"""Apple Trader on Fast&Furious: the momentum rules, driven by a scripted forecast.

These pin the RULES -- notebook 06's `trading._session`, long side -- without the
saved model or market data: when a forecast is a buy, how a fresh signal renews
the clock, the three exits, the re-arm after a stop -- plus how the strategy is
kept apart from everything written for the day-range one (the signature,
Orchestra, the tuning axes, a running agent's model switch).
`tests/test_fastfurious_model.py` pins the forecast itself.
"""

import math
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from agent_stonks import apple_models, clock, orchestra
from agent_stonks import apple_trader as at
from agent_stonks.apple_trader import AppleTraderConfig, config_signature
from agent_stonks.broker import Broker
from agent_stonks.decisions import DecisionTracker
from agent_stonks.state import AppState

TICKER = "AAPL"
# 10:30 ET on a Tuesday.
MIDSESSION = datetime(2026, 7, 21, 14, 30, tzinfo=timezone.utc)
THETA = 0.04
S120 = 0.005
DOWN = 0.4                      # the stop: 1.5 x 0.4 x 0.005 = 0.3% under the fill
BUNDLE = {"kind": "fastfurious", "path": "stub"}


class FakeBroker(Broker):
    def __init__(self, price: float = 100.0):
        self.price = price

    def get_current_price(self, symbol, key, secret, feed="iex") -> float:
        return self.price

    def submit_order(self, symbol, side, quantity, price) -> dict:
        return {"status": "filled", "filled_qty": quantity, "filled_price": price}


@pytest.fixture
def state() -> AppState:
    s = AppState()
    s.set_symbols([TICKER])
    s.api_key, s.api_secret, s.feed = "k", "s", "iex"
    return s


@pytest.fixture(autouse=True)
def market_open():
    clock.set_simulated(MIDSESSION)
    yield
    clock.clear()


class Tape:
    """Today's closed bars, and the forecast scripted for each of them:
    `append(close, y=..., low=..., down=...)`, minute 60 (10:30) onwards."""

    OPEN = pd.Timestamp("2026-07-21 09:30", tz="America/New_York")

    def __init__(self, monkeypatch, broker: FakeBroker):
        self.broker = broker
        self.rows: "list[dict]" = []
        self.index: "list[pd.Timestamp]" = []
        self.forecasts: "dict[int, dict | Exception]" = {}
        self.threshold = (THETA, 20)
        tape = self

        def forecast_at(bundle, ticker, day, minute, q, key=None, secret=None):
            found = tape.forecasts.get(minute)
            if isinstance(found, Exception):
                raise found
            if found is None:
                raise ValueError(f"no forecast at {minute}")
            theta = tape.threshold[0]
            y = found["y"]
            return {
                "minute": minute, "y_hat": y, "mom_bps": y * 0.003 / 30 * 1e4, "price": found["p"],
                "theta": theta, "theta_bps": theta * 0.003 / 30 * 1e4, "theta_sessions": 20,
                "signal": 1 if y >= theta else -1 if y <= -theta else 0,
                "up_hat": 0.5, "down_hat": found.get("down", DOWN), "s30": 0.003, "s120": S120,
                "pred_high": found["p"] * 1.002, "pred_low": found["p"] * 0.998,
                "range_problem": "",
            }

        def threshold(bundle, ticker, day, q, key=None, secret=None):
            if isinstance(tape.threshold, Exception):
                raise tape.threshold
            return tape.threshold

        fake = SimpleNamespace(
            forecast_at=forecast_at, threshold=threshold, LAST_ORIGIN=359,
            THRESHOLD_MIN_SESSIONS=5, is_half_day=lambda day: False,
            nyse_sessions=lambda days: list(days), warm=lambda *a, **k: None,
        )
        monkeypatch.setattr(at, "_fastfurious", lambda: fake)
        monkeypatch.setattr(at.momentum_regime, "minute_frame", lambda *a, **k: self.frame())

    def append(self, close: float, y: float = 0.0, low: "float | None" = None,
               down: "float | None" = None, error: "Exception | None" = None, minute=None):
        minute = 60 + len(self.rows) if minute is None else minute
        self.index.append(self.OPEN + pd.Timedelta(minutes=minute))
        self.rows.append({"open": close, "high": close, "low": close if low is None else low,
                          "close": close, "volume": 1e5})
        forecast = {"y": y, "p": close}
        if down is not None:
            forecast["down"] = down
        self.forecasts[minute] = error or forecast
        self.broker.price = close

    def frame(self) -> pd.DataFrame:
        return at.momentum_regime.add_session_columns(
            pd.DataFrame(self.rows, index=pd.DatetimeIndex(self.index))
        )


def ff_config(**kwargs) -> AppleTraderConfig:
    """Notebook 06's rules; no days off, so the calendar decides nothing here."""
    kwargs.setdefault("skip_events", ())
    return AppleTraderConfig(model_key=apple_models.FASTFURIOUS_KEY, ticker=TICKER, **kwargs)


def _setup(monkeypatch, state, **config):
    broker = FakeBroker(100.0)
    tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
    tape = Tape(monkeypatch, broker)
    trader = at.build_trader(ff_config(**config), BUNDLE)
    assert isinstance(trader, at.FastFuriousTrader)
    return trader, tracker, tape


def cycle(trader, state, tracker) -> str:
    return trader.run_cycle(BUNDLE, state, tracker)


def _exits(state) -> "list[str]":
    return [e.get("exit") for e in state.agent_log if e.get("type") == "decision" and e.get("action") == "sell"]


class TestEntry:
    def test_a_forecast_at_the_threshold_buys_with_a_stop_and_a_clock(self, monkeypatch, state):
        trader, tracker, tape = _setup(monkeypatch, state)
        tape.append(100.0, y=THETA)
        assert cycle(trader, state, tracker) == "bought"
        assert tracker.position_for(TICKER) > 0
        assert trader.entry["stop"] == pytest.approx(100.0 * math.exp(-1.5 * DOWN * S120))
        assert trader.entry["due"] == 60 + 30

    def test_a_forecast_under_the_threshold_buys_nothing(self, monkeypatch, state):
        trader, tracker, tape = _setup(monkeypatch, state)
        tape.append(100.0, y=THETA * 0.99)
        assert cycle(trader, state, tracker) == "hold"
        assert tracker.position_for(TICKER) == 0

    def test_a_bar_read_twice_is_acted_on_once(self, monkeypatch, state):
        trader, tracker, tape = _setup(monkeypatch, state)
        tape.append(100.0, y=THETA)
        cycle(trader, state, tracker)
        held = tracker.position_for(TICKER)
        assert cycle(trader, state, tracker) == "hold"
        assert tracker.position_for(TICKER) == held

    def test_no_range_forecast_no_entry_while_there_is_a_stop(self, monkeypatch, state):
        trader, tracker, tape = _setup(monkeypatch, state)
        tape.append(100.0, y=THETA, down=float("nan"))
        assert cycle(trader, state, tracker) == "hold"
        assert tracker.position_for(TICKER) == 0
        # With the stop off, the range is not needed.
        trader, tracker, tape = _setup(monkeypatch, state, ff_stop_k=0.0)
        tape.append(100.0, y=THETA, down=float("nan"))
        assert cycle(trader, state, tracker) == "bought"
        assert trader.entry["stop"] is None

    def test_a_forecast_that_fails_is_said_once_and_buys_nothing(self, monkeypatch, state):
        trader, tracker, tape = _setup(monkeypatch, state)
        for _ in range(3):
            tape.append(100.0, error=ValueError("IEX is down"))
            assert cycle(trader, state, tracker) == "hold"
        errors = [e for e in state.agent_log if e.get("type") == "error"]
        assert len(errors) == 1 and "IEX is down" in errors[0]["text"]

    def test_no_threshold_no_session(self, monkeypatch, state):
        trader, tracker, tape = _setup(monkeypatch, state)
        tape.threshold = (float("nan"), 3)
        tape.append(100.0, y=1.0)
        assert cycle(trader, state, tracker) == "no_data"
        assert trader.blocked is not None
        tape.append(100.0, y=1.0)
        assert cycle(trader, state, tracker) == "no_data"
        assert tracker.position_for(TICKER) == 0


class TestExits:
    def test_out_thirty_minutes_after_the_last_signal(self, monkeypatch, state):
        trader, tracker, tape = _setup(monkeypatch, state)
        tape.append(100.0, y=THETA)               # bar 60: bought, due 90
        cycle(trader, state, tracker)
        for _ in range(29):                       # bars 61..89
            tape.append(100.1, y=0.0)
            assert cycle(trader, state, tracker) == "hold"
        tape.append(100.1, y=0.0)                 # bar 90
        assert cycle(trader, state, tracker) == "sold"
        assert _exits(state) == [at.FastFuriousTrader.EXIT_TIME]

    def test_a_fresh_signal_restarts_the_clock(self, monkeypatch, state):
        trader, tracker, tape = _setup(monkeypatch, state)
        tape.append(100.0, y=THETA)               # bar 60
        cycle(trader, state, tracker)
        for _ in range(9):                        # bars 61..69
            tape.append(100.1, y=0.0)
            cycle(trader, state, tracker)
        tape.append(100.1, y=THETA * 2)           # bar 70: renewed, due 100
        assert cycle(trader, state, tracker) == "hold"
        assert trader.entry["due"] == 100
        for _ in range(29):                       # bars 71..99
            tape.append(100.1, y=0.0)
            assert cycle(trader, state, tracker) == "hold"
        tape.append(100.1, y=0.0)                 # bar 100
        assert cycle(trader, state, tracker) == "sold"

    def test_a_sell_signal_closes_the_long(self, monkeypatch, state):
        trader, tracker, tape = _setup(monkeypatch, state)
        tape.append(100.0, y=THETA)
        cycle(trader, state, tracker)
        tape.append(100.1, y=-THETA)
        assert cycle(trader, state, tracker) == "sold"
        assert _exits(state) == [at.FastFuriousTrader.EXIT_SELL_SIGNAL]
        # And it does not buy again on the same bar, or short.
        assert tracker.position_for(TICKER) == 0

    def test_without_the_sell_exit_a_sell_signal_is_nothing(self, monkeypatch, state):
        trader, tracker, tape = _setup(monkeypatch, state, ff_exit_on_sell_signal=False)
        tape.append(100.0, y=THETA)
        cycle(trader, state, tracker)
        tape.append(100.1, y=-THETA)
        assert cycle(trader, state, tracker) == "hold"
        assert tracker.position_for(TICKER) > 0

    def test_the_flatten_closes_and_buys_nothing(self, monkeypatch, state):
        trader, tracker, tape = _setup(monkeypatch, state)
        tape.append(100.0, y=THETA)
        cycle(trader, state, tracker)
        monkeypatch.setattr(at.FastFuriousTrader, "closing_soon", lambda self: True)
        tape.append(100.1, y=THETA)
        assert cycle(trader, state, tracker) == "sold"
        assert _exits(state) == [at.FastFuriousTrader.EXIT_FLATTEN]
        tape.append(100.1, y=THETA)
        assert cycle(trader, state, tracker) == "hold"
        assert tracker.position_for(TICKER) == 0


class TestStop:
    def _stopped(self, monkeypatch, state, y_on_stop_bar: float):
        trader, tracker, tape = _setup(monkeypatch, state)
        tape.append(100.0, y=THETA)                         # bought at 100, stop 99.70
        cycle(trader, state, tracker)
        tape.append(99.5, y=y_on_stop_bar, low=99.6)
        assert cycle(trader, state, tracker) == "sold"
        assert _exits(state) == [at.FastFuriousTrader.EXIT_STOP]
        return trader, tracker, tape

    def test_a_bar_through_the_stop_sells(self, monkeypatch, state):
        self._stopped(monkeypatch, state, 0.0)

    def test_a_bar_above_the_stop_holds(self, monkeypatch, state):
        trader, tracker, tape = _setup(monkeypatch, state)
        tape.append(100.0, y=THETA)
        cycle(trader, state, tracker)
        tape.append(99.8, y=0.0, low=99.75)
        assert cycle(trader, state, tracker) == "hold"

    def test_after_a_stop_the_buy_waits_for_the_forecast_to_leave_the_zone(self, monkeypatch, state):
        trader, tracker, tape = self._stopped(monkeypatch, state, THETA)   # still in the zone
        tape.append(99.5, y=THETA)
        assert cycle(trader, state, tracker) == "hold"                     # blocked
        tape.append(99.5, y=0.0)                                           # leaves the zone
        assert cycle(trader, state, tracker) == "hold"
        tape.append(99.5, y=THETA)
        assert cycle(trader, state, tracker) == "bought"

    def test_a_stop_bar_whose_forecast_is_out_of_the_zone_rearms_at_once(self, monkeypatch, state):
        trader, tracker, tape = self._stopped(monkeypatch, state, 0.0)
        tape.append(99.5, y=THETA)
        assert cycle(trader, state, tracker) == "bought"

    def test_a_stop_does_not_stop_the_run(self, monkeypatch, state):
        trader, _, _ = self._stopped(monkeypatch, state, 0.0)
        assert trader.halt is None


class TestRestart:
    def test_an_open_position_keeps_its_stop_and_clock(self, monkeypatch, state):
        trader, tracker, tape = _setup(monkeypatch, state)
        tape.append(100.0, y=THETA)
        cycle(trader, state, tracker)
        entry = dict(trader.entry)
        # A new run on the same ledger and the same day's record.
        again = at.build_trader(ff_config(), BUNDLE)
        tape.append(100.1, y=0.0)
        assert cycle(again, state, tracker) == "hold"
        assert again.entry["stop"] == pytest.approx(entry["stop"])
        assert again.entry["due"] == entry["due"]

    def test_a_day_range_runs_record_is_not_taken_up(self, monkeypatch, state):
        """A position a day-range run left open today has no momentum clock:
        it is adopted at the bar with a fresh one, not resumed and sold at once."""
        trader, tracker, tape = _setup(monkeypatch, state)
        tape.append(100.0, y=THETA)
        cycle(trader, state, tracker)                      # a long is held
        state.apple_trader_levels = {
            "ticker": TICKER, "date": pd.Timestamp("2026-07-21"),
            "config": AppleTraderConfig(model_key="dayrange"),
            "rows": [{"t": tape.index[0], "buy": 99.0, "sell": 101.0, "stop": 98.0}],
            "memory": {"entry": {"price": 100.0, "bars": 3, "risk": 2.0}, "stand_down": None},
        }
        again = at.build_trader(ff_config(), BUNDLE)
        tape.append(100.1, y=0.0)                          # bar 61
        assert cycle(again, state, tracker) == "hold"
        assert again.entry["due"] == 61 + 30 and again.entry["stop"] is None
        assert state.apple_trader_levels["config"].model_key == "fastfurious"

    def test_the_record_carries_the_forecast_and_the_stop(self, monkeypatch, state):
        trader, tracker, tape = _setup(monkeypatch, state)
        tape.append(100.0, y=THETA)
        cycle(trader, state, tracker)
        record = state.apple_trader_levels
        assert record["ticker"] == TICKER and record["config"].model_key == "fastfurious"
        row = record["rows"][-1]
        assert row["stop"] == pytest.approx(trader.entry["stop"])
        assert row["signal"] == 1 and row["mom_bps"] is not None
        assert record["memory"]["entry"]["due"] == 90


class TestKeptApart:
    """What is written for the day-range rules does not take Fast&Furious in."""

    def test_its_signature_reads_only_its_own_rules(self):
        base = ff_config()
        assert config_signature(base) == (
            "fastfurious_AAPL(q=0.95,stop=1.5dn,hold=30m,sellexit,size=95%)"
        )
        assert config_signature(replace(base, buy_k=0.9, sell_k=0.2)) == config_signature(base)
        assert config_signature(replace(base, ff_stop_k=0.0, ff_exit_on_sell_signal=False)) == (
            "fastfurious_AAPL(q=0.95,stop=off,hold=30m,size=95%)"
        )
        dayrange = AppleTraderConfig(model_key="dayrange")
        assert config_signature(replace(dayrange, ff_entry_quantile=0.9)) == config_signature(dayrange)

    def test_its_rules_are_checked(self):
        for bad in (dict(ff_entry_quantile=1.0), dict(ff_stop_k=-0.5), dict(ff_hold_minutes=0)):
            with pytest.raises(ValueError):
                ff_config(**bad)

    def test_orchestra_neither_offers_nor_takes_it(self):
        assert not [p for p in orchestra.all_pairs() if p.endswith(":fastfurious")]
        with pytest.raises(ValueError, match="day-range"):
            orchestra.OrchestraConfig([ff_config()])

    def test_tuning_offers_its_axes_only_on_it(self):
        from simlab import tuning

        assert set(tuning.axes_for("fastfurious", tuning.SWEEPABLE)) == set(tuning.MOMENTUM_AXES)
        assert not [a for a in tuning.axes_for("dayrange", tuning.SWEEPABLE) if a.startswith("ff_")]

    def test_a_running_day_range_agent_refuses_a_switch_to_it(self, monkeypatch, state):
        trader = at.DayRangeTrader(AppleTraderConfig(model_key="dayrange", skip_events=()))
        state.apple_trader_config = ff_config()
        frame = pd.DataFrame({"close": [100.0]}, index=pd.DatetimeIndex(
            [pd.Timestamp("2026-07-21 10:30", tz="America/New_York")]))
        assert trader._adopt_form_model(state, frame, pd.Timestamp("2026-07-21")) is False
        assert trader.config.model_key == "dayrange"
        assert any("own rules" in (e.get("text") or "") for e in state.agent_log)
