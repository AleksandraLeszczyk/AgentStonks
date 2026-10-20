"""Apple Trader on GoodMoments: the probability rules, driven by a scripted forecast.

These pin the RULES -- notebook 4's design 2, `trading._session` -- without the
saved model or market data: when an edge is a buy, the take and the stop in
ADRs from the fill, the 15-bar pause after a stop, the re-entry after a take,
the optional sale on a fading edge -- plus how the strategy is kept apart from
everything written for the day-range one (the signature, Orchestra, the tuning
axes, a running agent's model switch). `tests/test_goodmoments_model.py` pins
the forecast itself.
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
ADR = 0.02                       # take +0.4%, stop -1.5% (log terms) at the defaults
ENTER = 0.02
BUNDLE = {"kind": "goodmoments", "path": "stub"}


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
    """Today's closed bars and the edge scripted for each of them:
    `append(close, edge=..., low=..., high=...)`, minute 60 (10:30) onwards."""

    OPEN = pd.Timestamp("2026-07-21 09:30", tz="America/New_York")

    def __init__(self, monkeypatch, broker: FakeBroker):
        self.broker = broker
        self.rows: "list[dict]" = []
        self.index: "list[pd.Timestamp]" = []
        self.forecasts: "dict[int, dict | Exception]" = {}
        self.adr: "float | Exception" = ADR
        tape = self

        def forecast_at(bundle, ticker, day, minute, key=None, secret=None):
            found = tape.forecasts.get(minute)
            if isinstance(found, Exception):
                raise found
            if found is None:
                raise ValueError(f"no forecast at {minute}")
            p_tv = 0.5
            return {"minute": minute, "p": p_tv + found["edge"], "p_tv": p_tv, "edge": found["edge"],
                    "price": found["price"], "adr": ADR, "p_reach": 0.5}

        def session_adr(bundle, ticker, day, key=None, secret=None):
            if isinstance(tape.adr, Exception):
                raise tape.adr
            return tape.adr

        fake = SimpleNamespace(
            forecast_at=forecast_at, session_adr=session_adr, LAST_FORECAST=388, TARGET_ADR=0.2,
            is_half_day=lambda day: False, is_session=lambda day: True, warm=lambda *a, **k: None,
        )
        monkeypatch.setattr(at, "_goodmoments", lambda: fake)
        monkeypatch.setattr(at.momentum_regime, "minute_frame", lambda *a, **k: self.frame())

    def append(self, close: float, edge: float = 0.0, low: "float | None" = None,
               high: "float | None" = None, error: "Exception | None" = None, minute=None):
        minute = 60 + len(self.rows) if minute is None else minute
        self.index.append(self.OPEN + pd.Timedelta(minutes=minute))
        self.rows.append({"open": close, "high": close if high is None else high,
                          "low": close if low is None else low, "close": close, "volume": 1e5})
        self.forecasts[minute] = error or {"edge": edge, "price": close}
        self.broker.price = close

    def frame(self) -> pd.DataFrame:
        return at.momentum_regime.add_session_columns(
            pd.DataFrame(self.rows, index=pd.DatetimeIndex(self.index))
        )


def gm_config(**kwargs) -> AppleTraderConfig:
    """Notebook 4's design 2; no days off, so the calendar decides nothing here."""
    kwargs.setdefault("skip_events", ())
    return AppleTraderConfig(model_key=apple_models.GOODMOMENTS_KEY, ticker=TICKER, **kwargs)


def _setup(monkeypatch, state, **config):
    broker = FakeBroker(100.0)
    tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
    tape = Tape(monkeypatch, broker)
    trader = at.build_trader(gm_config(**config), BUNDLE)
    assert isinstance(trader, at.GoodMomentsTrader)
    return trader, tracker, tape


def cycle(trader, state, tracker) -> str:
    """One cycle as the live loop runs it: the rules, then the memory published."""
    outcome = trader.run_cycle(BUNDLE, state, tracker)
    trader.publish_memory(state)
    return outcome


def _exits(state) -> "list[str]":
    return [e.get("exit") for e in state.agent_log if e.get("type") == "decision" and e.get("action") == "sell"]


TAKE = 100.0 * math.exp(0.2 * ADR)
STOP = 100.0 * math.exp(-0.75 * ADR)


class TestEntry:
    def test_an_edge_at_the_entry_buys_with_a_take_and_a_stop_in_adrs(self, monkeypatch, state):
        trader, tracker, tape = _setup(monkeypatch, state)
        tape.append(100.0, edge=ENTER)
        assert cycle(trader, state, tracker) == "bought"
        assert tracker.position_for(TICKER) > 0
        assert trader.entry["take"] == pytest.approx(TAKE)
        assert trader.entry["stop"] == pytest.approx(STOP)

    def test_an_edge_under_the_entry_buys_nothing(self, monkeypatch, state):
        trader, tracker, tape = _setup(monkeypatch, state)
        tape.append(100.0, edge=ENTER - 0.001)
        assert cycle(trader, state, tracker) == "hold"
        assert tracker.position_for(TICKER) == 0

    def test_it_is_the_edge_not_the_probability(self, monkeypatch, state):
        """A probability of 0.9 that the clock and volatility alone also give
        is no reason to buy -- design 1's mistake."""
        trader, tracker, tape = _setup(monkeypatch, state)
        tape.append(100.0, edge=0.0)
        assert cycle(trader, state, tracker) == "hold"

    def test_a_bar_read_twice_is_acted_on_once(self, monkeypatch, state):
        trader, tracker, tape = _setup(monkeypatch, state)
        tape.append(100.0, edge=ENTER)
        cycle(trader, state, tracker)
        held = tracker.position_for(TICKER)
        assert cycle(trader, state, tracker) == "hold"
        assert tracker.position_for(TICKER) == held

    def test_no_entry_after_15_30(self, monkeypatch, state):
        trader, tracker, tape = _setup(monkeypatch, state)
        tape.append(100.0, edge=0.2, minute=361)
        assert cycle(trader, state, tracker) == "hold"
        assert tracker.position_for(TICKER) == 0
        trader, tracker, tape = _setup(monkeypatch, state)
        tape.append(100.0, edge=0.2, minute=360)
        assert cycle(trader, state, tracker) == "bought"

    def test_a_forecast_that_fails_is_said_once_and_buys_nothing(self, monkeypatch, state):
        trader, tracker, tape = _setup(monkeypatch, state)
        for _ in range(3):
            tape.append(100.0, error=ValueError("no option positioning"))
            assert cycle(trader, state, tracker) == "hold"
        errors = [e for e in state.agent_log if e.get("type") == "error"]
        assert len(errors) == 1 and "no option positioning" in errors[0]["text"]

    def test_no_adr_no_session(self, monkeypatch, state):
        trader, tracker, tape = _setup(monkeypatch, state)
        tape.adr = ValueError("only 40 SIP sessions of history")
        tape.append(100.0, edge=0.2)
        assert cycle(trader, state, tracker) == "no_data"
        assert trader.blocked is not None
        tape.append(100.0, edge=0.2)
        assert cycle(trader, state, tracker) == "no_data"
        assert tracker.position_for(TICKER) == 0


class TestExits:
    def _bought(self, monkeypatch, state, **config):
        trader, tracker, tape = _setup(monkeypatch, state, **config)
        tape.append(100.0, edge=ENTER)
        assert cycle(trader, state, tracker) == "bought"
        return trader, tracker, tape

    def test_a_bar_reaching_the_take_sells(self, monkeypatch, state):
        trader, tracker, tape = self._bought(monkeypatch, state)
        tape.append(100.2, high=TAKE + 0.01)
        assert cycle(trader, state, tracker) == "sold"
        assert _exits(state) == [at.GoodMomentsTrader.EXIT_TAKE]

    def test_after_a_take_the_same_bar_can_buy_again(self, monkeypatch, state):
        """The notebook frees the position on the take and reads the same
        bar's forecast for an entry."""
        trader, tracker, tape = self._bought(monkeypatch, state)
        tape.append(100.4, high=TAKE + 0.01, edge=ENTER)
        assert cycle(trader, state, tracker) == "bought"
        assert _exits(state) == [at.GoodMomentsTrader.EXIT_TAKE]
        assert trader.entry["take"] == pytest.approx(100.4 * math.exp(0.2 * ADR))

    def test_a_bar_through_the_stop_sells_and_pauses_entries(self, monkeypatch, state):
        trader, tracker, tape = self._bought(monkeypatch, state)        # bar 60
        tape.append(98.6, low=STOP - 0.01, edge=0.2)                    # bar 61: stopped
        assert cycle(trader, state, tracker) == "sold"
        assert _exits(state) == [at.GoodMomentsTrader.EXIT_STOP]
        assert trader.halt is None
        for _ in range(14):                                             # bars 62..75
            tape.append(98.6, edge=0.2)
            assert cycle(trader, state, tracker) == "hold"
        tape.append(98.6, edge=0.2)                                     # bar 76 = 61 + 15
        assert cycle(trader, state, tracker) == "bought"

    def test_a_bar_touching_both_is_a_stop(self, monkeypatch, state):
        trader, tracker, tape = self._bought(monkeypatch, state)
        tape.append(100.0, low=STOP - 0.01, high=TAKE + 0.01)
        assert cycle(trader, state, tracker) == "sold"
        assert _exits(state) == [at.GoodMomentsTrader.EXIT_STOP]

    def test_otherwise_it_holds_whatever_the_edge(self, monkeypatch, state):
        trader, tracker, tape = self._bought(monkeypatch, state)
        tape.append(100.1, edge=-0.3)
        assert cycle(trader, state, tracker) == "hold"
        assert tracker.position_for(TICKER) > 0

    def test_with_an_exit_a_fading_edge_sells(self, monkeypatch, state):
        trader, tracker, tape = self._bought(monkeypatch, state, gm_exit=0.0)
        tape.append(100.1, edge=0.001)
        assert cycle(trader, state, tracker) == "hold"
        tape.append(100.1, edge=-0.001)
        assert cycle(trader, state, tracker) == "sold"
        assert _exits(state) == [at.GoodMomentsTrader.EXIT_SIGNAL]
        assert tracker.position_for(TICKER) == 0

    def test_off_switches(self, monkeypatch, state):
        trader, tracker, tape = self._bought(monkeypatch, state, gm_take_k=0.0, gm_stop_k=0.0)
        assert trader.entry["take"] is None and trader.entry["stop"] is None
        tape.append(90.0, low=80.0, high=120.0)
        assert cycle(trader, state, tracker) == "hold"

    def test_the_flatten_closes_and_buys_nothing(self, monkeypatch, state):
        trader, tracker, tape = self._bought(monkeypatch, state)
        monkeypatch.setattr(at.GoodMomentsTrader, "closing_soon", lambda self: True)
        tape.append(100.1, edge=0.2)
        assert cycle(trader, state, tracker) == "sold"
        assert _exits(state) == [at.GoodMomentsTrader.EXIT_FLATTEN]
        tape.append(100.1, edge=0.2)
        assert cycle(trader, state, tracker) == "hold"
        assert tracker.position_for(TICKER) == 0


class TestRestart:
    def test_an_open_position_keeps_its_take_and_stop(self, monkeypatch, state):
        trader, tracker, tape = _setup(monkeypatch, state)
        tape.append(100.0, edge=ENTER)
        cycle(trader, state, tracker)
        entry = dict(trader.entry)
        again = at.build_trader(gm_config(), BUNDLE)
        tape.append(100.1)
        assert cycle(again, state, tracker) == "hold"
        assert again.entry["take"] == pytest.approx(entry["take"])
        assert again.entry["stop"] == pytest.approx(entry["stop"])

    def test_the_pause_after_a_stop_survives_a_restart(self, monkeypatch, state):
        trader, tracker, tape = _setup(monkeypatch, state)
        tape.append(100.0, edge=ENTER)
        cycle(trader, state, tracker)
        tape.append(98.0, low=STOP - 0.01)                 # bar 61: stopped, pause to 76
        cycle(trader, state, tracker)
        again = at.build_trader(gm_config(), BUNDLE)
        tape.append(98.0, edge=0.2)                        # bar 62
        assert cycle(again, state, tracker) == "hold"
        assert again.session["cool_until"] == 76

    def test_a_day_range_runs_record_is_not_taken_up(self, monkeypatch, state):
        """A position a day-range run left open today is adopted at the bar
        with a take and a stop of its own, not resumed on the other rules."""
        trader, tracker, tape = _setup(monkeypatch, state)
        tape.append(100.0, edge=ENTER)
        cycle(trader, state, tracker)
        state.apple_trader_levels = {
            "ticker": TICKER, "date": pd.Timestamp("2026-07-21"),
            "config": AppleTraderConfig(model_key="dayrange"),
            "rows": [{"t": tape.index[0], "buy": 99.0, "sell": 101.0, "stop": 98.0}],
            "memory": {"entry": {"price": 100.0, "bars": 3, "risk": 2.0}, "stand_down": None},
        }
        again = at.build_trader(gm_config(), BUNDLE)
        tape.append(101.0)                                 # bar 61
        assert cycle(again, state, tracker) == "hold"
        assert again.entry["take"] == pytest.approx(101.0 * math.exp(0.2 * ADR))
        assert state.apple_trader_levels["config"].model_key == "goodmoments"

    def test_the_record_carries_the_forecast_and_the_levels(self, monkeypatch, state):
        trader, tracker, tape = _setup(monkeypatch, state)
        tape.append(100.0, edge=ENTER)
        cycle(trader, state, tracker)
        record = state.apple_trader_levels
        assert record["ticker"] == TICKER and record["config"].model_key == "goodmoments"
        row = record["rows"][-1]
        assert row["edge"] == pytest.approx(ENTER) and row["p_tv"] == pytest.approx(0.5)
        assert row["take"] == pytest.approx(TAKE) and row["stop"] == pytest.approx(STOP)


class TestKeptApart:
    """What is written for the day-range rules does not take GoodMoments in."""

    def test_its_signature_reads_only_its_own_rules(self):
        base = gm_config()
        assert config_signature(base) == (
            "goodmoments_AAPL(enter=+0.02,exit=off,take=0.2A,stop=0.75A,size=95%)"
        )
        assert config_signature(replace(base, buy_k=0.9, sell_k=0.2, ff_stop_k=0.0)) == config_signature(base)
        assert config_signature(replace(base, gm_exit=-0.02, gm_take_k=0.0)) == (
            "goodmoments_AAPL(enter=+0.02,exit=-0.02,take=off,stop=0.75A,size=95%)"
        )
        dayrange = AppleTraderConfig(model_key="dayrange")
        assert config_signature(replace(dayrange, gm_enter=0.05)) == config_signature(dayrange)

    def test_its_rules_are_checked(self):
        for bad in (dict(gm_enter=1.5), dict(gm_exit=0.02), dict(gm_take_k=-0.1), dict(gm_stop_k=-1)):
            with pytest.raises(ValueError):
                gm_config(**bad)

    def test_orchestra_neither_offers_nor_takes_it(self):
        assert not [p for p in orchestra.all_pairs() if p.endswith(":goodmoments")]
        with pytest.raises(ValueError, match="day-range"):
            orchestra.OrchestraConfig([gm_config()])

    def test_tuning_offers_its_axes_only_on_it(self):
        from simlab import tuning

        assert set(tuning.axes_for("goodmoments", tuning.SWEEPABLE)) == set(tuning.PROBABILITY_AXES)
        assert not [a for a in tuning.axes_for("dayrange", tuning.SWEEPABLE) if a.startswith("gm_")]
        assert not [a for a in tuning.axes_for("fastfurious", tuning.SWEEPABLE) if a.startswith("gm_")]
        assert tuning.DEFAULT_AXES[gm_config().strategy] == ("gm_enter", "gm_stop_k")

    def test_a_running_day_range_agent_refuses_a_switch_to_it(self, monkeypatch, state):
        trader = at.DayRangeTrader(AppleTraderConfig(model_key="dayrange", skip_events=()))
        state.apple_trader_config = gm_config()
        frame = pd.DataFrame({"close": [100.0]}, index=pd.DatetimeIndex(
            [pd.Timestamp("2026-07-21 10:30", tz="America/New_York")]))
        assert trader._adopt_form_model(state, frame, pd.Timestamp("2026-07-21")) is False
        assert trader.config.model_key == "dayrange"
