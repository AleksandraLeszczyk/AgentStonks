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
            "goodmoments_AAPL(enter=+0.02,exit=off,take=0.2A,stop=0.75A,raise=0.05/stop,size=95%)"
        )
        assert config_signature(replace(base, buy_k=0.9, sell_k=0.2, ff_stop_k=0.0)) == config_signature(base)
        assert config_signature(replace(base, gm_exit=-0.02, gm_take_k=0.0, gm_stop_raise=0.0)) == (
            "goodmoments_AAPL(enter=+0.02,exit=-0.02,take=off,stop=0.75A,size=95%)"
        )
        dayrange = AppleTraderConfig(model_key="dayrange")
        assert config_signature(replace(dayrange, gm_enter=0.05)) == config_signature(dayrange)

    def test_its_rules_are_checked(self):
        for bad in (dict(gm_enter=1.5), dict(gm_exit=0.02), dict(gm_take_k=-0.1), dict(gm_stop_k=-1)):
            with pytest.raises(ValueError):
                gm_config(**bad)

    def test_each_ticker_starts_from_its_notebooks_stop(self, monkeypatch):
        """AMD's notebook 4 picked a 0.5 ADR stop, AAPL's 0.75; a stored or
        typed stop is kept whatever the ticker."""
        monkeypatch.setattr(at.base_configs, "stored", lambda *a, **k: None)
        amd = at.default_config("AMD", apple_models.GOODMOMENTS_KEY)
        assert amd.gm_stop_k == 0.5 and amd.gm_enter == 0.02 and amd.gm_take_k == 0.2
        assert at.default_config("AAPL", apple_models.GOODMOMENTS_KEY).gm_stop_k == 0.75
        assert AppleTraderConfig(model_key=apple_models.GOODMOMENTS_KEY, ticker="AMD", gm_stop_k=0.75).gm_stop_k == 0.75
        assert "stop=0.5A" in config_signature(amd)

    def test_amd_is_offered(self):
        assert apple_models.GOODMOMENTS_KEY in apple_models.keys_for("AMD")
        assert apple_models.HIGHLOW2_GOODMOMENTS_KEY not in apple_models.keys_for("AMD")

    def test_orchestra_offers_and_takes_it_but_not_fast_and_furious(self):
        # Since 2026-10-25 Orchestra races GoodMoments pairs too
        # (tests/test_orchestra.py); Fast&Furious's rules it is not written for.
        assert {"AAPL:goodmoments", "AMD:goodmoments"} <= set(orchestra.all_pairs())
        assert orchestra.OrchestraConfig([gm_config()]).keys == ["AAPL:goodmoments"]
        assert not [p for p in orchestra.all_pairs() if p.endswith(":fastfurious")]
        with pytest.raises(ValueError, match="runs its own"):
            orchestra.OrchestraConfig([AppleTraderConfig(model_key=apple_models.FASTFURIOUS_KEY)])

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


# A plausible time-of-day curve (09:30 ~3x midday, a short close ramp), so the
# adaptive take is pinned without the exported file.
CURVE = {
    "shape": {
        "params": {"a": 0.6, "b": 3.0, "alpha": 0.45, "c": 0.5, "kappa": 12.0},
        "t_domain": [2.5, 387.5],
    },
}
TAKE_MAX, TAKE_MIN = 0.4, 0.1


def adaptive_take(fill: float, minute: int) -> float:
    scale = at.intraday_vol_model.volatility_scale(CURVE)[minute]
    return fill * math.exp((TAKE_MIN + (TAKE_MAX - TAKE_MIN) * scale) * ADR)


class TestAdaptiveTake:
    """The take that follows the intraday volatility curve: `gm_take_k` at the
    session's most volatile minute, `gm_take_min_k` at its quietest, moved to
    each bar's minute and measured from the fill."""

    @pytest.fixture(autouse=True)
    def curve(self, monkeypatch):
        monkeypatch.setattr(at.intraday_vol_model, "load", lambda ticker: CURVE)

    def _bought(self, monkeypatch, state, minute=None, **config):
        config = {"gm_take_k": TAKE_MAX, "gm_take_min_k": TAKE_MIN, **config}
        trader, tracker, tape = _setup(monkeypatch, state, **config)
        tape.append(100.0, edge=ENTER, minute=minute)
        assert cycle(trader, state, tracker) == "bought"
        return trader, tracker, tape

    def test_the_take_runs_from_the_peak_take_to_the_trough_take(self):
        config = gm_config(gm_take_k=TAKE_MAX, gm_take_min_k=TAKE_MIN)
        assert at.gm_take_k_at(config, 1.0) == pytest.approx(TAKE_MAX)
        assert at.gm_take_k_at(config, 0.0) == pytest.approx(TAKE_MIN)
        assert at.gm_take_k_at(config, 0.5) == pytest.approx(0.25)
        assert at.gm_take_k_at(gm_config(), 0.5) == 0.2       # flat: the scale is not read

    def test_a_buy_at_the_open_rests_the_peak_take(self, monkeypatch, state):
        trader, tracker, tape = self._bought(monkeypatch, state, minute=0)
        assert trader.entry["take"] == pytest.approx(100.0 * math.exp(TAKE_MAX * ADR))
        assert trader.entry["stop"] == pytest.approx(STOP)

    def test_a_buy_at_the_quietest_minute_rests_the_trough_take(self, monkeypatch, state):
        trough = int(at.intraday_vol_model.volatility_scale(CURVE).argmin())
        trader, tracker, tape = self._bought(monkeypatch, state, minute=trough)
        assert trader.entry["take"] == pytest.approx(100.0 * math.exp(TAKE_MIN * ADR))

    def test_the_take_moves_every_bar_from_the_fill_and_the_stop_does_not(self, monkeypatch, state):
        trader, tracker, tape = self._bought(monkeypatch, state)            # bar 60
        assert trader.entry["take"] == pytest.approx(adaptive_take(100.0, 60))
        takes = []
        for minute in (61, 62, 63):
            tape.append(100.05)
            assert cycle(trader, state, tracker) == "hold"
            assert trader.entry["take"] == pytest.approx(adaptive_take(100.0, minute))
            assert trader.entry["stop"] == pytest.approx(STOP)
            takes.append(trader.entry["take"])
        assert takes == sorted(takes, reverse=True) and takes[0] > takes[-1]

    def test_a_bar_reaching_the_shrunken_take_sells(self, monkeypatch, state):
        """Bought at the open with 0.4 ADR asked; by 11:30 a smaller rise is
        enough -- one the flat 0.2 ADR take would not have sold on."""
        trader, tracker, tape = self._bought(monkeypatch, state, minute=0)
        later, flat = adaptive_take(100.0, 120), 100.0 * math.exp(0.2 * ADR)
        assert later < flat
        tape.append(100.0, high=(later + flat) / 2, minute=120)
        assert cycle(trader, state, tracker) == "sold"
        assert _exits(state) == [at.GoodMomentsTrader.EXIT_TAKE]
        sold = [e for e in state.agent_log if e.get("type") == "decision" and e.get("action") == "sell"]
        assert "at 11:30's volatility" in sold[-1]["reasoning"]

    def test_the_same_bar_early_does_not(self, monkeypatch, state):
        trader, tracker, tape = self._bought(monkeypatch, state, minute=0)
        tape.append(100.0, high=(adaptive_take(100.0, 120) + 100.0 * math.exp(0.2 * ADR)) / 2, minute=1)
        assert cycle(trader, state, tracker) == "hold"

    def test_the_record_carries_the_moving_take(self, monkeypatch, state):
        trader, tracker, tape = self._bought(monkeypatch, state)
        tape.append(100.05)
        cycle(trader, state, tracker)
        rows = state.apple_trader_levels["rows"]
        assert [r["take"] for r in rows] == pytest.approx([adaptive_take(100.0, 60), adaptive_take(100.0, 61)])

    def test_the_chart_draws_the_take_as_it_moved(self, monkeypatch, state):
        from agent_stonks import model_overlays

        trader, tracker, tape = self._bought(monkeypatch, state)
        tape.append(100.05)
        cycle(trader, state, tracker)
        record = state.apple_trader_levels
        items, problem = model_overlays._signal_trader_items(
            TICKER, (record["rows"], record["config"]), apple_models.GOODMOMENTS_KEY,
        )
        take = next(i for i in items if i["label"] == "Take-profit")
        assert not problem and take["step"]
        assert take["values"] == pytest.approx([adaptive_take(100.0, 60), adaptive_take(100.0, 61)])
        assert "0.4 ADR" in take["note"] and "0.1 at its quietest" in take["note"]

    def test_a_restart_moves_the_resumed_take_to_the_bar(self, monkeypatch, state):
        trader, tracker, tape = self._bought(monkeypatch, state, minute=0)
        again = at.build_trader(gm_config(gm_take_k=TAKE_MAX, gm_take_min_k=TAKE_MIN), BUNDLE)
        tape.append(100.05, minute=90)
        assert cycle(again, state, tracker) == "hold"
        assert again.entry["take"] == pytest.approx(adaptive_take(100.0, 90))

    def test_without_a_curve_the_session_is_not_traded(self, monkeypatch, state):
        monkeypatch.setattr(at.intraday_vol_model, "load", lambda ticker: None)
        trader, tracker, tape = _setup(monkeypatch, state, gm_take_k=TAKE_MAX, gm_take_min_k=TAKE_MIN)
        tape.append(100.0, edge=0.2)
        assert cycle(trader, state, tracker) == "no_data"
        assert trader.blocked is not None and tracker.position_for(TICKER) == 0
        errors = [e["text"] for e in state.agent_log if e.get("type") == "error"]
        assert errors and "intraday volatility curve" in errors[0]

    def test_with_the_take_off_no_curve_is_needed(self, monkeypatch, state):
        monkeypatch.setattr(at.intraday_vol_model, "load", lambda ticker: None)
        trader, tracker, tape = self._bought(monkeypatch, state, gm_take_k=0.0, gm_take_min_k=0.0)
        assert trader.entry["take"] is None

    def test_its_signature_and_its_checks(self):
        assert gm_config().gm_take_min_k is None
        assert config_signature(gm_config(gm_take_k=TAKE_MAX, gm_take_min_k=TAKE_MIN)) == (
            "goodmoments_AAPL(enter=+0.02,exit=off,take=0.4..0.1A@vol,stop=0.75A,raise=0.05/stop,size=95%)"
        )
        assert config_signature(gm_config(gm_take_k=0.0, gm_take_min_k=0.0)) == (
            "goodmoments_AAPL(enter=+0.02,exit=off,take=off,stop=0.75A,raise=0.05/stop,size=95%)"
        )
        for bad in (dict(gm_take_min_k=0.3), dict(gm_take_min_k=-0.05)):    # gm_take_k is 0.2
            with pytest.raises(ValueError, match="gm_take_min_k"):
                gm_config(**bad)

    def test_a_grid_cell_above_the_take_is_a_hole(self):
        from dataclasses import asdict

        from simlab import tuning

        assert "gm_take_min_k" in tuning.axes_for("goodmoments", tuning.SWEEPABLE)
        configs, refused = tuning.expand(
            gm_config(), [{"name": "gm_take_min_k", "values": [0.1, 0.3]}],
        )
        assert [c.gm_take_min_k for c in configs] == [0.1]
        assert [o for o, _ in refused] == [{"gm_take_min_k": 0.3}]
        old = {k: v for k, v in asdict(gm_config()).items() if k != "gm_take_min_k"}
        assert tuning.make_config(old, {}).gm_take_min_k is None


class TestForm:
    """The sidebar's knob for the adaptive take, in the form both apps render."""

    @staticmethod
    def _form() -> None:
        import streamlit as st

        from agent_stonks import apple_trader_ui

        copy = apple_trader_ui.FormCopy(
            prefix="t", unavailable_suffix="", instrument_help="", model_help=""
        )
        st.session_state["out"] = apple_trader_ui.params(["AAPL"], copy)

    @pytest.fixture
    def app(self, monkeypatch):
        from streamlit.testing.v1 import AppTest

        monkeypatch.setattr(apple_models, "load", lambda *a, **k: {"model": "stub"})
        app = AppTest.from_function(self._form, default_timeout=30)
        app.run()
        app.selectbox(key="t_model_AAPL").select(apple_models.GOODMOMENTS_KEY).run()
        assert not app.exception, app.exception
        return app

    def test_off_by_default_and_the_take_is_flat(self, app):
        assert app.checkbox(key="t_gm_take_adaptive_goodmoments_AAPL").value is False
        assert app.number_input(key="t_gm_take_min_k_goodmoments_AAPL").disabled
        assert app.session_state["out"].gm_take_min_k is None

    def test_ticked_the_take_follows_volatility(self, app):
        app.checkbox(key="t_gm_take_adaptive_goodmoments_AAPL").check().run()
        app.number_input(key="t_gm_take_k_goodmoments_AAPL").set_value(0.4).run()
        app.number_input(key="t_gm_take_min_k_goodmoments_AAPL").set_value(0.15).run()
        out = app.session_state["out"]
        assert (out.gm_take_k, out.gm_take_min_k) == pytest.approx((0.4, 0.15))
        assert not app.warning

    def test_the_momentum_confirmation_knob(self, app):
        knob = app.number_input(key="t_gm_confirm_bars_goodmoments_AAPL")
        assert knob.value == 0 and app.session_state["out"].gm_confirm_bars == 0
        knob.set_value(3).run()
        assert app.session_state["out"].gm_confirm_bars == 3

    def test_the_circuit_breaker_is_on_and_its_box_switches_it_off(self, app):
        assert app.checkbox(key="t_gm_stop_raise_on_goodmoments_AAPL").value is True
        assert app.session_state["out"].gm_stop_raise == pytest.approx(0.05)
        app.number_input(key="t_gm_stop_raise_goodmoments_AAPL").set_value(0.03).run()
        assert app.session_state["out"].gm_stop_raise == pytest.approx(0.03)
        app.checkbox(key="t_gm_stop_raise_on_goodmoments_AAPL").uncheck().run()
        assert app.session_state["out"].gm_stop_raise == 0.0
        assert app.number_input(key="t_gm_stop_raise_goodmoments_AAPL").disabled

    def test_a_trough_take_above_the_peak_take_is_refused(self, app):
        app.checkbox(key="t_gm_take_adaptive_goodmoments_AAPL").check().run()
        app.number_input(key="t_gm_take_min_k_goodmoments_AAPL").set_value(0.5).run()
        assert app.session_state["out"].gm_take_min_k is None
        assert "at most the take-profit" in app.warning[0].value


class TestCircuitBreaker:
    """Each stop raises the edge a buy needs for the rest of the session
    (`gm_stop_raise`, 2026-10-25), on top of the 15-bar pause."""

    RAISE = 0.05

    def _stopped(self, monkeypatch, state, **config):
        """Bought at bar 60, stopped at bar 61, paused through bar 75."""
        trader, tracker, tape = _setup(monkeypatch, state, **config)
        tape.append(100.0, edge=ENTER)
        assert cycle(trader, state, tracker) == "bought"
        tape.append(98.6, low=STOP - 0.01)
        assert cycle(trader, state, tracker) == "sold"
        for _ in range(14):
            tape.append(98.6)
            cycle(trader, state, tracker)
        return trader, tracker, tape

    def test_after_a_stop_a_buy_needs_the_raised_edge(self, monkeypatch, state):
        trader, tracker, tape = self._stopped(monkeypatch, state)
        assert trader.enter_at() == pytest.approx(ENTER + self.RAISE)
        tape.append(98.6, edge=ENTER + self.RAISE - 0.01)    # bar 76: over the entry, under the raise
        assert cycle(trader, state, tracker) == "hold"
        tape.append(98.6, edge=ENTER + self.RAISE)
        assert cycle(trader, state, tracker) == "bought"
        buy = [e for e in state.agent_log if e.get("type") == "decision" and e.get("action") == "buy"][-1]
        assert "raised by the circuit breaker after 1 stop today" in buy["reasoning"]

    def test_the_stop_says_what_the_next_buy_needs(self, monkeypatch, state):
        trader, tracker, tape = self._stopped(monkeypatch, state)
        stop = [e for e in state.agent_log if e.get("type") == "decision" and e.get("action") == "sell"][-1]
        assert f"a buy needs an edge of {ENTER + self.RAISE:+.3f}" in stop["reasoning"]

    def test_a_refusal_is_said_once(self, monkeypatch, state):
        trader, tracker, tape = self._stopped(monkeypatch, state)
        for _ in range(3):
            tape.append(98.6, edge=ENTER + 0.01)
            assert cycle(trader, state, tracker) == "hold"
        said = [e for e in state.agent_log if "the circuit breaker asks for" in (e.get("text") or "")]
        assert len(said) == 1

    def test_every_stop_raises_it_again(self, monkeypatch, state):
        trader, tracker, tape = self._stopped(monkeypatch, state)
        tape.append(98.6, edge=0.2)                            # bar 76: bought
        assert cycle(trader, state, tracker) == "bought"
        tape.append(96.0, low=98.6 * math.exp(-0.75 * ADR) - 0.01)   # bar 77: stopped again
        assert cycle(trader, state, tracker) == "sold"
        assert trader.enter_at() == pytest.approx(ENTER + 2 * self.RAISE)

    def test_switched_off_the_pause_is_all_a_stop_does(self, monkeypatch, state):
        trader, tracker, tape = self._stopped(monkeypatch, state, gm_stop_raise=0.0)
        assert trader.enter_at() == pytest.approx(ENTER)
        tape.append(98.6, edge=ENTER)
        assert cycle(trader, state, tracker) == "bought"

    def test_a_new_session_starts_from_the_entry(self, monkeypatch, state):
        trader, tracker, tape = self._stopped(monkeypatch, state)
        trader._roll_session(pd.Timestamp("2026-07-22"))
        assert trader.enter_at() == pytest.approx(ENTER)

    def test_the_raised_entry_survives_a_restart(self, monkeypatch, state):
        trader, tracker, tape = self._stopped(monkeypatch, state)
        again = at.build_trader(gm_config(), BUNDLE)
        tape.append(98.6, edge=ENTER + self.RAISE - 0.01)      # bar 76
        assert cycle(again, state, tracker) == "hold"
        assert again.enter_at() == pytest.approx(ENTER + self.RAISE)

    def test_the_record_carries_the_entry_of_each_bar(self, monkeypatch, state):
        trader, tracker, tape = self._stopped(monkeypatch, state)
        rows = state.apple_trader_levels["rows"]
        # The stop bar's row is written before its stop is read.
        assert [r["enter"] for r in rows[:3]] == pytest.approx([ENTER, ENTER, ENTER + self.RAISE])

    def test_its_signature_and_its_checks(self):
        from simlab.rule_agents import _apple_from_record

        assert ",raise=0.05/stop," in config_signature(gm_config())
        assert "raise=" not in config_signature(gm_config(gm_stop_raise=0.0))
        old = _apple_from_record({"model_key": "goodmoments", "ticker": TICKER, "gm_enter": 0.08})
        assert old.gm_stop_raise == 0.0 and "raise=" not in config_signature(old)
        with pytest.raises(ValueError, match="gm_stop_raise"):
            gm_config(gm_stop_raise=-0.01)

    def test_tuning_offers_it_on_goodmoments(self):
        from simlab import tuning

        assert "gm_stop_raise" in tuning.TUNABLES
        assert "gm_stop_raise" in tuning.axes_for(apple_models.GOODMOMENTS_KEY, ["gm_stop_raise"])
        assert tuning.axes_for("highlow", ["gm_stop_raise"]) == []

    def test_the_chart_steps_the_entry_threshold(self):
        from agent_stonks import model_overlays as mo

        stamps = [Tape.OPEN + pd.Timedelta(minutes=m) for m in (60, 61, 62, 63)]
        fc = pd.DataFrame(
            {"edge": [0.03, 0.03, 0.06, 0.08], "p": [0.5] * 4, "p_tv": [0.47] * 4,
             "price": [100.0] * 4},
            index=[60, 61, 62, 63],
        )
        rows = [{"t": stamps[0], "enter": 0.02}, {"t": stamps[2], "enter": 0.07}]
        result = {"fc": fc, "stamps": stamps, "enter": 0.02, "exit": None, "raise": 0.05,
                  "problem": ""}
        items, _ = mo._goodmoments_signal_items(result, rows)
        line = next(i for i in items if i["label"] == "Entry threshold")
        assert line["values"] == pytest.approx([2.0, 2.0, 7.0, 7.0])
        # 62 is under the raised entry: the 63 bar starts a new run of buy signals.
        markers = [i["note"] for i in items if i["kind"] == "event"]
        assert len(markers) == 2
        assert "+0.02 entry" in markers[0] and "+0.07 entry" in markers[1]
        # A replay's chart has no record to step: the note says what a stop does.
        items, _ = mo._goodmoments_signal_items(result)
        line = next(i for i in items if i["label"] == "Entry threshold")
        assert set(line["values"]) == {2.0} and "+0.05 more after each stop" in line["note"]


class TestMomentumConfirmation:
    """HighLow's behaviour table on GoodMoments' rules (`gm_confirm_bars`): no
    entry the table does not allow, and a reached take held while momentum is
    positive -- sold on the first bar it is not, wherever the price is."""

    N = 3
    MINUTE_MOVE = 0.10                      # neutral band ±0.01 $/bar

    def _setup(self, monkeypatch, state, **config):
        trader, tracker, tape = _setup(monkeypatch, state, gm_confirm_bars=self.N, **config)
        state.sym(TICKER).abs_mean_minute_momentum = self.MINUTE_MOVE
        return trader, tracker, tape

    @staticmethod
    def _run(trader, state, tracker, tape, closes, edge=0.0) -> str:
        """One bar per close, the edge only on the last; the last outcome."""
        for i, close in enumerate(closes):
            tape.append(close, edge=edge if i == len(closes) - 1 else 0.0)
            outcome = cycle(trader, state, tracker)
        return outcome

    def _bought_rising(self, monkeypatch, state, **config):
        trader, tracker, tape = self._setup(monkeypatch, state, **config)
        assert self._run(trader, state, tracker, tape, [100.0, 100.1, 100.2, 100.3, 100.4], ENTER) == "bought"
        return trader, tracker, tape

    @staticmethod
    def _statuses(state) -> "list[str]":
        return [e["text"] for e in state.agent_log if e.get("type") == "status"]

    def test_negative_momentum_does_not_buy_and_says_so_once(self, monkeypatch, state):
        trader, tracker, tape = self._setup(monkeypatch, state)
        assert self._run(trader, state, tracker, tape, [100.5, 100.4, 100.3, 100.2, 100.1], ENTER) == "hold"
        tape.append(100.0, edge=ENTER)
        assert cycle(trader, state, tracker) == "hold"
        assert tracker.position_for(TICKER) == 0
        refusals = [t for t in self._statuses(state) if "Not buying yet" in t]
        assert len(refusals) == 1 and "(negative)" in refusals[0]

    def test_positive_or_flat_momentum_buys(self, monkeypatch, state):
        self._bought_rising(monkeypatch, state)
        trader, tracker, tape = self._setup(monkeypatch, state)
        assert self._run(trader, state, tracker, tape, [100.0] * 5, ENTER) == "bought"

    def test_the_falling_stretch_ends_in_a_buy_once_momentum_turns(self, monkeypatch, state):
        trader, tracker, tape = self._setup(monkeypatch, state)
        self._run(trader, state, tracker, tape, [100.5, 100.4, 100.3, 100.2, 100.1], ENTER)
        assert self._run(trader, state, tracker, tape, [100.2, 100.3, 100.4], ENTER) == "bought"

    def test_before_the_look_back_it_does_not_buy(self, monkeypatch, state):
        trader, tracker, tape = self._setup(monkeypatch, state)
        assert self._run(trader, state, tracker, tape, [100.0, 100.1, 100.2], ENTER) == "hold"
        assert any("needs 5 bars" in t for t in self._statuses(state))

    def test_without_last_weeks_minute_move_it_does_not_buy(self, monkeypatch, state):
        trader, tracker, tape = self._setup(monkeypatch, state)
        state.sym(TICKER).abs_mean_minute_momentum = None
        assert self._run(trader, state, tracker, tape, [100.0, 100.1, 100.2, 100.3, 100.4], ENTER) == "hold"
        assert any("not known yet" in t for t in self._statuses(state))

    def test_a_reached_take_is_held_while_momentum_rises_then_sold(self, monkeypatch, state):
        trader, tracker, tape = self._bought_rising(monkeypatch, state)
        take = trader.entry["take"]
        tape.append(100.85, high=take + 0.1)                       # through the take, still rising
        assert cycle(trader, state, tracker) == "hold"
        assert trader.entry["take_reached"] and tracker.position_for(TICKER) > 0
        tape.append(100.95)
        assert cycle(trader, state, tracker) == "hold"
        assert len([t for t in self._statuses(state) if "Holding on" in t]) == 1
        tape.append(100.90)
        tape.append(100.80)                                        # under the take, momentum turned
        assert cycle(trader, state, tracker) == "sold"
        assert _exits(state) == [at.GoodMomentsTrader.EXIT_TAKE]
        assert tracker.position_for(TICKER) == 0

    def test_a_take_reached_on_fading_momentum_sells_at_once(self, monkeypatch, state):
        trader, tracker, tape = self._bought_rising(monkeypatch, state)
        take = trader.entry["take"]
        for close in (100.4, 100.4, 100.4):
            tape.append(close)
            cycle(trader, state, tracker)
        tape.append(100.4, high=take + 0.01)                       # a spike on a flat tape
        assert cycle(trader, state, tracker) == "sold"
        assert _exits(state) == [at.GoodMomentsTrader.EXIT_TAKE]

    def test_no_read_holds_the_take(self, monkeypatch, state):
        trader, tracker, tape = self._bought_rising(monkeypatch, state)
        state.sym(TICKER).abs_mean_minute_momentum = None
        tape.append(100.4, high=trader.entry["take"] + 0.01)
        assert cycle(trader, state, tracker) == "hold"
        assert tracker.position_for(TICKER) > 0

    def test_the_stop_is_never_held(self, monkeypatch, state):
        trader, tracker, tape = self._bought_rising(monkeypatch, state)
        tape.append(100.85, high=trader.entry["take"] + 0.1)
        assert cycle(trader, state, tracker) == "hold"
        tape.append(98.0, low=100.4 * math.exp(-0.75 * ADR) - 0.01)
        assert cycle(trader, state, tracker) == "sold"
        assert _exits(state) == [at.GoodMomentsTrader.EXIT_STOP]

    def test_the_flatten_is_never_held(self, monkeypatch, state):
        trader, tracker, tape = self._bought_rising(monkeypatch, state)
        tape.append(100.85, high=trader.entry["take"] + 0.1)
        monkeypatch.setattr(at.GoodMomentsTrader, "closing_soon", lambda self: True)
        assert cycle(trader, state, tracker) == "sold"
        assert _exits(state) == [at.GoodMomentsTrader.EXIT_FLATTEN]

    def test_a_held_take_survives_a_restart(self, monkeypatch, state):
        trader, tracker, tape = self._bought_rising(monkeypatch, state)
        tape.append(100.85, high=trader.entry["take"] + 0.1)
        cycle(trader, state, tracker)
        again = at.build_trader(gm_config(gm_confirm_bars=self.N), BUNDLE)
        tape.append(100.20)                                        # under the take, turned down
        assert cycle(again, state, tracker) == "sold"
        assert _exits(state) == [at.GoodMomentsTrader.EXIT_TAKE]

    def test_off_it_is_the_notebooks_rule(self, monkeypatch, state):
        trader, tracker, tape = _setup(monkeypatch, state)
        state.sym(TICKER).abs_mean_minute_momentum = None
        tape.append(100.0, edge=ENTER)
        assert cycle(trader, state, tracker) == "bought"

    def test_its_signature_checks_and_legacy(self):
        from dataclasses import asdict

        from simlab import tuning

        assert gm_config().gm_confirm_bars == 0
        assert config_signature(gm_config(gm_confirm_bars=3)) == (
            "goodmoments_AAPL(enter=+0.02,exit=off,take=0.2A,stop=0.75A,raise=0.05/stop,confirm=3b,size=95%)"
        )
        for bad in (-1, 1.5):
            with pytest.raises(ValueError, match="gm_confirm_bars"):
                gm_config(gm_confirm_bars=bad)
        assert "gm_confirm_bars" in tuning.axes_for("goodmoments", tuning.SWEEPABLE)
        old = {k: v for k, v in asdict(gm_config()).items() if k != "gm_confirm_bars"}
        assert tuning.make_config(old, {}).gm_confirm_bars == 0

    def test_the_live_chart_draws_its_look_back(self, state):
        from agent_stonks import ui

        state.llm_personality = at.APPLE_TRADER_KEY
        state.apple_trader_config = gm_config(gm_confirm_bars=4)
        assert ui._agent_momentum(state, state.sym(TICKER))[0] == 4
