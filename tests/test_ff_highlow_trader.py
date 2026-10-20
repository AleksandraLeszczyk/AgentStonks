"""Apple Trader on Fast&Furious + HighLow: Fast&Furious's signal for the first
five bars, HighLow's day-range rules after them.

Driven through both stubbed forecasts -- the day-range one `tests/test_apple_trader.py`
uses (a $10 ADR under a $110 predicted high: buy $102.50, sell $109.00) and a
scripted Fast&Furious forecast per minute -- so these pin the RULES: what the
window buys on, how it gets out inside the window, what is handed to the 9:35
levels, and how the model is kept apart from what it cannot do (Orchestra,
a mid-run model switch).
"""

import math
import threading
from dataclasses import replace
from datetime import date, datetime, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from agent_stonks import apple_models, base_configs, clock, model_overlays, orchestra
from agent_stonks import apple_trader as at
from agent_stonks.apple_trader import AppleTraderConfig, config_signature
from agent_stonks.decisions import DecisionTracker
from agent_stonks.state import AppState
from simlab import tuning as sim_tuning
from tests.test_apple_trader import (
    BUY_LEVEL, DAYRANGE_BUNDLE, SELL_LEVEL, TARGET_GAIN, FakeBroker, Tape, dayrange_config,
)

TICKER = "AAPL"
KEY = apple_models.FF_HIGHLOW_KEY
# 10:30 ET on a Tuesday: the clock the market check reads. The window is read
# off the bars' own times, so it does not need the clock at 09:31.
MIDSESSION = datetime(2026, 7, 21, 14, 30, tzinfo=timezone.utc)
THETA = 0.04
S120 = 0.005
DOWN = 0.4          # the window's stop: 1.5 x 0.4 x 0.005 = 0.3% under the fill
FF_BUNDLE = {"kind": "fastfurious", "path": "stub"}
# The 9:35 rules' stop is half the 0.65-ADR predicted gain under the fill.
HALF_GAIN = 0.5 * TARGET_GAIN


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


class Signals:
    """Fast&Furious's forecast scripted per minute (y=0 where not set), and
    every minute it was asked for."""

    def __init__(self, monkeypatch):
        self.y: "dict[int, float]" = {}
        self.down: "dict[int, float]" = {}
        self.threshold = (THETA, 20)
        self.asked: "list[int]" = []
        self.threshold_calls = 0
        self.warmed = threading.Event()
        signals = self

        def forecast_at(bundle, ticker, day, minute, q, key=None, secret=None):
            assert bundle is FF_BUNDLE
            signals.asked.append(minute)
            y = signals.y.get(minute, 0.0)
            theta = signals.threshold[0]
            return {
                "minute": minute, "y_hat": y, "mom_bps": y * 0.003 / 30 * 1e4, "price": 100.0,
                "theta": theta, "theta_bps": theta * 0.003 / 30 * 1e4, "theta_sessions": 20,
                "signal": 1 if y >= theta else -1 if y <= -theta else 0,
                "up_hat": 0.5, "down_hat": signals.down.get(minute, DOWN), "s30": 0.003,
                "s120": S120, "pred_high": 100.2, "pred_low": 99.8, "range_problem": "",
            }

        def threshold(bundle, ticker, day, q, key=None, secret=None):
            signals.threshold_calls += 1
            return signals.threshold

        fake = SimpleNamespace(
            forecast_at=forecast_at, threshold=threshold, LAST_ORIGIN=359,
            THRESHOLD_MIN_SESSIONS=5, is_half_day=lambda day: False,
            nyse_sessions=lambda days: list(days),
            warm=lambda *a, **k: signals.warmed.set(),
        )
        monkeypatch.setattr(at, "_fastfurious", lambda: fake)
        real = apple_models.load
        monkeypatch.setattr(
            apple_models, "load",
            lambda key, ticker=None: FF_BUNDLE if key == apple_models.FASTFURIOUS_KEY
            else real(key, ticker),
        )


def ffhl_config(**kwargs) -> AppleTraderConfig:
    """`dayrange_config`'s pinned rules (buy 0.75, sell 0.10, half-gain stop,
    no breaker, no confirmation, market buys) on this model, with
    Fast&Furious's notebook rules for the window."""
    return replace(dayrange_config(**kwargs), model_key=KEY)


def _setup(monkeypatch, **config):
    """A trader with the 09:30 bar in (close $101.00, low $100.90): the
    window's first bar, not yet read."""
    broker = FakeBroker(101.0)
    tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
    tape = Tape(monkeypatch, broker, minutes=1)
    signals = Signals(monkeypatch)
    trader = at.build_trader(ffhl_config(**config), DAYRANGE_BUNDLE)
    assert isinstance(trader, at.FastFuriousHighLowTrader)
    return trader, tracker, tape, signals


def cycle(trader, state, tracker) -> str:
    return trader.run_cycle(DAYRANGE_BUNDLE, state, tracker)


def _exits(state) -> "list[str]":
    return [
        e.get("exit") for e in state.agent_log
        if e.get("type") == "decision" and e.get("action") == "sell"
    ]


def _bought_at_0930(monkeypatch, state, **config):
    trader, tracker, tape, signals = _setup(monkeypatch, **config)
    signals.y[0] = THETA
    assert cycle(trader, state, tracker) == "bought"
    return trader, tracker, tape, signals


def _quiet_to_0934(trader, state, tracker, tape, close=101.2, low=101.0):
    """The 09:31 to 09:34 bars with no signal; returns the 09:34 outcome."""
    outcome = None
    for offset in (1, 2, 3, 4):
        tape.append(close, low=low, high=close + 0.2, offset=offset)
        outcome = cycle(trader, state, tracker)
    return outcome


class TestWindowEntry:
    def test_a_buy_signal_in_the_first_five_minutes_buys_at_market(self, monkeypatch, state):
        trader, tracker, tape, signals = _bought_at_0930(monkeypatch, state)
        fill = trader.entry["price"]
        assert trader.entry["early"] and trader.plan is None
        assert trader.entry["stop"] == pytest.approx(fill * math.exp(-1.5 * DOWN * S120))
        assert tape.forecast_calls == 0 and signals.threshold_calls == 1
        decision = tracker.snapshot()["decisions"][-1]
        assert decision.limit_price is None  # Fast&Furious buys at market
        assert "Fast&Furious window" in decision.reasoning

    def test_without_a_signal_nothing_is_bought_however_low_the_bar(self, monkeypatch, state):
        trader, tracker, tape, _ = _setup(monkeypatch)
        tape.rows[-1]["low"] = 90.0  # far through any level, which do not exist yet
        assert cycle(trader, state, tracker) == "hold"
        assert tracker.position_for(TICKER) == 0

    def test_after_the_window_the_signal_is_not_read(self, monkeypatch, state):
        trader, tracker, tape, signals = _setup(monkeypatch)
        cycle(trader, state, tracker)
        _quiet_to_0934(trader, state, tracker, tape, close=104.0, low=103.9)
        signals.y[5] = THETA
        tape.append(104.0, low=103.9, offset=5)  # above the $102.50 buy level
        assert cycle(trader, state, tracker) == "hold"
        assert tracker.position_for(TICKER) == 0
        assert signals.asked == [0, 1, 2, 3, 4]

    def test_a_buy_on_the_0934_bar_is_handed_over_at_once(self, monkeypatch, state):
        trader, tracker, tape, signals = _setup(monkeypatch)
        cycle(trader, state, tracker)
        for offset in (1, 2, 3):
            tape.append(101.2, low=101.0, offset=offset)
            cycle(trader, state, tracker)
        signals.y[4] = THETA
        tape.append(101.5, low=101.3, offset=4)
        assert cycle(trader, state, tracker) == "bought"
        assert trader.plan is not None and "early" not in trader.entry
        assert trader.entry["stop"] == pytest.approx(101.5 - HALF_GAIN)

    def test_a_run_started_after_the_window_never_reads_it(self, monkeypatch, state):
        trader, tracker, tape, signals = _setup(monkeypatch)
        for offset in (1, 2, 3, 4, 5):
            tape.append(104.0, low=103.9, offset=offset)
        cycle(trader, state, tracker)
        assert signals.asked == [] and trader.early is None and trader.plan is not None


class TestHandOver:
    def test_a_position_open_at_935_is_managed_by_highlows_rules(self, monkeypatch, state):
        trader, tracker, tape, _ = _bought_at_0930(monkeypatch, state)
        fill = trader.entry["price"]
        assert _quiet_to_0934(trader, state, tracker, tape) == "hold"
        assert trader.plan is not None
        for name in ("early", "due", "y_hat"):
            assert name not in trader.entry
        assert trader.entry["stop"] == pytest.approx(fill - HALF_GAIN)
        assert trader.entry["ts"] == Tape.OPEN  # the take is timed from the 09:30 fill
        handed = [e["text"] for e in state.agent_log if "are now managed by" in e.get("text", "")]
        assert handed and "Fast&Furious's opening window" in handed[0]

        tape.append(109.1, low=108.5, high=109.2, offset=5)
        assert cycle(trader, state, tracker) == "sold"
        assert _exits(state) == [at.DayRangeTrader.EXIT_TARGET]

    def test_highlows_stop_applies_after_935_and_stops_the_run(self, monkeypatch, state):
        trader, tracker, tape, _ = _bought_at_0930(monkeypatch, state)
        fill = trader.entry["price"]
        _quiet_to_0934(trader, state, tracker, tape)
        tape.append(fill - HALF_GAIN - 0.2, offset=5)
        assert cycle(trader, state, tracker) == "sold"
        assert _exits(state) == [at.DayRangeTrader.EXIT_STOP]
        assert trader.halt == at.DayRangeTrader.STOPPED_OUT

    def test_flat_at_935_the_levels_trade_as_on_highlow(self, monkeypatch, state):
        trader, tracker, tape, _ = _setup(monkeypatch)
        cycle(trader, state, tracker)
        _quiet_to_0934(trader, state, tracker, tape, close=104.0, low=103.9)
        tape.append(103.0, low=BUY_LEVEL - 0.01, offset=5)
        assert cycle(trader, state, tracker) == "bought"
        assert "early" not in trader.entry


class TestWindowExits:
    def test_the_windows_stop_sells_and_does_not_stop_the_run(self, monkeypatch, state):
        trader, tracker, tape, _ = _bought_at_0930(monkeypatch, state)
        stop = trader.entry["stop"]
        tape.append(stop - 0.1, low=stop - 0.2, offset=1)
        assert cycle(trader, state, tracker) == "sold"
        assert _exits(state) == [at.FastFuriousHighLowTrader.EXIT_WINDOW_STOP]
        assert trader.halt is None
        # The stop bar's own forecast was out of the buy zone: re-armed at once,
        # as under Fast&Furious.
        assert not trader.early["blocked"]

    def test_after_a_window_stop_the_buy_waits_for_the_forecast_to_leave_the_zone(
        self, monkeypatch, state
    ):
        trader, tracker, tape, signals = _bought_at_0930(monkeypatch, state)
        stop = trader.entry["stop"]
        signals.y.update({1: THETA, 2: THETA, 3: 0.0, 4: THETA})
        tape.append(stop - 0.1, low=stop - 0.2, offset=1)
        assert cycle(trader, state, tracker) == "sold"
        tape.append(100.5, low=100.4, offset=2)
        assert cycle(trader, state, tracker) == "hold"           # still in the zone
        tape.append(100.5, low=100.4, offset=3)
        assert cycle(trader, state, tracker) == "hold"           # leaves it
        tape.append(100.5, low=100.4, offset=4)
        assert cycle(trader, state, tracker) == "bought"

    def test_a_sell_signal_in_the_window_closes_the_long(self, monkeypatch, state):
        trader, tracker, tape, signals = _bought_at_0930(monkeypatch, state)
        signals.y[1] = -THETA
        tape.append(101.1, low=101.0, offset=1)
        assert cycle(trader, state, tracker) == "sold"
        assert _exits(state) == [at.FastFuriousHighLowTrader.EXIT_WINDOW_SELL_SIGNAL]
        assert tracker.position_for(TICKER) == 0

    def test_a_window_trade_does_not_trip_the_breaker(self, monkeypatch, state):
        """A round trip for a few cents is under any breaker, but the breaker
        is about the 9:35 levels: they stay armed."""
        trader, tracker, tape, signals = _bought_at_0930(monkeypatch, state, min_win_k=0.5)
        signals.y[1] = -THETA
        tape.append(101.05, low=101.0, offset=1)
        assert cycle(trader, state, tracker) == "sold"
        for offset in (2, 3, 4):
            tape.append(104.0, low=103.9, offset=offset)
            cycle(trader, state, tracker)
        assert trader.plan is not None and not trader.plan.get("stand_down")
        tape.append(103.0, low=BUY_LEVEL - 0.01, offset=5)
        assert cycle(trader, state, tracker) == "bought"

    def test_a_signal_while_long_renews_the_clock_and_buys_nothing_more(
        self, monkeypatch, state
    ):
        trader, tracker, tape, signals = _bought_at_0930(monkeypatch, state)
        held = tracker.position_for(TICKER)
        signals.y[2] = THETA
        tape.append(101.1, low=101.0, offset=1)
        cycle(trader, state, tracker)
        tape.append(101.1, low=101.0, offset=2)
        assert cycle(trader, state, tracker) == "hold"
        assert trader.entry["due"] == 2 + 30 and tracker.position_for(TICKER) == held


class TestWindowFailures:
    def test_no_threshold_costs_only_the_window(self, monkeypatch, state):
        trader, tracker, tape, signals = _setup(monkeypatch)
        signals.threshold = (float("nan"), 3)
        signals.y[0] = THETA
        assert cycle(trader, state, tracker) == "warming_up"
        errors = [e["text"] for e in state.agent_log if e.get("type") == "error"]
        assert len(errors) == 1
        assert "nothing is bought before HighLow model's 9:35 forecast" in errors[0]
        _quiet_to_0934(trader, state, tracker, tape, close=104.0, low=103.9)
        assert tracker.position_for(TICKER) == 0 and signals.threshold_calls == 1
        tape.append(103.0, low=BUY_LEVEL - 0.01, offset=5)
        assert cycle(trader, state, tracker) == "bought"

    def test_no_935_forecast_sells_what_the_window_bought(self, monkeypatch, state):
        trader, tracker, tape, _ = _bought_at_0930(monkeypatch, state)

        def boom(*a, **k):
            raise RuntimeError("history too short")

        monkeypatch.setattr(at._dayrange(), "forecast_session", boom)
        assert _quiet_to_0934(trader, state, tracker, tape) == "no_data"
        assert tracker.position_for(TICKER) == 0
        assert _exits(state) == [at.FastFuriousHighLowTrader.EXIT_EARLY_DROPPED]

    def test_a_day_sat_out_is_sat_out_from_the_first_bar(self, monkeypatch, state):
        monkeypatch.setattr(at.event_days, "load_calendar", lambda *a, **k: [{
            "date": "2026-07-21", "category": "cpi", "scope": "ALL",
            "event": "BLS Consumer Price Index release, 08:30 ET", "source": "",
        }])
        trader, tracker, tape, signals = _setup(
            monkeypatch, skip_events=at.event_days.CATEGORIES
        )
        signals.y[0] = THETA
        assert cycle(trader, state, tracker) == "no_data"
        assert trader.blocked["sit_out"] == ["cpi"]
        assert signals.asked == [] and tracker.position_for(TICKER) == 0

    def test_a_forecast_that_fails_is_said_once_and_buys_nothing(self, monkeypatch, state):
        trader, tracker, tape, signals = _setup(monkeypatch)

        def down(*a, **k):
            raise ValueError("IEX is down")

        monkeypatch.setattr(at._fastfurious(), "forecast_at", down)
        cycle(trader, state, tracker)
        tape.append(101.2, low=101.0, offset=1)
        cycle(trader, state, tracker)
        errors = [e["text"] for e in state.agent_log if e.get("type") == "error"]
        assert len(errors) == 1 and "IEX is down" in errors[0]
        assert tracker.position_for(TICKER) == 0


class TestRestartAndRecord:
    def test_a_restart_inside_the_window_keeps_the_positions_stop(self, monkeypatch, state):
        trader, tracker, tape, signals = _bought_at_0930(monkeypatch, state)
        entry = dict(trader.entry)
        again = at.build_trader(ffhl_config(), DAYRANGE_BUNDLE)
        tape.append(101.1, low=101.0, offset=1)
        assert cycle(again, state, tracker) == "hold"
        assert again.entry["early"] and again.entry["stop"] == pytest.approx(entry["stop"])

    def test_the_record_starts_at_935_and_remembers_the_window_position(
        self, monkeypatch, state
    ):
        trader, tracker, tape, _ = _bought_at_0930(monkeypatch, state)
        record = state.apple_trader_levels
        assert record["rows"] == [] and record["memory"]["entry"]["early"]
        _quiet_to_0934(trader, state, tracker, tape)
        tape.append(101.3, low=101.1, offset=5)
        cycle(trader, state, tracker)
        rows = state.apple_trader_levels["rows"]
        assert [(r["t"].minute, r["model_key"]) for r in rows] == [(34, KEY), (35, KEY)]
        assert rows[0]["buy"] == pytest.approx(BUY_LEVEL)
        assert rows[0]["sell"] == pytest.approx(SELL_LEVEL)
        assert rows[-1]["stop"] == pytest.approx(trader.entry["price"] - HALF_GAIN)

    def test_the_caches_are_warmed_before_the_window(self, monkeypatch, state):
        trader, tracker, _, signals = _setup(monkeypatch)
        clock.set_simulated(datetime(2026, 7, 21, 13, 20, tzinfo=timezone.utc))  # 09:20 ET
        cycle(trader, state, tracker)
        assert signals.warmed.wait(2.0)

    def test_a_running_highlow_agent_does_not_switch_to_it(self, monkeypatch, state):
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(104.0))
        tape = Tape(monkeypatch, tracker.broker)
        trader = at.DayRangeTrader(replace(ffhl_config(), model_key="highlow"))
        state.apple_trader_config = ffhl_config()
        tape.append(104.0, low=103.9)
        trader.run_cycle(DAYRANGE_BUNDLE, state, tracker)
        assert trader.config.model_key == "highlow"
        errors = [e["text"] for e in state.agent_log if e.get("type") == "error"]
        assert any("opening window differently" in t for t in errors)


class TestConfig:
    HIGHLOW_BASE = {
        "model_key": "highlow", "ticker": TICKER, "buy_k": 0.85, "sell_k": 0.1,
        "stop_gain_fraction": 0.8, "level_unit": "pred_range", "breach_update": "brownian",
        "min_win_k": 0.1,
        # A field since removed (HighLow_3m's window, 2026-10-20): a stored
        # base still carrying it opens all the same.
        "use_3m": True,
    }

    def _bases(self, monkeypatch, pairs: dict) -> None:
        monkeypatch.setattr(
            base_configs, "stored",
            lambda model_key, ticker: dict(pairs[model_key]) if model_key in pairs else None,
        )

    def test_it_opens_on_highlows_base_configuration(self, monkeypatch):
        self._bases(monkeypatch, {"highlow": self.HIGHLOW_BASE})
        config = at.default_config(TICKER, KEY)
        assert config.model_key == KEY
        assert (config.buy_k, config.sell_k, config.stop_gain_fraction) == (0.85, 0.1, 0.8)
        assert config.breach_update == "brownian" and config.level_unit == "pred_range"
        assert not hasattr(config, "use_3m")
        assert at.config_error(config) is None
        assert at.base_record(KEY, TICKER)[1] == "highlow"

    def test_its_own_base_wins(self, monkeypatch):
        own = {**self.HIGHLOW_BASE, "model_key": KEY, "buy_k": 0.6}
        self._bases(monkeypatch, {"highlow": self.HIGHLOW_BASE, KEY: own})
        assert at.default_config(TICKER, KEY).buy_k == 0.6
        assert at.base_record(KEY, TICKER)[1] == KEY

    def test_without_a_base_it_starts_from_highlows_tuned_numbers(self, monkeypatch):
        self._bases(monkeypatch, {})
        assert at.dayrange_levels(TICKER, KEY) == at.dayrange_levels(TICKER, "highlow")
        assert at.stop_for(TICKER, KEY) == at.stop_for(TICKER, "highlow")

    def test_the_signature_names_the_windows_rules(self):
        on = config_signature(ffhl_config())
        assert on.startswith(f"{KEY}_{TICKER}(")
        assert ",ff=[q=0.95,stop=1.5dn,hold=30m,sellexit]," in on
        assert "ff=" not in config_signature(replace(ffhl_config(), model_key="highlow"))

    def test_the_armed_line_says_what_the_window_does(self):
        line = at._armed_summary(ffhl_config(), apple_models.get(KEY), DAYRANGE_BUNDLE)
        assert "on the 09:30 to 09:34 bars" in line
        assert "handed to that forecast's levels" in line

    def test_it_needs_both_bundles(self, monkeypatch):
        monkeypatch.setattr(apple_models, "_load_highlow", lambda t: {"kind": "highlow"})
        monkeypatch.setattr(apple_models, "_load_fastfurious", lambda t: None)
        assert apple_models.load(KEY, TICKER) is None
        assert "Fast&Furious" in apple_models.unavailable_reason(KEY, TICKER)
        monkeypatch.setattr(apple_models, "_load_fastfurious", lambda t: FF_BUNDLE)
        assert apple_models.load(KEY, TICKER) == {"kind": "highlow"}

    def test_the_registry_names_its_parts(self):
        assert apple_models.forecast_key(KEY) == "highlow"
        assert apple_models.window_key(KEY) == "fastfurious"
        assert apple_models.parts(KEY) == ("highlow", "fastfurious")
        assert apple_models.parts("highlow") == ("highlow",)
        assert apple_models.strategy(KEY) == apple_models.STRATEGY_DAYRANGE
        assert apple_models.MODELS[KEY].tickers == (TICKER,)


class TestFenced:
    def test_orchestra_neither_offers_nor_takes_it(self):
        assert f"{TICKER}:{KEY}" not in orchestra.all_pairs()
        with pytest.raises(ValueError, match="opening window"):
            orchestra.OrchestraConfig([ffhl_config()])

    def test_tuning_offers_the_windows_threshold_and_stop(self):
        axes = sim_tuning.axes_for(KEY, sim_tuning.AXES)
        assert "ff_entry_quantile" in axes and "ff_stop_k" in axes and "buy_k" in axes
        assert "ff_hold_minutes" not in axes
        assert "ff_entry_quantile" not in sim_tuning.axes_for("highlow", sim_tuning.AXES)
        assert sim_tuning.axes_family(KEY) != sim_tuning.axes_family("highlow")

    def test_a_result_is_stale_when_either_model_changes(self):
        paths = sim_tuning._replay_inputs(ffhl_config(), [], "iex")
        names = {p.name for p in paths}
        assert any(n.startswith("highlow15m_AAPL") for n in names)
        assert any(n.startswith("fastfurious_AAPL") for n in names)

    def test_its_chart_draws_highlows_range_and_fast_and_furious_signals(self):
        keys = model_overlays.for_models([KEY], TICKER)["keys"]
        assert model_overlays.HIGHLOW_RANGE_KEY in keys
        assert model_overlays.FASTFURIOUS_SIGNAL_KEY in keys


# --- the form both apps render ---------------------------------------------------


def _form() -> None:
    import streamlit as st

    from agent_stonks import apple_trader_ui

    copy = apple_trader_ui.FormCopy(
        prefix="t", unavailable_suffix="", instrument_help="", model_help=""
    )
    st.session_state["out"] = apple_trader_ui.params(["AAPL"], copy)


class TestForm:
    def _picked(self, monkeypatch):
        from streamlit.testing.v1 import AppTest

        # No bundle is loaded to draw a form.
        monkeypatch.setattr(apple_models, "load", lambda *a, **k: {"model": "stub"})
        app = AppTest.from_function(_form, default_timeout=30)
        app.run()
        app.selectbox(key="t_model_AAPL").select(KEY).run()
        assert not app.exception, app.exception
        return app

    def test_it_draws_highlows_levels_and_the_windows_rules(self, monkeypatch):
        app = self._picked(monkeypatch)
        assert app.number_input(key=f"t_buy_k_{KEY}_AAPL").value == pytest.approx(
            at.dayrange_levels(TICKER, "highlow")[0]
        )
        quantile = app.number_input(key=f"t_ff_entry_quantile_{KEY}_AAPL")
        assert quantile.value == pytest.approx(0.95)
        quantile.set_value(0.98).run()
        out = app.session_state["out"]
        assert out.model_key == KEY and out.ff_entry_quantile == pytest.approx(0.98)

    def test_it_opens_on_highlows_base(self, monkeypatch):
        base = replace(
            AppleTraderConfig(ticker=TICKER, model_key="highlow"), buy_k=0.85, sell_k=0.15
        )
        from dataclasses import asdict

        base_configs.set_base(asdict(base))
        app = self._picked(monkeypatch)
        suffix = base_configs.key_suffix("highlow", TICKER)
        assert suffix
        assert app.number_input(key=f"t_buy_k_{KEY}_AAPL{suffix}").value == pytest.approx(0.85)
        assert app.session_state["out"].model_key == KEY
