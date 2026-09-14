"""Apple Trader: the rule-based loop over the day-range forecast.

Every cycle is driven through a stubbed forecast, so these pin the RULES --
when it buys, when it refuses to, and every way it gets back out -- without
depending on the saved artifact or on live market data.
"""

import threading
from dataclasses import replace
from datetime import datetime, timezone

import pandas as pd
import pytest

from agent_stonks import apple_models
from agent_stonks import apple_trader as at
from agent_stonks import clock
from agent_stonks import rule_agent
from agent_stonks.apple_trader import DEFAULT_TICKER as TICKER
from agent_stonks.apple_trader import AppleTraderConfig, config_signature
from agent_stonks.broker import Broker
from agent_stonks.decisions import DecisionTracker
from agent_stonks.state import AppState

# 10:30 ET on a Tuesday: mid-session, well clear of both the open and the close.
MIDSESSION = datetime(2026, 7, 21, 14, 30, tzinfo=timezone.utc)


class FakeBroker(Broker):
    def __init__(self, price: float = 100.0):
        self.price = price
        self.orders: list[tuple] = []

    def get_current_price(self, symbol, key, secret, feed="iex") -> float:
        return self.price

    def submit_order(self, symbol, side, quantity, price) -> dict:
        self.orders.append((symbol, side, quantity, price))
        return {"status": "filled", "filled_qty": quantity, "filled_price": price}


@pytest.fixture
def market_open():
    clock.set_simulated(MIDSESSION)
    yield
    clock.clear()


@pytest.fixture
def state() -> AppState:
    state = AppState()
    state.set_symbols([TICKER])
    state.api_key = "k"
    state.api_secret = "s"
    state.feed = "iex"
    return state


class TestCycleTiming:
    """The bar-aligned cadence, now shared by every rule agent."""

    def test_wakes_just_after_the_next_bar_closes(self):
        clock.set_simulated(datetime(2026, 7, 21, 14, 30, 20, tzinfo=timezone.utc))
        try:
            # 40s to the boundary, plus the lag that lets the bar arrive.
            assert rule_agent.seconds_to_next_bar(60, lag=5.0) == pytest.approx(45.0)
        finally:
            clock.clear()

    def test_the_lag_is_added_on_top_of_the_boundary(self):
        """Just past a boundary it waits for the NEXT one, never skipping a bar
        by landing before the lag."""
        clock.set_simulated(datetime(2026, 7, 21, 14, 30, 3, tzinfo=timezone.utc))
        try:
            assert rule_agent.seconds_to_next_bar(60, lag=5.0) == pytest.approx(62.0)
        finally:
            clock.clear()


# --------------------------------------------------------------------------
# The day-range rules (TimeToChange3): one forecast, two resting levels.
#
# Driven through a stubbed forecast: these pin the RULES -- when a level is a
# buy, when it is a sell, and what the opening window and the closing bell
# override -- without depending on the saved bundle. `tests/test_dayrange_model.py` pins the
# forecast itself.
# --------------------------------------------------------------------------

DAYRANGE_BUNDLE = {"opening_minutes": 5}

# The forecast the stub returns: a $10 average daily range around a predicted
# high of $110, so at the notebook's 0.75 / 0.10 -- which `dayrange_config`
# pins, whatever an instrument's own default is -- the levels land on round
# numbers.
FORECAST = {
    "pred_high": 110.0,
    "pred_low": 95.0,
    "prev_avg": 102.0,
    "adr14_abs": 10.0,
    "or_high": 103.0,
    "or_low": 101.0,
}
BUY_LEVEL = 102.5   # 110 - 0.75 x 10
SELL_LEVEL = 109.0  # 110 - 0.10 x 10


class Tape:
    """A growing frame of today's minute bars, as `minute_frame` returns it.

    The first five bars are the 09:30 opening window the forecast is built on;
    everything after them is a tradable bar the test appends one at a time.
    """

    OPEN = pd.Timestamp("2026-07-21 09:30", tz="America/New_York")

    def __init__(self, monkeypatch, broker=None, minutes: int = 5):
        self.broker = broker
        self.rows: list[dict] = []
        self.index: list[pd.Timestamp] = []
        self.forecast_calls = 0
        for i in range(minutes):
            self.append(101.0 + i * 0.1, low=100.9, high=101.5, offset=i)

        dayrange = at._dayrange()
        monkeypatch.setattr(
            at.momentum_regime, "minute_frame", lambda *a, **k: self.frame()
        )
        monkeypatch.setattr(at.historical, "fetch_daily_ohlc_bars", lambda *a, **k: [])
        monkeypatch.setattr(at.historical, "fetch_session_open", lambda *a, **k: None)
        monkeypatch.setattr(dayrange, "forecast_session", self._forecast)

    def _forecast(self, *args, **kwargs):
        self.forecast_calls += 1
        return dict(FORECAST)

    def append(self, close: float, low=None, high=None, offset=None):
        """One more closed bar. `offset` is minutes from the open; without it
        the bar lands at 10:30, comfortably past the opening window."""
        if offset is None:
            offset = 60 + len(self.rows)
        self.index.append(self.OPEN + pd.Timedelta(minutes=offset))
        self.rows.append(
            {
                "open": close,
                "high": close if high is None else high,
                "low": close if low is None else low,
                "close": close,
                "volume": 1.0e5,
                "minutes_from_open": float(offset),
            }
        )
        if self.broker is not None:
            self.broker.price = close

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, index=pd.DatetimeIndex(self.index))


def dayrange_config(**kwargs) -> AppleTraderConfig:
    kwargs.setdefault("buy_k", 0.75)
    kwargs.setdefault("sell_k", 0.10)
    return AppleTraderConfig(model_key="dayrange", **kwargs)


class TestDayRangeEntry:
    def _trader(self, **kwargs):
        return at.DayRangeTrader(dayrange_config(**kwargs))

    def test_a_bar_that_trades_down_to_the_buy_level_is_bought(
        self, state, market_open, monkeypatch
    ):
        broker = FakeBroker(103.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tape = Tape(monkeypatch, broker)
        trader = self._trader()

        tape.append(103.0, low=BUY_LEVEL - 0.01)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "bought"
        assert tracker.position_for(TICKER) > 0
        reasoning = tracker.snapshot()["decisions"][-1].reasoning
        assert "102.50" in reasoning and "110.00" in reasoning

    def test_a_bar_that_stays_above_the_buy_level_is_not(
        self, state, market_open, monkeypatch
    ):
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(104.0))
        tape = Tape(monkeypatch)
        trader = self._trader()

        tape.append(104.0, low=BUY_LEVEL + 0.01)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "hold"
        assert tracker.position_for(TICKER) == 0

    def test_the_levels_move_with_the_configured_distances(
        self, state, market_open, monkeypatch
    ):
        """The two knobs are the whole strategy: a shallower buy distance turns
        the same bar from a hold into a fill."""
        broker = FakeBroker(105.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tape = Tape(monkeypatch, broker)
        trader = self._trader(buy_k=0.4)  # buy level 106.0 rather than 102.5

        tape.append(105.0, low=105.0)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "bought"
        assert trader.plan["buy_level"] == pytest.approx(106.0)
        assert trader.plan["sell_level"] == pytest.approx(SELL_LEVEL)

    def test_nothing_trades_before_the_opening_window_closes(
        self, state, market_open, monkeypatch
    ):
        """The forecast does not exist before 9:35, so neither does the rule --
        even on a bar that is below where the buy level will turn out to be."""
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(100.0))
        tape = Tape(monkeypatch, minutes=3)
        trader = self._trader()

        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "warming_up"
        assert trader.plan is None
        assert tape.forecast_calls == 0
        assert tracker.position_for(TICKER) == 0

    def test_the_last_bar_of_the_opening_window_is_not_traded(
        self, state, market_open, monkeypatch
    ):
        """The forecast is built *from* that bar, so acting on it would be
        trading the same minute the model was just handed. The notebook skips
        it too (`start_after`)."""
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(100.0))
        tape = Tape(monkeypatch, minutes=5)
        tape.rows[-1]["low"] = BUY_LEVEL - 5  # deep enough to fill, if it counted
        trader = self._trader()

        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "warming_up"
        assert trader.plan is not None  # the forecast IS made on that bar
        assert tracker.position_for(TICKER) == 0

    def test_the_forecast_is_made_once_and_reused_all_day(
        self, state, market_open, monkeypatch
    ):
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(104.0))
        tape = Tape(monkeypatch)
        trader = self._trader()

        for _ in range(6):
            tape.append(104.0)
            trader.run_cycle(DAYRANGE_BUNDLE, state, tracker)
        assert tape.forecast_calls == 1

    def test_a_replayed_bar_does_not_buy_twice(self, state, market_open, monkeypatch):
        """A cycle that runs before a new bar closes sees the same one again."""
        broker = FakeBroker(103.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tape = Tape(monkeypatch, broker)
        trader = self._trader()

        tape.append(103.0, low=BUY_LEVEL - 0.01)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "bought"
        held = tracker.position_for(TICKER)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "hold"
        assert tracker.position_for(TICKER) == held


class TestDayRangeExit:
    def _entered(self, state, tracker, tape, **kwargs):
        trader = at.DayRangeTrader(dayrange_config(**kwargs))
        tape.append(103.0, low=BUY_LEVEL - 0.01)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "bought"
        return trader

    def test_a_bar_that_trades_up_to_the_sell_level_closes_the_position(
        self, state, market_open, monkeypatch
    ):
        broker = FakeBroker(103.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tape = Tape(monkeypatch, broker)
        trader = self._entered(state, tracker, tape)

        tape.append(108.8, high=SELL_LEVEL + 0.05)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "sold"
        assert tracker.position_for(TICKER) == 0
        assert "Target" in tracker.snapshot()["decisions"][-1].reasoning

    def test_a_position_that_never_reaches_the_target_is_simply_held(
        self, state, market_open, monkeypatch
    ):
        """No stop, by design: the forecast says where the day tops out, and
        bailing on weakness would be a second, unmeasured rule."""
        broker = FakeBroker(103.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tape = Tape(monkeypatch, broker)
        trader = self._entered(state, tracker, tape)

        for price in (101.0, 99.0, 96.0, 94.0):
            tape.append(price)
            assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "hold"
        assert tracker.position_for(TICKER) > 0

    def test_the_closing_bell_flattens_what_the_day_never_paid_out(
        self, state, market_open, monkeypatch
    ):
        broker = FakeBroker(103.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tape = Tape(monkeypatch, broker)
        trader = self._entered(state, tracker, tape)

        clock.set_simulated(datetime(2026, 7, 21, 19, 57, tzinfo=timezone.utc))  # 15:57 ET
        tape.append(104.0)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "sold"
        assert tracker.position_for(TICKER) == 0
        assert "flattened" in tracker.snapshot()["decisions"][-1].reasoning

    def test_the_rule_re_arms_after_a_sale(self, state, market_open, monkeypatch):
        """The levels are resting orders, not a one-shot: a day that dips,
        recovers and dips again is traded twice."""
        broker = FakeBroker(103.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tape = Tape(monkeypatch, broker)
        trader = self._entered(state, tracker, tape)

        tape.append(108.8, high=SELL_LEVEL + 0.05)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "sold"
        tape.append(103.0, low=BUY_LEVEL - 0.01)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "bought"
        fills = [d for d in tracker.snapshot()["decisions"] if d.status == "filled"]
        assert [d.action for d in fills] == ["buy", "sell", "buy"]


class TestDayRangeGuards:
    def test_no_entry_inside_the_closing_flatten_window(
        self, state, market_open, monkeypatch
    ):
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(103.0))
        tape = Tape(monkeypatch)
        trader = at.DayRangeTrader(dayrange_config())
        # Warm the plan up while the session still has hours left.
        tape.append(104.0)
        trader.run_cycle(DAYRANGE_BUNDLE, state, tracker)

        clock.set_simulated(datetime(2026, 7, 21, 19, 57, tzinfo=timezone.utc))
        tape.append(103.0, low=BUY_LEVEL - 0.01)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "hold"
        assert tracker.position_for(TICKER) == 0

    def test_a_forecast_that_cannot_be_made_stops_the_day_rather_than_the_bar(
        self, state, market_open, monkeypatch
    ):
        """Too little daily history at 9:35 is still too little at 14:00, so
        the refusal is logged once and the session is skipped -- not retried
        every minute for six and a half hours."""
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(103.0))
        tape = Tape(monkeypatch)
        calls = {"n": 0}

        def boom(*args, **kwargs):
            calls["n"] += 1
            raise ValueError("only 40 daily sessions of history")

        monkeypatch.setattr(at._dayrange(), "forecast_session", boom)
        trader = at.DayRangeTrader(dayrange_config())

        for _ in range(4):
            tape.append(103.0, low=BUY_LEVEL - 0.01)
            assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "no_data"
        assert calls["n"] == 1
        assert tracker.position_for(TICKER) == 0
        errors = [e for e in state.agent_log if e.get("type") == "error"]
        assert len(errors) == 1 and "40 daily sessions" in errors[0]["text"]

    def test_an_opening_window_the_buffer_never_saw_is_refused(
        self, state, market_open, monkeypatch
    ):
        """An agent started at 10:30 has a buffer that begins at 10:30. Taking
        its first five bars as "the open" would forecast confidently off the
        wrong five minutes."""
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(103.0))
        tape = Tape(monkeypatch, minutes=0)
        for _ in range(6):
            tape.append(103.0, low=BUY_LEVEL - 0.01)
        monkeypatch.setattr(at.agent_mod, "fetch_bars_window", lambda *a, **k: [])
        trader = at.DayRangeTrader(dayrange_config())

        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "no_data"
        assert tracker.position_for(TICKER) == 0
        assert any(
            "09:30 window" in e.get("text", "")
            for e in state.agent_log
            if e.get("type") == "error"
        )

    def test_a_new_session_forgets_yesterdays_levels(
        self, state, market_open, monkeypatch
    ):
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(104.0))
        tape = Tape(monkeypatch)
        trader = at.DayRangeTrader(dayrange_config())
        tape.append(104.0)
        trader.run_cycle(DAYRANGE_BUNDLE, state, tracker)
        assert tape.forecast_calls == 1

        clock.set_simulated(datetime(2026, 7, 22, 14, 30, tzinfo=timezone.utc))
        tape.append(104.0)
        trader.run_cycle(DAYRANGE_BUNDLE, state, tracker)
        assert tape.forecast_calls == 2


class TestStrategySelection:
    def test_the_model_chooses_the_state_machine(self):
        assert isinstance(
            at.build_trader(dayrange_config(), DAYRANGE_BUNDLE), at.DayRangeTrader
        )

    def test_the_levels_are_the_signature(self):
        base = config_signature(dayrange_config())
        assert base == "dayrange_AAPL(buy=H-0.75A,sell=H-0.1A,size=95%)"
        assert base != config_signature(dayrange_config(buy_k=0.8))
        assert base != config_signature(dayrange_config(sell_k=0.2))

    def test_a_sell_level_below_the_buy_level_is_refused(self):
        """Both are distances *below* the predicted high, so the sell distance
        has to be the smaller number. The other way round the rule would sell
        under its own entry on every bar."""
        with pytest.raises(ValueError, match="sell_k"):
            AppleTraderConfig(buy_k=0.5, sell_k=0.5)
        with pytest.raises(ValueError, match="sell_k"):
            AppleTraderConfig(buy_k=0.2, sell_k=0.6)


# ----------------------------------------------------------------- instrument
#
# Which symbol a run trades, and the one thing that constrains it: a model was
# fitted on a symbol or it was not. `UNMODELLED` is the case where nothing is.

NON_AAPL = "GOOGL"
UNMODELLED = "MSFT"


class TestDayRangeLevelDefaults:
    """Each instrument starts from its own swept pair, not the notebook's."""

    def test_each_instrument_starts_from_its_own_pair(self):
        for ticker, (buy_k, sell_k) in at.APPLE_TRADER_DAYRANGE_LEVELS.items():
            config = AppleTraderConfig(model_key="dayrange", ticker=ticker.lower())
            assert (config.buy_k, config.sell_k) == (buy_k, sell_k)

    def test_every_ticker_the_model_is_wired_up_for_has_a_pair(self):
        assert set(apple_models.DAYRANGE_TICKERS) <= set(at.APPLE_TRADER_DAYRANGE_LEVELS)

    def test_a_symbol_never_swept_falls_back_to_the_notebook_pair(self):
        assert at.dayrange_levels("ZZZZ") == (0.75, 0.10)
        config = AppleTraderConfig(ticker="ZZZZ")
        assert (config.buy_k, config.sell_k) == (0.75, 0.10)

    def test_a_level_given_explicitly_wins_and_the_other_keeps_its_default(self):
        _, googl_sell = at.dayrange_levels("GOOGL")
        config = AppleTraderConfig(model_key="dayrange", ticker="GOOGL", buy_k=0.9)
        assert (config.buy_k, config.sell_k) == (0.9, googl_sell)

    def test_the_default_pair_signs_the_run(self):
        buy_k, sell_k = at.dayrange_levels("AAPL")
        assert config_signature(AppleTraderConfig(model_key="dayrange")) == (
            f"dayrange_AAPL(buy=H-{buy_k:g}A,sell=H-{sell_k:g}A,size=95%)"
        )


class TestInstrument:
    def test_the_symbols_on_offer_are_the_ones_a_model_covers(self):
        for symbol in (TICKER, NON_AAPL, "INTC"):
            assert apple_models.keys_for(symbol) == ["dayrange"]
        assert apple_models.keys_for(UNMODELLED) == []

    def test_a_model_cannot_be_pointed_at_a_symbol_it_was_not_fitted_on(
        self, monkeypatch
    ):
        """The check that keeps 'Apple Trader on GOOGL' from meaning a model
        fitted on a different stock's tape.

        Stubbed back to AAPL-only, because the shipped model covers every
        shipped symbol and the machinery would otherwise go untested."""
        monkeypatch.setitem(
            apple_models.MODELS, "dayrange",
            replace(apple_models.MODELS["dayrange"], tickers=(TICKER,)),
        )
        error = at.model_ticker_error(
            AppleTraderConfig(model_key="dayrange", ticker=NON_AAPL)
        )
        assert error is not None
        assert "AAPL only" in error and NON_AAPL in error
        assert ".joblib" not in error

    def test_an_unmodelled_symbol_is_refused_with_no_alternative_offered(self):
        error = at.model_ticker_error(
            AppleTraderConfig(model_key="dayrange", ticker=UNMODELLED)
        )
        assert error is not None and "pick another instrument" in error

    def test_a_removed_model_is_refused_rather_than_replaced(self):
        """A stored config naming a model that has been taken out of the app
        must not run as the model that is left -- that would file one strategy's
        numbers under another's name."""
        for key in ("persistence", "nbeats", "momentum_change"):
            error = at.model_ticker_error(AppleTraderConfig(model_key=key))
            assert error is not None and key in error and "removed" in error

    def test_the_loop_refuses_a_removed_model_before_it_loads_anything(
        self, state, monkeypatch
    ):
        loaded: list = []
        monkeypatch.setattr(
            at.apple_models, "load",
            lambda key, ticker=None: loaded.append(key) or DAYRANGE_BUNDLE,
        )
        at._apple_trader_loop(
            state, tracker_for_loop(), AppleTraderConfig(model_key="nbeats"), 60,
            threading.Event(),
        )
        assert loaded == []
        assert state.agent_running is False
        assert any("removed" in e.get("text", "") for e in state.agent_log)

    def test_the_pairing_is_part_of_config_error(self):
        """One call is what every launch path checks, so the pairing cannot be
        enforced in the live loop and forgotten in SimLab."""
        config = AppleTraderConfig(model_key="dayrange", ticker=UNMODELLED)
        assert "cannot trade" in (at.config_error(config, DAYRANGE_BUNDLE) or "")

    def test_a_ticker_is_normalised(self):
        assert AppleTraderConfig(ticker=" googl ").ticker == "GOOGL"

    def test_the_signature_carries_the_symbol(self):
        """The same levels over two tapes are two experiments; filing them
        together would average them into one row in Results."""
        aapl = config_signature(AppleTraderConfig(model_key="dayrange"))
        googl = config_signature(AppleTraderConfig(model_key="dayrange", ticker=NON_AAPL))
        assert aapl.startswith("dayrange_AAPL(") and googl.startswith("dayrange_GOOGL(")
        assert aapl != googl

    def test_a_config_without_a_symbol_is_an_aapl_run(self):
        assert AppleTraderConfig(model_key="dayrange").ticker == TICKER

    def test_the_loop_refuses_the_pairing_before_it_loads_anything(
        self, state, monkeypatch
    ):
        """'There is no MSFT day-range model' rather than 'the file is
        missing': different problems, different fixes."""
        loaded: list = []
        monkeypatch.setattr(
            at.apple_models, "load",
            lambda key, ticker=None: loaded.append((key, ticker)) or DAYRANGE_BUNDLE,
        )
        at._apple_trader_loop(
            state, tracker_for_loop(), AppleTraderConfig(
                model_key="dayrange", ticker=UNMODELLED
            ), 60, threading.Event(),
        )
        assert loaded == []
        assert any("cannot trade" in e.get("text", "") for e in state.agent_log)
        assert state.agent_running is False

    def test_the_loop_loads_the_bundle_for_the_configured_symbol(
        self, state, monkeypatch
    ):
        asked: list = []
        monkeypatch.setattr(
            at.apple_models, "load",
            lambda key, ticker=None: asked.append((key, ticker)) or None,
        )
        at._apple_trader_loop(
            state, tracker_for_loop(), AppleTraderConfig(
                model_key="dayrange", ticker=NON_AAPL
            ), 60, threading.Event(),
        )
        assert asked == [("dayrange", NON_AAPL)]


def tracker_for_loop() -> DecisionTracker:
    return DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(100.0))
