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

from agent_stonks import apple_models, intraday_vol_model
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

    The momentum score is scripted per bar (`append(..., mom=)`, NaN when not
    given, as it is while the score warms up), so the exit rules can be pinned
    without engineering a price path that produces a particular score. Pass
    `real_momentum=True` to have the trader compute it from the closes instead.
    """

    OPEN = pd.Timestamp("2026-07-21 09:30", tz="America/New_York")

    def __init__(
        self, monkeypatch, broker=None, minutes: int = 5, real_momentum: bool = False
    ):
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
        if not real_momentum:
            monkeypatch.setattr(
                at.momentum_regime, "compute_momentum", lambda frame, *a, **k: frame
            )
        monkeypatch.setattr(at.historical, "fetch_daily_ohlc_bars", lambda *a, **k: [])
        monkeypatch.setattr(at.historical, "fetch_session_open", lambda *a, **k: None)
        monkeypatch.setattr(dayrange, "forecast_session", self._forecast)

    def _forecast(self, *args, **kwargs):
        self.forecast_calls += 1
        return dict(FORECAST)

    def append(self, close: float, low=None, high=None, offset=None, mom=None):
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
                "mom": float("nan") if mom is None else float(mom),
            }
        )
        if self.broker is not None:
            self.broker.price = close

    def frame(self) -> pd.DataFrame:
        # The same session bookkeeping `minute_frame` attaches, which is what
        # the momentum score groups on.
        return at.momentum_regime.add_session_columns(
            pd.DataFrame(self.rows, index=pd.DatetimeIndex(self.index))
        )


def dayrange_config(**kwargs) -> AppleTraderConfig:
    """The notebook's configuration, whatever the app's defaults happen to be.

    The levels are pinned to 0.75 / 0.10 rather than to AAPL's swept pair for
    the same reason `breach_update` is pinned off: these tests are about the
    rule as notebook 05 specified it, and a default that moves would rewrite
    what they assert. `TestIntradayRangeUpdate` is where the update is on.
    """
    kwargs.setdefault("buy_k", 0.75)
    kwargs.setdefault("sell_k", 0.10)
    kwargs.setdefault("breach_update", "off")
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

    def test_a_trade_that_goes_the_wrong_way_is_stopped_out(
        self, state, market_open, monkeypatch
    ):
        """Filled at 103 on a $10 ADR, the default 0.2 stop sits at 101.00 --
        a touch, like the levels, so the low decides."""
        broker = FakeBroker(103.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tape = Tape(monkeypatch, broker)
        trader = self._entered(state, tracker, tape)

        tape.append(101.5, low=101.01)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "hold"
        tape.append(101.2, low=100.99)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "sold"
        assert tracker.position_for(TICKER) == 0
        reasoning = tracker.snapshot()["decisions"][-1].reasoning
        assert "Stop loss" in reasoning and "101.00" in reasoning

    def test_the_stop_is_a_distance_in_adrs_from_the_fill(
        self, state, market_open, monkeypatch
    ):
        broker = FakeBroker(103.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tape = Tape(monkeypatch, broker)
        trader = self._entered(state, tracker, tape, stop_k=0.5)  # 103 - 5 = 98

        tape.append(99.0, low=98.01)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "hold"
        tape.append(98.0, low=97.9)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "sold"
        assert "98.00" in tracker.snapshot()["decisions"][-1].reasoning

    def test_after_a_stop_the_session_buys_nothing_more(
        self, state, market_open, monkeypatch
    ):
        """The buy level is right above a stopped-out price, so re-arming it
        would buy the same slide again, one stop lower each time."""
        broker = FakeBroker(103.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tape = Tape(monkeypatch, broker)
        trader = self._entered(state, tracker, tape)

        tape.append(100.5)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "sold"
        for _ in range(3):
            tape.append(100.0, low=BUY_LEVEL - 3)
            assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "hold"
        assert tracker.position_for(TICKER) == 0

    def test_with_the_exit_switched_off_a_losing_position_is_simply_held(
        self, state, market_open, monkeypatch
    ):
        """Notebook 05's rule as specified, and what a record written before
        the managed exit existed replays as: no stop and no momentum take,
        however far the price and the score fall."""
        broker = FakeBroker(103.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tape = Tape(monkeypatch, broker)
        trader = self._entered(state, tracker, tape, stop_k=0.0, momentum_drop=0.0)

        for price, mom in ((104.0, 2.0), (104.5, 0.5), (99.0, -1.0), (94.0, -2.5)):
            tape.append(price, mom=mom)
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


class TestDayRangeMomentumTake:
    """Banking gains short of the target when the move carrying them fades.

    Every test fills at 103 on the $10 ADR, so the 109 sell level is 0.6 ADR
    above the fill -- past the 0.3 worth keeping a runner for -- and the stop
    sits at 101. The momentum score is scripted per bar, except in the last
    test, which computes it from the closes.
    """

    def _entered(self, state, monkeypatch, **kwargs):
        broker = FakeBroker(103.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tape = Tape(monkeypatch, broker, real_momentum=kwargs.pop("real_momentum", False))
        trader = at.DayRangeTrader(dayrange_config(**kwargs))
        tape.append(103.0, low=BUY_LEVEL - 0.01, mom=0.0)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "bought"
        return trader, tracker, tape

    def _taken(self, state, monkeypatch, **kwargs):
        trader, tracker, tape = self._entered(state, monkeypatch, **kwargs)
        held = tracker.position_for(TICKER)
        tape.append(105.0, mom=2.0)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "hold"
        tape.append(105.2, mom=0.9)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "sold"
        return trader, tracker, tape, held

    def test_a_fading_move_in_profit_banks_most_and_keeps_a_runner(
        self, state, market_open, monkeypatch
    ):
        trader, tracker, tape = self._entered(state, monkeypatch)
        held = tracker.position_for(TICKER)

        tape.append(105.0, mom=2.0)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "hold"
        tape.append(105.5, mom=1.2)  # 0.8σ off the peak: a wobble, not a fade
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "hold"
        tape.append(105.2, mom=0.9)  # 1.1σ
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "sold"

        runner = tracker.position_for(TICKER)
        assert held - runner == int(held * 0.7)
        assert 0 < runner < held
        reasoning = tracker.snapshot()["decisions"][-1].reasoning
        assert "Momentum take" in reasoning and "Banking" in reasoning

    def test_the_share_taken_is_configurable(self, state, market_open, monkeypatch):
        _, tracker, _, held = self._taken(state, monkeypatch, take_fraction=0.5)
        assert tracker.position_for(TICKER) == held - int(held * 0.5)

    def test_the_runner_is_not_trimmed_again_and_waits_for_the_sell_level(
        self, state, market_open, monkeypatch
    ):
        trader, tracker, tape, _ = self._taken(state, monkeypatch)
        runner = tracker.position_for(TICKER)

        tape.append(104.0, mom=-2.5)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "hold"
        assert tracker.position_for(TICKER) == runner
        tape.append(108.8, high=SELL_LEVEL + 0.05)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "sold"
        assert tracker.position_for(TICKER) == 0
        assert "Target" in tracker.snapshot()["decisions"][-1].reasoning

    def test_the_runner_is_sold_if_the_price_comes_back_to_the_fill(
        self, state, market_open, monkeypatch
    ):
        trader, tracker, tape, _ = self._taken(state, monkeypatch)

        tape.append(103.5, low=103.01)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "hold"
        tape.append(103.2, low=102.99)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "sold"
        assert tracker.position_for(TICKER) == 0
        assert "Breakeven" in tracker.snapshot()["decisions"][-1].reasoning

        # A breakeven is not a stop: the forecast was not proven wrong, so the
        # buy level re-arms as it does after the target.
        tape.append(102.6, low=BUY_LEVEL - 0.01)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "bought"

    def test_a_target_too_close_to_wait_for_is_sold_whole(
        self, state, market_open, monkeypatch
    ):
        """0.6 ADR left to the sell level, under a 0.7 threshold."""
        trader, tracker, tape = self._entered(state, monkeypatch, hold_min_gain_k=0.7)

        tape.append(105.0, mom=2.0)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "hold"
        tape.append(104.5, mom=0.5)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "sold"
        assert tracker.position_for(TICKER) == 0
        assert "whole position" in tracker.snapshot()["decisions"][-1].reasoning

    def test_there_is_no_gain_to_take_under_water(self, state, market_open, monkeypatch):
        trader, tracker, tape = self._entered(state, monkeypatch)
        held = tracker.position_for(TICKER)

        tape.append(104.0, mom=2.0)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "hold"
        tape.append(102.9, mom=-1.0)  # below the fill, above the stop
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "hold"
        assert tracker.position_for(TICKER) == held

    def test_the_peak_is_this_positions_not_the_mornings(
        self, state, market_open, monkeypatch
    ):
        """A surge before the entry is not a move this trade was riding."""
        trader, tracker, tape = self._entered(state, monkeypatch)
        held = tracker.position_for(TICKER)
        for row in tape.rows[:-1]:
            row["mom"] = 3.0

        tape.append(104.0, mom=1.5)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "hold"
        assert tracker.position_for(TICKER) == held

    def test_the_score_is_computed_from_the_tape(self, state, market_open, monkeypatch):
        """End to end on the real momentum score: a zig-zag climb that never
        reaches the target, then a turn. The climb alone moves the smoothed
        score about 0.8σ off its first peak, under the 1σ trigger; the turn
        crosses it at 105.30, and the runner goes back out at the fill."""
        trader, tracker, tape = self._entered(state, monkeypatch, real_momentum=True)
        price = 103.0
        for step in [0.3, -0.1] * 15 + [-0.3, 0.1] * 15:
            price += step
            tape.append(round(price, 2))
            trader.run_cycle(DAYRANGE_BUNDLE, state, tracker)

        fills = [d for d in tracker.snapshot()["decisions"] if d.status == "filled"]
        assert [d.action for d in fills] == ["buy", "sell", "sell"]
        buy, take, breakeven = fills
        assert "Momentum take" in take.reasoning and take.price == pytest.approx(105.3)
        assert take.filled_quantity == int(buy.filled_quantity * 0.7)
        assert "Breakeven" in breakeven.reasoning
        assert tracker.position_for(TICKER) == 0


class TestIntradayRangeUpdate:
    """What a session that trades outside the forecast does to the two levels.

    The forecast is a statement about the width of the day and the levels hang
    off it, so a day that has traded through it has falsified both. These pin
    which of the three policies moves what, and when.
    """

    def _trader(self, policy, **kwargs):
        return at.DayRangeTrader(dayrange_config(breach_update=policy, **kwargs))

    def _run(self, policy, state, tracker, tape, **kwargs):
        trader = self._trader(policy, **kwargs)
        trader.run_cycle(DAYRANGE_BUNDLE, state, tracker)
        return trader

    def test_off_holds_the_935_forecast_through_a_breach(
        self, state, market_open, monkeypatch
    ):
        """The notebook's rule: one forecast, two levels, all day."""
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(112.0))
        tape = Tape(monkeypatch)
        tape.append(112.0, high=112.0)
        trader = self._run("off", state, tracker, tape)

        assert trader.plan["pred_high"] == FORECAST["pred_high"]
        assert trader.plan["buy_level"] == pytest.approx(BUY_LEVEL)
        assert trader.plan["sell_level"] == pytest.approx(SELL_LEVEL)
        assert "range_updates" not in trader.plan

    def test_extreme_moves_the_high_to_the_session_high_and_the_levels_with_it(
        self, state, market_open, monkeypatch
    ):
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(112.0))
        tape = Tape(monkeypatch)
        tape.append(111.5, high=112.0)
        trader = self._run("extreme", state, tracker, tape)

        assert trader.plan["pred_high"] == pytest.approx(112.0)
        assert trader.plan["pred_low"] == FORECAST["pred_low"]  # never breached
        # Both levels are rebuilt from the moved high: the whole point of the
        # update is that the strategy's numbers follow the forecast's.
        assert trader.plan["buy_level"] == pytest.approx(112.0 - 0.75 * 10)
        assert trader.plan["sell_level"] == pytest.approx(112.0 - 0.10 * 10)
        assert trader.plan["range_updates"] == 1

    def test_a_breach_of_the_low_moves_the_low(self, state, market_open, monkeypatch):
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(94.0))
        tape = Tape(monkeypatch)
        tape.append(94.5, low=94.0)
        trader = self._run("extreme", state, tracker, tape)

        assert trader.plan["pred_low"] == pytest.approx(94.0)
        assert trader.plan["pred_high"] == FORECAST["pred_high"]

    def test_brownian_extends_past_the_extreme(self, state, market_open, monkeypatch):
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(112.0))
        tape = Tape(monkeypatch)
        tape.append(111.5, high=112.0)   # the 10:35 bar: 325 minutes to the close
        trader = self._run("brownian", state, tracker, tape)

        reach = at._dayrange().brownian_reach(10.0, 325.0)
        assert reach > 0
        assert trader.plan["pred_high"] == pytest.approx(112.0 + reach)
        assert trader.plan["buy_level"] == pytest.approx(112.0 + reach - 7.5)

    def test_the_updated_levels_are_what_the_bar_is_then_measured_against(
        self, state, market_open, monkeypatch
    ):
        """The update runs before the rules, not after, so the entry it arms is
        the one the new high implies. A bar that dipped nowhere near the 9:35 buy
        level fills against the one its own breach just raised."""
        broker = FakeBroker(113.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tape = Tape(monkeypatch, broker)
        # Low 113 is $10.50 above the 9:35 buy level and would never have filled;
        # against a high moved to 121 the buy level is 113.50.
        tape.append(113.0, low=113.0, high=121.0)

        assert self._run("off", state, tracker, tape).plan["buy_level"] == pytest.approx(
            BUY_LEVEL
        )
        assert tracker.position_for(TICKER) == 0

        trader = self._trader("extreme")
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "bought"
        assert trader.plan["buy_level"] == pytest.approx(113.5)

    def test_extreme_still_sells_into_the_breach_but_brownian_holds_through_it(
        self, state, market_open, monkeypatch
    ):
        """The two policies differ where it costs money, and this is where.

        A bar that breaches the high also reaches a sell level moved only to
        that same high (the gap is `sell_k × ADR`, and the bar is *at* the
        extreme), so `extreme` banks the trade exactly as `off` does. The
        Brownian reach is bigger than that gap for most of the day, so the
        target moves out of the bar's way and the position rides on.
        """
        def entered(policy):
            broker = FakeBroker(103.0)
            tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
            tape = Tape(monkeypatch, broker)
            trader = self._trader(policy)
            tape.append(103.0, low=BUY_LEVEL - 0.01)
            assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "bought"
            tape.append(111.0, high=112.0)
            return trader, trader.run_cycle(DAYRANGE_BUNDLE, state, tracker), tracker

        for policy in ("off", "extreme"):
            _, outcome, tracker = entered(policy)
            assert outcome == "sold", policy
            assert tracker.position_for(TICKER) == 0

        trader, outcome, tracker = entered("brownian")
        assert outcome == "hold"
        assert tracker.position_for(TICKER) > 0
        assert trader.plan["sell_level"] > 112.0

    def test_the_high_only_ever_ratchets_up(self, state, market_open, monkeypatch):
        """A level that could walk back towards the price would turn one move
        into a stream of entries and exits chasing it."""
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(112.0))
        tape = Tape(monkeypatch)
        trader = self._trader("brownian")
        seen = []
        for close, high in ((111.5, 112.0), (108.0, 109.0), (105.0, 106.0)):
            tape.append(close, high=high)
            trader.run_cycle(DAYRANGE_BUNDLE, state, tracker)
            seen.append(trader.plan["pred_high"])
        assert seen == sorted(seen)
        assert trader.plan["range_updates"] == 1  # only the first bar breached

    def test_the_opening_window_never_triggers_an_update(
        self, state, market_open, monkeypatch
    ):
        """Its extremes are already in the forecast (`apply_open_constraint`),
        and the bar the plan was built on is not traded either."""
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(120.0))
        tape = Tape(monkeypatch, minutes=0)
        for i in range(5):
            tape.append(120.0, high=121.0, offset=i)
        trader = self._trader("extreme")

        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "warming_up"
        assert trader.plan["pred_high"] == FORECAST["pred_high"]

    def test_an_update_is_logged_once_with_both_the_old_and_new_levels(
        self, state, market_open, monkeypatch
    ):
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(112.0))
        tape = Tape(monkeypatch)
        tape.append(111.5, high=112.0)
        self._run("extreme", state, tracker, tape)

        lines = [e["text"] for e in state.agent_log if "forecast updated" in e.get("text", "")]
        assert len(lines) == 1
        assert "110.00" in lines[0] and "112.00" in lines[0]        # the high, before and after
        assert "102.50" in lines[0] and "104.50" in lines[0]        # the buy level, ditto

    def test_a_breach_of_the_low_alone_does_not_claim_the_levels_moved(
        self, state, market_open, monkeypatch
    ):
        """They are built from the predicted high, so a low that moves moves
        nothing this strategy rests on."""
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(94.0))
        tape = Tape(monkeypatch)
        tape.append(94.5, low=94.0)
        trader = self._run("extreme", state, tracker, tape)

        line = next(e["text"] for e in state.agent_log if "forecast updated" in e.get("text", ""))
        assert "they stay" in line and "rebuilt" not in line
        assert trader.plan["buy_level"] == pytest.approx(BUY_LEVEL)

    def test_a_new_session_starts_the_update_over(
        self, state, market_open, monkeypatch
    ):
        """The update is a fact about one session, like the forecast it corrects.

        The tape fixture keeps every bar under one index, so yesterday's breach
        is still visible to today's frame here -- what this pins is that the
        plan itself was rebuilt from a second forecast and the count restarted,
        not carried over. Live, `minute_frame` hands the trader today's bars only.
        """
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(112.0))
        tape = Tape(monkeypatch)
        tape.append(111.5, high=112.0)
        trader = self._run("extreme", state, tracker, tape)
        assert trader.plan["range_updates"] == 1 and tape.forecast_calls == 1

        clock.set_simulated(datetime(2026, 7, 22, 14, 30, tzinfo=timezone.utc))
        tape.append(104.0)
        trader.run_cycle(DAYRANGE_BUNDLE, state, tracker)
        assert tape.forecast_calls == 2
        assert trader.plan["date"] == pd.Timestamp("2026-07-22")
        assert trader.plan["range_updates"] == 1


# A hand-made IntradayVolatility shape, so these pin the levels rather than the
# fitted curve: vol(t) = 0.2 + 0.8 / (1.5 + t), which peaks at the open like the
# real one and settles onto a floor, but is readable by hand. The real fit is
# pinned in `tests/test_intraday_vol_model.py` against the exporter's own check
# block; what matters here is what the trader does with whatever curve it gets.
VOL_SHAPE = {
    "shape": {
        "params": {"a": 0.2, "b": 0.8, "alpha": 1.0, "c": 0.0, "kappa": 30.0},
        "t_domain": [0.0, 389.5],
    },
    "day_range": {"coef": {"const": 0.0, "lr_d": 0.0, "lr_w": 0.0, "lr_m": 0.0,
                           "abs_gap": 0.0}},
}
# The tape's 09:30 bar opens here, and `fetch_session_open` is stubbed to None,
# so this is the price the envelope is centred on.
SESSION_OPEN = 101.0


def expected_reference(minute: float, high=FORECAST["pred_high"], low=FORECAST["pred_low"]):
    """The upper envelope at one minute, straight from the model the chart uses."""
    upper, _ = intraday_vol_model.envelope_at(VOL_SHAPE, SESSION_OPEN, high, low, minute)
    return upper


class TestIntradayLevelSource:
    """The levels resting under the intraday band rather than under a flat high.

    Same forecast, read through IntradayVolatility's time-of-day shape: the
    reference is the upper curve of the "predicted intraday range x day range"
    overlay at the minute of the bar that just closed, so it is the predicted
    high at 09:30 and pulls in towards the open as the day quiets.
    """

    def _trader(self, monkeypatch, **kwargs):
        monkeypatch.setattr(at.intraday_vol_model, "load", lambda *a, **k: VOL_SHAPE)
        return at.DayRangeTrader(dayrange_config(level_source="intraday", **kwargs))

    def test_the_default_is_the_flat_predicted_high(self):
        assert AppleTraderConfig().level_source == "dayrange"

    def test_the_flat_source_rests_on_the_predicted_high_all_day(
        self, state, market_open, monkeypatch
    ):
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(104.0))
        tape = Tape(monkeypatch)
        trader = at.DayRangeTrader(dayrange_config())
        for offset in (65, 200):
            tape.append(104.0, offset=offset)
            trader.run_cycle(DAYRANGE_BUNDLE, state, tracker)
            assert trader.plan["reference"] == FORECAST["pred_high"]
            assert trader.plan["buy_level"] == pytest.approx(BUY_LEVEL)

    def test_the_reference_is_the_overlay_curve_at_this_minute(
        self, state, market_open, monkeypatch
    ):
        """Pinned against `intraday_vol_model` itself, not against a number
        copied here: the level Apple Trader rests on and the band a reader sees
        behind the candles have to be the same curve, or the chart is a lie."""
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(104.0))
        tape = Tape(monkeypatch)
        trader = self._trader(monkeypatch)

        tape.append(104.0, offset=65)
        trader.run_cycle(DAYRANGE_BUNDLE, state, tracker)
        assert trader.plan["reference"] == pytest.approx(expected_reference(65))

    def test_the_levels_are_the_same_two_distances_under_that_reference(
        self, state, market_open, monkeypatch
    ):
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(104.0))
        tape = Tape(monkeypatch)
        trader = self._trader(monkeypatch)

        tape.append(104.0, offset=65)
        trader.run_cycle(DAYRANGE_BUNDLE, state, tracker)
        plan = trader.plan
        assert plan["buy_level"] == pytest.approx(plan["reference"] - 0.75 * 10)
        assert plan["sell_level"] == pytest.approx(plan["reference"] - 0.10 * 10)

    def test_the_reference_pulls_in_from_the_predicted_high_as_the_day_quiets(
        self, state, market_open, monkeypatch
    ):
        """At the open the shape is 1 and the reference *is* the predicted high;
        by the afternoon it is a fraction of the way there."""
        assert expected_reference(0) == pytest.approx(FORECAST["pred_high"])
        seen = [expected_reference(m) for m in (0, 30, 120, 300)]
        assert seen == sorted(seen, reverse=True)
        assert seen[-1] < FORECAST["pred_high"]

    def test_the_levels_move_between_bars_with_no_breach_at_all(
        self, state, market_open, monkeypatch
    ):
        """The clock alone moves them, which is the whole difference from the
        flat source -- and is why the levels are rebuilt every bar."""
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(104.0))
        tape = Tape(monkeypatch)
        trader = self._trader(monkeypatch, breach_update="off")

        levels = []
        for offset in (65, 150, 300):
            tape.append(104.0, offset=offset)
            trader.run_cycle(DAYRANGE_BUNDLE, state, tracker)
            levels.append((trader.plan["buy_level"], trader.plan["sell_level"]))
        assert len(set(levels)) == 3
        assert [b for b, _ in levels] == sorted([b for b, _ in levels], reverse=True)
        assert trader.plan.get("range_updates") is None

    def test_the_sell_level_stays_above_the_buy_level_at_every_minute(
        self, state, market_open, monkeypatch
    ):
        """Both distances come off the same reference, so however it moves the
        pair cannot invert -- the failure that would buy and sell every bar."""
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(104.0))
        tape = Tape(monkeypatch)
        trader = self._trader(monkeypatch)
        for offset in range(65, 389, 20):
            tape.append(104.0, offset=offset)
            trader.run_cycle(DAYRANGE_BUNDLE, state, tracker)
            assert trader.plan["sell_level"] > trader.plan["buy_level"]

    def test_a_descending_target_can_close_a_position_the_price_never_reached(
        self, state, market_open, monkeypatch
    ):
        """The surprising half of this setting, pinned rather than discovered.

        Under the flat high a sell only fires when the price rises to it. Here
        the reference falls through the morning, so the target can come down to
        a flat price instead -- the model saying the day is no longer moving
        enough to reach the morning's number.
        """
        broker = FakeBroker(102.6)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tape = Tape(monkeypatch, broker)
        trader = self._trader(monkeypatch, breach_update="off")

        # 09:36, while the reference is still near the predicted high: a dip
        # deep enough to fill, closing back at 102.60.
        tape.append(102.6, low=97.2, high=102.6, offset=6)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "bought"
        entry_sell = trader.plan["sell_level"]
        assert entry_sell > 102.6    # the target was out of reach at the fill

        # 13:30, on a bar that never trades above that fill: flat price, and the
        # target has descended through it.
        tape.append(102.6, low=102.6, high=102.6, offset=240)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "sold"
        assert trader.plan["sell_level"] < 102.6 < entry_sell
        assert "Target" in tracker.snapshot()["decisions"][-1].reasoning

    def test_a_breach_moves_the_forecast_and_the_band_is_rebuilt_from_it(
        self, state, market_open, monkeypatch
    ):
        """The two settings compose: the breach update owns the forecast, the
        level source owns how it is read."""
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(112.0))
        tape = Tape(monkeypatch)
        trader = self._trader(monkeypatch, breach_update="extreme")

        tape.append(111.5, high=112.0, offset=65)
        trader.run_cycle(DAYRANGE_BUNDLE, state, tracker)
        assert trader.plan["pred_high"] == pytest.approx(112.0)
        assert trader.plan["reference"] == pytest.approx(expected_reference(65, high=112.0))

    def test_a_missing_shape_falls_back_to_the_flat_high_and_says_so(
        self, state, market_open, monkeypatch
    ):
        """`config_error` refuses a run whose symbol has no shape, so this is
        the narrow case of one that vanished between the launch and 9:35 --
        better a notebook-rule session than no levels at all."""
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(104.0))
        tape = Tape(monkeypatch)
        monkeypatch.setattr(at.intraday_vol_model, "load", lambda *a, **k: None)
        trader = at.DayRangeTrader(dayrange_config(level_source="intraday"))

        tape.append(104.0, offset=65)
        trader.run_cycle(DAYRANGE_BUNDLE, state, tracker)
        assert trader.plan["reference"] == FORECAST["pred_high"]
        assert trader.plan["buy_level"] == pytest.approx(BUY_LEVEL)
        assert any(
            "IntradayVolatility shape" in e.get("text", "")
            for e in state.agent_log if e.get("type") == "error"
        )

    def test_the_flat_source_never_loads_the_shape(self, state, market_open, monkeypatch):
        """A run that does not read it must not pay for the file."""
        calls = []
        monkeypatch.setattr(
            at.intraday_vol_model, "load",
            lambda *a, **k: calls.append(a) or VOL_SHAPE,
        )
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(104.0))
        tape = Tape(monkeypatch)
        trader = at.DayRangeTrader(dayrange_config())
        tape.append(104.0, offset=65)
        trader.run_cycle(DAYRANGE_BUNDLE, state, tracker)
        assert calls == []


class TestIntradayLevelSourceAvailability:
    """It is a second model, so it is a second thing that can be missing."""

    def test_a_symbol_the_shape_was_never_fitted_on_is_refused(self, monkeypatch):
        monkeypatch.setattr(at.apple_models, "covers", lambda *a, **k: True)
        config = dayrange_config(ticker="MSFT", level_source="intraday")
        error = at.config_error(config)
        assert error is not None and "MSFT" in error and "IntradayVolatility" in error

    def test_a_missing_export_is_refused_with_the_path_it_looked_for(self, monkeypatch):
        monkeypatch.setattr(at.intraday_vol_model, "load", lambda *a, **k: None)
        error = at.config_error(dayrange_config(level_source="intraday"))
        assert error is not None and "intravol_AAPL.json" in error

    def test_the_flat_source_needs_none_of_it(self, monkeypatch):
        monkeypatch.setattr(at.intraday_vol_model, "load", lambda *a, **k: None)
        assert at.config_error(dayrange_config()) is None

    def test_an_unknown_level_source_is_refused(self):
        with pytest.raises(ValueError, match="level_source"):
            dayrange_config(level_source="vwap")


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
        base = config_signature(dayrange_config(stop_k=0.0, momentum_drop=0.0))
        assert base == "dayrange_AAPL(buy=H-0.75A,sell=H-0.1A,size=95%)"
        assert base != config_signature(
            dayrange_config(buy_k=0.8, stop_k=0.0, momentum_drop=0.0)
        )
        assert base != config_signature(
            dayrange_config(sell_k=0.2, stop_k=0.0, momentum_drop=0.0)
        )

    def test_the_exit_is_in_the_signature_only_while_switched_on(self):
        on = config_signature(dayrange_config())
        assert on == (
            "dayrange_AAPL(buy=H-0.75A,sell=H-0.1A,size=95%,"
            "stop=E-0.2A,take=70%@mom-1,runner>=0.3A)"
        )
        for field, value in (
            ("stop_k", 0.3), ("momentum_drop", 1.5),
            ("take_fraction", 0.5), ("hold_min_gain_k", 0.5),
        ):
            assert config_signature(dayrange_config(**{field: value})) != on, field
        # With the take off its two knobs trade nothing, so they sign nothing.
        off = dayrange_config(momentum_drop=0.0)
        assert config_signature(off) == config_signature(
            replace(off, take_fraction=0.5, hold_min_gain_k=0.9)
        )

    def test_the_intraday_update_is_in_the_signature_only_while_switched_on(self):
        off = config_signature(dayrange_config(stop_k=0.0, momentum_drop=0.0))
        assert off == "dayrange_AAPL(buy=H-0.75A,sell=H-0.1A,size=95%)"
        for policy in ("extreme", "brownian"):
            signed = config_signature(
                dayrange_config(stop_k=0.0, momentum_drop=0.0, breach_update=policy)
            )
            assert signed == off[:-1] + f",breach={policy})"

    def test_the_level_source_is_in_the_signature_only_when_it_is_not_the_high(self):
        """It sits beside the two distances rather than at the end: "H" in
        `buy=H-0.75A` is whatever the source says it is."""
        flat = config_signature(dayrange_config(stop_k=0.0, momentum_drop=0.0))
        assert flat == "dayrange_AAPL(buy=H-0.75A,sell=H-0.1A,size=95%)"
        assert config_signature(
            dayrange_config(stop_k=0.0, momentum_drop=0.0, level_source="intraday")
        ) == "dayrange_AAPL(buy=H-0.75A,sell=H-0.1A,levels=intraday,size=95%)"

    def test_the_intraday_update_defaults_to_the_extreme_so_far(self):
        """`dayrange_config` pins it off; the app's own default does not."""
        assert AppleTraderConfig().breach_update == "extreme"

    def test_an_unknown_update_policy_is_refused(self):
        """Read as "off" once a record carries it, but refused while a config is
        being built -- the earliest place a typo can be reported is the best one."""
        with pytest.raises(ValueError, match="breach_update"):
            dayrange_config(breach_update="mean_reversion")

    def test_exit_distances_cannot_be_negative_and_the_take_is_a_share(self):
        for field in ("stop_k", "momentum_drop", "hold_min_gain_k"):
            with pytest.raises(ValueError, match=field):
                dayrange_config(**{field: -0.1})
        for fraction in (0.0, 1.5):
            with pytest.raises(ValueError, match="take_fraction"):
                dayrange_config(take_fraction=fraction)

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
        assert config_signature(AppleTraderConfig(model_key="dayrange")).startswith(
            f"dayrange_AAPL(buy=H-{buy_k:g}A,sell=H-{sell_k:g}A,size=95%"
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
