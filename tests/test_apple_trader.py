"""Apple Trader: the rule-based loop over the day-range forecast.

Every cycle is driven through a stubbed forecast, so these pin the RULES --
when it buys, when it refuses to, and every way it gets back out -- without
depending on the saved artifact or on live market data.
"""

import threading
from dataclasses import replace
from datetime import date, datetime, timezone

import pandas as pd
import pytest

from agent_stonks import apple_models, intraday_vol_model
from agent_stonks import apple_trader as at
from agent_stonks import clock
from agent_stonks import rule_agent
from agent_stonks.apple_trader import DEFAULT_TICKER as TICKER
from agent_stonks.apple_trader import AppleTraderConfig, config_signature
from agent_stonks.config import UNIT_ADR, UNIT_PRED_RANGE
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
# What a target exit is playing for, and what the stop is written against: the
# two levels are 0.65 ADR apart, so at the default half-the-gain stop a fill at
# 103 risks $3.25 of the $6.50 the target pays.
TARGET_GAIN = SELL_LEVEL - BUY_LEVEL          # 6.50
STOP_PRICE = 103.0 - 0.5 * TARGET_GAIN        # 99.75


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
        # yfinance is one of the consolidated tapes the opening window falls
        # back to (`TestOpeningWindowTape`); no test may reach it for real.
        monkeypatch.setattr(at.historical, "fetch_intraday_bars", lambda *a, **k: [])
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
    the same reason `breach_update` and `min_win_k` are pinned off: these tests
    are about the rule as notebook 05 specified it, and a default that moves
    would rewrite what they assert. `TestIntradayRangeUpdate`,
    `TestIntradayLevelSource` and `TestMinimumWin` are where each is switched on.

    `level_unit`, `contain_range`, `keep_width` and `breach_exit` are pinned to what came
    before today's defaults, for the same reason and with the same consequence:
    `BUY_LEVEL` and `SELL_LEVEL` above are ADR arithmetic, and the breach
    policies in `TestIntradayRangeUpdate` are only separable from each other
    while containment is off. `TestPredictedRangeUnit`, `TestRangeContainment`
    and `TestBreachExit` are where each of the three is switched on.

    `scale_in` is pinned off for the same reason: the notebook buys once and
    waits. `TestScaleIn` is where the ladder is switched on, on the half-way
    rung `buy_step_k` is pinned to; `TestBuyStep` is where the step is set. So is `max_fall_k`:
    the notebook buys whatever the speed of the fall. `TestNoBuyIntoAFall` is
    where it is switched on.

    `momentum_confirmation_bars` is pinned off too, and the take pinned to the
    15-bar negative streak it replaced: most of this file is about that take,
    which stored records still replay. `TestMomentumConfirmation` is where the
    confirmation is switched on.

    `take_after_minutes` is pinned to 0, the take from the first bar after the
    fill: what every take test here was written against.
    `TestMomentumConfirmation` is where the wait is switched on, and the legacy
    realised-gain gate (`take_min_gain_fraction`, 0 by default) with it.

    `take_in_loss` is pinned off, the take in profit only: what every take test
    here was written against. `TestMomentumConfirmation` is where it is
    switched on.

    `limit_entry` is pinned off, the market buy: most tests here enter on a bar
    that dipped to the level and closed above it, filling at the close, which
    a limit at the level refuses. `TestLimitEntry` is where it is switched on.

    `skip_events` is pinned empty: the notebook trades every session, and the
    days-off calendar would otherwise decide which of these tests' dates trade.
    `TestDaysOff` is where it is switched on.
    """
    kwargs.setdefault("skip_events", ())
    kwargs.setdefault("limit_entry", False)
    kwargs.setdefault("momentum_confirmation_bars", 0)
    kwargs.setdefault("take_after_minutes", 0)
    kwargs.setdefault("take_in_loss", False)
    if not kwargs["momentum_confirmation_bars"] and not (
        kwargs.get("momentum_fade_bars") or kwargs.get("momentum_drop")
    ):
        kwargs.setdefault("negative_momentum_bars", 15)
    kwargs.setdefault("buy_k", 0.75)
    kwargs.setdefault("sell_k", 0.10)
    kwargs.setdefault("breach_update", "off")
    kwargs.setdefault("min_win_k", 0.0)
    kwargs.setdefault("level_unit", UNIT_ADR)
    kwargs.setdefault("contain_range", False)
    kwargs.setdefault("keep_width", False)
    kwargs.setdefault("breach_exit", False)
    kwargs.setdefault("scale_in", False)
    kwargs.setdefault("buy_step_k", 0.0)
    kwargs.setdefault("max_fall_k", 0.0)
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
        """Filled at 103, the default stop sits at STOP_PRICE -- a touch, like
        the levels, so the low decides."""
        broker = FakeBroker(103.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tape = Tape(monkeypatch, broker)
        trader = self._entered(state, tracker, tape)

        tape.append(100.0, low=STOP_PRICE + 0.01)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "hold"
        tape.append(99.6, low=STOP_PRICE - 0.01)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "sold"
        assert tracker.position_for(TICKER) == 0
        reasoning = tracker.snapshot()["decisions"][-1].reasoning
        assert "Stop loss" in reasoning and f"{STOP_PRICE:,.2f}" in reasoning

    def test_after_a_stop_the_session_buys_nothing_more(
        self, state, market_open, monkeypatch
    ):
        """The buy level is right above a stopped-out price, so re-arming it
        would buy the same slide again, one stop lower each time."""
        broker = FakeBroker(103.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tape = Tape(monkeypatch, broker)
        trader = self._entered(state, tracker, tape)

        tape.append(STOP_PRICE - 0.5)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "sold"
        # The live loop stops the agent on this; a replay just holds.
        assert trader.halt == "stopped out"
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
        trader = self._entered(
            state, tracker, tape, stop_gain_fraction=0.0, negative_momentum_bars=0
        )

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


class TestStopIsAShareOfThePredictedGain:
    """The stop is written against what the trade is playing for.

    `sell_level - buy_level` is `(buy_k - sell_k) x ADR` at every minute --
    both levels hang off the same reference, so whatever moves the reference
    moves them together -- which is what lets a fraction of it be a fixed price
    once there is a fill, and lets the same number mean the same bet on every
    instrument.
    """

    def _entered(self, state, tracker, tape, **kwargs):
        trader = at.DayRangeTrader(dayrange_config(**kwargs))
        tape.append(103.0, low=BUY_LEVEL - 0.01)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "bought"
        return trader

    def test_the_default_risks_half_of_what_the_target_pays(self):
        assert AppleTraderConfig().stop_gain_fraction == 0.5

    def test_the_distance_is_the_fraction_of_the_two_levels_gap(self):
        config = dayrange_config(buy_k=0.75, sell_k=0.10, stop_gain_fraction=0.5)
        # 0.65 ADR between the levels, half of it under the fill, $10 an ADR.
        assert config.target_gain_k == pytest.approx(0.65)
        assert at.stop_distance(config, 10.0) == pytest.approx(3.25)

    def test_widening_the_levels_widens_the_stop_with_them(self):
        """The point of the reparameterisation: risk follows reward instead of
        being a distance that has to be re-picked whenever the levels move."""
        narrow = dayrange_config(buy_k=0.40, sell_k=0.25)
        wide = dayrange_config(buy_k=0.90, sell_k=0.05)
        assert at.stop_distance(narrow, 10.0) == pytest.approx(0.75)
        assert at.stop_distance(wide, 10.0) == pytest.approx(4.25)

    def test_zero_switches_it_off(self):
        config = dayrange_config(stop_gain_fraction=0.0)
        assert at.stop_distance(config, 10.0) == 0.0
        assert not config.has_stop

    def test_whether_there_is_a_stop_is_answerable_before_a_session_is(self):
        """The run opens its log before there is an ADR to measure against."""
        assert dayrange_config().has_stop
        assert dayrange_config(stop_gain_fraction=0.0, stop_k=0.2).has_stop

    def test_the_stop_fires_at_that_price(self, state, market_open, monkeypatch):
        broker = FakeBroker(103.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tape = Tape(monkeypatch, broker)
        trader = self._entered(state, tracker, tape, stop_gain_fraction=0.2)

        stop = 103.0 - 0.2 * TARGET_GAIN   # 101.70
        tape.append(102.0, low=stop + 0.01)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "hold"
        tape.append(101.6, low=stop - 0.01)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "sold"
        assert f"{stop:,.2f}" in tracker.snapshot()["decisions"][-1].reasoning

    def test_the_log_says_what_the_stop_was_written_as(
        self, state, market_open, monkeypatch
    ):
        broker = FakeBroker(103.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tape = Tape(monkeypatch, broker)
        self._entered(state, tracker, tape)
        entry = tracker.snapshot()["decisions"][-1].reasoning
        assert "0.5 × the predicted gain" in entry and "$3.25" in entry

    def test_a_replayed_record_reads_back_in_its_own_units(
        self, state, market_open, monkeypatch
    ):
        """A run recorded before the stop was written this way keeps saying
        ADRs -- the log has to agree with the form that produced it."""
        broker = FakeBroker(103.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tape = Tape(monkeypatch, broker)
        self._entered(state, tracker, tape, stop_gain_fraction=0.0, stop_k=0.2)
        entry = tracker.snapshot()["decisions"][-1].reasoning
        assert "0.2 × ADR" in entry and "$2.00" in entry

    def test_the_legacy_unit_still_stops_where_it_did(
        self, state, market_open, monkeypatch
    ):
        broker = FakeBroker(103.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tape = Tape(monkeypatch, broker)
        trader = self._entered(
            state, tracker, tape, stop_gain_fraction=0.0, stop_k=0.2
        )
        tape.append(101.5, low=101.01)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "hold"
        tape.append(101.2, low=100.99)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "sold"
        assert "101.00" in tracker.snapshot()["decisions"][-1].reasoning

    def test_the_two_units_cannot_both_be_set(self):
        """Not a wider stop or a narrower one -- a config that does not say
        which rule it means."""
        with pytest.raises(ValueError, match="only one may be set"):
            dayrange_config(stop_gain_fraction=0.5, stop_k=0.2)

    def test_each_unit_signs_as_itself(self):
        """A stored record's signature is its identity everywhere downstream,
        so replaying one must not rewrite it into the new units."""
        assert ",stop=E-0.5G" in config_signature(dayrange_config())
        assert ",stop=E-0.2A" in config_signature(
            dayrange_config(stop_gain_fraction=0.0, stop_k=0.2)
        )

    def test_a_stop_that_risks_more_than_the_target_pays_is_allowed(self):
        """Legal, and rarely meant -- the form warns rather than refusing,
        because the rule also exits on momentum and at the close."""
        config = dayrange_config(stop_gain_fraction=1.5)
        assert at.stop_distance(config, 10.0) > TARGET_GAIN


class TestDayRangeMomentumTake:
    """Banking gains short of the target once momentum has been negative for long enough.

    Every test fills at 103 on the $10 ADR, so the 109 sell level is 0.6 ADR
    above the fill -- past the 0.3 worth keeping a runner for -- and the stop
    sits at 101. Momentum is read from the closes: the 3-bar change `close -
    close[3 bars ago]`, which the take wants negative 2 bars in a row, so a
    test walks the price through a streak in a handful of bars. The opening
    window closes at 101.0 .. 101.4.
    """

    def _entered(self, state, monkeypatch, **kwargs):
        kwargs.setdefault("negative_momentum_bars", 3)
        kwargs.setdefault("negative_for_bars", 2)
        broker = FakeBroker(103.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tape = Tape(monkeypatch, broker, real_momentum=True)
        trader = at.DayRangeTrader(dayrange_config(**kwargs))
        tape.append(103.0, low=BUY_LEVEL - 0.01)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "bought"
        return trader, tracker, tape

    @staticmethod
    def _walk(trader, state, tracker, tape, closes, expect="hold"):
        for close in closes:
            tape.append(close)
            assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == expect, close

    def _rise(self, trader, state, tracker, tape):
        """Up off the fill and into profit: 105.0, 105.5, 104.8 -- momentum
        +3.7, +4.1, +1.8, none of it negative."""
        self._walk(trader, state, tracker, tape, (105.0, 105.5, 104.8))

    def _taken(self, state, monkeypatch, **kwargs):
        trader, tracker, tape = self._entered(state, monkeypatch, **kwargs)
        held = tracker.position_for(TICKER)
        self._rise(trader, state, tracker, tape)
        self._walk(trader, state, tracker, tape, (104.6,))  # -0.40 against 105.0
        self._walk(trader, state, tracker, tape, (104.4,), "sold")  # -1.10 against 105.5
        return trader, tracker, tape, held

    def test_negative_for_long_enough_in_profit_banks_most_and_keeps_a_runner(
        self, state, market_open, monkeypatch
    ):
        trader, tracker, tape, held = self._taken(state, monkeypatch)

        runner = tracker.position_for(TICKER)
        assert held - runner == int(held * 0.7)
        assert 0 < runner < held
        reasoning = tracker.snapshot()["decisions"][-1].reasoning
        assert "Momentum take" in reasoning and "Banking" in reasoning
        assert "3-bar momentum has been negative for the last 2 bars" in reasoning

    def test_one_negative_bar_is_not_long_enough_and_a_positive_one_restarts_the_count(
        self, state, market_open, monkeypatch
    ):
        trader, tracker, tape = self._entered(state, monkeypatch)
        held = tracker.position_for(TICKER)
        self._rise(trader, state, tracker, tape)
        # -0.40, then +0.50 against 105.5: the streak is broken at one ...
        self._walk(trader, state, tracker, tape, (104.6, 106.0, 105.8))
        # ... and has to be built again from nothing: -0.10, then -2.00.
        self._walk(trader, state, tracker, tape, (104.5,))
        assert tracker.position_for(TICKER) == held
        self._walk(trader, state, tracker, tape, (104.0,), "sold")

    def test_how_long_is_long_enough_is_configurable(self, state, market_open, monkeypatch):
        trader, tracker, tape = self._entered(state, monkeypatch, negative_for_bars=3)
        self._rise(trader, state, tracker, tape)
        self._walk(trader, state, tracker, tape, (104.6, 104.4))
        self._walk(trader, state, tracker, tape, (104.3,), "sold")  # -0.50 against 104.8
        assert "negative for the last 3 bars" in tracker.snapshot()["decisions"][-1].reasoning

    def test_the_fall_into_the_fill_does_not_count(self, state, market_open, monkeypatch):
        """The dip that reached the buy level is falling by construction: a
        streak it started is not this trade going wrong, so only bars after the
        fill count toward it."""
        broker = FakeBroker(103.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tape = Tape(monkeypatch, broker, real_momentum=True)
        trader = at.DayRangeTrader(
            dayrange_config(negative_momentum_bars=3, negative_for_bars=2)
        )
        for close in (106.0, 105.5, 105.0):
            tape.append(close)
        tape.append(103.0, low=BUY_LEVEL - 0.01)  # -3.00 against 106.0
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "bought"

        self._walk(trader, state, tracker, tape, (103.5,))  # -2.00: one bar, not two
        self._walk(trader, state, tracker, tape, (103.4,), "sold")  # -1.60: two

    def test_momentum_not_known_yet_is_not_negative(self, state, market_open, monkeypatch):
        """The first N bars of the session have nothing to compare against --
        the chart's panel leaves them blank -- and a blank is not a streak."""
        trader, tracker, tape = self._entered(
            state, monkeypatch, negative_momentum_bars=10
        )
        held = tracker.position_for(TICKER)
        self._walk(trader, state, tracker, tape, (105.0, 104.8, 104.6, 104.4))
        assert tracker.position_for(TICKER) == held

    def test_there_is_no_gain_to_take_under_water(self, state, market_open, monkeypatch):
        trader, tracker, tape = self._entered(state, monkeypatch)
        held = tracker.position_for(TICKER)
        self._rise(trader, state, tracker, tape)
        # Negative two bars running, but the second is below the fill (and
        # above the stop): that is the stop's business, not the take's.
        self._walk(trader, state, tracker, tape, (104.6, 102.9))
        assert tracker.position_for(TICKER) == held

    def test_the_share_taken_is_configurable(self, state, market_open, monkeypatch):
        _, tracker, _, held = self._taken(state, monkeypatch, take_fraction=0.5)
        assert tracker.position_for(TICKER) == held - int(held * 0.5)

    def test_the_runner_is_not_trimmed_again_and_waits_for_the_sell_level(
        self, state, market_open, monkeypatch
    ):
        trader, tracker, tape, _ = self._taken(state, monkeypatch)
        runner = tracker.position_for(TICKER)

        self._walk(trader, state, tracker, tape, (104.0,))  # still negative
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
        _, tracker, _, _ = self._taken(state, monkeypatch, hold_min_gain_k=0.7)
        assert tracker.position_for(TICKER) == 0
        assert "whole position" in tracker.snapshot()["decisions"][-1].reasoning

    def test_the_default_rule_end_to_end(self, state, market_open, monkeypatch):
        """At the shipped 15 bars / 5 in a row: a zig-zag climb that never
        reaches the target, then a zig-zag slide. The climb keeps the 15-bar
        change positive; the slide takes it under zero six bars down from the
        106.00 top, and the take waits out five bars of that before selling at
        104.70 -- three bars and 50 cents later than the legacy sigma turn on
        the same tape (`TestLegacyMomentumFade`). The runner goes back out at
        the fill."""
        trader, tracker, tape = self._entered(
            state, monkeypatch, negative_momentum_bars=15, negative_for_bars=5
        )
        price = 103.0
        for step in [0.3, -0.1] * 15 + [-0.3, 0.1] * 15:
            price += step
            tape.append(round(price, 2))
            trader.run_cycle(DAYRANGE_BUNDLE, state, tracker)

        fills = [d for d in tracker.snapshot()["decisions"] if d.status == "filled"]
        assert [d.action for d in fills] == ["buy", "sell", "sell"]
        buy, take, breakeven = fills
        assert "Momentum take" in take.reasoning and take.price == pytest.approx(104.7)
        assert "15-bar momentum has been negative for the last 5 bars" in take.reasoning
        assert take.filled_quantity == int(buy.filled_quantity * 0.7)
        assert "Breakeven" in breakeven.reasoning
        assert tracker.position_for(TICKER) == 0


class TestLegacyMomentumFade:
    """The two takes stored records replay under, now that neither is configured.

    `momentum_fade_bars` is the positive-to-balanced turn of the sigma score
    (2026-09-21 to -23), `momentum_drop` the fall from the peak before it. The
    score is scripted per bar, except in the last test, which computes it from
    the closes. Same fill, levels and stop as `TestDayRangeMomentumTake`.
    """

    def _entered(self, state, monkeypatch, **kwargs):
        kwargs.setdefault("negative_momentum_bars", 0)
        kwargs.setdefault("momentum_fade_bars", 15)
        broker = FakeBroker(103.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tape = Tape(monkeypatch, broker, real_momentum=kwargs.pop("real_momentum", False))
        trader = at.DayRangeTrader(dayrange_config(**kwargs))
        tape.append(103.0, low=BUY_LEVEL - 0.01, mom=0.0)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "bought"
        return trader, tracker, tape

    def test_a_fading_move_in_profit_banks_most_and_keeps_a_runner(
        self, state, market_open, monkeypatch
    ):
        trader, tracker, tape = self._entered(state, monkeypatch)
        held = tracker.position_for(TICKER)

        tape.append(105.0, mom=2.0)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "hold"
        tape.append(105.5, mom=1.2)  # off its peak, but still positive
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "hold"
        # Under the 0.9σ it took to turn positive, but a regime is left only
        # under 0.4σ: a score hovering at the line is not a turn.
        tape.append(105.4, mom=0.6)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "hold"
        tape.append(105.2, mom=0.3)  # balanced
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "sold"

        runner = tracker.position_for(TICKER)
        assert held - runner == int(held * 0.7)
        reasoning = tracker.snapshot()["decisions"][-1].reasoning
        assert "15-bar momentum" in reasoning and "turned balanced" in reasoning

    def test_a_turn_straight_to_negative_takes_too(self, state, market_open, monkeypatch):
        trader, tracker, tape = self._entered(state, monkeypatch)
        tape.append(105.0, mom=2.0)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "hold"
        tape.append(104.8, mom=-1.2)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "sold"
        assert "turned negative" in tracker.snapshot()["decisions"][-1].reasoning

    def test_momentum_that_never_turned_positive_has_not_faded(
        self, state, market_open, monkeypatch
    ):
        trader, tracker, tape = self._entered(state, monkeypatch)
        held = tracker.position_for(TICKER)
        for mom in (-1.5, 0.2, 0.8, 0.1):
            tape.append(104.5, mom=mom)
            assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "hold"
        assert tracker.position_for(TICKER) == held

    def test_the_look_back_is_the_momentum_horizon(self, state, market_open, monkeypatch):
        """N is handed to the score as its horizon, not applied after it."""
        seen = []
        trader, tracker, tape = self._entered(state, monkeypatch, momentum_fade_bars=7)
        monkeypatch.setattr(
            at.momentum_regime, "compute_momentum",
            lambda frame, params=None: seen.append(params) or frame,
        )
        tape.append(105.0, mom=2.0)
        trader.run_cycle(DAYRANGE_BUNDLE, state, tracker)
        assert seen and seen[-1]["horizon"] == 7

    def test_the_older_record_keeps_the_fall_from_the_peak(
        self, state, market_open, monkeypatch
    ):
        """1σ off the peak, whatever the regime says."""
        trader, tracker, tape = self._entered(
            state, monkeypatch, momentum_fade_bars=0, momentum_drop=1.0
        )
        tape.append(105.0, mom=2.0)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "hold"
        tape.append(105.5, mom=1.2)  # 0.8σ off the peak
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "hold"
        tape.append(105.2, mom=0.9)  # 1.1σ, still a positive regime
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "sold"
        assert "1σ is the trigger" in tracker.snapshot()["decisions"][-1].reasoning

    def test_the_peak_is_this_positions_not_the_mornings(
        self, state, market_open, monkeypatch
    ):
        trader, tracker, tape = self._entered(state, monkeypatch)
        held = tracker.position_for(TICKER)
        for row in tape.rows[:-1]:
            row["mom"] = 3.0

        tape.append(104.0, mom=0.2)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "hold"
        assert tracker.position_for(TICKER) == held

    def test_the_score_is_computed_from_the_tape(self, state, market_open, monkeypatch):
        """End to end on the real momentum score: the climb carries the 15-bar
        score positive (to about +1.95σ) and its zig-zag never takes it back
        under the 0.4σ exit line; the turn does, eight bars down from the
        106.00 top at 105.20, and the runner goes back out at the fill."""
        trader, tracker, tape = self._entered(state, monkeypatch, real_momentum=True)
        price = 103.0
        for step in [0.3, -0.1] * 15 + [-0.3, 0.1] * 15:
            price += step
            tape.append(round(price, 2))
            trader.run_cycle(DAYRANGE_BUNDLE, state, tracker)

        fills = [d for d in tracker.snapshot()["decisions"] if d.status == "filled"]
        assert [d.action for d in fills] == ["buy", "sell", "sell"]
        buy, take, breakeven = fills
        assert "Momentum take" in take.reasoning and take.price == pytest.approx(105.2)
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

    def test_shift_moves_both_sides_and_the_levels_translate_by_the_breach(
        self, state, market_open, monkeypatch
    ):
        """"Move to the extreme so far" (since 2026-09-23): up $2 through the
        $110 high moves the $95 low up $2 too. Under the predicted-range unit
        the width -- and so the unit -- is unchanged, and both levels move by
        exactly the breach."""
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(112.0))
        tape = Tape(monkeypatch)
        tape.append(111.5, high=112.0)
        trader = self._run("shift", state, tracker, tape, level_unit=UNIT_PRED_RANGE)

        assert trader.plan["pred_high"] == pytest.approx(112.0)
        assert trader.plan["pred_low"] == pytest.approx(97.0)
        assert trader.plan["buy_level"] == pytest.approx(110.0 - 0.75 * 15 + 2.0)
        assert trader.plan["sell_level"] == pytest.approx(110.0 - 0.10 * 15 + 2.0)
        # The log names the side the tape went through, not the one that followed.
        (line,) = [
            e["text"] for e in state.agent_log if "forecast updated" in e.get("text", "")
        ]
        assert "traded up through the $110.00 predicted high" in line
        assert "predicted low" not in line
        assert "and the other with it" in line

    def test_shift_on_a_breach_of_the_low_brings_the_high_down(
        self, state, market_open, monkeypatch
    ):
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(94.0))
        tape = Tape(monkeypatch)
        tape.append(94.5, low=94.0)
        trader = self._run("shift", state, tracker, tape)

        assert trader.plan["pred_low"] == pytest.approx(94.0)
        assert trader.plan["pred_high"] == pytest.approx(109.0)

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


class TestDayRangeIntradayModel:
    """The reference as a model rather than as a setting beside one.

    "Day Range × Intraday Volatility" is the same forecast and the same rules
    with the levels measured below the intraday curve, so it is one choice in
    the model picker instead of two choices the reader had to pair correctly.
    """

    KEY = "dayrange_intraday"

    def test_the_model_decides_what_the_levels_are_measured_below(self):
        assert AppleTraderConfig(model_key=self.KEY).level_source == "intraday"
        assert AppleTraderConfig(model_key="dayrange").level_source == "dayrange"

    def test_it_cannot_be_talked_out_of_its_own_reference(self):
        """A config naming this model and the flat high would be this model in
        name only -- the reference is the whole of what distinguishes it."""
        config = AppleTraderConfig(model_key=self.KEY, level_source="dayrange")
        assert config.level_source == "intraday"

    def test_the_model_key_says_it_so_the_signature_does_not_say_it_twice(self):
        sig = config_signature(AppleTraderConfig(model_key=self.KEY))
        assert sig.startswith("dayrange_intraday_AAPL(")
        assert "levels=" not in sig

    def test_a_record_from_when_it_was_a_setting_keeps_its_signature(self):
        """The pairing existed before this model did, as `dayrange` plus
        `level_source=intraday`. Such a record has to replay as what it was and
        file where it did -- rewriting it onto the new key would move it to a
        different row in Results."""
        from simlab.rule_agents import _apple_from_record

        old = _apple_from_record({
            "model_key": "dayrange", "buy_k": 0.75, "sell_k": 0.10,
            "level_source": "intraday",
        })
        assert old.model_key == "dayrange"
        assert old.level_source == "intraday"
        assert ",levels=intraday" in config_signature(old)

    def test_a_record_with_no_reference_at_all_is_still_the_flat_model(self):
        from simlab.rule_agents import _apple_from_record

        old = _apple_from_record({"model_key": "dayrange", "buy_k": 0.75})
        assert old.level_source == "dayrange"
        assert "levels=" not in config_signature(old)

    def test_it_is_offered_only_where_both_files_were_fitted(self):
        both = set(apple_models.DAYRANGE_TICKERS) & set(
            at.intraday_vol_model.TICKERS
        )
        assert set(apple_models.get(self.KEY).tickers) == both
        for symbol in both:
            assert self.KEY in apple_models.keys_for(symbol)

    def test_a_missing_shape_makes_the_model_unavailable_not_silently_flat(
        self, monkeypatch
    ):
        """The one outcome worth failing for: without the shape it would run as
        the flat model under this model's name."""
        monkeypatch.setattr(
            apple_models, "_load_dayrange", lambda ticker=TICKER: {"stub": True}
        )
        monkeypatch.setattr(
            at.intraday_vol_model, "load", lambda *a, **k: None
        )
        assert apple_models.load(self.KEY, TICKER) is None
        why = apple_models.unavailable_reason(self.KEY, TICKER)
        assert "IntradayVolatility" in why and "missing" in why

    def test_a_missing_day_range_bundle_says_so_instead(self, monkeypatch):
        monkeypatch.setattr(
            apple_models, "_load_dayrange", lambda ticker=TICKER: None
        )
        why = apple_models.unavailable_reason(self.KEY, TICKER)
        assert "day-range bundle" in why

    def test_both_models_drive_the_same_rule_set(self):
        assert (
            apple_models.get(self.KEY).strategy
            == apple_models.get("dayrange").strategy
        )


class TestHighLowModel:
    """HighLow changes the forecast and nothing else: the same state machine,
    the same levels arithmetic, fed HighLow's predicted high and range."""

    KEY = "highlow"
    BUNDLE = {"opening_minutes": 5, "kind": "highlow"}

    def _module(self):
        return at._highlow()

    def _stub_highlow(self, monkeypatch, forecast):
        calls = []

        def fake(bundle, ticker, opening, today, key=None, secret=None):
            calls.append({"ticker": ticker, "bars": len(opening), "key": key, "secret": secret})
            return dict(forecast)

        monkeypatch.setattr(self._module(), "forecast_session", fake)
        return calls

    def test_it_drives_the_day_range_trader(self):
        config = AppleTraderConfig(model_key=self.KEY)
        assert isinstance(at.build_trader(config, self.BUNDLE), at.DayRangeTrader)
        assert config.level_source == "dayrange"

    def test_the_levels_hang_off_the_highlow_forecast(self, state, market_open, monkeypatch):
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(104.0))
        tape = Tape(monkeypatch)
        calls = self._stub_highlow(monkeypatch, {**FORECAST, "pred_high": 120.0})
        trader = at.DayRangeTrader(replace(dayrange_config(), model_key=self.KEY))

        tape.append(104.0, low=104.0)
        trader.run_cycle(self.BUNDLE, state, tracker)
        # 120 - 0.75 x 10 and 120 - 0.10 x 10: HighLow's high, the same rule.
        assert trader.plan["buy_level"] == pytest.approx(112.5)
        assert trader.plan["sell_level"] == pytest.approx(119.0)
        # TimeToChange3 was never asked, and the SIP history got the run's keys.
        assert tape.forecast_calls == 0
        assert calls == [{"ticker": TICKER, "bars": self.BUNDLE["opening_minutes"],
                          "key": "k", "secret": "s"}]

    def test_a_replay_withholds_its_placeholder_keys(self, state, market_open, monkeypatch):
        """SimLab's state carries "simulated" keys; sending those to Alpaca is a
        401 on every session. None lets the history read the environment's."""
        state.bar_tape_override = "yfinance"
        state.api_key = state.api_secret = "simulated"
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(104.0))
        Tape(monkeypatch).append(104.0, low=104.0)
        calls = self._stub_highlow(monkeypatch, FORECAST)
        trader = at.DayRangeTrader(replace(dayrange_config(), model_key=self.KEY))
        trader.run_cycle(self.BUNDLE, state, tracker)
        assert (calls[0]["key"], calls[0]["secret"]) == (None, None)

    def test_under_the_predicted_range_unit_it_is_highlows_range(
        self, state, market_open, monkeypatch
    ):
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(104.0))
        Tape(monkeypatch).append(104.0, low=104.0)
        self._stub_highlow(monkeypatch, {**FORECAST, "pred_high": 110.0, "pred_low": 106.0})
        trader = at.DayRangeTrader(
            replace(dayrange_config(level_unit=UNIT_PRED_RANGE), model_key=self.KEY)
        )
        trader.run_cycle(self.BUNDLE, state, tracker)
        assert trader.plan["buy_level"] == pytest.approx(110.0 - 0.75 * 4.0)

    def test_a_highlow_failure_stands_the_session_down(self, state, market_open, monkeypatch):
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(104.0))
        tape = Tape(monkeypatch)

        def boom(*a, **k):
            raise ValueError("only 40 complete SIP sessions of history")

        monkeypatch.setattr(self._module(), "forecast_session", boom)
        trader = at.DayRangeTrader(replace(dayrange_config(), model_key=self.KEY))
        tape.append(104.0, low=90.0)
        assert trader.run_cycle(self.BUNDLE, state, tracker) == "no_data"
        assert "SIP sessions" in trader.blocked["reason"]
        assert tracker.position_for(TICKER) == 0

    def test_it_is_refused_on_a_symbol_it_was_not_fitted_on(self):
        config = AppleTraderConfig(ticker="GOOGL", model_key=self.KEY)
        assert "HighLow" in at.model_ticker_error(config)

    def test_it_signs_its_own_row_in_results(self):
        sig = config_signature(AppleTraderConfig(model_key=self.KEY))
        assert sig.startswith(f"{self.KEY}_AAPL(")


class TestHighLow2Model(TestHighLowModel):
    """HighLow2 likewise: every HighLow test above, its bundle dispatched to
    `highlow2_model` instead."""

    KEY = "highlow2"
    BUNDLE = {"opening_minutes": 5, "kind": "highlow2", "opening_feed": "iex"}

    def _module(self):
        return at._highlow2()

    def test_the_iex_caveat_does_not_describe_it(self, state, market_open, monkeypatch):
        """It reads its own IEX window, fitted on IEX volume, whatever tape the
        run streams -- where a SIP-opening HighLow bundle on an IEX tape is warned."""
        opening_window = at.DayRangeTrader._opening_window
        monkeypatch.setattr(
            at.DayRangeTrader, "_opening_window",
            lambda trader, *a, **k: (opening_window(trader, *a, **k)[0], "iex"),
        )
        Tape(monkeypatch).append(104.0, low=104.0)
        monkeypatch.setattr(at._highlow(), "forecast_session", lambda *a, **k: dict(FORECAST))
        self._stub_highlow(monkeypatch, FORECAST)

        def caveats(key, bundle):
            state.agent_log.clear()
            tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(104.0))
            trader = at.DayRangeTrader(replace(dayrange_config(), model_key=key))
            trader.run_cycle(bundle, state, tracker)
            assert trader.plan is not None
            return [e for e in state.agent_log if "caveat" in (e.get("text") or "")]

        assert caveats("highlow", TestHighLowModel.BUNDLE)
        assert caveats(self.KEY, self.BUNDLE) == []


class TestHighLow3mModel(TestHighLow2Model):
    """HighLow_3m likewise -- every HighLow and HighLow2 test above, its bundle
    dispatched to `highlow3m_model` and forecast after three minutes -- plus
    what its range being the one *after* the window changes."""

    KEY = "highlow3m"
    BUNDLE = {"opening_minutes": 3, "kind": "highlow3m", "opening_feed": "iex"}

    def _module(self):
        return at._highlow3m()

    def _breaches(self, monkeypatch, state, forecast, bundle, module):
        """Range updates after one bar past the window, with a high above the
        forecast in the window and nothing after it reaching it."""
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(104.0))
        tape = Tape(monkeypatch, minutes=bundle["opening_minutes"])  # window highs 101.5
        tape.append(101.0, high=101.1)
        monkeypatch.setattr(module, "forecast_session", lambda *a, **k: dict(forecast))
        trader = at.DayRangeTrader(
            replace(dayrange_config(breach_update="extreme"), model_key=bundle["kind"])
        )
        trader.run_cycle(bundle, state, tracker)
        return trader.plan.get("range_updates", 0), trader.plan["pred_high"]

    def test_the_windows_own_high_is_not_a_breach_of_a_range_after_it(
        self, state, market_open, monkeypatch
    ):
        forecast = {**FORECAST, "pred_high": 101.2, "pred_low": 99.0}
        updates, high = self._breaches(
            monkeypatch, state, {**forecast, "range_after_opening": True}, self.BUNDLE,
            self._module(),
        )
        assert (updates, high) == (0, 101.2)

    def test_a_whole_day_forecast_is_still_breached_by_the_window(
        self, state, market_open, monkeypatch
    ):
        """The other models' ranges cover the whole session: a window high
        above one has falsified it, as before."""
        forecast = {**FORECAST, "pred_high": 101.2, "pred_low": 99.0}
        updates, high = self._breaches(
            monkeypatch, state, forecast, TestHighLowModel.BUNDLE, at._highlow(),
        )
        assert (updates, high) == (1, pytest.approx(101.5))

    def test_a_breach_after_the_window_still_moves_it(self, state, market_open, monkeypatch):
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(104.0))
        tape = Tape(monkeypatch, minutes=3)
        tape.append(101.6, high=101.8)
        monkeypatch.setattr(self._module(), "forecast_session", lambda *a, **k: {
            **FORECAST, "pred_high": 101.2, "pred_low": 99.0, "range_after_opening": True,
        })
        trader = at.DayRangeTrader(
            replace(dayrange_config(breach_update="extreme"), model_key=self.KEY)
        )
        trader.run_cycle(self.BUNDLE, state, tracker)
        assert trader.plan["pred_high"] == pytest.approx(101.8)

    def test_the_caches_are_warmed_while_the_window_is_open(self, state, market_open, monkeypatch):
        """Once per session, in the background, with the run's keys; not once
        the window has closed (the forecast fetches then), never in a replay."""
        warmed, done = [], threading.Event()

        def warm(bundle, ticker, before, key=None, secret=None):
            warmed.append((ticker, before, key, secret))
            done.set()

        monkeypatch.setattr(self._module(), "warm_history", warm)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(104.0))
        Tape(monkeypatch, minutes=2)
        trader = at.DayRangeTrader(replace(dayrange_config(), model_key=self.KEY))
        assert trader.run_cycle(self.BUNDLE, state, tracker) == "warming_up"
        assert done.wait(5)
        trader.run_cycle(self.BUNDLE, state, tracker)
        assert [(t, k, s) for t, _, k, s in warmed] == [(TICKER, "k", "s")]

        warmed.clear()
        state.bar_tape_override = "yfinance"
        replay = at.DayRangeTrader(replace(dayrange_config(), model_key=self.KEY))
        replay.run_cycle(self.BUNDLE, state, tracker)
        assert warmed == []

    def test_the_armed_line_says_933_and_the_rest_of_the_day(self):
        bundle = {**self.BUNDLE, "trained_at": "2026-10-05",
                  "metadata": {"scores": {"test": {"mae_usd": 1.95}}}}
        line = at._armed_summary(AppleTraderConfig(model_key=self.KEY),
                                 apple_models.get(self.KEY), bundle)
        assert "at 9:33 it forecasts where the rest of today's AAPL high and low" in line
        assert "held-out mean error $1.95" in line
        other = at._armed_summary(AppleTraderConfig(model_key="highlow"),
                                  apple_models.get("highlow"), TestHighLowModel.BUNDLE)
        assert "at 9:35 it forecasts where today's AAPL high and low" in other

    def test_other_models_are_not_warmed(self, state, market_open, monkeypatch):
        monkeypatch.setattr(self._module(), "warm_history", lambda *a, **k: pytest.fail("warmed"))
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(104.0))
        Tape(monkeypatch, minutes=2)
        trader = at.DayRangeTrader(replace(dayrange_config(), model_key="highlow"))
        assert trader.run_cycle(TestHighLowModel.BUNDLE, state, tracker) == "warming_up"


class TestHighLow2RestHead:
    """A HighLow2 bundle with a rest-of-session head (INTC's) hands the trader
    the range after 9:35 as `pred_high` / `pred_low`: the reference high and
    the predicted-range unit are that range's, and the window's own extremes
    are not a breach of it."""

    BUNDLE = {**TestHighLow2Model.BUNDLE, "rest_head": True}
    # The rest of the session's 110 / 106 inside the day's 112 / 104.
    FORECAST = {**FORECAST, "pred_high": 110.0, "pred_low": 106.0, "day_high": 112.0,
                "day_low": 104.0, "range_after_opening": True}

    def _run(self, monkeypatch, state, forecast, **config):
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(104.0))
        tape = Tape(monkeypatch)  # window highs 101.5
        tape.append(101.0, high=101.1)
        monkeypatch.setattr(at._highlow2(), "forecast_session", lambda *a, **k: dict(forecast))
        trader = at.DayRangeTrader(
            replace(dayrange_config(**config), model_key=TestHighLow2Model.KEY)
        )
        trader.run_cycle(self.BUNDLE, state, tracker)
        return trader

    def test_the_levels_hang_off_the_rest_of_the_session(self, state, market_open, monkeypatch):
        trader = self._run(monkeypatch, state, self.FORECAST, level_unit=UNIT_PRED_RANGE)
        assert trader.plan["buy_level"] == pytest.approx(110.0 - 0.75 * 4.0)
        assert trader.plan["sell_level"] == pytest.approx(110.0 - 0.10 * 4.0)

    def test_the_windows_own_high_is_not_a_breach_of_it(self, state, market_open, monkeypatch):
        forecast = {**self.FORECAST, "pred_high": 101.2, "pred_low": 99.0}
        trader = self._run(monkeypatch, state, forecast, breach_update="extreme")
        assert (trader.plan.get("range_updates", 0), trader.plan["pred_high"]) == (0, 101.2)

    def test_the_log_says_which_range_it_trades(self, state, market_open, monkeypatch):
        self._run(monkeypatch, state, self.FORECAST)
        plan = next(e["text"] for e in state.agent_log if "forecast for" in (e.get("text") or ""))
        assert "forecast for the rest of the session after 09:34" in plan
        assert "high $110.00, low $106.00; the whole day's: high $112.00, low $104.00" in plan
        armed = at._armed_summary(AppleTraderConfig(ticker="INTC", model_key="highlow2"),
                                  apple_models.get("highlow2"), self.BUNDLE)
        assert "at 9:35 it forecasts where the rest of today's INTC high and low" in armed

    def test_a_day_only_bundle_logs_the_session(self, state, market_open, monkeypatch):
        day = {k: v for k, v in self.FORECAST.items()
               if k not in ("day_high", "day_low", "range_after_opening")}
        self._run(monkeypatch, state, day)
        plan = next(e["text"] for e in state.agent_log if "forecast for" in (e.get("text") or ""))
        assert "forecast for the session, from" in plan and "whole day" not in plan


class TestDaysOff:
    """A session the run sits out (`skip_events`, `event_days`) is never
    forecast: no model is asked, no order rests, and the log and the status
    line say which day it is. MIDSESSION, 2026-07-21, is an ordinary Tuesday;
    the calendar's own days are pinned in `tests/test_event_days.py`."""

    DAY = date(2026, 7, 21)

    def _run(self, state, monkeypatch, bars: int = 1, **kwargs):
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(104.0))
        tape = Tape(monkeypatch)
        trader = at.DayRangeTrader(dayrange_config(skip_events=at.event_days.CATEGORIES, **kwargs))
        outcomes = []
        for _ in range(bars):
            tape.append(104.0, low=90.0)  # through the buy level every bar
            outcomes.append(trader.run_cycle(DAYRANGE_BUNDLE, state, tracker))
        return trader, tape, tracker, outcomes

    def _flag(self, shock="geo", reason="US strikes on Iran overnight", hhmm="08:30"):
        made = pd.Timestamp(f"{self.DAY} {hhmm}").tz_localize("America/New_York").to_pydatetime()
        at.event_days.record_verdict(TICKER, shock, reason, made)

    def test_a_flagged_shock_day_is_never_forecast_or_traded(self, state, market_open, monkeypatch):
        self._flag()
        trader, tape, tracker, outcomes = self._run(state, monkeypatch, bars=3)
        assert outcomes == ["no_data"] * 3
        assert tape.forecast_calls == 0 and tracker.position_for(TICKER) == 0
        assert trader.blocked["sit_out"] == ["geo"]
        lines = [e["text"] for e in state.agent_log if "sits out" in e.get("text", "")]
        assert len(lines) == 1 and "US strikes on Iran overnight" in lines[0]
        assert trader.activity("no_data", tracker) == (
            at.rule_agent.WAITING, "Agent sits today out (Geopolitical shock)"
        )

    def test_a_calendar_day_needs_no_briefing(self, state, market_open, monkeypatch):
        monkeypatch.setattr(at.event_days, "load_calendar", lambda *a, **k: [{
            "date": str(self.DAY), "category": "cpi", "scope": "ALL",
            "event": "BLS Consumer Price Index release, 08:30 ET", "source": "",
        }])
        trader, tape, _, outcomes = self._run(state, monkeypatch)
        assert outcomes == ["no_data"] and tape.forecast_calls == 0
        assert trader.blocked["sit_out"] == ["cpi"]

    def test_an_ordinary_day_trades_and_says_what_it_could_not_check(self, state, market_open, monkeypatch):
        self._flag(shock="none", reason="")
        trader, tape, tracker, outcomes = self._run(state, monkeypatch)
        assert tape.forecast_calls == 1 and trader.plan is not None
        assert tracker.position_for(TICKER) > 0
        assert not [e for e in state.agent_log if "Days-off check" in e.get("text", "")]

    def test_a_briefing_still_being_written_is_waited_for_then_read(self, state, market_open, monkeypatch):
        state.premarket_pending = [TICKER]
        trader, tape, tracker, outcomes = self._run(state, monkeypatch, bars=2)
        assert outcomes == ["warming_up"] * 2 and tape.forecast_calls == 0
        waits = [e for e in state.agent_log if "Waiting for" in e.get("text", "")]
        assert len(waits) == 1  # once, not every minute
        self._flag()
        state.premarket_pending = []
        tape.append(104.0, low=90.0)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "no_data"
        assert trader.blocked["sit_out"] == ["geo"] and tape.forecast_calls == 0

    def test_a_briefing_that_never_lands_is_not_waited_for_all_day(self, state, market_open, monkeypatch):
        state.premarket_pending = [TICKER]
        trader, tape, _, outcomes = self._run(
            state, monkeypatch, bars=at.BRIEFING_WAIT_BARS + 1,
        )
        assert outcomes[:at.BRIEFING_WAIT_BARS - 1] == ["warming_up"] * (at.BRIEFING_WAIT_BARS - 1)
        assert tape.forecast_calls == 1 and trader.plan is not None

    def test_with_no_days_off_nothing_is_checked(self, state, market_open, monkeypatch):
        monkeypatch.setattr(at.event_days, "check", lambda *a, **k: pytest.fail("checked"))
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(104.0))
        tape = Tape(monkeypatch)
        trader = at.DayRangeTrader(dayrange_config())
        tape.append(104.0, low=104.0)
        trader.run_cycle(DAYRANGE_BUNDLE, state, tracker)
        assert tape.forecast_calls == 1

    def test_the_days_sign_the_run_and_an_old_record_has_none(self):
        from simlab.rule_agents import _apple_from_record

        sig = config_signature(AppleTraderConfig(skip_events=["geo", "earnings"]))
        assert sig.endswith(",skip=earn+geo)")
        assert ",skip=" not in config_signature(AppleTraderConfig(skip_events=()))
        assert _apple_from_record({"model_key": "dayrange"}).skip_events == ()
        assert AppleTraderConfig().skip_events == at.event_days.CATEGORIES

    def test_an_unknown_day_is_refused(self):
        with pytest.raises(ValueError, match="skip_events"):
            AppleTraderConfig(skip_events=("fomc",))


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


class TestMinimumWin:
    """The session circuit breaker: stop buying after a trade that barely paid.

    The forecast's claim is that the day is wide enough for the dip to be worth
    buying. A round trip that closed for almost nothing is that claim being
    tested and failing, so the levels are not re-armed on the same tape.

    On the fixture's $10 ADR a 0.2 threshold is $2.00 a share, which the tape's
    dollar-sized moves sit well under -- so these drive the threshold down
    rather than engineering huge price swings.

    The reference trade throughout is a 103.00 fill exited on the bar that
    reaches the 109.00 sell level, which fills at that bar's 108.80 close (the
    ledger is market-order only): $5.80 a share, 0.58 x ADR.
    """

    def _entered(self, state, monkeypatch, **kwargs):
        broker = FakeBroker(103.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tape = Tape(monkeypatch, broker)
        trader = at.DayRangeTrader(dayrange_config(**kwargs))
        tape.append(103.0, low=BUY_LEVEL - 0.01)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "bought"
        return trader, tracker, tape

    def _rearms(self, trader, tracker, tape, state) -> bool:
        """Whether the buy level still fills after the position closed."""
        tape.append(103.0, low=BUY_LEVEL - 0.01)
        return trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "bought"

    def test_off_by_default_in_these_tests_and_0_10_in_the_app(self):
        assert dayrange_config().min_win_k == 0.0
        # 0.10 on every instrument since 2026-09-23 -- what AAPL alone had
        # before, because its levels are only 0.15 ADR apart -- except BE
        # (0.01 since 2026-10-01, its HighLow levels are 0.10 apart).
        for ticker in ("AAPL", "GOOGL", "INTC"):
            assert AppleTraderConfig(ticker=ticker).min_win_k == 0.10, ticker

    def test_every_instruments_default_leaves_its_target_exit_alive(self):
        """The invariant AAPL's entry exists to keep: a trade that runs all the
        way to the sell level must clear the bar, or the breaker is really a
        one-trade-a-day rule wearing its name."""
        for model_key, model in apple_models.MODELS.items():
            for ticker in model.tickers:
                config = AppleTraderConfig(model_key=model_key, ticker=ticker)
                assert config.min_win_k < config.buy_k - config.sell_k, (model_key, ticker)

    def test_a_thin_win_stands_the_session_down(self, state, market_open, monkeypatch):
        # Sell level 109 is $6 over the 103 fill = 0.6 ADR, under the 0.8 bar.
        trader, tracker, tape = self._entered(state, monkeypatch, min_win_k=0.8)
        tape.append(108.8, high=SELL_LEVEL + 0.05)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "sold"

        assert trader.plan["stand_down"]
        # The breaker stands the session down but does not stop the agent.
        assert trader.halt is None
        assert not self._rearms(trader, tracker, tape, state)
        assert tracker.position_for(TICKER) == 0

    def test_a_win_over_the_bar_leaves_the_levels_armed(
        self, state, market_open, monkeypatch
    ):
        """The same trade, judged against a bar it clears: 0.6 ADR > 0.5."""
        trader, tracker, tape = self._entered(state, monkeypatch, min_win_k=0.5)
        tape.append(108.8, high=SELL_LEVEL + 0.05)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "sold"

        assert not trader.plan.get("stand_down")
        assert self._rearms(trader, tracker, tape, state)

    def test_the_bar_is_exclusive_so_exactly_the_threshold_stands_down(
        self, state, market_open, monkeypatch
    ):
        """"profit larger than k x ADR" -- landing exactly on it is not larger."""
        # Fill 103, target 109: exactly 0.6 ADR.
        trader, tracker, tape = self._entered(state, monkeypatch, min_win_k=0.6)
        tape.append(SELL_LEVEL, high=SELL_LEVEL)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "sold"
        assert trader.plan["stand_down"]

    def test_zero_switches_it_off_and_the_levels_re_arm_all_day(
        self, state, market_open, monkeypatch
    ):
        trader, tracker, tape = self._entered(state, monkeypatch, min_win_k=0.0)
        tape.append(108.8, high=SELL_LEVEL + 0.05)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "sold"
        assert not trader.plan.get("stand_down")
        assert self._rearms(trader, tracker, tape, state)

    def test_a_losing_trade_stands_down_whatever_closed_it(
        self, state, market_open, monkeypatch
    ):
        """A breakeven runner is the case the stop does not already cover: not a
        loss, but nothing to show for the risk either."""
        trader, tracker, tape = self._entered(
            state, monkeypatch, min_win_k=0.05, stop_k=0.0, negative_momentum_bars=0
        )
        # Flattened at the close, back at the fill.
        tape.append(103.0, high=103.0, low=103.0)
        monkeypatch.setattr(at.market_hours, "seconds_to_close", lambda *a, **k: 60.0)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "sold"
        assert trader.plan["stand_down"]

    def test_a_trade_taken_off_in_two_pieces_is_judged_on_all_of_it(
        self, state, market_open, monkeypatch
    ):
        """A momentum take banks 70% at one price and the runner leaves at
        another. Judging only the piece that closed the position would call a
        good trade bad (a runner sold back at the fill nets nothing) or a bad
        one good."""
        trader, tracker, tape = self._entered(
            state, monkeypatch, min_win_k=0.3, take_fraction=0.7,
            negative_momentum_bars=1, negative_for_bars=1,
        )
        held = tracker.position_for(TICKER)

        # Take 70% at 107.90 (+$4.90 a share) on the first bar down, then the
        # runner back at the 103 fill (+$0). Over the whole position that is
        # 0.7 x 4.90 = $3.43 a share, 0.343 ADR -- above the 0.3 bar, though the
        # closing piece alone made nothing.
        tape.append(108.0)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "hold"
        tape.append(107.9)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "sold"
        runner = tracker.position_for(TICKER)
        assert 0 < runner < held

        tape.append(103.0, low=102.9)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "sold"
        assert tracker.position_for(TICKER) == 0
        assert not trader.plan.get("stand_down")

    def test_a_partial_take_does_not_judge_the_trade_early(
        self, state, market_open, monkeypatch
    ):
        """The position is still open, so there is nothing to judge yet -- and
        blocking an entry while holding one would be meaningless anyway."""
        trader, tracker, tape = self._entered(
            state, monkeypatch, min_win_k=3.0,
            negative_momentum_bars=1, negative_for_bars=1,
        )
        tape.append(104.0)
        trader.run_cycle(DAYRANGE_BUNDLE, state, tracker)
        tape.append(103.9)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "sold"
        assert tracker.position_for(TICKER) > 0
        assert not trader.plan.get("stand_down")

    def test_the_stand_down_is_announced_once_with_the_number(
        self, state, market_open, monkeypatch
    ):
        trader, tracker, tape = self._entered(state, monkeypatch, min_win_k=0.8)
        tape.append(108.8, high=SELL_LEVEL + 0.05)
        trader.run_cycle(DAYRANGE_BUNDLE, state, tracker)
        for _ in range(3):
            tape.append(103.0, low=BUY_LEVEL - 0.01)
            trader.run_cycle(DAYRANGE_BUNDLE, state, tracker)

        lines = [
            e["text"] for e in state.agent_log
            if "no further entries today" in e.get("text", "")
        ]
        assert len(lines) == 1
        # 0.58, not 0.60: the ledger is market-order only and fills at the
        # closing 108.80 of the bar that reached the 109.00 level.
        assert "+0.58 × ADR" in lines[0] and "0.8 × ADR" in lines[0]

    def test_a_stop_still_ends_the_session_with_the_breaker_off(
        self, state, market_open, monkeypatch
    ):
        """The two rules are separate: switching the breaker off must not
        switch off the older refusal to re-enter after a stop."""
        trader, tracker, tape = self._entered(state, monkeypatch, min_win_k=0.0)
        tape.append(STOP_PRICE - 0.1, low=STOP_PRICE - 0.1)   # through the stop
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "sold"
        assert trader.plan["stand_down"] == "stopped out"
        assert not self._rearms(trader, tracker, tape, state)

    def test_a_stop_is_not_announced_twice_when_both_rules_agree(
        self, state, market_open, monkeypatch
    ):
        """A stop is a loss, so the breaker would fire on the same fill."""
        trader, tracker, tape = self._entered(state, monkeypatch, min_win_k=0.2)
        tape.append(STOP_PRICE - 0.1, low=STOP_PRICE - 0.1)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "sold"
        assert trader.plan["stand_down"] == "stopped out"
        lines = [
            e["text"] for e in state.agent_log
            if "no further entries today" in e.get("text", "")
        ]
        assert len(lines) == 1

    def test_a_new_session_starts_armed_again(self, state, market_open, monkeypatch):
        trader, tracker, tape = self._entered(state, monkeypatch, min_win_k=0.8)
        tape.append(108.8, high=SELL_LEVEL + 0.05)
        trader.run_cycle(DAYRANGE_BUNDLE, state, tracker)
        assert trader.plan["stand_down"]

        clock.set_simulated(datetime(2026, 7, 22, 14, 30, tzinfo=timezone.utc))
        tape.append(103.0, low=BUY_LEVEL - 0.01)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "bought"
        assert not trader.plan.get("stand_down")

    def test_the_read_summary_says_why_it_stopped(self, state, market_open, monkeypatch):
        trader, tracker, tape = self._entered(state, monkeypatch, min_win_k=0.8)
        tape.append(108.8, high=SELL_LEVEL + 0.05)
        trader.run_cycle(DAYRANGE_BUNDLE, state, tracker)
        tape.append(103.0)
        trader.run_cycle(DAYRANGE_BUNDLE, state, tracker)

        reads = [e["text"] for e in state.agent_log if " · " in e.get("text", "")]
        assert "no new entries today" in reads[-1]
        assert "+0.58 × ADR" in reads[-1]


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


class TestOpeningWindowTape:
    """Which tape the forecast's opening five minutes come from.

    `or_volume_share` is the one feature that reads minute volume and it was
    fitted on consolidated volume, so an IEX window (under 4% of the tape) is a
    bias in the forecast rather than a rounding error. The rule is: the best
    tape that will serve the 09:30 window, and IEX only when none of them will.
    """

    WANT = 5

    def _window(self, monkeypatch, minutes: int = 5):
        """A Tape whose buffer covers the open (minutes=5) or does not (0).

        With minutes=0 the bars start at 10:30 instead -- an agent launched
        mid-session, whose first five bars are not the session's first five.
        """
        tape = Tape(monkeypatch, minutes=minutes)
        if minutes == 0:
            for _ in range(self.WANT + 1):
                tape.append(103.0)
        return tape

    def _fetches(self, monkeypatch, answers: dict):
        """Record every tape asked for; answer from `answers` (feed -> bars)."""
        asked: list[str] = []

        def alpaca(symbol, timeframe, start, end, key, secret, feed="iex", **kw):
            asked.append(feed)
            return answers.get(feed, [])

        def yahoo(symbol, interval="1m"):
            asked.append("yfinance")
            return answers.get("yfinance", [])

        monkeypatch.setattr(at.agent_mod, "fetch_bars_window", alpaca)
        monkeypatch.setattr(at.historical, "fetch_intraday_bars", yahoo)
        return asked

    def _bars(self, count: int = 5, volume: float = 4.0e5) -> list[dict]:
        """`count` one-minute bars from the 09:30 open, in Alpaca's REST shape."""
        open_et = pd.Timestamp("2026-07-21 09:30", tz="America/New_York")
        return [
            {
                "t": (open_et + pd.Timedelta(minutes=i)).tz_convert("UTC").strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
                "o": 101.0, "h": 101.5, "l": 100.9, "c": 101.0, "v": volume,
            }
            for i in range(count)
        ]

    def test_a_consolidated_buffer_is_used_as_it_stands(self, state, market_open, monkeypatch):
        """The Finnhub stream is the consolidated tape, so the bars already in
        hand are the right ones and nothing is fetched."""
        tape = self._window(monkeypatch)
        asked = self._fetches(monkeypatch, {})
        state.data_source = "finnhub"

        window, source = at.fetch_opening_window(state, tape.frame(), self.WANT)

        assert source == "finnhub" and asked == []
        assert len(window) == self.WANT

    def test_an_iex_buffer_is_replaced_by_a_consolidated_tape(
        self, state, market_open, monkeypatch
    ):
        """The right five minutes on the wrong volume scale. SIP has the same
        five minutes on the scale the ridge was fitted on, so SIP wins even
        though the buffer needs no repair at all."""
        tape = self._window(monkeypatch)
        asked = self._fetches(monkeypatch, {"sip": self._bars()})
        state.data_source = "alpaca"
        state.feed = "iex"

        window, source = at.fetch_opening_window(state, tape.frame(), self.WANT)

        assert source == "sip" and asked == ["sip"]
        assert float(window["volume"].iloc[0]) == pytest.approx(4.0e5)

    def test_iex_is_asked_last_and_only_after_every_consolidated_tape(
        self, state, market_open, monkeypatch
    ):
        tape = self._window(monkeypatch, minutes=0)
        asked = self._fetches(monkeypatch, {"iex": self._bars(volume=1.5e4)})
        state.data_source = "alpaca"
        state.feed = "iex"

        window, source = at.fetch_opening_window(state, tape.frame(), self.WANT)

        # sip_delayed never reaches Alpaca: mid-session at 10:30, the 09:30
        # window is well outside the trailing 15 minutes a free plan refuses,
        # so it is asked -- as plain "sip", which is the feed Alpaca knows.
        assert asked == ["sip", "sip", "yfinance", "iex"]
        assert source == "iex" and len(window) == self.WANT

    def test_an_iex_buffer_is_kept_when_no_consolidated_tape_answers(
        self, state, market_open, monkeypatch
    ):
        """Before 9:50 on a free key every consolidated source is 15 minutes
        behind. A biased forecast beats no forecast; the caveat says which."""
        tape = self._window(monkeypatch)
        asked = self._fetches(monkeypatch, {})
        state.data_source = "alpaca"
        state.feed = "iex"

        window, source = at.fetch_opening_window(state, tape.frame(), self.WANT)

        assert source == "iex" and len(window) == self.WANT
        # A REST IEX window would hand back the bars already in the buffer, so
        # the round trip is not spent.
        assert "iex" not in asked

    def test_a_short_answer_is_not_an_answer(self, state, market_open, monkeypatch):
        """Three of the five minutes is not a cheaper forecast, so the tape that
        sent them is passed over rather than trusted."""
        tape = self._window(monkeypatch, minutes=0)
        asked = self._fetches(
            monkeypatch, {"sip": self._bars(count=3), "yfinance": self._bars()}
        )
        state.data_source = "alpaca"
        state.feed = "iex"

        _, source = at.fetch_opening_window(state, tape.frame(), self.WANT)

        assert source == "yfinance" and "iex" not in asked

    def test_delayed_sip_declines_a_window_it_cannot_serve(
        self, state, market_open, monkeypatch
    ):
        """At 9:35 the 09:30 window is inside the trailing 15 minutes a
        free/basic plan refuses, so delayed SIP never reaches Alpaca -- the
        request is not spent on a certain 403. The tapes behind it are still
        tried, in order, and real-time SIP is one of them."""
        tape = self._window(monkeypatch, minutes=0)
        asked = self._fetches(monkeypatch, {})
        clock.set_simulated(datetime(2026, 7, 21, 13, 35, tzinfo=timezone.utc))
        state.history_feed_resolved = "sip_delayed"

        with pytest.raises(ValueError, match="09:30 window"):
            at.fetch_opening_window(state, tape.frame(), self.WANT)

        assert asked == ["sip", "yfinance", "iex"]

    def test_a_replay_keeps_its_datasets_tape_and_does_not_shop(
        self, state, market_open, monkeypatch
    ):
        """In simulation every source is patched back to the same stored bars,
        so asking SIP for an IEX dataset's window would relabel it -- and lose
        the caveat that dataset had earned."""
        tape = self._window(monkeypatch)
        asked = self._fetches(monkeypatch, {"sip": self._bars()})
        state.bar_tape_override = "iex"

        window, source = at.fetch_opening_window(state, tape.frame(), self.WANT)

        assert source == "iex" and asked == []
        assert len(window) == self.WANT

    def test_a_replay_recovering_the_window_still_names_its_own_tape(
        self, state, market_open, monkeypatch
    ):
        """A mid-session start inside a replay does re-fetch -- out of the same
        dataset. The bars are the replay's whichever source name answered."""
        tape = self._window(monkeypatch, minutes=0)
        self._fetches(monkeypatch, {"sip": self._bars()})
        state.bar_tape_override = "iex"

        _, source = at.fetch_opening_window(state, tape.frame(), self.WANT)

        assert source == "iex"

    def test_the_caveat_names_the_tape_the_forecast_was_built_on(
        self, state, market_open, monkeypatch
    ):
        """Not `state.feed`, which the Finnhub default leaves at "iex" while the
        buffer is consolidated end to end -- a caveat on every clean forecast."""
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(103.0))
        tape = Tape(monkeypatch)
        self._fetches(monkeypatch, {})
        state.data_source = "finnhub"
        state.feed = "iex"
        tape.append(103.0)

        at.DayRangeTrader(dayrange_config()).run_cycle(DAYRANGE_BUNDLE, state, tracker)

        assert not [e for e in state.agent_log if "Forecast caveat" in e.get("text", "")]

    def test_an_iex_forecast_still_says_so(self, state, market_open, monkeypatch):
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(103.0))
        tape = Tape(monkeypatch)
        self._fetches(monkeypatch, {})
        state.data_source = "alpaca"
        state.feed = "iex"
        tape.append(103.0)

        at.DayRangeTrader(dayrange_config()).run_cycle(DAYRANGE_BUNDLE, state, tracker)

        caveats = [e for e in state.agent_log if "Forecast caveat" in e.get("text", "")]
        assert len(caveats) == 1 and "4% of consolidated" in caveats[0]["text"]


class TestStrategySelection:
    def test_the_model_chooses_the_state_machine(self):
        assert isinstance(
            at.build_trader(dayrange_config(), DAYRANGE_BUNDLE), at.DayRangeTrader
        )

    def test_the_levels_are_the_signature(self):
        base = config_signature(dayrange_config(stop_gain_fraction=0.0, negative_momentum_bars=0))
        assert base == "dayrange_AAPL(buy=H-0.75A,sell=H-0.1A,size=95%)"
        assert base != config_signature(
            dayrange_config(buy_k=0.8, stop_gain_fraction=0.0, negative_momentum_bars=0)
        )
        assert base != config_signature(
            dayrange_config(sell_k=0.2, stop_gain_fraction=0.0, negative_momentum_bars=0)
        )

    def test_the_exit_is_in_the_signature_only_while_switched_on(self):
        on = config_signature(dayrange_config())
        assert on == (
            "dayrange_AAPL(buy=H-0.75A,sell=H-0.1A,size=95%,"
            "stop=E-0.5G,take=70%@neg15b/5b,runner>=0.3A)"
        )
        # A legacy record's take signs in its own form, as it was filed.
        assert config_signature(
            dayrange_config(negative_momentum_bars=0, momentum_fade_bars=15)
        ) == on.replace("@neg15b/5b", "@fade15b")
        assert config_signature(
            dayrange_config(negative_momentum_bars=0, momentum_drop=1.0)
        ) == on.replace("@neg15b/5b", "@mom-1")
        for field, value in (
            ("stop_gain_fraction", 0.3), ("negative_momentum_bars", 20),
            ("negative_for_bars", 3), ("take_fraction", 0.5), ("hold_min_gain_k", 0.5),
        ):
            assert config_signature(dayrange_config(**{field: value})) != on, field
        # With the take off its knobs trade nothing, so they sign nothing.
        off = dayrange_config(negative_momentum_bars=0)
        assert config_signature(off) == config_signature(
            replace(off, negative_for_bars=9, take_fraction=0.5, hold_min_gain_k=0.9)
        )

    def test_only_one_momentum_take_may_be_set(self):
        """Today's streak and the two legacy takes are three ways of writing one
        rule; a config naming two does not say which it means."""
        for legacy in ({"momentum_fade_bars": 15}, {"momentum_drop": 1.0}):
            with pytest.raises(ValueError, match="only one may be set"):
                dayrange_config(negative_momentum_bars=15, **legacy)

    def test_the_streak_is_whole_bars_and_at_least_one(self):
        with pytest.raises(ValueError, match="negative_for_bars"):
            dayrange_config(negative_for_bars=0)
        with pytest.raises(ValueError, match="negative_for_bars"):
            dayrange_config(negative_for_bars=2.5)
        # Off, the streak is never read, so it is not held to anything.
        dayrange_config(negative_momentum_bars=0, negative_for_bars=0)

    def test_the_intraday_update_is_in_the_signature_only_while_switched_on(self):
        off = config_signature(dayrange_config(stop_gain_fraction=0.0, negative_momentum_bars=0))
        assert off == "dayrange_AAPL(buy=H-0.75A,sell=H-0.1A,size=95%)"
        for policy in ("extreme", "brownian"):
            signed = config_signature(
                dayrange_config(stop_gain_fraction=0.0, negative_momentum_bars=0, breach_update=policy)
            )
            assert signed == off[:-1] + f",breach={policy})"

    def test_the_level_source_is_in_the_signature_only_when_it_is_not_the_high(self):
        """It sits beside the two distances rather than at the end: "H" in
        `buy=H-0.75A` is whatever the source says it is."""
        flat = config_signature(dayrange_config(stop_gain_fraction=0.0, negative_momentum_bars=0))
        assert flat == "dayrange_AAPL(buy=H-0.75A,sell=H-0.1A,size=95%)"
        assert config_signature(
            dayrange_config(stop_gain_fraction=0.0, negative_momentum_bars=0, level_source="intraday")
        ) == "dayrange_AAPL(buy=H-0.75A,sell=H-0.1A,levels=intraday,size=95%)"

    def test_the_intraday_update_defaults_to_brownian(self):
        """`dayrange_config` pins it off; the app's own default does not. The
        Brownian extension since 2026-09-29 ("shift" before)."""
        assert AppleTraderConfig().breach_update == "brownian"

    def test_an_unknown_update_policy_is_refused(self):
        """Read as "off" once a record carries it, but refused while a config is
        being built -- the earliest place a typo can be reported is the best one."""
        with pytest.raises(ValueError, match="breach_update"):
            dayrange_config(breach_update="mean_reversion")

    def test_exit_distances_cannot_be_negative_and_the_take_is_a_share(self):
        for field in (
            "stop_gain_fraction", "momentum_drop", "momentum_fade_bars",
            "negative_momentum_bars", "hold_min_gain_k",
        ):
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
    """Each model and instrument starts from its own tuned pair, not the notebook's."""

    def test_each_tuned_pair_starts_from_its_own_levels(self):
        for (model_key, ticker), (buy_k, sell_k) in at.APPLE_TRADER_TUNED_LEVELS.items():
            config = AppleTraderConfig(model_key=model_key, ticker=ticker.lower())
            assert (config.buy_k, config.sell_k) == (buy_k, sell_k)

    def test_every_tuned_pair_is_a_model_the_instrument_is_wired_up_for(self):
        for model_key, ticker in at.APPLE_TRADER_TUNED_LEVELS:
            assert ticker in apple_models.MODELS[model_key].tickers

    def test_the_same_instrument_differs_by_model(self):
        """The same distance is a different price under a different forecast."""
        assert at.dayrange_levels("AAPL", "dayrange") != at.dayrange_levels("AAPL", "highlow")

    def test_a_pair_never_tuned_falls_back_to_the_instruments_sweep(self):
        untuned = apple_models.DAYRANGE_INTRADAY_KEY
        for ticker in apple_models.MODELS[untuned].tickers:
            assert (untuned, ticker) not in at.APPLE_TRADER_TUNED_LEVELS
            assert at.dayrange_levels(ticker, untuned) == at.APPLE_TRADER_DAYRANGE_LEVELS[ticker]

    def test_every_ticker_the_model_is_wired_up_for_has_a_pair(self):
        assert set(apple_models.DAYRANGE_TICKERS) <= set(at.APPLE_TRADER_DAYRANGE_LEVELS)

    def test_a_symbol_never_swept_falls_back_to_the_notebook_pair(self):
        assert at.dayrange_levels("ZZZZ") == (0.75, 0.10)
        config = AppleTraderConfig(ticker="ZZZZ")
        assert (config.buy_k, config.sell_k) == (0.75, 0.10)

    def test_a_level_given_explicitly_wins_and_the_other_keeps_its_default(self):
        _, googl_sell = at.dayrange_levels("GOOGL", "dayrange")
        config = AppleTraderConfig(model_key="dayrange", ticker="GOOGL", buy_k=0.9)
        assert (config.buy_k, config.sell_k) == (0.9, googl_sell)

    def test_the_default_pair_signs_the_run(self):
        buy_k, sell_k = at.dayrange_levels("AAPL", "dayrange")
        assert config_signature(AppleTraderConfig(model_key="dayrange")).startswith(
            f"dayrange_AAPL(buy=H-{buy_k:g}R,sell=H-{sell_k:g}R,size=95%"
        )


class TestStopDefaults:
    """The stop starts from the one tuned with the pair's levels, since 2026-10-04."""

    def test_each_tuned_pair_starts_from_its_own_stop(self):
        for (model_key, ticker), stop in at.APPLE_TRADER_TUNED_STOP.items():
            config = AppleTraderConfig(model_key=model_key, ticker=ticker.lower())
            assert config.stop_gain_fraction == stop, (model_key, ticker)

    def test_a_stop_is_tuned_only_with_the_levels_it_is_a_share_of(self):
        assert set(at.APPLE_TRADER_TUNED_STOP) <= set(at.APPLE_TRADER_TUNED_LEVELS)

    def test_a_pair_never_tuned_starts_from_the_shared_default(self):
        assert ("dayrange", "AAPL") not in at.APPLE_TRADER_TUNED_STOP
        assert AppleTraderConfig(model_key="dayrange", ticker="AAPL").stop_gain_fraction == (
            at.APPLE_TRADER_STOP_GAIN_FRACTION
        )
        assert at.stop_for("ZZZZ", "highlow") == at.APPLE_TRADER_STOP_GAIN_FRACTION

    def test_a_stop_given_explicitly_wins(self):
        assert AppleTraderConfig(
            model_key="highlow", ticker="AAPL", stop_gain_fraction=0.25
        ).stop_gain_fraction == 0.25
        # 0 is the stop switched off, not a missing value.
        assert not AppleTraderConfig(
            model_key="highlow", ticker="AAPL", stop_gain_fraction=0.0
        ).has_stop

    def test_the_tuned_stop_signs_the_run(self):
        config = AppleTraderConfig(model_key="highlow", ticker="AAPL")
        assert f",stop=E-{at.stop_for('AAPL', 'highlow'):g}G" in config_signature(config)


class TestRangeContainment:
    """The forecast is never left arguing with the tape.

    `apply_open_constraint` already clips the 9:35 prediction to contain the
    opening five minutes. This is the same rule for the rest of the session,
    and unlike the breach policies it is arithmetic: it never leads the tape,
    it only declines to keep a number the tape has passed.
    """

    def _trader(self, policy="off", **kwargs):
        return at.DayRangeTrader(
            dayrange_config(breach_update=policy, contain_range=True, **kwargs)
        )

    def test_off_no_longer_keeps_a_high_the_day_traded_through(
        self, state, market_open, monkeypatch
    ):
        """The case this rule exists for. Without it the 9:35 high stands all
        day under "off", so every level stays measured from a price the session
        has already been past."""
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(112.0))
        tape = Tape(monkeypatch)
        tape.append(111.0, high=112.0)
        trader = self._trader("off")
        trader.run_cycle(DAYRANGE_BUNDLE, state, tracker)

        assert trader.plan["pred_high"] == pytest.approx(112.0)
        assert trader.plan["sell_level"] == pytest.approx(112.0 - 0.10 * 10.0)

    def test_a_low_the_day_traded_through_is_pulled_down_too(
        self, state, market_open, monkeypatch
    ):
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(94.0))
        tape = Tape(monkeypatch)
        tape.append(94.5, low=94.0)
        trader = self._trader("off")
        trader.run_cycle(DAYRANGE_BUNDLE, state, tracker)

        assert trader.plan["pred_low"] == pytest.approx(94.0)
        assert trader.plan["pred_high"] == pytest.approx(110.0)   # never breached

    def test_it_only_ever_widens(self, state, market_open, monkeypatch):
        """A session trading quietly inside its forecast leaves it alone: the
        rule is a floor on the range, not a description of it."""
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(103.0))
        tape = Tape(monkeypatch)
        tape.append(103.0, high=104.0, low=102.0)
        trader = self._trader("off")
        trader.run_cycle(DAYRANGE_BUNDLE, state, tracker)

        assert trader.plan["pred_high"] == pytest.approx(FORECAST["pred_high"])
        assert trader.plan["pred_low"] == pytest.approx(FORECAST["pred_low"])

    def test_it_makes_off_agree_with_extreme(self, state, market_open, monkeypatch):
        """The consequence of applying it under every policy, stated outright
        so that it is a decision on the record rather than a surprise: the only
        policy still distinguishable with containment on is `brownian`."""
        def levels(policy):
            tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(112.0))
            tape = Tape(monkeypatch)
            tape.append(111.0, high=112.0)
            trader = self._trader(policy)
            trader.run_cycle(DAYRANGE_BUNDLE, state, tracker)
            return trader.plan["buy_level"], trader.plan["sell_level"]

        assert levels("off") == levels("extreme")
        assert levels("brownian") != levels("extreme")

    def test_the_model_helper_never_crosses_the_two_sides(self):
        """Whatever it is handed, it widens -- so it cannot put the low above
        the high, which would be a range no level could be read from."""
        dayrange = pytest.importorskip("agent_stonks.dayrange_model")
        high, low = dayrange.contain_session(
            110.0, 95.0, session_high=112.0, session_low=94.0
        )
        assert (high, low) == (112.0, 94.0)
        high, low = dayrange.contain_session(
            110.0, 95.0, session_high=None, session_low=None
        )
        assert (high, low) == (110.0, 95.0)

    def test_a_record_written_before_it_replays_without_it(self):
        from simlab.rule_agents import _apple_from_record

        old = _apple_from_record({"model_key": "dayrange", "buy_k": 0.75})
        assert old.contain_range is False
        assert "contain" not in config_signature(old)
        # Off by default since 2026-10-04, so the one that signs it says so.
        assert "contain" not in config_signature(AppleTraderConfig(model_key="dayrange"))
        assert "contain" in config_signature(
            AppleTraderConfig(model_key="dayrange", contain_range=True)
        )


class TestKeepWidth:
    """A breach keeps the range at the level unit's width (2026-09-25).

    FORECAST is $95 - $110 ($15) with a $10 ADR, so which of the two widths a
    breach keeps shows in where the side that followed ends up.
    """

    def _run(self, policy, state, tape, **kwargs):
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(94.0))
        trader = at.DayRangeTrader(
            dayrange_config(breach_update=policy, keep_width=True, **kwargs)
        )
        trader.run_cycle(DAYRANGE_BUNDLE, state, tracker)
        return trader

    def test_under_adr_the_high_follows_a_low_breach_one_adr_above_it(
        self, state, market_open, monkeypatch
    ):
        tape = Tape(monkeypatch)
        tape.append(94.5, low=94.0)
        trader = self._run("shift", state, tape)

        assert trader.plan["pred_low"] == pytest.approx(94.0)
        assert trader.plan["pred_high"] == pytest.approx(104.0)
        assert trader.plan["buy_level"] == pytest.approx(104.0 - 0.75 * 10)

    def test_under_the_predicted_range_it_keeps_the_935_width(
        self, state, market_open, monkeypatch
    ):
        tape = Tape(monkeypatch)
        tape.append(94.5, low=94.0)
        trader = self._run("brownian", state, tape, level_unit=UNIT_PRED_RANGE)

        reach = at._dayrange().brownian_reach(10.0, 325.0)
        assert trader.plan["pred_low"] == pytest.approx(94.0 - reach)
        assert trader.plan["pred_high"] - trader.plan["pred_low"] == pytest.approx(15.0)
        assert at.level_unit(trader.config, trader.plan) == pytest.approx(15.0)

    def test_a_record_written_before_it_replays_without_it(self):
        from simlab.rule_agents import _apple_from_record

        old = _apple_from_record(
            {"model_key": "dayrange", "buy_k": 0.75, "breach_update": "brownian"}
        )
        assert old.keep_width is False
        assert "keep_width" not in config_signature(old)
        new = AppleTraderConfig(model_key="dayrange", breach_update="brownian")
        assert "keep_width" in config_signature(new)
        # Nothing to keep when nothing moves.
        held = AppleTraderConfig(model_key="dayrange", breach_update="off")
        assert "keep_width" not in config_signature(held)


class TestBreachExit:
    """A bar through the predicted high closes an open position.

    The levels are a bet that the day tops out near the predicted high. A bar
    that trades through it has settled that bet in the position's favour, and
    at a better price than the sell level was offering.
    """

    def _trader(self, policy="brownian", **kwargs):
        return at.DayRangeTrader(
            dayrange_config(breach_update=policy, breach_exit=True, **kwargs)
        )

    def _entered(self, policy, state, monkeypatch, **kwargs):
        broker = FakeBroker(103.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tape = Tape(monkeypatch, broker)
        trader = self._trader(policy, **kwargs)
        tape.append(103.0, low=BUY_LEVEL - 0.01)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "bought"
        tape.append(111.0, high=112.0)
        return trader, trader.run_cycle(DAYRANGE_BUNDLE, state, tracker), tracker

    def test_brownian_no_longer_rides_through_the_breach(
        self, state, market_open, monkeypatch
    ):
        """The defect this rule fixes. The breach moved the forecast, which
        carried the sell level past the bar that breached, and the position was
        held on against a target that had stepped out of its own way."""
        trader, outcome, tracker = self._entered("brownian", state, monkeypatch)
        assert outcome == "sold"
        assert tracker.position_for(TICKER) == 0
        # The target did move out of the way -- that is what is being overruled.
        assert trader.plan["sell_level"] > 112.0

    def test_it_is_measured_against_the_high_the_bar_opened_under(
        self, state, market_open, monkeypatch
    ):
        """Not against the high the bar itself just pushed up. Testing a bar
        against a level it moved is lookahead whatever the policy is."""
        trader, outcome, _ = self._entered("brownian", state, monkeypatch)
        assert outcome == "sold"
        assert trader.plan["high_at_bar"] == pytest.approx(FORECAST["pred_high"])
        assert trader.plan["pred_high"] > 112.0        # moved, but not used

    def test_a_breach_that_reaches_the_target_is_logged_as_the_target(
        self, state, market_open, monkeypatch
    ):
        """Under "off" and "extreme" the sell level sits `sell_k` under the high
        the bar traded through, so it was always a target exit and should still
        read as one."""
        for policy in ("off", "extreme"):
            _, outcome, tracker = self._entered(policy, state, monkeypatch)
            assert outcome == "sold", policy
            assert "Target" in tracker.snapshot()["decisions"][-1].reasoning, policy

    def test_the_breach_exit_names_itself_and_the_high_it_cleared(
        self, state, market_open, monkeypatch
    ):
        _, outcome, tracker = self._entered("brownian", state, monkeypatch)
        reasoning = tracker.snapshot()["decisions"][-1].reasoning
        assert "Breach exit" in reasoning
        assert "110.00" in reasoning and "112.00" in reasoning

    def test_a_day_inside_the_forecast_is_not_an_exit(
        self, state, market_open, monkeypatch
    ):
        """Touching the predicted high is not trading through it, and the bar
        that merely reaches it is the target's business."""
        broker = FakeBroker(103.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tape = Tape(monkeypatch, broker)
        trader = self._trader("brownian")
        tape.append(103.0, low=BUY_LEVEL - 0.01)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "bought"
        tape.append(105.0, high=106.0)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "hold"
        assert tracker.position_for(TICKER) > 0

    def test_it_is_a_completed_trade_the_circuit_breaker_can_judge(
        self, state, market_open, monkeypatch
    ):
        """Unlike a stop it does not stand the session down by itself, so a
        breach exit that paid leaves the levels armed."""
        trader, outcome, _ = self._entered(
            "brownian", state, monkeypatch, min_win_k=0.0
        )
        assert outcome == "sold"
        # A stop stands the session down by itself; a breach exit banked a
        # profit, so the levels stay armed for the rest of the day.
        assert not trader.plan.get("stand_down")

    def test_switched_off_it_rides_through_as_it_used_to(
        self, state, market_open, monkeypatch
    ):
        broker = FakeBroker(103.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tape = Tape(monkeypatch, broker)
        trader = at.DayRangeTrader(dayrange_config(breach_update="brownian"))
        tape.append(103.0, low=BUY_LEVEL - 0.01)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "bought"
        tape.append(111.0, high=112.0)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "hold"
        assert tracker.position_for(TICKER) > 0

    def test_a_record_written_before_it_replays_without_it(self):
        from simlab.rule_agents import _apple_from_record

        old = _apple_from_record({"model_key": "dayrange", "buy_k": 0.75})
        assert old.breach_exit is False
        assert "breach_exit" not in config_signature(old)
        assert "breach_exit" in config_signature(AppleTraderConfig(model_key="dayrange"))


class TestPredictedRangeUnit:
    """The two distances counted in the model's own predicted range.

    The fixture separates the two units cleanly: the ADR is $10 and the
    predicted range is $15 (110 - 95), so every level, stop and threshold lands
    somewhere the other unit could not have put it.
    """

    def _config(self, **kwargs):
        kwargs.setdefault("level_unit", UNIT_PRED_RANGE)
        return dayrange_config(**kwargs)

    def _trader(self, **kwargs):
        return at.DayRangeTrader(self._config(**kwargs))

    PRED_RANGE = 15.0                     # 110 - 95
    BUY = 110.0 - 0.75 * PRED_RANGE       # 98.75, vs 102.50 under the ADR
    SELL = 110.0 - 0.10 * PRED_RANGE      # 108.50, vs 109.00 under the ADR

    def test_the_levels_are_the_two_distances_in_predicted_ranges(
        self, state, market_open, monkeypatch
    ):
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(103.0))
        Tape(monkeypatch)
        trader = self._trader()
        trader.run_cycle(DAYRANGE_BUNDLE, state, tracker)

        assert trader.plan["buy_level"] == pytest.approx(self.BUY)
        assert trader.plan["sell_level"] == pytest.approx(self.SELL)
        # And not where the ADR would have put them, which is the whole point.
        assert trader.plan["buy_level"] != pytest.approx(BUY_LEVEL)

    def test_the_unit_is_the_forecasts_width_not_the_trailing_average(self):
        """A forecast calling an unusually wide day gets unusually wide distances.

        Under the ADR the same `adr14_abs` would have produced the same two
        distances for both of these, which is the thing this unit exists to
        stop.
        """
        config = self._config()
        narrow = {**FORECAST, "pred_high": 110.0, "pred_low": 108.0}
        wide = {**FORECAST, "pred_high": 110.0, "pred_low": 80.0}
        assert at.level_unit(config, narrow) == pytest.approx(2.0)
        assert at.level_unit(config, wide) == pytest.approx(30.0)
        # The ADR unit ignores both and reads the trailing average.
        adr_config = self._config(level_unit=UNIT_ADR)
        assert at.level_unit(adr_config, narrow) == pytest.approx(10.0)
        assert at.level_unit(adr_config, wide) == pytest.approx(10.0)

    def test_a_forecast_with_no_width_falls_back_to_the_adr(self):
        """Rather than collapsing both levels onto the reference, which would
        trade as a single level and say nothing about why."""
        config = self._config()
        assert at.level_unit(config, {**FORECAST, "pred_low": 110.0}) == 10.0
        assert at.level_unit(config, {**FORECAST, "pred_low": 120.0}) == 10.0

    def test_a_breach_widens_the_unit_so_the_levels_spread_apart(
        self, state, market_open, monkeypatch
    ):
        """The consequence the ADR unit does not have.

        Under "adr" a breach shifts both levels by the same dollar and the gap
        between them is the same all day. Here the breach widens the forecast,
        which widens the unit, so the two levels move apart -- the day has been
        shown to be bigger than predicted and the trade plays for more of it.
        """
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(112.0))
        tape = Tape(monkeypatch)
        tape.append(111.5, high=112.0)
        trader = self._trader(breach_update="extreme")
        trader.run_cycle(DAYRANGE_BUNDLE, state, tracker)

        # 112 - 95 = 17 wide now, off a high of 112.
        assert trader.plan["pred_high"] == pytest.approx(112.0)
        gap = trader.plan["sell_level"] - trader.plan["buy_level"]
        assert gap == pytest.approx(0.65 * 17.0)
        assert gap > 0.65 * self.PRED_RANGE

        # Where the ADR unit holds the gap fixed through the same breach.
        adr_tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(112.0))
        adr_tape = Tape(monkeypatch)
        adr_tape.append(111.5, high=112.0)
        adr = at.DayRangeTrader(
            dayrange_config(breach_update="extreme", level_unit=UNIT_ADR)
        )
        adr.run_cycle(DAYRANGE_BUNDLE, state, adr_tracker)
        assert adr.plan["sell_level"] - adr.plan["buy_level"] == pytest.approx(6.5)

    def test_the_stop_is_a_share_of_the_gain_in_the_same_unit(self):
        """Not of a gain measured in ADRs, which would not be the gain."""
        config = self._config(stop_gain_fraction=0.5)
        plan = dict(FORECAST)
        # 0.65 of a 15-dollar range is 9.75; half of that is 4.875.
        assert at.stop_distance(
            config, at.stop_unit(config, plan)
        ) == pytest.approx(0.5 * 0.65 * 15.0)

    def test_the_legacy_adr_stop_stays_in_adrs_whatever_the_unit_says(self):
        """`stop_k` is a stored record's own number and means ADRs by
        construction; re-reading it against the predicted range would replay a
        different stop under the original's signature."""
        config = self._config(stop_k=0.2, stop_gain_fraction=0.0)
        plan = dict(FORECAST)
        assert at.stop_unit(config, plan) == pytest.approx(10.0)
        assert at.stop_distance(config, at.stop_unit(config, plan)) == pytest.approx(2.0)

    def test_an_open_positions_stop_does_not_loosen_when_the_forecast_widens(
        self, state, market_open, monkeypatch
    ):
        """The one direction a stop must never move on its own.

        The unit is a function of a forecast the breach policy moves, so a stop
        re-read each bar would step further from the fill every time the day
        made a new high. It is fixed in dollars at the fill instead.
        """
        broker = FakeBroker(98.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tape = Tape(monkeypatch, broker)
        trader = self._trader(breach_update="extreme", stop_gain_fraction=0.5)

        tape.append(98.0, low=98.0)          # through the 98.75 buy level
        trader.run_cycle(DAYRANGE_BUNDLE, state, tracker)
        assert tracker.position_for(TICKER) > 0
        risk = trader._risk()
        assert risk == pytest.approx(0.5 * 0.65 * 15.0)   # 4.875

        # A breach of the *low* side, which widens the range without going
        # anywhere near the 108.50 sell level. 94.00 clears both the frozen
        # stop (98 - 4.875 = 93.125) and the looser one a re-read would give,
        # so the only thing this asserts is which of the two is being used.
        tape.append(96.0, low=94.0)
        trader.run_cycle(DAYRANGE_BUNDLE, state, tracker)
        assert tracker.position_for(TICKER) > 0
        assert trader.plan["pred_low"] == pytest.approx(94.0)   # 16 wide now
        assert trader._risk() == pytest.approx(risk)
        # The re-read this is guarding against, for contrast: further from the
        # fill than the stop the trade was entered with.
        assert at.stop_distance(
            trader.config, at.stop_unit(trader.config, trader.plan)
        ) == pytest.approx(0.5 * 0.65 * 16.0)

    def test_a_breach_of_the_low_alone_now_moves_the_levels(
        self, state, market_open, monkeypatch
    ):
        """Which it does not do under the ADR unit.

        There the levels are built from the predicted high and a trailing
        average, so the low is not an input to either and a breach of it moves
        the forecast and nothing else. Here the low is half the unit.
        """
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(96.0))
        tape = Tape(monkeypatch)
        tape.append(96.0, low=94.0)
        trader = self._trader(breach_update="extreme")
        trader.run_cycle(DAYRANGE_BUNDLE, state, tracker)

        assert trader.plan["pred_high"] == pytest.approx(110.0)   # never breached
        assert trader.plan["pred_low"] == pytest.approx(94.0)
        assert trader.plan["buy_level"] == pytest.approx(110.0 - 0.75 * 16.0)
        assert trader.plan["sell_level"] == pytest.approx(110.0 - 0.10 * 16.0)

        adr_tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(96.0))
        adr_tape = Tape(monkeypatch)
        adr_tape.append(96.0, low=94.0)
        adr = at.DayRangeTrader(
            dayrange_config(breach_update="extreme", level_unit=UNIT_ADR)
        )
        adr.run_cycle(DAYRANGE_BUNDLE, state, adr_tracker)
        assert adr.plan["pred_low"] == pytest.approx(94.0)
        assert adr.plan["buy_level"] == pytest.approx(BUY_LEVEL)   # unmoved
        assert adr.plan["sell_level"] == pytest.approx(SELL_LEVEL)

    def test_the_unit_is_in_the_signature_on_every_k(self):
        """Two units are two strategies at the same numbers, so they must not
        share a row in Results."""
        adr = config_signature(dayrange_config(level_unit=UNIT_ADR, min_win_k=0.2))
        pred = config_signature(self._config(min_win_k=0.2))
        assert "buy=H-0.75A" in adr and "min_win=0.2A" in adr
        assert "buy=H-0.75R" in pred and "min_win=0.2R" in pred
        assert adr != pred

    def test_the_log_lines_name_the_unit_they_are_counted_in(
        self, state, market_open, monkeypatch
    ):
        broker = FakeBroker(98.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tape = Tape(monkeypatch, broker)
        tape.append(98.0, low=98.0)
        trader = self._trader()
        trader.run_cycle(DAYRANGE_BUNDLE, state, tracker)

        entry = tracker.snapshot()["decisions"][-1].reasoning
        assert "buy level" in entry
        assert "predicted range" in entry
        assert "average daily" not in entry
        # And the ADR unit still says "ADR", on the same line.
        adr_tracker = DecisionTracker(
            starting_cash=10_000.0, broker=FakeBroker(103.0)
        )
        adr_tape = Tape(monkeypatch, FakeBroker(103.0))
        adr_tape.append(102.0, low=102.0)
        adr = at.DayRangeTrader(dayrange_config(level_unit=UNIT_ADR))
        adr.run_cycle(DAYRANGE_BUNDLE, state, adr_tracker)
        assert "ADR" in adr_tracker.snapshot()["decisions"][-1].reasoning


class TestInstrument:
    def test_the_symbols_on_offer_are_the_ones_a_model_covers(self):
        # HighLow_5m and HighLow2_5m have been run on AAPL and INTC, not GOOGL;
        # HighLow_3m on AAPL alone.
        assert apple_models.keys_for(TICKER) == [
            "dayrange", "dayrange_intraday", "highlow", "highlow2", "highlow3m",
        ]
        assert apple_models.keys_for("INTC") == [
            "dayrange", "dayrange_intraday", "highlow", "highlow2",
        ]
        for symbol in (NON_AAPL,):
            assert apple_models.keys_for(symbol) == [
                "dayrange", "dayrange_intraday",
            ]
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


class TestStopOutEndsTheRun:
    def test_the_loop_stops_the_agent_after_a_stop_out(self, state, monkeypatch):
        cycles: list = []

        class StoppedOut:
            halt = None

            def run_cycle(self, bundle, state, tracker):
                cycles.append(1)
                self.halt = "stopped out"
                return "sold"

        monkeypatch.setattr(at.apple_models, "load", lambda key, ticker=None: DAYRANGE_BUNDLE)
        monkeypatch.setattr(at, "config_error", lambda config, bundle=None: None)
        monkeypatch.setattr(at, "build_trader", lambda config, bundle=None: StoppedOut())
        state.agent_running = True
        stop_event = threading.Event()
        loop = threading.Thread(
            target=at._apple_trader_loop,
            args=(state, tracker_for_loop(), AppleTraderConfig(), 60, stop_event),
            daemon=True,
        )
        loop.start()
        loop.join(timeout=10)

        assert not loop.is_alive()
        assert stop_event.is_set()
        assert cycles == [1]
        assert state.agent_running is False
        assert any("Press ▶ Start Agent" in e.get("text", "") for e in state.agent_log)


def tracker_for_loop() -> DecisionTracker:
    return DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(100.0))


class TestStatusLineActivity:
    """The agent's own segment of the Agent tab's status line: a dot and a
    phrase, set from each cycle's outcome (`BaseTrader.activity`)."""

    def _setup(self, monkeypatch, price=103.0, **tape):
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(price))
        return at.DayRangeTrader(dayrange_config()), tracker, Tape(monkeypatch, **tape)

    def _activity(self, trader, state, tracker):
        return trader.activity(trader.run_cycle(DAYRANGE_BUNDLE, state, tracker), tracker)

    def test_before_the_open_it_waits_for_the_session(self, state, monkeypatch):
        clock.set_simulated(datetime(2026, 7, 21, 12, 0, tzinfo=timezone.utc))  # 8:00 ET
        try:
            trader, tracker, _ = self._setup(monkeypatch)
            assert self._activity(trader, state, tracker) == (
                "🟠", "Agent waiting for session start"
            )
        finally:
            clock.clear()

    def test_inside_the_opening_window_it_waits_for_the_forecast(
        self, state, market_open, monkeypatch
    ):
        trader, tracker, tape = self._setup(monkeypatch, minutes=0)
        tape.append(104.0)
        assert self._activity(trader, state, tracker) == (
            "🟠", "Agent waiting for the opening forecast"
        )

    def test_flat_then_holding(self, state, market_open, monkeypatch):
        trader, tracker, tape = self._setup(monkeypatch)
        tape.append(104.0)
        assert self._activity(trader, state, tracker) == (
            "🟢", "Agent watching for an entry"
        )
        tape.append(103.0, low=BUY_LEVEL - 0.01)
        assert self._activity(trader, state, tracker) == ("🟢", f"Agent holding {TICKER}")

    def test_a_stand_down_says_so(self, state, market_open, monkeypatch):
        trader, tracker, tape = self._setup(monkeypatch)
        tape.append(104.0)
        trader.run_cycle(DAYRANGE_BUNDLE, state, tracker)
        trader.plan["stand_down"] = "min win"
        tape.append(104.0)
        assert self._activity(trader, state, tracker) == (
            "🟠", "Agent stood down for the day"
        )

    def test_a_day_that_cannot_be_forecast_is_red(self, state, market_open, monkeypatch):
        trader, tracker, tape = self._setup(monkeypatch)

        def boom(*args, **kwargs):
            raise ValueError("only 40 daily sessions of history")

        monkeypatch.setattr(at._dayrange(), "forecast_session", boom)
        tape.append(104.0)
        dot, text = self._activity(trader, state, tracker)
        assert dot == "🔴" and "cannot trade today" in text

    def test_the_loop_publishes_it_and_clears_it_on_the_way_out(
        self, state, monkeypatch
    ):
        seen: list = []

        class OneCycle:
            halt = None

            def run_cycle(self, bundle, state, tracker):
                self.halt = "stopped out"
                return "hold"

            def activity(self, outcome, tracker):
                return ("🟢", f"after {outcome}")

        def end(state, tracker, text):
            seen.append(state.agent_activity)
            original_end(state, tracker, text)

        original_end = at.rule_agent.end_session
        monkeypatch.setattr(at.rule_agent, "end_session", end)
        monkeypatch.setattr(at.apple_models, "load", lambda key, ticker=None: DAYRANGE_BUNDLE)
        monkeypatch.setattr(at, "config_error", lambda config, bundle=None: None)
        monkeypatch.setattr(at, "build_trader", lambda config, bundle=None: OneCycle())
        state.agent_running = True
        at._apple_trader_loop(
            state, tracker_for_loop(), AppleTraderConfig(), 60, threading.Event()
        )
        assert seen == [("🟢", "after hold")]
        assert state.agent_activity is None


class TestRecordedLevels:
    """What a live run publishes for the chart: one row per cycle, from the
    same plan the analysis line prints, so the two can never disagree."""

    def _cycle(self, trader, state, tracker, tape, close, **bar):
        tape.append(close, **bar)
        trader.run_cycle(DAYRANGE_BUNDLE, state, tracker)

    def _analysis(self, state):
        return [e["text"] for e in state.agent_log if e.get("type") == "analysis"]

    def test_the_first_row_is_the_forecasts_own_levels_at_the_window_end(
        self, state, market_open, monkeypatch
    ):
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(108.0))
        tape = Tape(monkeypatch)
        trader = at.DayRangeTrader(dayrange_config(breach_update="extreme"))
        self._cycle(trader, state, tracker, tape, 111.5, high=112.0)   # a breach

        rows = state.apple_trader_levels["rows"]
        assert [r["t"] for r in rows] == [tape.index[4], tape.index[-1]]
        assert rows[0]["buy"] == pytest.approx(BUY_LEVEL)
        assert rows[1]["buy"] == pytest.approx(112.0 - 7.5)
        assert state.apple_trader_levels["ticker"] == TICKER

    def test_every_row_quotes_what_that_cycles_analysis_line_quotes(
        self, state, market_open, monkeypatch
    ):
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(108.0))
        tape = Tape(monkeypatch)
        trader = at.DayRangeTrader(dayrange_config(breach_update="brownian"))
        for close, high in [(108.0, 108.2), (111.0, 112.5), (110.0, 110.5), (113.0, 114.0)]:
            self._cycle(trader, state, tracker, tape, close, high=high)
            row = state.apple_trader_levels["rows"][-1]
            line = self._analysis(state)[-1]
            assert f"buy ${row['buy']:,.2f}" in line
            assert f"sell ${row['sell']:,.2f}" in line


class TestLateStart:
    """A run started after 9:35 walks the bars it missed one at a time, so each
    revision of the forecast lands on the bar that breached it -- not on the
    minute ▶ Start was pressed -- and under "brownian" is extended by the reach
    left at that bar. It ends up exactly where a run up since 9:35 would be."""

    BARS = [(108.0, 108.2), (111.0, 112.5), (110.0, 110.5), (113.0, 114.0), (112.0, 112.4)]

    @staticmethod
    def _state() -> AppState:
        fresh = AppState()
        fresh.set_symbols([TICKER])
        fresh.api_key, fresh.api_secret, fresh.feed = "k", "s", "iex"
        return fresh

    @staticmethod
    def _levels(rows):
        return [(r["t"], r["pred_high"], r["pred_low"], r["buy"], r["sell"]) for r in rows]

    def test_a_late_start_matches_a_run_that_was_up_all_along(
        self, state, market_open, monkeypatch
    ):
        config = dayrange_config(breach_update="brownian", keep_width=True)
        tape = Tape(monkeypatch)
        early = at.DayRangeTrader(config)
        early_tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(108.0))
        for close, high in self.BARS:
            tape.append(close, high=high)
            early.run_cycle(DAYRANGE_BUNDLE, state, early_tracker)

        late_state = self._state()
        late = at.DayRangeTrader(config)
        late.run_cycle(
            DAYRANGE_BUNDLE, late_state,
            DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(108.0)),
        )
        # Row for row, the same arithmetic on the same bars.
        assert self._levels(late_state.apple_trader_levels["rows"]) == self._levels(
            state.apple_trader_levels["rows"]
        )
        for key in ("pred_high", "pred_low", "buy_level", "sell_level"):
            assert late.plan[key] == pytest.approx(early.plan[key])

    def test_the_revision_is_dated_at_the_breach_not_at_the_start(
        self, state, market_open, monkeypatch
    ):
        tape = Tape(monkeypatch)
        for close, high in self.BARS:
            tape.append(close, high=high)
        trader = at.DayRangeTrader(dayrange_config(breach_update="brownian"))
        trader.run_cycle(
            DAYRANGE_BUNDLE, state,
            DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(108.0)),
        )
        rows = {r["t"]: r["pred_high"] for r in state.apple_trader_levels["rows"]}
        first_breach, before_it = tape.index[6], tape.index[5]
        assert rows[before_it] == pytest.approx(FORECAST["pred_high"])
        assert rows[first_breach] > 112.5      # the extreme plus the reach left then
        caught_up = [e["text"] for e in state.agent_log if "caught up" in e["text"]]
        assert len(caught_up) == 1 and f"{first_breach:%H:%M}" in caught_up[0]


class TestSidebarLevelEdits:
    """A running agent takes the buy and sell distances the sidebar holds,
    from its next bar, and nothing else."""

    def _started(self, state, monkeypatch, **kwargs):
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(108.0))
        tape = Tape(monkeypatch)
        trader = at.DayRangeTrader(dayrange_config(**kwargs))
        state.apple_trader_config = trader.config
        tape.append(108.0)
        trader.run_cycle(DAYRANGE_BUNDLE, state, tracker)
        return trader, tracker, tape

    def test_an_edit_moves_the_levels_from_the_next_bar_and_is_logged(
        self, state, market_open, monkeypatch
    ):
        trader, tracker, tape = self._started(state, monkeypatch)
        state.apple_trader_config = dayrange_config(buy_k=0.9, sell_k=0.2)
        tape.append(108.0)
        trader.run_cycle(DAYRANGE_BUNDLE, state, tracker)

        assert (trader.config.buy_k, trader.config.sell_k) == (0.9, 0.2)
        assert trader.plan["buy_level"] == pytest.approx(110.0 - 9.0)
        assert trader.plan["sell_level"] == pytest.approx(110.0 - 2.0)
        rows = state.apple_trader_levels["rows"]
        assert [r["buy"] for r in rows] == pytest.approx([BUY_LEVEL, BUY_LEVEL, 101.0])
        assert state.apple_trader_levels["config"].buy_k == 0.9
        changed = [e["text"] for e in state.agent_log if "from the sidebar" in e["text"]]
        assert len(changed) == 1
        assert "buy 0.75 → 0.9" in changed[0] and "was $102.50" in changed[0]

    def test_other_settings_are_not_adopted(self, state, market_open, monkeypatch):
        trader, tracker, tape = self._started(state, monkeypatch)
        state.apple_trader_config = dayrange_config(position_pct=5.0, breach_update="extreme")
        tape.append(108.0)
        trader.run_cycle(DAYRANGE_BUNDLE, state, tracker)
        assert trader.config.position_pct != 5.0
        assert trader.config.breach_update == "off"

    def test_a_form_for_another_instrument_is_ignored(self, state, market_open, monkeypatch):
        trader, tracker, tape = self._started(state, monkeypatch)
        state.apple_trader_config = dayrange_config(ticker="GOOGL", buy_k=1.5, sell_k=0.9)
        tape.append(108.0)
        trader.run_cycle(DAYRANGE_BUNDLE, state, tracker)
        assert trader.config.buy_k == 0.75

    def test_distances_in_another_unit_are_refused_once(
        self, state, market_open, monkeypatch
    ):
        """0.9 × ADR and 0.9 × the predicted range are different orders."""
        trader, tracker, tape = self._started(state, monkeypatch)
        state.apple_trader_config = dayrange_config(
            buy_k=0.9, sell_k=0.2, level_unit=UNIT_PRED_RANGE
        )
        for _ in range(3):
            tape.append(108.0)
            trader.run_cycle(DAYRANGE_BUNDLE, state, tracker)
        assert trader.config.buy_k == 0.75
        refused = [e for e in state.agent_log if "needs ▶ Start" in e["text"]]
        assert len(refused) == 1

    def test_an_open_position_keeps_the_stop_it_was_filled_with(
        self, state, market_open, monkeypatch
    ):
        broker = FakeBroker(103.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tape = Tape(monkeypatch, broker)
        trader = at.DayRangeTrader(dayrange_config())
        state.apple_trader_config = trader.config
        tape.append(103.0, low=102.0)             # touches 102.50: bought at 103
        trader.run_cycle(DAYRANGE_BUNDLE, state, tracker)
        assert tracker.position_for(TICKER) > 0
        risk = trader.entry["risk"]

        state.apple_trader_config = dayrange_config(buy_k=0.9, sell_k=0.2)
        tape.append(104.0)
        trader.run_cycle(DAYRANGE_BUNDLE, state, tracker)
        assert trader.entry["risk"] == risk
        # The recorded stop is the position's, which is what the log line quotes.
        stop = state.apple_trader_levels["rows"][-1]["stop"]
        assert stop == pytest.approx(103.0 - risk)
        assert f"stop ${stop:,.2f}" in state.agent_log[-1]["text"]

    def test_a_restart_keeps_what_the_earlier_run_rested(
        self, state, market_open, monkeypatch
    ):
        trader, tracker, tape = self._started(state, monkeypatch)
        before = list(state.apple_trader_levels["rows"])

        # ▶ Start launches with what the sidebar holds.
        state.apple_trader_config = dayrange_config(buy_k=0.9, sell_k=0.2)
        restarted = at.DayRangeTrader(state.apple_trader_config)
        tape.append(108.0)
        restarted.run_cycle(DAYRANGE_BUNDLE, state, tracker)
        rows = state.apple_trader_levels["rows"]
        assert rows[: len(before)] == before
        assert rows[-1]["t"] == tape.index[-1]
        assert rows[-1]["buy"] == pytest.approx(101.0)
        assert len(rows) == len(before) + 1


class TestRestartMemory:
    """A restart the same day picks up what the earlier run knew and the ledger
    does not: the open position's own fill and stop, and a stand-down."""

    def _bought(self, state, monkeypatch):
        broker = FakeBroker(103.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tape = Tape(monkeypatch, broker)
        trader = at.DayRangeTrader(dayrange_config())
        tape.append(103.0, low=102.0)             # touches 102.50: bought at 103
        trader.run_cycle(DAYRANGE_BUNDLE, state, tracker)
        trader.publish_memory(state)
        assert tracker.position_for(TICKER) > 0
        return trader, tracker, tape

    def test_the_published_memory_carries_the_entry(self, state, market_open, monkeypatch):
        trader, _, _ = self._bought(state, monkeypatch)
        memory = state.apple_trader_levels["memory"]
        assert memory["entry"]["price"] == pytest.approx(103.0)
        assert memory["entry"]["risk"] == trader.entry["risk"]
        assert memory["stand_down"] is None

    def test_a_restart_resumes_the_position_at_its_own_fill(
        self, state, market_open, monkeypatch
    ):
        trader, tracker, tape = self._bought(state, monkeypatch)
        risk = trader.entry["risk"]

        restarted = at.DayRangeTrader(dayrange_config())
        tape.append(104.0)
        restarted.run_cycle(DAYRANGE_BUNDLE, state, tracker)
        # Not re-adopted at this bar's 104 close: the fill it was bought at.
        assert restarted.entry["price"] == pytest.approx(103.0)
        assert restarted.entry["risk"] == risk
        assert state.apple_trader_levels["rows"][-1]["stop"] == pytest.approx(103.0 - risk)

    def test_a_remembered_entry_is_dropped_when_the_ledger_is_flat(
        self, state, market_open, monkeypatch
    ):
        _, _, tape = self._bought(state, monkeypatch)
        fresh = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(104.0))
        restarted = at.DayRangeTrader(dayrange_config())
        tape.append(104.0)
        restarted.run_cycle(DAYRANGE_BUNDLE, state, fresh)
        assert restarted.entry is None

    def test_a_restart_keeps_the_breakers_stand_down(self, state, market_open, monkeypatch):
        trader, tracker, tape = self._bought(state, monkeypatch)
        trader.plan["stand_down"] = "closed for +0.05 × ADR"
        trader.publish_memory(state)

        restarted = at.DayRangeTrader(dayrange_config())
        tape.append(104.0)
        restarted.run_cycle(DAYRANGE_BUNDLE, state, tracker)
        assert restarted.plan["stand_down"] == "closed for +0.05 × ADR"
        assert any("still no new entries" in e.get("text", "") for e in state.agent_log)

    def test_start_after_a_stop_out_buys_again(self, state, market_open, monkeypatch):
        """A stop ends the run; pressing ▶ Start again is the say-so to trade
        the levels again today."""
        trader, tracker, tape = self._bought(state, monkeypatch)
        tape.append(STOP_PRICE - 0.5)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "sold"
        trader.publish_memory(state)
        assert state.apple_trader_levels["memory"]["stand_down"] == "stopped out"

        restarted = at.DayRangeTrader(dayrange_config())
        tape.append(100.0, low=BUY_LEVEL - 3)
        assert restarted.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "bought"
        assert not restarted.plan.get("stand_down")
        assert restarted.halt is None

    def test_the_memory_survives_the_session_file(
        self, state, market_open, monkeypatch, tmp_path
    ):
        """Through disk, as an app restart takes it: save, restore onto a new
        state, continue the ledger, restart the trader."""
        from agent_stonks import session_store

        _, tracker, tape = self._bought(state, monkeypatch)
        state.decision_tracker = tracker
        session_store.claim(state)
        assert session_store.save(state) is not None

        restored = AppState()
        restored.set_symbols([TICKER])
        restored.api_key, restored.api_secret, restored.feed = "k", "s", "iex"
        assert session_store.restore(restored) is not None
        continued = DecisionTracker(starting_cash=0.0, broker=FakeBroker(104.0))
        continued.carry_over(restored.decision_tracker)
        assert continued.position_for(TICKER) == tracker.position_for(TICKER)

        restarted = at.DayRangeTrader(dayrange_config())
        tape.append(104.0)
        restarted.run_cycle(DAYRANGE_BUNDLE, restored, continued)
        assert restarted.entry["price"] == pytest.approx(103.0)
        assert restarted.entry["ts"] == tape.index[-2]


class TestSidebarModelSwitch:
    """A running agent switches to the model the sidebar names, from its next
    bar: it re-forecasts the day with that model and rests the new model's
    levels, and the chart's record shows both, each for its own minutes."""

    HIGHLOW = {"kind": "highlow", "opening_minutes": 5}

    def _stub_highlow(self, monkeypatch, pred_high=120.0, fails=False):
        original = at.session_forecast

        def forecast(bundle, *args, **kwargs):
            if bundle.get("kind") != "highlow":
                return original(bundle, *args, **kwargs)
            if fails:
                raise RuntimeError("no SIP history")
            return {**FORECAST, "pred_high": pred_high}, None

        monkeypatch.setattr(at, "session_forecast", forecast)
        monkeypatch.setattr(
            at.apple_models, "load",
            lambda key, ticker: self.HIGHLOW if key == "highlow" else DAYRANGE_BUNDLE,
        )

    def _started(self, state, monkeypatch, broker=None, close=108.0, **bar):
        broker = broker or FakeBroker(close)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tape = Tape(monkeypatch, broker)
        trader = at.DayRangeTrader(dayrange_config())
        state.apple_trader_config = trader.config
        tape.append(close, **bar)
        trader.run_cycle(DAYRANGE_BUNDLE, state, tracker)
        return trader, tracker, tape

    def _switch(self, state, **kwargs):
        kwargs.setdefault("buy_k", 0.5)
        kwargs.setdefault("sell_k", 0.1)
        state.apple_trader_config = replace(dayrange_config(**kwargs), model_key="highlow")

    def test_a_switch_reforecasts_and_moves_the_levels_from_the_next_bar(
        self, state, market_open, monkeypatch
    ):
        self._stub_highlow(monkeypatch)
        trader, tracker, tape = self._started(state, monkeypatch)
        self._switch(state)
        tape.append(108.0)
        # The loop still hands over the bundle it was started with.
        trader.run_cycle(DAYRANGE_BUNDLE, state, tracker)

        assert trader.config.model_key == "highlow"
        assert (trader.config.buy_k, trader.config.sell_k) == (0.5, 0.1)
        assert trader.plan["pred_high"] == pytest.approx(120.0)
        assert trader.plan["buy_level"] == pytest.approx(120.0 - 5.0)
        assert trader.plan["sell_level"] == pytest.approx(120.0 - 1.0)
        rows = state.apple_trader_levels["rows"]
        assert [r["model_key"] for r in rows] == ["dayrange", "dayrange", "highlow"]
        assert [r["buy"] for r in rows] == pytest.approx([BUY_LEVEL, BUY_LEVEL, 115.0])
        assert state.apple_trader_levels["config"].model_key == "highlow"
        switched = [e["text"] for e in state.agent_log if "model switched" in e.get("text", "")]
        assert len(switched) == 1 and "was $102.50" in switched[0]

        # And it keeps trading on the new model: the forecast is not re-made.
        tape.append(108.0)
        trader.run_cycle(DAYRANGE_BUNDLE, state, tracker)
        assert state.apple_trader_levels["rows"][-1]["buy"] == pytest.approx(115.0)

    def test_an_open_position_keeps_its_fill_and_its_stop(
        self, state, market_open, monkeypatch
    ):
        self._stub_highlow(monkeypatch)
        broker = FakeBroker(103.0)
        trader, tracker, tape = self._started(
            state, monkeypatch, broker=broker, close=103.0, low=102.0
        )
        assert tracker.position_for(TICKER) > 0
        entry = dict(trader.entry)
        self._switch(state)
        tape.append(104.0)
        trader.run_cycle(DAYRANGE_BUNDLE, state, tracker)
        assert trader.entry["price"] == entry["price"]
        assert trader._stop_price() == pytest.approx(103.0 - entry["risk"])
        assert state.apple_trader_levels["rows"][-1]["stop"] == pytest.approx(
            103.0 - entry["risk"]
        )

    def test_a_model_that_cannot_forecast_is_refused_once_and_the_run_carries_on(
        self, state, market_open, monkeypatch
    ):
        self._stub_highlow(monkeypatch, fails=True)
        trader, tracker, tape = self._started(state, monkeypatch)
        self._switch(state)
        for _ in range(3):
            tape.append(108.0)
            trader.run_cycle(DAYRANGE_BUNDLE, state, tracker)
        assert trader.config.model_key == "dayrange"
        assert trader.blocked is None
        assert trader.plan["buy_level"] == pytest.approx(BUY_LEVEL)
        refused = [e for e in state.agent_log if "run stays on" in e.get("text", "")]
        assert len(refused) == 1 and "no SIP history" in refused[0]["text"]

    def test_a_missing_model_file_is_refused(self, state, market_open, monkeypatch):
        trader, tracker, tape = self._started(state, monkeypatch)
        monkeypatch.setattr(at.apple_models, "load", lambda key, ticker: None)
        self._switch(state)
        tape.append(108.0)
        trader.run_cycle(DAYRANGE_BUNDLE, state, tracker)
        assert trader.config.model_key == "dayrange"
        assert any("run stays on" in e.get("text", "") for e in state.agent_log)

    def test_before_the_forecast_the_new_model_simply_makes_it(
        self, state, market_open, monkeypatch
    ):
        self._stub_highlow(monkeypatch)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(108.0))
        tape = Tape(monkeypatch, minutes=3)
        trader = at.DayRangeTrader(dayrange_config())
        self._switch(state)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "warming_up"
        assert trader.config.model_key == "highlow" and trader.plan is None
        for i in (3, 4):
            tape.append(101.0, offset=i)
        trader.run_cycle(DAYRANGE_BUNDLE, state, tracker)
        assert trader.plan["pred_high"] == pytest.approx(120.0)

    def test_no_stop_is_recorded_before_a_buy(self, state, market_open, monkeypatch):
        trader, tracker, tape = self._started(state, monkeypatch)
        assert all(r["stop"] is None for r in state.apple_trader_levels["rows"])


class TestReadsClosedBarsOnly:
    """Through the real `minute_frame`, not `Tape`'s stand-in: with a Finnhub
    buffer the newest row is the minute still trading, and the agent must act
    on the one that just closed."""

    def test_a_dip_in_the_bar_that_just_closed_buys(self, state, monkeypatch):
        original = at.momentum_regime.minute_frame
        tape = Tape(monkeypatch)
        monkeypatch.setattr(at.momentum_regime, "minute_frame", original)
        clock.set_simulated(datetime(2026, 7, 21, 14, 30, 5, tzinfo=timezone.utc))  # 10:30:05 ET
        try:
            def bar(ts, close, low=None):
                return {"t": ts.tz_convert("UTC").isoformat(), "o": close, "h": close,
                        "l": close if low is None else low, "c": close, "v": 1.0e5}

            opening = [bar(ts, row["close"], row["low"]) for ts, row in zip(tape.index, tape.rows)]
            ten = Tape.OPEN + pd.Timedelta(hours=1)
            sym_state = state.sym(TICKER)
            sym_state.bars.extend(opening + [
                bar(ten - pd.Timedelta(minutes=1), 103.0, low=102.0),  # 10:29: touched 102.50
                bar(ten, 104.0),                                      # 10:30: 5 s old
            ])
            broker = FakeBroker(103.0)
            tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
            trader = at.DayRangeTrader(dayrange_config())
            assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "bought"
            line = [e["text"] for e in state.agent_log if e.get("type") == "analysis"][-1]
            assert line.startswith(f"{TICKER} 10:29 ")
        finally:
            clock.clear()


class _Ladder:
    """What both ladder classes drive a trader with: 50% of the cash a buy."""

    RISK = 0.5 * TARGET_GAIN  # 3.25

    def _trader(self, **kwargs):
        kwargs.setdefault("scale_in", True)
        kwargs.setdefault("position_pct", 50.0)
        return at.DayRangeTrader(dayrange_config(**kwargs))

    def _cycle(self, trader, state, tracker, tape, close, **bar):
        tape.append(close, **bar)
        return trader.run_cycle(DAYRANGE_BUNDLE, state, tracker)

    def _analysis(self, state):
        return [e["text"] for e in state.agent_log if e.get("type") == "analysis"]

    def _bought_once(self, state, monkeypatch, at_price=BUY_LEVEL, **kwargs):
        broker = FakeBroker(at_price)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tape = Tape(monkeypatch, broker)
        trader = self._trader(**kwargs)
        assert self._cycle(trader, state, tracker, tape, at_price) == "bought"
        return trader, tracker, tape


class TestScaleIn(_Ladder):
    """Buying again lower while the cash left over allows it, on the half-way
    rung (`buy_step_k` 0, pinned by `dayrange_config`).

    Notebook arithmetic (H $110, ADR $10, buy 0.75): the first buy rests at
    $102.50 and the bottom of the range is $100.00, one ADR under H, so the
    ladder is $102.50 → $101.25 → $100.625, each rung half-way down what is
    left. The stop is half the $6.50 predicted gain, $3.25, under the last
    actual fill -- so $99.25 under a fill at the buy level, and every rung here
    is above it.
    """

    RUNG_2 = 101.25   # 110 - 0.875 x 10
    RUNG_3 = 100.625  # 110 - 0.9375 x 10

    def test_after_a_buy_the_next_rests_half_way_to_the_bottom_of_the_range(
        self, state, market_open, monkeypatch
    ):
        trader, tracker, tape = self._bought_once(state, monkeypatch)
        assert trader.plan["buy_level"] == pytest.approx(self.RUNG_2)
        assert trader._stop_price() == pytest.approx(BUY_LEVEL - self.RISK)
        assert any("Next buy at $101.25" in line for line in self._analysis(state))

    def test_the_stop_hangs_under_the_actual_fill_not_the_buy_level(
        self, state, market_open, monkeypatch
    ):
        """A fill under the level -- the price went on falling until the
        momentum confirmed -- takes the stop down with it."""
        trader, tracker, tape = self._bought_once(state, monkeypatch, at_price=101.5)
        assert trader.entry["last_fill"] == pytest.approx(101.5)
        assert trader._stop_price() == pytest.approx(101.5 - self.RISK)

    def test_a_rung_at_or_under_the_stop_is_never_bought(
        self, state, market_open, monkeypatch
    ):
        """A stop of 0.1 x the gain is $0.65 under the $102.50 fill, $101.85 --
        above the $101.25 rung, so a bar reaching the rung stops out first."""
        trader, tracker, tape = self._bought_once(
            state, monkeypatch, stop_gain_fraction=0.1
        )
        assert trader._stop_price() == pytest.approx(BUY_LEVEL - 0.65)
        assert trader.plan["buy_level"] == pytest.approx(BUY_LEVEL)
        assert "at or under the stop" in self._analysis(state)[-1]
        assert self._cycle(trader, state, tracker, tape, 101.2) == "sold"

    def test_a_bar_that_reaches_the_next_rung_adds_to_the_position(
        self, state, market_open, monkeypatch
    ):
        trader, tracker, tape = self._bought_once(state, monkeypatch)
        first = tracker.position_for(TICKER)                       # 48 sh
        assert self._cycle(trader, state, tracker, tape, 102.0, low=101.9) == "hold"
        assert self._cycle(trader, state, tracker, tape, self.RUNG_2) == "bought"

        held = tracker.position_for(TICKER)
        added = held - first
        assert first == 48 and added == 25     # half of the $5,080 left
        average = (48 * BUY_LEVEL + 25 * self.RUNG_2) / 73
        assert trader.entry["price"] == pytest.approx(average)
        assert trader.plan["buy_level"] == pytest.approx(self.RUNG_3)
        # Under the add's own fill, not the average cost, which is above it.
        assert trader._stop_price() == pytest.approx(self.RUNG_2 - self.RISK)

    def test_an_add_must_close_under_the_last_fill(
        self, state, market_open, monkeypatch
    ):
        """A first buy at $100.90 is already under the $101.25 rung. The next
        bar reaches the rung but closes at $101.00, above the fill: no add.
        One closing at $100.50 is lower, and adds."""
        trader, tracker, tape = self._bought_once(state, monkeypatch, at_price=100.9)
        first = tracker.position_for(TICKER)
        assert self._cycle(trader, state, tracker, tape, 101.0) == "hold"
        assert tracker.position_for(TICKER) == first
        status = [e["text"] for e in state.agent_log if e.get("type") == "status"][-1]
        assert "not under the $100.90 last fill" in status
        assert self._cycle(trader, state, tracker, tape, 100.5) == "bought"
        assert trader.entry["last_fill"] == pytest.approx(100.5)

    def test_legacy_an_add_fills_at_any_price_on_the_rung(
        self, state, market_open, monkeypatch
    ):
        trader, tracker, tape = self._bought_once(
            state, monkeypatch, at_price=100.9, add_under_fill=False
        )
        assert self._cycle(trader, state, tracker, tape, 101.0) == "bought"

    def test_a_bar_through_the_fills_stop_sells_rather_than_adds(
        self, state, market_open, monkeypatch
    ):
        """$99.00 is through the $102.50 fill's $99.25 stop: the stop is read
        first, so the bar sells everything even though it passed the rung."""
        trader, tracker, tape = self._bought_once(state, monkeypatch)
        assert self._cycle(trader, state, tracker, tape, 101.0, low=99.0) == "sold"
        assert tracker.position_for(TICKER) == 0
        reasoning = tracker.snapshot()["decisions"][-1].reasoning
        assert "$99.25 stop" in reasoning and "$102.50 fill" in reasoning

    def test_legacy_the_bar_that_would_have_stopped_a_single_buy_adds_instead(
        self, state, market_open, monkeypatch
    ):
        """A record from 2026-09-23 to -28: the stop under the next rung, so
        $99.00 is above it and the bar adds at $101.25."""
        trader, tracker, tape = self._bought_once(
            state, monkeypatch, stop_under_next_buy=True
        )
        assert trader._stop_price() == pytest.approx(self.RUNG_2 - self.RISK)
        assert self._cycle(trader, state, tracker, tape, 101.0, low=99.0) == "bought"
        assert tracker.position_for(TICKER) > 48

    def test_legacy_a_bar_through_the_stop_under_the_next_rung_sells_everything(
        self, state, market_open, monkeypatch
    ):
        trader, tracker, tape = self._bought_once(
            state, monkeypatch, stop_under_next_buy=True
        )
        stop = self.RUNG_2 - self.RISK                                  # 98.00
        assert self._cycle(trader, state, tracker, tape, 97.5, low=97.9) == "sold"
        assert tracker.position_for(TICKER) == 0
        reasoning = tracker.snapshot()["decisions"][-1].reasoning
        assert f"${stop:,.2f} stop" in reasoning and "$101.25 next buy" in reasoning

    def test_the_target_sells_the_whole_ladder_and_resets_it(
        self, state, market_open, monkeypatch
    ):
        trader, tracker, tape = self._bought_once(state, monkeypatch)
        self._cycle(trader, state, tracker, tape, self.RUNG_2)
        assert self._cycle(trader, state, tracker, tape, 108.0, high=SELL_LEVEL) == "sold"
        assert tracker.position_for(TICKER) == 0
        self._cycle(trader, state, tracker, tape, 107.0)
        assert trader.plan["buy_level"] == pytest.approx(BUY_LEVEL)

    def test_when_the_cash_runs_out_the_stop_moves_under_the_last_fill(
        self, state, market_open, monkeypatch
    ):
        """At 95% the second buy is the last one the cash can pay for."""
        trader, tracker, tape = self._bought_once(state, monkeypatch, position_pct=95.0)
        assert self._cycle(trader, state, tracker, tape, self.RUNG_2) == "bought"
        held = tracker.position_for(TICKER)

        assert trader._stop_price() == pytest.approx(self.RUNG_2 - self.RISK)
        assert trader.plan["buy_level"] == pytest.approx(BUY_LEVEL)  # no next rung
        assert "No further buys" in self._analysis(state)[-1]
        assert self._cycle(trader, state, tracker, tape, self.RUNG_3) == "hold"
        assert tracker.position_for(TICKER) == held

    def test_the_bottom_of_a_predicted_range_is_the_predicted_low(
        self, state, market_open, monkeypatch
    ):
        """Unit $15 (110 - 95): the buy at $98.75, half-way to $95 is $96.875."""
        buy = 110.0 - 0.75 * 15.0
        trader, tracker, tape = self._bought_once(
            state, monkeypatch, at_price=buy, level_unit=UNIT_PRED_RANGE
        )
        assert trader.plan["buy_level"] == pytest.approx((buy + FORECAST["pred_low"]) / 2)

    def test_at_full_size_a_position_is_bought_once(
        self, state, market_open, monkeypatch
    ):
        trader, tracker, tape = self._bought_once(state, monkeypatch, position_pct=100.0)
        held = tracker.position_for(TICKER)
        assert trader.plan["buy_level"] == pytest.approx(BUY_LEVEL)
        assert trader._stop_price() == pytest.approx(BUY_LEVEL - self.RISK)
        assert self._cycle(trader, state, tracker, tape, 100.5) == "hold"
        assert tracker.position_for(TICKER) == held

    def test_switched_off_the_stop_is_the_fills_own(
        self, state, market_open, monkeypatch
    ):
        trader, tracker, tape = self._bought_once(state, monkeypatch, scale_in=False)
        assert trader.plan["buy_level"] == pytest.approx(BUY_LEVEL)
        assert trader._stop_price() == pytest.approx(BUY_LEVEL - self.RISK)
        assert not any("Next buy" in line for line in self._analysis(state))

    def test_the_chart_record_steps_to_each_rung_with_its_stop(
        self, state, market_open, monkeypatch
    ):
        trader, tracker, tape = self._bought_once(state, monkeypatch)
        self._cycle(trader, state, tracker, tape, 102.0)
        row = state.apple_trader_levels["rows"][-1]
        assert row["buy"] == pytest.approx(self.RUNG_2)
        assert row["stop"] == pytest.approx(BUY_LEVEL - self.RISK)
        line = [x for x in self._analysis(state) if " · " in x][-1]
        assert f"buy ${row['buy']:,.2f}" in line and f"stop ${row['stop']:,.2f}" in line

    def test_signed_only_where_it_can_change_a_trade(self):
        assert "adds=+0.1A,stop@fill" in config_signature(
            AppleTraderConfig(model_key="dayrange", level_unit=UNIT_ADR)
        )
        assert "adds=half,stop@fill" in config_signature(
            AppleTraderConfig(model_key="dayrange", buy_step_k=0.0)
        )
        legacy = config_signature(
            AppleTraderConfig(
                model_key="dayrange", stop_under_next_buy=True, buy_step_k=0.0
            )
        )
        assert "adds=half" in legacy and "stop@fill" not in legacy
        assert "stop@fill" not in config_signature(
            AppleTraderConfig(model_key="dayrange", stop_gain_fraction=0.0)
        )
        assert "add<fill" in config_signature(AppleTraderConfig(model_key="dayrange"))
        assert "add<fill" not in config_signature(
            AppleTraderConfig(model_key="dayrange", add_under_fill=False)
        )
        assert "add<fill" not in config_signature(
            AppleTraderConfig(model_key="dayrange", position_pct=100.0)
        )
        assert "adds=" not in config_signature(
            AppleTraderConfig(model_key="dayrange", position_pct=100.0)
        )
        assert "adds=" not in config_signature(
            AppleTraderConfig(model_key="dayrange", scale_in=False)
        )

    def test_a_record_from_before_the_ladder_replays_without_it(self):
        from simlab.rule_agents import _apple_from_record

        assert _apple_from_record({"position_pct": 50.0}).scale_in is False

    def test_a_record_from_before_the_fill_stop_replays_under_the_rung(self):
        from simlab.rule_agents import _apple_from_record

        old = _apple_from_record({"position_pct": 50.0, "scale_in": True})
        assert old.stop_under_next_buy is True
        assert "stop@fill" not in config_signature(old)
        assert old.add_under_fill is False and "add<fill" not in config_signature(old)


class TestBuyStep(_Ladder):
    """Each buy rests the next one `buy_step_k` units lower (2026-10-01).

    Same arithmetic as `TestScaleIn` at a 0.1 step: the first buy at $102.50
    (0.75), the next at $101.50 (0.85), then $100.50 (0.95) -- every rung above
    the $99.25 stop under a fill at the level.
    """

    RUNG_2 = 101.50   # 110 - 0.85 x 10
    RUNG_3 = 100.50   # 110 - 0.95 x 10

    def _trader(self, **kwargs):
        kwargs.setdefault("buy_step_k", 0.1)
        return super()._trader(**kwargs)

    def test_after_a_buy_the_next_rests_one_step_lower(
        self, state, market_open, monkeypatch
    ):
        trader, tracker, tape = self._bought_once(state, monkeypatch)
        assert trader.plan["buy_level"] == pytest.approx(self.RUNG_2)
        line = self._analysis(state)[-1]
        assert "Next buy at $101.50 (0.85 × ADR" in line and "0.1 lower" in line

    def test_each_add_steps_another_notch_down(self, state, market_open, monkeypatch):
        trader, tracker, tape = self._bought_once(state, monkeypatch)
        assert self._cycle(trader, state, tracker, tape, self.RUNG_2) == "bought"
        assert trader.entry["fills"] == 2
        assert trader.plan["buy_level"] == pytest.approx(self.RUNG_3)
        reasoning = tracker.snapshot()["decisions"][-1].reasoning
        assert "0.85 × the ADR" in reasoning and "0.1 lower" in reasoning

    def test_selling_everything_puts_the_buy_back_at_the_buy_distance(
        self, state, market_open, monkeypatch
    ):
        trader, tracker, tape = self._bought_once(state, monkeypatch)
        self._cycle(trader, state, tracker, tape, self.RUNG_2)
        assert self._cycle(trader, state, tracker, tape, 108.0, high=SELL_LEVEL) == "sold"
        assert tracker.position_for(TICKER) == 0
        self._cycle(trader, state, tracker, tape, 107.0)
        assert trader.plan["buy_level"] == pytest.approx(BUY_LEVEL)

    def test_the_step_is_not_capped_at_the_bottom_of_the_range(
        self, state, market_open, monkeypatch
    ):
        """Buy 0.95 at $100.50: the half-way rule would rest the next at
        0.975, the step puts it at 1.05, $99.50 -- under the $100 bottom."""
        trader, tracker, tape = self._bought_once(
            state, monkeypatch, at_price=100.5, buy_k=0.95
        )
        assert trader.plan["buy_level"] == pytest.approx(99.5)

    def test_a_record_from_before_the_step_replays_half_way(self):
        from simlab.rule_agents import _apple_from_record

        old = _apple_from_record({"position_pct": 50.0, "scale_in": True})
        assert old.buy_step_k == 0.0 and "adds=half" in config_signature(old)


class TestLimitEntry:
    """Buys are limit orders at the buy level (`limit_entry`).

    The order goes out after the bar that reached the level has closed, so on
    a bar that dipped and recovered a market order would pay the recovery. The
    limit refuses that, and the level stays armed for the next touch.
    """

    def _trader(self, **kwargs):
        kwargs.setdefault("limit_entry", True)
        return at.DayRangeTrader(dayrange_config(**kwargs))

    def _setup(self, monkeypatch, **kwargs):
        broker = FakeBroker(103.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        return self._trader(**kwargs), tracker, Tape(monkeypatch, broker), broker

    def test_on_for_every_new_config(self):
        assert AppleTraderConfig().limit_entry is True

    def test_a_dip_that_recovered_above_the_level_is_not_bought(
        self, state, market_open, monkeypatch
    ):
        trader, tracker, tape, broker = self._setup(monkeypatch)
        tape.append(103.0, low=BUY_LEVEL - 0.01)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "hold"
        assert broker.orders == []
        assert tracker.position_for(TICKER) == 0
        decision = tracker.snapshot()["decisions"][-1]
        assert decision.status == "rejected" and decision.limit_missed
        assert decision.limit_price == pytest.approx(BUY_LEVEL)
        assert "above the $102.50 limit" in decision.reasoning
        status = [e["text"] for e in state.agent_log if e.get("type") == "status"][-1]
        assert "stays armed" in status

    def test_the_level_stays_armed_and_the_next_touch_buys_at_or_under_it(
        self, state, market_open, monkeypatch
    ):
        trader, tracker, tape, broker = self._setup(monkeypatch)
        tape.append(103.0, low=BUY_LEVEL - 0.01)
        trader.run_cycle(DAYRANGE_BUNDLE, state, tracker)
        tape.append(102.3, low=102.2)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "bought"
        decision = tracker.snapshot()["decisions"][-1]
        assert decision.status == "filled" and decision.price == pytest.approx(102.3)
        assert decision.limit_price == pytest.approx(BUY_LEVEL)
        assert trader.entry["price"] == pytest.approx(102.3)

    def test_a_price_exactly_at_the_level_is_bought(
        self, state, market_open, monkeypatch
    ):
        trader, tracker, tape, _ = self._setup(monkeypatch)
        tape.append(BUY_LEVEL)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "bought"

    def test_legacy_a_market_buy_pays_the_close(
        self, state, market_open, monkeypatch
    ):
        trader, tracker, tape, _ = self._setup(monkeypatch, limit_entry=False)
        tape.append(103.0, low=BUY_LEVEL - 0.01)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "bought"
        assert tracker.snapshot()["decisions"][-1].price == pytest.approx(103.0)

    def test_a_missed_add_keeps_the_ladder(self, state, market_open, monkeypatch):
        """Scale-in notebook arithmetic: first buy at $102.50, next rung
        $101.25. A bar reaching the rung that closes at $102.00 is under the
        last fill but above the rung: not bought, and the rung is still there
        for the bar after."""
        trader, tracker, tape, _ = self._setup(
            monkeypatch, scale_in=True, position_pct=50.0
        )
        tape.append(BUY_LEVEL)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "bought"
        first = tracker.position_for(TICKER)
        rung = trader.plan["buy_level"]
        assert rung == pytest.approx(101.25)

        tape.append(102.0, low=rung - 0.01)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "hold"
        assert tracker.position_for(TICKER) == first
        assert trader.entry.get("can_add", True) is not False
        assert trader.plan["buy_level"] == pytest.approx(rung)

        tape.append(101.2)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "bought"
        assert tracker.position_for(TICKER) > first
        assert trader.entry["last_fill"] == pytest.approx(101.2)

    def test_signed_only_while_on(self):
        assert ",limit" in config_signature(AppleTraderConfig(model_key="dayrange"))
        assert ",limit" not in config_signature(
            AppleTraderConfig(model_key="dayrange", limit_entry=False)
        )

    def test_a_record_from_before_it_replays_with_market_buys(self):
        from simlab.rule_agents import _apple_from_record

        old = _apple_from_record({"position_pct": 50.0})
        assert old.limit_entry is False and ",limit" not in config_signature(old)


class TestMomentumReadTable:
    """`momentum_read`: the averages over N bars and the user's behaviour table."""

    # With N = 1 the averages are the plain 1-bar values: momentum is the last
    # move, change the last move minus the one before. A mean minute move of
    # 1.0 puts the neutral band at +-0.1.
    @staticmethod
    def read(moves):
        closes = [100.0]
        for m in moves:
            closes.append(closes[-1] + m)
        return at.momentum_read(closes, 1, 1.0)

    @pytest.mark.parametrize(
        "moves, mom, change, buy, take, sell_target",
        [
            ((-1.0, -0.5), "negative", "positive", False, False, True),
            ((-0.5, -0.5), "negative", "neutral", False, True, True),
            ((-0.5, -1.0), "negative", "negative", False, True, True),
            ((-0.5, 0.0), "neutral", "positive", True, False, True),
            ((0.0, 0.0), "neutral", "neutral", True, False, True),
            ((0.5, 0.0), "neutral", "negative", False, False, True),
            ((0.5, 1.0), "positive", "positive", True, False, False),
            ((0.5, 0.5), "positive", "neutral", True, False, False),
            ((1.0, 0.5), "positive", "negative", True, False, False),
        ],
    )
    def test_every_row_of_the_table(self, moves, mom, change, buy, take, sell_target):
        read = self.read(moves)
        assert (read["mom_class"], read["change_class"]) == (mom, change)
        row = read["row"]
        assert (row.buy, row.take, row.sell_target) == (buy, take, sell_target)

    def test_the_averages_over_n_bars(self):
        closes = [100.0, 101.0, 101.5, 101.0, 102.0, 101.0]
        read = at.momentum_read(closes, 3, 1.0)
        # (101 - 101.5) / 3, and ((101 - 102) - (101.5 - 101)) / 3.
        assert read["mom"] == pytest.approx(-0.5 / 3)
        assert read["change"] == pytest.approx(-1.5 / 3)

    def test_neutral_is_a_tenth_of_the_mean_minute_move(self):
        read = at.momentum_read([100.0, 100.0, 100.19], 1, 2.0)
        assert read["band"] == pytest.approx(0.2)
        assert read["mom_class"] == "neutral"
        assert at.momentum_read([100.0, 100.0, 100.21], 1, 2.0)["mom_class"] == "positive"

    def test_cannot_read_without_the_mean_move_or_enough_bars(self):
        assert at.momentum_read([100.0, 101.0, 102.0], 1, None) is None
        assert at.momentum_read([100.0, 101.0, 102.0], 1, 0.0) is None
        # The change over N bars needs N + 2 closes.
        assert at.momentum_read([100.0, 101.0, 102.0, 103.0], 3, 1.0) is None
        assert at.momentum_read([100.0, 101.0, 102.0, 103.0, 104.0], 3, 1.0) is not None


class TestMomentumConfirmation:
    """The behaviour table on the trader: what a buy at the buy level, a sell
    at the sell level and the take short of it each need.

    Notebook arithmetic (buy $102.50, sell $109.00), a 3-bar confirmation
    period and a mean minute move of $0.50, so the neutral band is +-$0.05.
    """

    N = 3
    MINUTE_MOVE = 0.5

    def _trader(self, **kwargs):
        kwargs.setdefault("momentum_confirmation_bars", self.N)
        return at.DayRangeTrader(dayrange_config(**kwargs))

    def _setup(self, state, monkeypatch, minute_move=MINUTE_MOVE, **kwargs):
        broker = FakeBroker(101.4)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tape = Tape(monkeypatch, broker)
        state.sym(TICKER).abs_mean_minute_momentum = minute_move
        return self._trader(**kwargs), tracker, tape

    def _step(self, trader, tracker, tape, state, close, **kw):
        tape.append(close, **kw)
        return trader.run_cycle(DAYRANGE_BUNDLE, state, tracker)

    def _status(self, state):
        return [e["text"] for e in state.agent_log if e.get("type") == "status"]

    def _enter(self, state, monkeypatch, **kwargs):
        """Bought at $101.80, on a rising bar under the buy level."""
        trader, tracker, tape = self._setup(state, monkeypatch, **kwargs)
        # The opening window closes at 101.0 ... 101.4; 101.8 is +0.17 a bar
        # over the last 3 -- positive momentum.
        assert self._step(trader, tracker, tape, state, 101.8) == "bought"
        return trader, tracker, tape

    @staticmethod
    def _force(monkeypatch, mom, change):
        """Pin the read to one row of the table, whatever the tape says."""
        row = at.MOMENTUM_TABLE[(mom, change)]
        read = {
            "n": 3, "mom": -1.0 if mom == "negative" else (1.0 if mom == "positive" else 0.0),
            "change": 0.0, "band": 0.05, "mom_class": mom, "change_class": change, "row": row,
        }
        monkeypatch.setattr(at, "momentum_read", lambda *a, **k: dict(read))

    # --- config ---------------------------------------------------------

    def test_on_by_default_and_signed(self):
        config = AppleTraderConfig()
        assert config.momentum_confirmation_bars == 3
        assert config.has_take
        signature = config_signature(config)
        assert ",confirm=3b" in signature and "@conf>=15m+loss," in signature

    def test_it_cannot_sit_beside_a_legacy_take(self):
        with pytest.raises(ValueError, match="only one may be set"):
            AppleTraderConfig(negative_momentum_bars=15)

    def test_off_signs_as_before(self):
        config = AppleTraderConfig(momentum_confirmation_bars=0)
        assert "confirm=" not in config_signature(config)
        assert not config.has_take

    # --- buy --------------------------------------------------------------

    def test_a_fall_to_the_buy_level_is_not_bought(self, state, market_open, monkeypatch):
        trader, tracker, tape = self._setup(state, monkeypatch)
        for close in (104.0, 103.5, 103.0):
            assert self._step(trader, tracker, tape, state, close) == "hold"
        assert self._step(trader, tracker, tape, state, 102.5) == "hold"
        assert tracker.position_for(TICKER) == 0
        (line,) = [t for t in self._status(state) if "Not buying yet" in t]
        assert "negative" in line and "$102.50 buy level" in line

    def test_a_rise_under_the_buy_level_is_bought(self, state, market_open, monkeypatch):
        _, tracker, _ = self._enter(state, monkeypatch)
        assert tracker.position_for(TICKER) > 0

    def test_neutral_momentum_turning_up_is_bought(self, state, market_open, monkeypatch):
        trader, tracker, tape = self._setup(state, monkeypatch)
        for close in (103.0, 103.0, 103.0):
            assert self._step(trader, tracker, tape, state, close) == "hold"
        # A slide under the buy level (negative momentum, not bought), then a
        # turn: over the last 3 bars the price is where it was (+0.017 a bar,
        # neutral) but the last move (+0.15) is far above the one 3 bars
        # earlier (-0.2) -- change +0.12, "about to start rising".
        for close in (102.2, 102.0, 101.95, 101.9):
            assert self._step(trader, tracker, tape, state, close) == "hold"
        assert tracker.position_for(TICKER) == 0
        assert self._step(trader, tracker, tape, state, 102.05) == "bought"

    def test_flat_at_the_buy_level_is_bought(self, state, market_open, monkeypatch):
        trader, tracker, tape = self._setup(state, monkeypatch)
        # Just above the level, then a tick onto it: -0.03 a bar and a change
        # of -0.03, both inside the +-0.05 band -- "still flat", bought.
        for close in (102.6, 102.6, 102.6, 102.6):
            assert self._step(trader, tracker, tape, state, close) == "hold"
        assert self._step(trader, tracker, tape, state, 102.5) == "bought"
        assert tracker.position_for(TICKER) > 0

    def test_nothing_is_bought_before_the_mean_move_is_known(
        self, state, market_open, monkeypatch
    ):
        trader, tracker, tape = self._setup(state, monkeypatch, minute_move=None)
        assert self._step(trader, tracker, tape, state, 101.8) == "hold"
        assert any("not known yet" in t for t in self._status(state))

    # --- sell at the sell level -------------------------------------------

    def test_the_target_is_held_while_momentum_is_positive(
        self, state, market_open, monkeypatch
    ):
        trader, tracker, tape = self._enter(state, monkeypatch)
        shares = tracker.position_for(TICKER)
        self._force(monkeypatch, "positive", "negative")
        assert self._step(trader, tracker, tape, state, 109.5, high=109.6) == "hold"
        assert tracker.position_for(TICKER) == shares
        (line,) = [t for t in self._status(state) if "Holding on" in t]
        assert "$109.00 sell level" in line and "going up" in line

    @pytest.mark.parametrize("mom", ["neutral", "negative"])
    def test_the_target_sells_once_momentum_is_not_positive(
        self, state, market_open, monkeypatch, mom
    ):
        trader, tracker, tape = self._enter(state, monkeypatch)
        self._force(monkeypatch, mom, "positive")
        assert self._step(trader, tracker, tape, state, 109.5, high=109.6) == "sold"
        assert tracker.position_for(TICKER) == 0
        assert "Target" in tracker.snapshot()["decisions"][-1].reasoning

    def test_a_real_run_up_is_held_and_then_sold(self, state, market_open, monkeypatch):
        trader, tracker, tape = self._enter(state, monkeypatch)
        for close in (104.0, 106.0):
            assert self._step(trader, tracker, tape, state, close) == "hold"
        # Through the sell level, still rising: held, and held while the last
        # 3 bars still climb.
        assert self._step(trader, tracker, tape, state, 109.5) == "hold"
        assert self._step(trader, tracker, tape, state, 109.5) == "hold"
        assert self._step(trader, tracker, tape, state, 109.5) == "hold"
        # Three bars flat at the top: neutral momentum, sold.
        assert self._step(trader, tracker, tape, state, 109.5) == "sold"
        assert "Target" in tracker.snapshot()["decisions"][-1].reasoning

    # --- the take, short of the sell level ------------------------------------

    @pytest.mark.parametrize("change", ["neutral", "negative"])
    def test_negative_momentum_takes_the_share_in_profit(
        self, state, market_open, monkeypatch, change
    ):
        trader, tracker, tape = self._enter(state, monkeypatch)
        shares = tracker.position_for(TICKER)
        self._force(monkeypatch, "negative", change)
        assert self._step(trader, tracker, tape, state, 105.0) == "sold"
        left = tracker.position_for(TICKER)
        assert 0 < left < shares
        assert "Momentum take" in tracker.snapshot()["decisions"][-1].reasoning

    def test_a_slowing_drop_is_not_taken(self, state, market_open, monkeypatch):
        trader, tracker, tape = self._enter(state, monkeypatch)
        shares = tracker.position_for(TICKER)
        self._force(monkeypatch, "negative", "positive")
        assert self._step(trader, tracker, tape, state, 105.0) == "hold"
        assert tracker.position_for(TICKER) == shares

    def test_no_take_at_a_loss(self, state, market_open, monkeypatch):
        trader, tracker, tape = self._enter(state, monkeypatch)
        shares = tracker.position_for(TICKER)
        self._force(monkeypatch, "negative", "negative")
        # Under the $101.80 fill, above the stop.
        assert self._step(trader, tracker, tape, state, 101.0) == "hold"
        assert tracker.position_for(TICKER) == shares

    def test_a_real_drop_is_taken(self, state, market_open, monkeypatch):
        trader, tracker, tape = self._enter(state, monkeypatch)
        shares = tracker.position_for(TICKER)
        for close in (104.0, 104.0, 104.0, 104.0):
            assert self._step(trader, tracker, tape, state, close) == "hold"
        # -1.00 against three flat bars: momentum -0.33 a bar, change -0.33.
        assert self._step(trader, tracker, tape, state, 103.0) == "sold"
        assert 0 < tracker.position_for(TICKER) < shares

    # --- the take at a loss -----------------------------------------------

    def test_the_take_at_a_loss_is_on_by_default_and_signed(self):
        config = AppleTraderConfig()
        assert config.take_in_loss
        assert "@conf>=15m+loss," in config_signature(config)
        assert "+loss" not in config_signature(replace(config, take_in_loss=False))
        assert "+loss" not in config_signature(
            AppleTraderConfig(momentum_confirmation_bars=0, stop_gain_fraction=0.0)
        )

    def test_the_take_at_a_loss_is_announced(self):
        config = dayrange_config(momentum_confirmation_bars=self.N, take_in_loss=True)
        summary = at._armed_summary(
            config, at.apple_models.get(config.model_key), DAYRANGE_BUNDLE
        )
        assert "take in profit or at a loss" in summary
        assert "a take at a loss leaves that to the stop" in summary

    def test_a_loss_is_taken_and_its_runner_left_to_the_stop(
        self, state, market_open, monkeypatch
    ):
        trader, tracker, tape = self._enter(state, monkeypatch, take_in_loss=True)
        shares = tracker.position_for(TICKER)
        self._force(monkeypatch, "negative", "negative")
        # Under the $101.80 fill, above the stop: the same share as in profit.
        assert self._step(trader, tracker, tape, state, 101.0) == "sold"
        left = tracker.position_for(TICKER)
        assert 0 < left < shares
        reasoning = tracker.snapshot()["decisions"][-1].reasoning
        assert "Momentum take" in reasoning and "at or under the $101.80 fill" in reasoning
        stop = 101.8 - 0.5 * TARGET_GAIN
        assert f"sold at the ${stop:,.2f} stop" in reasoning
        assert trader.entry["runner"] and trader.entry["loss_runner"]
        # Under the fill again: no breakeven for this runner, and no second take.
        assert self._step(trader, tracker, tape, state, 100.8, low=100.7) == "hold"
        assert tracker.position_for(TICKER) == left
        assert trader.plan["history"][-1]["stop"] == pytest.approx(stop)
        # The stop is its floor.
        assert self._step(trader, tracker, tape, state, 99.0, low=stop - 0.01) == "sold"
        assert tracker.position_for(TICKER) == 0
        assert "Stop loss" in tracker.snapshot()["decisions"][-1].reasoning

    def test_a_loss_runner_rides_to_the_sell_level(self, state, market_open, monkeypatch):
        trader, tracker, tape = self._enter(state, monkeypatch, take_in_loss=True)
        self._force(monkeypatch, "negative", "negative")
        assert self._step(trader, tracker, tape, state, 101.0) == "sold"
        self._force(monkeypatch, "neutral", "neutral")
        assert self._step(trader, tracker, tape, state, 109.5, high=109.6) == "sold"
        assert tracker.position_for(TICKER) == 0
        assert "Target" in tracker.snapshot()["decisions"][-1].reasoning

    def test_a_runner_kept_in_profit_still_has_its_breakeven(
        self, state, market_open, monkeypatch
    ):
        trader, tracker, tape = self._enter(state, monkeypatch, take_in_loss=True)
        self._force(monkeypatch, "negative", "negative")
        assert self._step(trader, tracker, tape, state, 105.0) == "sold"
        assert not trader.entry["loss_runner"]
        assert self._step(trader, tracker, tape, state, 102.0, low=101.7) == "sold"
        assert tracker.position_for(TICKER) == 0
        assert "Breakeven" in tracker.snapshot()["decisions"][-1].reasoning

    def test_a_loss_waits_for_the_time_too(self, state, market_open, monkeypatch):
        trader, tracker, tape = self._enter(
            state, monkeypatch, take_in_loss=True, take_after_minutes=3
        )
        shares = tracker.position_for(TICKER)
        self._force(monkeypatch, "negative", "negative")
        assert self._step(trader, tracker, tape, state, 101.0) == "hold"
        assert self._step(trader, tracker, tape, state, 101.0) == "hold"
        assert tracker.position_for(TICKER) == shares
        assert self._step(trader, tracker, tape, state, 101.0) == "sold"
        assert 0 < tracker.position_for(TICKER) < shares

    # --- the take waits for time since the fill -----------------------------

    def test_the_time_gate_is_on_by_default_and_signed(self):
        config = AppleTraderConfig()
        assert config.take_after_minutes == 15
        assert config.take_min_gain_fraction == 0.0
        assert "@conf>=15m+loss," in config_signature(config)
        assert ">=0m" not in config_signature(replace(config, take_after_minutes=0))
        assert ">=15m" not in config_signature(
            AppleTraderConfig(momentum_confirmation_bars=0, stop_gain_fraction=0.0)
        )
        with pytest.raises(ValueError, match="take_after_minutes"):
            AppleTraderConfig(take_after_minutes=-1)
        with pytest.raises(ValueError, match="must be whole"):
            AppleTraderConfig(take_after_minutes=2.5)

    def test_no_take_before_the_time_is_up(self, state, market_open, monkeypatch):
        trader, tracker, tape = self._enter(state, monkeypatch, take_after_minutes=3)
        shares = tracker.position_for(TICKER)
        self._force(monkeypatch, "negative", "negative")
        # Bought on the bar before: 1 and then 2 minutes after the fill.
        assert self._step(trader, tracker, tape, state, 105.0) == "hold"
        assert self._step(trader, tracker, tape, state, 105.0) == "hold"
        assert tracker.position_for(TICKER) == shares
        # 3 minutes after it.
        assert self._step(trader, tracker, tape, state, 105.0) == "sold"
        assert 0 < tracker.position_for(TICKER) < shares
        assert "Momentum take" in tracker.snapshot()["decisions"][-1].reasoning

    def test_the_wait_is_announced(self):
        config = dayrange_config(momentum_confirmation_bars=self.N, take_after_minutes=15)
        summary = at._armed_summary(
            config, at.apple_models.get(config.model_key), DAYRANGE_BUNDLE
        )
        assert "take in profit (not before 15 min after the fill)" in summary

    # --- the legacy gate: a share of the predicted gain ---------------------

    def test_the_legacy_gain_gate_is_signed_when_set(self):
        config = AppleTraderConfig(
            take_min_gain_fraction=0.2, take_after_minutes=0, take_in_loss=False
        )
        assert "@conf>=0.2G," in config_signature(config)
        assert ">=0G" not in config_signature(AppleTraderConfig())
        with pytest.raises(ValueError, match="take_min_gain_fraction"):
            AppleTraderConfig(take_min_gain_fraction=-0.1)

    def test_no_take_short_of_the_gain_gate(self, state, market_open, monkeypatch):
        # The predicted gain is $109.00 - $102.50 = $6.50, so 0.2 of it is
        # $1.30 over the $101.80 fill: $103.10.
        trader, tracker, tape = self._enter(state, monkeypatch, take_min_gain_fraction=0.2)
        shares = tracker.position_for(TICKER)
        self._force(monkeypatch, "negative", "negative")
        assert self._step(trader, tracker, tape, state, 103.0) == "hold"
        assert tracker.position_for(TICKER) == shares

    def test_the_take_fires_past_the_gain_gate(self, state, market_open, monkeypatch):
        trader, tracker, tape = self._enter(state, monkeypatch, take_min_gain_fraction=0.2)
        shares = tracker.position_for(TICKER)
        self._force(monkeypatch, "negative", "negative")
        assert self._step(trader, tracker, tape, state, 103.2) == "sold"
        assert 0 < tracker.position_for(TICKER) < shares
        assert "Momentum take" in tracker.snapshot()["decisions"][-1].reasoning

    # --- never gated ------------------------------------------------------

    def test_the_stop_is_not_gated(self, state, market_open, monkeypatch):
        trader, tracker, tape = self._enter(state, monkeypatch)
        self._force(monkeypatch, "positive", "positive")
        stop = 101.8 - 0.5 * TARGET_GAIN
        assert self._step(trader, tracker, tape, state, 102.0, low=stop - 0.01) == "sold"
        assert "Stop loss" in tracker.snapshot()["decisions"][-1].reasoning

    def test_the_flatten_is_not_gated(self, state, market_open, monkeypatch):
        trader, tracker, tape = self._enter(state, monkeypatch)
        self._force(monkeypatch, "positive", "positive")
        monkeypatch.setattr(trader, "closing_soon", lambda: True)
        assert self._step(trader, tracker, tape, state, 104.0) == "sold"
        assert "flattened" in tracker.snapshot()["decisions"][-1].reasoning


class TestNoBuyIntoAFall:
    """No buy -- first or add -- while the price is falling too fast to catch.

    Notebook arithmetic (H $110, ADR $10, buy $102.50) at `max_fall_k` 0.30:
    a close more than $3.00 under the close 15 bars earlier refuses the bar.
    """

    def _trader(self, **kwargs):
        kwargs.setdefault("max_fall_k", 0.30)
        return at.DayRangeTrader(dayrange_config(**kwargs))

    def _run(self, state, monkeypatch, pad: float, pad_bars: int = 20, **kwargs):
        """`pad_bars` bars at `pad`, then one at the buy level; the outcome of that last one."""
        broker = FakeBroker(pad)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tape = Tape(monkeypatch, broker)
        trader = self._trader(**kwargs)
        for _ in range(pad_bars):
            tape.append(pad)
            assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "hold"
        tape.append(BUY_LEVEL)
        outcome = trader.run_cycle(DAYRANGE_BUNDLE, state, tracker)
        return outcome, trader, tracker, tape

    def _status(self, state):
        return [e["text"] for e in state.agent_log if e.get("type") == "status"]

    def test_a_new_config_leaves_it_to_the_momentum_confirmation(self):
        # The legacy gate: records from 2026-09-23 to -24 replay it, nothing
        # new sets it, and it cannot sit beside the confirmation.
        assert AppleTraderConfig().max_fall_k == 0.0
        assert AppleTraderConfig().momentum_confirmation_bars > 0
        with pytest.raises(ValueError, match="momentum_confirmation_bars"):
            AppleTraderConfig(max_fall_k=0.1)

    def test_a_steep_fall_to_the_buy_level_is_not_bought(self, state, market_open, monkeypatch):
        # $106.00 -> $102.50 is $3.50 down over 15 bars: 0.35 ADR, past 0.30.
        outcome, _, tracker, _ = self._run(state, monkeypatch, pad=106.0)
        assert outcome == "hold"
        assert tracker.position_for(TICKER) == 0
        (line,) = [t for t in self._status(state) if "Not buying into the fall" in t]
        assert "$3.50" in line and "15 bars" in line and "-0.35" in line

    def test_a_gentle_fall_to_the_buy_level_is_bought(self, state, market_open, monkeypatch):
        # $104.50 -> $102.50 is $2.00: 0.20 ADR, inside 0.30.
        outcome, _, tracker, _ = self._run(state, monkeypatch, pad=104.5)
        assert outcome == "bought"
        assert tracker.position_for(TICKER) > 0

    def test_off_at_zero(self, state, market_open, monkeypatch):
        outcome, *_ = self._run(state, monkeypatch, pad=106.0, max_fall_k=0.0)
        assert outcome == "bought"

    def test_only_the_bar_is_refused_and_the_buy_follows_once_the_fall_is_out_of_the_window(
        self, state, market_open, monkeypatch
    ):
        outcome, trader, tracker, tape = self._run(state, monkeypatch, pad=106.0)
        assert outcome == "hold"
        outcomes = []
        for _ in range(15):
            tape.append(BUY_LEVEL)
            outcomes.append(trader.run_cycle(DAYRANGE_BUNDLE, state, tracker))
        # The $106 closes drop out of the 15-bar look-back one bar at a time;
        # the first bar whose look-back starts at $102.50 buys.
        assert outcomes == ["hold"] * 14 + ["bought"]

    def test_the_look_back_is_the_momentum_takes(self, state, market_open, monkeypatch):
        # The fall to $102.60 is six bars old: inside 15 bars, outside 5.
        def run(**kwargs):
            broker = FakeBroker(106.0)
            tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
            tape = Tape(monkeypatch, broker)
            trader = self._trader(**kwargs)
            for close in [106.0] * 20 + [102.6] * 6:
                tape.append(close)
                trader.run_cycle(DAYRANGE_BUNDLE, state, tracker)
            tape.append(BUY_LEVEL)
            return trader.run_cycle(DAYRANGE_BUNDLE, state, tracker)

        assert run(negative_momentum_bars=15) == "hold"
        assert run(negative_momentum_bars=5) == "bought"

    def test_with_the_take_off_the_look_back_is_the_default(self):
        from agent_stonks.config import APPLE_TRADER_NEGATIVE_MOMENTUM_BARS

        assert (
            dayrange_config(negative_momentum_bars=0).fall_bars
            == APPLE_TRADER_NEGATIVE_MOMENTUM_BARS
        )
        assert dayrange_config(negative_momentum_bars=8).fall_bars == 8
        # A record replaying the legacy turn keeps the look-back it signed.
        assert dayrange_config(negative_momentum_bars=0, momentum_fade_bars=12).fall_bars == 12

    def test_an_add_is_refused_too(self, state, market_open, monkeypatch):
        broker = FakeBroker(BUY_LEVEL)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tape = Tape(monkeypatch, broker)
        # $1.00 allowed: the first buy comes up from the opening window, the
        # add falls $1.75 from $103.00 to the $101.25 next rung.
        trader = self._trader(max_fall_k=0.10, scale_in=True, position_pct=50.0)
        tape.append(BUY_LEVEL)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "bought"
        first = tracker.position_for(TICKER)
        for _ in range(20):
            tape.append(103.0)
            trader.run_cycle(DAYRANGE_BUNDLE, state, tracker)
        tape.append(TestScaleIn.RUNG_2)
        assert trader.run_cycle(DAYRANGE_BUNDLE, state, tracker) == "hold"
        assert tracker.position_for(TICKER) == first
        assert any("Not buying into the fall" in t for t in self._status(state))

    def test_negative_is_refused(self):
        with pytest.raises(ValueError, match="max_fall_k"):
            dayrange_config(max_fall_k=-0.1)

    def test_in_the_signature_only_while_on(self):
        on = config_signature(dayrange_config(max_fall_k=0.3, negative_momentum_bars=12))
        assert ",nofall=0.3A/12b" in on
        assert "nofall" not in config_signature(dayrange_config())

    def test_a_record_from_before_it_existed_replays_with_it_off(self):
        from simlab.rule_agents import _apple_from_record

        assert _apple_from_record({}).max_fall_k == 0.0
        assert _apple_from_record({"max_fall_k": 0.25}).max_fall_k == 0.25


# --------------------------------------------------------------------------
# HighLow_3m's 9:33 window (`use_3m`).
# --------------------------------------------------------------------------

EARLY_BUNDLE = {"kind": "highlow3m", "opening_minutes": 3}
# HighLow_3m's 9:33 forecast of the rest of the session: a $10 ADR -- the unit
# `dayrange_config` pins -- under a predicted high of $106, so at the 0.40 /
# 0.25 the window is pinned to it buys at $102.00 and sells at $103.50. The
# 9:35 forecast is `FORECAST`: buy $102.50, sell $109.00.
EARLY_FORECAST = {
    "pred_high": 106.0,
    "pred_low": 98.0,
    "prev_avg": 102.0,
    "adr14_abs": 10.0,
    "or_high": 101.5,
    "or_low": 100.9,
    "range_after_opening": True,
}
EARLY_BUY, EARLY_SELL = 102.0, 103.5


class TestHighLow3mWindow:
    """The 09:33 and 09:34 bars traded on HighLow_3m's 9:33 forecast before the
    run's own 9:35 one exists: its own buy and sell, no stop, no momentum
    confirmation, no take -- and a position still open at 9:35 handed to the
    9:35 rules as if it had been bought then."""

    FILL = 101.8
    # The 9:35 rules' stop for that fill: half the 0.65-ADR predicted gain.
    STOP = FILL - 0.5 * TARGET_GAIN

    def _setup(self, monkeypatch, early=None, **kwargs):
        kwargs.setdefault("use_3m", True)
        kwargs.setdefault("buy_3m_k", 0.40)
        kwargs.setdefault("sell_3m_k", 0.25)
        broker = FakeBroker(101.2)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tape = Tape(monkeypatch, broker, minutes=3)
        calls: list[int] = []

        def forecast(bundle, ticker, opening, today, key=None, secret=None):
            calls.append(len(opening))
            if isinstance(early, Exception):
                raise early
            return dict(EARLY_FORECAST)

        stub = type("Stub", (), {})()
        stub.forecast_session = forecast
        stub.warm_history = lambda *a, **k: None
        monkeypatch.setattr(at, "_highlow3m", lambda: stub)
        real = apple_models.load
        monkeypatch.setattr(
            apple_models, "load",
            lambda key, ticker=None: dict(EARLY_BUNDLE)
            if key == apple_models.HIGHLOW3M_KEY else real(key, ticker),
        )
        # The opening window is never re-fetched from the network here.
        monkeypatch.setattr(at.agent_mod, "fetch_bars_window", lambda *a, **k: [])
        trader = at.DayRangeTrader(dayrange_config(**kwargs))
        return trader, tracker, tape, broker, calls

    @staticmethod
    def cycle(trader, state, tracker) -> str:
        return trader.run_cycle(DAYRANGE_BUNDLE, state, tracker)

    def _bought_at_0933(self, state, monkeypatch, **kwargs):
        trader, tracker, tape, broker, calls = self._setup(monkeypatch, **kwargs)
        assert self.cycle(trader, state, tracker) == "warming_up"
        tape.append(self.FILL, low=101.7, high=102.1, offset=3)
        assert self.cycle(trader, state, tracker) == "bought"
        return trader, tracker, tape, broker, calls

    @staticmethod
    def _decisions(state) -> list[dict]:
        return [e for e in state.agent_log if e.get("type") == "decision"]

    def test_the_forecast_is_made_on_the_0932_bar_and_not_traded_on_it(
        self, state, market_open, monkeypatch
    ):
        trader, tracker, tape, _, calls = self._setup(monkeypatch)
        tape.rows[-1]["low"] = EARLY_BUY - 5  # deep enough to fill, if it counted
        assert self.cycle(trader, state, tracker) == "warming_up"
        assert calls == [3] and trader.plan is None
        assert trader.early["buy_level"] == pytest.approx(EARLY_BUY)
        assert trader.early["sell_level"] == pytest.approx(EARLY_SELL)
        assert tracker.position_for(TICKER) == 0

    def test_the_0933_bar_buys_on_highlow3ms_level_with_no_stop(
        self, state, market_open, monkeypatch
    ):
        trader, tracker, *_ = self._bought_at_0933(state, monkeypatch)
        assert trader.entry["early"] and trader.entry["price"] == pytest.approx(self.FILL)
        assert trader._stop_price() is None
        reasoning = tracker.snapshot()["decisions"][-1].reasoning
        assert "HighLow_3m window" in reasoning and "$102.00" in reasoning

    def test_a_position_open_at_935_is_handed_to_the_935_rules(
        self, state, market_open, monkeypatch
    ):
        trader, tracker, tape, _, _ = self._bought_at_0933(state, monkeypatch)
        tape.append(102.4, low=102.0, high=102.9, offset=4)
        assert self.cycle(trader, state, tracker) == "hold"
        assert trader.plan is not None
        assert "early" not in trader.entry
        assert trader.entry["stop"] == pytest.approx(self.STOP)
        assert trader.entry["ts"] == Tape.OPEN + pd.Timedelta(minutes=3)
        handed = [e["text"] for e in state.agent_log if "are now managed by" in e.get("text", "")]
        assert handed and f"stop ${self.STOP:,.2f}" in handed[0]

        tape.append(109.1, low=108.5, high=109.2, offset=5)
        assert self.cycle(trader, state, tracker) == "sold"
        assert self._decisions(state)[-1]["exit"] == at.DayRangeTrader.EXIT_TARGET

    def test_there_is_no_stop_inside_the_window_and_the_935_one_applies_after(
        self, state, market_open, monkeypatch
    ):
        trader, tracker, tape, _, _ = self._bought_at_0933(state, monkeypatch)
        tape.append(97.0, low=96.5, high=101.0, offset=4)  # well through any stop
        assert self.cycle(trader, state, tracker) == "hold"
        assert tracker.position_for(TICKER) > 0
        tape.append(96.8, low=96.4, high=97.2, offset=5)
        assert self.cycle(trader, state, tracker) == "sold"
        assert self._decisions(state)[-1]["exit"] == at.DayRangeTrader.EXIT_STOP

    def test_the_window_sells_at_its_own_level_without_tripping_the_breaker(
        self, state, market_open, monkeypatch
    ):
        """A $0.10-a-share round trip is under any breaker, but the breaker is
        about the 9:35 levels: the window's trade leaves them armed."""
        trader, tracker, tape, _, _ = self._bought_at_0933(
            state, monkeypatch, min_win_k=0.5
        )
        tape.append(103.6, low=103.0, high=103.7, offset=4)
        assert self.cycle(trader, state, tracker) == "sold"
        assert tracker.position_for(TICKER) == 0
        assert self._decisions(state)[-1]["exit"] == at.DayRangeTrader.EXIT_EARLY_TARGET
        assert trader.plan is not None and not trader.plan.get("stand_down")
        assert trader.entry is None

        tape.append(BUY_LEVEL, offset=5)
        assert self.cycle(trader, state, tracker) == "bought"

    def test_no_momentum_confirmation_inside_the_window(
        self, state, market_open, monkeypatch
    ):
        """With the confirmation on and nothing to read it against yet, the
        9:35 rules would hold every buy; the window does not read it."""
        trader, tracker, *_ = self._bought_at_0933(
            state, monkeypatch, momentum_confirmation_bars=3
        )
        assert tracker.position_for(TICKER) > 0

    def test_a_limit_the_price_recovered_from_leaves_the_level_armed(
        self, state, market_open, monkeypatch
    ):
        trader, tracker, tape, _, _ = self._setup(monkeypatch, limit_entry=True)
        self.cycle(trader, state, tracker)
        tape.append(102.3, low=101.9, high=102.4, offset=3)
        assert self.cycle(trader, state, tracker) == "hold"
        decision = tracker.snapshot()["decisions"][-1]
        assert decision.limit_missed and decision.limit_price == pytest.approx(EARLY_BUY)
        tape.append(101.95, low=101.9, high=102.2, offset=4)
        assert self.cycle(trader, state, tracker) == "bought"
        assert tracker.snapshot()["decisions"][-1].price == pytest.approx(101.95)

    def test_off_by_default_and_nothing_is_bought_before_935(
        self, state, market_open, monkeypatch
    ):
        assert AppleTraderConfig().use_3m is False
        trader, tracker, tape, _, calls = self._setup(monkeypatch, use_3m=False)
        assert self.cycle(trader, state, tracker) == "warming_up"
        tape.append(self.FILL, low=101.7, high=102.1, offset=3)
        assert self.cycle(trader, state, tracker) == "warming_up"
        assert tracker.position_for(TICKER) == 0 and calls == []

    def test_a_run_started_after_the_window_never_trades_it(
        self, state, market_open, monkeypatch
    ):
        trader, tracker, tape, _, calls = self._setup(monkeypatch)
        for offset in (3, 4, 5):
            tape.append(101.0, low=99.0, offset=offset)
        self.cycle(trader, state, tracker)
        assert calls == [] and trader.early is None and trader.plan is not None

    def test_a_failed_933_forecast_costs_only_the_window(
        self, state, market_open, monkeypatch
    ):
        trader, tracker, tape, _, _ = self._setup(
            monkeypatch, early=RuntimeError("no option tables")
        )
        assert self.cycle(trader, state, tracker) == "warming_up"
        errors = [e["text"] for e in state.agent_log if e.get("type") == "error"]
        assert "nothing is traded before the 9:35 forecast" in errors[-1]
        tape.append(self.FILL, low=101.7, high=102.1, offset=3)
        assert self.cycle(trader, state, tracker) == "warming_up"
        tape.append(102.4, low=102.0, high=102.9, offset=4)
        self.cycle(trader, state, tracker)
        assert trader.plan is not None and tracker.position_for(TICKER) == 0

    def test_no_935_forecast_sells_what_the_window_bought(
        self, state, market_open, monkeypatch
    ):
        """Without a 9:35 forecast there is nothing to hand the position to, and
        it has no stop of its own: sold rather than left until the close."""
        trader, tracker, tape, _, _ = self._bought_at_0933(state, monkeypatch)

        def boom(*a, **k):
            raise RuntimeError("history too short")

        monkeypatch.setattr(at._dayrange(), "forecast_session", boom)
        tape.append(102.4, low=102.0, high=102.9, offset=4)
        assert self.cycle(trader, state, tracker) == "no_data"
        assert tracker.position_for(TICKER) == 0
        assert self._decisions(state)[-1]["exit"] == at.DayRangeTrader.EXIT_EARLY_DROPPED

    def test_the_windows_levels_open_the_record_the_chart_draws(
        self, state, market_open, monkeypatch
    ):
        trader, tracker, tape, _, _ = self._bought_at_0933(state, monkeypatch)
        tape.append(102.4, low=102.0, high=102.9, offset=4)
        self.cycle(trader, state, tracker)
        tape.append(102.6, offset=5)
        self.cycle(trader, state, tracker)
        rows = state.apple_trader_levels["rows"]
        assert [(r["t"].minute, r["model_key"]) for r in rows] == [
            (32, "highlow3m"), (33, "highlow3m"), (34, "dayrange"), (35, "dayrange"),
        ]
        assert rows[0]["buy"] == pytest.approx(EARLY_BUY)
        assert rows[0]["stop"] is None and rows[-1]["stop"] == pytest.approx(self.STOP)


class TestHighLow3mWindowConfig:
    def test_the_distances_start_from_highlow3ms_own_pair(self):
        config = AppleTraderConfig(model_key="highlow")
        assert (config.buy_3m_k, config.sell_3m_k) == at.dayrange_levels(
            TICKER, apple_models.HIGHLOW3M_KEY
        )

    def test_a_crossed_pair_is_refused(self):
        with pytest.raises(ValueError, match="sell_3m_k"):
            AppleTraderConfig(model_key="highlow", buy_3m_k=0.2, sell_3m_k=0.3)
        with pytest.raises(ValueError, match="buy_3m_k"):
            AppleTraderConfig(model_key="highlow", buy_3m_k=-0.1, sell_3m_k=-0.2)

    def test_signed_only_while_on(self):
        on = config_signature(dayrange_config(use_3m=True, buy_3m_k=0.4, sell_3m_k=0.25))
        assert ",3m=H-0.4A/H-0.25A," in on
        assert "3m=" not in config_signature(dayrange_config(buy_3m_k=0.6, sell_3m_k=0.1))

    def test_a_record_from_before_it_replays_with_it_off(self):
        from simlab.rule_agents import _apple_from_record

        old = _apple_from_record({"model_key": "highlow"})
        assert old.use_3m is False and "3m=" not in config_signature(old)

    def test_the_armed_line_says_so(self):
        config = dayrange_config(use_3m=True)
        line = at._armed_summary(config, apple_models.get("dayrange"), DAYRANGE_BUNDLE)
        assert "from 9:33 it trades HighLow_3m's 9:33 forecast" in line
        assert "9:33" not in at._armed_summary(
            dayrange_config(), apple_models.get("dayrange"), DAYRANGE_BUNDLE
        )

    def test_where_it_cannot_run_is_said_before_the_run(self, monkeypatch):
        real = apple_models.load
        present = {"bundle": dict(EARLY_BUNDLE)}
        monkeypatch.setattr(
            apple_models, "load",
            lambda key, ticker=None: present["bundle"]
            if key == apple_models.HIGHLOW3M_KEY else real(key, ticker),
        )
        ok = AppleTraderConfig(model_key="highlow", use_3m=True)
        assert at.early_window_error(ok) is None
        assert at.early_window_error(replace(ok, use_3m=False)) is None
        on_itself = AppleTraderConfig(model_key="highlow3m", use_3m=True)
        assert "forecasts at 9:33" in at.config_error(on_itself)
        intc = AppleTraderConfig(model_key="highlow", ticker="INTC", use_3m=True)
        assert "fitted on AAPL only" in at.early_window_error(intc)
        present["bundle"] = None
        assert "cannot run" in at.early_window_error(ok)
