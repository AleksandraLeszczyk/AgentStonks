"""Apple Trader: the rule-based loop over the momentum-persistence model.

Every cycle is driven through a stubbed model read, so these pin the RULES --
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
from agent_stonks.apple_trader import AppleTrader, AppleTraderConfig, config_signature
from agent_stonks.broker import Broker
from agent_stonks.decisions import DecisionTracker
from agent_stonks.state import AppState

# 10:30 ET on a Tuesday: mid-session, well clear of both the open and the close.
MIDSESSION = datetime(2026, 7, 21, 14, 30, tzinfo=timezone.utc)

BUNDLE = {"pipeline": None, "feature_columns": [], "seq_len": 20, "threshold": 0.07}


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


class Reads:
    """Feeds the trader a scripted sequence of model reads, one per cycle."""

    def __init__(self, monkeypatch, broker: "FakeBroker | None" = None):
        self.broker = broker
        self.minute = 0
        self.next_read: dict = {}
        monkeypatch.setattr(
            at.persistence_model, "minute_frame", lambda *a, **k: pd.DataFrame({"x": [1]})
        )
        monkeypatch.setattr(at.persistence_model, "read_latest", lambda *a, **k: self.next_read)

    def set(
        self,
        *,
        price: float = 100.0,
        high: "float | None" = None,
        mom: float = 1.2,
        regime: int = 1,
        prev_regime: "int | None" = 0,
        change: bool = False,
        proba: "float | None" = None,
        turn_proba: "float | None" = None,
        reversal_proba: "float | None" = None,
        pre_dwell: "int | None" = 20,
        bars_in_regime: int = 20,
        bars_today: int = 200,
        warming_up: bool = False,
        advance: bool = True,
    ) -> dict:
        """Stage the next bar. `advance=False` replays the SAME timestamp, the
        way a cycle that runs before a new bar has closed would see it."""
        if advance:
            self.minute += 1
        if self.broker is not None:
            self.broker.price = price
        self.next_read = {
            "ts": pd.Timestamp("2026-07-21 10:30", tz="America/New_York")
            + pd.Timedelta(minutes=self.minute),
            "price": price,
            "high": price if high is None else high,
            "mom": mom,
            "regime": regime,
            "prev_regime": prev_regime,
            "regime_change": change,
            "to_positive": change and regime == 1,
            "pre_dwell": pre_dwell if change else None,
            "bars_in_regime": bars_in_regime,
            "proba": proba,
            "turn_proba": turn_proba,
            "reversal_proba": reversal_proba,
            "bars_today": bars_today,
            "warming_up": warming_up,
        }
        return self.next_read

    def to_positive(self, *, proba: float, **kwargs) -> dict:
        """The bar `confirm` acts on: a change into positive, already printed."""
        return self.set(regime=1, prev_regime=0, change=True, proba=proba, **kwargs)

    def pre_turn(self, *, turn_proba: "float | None", regime: int = 0, **kwargs) -> dict:
        """The bar `anticipate` acts on: the regime has NOT turned positive, and
        the forecaster has been asked whether it is about to.

        `read_latest` only fills `turn_proba` in on such a bar, so a stub that
        set it beside `regime=1` would be testing a state the pipeline cannot
        produce.
        """
        return self.set(regime=regime, prev_regime=regime, turn_proba=turn_proba, **kwargs)


def confirm_config(**kwargs) -> AppleTraderConfig:
    """A rule set on the `confirm` entry, which is no longer the default.

    The trailing stop, the flatten rule and the guards are shared by both entry
    modes, so the suites below pin them through the mode whose trigger is the
    notebook's and whose stub is a single scripted bar.
    """
    return AppleTraderConfig(entry_mode=at.ENTRY_CONFIRM, **kwargs)


class TestEntry:
    """The `confirm` trigger: buy a change into positive that has already
    printed, if the model rates it likely to hold."""

    def _trader(self, **kwargs) -> AppleTrader:
        return AppleTrader(confirm_config(**kwargs), model_threshold=0.07)

    def test_buys_a_to_positive_change_the_model_backs(self, state, market_open, monkeypatch):
        broker = FakeBroker(100.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        reads = Reads(monkeypatch, broker)
        trader = self._trader(prob_threshold=0.5)

        reads.to_positive(proba=0.62)
        assert trader.run_cycle(BUNDLE, state, tracker) == "bought"
        assert tracker.position_for(TICKER) > 0
        assert "62%" in tracker.snapshot()["decisions"][-1].reasoning

    def test_a_probability_below_the_threshold_is_not_a_buy(self, state, market_open, monkeypatch):
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(100.0))
        reads = Reads(monkeypatch)
        trader = self._trader(prob_threshold=0.5)

        reads.to_positive(proba=0.49)
        assert trader.run_cycle(BUNDLE, state, tracker) == "hold"
        assert tracker.position_for(TICKER) == 0

    def test_the_bundles_own_threshold_applies_when_none_is_configured(
        self, state, market_open, monkeypatch
    ):
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(100.0))
        reads = Reads(monkeypatch)
        trader = AppleTrader(confirm_config(prob_threshold=None), model_threshold=0.07)
        assert trader.prob_threshold == pytest.approx(0.07)

        reads.to_positive(proba=0.10)  # under any sane default, over this model's
        assert trader.run_cycle(BUNDLE, state, tracker) == "bought"

    def test_a_change_out_of_positive_is_not_an_entry(self, state, market_open, monkeypatch):
        """Only changes INTO the positive regime are ever traded; the model is
        not even asked about the others, so `proba` is None on them."""
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(100.0))
        reads = Reads(monkeypatch)
        trader = self._trader(prob_threshold=0.5)

        for regime, prev in ((0, 1), (-1, 1)):
            reads.set(regime=regime, prev_regime=prev, change=True, proba=None)
            assert trader.run_cycle(BUNDLE, state, tracker) == "hold"
        assert tracker.position_for(TICKER) == 0

    def test_sitting_in_the_positive_regime_is_not_a_change(self, state, market_open, monkeypatch):
        """The signal is the transition, not the state: momentum can be
        strongly positive for an hour without the loop ever buying."""
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(100.0))
        reads = Reads(monkeypatch)
        trader = self._trader(prob_threshold=0.5)

        for _ in range(5):
            reads.set(regime=1, change=False, mom=2.5, proba=0.99)
            assert trader.run_cycle(BUNDLE, state, tracker) == "hold"
        assert tracker.position_for(TICKER) == 0

    def test_an_unscoreable_change_is_not_a_buy(self, state, market_open, monkeypatch):
        """A change whose 20-bar feature window hasn't warmed up leaves `proba`
        None. An unasked model is not a yes."""
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(100.0))
        reads = Reads(monkeypatch)
        trader = self._trader(prob_threshold=0.5)

        reads.to_positive(proba=None)
        assert trader.run_cycle(BUNDLE, state, tracker) == "hold"
        assert tracker.position_for(TICKER) == 0

    def test_does_not_trade_during_the_models_warm_up(self, state, market_open, monkeypatch):
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(100.0))
        reads = Reads(monkeypatch)
        trader = self._trader(prob_threshold=0.5)

        reads.to_positive(proba=None, warming_up=True, bars_today=12)
        assert trader.run_cycle(BUNDLE, state, tracker) == "warming_up"
        assert tracker.position_for(TICKER) == 0

    def test_re_reading_the_same_bar_does_not_re_enter(self, state, market_open, monkeypatch):
        """One closed bar is one decision. A cycle that runs before the next bar
        has arrived must not act on the same change twice."""
        broker = FakeBroker(100.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        reads = Reads(monkeypatch, broker)
        trader = self._trader(prob_threshold=0.5, trail_pct=0.5)

        reads.to_positive(proba=0.9)
        assert trader.run_cycle(BUNDLE, state, tracker) == "bought"
        # Same bar, deeper drawdown than the stop allows: the exit fires...
        reads.to_positive(proba=0.9, price=99.0, advance=False)
        assert trader.run_cycle(BUNDLE, state, tracker) == "sold"
        # ...and the stale bar cannot immediately buy the same change back.
        assert trader.run_cycle(BUNDLE, state, tracker) == "hold"
        assert tracker.position_for(TICKER) == 0

    def test_position_size_follows_the_configured_share_of_cash(
        self, state, market_open, monkeypatch
    ):
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(100.0), trade_cost=0.0)
        reads = Reads(monkeypatch)
        trader = self._trader(prob_threshold=0.5, position_pct=50.0)

        reads.to_positive(proba=0.9, price=100.0)
        trader.run_cycle(BUNDLE, state, tracker)
        assert tracker.position_for(TICKER) == pytest.approx(50.0)

    def test_no_entry_inside_the_closing_flatten_window(self, state, market_open, monkeypatch):
        """The notebook's simulator makes no decision on the last bar of a
        session; the same reasoning ends this loop's entries once the flatten
        rule is in force, since the position would be sold straight back."""
        broker = FakeBroker(100.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        reads = Reads(monkeypatch, broker)
        trader = self._trader(prob_threshold=0.5, flatten_before_close_min=5)

        clock.set_simulated(datetime(2026, 7, 21, 19, 57, tzinfo=timezone.utc))  # 15:57 ET
        reads.to_positive(proba=0.99)
        assert trader.run_cycle(BUNDLE, state, tracker) == "hold"
        assert tracker.position_for(TICKER) == 0
        assert any("Standing down" in e.get("text", "") for e in state.agent_log)

    def test_a_signal_just_before_the_window_still_trades(self, state, market_open, monkeypatch):
        broker = FakeBroker(100.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        reads = Reads(monkeypatch, broker)
        trader = self._trader(prob_threshold=0.5, flatten_before_close_min=5)

        clock.set_simulated(datetime(2026, 7, 21, 19, 54, tzinfo=timezone.utc))  # 15:54 ET
        reads.to_positive(proba=0.99)
        assert trader.run_cycle(BUNDLE, state, tracker) == "bought"

    def test_no_second_entry_while_already_long(self, state, market_open, monkeypatch):
        broker = FakeBroker(100.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        reads = Reads(monkeypatch, broker)
        trader = self._trader(prob_threshold=0.5)

        reads.to_positive(proba=0.9)
        trader.run_cycle(BUNDLE, state, tracker)
        size = tracker.position_for(TICKER)
        for _ in range(3):
            reads.to_positive(proba=0.9, price=100.5)
            trader.run_cycle(BUNDLE, state, tracker)
        assert tracker.position_for(TICKER) == size


class TestAnticipateEntry:
    """The default trigger: buy while the regime is still negative or balanced,
    on the forecast that it turns positive next bar.

    The whole point of this mode is *where in the transition* the order goes
    in, so these pin which bars can and cannot produce one -- not just the
    threshold arithmetic.
    """

    def _trader(self, **kwargs) -> AppleTrader:
        return AppleTrader(
            AppleTraderConfig(entry_mode=at.ENTRY_ANTICIPATE, **kwargs),
            model_threshold=0.05,
        )

    def test_buys_a_balanced_bar_the_forecast_expects_to_turn(
        self, state, market_open, monkeypatch
    ):
        broker = FakeBroker(100.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        reads = Reads(monkeypatch, broker)
        trader = self._trader(prob_threshold=0.2)

        reads.pre_turn(turn_proba=0.45, regime=0, mom=0.8)
        assert trader.run_cycle(BUNDLE, state, tracker) == "bought"
        assert tracker.position_for(TICKER) > 0

    def test_buys_out_of_the_negative_regime_too(self, state, market_open, monkeypatch):
        broker = FakeBroker(100.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        reads = Reads(monkeypatch, broker)
        trader = self._trader(prob_threshold=0.2)

        reads.pre_turn(turn_proba=0.31, regime=-1, mom=-0.5)
        assert trader.run_cycle(BUNDLE, state, tracker) == "bought"
        assert tracker.position_for(TICKER) > 0

    def test_the_regime_is_not_yet_positive_when_the_order_goes_in(
        self, state, market_open, monkeypatch
    ):
        """The regression this mode exists for. `confirm` can only ever buy a
        bar whose regime has already turned positive, which puts the entry
        after the momentum score has crossed its threshold and after the move
        that pushed it there. Here the ledger records a bar that has not."""
        broker = FakeBroker(100.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        reads = Reads(monkeypatch, broker)
        trader = self._trader(prob_threshold=0.2)

        read = reads.pre_turn(turn_proba=0.45, regime=0, mom=0.8, bars_in_regime=44)
        assert trader.run_cycle(BUNDLE, state, tracker) == "bought"
        assert read["regime"] != 1
        reasoning = tracker.snapshot()["decisions"][-1].reasoning
        assert "still balanced" in reasoning
        assert "held 44 bars" in reasoning
        assert "45%" in reasoning

    def test_a_forecast_below_the_threshold_is_not_a_buy(
        self, state, market_open, monkeypatch
    ):
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(100.0))
        reads = Reads(monkeypatch)
        trader = self._trader(prob_threshold=0.2)

        reads.pre_turn(turn_proba=0.19)
        assert trader.run_cycle(BUNDLE, state, tracker) == "hold"
        assert tracker.position_for(TICKER) == 0

    def test_an_unscoreable_bar_is_not_a_buy(self, state, market_open, monkeypatch):
        """A bar whose 20-bar window hasn't warmed up, or a bundle that cannot
        forecast at all, leaves `turn_proba` None. An unasked model is not a
        yes -- and it must not fall through to the persistence answer."""
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(100.0))
        reads = Reads(monkeypatch)
        trader = self._trader(prob_threshold=0.2)

        reads.pre_turn(turn_proba=None, proba=0.99)
        assert trader.run_cycle(BUNDLE, state, tracker) == "hold"
        assert tracker.position_for(TICKER) == 0

    def test_the_confirmed_change_is_no_longer_an_entry(
        self, state, market_open, monkeypatch
    ):
        """Once the change has printed, this mode has missed it and says so by
        standing aside -- it does not chase the bar `confirm` would have
        bought."""
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(100.0))
        reads = Reads(monkeypatch)
        trader = self._trader(prob_threshold=0.2)

        reads.to_positive(proba=0.99)
        assert trader.run_cycle(BUNDLE, state, tracker) == "hold"
        assert tracker.position_for(TICKER) == 0

    def test_no_entry_inside_the_closing_flatten_window(
        self, state, market_open, monkeypatch
    ):
        broker = FakeBroker(100.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        reads = Reads(monkeypatch, broker)
        trader = self._trader(prob_threshold=0.2, flatten_before_close_min=5)

        clock.set_simulated(datetime(2026, 7, 21, 19, 57, tzinfo=timezone.utc))  # 15:57 ET
        reads.pre_turn(turn_proba=0.99)
        assert trader.run_cycle(BUNDLE, state, tracker) == "hold"
        assert tracker.position_for(TICKER) == 0

    def test_re_reading_the_same_bar_does_not_re_enter(
        self, state, market_open, monkeypatch
    ):
        broker = FakeBroker(100.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        reads = Reads(monkeypatch, broker)
        trader = self._trader(prob_threshold=0.2, trail_pct=0.5)

        reads.pre_turn(turn_proba=0.9)
        assert trader.run_cycle(BUNDLE, state, tracker) == "bought"
        reads.pre_turn(turn_proba=0.9, price=99.0, advance=False)
        assert trader.run_cycle(BUNDLE, state, tracker) == "sold"
        assert trader.run_cycle(BUNDLE, state, tracker) == "hold"
        assert tracker.position_for(TICKER) == 0

    def test_the_trailing_stop_is_the_exit_here_too(
        self, state, market_open, monkeypatch
    ):
        """Nothing about the exit changes with the entry mode: once the
        position is on, only price decides when it comes off."""
        broker = FakeBroker(100.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        reads = Reads(monkeypatch, broker)
        trader = self._trader(prob_threshold=0.2, trail_pct=0.5)

        reads.pre_turn(turn_proba=0.9, price=100.0)
        assert trader.run_cycle(BUNDLE, state, tracker) == "bought"
        reads.set(price=102.0, regime=1)
        assert trader.run_cycle(BUNDLE, state, tracker) == "hold"
        reads.set(price=101.4, regime=1)
        assert trader.run_cycle(BUNDLE, state, tracker) == "sold"


def _enter(state, tracker, reads, config) -> AppleTrader:
    """Take a long at $100 so the exit rule has something to act on."""
    trader = AppleTrader(config, model_threshold=0.07)
    reads.to_positive(proba=0.9, price=100.0)
    assert trader.run_cycle(BUNDLE, state, tracker) == "bought"
    return trader


class TestTrailingStop:
    def test_sells_once_price_gives_back_the_configured_percent(
        self, state, market_open, monkeypatch
    ):
        broker = FakeBroker(100.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        reads = Reads(monkeypatch, broker)
        trader = _enter(state, tracker, reads, confirm_config(trail_pct=0.5))

        reads.set(price=99.6)  # -0.4% from the peak: still inside the give-back
        assert trader.run_cycle(BUNDLE, state, tracker) == "hold"
        reads.set(price=99.5)  # -0.5%: at the line
        assert trader.run_cycle(BUNDLE, state, tracker) == "sold"
        assert tracker.position_for(TICKER) == 0
        assert "Trailing stop" in tracker.snapshot()["decisions"][-1].reasoning

    def test_the_stop_trails_the_high_since_entry_not_the_entry(
        self, state, market_open, monkeypatch
    ):
        """A drop that would be harmless measured from the entry still sells
        once the position has been further ahead than that."""
        broker = FakeBroker(100.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        reads = Reads(monkeypatch, broker)
        trader = _enter(state, tracker, reads, confirm_config(trail_pct=0.5))

        reads.set(price=102.0)
        assert trader.run_cycle(BUNDLE, state, tracker) == "hold"
        # +1.4% on the trade, but 0.6% off the $102.00 peak.
        reads.set(price=101.4)
        assert trader.run_cycle(BUNDLE, state, tracker) == "sold"
        exit_reason = tracker.snapshot()["decisions"][-1].reasoning
        assert "102.00 high" in exit_reason and "+1.40%" in exit_reason

    def test_the_peak_ratchets_up_and_never_down(self, state, market_open, monkeypatch):
        broker = FakeBroker(100.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        reads = Reads(monkeypatch, broker)
        trader = _enter(state, tracker, reads, confirm_config(trail_pct=1.0))

        for price in (101.0, 100.4, 103.0, 102.5):
            reads.set(price=price)
            assert trader.run_cycle(BUNDLE, state, tracker) == "hold"
        assert trader.entry["peak"] == pytest.approx(103.0)

    def test_the_peak_comes_from_the_bar_high_not_its_close(
        self, state, market_open, monkeypatch
    ):
        """The stop trails the highest price the position actually traded at,
        so a bar that spiked and gave most of it back still raises the peak."""
        broker = FakeBroker(100.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        reads = Reads(monkeypatch, broker)
        trader = _enter(state, tracker, reads, confirm_config(trail_pct=0.5))

        reads.set(price=100.5, high=100.6)
        assert trader.run_cycle(BUNDLE, state, tracker) == "hold"
        assert trader.entry["peak"] == pytest.approx(100.6)
        # Still above the entry and only -0.45% off the last close: nothing but
        # the $100.60 print inside the previous bar explains this exit.
        reads.set(price=100.05, high=100.05)
        assert trader.run_cycle(BUNDLE, state, tracker) == "sold"
        assert "100.60 high" in tracker.snapshot()["decisions"][-1].reasoning

    def test_the_entry_bars_own_high_is_not_part_of_the_peak(
        self, state, market_open, monkeypatch
    ):
        """A spike earlier in the change bar happened before the position
        existed, so the stop does not trail from it."""
        broker = FakeBroker(100.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        reads = Reads(monkeypatch, broker)
        trader = AppleTrader(confirm_config(trail_pct=0.5), model_threshold=0.07)

        reads.to_positive(proba=0.9, price=100.0, high=102.0)
        assert trader.run_cycle(BUNDLE, state, tracker) == "bought"
        assert trader.entry["peak"] == pytest.approx(100.0)
        reads.set(price=99.7, high=99.9)
        assert trader.run_cycle(BUNDLE, state, tracker) == "hold"

    def test_before_any_new_high_the_stop_sits_under_the_entry(
        self, state, market_open, monkeypatch
    ):
        broker = FakeBroker(100.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        reads = Reads(monkeypatch, broker)
        trader = _enter(state, tracker, reads, confirm_config(trail_pct=0.5))

        reads.set(price=99.51)
        assert trader.run_cycle(BUNDLE, state, tracker) == "hold"
        reads.set(price=99.4)
        assert trader.run_cycle(BUNDLE, state, tracker) == "sold"

    def test_momentum_turning_negative_does_not_close_the_position(
        self, state, market_open, monkeypatch
    ):
        """The regime *having* turned is not an exit -- only price and the
        model's forecast are, and this bar carries neither.

        Momentum that has already gone negative is exactly the give-back the
        trailing stop is measuring, so acting on it as well would be the same
        rule twice at a worse level. What the forecast exit acts on is the bar
        *before* this one, which is the whole distinction (see TestReversalExit).
        """
        broker = FakeBroker(100.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        reads = Reads(monkeypatch, broker)
        trader = _enter(state, tracker, reads, confirm_config(trail_pct=2.0))

        for _ in range(5):
            reads.set(price=99.5, mom=-1.5, regime=-1, prev_regime=0, change=True)
            assert trader.run_cycle(BUNDLE, state, tracker) == "hold"
        assert tracker.position_for(TICKER) > 0

    def test_flattens_before_the_close(self, state, market_open, monkeypatch):
        broker = FakeBroker(100.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        reads = Reads(monkeypatch, broker)
        trader = _enter(state, tracker, reads, confirm_config(trail_pct=5.0))

        clock.set_simulated(datetime(2026, 7, 21, 19, 57, tzinfo=timezone.utc))  # 15:57 ET
        reads.set(price=100.2)
        assert trader.run_cycle(BUNDLE, state, tracker) == "sold"
        assert "flattened" in tracker.snapshot()["decisions"][-1].reasoning

    def test_adopts_a_position_it_did_not_open(self, state, market_open, monkeypatch):
        """Restarted onto a ledger that already holds shares: the stop has no
        peak it ever saw, so it starts trailing from the current price."""
        broker = FakeBroker(100.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tracker.record_trade(TICKER, "buy", 10, "seeded", "k", "s")
        reads = Reads(monkeypatch, broker)
        trader = AppleTrader(confirm_config(trail_pct=0.5), model_threshold=0.07)

        reads.set(price=100.0)
        assert trader.run_cycle(BUNDLE, state, tracker) == "hold"
        assert trader.entry["peak"] == pytest.approx(100.0)
        reads.set(price=99.4)
        assert trader.run_cycle(BUNDLE, state, tracker) == "sold"


def reversal_config(**kwargs) -> AppleTraderConfig:
    """A rule set whose forecast exit is armed, on the `confirm` entry so the
    position can be taken with one scripted bar."""
    kwargs.setdefault("reversal_threshold", 0.30)
    return confirm_config(**kwargs)


class TestReversalExit:
    """The second exit: the model calling the end of the regime it bought.

    The stub supplies `reversal_proba` the way `read_latest` does -- filled in
    only on a positive bar reached while holding, and None everywhere else --
    so these pin the rule without depending on a 200 MB forecaster.
    """

    def test_sells_when_the_forecast_clears_the_threshold(
        self, state, market_open, monkeypatch
    ):
        broker = FakeBroker(100.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        reads = Reads(monkeypatch, broker)
        trader = _enter(state, tracker, reads, reversal_config(trail_pct=5.0))

        reads.set(price=101.0, reversal_proba=0.29)
        assert trader.run_cycle(BUNDLE, state, tracker) == "hold"
        reads.set(price=101.0, reversal_proba=0.30)
        assert trader.run_cycle(BUNDLE, state, tracker) == "sold"
        assert tracker.position_for(TICKER) == 0
        assert "Forecast reversal" in tracker.snapshot()["decisions"][-1].reasoning

    def test_sells_at_the_high_before_any_give_back(
        self, state, market_open, monkeypatch
    ):
        """The point of the rule: out while the trade is still at its peak,
        which the trailing stop can never do by construction."""
        broker = FakeBroker(100.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        reads = Reads(monkeypatch, broker)
        trader = _enter(state, tracker, reads, reversal_config(trail_pct=0.5))

        reads.set(price=103.0, reversal_proba=0.9)
        assert trader.run_cycle(BUNDLE, state, tracker) == "sold"
        reasoning = tracker.snapshot()["decisions"][-1].reasoning
        assert "+3.00%" in reasoning and "90%" in reasoning

    def test_off_by_default_for_records_that_never_had_it(
        self, state, market_open, monkeypatch
    ):
        """`reversal_threshold=None` is the pre-existing strategy exactly: the
        forecast is ignored however loud it gets."""
        broker = FakeBroker(100.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        reads = Reads(monkeypatch, broker)
        trader = _enter(
            state, tracker, reads, confirm_config(trail_pct=5.0, reversal_threshold=None)
        )

        reads.set(price=101.0, reversal_proba=0.99)
        assert trader.run_cycle(BUNDLE, state, tracker) == "hold"
        assert tracker.position_for(TICKER) > 0

    def test_an_unasked_model_is_not_a_sell(self, state, market_open, monkeypatch):
        """`read_latest` leaves the number None on any bar that does not pose
        the question. An absent probability is not a quiet zero *or* a quiet
        yes -- it is a bar with nothing to say."""
        broker = FakeBroker(100.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        reads = Reads(monkeypatch, broker)
        trader = _enter(state, tracker, reads, reversal_config(trail_pct=5.0))

        reads.set(price=101.0, reversal_proba=None)
        assert trader.run_cycle(BUNDLE, state, tracker) == "hold"
        assert tracker.position_for(TICKER) > 0

    def test_the_trailing_stop_still_wins_a_bar_they_both_fire_on(
        self, state, market_open, monkeypatch
    ):
        """Both exits close the position, so the only thing at stake is the
        ledger's account of why -- and a give-back that actually happened is a
        better explanation than a forecast that agreed with it."""
        broker = FakeBroker(100.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        reads = Reads(monkeypatch, broker)
        trader = _enter(state, tracker, reads, reversal_config(trail_pct=0.5))

        reads.set(price=99.0, reversal_proba=0.99)
        assert trader.run_cycle(BUNDLE, state, tracker) == "sold"
        assert "Trailing stop" in tracker.snapshot()["decisions"][-1].reasoning

    def test_the_question_is_only_asked_while_holding(
        self, state, market_open, monkeypatch
    ):
        """Every ask costs a full forecast and nothing acts on the answer with
        the book flat, so `run_cycle` must not pay for it when it is not long."""
        broker = FakeBroker(100.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        asked: list[bool] = []
        monkeypatch.setattr(
            at.persistence_model, "minute_frame", lambda *a, **k: pd.DataFrame({"x": [1]})
        )
        reads = Reads(monkeypatch, broker)

        def spy(bundle, frame, holding=False):
            asked.append(holding)
            return reads.next_read

        monkeypatch.setattr(at.persistence_model, "read_latest", spy)
        trader = AppleTrader(reversal_config(trail_pct=5.0), model_threshold=0.07)

        reads.set(price=100.0)  # flat
        trader.run_cycle(BUNDLE, state, tracker)
        reads.to_positive(proba=0.9, price=100.0)
        assert trader.run_cycle(BUNDLE, state, tracker) == "bought"
        reads.set(price=101.0)  # long
        trader.run_cycle(BUNDLE, state, tracker)
        assert asked == [False, False, True]

    def test_a_disarmed_rule_never_pays_for_the_forecast(
        self, state, market_open, monkeypatch
    ):
        """Holding is not enough: with the rule off the question is pointless,
        and asking it would make switching the rule off cost the same as
        leaving it on."""
        broker = FakeBroker(100.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        asked: list[bool] = []
        monkeypatch.setattr(
            at.persistence_model, "minute_frame", lambda *a, **k: pd.DataFrame({"x": [1]})
        )
        reads = Reads(monkeypatch, broker)

        def spy(bundle, frame, holding=False):
            asked.append(holding)
            return reads.next_read

        monkeypatch.setattr(at.persistence_model, "read_latest", spy)
        trader = _enter(
            state, tracker, reads, confirm_config(trail_pct=5.0, reversal_threshold=None)
        )
        reads.set(price=101.0)
        trader.run_cycle(BUNDLE, state, tracker)
        assert asked == [False, False]

    def test_an_out_of_range_threshold_is_refused(self):
        with pytest.raises(ValueError, match="not a probability"):
            AppleTraderConfig(reversal_threshold=1.5)


class TestGuards:
    def test_does_nothing_when_the_market_is_closed(self, state, monkeypatch):
        clock.set_simulated(datetime(2026, 7, 21, 2, 0, tzinfo=timezone.utc))  # 22:00 ET Monday
        try:
            tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(100.0))
            reads = Reads(monkeypatch)
            reads.to_positive(proba=0.99)
            assert AppleTrader().run_cycle(BUNDLE, state, tracker) == "closed"
            assert tracker.position_for(TICKER) == 0
        finally:
            clock.clear()

    def test_reports_when_the_ticker_is_not_streamed(self, market_open, monkeypatch):
        state = AppState()
        state.set_symbols(["MSFT"])
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(100.0))
        Reads(monkeypatch)
        assert AppleTrader().run_cycle(BUNDLE, state, tracker) == "no_data"
        assert any(e["type"] == "error" for e in state.agent_log)

    def test_reports_when_there_is_not_enough_history(self, state, market_open, monkeypatch):
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(100.0))
        monkeypatch.setattr(
            at.persistence_model, "minute_frame", lambda *a, **k: pd.DataFrame({"x": [1]})
        )
        monkeypatch.setattr(at.persistence_model, "read_latest", lambda *a, **k: None)
        assert AppleTrader().run_cycle(BUNDLE, state, tracker) == "no_data"

    def test_missing_model_stops_the_loop_instead_of_trading(self, state, monkeypatch):
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(100.0))
        monkeypatch.setattr(at.persistence_model, "load_bundle", lambda *a, **k: None)
        stop_event = threading.Event()
        at._apple_trader_loop(
            state, tracker, confirm_config(model_key="persistence"), 60, stop_event
        )
        assert state.agent_running is False
        assert any("cannot run without it" in e.get("text", "") for e in state.agent_log)

    def test_anticipating_on_a_model_that_cannot_forecast_stops_the_loop(
        self, state, monkeypatch
    ):
        """The failure this prevents is the quiet one: `read_latest` leaves
        `turn_proba` None on a classifier, every bar reads as "not a buy", and
        the run finishes clean with an empty ledger that looks like a result."""
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(100.0))
        monkeypatch.setattr(at.apple_models, "load", lambda key, ticker=None: BUNDLE)
        config = AppleTraderConfig(
            model_key="persistence", entry_mode=at.ENTRY_ANTICIPATE
        )
        at._apple_trader_loop(state, tracker, config, 60, threading.Event())
        assert state.agent_running is False
        assert any(
            "cannot forecast" in e.get("text", "") and e["type"] == "error"
            for e in state.agent_log
        )

    def test_the_reversal_exit_on_a_model_that_cannot_forecast_stops_the_loop(
        self, state, monkeypatch
    ):
        """Quieter than the entry version and caught for the same reason: the
        run would trade normally and exit everything on the trailing stop,
        which is indistinguishable from a rule that just never triggered."""
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(100.0))
        monkeypatch.setattr(at.apple_models, "load", lambda key, ticker=None: BUNDLE)
        config = confirm_config(model_key="persistence", reversal_threshold=0.3)
        at._apple_trader_loop(state, tracker, config, 60, threading.Event())
        assert state.agent_running is False
        assert any(
            "cannot forecast the breakdown" in e.get("text", "") and e["type"] == "error"
            for e in state.agent_log
        )

    def test_clearing_the_reversal_exit_lets_that_model_run(self, state, monkeypatch):
        """The classifier is not disqualified -- only that one rule is."""
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(100.0))
        monkeypatch.setattr(at.apple_models, "load", lambda key, ticker=None: BUNDLE)
        config = confirm_config(model_key="persistence", reversal_threshold=None)
        assert at.config_error(config, BUNDLE) is None
        stop = threading.Event()
        stop.set()
        at._apple_trader_loop(state, tracker, config, 60, stop)
        assert not any(e["type"] == "error" for e in state.agent_log)

    def test_the_loop_loads_the_model_the_config_names(self, state, monkeypatch):
        """A config naming an unavailable model stops on *that* model rather
        than quietly running the one that happens to be loadable."""
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(100.0))
        monkeypatch.setattr(
            at.apple_models, "load", lambda key, ticker=None: None if key == "nbeats" else BUNDLE
        )
        at._apple_trader_loop(
            state, tracker, AppleTraderConfig(model_key="nbeats"), 60, threading.Event()
        )
        assert state.agent_running is False
        assert any(
            "timetochange2_nbeats_AAPL.pt" in e.get("text", "") for e in state.agent_log
        )


class TestConfigSignature:
    def test_every_rule_that_changes_behaviour_is_in_the_signature(self):
        base = config_signature(AppleTraderConfig(prob_threshold=0.2, trail_pct=0.5))
        assert base != config_signature(AppleTraderConfig(prob_threshold=0.3, trail_pct=0.5))
        assert base != config_signature(AppleTraderConfig(prob_threshold=0.2, trail_pct=0.8))
        assert base != config_signature(
            AppleTraderConfig(prob_threshold=0.2, trail_pct=0.5, model_key="persistence")
        )
        # The same model answering the other question is a different experiment.
        assert base != config_signature(
            confirm_config(prob_threshold=0.2, trail_pct=0.5)
        )
        assert base == config_signature(AppleTraderConfig(prob_threshold=0.2, trail_pct=0.5))

    def test_arming_the_reversal_exit_is_a_new_configuration(self):
        off = config_signature(AppleTraderConfig(prob_threshold=0.2, reversal_threshold=None))
        armed = config_signature(AppleTraderConfig(prob_threshold=0.2, reversal_threshold=0.3))
        assert off != armed
        assert "rev>=0.3" in armed
        # Retuning it is a new configuration too.
        assert armed != config_signature(
            AppleTraderConfig(prob_threshold=0.2, reversal_threshold=0.4)
        )

    def test_a_rule_set_without_it_signs_as_it_always_did(self):
        """Runs recorded before this exit existed and runs configured without
        it now are the same strategy, so Results must go on grouping them."""
        off = config_signature(AppleTraderConfig(prob_threshold=0.2, reversal_threshold=None))
        assert off == "nbeats_AAPL(anticipate,p>=0.2,trail=0.5%,size=95%)"

    def test_an_unset_threshold_names_the_model_that_supplies_it(self):
        config = AppleTraderConfig(prob_threshold=None)
        assert "p>=model" in config_signature(config)
        assert "p>=0.07" in config_signature(config, model_threshold=0.07)


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
# Driven through a stubbed forecast for the same reason the momentum suites
# stub the model read: these pin the RULES -- when a level is a buy, when it is
# a sell, and what the opening window and the closing bell override -- without
# depending on the saved bundle. `tests/test_dayrange_model.py` pins the
# forecast itself.
# --------------------------------------------------------------------------

DAYRANGE_BUNDLE = {"opening_minutes": 5}

# The forecast the stub returns: a $10 average daily range around a predicted
# high of $110, so at the shipped 0.75 / 0.10 the levels land on round numbers.
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
            at.persistence_model, "minute_frame", lambda *a, **k: self.frame()
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


# ------------------------------------------------------- the delta-momentum rules
#
# The third strategy: a signed bps/min forecast read against the regime the tape
# has already printed. The stub below feeds `momentum_change_model.read_latest`
# directly, exactly as `Reads` does for the momentum rules -- these pin the
# RULES, and `tests/test_momentum_change_model.py` pins the model behind them.

MOMENTUM_CHANGE_TICKER = "GOOGL"
MOMENTUM_CHANGE_BUNDLE = {
    "estimator": None,
    "feature_cols": ["mom_15"],
    "pipeline_params": {"persist": 15},
    "model_name": "RandomForest",
}


class MomReads:
    """Feeds `MomentumChangeTrader` a scripted sequence of reads, one per cycle."""

    def __init__(self, monkeypatch, broker: "FakeBroker | None" = None):
        self.broker = broker
        self.minute = 0
        self.next_read: "dict | None" = {}
        self.history_problem: "str | None" = None
        self.history_calls = 0
        momentum_change = at._momentum_change()

        def require_history(frame, session_date):
            self.history_calls += 1
            return self.history_problem

        monkeypatch.setattr(
            momentum_change, "session_frame", lambda *a, **k: pd.DataFrame({"x": [1]})
        )
        monkeypatch.setattr(momentum_change, "require_history", require_history)
        monkeypatch.setattr(momentum_change, "read_latest", lambda *a, **k: self.next_read)

    def set(
        self,
        *,
        price: float = 100.0,
        pred: "float | None" = 0.0,
        mom: "float | None" = -1.0,
        theta: "float | None" = 0.5,
        regime: int = -1,
        regime_before: "int | None" = -1,
        bars_today: int = 200,
        warming_up: bool = False,
        advance: bool = True,
    ) -> dict:
        """Stage the next bar. `advance=False` replays the SAME timestamp, the
        way a cycle running before a new bar has closed would see it."""
        if advance:
            self.minute += 1
        if self.broker is not None:
            self.broker.price = price
        self.next_read = {
            "ts": pd.Timestamp("2026-07-21 10:30", tz="America/New_York")
            + pd.Timedelta(minutes=self.minute),
            "price": price,
            "pred": pred,
            "mom": mom,
            "theta": theta,
            "regime": regime,
            "regime_before": regime_before,
            "bars_today": bars_today,
            "warming_up": warming_up,
        }
        return self.next_read

    def turn_up(self, *, pred: float = 0.9, **kwargs) -> dict:
        """The bar the entry acts on: the previous minute was still negative
        and the model calls the move upwards."""
        return self.set(regime_before=-1, regime=-1, pred=pred, **kwargs)


def momentum_change_config(**kwargs) -> AppleTraderConfig:
    kwargs.setdefault("ticker", MOMENTUM_CHANGE_TICKER)
    return AppleTraderConfig(model_key="momentum_change", **kwargs)


@pytest.fixture
def momentum_change_state() -> AppState:
    state = AppState()
    state.set_symbols([MOMENTUM_CHANGE_TICKER])
    state.api_key = "k"
    state.api_secret = "s"
    state.feed = "yfinance"
    return state


class TestMomentumChangeEntry:
    def _trader(self, **kwargs):
        return at.MomentumChangeTrader(momentum_change_config(**kwargs))

    def test_a_negative_minute_the_model_calls_up_is_bought(
        self, momentum_change_state, market_open, monkeypatch
    ):
        broker = FakeBroker(100.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        reads = MomReads(monkeypatch, broker)
        trader = self._trader()

        reads.turn_up(pred=0.9)
        assert trader.run_cycle(MOMENTUM_CHANGE_BUNDLE, momentum_change_state, tracker) == "bought"
        assert tracker.position_for(MOMENTUM_CHANGE_TICKER) > 0
        reasoning = tracker.snapshot()["decisions"][-1].reasoning
        assert "+0.90 bps/min" in reasoning and "negative momentum regime" in reasoning

    def test_a_prediction_below_the_threshold_is_not_a_buy(
        self, momentum_change_state, market_open, monkeypatch
    ):
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(100.0))
        reads = MomReads(monkeypatch)
        trader = self._trader(buy_thr=0.5)

        reads.turn_up(pred=0.49)
        assert trader.run_cycle(MOMENTUM_CHANGE_BUNDLE, momentum_change_state, tracker) == "hold"
        assert tracker.position_for(MOMENTUM_CHANGE_TICKER) == 0

    @pytest.mark.parametrize("regime_before", [0, 1])
    def test_only_a_negative_regime_is_bought(
        self, momentum_change_state, market_open, monkeypatch, regime_before
    ):
        """The model's timing is the weak half; the tape picks the situation.
        A large prediction on a balanced or positive minute is not a trade."""
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(100.0))
        reads = MomReads(monkeypatch)
        trader = self._trader()

        reads.set(regime_before=regime_before, regime=regime_before, pred=5.0)
        assert trader.run_cycle(MOMENTUM_CHANGE_BUNDLE, momentum_change_state, tracker) == "hold"
        assert tracker.position_for(MOMENTUM_CHANGE_TICKER) == 0

    def test_a_bar_with_no_prediction_yet_is_not_a_buy(
        self, momentum_change_state, market_open, monkeypatch
    ):
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(100.0))
        reads = MomReads(monkeypatch)
        trader = self._trader()

        reads.set(pred=None, warming_up=True)
        assert trader.run_cycle(MOMENTUM_CHANGE_BUNDLE, momentum_change_state, tracker) == "warming_up"
        assert tracker.position_for(MOMENTUM_CHANGE_TICKER) == 0

    def test_a_replayed_bar_does_not_buy_twice(
        self, momentum_change_state, market_open, monkeypatch
    ):
        broker = FakeBroker(100.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        reads = MomReads(monkeypatch, broker)
        trader = self._trader()

        reads.turn_up(pred=0.9)
        assert trader.run_cycle(MOMENTUM_CHANGE_BUNDLE, momentum_change_state, tracker) == "bought"
        held = tracker.position_for(MOMENTUM_CHANGE_TICKER)
        reads.turn_up(pred=0.9, advance=False)
        assert trader.run_cycle(MOMENTUM_CHANGE_BUNDLE, momentum_change_state, tracker) == "hold"
        assert tracker.position_for(MOMENTUM_CHANGE_TICKER) == held


def _mom_entered(state, tracker, reads, **kwargs):
    trader = at.MomentumChangeTrader(momentum_change_config(**kwargs))
    reads.turn_up(pred=0.9)
    assert trader.run_cycle(MOMENTUM_CHANGE_BUNDLE, state, tracker) == "bought"
    return trader


class TestMomentumChangeExit:
    def test_the_stop_is_measured_from_the_entry_and_does_not_trail(
        self, momentum_change_state, market_open, monkeypatch
    ):
        """Unlike the momentum rules' trailing stop, a run-up does not move
        this floor: it stays `stop_pct` under the price that was paid."""
        broker = FakeBroker(100.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        reads = MomReads(monkeypatch, broker)
        trader = _mom_entered(momentum_change_state, tracker, reads, stop_pct=0.5)

        reads.set(price=105.0)   # a run-up the stop must ignore
        assert trader.run_cycle(MOMENTUM_CHANGE_BUNDLE, momentum_change_state, tracker) == "hold"
        reads.set(price=99.6)    # 0.4% down: still above the floor
        assert trader.run_cycle(MOMENTUM_CHANGE_BUNDLE, momentum_change_state, tracker) == "hold"
        reads.set(price=99.4)    # 0.6% down: through it
        assert trader.run_cycle(MOMENTUM_CHANGE_BUNDLE, momentum_change_state, tracker) == "sold"
        assert "Stop" in tracker.snapshot()["decisions"][-1].reasoning

    def test_the_momentum_floor_scales_with_the_days_threshold(
        self, momentum_change_state, market_open, monkeypatch
    ):
        """`m1_mult` is a multiple of theta rather than a number of bps,
        because theta is set from yesterday's volatility."""
        broker = FakeBroker(100.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        reads = MomReads(monkeypatch, broker)
        trader = _mom_entered(momentum_change_state, tracker, reads, m1_mult=-2.0)

        reads.set(price=100.0, mom=-0.9, theta=0.5)   # floor -1.0
        assert trader.run_cycle(MOMENTUM_CHANGE_BUNDLE, momentum_change_state, tracker) == "hold"
        reads.set(price=100.0, mom=-1.1, theta=0.5)
        assert trader.run_cycle(MOMENTUM_CHANGE_BUNDLE, momentum_change_state, tracker) == "sold"
        assert "Momentum floor" in tracker.snapshot()["decisions"][-1].reasoning

    def test_the_model_closes_a_positive_regime_it_calls_over(
        self, momentum_change_state, market_open, monkeypatch
    ):
        broker = FakeBroker(100.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        reads = MomReads(monkeypatch, broker)
        trader = _mom_entered(momentum_change_state, tracker, reads, sell_thr=0.3)

        reads.set(price=100.5, regime_before=1, regime=1, mom=1.2, pred=-0.2)
        assert trader.run_cycle(MOMENTUM_CHANGE_BUNDLE, momentum_change_state, tracker) == "hold"
        reads.set(price=100.5, regime_before=1, regime=1, mom=1.2, pred=-0.4)
        assert trader.run_cycle(MOMENTUM_CHANGE_BUNDLE, momentum_change_state, tracker) == "sold"
        assert "Model exit" in tracker.snapshot()["decisions"][-1].reasoning

    def test_the_model_exit_needs_a_positive_regime(
        self, momentum_change_state, market_open, monkeypatch
    ):
        """A large negative prediction on a still-negative minute is the entry
        question read backwards, not an exit."""
        broker = FakeBroker(100.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        reads = MomReads(monkeypatch, broker)
        trader = _mom_entered(momentum_change_state, tracker, reads)

        reads.set(price=100.5, regime_before=-1, regime=-1, mom=-0.4, pred=-2.0)
        assert trader.run_cycle(MOMENTUM_CHANGE_BUNDLE, momentum_change_state, tracker) == "hold"

    def test_a_bar_where_two_exits_fire_is_reported_as_the_stop(
        self, momentum_change_state, market_open, monkeypatch
    ):
        """The notebook tests all three independently and exits on any, so the
        order only decides what the ledger is told -- and a fact about price
        beats a prediction."""
        broker = FakeBroker(100.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        reads = MomReads(monkeypatch, broker)
        trader = _mom_entered(momentum_change_state, tracker, reads, stop_pct=0.5)

        reads.set(price=99.0, regime_before=1, regime=1, mom=1.2, pred=-2.0)
        assert trader.run_cycle(MOMENTUM_CHANGE_BUNDLE, momentum_change_state, tracker) == "sold"
        assert "Stop" in tracker.snapshot()["decisions"][-1].reasoning

    def test_the_position_is_flattened_before_the_close(
        self, momentum_change_state, market_open, monkeypatch
    ):
        broker = FakeBroker(100.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        reads = MomReads(monkeypatch, broker)
        trader = _mom_entered(momentum_change_state, tracker, reads)

        clock.set_simulated(datetime(2026, 7, 21, 19, 57, tzinfo=timezone.utc))
        reads.set(price=100.2)
        assert trader.run_cycle(MOMENTUM_CHANGE_BUNDLE, momentum_change_state, tracker) == "sold"
        assert "Session ends" in tracker.snapshot()["decisions"][-1].reasoning


class TestMomentumChangeGuards:
    def test_no_entry_inside_the_closing_flatten_window(
        self, momentum_change_state, market_open, monkeypatch
    ):
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(100.0))
        reads = MomReads(monkeypatch)
        trader = at.MomentumChangeTrader(momentum_change_config())

        clock.set_simulated(datetime(2026, 7, 21, 19, 57, tzinfo=timezone.utc))
        reads.turn_up(pred=0.9)
        assert trader.run_cycle(MOMENTUM_CHANGE_BUNDLE, momentum_change_state, tracker) == "hold"
        assert tracker.position_for(MOMENTUM_CHANGE_TICKER) == 0

    def test_a_history_it_cannot_get_stops_the_day_rather_than_the_bar(
        self, momentum_change_state, market_open, monkeypatch
    ):
        """Six sessions missing at 09:31 are still missing at 14:00, so the
        refusal is logged once and the session is skipped -- not retried every
        minute for six and a half hours."""
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(100.0))
        reads = MomReads(monkeypatch)
        reads.history_problem = "only 2 of the 6 previous sessions"
        trader = at.MomentumChangeTrader(momentum_change_config())

        for _ in range(4):
            reads.turn_up(pred=0.9)
            assert trader.run_cycle(MOMENTUM_CHANGE_BUNDLE, momentum_change_state, tracker) == "no_data"
        assert reads.history_calls == 1
        assert tracker.position_for(MOMENTUM_CHANGE_TICKER) == 0
        errors = [e for e in momentum_change_state.agent_log if e.get("type") == "error"]
        assert len(errors) == 1 and "2 of the 6" in errors[0]["text"]

    def test_a_new_session_tries_the_history_again(
        self, momentum_change_state, market_open, monkeypatch
    ):
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(100.0))
        reads = MomReads(monkeypatch)
        reads.history_problem = "only 2 of the 6 previous sessions"
        trader = at.MomentumChangeTrader(momentum_change_config())
        reads.turn_up(pred=0.9)
        assert trader.run_cycle(MOMENTUM_CHANGE_BUNDLE, momentum_change_state, tracker) == "no_data"

        clock.set_simulated(datetime(2026, 7, 22, 14, 30, tzinfo=timezone.utc))
        reads.history_problem = None
        reads.turn_up(pred=0.9)
        assert trader.run_cycle(MOMENTUM_CHANGE_BUNDLE, momentum_change_state, tracker) == "bought"
        assert reads.history_calls == 2

    def test_does_nothing_when_the_market_is_closed(self, momentum_change_state, monkeypatch):
        clock.set_simulated(datetime(2026, 7, 21, 2, 0, tzinfo=timezone.utc))
        try:
            tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(100.0))
            reads = MomReads(monkeypatch)
            reads.turn_up(pred=0.9)
            trader = at.MomentumChangeTrader(momentum_change_config())
            assert trader.run_cycle(MOMENTUM_CHANGE_BUNDLE, momentum_change_state, tracker) == "closed"
        finally:
            clock.clear()

    def test_a_stop_at_zero_is_refused_by_the_config(self):
        """It would be breached by the bar that opened the trade, on every
        bar -- a rule that sells everything it buys."""
        with pytest.raises(ValueError, match="stop_pct"):
            AppleTraderConfig(stop_pct=0.0)


class TestMomentumChangeSignature:
    def test_every_knob_that_changes_behaviour_is_in_the_signature(self):
        base = config_signature(momentum_change_config())
        for changed in (
            momentum_change_config(buy_thr=0.4),
            momentum_change_config(sell_thr=0.4),
            momentum_change_config(m1_mult=-1.5),
            momentum_change_config(stop_pct=0.8),
            momentum_change_config(position_pct=50.0),
            momentum_change_config(ticker="INTC"),
        ):
            assert config_signature(changed) != base

    def test_the_momentum_knobs_are_left_out(self):
        """They are inert here, and a signature carrying them would split one
        strategy's runs into two configurations the first time somebody moved
        a knob that changes nothing."""
        base = config_signature(momentum_change_config())
        assert base == config_signature(momentum_change_config(trail_pct=9.0))
        assert base == config_signature(momentum_change_config(prob_threshold=0.9))
        assert base == config_signature(momentum_change_config(buy_k=1.5, sell_k=0.9))
        assert "trail" not in base and "buy=H-" not in base

    def test_it_is_not_confusable_with_the_other_strategies(self):
        assert config_signature(momentum_change_config()).startswith("momentum_change_GOOGL(")
        assert config_signature(dayrange_config()).startswith("dayrange_AAPL(")


# --------------------------------------------------------------------------
# The price-range rules (PriceRange2): one forecast at 09:35, quantile levels
# and a stop.
#
# Stubbed the same way the day-range suite is, and for the same reason: these
# pin the RULES, and `tests/test_pricerange_model.py` pins the forecast against
# the notebook.
# --------------------------------------------------------------------------

PRICERANGE_BUNDLE = {"opening_minutes": 5}

# A forecast whose quantile edges land on round numbers at the shipped buffers:
# buy 100 x 1.0015 = 100.15, sell 108 x 1.0 = 108.
PRICE_FORECAST = {
    "pred_high": 110.0,
    "pred_low": 98.0,
    "pred_range_pct": 0.12,
    "ref": 104.0,
    "open_high": 105.0,
    "open_low": 103.0,
    "buy_edge": 100.0,
    "sell_edge": 108.0,
}
PR_BUY_LEVEL = 100.15
PR_SELL_LEVEL = 108.0


class PriceTape(Tape):
    """`Tape`, with the price-range model stubbed instead of the day-range one."""

    def __init__(self, monkeypatch, broker=None, minutes: int = 5, forecast=None):
        self.forecast = dict(forecast or PRICE_FORECAST)
        super().__init__(monkeypatch, broker, minutes)
        pricerange = at._pricerange()
        monkeypatch.setattr(pricerange, "forecast_session", self._forecast)
        monkeypatch.setattr(pricerange, "fetch_cross_frame", lambda *a, **k: None)
        monkeypatch.setattr(
            pricerange, "fetch_opening_volume_history", lambda *a, **k: None
        )
        monkeypatch.setattr(pricerange, "daily_frame_from_bars", lambda bars: None)

    def _forecast(self, *args, **kwargs):
        self.forecast_calls += 1
        return dict(self.forecast)


def pricerange_config(**kwargs) -> AppleTraderConfig:
    return AppleTraderConfig(model_key="pricerange", **kwargs)


class TestPriceRangeEntry:
    def _trader(self, **kwargs):
        return at.PriceRangeTrader(pricerange_config(**kwargs))

    def test_it_rests_on_the_quantile_edges_not_the_point_forecast(
        self, state, market_open, monkeypatch
    ):
        """The whole point of this strategy: the median edges lose money in
        every one of PriceRange2's 112 sweep cells, so the levels come off the
        75th/25th percentiles instead."""
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(104.0))
        tape = PriceTape(monkeypatch)
        trader = self._trader()

        tape.append(104.0)
        trader.run_cycle(PRICERANGE_BUNDLE, state, tracker)
        assert trader.plan["buy_level"] == pytest.approx(PR_BUY_LEVEL)
        assert trader.plan["sell_level"] == pytest.approx(PR_SELL_LEVEL)
        # ...and not the point forecast, which is far wider.
        assert trader.plan["buy_level"] > PRICE_FORECAST["pred_low"]
        assert trader.plan["sell_level"] < PRICE_FORECAST["pred_high"]

    def test_a_bar_that_trades_down_to_the_buy_level_is_bought(
        self, state, market_open, monkeypatch
    ):
        broker = FakeBroker(101.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tape = PriceTape(monkeypatch, broker)
        trader = self._trader()

        tape.append(101.0, low=PR_BUY_LEVEL - 0.01)
        assert trader.run_cycle(PRICERANGE_BUNDLE, state, tracker) == "bought"
        reasoning = tracker.snapshot()["decisions"][-1].reasoning
        assert "100.15" in reasoning and "75th-percentile" in reasoning

    def test_a_bar_that_stays_above_the_buy_level_is_not(
        self, state, market_open, monkeypatch
    ):
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(104.0))
        tape = PriceTape(monkeypatch)
        trader = self._trader()

        tape.append(104.0, low=PR_BUY_LEVEL + 0.01)
        assert trader.run_cycle(PRICERANGE_BUNDLE, state, tracker) == "hold"
        assert tracker.position_for(TICKER) == 0

    def test_the_buffers_move_the_levels_inward(self, state, market_open, monkeypatch):
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(104.0))
        tape = PriceTape(monkeypatch)
        trader = self._trader(entry_buffer=1.0, exit_buffer=2.0)

        tape.append(104.0)
        trader.run_cycle(PRICERANGE_BUNDLE, state, tracker)
        assert trader.plan["buy_level"] == pytest.approx(101.0)   # 100 x 1.01
        assert trader.plan["sell_level"] == pytest.approx(105.84)  # 108 x 0.98

    def test_nothing_trades_before_the_opening_window_closes(
        self, state, market_open, monkeypatch
    ):
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(100.0))
        tape = PriceTape(monkeypatch, minutes=3)
        trader = self._trader()

        assert trader.run_cycle(PRICERANGE_BUNDLE, state, tracker) == "warming_up"
        assert trader.plan is None and tape.forecast_calls == 0

    def test_the_forecast_is_made_once_and_reused_all_day(
        self, state, market_open, monkeypatch
    ):
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(104.0))
        tape = PriceTape(monkeypatch)
        trader = self._trader()
        for _ in range(6):
            tape.append(104.0)
            trader.run_cycle(PRICERANGE_BUNDLE, state, tracker)
        assert tape.forecast_calls == 1

    def test_a_bundle_without_quantiles_stands_down_for_the_session(
        self, state, market_open, monkeypatch
    ):
        """The levels *are* the quantiles, so a point-only bundle has no rule.

        Falling back to the median edges would be running the one configuration
        that project measured as never profitable.
        """
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(100.0))
        tape = PriceTape(
            monkeypatch,
            forecast={**PRICE_FORECAST, "buy_edge": None, "sell_edge": None},
        )
        trader = self._trader()

        tape.append(100.0, low=90.0)  # would fill on any level
        assert trader.run_cycle(PRICERANGE_BUNDLE, state, tracker) == "no_data"
        assert trader.plan is None and trader.blocked is not None
        assert tracker.position_for(TICKER) == 0

    def test_crossed_levels_stand_the_session_down(
        self, state, market_open, monkeypatch
    ):
        """A forecast band narrower than the buffers is a real forecast about a
        very quiet day -- and a buy above the sell would round-trip forever."""
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(104.0))
        tape = PriceTape(
            monkeypatch,
            forecast={**PRICE_FORECAST, "buy_edge": 104.0, "sell_edge": 104.0},
        )
        trader = self._trader(entry_buffer=1.0)

        tape.append(104.0, low=100.0)
        assert trader.run_cycle(PRICERANGE_BUNDLE, state, tracker) == "no_data"
        assert trader.plan is None and trader.blocked is not None


class TestPriceRangeExit:
    def _entered(self, state, tracker, tape, **kwargs):
        trader = at.PriceRangeTrader(pricerange_config(**kwargs))
        tape.append(101.0, low=PR_BUY_LEVEL - 0.01)
        assert trader.run_cycle(PRICERANGE_BUNDLE, state, tracker) == "bought"
        return trader

    def test_the_sell_level_closes_the_position(self, state, market_open, monkeypatch):
        broker = FakeBroker(101.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tape = PriceTape(monkeypatch, broker)
        trader = self._entered(state, tracker, tape)

        tape.append(108.5, high=PR_SELL_LEVEL + 0.1)
        assert trader.run_cycle(PRICERANGE_BUNDLE, state, tracker) == "sold"
        assert tracker.position_for(TICKER) == 0
        assert "Target" in tracker.snapshot()["decisions"][-1].reasoning

    def test_the_stop_closes_it_too(self, state, market_open, monkeypatch):
        broker = FakeBroker(101.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tape = PriceTape(monkeypatch, broker)
        trader = self._entered(state, tracker, tape)
        entry = trader.entry["price"]

        tape.append(entry * 0.985, low=entry * 0.98)
        assert trader.run_cycle(PRICERANGE_BUNDLE, state, tracker) == "sold"
        assert "Stop" in tracker.snapshot()["decisions"][-1].reasoning

    def test_a_bar_holding_both_levels_takes_the_stop(
        self, state, market_open, monkeypatch
    ):
        """The rule, not an implementation detail: a minute bar does not record
        which came first, and assuming the good one is how a backtest invents
        money. PriceRange2's simulator takes the stop; so does this."""
        broker = FakeBroker(101.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tape = PriceTape(monkeypatch, broker)
        trader = self._entered(state, tracker, tape)
        entry = trader.entry["price"]

        tape.append(104.0, low=entry * 0.98, high=PR_SELL_LEVEL + 1.0)
        assert trader.run_cycle(PRICERANGE_BUNDLE, state, tracker) == "sold"
        assert "Stop" in tracker.snapshot()["decisions"][-1].reasoning

    def test_the_buy_is_spent_on_the_entry_not_the_exit(self, state, market_open, monkeypatch):
        """Where `simulate.simulate_session` disarms, and it matters.

        Disarming on the exit would spend the session's one trade on a sell the
        ledger refused; disarming on a buy that filled nothing would spend it on
        no trade at all.
        """
        broker = FakeBroker(101.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tape = PriceTape(monkeypatch, broker)
        trader = self._entered(state, tracker, tape)
        assert trader.armed is False  # already spent, while still holding

    def test_a_buy_that_fills_nothing_does_not_spend_the_session(
        self, state, market_open, monkeypatch
    ):
        broker = FakeBroker(101.0)
        tracker = DecisionTracker(starting_cash=0.0, broker=broker)
        tape = PriceTape(monkeypatch, broker)
        trader = at.PriceRangeTrader(pricerange_config())

        tape.append(101.0, low=PR_BUY_LEVEL - 0.01)
        assert trader.run_cycle(PRICERANGE_BUNDLE, state, tracker) == "hold"
        assert trader.armed is True

    def test_the_buy_does_not_re_arm_by_default(self, state, market_open, monkeypatch):
        broker = FakeBroker(101.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tape = PriceTape(monkeypatch, broker)
        trader = self._entered(state, tracker, tape)

        tape.append(108.5, high=PR_SELL_LEVEL + 0.1)
        assert trader.run_cycle(PRICERANGE_BUNDLE, state, tracker) == "sold"
        assert trader.armed is False
        tape.append(101.0, low=PR_BUY_LEVEL - 0.5)
        assert trader.run_cycle(PRICERANGE_BUNDLE, state, tracker) == "hold"
        assert tracker.position_for(TICKER) == 0

    def test_re_entry_can_be_switched_on(self, state, market_open, monkeypatch):
        broker = FakeBroker(101.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tape = PriceTape(monkeypatch, broker)
        trader = self._entered(state, tracker, tape, allow_reentry=True)

        tape.append(108.5, high=PR_SELL_LEVEL + 0.1)
        assert trader.run_cycle(PRICERANGE_BUNDLE, state, tracker) == "sold"
        assert trader.armed is True
        tape.append(101.0, low=PR_BUY_LEVEL - 0.5)
        assert trader.run_cycle(PRICERANGE_BUNDLE, state, tracker) == "bought"


class TestPriceRangeSignature:
    def test_the_levels_and_the_stop_are_the_signature(self):
        base = config_signature(pricerange_config())
        assert base == (
            "pricerange_AAPL(buy=L75+0.15%,sell=H25-0%,stop=1%,size=95%)"
        )
        assert base != config_signature(pricerange_config(entry_buffer=0.3))
        assert base != config_signature(pricerange_config(exit_buffer=0.1))
        assert base != config_signature(pricerange_config(range_stop_pct=1.5))
        assert base != config_signature(pricerange_config(allow_reentry=True))
        assert base != config_signature(pricerange_config(ticker="INTC"))

    def test_the_other_strategies_knobs_are_inert(self):
        """Including `stop_pct`, which belongs to the delta-momentum rules.

        The two stops were fitted separately and mean different things, which
        is why they are separate fields -- and why moving one must not split
        the other strategy's runs into two configurations in Results.
        """
        base = config_signature(pricerange_config())
        assert base == config_signature(
            pricerange_config(trail_pct=2.0, prob_threshold=0.9, buy_k=1.5,
                              stop_pct=2.0, buy_thr=0.9)
        )

    def test_the_two_stops_do_not_share_a_field(self):
        """Retuning the price-range stop must leave a momentum_change run's
        signature untouched, and vice versa."""
        momentum = config_signature(momentum_change_config())
        assert momentum == config_signature(momentum_change_config(range_stop_pct=3.0))
        price = config_signature(pricerange_config())
        assert price == config_signature(pricerange_config(stop_pct=3.0))


class TestStrategyCopy:
    """Both apps must describe every strategy they can run.

    There is no exception raised when they do not: `copy.intro.get(...)` and
    `copy.help.get(...)` return None, `_caption(None)` renders nothing, and the
    form comes up with an unexplained set of number inputs. So a strategy added
    to the registry without its copy fails silently in exactly the place a user
    is deciding what a knob does -- which is why this is a test rather than a
    convention.
    """

    def _copies(self):
        pytest.importorskip("streamlit")
        from agent_stonks.ui import _APPLE_TRADER_COPY
        from simlab.app import _APPLE_TRADER_COPY_FIELDS

        return {
            "live": (_APPLE_TRADER_COPY.intro, _APPLE_TRADER_COPY.outro,
                     _APPLE_TRADER_COPY.help),
            "simlab": (_APPLE_TRADER_COPY_FIELDS["intro"],
                       _APPLE_TRADER_COPY_FIELDS["outro"],
                       _APPLE_TRADER_COPY_FIELDS["help"]),
        }

    def test_every_level_based_strategy_is_introduced_in_both_apps(self):
        """The three whose form is a set of price levels with no other context.

        Not every strategy in the registry: the live app has never carried a
        `momentum` intro, because that form's four knobs are each explained by
        their own `help` and the model summary sits directly above them. These
        three are the ones where the numbers mean nothing without a sentence
        saying what they are distances *from*.
        """
        level_based = {"dayrange", "momentum_change", "pricerange"}
        for app, (intro, outro, _) in self._copies().items():
            assert not level_based - set(intro), f"{app} is missing an intro"
            assert not level_based - set(outro), f"{app} is missing an outro"

    def test_every_price_range_knob_has_help_in_both_apps(self):
        """The four fields `pricerange_params` renders with a `help=` argument."""
        knobs = {"entry_buffer", "exit_buffer", "range_stop_pct", "allow_reentry"}
        for app, (_, _, help_) in self._copies().items():
            missing = knobs - set(help_)
            assert not missing, f"{app} has no help for {sorted(missing)}"

    def test_the_rules_own_verdict_on_itself_reaches_the_user(self):
        """PriceRange2 measured this rule as having no edge over holding, and
        both apps say so where the rule is configured. That finding is the most
        important thing about the strategy and the easiest to leave out."""
        for app, (_, outro, _) in self._copies().items():
            text = outro["pricerange"].lower()
            assert "no edge" in text or "reduced exposure" in text, app


class TestStrategySelection:
    def test_the_model_chooses_the_state_machine(self):
        assert isinstance(
            at.build_trader(dayrange_config(), DAYRANGE_BUNDLE), at.DayRangeTrader
        )
        assert isinstance(
            at.build_trader(momentum_change_config(), MOMENTUM_CHANGE_BUNDLE), at.MomentumChangeTrader
        )
        assert isinstance(
            at.build_trader(pricerange_config(), PRICERANGE_BUNDLE), at.PriceRangeTrader
        )
        assert isinstance(at.build_trader(AppleTraderConfig(), BUNDLE), AppleTrader)

    def test_the_momentum_pairing_checks_do_not_fire_on_the_other_strategy(self):
        """`anticipate` and the reversal exit are momentum concepts. A
        day-range config carries their defaults and must not be rejected for
        them -- the bundle it runs on cannot answer either question and is
        never asked."""
        assert at.config_error(dayrange_config(), DAYRANGE_BUNDLE) is None

    def test_the_levels_are_the_signature_and_the_momentum_knobs_are_not(self):
        base = config_signature(dayrange_config())
        assert base == "dayrange_AAPL(buy=H-0.75A,sell=H-0.1A,size=95%)"
        assert base != config_signature(dayrange_config(buy_k=0.8))
        assert base != config_signature(dayrange_config(sell_k=0.2))
        # Inert knobs must not split one strategy's runs into two
        # configurations in Results.
        assert base == config_signature(dayrange_config(trail_pct=2.0, prob_threshold=0.9))

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
# fitted on a symbol or it was not. Between 2026-09-07 and the PriceRange2
# integration every model covered every shipped symbol, which made `UNMODELLED`
# the only pairing the shipped registry refused. That is no longer true, and
# the reason is worth keeping: PriceRange2 was run on **GOOG** where the three
# TimeToChange projects were run on **GOOGL**, and the two are different share
# classes rather than spellings of one symbol. So the registry now narrows in
# both directions on its own, and these tests say so.

NON_AAPL = "GOOGL"
DAYRANGE_ONLY = NON_AAPL  # kept for the tests written before there were two
PRICERANGE_ONLY = "GOOG"
UNMODELLED = "MSFT"

# The four fitted by the TimeToChange projects, which share a symbol set.
TIMETOCHANGE_MODELS = ["persistence", "nbeats", "dayrange", "momentum_change"]
ALL_MODELS = TIMETOCHANGE_MODELS + ["pricerange"]


class TestInstrument:
    def test_the_symbols_on_offer_are_the_ones_a_model_covers(self):
        # AAPL and INTC are the two symbols every project was run on.
        for symbol in (TICKER, "INTC"):
            assert apple_models.keys_for(symbol) == ALL_MODELS
        assert apple_models.keys_for(UNMODELLED) == []

    def test_the_two_google_share_classes_are_not_interchangeable(self):
        """GOOGL runs the TimeToChange models; GOOG runs PriceRange2's.

        The pairing that would be easiest to get wrong and hardest to notice:
        the two tickers track nearly the same price, so a model quietly
        answering for the wrong one would look entirely plausible in the log.
        """
        assert apple_models.keys_for(NON_AAPL) == TIMETOCHANGE_MODELS
        assert apple_models.keys_for(PRICERANGE_ONLY) == ["pricerange"]
        assert "pricerange" not in apple_models.keys_for(NON_AAPL)
        # And the refusal names the symbol it *was* fitted on, so the reader is
        # not left guessing which of the two the file belongs to.
        reason = apple_models.unavailable_reason("pricerange", NON_AAPL)
        assert PRICERANGE_ONLY in reason and NON_AAPL in reason

    def test_a_model_cannot_be_pointed_at_a_symbol_it_was_not_fitted_on(
        self, monkeypatch
    ):
        """The check that keeps 'Apple Trader on GOOGL' from meaning a model
        fitted on a different stock's tape.

        Stubbed back to AAPL-only, because every shipped model now covers every
        shipped symbol and the machinery would otherwise go untested until the
        next model arrives for one ticker ahead of the others."""
        monkeypatch.setitem(
            apple_models.MODELS, "nbeats",
            replace(apple_models.MODELS["nbeats"], tickers=(TICKER,)),
        )
        error = at.model_ticker_error(
            AppleTraderConfig(model_key="nbeats", ticker=DAYRANGE_ONLY)
        )
        assert error is not None
        assert "AAPL only" in error and DAYRANGE_ONLY in error
        # ...and it names what that symbol *can* run.
        assert "Day-range forecast" in error

    def test_every_model_covers_the_retrained_symbols(self):
        """Every project has been re-run per ticker, on the symbols it names.

        Two symbol sets rather than one, because PriceRange2 was run on GOOG
        and the TimeToChange projects on GOOGL.
        """
        for key in TIMETOCHANGE_MODELS:
            for symbol in (TICKER, DAYRANGE_ONLY, "INTC"):
                config = AppleTraderConfig(model_key=key, ticker=symbol)
                assert at.model_ticker_error(config) is None
        for symbol in (TICKER, PRICERANGE_ONLY, "INTC"):
            config = AppleTraderConfig(model_key="pricerange", ticker=symbol)
            assert at.model_ticker_error(config) is None

    def test_an_unmodelled_symbol_is_refused_with_no_alternative_offered(self):
        error = at.model_ticker_error(
            AppleTraderConfig(model_key="dayrange", ticker=UNMODELLED)
        )
        assert error is not None and "pick another instrument" in error

    def test_the_pairing_is_part_of_config_error(self):
        """One call is what every launch path checks, so the pairing cannot be
        enforced in the live loop and forgotten in SimLab."""
        config = AppleTraderConfig(model_key="nbeats", ticker=UNMODELLED)
        assert "cannot trade" in (at.config_error(config, BUNDLE) or "")

    def test_a_ticker_is_normalised(self):
        assert AppleTraderConfig(ticker=" googl ").ticker == "GOOGL"

    def test_the_signature_carries_the_symbol(self):
        """The same levels over two tapes are two experiments; filing them
        together would average them into one row in Results."""
        aapl = config_signature(AppleTraderConfig(model_key="dayrange"))
        googl = config_signature(
            AppleTraderConfig(model_key="dayrange", ticker=DAYRANGE_ONLY)
        )
        assert aapl.startswith("dayrange_AAPL(") and googl.startswith("dayrange_GOOGL(")
        assert aapl != googl

    def test_a_record_written_before_the_instrument_existed_is_an_aapl_run(self):
        assert AppleTraderConfig(model_key="persistence").ticker == TICKER

    def test_the_configured_symbol_is_the_one_traded(self, market_open, monkeypatch):
        """Every read, order and log line follows the config, not the module."""
        state = AppState()
        state.set_symbols([DAYRANGE_ONLY])
        state.api_key, state.api_secret, state.feed = "k", "s", "iex"
        broker = FakeBroker(100.0)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        reads = Reads(monkeypatch)
        reads.to_positive(proba=0.99)

        trader = AppleTrader(confirm_config(ticker=DAYRANGE_ONLY))
        assert trader.run_cycle(BUNDLE, state, tracker) == "bought"
        assert tracker.position_for(DAYRANGE_ONLY) > 0
        assert tracker.position_for(TICKER) == 0
        assert broker.orders[0][0] == DAYRANGE_ONLY

    def test_a_symbol_that_is_not_streamed_says_so(self, market_open, monkeypatch):
        state = AppState()
        state.set_symbols([TICKER])
        state.api_key, state.api_secret, state.feed = "k", "s", "iex"
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(100.0))
        Reads(monkeypatch)
        trader = AppleTrader(confirm_config(ticker=DAYRANGE_ONLY))
        assert trader.run_cycle(BUNDLE, state, tracker) == "no_data"
        assert DAYRANGE_ONLY in state.agent_log[-1]["text"]

    def test_the_loop_refuses_the_pairing_before_it_loads_anything(
        self, state, monkeypatch
    ):
        """'There is no MSFT N-BEATS model' rather than 'the file is missing':
        different problems, different fixes."""
        loaded: list = []
        monkeypatch.setattr(
            at.apple_models, "load",
            lambda key, ticker=None: loaded.append((key, ticker)) or BUNDLE,
        )
        at._apple_trader_loop(
            state, tracker_for_loop(), AppleTraderConfig(
                model_key="nbeats", ticker=UNMODELLED
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
                model_key="dayrange", ticker=DAYRANGE_ONLY
            ), 60, threading.Event(),
        )
        assert asked == [("dayrange", DAYRANGE_ONLY)]


def tracker_for_loop() -> DecisionTracker:
    return DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(100.0))
