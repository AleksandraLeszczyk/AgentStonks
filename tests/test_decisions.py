import pytest

from marketview.broker import Broker
from marketview.decisions import DecisionTracker


class FakeBroker(Broker):
    def __init__(self, price: float = 100.0):
        self.price = price
        self.orders: list[tuple] = []

    def get_current_price(self, symbol, key, secret, feed="iex") -> float:
        return self.price

    def submit_order(self, symbol, side, quantity, price) -> dict:
        self.orders.append((symbol, side, quantity, price))
        return {"status": "filled", "filled_qty": quantity, "filled_price": price}


class TestRecordSleep:
    def test_records_noop_decision_without_price(self):
        tracker = DecisionTracker(starting_cash=1000.0, broker=FakeBroker())
        decision = tracker.record_sleep("AAPL", "no clear edge")
        assert decision.action == "sleep"
        assert decision.status == "noop"
        assert decision.price is None
        assert decision.filled_quantity == 0
        assert decision.fee == 0.0
        assert tracker.cash == 1000.0
        assert tracker.position == 0.0


class TestRecordTradeBuy:
    def test_buy_deducts_cash_and_adds_position(self):
        tracker = DecisionTracker(starting_cash=1000.0, broker=FakeBroker(price=100.0), trade_cost=0.0)
        decision = tracker.record_trade("AAPL", "buy", 5, "bullish breakout", "k", "s")
        assert decision.status == "filled"
        assert decision.filled_quantity == 5
        assert decision.price == 100.0
        assert tracker.cash == 500.0
        assert tracker.position == 5.0

    def test_buy_clamps_to_affordable_quantity(self):
        tracker = DecisionTracker(starting_cash=250.0, broker=FakeBroker(price=100.0), trade_cost=0.0)
        decision = tracker.record_trade("AAPL", "buy", 10, "go big", "k", "s")
        assert decision.filled_quantity == 2.5
        assert tracker.cash == 0.0
        assert tracker.position == 2.5

    def test_buy_with_zero_cash_is_rejected(self):
        tracker = DecisionTracker(starting_cash=0.0, broker=FakeBroker(price=100.0))
        decision = tracker.record_trade("AAPL", "buy", 5, "go big", "k", "s")
        assert decision.status == "rejected"
        assert decision.filled_quantity == 0


class TestRecordTradeSell:
    def test_sell_clamps_to_current_position(self):
        broker = FakeBroker(price=100.0)
        tracker = DecisionTracker(starting_cash=0.0, broker=broker, trade_cost=0.0)
        tracker.position = 3.0
        decision = tracker.record_trade("AAPL", "sell", 10, "take profit", "k", "s")
        assert decision.filled_quantity == 3.0
        assert tracker.position == 0.0
        assert tracker.cash == 300.0

    def test_sell_with_no_position_is_rejected(self):
        tracker = DecisionTracker(starting_cash=0.0, broker=FakeBroker(price=100.0))
        decision = tracker.record_trade("AAPL", "sell", 5, "take profit", "k", "s")
        assert decision.status == "rejected"
        assert decision.filled_quantity == 0

    def test_invalid_action_raises(self):
        tracker = DecisionTracker(broker=FakeBroker())
        with pytest.raises(ValueError):
            tracker.record_trade("AAPL", "hold", 5, "?", "k", "s")


class TestTradeCostFee:
    def test_default_trade_cost_is_1_15(self):
        tracker = DecisionTracker(broker=FakeBroker())
        assert tracker.trade_cost == 1.15

    def test_buy_deducts_fee_on_top_of_cost(self):
        tracker = DecisionTracker(starting_cash=1000.0, broker=FakeBroker(price=100.0), trade_cost=1.0)
        decision = tracker.record_trade("AAPL", "buy", 5, "bullish breakout", "k", "s")
        assert decision.fee == 1.0
        assert tracker.cash == 1000.0 - 5 * 100.0 - 1.0
        assert tracker.position == 5.0

    def test_buy_affordable_quantity_reserves_fee(self):
        tracker = DecisionTracker(starting_cash=101.0, broker=FakeBroker(price=100.0), trade_cost=1.0)
        decision = tracker.record_trade("AAPL", "buy", 5, "go big", "k", "s")
        assert decision.filled_quantity == 1.0
        assert tracker.cash == 0.0

    def test_sell_deducts_fee_from_proceeds(self):
        tracker = DecisionTracker(starting_cash=0.0, broker=FakeBroker(price=100.0), trade_cost=1.0)
        tracker.position = 3.0
        decision = tracker.record_trade("AAPL", "sell", 3, "take profit", "k", "s")
        assert decision.fee == 1.0
        assert tracker.cash == 3 * 100.0 - 1.0

    def test_rejected_trade_has_no_fee(self):
        tracker = DecisionTracker(starting_cash=0.0, broker=FakeBroker(price=100.0), trade_cost=1.0)
        decision = tracker.record_trade("AAPL", "buy", 5, "go big", "k", "s")
        assert decision.status == "rejected"
        assert decision.fee == 0.0
        assert tracker.cash == 0.0


class TestSnapshotAndMarkers:
    def test_snapshot_reflects_cash_and_position(self):
        tracker = DecisionTracker(starting_cash=1000.0, broker=FakeBroker(price=50.0), trade_cost=0.0)
        tracker.record_trade("AAPL", "buy", 4, "x", "k", "s")
        snap = tracker.snapshot()
        assert snap["cash"] == 800.0
        assert snap["position"] == 4.0
        assert len(snap["decisions"]) == 1

    def test_trade_markers_excludes_sleep_and_rejected(self):
        tracker = DecisionTracker(starting_cash=1000.0, broker=FakeBroker(price=50.0), trade_cost=0.0)
        tracker.record_sleep("AAPL", "no edge")
        tracker.record_trade("AAPL", "buy", 4, "x", "k", "s")
        tracker.record_trade("AAPL", "sell", 100, "y", "k", "s")  # filled (clamped to 4)
        tracker.record_trade("AAPL", "sell", 5, "z", "k", "s")  # rejected, no position left
        markers = tracker.trade_markers()
        assert len(markers) == 2
        assert {m["action"] for m in markers} == {"buy", "sell"}
