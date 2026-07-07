from datetime import datetime, timezone

import pytest

from marketview.performance import compute_equity_curve, decision_markers, summarize, total_fees_paid

SESSION_START = datetime(2024, 1, 15, 13, 20, tzinfo=timezone.utc)

BARS = [
    {"t": "2024-01-15T14:00:00Z", "o": 100.0, "h": 102.0, "l": 99.0, "c": 100.0, "v": 5000},
    {"t": "2024-01-15T14:01:00Z", "o": 100.0, "h": 103.0, "l": 100.0, "c": 102.0, "v": 3000},
    {"t": "2024-01-15T14:02:00Z", "o": 102.0, "h": 105.0, "l": 101.0, "c": 104.0, "v": 2000},
]


def _decision(ts: str, action: str, price: float, cash_after: float, position_after: float, status: str = "filled", fee: float = 1.15) -> dict:
    return {
        "ts": ts,
        "symbol": "AAPL",
        "action": action,
        "requested_quantity": 1,
        "filled_quantity": 1,
        "price": price,
        "reasoning": "x",
        "status": status,
        "cash_after": cash_after,
        "position_after": position_after,
        "fee": fee,
    }


class TestComputeEquityCurve:
    def test_empty_bars_returns_empty(self):
        assert compute_equity_curve({}, [], 1000.0, SESSION_START) == []

    def test_no_decisions_uses_starting_cash_throughout(self):
        points = compute_equity_curve({"AAPL": BARS}, [], 1000.0, SESSION_START)
        assert len(points) == 3
        assert all(p["value"] == 1000.0 for p in points)
        assert all(p["position"] == 0.0 for p in points)

    def test_decision_changes_value_from_its_timestamp_onward(self):
        decisions = [_decision("2024-01-15T14:00:30Z", "buy", 100.0, cash_after=898.85, position_after=1.0)]
        points = compute_equity_curve({"AAPL": BARS}, decisions, 1000.0, SESSION_START)
        assert points[0]["value"] == 1000.0  # before the decision
        assert points[1]["value"] == 898.85 + 1.0 * 102.0
        assert points[2]["value"] == 898.85 + 1.0 * 104.0

    def test_bars_before_session_start_are_excluded(self):
        old_bar = {"t": "2024-01-14T10:00:00Z", "o": 50.0, "h": 51.0, "l": 49.0, "c": 50.0, "v": 1000}
        points = compute_equity_curve({"AAPL": [old_bar] + BARS}, [], 1000.0, SESSION_START)
        assert len(points) == 3

    def test_live_price_appends_trailing_point_priced_off_live_price(self):
        decisions = [_decision("2024-01-15T14:00:30Z", "buy", 100.0, cash_after=898.85, position_after=1.0)]
        points = compute_equity_curve({"AAPL": BARS}, decisions, 1000.0, SESSION_START, live_prices={"AAPL": 110.0})
        assert len(points) == 4
        assert points[-1]["price"] == 110.0
        assert points[-1]["value"] == 898.85 + 1.0 * 110.0
        assert points[-1]["position"] == 1.0

    def test_live_price_works_with_no_bars(self):
        points = compute_equity_curve({}, [], 1000.0, SESSION_START, live_prices={"AAPL": 105.0})
        assert len(points) == 1
        assert points[0]["value"] == 1000.0

    def test_no_live_price_and_no_bars_returns_empty(self):
        assert compute_equity_curve({}, [], 1000.0, SESSION_START, live_prices=None) == []


class TestDecisionMarkers:
    def test_filters_out_sleep_and_rejected(self):
        decisions = [
            _decision("2024-01-15T14:00:30Z", "buy", 100.0, 898.85, 1.0),
            {"ts": "2024-01-15T14:01:00Z", "action": "sleep", "status": "noop", "price": None, "cash_after": 898.85, "position_after": 1.0},
            _decision("2024-01-15T14:01:30Z", "sell", 102.0, 999.85, 0.0, status="rejected"),
        ]
        markers = decision_markers(decisions, SESSION_START)
        assert len(markers) == 1
        assert markers[0]["action"] == "buy"
        assert markers[0]["value"] == 898.85 + 1.0 * 100.0

    def test_filters_out_decisions_before_session_start(self):
        decisions = [_decision("2024-01-15T10:00:00Z", "buy", 100.0, 898.85, 1.0)]
        assert decision_markers(decisions, SESSION_START) == []


class TestTotalFeesPaid:
    def test_sums_fees_across_decisions(self):
        decisions = [
            _decision("2024-01-15T14:00:30Z", "buy", 100.0, 898.85, 1.0, fee=1.15),
            _decision("2024-01-15T14:01:30Z", "sell", 102.0, 999.7, 0.0, fee=1.15),
        ]
        assert total_fees_paid(decisions) == 2.3

    def test_zero_for_no_decisions(self):
        assert total_fees_paid([]) == 0.0


class TestSummarize:
    def test_uses_starting_cash_when_no_points(self):
        stats = summarize([], [], 1000.0)
        assert stats["current_value"] == 1000.0
        assert stats["return_pct"] == 0.0

    def test_computes_return_and_fees(self):
        points = [{"ts": "t", "price": 100.0, "cash": 0.0, "position": 10.0, "value": 1100.0}]
        decisions = [_decision("2024-01-15T14:00:30Z", "buy", 100.0, 0.0, 10.0, fee=1.15)]
        stats = summarize(points, decisions, 1000.0)
        assert stats["current_value"] == 1100.0
        assert stats["return_pct"] == pytest.approx(10.0)
        assert stats["total_fees"] == 1.15
