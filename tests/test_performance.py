from datetime import datetime, timezone

import pytest

from agent_stonks.performance import (
    append_live_point,
    compute_equity_curve,
    thin_live_points,
    compute_symbol_curves,
    decision_markers,
    holdings,
    summarize,
    symbol_books,
    total_fees_paid,
)

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

    def test_carries_what_the_hover_shows(self):
        (marker,) = decision_markers([_decision("2024-01-15T14:00:30Z", "buy", 100.0, 898.85, 1.0)], SESSION_START)
        assert (marker["price"], marker["filled_quantity"], marker["reasoning"]) == (100.0, 1, "x")

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

    def test_an_explicit_current_value_overrides_the_curve(self):
        # A real broker account knows things this replayed curve cannot: what
        # it held before the session, symbols the app never streamed, cash that
        # moved elsewhere. When it says what the portfolio is worth, it wins.
        points = [{"ts": "t", "price": 100.0, "cash": 0.0, "position": 10.0, "value": 1100.0}]
        stats = summarize(points, [], 1000.0, 1250.0)
        assert stats["current_value"] == 1250.0
        assert stats["return_pct"] == pytest.approx(25.0)

    def test_no_override_falls_back_to_the_curve(self):
        points = [{"ts": "t", "price": 100.0, "cash": 0.0, "position": 10.0, "value": 1100.0}]
        assert summarize(points, [], 1000.0, None)["current_value"] == 1100.0


# --- per instrument -------------------------------------------------------

NVDA_BARS = [
    {"t": "2024-01-15T14:00:00Z", "o": 50.0, "h": 51.0, "l": 49.0, "c": 50.0, "v": 9000},
    {"t": "2024-01-15T14:01:00Z", "o": 50.0, "h": 52.0, "l": 50.0, "c": 51.0, "v": 8000},
    {"t": "2024-01-15T14:02:00Z", "o": 51.0, "h": 51.0, "l": 48.0, "c": 49.0, "v": 7000},
]


def _fill(ts: str, symbol: str, action: str, qty: float, price: float,
          cash_after: float = 0.0, positions_after: dict | None = None, fee: float = 1.0) -> dict:
    positions_after = positions_after or {}
    return {
        "ts": ts,
        "symbol": symbol,
        "action": action,
        "requested_quantity": qty,
        "filled_quantity": qty,
        "price": price,
        "reasoning": "x",
        "status": "filled",
        "cash_after": cash_after,
        "position_after": positions_after.get(symbol, 0.0),
        "positions_after": positions_after,
        "fee": fee,
    }


# Starting with $10,000 and a $1 fee a fill: AAPL bought and sold, NVDA bought.
BASKET = [
    _fill("2024-01-15T14:00:30Z", "AAPL", "buy", 10, 100.5, 8994.0, {"AAPL": 10.0}),
    _fill("2024-01-15T14:00:40Z", "NVDA", "buy", 20, 50.5, 7983.0, {"AAPL": 10.0, "NVDA": 20.0}),
    _fill("2024-01-15T14:01:30Z", "AAPL", "sell", 10, 103.0, 9012.0, {"AAPL": 0.0, "NVDA": 20.0}),
]


class TestSymbolBooks:
    def test_average_cost_and_what_the_sells_realized(self):
        book = symbol_books([
            _fill("2024-01-15T14:00:10Z", "AAPL", "buy", 10, 100.0),
            _fill("2024-01-15T14:00:20Z", "AAPL", "buy", 10, 110.0),
            _fill("2024-01-15T14:00:30Z", "AAPL", "sell", 5, 120.0),
        ])["AAPL"]
        assert book.shares == 15
        assert book.avg_cost == pytest.approx(105.0)
        assert book.realized == pytest.approx(75.0)  # 5 × (120 − 105)
        assert (book.fees, book.buys, book.sells) == (3.0, 2, 1)
        assert book.open_pnl(100.0) == pytest.approx(-75.0)
        assert book.pnl(100.0) == pytest.approx(75.0 - 75.0 - 3.0)

    def test_selling_more_than_the_ledger_bought_realizes_only_what_it_bought(self):
        # A real account can sell shares it held before the run; their cost is
        # not in this ledger, so their sale realizes nothing here.
        book = symbol_books([
            _fill("2024-01-15T14:00:10Z", "AAPL", "buy", 5, 100.0, fee=0.0),
            _fill("2024-01-15T14:00:20Z", "AAPL", "sell", 8, 110.0, fee=0.0),
        ])["AAPL"]
        assert book.realized == pytest.approx(50.0)
        assert book.shares == 0 and book.avg_cost is None
        assert book.open_pnl(120.0) is None

    def test_only_fills_count(self):
        rejected = {**_fill("2024-01-15T14:00:10Z", "AAPL", "buy", 5, 100.0), "status": "rejected"}
        tactics = {**_fill("2024-01-15T14:00:20Z", "AAPL", "tactics", 0, 100.0), "status": "armed"}
        assert symbol_books([rejected, tactics]) == {}


class TestHoldings:
    def test_held_first_by_market_value_then_the_symbols_traded_flat(self):
        decisions = [
            _fill("2024-01-15T14:00:10Z", "AAPL", "buy", 10, 100.0),
            _fill("2024-01-15T14:00:20Z", "NVDA", "buy", 100, 50.0),
            _fill("2024-01-15T14:00:30Z", "MU", "buy", 5, 20.0),
            _fill("2024-01-15T14:00:40Z", "MU", "sell", 5, 22.0),
        ]
        rows = holdings(
            decisions,
            {"AAPL": 10.0, "NVDA": 100.0, "MU": 0.0},
            {"AAPL": 110.0, "NVDA": 52.0, "MU": 21.0},
            portfolio_value=10_000.0,
        )
        assert [r["symbol"] for r in rows] == ["NVDA", "AAPL", "MU"]
        aapl = rows[1]
        assert aapl["market_value"] == pytest.approx(1100.0)
        assert aapl["weight_pct"] == pytest.approx(11.0)
        assert aapl["avg_cost"] == pytest.approx(100.0)
        assert aapl["open_pnl"] == pytest.approx(100.0)
        assert aapl["open_pct"] == pytest.approx(10.0)
        assert aapl["net_pnl"] == pytest.approx(99.0)
        mu = rows[2]
        assert mu["shares"] == 0 and mu["open_pnl"] is None and mu["avg_cost"] is None
        assert mu["closed_pnl"] == pytest.approx(10.0)
        assert mu["net_pnl"] == pytest.approx(8.0)
        assert (mu["buys"], mu["sells"]) == (1, 1)

    def test_shares_no_fill_bought_are_untracked(self):
        # The account holds 15; this ledger bought 10 of them.
        rows = holdings(
            [_fill("2024-01-15T14:00:10Z", "AAPL", "buy", 10, 100.0, fee=0.0)],
            {"AAPL": 15.0},
            {"AAPL": 110.0},
        )
        (aapl,) = rows
        assert aapl["shares"] == 15 and aapl["untracked"] == 5
        assert aapl["market_value"] == pytest.approx(1650.0)
        assert aapl["open_pnl"] == pytest.approx(100.0)  # on the 10 it bought
        assert aapl["weight_pct"] is None  # no portfolio value given

    def test_a_held_symbol_without_a_price_has_no_value(self):
        (row,) = holdings([], {"TSLA": 3.0}, {"TSLA": None}, portfolio_value=1000.0)
        assert row["market_value"] is None and row["weight_pct"] is None
        assert row["untracked"] == 3


class TestComputeSymbolCurves:
    BARS_BY_SYMBOL = {"AAPL": BARS, "NVDA": NVDA_BARS}

    def test_symbol_pnls_add_up_to_the_equity_curve(self):
        curves = compute_symbol_curves(self.BARS_BY_SYMBOL, BASKET, 10_000.0, SESSION_START)
        equity = compute_equity_curve(self.BARS_BY_SYMBOL, BASKET, 10_000.0, SESSION_START)
        # The equity curve's value once every symbol's bar at a time is in.
        value_at = {p["ts"]: p["value"] for p in equity}
        bar_points = [p for p in curves["points"] if p["ts"] in value_at]
        assert [p["ts"][11:16] for p in bar_points] == ["14:00", "14:01", "14:02"]
        for point in bar_points:
            assert sum(point["pnl"].values()) == pytest.approx(value_at[point["ts"]] - 10_000.0)
        last = bar_points[-1]
        assert last["pnl"] == pytest.approx({"AAPL": 23.0, "NVDA": -31.0})
        assert last["market_value"] == pytest.approx({"AAPL": 0.0, "NVDA": 980.0})
        assert last["cash"] == 9012.0

    def test_a_point_at_every_fill_and_each_fill_on_its_own_line(self):
        curves = compute_symbol_curves(self.BARS_BY_SYMBOL, BASKET, 10_000.0, SESSION_START)
        assert len(curves["points"]) == 3 + 3  # three bar times, three fills
        assert [(f["symbol"], f["action"]) for f in curves["fills"]] == [
            ("AAPL", "buy"), ("NVDA", "buy"), ("AAPL", "sell"),
        ]
        # Right after a buy the symbol has lost only its fee.
        assert curves["fills"][0]["pnl"] == pytest.approx(-1.0)
        assert curves["fills"][2]["pnl"] == pytest.approx(10 * (103.0 - 100.5) - 2.0)

    def test_fills_before_session_start_count_but_are_not_marked(self):
        late_start = datetime(2024, 1, 15, 14, 0, 45, tzinfo=timezone.utc)
        curves = compute_symbol_curves(self.BARS_BY_SYMBOL, BASKET, 10_000.0, late_start)
        assert [f["action"] for f in curves["fills"]] == ["sell"]
        first = curves["points"][0]
        assert set(first["pnl"]) == {"AAPL", "NVDA"}

    def test_live_prices_add_a_now_point(self):
        curves = compute_symbol_curves(
            self.BARS_BY_SYMBOL, BASKET, 10_000.0, SESSION_START, live_prices={"NVDA": 55.0},
        )
        now = curves["points"][-1]
        assert now["pnl"]["NVDA"] == pytest.approx(20 * 55.0 - 1010.0 - 1.0)
        assert now["market_value"]["NVDA"] == pytest.approx(1100.0)

    def test_nothing_traded_is_empty_per_symbol(self):
        curves = compute_symbol_curves(self.BARS_BY_SYMBOL, [], 10_000.0, SESSION_START)
        assert all(p["pnl"] == {} and p["market_value"] == {} for p in curves["points"])
        assert curves["fills"] == []


class TestAppendLivePoint:
    """The Portfolio value chart's once-a-minute live reading."""

    @staticmethod
    def _never_open(ts):
        return False

    @staticmethod
    def _point(minute, value=4001.8, cash=4001.8, position=0):
        return {
            "ts": f"2026-11-13T06:{minute:02d}:24+00:00",
            "price": None, "cash": cash, "position": position, "value": value,
        }

    def test_an_unchanged_night_keeps_its_first_and_newest_point(self):
        history = []
        for minute in range(33, 40):
            append_live_point(history, self._point(minute), self._never_open)
        assert [p["ts"][11:16] for p in history] == ["06:33", "06:39"]

    def test_a_change_starts_a_new_stretch(self):
        history = []
        for minute in (33, 34, 35):
            append_live_point(history, self._point(minute), self._never_open)
        for minute in (36, 37, 38):
            append_live_point(history, self._point(minute, value=4005.0), self._never_open)
        assert [(p["ts"][11:16], p["value"]) for p in history] == [
            ("06:33", 4001.8), ("06:35", 4001.8), ("06:36", 4005.0), ("06:38", 4005.0),
        ]

    def test_in_the_session_every_reading_is_kept(self):
        history = []
        for minute in range(33, 37):
            append_live_point(history, self._point(minute), lambda ts: True)
        assert len(history) == 4

    def test_points_recorded_before_are_thinned_the_same_way(self):
        from agent_stonks import market_hours

        night = [self._point(minute) for minute in range(33, 40)]
        session = [
            {**self._point(minute), "ts": f"2026-11-12T15:{minute:02d}:24+00:00"}
            for minute in range(0, 3)
        ]
        kept = thin_live_points(session + night, market_hours.is_market_open)
        assert [p["ts"][:16] for p in kept] == [
            "2026-11-12T15:00", "2026-11-12T15:01", "2026-11-12T15:02",
            "2026-11-13T06:33", "2026-11-13T06:39",
        ]
