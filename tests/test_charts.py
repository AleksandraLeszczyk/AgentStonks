from datetime import datetime, timezone

import pandas as pd
import plotly.graph_objects as go
import pytest

from agent_stonks.charts import build_chart, build_historical_chart, build_performance_chart, empty_chart


SESSION_START = datetime(2024, 1, 15, 13, 20, tzinfo=timezone.utc)

BARS = [
    {"t": "2024-01-15T14:00:00Z", "o": 100.0, "h": 102.0, "l": 99.0, "c": 101.0, "v": 5000},
    {"t": "2024-01-15T14:01:00Z", "o": 101.0, "h": 103.0, "l": 100.5, "c": 102.0, "v": 3000},
    {"t": "2024-01-15T14:02:00Z", "o": 102.0, "h": 102.5, "l": 101.0, "c": 101.5, "v": 2000},
]

TRADES = [
    {"p": 100.5, "s": 100, "t": "2024-01-15T14:00:10Z"},
    {"p": 101.0, "s": 200, "t": "2024-01-15T14:00:30Z"},
    {"p": 101.5, "s": 150, "t": "2024-01-15T14:01:05Z"},
]

NEWS = [
    {
        "headline": "Apple beats earnings",
        "created_at": "2024-01-15T14:00:00Z",
        "url": "http://example.com",
    }
]


class TestEmptyChart:
    def test_returns_figure(self):
        assert isinstance(empty_chart(), go.Figure)

    def test_contains_custom_message(self):
        fig = empty_chart("Test message")
        texts = [a["text"] for a in fig.layout.annotations]
        assert "Test message" in texts

    def test_default_message(self):
        fig = empty_chart()
        texts = [a["text"] for a in fig.layout.annotations]
        assert any("symbol" in t.lower() or "start" in t.lower() for t in texts)


class TestBuildChart:
    def test_returns_figure_with_bars(self):
        fig = build_chart(BARS, [], [], "AAPL", SESSION_START)
        assert isinstance(fig, go.Figure)

    def test_empty_bars_returns_waiting_chart(self):
        fig = build_chart([], [], [], "AAPL", SESSION_START)
        texts = [a["text"] for a in fig.layout.annotations]
        assert any("Waiting" in t for t in texts)

    def test_title_contains_symbol_and_price(self):
        fig = build_chart(BARS, [], [], "TSLA", SESSION_START)
        assert "TSLA" in fig.layout.title.text
        assert "101.50" in fig.layout.title.text

    def test_works_with_empty_trades(self):
        fig = build_chart(BARS, [], [], "AAPL", SESSION_START)
        assert isinstance(fig, go.Figure)

    def test_works_with_empty_news(self):
        fig = build_chart(BARS, [], TRADES, "AAPL", SESSION_START)
        assert isinstance(fig, go.Figure)

    def test_works_with_trades_and_news(self):
        fig = build_chart(BARS, NEWS, TRADES, "AAPL", SESSION_START)
        assert isinstance(fig, go.Figure)

    def test_bars_before_session_start_are_filtered(self):
        old_bar = {"t": "2024-01-14T10:00:00Z", "o": 50.0, "h": 51.0, "l": 49.0, "c": 50.5, "v": 1000}
        fig = build_chart([old_bar] + BARS, [], [], "AAPL", SESSION_START)
        # Should still render (BARS are after session_start)
        assert "AAPL" in fig.layout.title.text

    def test_only_old_bars_returns_waiting_chart(self):
        old_bar = {"t": "2024-01-14T10:00:00Z", "o": 50.0, "h": 51.0, "l": 49.0, "c": 50.5, "v": 1000}
        fig = build_chart([old_bar], [], [], "AAPL", SESSION_START)
        texts = [a["text"] for a in fig.layout.annotations]
        assert any("Waiting" in t for t in texts)

    def test_decision_markers_plot_buy_and_sell(self):
        decisions = [
            {"ts": "2024-01-15T14:00:30Z", "action": "buy", "price": 100.5, "filled_quantity": 2, "status": "filled"},
            {"ts": "2024-01-15T14:01:30Z", "action": "sell", "price": 102.0, "filled_quantity": 2, "status": "filled"},
        ]
        fig = build_chart(BARS, [], [], "AAPL", SESSION_START, decisions=decisions)
        names = [t.name for t in fig.data]
        assert "Agent buy" in names
        assert "Agent sell" in names

    def test_decision_markers_ignored_when_no_price(self):
        decisions = [{"ts": "2024-01-15T14:00:30Z", "action": "sleep", "price": None, "filled_quantity": 0}]
        fig = build_chart(BARS, [], [], "AAPL", SESSION_START, decisions=decisions)
        names = [t.name for t in fig.data]
        assert "Agent sleep" not in names

    def test_no_decisions_does_not_error(self):
        fig = build_chart(BARS, [], [], "AAPL", SESSION_START, decisions=None)
        assert isinstance(fig, go.Figure)

    def test_price_alerts_plot_as_shapes(self):
        alerts = [
            {"field": "last_price", "condition": "above", "value": 150.0},
            {"field": "day_low", "condition": "below", "value": 95.0},
        ]
        fig = build_chart(BARS, [], [], "AAPL", SESSION_START, price_alerts=alerts)
        shape_levels = [s.y0 for s in fig.layout.shapes]
        assert 150.0 in shape_levels
        assert 95.0 in shape_levels
        texts = [a["text"] for a in fig.layout.annotations]
        assert any("above" in t and "150.00" in t for t in texts)
        assert any("below" in t and "95.00" in t for t in texts)

    def test_no_price_alerts_does_not_error(self):
        fig = build_chart(BARS, [], [], "AAPL", SESSION_START, price_alerts=None)
        assert isinstance(fig, go.Figure)


class TestBuildPerformanceChart:
    def test_no_points_returns_placeholder(self):
        fig = build_performance_chart([], [], "AAPL")
        texts = [a["text"] for a in fig.layout.annotations]
        assert any("No agent performance" in t for t in texts)

    def test_returns_figure_with_value_line(self):
        points = [
            {"ts": "2024-01-15T14:00:00Z", "price": 100.0, "cash": 1000.0, "position": 0.0, "value": 1000.0},
            {"ts": "2024-01-15T14:01:00Z", "price": 102.0, "cash": 1000.0, "position": 0.0, "value": 1000.0},
        ]
        fig = build_performance_chart(points, [], "AAPL")
        names = [t.name for t in fig.data]
        assert "Portfolio value" in names

    def test_markers_for_buy_and_sell_decisions(self):
        points = [{"ts": "2024-01-15T14:00:00Z", "price": 100.0, "cash": 900.0, "position": 1.0, "value": 1000.0}]
        markers = [
            {"ts": "2024-01-15T14:00:00Z", "action": "buy", "value": 1000.0},
            {"ts": "2024-01-15T14:00:30Z", "action": "sell", "value": 1005.0},
        ]
        fig = build_performance_chart(points, markers, "AAPL")
        names = [t.name for t in fig.data]
        assert "Agent buy" in names
        assert "Agent sell" in names


def _close_series(values: list[float], start: str = "2024-01-01") -> pd.Series:
    idx = pd.date_range(start, periods=len(values), freq="D")
    return pd.Series(values, index=idx)


class TestBuildHistoricalChart:
    def test_empty_ticker_returns_placeholder(self):
        fig = build_historical_chart(pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float), "AAPL", "1 Year")
        texts = [a["text"] for a in fig.layout.annotations]
        assert any("No historical data" in t for t in texts)

    def test_returns_figure_with_data(self):
        ticker = _close_series([100, 102, 105])
        spy = _close_series([400, 404, 410])
        vix = _close_series([15, 16, 14])
        fig = build_historical_chart(ticker, spy, vix, "AAPL", "1 Year")
        assert isinstance(fig, go.Figure)
        names = [t.name for t in fig.data]
        assert "AAPL" in names
        assert "SPY" in names
        assert "VIX" in names

    def test_ticker_normalized_to_percentage(self):
        ticker = _close_series([100, 110, 90])
        fig = build_historical_chart(ticker, pd.Series(dtype=float), pd.Series(dtype=float), "AAPL", "1 Year")
        ticker_trace = next(t for t in fig.data if t.name == "AAPL")
        assert list(ticker_trace.y) == pytest.approx([0.0, 10.0, -10.0])

    def test_price_targets_add_per_firm_step_lines(self):
        ticker = _close_series([100, 102, 105])
        targets = pd.DataFrame(
            {
                "firm": ["Morgan Stanley", "Morgan Stanley", "Wedbush"],
                "date": pd.to_datetime(["2023-12-30", "2024-01-02", "2024-01-01"]),
                "target": [110.0, 120.0, 130.0],
            }
        )
        fig = build_historical_chart(
            ticker, pd.Series(dtype=float), pd.Series(dtype=float), "AAPL", "1 Year",
            price_targets=targets,
        )
        names = [t.name for t in fig.data]
        assert "🎯 Morgan Stanley" in names
        assert "🎯 Wedbush" in names

        ms = next(t for t in fig.data if t.name == "🎯 Morgan Stanley")
        assert ms.line.shape == "hv"
        # Targets are on the % change scale relative to the first close (100),
        # the carry-in date is clipped to the plotted range, and the last
        # target is extended to the final close date.
        assert list(ms.y) == pytest.approx([10.0, 20.0, 20.0])
        assert pd.Timestamp(ms.x[0]) == ticker.index[0]
        assert pd.Timestamp(ms.x[-1]) == ticker.index[-1]

    def test_no_price_targets_adds_no_target_traces(self):
        ticker = _close_series([100, 102, 105])
        fig = build_historical_chart(
            ticker, pd.Series(dtype=float), pd.Series(dtype=float), "AAPL", "1 Year",
        )
        assert not any(t.name.startswith("🎯") for t in fig.data if t.name)

    def test_earnings_and_dividends_add_vlines(self):
        ticker = _close_series([100, 102, 105])
        earnings = pd.DataFrame({"EPS Estimate": [1.5]}, index=pd.DatetimeIndex(["2024-01-02"]))
        dividends = pd.Series([0.5], index=pd.DatetimeIndex(["2024-01-03"]))
        fig = build_historical_chart(
            ticker, pd.Series(dtype=float), pd.Series(dtype=float), "AAPL", "1 Year",
            dividends=dividends, earnings=earnings,
        )
        assert len(fig.layout.shapes) == 2


class TestFillIntradayGaps:
    GAPPY_BARS = [
        {"t": "2024-01-15T14:00:00Z", "o": 100.0, "h": 102.0, "l": 99.0, "c": 101.0, "v": 5000},
        {"t": "2024-01-15T14:01:00Z", "o": 101.0, "h": 103.0, "l": 100.5, "c": 102.0, "v": 3000},
        # 14:02 and 14:03 missing (no trades on the feed)
        {"t": "2024-01-15T14:04:00Z", "o": 102.0, "h": 102.5, "l": 101.0, "c": 101.5, "v": 2000},
    ]

    def _df(self, bars):
        df = pd.DataFrame(bars)
        df["t"] = pd.to_datetime(df["t"], utc=True)
        return df.sort_values("t").reset_index(drop=True)

    def test_fills_missing_buckets_with_flat_zero_volume_bars(self):
        from agent_stonks.charts import _fill_intraday_gaps

        filled = _fill_intraday_gaps(self._df(self.GAPPY_BARS))
        assert len(filled) == 5
        synth = filled[filled["synthetic"]]
        assert list(synth["t"].dt.strftime("%H:%M")) == ["14:02", "14:03"]
        # Flat at the previous close, zero volume
        assert (synth["o"] == 102.0).all()
        assert (synth["c"] == 102.0).all()
        assert (synth["v"] == 0).all()
        # Real bars untouched
        assert filled[~filled["synthetic"]]["v"].tolist() == [5000, 3000, 2000]

    def test_does_not_fill_across_days(self):
        from agent_stonks.charts import _fill_intraday_gaps

        bars = self.GAPPY_BARS + [
            {"t": "2024-01-16T14:00:00Z", "o": 103.0, "h": 104.0, "l": 102.0, "c": 103.5, "v": 1000},
        ]
        filled = _fill_intraday_gaps(self._df(bars))
        # Only the two intraday holes are filled -- not the overnight gap.
        assert int(filled["synthetic"].sum()) == 2

    def test_build_chart_with_fill_gaps_adds_no_trades_markers(self):
        fig = build_chart(self.GAPPY_BARS, [], [], "AAPL", SESSION_START, fill_gaps=True)
        names = [tr.name for tr in fig.data]
        assert "No trades" in names

    def test_build_chart_without_fill_gaps_has_no_markers(self):
        fig = build_chart(self.GAPPY_BARS, [], [], "AAPL", SESSION_START, fill_gaps=False)
        names = [tr.name for tr in fig.data]
        assert "No trades" not in names

    def test_build_chart_dedupes_mixed_timestamp_formats(self):
        # Same bucket delivered twice: REST 'Z' format and stream '+00:00' format.
        dup = dict(self.GAPPY_BARS[1], t="2024-01-15T14:01:00+00:00", c=999.0)
        fig = build_chart(self.GAPPY_BARS + [dup], [], [], "AAPL", SESSION_START)
        assert isinstance(fig, go.Figure)
