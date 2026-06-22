from datetime import datetime, timezone

import pandas as pd
import plotly.graph_objects as go
import pytest

from marketview.charts import build_chart, build_historical_chart, empty_chart


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

    def test_earnings_and_dividends_add_vlines(self):
        ticker = _close_series([100, 102, 105])
        earnings = pd.DataFrame({"EPS Estimate": [1.5]}, index=pd.DatetimeIndex(["2024-01-02"]))
        dividends = pd.Series([0.5], index=pd.DatetimeIndex(["2024-01-03"]))
        fig = build_historical_chart(
            ticker, pd.Series(dtype=float), pd.Series(dtype=float), "AAPL", "1 Year",
            dividends=dividends, earnings=earnings,
        )
        assert len(fig.layout.shapes) == 2
