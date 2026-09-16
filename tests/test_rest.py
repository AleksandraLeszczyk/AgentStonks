import pytest
import requests

from datetime import datetime, timedelta, timezone

from agent_stonks.rest import (
    fetch_bars,
    fetch_bars_window,
    fetch_corporate_actions,
    fetch_news,
    fetch_trades,
    _headers,
)


def test_headers():
    assert _headers("k", "s") == {"APCA-API-KEY-ID": "k", "APCA-API-SECRET-KEY": "s"}


class TestFetchBars:
    def test_returns_bars_for_symbol(self, requests_mock):
        requests_mock.get(
            "https://data.alpaca.markets/v2/stocks/bars",
            json={"bars": {"AAPL": [{"t": "2024-01-01T14:00:00Z", "o": 100, "h": 101, "l": 99, "c": 100.5, "v": 1000}]}},
        )
        bars = fetch_bars("AAPL", "1Min", 100, "key", "secret")
        assert len(bars) == 1
        assert bars[0]["c"] == 100.5

    def test_returns_empty_list_when_symbol_missing(self, requests_mock):
        requests_mock.get(
            "https://data.alpaca.markets/v2/stocks/bars",
            json={"bars": {}},
        )
        assert fetch_bars("AAPL", "1Min", 100, "key", "secret") == []

    def test_raises_on_http_error(self, requests_mock):
        requests_mock.get(
            "https://data.alpaca.markets/v2/stocks/bars",
            status_code=403,
        )
        with pytest.raises(requests.HTTPError):
            fetch_bars("AAPL", "1Min", 100, "key", "secret")

    def test_asks_for_the_newest_bars_of_the_window(self, requests_mock):
        """`limit` is a cap on one page, and a lookback of minute bars routinely
        exceeds it -- 16 hours is over 600 one-minute bars on a liquid symbol.
        Ascending (Alpaca's default) would return the *start* of the window and
        throw away the page token carrying the rest, so a buffer seed or a
        backfill would stop hours short of now and never fill a recent hole."""
        requests_mock.get(
            "https://data.alpaca.markets/v2/stocks/bars", json={"bars": {"AAPL": []}}
        )
        fetch_bars("AAPL", "1Min", 420, "key", "secret", "sip", lookback_hours=16)
        assert requests_mock.last_request.qs["sort"] == ["desc"]

    def test_returns_them_oldest_first_however_they_arrived(self, requests_mock):
        """Everything downstream reads these ascending -- `bars[-1]` is the
        latest bar, the volume profile walks the session forward."""
        requests_mock.get(
            "https://data.alpaca.markets/v2/stocks/bars",
            json={"bars": {"AAPL": [
                {"t": "2024-01-01T14:02:00Z", "c": 3},
                {"t": "2024-01-01T14:01:00Z", "c": 2},
                {"t": "2024-01-01T14:00:00Z", "c": 1},
            ]}},
        )
        bars = fetch_bars("AAPL", "1Min", 3, "key", "secret")
        assert [b["c"] for b in bars] == [1, 2, 3]


class TestFetchBarsWindow:
    """`keep` decides which end of an over-long window survives `limit`."""

    START = datetime(2024, 1, 1, 14, 0, tzinfo=timezone.utc)

    def _window(self, requests_mock, **kwargs):
        requests_mock.get(
            "https://data.alpaca.markets/v2/stocks/bars", json={"bars": {"AAPL": []}}
        )
        return fetch_bars_window(
            "AAPL", "1Min", self.START, self.START + timedelta(hours=16),
            "key", "secret", "sip", limit=420, **kwargs,
        )

    def test_a_window_anchored_on_its_start_keeps_the_oldest(self, requests_mock):
        """The default, and what the 09:30 opening range needs: the first five
        minutes of the session, not the last five of the window."""
        self._window(requests_mock)
        assert requests_mock.last_request.qs["sort"] == ["asc"]

    def test_a_window_anchored_on_now_keeps_the_newest(self, requests_mock):
        self._window(requests_mock, keep="newest")
        assert requests_mock.last_request.qs["sort"] == ["desc"]

    def test_both_come_back_oldest_first(self, requests_mock):
        requests_mock.get(
            "https://data.alpaca.markets/v2/stocks/bars",
            json={"bars": {"AAPL": [
                {"t": "2024-01-01T14:01:00Z", "c": 2},
                {"t": "2024-01-01T14:00:00Z", "c": 1},
            ]}},
        )
        bars = fetch_bars_window(
            "AAPL", "1Min", self.START, self.START + timedelta(hours=1),
            "key", "secret", "sip", keep="newest",
        )
        assert [b["c"] for b in bars] == [1, 2]


class TestFetchTrades:
    def test_returns_trades_for_symbol(self, requests_mock):
        requests_mock.get(
            "https://data.alpaca.markets/v2/stocks/trades",
            json={"trades": {"MSFT": [{"p": 300.0, "s": 10, "t": "2024-01-01T14:01:00Z"}]}},
        )
        trades = fetch_trades("MSFT", "key", "secret")
        assert len(trades) == 1
        assert trades[0]["p"] == 300.0

    def test_raises_on_http_error(self, requests_mock):
        requests_mock.get(
            "https://data.alpaca.markets/v2/stocks/trades",
            status_code=401,
        )
        with pytest.raises(requests.HTTPError):
            fetch_trades("MSFT", "key", "secret")


class TestFetchCorporateActions:
    def test_flattens_grouped_response_chronologically(self, requests_mock):
        requests_mock.get(
            "https://data.alpaca.markets/v1beta1/corporate-actions",
            json={
                "corporate_actions": {
                    "cash_dividends": [
                        {"symbol": "AAPL", "rate": 0.25, "ex_date": "2026-07-20", "payable_date": "2026-08-01"}
                    ],
                    "forward_splits": [
                        {"symbol": "AAPL", "new_rate": 4, "old_rate": 1, "ex_date": "2026-07-15"}
                    ],
                }
            },
        )
        actions = fetch_corporate_actions("AAPL", "key", "secret")
        assert [a["type"] for a in actions] == ["forward_split", "cash_dividend"]
        assert [a["date"] for a in actions] == ["2026-07-15", "2026-07-20"]
        assert actions[1]["rate"] == 0.25

    def test_falls_back_through_date_fields(self, requests_mock):
        requests_mock.get(
            "https://data.alpaca.markets/v1beta1/corporate-actions",
            json={
                "corporate_actions": {
                    "cash_mergers": [
                        {"acquiree_symbol": "AAPL", "rate": 10.0, "effective_date": "2026-07-18"}
                    ],
                    "name_changes": [
                        {"old_symbol": "AAPL", "new_symbol": "AAPL2", "process_date": "2026-07-16"}
                    ],
                }
            },
        )
        actions = fetch_corporate_actions("AAPL", "key", "secret")
        assert [a["date"] for a in actions] == ["2026-07-16", "2026-07-18"]

    def test_returns_empty_when_nothing_scheduled(self, requests_mock):
        requests_mock.get(
            "https://data.alpaca.markets/v1beta1/corporate-actions",
            json={"corporate_actions": {}},
        )
        assert fetch_corporate_actions("AAPL", "key", "secret") == []

    def test_raises_on_http_error(self, requests_mock):
        requests_mock.get(
            "https://data.alpaca.markets/v1beta1/corporate-actions",
            status_code=500,
        )
        with pytest.raises(requests.HTTPError):
            fetch_corporate_actions("AAPL", "key", "secret")


class TestFetchNews:
    def test_returns_news_list(self, requests_mock):
        requests_mock.get(
            "https://data.alpaca.markets/v1beta1/news",
            json={"news": [{"headline": "AAPL rises", "summary": "Apple stock up", "created_at": "2024-01-01T12:00:00Z", "url": "http://example.com"}]},
        )
        news = fetch_news("AAPL", "key", "secret")
        assert len(news) == 1
        assert news[0]["headline"] == "AAPL rises"

    def test_returns_empty_when_no_news(self, requests_mock):
        requests_mock.get(
            "https://data.alpaca.markets/v1beta1/news",
            json={"news": []},
        )
        assert fetch_news("AAPL", "key", "secret") == []

    def test_raises_on_http_error(self, requests_mock):
        requests_mock.get(
            "https://data.alpaca.markets/v1beta1/news",
            status_code=500,
        )
        with pytest.raises(requests.HTTPError):
            fetch_news("AAPL", "key", "secret")
