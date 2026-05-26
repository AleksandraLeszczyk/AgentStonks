import pytest
import requests

from marketview.rest import fetch_bars, fetch_news, fetch_trades, _headers


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
