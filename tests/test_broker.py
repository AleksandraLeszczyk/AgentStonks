import pytest
import requests

from agent_stonks.broker import PaperBroker


class TestPaperBrokerGetCurrentPrice:
    def test_returns_latest_trade_price(self, requests_mock):
        requests_mock.get(
            "https://data.alpaca.markets/v2/stocks/AAPL/trades/latest",
            json={"trade": {"p": 123.45, "s": 10}},
        )
        broker = PaperBroker()
        assert broker.get_current_price("AAPL", "key", "secret") == 123.45

    def test_raises_when_no_trade_price(self, requests_mock):
        requests_mock.get(
            "https://data.alpaca.markets/v2/stocks/AAPL/trades/latest",
            json={"trade": {}},
        )
        broker = PaperBroker()
        with pytest.raises(RuntimeError):
            broker.get_current_price("AAPL", "key", "secret")

    def test_raises_on_http_error(self, requests_mock):
        requests_mock.get(
            "https://data.alpaca.markets/v2/stocks/AAPL/trades/latest",
            status_code=403,
        )
        broker = PaperBroker()
        with pytest.raises(requests.HTTPError):
            broker.get_current_price("AAPL", "key", "secret")


class TestPaperBrokerSubmitOrder:
    def test_returns_filled_report(self):
        broker = PaperBroker()
        report = broker.submit_order("AAPL", "buy", 10, 123.45)
        assert report == {"status": "filled", "filled_qty": 10, "filled_price": 123.45}
