import pytest

from agent_stonks.historical import (
    estimate_dividend_return_10y,
    estimate_total_return,
    fetch_static_analysis,
)


class FakeTicker:
    def __init__(self, info):
        self.info = info


class TestFetchStaticAnalysis:
    def test_returns_pe_dividend_yield_and_earnings_growth(self, monkeypatch):
        monkeypatch.setattr(
            "agent_stonks.historical.yf.Ticker",
            lambda symbol: FakeTicker(
                {
                    "trailingPE": 25.0,
                    "forwardPE": 22.0,
                    "trailingAnnualDividendYield": 0.02,
                    "earningsGrowth": 0.15,
                    "revenueGrowth": 0.1,
                }
            ),
        )
        result = fetch_static_analysis("AAPL")
        assert result == {
            "pe_ratio": 25.0,
            "forward_pe": 22.0,
            "dividend_yield": 0.02,
            "growth_rate": 0.15,
        }

    def test_falls_back_to_revenue_growth_when_earnings_growth_missing(self, monkeypatch):
        monkeypatch.setattr(
            "agent_stonks.historical.yf.Ticker",
            lambda symbol: FakeTicker({"revenueGrowth": 0.08}),
        )
        result = fetch_static_analysis("AAPL")
        assert result["growth_rate"] == 0.08

    def test_returns_none_values_on_failure(self, monkeypatch):
        def raise_error(symbol):
            raise RuntimeError("network error")

        monkeypatch.setattr("agent_stonks.historical.yf.Ticker", raise_error)
        result = fetch_static_analysis("AAPL")
        assert result == {
            "pe_ratio": None,
            "forward_pe": None,
            "dividend_yield": None,
            "growth_rate": None,
        }


class TestEstimateTotalReturn:
    def test_sums_dividend_yield_and_growth(self):
        assert estimate_total_return(0.02, 0.1) == pytest.approx(0.12)

    def test_returns_none_when_inputs_missing(self):
        assert estimate_total_return(None, 0.1) is None
        assert estimate_total_return(0.02, None) is None


class TestEstimateDividendReturn10y:
    def test_compounds_dividend_yield_over_years(self):
        result = estimate_dividend_return_10y(0.05, 0.0, years=3)
        assert result == pytest.approx(0.15)

    def test_accounts_for_growth(self):
        result = estimate_dividend_return_10y(0.05, 0.1, years=2)
        assert result == pytest.approx(0.05 + 0.05 * 1.1)

    def test_returns_none_when_inputs_missing(self):
        assert estimate_dividend_return_10y(None, 0.1) is None
        assert estimate_dividend_return_10y(0.05, None) is None
