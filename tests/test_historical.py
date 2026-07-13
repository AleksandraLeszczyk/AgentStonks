import pandas as pd
import pytest

from agent_stonks.historical import (
    estimate_dividend_return_10y,
    estimate_total_return,
    fetch_price_target_history,
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


class FakeTargetTicker:
    def __init__(self, upgrades_downgrades):
        self.upgrades_downgrades = upgrades_downgrades


def _target_actions(rows: list[tuple[str, int, float]]) -> pd.DataFrame:
    """Rows of (firm, days_ago, target) in yfinance upgrades_downgrades shape."""
    now = pd.Timestamp.now()
    return pd.DataFrame(
        {
            "Firm": [firm for firm, _, _ in rows],
            "ToGrade": ["Buy"] * len(rows),
            "FromGrade": ["Buy"] * len(rows),
            "Action": ["main"] * len(rows),
            "priceTargetAction": ["Raises"] * len(rows),
            "currentPriceTarget": [target for _, _, target in rows],
            "priorPriceTarget": [0.0] * len(rows),
        },
        index=pd.DatetimeIndex([now - pd.Timedelta(days=d) for _, d, _ in rows], name="GradeDate"),
    )


class TestFetchPriceTargetHistory:
    @pytest.fixture(autouse=True)
    def clear_cache(self, monkeypatch):
        monkeypatch.setattr("agent_stonks.historical._price_target_cache", {})

    def _patch(self, monkeypatch, actions):
        monkeypatch.setattr(
            "agent_stonks.historical.yf.Ticker",
            lambda symbol: FakeTargetTicker(actions),
        )

    def test_windows_events_and_carries_in_standing_target(self, monkeypatch):
        self._patch(monkeypatch, _target_actions([("Morgan Stanley", 100, 200.0), ("Morgan Stanley", 10, 250.0)]))
        result = fetch_price_target_history("AAPL", days=30)
        assert result["target"].tolist() == [200.0, 250.0]
        window_start = pd.Timestamp.now() - pd.Timedelta(days=30)
        # The stale event is carried in at the window start, not its real date.
        assert abs((result["date"].iloc[0] - window_start).total_seconds()) < 60
        assert result["date"].iloc[1] > window_start

    def test_drops_rows_without_a_published_target(self, monkeypatch):
        self._patch(monkeypatch, _target_actions([("Wedbush", 5, 0.0), ("Wedbush", 3, 300.0)]))
        result = fetch_price_target_history("AAPL", days=30)
        assert result["target"].tolist() == [300.0]

    def test_limits_to_most_recently_active_firms(self, monkeypatch):
        self._patch(
            monkeypatch,
            _target_actions([("Old Firm", 25, 100.0), ("Fresh Firm", 2, 200.0), ("Mid Firm", 10, 150.0)]),
        )
        result = fetch_price_target_history("AAPL", days=30, max_firms=2)
        assert set(result["firm"]) == {"Fresh Firm", "Mid Firm"}

    def test_returns_empty_frame_on_fetch_failure(self, monkeypatch):
        def raise_error(symbol):
            raise RuntimeError("network error")

        monkeypatch.setattr("agent_stonks.historical.yf.Ticker", raise_error)
        result = fetch_price_target_history("AAPL", days=30)
        assert result.empty
        assert list(result.columns) == ["firm", "date", "target"]

    def test_returns_empty_frame_when_no_actions(self, monkeypatch):
        self._patch(monkeypatch, None)
        result = fetch_price_target_history("AAPL", days=30)
        assert result.empty


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
