import pandas as pd
import pytest

from agent_stonks.historical import (
    estimate_dividend_return_10y,
    estimate_total_return,
    fetch_analyst_targets,
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


class FakeAnalystTicker:
    def __init__(self, info=None, upgrades_downgrades=None):
        if info is None:
            raise RuntimeError("no info")
        self.info = info
        self.upgrades_downgrades = upgrades_downgrades


class TestFetchAnalystTargets:
    @pytest.fixture(autouse=True)
    def clear_caches(self, monkeypatch):
        monkeypatch.setattr("agent_stonks.historical._analyst_targets_cache", {})
        monkeypatch.setattr("agent_stonks.historical._price_target_cache", {})

    def _patch(self, monkeypatch, info=None, actions=None):
        monkeypatch.setattr(
            "agent_stonks.historical.yf.Ticker",
            lambda symbol: FakeAnalystTicker(info=info, upgrades_downgrades=actions),
        )

    def _info(self, **overrides):
        base = {
            "targetMeanPrice": 210.0,
            "targetMedianPrice": 208.0,
            "targetHighPrice": 250.0,
            "targetLowPrice": 180.0,
            "numberOfAnalystOpinions": 30,
            "recommendationKey": "buy",
            "currentPrice": 200.0,
        }
        base.update(overrides)
        return base

    def test_consensus_and_tracked_firm_targets_with_upside(self, monkeypatch):
        actions = _target_actions(
            [("UBS", 5, 220.0), ("Morgan Stanley", 8, 205.0), ("Barclays", 3, 215.0), ("Wedbush", 2, 260.0)]
        )
        self._patch(monkeypatch, info=self._info(), actions=actions)

        result = fetch_analyst_targets("AAPL", current_price=200.0)

        assert result["consensus"]["mean"] == 210.0
        assert result["consensus"]["mean_upside_pct"] == 5.0
        assert result["consensus"]["num_analysts"] == 30
        # Only the tracked bulge-bracket firms are surfaced individually.
        assert set(result["firms"]) == {"UBS", "Morgan Stanley", "Barclays"}
        assert result["firms"]["UBS"]["target"] == 220.0
        assert result["firms"]["UBS"]["upside_pct"] == 10.0

    def test_takes_most_recent_target_per_firm(self, monkeypatch):
        actions = _target_actions([("UBS", 40, 190.0), ("UBS", 4, 230.0)])
        self._patch(monkeypatch, info=self._info(), actions=actions)

        result = fetch_analyst_targets("AAPL", current_price=200.0)
        assert result["firms"]["UBS"]["target"] == 230.0

    def test_current_price_arg_overrides_yahoo_price(self, monkeypatch):
        self._patch(monkeypatch, info=self._info(currentPrice=100.0), actions=None)

        # 210 mean vs 200 live = +5%, not the +110% Yahoo's stale 100 would imply.
        result = fetch_analyst_targets("AAPL", current_price=200.0)
        assert result["current_price"] == 200.0
        assert result["consensus"]["mean_upside_pct"] == 5.0

    def test_falls_back_to_yahoo_price_when_arg_missing(self, monkeypatch):
        self._patch(monkeypatch, info=self._info(currentPrice=200.0), actions=None)

        result = fetch_analyst_targets("AAPL")
        assert result["current_price"] == 200.0
        assert result["consensus"]["mean_upside_pct"] == 5.0

    def test_price_above_mean_flags_exhausted_upside(self, monkeypatch):
        self._patch(monkeypatch, info=self._info(), actions=None)

        result = fetch_analyst_targets("AAPL", current_price=215.0)
        assert result["consensus"]["mean_upside_pct"] == pytest.approx(-2.3, abs=0.1)
        assert any("ABOVE the consensus mean" in i for i in result["insights"])

    def test_price_above_highest_target_flagged(self, monkeypatch):
        self._patch(monkeypatch, info=self._info(), actions=None)

        result = fetch_analyst_targets("AAPL", current_price=300.0)
        assert any("HIGHEST analyst target" in i for i in result["insights"])

    def test_no_data_returns_empty_with_summary(self, monkeypatch):
        def raise_error(symbol):
            raise RuntimeError("network error")

        monkeypatch.setattr("agent_stonks.historical.yf.Ticker", raise_error)
        result = fetch_analyst_targets("AAPL", current_price=200.0)
        assert result["consensus"]["mean"] is None
        assert result["firms"] == {}
        assert result["summary"] == "No analyst price targets available."
        assert result["insights"] == []


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
