"""Finnhub alternative-data fetchers, summarisers, and the briefing block.

Every response here is mocked; nothing reaches Finnhub.
"""
import pytest

from agent_stonks import finnhub_rest as fh
from agent_stonks import premarket

BASE = "https://finnhub.io/api/v1"


@pytest.fixture(autouse=True)
def _clear_cache():
    """The module caches per (endpoint, symbol, window) for half an hour, which
    would leak one test's fixture into the next."""
    fh.clear_cache()
    yield
    fh.clear_cache()


def _tx(code, change, price=100.0, name="COOK TIMOTHY D", date="2026-08-25"):
    return {
        "name": name, "change": change, "transactionCode": code,
        "transactionPrice": price, "transactionDate": date,
    }


class TestHttpErrors:
    def test_missing_token_fails_without_a_request(self):
        with pytest.raises(fh.FinnhubError, match="no FINNHUB_API_KEY"):
            fh._get("/stock/lobbying", "")

    def test_premium_endpoint_says_so_rather_than_looking_empty(self, requests_mock):
        # A 403 reported as "no data" would read as "nothing is happening at
        # this company", which is a different and wrong statement.
        requests_mock.get(f"{BASE}/stock/lobbying", status_code=403, json={})
        with pytest.raises(fh.FinnhubError, match="not available on this Finnhub plan"):
            fh._get("/stock/lobbying", "tok")

    def test_bad_key_is_distinguishable(self, requests_mock):
        requests_mock.get(f"{BASE}/stock/lobbying", status_code=401, json={})
        with pytest.raises(fh.FinnhubError, match="rejected the API key"):
            fh._get("/stock/lobbying", "tok")

    def test_rate_limit_is_distinguishable(self, requests_mock):
        requests_mock.get(f"{BASE}/stock/lobbying", status_code=429, json={})
        with pytest.raises(fh.FinnhubError, match="rate limit"):
            fh._get("/stock/lobbying", "tok")

    def test_sends_the_symbol_and_a_date_window(self, requests_mock):
        requests_mock.get(f"{BASE}/stock/insider-transactions", json={"data": []})
        fh.fetch_insider_transactions("aapl", "tok", days=30)
        q = requests_mock.last_request.qs  # requests_mock lower-cases values
        assert q["symbol"] == ["aapl"]  # sent upper-cased; compared lower here
        assert q["token"] == ["tok"]
        assert "from" in q and "to" in q
        assert "AAPL" in requests_mock.last_request.url  # the wire really is upper

    def test_caches_within_the_ttl(self, requests_mock):
        m = requests_mock.get(f"{BASE}/stock/lobbying", json={"data": [{"year": 2025}]})
        fh.fetch_lobbying("AAPL", "tok")
        fh.fetch_lobbying("AAPL", "tok")
        assert m.call_count == 1

    def test_caches_per_symbol(self, requests_mock):
        m = requests_mock.get(f"{BASE}/stock/lobbying", json={"data": []})
        fh.fetch_lobbying("AAPL", "tok")
        fh.fetch_lobbying("TSLA", "tok")
        assert m.call_count == 2


class TestInsiderTransactions:
    def test_separates_open_market_trades_from_compensation(self):
        # The classic way to misread an insider feed: a scheduled vest (A) or
        # tax withholding (F) looks like a huge trade nobody decided to make.
        rows = [
            _tx("P", 1000), _tx("S", -500),
            _tx("A", 50_000), _tx("M", 20_000), _tx("F", -8_000), _tx("G", -100),
        ]
        out = fh.summarize_insider_transactions(rows)
        assert out["buy_count"] == 1
        assert out["sell_count"] == 1
        assert out["other_count"] == 4
        assert out["net_shares"] == 500  # 1000 bought - 500 sold, comp excluded

    def test_values_the_two_sides_in_dollars(self):
        rows = [_tx("P", 100, price=10.0), _tx("S", -50, price=20.0)]
        out = fh.summarize_insider_transactions(rows)
        assert out["buy_value"] == 1000.0
        assert out["sell_value"] == 1000.0

    def test_uses_absolute_share_counts_per_side(self):
        # Alpaca-style signed `change` must not make sell_shares negative.
        out = fh.summarize_insider_transactions([_tx("S", -400)])
        assert out["sell_shares"] == 400.0

    def test_names_the_most_active_insiders(self):
        rows = [_tx("S", -1, name="COOK TIMOTHY D") for _ in range(3)]
        rows.append(_tx("P", 1, name="LEVINSON ARTHUR D"))
        out = fh.summarize_insider_transactions(rows)
        assert out["insiders"][0] == "COOK TIMOTHY D"

    def test_compensation_only_feed_reports_no_trades(self):
        out = fh.summarize_insider_transactions([_tx("A", 10_000), _tx("F", -3_000)])
        assert out["buy_count"] == 0 and out["sell_count"] == 0
        assert out["other_count"] == 2

    def test_handles_missing_prices_without_raising(self):
        out = fh.summarize_insider_transactions(
            [{"transactionCode": "S", "change": -10}]
        )
        assert out["sell_value"] == 0.0
        assert out["sell_shares"] == 10.0


class TestInsiderSentiment:
    def test_orders_by_period_so_latest_is_the_newest(self):
        rows = [
            {"year": 2026, "month": 3, "mspr": -20.0, "change": -100},
            {"year": 2026, "month": 8, "mspr": -100.0, "change": -500},
            {"year": 2026, "month": 2, "mspr": 25.0, "change": 50},
        ]
        out = fh.summarize_insider_sentiment(rows)
        assert out["latest_period"] == "2026-08"
        assert out["latest_mspr"] == -100.0
        assert out["months"] == 3
        assert out["positive_months"] == 1

    def test_averages_across_the_window(self):
        rows = [
            {"year": 2026, "month": 1, "mspr": 10.0, "change": 0},
            {"year": 2026, "month": 2, "mspr": 30.0, "change": 0},
        ]
        assert fh.summarize_insider_sentiment(rows)["mean_mspr"] == 20.0

    def test_empty_when_no_readings(self):
        assert fh.summarize_insider_sentiment([]) == {}
        assert fh.summarize_insider_sentiment([{"year": 2026, "month": 1}]) == {}


class TestUsaSpending:
    def test_totals_awards_and_ranks_agencies(self):
        rows = [
            {"obligatedAmount": 1000, "potentialAmount": 2000,
             "awardingAgencyName": "Department of State", "actionDate": "2026-07-30"},
            {"obligatedAmount": 500, "potentialAmount": 500,
             "awardingAgencyName": "Department of State", "actionDate": "2026-01-02"},
            {"obligatedAmount": 250, "potentialAmount": 250,
             "awardingAgencyName": "NASA", "actionDate": "2026-03-03"},
        ]
        out = fh.summarize_usa_spending(rows)
        assert out["award_count"] == 3
        assert out["total_obligated"] == 1750
        assert out["top_agencies"][0] == "Department of State (2)"
        assert out["latest_date"] == "2026-07-30"

    def test_empty_stays_empty(self):
        assert fh.summarize_usa_spending([]) == {}


class TestEmptyFeedsAreUniform:
    """Every summariser must return {} for no rows, because `fetch_all` uses
    truthiness to decide whether a dataset has anything to say."""

    @pytest.mark.parametrize("fn", [
        fh.summarize_insider_transactions,
        fh.summarize_insider_sentiment,
        fh.summarize_usa_spending,
        fh.summarize_lobbying,
        fh.summarize_uspto_patents,
        fh.summarize_h1b_visa,
    ])
    def test_no_rows_gives_no_summary(self, fn):
        assert fn([]) == {}


class TestLobbying:
    def test_uses_expenses_when_present_and_income_otherwise(self):
        # A company reports `expenses`; a hired registrant reports `income`.
        rows = [
            {"year": 2025, "expenses": 900_000, "income": None},
            {"year": 2025, "expenses": None, "income": 60_000},
        ]
        out = fh.summarize_lobbying(rows)
        assert out["per_year"][2025] == 960_000
        assert out["total"] == 960_000

    def test_totals_per_year_so_a_trend_is_visible(self):
        rows = [
            {"year": 2024, "expenses": 100},
            {"year": 2025, "expenses": 300},
            {"year": 2025, "expenses": 200},
        ]
        out = fh.summarize_lobbying(rows)
        assert out["per_year"] == {2024: 100.0, 2025: 500.0}
        assert out["latest_year"] == 2025
        assert out["filing_count"] == 3

    def test_empty_stays_empty(self):
        assert fh.summarize_lobbying([]) == {}


class TestUsptoPatents:
    def test_buckets_filings_by_year(self):
        rows = [
            {"filingDate": "2024-06-13 00:00:00", "description": "A lens"},
            {"filingDate": "2024-01-02 00:00:00", "description": "A radio"},
            {"filingDate": "2023-05-05 00:00:00", "description": "A screen"},
        ]
        out = fh.summarize_uspto_patents(rows)
        assert out["per_year"] == {2023: 1, 2024: 2}
        assert out["latest_filing"] == "2024-06-13"
        assert out["recent_titles"][0] == "A lens"

    def test_collapses_whitespace_in_titles(self):
        rows = [{"filingDate": "2024-06-13", "description": "A   very\n  spaced   title"}]
        assert fh.summarize_uspto_patents(rows)["recent_titles"] == ["A very spaced title"]

    def test_tolerates_an_unparseable_filing_date(self):
        out = fh.summarize_uspto_patents([{"filingDate": "", "description": "x"}])
        assert out["filing_count"] == 1
        assert out["per_year"] == {}

    def test_empty_stays_empty(self):
        assert fh.summarize_uspto_patents([]) == {}


class TestH1bVisa:
    def _row(self, **kw):
        base = {"caseStatus": "Certified", "wageRangeFrom": 100_000,
                "jobTitle": "Engineer", "worksiteCity": "Cupertino",
                "worksiteState": "CA", "year": 2026}
        base.update(kw)
        return base

    def test_flags_the_500_row_cap_so_the_count_reads_as_a_floor(self):
        out = fh.summarize_h1b_visa([self._row() for _ in range(500)])
        assert out["capped"] is True

    def test_a_short_response_is_not_capped(self):
        assert fh.summarize_h1b_visa([self._row()])["capped"] is False

    def test_reports_median_wage_and_top_roles(self):
        rows = [
            self._row(wageRangeFrom=100_000, jobTitle="Engineer"),
            self._row(wageRangeFrom=200_000, jobTitle="Engineer"),
            self._row(wageRangeFrom=300_000, jobTitle="Designer"),
        ]
        out = fh.summarize_h1b_visa(rows)
        assert out["median_wage"] == 200_000
        assert out["top_titles"][0] == "Engineer (2)"
        assert out["top_sites"][0] == "Cupertino, CA (3)"

    def test_counts_only_certified_separately(self):
        rows = [self._row(), self._row(caseStatus="Denied")]
        out = fh.summarize_h1b_visa(rows)
        assert out["filing_count"] == 2
        assert out["certified_count"] == 1

    def test_ignores_zero_wages_in_the_median(self):
        rows = [self._row(wageRangeFrom=0), self._row(wageRangeFrom=150_000)]
        assert fh.summarize_h1b_visa(rows)["median_wage"] == 150_000

    def test_empty_stays_empty(self):
        assert fh.summarize_h1b_visa([]) == {}


class TestFetchAll:
    def test_no_token_fetches_nothing(self):
        assert fh.fetch_all("AAPL", "") == {}

    def test_one_dataset_failing_does_not_lose_the_others(self, requests_mock):
        # These are supplementary; no single feed is worth failing a briefing over.
        requests_mock.get(f"{BASE}/stock/insider-transactions",
                          json={"data": [_tx("P", 100)]})
        requests_mock.get(f"{BASE}/stock/insider-sentiment", status_code=403, json={})
        requests_mock.get(f"{BASE}/stock/usa-spending", status_code=500, json={})
        requests_mock.get(f"{BASE}/stock/lobbying", json={"data": [{"year": 2025, "expenses": 10}]})
        requests_mock.get(f"{BASE}/stock/uspto-patent", json={"data": []})
        requests_mock.get(f"{BASE}/stock/visa-application", json={"data": []})

        out = fh.fetch_all("AAPL", "tok")

        assert "insider_transactions" in out
        assert "lobbying" in out
        assert "insider_sentiment" not in out  # premium refusal, skipped
        assert "usa_spending" not in out       # server error, skipped

    def test_empty_datasets_are_omitted_rather_than_reported_as_zero(self, requests_mock):
        for path in ("insider-transactions", "insider-sentiment", "usa-spending",
                     "lobbying", "uspto-patent", "visa-application"):
            requests_mock.get(f"{BASE}/stock/{path}", json={"data": []})
        assert fh.fetch_all("AAPL", "tok") == {}

    def test_accepts_a_bare_list_payload(self, requests_mock):
        for path in ("insider-sentiment", "usa-spending", "lobbying",
                     "uspto-patent", "visa-application"):
            requests_mock.get(f"{BASE}/stock/{path}", json={"data": []})
        requests_mock.get(f"{BASE}/stock/insider-transactions", json=[_tx("P", 5)])
        out = fh.fetch_all("AAPL", "tok")
        assert out["insider_transactions"]["buy_count"] == 1


class TestAltDataBlock:
    def test_no_token_means_no_block(self):
        assert premarket._alt_data_block("AAPL", "") == ""

    def test_no_data_means_no_block(self, monkeypatch):
        # A heading with nothing under it invites the model to speculate about
        # the silence.
        monkeypatch.setattr(premarket.finnhub_rest, "fetch_all", lambda *a: {})
        assert premarket._alt_data_block("AAPL", "tok") == ""

    def test_a_fetch_failure_degrades_to_no_block(self, monkeypatch):
        def _boom(*a):
            raise RuntimeError("finnhub down")
        monkeypatch.setattr(premarket.finnhub_rest, "fetch_all", _boom)
        assert premarket._alt_data_block("AAPL", "tok") == ""

    def test_renders_each_dataset_that_has_content(self, monkeypatch):
        monkeypatch.setattr(premarket.finnhub_rest, "fetch_all", lambda *a: {
            "insider_transactions": {
                "buy_count": 0, "sell_count": 15, "buy_value": 0.0,
                "sell_value": 112_595_414.0, "net_shares": -400_637.0,
                "insiders": ["COOK TIMOTHY D"], "latest_date": "2026-08-25",
                "other_count": 27, "buy_shares": 0.0, "sell_shares": 400_637.0,
            },
            "insider_sentiment": {
                "months": 12, "latest_mspr": -100.0, "latest_period": "2026-08",
                "mean_mspr": -38.8, "positive_months": 3, "net_change": -1_128_721.0,
            },
            "usa_spending": {
                "award_count": 6, "total_obligated": 112_763.68,
                "total_potential": 112_763.68,
                "top_agencies": ["Department of State (4)"], "latest_date": "2026-07-30",
            },
            "lobbying": {
                "filing_count": 57, "per_year": {2024: 4_880_000.0, 2025: 13_140_000.0},
                "latest_year": 2025, "total": 18_020_000.0,
            },
            "uspto_patents": {
                "filing_count": 250, "per_year": {2024: 250},
                "recent_titles": ["Fluid-Filled Tunable Lens"], "latest_filing": "2024-06-13",
            },
            "h1b_visa": {
                "filing_count": 500, "capped": True, "certified_count": 500,
                "median_wage": 177_685.5, "top_titles": ["Software Development (57)"],
                "top_sites": ["Cupertino, CA (225)"], "per_year": {2026: 500},
            },
        })
        block = premarket._alt_data_block("AAPL", "tok")

        assert "NOT intraday catalysts" in block
        assert "15 sells ($112.6M)" in block
        assert "27 further filings" in block  # compensation exclusion is stated
        assert "MSPR" in block
        assert "$112,764 obligated" in block
        assert "$18.0M total" in block
        assert "lags filing by roughly two years" in block
        assert "500+ filings" in block  # the cap is surfaced as a floor

    def test_omits_datasets_that_came_back_empty(self, monkeypatch):
        monkeypatch.setattr(premarket.finnhub_rest, "fetch_all", lambda *a: {
            "lobbying": {"filing_count": 2, "per_year": {2025: 1000.0},
                         "latest_year": 2025, "total": 1000.0},
        })
        block = premarket._alt_data_block("AAPL", "tok")
        assert "Senate lobbying" in block
        assert "MSPR" not in block
        assert "H-1B" not in block


class TestMoneyFormat:
    @pytest.mark.parametrize("value, expected", [
        (112_595_414.0, "$112.6M"),
        (18_020_000.0, "$18.0M"),
        (2_400_000_000.0, "$2.4B"),
        (112_763.68, "$112,764"),
        (0.0, "$0"),
    ])
    def test_scales_to_a_readable_unit(self, value, expected):
        assert premarket._money(value) == expected
