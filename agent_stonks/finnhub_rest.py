"""Finnhub's alternative-data REST endpoints, summarised for a briefing prompt.

Six datasets about what a company is *doing* rather than what its stock is doing:
who inside it is buying or selling, what it is winning from the US government,
what it is spending on lobbyists, what it is patenting, and who it is hiring
from abroad.

Each fetcher returns Finnhub's raw rows; each summariser reduces them to the
handful of numbers worth putting in front of a model. The split matters --
these endpoints return hundreds of rows (H-1B is capped at 500 and hits the cap
routinely) and pasting them into a prompt would bury the price action they are
supposed to add context to.

**Every one of these is slow-moving.** Lobbying and government awards are
quarterly, patents publish on a lag measured in years, H-1B filings are
seasonal. None of them moves a stock intraday, and the briefing prompt says so
explicitly -- they are conviction modifiers behind a thesis built on price, news
and flow, not catalysts of their own. Insider transactions are the partial
exception: a cluster of open-market buys is a days-to-weeks signal.

Measured against the live API on 2026-09-12 for AAPL, which is where the default
windows come from:

    insider transactions   8 rows / 4 months
    insider sentiment      5 monthly points / 6 months
    usa spending           6 awards / 12 months
    lobbying              22 filings / 12 months
    uspto patents          0 rows over 24 months, 250 over 5 years  <- big lag
    h1-b visa            500 rows (the cap) for a single year
"""
from __future__ import annotations

import logging
import statistics
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import requests

logger = logging.getLogger(__name__)

FINNHUB_REST = "https://finnhub.io/api/v1"

# Open-market trades, where an insider chose to put money in or take it out.
# Everything else on a Form 4 is compensation mechanics -- a grant vesting (A),
# an option exercise (M), shares withheld for tax (F), a gift (G) -- and treating
# those as sentiment is the classic way to read an insider feed wrong: a big
# scheduled vest shows up as a huge "sale" that the insider never decided to make.
OPEN_MARKET_CODES = {"P": "purchase", "S": "sale"}

# Session cache. These datasets update quarterly at best, and a briefing is
# regenerated on demand -- re-fetching six endpoints per symbol on every
# regenerate spends rate limit on data that cannot have changed.
_CACHE_TTL_SEC = 30 * 60
_cache: dict[tuple, tuple[float, object]] = {}
_cache_lock = threading.Lock()


class FinnhubError(RuntimeError):
    """A Finnhub REST call that failed, with the API's own message where it gave one."""


def _get(path: str, token: str, **params) -> dict:
    if not token:
        raise FinnhubError("no FINNHUB_API_KEY configured")
    response = requests.get(
        f"{FINNHUB_REST}{path}", params={**params, "token": token}, timeout=20
    )
    if response.status_code == 401:
        raise FinnhubError("Finnhub rejected the API key")
    if response.status_code == 403:
        # Several of these are premium on some plans; say so rather than
        # reporting an empty dataset, which would read as "nothing happening".
        raise FinnhubError(f"{path} is not available on this Finnhub plan")
    if response.status_code == 429:
        raise FinnhubError("Finnhub rate limit reached")
    if response.status_code >= 400:
        raise FinnhubError(f"{path} failed ({response.status_code})")
    payload = response.json()
    return payload if isinstance(payload, dict) else {"data": payload}


def _window(days: int) -> tuple[str, str]:
    today = datetime.now(timezone.utc).date()
    return (today - timedelta(days=days)).isoformat(), today.isoformat()


def _rows(path: str, symbol: str, token: str, days: int) -> list[dict]:
    """Fetch one dataset's rows, memoised per (endpoint, symbol, window)."""
    frm, to = _window(days)
    key = (path, symbol.upper(), frm, to)
    now = time.monotonic()
    with _cache_lock:
        hit = _cache.get(key)
        if hit and now - hit[0] < _CACHE_TTL_SEC:
            return hit[1]
    data = _get(path, token, symbol=symbol.upper(), **{"from": frm, "to": to}).get("data") or []
    with _cache_lock:
        _cache[key] = (now, data)
    return data


def clear_cache() -> None:
    with _cache_lock:
        _cache.clear()


# --- fetchers ------------------------------------------------------------
# Default windows are set from what each endpoint actually returns (see the
# module docstring), not from what would be tidy.

def fetch_insider_transactions(symbol: str, token: str, days: int = 180) -> list[dict]:
    return _rows("/stock/insider-transactions", symbol, token, days)


def fetch_insider_sentiment(symbol: str, token: str, days: int = 365) -> list[dict]:
    return _rows("/stock/insider-sentiment", symbol, token, days)


def fetch_usa_spending(symbol: str, token: str, days: int = 365) -> list[dict]:
    return _rows("/stock/usa-spending", symbol, token, days)


def fetch_lobbying(symbol: str, token: str, days: int = 730) -> list[dict]:
    return _rows("/stock/lobbying", symbol, token, days)


def fetch_uspto_patents(symbol: str, token: str, days: int = 1825) -> list[dict]:
    """Five years by default: USPTO publication lags filing by roughly two, so a
    12-month window comes back empty even for a company filing constantly."""
    return _rows("/stock/uspto-patent", symbol, token, days)


def fetch_h1b_visa(symbol: str, token: str, days: int = 730) -> list[dict]:
    return _rows("/stock/visa-application", symbol, token, days)


# --- summarisers ---------------------------------------------------------
# Pure functions over the raw rows, so the arithmetic that decides what a
# dataset "says" is testable without a network.

def _num(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def summarize_insider_transactions(rows: list[dict]) -> dict:
    """Open-market insider buying vs selling, with compensation noise removed.

    Reports the two populations separately: `open_market` is the signal (someone
    chose to trade), `other` is grants, vests, option exercises and tax
    withholding, which are scheduled and say nothing about conviction.
    """
    if not rows:
        # Every other summariser here returns {} for no data, and `fetch_all`
        # relies on that to omit a dataset rather than reporting it as zeros --
        # "0 buys vs 0 sells" reads as a finding when it is just an empty feed.
        return {}
    open_market = [r for r in rows if str(r.get("transactionCode", "")).upper() in OPEN_MARKET_CODES]
    other = [r for r in rows if r not in open_market]
    buys = [r for r in open_market if str(r.get("transactionCode", "")).upper() == "P"]
    sells = [r for r in open_market if str(r.get("transactionCode", "")).upper() == "S"]

    def value(batch: list[dict]) -> float:
        return sum(abs(_num(r.get("change"))) * _num(r.get("transactionPrice")) for r in batch)

    names = [str(r.get("name") or "").strip() for r in open_market if r.get("name")]
    return {
        "buy_count": len(buys),
        "sell_count": len(sells),
        "buy_shares": sum(abs(_num(r.get("change"))) for r in buys),
        "sell_shares": sum(abs(_num(r.get("change"))) for r in sells),
        "buy_value": value(buys),
        "sell_value": value(sells),
        "net_shares": sum(_num(r.get("change")) for r in open_market),
        "insiders": [n for n, _ in Counter(names).most_common(4)],
        "latest_date": max((str(r.get("transactionDate") or "") for r in open_market), default=""),
        "other_count": len(other),
    }


def summarize_insider_sentiment(rows: list[dict]) -> dict:
    """Finnhub's MSPR (monthly share purchase ratio, -100..+100) over time.

    The level of the latest month matters less than whether the recent months
    lean one way, so both the newest reading and the mean are reported.
    """
    points = sorted(
        (r for r in rows if r.get("mspr") is not None),
        key=lambda r: (_num(r.get("year")), _num(r.get("month"))),
    )
    if not points:
        return {}
    mspr = [_num(r.get("mspr")) for r in points]
    latest = points[-1]
    return {
        "months": len(points),
        "latest_mspr": mspr[-1],
        "latest_period": f"{int(_num(latest.get('year')))}-{int(_num(latest.get('month'))):02d}",
        "mean_mspr": statistics.fmean(mspr),
        "positive_months": sum(1 for v in mspr if v > 0),
        "net_change": sum(_num(r.get("change")) for r in points),
    }


def summarize_usa_spending(rows: list[dict]) -> dict:
    """Federal contract awards: how much, from whom, how recently."""
    if not rows:
        return {}
    agencies = Counter(
        str(r.get("awardingAgencyName") or "unknown").strip() for r in rows
    )
    return {
        "award_count": len(rows),
        "total_obligated": sum(_num(r.get("obligatedAmount")) for r in rows),
        "total_potential": sum(_num(r.get("potentialAmount")) for r in rows),
        "top_agencies": [f"{a} ({n})" for a, n in agencies.most_common(3)],
        "latest_date": max((str(r.get("actionDate") or "") for r in rows), default=""),
    }


def summarize_lobbying(rows: list[dict]) -> dict:
    """Senate lobbying disclosures, totalled per year so a trend is visible.

    A filing carries `expenses` when the company reports its own spend and
    `income` when a hired registrant reports what it was paid; whichever is
    present is the money that moved.
    """
    if not rows:
        return {}
    per_year: dict[int, float] = {}
    for row in rows:
        year = int(_num(row.get("year")))
        amount = _num(row.get("expenses")) or _num(row.get("income"))
        per_year[year] = per_year.get(year, 0.0) + amount
    years = sorted(per_year)
    return {
        "filing_count": len(rows),
        "per_year": {y: per_year[y] for y in years},
        "latest_year": years[-1] if years else None,
        "total": sum(per_year.values()),
    }


def summarize_uspto_patents(rows: list[dict]) -> dict:
    """Patent filings by year, plus a few recent titles as a sense of direction."""
    if not rows:
        return {}
    per_year = Counter()
    for row in rows:
        stamp = str(row.get("filingDate") or "")[:4]
        if stamp.isdigit():
            per_year[int(stamp)] += 1
    recent = sorted(rows, key=lambda r: str(r.get("filingDate") or ""), reverse=True)
    titles = [
        " ".join(str(r.get("description") or "").split())[:90]
        for r in recent[:3]
        if r.get("description")
    ]
    return {
        "filing_count": len(rows),
        "per_year": dict(sorted(per_year.items())),
        "recent_titles": titles,
        "latest_filing": str(recent[0].get("filingDate") or "")[:10] if recent else "",
    }


def summarize_h1b_visa(rows: list[dict]) -> dict:
    """Hiring intent from H-1B filings: how many, for what, at what wage.

    Finnhub caps this response at 500 rows and a large employer hits the cap, so
    the count is reported as a floor rather than a total.
    """
    if not rows:
        return {}
    certified = [r for r in rows if str(r.get("caseStatus", "")).lower().startswith("certified")]
    wages = [_num(r.get("wageRangeFrom")) for r in rows if _num(r.get("wageRangeFrom")) > 0]
    titles = Counter(str(r.get("jobTitle") or "").strip() for r in rows if r.get("jobTitle"))
    sites = Counter(
        f"{r.get('worksiteCity')}, {r.get('worksiteState')}"
        for r in rows
        if r.get("worksiteCity")
    )
    years = Counter(int(_num(r.get("year"))) for r in rows if r.get("year"))
    return {
        "filing_count": len(rows),
        "capped": len(rows) >= 500,
        "certified_count": len(certified),
        "median_wage": statistics.median(wages) if wages else 0.0,
        "top_titles": [f"{t} ({n})" for t, n in titles.most_common(3)],
        "top_sites": [f"{s} ({n})" for s, n in sites.most_common(2)],
        "per_year": dict(sorted(years.items())),
    }


# --- one call for all six ------------------------------------------------

_DATASETS: tuple[tuple[str, object, object], ...] = (
    ("insider_transactions", fetch_insider_transactions, summarize_insider_transactions),
    ("insider_sentiment", fetch_insider_sentiment, summarize_insider_sentiment),
    ("usa_spending", fetch_usa_spending, summarize_usa_spending),
    ("lobbying", fetch_lobbying, summarize_lobbying),
    ("uspto_patents", fetch_uspto_patents, summarize_uspto_patents),
    ("h1b_visa", fetch_h1b_visa, summarize_h1b_visa),
)


def fetch_all(symbol: str, token: str) -> dict[str, dict]:
    """Every dataset for one symbol, fetched concurrently and summarised.

    Returns `{name: summary}`, omitting any dataset that failed or came back
    empty. Six sequential round trips would add ten-odd seconds to a briefing
    that already makes a dozen calls, so they go out together; one failing is
    recorded and skipped rather than losing the other five, because these are
    supplementary context and no single one is worth failing a briefing over.
    """
    if not token:
        return {}

    def one(entry) -> tuple[str, dict]:
        name, fetch, summarize = entry
        try:
            return name, summarize(fetch(symbol, token))
        except FinnhubError as exc:
            logger.info("%s: %s unavailable — %s", symbol, name, exc)
        except Exception as exc:  # noqa: BLE001 - supplementary data, never fatal
            logger.warning("%s: %s failed — %s", symbol, name, exc)
        return name, {}

    with ThreadPoolExecutor(max_workers=len(_DATASETS)) as pool:
        results = list(pool.map(one, _DATASETS))
    return {name: summary for name, summary in results if summary}
