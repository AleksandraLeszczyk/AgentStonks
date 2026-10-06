"""What each replayed session looked like before it started: the VIX at the
open, and a pre-market briefing's bullish / neutral / bearish call.

Read by the Tuning tab's "The pick, session by session" chart, so a day the
pick lost money can be set against what the morning looked like. The cached
briefings are also the one input a replay takes from here: Orchestra's 09:34
candidate selection reads them (`ReplaySources`), together with each symbol's
earnings calendar, kept here too. A replay only ever reads a cached briefing
and never writes one, so the same morning makes the same choice every time.

That is why both live under `data/simlab/session_context/` and not in the
rolling store: `tuning.run_is_stale` compares the store's mtimes against every
stored run, so writing a VIX open into `market/indicators.json.gz` would mark
every run in the store stale and have the next tuning job replay all of them.

The VIX open is Yahoo's daily Open for ^VIX -- the 9:30 print, not the
previous close. Fetched for the days asked about and kept, so a session is
downloaded once.

The briefing is the live app's pre-market briefing
(`premarket.generate_premarket_from_data`, same system prompt) written as of
09:25 ET on the session's day, from data clipped to that moment: stored daily
closes and SPY/VIX/VIX3M closes of the days before, news published before
09:25, and the earnings schedule. It is an LLM call, so it is not
reproducible; the first answer per (symbol, day, provider, model) is cached
and that is the one shown from then on.
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import requests

from agent_stonks import premarket
from agent_stonks.config import DATA_REST
from agent_stonks.llm import ENV_KEYS, PROVIDERS
from agent_stonks.market_hours import MARKET_TZ

from . import data as sim_data

CONTEXT_DIR = sim_data.SIMLAB_DIR / "session_context"
VIX_OPEN_PATH = CONTEXT_DIR / "vix_open.json"
BRIEFING_DIR = CONTEXT_DIR / "briefings"

# Five minutes before the bell: what a trader reading the briefing at the open
# would have had, and before the auction print it would otherwise be told.
BRIEFING_TIME = time(9, 25)
# How far back the briefing's news reaches, and how many articles it gets --
# the live briefing asks Alpaca for its latest 20.
NEWS_LOOKBACK_DAYS = 7
NEWS_LIMIT = 20

BIASES = ("bullish", "neutral", "bearish")


# ---------------------------------------------------------------------------
# VIX at the open
# ---------------------------------------------------------------------------

def _fetch_vix_opens(start: date, end: date) -> dict[str, float]:
    """{YYYY-MM-DD: ^VIX daily open} over [start, end] from yfinance."""
    import yfinance as yf

    frame = yf.Ticker("^VIX").history(
        start=start.isoformat(), end=(end + timedelta(days=1)).isoformat(),
        interval="1d", auto_adjust=False,
    )
    if frame.empty:
        return {}
    return {idx.date().isoformat(): float(v) for idx, v in frame["Open"].dropna().items()}


def _load_vix_opens() -> dict[str, float]:
    try:
        return json.loads(VIX_OPEN_PATH.read_text())
    except (OSError, ValueError):
        return {}


def vix_opens(
    days: list[str], fetch: Callable[[date, date], dict[str, float]] = _fetch_vix_opens
) -> dict[str, float]:
    """The VIX's opening print for each of `days` that has one.

    Only the days not already kept are downloaded, in one request spanning
    them. A failed download leaves those days out rather than raising -- the
    chart draws what it has.
    """
    kept = _load_vix_opens()
    missing = sorted(d for d in days if d not in kept)
    if missing:
        try:
            fetched = fetch(date.fromisoformat(missing[0]), date.fromisoformat(missing[-1]))
        except Exception:
            fetched = {}
        if fetched:
            kept.update(fetched)
            CONTEXT_DIR.mkdir(parents=True, exist_ok=True)
            VIX_OPEN_PATH.write_text(json.dumps(kept, sort_keys=True))
    return {d: kept[d] for d in days if d in kept}


# ---------------------------------------------------------------------------
# Pre-market bias
# ---------------------------------------------------------------------------

def briefing_provider() -> Optional[tuple[str, str, str]]:
    """(provider, model, api key) for the briefings: the pre-market default
    provider when its key is in the environment, else the first provider with
    one, on its pre-market default model. None without any."""
    default = premarket.DEFAULT_PREMARKET_PROVIDER
    for provider in (default, *(p for p in PROVIDERS if p != default)):
        key = os.getenv(ENV_KEYS[provider], "")
        if key:
            return provider, premarket.DEFAULT_PREMARKET_MODELS[provider], key
    return None


def briefing_as_of(day: str) -> datetime:
    return datetime.combine(date.fromisoformat(day), BRIEFING_TIME, tzinfo=MARKET_TZ)


def briefing_path(symbol: str, day: str, provider: str, model: str) -> Path:
    return BRIEFING_DIR / symbol.upper() / f"{day}.{provider}.{model}.json"


def cached_briefing(symbol: str, day: str, provider: str, model: str) -> Optional[dict]:
    try:
        return json.loads(briefing_path(symbol, day, provider, model).read_text())
    except (OSError, ValueError):
        return None


def _closes_before(symbol: str, feed: str, day: str) -> pd.Series:
    """Stored daily closes of the sessions before `day`, indexed by ET date.

    Daily bars are stamped at midnight ET in UTC, so the session is the bar's
    ET date (see `app._tuning_session_bars`).
    """
    rows = {}
    for bar in sim_data.load_daily_bars(symbol, feed):
        bar_day = pd.Timestamp(bar["t"]).tz_convert(MARKET_TZ).date().isoformat()
        if bar_day < day:
            rows[pd.Timestamp(bar_day)] = float(bar["c"])
    return pd.Series(rows, dtype=float).sort_index()


def _indicators_before(day: str) -> dict[str, pd.Series]:
    """SPY/VIX/VIX3M closes of the days before `day` -- the morning's macro."""
    out = {}
    for name, rows in sim_data.load_market_indicators().items():
        kept = [(r["date"], float(r["close"])) for r in rows if r["date"] < day]
        out[name] = pd.Series(
            [c for _, c in kept], index=pd.to_datetime([d for d, _ in kept]), dtype=float
        )
    return out


def _fetch_news_before(
    symbol: str, as_of: datetime, key: str, secret: str
) -> list[dict]:
    """Alpaca's newest articles on `symbol` in the week before `as_of`."""
    r = requests.get(
        f"{DATA_REST}/v1beta1/news",
        headers=sim_data._headers(key, secret),
        params=dict(
            symbols=symbol,
            start=(as_of - timedelta(days=NEWS_LOOKBACK_DAYS)).astimezone(timezone.utc).isoformat(),
            end=as_of.astimezone(timezone.utc).isoformat(),
            limit=NEWS_LIMIT, sort="desc",
        ),
        timeout=30,
    )
    r.raise_for_status()
    return r.json().get("news", [])


def _stored_news_before(symbol: str, as_of: datetime) -> list[dict]:
    """The same week out of the news store -- which only holds a dataset's own
    days, so thinner than Alpaca's. The fallback without Alpaca keys."""
    cutoff = as_of.astimezone(timezone.utc)
    items = []
    for back in range(NEWS_LOOKBACK_DAYS + 1):
        for item in sim_data.load_news(symbol, as_of.date() - timedelta(days=back)):
            created = pd.to_datetime(item.get("created_at"), utc=True, errors="coerce")
            if not pd.isna(created) and created.to_pydatetime() < cutoff:
                items.append(item)
    items.sort(key=lambda item: item.get("created_at") or "", reverse=True)
    return items[:NEWS_LIMIT]


def generate_briefing(
    symbol: str, day: str, feed: str, provider: str, model: str, api_key: str,
    alpaca_key: str = "", alpaca_secret: str = "",
) -> dict:
    """Brief one session as of 09:25 ET that day, cache it, and return it.

    Raises when the model gives nothing back, so the caller can say which day
    failed; a failed day is not cached and is asked again next time.
    """
    sym = symbol.upper()
    as_of = briefing_as_of(day)
    if alpaca_key and alpaca_secret:
        news = _fetch_news_before(sym, as_of, alpaca_key, alpaca_secret)
        news_source = "alpaca"
    else:
        news = _stored_news_before(sym, as_of)
        news_source = "store"
    briefing = premarket.generate_premarket_from_data(
        sym, provider, api_key, as_of=as_of,
        closes=_closes_before(sym, feed, day),
        indicators=_indicators_before(day),
        news_items=news,
        earnings_text=premarket._earnings_block(sym, as_of),
        model=model,
    )
    if briefing is None:
        raise RuntimeError("the model returned no briefing")
    record = {
        "symbol": sym, "day": day, "as_of": as_of.isoformat(),
        "provider": provider, "model": model,
        "bias": briefing.overall_bias, "confidence": briefing.confidence,
        "summary": briefing.summary,
        "news": len(news), "news_source": news_source,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    path = briefing_path(sym, day, provider, model)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1))
    return record


def session_biases(
    symbol: str, days_by_feed: dict[str, list[str]], provider: str, model: str,
    api_key: str, alpaca_key: str = "", alpaca_secret: str = "",
    generate: Callable[..., dict] = generate_briefing, workers: int = 4,
) -> tuple[dict[str, dict], dict[str, str]]:
    """({day: briefing record}, {day: error}) for every day, cached ones read
    and the rest generated in parallel.

    `days_by_feed` because a tuning job's two datasets can be on different
    tapes, and the daily closes a briefing reads come from the day's own feed.
    """
    records: dict[str, dict] = {}
    todo: list[tuple[str, str]] = []
    for feed, days in days_by_feed.items():
        for day in days:
            cached = cached_briefing(symbol, day, provider, model)
            if cached is not None:
                records[day] = cached
            else:
                todo.append((feed, day))
    errors: dict[str, str] = {}
    if todo:
        def one(item):
            feed, day = item
            try:
                return day, generate(
                    symbol, day, feed, provider, model, api_key, alpaca_key, alpaca_secret
                ), None
            except Exception as exc:
                return day, None, str(exc) or type(exc).__name__

        with ThreadPoolExecutor(max_workers=max(1, min(workers, len(todo)))) as pool:
            for day, record, error in pool.map(one, todo):
                if record is not None:
                    records[day] = record
                else:
                    errors[day] = error
    return records, errors


# ---------------------------------------------------------------------------
# Orchestra's 09:34 candidate selection, replayed
# ---------------------------------------------------------------------------

EARNINGS_DIR = CONTEXT_DIR / "earnings"
# The briefings an Orchestra setup that names none is shown coverage for: the
# pre-market default provider on its default model.
DEFAULT_BRIEFING = (
    premarket.DEFAULT_PREMARKET_PROVIDER,
    premarket.DEFAULT_PREMARKET_MODELS[premarket.DEFAULT_PREMARKET_PROVIDER],
)


def cached_briefing_models() -> list[tuple[str, str]]:
    """Every (provider, model) with briefings in the cache, the most-cached
    first: what a replayed selection can read."""
    counts: dict[tuple[str, str], int] = {}
    for path in BRIEFING_DIR.glob("*/*.json"):
        parts = path.name[: -len(".json")].split(".", 2)
        if len(parts) == 3:
            key = (parts[1], parts[2])
            counts[key] = counts.get(key, 0) + 1
    return sorted(counts, key=lambda key: (-counts[key], key))


def briefing_coverage(
    symbols: list[str], days: list[str], provider: str, model: str
) -> tuple[int, list[tuple[str, str]]]:
    """(cached, [(symbol, day) missing]) over every symbol and day."""
    missing = [
        (symbol, day) for symbol in symbols for day in days
        if not briefing_path(symbol, day, provider, model).exists()
    ]
    return len(symbols) * len(days) - len(missing), missing


def earnings_dates(symbol: str, fetch: Optional[Callable[[str], list]] = None) -> Optional[list[str]]:
    """`symbol`'s earnings report times (ISO), from the cache, fetched from
    yfinance and kept the first time they are asked for. None when they cannot
    be read.

    Kept rather than re-fetched so a replay reads the same calendar every
    time. Past reports stay on Yahoo's list and future ones are scheduled
    weeks ahead, so the date after a replayed morning was known that morning.
    """
    path = EARNINGS_DIR / f"{symbol.upper()}.json"
    try:
        return json.loads(path.read_text())["dates"]
    except (OSError, ValueError, KeyError):
        pass
    if fetch is None:
        from agent_stonks import historical

        def fetch(sym: str) -> list:
            frame = historical.fetch_earnings_dates(sym, days=2000)
            return [] if frame is None else [pd.Timestamp(t).isoformat() for t in frame.index]
    try:
        dates = fetch(symbol.upper())
    except Exception:
        return None
    if not dates:
        # Nothing on record is not the same as a failed read; neither is kept,
        # so the next replay asks again.
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "symbol": symbol.upper(), "dates": list(dates),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }, indent=1))
    return list(dates)


class ReplaySources:
    """The briefing and the earnings calendar for a replayed 09:34 selection
    (`agent_stonks.candidates`): the briefing cached for the day by `provider`
    and `model`, never a new one -- an LLM answer is not reproducible, and a
    replay must make the same choice every time it is run."""

    def __init__(self, provider: str = "", model: str = "") -> None:
        self.provider = provider
        self.model = model

    def briefing(self, ticker: str, day: date, state) -> tuple[Optional[dict], str]:
        if not (self.provider and self.model):
            return None, "no briefing source chosen"
        record = cached_briefing(ticker, day.isoformat(), self.provider, self.model)
        if record is None:
            return None, f"no cached {self.provider}/{self.model} briefing"
        return {"bias": record.get("bias"), "confidence": record.get("confidence")}, ""

    def earnings(self, ticker: str, day: date) -> Optional[list]:
        return earnings_dates(ticker)
