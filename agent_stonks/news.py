"""
News analysis pipeline using Alpaca, Yahoo Finance (yfinance), WorldNews API,
and an LLM (Gemini, OpenAI, or Anthropic — see `agent_stonks.llm`).

This module is optional — only needed for LLM-based impact scoring.
Required env vars: one of GEMINI_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY,
plus WORLD_NEWS_API_KEY.

A live session's news comes from two feeds at once (`fetch_live_news`, then
`agent_stonks.stream`): Alpaca, whose news is Benzinga's wire only, and Yahoo
Finance, which carries Reuters, Barron's, WSJ, IBD and the rest. The two
overlap -- Benzinga pieces are on Yahoo too, under Yahoo's own id -- so they
are combined with `merge_news`, which drops a repeated headline.
"""
from __future__ import annotations

import html
import re
from datetime import datetime, timedelta, timezone
from typing import Literal, Optional

import pandas as pd
import requests
from pydantic import BaseModel, field_validator

from . import observability as obs
from .config import YF_NEWS_COUNT
from .datalog import log_fetch, log_fetch_failure
from .llm import DEFAULT_NEWS_MODELS, parse_structured
from .rest import fetch_news as _fetch_alpaca_news


ImpactLabel = Literal["positive", "negative", "neutral", "small", "unknown"]

_VALID_LABELS: frozenset[str] = frozenset({"positive", "negative", "neutral", "small", "unknown"})


class Impact(BaseModel):
    impact_type: Optional[Literal["positive", "negative", "neutral"]] = None
    impact_scale: Optional[Literal["small", "medium", "large"]] = None


class _SingleImpact(BaseModel):
    index: int
    impact: str  # keep permissive; normalised below

    @field_validator("impact", mode="before")
    @classmethod
    def normalise(cls, v: object) -> str:
        s = str(v).lower().strip()
        return s if s in _VALID_LABELS else "unknown"


class _BatchImpact(BaseModel):
    scores: list[_SingleImpact]


class News(BaseModel):
    title: str
    text: str
    timestamp: str
    url: str
    sentiment: Optional[float] = None
    impact: Optional[Impact] = None


class SelectedNews(BaseModel):
    title: str
    text: str


class ListOfSelectedNews(BaseModel):
    news: list[SelectedNews]


def _clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    words = re.split(r"\s+", text, flags=re.UNICODE)
    return " ".join(words[:256])


def _today() -> str:
    return datetime.today().strftime("%Y-%m-%d")


def _week_ago() -> str:
    return (datetime.today() - timedelta(days=7)).strftime("%Y-%m-%d")


def _alpaca_headers(key: str, secret: str) -> dict[str, str]:
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}


def get_latest_news(symbols: str, key: str, secret: str, limit: int = 10) -> list[News]:
    """Fetch recent Alpaca news articles for the given symbol(s)."""
    url = "https://data.alpaca.markets/v1beta1/news?sort=desc"
    response = requests.get(
        url, headers=_alpaca_headers(key, secret), params={"symbols": symbols, "limit": limit}
    )
    response.raise_for_status()
    return [
        News(
            title=item["headline"],
            text=_clean_text(item.get("summary") or item.get("content") or ""),
            timestamp=item["created_at"],
            url=item["url"],
        )
        for item in response.json().get("news", [])
    ]


def get_last_week_news(keywords: str, worldnews_api_key: str) -> list[News]:
    """Search WorldNews API for articles from the past week matching keywords."""
    try:
        import worldnewsapi

        configuration = worldnewsapi.Configuration()
        configuration.api_key["apiKey"] = worldnews_api_key
        client = worldnewsapi.NewsApi(worldnewsapi.ApiClient(configuration))
        response = client.search_news(
            text=keywords,
            source_country="us",
            language="en",
            earliest_publish_date=_week_ago(),
            latest_publish_date=_today(),
            categories="politics,business,technology,other",
            sort="publish-time",
            sort_direction="desc",
            min_sentiment=-0.9,
            max_sentiment=0.9,
            offset=0,
            number=100,
        )
    except Exception as exc:
        log_fetch_failure(
            "news (weekly search)",
            [("WorldNews API", exc)],
            symbol=keywords,
            consequence="returning no articles",
        )
        return []

    df = pd.DataFrame.from_records([i.to_dict() for i in response.news])
    df = df.drop_duplicates(subset=["title", "text"], keep="first")
    df["text"] = df["text"].fillna("")
    df["summary"] = df["summary"].fillna("")
    return [
        News(
            title=row.title,
            text=_clean_text(row.summary or row.text or " "),
            timestamp=row.publish_date,
            url=row.url,
            sentiment=row.sentiment,
        )
        for row in df.itertuples()
    ]


def fetch_news_with_fallback(
    symbol: str,
    alpaca_key: str,
    alpaca_secret: str,
    worldnews_api_key: str,
    limit: int = 15,
) -> list[dict]:
    """Fetch news from Alpaca, falling back to WorldNews API on failure (e.g. rate limit).

    Returns the same dict shape as `rest.fetch_news` (headline/summary/created_at/url/source)
    regardless of which provider served the result, so callers don't need to branch.
    """
    try:
        articles = _fetch_alpaca_news(symbol, alpaca_key, alpaca_secret, limit=limit)
        log_fetch("news", "Alpaca news API", symbol=symbol, detail=f"{len(articles)} articles")
        return articles
    except Exception as exc:
        alpaca_failure = ("Alpaca news API", exc)

    if not worldnews_api_key:
        log_fetch_failure(
            "news",
            [alpaca_failure],
            symbol=symbol,
            consequence="no WorldNews API key configured; returning no articles",
        )
        return []

    fallback = get_last_week_news(keywords=symbol, worldnews_api_key=worldnews_api_key)
    log_fetch(
        "news",
        "WorldNews API",
        symbol=symbol,
        detail=f"{len(fallback[:limit])} articles",
        failures=[alpaca_failure],
    )
    return [
        {
            "id": f"worldnews-{symbol}-{i}",
            "headline": item.title,
            "summary": item.text,
            "created_at": item.timestamp,
            "url": item.url,
            "source": "worldnewsapi",
        }
        for i, item in enumerate(fallback[:limit])
    ]


YFINANCE_FEED = "yfinance"


def fetch_yfinance_news(symbol: str, count: int = YF_NEWS_COUNT) -> list[dict]:
    """Recent Yahoo Finance articles for `symbol`, in `rest.fetch_news`'s shape
    (id/headline/summary/created_at/url/source), plus `feed: "yfinance"`.
    `source` is the publisher (Reuters, Barrons.com, ...). Raises on failure.

    Yahoo does not return its articles in time order (it pins some), so callers
    sort -- `merge_news` does."""
    import yfinance as yf

    # A fresh Ticker on every call: Ticker.get_news caches its first answer on
    # the instance, so a kept one would never see a new article.
    items = yf.Ticker(symbol).get_news(count=count, tab="news")
    articles = []
    for item in items or []:
        content = item.get("content") or {}
        title = content.get("title")
        published = content.get("pubDate") or content.get("displayTime")
        article_id = content.get("id") or item.get("id")
        if not (title and published and article_id):
            continue
        url = (
            (content.get("canonicalUrl") or {}).get("url")
            or (content.get("clickThroughUrl") or {}).get("url")
            or content.get("previewUrl")
            or ""
        )
        articles.append({
            "id": f"yf-{article_id}",
            "headline": title,
            "summary": content.get("summary") or content.get("description") or "",
            "created_at": published,
            "url": url,
            "source": (content.get("provider") or {}).get("displayName") or "Yahoo Finance",
            "feed": YFINANCE_FEED,
        })
    return articles


def headline_key(article: dict) -> str:
    """`article`'s headline as lowercase words: what two feeds' copies of one
    story share (Alpaca's headlines carry HTML entities, Yahoo's do not)."""
    text = html.unescape(str(article.get("headline") or "")).lower()
    return " ".join(re.findall(r"[a-z0-9]+", text))


_NO_TIME = datetime.min.replace(tzinfo=timezone.utc)


def published_at(article: dict) -> datetime:
    """`article`'s created_at as an aware UTC datetime; the earliest possible
    time when it is missing or unreadable, so such an article sorts last."""
    try:
        ts = datetime.fromisoformat(str(article.get("created_at")))
    except ValueError:
        return _NO_TIME
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def merge_news(existing: list[dict], fresh: list[dict]) -> tuple[list[dict], list[dict]]:
    """Combine `existing` with the articles of `fresh` it does not have yet.

    Returns (every article newest first, the ones `fresh` added). An article
    is a repeat when its id or its headline (`headline_key`) is already there:
    the same Benzinga story reaches a session through Alpaca and through Yahoo,
    under a different id each time, and the first copy to arrive is kept.

    A new list is returned rather than `existing` changed, so a caller can swap
    it onto SymbolState.news whole and a reader never sees it half-sorted.
    """
    seen_ids = {str(a.get("id")) for a in existing if a.get("id") is not None}
    seen_headlines = {headline_key(a) for a in existing} - {""}
    added = []
    for article in fresh:
        article_id = article.get("id")
        headline = headline_key(article)
        if article_id is not None and str(article_id) in seen_ids:
            continue
        if headline and headline in seen_headlines:
            continue
        added.append(article)
        if article_id is not None:
            seen_ids.add(str(article_id))
        if headline:
            seen_headlines.add(headline)
    # Sorted even when nothing was added: Alpaca's REST answer is not in
    # publication order either. The sort is stable, so ties keep their order.
    merged = sorted([*existing, *added], key=published_at, reverse=True)
    return merged, added


def fetch_live_news(
    symbol: str,
    alpaca_key: str,
    alpaca_secret: str,
    worldnews_api_key: str,
    limit: int = 15,
) -> list[dict]:
    """A live session's opening news for `symbol`, newest first: Alpaca's
    (WorldNews when Alpaca fails, see `fetch_news_with_fallback`) merged with
    Yahoo Finance's. A Yahoo failure leaves Alpaca's articles alone; the
    stream's Yahoo poll fills them in later."""
    articles = fetch_news_with_fallback(
        symbol, alpaca_key, alpaca_secret, worldnews_api_key, limit=limit
    )
    try:
        yahoo = fetch_yfinance_news(symbol)
    except Exception as exc:
        log_fetch_failure(
            "news (Yahoo Finance)",
            [("yfinance", exc)],
            symbol=symbol,
            consequence="Alpaca's articles only until the next Yahoo poll",
        )
        yahoo = []
    else:
        log_fetch("news (Yahoo Finance)", "yfinance", symbol=symbol, detail=f"{len(yahoo)} articles")
    merged, _ = merge_news(articles, yahoo)
    return merged


_IMPACT_SYSTEM = (
    "You are a financial market expert and news analyst who estimate news impact "
    "on stock market for a given symbol."
)
_IMPACT_FORMAT = (
    'Answer only in JSON format with keys '
    '"impact_type": "positive"|"negative"|"neutral", '
    '"impact_scale": "small"|"medium"|"large"'
)


def estimate_impact_news(symbol: str, news: list[News], provider: str, api_key: str) -> list[News]:
    """Score each news item for market impact using the configured LLM provider."""
    model = DEFAULT_NEWS_MODELS[provider]
    results = []
    for item in news:
        impact = parse_structured(
            provider,
            api_key,
            model,
            _IMPACT_SYSTEM,
            f"What impact has this news on {symbol} stock? "
            f"The news: {item.title} {item.text}. {_IMPACT_FORMAT}",
            Impact,
        )
        results.append(item.model_copy(update={"impact": impact}))
    return results


_SELECTION_SYSTEM = (
    "You are a financial market expert and news analyst who select the most "
    "important news for a given symbol."
)
_SELECTION_FORMAT = r'Answer only as JSON: {"news": [{"title": str, "text": str}]}'


def select_important_news(
    symbol: str, news: list[News], provider: str, api_key: str, top_n: int = 10
) -> list[News]:
    """Use LLM to pick the most market-relevant articles from a larger list."""
    model = DEFAULT_NEWS_MODELS[provider]
    combined = " ".join(f"title: {i.title} text: {i.text}" for i in news)
    selected = parse_structured(
        provider,
        api_key,
        model,
        _SELECTION_SYSTEM,
        f"News: {combined}. Choose up to {top_n} most important pieces "
        f"that impact stock market symbol {symbol}. {_SELECTION_FORMAT}",
        ListOfSelectedNews,
    )
    if selected is None:
        return []

    # Re-attach original metadata (url, timestamp) by matching title + text
    news_index = {(i.title.lower(), i.text.lower()): i for i in news}
    return [
        news_index[(s.title.lower(), s.text.lower())]
        for s in selected.news
        if (s.title.lower(), s.text.lower()) in news_index
    ]


def get_most_important_news_week(
    keyword: str, symbol: str, worldnews_api_key: str, provider: str, api_key: str
) -> list[News]:
    """Fetch a week of news by keyword, then filter to the most impactful ones."""
    last_week = get_last_week_news(keywords=keyword, worldnews_api_key=worldnews_api_key)
    return select_important_news(symbol=symbol, news=last_week, provider=provider, api_key=api_key)


_SCORE_SYSTEM = """\
You are a financial analyst. For each numbered news item, determine its impact on the given stock symbol.

REASONING RULES (apply in order):
1. If the news is DIRECTLY about the symbol → assess sentiment normally.
2. If the news is about a DIRECT COMPETITOR → invert: competitor's good news = negative for symbol; competitor's bad news = positive for symbol.
3. If the news is about the BROADER SECTOR or MACRO → assess indirect relevance to the symbol.
4. If it is unclear → use "unknown".

Impact labels:
  "positive" – likely pushes the symbol's stock UP
  "negative" – likely pushes the symbol's stock DOWN
  "neutral"  – unlikely to move the symbol significantly
  "small"    – minor effect expected (either direction)
  "unknown"  – cannot determine impact on this symbol
"""


@obs.observe(name="score-news-impacts")
def score_news_impacts(
    symbol: str, news_items: list[dict], provider: str, api_key: str
) -> dict[str, str]:
    """Score all news items in a single LLM call. Returns {news_id: impact_label}."""
    if not news_items:
        return {}

    model = DEFAULT_NEWS_MODELS[provider]
    numbered = "\n".join(
        f"{i}. Headline: {item.get('headline', '')} | "
        f"Summary: {_clean_text(item.get('summary') or item.get('content') or '')}"
        for i, item in enumerate(news_items)
    )

    scored = parse_structured(
        provider,
        api_key,
        model,
        _SCORE_SYSTEM,
        f"Symbol: {symbol}\n\n"
        f"News articles:\n{numbered}\n\n"
        f"Return a JSON object with key \"scores\" containing an array of "
        f"{{\"index\": <int>, \"impact\": <label>}} for each article.",
        _BatchImpact,
    )
    if scored is None:
        return {}
    result: dict[str, str] = {}
    for entry in scored.scores:
        if 0 <= entry.index < len(news_items):
            news_id = str(news_items[entry.index].get("id", ""))
            if news_id:
                result[news_id] = entry.impact
    return result
