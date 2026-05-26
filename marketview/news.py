"""
News analysis pipeline using Alpaca, WorldNews API, and an LLM (Gemini via OpenAI-compat).

This module is optional — only needed for LLM-based impact scoring.
Required env vars: GEMINI_API_KEY, WORLD_NEWS_API_KEY
"""
from __future__ import annotations

import os
import re
from datetime import datetime, timedelta
from typing import Literal

import pandas as pd
import requests
from openai import OpenAI
from pydantic import BaseModel


class Impact(BaseModel):
    impact_type: Literal["positive", "negative", "neutral"] | None
    impact_scale: Literal["small", "medium", "large"] | None


class News(BaseModel):
    title: str
    text: str
    timestamp: str
    url: str
    sentiment: float | None = None
    impact: Impact | None = None


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
        print(f"WorldNews API error: {exc}")
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


_IMPACT_SYSTEM = (
    "You are a financial market expert and news analyst who estimate news impact "
    "on stock market for a given symbol."
)
_IMPACT_FORMAT = (
    'Answer only in JSON format with keys '
    '"impact_type": "positive"|"negative"|"neutral", '
    '"impact_scale": "small"|"medium"|"large"'
)


def estimate_impact_news(symbol: str, news: list[News]) -> list[News]:
    """Score each news item for market impact using Gemini via OpenAI-compat API."""
    client = OpenAI(
        api_key=os.environ["GEMINI_API_KEY"],
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    results = []
    for item in news:
        completion = client.chat.completions.parse(
            model="gemini-2.0-flash",
            messages=[
                {"role": "system", "content": _IMPACT_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"What impact has this news on {symbol} stock? "
                        f"The news: {item.title} {item.text}. {_IMPACT_FORMAT}"
                    ),
                },
            ],
            response_format=Impact,
        )
        impact = completion.choices[0].message.parsed
        results.append(item.model_copy(update={"impact": impact}))
    return results


_SELECTION_SYSTEM = (
    "You are a financial market expert and news analyst who select the most "
    "important news for a given symbol."
)
_SELECTION_FORMAT = r'Answer only as JSON: {"news": [{"title": str, "text": str}]}'


def select_important_news(symbol: str, news: list[News], top_n: int = 10) -> list[News]:
    """Use LLM to pick the most market-relevant articles from a larger list."""
    client = OpenAI(
        api_key=os.environ["GEMINI_API_KEY"],
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    combined = " ".join(f"title: {i.title} text: {i.text}" for i in news)
    completion = client.chat.completions.parse(
        model="gemini-2.0-flash",
        messages=[
            {"role": "system", "content": _SELECTION_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"News: {combined}. Choose up to {top_n} most important pieces "
                    f"that impact stock market symbol {symbol}. {_SELECTION_FORMAT}"
                ),
            },
        ],
        response_format=ListOfSelectedNews,
    )
    selected = completion.choices[0].message.parsed

    # Re-attach original metadata (url, timestamp) by matching title + text
    news_index = {(i.title.lower(), i.text.lower()): i for i in news}
    return [
        news_index[(s.title.lower(), s.text.lower())]
        for s in selected.news
        if (s.title.lower(), s.text.lower()) in news_index
    ]


def get_most_important_news_week(keyword: str, symbol: str, worldnews_api_key: str) -> list[News]:
    """Fetch a week of news by keyword, then filter to the most impactful ones."""
    last_week = get_last_week_news(keywords=keyword, worldnews_api_key=worldnews_api_key)
    return select_important_news(symbol=symbol, news=last_week)
