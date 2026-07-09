"""
Pre-market analysis: synthesizes recent news, historical price context, macro
indicators, and fundamental data into a structured morning briefing via LLM.

Optional: requires one of GEMINI_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY.
News sources are the same as the live news panel (Alpaca + WorldNews fallback).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Literal, Optional

from pydantic import BaseModel

from . import observability as obs
from .historical import (
    fetch_close_series,
    fetch_earnings_dates,
    fetch_market_indicators,
    fetch_static_analysis,
)
from .llm import DEFAULT_NEWS_MODELS, parse_structured
from .news import fetch_news_with_fallback, get_last_week_news


DEFAULT_PREMARKET_MODELS: dict[str, str] = {
    "gemini": "gemini-3.5-flash",
    "openai": "gpt-5.4-mini",
    "anthropic": "claude-sonnet-4-6",
}


class Catalyst(BaseModel):
    headline: str
    impact: Literal["positive", "negative", "neutral"]
    relevance: str


class TechnicalLevel(BaseModel):
    level: float
    role: str
    note: str


class PremarketBriefing(BaseModel):
    overall_bias: Literal["bullish", "bearish", "neutral"]
    confidence: Literal["high", "medium", "low"]
    summary: str
    catalysts: list[Catalyst]
    technical_levels: list[TechnicalLevel]
    risk_factors: list[str]
    macro_context: str
    key_levels_to_watch: list[str]


_PREMARKET_SYSTEM = """\
You are a senior equity analyst preparing a pre-market briefing for a day trader.
You have recent news, historical price action, and macro market indicators.

Your job:
1. Form a directional morning bias (bullish / bearish / neutral) with a confidence rating.
2. Write a 2-3 sentence executive summary (be specific, cite concrete data).
3. List 2-5 catalysts that drive the bias — only keep news that is DIRECTLY relevant.
4. Call out 2-4 key price levels the trader must monitor today (support, resistance, pivots).
5. List 2-4 tail risks that could invalidate the thesis.
6. Provide 1-2 sentences of macro context (SPY trend + VIX regime).
7. List 2-3 things to watch during the session as actionable cues.

Be concise. If data is thin, lower confidence to "low" and say so in the summary.
"""


def _fmt_price(value: float) -> str:
    """Format a dollar price for the LLM prompt: whole dollars above $100, cents below.
    Keeps the model reasoning in round numbers for higher-priced names while the UI
    (which formats independently) continues to show exact prices."""
    return f"{value:.0f}" if value > 100 else f"{value:.2f}"


def _price_context(symbol: str) -> tuple[str, dict[str, float]]:
    """Fetch daily closes for multiple lookback windows. Returns (text, last_closes_by_period)."""
    lines: list[str] = []
    last_close: dict[str, float] = {}
    for label, days in [("7d", 7), ("30d", 30), ("90d", 90), ("1y", 365)]:
        try:
            series = fetch_close_series(symbol, days)
            if series.empty or len(series) < 2:
                continue
            start, end = float(series.iloc[0]), float(series.iloc[-1])
            pct = (end - start) / start * 100
            hi, lo = float(series.max()), float(series.min())
            lines.append(
                f"  {label}: {pct:+.1f}%  (range {_fmt_price(lo)}–{_fmt_price(hi)}, "
                f"last close {_fmt_price(end)})"
            )
            last_close[label] = end
        except Exception:
            pass
    header = f"Price history — {symbol}:"
    return (header + "\n" + "\n".join(lines)) if lines else f"No price history for {symbol}.", last_close


def _macro_context(days: int = 30) -> str:
    try:
        mkt = fetch_market_indicators(days=days)
    except Exception:
        return "Macro data unavailable."

    parts: list[str] = []
    spy = mkt.get("spy")
    vix = mkt.get("vix")
    vix3m = mkt.get("vix3m")

    if spy is not None and not spy.empty and len(spy) >= 5:
        w1 = (spy.iloc[-1] - spy.iloc[-5]) / spy.iloc[-5] * 100
        m1 = (spy.iloc[-1] - spy.iloc[0]) / spy.iloc[0] * 100 if len(spy) >= 20 else None
        spy_str = f"SPY 5d {w1:+.1f}%"
        if m1 is not None:
            spy_str += f" / 1mo {m1:+.1f}%"
        spy_str += f" (close {_fmt_price(float(spy.iloc[-1]))})"
        parts.append(spy_str)

    if vix is not None and not vix.empty:
        v = float(vix.iloc[-1])
        regime = (
            "fear/spike" if v > 30
            else "elevated" if v > 20
            else "moderate" if v > 15
            else "complacency"
        )
        vix_part = f"VIX {v:.1f} ({regime})"
        if vix3m is not None and not vix3m.empty:
            v3 = float(vix3m.iloc[-1])
            term_structure = "contango" if v3 > v else "backwardation"
            vix_part += f", VIX3M {v3:.1f} ({term_structure})"
        parts.append(vix_part)

    return "Macro: " + " | ".join(parts) if parts else "Macro data unavailable."


def _news_block(news_items: list[dict], max_items: int = 20) -> str:
    if not news_items:
        return "No recent news."
    lines = []
    for i, item in enumerate(news_items[:max_items]):
        date = (item.get("created_at") or "")[:10]
        headline = item.get("headline", "").strip()
        summary = (item.get("summary") or "")[:160].rstrip()
        lines.append(f"{i + 1}. [{date}] {headline}\n   {summary}")
    return "Recent news:\n" + "\n".join(lines)


def _fundamentals_block(symbol: str) -> str:
    try:
        info = fetch_static_analysis(symbol)
    except Exception:
        return ""
    parts: list[str] = []
    if info.get("pe_ratio") is not None:
        parts.append(f"Trailing P/E {info['pe_ratio']:.1f}")
    if info.get("forward_pe") is not None:
        parts.append(f"Forward P/E {info['forward_pe']:.1f}")
    if info.get("dividend_yield") is not None:
        parts.append(f"Div yield {info['dividend_yield'] * 100:.2f}%")
    if info.get("growth_rate") is not None:
        parts.append(f"Growth rate {info['growth_rate'] * 100:.1f}%")
    return "Fundamentals: " + ", ".join(parts) if parts else ""


def _earnings_block(symbol: str) -> str:
    try:
        df = fetch_earnings_dates(symbol, days=60)
    except Exception:
        return ""
    if df.empty:
        return ""
    now = datetime.now(tz=df.index.tz)
    upcoming = df[df.index >= now]
    if upcoming.empty:
        return ""
    next_date = upcoming.index[0]
    days_away = (next_date - now).days
    return f"Next earnings: {next_date.strftime('%Y-%m-%d')} (~{days_away}d away)"


@obs.observe(name="generate-premarket-analysis")
def generate_premarket_analysis(
    symbol: str,
    provider: str,
    api_key: str,
    alpaca_key: str = "",
    alpaca_secret: str = "",
    worldnews_key: str = "",
    model: Optional[str] = None,
) -> Optional[PremarketBriefing]:
    """Generate a structured pre-market briefing by gathering multi-source context and calling the LLM."""
    sym = symbol.strip().upper()

    # --- News (Alpaca primary, WorldNews 30-day supplement) ---
    news_items: list[dict] = []
    if alpaca_key and alpaca_secret:
        try:
            news_items = fetch_news_with_fallback(sym, alpaca_key, alpaca_secret, worldnews_key, limit=20)
        except Exception:
            pass

    # If Alpaca gave nothing, try 30 days of WorldNews directly for broader context
    if not news_items and worldnews_key:
        try:
            month_ago = (datetime.today() - timedelta(days=30)).strftime("%Y-%m-%d")
            week_news = get_last_week_news(keywords=sym, worldnews_api_key=worldnews_key)
            news_items = [
                {
                    "headline": n.title,
                    "summary": n.text,
                    "created_at": n.timestamp,
                    "source": "worldnewsapi",
                }
                for n in week_news[:20]
            ]
        except Exception:
            pass

    price_text, _ = _price_context(sym)
    macro_text = _macro_context(days=60)
    news_text = _news_block(news_items)
    fundamentals_text = _fundamentals_block(sym)
    earnings_text = _earnings_block(sym)

    context_parts = [
        f"Symbol: {sym}",
        f"Analysis date: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
    ]
    for block in [earnings_text, fundamentals_text, price_text, macro_text, news_text]:
        if block:
            context_parts.append(block)
    context = "\n\n".join(context_parts)

    chosen_model = model or DEFAULT_PREMARKET_MODELS.get(provider, DEFAULT_NEWS_MODELS[provider])
    return parse_structured(
        provider,
        api_key,
        chosen_model,
        _PREMARKET_SYSTEM,
        f"Generate a pre-market briefing for {sym} based on this context:\n\n{context}",
        PremarketBriefing,
    )
