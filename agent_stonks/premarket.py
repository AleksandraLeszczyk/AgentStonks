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
    fetch_analyst_targets,
    fetch_close_series,
    fetch_earnings_dates,
    fetch_market_indicators,
    fetch_static_analysis,
)
from .llm import DEFAULT_NEWS_MODELS, parse_structured
from .news import fetch_news_with_fallback, get_last_week_news
from .rest import fetch_corporate_actions


DEFAULT_PREMARKET_MODELS: dict[str, str] = {
    "gemini": "gemini-3.5-flash",
    "openai": "gpt-5.6-luna",
    "anthropic": "claude-haiku-4-5-20251001",
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

You have, when available, current Wall Street price targets: the yfinance
consensus (mean/high/low across all covering analysts, with implied upside vs
the latest close) and the standing target from UBS, Morgan Stanley, and
Barclays. Use them to frame upside/downside: little room to the consensus mean
(or price already above it) argues against chasing a gap-up; a wide gap to the
mean leaves room to run; price outside the whole high-low range is a valuation
extreme worth flagging.

You may also have incoming corporate actions (ex-dividend dates, splits,
mergers, spin-offs) over the next two weeks. Treat them as scheduled catalysts:
an imminent ex-dividend date mechanically lowers the open by roughly the
dividend (not a bearish signal), splits reset every price level and often draw
retail flow, and merger/spin-off terms can pin or reprice the stock. Fold them
into the bias, catalysts, and risk factors where relevant.

Your job:
1. Form a directional morning bias (bullish / bearish / neutral) with a confidence rating.
2. Write a 2-3 sentence executive summary (be specific, cite concrete data).
3. List 2-5 catalysts that drive the bias — only keep news that is DIRECTLY relevant.
4. Call out 2-4 key price levels the trader must monitor today (support, resistance, pivots).
   Fold the analyst targets in where relevant — the consensus mean and the tracked
   firms' targets are natural resistance/objective levels.
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


def _targets_block(symbol: str, current_price: Optional[float] = None) -> str:
    """Analyst price targets: the yfinance consensus (mean/high/low + upside)
    and the tracked firms' standing targets (UBS, Morgan Stanley, Barclays)."""
    try:
        data = fetch_analyst_targets(symbol, current_price=current_price)
    except Exception:
        return ""
    cons = data.get("consensus") or {}
    firms = data.get("firms") or {}
    if cons.get("mean") is None and not firms:
        return ""

    lines: list[str] = []
    mean = cons.get("mean")
    if mean is not None:
        up = cons.get("mean_upside_pct")
        parts = [f"mean {_fmt_price(mean)}" + (f" ({up:+.1f}%)" if up is not None else "")]
        if cons.get("high") is not None:
            parts.append(f"high {_fmt_price(cons['high'])}")
        if cons.get("low") is not None:
            parts.append(f"low {_fmt_price(cons['low'])}")
        if cons.get("num_analysts"):
            parts.append(f"{cons['num_analysts']} analysts")
        if cons.get("recommendation"):
            parts.append(f"rec {cons['recommendation']}")
        lines.append("  Consensus (yfinance): " + ", ".join(parts))
    for name, f in firms.items():
        up = f.get("upside_pct")
        lines.append(
            f"  {name}: {_fmt_price(f['target'])}"
            + (f" ({up:+.1f}%)" if up is not None else "")
            + f", set {f['date']}"
        )
    for insight in data.get("insights", []):
        lines.append(f"  • {insight}")
    return "Analyst price targets:\n" + "\n".join(lines)


def _corporate_actions_block(symbol: str, alpaca_key: str, alpaca_secret: str, days_ahead: int = 14) -> str:
    """Incoming corporate actions (ex-dividends, splits, mergers, ...) from Alpaca,
    one chronological line per action. Empty when keys are missing or nothing is scheduled."""
    if not alpaca_key or not alpaca_secret:
        return ""
    try:
        actions = fetch_corporate_actions(symbol, alpaca_key, alpaca_secret, days_ahead=days_ahead)
    except Exception:
        return ""
    if not actions:
        return ""
    lines: list[str] = []
    for action in actions[:10]:
        details = ", ".join(
            f"{field} {value}"
            for field, value in action.items()
            if field not in ("type", "date") and value not in (None, "", False)
        )
        lines.append(
            f"  {action.get('date') or 'date TBD'}: {action['type'].replace('_', ' ')}"
            + (f" ({details})" if details else "")
        )
    return f"Incoming corporate actions (next {days_ahead} days):\n" + "\n".join(lines)


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

    price_text, last_close = _price_context(sym)
    macro_text = _macro_context(days=60)
    news_text = _news_block(news_items)
    fundamentals_text = _fundamentals_block(sym)
    earnings_text = _earnings_block(sym)
    corporate_actions_text = _corporate_actions_block(sym, alpaca_key, alpaca_secret)
    # The most recent daily close anchors the target upside math (no live
    # pre-market print is available in this offline briefing path).
    targets_text = _targets_block(sym, current_price=last_close.get("7d"))

    context_parts = [
        f"Symbol: {sym}",
        f"Analysis date: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
    ]
    for block in [
        earnings_text,
        corporate_actions_text,
        fundamentals_text,
        targets_text,
        price_text,
        macro_text,
        news_text,
    ]:
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
