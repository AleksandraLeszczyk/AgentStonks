"""
Pre-market analysis: synthesizes recent news, historical price context, macro
indicators, and fundamental data into a structured morning briefing via LLM.

Optional: requires one of GEMINI_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY.
News sources are the same as the live news panel (Alpaca + WorldNews fallback).
"""
from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from typing import Literal, Optional

from pydantic import BaseModel

from . import clock
from . import market_hours
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
from .state import current_volume_ratio


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


_SYSTEM_BASE = """\
You are a senior equity analyst briefing a day trader.
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
1. Form a directional bias (bullish / bearish / neutral) with a confidence rating.
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

# What the briefing is *about* depends on when it is generated. The app now
# produces one automatically when the data stream starts, and that can be any
# time of day -- so the framing has to say which session is in question and what
# the trader can still act on. A briefing generated at 11:00 that talks about
# "the open" is describing something that already happened and offering levels
# the tape has already tested.
_PHASE_FRAMING: dict[str, str] = {
    "premarket": """\
TIMING: it is before today's opening bell. This is a pre-market briefing: the
session has not started, so everything you say is about what to expect from the
open and how to trade the day ahead.
""",
    "open": """\
TIMING: the regular session is ALREADY UNDERWAY. This is an intraday situation
briefing, not a pre-market one. You are given how the session has traded so far
-- the opening print, the range, the last price, volume against a normal day's
pace -- and your job is to brief the trader on the situation as it stands RIGHT
NOW and what is still actionable for the remainder of the session.

Concretely: treat the levels the tape has already made today (the session open,
high and low) as the primary structure, and say where price sits inside that
range. Do NOT offer levels that have already been passed as if they were still
ahead, and do NOT frame the bias as a prediction about the open -- the open is a
fact you have been given. If today's action has already invalidated what the
overnight news would have implied, say so explicitly.
""",
    "after_hours": """\
TIMING: today's regular session has CLOSED. Today's full range and close are
given to you as fact. Brief the trader on where the day left the stock and what
that sets up for the NEXT session -- the levels today established are the
structure the next open will be measured against.
""",
    "weekend": """\
TIMING: the market is closed for the weekend. Brief the trader on where the last
session left the stock and what to expect at the NEXT open, weighing any news
that has landed since the close.
""",
}


def _system_prompt(phase: str) -> str:
    """The analyst brief, framed for the moment it is being generated in."""
    framing = _PHASE_FRAMING.get(phase, _PHASE_FRAMING["premarket"])
    return f"{framing}\n{_SYSTEM_BASE}"


# Human wording for each phase, used in the prompt's header line and in the UI
# so the reader knows which session a briefing is talking about.
_PHASE_LABELS: dict[str, str] = {
    "premarket": "before today's open",
    "open": "regular session in progress",
    "after_hours": "after today's close",
    "weekend": "market closed for the weekend",
}

# Minutes in a US regular session (09:30-16:00 ET), the denominator for the
# elapsed fraction the relative-volume pace is projected with.
_RTH_MINUTES = 390

PHASE_TITLES: dict[str, str] = {
    "premarket": "Pre-Market Briefing",
    "open": "Intraday Situation Briefing",
    "after_hours": "Post-Close Briefing",
    "weekend": "Weekend Briefing",
}


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


def _intraday_block(
    bars: list[dict],
    prev_close: Optional[float] = None,
    daily_bars: Optional[list[dict]] = None,
) -> str:
    """What today's session has done so far, from the app's own live bar series.

    Supplied only while the session is actually running. Outside it this block
    is worse than nothing: the live buffer holds whatever the last REST lookback
    happened to catch -- on a Saturday that is a sliver of Friday's
    extended-hours tape -- and summarising sixteen thin after-hours minutes as
    "the session" invites the model to report a 8-cent range as the day's
    structure. The completed day is already described, correctly, by the daily
    close series in `_price_context`.

    The bars come from the caller (the live buffer the chart is already drawing)
    rather than being re-fetched, so the briefing and the chart cannot disagree
    about what today looks like.

    Returns "" when the market is closed, or when nothing from today's regular
    session has printed yet.
    """
    session_start = market_hours.session_open()
    if session_start is None:
        return ""
    todays = []
    for bar in bars:
        ts = clock.parse_iso(bar.get("t"))
        if ts is not None and ts >= session_start:
            todays.append(bar)
    if not todays:
        return ""

    opens = [b["o"] for b in todays if b.get("o") is not None]
    highs = [b["h"] for b in todays if b.get("h") is not None]
    lows = [b["l"] for b in todays if b.get("l") is not None]
    closes = [b["c"] for b in todays if b.get("c") is not None]
    if not (opens and highs and lows and closes):
        return ""
    session_open_px, high, low, last = opens[0], max(highs), min(lows), closes[-1]
    volume = sum(float(b.get("v") or 0.0) for b in todays)

    # Cents, always -- unlike `_fmt_price`, which rounds above $100 to keep the
    # model reasoning in round numbers over multi-month history. These are the
    # levels the trader acts on this afternoon, and on a $332 stock "330 - 333"
    # is not a range anyone can place an order against.
    def px(value: float) -> str:
        return f"{value:.2f}"

    lines = [
        "TODAY'S SESSION SO FAR (live tape):",
        f"- Opening print: {px(session_open_px)}",
        f"- Range so far: {px(low)} - {px(high)}",
        f"- Last price: {px(last)}",
    ]
    span = high - low
    if span > 0:
        # Where in the day's range price sits: near the high is strength held,
        # near the low is a failed bounce -- the single most useful read.
        pos = (last - low) / span * 100.0
        lines.append(f"- Position in today's range: {pos:.0f}% (0% = at the low, 100% = at the high)")
    if session_open_px:
        lines.append(f"- Change from the open: {(last - session_open_px) / session_open_px * 100:+.2f}%")
    if prev_close:
        lines.append(f"- Change from the previous close ({px(prev_close)}): "
                     f"{(last - prev_close) / prev_close * 100:+.2f}%")
    minutes_in = len(todays)
    lines.append(f"- Volume so far: {volume:,.0f} shares over {minutes_in} minute bars")
    lines.append(
        f"- Elapsed: roughly {minutes_in} minutes of the {_RTH_MINUTES}-minute regular session"
    )

    # Raw share count means nothing without a yardstick -- 29M shares is heavy
    # for one name and nothing for another, and heavy by 10:00 is a different
    # statement from heavy by 15:30. Project today's volume to a full session
    # and compare with a normal day, which is the number that tells the model
    # whether the move it is looking at has participation behind it.
    _, baseline = current_volume_ratio(volume, daily_bars or [])
    if baseline and minutes_in:
        pace = (volume / (minutes_in / _RTH_MINUTES)) / baseline
        lines.append(
            f"- Relative volume pace: {pace:.2f}x (today's volume projected to a full "
            f"session vs the {baseline:,.0f}-share average day; 1.0 = normal, "
            "1.5+ = clearly elevated participation)"
        )
    return "\n".join(lines)


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
    phase: Optional[str] = None,
    intraday_bars: Optional[list[dict]] = None,
    prev_close: Optional[float] = None,
    daily_bars: Optional[list[dict]] = None,
) -> Optional[PremarketBriefing]:
    """Generate a structured briefing by gathering multi-source context and calling the LLM.

    `phase` is where the clock sits relative to the regular session (see
    `market_hours.session_phase`); it defaults to reading the clock now. It
    decides what the briefing is *about*: before the open it is the pre-market
    briefing this module was written for, and once the tape is running it is an
    intraday situation briefing about the session in progress.

    `intraday_bars` is the live bar buffer for the symbol, used only to describe
    what today has actually done so far. Passing the app's own series (rather
    than re-fetching) keeps the briefing and the chart telling the same story.
    """
    sym = symbol.strip().upper()
    phase = phase or market_hours.session_phase()

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
    intraday_text = _intraday_block(intraday_bars or [], prev_close, daily_bars)
    # Anchor the target upside math on the live price when the tape is running,
    # and on the most recent daily close otherwise. Quoting analyst upside
    # against a stale close while the stock is 3% up on the day would misstate
    # every distance-to-target in the briefing.
    anchor = None
    if phase == "open" and intraday_bars:
        closes = [b.get("c") for b in intraday_bars if b.get("c") is not None]
        anchor = closes[-1] if closes else None
    targets_text = _targets_block(sym, current_price=anchor or last_close.get("7d"))

    now_et = datetime.now(timezone.utc).astimezone(market_hours.MARKET_TZ)
    context_parts = [
        f"Symbol: {sym}",
        f"Analysis time: {now_et.strftime('%Y-%m-%d %H:%M')} ET ({_PHASE_LABELS.get(phase, phase)})",
    ]
    for block in [
        intraday_text,
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
    subject = (
        f"an intraday situation briefing for {sym}"
        if phase == "open"
        else f"a pre-market briefing for {sym}"
    )
    return parse_structured(
        provider,
        api_key,
        chosen_model,
        _system_prompt(phase),
        f"Generate {subject} based on this context:\n\n{context}",
        PremarketBriefing,
    )


def generate_for_symbols(
    app,
    symbols: list[str],
    provider: str,
    api_key: str,
    alpaca_key: str = "",
    alpaca_secret: str = "",
    worldnews_key: str = "",
    model: Optional[str] = None,
) -> None:
    """Brief every symbol in turn, publishing each one onto `app` as it lands.

    Results are written per symbol rather than in one batch at the end so the
    panel fills in progressively -- briefing a basket is several seconds of LLM
    time per name, and a trader watching the tab should not have to wait for the
    slowest symbol to see the first.

    Never raises: one symbol failing (a thin ticker, a provider hiccup) records
    its error and leaves the rest to run. The caller is a background thread with
    nowhere to propagate an exception to.
    """
    phase = market_hours.session_phase()
    app.premarket_phase = phase
    app.premarket_status = f"Generating {PHASE_TITLES.get(phase, 'briefing')}…"
    app.premarket_errors = {}
    app.premarket_briefings = {}
    app.premarket_pending = list(symbols)

    for sym in symbols:
        state = app.sym(sym)
        with state.lock:
            bars = list(state.bars)
            prev_close = state.prev_close
            daily_bars = list(state.daily_bars)
        try:
            briefing = generate_premarket_analysis(
                symbol=sym,
                provider=provider,
                api_key=api_key,
                alpaca_key=alpaca_key,
                alpaca_secret=alpaca_secret,
                worldnews_key=worldnews_key,
                model=model,
                phase=phase,
                intraday_bars=bars,
                prev_close=prev_close,
                daily_bars=daily_bars,
            )
        except Exception as exc:
            app.premarket_errors = {**app.premarket_errors, sym: str(exc)}
        else:
            if briefing is not None:
                app.premarket_briefings = {**app.premarket_briefings, sym: briefing}
            else:
                app.premarket_errors = {
                    **app.premarket_errors, sym: "the model returned no briefing"
                }
        app.premarket_pending = [s for s in app.premarket_pending if s != sym]

    app.premarket_generated_at = datetime.now(timezone.utc)
    done, failed = len(app.premarket_briefings), len(app.premarket_errors)
    app.premarket_status = (
        f"{PHASE_TITLES.get(phase, 'Briefing')} ready ({done} of {done + failed} symbols)"
        if done
        else "Briefing failed"
    )


def launch_premarket_analysis(
    app,
    symbols: list[str],
    provider: str,
    api_key: str,
    alpaca_key: str = "",
    alpaca_secret: str = "",
    worldnews_key: str = "",
    model: Optional[str] = None,
) -> bool:
    """Start `generate_for_symbols` on a background thread. Returns False (and
    records why) when there is no LLM key to run it with.

    Backgrounded because this is several seconds of LLM work per symbol and it
    is kicked off by the same click that starts the data stream -- the tape must
    not wait on an analyst.
    """
    if not symbols:
        return False
    if not api_key:
        app.premarket_status = (
            f"No API key for {provider} — set {provider.upper()}_API_KEY to get a briefing "
            "automatically when the stream starts."
        )
        app.premarket_briefings = {}
        app.premarket_errors = {}
        app.premarket_pending = []
        return False
    threading.Thread(
        target=generate_for_symbols,
        args=(app, list(symbols), provider, api_key,
              alpaca_key, alpaca_secret, worldnews_key, model),
        daemon=True,
    ).start()
    return True
