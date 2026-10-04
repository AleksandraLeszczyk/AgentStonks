import base64
import html
import logging
import os
import re
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, fields, replace as dc_replace
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from . import (
    apple_models,
    candidates,
    candle_patterns,
    last_setup,
    market_hours,
    model_overlays,
    session_store,
)
from . import apple_trader_ui
from .model_catalogue_ui import model_catalogue_panel
from .agent import (
    AGENT_PERSONALITIES,
    DEFAULT_PERSONALITY,
    PREMARKET_PERSONALITY,
    launch_agent,
    selectable_personalities,
    sell_everything_and_stop,
    stop_agent,
)
from .apple_trader import (
    APPLE_TRADER_AVATAR,
    APPLE_TRADER_KEY,
    APPLE_TRADER_LABEL,
    AppleTraderConfig,
    launch_apple_trader,
)
from .apple_trader import DEFAULT_TICKER as APPLE_TRADER_TICKER
from .orchestra import (
    ORCHESTRA_AVATAR,
    ORCHESTRA_KEY,
    ORCHESTRA_LABEL,
    OrchestraConfig,
    launch_orchestra,
    racer_label,
)
from .automatic import AUTOMATIC_AVATAR, AUTOMATIC_KEY, AUTOMATIC_LABEL, launch_automatic
from .charts import (
    build_analysis_gauges,
    build_chart,
    build_gamma_chart,
    build_performance_chart,
    empty_chart,
)
from .config import (
    AGENT_CYCLE_SEC,
    AGENT_EQUITY_HISTORY_MAXLEN,
    AGENT_LOG_POLL_SEC,
    AGENT_PERFORMANCE_POLL_SEC,
    APPLE_TRADER_CYCLE_SEC,
    CHART_POLL_SEC,
    DEFAULT_DATA_SOURCE,
    DEFAULT_HISTORY_FEED,
    DEFAULT_LIVE_SOURCE,
    DEFAULT_TRADING_MODE,
    HISTORY_FEEDS,
    LIVE_SOURCE_LABELS,
    LIVE_SOURCES,
    LIVE_TRADING_ENV_FLAG,
    TRADING_MODES,
    MAX_BARS,
    NEWS_IMPACT_COLORS,
    OPTIONS_POLL_SEC,
    OPTIONS_WALL_HISTORY_MAXLEN,
    PAPER_STARTING_CASH,
    PALETTE,
    POLL_SEC,
    PREMARKET_POLL_SEC,
    ORCHESTRA_CANDIDATES_POLL_SEC,
    SESSION_START,
    TACTICS_MOMENTUM_WINDOW_MIN,
    TIMEFRAMES,
    TRADE_FIXED_COST,
)
from .datalog import log_fetch, log_fetch_failure
from .decisions import DecisionTracker
from .historical import fetch_intraday_history_bars, fetch_market_indicators
from .llm import DEFAULT_AGENT_MODELS, DEFAULT_NEWS_MODELS, ENV_KEYS, PROVIDERS, models_for
from .news import YFINANCE_FEED, fetch_live_news, score_news_impacts
from .premarket import (
    DEFAULT_PREMARKET_MODELS,
    PHASE_TITLES,
    PremarketBriefing,
    launch_premarket_analysis,
)
from .options import fetch_options_walls_data, net_gamma_exposure
from .performance import compute_equity_curve, decision_markers, summarize
from .profile_model import predicted_open_profile
from .report import build_report_html
from .rest import fetch_bars, fetch_daily_bars, fetch_trades
from .state import (
    PRICE_AXIS_ALERT_FIELDS,
    AppState,
    SymbolState,
    append_agent_log,
    format_alert,
    format_tool_kv,
    momentum_pct,
    today_daily_bar,
)
from .tactics import tactic_price_levels, tactics_summaries
from . import bar_history, gamma_history, minute_momentum, newsimpact_model, stream_common
from .trade_sound import next_trade_cue, play_trade_sound
from .page_watchdog import WATCHDOG_BEAT_SEC, describe_reload, page_watchdog
from .quote_card import quote_card
from .trading_mode import (
    ENV_KEYS as TRADING_ENV_KEYS,
    MODE_LABELS,
    PAPER_FALLBACK_ENV,
    credentials_for,
    live_trading_enabled,
    resolve_broker,
)
from .stream_common import TF_MINUTES
from .stream import (
    adopt_orphaned_session,
    backfill_bars,
    launch_stream,
    launch_stream_news,
    reap_dead_sessions,
    register_live_session,
    stop_streams,
)
from .technical_analysis import (
    analyze_intraday,
    analyze_market,
    analyze_trend,
    get_put_call_walls_and_gamma,
)
from .volume_baseline import (
    BAND_WINDOWS,
    DEFAULT_VOLUME_BASELINE,
    VOLUME_BASELINE_WINDOWS,
    lookback_days,
    minute_volume_baseline,
    volume_band,
)


def _volume_baseline(symbol: str, bars: "list[dict]", state: AppState) -> "dict | None":
    """The volume panel's "usual volume" reference for the selected window.

    The prior sessions are fetched here rather than kept on `SymbolState`, for
    the reason the day-range overlay fetches its own history: the live buffer
    holds today only, and the baseline wants a week of *consolidated* minutes,
    which is not the tape the stream is on. `fetch_intraday_history_bars`
    caches for an hour and swallows its own failures, so the fragment's poll
    does not turn into a download per rerun, and a window that cannot be built
    draws nothing.
    """
    window = state.volume_baseline_window
    if window not in VOLUME_BASELINE_WINDOWS or window == "off":
        return None
    if window in BAND_WINDOWS:
        return _volume_band_baseline(symbol, bars, state, window)
    days = lookback_days(window)
    history = fetch_intraday_history_bars(symbol, days) if days else []
    return minute_volume_baseline(window, bars, history)


# Which history each bar source's band is built from, best first. Finnhub has
# no minute history of its own; it streams the consolidated tape, so its bars
# are read against SIP, or yfinance's rendering of the same tape without SIP.
_BAND_HISTORY_FOR_SOURCE: dict[str, tuple[str, ...]] = {
    "yfinance": ("yfinance",),
    "sip": ("sip",),
    "iex": ("iex",),
    "finnhub": ("sip", "yfinance"),
}


def _volume_band_baseline(
    symbol: str, bars: "list[dict]", state: AppState, window: str
) -> "dict | None":
    """The mean + 1 sigma volume band, one per tape the chart's bars came off.

    Each bar carries its source (`bar["src"]`, see
    `bar_history.BAR_SOURCE_OF_FEED`), and the band behind it is built from
    that source's own last trading week: IEX is ~4% of SIP, so one band for a
    buffer mixing the two would be wrong for one of them everywhere. A bar with
    no recorded source (a buffer loaded before sources were recorded) gets no
    band.
    """
    span = stream_common.TF_MINUTES.get(state.timeframe)
    if not span or span > 60:
        return None
    sources = {b.get("src") for b in bars} & set(_BAND_HISTORY_FOR_SOURCE)
    if not sources:
        return None
    dates = _session_dates_of(bars)
    today = dates[-1] if dates else None
    if today is None:
        return None
    days = lookback_days(window)
    built: dict[str, "dict | None"] = {}

    def band_of(history_source: str) -> "dict | None":
        if history_source not in built:
            if history_source == "yfinance":
                history = fetch_intraday_history_bars(symbol, days)
            else:
                history = bar_history.fetch_week_minute_bars(
                    symbol, history_source, state.api_key, state.api_secret, days
                )
            built[history_source] = _cached_volume_band(
                symbol, history_source, history, today, span
            )
        return built[history_source]

    by_source: dict[str, dict] = {}
    for src in sorted(sources):
        for history_source in _BAND_HISTORY_FOR_SOURCE[src]:
            band = band_of(history_source)
            if band is not None:
                by_source[src] = band
                break
    if not by_source:
        return None
    return {
        "key": window,
        "label": VOLUME_BASELINE_WINDOWS[window],
        "span": span,
        "band_by_source": by_source,
    }


# Bands by (symbol, history source, ET day, span), each with a signature of the
# history it was built from. Building one parses a week of minute bars (~0.3 s
# for SIP's ~8,000), and the chart fragment reruns every few seconds over the
# same, day-cached history -- so it is rebuilt only when that history changes.
_VOLUME_BAND_CACHE: "dict[tuple[str, str, str, int], tuple[tuple, dict | None]]" = {}


def _cached_volume_band(
    symbol: str, history_source: str, history: "list[dict]", today: str, span: int
) -> "dict | None":
    signature = (
        len(history),
        history[0].get("t") if history else None,
        history[-1].get("t") if history else None,
    )
    key = (symbol, history_source, today, span)
    hit = _VOLUME_BAND_CACHE.get(key)
    if hit is not None and hit[0] == signature:
        return hit[1]
    band = volume_band(history, today, span=span)
    result = {**band, "history": history_source} if band is not None else None
    _VOLUME_BAND_CACHE[key] = (signature, result)
    return result


def _session_dates_of(bars: "list[dict]") -> "list[str]":
    """The ET session dates in `bars`, oldest first."""
    return sorted({
        pd.Timestamp(b["t"]).tz_convert(market_hours.MARKET_TZ).strftime("%Y-%m-%d")
        for b in bars
        if b.get("t")
    })


def _session_id() -> str:
    """This Streamlit session's id, or "" outside a script run."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        ctx = get_script_run_ctx()
        return ctx.session_id if ctx else ""
    except Exception:
        return ""


def _session_is_active(session_id: str) -> bool:
    try:
        from streamlit.runtime import Runtime

        return Runtime.instance().is_active_session(session_id)
    except Exception:
        # No runtime to ask: never stop a session on a guess.
        return True


def _get_state() -> AppState:
    # Any rerun, in any session, stops the streams of sessions whose browser
    # has gone for good (see stream.reap_dead_sessions).
    reap_dead_sessions(_session_is_active)
    if "app_state" not in st.session_state:
        # A browser coming back from a dropped connection, or a reload, is a
        # new session: it takes over the state that kept running without it
        # rather than opening a stopped copy beside it.
        session_id = _session_id()
        adopted = (
            adopt_orphaned_session(session_id, _session_is_active) if session_id else None
        )
        if adopted is not None:
            if not (adopted.recovery or {}).get("pending"):
                adopted.recovery = {"kind": "reconnect", "at": time.time()}
            st.session_state["app_state"] = adopted
    if "app_state" not in st.session_state:
        fresh = AppState()
        # Today's session as it was saved before a restart (or in another tab):
        # the ledger, the log and the levels come back; the agent stays stopped
        # until ▶ Start, which continues them.
        try:
            session_store.restore(fresh)
        except Exception:
            logging.getLogger(__name__).exception("Restoring today's saved session failed")
        st.session_state["app_state"] = fresh
    state = st.session_state["app_state"]
    session_store.start_autosave(state)
    # Streamlit's dev-mode autoreload reruns this script on every save but keeps
    # the same AppState instance alive in session_state. If a field was added to
    # AppState after this instance was constructed, the instance's __class__ (and
    # therefore __getattr__) still points at the pre-edit definition, so reading
    # the new field raises AttributeError instead of falling back to a default.
    # Writing straight into __dict__ sidesteps the class entirely.
    state.__dict__.setdefault("symbols", [])
    state.__dict__.setdefault("symbol_states", {})
    return state


def _parse_symbols(text: str) -> list[str]:
    """'aapl, tsla msft' -> ['AAPL', 'TSLA', 'MSFT'] (deduped, order kept)."""
    seen: list[str] = []
    for raw in re.split(r"[,;\s]+", text or ""):
        sym = raw.strip().upper()
        if sym and sym not in seen:
            seen.append(sym)
    return seen


def _effective_symbols(state: AppState, symbols_input: str) -> list[str]:
    """Symbols the panels should render: the sidebar input, falling back to
    whatever is currently streamed."""
    return _parse_symbols(symbols_input) or list(state.symbols)


# Agents that aren't LLM personalities and so have no entry in
# AGENT_PERSONALITIES: the Automatic orchestrator and the rule-based Apple
# Trader and Orchestra. They still need a label and a face in the picker.
_NON_LLM_AGENTS: dict[str, tuple[str, str]] = {
    AUTOMATIC_KEY: (AUTOMATIC_LABEL, AUTOMATIC_AVATAR),
    APPLE_TRADER_KEY: (APPLE_TRADER_LABEL, APPLE_TRADER_AVATAR),
    ORCHESTRA_KEY: (ORCHESTRA_LABEL, ORCHESTRA_AVATAR),
}

# The agents that place their own orders from a fixed loop: no LLM key needed,
# and their own parameter panel instead of provider/model.
RULE_AGENT_KEYS = (APPLE_TRADER_KEY, ORCHESTRA_KEY)


def _personality_label(key: str) -> str:
    """Display label for a personality key, including the non-LLM agents."""
    if key in _NON_LLM_AGENTS:
        return _NON_LLM_AGENTS[key][0]
    entry = AGENT_PERSONALITIES.get(key) or AGENT_PERSONALITIES[DEFAULT_PERSONALITY]
    return entry["label"]


AVATAR_DIR = Path(__file__).resolve().parent.parent / "data" / "avatars"


@lru_cache(maxsize=None)
def _avatar_data_uri(key: str) -> Optional[str]:
    """Base64 data URI for a personality's avatar PNG, or None if the file is missing."""
    if key in _NON_LLM_AGENTS:
        filename = _NON_LLM_AGENTS[key][1]
    else:
        entry = AGENT_PERSONALITIES.get(key) or AGENT_PERSONALITIES[DEFAULT_PERSONALITY]
        filename = entry["avatar"]
    try:
        data = (AVATAR_DIR / filename).read_bytes()
    except OSError:
        return None
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def _parse_ma_periods(ma_selection: list[str]) -> list[int]:
    mapping = {"VWMA(5)": 5, "VWMA(15)": 15, "VWMA(60)": 60}
    return [mapping[s] for s in ma_selection if s in mapping]


def _parse_avg_flags(ma_selection: list[str]) -> tuple[bool, bool, bool]:
    return "7d Avg" in ma_selection, "28d Avg" in ma_selection, "1y Avg" in ma_selection


def _strip_html(text: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", " ", text)).strip()


def wrap_text(text: Optional[str], width: int = 80) -> str:
    """Insert <br> tags to wrap text at the given character width."""
    if not text:
        return ""
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    length = 0
    for word in words:
        if length + len(word) + (1 if current else 0) > width and current:
            lines.append(" ".join(current))
            current = [word]
            length = len(word)
        else:
            current.append(word)
            length += len(word) + (1 if len(current) > 1 else 0)
    if current:
        lines.append(" ".join(current))
    return "<br>".join(lines)


_IMPACT_STYLE: dict[str, dict[str, str]] = {
    "positive": {"label": "positive impact", "dot": NEWS_IMPACT_COLORS["positive"], "bg": "#0d2b24", "border": "#1a4a3d", "text": "#26c6a2"},
    "negative": {"label": "negative impact", "dot": NEWS_IMPACT_COLORS["negative"], "bg": "#2b0d0d", "border": "#4a1a1a", "text": "#ef5350"},
    "neutral":  {"label": "neutral impact",  "dot": NEWS_IMPACT_COLORS["neutral"],  "bg": "#1e1e2e", "border": "#2a2d3a", "text": "#888888"},
    "small":    {"label": "small impact",    "dot": NEWS_IMPACT_COLORS["small"],    "bg": "#2b1a0d", "border": "#4a2d1a", "text": "#fb923c"},
    "unknown":  {"label": "unknown impact",  "dot": NEWS_IMPACT_COLORS["unknown"],  "bg": "#1a1d27", "border": "#2a2d3a", "text": "#555555"},
}


def _impact_badge(impact: str, detail: Optional[dict] = None) -> str:
    cfg = _IMPACT_STYLE.get(impact, _IMPACT_STYLE["unknown"])
    label = cfg["label"]
    title = ""
    if detail and detail.get("source") == newsimpact_model.SOURCE_MODEL:
        # Say which estimator spoke, and put its numbers -- or why it could not
        # score the article -- in the tooltip.
        if detail.get("status") != newsimpact_model.STATUS_SCORED:
            label = detail.get("short") or "unknown"
        label = f"{label} · model"
        title = f' title="{html.escape(detail.get("reason", ""), quote=True)}"'
    return (
        f'<span{title} style="display:inline-flex;align-items:center;gap:5px;'
        f'padding:3px 9px;border-radius:12px;background:{cfg["bg"]};'
        f'border:1px solid {cfg["border"]};font-size:10px;font-weight:600;'
        f'color:{cfg["text"]};white-space:nowrap;letter-spacing:0.02em;">'
        f'<span style="width:6px;height:6px;border-radius:50%;'
        f'background:{cfg["dot"]};display:inline-block;flex-shrink:0;"></span>'
        f'{html.escape(label)}</span>'
    )


def _news_html(
    news: list[dict],
    symbol: str,
    impacts: Optional[dict] = None,
    details: Optional[dict] = None,
) -> str:
    if not news:
        return (
            f"<p style='color:{PALETTE['muted']};padding:12px'>"
            f"No recent news for {symbol}.</p>"
        )
    impacts = impacts or {}
    details = details or {}
    cards = []
    for item in news[:12]:
        ts = pd.to_datetime(item.get("created_at")).strftime("%b %d  %H:%M")
        src = html.escape(item.get("source", ""))
        if item.get("feed") == YFINANCE_FEED:
            src += f'<span style="color:{PALETTE["muted"]}"> via Yahoo</span>'
        headline = html.escape(_strip_html(item.get("headline", "")))
        summary = _strip_html(item.get("summary") or "")[:180].rstrip()
        summary = html.escape(summary)
        url = html.escape(item.get("url", "#"))
        news_id = str(item.get("id", ""))
        impact = impacts.get(news_id)
        badge = _impact_badge(impact or "unknown", details.get(news_id))
        cards.append(
            f"""
        <div style="background:{PALETTE['panel']}; border-radius:8px; padding:12px 14px;
                    border:1px solid {PALETTE['grid']}; display:flex; flex-direction:column;
                    gap:6px; min-width:0;">
          <div style="font-size:11px; color:{PALETTE['muted']}; display:flex; align-items:center;
                      justify-content:space-between; gap:8px; flex-wrap:wrap;">
            <span>{ts} · <span style="color:{PALETTE['accent']}">{src}</span></span>
            {badge}
          </div>
          <a href="{url}" target="_blank"
             style="color:{PALETTE['text']}; font-weight:600;
                    text-decoration:none; font-size:13px; line-height:1.4">
            {headline}
          </a>
          <div style="font-size:12px; color:{PALETTE['muted']}; line-height:1.5">
            {summary}{"…" if summary else ""}
          </div>
        </div>"""
        )
    return f"""
    <div style="font-family:Inter,sans-serif; padding:4px 0 12px;">
      <h3 style="color:{PALETTE['text']}; font-size:14px; margin:0 0 10px 0">
        📰 Latest news · <b style="color:{PALETTE['accent']}">{symbol}</b>
      </h3>
      <div style="display:grid; grid-template-columns:repeat(auto-fill,minmax(340px,1fr));
                  gap:10px;">
        {''.join(cards)}
      </div>
    </div>"""


build_news_html = _news_html


def _today_range(
    daily_bars: list[dict], intraday_bars: list[dict]
) -> tuple[float | None, float | None, float | None]:
    """(low, high, open) for today's session.

    Prefers today's still-forming daily bar; falls back to today's intraday bars
    (for pre-open/lagging feeds where the daily bar hasn't published yet)."""
    bar = today_daily_bar(daily_bars)
    if bar is not None:
        try:
            return float(bar["l"]), float(bar["h"]), float(bar["o"])
        except (KeyError, TypeError, ValueError):
            pass

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    todays = [b for b in intraday_bars if str(b.get("t", "")).startswith(today)]
    if not todays:
        return None, None, None
    try:
        low = min(float(b["l"]) for b in todays)
        high = max(float(b["h"]) for b in todays)
        open_ = float(todays[0]["o"])
    except (KeyError, TypeError, ValueError):
        return None, None, None
    return low, high, open_


def _signed_dollars(value: float) -> str:
    """Compact signed dollar amount: +$1.23B, -$456.70M, +$12.30K."""
    sign = "+" if value >= 0 else "-"
    mag = abs(value)
    for div, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if mag >= div:
            return f"{sign}${mag / div:,.2f}{suffix}"
    return f"{sign}${mag:,.0f}"


def _live_net_gamma(sym_state: SymbolState, spot: float | None = None) -> float | None:
    """Total net dealer gamma ($ per 1% move) from the symbol's latest options
    chain, at `spot` when given (so the card and the chart panel's last bar
    agree), or None before a chain has arrived. Also kicks off a background
    refresh."""
    _refresh_option_chain(sym_state)
    with sym_state.lock:
        data = sym_state.options_chain
    if not data or not data.get("strikes"):
        return None
    if spot:
        at_spot = net_gamma_exposure(data, [spot])
        if at_spot is not None:
            return float(at_spot[0])
    return float(sum(data["calls_gamma_exposure"]) + sum(data["puts_gamma_exposure"]))


def _live_gamma_series(sym_state: SymbolState, bars: list[dict], timeframe: str) -> dict:
    """The chart's net gamma panel: each bar's value taken once, at its close,
    with the chain the app had then (`gamma_history`), the forming bar's at its
    current close -- or a note saying why there is none."""
    _refresh_option_chain(sym_state)
    with sym_state.lock:
        data = sym_state.options_chain
    return gamma_history.series(sym_state.symbol, timeframe, bars, data)


def _quote_html(
    price: float | None,
    prev_close: float | None,
    bid: float | None,
    bid_size: float | None,
    ask: float | None,
    ask_size: float | None,
    symbol: str,
    today_low: float | None = None,
    today_high: float | None = None,
    current_momentum: float | None = None,
    daily_momentum: float | None = None,
    net_gamma: float | None = None,
    day_lines: "set[str] | frozenset[str]" = frozenset(),
) -> str:
    """`day_lines` holds which of "day_low" / "day_high" are drawn on the price
    chart; those cards get an accent border. Both cards carry a `data-toggle`
    that `quote_card` reports when clicked."""
    if price is None and bid is None and ask is None:
        return ""
    change = price - prev_close if price is not None and prev_close else None
    pct = (change / prev_close * 100) if change is not None and prev_close else None
    if change is None:
        arrow, chg_color, delta_str = "", PALETTE["muted"], ""
    elif change >= 0:
        arrow, chg_color = "▲", "#26c6a2"
        delta_str = f"+{change:.2f} ({pct:+.2f}%)"
    else:
        arrow, chg_color = "▼", "#ef5350"
        delta_str = f"{change:.2f} ({pct:.2f}%)"

    symbol_chip = (
        f'<span style="font-size:13px;font-weight:700;color:{PALETTE["accent"]};'
        f'letter-spacing:0.04em;margin-right:10px;">{html.escape(symbol)}</span>'
    )
    price_row = ""
    if price is not None:
        price_row = (
            f'<div style="display:flex;align-items:baseline;gap:12px;margin-bottom:10px;">'
            f'{symbol_chip}'
            f'<span style="font-size:28px;font-weight:700;color:{PALETTE["text"]};'
            f'letter-spacing:-0.5px;">${price:,.4f}</span>'
            f'<span style="font-size:14px;font-weight:600;color:{chg_color};">'
            f'{arrow} {delta_str}</span>'
            f'</div>'
        )
    else:
        price_row = f'<div style="margin-bottom:6px;">{symbol_chip}</div>'

    def _side(
        label: str, p: float | None, sz: float | None, color: str,
        toggle: str | None = None,
    ) -> str:
        if p is None:
            return ""
        size_str = f'<span style="font-size:11px;color:{PALETTE["muted"]};margin-left:4px;">{sz:,.0f}</span>' if sz else ""
        border = PALETTE["grid"]
        extra = ""
        if toggle is not None:
            shown = toggle in day_lines
            if shown:
                border = PALETTE["accent"]
            verb = "Hide" if shown else "Show"
            extra = f' data-toggle="{toggle}" title="{verb} this line on the price chart"'
        cursor = "cursor:pointer;" if toggle is not None else ""
        return (
            f'<div{extra} style="display:flex;flex-direction:column;align-items:center;'
            f'background:{PALETTE["panel"]};border:1px solid {border};{cursor}'
            f'border-radius:8px;padding:8px 16px;min-width:100px;">'
            f'<span style="font-size:10px;font-weight:600;color:{PALETTE["muted"]};'
            f'letter-spacing:0.08em;text-transform:uppercase;margin-bottom:4px;">{label}</span>'
            f'<span style="font-size:18px;font-weight:700;color:{color};">${p:,.4f}</span>'
            f'{size_str}'
            f'</div>'
        )

    low_card = _side("Day Low", today_low, None, PALETTE["muted"], toggle="day_low")
    bid_card = _side("Bid", bid, bid_size, "#ef5350")
    ask_card = _side("Ask", ask, ask_size, "#26c6a2")
    high_card = _side("Day High", today_high, None, PALETTE["muted"], toggle="day_high")
    spread_row = ""
    if bid is not None and ask is not None:
        spread = ask - bid
        spread_row = (
            f'<span style="font-size:11px;color:{PALETTE["muted"]};align-self:center;">'
            f'spread {spread:.4f}</span>'
        )

    ba_row = ""
    if bid_card or ask_card or low_card or high_card:
        ba_row = (
            f'<div style="display:flex;gap:10px;align-items:stretch;">'
            f'{low_card}{bid_card}{spread_row}{ask_card}{high_card}'
            f'</div>'
        )

    def _stat_card(label: str, value: str, color: str) -> str:
        return (
            f'<div style="display:flex;flex-direction:column;align-items:center;'
            f'background:{PALETTE["panel"]};border:1px solid {PALETTE["grid"]};'
            f'border-radius:8px;padding:8px 16px;min-width:100px;">'
            f'<span style="font-size:10px;font-weight:600;color:{PALETTE["muted"]};'
            f'letter-spacing:0.08em;text-transform:uppercase;margin-bottom:4px;">{label}</span>'
            f'<span style="font-size:18px;font-weight:700;color:{color};">{value}</span>'
            f'</div>'
        )

    def _momentum_color(m: float | None) -> str:
        if m is None or abs(m) < 1e-9:
            return PALETTE["muted"]
        return "#26c6a2" if m > 0 else "#ef5350"

    cur_mom_card = (
        _stat_card("Momentum (10m)", f"{current_momentum:+.2f}%", _momentum_color(current_momentum))
        if current_momentum is not None else ""
    )
    day_mom_card = (
        _stat_card("Momentum (day)", f"{daily_momentum:+.2f}%", _momentum_color(daily_momentum))
        if daily_momentum is not None else ""
    )
    # Dollar gamma per 1% move, summed over the nearest expiry's strikes (the
    # same total behind the Options tab's "Net gamma regime").
    gamma_card = (
        _stat_card("Net Gamma (1%)", _signed_dollars(net_gamma), _momentum_color(net_gamma))
        if net_gamma is not None else ""
    )

    stats_row = ""
    if cur_mom_card or day_mom_card or gamma_card:
        stats_row = (
            f'<div style="display:flex;gap:10px;align-items:stretch;margin-top:10px;">'
            f'{cur_mom_card}{day_mom_card}{gamma_card}'
            f'</div>'
        )

    return (
        f'<div style="font-family:Inter,monospace;padding:4px 0 8px;">'
        f'{price_row}{ba_row}{stats_row}'
        f'</div>'
    )


# Day Low / Day High lines on the Live price chart, switched by clicking those
# cards in the quote: {symbol: {"day_low", "day_high"} subset}.
_DAY_LINES_STATE_KEY = "live_day_lines"
_DAY_LINE_KEYS = ("day_low", "day_high")


def _day_lines_shown(symbol: str) -> "set[str]":
    return set(st.session_state.get(_DAY_LINES_STATE_KEY, {}).get(symbol, ()))


def _day_range_lines(symbol: str, daily_bars: list[dict], bars: list[dict]) -> "dict | None":
    """{"day_low": ..., "day_high": ...} for the lines switched on for
    `symbol`, from the same `_today_range` the quote card shows; None when none
    are on."""
    shown = _day_lines_shown(symbol)
    if not shown:
        return None
    low, high, _ = _today_range(daily_bars, bars)
    levels = {"day_low": low, "day_high": high}
    return {k: levels[k] for k in _DAY_LINE_KEYS if k in shown}


@st.fragment(run_every=POLL_SEC)
def _price_ticker() -> None:
    state = _get_state()
    st.caption(f"Status: {state.status}")
    if state.news_status not in ("Idle", state.status):
        st.caption(f"News: {state.news_status}")
    for sym_state in state.iter_symbol_states():
        with sym_state.lock:
            last_price = sym_state.last_price
            prev_close = sym_state.prev_close
            bid_price = sym_state.bid_price
            bid_size = sym_state.bid_size
            ask_price = sym_state.ask_price
            ask_size = sym_state.ask_size
            bars = list(sym_state.bars)
        today_low, today_high, today_open = _today_range(sym_state.daily_bars, bars)
        current_momentum = momentum_pct(sym_state)
        daily_momentum = (
            (last_price / today_open - 1.0) * 100.0
            if last_price is not None and today_open
            else None
        )
        day_lines = _day_lines_shown(sym_state.symbol)
        quote = _quote_html(
            last_price, prev_close, bid_price, bid_size, ask_price, ask_size,
            sym_state.symbol,
            today_low=today_low, today_high=today_high,
            current_momentum=current_momentum, daily_momentum=daily_momentum,
            net_gamma=_live_net_gamma(sym_state, last_price),
            day_lines=day_lines,
        )
        if not quote:
            continue
        clicked = quote_card(quote, key=f"live_quote_{sym_state.symbol}")
        if clicked in _DAY_LINE_KEYS:
            st.session_state[_DAY_LINES_STATE_KEY] = {
                **st.session_state.get(_DAY_LINES_STATE_KEY, {}),
                sym_state.symbol: day_lines ^ {clicked},
            }
            # The chart fragment only polls every CHART_POLL_SEC; rerun the
            # whole app so the line appears (or goes) straight away.
            st.rerun(scope="app")


# How far before the 09:30 open the live chart starts, unless pre-market is on.
CHART_LEAD_MIN = 5


def _chart_start(bars: list[dict], pre_market: bool) -> datetime:
    """Where the live chart's time axis starts, in UTC: CHART_LEAD_MIN before
    the open of the latest bar's ET trading day, or that day's midnight -- so
    every pre-market bar -- with `pre_market` on.

    From the bars' own date rather than the clock (or `SESSION_START`, fixed
    at import in UTC), so a chart left running overnight, or viewed in the
    evening, is still about the day its bars are from, and 09:25 is 09:25 ET
    in winter too.
    """
    last = pd.Timestamp(bars[-1]["t"])
    if last.tzinfo is None:
        last = last.tz_localize("UTC")
    day = last.tz_convert(market_hours.MARKET_TZ).normalize()
    if pre_market:
        start = day
    else:
        start = day + timedelta(
            hours=market_hours.MARKET_OPEN.hour, minutes=market_hours.MARKET_OPEN.minute,
        ) - timedelta(minutes=CHART_LEAD_MIN)
    return start.tz_convert("UTC").to_pydatetime()


@st.fragment(run_every=CHART_POLL_SEC)
def _chart_panel() -> None:
    state = _get_state()
    tracker = state.decision_tracker
    rendered = False
    for sym_state in state.iter_symbol_states():
        sym = sym_state.symbol
        with sym_state.lock:
            bars = list(sym_state.bars)
            # Only price-axis alerts (price/bid/ask/day high/low) can be drawn as
            # horizontal lines; volume/spread/portfolio alerts have no price level.
            price_alerts = [
                a for a in sym_state.alerts if a.get("field") in PRICE_AXIS_ALERT_FIELDS
            ]

        if not bars:
            continue
        rendered = True
        chart_start = _chart_start(bars, state.show_pre_market)
        if pd.Timestamp(bars[-1]["t"]) <= pd.Timestamp(chart_start):
            st.plotly_chart(
                empty_chart(
                    f"{sym}: pre-market — the chart starts at "
                    f"{pd.Timestamp(chart_start).tz_convert(market_hours.MARKET_TZ):%H:%M} ET "
                    "(Chart Settings → Pre-market shows it now)"
                ),
                width='stretch', key=f"live_chart_{sym}",
            )
            continue

        # Price levels at which armed tactics (standing conditional orders) execute.
        tactic_levels = tactic_price_levels(sym_state.tactics)
        decisions = tracker.trade_markers(symbol=sym) if tracker else None

        predicted_profile = (
            predicted_open_profile(sym_state, bars)
            if state.show_predicted_profile
            else None
        )
        overlays = model_overlays.live_overlays(
            sym_state, bars, state.model_overlay_keys
        )
        option_walls = _live_option_walls(sym_state, state.option_walls)
        day_range_lines = _day_range_lines(sym, sym_state.daily_bars, bars)
        net_gamma = (
            _live_gamma_series(sym_state, bars, state.timeframe)
            if state.show_net_gamma else None
        )

        fig = build_chart(
            bars,
            sym_state.news,
            sym_state.trades,
            sym,
            chart_start,
            ma_periods=state.ma_periods,
            show_fib=state.show_fib,
            show_7d_avg=state.show_7d_avg,
            show_28d_avg=state.show_28d_avg,
            show_1y_avg=state.show_1y_avg,
            mixture_distribution=state.mixture_distribution,
            mixture_max_components=state.mixture_max_components,
            predicted_profile=predicted_profile,
            mixture_fit_target=state.mixture_fit_target,
            daily_bars=sym_state.daily_bars,
            vwap_style=state.vwap_style,
            show_candle_body=state.show_candle_body,
            show_percentile_body=state.show_percentile_body,
            show_whiskers=state.show_whiskers,
            decisions=decisions,
            price_alerts=price_alerts,
            tactic_levels=tactic_levels,
            news_impacts=sym_state.news_impacts,
            fill_gaps=state.fill_gaps,
            model_overlays=overlays["items"],
            candle_patterns=_live_candle_patterns(state, bars),
            show_momentum=state.show_momentum,
            minute_momentum_profile=sym_state.minute_momentum_profile,
            minute_momentum_change_profile=sym_state.minute_momentum_change_profile,
            **_agent_momentum_kwargs(state, sym_state),
            volume_baseline=_volume_baseline(sym, bars, state),
            option_walls=option_walls,
            day_range_lines=day_range_lines,
            net_gamma=net_gamma,
        )
        st.plotly_chart(fig, width='stretch', key=f"live_chart_{sym}")
        for note in overlays["notes"]:
            st.caption(f":material/info: {sym} — {note}")
    if not rendered:
        st.plotly_chart(empty_chart(), width='stretch', key="live_chart_empty")


# The look-back drawn over the momentum panels when the selected agent decides
# on no momentum at all, in minutes.
FALLBACK_MOMENTUM_MIN = 5


def _agent_momentum(state, sym_state) -> "tuple[int, str]":
    """How many chart bars the selected agent reads momentum over, and why.

    * Apple Trader -- its momentum confirmation period, which both sides read
      the behaviour table over (the running agent's config while one runs,
      since that setting takes ▶ Start; else the form's). A legacy config
      without it: the take / fall look-back, `fall_bars`.
    * Orchestra -- the same, read off the pair this symbol's chart draws
      (`model_overlays.live_trader_view`); every pair shares the setting.
    * LLM personalities and Automatic -- an armed tactic or pending alert on
      `momentum_pct` compares against the close `TACTICS_MOMENTUM_WINDOW_MIN`
      minutes back.

    The rule trader counts bars of the stream they run on, which is the
    chart's timeframe; the tactic window is in minutes, so it is converted.
    Anything else -- including an Apple Trader with the take and the fall rule
    both off -- gets `FALLBACK_MOMENTUM_MIN` minutes.
    """
    bar_min = TF_MINUTES.get(getattr(state, "timeframe", "1Min"), 1)

    def minutes_to_bars(minutes: int) -> int:
        return max(1, round(minutes / bar_min))

    personality = getattr(state, "llm_personality", None)
    if personality in (APPLE_TRADER_KEY, ORCHESTRA_KEY):
        if personality == ORCHESTRA_KEY:
            # The racer this symbol's chart draws: they share these settings.
            config, _ = model_overlays.live_trader_view(state, sym_state.symbol)
        else:
            running = (getattr(state, "apple_trader_levels", None) or {}) if state.agent_running else {}
            config = running.get("config") or getattr(state, "apple_trader_config", None)
        name = _personality_label(personality).split(" (")[0]
        confirm = int(getattr(config, "momentum_confirmation_bars", 0) or 0)
        if confirm:
            return confirm, name
        if config is not None and (config.has_take or getattr(config, "max_fall_k", 0) > 0):
            return int(config.fall_bars), name
    elif _reads_momentum_pct(sym_state):
        return minutes_to_bars(TACTICS_MOMENTUM_WINDOW_MIN), "armed tactic"
    return minutes_to_bars(FALLBACK_MOMENTUM_MIN), f"{FALLBACK_MOMENTUM_MIN} min"


def _reads_momentum_pct(sym_state) -> bool:
    """Whether an armed tactic or a pending alert on this symbol waits on
    `momentum_pct`."""
    tactics = getattr(sym_state, "tactics", None)
    if tactics is not None and tactics.status == "armed":
        if any(c.field == "momentum_pct" for a in tactics.actions for c in a.conditions):
            return True
    return any(a.get("field") == "momentum_pct" for a in getattr(sym_state, "alerts", None) or [])


def _agent_momentum_kwargs(state, sym_state) -> dict:
    bars, label = _agent_momentum(state, sym_state)
    return {"agent_momentum_bars": bars, "agent_momentum_label": label}


def _live_chart_controls() -> None:
    state = _get_state()
    with st.expander("Chart Settings"):
        st.selectbox("Timeframe", TIMEFRAMES, index=0, key="live_timeframe")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Candle**")
            show_candle_body = st.checkbox("Open-Close", value=True, key="chart_candle_body")
            show_percentile_body = st.checkbox("20%-80%", value=False, key="chart_percentile_body")
            show_whiskers = st.checkbox("Whiskers", value=True, key="chart_whiskers")
            fill_gaps = st.checkbox(
                "Fill no-trade gaps",
                value=True,
                key="chart_fill_gaps",
                help="Draw flat zero-volume placeholder bars for feed minutes "
                "without any trade (common on IEX for thin symbols).",
            )
            show_vwap = st.checkbox("VWAP", value=False, key="chart_vwap")
            show_pre_market = st.checkbox(
                "Pre-market",
                value=False,
                key="chart_pre_market",
                help=f"Show the whole day from its first bar. Off, the chart starts "
                f"{CHART_LEAD_MIN} minutes before the 09:30 ET open, and every "
                "panel under it — momentum, net gamma — starts there too.",
            )
        with c2:
            st.markdown("**Overlays**")
            with _help_row(
                "Volume-weighted moving averages of the close over the last 5, "
                "15 or 60 bars.",
                icon_ratio=_HALF_WIDTH_ICON,
            ):
                vwma_selection = st.multiselect(
                    "VWMA",
                    ["VWMA(5)", "VWMA(15)", "VWMA(60)"],
                    default=[],
                    key="chart_vwma",
                    placeholder="VWMA",
                    label_visibility="collapsed",
                )
            with _help_row(
                "Flat lines at the volume-weighted average close over the last "
                "7 days, 28 days or year of daily bars.",
                icon_ratio=_HALF_WIDTH_ICON,
            ):
                avg_selection = st.multiselect(
                    "Average Lines",
                    ["7d Avg", "28d Avg", "1y Avg"],
                    default=[],
                    key="chart_avg_lines",
                    placeholder="Average lines",
                    label_visibility="collapsed",
                )
            with _help_row(
                "The strikes with the most call open interest (Call wall, likely "
                "resistance) and put open interest (Put wall, likely support) in "
                "the nearest expiry within 45 days, from yfinance and refreshed "
                "every minute: the same walls as the Put/Call Walls tab. A wall "
                "far outside today's range is a label in the chart's corner "
                "rather than a line, so it doesn't flatten the candles.",
                icon_ratio=_HALF_WIDTH_ICON,
            ):
                wall_selection = st.multiselect(
                    "Options walls",
                    list(_OPTION_WALL_OPTIONS),
                    default=[],
                    key="chart_option_walls",
                    placeholder="Options walls",
                    label_visibility="collapsed",
                )
            baseline_keys = list(VOLUME_BASELINE_WINDOWS)
            with _help_row(
                "What the volume panel compares today's bars against.\n\n"
                "- **Last trading week, mean + 1σ** (default) — a dark band behind "
                "the bars: for each bar's clock bucket, the mean + one standard "
                "deviation of its volume over the last five sessions, pooled with "
                "the buckets 5 minutes either side. Built from **the same source "
                "as the bar** — IEX bars against IEX history, SIP against SIP, "
                "yfinance against yfinance (Finnhub bars against SIP) — since IEX "
                "is only ~4% of the consolidated volume.\n"
                "- The other windows draw a dashed line at the window's mean "
                "volume per bar and a shaded backdrop at each clock minute's "
                "average, from yfinance's consolidated tape. \"This session\" "
                "has no backdrop, since its shape is the bars themselves.",
                icon_ratio=_HALF_WIDTH_ICON,
            ):
                volume_baseline_window = st.selectbox(
                    "Usual volume",
                    baseline_keys,
                    index=baseline_keys.index(DEFAULT_VOLUME_BASELINE),
                    format_func=lambda k: f"Usual volume: {VOLUME_BASELINE_WINDOWS[k]}",
                    key="chart_volume_baseline",
                    label_visibility="collapsed",
                )
            show_momentum = st.checkbox(
                "Momentum panel",
                value=True,
                key="chart_momentum",
                help="Two panels under the volume, one bar per chart bar, so "
                "they follow the timeframe (a bar a minute on 1Min, one per five "
                "minutes on 5Min):\n\n"
                "- **Momentum** — the close minus the previous close, in dollars. "
                "On 1Min bars the dark band at ± is the mean + 1σ of the absolute "
                "one-minute move at that time of day over the last five sessions "
                "(±5 minutes pooled), measured once a day on ▶ Start.\n"
                "- **Momentum Δ** — this bar's momentum minus the previous bar's: "
                "above zero the move is speeding up (or a fall is easing), below "
                "zero it is slowing (or a fall is steepening). Its dark band is the "
                "same mean + 1σ, of the absolute Δ.\n\n"
                "The semitransparent blue lines are the same two measures averaged "
                "per bar over the look-back the selected agent decides on — Apple "
                "Trader's momentum confirmation period or an LLM agent's armed "
                "momentum tactic — and over 5 "
                "minutes when it reads no momentum.\n\n"
                "Regular session only, and each day starts fresh.",
            )
            show_net_gamma = st.checkbox(
                "Net gamma panel",
                value=True,
                key="chart_net_gamma",
                help="A panel at the bottom, under Momentum Δ: the options "
                "market's net dealer gamma, in $M per 1% move, at each bar's "
                "close. Green where dealers are long gamma — their hedging sells "
                "rallies and buys dips, damping moves — red where they are short "
                "and their hedging amplifies them.\n\n"
                "From the nearest expiry within 45 days (yfinance, refreshed every "
                "minute — the same chain as the Put/Call Walls tab and the Net "
                "Gamma card). Each bar is priced once, at its close, with the "
                "chain the app had then, and keeps that value — across a reload "
                "or a restart too. Only the bar still forming moves. Bars that "
                "closed before the first chain arrived (a start mid-day) are "
                "priced with that first chain: open interest only changes "
                "overnight, so it is re-priced at their closes, holding its "
                "implied volatility.",
            )

        st.markdown("**Price Profile Fit**")
        with _help_row(
            "- **Gaussian / Cauchy mixture** — fits a mixture to the "
            "volume-at-price profile. Both can be drawn at once; **Components** "
            "sets how many.\n"
            "- **ML predicted profile** — where today's volume is predicted to "
            "trade (LevelsML density model: per-quantile LightGBM at the open, "
            "from daily-bar features). Needs the trained pack in ../Models and "
            "daily bars.\n"
            "- **Fibonacci levels** — retracement lines across the session's "
            "high-low range, drawn on the candles."
        ):
            profile_selection = st.multiselect(
                "Price profile fit",
                list(_PROFILE_FIT_OPTIONS),
                default=[],
                key="chart_profile_fit",
                placeholder="None",
                label_visibility="collapsed",
            )
        mixture_dists = [
            _PROFILE_FIT_OPTIONS[o] for o in profile_selection
            if _PROFILE_FIT_OPTIONS[o] in ("gaussian", "cauchy")
        ]
        show_fib = "Fibonacci levels" in profile_selection
        show_predicted = "ML predicted profile" in profile_selection
        max_components = 0
        fit_target_choice = "Live volume"
        if mixture_dists:
            max_components = st.slider(
                "Components", min_value=1, max_value=5, value=1,
                key="chart_mixture_components",
            )
            if show_predicted:
                fit_target_choice = st.selectbox(
                    "Fit to",
                    ["Live volume", "Predicted profile"],
                    index=0,
                    key="chart_mixture_fit_target",
                    help="Which profile the mixture is fitted to.",
                )

        overlay_keys = _model_overlay_controls(state)
        pattern_keys, fvg_min_size, fvg_hide_filled = _candle_pattern_controls()

        backfill_clicked = st.button(
            "⟲ Backfill missing bars",
            disabled=not (state.symbols and state.api_key),
            help="Re-fetch each symbol's session bars via REST and merge any that the "
            "stream missed, e.g. while it was stopped. Bars older than 15 minutes come "
            "from yfinance (regular session) or the history feed (pre/post-market); "
            "younger ones from IEX, replaced once they are 15 minutes old. The minute "
            "in progress is left to the live stream.",
        )
        if backfill_clicked:
            with st.spinner("Backfilling…"):
                notes = []
                for sym_state in state.iter_symbol_states():
                    try:
                        added, source = backfill_bars(
                            sym_state.symbol, state.api_key, state.api_secret,
                            state.history_feed_resolved or state.history_feed,
                            sym_state, state.timeframe,
                        )
                    except Exception as exc:
                        st.error(f"Backfill failed for {sym_state.symbol}: {exc}")
                    else:
                        notes.append(
                            f"{sym_state.symbol}: {added} bar(s) via {source}"
                            if added else f"{sym_state.symbol}: no missing bars"
                        )
                if notes:
                    st.caption(" · ".join(notes))

    state.ma_periods = _parse_ma_periods(vwma_selection)
    state.show_7d_avg, state.show_28d_avg, state.show_1y_avg = _parse_avg_flags(avg_selection)
    state.show_candle_body = show_candle_body
    state.show_percentile_body = show_percentile_body
    state.show_whiskers = show_whiskers
    state.show_momentum = show_momentum
    state.show_net_gamma = show_net_gamma
    state.show_pre_market = show_pre_market
    state.volume_baseline_window = volume_baseline_window
    state.fill_gaps = fill_gaps
    state.vwap_style = "dot" if show_vwap else "hide"
    state.show_fib = show_fib
    state.option_walls = [_OPTION_WALL_OPTIONS[o] for o in wall_selection]
    state.mixture_distribution = mixture_dists
    state.mixture_max_components = max_components
    state.show_predicted_profile = show_predicted
    state.mixture_fit_target = (
        "predicted"
        if show_predicted and fit_target_choice == "Predicted profile"
        else "live"
    )
    state.model_overlay_keys = overlay_keys
    state.candle_pattern_keys = pattern_keys
    state.fvg_min_size = fvg_min_size
    state.fvg_hide_filled = fvg_hide_filled



# `_help_row`'s icon column inside one of Chart Settings' two half-width columns.
_HALF_WIDTH_ICON = 0.16


def _help_row(help_text: str, icon_ratio: float = 0.08):
    """A column for a label-less widget, with its help icon to its right.

    A widget whose label is collapsed loses its own help icon along with the
    label, so the icon goes in a narrow column of its own beside it. Use as
    `with _help_row("..."):` around the widget. The icon column needs ~30px or
    the icon wraps below the widget's middle, so a row inside a half-width
    column passes a larger `icon_ratio` than a full-width one.
    """
    field, icon = st.columns([1, icon_ratio], gap="small", vertical_alignment="center")
    with icon:
        # A zero-width body: anything visible wraps the icon onto a second
        # line in a column this narrow, dropping it below the widget's middle.
        st.markdown("\u200b", help=help_text)
    return field


# Options walls multiselect: option label -> key in the walls analysis.
_OPTION_WALL_OPTIONS = {"Call wall": "call_wall", "Put wall": "put_wall"}

# Symbols whose options chain is being fetched for the live chart right now,
# and when each was last tried -- see `_live_option_walls`.
_option_fetch_lock = threading.Lock()
_option_fetch_running: set[str] = set()
_option_fetch_tried: dict[str, float] = {}


def _refresh_option_chain(sym_state: SymbolState) -> None:
    """Refresh `sym_state.options_chain` in a background thread, at most once
    per OPTIONS_POLL_SEC per symbol.

    The chart fragment re-runs every few seconds and a yfinance chain fetch
    takes a second or two, so fetching inline would stall the live chart. A
    failure is logged by `fetch_option_chain` and simply retried on the next
    interval; the chart keeps drawing the last chain it had.
    """
    sym = sym_state.symbol
    now = time.monotonic()
    with _option_fetch_lock:
        if sym in _option_fetch_running or now - _option_fetch_tried.get(sym, -1e9) < OPTIONS_POLL_SEC:
            return
        _option_fetch_running.add(sym)
        _option_fetch_tried[sym] = now

    def fetch() -> None:
        try:
            with sym_state.lock:
                spot = sym_state.last_price
            data = fetch_options_walls_data(sym, spot=spot)
        except Exception:
            pass
        else:
            with sym_state.lock:
                sym_state.options_chain = data
        finally:
            with _option_fetch_lock:
                _option_fetch_running.discard(sym)

    threading.Thread(target=fetch, name=f"option-walls-{sym}", daemon=True).start()


def _live_option_walls(sym_state: SymbolState, keys: "list[str]") -> "dict | None":
    """The selected walls ({"call_wall": ..., "put_wall": ...}) from the symbol's
    latest options chain, or None when none are selected or no chain has
    arrived yet. Also kicks off a refresh of that chain."""
    if not keys:
        return None
    _refresh_option_chain(sym_state)
    with sym_state.lock:
        data = sym_state.options_chain
    if not data or not data.get("strikes"):
        return None
    analysis = get_put_call_walls_and_gamma(
        strikes=data["strikes"],
        calls_oi=data["calls_oi"],
        puts_oi=data["puts_oi"],
        calls_gamma_exposure=data["calls_gamma_exposure"],
        puts_gamma_exposure=data["puts_gamma_exposure"],
        spot=data["spot"],
    )
    return {k: analysis.get(k) for k in keys}


# Price Profile Fit multiselect: option label -> mixture distribution, or
# "predicted" for the ML profile curve, or "fib" for the Fibonacci levels.
_PROFILE_FIT_OPTIONS = {
    "Gaussian mixture": "gaussian",
    "Cauchy mixture": "cauchy",
    "ML predicted profile": "predicted",
    "Fibonacci levels": "fib",
}


def _model_overlay_controls(state: AppState) -> "list[str]":
    """Which model predictions the price chart draws.

    Offered per the symbols being streamed rather than globally: the day-range
    model was fitted on a fixed set of tickers, and an overlay no chart here
    could show is a checkbox that does nothing. The profile model claims to
    transfer, so it is always on the list.

    The widget is driven by `key=` alone rather than by `default=`. A default
    that changes with the selection re-creates the widget on the next run and
    loses it, and the streamed symbols can change under a stored selection --
    so the session value is seeded once and then pruned to what is still on
    offer.
    """
    available: list[str] = []
    for sym in state.symbols or []:
        for key in model_overlays.keys_for(sym):
            if key not in available:
                available.append(key)
    if not available:
        available = model_overlays.keys()

    stored = [k for k in st.session_state.get("model_overlay_keys", []) if k in available]
    st.session_state["model_overlay_keys"] = stored

    st.markdown("**Model Predictions**")
    with _help_row(
        "Draws what the trained models predict for this session:\n"
        "- **TimeToChange3**, **HighLow**, **HighLow2**, **LevelsML** — price ranges: "
        "horizontal lines, in the candles and in the profile beside them.\n"
        "- **IntradayVolatility** (alone or × TimeToChange3) — time-of-day "
        "ranges: a shaded envelope, widest at the open and narrowing through "
        "midday.\n"
        "- **Apple Trader** — buy/sell levels, not a forecast: the two orders "
        "the agent configured below would rest, following whatever settings "
        "that form is holding."
    ):
        selected = st.multiselect(
            "Model predictions",
            available,
            format_func=model_overlays.name,
            key="model_overlay_keys",
            placeholder="None",
            label_visibility="collapsed",
        )
    for key in selected:
        overlay = model_overlays.get(key)
        if overlay:
            st.caption(f"{overlay.name} — {overlay.summary}")
    return selected


def _candle_pattern_controls() -> "tuple[list[str], float, bool]":
    """Which candle patterns the price chart draws, and the FVG filters.

    Returns `(keys, fvg_min_size, fvg_hide_filled)`. The filters only show
    once a fair value gap is selected; the defaults stand otherwise.
    """
    st.markdown("**Candle Patterns**")
    with _help_row(
        "Shapes read off the candles themselves — no model, no forecast.\n"
        "- **Fair Value Gap (FVG)** — three candles where candle 1's wick and "
        "candle 3's wick don't overlap. The gap between them is boxed from "
        "candle 1: green for a gap up, red for a gap down, running right until "
        "price trades through it, then faded. Hover a box's left edge for its "
        "range and when it formed and filled. Regular session only."
    ):
        selected = st.multiselect(
            "Candle patterns",
            candle_patterns.keys(),
            format_func=candle_patterns.label,
            key="candle_pattern_keys",
            placeholder="None",
            label_visibility="collapsed",
        )
    min_size = candle_patterns.DEFAULT_FVG_MIN_SIZE
    hide_filled = False
    if candle_patterns.FVG_KEY in selected:
        c1, c2 = st.columns(2, vertical_alignment="bottom")
        with c1:
            min_size = st.slider(
                "Min FVG size (× avg bar range)",
                min_value=0.0, max_value=2.0, step=0.25,
                value=candle_patterns.DEFAULT_FVG_MIN_SIZE,
                key="fvg_min_size",
                help="Drops gaps narrower than this many times the average "
                "high-low range of the 14 bars before them. 1-minute bars leave "
                "100+ tiny gaps a session, most filled within a few bars; 0 "
                "draws every one.",
            )
        with c2:
            hide_filled = st.checkbox(
                "Hide filled gaps",
                key="fvg_hide_filled",
                help="Only draw gaps price has not yet traded through.",
            )
    return selected, float(min_size), bool(hide_filled)


def _live_candle_patterns(state: AppState, bars: "list[dict]") -> "list[dict]":
    """The selected candle patterns over one symbol's live bars."""
    return candle_patterns.compute(
        state.candle_pattern_keys, bars,
        min_size=state.fvg_min_size, hide_filled=state.fvg_hide_filled,
    )


def _news_model_credentials(state: AppState) -> "tuple[str, str, str]":
    """Alpaca key, secret and bar feed for the news-impact model's bar fetches."""
    return (
        state.api_key or os.getenv("ALPACA_API_KEY", ""),
        state.api_secret or os.getenv("ALPACA_SECRET", ""),
        state.history_feed_resolved or state.history_feed,
    )


def _refresh_model_impacts(state: AppState) -> None:
    """Keep model-scored symbols' badges current; drop them where the model is off.

    Runs on every news-panel poll. A refresh only starts when some article has
    no settled verdict (a new one from the stream, or one waiting for its
    bars), and it runs on a background thread: the first one of the day
    downloads weeks of minute bars.
    """
    key, secret, feed = _news_model_credentials(state)
    for sym_state in state.iter_symbol_states():
        if not sym_state.news:
            continue
        if newsimpact_model.uses_model(sym_state.symbol, state.news_impact_method):
            newsimpact_model.launch_refresh(sym_state, key, secret, feed)
        else:
            newsimpact_model.clear_model_impacts(sym_state)


def _news_analysis_controls(symbols: list[str]) -> None:
    state = _get_state()
    methods = list(newsimpact_model.IMPACT_METHODS)
    method = st.selectbox(
        "Impact estimate",
        methods,
        index=methods.index(state.news_impact_method) if state.news_impact_method in methods else 0,
        format_func=newsimpact_model.IMPACT_METHODS.get,
        key="news_impact_method_select",
        help=(
            "The news-impact model reads the price's momentum over the 15 minutes before an "
            "intraday release and estimates the momentum state over the 15 minutes after it; "
            "it does not read the article's text. Releases outside regular hours (or in the "
            "first 15 minutes) stay unknown. It is used for symbols that have their own model "
            "and scores new articles automatically; every other symbol uses the LLM."
        ),
    )
    state.news_impact_method = method
    states = [s for s in state.iter_symbol_states() if s.symbol in symbols] or list(
        state.iter_symbol_states()
    )
    model_states = [s for s in states if newsimpact_model.uses_model(s.symbol, method)]
    llm_states = [s for s in states if s not in model_states]
    if model_states:
        caption = f"Model: {', '.join(s.symbol for s in model_states)}"
        if llm_states:
            caption += f" · LLM: {', '.join(s.symbol for s in llm_states)}"
        st.caption(caption)
        for sym_state in model_states:
            if sym_state.news_impact_error:
                st.caption(f"⚠️ News-impact model for {sym_state.symbol}: {sym_state.news_impact_error}")

    provider = state.news_llm_provider
    llm_key = ""
    env_var = ENV_KEYS[provider]
    if llm_states or not states:
        provider = st.selectbox(
            "Provider",
            PROVIDERS,
            index=PROVIDERS.index(state.news_llm_provider),
            key="news_llm_provider_select",
            help=f"Model used: {', '.join(f'{p}={m}' for p, m in DEFAULT_NEWS_MODELS.items())}",
        )
        state.news_llm_provider = provider
        env_var = ENV_KEYS[provider]
        llm_key = os.getenv(env_var, "")
        if not llm_key:
            st.caption(f"⚠️ {env_var} is not set.")

    analyze_clicked = st.button("🔍 Analyze News", key="news_analyze_btn")
    if analyze_clicked:
        if not any(s.news for s in states):
            st.warning("No news loaded yet. Start the Live stream for the symbols first.")
            return
        key, secret, feed = _news_model_credentials(state)
        for sym_state in model_states:
            if sym_state.news:
                newsimpact_model.launch_refresh(sym_state, key, secret, feed, force=True)
        llm_targets = [s for s in llm_states if s.news]
        if llm_targets and not llm_key:
            st.error(f"{env_var} is not set; news analysis needs an LLM key.")
        elif llm_targets:
            with st.spinner("Scoring news impact…"):
                for sym_state in llm_targets:
                    try:
                        impacts = score_news_impacts(
                            sym_state.symbol, sym_state.news, provider, llm_key
                        )
                    except Exception as exc:
                        st.error(f"News impact scoring failed for {sym_state.symbol}: {exc}")
                        continue
                    with sym_state.lock:
                        sym_state.news_impacts = impacts
                        sym_state.news_impact_details = {}


@st.fragment(run_every=CHART_POLL_SEC)
def _news_panel(symbols: list[str]) -> None:
    state = _get_state()
    if state.news_status not in ("Idle", state.status):
        st.caption(f"News: {state.news_status}")
    _refresh_model_impacts(state)
    _news_analysis_controls(symbols)
    # The impact method and provider are drawn in this fragment, whose reruns
    # skip the full run's save.
    last_setup.remember()
    rendered = False
    for sym_state in state.iter_symbol_states():
        st.html(
            _news_html(
                sym_state.news,
                sym_state.symbol,
                sym_state.news_impacts,
                sym_state.news_impact_details,
            )
        )
        rendered = True
    if not rendered:
        st.info("Start the Live stream to load news for your symbols.")


def _live_panel() -> None:
    _live_chart_controls()
    _price_ticker()
    _chart_panel()


@st.fragment(run_every=CHART_POLL_SEC)
def _technical_analysis_panel(symbols: list[str]) -> None:
    """Visualizes the three human-readable reads from `technical_analysis` for
    each symbol: the daily trend regime, intraday momentum, and (once, shared)
    the broad market environment."""
    state = _get_state()

    states = [s for s in state.iter_symbol_states() if s.symbol in symbols] or list(
        state.iter_symbol_states()
    )
    states = [s for s in states if s.bars]
    if not states:
        st.info("Start the Live stream for your symbols first so there's data to analyze.")
        return

    # The broad-market read is symbol-independent -- compute it once.
    try:
        market_series = fetch_market_indicators()
        market = analyze_market(market_series.get("vix"), market_series.get("spy"), market_series.get("vix3m"))
    except Exception as exc:
        market = {"note": f"market indicators unavailable: {exc}"}

    for sym_state in states:
        sym = sym_state.symbol
        with sym_state.lock:
            bars = list(sym_state.bars)
        daily_bars = sym_state.daily_bars

        st.subheader(sym)
        trend = analyze_trend(daily_bars if daily_bars else bars)
        intraday = analyze_intraday(bars)

        st.plotly_chart(
            build_analysis_gauges(trend, intraday, market),
            width='stretch',
            key=f"analysis_gauges_{sym}",
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**📈 Trend (daily)**")
            if "note" in trend:
                st.caption(trend["note"])
            else:
                st.metric(
                    "Regime",
                    f"{trend['regime'].capitalize()} ({trend['trend_strength']})",
                    f"{trend['pct_change_over_period']:+.1f}%",
                )
                st.caption(trend["summary"])
        with c2:
            st.markdown("**⚡ Intraday Momentum**")
            if "note" in intraday:
                st.caption(intraday["note"])
            else:
                st.metric("Window change", f"{intraday['pct_change_in_window']:+.2f}%", intraday["momentum_pattern"])
                st.caption(intraday["summary"])
        with c3:
            st.markdown("**🌍 Market Environment**")
            if "note" in market:
                st.caption(market["note"])
            else:
                st.metric("Risk environment", market["risk_environment"].capitalize(), f"score {market['risk_score']:+d}")
                st.caption(market["summary"])
                for insight in market.get("insights", []):
                    st.markdown(f"- {insight}")
        st.divider()


def _record_wall_snapshot(sym_state: SymbolState, call_wall: float, put_wall: float) -> list[dict]:
    """Append a {call_wall, put_wall} snapshot if it differs from the last one, so the
    agent's trend read (rising/falling walls) reflects real shifts, not poll noise."""
    with sym_state.lock:
        history = list(sym_state.options_wall_history)
        last = history[-1] if history else None
        if last is None or last.get("call_wall") != call_wall or last.get("put_wall") != put_wall:
            history.append(
                {
                    "ts": pd.Timestamp.now(tz="UTC").isoformat(),
                    "call_wall": call_wall,
                    "put_wall": put_wall,
                }
            )
            history = history[-OPTIONS_WALL_HISTORY_MAXLEN:]
            sym_state.options_wall_history = history
        return history


@st.fragment(run_every=OPTIONS_POLL_SEC)
def _options_walls_panel(symbols: list[str]) -> None:
    """Independently fetches/refreshes each symbol's options chain (cached, on its
    own poll loop -- never triggered by the agent) and renders the Call Wall /
    Put Wall / gamma read. The agent's get_put_call_walls tool only reads whatever
    this last stored on the SymbolState."""
    state = _get_state()
    if not symbols:
        st.plotly_chart(empty_chart("Enter symbols in the sidebar"), width='stretch')
        return

    for sym in symbols:
        sym_state = state.sym(sym)
        st.subheader(sym)
        live_spot = None
        if sym_state is not None:
            with sym_state.lock:
                live_spot = sym_state.last_price

        try:
            data = fetch_options_walls_data(sym, spot=live_spot)
        except Exception as exc:
            data = None
            if sym_state is not None:
                with sym_state.lock:
                    data = sym_state.options_chain
            if not data:
                st.error(f"Failed to fetch options chain for {sym}: {exc}")
                continue
            st.warning(f"Using last successful options fetch for {sym} -- refresh failed: {exc}")
        else:
            if sym_state is not None:
                with sym_state.lock:
                    sym_state.options_chain = data

        prior_history: list[dict] = []
        if sym_state is not None:
            with sym_state.lock:
                prior_history = list(sym_state.options_wall_history)
        analysis = get_put_call_walls_and_gamma(
            strikes=data["strikes"],
            calls_oi=data["calls_oi"],
            puts_oi=data["puts_oi"],
            calls_gamma_exposure=data["calls_gamma_exposure"],
            puts_gamma_exposure=data["puts_gamma_exposure"],
            spot=data["spot"],
            wall_history=prior_history,
        )
        if sym_state is not None:
            _record_wall_snapshot(sym_state, analysis["call_wall"], analysis["put_wall"])

        st.caption(f"Expiry {data['expiry']} · fetched {data['fetched_at']}")
        fig = build_gamma_chart(data, analysis, sym)
        st.plotly_chart(fig, width='stretch', key=f"gamma_chart_{sym}")

        c1, c2, c3 = st.columns(3)
        c1.metric("Call Wall (resistance)", f"${analysis['call_wall']:.2f}", analysis["call_wall_trend"] or "")
        c2.metric("Put Wall (support)", f"${analysis['put_wall']:.2f}", analysis["put_wall_trend"] or "")
        c3.metric("Net gamma regime", analysis["gamma_regime"].split(" ")[0].capitalize())

        st.caption(analysis["summary"])
        for insight in analysis["insights"]:
            st.markdown(f"- {insight}")
        st.divider()


def _agent_entry_style(entry: dict) -> tuple[str, str, str]:
    """Return (icon, accent_color, label) for an agent log entry."""
    etype = entry.get("type")
    symbol = entry.get("symbol")
    suffix = f" {symbol}" if symbol else ""
    if etype == "decision":
        action = entry.get("action")
        if action == "buy":
            return "🟢", PALETTE["up"], f"BUY{suffix}"
        if action == "sell":
            return "🔴", PALETTE["down"], f"SELL{suffix}"
        if action == "alert":
            return "⏰", PALETTE["accent"], "ALERT"
        return "💤", PALETTE["muted"], "SLEEP"
    if etype == "tactics_set":
        if entry.get("cancelled") is not None:
            return "🎯", PALETTE["muted"], f"TACTICS CANCELLED{suffix}"
        return "🎯", PALETTE["orange"], f"TACTICS SET{suffix}"
    if etype == "tactics_execution":
        action = entry.get("action")
        if action == "buy":
            return "🎯", PALETTE["up"], f"TACTICS → BUY{suffix}"
        if action == "sell":
            return "🎯", PALETTE["down"], f"TACTICS → SELL{suffix}"
        return "🎯", PALETTE["orange"], f"TACTICS{suffix}"
    if etype == "regime_select":
        return "🤖", PALETTE["orange"], "REGIME → STRATEGY"
    if etype == "stand_down":
        return "🛑", PALETTE["orange"], "STAND DOWN"
    if etype == "tool_call":
        return "🛠️", PALETTE["accent"], str(entry.get("name", "tool"))
    if etype == "analysis":
        return "🧠", PALETTE["text"], "analysis"
    if etype == "cycle_start":
        return "🔄", PALETTE["accent"], "cycle start"
    if etype == "news_alert":
        return "📰", PALETTE["accent"], "NEWS ALERT"
    if etype == "error":
        return "⚠️", PALETTE["down"], "error"
    return "ℹ️", PALETTE["muted"], "status"


def _kv_row_html(data: dict) -> str:
    """Render a tool call's args/result dict as compact monospace key=value
    chips instead of a raw (and often mid-token truncated) JSON blob."""
    pairs = " &nbsp;·&nbsp; ".join(
        f"<span style='color:{PALETTE['muted']}'>{html.escape(k)}</span>="
        f"<span>{html.escape(v)}</span>"
        for k, v in format_tool_kv(data)
    )
    return f"<span style='font-family:monospace;font-size:11px'>{pairs}</span>"


def _agent_entry_body(entry: dict) -> str:
    etype = entry.get("type")
    if etype == "decision":
        price = entry.get("price")
        price_str = f"${price:,.4f}" if price is not None else "—"
        qty = entry.get("quantity") or 0
        regime = html.escape(str(entry.get("regime", "unknown")))
        reasoning = html.escape(entry.get("reasoning", ""))
        extra = ""
        if entry.get("action") == "alert" and entry.get("alerts"):
            levels = " or ".join(
                f"<b>{html.escape(format_alert(a))}</b>" for a in entry["alerts"]
            )
            extra = f" · Wake when {levels}"
        return (
            f"<div>Regime: <b>{regime}</b> · Qty: <b>{qty:.2f}</b> · Price: <b>{price_str}</b>{extra}</div>"
            f"<div style='margin-top:4px;color:{PALETTE['muted']}'>{reasoning}</div>"
        )
    if etype == "tactics_set":
        cancelled = entry.get("cancelled")
        reasoning = html.escape(entry.get("reasoning", ""))
        if cancelled is not None:
            what = " · ".join(html.escape(t) for t in cancelled) or "none armed"
            return (
                f"<div>Cancelled: {what}</div>"
                f"<div style='margin-top:4px;color:{PALETTE['muted']}'>{reasoning}</div>"
            )
        rows = "".join(f"<div>🎯 <b>{html.escape(t)}</b></div>" for t in entry.get("tactics") or [])
        replaced = entry.get("replaced") or []
        replaced_html = (
            f"<div style='color:{PALETTE['muted']}'>replaced: {' · '.join(html.escape(t) for t in replaced)}</div>"
            if replaced
            else ""
        )
        return f"{rows}{replaced_html}<div style='margin-top:4px;color:{PALETTE['muted']}'>{reasoning}</div>"
    if etype == "tactics_execution":
        price = entry.get("price")
        price_str = f"${price:,.4f}" if price is not None else "—"
        qty = entry.get("quantity") or 0
        status = html.escape(str(entry.get("status", "")))
        tactic = html.escape(entry.get("tactic", ""))
        triggered = html.escape(entry.get("triggered_by", ""))
        error = entry.get("error")
        error_html = (
            f"<div style='color:{PALETTE['down']}'>{html.escape(str(error))}</div>" if error else ""
        )
        return (
            f"<div>Executed <b>{tactic}</b> · Status: <b>{status}</b> · Qty: <b>{qty:.2f}</b> · Price: <b>{price_str}</b></div>"
            f"{error_html}"
            f"<div style='margin-top:4px;color:{PALETTE['muted']}'>Triggered by {triggered}</div>"
        )
    if etype == "tool_call":
        args = entry.get("args") or {}
        result = entry.get("result") or {}
        parts = []
        if args:
            parts.append(f"<div style='margin-bottom:2px'>{_kv_row_html(args)}</div>")
        if "error" in result:
            parts.append(
                f"<div style='color:{PALETTE['down']}'>⚠ {html.escape(str(result['error']))}</div>"
            )
        elif set(result.keys()) <= {"note"} and result.get("note"):
            parts.append(
                f"<div style='color:{PALETTE['muted']};font-style:italic'>{html.escape(str(result['note']))}</div>"
            )
        elif result:
            parts.append(f"<div>{_kv_row_html(result)}</div>")
        return "".join(parts)
    if etype == "regime_select":
        label = html.escape(str(entry.get("label", entry.get("strategy", ""))))
        regime = html.escape(str(entry.get("regime", "unknown")))
        reasoning = html.escape(entry.get("reasoning", ""))
        return (
            f"<div>Activated <b>{label}</b> · Regime: <b>{regime}</b></div>"
            f"<div style='margin-top:4px;color:{PALETTE['muted']}'>{reasoning}</div>"
        )
    if etype == "stand_down":
        label = html.escape(_personality_label(str(entry.get("personality", ""))))
        quiet = entry.get("expected_quiet_minutes")
        quiet_str = f" · ~{quiet:g} min quiet expected" if isinstance(quiet, (int, float)) else ""
        reasoning = html.escape(entry.get("reasoning", ""))
        return (
            f"<div><b>{label}</b> relinquished control{quiet_str}</div>"
            f"<div style='margin-top:4px;color:{PALETTE['muted']}'>{reasoning}</div>"
        )
    if etype in ("analysis", "error", "status", "cycle_start", "news_alert"):
        return f"<div>{html.escape(entry.get('text', ''))}</div>"
    return ""


def _agent_log_html(log: list[dict]) -> str:
    if not log:
        return f"<p style='color:{PALETTE['muted']};padding:12px'>No agent activity yet.</p>"
    cards = []
    for entry in reversed(log):
        icon, color, label = _agent_entry_style(entry)
        if entry.get("racer"):
            # An Orchestra line: which (ticker, model) pair wrote it.
            label = f"{label} · {entry['racer']}"
        try:
            ts_fmt = pd.to_datetime(entry.get("ts", "")).strftime("%H:%M:%S")
        except Exception:
            ts_fmt = str(entry.get("ts", ""))
        cards.append(
            f"""
        <div style="background:{PALETTE['panel']}; border-radius:8px; padding:10px 14px;
                    border-left:3px solid {color}; border-top:1px solid {PALETTE['grid']};
                    border-right:1px solid {PALETTE['grid']}; border-bottom:1px solid {PALETTE['grid']};
                    margin-bottom:8px; font-size:12px; color:{PALETTE['text']};">
          <div style="display:flex; justify-content:space-between; color:{PALETTE['muted']}; font-size:11px; margin-bottom:4px;">
            <span>{icon} <b style="color:{color}">{html.escape(label)}</b></span>
            <span>{ts_fmt}</span>
          </div>
          {_agent_entry_body(entry)}
        </div>"""
        )
    return (
        "<div style='font-family:Inter,sans-serif; max-height:420px; overflow-y:auto;'>"
        f"{''.join(cards)}</div>"
    )


def _current_tactics_html(state: AppState) -> str:
    """The 'current tactics' card for the Agent tab: every armed conditional
    action (across all symbols) with each plan's reasoning and arming time, or
    an explicit nothing-armed note while the agent runs (idle agent renders
    nothing)."""
    armed_blocks: list[str] = []
    for sym_state in state.iter_symbol_states():
        tactics = sym_state.tactics
        armed = tactics_summaries(tactics)
        if not armed:
            continue
        try:
            since = f" · armed {pd.to_datetime(tactics.ts).strftime('%H:%M:%S')}" if tactics.ts else ""
        except Exception:
            since = ""
        items = "".join(f"<div>🎯 <b>{html.escape(t)}</b></div>" for t in armed)
        reasoning = (
            f"<div style='margin-top:4px;color:{PALETTE['muted']}'>{html.escape(tactics.reasoning)}</div>"
            if tactics.reasoning
            else ""
        )
        armed_blocks.append(
            f"<div style='margin-bottom:6px'>"
            f"<div style='color:{PALETTE['orange']};font-size:11px;margin-bottom:4px'>"
            f"{html.escape(sym_state.symbol)}{since}</div>"
            f"{items}{reasoning}</div>"
        )

    if not armed_blocks:
        if not state.agent_running:
            return ""
        return (
            f"<div style='background:{PALETTE['panel']};border:1px dashed {PALETTE['grid']};"
            "border-radius:8px;padding:8px 14px;margin-bottom:8px;font-size:12px;"
            f"color:{PALETTE['muted']}'>🎯 No tactics armed — the agent has not set "
            "buy/sell conditions on any symbol; it will act only when woken "
            "(alert, news, or timer).</div>"
        )
    return (
        f"<div style='background:{PALETTE['panel']};border:1px solid {PALETTE['orange']};"
        "border-radius:8px;padding:8px 14px;margin-bottom:8px;font-size:12px;"
        f"color:{PALETTE['text']}'>"
        f"<div style='color:{PALETTE['orange']};font-size:11px;margin-bottom:4px'>"
        f"ARMED TACTICS — execute automatically, then wake the agent</div>"
        f"{''.join(armed_blocks)}</div>"
    )


@st.fragment(run_every=AGENT_LOG_POLL_SEC)
def _trade_sound_fragment() -> None:
    """Poll the ledger and chime once per newly filled trade.

    On its own fragment because the trade happens on a background thread that
    the Streamlit script never waits for: the only way the page learns about a
    fill is by looking. It renders nothing visible, and it is mounted from the
    Agent tab — which Streamlit executes even when another tab is on screen, so
    the cue still fires for a user who has navigated away.
    """
    state = _get_state()
    tracker = state.decision_tracker
    if tracker is None or state.trade_sound_volume <= 0:
        # Forget the position in the ledger while sound is off, so turning it
        # back on adopts the trades that happened meanwhile instead of firing a
        # chime for each of them.
        st.session_state.pop("trade_sound_seen", None)
        return
    cue, seen = next_trade_cue(
        tracker.snapshot()["decisions"], st.session_state.get("trade_sound_seen")
    )
    st.session_state["trade_sound_seen"] = seen
    play_trade_sound(cue, volume=state.trade_sound_volume)


@st.fragment(run_every=AGENT_LOG_POLL_SEC)
def _agent_status_line() -> None:
    """The one-line run status: running or idle, the venue, the symbols, and
    what the agent itself is doing. Polled, because the last of those changes
    on the agent's thread -- the session opening, a buy, a stand-down -- and a
    run that ends on its own (a stop-out) has to stop saying "running"."""
    state = _get_state()
    status = "🟢 running" if state.agent_running else "⚪ idle"
    watching = (
        f" — watching {', '.join(state.symbols)}"
        if state.agent_running and state.symbols
        else ""
    )
    # Where orders go belongs in the status line rather than in a banner of its
    # own: this line is on screen whether the agent is running, idle or stopped,
    # and "is this moving real money" is not a question that stops mattering the
    # moment a run ends. Before the first Start nothing has been resolved yet,
    # so the line says that instead of naming a venue it has not chosen.
    if state.trading_mode_requested:
        venue = f" · {_venue_badge(state.trading_mode)}"
    else:
        venue = " · no venue resolved yet — press ▶ Start"
    activity = state.agent_activity if state.agent_running else None
    doing = f" · {activity[0]} {activity[1]}" if activity else ""
    st.caption(f"Status: {status}{venue}{watching}{doing}")


@st.fragment(run_every=AGENT_LOG_POLL_SEC)
def _agent_identity_panel() -> None:
    """Avatar card for the personality currently in charge. Under Automatic the
    face shown is the strategy Automatic activated, not Automatic itself; while
    it is still classifying the regime (or idle) the orchestrator's own avatar
    shows. Polled so the card follows Automatic's strategy switches live."""
    state = _get_state()
    selected = state.llm_personality
    display_key = selected
    note = ""
    assignments: dict = {}
    if selected == AUTOMATIC_KEY and state.agent_running:
        assignments = state.automatic_assignments or {}
        active = state.automatic_active_strategy
        if active:
            display_key = active
            regime = f" — {state.automatic_regime} market" if state.automatic_regime else ""
            distinct = {a.get("strategy") for a in assignments.values()}
            note = (
                # The card can only wear one face; when the orchestrator is
                # running several strategies at once, say so rather than
                # letting the dominant one stand for the whole basket.
                f"🤖 most of the basket — {len(distinct)} strategies running{regime}"
                if len(distinct) > 1
                else f"🤖 picked by Automatic{regime}"
            )
        else:
            note = "🤖 Automatic is assessing each ticker…"
    avatar = _avatar_data_uri(display_key)
    img = (
        f"<img src='{avatar}' alt='' style='width:56px;height:56px;border-radius:50%;flex:none'/>"
        if avatar
        else ""
    )
    note_html = (
        f"<div style='color:{PALETTE['muted']};font-size:0.85rem'>{html.escape(note)}</div>"
        if note
        else ""
    )
    st.html(
        f"<div style='display:flex;align-items:center;gap:14px;background:{PALETTE['panel']};"
        f"border:1px solid {PALETTE['grid']};border-radius:12px;padding:10px 16px;margin:4px 0'>"
        f"{img}"
        f"<div>"
        f"<div style='color:{PALETTE['text']};font-weight:600;font-size:1.05rem'>"
        f"{html.escape(_personality_label(display_key))}</div>"
        f"{note_html}"
        f"</div></div>"
    )
    # Per-ticker assignments, when the orchestrator split the basket. One line
    # each, because "which strategy is trading my TSLA" has no answer in the
    # single-avatar card above once the strategies differ.
    if len(assignments) > 1:
        rows = " · ".join(
            f"<b style='color:{PALETTE['accent']}'>{html.escape(sym)}</b> "
            f"{html.escape(_personality_label(entry.get('strategy', '')))}"
            f"<span style='color:{PALETTE['muted']}'> ({html.escape(str(entry.get('regime') or '—'))})</span>"
            for sym, entry in sorted(assignments.items())
        )
        st.html(
            f"<div style='background:{PALETTE['panel']};border:1px solid {PALETTE['grid']};"
            f"border-radius:10px;padding:8px 16px;margin:0 0 6px;font-size:0.85rem;"
            f"color:{PALETTE['text']}'>{rows}</div>"
        )


# Which account the money figures on this page belong to. Spelled out rather
# than left as a bare "Portfolio value": paper and live are two separate Alpaca
# accounts with separate balances, and a number that does not say which one it
# came from is a number the user has to guess about.
_ACCOUNT_SUFFIX: dict[str, str] = {
    "alpaca_paper": " (Alpaca paper)",
    "alpaca_live": " (Alpaca LIVE)",
}


def _portfolio_value_label(state: AppState) -> str:
    return "Portfolio value" + _ACCOUNT_SUFFIX.get(state.trading_mode, "")


def _starting_value_label(state: AppState) -> str:
    """On a real venue this is the account's value when the run started, not a
    budget anyone chose -- calling it a budget there invites the reading that
    the app is trading some carved-out slice of the account."""
    return "Starting budget" if state.trading_mode == "local" else "Value at start"


# One badge per venue, used everywhere the app has to say where orders are
# going. Short enough to sit inside the status line, which is the only place
# guaranteed to be on screen whether the agent is running, idle or stopped.
_VENUE_BADGE: dict[str, str] = {
    "local": "💻 local simulation",
    "alpaca_paper": "📝 Alpaca paper",
    "alpaca_live": "🔴 Alpaca LIVE",
}


def _venue_badge(mode: str) -> str:
    return _VENUE_BADGE.get(mode, mode)


def _cash_label(state: AppState) -> str:
    if state.trading_mode == "alpaca_live":
        return "Alpaca LIVE cash"
    if state.trading_mode == "alpaca_paper":
        return "Alpaca paper cash"
    return "Paper cash"


@st.fragment(run_every=AGENT_LOG_POLL_SEC)
def _agent_log_panel() -> None:
    state = _get_state()
    tracker = state.decision_tracker
    if tracker:
        snap = tracker.snapshot()
        positions = {s: q for s, q in snap["positions"].items() if q}
        c1, c2, c3 = st.columns(3)
        c1.metric(_cash_label(state), f"${snap['cash']:,.2f}")
        c2.metric(
            "Positions",
            " · ".join(f"{s} {q:.2f} sh" for s, q in positions.items()) if positions else "flat",
        )
        c3.metric("Decisions", len(snap["decisions"]))
    tactics_html = _current_tactics_html(state)
    if tactics_html:
        st.html(tactics_html)
    _orchestra_board(state)
    with state.lock:
        log = list(state.agent_log)
    st.html(_agent_log_html(log[-50:]))


def _orchestra_board(state: AppState) -> None:
    """Where every pair of today's Orchestra stands (`AppState.orchestra`):
    its status and how far its last close is above its buy level, counted in
    its own level unit so pairs at different prices compare."""
    race = getattr(state, "orchestra", None) or {}
    board = race.get("board") or []
    if not board or state.llm_personality != ORCHESTRA_KEY:
        return
    holder = race.get("holder")
    title = (
        f"🎼 Orchestra — following **{(race.get('labels') or {}).get(holder, holder)}**"
        if holder else "🎼 Orchestra — open, the first buy to fill takes it"
    )
    if not race.get("running"):
        title += " (stopped)"
    st.markdown(title)
    st.dataframe(
        pd.DataFrame([
            {
                "Pair": row["label"],
                "Status": row["status"],
                "Above buy (units)": row["to_buy"],
                "Last": row["close"],
                "Buy": row["buy"],
                "Sell": row["sell"],
            }
            for row in board
        ]),
        hide_index=True,
        width="stretch",
        column_config={
            "Last": st.column_config.NumberColumn(format="$%.2f"),
            "Buy": st.column_config.NumberColumn(format="$%.2f"),
            "Sell": st.column_config.NumberColumn(format="$%.2f"),
            "Above buy (units)": st.column_config.NumberColumn(
                format="%+.2f",
                help="(last close − buy level) ÷ the pair's level unit. The pair "
                "nearest 0 is the nearest to buying.",
            ),
        },
    )


def _record_live_equity_point(state: AppState) -> None:
    """Append a snapshot of the agent's current total value (all positions marked
    to their live prices), so the chart keeps advancing every poll instead of
    waiting on a full bar to close (bars can lag a minute or more behind)."""
    value = state.mark_to_market()
    if value is None:
        return
    tracker = state.decision_tracker
    snap = tracker.snapshot() if tracker else {"cash": 0.0, "positions": {}}
    with state.lock:
        state.agent_equity_history.append(
            {
                "ts": pd.Timestamp.now(tz="UTC").isoformat(),
                "price": None,
                "cash": snap["cash"],
                "position": sum(snap["positions"].values()),
                "value": value,
            }
        )
        if len(state.agent_equity_history) > AGENT_EQUITY_HISTORY_MAXLEN:
            state.agent_equity_history = state.agent_equity_history[-AGENT_EQUITY_HISTORY_MAXLEN:]


def _merge_live_history(state: AppState, points: list[dict], agent_start: datetime) -> list[dict]:
    with state.lock:
        history = [h for h in state.agent_equity_history if pd.Timestamp(h["ts"]) > pd.Timestamp(agent_start)]
    return sorted(points + history, key=lambda p: p["ts"])


def _bars_by_symbol(state: AppState) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {}
    for sym_state in state.iter_symbol_states():
        with sym_state.lock:
            result[sym_state.symbol] = list(sym_state.bars)
    return result


@st.fragment(run_every=AGENT_PERFORMANCE_POLL_SEC)
def _agent_performance_panel(symbols: list[str]) -> None:
    state = _get_state()
    tracker = state.decision_tracker
    if not tracker:
        st.plotly_chart(empty_chart("Start the agent to track performance"), width='stretch')
        return

    # This panel *is* the portfolio-value display, it runs once a minute, and it
    # runs on Streamlit's own thread -- so read the account's value fresh here
    # rather than taking whatever the background refresh last left cached.
    tracker.refresh_venue_value()
    snap = tracker.snapshot()
    decisions = [asdict(d) for d in snap["decisions"]]
    bars_by_symbol = _bars_by_symbol(state)

    agent_start = state.agent_start_time or SESSION_START
    _record_live_equity_point(state)
    points = compute_equity_curve(bars_by_symbol, decisions, state.starting_budget, agent_start)
    points = _merge_live_history(state, points, agent_start)
    markers = decision_markers(decisions, agent_start, points)
    # On a real Alpaca account the account itself is what the portfolio is
    # worth; the curve above is a replay of this session's decisions and cannot
    # see holdings it did not open, unstreamed symbols or outside cash moves.
    stats = summarize(points, decisions, state.starting_budget, snap.get("venue_value"))

    c1, c2, c3 = st.columns(3)
    c1.metric(
        _portfolio_value_label(state),
        f"${stats['current_value']:,.2f}",
        f"{stats['return_pct']:+.2f}%",
    )
    c2.metric("Fees paid", f"${stats['total_fees']:,.2f}")
    c3.metric(_starting_value_label(state), f"${stats['starting_cash']:,.2f}")

    label = ", ".join(symbols or state.symbols)
    fig = build_performance_chart(points, markers, label)
    st.plotly_chart(fig, width='stretch')


def _report_briefing(state: AppState, syms: list[str]) -> dict:
    """The Pre-Market tab's briefing cards for the report: title, the
    generated-at note (with any symbol that failed), and one card per symbol."""
    briefings = dict(state.premarket_briefings or {})
    errors = dict(state.premarket_errors or {})
    phase = state.premarket_phase
    notes = []
    if state.premarket_generated_at is not None:
        generated_et = state.premarket_generated_at.astimezone(market_hours.MARKET_TZ)
        notes.append(f"Generated {generated_et.strftime('%Y-%m-%d %H:%M')} ET")
    notes += [f"failed for {sym}: {errors[sym]}" for sym in syms if sym in errors]
    return {
        "briefing_title": PHASE_TITLES.get(phase, "Pre-Market Briefing"),
        "briefing_note": " · ".join(notes),
        "briefing_cards": [
            _premarket_briefing_html(briefings[sym], sym, phase)
            for sym in syms
            if sym in briefings
        ],
    }


def _report_news_cards(state: AppState, syms: list[str]) -> list[str]:
    """The News tab's cards for the report, one per streamed symbol."""
    cards = []
    for sym in syms:
        sym_state = state.sym(sym)
        if sym_state is None:
            continue
        with sym_state.lock:
            news = list(sym_state.news)
            impacts = dict(sym_state.news_impacts)
            details = dict(sym_state.news_impact_details)
        cards.append(_news_html(news, sym, impacts, details))
    return cards


def _report_option_walls(state: AppState, syms: list[str]) -> list[dict]:
    """The Put/Call Walls tab's chart and read per symbol, from the chain that
    tab (or the live chart's wall overlay) last stored. A symbol with none yet
    gets one fetch through the same 5-minute cache; one that fails is left out."""
    walls = []
    for sym in syms:
        sym_state = state.sym(sym)
        data, history, spot = None, [], None
        if sym_state is not None:
            with sym_state.lock:
                data = sym_state.options_chain
                history = list(sym_state.options_wall_history)
                spot = sym_state.last_price
        if not data:
            try:
                data = fetch_options_walls_data(sym, spot=spot)
            except Exception:
                continue
        if not data or not data.get("strikes"):
            continue
        analysis = get_put_call_walls_and_gamma(
            strikes=data["strikes"],
            calls_oi=data["calls_oi"],
            puts_oi=data["puts_oi"],
            calls_gamma_exposure=data["calls_gamma_exposure"],
            puts_gamma_exposure=data["puts_gamma_exposure"],
            spot=data["spot"],
            wall_history=history,
        )
        walls.append(
            {
                "symbol": sym,
                "fig": build_gamma_chart(data, analysis, sym),
                "expiry": data.get("expiry", ""),
                "fetched_at": data.get("fetched_at", ""),
                "analysis": analysis,
            }
        )
    return walls


def _build_agent_report_html(state: AppState, symbols: list[str]) -> str:
    syms = symbols or list(state.symbols)

    with state.lock:
        agent_log = list(state.agent_log)

    tracker = state.decision_tracker
    tracker_snap = tracker.snapshot() if tracker else None
    decisions = [asdict(d) for d in tracker_snap["decisions"]] if tracker_snap else []

    live_figs: list[tuple[str, object]] = []
    for sym in syms:
        sym_state = state.sym(sym)
        if sym_state is None:
            continue
        with sym_state.lock:
            bars = list(sym_state.bars)
            trades = list(sym_state.trades)
        if not bars:
            continue
        live_figs.append(
            (
                sym,
                build_chart(
                    bars,
                    sym_state.news,
                    trades,
                    sym,
                    _chart_start(bars, state.show_pre_market),
                    ma_periods=state.ma_periods,
                    show_fib=state.show_fib,
                    show_7d_avg=state.show_7d_avg,
                    show_28d_avg=state.show_28d_avg,
                    show_1y_avg=state.show_1y_avg,
                    mixture_distribution=state.mixture_distribution,
                    mixture_max_components=state.mixture_max_components,
                    predicted_profile=(
                        predicted_open_profile(sym_state, bars)
                        if state.show_predicted_profile
                        else None
                    ),
                    mixture_fit_target=state.mixture_fit_target,
                    daily_bars=sym_state.daily_bars,
                    vwap_style=state.vwap_style,
                    show_candle_body=state.show_candle_body,
                    show_percentile_body=state.show_percentile_body,
                    show_whiskers=state.show_whiskers,
                    decisions=tracker.trade_markers(symbol=sym) if tracker else None,
                    news_impacts=sym_state.news_impacts,
                    fill_gaps=state.fill_gaps,
                    volume_baseline=_volume_baseline(sym, bars, state),
                    model_overlays=model_overlays.live_overlays(
                        sym_state, bars, state.model_overlay_keys,
                    )["items"],
                    candle_patterns=_live_candle_patterns(state, bars),
                    show_momentum=state.show_momentum,
                    minute_momentum_profile=sym_state.minute_momentum_profile,
                    minute_momentum_change_profile=sym_state.minute_momentum_change_profile,
                    **_agent_momentum_kwargs(state, sym_state),
                ),
            )
        )

    performance_fig = None
    performance_stats = None
    agent_start = state.agent_start_time or SESSION_START
    if tracker:
        points = compute_equity_curve(_bars_by_symbol(state), decisions, state.starting_budget, agent_start)
        points = _merge_live_history(state, points, agent_start)
        markers = decision_markers(decisions, agent_start, points)
        performance_stats = summarize(
            points, decisions, state.starting_budget, tracker_snap.get("venue_value")
        )
        performance_fig = build_performance_chart(points, markers, ", ".join(syms))

    return build_report_html(
        symbols=syms,
        feed=state.feed,
        timeframe=state.timeframe,
        session_start=agent_start,
        starting_budget=state.starting_budget,
        trading_venue=MODE_LABELS.get(state.trading_mode, state.trading_mode),
        trade_fixed_cost=TRADE_FIXED_COST,
        llm_provider=state.llm_provider,
        llm_model=state.llm_model,
        llm_personality=_personality_label(state.llm_personality),
        agent_running=state.agent_running,
        live_figs=live_figs,
        performance_fig=performance_fig,
        performance_stats=performance_stats,
        decisions=decisions,
        agent_log=agent_log,
        news_cards=_report_news_cards(state, syms),
        option_walls=_report_option_walls(state, syms),
        **_report_briefing(state, syms),
    )


def _agent_report_section(symbols: list[str]) -> None:
    state = _get_state()
    st.divider()
    st.caption(
        "Save everything about this run — starting conditions, the briefing and news, "
        "the charts (put/call walls included), and the full decision history — to a "
        "single HTML file."
    )
    if st.button("📄 Generate Report", key="agent_generate_report"):
        with st.spinner("Building report…"):
            try:
                report_html = _build_agent_report_html(state, symbols)
            except Exception as exc:
                st.error(f"Failed to build report: {exc}")
            else:
                label = "_".join(symbols or state.symbols) or "agent"
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.session_state["agent_report_html"] = report_html
                st.session_state["agent_report_name"] = f"{label}_agent_report_{ts}.html"

    report_html = st.session_state.get("agent_report_html")
    if report_html:
        st.download_button(
            "💾 Save Report (.html)",
            data=report_html,
            file_name=st.session_state.get("agent_report_name", "agent_report.html"),
            mime="text/html",
            key="agent_report_download",
        )


# The dashboard's half of the Apple Trader form (see `apple_trader_ui`): the
# wording a run about to start needs, as opposed to SimLab's, which is about
# what is worth sweeping.
_APPLE_TRADER_COPY = apple_trader_ui.FormCopy(
    prefix="apple_trader",
    unavailable_suffix="not streamed",
    instrument_help=(
        "The one symbol this run trades.\n\n"
        "- Only symbols a saved model covers are listed — without a model there is "
        "no strategy to run.\n"
        "- It must also be streamed: add it to the sidebar symbols before starting."
    ),
    model_help=(
        "Which saved model the agent runs on — and, with it, which rules. Only the "
        "models fitted on the instrument above are listed."
    ),
    sections={
        "dayrange": (
            "At 9:35 the model forecasts where today's high **H** will land, and two "
            "levels are set below it.\n\n"
            "- **Buy** — rests well under H, so it fills on a dip.\n"
            "- **Sell** — rests just beneath H.\n"
            "- **Unit** — both distances are counted in the trailing 14-day ADR (they "
            "scale with how wide recent sessions were) or in the forecast's own "
            "predicted range (they scale with how wide the model thinks *today* will be)."
        ),
        "dayrange_breach": (
            "**H** is a forecast, and the day can trade straight through it.\n\n"
            "- What happens then is chosen below.\n"
            "- Both levels are rebuilt from the predicted high every time it moves — "
            "live, including under a position that is already open."
        ),
        "dayrange_exits": (
            "How a position gets out before the sell level. Everything is measured from "
            "the fill.\n\n"
            "- **Stop** — under the fill, as a share of what the trade is playing for.\n"
            "- **Momentum take** — part of the position, when momentum has been negative "
            "for long enough while the trade is in profit.\n"
            "- **Runner** — the rest, kept for the sell level only if that is still far "
            "enough away; sold if the price comes back to the fill.\n\n"
            "A trade closes at the sell level, the stop, a momentum take (its runner at "
            "the sell level or back at the fill), or the closing flatten. None of this is "
            "in the notebook's results."
        ),
        "dayrange_skip": (
            "Sessions the agent does not trade at all: no forecast, no order. The forecast "
            "is built for an ordinary day, and on these the news sets the range — "
            "HighLow2 was fitted with most of them left out."
        ),
        "dayrange_breaker": (
            "When to stop buying for the day. Two things end it:\n\n"
            "- **A stop** — the day has not gone the way the forecast said.\n"
            "- **A trade that barely paid** — the forecast said the day was wide enough "
            "for the dip to be worth buying, and the trade says otherwise."
        ),
    },
    help={
        "model_dayrange": (
            "The levels hang off the predicted high — one number for the whole session. "
            "The notebook's rule, and what the buy and sell distances were swept against."
        ),
        "model_highlow": (
            "The day-range rules, unchanged, on HighLow's predicted high and predicted "
            "range instead of TimeToChange3's.\n\n"
            "- Its range is usually **narrower** (it anchors on the 9:35 price), so "
            "distances counted in the predicted range sit closer together.\n"
            "- The shipped buy/sell distances were swept on TimeToChange3's forecast, "
            "not this one.\n"
            "- At 9:35 it needs ~150 sessions of Alpaca SIP minute history: the first "
            "run fetches it (tens of seconds) and caches it under `data/highlow/`."
        ),
        "skip_events": (
            "Checked when the opening window closes, before the forecast.\n\n"
            "- **Day after earnings** — the first session after the symbol's report "
            "(Yahoo's earnings dates).\n"
            "- **CPI release**, **Jobs report (NFP)** — BLS days, 08:30 ET, from "
            "`agent_stonks/calendars/shock_days.csv`. BLS's dates run to December 2026; "
            "add 2027's when they are published (the log says when they run out).\n"
            "- **Market shock**, **Geopolitical shock** — flagged by the pre-market "
            "briefing for the symbol (shown in red on the Pre-Market tab). Without a "
            "briefing only the calendar can say so, and its shock rows end on 25 Sep 2026. "
            "A briefing still being written at 9:35 is waited for up to 10 minutes."
        ),
        "model_highlow2": (
            "The day-range rules, unchanged, on HighLow2's predicted high and predicted "
            "range.\n\n"
            "- It reads this morning from **IEX** — the first five minutes and the open "
            "— plus the pre-market (SIP to 09:19, IEX to 09:29) and last evening's "
            "after-hours, so the stream's feed does not change its forecast.\n"
            "- It was fitted **without shock days** (the day after earnings, jobs-report "
            "days): on those it forecasts an ordinary day's width.\n"
            "- No buy/sell distances were ever swept on it: it starts from the pair "
            "TimeToChange3's notebook 05 swept for the instrument. Sweep them on "
            "SimLab's Tuning tab.\n"
            "- At 9:35 it needs ~150 sessions of Alpaca SIP and IEX minute history, "
            "extended hours included: the first run fetches it (tens of seconds) and "
            "caches it under `data/highlow2/`."
        ),
        "model_dayrange_intraday": (
            "The levels rest under the top of the intraday band.\n\n"
            "- The band *is* the predicted high at 09:30, pulls in to about a fifth of "
            "the distance to the open by midday, and opens back up into the last half hour.\n"
            "- Both levels move with it, so an entry needs a deeper dip as the day quiets.\n"
            "- **The target comes down too** — a morning position can be closed by a sell "
            "level that fell to it. That is the model's claim, not a bug, but it is a "
            "different strategy from the flat model and has not been swept.\n"
            "- The band is anchored on the session's **open**, not the price: a day that "
            "trends away from the open leaves the levels behind, so the stop matters more here."
        ),
        "buy_k": (
            "How far under the predicted high the entry rests.\n\n"
            "- {ticker} starts at **{buy_k}** on this model — the most profitable cell of "
            "a buy × sell grid replayed in SimLab over two or three weeks of {ticker} tape and "
            "summed. A model never tuned on {ticker} starts from notebook 05's sweep.\n"
            "- Shallower entries fill on more days; deeper ones pay a better price on fewer.\n"
            "- It was picked on the sessions it was scored on: the best-evidenced starting "
            "point, not an edge."
        ),
        "sell_k": (
            "Where the exit rests, below the same predicted high.\n\n"
            "- {ticker} starts at **{sell_k}** on this model, from the same sweep.\n"
            "- It must be the smaller of the two numbers — the higher price.\n"
            "- A day that never reaches it is held to the closing flatten."
        ),
        "level_unit": (
            "What one unit of `buy` and `sell` is worth in dollars.\n\n"
            "- **ADR** — this symbol's trailing 14-day average daily range. Fixed for "
            "the session, and no part of the model's output.\n"
            "- **Predicted range** — `predicted high − predicted low` from the same "
            "forecast, so a day the model calls wide gets wider distances. With an "
            "intraday update on it *widens* as the day breaches its forecast, spreading "
            "the levels apart; a stop stays fixed in dollars at the fill.\n\n"
            "The shipped distances were swept in ADRs — under the predicted range they "
            "are starting points only."
        ),
        "breach_update": (
            "What replaces the predicted high once the session trades through it. The "
            "model cannot be re-run intraday — its features are a fixed five-minute window.\n\n"
            "- **Hold the 9:35 forecast** — it stands whatever the tape does. The "
            "notebook's rule, and the one the distances were swept under.\n"
            "- **Move to the extreme so far** — the breached side moves to the session's "
            "own high (or low) and the other side moves with it, so the range keeps its "
            "width and follows the day. The other side never passes "
            "a price the session has already printed. It never leads the tape: one move "
            "late, but it never claims a high the day has not made.\n"
            "- **Brownian extension** — past the extreme, by what a driftless random walk "
            "with this ADR's volatility would still cover before the close: half an ADR "
            "with the whole day left, nothing at the bell. It leads the tape, and pays "
            "for that with a sell level the day may never come back up to.\n\n"
            "Both moving policies keep the range as wide as the unit the levels are "
            "counted in — the 9:35 predicted range, or the ADR — and move the other side "
            "with the breached one. The range only widens when that would put the other "
            "side past a price the session has already traded. "
            "Neither updating policy has been swept."
        ),
        "contain_range": (
            "Keeps the forecast wide enough to hold what the session has printed.\n\n"
            "- The 9:35 forecast is already clipped to the first five minutes; this keeps "
            "that true all day.\n"
            "- If the session trades above the predicted high or below the predicted low, "
            "the forecast widens to hold it.\n"
            "- It never leads the tape — it only stops the levels being measured from a "
            "price the day has already passed.\n\n"
            "With it on, *hold the 9:35 forecast* moves a breached side to the extreme "
            "too; *move to the extreme so far* still differs by moving the other side "
            "with it."
        ),
        "breach_exit": (
            "The position is a bet that the day tops out near the predicted high.\n\n"
            "- **On** — a bar that trades clean through that high closes the position at "
            "market: the bet is settled, at a better price than the sell level.\n"
            "- **Off** — the breach moves the forecast instead. Under *Brownian "
            "extension* that carries the sell level past the bar, so the position rides "
            "on and can give the gain back.\n\n"
            "Measured against the high as it stood when the bar opened, and checked after "
            "the sell level — a breach that also reached the target logs as the target."
        ),
        "position_pct": "The share of available cash each entry spends.",
        "scale_in": (
            "Adds to an open position on the way down, while the cash left over can "
            "pay for at least one more share.\n\n"
            "- Each buy spends *Position size* of the cash that is left.\n"
            "- After every buy the next one rests **Next buy step** lower — 0.55, "
            "0.65, 0.75… at a 0.1 step — and selling the whole position puts it back "
            "at the buy distance.\n"
            "- The stop sits its usual distance under **the last actual fill**, so "
            "the next buy is placed only while it is above that stop.\n"
            "- An add fills only on a bar that closes **under the last fill**, so "
            "every buy is lower than the one before.\n"
            "- The target, the momentum take and the breakeven measure from the "
            "average cost.\n"
            "- Needs a position size under 100%."
        ),
        "buy_step_k": (
            "How much further below H each buy moves the next one, in the same units "
            "as the distances.\n\n"
            "- At 0.1 a buy distance of 0.55 becomes 0.65 after the first buy, 0.75 "
            "after the second…\n"
            "- Selling the whole position resets it to the buy distance; a partial "
            "take does not.\n"
            "- A step at or past the stop distance never adds: the stop is hit first.\n"
            "- **0** — the old rule: each buy half-way to the bottom of the range "
            "(the predicted low, or one ADR under H). 0.40 → 0.70 → 0.85…"
        ),
        "stop_gain_fraction": (
            "Sells everything when a bar's low reaches this share of the **predicted "
            "gain** under the fill.\n\n"
            "- Predicted gain = `buy − sell`: the gap between the levels, and the most a "
            "target exit can pay.\n"
            "- {stop_gain_fraction} risks \\$0.50 for every \\$1.00 the trade plays for — "
            "on any instrument, at any levels.\n"
            "- After a stop the agent stops; press ▶ Start Agent to trade again.\n"
            "- 0 switches the stop off."
        ),
        "momentum_confirmation_bars": (
            "One look-back, in bars, over which both the buy and the sells read the "
            "momentum behaviour table.\n\n"
            "- **Momentum** — the average move per bar over the period, "
            "`(close − close N bars ago) / N`. **Change** — the average bar-to-bar "
            "change of the 1-bar momentum, `(m1 − m1 N bars ago) / N`.\n"
            "- Each is **neutral** while its size is under 0.1 × the ticker's mean "
            "absolute one-minute move over last week (`abs_mean_minute_momentum`), "
            "else positive or negative.\n"
            "- **Buy** at the buy level only on positive momentum, or neutral momentum "
            "whose change is not negative (flat, or about to rise).\n"
            "- **At or above the sell level**, sell unless momentum is still positive "
            "— then hold on while the price keeps rising. The breach exit likewise.\n"
            "- **Below the sell level**, in profit: sell the *Take on negative "
            "momentum* share while momentum is negative and its change neutral or "
            "negative (still dropping, or dropping faster).\n"
            "- Never gates the stop loss, the runner's breakeven or the closing "
            "flatten.\n"
            "- Until it can be read (the first N + 2 bars, or before last week's "
            "mean move is known) nothing is bought or sold at the levels.\n"
            "- 0 switches it off, and with it the momentum take and the runner."
        ),
        "take_fraction": (
            "How much of the position a momentum take sells when the rest is kept as a "
            "runner.\n\n"
            "- The take fires below the sell level, in profit, while momentum over the "
            "*Momentum confirmation period* is negative and its change neutral or "
            "negative.\n"
            "- Whole shares, rounded down, and at least one."
        ),
        "take_after_minutes": (
            "The momentum take fires only once this many minutes have passed since the "
            "last buy.\n\n"
            "- Before that a negative momentum read is left alone; the stop covers the "
            "downside.\n"
            "- A buy lower down (scale-in) starts the wait again.\n"
            "- 0 takes from the first bar after the buy."
        ),
        "hold_min_gain_k": (
            "Keep a runner only if the sell level is still this many × {unit} above the fill; "
            "otherwise a momentum take sells everything.\n\n"
            "- The gap is at most `buy − sell`: 0.15 on AAPL's default pair, 0.60 on "
            "GOOGL's — so 0.30 keeps a runner on GOOGL and INTC, never on AAPL.\n"
            "- A runner is sold at the sell level, at the flatten, or back at the fill."
        ),
        "min_win_k": (
            "After a trade closes for no more than this many × {unit} **per share**, nothing "
            "else is bought today.\n\n"
            "- {ticker} starts at **{min_win_k}** — per instrument, since it only means "
            "something against that symbol's own buy/sell pair.\n"
            "- Judged over the whole position: a momentum take and its runner count as "
            "one trade.\n"
            "- Catches trades that were not losses but were not worth the risk either, "
            "including a runner sold back at the fill.\n"
            "- At or above `buy − sell`, every trade stands the session down — a "
            "one-trade-a-day rule.\n"
            "- 0 switches it off and lets the levels re-arm all day."
        ),
    },
)

# Orchestra's form: the same knobs (and the same wording for them) as Apple
# Trader's under its own widget-key prefix, so its settings are its own, plus
# the wording for its pairs.
_ORCHESTRA_COPY = dc_replace(
    _APPLE_TRADER_COPY,
    prefix="orchestra",
    sections={
        **_APPLE_TRADER_COPY.sections,
        "selection": (
            "Once a session, at **09:34 ET** — four opening minutes in, one bar before "
            "the forecasts — Orchestra narrows its pairs to the day's candidates. The "
            "rest sit the day out: they are not read and never forecast.\n\n"
            "- Reads the pre-market briefing (bias, confidence), the earnings calendar, "
            "and the opening gap on the daily ADR.\n"
            "- Ranks what is left by what a target exit pays as a share of the price "
            "(`buy − sell` ADRs over the price), best first.\n"
            "- The Candidates tab shows the pick — provisional until 09:34, then the "
            "one Orchestra made. A restart keeps it."
        ),
        "pairs": (
            "Every pair forecasts its own **H** at 9:35 and rests its own buy and sell "
            "below it, in the same unit.\n\n"
            "- **Buy / Sell** — each pair's own distances, starting from its tuned pair.\n"
            "- **Min win** — each pair's circuit breaker: a trade that closes for no more "
            "than this stands *that pair* down for the day; the others keep racing.\n"
            "- Everything below the table is shared by every pair."
        ),
    },
    help={
        **_APPLE_TRADER_COPY.help,
        "pairs": (
            "The (ticker, model) pairs Orchestra watches — AAPL on the day-range model "
            "and AAPL on HighLow are two pairs. The defaults are the pairs SimLab's "
            "tuning has picked levels for. The order is the tie-break: when two buys "
            "would fill on the same bar, the pair listed first takes it. Every pair's "
            "ticker must be streamed."
        ),
        "pair_min_win_k": (
            "Per pair: a trade that closes for no more than this many units a share "
            "stands that pair down for the rest of the day. 0 is off."
        ),
        "select_on": (
            "Off, Orchestra races every pair all day. On, it picks the day's "
            "candidates once at 09:34 and only those race."
        ),
        "select_max_pairs": (
            "How many pairs race after the rules below have left out what they "
            "leave out — the best by target as a share of the price. 0 keeps every "
            "eligible pair."
        ),
        "select_max_gap_adr": (
            "Leave out a symbol whose official open is further than this many average "
            "daily ranges from yesterday's close, up or down. 0 is no limit."
        ),
        "select_bearish": (
            "The strategy only buys dips and only goes long, so a confidently bearish "
            "morning is the one most likely to walk through the buy level into the stop. "
            "A symbol whose briefing is not ready by 09:34 is not left out for it."
        ),
        "select_earnings": (
            "Leave out a symbol with an earnings report between yesterday's close and "
            "today's — a day unlike the ones the levels were tuned on."
        ),
    },
)


# Both rule forms are fragments: a knob change reruns the form alone rather
# than the whole page, every tab of which Streamlit would otherwise re-render
# (and a full rerun is what made each edit freeze the screen). What they return
# is only read by ▶ Start, whose click is a full rerun that runs these again
# and so returns the values on screen. The chart overlay reads the config from
# the state instead, so it is published from inside the fragment.
@st.fragment
def _apple_trader_params(symbols: list[str]) -> AppleTraderConfig:
    """Apple Trader's instrument and tunables, inside the dashboard's expander."""
    state = _get_state()
    levels = getattr(state, "apple_trader_levels", None) or {}
    running = levels if state.agent_running else {}
    # After a restart the form reopens on the restored run's buy and sell, so
    # the ▶ Start that continues it does not quietly move them -- unless the
    # pair was changed on screen, when last_setup has already put the last
    # values in (which a running agent had adopted anyway).
    seed = running.get("config") or (
        levels.get("config") if getattr(state, "session_restored", None) else None
    )
    with st.expander("Apple Trader rules", expanded=True):
        config = apple_trader_ui.params(symbols, _APPLE_TRADER_COPY, seed=seed)
        if running.get("config") is not None:
            st.caption(
                "The running agent picks up a change to the model or to the buy or "
                "sell distance at its next bar. Every other setting takes effect on "
                "▶ Start."
            )
    # Published for the chart's buy/sell overlay, which is drawn from a
    # configuration rather than from a model and should show the one on screen,
    # and for the running agent, which reads the two distances back from it.
    state.apple_trader_config = config
    # A knob change reruns this fragment alone, so the end of the full run
    # that would otherwise save it never comes.
    last_setup.remember()
    return config


def _running_orchestra(state: AppState) -> "OrchestraConfig | None":
    """The running Orchestra's configuration as its racers now hold it, or None."""
    race = getattr(state, "orchestra", None) or {}
    if not (state.agent_running and race.get("running")):
        return None
    records = getattr(state, "orchestra_levels", None) or {}
    racers = [
        records[key]["config"] for key in race.get("order") or []
        if (records.get(key) or {}).get("config") is not None
    ]
    try:
        return OrchestraConfig(racers) if racers else None
    except ValueError:
        return None


@st.fragment
def _orchestra_params(symbols: list[str]) -> "OrchestraConfig | None":
    """Orchestra's pairs, each pair's numbers and the rules they share, inside
    the dashboard's expander. A fragment for the same reason as Apple Trader's."""
    state = _get_state()
    seed = _running_orchestra(state)
    with st.expander("Orchestra rules", expanded=True):
        race = apple_trader_ui.orchestra_params(symbols, _ORCHESTRA_COPY, seed=seed)
        if seed is not None:
            st.caption(
                "The running Orchestra picks up a change to a pair's buy or sell distance "
                "at its next bar. Every other setting — and adding or removing a pair — "
                "takes effect on ▶ Start."
            )
        if race is not None:
            missing = [t for t in race.tickers if t not in symbols]
            if missing:
                st.warning(
                    f"{', '.join(missing)} {'is' if len(missing) == 1 else 'are'} not in "
                    "the sidebar symbols, so Orchestra cannot read "
                    f"{'its' if len(missing) == 1 else 'their'} bars.",
                    icon=":material/warning:",
                )
                if st.button(
                    f"Add {', '.join(missing)} to the symbols",
                    # Not under the "orchestra_" prefix: last_setup keeps
                    # those, and a button's value cannot be restored.
                    key="add_symbols_for_orchestra",
                    on_click=_add_sidebar_symbols,
                    args=(missing,),
                    help="Adds them to the sidebar's symbol list. ▶ Start Agent then "
                    "restarts the live stream with them in it.",
                ):
                    # This form is a fragment: only a full rerun redraws the
                    # sidebar and hands the form the longer symbol list.
                    st.rerun(scope="app")
    # Published for the chart (each racer's levels) and for the running
    # Orchestra, whose racers read their two distances back from here.
    state.orchestra_configs = (
        {f"{r.ticker}:{r.model_key}": r for r in race.racers} if race is not None else {}
    )
    # And whole, for the Candidates tab's provisional pick.
    state.orchestra_form = race
    last_setup.remember()
    return race


def _add_sidebar_symbols(symbols: "list[str]") -> None:
    """Append `symbols` to the sidebar's symbol box (an on_click callback: a
    widget's value can only be set before it renders)."""
    current = _parse_symbols(st.session_state.get("sidebar_symbols", ""))
    st.session_state["sidebar_symbols"] = ", ".join(
        current + [s for s in symbols if s not in current]
    )


def _execution_controls() -> str:
    """Where this agent run's orders go. Returns the requested mode.

    Rendered above the Start button rather than tucked in an expander: which
    account an automated strategy is about to trade is the single most
    consequential setting on this page, and it should not be possible to press
    ▶ Start without having seen it.
    """
    mode = st.selectbox(
        "Order execution",
        TRADING_MODES,
        index=TRADING_MODES.index(DEFAULT_TRADING_MODE),
        format_func=lambda m: MODE_LABELS.get(m, m),
        key="agent_trading_mode",
        help=(
            "**Local simulation** keeps the in-memory ledger this app has always "
            "used — nothing leaves the process.\n\n"
            "**Alpaca paper** sends real orders to your paper account: real "
            "routing, real fills, real rejections, fake money. Needs "
            "`ALPACA_PAPER_API_KEY` / `ALPACA_PAPER_SECRET`.\n\n"
            "**Alpaca LIVE** sends real orders with real money. Needs "
            f"`{LIVE_TRADING_ENV_FLAG}=true` in the environment."
        ),
    )

    # Whether this venue can actually be reached is knowable right here, from
    # the environment alone, and telling the user now beats letting them press
    # ▶ Start and discover the run quietly degraded to simulation.
    keyed = credentials_for(mode).configured if mode != "local" else True

    if mode == "alpaca_live":
        if not live_trading_enabled():
            st.error(
                f"Live trading is disabled. Set `{LIVE_TRADING_ENV_FLAG}=true` in your "
                "environment and restart the app to enable it. Until then this run "
                "falls back to local simulation."
            )
        elif not keyed:
            key_var, secret_var = TRADING_ENV_KEYS["alpaca_live"]
            st.error(
                f"**No live credentials.** `{key_var}` and `{secret_var}` are not set, "
                "so this run falls back to local simulation and sends nothing "
                "anywhere. Live deliberately does not borrow `ALPACA_API_KEY`: if "
                "that happened to be a live key, whether real money moved would come "
                "down to which variable was already set."
            )
        elif _get_state().trading_mode != "alpaca_live":
            # A pre-Start caution only: once a live run is on, the status block
            # under the Start button says so and this would be a second banner.
            st.warning(
                "**This will trade real money.** Pressing ▶ Start arms an automated "
                "strategy on your live Alpaca account, and it will place orders "
                "without asking again per trade."
            )
    elif mode == "alpaca_paper":
        if not keyed:
            key_var, secret_var = TRADING_ENV_KEYS["alpaca_paper"]
            st.error(
                f"**No paper credentials.** Set `{key_var}` / `{secret_var}` (or "
                f"`{PAPER_FALLBACK_ENV[0]}` / `{PAPER_FALLBACK_ENV[1]}`, if those are "
                "your paper keys). Until then this run falls back to local simulation."
            )
        else:
            st.caption(
                "Orders go to your Alpaca **paper** account — no real money, but real "
                "order routing, so rejections and partial fills are real too."
            )
    return mode


def _start_agent(
    state: AppState,
    syms: "list[str]",
    *,
    personality: str,
    provider: str,
    model: str,
    apple_config: "AppleTraderConfig | OrchestraConfig | None",
    trading_mode_choice: str,
    starting_budget: float,
    continue_today: bool,
    alpaca_key: str = "",
    alpaca_secret: str = "",
    feed: str = "iex",
    data_source: str = DEFAULT_DATA_SOURCE,
    finnhub_token: str = "",
    history_feed: str = DEFAULT_HISTORY_FEED,
) -> bool:
    """What ▶ Start Agent does: check the setup, start the live stream when it
    is not running for `syms`, resolve the venue, open or continue today's
    ledger and launch `personality`. True when the agent was launched.

    Shared by the button and by `_resume_after_restart`, which starts again
    the run a restart interrupted from what `state.run_spec` recorded of it."""
    env_var = ENV_KEYS[provider]
    llm_key = os.getenv(env_var, "")
    is_apple_trader = personality == APPLE_TRADER_KEY
    is_rule_agent = personality in RULE_AGENT_KEYS
    is_orchestra = personality == ORCHESTRA_KEY
    # The symbols a rule agent trades: Apple Trader's one, or every pair's
    # under Orchestra. They have to be streamed, or there are no bars to read
    # and the agent would idle all session.
    rule_tickers = [APPLE_TRADER_TICKER]
    if is_orchestra and isinstance(apple_config, OrchestraConfig):
        rule_tickers = apple_config.tickers
    elif is_apple_trader and apple_config is not None:
        rule_tickers = [apple_config.ticker]
    unstreamed = [t for t in rule_tickers if t not in syms]
    stream_ready = False
    if not syms:
        st.error("Enter at least one symbol in the sidebar first.")
    elif is_orchestra and not isinstance(apple_config, OrchestraConfig):
        st.error("Pick at least one (ticker, model) pair for Orchestra.")
    elif is_rule_agent and unstreamed:
        st.error(
            f"{_personality_label(personality)} is configured to trade "
            f"{', '.join(rule_tickers)}; add {', '.join(unstreamed)} to the symbols in "
            "the sidebar."
        )
    elif not llm_key and not is_rule_agent:
        st.error(f"{env_var} is not set; the agent needs an LLM key to reason about decisions.")
    else:
        # The live stream feeds every tool the agent reads. If it isn't
        # running for these symbols yet, start it here rather than sending
        # the user back to the sidebar first.
        stream_ready = bool(state.api_key) and all(state.sym(s) is not None for s in syms)
        if not stream_ready:
            key = alpaca_key.strip() or os.getenv("ALPACA_API_KEY", "")
            secret = alpaca_secret.strip() or os.getenv("ALPACA_SECRET", "")
            if not key or not secret:
                st.error(
                    "Alpaca API key and secret are required to start the live stream "
                    "(sidebar Connection expander, or the ALPACA_API_KEY / "
                    "ALPACA_SECRET environment variables)."
                )
            else:
                timeframe = st.session_state.get("live_timeframe", TIMEFRAMES[0])
                stream_ready = _start_live_session(
                    state, syms, key, secret, feed, timeframe,
                    data_source=data_source, finnhub_token=finnhub_token,
                    history_feed=history_feed,
                )
    if stream_ready:
        if not market_hours.is_market_open():
            open_et = market_hours.next_market_open().astimezone(market_hours.MARKET_TZ)
            st.info(
                f"The trading session hasn't started yet (next open: "
                f"{open_et.strftime('%a %Y-%m-%d %H:%M')} ET). The agent is told the "
                "market is closed and adapts: the Premarket Analyst prepares opening "
                "tactics, other strategies study structure and arm plans for the "
                "open instead of trading the stale tape."
                + (
                    " The Apple Traders simply idle until the bell — they score "
                    "closed minute bars and there are none."
                    if is_rule_agent
                    else ""
                )
            )
        # Resolve the venue before anything starts. Every refusal inside
        # resolve_broker degrades to local simulation and says so, so a
        # misconfigured or blocked account can never silently become a
        # different account than the one the user picked.
        live_broker, effective_mode, broker_message = resolve_broker(
            trading_mode_choice
        )
        # Read before `trading_mode` becomes the new venue: a ledger is only
        # continued on the venue it was kept on.
        continuing = continue_today and session_store.continues(state, effective_mode)
        state.trading_mode = effective_mode
        state.trading_mode_requested = trading_mode_choice
        state.trading_status = broker_message
        # Only local simulation is announced here: a degraded, paper or live
        # venue gets its banner from the persistent status block below,
        # and announcing it here too would stack two banners saying the same.
        if effective_mode == trading_mode_choice == "local":
            st.success(broker_message)

        # Today's ledger goes on (session_store): a restart or a Stop is not
        # the end of the trading day.
        prior = state.decision_tracker
        tracker = DecisionTracker(
            starting_cash=starting_budget,
            broker=live_broker,
            # A real venue applies its own costs inside the cash it reports;
            # the modelled per-trade cost belongs to the simulation only.
            trade_cost=TRADE_FIXED_COST if effective_mode == "local" else 0.0,
        )
        if continuing:
            tracker.carry_over(prior)
        else:
            # Starting over must not destroy what the day already did.
            session_store.archive()
            state.starting_budget = starting_budget
            levels = getattr(state, "apple_trader_levels", None)
            if levels:
                # The rows stay on the chart; a fresh ledger has no position
                # and no stand-down to resume.
                state.apple_trader_levels = {**levels, "memory": None}
            state.orchestra_levels = {
                key: {**record, "memory": None}
                for key, record in (getattr(state, "orchestra_levels", None) or {}).items()
            }
            if getattr(state, "orchestra", None):
                state.orchestra = {**state.orchestra, "holder": None}
        state.decision_tracker = tracker
        state.session_date = session_store.session_date()
        state.session_restored = None
        session_store.claim(state)
        if effective_mode != "local":
            # Open on the account's real balance and holdings rather than a
            # configured budget, and surface a position the app did not open
            # (left over from a previous run, or placed in Alpaca directly).
            if state.decision_tracker.sync_from_broker():
                snap = state.decision_tracker.snapshot()
                # The baseline every return percentage is measured against
                # is the account's *value*, not its cash. On an account that
                # already holds something, cash is only part of what it is
                # worth, and using it would report a return the moment the
                # agent did nothing at all. A continued day keeps the
                # baseline it opened on.
                if not continuing:
                    state.starting_budget = (
                        snap["venue_value"]
                        if snap.get("venue_value") is not None
                        else snap["cash"]
                    )
                held = {s: q for s, q in snap["positions"].items() if q}
                if held:
                    st.info(
                        "Existing positions on this account: "
                        + ", ".join(f"{s} {q:g}" for s, q in held.items())
                        + " — the agent starts from these, not from flat."
                    )
            else:
                st.warning(
                    "Could not read the account balance; the ledger starts from "
                    "the configured budget and will reconcile on the first order."
                )
        if continuing:
            snap = state.decision_tracker.snapshot()
            held = {s: q for s, q in snap["positions"].items() if q}
            append_agent_log(
                state,
                {
                    "type": "status",
                    "text": (
                        f"Continuing today's session: {len(snap['decisions'])} "
                        f"decisions so far, cash ${snap['cash']:,.2f}, "
                        + (
                            "holding " + ", ".join(f"{s} {q:g}" for s, q in held.items())
                            if held
                            else "flat"
                        )
                        + "."
                    ),
                },
            )
            if state.agent_start_time is None:
                state.agent_start_time = datetime.now(tz=timezone.utc)
        else:
            state.agent_log = []
            state.agent_start_time = datetime.now(tz=timezone.utc)
            state.agent_equity_history = []
        # How this run was started, saved with the day while it runs, so a
        # restart of the app can start it again the same way.
        state.run_spec = {
            "personality": personality,
            "provider": provider,
            "model": model,
            "symbols": list(syms),
            "trading_mode": trading_mode_choice,
            "starting_budget": float(starting_budget),
            "apple_config": asdict(apple_config) if apple_config is not None else None,
        }
        if is_orchestra:
            launch_orchestra(
                state,
                state.decision_tracker,
                apple_config,
                cycle_sec=APPLE_TRADER_CYCLE_SEC,
            )
        elif is_apple_trader:
            launch_apple_trader(
                state,
                state.decision_tracker,
                config=apple_config or AppleTraderConfig(),
                cycle_sec=APPLE_TRADER_CYCLE_SEC,
            )
        elif personality == AUTOMATIC_KEY:
            launch_automatic(
                state,
                state.decision_tracker,
                syms,
                llm_key,
                provider=provider,
                model=model or None,
                cycle_sec=AGENT_CYCLE_SEC,
            )
        else:
            launch_agent(
                state,
                state.decision_tracker,
                syms,
                llm_key,
                provider=provider,
                model=model or None,
                cycle_sec=AGENT_CYCLE_SEC,
                personality=personality,
            )
        # On disk now rather than at the next autosave: a crash in the next
        # few seconds must still find the run to start again.
        session_store.save(state)
    return stream_ready


def _agent_panel(
    symbols: list[str],
    alpaca_key: str = "",
    alpaca_secret: str = "",
    feed: str = "iex",
    data_source: str = DEFAULT_DATA_SOURCE,
    finnhub_token: str = "",
    history_feed: str = DEFAULT_HISTORY_FEED,
) -> None:
    state = _get_state()
    st.caption(
        "Runs an LLM research agent that reads already-fetched data for every streamed "
        "ticker and makes paper buy/sell/alert calls on a fixed interval, allocating one "
        "shared cash balance across the whole basket. Instead of trading only at the "
        "current price, the agent prefers to arm tactics -- standing conditional orders "
        "like 'buy 10 sh AAPL if price below X' or 'sell 20% of TSLA shares if price above Y' "
        "with multiple AND-ed conditions (price, volume, VIX, momentum, ...) -- that a "
        "background executor fills the instant they trigger, waking the agent to reevaluate. "
        "When it doesn't want to trade, the agent sets condition alerts on any ticker's "
        "continuously-updated values to wake up early the moment one is crossed, and it "
        "always wakes up early when fresh news breaks for any of its tickers. No real "
        "orders are ever placed. "
        f"Each filled buy/sell costs a fixed ${TRADE_FIXED_COST:.2f}. "
        "The exceptions are Apple Trader and Orchestra, which have no LLM at all: "
        "Apple Trader is a fixed loop over a saved day-range forecast on one "
        "instrument, and Orchestra runs those rules on several (ticker, model) pairs "
        "at once, following the first whose buy fills."
    )
    with st.expander("LLM", expanded=True):
        # Automatic first: it's the regime-adaptive orchestrator that picks and
        # switches between the individual strategies on its own; the rule-based
        # Apple Trader last, since it is the odd one out (no model reasoning).
        personality_keys = [
            AUTOMATIC_KEY, *selectable_personalities(), *RULE_AGENT_KEYS
        ]
        personality = st.selectbox(
            "Personality",
            personality_keys,
            index=personality_keys.index(state.llm_personality)
            if state.llm_personality in personality_keys
            else personality_keys.index(DEFAULT_PERSONALITY),
            format_func=_personality_label,
            key="agent_llm_personality",
        )
        state.llm_personality = personality
        if personality == AUTOMATIC_KEY:
            st.caption(
                "🤖 Automatic detects the market regime and activates the best-fitting "
                "strategy. That strategy trades until it sees no opportunities in the near "
                "term and stands down, waking Automatic to re-assess and switch. Before "
                "the session starts it activates the Premarket Analyst instead."
            )
        elif personality == PREMARKET_PERSONALITY:
            st.caption(
                "🌅 The Premarket Analyst doesn't analyze on start: it holds until "
                "~2 minutes before the opening bell, then runs one pre-market read and "
                "arms opening tactics — how much of each ticker to buy/sell and at what "
                "price for the later trades to be profitable. Once a tactic executes "
                "(simulated at the opening prints), the analyst retires and the agent "
                "disables itself."
            )
        if personality == APPLE_TRADER_KEY:
            st.caption(
                "🍎 Apple Trader runs no LLM. It trades the **day-range** model's rules: "
                "at 9:35 it forecasts where the whole session's high and low will land, "
                "then rests a buy well below the predicted high and a sell just under it "
                "for the rest of the day. It trades **one symbol**, picked below out of "
                "the ones the model was fitted on — every rule here is a model's output, "
                "so the instrument and the model constrain each other."
            )
        if personality == ORCHESTRA_KEY:
            st.caption(
                "🎼 Orchestra runs no LLM. It plays Apple Trader's rules on several "
                "**(ticker, model) pairs** at once — AAPL on the day-range model and AAPL "
                "on HighLow are two pairs. Each forecasts its own day at 9:35; the first "
                "whose buy fills is followed alone until its position is closed, then the "
                "others may buy again. A stop-out stops the agent."
            )
        if personality in RULE_AGENT_KEYS:
            # No LLM in the loop, so no LLM settings on screen. The stored
            # choice is left untouched for when an LLM personality is picked again.
            provider, model = state.llm_provider, state.llm_model
        else:
            provider = st.selectbox(
                "Provider", PROVIDERS, index=PROVIDERS.index(state.llm_provider), key="agent_llm_provider"
            )
            default_model = DEFAULT_AGENT_MODELS[provider]
            model_options = models_for(provider, default=default_model)
            current_model = state.llm_model if state.llm_model in model_options else default_model
            model = st.selectbox(
                "Model",
                model_options,
                index=model_options.index(current_model),
                # Key is provider-scoped so switching providers rebuilds the widget
                # instead of carrying a stale value that isn't in the new options.
                key=f"agent_llm_model_{provider}",
                help=f"Default: {default_model}",
            )
            state.llm_provider = provider
            state.llm_model = model
            env_var = ENV_KEYS[provider]
            if not os.getenv(env_var):
                st.caption(f"⚠️ {env_var} is not set.")

    # Each rule agent publishes its form for the chart; the other's is cleared,
    # so the chart never draws the levels of an agent that is not selected.
    if personality == APPLE_TRADER_KEY:
        apple_config = _apple_trader_params(symbols)
        state.orchestra_configs, state.orchestra_form = {}, None
    elif personality == ORCHESTRA_KEY:
        apple_config = _orchestra_params(symbols)
        state.apple_trader_config = None
    else:
        apple_config = None
        state.apple_trader_config = None
        state.orchestra_configs, state.orchestra_form = {}, None

    trading_mode_choice = _execution_controls()

    state.trade_sound_volume = st.slider(
        "Trade sound",
        min_value=0.0,
        max_value=0.60,
        value=state.trade_sound_volume,
        step=0.05,
        key="agent_trade_sound_volume",
        help="Volume of a short chime whenever an order fills — rising for a buy, "
        "falling for a sell. 0 (the default) is off. Your browser only allows "
        "sound after you interact with the page, which starting the agent satisfies.",
    )

    # A ledger from earlier today -- this run's, or one restored after a restart
    # -- is carried on by ▶ Start unless this is unticked.
    continue_today = True
    if (
        state.decision_tracker is not None
        and getattr(state, "session_date", "") == session_store.session_date()
    ):
        kept = len(state.decision_tracker.decisions)
        continue_today = st.checkbox(
            f"Continue today's session ({kept} decision{'s' if kept != 1 else ''} so far)",
            value=True,
            key="agent_continue_today",
            help="▶ Start carries on today's ledger — cash, positions and every "
            "decision — with the agent log, the equity curve and Apple Trader's "
            "levels, an open position and a stand-down. Unticked, Start opens a "
            "fresh ledger on the starting budget; today's saved session is kept "
            "beside it in data/sessions/ rather than overwritten. A different "
            "venue always starts fresh.",
        )
    st.checkbox(
        "Resume automatically after a restart",
        value=True,
        key=AUTO_RESUME_KEY,
        help="If the app goes down while the live stream or the agent is running "
        "— a crash, a hang the supervisor (run_app.py) restarted, or a restart "
        "of the server — the first page to open afterwards starts them again "
        "as they were, on the same venue, continuing today's ledger. A run "
        "that was stopped (⏹ Stop Agent, a stop-out) stays stopped. A page "
        "that only lost its connection needs none of this: it takes back the "
        "session that kept running.",
    )
    c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1.3])
    starting_budget = c1.number_input(
        "Starting budget ($)",
        min_value=0.0,
        value=PAPER_STARTING_CASH,
        step=100.0,
        key="agent_starting_budget",
        help="Only used by local simulation. On an Alpaca account this is ignored: "
        "the balance, the holdings and the portfolio value all come from that "
        "account, read from Alpaca and refreshed while the agent runs.",
        disabled=trading_mode_choice != "local",
    )
    start_clicked = c2.button("▶ Start Agent", type="primary", width='stretch', key="agent_start")
    stop_clicked = c3.button("⏹ Stop Agent", width='stretch', key="agent_stop")
    sell_all_clicked = c4.button(
        "Sell everything and stop",
        width='stretch',
        key="agent_sell_all_stop",
        disabled=state.decision_tracker is None,
        help="Stops the agent, then sells every open position at market. On an "
        "Alpaca account that is everything the account holds, not only what this "
        "session bought.",
    )

    if start_clicked:
        _start_agent(
            state,
            list(symbols or state.symbols),
            personality=personality,
            provider=provider,
            model=model,
            apple_config=apple_config,
            trading_mode_choice=trading_mode_choice,
            starting_budget=starting_budget,
            continue_today=continue_today,
            alpaca_key=alpaca_key,
            alpaca_secret=alpaca_secret,
            feed=feed,
            data_source=data_source,
            finnhub_token=finnhub_token,
            history_feed=history_feed,
        )

    if stop_clicked:
        stop_agent(state)
        # At once, not at the next autosave: a crash right after a Stop must
        # not start the stopped run again.
        session_store.save(state)

    if sell_all_clicked:
        sold, errors = sell_everything_and_stop(state)
        session_store.save(state)
        filled = [d for d in sold if d.status == "filled"]
        refused = [d for d in sold if d.status != "filled"]
        if filled:
            st.success(
                "Agent stopped and sold "
                + ", ".join(
                    f"{d.filled_quantity:g} {d.symbol} @ ${d.price:,.2f}" for d in filled
                )
                + "."
            )
        if refused or errors:
            st.error(
                "Not sold: "
                + "; ".join([f"{d.symbol} ({d.reasoning})" for d in refused] + errors)
            )
        if not sold and not errors:
            st.info("Agent stopped. There were no open positions to sell.")

    _agent_status_line()
    restored = getattr(state, "session_restored", None)
    if (
        restored
        and not state.agent_running
        and (restored.get("decisions") or restored.get("log"))
    ):
        held = restored.get("positions") or {}
        saved = str(restored.get("saved_at") or "")
        try:
            saved = pd.Timestamp(saved).tz_convert(market_hours.MARKET_TZ).strftime("%H:%M:%S ET")
        except (ValueError, TypeError):
            pass
        st.info(
            f"Restored today's session ({restored['date']}, saved {saved}): "
            f"{restored['decisions']} decisions, {restored['fills']} fills, "
            + ("holding " + ", ".join(f"{s} {q:g}" for s, q in held.items()) if held else "flat")
            + ". ▶ Start continues it."
        )
    _trade_sound_fragment()

    # A run that did not get the venue it asked for. This is the whole reason
    # the requested mode is kept: resolve_broker degrades toward simulation on
    # every misconfiguration, and saying so once in a toast leaves the dropdown
    # reading "Alpaca LIVE" over a session that is sending nothing anywhere.
    # It stays on screen until the next Start changes the answer.
    if state.trading_mode_requested and state.trading_mode != state.trading_mode_requested:
        st.warning(
            f"**Not {_venue_badge(state.trading_mode_requested)}.** This run is on "
            f"{_venue_badge(state.trading_mode)} instead — {state.trading_status}"
        )
    elif state.trading_mode == "alpaca_paper":
        st.info(
            f"📝 Orders routed to your Alpaca **paper** account. {state.trading_status}"
        )
    _agent_identity_panel()

    _agent_performance_panel(symbols)
    _agent_log_panel()
    _agent_report_section(symbols)


_BIAS_STYLE: dict[str, dict[str, str]] = {
    "bullish":  {"color": "#26c6a2", "bg": "#0d2b24", "border": "#1a4a3d", "icon": "▲"},
    "bearish":  {"color": "#ef5350", "bg": "#2b0d0d", "border": "#4a1a1a", "icon": "▼"},
    "neutral":  {"color": "#888888", "bg": "#1e1e2e", "border": "#2a2d3a", "icon": "→"},
}

_CONF_COLOR: dict[str, str] = {"high": "#26c6a2", "medium": "#fb923c", "low": "#888888"}

_IMPACT_ICON: dict[str, str] = {"positive": "↑", "negative": "↓", "neutral": "→"}


def _premarket_briefing_html(
    briefing: PremarketBriefing, symbol: str, phase: str = "premarket"
) -> str:
    # The card names the phase it was written in: a briefing generated at 11:00
    # that calls itself "Pre-Market" is claiming to be about something it isn't.
    title = PHASE_TITLES.get(phase, "Briefing")
    bias = briefing.overall_bias
    b = _BIAS_STYLE.get(bias, _BIAS_STYLE["neutral"])
    conf_color = _CONF_COLOR.get(briefing.confidence, "#888")

    # Header
    header = (
        f'<div style="background:{b["bg"]};border:1px solid {b["border"]};border-radius:10px;'
        f'padding:14px 18px;margin-bottom:12px;font-family:Inter,sans-serif;">'
        f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">'
        f'<span style="font-size:22px;font-weight:800;color:{b["color"]};letter-spacing:-0.5px;">'
        f'{b["icon"]} {bias.upper()}</span>'
        f'<span style="font-size:11px;font-weight:600;color:{conf_color};'
        f'border:1px solid {conf_color};border-radius:10px;padding:2px 8px;">'
        f'{briefing.confidence.upper()} CONFIDENCE</span>'
        f'<span style="font-size:11px;color:{PALETTE["muted"]};margin-left:auto;">{symbol}</span>'
        f'</div>'
        f'<p style="margin:0;color:{PALETTE["text"]};font-size:13px;line-height:1.6;">'
        f'{html.escape(briefing.summary)}</p>'
        f'</div>'
    )

    # Macro, after the shock flag when there is one: Apple Trader sits the
    # session out on it (`event_days`), so it is the line to read first.
    shock = getattr(briefing, "shock", "none")
    shock_label = {"geo": "Geopolitical shock day", "market": "Market shock day"}.get(shock)
    macro = (
        (
            f'<div style="background:{PALETTE["panel"]};border:1px solid #f87171;'
            f'border-radius:8px;padding:10px 14px;margin-bottom:10px;font-size:12px;'
            f'color:{PALETTE["text"]};font-family:Inter,sans-serif;">'
            f'⚡ <b>{shock_label}</b> — {html.escape(briefing.shock_reason or "flagged")}. '
            f'Apple Trader sits such sessions out by default.</div>'
        ) if shock_label else ""
    ) + (
        f'<div style="background:{PALETTE["panel"]};border:1px solid {PALETTE["grid"]};'
        f'border-radius:8px;padding:10px 14px;margin-bottom:10px;font-size:12px;'
        f'color:{PALETTE["muted"]};font-family:Inter,sans-serif;">'
        f'🌍 {html.escape(briefing.macro_context)}</div>'
    )

    # Catalysts
    catalyst_cards = []
    for c in briefing.catalysts:
        imp = c.impact
        imp_color = _IMPACT_STYLE.get(imp, _IMPACT_STYLE["unknown"])
        icon = _IMPACT_ICON.get(imp, "→")
        catalyst_cards.append(
            f'<div style="background:{imp_color["bg"]};border:1px solid {imp_color["border"]};'
            f'border-radius:8px;padding:10px 12px;font-family:Inter,sans-serif;">'
            f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:4px;">'
            f'<span style="color:{imp_color["text"]};font-weight:700;font-size:12px;">{icon}</span>'
            f'<span style="color:{PALETTE["text"]};font-weight:600;font-size:12px;">'
            f'{html.escape(c.headline)}</span>'
            f'</div>'
            f'<div style="font-size:11px;color:{PALETTE["muted"]};line-height:1.4;">'
            f'{html.escape(c.relevance)}</div>'
            f'</div>'
        )
    catalysts_section = (
        f'<div style="margin-bottom:12px;">'
        f'<div style="font-size:12px;font-weight:700;color:{PALETTE["muted"]};'
        f'letter-spacing:0.06em;text-transform:uppercase;margin-bottom:6px;">Key Catalysts</div>'
        f'<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:8px;">'
        f"{''.join(catalyst_cards)}"
        f'</div></div>'
    ) if catalyst_cards else ""

    # Technical levels
    level_rows = ""
    for lvl in briefing.technical_levels:
        role_color = "#26c6a2" if "support" in lvl.role.lower() else "#ef5350" if "resist" in lvl.role.lower() else PALETTE["accent"]
        level_rows += (
            f'<tr>'
            f'<td style="padding:5px 10px;font-weight:700;color:{role_color};">${lvl.level:.2f}</td>'
            f'<td style="padding:5px 10px;color:{PALETTE["muted"]};text-transform:capitalize;">{html.escape(lvl.role)}</td>'
            f'<td style="padding:5px 10px;color:{PALETTE["text"]};">{html.escape(lvl.note)}</td>'
            f'</tr>'
        )
    levels_section = (
        f'<div style="margin-bottom:12px;">'
        f'<div style="font-size:12px;font-weight:700;color:{PALETTE["muted"]};'
        f'letter-spacing:0.06em;text-transform:uppercase;margin-bottom:6px;">Technical Levels</div>'
        f'<table style="width:100%;border-collapse:collapse;font-family:Inter,monospace;font-size:12px;'
        f'background:{PALETTE["panel"]};border-radius:8px;overflow:hidden;">'
        f'{level_rows}</table></div>'
    ) if level_rows else ""

    # Risk factors + watch list side by side
    def _bullet_list(items: list[str], title: str, icon: str = "⚠️") -> str:
        if not items:
            return ""
        lis = "".join(
            f'<li style="margin-bottom:4px;line-height:1.4;">{html.escape(r)}</li>'
            for r in items
        )
        return (
            f'<div style="flex:1;background:{PALETTE["panel"]};border:1px solid {PALETTE["grid"]};'
            f'border-radius:8px;padding:10px 14px;font-family:Inter,sans-serif;">'
            f'<div style="font-size:12px;font-weight:700;color:{PALETTE["muted"]};'
            f'letter-spacing:0.06em;text-transform:uppercase;margin-bottom:6px;">{icon} {title}</div>'
            f'<ul style="margin:0;padding-left:16px;font-size:12px;color:{PALETTE["text"]};">{lis}</ul>'
            f'</div>'
        )

    risks = _bullet_list(briefing.risk_factors, "Risk Factors", "⚠️")
    watch = _bullet_list(briefing.key_levels_to_watch, "Watch During Session", "👁️")
    bottom_row = (
        f'<div style="display:flex;gap:10px;margin-bottom:12px;">{risks}{watch}</div>'
        if (risks or watch) else ""
    )

    return (
        f'<div style="font-family:Inter,sans-serif;padding:4px 0 16px;">'
        f'<h3 style="color:{PALETTE["text"]};font-size:14px;margin:0 0 10px 0">'
        f'🌅 {title} · <b style="color:{PALETTE["accent"]}">{symbol}</b>'
        f'</h3>'
        f'{header}{macro}{catalysts_section}{levels_section}{bottom_row}'
        f'</div>'
    )


def premarket_llm_settings(state: AppState) -> tuple[str, str, str]:
    """The provider/key/model the automatic briefing will run on.

    Read from the widgets the user sees in the Pre-Market tab (falling back to
    the stored provider and the env key), so the sidebar's ▶ Start and the tab's
    ↻ Regenerate both launch the same configuration.
    """
    provider = st.session_state.get("premarket_provider") or state.news_llm_provider
    if provider not in PROVIDERS:
        provider = state.news_llm_provider
    llm_key = os.getenv(ENV_KEYS[provider], "")
    model = st.session_state.get(f"premarket_model_{provider}") or None
    return provider, llm_key, model


def _launch_premarket(state: AppState, symbols: list[str]) -> None:
    """Kick off the automatic briefing for `symbols` on a background thread."""
    provider, llm_key, model = premarket_llm_settings(state)
    launch_premarket_analysis(
        state,
        symbols,
        provider=provider,
        api_key=llm_key,
        alpaca_key=state.api_key or os.getenv("ALPACA_API_KEY", ""),
        alpaca_secret=state.api_secret or os.getenv("ALPACA_SECRET", ""),
        worldnews_key=os.getenv("WORLD_NEWS_API_KEY", ""),
        # Same key the live tape uses; it also unlocks Finnhub's six
        # alternative-data feeds that the briefing folds in as structural context.
        finnhub_token=state.finnhub_token or os.getenv("FINNHUB_API_KEY", ""),
        model=model,
    )


@st.fragment(run_every=PREMARKET_POLL_SEC)
def _premarket_results(symbols: list[str], llm_key: str) -> None:
    """Render whatever the background briefing has produced so far.

    On its own re-run loop because the briefing arrives symbol by symbol from a
    thread the Streamlit script does not wait on: the tab has to be able to fill
    in while the user is looking at it. The Regenerate control lives in here
    too -- outside the fragment its disabled-while-busy state would freeze at
    whatever the last full script run saw, leaving the button greyed out long
    after the briefing had finished.
    """
    state = _get_state()
    briefings = state.premarket_briefings or {}
    errors = state.premarket_errors or {}
    pending = state.premarket_pending or []

    if symbols:
        # A refresh, not the old generate-on-demand step: the first briefing has
        # already run by the time this appears. It exists because the briefing is
        # a snapshot of one moment, and a trader who has been streaming since
        # before the bell needs a midday read without restarting the stream.
        if st.button(
            f"↻ Regenerate now ({', '.join(symbols)})",
            key="premarket_regenerate",
            disabled=bool(pending) or not llm_key,
            help="Re-take the briefing against the market as it stands right now.",
        ):
            _launch_premarket(state, symbols)
            st.rerun(scope="fragment")
        if pending:
            st.caption("A briefing is already running…")

    if state.premarket_generated_at is not None:
        generated_et = state.premarket_generated_at.astimezone(market_hours.MARKET_TZ)
        title = PHASE_TITLES.get(state.premarket_phase, "Briefing")
        st.caption(
            f"**{title}** — generated {generated_et.strftime('%Y-%m-%d %H:%M')} ET, "
            "from the market situation at that moment. Restart the stream (or ↻ Regenerate) "
            "for a fresh read."
        )
    elif pending:
        st.caption(f"⏳ {state.premarket_status}")
    elif state.premarket_status and state.premarket_status != "Idle":
        st.caption(state.premarket_status)

    for sym in pending:
        st.caption(f"⏳ Briefing {sym}…")
    for sym, briefing in briefings.items():
        st.html(_premarket_briefing_html(briefing, sym, state.premarket_phase))
    for sym, message in errors.items():
        st.warning(f"Briefing failed for {sym}: {message}")

    if not briefings and not pending and not errors:
        st.info(
            "The briefing is generated automatically when you press ▶ Start in the sidebar. "
            "It reflects the market situation at that moment — before the bell it is a "
            "pre-market briefing, and if you start mid-session it reports where the stock "
            "stands in today's range instead."
        )


@st.fragment(run_every=ORCHESTRA_CANDIDATES_POLL_SEC)
def _orchestra_candidates_panel() -> None:
    """Orchestra's candidates for today: the pick it made at 09:34, or -- before
    then, while Orchestra is the selected agent -- the pick as it would come
    out now, from what is known so far."""
    state = _get_state()
    today = session_store.session_date()
    decided = (getattr(state, "orchestra", None) or {}).get("selection") or {}
    form = getattr(state, "orchestra_form", None)
    rows: "list[dict]" = []
    if decided.get("date") == today and decided.get("rows"):
        rows = decided["rows"]
        st.markdown("**🎼 Orchestra's candidates for today** — picked at 09:34 ET")
    elif form is not None and form.selection is None:
        st.markdown("**🎼 Orchestra's candidates for today**")
        st.caption(
            "The selection is off in Orchestra's rules, so it races every pair all day: "
            + ", ".join(racer_label(r) for r in form.racers) + "."
        )
        return
    elif form is not None:
        st.markdown("**🎼 Orchestra's candidates for today** — provisional")
        st.caption(
            "Orchestra decides once, at 09:34 ET — four opening minutes in, one bar "
            "before the forecasts. Until then this is the pick as it would come out now: "
            "the open and the first minutes are not in before 09:30, and a briefing that "
            "is still being written counts as none."
        )
        try:
            today_date = date.fromisoformat(today)
            facts = [
                candidates.gather_facts(r, racer_label(r), state, today_date, candidates.LIVE_SOURCES)
                for r in form.racers
            ]
            rows = candidates.to_rows(candidates.select_candidates(facts, form.selection))
        except Exception as exc:  # a fetch that failed must not take the tab down
            st.warning(f"Could not work out the candidates: {exc}")
            return
    else:
        st.info(
            "Nothing to show yet. Pick **Orchestra** as the agent in the 🤖 Agent tab to "
            "see the pick as it would come out now; the one it makes at 09:34 ET stays "
            "here for the rest of the day."
        )
        return
    st.dataframe(
        pd.DataFrame([
            {
                "Pair": row["label"],
                "Races": "✓" if row.get("selected") else "—",
                "Why": row.get("reason"),
                "Bias": (
                    f"{row['bias']} ({row.get('confidence') or '?'})" if row.get("bias")
                    else row.get("briefing_note") or "—"
                ),
                "Earnings": row.get("earnings") or ("—" if row.get("earnings_known", True) else "unknown"),
                "Gap (ADR)": row.get("gap_adr"),
                "First minutes (ADR)": row.get("move_adr"),
                "Target (% of price)": row.get("target_pct"),
                "ADR": row.get("adr"),
            }
            for row in rows
        ]),
        hide_index=True,
        width="stretch",
        # Every pair on screen at once: a pick is read as a whole.
        height=36 * (len(rows) + 1) + 3,
        column_config={
            "Gap (ADR)": st.column_config.NumberColumn(
                format="%+.2f", help="Official open minus yesterday's close, in 14-day ADRs."
            ),
            "First minutes (ADR)": st.column_config.NumberColumn(
                format="%+.2f", help="09:30 to 09:33 close, from the open, in ADRs."
            ),
            "Target (% of price)": st.column_config.NumberColumn(
                format="%.2f%%",
                help="What a target exit pays — buy − sell distance, in ADRs — as a "
                "share of the price. The ranking.",
            ),
            "ADR": st.column_config.NumberColumn(format="$%.2f"),
        },
    )


def _premarket_panel(symbols: list[str]) -> None:
    state = _get_state()

    st.caption(
        "Synthesizes recent news, historical price action, macro indicators, and fundamentals "
        "into a structured briefing per symbol. Generated automatically when the data stream "
        "starts, and framed by the moment it runs in: a pre-market briefing before the bell, "
        "an intraday situation briefing once the session is underway."
    )

    c1, c2 = st.columns([1, 2])
    with c1:
        provider = st.selectbox(
            "Provider",
            PROVIDERS,
            index=PROVIDERS.index(state.news_llm_provider),
            key="premarket_provider",
            help=f"Model: {', '.join(f'{p}={m}' for p, m in DEFAULT_PREMARKET_MODELS.items())}",
        )
        state.news_llm_provider = provider
        env_var = ENV_KEYS[provider]
        llm_key = os.getenv(env_var, "")
        if not llm_key:
            st.caption(f"⚠️ {env_var} is not set.")
        default_model = DEFAULT_PREMARKET_MODELS.get(provider, DEFAULT_NEWS_MODELS[provider])
        premarket_models = models_for(provider, default=default_model)
        st.selectbox(
            "Model",
            premarket_models,
            index=0,
            key=f"premarket_model_{provider}",
            help=f"Default: {default_model}. Applies to the next briefing — the one ▶ Start "
            "launches, or ↻ Regenerate below.",
        )
    with c2:
        if not symbols:
            st.info("Enter symbols in the sidebar first.")

    _premarket_results(symbols, llm_key)


def _start_live_session(
    state: AppState,
    syms: list[str],
    key: str,
    secret: str,
    feed: str,
    timeframe: str,
    data_source: str = DEFAULT_DATA_SOURCE,
    finnhub_token: str = "",
    history_feed: str = DEFAULT_HISTORY_FEED,
) -> bool:
    """Load history for every symbol and launch the bars + news streams.

    Shared by the sidebar ▶ Start and the agent panel's ▶ Start Agent (which
    starts the stream itself when it isn't running yet). Returns True when the
    streams were launched, False when loading any symbol failed.

    History, news and daily bars come from Alpaca REST whichever live source is
    chosen (news from Yahoo Finance too); `data_source` only decides which
    WebSocket takes over from there.

    A symbol that already has bars at this timeframe and history feed (a Stop
    followed by a Start) keeps them: the fetched history is merged in, filling
    the gap the stop left, and replacing only provisional bars."""
    prior_timeframe = state.timeframe
    prior_history_feed = state.history_feed_resolved
    state.set_symbols(syms)
    state.feed = feed
    state.timeframe = timeframe
    state.api_key = key
    state.api_secret = secret
    state.finnhub_token = finnhub_token
    state.history_feed = history_feed
    # Resolved once, here, and reused by the backfill and the fallback poll: the
    # whole buffer has to carry one set of volume units (see bar_history).
    state.history_feed_resolved = bar_history.resolve_history_feed(
        history_feed, syms[0], key, secret, timeframe
    )
    loaded: list[str] = []
    for sym in syms:
        sym_state = state.sym(sym)
        with st.spinner(f"Loading history for {sym}…"):
            try:
                historical = bar_history.fetch_live_bars(
                    sym, timeframe, key, secret, state.history_feed_resolved,
                    limit=MAX_BARS, what="bars (initial load)",
                )
                historical_trades = fetch_trades(sym, key, secret, feed)
                log_fetch(
                    "trades (initial load)", "Alpaca REST", symbol=sym,
                    detail=f"{len(historical_trades)} trades",
                )
                # Alpaca's and Yahoo Finance's, merged; the stream keeps
                # both coming (see launch_stream_news).
                news = fetch_live_news(
                    sym, key, secret, os.getenv("WORLD_NEWS_API_KEY", "")
                )
                # Same feed as the intraday bars: this series is the baseline
                # every volume comparison divides by, so an IEX daily average
                # under a consolidated intraday series reports an ordinary
                # session as 30-40x normal participation.
                daily_bars, _ = bar_history.fetch_daily(
                    sym, key, secret, state.history_feed_resolved
                )
            except Exception as exc:
                log_fetch_failure(
                    "initial data load", [("Alpaca REST", exc)], symbol=sym,
                    consequence="start aborted",
                )
                st.error(f"Failed to load data for {sym}: {exc}")
                state.status = f"Failed ({sym}): {exc}"
                return False
            sym_state.daily_bars = daily_bars
            with sym_state.lock:
                cached = bool(sym_state.bars)
            reuse = (
                cached
                and prior_timeframe == timeframe
                and prior_history_feed == state.history_feed_resolved
            )
            if reuse:
                added, replaced = stream_common.merge_live_bars(
                    sym_state, historical.bars, historical.provisional
                )
                detail = (
                    f"cached buffer kept; {added} {timeframe} bar(s) added, "
                    f"{replaced} provisional replaced"
                )
            else:
                stream_common.load_live_bars(
                    sym_state, historical.bars, historical.provisional
                )
                detail = f"{len(historical.bars)} {timeframe} bars"
            log_fetch(
                "bars (initial load)", historical.source, symbol=sym,
                detail=detail, failures=historical.failures,
            )
            with sym_state.lock:
                sym_state.trades.clear()
                sym_state.trades.extend(historical_trades)
            sym_state.news = news
            sym_state.news_impacts = {}
            sym_state.news_impact_details = {}
            loaded.append(sym)

    # A symbol with its own news-impact model is scored by it, on a background
    # thread (its first run downloads weeks of minute bars, which must not hold
    # up the stream); the rest go to the LLM.
    model_syms = [
        sym for sym in loaded
        if newsimpact_model.uses_model(sym, state.news_impact_method)
    ]
    for sym in model_syms:
        if state.sym(sym).news:
            newsimpact_model.launch_refresh(
                state.sym(sym), key, secret, state.history_feed_resolved, force=True
            )
    # Once per ET day per ticker: the first start computes it from last week's
    # minute bars and stores it, later starts read it back.
    minute_momentum.launch_refresh(state.sym(sym) for sym in loaded)
    news_llm_provider = state.news_llm_provider
    llm_key = os.getenv(ENV_KEYS[news_llm_provider], "")
    if llm_key:
        for sym in loaded:
            sym_state = state.sym(sym)
            if not sym_state.news or sym in model_syms:
                continue
            try:
                sym_state.news_impacts = score_news_impacts(
                    sym, sym_state.news, news_llm_provider, llm_key
                )
            except Exception as exc:
                state.status = f"News impact scoring failed for {sym}: {exc}"
    state.status = "Connecting WebSocket…"
    session_id = _session_id()
    if session_id:
        register_live_session(session_id, state)
    launch_stream(
        syms, key, secret, feed, state, timeframe,
        data_source=data_source, finnhub_token=finnhub_token,
        history_feed=state.history_feed_resolved,
    )
    launch_stream_news(
        syms, key, secret, state, worldnews_key=os.getenv("WORLD_NEWS_API_KEY", "")
    )
    # The briefing rides the same click that starts the tape, on its own thread:
    # it is several seconds of LLM work per symbol and the stream must not wait
    # for it. What it says is anchored to *this* moment -- before the bell that
    # is a pre-market briefing, mid-session it reports where the stock stands in
    # today's range instead (see agent_stonks.premarket).
    _launch_premarket(state, syms)
    return True


# --- recovery after a dropped connection or a restart ---------------------------

# The Agent tab's "Resume automatically after a restart" box (kept in last_setup).
AUTO_RESUME_KEY = "agent_auto_resume"
# How often a resume that could not start (no network yet) is tried again.
RESUME_RETRY_SEC = 30
# How long the banner saying what was recovered stays above the tabs.
RECOVERY_BANNER_SEC = 10 * 60


def _apple_config_from(raw: "dict | None") -> "AppleTraderConfig | OrchestraConfig | None":
    """The rule agent's configuration a run was started with, back from its
    saved `run_spec`: Apple Trader's, or Orchestra's (`{"racers": [...]}`)."""
    if not raw:
        return None
    known = {f.name for f in fields(AppleTraderConfig)}
    if "racers" in raw:
        return OrchestraConfig([
            AppleTraderConfig(**{k: v for k, v in racer.items() if k in known})
            for racer in raw["racers"]
        ])
    return AppleTraderConfig(**{k: v for k, v in raw.items() if k in known})


def _run_label(run: dict) -> str:
    """'the live stream and Apple Trader' -- what `run` started, for the banner."""
    parts = []
    if run.get("stream"):
        parts.append(f"the live stream ({', '.join(run['stream'].get('symbols') or [])})")
    if run.get("agent"):
        parts.append(_personality_label(run["agent"].get("personality", "")))
    return " and ".join(parts) or "nothing"


def _resume_after_restart(
    state: AppState, alpaca_key: str, alpaca_secret: str, finnhub_token: str
) -> None:
    """Start again what the app went down in the middle of.

    The day's file says what was running when the process last saved
    (`session_store.take_resume`): the live stream, the agent, or both. The
    first session after a restart starts them as they were -- the agent on the
    same venue, continuing today's ledger, Apple Trader taking back its open
    position and stand-down -- unless the Agent tab's "Resume automatically
    after a restart" is unticked. A stream that cannot start yet (whatever
    took the app down may have taken the network with it) is tried again
    every RESUME_RETRY_SEC; an agent the setup refuses is reported, not retried.
    """
    if not st.session_state.get(AUTO_RESUME_KEY, True):
        if (state.recovery or {}).get("pending"):
            state.recovery = None
        return
    pending = (state.recovery or {}).get("pending")
    if pending is not None and time.time() < (state.recovery or {}).get("retry_at", 0):
        return
    run = pending or session_store.take_resume(state)
    if not run:
        return
    attempts = (state.recovery or {}).get("attempts", 0) + 1 if pending else 1
    stream, agent = run.get("stream") or {}, run.get("agent")
    key = alpaca_key.strip() or os.getenv("ALPACA_API_KEY", "")
    secret = alpaca_secret.strip() or os.getenv("ALPACA_SECRET", "")
    if not key or not secret:
        state.recovery = {
            "kind": "restart", "at": time.time(), "run": run,
            "error": "the Alpaca API key and secret are not in the environment "
            "(ALPACA_API_KEY / ALPACA_SECRET), so nothing was started — enter them "
            "and press ▶ Start.",
        }
        return
    connection = {
        "feed": stream.get("feed") or "iex",
        "data_source": stream.get("data_source") or DEFAULT_DATA_SOURCE,
        "history_feed": stream.get("history_feed") or DEFAULT_HISTORY_FEED,
    }
    error = ""
    started = True
    if stream:
        try:
            started = _start_live_session(
                state, list(stream["symbols"]), key, secret, connection["feed"],
                stream.get("timeframe") or TIMEFRAMES[0],
                data_source=connection["data_source"], finnhub_token=finnhub_token,
                history_feed=connection["history_feed"],
            )
        except Exception as exc:
            logging.getLogger(__name__).exception("Resuming the live stream failed")
            started, error = False, f"{type(exc).__name__}: {exc}"
        if not started:
            state.recovery = {
                "kind": "restart", "at": time.time(), "run": run, "pending": run,
                "attempts": attempts, "retry_at": time.time() + RESUME_RETRY_SEC,
                "error": error or state.status,
            }
            # The day's run is handed out once per process, to this state: a
            # reload has to find it again (adopt_orphaned_session) to go on trying.
            session_id = _session_id()
            if session_id:
                register_live_session(session_id, state)
            return
    if agent:
        try:
            started = _start_agent(
                state,
                list(agent.get("symbols") or stream.get("symbols") or []),
                personality=agent.get("personality") or DEFAULT_PERSONALITY,
                provider=agent.get("provider") or state.llm_provider,
                model=agent.get("model") or "",
                apple_config=_apple_config_from(agent.get("apple_config")),
                trading_mode_choice=agent.get("trading_mode") or DEFAULT_TRADING_MODE,
                starting_budget=float(agent.get("starting_budget") or PAPER_STARTING_CASH),
                continue_today=True,
                alpaca_key=key,
                alpaca_secret=secret,
                finnhub_token=finnhub_token,
                **connection,
            )
        except Exception as exc:
            logging.getLogger(__name__).exception("Resuming the agent failed")
            started, error = False, f"{type(exc).__name__}: {exc}"
        if started:
            append_agent_log(state, {"type": "status", "text": (
                f"{_personality_label(agent.get('personality', ''))} started again "
                "automatically: the app restarted while it was running."
            )})
        else:
            error = error or "the agent's setup was refused — see the message above."
    state.recovery = {"kind": "restart", "at": time.time(), "run": run, "error": error}


@st.fragment(run_every=WATCHDOG_BEAT_SEC)
def _page_watchdog() -> None:
    """The heartbeat of the page's own watchdog (see page_watchdog), which
    reloads this tab when Streamlit's frontend has died under a healthy server.
    The first page after such a reload says why; the full run draws that in
    the recovery banner."""
    report = page_watchdog()
    if report:
        state = _get_state()
        recovery = state.recovery or {"kind": "reconnect", "at": time.time()}
        state.recovery = {**recovery, "reload": report}
        st.rerun(scope="app")


@st.fragment(run_every=RESUME_RETRY_SEC)
def _resume_retry_timer() -> None:
    """Rerun the whole app once a pending resume is due to be tried again --
    only a full run can start a stream. Drawn only while one is pending."""
    recovery = _get_state().recovery or {}
    if recovery.get("pending") and time.time() >= recovery.get("retry_at", 0):
        st.rerun(scope="app")


def _recovery_banner(state: AppState) -> None:
    """Say, above the tabs, what came back after a dropped connection or a
    restart, for RECOVERY_BANNER_SEC -- or that it has not yet."""
    recovery = state.recovery
    if not recovery:
        return
    run = recovery.get("run") or {}
    if recovery.get("pending"):
        st.warning(
            f"♻️ The app restarted while {_run_label(run)} was running. Starting it "
            f"again failed ({recovery.get('error') or 'unknown error'}); trying again "
            f"every {RESUME_RETRY_SEC} s (attempt {recovery.get('attempts', 1)}). "
            "▶ Start or ⏹ Stop takes over."
        )
        _resume_retry_timer()
        return
    if time.time() - recovery.get("at", 0) > RECOVERY_BANNER_SEC:
        return
    if recovery["kind"] == "reconnect":
        running = [
            label for label, on in (
                ("the live stream", session_store.is_streaming(state)),
                (_personality_label(state.llm_personality), state.agent_running),
            ) if on
        ]
        reload = recovery.get("reload")
        what = (
            f"This page reloaded itself because {describe_reload(reload)}"
            if reload else "The connection to the app dropped and came back"
        )
        if running:
            st.info(
                f"🔌 {what}. This page took over the session that kept running "
                f"meanwhile: {' and '.join(running)} never stopped."
            )
        elif reload:
            st.info(f"🔌 {what}.")
    elif recovery.get("error"):
        st.error(
            f"♻️ The app restarted while {_run_label(run)} was running, and could not "
            f"start all of it again: {recovery['error']}"
        )
    else:
        st.success(
            f"♻️ The app restarted while {_run_label(run)} was running, and started it "
            "again" + (" — continuing today's ledger" if run.get("agent") else "")
            + ". Untick \"Resume automatically after a restart\" in the Agent tab to "
            "leave it stopped next time."
        )


@contextmanager
def _panel_guard(label: str):
    """Keep one tab's failure from taking the whole page down.

    An exception escaping a tab used to end the script run where it was raised:
    every tab after it -- the Agent tab among them -- was left undrawn behind a
    traceback, and a dropped connection to any one data source was enough. The
    error is shown in its own tab instead and the run goes on. Streamlit's own
    rerun/stop signals are BaseExceptions and pass through."""
    try:
        yield
    except Exception as exc:
        logging.getLogger(__name__).exception("%s failed to draw", label)
        st.warning(
            f"⚠️ {label} hit an error and was skipped this time ({type(exc).__name__}: "
            f"{exc}). The rest of the app, the stream and the agent are not affected; "
            "it is drawn again on the next refresh."
        )
        with st.expander("Error details"):
            st.exception(exc)


def build_ui() -> None:
    st.set_page_config(
        page_title="Agent Stonks",
        page_icon="📈",
        layout="wide",
    )
    # The setup the last session left -- symbols, connection, chart settings,
    # personality, Apple Trader's rules, venue -- before any widget is drawn,
    # so a restart or a reconnect opens where it was (see last_setup).
    last_setup.restore()

    with st.sidebar:
        st.header("Controls")
        c1, c2 = st.columns(2)
        start_clicked = c1.button("▶ Start", type="primary", width='stretch')
        stop_clicked = c2.button("⏹ Stop", width='stretch')
        symbols_input = st.text_input(
            "Symbols",
            value="AAPL",
            key="sidebar_symbols",
            placeholder="AAPL, TSLA, MSFT…",
            help="One or more tickers, comma- or space-separated. All live plots, "
            "analyses, and the trading agent cover every listed symbol.",
        )
        with st.expander("Connection"):
            live_source = st.selectbox(
                "Live data source",
                list(LIVE_SOURCES),
                index=list(LIVE_SOURCES).index(DEFAULT_LIVE_SOURCE),
                format_func=lambda s: LIVE_SOURCE_LABELS.get(s, s),
                key="sidebar_live_source",
                help=(
                    "Which WebSocket fills the live bar series, and — for Alpaca — which "
                    "of its feeds. **Finnhub** streams the consolidated trade tape and the "
                    "candles are built from it here, so the newest candle is the minute in "
                    "progress rather than the last one to close; it needs FINNHUB_API_KEY. "
                    "**Alpaca** streams ready-made bars off the named feed and is the only "
                    "one of the two that also streams bid/ask.\n\n"
                    "The feed travels further than the socket: the bid/ask quote poll and "
                    "the price every agent fills at read it whichever source is streaming, "
                    "so Finnhub rides on IEX quotes — the one Alpaca feed served on every "
                    "plan. Alpaca credentials are required either way, because the REST "
                    "fallback, the bar backfill and (under Finnhub) the quote poll all run "
                    "on them. Historical bars have their own setting below."
                ),
            )
            data_source, feed = LIVE_SOURCES[live_source]
            history_feed = st.selectbox(
                "History / backfill source",
                HISTORY_FEEDS,
                index=0,
                key="sidebar_history_feed",
                help=(
                    "Where REST bars come from — the initial history load, the timeframe "
                    "reload, the periodic backfill and the stream-down fallback poll, which "
                    "all fill the same buffer the live socket fills.\n\n"
                    "**IEX carries under 4% of consolidated volume** (measured on AAPL: 1.56M "
                    "vs 41.6M shares over the same 390 minutes), so IEX history next to a "
                    "consolidated live stream puts a ~26x volume step mid-series that "
                    "relative volume, the volume profile and the models' volume features all "
                    "sum across.\n\n"
                    "**auto** (recommended) uses Alpaca SIP — real-time on a paid plan, or "
                    "held back 16 minutes on a free/basic plan, which refuses only the "
                    "trailing 15 minutes. Delayed SIP is still the right backfill source: "
                    "backfill repairs *holes*, and the live stream already owns the recent "
                    "window. Failing that it uses yfinance (within 1.5% of SIP, free, ~15 min "
                    "delayed, ~7 days of minute history), and IEX only when neither answers.\n\n"
                    "Whatever is chosen here, the live buffer takes regular-session bars "
                    "older than 15 minutes from yfinance, and this source serves the rest "
                    "(pre/post-market, where yfinance reports zero volume). The last 15 "
                    "minutes come from IEX and are replaced once they settle; the minute in "
                    "progress is the live stream's. A Stop then Start keeps the buffered "
                    "bars and fills only the gap."
                ),
            )
            finnhub_token_input = st.text_input(
                "Finnhub API Key",
                type="password",
                placeholder="From env FINNHUB_API_KEY if blank",
            )
            api_key = st.text_input(
                "Alpaca API Key",
                type="password",
                placeholder="From env ALPACA_API_KEY if blank",
            )
            api_secret = st.text_input(
                "Alpaca Secret",
                type="password",
                placeholder="From env ALPACA_SECRET if blank",
            )
        # Invisible; ahead of the tabs, so nothing they draw can keep the page
        # from watching itself.
        _page_watchdog()
    finnhub_token = finnhub_token_input.strip() or os.getenv("FINNHUB_API_KEY", "")
    state = _get_state()
    with _panel_guard("Restart recovery"):
        _resume_after_restart(state, api_key, api_secret, finnhub_token)
        _recovery_banner(state)
    symbols = _effective_symbols(state, symbols_input)

    (
        tab_agent, tab_live, tab_news, tab_premarket, tab_candidates, tab_analysis,
        tab_walls, tab_models,
    ) = st.tabs(
        [
            "🤖 Agent", "📡 Live", "📰 News", "🌅 Pre-Market", "🎼 Candidates",
            "🔬 Technical Analysis", "🧱 Put/Call Walls", "🧠 ML Models",
        ]
    )

    with tab_live, _panel_guard("The Live tab"):
        st.caption(
            "Live candles are built locally from the Finnhub trade tape by default, so the "
            "newest candle is the minute in progress. Bid/ask, the bar backfill and the "
            "stream-down fallback still come from Alpaca REST — and on a free Alpaca "
            "account the IEX feed only serves US market hours (9:30–16:00 ET). "
            "Switch sources in the sidebar's Connection expander."
        )
        timeframe = st.session_state.get("live_timeframe", TIMEFRAMES[0])

        timeframe_changed = (
            state.symbols
            and state.api_key
            and timeframe != state.timeframe
            and not start_clicked
            and not stop_clicked
        )
        if timeframe_changed:
            with st.spinner(f"Reloading {', '.join(state.symbols)} at {timeframe}…"):
                reloaded = True
                for sym_state in state.iter_symbol_states():
                    sym = sym_state.symbol
                    try:
                        historical = bar_history.fetch_live_bars(
                            sym, timeframe, state.api_key, state.api_secret,
                            bar_history.resolve_history_feed(
                                state.history_feed, sym, state.api_key,
                                state.api_secret, timeframe,
                            ),
                            limit=MAX_BARS, what="bars (timeframe reload)",
                        )
                    except Exception as exc:
                        log_fetch_failure(
                            "bars (timeframe reload)", [("Alpaca REST", exc)], symbol=sym,
                            consequence=f"staying on {state.timeframe}",
                        )
                        st.error(f"Failed to reload bars for {sym}: {exc}")
                        reloaded = False
                        break
                    log_fetch(
                        "bars (timeframe reload)", historical.source, symbol=sym,
                        detail=f"{len(historical.bars)} {timeframe} bars",
                        failures=historical.failures,
                    )
                    stream_common.load_live_bars(
                        sym_state, historical.bars, historical.provisional
                    )
                if reloaded:
                    state.timeframe = timeframe
                    launch_stream(
                        list(state.symbols), state.api_key, state.api_secret,
                        state.feed, state, timeframe,
                        data_source=state.data_source,
                        finnhub_token=state.finnhub_token,
                    )

        if start_clicked:
            syms = _parse_symbols(symbols_input)
            key = api_key.strip() or os.getenv("ALPACA_API_KEY", "")
            secret = api_secret.strip() or os.getenv("ALPACA_SECRET", "")

            if not syms:
                st.error("Please enter at least one symbol.")
            elif not key or not secret:
                st.error("API key and secret are required.")
            else:
                _start_live_session(
                    state, syms, key, secret, feed, timeframe,
                    data_source=data_source, finnhub_token=finnhub_token,
                    history_feed=history_feed,
                )

        if stop_clicked:
            stop_streams(state)
            session_store.save(state)

        _live_panel()

    with tab_news, _panel_guard("The News tab"):
        _news_panel(symbols)

    with tab_premarket, _panel_guard("The Pre-Market tab"):
        _premarket_panel(symbols)

    with tab_analysis, _panel_guard("The Technical Analysis tab"):
        _technical_analysis_panel(symbols)

    with tab_walls, _panel_guard("The Put/Call Walls tab"):
        _options_walls_panel(symbols)

    with tab_agent, _panel_guard("The Agent tab"):
        _agent_panel(
            symbols,
            alpaca_key=api_key,
            alpaca_secret=api_secret,
            feed=feed,
            data_source=data_source,
            finnhub_token=finnhub_token,
            history_feed=history_feed,
        )

    # After the Agent tab: Orchestra's form there publishes the setup the
    # provisional pick is worked out from.
    with tab_candidates, _panel_guard("The Candidates tab"):
        _orchestra_candidates_panel()

    with tab_models, _panel_guard("The ML Models tab"):
        model_catalogue_panel()

    last_setup.remember()
