import html
import os
import re
from typing import Optional

import pandas as pd
import streamlit as st

from .charts import build_chart, empty_chart
from .config import CHART_POLL_SEC, FEEDS, MAX_BARS, PALETTE, POLL_SEC, SESSION_START, TIMEFRAMES
from .news import score_news_impacts
from .rest import fetch_bars, fetch_daily_bars, fetch_news, fetch_trades
from .state import AppState
from .stream import launch_stream, launch_stream_news


def _get_state() -> AppState:
    if "app_state" not in st.session_state:
        st.session_state["app_state"] = AppState()
    return st.session_state["app_state"]


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
    "positive": {"label": "positive impact", "dot": "#26c6a2", "bg": "#0d2b24", "border": "#1a4a3d", "text": "#26c6a2"},
    "negative": {"label": "negative impact", "dot": "#ef5350", "bg": "#2b0d0d", "border": "#4a1a1a", "text": "#ef5350"},
    "neutral":  {"label": "neutral impact",  "dot": "#888",    "bg": "#1e1e2e", "border": "#2a2d3a", "text": "#888888"},
    "small":    {"label": "small impact",    "dot": "#fb923c", "bg": "#2b1a0d", "border": "#4a2d1a", "text": "#fb923c"},
    "unknown":  {"label": "unknown impact",  "dot": "#555",    "bg": "#1a1d27", "border": "#2a2d3a", "text": "#555555"},
}


def _impact_badge(impact: str) -> str:
    cfg = _IMPACT_STYLE.get(impact, _IMPACT_STYLE["unknown"])
    return (
        f'<span style="display:inline-flex;align-items:center;gap:5px;'
        f'padding:3px 9px;border-radius:12px;background:{cfg["bg"]};'
        f'border:1px solid {cfg["border"]};font-size:10px;font-weight:600;'
        f'color:{cfg["text"]};white-space:nowrap;letter-spacing:0.02em;">'
        f'<span style="width:6px;height:6px;border-radius:50%;'
        f'background:{cfg["dot"]};display:inline-block;flex-shrink:0;"></span>'
        f'{cfg["label"]}</span>'
    )


def _news_html(news: list[dict], symbol: str, impacts: Optional[dict] = None) -> str:
    if not news:
        return (
            f"<p style='color:{PALETTE['muted']};padding:12px'>"
            f"No recent news for {symbol}.</p>"
        )
    impacts = impacts or {}
    cards = []
    for item in news[:12]:
        ts = pd.to_datetime(item.get("created_at")).strftime("%b %d  %H:%M")
        src = html.escape(item.get("source", ""))
        headline = html.escape(_strip_html(item.get("headline", "")))
        summary = _strip_html(item.get("summary") or "")[:180].rstrip()
        summary = html.escape(summary)
        url = html.escape(item.get("url", "#"))
        news_id = str(item.get("id", ""))
        impact = impacts.get(news_id)
        badge = _impact_badge(impact) if impact else _impact_badge("unknown")
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


def _quote_html(
    price: float | None,
    prev_close: float | None,
    bid: float | None,
    bid_size: float | None,
    ask: float | None,
    ask_size: float | None,
    symbol: str,
) -> str:
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

    price_row = ""
    if price is not None:
        price_row = (
            f'<div style="display:flex;align-items:baseline;gap:12px;margin-bottom:10px;">'
            f'<span style="font-size:28px;font-weight:700;color:{PALETTE["text"]};'
            f'letter-spacing:-0.5px;">${price:,.4f}</span>'
            f'<span style="font-size:14px;font-weight:600;color:{chg_color};">'
            f'{arrow} {delta_str}</span>'
            f'</div>'
        )

    def _side(label: str, p: float | None, sz: float | None, color: str) -> str:
        if p is None:
            return ""
        size_str = f'<span style="font-size:11px;color:{PALETTE["muted"]};margin-left:4px;">{sz:,.0f}</span>' if sz else ""
        return (
            f'<div style="display:flex;flex-direction:column;align-items:center;'
            f'background:{PALETTE["panel"]};border:1px solid {PALETTE["grid"]};'
            f'border-radius:8px;padding:8px 16px;min-width:100px;">'
            f'<span style="font-size:10px;font-weight:600;color:{PALETTE["muted"]};'
            f'letter-spacing:0.08em;text-transform:uppercase;margin-bottom:4px;">{label}</span>'
            f'<span style="font-size:18px;font-weight:700;color:{color};">${p:,.4f}</span>'
            f'{size_str}'
            f'</div>'
        )

    bid_card = _side("Bid", bid, bid_size, "#ef5350")
    ask_card = _side("Ask", ask, ask_size, "#26c6a2")
    spread_row = ""
    if bid is not None and ask is not None:
        spread = ask - bid
        spread_row = (
            f'<span style="font-size:11px;color:{PALETTE["muted"]};align-self:center;">'
            f'spread {spread:.4f}</span>'
        )

    ba_row = ""
    if bid_card or ask_card:
        ba_row = (
            f'<div style="display:flex;gap:10px;align-items:stretch;">'
            f'{bid_card}{spread_row}{ask_card}'
            f'</div>'
        )

    return (
        f'<div style="font-family:Inter,monospace;padding:4px 0 8px;">'
        f'{price_row}{ba_row}'
        f'</div>'
    )


@st.fragment(run_every=POLL_SEC)
def _price_ticker() -> None:
    state = _get_state()
    st.caption(f"Status: {state.status}")
    if state.symbol:
        with state.lock:
            last_price = state.last_price
            prev_close = state.prev_close
            bid_price = state.bid_price
            bid_size = state.bid_size
            ask_price = state.ask_price
            ask_size = state.ask_size
        html = _quote_html(last_price, prev_close, bid_price, bid_size, ask_price, ask_size, state.symbol)
        if html:
            st.html(html)


@st.fragment(run_every=CHART_POLL_SEC)
def _chart_panel() -> None:
    state = _get_state()
    with state.lock:
        bars = list(state.bars)

    fig = (
        build_chart(
            bars,
            state.news,
            state.trades,
            state.symbol,
            SESSION_START,
            ma_periods=state.ma_periods,
            show_fib=state.show_fib,
            show_7d_avg=state.show_7d_avg,
            show_28d_avg=state.show_28d_avg,
            show_1y_avg=state.show_1y_avg,
            gaussian_max_components=state.gaussian_max_components,
            show_gaussian_centers=state.show_gaussian_centers,
            daily_bars=state.daily_bars,
            vwap_style=state.vwap_style,
            show_candle_body=state.show_candle_body,
            show_percentile_body=state.show_percentile_body,
            show_whiskers=state.show_whiskers,
        )
        if (state.symbol and bars)
        else empty_chart()
    )
    st.plotly_chart(fig, use_container_width=True)
    st.html(_news_html(state.news, state.symbol, state.news_impacts))


def _live_panel() -> None:
    _price_ticker()
    _chart_panel()


def build_ui() -> None:
    st.set_page_config(
        page_title="Market Stream",
        page_icon="📈",
        layout="wide",
    )

    st.markdown(
        f"<h1 style='color:{PALETTE['text']};font-family:Inter,sans-serif;margin:0 0 4px'>"
        "📈 Market Stream</h1>"
        f"<p style='color:{PALETTE['muted']};font-size:13px;margin:0'>"
        "Real-time candlestick bars &amp; news via Alpaca streaming API</p>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Controls")
        c1, c2 = st.columns(2)
        start_clicked = c1.button("▶ Start", type="primary", use_container_width=True)
        stop_clicked = c2.button("⏹ Stop", use_container_width=True)
        symbol = st.text_input("Symbol", value="AAPL", placeholder="AAPL, TSLA, MSFT…")
        timeframe = st.selectbox("Timeframe", TIMEFRAMES, index=0)
        with st.expander("Connection"):
            feed = st.selectbox("Feed", FEEDS, index=0)
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
        st.subheader("Candle")
        show_candle_body = st.checkbox("Open-Close", value=True)
        show_percentile_body = st.checkbox("20%-80%", value=False)
        show_whiskers = st.checkbox("Whiskers", value=True)
        vwap_style = st.selectbox("VWAP", ["hide", "dot", "line"], index=0)

        st.subheader("Overlays")
        vwma_selection = st.multiselect(
            "VWMA",
            ["VWMA(5)", "VWMA(15)", "VWMA(60)"],
            default=[],
        )
        avg_selection = st.multiselect(
            "Average Lines",
            ["7d Avg", "28d Avg", "1y Avg"],
            default=[],
        )
        show_fib = st.checkbox("Fibonacci levels", value=False)

        with st.expander("Price Profile Fit"):
            fit_enabled = st.checkbox("Fit Gaussian mixture", value=False)
            max_components = st.slider(
                "Components",
                min_value=1,
                max_value=5,
                value=1,
                disabled=not fit_enabled,
            )
            show_gaussian_centers = st.checkbox(
                "Show centers on candle chart",
                value=False,
                disabled=not fit_enabled,
            )


    state = _get_state()
    state.ma_periods = _parse_ma_periods(vwma_selection)
    state.show_7d_avg, state.show_28d_avg, state.show_1y_avg = _parse_avg_flags(avg_selection)
    state.show_candle_body = show_candle_body
    state.show_percentile_body = show_percentile_body
    state.show_whiskers = show_whiskers
    state.vwap_style = vwap_style
    state.show_fib = show_fib
    state.gaussian_max_components = max_components if fit_enabled else 0
    state.show_gaussian_centers = show_gaussian_centers if fit_enabled else False

    timeframe_changed = (
        state.symbol
        and state.api_key
        and timeframe != state.timeframe
        and not start_clicked
        and not stop_clicked
    )
    if timeframe_changed:
        with st.spinner(f"Reloading {state.symbol} at {timeframe}…"):
            try:
                historical_bars = fetch_bars(state.symbol, timeframe, MAX_BARS, state.api_key, state.api_secret, state.feed)
            except Exception as exc:
                st.error(f"Failed to reload bars: {exc}")
            else:
                state.timeframe = timeframe
                with state.lock:
                    state.bars.clear()
                    state.bars.extend(historical_bars)
                launch_stream(state.symbol, state.api_key, state.api_secret, state.feed, state, timeframe)

    if start_clicked:
        sym = symbol.strip().upper()
        key = api_key.strip() or os.getenv("ALPACA_API_KEY", "")
        secret = api_secret.strip() or os.getenv("ALPACA_SECRET", "")

        if not sym:
            st.error("Please enter a symbol.")
        elif not key or not secret:
            st.error("API key and secret are required.")
        else:
            with st.spinner(f"Loading history for {sym}…"):
                try:
                    historical_bars = fetch_bars(sym, timeframe, MAX_BARS, key, secret, feed)
                    historical_trades = fetch_trades(sym, key, secret, feed)
                    news = fetch_news(sym, key, secret)
                    daily_bars = fetch_daily_bars(sym, key, secret, feed)
                except Exception as exc:
                    st.error(f"Failed to load data: {exc}")
                    state.status = f"Failed: {exc}"
                else:
                    state.symbol = sym
                    state.feed = feed
                    state.timeframe = timeframe
                    state.api_key = key
                    state.api_secret = secret
                    state.daily_bars = daily_bars
                    with state.lock:
                        state.bars.clear()
                        state.bars.extend(historical_bars)
                        state.trades.clear()
                        state.trades.extend(historical_trades)
                    state.news = news
                    state.news_impacts = {}
                    gemini_key = os.getenv("GEMINI_API_KEY", "")
                    if gemini_key and news:
                        try:
                            state.news_impacts = score_news_impacts(sym, news, gemini_key)
                        except Exception as exc:
                            state.status = f"News impact scoring failed: {exc}"
                    state.status = "Connecting WebSocket…"
                    launch_stream(sym, key, secret, feed, state, timeframe)
                    launch_stream_news(sym, key, secret, state)

    if stop_clicked:
        if state.ws:
            try:
                state.ws.close()
            except Exception:
                pass
        state.status = "Stopped"

    st.caption(
        "⚠️ Free Alpaca accounts: IEX feed available during US market hours (9:30–16:00 ET). "
    )

    _live_panel()
