import html
import os
import re

import pandas as pd
import streamlit as st

from .charts import build_chart, empty_chart
from .config import FEEDS, MAX_BARS, PALETTE, POLL_SEC, SESSION_START, TIMEFRAMES
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


def _news_html(news: list[dict], symbol: str) -> str:
    if not news:
        return (
            f"<p style='color:{PALETTE['muted']};padding:12px'>"
            f"No recent news for {symbol}.</p>"
        )
    cards = []
    for item in news[:12]:
        ts = pd.to_datetime(item.get("created_at")).strftime("%b %d  %H:%M")
        src = html.escape(item.get("source", ""))
        headline = html.escape(_strip_html(item.get("headline", "")))
        summary = _strip_html(item.get("summary") or "")[:180].rstrip()
        summary = html.escape(summary)
        url = html.escape(item.get("url", "#"))
        cards.append(
            f"""
        <div style="background:{PALETTE['panel']}; border-radius:8px; padding:12px 14px;
                    border:1px solid {PALETTE['grid']}; display:flex; flex-direction:column;
                    gap:6px; min-width:0;">
          <div style="font-size:11px; color:{PALETTE['muted']};">
            {ts} · <span style="color:{PALETTE['accent']}">{src}</span>
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


@st.fragment(run_every=POLL_SEC)
def _live_panel() -> None:
    state = _get_state()
    st.caption(f"Status: {state.status}")

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
        )
        if (state.symbol and bars)
        else empty_chart()
    )
    st.plotly_chart(fig, use_container_width=True)
    st.html(_news_html(state.news, state.symbol))


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
        symbol = st.text_input("Symbol", value="AAPL", placeholder="AAPL, TSLA, MSFT…")
        timeframe = st.selectbox("Timeframe", TIMEFRAMES, index=0)
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

        c1, c2 = st.columns(2)
        start_clicked = c1.button("▶ Start", type="primary", use_container_width=True)
        stop_clicked = c2.button("⏹ Stop", use_container_width=True)

    state = _get_state()
    state.ma_periods = _parse_ma_periods(vwma_selection)
    state.show_7d_avg, state.show_28d_avg, state.show_1y_avg = _parse_avg_flags(avg_selection)
    state.show_fib = show_fib
    state.gaussian_max_components = max_components if fit_enabled else 0
    state.show_gaussian_centers = show_gaussian_centers if fit_enabled else False

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
                    state.daily_bars = daily_bars
                    with state.lock:
                        state.bars.clear()
                        state.bars.extend(historical_bars)
                        state.trades.clear()
                        state.trades.extend(historical_trades)
                    state.news = news
                    state.status = "Connecting WebSocket…"
                    launch_stream(sym, key, secret, feed, state)
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
