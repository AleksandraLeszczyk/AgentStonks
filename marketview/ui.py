import os
import textwrap

import gradio as gr
import pandas as pd

from .charts import build_chart, empty_chart
from .config import (
    CUSTOM_CSS,
    FEEDS,
    MAX_BARS,
    PALETTE,
    POLL_SEC,
    SESSION_START,
    TIMEFRAMES,
)
from .rest import fetch_bars, fetch_news, fetch_trades
from .state import AppState
from .stream import launch_stream, launch_stream_news

state = AppState()


def wrap_text(text: str, width: int = 30) -> str:
    if not text:
        return ""
    return "<br>".join(textwrap.wrap(text, width=width))


def build_news_html(news: list[dict], symbol: str) -> str:
    if not news:
        return (
            f"<p style='color:{PALETTE['muted']};padding:12px'>"
            f"No recent news for {symbol}.</p>"
        )

    rows = []
    for item in news[:12]:
        ts = pd.to_datetime(item.get("created_at")).strftime("%b %d  %H:%M")
        src = item.get("source", "")
        summary = (item.get("summary") or "")[:220].rstrip()
        rows.append(
            f"""
        <div style="padding:10px 0; border-bottom:1px solid {PALETTE['grid']};">
          <div style="font-size:11px; color:{PALETTE['muted']}; margin-bottom:4px;">
            {ts} · <span style="color:{PALETTE['accent']}">{src}</span>
          </div>
          <a href="{item.get('url', '#')}" target="_blank"
             style="color:{PALETTE['text']}; font-weight:600;
                    text-decoration:none; font-size:14px; line-height:1.4">
            {item.get('headline', '')}
          </a>
          <div style="font-size:12px; color:{PALETTE['muted']}; margin-top:5px; line-height:1.5">
            {summary}…
          </div>
        </div>"""
        )

    return f"""
    <div style="background:{PALETTE['panel']}; border-radius:8px;
                padding:0 14px; font-family:Inter,sans-serif; max-height:500px;
                overflow-y:auto;">
      <h3 style="color:{PALETTE['text']}; font-size:14px; padding:12px 0 0; margin:0">
        📰 Latest news · <b style="color:{PALETTE['accent']}">{symbol}</b>
      </h3>
      {''.join(rows)}
    </div>"""


def on_start(
    symbol: str, timeframe: str, feed: str, api_key: str, api_secret: str
) -> tuple:
    symbol = symbol.strip().upper()
    key = api_key.strip() or os.getenv("ALPACA_API_KEY", "")
    secret = api_secret.strip() or os.getenv("ALPACA_SECRET", "")

    if not symbol:
        return empty_chart("⚠️ Please enter a symbol"), "<p>Enter a symbol first.</p>", "No symbol"
    if not key or not secret:
        return (
            empty_chart("⚠️ API credentials required"),
            "<p>Set API key and secret.</p>",
            "No credentials",
        )

    state.symbol = symbol
    state.feed = feed
    state.status = f"Loading history for {symbol}…"

    try:
        historical_bars = fetch_bars(symbol, timeframe, MAX_BARS, key, secret, feed)
        with state.lock:
            state.bars.clear()
            state.bars.extend(historical_bars)

        historical_trades = fetch_trades(symbol, key, secret, feed)
        with state.lock:
            state.trades.clear()
            state.trades.extend(historical_trades)

        state.news = fetch_news(symbol, key, secret)
    except Exception as exc:
        state.status = f"Failed to load data: {exc}"
        return empty_chart(f"⚠️ {exc}"), "<p>Failed to load data.</p>", state.status

    state.status = "Connecting WebSocket…"
    launch_stream(symbol, key, secret, feed, state)
    launch_stream_news(symbol, key, secret, state)

    fig = build_chart(list(state.bars), state.news, state.trades, symbol, SESSION_START)
    return fig, build_news_html(state.news, symbol), state.status


def on_stop() -> str:
    if state.ws:
        try:
            state.ws.close()
        except Exception:
            pass
    state.status = "Stopped"
    return state.status


def on_poll() -> tuple:
    """Refresh callback triggered by the Gradio timer."""
    with state.lock:
        bars = list(state.bars)
    fig = (
        build_chart(bars, state.news, state.trades, state.symbol, SESSION_START)
        if state.symbol
        else empty_chart()
    )
    return fig, build_news_html(state.news, state.symbol), state.status


def build_ui() -> gr.Blocks:
    with gr.Blocks(css=CUSTOM_CSS, title="Market Stream") as demo:
        gr.Markdown(
            f"<h1 style='color:{PALETTE['text']};font-family:Inter,sans-serif;"
            f"margin:0 0 4px'>📈 Market Stream</h1>"
            f"<p style='color:{PALETTE['muted']};font-size:13px;margin:0'>"
            "Real-time candlestick bars &amp; news via Alpaca streaming API</p>"
        )

        with gr.Row():
            sym_inp = gr.Textbox(
                label="Symbol", value="AAPL", placeholder="AAPL, TSLA, MSFT…", scale=2
            )
            tf_inp = gr.Dropdown(label="Timeframe", choices=TIMEFRAMES, value="1Min", scale=1)
            feed_inp = gr.Dropdown(label="Feed", choices=FEEDS, value="iex", scale=1)
            key_inp = gr.Textbox(
                label="Alpaca API Key",
                placeholder="From env ALPACA_API_KEY if blank",
                type="password",
                scale=3,
            )
            secret_inp = gr.Textbox(
                label="Alpaca Secret",
                placeholder="From env ALPACA_SECRET if blank",
                type="password",
                scale=3,
            )

        with gr.Row():
            with gr.Column(scale=1):
                start_btn = gr.Button("▶  Start", variant="primary")
                stop_btn = gr.Button("⏹  Stop", variant="secondary")
            with gr.Column(scale=5):
                status_bar = gr.Textbox(label="Status", interactive=False, value="Idle")

        with gr.Row():
            with gr.Column(scale=3):
                chart_out = gr.Plot(label="", show_label=False)
            with gr.Column(scale=1):
                news_out = gr.HTML(label="News")

        timer = gr.Timer(value=POLL_SEC, active=False)

        start_btn.click(
            fn=on_start,
            inputs=[sym_inp, tf_inp, feed_inp, key_inp, secret_inp],
            outputs=[chart_out, news_out, status_bar],
        ).then(lambda: gr.Timer(active=True), outputs=timer)

        stop_btn.click(fn=on_stop, outputs=status_bar).then(
            lambda: gr.Timer(active=False), outputs=timer
        )

        timer.tick(fn=on_poll, outputs=[chart_out, news_out, status_bar])

        gr.Markdown(
            f"<p style='color:{PALETTE['muted']};font-size:11px;margin-top:8px'>"
            "⚠️ Free Alpaca accounts: IEX feed available during US market hours (9:30–16:00 ET). "
            "Historical bars are loaded on start; live bars stream via WebSocket. "
            "Orange dotted lines mark news events on the chart.</p>"
        )

    return demo
