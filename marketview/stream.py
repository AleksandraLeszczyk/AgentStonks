import json
import threading
import time
from typing import Any

import websocket

from .config import BARS_STREAM_URL, NEWS_STREAM_URL
from .state import AppState


def _start_stream(symbol: str, key: str, secret: str, feed: str, state: AppState) -> None:
    """Open Alpaca WebSocket and stream real-time bars and trades into state."""

    def on_open(ws: websocket.WebSocketApp) -> None:
        ws.send(json.dumps({"action": "auth", "key": key, "secret": secret}))

    def on_message(ws: websocket.WebSocketApp, raw: str) -> None:
        try:
            messages: list[dict] = json.loads(raw)
        except json.JSONDecodeError:
            return

        for msg in messages:
            t = msg.get("T")
            if t == "connected":
                state.status = "Connected – authenticating…"
            elif t == "success" and msg.get("msg") == "authenticated":
                state.status = f"Authenticated – subscribing to {symbol} bars…"
                ws.send(
                    json.dumps({"action": "subscribe", "bars": [symbol], "trades": [symbol]})
                )
            elif t == "subscription":
                state.status = f"✅ Streaming {symbol} ({feed.upper()})"
            elif t == "b" and msg.get("S") == symbol:
                bar = {k: msg[k] for k in ("t", "o", "h", "l", "c", "v") if k in msg}
                with state.lock:
                    state.bars.append(bar)
            elif t == "t" and msg.get("S") == symbol:
                trade = {k: msg[k] for k in ("i", "x", "p", "s", "t", "c") if k in msg}
                with state.lock:
                    state.trades.append(trade)
            elif t == "error":
                state.status = f"Stream error: {msg.get('msg')}"

    def on_error(ws: websocket.WebSocketApp, err: Exception) -> None:
        state.status = f"WS error: {err}"

    def on_close(ws: websocket.WebSocketApp, *_: Any) -> None:
        if state.status.startswith("✅"):
            state.status = "Stream closed"

    ws = websocket.WebSocketApp(
        BARS_STREAM_URL.format(feed=feed),
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )
    state.ws = ws
    ws.run_forever(ping_interval=20, ping_timeout=10)


def launch_stream(symbol: str, key: str, secret: str, feed: str, state: AppState) -> None:
    """Close any existing bars/trades stream and start a new background thread."""
    if state.ws:
        try:
            state.ws.close()
        except Exception:
            pass
        time.sleep(0.5)

    threading.Thread(
        target=_start_stream, args=(symbol, key, secret, feed, state), daemon=True
    ).start()


def _start_stream_news(symbol: str, key: str, secret: str, state: AppState) -> None:
    """Open Alpaca news WebSocket and stream real-time news articles into state."""

    def on_open(ws: websocket.WebSocketApp) -> None:
        ws.send(json.dumps({"action": "auth", "key": key, "secret": secret}))

    def on_message(ws: websocket.WebSocketApp, raw: str) -> None:
        try:
            messages: list[dict] = json.loads(raw)
        except json.JSONDecodeError:
            return

        for msg in messages:
            t = msg.get("T")
            if t == "connected":
                state.status = "Connected – authenticating…"
            elif t == "success" and msg.get("msg") == "authenticated":
                state.status = "Authenticated – subscribing to news…"
                ws.send(json.dumps({"action": "subscribe", "news": [symbol]}))
            elif t == "n" and msg.get("S") == symbol:
                article = {
                    k: msg[k]
                    for k in ("id", "headline", "summary", "created_at", "url", "source")
                    if k in msg
                }
                with state.lock:
                    state.news.append(article)
            elif t == "error":
                state.status = f"News stream error: {msg.get('msg')}"

    def on_error(ws: websocket.WebSocketApp, err: Exception) -> None:
        state.status = f"WS error: {err}"

    def on_close(ws: websocket.WebSocketApp, *_: Any) -> None:
        if state.status.startswith("✅"):
            state.status = "Stream closed"

    ws = websocket.WebSocketApp(
        NEWS_STREAM_URL,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )
    state.ws_news = ws
    ws.run_forever(ping_interval=20, ping_timeout=10)


def launch_stream_news(symbol: str, key: str, secret: str, state: AppState) -> None:
    """Close any existing news stream and start a new background thread."""
    if state.ws_news:
        try:
            state.ws_news.close()
        except Exception:
            pass
        time.sleep(0.5)

    threading.Thread(
        target=_start_stream_news, args=(symbol, key, secret, state), daemon=True
    ).start()
