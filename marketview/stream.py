import json
import threading
import time
from datetime import datetime
from typing import Any

import websocket

from .config import BARS_STREAM_URL, NEWS_STREAM_URL
from .state import AppState, alert_triggered


_TF_MINUTES: dict[str, int] = {
    "1Min": 1, "5Min": 5, "15Min": 15, "30Min": 30, "1Hour": 60, "1Day": 1440,
}


def _floor_ts(ts: str, minutes: int) -> str:
    """Floor an ISO timestamp to the nearest N-minute bucket."""
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    total = dt.hour * 60 + dt.minute
    floored = (total // minutes) * minutes
    dt = dt.replace(hour=floored // 60, minute=floored % 60, second=0, microsecond=0)
    return dt.isoformat()


def _start_stream(symbol: str, key: str, secret: str, feed: str, state: AppState, timeframe: str = "1Min") -> None:
    """Open Alpaca WebSocket and stream real-time bars and trades into state."""
    tf_minutes = _TF_MINUTES.get(timeframe, 1)

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
                    json.dumps({"action": "subscribe", "bars": [symbol], "trades": [symbol], "quotes": [symbol]})
                )
            elif t == "subscription":
                state.status = f"✅ Streaming {symbol} ({feed.upper()})"
            elif t == "b" and msg.get("S") == symbol:
                bar = {k: msg[k] for k in ("t", "o", "h", "l", "c", "v", "vw") if k in msg}
                if tf_minutes == 1:
                    with state.lock:
                        state.bars.append(bar)
                else:
                    bucket = _floor_ts(bar["t"], tf_minutes)
                    with state.lock:
                        if state.bars and state.bars[-1]["t"] == bucket:
                            last = state.bars[-1]
                            last["h"] = max(last["h"], bar["h"])
                            last["l"] = min(last["l"], bar["l"])
                            last["c"] = bar["c"]
                            new_v = last["v"] + bar["v"]
                            if "vw" in bar and "vw" in last and new_v > 0:
                                last["vw"] = (last["vw"] * last["v"] + bar["vw"] * bar["v"]) / new_v
                            last["v"] = new_v
                        else:
                            state.bars.append({**bar, "t": bucket})
                with state.lock:
                    if "h" in bar:
                        state.day_high = max(state.day_high, bar["h"]) if state.day_high is not None else bar["h"]
                    if "l" in bar:
                        state.day_low = min(state.day_low, bar["l"]) if state.day_low is not None else bar["l"]
            elif t == "t" and msg.get("S") == symbol:
                trade = {k: msg[k] for k in ("i", "x", "p", "s", "t", "c") if k in msg}
                with state.lock:
                    state.trades.append(trade)
                    if "p" in msg:
                        state.last_price = float(msg["p"])
                    price = state.last_price
                    alerts = state.price_alerts
                    tracker = state.decision_tracker

                # Keep portfolio value marked-to-market independently of the
                # agent loop -- it never has to fetch or compute this itself.
                if tracker is not None and price is not None:
                    snap = tracker.snapshot()
                    state.portfolio_value = snap["cash"] + snap["position"] * price

                if price is not None and alerts:
                    hit = next((a for a in alerts if alert_triggered(price, a)), None)
                    if hit is not None:
                        state.price_alerts = []
                        state.agent_wake_reason = (
                            f"Price alert hit at {price} ({hit['condition']} {hit['price']})."
                        )
                        state.agent_wake_event.set()
            elif t == "q" and msg.get("S") == symbol:
                if "bp" in msg:
                    state.bid_price = float(msg["bp"])
                if "bs" in msg:
                    state.bid_size = float(msg["bs"])
                if "ap" in msg:
                    state.ask_price = float(msg["ap"])
                if "as" in msg:
                    state.ask_size = float(msg["as"])
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
    # reconnect=5: without this, any drop (ping/pong timeout, network blip,
    # server-side close) ends run_forever for good and bars silently stop
    # arriving -- nothing else in the UI depends on this socket, so there's
    # no other signal that it died. ws.close() (Stop button) still ends the
    # retry loop via keep_running.
    ws.run_forever(ping_interval=20, ping_timeout=10, reconnect=5)


def launch_stream(symbol: str, key: str, secret: str, feed: str, state: AppState, timeframe: str = "1Min") -> None:
    """Close any existing bars/trades stream and start a new background thread."""
    if state.ws:
        try:
            state.ws.close()
        except Exception:
            pass
        time.sleep(0.5)

    with state.lock:
        if state.bars:
            state.prev_close = state.bars[-1].get("c")
        if state.daily_bars:
            today_bar = state.daily_bars[-1]
            state.day_high = today_bar.get("h")
            state.day_low = today_bar.get("l")
        else:
            state.day_high = None
            state.day_low = None
        state.last_price = None
        state.bid_price = None
        state.bid_size = None
        state.ask_price = None
        state.ask_size = None

    threading.Thread(
        target=_start_stream, args=(symbol, key, secret, feed, state, timeframe), daemon=True
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
                headline = article.get("headline", "")
                text = "Fresh news arrived for the ticker."
                if headline:
                    text += f" Latest: {headline}"
                state.agent_wake_reason = text
                state.agent_wake_event.set()
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
    ws.run_forever(ping_interval=20, ping_timeout=10, reconnect=5)


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
