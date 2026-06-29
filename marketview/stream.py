import json
import logging
import socket
import threading
import time
from datetime import datetime
from typing import Any

import websocket

from .config import BARS_STREAM_URL, FALLBACK_POLL_SEC, MAX_BARS, NEWS_FALLBACK_POLL_SEC, NEWS_STREAM_URL
from .historical import fetch_intraday_bars
from .news import fetch_news_with_fallback
from .rest import fetch_bars, fetch_latest_quote, fetch_trades
from .state import AppState, alert_triggered, current_volume_ratio, today_daily_bar, today_daily_volume

logger = logging.getLogger(__name__)


_TF_MINUTES: dict[str, int] = {
    "1Min": 1, "5Min": 5, "15Min": 15, "30Min": 30, "1Hour": 60, "1Day": 1440,
}


def _keepalive_sockopt() -> list[tuple]:
    """TCP keepalive options so a silently-dead connection (NAT/proxy idle
    timeout dropping the TCP session without a FIN/close frame) is detected
    and torn down in seconds rather than leaving the stream hung until the
    next read happens to fail. Names differ by OS -- Linux exposes
    TCP_KEEPIDLE, macOS exposes TCP_KEEPALIVE instead -- so probe for both.
    """
    opts = [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)]
    idle_opt = getattr(socket, "TCP_KEEPIDLE", getattr(socket, "TCP_KEEPALIVE", None))
    if idle_opt is not None:
        opts.append((socket.IPPROTO_TCP, idle_opt, 30))
    if hasattr(socket, "TCP_KEEPINTVL"):
        opts.append((socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 10))
    if hasattr(socket, "TCP_KEEPCNT"):
        opts.append((socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3))
    return opts


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
                state.bars_connected = True
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
                    if "v" in bar:
                        state.day_volume = (state.day_volume or 0.0) + float(bar["v"])
                    day_volume = state.day_volume
                    vol_enabled = state.volume_alert_enabled
                    vol_multiplier = state.volume_alert_multiplier
                    vol_triggered = state.volume_alert_triggered
                    daily_bars = state.daily_bars

                # High-volume alert: today's cumulative volume crossing
                # multiplier x average daily volume. Latched (vol_triggered) so
                # it fires once per session rather than on every later bar.
                if vol_enabled and not vol_triggered and day_volume is not None:
                    ratio, baseline = current_volume_ratio(day_volume, daily_bars)
                    if ratio is not None and ratio >= vol_multiplier:
                        state.volume_alert_triggered = True
                        state.volume_alert_ratio = ratio
                        state.agent_wake_reason = (
                            f"High-volume alert: today's volume {day_volume:,.0f} is "
                            f"{ratio:.2f}x average daily volume ({baseline:,.0f}), above "
                            f"the {vol_multiplier:.2f}x threshold."
                        )
                        state.agent_wake_event.set()
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
                # Single lock acquisition so readers (e.g. the UI's quote
                # snapshot) never see a torn mix of this tick's bid with the
                # previous tick's ask, or vice versa.
                with state.lock:
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
                state.bars_connected = False

    def on_error(ws: websocket.WebSocketApp, err: Exception) -> None:
        logger.warning("Bars stream error for %s: %s", symbol, err)
        state.status = f"WS error: {err}"
        state.bars_connected = False

    def on_close(ws: websocket.WebSocketApp, *_: Any) -> None:
        logger.info("Bars stream closed for %s, reconnecting…", symbol)
        state.bars_connected = False
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
    #
    # sockopt enables TCP keepalive: the most common cause of repeated
    # "Connection to remote host was lost." drops is a NAT/proxy/load-balancer
    # between this process and Alpaca silently killing an idle TCP session --
    # neither side sends a close frame, so the app only notices on the next
    # read, which raises immediately. Keepalive probes generate traffic so
    # the OS detects and recovers (or reports) a dead socket within ~30-50s
    # instead of leaving it to rot.
    ws.run_forever(ping_interval=20, ping_timeout=10, reconnect=5, sockopt=_keepalive_sockopt())


def _fallback_bars_loop(
    symbol: str, key: str, secret: str, feed: str, state: AppState, timeframe: str, stop_event: threading.Event
) -> None:
    """REST-polling fallback that keeps prices flowing while the bars/trades WS isn't
    connected. Alpaca's per-key streaming connection limit doesn't apply to REST calls,
    so this keeps working even while `_start_stream` is stuck retrying a rejected socket
    (e.g. another session/tab holding the one streaming slot Alpaca allows per key).

    Falls back further to yfinance (no API key, delayed quotes) if Alpaca's REST API
    itself is also unavailable.
    """
    while not stop_event.wait(FALLBACK_POLL_SEC):
        if state.bars_connected:
            continue
        try:
            bars = fetch_bars(symbol, timeframe, MAX_BARS, key, secret, feed, lookback_hours=16)
            source = "Alpaca REST"
        except Exception as exc:
            logger.warning("Bars REST fallback failed for %s, trying yfinance: %s", symbol, exc)
            try:
                bars = fetch_intraday_bars(symbol)
                source = "yfinance (delayed)"
            except Exception as exc2:
                logger.warning("yfinance fallback also failed for %s: %s", symbol, exc2)
                continue
        if not bars:
            continue

        last_price = bars[-1].get("c")
        try:
            latest_trade = fetch_trades(symbol, key, secret, feed, lookback_hours=1)
            if latest_trade:
                last_price = latest_trade[-1].get("p", last_price)
        except Exception:
            pass  # last bar's close is still a reasonable last_price

        quote = None
        try:
            quote = fetch_latest_quote(symbol, key, secret, feed)
        except Exception:
            pass  # bid/ask just won't refresh this cycle -- stays at its last known value

        with state.lock:
            state.bars.clear()
            state.bars.extend(bars[-MAX_BARS:])
            state.last_price = last_price
            state.day_high = max(b["h"] for b in bars if "h" in b)
            state.day_low = min(b["l"] for b in bars if "l" in b)
            state.day_volume = sum(float(b.get("v") or 0.0) for b in bars)
            if quote:
                if "bp" in quote:
                    state.bid_price = float(quote["bp"])
                if "bs" in quote:
                    state.bid_size = float(quote["bs"])
                if "ap" in quote:
                    state.ask_price = float(quote["ap"])
                if "as" in quote:
                    state.ask_size = float(quote["as"])
        state.status = f"⚠️ Fallback: polling {symbol} via {source} (stream down)"


def launch_stream(symbol: str, key: str, secret: str, feed: str, state: AppState, timeframe: str = "1Min") -> None:
    """Close any existing bars/trades stream and start a new background thread, plus a
    REST-polling fallback that activates whenever the WS stream isn't connected."""
    if state.ws:
        try:
            state.ws.close()
        except Exception:
            pass
        time.sleep(0.5)
    if state.bars_fallback_stop_event:
        state.bars_fallback_stop_event.set()

    state.bars_connected = False

    with state.lock:
        if state.bars:
            state.prev_close = state.bars[-1].get("c")
        today_bar = today_daily_bar(state.daily_bars)
        state.day_high = today_bar.get("h") if today_bar else None
        state.day_low = today_bar.get("l") if today_bar else None
        # Seed today's running volume from today's partial daily bar (0 if the
        # latest daily bar isn't today, e.g. pre-open/weekend) and clear the
        # one-shot alert latch for the new session.
        state.day_volume = today_daily_volume(state.daily_bars)
        state.volume_alert_triggered = False
        state.volume_alert_ratio = None
        state.last_price = None
        state.bid_price = None
        state.bid_size = None
        state.ask_price = None
        state.ask_size = None

    stop_event = threading.Event()
    state.bars_fallback_stop_event = stop_event

    threading.Thread(
        target=_start_stream, args=(symbol, key, secret, feed, state, timeframe), daemon=True
    ).start()
    threading.Thread(
        target=_fallback_bars_loop, args=(symbol, key, secret, feed, state, timeframe, stop_event), daemon=True
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
                state.news_status = "Connected – authenticating…"
            elif t == "success" and msg.get("msg") == "authenticated":
                state.news_status = "Authenticated – subscribing to news…"
                ws.send(json.dumps({"action": "subscribe", "news": [symbol]}))
            elif t == "subscription":
                state.news_status = f"✅ Streaming news ({symbol})"
                state.news_connected = True
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
                state.news_status = f"News stream error: {msg.get('msg')}"
                state.news_connected = False

    def on_error(ws: websocket.WebSocketApp, err: Exception) -> None:
        logger.warning("News stream error for %s: %s", symbol, err)
        state.news_status = f"WS error: {err}"
        state.news_connected = False

    def on_close(ws: websocket.WebSocketApp, *_: Any) -> None:
        logger.info("News stream closed for %s, reconnecting…", symbol)
        state.news_connected = False
        if state.news_status.startswith("✅"):
            state.news_status = "Stream closed"

    ws = websocket.WebSocketApp(
        NEWS_STREAM_URL,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )
    state.ws_news = ws
    ws.run_forever(ping_interval=20, ping_timeout=10, reconnect=5, sockopt=_keepalive_sockopt())


def _fallback_news_loop(
    symbol: str, key: str, secret: str, worldnews_key: str, state: AppState, stop_event: threading.Event
) -> None:
    """REST-polling fallback that keeps news flowing while the news WS isn't connected.

    Alpaca's per-key streaming connection limit doesn't apply to REST calls, so this
    keeps working even while `_start_stream_news` is stuck retrying a rejected socket.
    """
    while not stop_event.wait(NEWS_FALLBACK_POLL_SEC):
        if state.news_connected:
            continue
        try:
            fresh = fetch_news_with_fallback(symbol, key, secret, worldnews_key)
        except Exception as exc:
            logger.warning("News fallback poll failed for %s: %s", symbol, exc)
            continue
        with state.lock:
            seen = {a.get("id") for a in state.news}
            new_articles = [a for a in fresh if a.get("id") not in seen]
            state.news.extend(new_articles)
        if new_articles:
            state.news_status = f"⚠️ Fallback polling news for {symbol} (stream down)"
            headline = new_articles[0].get("headline", "")
            text = "Fresh news arrived for the ticker (via fallback poll)."
            if headline:
                text += f" Latest: {headline}"
            state.agent_wake_reason = text
            state.agent_wake_event.set()


def launch_stream_news(symbol: str, key: str, secret: str, state: AppState, worldnews_key: str = "") -> None:
    """Close any existing news stream and start a new background thread, plus a
    REST-polling fallback that activates whenever the WS stream isn't connected."""
    if state.ws_news:
        try:
            state.ws_news.close()
        except Exception:
            pass
        time.sleep(0.5)
    if state.news_fallback_stop_event:
        state.news_fallback_stop_event.set()

    state.news_connected = False
    stop_event = threading.Event()
    state.news_fallback_stop_event = stop_event

    threading.Thread(
        target=_start_stream_news, args=(symbol, key, secret, state), daemon=True
    ).start()
    threading.Thread(
        target=_fallback_news_loop, args=(symbol, key, secret, worldnews_key, state, stop_event), daemon=True
    ).start()
