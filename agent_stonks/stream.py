import json
import logging
import socket
import threading
import time
from datetime import datetime
from typing import Any

import websocket

from .config import (
    BACKFILL_POLL_SEC,
    BARS_STREAM_URL,
    FALLBACK_POLL_SEC,
    MAX_BARS,
    NEWS_FALLBACK_POLL_SEC,
    NEWS_STREAM_URL,
)
from .datalog import log_fetch, log_fetch_failure
from .historical import fetch_intraday_bars
from .news import fetch_news_with_fallback
from .rest import fetch_bars, fetch_latest_quote, fetch_trades
from .state import (
    AppState,
    SymbolState,
    alert_field_value,
    alert_triggered,
    current_volume_ratio,
    format_alert,
    today_daily_volume,
)

logger = logging.getLogger(__name__)


def _fire_due_alerts(sym_state: SymbolState) -> None:
    """Check every pending condition alert (across ALL streamed symbols) against
    current state and, if any is met, clear the set and wake the agent early.
    Called after each kind of tick (bars, trades, quotes) so an alert on any
    continuously-updated field -- price, bid/ask, spread, day volume, volume
    ratio, portfolio value -- fires as soon as its field crosses the threshold,
    regardless of which symbol's tick moved it (a trade on one symbol moves the
    shared portfolio value, for instance).

    Armed tactics ride the same tick: the ticking symbol's executor is nudged so
    a conditional trade fires as soon as its conditions are met, not on its slow
    fallback poll.
    """
    executor = sym_state.tactics_executor
    if executor is not None and sym_state.tactics is not None:
        executor.notify()
    app = sym_state.app
    pairs = app.iter_alerts()
    if not pairs:
        return
    hit = next(((ss, a) for ss, a in pairs if alert_triggered(ss, a)), None)
    if hit is not None:
        ss_hit, alert = hit
        app.clear_alerts()
        value = alert_field_value(ss_hit, alert.get("field"))
        value_str = f"{value:,.4f}" if isinstance(value, (int, float)) else "n/a"
        app.agent_wake_reason = f"Alert hit: {format_alert(alert)} (now {value_str})."
        app.agent_wake_event.set()


def _apply_quote(state: SymbolState, quote: dict) -> None:
    """Copy an Alpaca quote (WS message or REST payload -- same field names)
    into the symbol's state. Caller must hold state.lock.

    Alpaca reports a one-sided book as bp/ap = 0; store None for that side so
    a bogus 0.0 never reaches the spread computation or a bid/ask alert. The
    quote timestamp is kept so consumers can tell a live quote from an
    hours-old off-session snapshot.
    """
    if "bp" in quote:
        price = float(quote["bp"])
        state.bid_price = price if price > 0 else None
    if "bs" in quote:
        state.bid_size = float(quote["bs"])
    if "ap" in quote:
        price = float(quote["ap"])
        state.ask_price = price if price > 0 else None
    if "as" in quote:
        state.ask_size = float(quote["as"])
    if "t" in quote:
        state.quote_ts = str(quote["t"])


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
    """Floor an ISO timestamp to the nearest N-minute bucket.

    Emits the same 'Z'-suffixed RFC-3339 format Alpaca uses for bar timestamps,
    so a bucket built here compares equal to a REST bar for the same period
    (isoformat()'s '+00:00' suffix broke that, duplicating buckets after a
    fallback refresh).
    """
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    total = dt.hour * 60 + dt.minute
    floored = (total // minutes) * minutes
    dt = dt.replace(hour=floored // 60, minute=floored % 60, second=0, microsecond=0)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _bar_ts_key(ts: object) -> str:
    """Normalize a bar timestamp for cross-source comparison ('Z' vs '+00:00')."""
    return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).isoformat()


def merge_missing_bars(state: SymbolState, fetched: list[dict]) -> int:
    """Insert fetched bars whose timestamps are absent from state.bars.

    On a timestamp collision the existing (streamed) bar always wins, so a
    live in-progress bar is never clobbered by an older REST snapshot.
    Returns the number of bars added.
    """
    if not fetched:
        return 0
    with state.lock:
        have = {_bar_ts_key(b["t"]) for b in state.bars if "t" in b}
        missing = [b for b in fetched if "t" in b and _bar_ts_key(b["t"]) not in have]
        if not missing:
            return 0
        merged = sorted(list(state.bars) + missing, key=lambda b: _bar_ts_key(b["t"]))
        state.bars.clear()
        state.bars.extend(merged[-MAX_BARS:])
    return len(missing)


# yfinance equivalents of Alpaca timeframes, for the secondary backfill source.
_YF_INTERVALS: dict[str, str] = {
    "1Min": "1m", "5Min": "5m", "15Min": "15m", "30Min": "30m", "1Hour": "60m",
}


def backfill_bars(
    symbol: str, key: str, secret: str, feed: str, state: SymbolState, timeframe: str
) -> tuple[int, str]:
    """Repair holes in one symbol's live bar series from a slower-but-complete source.

    The WS stream never re-delivers bars that closed while the socket was down,
    and on the IEX feed a minute without an IEX trade produces no bar at all --
    both leave permanent gaps in state.bars. Primary source is Alpaca REST
    (same feed as the stream, so volumes stay comparable); if that fails,
    delayed consolidated yfinance bars are used where the timeframe has an
    equivalent. Only missing timestamps are inserted -- streamed bars are never
    overwritten. Returns (bars_added, source_name).
    """
    failures: list[tuple[str, object]] = []
    consequence = "gaps in the bar series stay unrepaired until the next backfill"
    try:
        fetched = fetch_bars(symbol, timeframe, MAX_BARS, key, secret, feed, lookback_hours=16)
        source = "Alpaca REST"
    except Exception as exc:
        failures.append(("Alpaca REST", exc))
        yf_interval = _YF_INTERVALS.get(timeframe)
        if yf_interval is None:
            log_fetch_failure(
                "bar backfill",
                failures,
                symbol=symbol,
                consequence=f"no yfinance equivalent for {timeframe}; {consequence}",
            )
            raise
        try:
            fetched = fetch_intraday_bars(symbol, interval=yf_interval)
        except Exception as exc2:
            log_fetch_failure(
                "bar backfill",
                failures + [("yfinance", exc2)],
                symbol=symbol,
                consequence=consequence,
            )
            raise
        source = "yfinance (delayed)"
    added = merge_missing_bars(state, fetched)
    log_fetch(
        "bar backfill",
        source,
        symbol=symbol,
        detail=f"{added} missing {timeframe} bar(s) added",
        failures=failures,
    )
    return added, source


def _backfill_quietly(
    symbol: str, key: str, secret: str, feed: str, state: SymbolState, timeframe: str
) -> None:
    """backfill_bars wrapped for background use: swallow failures (already logged)."""
    try:
        backfill_bars(symbol, key, secret, feed, state, timeframe)
    except Exception:
        pass


def _backfill_all_quietly(
    symbols: list[str], key: str, secret: str, feed: str, app: AppState, timeframe: str
) -> None:
    for symbol in symbols:
        sym_state = app.sym(symbol)
        if sym_state is not None:
            _backfill_quietly(symbol, key, secret, feed, sym_state, timeframe)


def _start_stream(
    symbols: list[str], key: str, secret: str, feed: str, app: AppState, timeframe: str = "1Min"
) -> None:
    """Open one Alpaca WebSocket and stream real-time bars/trades/quotes for
    every subscribed symbol into its SymbolState."""
    tf_minutes = _TF_MINUTES.get(timeframe, 1)
    symbols_label = ", ".join(symbols)

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
                app.status = "Connected – authenticating…"
                continue
            if t == "success" and msg.get("msg") == "authenticated":
                app.status = f"Authenticated – subscribing to {symbols_label} bars…"
                ws.send(
                    json.dumps(
                        {"action": "subscribe", "bars": symbols, "trades": symbols, "quotes": symbols}
                    )
                )
                continue
            if t == "subscription":
                app.status = f"✅ Streaming {symbols_label} ({feed.upper()})"
                app.bars_connected = True
                # This also fires on every reconnect. The stream only pushes
                # bars from now on -- anything that closed while the socket was
                # down is gone unless fetched again, so repair the holes now.
                threading.Thread(
                    target=_backfill_all_quietly,
                    args=(symbols, key, secret, feed, app, timeframe),
                    daemon=True,
                ).start()
                continue
            if t == "error":
                app.status = f"Stream error: {msg.get('msg')}"
                app.bars_connected = False
                continue

            state = app.sym(msg.get("S", ""))
            if state is None:
                continue
            symbol = state.symbol

            if t == "b":
                bar = {k: msg[k] for k in ("t", "o", "h", "l", "c", "v", "vw") if k in msg}
                # First bar after a (re)connect logs at INFO; identical repeats
                # log at DEBUG (see datalog de-duplication).
                log_fetch(
                    "bars",
                    f"Alpaca WebSocket stream ({feed} feed)",
                    symbol=symbol,
                    detail=f"bar t={bar.get('t')} c={bar.get('c')}",
                )
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
                        state.previous_minute_high = bar["h"]
                    if "l" in bar:
                        state.previous_minute_low = bar["l"]
                    if "v" in bar:
                        state.day_volume = (state.day_volume or 0.0) + float(bar["v"])
                    day_volume = state.day_volume
                    vol_triggered = state.volume_alert_triggered
                    daily_bars = state.daily_bars
                vol_enabled = app.volume_alert_enabled
                vol_multiplier = app.volume_alert_multiplier

                # High-volume alert: today's cumulative volume crossing
                # multiplier x average daily volume. Latched per symbol
                # (vol_triggered) so it fires once per session rather than on
                # every later bar.
                if vol_enabled and not vol_triggered and day_volume is not None:
                    ratio, baseline = current_volume_ratio(day_volume, daily_bars)
                    if ratio is not None and ratio >= vol_multiplier:
                        state.volume_alert_triggered = True
                        state.volume_alert_ratio = ratio
                        app.agent_wake_reason = (
                            f"High-volume alert for {symbol}: today's volume {day_volume:,.0f} is "
                            f"{ratio:.2f}x average daily volume ({baseline:,.0f}), above "
                            f"the {vol_multiplier:.2f}x threshold."
                        )
                        app.agent_wake_event.set()

                # Generic condition alerts: a bar moves previous_minute_high/low/day_volume
                # (and the derived volume_ratio), so re-check after every bar.
                _fire_due_alerts(state)
            elif t == "t":
                trade = {k: msg[k] for k in ("i", "x", "p", "s", "t", "c") if k in msg}
                log_fetch(
                    "last price",
                    f"Alpaca WebSocket stream ({feed} feed)",
                    symbol=symbol,
                    detail=f"price={trade.get('p')}",
                )
                with state.lock:
                    state.trades.append(trade)
                    if "p" in msg:
                        state.last_price = float(msg["p"])
                        state.recent_prices.append((time.monotonic(), state.last_price))

                # Keep portfolio value marked-to-market independently of the
                # agent loop -- it never has to fetch or compute this itself.
                if app.decision_tracker is not None:
                    app.mark_to_market()

                # A trade moves last_price and portfolio_value -- re-check alerts.
                _fire_due_alerts(state)
            elif t == "q":
                log_fetch(
                    "ask/bid price",
                    f"Alpaca WebSocket stream ({feed} feed)",
                    symbol=symbol,
                    detail=f"bid={msg.get('bp')}, ask={msg.get('ap')}",
                )
                # Single lock acquisition so readers (e.g. the UI's quote
                # snapshot) never see a torn mix of this tick's bid with the
                # previous tick's ask, or vice versa.
                with state.lock:
                    _apply_quote(state, msg)

                # A quote moves bid/ask price+size and the derived spread.
                _fire_due_alerts(state)

    def on_error(ws: websocket.WebSocketApp, err: Exception) -> None:
        logger.warning("Bars stream error for %s: %s", symbols_label, err)
        app.status = f"WS error: {err}"
        app.bars_connected = False

    def on_close(ws: websocket.WebSocketApp, *_: Any) -> None:
        logger.info("Bars stream closed for %s, reconnecting…", symbols_label)
        app.bars_connected = False
        if app.status.startswith("✅"):
            app.status = "Stream closed"

    ws = websocket.WebSocketApp(
        BARS_STREAM_URL.format(feed=feed),
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )
    app.ws = ws
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


def _poll_symbol_via_rest(
    symbol: str, key: str, secret: str, feed: str, state: SymbolState, timeframe: str
) -> "str | None":
    """One REST fallback refresh of a single symbol's bars/price/quote.
    Returns the bar source name on success, None when no bars were available."""
    bar_failures: list[tuple[str, object]] = []
    try:
        bars = fetch_bars(symbol, timeframe, MAX_BARS, key, secret, feed, lookback_hours=16)
        source = "Alpaca REST"
    except Exception as exc:
        bar_failures.append(("Alpaca REST", exc))
        try:
            bars = fetch_intraday_bars(symbol)
            source = "yfinance (delayed)"
        except Exception as exc2:
            log_fetch_failure(
                "bars",
                bar_failures + [("yfinance", exc2)],
                symbol=symbol,
                consequence="no price data this cycle",
            )
            return None
    if not bars:
        log_fetch(
            "bars", source, symbol=symbol, detail="0 bars returned", failures=bar_failures
        )
        return None
    log_fetch(
        "bars", source, symbol=symbol, detail=f"{len(bars)} bars", failures=bar_failures
    )

    last_price = bars[-1].get("c")
    price_source = f"{source} (last bar close)"
    price_failures: list[tuple[str, object]] = []
    try:
        latest_trade = fetch_trades(symbol, key, secret, feed, lookback_hours=1)
        if latest_trade:
            last_price = latest_trade[-1].get("p", last_price)
            price_source = "Alpaca REST (latest trade)"
    except Exception as exc:
        # last bar's close is still a reasonable last_price
        price_failures.append(("Alpaca REST trades", exc))
    log_fetch(
        "last price",
        price_source,
        symbol=symbol,
        detail=f"price={last_price}",
        failures=price_failures,
    )

    quote = None
    try:
        quote = fetch_latest_quote(symbol, key, secret, feed)
    except Exception as exc:
        # bid/ask just won't refresh this cycle -- stays at its last known value
        log_fetch_failure(
            "ask/bid price",
            [("Alpaca REST /quotes/latest", exc)],
            symbol=symbol,
            consequence="no fallback source provides quotes; keeping last known bid/ask",
        )
    if quote:
        log_fetch(
            "ask/bid price",
            "Alpaca REST /quotes/latest",
            symbol=symbol,
            detail=f"bid={quote.get('bp')}, ask={quote.get('ap')}",
        )

    with state.lock:
        state.bars.clear()
        state.bars.extend(bars[-MAX_BARS:])
        state.last_price = last_price
        if last_price is not None:
            state.recent_prices.append((time.monotonic(), float(last_price)))
        last_bar = bars[-1] if bars else {}
        state.previous_minute_high = last_bar.get("h")
        state.previous_minute_low = last_bar.get("l")
        state.day_volume = sum(float(b.get("v") or 0.0) for b in bars)
        if quote:
            _apply_quote(state, quote)
    if state.app.decision_tracker is not None:
        state.app.mark_to_market()
    # Keep alerts live even when the WS is down and prices come from REST.
    _fire_due_alerts(state)
    return source


def _fallback_bars_loop(
    symbols: list[str],
    key: str,
    secret: str,
    feed: str,
    app: AppState,
    timeframe: str,
    stop_event: threading.Event,
) -> None:
    """REST-polling fallback that keeps prices flowing for every symbol while the
    bars/trades WS isn't connected. Alpaca's per-key streaming connection limit
    doesn't apply to REST calls, so this keeps working even while `_start_stream`
    is stuck retrying a rejected socket (e.g. another session/tab holding the one
    streaming slot Alpaca allows per key).

    Falls back further to yfinance (no API key, delayed quotes) if Alpaca's REST
    API itself is also unavailable.

    While the WS *is* connected this loop instead runs a periodic backfill
    (every BACKFILL_POLL_SEC) that merges only-missing bars, repairing holes
    left by reconnects and by feed minutes without any trade.
    """
    last_backfill = 0.0
    while not stop_event.wait(FALLBACK_POLL_SEC):
        if app.bars_connected:
            now = time.monotonic()
            if now - last_backfill >= BACKFILL_POLL_SEC:
                last_backfill = now
                _backfill_all_quietly(symbols, key, secret, feed, app, timeframe)
            continue
        polled_sources: list[str] = []
        for symbol in symbols:
            sym_state = app.sym(symbol)
            if sym_state is None:
                continue
            source = _poll_symbol_via_rest(symbol, key, secret, feed, sym_state, timeframe)
            if source:
                polled_sources.append(source)
        if polled_sources:
            app.status = (
                f"⚠️ Fallback: polling {', '.join(symbols)} via "
                f"{polled_sources[0]} (stream down)"
            )


def launch_stream(
    symbols: list[str], key: str, secret: str, feed: str, app: AppState, timeframe: str = "1Min"
) -> None:
    """Close any existing bars/trades stream and start a new background thread
    streaming every symbol over one socket, plus a REST-polling fallback that
    activates whenever the WS stream isn't connected."""
    if app.ws:
        try:
            app.ws.close()
        except Exception:
            pass
        time.sleep(0.5)
    if app.bars_fallback_stop_event:
        app.bars_fallback_stop_event.set()

    app.bars_connected = False

    for state in app.iter_symbol_states():
        with state.lock:
            if state.bars:
                state.prev_close = state.bars[-1].get("c")
            last_bar = state.bars[-1] if state.bars else None
            state.previous_minute_high = last_bar.get("h") if last_bar else None
            state.previous_minute_low = last_bar.get("l") if last_bar else None
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
            state.quote_ts = None

    stop_event = threading.Event()
    app.bars_fallback_stop_event = stop_event

    threading.Thread(
        target=_start_stream, args=(symbols, key, secret, feed, app, timeframe), daemon=True
    ).start()
    threading.Thread(
        target=_fallback_bars_loop,
        args=(symbols, key, secret, feed, app, timeframe, stop_event),
        daemon=True,
    ).start()


def _news_message_states(app: AppState, msg: dict) -> list[SymbolState]:
    """SymbolStates an Alpaca news message applies to. The news stream tags
    articles with a `symbols` list (older payloads used a single `S`)."""
    tagged = msg.get("symbols")
    if not isinstance(tagged, list):
        tagged = []
    single = msg.get("S")
    if single:
        tagged = [*tagged, single]
    states = []
    for raw in tagged:
        state = app.sym(str(raw))
        if state is not None and state not in states:
            states.append(state)
    return states


def _start_stream_news(symbols: list[str], key: str, secret: str, app: AppState) -> None:
    """Open Alpaca news WebSocket and stream real-time news articles into every
    matching symbol's state."""
    symbols_label = ", ".join(symbols)

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
                app.news_status = "Connected – authenticating…"
            elif t == "success" and msg.get("msg") == "authenticated":
                app.news_status = "Authenticated – subscribing to news…"
                ws.send(json.dumps({"action": "subscribe", "news": symbols}))
            elif t == "subscription":
                app.news_status = f"✅ Streaming news ({symbols_label})"
                app.news_connected = True
            elif t == "n":
                article = {
                    k: msg[k]
                    for k in ("id", "headline", "summary", "created_at", "url", "source")
                    if k in msg
                }
                for state in _news_message_states(app, msg):
                    log_fetch(
                        "news",
                        "Alpaca WebSocket news stream",
                        symbol=state.symbol,
                        detail=f"headline: {article.get('headline', '')[:80]}",
                    )
                    with state.lock:
                        if any(a.get("id") == article.get("id") for a in state.news):
                            continue
                        state.news.append(article)
                    headline = article.get("headline", "")
                    text = f"Fresh news arrived for {state.symbol}."
                    if headline:
                        text += f" Latest: {headline}"
                    app.agent_wake_reason = text
                    app.agent_wake_event.set()
            elif t == "error":
                app.news_status = f"News stream error: {msg.get('msg')}"
                app.news_connected = False

    def on_error(ws: websocket.WebSocketApp, err: Exception) -> None:
        logger.warning("News stream error for %s: %s", symbols_label, err)
        app.news_status = f"WS error: {err}"
        app.news_connected = False

    def on_close(ws: websocket.WebSocketApp, *_: Any) -> None:
        logger.info("News stream closed for %s, reconnecting…", symbols_label)
        app.news_connected = False
        if app.news_status.startswith("✅"):
            app.news_status = "Stream closed"

    ws = websocket.WebSocketApp(
        NEWS_STREAM_URL,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )
    app.ws_news = ws
    ws.run_forever(ping_interval=20, ping_timeout=10, reconnect=5, sockopt=_keepalive_sockopt())


def _fallback_news_loop(
    symbols: list[str],
    key: str,
    secret: str,
    worldnews_key: str,
    app: AppState,
    stop_event: threading.Event,
) -> None:
    """REST-polling fallback that keeps news flowing for every symbol while the
    news WS isn't connected.

    Alpaca's per-key streaming connection limit doesn't apply to REST calls, so this
    keeps working even while `_start_stream_news` is stuck retrying a rejected socket.
    """
    while not stop_event.wait(NEWS_FALLBACK_POLL_SEC):
        if app.news_connected:
            continue
        for symbol in symbols:
            state = app.sym(symbol)
            if state is None:
                continue
            try:
                fresh = fetch_news_with_fallback(symbol, key, secret, worldnews_key)
            except Exception as exc:
                log_fetch_failure(
                    "news",
                    [("news fallback poll", exc)],
                    symbol=symbol,
                    consequence="retrying next poll",
                )
                continue
            with state.lock:
                seen = {a.get("id") for a in state.news}
                new_articles = [a for a in fresh if a.get("id") not in seen]
                state.news.extend(new_articles)
            if new_articles:
                app.news_status = f"⚠️ Fallback polling news for {symbol} (stream down)"
                headline = new_articles[0].get("headline", "")
                text = f"Fresh news arrived for {symbol} (via fallback poll)."
                if headline:
                    text += f" Latest: {headline}"
                app.agent_wake_reason = text
                app.agent_wake_event.set()


def launch_stream_news(
    symbols: list[str], key: str, secret: str, app: AppState, worldnews_key: str = ""
) -> None:
    """Close any existing news stream and start a new background thread covering
    every symbol, plus a REST-polling fallback that activates whenever the WS
    stream isn't connected."""
    if app.ws_news:
        try:
            app.ws_news.close()
        except Exception:
            pass
        time.sleep(0.5)
    if app.news_fallback_stop_event:
        app.news_fallback_stop_event.set()

    app.news_connected = False
    stop_event = threading.Event()
    app.news_fallback_stop_event = stop_event

    threading.Thread(
        target=_start_stream_news, args=(symbols, key, secret, app), daemon=True
    ).start()
    threading.Thread(
        target=_fallback_news_loop,
        args=(symbols, key, secret, worldnews_key, app, stop_event),
        daemon=True,
    ).start()
