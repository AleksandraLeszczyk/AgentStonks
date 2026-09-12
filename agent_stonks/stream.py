"""Alpaca's WebSocket data source, plus the REST safety nets every source shares.

Two jobs live here. The first is the Alpaca socket itself: bars, trades and
quotes arrive ready-made and go straight into `SymbolState`. The second is
source-independent and serves whichever socket is running -- the REST fallback
that keeps prices flowing while no socket is connected, the periodic backfill
that repairs holes in the bar series while one is, and the quote poll that gives
the Finnhub source a book it does not otherwise have.

`launch_stream` is the entry point for both sources; which socket it opens is
`data_source` (see `agent_stonks.finnhub_stream` for the other one). The pieces
both sockets share on the tick path live in `agent_stonks.stream_common`.
"""
import json
import logging
import threading
import time
from typing import Any

import websocket

from .config import (
    BACKFILL_POLL_SEC,
    BARS_STREAM_URL,
    DEFAULT_DATA_SOURCE,
    FALLBACK_POLL_SEC,
    MAX_BARS,
    NEWS_FALLBACK_POLL_SEC,
    NEWS_STREAM_URL,
)
from . import finnhub_stream
from . import scoring
from . import stream_common
from .datalog import log_fetch, log_fetch_failure
from .historical import fetch_intraday_bars
from .news import fetch_news_with_fallback
from .rest import fetch_bars, fetch_latest_quote, fetch_trades
from .state import AppState, SymbolState
from .stream_common import merge_missing_bars  # noqa: F401  (re-export)

logger = logging.getLogger(__name__)

# The shared tick-path helpers used to live in this module and are still reached
# through it by callers and tests; `stream_common` is where they are defined.
_fire_due_alerts = stream_common.fire_due_alerts
_apply_quote = stream_common.apply_quote
_keepalive_sockopt = stream_common.keepalive_sockopt
_floor_ts = stream_common.floor_ts
_bar_ts_key = stream_common.bar_ts_key
_TF_MINUTES = stream_common.TF_MINUTES


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
                if "v" in bar:
                    with state.lock:
                        state.day_volume = (state.day_volume or 0.0) + float(bar["v"])

                # Every bar Alpaca pushes has already closed, so it is always the
                # "last completed bar" -- publish it as such and let the
                # profit-potential tracker see its extremes.
                stream_common.record_bar_close(state, bar)
                stream_common.check_volume_alert(state)

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
                if "p" in msg:
                    scoring.record_price(app, symbol, msg["p"])

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
        state.previous_minute_close = last_bar.get("c")
        state.day_volume = sum(float(b.get("v") or 0.0) for b in bars)
        if quote:
            _apply_quote(state, quote)
    # Session profit-potential tracking off the freshest bar and price; older
    # bars are skipped -- replaying history would break the tracker's
    # arrival-order assumption (and may predate the session anyway).
    scoring.record_price(state.app, symbol, last_bar.get("l"))
    scoring.record_price(state.app, symbol, last_bar.get("h"))
    scoring.record_price(state.app, symbol, last_price)
    if state.app.decision_tracker is not None:
        state.app.mark_to_market()
    # Keep alerts live even when the WS is down and prices come from REST.
    _fire_due_alerts(state)
    return source


def _refresh_quotes_via_rest(
    symbols: list[str], key: str, secret: str, feed: str, app: AppState
) -> None:
    """Refresh every symbol's bid/ask from Alpaca REST.

    Only the Finnhub source needs this: its socket carries the trade tape and
    nothing else, so `bid_price`, `ask_price`, their sizes and the derived
    spread have no live source at all while it is running. Alerts and tactic
    conditions on any of those would sit on whatever the initial load left
    behind -- silently stale rather than visibly absent -- so they are polled
    here instead, at the fallback cadence. Failures are logged and skipped: a
    missed poll leaves the last known quote in place, which is exactly what the
    Alpaca path does when its quote fetch fails.
    """
    if not key or not secret:
        return
    for symbol in symbols:
        state = app.sym(symbol)
        if state is None:
            continue
        try:
            quote = fetch_latest_quote(symbol, key, secret, feed)
        except Exception as exc:
            log_fetch_failure(
                "ask/bid price",
                [("Alpaca REST /quotes/latest", exc)],
                symbol=symbol,
                consequence="the Finnhub tape carries no quotes; keeping last known bid/ask",
            )
            continue
        if not quote:
            continue
        log_fetch(
            "ask/bid price",
            "Alpaca REST /quotes/latest",
            symbol=symbol,
            detail=f"bid={quote.get('bp')}, ask={quote.get('ap')}",
        )
        with state.lock:
            _apply_quote(state, quote)
        _fire_due_alerts(state)


def _fallback_bars_loop(
    symbols: list[str],
    key: str,
    secret: str,
    feed: str,
    app: AppState,
    timeframe: str,
    stop_event: threading.Event,
    data_source: str = DEFAULT_DATA_SOURCE,
) -> None:
    """REST-polling fallback that keeps prices flowing for every symbol while the
    bars/trades WS isn't connected. Alpaca's per-key streaming connection limit
    doesn't apply to REST calls, so this keeps working even while `_start_stream`
    is stuck retrying a rejected socket (e.g. another session/tab holding the one
    streaming slot Alpaca allows per key) -- and it is the safety net under the
    Finnhub socket too, since neither socket's outage affects Alpaca REST.

    Falls back further to yfinance (no API key, delayed quotes) if Alpaca's REST
    API itself is also unavailable.

    While a WS *is* connected this loop instead maintains what that socket
    doesn't deliver: a periodic backfill (every BACKFILL_POLL_SEC) that merges
    only-missing bars, repairing holes left by reconnects and by feed minutes
    without any trade, plus -- on the Finnhub source only -- a quote refresh on
    every tick, because that socket carries no book.
    """
    last_backfill = 0.0
    while not stop_event.wait(FALLBACK_POLL_SEC):
        if app.bars_connected:
            now = time.monotonic()
            if now - last_backfill >= BACKFILL_POLL_SEC:
                last_backfill = now
                _backfill_all_quietly(symbols, key, secret, feed, app, timeframe)
            if data_source == "finnhub":
                _refresh_quotes_via_rest(symbols, key, secret, feed, app)
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


def resolve_data_source(data_source: str, finnhub_token: str) -> str:
    """Which live source will actually be used, given the token situation.

    Finnhub is the default and needs a token; without one there is nothing to
    connect to, so the choice silently degrades to Alpaca rather than leaving
    the app on a socket that can only fail. Kept separate from `launch_stream`
    so the UI can say which source is running before anything is launched.
    """
    if data_source == "finnhub" and not finnhub_token:
        return "alpaca"
    return data_source if data_source in ("finnhub", "alpaca") else DEFAULT_DATA_SOURCE


def launch_stream(
    symbols: list[str],
    key: str,
    secret: str,
    feed: str,
    app: AppState,
    timeframe: str = "1Min",
    data_source: str = DEFAULT_DATA_SOURCE,
    finnhub_token: str = "",
) -> None:
    """Close any existing bars/trades stream and start a new background thread
    streaming every symbol over one socket, plus a REST-polling fallback that
    activates whenever the WS stream isn't connected.

    `data_source` picks the socket: "finnhub" (the default) streams the
    consolidated trade tape and builds candles locally; "alpaca" streams
    ready-made bars, trades and quotes from the chosen `feed`. Either way the
    Alpaca credentials are still required -- the REST fallback, the backfill and
    (under Finnhub) the quote poll all run on them.
    """
    source = resolve_data_source(data_source, finnhub_token)
    app.data_source = source
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
        stream_common.reset_symbol_for_new_stream(state)

    stop_event = threading.Event()
    app.bars_fallback_stop_event = stop_event

    if source == "finnhub":
        finnhub_stream.launch(symbols, finnhub_token, app, timeframe, stop_event)
    else:
        threading.Thread(
            target=_start_stream, args=(symbols, key, secret, feed, app, timeframe), daemon=True
        ).start()
    threading.Thread(
        target=_fallback_bars_loop,
        args=(symbols, key, secret, feed, app, timeframe, stop_event, source),
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
