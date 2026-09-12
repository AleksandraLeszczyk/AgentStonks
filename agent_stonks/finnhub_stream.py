"""Finnhub WebSocket data source: raw trades in, locally-built candles out.

Finnhub's real-time socket (`wss://ws.finnhub.io?token=...`) serves exactly one
thing for US equities -- the trade tape:

    {"type": "trade", "data": [{"s": "AAPL", "p": 226.14, "t": 1757683200123,
                                "v": 100, "c": ["12"]}]}

No bars, no book. Everything the app draws and reasons over is a bar, so the
bars are aggregated here, tick by tick, into the same `{t,o,h,l,c,v,n,vw}` dicts
Alpaca's bar stream delivers. That is the whole difference between this module
and `agent_stonks.stream`; once a bar exists, the shared half of the tick path
(`agent_stonks.stream_common`) treats it identically.

Three things follow from building bars locally rather than receiving them:

* **The in-progress bar is visible.** Alpaca pushes a bar once it has closed, so
  the newest candle on the chart is always a completed minute. Here the newest
  candle is the minute happening right now, updated on every trade. That is the
  point -- it is the fastest view of the tape this app can draw -- but it also
  means the last candle is not a settled number, so `previous_minute_*` (which
  alerts, tactics and the rule traders read) is only ever published from a bar
  that has actually closed. See `CandleBuilder` and
  `stream_common.record_bar_close`.

* **A bar has to be closed by the clock, not by the next trade.** A thin symbol
  can go minutes without a print; waiting for the next trade to roll the bucket
  would stall `previous_minute_close` for exactly as long. `_flush_loop` closes
  any bucket whose minute has elapsed, whether or not a trade followed it.

* **There are no quotes.** Bid/ask, spread, and every alert or tactic condition
  built on them have no Finnhub source at all, so `stream` keeps refreshing them
  from Alpaca REST while this socket is the bar source (see
  `stream._refresh_quotes_via_rest`). Bars are likewise still backfilled from
  Alpaca REST, which repairs anything missed while the socket was down.

Volume is Finnhub's consolidated tape rather than a single venue's, so it runs
*higher* than the IEX feed's volume for the same minute and is not directly
comparable with historical IEX bars sitting earlier in the same buffer. It is
much closer to the truth than an IEX-only count; the mismatch is at the seam
between backfilled and streamed bars, not within either.
"""
import json
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any

import websocket

from . import clock
from . import scoring
from .config import FINNHUB_BAR_FLUSH_SEC, FINNHUB_STREAM_URL
from .datalog import log_fetch
from .state import AppState, SymbolState
from .stream_common import (
    TF_MINUTES,
    bar_ts_key,
    check_volume_alert,
    fire_due_alerts,
    floor_ts,
    keepalive_sockopt,
    record_bar_close,
)

logger = logging.getLogger(__name__)

SOURCE_LABEL = "Finnhub WebSocket stream"


def _iso_from_millis(millis: object) -> str:
    """RFC-3339 'Z' timestamp from Finnhub's epoch-milliseconds trade time.

    Every other timestamp in the app is the 'Z'-suffixed string Alpaca uses, and
    bars built here are compared against Alpaca REST bars by that string, so the
    conversion happens once, here, at the edge.
    """
    dt = datetime.fromtimestamp(float(millis) / 1000.0, tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def normalize_trade(raw: dict) -> "dict | None":
    """One Finnhub trade -> the Alpaca-shaped trade dict the rest of the app reads.

    The field names collide misleadingly: Finnhub's `s` is the *symbol* and `v`
    is the size, while Alpaca's `s` is the *size*. Charts and the agent read the
    Alpaca spelling, so the swap has to happen here or trade sizes silently
    become ticker strings.

    Returns None for a record missing a price or symbol -- a trade without a
    price is not a trade.
    """
    symbol = raw.get("s")
    price = raw.get("p")
    if not symbol or price is None:
        return None
    trade: dict[str, Any] = {"S": str(symbol), "p": float(price)}
    if raw.get("v") is not None:
        trade["s"] = float(raw["v"])
    if raw.get("t") is not None:
        trade["t"] = _iso_from_millis(raw["t"])
    if raw.get("c"):
        trade["c"] = list(raw["c"])
    return trade


class CandleBuilder:
    """Aggregates one symbol's trade stream into its live bar series.

    Writes straight into `state.bars` -- the newest entry is the bar currently
    being filled, rewritten in place on every trade, so the chart shows the
    minute as it happens.

    The only state kept here is `open_bucket`: the timestamp of the bar we are
    filling, or None when there isn't one. It is what separates "a bar this
    builder owns and may still close" from "whatever was in the buffer before we
    started" -- the buffer is seeded with backfilled Alpaca REST bars, and those
    are already closed and already published. Without that distinction the
    flush timer would keep re-publishing the last backfilled bar every couple of
    seconds, re-firing alerts and re-feeding the profit-potential tracker.
    """

    def __init__(self, state: SymbolState, tf_minutes: int) -> None:
        self.state = state
        self.tf_minutes = tf_minutes
        self.open_bucket: "str | None" = None

    def _open_bar_locked(self) -> "dict | None":
        """The bar this builder is filling, or None if it is no longer there.

        Normally that is simply `state.bars[-1]`, but the bar buffer is not ours
        alone: the periodic REST backfill merges missing bars into it and
        re-sorts, and `_poll_symbol_via_rest` replaces it wholesale. Matching on
        the timestamp means a bar that moved, or got evicted, is treated as "no
        open bar" and a fresh one is started -- rather than this builder
        silently writing its ticks into somebody else's bar.

        Caller must hold `state.lock`.
        """
        if self.open_bucket is None or not self.state.bars:
            return None
        last = self.state.bars[-1]
        if "t" not in last or bar_ts_key(last["t"]) != bar_ts_key(self.open_bucket):
            return None
        return last

    def add_trade(self, price: float, size: float, ts: object) -> "dict | None":
        """Fold one trade in. Returns the bar it displaced, if it closed one.

        A trade stamped *before* the bar being filled -- an out-of-order print,
        or a late arrival after the timer already closed that minute -- is
        dropped rather than rewriting a settled bar: bars the app has already
        reasoned over must not move underneath it.
        """
        bucket = floor_ts(ts, self.tf_minutes)
        state = self.state
        with state.lock:
            last = self._open_bar_locked()
            if self.open_bucket is not None and last is not None:
                if bar_ts_key(self.open_bucket) == bar_ts_key(bucket):
                    last["h"] = max(last["h"], price)
                    last["l"] = min(last["l"], price)
                    last["c"] = price
                    prior_v = float(last.get("v") or 0.0)
                    new_v = prior_v + size
                    if new_v > 0:
                        prior_vw = float(last.get("vw") or price)
                        last["vw"] = (prior_vw * prior_v + price * size) / new_v
                    last["v"] = new_v
                    last["n"] = int(last.get("n") or 0) + 1
                    return None
                if bar_ts_key(self.open_bucket) > bar_ts_key(bucket):
                    return None
                closed = dict(last)
            else:
                # Nothing of ours is open. Still refuse a trade that belongs to a
                # bucket the buffer already holds (the seeded history's last
                # bar), which would otherwise append an out-of-order duplicate.
                closed = None
                newest = state.bars[-1] if state.bars else None
                if newest is not None and "t" in newest and bar_ts_key(newest["t"]) >= bar_ts_key(bucket):
                    return None
            state.bars.append(
                {"t": bucket, "o": price, "h": price, "l": price, "c": price,
                 "v": size, "vw": price, "n": 1}
            )
            self.open_bucket = bucket
        return closed

    def close_if_elapsed(self, now: "float | None" = None) -> "dict | None":
        """Close the in-progress bar if its bucket has ended. Returns it, or None.

        Unlike `add_trade` this does not open a replacement: a minute with no
        trades produces no bar, exactly as a feed minute with no trade produces
        none on Alpaca, and the chart's gap filling draws the hole.
        """
        if self.open_bucket is None:
            return None
        started = clock.parse_iso_strict(self.open_bucket).timestamp()
        wall = now if now is not None else time.time()
        if wall < started + self.tf_minutes * 60:
            return None
        state = self.state
        with state.lock:
            last = self._open_bar_locked()
            closed = dict(last) if last is not None else None
        self.open_bucket = None
        return closed


def _publish_closed_bar(state: SymbolState, bar: "dict | None", detail_suffix: str = "") -> None:
    """Common tail for a bar that just closed, from either a trade or the timer."""
    if bar is None:
        return
    log_fetch(
        "bars",
        SOURCE_LABEL,
        symbol=state.symbol,
        detail=f"bar t={bar.get('t')} c={bar.get('c')}{detail_suffix}",
    )
    record_bar_close(state, bar)


def apply_trade(state: SymbolState, builder: CandleBuilder, trade: dict) -> None:
    """Everything one Finnhub trade does to a symbol's state.

    Order matters: the bar series is folded first (which may close the previous
    bar and publish it as the last completed one), then the live scalars --
    `last_price`, the recent-price ring the price alerts scan, the running day
    volume -- then the portfolio mark and the alert sweep, so an alert firing on
    this tick sees every field this tick moved.
    """
    price = float(trade["p"])
    size = float(trade.get("s") or 0.0)
    ts = trade.get("t") or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    app = state.app

    _publish_closed_bar(state, builder.add_trade(price, size, ts))

    with state.lock:
        state.trades.append({k: v for k, v in trade.items() if k != "S"})
        state.last_price = price
        state.recent_prices.append((time.monotonic(), price))
        if size:
            state.day_volume = (state.day_volume or 0.0) + size
    scoring.record_price(app, state.symbol, price)
    if size:
        check_volume_alert(state)

    # Keep portfolio value marked-to-market independently of the agent loop --
    # it never has to fetch or compute this itself.
    if app.decision_tracker is not None:
        app.mark_to_market()

    fire_due_alerts(state)


def _flush_loop(
    builders: dict[str, CandleBuilder], app: AppState, stop_event: threading.Event
) -> None:
    """Close elapsed in-progress bars on a timer for every streamed symbol.

    Runs only while the Finnhub socket is the live source; when it drops, the
    REST fallback in `stream` owns the bar series instead and rewrites it whole,
    so there is no in-progress bar of ours left to close.
    """
    while not stop_event.wait(FINNHUB_BAR_FLUSH_SEC):
        if not app.bars_connected:
            continue
        for symbol, builder in builders.items():
            state = app.sym(symbol)
            if state is None:
                continue
            closed = builder.close_if_elapsed()
            if closed is not None:
                _publish_closed_bar(state, closed, detail_suffix=" (closed on timer)")
                fire_due_alerts(state)


def start_stream(
    symbols: list[str],
    token: str,
    app: AppState,
    timeframe: str,
    builders: dict[str, CandleBuilder],
) -> None:
    """Open one Finnhub WebSocket and stream the trade tape for every symbol,
    aggregating it into that symbol's bar series. Blocks until the socket is
    closed for good (`ws.close()` from the Stop button)."""
    symbols_label = ", ".join(symbols)

    def on_open(ws: websocket.WebSocketApp) -> None:
        # Finnhub authenticates in the handshake query string, so an accepted
        # socket is already an authenticated one -- there is no auth round trip
        # to wait for and no subscription acknowledgement to come back. One
        # subscribe frame per symbol is the whole protocol.
        for symbol in symbols:
            ws.send(json.dumps({"type": "subscribe", "symbol": symbol}))
        app.bars_connected = True
        app.status = f"✅ Streaming {symbols_label} (Finnhub trades → local candles)"

    def on_message(ws: websocket.WebSocketApp, raw: str) -> None:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return
        if not isinstance(payload, dict):
            return

        kind = payload.get("type")
        if kind == "ping":
            return
        if kind == "error":
            app.status = f"Finnhub stream error: {payload.get('msg')}"
            app.bars_connected = False
            logger.warning("Finnhub stream error for %s: %s", symbols_label, payload.get("msg"))
            return
        if kind != "trade":
            return

        for raw_trade in payload.get("data") or []:
            trade = normalize_trade(raw_trade)
            if trade is None:
                continue
            state = app.sym(trade["S"])
            builder = builders.get(trade["S"])
            if state is None or builder is None:
                continue
            log_fetch(
                "last price",
                SOURCE_LABEL,
                symbol=state.symbol,
                detail=f"price={trade['p']}",
            )
            apply_trade(state, builder, trade)

    def on_error(ws: websocket.WebSocketApp, err: Exception) -> None:
        logger.warning("Finnhub stream error for %s: %s", symbols_label, err)
        app.status = f"Finnhub WS error: {err}"
        app.bars_connected = False

    def on_close(ws: websocket.WebSocketApp, *_: Any) -> None:
        logger.info("Finnhub stream closed for %s, reconnecting…", symbols_label)
        app.bars_connected = False
        if app.status.startswith("✅"):
            app.status = "Stream closed"

    app.status = "Connecting to Finnhub…"
    ws = websocket.WebSocketApp(
        FINNHUB_STREAM_URL.format(token=token),
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )
    app.ws = ws
    # Same reconnect/keepalive reasoning as the Alpaca socket: without
    # reconnect, one blip ends run_forever for good and the tape silently stops,
    # and without TCP keepalive a NAT/proxy can kill an idle session without
    # either side sending a close frame.
    ws.run_forever(ping_interval=20, ping_timeout=10, reconnect=5, sockopt=keepalive_sockopt())


def launch(
    symbols: list[str],
    token: str,
    app: AppState,
    timeframe: str,
    stop_event: threading.Event,
) -> None:
    """Start the Finnhub socket and its bar-flush timer as background threads.

    The builders are created here, one per symbol, and shared by the two threads:
    the socket thread fills them, the timer thread closes them out.
    """
    tf_minutes = TF_MINUTES.get(timeframe, 1)
    builders: dict[str, CandleBuilder] = {}
    for symbol in symbols:
        state = app.sym(symbol)
        if state is not None:
            builders[symbol] = CandleBuilder(state, tf_minutes)

    threading.Thread(
        target=start_stream, args=(symbols, token, app, timeframe, builders), daemon=True
    ).start()
    threading.Thread(
        target=_flush_loop, args=(builders, app, stop_event), daemon=True
    ).start()
