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

Streamed candles stand only for the last SETTLED_BAR_AGE_MIN minutes. The
backfill puts IEX bars in that window where the socket missed a minute, flagged
provisional, and once a minute is older it swaps whatever the buffer holds --
IEX bar or streamed candle -- for the settled SIP bar (see
`bar_history.fetch_live_bars`, `stream_common.merge_live_bars`).
"""
import json
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any

from . import clock
from . import scoring
from .config import FINNHUB_BAR_FLUSH_SEC, FINNHUB_STREAM_URL
from .datalog import log_fetch
from .state import AppState, SymbolState
from .stream_common import (
    TF_MINUTES,
    bar_ts_key,
    fire_due_alerts,
    floor_ts,
    record_bar_close,
)
from .ws_reconnect import ReconnectingSocket

logger = logging.getLogger(__name__)

SOURCE_LABEL = "Finnhub WebSocket stream"
SOCKET_LABEL = "Finnhub stream"


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
        # The first bar this builder opens starts at whatever trade arrived
        # first, not at the top of its minute, so it is flagged provisional for
        # the backfill to replace once a settled bar for it exists.
        self.opened_any = False

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
                 "v": size, "vw": price, "n": 1, "src": "finnhub"}
            )
            self.open_bucket = bucket
            if not self.opened_any:
                self.opened_any = True
                state.provisional_bar_keys.add(bar_ts_key(bucket))
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
    REST fallback in `stream` merges into the bar series instead and flags the
    bar we were building as provisional, so there is nothing of ours to close.
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


class FinnhubSubscription:
    """One session's claim on the shared Finnhub socket, kept on `app.ws`.

    `close()` (Stop, a restarted stream, a reaped session) drops this session's
    symbols. The socket itself closes only when the last subscription does.
    """

    def __init__(
        self,
        hub: "FinnhubHub",
        app: AppState,
        symbols: list[str],
        builders: dict[str, CandleBuilder],
    ) -> None:
        self.hub = hub
        self.app = app
        self.symbols = list(symbols)
        self.builders = builders
        self.status = f"✅ Streaming {', '.join(symbols)} (Finnhub trades → local candles)"

    def close(self) -> None:
        self.hub.remove(self)


class FinnhubHub:
    """The process's one Finnhub socket for an API key, shared by every session.

    Finnhub allows one socket per key, and a second connection gets one of them
    dropped. Each Streamlit session used to open its own socket: a second tab,
    a page refresh, or a SimLab and a live app on the same key. The sockets then
    dropped each other, and each reconnected every 5 s, until Finnhub's
    5-handshakes-per-window limit refused them with 429. Here each session
    registers a `FinnhubSubscription`. The socket subscribes to the union of
    their symbols and hands every trade to each session that streams that symbol.
    """

    def __init__(self, token: str) -> None:
        self.token = token
        self._subs: list[FinnhubSubscription] = []
        # Symbols the live connection has been sent a subscribe frame for.
        # Cleared on every (re)connect, when on_open subscribes to everything again.
        self._subscribed: set[str] = set()
        self._connected = False
        self.socket = ReconnectingSocket(
            SOCKET_LABEL,
            FINNHUB_STREAM_URL.format(token=token),
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )

    # -- subscriptions ------------------------------------------------------

    def _sync_subscriptions(self) -> None:
        """Bring the live connection's subscriptions in line with the sessions'.
        Caller holds `_lock`. A no-op while disconnected: on_open sends everything."""
        if not self._connected:
            return
        wanted = {s for sub in self._subs for s in sub.symbols}
        for symbol in sorted(wanted - self._subscribed):
            if self.socket.send(json.dumps({"type": "subscribe", "symbol": symbol})):
                self._subscribed.add(symbol)
        for symbol in sorted(self._subscribed - wanted):
            self.socket.send(json.dumps({"type": "unsubscribe", "symbol": symbol}))
            self._subscribed.discard(symbol)

    def add(self, sub: FinnhubSubscription) -> None:
        with _lock:
            self._subs.append(sub)
            self._sync_subscriptions()
            if self._connected:
                sub.app.bars_connected = True
                sub.app.status = sub.status

    def remove(self, sub: FinnhubSubscription) -> None:
        with _lock:
            if sub not in self._subs:
                return
            self._subs.remove(sub)
            sub.app.bars_connected = False
            if self._subs:
                self._sync_subscriptions()
                return
            if _hubs.get(self.token) is self:
                del _hubs[self.token]
        self.socket.close()

    # -- socket handlers ----------------------------------------------------

    def _on_open(self, ws: Any) -> None:
        # Finnhub authenticates in the handshake query string, so an accepted
        # socket is already an authenticated one. There is no auth round trip to
        # wait for and no subscription acknowledgement. One subscribe frame per
        # symbol is the whole protocol, and it is sent on every (re)connect.
        with _lock:
            self._connected = True
            self._subscribed = set()
            self._sync_subscriptions()
            for sub in self._subs:
                sub.app.bars_connected = True
                sub.app.status = sub.status

    def _mark_down(self, status: str, only_if_streaming: bool = False) -> None:
        with _lock:
            for sub in self._subs:
                sub.app.bars_connected = False
                if not only_if_streaming or sub.app.status.startswith("✅"):
                    sub.app.status = status

    def _on_message(self, ws: Any, raw: str) -> None:
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
            logger.warning("%s: error frame: %s", SOCKET_LABEL, payload.get("msg"))
            self._mark_down(f"Finnhub stream error: {payload.get('msg')}")
            return
        if kind != "trade":
            return

        with _lock:
            subs = list(self._subs)
        for raw_trade in payload.get("data") or []:
            trade = normalize_trade(raw_trade)
            if trade is None:
                continue
            for sub in subs:
                builder = sub.builders.get(trade["S"])
                state = sub.app.sym(trade["S"])
                if state is None or builder is None:
                    continue
                log_fetch(
                    "last price",
                    SOURCE_LABEL,
                    symbol=state.symbol,
                    detail=f"price={trade['p']}",
                )
                apply_trade(state, builder, dict(trade))

    def _on_error(self, ws: Any, err: Exception) -> None:
        self._mark_down(f"Finnhub WS error: {err}")

    def _on_close(self, ws: Any, *_: Any) -> None:
        with _lock:
            self._connected = False
            self._subscribed = set()
        self._mark_down("Stream closed", only_if_streaming=True)


# One hub per API key, for the whole process. `_lock` guards this registry and
# every hub's subscriber list.
_hubs: dict[str, FinnhubHub] = {}
_lock = threading.RLock()


def _start_socket_thread(hub: FinnhubHub) -> None:
    threading.Thread(target=hub.socket.run_forever, daemon=True).start()


def subscribe(
    symbols: list[str], token: str, app: AppState, builders: dict[str, CandleBuilder]
) -> FinnhubSubscription:
    """Register `app` for `symbols` on the key's shared socket, opening the
    socket if this is the first subscription."""
    with _lock:
        hub = _hubs.get(token)
        created = hub is None
        if created:
            hub = _hubs[token] = FinnhubHub(token)
            app.status = "Connecting to Finnhub…"
        sub = FinnhubSubscription(hub, app, symbols, builders)
        hub.add(sub)
    if created:
        _start_socket_thread(hub)
    return sub


def launch(
    symbols: list[str],
    token: str,
    app: AppState,
    timeframe: str,
    stop_event: threading.Event,
) -> None:
    """Join the shared Finnhub socket and start this session's bar-flush timer.

    The builders are created here, one per symbol. The socket's thread fills
    them and the timer thread closes them out. `app.ws` holds the subscription,
    so the usual `app.ws.close()` releases it.
    """
    tf_minutes = TF_MINUTES.get(timeframe, 1)
    builders: dict[str, CandleBuilder] = {}
    for symbol in symbols:
        state = app.sym(symbol)
        if state is not None:
            builders[symbol] = CandleBuilder(state, tf_minutes)

    app.ws = subscribe(symbols, token, app, builders)
    threading.Thread(
        target=_flush_loop, args=(builders, app, stop_event), daemon=True
    ).start()
