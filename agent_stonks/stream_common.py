"""Primitives shared by every live data source.

There is more than one way to get a live tape into `SymbolState` -- Alpaca
pushes ready-made bars, quotes and trades (`agent_stonks.stream`), Finnhub
pushes only raw trades and the bars are built here (`agent_stonks.finnhub_stream`)
-- but what happens *once a tick lands* is the same either way: alerts and armed
tactics get re-checked, the profit-potential tracker sees the price, the
high-volume latch gets a look, bar timestamps get bucketed and de-duplicated the
same way.

That common half lives here so the two sources can't drift apart on it, and so
neither has to import the other. `stream.py` re-exports the names it has always
exposed under their old private spellings, so existing callers and tests are
unaffected.
"""
import socket

from . import clock
from . import scoring
from .config import MAX_BARS
from .state import (
    SymbolState,
    alert_field_value,
    alert_triggered,
    format_alert,
    today_daily_volume,
)

# Minutes per Alpaca timeframe label, used to bucket ticks into bars.
TF_MINUTES: dict[str, int] = {
    "1Min": 1, "5Min": 5, "15Min": 15, "30Min": 30, "1Hour": 60, "1Day": 1440,
}


def fire_due_alerts(sym_state: SymbolState) -> None:
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


def record_bar_close(state: SymbolState, bar: dict) -> None:
    """Publish a just-completed bar as the symbol's "last completed bar".

    `previous_minute_high/low/close` are what alerts, tactics and the rule-based
    traders read when they want a level a full bar has actually closed through,
    so they must only ever move on a bar that is finished -- never on the
    in-progress one. The bar's low and high also bound what an oracle could have
    traded that minute, which is what the profit-potential tracker wants; order
    within the bar is unknown, and low-then-high assumes the optimistic
    buy-low-sell-high ordering.
    """
    with state.lock:
        if "h" in bar:
            state.previous_minute_high = bar["h"]
        if "l" in bar:
            state.previous_minute_low = bar["l"]
        if "c" in bar:
            state.previous_minute_close = bar["c"]
    if "l" in bar:
        scoring.record_price(state.app, state.symbol, bar["l"])
    if "h" in bar:
        scoring.record_price(state.app, state.symbol, bar["h"])


def apply_quote(state: SymbolState, quote: dict) -> None:
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


def keepalive_sockopt() -> list[tuple]:
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


def floor_ts(ts: object, minutes: int) -> str:
    """Floor an ISO timestamp to the nearest N-minute bucket.

    Emits the same 'Z'-suffixed RFC-3339 format Alpaca uses for bar timestamps,
    so a bucket built here compares equal to a REST bar for the same period
    (isoformat()'s '+00:00' suffix broke that, duplicating buckets after a
    fallback refresh).
    """
    dt = clock.parse_iso_strict(ts)
    total = dt.hour * 60 + dt.minute
    floored = (total // minutes) * minutes
    dt = dt.replace(hour=floored // 60, minute=floored % 60, second=0, microsecond=0)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def bar_ts_key(ts: object) -> str:
    """Normalize a bar timestamp for cross-source comparison ('Z' vs '+00:00')."""
    return clock.parse_iso_strict(ts).isoformat()


def merge_missing_bars(state: SymbolState, fetched: list[dict]) -> int:
    """Insert fetched bars whose timestamps are absent from state.bars.

    On a timestamp collision the existing (streamed) bar always wins, so a
    live in-progress bar is never clobbered by an older REST snapshot.

    Returns the number of bars that are in the buffer now and were not before
    -- which is not always every bar that was missing. The buffer is a ring of
    the newest `MAX_BARS`, so a fetched bar older than everything already in a
    full one has nowhere to go and is dropped again by the same truncation that
    keeps the ring bounded. Counting it as added would tell the caller (and the
    Backfill button's caption) that a bar had been restored while the chart
    still had a hole where it belongs.
    """
    if not fetched:
        return 0
    with state.lock:
        have = {bar_ts_key(b["t"]) for b in state.bars if "t" in b}
        missing = [b for b in fetched if "t" in b and bar_ts_key(b["t"]) not in have]
        if not missing:
            return 0
        merged = sorted(list(state.bars) + missing, key=lambda b: bar_ts_key(b["t"]))
        kept = merged[-MAX_BARS:]
        state.bars.clear()
        state.bars.extend(kept)
        kept_keys = {bar_ts_key(b["t"]) for b in kept}
        return sum(1 for b in missing if bar_ts_key(b["t"]) in kept_keys)


def merge_live_bars(
    state: SymbolState, fetched: list[dict], provisional: "set[str] | None" = None
) -> "tuple[int, int]":
    """Merge a `bar_history.fetch_live_bars` result into the live buffer.

    A fetched bar goes in where its timestamp is missing, and replaces an
    existing bar only when that bar is provisional and the fetched one is not
    -- a settled consolidated bar over an IEX bar or a partial live minute.
    Everything else already in the buffer wins, so a streamed bar is never
    overwritten by REST and IEX never overwrites anything.

    Returns (added, replaced), counting only bars still in the buffer after
    the ring's truncation (see `merge_missing_bars`).
    """
    provisional = provisional or set()
    if not fetched:
        return 0, 0
    with state.lock:
        by_key = {bar_ts_key(b["t"]): b for b in state.bars if "t" in b}
        added: set[str] = set()
        replaced: set[str] = set()
        for bar in fetched:
            if "t" not in bar:
                continue
            k = bar_ts_key(bar["t"])
            if k not in by_key:
                by_key[k] = bar
                added.add(k)
                if k in provisional:
                    state.provisional_bar_keys.add(k)
            elif k in state.provisional_bar_keys and k not in provisional:
                by_key[k] = bar
                replaced.add(k)
                state.provisional_bar_keys.discard(k)
        if not added and not replaced:
            return 0, 0
        kept = [by_key[k] for k in sorted(by_key)][-MAX_BARS:]
        state.bars.clear()
        state.bars.extend(kept)
        kept_keys = {bar_ts_key(b["t"]) for b in kept}
        state.provisional_bar_keys &= kept_keys
        return len(added & kept_keys), len(replaced & kept_keys)


def load_live_bars(state: SymbolState, fetched: list[dict], provisional: "set[str]") -> None:
    """Replace the whole live buffer with a `fetch_live_bars` result -- a first
    load, or a timeframe change, where nothing already buffered still applies."""
    with state.lock:
        state.bars.clear()
        state.bars.extend(fetched[-MAX_BARS:])
        kept_keys = {bar_ts_key(b["t"]) for b in state.bars if "t" in b}
        state.provisional_bar_keys = set(provisional) & kept_keys


def mark_newest_bar_provisional(state: SymbolState) -> None:
    """Flag the buffer's newest bar for replacement by settled data.

    Called where a live stream stops or starts: the minute it was building when
    it stopped holds only the trades before the stop, and nothing else would
    ever top it up. Flagging a bar that happens to be complete costs nothing:
    it is later replaced by a consolidated bar for the same minute.
    """
    with state.lock:
        if state.bars and "t" in state.bars[-1]:
            state.provisional_bar_keys.add(bar_ts_key(state.bars[-1]["t"]))


def reset_symbol_for_new_stream(state: SymbolState) -> None:
    """Clear the per-session live fields before a (re)start of any data source.

    Everything here is either seeded by the first tick of the new stream or is
    an explicit "we don't know yet" -- carrying the previous session's last
    price or bid/ask across a restart would make a stale number look live.
    The newest bar is the last minute the previous stream was building, cut
    short by the stop, so it is flagged for replacement.
    """
    mark_newest_bar_provisional(state)
    with state.lock:
        if state.bars:
            state.prev_close = state.bars[-1].get("c")
        last_bar = state.bars[-1] if state.bars else None
        state.previous_minute_high = last_bar.get("h") if last_bar else None
        state.previous_minute_low = last_bar.get("l") if last_bar else None
        state.previous_minute_close = last_bar.get("c") if last_bar else None
        # Seed today's running volume from today's partial daily bar (0 if the
        # latest daily bar isn't today, e.g. pre-open/weekend).
        state.day_volume = today_daily_volume(state.daily_bars)
        state.last_price = None
        state.bid_price = None
        state.bid_size = None
        state.ask_price = None
        state.ask_size = None
        state.quote_ts = None
