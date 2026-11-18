"""Whether SPY is down from its open by more than it usually moves.

The days-off check reads the market once, before the session: a market that
turns after the open is not in it. This is the same question asked during the
session, for Apple Trader's "no buys while SPY is falling" rule
(`AppleTraderConfig.no_buy_spy_fall_k`).

"Usually" is per clock minute, because a move from the open grows through the
day: 0.2% is a lot at 09:40 and nothing at 15:00. The usual move at a minute is
the **median of SPY's absolute move from its open** over the last
`USUAL_SESSIONS` completed sessions, pooled over `SMOOTH_MINUTES` either side.
A median rather than a mean or a sigma so that one crash does not make the next
month's falls look ordinary: 2026-11-05's -4% took a 20-session sigma to twice
its size, and 11-17's -1.4% then read as an unremarkable day.

Each read is the tape the live app has at the close of that bar, in a replay as
well: SIP's bars from 15 minutes back (the basic plan's delay) and IEX's up to
the bar. The open is SIP's first bar once it can be had and IEX's before that;
the price is IEX's newest bar. The tapes are `fastfurious_model`'s, so SPY's
history is cached with the other symbols' under data/fastfurious/.

Nothing here reads the wall clock to decide what a replay may see: a past
session is read whole from the cache and cut at the bar it is asked about.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from .market_hours import MARKET_TZ

SYMBOL = "SPY"

# The sessions "usual" is measured over, and the fewest it is believed from.
USUAL_SESSIONS = 20
MIN_USUAL_SESSIONS = 10
# Enough calendar days to hold USUAL_SESSIONS through a run of holidays.
HISTORY_CALENDAR_DAYS = 45

# Clock minutes pooled either side of the one asked about: 20 sessions are 20
# samples a minute, which leaves a jagged median. As `minute_momentum`'s band.
SMOOTH_MINUTES = 5

# The regular session's bars, 09:30 = 0.
SESSION_MINUTES = 390
_OPEN_MINUTE = 9 * 60 + 30

# A session in the history needs its opening bar and this many bars (a half day
# has 210): anything less is a hole in the tape, not a session.
_MIN_SESSION_BARS = 60

# How old SPY's newest bar may be, in minutes, and still be "now".
MAX_BAR_AGE = 3

# How long a failed read is remembered, so an outage costs one request a minute.
_RETRY_SEC = 60.0
_MEMO_MAX = 32


class Unavailable(ValueError):
    """SPY's move cannot be read for this bar; the message says why."""


@dataclass(frozen=True)
class SpyMove:
    """SPY at the close of one bar of the session."""

    minute: int
    open: float
    price: float
    move_pct: float   # from the open, signed
    usual_pct: float  # the usual size of that move at this minute, positive

    def is_fall_over(self, k: float) -> bool:
        """Whether SPY is down by more than `k` times its usual move."""
        return self.move_pct < 0 and -self.move_pct > float(k) * self.usual_pct


def _ff():
    # Late: the forecast module is heavy and only this rule needs it here.
    from . import fastfurious_model

    return fastfurious_model


def minute_of(ts) -> int:
    """The bar's minutes past 09:30 ET; negative before the open."""
    stamp = pd.Timestamp(ts).tz_convert(MARKET_TZ)
    return stamp.hour * 60 + stamp.minute - _OPEN_MINUTE


def _minutes(bars: pd.DataFrame) -> np.ndarray:
    stamps = bars.index.tz_convert(MARKET_TZ)
    return np.asarray(stamps.hour * 60 + stamps.minute - _OPEN_MINUTE)


def session_moves(bars: pd.DataFrame) -> pd.DataFrame:
    """Each session's % move of the close from its open, one row a session and
    one column a minute of it (0-389). A minute with no bar carries the last
    close; the minutes after a half day's close are empty."""
    columns = range(SESSION_MINUTES)
    if not len(bars):
        return pd.DataFrame(columns=columns, dtype=float)
    minutes = _minutes(bars)
    inside = (minutes >= 0) & (minutes < SESSION_MINUTES)
    bars, minutes = bars[inside], minutes[inside]
    rows = {}
    for session, at in pd.Series(np.arange(len(bars))).groupby(
        bars.index.tz_convert(MARKET_TZ).strftime("%Y-%m-%d")
    ):
        where = at.to_numpy()
        own = minutes[where]
        if own.min() > 0 or len(where) < _MIN_SESSION_BARS:
            continue
        opened = float(bars["open"].iloc[where[0]])
        if not opened:
            continue
        closes = bars["close"].to_numpy(dtype=float)[where]
        path = pd.Series((closes / opened - 1.0) * 100.0, index=own)
        rows[session] = path.reindex(range(int(own.max()) + 1)).ffill()
    return pd.DataFrame(rows).T.sort_index().reindex(columns=columns)


def usual_moves(bars: pd.DataFrame, day: date) -> np.ndarray:
    """The usual absolute % move from the open at each minute of `day`, from
    the `USUAL_SESSIONS` sessions before it in `bars`. Raises `Unavailable`
    with fewer than `MIN_USUAL_SESSIONS` of them."""
    moves = session_moves(bars)
    moves = moves[moves.index < day.isoformat()].tail(USUAL_SESSIONS)
    if len(moves) < MIN_USUAL_SESSIONS:
        raise Unavailable(
            f"only {len(moves)} complete {SYMBOL} sessions before {day} "
            f"({MIN_USUAL_SESSIONS} are needed to say what is usual)"
        )
    values = np.abs(moves.to_numpy(dtype=float))
    usual = np.full(SESSION_MINUTES, np.nan)
    for minute in range(SESSION_MINUTES):
        pooled = values[:, max(0, minute - SMOOTH_MINUTES): minute + SMOOTH_MINUTES + 1]
        pooled = pooled[np.isfinite(pooled)]
        if pooled.size:
            usual[minute] = float(np.median(pooled))
    return usual


def move_at(sip: pd.DataFrame, iex: pd.DataFrame, minute: int, usual: np.ndarray) -> SpyMove:
    """SPY at the close of bar `minute`, from the two tapes as they stand then
    (`sip` cut 15 bars back, `iex` at the bar)."""
    tape = iex if len(iex) else sip
    if not len(tape):
        raise Unavailable(f"{SYMBOL} has printed no bar yet today")
    newest = int(_minutes(tape)[-1])
    if minute - newest > MAX_BAR_AGE:
        raise Unavailable(f"{SYMBOL}'s newest bar is {minute - newest} minutes old")
    typical = float(usual[minute]) if 0 <= minute < SESSION_MINUTES else float("nan")
    if not typical > 0:
        raise Unavailable(f"no usual {SYMBOL} move is known for this minute of the session")
    opened = float((sip if len(sip) else iex)["open"].iloc[0])
    price = float(tape["close"].iloc[-1])
    if not opened:
        raise Unavailable(f"{SYMBOL}'s opening price is not known")
    return SpyMove(minute, opened, price, (price / opened - 1.0) * 100.0, typical)


# --- reading it, once per day and per bar --------------------------------------

_lock = threading.RLock()
_usual: "dict[date, np.ndarray]" = {}
_sessions: "dict[date, tuple[pd.DataFrame, pd.DataFrame]]" = {}
_live: "dict[tuple[date, int], SpyMove]" = {}
_failed: "dict[tuple, tuple[float, str]]" = {}


def clear_memos() -> None:
    """Forget what has been read (tests; a process otherwise keeps it)."""
    with _lock:
        _usual.clear()
        _sessions.clear()
        _live.clear()
        _failed.clear()


def _remember(store: dict, key, value) -> None:
    if len(store) >= _MEMO_MAX:
        store.clear()
    store[key] = value


def _usual_for(day: date, key, secret) -> np.ndarray:
    found = _usual.get(day)
    if found is None:
        first = day - timedelta(days=HISTORY_CALENDAR_DAYS)
        found = usual_moves(_ff().raw_history(SYMBOL, first, day, key, secret)["sip"], day)
        _remember(_usual, day, found)
    return found


def _past_tapes(day: date, minute: int, key, secret) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """A past session's tapes as they stood at the close of bar `minute`."""
    ff = _ff()
    whole = _sessions.get(day)
    if whole is None:
        raw = ff.raw_history(SYMBOL, day, day + timedelta(days=1), key, secret)
        whole = (raw["sip"], raw["iex"])
        _remember(_sessions, day, whole)
    sip, iex = whole
    return sip[_minutes(sip) <= minute - ff.L], iex[_minutes(iex) <= minute]


def read(day: date, minute: int, key: "str | None" = None, secret: "str | None" = None) -> SpyMove:
    """SPY's move at the close of bar `minute` of session `day`.

    Raises `Unavailable` when it cannot be read. Live, a bar IEX has not
    published yet is not waited for: the newest one stands in, and is read
    again on the next ask rather than remembered.
    """
    memo = (day, minute)
    with _lock:
        hit = _live.get(memo)
        if hit is not None:
            return hit
        failed = _failed.get(memo)
        if failed is not None and time.monotonic() - failed[0] < _RETRY_SEC:
            raise Unavailable(failed[1])
        try:
            usual = _usual_for(day, key, secret)
            ff = _ff()
            if day >= ff._today_et():
                sip, iex, settled = ff.live_tapes(SYMBOL, day, minute, key, secret, retries=0)
                move = move_at(sip, iex, minute, usual)
                if settled:
                    _remember(_live, memo, move)
                return move
            sip, iex = _past_tapes(day, minute, key, secret)
            return move_at(sip, iex, minute, usual)
        except Exception as exc:  # noqa: BLE001 -- any failure means "not known"
            why = str(exc) or type(exc).__name__
            _remember(_failed, memo, (time.monotonic(), why))
            raise Unavailable(why) from exc
