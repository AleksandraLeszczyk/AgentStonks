"""
US equity regular-session clock (09:30-16:00 ET, Monday-Friday).

Exchange holidays are not modeled: on a holiday the helpers treat the day as a
normal weekday, so a premarket window armed for it simply produces tactics that
cannot fill until the next real session -- acceptable for a paper-trading
sandbox, and it avoids shipping (and maintaining) a holiday calendar.
"""
from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

from . import clock

MARKET_TZ = ZoneInfo("America/New_York")
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)


def _as_market_time(now: "datetime | None") -> datetime:
    return (now or clock.now()).astimezone(MARKET_TZ)


def is_market_open(now: "datetime | None" = None) -> bool:
    """Whether the regular US equity session is in progress."""
    et = _as_market_time(now)
    return et.weekday() < 5 and MARKET_OPEN <= et.time() < MARKET_CLOSE


def session_phase(now: "datetime | None" = None) -> str:
    """Where `now` sits relative to the regular session:

        "open"        the bell has rung and the session is running
        "premarket"   a weekday before 09:30 ET
        "after_hours" a weekday after 16:00 ET
        "weekend"     Saturday or Sunday

    Used to decide what a briefing is actually *about*. Before the open (or on a
    day with no session left) the honest subject is the session that has not
    started yet; once the tape is running, it is the session in progress, and a
    briefing that still talks about "the open" is describing something that
    already happened.
    """
    et = _as_market_time(now)
    if et.weekday() >= 5:
        return "weekend"
    if et.time() < MARKET_OPEN:
        return "premarket"
    if et.time() >= MARKET_CLOSE:
        return "after_hours"
    return "open"


def next_market_open(now: "datetime | None" = None) -> datetime:
    """The next regular-session open strictly after `now`, in UTC.

    Mid-session (or after the close) this is the NEXT session's open -- the
    opening bell already rung today is never returned.
    """
    et = _as_market_time(now)
    candidate = et.replace(
        hour=MARKET_OPEN.hour, minute=MARKET_OPEN.minute, second=0, microsecond=0
    )
    if et.time() >= MARKET_OPEN:
        candidate += timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)
    return candidate.astimezone(timezone.utc)


def session_open(now: "datetime | None" = None) -> "datetime | None":
    """Open time (UTC) of the session currently in progress, or None when the
    market is closed."""
    if not is_market_open(now):
        return None
    et = _as_market_time(now)
    return et.replace(
        hour=MARKET_OPEN.hour, minute=MARKET_OPEN.minute, second=0, microsecond=0
    ).astimezone(timezone.utc)


def seconds_until_next_open(now: "datetime | None" = None) -> float:
    """Seconds from `now` to the next regular-session open (always > 0)."""
    base = (now or clock.now()).astimezone(timezone.utc)
    return max(0.0, (next_market_open(now) - base).total_seconds())


def seconds_to_close(now: "datetime | None" = None) -> "float | None":
    """Seconds from `now` to today's regular-session close, or None when the
    market is closed."""
    if not is_market_open(now):
        return None
    et = _as_market_time(now)
    close = et.replace(
        hour=MARKET_CLOSE.hour, minute=MARKET_CLOSE.minute, second=0, microsecond=0
    )
    return (close - et).total_seconds()
