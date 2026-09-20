"""What a minute's volume *usually* is, for the live chart's volume panel.

A raw volume bar says 41,000 shares. Whether that is a lot depends entirely on
the minute it lands in: the first and last ten minutes of a session routinely
carry five to ten times what the middle carries, so a bar that would be a
screaming outlier at 12:40 is unremarkable at 09:31. This module turns prior
sessions into the two references that make the panel readable:

* a flat **mean** -- one number for the whole window, the line that answers
  "is this minute busier than a typical minute?"
* a **per-minute shape** -- the average volume of *this* clock minute across
  the window, which answers the sharper question "is this minute busier than
  this minute usually is?"

Both are built from the same window so the panel never mixes references, and
the window is the user's choice (`VOLUME_BASELINE_WINDOWS`). Everything here is
a pure function over bars; the fetching lives in `agent_stonks.historical`.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from .market_hours import MARKET_TZ

# The windows offered in Chart Settings. "session" is deliberately the odd one
# out: its per-minute shape would be a tracing of the very bars it is drawn
# over, so it contributes the mean line only (see `minute_volume_baseline`).
VOLUME_BASELINE_WINDOWS: dict[str, str] = {
    "off": "Off",
    "session": "This session",
    "yesterday": "Yesterday",
    "week": "Last trading week",
}
DEFAULT_VOLUME_BASELINE = "week"

# "Last trading week" is five *sessions*, not seven days: a holiday-shortened
# week should average the days that traded, not dilute itself with the ones
# that did not.
WEEK_SESSIONS = 5

# How many calendar days of minute history to ask for to find those sessions.
# Enough to clear a long weekend plus a holiday and still land five sessions.
WEEK_LOOKBACK_DAYS = 12


def _session_dates(bars: "list[dict]") -> "list[str]":
    """The ET dates present in `bars`, oldest first, de-duplicated."""
    seen: dict[str, None] = {}
    for bar in bars:
        date = _bar_session_date(bar)
        if date:
            seen[date] = None
    return sorted(seen)


def _bar_session_date(bar: dict) -> "str | None":
    ts = _bar_time(bar)
    return None if ts is None else ts.astimezone(MARKET_TZ).strftime("%Y-%m-%d")


def _bar_time(bar: dict) -> "datetime | None":
    raw = bar.get("t")
    if raw is None:
        return None
    try:
        ts = pd.Timestamp(raw)
    except (ValueError, TypeError):
        return None
    if pd.isna(ts):
        return None
    return ts.tz_localize(timezone.utc) if ts.tzinfo is None else ts.to_pydatetime()


def _minute_of_day(bar: dict) -> "int | None":
    """The bar's start as minutes past ET midnight, the key both series share.

    The market clock, not UTC: the whole point is to compare 09:31 against
    09:31, and the two frames drift apart by an hour twice a year.
    """
    ts = _bar_time(bar)
    if ts is None:
        return None
    et = ts.astimezone(MARKET_TZ)
    return et.hour * 60 + et.minute


def _volume(bar: dict) -> "float | None":
    try:
        vol = float(bar.get("v"))
    except (TypeError, ValueError):
        return None
    return None if vol != vol else vol


def lookback_days(window: str) -> int:
    """Calendar days of minute history `window` needs fetched."""
    if window == "yesterday":
        # A Monday's "yesterday" is the previous Friday, and a Tuesday after a
        # long weekend reaches back four days.
        return 5
    return WEEK_LOOKBACK_DAYS if window == "week" else 0


def minute_volume_baseline(
    window: str,
    session_bars: "list[dict]",
    history_bars: "list[dict] | None" = None,
    today: "str | None" = None,
) -> "dict | None":
    """The volume reference for `window`, or None when it cannot be built.

    `session_bars` are today's bars (the ones the chart is drawing);
    `history_bars` are minute bars spanning enough prior sessions, which only
    the "yesterday" and "week" windows read. `today` is the ET session date to
    measure *back from*, so a chart of an earlier day compares against the days
    before it rather than the days before now.

    The result is::

        {"key", "label", "mean_per_minute", "per_minute", "sessions", "dates"}

    where `per_minute` maps minutes-past-ET-midnight to the average volume that
    minute carried across the window's sessions. It is empty for "session",
    whose shape is the chart's own bars. Returns None when the window is off or
    the history is too thin to average anything -- callers draw nothing rather
    than a line built from one stray bar.
    """
    if window not in VOLUME_BASELINE_WINDOWS or window == "off":
        return None
    label = VOLUME_BASELINE_WINDOWS[window]

    if window == "session":
        volumes = [v for v in (_volume(b) for b in session_bars or []) if v is not None]
        if not volumes:
            return None
        return {
            "key": window,
            "label": label,
            "mean_per_minute": sum(volumes) / len(volumes),
            "per_minute": {},
            "sessions": 1,
            "dates": _session_dates(session_bars or [])[-1:],
        }

    if today is None:
        dates_today = _session_dates(session_bars or [])
        today = (
            dates_today[-1]
            if dates_today
            else datetime.now(timezone.utc).astimezone(MARKET_TZ).strftime("%Y-%m-%d")
        )

    # Only *completed* prior sessions: today's own bars leaking into the
    # baseline would make the reference partly a copy of what it references.
    prior = [
        bar
        for bar in (history_bars or [])
        if (_bar_session_date(bar) or "") < today
    ]
    wanted = 1 if window == "yesterday" else WEEK_SESSIONS
    dates = _session_dates(prior)[-wanted:]
    if not dates:
        return None
    chosen = set(dates)

    totals: dict[int, float] = {}
    counts: dict[int, int] = {}
    for bar in prior:
        if _bar_session_date(bar) not in chosen:
            continue
        minute = _minute_of_day(bar)
        volume = _volume(bar)
        if minute is None or volume is None:
            continue
        totals[minute] = totals.get(minute, 0.0) + volume
        counts[minute] = counts.get(minute, 0) + 1
    if not totals:
        return None

    # Divided by the days that actually reported the minute, not by the number
    # of sessions in the window: a minute that only printed on three of five
    # days is a thin minute, not a minute that averages 40% less.
    per_minute = {m: totals[m] / counts[m] for m in totals}
    return {
        "key": window,
        "label": label,
        "mean_per_minute": sum(per_minute.values()) / len(per_minute),
        "per_minute": per_minute,
        "sessions": len(dates),
        "dates": dates,
    }
