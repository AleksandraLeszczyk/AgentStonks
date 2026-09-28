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
the window is the user's choice (`VOLUME_BASELINE_WINDOWS`).

The default window, "week_band", replaces both with a **band**: per bar, the
mean + 1 sigma of the volume that clock bucket carried over the last trading
week, drawn as a dark backdrop behind the bars (`volume_band`). It is built
from the same tape as the bar it sits behind -- an IEX bar against IEX
history, a SIP bar against SIP history, a yfinance bar against yfinance -- so
the chart's mixed-source buffer never compares one venue's slice with the
whole market's volume. Everything here is
a pure function over bars; the fetching lives in `agent_stonks.historical`.
"""

from __future__ import annotations

import statistics
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
    "week_band": "Last trading week, mean + 1\u03c3",
}
DEFAULT_VOLUME_BASELINE = "week_band"

# Windows drawn as a per-source mean + sigma band (`volume_band`) rather than
# the mean line and per-minute shape of `minute_volume_baseline`.
BAND_WINDOWS = ("week_band",)

# `volume_band` pools each bucket with the buckets this many *minutes* either
# side of it (so +/-5 one-minute bars, +/-1 five-minute bar, none coarser):
# five sessions are five samples a bucket, too few for a sigma. Same width as
# the momentum bands (`minute_momentum.SMOOTH_MINUTES`).
BAND_POOL_MINUTES = 5
BAND_SIGMAS = 1.0

# Minutes past ET midnight of the regular session's open and close.
_OPEN_MINUTE = 9 * 60 + 30
_CLOSE_MINUTE = 16 * 60

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
    return WEEK_LOOKBACK_DAYS if window in ("week", "week_band") else 0


def _band_segment(minute: int, span: int) -> str:
    """Which stretch of the day a bucket starting at `minute` belongs to.

    `volume_band` never pools across these: pre-market minutes carry a few
    percent of a regular minute, so a window reaching over the open would
    inflate the pre-market's last minutes and sink the open's. On one-minute
    bars the two auction minutes are stretches of their own -- SIP's 16:00 bar
    holds the closing cross, often more than the whole last half hour -- so
    their neighbours do not inherit it.
    """
    if span == 1 and minute in (_OPEN_MINUTE, _CLOSE_MINUTE):
        return str(minute)
    if minute < _OPEN_MINUTE:
        return "pre"
    return "regular" if minute < _CLOSE_MINUTE else "post"


def volume_band(
    history_bars: "list[dict]",
    today: str,
    span: int = 1,
    pool_minutes: int = BAND_POOL_MINUTES,
    sigmas: float = BAND_SIGMAS,
) -> "dict | None":
    """Mean + `sigmas` sigma of the volume per `span`-minute bucket over the
    last `WEEK_SESSIONS` sessions before `today`.

    `history_bars` are one-minute bars of a single source. Each session's
    minutes are summed into buckets starting on multiples of `span` past ET
    midnight -- where Alpaca and yfinance start their coarser bars -- so a
    bucket's samples are that bucket's volume, one per session that traded it.
    Each bucket then pools the samples of the buckets within `pool_minutes`
    of it in the same stretch of the day (`_band_segment`), and the band is
    their mean plus `sigmas` sample standard deviations (zero for a single
    sample).

    Returns ``{"per_minute": {bucket start minute: level}, "sessions",
    "dates"}``, or None when no prior session has a bar.
    """
    span = max(int(span), 1)
    prior = [bar for bar in history_bars or [] if (_bar_session_date(bar) or "") < today]
    dates = _session_dates(prior)[-WEEK_SESSIONS:]
    chosen = set(dates)
    sums: dict[tuple[str, int], float] = {}
    for bar in prior:
        session = _bar_session_date(bar)
        minute = _minute_of_day(bar)
        volume = _volume(bar)
        if session not in chosen or minute is None or volume is None:
            continue
        bucket = minute // span * span
        sums[(session, bucket)] = sums.get((session, bucket), 0.0) + volume
    if not sums:
        return None
    samples: dict[int, list[float]] = {}
    for (_, bucket), volume in sums.items():
        samples.setdefault(bucket, []).append(volume)

    reach = pool_minutes // span * span
    per_minute: dict[int, float] = {}
    for bucket in sorted(samples):
        segment = _band_segment(bucket, span)
        pooled = [
            v
            for other in range(bucket - reach, bucket + reach + 1, span)
            if other in samples and _band_segment(other, span) == segment
            for v in samples[other]
        ]
        spread = statistics.stdev(pooled) if len(pooled) > 1 else 0.0
        per_minute[bucket] = statistics.fmean(pooled) + sigmas * spread
    return {"per_minute": per_minute, "sessions": len(dates), "dates": dates}


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
    whose shape is the chart's own bars. Returns None for a band window (see
    `volume_band`), when the window is off, or
    the history is too thin to average anything -- callers draw nothing rather
    than a line built from one stray bar.
    """
    if window not in VOLUME_BASELINE_WINDOWS or window == "off" or window in BAND_WINDOWS:
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
