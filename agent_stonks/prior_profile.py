"""Yesterday's volume profile on the live chart: its point of control and peaks.

The live chart can draw two kinds of line from the previous session's
volume-by-price profile: the point of control (POC), the price that traded the
most volume, and the profile's other peaks -- prices where volume piled up
again, separated from the POC by a thinner stretch.

- **"Yesterday" is the session before the chart's day**, the latest completed
  daily bar before it -- Friday on a Monday, the last session before a
  holiday. Without daily bars it is the nearest weekday before, skipping back
  over a day with no bars.
- **From yfinance's 1-minute bars of that session**, regular hours only (the
  consolidated tape, as the agent's `analyze_volume_profile` reads a past date).
  A minute's volume is spread evenly over its low-high range: one bar says how
  much traded between those prices, not where.
- **N_BINS equal slices of the day's range, lightly smoothed** (a Gaussian
  SMOOTH_BINS slices wide), so a single slice that a few minutes happened to
  stack on doesn't read as a peak of its own.
- **The POC is the profile's tallest point.** A peak is any other local
  maximum standing at least MIN_PROMINENCE of the POC's height above the lower
  of the two troughs that separate it from taller ground -- on six tickers'
  sessions of 2026-10-01..05 that kept none to four besides the POC, the
  bumps the eye picks out, and dropped the ripples.
- **Fetched once, in the background.** The chart redraws every few seconds and
  a yfinance call takes one or two; the profile of a finished session never
  changes, so it is kept for the process. A failed fetch is retried after
  RETRY_SEC, and the chart draws nothing until there is a profile.
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd

from . import clock
from .historical import fetch_intraday_bars_for_date
from .market_hours import MARKET_TZ

logger = logging.getLogger(__name__)

N_BINS = 100
SMOOTH_BINS = 1.5
MIN_PROMINENCE = 0.15
RETRY_SEC = 300.0
# Weekdays looked back over for a session when there are no daily bars.
_MAX_LOOKBACK_DAYS = 7

_lock = threading.Lock()
# (symbol, chart's ET day) -> profile levels, or None for a session with no bars.
_kept: dict[tuple[str, str], "dict | None"] = {}
_running: set[tuple[str, str]] = set()
_tried: dict[tuple[str, str], float] = {}


def volume_at_price(bars: "list[dict]", n_bins: int = N_BINS) -> "tuple[np.ndarray, np.ndarray] | None":
    """(slice centres, volume per slice) over `bars`' low-high range, each
    bar's volume spread evenly over its own low-high; None with no range or no
    volume."""
    rows = [
        (float(b["l"]), float(b["h"]), float(b.get("v") or 0.0))
        for b in bars
        if b.get("l") is not None and b.get("h") is not None
    ]
    if not rows:
        return None
    low = min(r[0] for r in rows)
    high = max(r[1] for r in rows)
    if not high > low:
        return None
    edges = np.linspace(low, high, n_bins + 1)
    volume = np.zeros(n_bins)
    for l, h, v in rows:
        if v <= 0:
            continue
        if h > l:
            overlap = np.clip(np.minimum(edges[1:], h) - np.maximum(edges[:-1], l), 0.0, None)
            volume += v * overlap / (h - l)
        else:
            volume[min(n_bins - 1, int((l - low) / (high - low) * n_bins))] += v
    if not volume.sum() > 0:
        return None
    return (edges[:-1] + edges[1:]) / 2, volume


def _smooth(values: np.ndarray, sigma: float) -> np.ndarray:
    """`values` convolved with a Gaussian `sigma` slices wide, zero beyond the ends."""
    if sigma <= 0:
        return values
    reach = int(np.ceil(3 * sigma))
    x = np.arange(-reach, reach + 1)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    return np.convolve(np.pad(values, reach), kernel / kernel.sum(), mode="valid")


def _prominences(values: np.ndarray) -> "list[tuple[int, float]]":
    """(index, prominence) of every local maximum of `values`: its height over
    the higher of the lowest points between it and taller ground on either
    side. Beyond the ends counts as zero -- nothing traded past the day's high
    or low -- so volume piled at an extreme is a peak too. A flat top counts
    once, at its first slice."""
    n = len(values)
    out = []
    for i in range(n):
        if (i > 0 and values[i - 1] >= values[i]) or (i < n - 1 and values[i + 1] > values[i]):
            continue
        troughs = []
        for step in (-1, 1):
            j, lowest = i + step, values[i]
            while 0 <= j < n and values[j] <= values[i]:
                lowest = min(lowest, values[j])
                j += step
            troughs.append(lowest if 0 <= j < n else 0.0)
        out.append((i, float(values[i] - max(troughs))))
    return out


def profile_levels(bars: "list[dict]") -> "dict | None":
    """{"poc", "peaks"} of `bars`' volume profile -- the POC and the other peaks
    (ascending) as prices -- or None when there is no profile."""
    profile = volume_at_price(bars)
    if profile is None:
        return None
    centers, volume = profile
    smoothed = _smooth(volume, SMOOTH_BINS)
    top = int(np.argmax(smoothed))
    peaks = [
        float(centers[i])
        for i, prominence in _prominences(smoothed)
        if i != top and prominence >= MIN_PROMINENCE * smoothed[top]
    ]
    return {"poc": float(centers[top]), "peaks": sorted(peaks)}


def previous_session(chart_day: date, daily_bars: "list[dict] | None" = None) -> "date | None":
    """The latest daily bar's date before `chart_day`, or None without one."""
    days = sorted(
        {str(b.get("t", ""))[:10] for b in daily_bars or [] if str(b.get("t", ""))[:10] < chart_day.isoformat()}
    )
    return date.fromisoformat(days[-1]) if days else None


def _weekdays_before(chart_day: date) -> "list[date]":
    days, day = [], chart_day
    while len(days) < _MAX_LOOKBACK_DAYS:
        day -= timedelta(days=1)
        if day.weekday() < 5:
            days.append(day)
    return days


def chart_day(bars: "list[dict]") -> date:
    """The ET day of the chart: its latest bar's, or today's without bars."""
    if bars:
        stamp = pd.Timestamp(bars[-1]["t"])
        stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp
        return stamp.tz_convert(MARKET_TZ).date()
    return clock.now().astimezone(MARKET_TZ).date()


def _fetch(symbol: str, day: date, daily_bars: "list[dict]") -> "dict | None":
    """The profile of the session before `day`, with its date. Raises when the
    session the daily bars name has no minute bars (a fetch that came back
    empty is worth retrying), or no session before `day` has any."""
    known = previous_session(day, daily_bars)
    for session in [known] if known else _weekdays_before(day):
        bars = fetch_intraday_bars_for_date(symbol, session.isoformat())
        if bars:
            levels = profile_levels(bars)
            return dict(levels, date=session.isoformat()) if levels else None
    raise LookupError(f"no minute bars for {symbol}'s session before {day}")


def levels(symbol: str, day: date, daily_bars: "list[dict] | None" = None) -> "dict | None":
    """The previous session's {"date", "poc", "peaks"} for the
    chart of `symbol` on `day`, or None until it has been fetched. Starts that
    fetch in the background, at most once per RETRY_SEC."""
    key = (symbol, day.isoformat())
    now = time.monotonic()
    with _lock:
        if key in _kept:
            return _kept[key]
        if key in _running or now - _tried.get(key, -1e9) < RETRY_SEC:
            return None
        _running.add(key)
        _tried[key] = now

    def fetch() -> None:
        try:
            result = _fetch(symbol, day, list(daily_bars or []))
        except Exception as exc:
            logger.warning("Previous session's profile for %s not built: %s", symbol, exc)
        else:
            with _lock:
                _kept[key] = result
        finally:
            with _lock:
                _running.discard(key)

    threading.Thread(target=clock.inherit(fetch), name=f"prior-profile-{symbol}", daemon=True).start()
    return None
