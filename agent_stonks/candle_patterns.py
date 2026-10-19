"""Candle patterns, in the shape a price chart can draw.

The sibling of `model_overlays`: that module draws what a trained model *said*
about a session, this one draws what the tape itself *did* -- shapes read off
the candles with no model and no forecast. Keeping the two apart keeps the
pickers honest: nothing on the "Model Predictions" list is a pattern, and
nothing here claims to predict.

Three patterns
--------------
`fvg`   Fair value gap: three candles where candle 1's wick and candle 3's
        wick do not overlap, leaving a price zone candle 2 crossed without
        trading on both sides of it. Drawn as a box from candle 1 over the
        gap's price range, reaching right until price trades through its far
        edge (then faded and ended at that bar) or to the session's last bar
        while it is still open. Detection is
        `technical_analysis.fair_value_gaps`.
`stall` Volume-node stall (absorption): price runs into a price where a lot
        of volume already traded and its N-bar average momentum falls to
        about zero. Marked at the bar where that happens, with a dashed line
        at the node's price. See "The stall" below.
`zone`  Stall-zone level: a stretch where price sat in a narrow band for a
        while, boxed, with a line at the price that traded the most volume
        inside it -- the peak of that stretch's own price profile. Read off
        the volume at each price rather than the candles' shape. See "The
        stall zone" below.

Two filters, because the raw pattern is everywhere on 1-minute bars
------------------------------------------------------------------
Measured on stored 2026-07 sessions: 100-230 FVGs per regular session, and
the median one is filled again within 4-7 bars. Drawn unfiltered that is a
wall of boxes. So:

`min_size`     a gap narrower than this many times the average bar range
               (mean high-low of the 14 bars before candle 2) is dropped.
               0.5 keeps roughly a third to a half of AAPL's; 1.0 about a
               tenth. Relative rather than in dollars, so one setting serves a
               $30 and a $300 stock, and a quiet midday and a busy open.
`hide_filled`  drop gaps price has already traded through, leaving only the
               imbalances still open on the chart.

Only regular-session bars are read, and each exchange-local day on its own: a
thin pre-market tape leaves gaps between almost every pair of bars, and three
candles straddling an overnight close are an opening gap, not an FVG.

The newest bar on a live chart is still forming, so a gap whose candle 3 is
that bar can appear and vanish before the minute closes.

The stall
---------
Read at every bar `t` of a session, with `m(t) = (c[t] - c[t-N]) / N` the
N-bar average momentum the chart's agent line draws and `u` the typical
minute move (the ticker's `abs_mean_minute_momentum`, or the session's own
mean |1-bar move| so far when the caller has none):

1. **Stalled**: `|m(t)| < 0.1 u` -- Apple Trader's neutral band
   (`MOMENTUM_NEUTRAL_FRACTION`).
2. **After a move**: the N bars before that, `m(t-N)`, ran at least
   `approach` x u up (the node is resistance) or down (support). Without it
   every quiet minute beside the POC would be marked, all midday.
3. **At a volume node that was already there**: the volume profile of the
   bars *before the approach began* (`prior_profile.profile_peaks`, the peak
   rule of yesterday's lines) has its POC or a peak within STALL_NEAR x u of
   where price stalled -- at or above the close for resistance, at or below
   it for support, and no further out than the stall's high (low) plus that
   distance. The approach's own bars are left out because they would build
   the node they then "find".
4. Optionally **on heavy volume**: the stall's bars average at least
   `min_volume_ratio` x the volume of the STALL_VOLUME_LOOKBACK bars before
   the approach -- effort without result. 0 turns the check off.

One mark per visit: after a mark nothing is marked again until a close lands
more than STALL_REARM x u from its node. Everything a mark is decided on is at
or before its bar, so it is the same live and in a replay; only the dashed
line's length looks ahead -- it runs right until a close goes through the node
by more than STALL_NEAR x u, or STALL_LEVEL_SPAN_BARS bars, or the session's
last bar, so the chart shows whether the node held.

Measured on 267 stored yfinance sessions of nine tickers (2026-08-03..10-09,
u = last week's value as SimMarket reads it), outcome = which comes first
within 30 bars, a 2u or 4u move back against the approach or on with it:

- **How often**: N=5 marks 4.9 / 2.2 / 0.4 a session at approach 0.25 / 0.5
  / 1.0; N=3 (Apple Trader's default look-back) 4.8 / 2.8 / 0.6.
- **At approach 0.25-0.5 it predicts nothing**: back-first and on-first are
  both ~47-49% (2u), the same as every bar after an approach, a stall with
  no node near, or a node price is still moving through.
- **At approach 1.0 (the default) there is a hint, on the support side only**:
  N=5 support marks went 2u back first 59% vs on 37% (54 marks), N=3 57% vs
  38% (87); resistance marks were even. Small samples picked from 27 settings,
  so not proof.
- **The volume check did not help**: with it on, marks leaned towards price
  going on *through* the node, not back. Off by default.
- The node was closed through within 30 bars after 60-70% of marks.

The stall zone
--------------
A zone is a run of at least `min_bars` bars whose closes all lie within
`width` x u of each other (u as for the stall, read at the bar the run
reaches `min_bars`): price going nowhere for a while. Runs are found left to
right, greedily -- a zone grows bar by bar until a close breaks the band, and
the next search starts at that bar -- so zones never overlap.

Its level is the peak of the zone's own volume-at-price profile
(`prior_profile.profile_peaks` over the zone's bars, each bar's volume spread
over its low-high): where the stalled trading actually piled up, which need
not be the middle of the band. What the level is follows from how price left:

- **resistance** -- the band broke downwards, so the level is overhead;
- **support** -- it broke upwards;
- **forming** -- price is still inside the band at the session's last bar
  (live: the newest bar), so the zone and its level can still grow.

The box covers the zone's bars and close band; the line runs from the zone's
first bar until a close goes through the level by more than ZONE_NEAR x u from
the side price left on, or to the session's last bar. The zone is known only
from its bars, so it reads the same live and in a replay -- except that a
forming zone's level moves as it grows.

Counted on the same 267 sessions: with a 3u band, 15 / 20 / 30-bar zones
come 7.4 / 4.0 / 1.3 a session (default 20: median 24 bars, about 106 minutes
of a session inside one); a 2u band 3.3 / 1.3 / 0.2, a 4u band 9.5 / 6.1 /
2.7. Almost none start before 10:00 -- the open moves several times last
week's typical minute -- and about 70% of levels are closed through before
the session ends. How often price turns at a level it comes back to is not
measured yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Mapping

import numpy as np
import pandas as pd

from . import market_hours
from .config import MOMENTUM_NEUTRAL_FRACTION
from .prior_profile import profile_peaks
from .technical_analysis import fair_value_gaps

FVG_KEY = "fvg"
STALL_KEY = "stall"
ZONE_KEY = "zone"

# The bars before candle 2 whose mean high-low range sizes a gap.
FVG_RANGE_LOOKBACK = 14
DEFAULT_FVG_MIN_SIZE = 0.5

# The stall's momentum look-back when the caller has none: the live chart's
# fallback for an agent that reads no momentum, five 1-minute bars.
DEFAULT_STALL_BARS = 5
# The approach's N-bar average momentum, in typical minute moves.
DEFAULT_STALL_APPROACH = 1.0
DEFAULT_STALL_MIN_VOLUME = 0.0
# How far from the stall a volume node may be, in typical minute moves; a
# close this far past the node also counts as going through it.
STALL_NEAR = 1.0
# A close this many typical minute moves from the last node re-arms the mark.
STALL_REARM = 2.0
# The bars the profile needs before it is read at all, and the bars before
# the approach the stall's volume is compared with.
STALL_MIN_PROFILE_BARS = 15
STALL_VOLUME_LOOKBACK = 30
# The dashed line's longest run right of the mark.
STALL_LEVEL_SPAN_BARS = 30
# The marker sits this many typical minute moves above (below) the bar.
STALL_MARK_OFFSET = 0.6

# A stall zone: at least this many bars with every close inside a band this
# many typical minute moves wide.
DEFAULT_ZONE_MIN_BARS = 20
DEFAULT_ZONE_WIDTH = 3.0
# A close this many typical minute moves past a zone's level, from the side
# price left on, ends its line.
ZONE_NEAR = 1.0


@dataclass(frozen=True)
class CandlePattern:
    key: str
    label: str
    # One line for a picker: what the overlay shows.
    summary: str


PATTERNS: "dict[str, CandlePattern]" = {
    FVG_KEY: CandlePattern(
        key=FVG_KEY,
        label="Fair Value Gap (FVG)",
        summary=(
            "Three candles where candle 1's wick and candle 3's wick don't overlap. "
            "The gap between them is boxed from candle 1 until price trades through "
            "it: green for a gap up, red for a gap down, faded once filled."
        ),
    ),
    STALL_KEY: CandlePattern(
        key=STALL_KEY,
        label="Volume-node stall",
        summary=(
            "Price ran into a price where a lot of volume already traded and its "
            "average momentum fell to about zero: ▼ where a rise stalled under "
            "resistance, ▲ where a fall stalled on support, with a dashed line at "
            "the node until price closes through it."
        ),
    ),
    ZONE_KEY: CandlePattern(
        key=ZONE_KEY,
        label="Stall-zone levels",
        summary=(
            "Stretches where price stayed in a narrow band for a while, boxed, with a "
            "line at the price that traded the most volume inside each (the peak of "
            "its own price profile): orange when price left below it (resistance "
            "overhead), blue when it left above (support), grey while still forming. "
            "The line runs right until price closes through it."
        ),
    ),
}


def keys() -> "list[str]":
    return list(PATTERNS)


def get(key: "str | None") -> "CandlePattern | None":
    return PATTERNS.get(key or "")


def label(key: "str | None") -> str:
    pattern = get(key)
    return pattern.label if pattern else str(key)


def compute(
    pattern_keys: "list[str] | tuple[str, ...] | None",
    bars: "list[dict]",
    min_size: float = DEFAULT_FVG_MIN_SIZE,
    hide_filled: bool = False,
    stall_bars: int = DEFAULT_STALL_BARS,
    minute_move: "float | Mapping[date, float | None] | None" = None,
    stall_approach: float = DEFAULT_STALL_APPROACH,
    stall_min_volume: float = DEFAULT_STALL_MIN_VOLUME,
    zone_min_bars: int = DEFAULT_ZONE_MIN_BARS,
    zone_width: float = DEFAULT_ZONE_WIDTH,
) -> "list[dict]":
    """Draw instructions for the requested patterns over `bars`.

    `bars` are 1-minute bars (`{"t","o","h","l","c",...}`), one session or
    several -- each exchange-local day is read on its own. Returns a flat list
    of items, each carrying the pattern `key`; `charts.add_candle_patterns` is
    the renderer.

    The stall reads `stall_bars` bars of momentum against `minute_move`, the
    typical one-minute move in dollars: one number, or one per exchange-local
    day for a replay of several; a day without one is measured on its own
    bars. `stall_approach` and `stall_min_volume` are its conditions 2 and 4.
    The stall zone reads the same `minute_move`, with `zone_min_bars` and
    `zone_width` its length and band.
    """
    wanted = [k for k in (pattern_keys or []) if k in PATTERNS]
    if not wanted or not bars:
        return []
    items: "list[dict]" = []
    for session in _regular_sessions(bars):
        if FVG_KEY in wanted:
            items.extend(_fvg_items(session, min_size, hide_filled))
        day = session[0]["_utc"].tz_convert(market_hours.MARKET_TZ).date()
        move = minute_move.get(day) if isinstance(minute_move, Mapping) else minute_move
        if STALL_KEY in wanted:
            items.extend(_stall_items(session, int(stall_bars), move, stall_approach, stall_min_volume))
        if ZONE_KEY in wanted:
            items.extend(_zone_items(session, move, int(zone_min_bars), zone_width))
    return items


def _regular_sessions(bars: "list[dict]") -> "list[list[dict]]":
    """`bars` split into exchange-local days, regular session only, in time order."""
    stamps = pd.to_datetime(pd.Series([b["t"] for b in bars]), utc=True)
    local = stamps.dt.tz_convert(market_hours.MARKET_TZ)
    clock = local.dt.time
    regular = (clock >= market_hours.MARKET_OPEN) & (clock < market_hours.MARKET_CLOSE)
    order = stamps[regular].sort_values().index
    days: "dict[object, list[dict]]" = {}
    for i in order:
        days.setdefault(local[i].date(), []).append({**bars[i], "_utc": stamps[i]})
    return [days[d] for d in sorted(days)]


def _fvg_items(session: "list[dict]", min_size: float, hide_filled: bool) -> "list[dict]":
    ranges = [float(b["h"]) - float(b["l"]) for b in session]
    items: "list[dict]" = []
    for gap in fair_value_gaps(session):
        i = gap["index"]
        before = ranges[max(0, i - FVG_RANGE_LOOKBACK):i]
        typical = sum(before) / len(before)
        size = gap["top"] - gap["bottom"]
        ratio = size / typical if typical > 0 else float("inf")
        if ratio < min_size:
            continue
        filled_at = gap["filled_at"]
        if hide_filled and filled_at is not None:
            continue
        items.append({
            "kind": "fvg",
            "key": FVG_KEY,
            "direction": gap["type"],
            # Candle 1, where the gap's near edge was set.
            "x0": _iso(session[i - 1]["_utc"]),
            # The bar that traded through it, or the session's last bar while
            # it is still open -- never past it: each day is read on its own, so
            # a replay of several days must not draw Monday's gap over Friday.
            "x1": _iso(session[-1 if filled_at is None else filled_at]["_utc"]),
            "y0": gap["bottom"],
            "y1": gap["top"],
            "size": round(size, 4),
            "size_ratio": None if ratio == float("inf") else round(ratio, 2),
            # Candle 3: the gap exists once it prints.
            "formed": _iso(session[i + 1]["_utc"]),
            "touched": (
                None if gap["touched_at"] is None
                else _iso(session[gap["touched_at"]]["_utc"])
            ),
            "filled": filled_at is not None,
        })
    return items


def _stall_items(
    session: "list[dict]",
    n: int,
    minute_move: "float | None",
    approach: float,
    min_volume_ratio: float,
) -> "list[dict]":
    """The stalls of one regular session; see "The stall" in the module doc."""
    count = len(session)
    if n < 1 or count < STALL_MIN_PROFILE_BARS + 2 * n:
        return []
    close = np.array([float(b["c"]) for b in session])
    high = np.array([float(b["h"]) for b in session])
    low = np.array([float(b["l"]) for b in session])
    volume = np.nan_to_num(np.array([float(b.get("v") or 0.0) for b in session]))
    units = _units(close, minute_move)

    items: "list[dict]" = []
    last_node: "float | None" = None
    # The first bar whose approach starts after the profile's minimum bars.
    for t in range(STALL_MIN_PROFILE_BARS - 1 + 2 * n, count):
        unit = units[t]
        if not unit > 0:
            continue
        if last_node is not None:
            if abs(close[t] - last_node) <= STALL_REARM * unit:
                continue
            last_node = None
        start = t - 2 * n
        stall = (close[t] - close[t - n]) / n
        before = (close[t - n] - close[start]) / n
        if abs(stall) >= MOMENTUM_NEUTRAL_FRACTION * unit:
            continue
        if before >= approach * unit:
            side = "resistance"
        elif before <= -approach * unit:
            side = "support"
        else:
            continue
        ratio = None
        prior = volume[max(0, start - STALL_VOLUME_LOOKBACK):start]
        if prior.size and prior.mean() > 0:
            ratio = float(volume[t - n + 1:t + 1].mean() / prior.mean())
        if min_volume_ratio > 0 and not (ratio is not None and ratio >= min_volume_ratio):
            continue

        found = profile_peaks(session[:start + 1])
        if found is None:
            continue
        centers, smoothed, peaks = found
        near = STALL_NEAR * unit
        if side == "resistance":
            lo, hi = close[t] - near, high[t - n:t + 1].max() + near
        else:
            lo, hi = low[t - n:t + 1].min() - near, close[t] + near
        candidates = [i for i in peaks if lo <= centers[i] <= hi]
        if not candidates:
            continue
        node_slice = min(candidates, key=lambda i: abs(centers[i] - close[t]))
        node = float(centers[node_slice])
        last_node = node
        around = np.abs(centers - node) <= near
        share = float(smoothed[around].sum() / smoothed.sum())

        sign = 1.0 if side == "resistance" else -1.0
        end = min(count - 1, t + STALL_LEVEL_SPAN_BARS)
        broken_at = next(
            (j for j in range(t + 1, end + 1) if sign * (close[j] - node) > near), None
        )
        mark_y = (
            high[t] + STALL_MARK_OFFSET * unit if side == "resistance"
            else low[t] - STALL_MARK_OFFSET * unit
        )
        items.append({
            "kind": "stall",
            "key": STALL_KEY,
            "side": side,
            "level": round(node, 4),
            "poc": node_slice == peaks[0],
            # Share of the profile's volume within STALL_NEAR of the node.
            "node_share": round(share, 4),
            # The mark: the bar the momentum stalled at.
            "x": _iso(session[t]["_utc"]),
            "y": round(float(mark_y), 4),
            # The line: from the stall's first bar to where price went through
            # the node, or as far as it is drawn.
            "x0": _iso(session[t - n]["_utc"]),
            "x1": _iso(session[end if broken_at is None else broken_at]["_utc"]),
            "broken": broken_at is not None,
            "bars": n,
            "unit": round(float(unit), 4),
            "near": round(float(near), 4),
            "approach": round(float(close[t - n] - close[start]), 4),
            "stall_move": round(float(close[t] - close[t - n]), 4),
            "volume_ratio": None if ratio is None else round(ratio, 2),
        })
    return items


def _units(close: np.ndarray, minute_move: "float | None") -> np.ndarray:
    """The typical minute move as of each bar: `minute_move` throughout when
    there is one, else the session's running mean |1-bar move| through that
    bar (NaN at the first)."""
    if minute_move and minute_move > 0:
        return np.full(len(close), float(minute_move))
    count = len(close)
    return np.concatenate([[np.nan], np.cumsum(np.abs(np.diff(close))) / np.arange(1, count)])


def _zone_items(
    session: "list[dict]",
    minute_move: "float | None",
    min_bars: int,
    width_k: float,
) -> "list[dict]":
    """The stall zones of one regular session; see "The stall zone" in the
    module doc."""
    count = len(session)
    if min_bars < 2 or count < min_bars:
        return []
    close = np.array([float(b["c"]) for b in session])
    volume = np.nan_to_num(np.array([float(b.get("v") or 0.0) for b in session]))
    units = _units(close, minute_move)
    cum_volume = np.cumsum(volume)

    items: "list[dict]" = []
    i = 0
    while i + min_bars <= count:
        unit = units[i + min_bars - 1]
        if not unit > 0:
            i += 1
            continue
        width = width_k * unit
        lo = hi = close[i]
        j = i
        while j + 1 < count and max(hi, close[j + 1]) - min(lo, close[j + 1]) <= width:
            j += 1
            lo, hi = min(lo, close[j]), max(hi, close[j])
        if j - i + 1 < min_bars:
            i += 1
            continue
        found = profile_peaks(session[i:j + 1])
        if found is not None:
            items.append(_zone_item(session, close, volume, cum_volume, i, j, lo, hi, unit, found))
        i = j + 1
    return items


def _zone_item(session, close, volume, cum_volume, i, j, lo, hi, unit, found) -> dict:
    """The draw instruction for the zone over bars `i`..`j`."""
    centers, _, peaks = found
    level = float(centers[peaks[0]])
    count = len(session)
    near = ZONE_NEAR * unit
    if j == count - 1:
        role, end, broken_at = "forming", j, None
    else:
        role = "resistance" if close[j + 1] < lo else "support"
        sign = 1.0 if role == "resistance" else -1.0
        broken_at = next(
            (k for k in range(j + 1, count) if sign * (close[k] - level) > near), None
        )
        end = count - 1 if broken_at is None else broken_at
    zone_volume = float(volume[i:j + 1].sum())
    return {
        "kind": "zone",
        "key": ZONE_KEY,
        "role": role,
        "level": round(level, 4),
        # The box: the zone's bars and close band.
        "x0": _iso(session[i]["_utc"]),
        "x_end": _iso(session[j]["_utc"]),
        "y0": round(float(lo), 4),
        "y1": round(float(hi), 4),
        # The line: from the zone's first bar to where a close went through the
        # level, or the session's last bar.
        "x1": _iso(session[end]["_utc"]),
        "broken": broken_at is not None,
        "left_at": None if role == "forming" else _iso(session[j + 1]["_utc"]),
        "bars": j - i + 1,
        "unit": round(float(unit), 4),
        "near": round(float(near), 4),
        "volume": round(zone_volume),
        # The zone's share of everything the session traded up to its last bar.
        "volume_share": round(zone_volume / cum_volume[j], 4) if cum_volume[j] > 0 else None,
    }


def _iso(stamp: pd.Timestamp) -> str:
    """UTC ISO-8601, the form `model_overlays` items carry too."""
    return pd.Timestamp(stamp).tz_convert("UTC").isoformat()
