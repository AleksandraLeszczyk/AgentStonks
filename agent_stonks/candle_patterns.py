"""Candle patterns, in the shape a price chart can draw.

The sibling of `model_overlays`: that module draws what a trained model *said*
about a session, this one draws what the tape itself *did* -- shapes read off
the candles with no model and no forecast. Keeping the two apart keeps the
pickers honest: nothing on the "Model Predictions" list is a pattern, and
nothing here claims to predict.

Two patterns
------------
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
    """
    wanted = [k for k in (pattern_keys or []) if k in PATTERNS]
    if not wanted or not bars:
        return []
    items: "list[dict]" = []
    for session in _regular_sessions(bars):
        if FVG_KEY in wanted:
            items.extend(_fvg_items(session, min_size, hide_filled))
        if STALL_KEY in wanted:
            day = session[0]["_utc"].tz_convert(market_hours.MARKET_TZ).date()
            move = minute_move.get(day) if isinstance(minute_move, Mapping) else minute_move
            items.extend(_stall_items(session, int(stall_bars), move, stall_approach, stall_min_volume))
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
    # Running mean |1-bar move| through each bar: the fallback unit.
    own_unit = np.concatenate([[np.nan], np.cumsum(np.abs(np.diff(close))) / np.arange(1, count)])
    fixed_unit = float(minute_move) if minute_move and minute_move > 0 else None

    items: "list[dict]" = []
    last_node: "float | None" = None
    # The first bar whose approach starts after the profile's minimum bars.
    for t in range(STALL_MIN_PROFILE_BARS - 1 + 2 * n, count):
        unit = fixed_unit or own_unit[t]
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


def _iso(stamp: pd.Timestamp) -> str:
    """UTC ISO-8601, the form `model_overlays` items carry too."""
    return pd.Timestamp(stamp).tz_convert("UTC").isoformat()
