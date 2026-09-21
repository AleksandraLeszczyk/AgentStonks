"""Candle patterns, in the shape a price chart can draw.

The sibling of `model_overlays`: that module draws what a trained model *said*
about a session, this one draws what the tape itself *did* -- shapes read off
the candles with no model and no forecast. Keeping the two apart keeps the
pickers honest: nothing on the "Model Predictions" list is a pattern, and
nothing here claims to predict.

One pattern so far
------------------
`fvg`   Fair value gap: three candles where candle 1's wick and candle 3's
        wick do not overlap, leaving a price zone candle 2 crossed without
        trading on both sides of it. Drawn as a box from candle 1 over the
        gap's price range, reaching right until price trades through its far
        edge (then faded and ended at that bar) or to the session's last bar
        while it is still open. Detection is
        `technical_analysis.fair_value_gaps`, the same definition the agent's
        `analyze_fair_value_gaps` tool reads.

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
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import market_hours
from .technical_analysis import fair_value_gaps

FVG_KEY = "fvg"

# The bars before candle 2 whose mean high-low range sizes a gap.
FVG_RANGE_LOOKBACK = 14
DEFAULT_FVG_MIN_SIZE = 0.5


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
) -> "list[dict]":
    """Draw instructions for the requested patterns over `bars`.

    `bars` are 1-minute bars (`{"t","o","h","l","c",...}`), one session or
    several -- each exchange-local day is read on its own. Returns a flat list
    of items, each carrying the pattern `key`; `charts.add_candle_patterns` is
    the renderer.
    """
    wanted = [k for k in (pattern_keys or []) if k in PATTERNS]
    if not wanted or not bars:
        return []
    items: "list[dict]" = []
    for session in _regular_sessions(bars):
        if FVG_KEY in wanted:
            items.extend(_fvg_items(session, min_size, hide_filled))
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


def _iso(stamp: pd.Timestamp) -> str:
    """UTC ISO-8601, the form `model_overlays` items carry too."""
    return pd.Timestamp(stamp).tz_convert("UTC").isoformat()
