"""Price zones read off the candles: order blocks and fair value gaps.

What is left of the ICT concepts the Smart Money agent was built on (removed
2026-10-10): the daily order blocks the Strategist reads as demand/supply
context, and the fair-value-gap detector the chart's candle-pattern overlay
draws. Liquidity pools, premium/discount and the composite setup read went
with the agent -- nothing else used them.
"""

from __future__ import annotations


def _order_block(bars: list[dict], i: int, kind: str, highs: list[float], lows: list[float]) -> dict:
    """One order-block zone descriptor anchored on the origin candle at index `i`.

    The zone is the full high-low range of that candle -- the price band
    institutions are presumed to defend on a return.
    """
    top = highs[i]
    bottom = lows[i]
    return {
        "type": kind,
        "index": i,
        "top": round(top, 4),
        "bottom": round(bottom, 4),
        "mid": round((top + bottom) / 2.0, 4),
        "timestamp": bars[i].get("t"),
    }

def find_order_blocks(bars: list[dict], swing: int = 5, lookahead: int = 3, max_blocks: int = 6) -> dict:
    """Locate institutional order blocks: the last opposing candle before a
    displacement move that breaks structure.

    A *bullish* order block is the last down-close candle before an up-move that
    takes out the prior `swing`-bar high (a bullish break of structure) -- the
    footprint of institutions absorbing supply before driving price up, and a
    zone they tend to defend on a return. A *bearish* order block is the mirror:
    the last up-close candle before a down-move that breaks the prior swing low.
    Each block's zone is the full high-low range of its origin candle. A block is
    "mitigated" once price has traded back into its zone after forming (its first
    defence has already been tested), which makes a *fresh, unmitigated* block the
    higher-quality one to trade a return into.
    """
    n = len(bars)
    if n < swing + lookahead + 1:
        return {"note": "not enough bars to locate order blocks", "order_blocks": []}

    o = [float(b["o"]) for b in bars]
    h = [float(b["h"]) for b in bars]
    l = [float(b["l"]) for b in bars]
    c = [float(b["c"]) for b in bars]

    blocks: list[dict] = []
    for i in range(swing, n - lookahead):
        swing_high = max(h[i - swing : i])
        swing_low = min(l[i - swing : i])
        impulse_high = max(h[i + 1 : i + 1 + lookahead])
        impulse_low = min(l[i + 1 : i + 1 + lookahead])
        if c[i] < o[i] and impulse_high > swing_high:
            blocks.append(_order_block(bars, i, "bullish", h, l))
        elif c[i] > o[i] and impulse_low < swing_low:
            blocks.append(_order_block(bars, i, "bearish", h, l))

    for blk in blocks:
        idx = blk["index"]
        # Mitigated if any bar *after* the displacement window traded back into the zone.
        blk["mitigated"] = any(
            l[j] <= blk["top"] and h[j] >= blk["bottom"] for j in range(idx + lookahead + 1, n)
        )
        blk["bars_ago"] = n - 1 - idx

    return {"order_blocks": blocks[-max_blocks:], "bar_count": n}

def _nearest_bullish_demand(blocks: list[dict], spot: float) -> "dict | None":
    """The bullish order block closest below (or containing) `spot` -- the demand
    zone price would return *down* into. Highest such zone wins (nearest support)."""
    candidates = [b for b in blocks if b["type"] == "bullish" and b["bottom"] <= spot]
    if not candidates:
        return None
    return max(candidates, key=lambda b: b["top"])

def _nearest_bearish_supply(blocks: list[dict], spot: float) -> "dict | None":
    """The bearish order block closest above `spot` -- the supply zone that makes
    a natural upside target. Lowest such zone wins (nearest overhead resistance)."""
    candidates = [b for b in blocks if b["type"] == "bearish" and b["top"] >= spot]
    if not candidates:
        return None
    return min(candidates, key=lambda b: b["bottom"])

def analyze_order_blocks(bars: list[dict], spot: "float | None" = None) -> dict:
    """Order-block read: every detected block plus the nearest bullish demand zone
    at/below price and the nearest bearish supply zone above it.

    The demand zone is the candidate entry on a return; the supply zone is a
    natural structural target. Returns labeled values plus a one-line summary.
    """
    found = find_order_blocks(bars)
    if "note" in found:
        return found
    blocks = found["order_blocks"]
    if spot is None:
        spot = float(bars[-1]["c"])

    demand = _nearest_bullish_demand(blocks, spot)
    supply = _nearest_bearish_supply(blocks, spot)

    summary_parts = [f"Found {len(blocks)} order block(s) over {found['bar_count']} bars; spot {spot:.2f}."]
    if demand is not None:
        state = "unmitigated" if not demand["mitigated"] else "mitigated"
        summary_parts.append(
            f"Nearest bullish demand block {demand['bottom']:.2f}-{demand['top']:.2f} "
            f"({state}, {demand['bars_ago']} bars ago)."
        )
    else:
        summary_parts.append("No bullish demand block at/below price.")
    if supply is not None:
        summary_parts.append(f"Nearest bearish supply block {supply['bottom']:.2f}-{supply['top']:.2f} (target).")

    return {
        "spot": round(spot, 4),
        "order_blocks": blocks,
        "nearest_bullish_ob": demand,
        "nearest_bearish_ob": supply,
        "summary": " ".join(summary_parts),
    }

def fair_value_gaps(bars: list[dict]) -> list[dict]:
    """Every fair value gap in `bars`, oldest first, with when it was revisited.

    The definition the chart's candle-pattern overlay (`candle_patterns`) draws.
    A *bullish* FVG forms when
    candle 3's low sits above candle 1's high (`low[i+1] > high[i-1]`), leaving the
    zone `(high[i-1], low[i+1])` that candle 2 crossed without the wicks either side
    overlapping. A *bearish* FVG is the mirror (`high[i+1] < low[i-1]`).

    `index` is candle 2. Two later moments are recorded, because "filled" means
    different things to different readers:

    `touched_at`   the first later bar that trades into the zone at all (price
                   came back to it).
    `filled_at`    the first later bar that trades through the far edge (below
                   a bullish gap's bottom, above a bearish gap's top), after which
                   there is no imbalance left to draw.

    Both are None while it has not happened yet.
    """
    h = [float(b["h"]) for b in bars]
    l = [float(b["l"]) for b in bars]
    n = len(bars)

    gaps: list[dict] = []
    for i in range(1, n - 1):
        if l[i + 1] > h[i - 1]:
            kind, bottom, top = "bullish", h[i - 1], l[i + 1]
        elif h[i + 1] < l[i - 1]:
            kind, bottom, top = "bearish", h[i + 1], l[i - 1]
        else:
            continue
        later = range(i + 2, n)
        touched = next((j for j in later if l[j] <= top and h[j] >= bottom), None)
        filled = next(
            (j for j in later if (l[j] <= bottom if kind == "bullish" else h[j] >= top)),
            None,
        )
        gaps.append({
            "type": kind, "index": i, "bottom": round(bottom, 4), "top": round(top, 4),
            "touched_at": touched, "filled_at": filled,
        })
    return gaps
