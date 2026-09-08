"""The ICT / \"smart money\" concepts, as one family.

Order blocks, fair-value gaps, liquidity pools and premium/discount: one
vocabulary with its own definitions, kept together because the setup read at
the bottom is assembled out of all of them.
"""

from __future__ import annotations


from .indicators import _rejection_candle, analyze_trend


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

def find_fair_value_gaps(bars: list[dict], max_gaps: int = 6) -> dict:
    """Locate fair value gaps (FVGs): three-candle price imbalances institutions
    tend to revisit.

    A *bullish* FVG forms when a strong up-candle leaves a gap between the high of
    the candle before it and the low of the candle after it (`low[i+1] > high[i-1]`);
    the gap zone `(high[i-1], low[i+1])` is an unfilled imbalance that often acts as
    support on a pullback. A *bearish* FVG is the mirror (`high[i+1] < low[i-1]`).
    A gap is "filled" once a later bar trades back through the zone.
    """
    n = len(bars)
    if n < 3:
        return {"note": "not enough bars to locate fair value gaps", "fair_value_gaps": []}

    h = [float(b["h"]) for b in bars]
    l = [float(b["l"]) for b in bars]

    gaps: list[dict] = []
    for i in range(1, n - 1):
        if l[i + 1] > h[i - 1]:
            gaps.append({"type": "bullish", "index": i, "bottom": round(h[i - 1], 4), "top": round(l[i + 1], 4)})
        elif h[i + 1] < l[i - 1]:
            gaps.append({"type": "bearish", "index": i, "bottom": round(h[i + 1], 4), "top": round(l[i - 1], 4)})

    for g in gaps:
        idx = g["index"]
        g["filled"] = any(l[j] <= g["top"] and h[j] >= g["bottom"] for j in range(idx + 2, n))
        g["bars_ago"] = n - 1 - idx

    return {"fair_value_gaps": gaps[-max_gaps:], "bar_count": n}

def analyze_fair_value_gaps(bars: list[dict], spot: "float | None" = None) -> dict:
    """Fair-value-gap read: detected gaps plus the nearest bullish FVG at/below
    price (a support imbalance price may be filling now). Returns a one-line summary."""
    found = find_fair_value_gaps(bars)
    if "note" in found:
        return found
    gaps = found["fair_value_gaps"]
    if spot is None:
        spot = float(bars[-1]["c"])

    bullish_below = [g for g in gaps if g["type"] == "bullish" and g["bottom"] <= spot * 1.001]
    nearest = max(bullish_below, key=lambda g: g["top"]) if bullish_below else None

    summary_parts = [f"Found {len(gaps)} fair value gap(s); spot {spot:.2f}."]
    if nearest is not None:
        state = "filled" if nearest["filled"] else "unfilled"
        summary_parts.append(f"Nearest bullish FVG {nearest['bottom']:.2f}-{nearest['top']:.2f} ({state}).")
    else:
        summary_parts.append("No bullish FVG at/below price.")

    return {
        "spot": round(spot, 4),
        "fair_value_gaps": gaps,
        "nearest_bullish_fvg": nearest,
        "summary": " ".join(summary_parts),
    }

def _intraday_break_of_structure(bars: list[dict], swing: int = 5) -> bool:
    """Whether intraday price has made a bullish break of structure and is holding it:
    the recent `swing` bars took out the prior `swing`-bar high and price still sits
    above that broken level. This is the lightweight 'breaker / structure flip'
    confirmation -- old resistance reclaimed as support."""
    if len(bars) < 2 * swing:
        return False
    h = [float(b["h"]) for b in bars]
    c = [float(b["c"]) for b in bars]
    prior_high = max(h[-2 * swing : -swing])
    recent_high = max(h[-swing:])
    return recent_high > prior_high and c[-1] > prior_high

def smart_money_trade_geometry(
    entry: float, stop: float, target: float, min_reward_risk: float = 3.0
) -> dict:
    """Mechanical entry/stop/target math for a long Smart Money setup.

    Entry is at the higher-timeframe demand (order block) on a return, the stop
    sits just beyond the block, and the target is the next opposing structural
    level. `meets_min_reward_risk` flags whether the reward-to-risk clears the
    3:1 minimum the Smart Money setup demands (it typically runs 3:1 to 5:1).
    """
    if entry <= 0 or stop <= 0 or target <= 0:
        return {"note": "entry, stop, and target must be positive prices"}
    if not (stop < entry < target):
        return {"note": "a long smart-money setup needs stop < entry < target"}

    risk = entry - stop
    reward = target - entry
    reward_risk = reward / risk if risk > 0 else None
    meets_min_rr = reward_risk is not None and reward_risk >= min_reward_risk

    summary = (
        f"Long smart-money setup: entry {entry:.2f}, stop {stop:.2f} (just beyond the block), "
        f"target {target:.2f}. Reward/risk {reward_risk:.2f}:1. "
        + (
            f"Meets the {min_reward_risk:.0f}:1 minimum."
            if meets_min_rr
            else f"Below the {min_reward_risk:.0f}:1 minimum -- skip, or wait for a deeper return into the block."
        )
    )

    return {
        "entry": round(entry, 4),
        "stop": round(stop, 4),
        "target": round(target, 4),
        "risk_per_share": round(risk, 4),
        "reward_per_share": round(reward, 4),
        "reward_risk_ratio": round(reward_risk, 2) if reward_risk is not None else None,
        "meets_min_reward_risk": meets_min_rr,
        "summary": summary,
    }

def analyze_premium_discount(
    bars: list[dict], lookback: int = 20, spot: "float | None" = None
) -> dict:
    """Premium / discount read over the recent dealing range.

    Smart Money buys in *discount* (below the range midpoint, "equilibrium") and
    sells in *premium* (above it). The dealing range is the highest high and
    lowest low over the last `lookback` bars; its midpoint is equilibrium. A
    return into a bullish demand block that *also* sits in discount is a
    higher-quality long than the same block sitting in premium -- price is cheap
    relative to where institutions accumulated. The deep-discount "OTE" zone is
    the 0.618-0.79 retracement down from the range high, where institutional
    longs are statistically filled.
    """
    if not bars or len(bars) < 3:
        return {"note": "not enough bars for a premium/discount read"}

    window = bars[-lookback:] if len(bars) >= lookback else bars
    range_high = max(float(b["h"]) for b in window)
    range_low = min(float(b["l"]) for b in window)
    rng = range_high - range_low
    if rng <= 0:
        return {"note": "flat dealing range; premium/discount undefined"}

    if spot is None:
        spot = float(bars[-1]["c"])
    equilibrium = (range_high + range_low) / 2.0
    # 0.0 = range low, 1.0 = range high.
    position = (spot - range_low) / rng

    if position < 0.45:
        zone = "discount"
    elif position > 0.55:
        zone = "premium"
    else:
        zone = "equilibrium"

    # Optimal Trade Entry: the 0.618-0.79 retracement down from the high.
    ote_top = round(range_high - 0.618 * rng, 4)
    ote_bottom = round(range_high - 0.79 * rng, 4)
    in_ote = ote_bottom <= spot <= ote_top

    summary = (
        f"Dealing range {range_low:.2f}-{range_high:.2f}, equilibrium {equilibrium:.2f}; "
        f"spot {spot:.2f} is in the {zone} zone ({position * 100:.0f}% of range). "
        + (
            "Inside the deep-discount OTE zone -- prime institutional long area."
            if in_ote
            else ("Below equilibrium -- favourable for longs." if zone == "discount"
                  else "At/above equilibrium -- longs are buying retail-expensive prices.")
        )
    )

    return {
        "spot": round(spot, 4),
        "range_high": round(range_high, 4),
        "range_low": round(range_low, 4),
        "equilibrium": round(equilibrium, 4),
        "range_position": round(position, 3),
        "zone": zone,
        "in_discount": zone == "discount",
        "ote_zone": {"bottom": ote_bottom, "top": ote_top},
        "in_ote_zone": in_ote,
        "summary": summary,
    }

def _cluster_levels(levels: list[float], tol_pct: float) -> list[dict]:
    """Group near-equal price levels into liquidity pools.

    Levels within `tol_pct` of a running cluster anchor are merged; a pool with
    two or more members is an *equal-highs/lows* cluster -- a stronger resting
    pool of liquidity (more stops bunched at one price)."""
    pools: list[dict] = []
    for price in sorted(levels):
        if pools and price <= pools[-1]["anchor"] * (1 + tol_pct):
            pool = pools[-1]
            pool["members"].append(price)
            pool["price"] = sum(pool["members"]) / len(pool["members"])
        else:
            pools.append({"anchor": price, "price": price, "members": [price]})
    return [
        {"price": round(p["price"], 4), "count": len(p["members"]), "equal": len(p["members"]) >= 2}
        for p in pools
    ]

def analyze_liquidity(
    bars: list[dict], swing: int = 3, tol_pct: float = 0.0015, recent: int = 5, spot: "float | None" = None
) -> dict:
    """Liquidity pools and recent sweeps -- the core Smart Money 'stop hunt' read.

    Liquidity rests where retail stops cluster: just above swing highs (buy-side
    liquidity, BSL) and just below swing lows (sell-side liquidity, SSL).
    Institutions push price through these pools to fill large orders, then
    reverse -- a *liquidity sweep* / stop run. A bullish SSL sweep (price
    undercuts a prior swing low then closes back above it) is exactly the trap
    that precedes an institutional markup, and one of the strongest
    confirmations for a long off a demand block. Near-equal highs/lows within
    `tol_pct` are merged into a single, stronger resting-liquidity pool.
    """
    n = len(bars)
    if n < 2 * swing + 2:
        return {"note": "not enough bars to locate liquidity", "buy_side_liquidity": [], "sell_side_liquidity": []}

    h = [float(b["h"]) for b in bars]
    l = [float(b["l"]) for b in bars]
    c = [float(b["c"]) for b in bars]

    swing_highs: list[tuple[int, float]] = []
    swing_lows: list[tuple[int, float]] = []
    for i in range(swing, n - swing):
        if h[i] == max(h[i - swing : i + swing + 1]):
            swing_highs.append((i, h[i]))
        if l[i] == min(l[i - swing : i + swing + 1]):
            swing_lows.append((i, l[i]))

    if spot is None:
        spot = float(bars[-1]["c"])

    bsl = _cluster_levels([p for _, p in swing_highs], tol_pct)  # buy-side (above)
    ssl = _cluster_levels([p for _, p in swing_lows], tol_pct)  # sell-side (below)

    bsl_above = [pool for pool in bsl if pool["price"] >= spot]
    ssl_below = [pool for pool in ssl if pool["price"] <= spot]
    nearest_bsl = min(bsl_above, key=lambda p: p["price"]) if bsl_above else None
    nearest_ssl = max(ssl_below, key=lambda p: p["price"]) if ssl_below else None

    # Recent sweep: a bar in the last `recent` that pierced a *prior* swing level
    # and closed back on the other side of it (the reversal that defines a sweep).
    recent_sweep: "dict | None" = None
    for j in range(max(swing, n - recent), n):
        prior_lows = [pl for k, pl in swing_lows if k <= j - 1]
        for pl in prior_lows:
            if l[j] < pl and c[j] > pl:
                recent_sweep = {"type": "bullish", "level": round(pl, 4), "bars_ago": n - 1 - j}
                break
        if recent_sweep is not None:
            continue
        prior_highs = [ph for k, ph in swing_highs if k <= j - 1]
        for ph in prior_highs:
            if h[j] > ph and c[j] < ph:
                recent_sweep = {"type": "bearish", "level": round(ph, 4), "bars_ago": n - 1 - j}
                break

    bullish_sweep = recent_sweep is not None and recent_sweep["type"] == "bullish"

    parts = [f"{len(bsl)} buy-side and {len(ssl)} sell-side liquidity pool(s); spot {spot:.2f}."]
    if nearest_bsl is not None:
        eq = " (equal highs)" if nearest_bsl["equal"] else ""
        parts.append(f"Nearest overhead BSL {nearest_bsl['price']:.2f}{eq} -- liquidity/target above.")
    if nearest_ssl is not None:
        eq = " (equal lows)" if nearest_ssl["equal"] else ""
        parts.append(f"Nearest SSL {nearest_ssl['price']:.2f}{eq} below -- stops resting there.")
    if recent_sweep is not None:
        parts.append(
            f"Recent {recent_sweep['type']} sweep of {recent_sweep['level']:.2f} "
            f"({recent_sweep['bars_ago']} bars ago)"
            + (" -- bullish stop-run, supports a long." if bullish_sweep else ".")
        )
    else:
        parts.append("No recent sweep.")

    return {
        "spot": round(spot, 4),
        "buy_side_liquidity": bsl,
        "sell_side_liquidity": ssl,
        "nearest_bsl_above": nearest_bsl,
        "nearest_ssl_below": nearest_ssl,
        "recent_sweep": recent_sweep,
        "bullish_sweep": bullish_sweep,
        "summary": " ".join(parts),
    }

def analyze_smart_money_setup(
    daily_bars: list[dict],
    intraday_bars: "list[dict] | None" = None,
    spot: "float | None" = None,
    min_reward_risk: float = 3.0,
    stop_buffer: float = 0.1,
) -> dict:
    """The composite Smart Money setup: a higher-timeframe bullish order block that
    price is returning into, confirmed by an intraday signal.

    Combines the higher-timeframe structure (daily order blocks + trend regime)
    with intraday confirmation (a bullish rejection candle, a bullish FVG that price
    has filled, or an intraday break-of-structure/breaker). A `long_setup` requires
    all of: a bullish demand block at/below price, a non-bearish daily regime, price
    actually inside that block, at least one intraday confirmation, and a target
    (the next opposing structural level) that clears the 3:1 reward-to-risk minimum.
    Anything short of that with a valid block reads as `watching`; no qualifying
    block at all is `no_setup`.
    """
    if not daily_bars or len(daily_bars) < 8:
        return {"note": "not enough daily bars for a smart-money structural read", "signal": "no_setup"}

    if spot is None:
        ref = intraday_bars if intraday_bars else daily_bars
        spot = float(ref[-1]["c"])

    ob = analyze_order_blocks(daily_bars, spot=spot)
    demand = ob.get("nearest_bullish_ob")
    supply = ob.get("nearest_bearish_ob")
    regime = analyze_trend(daily_bars).get("regime", "neutral")

    price_in_ob = demand is not None and demand["bottom"] <= spot <= demand["top"]

    # Premium/discount context: smart money buys discount (below equilibrium).
    pd_read = analyze_premium_discount(daily_bars, spot=spot)
    in_discount = bool(pd_read.get("in_discount"))

    # Intraday confirmation signals (any one qualifies; more = higher quality).
    confirmations: list[str] = []
    nearest_fvg = None
    liquidity = None
    if intraday_bars and len(intraday_bars) >= 3:
        if _rejection_candle(intraday_bars[-1]) == "bullish_rejection":
            confirmations.append("rejection_candle")
        fvg = analyze_fair_value_gaps(intraday_bars, spot=spot)
        nearest_fvg = fvg.get("nearest_bullish_fvg")
        if nearest_fvg is not None and nearest_fvg.get("filled"):
            confirmations.append("fvg_fill")
        if _intraday_break_of_structure(intraday_bars):
            confirmations.append("breaker")
        liquidity = analyze_liquidity(intraday_bars, spot=spot)
        if liquidity.get("bullish_sweep"):
            confirmations.append("liquidity_sweep")

    # Entry, stop, and target geometry.
    entry = round(spot if price_in_ob else (demand["top"] if demand else spot), 4)
    suggested_stop = None
    structural_target = None
    geometry = None
    if demand is not None:
        zone_height = demand["top"] - demand["bottom"]
        suggested_stop = round(demand["bottom"] - stop_buffer * max(zone_height, entry * 0.001), 4)
        if supply is not None and supply["bottom"] > entry:
            structural_target = supply["bottom"]
        else:
            recent_high = max(b["h"] for b in daily_bars[-20:])
            structural_target = round(recent_high, 4) if recent_high > entry else None
        if structural_target is not None:
            geometry = smart_money_trade_geometry(entry, suggested_stop, structural_target, min_reward_risk)

    meets_rr = bool(geometry and geometry.get("meets_min_reward_risk"))

    if demand is None or regime == "bearish":
        signal = "no_setup"
    elif price_in_ob and confirmations and meets_rr:
        signal = "long_setup"
    else:
        signal = "watching"

    if signal == "long_setup" and len(confirmations) >= 2 and regime == "bullish" and in_discount and (
        geometry and geometry["reward_risk_ratio"] >= 4.0
    ):
        quality = "A+"
    elif signal == "long_setup":
        quality = "B"
    else:
        quality = "C"

    summary_parts = [f"HTF regime {regime}; spot {spot:.2f}."]
    if demand is not None:
        loc = "inside" if price_in_ob else ("above" if spot > demand["top"] else "below")
        summary_parts.append(
            f"Bullish demand block {demand['bottom']:.2f}-{demand['top']:.2f} "
            f"({'unmitigated' if not demand['mitigated'] else 'mitigated'}); price is {loc} it."
        )
    else:
        summary_parts.append("No bullish demand block at/below price.")
    summary_parts.append(f"Price in {pd_read.get('zone', 'n/a')} zone (eq {pd_read.get('equilibrium', float('nan')):.2f}).")
    summary_parts.append(
        f"Intraday confirmation: {', '.join(confirmations) if confirmations else 'none'}."
    )
    if geometry is not None and "reward_risk_ratio" in geometry:
        summary_parts.append(
            f"Geometry entry {entry:.2f} / stop {suggested_stop:.2f} / target {structural_target:.2f} "
            f"= {geometry['reward_risk_ratio']:.1f}:1."
        )
    summary_parts.append(
        {
            "long_setup": f"LONG setup ({quality}): return into demand with confirmation and ≥{min_reward_risk:.0f}:1 target.",
            "watching": "Watching: a valid demand block exists but price/confirmation/RR isn't all there yet.",
            "no_setup": "No setup: no bullish demand block at/below price, or the daily regime is bearish.",
        }[signal]
    )

    return {
        "signal": signal,
        "quality": quality,
        "htf_regime": regime,
        "spot": round(spot, 4),
        "order_block": demand,
        "supply_block": supply,
        "price_in_order_block": price_in_ob,
        "premium_discount_zone": pd_read.get("zone"),
        "in_discount": in_discount,
        "equilibrium": pd_read.get("equilibrium"),
        "intraday_confirmation": confirmations,
        "confirmed": bool(confirmations),
        "recent_sweep": liquidity.get("recent_sweep") if liquidity else None,
        "nearest_bullish_fvg": nearest_fvg,
        "suggested_entry": entry if demand is not None else None,
        "suggested_stop": suggested_stop,
        "structural_target": structural_target,
        "reward_risk_to_target": geometry["reward_risk_ratio"] if geometry and "reward_risk_ratio" in geometry else None,
        "meets_min_reward_risk": meets_rr,
        "summary": " ".join(summary_parts),
    }
