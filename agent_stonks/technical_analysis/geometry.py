"""Pure trade arithmetic: risk, reward, and the session clock.

Nothing here reads a bar. These take levels an agent has already chosen and
report what the trade they imply is worth -- which is why they are the only
functions in the package with no market data in their signature.
"""

from __future__ import annotations

from .. import clock, market_hours

_ET = market_hours.MARKET_TZ

from .indicators import atr


def session_time_window(latest_bar_ts: "str | None" = None) -> dict:
    """Classify the current point in the trading day for breakout timing discipline.

    Breakouts in the first 90 minutes or the final hour of the regular session
    are historically the most reliable; the 12:00-14:00 ET stretch is a
    notorious fakeout zone. Uses the timestamp of the latest bar (Alpaca bars
    are UTC ISO strings) when given, otherwise the current time.
    """
    dt = clock.parse_iso(latest_bar_ts) if latest_bar_ts else None
    if dt is None:
        dt = clock.now()
    et = dt.astimezone(_ET)
    minutes = et.hour * 60 + et.minute

    open_ = 9 * 60 + 30
    morning_end = 11 * 60
    dead_start = 12 * 60
    dead_end = 14 * 60
    power_hour_start = 15 * 60
    close_ = 16 * 60

    if minutes < open_ or minutes >= close_:
        window, favorable = "outside_regular_hours", False
        note = "Outside the 9:30-16:00 ET regular session."
    elif minutes < morning_end:
        window, favorable = "opening_window", True
        note = "First 90 minutes of the session -- historically the most reliable window for breakouts."
    elif dead_start <= minutes < dead_end:
        window, favorable = "midday_dead_zone", False
        note = "12:00-14:00 ET dead zone -- breakouts here are notoriously prone to fakeouts; demand stronger confirmation or stand aside."
    elif minutes >= power_hour_start:
        window, favorable = "power_hour", True
        note = "Final hour of the session -- a favorable window for breakouts."
    else:
        window, favorable = "other_session_hours", True
        note = "Mid-morning/early-afternoon -- acceptable but not the highest-conviction window."

    return {
        "et_time": et.strftime("%H:%M"),
        "window": window,
        "favorable_for_breakouts": favorable,
        "summary": f"{et.strftime('%H:%M')} ET -- {window.replace('_', ' ')}. {note}",
    }

def breakout_trade_geometry(
    entry: float,
    stop: float,
    base_height: "float | None" = None,
    atr: "float | None" = None,
    overhead_resistance: "float | None" = None,
) -> dict:
    """Mechanical entry/stop/target math for a long breakout trade.

    Projects targets by adding 1x and 2x the base height (the classic
    "measured move") and/or 1x and 2x ATR above entry, then expresses each as
    a reward-to-risk multiple of the entry-to-stop distance. `meets_min_reward_risk`
    flags whether the best available target clears the 2:1 minimum breakout
    traders require before taking the trade.

    `overhead_resistance` -- the nearest structural level above the entry (from
    key_levels) -- caps the realistic first target: `room_to_run` is true only
    when that ceiling sits at least 2x the stop distance above the entry.
    Buying with `room_to_run` false is buying into resistance; the better play
    is arming the entry at a break of that level instead.
    """
    if entry <= 0 or stop <= 0:
        return {"note": "entry and stop must be positive prices"}
    risk_per_share = entry - stop
    if risk_per_share <= 0:
        return {"note": "stop must be below entry for a long breakout setup"}

    targets: dict[str, float] = {}
    if base_height is not None and base_height > 0:
        targets["target1_base_height"] = round(entry + base_height, 4)
        targets["target2_base_height"] = round(entry + 2 * base_height, 4)
    if atr is not None and atr > 0:
        targets["target1_atr"] = round(entry + atr, 4)
        targets["target2_atr"] = round(entry + 2 * atr, 4)

    reward_risk: dict[str, float] = {}
    for key, target in targets.items():
        reward = target - entry
        reward_risk[key.replace("target", "rr")] = round(reward / risk_per_share, 2)

    best_rr = max(reward_risk.values()) if reward_risk else None
    meets_min_rr = best_rr is not None and best_rr >= 2.0

    room_to_run: "bool | None" = None
    rr_at_resistance: "float | None" = None
    resistance_note: "str | None" = None
    if overhead_resistance is not None and overhead_resistance > 0:
        if overhead_resistance <= entry:
            room_to_run = True
            resistance_note = (
                f"The given resistance {overhead_resistance:.2f} is at/below the entry -- "
                "already cleared, it does not cap the trade (it becomes support on a retest)."
            )
        else:
            rr_at_resistance = round((overhead_resistance - entry) / risk_per_share, 2)
            room_to_run = rr_at_resistance >= 2.0
            if room_to_run:
                resistance_note = (
                    f"Nearest overhead resistance {overhead_resistance:.2f} leaves "
                    f"{rr_at_resistance:.1f}:1 reward-to-risk -- room to run."
                )
            else:
                resistance_note = (
                    f"Nearest overhead resistance {overhead_resistance:.2f} caps reward at "
                    f"{rr_at_resistance:.1f}:1 -- below the 2:1 minimum. Do NOT buy into this "
                    "ceiling; arm the entry at a break of that level instead."
                )

    summary_parts = [f"Risk per share {risk_per_share:.2f} (entry {entry:.2f}, stop {stop:.2f})."]
    if targets:
        labelled = ", ".join(
            f"{k}={v:.2f} (R:R {reward_risk[k.replace('target', 'rr')]:.1f})" for k, v in targets.items()
        )
        summary_parts.append(labelled + ".")
        summary_parts.append(
            "Meets the 2:1 minimum reward-to-risk."
            if meets_min_rr
            else "Does NOT meet the 2:1 minimum reward-to-risk -- skip, or wait for a tighter stop/better entry."
        )
    else:
        summary_parts.append("No base_height or atr given -- cannot project a target.")
    if resistance_note:
        summary_parts.append(resistance_note)

    result = {
        "risk_per_share": round(risk_per_share, 4),
        **targets,
        **reward_risk,
        "best_reward_risk_ratio": best_rr,
        "meets_min_reward_risk": meets_min_rr,
        "summary": " ".join(summary_parts),
    }
    if room_to_run is not None:
        result["overhead_resistance"] = round(overhead_resistance, 4)
        result["rr_at_overhead_resistance"] = rr_at_resistance
        result["room_to_run"] = room_to_run
    return result

def vwap_reversion_geometry(
    entry: float,
    vwap: float,
    std_dev: float,
    side: str = "long",
    min_reward_risk: float = 1.5,
) -> dict:
    """Mechanical entry/stop/target math for a VWAP mean-reversion trade.

    The target is always VWAP (the mean price is expected to revert to). The
    stop sits one standard deviation beyond the entry -- i.e. past the next
    band -- so a long entered near the -2σ band stops out below -3σ, a short
    entered near +2σ stops out above +3σ. Returns the reward-to-risk ratio and
    whether it clears the mean-reversion minimum (1.5:1 by default; these are
    tighter-R:R, higher-win-rate trades than breakouts).
    """
    if entry <= 0 or vwap <= 0 or std_dev <= 0:
        return {"note": "entry, vwap, and std_dev must be positive"}
    side = side.lower()
    if side not in ("long", "short"):
        return {"note": "side must be 'long' or 'short'"}

    if side == "long":
        if entry >= vwap:
            return {"note": "for a long reversion, entry must be below VWAP (price stretched down to the band)"}
        stop = entry - std_dev
        target = vwap
        reward = target - entry
        risk = entry - stop
    else:  # short
        if entry <= vwap:
            return {"note": "for a short reversion, entry must be above VWAP (price stretched up to the band)"}
        stop = entry + std_dev
        target = vwap
        reward = entry - target
        risk = stop - entry

    reward_risk = reward / risk if risk > 0 else None
    meets_min_rr = reward_risk is not None and reward_risk >= min_reward_risk

    summary = (
        f"{side.capitalize()} reversion: entry {entry:.2f}, stop {stop:.2f} "
        f"(1σ={std_dev:.3f} beyond entry), target VWAP {target:.2f}. "
        f"Reward/risk {reward_risk:.2f}:1. "
        + (
            f"Meets the {min_reward_risk:.1f}:1 minimum."
            if meets_min_rr
            else f"Below the {min_reward_risk:.1f}:1 minimum -- skip or wait for a deeper stretch / closer entry."
        )
    )

    return {
        "side": side,
        "entry": round(entry, 4),
        "stop": round(stop, 4),
        "target": round(target, 4),
        "risk_per_share": round(risk, 4),
        "reward_per_share": round(reward, 4),
        "reward_risk_ratio": round(reward_risk, 2) if reward_risk is not None else None,
        "meets_min_reward_risk": meets_min_rr,
        "summary": summary,
    }
