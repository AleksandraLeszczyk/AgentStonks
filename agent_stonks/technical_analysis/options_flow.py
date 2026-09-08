"""What the options book says about where price is pinned.

Put and call walls, and the dealer gamma profile around them.
"""

from __future__ import annotations


def _gamma_flip_strike(strikes: list[float], net_gamma_by_strike: list[float]) -> "float | None":
    """Approximate zero-gamma level: the strike where cumulative net dealer gamma,
    scanned from the lowest strike up, crosses from negative to positive.

    Below this level dealer hedging tends to amplify moves (negative gamma);
    above it, dampen them (positive gamma). Linearly interpolated between the two
    bracketing strikes.
    """
    cumulative = 0.0
    prev_strike: "float | None" = None
    prev_cumulative: "float | None" = None
    for strike, gamma in zip(strikes, net_gamma_by_strike):
        cumulative += gamma
        if prev_cumulative is not None and prev_cumulative < 0 <= cumulative:
            span = cumulative - prev_cumulative
            if span:
                frac = -prev_cumulative / span
                return prev_strike + frac * (strike - prev_strike)
            return strike
        prev_strike, prev_cumulative = strike, cumulative
    return None

def _wall_trend(history: "list[dict] | None", key: str) -> "str | None":
    """Direction of `key` (call_wall/put_wall) across recorded snapshots, oldest to newest."""
    if not history or len(history) < 2:
        return None
    first = history[0].get(key)
    last = history[-1].get(key)
    if first is None or last is None:
        return None
    if last > first:
        return "rising"
    if last < first:
        return "falling"
    return "flat"

def get_put_call_walls_and_gamma(
    strikes: list[float],
    calls_oi: list[float],
    puts_oi: list[float],
    calls_gamma_exposure: list[float],
    puts_gamma_exposure: list[float],
    spot: float,
    wall_history: "list[dict] | None" = None,
) -> dict:
    """Options-derived support/resistance and dealer-gamma regime.

    The Call Wall (strike with the most call open interest) marks likely
    resistance; the Put Wall (most put open interest) marks likely support.
    Net dealer gamma across strikes says whether hedging flows should dampen
    price action (positive gamma) or amplify it (negative gamma) -- most
    dangerous right around a wall breach. `wall_history` (oldest-first
    `{"call_wall", "put_wall"}` snapshots) lets a rising call wall / falling put
    wall read as a bullish/bearish momentum tell, independent of current price.
    """
    if not strikes:
        return {"note": "no options chain data available"}

    call_wall_idx = max(range(len(strikes)), key=lambda i: calls_oi[i])
    put_wall_idx = max(range(len(strikes)), key=lambda i: puts_oi[i])
    call_wall = float(strikes[call_wall_idx])
    put_wall = float(strikes[put_wall_idx])

    net_gamma_by_strike = [c + p for c, p in zip(calls_gamma_exposure, puts_gamma_exposure)]
    total_net_gamma = sum(net_gamma_by_strike)
    gamma_regime = "positive (dampening)" if total_net_gamma >= 0 else "negative (amplifying)"
    gamma_flip = _gamma_flip_strike(strikes, net_gamma_by_strike)

    in_range = put_wall <= spot <= call_wall
    range_width = call_wall - put_wall
    range_position_pct = (spot - put_wall) / range_width * 100 if range_width > 0 else None

    near_call_wall = call_wall > 0 and abs(spot - call_wall) / call_wall < 0.01
    near_put_wall = put_wall > 0 and abs(spot - put_wall) / put_wall < 0.01

    insights: list[str] = []
    if in_range:
        insights.append(
            f"Spot {spot:.2f} sits inside the {put_wall:.2f}-{call_wall:.2f} put/call wall range"
            + (f" ({range_position_pct:.0f}% of the way from put wall to call wall)." if range_position_pct is not None else ".")
        )
    elif spot > call_wall:
        insights.append(
            f"Spot {spot:.2f} is already above the call wall ({call_wall:.2f}) -- resistance has "
            "been breached; a held break above often runs further as call gamma overhead thins out."
        )
    else:
        insights.append(
            f"Spot {spot:.2f} is already below the put wall ({put_wall:.2f}) -- support has given way; "
            "a held break below often accelerates as put gamma underneath thins out."
        )

    if near_call_wall:
        insights.append(f"Price is within 1% of the call wall ({call_wall:.2f}) -- watch for a stall or reversal here.")
    if near_put_wall:
        insights.append(f"Price is within 1% of the put wall ({put_wall:.2f}) -- watch for a bounce here.")

    if total_net_gamma >= 0:
        insights.append(
            "Net dealer gamma is positive: hedging flows tend to dampen moves (sell rallies, buy dips), "
            "favoring range-bound, mean-reverting price action between the walls."
        )
    else:
        insights.append(
            "Net dealer gamma is negative: hedging flows tend to amplify moves in the prevailing direction "
            "(buy strength, sell weakness), raising breakout/breakdown risk if a wall gives way."
        )

    call_wall_trend = _wall_trend(wall_history, "call_wall")
    put_wall_trend = _wall_trend(wall_history, "put_wall")
    if call_wall_trend == "rising":
        insights.append("Call wall has been rising -- bullish sentiment, resistance migrating higher.")
    elif call_wall_trend == "falling":
        insights.append("Call wall has been falling -- resistance compressing lower.")
    if put_wall_trend == "falling":
        insights.append("Put wall has been falling -- bearish sentiment, support migrating lower.")
    elif put_wall_trend == "rising":
        insights.append("Put wall has been rising -- support migrating higher, a bullish tell.")

    summary_parts = [
        f"Call wall {call_wall:.2f} (resistance), put wall {put_wall:.2f} (support), net gamma {gamma_regime}.",
    ]
    if gamma_flip is not None:
        summary_parts.append(f"Gamma flip near {gamma_flip:.2f}.")
    summary_parts.append(insights[0])

    return {
        "call_wall": call_wall,
        "put_wall": put_wall,
        "spot": spot,
        "gamma_flip": gamma_flip,
        "net_gamma": total_net_gamma,
        "gamma_regime": gamma_regime,
        "in_range": in_range,
        "range_position_pct": round(range_position_pct, 1) if range_position_pct is not None else None,
        "call_wall_trend": call_wall_trend,
        "put_wall_trend": put_wall_trend,
        "insights": insights,
        "summary": " ".join(summary_parts),
    }
