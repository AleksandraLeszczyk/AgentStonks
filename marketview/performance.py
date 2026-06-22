"""
Portfolio value and agent performance tracking.

Replays the agent's recorded decisions (from `DecisionTracker`) against the
streamed price bars to reconstruct portfolio value at each bar. This is pure
post-hoc bookkeeping, independent of `DecisionTracker`'s live cash/position
state, so the equity curve can be recomputed at any point from just `bars` +
`tracker.snapshot()["decisions"]`.
"""
from __future__ import annotations

import bisect
from datetime import datetime
from typing import Any

import pandas as pd


def compute_equity_curve(
    bars: list[dict],
    decisions: list[dict],
    starting_cash: float,
    session_start: datetime,
) -> list[dict]:
    """One point per bar at/after `session_start`: portfolio value = cash + position * close.

    Before the first decision, the portfolio is `starting_cash` cash and no position.
    From each decision onward, value uses that decision's post-trade cash/position.
    """
    if not bars:
        return []

    df = pd.DataFrame(bars)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    df = df[df["t"] > pd.Timestamp(session_start)].sort_values("t")
    if df.empty:
        return []

    ordered = sorted(decisions, key=lambda d: d["ts"])
    decision_ts = [pd.Timestamp(d["ts"]) for d in ordered]

    points: list[dict] = []
    for row in df.itertuples():
        idx = bisect.bisect_right(decision_ts, row.t) - 1
        if idx >= 0:
            cash, position = ordered[idx]["cash_after"], ordered[idx]["position_after"]
        else:
            cash, position = starting_cash, 0.0
        price = float(row.c)
        points.append(
            {
                "ts": row.t.isoformat(),
                "price": price,
                "cash": cash,
                "position": position,
                "value": cash + position * price,
            }
        )
    return points


def decision_markers(decisions: list[dict], session_start: datetime) -> list[dict]:
    """Filled buy/sell decisions shaped for plotting on the equity curve (value, not price)."""
    cutoff = pd.Timestamp(session_start)
    markers: list[dict] = []
    for d in decisions:
        if d.get("status") != "filled" or d.get("price") is None:
            continue
        if pd.Timestamp(d["ts"]) <= cutoff:
            continue
        markers.append(
            {
                "ts": d["ts"],
                "action": d["action"],
                "value": d["cash_after"] + d["position_after"] * d["price"],
            }
        )
    return markers


def total_fees_paid(decisions: list[dict]) -> float:
    return sum(d.get("fee", 0.0) for d in decisions)


def summarize(points: list[dict], decisions: list[dict], starting_cash: float) -> dict[str, Any]:
    """Headline performance stats for the current equity curve."""
    current_value = points[-1]["value"] if points else starting_cash
    return {
        "starting_cash": starting_cash,
        "current_value": current_value,
        "return_pct": ((current_value / starting_cash) - 1.0) * 100 if starting_cash else 0.0,
        "total_fees": total_fees_paid(decisions),
    }
