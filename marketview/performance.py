"""
Portfolio value and agent performance tracking.

Replays the agent's recorded decisions (from `DecisionTracker`) against the
streamed price bars of every symbol to reconstruct portfolio value over time.
This is pure post-hoc bookkeeping, independent of `DecisionTracker`'s live
cash/position state, so the equity curve can be recomputed at any point from
just the per-symbol bars + `tracker.snapshot()["decisions"]`.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd


def _decision_positions(decision: dict) -> dict[str, float]:
    """Positions snapshot after a decision. Falls back to the single-symbol
    `position_after` field for decisions recorded before multi-symbol support."""
    positions = decision.get("positions_after")
    if isinstance(positions, dict) and positions:
        return {str(s): float(q) for s, q in positions.items()}
    symbol = str(decision.get("symbol") or "")
    return {symbol: float(decision.get("position_after") or 0.0)} if symbol else {}


def _bar_events(
    bars_by_symbol: dict[str, list[dict]], session_start: datetime
) -> list[tuple[pd.Timestamp, str, float]]:
    """(ts, symbol, close) for every bar after `session_start`, time-ordered."""
    cutoff = pd.Timestamp(session_start)
    events: list[tuple[pd.Timestamp, str, float]] = []
    for symbol, bars in bars_by_symbol.items():
        for bar in bars:
            try:
                ts = pd.Timestamp(bar["t"])
                close = float(bar["c"])
            except (KeyError, TypeError, ValueError):
                continue
            if ts > cutoff:
                events.append((ts, symbol, close))
    events.sort(key=lambda e: e[0])
    return events


def compute_equity_curve(
    bars_by_symbol: dict[str, list[dict]],
    decisions: list[dict],
    starting_cash: float,
    session_start: datetime,
    live_prices: dict[str, float] | None = None,
) -> list[dict]:
    """One point per bar (of any symbol) at/after `session_start`: portfolio
    value = cash + every position marked to its symbol's latest known close,
    plus a trailing "now" point priced off `live_prices` so the curve keeps
    advancing between bar closes and agent decisions.

    Before the first decision, the portfolio is `starting_cash` cash and no
    positions. From each decision onward, value uses that decision's post-trade
    cash and per-symbol positions; a decision's own fill price also updates the
    marking price of its symbol.
    """
    ordered = sorted(decisions, key=lambda d: d["ts"])
    decision_ts = [pd.Timestamp(d["ts"]) for d in ordered]

    cash = starting_cash
    positions: dict[str, float] = {}
    prices: dict[str, float] = {}
    di = 0

    def _apply_decisions_up_to(ts: pd.Timestamp) -> None:
        nonlocal cash, positions, di
        while di < len(ordered) and decision_ts[di] <= ts:
            d = ordered[di]
            cash = float(d.get("cash_after") or 0.0)
            positions = _decision_positions(d)
            if d.get("price") is not None and d.get("symbol"):
                prices[str(d["symbol"])] = float(d["price"])
            di += 1

    def _value() -> float:
        return cash + sum(
            qty * prices[sym] for sym, qty in positions.items() if qty and sym in prices
        )

    points: list[dict] = []
    for ts, symbol, close in _bar_events(bars_by_symbol, session_start):
        _apply_decisions_up_to(ts)
        prices[symbol] = close
        points.append(
            {
                "ts": ts.isoformat(),
                "price": close,
                "cash": cash,
                "position": positions.get(symbol, 0.0),
                "value": _value(),
            }
        )

    if live_prices:
        _apply_decisions_up_to(pd.Timestamp.now(tz="UTC"))
        for sym, price in live_prices.items():
            if price is not None:
                prices[sym] = float(price)
        points.append(
            {
                "ts": pd.Timestamp.now(tz="UTC").isoformat(),
                "price": next(iter(live_prices.values()), None),
                "cash": cash,
                "position": sum(positions.values()),
                "value": _value(),
            }
        )
    return points


def decision_markers(
    decisions: list[dict],
    session_start: datetime,
    points: list[dict] | None = None,
) -> list[dict]:
    """Filled buy/sell decisions -- plus tactics-armed moments -- shaped for
    plotting on the equity curve (value, not price). When the computed curve
    `points` are provided, each marker sits on the curve value at its time;
    otherwise the marker value is approximated from the decision's own fill
    price and positions snapshot."""
    cutoff = pd.Timestamp(session_start)
    point_ts = [pd.Timestamp(p["ts"]) for p in (points or [])]
    markers: list[dict] = []
    for d in decisions:
        is_tactics = d.get("action") == "tactics"
        if (d.get("status") != "filled" and not is_tactics) or d.get("price") is None:
            continue
        ts = pd.Timestamp(d["ts"])
        if ts <= cutoff:
            continue
        value = None
        if point_ts:
            # Nearest curve point at/after the decision, so the marker sits on
            # the plotted curve even when other symbols move the total value.
            idx = min(range(len(point_ts)), key=lambda i: abs((point_ts[i] - ts).total_seconds()))
            value = points[idx]["value"]
        if value is None:
            positions = _decision_positions(d)
            value = float(d.get("cash_after") or 0.0) + positions.get(
                str(d.get("symbol") or ""), 0.0
            ) * float(d["price"])
        markers.append(
            {
                "ts": d["ts"],
                "action": d["action"],
                "symbol": d.get("symbol"),
                "value": value,
                "label": " · ".join(d.get("tactics") or []) if is_tactics else None,
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
