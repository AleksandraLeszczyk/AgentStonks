"""
Portfolio value and agent performance tracking.

Replays the agent's recorded decisions (from `DecisionTracker`) against the
streamed price bars of every symbol to reconstruct portfolio value over time.
This is pure post-hoc bookkeeping, independent of `DecisionTracker`'s live
cash/position state, so the equity curve can be recomputed at any point from
just the per-symbol bars + `tracker.snapshot()["decisions"]`.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from itertools import groupby
from typing import Any

import pandas as pd

from . import clock


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
        now = pd.Timestamp(clock.now())
        _apply_decisions_up_to(now)
        for sym, price in live_prices.items():
            if price is not None:
                prices[sym] = float(price)
        points.append(
            {
                "ts": now.isoformat(),
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
                # For the marker's hover, as on the live chart's.
                "price": d.get("price"),
                "filled_quantity": d.get("filled_quantity"),
                "reasoning": d.get("reasoning"),
            }
        )
    return markers


def total_fees_paid(decisions: list[dict]) -> float:
    return sum(d.get("fee", 0.0) for d in decisions)


def summarize(
    points: list[dict],
    decisions: list[dict],
    starting_cash: float,
    current_value: float | None = None,
) -> dict[str, Any]:
    """Headline performance stats for the current equity curve.

    `current_value` overrides the curve's last point. Pass it when something
    else is authoritative about what the portfolio is worth -- a real broker
    account, which knows about holdings and cash movements this curve is
    reconstructed without -- so the headline figure matches the account rather
    than the replay.
    """
    if current_value is None:
        current_value = points[-1]["value"] if points else starting_cash
    return {
        "starting_cash": starting_cash,
        "current_value": current_value,
        "return_pct": ((current_value / starting_cash) - 1.0) * 100 if starting_cash else 0.0,
        "total_fees": total_fees_paid(decisions),
    }


# --- per instrument -------------------------------------------------------
# The Portfolio tab splits the basket by symbol: what each one holds, what it
# cost, and what it has made. Every figure here is replayed from the ledger's
# own fills, on average cost.


def _is_fill(decision: dict) -> bool:
    return (
        decision.get("status") == "filled"
        and decision.get("action") in ("buy", "sell")
        and decision.get("price") is not None
        and bool(decision.get("symbol"))
        and float(decision.get("filled_quantity") or 0.0) > 0
    )


@dataclass
class SymbolBook:
    """One symbol's fills on average cost: the shares the ledger bought and has
    not sold yet, what they cost, and what its sells realized."""

    shares: float = 0.0
    cost: float = 0.0
    realized: float = 0.0
    fees: float = 0.0
    buys: int = 0
    sells: int = 0

    def apply(self, decision: dict) -> None:
        qty = float(decision["filled_quantity"])
        price = float(decision["price"])
        self.fees += float(decision.get("fee") or 0.0)
        if decision["action"] == "buy":
            self.shares += qty
            self.cost += qty * price
            self.buys += 1
            return
        self.sells += 1
        # Only shares the ledger bought have a cost it knows. A sell of more
        # than that -- shares a real account held before the run -- realizes
        # nothing measurable on the excess, so the excess is left out.
        matched = min(qty, self.shares)
        if matched <= 0:
            return
        avg = self.cost / self.shares
        self.realized += matched * (price - avg)
        self.shares -= matched
        self.cost -= matched * avg
        if self.shares < 1e-9:
            self.shares, self.cost = 0.0, 0.0

    @property
    def avg_cost(self) -> "float | None":
        return self.cost / self.shares if self.shares > 0 else None

    def open_pnl(self, price: "float | None") -> "float | None":
        """The open shares' gain at `price`; None while none are open or
        there is no price to mark them at."""
        if not self.shares or price is None:
            return None
        return self.shares * price - self.cost

    def pnl(self, price: "float | None") -> float:
        """Realized + open − fees. Across symbols these add up to the equity
        curve's value − starting cash whenever the ledger opened every
        position itself (always, in local simulation)."""
        return self.realized + (self.open_pnl(price) or 0.0) - self.fees


def symbol_books(decisions: list[dict]) -> dict[str, SymbolBook]:
    """Every symbol's book after all of `decisions`' fills, in time order."""
    books: dict[str, SymbolBook] = {}
    for d in sorted(decisions, key=lambda d: d["ts"]):
        if _is_fill(d):
            books.setdefault(str(d["symbol"]), SymbolBook()).apply(d)
    return books


def holdings(
    decisions: list[dict],
    positions: dict[str, float],
    prices: dict[str, "float | None"],
    portfolio_value: "float | None" = None,
) -> list[dict]:
    """One row per symbol the ledger holds or has traded: held positions first,
    largest market value first, then the symbols traded flat, by name.

    `shares` is the ledger's position -- on a real venue, the account's -- and
    `untracked` the part of it no fill here bought (held before the run, or
    bought outside the app), which has no cost here and so no open P&L.
    `weight_pct` is the market value's share of `portfolio_value`.
    """
    books = symbol_books(decisions)
    symbols = set(books) | {s for s, q in positions.items() if q}
    rows = []
    for sym in symbols:
        book = books.get(sym) or SymbolBook()
        shares = float(positions.get(sym) or 0.0)
        last = prices.get(sym)
        market_value = shares * last if last is not None else None
        open_pnl = book.open_pnl(last)
        avg = book.avg_cost
        rows.append(
            {
                "symbol": sym,
                "shares": shares,
                "avg_cost": avg,
                "last": last,
                "market_value": market_value,
                "weight_pct": (
                    market_value / portfolio_value * 100
                    if market_value is not None and portfolio_value else None
                ),
                "open_pnl": open_pnl,
                "open_pct": (last / avg - 1) * 100 if avg and last is not None else None,
                "closed_pnl": book.realized,
                "fees": book.fees,
                "net_pnl": book.pnl(last),
                "buys": book.buys,
                "sells": book.sells,
                "untracked": max(0.0, shares - book.shares),
            }
        )
    held = sorted(
        (r for r in rows if r["shares"]),
        key=lambda r: -abs(r["market_value"] or 0.0),
    )
    flat = sorted((r for r in rows if not r["shares"]), key=lambda r: r["symbol"])
    return held + flat


def compute_symbol_curves(
    bars_by_symbol: dict[str, list[dict]],
    decisions: list[dict],
    starting_cash: float,
    session_start: datetime,
    live_prices: "dict[str, float | None] | None" = None,
) -> dict:
    """`compute_equity_curve` split by symbol.

    Returns {"points": [...], "fills": [...]}. One point per bar time at/after
    `session_start` and one at every fill, each {"ts", "cash", "pnl": {symbol:
    $}, "market_value": {symbol: $}} over every symbol traded or held so far:
    P&L from that symbol's book (`SymbolBook.pnl`), market value the ledger's
    position marked to the symbol's latest close (or the fill's own price, as
    on the equity curve). Plus a trailing "now" point priced off `live_prices`.
    `fills` are the buys/sells after `session_start`, each with its symbol's
    P&L just after it, to sit on that symbol's line.
    """
    ordered = sorted(decisions, key=lambda d: d["ts"])
    decision_ts = [pd.Timestamp(d["ts"]) for d in ordered]
    cutoff = pd.Timestamp(session_start)

    cash = starting_cash
    positions: dict[str, float] = {}
    prices: dict[str, float] = {}
    books: dict[str, SymbolBook] = {}
    points: list[dict] = []
    fills: list[dict] = []
    di = 0

    def _point(ts: pd.Timestamp) -> dict:
        symbols = sorted(set(books) | {s for s, q in positions.items() if q})
        return {
            "ts": ts.isoformat(),
            "cash": cash,
            "pnl": {s: books[s].pnl(prices.get(s)) if s in books else 0.0 for s in symbols},
            "market_value": {
                s: positions.get(s, 0.0) * prices[s] if s in prices else 0.0 for s in symbols
            },
        }

    def _apply_decisions_up_to(ts: pd.Timestamp) -> None:
        nonlocal cash, positions, di
        while di < len(ordered) and decision_ts[di] <= ts:
            d = ordered[di]
            cash = float(d.get("cash_after") or 0.0)
            positions = _decision_positions(d)
            if d.get("price") is not None and d.get("symbol"):
                prices[str(d["symbol"])] = float(d["price"])
            if _is_fill(d):
                sym = str(d["symbol"])
                books.setdefault(sym, SymbolBook()).apply(d)
                if decision_ts[di] > cutoff:
                    points.append(_point(decision_ts[di]))
                    fills.append(
                        {
                            "ts": d["ts"],
                            "symbol": sym,
                            "action": d["action"],
                            "pnl": points[-1]["pnl"][sym],
                            "price": d.get("price"),
                            "filled_quantity": d.get("filled_quantity"),
                            "reasoning": d.get("reasoning"),
                        }
                    )
            di += 1

    for ts, group in groupby(_bar_events(bars_by_symbol, session_start), key=lambda e: e[0]):
        _apply_decisions_up_to(ts)
        for _, symbol, close in group:
            prices[symbol] = close
        points.append(_point(ts))

    if live_prices:
        now = pd.Timestamp(clock.now())
        _apply_decisions_up_to(now)
        for sym, price in live_prices.items():
            if price is not None:
                prices[sym] = float(price)
        points.append(_point(now))
    return {"points": points, "fills": fills}
