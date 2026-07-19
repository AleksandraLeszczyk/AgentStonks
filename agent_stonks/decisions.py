"""
Independent decision-tracking ledger for the trading agent.

This module is deliberately separate from the agent's reasoning: when a buy
or sell is decided, the fill price is fetched here, fresh, via `Broker`,
rather than trusting whatever price the agent happened to have looked at
during analysis. The agent only ever influences *what* to do (symbol, action,
quantity, reasoning) — never the price a trade is recorded at.

One tracker serves the whole symbol basket: cash is a single shared balance,
positions are held per symbol.
"""
from __future__ import annotations

import threading
from dataclasses import asdict, dataclass, field
from typing import Optional

from . import clock
from .broker import Broker, PaperBroker
from .config import TRADE_FIXED_COST


@dataclass
class Decision:
    ts: str
    symbol: str
    action: str  # "buy" | "sell" | "alert" | "tactics"; "sleep" only as an internal no-op fallback
    requested_quantity: float
    filled_quantity: float
    price: Optional[float]
    reasoning: str
    status: str  # "filled" | "rejected" | "noop" | "armed"
    cash_after: float
    position_after: float  # position in THIS decision's symbol after the trade
    fee: float = 0.0
    alerts: Optional[list[dict]] = None
    # Human-readable one-liner per armed conditional action, for action="tactics".
    tactics: Optional[list[str]] = None
    # Snapshot of every symbol's position after the trade, so the multi-symbol
    # equity curve can be replayed from decisions alone.
    positions_after: dict[str, float] = field(default_factory=dict)


class DecisionTracker:
    """Tracks a mock paper cash balance, per-symbol positions, and every decision
    made against them."""

    def __init__(
        self,
        starting_cash: float = 100_000.0,
        broker: Optional[Broker] = None,
        trade_cost: float = TRADE_FIXED_COST,
    ) -> None:
        self.broker = broker or PaperBroker()
        self.lock = threading.Lock()
        self.cash = starting_cash
        self.positions: dict[str, float] = {}
        self.trade_cost = trade_cost
        self.decisions: list[Decision] = []

    def position_for(self, symbol: str) -> float:
        with self.lock:
            return self.positions.get(symbol, 0.0)

    def _noop_decision(self, symbol: str, action: str, reasoning: str, **extra) -> Decision:
        return Decision(
            ts=clock.now().isoformat(),
            symbol=symbol,
            action=action,
            requested_quantity=0,
            filled_quantity=0,
            price=extra.pop("price", None),
            reasoning=reasoning,
            status=extra.pop("status", "noop"),
            cash_after=self.cash,
            position_after=self.positions.get(symbol, 0.0),
            positions_after=dict(self.positions),
            **extra,
        )

    def record_sleep(self, symbol: str, reasoning: str) -> Decision:
        """Record an internal no-op cycle. The agent can no longer *choose* to sleep --
        when it doesn't want to trade it must set an alert -- so this now only backs the
        forced fallback when a cycle ends without any finalized decision."""
        decision = self._noop_decision(symbol, "sleep", reasoning)
        with self.lock:
            self.decisions.append(decision)
        return decision

    def record_alert(self, symbol: str, alerts: list[dict], reasoning: str) -> Decision:
        """Record a no-op cycle where the agent set one or more condition alerts instead of trading.

        Each entry is shaped {"symbol": str, "field": str, "condition": "above" | "below",
        "value": float}, watching a continuously-updated per-symbol field (price, bid/ask,
        spread, day volume, volume ratio, portfolio value, ...) to wake the agent early
        when it crosses the value.
        """
        decision = self._noop_decision(symbol, "alert", reasoning, alerts=alerts)
        with self.lock:
            self.decisions.append(decision)
        return decision

    def record_tactics(
        self, symbol: str, summaries: list[str], reasoning: str, price: Optional[float] = None
    ) -> Decision:
        """Record the arming of a conditional trade plan (see agent_stonks.tactics).

        No cash or position changes here -- the trade happens later, via
        `record_trade`, when the TacticsExecutor sees the conditions met. `price`
        is the last seen price at arming time, kept so the moment can be marked
        on the portfolio-value chart.
        """
        decision = self._noop_decision(
            symbol, "tactics", reasoning, price=price, status="armed", tactics=summaries
        )
        with self.lock:
            self.decisions.append(decision)
        return decision

    def record_trade(
        self,
        symbol: str,
        action: str,
        quantity: float,
        reasoning: str,
        key: str,
        secret: str,
        feed: str = "iex",
    ) -> Decision:
        """Record a buy/sell decision for one symbol. Fetches the fill price
        independently via `broker`. Cash is shared across symbols; the position
        change applies to `symbol` only."""
        if action not in ("buy", "sell"):
            raise ValueError(f"action must be 'buy' or 'sell', got {action!r}")

        price = self.broker.get_current_price(symbol, key, secret, feed)

        with self.lock:
            position = self.positions.get(symbol, 0.0)
            filled_qty = 0.0
            status = "rejected"
            fee = 0.0
            if action == "buy":
                affordable_cash = max(0.0, self.cash - self.trade_cost)
                affordable = affordable_cash / price if price > 0 else 0.0
                filled_qty = max(0.0, min(quantity, affordable))
                if filled_qty > 0:
                    self.broker.submit_order(symbol, "buy", filled_qty, price)
                    fee = self.trade_cost
                    self.cash -= filled_qty * price + fee
                    position += filled_qty
                    status = "filled"
            else:  # sell
                filled_qty = max(0.0, min(quantity, position))
                if filled_qty > 0:
                    self.broker.submit_order(symbol, "sell", filled_qty, price)
                    fee = self.trade_cost
                    self.cash += filled_qty * price - fee
                    position -= filled_qty
                    status = "filled"
            self.positions[symbol] = position

            decision = Decision(
                ts=clock.now().isoformat(),
                symbol=symbol,
                action=action,
                requested_quantity=quantity,
                filled_quantity=filled_qty,
                price=price,
                reasoning=reasoning,
                status=status,
                cash_after=self.cash,
                position_after=position,
                fee=fee,
                positions_after=dict(self.positions),
            )
            self.decisions.append(decision)
        return decision

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "cash": self.cash,
                "positions": dict(self.positions),
                "decisions": list(self.decisions),
            }

    def trade_markers(self, symbol: "str | None" = None) -> list[dict]:
        """Filled buy/sell decisions only, shaped for plotting on the price chart.
        Pass `symbol` to restrict markers to one ticker's chart."""
        with self.lock:
            return [
                asdict(d)
                for d in self.decisions
                if d.status == "filled"
                and d.price is not None
                and (symbol is None or d.symbol == symbol)
            ]
