"""
Independent decision-tracking ledger for the trading agent.

This module is deliberately separate from the agent's reasoning: when a buy
or sell is decided, the fill price is fetched here, fresh, via `Broker`,
rather than trusting whatever price the agent happened to have looked at
during analysis. The agent only ever influences *what* to do (action,
quantity, reasoning) — never the price a trade is recorded at.
"""
from __future__ import annotations

import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Optional

from .broker import Broker, PaperBroker
from .config import TRADE_FIXED_COST


@dataclass
class Decision:
    ts: str
    symbol: str
    action: str  # "buy" | "sell" | "sleep"
    requested_quantity: float
    filled_quantity: float
    price: Optional[float]
    reasoning: str
    status: str  # "filled" | "rejected" | "noop"
    cash_after: float
    position_after: float
    fee: float = 0.0


class DecisionTracker:
    """Tracks a mock paper position/cash balance and every decision made against it."""

    def __init__(
        self,
        starting_cash: float = 100_000.0,
        broker: Optional[Broker] = None,
        trade_cost: float = TRADE_FIXED_COST,
    ) -> None:
        self.broker = broker or PaperBroker()
        self.lock = threading.Lock()
        self.cash = starting_cash
        self.position = 0.0
        self.trade_cost = trade_cost
        self.decisions: list[Decision] = []

    def record_sleep(self, symbol: str, reasoning: str) -> Decision:
        decision = Decision(
            ts=datetime.now(timezone.utc).isoformat(),
            symbol=symbol,
            action="sleep",
            requested_quantity=0,
            filled_quantity=0,
            price=None,
            reasoning=reasoning,
            status="noop",
            cash_after=self.cash,
            position_after=self.position,
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
        """Record a buy/sell decision. Fetches the fill price independently via `broker`."""
        if action not in ("buy", "sell"):
            raise ValueError(f"action must be 'buy' or 'sell', got {action!r}")

        price = self.broker.get_current_price(symbol, key, secret, feed)

        with self.lock:
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
                    self.position += filled_qty
                    status = "filled"
            else:  # sell
                filled_qty = max(0.0, min(quantity, self.position))
                if filled_qty > 0:
                    self.broker.submit_order(symbol, "sell", filled_qty, price)
                    fee = self.trade_cost
                    self.cash += filled_qty * price - fee
                    self.position -= filled_qty
                    status = "filled"

            decision = Decision(
                ts=datetime.now(timezone.utc).isoformat(),
                symbol=symbol,
                action=action,
                requested_quantity=quantity,
                filled_quantity=filled_qty,
                price=price,
                reasoning=reasoning,
                status=status,
                cash_after=self.cash,
                position_after=self.position,
                fee=fee,
            )
            self.decisions.append(decision)
        return decision

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "cash": self.cash,
                "position": self.position,
                "decisions": list(self.decisions),
            }

    def trade_markers(self) -> list[dict]:
        """Filled buy/sell decisions only, shaped for plotting on the price chart."""
        with self.lock:
            return [
                asdict(d)
                for d in self.decisions
                if d.status == "filled" and d.price is not None
            ]
