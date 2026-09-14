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

import math
import threading
from dataclasses import asdict, dataclass, field
from typing import Optional

from . import clock
from .broker import Broker, PaperBroker
from .config import TRADE_FIXED_COST


def whole_shares(quantity: float) -> float:
    """`quantity` rounded down to a whole number of shares (0.0 when not positive).

    Every agent trades whole shares only. Down, never to nearest: rounding up
    would buy with cash the sizing did not allow for, or sell shares that are
    not held. The epsilon keeps float noise such as 30% of 10 = 2.9999999999999996
    from losing a share.
    """
    if not quantity or not quantity > 0:
        return 0.0
    return float(math.floor(quantity + 1e-9))


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
        # Set once the book has been sold off on request (`halt_buys`). Stopping
        # an agent only asks its loop to stop, so a cycle already past that
        # check must not be able to buy straight back in. A new run builds a new
        # tracker, which is what lifts it.
        self.buys_halted: "str | None" = None

    def position_for(self, symbol: str) -> float:
        with self.lock:
            return self.positions.get(symbol, 0.0)

    def halt_buys(self, reason: str) -> None:
        """Refuse every buy from here on, whoever asks; sells still go through."""
        with self.lock:
            self.buys_halted = reason

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
        change applies to `symbol` only.

        Only whole shares are ever traded: the request, and every clamp applied
        to it (affordable cash, position held, venue ceiling), is rounded down
        with `whole_shares`. A request below one share is rejected."""
        if action not in ("buy", "sell"):
            raise ValueError(f"action must be 'buy' or 'sell', got {action!r}")

        if action == "buy" and self.buys_halted:
            with self.lock:
                decision = self._noop_decision(
                    symbol, action, f"{reasoning} [not placed: {self.buys_halted}]",
                    status="rejected",
                )
                decision.requested_quantity = quantity
                self.decisions.append(decision)
            return decision

        price = self.broker.get_current_price(symbol, key, secret, feed)

        if not self.broker.is_simulated:
            return self._record_live_trade(symbol, action, quantity, reasoning, price)

        with self.lock:
            position = self.positions.get(symbol, 0.0)
            filled_qty = 0.0
            status = "rejected"
            fee = 0.0
            if action == "buy":
                affordable_cash = max(0.0, self.cash - self.trade_cost)
                affordable = affordable_cash / price if price > 0 else 0.0
                filled_qty = whole_shares(min(quantity, affordable))
                if filled_qty > 0:
                    self.broker.submit_order(symbol, "buy", filled_qty, price)
                    fee = self.trade_cost
                    self.cash -= filled_qty * price + fee
                    position += filled_qty
                    status = "filled"
            else:  # sell
                filled_qty = whole_shares(min(quantity, position))
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

    def _record_live_trade(
        self, symbol: str, action: str, quantity: float, reasoning: str, price: float
    ) -> Decision:
        """Route one order to a real venue and record what it did.

        The shape is deliberately the inverse of the simulated path. There, the
        ledger decides what is affordable, fills it at the quoted price, and the
        broker is told afterwards. Here the venue decides everything: how much
        buying power there really is, whether the symbol takes a fractional
        quantity, what price the order actually got, and whether it filled at
        all. The ledger is written *from* the answer.

        So cash and positions come from an account read after the order rather
        than from arithmetic on the request. That is the only way the two stay
        in step across the things a real broker does and a simulation never
        does -- partial fills, slippage, queued orders, and a position that
        moved because something outside this app touched the same account.
        """
        requested = quantity
        quantity = whole_shares(quantity)
        # Ask the venue what it would allow before asking for it, so an
        # oversized request is trimmed rather than rejected outright.
        ceiling = self.broker.max_quantity(symbol, action, price)
        if ceiling is not None and quantity > 0:
            quantity = whole_shares(min(quantity, ceiling))

        if quantity <= 0:
            if whole_shares(requested) <= 0:
                reason = "less than one whole share requested"
            elif action == "buy":
                reason = "no buying power at the broker for one whole share"
            else:
                reason = "no whole share held at the broker to sell"
            return self._record_broker_decision(
                symbol, action, requested, 0.0, price, f"{reasoning} [{reason}]", "rejected"
            )

        report = self.broker.submit_order(symbol, action, quantity, price)
        filled_qty = float(report.get("filled_qty") or 0.0)
        fill_price = float(report.get("filled_price") or price)
        status = "filled" if filled_qty > 0 else "rejected"
        note = report.get("reason")
        if note:
            reasoning = f"{reasoning} [{self.broker.venue}: {note}]"

        return self._record_broker_decision(
            symbol, action, requested, filled_qty, fill_price, reasoning, status
        )

    def _record_broker_decision(
        self,
        symbol: str,
        action: str,
        requested: float,
        filled_qty: float,
        price: float,
        reasoning: str,
        status: str,
    ) -> Decision:
        """Append a decision whose cash and positions come from the venue.

        Falls back to applying the fill to the local ledger only when the
        account read fails -- an unreachable broker should leave the app with a
        stale-but-plausible ledger rather than a zeroed one.
        """
        snapshot = self.broker.account_snapshot()
        with self.lock:
            if snapshot is not None:
                self.cash = snapshot["cash"]
                self.positions = dict(snapshot["positions"])
            elif filled_qty > 0:
                signed = filled_qty if action == "buy" else -filled_qty
                self.cash -= signed * price
                self.positions[symbol] = self.positions.get(symbol, 0.0) + signed
            decision = Decision(
                ts=clock.now().isoformat(),
                symbol=symbol,
                action=action,
                requested_quantity=requested,
                filled_quantity=filled_qty,
                price=price,
                reasoning=reasoning,
                status=status,
                cash_after=self.cash,
                position_after=self.positions.get(symbol, 0.0),
                # Alpaca charges no commission on US equities, and the real
                # regulatory fees are already inside the cash the account
                # reports -- modelling TRADE_FIXED_COST on top would double-count
                # a cost the broker has itself applied.
                fee=0.0,
                positions_after=dict(self.positions),
            )
            self.decisions.append(decision)
        return decision

    def sync_from_broker(self) -> bool:
        """Adopt the venue's cash and positions as the ledger's.

        Called when a real broker is attached, so the app opens on the account's
        actual balance instead of a configured starting budget, and again on
        demand -- positions move for reasons this process never sees (a fill
        from an order left working, a manual trade in Alpaca's own UI, a
        corporate action). Returns False when the account could not be read.
        """
        snapshot = self.broker.account_snapshot()
        if snapshot is None:
            return False
        with self.lock:
            self.cash = snapshot["cash"]
            self.positions = dict(snapshot["positions"])
        return True

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
