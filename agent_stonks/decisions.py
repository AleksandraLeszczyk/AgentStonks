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

import logging
import math
import threading
from dataclasses import asdict, dataclass, field
from typing import Optional

from . import clock
from .broker import Broker, PaperBroker
from .config import TRADE_FIXED_COST, VENUE_VALUE_REFRESH_SEC

logger = logging.getLogger(__name__)


def _beyond_limit(action: str, price: float, limit_price: float) -> bool:
    """Whether `price` is on the side of the limit an order must not fill on."""
    return price > limit_price if action == "buy" else price < limit_price


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
    # The price a limited order could not pay more than (a buy) or take less
    # than (a sell), or None for a market order. See `record_trade`.
    limit_price: Optional[float] = None
    # True when the order went unfilled because the price was on the wrong
    # side of `limit_price` -- a miss, not a refusal: the same order may fill
    # on the next try, which a rejection for cash or tradability will not.
    limit_missed: bool = False


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
        # Last value the venue itself reported for the account, and when (a
        # monotonic reading). Only a real broker ever fills these; see
        # `venue_value`.
        self._venue_value: "float | None" = None
        self._venue_value_at: float = 0.0
        self._venue_value_lock = threading.Lock()
        self._venue_value_refreshing = False

    # --- the venue's own account value ------------------------------------
    def venue_value(self) -> "float | None":
        """What the venue says this account is worth right now, or None when
        there is no venue (a simulated broker) or it has never answered.

        Never blocks. Portfolio value is marked to market from the websocket
        thread on every streamed trade, and an HTTP call there would stall the
        tape; so this returns the last value the venue gave and kicks off a
        background refresh once it is older than VENUE_VALUE_REFRESH_SEC.

        A stale reading is returned in preference to None because the
        alternative is worse: callers fall back to summing the local ledger,
        and that number is a different quantity, not an older one -- it misses
        positions in symbols the app does not stream and any cash movement made
        outside the app. Swapping between the two would make the displayed
        portfolio value jump every time a request failed.
        """
        if self.broker.is_simulated:
            return None
        with self._venue_value_lock:
            # Staleness is about the clock, never about whether the last read
            # succeeded: a venue that keeps failing would otherwise be retried
            # on every single streamed trade, which is thousands of requests a
            # session against an account that is already not answering.
            never_read = self._venue_value_at == 0.0
            age = clock.monotonic() - self._venue_value_at
            stale = never_read or age >= VENUE_VALUE_REFRESH_SEC
            if stale and not self._venue_value_refreshing:
                self._venue_value_refreshing = True
                threading.Thread(
                    target=self._refresh_venue_value_bg, daemon=True,
                    name="venue-value-refresh",
                ).start()
            return self._venue_value

    def refresh_venue_value(self) -> "float | None":
        """Read the venue's account value now, blocking. For callers already
        off the hot path (session start, the agent's own cycle)."""
        if self.broker.is_simulated:
            return None
        value = self.broker.account_value()
        self._store_venue_value(value)
        return value

    def _refresh_venue_value_bg(self) -> None:
        try:
            self._store_venue_value(self.broker.account_value())
        except Exception:  # never let a background read die unremarked
            logger.exception("venue account value refresh failed")
            self._store_venue_value(None)
        finally:
            # Only the thread that claimed the slot releases it. A blocking
            # `refresh_venue_value` running alongside must not clear a flag it
            # never set, or two background reads end up in flight at once.
            with self._venue_value_lock:
                self._venue_value_refreshing = False

    def _store_venue_value(self, value: "float | None") -> None:
        """Record a reading. A failed read (None) still resets the clock, so a
        broker that is down is retried on the same interval rather than on every
        single streamed trade."""
        with self._venue_value_lock:
            if value is not None:
                self._venue_value = float(value)
            self._venue_value_at = clock.monotonic()

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
        limit_price: Optional[float] = None,
    ) -> Decision:
        """Record a buy/sell decision for one symbol. Fetches the fill price
        independently via `broker`. Cash is shared across symbols; the position
        change applies to `symbol` only.

        Only whole shares are ever traded: the request, and every clamp applied
        to it (affordable cash, position held, venue ceiling), is rounded down
        with `whole_shares`. A request below one share is rejected.

        `limit_price` makes it a limit order: a buy never pays more, a sell
        never takes less. A simulated broker fills at the quoted price or not
        at all, so the limit is applied here, to that price -- an order the
        quote is on the wrong side of fills nothing, recorded as a rejected
        decision with `limit_missed` set. It is not left resting: the ledger
        has no working orders, and the caller asks again if it still wants to.
        A real venue is sent the limit itself (`_record_live_trade`)."""
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
            return self._record_live_trade(
                symbol, action, quantity, reasoning, price, limit_price
            )

        if limit_price is not None and _beyond_limit(action, price, limit_price):
            side = "above" if action == "buy" else "below"
            with self.lock:
                decision = self._noop_decision(
                    symbol, action,
                    f"{reasoning} [not placed: ${price:,.2f} is {side} the "
                    f"${limit_price:,.2f} limit]",
                    status="rejected", price=price,
                    limit_price=limit_price, limit_missed=True,
                )
                decision.requested_quantity = quantity
                self.decisions.append(decision)
            return decision

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
                limit_price=limit_price,
            )
            self.decisions.append(decision)
        return decision

    def _record_live_trade(
        self,
        symbol: str,
        action: str,
        quantity: float,
        reasoning: str,
        price: float,
        limit_price: Optional[float] = None,
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
                symbol, action, requested, 0.0, price, f"{reasoning} [{reason}]", "rejected",
                limit_price=limit_price,
            )

        # Only when there is one, so a broker that predates limits (and every
        # test double) is called exactly as before.
        extra = {} if limit_price is None else {"limit_price": limit_price}
        report = self.broker.submit_order(symbol, action, quantity, price, **extra)
        filled_qty = float(report.get("filled_qty") or 0.0)
        fill_price = float(report.get("filled_price") or price)
        status = "filled" if filled_qty > 0 else "rejected"
        note = report.get("reason")
        if note:
            reasoning = f"{reasoning} [{self.broker.venue}: {note}]"

        return self._record_broker_decision(
            symbol, action, requested, filled_qty, fill_price, reasoning, status,
            limit_price=report.get("limit_price", limit_price),
            limit_missed=bool(report.get("limit_missed")),
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
        limit_price: Optional[float] = None,
        limit_missed: bool = False,
    ) -> Decision:
        """Append a decision whose cash and positions come from the venue.

        Falls back to applying the fill to the local ledger only when the
        account read fails -- an unreachable broker should leave the app with a
        stale-but-plausible ledger rather than a zeroed one.
        """
        snapshot = self.broker.account_snapshot()
        if snapshot is not None:
            # The snapshot already carries the account's equity -- adopt it, so
            # the portfolio value moves the instant a fill lands instead of
            # waiting out the refresh interval.
            self._store_venue_value(snapshot.get("equity"))
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
                limit_price=limit_price,
                limit_missed=limit_missed,
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
        self._store_venue_value(snapshot.get("equity"))
        with self.lock:
            self.cash = snapshot["cash"]
            self.positions = dict(snapshot["positions"])
        return True

    def carry_over(self, prior: "DecisionTracker") -> None:
        """Continue `prior`'s ledger on this tracker: its cash, positions and
        every decision so far. ▶ Start does this within a trading day (see
        agent_stonks.session_store) instead of opening a new ledger. `buys_halted`
        is not carried -- a new run is what lifts it. On a real venue call
        `sync_from_broker` afterwards: the account, not the copy, is the truth."""
        with prior.lock:
            cash = prior.cash
            positions = dict(prior.positions)
            decisions = list(prior.decisions)
        with self.lock:
            self.cash = cash
            self.positions = positions
            self.decisions = decisions

    def snapshot(self) -> dict:
        # Read outside the ledger lock: venue_value takes its own, and it must
        # not be possible to order the two differently anywhere.
        venue_value = self.venue_value()
        with self.lock:
            return {
                "cash": self.cash,
                "positions": dict(self.positions),
                "decisions": list(self.decisions),
                # The venue's own account value, or None in local simulation
                # where the ledger above is the whole truth.
                "venue_value": venue_value,
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
