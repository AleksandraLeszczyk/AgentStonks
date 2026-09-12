"""
Order execution abstraction.

`DecisionTracker` talks to a `Broker`, not directly to Alpaca, so the same
decision-tracking logic serves three very different places an order can go:

    PaperBroker   real prices, invented fills, nothing leaves the process
    SimBroker     (simlab) fills against a stored historical tape
    AlpacaBroker  a real order to a real broker, paper or live account

The first two are *simulated*: they always say yes, at the price asked, for the
quantity asked. The third can say no. That difference is the whole reason
`is_simulated` exists -- a simulated broker lets the local ledger stay the
record of truth, while a real one makes the broker's account the truth and
demotes the ledger to a mirror of it. See `DecisionTracker.record_trade`.
"""
from __future__ import annotations

import abc
import logging
import uuid

from .config import ORDER_FILL_TIMEOUT_SEC
from . import trading_rest
from .rest import fetch_latest_trade
from .trading_rest import TradingCredentials, TradingError

logger = logging.getLogger(__name__)


class Broker(abc.ABC):
    @abc.abstractmethod
    def get_current_price(self, symbol: str, key: str, secret: str, feed: str = "iex") -> float:
        """Return a fresh current price for `symbol`, independent of any cached data."""

    @abc.abstractmethod
    def submit_order(self, symbol: str, side: str, quantity: float, price: float) -> dict:
        """Execute an order and return a fill report.

        The report carries `status` ("filled" | "rejected" | "partial" | ...),
        `filled_qty` and `filled_price`. A simulated broker echoes back what it
        was asked for; a real one reports what the venue actually did, which may
        be less, at a different price, or nothing at all.
        """

    @property
    def is_simulated(self) -> bool:
        """True when fills are invented locally and no order leaves the process.

        `DecisionTracker` uses this to decide who owns the cash balance: itself,
        or the venue.
        """
        return True

    @property
    def venue(self) -> str:
        """Human name for where orders go, for logs and the UI badge."""
        return "local simulation"

    def account_snapshot(self) -> "dict | None":
        """Authoritative {"cash": float, "positions": {symbol: qty}} from the
        venue, or None when the local ledger is the only record there is."""
        return None

    def max_quantity(self, symbol: str, side: str, price: float) -> "float | None":
        """Largest quantity the venue would accept, or None for "no opinion"
        (the caller falls back to its own cash arithmetic)."""
        return None


class PaperBroker(Broker):
    """Mock broker: prices are real (fetched from Alpaca), but orders only
    mutate an in-memory ledger via DecisionTracker — no real trades are placed."""

    def get_current_price(self, symbol: str, key: str, secret: str, feed: str = "iex") -> float:
        trade = fetch_latest_trade(symbol, key, secret, feed)
        price = trade.get("p")
        if price is None:
            raise RuntimeError(f"No latest trade price available for {symbol}")
        return float(price)

    def submit_order(self, symbol: str, side: str, quantity: float, price: float) -> dict:
        return {"status": "filled", "filled_qty": quantity, "filled_price": price}


class AlpacaBroker(Broker):
    """Routes orders to a real Alpaca account -- paper or live.

    Unlike the simulated brokers, everything here can refuse. Buying power is
    the account's, not a number this app chose; a symbol may not be tradable or
    may not accept the fractional quantity the sizing math produced; an order
    submitted while the market is closed queues instead of filling; a market
    order fills asynchronously and can come back partial. Each of those is
    reported rather than smoothed over, because a strategy that believes it
    bought 100 shares when it holds 40 is worse off than one told it got 40.

    The account is the source of truth for cash and positions. After every
    order this broker re-reads both, so the app's ledger converges on the
    broker's even when a fill lands differently than the request implied.
    """

    def __init__(
        self,
        creds: TradingCredentials,
        fill_timeout_sec: float = ORDER_FILL_TIMEOUT_SEC,
        sleep: "callable | None" = None,
    ) -> None:
        self.creds = creds
        self.fill_timeout_sec = fill_timeout_sec
        self._sleep = sleep
        # Asset tradability changes about as often as a listing does, so it is
        # fetched once per symbol per session rather than on every order.
        self._asset_cache: dict[str, dict] = {}

    # --- identity -------------------------------------------------------
    @property
    def is_simulated(self) -> bool:
        return False

    @property
    def venue(self) -> str:
        return self.creds.venue

    @property
    def is_live(self) -> bool:
        return self.creds.live

    # --- prices ---------------------------------------------------------
    def get_current_price(self, symbol: str, key: str, secret: str, feed: str = "iex") -> float:
        """Still the market-data API: the trading account does not publish a tape,
        and the app's own data credentials are what is entitled to read one."""
        trade = fetch_latest_trade(symbol, key, secret, feed)
        price = trade.get("p")
        if price is None:
            raise RuntimeError(f"No latest trade price available for {symbol}")
        return float(price)

    # --- account --------------------------------------------------------
    def account_snapshot(self) -> "dict | None":
        try:
            account = trading_rest.get_account(self.creds)
            positions = trading_rest.positions_by_symbol(self.creds)
        except TradingError as exc:
            # A failed read must not be mistaken for "the account is empty":
            # returning None leaves the ledger on its last known values rather
            # than zeroing cash and every position.
            logger.warning("%s: account snapshot failed: %s", self.venue, exc)
            return None
        return {
            "cash": float(account.get("cash") or 0.0),
            "positions": positions,
            "buying_power": float(account.get("buying_power") or 0.0),
            "equity": float(account.get("equity") or 0.0),
            "blocked": bool(
                account.get("trading_blocked")
                or account.get("account_blocked")
                or account.get("transfers_blocked")
            ),
            "status": account.get("status"),
        }

    def asset(self, symbol: str) -> dict:
        symbol = symbol.upper()
        if symbol not in self._asset_cache:
            self._asset_cache[symbol] = trading_rest.get_asset(symbol, self.creds)
        return self._asset_cache[symbol]

    def normalize_quantity(self, symbol: str, quantity: float) -> float:
        """Round a requested quantity to something the venue will accept.

        The app sizes positions as a fraction of cash, so it asks for things like
        12.8371 shares. Alpaca takes that on a fractionable symbol and rejects it
        on any other, where the order has to be whole shares -- rounded *down*,
        since rounding up would spend money the sizing did not allow for.
        """
        if quantity <= 0:
            return 0.0
        try:
            fractionable = bool(self.asset(symbol).get("fractionable"))
        except TradingError as exc:
            # Unknown: treat as whole-share-only, the choice that cannot produce
            # a rejection.
            logger.warning("%s: asset lookup failed for %s: %s", self.venue, symbol, exc)
            fractionable = False
        if not fractionable:
            return float(int(quantity))
        return round(quantity, 9)

    def max_quantity(self, symbol: str, side: str, price: float) -> "float | None":
        """What the account can actually do, which is not what the local ledger
        thinks: buying power reflects settled cash and margin, and a sell is
        capped by the position the broker says is held."""
        snap = self.account_snapshot()
        if snap is None:
            return None
        if side == "buy":
            return snap["buying_power"] / price if price > 0 else 0.0
        return max(0.0, snap["positions"].get(symbol.upper(), 0.0))

    # --- orders ---------------------------------------------------------
    def submit_order(self, symbol: str, side: str, quantity: float, price: float) -> dict:
        """Place a market order and report what the venue actually did.

        Never raises for an ordinary refusal -- a rejected order is a result,
        not an error, and the tracker records it as a rejected decision with the
        broker's own reason attached.
        """
        symbol = symbol.upper()
        qty = self.normalize_quantity(symbol, quantity)
        if qty <= 0:
            return {
                "status": "rejected",
                "filled_qty": 0.0,
                "filled_price": price,
                "reason": (
                    f"{symbol} is not fractionable and the requested {quantity:.4f} "
                    "shares rounds down to zero whole shares"
                ),
            }
        try:
            order = trading_rest.submit_order(
                symbol, side, qty, self.creds,
                client_order_id=f"agentstonks-{uuid.uuid4().hex[:20]}",
            )
        except (TradingError, ValueError) as exc:
            return {
                "status": "rejected", "filled_qty": 0.0, "filled_price": price,
                "reason": str(exc),
            }

        order_id = str(order.get("id") or "")
        try:
            final = trading_rest.wait_for_fill(
                order_id, self.creds, self.fill_timeout_sec, sleep=self._sleep
            )
        except TradingError as exc:
            # The order is out there; we just could not read it back. Report
            # nothing filled rather than inventing a fill -- the next account
            # snapshot will reconcile whatever it turns into.
            logger.warning("%s: could not poll order %s: %s", self.venue, order_id, exc)
            return {
                "status": "unknown", "filled_qty": 0.0, "filled_price": price,
                "order_id": order_id, "reason": str(exc),
            }

        filled_qty = float(final.get("filled_qty") or 0.0)
        filled_price = final.get("filled_avg_price")
        status = str(final.get("status") or "")
        report = {
            "status": "filled" if filled_qty > 0 else "rejected",
            "filled_qty": filled_qty,
            # Alpaca's average fill, not the pre-trade quote -- slippage between
            # the two is exactly what routing a real order is meant to reveal.
            "filled_price": float(filled_price) if filled_price else price,
            "order_id": order_id,
            "broker_status": status,
        }
        if filled_qty <= 0:
            report["reason"] = _no_fill_reason(final, status, self.fill_timeout_sec)
        elif filled_qty < qty:
            report["status"] = "filled"
            report["reason"] = (
                f"partial fill: {filled_qty:g} of {qty:g} shares "
                f"(order {status}); the rest is still with the broker"
            )
        return report


def _no_fill_reason(order: dict, status: str, timeout_sec: float) -> str:
    """Why an order produced nothing, in the venue's words where it has any."""
    if order.get("reject_reason"):
        return str(order["reject_reason"])
    if status in ("new", "accepted", "pending_new", "partially_filled"):
        return (
            f"still working after {timeout_sec:.0f}s (order {status}) — it was NOT "
            "cancelled and may yet fill; the next account sync will pick it up"
        )
    return f"order finished as {status or 'unknown'} with nothing filled"
