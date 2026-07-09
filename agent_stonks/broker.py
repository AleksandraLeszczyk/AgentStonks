"""
Order execution abstraction.

`DecisionTracker` talks to a `Broker`, not directly to Alpaca, so the agent can
stay on paper trading today (`PaperBroker`) while leaving room to plug in a
live order-routing broker later without touching the decision-tracking logic.
"""
from __future__ import annotations

import abc

from .rest import fetch_latest_trade


class Broker(abc.ABC):
    @abc.abstractmethod
    def get_current_price(self, symbol: str, key: str, secret: str, feed: str = "iex") -> float:
        """Return a fresh current price for `symbol`, independent of any cached data."""

    @abc.abstractmethod
    def submit_order(self, symbol: str, side: str, quantity: float, price: float) -> dict:
        """Execute an order and return a fill report."""


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
