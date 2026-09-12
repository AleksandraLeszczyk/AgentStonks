"""Alpaca Trading API client: accounts, positions and real orders.

Distinct from `agent_stonks.rest`, which reads *market data*. This module is the
only place in the app that can move money, and it talks to a different host:

    paper   https://paper-api.alpaca.markets     fake money, real plumbing
    live    https://api.alpaca.markets           real money

Those are separate accounts with separate credentials -- a paper key is not
valid against the live host and vice versa -- so the endpoint and the key always
travel together, in a `TradingCredentials`, rather than being passed as loose
strings that could drift apart. Nothing here reads an environment variable or
picks a venue on its own; the caller says which account it means, every time.

Market data keys are a third thing again: `rest.py` keeps using those, and a
data subscription has no bearing on what this module is allowed to trade.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import requests

from .config import ORDER_POLL_SEC, TRADING_REST_LIVE, TRADING_REST_PAPER

logger = logging.getLogger(__name__)

# Order states Alpaca will not move away from on its own. Anything else is still
# working and worth polling; `filled` is the only one of these that means the
# trade happened.
TERMINAL_ORDER_STATUSES = frozenset(
    {"filled", "canceled", "expired", "rejected", "done_for_day", "suspended", "stopped"}
)


class TradingError(RuntimeError):
    """An order or account call the caller has to handle -- rejected order,
    insufficient buying power, untradable symbol, bad credentials."""


@dataclass(frozen=True)
class TradingCredentials:
    """One Alpaca trading account: which host, and the key pair for it.

    `live` is carried explicitly rather than inferred from the URL so that every
    log line, UI badge and confirmation prompt downstream can say which kind of
    money is at stake without re-parsing a hostname.
    """

    key: str
    secret: str
    live: bool = False

    @property
    def base_url(self) -> str:
        return TRADING_REST_LIVE if self.live else TRADING_REST_PAPER

    @property
    def venue(self) -> str:
        return "Alpaca LIVE" if self.live else "Alpaca paper"

    @property
    def configured(self) -> bool:
        return bool(self.key and self.secret)


def _headers(creds: TradingCredentials) -> dict[str, str]:
    return {
        "APCA-API-KEY-ID": creds.key,
        "APCA-API-SECRET-KEY": creds.secret,
        "accept": "application/json",
    }


def _request(
    method: str, path: str, creds: TradingCredentials, *, timeout: float = 10.0, **kwargs
) -> object:
    """One Trading API call, with Alpaca's error body surfaced.

    `raise_for_status` reports only the status line, which for this API throws
    away the part that matters: a rejected order says *why* in the body
    ("insufficient buying power", "asset not tradable", "account is restricted").
    Losing that turns every failure into an indistinguishable 403.
    """
    if not creds.configured:
        raise TradingError("No Alpaca trading credentials configured")
    url = f"{creds.base_url}{path}"
    response = requests.request(
        method, url, headers=_headers(creds), timeout=timeout, **kwargs
    )
    if response.status_code >= 400:
        detail = ""
        try:
            body = response.json()
            detail = body.get("message") or str(body)
        except Exception:
            detail = (response.text or "").strip()[:300]
        raise TradingError(
            f"{creds.venue} {method} {path} failed ({response.status_code}): {detail}"
        )
    if response.status_code == 204 or not response.content:
        return None
    return response.json()


def get_account(creds: TradingCredentials) -> dict:
    """The trading account: cash, buying power, equity, and its restrictions."""
    return _request("GET", "/v2/account", creds)


def get_positions(creds: TradingCredentials) -> list[dict]:
    """Every open position on the account."""
    return _request("GET", "/v2/positions", creds) or []


def get_clock(creds: TradingCredentials) -> dict:
    """Alpaca's own market clock -- the authority on whether an order submitted
    now trades now or queues for the next open."""
    return _request("GET", "/v2/clock", creds)


def get_asset(symbol: str, creds: TradingCredentials) -> dict:
    """Tradability of one symbol, including whether it accepts fractional
    quantities -- the app sizes positions as a fraction of cash, so it routinely
    wants 12.837 shares, which most-but-not-all US equities will accept."""
    return _request("GET", f"/v2/assets/{symbol.upper()}", creds)


def submit_order(
    symbol: str,
    side: str,
    quantity: float,
    creds: TradingCredentials,
    *,
    order_type: str = "market",
    time_in_force: str = "day",
    limit_price: "float | None" = None,
    client_order_id: "str | None" = None,
) -> dict:
    """Place one order. Returns Alpaca's order object (not necessarily filled).

    A market order is *accepted* here, not completed: fills arrive asynchronously
    and `filled_avg_price` is empty on the way back. Use `wait_for_fill` to find
    out what actually happened.
    """
    side = side.lower()
    if side not in ("buy", "sell"):
        raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
    if quantity <= 0:
        raise ValueError(f"quantity must be positive, got {quantity!r}")

    payload: dict[str, object] = {
        "symbol": symbol.upper(),
        # Alpaca wants the quantity as a string; a float repr like 1e-05 is
        # rejected, and trailing zeros on a fractional qty are not.
        "qty": f"{quantity:.9f}".rstrip("0").rstrip("."),
        "side": side,
        "type": order_type,
        "time_in_force": time_in_force,
    }
    if limit_price is not None:
        payload["limit_price"] = f"{limit_price:.2f}"
    if client_order_id:
        payload["client_order_id"] = client_order_id
    logger.info("%s: submitting %s %s %s", creds.venue, side, payload["qty"], symbol)
    return _request("POST", "/v2/orders", creds, json=payload)


def get_order(order_id: str, creds: TradingCredentials) -> dict:
    return _request("GET", f"/v2/orders/{order_id}", creds)


def cancel_order(order_id: str, creds: TradingCredentials) -> None:
    _request("DELETE", f"/v2/orders/{order_id}", creds)


def wait_for_fill(
    order_id: str,
    creds: TradingCredentials,
    timeout_sec: float,
    sleep: "callable | None" = None,
) -> dict:
    """Poll one order until it reaches a terminal state or `timeout_sec` elapses.

    Returns the last order object seen, filled or not -- deciding what a partial
    fill or a still-working order means is the caller's business, and this
    function never cancels on its own.

    `sleep` is injectable so tests can drive the loop without real delays.
    """
    import time as _time

    naptime = sleep or _time.sleep
    deadline = _time.monotonic() + timeout_sec
    order = get_order(order_id, creds)
    while order.get("status") not in TERMINAL_ORDER_STATUSES:
        if _time.monotonic() >= deadline:
            break
        naptime(ORDER_POLL_SEC)
        order = get_order(order_id, creds)
    return order


def positions_by_symbol(creds: TradingCredentials) -> dict[str, float]:
    """Open positions as {symbol: signed quantity}, the shape the app's ledger
    keeps. A short position comes back negative, as Alpaca reports it."""
    return {
        str(p["symbol"]).upper(): float(p.get("qty") or 0.0)
        for p in get_positions(creds)
    }
