"""Choosing where orders go, and refusing to route them anywhere by accident.

One function matters here: `resolve_broker` turns a requested trading mode into
an actual `Broker`, or refuses and says why. Everything else is the refusing.

The asymmetry is deliberate. Falling back from a real venue to the local ledger
is always safe -- the worst case is a simulated trade the user thought was real,
which is visible the moment they look at their Alpaca account. Falling the other
way is not recoverable: a live order cannot be un-sent. So every failure here
degrades toward simulation, and live trading is reachable only by passing two
independent gates that both have to be set deliberately:

    1. the LIVE_TRADING_ENV_FLAG environment variable, set outside the app
    2. a phrase typed into the UI for this session

Neither is remembered for the user. An agent loop that starts itself on a timer
must not be able to reach the live account because someone ticked a box once.
"""
from __future__ import annotations

import logging
import os

from .broker import AlpacaBroker, Broker, PaperBroker
from .config import (
    DEFAULT_TRADING_MODE,
    LIVE_TRADING_CONFIRM_PHRASE,
    LIVE_TRADING_ENV_FLAG,
    TRADING_MODES,
)
from .trading_rest import TradingCredentials, TradingError, get_account

logger = logging.getLogger(__name__)

MODE_LABELS: dict[str, str] = {
    "local": "Local simulation (no orders sent)",
    "alpaca_paper": "Alpaca paper account (real orders, fake money)",
    "alpaca_live": "Alpaca LIVE account (real orders, real money)",
}

# Env vars per venue. Paper and live are separate Alpaca accounts with separate
# key pairs.
ENV_KEYS: dict[str, tuple[str, str]] = {
    "alpaca_paper": ("ALPACA_PAPER_API_KEY", "ALPACA_PAPER_SECRET"),
    "alpaca_live": ("ALPACA_LIVE_API_KEY", "ALPACA_LIVE_SECRET"),
}

# Paper -- and ONLY paper -- may fall back to the market-data credentials. An
# Alpaca paper key is usually the same key people already have in
# ALPACA_API_KEY, and making them paste it a second time buys nothing: the worst
# case is a data-only or live key that the paper host rejects with a 401, which
# degrades to local simulation like any other misconfiguration.
#
# Live gets no such fallback, deliberately. If someone's ALPACA_API_KEY happened
# to be a live key, an inherited credential would mean the difference between
# simulated and real money came down to which variable was already set — and
# that is precisely the accident the two gates above exist to prevent. Live
# credentials must be named as live credentials.
PAPER_FALLBACK_ENV: tuple[str, str] = ("ALPACA_API_KEY", "ALPACA_SECRET")


def live_trading_enabled() -> bool:
    """Whether the environment permits live trading at all."""
    return os.getenv(LIVE_TRADING_ENV_FLAG, "").strip().lower() in ("1", "true", "yes", "on")


def confirmation_ok(phrase: str) -> bool:
    return phrase.strip().upper() == LIVE_TRADING_CONFIRM_PHRASE


def credentials_for(mode: str) -> TradingCredentials:
    """Trading credentials for a venue, from its own environment variables.

    Paper falls back to the market-data pair when its own is unset; live never
    does. See PAPER_FALLBACK_ENV.
    """
    key_var, secret_var = ENV_KEYS.get(mode, ("", ""))
    key = os.getenv(key_var, "").strip()
    secret = os.getenv(secret_var, "").strip()
    if mode == "alpaca_paper" and not (key and secret):
        fallback_key, fallback_secret = PAPER_FALLBACK_ENV
        key = os.getenv(fallback_key, "").strip()
        secret = os.getenv(fallback_secret, "").strip()
    return TradingCredentials(key=key, secret=secret, live=(mode == "alpaca_live"))


def verify_account(creds: TradingCredentials) -> "tuple[bool, str]":
    """Can we actually trade this account right now? Returns (ok, message).

    Checked before any order rather than discovered on the first one: a blocked
    or restricted account rejects every order it is sent, and finding that out
    from a strategy's first entry is a worse way to learn it.
    """
    try:
        account = get_account(creds)
    except TradingError as exc:
        return False, str(exc)
    if account.get("trading_blocked") or account.get("account_blocked"):
        return False, f"{creds.venue} account is blocked from trading (status {account.get('status')})"
    cash = account.get("cash")
    equity = account.get("equity")
    return True, (
        f"{creds.venue} account ready — status {account.get('status')}, "
        f"cash ${float(cash or 0):,.2f}, equity ${float(equity or 0):,.2f}"
    )


def resolve_broker(
    mode: str, confirm_phrase: str = ""
) -> "tuple[Broker, str, str]":
    """Turn a requested mode into (broker, effective_mode, message).

    `effective_mode` is what the caller actually got, which is not always what
    it asked for: every refusal below returns the local simulation instead, so
    a misconfigured venue produces simulated trades and a clear message rather
    than an exception in the middle of a strategy's entry.
    """
    if mode not in TRADING_MODES:
        mode = DEFAULT_TRADING_MODE

    if mode == "local":
        return PaperBroker(), "local", "Local simulation — no orders leave this app."

    if mode == "alpaca_live":
        if not live_trading_enabled():
            return (
                PaperBroker(), "local",
                f"LIVE trading refused: {LIVE_TRADING_ENV_FLAG} is not set in the "
                "environment. Falling back to local simulation.",
            )
        if not confirmation_ok(confirm_phrase):
            return (
                PaperBroker(), "local",
                f'LIVE trading refused: type "{LIVE_TRADING_CONFIRM_PHRASE}" to confirm. '
                "Falling back to local simulation.",
            )

    creds = credentials_for(mode)
    if not creds.configured:
        key_var, secret_var = ENV_KEYS[mode]
        extra = (
            f" (or {PAPER_FALLBACK_ENV[0]} / {PAPER_FALLBACK_ENV[1]}, if those are "
            "your paper keys)"
            if mode == "alpaca_paper"
            else ""
        )
        return (
            PaperBroker(), "local",
            f"{MODE_LABELS[mode]} needs {key_var} and {secret_var} in the "
            f"environment{extra}. Falling back to local simulation.",
        )

    ok, message = verify_account(creds)
    if not ok:
        return PaperBroker(), "local", f"{message} Falling back to local simulation."

    broker = AlpacaBroker(creds)
    if mode == "alpaca_live":
        logger.warning("LIVE TRADING ARMED — orders will be sent to the real account")
    return broker, mode, message
