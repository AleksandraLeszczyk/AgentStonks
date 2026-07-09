"""Console logging of where market data comes from and whether fetching worked.

Every data-acquisition path (WebSocket stream, Alpaca REST, yfinance, WorldNews)
reports through :func:`log_fetch` / :func:`log_fetch_failure` so the console
always shows, in one consistent format, which source served each kind of data
and which sources were tried and failed first, e.g.::

    AAPL: bars fetched from yfinance (delayed) (fetching from Alpaca REST failed: 403 Forbidden)
    AAPL: ask/bid price fetch failed — tried Alpaca REST /quotes/latest: timeout — keeping last known values

Recurring fetches (the 15s REST fallback poll, per-tick stream messages) would
flood the console if every identical outcome were printed, so outcomes are
de-duplicated by *signature* — the (what, symbol, source, failed-sources)
combination. A repeat of the same outcome logs at DEBUG; any change (source
switched, a fallback kicked in, recovery after failure) logs at INFO/WARNING
again. Values that change every tick (prices, row counts) go in ``detail`` and
are excluded from the signature.
"""
from __future__ import annotations

import logging
import threading

logger = logging.getLogger("agent_stonks.data")

# Last outcome signature per (what, symbol) — see module docstring.
_last_outcome: dict[tuple[str, str], tuple] = {}
_lock = threading.Lock()


def _dedup_level(what: str, symbol: str, signature: tuple, level: int) -> int:
    """`level` the first time this outcome is seen for (what, symbol), DEBUG on repeats."""
    key = (what, symbol)
    with _lock:
        if _last_outcome.get(key) == signature:
            return logging.DEBUG
        _last_outcome[key] = signature
    return level


def _failures_text(failures: "list[tuple[str, object]] | None") -> str:
    """'(fetching from a, b failed: err_a; err_b)' or '' when nothing failed."""
    if not failures:
        return ""
    sources = ", ".join(source for source, _ in failures)
    errors = "; ".join(str(error) for _, error in failures)
    return f" (fetching from {sources} failed: {errors})"


def log_fetch(
    what: str,
    source: str,
    *,
    symbol: str = "",
    detail: str = "",
    failures: "list[tuple[str, object]] | None" = None,
) -> None:
    """Log that `what` was successfully fetched from `source`.

    `failures` lists (source, error) pairs that were attempted and failed before
    `source` succeeded, e.g. the Alpaca REST error that pushed a fetch to the
    yfinance fallback. `detail` carries per-fetch values (prices, counts) and
    does not affect de-duplication.
    """
    prefix = f"{symbol}: " if symbol else ""
    message = f"{prefix}{what} fetched from {source}"
    if detail:
        message += f" ({detail})"
    message += _failures_text(failures)
    signature = ("ok", source, tuple(s for s, _ in failures or []))
    logger.log(_dedup_level(what, symbol, signature, logging.INFO), message)


def log_fetch_failure(
    what: str,
    failures: "list[tuple[str, object]]",
    *,
    symbol: str = "",
    consequence: str = "",
) -> None:
    """Log that fetching `what` failed from every attempted source.

    `failures` lists every (source, error) attempted; `consequence` says what the
    app does about it ("keeping last known values", "panel stays empty", …).
    """
    prefix = f"{symbol}: " if symbol else ""
    tried = "; ".join(f"{source}: {error}" for source, error in failures)
    message = f"{prefix}{what} fetch failed — tried {tried}"
    if consequence:
        message += f" — {consequence}"
    signature = ("failed", tuple(s for s, _ in failures))
    logger.log(_dedup_level(what, symbol, signature, logging.WARNING), message)
