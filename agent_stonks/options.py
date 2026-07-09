"""
Options chain fetching for put/call wall + gamma exposure analysis.

Mirrors `historical.py`'s relationship to `technical_analysis.py`: this module
only fetches and shapes raw data (open interest and a Black-Scholes gamma per
strike, independent of any agent call); `technical_analysis.get_put_call_walls_and_gamma`
turns that into a labeled read. Refreshing here happens on the UI's own poll
loop (see `ui.py`), never inside an agent tool call -- the agent only ever
reads whatever was fetched most recently from `AppState`.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

import yfinance as yf

from .datalog import log_fetch, log_fetch_failure

RISK_FREE_RATE = 0.045
CONTRACT_MULTIPLIER = 100
DEFAULT_MAX_DTE = 45

_walls_cache: dict[str, dict] = {}


def _select_expiry(expirations: list[str], max_dte: int = DEFAULT_MAX_DTE) -> str:
    """Pick the nearest future expiry, preferring one within `max_dte` days."""
    if not expirations:
        raise ValueError("no option expirations available")
    today = datetime.now(timezone.utc).date()
    parsed = [(datetime.strptime(e, "%Y-%m-%d").date(), e) for e in expirations]
    future = [(d, e) for d, e in parsed if (d - today).days >= 0] or parsed
    within_window = [(d, e) for d, e in future if (d - today).days <= max_dte]
    candidates = within_window or future
    return min(candidates, key=lambda pair: pair[0])[1]


def _bs_gamma(spot: float, strike: float, t_years: float, vol: float, r: float = RISK_FREE_RATE) -> float:
    """Black-Scholes gamma -- identical for calls and puts at the same strike/maturity."""
    if spot <= 0 or strike <= 0 or t_years <= 0 or vol <= 0:
        return 0.0
    d1 = (math.log(spot / strike) + (r + 0.5 * vol**2) * t_years) / (vol * math.sqrt(t_years))
    pdf = math.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi)
    return pdf / (spot * vol * math.sqrt(t_years))


def _fetch_spot(ticker: "yf.Ticker") -> float:
    symbol = getattr(ticker, "ticker", "")
    failures: list[tuple[str, object]] = []
    try:
        price = ticker.fast_info.get("lastPrice")
        if price:
            price = float(price)
            log_fetch("spot price", "yfinance fast_info", symbol=symbol, detail=f"{price}")
            return price
        failures.append(("yfinance fast_info", "no lastPrice in response"))
    except Exception as exc:
        failures.append(("yfinance fast_info", exc))
    hist = ticker.history(period="1d")
    if not hist.empty:
        price = float(hist["Close"].iloc[-1])
        log_fetch(
            "spot price",
            "yfinance daily history",
            symbol=symbol,
            detail=f"{price}",
            failures=failures,
        )
        return price
    failures.append(("yfinance daily history", "no rows returned"))
    log_fetch_failure("spot price", failures, symbol=symbol)
    raise ValueError("could not determine spot price")


def _oi(table, strike: float) -> float:
    if strike not in table.index:
        return 0.0
    value = table.loc[strike, "openInterest"]
    return float(value) if value == value else 0.0  # NaN check


def _iv(table, strike: float) -> float:
    if strike not in table.index:
        return 0.0
    value = table.loc[strike, "impliedVolatility"]
    return float(value) if value == value else 0.0  # NaN check


def fetch_option_chain(
    symbol: str,
    spot: "float | None" = None,
    expiry: "str | None" = None,
    max_dte: int = DEFAULT_MAX_DTE,
) -> dict:
    """Fetch the options chain for `symbol` and compute open interest + dollar gamma
    exposure per strike. Raises on failure (no expirations, no chain data, etc.)."""
    ticker = yf.Ticker(symbol)
    try:
        expirations = list(ticker.options)
        chosen_expiry = expiry or _select_expiry(expirations, max_dte)
        chain = ticker.option_chain(chosen_expiry)
    except Exception as exc:
        log_fetch_failure(
            "options chain",
            [("yfinance", exc)],
            symbol=symbol,
            consequence="no put/call wall data",
        )
        raise
    log_fetch(
        "options chain",
        "yfinance",
        symbol=symbol,
        detail=f"expiry {chosen_expiry}, {len(chain.calls)} calls / {len(chain.puts)} puts",
    )
    calls = chain.calls.set_index("strike")
    puts = chain.puts.set_index("strike")

    if spot is None:
        spot = _fetch_spot(ticker)

    today = datetime.now(timezone.utc).date()
    expiry_date = datetime.strptime(chosen_expiry, "%Y-%m-%d").date()
    t_years = max((expiry_date - today).days, 1) / 365.0

    strikes = sorted(set(calls.index) | set(puts.index))
    calls_oi, puts_oi, calls_gamma_exposure, puts_gamma_exposure = [], [], [], []
    for k in strikes:
        c_oi, p_oi = _oi(calls, k), _oi(puts, k)
        c_gamma = _bs_gamma(spot, k, t_years, _iv(calls, k))
        p_gamma = _bs_gamma(spot, k, t_years, _iv(puts, k))
        # Dollar gamma exposure per 1% underlying move; dealers are assumed net long
        # calls (positive gamma contribution) and net short puts (negative contribution)
        # -- the standard convention used by retail gamma-exposure trackers.
        calls_oi.append(c_oi)
        puts_oi.append(p_oi)
        calls_gamma_exposure.append(c_gamma * c_oi * CONTRACT_MULTIPLIER * spot**2 * 0.01)
        puts_gamma_exposure.append(-p_gamma * p_oi * CONTRACT_MULTIPLIER * spot**2 * 0.01)

    return {
        "symbol": symbol,
        "expiry": chosen_expiry,
        "spot": float(spot),
        "strikes": strikes,
        "calls_oi": calls_oi,
        "puts_oi": puts_oi,
        "calls_gamma_exposure": calls_gamma_exposure,
        "puts_gamma_exposure": puts_gamma_exposure,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


def fetch_options_walls_data(symbol: str, spot: "float | None" = None, ttl_sec: int = 300) -> dict:
    """Cached wrapper around `fetch_option_chain` -- options open interest moves slowly
    relative to a poll loop, so avoid re-hitting yfinance on every refresh."""
    now = datetime.now(timezone.utc)
    cached = _walls_cache.get(symbol)
    if cached is not None and (now - cached["ts"]).total_seconds() < ttl_sec:
        return cached["data"]
    data = fetch_option_chain(symbol, spot=spot)
    _walls_cache[symbol] = {"ts": now, "data": data}
    return data
