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

import numpy as np
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
        # Yahoo now and then answers a dated chain request with no contracts,
        # which yfinance hands back as calls=None / puts=None rather than
        # raising. It is transient (the same request a minute later is fine),
        # so ask once more before giving up.
        if chain.calls is None or chain.puts is None:
            chain = ticker.option_chain(chosen_expiry)
        if chain.calls is None or chain.puts is None:
            raise ValueError(f"yfinance returned no contracts for expiry {chosen_expiry}")
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
    calls_iv, puts_iv = [], []
    for k in strikes:
        c_oi, p_oi = _oi(calls, k), _oi(puts, k)
        c_iv, p_iv = _iv(calls, k), _iv(puts, k)
        calls_iv.append(c_iv)
        puts_iv.append(p_iv)
        c_gamma = _bs_gamma(spot, k, t_years, c_iv)
        p_gamma = _bs_gamma(spot, k, t_years, p_iv)
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
        # What `net_gamma_exposure` re-prices the chain with at another spot.
        "calls_iv": calls_iv,
        "puts_iv": puts_iv,
        "t_years": t_years,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


def net_gamma_exposure(data: dict, spots) -> "np.ndarray | None":
    """Net dealer gamma ($ per 1% move) of the chain `data` at each price in
    `spots`, or None for a chain fetched before it carried its IVs.

    The same sum as `calls_gamma_exposure + puts_gamma_exposure`, which is this
    at the spot the chain was fetched at, but re-priced at other spots: open
    interest only changes overnight, so one chain says what the dealers' gamma
    was at every price the session has traded. The IVs are held at the fetch's,
    which is the approximation.
    """
    if not data or "calls_iv" not in data or not data.get("strikes"):
        return None
    s = np.asarray(spots, dtype=float).reshape(-1, 1)
    k = np.asarray(data["strikes"], dtype=float).reshape(1, -1)
    t = float(data["t_years"])

    def gamma(iv) -> np.ndarray:
        vol = np.asarray(iv, dtype=float).reshape(1, -1)
        ok = (s > 0) & (k > 0) & (vol > 0) & (t > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            root = vol * math.sqrt(t)
            d1 = (np.log(s / k) + (RISK_FREE_RATE + 0.5 * vol**2) * t) / root
            g = np.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi) / (s * root)
        return np.where(ok, g, 0.0)

    scale = CONTRACT_MULTIPLIER * s[:, 0] ** 2 * 0.01
    calls = gamma(data["calls_iv"]) @ np.asarray(data["calls_oi"], dtype=float)
    puts = gamma(data["puts_iv"]) @ np.asarray(data["puts_oi"], dtype=float)
    return (calls - puts) * scale


# `gamma_flip` scans log price this far either side of the price, in steps of
# 0.05% -- fine against a next-day expiry's gamma, ~1.5% of price wide. Each
# scan is a few milliseconds, so the live chart can afford one per redraw.
FLIP_WINDOW = 0.25
_FLIP_STEP = 0.0005


def gamma_flip(data: dict, near: "float | None" = None) -> "float | None":
    """The price nearest `near` (default: the chain's own spot) at which the
    chain's net dealer gamma (`net_gamma_exposure`) changes sign, or None when
    it keeps one sign within FLIP_WINDOW (log) of that price, or the chain has
    no IVs.

    The price where the live chart's net gamma panel changes colour, and how
    HighLow_3m's `gamma_flip` feature reads its own chains. Not the Put/Call
    Walls tab's Gamma Flip (`technical_analysis.get_put_call_walls_and_gamma`):
    that is the strike where per-strike gamma at today's spot, summed from the
    lowest strike up, turns positive, which on a near expiry can sit on the
    other side of the price from this one, or not exist.
    """
    if not data or not data.get("strikes"):
        return None
    near = float(near if near is not None else data.get("spot") or 0.0)
    if not near > 0:
        return None
    n = int(round(2 * FLIP_WINDOW / _FLIP_STEP)) + 1
    x = math.log(near) + np.linspace(-FLIP_WINDOW, FLIP_WINDOW, n)
    values = net_gamma_exposure(data, np.exp(x))
    if values is None:
        return None
    # A spot far from every strike can price to exactly zero; a change of sign
    # across such a stretch is still a crossing, between the samples either side.
    keep = values != 0
    x, values = x[keep], values[keep]
    left, right = values[:-1], values[1:]
    at = np.nonzero(np.sign(left) != np.sign(right))[0]
    if not at.size:
        return None
    # Linear in log price between the two samples either side of the change.
    cross = x[at] - left[at] * (x[at + 1] - x[at]) / (right[at] - left[at])
    return float(np.exp(cross[np.argmin(np.abs(cross - math.log(near)))]))


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
