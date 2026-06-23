from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import yfinance as yf

HISTORICAL_PERIODS: dict[str, int] = {
    "7 Days": 7,
    "28 Days": 28,
    "Quarter": 91,
    "1 Year": 365,
    "5 Years": 1825,
}

VIX_SYMBOL = "^VIX"
VIX3M_SYMBOL = "^VIX3M"
SPY_SYMBOL = "SPY"

# Broad-market indicator series change slowly relative to the agent's cycle and
# are identical for every ticker, so cache them briefly to avoid re-hitting
# yfinance on each agent call.
_market_cache: dict = {"ts": None, "data": None}


def fetch_close_series(symbol: str, days: int) -> pd.Series:
    """Fetch daily close prices for `symbol` over the trailing `days`. Raises on failure."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    df = yf.download(symbol, start=start, end=end, interval="1d", auto_adjust=True, progress=False)
    if df.empty:
        return pd.Series(dtype=float)
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close.index = pd.to_datetime(close.index)
    return close.dropna()


def fetch_market_indicators(days: int = 365, ttl_sec: int = 300) -> dict:
    """Fetch the broad-market condition series (SPY, VIX, VIX3M) for `analyze_market`.

    Returns a dict of {"spy", "vix", "vix3m"} -> daily close Series. A failed or
    unavailable symbol yields an empty Series rather than raising, so one bad
    feed never sinks the whole read. Results are cached for `ttl_sec` seconds.
    """
    now = datetime.now(timezone.utc)
    cached = _market_cache["data"]
    cached_ts = _market_cache["ts"]
    if cached is not None and cached_ts is not None and (now - cached_ts).total_seconds() < ttl_sec:
        return cached

    data: dict[str, pd.Series] = {}
    for key, symbol in (("spy", SPY_SYMBOL), ("vix", VIX_SYMBOL), ("vix3m", VIX3M_SYMBOL)):
        try:
            data[key] = fetch_close_series(symbol, days)
        except Exception:
            data[key] = pd.Series(dtype=float)

    _market_cache["ts"] = now
    _market_cache["data"] = data
    return data


def fetch_dividends(symbol: str, days: int) -> pd.Series:
    """Fetch dividend payouts for `symbol` over the trailing `days`."""
    div = yf.Ticker(symbol).dividends
    if div.empty:
        return div
    cutoff = pd.Timestamp.now(tz=div.index.tz) - pd.Timedelta(days=days)
    return div[div.index >= cutoff]


def fetch_earnings_dates(symbol: str, days: int) -> pd.DataFrame:
    """Fetch past and upcoming earnings dates for `symbol` over the trailing `days`."""
    try:
        earnings = yf.Ticker(symbol).get_earnings_dates(limit=20)
    except Exception:
        return pd.DataFrame()
    if earnings is None or earnings.empty:
        return pd.DataFrame()
    cutoff = pd.Timestamp.now(tz=earnings.index.tz) - pd.Timedelta(days=days)
    return earnings[earnings.index >= cutoff]


def fetch_static_analysis(symbol: str) -> dict:
    """Fetch the raw inputs (P/E, growth, dividend yield) for a simple static valuation estimate."""
    try:
        info = yf.Ticker(symbol).info
    except Exception:
        info = {}
    growth_rate = info.get("earningsGrowth")
    if growth_rate is None:
        growth_rate = info.get("revenueGrowth")
    return {
        "pe_ratio": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "dividend_yield": info.get("trailingAnnualDividendYield"),
        "growth_rate": growth_rate,
    }


def estimate_total_return(dividend_yield: Optional[float], growth_rate: Optional[float]) -> Optional[float]:
    """Rough estimate of annual total return on the asset: dividend yield + earnings/revenue growth."""
    if dividend_yield is None or growth_rate is None:
        return None
    return dividend_yield + growth_rate


def estimate_dividend_return_10y(
    dividend_yield: Optional[float], growth_rate: Optional[float], years: int = 10
) -> Optional[float]:
    """Cumulative dividends collected over `years`, as a fraction of the current price.

    Assumes the dividend grows annually at `growth_rate` from today's yield.
    """
    if dividend_yield is None or growth_rate is None:
        return None
    return sum(dividend_yield * (1 + growth_rate) ** t for t in range(years))
