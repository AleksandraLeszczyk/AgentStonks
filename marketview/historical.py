from datetime import datetime, timedelta, timezone

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
SPY_SYMBOL = "SPY"


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
