from datetime import datetime, timedelta, timezone

import requests

from .config import DATA_REST


def _headers(key: str, secret: str) -> dict[str, str]:
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}


def fetch_bars(
    symbol: str,
    timeframe: str,
    limit: int,
    key: str,
    secret: str,
    feed: str = "iex",
    lookback_hours: int = 6,
) -> list[dict]:
    """Fetch historical OHLCV bars from Alpaca. Raises on HTTP error."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=lookback_hours)
    r = requests.get(
        f"{DATA_REST}/v2/stocks/bars",
        headers=_headers(key, secret),
        params=dict(
            symbols=symbol,
            timeframe=timeframe,
            start=start.isoformat(),
            end=end.isoformat(),
            limit=limit,
            feed=feed,
        ),
        timeout=10,
    )
    r.raise_for_status()
    return r.json().get("bars", {}).get(symbol, [])


def fetch_trades(
    symbol: str,
    key: str,
    secret: str,
    feed: str = "iex",
    lookback_hours: int = 8,
) -> list[dict]:
    """Fetch recent trades from Alpaca. Raises on HTTP error."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=lookback_hours)
    r = requests.get(
        f"{DATA_REST}/v2/stocks/trades",
        headers=_headers(key, secret),
        params=dict(
            symbols=symbol,
            start=start.isoformat(),
            end=end.isoformat(),
            limit=10000,
            feed=feed,
        ),
        timeout=10,
    )
    r.raise_for_status()
    return r.json().get("trades", {}).get(symbol, [])


def fetch_daily_bars(
    symbol: str,
    key: str,
    secret: str,
    feed: str = "iex",
    lookback_days: int = 365,
) -> list[dict]:
    """Fetch daily OHLCV bars going back lookback_days for multi-day avg computations."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)
    r = requests.get(
        f"{DATA_REST}/v2/stocks/bars",
        headers=_headers(key, secret),
        params=dict(
            symbols=symbol,
            timeframe="1Day",
            start=start.isoformat(),
            end=end.isoformat(),
            limit=lookback_days + 10,
            feed=feed,
        ),
        timeout=10,
    )
    r.raise_for_status()
    return r.json().get("bars", {}).get(symbol, [])


def fetch_latest_trade(
    symbol: str,
    key: str,
    secret: str,
    feed: str = "iex",
) -> dict:
    """Fetch the single most recent trade for a symbol. Raises on HTTP error."""
    r = requests.get(
        f"{DATA_REST}/v2/stocks/{symbol}/trades/latest",
        headers=_headers(key, secret),
        params=dict(feed=feed),
        timeout=10,
    )
    r.raise_for_status()
    return r.json().get("trade", {})


def fetch_latest_quote(
    symbol: str,
    key: str,
    secret: str,
    feed: str = "iex",
) -> dict:
    """Fetch the single most recent NBBO quote for a symbol. Raises on HTTP error."""
    r = requests.get(
        f"{DATA_REST}/v2/stocks/{symbol}/quotes/latest",
        headers=_headers(key, secret),
        params=dict(feed=feed),
        timeout=10,
    )
    r.raise_for_status()
    return r.json().get("quote", {})


def fetch_news(
    symbol: str,
    key: str,
    secret: str,
    limit: int = 15,
) -> list[dict]:
    """Fetch recent news articles from Alpaca. Raises on HTTP error."""
    r = requests.get(
        f"{DATA_REST}/v1beta1/news",
        headers=_headers(key, secret),
        params=dict(symbols=symbol, limit=limit),
        timeout=10,
    )
    r.raise_for_status()
    return r.json().get("news", [])
