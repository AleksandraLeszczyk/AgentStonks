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


def fetch_corporate_actions(
    symbol: str,
    key: str,
    secret: str,
    days_ahead: int = 14,
) -> list[dict]:
    """Fetch incoming corporate actions (dividends, splits, mergers, spin-offs, ...)
    taking effect between today and today+days_ahead from Alpaca. The response is
    grouped by action type; flatten it into one chronological list of dicts, each
    carrying "type", its anchor "date" (ex/effective/process date), and the raw
    per-action fields. Raises on HTTP error."""
    start = datetime.now(timezone.utc).date()
    end = start + timedelta(days=days_ahead)
    r = requests.get(
        f"{DATA_REST}/v1beta1/corporate-actions",
        headers=_headers(key, secret),
        params=dict(
            symbols=symbol,
            start=start.isoformat(),
            end=end.isoformat(),
            limit=1000,
            sort="asc",
        ),
        timeout=10,
    )
    r.raise_for_status()
    grouped = r.json().get("corporate_actions") or {}
    actions: list[dict] = []
    for group, entries in grouped.items():
        kind = group[:-1] if group.endswith("s") else group  # "cash_dividends" -> "cash_dividend"
        for entry in entries or []:
            date = next(
                (
                    entry[field]
                    for field in ("ex_date", "effective_date", "process_date", "payable_date")
                    if entry.get(field)
                ),
                None,
            )
            actions.append({"type": kind, "date": date, **entry})
    actions.sort(key=lambda action: action["date"] or "9999-12-31")
    return actions


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
