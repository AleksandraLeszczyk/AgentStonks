"""Dataset download and storage for SimLab.

Everything a simulated session needs is downloaded once and kept in a local
file store, deduplicated at the (symbol, trading day) level so overlapping
datasets never re-download or re-store the same day:

    data/simlab/
      store/
        bars/{SYMBOL}/{YYYY-MM-DD}.json.gz   1-minute bars, 04:00-20:00 ET
        daily/{SYMBOL}.json.gz               daily bars (range in the payload)
        news/{SYMBOL}/{YYYY-MM-DD}.json.gz   news articles created that day
        market/indicators.json.gz            SPY/VIX/VIX3M daily closes
      datasets.json                          named dataset manifest

A *dataset* is a named bundle: symbols + an inclusive date range. Creating one
walks the range and fills only the store files that are missing; deleting one
only removes the manifest entry (the store is shared).

Bars/news keep Alpaca's native dict shapes ({"t","o","h","l","c","v",...}) so
everything downstream (SymbolState, technical_analysis) consumes them as-is.
"""
from __future__ import annotations

import gzip
import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional

import requests

from agent_stonks.config import DATA_REST
from agent_stonks.market_hours import MARKET_TZ

SIMLAB_DIR = Path(__file__).resolve().parent.parent / "data" / "simlab"
STORE_DIR = SIMLAB_DIR / "store"
MANIFEST_PATH = SIMLAB_DIR / "datasets.json"

# Stored minute-bar window per trading day, ET: full pre-market through
# post-market so premarket reads and opening tactics have real tape.
DAY_START_ET = time(4, 0)
DAY_END_ET = time(20, 0)

# Daily-bar history stored per symbol: enough for analyze_daily_trend (1y),
# order blocks, and the ADV/rvol baselines at any simulated day in range.
DAILY_LOOKBACK_DAYS = 420

MARKET_INDICATOR_SYMBOLS = {"spy": "SPY", "vix": "^VIX", "vix3m": "^VIX3M"}

_manifest_lock = threading.Lock()

ProgressCb = Callable[[str], None]


def _noop_progress(_msg: str) -> None:
    return None


# ---------------------------------------------------------------------------
# Low-level store
# ---------------------------------------------------------------------------

def _read_gz(path: Path) -> object:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def _write_gz(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    tmp.replace(path)


def bars_path(symbol: str, day: date) -> Path:
    return STORE_DIR / "bars" / symbol.upper() / f"{day.isoformat()}.json.gz"


def news_path(symbol: str, day: date) -> Path:
    return STORE_DIR / "news" / symbol.upper() / f"{day.isoformat()}.json.gz"


def daily_path(symbol: str) -> Path:
    return STORE_DIR / "daily" / f"{symbol.upper()}.json.gz"


def market_path() -> Path:
    return STORE_DIR / "market" / "indicators.json.gz"


def load_day_bars(symbol: str, day: date) -> list[dict]:
    """Stored 1-minute bars for one (symbol, day); [] when the day has no
    session (holiday) or hasn't been downloaded."""
    path = bars_path(symbol, day)
    if not path.exists():
        return []
    return _read_gz(path)  # type: ignore[return-value]


def load_daily_bars(symbol: str) -> list[dict]:
    path = daily_path(symbol)
    if not path.exists():
        return []
    return _read_gz(path).get("bars", [])  # type: ignore[union-attr]


def load_news(symbol: str, day: date) -> list[dict]:
    path = news_path(symbol, day)
    if not path.exists():
        return []
    return _read_gz(path)  # type: ignore[return-value]


def load_market_indicators() -> dict[str, list[dict]]:
    """{"spy"|"vix"|"vix3m": [{"date": "YYYY-MM-DD", "close": float}, ...]}"""
    path = market_path()
    if not path.exists():
        return {}
    return _read_gz(path)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Dataset manifest
# ---------------------------------------------------------------------------

@dataclass
class Dataset:
    name: str
    symbols: list[str]
    start: str  # inclusive, YYYY-MM-DD
    end: str  # inclusive, YYYY-MM-DD
    created_at: str = ""
    # Trading days (YYYY-MM-DD) that actually have bars for at least one
    # symbol -- weekends/holidays in the range are absent.
    days: list[str] = field(default_factory=list)

    def date_range(self) -> tuple[date, date]:
        return date.fromisoformat(self.start), date.fromisoformat(self.end)


def list_datasets() -> list[Dataset]:
    if not MANIFEST_PATH.exists():
        return []
    raw = json.loads(MANIFEST_PATH.read_text())
    return [Dataset(**entry) for entry in raw]


def get_dataset(name: str) -> Optional[Dataset]:
    return next((d for d in list_datasets() if d.name == name), None)


def _save_manifest(datasets: list[Dataset]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps([asdict(d) for d in datasets], indent=2))


def delete_dataset(name: str) -> None:
    """Remove a dataset from the manifest. Store files are shared across
    datasets and deliberately kept."""
    with _manifest_lock:
        _save_manifest([d for d in list_datasets() if d.name != name])


# ---------------------------------------------------------------------------
# Alpaca fetchers (range-based, paginated -- rest.py's live helpers are
# anchored to "now", which is exactly what a downloader must not be)
# ---------------------------------------------------------------------------

def _headers(key: str, secret: str) -> dict[str, str]:
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}


def _paged_get(url: str, params: dict, key: str, secret: str, item_key: str, symbol: str) -> list:
    """Follow Alpaca's next_page_token pagination until exhausted."""
    out: list = []
    token: Optional[str] = None
    while True:
        page_params = dict(params)
        if token:
            page_params["page_token"] = token
        r = requests.get(url, headers=_headers(key, secret), params=page_params, timeout=30)
        r.raise_for_status()
        payload = r.json()
        container = payload.get(item_key) or {}
        items = container.get(symbol, []) if isinstance(container, dict) else container
        out.extend(items or [])
        token = payload.get("next_page_token")
        if not token:
            return out


def fetch_minute_bars_day(symbol: str, day: date, key: str, secret: str, feed: str = "iex") -> list[dict]:
    """All 1-minute bars for one trading day, 04:00-20:00 ET."""
    start = datetime.combine(day, DAY_START_ET, tzinfo=MARKET_TZ).astimezone(timezone.utc)
    end = datetime.combine(day, DAY_END_ET, tzinfo=MARKET_TZ).astimezone(timezone.utc)
    return _paged_get(
        f"{DATA_REST}/v2/stocks/bars",
        dict(
            symbols=symbol,
            timeframe="1Min",
            start=start.isoformat(),
            end=end.isoformat(),
            limit=10000,
            feed=feed,
        ),
        key,
        secret,
        "bars",
        symbol,
    )


def fetch_daily_bars_range(
    symbol: str, start: date, end: date, key: str, secret: str, feed: str = "iex"
) -> list[dict]:
    return _paged_get(
        f"{DATA_REST}/v2/stocks/bars",
        dict(
            symbols=symbol,
            timeframe="1Day",
            start=datetime.combine(start, time(0, 0), tzinfo=timezone.utc).isoformat(),
            end=datetime.combine(end, time(23, 59), tzinfo=timezone.utc).isoformat(),
            limit=10000,
            feed=feed,
        ),
        key,
        secret,
        "bars",
        symbol,
    )


def fetch_news_day(symbol: str, day: date, key: str, secret: str) -> list[dict]:
    """News articles for `symbol` created during `day` (UTC)."""
    start = datetime.combine(day, time(0, 0), tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    r = requests.get(
        f"{DATA_REST}/v1beta1/news",
        headers=_headers(key, secret),
        params=dict(
            symbols=symbol, start=start.isoformat(), end=end.isoformat(), limit=50, sort="desc"
        ),
        timeout=30,
    )
    r.raise_for_status()
    return r.json().get("news", [])


def fetch_market_indicator_closes(start: date, end: date) -> dict[str, list[dict]]:
    """Daily closes for SPY/VIX/VIX3M over [start, end] via yfinance."""
    import yfinance as yf

    out: dict[str, list[dict]] = {}
    for name, ticker in MARKET_INDICATOR_SYMBOLS.items():
        try:
            frame = yf.Ticker(ticker).history(
                start=start.isoformat(),
                end=(end + timedelta(days=1)).isoformat(),
                interval="1d",
                auto_adjust=False,
            )
            closes = frame["Close"].dropna()
            out[name] = [
                {"date": idx.date().isoformat(), "close": float(value)}
                for idx, value in closes.items()
            ]
        except Exception:
            out[name] = []
    return out


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------

def weekdays(start: date, end: date) -> Iterable[date]:
    day = start
    while day <= end:
        if day.weekday() < 5:
            yield day
        day += timedelta(days=1)


def coverage(symbols: list[str], start: date, end: date) -> dict[str, dict[str, bool]]:
    """{day -> {symbol -> already stored}} for every weekday in range."""
    return {
        day.isoformat(): {sym: bars_path(sym, day).exists() for sym in symbols}
        for day in weekdays(start, end)
    }


def _daily_covers(symbol: str, start: date, end: date) -> bool:
    """Whether the stored daily-bar file spans [start - lookback, end]."""
    path = daily_path(symbol)
    if not path.exists():
        return False
    meta = _read_gz(path)
    try:
        have_start = date.fromisoformat(meta["start"])
        have_end = date.fromisoformat(meta["end"])
    except (KeyError, ValueError, TypeError):
        return False
    return have_start <= start - timedelta(days=DAILY_LOOKBACK_DAYS) and have_end >= end


def _market_covers(start: date, end: date) -> bool:
    data = load_market_indicators()
    spy = data.get("spy") or []
    if not spy:
        return False
    dates = [row["date"] for row in spy]
    return dates[0] <= (start - timedelta(days=DAILY_LOOKBACK_DAYS)).isoformat() and dates[-1] >= (
        end - timedelta(days=4)
    ).isoformat()


def create_dataset(
    name: str,
    symbols: list[str],
    start: date,
    end: date,
    key: str,
    secret: str,
    feed: str = "iex",
    progress: ProgressCb = _noop_progress,
) -> Dataset:
    """Create (or refresh) a named dataset, downloading only what the store
    is missing. Returns the manifest entry with its resolved trading days."""
    symbols = [s.strip().upper() for s in symbols if s.strip()]
    if not symbols:
        raise ValueError("dataset needs at least one symbol")
    if end < start:
        raise ValueError("dataset end date is before its start date")

    # Daily bars first: they double as the trading-day calendar for the range.
    daily_start = start - timedelta(days=DAILY_LOOKBACK_DAYS)
    for sym in symbols:
        if _daily_covers(sym, start, end):
            progress(f"daily bars {sym}: already stored")
            continue
        progress(f"daily bars {sym}: downloading {daily_start} … {end}")
        bars = fetch_daily_bars_range(sym, daily_start, end, key, secret, feed)
        _write_gz(
            daily_path(sym),
            {"symbol": sym, "start": daily_start.isoformat(), "end": end.isoformat(), "bars": bars},
        )

    if _market_covers(start, end):
        progress("market indicators (SPY/VIX/VIX3M): already stored")
    else:
        progress("market indicators (SPY/VIX/VIX3M): downloading")
        _write_gz(market_path(), fetch_market_indicator_closes(daily_start, end))

    session_days: list[str] = []
    for day in weekdays(start, end):
        day_has_bars = False
        for sym in symbols:
            path = bars_path(sym, day)
            if path.exists():
                day_has_bars = day_has_bars or bool(load_day_bars(sym, day))
                continue
            progress(f"minute bars {sym} {day}: downloading")
            bars = fetch_minute_bars_day(sym, day, key, secret, feed)
            _write_gz(path, bars)
            day_has_bars = day_has_bars or bool(bars)

            npath = news_path(sym, day)
            if not npath.exists():
                try:
                    _write_gz(npath, fetch_news_day(sym, day, key, secret))
                except Exception as exc:  # news is nice-to-have, never fatal
                    progress(f"news {sym} {day}: failed ({exc}); storing empty")
                    _write_gz(npath, [])
        if not day_has_bars:
            # Existing empty files (holiday) or fresh empty downloads.
            day_has_bars = any(load_day_bars(sym, day) for sym in symbols)
        if day_has_bars:
            session_days.append(day.isoformat())

    dataset = Dataset(
        name=name,
        symbols=symbols,
        start=start.isoformat(),
        end=end.isoformat(),
        created_at=datetime.now(timezone.utc).isoformat(),
        days=session_days,
    )
    with _manifest_lock:
        existing = [d for d in list_datasets() if d.name != name]
        _save_manifest([*existing, dataset])
    progress(f"dataset '{name}' ready: {len(session_days)} trading day(s)")
    return dataset


def store_size_bytes() -> int:
    if not STORE_DIR.exists():
        return 0
    return sum(p.stat().st_size for p in STORE_DIR.rglob("*") if p.is_file())
