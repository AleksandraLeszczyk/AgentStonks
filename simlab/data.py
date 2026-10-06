"""Dataset download and storage for SimLab.

Everything a simulated session needs is downloaded once and kept in a local
file store, deduplicated at the (feed, symbol, trading day) level so
overlapping datasets never re-download or re-store the same day:

    data/simlab/
      store/
        bars/{FEED}/{SYMBOL}/{YYYY-MM-DD}.json.gz   1-minute bars, 04:00-20:00 ET
        daily/{FEED}/{SYMBOL}.json.gz               daily bars (range in the payload)
        news/{SYMBOL}/{YYYY-MM-DD}.json.gz          news articles created that day
        market/indicators.json.gz                   SPY/VIX/VIX3M daily closes
      datasets.json                                 named dataset manifest

A *dataset* is a named bundle: symbols + an inclusive date range + the feed its
bars came from. Creating one walks the range and fills only the store files
that are missing; deleting one only removes the manifest entry (the store is
shared).

Why the feed is part of the key
-------------------------------
`yfinance`, `iex` and `sip` are not three routes to the same numbers. IEX is
one venue -- about 4% of consolidated volume on a large-cap name -- so its
minute bars carry different closes (a few cents), different volumes (~25x
smaller) and sometimes an extra or missing bar. Any agent whose rules are
thresholds over those bars can trade a different day on each tape, and a model
reading volume or volatility features sees genuinely different inputs.

Keying the store on (symbol, day) alone made that invisible *and*
unfixable: a day already downloaded on IEX satisfied the "already stored"
check, so asking for `sip` silently reused the IEX bars and produced a dataset
that claimed a tape it did not have. The feed is therefore part of the path,
recorded on the manifest entry, and carried into the run record.

Days downloaded before feeds were tracked were all `iex` (it was the only
default), and they stay where they are -- the readers fall back to the pre-feed
layout for `iex` rather than forcing a re-download.

Bars/news keep Alpaca's native dict shapes ({"t","o","h","l","c","v",...}) so
everything downstream (SymbolState, technical_analysis) consumes them as-is;
the yfinance fetchers convert into that same shape.

A stored day is whole only if it was fetched after it ended
-----------------------------------------------------------
A day downloaded while it is still trading is a prefix of that day, and on disk
it looked exactly like the whole day. 2026-09-09's AAPL/GOOG/GOOGL/ORCL/SPY bars
were fetched at 10:29 ET and stopped there, so every replay of that session ran
out of tape at 10:29 -- no flatten, the position carried overnight. A day fetched
before it began stored empty and read as a holiday (2026-08-21, fetched 08-20).
Both were then "already stored" for every later dataset and never fetched again.

A bar file does not say when it was fetched, but its mtime does: `_write_gz` is
the only writer and replaces a file whole. So a stored day counts as stored
only when it was written after `day_final_at(day)`; earlier is *premature*,
and is re-downloaded by `create_dataset` and by `repair_incomplete` (run by the
SimLab app, the experiment runner and tuning jobs) once the day can be had
whole. A day stored empty although its daily history shows a session is
incomplete too (a failed fetch reads as `[]`). A re-download never replaces a
stored day with fewer bars, so a day Yahoo no longer serves keeps the part it
has. The daily history and the market indicators follow the same rule through
their own fetch time.
"""
from __future__ import annotations

import gzip
import json
import os
import threading
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional

import requests

from agent_stonks import minute_momentum
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

# The tapes a dataset's bars can come from.
#
# `yfinance` is the default and the source of every newly downloaded minute
# bar: consolidated-tape OHLCV, free, no Alpaca subscription, and the same
# source the live volume tools read (agent_stonks.historical), so a simulated
# volume ratio is like-for-like with the live one. Its cost is reach -- Yahoo
# serves 1-minute history for the last 30 days only (see YF_MINUTE_WINDOW_DAYS)
# and publishes no per-bar VWAP, so bars carry no `vw` field.
#
# `iex` and `sip` are Alpaca's tapes, kept so every dataset downloaded before
# this still loads and can still be re-run. `iex` is the free single-venue feed
# and what every day stored before the feed was tracked contains. `sip` is
# Alpaca's consolidated tape and needs a paid data subscription; without one
# Alpaca rejects the request rather than quietly downgrading.
FEEDS = ("yfinance", "iex", "sip")
DEFAULT_FEED = "yfinance"

# The feed to assume where none was recorded: the pre-feed store layout and
# manifest entries written before `feed` existed. Those are all Alpaca IEX --
# it was the downloader's only option at the time -- and must never be read as
# anything else.
LEGACY_FEED = "iex"

# Yahoo serves 1-minute bars for roughly the last 30 calendar days and refuses
# anything older ("The requested range must be within the last 30 days"), so a
# yfinance dataset cannot reach further back than that. Alpaca has no such
# limit, which is why `iex`/`sip` remain the only way to build a dataset over
# an older window.
YF_MINUTE_WINDOW_DAYS = 30

# How long after the stored window closes (20:00 ET) a day's tape is taken as
# final -- the vendors' own lag (Alpaca withholds the last 15 minutes of SIP
# from a free plan) with room to spare. A file written before then is a prefix.
DAY_SETTLE = timedelta(minutes=30)

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
    # One temp name per writer: the app, an experiment runner and a tuning job
    # can all repair the same premature day at once, and a shared temp file
    # would interleave their writes.
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    tmp.replace(path)


def bars_path(symbol: str, day: date, feed: str = DEFAULT_FEED) -> Path:
    """Where a freshly downloaded day is written."""
    return STORE_DIR / "bars" / feed / symbol.upper() / f"{day.isoformat()}.json.gz"


def news_path(symbol: str, day: date) -> Path:
    # No feed: news is the same articles whichever tape the bars came from.
    return STORE_DIR / "news" / symbol.upper() / f"{day.isoformat()}.json.gz"


def daily_path(symbol: str, feed: str = DEFAULT_FEED) -> Path:
    return STORE_DIR / "daily" / feed / f"{symbol.upper()}.json.gz"


def market_path() -> Path:
    return STORE_DIR / "market" / "indicators.json.gz"


def _legacy_bars_path(symbol: str, day: date) -> Path:
    return STORE_DIR / "bars" / symbol.upper() / f"{day.isoformat()}.json.gz"


def _legacy_daily_path(symbol: str) -> Path:
    return STORE_DIR / "daily" / f"{symbol.upper()}.json.gz"


def stored_bars_path(symbol: str, day: date, feed: str = DEFAULT_FEED) -> Path:
    """Where this day actually is on disk, feed-scoped or pre-feed.

    Everything downloaded before the feed was tracked is IEX and sits in the
    old flat layout. Falling back to it (for `iex` only) keeps those datasets
    working without a re-download, and cannot mislabel anything: `sip` and
    `yfinance` never resolve to a file that was fetched as `iex`.
    """
    path = bars_path(symbol, day, feed)
    if not path.exists() and feed == LEGACY_FEED:
        legacy = _legacy_bars_path(symbol, day)
        if legacy.exists():
            return legacy
    return path


def stored_daily_path(symbol: str, feed: str = DEFAULT_FEED) -> Path:
    path = daily_path(symbol, feed)
    if not path.exists() and feed == LEGACY_FEED:
        legacy = _legacy_daily_path(symbol)
        if legacy.exists():
            return legacy
    return path


def load_day_bars(symbol: str, day: date, feed: str = DEFAULT_FEED) -> list[dict]:
    """Stored 1-minute bars for one (symbol, day, feed); [] when the day has no
    session (holiday) or hasn't been downloaded on that feed."""
    path = stored_bars_path(symbol, day, feed)
    if not path.exists():
        return []
    return _read_gz(path)  # type: ignore[return-value]


def load_daily_bars(symbol: str, feed: str = DEFAULT_FEED) -> list[dict]:
    path = stored_daily_path(symbol, feed)
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
# Completeness -- see "A stored day is whole only if it was fetched after it
# ended" in the module docstring
# ---------------------------------------------------------------------------

def day_final_at(day: date) -> datetime:
    """When `day`'s stored window (to 20:00 ET) can be downloaded whole."""
    return datetime.combine(day, DAY_END_ET, tzinfo=MARKET_TZ) + DAY_SETTLE


def news_final_at(day: date) -> datetime:
    """The same for a news file, which holds a UTC day (`fetch_news_day`)."""
    return datetime.combine(day + timedelta(days=1), time(0, 0), tzinfo=timezone.utc) + DAY_SETTLE


def last_final_day(moment: datetime) -> date:
    """The newest day whose tape was whole at `moment`."""
    day = moment.astimezone(MARKET_TZ).date()
    return day if moment >= day_final_at(day) else day - timedelta(days=1)


def _last_weekday(day: date) -> date:
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return day


def written_at(path: Path) -> Optional[datetime]:
    """When `path` was written (`_write_gz` replaces files whole), or None."""
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
    except OSError:
        return None


def _stored_empty(path: Path) -> bool:
    # A gzipped "[]" is a few dozen bytes and a stored session is kilobytes,
    # so only the smallest files are worth opening.
    try:
        if path.stat().st_size > 512:
            return False
        return not _read_gz(path)
    except (OSError, ValueError):
        return False


def _premature(written: datetime) -> str:
    return f"downloaded {written.astimezone(MARKET_TZ):%Y-%m-%d %H:%M} ET, before the day was over"


def incomplete_reason(symbol: str, day: date, feed: str = DEFAULT_FEED) -> Optional[str]:
    """Why the stored (symbol, day, feed) is not the whole day, or None when it
    is -- or is not stored at all, which `stored_bars_path(...).exists()` says."""
    path = stored_bars_path(symbol, day, feed)
    written = written_at(path)
    if written is None:
        return None
    if written < day_final_at(day):
        return _premature(written)
    if _stored_empty(path) and any(
        str(bar.get("t", ""))[:10] == day.isoformat()
        for bar in load_daily_bars(symbol, feed)
    ):
        return "stored empty, but the daily history has a session that day"
    return None


def stored_day_complete(symbol: str, day: date, feed: str = DEFAULT_FEED) -> bool:
    """Whether (symbol, day, feed) is stored and holds the whole day."""
    return (
        stored_bars_path(symbol, day, feed).exists()
        and incomplete_reason(symbol, day, feed) is None
    )


def refetchable(day: date, feed: str = DEFAULT_FEED, now: Optional[datetime] = None) -> bool:
    """Whether downloading `day` now would bring it back whole: it is over, and
    on yfinance still inside Yahoo's 1-minute window."""
    now = now or datetime.now(timezone.utc)
    if now < day_final_at(day):
        return False
    if feed == "yfinance":
        horizon = now.astimezone(MARKET_TZ).date() - timedelta(days=YF_MINUTE_WINDOW_DAYS)
        return day >= horizon
    return True


def _news_incomplete(symbol: str, day: date) -> bool:
    written = written_at(news_path(symbol, day))
    return written is not None and written < news_final_at(day)


def _daily_fetched_at(path: Path, meta: dict) -> Optional[datetime]:
    """When a daily-history file was fetched: recorded since 2026-10-02, the
    file's mtime before that."""
    try:
        return datetime.fromisoformat(meta["fetched_at"])
    except (KeyError, TypeError, ValueError):
        return written_at(path)


def _daily_whole_through(path: Path, meta: dict) -> date:
    """The last day a daily-history file holds whole: its requested end, unless
    it was fetched before that day was over (its row is then a partial bar)."""
    end = date.fromisoformat(meta["end"])
    fetched = _daily_fetched_at(path, meta)
    return min(end, last_final_day(fetched)) if fetched is not None else end


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
    # Which tape these bars are. Defaulted rather than required so manifest
    # entries written before feeds were tracked still load -- and `iex` is the
    # honest value for them, since it was the only feed the downloader used.
    feed: str = LEGACY_FEED

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


def _fetch_minute_bars_day_alpaca(
    symbol: str, day: date, key: str, secret: str, feed: str
) -> list[dict]:
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


def _fetch_daily_bars_range_alpaca(
    symbol: str, start: date, end: date, key: str, secret: str, feed: str
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


# ---------------------------------------------------------------------------
# yfinance fetchers (the default source for new datasets)
#
# yfinance hands back a DataFrame; the store speaks Alpaca's bar dicts, so
# everything here converts. Two differences from an Alpaca day are permanent
# and worth knowing before comparing runs across feeds:
#
# - No `vw`. Yahoo publishes no per-bar VWAP, so yfinance bars omit the key
#   rather than carry a fabricated one. `technical_analysis.analyze_intraday`
#   and `charts` already branch on its presence, so the VWAP line/note is
#   simply absent on a yfinance run instead of being wrong.
# - Zero volume outside 09:30-16:00. `prepost=True` is required to reach the
#   stored 04:00-20:00 window, and Yahoo does return real, moving prices for
#   those minutes -- but reports every one of them at volume 0. The bars are
#   kept: their prices are what a pre-market read or an opening tactic acts on,
#   and dropping them would cut the stored day down to the regular session.
#   The consequence is that volume-threshold rules never fire outside regular
#   hours on this feed, where on Alpaca they can.
# ---------------------------------------------------------------------------

def _num(value: object, default: float = 0.0) -> float:
    """float(value), with NaN/None/junk collapsing to `default` -- `x or 0.0`
    does not, since NaN is truthy."""
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return default if out != out else out


def _yf_frame(symbol: str, start: date, end: date, interval: str, prepost: bool):
    """yfinance download, flattened to single-level columns. `end` exclusive."""
    import yfinance as yf

    frame = yf.download(
        symbol,
        start=start.isoformat(),
        end=end.isoformat(),
        interval=interval,
        auto_adjust=False,
        prepost=prepost,
        progress=False,
    )
    if frame is None or frame.empty:
        return None
    if hasattr(frame.columns, "get_level_values") and frame.columns.nlevels > 1:
        frame.columns = frame.columns.get_level_values(0)
    return frame


def _fetch_minute_bars_day_yfinance(symbol: str, day: date) -> list[dict]:
    """All 1-minute bars for one trading day from yfinance, 04:00-20:00 ET.

    Returns [] for a non-trading day and for any day outside Yahoo's 30-day
    1-minute window -- the same "nothing stored for this day" signal a holiday
    produces, which `create_dataset` already handles.
    """
    frame = _yf_frame(symbol, day, day + timedelta(days=1), "1m", prepost=True)
    if frame is None:
        return []
    index = frame.index
    if index.tz is None:
        index = index.tz_localize(MARKET_TZ)
    index = index.tz_convert(timezone.utc)

    bars: list[dict] = []
    for ts, row in zip(index, frame.itertuples(index=False)):
        if row.Close != row.Close:  # NaN: no price for this minute at all
            continue
        # Yahoo can serve a few minutes either side of the requested day around
        # DST changes; the store's contract is one ET day per file.
        if ts.astimezone(MARKET_TZ).date() != day:
            continue
        bars.append(
            {
                "t": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "o": float(row.Open),
                "h": float(row.High),
                "l": float(row.Low),
                "c": float(row.Close),
                "v": _num(row.Volume),
            }
        )
    return bars


def _fetch_daily_bars_range_yfinance(symbol: str, start: date, end: date) -> list[dict]:
    """Daily bars over [start, end] inclusive, in the stored bar shape.

    Sourced from yfinance so a yfinance dataset's daily baselines (ADV, rvol,
    prev close) are measured on the same tape as its minute bars.
    """
    frame = _yf_frame(symbol, start, end + timedelta(days=1), "1d", prepost=False)
    if frame is None:
        return []
    return [
        {
            # Alpaca stamps a daily bar at the session open in UTC; the store
            # only ever reads the date part, and `SimMarket.daily_bars_at`
            # builds today's partial bar with the same 05:00Z stamp.
            "t": f"{ts.date().isoformat()}T05:00:00Z",
            "o": float(row.Open),
            "h": float(row.High),
            "l": float(row.Low),
            "c": float(row.Close),
            "v": _num(row.Volume),
        }
        for ts, row in zip(frame.index, frame.itertuples(index=False))
        if row.Close == row.Close  # drop NaN rows (non-trading days)
    ]


# ---------------------------------------------------------------------------
# Feed dispatch -- the one place that decides which vendor a feed means
# ---------------------------------------------------------------------------

def fetch_minute_bars_day(
    symbol: str, day: date, key: str = "", secret: str = "", feed: str = DEFAULT_FEED
) -> list[dict]:
    """All 1-minute bars for one trading day, 04:00-20:00 ET, from `feed`.

    `key`/`secret` are only used by the Alpaca feeds; the yfinance default
    needs no credentials.
    """
    if feed == "yfinance":
        return _fetch_minute_bars_day_yfinance(symbol, day)
    return _fetch_minute_bars_day_alpaca(symbol, day, key, secret, feed)


def fetch_daily_bars_range(
    symbol: str, start: date, end: date, key: str = "", secret: str = "",
    feed: str = DEFAULT_FEED,
) -> list[dict]:
    if feed == "yfinance":
        return _fetch_daily_bars_range_yfinance(symbol, start, end)
    return _fetch_daily_bars_range_alpaca(symbol, start, end, key, secret, feed)


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


def coverage(
    symbols: list[str], start: date, end: date, feed: str = DEFAULT_FEED
) -> dict[str, dict[str, bool]]:
    """{day -> {symbol -> already stored on this feed}} for every weekday."""
    return {
        day.isoformat(): {
            sym: stored_bars_path(sym, day, feed).exists() for sym in symbols
        }
        for day in weekdays(start, end)
    }


def _daily_covers(symbol: str, start: date, end: date, feed: str = DEFAULT_FEED) -> bool:
    """Whether the stored daily-bar file spans [start - lookback, end]."""
    path = stored_daily_path(symbol, feed)
    if not path.exists():
        return False
    meta = _read_gz(path)
    try:
        have_start = date.fromisoformat(meta["start"])
        whole_through = _daily_whole_through(path, meta)
    except (KeyError, ValueError, TypeError):
        return False
    return (
        have_start <= start - timedelta(days=DAILY_LOOKBACK_DAYS)
        and whole_through >= _last_weekday(end)
    )


def _market_covers(start: date, end: date) -> bool:
    data = load_market_indicators()
    spy = data.get("spy") or []
    if not spy:
        return False
    dates = [row["date"] for row in spy]
    fetched = written_at(market_path())
    return (
        dates[0] <= (start - timedelta(days=DAILY_LOOKBACK_DAYS)).isoformat()
        and dates[-1] >= (end - timedelta(days=4)).isoformat()
        # Fetched mid-session, the last close is a price, not a close.
        and fetched is not None and last_final_day(fetched) >= _last_weekday(end)
    )


def _needs_download(symbol: str, day: date, feed: str, progress: ProgressCb) -> bool:
    """Whether `create_dataset` should fetch this day: not stored, or stored but
    not whole and still within reach."""
    if not stored_bars_path(symbol, day, feed).exists():
        return True
    reason = incomplete_reason(symbol, day, feed)
    if reason is None:
        return False
    if feed == "yfinance" and day < date.today() - timedelta(days=YF_MINUTE_WINDOW_DAYS):
        progress(
            f"minute bars {symbol} {day} [{feed}]: incomplete ({reason}) and Yahoo "
            "no longer serves that day; keeping what is stored"
        )
        return False
    progress(f"minute bars {symbol} {day} [{feed}]: incomplete ({reason}); re-downloading")
    return True


def _store_day_bars(
    symbol: str, day: date, feed: str, bars: list[dict], progress: ProgressCb
) -> list[dict]:
    """Store a downloaded day -- unless the copy on disk has more bars, which a
    whole day never does: then the download failed or Yahoo has let the day
    go, and the stored part is worth more than none. Returns what is stored."""
    path = stored_bars_path(symbol, day, feed)
    if path.exists():
        kept = _read_gz(path)
        if len(bars) < len(kept):  # type: ignore[arg-type]
            progress(
                f"minute bars {symbol} {day} [{feed}]: the download has {len(bars)} "
                f"bar(s), fewer than the {len(kept)} stored; kept the stored copy"  # type: ignore[arg-type]
            )
            return kept  # type: ignore[return-value]
    _write_gz(bars_path(symbol, day, feed), bars)
    return bars


def _store_news(symbol: str, day: date, key: str, secret: str, progress: ProgressCb) -> None:
    """Fetch and store one day's news. News is nice-to-have and never fatal,
    and a failure never replaces articles already stored."""
    npath = news_path(symbol, day)
    if not (key and secret):
        # News is Alpaca-only; a credential-free yfinance dataset simply has
        # none rather than failing the download.
        if not npath.exists():
            _write_gz(npath, [])
        return
    try:
        articles = fetch_news_day(symbol, day, key, secret)
    except Exception as exc:
        if npath.exists():
            progress(f"news {symbol} {day}: failed ({exc}); kept the stored articles")
        else:
            progress(f"news {symbol} {day}: failed ({exc}); storing empty")
            _write_gz(npath, [])
        return
    if npath.exists() and len(articles) < len(load_news(symbol, day)):
        return
    _write_gz(npath, articles)


def _store_daily(
    symbol: str, start: date, end: date, key: str, secret: str, feed: str,
    progress: ProgressCb = _noop_progress,
) -> bool:
    """Download and store a symbol's daily history over [start, end], with the
    moment it was fetched -- a row for a day still trading is a partial bar,
    and `_daily_whole_through` needs to know which rows those are. An empty
    download never replaces a stored history. Returns whether it wrote."""
    fetched = datetime.now(timezone.utc)
    bars = fetch_daily_bars_range(symbol, start, end, key, secret, feed)
    if not bars and load_daily_bars(symbol, feed):
        progress(f"daily bars {symbol} [{feed}]: the download came back empty; kept the stored history")
        return False
    _write_gz(
        daily_path(symbol, feed),
        {"symbol": symbol, "start": start.isoformat(), "end": end.isoformat(),
         "feed": feed, "fetched_at": fetched.isoformat(), "bars": bars},
    )
    return True


def download_days(
    symbols: list[str],
    start: date,
    end: date,
    key: str = "",
    secret: str = "",
    feed: str = DEFAULT_FEED,
    progress: ProgressCb = _noop_progress,
) -> list[str]:
    """Fill the store with everything a replay of [start, end] reads -- the
    daily history, the market indicators, the week before the first session
    and every day's bars and news -- downloading only what is missing or not
    whole. Returns the session days (ISO) that have bars.

    `create_dataset` without the manifest entry: the live app's replay of a
    past session (`agent_stonks.replay`) reads the same store, and has no
    business adding a dataset to SimLab's list every time it is started.
    Arguments are taken as validated (`create_dataset` checks them).
    """
    # Daily bars first: they double as the trading-day calendar for the range.
    daily_start = start - timedelta(days=DAILY_LOOKBACK_DAYS)
    for sym in symbols:
        if _daily_covers(sym, start, end, feed):
            progress(f"daily bars {sym} [{feed}]: already stored")
            continue
        progress(f"daily bars {sym} [{feed}]: downloading {daily_start} … {end}")
        _store_daily(sym, daily_start, end, key, secret, feed, progress)

    if _market_covers(start, end):
        progress("market indicators (SPY/VIX/VIX3M): already stored")
    else:
        progress("market indicators (SPY/VIX/VIX3M): downloading")
        _write_gz(market_path(), fetch_market_indicator_closes(daily_start, end))

    # The week before the first session, bars only: what a replay measures
    # `abs_mean_minute_momentum` from (`SimMarket.abs_mean_minute_momentum`),
    # so the dataset's first days are read like the rest. Not part of the
    # dataset's days, and a failure costs only that measure, never the dataset.
    for day in minute_momentum.prior_week_days(start):
        for sym in symbols:
            if not _needs_download(sym, day, feed, progress):
                continue
            progress(f"minute bars {sym} {day} [{feed}]: downloading (the week before)")
            try:
                _store_day_bars(sym, day, feed, fetch_minute_bars_day(sym, day, key, secret, feed), progress)
            except Exception as exc:
                progress(f"minute bars {sym} {day}: failed ({exc}); skipped")

    session_days: list[str] = []
    for day in weekdays(start, end):
        day_has_bars = False
        for sym in symbols:
            # The *resolved* path, so a day already held in the pre-feed layout
            # counts as stored instead of being downloaded a second time -- and
            # only when it is the whole day (`incomplete_reason`).
            downloaded = _needs_download(sym, day, feed, progress)
            if downloaded:
                progress(f"minute bars {sym} {day} [{feed}]: downloading")
                bars = _store_day_bars(
                    sym, day, feed, fetch_minute_bars_day(sym, day, key, secret, feed), progress
                )
                day_has_bars = day_has_bars or bool(bars)
            else:
                day_has_bars = day_has_bars or bool(load_day_bars(sym, day, feed))
            if (downloaded and not news_path(sym, day).exists()) or _news_incomplete(sym, day):
                _store_news(sym, day, key, secret, progress)
        if not day_has_bars:
            # Existing empty files (holiday) or fresh empty downloads.
            day_has_bars = any(load_day_bars(sym, day, feed) for sym in symbols)
        if day_has_bars:
            session_days.append(day.isoformat())

    return session_days


def create_dataset(
    name: str,
    symbols: list[str],
    start: date,
    end: date,
    key: str = "",
    secret: str = "",
    feed: str = DEFAULT_FEED,
    progress: ProgressCb = _noop_progress,
) -> Dataset:
    """Create (or refresh) a named dataset, downloading only what the store
    is missing. Returns the manifest entry with its resolved trading days.

    `key`/`secret` are Alpaca credentials. On the default yfinance feed they
    are only needed for news, which is skipped (stored empty) without them; the
    Alpaca feeds require them for bars.
    """
    symbols = [s.strip().upper() for s in symbols if s.strip()]
    if not symbols:
        raise ValueError("dataset needs at least one symbol")
    if end < start:
        raise ValueError("dataset end date is before its start date")
    if feed not in FEEDS:
        raise ValueError(f"unknown feed {feed!r}; expected one of {', '.join(FEEDS)}")
    if feed != "yfinance" and not (key and secret):
        raise ValueError(f"the {feed!r} feed needs Alpaca credentials")

    # Yahoo's 1-minute history stops ~30 days back. Say so up front rather than
    # letting the run finish with a handful of silently empty days: those days
    # are indistinguishable from holidays once stored.
    if feed == "yfinance":
        horizon = date.today() - timedelta(days=YF_MINUTE_WINDOW_DAYS)
        if start < horizon:
            progress(
                f"warning: yfinance serves 1-minute bars only back to {horizon} "
                f"({YF_MINUTE_WINDOW_DAYS} days); days before that will store empty. "
                "Use the 'sip' feed (paid Alpaca data) for an older window."
            )

    session_days = download_days(symbols, start, end, key, secret, feed, progress)

    dataset = Dataset(
        name=name,
        symbols=symbols,
        start=start.isoformat(),
        end=end.isoformat(),
        created_at=datetime.now(timezone.utc).isoformat(),
        days=session_days,
        feed=feed,
    )
    with _manifest_lock:
        existing = [d for d in list_datasets() if d.name != name]
        _save_manifest([*existing, dataset])
    progress(f"dataset '{name}' ready: {len(session_days)} trading day(s)")
    return dataset


# ---------------------------------------------------------------------------
# Repairing what was stored before it was whole
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IncompleteFile:
    """One stored file that is not the whole of what it names."""

    kind: str  # "bars" | "news" | "daily" | "market"
    path: Path
    feed: str  # "" for news and the market indicators
    symbol: str  # "" for the market indicators
    day: Optional[date]  # the day that is not whole
    reason: str
    fixable: bool  # a download now would bring it back whole

    def label(self) -> str:
        if self.kind == "bars":
            return f"minute bars {self.symbol} {self.day} [{self.feed}]"
        if self.kind == "news":
            return f"news {self.symbol} {self.day}"
        if self.kind == "daily":
            return f"daily bars {self.symbol} [{self.feed}]"
        return "market indicators (SPY/VIX/VIX3M)"


def _stored_bar_days(feed: Optional[str] = None) -> "set[tuple[str, str, date]]":
    """(feed, symbol, day) for every stored minute-bar file, in both layouts."""
    root = STORE_DIR / "bars"
    found: "set[tuple[str, str, date]]" = set()
    if not root.exists():
        return found
    for top in root.iterdir():
        if not top.is_dir():
            continue
        if top.name in FEEDS:
            dirs = [(top.name, sub) for sub in top.iterdir() if sub.is_dir()]
        else:  # the pre-feed flat layout, bars/{SYMBOL}/ -- all IEX
            dirs = [(LEGACY_FEED, top)]
        for tape, sym_dir in dirs:
            if feed and tape != feed:
                continue
            for path in sym_dir.glob("*.json.gz"):
                try:
                    found.add((tape, sym_dir.name.upper(), date.fromisoformat(path.name[:10])))
                except ValueError:
                    continue
    return found


def incomplete_files(
    symbols: "Optional[Iterable[str]]" = None,
    days: "Optional[Iterable[date]]" = None,
    feed: Optional[str] = None,
    now: Optional[datetime] = None,
) -> "list[IncompleteFile]":
    """Every stored file that is not whole, narrowed to `symbols` / `days` /
    `feed` when given. The daily histories and the market indicators are not
    per day, so `days` does not narrow them."""
    now = now or datetime.now(timezone.utc)
    want_symbols = {s.upper() for s in symbols} if symbols is not None else None
    want_days = set(days) if days is not None else None
    found: "list[IncompleteFile]" = []

    for tape, symbol, day in sorted(_stored_bar_days(feed)):
        if want_symbols is not None and symbol not in want_symbols:
            continue
        if want_days is not None and day not in want_days:
            continue
        reason = incomplete_reason(symbol, day, tape)
        if reason:
            found.append(IncompleteFile(
                "bars", stored_bars_path(symbol, day, tape), tape, symbol, day,
                reason, refetchable(day, tape, now),
            ))

    news_root = STORE_DIR / "news"
    for sym_dir in sorted(news_root.iterdir()) if news_root.exists() else ():
        symbol = sym_dir.name.upper()
        if want_symbols is not None and symbol not in want_symbols:
            continue
        for path in sorted(sym_dir.glob("*.json.gz")):
            try:
                day = date.fromisoformat(path.name[:10])
            except ValueError:
                continue
            if want_days is not None and day not in want_days:
                continue
            written = written_at(path)
            if written is not None and written < news_final_at(day):
                found.append(IncompleteFile(
                    "news", path, "", symbol, day, _premature(written), now >= news_final_at(day),
                ))

    daily_root = STORE_DIR / "daily"
    daily_files = [(tape, p) for tape in FEEDS for p in sorted((daily_root / tape).glob("*.json.gz"))]
    daily_files += [(LEGACY_FEED, p) for p in sorted(daily_root.glob("*.json.gz"))]
    for tape, path in daily_files:
        symbol = path.name[: -len(".json.gz")].upper()
        if (feed and tape != feed) or (want_symbols is not None and symbol not in want_symbols):
            continue
        if path != stored_daily_path(symbol, tape):
            continue  # a pre-feed file shadowed by a feed-scoped one
        try:
            meta = _read_gz(path)
            last = date.fromisoformat(str(meta["bars"][-1]["t"])[:10])  # type: ignore[index]
        except (OSError, ValueError, KeyError, IndexError, TypeError):
            continue
        fetched = _daily_fetched_at(path, meta)  # type: ignore[arg-type]
        if fetched is not None and fetched < day_final_at(last):
            found.append(IncompleteFile(
                "daily", path, tape, symbol, last,
                f"its {last} bar is partial ({_premature(fetched)})", now >= day_final_at(last),
            ))

    spy = load_market_indicators().get("spy") or []
    written = written_at(market_path())
    if spy and written is not None:
        last = date.fromisoformat(spy[-1]["date"])
        if written < day_final_at(last):
            found.append(IncompleteFile(
                "market", market_path(), "", "", last,
                f"its {last} close is a price, not a close ({_premature(written)})",
                now >= day_final_at(last),
            ))
    return found


def _repair(item: IncompleteFile, key: str, secret: str, progress: ProgressCb) -> bool:
    """Re-download one incomplete file; whether it now holds more than before."""
    if item.kind == "bars":
        assert item.day is not None
        fresh = fetch_minute_bars_day(item.symbol, item.day, key, secret, item.feed)
        return _store_day_bars(item.symbol, item.day, item.feed, fresh, progress) is fresh
    if item.kind == "news":
        assert item.day is not None
        _store_news(item.symbol, item.day, key, secret, progress)
        return not _news_incomplete(item.symbol, item.day)
    if item.kind == "daily":
        meta = _read_gz(item.path)
        return _store_daily(
            item.symbol, date.fromisoformat(meta["start"]), date.fromisoformat(meta["end"]),  # type: ignore[index]
            key, secret, item.feed, progress,
        )
    stored = load_market_indicators()
    spy = stored.get("spy") or []
    fresh = fetch_market_indicator_closes(
        date.fromisoformat(spy[0]["date"]), date.fromisoformat(spy[-1]["date"])
    )
    # The fetch swallows a failure per series as []; never trade a stored
    # series for an empty one.
    if any(rows and not fresh.get(name) for name, rows in stored.items()):
        progress("market indicators: a series came back empty; kept the stored closes")
        return False
    _write_gz(market_path(), fresh)
    return True


def repair_incomplete(
    key: str = "",
    secret: str = "",
    progress: ProgressCb = _noop_progress,
    symbols: "Optional[Iterable[str]]" = None,
    days: "Optional[Iterable[date]]" = None,
    feed: Optional[str] = None,
) -> "list[IncompleteFile]":
    """Re-download every incomplete stored file a download can now make whole
    (`incomplete_files`, narrowed the same way). Returns the ones repaired.

    Alpaca credentials fall back to the environment (news and the `iex`/`sip`
    tapes need them). Nothing here is fatal: a failed download keeps what is
    stored, and is tried again on the next call.
    """
    key = key or os.getenv("ALPACA_API_KEY", "")
    secret = secret or os.getenv("ALPACA_SECRET", "")
    repaired: "list[IncompleteFile]" = []
    for item in incomplete_files(symbols, days, feed):
        if not item.fixable:
            continue
        alpaca = item.kind == "news" or (item.kind in ("bars", "daily") and item.feed != "yfinance")
        if alpaca and not (key and secret):
            progress(f"{item.label()}: incomplete ({item.reason}); no Alpaca credentials to re-download it")
            continue
        progress(f"{item.label()}: incomplete ({item.reason}); re-downloading")
        try:
            if _repair(item, key, secret, progress):
                repaired.append(item)
        except Exception as exc:
            progress(f"{item.label()}: re-download failed ({exc}); kept as stored")
    return repaired


def repair_for_replay(
    symbols: "Iterable[str]", days: "Iterable[date]", feed: str,
    progress: ProgressCb = _noop_progress,
) -> None:
    """`repair_incomplete` for what one replay reads: its symbols' sessions and
    the week before each (`abs_mean_minute_momentum`), their daily histories
    and the market indicators. Never raises -- the replay goes ahead on what
    is stored."""
    days = list(days)
    wanted = set(days) | {prior for day in days for prior in minute_momentum.prior_week_days(day)}
    try:
        repair_incomplete(progress=progress, symbols=symbols, days=wanted, feed=feed)
    except Exception as exc:
        progress(f"Checking the stored data for incomplete days failed ({exc}); replaying what is stored.")


def reconcile_dataset_days() -> "list[str]":
    """Add to each dataset's `days` any weekday in its range that now has bars.

    A day stored before it had begun was stored empty and left out as a
    holiday; once repaired it is a session the dataset should offer. Returns
    the names of the datasets that changed.
    """
    with _manifest_lock:
        datasets = list_datasets()
        changed: "list[str]" = []
        for ds in datasets:
            have = set(ds.days)
            lo, hi = ds.date_range()
            gained = {
                day.isoformat() for day in weekdays(lo, hi)
                if day.isoformat() not in have
                and any(load_day_bars(sym, day, ds.feed) for sym in ds.symbols)
            }
            if gained:
                ds.days = sorted(have | gained)
                changed.append(ds.name)
        if changed:
            _save_manifest(datasets)
    return changed


def store_size_bytes() -> int:
    if not STORE_DIR.exists():
        return 0
    return sum(p.stat().st_size for p in STORE_DIR.rglob("*") if p.is_file())
