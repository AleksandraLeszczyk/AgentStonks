"""How far a streamed ticker's price *usually* moves in one minute.

`abs_mean_minute_momentum` is the mean absolute one-minute close-to-close
change, in dollars, over the last trading week -- the 1-bar case of the
`close - close[N bars ago]` momentum the live chart's panel draws. It puts a
number on "is this minute's move big for this ticker?": 0.12 means AAPL's
typical minute moves twelve cents either way.

It is measured once per ET day per ticker, on a streaming start. The window is
made of completed sessions only, so it cannot change during the day: the first
start of the day computes it and writes it under `data/minute_momentum/`, and
every later start that day (a restart, a timeframe reload, a second Streamlit
session) reads that file back instead of downloading a week of minute bars
again.

The history is the same yfinance week the volume baseline uses
(`volume_baseline.WEEK_SESSIONS` sessions, regular hours, consolidated), so the
two references describe the same five days.
"""

from __future__ import annotations

import json
import threading
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from . import historical
from .datalog import log_fetch, log_fetch_failure
from .market_hours import MARKET_TZ
from .volume_baseline import (
    WEEK_LOOKBACK_DAYS,
    WEEK_SESSIONS,
    _bar_session_date,
    _minute_of_day,
    _session_dates,
)

# One JSON file per symbol, overwritten by the first start of each day.
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "minute_momentum"

# The regular session in minutes past ET midnight, [09:30, 16:00): the bars a
# minute's move is measured over. The live history is regular-hours already;
# SimLab's stored days carry pre- and post-market minutes too, whose thin
# prints would move the mean.
_RTH_MINUTES = (9 * 60 + 30, 16 * 60)


def prior_week_days(day: date) -> "list[date]":
    """The weekdays whose sessions can make up the week before `day`: the
    `WEEK_LOOKBACK_DAYS` calendar days the live read fetches, oldest first."""
    return [
        d
        for d in (day - timedelta(days=k) for k in range(WEEK_LOOKBACK_DAYS, 0, -1))
        if d.weekday() < 5
    ]


def _close(bar: dict) -> "float | None":
    try:
        close = float(bar.get("c"))
    except (TypeError, ValueError):
        return None
    return None if close != close else close


def compute(history_bars: "list[dict]", today: str) -> "dict | None":
    """The mean absolute 1-minute close change over the week before `today`.

    `history_bars` are minute bars (oldest first or not -- they are sorted
    here) spanning enough calendar days to hold `WEEK_SESSIONS` sessions;
    `today` is the ET date to measure back from, and its own bars are ignored.

    Changes are taken between consecutive bars *of the same session*: the
    overnight gap from one close to the next open is not a minute's move and
    would dominate the mean on any gap day.

    Returns ``{"abs_mean_minute_momentum", "changes", "sessions", "dates"}``,
    or None when no prior session carries two bars.
    """
    prior = [
        bar
        for bar in history_bars or []
        if (_bar_session_date(bar) or "") < today
        and _RTH_MINUTES[0] <= (_minute_of_day(bar) or -1) < _RTH_MINUTES[1]
    ]
    dates = _session_dates(prior)[-WEEK_SESSIONS:]
    chosen = set(dates)
    by_session: dict[str, list[dict]] = {}
    for bar in prior:
        session = _bar_session_date(bar)
        if session in chosen:
            by_session.setdefault(session, []).append(bar)

    total = 0.0
    count = 0
    for bars in by_session.values():
        bars.sort(key=lambda bar: str(bar.get("t")))
        closes = [c for c in (_close(bar) for bar in bars) if c is not None]
        for prev, cur in zip(closes, closes[1:]):
            total += abs(cur - prev)
            count += 1
    if not count:
        return None
    return {
        "abs_mean_minute_momentum": total / count,
        "changes": count,
        "sessions": len(dates),
        "dates": dates,
    }


def _today_et() -> str:
    return datetime.now(timezone.utc).astimezone(MARKET_TZ).strftime("%Y-%m-%d")


def _cache_path(symbol: str) -> Path:
    return CACHE_DIR / f"{symbol.upper()}.json"


def _read_cached(symbol: str, today: str) -> "dict | None":
    """Today's stored result for `symbol`, or None if it is missing or stale."""
    try:
        record = json.loads(_cache_path(symbol).read_text())
    except (OSError, ValueError):
        return None
    if record.get("computed_on") != today:
        return None
    if not isinstance(record.get("abs_mean_minute_momentum"), (int, float)):
        return None
    return record


def _write_cached(symbol: str, record: dict) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _cache_path(symbol).write_text(json.dumps(record, indent=2))
    except OSError:
        # A read-only checkout still gets the value for this run; the next
        # start just computes it again.
        pass


def load_or_compute(symbol: str, today: "str | None" = None) -> "dict | None":
    """`symbol`'s record for `today` (ET), computing and storing it at most once.

    None when the week's history cannot be fetched -- nothing is stored then,
    so the next start tries again rather than keeping a gap for the day.
    """
    today = today or _today_et()
    cached = _read_cached(symbol, today)
    if cached is not None:
        return cached
    bars = historical.fetch_intraday_history_bars(symbol, WEEK_LOOKBACK_DAYS)
    result = compute(bars, today)
    if result is None:
        log_fetch_failure(
            "abs mean minute momentum", [("yfinance", "no prior-week minute bars")],
            symbol=symbol, consequence="left unset until the next start",
        )
        return None
    record = {"symbol": symbol.upper(), "computed_on": today, **result}
    _write_cached(symbol, record)
    log_fetch(
        "abs mean minute momentum", "yfinance", symbol=symbol,
        detail=f"${result['abs_mean_minute_momentum']:.4f} over "
        f"{result['sessions']} sessions ({result['changes']} changes)",
    )
    return record


def refresh(state) -> "float | None":
    """Set `state.abs_mean_minute_momentum` (a `SymbolState`) and return it."""
    try:
        record = load_or_compute(state.symbol)
    except Exception as exc:
        log_fetch_failure(
            "abs mean minute momentum", [("yfinance", exc)],
            symbol=state.symbol, consequence="left unset until the next start",
        )
        return None
    value = record["abs_mean_minute_momentum"] if record else None
    if value is not None:
        state.abs_mean_minute_momentum = value
    return value


def launch_refresh(states) -> None:
    """Refresh every state in `states` on one background thread.

    Off the start path because a first start of the day downloads a week of
    minute bars per ticker, and the stream must not wait for that.
    """
    states = list(states)
    if states:
        threading.Thread(
            target=lambda: [refresh(s) for s in states], daemon=True
        ).start()
