"""The live chart's net gamma panel, one value per bar, taken once.

The panel shows the options chain's net dealer gamma at each bar's close
(`options.net_gamma_exposure`). It used to re-price *every* bar with the latest
chain on every redraw, so a bar already in the past moved whenever the chain
was refetched (new IVs, new open interest, about once a minute): the 13:40 bar
read one value at 13:41 and another at 13:45. A bar's value is now taken once,
at its close, with the chain the app had at that moment, and kept.

- **When a bar is taken.** Once it is complete: a later bar exists, or its
  period has ended. The bar still forming is re-priced on every redraw at its
  current close, so it agrees with the Net Gamma card, and is taken when it
  closes.
- **Bars that closed before there was a chain** (a start mid-day, or the first
  seconds before yfinance answers) are priced with the first chain that
  arrives -- the earliest the app can -- and kept from then on.
- **A later revision of a bar is not followed.** The bar tiers replace a
  provisional close with the consolidated one minutes later
  (`bar_history`); the value stays the one taken at the close.
- **Kept for the process and on disk.** Every Streamlit session reads the same
  values, and they are written under ``data/net_gamma/`` (one file per symbol,
  timeframe and ET day), so a browser reload or an app restart draws the bars
  it already had rather than re-pricing them with a newer chain.
- **Only the latest bar's ET day.** Earlier days' bars are not drawn, and
  today's chain says nothing about them.

Apple Trader's "no buy on a falling, negative-gamma day" reads the same values,
one bar at a time (`at_close`): a bar the panel already took is read back, and
one it has not is taken by the trader, by the same rule. A SimLab replay or
the live app's replay of a past session has no chain of that day, only what
the live app kept then. The chain itself is refreshed here (`refresh_chain`)
for the panel and the trader alike, so a trader running with no page open
still reads a current one.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import OPTIONS_POLL_SEC
from .market_hours import MARKET_TZ
from .options import fetch_options_walls_data, net_gamma_exposure
from .stream_common import TF_MINUTES

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "net_gamma"

_lock = threading.Lock()
# (symbol, timeframe, ET date) -> {bar time, UTC ISO: net gamma at its close}.
_kept: dict[tuple[str, str, str], dict[str, float]] = {}

# The background chain refresh (`refresh_chain`): one fetch at a time per
# symbol, at most once per OPTIONS_POLL_SEC.
_fetch_lock = threading.Lock()
_fetch_running: set[str] = set()
_fetch_tried: dict[str, float] = {}


def _utc(t) -> pd.Timestamp:
    stamp = pd.Timestamp(t)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def kept_path(symbol: str, timeframe: str, day: str) -> Path:
    """The file one symbol, timeframe and ET day's kept values live in."""
    return CACHE_DIR / f"{symbol}_{timeframe}_{day}.json"


def _load(symbol: str, timeframe: str, day: str) -> dict[str, float]:
    path = kept_path(symbol, timeframe, day)
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return {str(t): float(v) for t, v in raw.items()}
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        logger.warning("Kept net gamma %s not read: %s", path, exc)
        return {}


def _save(symbol: str, timeframe: str, day: str, values: dict[str, float]) -> None:
    path = kept_path(symbol, timeframe, day)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(values), encoding="utf-8")
        os.replace(tmp, path)
    except OSError as exc:  # the panel still draws from memory
        logger.warning("Kept net gamma %s not written: %s", path, exc)


def _day_values(key: "tuple[str, str, str]") -> dict[str, float]:
    """The values kept for (symbol, timeframe, ET date), read from disk the
    first time. Called under `_lock`."""
    kept = _kept.get(key)
    if kept is None:
        # A new day: the symbol's earlier days are not drawn any more.
        for old in [k for k in _kept if k[:2] == key[:2]]:
            del _kept[old]
        kept = _kept[key] = _load(*key)
    return kept


def series(
    symbol: str,
    timeframe: str,
    bars: "list[dict]",
    chain: "dict | None",
    now: "datetime | None" = None,
) -> dict:
    """The net gamma panel for `bars`: ``{"t", "value", "note"}``.

    Complete bars of the latest bar's ET day get the value kept for them, or
    are priced now with `chain` and kept; the bar still forming is priced with
    `chain` and not kept. A bar with no kept value while there is no usable
    chain is left out, and `note` says why when nothing is left."""
    if not bars:
        return {"t": [], "value": [], "note": "No bars yet"}
    now = now or datetime.now(timezone.utc)
    period = pd.Timedelta(minutes=TF_MINUTES.get(timeframe, 1))
    stamps = [_utc(b["t"]) for b in bars]
    day = stamps[-1].tz_convert(MARKET_TZ).date().isoformat()
    first = next(
        i for i, s in enumerate(stamps) if s.tz_convert(MARKET_TZ).date().isoformat() == day
    )
    todays = list(zip(stamps[first:], bars[first:]))
    last = len(todays) - 1

    def complete(i: int) -> bool:
        return i < last or todays[i][0] + period <= now

    key = (symbol, timeframe, day)
    with _lock:
        kept = _day_values(key)
        # Every bar without a kept value, and the one still forming: the
        # complete ones are kept, the forming one is only drawn.
        todo = [
            (i, stamp.isoformat(), float(bar["c"]))
            for i, (stamp, bar) in enumerate(todays)
            if not complete(i) or stamp.isoformat() not in kept
        ]
        priced: dict[str, float] = {}
        values = net_gamma_exposure(chain, [c for _, _, c in todo]) if todo else None
        if values is not None:
            added = False
            for (i, iso, _), value in zip(todo, values.tolist()):
                priced[iso] = value
                if complete(i):
                    kept[iso] = value
                    added = True
            if added:
                _save(symbol, timeframe, day, kept)
        drawn = {**kept, **priced}

    t, value = [], []
    for stamp, bar in todays:
        iso = stamp.isoformat()
        if iso in drawn:
            t.append(bar["t"])
            value.append(drawn[iso])
    if value:
        note = ""
    elif not chain or not chain.get("strikes"):
        note = "Waiting for the options chain (yfinance)"
    else:
        note = "Waiting for the next options chain refresh (within a minute)"
    return {"t": t, "value": value, "note": note}


def at_close(
    symbol: str,
    timeframe: str,
    t,
    close: float,
    chain: "dict | None",
) -> "float | None":
    """The net gamma of the complete bar that opened at `t`: the value kept
    for it, else priced now at its `close` with `chain` and kept -- what the
    panel would take for it -- else None, with neither a kept value nor a
    usable chain."""
    stamp = _utc(t)
    day = stamp.tz_convert(MARKET_TZ).date().isoformat()
    iso = stamp.isoformat()
    with _lock:
        kept = _day_values((symbol, timeframe, day))
        if iso in kept:
            return kept[iso]
        values = net_gamma_exposure(chain, [close]) if chain else None
        if values is None:
            return None
        kept[iso] = float(values[0])
        _save(symbol, timeframe, day, kept)
        return kept[iso]


def refresh_chain(sym_state) -> None:
    """Refresh `sym_state.options_chain` in a background thread, at most once
    per OPTIONS_POLL_SEC per symbol.

    The chart fragment re-runs every few seconds and a yfinance chain fetch
    takes a second or two, so fetching inline would stall the live chart. A
    failure is logged by `fetch_option_chain` and simply retried on the next
    interval; the chart keeps drawing the last chain it had.

    Nothing is fetched for the live app's replay of a past session (nor in
    SimLab, which patches this out): today's chain over a past session's tape
    would be a made-up read, so a replay reads only the net gamma the live
    app kept that day.
    """
    if getattr(getattr(sym_state, "app", None), "replay", None) is not None:
        return
    sym = sym_state.symbol
    now = time.monotonic()
    with _fetch_lock:
        if sym in _fetch_running or now - _fetch_tried.get(sym, -1e9) < OPTIONS_POLL_SEC:
            return
        _fetch_running.add(sym)
        _fetch_tried[sym] = now

    def fetch() -> None:
        try:
            with sym_state.lock:
                spot = sym_state.last_price
            data = fetch_options_walls_data(sym, spot=spot)
        except Exception:
            pass
        else:
            with sym_state.lock:
                sym_state.options_chain = data
        finally:
            with _fetch_lock:
                _fetch_running.discard(sym)

    threading.Thread(target=fetch, name=f"option-walls-{sym}", daemon=True).start()
