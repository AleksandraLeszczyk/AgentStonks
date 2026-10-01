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
"""
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .market_hours import MARKET_TZ
from .options import net_gamma_exposure
from .stream_common import TF_MINUTES

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "net_gamma"

_lock = threading.Lock()
# (symbol, timeframe, ET date) -> {bar time, UTC ISO: net gamma at its close}.
_kept: dict[tuple[str, str, str], dict[str, float]] = {}


def _utc(t) -> pd.Timestamp:
    stamp = pd.Timestamp(t)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _path(symbol: str, timeframe: str, day: str) -> Path:
    return CACHE_DIR / f"{symbol}_{timeframe}_{day}.json"


def _load(symbol: str, timeframe: str, day: str) -> dict[str, float]:
    path = _path(symbol, timeframe, day)
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return {str(t): float(v) for t, v in raw.items()}
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        logger.warning("Kept net gamma %s not read: %s", path, exc)
        return {}


def _save(symbol: str, timeframe: str, day: str, values: dict[str, float]) -> None:
    path = _path(symbol, timeframe, day)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(values), encoding="utf-8")
        os.replace(tmp, path)
    except OSError as exc:  # the panel still draws from memory
        logger.warning("Kept net gamma %s not written: %s", path, exc)


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
        kept = _kept.get(key)
        if kept is None:
            # A new day: the symbol's earlier days are not drawn any more.
            for old in [k for k in _kept if k[:2] == key[:2]]:
                del _kept[old]
            kept = _kept[key] = _load(symbol, timeframe, day)
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
