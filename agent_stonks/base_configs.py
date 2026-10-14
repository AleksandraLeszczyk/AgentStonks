"""Apple Trader's base configurations: a tuning result made the default for its pair.

SimLab's Tuning tab ends on one configuration -- the pick, or a square clicked
on a heatmap -- and its "Set as base configuration" button makes that what a
new run of the same (model, ticker) pair starts from, in the live app and in
SimLab alike. Until then a pick reached the defaults only by hand, its numbers
copied into `config.APPLE_TRADER_TUNED_LEVELS` and `APPLE_TRADER_TUNED_STOP`;
those tables stay the shipped defaults, and a pair with nothing stored here
still starts from them.

What is kept is the whole configuration, not only the numbers the grid swept:
a pick was measured under every rule of its job's base -- the confirmation
period, the breach policy, the fill rule -- and the same levels under other
rules are another strategy. One per pair, in ``data/apple_trader_base.json``::

    {"version": 1, "pairs": {"highlow/AAPL": {
        "config": {... every AppleTraderConfig field ...},
        "set_at": "2026-10-14T...",
        "source": {"job_id": ..., "overrides": {...}, "label": ...}}}}

Who reads it
------------
* `apple_trader.dayrange_levels`, `stop_for` and `min_win_for` answer a pair's
  numbers from here first, so everything that resolves a pair's defaults --
  a config built with them left None, Orchestra's pairs -- follows it.
* `apple_trader.default_config` decodes the whole of it: what both apps'
  Apple Trader form opens on (`apple_trader_ui.params`).
* The form's widget keys carry `key_suffix`, a digest of the stored config, so
  a new base re-seeds every knob of the pair on the next rerun of any open
  page, and the live app's remembered setup (`last_setup`), kept under the
  keys the old defaults were drawn with, does not put a value back over it.

A stored record -- a Results run, a day file, a tuning job -- carries its
numbers explicitly, so nothing here changes how a past run replays.

Plain JSON only: this module imports nothing of the trader's, so that
`apple_trader` can import it.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

BASE_PATH = Path(__file__).resolve().parent.parent / "data" / "apple_trader_base.json"
_VERSION = 1

# Serialises the read-merge-write within a process; across processes the write
# is atomic (`os.replace`), and SimLab is the only writer.
_write_lock = threading.Lock()
# ((path, mtime_ns, size), pairs). A pair's numbers are looked up whenever a
# config is built with them left None, so a lookup costs a stat, not a parse;
# a write from the other app shows up as a new mtime.
_cache: "tuple[tuple, dict] | None" = None


def pair_key(model_key: str, ticker: str) -> str:
    return f"{model_key}/{(ticker or '').strip().upper()}"


def _read(path: Path) -> dict:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        logger.warning("Could not read the base configurations %s: %s", path, exc)
        return {}
    if not isinstance(record, dict) or record.get("version") != _VERSION:
        return {}
    pairs = record.get("pairs")
    if not isinstance(pairs, dict):
        return {}
    return {
        key: entry for key, entry in pairs.items()
        if isinstance(entry, dict) and isinstance(entry.get("config"), dict)
    }


def load() -> dict:
    """Every stored base, by `pair_key`. Shared with the cache: read, never change."""
    global _cache
    path = BASE_PATH
    try:
        status = path.stat()
    except OSError:
        return {}
    stamp = (str(path), status.st_mtime_ns, status.st_size)
    cached = _cache
    if cached is not None and cached[0] == stamp:
        return cached[1]
    pairs = _read(path)
    _cache = (stamp, pairs)
    return pairs


def entry(model_key: str, ticker: str) -> "dict | None":
    """The pair's stored base -- its config, when it was set and from what -- or None."""
    return load().get(pair_key(model_key, ticker))


def stored(model_key: str, ticker: str) -> "dict | None":
    """The pair's base configuration as a record (a copy), or None."""
    found = entry(model_key, ticker)
    return dict(found["config"]) if found else None


def digest(config: dict) -> str:
    """A short fingerprint of a configuration record."""
    text = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


def key_suffix(model_key: str, ticker: str) -> str:
    """What the form appends to the pair's widget keys: empty with no base, so
    a pair never given one keeps the keys (and remembered values) it always had."""
    found = entry(model_key, ticker)
    return f"_base{digest(found['config'])}" if found else ""


def _write(pairs: dict) -> None:
    global _cache
    path = BASE_PATH
    text = json.dumps({"version": _VERSION, "pairs": pairs}, indent=1, sort_keys=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    _cache = None


def set_base(config: dict, source: "dict | None" = None) -> dict:
    """Make `config` -- a whole configuration record -- its pair's base; returns
    the entry written. `source` says where it came from (a tuning job, a cell)."""
    plain = json.loads(json.dumps(config))  # a copy, tuples as lists
    found = {
        "config": plain,
        "set_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": json.loads(json.dumps(source or {}, default=str)),
    }
    with _write_lock:
        pairs = dict(_read(BASE_PATH))
        pairs[pair_key(str(plain["model_key"]), str(plain["ticker"]))] = found
        _write(pairs)
    return found


def clear(model_key: str, ticker: str) -> bool:
    """Drop the pair's base, back to the shipped defaults. False if it had none."""
    with _write_lock:
        pairs = dict(_read(BASE_PATH))
        if pairs.pop(pair_key(model_key, ticker), None) is None:
            return False
        _write(pairs)
    return True
