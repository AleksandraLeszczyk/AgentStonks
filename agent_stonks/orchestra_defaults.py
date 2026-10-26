"""Orchestra's default pairs: the (ticker, model) pairs its form opens on, set
from SimLab's ML summary.

SimLab's ML summary names each instrument's best model; its "Set Orchestra"
button makes the pairs ticked there what Orchestra's pair picker opens on, in
the live app and in SimLab alike. With nothing stored, Orchestra opens on every
pair SimLab's tuning has picked levels for (`orchestra.default_pairs`). Kept in
``data/orchestra_pairs.json``::

    {"version": 1, "pairs": ["AMD:highlow", "AAPL:goodmoments"],
     "set_at": "2026-10-26T...", "source": {"window": "month"}}

The pairs are racer keys (`orchestra.racer_key`). The pair picker's widget key
carries `key_suffix`, a digest of what is stored, so a new default re-seeds the
picker on the next rerun of any open page, and the live app's remembered setup
(`last_setup`), kept under the key the old default was drawn with, does not
put the old pairs back over it.

Plain JSON only, like `base_configs`: this module imports nothing of the
trader's.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

PAIRS_PATH = Path(__file__).resolve().parent.parent / "data" / "orchestra_pairs.json"
_VERSION = 1


def load() -> "dict | None":
    """The stored default -- its pairs, when it was set and from what -- or None."""
    try:
        record = json.loads(PAIRS_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        logger.warning("Could not read Orchestra's default pairs %s: %s", PAIRS_PATH, exc)
        return None
    if not isinstance(record, dict) or record.get("version") != _VERSION:
        return None
    pairs = record.get("pairs")
    if not isinstance(pairs, list) or not all(isinstance(p, str) for p in pairs):
        return None
    return record


def pairs() -> "list[str] | None":
    """The stored default pairs, or None with nothing stored."""
    record = load()
    return list(record["pairs"]) if record else None


def key_suffix() -> str:
    """What the pair picker appends to its widget key: empty with nothing
    stored, so the picker keeps the key (and remembered value) it always had."""
    found = pairs()
    if found is None:
        return ""
    text = json.dumps(found)
    return f"_set{hashlib.sha1(text.encode('utf-8')).hexdigest()[:8]}"


def set_pairs(keys: "list[str]", source: "dict | None" = None) -> dict:
    """Make `keys` -- racer keys, "TICKER:model" -- Orchestra's default pairs;
    returns the record written. `source` says where they came from."""
    if not keys:
        raise ValueError("Orchestra needs at least one (ticker, model) pair")
    record = {
        "version": _VERSION,
        "pairs": list(dict.fromkeys(str(k) for k in keys)),
        "set_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": json.loads(json.dumps(source or {}, default=str)),
    }
    PAIRS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = PAIRS_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=1), encoding="utf-8")
    os.replace(tmp, PAIRS_PATH)
    return record


def clear() -> None:
    """Back to the shipped default: every tuned pair."""
    try:
        PAIRS_PATH.unlink()
    except FileNotFoundError:
        pass
