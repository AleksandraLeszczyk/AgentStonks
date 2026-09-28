"""Today's live session on disk, so a restart of the app picks it back up.

Everything a run builds up during the day lives on the Streamlit session's
`AppState` -- the decision ledger (cash, positions, every decision), the agent
log, the equity curve and the levels Apple Trader rested -- and all of it was
gone the moment the app restarted or the browser tab was reloaded. This module
keeps one JSON file per ET trading day under ``data/sessions/`` and puts it back
on a fresh `AppState`.

- **What restore gives back.** The ledger, the log, the equity history, the
  venue the run was on and Apple Trader's recorded levels plus the trader's own
  memory of an open position (`DayRangeTrader.memory`: its fill, stop, ladder
  and a session stand-down). Nothing is re-started: the agent stays stopped
  until ▶ Start, which then *continues* the day (`continues`) rather than
  opening a new ledger -- the user's choice, 2026-09-28. A new ET day, or a
  different venue, starts fresh.
- **The ledger is restored onto a simulated broker.** Only ▶ Start resolves a
  venue; until then an Alpaca run's numbers are the ledger as last saved, and
  the Start that continues it re-reads the account (`sync_from_broker`).
- **One writer per day, per process.** Every browser tab has its own
  `AppState`, and each restores the same file. Only the state that owns the day
  writes it: the first to restore it after a restart, or whichever last pressed
  ▶ Start (`claim`). A second tab that is only looking never overwrites the one
  that is trading.
- **Written by a background thread, not the UI.** The agent runs on its own
  thread and keeps trading with no browser attached, so the save cannot hang
  off a rerun. `start_autosave` compares a cheap signature every few seconds
  and writes on a change, atomically (temp file + rename), so a crash mid-write
  leaves the previous file whole.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
import weakref
from dataclasses import asdict, fields, is_dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from . import clock
from .broker import PaperBroker
from .decisions import Decision, DecisionTracker
from .market_hours import MARKET_TZ

logger = logging.getLogger(__name__)

SESSION_DIR = Path(__file__).resolve().parent.parent / "data" / "sessions"
AUTOSAVE_SEC = 5.0
_VERSION = 1

# date -> owner token of the AppState allowed to write that day's file.
# Process-wide on purpose: every Streamlit session lives in this one process.
_owners: dict[str, str] = {}
_owners_lock = threading.Lock()
# Serialises writes, so two threads never interleave on the temp file.
_write_lock = threading.Lock()

_DECISION_FIELDS = {f.name for f in fields(Decision)}


def session_date(now: "datetime | None" = None) -> str:
    """Today's ET trading date, YYYY-MM-DD: the day a session file is for."""
    return (now or clock.now()).astimezone(MARKET_TZ).date().isoformat()


def path_for(day: str) -> Path:
    return SESSION_DIR / f"{day}.json"


# --- encoding -----------------------------------------------------------------


def _default(value: object) -> object:
    """JSON for what `json` cannot write itself. Timestamps are tagged so they
    come back as `pd.Timestamp` -- the levels rows and the trader's entry
    compare them against bar timestamps."""
    if isinstance(value, (pd.Timestamp, datetime, date)):
        stamp = pd.Timestamp(value)
        tagged = {"$ts": stamp.isoformat()}
        # A named zone survives the trip (an ISO string only keeps the offset),
        # so a restored bar time prints and converts like a live one.
        zone = str(stamp.tz) if stamp.tz is not None else ""
        if "/" in zone or zone == "UTC":
            tagged["tz"] = zone
        return tagged
    if hasattr(value, "item"):  # numpy scalar
        return value.item()
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    return str(value)


def _object_hook(raw: dict) -> object:
    if "$ts" in raw and set(raw) <= {"$ts", "tz"}:
        stamp = pd.Timestamp(raw["$ts"])
        if raw.get("tz") and stamp.tz is not None:
            stamp = stamp.tz_convert(raw["tz"])
        return stamp
    return raw


def dumps(record: dict) -> str:
    return json.dumps(record, default=_default)


def loads(text: str) -> dict:
    return json.loads(text, object_hook=_object_hook)


# --- ownership ------------------------------------------------------------------


def owner_token(state) -> str:
    token = state.__dict__.get("_session_owner")
    if not token:
        token = uuid.uuid4().hex
        state.__dict__["_session_owner"] = token
    return token


def claim(state, day: "str | None" = None) -> None:
    """Make `state` the one that writes `day`'s file (today by default)."""
    with _owners_lock:
        _owners[day or session_date()] = owner_token(state)


def _claim_if_free(state, day: str) -> None:
    with _owners_lock:
        _owners.setdefault(day, owner_token(state))


def owns(state, day: "str | None" = None) -> bool:
    with _owners_lock:
        return _owners.get(day or session_date()) == owner_token(state)


# --- capture / save -------------------------------------------------------------


def _levels_record(levels: "dict | None") -> "dict | None":
    if not levels:
        return None
    config = levels.get("config")
    return {
        "ticker": levels.get("ticker"),
        "date": levels.get("date"),
        "config": asdict(config) if is_dataclass(config) else None,
        "rows": list(levels.get("rows") or []),
        "memory": levels.get("memory"),
    }


def capture(state) -> dict:
    """The part of `state` a restart would lose, as one JSON-able record."""
    tracker = state.decision_tracker
    ledger = None
    if tracker is not None:
        with tracker.lock:
            ledger = {
                "cash": tracker.cash,
                "positions": dict(tracker.positions),
                "trade_cost": tracker.trade_cost,
                "decisions": [asdict(d) for d in tracker.decisions],
            }
    with state.lock:
        log = list(state.agent_log)
        equity = list(state.agent_equity_history)
    start = state.agent_start_time
    return {
        "version": _VERSION,
        "date": getattr(state, "session_date", "") or session_date(),
        "saved_at": clock.now().isoformat(),
        "trading_mode": state.trading_mode,
        "trading_mode_requested": state.trading_mode_requested,
        "trading_status": state.trading_status,
        "llm_personality": state.llm_personality,
        "starting_budget": state.starting_budget,
        "agent_start_time": start.isoformat() if start is not None else None,
        "agent_log": log,
        "agent_equity_history": equity,
        "tracker": ledger,
        "apple_trader_levels": _levels_record(getattr(state, "apple_trader_levels", None)),
    }


def _has_content(record: dict) -> bool:
    return bool(record.get("tracker") or record.get("agent_log"))


def save(state, *, force: bool = False) -> "Path | None":
    """Write `state`'s session to its day's file, atomically.

    Skipped (None) when another state owns the day -- unless `force` -- or
    when there is nothing to keep yet."""
    day = getattr(state, "session_date", "") or session_date()
    with _owners_lock:
        owner = _owners.get(day)
    if not force and owner is not None and owner != owner_token(state):
        return None
    record = capture(state)
    if not _has_content(record):
        return None
    if owner is None:
        # Nobody owns the day yet (no file was restored this process): the
        # first state with something to keep takes it.
        _claim_if_free(state, day)
        if not owns(state, day):
            return None
    path = path_for(day)
    text = dumps(record)
    with _write_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    return path


def archive(day: "str | None" = None) -> "Path | None":
    """Move `day`'s file aside before a fresh ledger replaces it, so starting
    over never destroys what the day already did. Returns the new path."""
    path = path_for(day or session_date())
    if not path.exists():
        return None
    stamp = clock.now().astimezone(MARKET_TZ).strftime("%H%M%S")
    target = path.with_name(f"{path.stem}-replaced-{stamp}.json")
    with _write_lock:
        os.replace(path, target)
    return target


# --- load / restore -------------------------------------------------------------


def load(day: "str | None" = None) -> "dict | None":
    """`day`'s saved session (today by default), or None when there is none
    or it cannot be read."""
    day = day or session_date()
    path = path_for(day)
    if not path.exists():
        return None
    try:
        record = loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Could not read the saved session %s: %s", path, exc)
        return None
    if record.get("version") != _VERSION or record.get("date") != day:
        return None
    return record


def _tracker_from(ledger: dict) -> DecisionTracker:
    tracker = DecisionTracker(
        starting_cash=float(ledger.get("cash") or 0.0),
        broker=PaperBroker(),
        trade_cost=float(ledger.get("trade_cost") or 0.0),
    )
    tracker.positions = {
        str(s): float(q) for s, q in (ledger.get("positions") or {}).items()
    }
    tracker.decisions = [
        Decision(**{k: v for k, v in raw.items() if k in _DECISION_FIELDS})
        for raw in ledger.get("decisions") or []
    ]
    return tracker


def _levels_from(record: "dict | None") -> "dict | None":
    if not record or not record.get("rows"):
        return None
    from .apple_trader import AppleTraderConfig

    config = None
    raw = record.get("config")
    if raw:
        known = {f.name for f in fields(AppleTraderConfig)}
        try:
            config = AppleTraderConfig(**{k: v for k, v in raw.items() if k in known})
        except (TypeError, ValueError) as exc:
            logger.warning("Saved Apple Trader config not restored: %s", exc)
    return {
        "ticker": record.get("ticker"),
        "date": record.get("date"),
        "config": config,
        "rows": list(record["rows"]),
        "memory": record.get("memory"),
    }


def restore(state, record: "dict | None" = None) -> "dict | None":
    """Put today's saved session back on a fresh `state`.

    Returns a short summary for the UI, or None when there was nothing for
    today. The first state to restore a day owns it (see the module note)."""
    record = record if record is not None else load()
    if not record:
        return None
    day = record["date"]
    ledger = record.get("tracker")
    state.decision_tracker = _tracker_from(ledger) if ledger else None
    state.session_date = day
    state.trading_mode = record.get("trading_mode") or "local"
    state.trading_mode_requested = record.get("trading_mode_requested") or ""
    state.trading_status = record.get("trading_status") or ""
    state.starting_budget = float(record.get("starting_budget") or state.starting_budget)
    start = record.get("agent_start_time")
    state.agent_start_time = pd.Timestamp(start).to_pydatetime() if start else None
    with state.lock:
        state.agent_log = list(record.get("agent_log") or [])
        state.agent_equity_history = list(record.get("agent_equity_history") or [])
    levels = _levels_from(record.get("apple_trader_levels"))
    if levels is not None:
        state.apple_trader_levels = levels
    _claim_if_free(state, day)
    tracker = state.decision_tracker
    summary = {
        "date": day,
        "saved_at": record.get("saved_at"),
        "path": str(path_for(day)),
        "decisions": len(tracker.decisions) if tracker else 0,
        "fills": sum(1 for d in tracker.decisions if d.status == "filled") if tracker else 0,
        "positions": {s: q for s, q in tracker.positions.items() if q} if tracker else {},
        "cash": tracker.cash if tracker else None,
    }
    state.session_restored = summary
    logger.info(
        "Restored the %s session from %s: %d decisions, positions %s",
        day, summary["path"], summary["decisions"], summary["positions"] or "flat",
    )
    return summary


def continues(state, venue: str, day: "str | None" = None) -> bool:
    """Whether ▶ Start should carry on `state`'s ledger rather than open a new
    one: there is a ledger, it is today's, and it was kept on this venue."""
    return (
        state.decision_tracker is not None
        and getattr(state, "session_date", "") == (day or session_date())
        and state.trading_mode == venue
    )


# --- autosave -------------------------------------------------------------------


def _signature(state) -> tuple:
    """Changes whenever something worth saving does. The equity history is left
    out on purpose: the performance panel appends to it on every poll, even in
    a tab that is only looking, and it is saved along with everything else."""
    tracker = state.decision_tracker
    if tracker is not None:
        with tracker.lock:
            ledger = (
                id(tracker),
                len(tracker.decisions),
                tracker.cash,
                tuple(sorted(tracker.positions.items())),
            )
    else:
        ledger = None
    levels = getattr(state, "apple_trader_levels", None) or {}
    rows = levels.get("rows") or []
    return (
        ledger,
        len(state.agent_log),
        getattr(state, "session_date", ""),
        state.trading_mode,
        state.starting_budget,
        state.agent_start_time,
        len(rows),
        rows[-1].get("t") if rows else None,
        repr(levels.get("memory")),
    )


def start_autosave(state, interval: float = AUTOSAVE_SEC) -> None:
    """Save `state` on a background thread whenever it changes. Idempotent.

    Holds only a weak reference, so the thread ends with the session."""
    thread = state.__dict__.get("_session_autosave")
    if thread is not None and thread.is_alive():
        return
    # A state made before this module existed (a hot reload over a running
    # session) has a ledger but no date: it is the day its run started on.
    start = getattr(state, "agent_start_time", None)
    if (
        not getattr(state, "session_date", "")
        and getattr(state, "decision_tracker", None) is not None
        and start is not None
    ):
        state.session_date = session_date(start)
    ref = weakref.ref(state)
    first = _signature(state)

    def run() -> None:
        last = first
        while True:
            time.sleep(interval)
            current = ref()
            if current is None:
                return
            try:
                sig = _signature(current)
                if sig != last and save(current) is not None:
                    last = sig
            except Exception:  # a failed save must not end the saving
                logger.exception("Saving the session failed")
            del current

    thread = threading.Thread(target=run, daemon=True, name="session-autosave")
    state.__dict__["_session_autosave"] = thread
    thread.start()
