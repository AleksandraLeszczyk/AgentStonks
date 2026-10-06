"""Swappable time source for the agent path.

Everything the trading agent touches (cycle prompts, market-hours gating,
decision timestamps, tactics hold timers, intraday-pace math) asks *this*
module for the current time instead of calling ``datetime.now`` directly. In
the live app that is exactly the wall clock. Under the simulation harness
(``simlab``) the clock is pinned to the historical moment being replayed, so
every "what time is it / what date is today" read inside a simulated cycle
lands on the simulated session rather than the real one.

Only the agent path is routed through here. UI rendering, stream plumbing,
and report generation keep the real wall clock -- they describe the live app,
not a simulated tape.

Two ways to move it, for two different owners:

- **the pin** (`set_simulated` / `clear`) is process-wide. SimLab owns its
  whole process, so pinning every thread at once is exactly right there.
- **a thread scope** (`bind` / `unbind` / `inherit`) moves only the threads
  bound to it. The live app's replay of a past session (`agent_stonks.replay`)
  runs inside the same process as a live session that must keep the real
  time, so its clock is bound to the replay's own threads -- the page's script
  run, its data feed, its agent and whatever those start through `inherit` --
  and to nothing else.
"""
from __future__ import annotations

import threading
import time as _time
from datetime import datetime, timezone

# Pinned simulation time; None = live (wall clock). A plain module global:
# a simulation owns its whole process (the sim UI runs separately from
# main.py), so no thread-local indirection is needed.
_sim_now: "datetime | None" = None

# Monotonic anchor mirroring _sim_now, so monotonic() stays consistent with
# now() while pinned (used by hold_sec timers and recent-price windows).
_sim_monotonic: "float | None" = None

# The scope this thread is bound to: anything with a `now()` returning an
# aware datetime. Unset = not bound.
_local = threading.local()


def now() -> datetime:
    """Current time (UTC): this thread's scope, else the pinned simulation
    time, else the wall clock."""
    scope = getattr(_local, "scope", None)
    if scope is not None:
        return scope.now()
    return _sim_now or datetime.now(timezone.utc)


def monotonic() -> float:
    """Monotonic seconds consistent with :func:`now` while pinned."""
    return _sim_monotonic if _sim_monotonic is not None else _time.monotonic()


def is_simulated() -> bool:
    return _sim_now is not None


def set_simulated(dt: datetime) -> None:
    """Pin the clock to `dt` (must be tz-aware). Advancing time = calling again."""
    global _sim_now, _sim_monotonic
    if dt.tzinfo is None:
        raise ValueError("simulated time must be timezone-aware")
    _sim_now = dt.astimezone(timezone.utc)
    _sim_monotonic = _sim_now.timestamp()


def clear() -> None:
    """Return to the wall clock."""
    global _sim_now, _sim_monotonic
    _sim_now = None
    _sim_monotonic = None


# --------------------------------------------------------------------------
# Thread scopes
#
# `monotonic()` deliberately ignores the scope: the one scope there is, the
# live replay, runs at the speed of the wall clock, so real monotonic seconds
# already are its seconds -- and the stream stamps `recent_prices` with
# `time.monotonic()` directly.
# --------------------------------------------------------------------------

def bind(scope: object) -> None:
    """Bind this thread to `scope` (anything with `now()`), until `unbind`."""
    _local.scope = scope


def unbind() -> None:
    """Return this thread to the pin / the wall clock."""
    _local.scope = None


def scope() -> "object | None":
    """The scope this thread is bound to, or None."""
    return getattr(_local, "scope", None)


def inherit(target):
    """`target` wrapped to run bound to the scope of the thread calling this.

    For `threading.Thread(target=clock.inherit(fn))`: a thread starts unbound,
    so a replay's agent would otherwise read the wall clock the moment it
    left the thread that launched it. Unbound callers get a wrapper that runs
    `target` unbound, i.e. the live behaviour unchanged.
    """
    bound = scope()

    def run(*args, **kwargs):
        _local.scope = bound
        try:
            return target(*args, **kwargs)
        finally:
            _local.scope = None

    return run


# --------------------------------------------------------------------------
# Timestamp parsing
#
# Not a time *source* -- these read a timestamp someone else produced, so the
# simulation pin above does not apply to them. They live here because this is
# the package's one dependency-free time module, and because the alternative
# was the same three lines inlined at twenty call sites.
#
# Alpaca stamps bars and quotes RFC-3339 with a 'Z' suffix, which
# `datetime.fromisoformat` did not accept before Python 3.11; the replace()
# keeps the parse working on either. A timestamp that arrives naive is read as
# UTC -- the convention every feed and tool output in this codebase uses, and
# the reading that keeps a later `.astimezone()` from silently interpreting it
# as the host's local time.
# --------------------------------------------------------------------------

def parse_iso_strict(raw: object) -> datetime:
    """Aware UTC datetime from an RFC-3339 string; raises on anything else.

    For call sites that cannot proceed without a timestamp -- a sort key or a
    dict key, where a None would fail later and further away.
    """
    parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def parse_iso(raw: object) -> "datetime | None":
    """Aware UTC datetime from an RFC-3339 string, or None if unparseable."""
    try:
        return parse_iso_strict(raw)
    except (TypeError, ValueError):
        return None


def bar_dt(bar: object) -> "datetime | None":
    """Aware UTC datetime of a bar's `t` field, or None if absent/unparseable."""
    try:
        raw = bar["t"]
    except (KeyError, TypeError, IndexError):
        return None
    return parse_iso(raw) if raw else None
