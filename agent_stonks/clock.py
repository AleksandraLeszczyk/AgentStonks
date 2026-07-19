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
"""
from __future__ import annotations

import time as _time
from datetime import datetime, timezone

# Pinned simulation time; None = live (wall clock). A plain module global:
# a simulation owns its whole process (the sim UI runs separately from
# main.py), so no thread-local indirection is needed.
_sim_now: "datetime | None" = None

# Monotonic anchor mirroring _sim_now, so monotonic() stays consistent with
# now() while pinned (used by hold_sec timers and recent-price windows).
_sim_monotonic: "float | None" = None


def now() -> datetime:
    """Current time (UTC): the pinned simulation time, else the wall clock."""
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
