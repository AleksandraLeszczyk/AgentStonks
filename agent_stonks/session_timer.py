"""The sidebar's session timer: how far into the regular session it is.

It reads 0:00:00 at the 09:30 ET bell and counts up through the session. Before
the bell it counts down to it as a negative time -- −0:12:34 is twelve and a
half minutes to go -- and so it does from the close on: the evening and the
weekend count down to the next session's open. The time is the app's clock
(`agent_stonks.clock`), so under Dummy data it is the replayed session's, and
it stands still while the replay waits for ▶ Start Agent.

Sessions are `market_hours`' (09:30-16:00 ET, Monday-Friday, no holidays).

The browser counts the seconds itself, in an inline Custom Component v2: a
fragment rerun every second only to move a clock would be a page round trip a
second. The server sends the time now and then (`SYNC_SEC`, and every full
rerun, so a click on ▶ Start Agent starts a paused replay's timer at once);
in between the component counts on from the moment the time arrived.
"""
from __future__ import annotations

from datetime import date, datetime

import streamlit as st

from . import market_hours
from .config import PALETTE
from .market_hours import MARKET_CLOSE, MARKET_OPEN, MARKET_TZ

# How often the sidebar re-sends the time. The browser keeps it to the second
# in between; a sync only has to catch the close (the timer reads "closed"
# from 16:00 until one turns it into the next open's countdown) and a replay
# that stops on its own.
SYNC_SEC = 30

# The regular session's length, where the count up ends.
SESSION_SEC = (
    datetime.combine(date.min, MARKET_CLOSE) - datetime.combine(date.min, MARKET_OPEN)
).total_seconds()


def session_seconds(now: datetime) -> float:
    """Seconds since the open of the session in progress at `now`, or minus
    the seconds to the next open when no session is."""
    opened = market_hours.session_open(now)
    if opened is not None:
        return (now - opened).total_seconds()
    return -market_hours.seconds_until_next_open(now)


def before_label(now: datetime) -> str:
    """What a negative time counts down to: today's open, or the next
    weekday's, named, when that is another day."""
    nxt = market_hours.next_market_open(now).astimezone(MARKET_TZ)
    day = "" if nxt.date() == now.astimezone(MARKET_TZ).date() else f"{nxt:%a} "
    return f"Before the open · {day}{nxt:%H:%M} ET"


_CSS = f"""
.timer {{
  font-family: var(--st-font);
  color: var(--st-text-color);
  line-height: 1.25;
}}
.label {{
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 0.8rem;
}}
.text {{
  opacity: 0.7;
}}
.dot {{
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: {PALETTE['muted']};
}}
.timer[data-phase="open"] .dot {{
  background: {PALETTE['up']};
}}
.timer[data-phase="before"] .dot {{
  background: {PALETTE['orange']};
}}
.time {{
  font-size: 1.75rem;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
  letter-spacing: 0.01em;
}}
"""

_JS = """
const TIMERS = new WeakMap()

const pad = (n) => String(n).padStart(2, "0")

// h:mm:ss, with the days in front past 24 hours (a weekend's countdown).
// Elapsed time rounds down and a countdown up, so the bell is the one moment
// that reads 0:00:00 and a countdown never shows −0:00:00.
function format(sec) {
  const whole = sec < 0 ? Math.ceil(-sec) : Math.floor(sec)
  const d = Math.floor(whole / 86400)
  const h = Math.floor((whole % 86400) / 3600)
  const hms = `${pad(Math.floor((whole % 3600) / 60))}:${pad(whole % 60)}`
  return (sec < 0 ? "−" : "") + (d ? `${d}d ${pad(h)}:${hms}` : `${h}:${hms}`)
}

function draw(timer) {
  const counted = timer.ticking ? (performance.now() - timer.syncedAt) / 1000 : 0
  const sec = timer.seconds + counted
  let phase = "open"
  let shown = sec
  if (sec < 0) {
    phase = "before"
  } else if (sec >= timer.closeAt) {
    // From the close until a sync brings the next session's countdown.
    phase = "closed"
    shown = timer.closeAt
  }
  const text = format(shown)
  if (text !== timer.shownText) {
    timer.time.textContent = text
    timer.shownText = text
  }
  const label = timer.labels[phase]
  if (phase !== timer.root.dataset.phase || label !== timer.text.textContent) {
    timer.root.dataset.phase = phase
    timer.text.textContent = label
  }
}

function install(parentElement) {
  const timer = {
    root: parentElement.querySelector(".timer"),
    text: parentElement.querySelector(".text"),
    time: parentElement.querySelector(".time"),
  }
  // A quarter second, so the shown second turns within that of the real one.
  timer.interval = setInterval(() => draw(timer), 250)
  TIMERS.set(parentElement, timer)
  return timer
}

export default function (component) {
  const { data, parentElement } = component
  if (!data) return
  const timer = TIMERS.get(parentElement) || install(parentElement)
  // Count on from the moment a new time arrived. The same time drawn again is
  // not a new one: restarting from it would set the timer back.
  if (data.seconds !== timer.sentSeconds || data.ticking !== timer.ticking) {
    timer.sentSeconds = data.seconds
    timer.seconds = data.seconds
    timer.ticking = data.ticking
    timer.syncedAt = performance.now()
  }
  timer.closeAt = data.close_at
  timer.labels = data.labels
  draw(timer)
  // Streamlit runs the last of these when the component unmounts.
  return () => {
    clearInterval(timer.interval)
    TIMERS.delete(parentElement)
  }
}
"""

# Registered once at import, like the trade sound.
_SESSION_TIMER = st.components.v2.component(
    "agent_stonks_session_timer",
    html=(
        '<div class="timer">'
        '<div class="label"><span class="dot"></span><span class="text"></span></div>'
        '<div class="time"></div>'
        "</div>"
    ),
    css=_CSS,
    js=_JS,
)


def session_timer(now: datetime, *, ticking: bool, key: str) -> None:
    """Show the session time at `now`, counted on in the browser while
    `ticking` (the clock is running) until the next call re-syncs it."""
    _SESSION_TIMER(
        data={
            "seconds": session_seconds(now),
            "ticking": ticking,
            "close_at": SESSION_SEC,
            "labels": {
                "open": f"Session · since {MARKET_OPEN:%H:%M} ET",
                "before": before_label(now),
                "closed": f"Session closed · {MARKET_CLOSE:%H:%M} ET",
            },
        },
        key=key,
    )
