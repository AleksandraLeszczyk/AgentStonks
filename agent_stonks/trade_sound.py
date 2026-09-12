"""An audible cue when a trade fills.

The agent trades on its own schedule while the user is doing something else --
another tab, another window, reading the news panel. A filled order is the one
event in this app worth interrupting for, and a chime is the only way to deliver
it to someone who is not looking at the screen.

Implemented as an inline Custom Component v2 rather than `st.audio`, for two
reasons. `st.audio(autoplay=True)` renders a visible player widget, which is
wrong furniture for a notification -- it would accumulate a row of dead
transport controls down the page, one per trade. And synthesising the tone in
the browser with the Web Audio API means there is no audio file to ship, no
binary in the repo, and no fetch to a CDN that the notification depends on.

Buy and sell get different shapes on purpose: a rising interval for a buy, a
falling one for a sell. The point is to be able to tell what happened without
looking, which a single undifferentiated beep cannot do.

Browsers refuse to play audio before the user has interacted with the page. In
practice that is satisfied by the click on "Start agent" -- nothing sounds
before then anyway, since there are no trades. The AudioContext is still
resumed defensively on every cue, because a tab restored from bfcache can come
back suspended.
"""
from __future__ import annotations

import streamlit as st

from .config import TRADE_SOUND_VOLUME

# Distinct two-note motifs. Buy rises, sell falls; both resolve quickly so the
# cue is over before it becomes annoying at the twentieth trade of a session.
_JS = """
let audioCtx = null

function context() {
  if (audioCtx === null) {
    const Ctor = window.AudioContext || window.webkitAudioContext
    if (!Ctor) return null
    audioCtx = new Ctor()
  }
  // A tab restored from bfcache, or one that autoplay policy never unlocked,
  // comes back suspended; resuming is a no-op when it is already running.
  if (audioCtx.state === "suspended") audioCtx.resume()
  return audioCtx
}

function chime(ctx, freqs, gainPeak) {
  const noteLen = 0.16
  freqs.forEach((freq, i) => {
    const osc = ctx.createOscillator()
    const gain = ctx.createGain()
    osc.type = "sine"
    osc.frequency.value = freq
    const t0 = ctx.currentTime + i * 0.085
    // Exponential ramps from a near-zero floor rather than linear from 0:
    // a hard start/stop on a sine produces an audible click.
    gain.gain.setValueAtTime(0.0001, t0)
    gain.gain.exponentialRampToValueAtTime(gainPeak, t0 + 0.012)
    gain.gain.exponentialRampToValueAtTime(0.0001, t0 + noteLen)
    osc.connect(gain)
    gain.connect(ctx.destination)
    osc.start(t0)
    osc.stop(t0 + noteLen + 0.02)
  })
}

const PLAYED = new WeakMap()

export default function (component) {
  const { data, parentElement } = component
  const cue = data && data.cue
  if (!cue || !cue.id) return

  // Python already de-duplicates, but a fragment can re-render with the same
  // data (a resize, a theme change) and that must not re-fire the chime.
  if (PLAYED.get(parentElement) === cue.id) return
  PLAYED.set(parentElement, cue.id)

  const ctx = context()
  if (!ctx) return
  const gainPeak = typeof cue.volume === "number" ? cue.volume : 0.22
  chime(ctx, cue.side === "sell" ? [880, 587] : [587, 880], gainPeak)
}
"""

# Registered once at import. Re-registering the same name from inside a
# function that runs every rerun produces confusing behaviour.
_TRADE_SOUND = st.components.v2.component(
    "agent_stonks_trade_sound",
    html="<span hidden></span>",
    js=_JS,
)


def play_trade_sound(cue: "dict | None", volume: float = TRADE_SOUND_VOLUME) -> None:
    """Mount the (invisible) sound component, sounding `cue` if there is one.

    `cue` is `{"id": str, "side": "buy" | "sell"}` for a trade that has not been
    announced yet, or None. Mounting unconditionally -- with or without a cue --
    keeps the component in the DOM across reruns, so its AudioContext survives
    and the first trade of a session is not the one that pays to create it.
    """
    payload = dict(cue) if cue else None
    if payload is not None:
        payload["volume"] = volume
    _TRADE_SOUND(data={"cue": payload}, key="agent_trade_sound")


def next_trade_cue(decisions: list, seen: "int | None") -> "tuple[dict | None, int]":
    """Decide what (if anything) to sound. Returns (cue, new_seen_count).

    `decisions` is the tracker's decision list; `seen` is how many filled trades
    this session has already announced, or None the first time it is asked.

    Three cases the caller must not get wrong, which is why this is a pure
    function with its own tests rather than logic inline in a fragment:

    * **First look** (`seen is None`): adopt the history silently. A session
      resumed with twenty trades behind it must not fire twenty chimes.
    * **Fewer trades than before**: the tracker was replaced (a new agent run),
      so the count resets rather than going negative.
    * **Several trades since the last poll**: the newest one sounds, once. The
      poll is a few seconds wide and a strategy can fill more than one order
      inside it; chiming per trade would overlap into noise.
    """
    filled = [
        d for d in decisions
        if getattr(d, "action", None) in ("buy", "sell")
        and getattr(d, "status", None) == "filled"
    ]
    count = len(filled)
    if seen is None or count < seen:
        return None, count
    if count == seen:
        return None, seen
    newest = filled[-1]
    # The id has to change per trade, and two trades can share a timestamp at
    # this resolution, so the running count goes in it too.
    return {"id": f"{newest.ts}#{count}", "side": newest.action}, count
