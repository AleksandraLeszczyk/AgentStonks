"""The Agent tab's log, which holds still while it is being read.

The log is newest-first in a scroll box, re-sent by a fragment every few
seconds. Drawn with `st.html`, each new entry replaced the box and threw a
reader who had scrolled down to older entries back to the top. This inline
Custom Component v2 shows the same HTML but decides in the browser -- the only
place that knows where the box is scrolled -- when to take a new copy:

* at the top of the box, every copy is shown as it arrives, as before;
* scrolled down, the box keeps what it shows (no new entries above, no old
  ones dropping off below) and a pill says there is newer activity; scrolling
  back to the top, or clicking the pill, shows the newest copy;
* while its tab is hidden it follows every copy, and it comes back at the top
  -- going to another tab and back is the other way to catch up.
"""
from __future__ import annotations

import streamlit as st

from .config import PALETTE

_CSS = f"""
.log {{
  position: relative;
}}
#box {{
  font-family: Inter, sans-serif;
  max-height: 420px;
  overflow-y: auto;
}}
#pill {{
  position: absolute;
  top: 6px;
  left: 50%;
  transform: translateX(-50%);
  z-index: 1;
  background: {PALETTE['accent']};
  color: {PALETTE['bg']};
  border: none;
  border-radius: 999px;
  padding: 3px 12px;
  font: 600 11px Inter, sans-serif;
  cursor: pointer;
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.4);
}}
#pill[hidden] {{
  display: none;
}}
"""

_JS = """
const VIEWS = new WeakMap()

// A zoomed page scrolls in fractions of a pixel, so "at the top" has a margin.
const TOP_PX = 2

function atTop(box) {
  return box.scrollTop <= TOP_PX
}

// Its tab is not on screen (display: none somewhere above it).
function hidden(box) {
  return box.getClientRects().length === 0
}

function showLatest(view) {
  if (view.latest !== view.shown) {
    view.box.innerHTML = view.latest
    view.shown = view.latest
  }
  view.pill.hidden = true
}

function install(parentElement) {
  const box = parentElement.querySelector("#box")
  const pill = parentElement.querySelector("#pill")
  const view = { box, pill, latest: "", shown: null, wasHidden: false }
  box.addEventListener(
    "scroll",
    () => {
      if (atTop(box)) showLatest(view)
    },
    { passive: true },
  )
  pill.addEventListener("click", () => box.scrollTo({ top: 0 }))
  // A hidden box has no size, so its tab being left and entered again both
  // resize it.
  view.watcher = new ResizeObserver(() => {
    if (hidden(box)) {
      view.wasHidden = true
    } else if (view.wasHidden) {
      view.wasHidden = false
      box.scrollTop = 0
      showLatest(view)
    }
  })
  view.watcher.observe(box)
  VIEWS.set(parentElement, view)
  return view
}

export default function (component) {
  const { data, parentElement } = component
  const view = VIEWS.get(parentElement) || install(parentElement)
  view.latest = (data && data.html) || ""
  if (view.shown === null || atTop(view.box) || hidden(view.box)) {
    showLatest(view)
  } else if (view.latest !== view.shown) {
    view.pill.hidden = false
  }
  // Streamlit runs the last of these when the component unmounts.
  return () => {
    view.watcher.disconnect()
    VIEWS.delete(parentElement)
  }
}
"""

# Registered once at import, like the trade sound.
_AGENT_LOG = st.components.v2.component(
    "agent_stonks_agent_log",
    html=(
        '<div class="log">'
        '<button id="pill" type="button" hidden>↑ New activity</button>'
        '<div id="box"></div>'
        "</div>"
    ),
    css=_CSS,
    js=_JS,
)


def agent_log_view(log_html: str, *, key: str) -> None:
    """Show `log_html` (the log's cards, newest first) in a scroll box that
    takes a new copy only while it is scrolled to the top."""
    _AGENT_LOG(data={"html": log_html}, key=key)
