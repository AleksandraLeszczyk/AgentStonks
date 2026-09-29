"""The Live tab's quote card, with clickable cards.

The card itself is HTML built by `ui._quote_html`; `st.html` can show it but
cannot report a click back to Python. This inline Custom Component v2 shows the
same HTML and reports a click on any element carrying a `data-toggle`
attribute as a one-rerun trigger holding that attribute's value -- the Day Low
and Day High cards use it to switch their lines on the price chart.
"""
from __future__ import annotations

import streamlit as st

_JS = """
const SHOWN = new WeakMap()

export default function (component) {
  const { data, parentElement, setTriggerValue } = component
  const root = parentElement.querySelector("#quote")
  if (!root) return

  // The ticker re-renders every few seconds; only touch the DOM when the card
  // changed, so a click landing mid-render isn't lost.
  const html = (data && data.html) || ""
  if (SHOWN.get(parentElement) !== html) {
    root.innerHTML = html
    SHOWN.set(parentElement, html)
  }

  root.onclick = (e) => {
    const el = e.target.closest("[data-toggle]")
    if (el) setTriggerValue("toggled", el.dataset.toggle)
  }
}
"""

# Registered once at import, like the trade sound.
_QUOTE_CARD = st.components.v2.component(
    "agent_stonks_quote_card",
    html='<div id="quote"></div>',
    js=_JS,
)


def quote_card(card_html: str, *, key: str) -> "str | None":
    """Show `card_html`; return the `data-toggle` value of the element clicked
    since the last run, or None."""
    result = _QUOTE_CARD(
        data={"html": card_html},
        key=key,
        on_toggled_change=lambda: None,
    )
    return result.toggled
