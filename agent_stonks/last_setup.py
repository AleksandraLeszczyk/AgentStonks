"""The live dashboard's setup on disk, so a restart opens where it was left.

Every choice on the page lives in Streamlit's session state, and that belongs
to one browser session. A restart of the app, or a dropped connection that the
browser comes back from as a new session, opened every one of them on its
shipped default again -- the symbols, the connection, the chart settings, the
personality, Apple Trader's rules, the venue -- and setting them all up again
was the slow part of getting back to trading. This module keeps them in
``data/last_setup.json`` and puts them back on a new session before its
widgets are drawn.

- **What is kept** is a widget's value, by its key: `KEYS`, and every key under
  `PREFIXES` (the keys Apple Trader's form and the model pickers build from the
  instrument or the provider). Never the credential fields -- those come from
  the environment, and a plain file under data/ is no place for them -- nor
  buttons, nor the day's "Continue today's session" box, which is not a setting.
- **Seeded, never forced.** A value goes into session state only where the key
  is missing, so it is the widget's starting value and the widget's own changes
  win. A stored value a widget can no longer take -- a model since removed, a
  number outside a range since narrowed -- is put back to that widget's default
  by Streamlit itself (1.59 resets an out-of-options or out-of-range session
  value rather than raising).
- **Filled in on every full run**, not only a session's first: Streamlit drops
  a keyed widget's value on the first run it is not drawn, so picking another
  personality and coming back reopened Apple Trader's form on its defaults.
- **Only what a session changed is written.** Every browser tab is its own
  session, and one that is only looking still holds the values it opened with;
  writing all of them would put back what another tab had just changed. A value
  is written once it differs from what this session first saw for it, so a
  default nobody touched is never stored either -- and a shipped default that
  changes later (a re-tuned buy/sell pair) still reaches the form.
- **Nothing is started.** The stream and the agent wait for ▶ Start, and so
  does the venue: a restored Alpaca LIVE comes back selected, with its warning
  above the button, and trades nothing until that is pressed.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import MutableMapping
from pathlib import Path

import streamlit as st

logger = logging.getLogger(__name__)

SETUP_PATH = Path(__file__).resolve().parent.parent / "data" / "last_setup.json"
_VERSION = 1

KEYS = frozenset({
    # Sidebar
    "sidebar_symbols",
    "sidebar_live_source",
    "sidebar_history_feed",
    # Live tab, Chart Settings
    "live_timeframe",
    "chart_candle_body",
    "chart_percentile_body",
    "chart_whiskers",
    "chart_fill_gaps",
    "chart_vwap",
    "chart_pre_market",
    "chart_vwma",
    "chart_avg_lines",
    "chart_option_walls",
    "chart_volume_baseline",
    "chart_momentum",
    "chart_net_gamma",
    "chart_profile_fit",
    "chart_mixture_components",
    "chart_mixture_fit_target",
    "model_overlay_keys",
    "candle_pattern_keys",
    "fvg_min_size",
    "fvg_hide_filled",
    # News, Pre-Market and Historical tabs
    "news_impact_method_select",
    "news_llm_provider_select",
    "premarket_provider",
    "hist_period",
    # Agent tab
    "agent_llm_personality",
    "agent_llm_provider",
    "agent_trading_mode",
    "agent_trade_sound_volume",
    "agent_starting_budget",
})
# Keys built from the instrument, the model or the provider on screen.
PREFIXES = ("apple_trader_", "agent_llm_model_", "premarket_model_")

# Session-state key of the value this session last saw under each kept key.
_SEEN = "_last_setup_seen"
_UNWRITABLE = object()
# Serialises the read-merge-write, so two tabs never interleave on the file.
_write_lock = threading.Lock()


def is_kept(key: object) -> bool:
    return isinstance(key, str) and (key in KEYS or key.startswith(PREFIXES))


def load(path: "Path | None" = None) -> dict:
    """The stored values, by widget key; empty when there are none or the
    file cannot be read."""
    path = path or SETUP_PATH
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        logger.warning("Could not read the last setup %s: %s", path, exc)
        return {}
    if not isinstance(record, dict) or record.get("version") != _VERSION:
        return {}
    values = record.get("values")
    if not isinstance(values, dict):
        return {}
    return {k: v for k, v in values.items() if is_kept(k)}


def _write(values: dict, path: Path) -> None:
    text = json.dumps({"version": _VERSION, "values": values}, indent=1, sort_keys=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _plain(value: object) -> object:
    """`value` as it comes back from JSON -- a copy, and a tuple as a list, so
    it compares equal to what a later load gives -- or `_UNWRITABLE`."""
    try:
        return json.loads(json.dumps(value))
    except (TypeError, ValueError):
        return _UNWRITABLE


def _seen(session: MutableMapping) -> dict:
    seen = session.get(_SEEN)
    if not isinstance(seen, dict):
        seen = {}
        session[_SEEN] = seen
    return seen


def restore(session: "MutableMapping | None" = None, path: "Path | None" = None) -> int:
    """Fill the kept keys `session` is missing from the stored setup.

    Call on every full run, before its first widget: a key can only be seeded
    before the widget it belongs to is drawn. Returns how many were filled."""
    session = st.session_state if session is None else session
    seen = _seen(session)
    filled = 0
    for key, value in load(path).items():
        if key not in session:
            session[key] = value
            seen[key] = value
            filled += 1
    return filled


def remember(session: "MutableMapping | None" = None, path: "Path | None" = None) -> bool:
    """Write the kept values `session` has changed since it first saw them.

    Call once the widgets are drawn: at the end of a full run, and at the end
    of a fragment that draws kept widgets, since a change inside one reruns the
    fragment alone. A key seen for the first time only has its value noted --
    that is the widget's default, not a choice. True when the file was written."""
    session = st.session_state if session is None else session
    seen = _seen(session)
    changed = {}
    for key in list(session.keys()):
        if not is_kept(key):
            continue
        value = _plain(session[key])
        if value is _UNWRITABLE:
            continue
        if key not in seen:
            seen[key] = value
        elif seen[key] != value:
            seen[key] = value
            changed[key] = value
    if not changed:
        return False
    path = path or SETUP_PATH
    with _write_lock:
        values = load(path)
        values.update(changed)
        try:
            _write(values, path)
        except OSError as exc:
            logger.warning("Could not save the last setup %s: %s", path, exc)
            return False
    return True


class _SeededWidgetWarning(logging.Filter):
    """Drops Streamlit's "created with a default value but also had its value
    set via the Session State API" for the keys seeded here. That is what
    seeding is, and the warning is logged with a whole stack on every start."""

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args if isinstance(record.args, tuple) else ()
        return not (
            str(record.msg).startswith("The widget with key")
            and args
            and is_kept(args[0])
        )


_policy_logger = logging.getLogger("streamlit.elements.lib.policies")
# Once per process: a dev-mode reload re-imports this module.
if not getattr(_policy_logger, "_last_setup_filter", False):
    _policy_logger.addFilter(_SeededWidgetWarning())
    _policy_logger._last_setup_filter = True
