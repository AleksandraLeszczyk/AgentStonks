"""Bring a dead browser tab back by itself.

The agent, the streams and the server all run on without a browser, but the
page is the only way to watch them or stop them -- and Streamlit's page can die
on its own with the server perfectly healthy. On 2026-10-02 it did, with
"Failed to process a Websocket message. Error: Cached ForwardMsg MISS": a fatal
error puts Streamlit's frontend in a state it never leaves (no retry, no
reconnect) until someone reloads the tab. run_app.py could not help -- the
server was still answering -- and the server log had no trace of it.

That particular error is gone at the source (.streamlit/config.toml turns off
the message cache it came from), but other fatal errors end the same way. So
the page watches itself: an invisible Custom Component v2 installs one
watchdog on the page, and a `run_every` fragment re-renders it every
WATCHDOG_BEAT_SEC, which is its heartbeat -- it only arrives while the server
is reaching the page. The page reloads itself when either:

* Streamlit's own connection badge ("Connecting" / "Error") has been up for
  OFFLINE_RELOAD_SEC: the frontend has given up, or is not getting back; or
* no heartbeat has arrived for SILENT_RELOAD_SEC (longer while a run is in
  progress or the tab is in the background, where Chrome lets timers fire only
  once a minute): the page is stuck some other way.

Either way only once the server has answered /_stcore/health for
HEALTHY_GRACE_SEC: while it is down or restarting there is nothing a reload
can do, and Streamlit reconnects by itself when it is back. Reloads back off
(1, 2, 4 ... 10 min apart within RELOAD_WINDOW_SEC) so a page that cannot come
up is retried, never hammered, and never given up on.

A reloaded tab is a new session, which takes over the state that kept running
(`stream.adopt_orphaned_session`). It reports why it was reloaded, once: the
report is logged on the server, so the cause is in data/logs, and shown in the
recovery banner.
"""
from __future__ import annotations

import logging
import time

import streamlit as st

logger = logging.getLogger(__name__)

# The heartbeat: how often the fragment holding the watchdog re-renders it.
WATCHDOG_BEAT_SEC = 5
# How often the page checks on itself.
CHECK_EVERY_SEC = 5
# Streamlit's connection badge up this long means its frontend is not coming
# back by itself. It shows "Connecting" for a second or two on any reconnect.
OFFLINE_RELOAD_SEC = 20
# No heartbeat for this long (18 beats) means the page is stuck. A run in
# progress holds the fragments back -- a full rerun can take a while -- and a
# background tab may only get to run timers once a minute.
SILENT_RELOAD_SEC = 90
SILENT_RELOAD_RUNNING_SEC = 5 * 60
SILENT_RELOAD_HIDDEN_SEC = 3 * 60
# The server must have been answering this long before the page reloads, so a
# server just back from a restart gets the chance to reconnect the page itself.
HEALTHY_GRACE_SEC = 10
HEALTH_TIMEOUT_SEC = 4
# A second reload within RELOAD_WINDOW_SEC of the first waits RELOAD_BACKOFF_MIN_SEC
# after it, a third twice that, and so on up to RELOAD_BACKOFF_MAX_SEC.
RELOAD_WINDOW_SEC = 30 * 60
RELOAD_BACKOFF_MIN_SEC = 60
RELOAD_BACKOFF_MAX_SEC = 10 * 60
# A report older than this is not about this page load (a closed tab reopened
# brings its sessionStorage back with it).
REPORT_MAX_AGE_SEC = 5 * 60

_JS = """
// One watchdog per page however often the component renders: it lives on
// window, and each render only hands it the settings and a heartbeat.
const WATCHDOG = "__agentStonksPageWatchdog"
const STORE = "agentStonksPageWatchdog"

function load() {
  try {
    return JSON.parse(sessionStorage.getItem(STORE) || "{}") || {}
  } catch (e) {
    return {}
  }
}

function save(store) {
  try {
    sessionStorage.setItem(STORE, JSON.stringify(store))
  } catch (e) {}
}

// The message of Streamlit's "Connection error" dialog, when it is up.
function errorText() {
  const dialog = document.querySelector('[role="dialog"]')
  const text = (dialog && dialog.textContent) || ""
  if (!text.includes("Connection error")) return ""
  return text.replace("Connection error", "").trim().slice(0, 400)
}

async function serverAnswers(cfg) {
  const abort = new AbortController()
  const timer = setTimeout(() => abort.abort(), cfg.health_timeout_ms)
  try {
    const response = await fetch(cfg.health_url, { cache: "no-store", signal: abort.signal })
    return response.ok
  } catch (e) {
    return false
  } finally {
    clearTimeout(timer)
  }
}

// What is wrong with the page, or "" when nothing is.
function problem(w, now) {
  const cfg = w.cfg
  const badge = document.querySelector('[data-testid="stConnectionStatus"]')
  w.offlineSince = badge ? (w.offlineSince ?? now) : null
  if (badge && now - w.offlineSince >= cfg.offline_ms) {
    const label = badge.textContent.trim().toLowerCase() || "offline"
    return `lost its connection to the server (${label} for ${Math.round((now - w.offlineSince) / 1000)} s)`
  }
  const running = !!document.querySelector('[data-testid="stStatusWidgetRunningIcon"]')
  const limit = running ? cfg.silent_running_ms : document.hidden ? cfg.silent_hidden_ms : cfg.silent_ms
  const silent = now - w.lastBeat
  if (silent >= limit) return `had no update from the server for ${Math.round(silent / 1000)} s`
  return ""
}

// How long a reload now still has to wait, given the ones before it.
function backoffLeft(cfg, reloads, now) {
  if (!reloads.length) return 0
  const wait = Math.min(cfg.backoff_min_ms * 2 ** (reloads.length - 1), cfg.backoff_max_ms)
  return Math.max(0, reloads[reloads.length - 1] + wait - now)
}

async function check(w) {
  const cfg = w.cfg
  const now = Date.now()
  const reason = problem(w, now)
  if (!reason) {
    w.healthySince = null
    return
  }
  // A server that is down or restarting is run_app.py's to bring back, and
  // Streamlit reconnects the page by itself once it answers.
  if (!(await serverAnswers(cfg))) {
    w.healthySince = null
    return
  }
  w.healthySince = w.healthySince ?? now
  if (now - w.healthySince < cfg.healthy_grace_ms) return
  const store = load()
  const reloads = (store.reloads || []).filter((t) => now - t < cfg.reload_window_ms)
  if (backoffLeft(cfg, reloads, now) > 0) return
  store.reloads = [...reloads, now]
  store.report = { at: now, reason, detail: errorText(), reloads: store.reloads.length }
  save(store)
  console.warn(`Page watchdog: this page ${reason}; reloading`)
  window.location.reload()
}

function install(cfg) {
  const w = { cfg, lastBeat: Date.now(), offlineSince: null, healthySince: null }
  const tick = async () => {
    try {
      await check(w)
    } catch (e) {
      console.error("Page watchdog check failed", e)
    }
    setTimeout(tick, w.cfg.check_ms)
  }
  setTimeout(tick, cfg.check_ms)
  // The reload that brought this page here, said once to the server.
  const store = load()
  if (store.report) {
    w.report = Date.now() - store.report.at < cfg.report_max_age_ms ? store.report : null
    delete store.report
    save(store)
  }
  return w
}

export default function (component) {
  const { data, setTriggerValue } = component
  const cfg = data && data.config
  if (!cfg) return
  const w = window[WATCHDOG] || (window[WATCHDOG] = install(cfg))
  w.cfg = cfg
  w.lastBeat = Date.now()
  if (w.report) {
    const report = w.report
    w.report = null
    setTimeout(() => setTriggerValue("recovered", report), 0)
  }
}
"""

# Registered once at import, like the trade sound.
_WATCHDOG = st.components.v2.component(
    "agent_stonks_page_watchdog",
    html="<span hidden></span>",
    js=_JS,
)


def health_url(base_url_path: str) -> str:
    """The server's health route, under Streamlit's `server.baseUrlPath`."""
    base = base_url_path.strip("/")
    return f"/{base}/_stcore/health" if base else "/_stcore/health"


def watchdog_config(base_url_path: str = "") -> dict:
    """The watchdog's settings, in the milliseconds the page counts in."""
    return {
        "check_ms": CHECK_EVERY_SEC * 1000,
        "offline_ms": OFFLINE_RELOAD_SEC * 1000,
        "silent_ms": SILENT_RELOAD_SEC * 1000,
        "silent_running_ms": SILENT_RELOAD_RUNNING_SEC * 1000,
        "silent_hidden_ms": SILENT_RELOAD_HIDDEN_SEC * 1000,
        "healthy_grace_ms": HEALTHY_GRACE_SEC * 1000,
        "health_timeout_ms": HEALTH_TIMEOUT_SEC * 1000,
        "health_url": health_url(base_url_path),
        "reload_window_ms": RELOAD_WINDOW_SEC * 1000,
        "backoff_min_ms": RELOAD_BACKOFF_MIN_SEC * 1000,
        "backoff_max_ms": RELOAD_BACKOFF_MAX_SEC * 1000,
        "report_max_age_ms": REPORT_MAX_AGE_SEC * 1000,
    }


def describe_reload(report: dict) -> str:
    """'it lost its connection to the server (error for 20 s): Failed to ...'"""
    text = f"it {report.get('reason') or 'stopped'}"
    # Callers end the sentence themselves.
    detail = (report.get("detail") or "").strip().rstrip(".")
    return f"{text}: {detail}" if detail else text


def page_watchdog(*, key: str = "page_watchdog") -> "dict | None":
    """Mount the watchdog with a fresh heartbeat. Returns, in the one run it
    arrives in, the report of the reload that brought this page back --
    `{"at": ms, "reason": str, "detail": str, "reloads": n}` -- else None.
    Call it from a `run_every=WATCHDOG_BEAT_SEC` fragment."""
    try:
        base_url_path = str(st.get_option("server.baseUrlPath") or "")
    except Exception:
        base_url_path = ""
    result = _WATCHDOG(
        data={"config": watchdog_config(base_url_path), "beat": time.time()},
        key=key,
        on_recovered_change=lambda: None,
    )
    report = result.recovered
    if not isinstance(report, dict):
        return None
    logger.warning(
        "A browser tab reloaded itself because %s (reload %s in the last %d min)",
        describe_reload(report), report.get("reloads", 1), RELOAD_WINDOW_SEC // 60,
    )
    return report
