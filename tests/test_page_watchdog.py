"""The page bringing itself back after Streamlit's frontend dies (2026-10-02,
"Cached ForwardMsg MISS"), and the message cache that error came from."""
import logging
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
from streamlit.runtime import forward_msg_cache

from agent_stonks import page_watchdog, session_store, ui
from agent_stonks.state import AppState

CONFIG = Path(__file__).resolve().parent.parent / ".streamlit" / "config.toml"

REPORT = {
    "at": 1_790_958_500_782,
    "reason": "lost its connection to the server (error for 30 s)",
    "detail": "Failed to process a Websocket message. Error: Cached ForwardMsg MISS [hash=00].",
    "reloads": 1,
}


def _big_element() -> ForwardMsg:
    msg = ForwardMsg()
    msg.delta.new_element.markdown.body = "x" * 50_000
    return msg


def _cacheable_with(monkeypatch, min_size) -> bool:
    real = forward_msg_cache.config.get_option
    monkeypatch.setattr(
        forward_msg_cache.config, "get_option",
        lambda key: min_size if key == "global.minCachedMessageSize" else real(key),
    )
    msg = _big_element()
    forward_msg_cache.populate_hash_if_needed(msg)
    return msg.metadata.cacheable


def test_the_project_config_turns_off_the_message_cache(monkeypatch):
    # Nothing cacheable means the server never sends a reference the browser
    # may already have dropped -- the MISS that killed the page.
    min_size = tomllib.loads(CONFIG.read_text())["global"]["minCachedMessageSize"]
    assert _cacheable_with(monkeypatch, 10e3), "Streamlit's default caches it"
    assert not _cacheable_with(monkeypatch, min_size)
    int(min_size)  # Streamlit int()s it: inf would break every message


def test_the_health_route_follows_the_base_url_path():
    assert page_watchdog.health_url("") == "/_stcore/health"
    assert page_watchdog.health_url("/desk/") == "/desk/_stcore/health"


def test_a_background_tab_is_not_reloaded_for_being_throttled():
    # Chrome runs a background tab's timers once a minute: the heartbeat and
    # the check can each be a minute late.
    cfg = page_watchdog.watchdog_config()
    assert cfg["silent_hidden_ms"] > 2 * 60_000
    assert cfg["silent_ms"] >= 10 * page_watchdog.WATCHDOG_BEAT_SEC * 1000
    assert cfg["silent_running_ms"] >= cfg["silent_ms"]


def test_a_reload_report_is_logged_once_and_returned(monkeypatch, caplog):
    renders = []

    def mount(**kwargs):
        renders.append(kwargs)
        return SimpleNamespace(recovered=REPORT)

    monkeypatch.setattr(page_watchdog, "_WATCHDOG", mount)
    with caplog.at_level(logging.WARNING, logger="agent_stonks.page_watchdog"):
        assert page_watchdog.page_watchdog() == REPORT
    assert "Cached ForwardMsg MISS" in caplog.text
    assert "lost its connection" in caplog.text
    # Every render is a heartbeat: the data has to change for the page to see it.
    assert "beat" in renders[0]["data"] and "config" in renders[0]["data"]


def test_no_report_is_no_log(monkeypatch, caplog):
    monkeypatch.setattr(
        page_watchdog, "_WATCHDOG", lambda **_: SimpleNamespace(recovered=None)
    )
    with caplog.at_level(logging.WARNING, logger="agent_stonks.page_watchdog"):
        assert page_watchdog.page_watchdog() is None
    assert not caplog.text


@pytest.fixture
def shown(monkeypatch):
    out: list[str] = []
    monkeypatch.setattr(ui.st, "info", lambda text: out.append(text))
    return out


def test_the_banner_says_why_the_page_reloaded(shown, monkeypatch):
    state = AppState()
    state.agent_running = True
    monkeypatch.setattr(session_store, "is_streaming", lambda _: True)
    state.recovery = {"kind": "reconnect", "at": ui.time.time(), "reload": REPORT}

    ui._recovery_banner(state)

    assert "reloaded itself because it lost its connection" in shown[0]
    assert "Cached ForwardMsg MISS" in shown[0]
    assert "never stopped" in shown[0]


def test_a_reload_with_nothing_running_still_says_so(shown, monkeypatch):
    monkeypatch.setattr(session_store, "is_streaming", lambda _: False)
    state = AppState()
    state.recovery = {"kind": "reconnect", "at": ui.time.time(), "reload": REPORT}

    ui._recovery_banner(state)

    assert "reloaded itself" in shown[0]


@pytest.mark.parametrize("before", [
    None,
    {"kind": "reconnect", "at": 1.0},
    {"kind": "restart", "at": 1.0, "pending": {"stream": {}}, "attempts": 2},
])
def test_a_report_joins_the_recovery_without_dropping_it(before, monkeypatch):
    # A reload in the middle of a pending resume must not end the retries.
    state = AppState()
    state.recovery = dict(before) if before else None
    reruns = []
    monkeypatch.setattr(ui, "_get_state", lambda: state)
    monkeypatch.setattr(ui, "page_watchdog", lambda: REPORT)
    monkeypatch.setattr(ui.st, "rerun", lambda **kw: reruns.append(kw))

    ui._page_watchdog.__wrapped__()

    assert state.recovery["reload"] == REPORT
    assert state.recovery["kind"] == (before or {"kind": "reconnect"})["kind"]
    if before and before.get("pending"):
        assert state.recovery["pending"] == before["pending"]
    assert reruns == [{"scope": "app"}]  # the banner is drawn by the full run
