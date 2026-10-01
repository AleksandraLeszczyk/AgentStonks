import threading

import pytest

from agent_stonks import stream
from agent_stonks.state import AppState


class _Closable:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def _live_app() -> AppState:
    app = AppState()
    app.ws, app.ws_news = _Closable(), _Closable()
    app.bars_fallback_stop_event = threading.Event()
    app.news_fallback_stop_event = threading.Event()
    app.bars_connected = app.news_connected = True
    return app


@pytest.fixture(autouse=True)
def _clean_registry():
    stream._live_sessions.clear()
    stream._gone_since.clear()
    yield
    stream._live_sessions.clear()
    stream._gone_since.clear()


def test_stop_streams_closes_both_sockets_and_both_fallbacks():
    app = _live_app()

    stream.stop_streams(app)

    assert app.ws.closed and app.ws_news.closed
    assert app.bars_fallback_stop_event.is_set()
    assert app.news_fallback_stop_event.is_set()
    assert (app.status, app.bars_connected, app.news_connected) == ("Stopped", False, False)


def test_a_session_gone_past_the_grace_period_is_stopped():
    app = _live_app()
    stream.register_live_session("s1", app)

    assert stream.reap_dead_sessions(lambda sid: False, now=0.0) == []
    assert not app.ws.closed
    reaped = stream.reap_dead_sessions(
        lambda sid: False, now=stream.SESSION_REAP_AFTER_SEC
    )

    assert reaped == ["s1"]
    assert app.ws.closed and app.bars_fallback_stop_event.is_set()
    assert stream._live_sessions == {}


def test_a_session_that_comes_back_is_left_alone():
    # A laptop sleep or network blip: gone for a while, then reconnected.
    app = _live_app()
    stream.register_live_session("s1", app)
    stream.reap_dead_sessions(lambda sid: False, now=0.0)
    stream.reap_dead_sessions(lambda sid: True, now=100.0)

    reaped = stream.reap_dead_sessions(
        lambda sid: False, now=stream.SESSION_REAP_AFTER_SEC + 1
    )

    assert reaped == []
    assert not app.ws.closed


def test_an_active_session_is_never_stopped():
    app = _live_app()
    stream.register_live_session("s1", app)

    assert stream.reap_dead_sessions(lambda sid: True, now=10_000.0) == []
    assert not app.ws.closed


def test_a_session_whose_agent_still_trades_keeps_its_streams():
    # The agent reads the streams; stopping them under it would leave a
    # position on a tape that no longer moves, with its stop unable to fire.
    app = _live_app()
    app.agent_running = True
    stream.register_live_session("s1", app)
    stream.reap_dead_sessions(lambda sid: False, now=0.0)

    assert stream.reap_dead_sessions(lambda sid: False, now=10_000.0) == []
    assert not app.ws.closed

    app.agent_running = False  # the agent stopped; nobody came back
    assert stream.reap_dead_sessions(lambda sid: False, now=10_001.0) == ["s1"]
    assert app.ws.closed


def test_a_new_session_takes_over_a_running_one_whose_browser_went():
    app = _live_app()
    app.agent_running = True
    stream.register_live_session("old", app)
    stream.reap_dead_sessions(lambda sid: False, now=0.0)

    adopted = stream.adopt_orphaned_session("new", lambda sid: sid == "new")

    assert adopted is app
    assert stream._live_sessions == {"new": app}
    assert stream._gone_since == {}
    assert not app.ws.closed


def test_a_session_still_connected_is_not_taken_over():
    # A second tab opened beside a live one keeps a state of its own.
    app = _live_app()
    stream.register_live_session("first", app)

    assert stream.adopt_orphaned_session("second", lambda sid: True) is None
    assert stream._live_sessions == {"first": app}


def test_a_running_agent_is_taken_over_before_a_newer_idle_stream():
    trading, idle = _live_app(), _live_app()
    trading.agent_running = True
    stream.register_live_session("trading", trading)
    stream.register_live_session("idle", idle)

    assert stream.adopt_orphaned_session("new", lambda sid: sid == "new") is trading
    assert stream._live_sessions == {"idle": idle, "new": trading}
    # The next session gets the other one.
    assert stream.adopt_orphaned_session("newer", lambda sid: sid in ("new", "newer")) is idle


def test_nothing_to_take_over_gives_none():
    assert stream.adopt_orphaned_session("new", lambda sid: sid == "new") is None


def test_a_session_already_registered_gets_its_own_state_back():
    app = _live_app()
    stream.register_live_session("s1", app)

    assert stream.adopt_orphaned_session("s1", lambda sid: True) is app
