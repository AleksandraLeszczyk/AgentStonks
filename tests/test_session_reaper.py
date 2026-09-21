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
