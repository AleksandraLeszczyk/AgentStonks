"""Starting again, after a restart of the app, what was running when it went down."""
import threading
from datetime import datetime, timezone

import pytest

from agent_stonks import clock, session_store, ui
from agent_stonks.apple_trader import AppleTraderConfig
from agent_stonks.decisions import DecisionTracker
from agent_stonks.state import AppState

NOW = datetime(2026, 9, 28, 15, 0, tzinfo=timezone.utc)  # 11:00 ET

RUN_SPEC = {
    "personality": "apple_trader", "provider": "openai", "model": "",
    "symbols": ["AAPL"], "trading_mode": "alpaca_paper", "starting_budget": 25_000.0,
    "apple_config": {"ticker": "AAPL", "buy_k": 0.9, "sell_k": 0.2, "not_a_field": 1},
}


@pytest.fixture(autouse=True)
def frozen_clock(monkeypatch):
    monkeypatch.setattr(clock, "now", lambda: NOW)


@pytest.fixture
def widgets(monkeypatch):
    """The page's session state, as `_resume_after_restart` reads it."""
    values: dict = {}
    monkeypatch.setattr(ui.st, "session_state", values)
    return values


@pytest.fixture
def keys(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET", "s")


class Starts:
    """Stands in for the stream and agent starters, recording their calls."""

    def __init__(self, stream_ok=True, agent_ok=True):
        self.stream_ok, self.agent_ok = stream_ok, agent_ok
        self.streams: list[tuple] = []
        self.agents: list[dict] = []

    def stream(self, state, syms, key, secret, feed, timeframe, **kw):
        self.streams.append((syms, key, secret, feed, timeframe, kw))
        return self.stream_ok

    def agent(self, state, syms, **kw):
        self.agents.append({"syms": syms, **kw})
        if self.agent_ok:
            state.agent_running = True
        return self.agent_ok


@pytest.fixture
def starts(monkeypatch):
    fake = Starts()
    monkeypatch.setattr(ui, "_start_live_session", fake.stream)
    monkeypatch.setattr(ui, "_start_agent", fake.agent)
    return fake


def _saved_mid_run() -> None:
    """Today's file as a process that went down mid-run left it."""
    state = AppState()
    state.decision_tracker = DecisionTracker(starting_cash=25_000.0)
    state.agent_log = [{"type": "status", "text": "armed"}]
    state.set_symbols(["AAPL", "MSFT"])
    state.timeframe, state.feed = "1Min", "iex"
    state.data_source, state.history_feed = "finnhub", "auto"
    state.bars_fallback_stop_event = threading.Event()
    state.run_spec = dict(RUN_SPEC)
    state.agent_running = True
    session_store.claim(state)
    session_store.save(state)


def _after_restart() -> AppState:
    session_store._owners.clear()
    session_store._resumed.clear()
    state = AppState()
    session_store.restore(state)
    return state


def test_the_stream_and_the_agent_start_again_as_they_were(widgets, keys, starts):
    _saved_mid_run()
    state = _after_restart()

    ui._resume_after_restart(state, "", "", "fh-token")

    [(syms, key, secret, feed, timeframe, kw)] = starts.streams
    assert (syms, key, secret, feed, timeframe) == (["AAPL", "MSFT"], "k", "s", "iex", "1Min")
    assert kw == {"data_source": "finnhub", "finnhub_token": "fh-token", "history_feed": "auto"}
    [agent] = starts.agents
    assert agent["syms"] == ["AAPL"]
    assert agent["personality"] == "apple_trader"
    assert agent["trading_mode_choice"] == "alpaca_paper"
    assert agent["starting_budget"] == 25_000.0
    assert agent["continue_today"] is True
    assert agent["apple_config"] == AppleTraderConfig(ticker="AAPL", buy_k=0.9, sell_k=0.2)
    assert agent["data_source"] == "finnhub" and agent["feed"] == "iex"
    assert state.recovery["kind"] == "restart" and not state.recovery["error"]
    assert "started again automatically" in state.agent_log[-1]["text"]

    # Once: a later rerun of the same page starts nothing more.
    ui._resume_after_restart(state, "", "", "fh-token")
    assert len(starts.streams) == 1 and len(starts.agents) == 1


def test_a_stream_that_cannot_start_yet_is_tried_again(widgets, keys, starts, monkeypatch):
    _saved_mid_run()
    state = _after_restart()
    starts.stream_ok = False
    now = [1_000.0]
    monkeypatch.setattr(ui.time, "time", lambda: now[0])

    ui._resume_after_restart(state, "", "", "")
    assert state.recovery["pending"] and state.recovery["attempts"] == 1
    assert starts.agents == []

    now[0] += ui.RESUME_RETRY_SEC - 1          # not due yet
    ui._resume_after_restart(state, "", "", "")
    assert len(starts.streams) == 1

    starts.stream_ok = True
    now[0] += 2
    ui._resume_after_restart(state, "", "", "")
    assert len(starts.streams) == 2 and len(starts.agents) == 1
    assert "pending" not in state.recovery and not state.recovery["error"]


def test_unticked_nothing_is_started(widgets, keys, starts):
    _saved_mid_run()
    state = _after_restart()
    widgets[ui.AUTO_RESUME_KEY] = False

    ui._resume_after_restart(state, "", "", "")

    assert starts.streams == [] and starts.agents == []
    assert state.recovery is None


def test_unticking_drops_a_pending_retry(widgets, keys, starts):
    _saved_mid_run()
    state = _after_restart()
    starts.stream_ok = False
    ui._resume_after_restart(state, "", "", "")
    assert state.recovery["pending"]

    widgets[ui.AUTO_RESUME_KEY] = False
    ui._resume_after_restart(state, "", "", "")
    assert state.recovery is None


def test_without_credentials_it_says_so_and_starts_nothing(widgets, starts, monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET", raising=False)
    _saved_mid_run()
    state = _after_restart()

    ui._resume_after_restart(state, "", "", "")

    assert starts.streams == [] and starts.agents == []
    assert "ALPACA_API_KEY" in state.recovery["error"]


def test_an_agent_the_setup_refuses_is_reported_not_retried(widgets, keys, starts):
    _saved_mid_run()
    state = _after_restart()
    starts.agent_ok = False

    ui._resume_after_restart(state, "", "", "")

    assert state.recovery["error"] and "pending" not in state.recovery
    ui._resume_after_restart(state, "", "", "")
    assert len(starts.agents) == 1


def test_a_day_saved_while_stopped_starts_nothing(widgets, keys, starts):
    state = AppState()
    state.decision_tracker = DecisionTracker(starting_cash=1.0)
    session_store.claim(state)
    session_store.save(state)

    ui._resume_after_restart(_after_restart(), "", "", "")

    assert starts.streams == [] and starts.agents == []


def test_a_failing_tab_is_contained(monkeypatch):
    shown = []
    monkeypatch.setattr(ui.st, "warning", lambda text: shown.append(text))
    monkeypatch.setattr(ui.st, "expander", lambda *_: _Nothing())
    monkeypatch.setattr(ui.st, "exception", lambda exc: None)

    with ui._panel_guard("The News tab"):
        raise ConnectionError("HTTPSConnectionPool: Max retries exceeded")

    assert "The News tab" in shown[0] and "ConnectionError" in shown[0]


def test_streamlit_control_flow_passes_through_the_guard():
    from streamlit.runtime.scriptrunner_utils.exceptions import RerunException

    with pytest.raises(RerunException):
        with ui._panel_guard("The Live tab"):
            raise RerunException(None)


class _Nothing:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_a_pending_resume_can_be_taken_over_by_a_reload(widgets, keys, starts, monkeypatch):
    # The run is handed out once per process; a reload of the page must find
    # the state still trying, not a fresh one with nothing left to resume.
    from agent_stonks import stream

    monkeypatch.setattr(stream, "_live_sessions", {})
    monkeypatch.setattr(stream, "_gone_since", {})
    monkeypatch.setattr(ui, "_session_id", lambda: "before-reload")
    _saved_mid_run()
    state = _after_restart()
    starts.stream_ok = False

    ui._resume_after_restart(state, "", "", "")

    assert stream.adopt_orphaned_session("after-reload", lambda sid: sid == "after-reload") is state
    assert state.recovery["pending"]
