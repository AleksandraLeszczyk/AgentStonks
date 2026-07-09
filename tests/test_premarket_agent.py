"""Tests for the Premarket Analyst personality: the pre-open window gate, the
analyze_premarket tool, the one-shot session driver, and the Automatic
orchestrator routing to it while the session hasn't started."""
import json
import threading
import time
from datetime import datetime, timezone
from types import SimpleNamespace

from agent_stonks import agent as agent_module
from agent_stonks import automatic, market_hours
from agent_stonks.agent import (
    AGENT_PERSONALITIES,
    PREMARKET_PERSONALITY,
    PREMARKET_TOOLS,
    _tool_analyze_premarket,
    _wait_for_premarket_window,
    run_agent_cycle,
    run_premarket_session,
)
from agent_stonks.automatic import SELECTABLE_STRATEGIES
from agent_stonks.broker import Broker
from agent_stonks.decisions import DecisionTracker
from agent_stonks.state import AppState


class FakeBroker(Broker):
    def __init__(self, price: float = 100.0):
        self.price = price

    def get_current_price(self, symbol, key, secret, feed="iex") -> float:
        return self.price

    def submit_order(self, symbol, side, quantity, price) -> dict:
        return {"status": "filled", "filled_qty": quantity, "filled_price": price}


def _tool_call(call_id: str, name: str, arguments: dict) -> SimpleNamespace:
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=json.dumps(arguments)))


def _response(content: str | None = None, tool_calls: list | None = None) -> SimpleNamespace:
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeClient:
    def __init__(self, responses: list):
        self._responses = list(responses)
        self.calls: list = []
        self.tools_seen: list = []
        outer = self

        class _Completions:
            def create(self, model, messages, tools, tool_choice):
                outer.calls.append(messages)
                outer.tools_seen.append(tools)
                return outer._responses.pop(0)

        class _Chat:
            def __init__(self) -> None:
                self.completions = _Completions()

        self.chat = _Chat()


def _base_state() -> AppState:
    state = AppState()
    state.set_symbols(["AAPL"])
    state.api_key = "k"
    state.api_secret = "s"
    state.feed = "iex"
    return state


class TestPremarketRegistration:
    def test_personality_registered_with_tools(self):
        assert PREMARKET_PERSONALITY in AGENT_PERSONALITIES
        names = {t["function"]["name"] for t in PREMARKET_TOOLS}
        assert {"analyze_premarket", "get_quote", "get_news", "get_position",
                "set_tactics", "submit_decision"} <= names

    def test_not_selectable_by_the_regime_cycle(self):
        assert PREMARKET_PERSONALITY not in SELECTABLE_STRATEGIES
        assert "momentum" in SELECTABLE_STRATEGIES

    def test_cycle_uses_premarket_tools(self):
        state = _base_state()
        tracker = DecisionTracker(broker=FakeBroker())
        responses = [
            _response(
                tool_calls=[
                    _tool_call(
                        "c1",
                        "submit_decision",
                        {
                            "action": "alert",
                            "reasoning": "thin evidence",
                            "alerts": [{"field": "last_price", "condition": "above", "value": 1.0}],
                        },
                    )
                ]
            )
        ]
        client = FakeClient(responses)
        run_agent_cycle(client, "m", ["AAPL"], state, tracker, max_iters=3, personality=PREMARKET_PERSONALITY)
        names = {t["function"]["name"] for t in client.tools_seen[0]}
        assert "analyze_premarket" in names
        assert "stand_down" not in names


class TestAnalyzePremarketTool:
    def test_reads_gap_and_premarket_bars(self, monkeypatch):
        open_dt = datetime(2026, 7, 6, 13, 30, tzinfo=timezone.utc)  # 09:30 ET
        monkeypatch.setattr(market_hours, "session_open", lambda now=None: None)
        monkeypatch.setattr(market_hours, "next_market_open", lambda now=None: open_dt)
        monkeypatch.setattr(market_hours, "is_market_open", lambda now=None: False)
        state = _base_state().sym("AAPL")
        state.prev_close = 100.0
        state.last_price = 105.0
        # Prior day's bar is excluded; the same-day pre-open bar counts.
        state.bars.append({"t": "2026-07-02T13:00:00Z", "o": 99, "h": 100, "l": 98, "c": 99, "v": 900})
        state.bars.append({"t": "2026-07-06T13:00:00Z", "o": 104, "h": 106, "l": 103, "c": 105, "v": 5000})

        result = _tool_analyze_premarket(state)
        assert result["implied_gap_pct"] == 5.0
        assert result["market_is_open"] is False
        pre = result["premarket_session"]
        assert pre["bars"] == 1
        assert pre["high"] == 106.0
        assert pre["low"] == 103.0
        assert pre["volume"] == 5000.0

    def test_no_bars_yet_returns_note(self, monkeypatch):
        monkeypatch.setattr(market_hours, "session_open", lambda now=None: None)
        monkeypatch.setattr(
            market_hours,
            "next_market_open",
            lambda now=None: datetime(2026, 7, 6, 13, 30, tzinfo=timezone.utc),
        )
        state = _base_state().sym("AAPL")
        result = _tool_analyze_premarket(state)
        assert "note" in result["premarket_session"]


class TestPremarketWindow:
    def test_holds_until_lead_window(self, monkeypatch):
        remaining = iter([500.0, 60.0])
        monkeypatch.setattr(market_hours, "seconds_until_next_open", lambda now=None: next(remaining))
        monkeypatch.setattr(agent_module, "PREMARKET_WAIT_POLL_SEC", 0.01)
        state = _base_state()
        assert _wait_for_premarket_window(state, threading.Event()) is True
        with state.lock:
            texts = [e.get("text", "") for e in state.agent_log]
        assert any("holding until" in t for t in texts)

    def test_stop_while_holding_returns_false(self, monkeypatch):
        monkeypatch.setattr(market_hours, "seconds_until_next_open", lambda now=None: 10_000.0)
        monkeypatch.setattr(agent_module, "PREMARKET_WAIT_POLL_SEC", 0.01)
        state = _base_state()
        stop = threading.Event()
        threading.Timer(0.05, stop.set).start()
        assert _wait_for_premarket_window(state, stop) is False


class TestRunPremarketSession:
    def _tactics_cycle_responses(self) -> list:
        return [
            _response(
                tool_calls=[
                    _tool_call(
                        "c1",
                        "set_tactics",
                        {
                            "actions": [
                                {
                                    "action": "buy",
                                    "quantity": 5,
                                    "conditions": [
                                        {"field": "last_price", "condition": "below", "value": 99.0}
                                    ],
                                    "note": "opening entry",
                                }
                            ],
                            "reasoning": "gap should pull back before following through",
                        },
                    )
                ]
            ),
            _response(
                tool_calls=[
                    _tool_call(
                        "c2",
                        "submit_decision",
                        {"action": "alert", "reasoning": "opening bracket armed", "alerts": []},
                    )
                ]
            ),
        ]

    def test_retires_after_tactic_executes(self, monkeypatch):
        monkeypatch.setattr(agent_module, "_wait_for_premarket_window", lambda state, stop: True)
        state = _base_state()
        tracker = DecisionTracker(broker=FakeBroker())
        client = FakeClient(self._tactics_cycle_responses())
        stop = threading.Event()
        result: list = []

        thread = threading.Thread(
            target=lambda: result.append(
                run_premarket_session(client, "m", ["AAPL"], state, tracker, stop)
            )
        )
        thread.start()
        deadline = time.monotonic() + 5
        while not state.any_tactics() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert state.any_tactics()
        # Simulate the TacticsExecutor performing the opening trade.
        state.sym("AAPL").tactics = None
        state.agent_wake_reason = "Tactics executed: buy 5 sh -> filled."
        state.agent_wake_event.set()
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert result == ["executed"]
        with state.lock:
            texts = [e.get("text", "") for e in state.agent_log]
        assert any("retiring" in t for t in texts)

    def test_done_when_nothing_armed_at_the_bell(self, monkeypatch):
        monkeypatch.setattr(agent_module, "_wait_for_premarket_window", lambda state, stop: True)
        monkeypatch.setattr(market_hours, "is_market_open", lambda now=None: True)
        state = _base_state()
        tracker = DecisionTracker(broker=FakeBroker())
        responses = [
            _response(
                tool_calls=[
                    _tool_call(
                        "c1",
                        "submit_decision",
                        {
                            "action": "alert",
                            "reasoning": "no gap, no catalyst",
                            "alerts": [{"field": "last_price", "condition": "above", "value": 200.0}],
                        },
                    )
                ]
            )
        ]
        client = FakeClient(responses)
        assert run_premarket_session(client, "m", ["AAPL"], state, tracker, threading.Event()) == "done"

    def test_stopped_while_waiting_for_the_window(self):
        state = _base_state()
        tracker = DecisionTracker(broker=FakeBroker())
        stop = threading.Event()
        stop.set()
        client = FakeClient([])
        assert run_premarket_session(client, "m", ["AAPL"], state, tracker, stop) == "stopped"


class TestAutomaticPremarketRouting:
    def test_activates_premarket_when_session_not_started(self, monkeypatch):
        state = _base_state()
        tracker = DecisionTracker(broker=FakeBroker())
        seen: dict = {}

        def fake_session(client, model, symbols, st, tr, stop_event):
            seen["active_strategy"] = st.automatic_active_strategy
            seen["regime"] = st.automatic_regime
            stop_event.set()
            return "stopped"

        monkeypatch.setattr(market_hours, "is_market_open", lambda now=None: False)
        monkeypatch.setattr(automatic, "run_premarket_session", fake_session)
        monkeypatch.setattr(automatic, "get_agent_client", lambda provider, api_key: object())

        automatic._automatic_loop(
            state, tracker, ["AAPL"], "openai", "key", "model", 1, threading.Event()
        )
        assert seen["active_strategy"] == PREMARKET_PERSONALITY
        assert seen["regime"] == "premarket"
        assert state.automatic_active_strategy is None
        with state.lock:
            texts = [e.get("text", "") for e in state.agent_log]
        assert any("session hasn't started" in t for t in texts)
