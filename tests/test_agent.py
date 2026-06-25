import json
import threading
from types import SimpleNamespace

from marketview.agent import (
    BREAKOUT_TOOLS,
    MOMENTUM_TOOLS,
    PERSONALITY_TOOLS,
    _dispatch_tool,
    _tool_analyze_volume,
    _tool_breakout_trade_geometry,
    _tool_get_news,
    _tool_get_quote,
    _wait_for_next_cycle,
    run_agent_cycle,
)
from marketview.broker import Broker
from marketview.decisions import DecisionTracker
from marketview.state import AppState, alert_triggered


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
    """Returns canned chat-completion responses in sequence."""

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


class TestToolHandlers:
    def test_analyze_volume_with_no_bars_returns_note(self):
        state = AppState()
        assert "note" in _tool_analyze_volume(state)

    def test_get_quote_reads_state(self):
        state = AppState()
        state.last_price = 101.0
        state.bid_price = 100.5
        result = _tool_get_quote(state)
        assert result["last_price"] == 101.0
        assert result["bid_price"] == 100.5

    def test_get_news_maps_impact_labels(self):
        state = AppState()
        state.news = [{"id": "1", "headline": "h", "summary": "s", "created_at": "t", "source": "src"}]
        state.news_impacts = {"1": "positive"}
        result = _tool_get_news(state)
        assert result["articles"][0]["impact"] == "positive"

    def test_dispatch_unknown_tool_returns_error(self):
        state = AppState()
        tracker = DecisionTracker()
        result = _dispatch_tool("nonexistent", {}, state, tracker)
        assert "error" in result

    def test_breakout_trade_geometry_tool_computes_targets(self):
        state = AppState()
        result = _tool_breakout_trade_geometry(state, entry=100.0, stop=98.0, atr=4.0)
        assert result["meets_min_reward_risk"] is True

    def test_breakout_personality_uses_breakout_tools(self):
        assert PERSONALITY_TOOLS["breakout"] is BREAKOUT_TOOLS
        names = {t["function"]["name"] for t in BREAKOUT_TOOLS}
        assert {"analyze_opening_range", "analyze_volume", "breakout_trade_geometry"} <= names
        assert "analyze_daily_trend" not in names
        assert "analyze_market" not in names
        assert "get_put_call_walls" not in names

    def test_momentum_personality_uses_momentum_tools(self):
        assert PERSONALITY_TOOLS["momentum"] is MOMENTUM_TOOLS
        names = {t["function"]["name"] for t in MOMENTUM_TOOLS}
        assert {"analyze_intraday_momentum", "analyze_volume", "get_news", "get_quote"} <= names
        assert "analyze_daily_trend" not in names
        assert "analyze_market" not in names
        assert "analyze_opening_range" not in names
        assert "get_put_call_walls" not in names


class TestRunAgentCycle:
    def test_records_buy_decision_and_logs_tool_calls(self):
        state = AppState()
        state.symbol = "AAPL"
        state.api_key = "k"
        state.api_secret = "s"
        state.feed = "iex"
        tracker = DecisionTracker(starting_cash=1000.0, broker=FakeBroker(price=100.0), trade_cost=0.0)

        responses = [
            _response(tool_calls=[_tool_call("c1", "analyze_daily_trend", {"limit": 60})]),
            _response(
                tool_calls=[
                    _tool_call(
                        "c2",
                        "submit_decision",
                        {"action": "buy", "quantity": 2, "regime": "bullish", "reasoning": "uptrend confirmed"},
                    )
                ]
            ),
        ]
        client = FakeClient(responses)

        run_agent_cycle(client, "gemini-2.0-flash", "AAPL", state, tracker, max_iters=5)

        snap = tracker.snapshot()
        assert snap["position"] == 2.0
        assert snap["cash"] == 800.0
        assert len(snap["decisions"]) == 1
        assert snap["decisions"][0].action == "buy"

        with state.lock:
            log_types = [e["type"] for e in state.agent_log]
        assert "tool_call" in log_types
        assert "decision" in log_types

    def test_breakout_personality_passes_breakout_tools_to_client(self):
        state = AppState()
        state.symbol = "AAPL"
        state.api_key = "k"
        state.api_secret = "s"
        tracker = DecisionTracker(broker=FakeBroker())

        responses = [
            _response(
                tool_calls=[_tool_call("c1", "submit_decision", {"action": "sleep", "reasoning": "no setup yet"})]
            )
        ]
        client = FakeClient(responses)

        run_agent_cycle(client, "gemini-2.0-flash", "AAPL", state, tracker, max_iters=3, personality="breakout")

        assert client.tools_seen[0] is BREAKOUT_TOOLS

    def test_forces_sleep_when_max_iters_reached_without_decision(self):
        state = AppState()
        state.symbol = "AAPL"
        state.api_key = "k"
        state.api_secret = "s"
        tracker = DecisionTracker(broker=FakeBroker())

        responses = [_response(content="still thinking...") for _ in range(3)]
        client = FakeClient(responses)

        run_agent_cycle(client, "gemini-2.0-flash", "AAPL", state, tracker, max_iters=3)

        snap = tracker.snapshot()
        assert len(snap["decisions"]) == 1
        assert snap["decisions"][0].action == "sleep"

    def test_sleep_decision_from_model_records_no_price(self):
        state = AppState()
        state.symbol = "AAPL"
        state.api_key = "k"
        state.api_secret = "s"
        tracker = DecisionTracker(broker=FakeBroker())

        responses = [
            _response(
                tool_calls=[_tool_call("c1", "submit_decision", {"action": "sleep", "reasoning": "no clear edge"})]
            )
        ]
        client = FakeClient(responses)

        run_agent_cycle(client, "gemini-2.0-flash", "AAPL", state, tracker, max_iters=3)

        snap = tracker.snapshot()
        assert len(snap["decisions"]) == 1
        assert snap["decisions"][0].action == "sleep"
        assert snap["decisions"][0].price is None

    def test_alert_decision_sets_state_price_alert_and_records_no_trade(self):
        state = AppState()
        state.symbol = "AAPL"
        state.api_key = "k"
        state.api_secret = "s"
        tracker = DecisionTracker(broker=FakeBroker())

        responses = [
            _response(
                tool_calls=[
                    _tool_call(
                        "c1",
                        "submit_decision",
                        {
                            "action": "alert",
                            "reasoning": "wait for breakout above resistance",
                            "alert_high_price": 150.0,
                        },
                    )
                ]
            )
        ]
        client = FakeClient(responses)

        run_agent_cycle(client, "gemini-2.0-flash", "AAPL", state, tracker, max_iters=3)

        snap = tracker.snapshot()
        assert len(snap["decisions"]) == 1
        decision = snap["decisions"][0]
        assert decision.action == "alert"
        assert decision.status == "noop"
        assert decision.price is None
        assert decision.alerts == [{"price": 150.0, "condition": "above"}]
        assert state.price_alerts == [{"price": 150.0, "condition": "above"}]

    def test_alert_decision_supports_both_low_and_high_levels(self):
        state = AppState()
        state.symbol = "AAPL"
        state.api_key = "k"
        state.api_secret = "s"
        tracker = DecisionTracker(broker=FakeBroker())

        responses = [
            _response(
                tool_calls=[
                    _tool_call(
                        "c1",
                        "submit_decision",
                        {
                            "action": "alert",
                            "reasoning": "watching a breakout above resistance or a breakdown below support",
                            "alert_low_price": 95.0,
                            "alert_high_price": 150.0,
                        },
                    )
                ]
            )
        ]
        client = FakeClient(responses)

        run_agent_cycle(client, "gemini-2.0-flash", "AAPL", state, tracker, max_iters=3)

        snap = tracker.snapshot()
        decision = snap["decisions"][0]
        assert decision.action == "alert"
        assert decision.alerts == [
            {"price": 95.0, "condition": "below"},
            {"price": 150.0, "condition": "above"},
        ]
        assert state.price_alerts == decision.alerts

    def test_alert_with_missing_fields_is_rejected_and_retried(self):
        """An incomplete alert call (no price levels) must not be silently downgraded to
        sleep -- the model gets an error back and a chance to correct itself, since it
        otherwise has no way to know its alert was dropped."""
        state = AppState()
        state.symbol = "AAPL"
        state.api_key = "k"
        state.api_secret = "s"
        tracker = DecisionTracker(broker=FakeBroker())

        responses = [
            _response(
                tool_calls=[
                    _tool_call("c1", "submit_decision", {"action": "alert", "reasoning": "no level chosen"})
                ]
            ),
            _response(
                tool_calls=[
                    _tool_call(
                        "c2",
                        "submit_decision",
                        {
                            "action": "alert",
                            "reasoning": "watching for a breakout above resistance",
                            "alert_high_price": 150.0,
                        },
                    )
                ]
            ),
        ]
        client = FakeClient(responses)

        run_agent_cycle(client, "gpt-4.1-mini", "AAPL", state, tracker, max_iters=3)

        snap = tracker.snapshot()
        assert len(snap["decisions"]) == 1
        assert snap["decisions"][0].action == "alert"
        assert snap["decisions"][0].alerts == [{"price": 150.0, "condition": "above"}]
        assert state.price_alerts == [{"price": 150.0, "condition": "above"}]

        # the rejected first attempt must have been surfaced back to the model as a tool result
        first_call_messages = client.calls[1]
        tool_results = [m["content"] for m in first_call_messages if m.get("role") == "tool"]
        assert any("requires alert_low_price" in c for c in tool_results)

    def test_alert_with_missing_fields_falls_back_to_sleep_if_never_corrected(self):
        state = AppState()
        state.symbol = "AAPL"
        state.api_key = "k"
        state.api_secret = "s"
        tracker = DecisionTracker(broker=FakeBroker())

        responses = [
            _response(
                tool_calls=[
                    _tool_call("c1", "submit_decision", {"action": "alert", "reasoning": "no level chosen"})
                ]
            )
            for _ in range(3)
        ]
        client = FakeClient(responses)

        run_agent_cycle(client, "gpt-4.1-mini", "AAPL", state, tracker, max_iters=3)

        snap = tracker.snapshot()
        assert len(snap["decisions"]) == 1
        assert snap["decisions"][0].action == "sleep"
        assert state.price_alerts == []


class TestAlertTrigger:
    def test_alert_triggered_above(self):
        assert alert_triggered(151.0, {"price": 150.0, "condition": "above"}) is True
        assert alert_triggered(149.0, {"price": 150.0, "condition": "above"}) is False

    def test_alert_triggered_below(self):
        assert alert_triggered(99.0, {"price": 100.0, "condition": "below"}) is True
        assert alert_triggered(101.0, {"price": 100.0, "condition": "below"}) is False

    def test_wait_returns_early_when_alert_fires(self):
        state = AppState()
        state.last_price = 151.0
        state.price_alerts = [{"price": 150.0, "condition": "above"}]
        stop_event = threading.Event()

        start = threading.Event()
        finished = threading.Event()

        def run():
            start.set()
            _wait_for_next_cycle(state, stop_event, cycle_sec=60)
            finished.set()

        thread = threading.Thread(target=run)
        thread.start()
        thread.join(timeout=5)

        assert finished.is_set()
        assert state.price_alerts == []  # cleared once triggered
        with state.lock:
            log_types = [e["type"] for e in state.agent_log]
        assert "status" in log_types

    def test_wait_returns_early_when_low_side_of_bracket_fires(self):
        state = AppState()
        state.last_price = 94.0
        state.price_alerts = [
            {"price": 95.0, "condition": "below"},
            {"price": 150.0, "condition": "above"},
        ]
        stop_event = threading.Event()

        start = threading.Event()
        finished = threading.Event()

        def run():
            start.set()
            _wait_for_next_cycle(state, stop_event, cycle_sec=60)
            finished.set()

        thread = threading.Thread(target=run)
        thread.start()
        thread.join(timeout=5)

        assert finished.is_set()
        assert state.price_alerts == []  # both levels cleared once either fires
        with state.lock:
            log_types = [e["type"] for e in state.agent_log]
        assert "status" in log_types

    def test_wait_returns_early_when_woken_externally(self):
        """The actual news/alert detection now lives in the stream callbacks (see
        stream.py), which signal `agent_wake_event` directly. This simulates that
        external signal to verify `_wait_for_next_cycle` is a real, event-driven
        block rather than a self-polling loop."""
        state = AppState()
        stop_event = threading.Event()

        def signal_after_start():
            start.wait()
            state.agent_wake_reason = "Fresh news arrived for the ticker."
            state.agent_wake_event.set()

        start = threading.Event()
        finished = threading.Event()

        def run():
            start.set()
            _wait_for_next_cycle(state, stop_event, cycle_sec=60)
            finished.set()

        signaler = threading.Thread(target=signal_after_start)
        thread = threading.Thread(target=run)
        thread.start()
        signaler.start()
        thread.join(timeout=5)
        signaler.join(timeout=5)

        assert finished.is_set()
        with state.lock:
            log_types = [e["type"] for e in state.agent_log]
        assert "status" in log_types

    def test_wait_respects_stop_event_without_alert(self):
        state = AppState()
        stop_event = threading.Event()
        stop_event.set()  # already stopped, should return immediately

        _wait_for_next_cycle(state, stop_event, cycle_sec=60)  # should not hang
