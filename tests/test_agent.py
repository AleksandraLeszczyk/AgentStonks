import json
import threading
from types import SimpleNamespace

from marketview.agent import (
    _alert_triggered,
    _dispatch_tool,
    _tool_get_news,
    _tool_get_quote,
    _tool_get_volume_stats,
    _wait_for_next_cycle,
    run_agent_cycle,
)
from marketview.broker import Broker
from marketview.decisions import DecisionTracker
from marketview.state import AppState


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
        outer = self

        class _Completions:
            def create(self, model, messages, tools, tool_choice):
                outer.calls.append(messages)
                return outer._responses.pop(0)

        class _Chat:
            def __init__(self) -> None:
                self.completions = _Completions()

        self.chat = _Chat()


class TestToolHandlers:
    def test_get_volume_stats_with_no_bars_returns_note(self):
        state = AppState()
        assert "note" in _tool_get_volume_stats(state)

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


class TestRunAgentCycle:
    def test_records_buy_decision_and_logs_tool_calls(self):
        state = AppState()
        state.symbol = "AAPL"
        state.api_key = "k"
        state.api_secret = "s"
        state.feed = "iex"
        tracker = DecisionTracker(starting_cash=1000.0, broker=FakeBroker(price=100.0), trade_cost=0.0)

        responses = [
            _response(tool_calls=[_tool_call("c1", "get_daily_bars", {"limit": 60})]),
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
                            "alert_price": 150.0,
                            "alert_condition": "above",
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
        assert decision.alert_price == 150.0
        assert decision.alert_condition == "above"
        assert state.price_alert == {
            "price": 150.0,
            "condition": "above",
            "reasoning": "wait for breakout above resistance",
        }

    def test_alert_with_missing_fields_falls_back_to_sleep(self):
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
        ]
        client = FakeClient(responses)

        run_agent_cycle(client, "gemini-2.0-flash", "AAPL", state, tracker, max_iters=3)

        snap = tracker.snapshot()
        assert snap["decisions"][0].action == "sleep"
        assert state.price_alert is None


class TestAlertTrigger:
    def test_alert_triggered_above(self):
        assert _alert_triggered(151.0, {"price": 150.0, "condition": "above"}) is True
        assert _alert_triggered(149.0, {"price": 150.0, "condition": "above"}) is False

    def test_alert_triggered_below(self):
        assert _alert_triggered(99.0, {"price": 100.0, "condition": "below"}) is True
        assert _alert_triggered(101.0, {"price": 100.0, "condition": "below"}) is False

    def test_wait_returns_early_when_alert_fires(self):
        state = AppState()
        state.last_price = 151.0
        state.price_alert = {"price": 150.0, "condition": "above", "reasoning": "breakout watch"}
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
        assert state.price_alert is None  # cleared once triggered
        with state.lock:
            log_types = [e["type"] for e in state.agent_log]
        assert "status" in log_types

    def test_wait_respects_stop_event_without_alert(self):
        state = AppState()
        stop_event = threading.Event()
        stop_event.set()  # already stopped, should return immediately

        _wait_for_next_cycle(state, stop_event, cycle_sec=60)  # should not hang
