import json
from types import SimpleNamespace

from marketview.agent import _dispatch_tool, _tool_get_news, _tool_get_quote, _tool_get_volume_stats, run_agent_cycle
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
