import json
import threading
from types import SimpleNamespace

from marketview.agent import (
    BREAKOUT_TOOLS,
    MOMENTUM_TOOLS,
    PERSONALITY_TOOLS,
    REVERSAL_TOOLS,
    SMART_MONEY_TOOLS,
    _dispatch_tool,
    _tool_analyze_volume,
    _tool_breakout_trade_geometry,
    _tool_get_news,
    _tool_get_quote,
    _tool_smart_money_trade_geometry,
    _tool_vwap_reversion_geometry,
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

    def test_get_quote_tight_fresh_quote_has_no_warning(self):
        from datetime import datetime, timezone

        state = AppState()
        state.last_price = 100.01
        state.bid_price, state.ask_price = 100.0, 100.02
        state.quote_ts = datetime.now(timezone.utc).isoformat()
        result = _tool_get_quote(state)
        assert result["spread"] == 0.02
        assert result["quote_age_sec"] < 5
        assert "warning" not in result

    def test_get_quote_flags_placeholder_wide_quote(self):
        # Real off-hours IEX quote: ~10% wide around the mid, useless as a price.
        state = AppState()
        state.last_price = 288.64
        state.bid_price, state.ask_price = 274.46, 305.29
        result = _tool_get_quote(state)
        assert result["spread_pct"] > 10
        assert "use last_price" in result["warning"]

    def test_get_quote_flags_stale_quote(self):
        state = AppState()
        state.bid_price, state.ask_price = 100.0, 100.02
        state.quote_ts = "2020-01-01T00:00:00Z"
        result = _tool_get_quote(state)
        assert result["quote_age_sec"] > 3600
        assert "stale" in result["warning"]

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

    def test_reversal_personality_uses_reversal_tools(self):
        assert PERSONALITY_TOOLS["reversal"] is REVERSAL_TOOLS
        names = {t["function"]["name"] for t in REVERSAL_TOOLS}
        assert {"analyze_vwap_bands", "vwap_reversion_geometry", "analyze_volume", "get_quote"} <= names
        assert "analyze_daily_trend" not in names
        assert "analyze_opening_range" not in names
        assert "get_put_call_walls" not in names

    def test_vwap_reversion_geometry_tool_computes_reward_risk(self):
        state = AppState()
        result = _tool_vwap_reversion_geometry(state, entry=98.0, vwap=100.0, std_dev=1.0, side="long")
        assert result["reward_risk_ratio"] == 2.0
        assert result["meets_min_reward_risk"] is True

    def test_analyze_vwap_bands_dispatches(self):
        state = AppState()
        result = _dispatch_tool("analyze_vwap_bands", {}, state, DecisionTracker())
        assert "note" in result  # no bars yet

    def test_smart_money_personality_uses_smart_money_tools(self):
        assert PERSONALITY_TOOLS["smart_money"] is SMART_MONEY_TOOLS
        names = {t["function"]["name"] for t in SMART_MONEY_TOOLS}
        assert {
            "analyze_daily_trend",
            "analyze_order_blocks",
            "analyze_smart_money_setup",
            "analyze_fair_value_gaps",
            "smart_money_trade_geometry",
        } <= names
        assert "analyze_market" not in names
        assert "get_put_call_walls" not in names
        assert "analyze_opening_range" not in names
        assert "analyze_vwap_bands" not in names

    def test_smart_money_trade_geometry_tool_computes_reward_risk(self):
        state = AppState()
        result = _tool_smart_money_trade_geometry(state, entry=100.0, stop=98.0, target=110.0)
        assert result["reward_risk_ratio"] == 5.0
        assert result["meets_min_reward_risk"] is True

    def test_analyze_smart_money_setup_dispatches_without_daily_bars(self):
        result = _dispatch_tool("analyze_smart_money_setup", {}, AppState(), DecisionTracker())
        assert "note" in result  # no daily bars yet

    def test_analyze_order_blocks_dispatches_without_daily_bars(self):
        result = _dispatch_tool("analyze_order_blocks", {}, AppState(), DecisionTracker())
        assert "note" in result  # no daily bars yet


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
                tool_calls=[
                    _tool_call(
                        "c1",
                        "submit_decision",
                        {
                            "action": "alert",
                            "reasoning": "no setup yet, watch the opening-range high",
                            "alerts": [{"field": "last_price", "condition": "above", "value": 150.0}],
                        },
                    )
                ]
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

    def test_removed_sleep_action_is_rejected_and_retried(self):
        """The agent can no longer choose to sleep. A model that still reaches for the
        old "sleep" action gets an error back and must finalize with a real decision --
        here it corrects to an alert."""
        state = AppState()
        state.symbol = "AAPL"
        state.api_key = "k"
        state.api_secret = "s"
        tracker = DecisionTracker(broker=FakeBroker())

        responses = [
            _response(
                tool_calls=[_tool_call("c1", "submit_decision", {"action": "sleep", "reasoning": "no clear edge"})]
            ),
            _response(
                tool_calls=[
                    _tool_call(
                        "c2",
                        "submit_decision",
                        {
                            "action": "alert",
                            "reasoning": "stand aside until price reclaims resistance",
                            "alerts": [{"field": "last_price", "condition": "above", "value": 150.0}],
                        },
                    )
                ]
            ),
        ]
        client = FakeClient(responses)

        run_agent_cycle(client, "gemini-2.0-flash", "AAPL", state, tracker, max_iters=3)

        snap = tracker.snapshot()
        assert len(snap["decisions"]) == 1
        assert snap["decisions"][0].action == "alert"
        assert state.alerts == [{"field": "last_price", "condition": "above", "value": 150.0}]

        # the rejected "sleep" attempt must have been surfaced back to the model
        second_call_messages = client.calls[1]
        tool_results = [m["content"] for m in second_call_messages if m.get("role") == "tool"]
        assert any("'buy', 'sell', or 'alert'" in c for c in tool_results)

    def test_alert_decision_sets_state_alert_and_records_no_trade(self):
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
                            "alerts": [{"field": "last_price", "condition": "above", "value": 150.0}],
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
        assert decision.alerts == [{"field": "last_price", "condition": "above", "value": 150.0}]
        assert state.alerts == [{"field": "last_price", "condition": "above", "value": 150.0}]

    def test_alert_decision_supports_multiple_conditions_across_fields(self):
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
                            "reasoning": "watch a breakdown below support or a real volume surge",
                            "alerts": [
                                {"field": "last_price", "condition": "below", "value": 95.0},
                                {"field": "volume_ratio", "condition": "above", "value": 2.0},
                            ],
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
            {"field": "last_price", "condition": "below", "value": 95.0},
            {"field": "volume_ratio", "condition": "above", "value": 2.0},
        ]
        assert state.alerts == decision.alerts

    def test_alert_with_missing_fields_is_rejected_and_retried(self):
        """An incomplete alert call (no conditions) must not be silently downgraded to
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
                            "alerts": [{"field": "last_price", "condition": "above", "value": 150.0}],
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
        assert snap["decisions"][0].alerts == [{"field": "last_price", "condition": "above", "value": 150.0}]
        assert state.alerts == [{"field": "last_price", "condition": "above", "value": 150.0}]

        # the rejected first attempt must have been surfaced back to the model as a tool result
        first_call_messages = client.calls[1]
        tool_results = [m["content"] for m in first_call_messages if m.get("role") == "tool"]
        assert any("requires a non-empty 'alerts' array" in c for c in tool_results)

    def test_alert_with_invalid_field_is_rejected(self):
        """A condition naming a field that isn't continuously tracked is dropped, so an
        otherwise-empty alert is rejected rather than silently watching nothing. The model
        then corrects to a valid, watchable condition."""
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
                            "reasoning": "bad field",
                            "alerts": [{"field": "rsi", "condition": "above", "value": 70}],
                        },
                    )
                ]
            ),
            _response(
                tool_calls=[
                    _tool_call(
                        "c2",
                        "submit_decision",
                        {
                            "action": "alert",
                            "reasoning": "stand aside until a real breakdown",
                            "alerts": [{"field": "last_price", "condition": "below", "value": 95.0}],
                        },
                    )
                ]
            ),
        ]
        client = FakeClient(responses)

        run_agent_cycle(client, "gpt-4.1-mini", "AAPL", state, tracker, max_iters=3)

        snap = tracker.snapshot()
        assert snap["decisions"][0].action == "alert"
        assert state.alerts == [{"field": "last_price", "condition": "below", "value": 95.0}]

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
        assert state.alerts == []


class TestAlertTrigger:
    def test_alert_triggered_above(self):
        state = AppState()
        state.last_price = 151.0
        assert alert_triggered(state, {"field": "last_price", "condition": "above", "value": 150.0}) is True
        state.last_price = 149.0
        assert alert_triggered(state, {"field": "last_price", "condition": "above", "value": 150.0}) is False

    def test_alert_triggered_below(self):
        state = AppState()
        state.last_price = 99.0
        assert alert_triggered(state, {"field": "last_price", "condition": "below", "value": 100.0}) is True
        state.last_price = 101.0
        assert alert_triggered(state, {"field": "last_price", "condition": "below", "value": 100.0}) is False

    def test_alert_triggered_on_non_price_field(self):
        state = AppState()
        state.day_volume = 6_000_000
        assert alert_triggered(state, {"field": "day_volume", "condition": "above", "value": 5_000_000}) is True
        state.bid_price, state.ask_price = 10.0, 10.05
        assert alert_triggered(state, {"field": "spread", "condition": "above", "value": 0.04}) is True
        assert alert_triggered(state, {"field": "spread", "condition": "below", "value": 0.04}) is False

    def test_wait_returns_early_when_alert_fires(self):
        state = AppState()
        state.last_price = 151.0
        state.alerts = [{"field": "last_price", "condition": "above", "value": 150.0}]
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
        assert state.alerts == []  # cleared once triggered
        with state.lock:
            log_types = [e["type"] for e in state.agent_log]
        assert "status" in log_types

    def test_wait_returns_early_when_low_side_of_bracket_fires(self):
        state = AppState()
        state.last_price = 94.0
        state.alerts = [
            {"field": "last_price", "condition": "below", "value": 95.0},
            {"field": "last_price", "condition": "above", "value": 150.0},
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
        assert state.alerts == []  # both levels cleared once either fires
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
