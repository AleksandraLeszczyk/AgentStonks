import json
from types import SimpleNamespace

import pytest

import agent_stonks.tactics as tactics_mod
from agent_stonks.agent import run_agent_cycle
from agent_stonks.broker import Broker
from agent_stonks.decisions import DecisionTracker
from agent_stonks.state import AppState
from agent_stonks.tactics import (
    TacticsExecutor,
    format_tactic_action,
    momentum_pct,
    normalize_tactics,
    tactic_price_levels,
    tactics_summaries,
)


class FakeBroker(Broker):
    def __init__(self, price: float = 100.0):
        self.price = price
        self.orders: list = []

    def get_current_price(self, symbol, key, secret, feed="iex") -> float:
        return self.price

    def submit_order(self, symbol, side, quantity, price) -> dict:
        self.orders.append((side, quantity, price))
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


def _entry_action(quantity=10.0, quantity_pct=None, action="buy", conditions=None, note=""):
    entry: dict = {"action": action, "conditions": conditions or [{"field": "last_price", "condition": "below", "value": 90.0}]}
    if quantity is not None:
        entry["quantity"] = quantity
    if quantity_pct is not None:
        entry["quantity_pct"] = quantity_pct
    if note:
        entry["note"] = note
    return entry


class TestNormalizeTactics:
    def test_valid_multi_action_multi_condition(self):
        raw = [
            _entry_action(note="entry"),
            {
                "action": "sell",
                "quantity_pct": 20,
                "conditions": [
                    {"field": "last_price", "condition": "above", "value": 120},
                    {"field": "vix", "condition": "below", "value": 20},
                ],
                "note": "take profit",
            },
        ]
        tactics, error = normalize_tactics("AAPL", raw, "bracket")
        assert error is None
        assert tactics.status == "armed"
        assert len(tactics.actions) == 2
        assert tactics.actions[1].quantity_pct == 20
        assert len(tactics.actions[1].conditions) == 2

    def test_rejects_empty_actions(self):
        tactics, error = normalize_tactics("AAPL", [], "x")
        assert tactics is None and "non-empty" in error

    def test_rejects_unknown_field(self):
        raw = [_entry_action(conditions=[{"field": "moon_phase", "condition": "above", "value": 1}])]
        tactics, error = normalize_tactics("AAPL", raw, "x")
        assert tactics is None and "field" in error

    def test_rejects_both_quantity_and_pct(self):
        raw = [_entry_action(quantity=10, quantity_pct=50)]
        tactics, error = normalize_tactics("AAPL", raw, "x")
        assert tactics is None and "exactly one" in error

    def test_rejects_missing_quantity(self):
        raw = [_entry_action(quantity=None)]
        tactics, error = normalize_tactics("AAPL", raw, "x")
        assert tactics is None and "exactly one" in error

    def test_rejects_pct_out_of_range(self):
        raw = [_entry_action(quantity=None, quantity_pct=150, action="sell")]
        tactics, error = normalize_tactics("AAPL", raw, "x")
        assert tactics is None and "quantity_pct" in error

    def test_rejects_empty_conditions(self):
        raw = [{"action": "buy", "quantity": 5, "conditions": []}]
        tactics, error = normalize_tactics("AAPL", raw, "x")
        assert tactics is None and "conditions" in error


class TestFormattingAndLevels:
    def test_format_action_shares(self):
        tactics, _ = normalize_tactics("AAPL", [_entry_action()], "x")
        assert format_tactic_action(tactics.actions[0]) == "buy 10 sh when last_price below 90"

    def test_format_action_pct_of_position(self):
        raw = [
            {
                "action": "sell",
                "quantity_pct": 20,
                "conditions": [{"field": "last_price", "condition": "above", "value": 120}],
            }
        ]
        tactics, _ = normalize_tactics("AAPL", raw, "x")
        assert format_tactic_action(tactics.actions[0]) == "sell 20% of position when last_price above 120"

    def test_price_levels_skip_non_price_fields(self):
        raw = [
            {
                "action": "buy",
                "quantity": 10,
                "conditions": [
                    {"field": "last_price", "condition": "below", "value": 90},
                    {"field": "vix", "condition": "below", "value": 20},
                ],
            }
        ]
        tactics, _ = normalize_tactics("AAPL", raw, "x")
        levels = tactic_price_levels(tactics)
        assert len(levels) == 1
        assert levels[0]["field"] == "last_price"
        assert levels[0]["action"] == "buy"
        assert levels[0]["value"] == 90

    def test_summaries_empty_when_none_armed(self):
        assert tactics_summaries(None) == []


class TestMomentumPct:
    def test_computes_pct_change_over_window(self):
        app = AppState()
        app.set_symbols(["AAPL"])
        state = app.sym("AAPL")
        state.bars.extend(
            [
                {"t": "2026-07-02T14:00:00Z", "c": 100.0},
                {"t": "2026-07-02T14:05:00Z", "c": 102.0},
                {"t": "2026-07-02T14:12:00Z", "c": 110.0},
            ]
        )
        # Baseline is the first bar at/beyond the 10-minute lookback: 100 -> 110 = +10%.
        assert momentum_pct(state, window_min=10) == pytest.approx(10.0)

    def test_none_without_enough_bars(self):
        app = AppState()
        app.set_symbols(["AAPL"])
        state = app.sym("AAPL")
        state.bars.append({"t": "2026-07-02T14:00:00Z", "c": 100.0})
        assert momentum_pct(state) is None


def _armed_state(symbol="AAPL", raw_actions=None, last_price=None):
    app = AppState()
    app.set_symbols([symbol])
    state = app.sym(symbol)
    state.last_price = last_price
    tactics, error = normalize_tactics(symbol, raw_actions, "test plan")
    assert error is None
    state.tactics = tactics
    return state


class TestTacticsExecutor:
    def test_no_tactics_no_execution(self):
        app = AppState()
        app.set_symbols(["AAPL"])
        state = app.sym("AAPL")
        tracker = DecisionTracker(starting_cash=1000, broker=FakeBroker())
        executor = TacticsExecutor(state, tracker)
        assert executor.check_now() is None

    def test_buy_fires_when_condition_met(self):
        state = _armed_state(raw_actions=[_entry_action(quantity=5)], last_price=85.0)
        broker = FakeBroker(price=85.0)
        tracker = DecisionTracker(starting_cash=10_000, broker=broker)
        executor = TacticsExecutor(state, tracker)

        fired = executor.check_now()

        assert fired is not None and fired.action == "buy"
        assert broker.orders == [("buy", 5.0, 85.0)]
        assert state.tactics is None  # disarmed after execution
        assert state.agent_wake_event.is_set()
        assert "Tactics executed" in state.agent_wake_reason
        decision = tracker.snapshot()["decisions"][-1]
        assert decision.action == "buy" and decision.status == "filled"
        entry = state.agent_log[-1]
        assert entry["type"] == "tactics_execution" and entry["status"] == "filled"

    def test_does_not_fire_when_condition_not_met(self):
        state = _armed_state(raw_actions=[_entry_action(quantity=5)], last_price=95.0)
        tracker = DecisionTracker(starting_cash=10_000, broker=FakeBroker())
        executor = TacticsExecutor(state, tracker)
        assert executor.check_now() is None
        assert state.tactics is not None
        assert not state.agent_wake_event.is_set()

    def test_all_conditions_must_hold(self, monkeypatch):
        raw = [
            {
                "action": "buy",
                "quantity": 5,
                "conditions": [
                    {"field": "last_price", "condition": "below", "value": 90},
                    {"field": "vix", "condition": "below", "value": 20},
                ],
            }
        ]
        state = _armed_state(raw_actions=raw, last_price=85.0)
        tracker = DecisionTracker(starting_cash=10_000, broker=FakeBroker(price=85.0))
        executor = TacticsExecutor(state, tracker)

        monkeypatch.setattr(tactics_mod, "fetch_vix_level", lambda: 25.0)
        assert executor.check_now() is None

        monkeypatch.setattr(tactics_mod, "fetch_vix_level", lambda: 18.0)
        fired = executor.check_now()
        assert fired is not None and fired.action == "buy"

    def test_sell_pct_resolves_from_position(self):
        raw = [
            {
                "action": "sell",
                "quantity_pct": 20,
                "conditions": [{"field": "last_price", "condition": "above", "value": 120}],
            }
        ]
        state = _armed_state(raw_actions=raw, last_price=125.0)
        broker = FakeBroker(price=125.0)
        tracker = DecisionTracker(starting_cash=0, broker=broker)
        tracker.positions["AAPL"] = 50.0

        TacticsExecutor(state, tracker).check_now()

        assert broker.orders == [("sell", 10.0, 125.0)]  # 20% of 50 shares

    def test_buy_pct_resolves_from_cash(self):
        raw = [
            {
                "action": "buy",
                "quantity_pct": 50,
                "conditions": [{"field": "last_price", "condition": "below", "value": 110}],
            }
        ]
        state = _armed_state(raw_actions=raw, last_price=100.0)
        broker = FakeBroker(price=100.0)
        tracker = DecisionTracker(starting_cash=10_000, broker=broker, trade_cost=0.0)

        TacticsExecutor(state, tracker).check_now()

        assert broker.orders and broker.orders[0][0] == "buy"
        assert broker.orders[0][1] == 50.0  # 50% of $10k at $100/sh

    def test_unfillable_sell_logs_error_and_wakes_agent(self):
        raw = [
            {
                "action": "sell",
                "quantity_pct": 50,
                "conditions": [{"field": "last_price", "condition": "above", "value": 120}],
            }
        ]
        state = _armed_state(raw_actions=raw, last_price=125.0)
        tracker = DecisionTracker(starting_cash=100, broker=FakeBroker(price=125.0))  # no position

        TacticsExecutor(state, tracker).check_now()

        assert state.tactics is None
        assert state.agent_wake_event.is_set()
        entry = state.agent_log[-1]
        assert entry["type"] == "tactics_execution" and entry["status"] == "error"

    def test_first_matching_action_wins_and_disarms_rest(self):
        raw = [
            _entry_action(quantity=5),  # buy below 90 -- met
            {
                "action": "sell",
                "quantity": 5,
                "conditions": [{"field": "last_price", "condition": "below", "value": 90}],  # also met
            },
        ]
        state = _armed_state(raw_actions=raw, last_price=85.0)
        broker = FakeBroker(price=85.0)
        tracker = DecisionTracker(starting_cash=10_000, broker=broker)

        TacticsExecutor(state, tracker).check_now()

        assert len(broker.orders) == 1 and broker.orders[0][0] == "buy"
        assert state.tactics is None


class TestAgentSetTactics:
    def _run(self, state, tracker, responses):
        client = FakeClient(responses)
        return run_agent_cycle(client, "test-model", ["AAPL"], state, tracker, personality="momentum")

    def test_set_tactics_arms_and_allows_bare_alert(self):
        state = AppState()
        state.set_symbols(["AAPL"])
        state.sym("AAPL").last_price = 100.0
        tracker = DecisionTracker(starting_cash=10_000, broker=FakeBroker())
        responses = [
            _response(
                tool_calls=[
                    _tool_call(
                        "c1",
                        "set_tactics",
                        {"actions": [_entry_action(quantity=5)], "reasoning": "buy the dip"},
                    )
                ]
            ),
            _response(
                tool_calls=[
                    _tool_call(
                        "c2",
                        "submit_decision",
                        {"action": "alert", "reasoning": "tactics armed, sleeping", "alerts": []},
                    )
                ]
            ),
        ]

        result = self._run(state, tracker, responses)

        assert result == "decided"
        armed = state.sym("AAPL").tactics
        assert armed is not None and len(armed.actions) == 1
        actions = [d.action for d in tracker.snapshot()["decisions"]]
        assert actions == ["tactics", "alert"]
        types = [e["type"] for e in state.agent_log]
        assert "tactics_set" in types
        armed_decision = tracker.snapshot()["decisions"][0]
        assert armed_decision.status == "armed"
        assert armed_decision.price == 100.0
        assert armed_decision.tactics == ["buy 5 sh AAPL when last_price below 90"]

    def test_bare_alert_without_tactics_still_rejected(self):
        state = AppState()
        state.set_symbols(["AAPL"])
        tracker = DecisionTracker(starting_cash=10_000, broker=FakeBroker())
        responses = [
            _response(
                tool_calls=[
                    _tool_call("c1", "submit_decision", {"action": "alert", "reasoning": "nap", "alerts": []})
                ]
            ),
            _response(
                tool_calls=[
                    _tool_call(
                        "c2",
                        "submit_decision",
                        {
                            "action": "alert",
                            "reasoning": "nap",
                            "alerts": [{"field": "last_price", "condition": "above", "value": 120}],
                        },
                    )
                ]
            ),
        ]

        result = self._run(state, tracker, responses)

        assert result == "decided"
        assert [d.action for d in tracker.snapshot()["decisions"]] == ["alert"]

    def test_invalid_tactics_returns_error_for_retry(self):
        state = AppState()
        state.set_symbols(["AAPL"])
        tracker = DecisionTracker(starting_cash=10_000, broker=FakeBroker())
        responses = [
            _response(
                tool_calls=[
                    _tool_call(
                        "c1",
                        "set_tactics",
                        {
                            "actions": [{"action": "buy", "conditions": [{"field": "last_price", "condition": "below", "value": 90}]}],
                            "reasoning": "no size given",
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
                            "reasoning": "standing aside",
                            "alerts": [{"field": "last_price", "condition": "below", "value": 90}],
                        },
                    )
                ]
            ),
        ]
        client = FakeClient(responses)

        run_agent_cycle(client, "test-model", ["AAPL"], state, tracker, personality="momentum")

        assert state.sym("AAPL").tactics is None
        # The error travelled back as the tool result on the second LLM turn.
        tool_msgs = [m for m in client.calls[1] if m.get("role") == "tool"]
        assert any("exactly one" in m["content"] for m in tool_msgs)

    def test_set_tactics_with_empty_actions_cancels(self):
        state = AppState()
        state.set_symbols(["AAPL"])
        state.sym("AAPL").last_price = 100.0
        tracker = DecisionTracker(starting_cash=10_000, broker=FakeBroker())
        tactics, _ = normalize_tactics("AAPL", [_entry_action(quantity=5)], "old plan")
        state.sym("AAPL").tactics = tactics
        responses = [
            _response(
                tool_calls=[
                    _tool_call("c1", "set_tactics", {"actions": [], "reasoning": "plan invalidated"})
                ]
            ),
            _response(
                tool_calls=[
                    _tool_call(
                        "c2",
                        "submit_decision",
                        {
                            "action": "alert",
                            "reasoning": "watching",
                            "alerts": [{"field": "last_price", "condition": "below", "value": 90}],
                        },
                    )
                ]
            ),
        ]

        self._run(state, tracker, responses)

        assert state.sym("AAPL").tactics is None
        cancelled = [e for e in state.agent_log if e["type"] == "tactics_set"]
        assert cancelled and cancelled[0]["cancelled"] == ["buy 5 sh AAPL when last_price below 90"]

    def test_all_personalities_expose_set_tactics(self):
        from agent_stonks.agent import PERSONALITY_TOOLS

        for personality, tools in PERSONALITY_TOOLS.items():
            names = [t["function"]["name"] for t in tools]
            assert "set_tactics" in names, personality
