import json
from types import SimpleNamespace

from agent_stonks.agent import (
    DISABLED_PERSONALITIES,
    _TOOL_STAND_DOWN,
    run_agent_cycle,
)
from agent_stonks.automatic import (
    AUTOMATIC_KEY,
    REGIME_TOOLS,
    SELECTABLE_STRATEGIES,
    run_regime_cycle,
)
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


class TestStandDown:
    def test_stand_down_tool_only_added_under_automatic(self):
        state = _base_state()
        tracker = DecisionTracker(broker=FakeBroker())
        # Normal mode: model finalizes with a regular alert.
        responses = [
            _response(
                tool_calls=[
                    _tool_call(
                        "c1",
                        "submit_decision",
                        {
                            "action": "alert",
                            "reasoning": "waiting",
                            "alerts": [{"field": "last_price", "condition": "above", "value": 150.0}],
                        },
                    )
                ]
            )
        ]
        client = FakeClient(responses)
        run_agent_cycle(client, "m", ["AAPL"], state, tracker, max_iters=3, personality="breakout")
        names = {t["function"]["name"] for t in client.tools_seen[0]}
        assert "stand_down" not in names

    def test_stand_down_added_and_returns_signal(self):
        state = _base_state()
        tracker = DecisionTracker(broker=FakeBroker())
        responses = [
            _response(
                tool_calls=[
                    _tool_call(
                        "c1",
                        "stand_down",
                        {"reasoning": "range has resolved into a strong trend", "expected_quiet_minutes": 45},
                    )
                ]
            )
        ]
        client = FakeClient(responses)
        result = run_agent_cycle(
            client, "m", "AAPL", state, tracker, max_iters=3, personality="reversal", under_automatic=True
        )
        assert result == "stand_down"
        # stand_down tool was exposed to the model
        names = {t["function"]["name"] for t in client.tools_seen[0]}
        assert "stand_down" in names
        # No trade/alert decision was recorded -- it's a relinquish, not a decision.
        snap = tracker.snapshot()
        assert snap["decisions"] == []
        # It was logged as a stand_down event.
        with state.lock:
            types = [e["type"] for e in state.agent_log]
        assert "stand_down" in types

    def test_normal_decision_returns_decided(self):
        state = _base_state()
        tracker = DecisionTracker(starting_cash=1000.0, broker=FakeBroker(price=100.0), trade_cost=0.0)
        responses = [
            _response(
                tool_calls=[
                    _tool_call(
                        "c1", "submit_decision",
                        {"action": "buy", "quantity": 1, "reasoning": "go", "wake_in_minutes": 5},
                    )
                ]
            )
        ]
        client = FakeClient(responses)
        result = run_agent_cycle(
            client, "m", "AAPL", state, tracker, max_iters=3, personality="momentum", under_automatic=True
        )
        assert result == "decided"
        assert tracker.snapshot()["positions"] == {"AAPL": 1.0}


class TestRunRegimeCycle:
    def test_selects_strategy_after_analysis(self):
        state = _base_state()
        tracker = DecisionTracker(broker=FakeBroker())
        responses = [
            _response(tool_calls=[_tool_call("c1", "analyze_daily_trend", {})]),
            _response(
                tool_calls=[
                    _tool_call(
                        "c2",
                        "select_strategy",
                        {
                            "symbol": "AAPL",
                            "strategy": "momentum",
                            "regime": "bullish_trend",
                            "market_regime": "volatile",
                            "reasoning": "fresh gap on 3x volume with a catalyst",
                        },
                    )
                ]
            ),
        ]
        client = FakeClient(responses)
        assignments = run_regime_cycle(client, "m", ["AAPL"], state, tracker, max_iters=5)
        assert assignments["AAPL"]["strategy"] == "momentum"
        assert assignments["AAPL"]["regime"] == "bullish_trend"
        # The ticker's own regime and the shared market backdrop are separate.
        assert assignments["AAPL"]["market_regime"] == "volatile"
        assert client.tools_seen[0] is REGIME_TOOLS
        with state.lock:
            types = [e["type"] for e in state.agent_log]
        assert "regime_select" in types

    def test_invalid_strategy_is_rejected_then_corrected(self):
        state = _base_state()
        tracker = DecisionTracker(broker=FakeBroker())
        responses = [
            _response(tool_calls=[_tool_call("c1", "select_strategy", {"symbol": "AAPL", "strategy": "scalping", "reasoning": "x"})]),
            _response(
                tool_calls=[
                    _tool_call("c2", "select_strategy", {"symbol": "AAPL", "strategy": "momentum", "regime": "ranging", "reasoning": "mixed"})
                ]
            ),
        ]
        client = FakeClient(responses)
        assignments = run_regime_cycle(client, "m", ["AAPL"], state, tracker, max_iters=5)
        assert assignments["AAPL"]["strategy"] == "momentum"
        # the rejected attempt was surfaced back to the model
        second_call = client.calls[1]
        tool_results = [m["content"] for m in second_call if m.get("role") == "tool"]
        assert any("strategy must be one of" in c for c in tool_results)

    def test_returns_nothing_when_never_selects(self):
        state = _base_state()
        tracker = DecisionTracker(broker=FakeBroker())
        responses = [_response(content="thinking") for _ in range(3)]
        client = FakeClient(responses)
        assert run_regime_cycle(client, "m", ["AAPL"], state, tracker, max_iters=3) == {}

    def test_regime_tools_include_corporate_actions_but_no_trading(self):
        names = {t["function"]["name"] for t in REGIME_TOOLS}
        assert "get_corporate_actions" in names
        assert "submit_decision" not in names
        assert "set_tactics" not in names

    def test_selectable_strategies_match_personalities(self):
        # Orchestrator can pick any tradeable personality, and automatic is not
        # itself selectable.
        assert AUTOMATIC_KEY not in SELECTABLE_STRATEGIES
        assert "momentum" in SELECTABLE_STRATEGIES
        assert "breakout" in SELECTABLE_STRATEGIES
        # Premarket is activated deterministically before the open, never by
        # the regime cycle.
        assert "premarket" not in SELECTABLE_STRATEGIES
        # Switched-off personalities are never activated either.
        assert not DISABLED_PERSONALITIES & set(SELECTABLE_STRATEGIES)
        enum = _TOOL_STAND_DOWN["function"]["parameters"]["properties"]["reasoning"]
        assert enum["type"] == "string"


class TestBreakoutActivationGate:
    """select_strategy('breakout') is deterministically rejected when the ORB
    strategy has nothing real to trade: an unfavorable session window, or no
    measurable opening range for today."""

    def test_gated_breakout_pick_is_rejected_then_corrected(self, monkeypatch):
        import agent_stonks.automatic as automatic_mod

        monkeypatch.setattr(
            automatic_mod,
            "breakout_preconditions",
            lambda app, symbols, minutes=15: "breakout is not selectable right now: midday dead zone",
        )
        state = _base_state()
        tracker = DecisionTracker(broker=FakeBroker())
        responses = [
            _response(
                tool_calls=[
                    _tool_call("c1", "select_strategy", {"symbol": "AAPL", "strategy": "breakout", "regime": "breakout_pending", "reasoning": "x"})
                ]
            ),
            _response(
                tool_calls=[
                    _tool_call("c2", "select_strategy", {"symbol": "AAPL", "strategy": "reversal", "regime": "ranging", "reasoning": "adx 15"})
                ]
            ),
        ]
        client = FakeClient(responses)
        assignments = run_regime_cycle(client, "m", ["AAPL"], state, tracker, max_iters=5)
        assert assignments["AAPL"]["strategy"] == "reversal"
        tool_results = [m["content"] for m in client.calls[1] if m.get("role") == "tool"]
        assert any("not selectable" in c for c in tool_results)

    def test_preconditions_block_unfavorable_window(self, monkeypatch):
        import agent_stonks.technical_analysis as ta_mod
        from agent_stonks.agent import breakout_preconditions

        monkeypatch.setattr(
            ta_mod,
            "session_time_window",
            lambda *a, **k: {"favorable_for_breakouts": False, "summary": "12:30 ET -- midday dead zone."},
        )
        state = _base_state()
        reason = breakout_preconditions(state, ["AAPL"])
        assert reason is not None and "dead zone" in reason

    def test_preconditions_block_missing_opening_range(self, monkeypatch):
        import agent_stonks.technical_analysis as ta_mod
        from agent_stonks.agent import breakout_preconditions

        monkeypatch.setattr(
            ta_mod,
            "session_time_window",
            lambda *a, **k: {"favorable_for_breakouts": True, "summary": "10:00 ET -- opening window."},
        )
        state = AppState()
        state.set_symbols(["AAPL"])  # no API keys -> no REST recovery, no bars
        reason = breakout_preconditions(state, ["AAPL"])
        assert reason is not None and "opening range" in reason

    def test_preconditions_pass_with_cached_range(self, monkeypatch):
        from datetime import datetime, timezone
        from zoneinfo import ZoneInfo

        import agent_stonks.technical_analysis as ta_mod
        from agent_stonks.agent import breakout_preconditions

        monkeypatch.setattr(
            ta_mod,
            "session_time_window",
            lambda *a, **k: {"favorable_for_breakouts": True, "summary": "10:00 ET -- opening window."},
        )
        state = AppState()
        state.set_symbols(["AAPL"])
        today = datetime.now(timezone.utc).astimezone(ZoneInfo("America/New_York")).date().isoformat()
        state.sym("AAPL").opening_range = {
            "date": today, "minutes": 15, "high": 102.0, "low": 100.0,
            "bar_count": 15, "avg_volume": 1000.0, "complete": True,
        }
        assert breakout_preconditions(state, ["AAPL"]) is None


class TestPerSymbolAssignment:
    """Different tickers can be in genuinely different states, so each gets its
    own strategy rather than the basket sharing one pick."""

    def _state(self, *symbols) -> AppState:
        state = AppState()
        state.set_symbols(list(symbols))
        state.api_key = "k"
        state.api_secret = "s"
        return state

    def test_assigns_each_ticker_separately(self):
        state = self._state("AAPL", "TSLA")
        tracker = DecisionTracker(broker=FakeBroker())
        responses = [
            _response(tool_calls=[
                _tool_call("c1", "select_strategy", {
                    "symbol": "AAPL", "strategy": "momentum", "regime": "bullish_trend",
                    "market_regime": "volatile", "reasoning": "gap on 3x volume"}),
                _tool_call("c2", "select_strategy", {
                    "symbol": "TSLA", "strategy": "reversal", "regime": "ranging",
                    "market_regime": "volatile", "reasoning": "adx 14, stretched from vwap"}),
            ]),
        ]
        client = FakeClient(responses)
        out = run_regime_cycle(client, "m", ["AAPL", "TSLA"], state, tracker, max_iters=4)

        assert out["AAPL"]["strategy"] == "momentum"
        assert out["TSLA"]["strategy"] == "reversal"
        # The market backdrop is shared; the ticker regimes are not.
        assert out["AAPL"]["market_regime"] == out["TSLA"]["market_regime"] == "volatile"
        assert out["AAPL"]["regime"] != out["TSLA"]["regime"]

    def test_keeps_going_until_every_ticker_is_assigned(self):
        state = self._state("AAPL", "TSLA")
        tracker = DecisionTracker(broker=FakeBroker())
        responses = [
            _response(tool_calls=[_tool_call("c1", "select_strategy", {
                "symbol": "AAPL", "strategy": "momentum", "reasoning": "x"})]),
            _response(tool_calls=[_tool_call("c2", "select_strategy", {
                "symbol": "TSLA", "strategy": "reversal", "reasoning": "y"})]),
        ]
        client = FakeClient(responses)
        out = run_regime_cycle(client, "m", ["AAPL", "TSLA"], state, tracker, max_iters=5)
        assert set(out) == {"AAPL", "TSLA"}

    def test_a_partial_round_still_trades_what_it_assigned(self):
        # Three of four assigned should trade those three, not discard the round.
        state = self._state("AAPL", "TSLA")
        tracker = DecisionTracker(broker=FakeBroker())
        responses = [
            _response(tool_calls=[_tool_call("c1", "select_strategy", {
                "symbol": "AAPL", "strategy": "momentum", "reasoning": "x"})]),
            _response(content="I am done"),
            _response(content="still done"),
        ]
        client = FakeClient(responses)
        out = run_regime_cycle(client, "m", ["AAPL", "TSLA"], state, tracker, max_iters=3)
        assert set(out) == {"AAPL"}

    def test_an_unknown_symbol_is_rejected(self):
        state = self._state("AAPL")
        tracker = DecisionTracker(broker=FakeBroker())
        responses = [
            _response(tool_calls=[_tool_call("c1", "select_strategy", {
                "symbol": "NVDA", "strategy": "momentum", "reasoning": "x"})]),
            _response(tool_calls=[_tool_call("c2", "select_strategy", {
                "symbol": "AAPL", "strategy": "momentum", "reasoning": "x"})]),
        ]
        client = FakeClient(responses)
        out = run_regime_cycle(client, "m", ["AAPL"], state, tracker, max_iters=4)
        assert set(out) == {"AAPL"}
        results = [m["content"] for m in client.calls[1] if m.get("role") == "tool"]
        assert any("must be one of the tickers" in c for c in results)

    def test_breakout_gate_is_checked_per_ticker(self, monkeypatch):
        # One ticker lacking a measurable opening range says nothing about
        # another's, so the gate must be evaluated for the named symbol only.
        import agent_stonks.automatic as automatic_mod

        seen = []

        def _gate(app, symbols, minutes=15):
            seen.append(list(symbols))
            return "no opening range" if symbols == ["TSLA"] else None

        monkeypatch.setattr(automatic_mod, "breakout_preconditions", _gate)
        state = self._state("AAPL", "TSLA")
        tracker = DecisionTracker(broker=FakeBroker())
        responses = [
            _response(tool_calls=[
                _tool_call("c1", "select_strategy", {
                    "symbol": "AAPL", "strategy": "breakout", "reasoning": "x"}),
                _tool_call("c2", "select_strategy", {
                    "symbol": "TSLA", "strategy": "breakout", "reasoning": "y"}),
            ]),
            _response(tool_calls=[_tool_call("c3", "select_strategy", {
                "symbol": "TSLA", "strategy": "reversal", "reasoning": "z"})]),
        ]
        client = FakeClient(responses)
        out = run_regime_cycle(client, "m", ["AAPL", "TSLA"], state, tracker, max_iters=4)

        assert seen == [["AAPL"], ["TSLA"]]      # per ticker, never the basket
        assert out["AAPL"]["strategy"] == "breakout"   # allowed
        assert out["TSLA"]["strategy"] == "reversal"   # gated, re-picked


class TestGroupByStrategy:
    def test_tickers_sharing_a_strategy_are_traded_together(self):
        from agent_stonks.automatic import group_by_strategy

        groups = group_by_strategy({
            "AAPL": {"strategy": "momentum"},
            "MSFT": {"strategy": "momentum"},
            "TSLA": {"strategy": "reversal"},
        })
        as_dict = dict(groups)
        assert as_dict["momentum"] == ["AAPL", "MSFT"]
        assert as_dict["reversal"] == ["TSLA"]
        # One cycle per distinct strategy, not per ticker.
        assert len(groups) == 2

    def test_empty_assignments_produce_no_groups(self):
        from agent_stonks.automatic import group_by_strategy

        assert group_by_strategy({}) == []


class TestPublishAssignments:
    def test_single_strategy_keeps_the_legacy_summary_fields(self):
        from agent_stonks.automatic import _publish_assignments

        state = AppState()
        _publish_assignments(state, {
            "AAPL": {"strategy": "momentum", "regime": "bullish_trend",
                     "market_regime": "quiet", "reasoning": "gap on volume"},
        })
        assert state.automatic_active_strategy == "momentum"
        assert state.automatic_regime == "quiet"       # the market's, not the ticker's
        assert state.automatic_reason == "gap on volume"

    def test_mixed_strategies_summarise_as_the_dominant_one(self):
        from agent_stonks.automatic import _publish_assignments

        state = AppState()
        _publish_assignments(state, {
            "AAPL": {"strategy": "momentum", "regime": "bullish_trend",
                     "market_regime": "quiet", "reasoning": "a"},
            "MSFT": {"strategy": "momentum", "regime": "bullish_trend",
                     "market_regime": "quiet", "reasoning": "b"},
            "TSLA": {"strategy": "reversal", "regime": "ranging",
                     "market_regime": "quiet", "reasoning": "c"},
        })
        assert state.automatic_active_strategy == "momentum"  # covers 2 of 3
        assert "TSLA" in state.automatic_reason               # per-ticker map instead
        assert len(state.automatic_assignments) == 3

    def test_clearing_resets_every_field(self):
        from agent_stonks.automatic import _publish_assignments

        state = AppState()
        _publish_assignments(state, {"AAPL": {"strategy": "momentum", "reasoning": "a"}})
        _publish_assignments(state, {})
        assert state.automatic_assignments == {}
        assert state.automatic_active_strategy is None
        assert state.automatic_regime is None


def _assignment(strategy: str) -> dict:
    return {"strategy": strategy, "regime": "ranging", "market_regime": "quiet", "reasoning": "r"}


class TestAutomaticLoopWakes:
    """No cycle timer: a round runs after a wake, or straight after a
    stand-down to re-assess the freed tickers."""

    def test_stand_down_reassesses_now_and_leaves_other_groups_asleep(self, monkeypatch):
        import threading

        from agent_stonks import automatic, market_hours

        state = AppState()
        state.set_symbols(["AAPL", "KO"])
        tracker = DecisionTracker(broker=FakeBroker())
        stop = threading.Event()
        regime_rounds = iter([
            {"AAPL": _assignment("momentum"), "KO": _assignment("reversal")},
            {"AAPL": _assignment("breakout")},
        ])
        regime_calls: list = []
        cycles: list = []
        waits: list = []

        def fake_regime(client, model, symbols, st, tr):
            regime_calls.append(list(symbols))
            return next(regime_rounds)

        def fake_cycle(client, model, symbols, st, tr, personality, under_automatic):
            cycles.append((personality, list(symbols)))
            return "stand_down" if personality == "momentum" else "decided"

        def fake_wait(st, stop_event):
            waits.append(True)
            stop_event.set()

        monkeypatch.setattr(market_hours, "is_market_open", lambda now=None: True)
        monkeypatch.setattr(automatic, "get_agent_client", lambda provider, api_key: object())
        monkeypatch.setattr(automatic, "run_regime_cycle", fake_regime)
        monkeypatch.setattr(automatic, "run_agent_cycle", fake_cycle)
        monkeypatch.setattr(automatic, "_wait_for_next_cycle", fake_wait)

        automatic._automatic_loop(state, tracker, ["AAPL", "KO"], "openai", "key", "model", stop)

        assert regime_calls == [["AAPL", "KO"], ["AAPL"]]
        # KO's reversal agent decided in round one and nothing has woken it,
        # so the round after AAPL's stand-down runs only AAPL's new strategy.
        assert cycles == [
            ("momentum", ["AAPL"]),
            ("reversal", ["KO"]),
            ("breakout", ["AAPL"]),
        ]
        assert len(waits) == 1

    def test_unassigned_tickers_get_a_retry(self, monkeypatch):
        import threading

        from agent_stonks import automatic, market_hours

        state = AppState()
        state.set_symbols(["AAPL", "KO"])
        tracker = DecisionTracker(broker=FakeBroker())
        stop = threading.Event()
        seen: dict = {}

        def fake_wait(st, stop_event):
            seen["wake_at"] = st.agent_wake_at
            stop_event.set()

        monkeypatch.setattr(market_hours, "is_market_open", lambda now=None: True)
        monkeypatch.setattr(automatic, "get_agent_client", lambda provider, api_key: object())
        monkeypatch.setattr(
            automatic, "run_regime_cycle",
            lambda client, model, symbols, st, tr: {"AAPL": _assignment("momentum")},
        )
        monkeypatch.setattr(
            automatic, "run_agent_cycle",
            lambda client, model, symbols, st, tr, personality, under_automatic: "decided",
        )
        monkeypatch.setattr(automatic, "_wait_for_next_cycle", fake_wait)

        automatic._automatic_loop(state, tracker, ["AAPL", "KO"], "openai", "key", "model", stop)

        assert seen["wake_at"] is not None


class TestBasketScopedCleanup:
    """Under the orchestrator several strategies share one AppState: a cycle
    or a stand-down on one basket must not disarm another's."""

    def test_a_cycle_clears_only_its_own_tickers_alerts(self):
        state = AppState()
        state.set_symbols(["AAPL", "KO"])
        ko_alert = {"symbol": "KO", "field": "last_price", "condition": "above", "value": 70.0}
        state.sym("KO").alerts = [ko_alert]
        tracker = DecisionTracker(broker=FakeBroker())
        client = FakeClient([
            _response(tool_calls=[_tool_call("c1", "stand_down", {"reasoning": "no edge"})]),
        ])

        run_agent_cycle(client, "m", ["AAPL"], state, tracker, max_iters=3, under_automatic=True)

        assert state.sym("KO").alerts == [ko_alert]

    def test_stand_down_disarms_only_its_own_tickers_tactics(self):
        state = AppState()
        state.set_symbols(["AAPL", "KO"])
        state.sym("AAPL").tactics = object()
        ko_plan = object()
        state.sym("KO").tactics = ko_plan
        tracker = DecisionTracker(broker=FakeBroker())
        client = FakeClient([
            _response(tool_calls=[_tool_call("c1", "stand_down", {"reasoning": "no edge"})]),
        ])

        run_agent_cycle(client, "m", ["AAPL"], state, tracker, max_iters=3, under_automatic=True)

        assert state.sym("AAPL").tactics is None
        assert state.sym("KO").tactics is ko_plan
