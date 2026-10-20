"""🔄 Refresh Agent, and the handover from a running agent to the next run
that it shares with ▶ Start Agent."""
import threading
import time
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from agent_stonks import clock, rule_agent, session_store, ui
from agent_stonks.agent import stop_agent
from agent_stonks.apple_trader import AppleTraderConfig
from agent_stonks.broker import Broker

NOW = datetime(2026, 10, 20, 15, 0, tzinfo=timezone.utc)  # 11:00 ET


class AlpacaLike(Broker):
    """A venue that is not a simulation, without the network."""

    @property
    def is_simulated(self) -> bool:
        return False

    def get_current_price(self, symbol, key, secret, feed="iex") -> float:
        return 100.0

    def submit_order(self, symbol, side, quantity, price) -> dict:
        return {"status": "filled", "filled_qty": quantity, "filled_price": price}


@pytest.fixture(autouse=True)
def frozen_clock(monkeypatch):
    monkeypatch.setattr(clock, "now", lambda: NOW)


@pytest.fixture
def shown(monkeypatch):
    """What the page says, by kind; no Dummy data."""
    said = {name: [] for name in ("info", "success", "warning", "error")}
    for name, texts in said.items():
        monkeypatch.setattr(ui.st, name, lambda text, *a, _t=texts, **k: _t.append(text))
    monkeypatch.setattr(ui.st, "session_state", {})
    return said


@pytest.fixture
def launches(monkeypatch):
    """Apple Trader launched through `rule_agent.launch` with a stand-in loop
    whose wind-down, like the real one, comes a moment after it is asked to
    stop and clears the running flag."""
    configs: list = []

    def loop(state, tracker, config, cycle_sec, stop_event):
        stop_event.wait()
        time.sleep(0.2)  # the cycle that was in flight
        state.agent_running = False

    def launch(state, tracker, config=None, cycle_sec=60):
        configs.append(config)
        rule_agent.launch(
            state, tracker, "apple_trader", config.ticker,
            target=loop, args=(state, tracker, config, cycle_sec), stop_agent=stop_agent,
        )

    monkeypatch.setattr(ui, "launch_apple_trader", launch)
    yield configs


@pytest.fixture
def state():
    app = ui.AppState()
    app.set_symbols(["AAPL"])
    app.api_key = "k"  # the stream is up
    yield app
    stop_agent(app)


def _settings(**overrides) -> dict:
    return {
        "personality": "apple_trader", "provider": "openai", "model": "",
        "apple_config": AppleTraderConfig(), "trading_mode_choice": "local",
        "starting_budget": 10_000.0, **overrides,
    }


def _started(state) -> None:
    assert ui._start_agent(state, ["AAPL"], continue_today=True, **_settings())
    assert state.agent_running


def test_start_while_running_waits_for_the_old_loop(state, shown, launches):
    """The old loop's wind-down used to land after the new run had started,
    leaving it trading under an idle status line."""
    _started(state)
    assert ui._start_agent(state, ["AAPL"], continue_today=True, **_settings())
    time.sleep(0.4)  # longer than the old loop's wind-down
    assert state.agent_running
    assert len(launches) == 2


def test_refresh_carries_the_ledger_on_with_the_new_rules(state, shown, launches, monkeypatch):
    _started(state)
    tracker = state.decision_tracker
    tracker.positions = {"AAPL": 5.0}
    monkeypatch.setattr(session_store, "archive", lambda *a, **k: pytest.fail("archived"))
    old_buy_k = AppleTraderConfig().buy_k
    rules = replace(AppleTraderConfig(), buy_k=old_buy_k + 0.1)

    assert ui._refresh_agent(state, ["AAPL"], **_settings(apple_config=rules))

    assert state.agent_running and launches[-1] is rules
    assert state.decision_tracker is not tracker
    assert state.decision_tracker.snapshot()["positions"] == {"AAPL": 5.0}
    assert state.decision_tracker.buys_halted is None
    text = state.agent_log[-1]["text"]
    assert text.startswith("🔄 Agent refreshed: rules buy_k")
    assert f"{old_buy_k} → {old_buy_k + 0.1}" in text
    assert shown["success"] and not shown["error"]


def test_refresh_with_nothing_changed_says_so(state, shown, launches):
    _started(state)
    assert ui._refresh_agent(state, ["AAPL"], **_settings())
    assert "the settings were the ones it was already on" in state.agent_log[-1]["text"]


def test_settings_that_cannot_start_leave_the_run_trading(state, shown, launches):
    _started(state)
    stop_event = state.agent_stop_event
    msft = replace(AppleTraderConfig(), ticker="MSFT")
    assert not ui._refresh_agent(state, ["AAPL"], **_settings(apple_config=msft))
    assert state.agent_running and not stop_event.is_set()
    assert "add MSFT" in shown["error"][0] and "goes on as it was" in shown["error"][0]
    assert len(launches) == 1


def test_moving_off_an_alpaca_account_that_holds_positions_is_refused(state, shown, launches):
    _started(state)
    state.trading_mode = "alpaca_paper"
    state.decision_tracker.broker = AlpacaLike()
    state.decision_tracker.positions = {"AAPL": 5.0}
    stop_event = state.agent_stop_event
    assert not ui._refresh_agent(state, ["AAPL"], **_settings(trading_mode_choice="local"))
    assert state.agent_running and not stop_event.is_set()
    assert "AAPL 5" in shown["error"][0] and "Alpaca paper" in shown["error"][0]


def test_a_flat_alpaca_account_can_be_left(state, shown, launches):
    _started(state)
    state.trading_mode = "alpaca_paper"
    state.decision_tracker.broker = AlpacaLike()
    assert ui._refresh_agent(state, ["AAPL"], **_settings(trading_mode_choice="local"))
    assert state.trading_mode == "local" and state.agent_running
    assert "venue" in state.agent_log[-1]["text"]


def test_a_stopped_agent_is_not_refreshed(state, shown, launches):
    assert not ui._refresh_agent(state, ["AAPL"], **_settings())
    assert "▶ Start Agent" in shown["info"][0]
    assert launches == []


def test_refresh_changes_name_venue_model_and_symbols():
    before = {"personality": "momentum", "provider": "openai", "model": "a", "symbols": ["AAPL"]}
    after = {**before, "model": "b", "symbols": ["AAPL", "MSFT"]}
    changes = ui._refresh_changes(before, after, "local", "alpaca_paper")
    assert changes[0].startswith("venue ") and "Alpaca paper" in changes[0]
    assert changes[1] == "model openai a → openai b"
    assert changes[2] == "symbols AAPL → AAPL, MSFT"
