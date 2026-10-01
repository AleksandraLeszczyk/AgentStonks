"""Today's session on disk: what a restart of the app gets back."""
from datetime import datetime, timezone

import pandas as pd
import pytest

from agent_stonks import clock, session_store
from agent_stonks.apple_trader import AppleTraderConfig
from agent_stonks.decisions import Decision, DecisionTracker
from agent_stonks.state import AppState

NOW = datetime(2026, 9, 28, 15, 0, tzinfo=timezone.utc)  # 11:00 ET


@pytest.fixture(autouse=True)
def frozen_clock(monkeypatch):
    monkeypatch.setattr(clock, "now", lambda: NOW)


def _decision(action: str, qty: float, price: float, cash: float, pos: float) -> Decision:
    return Decision(
        ts=NOW.isoformat(), symbol="AAPL", action=action, requested_quantity=qty,
        filled_quantity=qty, price=price, reasoning=f"{action} it", status="filled",
        cash_after=cash, position_after=pos, fee=1.0, positions_after={"AAPL": pos},
    )


def _running_state() -> AppState:
    state = AppState()
    tracker = DecisionTracker(starting_cash=10_000.0, trade_cost=1.0)
    tracker.cash = 6_999.0
    tracker.positions = {"AAPL": 10.0, "MSFT": 0.0}
    tracker.decisions = [_decision("buy", 10, 300.0, 6_999.0, 10.0)]
    state.decision_tracker = tracker
    state.session_date = "2026-09-28"
    state.starting_budget = 10_000.0
    state.agent_start_time = NOW
    state.trading_mode = "local"
    state.trading_mode_requested = "local"
    state.agent_log = [{"ts": NOW.isoformat(), "type": "status", "text": "armed"}]
    state.agent_equity_history = [{"ts": NOW.isoformat(), "value": 10_050.0}]
    bar = pd.Timestamp("2026-09-28 10:59", tz="America/New_York")
    state.apple_trader_levels = {
        "ticker": "AAPL",
        "date": pd.Timestamp("2026-09-28"),
        "config": AppleTraderConfig(buy_k=0.9, sell_k=0.2),
        "rows": [{"t": bar, "model_key": "dayrange", "buy": 295.0, "sell": 305.0,
                  "stop": None, "reference": 310.0, "pred_high": 310.0, "pred_low": 290.0}],
        "memory": {"entry": {"price": 300.0, "bars": 3, "ts": bar, "risk": 2.5},
                   "stand_down": None},
        "seed": {"forecast": {"pred_high": 310.0, "pred_low": 290.0, "adr14_abs": 6.0},
                 "opening_end": pd.Timestamp("2026-09-28 09:34", tz="America/New_York"),
                 "open_price": 300.5},
    }
    session_store.claim(state)
    return state


def test_session_date_is_the_et_trading_day():
    late = datetime(2026, 9, 29, 2, 0, tzinfo=timezone.utc)  # 22:00 ET on the 28th
    assert session_store.session_date(late) == "2026-09-28"


def test_a_saved_session_comes_back_whole():
    state = _running_state()
    path = session_store.save(state)
    assert path == session_store.path_for("2026-09-28") and path.exists()

    restored = AppState()
    summary = session_store.restore(restored)
    assert summary["decisions"] == 1 and summary["fills"] == 1
    assert summary["positions"] == {"AAPL": 10.0}

    tracker = restored.decision_tracker
    assert tracker.cash == 6_999.0 and tracker.trade_cost == 1.0
    assert tracker.positions == {"AAPL": 10.0, "MSFT": 0.0}
    assert tracker.decisions == state.decision_tracker.decisions
    assert tracker.broker.is_simulated

    assert restored.session_date == "2026-09-28"
    assert restored.starting_budget == 10_000.0
    assert restored.agent_start_time == NOW
    assert restored.agent_log == state.agent_log
    assert restored.agent_equity_history == state.agent_equity_history

    levels = restored.apple_trader_levels
    assert levels["config"] == state.apple_trader_levels["config"]
    assert levels["date"] == pd.Timestamp("2026-09-28")
    row = levels["rows"][0]
    assert row == state.apple_trader_levels["rows"][0]
    assert str(row["t"].tz) == "America/New_York"
    assert levels["memory"]["entry"]["ts"] == row["t"]
    # What the chart carries the forecast on from after ▶ Stop.
    assert levels["seed"] == state.apple_trader_levels["seed"]


def test_nothing_is_restored_from_another_day(monkeypatch):
    session_store.save(_running_state())
    tomorrow = datetime(2026, 9, 29, 15, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(clock, "now", lambda: tomorrow)
    assert session_store.restore(AppState()) is None


def test_an_unreadable_file_restores_nothing():
    path = session_store.path_for("2026-09-28")
    path.parent.mkdir(parents=True)
    path.write_text("{not json")
    assert session_store.restore(AppState()) is None


def test_only_the_owner_writes_the_day():
    owner = _running_state()
    session_store.save(owner)

    looking = AppState()
    session_store.restore(looking)            # the day is already owned
    looking.agent_log.append({"ts": NOW.isoformat(), "type": "status", "text": "x"})
    assert session_store.save(looking) is None

    session_store.claim(looking)              # ▶ Start in that tab
    assert session_store.save(looking) is not None
    assert session_store.save(owner) is None


def test_the_first_state_to_restore_after_a_restart_owns_the_day():
    session_store.save(_running_state())
    session_store._owners.clear()             # a new process
    first, second = AppState(), AppState()
    session_store.restore(first)
    session_store.restore(second)
    assert session_store.owns(first) and not session_store.owns(second)


def test_an_empty_session_is_not_written():
    state = AppState()
    session_store.claim(state)
    assert session_store.save(state) is None


def test_starting_over_keeps_the_days_file_aside():
    session_store.save(_running_state())
    moved = session_store.archive()
    assert moved.exists() and moved.name == "2026-09-28-replaced-110000.json"
    assert not session_store.path_for("2026-09-28").exists()
    assert session_store.archive() is None


def test_start_continues_only_todays_ledger_on_the_same_venue():
    state = _running_state()
    assert session_store.continues(state, "local")
    assert not session_store.continues(state, "alpaca_paper")
    state.session_date = "2026-09-25"
    assert not session_store.continues(state, "local")
    assert not session_store.continues(AppState(), "local")


def test_carry_over_continues_the_ledger_but_not_a_halt():
    prior = _running_state().decision_tracker
    prior.halt_buys("sold everything")
    tracker = DecisionTracker(starting_cash=123.0)
    tracker.carry_over(prior)
    assert tracker.cash == 6_999.0
    assert tracker.positions == prior.positions
    assert tracker.decisions == prior.decisions
    assert tracker.decisions is not prior.decisions
    assert tracker.buys_halted is None


def test_the_autosave_signature_ignores_the_equity_polls():
    state = _running_state()
    before = session_store._signature(state)
    state.agent_equity_history.append({"ts": NOW.isoformat(), "value": 1.0})
    assert session_store._signature(state) == before
    state.agent_log.append({"ts": NOW.isoformat(), "type": "status", "text": "y"})
    assert session_store._signature(state) != before


def test_autosave_writes_a_change(monkeypatch):
    state = _running_state()
    session_store.start_autosave(state, interval=0.01)
    state.agent_log.append({"ts": NOW.isoformat(), "type": "status", "text": "later"})
    path = session_store.path_for("2026-09-28")
    import time

    for _ in range(200):
        if path.exists() and "later" in path.read_text():
            break
        time.sleep(0.01)
    assert "later" in path.read_text()


def test_the_first_state_with_something_to_keep_owns_an_unowned_day():
    """No file was restored this process -- e.g. a run already going when this
    module was hot-reloaded in: its first save takes the day."""
    state = _running_state()
    session_store._owners.clear()
    idle = AppState()
    assert session_store.save(idle) is None           # nothing to keep, no claim
    assert session_store.save(state) is not None
    assert session_store.owns(state)


def test_a_running_session_without_a_date_is_dated_by_its_start():
    state = _running_state()
    state.session_date = ""
    session_store.start_autosave(state, interval=60)
    assert state.session_date == "2026-09-28"


# --- what was running, and starting it again after a restart -------------------

_RUN_SPEC = {
    "personality": "apple_trader", "provider": "openai", "model": "", "symbols": ["AAPL"],
    "trading_mode": "alpaca_paper", "starting_budget": 10_000.0,
    "apple_config": {"ticker": "AAPL", "buy_k": 0.9, "sell_k": 0.2},
}


def _streaming(state: AppState) -> AppState:
    import threading

    state.set_symbols(["AAPL", "MSFT"])
    state.timeframe, state.feed = "1Min", "iex"
    state.data_source, state.history_feed = "finnhub", "auto"
    state.bars_fallback_stop_event = threading.Event()
    return state


def _trading(state: AppState) -> AppState:
    _streaming(state)
    state.run_spec = dict(_RUN_SPEC)
    state.agent_running = True
    return state


def _restart_and_restore() -> AppState:
    session_store._owners.clear()             # a new process
    session_store._resumed.clear()
    state = AppState()
    session_store.restore(state)
    return state


def test_a_running_stream_and_agent_are_saved_with_the_day():
    session_store.save(_trading(_running_state()))
    run = session_store.load()["run"]
    assert run["agent"] == _RUN_SPEC
    assert run["stream"] == {
        "symbols": ["AAPL", "MSFT"], "timeframe": "1Min", "feed": "iex",
        "data_source": "finnhub", "history_feed": "auto",
    }


def test_a_stopped_agent_and_stream_leave_nothing_to_resume():
    state = _trading(_running_state())
    state.agent_running = False               # ⏹ Stop Agent, a stop-out, the end of a run
    session_store.save(state)
    assert session_store.load()["run"]["agent"] is None

    state.bars_fallback_stop_event.set()      # ⏹ Stop
    session_store.save(state)
    assert session_store.load()["run"] is None


def test_stopping_changes_the_autosave_signature():
    state = _trading(_running_state())
    before = session_store._signature(state)
    state.agent_running = False
    assert session_store._signature(state) != before
    stopped = session_store._signature(state)
    state.bars_fallback_stop_event.set()
    assert session_store._signature(state) != stopped


def test_a_run_interrupted_by_a_restart_is_handed_out_once():
    session_store.save(_trading(_running_state()))

    first = _restart_and_restore()
    second = AppState()
    session_store.restore(second)             # another tab, same process

    run = session_store.take_resume(first)
    assert run["agent"] == _RUN_SPEC and run["stream"]["symbols"] == ["AAPL", "MSFT"]
    assert session_store.take_resume(first) is None    # a later rerun
    assert session_store.take_resume(second) is None   # not the owner


def test_only_the_owner_of_the_day_resumes_it():
    session_store.save(_trading(_running_state()))
    session_store._owners.clear()
    session_store._resumed.clear()
    looking, owner = AppState(), AppState()
    session_store.restore(looking)
    session_store.claim(owner)                # pressed ▶ Start elsewhere first
    assert session_store.take_resume(looking) is None


def test_nothing_is_resumed_from_a_day_saved_while_stopped():
    session_store.save(_running_state())      # a ledger, nothing running
    assert session_store.take_resume(_restart_and_restore()) is None


def test_a_stream_alone_is_kept_and_resumed_without_a_ledger():
    state = _streaming(AppState())
    path = session_store.save(state)
    assert path is not None and session_store.load()["tracker"] is None

    restored = _restart_and_restore()
    assert restored.decision_tracker is None
    assert restored.session_restored["decisions"] == restored.session_restored["log"] == 0
    assert session_store.take_resume(restored)["stream"]["symbols"] == ["AAPL", "MSFT"]


def test_a_day_with_only_a_stream_is_not_archived_on_a_fresh_start():
    session_store.save(_streaming(AppState()))
    assert session_store.archive() is None
    assert session_store.path_for("2026-09-28").exists()
