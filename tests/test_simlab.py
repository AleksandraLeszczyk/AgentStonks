"""Tests for the SimLab simulation suite (clock, store, market, engine, scores)."""
import json
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from agent_stonks import clock
from simlab import data as sim_data
from simlab import prompts as sim_prompts
from simlab import results as sim_results
from simlab.engine import SimulationConfig, SimulationEngine
from simlab.judge import _entry_context, _first_exit_after
from simlab.market import SimMarket
from simlab.patches import simulation_context

DAY = date(2026, 6, 15)  # a Monday
OPEN_UTC = datetime(2026, 6, 15, 13, 30, tzinfo=timezone.utc)  # 09:30 EDT


def _bar(ts: datetime, close: float, volume: float = 1000.0) -> dict:
    return {
        "t": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "o": close - 0.05,
        "h": close + 0.1,
        "l": close - 0.1,
        "c": close,
        "v": volume,
    }


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """Point the SimLab store at a temp dir and populate one synthetic day:
    60 minute bars ramping 100.0 -> 105.9, plus 30 prior daily bars."""
    monkeypatch.setattr(sim_data, "STORE_DIR", tmp_path / "store")
    monkeypatch.setattr(sim_data, "MANIFEST_PATH", tmp_path / "datasets.json")
    minute_bars = [
        _bar(OPEN_UTC + timedelta(minutes=i), 100.0 + 0.1 * i) for i in range(60)
    ]
    sim_data._write_gz(sim_data.bars_path("TEST", DAY), minute_bars)
    daily = [
        _bar(datetime(2026, 6, 15, tzinfo=timezone.utc) - timedelta(days=i), 99.0)
        for i in range(30, 0, -1)
    ]
    sim_data._write_gz(sim_data.daily_path("TEST"), {
        "symbol": "TEST", "start": "2026-05-16", "end": "2026-06-15", "bars": daily,
    })
    return tmp_path


class TestClock:
    def test_pin_and_clear(self):
        pinned = datetime(2026, 6, 15, 14, 0, tzinfo=timezone.utc)
        try:
            clock.set_simulated(pinned)
            assert clock.now() == pinned
            assert clock.monotonic() == pinned.timestamp()
            assert clock.is_simulated()
        finally:
            clock.clear()
        assert not clock.is_simulated()
        assert abs((clock.now() - datetime.now(timezone.utc)).total_seconds()) < 5

    def test_rejects_naive_datetime(self):
        with pytest.raises(ValueError):
            clock.set_simulated(datetime(2026, 6, 15, 14, 0))


class TestMarket:
    def test_completed_bars_respect_bar_completion(self, store):
        market = SimMarket(["TEST"], [DAY])
        # At 13:31:00 exactly bar 0 (13:30) has completed; bar 1 has not.
        t = OPEN_UTC + timedelta(minutes=1)
        bars = market.completed_bars("TEST", t)
        assert len(bars) == 1
        assert market.price_at("TEST", t) == 100.0

    def test_daily_bars_include_partial_today(self, store):
        market = SimMarket(["TEST"], [DAY])
        t = OPEN_UTC + timedelta(minutes=10)
        daily = market.daily_bars_at("TEST", t)
        assert daily[-1]["t"].startswith("2026-06-15")
        assert daily[-1]["c"] == pytest.approx(100.9)  # close of bar 9
        assert market.prev_close("TEST", t) == 99.0

    def test_step_times_cover_every_bar(self, store):
        market = SimMarket(["TEST"], [DAY])
        steps = market.step_times(DAY)
        assert len(steps) == 60
        assert steps[0] == OPEN_UTC + timedelta(minutes=1)


class TestDatasetStore:
    def test_create_dataset_downloads_only_missing_days(self, store, monkeypatch):
        calls = []

        def fake_minute(symbol, day, key, secret, feed="iex"):
            calls.append(day)
            return [_bar(OPEN_UTC, 100.0)] if day.weekday() < 5 else []

        monkeypatch.setattr(sim_data, "fetch_minute_bars_day", fake_minute)
        monkeypatch.setattr(sim_data, "fetch_news_day", lambda *a, **k: [])
        monkeypatch.setattr(sim_data, "fetch_daily_bars_range", lambda *a, **k: [_bar(OPEN_UTC, 99.0)])
        monkeypatch.setattr(sim_data, "fetch_market_indicator_closes", lambda *a, **k: {"spy": [], "vix": [], "vix3m": []})

        ds = sim_data.create_dataset("wk", ["TEST"], date(2026, 6, 15), date(2026, 6, 17), "k", "s")
        # 2026-06-15 already in the store (fixture) -- only 16th and 17th fetched.
        assert calls == [date(2026, 6, 16), date(2026, 6, 17)]
        assert ds.days == ["2026-06-15", "2026-06-16", "2026-06-17"]
        assert sim_data.get_dataset("wk").symbols == ["TEST"]
        sim_data.delete_dataset("wk")
        assert sim_data.get_dataset("wk") is None


def _tool_call(call_id, name, arguments):
    return SimpleNamespace(
        id=call_id, function=SimpleNamespace(name=name, arguments=json.dumps(arguments))
    )


def _response(tool_calls, content=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        outer = self

        class _Completions:
            def create(self, model, messages, tools, tool_choice):
                outer.calls.append(messages)
                return outer._responses.pop(0)

        self.chat = SimpleNamespace(completions=_Completions())


def _scripted_client():
    """Cycle 1: arm a buy-the-breakout tactic at 103 and finalize with alert.
    Cycle 2 (woken by the fill): sell everything. Cycle 3 (cycle timer): stand
    aside on a far-away alert for the rest of the day."""
    cycle1 = _response([
        _tool_call("c1a", "set_tactics", {
            "symbol": "TEST",
            "actions": [{
                "action": "buy", "quantity": 10,
                "conditions": [{"field": "last_price", "condition": "above", "value": 103.0}],
                "note": "breakout entry",
            }],
            "reasoning": "buy the break of 103",
        }),
        _tool_call("c1b", "submit_decision", {
            "action": "alert", "regime": "bullish", "reasoning": "waiting for the break", "alerts": [],
        }),
    ])
    cycle2 = _response([
        _tool_call("c2a", "submit_decision", {
            "action": "sell", "symbol": "TEST", "quantity": 10,
            "regime": "bullish", "reasoning": "taking the breakout profit",
        }),
    ])
    cycle3 = _response([
        _tool_call("c3a", "submit_decision", {
            "action": "alert", "regime": "neutral", "reasoning": "nothing to do",
            "alerts": [{"symbol": "TEST", "field": "last_price", "condition": "above", "value": 99999}],
        }),
    ])
    return FakeClient([cycle1, cycle2, cycle3])


def _run_sim(store, cycle_minutes=5):
    market = SimMarket(["TEST"], [DAY])
    config = SimulationConfig(
        personality="momentum", provider="openai", model="fake", api_key="",
        symbols=["TEST"], days=[DAY], starting_cash=10_000.0, cycle_minutes=cycle_minutes,
    )
    engine = SimulationEngine(market, config)
    result = engine.run(client=_scripted_client())
    return market, result


class TestEngine:
    def test_full_session_replay(self, store):
        market, result = _run_sim(store)
        assert result.error is None
        assert result.cycles_run == 3

        actions = [(d["action"], d["status"]) for d in result.decisions]
        assert actions == [
            ("tactics", "armed"),  # cycle 1: arm the breakout entry
            ("alert", "noop"),  # cycle 1: finalize (empty alerts, tactics armed)
            ("buy", "filled"),  # tactic fires mid-fast-forward
            ("sell", "filled"),  # cycle 2, woken by the fill
            ("alert", "noop"),  # cycle 3, cycle timer
        ]
        buy, sell = result.decisions[2], result.decisions[3]
        # The tactic fired on the first bar closing >= 103 (bar 30, completes
        # 14:01Z) and filled at that bar's close -- simulated time throughout.
        assert buy["price"] == pytest.approx(103.0)
        assert buy["ts"].startswith("2026-06-15T14:01")
        assert "Tactics triggered" in buy["reasoning"]
        assert sell["price"] >= buy["price"]  # ramping tape: sold at/above the entry

        # Equity: one point per completed bar, valued in simulated time.
        assert len(result.equity) == 60
        assert result.equity[0]["ts"].startswith("2026-06-15")
        fees = 2 * 1.15
        expected_final = 10_000.0 + 10 * (sell["price"] - buy["price"]) - fees
        assert result.final_value == pytest.approx(expected_final)

    def test_clock_restored_after_run(self, store):
        _run_sim(store)
        assert not clock.is_simulated()

    def test_summary_and_oracle(self, store):
        market, result = _run_sim(store)
        summary = sim_results.summarize_run(result, market)
        assert summary["trades_filled"] == 2
        # Oracle: buy 100.0, sell 105.9 -> 5.9%.
        assert summary["oracle_ceiling_pct"] == pytest.approx(5.9, abs=0.01)
        assert summary["profit_efficiency"] is not None
        assert summary["return_pct"] == pytest.approx(
            (result.final_value / 10_000.0 - 1.0) * 100.0
        )


class TestOracle:
    def test_best_round_trip_orders_matter(self):
        assert sim_results.oracle_best_round_trip([105, 100, 104]) == pytest.approx(4.0)
        assert sim_results.oracle_best_round_trip([105, 104, 103]) == 0.0
        assert sim_results.oracle_best_round_trip([]) == 0.0


class TestJudgeContext:
    def test_entry_context_includes_tape_and_outcome(self, store):
        market, result = _run_sim(store)
        buy = result.decisions[2]
        exit_decision = _first_exit_after(buy, result.decisions)
        assert exit_decision is not None and exit_decision["action"] == "sell"
        context = _entry_context(buy, exit_decision, market)
        assert "ENTRY: buy" in context
        assert "TAPE BEFORE ENTRY" in context
        assert "max favorable excursion" in context
        assert "EXIT: sold" in context


class TestPatches:
    def test_market_indicators_clip_to_sim_time(self, store):
        sim_data._write_gz(sim_data.market_path(), {
            "spy": [
                {"date": "2026-06-12", "close": 500.0},
                {"date": "2026-06-15", "close": 501.0},
                {"date": "2026-06-16", "close": 502.0},
            ],
            "vix": [{"date": "2026-06-15", "close": 15.0}],
            "vix3m": [],
        })
        market = SimMarket(["TEST"], [DAY])
        from agent_stonks import historical

        with simulation_context(market):
            clock.set_simulated(OPEN_UTC)
            series = historical.fetch_market_indicators()
            # 2026-06-16 is the future from the pinned clock -- must be clipped.
            assert list(series["spy"].values) == [500.0, 501.0]
            assert float(series["vix"].iloc[-1]) == 15.0
        assert not clock.is_simulated()

    def test_patches_are_restored(self, store):
        from agent_stonks import historical

        original = historical.fetch_market_indicators
        with simulation_context(SimMarket(["TEST"], [DAY])):
            assert historical.fetch_market_indicators is not original
        assert historical.fetch_market_indicators is original


class TestPrompts:
    def test_override_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sim_prompts, "PROMPTS_DIR", tmp_path / "prompts")
        assert sim_prompts.get_prompt("momentum") == sim_prompts.default_prompt("momentum")
        assert not sim_prompts.has_override("momentum")
        sim_prompts.save_override("momentum", "You are a test agent.")
        assert sim_prompts.get_prompt("momentum") == "You are a test agent."
        sim_prompts.reset_override("momentum")
        assert sim_prompts.get_prompt("momentum") == sim_prompts.default_prompt("momentum")

    def test_unknown_personality_rejected(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sim_prompts, "PROMPTS_DIR", tmp_path / "prompts")
        with pytest.raises(KeyError):
            sim_prompts.save_override("nope", "x")
