"""Apple Trader on HighLow2 + GoodMoments: GoodMoments' rules, HighLow2 voting.

These pin the hybrid's RULES with both forecasts scripted: the opening window
traded on GoodMoments' edge alone, every later buy needing the bar's close at or
under HighLow2's buy level as well, that level moved through a breach as a
HighLow2 run would move it, every exit GoodMoments' -- plus its defaults, its
signature, its tuning axes and its chart overlays.
`tests/test_goodmoments_trader.py` pins GoodMoments' rules themselves.
"""

import math
from dataclasses import asdict, replace
from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from agent_stonks import apple_models, base_configs, clock, model_overlays as mo, orchestra
from agent_stonks import apple_trader as at
from agent_stonks.apple_trader import AppleTraderConfig, config_signature
from agent_stonks.broker import Broker
from agent_stonks.config import MODEL_OVERLAY_COLORS
from agent_stonks.decisions import DecisionTracker
from agent_stonks.state import AppState

TICKER = "AAPL"
KEY = apple_models.HIGHLOW2_GOODMOMENTS_KEY
# 10:30 ET on a Tuesday.
MIDSESSION = datetime(2026, 7, 21, 14, 30, tzinfo=timezone.utc)
ADR = 0.02                       # GoodMoments' take +0.4%, stop -1.5% (log terms)
ENTER = 0.02
HL2 = {"kind": "highlow2", "opening_minutes": 5}
BUNDLE = {"kind": "goodmoments", "path": "stub", apple_models.HIGHLOW2_KEY: HL2}
# HighLow2's 9:35 forecast: a $4 predicted range, so the 0.6 buy level rests at
# 102 - 0.6 x 4 = 99.60.
FORECAST = {"pred_high": 102.0, "pred_low": 98.0, "adr14_abs": 3.0, "or_high": 100.5, "or_low": 99.5}
BUY_LEVEL = 99.6


class FakeBroker(Broker):
    def __init__(self, price: float = 100.0):
        self.price = price

    def get_current_price(self, symbol, key, secret, feed="iex") -> float:
        return self.price

    def submit_order(self, symbol, side, quantity, price) -> dict:
        return {"status": "filled", "filled_qty": quantity, "filled_price": price}


@pytest.fixture
def state() -> AppState:
    s = AppState()
    s.set_symbols([TICKER])
    s.api_key, s.api_secret, s.feed = "k", "s", "iex"
    return s


@pytest.fixture(autouse=True)
def market_open():
    clock.set_simulated(MIDSESSION)
    yield
    clock.clear()


@pytest.fixture(autouse=True)
def no_bases(monkeypatch, tmp_path):
    """No base configuration on disk but the ones a test writes."""
    monkeypatch.setattr(base_configs, "BASE_PATH", tmp_path / "apple_trader_base.json")
    monkeypatch.setattr(base_configs, "_cache", None)


class Tape:
    """Today's closed bars from 09:30, the edge scripted for each, and
    HighLow2's forecast (`FORECAST`, or an exception) for the opening window."""

    OPEN = pd.Timestamp("2026-07-21 09:30", tz="America/New_York")

    def __init__(self, monkeypatch, broker: FakeBroker):
        self.broker = broker
        self.rows: "list[dict]" = []
        self.index: "list[pd.Timestamp]" = []
        self.forecasts: "dict[int, dict]" = {}
        self.highlow2: "dict | Exception" = dict(FORECAST)
        self.windows: "list[int]" = []
        tape = self

        def forecast_at(bundle, ticker, day, minute, key=None, secret=None):
            found = tape.forecasts.get(minute)
            if found is None:
                raise ValueError(f"no forecast at {minute}")
            return {"minute": minute, "p": 0.5 + found["edge"], "p_tv": 0.5, "edge": found["edge"],
                    "price": found["price"], "adr": ADR, "p_reach": 0.5}

        fake = SimpleNamespace(
            forecast_at=forecast_at, session_adr=lambda *a, **k: ADR, LAST_FORECAST=388,
            TARGET_ADR=0.2, is_half_day=lambda day: False, is_session=lambda day: True,
            warm=lambda *a, **k: None,
        )
        monkeypatch.setattr(at, "_goodmoments", lambda: fake)
        monkeypatch.setattr(at.momentum_regime, "minute_frame", lambda *a, **k: self.frame())

        def opening_window(state, frame, want, ticker=TICKER):
            tape.windows.append(want)
            return frame.iloc[:want], "iex"

        def session_forecast(bundle, ticker, opening, today, key=None, secret=None):
            assert bundle is HL2
            if isinstance(tape.highlow2, Exception):
                raise tape.highlow2
            return dict(tape.highlow2), None

        monkeypatch.setattr(at, "fetch_opening_window", opening_window)
        monkeypatch.setattr(at, "session_forecast", session_forecast)

    def append(self, close: float, edge: float = 0.0, low: "float | None" = None,
               high: "float | None" = None):
        minute = len(self.rows)
        self.index.append(self.OPEN + pd.Timedelta(minutes=minute))
        self.rows.append({"open": close, "high": close if high is None else high,
                          "low": close if low is None else low, "close": close, "volume": 1e5})
        self.forecasts[minute] = {"edge": edge, "price": close}
        self.broker.price = close

    def window(self, close: float = 100.0) -> None:
        """The five opening bars, inside the forecast, with no edge."""
        for _ in range(5):
            self.append(close)

    def frame(self) -> pd.DataFrame:
        return at.momentum_regime.add_session_columns(
            pd.DataFrame(self.rows, index=pd.DatetimeIndex(self.index))
        )


def hybrid_config(**kwargs) -> AppleTraderConfig:
    """Notebook 4's GoodMoments rules and HighLow2's AAPL buy distance under
    the shipped breach rule; no days off."""
    kwargs.setdefault("skip_events", ())
    kwargs.setdefault("buy_k", 0.6)
    return AppleTraderConfig(model_key=KEY, ticker=TICKER, **kwargs)


def _setup(monkeypatch, state, **config):
    broker = FakeBroker(100.0)
    tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
    tape = Tape(monkeypatch, broker)
    trader = at.build_trader(hybrid_config(**config), BUNDLE)
    assert isinstance(trader, at.HighLow2GoodMomentsTrader)
    return trader, tracker, tape


def cycle(trader, state, tracker) -> str:
    outcome = trader.run_cycle(BUNDLE, state, tracker)
    trader.publish_memory(state)
    return outcome


def _texts(state, kind: str) -> "list[str]":
    return [e["text"] for e in state.agent_log if e.get("type") == kind]


def _exits(state) -> "list[str]":
    return [e.get("exit") for e in state.agent_log if e.get("type") == "decision" and e.get("action") == "sell"]


class TestOpeningWindow:
    def test_the_edge_alone_buys_in_the_first_five_minutes(self, monkeypatch, state):
        """09:32, well above where HighLow2 will put its buy: GoodMoments'
        signal is the only one there is."""
        trader, tracker, tape = _setup(monkeypatch, state)
        tape.append(101.0)
        tape.append(101.0)
        assert cycle(trader, state, tracker) == "hold"
        tape.append(101.5, edge=ENTER)
        assert cycle(trader, state, tracker) == "bought"
        decision = [e for e in state.agent_log if e.get("type") == "decision"][-1]
        assert "opening window" in decision["reasoning"]
        assert trader.levels is None

    def test_the_forecast_is_made_at_the_close_of_the_window(self, monkeypatch, state):
        trader, tracker, tape = _setup(monkeypatch, state)
        for minute in range(4):
            tape.append(100.0)
            cycle(trader, state, tracker)
            assert trader.levels is None
        tape.append(100.0)                                              # 09:34
        cycle(trader, state, tracker)
        assert tape.windows == [5]
        assert trader.levels.plan["buy_level"] == pytest.approx(BUY_LEVEL)
        assert any("HighLow2's forecast" in t for t in _texts(state, "analysis"))

    def test_the_last_bar_of_the_window_is_still_goodmoments_alone(self, monkeypatch, state):
        trader, tracker, tape = _setup(monkeypatch, state)
        for _ in range(4):
            tape.append(101.0)
        tape.append(101.0, edge=ENTER)                                  # 09:34, above the level
        assert cycle(trader, state, tracker) == "bought"


class TestBothHaveToAgree:
    def _after_window(self, monkeypatch, state, **config):
        trader, tracker, tape = _setup(monkeypatch, state, **config)
        tape.window()
        cycle(trader, state, tracker)
        return trader, tracker, tape

    def test_an_edge_above_the_level_buys_nothing_and_says_so_once(self, monkeypatch, state):
        trader, tracker, tape = self._after_window(monkeypatch, state)
        for _ in range(3):
            tape.append(100.0, edge=0.2)
            assert cycle(trader, state, tracker) == "hold"
        assert tracker.position_for(TICKER) == 0
        refusals = [t for t in _texts(state, "status") if "both models have to agree" in t]
        assert len(refusals) == 1 and "$99.60 buy level" in refusals[0]

    def test_an_edge_with_the_close_at_the_level_buys(self, monkeypatch, state):
        trader, tracker, tape = self._after_window(monkeypatch, state)
        tape.append(BUY_LEVEL, edge=ENTER)
        assert cycle(trader, state, tracker) == "bought"
        decision = [e for e in state.agent_log if e.get("type") == "decision"][-1]
        assert "HighLow2 agrees" in decision["reasoning"]

    def test_the_level_without_the_edge_buys_nothing(self, monkeypatch, state):
        trader, tracker, tape = self._after_window(monkeypatch, state)
        tape.append(99.0, edge=ENTER - 0.001)
        assert cycle(trader, state, tracker) == "hold"
        assert tracker.position_for(TICKER) == 0

    def test_it_is_the_close_not_a_dip_to_the_level(self, monkeypatch, state):
        """A market order after a bar that dipped to the level and recovered
        pays the recovery -- HighLow2's own limit would not have filled."""
        trader, tracker, tape = self._after_window(monkeypatch, state)
        tape.append(100.0, edge=0.2, low=99.0)
        assert cycle(trader, state, tracker) == "hold"
        tape.append(99.5, edge=0.2)
        assert cycle(trader, state, tracker) == "bought"

    def test_a_new_stretch_of_refusals_is_said_again(self, monkeypatch, state):
        trader, tracker, tape = self._after_window(monkeypatch, state)
        tape.append(100.0, edge=0.2)
        first = cycle(trader, state, tracker)
        tape.append(100.0, edge=0.0)
        cycle(trader, state, tracker)
        tape.append(100.0, edge=0.2)
        cycle(trader, state, tracker)
        assert first == "hold"
        assert len([t for t in _texts(state, "status") if "both models have to agree" in t]) == 2

    def test_a_breach_moves_the_level_as_a_highlow2_run_would(self, monkeypatch, state):
        trader, tracker, tape = self._after_window(monkeypatch, state)
        walker = at.DayRangeTrader(hybrid_config())
        walker.plan = {"date": pd.Timestamp("2026-07-21"), "opening_end": tape.index[4],
                       "open_price": 100.0, **FORECAST}
        tape.append(103.0, high=103.5)                                  # through the 102 high
        cycle(trader, state, tracker)
        walker._move_range(tape.frame(), tape.index[-1])
        walker._set_levels(tape.index[-1])
        assert trader.levels.plan["buy_level"] > BUY_LEVEL
        assert trader.levels.plan["buy_level"] == pytest.approx(walker.plan["buy_level"])
        assert any("moved at 09:35" in t for t in _texts(state, "analysis"))
        assert state.apple_trader_levels["rows"][-1]["buy"] == pytest.approx(walker.plan["buy_level"])

    def test_the_record_has_the_level_only_after_the_window(self, monkeypatch, state):
        trader, tracker, tape = self._after_window(monkeypatch, state)
        tape.append(100.0)
        cycle(trader, state, tracker)
        rows = state.apple_trader_levels["rows"]
        assert rows[-2]["buy"] is None                                  # 09:34
        assert rows[-1]["buy"] == pytest.approx(BUY_LEVEL)              # 09:35
        assert rows[-1]["pred_high"] == pytest.approx(102.0)


class TestNoHighLow2Forecast:
    def test_the_window_still_trades_and_nothing_after_it(self, monkeypatch, state):
        trader, tracker, tape = _setup(monkeypatch, state)
        tape.highlow2 = ValueError("no IEX opening")
        for _ in range(5):
            tape.append(100.0)
            cycle(trader, state, tracker)
        errors = _texts(state, "error")
        assert len(errors) == 1 and "no IEX opening" in errors[0]
        tape.append(95.0, edge=0.2)                                     # way under any level
        assert cycle(trader, state, tracker) == "hold"
        assert any("HighLow2 has no forecast today (no IEX opening)" in t for t in _texts(state, "status"))
        tape.append(95.0, edge=0.2)
        cycle(trader, state, tracker)
        assert len(_texts(state, "error")) == 1                         # said once

    def test_a_position_bought_in_the_window_keeps_its_exits(self, monkeypatch, state):
        trader, tracker, tape = _setup(monkeypatch, state)
        tape.highlow2 = ValueError("no IEX opening")
        tape.append(100.0, edge=ENTER)
        assert cycle(trader, state, tracker) == "bought"
        for _ in range(4):
            tape.append(100.0)
            cycle(trader, state, tracker)
        tape.append(100.5, high=100.0 * math.exp(0.2 * ADR) + 0.01)
        assert cycle(trader, state, tracker) == "sold"
        assert _exits(state) == [at.GoodMomentsTrader.EXIT_TAKE]


class TestGoodMomentsExits:
    def _bought(self, monkeypatch, state, **config):
        trader, tracker, tape = _setup(monkeypatch, state, **config)
        tape.window()
        cycle(trader, state, tracker)
        tape.append(99.5, edge=ENTER)
        assert cycle(trader, state, tracker) == "bought"
        return trader, tracker, tape

    def test_the_take_is_goodmoments_not_highlow2s_sell(self, monkeypatch, state):
        trader, tracker, tape = self._bought(monkeypatch, state)
        assert trader.entry["take"] == pytest.approx(99.5 * math.exp(0.2 * ADR))
        assert trader.entry["stop"] == pytest.approx(99.5 * math.exp(-0.75 * ADR))
        tape.append(99.8, high=trader.entry["take"] + 0.01)
        assert cycle(trader, state, tracker) == "sold"
        assert _exits(state) == [at.GoodMomentsTrader.EXIT_TAKE]

    def test_a_stop_pauses_entries_and_does_not_end_the_run(self, monkeypatch, state):
        trader, tracker, tape = self._bought(monkeypatch, state)        # bar 5
        tape.append(98.0, low=trader.entry["stop"] - 0.01, edge=0.2)    # bar 6: stopped
        assert cycle(trader, state, tracker) == "sold"
        assert _exits(state) == [at.GoodMomentsTrader.EXIT_STOP]
        assert trader.halt is None
        for _ in range(14):                                             # bars 7..20
            tape.append(98.0, edge=0.2)
            assert cycle(trader, state, tracker) == "hold"
        tape.append(98.0, edge=0.2)                                     # bar 21 = 6 + 15
        assert cycle(trader, state, tracker) == "bought"


class TestLateStart:
    def test_a_run_started_at_10_reads_the_window_and_walks_the_breaches(self, monkeypatch, state):
        trader, tracker, tape = _setup(monkeypatch, state)
        tape.window()
        tape.append(103.0, high=103.5)                                  # 09:35 breach
        for _ in range(25):
            tape.append(102.5)
        cycle(trader, state, tracker)
        assert tape.windows == [5]
        assert trader.levels.plan["buy_level"] > BUY_LEVEL
        assert trader.levels.plan.get("range_updates")


class TestWiring:
    def test_it_is_offered_where_both_models_are(self):
        assert KEY in apple_models.keys_for(TICKER)
        assert apple_models.get(KEY).tickers == (TICKER,)
        assert apple_models.strategy(KEY) == apple_models.STRATEGY_PROBABILITY
        assert apple_models.signal_of(KEY) == apple_models.GOODMOMENTS_KEY
        assert apple_models.levels_of(KEY) == apple_models.HIGHLOW2_KEY

    def test_it_starts_from_goodmoments_rules_and_highlow2s_buy(self):
        base_configs.set_base({
            **asdict(AppleTraderConfig(model_key="goodmoments", ticker=TICKER)),
            "gm_enter": 0.08, "gm_stop_k": 0.3, "skip_events": [],
        })
        config = at.default_config(TICKER, KEY)
        assert (config.model_key, config.gm_enter, config.gm_stop_k, config.skip_events) == (KEY, 0.08, 0.3, ())
        hl2 = at.default_config(TICKER, apple_models.HIGHLOW2_KEY)
        assert (config.buy_k, config.sell_k) == (hl2.buy_k, 0.0)
        assert (config.level_unit, config.breach_update) == (hl2.level_unit, hl2.breach_update)
        assert AppleTraderConfig(model_key=KEY).buy_k == pytest.approx(0.60)

    def test_its_own_base_wins(self):
        base_configs.set_base({**asdict(hybrid_config(buy_k=0.9, gm_enter=0.05))})
        config = at.default_config(TICKER, KEY)
        assert (config.buy_k, config.gm_enter) == (0.9, 0.05)

    def test_its_signature_adds_the_buy_level_to_goodmoments(self):
        config = hybrid_config(skip_events=())
        assert config_signature(config) == (
            "highlow2_goodmoments_AAPL(enter=+0.02,exit=off,take=0.2A,stop=0.75A,raise=0.05/stop,"
            "buy<=H-0.6R,breach=brownian,keep_width,size=95%)"
        )
        assert config_signature(replace(config, buy_k=0.8)) != config_signature(config)
        assert config_signature(replace(config, sell_k=0.3, stop_gain_fraction=0.9)) == config_signature(config)
        goodmoments = AppleTraderConfig(model_key="goodmoments", skip_events=())
        assert "buy<=" not in config_signature(goodmoments)

    def test_tuning_offers_goodmoments_axes_and_the_buy_level(self):
        from simlab import tuning

        offered = set(tuning.axes_for(KEY, tuning.SWEEPABLE))
        assert offered == set(tuning.PROBABILITY_AXES) | {"buy_k", "breach_update"}
        assert tuning.default_axes(KEY) == ("gm_enter", "buy_k")
        assert tuning.default_axes("goodmoments") == ("gm_enter", "gm_stop_k")
        assert tuning.axes_family(KEY) != tuning.axes_family("goodmoments")

    def test_orchestra_does_not_race_it(self):
        assert not [p for p in orchestra.all_pairs() if p.endswith(f":{KEY}")]

    def test_its_runs_draw_both_models(self):
        drawn = mo.for_models([KEY], TICKER)["keys"]
        assert mo.GOODMOMENTS_SIGNAL_KEY in drawn and mo.HIGHLOW2_RANGE_KEY in drawn

    def test_a_runs_record_draws_the_buy_level_beside_the_take_and_stop(self):
        stamps = [pd.Timestamp("2026-07-21 09:34", tz="America/New_York") + pd.Timedelta(minutes=i)
                  for i in range(3)]
        rows = [
            {"t": stamps[0], "buy": None, "take": None, "stop": None},
            {"t": stamps[1], "buy": 99.6, "take": None, "stop": None},
            {"t": stamps[2], "buy": 99.7, "take": 100.2, "stop": 98.1},
        ]
        items, note = mo._signal_trader_items(TICKER, (rows, hybrid_config()), KEY)
        labels = [item["label"] for item in items]
        assert note == "" and labels == ["HighLow2 buy level", "Take-profit", "Stop level"]

    def test_a_live_runs_record_draws_highlow2s_range_from_the_rows_with_a_forecast(self, monkeypatch):
        """The record starts at 09:30, five rows before HighLow2 has a forecast;
        those are not a range to draw."""
        monkeypatch.setattr(mo, "_highlow_forecast", lambda *a, **k: pytest.fail("forecast"))
        open_ = pd.Timestamp("2026-07-21 09:30", tz="America/New_York")
        stamps = [open_ + pd.Timedelta(minutes=i) for i in range(10)]
        bars = [{"t": s.tz_convert("UTC").isoformat(), "o": 100.0, "h": 100.2, "l": 99.8,
                 "c": 100.0, "v": 1e5} for s in stamps]
        rows = [{"t": s, "model_key": KEY, "price": 100.0, "edge": 0.0, "take": None, "stop": None,
                 "buy": None if i < 5 else BUY_LEVEL,
                 "pred_high": None if i < 5 else 102.0,
                 "pred_low": None if i < 5 else 98.0} for i, s in enumerate(stamps)]
        history = {"ticker": TICKER, "date": pd.Timestamp("2026-07-21"), "config": hybrid_config(),
                   "rows": rows, "seed": None}
        out = mo.compute([mo.HIGHLOW2_RANGE_KEY, mo.TRADER_LEVELS_KEY], TICKER, bars, daily_bars=[],
                         session_date="2026-07-21", trader_config=hybrid_config(),
                         trader_history=history)
        assert out["notes"] == []
        assert {item["key"] for item in out["items"]} == {mo.HIGHLOW2_RANGE_KEY, mo.TRADER_LEVELS_KEY}
        assert any(item.get("label") == "HighLow2 buy level" for item in out["items"])

    def test_without_a_run_the_buy_level_is_walked(self):
        session = pd.DataFrame(
            {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1e5},
            index=pd.date_range("2026-07-21 09:30", periods=8, freq="1min", tz="America/New_York"),
        )
        session = at.momentum_regime.add_session_columns(session)
        result = {"forecast": dict(FORECAST), "made_at": session.index[4], "problem": ""}
        walked = mo._hybrid_buy_walk(TICKER, session, result, hybrid_config(), KEY)
        items, note = mo._signal_trader_items(TICKER, None, KEY, walked)
        assert note == "" and items[0]["label"] == "HighLow2 buy level"
        assert items[0]["values"][0] == pytest.approx(BUY_LEVEL)
        # On the live chart with no agent running, the walk is gray.
        idle, _ = mo._signal_trader_items(TICKER, None, KEY, walked, idle=True)
        assert items[0]["color"] == MODEL_OVERLAY_COLORS["trader_levels"]
        assert idle[0]["color"] == MODEL_OVERLAY_COLORS["trader_idle"]
        failed = {"forecast": None, "made_at": None, "problem": "no IEX opening"}
        items, note = mo._signal_trader_items(
            TICKER, None, KEY, mo._hybrid_buy_walk(TICKER, session, failed, hybrid_config(), KEY)
        )
        assert items == [] and "no IEX opening" in note
