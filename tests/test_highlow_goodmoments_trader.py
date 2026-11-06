"""Apple Trader on HighLow + GoodMoments: GoodMoments' rules, HighLow voting.

The hybrid is HighLow2 + GoodMoments' trader with HighLow's forecast in the
vote (`apple_trader.HybridGoodMomentsTrader`), so these pin what differs --
whose forecast the vote reads and names, the defaults and signature (on AMD,
the first ticker it was asked for), the wiring on all four tickers both models
are fitted on -- and what both hybrids now share: a restart the same day votes with
the forecast the earlier run made, not a new one.
`tests/test_highlow2_goodmoments_trader.py` pins the rules themselves.
"""

import math
from dataclasses import asdict, replace
from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from agent_stonks import apple_models, base_configs, clock, model_catalogue, model_overlays as mo
from agent_stonks import orchestra, session_store
from agent_stonks import apple_trader as at
from agent_stonks.apple_trader import AppleTraderConfig, config_signature
from agent_stonks.broker import Broker
from agent_stonks.decisions import DecisionTracker
from agent_stonks.state import AppState

TICKER = "AMD"
KEY = apple_models.HIGHLOW_GOODMOMENTS_KEY
# 10:30 ET on a Tuesday.
MIDSESSION = datetime(2026, 7, 21, 14, 30, tzinfo=timezone.utc)
ADR = 0.02
ENTER = 0.02
HL = {"kind": "highlow", "opening_minutes": 5}
# HighLow's 9:35 forecast: a $4 predicted range, so the 0.6 buy level rests at
# 102 - 0.6 x 4 = 99.60.
FORECAST = {"pred_high": 102.0, "pred_low": 98.0, "adr14_abs": 3.0, "or_high": 100.5, "or_low": 99.5}
BUY_LEVEL = 99.6
# What the same model says when asked again after 09:50, on SIP's windows.
REMADE = {**FORECAST, "pred_high": 103.0, "pred_low": 98.5}


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
    s.set_symbols([TICKER, "AAPL"])
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
    """Today's closed bars from 09:30, the edge scripted for each, and the
    levels model's forecast (`forecast`, or an exception) for the opening
    window, counted."""

    OPEN = pd.Timestamp("2026-07-21 09:30", tz="America/New_York")

    def __init__(self, monkeypatch, broker: FakeBroker, part: dict):
        self.broker = broker
        self.rows: "list[dict]" = []
        self.index: "list[pd.Timestamp]" = []
        self.edges: "dict[int, float]" = {}
        self.forecast: "dict | Exception" = dict(FORECAST)
        self.forecasts_made = 0
        tape = self

        def forecast_at(bundle, ticker, day, minute, key=None, secret=None):
            if minute not in tape.edges:
                raise ValueError(f"no forecast at {minute}")
            edge = tape.edges[minute]
            return {"minute": minute, "p": 0.5 + edge, "p_tv": 0.5, "edge": edge,
                    "price": tape.rows[minute]["close"], "adr": ADR, "p_reach": 0.5}

        fake = SimpleNamespace(
            forecast_at=forecast_at, session_adr=lambda *a, **k: ADR, LAST_FORECAST=388,
            TARGET_ADR=0.2, is_half_day=lambda day: False, is_session=lambda day: True,
            warm=lambda *a, **k: None,
        )
        monkeypatch.setattr(at, "_goodmoments", lambda: fake)
        monkeypatch.setattr(at.momentum_regime, "minute_frame", lambda *a, **k: self.frame())
        monkeypatch.setattr(
            at, "fetch_opening_window", lambda state, frame, want, ticker=TICKER: (frame.iloc[:want], "iex")
        )

        def session_forecast(bundle, ticker, opening, today, key=None, secret=None):
            assert bundle is part
            tape.forecasts_made += 1
            if isinstance(tape.forecast, Exception):
                raise tape.forecast
            return dict(tape.forecast), None

        monkeypatch.setattr(at, "session_forecast", session_forecast)

    def append(self, close: float, edge: float = 0.0, low: "float | None" = None,
               high: "float | None" = None):
        minute = len(self.rows)
        self.index.append(self.OPEN + pd.Timedelta(minutes=minute))
        self.rows.append({"open": close, "high": close if high is None else high,
                          "low": close if low is None else low, "close": close, "volume": 1e5})
        self.edges[minute] = edge
        self.broker.price = close

    def window(self, close: float = 100.0) -> None:
        for _ in range(5):
            self.append(close)

    def frame(self) -> pd.DataFrame:
        return at.momentum_regime.add_session_columns(
            pd.DataFrame(self.rows, index=pd.DatetimeIndex(self.index))
        )


def hybrid_config(key: str = KEY, ticker: str = TICKER, **kwargs) -> AppleTraderConfig:
    kwargs.setdefault("skip_events", ())
    kwargs.setdefault("buy_k", 0.6)
    return AppleTraderConfig(model_key=key, ticker=ticker, **kwargs)


def _bundle(key: str = KEY) -> "tuple[dict, dict]":
    """The hybrid's bundle and its levels model's part in it."""
    part = {**HL, "kind": "highlow2"} if key == apple_models.HIGHLOW2_GOODMOMENTS_KEY else HL
    return {"kind": "goodmoments", "path": "stub", apple_models.levels_of(key): part}, part


def _setup(monkeypatch, state, key: str = KEY, ticker: str = TICKER, **config):
    broker = FakeBroker(100.0)
    tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
    bundle, part = _bundle(key)
    tape = Tape(monkeypatch, broker, part)
    trader = at.build_trader(hybrid_config(key, ticker, **config), bundle)
    return trader, tracker, tape, bundle


def cycle(trader, state, tracker, bundle) -> str:
    outcome = trader.run_cycle(bundle, state, tracker)
    trader.publish_memory(state)
    return outcome


def _texts(state, kind: str) -> "list[str]":
    return [e["text"] for e in state.agent_log if e.get("type") == kind]


def _exits(state) -> "list[str]":
    return [e.get("exit") for e in state.agent_log if e.get("type") == "decision" and e.get("action") == "sell"]


class TestHighLowVotes:
    def test_the_forecast_is_highlows_made_at_the_close_of_the_window(self, monkeypatch, state):
        trader, tracker, tape, bundle = _setup(monkeypatch, state)
        assert isinstance(trader, at.HighLowGoodMomentsTrader)
        for _ in range(4):
            tape.append(100.0)
            cycle(trader, state, tracker, bundle)
            assert trader.levels is None
        tape.append(100.0)                                              # 09:34
        cycle(trader, state, tracker, bundle)
        assert tape.forecasts_made == 1
        assert trader.levels.plan["buy_level"] == pytest.approx(BUY_LEVEL)
        assert any(t.startswith("HighLow's forecast for AMD at the 09:34 bar")
                   for t in _texts(state, "analysis"))
        assert any(t.startswith("Hybrid with HighLow:") for t in _texts(state, "analysis"))

    def test_the_window_buys_on_the_edge_alone(self, monkeypatch, state):
        trader, tracker, tape, bundle = _setup(monkeypatch, state)
        tape.append(101.0)
        tape.append(101.5, edge=ENTER)
        assert cycle(trader, state, tracker, bundle) == "bought"
        decision = [e for e in state.agent_log if e.get("type") == "decision"][-1]
        assert "before HighLow's forecast" in decision["reasoning"]

    def test_after_it_both_have_to_agree(self, monkeypatch, state):
        trader, tracker, tape, bundle = _setup(monkeypatch, state)
        tape.window()
        cycle(trader, state, tracker, bundle)
        tape.append(100.0, edge=0.2)
        assert cycle(trader, state, tracker, bundle) == "hold"
        refusal = [t for t in _texts(state, "status") if "both models have to agree" in t]
        assert len(refusal) == 1 and "above HighLow's $99.60 buy level" in refusal[0]
        tape.append(BUY_LEVEL, edge=ENTER)
        assert cycle(trader, state, tracker, bundle) == "bought"
        decision = [e for e in state.agent_log if e.get("type") == "decision"][-1]
        assert "HighLow agrees" in decision["reasoning"]
        assert "HighLow buy $99.60" in _texts(state, "analysis")[-1]

    def test_the_exits_are_goodmoments_with_amds_stop(self, monkeypatch, state):
        trader, tracker, tape, bundle = _setup(monkeypatch, state)
        tape.window()
        cycle(trader, state, tracker, bundle)
        tape.append(99.5, edge=ENTER)
        assert cycle(trader, state, tracker, bundle) == "bought"
        assert trader.config.gm_stop_k == 0.5
        assert trader.entry["stop"] == pytest.approx(99.5 * math.exp(-0.5 * ADR))
        tape.append(99.8, high=trader.entry["take"] + 0.01)
        assert cycle(trader, state, tracker, bundle) == "sold"
        assert _exits(state) == [at.GoodMomentsTrader.EXIT_TAKE]

    def test_without_highlows_forecast_nothing_is_bought_after_the_window(self, monkeypatch, state):
        trader, tracker, tape, bundle = _setup(monkeypatch, state)
        tape.forecast = ValueError("no SIP history")
        tape.window()
        cycle(trader, state, tracker, bundle)
        errors = _texts(state, "error")
        assert len(errors) == 1 and errors[0].startswith("HighLow cannot forecast today's AMD range")
        tape.append(95.0, edge=0.2)
        assert cycle(trader, state, tracker, bundle) == "hold"
        assert any("HighLow has no forecast today (no SIP history)" in t for t in _texts(state, "status"))


class TestRestart:
    """A restart the same day votes with the forecast the earlier run made:
    HighLow asked again after 09:50 reads SIP where 9:35 read IEX."""

    def _restarted(self, monkeypatch, state, key=KEY, ticker=TICKER, breach=False):
        trader, tracker, tape, bundle = _setup(monkeypatch, state, key, ticker)
        tape.window()
        cycle(trader, state, tracker, bundle)
        if breach:
            tape.append(103.0, high=103.5)                              # through the 102 high
            cycle(trader, state, tracker, bundle)
        before = dict(trader.levels.plan)
        for _ in range(3):
            tape.append(101.0)
        tape.forecast = dict(REMADE)
        again = at.build_trader(hybrid_config(key, ticker), bundle)
        cycle(again, state, tracker, bundle)
        return again, before, tape

    @pytest.mark.parametrize("key, ticker", [
        (KEY, TICKER), (apple_models.HIGHLOW2_GOODMOMENTS_KEY, "AAPL"),
    ])
    def test_it_votes_with_the_earlier_runs_forecast(self, monkeypatch, state, key, ticker):
        again, before, tape = self._restarted(monkeypatch, state, key, ticker)
        assert tape.forecasts_made == 1
        assert again.levels.plan["pred_high"] == pytest.approx(102.0)
        assert again.levels.plan["buy_level"] == pytest.approx(BUY_LEVEL)
        assert again.levels.plan["opening_end"] == before["opening_end"]
        assert any("(taken back from today's earlier run)" in t for t in _texts(state, "analysis"))

    def test_a_breach_since_is_walked_again_from_it(self, monkeypatch, state):
        again, before, tape = self._restarted(monkeypatch, state, breach=True)
        assert before["buy_level"] > BUY_LEVEL
        assert again.levels.plan["buy_level"] == pytest.approx(before["buy_level"])

    def test_the_seed_survives_the_session_file(self, monkeypatch, state):
        trader, tracker, tape, bundle = _setup(monkeypatch, state)
        tape.window()
        cycle(trader, state, tracker, bundle)
        memory = session_store.loads(session_store.dumps({"memory": trader.memory()}))["memory"]
        seed = memory["levels_seed"]
        assert seed["forecast"]["pred_high"] == 102.0 and seed["model_key"] == KEY
        assert seed["opening_end"] == tape.index[4]

    def test_another_hybrids_or_days_seed_is_not_taken_back(self, monkeypatch, state):
        trader, tracker, tape, bundle = _setup(monkeypatch, state)
        tape.window()
        frame, today = tape.frame(), pd.Timestamp("2026-07-21")
        seed = {"forecast": dict(FORECAST), "opening_end": tape.index[4], "open_price": 100.0,
                "model_key": KEY}
        trader._levels_seed = seed
        assert trader._taken_back_seed(frame, today) is seed
        trader._levels_seed = {**seed, "model_key": apple_models.HIGHLOW2_GOODMOMENTS_KEY}
        assert trader._taken_back_seed(frame, today) is None
        trader._levels_seed = {**seed, "opening_end": tape.index[4] - pd.Timedelta(days=1)}
        assert trader._taken_back_seed(frame, today) is None

    def test_a_goodmoments_runs_record_has_none_to_take_back(self, monkeypatch, state):
        """GoodMoments alone ran earlier today: the hybrid makes HighLow's
        forecast itself."""
        trader, tracker, tape, bundle = _setup(monkeypatch, state)
        tape.window()
        plain = at.build_trader(AppleTraderConfig(model_key="goodmoments", ticker=TICKER, skip_events=()))
        cycle(plain, state, tracker, bundle)
        assert "levels_seed" not in state.apple_trader_levels["memory"]
        cycle(trader, state, tracker, bundle)
        assert tape.forecasts_made == 1
        assert not any("taken back" in t for t in _texts(state, "analysis"))


class TestWiring:
    def test_it_is_offered_wherever_both_models_are(self):
        assert apple_models.get(KEY).tickers == ("AAPL", "AMD", "NVDA", "MU")
        for ticker in ("AAPL", "AMD", "NVDA", "MU"):
            assert KEY in apple_models.keys_for(ticker)
        for ticker in ("INTC", "AMGN", "TMO", "GOOG"):                 # HighLow but no GoodMoments
            assert KEY not in apple_models.keys_for(ticker)
        assert apple_models.strategy(KEY) == apple_models.STRATEGY_PROBABILITY
        assert apple_models.signal_of(KEY) == apple_models.GOODMOMENTS_KEY
        assert apple_models.levels_of(KEY) == apple_models.HIGHLOW_KEY

    def test_each_hybrid_has_its_own_trader(self):
        amd = at.build_trader(hybrid_config())
        aapl = at.build_trader(hybrid_config(apple_models.HIGHLOW2_GOODMOMENTS_KEY, "AAPL"))
        assert type(amd) is at.HighLowGoodMomentsTrader and amd.levels_name == "HighLow"
        assert type(aapl) is at.HighLow2GoodMomentsTrader and aapl.levels_name == "HighLow2"
        assert at.HighLowGoodMomentsTrader().config.model_key == KEY

    def test_its_bundle_is_goodmoments_with_highlows_beside_it(self, monkeypatch):
        monkeypatch.setattr(apple_models, "_load_goodmoments", lambda t: {"kind": "goodmoments"})
        monkeypatch.setattr(apple_models, "_load_highlow", lambda t: {"kind": "highlow"})
        assert apple_models.load(KEY, TICKER) == {
            "kind": "goodmoments", apple_models.HIGHLOW_KEY: {"kind": "highlow"},
        }
        monkeypatch.setattr(apple_models, "_load_highlow", lambda t: None)
        assert apple_models.load(KEY, TICKER) is None
        reason = apple_models.unavailable_reason(KEY, TICKER)
        assert "highlow15m_AMD.joblib" in reason and "goodmoments" not in reason

    def test_a_replay_is_stale_when_either_models_files_change(self, monkeypatch, tmp_path):
        for name in ("goodmoments_AMD.joblib", "goodmoments_AMD_timevol.joblib",
                     "highlow15m_AMD.joblib", "highlow15m_AMD_nbeats.pt"):
            (tmp_path / name).write_text("")
        for key, stem in ((apple_models.GOODMOMENTS_KEY, "goodmoments_AMD"),
                          (apple_models.HIGHLOW_KEY, "highlow15m_AMD")):
            monkeypatch.setitem(apple_models.MODELS, key, replace(
                apple_models.MODELS[key], path=lambda t, stem=stem: tmp_path / f"{stem}.joblib",
            ))
        model = apple_models.MODELS[KEY]
        monkeypatch.setitem(apple_models.MODELS, KEY, replace(
            model, path=apple_models.MODELS[apple_models.GOODMOMENTS_KEY].path,
        ))
        assert [p.name for p in apple_models.bundle_files(KEY, TICKER)] == [
            "goodmoments_AMD.joblib", "goodmoments_AMD_timevol.joblib",
            "highlow15m_AMD.joblib", "highlow15m_AMD_nbeats.pt",
        ]

    @pytest.mark.parametrize("ticker", ["AAPL", "AMD", "NVDA", "MU"])
    def test_each_ticker_opens_on_its_two_models_shipped_rules(self, ticker):
        """No base anywhere: GoodMoments' rules as its notebook picked them on
        the ticker (NVDA's +0.10 entry, MU's -0.02 exit, AMD's 0.5 stop), and
        HighLow's buy distance on it."""
        config = at.default_config(ticker, KEY)
        goodmoments = at.default_config(ticker, apple_models.GOODMOMENTS_KEY)
        highlow = at.default_config(ticker, apple_models.HIGHLOW_KEY)
        assert (config.model_key, config.ticker) == (KEY, ticker)
        assert (config.gm_enter, config.gm_exit, config.gm_take_k, config.gm_stop_k) == (
            goodmoments.gm_enter, goodmoments.gm_exit, goodmoments.gm_take_k, goodmoments.gm_stop_k,
        )
        assert (config.buy_k, config.sell_k, config.breach_update) == (
            highlow.buy_k, 0.0, highlow.breach_update,
        )
        assert type(at.build_trader(config)) is at.HighLowGoodMomentsTrader

    def test_it_starts_from_goodmoments_rules_and_highlows_buy(self):
        base_configs.set_base({
            **asdict(AppleTraderConfig(model_key="goodmoments", ticker=TICKER)),
            "gm_enter": 0.03, "gm_take_k": 0.1, "gm_stop_k": 0.8, "skip_events": [],
        })
        base_configs.set_base({
            **asdict(AppleTraderConfig(model_key="highlow", ticker=TICKER)),
            "buy_k": 0.45, "sell_k": 0.3, "breach_update": at.BREACH_BROWNIAN, "keep_width": True,
        })
        config = at.default_config(TICKER, KEY)
        assert (config.model_key, config.gm_enter, config.gm_take_k, config.gm_stop_k) == (KEY, 0.03, 0.1, 0.8)
        assert (config.buy_k, config.sell_k) == (0.45, 0.0)
        assert (config.breach_update, config.keep_width) == (at.BREACH_BROWNIAN, True)

    def test_its_signature_names_it_and_adds_the_buy_level(self):
        config = hybrid_config(breach_update=at.BREACH_BROWNIAN, keep_width=True)
        assert config_signature(config) == (
            "highlow_goodmoments_AMD(enter=+0.02,exit=off,take=0.2A,stop=0.5A,raise=0.05/stop,"
            "buy<=H-0.6R,breach=brownian,keep_width,size=95%)"
        )

    def test_tuning_offers_goodmoments_axes_and_the_buy_level(self):
        from simlab import tuning

        offered = set(tuning.axes_for(KEY, tuning.SWEEPABLE))
        assert offered == set(tuning.PROBABILITY_AXES) | {"buy_k", "breach_update"}
        assert tuning.default_axes(KEY) == ("gm_enter", "buy_k")

    def test_orchestra_races_it_under_its_short_name(self):
        assert sorted(p for p in orchestra.all_pairs() if p.endswith(f":{KEY}")) == [
            f"{t}:{KEY}" for t in ("AAPL", "AMD", "MU", "NVDA")
        ]
        assert orchestra.racer_label(f"{TICKER}:{KEY}") == "AMD · HighLow + GoodMoments"
        assert orchestra.pair_fields(KEY) == orchestra.HYBRID_PAIR_FIELDS
        assert "under HighLow's buy 0.6" in orchestra.racer_terms(hybrid_config())

    def test_its_runs_draw_highlows_range_and_the_edge(self):
        drawn = mo.for_models([KEY], TICKER)["keys"]
        assert mo.GOODMOMENTS_SIGNAL_KEY in drawn and mo.HIGHLOW_RANGE_KEY in drawn
        assert mo.HIGHLOW2_RANGE_KEY not in drawn

    def test_a_runs_record_draws_highlows_buy_level(self):
        stamps = [Tape.OPEN + pd.Timedelta(minutes=4 + i) for i in range(2)]
        rows = [{"t": stamps[0], "buy": None, "take": None, "stop": None},
                {"t": stamps[1], "buy": 99.6, "take": 100.2, "stop": 98.1}]
        items, note = mo._signal_trader_items(TICKER, (rows, hybrid_config()), KEY)
        assert note == "" and [i["label"] for i in items] == ["HighLow buy level", "Take-profit", "Stop level"]

    def test_the_catalogue_names_it_on_both_cards_where_it_exists(self):
        highlow = model_catalogue._hybrid_consumers(apple_models.HIGHLOW_KEY, TICKER)
        goodmoments = model_catalogue._hybrid_consumers(apple_models.GOODMOMENTS_KEY, TICKER)
        assert highlow == ("Apple Trader — HighLow + GoodMoments (its buy level votes on "
                           "GoodMoments' buys after 9:35)",)
        assert goodmoments == ("Apple Trader — HighLow + GoodMoments (GoodMoments' rules, "
                               "HighLow voting on buys after 9:35)",)
        assert model_catalogue._hybrid_consumers(apple_models.HIGHLOW_KEY, "INTC") == ()
        assert model_catalogue._hybrid_consumers(apple_models.HIGHLOW2_KEY, "GOOG") == ()
        # AAPL has both hybrids, so its GoodMoments card names both.
        assert [line.split(" (")[0] for line in
                model_catalogue._hybrid_consumers(apple_models.GOODMOMENTS_KEY, "AAPL")] == [
            "Apple Trader — HighLow2 + GoodMoments", "Apple Trader — HighLow + GoodMoments",
        ]

    def test_both_apps_name_the_voting_model_in_the_section_help(self):
        from agent_stonks import ui
        from simlab import app as simlab_app

        for text in (ui._APPLE_TRADER_COPY.sections["hybrid_levels"],
                     simlab_app._APPLE_TRADER_COPY_FIELDS["sections"]["hybrid_levels"]):
            said = text.format(model="HighLow")
            assert "HighLow's" in said and "HighLow2" not in said
        assert "model_highlow_goodmoments" in ui._APPLE_TRADER_COPY.help
        assert "model_highlow_goodmoments" in simlab_app._APPLE_TRADER_COPY_FIELDS["help"]


class TestForm:
    @staticmethod
    def _form() -> None:
        import streamlit as st

        from agent_stonks import apple_trader_ui

        copy = apple_trader_ui.FormCopy(
            prefix="t", unavailable_suffix="", instrument_help="", model_help="",
            sections={"hybrid_levels": "The vote of {model}."},
        )
        st.session_state["out"] = apple_trader_ui.params(["AMD", "AAPL"], copy)

    @pytest.fixture
    def app(self, monkeypatch):
        from streamlit.testing.v1 import AppTest

        monkeypatch.setattr(apple_models, "load", lambda *a, **k: {"model": "stub"})
        app = AppTest.from_function(self._form, default_timeout=30)
        app.run()
        return app

    def test_the_form_opens_on_highlows_buy_level(self, app):
        app.selectbox(key="t_ticker").set_value(TICKER).run()
        app.selectbox(key="t_model_AMD").select(KEY).run()
        assert not app.exception, app.exception
        out = app.session_state["out"]
        assert out.model_key == KEY and out.sell_k == 0.0
        assert out.buy_k == at.default_config(TICKER, KEY).buy_k
        headings = [m.value for m in app.markdown]
        assert "**HighLow buy level**" in headings

    def test_aapl_offers_both_hybrids_each_with_its_own_vote(self, app):
        app.selectbox(key="t_ticker").set_value("AAPL").run()
        offered = set(app.selectbox(key="t_model_AAPL").options)
        assert {"HighLow + GoodMoments", "HighLow2 + GoodMoments"} <= {o.split(" (")[0] for o in offered}
        for key, heading in ((KEY, "**HighLow buy level**"),
                             (apple_models.HIGHLOW2_GOODMOMENTS_KEY, "**HighLow2 buy level**")):
            app.selectbox(key="t_model_AAPL").select(key).run()
            assert not app.exception, app.exception
            assert app.session_state["out"].model_key == key
            assert heading in [m.value for m in app.markdown]
