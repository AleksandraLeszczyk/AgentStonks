"""Orchestra: Apple Trader's rules on several (ticker, model) pairs over one ledger.

Each racer is a real `DayRangeTrader` on a stubbed forecast, so these pin what
the race adds -- who may buy, who holds, when the race reopens -- and that it
changes nothing about any one racer's own rules: a race of one trades exactly
as a single run does.
"""
from dataclasses import field, fields, make_dataclass, replace
from datetime import datetime, timezone

import pandas as pd
import pytest

from agent_stonks import orchestra as ar
from agent_stonks import apple_trader as at
from agent_stonks import clock, model_overlays, session_store
from agent_stonks.broker import Broker
from agent_stonks.decisions import DecisionTracker
from agent_stonks.state import AppState
from tests.test_apple_trader import dayrange_config

MIDSESSION = datetime(2026, 7, 21, 14, 30, tzinfo=timezone.utc)
OPEN = pd.Timestamp("2026-07-21 09:30", tz="America/New_York")
BUNDLE = {"opening_minutes": 5}

# One forecast per symbol, each a $10 ADR under a $110 high, so at the
# notebook's 0.75 / 0.10 every racer buys at 102.50 and sells at 109.00.
FORECAST = {
    "pred_high": 110.0, "pred_low": 95.0, "prev_avg": 102.0,
    "adr14_abs": 10.0, "or_high": 103.0, "or_low": 101.0,
}
BUY_LEVEL, SELL_LEVEL = 102.5, 109.0


class Broker2(Broker):
    """Fills every order at the symbol's own last price."""

    def __init__(self):
        self.prices: dict[str, float] = {}

    def get_current_price(self, symbol, key, secret, feed="iex") -> float:
        return self.prices[symbol]

    def submit_order(self, symbol, side, quantity, price) -> dict:
        return {"status": "filled", "filled_qty": quantity, "filled_price": price}


class Tapes:
    """Today's minute bars per symbol, each opening with the 09:30 window the
    forecast is built on; `bar` appends the next tradable minute to all of
    them at once, as the clock would."""

    def __init__(self, monkeypatch, symbols, broker=None, open_=OPEN, opening=5):
        self.open = open_
        self.broker = broker
        self.rows = {s: [] for s in symbols}
        self.index: list[pd.Timestamp] = []
        self.forecasts: list[str] = []
        for _ in range(opening):
            self.opening_bar()
        monkeypatch.setattr(at.momentum_regime, "minute_frame", lambda ss: self.frame(ss.symbol))
        monkeypatch.setattr(at.momentum_regime, "compute_momentum", lambda frame, *a, **k: frame)
        monkeypatch.setattr(at, "session_forecast", self._forecast)
        monkeypatch.setattr(at.historical, "fetch_intraday_bars", lambda *a, **k: [])

    def _forecast(self, bundle, ticker, opening, today, key=None, secret=None):
        self.forecasts.append(ticker)
        return dict(FORECAST), None

    def opening_bar(self):
        """The next minute of the 09:30 opening window, on every symbol."""
        i = len(self.index)
        self._stamp(i)
        for s in self.rows:
            self._row(s, 101.0 + i * 0.1, low=100.9, high=101.5)

    def _stamp(self, offset):
        self.index.append(self.open + pd.Timedelta(minutes=offset))

    def _row(self, symbol, close, low=None, high=None):
        self.rows[symbol].append({
            "open": close, "high": close if high is None else high,
            "low": close if low is None else low, "close": close,
            "volume": 1.0e5, "mom": float("nan"),
        })
        if self.broker is not None:
            self.broker.prices[symbol] = close

    def bar(self, **closes):
        """The next minute. `closes[symbol]` is `close` or `(close, low)` or
        `(close, low, high)`; a symbol not named prints a flat 104."""
        self._stamp(60 + len(self.index))
        for symbol in self.rows:
            spec = closes.get(symbol, 104.0)
            spec = spec if isinstance(spec, tuple) else (spec,)
            self._row(symbol, *spec)

    def frame(self, symbol) -> pd.DataFrame:
        rows = self.rows[symbol]
        return at.momentum_regime.add_session_columns(
            pd.DataFrame(rows, index=pd.DatetimeIndex(self.index[: len(rows)]))
        )


@pytest.fixture
def market_open():
    clock.set_simulated(MIDSESSION)
    yield
    clock.clear()


def make_state(symbols) -> AppState:
    state = AppState()
    state.set_symbols(symbols)
    state.api_key, state.api_secret, state.feed = "k", "s", "iex"
    # The bar buffer is taken as the opening window rather than re-fetched.
    state.bar_tape_override = "yfinance"
    return state


def racer(ticker, model_key="dayrange", **kwargs):
    return replace(dayrange_config(**kwargs), ticker=ticker, model_key=model_key)


def make_race(*configs, selection=None, sources=None) -> ar.Orchestra:
    race = ar.OrchestraConfig(list(configs), selection=selection)
    return ar.build_orchestra(race, {k: BUNDLE for k in race.keys}, sources=sources)


def fills(tracker):
    return [(d.symbol, d.action, d.filled_quantity) for d in tracker.decisions if d.status == "filled"]


class TestDaysOff:
    def test_a_pair_sitting_the_day_out_leaves_the_race_to_the_rest(self, market_open, monkeypatch):
        """Each racer checks its own symbol: a geo shock flagged in AAPL's
        briefing benches AAPL's pair, and INTC's still races."""
        broker = Broker2()
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tapes = Tapes(monkeypatch, ["AAPL", "INTC"], broker)
        state = make_state(["AAPL", "INTC"])
        made = pd.Timestamp("2026-07-21 08:30", tz="America/New_York").to_pydatetime()
        at.event_days.record_verdict("AAPL", "geo", "strikes", made)
        at.event_days.record_verdict("INTC", "none", "", made)
        days = at.event_days.CATEGORIES
        race = make_race(racer("AAPL", skip_events=days), racer("INTC", skip_events=days))

        race.run_cycle(state, tracker)
        tapes.bar(AAPL=(103.0, BUY_LEVEL - 0.01), INTC=(103.0, BUY_LEVEL - 0.01))
        race.run_cycle(state, tracker)

        board = {row["key"]: row["status"] for row in state.orchestra["board"]}
        assert board["AAPL:dayrange"] == "sits out today (Geopolitical shock)"
        assert [f[:2] for f in fills(tracker)] == [("INTC", "buy")]


class TestFirstFillTakesTheRace:
    def test_the_first_racer_to_fill_holds_and_the_rest_wait(self, market_open, monkeypatch):
        broker = Broker2()
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tapes = Tapes(monkeypatch, ["AAPL", "INTC"], broker)
        state = make_state(["AAPL", "INTC"])
        race = make_race(racer("AAPL"), racer("INTC"))

        race.run_cycle(state, tracker)  # the forecasts
        # Both dip through their buy level on the same bar: the earlier racer
        # in the race's order is read first and takes it.
        tapes.bar(AAPL=(103.0, BUY_LEVEL - 0.01), INTC=(103.0, BUY_LEVEL - 0.01))
        assert race.run_cycle(state, tracker) == "bought"

        assert race.holder == "AAPL:dayrange"
        assert [f[:2] for f in fills(tracker)] == [("AAPL", "buy")]
        assert tracker.position_for("INTC") == 0
        refusals = [e["text"] for e in state.agent_log if "Orchestra is following" in e.get("text", "")]
        assert len(refusals) == 1 and refusals[0].startswith("INTC reached its buy level")
        board = {row["key"]: row["status"] for row in state.orchestra["board"]}
        assert board == {"AAPL:dayrange": "holding", "INTC:dayrange": "waiting for the holder"}

    def test_a_gated_racer_is_reported_once_per_holding(self, market_open, monkeypatch):
        broker = Broker2()
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tapes = Tapes(monkeypatch, ["AAPL", "INTC"], broker)
        state = make_state(["AAPL", "INTC"])
        race = make_race(racer("AAPL"), racer("INTC"))
        race.run_cycle(state, tracker)
        tapes.bar(AAPL=(103.0, BUY_LEVEL - 0.01))
        race.run_cycle(state, tracker)
        for _ in range(3):
            tapes.bar(AAPL=104.0, INTC=(103.0, BUY_LEVEL - 0.01))
            race.run_cycle(state, tracker)
        refusals = [e for e in state.agent_log if "Orchestra is following" in e.get("text", "")]
        assert len(refusals) == 1
        assert refusals[0]["racer"] == "INTC · Day Range"


class TestTheRaceReopens:
    def test_after_the_holder_sells_at_its_target_another_racer_can_buy(
        self, market_open, monkeypatch
    ):
        broker = Broker2()
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tapes = Tapes(monkeypatch, ["AAPL", "INTC"], broker)
        state = make_state(["AAPL", "INTC"])
        race = make_race(racer("AAPL"), racer("INTC"))
        race.run_cycle(state, tracker)

        tapes.bar(AAPL=(103.0, BUY_LEVEL - 0.01))
        race.run_cycle(state, tracker)
        tapes.bar(AAPL=(109.5, 109.0, SELL_LEVEL + 0.5))
        assert race.run_cycle(state, tracker) == "sold"
        assert race.holder is None
        assert any("the race is open" in e.get("text", "") for e in state.agent_log)

        tapes.bar(INTC=(103.0, BUY_LEVEL - 0.01))
        assert race.run_cycle(state, tracker) == "bought"
        assert race.holder == "INTC:dayrange"
        assert [f[:2] for f in fills(tracker)] == [
            ("AAPL", "buy"), ("AAPL", "sell"), ("INTC", "buy"),
        ]

    def test_the_holder_is_read_first_so_the_race_reopens_on_its_exit_bar(
        self, market_open, monkeypatch
    ):
        """INTC is first in the order, but AAPL holds: AAPL's sale on this bar
        frees the cash before INTC's dip on the same bar is judged."""
        broker = Broker2()
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tapes = Tapes(monkeypatch, ["INTC", "AAPL"], broker)
        state = make_state(["INTC", "AAPL"])
        race = make_race(racer("INTC"), racer("AAPL"))
        race.run_cycle(state, tracker)
        tapes.bar(AAPL=(103.0, BUY_LEVEL - 0.01))
        race.run_cycle(state, tracker)
        assert race.holder == "AAPL:dayrange"

        tapes.bar(AAPL=(109.5, 109.0, SELL_LEVEL + 0.5), INTC=(103.0, BUY_LEVEL - 0.01))
        race.run_cycle(state, tracker)
        assert race.holder == "INTC:dayrange"
        assert [f[:2] for f in fills(tracker)][-2:] == [("AAPL", "sell"), ("INTC", "buy")]

    def test_a_breaker_stand_down_keeps_the_others_racing(self, market_open, monkeypatch):
        broker = Broker2()
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tapes = Tapes(monkeypatch, ["AAPL", "INTC"], broker)
        state = make_state(["AAPL", "INTC"])
        # A trade has to clear 1 ADR a share; the target pays 0.65, so any
        # closed AAPL trade stands AAPL down.
        race = make_race(racer("AAPL", min_win_k=1.0), racer("INTC", min_win_k=1.0))
        race.run_cycle(state, tracker)
        tapes.bar(AAPL=(103.0, BUY_LEVEL - 0.01))
        race.run_cycle(state, tracker)
        tapes.bar(AAPL=(109.5, 109.0, SELL_LEVEL + 0.5))
        race.run_cycle(state, tracker)

        tapes.bar(AAPL=(103.0, BUY_LEVEL - 0.01), INTC=(103.0, BUY_LEVEL - 0.01))
        assert race.run_cycle(state, tracker) == "bought"
        assert race.holder == "INTC:dayrange"
        statuses = {row["key"]: row["status"] for row in state.orchestra["board"]}
        assert statuses["AAPL:dayrange"].startswith("stood down")


class TestOneSymbolTwoModels:
    def test_a_racer_on_the_holders_symbol_does_not_take_its_position(
        self, market_open, monkeypatch
    ):
        """AAPL on two models: the second sees AAPL as flat while the first
        holds it, so it neither adopts the position nor manages it."""
        broker = Broker2()
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tapes = Tapes(monkeypatch, ["AAPL"], broker)
        state = make_state(["AAPL"])
        first = racer("AAPL", "dayrange")
        # A shallower buy and an earlier sell under the second model.
        second = racer("AAPL", "highlow", buy_k=0.40, sell_k=0.30)
        race = make_race(first, second)
        race.run_cycle(state, tracker)

        tapes.bar(AAPL=(103.0, BUY_LEVEL - 0.01))
        race.run_cycle(state, tracker)
        assert race.holder == "AAPL:dayrange"
        assert race.by_key["AAPL:highlow"].trader.entry is None

        # Through the second racer's sell level (107.00) but not the holder's:
        # nothing is sold, because the second racer holds nothing.
        tapes.bar(AAPL=(107.5, 107.0, 107.5))
        race.run_cycle(state, tracker)
        assert [f[:2] for f in fills(tracker)] == [("AAPL", "buy")]
        assert race.by_key["AAPL:highlow"].trader.entry is None
        # Each racer keeps its own record, under its own key.
        assert set(state.orchestra_levels) == {"AAPL:dayrange", "AAPL:highlow"}
        assert state.apple_trader_levels is None


class TestStopOut:
    def test_a_stop_out_halts_the_race_for_the_day(self, market_open, monkeypatch):
        broker = Broker2()
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tapes = Tapes(monkeypatch, ["AAPL", "INTC"], broker)
        state = make_state(["AAPL", "INTC"])
        race = make_race(racer("AAPL", stop_gain_fraction=0.5), racer("INTC", stop_gain_fraction=0.5))
        race.run_cycle(state, tracker)
        tapes.bar(AAPL=(103.0, BUY_LEVEL - 0.01))
        race.run_cycle(state, tracker)
        tapes.bar(AAPL=(99.0, 98.0, 99.5))
        assert race.run_cycle(state, tracker) == "sold"
        assert race.halt and race.halt.startswith("AAPL · Day Range")

        # A replay keeps going bar by bar: the race stays down all session.
        tapes.bar(INTC=(103.0, BUY_LEVEL - 0.01))
        assert race.run_cycle(state, tracker) == "hold"
        assert tracker.position_for("INTC") == 0

        # The next session is a new race.
        clock.set_simulated(datetime(2026, 7, 22, 14, 30, tzinfo=timezone.utc))
        tapes = Tapes(
            monkeypatch, ["AAPL", "INTC"], broker,
            open_=pd.Timestamp("2026-07-22 09:30", tz="America/New_York"),
        )
        race.run_cycle(state, tracker)
        tapes.bar(INTC=(103.0, BUY_LEVEL - 0.01))
        assert race.run_cycle(state, tracker) == "bought"
        assert race.halt is None and race.holder == "INTC:dayrange"


class TestARaceOfOneIsASingleRun:
    """The race adds nothing to one racer: same decisions, same log."""

    PATH = [
        dict(AAPL=104.0),
        dict(AAPL=(103.0, BUY_LEVEL - 0.01)),
        dict(AAPL=104.0),
        dict(AAPL=(109.5, 109.0, SELL_LEVEL + 0.5)),
        dict(AAPL=(103.0, BUY_LEVEL - 0.01)),
        dict(AAPL=(101.0, 99.0, 103.0)),
        dict(AAPL=(97.0, 96.0, 99.0)),
        dict(AAPL=(104.0, 103.0, 104.0)),
    ]

    @pytest.mark.parametrize("kwargs", [
        {},
        {"stop_gain_fraction": 0.5, "min_win_k": 0.3},
        {"scale_in": True, "position_pct": 50.0, "buy_step_k": 0.1},
    ])
    def test_same_decisions_and_log(self, market_open, monkeypatch, kwargs):
        def run(as_race: bool):
            broker = Broker2()
            tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
            tapes = Tapes(monkeypatch, ["AAPL"], broker)
            state = make_state(["AAPL"])
            config = racer("AAPL", **kwargs)
            if as_race:
                race = make_race(config)
                cycle = lambda: race.run_cycle(state, tracker)  # noqa: E731
            else:
                trader = at.DayRangeTrader(config)
                cycle = lambda: trader.run_cycle(BUNDLE, state, tracker)  # noqa: E731
            outcomes = [cycle()]
            for bar in self.PATH:
                tapes.bar(**bar)
                outcomes.append(cycle())
            decisions = [
                (d.symbol, d.action, d.status, d.filled_quantity, d.price, d.reasoning)
                for d in tracker.decisions
            ]
            log = [(e["type"], e.get("text")) for e in state.agent_log]
            return outcomes, decisions, log

        assert run(as_race=True) == run(as_race=False)


class TestOrchestraConfig:
    def test_refuses_a_pair_twice(self):
        with pytest.raises(ValueError, match="more than once"):
            ar.OrchestraConfig([racer("AAPL"), racer("AAPL")])

    def test_refuses_racers_whose_shared_rules_differ(self):
        with pytest.raises(ValueError, match="position_pct"):
            ar.OrchestraConfig([racer("AAPL"), racer("INTC", position_pct=50.0)])

    def test_takes_racers_of_the_class_before_a_module_reload(self):
        # Saving a file makes Streamlit unload the app's modules, so a race
        # started before the save keeps publishing configs of the old class --
        # here one that has not gained `take_trigger` yet and still has a field since
        # removed. Rebuilding the running race from them raised TypeError.
        aapl, intc = racer("AAPL"), racer("INTC")
        old_class = make_dataclass(
            "AppleTraderConfig",
            [(f.name, object, field(default=None))
             for f in fields(at.AppleTraderConfig) if f.name != "take_trigger"]
            + [("removed_since", object, field(default=None))],
        )
        stale = [
            old_class(**{f.name: getattr(c, f.name) for f in fields(old_class) if hasattr(c, f.name)})
            for c in (aapl, intc)
        ]
        assert not isinstance(stale[0], at.AppleTraderConfig)
        race = ar.OrchestraConfig(stale)
        assert race.racers == [aapl, intc]
        assert ar.racer_label(stale[1]) == ar.racer_label("INTC:dayrange")

    def test_build_takes_each_pairs_own_tuned_numbers(self):
        race = ar.build_orchestra_config(["INTC:dayrange", "AAPL:highlow"], at.AppleTraderConfig())
        intc, aapl = race.racers
        assert (intc.buy_k, intc.sell_k) == at.dayrange_levels("INTC", "dayrange")
        assert (aapl.buy_k, aapl.sell_k) == at.dayrange_levels("AAPL", "highlow")
        assert race.tickers == ["INTC", "AAPL"]

    def test_build_takes_edited_numbers_where_given(self):
        race = ar.build_orchestra_config(
            ["MU:highlow"], at.AppleTraderConfig(), {"MU:highlow": (0.9, 0.2, 0.05)}
        )
        mu = race.racers[0]
        assert (mu.buy_k, mu.sell_k, mu.min_win_k) == (0.9, 0.2, 0.05)

    def test_signature_names_every_pair_and_the_shared_rules_once(self):
        race = ar.build_orchestra_config(["AAPL:dayrange", "MU:highlow"], at.AppleTraderConfig())
        sig = ar.orchestra_signature(race)
        assert sig.startswith("orchestra[dayrange_AAPL@")
        assert ",highlow_MU@" in sig
        assert sig.count("size=") == 1
        assert "min_win=" not in sig

    def test_default_pairs_are_the_tuned_ones(self):
        assert set(ar.default_pairs()) == {
            f"{ticker}:{model}" for model, ticker in at.APPLE_TRADER_TUNED_LEVELS
        }


class TestTheChartAndTheSessionFile:
    def _raced(self, monkeypatch):
        broker = Broker2()
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tapes = Tapes(monkeypatch, ["AAPL", "INTC"], broker)
        state = make_state(["AAPL", "INTC"])
        race = make_race(racer("AAPL", "dayrange"), racer("AAPL", "highlow"), racer("INTC"))
        race.run_cycle(state, tracker)
        tapes.bar(INTC=(103.0, BUY_LEVEL - 0.01))
        race.run_cycle(state, tracker)
        race.publish_memory(state)
        return state, tracker

    def test_a_symbols_chart_draws_its_first_racer_or_the_holder(self, market_open, monkeypatch):
        state, _ = self._raced(monkeypatch)
        config, record = model_overlays.live_trader_view(state, "AAPL")
        assert config.model_key == "dayrange" and record["ticker"] == "AAPL"
        config, record = model_overlays.live_trader_view(state, "INTC")
        assert config.ticker == "INTC" and record["memory"]["entry"] is not None
        assert model_overlays.live_trader_view(state, "MU") == (None, None)

    def test_the_session_file_keeps_every_racer_and_the_holder(
        self, market_open, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(session_store, "SESSION_DIR", tmp_path)
        state, _ = self._raced(monkeypatch)
        record = session_store.loads(session_store.dumps(session_store.capture(state)))
        fresh = AppState()
        session_store.restore(fresh, record)
        assert set(fresh.orchestra_levels) == {"AAPL:dayrange", "AAPL:highlow", "INTC:dayrange"}
        assert fresh.orchestra["holder"] == "INTC:dayrange"
        assert fresh.orchestra["running"] is False
        assert fresh.orchestra_levels["INTC:dayrange"]["memory"]["entry"] is not None

    def test_a_restarted_race_follows_the_racer_the_ledger_still_holds(
        self, market_open, monkeypatch
    ):
        state, tracker = self._raced(monkeypatch)
        race = make_race(racer("AAPL", "dayrange"), racer("AAPL", "highlow"), racer("INTC"))
        race.run_cycle(state, tracker)
        assert race.holder == "INTC:dayrange"
        # Its own memory came back with it: the fill, not a re-adoption.
        assert race.by_key["INTC:dayrange"].trader.entry["price"] == pytest.approx(103.0)


class TestTheLiveLoop:
    def _stub(self):
        class StoppedOut:
            blocked = None
            plan = None
            last_close = None
            last_bar_ts = None
            halt = None

            def __init__(self, config, bundle=None):
                self.ticker = config.ticker

            def run_cycle(self, bundle, state, tracker):
                self.halt = "stopped out"
                return "sold"

            def activity(self, outcome, tracker):
                return "🟢", "Agent holding"

            def publish_memory(self, state):
                pass

        return StoppedOut

    def test_a_stop_out_stops_the_agent(self, market_open, monkeypatch):
        import threading

        monkeypatch.setattr(ar.apple_models, "load", lambda key, ticker=None: BUNDLE)
        monkeypatch.setattr(at, "config_error", lambda config, bundle=None: None)
        monkeypatch.setattr(at, "build_trader", self._stub())
        state = make_state(["AAPL", "INTC"])
        state.agent_running = True
        stop_event = threading.Event()
        state.agent_stop_event = stop_event
        race = ar.OrchestraConfig([racer("AAPL"), racer("INTC")])
        tracker = DecisionTracker(starting_cash=10_000.0, broker=Broker2())
        loop = threading.Thread(
            target=ar._orchestra_loop, args=(state, tracker, race, 60, stop_event), daemon=True,
        )
        loop.start()
        loop.join(timeout=10)

        assert not loop.is_alive() and stop_event.is_set()
        assert state.agent_running is False
        assert state.orchestra["running"] is False
        assert any("Press ▶ Start Agent" in e.get("text", "") for e in state.agent_log)

    def test_a_racer_whose_model_cannot_load_is_left_out(self, market_open, monkeypatch):
        import threading

        monkeypatch.setattr(
            ar.apple_models, "load",
            lambda key, ticker=None: None if ticker == "INTC" else BUNDLE,
        )
        monkeypatch.setattr(at, "config_error", lambda config, bundle=None: None)
        monkeypatch.setattr(at, "build_trader", self._stub())
        state = make_state(["AAPL", "INTC"])
        stop_event = threading.Event()
        state.agent_stop_event = stop_event
        race = ar.OrchestraConfig([racer("AAPL"), racer("INTC")])
        tracker = DecisionTracker(starting_cash=10_000.0, broker=Broker2())
        loop = threading.Thread(
            target=ar._orchestra_loop, args=(state, tracker, race, 60, stop_event), daemon=True,
        )
        loop.start()
        loop.join(timeout=10)

        errors = [e["text"] for e in state.agent_log if e["type"] == "error"]
        assert any(t.startswith("INTC · Day Range is left out of Orchestra") for t in errors)
        assert state.orchestra["order"] == ["AAPL:dayrange"]



class _Sources:
    """Briefings and earnings for the 09:34 selection, per ticker."""

    def __init__(self, briefings=None, earnings=None):
        self.briefings = briefings or {}
        self.earnings_by = earnings or {}

    def briefing(self, ticker, day, state):
        b = self.briefings.get(ticker)
        return (b, "") if b else (None, "no briefing")

    def earnings(self, ticker, day):
        return self.earnings_by.get(ticker, [])


def _daily(monkeypatch, adr_by_ticker):
    """Fourteen completed sessions per symbol at a $100 close and the given
    daily range -- what the selection's ADR and gap are read from."""
    from agent_stonks import candidates as cd

    def bars(ticker, *a, **k):
        adr = adr_by_ticker.get(ticker, 1.0)
        return [
            {"t": f"2026-07-{d:02d}", "o": 100.0, "h": 100.0 + adr / 2, "l": 100.0 - adr / 2, "c": 100.0}
            for d in range(1, 15)
        ]

    monkeypatch.setattr(cd.historical, "fetch_daily_ohlc_bars", bars)
    monkeypatch.setattr(cd.historical, "fetch_session_open", lambda *a, **k: None)


class TestTheMorningPick:
    """Orchestra's 09:34 candidate selection (`agent_stonks.candidates`)."""

    def test_picks_once_at_0934_and_the_rest_never_forecast(self, market_open, monkeypatch):
        from agent_stonks.candidates import SelectionRules

        broker = Broker2()
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tapes = Tapes(monkeypatch, ["AAPL", "INTC", "MU"], broker, opening=3)
        # The same levels on all three; INTC's wider day makes its target worth
        # the most of its price.
        _daily(monkeypatch, {"AAPL": 2.0, "INTC": 6.0, "MU": 4.0})
        state = make_state(["AAPL", "INTC", "MU"])
        race = make_race(
            racer("AAPL"), racer("INTC"), racer("MU"),
            selection=SelectionRules(max_pairs=1), sources=_Sources(),
        )

        # 09:33: three opening minutes in -- nothing is decided, nothing read.
        assert race.run_cycle(state, tracker) == "warming_up"
        assert state.orchestra["selection"] is None
        assert tapes.forecasts == []
        assert race.activity("warming_up", tracker)[1] == "Agent waiting to pick its candidates at 09:34"

        # 09:34: the fourth is in, and the pick is made.
        tapes.opening_bar()
        race.run_cycle(state, tracker)
        picked = state.orchestra["selection"]
        assert [r["key"] for r in picked["rows"] if r["selected"]] == ["INTC:dayrange"]
        assert set(race.benched) == {"AAPL:dayrange", "MU:dayrange"}
        line = next(e["text"] for e in state.agent_log if e["text"].startswith("Candidates at 09:34"))
        assert "INTC · Day Range" in line and "Left out: AAPL · Day Range" in line

        # 09:35: only the candidate forecasts, and only it may buy.
        tapes.opening_bar()
        race.run_cycle(state, tracker)
        assert tapes.forecasts == ["INTC"]
        tapes.bar(AAPL=(103.0, BUY_LEVEL - 0.01), INTC=104.0)
        assert race.run_cycle(state, tracker) == "hold"
        tapes.bar(INTC=(103.0, BUY_LEVEL - 0.01))
        assert race.run_cycle(state, tracker) == "bought"
        assert [f[:2] for f in fills(tracker)] == [("INTC", "buy")]
        statuses = {row["key"]: row["status"] for row in state.orchestra["board"]}
        assert statuses["AAPL:dayrange"].startswith("sits out today (ranked #")

    def test_a_confidently_bearish_morning_keeps_a_pair_out(self, market_open, monkeypatch):
        from agent_stonks.candidates import SelectionRules

        tracker = DecisionTracker(starting_cash=10_000.0, broker=Broker2())
        Tapes(monkeypatch, ["AAPL", "INTC"], Broker2(), opening=4)
        _daily(monkeypatch, {"AAPL": 6.0, "INTC": 2.0})
        state = make_state(["AAPL", "INTC"])
        race = make_race(
            racer("AAPL"), racer("INTC"),
            selection=SelectionRules(max_pairs=0),
            sources=_Sources(briefings={"AAPL": {"bias": "bearish", "confidence": "high"}}),
        )
        race.run_cycle(state, tracker)
        assert race.benched == {"AAPL:dayrange": "bearish briefing, high confidence"}

    def test_earnings_since_the_last_close_keep_a_pair_out(self, market_open, monkeypatch):
        from agent_stonks.candidates import SelectionRules

        tracker = DecisionTracker(starting_cash=10_000.0, broker=Broker2())
        Tapes(monkeypatch, ["AAPL", "INTC"], Broker2(), opening=4)
        _daily(monkeypatch, {})
        state = make_state(["AAPL", "INTC"])
        race = make_race(
            racer("AAPL"), racer("INTC"),
            selection=SelectionRules(max_pairs=0),
            sources=_Sources(earnings={"INTC": [pd.Timestamp("2026-07-20 16:05-04:00")]}),
        )
        race.run_cycle(state, tracker)
        assert list(race.benched) == ["INTC:dayrange"]
        assert race.benched["INTC:dayrange"].startswith("earnings 2026-07-20 16:05")

    def test_a_restart_keeps_the_mornings_pick(self, market_open, monkeypatch):
        from agent_stonks.candidates import SelectionRules

        tracker = DecisionTracker(starting_cash=10_000.0, broker=Broker2())
        Tapes(monkeypatch, ["AAPL", "INTC"], Broker2(), opening=4)
        _daily(monkeypatch, {"AAPL": 2.0, "INTC": 6.0})
        state = make_state(["AAPL", "INTC"])
        rules = SelectionRules(max_pairs=1)
        make_race(racer("AAPL"), racer("INTC"), selection=rules, sources=_Sources()).run_cycle(state, tracker)
        assert [r["key"] for r in state.orchestra["selection"]["rows"] if r["selected"]] == ["INTC:dayrange"]

        # Later the same morning the facts would say otherwise; the pick stands.
        _daily(monkeypatch, {"AAPL": 9.0, "INTC": 1.0})
        again = make_race(racer("AAPL"), racer("INTC"), selection=rules, sources=_Sources())
        again.run_cycle(state, tracker)
        assert list(again.benched) == ["AAPL:dayrange"]
        assert any(e["text"].startswith("Keeping today's candidates") for e in state.agent_log)

    def test_the_selection_is_in_the_signature_and_the_record(self):
        from agent_stonks.candidates import SelectionRules
        from simlab.rule_agents import rule_agent

        agent = rule_agent(ar.ORCHESTRA_KEY)
        race = ar.build_orchestra_config(
            ["AAPL:dayrange", "INTC:dayrange"], at.AppleTraderConfig(),
            selection=SelectionRules(max_pairs=1, briefing_provider="gemini", briefing_model="g"),
        )
        assert ",select=1,earn,bear=high,brief=gemini/g)" in ar.orchestra_signature(race)
        assert agent.from_record(agent.to_record(race)) == race
        # A record from before the selection existed raced every pair all day.
        legacy = agent.to_record(race)
        del legacy["selection"]
        assert agent.from_record(legacy).selection is None


# --- the split: every pair at once, each on its own share of the budget --------


def make_split(shares: dict, *configs) -> ar.Orchestra:
    race = ar.OrchestraConfig(list(configs), mode=ar.MODE_SPLIT, budget_pct=shares)
    return ar.build_orchestra(race, {k: BUNDLE for k in race.keys})


def flat_racer(ticker, model_key="dayrange", **kwargs):
    """A racer with no stop unless a test gives it one: a bar that dips to one
    pair's buy level must not stop out another pair already holding."""
    kwargs.setdefault("stop_gain_fraction", 0.0)
    return racer(ticker, model_key, **kwargs)


class TestSplit:
    def test_every_pair_buys_at_once_on_its_own_share(self, market_open, monkeypatch):
        broker = Broker2()
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tapes = Tapes(monkeypatch, ["AAPL", "INTC"], broker)
        state = make_state(["AAPL", "INTC"])
        race = make_split(
            {"AAPL:dayrange": 60.0, "INTC:dayrange": 40.0}, flat_racer("AAPL"), flat_racer("INTC"),
        )
        race.run_cycle(state, tracker)
        assert race.budget_base == 10_000.0

        tapes.bar(AAPL=(103.0, BUY_LEVEL - 0.01), INTC=(103.0, BUY_LEVEL - 0.01))
        assert race.run_cycle(state, tracker) == "bought"

        # 95% of each pair's own share, at the $103 close: 5,700 / 103 and 3,800 / 103.
        assert fills(tracker) == [("AAPL", "buy", 55.0), ("INTC", "buy", 36.0)]
        fee = tracker.trade_cost
        aapl, intc = race.accounts["AAPL:dayrange"], race.accounts["INTC:dayrange"]
        assert aapl.cash == pytest.approx(6_000.0 - 55 * 103.0 - fee)
        assert intc.cash == pytest.approx(4_000.0 - 36 * 103.0 - fee)
        assert (aapl.shares, intc.shares) == (55.0, 36.0)
        assert race.holder is None and state.orchestra["holders"] == ["AAPL:dayrange", "INTC:dayrange"]
        assert not any("Orchestra is following" in e.get("text", "") for e in state.agent_log)
        board = {row["key"]: row for row in state.orchestra["board"]}
        assert board["AAPL:dayrange"]["status"] == board["INTC:dayrange"]["status"] == "holding"
        assert board["INTC:dayrange"]["budget_pct"] == 40.0

    def test_what_no_pair_is_given_stays_in_cash(self, market_open, monkeypatch):
        broker = Broker2()
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tapes = Tapes(monkeypatch, ["AAPL"], broker)
        state = make_state(["AAPL"])
        race = make_split({"AAPL:dayrange": 30.0}, flat_racer("AAPL"))
        race.run_cycle(state, tracker)
        tapes.bar(AAPL=(103.0, BUY_LEVEL - 0.01))
        race.run_cycle(state, tracker)
        # 95% of $3,000.
        assert fills(tracker) == [("AAPL", "buy", 27.0)]

    def test_two_pairs_on_one_symbol_each_hold_and_sell_their_own(self, market_open, monkeypatch):
        broker = Broker2()
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tapes = Tapes(monkeypatch, ["AAPL"], broker)
        state = make_state(["AAPL"])
        # HighLow buys at 106.00 and sells at 107.00; the day range at 102.50 / 109.00.
        race = make_split(
            {"AAPL:dayrange": 50.0, "AAPL:highlow": 50.0},
            flat_racer("AAPL", "dayrange"),
            flat_racer("AAPL", "highlow", buy_k=0.40, sell_k=0.30),
        )
        race.run_cycle(state, tracker)

        tapes.bar(AAPL=(106.5, 105.9, 106.6))
        race.run_cycle(state, tracker)
        tapes.bar(AAPL=(103.0, BUY_LEVEL - 0.01, 103.5))
        race.run_cycle(state, tracker)
        assert fills(tracker) == [("AAPL", "buy", 44.0), ("AAPL", "buy", 46.0)]
        assert tracker.position_for("AAPL") == 90.0

        # Through HighLow's sell, short of the day range's: HighLow sells its
        # 44 and nothing more; the day range keeps its own 46.
        tapes.bar(AAPL=(107.5, 107.0, 107.5))
        assert race.run_cycle(state, tracker) == "sold"
        assert fills(tracker)[-1] == ("AAPL", "sell", 44.0)
        assert tracker.position_for("AAPL") == 46.0
        assert race.accounts["AAPL:highlow"].shares == 0.0
        assert race.by_key["AAPL:dayrange"].trader.entry["price"] == pytest.approx(103.0)
        assert state.orchestra["holders"] == ["AAPL:dayrange"]

    def test_a_stop_out_stands_only_that_pair_down(self, market_open, monkeypatch):
        broker = Broker2()
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tapes = Tapes(monkeypatch, ["AAPL", "INTC"], broker)
        state = make_state(["AAPL", "INTC"])
        race = make_split(
            {"AAPL:dayrange": 50.0, "INTC:dayrange": 50.0},
            racer("AAPL", stop_gain_fraction=0.5), racer("INTC", stop_gain_fraction=0.5),
        )
        race.run_cycle(state, tracker)
        tapes.bar(AAPL=(103.0, BUY_LEVEL - 0.01))
        race.run_cycle(state, tracker)
        tapes.bar(AAPL=(99.0, 98.0, 99.5))
        assert race.run_cycle(state, tracker) == "sold"
        # Nothing stops the agent: the others have positions to manage.
        assert race.halt is None
        assert any(
            e.get("text", "").startswith("AAPL · Day Range stopped out: it sits out the rest")
            for e in state.agent_log
        )

        tapes.bar(AAPL=(103.0, BUY_LEVEL - 0.01), INTC=(103.0, BUY_LEVEL - 0.01))
        assert race.run_cycle(state, tracker) == "bought"
        assert [f[:2] for f in fills(tracker)] == [("AAPL", "buy"), ("AAPL", "sell"), ("INTC", "buy")]
        statuses = {row["key"]: row["status"] for row in state.orchestra["board"]}
        assert statuses["AAPL:dayrange"] == "stood down (stopped out)"

        # Started again the same day (a crash, or ▶ Stop and ▶ Start): the
        # stopped pair stays out -- the run never stopped, so this is no say-so.
        race.publish_memory(state)
        again = make_split(
            {"AAPL:dayrange": 50.0, "INTC:dayrange": 50.0},
            racer("AAPL", stop_gain_fraction=0.5), racer("INTC", stop_gain_fraction=0.5),
        )
        again.run_cycle(state, tracker)
        tapes.bar(AAPL=(103.0, BUY_LEVEL - 0.01), INTC=104.0)
        again.run_cycle(state, tracker)
        assert [f[:2] for f in fills(tracker)][-1] == ("INTC", "buy")
        assert again.by_key["AAPL:dayrange"].trader.plan["stand_down"] == "stopped out"

    def test_a_restart_continues_the_days_budget_and_accounts(self, market_open, monkeypatch):
        broker = Broker2()
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tapes = Tapes(monkeypatch, ["AAPL", "INTC"], broker)
        state = make_state(["AAPL", "INTC"])
        shares = {"AAPL:dayrange": 50.0, "INTC:dayrange": 40.0}
        race = make_split(shares, flat_racer("AAPL"), flat_racer("INTC"))
        race.run_cycle(state, tracker)
        tapes.bar(AAPL=(103.0, BUY_LEVEL - 0.01))
        race.run_cycle(state, tracker)
        race.publish_memory(state)
        aapl = race.accounts["AAPL:dayrange"]

        # The ledger is now mostly AAPL; the budget is not re-read off it, and
        # INTC's share of it is moved up, out of the cash nobody was given.
        again = make_split({**shares, "INTC:dayrange": 50.0}, flat_racer("AAPL"), flat_racer("INTC"))
        again.run_cycle(state, tracker)
        assert again.budget_base == 10_000.0
        assert again.accounts["AAPL:dayrange"] == aapl
        assert again.accounts["INTC:dayrange"].cash == pytest.approx(5_000.0)
        assert again.by_key["AAPL:dayrange"].trader.entry["price"] == pytest.approx(103.0)

        tapes.bar(INTC=(103.0, BUY_LEVEL - 0.01))
        again.run_cycle(state, tracker)
        # 95% of $5,000 at $103.
        assert fills(tracker)[-1] == ("INTC", "buy", 46.0)

    def test_the_session_file_keeps_the_budget(self, market_open, monkeypatch, tmp_path):
        monkeypatch.setattr(session_store, "SESSION_DIR", tmp_path)
        broker = Broker2()
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tapes = Tapes(monkeypatch, ["AAPL", "INTC"], broker)
        state = make_state(["AAPL", "INTC"])
        race = make_split(
            {"AAPL:dayrange": 50.0, "INTC:dayrange": 25.0}, flat_racer("AAPL"), flat_racer("INTC"),
        )
        race.run_cycle(state, tracker)
        tapes.bar(INTC=(103.0, BUY_LEVEL - 0.01))
        race.run_cycle(state, tracker)

        record = session_store.loads(session_store.dumps(session_store.capture(state)))
        fresh = AppState()
        session_store.restore(fresh, record)
        assert fresh.orchestra["mode"] == "split"
        assert fresh.orchestra["holders"] == ["INTC:dayrange"]
        budget = fresh.orchestra["budget"]
        assert budget["base"] == 10_000.0
        # 95% of $2,500 at $103.
        assert budget["accounts"]["INTC:dayrange"]["shares"] == 23.0
        assert budget["accounts"]["AAPL:dayrange"] == {"pct": 50.0, "cash": 5_000.0, "shares": 0.0}

    def test_a_ledger_already_holding_is_booked_to_the_first_pair_on_it(
        self, market_open, monkeypatch
    ):
        broker = Broker2()
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tapes = Tapes(monkeypatch, ["AAPL", "INTC"], broker)
        state = make_state(["AAPL", "INTC"])
        broker.prices["AAPL"] = 100.0
        tracker.record_trade("AAPL", "buy", 20, "earlier", "k", "s")
        monkeypatch.setattr(state, "mark_price", lambda symbol: 100.0)
        race = make_split(
            {"AAPL:dayrange": 50.0, "INTC:dayrange": 50.0}, flat_racer("AAPL"), flat_racer("INTC"),
        )
        race.run_cycle(state, tracker)
        cash = 10_000.0 - 2_000.0 - tracker.trade_cost
        assert race.budget_base == pytest.approx(cash + 2_000.0)
        assert race.accounts["AAPL:dayrange"].shares == 20.0
        assert race.accounts["AAPL:dayrange"].cash == pytest.approx(race.budget_base / 2 - 2_000.0)
        assert race.accounts["INTC:dayrange"].cash == pytest.approx(race.budget_base / 2)
        del tapes

    def test_shares_sold_outside_the_app_come_off_the_last_pair_on_the_symbol(
        self, market_open, monkeypatch
    ):
        broker = Broker2()
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tapes = Tapes(monkeypatch, ["AAPL"], broker)
        state = make_state(["AAPL"])
        race = make_split(
            {"AAPL:dayrange": 50.0, "AAPL:highlow": 50.0},
            flat_racer("AAPL", "dayrange"),
            flat_racer("AAPL", "highlow", buy_k=0.40, sell_k=0.30),
        )
        race.run_cycle(state, tracker)
        tapes.bar(AAPL=(106.5, 105.9, 106.6))
        race.run_cycle(state, tracker)
        tapes.bar(AAPL=(103.0, BUY_LEVEL - 0.01, 103.5))
        race.run_cycle(state, tracker)
        assert tracker.position_for("AAPL") == 90.0

        # 50 of the 90 sold by hand: HighLow (listed last) loses its 44 first.
        tracker.record_trade("AAPL", "sell", 50, "by hand", "k", "s")
        # Above HighLow's 106.00 buy, so the flat pair does not buy straight back.
        tapes.bar(AAPL=(106.8, 106.2, 106.9))
        race.run_cycle(state, tracker)
        assert race.accounts["AAPL:highlow"].shares == 0.0
        assert race.accounts["AAPL:dayrange"].shares == 40.0
        assert state.orchestra["holders"] == ["AAPL:dayrange"]

    def test_a_restart_hands_a_dropped_pairs_shares_to_the_next_pair_on_it(
        self, market_open, monkeypatch
    ):
        broker = Broker2()
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tapes = Tapes(monkeypatch, ["AAPL"], broker)
        state = make_state(["AAPL"])
        monkeypatch.setattr(state, "mark_price", lambda symbol: 106.5)
        race = make_split(
            {"AAPL:highlow": 50.0, "AAPL:dayrange": 50.0},
            flat_racer("AAPL", "highlow", buy_k=0.40, sell_k=0.30),
            flat_racer("AAPL", "dayrange"),
        )
        race.run_cycle(state, tracker)
        tapes.bar(AAPL=(106.5, 105.9, 106.6))
        race.run_cycle(state, tracker)
        assert race.accounts["AAPL:highlow"].shares == 44.0

        # Started again without HighLow: its 44 are the day range's to manage.
        again = make_split({"AAPL:dayrange": 50.0}, flat_racer("AAPL", "dayrange"))
        again.run_cycle(state, tracker)
        assert again.accounts["AAPL:dayrange"].shares == 44.0
        assert again.accounts["AAPL:dayrange"].cash == pytest.approx(5_000.0 - 44 * 106.5)
        assert any("no pair bought" in e.get("text", "") for e in state.agent_log)

    def test_the_chart_draws_the_pair_holding_the_symbol(self, market_open, monkeypatch):
        broker = Broker2()
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
        tapes = Tapes(monkeypatch, ["AAPL"], broker)
        state = make_state(["AAPL"])
        race = make_split(
            {"AAPL:dayrange": 50.0, "AAPL:highlow": 50.0},
            flat_racer("AAPL", "dayrange"),
            flat_racer("AAPL", "highlow", buy_k=0.40, sell_k=0.30),
        )
        race.run_cycle(state, tracker)
        tapes.bar(AAPL=(106.5, 105.9, 106.6))
        race.run_cycle(state, tracker)
        race.publish_memory(state)
        config, record = model_overlays.live_trader_view(state, "AAPL")
        assert config.model_key == "highlow" and record["memory"]["entry"] is not None


class TestASplitOfOneIsASingleRun:
    """One pair given the whole budget: the same decisions and log as an
    Apple Trader run -- its account is the ledger's cash to the cent."""

    @pytest.mark.parametrize("kwargs", [
        {},
        {"stop_gain_fraction": 0.5, "min_win_k": 0.3},
        {"scale_in": True, "position_pct": 50.0, "buy_step_k": 0.1},
    ])
    def test_same_decisions_and_log(self, market_open, monkeypatch, kwargs):
        def run(as_split: bool):
            broker = Broker2()
            tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)
            tapes = Tapes(monkeypatch, ["AAPL"], broker)
            state = make_state(["AAPL"])
            config = racer("AAPL", **kwargs)
            if as_split:
                race = make_split({"AAPL:dayrange": 100.0}, config)
                cycle = lambda: race.run_cycle(state, tracker)  # noqa: E731
            else:
                trader = at.DayRangeTrader(config)
                cycle = lambda: trader.run_cycle(BUNDLE, state, tracker)  # noqa: E731
            outcomes = [cycle()]
            for bar in TestARaceOfOneIsASingleRun.PATH:
                tapes.bar(**bar)
                outcomes.append(cycle())
            decisions = [
                (d.symbol, d.action, d.status, d.filled_quantity, d.price, d.reasoning)
                for d in tracker.decisions
            ]
            log = [(e["type"], e.get("text")) for e in state.agent_log]
            return outcomes, decisions, log

        assert run(as_split=True) == run(as_split=False)


class TestSplitConfig:
    def test_refuses_shares_over_100(self):
        with pytest.raises(ValueError, match="at most 100%"):
            ar.OrchestraConfig(
                [racer("AAPL"), racer("INTC")], mode=ar.MODE_SPLIT,
                budget_pct={"AAPL:dayrange": 60.0, "INTC:dayrange": 50.0},
            )

    def test_refuses_a_pair_without_a_share(self):
        with pytest.raises(ValueError, match="INTC · Day Range has no share"):
            ar.OrchestraConfig(
                [racer("AAPL"), racer("INTC")], mode=ar.MODE_SPLIT,
                budget_pct={"AAPL:dayrange": 60.0},
            )

    def test_refuses_the_morning_pick(self):
        from agent_stonks.candidates import SelectionRules

        with pytest.raises(ValueError, match="09:34"):
            ar.OrchestraConfig(
                [racer("AAPL")], mode=ar.MODE_SPLIT, budget_pct={"AAPL:dayrange": 100.0},
                selection=SelectionRules(),
            )

    def test_keeps_only_its_own_pairs_shares(self):
        # A pair left out of the run takes its share into cash.
        race = ar.OrchestraConfig(
            [racer("AAPL")], mode=ar.MODE_SPLIT,
            budget_pct={"AAPL:dayrange": 60.0, "INTC:dayrange": 40.0},
        )
        assert race.budget_pct == {"AAPL:dayrange": 60.0}

    def test_a_race_carries_no_shares(self):
        race = ar.OrchestraConfig([racer("AAPL")], budget_pct={"AAPL:dayrange": 60.0})
        assert race.mode == ar.MODE_RACE and race.budget_pct == {}

    def test_signature_and_record(self):
        from simlab.rule_agents import rule_agent

        agent = rule_agent(ar.ORCHESTRA_KEY)
        split = ar.build_orchestra_config(
            ["AAPL:dayrange", "MU:highlow"], at.AppleTraderConfig(),
            mode=ar.MODE_SPLIT, budget_pct={"AAPL:dayrange": 60.0, "MU:highlow": 40.0},
        )
        sig = ar.orchestra_signature(split)
        assert sig.startswith("orchestra[split:dayrange_AAPL@") and "=60%," in sig
        assert sig.split("](")[0].endswith("=40%")
        assert agent.from_record(agent.to_record(split)) == split
        # A race signs as it always has, and a record from before the split
        # existed is a race.
        race = ar.build_orchestra_config(["AAPL:dayrange", "MU:highlow"], at.AppleTraderConfig())
        pairs = ar.orchestra_signature(race).split("](")[0]
        assert pairs.startswith("orchestra[dayrange_AAPL@") and "%" not in pairs
        legacy = agent.to_record(race)
        del legacy["mode"], legacy["budget_pct"]
        assert agent.from_record(legacy) == race


# --- the pair table's min win box ---------------------------------------------


def _orchestra_form() -> None:
    import streamlit as st

    from agent_stonks import apple_trader_ui

    copy = apple_trader_ui.FormCopy(
        prefix="o", unavailable_suffix="", instrument_help="", model_help=""
    )
    st.session_state["out"] = apple_trader_ui.orchestra_params(None, copy)


class TestPairMinWinBox:
    """Each pair's circuit breaker is switched off by its own box, like Apple
    Trader's; the threshold is kept, greyed out."""

    PAIRS = ["AAPL:dayrange", "INTC:dayrange"]
    KEY, BOX = "o_min_win_k_dayrange_INTC", "o_min_win_on_dayrange_INTC"

    def _run(self, **state):
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_function(_orchestra_form, default_timeout=30)
        app.session_state["o_pairs"] = self.PAIRS
        for key, value in state.items():
            app.session_state[key] = value
        app.run()
        assert not app.exception, app.exception
        return app

    @staticmethod
    def _min_wins(app) -> "dict[str, float]":
        return {r.ticker: r.min_win_k for r in app.session_state["out"].racers}

    def test_every_pair_is_on_by_default(self):
        app = self._run()
        assert app.checkbox(key=self.BOX).value is True
        assert not app.number_input(key=self.KEY).disabled
        assert self._min_wins(app) == {
            "AAPL": pytest.approx(at.min_win_for("AAPL", "dayrange")),
            "INTC": pytest.approx(at.min_win_for("INTC", "dayrange")),
        }

    def test_unticking_one_pair_leaves_the_other_on(self):
        app = self._run()
        app.number_input(key=self.KEY).set_value(0.07).run()
        app.checkbox(key=self.BOX).uncheck().run()
        assert app.number_input(key=self.KEY).disabled
        wins = self._min_wins(app)
        assert wins["INTC"] == 0.0
        assert wins["AAPL"] == pytest.approx(at.min_win_for("AAPL", "dayrange"))
        sig = ar.orchestra_signature(app.session_state["out"])
        assert "dayrange_AAPL@" in sig and "/mw" in sig.split("dayrange_INTC@")[0]
        assert "/mw" not in sig.split("dayrange_INTC@")[1].split(",")[0]
        app.checkbox(key=self.BOX).check().run()
        assert self._min_wins(app)["INTC"] == pytest.approx(0.07)

    def test_a_kept_zero_opens_it_unticked(self):
        """A 0 remembered from before the box (`last_setup`) meant "off"."""
        app = self._run(**{self.KEY: 0.0})
        assert app.checkbox(key=self.BOX).value is False
        assert app.number_input(key=self.KEY).value == pytest.approx(at.min_win_for("INTC"))
        assert self._min_wins(app)["INTC"] == 0.0
