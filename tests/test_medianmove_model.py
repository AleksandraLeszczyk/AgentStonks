"""MedianMove: HighLow's 9:35 question with no model, the notebooks' baseline.

The rule is arithmetic, so most of these pin it on a synthetic history: the
medians are of HighLow's own targets over the last 60 sessions, a trend day
does not move them, the forecast holds the opening range, a short history is
refused. One pins it against HighLow's panel -- the same `prev_avg`, `adr14`
and targets HighLow's features compute -- and the rest pin the seams: the
history read under HighLow's settings for the ticker (one cache, never
rebuilt by the other reader), the registry, the trader's log and the chart,
and the rules: a pair with no base of its own opens on its HighLow model's.
"""

import json
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import pytest

from agent_stonks import apple_models, base_configs
from agent_stonks import apple_trader as at
from agent_stonks import medianmove_model as mm
from agent_stonks import model_overlays as mo
from agent_stonks.decisions import DecisionTracker

from tests.test_apple_trader import (  # noqa: F401 -- fixtures
    FORECAST, FakeBroker, Tape, dayrange_config, market_open, state,
)
from tests.test_base_configs import _base, form  # noqa: F401 -- fixture

DAY = pd.Timestamp("2026-07-21")
SIP = {"opening_feed": "sip", "min_bars": 385, "max_first_minute": 0, "broken_dates": ()}


def bundle(**overrides) -> dict:
    return {
        "kind": mm.KIND, "ticker": "AAPL", "opening_minutes": 5, **SIP,
        "lookback": mm.LOOKBACK, "min_sessions": mm.MIN_SESSIONS, "reads_volume": False,
        **overrides,
    }


def history(n: int = 80, up: float = 0.010, down: float = 0.015, price: float = 100.0,
            end=DAY) -> pd.DataFrame:
    """`n` sessions before `end`, each with its high `up` and its low `down`
    (log) from its own 9:35 price -- `highlow_model.history_frame`'s columns."""
    dates = pd.bdate_range(end=end - pd.Timedelta(days=1), periods=n, name="date")
    close5 = np.full(n, price)
    return pd.DataFrame(
        {
            "open": close5 * 0.999, "high": close5 * np.exp(up), "low": close5 * np.exp(-down),
            "close": close5 * 1.001, "volume": 1e6, "rv": 0.01,
            "open5": close5 * 0.999, "high5": close5 * 1.001, "low5": close5 * 0.998,
            "close5": close5,
        },
        index=dates,
    )


def medianmove_config(**kwargs):
    return replace(dayrange_config(**kwargs), model_key=apple_models.MEDIANMOVE_KEY)


def opening(close5: float = 200.0, high5: float = 200.5, low5: float = 199.5,
            day=DAY, start: str = "09:30", n: int = 5) -> pd.DataFrame:
    """Today's first `n` minute bars, exchange-local, ending on `close5`."""
    idx = pd.date_range(f"{day.date()} {start}", periods=n, freq="1min", tz="America/New_York")
    bars = pd.DataFrame(
        {"open": close5, "high": close5, "low": close5, "close": close5, "volume": 1e4}, index=idx,
    )
    bars.iloc[0, bars.columns.get_loc("high")] = high5
    bars.iloc[1, bars.columns.get_loc("low")] = low5
    return bars


class TestTheRule:
    def test_identical_sessions_forecast_their_own_move(self):
        """Every past high 1% over its 9:35 price and every low 1.5% under: in
        14-day ranges that is 0.4 and 0.6, and today's range is the same, so
        the forecast is today's price moved 1% and 1.5%."""
        out = mm.forecast_from(bundle(), history(), opening(), DAY)

        assert out["median_move"]["up"] == pytest.approx(0.4)
        assert out["median_move"]["down"] == pytest.approx(0.6)
        assert out["pred_high"] == pytest.approx(200.0 * np.exp(0.010))
        assert out["pred_low"] == pytest.approx(200.0 * np.exp(-0.015))
        # What the plan summary quotes, as HighLow reports it.
        assert out["prev_avg"] == pytest.approx(100.0)
        assert out["adr14_abs"] == pytest.approx(100.0 * (np.exp(0.010) - np.exp(-0.015)))
        assert (out["or_high"], out["or_low"]) == (200.5, 199.5)
        assert out["median_move"]["sessions"] == mm.LOOKBACK
        assert out["median_move"]["price"] == 200.0

    def test_a_trend_day_does_not_move_the_median(self):
        """The reason for a median: one session that ran 6% would drag a mean
        of the 60 excursions up by a twentieth. (Older than the 14 sessions
        today's range is read from, so only the median is in question.)"""
        quiet = history()
        trend = quiet.copy()
        trend.iloc[-40, trend.columns.get_loc("high")] = 100.0 * np.exp(0.06)

        out = mm.forecast_from(bundle(), trend, opening(), DAY)
        assert out["pred_high"] == pytest.approx(
            mm.forecast_from(bundle(), quiet, opening(), DAY)["pred_high"]
        )
        assert mm.excursions(trend).tail(60)["up"].mean() > 1.03 * out["median_move"]["up"]

    def test_only_the_last_60_sessions_count(self):
        """Older sessions that moved twice as far are outside the window (and
        outside the 14 sessions today's range is read from)."""
        recent = history(n=80)
        older = history(n=40, up=0.02, down=0.03, end=recent.index[0])
        out = mm.forecast_from(bundle(), pd.concat([older, recent]), opening(), DAY)

        assert out["median_move"]["from"] == recent.index[20].strftime("%Y-%m-%d")
        assert out["pred_high"] == pytest.approx(200.0 * np.exp(0.010))

    def test_a_short_history_is_refused(self):
        """30 sessions leave 16 with 14 before them -- under the 20 a median needs."""
        with pytest.raises(ValueError, match="need 20"):
            mm.forecast_from(bundle(), history(n=30), opening(), DAY)

    def test_the_forecast_holds_the_opening_range(self):
        """Clipped as HighLow clips: the session's high cannot finish under
        what the first five minutes printed."""
        out = mm.forecast_from(bundle(), history(), opening(high5=205.0, low5=190.0), DAY)

        assert (out["pred_high"], out["pred_low"]) == (205.0, 190.0)

    def test_a_session_without_an_opening_sets_the_range_but_not_a_move(self):
        """An IEX opening feed that printed nothing: no 9:35 price, so no
        excursion, but its high and low are still part of the 14-day range."""
        past = history()
        past.iloc[-1, past.columns.get_loc("close5")] = np.nan
        past.iloc[-1, past.columns.get_loc("high")] = 100.0 * np.exp(0.110)

        out = mm.forecast_from(bundle(), past, opening(), DAY)

        assert out["median_move"]["to"] == past.index[-2].strftime("%Y-%m-%d")
        assert out["median_move"]["adr14"] == pytest.approx((13 * 0.025 + 0.125) / 14)

    def test_history_on_or_after_the_day_is_never_read(self):
        later = history(n=5, up=0.05, down=0.05, end=DAY + pd.Timedelta(days=8))
        out = mm.forecast_from(bundle(), pd.concat([history(), later]), opening(), DAY)

        assert out["pred_high"] == pytest.approx(200.0 * np.exp(0.010))

    def test_a_window_that_misses_the_open_is_refused(self):
        with pytest.raises(ValueError, match="does not start at 09:30"):
            mm.forecast_from(bundle(), history(), opening(start="09:31"), DAY)
        with pytest.raises(ValueError, match="only 4 bars"):
            mm.forecast_from(bundle(), history(), opening(n=4), DAY)

    def test_an_iex_ticker_reads_the_price_off_its_own_window(self):
        """MU: HighLow's history holds IEX openings, so today's price is IEX's
        too, whatever tape the caller's bars are on."""
        iex = bundle(opening_feed="iex")
        out = mm.forecast_from(iex, history(), opening(), DAY, opening(close5=210.0, n=3))

        assert out["median_move"]["price"] == 210.0
        with pytest.raises(ValueError, match="IEX printed nothing"):
            mm.forecast_from(iex, history(), opening(), DAY, None)


class TestAgainstHighLow:
    """The numbers HighLow's own features compute from the same history."""

    def test_the_range_and_targets_are_highlows(self):
        H = pytest.importorskip("agent_stonks.highlow_model")
        rng = np.random.default_rng(7)
        past = history(n=90)
        past["close5"] = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, len(past))))
        past["high"] = past["close5"] * np.exp(rng.uniform(0.002, 0.03, len(past)))
        past["low"] = past["close5"] * np.exp(-rng.uniform(0.002, 0.03, len(past)))
        past["open"] = past["close5"] * np.exp(rng.normal(0, 0.003, len(past)))
        past["close"] = past["close5"] * np.exp(rng.normal(0, 0.005, len(past)))
        past["open5"], past["high5"], past["low5"] = past["open"], past["high"], past["low"]
        # The opening summary's other columns, which HighLow's panel reads and
        # this rule does not.
        for col in H._OPENING_COLS:
            if col not in past:
                past[col] = 1e4 if col == "volume5" else 0.01
        today = opening(close5=float(past["close5"].iloc[-1]))
        out = mm.forecast_from(bundle(), past, today, DAY)

        minute = H.add_session_columns(today)
        daily = pd.concat([past[H._ROLLUP_COLS], H.daily_bars(minute)[H._ROLLUP_COLS]])
        opening_rows = pd.concat([past[H._OPENING_COLS], H.opening_features(minute)[H._OPENING_COLS]])
        panel = H.panel_from(daily, opening_rows)
        row = panel.loc[DAY]
        assert out["prev_avg"] == pytest.approx(row["prev_avg"], rel=1e-12)
        assert out["adr14_abs"] == pytest.approx(row["adr14_usd"], rel=1e-12)
        assert out["median_move"]["adr14"] == pytest.approx(row["adr14"], rel=1e-12)

        before = panel[panel.index < DAY]
        up = (np.log(past.loc[before.index, "high"] / before["close5"]) / before["adr14"]).dropna()
        down = (np.log(before["close5"] / past.loc[before.index, "low"]) / before["adr14"]).dropna()
        pred = H.to_prices(np.array([[up.tail(60).median(), down.tail(60).median()]]), panel.loc[[DAY]])
        assert out["pred_high"] == pytest.approx(float(pred["pred_high"].iloc[0]), rel=1e-12)
        assert out["pred_low"] == pytest.approx(float(pred["pred_low"].iloc[0]), rel=1e-12)


class TestHistory:
    def test_the_cache_is_read_under_highlows_settings_for_the_ticker(self, monkeypatch, tmp_path):
        """One file per ticker, written under one set of settings: read under
        any other, HighLow's cache would be rebuilt -- and rebuilt back by
        HighLow the next morning."""
        H = pytest.importorskip("agent_stonks.highlow_model")
        sidecar = tmp_path / "highlow15m_MU.json"
        sidecar.write_text(json.dumps({"opening_feed": "iex", "ticker": "MU"}))
        monkeypatch.setattr(H, "model_path", lambda ticker=None: tmp_path / f"highlow15m_{ticker}.joblib")
        monkeypatch.setattr(H, "metadata_path", lambda path=None: path.with_suffix(".json"))

        assert H.history_settings("MU") == {**SIP, "opening_feed": "iex"}
        # No sidecar: the notebook's thresholds, SIP openings.
        assert H.history_settings("TMO") == {**SIP, "min_bars": 260, "max_first_minute": 1}
        assert H.history_settings("AMGN")["min_bars"] == 310
        assert H.history_settings("MCD") == {**SIP, "min_bars": 360, "broken_dates": ("2023-01-24",)}
        assert H.history_settings("GOOG") == SIP
        assert mm.load_bundle("mu")["opening_feed"] == "iex"

        asked = []

        def history_frame(symbol, before, key=None, secret=None, **kwargs):
            asked.append((symbol, kwargs))
            return history()

        monkeypatch.setattr(H, "history_frame", history_frame)
        monkeypatch.setattr(mm, "_forecast_cache", {})
        tmo = mm.load_bundle("TMO")
        first = mm.forecast_session(tmo, "TMO", opening(), DAY)
        again = mm.forecast_session(tmo, "TMO", opening(), DAY)

        assert asked == [("TMO", {**SIP, "min_bars": 260, "max_first_minute": 1})]
        assert again == first and again is not first

    def test_a_missing_credential_says_whose_history_it_is(self, monkeypatch):
        H = pytest.importorskip("agent_stonks.highlow_model")

        def history_frame(*args, **kwargs):
            raise ValueError("no Alpaca credentials are available")

        monkeypatch.setattr(H, "history_frame", history_frame)
        monkeypatch.setattr(mm, "_forecast_cache", {})
        with pytest.raises(ValueError, match="MedianMove reads the SIP history HighLow keeps"):
            mm.forecast_session(bundle(), "AAPL", opening(), DAY)


class TestRegistry:
    def test_every_highlow_instrument_has_its_baseline(self):
        for key in (apple_models.HIGHLOW_KEY, apple_models.HIGHLOW2_KEY, apple_models.HIGHLOW_CHAIN_KEY):
            for ticker in apple_models.MODELS[key].tickers:
                assert apple_models.covers(apple_models.MEDIANMOVE_KEY, ticker), (key, ticker)
        assert apple_models.strategy(apple_models.MEDIANMOVE_KEY) == apple_models.STRATEGY_DAYRANGE

    def test_the_bundle_is_the_rules_settings(self):
        pytest.importorskip("agent_stonks.highlow_model")
        found = apple_models.load(apple_models.MEDIANMOVE_KEY, "goog")
        assert found["kind"] == "medianmove" and found["ticker"] == "GOOG"
        assert found["opening_minutes"] == 5 and not found["reads_volume"]

    def test_a_stored_replay_is_fingerprinted_on_the_rule(self):
        """Nothing saved: the rule's source is what a replay was made from."""
        assert apple_models.bundle_files(apple_models.MEDIANMOVE_KEY, "AAPL") == [mm.rule_path()]

    def test_the_chart_draws_its_range(self):
        assert mo.for_models([apple_models.MEDIANMOVE_KEY], "AAPL")["keys"] == [mo.MEDIANMOVE_RANGE_KEY]
        assert mo.OVERLAYS[mo.MEDIANMOVE_RANGE_KEY].covers("GOOG")


class TestTrader:
    def _medianmove(self, monkeypatch, **overrides) -> dict:
        def forecast(bundle, ticker, opening_bars, today, key=None, secret=None):
            return {**FORECAST, "median_move": {
                "up": 0.41, "down": 0.52, "sessions": 60, "from": "2026-04-24",
                "to": "2026-07-20", "price": 101.4, "adr14": 0.02,
            }}

        monkeypatch.setattr(mm, "forecast_session", forecast)
        return bundle(**overrides)

    def test_it_trades_the_day_range_rules_and_says_where_the_range_came_from(
        self, state, market_open, monkeypatch
    ):
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(103.0))
        tape = Tape(monkeypatch)
        tape.append(103.0)
        trader = at.DayRangeTrader(medianmove_config())

        trader.run_cycle(self._medianmove(monkeypatch), state, tracker)

        assert (trader.plan["buy_level"], trader.plan["sell_level"]) == (102.5, 109.0)
        summary = [e["text"] for e in state.agent_log if "No model:" in e.get("text", "")]
        assert len(summary) == 1
        assert "09:35 price $101.40" in summary[0] and "last 60 sessions (0.41" in summary[0]

    @pytest.mark.parametrize("reads_volume, caveats", [(False, 0), (True, 1)])
    def test_no_volume_caveat_for_a_rule_that_reads_no_volume(
        self, state, market_open, monkeypatch, reads_volume, caveats
    ):
        """HighLow's IEX caveat is about its opening-volume features; this rule
        has none. (The same run on a bundle that read volume would get it.)"""
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(103.0))
        tape = Tape(monkeypatch)
        monkeypatch.setattr(at.agent_mod, "fetch_bars_window", lambda *a, **k: [])
        monkeypatch.setattr(at.historical, "fetch_intraday_bars", lambda *a, **k: [])
        state.data_source, state.feed = "alpaca", "iex"
        tape.append(103.0)

        at.DayRangeTrader(medianmove_config()).run_cycle(
            self._medianmove(monkeypatch, reads_volume=reads_volume), state, tracker
        )

        assert state.agent_log and caveats == len(
            [e for e in state.agent_log if "Forecast caveat" in e.get("text", "")]
        )


class TestRules:
    """MedianMove trades its HighLow model's rules (`apple_models.rules_of`):
    a pair with no base of its own opens on that model's, so the two differ
    in the forecast alone. The conftest's base store starts empty."""

    MM = apple_models.MEDIANMOVE_KEY

    def test_its_highlow_model_on_every_instrument(self):
        for ticker in apple_models.MODELS[self.MM].tickers:
            parent = apple_models.rules_of(self.MM, ticker)
            assert parent == ("highlow2" if ticker == "GOOG" else "highlow"), ticker
            assert apple_models.covers(parent, ticker)
            assert apple_models.strategy(parent) == apple_models.STRATEGY_DAYRANGE
        assert apple_models.rules_of(apple_models.HIGHLOW_KEY, "AAPL") is None

    @pytest.mark.parametrize("ticker, parent", [("INTC", "highlow"), ("GOOG", "highlow2")])
    def test_a_pair_without_a_base_opens_on_its_highlow_models(self, ticker, parent):
        base_configs.set_base(asdict(_base(
            ticker=ticker, model_key=parent, breach_update="conditional", skip_events=("nfp",),
        )), source={"job_id": "j-hl"})
        assert at.default_config(ticker, self.MM) == replace(
            at.default_config(ticker, parent), model_key=self.MM
        )
        assert at.dayrange_levels(ticker, self.MM) == (0.85, 0.15)
        assert at.stop_for(ticker, self.MM) == 0.7
        assert at.breach_for(ticker, self.MM) == "conditional"
        assert at.confirmation_for(ticker, self.MM) == 7
        assert base_configs.entry(self.MM, ticker) is None  # nothing written for it

    def test_its_own_base_wins(self):
        base_configs.set_base(asdict(_base(ticker="INTC")))
        base_configs.set_base(asdict(_base(ticker="INTC", model_key=self.MM, buy_k=0.5)))
        assert at.default_config("INTC", self.MM).buy_k == 0.5
        assert at.default_config("INTC", "highlow").buy_k == 0.85

    def test_without_either_base_the_highlow_models_shipped_tuning(self):
        for ticker in ("AAPL", "INTC", "MU", "GOOG"):
            parent = apple_models.rules_of(self.MM, ticker)
            assert at.shipped_config(ticker, self.MM) == replace(
                at.shipped_config(ticker, parent), model_key=self.MM
            )
            assert at.default_config(ticker, self.MM) == replace(
                at.default_config(ticker, parent), model_key=self.MM
            )

    def test_the_form_opens_on_highlows_base_under_its_keys(self, form):
        base_configs.set_base(asdict(_base()))
        app = form()
        app.selectbox(key="t_model_AAPL").select(self.MM).run()
        assert not app.exception
        suffix = base_configs.key_suffix("highlow", "AAPL")
        assert suffix and app.number_input(key=f"t_buy_k_{self.MM}_AAPL{suffix}").value == 0.85
        assert app.session_state["out"] == replace(_base(), model_key=self.MM)

    def test_the_help_names_whose_base_it_is(self):
        from agent_stonks import apple_trader_ui

        assert apple_trader_ui.base_help(self.MM, "AAPL") == ""
        base_configs.set_base(asdict(_base()), source={"job_id": "j-hl"})
        text = apple_trader_ui.base_help(self.MM, "AAPL")
        assert "HighLow model's, whose rules this model trades" in text and "j-hl" in text
