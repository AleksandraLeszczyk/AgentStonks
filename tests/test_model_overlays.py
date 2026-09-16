"""Tests for the model-prediction chart overlays (agent_stonks/model_overlays.py).

Two halves. The first drives `compute` with stubbed models, so what is pinned
is the *shape* of the answer -- which item kinds each overlay produces, and that a
missing model becomes a note rather than an exception. The second drives the renderer, so what is pinned is
that each kind reaches the figure the way its idiom requires: a level as a line
in the price panel AND in the profile beside it, a time span as a
semi-transparent background, a moment as a vertical line plus a marker.

The real bundles are not required (and are deliberately not used): the models
themselves have their own tests, and an overlay test that needed a 200 MB
checkpoint would only ever run on one machine.
"""

from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from agent_stonks import apple_models, model_overlays as mo
from agent_stonks.charts import add_model_overlays, build_chart, overlay_x_max

SESSION = "2026-08-07"
SESSION_START = datetime(2026, 8, 7, 13, 25, tzinfo=timezone.utc)


def minute_bars(n=180, seed=7, base=200.0):
    """A session of 1-minute bars starting at the 09:30 open.

    A random walk with a sine wave on top, so the session has a real range.
    """
    rng = np.random.default_rng(seed)
    start = pd.Timestamp(f"{SESSION} 09:30", tz="America/New_York")
    walk = base + np.cumsum(rng.normal(0, 0.08, n)) + np.sin(np.arange(n) / 11) * 1.1
    bars = []
    for i, price in enumerate(walk):
        close = price + rng.normal(0, 0.03)
        bars.append({
            "t": (start + pd.Timedelta(minutes=i)).tz_convert("UTC").isoformat(),
            "o": float(price),
            "h": float(max(price, close) + 0.06),
            "l": float(min(price, close) - 0.06),
            "c": float(close),
            "v": float(rng.integers(1_000, 9_000)),
        })
    return bars


class TestCatalogue:
    def test_every_overlay_is_offered_for_the_symbol_all_models_cover(self):
        assert mo.keys_for("AAPL") == mo.keys()

    def test_a_symbol_only_the_profile_model_covers_gets_only_that(self):
        assert mo.keys_for("MSFT") == [mo.PROFILE_RANGE_KEY]

    def test_each_overlay_follows_its_models_tickers(self):
        """The catalogue reads `apple_models`, so re-running a notebook for a
        new symbol widens the picker without a change here; only a symbol
        nothing was fitted on is excluded."""
        for symbol in ("AAPL", "GOOGL", "INTC"):
            assert mo.OVERLAYS[mo.DAY_RANGE_KEY].covers(symbol), symbol
        assert not mo.OVERLAYS[mo.DAY_RANGE_KEY].covers("MSFT")
        # ...except the one model that claims to transfer, which covers every
        # symbol because it was fitted without ticker dummies.
        assert mo.OVERLAYS[mo.PROFILE_RANGE_KEY].covers("MSFT")

    def test_label_falls_back_to_the_key_for_an_unknown_overlay(self):
        assert mo.label("nope") == "nope"


class TestForModels:
    """`for_models` is the seam that lets a chart of a past run open showing
    what the model behind it said, without the caller knowing which overlay
    draws which model."""

    def test_the_day_range_model_selects_the_day_range_overlay(self):
        assert mo.for_models([apple_models.DAYRANGE_KEY], "AAPL")["keys"] == [
            mo.DAY_RANGE_KEY
        ]

    def test_an_overlay_is_never_selected_for_a_symbol_it_has_no_model_for(self):
        """A rule run's stored symbol list can be wider than the one instrument
        it traded, so this is asked about tabs the model was never fitted on --
        where the picker has no such option to select."""
        assert mo.for_models([apple_models.DAYRANGE_KEY], "MSFT")["keys"] == []

    def test_no_symbol_asks_the_question_without_one(self):
        """The default is "which overlays draw these models", not "none" --
        filtering is what naming a symbol adds."""
        assert mo.for_models([apple_models.DAYRANGE_KEY])["keys"] == [
            mo.DAY_RANGE_KEY
        ]

    def test_the_transferable_profile_model_is_never_auto_selected(self):
        """It drives no agent and is not in `apple_models`, so nothing can name
        it -- selecting it would be the chart's opinion, not the run's."""
        for key in apple_models.keys():
            assert mo.PROFILE_RANGE_KEY not in mo.for_models([key], "AAPL")["keys"]

    def test_no_models_is_no_selection(self):
        for empty in ([], None, [""]):
            out = mo.for_models(empty, "AAPL")
            assert out == {"keys": [], "unmatched": []}

    def test_a_retired_model_key_is_unmatched_not_a_crash(self):
        """Records are JSON on disk and outlive the registry."""
        out = mo.for_models(["retired_model"], "AAPL")
        assert out["keys"] == [] and out["unmatched"] == ["retired_model"]


class TestCompute:
    def test_no_keys_is_no_work(self):
        assert mo.compute([], "AAPL", minute_bars()) == {"items": [], "notes": []}

    def test_no_bars_is_no_work(self):
        assert mo.compute(mo.keys(), "AAPL", []) == {"items": [], "notes": []}

    def test_unknown_keys_are_ignored(self):
        assert mo.compute(["nope"], "AAPL", minute_bars())["items"] == []

    def test_a_symbol_the_model_does_not_cover_is_a_note_not_an_error(self):
        result = mo.compute([mo.DAY_RANGE_KEY], "MSFT", minute_bars())
        assert result["items"] == []
        assert "MSFT" in result["notes"][0]
        assert "fitted on" in result["notes"][0]

    def test_a_missing_bundle_is_a_note_not_an_exception(self, monkeypatch):
        monkeypatch.setattr(mo.apple_models, "load", lambda *a, **k: None)
        result = mo.compute([mo.DAY_RANGE_KEY], "AAPL", minute_bars())
        assert result["items"] == []
        assert result["notes"] and "Predicted day range" in result["notes"][0]

    def test_a_model_that_raises_is_a_note_not_an_exception(self, monkeypatch):
        def boom(*_a, **_k):
            raise RuntimeError("the checkpoint is corrupt")

        monkeypatch.setattr(mo.apple_models, "load", boom)
        result = mo.compute([mo.DAY_RANGE_KEY], "AAPL", minute_bars())
        assert result["items"] == []
        assert "the checkpoint is corrupt" in result["notes"][0]

    def test_bars_from_another_day_produce_a_note(self):
        result = mo.compute([mo.DAY_RANGE_KEY], "AAPL", minute_bars(),
                            session_date="2026-08-06")
        assert result["items"] == []
        assert "2026-08-06" in result["notes"][0]


class TestProfileRangeOverlay:
    """The LevelsML pack reduced to levels.

    `profile_model` is stubbed here for the same reason the day-range forecast is:
    its own tests pin the quantiles, and these pin what the chart does with
    them.
    """

    def stub_pack(self, monkeypatch, quantiles=(-40.0, 0.0, 60.0)):
        monkeypatch.setattr(
            mo.profile_model, "load_pack", lambda: {"p_levels": [5, 50, 95]}
        )
        monkeypatch.setattr(
            mo.profile_model, "compute_features", lambda *a, **k: {"stub": 1.0}
        )
        monkeypatch.setattr(
            mo.profile_model, "predict_quantiles",
            lambda pack, feats: np.array(quantiles),
        )

    def test_produces_two_bounds_a_poc_and_a_band(self, monkeypatch):
        self.stub_pack(monkeypatch)
        items = mo.compute([mo.PROFILE_RANGE_KEY], "MSFT", minute_bars(),
                           daily_bars=[])["items"]
        kinds = [i["kind"] for i in items]
        assert kinds.count("level") == 3
        assert kinds.count("span") == 1
        band = next(i for i in items if i["kind"] == "span")
        assert band["y0"] < band["y1"]
        assert band["forward"] is False  # a claim about today, clipped to today

    def test_the_levels_name_the_session_they_belong_to(self, monkeypatch):
        self.stub_pack(monkeypatch)
        items = mo.compute([mo.PROFILE_RANGE_KEY], "MSFT", minute_bars(),
                           daily_bars=[])["items"]
        for item in items:
            if item["kind"] == "level":
                assert pd.Timestamp(item["x0"]).date() == pd.Timestamp(SESSION).date()
                assert pd.Timestamp(item["x1"]).date() == pd.Timestamp(SESSION).date()

    def test_the_bounds_are_the_outer_quantiles_in_price(self, monkeypatch):
        self.stub_pack(monkeypatch, quantiles=(-40.0, 0.0, 60.0))
        bars = minute_bars()
        open_px = bars[0]["o"]
        items = mo.compute([mo.PROFILE_RANGE_KEY], "MSFT", bars, daily_bars=[])["items"]
        values = {i["label"]: i["value"] for i in items if i["kind"] == "level"}
        assert values["Pred. q5"] == pytest.approx(open_px * np.exp(-40.0 / 1e4))
        assert values["Pred. q95"] == pytest.approx(open_px * np.exp(60.0 / 1e4))

    def test_a_missing_pack_is_a_note(self, monkeypatch):
        monkeypatch.setattr(mo.profile_model, "load_pack", lambda: None)
        result = mo.compute([mo.PROFILE_RANGE_KEY], "MSFT", minute_bars())
        assert result["items"] == []
        assert "Predicted price profile range" in result["notes"][0]

    def test_too_little_daily_history_is_a_note(self, monkeypatch):
        monkeypatch.setattr(mo.profile_model, "load_pack", lambda: {"p_levels": [5, 95]})
        monkeypatch.setattr(mo.profile_model, "compute_features", lambda *a, **k: None)
        result = mo.compute([mo.PROFILE_RANGE_KEY], "MSFT", minute_bars(), daily_bars=[])
        assert result["items"] == []
        assert "daily bars" in result["notes"][0]


class TestDayRangeOverlay:
    def stub_forecast(self, monkeypatch, high=210.0, low=198.0):
        import agent_stonks.dayrange_model as dr

        monkeypatch.setattr(mo.apple_models, "load", lambda *a, **k: {"stub": True})
        monkeypatch.setattr(dr, "opening_minutes", lambda bundle=None: 5)
        monkeypatch.setattr(dr, "daily_frame_from_bars", lambda bars: pd.DataFrame())
        monkeypatch.setattr(
            dr, "forecast_session",
            lambda *a, **k: {
                "pred_high": high, "pred_low": low, "prev_avg": 200.0,
                "adr14_abs": 3.0, "or_high": 201.0, "or_low": 199.0,
            },
        )

    def test_produces_two_levels_and_the_band_between_them(self, monkeypatch):
        pytest.importorskip("agent_stonks.dayrange_model")
        self.stub_forecast(monkeypatch)
        items = mo.compute([mo.DAY_RANGE_KEY], "AAPL", minute_bars(),
                           daily_bars=[], session_date=SESSION)["items"]
        levels = {i["label"]: i["value"] for i in items if i["kind"] == "level"}
        assert levels == {"Pred. high": 210.0, "Pred. low": 198.0}
        band = next(i for i in items if i["kind"] == "span")
        assert (band["y0"], band["y1"]) == (198.0, 210.0)

    def test_the_band_runs_from_the_forecast_to_the_closing_bell(self, monkeypatch):
        pytest.importorskip("agent_stonks.dayrange_model")
        self.stub_forecast(monkeypatch)
        items = mo.compute([mo.DAY_RANGE_KEY], "AAPL", minute_bars(),
                           daily_bars=[], session_date=SESSION)["items"]
        band = next(i for i in items if i["kind"] == "span")
        close = pd.Timestamp(band["x1"]).tz_convert("America/New_York")
        assert (close.hour, close.minute) == (16, 0)
        # The 5-minute opening window ends on the 09:34 bar.
        start = pd.Timestamp(band["x0"]).tz_convert("America/New_York")
        assert (start.hour, start.minute) == (9, 34)

    def test_bars_that_miss_the_open_refuse_rather_than_forecast(self, monkeypatch):
        pytest.importorskip("agent_stonks.dayrange_model")
        self.stub_forecast(monkeypatch)
        late = minute_bars()[30:]
        result = mo.compute([mo.DAY_RANGE_KEY], "AAPL", late,
                            daily_bars=[], session_date=SESSION)
        assert result["items"] == []
        assert "09:30 open" in result["notes"][0]

    def test_fewer_bars_than_the_opening_window_is_a_note(self, monkeypatch):
        pytest.importorskip("agent_stonks.dayrange_model")
        self.stub_forecast(monkeypatch)
        result = mo.compute([mo.DAY_RANGE_KEY], "AAPL", minute_bars(n=3),
                            daily_bars=[], session_date=SESSION)
        assert result["items"] == []
        assert "first 5 minutes" in result["notes"][0]


INTRAVOL = {
    "shape": {
        "params": {"a": 0.6, "b": 3.0, "alpha": 0.45, "c": 0.5, "kappa": 12.0},
        "t_domain": [2.5, 387.5],
    },
    "day_range": {
        "coef": {"const": -1.0, "lr_d": 0.2, "lr_w": 0.3, "lr_m": 0.3, "abs_gap": 5.0},
    },
}


def daily_history(n=30, width=0.02, close=200.0):
    days = pd.bdate_range(end=pd.Timestamp(SESSION) - pd.Timedelta(days=1), periods=n)
    return [
        {"t": str(d.date()), "o": close, "h": close * (1 + width), "l": close, "c": close}
        for d in days
    ]


def stub_intravol(monkeypatch, model=INTRAVOL):
    monkeypatch.setattr(mo.intraday_vol_model, "load", lambda ticker=None: model)


def the_band(items):
    return next(i for i in items if i["kind"] == "band")


def et(stamp):
    return pd.Timestamp(stamp).tz_convert("America/New_York")


class TestIntradayRangeOverlay:
    """IntradayVolatility alone: its own day-range forecast, shaped by time of day."""

    def compute(self, bars=None, **kwargs):
        kwargs.setdefault("daily_bars", daily_history())
        return mo.compute([mo.INTRADAY_RANGE_KEY], "AAPL", bars or minute_bars(),
                          session_date=SESSION, **kwargs)

    def test_is_one_band_over_the_whole_session(self, monkeypatch):
        stub_intravol(monkeypatch)
        items = self.compute()["items"]
        assert [i["kind"] for i in items] == ["band"]
        band = items[0]
        assert len(band["t"]) == len(band["upper"]) == len(band["lower"]) == 391
        assert (et(band["t"][0]).hour, et(band["t"][0]).minute) == (9, 30)
        assert (et(band["t"][-1]).hour, et(band["t"][-1]).minute) == (16, 0)
        assert band["forward"] is False

    def test_it_is_widest_at_the_open_and_narrows_through_midday(self, monkeypatch):
        stub_intravol(monkeypatch)
        band = the_band(self.compute()["items"])
        width = np.array(band["upper"]) - np.array(band["lower"])
        assert int(np.argmax(width)) == 0
        assert width[180] < 0.5 * width[0]

    def test_it_splits_its_forecast_range_evenly_around_the_open(self, monkeypatch):
        stub_intravol(monkeypatch)
        bars = minute_bars()
        band = the_band(self.compute(bars)["items"])
        assert band["upper"][0] * band["lower"][0] == pytest.approx(bars[0]["o"] ** 2)

    def test_the_official_opening_print_is_the_centre_when_supplied(self, monkeypatch):
        stub_intravol(monkeypatch)
        band = the_band(self.compute(open_price=205.0)["items"])
        assert band["upper"][0] * band["lower"][0] == pytest.approx(205.0 ** 2)

    def test_too_little_daily_history_is_a_note(self, monkeypatch):
        stub_intravol(monkeypatch)
        result = self.compute(daily_bars=daily_history(n=5))
        assert result["items"] == []
        assert "22 completed daily bars" in result["notes"][0]

    def test_a_missing_export_is_a_note_that_says_how_to_make_one(self, monkeypatch):
        stub_intravol(monkeypatch, model=None)
        result = self.compute()
        assert result["items"] == []
        assert "Predicted intraday range" in result["notes"][0]
        assert "export_app_model.py" in result["notes"][0]

    def test_bars_that_miss_the_open_with_no_opening_print_are_a_note(self, monkeypatch):
        stub_intravol(monkeypatch)
        result = self.compute(minute_bars()[30:])
        assert result["items"] == []
        assert "09:30 open" in result["notes"][0]


class TestIntradayDayRangeOverlay:
    """The same curve, stretched to TimeToChange3's predicted high and low."""

    def stub(self, monkeypatch, high=210.0, low=198.0):
        pytest.importorskip("agent_stonks.dayrange_model")
        stub_intravol(monkeypatch)
        TestDayRangeOverlay().stub_forecast(monkeypatch, high=high, low=low)

    def test_it_tops_out_at_the_predicted_high_and_bottoms_out_at_the_low(
        self, monkeypatch
    ):
        self.stub(monkeypatch)
        items = mo.compute([mo.INTRADAY_DAYRANGE_KEY], "AAPL", minute_bars(),
                           daily_bars=[], session_date=SESSION)["items"]
        band = the_band(items)
        assert max(band["upper"]) == pytest.approx(210.0)
        assert min(band["lower"]) == pytest.approx(198.0)
        width = np.array(band["upper"]) - np.array(band["lower"])
        assert int(np.argmax(width)) == 0 and width[180] < width[0]

    def test_selecting_both_day_range_overlays_forecasts_once(self, monkeypatch):
        import agent_stonks.dayrange_model as dr

        self.stub(monkeypatch)
        calls = []
        forecast = dr.forecast_session
        monkeypatch.setattr(
            dr, "forecast_session", lambda *a, **k: calls.append(1) or forecast(*a, **k)
        )
        items = mo.compute([mo.DAY_RANGE_KEY, mo.INTRADAY_DAYRANGE_KEY], "AAPL",
                           minute_bars(), daily_bars=[], session_date=SESSION)["items"]
        assert len(calls) == 1
        assert {i["kind"] for i in items} == {"level", "span", "band"}

    def test_no_forecast_is_a_note_naming_this_overlay(self, monkeypatch):
        self.stub(monkeypatch)
        monkeypatch.setattr(mo.apple_models, "load", lambda *a, **k: None)
        result = mo.compute([mo.INTRADAY_DAYRANGE_KEY], "AAPL", minute_bars(),
                            daily_bars=[], session_date=SESSION)
        assert result["items"] == []
        assert result["notes"][0].startswith("Predicted intraday range × day range:")


class TestTraderLevelsOverlay:
    """Apple Trader's resting orders, drawn from a configuration.

    The odd overlay out: every other one draws what a model said, this one draws
    what an agent would do about it. So what is pinned here is that it is the
    *agent's* answer -- the same levels `DayRangeTrader` would rest, moving when
    its settings say they move -- rather than a second derivation that could
    drift from the loop.
    """

    # 210/198 forecast on a $3 ADR, so the notebook's 0.75/0.10 put the buy at
    # 207.75 and the sell at 209.70.
    def stub_forecast(self, monkeypatch, high=210.0, low=198.0):
        TestDayRangeOverlay().stub_forecast(monkeypatch, high=high, low=low)

    def config(self, **kwargs):
        from agent_stonks.apple_trader import AppleTraderConfig

        kwargs.setdefault("ticker", "AAPL")
        kwargs.setdefault("buy_k", 0.75)
        kwargs.setdefault("sell_k", 0.10)
        kwargs.setdefault("breach_update", "off")
        return AppleTraderConfig(**kwargs)

    def items(self, monkeypatch, config=None, bars=None):
        pytest.importorskip("agent_stonks.dayrange_model")
        self.stub_forecast(monkeypatch)
        return mo.compute(
            [mo.TRADER_LEVELS_KEY], "AAPL", bars if bars is not None else minute_bars(),
            daily_bars=[], session_date=SESSION, trader_config=config,
        )

    def test_levels_that_never_move_are_drawn_as_two_flat_lines(self, monkeypatch):
        """Flat is the notebook's shape, and a level (unlike a band) mirrors
        into the price profile -- where a resting order is exactly the thing to
        read against traded volume."""
        items = self.items(monkeypatch, self.config())["items"]
        levels = {i["label"]: i["value"] for i in items if i["kind"] == "level"}
        assert levels == {"Buy level": pytest.approx(207.75),
                          "Sell level": pytest.approx(209.70)}
        assert not [i for i in items if i["kind"] == "band"]

    def test_the_two_distances_are_the_configured_ones(self, monkeypatch):
        items = self.items(monkeypatch, self.config(buy_k=1.0, sell_k=0.5))["items"]
        levels = {i["label"]: i["value"] for i in items if i["kind"] == "level"}
        assert levels == {"Buy level": pytest.approx(207.0),
                          "Sell level": pytest.approx(208.5)}

    def test_the_flat_lines_run_from_the_forecast_to_the_closing_bell(self, monkeypatch):
        """Nothing is resting before 09:35 -- the forecast does not exist yet."""
        items = self.items(monkeypatch, self.config())["items"]
        buy = next(i for i in items if i["label"] == "Buy level")
        start = pd.Timestamp(buy["x0"]).tz_convert("America/New_York")
        close = pd.Timestamp(buy["x1"]).tz_convert("America/New_York")
        assert (start.hour, start.minute) == (9, 34)
        assert (close.hour, close.minute) == (16, 0)

    def test_a_reference_that_moves_is_drawn_as_a_band(self, monkeypatch):
        """Under the intraday source the levels follow the clock, and two flat
        lines would be a picture of a strategy the run is not using."""
        monkeypatch.setattr(
            mo.intraday_vol_model, "load",
            lambda *a, **k: {
                "shape": {"params": {"a": 0.2, "b": 0.8, "alpha": 1.0, "c": 0.0,
                                     "kappa": 30.0},
                          "t_domain": [0.0, 389.5]},
                "day_range": {"coef": {}},
            },
        )
        items = self.items(monkeypatch, self.config(level_source="intraday"))["items"]
        band = next(i for i in items if i["kind"] == "band")
        assert not [i for i in items if i["kind"] == "level"]
        assert len(band["t"]) == len(band["lower"]) == len(band["upper"])
        # Buy under sell at every minute, and the pair pulls in through the day.
        assert all(lo < up for lo, up in zip(band["lower"], band["upper"]))
        assert band["lower"][-1] < band["lower"][0]

    def test_a_breach_moves_the_levels_and_switches_it_to_a_band(self, monkeypatch):
        """The same settings that move the agent's orders move the drawing --
        which is the whole reason this asks the trader rather than deriving it."""
        pytest.importorskip("agent_stonks.dayrange_model")
        self.stub_forecast(monkeypatch)
        bars = minute_bars()
        # Push the last bar clean through the 210.0 predicted high.
        bars[-1] = {**bars[-1], "h": 215.0, "c": 214.0}

        flat = mo.compute([mo.TRADER_LEVELS_KEY], "AAPL", bars, daily_bars=[],
                          session_date=SESSION,
                          trader_config=self.config(breach_update="off"))["items"]
        assert all(i["kind"] == "level" for i in flat)

        moved = mo.compute([mo.TRADER_LEVELS_KEY], "AAPL", bars, daily_bars=[],
                           session_date=SESSION,
                           trader_config=self.config(breach_update="extreme"))["items"]
        band = next(i for i in moved if i["kind"] == "band")
        assert band["lower"][-1] == pytest.approx(215.0 - 0.75 * 3.0)

    def test_no_config_draws_the_instruments_shipped_levels(self, monkeypatch):
        """A chart with nothing configured shows what the agent would do if
        started now, which is the only answer that is not a guess."""
        from agent_stonks.apple_trader import AppleTraderConfig

        items = self.items(monkeypatch, None)["items"]
        shipped = AppleTraderConfig(ticker="AAPL")
        buy = next(i for i in items if i["label"] == "Buy level")
        assert buy["value"] == pytest.approx(210.0 - shipped.buy_k * 3.0)

    def test_a_config_for_another_symbol_is_not_used_on_this_chart(self, monkeypatch):
        """Its distances were swept on that symbol's tape, and the caption would
        read right over a picture that was wrong."""
        from agent_stonks.apple_trader import AppleTraderConfig

        items = self.items(monkeypatch, self.config(ticker="GOOGL", buy_k=1.5,
                                                    sell_k=0.05))["items"]
        shipped = AppleTraderConfig(ticker="AAPL")
        buy = next(i for i in items if i["label"] == "Buy level")
        assert buy["value"] == pytest.approx(210.0 - shipped.buy_k * 3.0)

    def test_no_forecast_is_a_note_rather_than_an_empty_chart(self, monkeypatch):
        monkeypatch.setattr(mo.apple_models, "load", lambda *a, **k: None)
        result = mo.compute([mo.TRADER_LEVELS_KEY], "AAPL", minute_bars(),
                            daily_bars=[], session_date=SESSION)
        assert result["items"] == []
        assert result["notes"] and "Apple Trader" in result["notes"][0]

    def test_a_session_still_inside_the_opening_window_rests_nothing(self, monkeypatch):
        result = self.items(monkeypatch, self.config(), bars=minute_bars(n=5))
        assert result["items"] == []
        assert "nothing is resting" in result["notes"][0] or "nothing resting" in result["notes"][0]

    def test_it_is_never_auto_selected_by_a_runs_model(self):
        """`for_models` pre-selects what a model said. These are one agent's
        orders, and an Apple Trader 2 run that merely reads the same forecast
        rested nothing of the kind."""
        for key in apple_models.keys():
            assert mo.TRADER_LEVELS_KEY not in mo.for_models([key], "AAPL")["keys"]


class TestLiveOverlays:
    LONG_HISTORY = [{"t": "2026-08-06", "o": 1.0, "h": 1.0, "l": 1.0, "c": 1.0, "v": 1.0}]

    @pytest.fixture(autouse=True)
    def no_network(self, monkeypatch):
        self.fetched = []
        monkeypatch.setattr(
            mo.historical, "fetch_daily_ohlc_bars",
            lambda sym, *a, **k: self.fetched.append(sym) or self.LONG_HISTORY,
        )
        monkeypatch.setattr(mo.historical, "fetch_session_open", lambda *a, **k: 201.5)

    def make_state(self, bars):
        return SimpleNamespace(
            symbol="AAPL", daily_bars=[], bars=bars, model_overlay_cache=None
        )

    def capture_compute(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            mo, "compute",
            lambda *a, **k: calls.append(k) or {"items": [], "notes": []},
        )
        return calls

    def test_the_day_range_forecast_gets_the_traders_long_history(self, monkeypatch):
        """The live buffer's 365-day baseline is ~250 sessions, under the 253
        the forecast requires -- the overlay drew nothing live until the
        forecast was handed the trader's 420-day history instead."""
        calls = self.capture_compute(monkeypatch)
        bars = minute_bars(n=40)
        mo.live_overlays(self.make_state(bars), bars, [mo.DAY_RANGE_KEY])
        assert calls[0]["dayrange_daily_bars"] == self.LONG_HISTORY
        assert calls[0]["daily_bars"] == []
        assert self.fetched == ["AAPL"]

    def test_the_opening_print_is_only_used_for_todays_session(self, monkeypatch):
        calls = self.capture_compute(monkeypatch)
        bars = minute_bars(n=40)  # SESSION is in the past
        mo.live_overlays(self.make_state(bars), bars, [mo.INTRADAY_DAYRANGE_KEY])
        assert calls[0]["open_price"] is None

    def test_overlays_without_the_forecast_do_not_fetch_it(self, monkeypatch):
        calls = self.capture_compute(monkeypatch)
        bars = minute_bars(n=40)
        mo.live_overlays(self.make_state(bars), bars, [mo.PROFILE_RANGE_KEY])
        assert "dayrange_daily_bars" not in calls[0]
        assert self.fetched == []

    def test_compute_hands_the_forecast_its_own_history(self, monkeypatch):
        seen = []
        monkeypatch.setattr(
            mo, "_day_range_forecast",
            lambda symbol, session, daily, *a: seen.append(daily)
            or {"forecast": None, "made_at": None, "problem": "stub"},
        )
        mo.compute([mo.DAY_RANGE_KEY], "AAPL", minute_bars(), daily_bars=[],
                   session_date=SESSION, dayrange_daily_bars=self.LONG_HISTORY)
        assert seen == [self.LONG_HISTORY]

    def test_the_answer_is_cached_until_a_new_bar_arrives(self, monkeypatch):
        calls = []

        def fake_compute(*args, **kwargs):
            calls.append(kwargs)
            return {"items": [], "notes": []}

        monkeypatch.setattr(mo, "compute", fake_compute)
        bars = minute_bars(n=40)
        state = self.make_state(bars)

        mo.live_overlays(state, bars, [mo.DAY_RANGE_KEY])
        mo.live_overlays(state, bars, [mo.DAY_RANGE_KEY])
        assert len(calls) == 1

        mo.live_overlays(state, bars + minute_bars(n=1, base=210.0), [mo.DAY_RANGE_KEY])
        assert len(calls) == 2

    def test_the_agents_configuration_is_part_of_the_cache_key(self, monkeypatch):
        """`trader_levels` is a picture of the sidebar, so moving a distance
        there has to move the lines on the next rerun rather than the next bar."""
        from agent_stonks.apple_trader import AppleTraderConfig

        calls = self.capture_compute(monkeypatch)
        bars = minute_bars(n=40)
        state = self.make_state(bars)
        state.app = SimpleNamespace(apple_trader_config=AppleTraderConfig(buy_k=0.5))

        mo.live_overlays(state, bars, [mo.TRADER_LEVELS_KEY])
        mo.live_overlays(state, bars, [mo.TRADER_LEVELS_KEY])
        assert len(calls) == 1                     # same bars, same config
        assert calls[0]["trader_config"].buy_k == 0.5

        state.app.apple_trader_config = AppleTraderConfig(buy_k=0.9)
        mo.live_overlays(state, bars, [mo.TRADER_LEVELS_KEY])
        assert len(calls) == 2
        assert calls[1]["trader_config"].buy_k == 0.9

    def test_no_apple_trader_selected_passes_no_configuration(self, monkeypatch):
        """Another personality means the form is not rendered; the overlay then
        falls back to the instrument's shipped levels rather than the last
        symbol's numbers."""
        calls = self.capture_compute(monkeypatch)
        bars = minute_bars(n=40)
        mo.live_overlays(self.make_state(bars), bars, [mo.TRADER_LEVELS_KEY])
        assert calls[0]["trader_config"] is None

    def test_changing_the_selection_recomputes(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            mo, "compute", lambda *a, **k: calls.append(1) or {"items": [], "notes": []}
        )
        bars = minute_bars(n=40)
        state = self.make_state(bars)
        mo.live_overlays(state, bars, [mo.DAY_RANGE_KEY])
        mo.live_overlays(state, bars, [mo.DAY_RANGE_KEY, mo.PROFILE_RANGE_KEY])
        assert len(calls) == 2

    def test_nothing_selected_does_not_touch_the_models(self, monkeypatch):
        monkeypatch.setattr(
            mo, "compute", lambda *a, **k: pytest.fail("should not be called")
        )
        state = self.make_state(minute_bars(n=40))
        assert mo.live_overlays(state, state.bars, []) == {"items": [], "notes": []}

    def test_session_date_of_reads_the_last_bars_exchange_local_day(self):
        assert mo.session_date_of(minute_bars(n=3)).date() == pd.Timestamp(SESSION).date()

    def test_session_date_of_no_bars_is_none(self):
        assert mo.session_date_of([]) is None


# --- the renderer -----------------------------------------------------------

BARS = minute_bars(n=60)


def level(value=200.0, color="#22d3ee"):
    return {"kind": "level", "key": "k", "label": "Pred. high", "value": value,
            "color": color, "dash": "dash", "note": "why"}


def band(y0=199.0, y1=201.0):
    return {"kind": "span", "key": "k", "label": "Range", "x0": BARS[5]["t"],
            "x1": BARS[40]["t"], "y0": y0, "y1": y1, "color": "#22d3ee",
            "note": "why", "forward": False}


def window(forward=True):
    return {"kind": "span", "key": "m", "label": "Holds", "x0": BARS[50]["t"],
            "x1": (pd.Timestamp(BARS[59]["t"]) + pd.Timedelta(minutes=10)).isoformat(),
            "y0": None, "y1": None, "color": "#26c6a2", "note": "why",
            "forward": forward}


def moment():
    return {"kind": "event", "key": "m",
            "group": "Moments",
            "label": "→ positive 90%", "ts": BARS[50]["t"],
            "color": "#26c6a2", "icon": "▲", "price": 200.0, "dash": "dot",
            "note": "why", "forward": False}


def envelope(start=0, stop=60, extra_minutes=0):
    """A band over BARS[start:stop], optionally running on past the last bar."""
    stamps = [pd.Timestamp(b["t"]) for b in BARS[start:stop]]
    stamps += [stamps[-1] + pd.Timedelta(minutes=m + 1) for m in range(extra_minutes)]
    mid = np.linspace(1.0, 0.3, len(stamps))
    return {"kind": "band", "key": "iv", "label": "Pred. intraday range",
            "group": "Predicted intraday range",
            "t": [s.isoformat() for s in stamps],
            "upper": list(200.0 + 2.0 * mid), "lower": list(200.0 - 1.5 * mid),
            "color": "#facc15", "dash": "dot", "note": "why", "forward": False}


class TestRenderer:
    def chart(self, items):
        return build_chart(BARS, [], [], "AAPL", SESSION_START, model_overlays=items)

    def test_a_level_is_drawn_in_the_price_panel_and_in_the_profile(self):
        fig = self.chart([level()])
        refs = {(s.type, s.xref, s.yref) for s in fig.layout.shapes}
        assert ("line", "x domain", "y") in refs, "the candle panel"
        assert ("line", "x2 domain", "y2") in refs, "the price profile beside it"

    def test_a_level_bound_to_a_session_is_drawn_only_over_it(self):
        """SimLab replays several days into one figure.

        Five predicted highs drawn across all five days would each claim to be
        about days they were not, so a level that names its session is a
        segment rather than a full-width line.
        """
        bounded = {**level(), "x0": BARS[10]["t"], "x1": BARS[30]["t"]}
        fig = self.chart([bounded])
        segment = next(
            s for s in fig.layout.shapes
            if s.type == "line" and s.xref == "x" and s.line.color == "#22d3ee"
        )
        assert pd.Timestamp(segment.x0) == pd.Timestamp(BARS[10]["t"])
        assert pd.Timestamp(segment.x1) == pd.Timestamp(BARS[30]["t"])
        assert segment.y0 == segment.y1 == 200.0
        # Still mirrored across the whole profile panel, which has no time axis.
        assert any(s.xref == "x2 domain" for s in fig.layout.shapes)

    def test_a_level_whose_session_is_off_screen_is_dropped(self):
        elsewhere = {**level(), "x0": f"{SESSION}T10:00:00+00:00",
                     "x1": f"{SESSION}T11:00:00+00:00"}
        fig = self.chart([elsewhere])
        assert not [
            s for s in fig.layout.shapes
            if s.type == "line" and s.xref == "x" and s.line.color == "#22d3ee"
        ]

    def test_a_level_is_labelled_with_its_value(self):
        fig = self.chart([level(value=207.25)])
        assert any("207.25" in a["text"] for a in fig.layout.annotations)

    def test_a_price_bounded_span_is_a_semi_transparent_band_in_both_panels(self):
        fig = self.chart([band()])
        rects = [s for s in fig.layout.shapes if s.type == "rect"]
        assert len(rects) == 2
        for rect in rects:
            assert rect.fillcolor.startswith("rgba(")
            assert float(rect.fillcolor.rsplit(",", 1)[1].rstrip(")")) < 0.2
            assert rect.line.width == 0

    def test_a_time_only_span_is_a_full_height_column_over_price_and_volume(self):
        fig = self.chart([window()])
        rects = [s for s in fig.layout.shapes if s.type == "rect"]
        assert {r.yref for r in rects} == {"y domain", "y3 domain"}

    def test_a_span_is_drawn_behind_the_candles(self):
        fig = self.chart([band(), window()])
        assert all(s.layer == "below" for s in fig.layout.shapes if s.type == "rect")

    def test_an_event_without_a_line_is_a_marker_only(self):
        fig = self.chart([{**moment(), "line": False}])
        assert not [
            s for s in fig.layout.shapes
            if s.type == "line" and s.yref == "y domain" and s.line.color == "#26c6a2"
        ]
        assert [t for t in fig.data if getattr(t, "mode", None) == "markers+text"]

    def test_an_event_is_a_vertical_line_plus_a_marker_that_explains_itself(self):
        fig = self.chart([moment()])
        assert any(
            s.type == "line" and s.yref == "y domain" for s in fig.layout.shapes
        )
        trace = next(
            t for t in fig.data if getattr(t, "mode", None) == "markers+text"
        )
        assert list(trace.text) == ["▲"]
        assert trace.customdata[0][1] == "why"

    def test_events_of_one_overlay_share_a_single_legend_entry(self):
        second = {**moment(), "ts": BARS[55]["t"], "label": "→ negative"}
        fig = self.chart([moment(), second])
        marks = [t for t in fig.data if getattr(t, "mode", None) == "markers+text"]
        assert len(marks) == 1
        assert len(marks[0].x) == 2
        # Named for the overlay, not for whichever moment happened to be first.
        assert marks[0].name == "Moments"

    def test_a_forward_span_widens_the_time_axis_to_show_it(self):
        fig = self.chart([window(forward=True)])
        assert pd.Timestamp(fig.layout.xaxis.range[1]) > pd.Timestamp(BARS[-1]["t"])

    def test_a_span_about_the_rest_of_the_day_does_not_widen_the_axis(self):
        """A day-range forecast claims something about 16:00 at 09:35.

        Letting that stretch the axis would squash an hour of tape into a
        corner, so a non-forward span is clipped to the bars in hand instead.
        """
        far = {**band(), "x1": f"{SESSION}T20:00:00+00:00"}
        fig = self.chart([far])
        assert pd.Timestamp(fig.layout.xaxis.range[1]) == pd.Timestamp(BARS[-1]["t"])

    def test_a_span_entirely_outside_the_view_is_dropped(self):
        stale = {**band(), "x0": f"{SESSION}T10:00:00+00:00",
                 "x1": f"{SESSION}T11:00:00+00:00"}
        fig = self.chart([stale])
        assert not [s for s in fig.layout.shapes if s.type == "rect"]

    def test_no_overlays_changes_nothing(self):
        plain = build_chart(BARS, [], [], "AAPL", SESSION_START)
        empty = build_chart(BARS, [], [], "AAPL", SESSION_START, model_overlays=[])
        assert len(empty.data) == len(plain.data)
        assert len(empty.layout.shapes) == len(plain.layout.shapes)
        assert empty.layout.xaxis.range == plain.layout.xaxis.range

    def test_the_renderer_works_on_a_plain_figure_without_a_profile_panel(self):
        """SimLab's chart is one candlestick trace, not a 2x2 grid."""
        fig = go.Figure(
            go.Candlestick(
                x=[b["t"] for b in BARS],
                open=[b["o"] for b in BARS], high=[b["h"] for b in BARS],
                low=[b["l"] for b in BARS], close=[b["c"] for b in BARS],
            )
        )
        add_model_overlays(
            [level(), band(), window(), moment()], fig,
            pd.Timestamp(BARS[0]["t"]), pd.Timestamp(BARS[-1]["t"]),
            row=None, col=None,
        )
        assert len(fig.layout.shapes) == 4  # level, band, window, event line
        assert [t.name for t in fig.data if getattr(t, "mode", None) == "markers+text"]

    def test_a_band_is_two_edges_with_the_range_between_them_tinted(self):
        fig = self.chart([envelope()])
        edges = [t for t in fig.data if t.legendgroup == "iv"]
        assert len(edges) == 2
        upper, lower = edges
        assert upper.fill in (None, "none") and lower.fill == "tonexty"
        assert float(lower.fillcolor.rsplit(",", 1)[1].rstrip(")")) < 0.2

    def test_a_band_is_drawn_behind_the_candles(self):
        """The live chart's candle bodies are a bar trace named for the symbol."""
        fig = self.chart([envelope()])
        candles = next(i for i, t in enumerate(fig.data) if t.name == "AAPL")
        band_at = [i for i, t in enumerate(fig.data) if t.legendgroup == "iv"]
        assert max(band_at) < candles
        assert band_at == [band_at[0], band_at[0] + 1]  # the fill needs them adjacent

    def test_bands_of_one_overlay_share_a_single_legend_entry(self):
        """A replay draws one band per session; the legend names the overlay once."""
        first, second = envelope(0, 30), envelope(30, 60)
        fig = self.chart([first, second])
        edges = [t for t in fig.data if t.legendgroup == "iv"]
        assert len(edges) == 4
        assert sum(bool(t.showlegend) for t in edges) == 1

    def test_a_band_is_clipped_to_the_bars_in_hand(self):
        """Like a day-range span: a curve about 16:00 must not widen a live chart."""
        fig = self.chart([envelope(0, 60, extra_minutes=90)])
        upper = next(t for t in fig.data if t.legendgroup == "iv")
        # Plotly keeps datetimes as naive UTC wall clock, as it does the candles'.
        last_bar = pd.Timestamp(BARS[-1]["t"]).tz_convert("UTC").tz_localize(None)
        assert pd.Timestamp(max(upper.x)) <= last_bar
        assert pd.Timestamp(fig.layout.xaxis.range[1]) == pd.Timestamp(BARS[-1]["t"])

    def test_a_band_draws_on_simlabs_plain_figure(self):
        fig = go.Figure(
            go.Candlestick(
                x=[b["t"] for b in BARS],
                open=[b["o"] for b in BARS], high=[b["h"] for b in BARS],
                low=[b["l"] for b in BARS], close=[b["c"] for b in BARS],
            )
        )
        add_model_overlays([envelope()], fig, pd.Timestamp(BARS[0]["t"]),
                           pd.Timestamp(BARS[-1]["t"]), row=None, col=None)
        assert [t.type for t in fig.data] == ["scatter", "scatter", "candlestick"]

    def test_overlay_x_max_ignores_items_that_do_not_reach_forward(self):
        last = pd.Timestamp(BARS[-1]["t"])
        assert overlay_x_max([band(), level(), moment()], last) == last

    def test_overlay_x_max_handles_a_naive_reference(self):
        naive = pd.Timestamp(BARS[-1]["t"]).tz_localize(None)
        assert overlay_x_max([window()], naive).tzinfo is None
