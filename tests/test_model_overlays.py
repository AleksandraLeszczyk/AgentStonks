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


@pytest.fixture(autouse=True)
def _no_days_off(monkeypatch, request):
    """SESSION -- TimeToChange3's held-out day once, which the recorded
    forecasts here were pinned to -- is a jobs-report day, which every default
    configuration sits out. These tests are about how levels are drawn, so the
    calendar is emptied for them; `TestTraderLevelsOnDaysOff` reads it."""
    if request.node.cls is not None and request.node.cls.__name__ == "TestTraderLevelsOnDaysOff":
        return
    monkeypatch.setattr(mo.event_days, "load_calendar", lambda *a, **k: [])


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
        for symbol in apple_models.HIGHLOW_TICKERS:
            assert mo.OVERLAYS[mo.HIGHLOW_RANGE_KEY].covers(symbol), symbol
        assert not mo.OVERLAYS[mo.HIGHLOW_RANGE_KEY].covers("MSFT")
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
        assert mo.for_models([apple_models.HIGHLOW_KEY], "AAPL")["keys"] == [
            mo.HIGHLOW_RANGE_KEY
        ]

    def test_an_overlay_is_never_selected_for_a_symbol_it_has_no_model_for(self):
        """A rule run's stored symbol list can be wider than the one instrument
        it traded, so this is asked about tabs the model was never fitted on --
        where the picker has no such option to select."""
        assert mo.for_models([apple_models.HIGHLOW_KEY], "MSFT")["keys"] == []

    def test_no_symbol_asks_the_question_without_one(self):
        """The default is "which overlays draw these models", not "none" --
        filtering is what naming a symbol adds."""
        assert mo.for_models([apple_models.HIGHLOW_KEY])["keys"] == [
            mo.HIGHLOW_RANGE_KEY
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
        result = mo.compute([mo.HIGHLOW_RANGE_KEY], "MSFT", minute_bars())
        assert result["items"] == []
        assert "MSFT" in result["notes"][0]
        assert "fitted on" in result["notes"][0]

    def test_a_missing_bundle_is_a_note_not_an_exception(self, monkeypatch):
        monkeypatch.setattr(mo.apple_models, "load", lambda *a, **k: None)
        result = mo.compute([mo.HIGHLOW_RANGE_KEY], "AAPL", minute_bars())
        assert result["items"] == []
        assert result["notes"] and "Predicted day range" in result["notes"][0]

    def test_a_model_that_raises_is_a_note_not_an_exception(self, monkeypatch):
        def boom(*_a, **_k):
            raise RuntimeError("the checkpoint is corrupt")

        monkeypatch.setattr(mo.apple_models, "load", boom)
        result = mo.compute([mo.HIGHLOW_RANGE_KEY], "AAPL", minute_bars())
        assert result["items"] == []
        assert "the checkpoint is corrupt" in result["notes"][0]

    def test_bars_from_another_day_produce_a_note(self):
        result = mo.compute([mo.HIGHLOW_RANGE_KEY], "AAPL", minute_bars(),
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
    """A day-range forecast drawn: flat while the session stays inside it, a
    stepped band once it trades outside. On HighLow's overlay; HighLow2's is
    drawn by the same code."""

    def stub_forecast(self, monkeypatch, high=210.0, low=198.0):
        hl = pytest.importorskip("agent_stonks.highlow_model")

        monkeypatch.setattr(mo.apple_models, "load", lambda *a, **k: {"stub": True})
        monkeypatch.setattr(hl, "opening_minutes", lambda bundle=None: 5)
        monkeypatch.setattr(
            hl, "forecast_session",
            lambda *a, **k: {
                "pred_high": high, "pred_low": low, "prev_avg": 200.0,
                "adr14_abs": 3.0, "or_high": 201.0, "or_low": 199.0,
            },
        )

    def test_produces_two_levels_and_the_band_between_them(self, monkeypatch):
        """A session that stays inside its forecast leaves it flat all day."""
        self.stub_forecast(monkeypatch, low=196.0)   # the tape bottoms at 196.80
        items = mo.compute([mo.HIGHLOW_RANGE_KEY], "AAPL", minute_bars(),
                           daily_bars=[], session_date=SESSION)["items"]
        levels = {i["label"]: i["value"] for i in items if i["kind"] == "level"}
        assert levels == {"Pred. high": 210.0, "Pred. low": 196.0}
        band = next(i for i in items if i["kind"] == "span")
        assert (band["y0"], band["y1"]) == (196.0, 210.0)

    def test_the_band_runs_from_the_forecast_to_the_closing_bell(self, monkeypatch):
        self.stub_forecast(monkeypatch, low=196.0)
        items = mo.compute([mo.HIGHLOW_RANGE_KEY], "AAPL", minute_bars(),
                           daily_bars=[], session_date=SESSION)["items"]
        band = next(i for i in items if i["kind"] == "span")
        close = pd.Timestamp(band["x1"]).tz_convert("America/New_York")
        assert (close.hour, close.minute) == (16, 0)
        # The 5-minute opening window ends on the 09:34 bar.
        start = pd.Timestamp(band["x0"]).tz_convert("America/New_York")
        assert (start.hour, start.minute) == (9, 34)

    def test_a_breached_forecast_is_drawn_as_a_stepped_band(self, monkeypatch):
        """The tape bottoms at 196.80, under the 198.00 predicted low, so the
        low is revised and the flat pair becomes two curves -- and under
        "move to the extreme so far" the high follows it down."""
        from agent_stonks.apple_trader import AppleTraderConfig

        self.stub_forecast(monkeypatch)
        items = mo.compute([mo.HIGHLOW_RANGE_KEY], "AAPL", minute_bars(),
                           daily_bars=[], session_date=SESSION,
                           trader_config=AppleTraderConfig(
                               ticker="AAPL", breach_update="shift"))["items"]
        assert [i["kind"] for i in items] == ["band"]
        band = items[0]
        assert band["label"] == "Pred. high / low"
        assert len(band["t"]) == len(band["lower"]) == len(band["upper"])
        # Starts at the 9:35 forecast and ends at the session's own low.
        assert band["lower"][0] == pytest.approx(198.0)
        assert band["lower"][-1] == pytest.approx(196.80, abs=0.01)
        # The high was never traded through, but moves down with the low by
        # the same $1.20, keeping the forecast's width.
        assert band["upper"][0] == pytest.approx(210.0)
        assert band["upper"][-1] == pytest.approx(208.80, abs=0.01)
        assert band["upper"][-1] - band["lower"][-1] == pytest.approx(12.0)

    def test_the_default_brownian_revision_ends_at_the_last_bar(self, monkeypatch):
        """Under the default Brownian update the breached low goes past the
        session's own 196.80 by the reach still to come and the high follows at
        the forecast's width -- and nothing is drawn past the newest bar."""
        self.stub_forecast(monkeypatch)
        bars = minute_bars()
        band = mo.compute([mo.HIGHLOW_RANGE_KEY], "AAPL", bars,
                          daily_bars=[], session_date=SESSION)["items"][0]
        assert band["kind"] == "band" and not band["forward"]
        assert band["lower"][-1] < 196.80
        assert band["upper"][-1] - band["lower"][-1] == pytest.approx(12.0)
        assert pd.Timestamp(band["t"][-1]) == pd.Timestamp(bars[-1]["t"])

    def test_a_breach_on_the_first_bar_still_shows_the_935_forecast_before_it(
        self, monkeypatch
    ):
        """The walk reads each bar before recording it, so a breach on the very
        first bar after the forecast used to be in every row: the band came out
        flat at the revised high, as if that had been the forecast all along."""
        self.stub_forecast(monkeypatch, low=196.0)
        bars = minute_bars()
        bars[5] = {**bars[5], "h": 215.0}    # the 09:35 bar, first after the window
        band = mo.compute([mo.HIGHLOW_RANGE_KEY], "AAPL", bars,
                          daily_bars=[], session_date=SESSION)["items"][0]
        assert band["kind"] == "band" and band["step"]
        start = pd.Timestamp(band["t"][0]).tz_convert("America/New_York")
        assert (start.hour, start.minute) == (9, 34)
        assert band["upper"][0] == pytest.approx(210.0)
        assert band["upper"][1] > 210.0

    def test_the_low_only_steps_down_and_never_back_up(self, monkeypatch):
        """Piecewise, not a curve fitted to the tape: each step holds until the
        session trades through it again."""
        self.stub_forecast(monkeypatch)
        band = mo.compute([mo.HIGHLOW_RANGE_KEY], "AAPL", minute_bars(),
                          daily_bars=[], session_date=SESSION)["items"][0]
        lows = band["lower"]
        assert all(b <= a for a, b in zip(lows, lows[1:]))
        assert len(set(lows)) > 1          # it really did step

    def test_the_drawn_high_is_the_one_the_trader_levels_hang_off(
        self, monkeypatch
    ):
        """The reason both overlays share one walk. A chart showing the buy and
        sell stepping up under a predicted high that stood still would have the
        model's own line contradicting the orders derived from it."""
        self.stub_forecast(monkeypatch)
        bars = minute_bars()
        bars[-1] = {**bars[-1], "h": 215.0, "c": 214.0}
        items = mo.compute([mo.HIGHLOW_RANGE_KEY, mo.TRADER_LEVELS_KEY], "AAPL", bars,
                           daily_bars=[], session_date=SESSION)["items"]
        ranges = next(i for i in items if i["label"] == "Pred. high / low")
        levels = next(i for i in items if i["label"] == "Buy/sell levels")
        from agent_stonks.apple_trader import AppleTraderConfig

        shipped = AppleTraderConfig(ticker="AAPL")
        # sell = pred_high - sell_k x unit, on the same bar, from the same walk.
        unit = ranges["upper"][-1] - ranges["lower"][-1]
        assert levels["upper"][-1] == pytest.approx(
            ranges["upper"][-1] - shipped.sell_k * unit
        )

    def test_bars_that_miss_the_open_refuse_rather_than_forecast(self, monkeypatch):
        self.stub_forecast(monkeypatch)
        late = minute_bars()[30:]
        result = mo.compute([mo.HIGHLOW_RANGE_KEY], "AAPL", late,
                            daily_bars=[], session_date=SESSION)
        assert result["items"] == []
        assert "09:30 open" in result["notes"][0]

    def test_fewer_bars_than_the_opening_window_is_a_note(self, monkeypatch):
        self.stub_forecast(monkeypatch)
        result = mo.compute([mo.HIGHLOW_RANGE_KEY], "AAPL", minute_bars(n=3),
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
        from agent_stonks.config import UNIT_ADR

        kwargs.setdefault("ticker", "AAPL")
        kwargs.setdefault("buy_k", 0.75)
        kwargs.setdefault("sell_k", 0.10)
        kwargs.setdefault("breach_update", "off")
        # Pinned with the rest: the numbers in this class are ADR arithmetic
        # off the $3 ADR above. `test_the_unit_is_the_agents_own` is where the
        # other unit is switched on.
        kwargs.setdefault("level_unit", UNIT_ADR)
        # And containment off, so "off" still means levels that never move --
        # `test_containment_makes_off_draw_a_band` is where it is switched on.
        kwargs.setdefault("contain_range", False)
        kwargs.setdefault("breach_exit", False)
        # And no stop, so the buy/sell tests see two lines -- the stop tests
        # switch it on.
        kwargs.setdefault("stop_gain_fraction", 0.0)
        return AppleTraderConfig(**kwargs)

    def first_buy(self, items):
        """The chart's opening buy level, however the overlay chose to draw it.

        A config whose levels never move is drawn as two flat lines and one
        whose levels move as a band, and which of those the shipped settings
        produce is not what these two tests are about."""
        level = [i for i in items if i.get("label") == "Buy level"]
        if level:
            return level[0]["value"]
        return next(i for i in items if i["kind"] == "band")["lower"][0]

    def shipped_unit(self):
        """What one k is worth under the shipped config, on the stubbed forecast.

        Read through `level_unit` rather than written out, so that these two
        stay tests of *which config the chart falls back to* and do not have to
        be re-derived every time the shipped unit changes."""
        from agent_stonks.apple_trader import AppleTraderConfig, level_unit

        return level_unit(
            AppleTraderConfig(ticker="AAPL"),
            {"adr14_abs": 3.0, "pred_high": 210.0, "pred_low": 198.0},
        )

    def items(self, monkeypatch, config=None, bars=None):
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

    def test_no_stop_is_drawn_before_a_buy(self, monkeypatch):
        """The stop hangs under the actual fill, and a walk fills nothing --
        whatever the stop is written in."""
        for config in (self.config(stop_gain_fraction=0.5),
                       self.config(stop_gain_fraction=0.0, stop_k=0.2)):
            items = self.items(monkeypatch, config)["items"]
            assert not [i for i in items if i["label"] == "Stop level"]
            assert {i["label"] for i in items} == {"Buy level", "Sell level"}

    def test_moving_levels_carry_no_stop_either(self, monkeypatch):
        self.stub_forecast(monkeypatch)
        bars = minute_bars()
        bars[-1] = {**bars[-1], "h": 215.0, "c": 214.0}
        items = mo.compute(
            [mo.TRADER_LEVELS_KEY], "AAPL", bars, daily_bars=[], session_date=SESSION,
            trader_config=self.config(breach_update="extreme", stop_gain_fraction=0.5),
        )["items"]
        assert [i["kind"] for i in items] == ["band"]

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

    def test_a_breach_moves_the_levels_and_switches_it_to_a_band(self, monkeypatch):
        """The same settings that move the agent's orders move the drawing --
        which is the whole reason this asks the trader rather than deriving it."""
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

    def test_a_breach_on_the_first_bar_steps_the_levels_at_that_bar(self, monkeypatch):
        """The live case that drew two flat lines: the tape went through the
        predicted high on the bar straight after the forecast, so every walked
        row already held the revised levels. The forecast's own levels lead the
        band now, and the step lands on the bar that breached."""
        bars = minute_bars()
        bars[5] = {**bars[5], "h": 215.0}    # the 09:35 bar, first after the window
        items = self.items(monkeypatch, self.config(breach_update="extreme"),
                           bars=bars)["items"]
        band = next(i for i in items if i["kind"] == "band")
        assert band["step"]
        assert band["lower"][0] == pytest.approx(207.75)     # the 09:35 buy
        assert band["lower"][1] == pytest.approx(215.0 - 0.75 * 3.0)
        stamps = [pd.Timestamp(t).tz_convert("America/New_York") for t in band["t"][:2]]
        assert [(t.hour, t.minute) for t in stamps] == [(9, 34), (9, 35)]

    def test_no_config_draws_the_instruments_shipped_levels(self, monkeypatch):
        """A chart with nothing configured shows what the agent would do if
        started now, which is the only answer that is not a guess."""
        from agent_stonks.apple_trader import AppleTraderConfig

        items = self.items(monkeypatch, None)["items"]
        shipped = AppleTraderConfig(ticker="AAPL")
        assert self.first_buy(items) == pytest.approx(
            210.0 - shipped.buy_k * self.shipped_unit()
        )

    def test_a_config_for_another_symbol_is_not_used_on_this_chart(self, monkeypatch):
        """Its distances were swept on that symbol's tape, and the caption would
        read right over a picture that was wrong."""
        from agent_stonks.apple_trader import AppleTraderConfig

        items = self.items(monkeypatch, self.config(ticker="GOOGL", buy_k=1.5,
                                                    sell_k=0.05))["items"]
        shipped = AppleTraderConfig(ticker="AAPL")
        assert self.first_buy(items) == pytest.approx(
            210.0 - shipped.buy_k * self.shipped_unit()
        )

    def test_containment_makes_off_draw_a_band(self, monkeypatch):
        """The chart asks the trader, so a rule that moves the levels under
        "off" has to move the drawing under "off" too -- otherwise the picture
        shows orders resting where the agent is no longer resting them."""
        self.stub_forecast(monkeypatch)
        bars = minute_bars()
        bars[-1] = {**bars[-1], "h": 215.0, "c": 214.0}

        held = mo.compute([mo.TRADER_LEVELS_KEY], "AAPL", bars, daily_bars=[],
                          session_date=SESSION,
                          trader_config=self.config(breach_update="off"))["items"]
        assert all(i["kind"] == "level" for i in held)

        contained = mo.compute(
            [mo.TRADER_LEVELS_KEY], "AAPL", bars, daily_bars=[], session_date=SESSION,
            trader_config=self.config(breach_update="off", contain_range=True),
        )["items"]
        band = next(i for i in contained if i["kind"] == "band")
        # The high is pinned to the 215.0 that printed, so the sell curve ends
        # exactly `sell_k x ADR` under it rather than under the stale 210.0.
        assert band["upper"][-1] == pytest.approx(215.0 - 0.10 * 3.0)

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
        orders, not a forecast of the day."""
        for key in apple_models.keys():
            assert mo.TRADER_LEVELS_KEY not in mo.for_models([key], "AAPL")["keys"]

    def test_a_live_chart_with_no_agent_running_draws_them_gray(self, monkeypatch):
        """Not started, the walk is where the agent would rest its orders,
        not orders in the book. SimLab's walk (no `trader_running`) is the run
        being replayed, and keeps the agent's orange."""
        from agent_stonks.config import MODEL_OVERLAY_COLORS

        self.stub_forecast(monkeypatch)

        def colors(running):
            items = mo.compute(
                [mo.TRADER_LEVELS_KEY], "AAPL", minute_bars(), daily_bars=[],
                session_date=SESSION, trader_config=self.config(), trader_running=running,
            )["items"]
            assert {i["label"] for i in items} == {"Buy level", "Sell level"}
            return {i["color"] for i in items}

        assert colors(False) == {MODEL_OVERLAY_COLORS["trader_idle"]}
        assert colors(None) == {MODEL_OVERLAY_COLORS["trader_levels"]}


class TestTraderLevelsOnDaysOff:
    """A session the agent sits out draws no levels, and says why -- not a walk
    of orders it would never rest."""

    def test_a_jobs_report_day_rests_nothing_and_says_so(self, monkeypatch):
        monkeypatch.setattr(mo, "_highlow_forecast", lambda *a, **k: pytest.fail("forecast"))
        out = mo.compute([mo.TRADER_LEVELS_KEY], "AAPL", minute_bars(), daily_bars=[],
                         session_date=SESSION)
        assert out["items"] == []
        assert "sits this session out" in out["notes"][0] and "Jobs report" in out["notes"][0]

    def test_a_run_that_trades_every_day_still_draws_its_levels(self, monkeypatch):
        from agent_stonks.apple_trader import AppleTraderConfig

        sat_out = mo._sat_out("AAPL", pd.Timestamp(SESSION), AppleTraderConfig(skip_events=()))
        assert sat_out is None
        assert mo._sat_out("AAPL", pd.Timestamp(SESSION), AppleTraderConfig(skip_events=("cpi",))) is None


class TestRecordedTraderLevels:
    """A live run's own record of its levels is what the chart draws, so
    sidebar edits show at the minute they were adopted and the numbers are
    the ones the agent's log quotes."""

    def history(self, buys, sells=None, highs=None, day=SESSION, ticker="AAPL",
                model_key="highlow"):
        from agent_stonks.apple_trader import AppleTraderConfig

        start = pd.Timestamp(f"{day} 09:34", tz="America/New_York")
        sells = sells or [b + 2.0 for b in buys]
        highs = highs or [210.0] * len(buys)
        rows = [
            {"t": start + pd.Timedelta(minutes=i), "buy": b, "sell": s, "stop": b - 1.0,
             "reference": h, "pred_high": h, "pred_low": 198.0}
            for i, (b, s, h) in enumerate(zip(buys, sells, highs))
        ]
        return {"ticker": ticker, "date": pd.Timestamp(day),
                "config": AppleTraderConfig(ticker=ticker, model_key=model_key),
                "rows": rows}

    def compute(self, keys, history, monkeypatch):
        # The walk and the forecast must not be needed: the record replaces both.
        monkeypatch.setattr(mo, "_highlow_forecast", lambda *a, **k: pytest.fail("forecast"))
        return mo.compute(keys, "AAPL", minute_bars(), daily_bars=[],
                          session_date=SESSION, trader_history=history)["items"]

    def test_a_stopped_runs_forecast_carries_on_to_the_newest_bar(self, monkeypatch):
        """A run stopped at 09:36 recorded three rows. The prediction is about
        the session so far, so the bars after them are walked from the run's
        own forecast (its seed) -- the low revised at the bar that first traded
        under it, and nothing drawn past the newest bar."""
        bars = minute_bars()
        history = self.history([207.0] * 3, highs=[210.0] * 3)
        history["seed"] = {
            "forecast": {"pred_high": 210.0, "pred_low": 198.0, "prev_avg": 200.0,
                         "adr14_abs": 3.0, "or_high": 201.0, "or_low": 199.0},
            "opening_end": pd.Timestamp(f"{SESSION} 09:34", tz="America/New_York"),
            "open_price": bars[0]["o"],
        }
        band = next(i for i in self.compute([mo.HIGHLOW_RANGE_KEY], history, monkeypatch)
                    if i["kind"] == "band")
        stamps = [pd.Timestamp(t) for t in band["t"]]
        assert stamps[-1] == pd.Timestamp(bars[-1]["t"])
        assert band["upper"][:3] == [210.0, 210.0, 210.0]
        first_under = next(pd.Timestamp(b["t"]) for b in bars if b["l"] < 198.0)
        at = stamps.index(first_under)
        assert band["lower"][at - 1] == pytest.approx(198.0)
        assert band["lower"][at] < 198.0

    def test_a_stopped_runs_range_carries_on_from_where_the_run_left_it(self, monkeypatch):
        """Not from a walk of the day again: by now the buffer has swapped the
        09:36 bar for SIP's, $1 under the $198 low the run held through 09:40,
        and walked again from 09:35 that is a breach -- the band would drop at
        09:41 with nothing traded near it."""
        bars = minute_bars(n=40)
        assert min(b["l"] for b in bars) > 198.0
        bars[6]["l"] = 197.0
        history = self.history([207.0] * 7, highs=[210.0] * 7)
        history["seed"] = {
            "forecast": {"pred_high": 210.0, "pred_low": 198.0, "prev_avg": 200.0,
                         "adr14_abs": 3.0, "or_high": 201.0, "or_low": 199.0},
            "opening_end": pd.Timestamp(f"{SESSION} 09:34", tz="America/New_York"),
            "open_price": bars[0]["o"],
        }
        monkeypatch.setattr(mo, "_highlow_forecast", lambda *a, **k: pytest.fail("forecast"))
        items = mo.compute([mo.HIGHLOW_RANGE_KEY], "AAPL", bars, daily_bars=[],
                           session_date=SESSION, trader_history=history)["items"]
        # Held all session, so drawn flat: two levels, no band.
        assert not [i for i in items if i["kind"] == "band"]
        levels = {i["label"]: i["value"] for i in items if i["kind"] == "level"}
        assert levels == {"Pred. high": 210.0, "Pred. low": 198.0}

    def stopped(self, bars):
        """A run stopped at 09:36: three rows from 09:34, and the seed its
        levels can be carried on from."""
        history = self.history([207.0] * 3, highs=[210.0] * 3)
        history["seed"] = {
            "forecast": {"pred_high": 210.0, "pred_low": 198.0, "prev_avg": 200.0,
                         "adr14_abs": 3.0, "or_high": 201.0, "or_low": 199.0},
            "opening_end": pd.Timestamp(f"{SESSION} 09:34", tz="America/New_York"),
            "open_price": bars[0]["o"],
        }
        return history

    @staticmethod
    def stretch(item) -> "tuple[pd.Timestamp, pd.Timestamp]":
        start, end = (item["x0"], item["x1"]) if item["kind"] == "level" \
            else (item["t"][0], item["t"][-1])
        return pd.Timestamp(start), pd.Timestamp(end)

    def test_a_stopped_runs_levels_carry_on_gray_to_the_newest_bar(self, monkeypatch):
        """Orange while it ran, then the levels it would have kept resting --
        carried on from its record like the range -- gray from its last row to
        the newest bar."""
        from agent_stonks.config import MODEL_OVERLAY_COLORS

        bars = minute_bars()
        monkeypatch.setattr(mo, "_highlow_forecast", lambda *a, **k: pytest.fail("forecast"))
        items = mo.compute([mo.TRADER_LEVELS_KEY], "AAPL", bars, daily_bars=[],
                           session_date=SESSION, trader_history=self.stopped(bars),
                           trader_running=False)["items"]
        stopped_at = pd.Timestamp(f"{SESSION} 09:36", tz="America/New_York")
        ran = [i for i in items if i["color"] == MODEL_OVERLAY_COLORS["trader_levels"]]
        idle = [i for i in items if i["color"] == MODEL_OVERLAY_COLORS["trader_idle"]]
        assert {i["label"] for i in ran} == {"Buy level", "Sell level"}
        assert all(self.stretch(i)[1] == stopped_at for i in ran)
        assert idle and all(self.stretch(i)[0] == stopped_at for i in idle)
        band = next((i for i in idle if i["kind"] == "band"), None)
        if band is not None:
            assert band["lower"][0] == 207.0
            assert pd.Timestamp(band["t"][-1]) == pd.Timestamp(bars[-1]["t"])
        # The stop is the run's, while it ran.
        stop = next(i for i in items if i["label"] == "Stop level")
        assert self.stretch(stop)[1] == stopped_at

    def test_a_running_agents_levels_are_its_record_alone(self, monkeypatch):
        """Its record is up to the bar it last read; the live buffer's newer,
        still-forming bar is not a minute it was idle."""
        from agent_stonks.config import MODEL_OVERLAY_COLORS

        bars = minute_bars()
        monkeypatch.setattr(mo, "_highlow_forecast", lambda *a, **k: pytest.fail("forecast"))
        items = mo.compute([mo.TRADER_LEVELS_KEY], "AAPL", bars, daily_bars=[],
                           session_date=SESSION, trader_history=self.stopped(bars),
                           trader_running=True)["items"]
        assert MODEL_OVERLAY_COLORS["trader_idle"] not in {i["color"] for i in items}
        levels = {i["label"]: i["value"] for i in items if i["kind"] == "level"}
        assert levels == {"Buy level": 207.0, "Sell level": 209.0}

    def idle_and_ran(self, items):
        from agent_stonks.config import MODEL_OVERLAY_COLORS

        def of(color):
            return [i for i in items if i["color"] == MODEL_OVERLAY_COLORS[color]]
        return of("trader_idle"), of("trader_levels")

    def test_a_stopped_runs_levels_carry_on_under_the_settings_on_screen(self, monkeypatch):
        """Edited after the run stopped: from its last row the gray levels are
        where ▶ Start would rest them now, and the orange ones are still what
        it rested."""
        from dataclasses import replace

        bars = minute_bars()
        history = self.stopped(bars)
        monkeypatch.setattr(mo, "_highlow_forecast", lambda *a, **k: pytest.fail("forecast"))

        def draw(form):
            items = mo.compute([mo.TRADER_LEVELS_KEY], "AAPL", bars, daily_bars=[],
                               session_date=SESSION, trader_config=form,
                               trader_history=history, trader_running=False)["items"]
            idle, ran = self.idle_and_ran(items)
            band = next(i for i in idle if i["kind"] == "band")
            return band, sorted((i["label"], i["value"]) for i in ran)

        ran_config = history["config"]
        before, ran_before = draw(ran_config)
        after, ran_after = draw(replace(ran_config, buy_k=ran_config.buy_k + 0.1))
        assert ran_after == ran_before
        assert after["lower"][0] == before["lower"][0] == 207.0  # the run's last row
        assert all(new < old for new, old in zip(after["lower"][1:], before["lower"][1:]))
        assert after["upper"] == before["upper"]
        assert after["t"] == before["t"]

    def test_a_stopped_run_switched_to_another_model_carries_on_from_that_models_forecast(
        self, monkeypatch,
    ):
        """▶ Start on another model forecasts the day again, so the gray levels
        hang off that model's forecast rather than the run's seed."""
        from dataclasses import replace

        bars = minute_bars()
        history = self.stopped(bars)
        made_at = pd.Timestamp(f"{SESSION} 09:34", tz="America/New_York")
        asked = []

        def forecast(symbol, session, day, credentials, model_key):
            asked.append(model_key)
            return {"forecast": {"pred_high": 230.0, "pred_low": 190.0, "prev_avg": 200.0,
                                 "adr14_abs": 3.0, "or_high": 201.0, "or_low": 199.0},
                    "made_at": made_at, "problem": ""}

        monkeypatch.setattr(mo, "_highlow_forecast", forecast)
        form = replace(history["config"], model_key="highlow2")
        items = mo.compute([mo.TRADER_LEVELS_KEY], "AAPL", bars, daily_bars=[],
                           session_date=SESSION, trader_config=form,
                           trader_history=history, trader_running=False)["items"]
        idle, _ = self.idle_and_ran(items)
        assert asked == ["highlow2"]
        from agent_stonks.apple_trader import session_levels

        _, session, _ = mo._session_of(bars, SESSION)
        stopped_at = pd.Timestamp(f"{SESSION} 09:36", tz="America/New_York")
        walk = [row for row in session_levels(form, forecast(*[None] * 5)["forecast"],
                                              session, made_at)
                if pd.Timestamp(row["t"]) > stopped_at]
        buy = next(i for i in idle if i["label"] in ("Buy level", "Buy/sell levels"))
        drawn = buy["lower"][1:] if buy["kind"] == "band" else [buy["value"]] * len(walk)
        assert drawn == [row["buy"] for row in walk]
        assert drawn[0] != 207.0

    def test_a_running_agent_shows_new_distances_from_the_forming_bar(self, monkeypatch):
        """The run takes the form's buy and sell at its next bar, the one
        forming now: drawn from there in the run's colour, under the stop the
        position was filled with, not a minute later."""
        from dataclasses import replace

        bars = minute_bars(n=8)  # 09:30 – 09:37; the record ends at 09:36
        history = self.stopped(bars)
        monkeypatch.setattr(mo, "_highlow_forecast", lambda *a, **k: pytest.fail("forecast"))
        ran = history["config"]

        def draw(form):
            return mo.compute([mo.TRADER_LEVELS_KEY], "AAPL", bars, daily_bars=[],
                              session_date=SESSION, trader_config=form,
                              trader_history=history, trader_running=True)["items"]

        items = draw(replace(ran, buy_k=ran.buy_k + 0.1))
        idle, drawn = self.idle_and_ran(items)
        assert not idle
        band = next(i for i in drawn if i["kind"] == "band")
        stamps = [pd.Timestamp(t) for t in band["t"]]
        assert stamps[-1] == pd.Timestamp(bars[-1]["t"])
        assert band["lower"][:3] == [207.0] * 3 and band["lower"][-1] != 207.0
        stop = next(i for i in items if i["label"] == "Stop level")
        assert stop["values"][-1] == stop["values"][-2] == 206.0

        # Unchanged, or in a unit the run does not count in: the record alone.
        other_unit = next(u for u in ("adr", "pred_range") if u != ran.level_unit)
        for form in (ran, replace(ran, buy_k=ran.buy_k + 0.1, level_unit=other_unit)):
            levels = {i["label"]: i["value"] for i in draw(form) if i["kind"] == "level"}
            assert levels == {"Buy level": 207.0, "Sell level": 209.0}

    def test_an_edit_is_drawn_as_a_step_at_the_minute_it_was_adopted(self, monkeypatch):
        items = self.compute([mo.TRADER_LEVELS_KEY],
                             self.history([207.0, 207.0, 205.5, 205.5]), monkeypatch)
        band = next(i for i in items if i["kind"] == "band")
        assert band["lower"] == [207.0, 207.0, 205.5, 205.5]
        assert band["step"]
        stamps = [pd.Timestamp(t).tz_convert("America/New_York") for t in band["t"]]
        assert [(t.hour, t.minute) for t in stamps] == [(9, 34), (9, 35), (9, 36), (9, 37)]
        stop = next(i for i in items if i["kind"] == "path")
        assert stop["values"] == [206.0, 206.0, 204.5, 204.5]

    def test_the_stop_is_drawn_only_while_a_position_was_open(self, monkeypatch):
        """Flat until 09:35, long 09:35 to 09:36 under a $205.00 stop, flat
        again: the stop line has gaps where there was no position."""
        import math

        history = self.history([207.0] * 4)
        for row, stop in zip(history["rows"], [None, 205.0, 205.0, None]):
            row["stop"] = stop
        items = self.compute([mo.TRADER_LEVELS_KEY], history, monkeypatch)
        stop = next(i for i in items if i["label"] == "Stop level")
        assert stop["kind"] == "path" and stop["step"]
        assert math.isnan(stop["values"][0]) and math.isnan(stop["values"][-1])
        assert stop["values"][1:3] == [205.0, 205.0]
        levels = {i["label"]: i["value"] for i in items if i["kind"] == "level"}
        assert levels == {"Buy level": 207.0, "Sell level": 209.0}

    def test_a_flat_record_draws_no_stop(self, monkeypatch):
        history = self.history([207.0] * 3)
        for row in history["rows"]:
            row["stop"] = None
        items = self.compute([mo.TRADER_LEVELS_KEY], history, monkeypatch)
        assert not [i for i in items if i["label"] == "Stop level"]

    def test_a_switched_run_draws_each_range_only_for_its_own_minutes(self, monkeypatch):
        """HighLow2 until 09:36, HighLow from then: HighLow's range starts at
        the switch, not at HighLow2's forecast."""
        history = self.history([207.0] * 4, highs=[210.0, 210.0, 206.0, 206.0],
                               model_key="highlow")
        for row, model in zip(history["rows"], ["highlow2"] * 2 + ["highlow"] * 2):
            row["model_key"] = model
        items = self.compute([mo.HIGHLOW_RANGE_KEY], history, monkeypatch)
        high = next(i for i in items if i["label"] == "Pred. high")
        assert high["value"] == 206.0
        stamp = pd.Timestamp(high["x0"]).tz_convert("America/New_York")
        assert (stamp.hour, stamp.minute) == (9, 36)

    def test_levels_that_never_moved_are_still_two_flat_lines(self, monkeypatch):
        items = self.compute([mo.TRADER_LEVELS_KEY], self.history([207.0] * 3), monkeypatch)
        levels = {i["label"]: i["value"] for i in items if i["kind"] == "level"}
        assert levels["Buy level"] == 207.0 and levels["Sell level"] == 209.0

    def test_the_day_range_is_the_runs_own_forecast(self, monkeypatch):
        items = self.compute([mo.HIGHLOW_RANGE_KEY],
                             self.history([207.0] * 3, highs=[210.0, 212.0, 212.0]),
                             monkeypatch)
        band = next(i for i in items if i["kind"] == "band")
        assert band["upper"] == [210.0, 212.0, 212.0]

    def test_a_highlow2_run_draws_its_range_on_the_highlow2_overlay_only(self, monkeypatch):
        history = self.history([207.0] * 3, highs=[210.0, 212.0, 212.0], model_key="highlow2")
        items = self.compute([mo.HIGHLOW2_RANGE_KEY], history, monkeypatch)
        assert next(i for i in items if i["kind"] == "band")["upper"][-1] == 212.0
        # The HighLow overlay is not this run's model, so it forecasts as before.
        monkeypatch.setattr(mo, "_highlow_forecast", lambda *a, **k: {
            "forecast": None, "made_at": None, "problem": "stub"})
        result = mo.compute([mo.HIGHLOW_RANGE_KEY], "AAPL", minute_bars(), daily_bars=[],
                            session_date=SESSION, trader_history=history)
        assert result["items"] == [] and "stub" in result["notes"][0]

    def test_another_days_or_symbols_record_is_ignored(self, monkeypatch):
        calls = []
        monkeypatch.setattr(mo, "_trader_levels_items",
                            lambda *a, **k: calls.append(k) or ([], ""))
        monkeypatch.setattr(mo, "_highlow_forecast", lambda *a, **k: {
            "forecast": None, "made_at": None, "problem": "stub"})
        for history in (self.history([207.0], day="2026-08-06"),
                        self.history([207.0], ticker="GOOGL")):
            mo.compute([mo.TRADER_LEVELS_KEY], "AAPL", minute_bars(), daily_bars=[],
                       session_date=SESSION, trader_history=history)
        assert all(k.get("levels") is None for k in calls) and len(calls) == 2


class TestHighLowOverlay:
    """Each range model's forecast is drawn under its own key, and an agent
    configured on one has its levels drawn under that model's high -- HighLow's
    or HighLow2's, never the other's."""

    def stub_both(self, monkeypatch, highlow_high=206.0, highlow2_high=210.0):
        hl = pytest.importorskip("agent_stonks.highlow_model")
        hl2 = pytest.importorskip("agent_stonks.highlow2_model")

        monkeypatch.setattr(mo.apple_models, "load", lambda *a, **k: {"stub": True})
        base = {"prev_avg": 200.0, "adr14_abs": 3.0, "or_high": 201.0, "or_low": 199.0}
        seen = []

        def highlow(bundle, ticker, opening, day, key=None, secret=None):
            seen.append((key, secret))
            return {**base, "pred_high": highlow_high, "pred_low": 190.0}

        monkeypatch.setattr(hl, "forecast_session", highlow)
        monkeypatch.setattr(hl2, "forecast_session",
                            lambda *a, **k: {**base, "pred_high": highlow2_high, "pred_low": 190.0})
        return seen

    def config(self, **kwargs):
        from agent_stonks.apple_trader import AppleTraderConfig
        from agent_stonks.config import UNIT_ADR

        return AppleTraderConfig(
            ticker="AAPL", buy_k=0.75, sell_k=0.10, breach_update="off",
            level_unit=UNIT_ADR, contain_range=False, **kwargs,
        )

    def test_the_highlow_range_is_two_levels_under_its_own_key(self, monkeypatch):
        self.stub_both(monkeypatch)
        items = mo.compute([mo.HIGHLOW_RANGE_KEY], "AAPL", minute_bars(),
                           daily_bars=[], session_date=SESSION)["items"]
        levels = {i["label"]: i["value"] for i in items if i["kind"] == "level"}
        assert levels == {"Pred. high": 206.0, "Pred. low": 190.0}
        assert {i["key"] for i in items} == {mo.HIGHLOW_RANGE_KEY}

    def test_the_agents_levels_follow_the_model_it_is_configured_on(self, monkeypatch):
        self.stub_both(monkeypatch)

        def buy(model_key):
            items = mo.compute(
                [mo.TRADER_LEVELS_KEY], "AAPL", minute_bars(), daily_bars=[],
                session_date=SESSION, trader_config=self.config(model_key=model_key),
            )["items"]
            return next(i["value"] for i in items if i["label"] == "Buy level")

        assert buy("highlow2") == pytest.approx(210.0 - 0.75 * 3.0)
        assert buy("highlow") == pytest.approx(206.0 - 0.75 * 3.0)

    def test_a_symbol_only_highlow_covers_draws_highlows_levels(self, monkeypatch):
        """MU has no HighLow2 bundle: the agent's levels are offered there, and
        hang off HighLow's forecast with or without a MU configuration."""
        from agent_stonks.apple_trader import AppleTraderConfig

        self.stub_both(monkeypatch)

        def no_highlow2(*a, **k):
            raise AssertionError("MU has no HighLow2 bundle to ask")

        monkeypatch.setattr(pytest.importorskip("agent_stonks.highlow2_model"),
                            "forecast_session", no_highlow2)
        assert mo.TRADER_LEVELS_KEY in mo.keys_for("MU")
        assert mo.HIGHLOW2_RANGE_KEY not in mo.keys_for("MU")

        def buy(config):
            items = mo.compute(
                [mo.TRADER_LEVELS_KEY], "MU", minute_bars(), daily_bars=[],
                session_date=SESSION, trader_config=config,
            )["items"]
            return next(i["value"] for i in items if i["label"] == "Buy level")

        mu_highlow = buy(AppleTraderConfig(ticker="MU", model_key="highlow"))
        assert buy(None) == mu_highlow
        assert buy(self.config(model_key="highlow2")) == mu_highlow  # an AAPL agent's

    def test_the_credentials_reach_the_sip_history(self, monkeypatch):
        seen = self.stub_both(monkeypatch)
        mo.compute([mo.HIGHLOW_RANGE_KEY], "AAPL", minute_bars(), daily_bars=[],
                   session_date=SESSION, credentials=("k", "s"))
        assert seen == [("k", "s")]

    def test_a_highlow_run_opens_showing_its_range(self):
        assert mo.for_models([apple_models.HIGHLOW_KEY], "AAPL")["keys"] == [mo.HIGHLOW_RANGE_KEY]
        assert mo.HIGHLOW_RANGE_KEY not in mo.keys_for("GOOGL")


class TestSessionForecast:
    """`session_forecast` hands back the numbers `trader_levels` hangs the levels
    off -- the configured model's forecast -- and which overlay draws it."""

    def test_a_highlow2_agent_gets_highlow2s_forecast(self, monkeypatch):
        TestHighLowOverlay().stub_both(monkeypatch)
        result = mo.session_forecast(apple_models.HIGHLOW2_KEY, "aapl", minute_bars(),
                                     session_date=SESSION)
        assert result["overlay"] == mo.HIGHLOW2_RANGE_KEY
        assert result["forecast"]["pred_high"] == 210.0
        made = pd.Timestamp(result["made_at"]).tz_convert("America/New_York")
        assert (made.hour, made.minute) == (9, 34)

    def test_a_highlow_agent_gets_highlows_and_never_asks_highlow2(self, monkeypatch):
        TestHighLowOverlay().stub_both(monkeypatch)
        monkeypatch.setattr(pytest.importorskip("agent_stonks.highlow2_model"),
                            "forecast_session", lambda *a, **k: pytest.fail("highlow2"))
        result = mo.session_forecast(apple_models.HIGHLOW_KEY, "AAPL", minute_bars(),
                                     session_date=SESSION)
        assert result["overlay"] == mo.HIGHLOW_RANGE_KEY
        assert result["forecast"]["pred_high"] == 206.0

    def test_a_removed_model_is_a_reason_not_the_default_models_forecast(self, monkeypatch):
        """`apple_models.get` falls back to the default model for an unknown
        key; a tuning job on a removed model must not be drawn under HighLow's
        range as though it were its own."""
        TestHighLowOverlay().stub_both(monkeypatch)
        result = mo.session_forecast("dayrange", "AAPL", minute_bars(), session_date=SESSION)
        assert result["forecast"] is None
        assert "not a model this app can run" in result["problem"]

    def test_a_day_with_no_bars_is_a_reason_not_a_forecast(self):
        result = mo.session_forecast(apple_models.HIGHLOW_KEY, "AAPL", minute_bars(),
                                     session_date="2026-08-10")
        assert result["forecast"] is None
        assert result["problem"] == "No bars for 2026-08-10."


class TestHighLow2Overlay:
    """HighLow2's range is drawn like the other two, under its own key, and an
    agent configured on it has its levels drawn under HighLow2's high."""

    @pytest.fixture()
    def stubbed(self, monkeypatch):
        pytest.importorskip("agent_stonks.highlow_model")
        hl2 = pytest.importorskip("agent_stonks.highlow2_model")
        seen = TestHighLowOverlay().stub_both(monkeypatch, highlow_high=206.0)
        base = {"prev_avg": 200.0, "adr14_abs": 3.0, "or_high": 201.0, "or_low": 199.0}

        def highlow2(bundle, ticker, opening, day, key=None, secret=None):
            seen.append(("highlow2", key, secret))
            return {**base, "pred_high": 204.0, "pred_low": 192.0}

        monkeypatch.setattr(hl2, "forecast_session", highlow2)
        return seen

    def test_the_highlow2_range_is_two_levels_under_its_own_key(self, stubbed):
        items = mo.compute([mo.HIGHLOW2_RANGE_KEY], "AAPL", minute_bars(),
                           daily_bars=[], session_date=SESSION, credentials=("k", "s"))["items"]
        levels = {i["label"]: i["value"] for i in items if i["kind"] == "level"}
        assert levels == {"Pred. high": 204.0, "Pred. low": 192.0}
        assert {i["key"] for i in items} == {mo.HIGHLOW2_RANGE_KEY}
        assert stubbed == [("highlow2", "k", "s")]  # HighLow was never asked

    def test_the_agents_levels_follow_highlow2(self, stubbed):
        items = mo.compute(
            [mo.TRADER_LEVELS_KEY], "AAPL", minute_bars(), daily_bars=[], session_date=SESSION,
            trader_config=TestHighLowOverlay().config(model_key="highlow2"),
        )["items"]
        buy = next(i for i in items if i["label"] == "Buy level")
        assert buy["value"] == pytest.approx(204.0 - 0.75 * 3.0)
        assert "(HighLow2)" in buy["note"]

    def test_both_highlow_ranges_draw_together_from_their_own_models(self, stubbed):
        items = mo.compute([mo.HIGHLOW_RANGE_KEY, mo.HIGHLOW2_RANGE_KEY], "AAPL", minute_bars(),
                           daily_bars=[], session_date=SESSION)["items"]
        highs = {i["key"]: i["value"] for i in items if i.get("label") == "Pred. high"}
        assert highs == {mo.HIGHLOW_RANGE_KEY: 206.0, mo.HIGHLOW2_RANGE_KEY: 204.0}

    def test_a_highlow2_run_opens_showing_its_range(self):
        assert mo.for_models([apple_models.HIGHLOW2_KEY], "AAPL")["keys"] == [mo.HIGHLOW2_RANGE_KEY]
        assert mo.HIGHLOW2_RANGE_KEY not in mo.keys_for("MU")

    def test_a_rest_of_session_forecast_is_drawn_as_one(self, monkeypatch, stubbed):
        """A bundle with a rest head (INTC's had one) hands over the range after
        9:35: the same two levels, labelled as what they are, and the agent's
        levels under its high."""
        hl2 = pytest.importorskip("agent_stonks.highlow2_model")
        base = {"prev_avg": 200.0, "adr14_abs": 3.0, "or_high": 205.0, "or_low": 199.0}
        monkeypatch.setattr(hl2, "forecast_session", lambda *a, **k: {
            **base, "pred_high": 203.0, "pred_low": 196.0, "day_high": 205.0, "day_low": 195.0,
            "range_after_opening": True,
        })
        from agent_stonks.apple_trader import AppleTraderConfig
        from agent_stonks.config import UNIT_ADR

        config = AppleTraderConfig(
            ticker="AAPL", model_key="highlow2", buy_k=0.75, sell_k=0.10, breach_update="off",
            level_unit=UNIT_ADR, contain_range=False,
        )
        items = mo.compute([mo.HIGHLOW2_RANGE_KEY, mo.TRADER_LEVELS_KEY], "AAPL", minute_bars(),
                           daily_bars=[], session_date=SESSION, trader_config=config)["items"]
        mine = [i for i in items if i["key"] == mo.HIGHLOW2_RANGE_KEY]
        levels = {i["label"]: i for i in mine if i["kind"] == "level"}
        assert {k: v["value"] for k, v in levels.items()} == {"Pred. high": 203.0, "Pred. low": 196.0}
        assert levels["Pred. high"]["note"].startswith("predicted high after ")
        assert "session" not in levels["Pred. low"]["note"]
        buy = next(i for i in items if i["label"] == "Buy level")
        assert buy["value"] == pytest.approx(203.0 - 0.75 * 3.0)


class TestFastFuriousOverlay:
    """Fast&Furious trades on its 30-minute momentum forecast, so that is what
    its overlay draws -- a line on the momentum panel between the two entry
    thresholds, and the signals on the candles. The 2-hour range, which only
    sizes the stop, is an overlay of its own that nothing selects for you."""

    THETA = 0.04

    @pytest.fixture()
    def stubbed(self, monkeypatch):
        ff = pytest.importorskip("agent_stonks.fastfurious_model")
        calls: list = []
        minutes = np.arange(1, 41)
        y = np.zeros(len(minutes))
        y[10:13] = 0.05   # a buy run of three minutes: one marker
        y[25] = -0.06     # one sell minute
        s30 = np.linspace(0.006, 0.003, len(minutes))
        self.fc = pd.DataFrame(
            {"y_hat": y, "mom_bps": y * s30 / 30 * 1e4, "p": 200.0 + minutes * 0.01,
             "s30": s30, "pred_high": 202.0, "pred_low": 198.0},
            index=pd.Index(minutes),
        )
        self.theta = (self.THETA, 20)
        monkeypatch.setattr(apple_models, "load", lambda key, ticker=None: {"kind": "fastfurious"})

        def forecasts(bundle, symbol, day, last, key=None, secret=None, wait=True):
            calls.append(last)
            return self.fc.copy()

        monkeypatch.setattr(ff, "session_forecasts", forecasts)
        monkeypatch.setattr(ff, "threshold", lambda *a, **k: self.theta)
        return calls

    def compute(self, keys):
        return mo.compute(keys, "AAPL", minute_bars(n=60), session_date=SESSION)

    def test_the_signal_overlay_is_the_momentum_forecast_not_the_range(self, stubbed):
        result = self.compute([mo.FASTFURIOUS_SIGNAL_KEY])
        assert result["notes"] == []
        assert not [i for i in result["items"] if i["kind"] == "band"]
        series = {i["label"]: i for i in result["items"] if i["kind"] == "series"}
        assert set(series) == {"30-min momentum forecast", "Buy threshold", "Sell threshold"}
        assert {i["unit"] for i in series.values()} == {"bps/min"}
        line = series["30-min momentum forecast"]["values"]
        assert line == pytest.approx(list(self.fc["mom_bps"]))
        # The threshold is fixed in y, so in bps per minute it follows s30.
        edge = self.THETA * self.fc["s30"] / 30 * 1e4
        assert series["Buy threshold"]["values"] == pytest.approx(list(edge))
        assert series["Sell threshold"]["values"] == pytest.approx(list(-edge))

    def test_the_line_is_drawn_at_the_end_of_the_window_it_forecasts(self, stubbed):
        items = self.compute([mo.FASTFURIOUS_SIGNAL_KEY])["items"]
        open_ = pd.Timestamp(f"{SESSION} 09:30", tz="America/New_York")
        made = [open_ + pd.Timedelta(minutes=int(m)) for m in self.fc.index]
        due = [(t + pd.Timedelta(minutes=30)).tz_convert("UTC").isoformat() for t in made]
        for item in (i for i in items if i["kind"] == "series"):
            assert item["t"] == due, item["label"]
        line = next(i for i in items if i["label"] == "30-min momentum forecast")
        assert line["text"][0].startswith("made at the 09:31 bar · ")
        # The signals stay at the bar the trader acts on.
        buy = next(i for i in items if i["kind"] == "event")
        assert buy["ts"] == made[10].tz_convert("UTC").isoformat()

    def test_a_marker_sits_30_minutes_before_the_line_crosses_a_threshold(self, stubbed):
        items = self.compute([mo.FASTFURIOUS_SIGNAL_KEY])["items"]
        events = [i for i in items if i["kind"] == "event"]
        assert [e["icon"] for e in events] == ["▲", "▼"]
        series = {i["label"]: i for i in items if i["kind"] == "series"}
        at = {t: n for n, t in enumerate(series["30-min momentum forecast"]["t"])}
        line = series["30-min momentum forecast"]["values"]
        buy, sell = (
            at[(pd.Timestamp(e["ts"]) + pd.Timedelta(minutes=30)).isoformat()] for e in events
        )
        assert line[buy] >= series["Buy threshold"]["values"][buy]
        assert line[buy - 1] < series["Buy threshold"]["values"][buy - 1]
        assert line[sell] <= series["Sell threshold"]["values"][sell]
        # Markers stay on the candles, at the price the forecast was read at.
        assert events[0]["price"] == pytest.approx(self.fc["p"].iloc[buy])

    def test_the_range_is_its_own_overlay(self, stubbed):
        result = self.compute([mo.FASTFURIOUS_RANGE_KEY])
        assert [(i["kind"], i["key"]) for i in result["items"]] == [
            ("band", mo.FASTFURIOUS_RANGE_KEY)
        ]
        band_ = result["items"][0]
        assert set(band_["upper"]) == {202.0} and set(band_["lower"]) == {198.0}

    def test_both_overlays_read_one_forecast(self, stubbed):
        self.compute([mo.FASTFURIOUS_SIGNAL_KEY, mo.FASTFURIOUS_RANGE_KEY])
        assert len(stubbed) == 1

    def test_a_fastfurious_run_opens_without_the_range(self):
        assert mo.for_models([apple_models.FASTFURIOUS_KEY], "AAPL")["keys"] == [
            mo.FASTFURIOUS_SIGNAL_KEY
        ]
        assert mo.FASTFURIOUS_RANGE_KEY in mo.keys_for("AAPL")

    def test_no_threshold_still_draws_the_line_and_says_why(self, stubbed):
        self.theta = (float("nan"), 3)
        result = self.compute([mo.FASTFURIOUS_SIGNAL_KEY])
        assert [i["label"] for i in result["items"]] == ["30-min momentum forecast"]
        assert "no entry threshold (3 earlier sessions)" in result["notes"][0]

    def test_a_missing_option_table_only_empties_the_range(self, stubbed):
        self.fc[["pred_high", "pred_low"]] = np.nan
        self.fc.attrs["range_problem"] = "no option table for 2026-08-06"
        result = self.compute([mo.FASTFURIOUS_SIGNAL_KEY, mo.FASTFURIOUS_RANGE_KEY])
        assert {i["key"] for i in result["items"]} == {mo.FASTFURIOUS_SIGNAL_KEY}
        assert result["notes"] == [
            f"{mo.label(mo.FASTFURIOUS_RANGE_KEY)}: no 2-hour range "
            "(no option table for 2026-08-06)."
        ]


class TestLiveOverlays:
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

    def test_a_new_row_in_the_agents_record_redraws_within_the_bar(self, monkeypatch):
        """A sidebar edit is adopted mid-bar; the chart must not wait a minute."""
        calls = self.capture_compute(monkeypatch)
        bars = minute_bars(n=40)
        rows = [{"t": 1, "buy": 207.0}]
        sym_state = self.make_state(bars)
        sym_state.app = SimpleNamespace(
            apple_trader_config=None, apple_trader_levels={"rows": rows},
            api_key="", api_secret="",
        )
        mo.live_overlays(sym_state, bars, [mo.TRADER_LEVELS_KEY])
        mo.live_overlays(sym_state, bars, [mo.TRADER_LEVELS_KEY])
        rows[-1] = {"t": 1, "buy": 205.5}
        mo.live_overlays(sym_state, bars, [mo.TRADER_LEVELS_KEY])
        assert len(calls) == 2
        assert calls[-1]["trader_history"]["rows"][-1]["buy"] == 205.5

    def test_starting_the_agent_redraws_the_levels_within_the_bar(self, monkeypatch):
        """▶ Start turns the gray levels orange at once, and only an Apple
        Trader or an Orchestra race is an agent whose levels these are."""
        calls = self.capture_compute(monkeypatch)
        bars = minute_bars(n=40)
        sym_state = self.make_state(bars)
        sym_state.app = SimpleNamespace(
            apple_trader_config=None, apple_trader_levels=None, api_key="", api_secret="",
            agent_running=False, llm_personality="apple_trader",
        )
        mo.live_overlays(sym_state, bars, [mo.TRADER_LEVELS_KEY])
        sym_state.app.agent_running = True
        mo.live_overlays(sym_state, bars, [mo.TRADER_LEVELS_KEY])
        assert [c["trader_running"] for c in calls] == [False, True]
        for personality, running in (("orchestra", True), ("momentum", False)):
            sym_state.app.llm_personality = personality
            assert mo.live_trader_running(sym_state.app) is running

    def test_the_answer_is_cached_until_a_new_bar_arrives(self, monkeypatch):
        calls = []

        def fake_compute(*args, **kwargs):
            calls.append(kwargs)
            return {"items": [], "notes": []}

        monkeypatch.setattr(mo, "compute", fake_compute)
        bars = minute_bars(n=40)
        state = self.make_state(bars)

        mo.live_overlays(state, bars, [mo.HIGHLOW_RANGE_KEY])
        mo.live_overlays(state, bars, [mo.HIGHLOW_RANGE_KEY])
        assert len(calls) == 1

        mo.live_overlays(state, bars + minute_bars(n=1, base=210.0), [mo.HIGHLOW_RANGE_KEY])
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
        mo.live_overlays(state, bars, [mo.HIGHLOW_RANGE_KEY])
        mo.live_overlays(state, bars, [mo.HIGHLOW_RANGE_KEY, mo.PROFILE_RANGE_KEY])
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

    def test_a_path_is_one_unfilled_line_behind_the_candles_in_its_bands_legend(self):
        """A moving stop beside the moving buy/sell band: its own curve, no
        tint, and hidden with the band from the one legend entry."""
        stop = {**envelope(), "kind": "path", "label": "Stop level",
                "values": envelope()["lower"], "color": "#f87171"}
        del stop["upper"], stop["lower"]
        fig = self.chart([envelope(), stop])
        traces = [t for t in fig.data if t.legendgroup == "iv"]
        assert len(traces) == 3
        path = traces[-1]
        assert path.fill in (None, "none") and path.line.color == "#f87171"
        assert "Stop level" in path.hovertemplate
        assert sum(bool(t.showlegend) for t in traces) == 1
        candles = next(i for i, t in enumerate(fig.data) if t.name == "AAPL")
        assert fig.data.index(path) < candles

    def test_a_stepped_band_holds_each_value_until_the_next(self):
        """A revised forecast changes at a bar; a sloped line into the revision
        would hide the minute it happened. A smooth envelope stays linear."""
        fig = self.chart([{**envelope(), "step": True}])
        edges = [t for t in fig.data if t.legendgroup == "iv"]
        assert {t.line.shape for t in edges} == {"hv"}
        fig = self.chart([envelope()])
        edges = [t for t in fig.data if t.legendgroup == "iv"]
        assert {t.line.shape for t in edges} == {"linear"}

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

    def test_a_bands_two_edges_are_named_in_the_hover(self):
        """A band that stands in for two named levels has to keep their names.
        The legend still groups the pair under the overlay, but hovering the
        top edge must say "Pred. high", not "Pred. high / low upper"."""
        named = {**envelope(), "label": "Pred. high / low",
                 "lower_label": "Pred. low", "upper_label": "Pred. high"}
        fig = self.chart([named])
        hovers = [t.hovertemplate for t in fig.data if getattr(t, "legendgroup", None) == "iv"]
        assert any("<b>Pred. high</b>" in h for h in hovers)
        assert any("<b>Pred. low</b>" in h for h in hovers)
        assert not any("upper" in h or "lower" in h for h in hovers)

    def test_a_band_without_edge_names_still_says_which_edge(self):
        """An envelope's edges have no names of their own, so the generic pair
        is the right answer there rather than a blank."""
        fig = self.chart([envelope()])
        hovers = [t.hovertemplate for t in fig.data if getattr(t, "legendgroup", None) == "iv"]
        assert any("Pred. intraday range upper" in h for h in hovers)
        assert any("Pred. intraday range lower" in h for h in hovers)

    def test_overlay_x_max_ignores_items_that_do_not_reach_forward(self):
        last = pd.Timestamp(BARS[-1]["t"])
        assert overlay_x_max([band(), level(), moment()], last) == last

    def test_overlay_x_max_handles_a_naive_reference(self):
        naive = pd.Timestamp(BARS[-1]["t"]).tz_localize(None)
        assert overlay_x_max([window()], naive).tzinfo is None


def forecast_line(key="ff", label="Forecast", values=None, start=1):
    """A `series` item over BARS[start:], in bps per minute."""
    stamps = [b["t"] for b in BARS[start:]]
    if values is None:
        values = list(np.linspace(-0.05, 0.12, len(stamps)))
    return {"kind": "series", "key": key, "label": label, "group": "Fast&Furious",
            "t": stamps, "values": values, "unit": "bps/min", "color": "#facc15",
            "dash": "solid", "width": 1.8, "note": "why", "text": None, "forward": False}


class TestSeriesRenderer:
    """A forecast that is not a price goes on the momentum panel, on a value
    axis of its own whose zero lines up with the bars'."""

    def chart(self, items, **kwargs):
        kwargs.setdefault("show_momentum", True)
        return build_chart(BARS, [], [], "AAPL", SESSION_START, model_overlays=items, **kwargs)

    @staticmethod
    def line(fig, name="Fast&Furious"):
        return next(t for t in fig.data if t.name == name and t.type == "scatter")

    def test_the_line_is_on_its_own_axis_over_the_momentum_panel(self):
        fig = self.chart([forecast_line()])
        panel = fig.get_subplot(3, 1)
        trace = self.line(fig)
        axis = fig.layout["yaxis" + trace.yaxis[1:]]
        assert axis.overlaying == "y" + panel.yaxis.plotly_name[len("yaxis"):]
        assert trace.xaxis == "x" + panel.xaxis.plotly_name[len("xaxis"):]
        assert axis.side == "right" and axis.title.text == "bps/min"

    def test_the_other_subplots_keep_their_axis_ids(self):
        """`make_subplots(secondary_y=...)` would have renumbered them, and the
        price profile is addressed as x2/y2."""
        fig = self.chart([forecast_line()])
        assert fig.get_subplot(1, 2).xaxis.plotly_name == "xaxis2"
        assert fig.get_subplot(1, 2).yaxis.plotly_name == "yaxis2"

    def test_both_axes_are_centred_on_zero(self):
        fig = self.chart([forecast_line()])
        own = fig.layout["yaxis" + self.line(fig).yaxis[1:]].range
        bars = fig.layout[fig.get_subplot(3, 1).yaxis.plotly_name].range
        assert own[0] == pytest.approx(-own[1]) and own[1] >= 0.12
        assert bars[0] == pytest.approx(-bars[1]) and bars[1] > 0

    def test_a_window_centres_on_what_it_shows(self):
        values = [0.5] + [0.02] * (len(BARS) - 2)
        fig = self.chart([forecast_line(values=values)], window_minutes=30)
        own = fig.layout["yaxis" + self.line(fig).yaxis[1:]].range
        assert own[1] == pytest.approx(0.02 * 1.05)

    def test_without_a_momentum_panel_nothing_is_drawn(self):
        fig = self.chart([forecast_line()], show_momentum=False)
        assert not [t for t in fig.data if t.name == "Fast&Furious"]
        assert not [k for k, v in fig.layout.to_plotly_json().items()
                    if k.startswith("yaxis") and v.get("overlaying")]

    def test_simlabs_plain_figure_skips_it(self):
        fig = go.Figure(go.Scatter(x=[b["t"] for b in BARS], y=[b["c"] for b in BARS]))
        add_model_overlays([forecast_line()], fig, pd.Timestamp(BARS[0]["t"]),
                           pd.Timestamp(BARS[-1]["t"]), row=None, col=None)
        assert len(fig.data) == 1

    def test_the_line_joins_its_markers_legend_entry(self):
        marker = {**moment(), "key": "ff", "group": "Fast&Furious"}
        fig = self.chart([forecast_line(), forecast_line(label="Buy threshold"), marker])
        lines = [t for t in fig.data if t.yaxis not in (None, "y") and t.legendgroup == "ff"]
        assert len(lines) == 2 and not any(t.showlegend for t in lines)
        markers = next(t for t in fig.data if getattr(t, "mode", None) == "markers+text")
        assert markers.legendgroup == "ff" and markers.showlegend

    def test_with_no_markers_the_line_has_the_legend_entry(self):
        fig = self.chart([forecast_line(), forecast_line(label="Buy threshold")])
        lines = [t for t in fig.data if t.legendgroup == "ff"]
        assert [t.showlegend for t in lines] == [True, False]

    def test_other_overlays_markers_keep_no_legend_group(self):
        fig = self.chart([moment()])
        markers = next(t for t in fig.data if getattr(t, "mode", None) == "markers+text")
        assert markers.legendgroup is None
