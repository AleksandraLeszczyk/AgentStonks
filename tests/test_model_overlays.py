"""Tests for the model-prediction chart overlays (agent_stonks/model_overlays.py).

Two halves. The first drives `compute` with stubbed models, so what is pinned
is the *shape* of the answer -- which item kinds each overlay produces, which
bars the momentum model is asked about, and that a missing model becomes a note
rather than an exception. The second drives the renderer, so what is pinned is
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

from agent_stonks import model_overlays as mo
from agent_stonks.charts import add_model_overlays, build_chart, overlay_x_max

SESSION = "2026-08-07"
SESSION_START = datetime(2026, 8, 7, 13, 25, tzinfo=timezone.utc)


def minute_bars(n=180, seed=7, base=200.0):
    """A session of 1-minute bars starting at the 09:30 open.

    Momentum is a volatility-normalised trailing return, so a tape with no
    swings has no regime changes and the momentum overlay would have nothing to
    draw. The sine wave is there to guarantee some.
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

    def test_the_day_range_overlay_follows_its_models_tickers(self):
        assert mo.OVERLAYS[mo.DAY_RANGE_KEY].covers("GOOGL")
        assert not mo.OVERLAYS[mo.MOMENTUM_KEY].covers("GOOGL")

    def test_label_falls_back_to_the_key_for_an_unknown_overlay(self):
        assert mo.label("nope") == "nope"


class TestCompute:
    def test_no_keys_is_no_work(self):
        assert mo.compute([], "AAPL", minute_bars()) == {"items": [], "notes": []}

    def test_no_bars_is_no_work(self):
        assert mo.compute(mo.keys(), "AAPL", []) == {"items": [], "notes": []}

    def test_unknown_keys_are_ignored(self):
        assert mo.compute(["nope"], "AAPL", minute_bars())["items"] == []

    def test_a_symbol_the_model_does_not_cover_is_a_note_not_an_error(self):
        result = mo.compute([mo.MOMENTUM_KEY], "MSFT", minute_bars())
        assert result["items"] == []
        assert "MSFT" in result["notes"][0]
        assert "fitted on" in result["notes"][0]

    def test_a_missing_bundle_is_a_note_not_an_exception(self, monkeypatch):
        monkeypatch.setattr(mo.apple_models, "load", lambda *a, **k: None)
        result = mo.compute([mo.MOMENTUM_KEY], "AAPL", minute_bars())
        assert result["items"] == []
        assert result["notes"] and "Momentum regime changes" in result["notes"][0]

    def test_a_model_that_raises_is_a_note_not_an_exception(self, monkeypatch):
        def boom(*_a, **_k):
            raise RuntimeError("the checkpoint is corrupt")

        monkeypatch.setattr(mo.apple_models, "load", boom)
        result = mo.compute([mo.MOMENTUM_KEY], "AAPL", minute_bars())
        assert result["items"] == []
        assert "the checkpoint is corrupt" in result["notes"][0]

    def test_bars_from_another_day_produce_a_note(self):
        result = mo.compute([mo.MOMENTUM_KEY], "AAPL", minute_bars(),
                            session_date="2026-08-06")
        assert result["items"] == []
        assert "2026-08-06" in result["notes"][0]


def stub_momentum_bundle(monkeypatch, proba=0.9, turn=None, threshold=0.05,
                         min_dwell=15):
    """A momentum bundle that answers every sequence with a fixed number.

    The point of the momentum overlay is *which bars get asked* and what the
    answer becomes on the chart, not what the answer is -- so the model is a
    constant and the two TimeToChange2 bundles' own tests keep the numbers
    honest.
    """
    from agent_stonks import persistence_model as pm

    bundle = {
        "feature_columns": pm.FEATURE_COLUMNS,
        "seq_len": 20,
        "threshold": threshold,
        "settings": {"persistence": {"min_dwell": min_dwell}},
        "asked": [],
    }
    monkeypatch.setattr(mo.apple_models, "load", lambda *a, **k: bundle)
    monkeypatch.setattr(
        mo.persistence_model, "predict_proba",
        lambda b, X: np.full(len(np.atleast_3d(X)), proba),
    )
    monkeypatch.setattr(mo.persistence_model, "anticipates", lambda b: turn is not None)
    monkeypatch.setattr(
        mo.persistence_model, "predict_turn_proba", lambda b, X: np.array([turn])
    )
    return bundle


class TestMomentumOverlay:
    def test_marks_every_regime_change_as_an_event(self, monkeypatch):
        from agent_stonks import persistence_model as pm

        stub_momentum_bundle(monkeypatch)
        bars = minute_bars()
        result = mo.compute([mo.MOMENTUM_KEY], "AAPL", bars)

        frame = pm.frame_from_bars(bars)
        scored = pm.add_momentum_regimes(frame, pm.momentum_params(None))
        expected = int(scored["regime_change"].fillna(False).sum())

        events = [i for i in result["items"] if i["kind"] == "event"]
        assert expected > 0
        assert len(events) == expected

    def test_a_backed_change_into_positive_shades_the_bars_it_should_hold(
        self, monkeypatch
    ):
        stub_momentum_bundle(monkeypatch, proba=0.9, threshold=0.05, min_dwell=15)
        result = mo.compute([mo.MOMENTUM_KEY], "AAPL", minute_bars())
        spans = [i for i in result["items"] if i["kind"] == "span"]
        assert spans, "a change the model backs should carry a hold window"
        span = spans[0]
        assert span["forward"] is True
        assert span["y0"] is None and span["y1"] is None  # full height, a moment
        length = pd.Timestamp(span["x1"]) - pd.Timestamp(span["x0"])
        assert length == pd.Timedelta(minutes=15)

    def test_only_a_prediction_gets_a_vertical_line(self, monkeypatch):
        """A regime change is context; a change the model backs is a claim.

        On a chart covering several sessions the difference is between a
        readable picture and forty vertical rules.
        """
        stub_momentum_bundle(monkeypatch, proba=0.9, threshold=0.05)
        items = mo.compute([mo.MOMENTUM_KEY], "AAPL", minute_bars())["items"]
        for event in (i for i in items if i["kind"] == "event"):
            backed = "%" in event["label"]
            assert event["line"] is backed, event["label"]

    def test_a_change_below_the_threshold_gets_no_hold_window(self, monkeypatch):
        stub_momentum_bundle(monkeypatch, proba=0.01, threshold=0.5)
        result = mo.compute([mo.MOMENTUM_KEY], "AAPL", minute_bars())
        assert not [i for i in result["items"] if i["kind"] == "span"]
        assert any("fades" in i.get("note", "") for i in result["items"])

    def test_only_changes_into_positive_are_scored(self, monkeypatch):
        stub_momentum_bundle(monkeypatch, proba=0.9)
        items = mo.compute([mo.MOMENTUM_KEY], "AAPL", minute_bars())["items"]
        for item in items:
            if item["kind"] != "event":
                continue
            if "positive" not in item["label"]:
                assert "%" not in item["label"], "a non-positive change has no probability"

    def test_a_classifier_bundle_never_forecasts_a_turn(self, monkeypatch):
        stub_momentum_bundle(monkeypatch, turn=None)
        items = mo.compute([mo.MOMENTUM_KEY], "AAPL", minute_bars())["items"]
        assert not [i for i in items if i.get("icon") == "⤴"]

    def test_a_forecasting_bundle_marks_the_newest_bar_when_it_is_not_positive(
        self, monkeypatch
    ):
        from agent_stonks import persistence_model as pm

        stub_momentum_bundle(monkeypatch, turn=0.8, threshold=0.05)
        bars = minute_bars()
        scored = pm.add_momentum_regimes(
            pm.frame_from_bars(bars), pm.momentum_params(None)
        )
        if int(scored["regime"].iloc[-1]) == 1:
            pytest.skip("this tape ends in the positive regime; nothing to anticipate")

        items = mo.compute([mo.MOMENTUM_KEY], "AAPL", bars)["items"]
        turns = [i for i in items if i.get("icon") == "⤴"]
        assert len(turns) == 1
        assert turns[0]["forward"] is True
        assert pd.Timestamp(turns[0]["ts"]) == pd.Timestamp(bars[-1]["t"])

    def test_too_few_bars_to_have_momentum_is_a_note(self, monkeypatch):
        stub_momentum_bundle(monkeypatch)
        result = mo.compute([mo.MOMENTUM_KEY], "AAPL", minute_bars(n=5))
        assert result["items"] == []
        assert "bars" in result["notes"][0]

    def test_the_momentum_model_choice_is_passed_through(self, monkeypatch):
        seen = {}

        def load(key, ticker=None):
            seen["key"] = key
            return None

        monkeypatch.setattr(mo.apple_models, "load", load)
        mo.compute([mo.MOMENTUM_KEY], "AAPL", minute_bars(), momentum_model="nbeats")
        assert seen["key"] == "nbeats"

    def test_a_non_momentum_model_choice_falls_back_to_the_classifier(self, monkeypatch):
        seen = {}

        def load(key, ticker=None):
            seen["key"] = key
            return None

        monkeypatch.setattr(mo.apple_models, "load", load)
        mo.compute([mo.MOMENTUM_KEY], "AAPL", minute_bars(),
                   momentum_model=mo.apple_models.DAYRANGE_KEY)
        assert seen["key"] == mo.apple_models.PERSISTENCE_KEY


class TestProfileRangeOverlay:
    """The LevelsML pack reduced to levels.

    `profile_model` is stubbed here for the same reason the momentum bundle is:
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


class TestLiveOverlays:
    def make_state(self, bars):
        return SimpleNamespace(
            symbol="AAPL", daily_bars=[], bars=bars, model_overlay_cache=None
        )

    def test_the_answer_is_cached_until_a_new_bar_arrives(self, monkeypatch):
        calls = []

        def fake_compute(*args, **kwargs):
            calls.append(kwargs)
            return {"items": [], "notes": []}

        monkeypatch.setattr(mo, "compute", fake_compute)
        bars = minute_bars(n=40)
        state = self.make_state(bars)

        mo.live_overlays(state, bars, [mo.MOMENTUM_KEY])
        mo.live_overlays(state, bars, [mo.MOMENTUM_KEY])
        assert len(calls) == 1

        mo.live_overlays(state, bars + minute_bars(n=1, base=210.0), [mo.MOMENTUM_KEY])
        assert len(calls) == 2

    def test_changing_the_selection_recomputes(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            mo, "compute", lambda *a, **k: calls.append(1) or {"items": [], "notes": []}
        )
        bars = minute_bars(n=40)
        state = self.make_state(bars)
        mo.live_overlays(state, bars, [mo.MOMENTUM_KEY])
        mo.live_overlays(state, bars, [mo.MOMENTUM_KEY, mo.PROFILE_RANGE_KEY])
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
    return {"kind": "event", "key": mo.MOMENTUM_KEY,
            "group": mo.OVERLAYS[mo.MOMENTUM_KEY].label,
            "label": "→ positive 90%", "ts": BARS[50]["t"],
            "color": "#26c6a2", "icon": "▲", "price": 200.0, "dash": "dot",
            "note": "why", "forward": False}


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
        assert marks[0].name == "Momentum regime changes"

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

    def test_overlay_x_max_ignores_items_that_do_not_reach_forward(self):
        last = pd.Timestamp(BARS[-1]["t"])
        assert overlay_x_max([band(), level(), moment()], last) == last

    def test_overlay_x_max_handles_a_naive_reference(self):
        naive = pd.Timestamp(BARS[-1]["t"]).tz_localize(None)
        assert overlay_x_max([window()], naive).tzinfo is None
