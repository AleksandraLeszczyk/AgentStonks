"""SimLab's model drift view (simlab/drift.py).

The grouping and the cutoff split are pure and pinned on hand-made rows. The
training cutoffs are read from each model's saved metadata, pinned on dicts in
that shape. Each model's scoring runs on a synthetic store with the model
stubbed -- the real bundles have their own tests -- so what is pinned is the
arithmetic of the metric and the point-in-time rule: a session is scored from
what existed before it and compared with what the day then did.
"""
import math
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from simlab import data as sim_data
from simlab import drift as dr

TICKER = "AAPL"
FEED = sim_data.DEFAULT_FEED


# --- grouping ---------------------------------------------------------------


def rows(values: dict) -> list[dict]:
    return [{"date": d, "mae": v} for d, v in values.items()]


class TestAggregate:
    VALUES = {
        "2026-08-03": 1.0, "2026-08-04": 3.0, "2026-08-07": 2.0,   # week of 3 Aug
        "2026-08-10": 5.0,                                          # week of 10 Aug
        "2026-08-18": None,                                         # unscored
    }

    def test_by_day_is_one_group_per_scored_session(self):
        groups = dr.aggregate(rows(self.VALUES), "mae", dr.DAY)
        assert [g["start"] for g in groups] == ["2026-08-03", "2026-08-04", "2026-08-07", "2026-08-10"]
        assert all(g["n"] == 1 for g in groups)

    def test_by_week_groups_monday_to_sunday(self):
        groups = dr.aggregate(rows(self.VALUES), "mae", dr.WEEK)
        assert [g["start"] for g in groups] == ["2026-08-03", "2026-08-10"]
        first = groups[0]
        assert (first["n"], first["mean"], first["min"], first["max"]) == (3, 2.0, 1.0, 3.0)
        assert first["end"] == "2026-08-07"  # the last session in it, not the Sunday
        assert "3 Aug" in first["label"]

    def test_the_split_puts_a_cutoff_day_inside_the_training_data(self):
        cutoffs = [dr.Cutoff("2026-07-09", "Daily models"), dr.Cutoff("2026-08-04", "Ridge")]
        split = dr.split_by_cutoff(rows(self.VALUES), "mae", cutoffs)
        assert split["inside"] == {"mean": 2.0, "n": 2}
        assert split["after"] == {"mean": 3.5, "n": 2}

    def test_no_cutoffs_means_nothing_is_known_to_be_in_training(self):
        split = dr.split_by_cutoff(rows(self.VALUES), "mae", [])
        assert split["inside"]["n"] == 0 and split["after"]["n"] == 4


# --- training cutoffs from saved metadata -----------------------------------


DAYRANGE_META = {
    "daily_fit_through": "2026-07-09",
    "opening_correction": True,
    "opening_fit_sessions": 35,
    "held_out": "2026-08-28",
    "test_metrics_ensemble": {
        "mae_mean": 0.0077, "mae_y_high": 0.0078, "mae_y_low": 0.0076,
        "bias_y_high": -0.0017, "bias_y_low": -0.0006, "mae_usd_mean": 2.11,
    },
}


class TestTraining:
    def test_day_range_cutoffs_are_the_daily_fit_and_the_opening_ridge(self):
        training = dr.dayrange_training_from_metadata(DAYRANGE_META)
        cutoffs = {c.date: c for c in training["cutoffs"]}
        assert set(cutoffs) == {"2026-07-09", "2026-08-27"}
        assert "35" in cutoffs["2026-08-27"].note
        refs = {r.metric: r.value for r in training["references"]}
        assert refs["mae"] == 0.0077 and refs["bias_high"] == -0.0017 and refs["mae_usd"] == 2.11

    def test_a_bundle_without_an_opening_ridge_has_only_the_daily_cutoff(self):
        training = dr.dayrange_training_from_metadata({**DAYRANGE_META, "opening_correction": False})
        assert [c.date for c in training["cutoffs"]] == ["2026-07-09"]

    def test_the_ridge_cutoff_is_the_last_weekday_before_the_held_out_day(self):
        training = dr.dayrange_training_from_metadata({**DAYRANGE_META, "held_out": "2026-08-10"})
        assert "2026-08-07" in [c.date for c in training["cutoffs"]]  # a Monday -> Friday

    def test_missing_metadata_is_no_cutoffs_rather_than_an_error(self):
        assert dr.dayrange_training_from_metadata({}) == {"cutoffs": [], "references": []}

    def test_intraday_vol_cutoffs_are_its_two_samples(self):
        model = {"shape": {"sample": ["2023-01-03", "2026-09-11"]},
                 "day_range": {"sample": ["2019-02-04", "2026-09-10"], "residual_sd": 0.3}}
        training = dr.intraday_vol_training_from_model(model)
        assert sorted(c.date for c in training["cutoffs"]) == ["2026-09-10", "2026-09-11"]
        [ref] = training["references"]
        assert ref.metric == "abs_log_range_err"
        assert ref.value == pytest.approx(0.3 * math.sqrt(2 / math.pi))


# --- scoring on a synthetic store --------------------------------------------


DAYS = [date(2026, 8, 3), date(2026, 8, 4)]


def _minute_bars(day: date, n: int = 60, base: float = 101.0) -> list[dict]:
    open_utc = datetime(day.year, day.month, day.day, 13, 30, tzinfo=timezone.utc)
    return [
        {"t": (open_utc + timedelta(minutes=i)).strftime("%Y-%m-%dT%H:%M:%SZ"),
         "o": base, "h": base + 0.2 + 0.01 * (i % 7), "l": base - 0.2 - 0.01 * (i % 5),
         "c": base, "v": 1000.0 + i}
        for i in range(n)
    ]


def _daily_bars(end: date, n: int, width: float = 0.02) -> list[dict]:
    days = pd.bdate_range(end=end, periods=n)
    return [{"t": str(d.date()), "o": 100.0, "h": 100.0 * (1 + width), "l": 100.0,
             "c": 100.0, "v": 1e6} for d in days]


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(sim_data, "STORE_DIR", tmp_path / "store")
    daily = _daily_bars(DAYS[-1], 40)
    # The scored days' own bars: opened at 101, traded 100..106.
    for bar in daily:
        if bar["t"] in {str(d) for d in DAYS}:
            bar.update(o=101.0, h=106.0, l=100.0, c=103.0)
    sim_data._write_gz(sim_data.daily_path(TICKER, FEED), {"symbol": TICKER, "bars": daily})
    for day in DAYS:
        sim_data._write_gz(sim_data.bars_path(TICKER, day, FEED), _minute_bars(day))
    return tmp_path


class TestStore:
    def test_stored_minute_days_skip_empty_files(self, store):
        sim_data._write_gz(sim_data.bars_path(TICKER, date(2026, 8, 5), FEED), [])
        assert dr.stored_minute_days(TICKER, FEED) == DAYS

    def test_stored_symbols_on_a_feed(self, store):
        assert dr.stored_symbols(FEED) == [TICKER]
        assert dr.stored_symbols("sip") == []

    def test_the_signature_changes_when_a_day_is_added(self, store):
        before = dr.store_signature(TICKER, FEED)
        sim_data._write_gz(sim_data.bars_path(TICKER, date(2026, 8, 6), FEED), _minute_bars(date(2026, 8, 6)))
        assert dr.store_signature(TICKER, FEED) != before


class TestDayRange:
    @pytest.fixture()
    def stubbed(self, monkeypatch):
        dayrange = pytest.importorskip("agent_stonks.dayrange_model")
        seen = []
        monkeypatch.setattr(dr.apple_models, "load", lambda *a, **k: {"opening_minutes": 5})

        def forecast(bundle, history, opening, session_date, open_price=None):
            seen.append({"history_last": history.index.max(), "opening": len(opening),
                         "open": open_price, "day": pd.Timestamp(session_date)})
            return {"pred_high": 105.0, "pred_low": 99.0, "prev_avg": 100.0,
                    "adr14_abs": 2.0, "or_high": 101.2, "or_low": 100.8}

        monkeypatch.setattr(dayrange, "forecast_session", forecast)
        return seen

    def test_each_minute_session_is_scored_against_its_own_high_and_low(self, store, stubbed):
        result = dr.evaluate_dayrange(TICKER, FEED)
        assert [r["date"] for r in result["rows"]] == [str(d) for d in DAYS]
        row = result["rows"][0]
        assert row["bias_high"] == pytest.approx(math.log(105.0 / 106.0))
        assert row["bias_low"] == pytest.approx(math.log(99.0 / 100.0))
        assert row["mae"] == pytest.approx(
            (abs(math.log(105.0 / 106.0)) + abs(math.log(99.0 / 100.0))) / 2
        )
        assert row["mae_usd"] == pytest.approx((1.0 + 1.0) / 2)

    def test_the_forecast_sees_only_the_past_plus_the_open(self, store, stubbed):
        dr.evaluate_dayrange(TICKER, FEED)
        for call in stubbed:
            assert call["history_last"] < call["day"]
            assert call["opening"] == 5 and call["open"] == 101.0

    def test_a_missing_bundle_is_a_note(self, store, monkeypatch):
        monkeypatch.setattr(dr.apple_models, "load", lambda *a, **k: None)
        result = dr.evaluate_dayrange(TICKER, FEED)
        assert result["rows"] == [] and result["notes"]


INTRAVOL = {
    "shape": {"params": {"a": 0.6, "b": 3.0, "alpha": 0.45, "c": 0.5, "kappa": 12.0},
              "t_domain": [2.5, 387.5], "sample": ["2023-01-03", "2026-09-11"]},
    "day_range": {"coef": {"const": -1.0, "lr_d": 0.2, "lr_w": 0.3, "lr_m": 0.3,
                           "abs_gap": 5.0},
                  "sample": ["2019-02-04", "2026-09-11"], "residual_sd": 0.3},
}


class TestIntradayVol:
    @pytest.fixture()
    def stubbed(self, monkeypatch):
        monkeypatch.setattr(dr.intraday_vol_model, "load", lambda ticker=None: INTRAVOL)

    def test_every_daily_session_with_a_months_history_is_scored(self, store, stubbed):
        result = dr.evaluate_intraday_vol(TICKER, FEED)
        # 40 stored daily bars; the first 22 are history for the rest.
        assert len(result["rows"]) == 40 - 22

    def test_the_error_is_in_log_day_range(self, store, stubbed):
        row = next(r for r in dr.evaluate_intraday_vol(TICKER, FEED)["rows"] if r["date"] == "2026-07-31")
        daily = sim_data.load_daily_bars(TICKER, FEED)
        i = [b["t"] for b in daily].index("2026-07-31")
        feats = dr.intraday_vol_model.day_range_features(daily[:i], "2026-07-31", 100.0)
        predicted = dr.intraday_vol_model.predict_log_range(INTRAVOL, feats)
        assert row["log_range_bias"] == pytest.approx(predicted - math.log(math.log(1.02)))
        assert row["abs_log_range_err"] == pytest.approx(abs(row["log_range_bias"]))

    def test_the_shape_is_scored_only_where_minute_bars_exist(self, store, stubbed):
        by_date = {r["date"]: r for r in dr.evaluate_intraday_vol(TICKER, FEED)["rows"]}
        assert by_date["2026-07-31"]["shape_corr"] is None
        assert -1.0 <= by_date["2026-08-04"]["shape_corr"] <= 1.0

    def test_sessions_all_inside_the_sample_say_so(self, store, stubbed):
        notes = dr.evaluate_intraday_vol(TICKER, FEED)["notes"]
        assert any("inside this model's training sample" in n for n in notes)


def _bar_at(day: date, minute: int, price: float, volume: float) -> dict:
    ts = datetime(day.year, day.month, day.day, 13, 30, tzinfo=timezone.utc) + timedelta(minutes=minute)
    return {"t": ts.strftime("%Y-%m-%dT%H:%M:%SZ"), "o": price, "h": price, "l": price,
            "c": price, "v": volume}


class TestOpenProfile:
    DAY = date(2026, 8, 3)

    def bars(self):
        return [_bar_at(self.DAY, 0, 99.0, 1.0), _bar_at(self.DAY, 1, 100.0, 2.0),
                _bar_at(self.DAY, 2, 101.0, 1.0)]

    def test_weights_are_the_training_scripts_trapezoid(self):
        np.testing.assert_allclose(dr.emd_weights([5, 50, 95]), [0.275, 0.45, 0.275])

    def test_realised_quantiles_interpolate_the_cumulative_volume(self):
        q = dr.realised_quantiles(self.bars(), 100.0, [5, 50, 95])
        rel = np.log(np.array([99.0, 100.0, 101.0]) / 100.0) * 1e4
        expected = np.interp([0.05, 0.5, 0.95], [0.25, 0.75, 1.0], rel)
        np.testing.assert_allclose(q, expected)

    def test_the_volume_inside_a_band(self):
        assert dr.volume_share_inside(self.bars(), 99.5, 100.5) == pytest.approx(0.5)

    def test_no_volume_is_no_profile(self):
        silent = [{**b, "v": 0.0} for b in self.bars()]
        assert dr.realised_quantiles(silent, 100.0, [5, 50, 95]) is None

    def test_sessions_are_scored_from_the_past_and_their_own_volume(self, store, monkeypatch):
        seen = []
        monkeypatch.setattr(dr.profile_model, "load_pack", lambda: {"p_levels": [5, 50, 95]})
        monkeypatch.setattr(
            dr.profile_model, "compute_features",
            lambda daily, open_px, today: seen.append((max(b["t"] for b in daily), today, open_px))
            or {"x": 1.0},
        )
        monkeypatch.setattr(dr.profile_model, "predict_quantiles",
                            lambda pack, feats: np.array([-50.0, 0.0, 50.0]))
        result = dr.evaluate_open_profile(TICKER, FEED)
        assert [r["date"] for r in result["rows"]] == [str(d) for d in DAYS]
        for last_history, today, open_px in seen:
            assert last_history < today and open_px == 101.0
        row = result["rows"][0]
        assert row["emd_bps"] > 0 and 0.0 <= row["inside_band"] <= 100.0

    def test_cutoff_and_references_from_the_pack(self):
        pack = {"p_levels": [5, 50, 95], "metadata": {
            "date_range": ["2023-11-13", "2026-07-17"], "universe": ["SPY", "AAPL"],
            "walk_forward_emd_bps": {"lgbm live-features": 76.7, "ATR climatology": 77.8},
        }}
        training = dr.open_profile_training_from_pack(pack)
        assert [c.date for c in training["cutoffs"]] == ["2026-07-17"]
        refs = {(r.metric, r.label): r.value for r in training["references"]}
        assert refs[("emd_bps", "Walk-forward EMD")] == 76.7
        assert refs[("inside_band", "Nominal")] == 90.0


class TestCatalogue:
    def test_every_model_has_metrics_and_callables(self):
        for model in dr.MODELS.values():
            assert model.metrics and callable(model.evaluate) and callable(model.training)
            assert len({m.key for m in model.metrics}) == len(model.metrics)

    def test_real_training_metadata_reads_without_error(self):
        """Against whatever bundles this machine has; an absent file is no cutoffs."""
        for key, model in dr.MODELS.items():
            for ticker in (model.tickers or ("AAPL",)):
                training = model.training(ticker)
                assert set(training) == {"cutoffs", "references"}
