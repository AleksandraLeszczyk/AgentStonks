"""SimLab's model drift view (simlab/drift.py).

The grouping and the cutoff split are pure and pinned on hand-made rows. The
training cutoffs are read from each model's saved metadata, pinned on dicts in
that shape. Each model's scoring runs on a synthetic store with the model
stubbed -- the real bundles have their own tests -- so what is pinned is the
arithmetic of the metric and the point-in-time rule: a session is scored from
what existed before it and compared with what the day then did.

The two drift tests are pinned twice over: against series whose answer is not in
doubt (a ramp, a step, a flat line), and -- in `TestAgainstExactNull` -- against
the exact permutation null they approximate, enumerated here rather than taken
from SciPy, which this package does not depend on.
"""
import itertools
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
    # A normal day is 2% of the price here, so the held-out MAE is 38.5% of one.
    "sim_date_forecast": {"date": "2026-08-28", "prev_avg": 300.0, "adr14_abs": 6.0},
}


class TestTraining:
    def test_day_range_cutoffs_are_the_daily_fit_and_the_opening_ridge(self):
        training = dr.dayrange_training_from_metadata(DAYRANGE_META)
        cutoffs = {c.date: c for c in training["cutoffs"]}
        assert set(cutoffs) == {"2026-07-09", "2026-08-27"}
        assert "35" in cutoffs["2026-08-27"].note
        refs = {r.metric: r.value for r in training["references"]}
        assert refs["mae"] == 0.0077 and refs["bias_high"] == -0.0017 and refs["mae_usd"] == 2.11

    def test_the_headline_reference_is_the_ml_models_tab_number(self):
        """The dotted line on the chart is what the other page prints, so the two
        are read off the same arithmetic rather than each doing its own."""
        training = dr.dayrange_training_from_metadata(DAYRANGE_META)
        refs = {r.metric: r for r in training["references"]}
        assert refs["mae_pct_adr"].value == pytest.approx(38.5)
        assert refs["mae_pct_adr"].value == pytest.approx(
            dr.model_catalogue.dayrange_error_pct_adr(DAYRANGE_META)
        )

    def test_a_sidecar_without_a_simulation_day_has_no_adr_reference(self):
        """Nothing to divide by, so no line -- the series is still scored."""
        meta = {k: v for k, v in DAYRANGE_META.items() if k != "sim_date_forecast"}
        assert "mae_pct_adr" not in {r.metric for r in
                                     dr.dayrange_training_from_metadata(meta)["references"]}

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
        # Against the session's own 14-day range ($2.00), not the one day's ADR
        # the saved file records: a $1.00 miss is half a normal day.
        assert row["mae_pct_adr"] == pytest.approx(50.0)

    def test_a_session_with_no_average_range_is_left_unscored(self, store, monkeypatch, stubbed):
        """A gap in the series rather than a division by zero dressed as a number."""
        dayrange = pytest.importorskip("agent_stonks.dayrange_model")
        monkeypatch.setattr(dayrange, "forecast_session", lambda *a, **k: {
            "pred_high": 105.0, "pred_low": 99.0, "prev_avg": 100.0,
            "adr14_abs": 0.0, "or_high": 101.2, "or_low": 100.8,
        })
        rows = dr.evaluate_dayrange(TICKER, FEED)["rows"]
        assert rows and all(r["mae_pct_adr"] is None and r["mae_usd"] is not None for r in rows)

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


# --- has it moved? ----------------------------------------------------------


def series(values, start=date(2026, 1, 5), metric="mae") -> list[dict]:
    """One value per weekday from `start`, which is a Monday."""
    out, day = [], start
    for value in values:
        out.append({"date": day.isoformat(), metric: value})
        day += timedelta(days=1 if day.weekday() < 4 else 3)
    return out


class TestTrendTest:
    def test_a_ramp_is_a_trend_and_carries_its_slope(self):
        got = dr.trend_test(series([float(i) for i in range(12)]), "mae")
        assert got.statistic == pytest.approx(1.0) and got.p < 1e-4
        assert got.significant and got.direction == "up"
        # The slope is in the metric's units per 30 calendar days, and this ramp
        # climbs one unit per session -- of which a 30-day stretch holds ~21.
        assert 20.0 < got.effect < 23.0

    def test_a_falling_series_is_the_same_test_the_other_way(self):
        got = dr.trend_test(series([float(-i) for i in range(12)]), "mae")
        assert got.statistic == pytest.approx(-1.0) and got.direction == "down"
        assert got.effect < 0

    def test_noise_around_a_level_is_not_a_trend(self):
        values = [1.0, 3.0, 2.0, 4.0, 1.5, 3.5, 2.5, 1.0, 3.0, 2.0, 4.0, 2.5]
        got = dr.trend_test(series(values), "mae")
        assert not got.significant and got.direction == "flat"

    def test_a_flat_series_has_nothing_to_test_rather_than_a_p_value(self):
        got = dr.trend_test(series([2.0] * 12), "mae")
        assert got.p is None and got.statistic is None and "same value" in got.note

    def test_too_few_sessions_says_so_and_names_the_bar(self):
        got = dr.trend_test(series([1.0, 2.0, 3.0]), "mae")
        assert got.p is None and got.n == 3
        assert str(dr.MIN_TREND_N) in got.note

    def test_unscored_sessions_are_left_out(self):
        rows = series([float(i) for i in range(12)])
        rows += [{"date": "2026-02-02", "mae": None}]
        assert dr.trend_test(rows, "mae").n == 12


class TestSplitTest:
    CUTOFFS = [dr.Cutoff("2026-01-16", "Trained through")]

    def rows(self, inside, after):
        return series(inside) + series(after, start=date(2026, 1, 19))

    def test_a_step_up_after_the_cutoff_is_found(self):
        got = dr.split_test(self.rows([1.0] * 4 + [1.2, 0.9, 1.1, 1.0],
                                      [3.0, 3.2, 2.9, 3.1, 3.0, 2.8, 3.3, 3.1]),
                            "mae", self.CUTOFFS)
        assert got.significant and got.direction == "up"
        assert got.statistic == pytest.approx(1.0)  # every after-session above every inside one
        assert got.effect == pytest.approx(3.05 - 1.0, abs=0.06)

    def test_the_same_distribution_either_side_is_not_a_change(self):
        both = [1.0, 1.4, 0.8, 1.2, 1.1, 0.9, 1.3, 1.0]
        got = dr.split_test(self.rows(both, both), "mae", self.CUTOFFS)
        assert got.p == pytest.approx(1.0, abs=0.05) and not got.significant

    def test_a_model_with_no_cutoff_cannot_be_split(self):
        got = dr.split_test(self.rows([1.0] * 8, [2.0] * 8), "mae", [])
        assert got.p is None and "no training cutoff" in got.note

    def test_a_store_entirely_on_one_side_says_which_side(self):
        after = dr.split_test(series([1.0] * 12, start=date(2026, 1, 19)), "mae", self.CUTOFFS)
        assert after.p is None and after.note == "all 12 sessions after the cutoff"
        inside = dr.split_test(series([1.0] * 6), "mae", self.CUTOFFS)
        assert inside.p is None and inside.note == "all 6 sessions inside the cutoff"

    def test_too_few_on_one_side_counts_both(self):
        rows = series([1.0] * 8) + series([2.0] * 3, start=date(2026, 1, 19))
        assert dr.split_test(rows, "mae", self.CUTOFFS).note == (
            f"8 inside, 3 after — needs {dr.MIN_SPLIT_N} each side"
        )


class TestAgainstExactNull:
    """The normal approximations against the permutation nulls they stand in for."""

    @staticmethod
    def _kendall_s(values):
        values = np.asarray(values, float)
        return float(np.triu(np.sign(values[None, :] - values[:, None]), 1).sum())

    def test_mann_kendall_matches_the_exact_permutation_p(self):
        values = [0.4, -1.2, 0.9, 0.1, 1.6, -0.3, 1.1, 2.0]
        observed = abs(self._kendall_s(values))
        orderings = list(itertools.permutations(values))
        exact = sum(abs(self._kendall_s(o)) >= observed for o in orderings) / len(orderings)
        assert dr.trend_test(series(values), "mae").p == pytest.approx(exact, abs=0.02)

    def test_mann_whitney_matches_the_exact_permutation_p(self):
        inside = [1.0, 1.4, 0.8, 1.2, 1.1, 0.9]
        after = [1.3, 1.9, 1.5, 1.2, 2.1, 1.6, 1.7]
        rows = series(inside) + series(after, start=date(2026, 1, 19))
        got = dr.split_test(rows, "mae", TestSplitTest.CUTOFFS)
        both = np.array(after + inside, float)
        ranks = dr._ranks(both)
        n1, n2 = len(after), len(inside)
        observed = abs(float(ranks[:n1].sum()) - n1 * (n1 + 1) / 2 - n1 * n2 / 2)
        hits = total = 0
        for combo in itertools.combinations(range(n1 + n2), n1):
            u = sum(ranks[i] for i in combo) - n1 * (n1 + 1) / 2
            hits += abs(u - n1 * n2 / 2) >= observed - 1e-9
            total += 1
        assert got.p == pytest.approx(hits / total, abs=0.02)


class TestVerdict:
    def test_rising_is_worse_only_where_lower_is_better(self):
        lower = dr.Metric("e", "Error")
        higher = dr.Metric("c", "Correlation", better="higher")
        bias = dr.Metric("b", "Bias", better="zero")
        assert dr.worsened(lower, 0.4) is True and dr.worsened(lower, -0.4) is False
        assert dr.worsened(higher, 0.4) is False and dr.worsened(higher, -0.4) is True
        assert dr.worsened(bias, 0.4) is None
        assert dr.worsened(lower, None) is None

    def test_assess_returns_both_tests_and_the_split(self):
        rows = series([1.0] * 8) + series([3.0] * 8, start=date(2026, 1, 19))
        got = dr.assess(rows, dr.Metric("mae", "MAE"), TestSplitTest.CUTOFFS)
        assert got["n"] == 16
        assert got["trend"].kind == dr.TREND and got["split"].kind == dr.SPLIT
        assert got["parts"]["inside"]["mean"] == 1.0 and got["parts"]["after"]["mean"] == 3.0


# --- every model at once -----------------------------------------------------


class TestPairs:
    def test_every_model_is_paired_with_the_symbols_it_exists_for(self, store):
        got = dr.pairs(FEED, [TICKER])
        assert got == [(key, TICKER) for key in dr.MODELS]
        assert [k for k, _ in got] == list(dr.MODELS)  # model-major, catalogue order

    def test_a_symbol_the_store_does_not_carry_is_dropped(self, store):
        assert dr.pairs(FEED, ["NVDA"]) == []

    def test_a_per_ticker_model_only_takes_its_own_symbols(self, store, monkeypatch):
        monkeypatch.setattr(sim_data, "STORE_DIR", dr.sim_data.STORE_DIR)
        sim_data._write_gz(sim_data.daily_path("NVDA", FEED), {"symbol": "NVDA", "bars": []})
        keys = {k for k, t in dr.pairs(FEED, ["NVDA"])}
        assert keys == {k for k, m in dr.MODELS.items() if m.tickers is None}

    def test_the_default_symbols_are_the_ones_a_model_was_fitted_for(self, store):
        assert dr.default_symbols(FEED) == [TICKER]


class TestHeadline:
    def test_every_model_names_a_metric_it_actually_has(self):
        for model in dr.MODELS.values():
            assert model.headline in {m.key for m in model.metrics}
            assert model.headline_metric.key == model.headline

    def test_every_model_says_how_it_lines_up_with_the_ml_models_tab(self):
        for model in dr.MODELS.values():
            assert model.catalogue_metric


class TestSignature:
    """A scoring depends on two things, and both belong in its cache key."""

    def test_the_signature_carries_the_bars_and_the_model_file(self, store):
        bars, model = dr.signature(dr.model_catalogue.OPEN_PROFILE_KEY, TICKER, FEED)
        assert bars == dr.store_signature(TICKER, FEED)
        assert model == dr.model_signature(dr.model_catalogue.OPEN_PROFILE_KEY, TICKER)

    def test_retraining_a_model_changes_it_without_the_store_moving(self, store, monkeypatch, tmp_path):
        retrained = tmp_path / "open_profile_lgbm.json.gz"
        retrained.write_bytes(b"x")
        monkeypatch.setenv("OPEN_PROFILE_MODEL", str(retrained))
        before = dr.signature(dr.model_catalogue.OPEN_PROFILE_KEY, TICKER, FEED)
        retrained.write_bytes(b"xx")
        after = dr.signature(dr.model_catalogue.OPEN_PROFILE_KEY, TICKER, FEED)
        assert after != before and after[0] == before[0]  # the bars did not move

    def test_a_model_with_no_files_on_disk_is_an_empty_signature(self, monkeypatch, tmp_path):
        monkeypatch.setenv("OPEN_PROFILE_MODEL", str(tmp_path / "nothing.json.gz"))
        assert dr.model_signature(dr.model_catalogue.OPEN_PROFILE_KEY, TICKER) == ()
