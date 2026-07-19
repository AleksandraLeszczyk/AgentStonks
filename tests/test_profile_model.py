"""Tests for the ML predicted price profile (agent_stonks/profile_model.py)."""

import math
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from agent_stonks import profile_model as pm
from agent_stonks.charts import build_chart

TODAY = "2026-07-20"


def make_daily_bars(n=300, seed=1, last_close=None):
    rng = np.random.default_rng(seed)
    closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    if last_close is not None:
        closes = closes * (last_close / closes[-1])
    dates = pd.bdate_range(end="2026-07-17", periods=n)
    bars = []
    for d, c in zip(dates, closes):
        o = c * (1 + rng.normal(0, 0.003))
        h = max(o, c) * 1.005
        low = min(o, c) * 0.995
        bars.append({
            "t": d.strftime("%Y-%m-%dT05:00:00Z"),
            "o": o, "h": h, "l": low, "c": c,
            "v": float(rng.integers(1_000_000, 5_000_000)),
        })
    return bars


class TestComputeFeatures:
    def test_returns_expected_hand_computed_values(self):
        bars = make_daily_bars()
        open_px = bars[-1]["c"] * 1.004
        feats = pm.compute_features(bars, open_px, today=TODAY)
        assert feats is not None
        prev = bars[-1]
        assert feats["open_gap"] == pytest.approx(math.log(open_px / prev["c"]))
        assert feats["prev_range"] == pytest.approx((prev["h"] - prev["l"]) / prev["c"])
        assert feats["prev_ret"] == pytest.approx(math.log(prev["c"] / prev["o"]))
        assert feats["ret_5d"] == pytest.approx(math.log(prev["c"] / bars[-6]["c"]))
        # 2026-07-20 is a Monday
        assert feats["day_of_week"] == 0

    def test_all_pack_features_finite_with_enough_history(self):
        bars = make_daily_bars()
        feats = pm.compute_features(bars, bars[-1]["c"], today=TODAY)
        needed = [
            "prev_range", "prev_ret", "prev_volume_z", "prev_close_pos_in_range",
            "ret_5d", "ret_20d", "vol_5d", "vol_20d", "atr_14",
            "dist_high_20d", "dist_low_20d", "dist_high_252d", "dist_low_252d",
            "trend_5d", "trend_20d", "day_of_week", "open_gap",
        ]
        assert all(np.isfinite(feats[c]) for c in needed)

    def test_excludes_todays_forming_bar(self):
        bars = make_daily_bars()
        forming = {"t": f"{TODAY}T05:00:00Z", "o": 999.0, "h": 999.0, "l": 999.0,
                   "c": 999.0, "v": 1.0}
        open_px = bars[-1]["c"]
        with_forming = pm.compute_features(bars + [forming], open_px, today=TODAY)
        without = pm.compute_features(bars, open_px, today=TODAY)
        assert with_forming == without

    def test_insufficient_history_returns_none(self):
        assert pm.compute_features(make_daily_bars(n=10), 100.0, today=TODAY) is None

    def test_bad_open_returns_none(self):
        bars = make_daily_bars()
        assert pm.compute_features(bars, 0.0, today=TODAY) is None
        assert pm.compute_features(bars, None, today=TODAY) is None


class TestDensityFromQuantiles:
    P = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]

    def test_integrates_to_one_and_nonnegative(self):
        q = np.linspace(-80, 80, 11)
        grid, dens = pm.density_from_quantiles(q, self.P)
        assert (dens >= 0).all()
        assert np.trapezoid(dens, grid) == pytest.approx(1.0, abs=0.03)

    def test_mode_where_quantiles_bunch(self):
        # mass concentrated near +50 bps: quantiles bunch there
        q = np.array([-120, -80, -20, 30, 42, 48, 52, 58, 70, 110, 150], dtype=float)
        grid, dens = pm.density_from_quantiles(q, self.P)
        assert 30 <= grid[dens.argmax()] <= 70

    def test_zero_outside_padded_range(self):
        q = np.linspace(-50, 50, 11)
        grid, dens = pm.density_from_quantiles(q, self.P)
        assert dens[0] == pytest.approx(0.0, abs=1e-9)
        assert dens[-1] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.skipif(pm.load_pack() is None, reason="model pack not available")
class TestWithTrainedPack:
    def test_predict_quantiles_monotone_and_sane(self):
        pack = pm.load_pack()
        feats = pm.compute_features(make_daily_bars(), 100.0 * 1.002, today=TODAY)
        # scale synthetic prices so open_gap etc. stay realistic
        q = pm.predict_quantiles(pack, feats)
        assert q is not None and len(q) == len(pack["p_levels"])
        assert (np.diff(q) >= 0).all()
        assert abs(q).max() < 2000  # within ±20% of the open

    def test_predicted_open_profile_and_cache(self, monkeypatch):
        monkeypatch.setattr(
            pm.clock, "now",
            lambda: datetime(2026, 7, 20, 14, 0, tzinfo=timezone.utc),
        )
        bars = make_daily_bars()
        sym_state = SimpleNamespace(daily_bars=bars, predicted_profile_cache=None)
        intraday = [{"t": f"{TODAY}T13:30:00Z", "o": bars[-1]["c"], "h": 1, "l": 1,
                     "c": 1, "v": 1}]
        prof = pm.predicted_open_profile(sym_state, intraday)
        assert prof is not None
        assert prof["density"].max() == pytest.approx(1.0)
        assert len(prof["prices"]) == len(prof["density"])
        assert prof["prices"].min() < prof["open"] < prof["prices"].max()
        again = pm.predicted_open_profile(sym_state, intraday)
        assert again is prof  # served from the per-(day, open) cache

    def test_missing_daily_bars_gives_none(self):
        sym_state = SimpleNamespace(daily_bars=[], predicted_profile_cache=None)
        assert pm.predicted_open_profile(sym_state, []) is None


class TestChartIntegration:
    SESSION_START = datetime(2024, 1, 15, 13, 20, tzinfo=timezone.utc)
    BARS = [
        {"t": "2024-01-15T14:00:00Z", "o": 100.0, "h": 102.0, "l": 99.0, "c": 101.0, "v": 5000},
        {"t": "2024-01-15T14:01:00Z", "o": 101.0, "h": 103.0, "l": 100.5, "c": 102.0, "v": 3000},
    ]
    TRADES = [
        {"p": 100.5, "s": 100, "t": "2024-01-15T14:00:10Z"},
        {"p": 101.0, "s": 200, "t": "2024-01-15T14:00:30Z"},
        {"p": 101.5, "s": 150, "t": "2024-01-15T14:01:05Z"},
    ]

    @staticmethod
    def make_profile(center=101.0, width_bps=60.0):
        grid = np.linspace(-3 * width_bps, 3 * width_bps, 121)
        dens = np.exp(-0.5 * (grid / width_bps) ** 2)
        return {
            "prices": center * np.exp(grid / 1e4),
            "density": dens / dens.max(),
            "quantiles_bps": grid[::12],
            "open": center,
            "poc_price": center,
        }

    def _trace_names(self, fig):
        return [t.name for t in fig.data if t.name]

    def test_predicted_trace_drawn_with_trades(self):
        fig = build_chart(self.BARS, [], self.TRADES, "AAPL", self.SESSION_START,
                          predicted_profile=self.make_profile())
        assert "ML predicted profile" in self._trace_names(fig)

    def test_predicted_trace_drawn_without_trades(self):
        fig = build_chart(self.BARS, [], [], "AAPL", self.SESSION_START,
                          predicted_profile=self.make_profile())
        assert "ML predicted profile" in self._trace_names(fig)

    def test_no_predicted_trace_by_default(self):
        fig = build_chart(self.BARS, [], self.TRADES, "AAPL", self.SESSION_START)
        assert "ML predicted profile" not in self._trace_names(fig)

    def test_fit_mixture_to_predicted_profile(self):
        prof = self.make_profile(center=101.0, width_bps=60.0)
        fig = build_chart(self.BARS, [], self.TRADES, "AAPL", self.SESSION_START,
                          mixture_distribution="gaussian", mixture_max_components=1,
                          predicted_profile=prof, mixture_fit_target="predicted")
        gauss = [t for t in fig.data if t.name and t.name.startswith("G1")]
        assert gauss, "expected a fitted Gaussian component trace"
        # fitted location should sit at the predicted profile's mode, not at
        # the (offset) trade histogram
        mu = float(gauss[0].name.split("μ=")[1].split()[0])
        assert mu == pytest.approx(101.0, abs=0.05)

    def test_fit_to_live_still_works_with_profile_shown(self):
        fig = build_chart(self.BARS, [], self.TRADES, "AAPL", self.SESSION_START,
                          mixture_distribution="cauchy", mixture_max_components=2,
                          predicted_profile=self.make_profile(),
                          mixture_fit_target="live")
        assert any(t.name and t.name.startswith("C") for t in fig.data)
