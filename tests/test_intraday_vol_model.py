"""IntradayVolatility in the app (agent_stonks/intraday_vol_model.py).

Three halves. The first reproduces the exporter's own numbers from the `check`
block every exported file carries -- the mirror contract, pinned to floating-
point precision, and skipped only when the export is not on this machine. The
second pins the envelope the charts draw on a hand-written model, so it needs
no file. The third pins how the day-range inputs refuse what they cannot use.
"""

import json
import math

import numpy as np
import pandas as pd
import pytest

from agent_stonks import intraday_vol_model as ivm

# A plausible shape (09:30 ~5x midday, a short close ramp) and HAR.
MODEL = {
    "shape": {
        "params": {"a": 0.6, "b": 3.0, "alpha": 0.45, "c": 0.5, "kappa": 12.0},
        "t_domain": [2.5, 387.5],
    },
    "day_range": {
        "coef": {"const": -1.0, "lr_d": 0.2, "lr_w": 0.3, "lr_m": 0.3, "abs_gap": 5.0},
    },
}
SESSION = "2026-08-07"


def daily_bars(n=30, end=SESSION, width=0.02, close=100.0):
    """`n` completed business days ending the day before `end`."""
    days = pd.bdate_range(end=pd.Timestamp(end) - pd.Timedelta(days=1), periods=n)
    return [
        {"t": str(d.date()), "o": close, "h": close * (1 + width), "l": close, "c": close}
        for d in days
    ]


def exported(ticker):
    model = ivm.load(ticker)
    if model is None:
        pytest.skip(f"no IntradayVolatility export at {ivm.model_path(ticker)}")
    return model


@pytest.mark.parametrize("ticker", ivm.TICKERS)
class TestAgainstTheExporter:
    def test_the_day_range_features_and_forecast_reproduce_exactly(self, ticker):
        model = exported(ticker)
        check = model["check"]
        features = ivm.day_range_features(
            check["daily_bars"], check["session_date"], check["open"]
        )
        for name, value in check["features"].items():
            assert features[name] == pytest.approx(value, rel=0, abs=1e-12), name
        assert ivm.predict_log_range(model, features) == pytest.approx(
            check["pred_log_range"], rel=0, abs=1e-12
        )

    def test_the_shape_reproduces_exactly(self, ticker):
        model = exported(ticker)
        check = model["check"]
        np.testing.assert_allclose(
            ivm.volatility_shape(model, check["shape_minutes"]),
            check["shape_values"], rtol=0, atol=1e-12,
        )

    def test_the_fitted_shape_is_an_l_with_a_close_ramp(self, ticker):
        """What notebook 02 measured, as the exported curve has to reproduce it."""
        w = ivm.volatility_shape(exported(ticker))
        assert int(np.argmax(w)) == 0 and w[0] == pytest.approx(1.0)
        assert w[180] < 0.5 * w[0]          # midday is a fraction of the open
        assert w[389] > w[330]              # and it turns up into the close


class TestShape:
    def test_the_shape_peaks_at_one(self):
        w = ivm.volatility_shape(MODEL)
        assert w.max() == pytest.approx(1.0)
        assert len(w) == ivm.SESSION_MINUTES + 1

    def test_the_curve_is_held_flat_outside_the_fitted_bin_centres(self):
        """Fitted at 2.5 ... 387.5 minutes; the auction minute is not extrapolated."""
        w = ivm.volatility_shape(MODEL, [0, 1, 2])
        assert w[0] == w[1] == w[2]


class TestEnvelope:
    def test_the_upper_edge_tops_out_at_the_high_and_the_lower_at_the_low(self):
        upper, lower = ivm.envelope(MODEL, 200.0, 206.0, 197.0)
        assert upper.max() == pytest.approx(206.0)
        assert lower.min() == pytest.approx(197.0)
        assert (upper >= 200.0).all() and (lower <= 200.0).all()

    def test_its_width_follows_the_volatility_shape(self):
        upper, lower = ivm.envelope(MODEL, 200.0, 206.0, 197.0)
        width = upper - lower
        np.testing.assert_allclose(width / width.max(), ivm.volatility_shape(MODEL))
        assert width[195] < width[389] < width[0]

    def test_each_side_keeps_its_own_distance(self):
        """An asymmetric forecast -- the high twice as far as the low -- stays so."""
        upper, lower = ivm.envelope(MODEL, 200.0, 206.0, 197.0)
        np.testing.assert_allclose(upper - 200.0, 2 * (200.0 - lower))

    def test_an_open_outside_the_forecast_is_clamped_into_it(self):
        upper, lower = ivm.envelope(MODEL, 207.0, 206.0, 197.0)
        assert upper.max() == pytest.approx(206.0)
        assert lower.min() == pytest.approx(197.0)
        assert (upper >= lower).all()


class TestDayRange:
    def test_features_from_a_flat_history(self):
        feats = ivm.day_range_features(daily_bars(width=0.02), SESSION, 101.0)
        expected = math.log(math.log(1.02))
        assert feats["lr_d"] == pytest.approx(expected)
        assert feats["lr_w"] == pytest.approx(expected)
        assert feats["lr_m"] == pytest.approx(expected)
        assert feats["abs_gap"] == pytest.approx(math.log(101.0 / 100.0))

    def test_the_sessions_own_bar_cannot_leak_into_its_forecast(self):
        bars = daily_bars()
        leaked = bars + [{"t": SESSION, "o": 100, "h": 150, "l": 50, "c": 140}]
        assert ivm.day_range_features(leaked, SESSION, 100.0) == ivm.day_range_features(
            bars, SESSION, 100.0
        )

    def test_too_little_history_is_refused(self):
        with pytest.raises(ValueError, match="22 completed daily bars"):
            ivm.day_range_features(daily_bars(n=10), SESSION, 100.0)

    def test_a_missing_open_is_refused(self):
        with pytest.raises(ValueError, match="opening price"):
            ivm.day_range_features(daily_bars(), SESSION, 0.0)

    def test_the_implied_extremes_split_the_range_evenly_around_the_open(self):
        high, low = ivm.predicted_extremes(MODEL, daily_bars(), SESSION, 100.0)
        feats = ivm.day_range_features(daily_bars(), SESSION, 100.0)
        assert math.log(high / low) == pytest.approx(
            math.exp(ivm.predict_log_range(MODEL, feats))
        )
        assert high * low == pytest.approx(100.0 ** 2)


class TestLoading:
    def test_a_missing_file_is_unavailable(self, tmp_path):
        assert ivm._build(tmp_path / "nope.json") is None

    def test_a_file_without_the_model_is_unavailable(self, tmp_path):
        path = tmp_path / "intravol_AAPL.json"
        path.write_text(json.dumps({"shape": {"params": {"a": 1}}}))
        assert ivm._build(path) is None

    def test_a_complete_file_loads(self, tmp_path):
        path = tmp_path / "intravol_AAPL.json"
        path.write_text(json.dumps(MODEL))
        assert ivm._build(path)["shape"]["params"]["alpha"] == 0.45
