"""The GoodMoments mirror: the cleaning, the live tape's seam, the analyst
window, the bundle -- and, where the notebook's data is on this machine, every
feature, every forecast and the option positioning against what the notebook's
own package built.

The live path's claim is that a minute's features read SIP through 15 bars
back and IEX through the bar itself, joined, and nothing else of the session;
with the seam at 0 they are the notebook's own. The synthetic tests pin the
join; the notebook-backed ones pin the arithmetic to its panel.
"""

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

G = pytest.importorskip("agent_stonks.goodmoments_model")

NOTEBOOK = Path("/Users/aleksandra/Documents/playground/Code/FinNotebooks/GoodMoments/data")
TZ = "America/New_York"


# --- synthetic tapes ------------------------------------------------------------


def _bars(day: str, minutes, seed: int = 0, price: float = 100.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    stamps = pd.DatetimeIndex(
        [pd.Timestamp(f"{day} 09:30", tz=TZ) + pd.Timedelta(minutes=int(m)) for m in minutes], name="ts"
    )
    n = len(stamps)
    close = price * np.exp(np.cumsum(rng.normal(0, 0.0007, n)))
    open_ = np.r_[price, close[:-1]]
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.0003, n)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.0003, n)))
    return pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close,
        "volume": rng.integers(1_000, 50_000, n).astype(float), "vwap": (high + low + close) / 3,
    }, index=stamps)


class TestCleaning:
    def test_a_full_session_is_kept_and_a_short_one_is_cut_at_13_00(self):
        full = _bars("2026-09-15", range(390))
        short = _bars("2026-09-16", range(300))
        grid, report = G.clean_sessions(pd.concat([full, short]))
        assert bool(report.loc["2026-09-15", "kept"]) and not bool(report.loc["2026-09-16", "kept"])
        assert grid[grid["date"] == "2026-09-16"]["minute"].max() == G.HALF_DAY_MINUTES - 1

    def test_a_missing_minute_takes_the_last_close_and_no_volume(self):
        bars = _bars("2026-09-15", [m for m in range(390) if m != 100])
        grid, _ = G.clean_sessions(bars)
        row = grid[grid["minute"] == 100].iloc[0]
        before = grid[grid["minute"] == 99].iloc[0]
        assert row["volume"] == 0 and row["filled"]
        assert row["open"] == row["high"] == row["low"] == row["close"] == before["close"]


class TestLiveSeam:
    """At the close of bar t: SIP through t-15, IEX after it (volume x k)."""

    K, SIGMA = 23.0, 0.0006

    def _tapes(self):
        sip = G.session_tape(_bars("2026-09-15", range(390), seed=1))
        iex = G.session_tape(_bars("2026-09-15", range(390), seed=2))
        return sip, iex

    def test_every_minute_is_the_notebooks_arithmetic_on_the_joined_tape(self):
        sip, iex = self._tapes()
        live = G.session_stats(sip, iex, self.K, self.SIGMA, seam=G.L)
        for t in list(range(0, 20)) + list(range(20, 389, 23)):
            j = t - G.L

            def joined(field, scale=1.0):
                s, i = getattr(sip, field), getattr(iex, field) * scale
                head = s[:max(j + 1, 0)]
                tail = i[max(j + 1, 0):t + 1]
                pad = np.full(389 - t, 0.0 if field == "volume" else i[t])
                return np.concatenate([head, tail, pad])

            tape = G.Tape(joined("open"), joined("high"), joined("low"), joined("close"),
                          joined("volume", self.K), joined("vwap"))
            ref = G.session_stats(tape, None, self.K, self.SIGMA, seam=0)
            for name in ("c", "s_open", "hi", "lo", "vwap", "cum_var", "vol_15",
                         "bvc_sess", "clv_sess", "dev_poc"):
                assert live[name][t] == pytest.approx(ref[name][t], rel=1e-12, abs=1e-15), (name, t)

    def test_before_bar_15_the_session_is_iexs_alone(self):
        sip, iex = self._tapes()
        live = G.session_stats(sip, iex, self.K, self.SIGMA, seam=G.L)
        assert live["s_open"][14] == iex.open[0] and live["s_open"][15] == sip.open[0]
        assert live["hi"][14] == iex.high[:15].max()

    def test_the_price_is_iexs_last_trade(self):
        sip, iex = self._tapes()
        live = G.session_stats(sip, iex, self.K, self.SIGMA, seam=G.L)
        assert np.array_equal(live["c"], iex.close)


class TestAnalysts:
    def test_a_note_counts_from_its_stamp_for_ninety_days(self):
        raw = pd.DataFrame(
            {"currentPriceTarget": [200.0, 0.0, 300.0]},
            index=pd.DatetimeIndex(
                ["2026-06-01 10:00:30", "2026-09-15 08:00", "2026-09-15 10:00:30"], name="GradeDate"
            ),
        )
        events = G.analyst_events(raw)
        day = pd.Timestamp("2026-09-15")
        ts = G.bar_close_times(day, np.array([29, 30]))       # 10:00 and 10:01 closes
        gap = G.pt_gap(ts, np.array([100.0, 100.0]), events)
        # 10:00: the June note has aged out, the 0 target is no target, the
        # 10:00:30 note is not out yet -- nothing; 10:01: that note alone.
        assert np.isnan(gap[0])
        assert gap[1] == pytest.approx(np.log(3.0))


class TestBundle:
    @pytest.fixture
    def store(self, tmp_path, monkeypatch):
        lgb = pytest.importorskip("lightgbm")
        joblib = pytest.importorskip("joblib")
        rng = np.random.default_rng(0)

        def booster(cols):
            X = pd.DataFrame(rng.normal(size=(400, len(cols))), columns=cols)
            y = (X.iloc[:, 0] > 0).astype(int)
            return lgb.train({"objective": "binary", "verbose": -1, "num_leaves": 3},
                             lgb.Dataset(X, y), num_boost_round=3)

        path = tmp_path / "goodmoments_AAPL.joblib"
        monkeypatch.setenv(G.MODEL_PATH_ENV, str(path))
        G.reset_bundle_cache()
        yield path, booster, joblib
        G.reset_bundle_cache()

    def test_it_loads_with_its_twin_and_refuses_without(self, store):
        path, booster, joblib = store
        main_cols = ["p_reach", "frac_left", "d_poc"]
        joblib.dump({"model": booster(main_cols), "features": main_cols}, path)
        assert G.load_bundle("AAPL") is None
        assert "export_timevol" in G.unavailable_reason("AAPL")
        G.reset_bundle_cache()
        joblib.dump({"model": booster(["frac_left", "p_reach"]), "features": ["frac_left", "p_reach"]},
                    G.timevol_path(path))
        bundle = G.load_bundle("AAPL")
        assert bundle is not None and bundle["kind"] == "goodmoments"
        assert not bundle["reads_options"] and not bundle["reads_analysts"]

    def test_a_feature_this_app_does_not_compute_is_refused(self, store):
        path, booster, joblib = store
        joblib.dump({"model": booster(["p_reach", "mom_5m"]), "features": ["p_reach", "mom_5m"]}, path)
        joblib.dump({"model": booster(["frac_left", "p_reach"]), "features": ["frac_left", "p_reach"]},
                    G.timevol_path(path))
        assert G.load_bundle("AAPL") is None


# --- against the notebook ----------------------------------------------------------

needs_notebook = pytest.mark.skipif(
    not (NOTEBOOK / "panel.parquet").exists(), reason="the GoodMoments notebook's data is not on this machine"
)


@pytest.fixture(scope="module")
def notebook():
    minutes = pd.read_parquet(NOTEBOOK / "minutes.parquet")
    return {
        "minutes": minutes,
        "sessions": pd.read_parquet(NOTEBOOK / "sessions.parquet"),
        "daily": G.daily_candles(minutes),
        "positioning": pd.read_parquet(NOTEBOOK / "options" / "positioning.parquet"),
        "events": G.analyst_events(pd.read_parquet(NOTEBOOK / "analysts_raw.parquet")),
        "panel": pd.read_parquet(NOTEBOOK / "panel.parquet"),
    }


def _position_row(pos: pd.DataFrame, day: pd.Timestamp) -> dict:
    p = pos.loc[day]
    return {"spot": p["spot"], "iv_30d": p["iv_30d"], "pc_volume": p["pc_volume"],
            "gex": [p[f"gex_{k}"] for k in range(len(G.GEX_GRID))]}


@needs_notebook
class TestAgainstTheNotebook:
    DAYS = ["2024-06-03", "2025-03-14", "2025-11-20", "2026-04-07", "2026-09-30", "2026-10-16"]

    def test_the_cleaning_and_the_daily_candles_are_the_notebooks(self, notebook):
        raw = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(NOTEBOOK / "raw" / "AAPL_*.parquet")))])
        grid, report = G.clean_sessions(raw.sort_index()[G.BAR_COLS])
        cols = ["date", "minute", *G.BAR_COLS]
        assert np.array_equal(grid[cols[2:]].to_numpy(float), notebook["minutes"][cols[2:]].to_numpy(float))
        assert report["kept"].equals(notebook["sessions"]["kept"])
        daily = pd.read_parquet(NOTEBOOK / "daily.parquet")
        assert np.array_equal(notebook["daily"].to_numpy(), daily[notebook["daily"].columns].to_numpy(),
                              equal_nan=True)

    @pytest.mark.parametrize("day", DAYS)
    def test_every_feature_and_forecast_is_the_notebooks(self, notebook, day):
        bundle = G.load_bundle("AAPL")
        if bundle is None:
            pytest.skip("the GoodMoments bundle is not installed")
        d = pd.Timestamp(day)
        drow, norms = G.day_inputs(notebook["minutes"], notebook["sessions"], notebook["daily"], d)
        f = G.session_features(
            G._tape_of(notebook["minutes"], d), None, d, drow, norms,
            _position_row(notebook["positioning"], d), notebook["events"], seam=0,
        )
        ref = notebook["panel"][notebook["panel"]["date"] == d].set_index("minute")
        mine = f.loc[ref.index]
        for col in G.FEATURE_COLS:
            np.testing.assert_array_equal(mine[col].to_numpy(float), ref[col].to_numpy(float), err_msg=col)
        np.testing.assert_array_equal(
            bundle["model"].predict(mine[bundle["features"]]), bundle["model"].predict(ref[bundle["features"]])
        )

    @pytest.mark.parametrize("day", ["2024-09-16", "2025-06-02", "2026-03-10", "2026-08-14"])
    def test_the_option_row_is_the_notebooks(self, notebook, day):
        """`options.positioning` on the notebook's own option bars."""
        options = NOTEBOOK / "options"
        contracts = pd.read_parquet(options / "contracts.parquet")
        bars = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(options / "bars" / "*.parquet")))],
                         ignore_index=True)
        bars = bars.merge(contracts[["symbol", "type", "strike", "expiry"]], on="symbol")
        d = pd.Timestamp(day)
        sessions = pd.DatetimeIndex(notebook["daily"].index)
        prev = sessions[sessions < d].max()
        row = G.positioning_row(bars[bars["date"] <= prev], notebook["daily"], sessions, prev, d)
        ref = notebook["positioning"].loc[d]
        assert row["spot"] == ref["spot"]
        assert row["iv_30d"] == ref["iv_30d"]
        assert row["pc_volume"] == ref["pc_volume"]
        gex = np.array([ref[f"gex_{k}"] for k in range(len(G.GEX_GRID))])
        np.testing.assert_allclose(row["gex"], gex, rtol=1e-10)
