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
C = pytest.importorskip("agent_stonks.goodmoments_context")

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
        "trades": rng.integers(0, 400, n).astype(float),
    }, index=stamps)


class TestCleaning:
    def test_a_full_session_is_kept_and_a_short_one_is_cut_at_13_00(self):
        full = _bars("2026-09-15", range(390))
        short = _bars("2026-09-16", range(300))
        grid, report = G.clean_sessions(pd.concat([full, short]))
        assert bool(report.loc["2026-09-15", "kept"]) and not bool(report.loc["2026-09-16", "kept"])
        assert grid[grid["date"] == "2026-09-16"]["minute"].max() == G.HALF_DAY_MINUTES - 1

    def test_an_exchange_half_day_is_not_kept_however_busy_after_13_00(self):
        """NVDA trades so much after a 13:00 close that a bar count alone
        would keep the session (`data.early_closes`)."""
        grid, report = G.clean_sessions(_bars("2025-11-28", range(390)))
        assert not bool(report.loc["2025-11-28", "kept"])
        assert grid["minute"].max() == G.HALF_DAY_MINUTES - 1

    def test_a_context_series_keeps_a_thin_session_whole(self):
        grid, report = G.clean_sessions(_bars("2026-09-16", range(0, 390, 2)), min_bars=0)
        assert bool(report.loc["2026-09-16", "kept"]) and grid["minute"].max() == 389

    def test_trade_counts_ride_along_and_a_missing_minute_has_none(self):
        grid, _ = G.clean_sessions(_bars("2026-09-15", [m for m in range(390) if m != 100]))
        assert grid[grid["minute"] == 100]["trades"].iloc[0] == 0
        tape = G.session_tape(_bars("2026-09-15", [m for m in range(390) if m != 7]))
        assert tape.trades is not None and tape.trades[7] == 0

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
            for name in ("c", "s_open", "hi", "lo", "ref_1h", "vwap", "cum_vol", "cum_var", "vol_15",
                         "var_60", "bvc_1h", "bvc_sess", "clv_sess", "dev_poc"):
                assert live[name][t] == pytest.approx(ref[name][t], rel=1e-12, abs=1e-15), (name, t)

    def test_the_trade_size_is_sips_own_15_bars_back(self):
        """IEX's trade size is no guide to SIP's: the joined tape reads SIP's
        last 15 bars through t-15, against that minute's norm."""
        sip, iex = self._tapes()
        live = G.session_stats(sip, iex, self.K, self.SIGMA, seam=G.L)
        own = G.session_stats(sip, None, self.K, self.SIGMA, seam=0)
        assert np.isnan(live["size_15"][:G.L]).all() and (live["size_at"][:G.L] == -1).all()
        t = np.arange(G.L, 390)
        np.testing.assert_array_equal(live["size_15"][t], own["size_15"][t - G.L])
        np.testing.assert_array_equal(live["size_at"][t], t - G.L)

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
    def test_the_counts_read_the_last_30_and_7_days(self):
        raw = pd.DataFrame(
            {"currentPriceTarget": [200.0, 210.0, 190.0], "Action": ["up", "main", "down"],
             "priceTargetAction": ["Raises", "Raises", "Lowers"]},
            index=pd.DatetimeIndex(["2026-08-20 09:00", "2026-09-10 09:00", "2026-09-14 09:00"], name="GradeDate"),
        )
        events = G.analyst_events(raw)
        counts = G.analyst_counts(G.bar_close_times(pd.Timestamp("2026-09-15"), np.array([0])), events)
        # 30 days back reaches the August upgrade; 7 days, the last two notes.
        assert counts["rating_30d"][0] == 0.0          # +1 in August, -1 on the 14th
        assert counts["target_7d"][0] == 0.0           # raised on the 10th, lowered on the 14th
        assert counts["n_actions_7d"][0] == 2.0

    def test_nvda_targets_before_the_split_are_divided(self):
        raw = pd.DataFrame({"currentPriceTarget": [1200.0, 150.0]},
                           index=pd.DatetimeIndex(["2024-06-07 09:00", "2024-06-11 09:00"], name="GradeDate"))
        assert G.analyst_events(raw, "NVDA")["price_target"].tolist() == [120.0, 150.0]
        assert G.analyst_events(raw, "AAPL")["price_target"].tolist() == [1200.0, 150.0]

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

    def test_another_tickers_own_feature_is_refused(self, store):
        """NVDA's own features are NVDA's definitions: AAPL has no peers
        list, no QQQ beta of its own."""
        path, booster, joblib = store
        joblib.dump({"model": booster(["p_reach", "qqq_today"]), "features": ["p_reach", "qqq_today"]}, path)
        joblib.dump({"model": booster(["frac_left", "p_reach"]), "features": ["frac_left", "p_reach"]},
                    G.timevol_path(path))
        assert G.load_bundle("AAPL") is None


class TestNeeds:
    def test_nvda_reads_the_market_the_peers_the_calendar_and_the_trades(self):
        need = C.needs("NVDA", ["qqq_today", "resid_smh_today", "peers_today", "days_to_earn",
                                "d_pin", "trade_size_15m", "p_reach"])
        assert need.context == ("QQQ", "SMH", "AMD", "AVGO", "TSM", "MU")
        assert need.earnings and need.surface and need.trades

    def test_mus_clocks_read_smh_alone(self):
        need = C.needs("MU", ["frac_left", "p_reach_60m", "p_reach_smh"])
        assert need == C.Needs(context=("SMH",))

    def test_the_shared_features_read_nothing_here(self):
        assert not C.needs("AAPL", G.FEATURE_COLS[:3] + ["d_poc", "gex"]).any


class TestEarningsClock:
    SESSIONS = pd.DatetimeIndex(pd.bdate_range("2026-07-01", "2026-10-30")).as_unit("ns")

    def test_a_report_after_the_close_is_the_next_sessions(self):
        earn = C.report_sessions(["2026-08-26T16:20:00-04:00", "2026-10-28T08:00:00-04:00"], self.SESSIONS)
        assert list(earn) == [pd.Timestamp("2026-08-27"), pd.Timestamp("2026-10-28")]

    def test_sessions_to_and_since_are_capped(self):
        earn = pd.DatetimeIndex(["2026-08-27", "2026-10-28"])
        on = C.earnings_clock(earn, self.SESSIONS, pd.Timestamp("2026-08-27"))
        assert on == {"days_to_earn": 0.0, "days_since_earn": 0.0}
        later = C.earnings_clock(earn, self.SESSIONS, pd.Timestamp("2026-09-01"))
        assert later == {"days_to_earn": 15.0, "days_since_earn": 3.0}
        near = C.earnings_clock(earn, self.SESSIONS, pd.Timestamp("2026-10-26"))
        assert near["days_to_earn"] == 2.0


# --- against the notebook ----------------------------------------------------------


def _notebook_dir(ticker: str) -> Path:
    """`goodmoments.config.DATA_DIR`: AAPL keeps the original cache, every
    other ticker its own folder under it."""
    return NOTEBOOK if ticker == "AAPL" else NOTEBOOK / ticker


_NOTEBOOKS: dict = {}


def notebook(ticker: str) -> dict:
    """One ticker's notebook data, read once per run (AMD's panel is 90 MB)."""
    if ticker not in _NOTEBOOKS:
        folder = _notebook_dir(ticker)
        if not (folder / "panel.parquet").exists():
            pytest.skip(f"the GoodMoments notebook's {ticker} data is not on this machine")
        minutes = pd.read_parquet(folder / "minutes.parquet")
        surface = folder / "options" / "surface.parquet"
        earnings = folder / "earnings.parquet"
        _NOTEBOOKS[ticker] = {
            "folder": folder,
            "minutes": minutes,
            "sessions": pd.read_parquet(folder / "sessions.parquet"),
            "daily": G.daily_candles(minutes),
            "positioning": pd.read_parquet(folder / "options" / "positioning.parquet"),
            "surface": pd.read_parquet(surface) if surface.exists() else None,
            "earnings": list(pd.read_parquet(earnings)["Earnings Date"]) if earnings.exists() else None,
            "events": G.analyst_events(pd.read_parquet(folder / "analysts_raw.parquet"), ticker),
            "panel": pd.read_parquet(folder / "panel.parquet", filters=[("date", "in", [pd.Timestamp(d) for d in DAYS[ticker]])]),
            "context": {},
        }
    return _NOTEBOOKS[ticker]


def _context_minutes(nb: dict, symbol: str) -> pd.DataFrame:
    if symbol not in nb["context"]:
        nb["context"][symbol] = pd.read_parquet(nb["folder"] / "context" / f"{symbol}_minutes.parquet")
    return nb["context"][symbol]


def _position_row(nb: dict, day: pd.Timestamp) -> dict:
    p = nb["positioning"].loc[day]
    row = {"spot": p["spot"], "call_wall": p["call_wall"], "put_wall": p["put_wall"], "iv_30d": p["iv_30d"],
           "pc_volume": p["pc_volume"], "gex": [p[f"gex_{k}"] for k in range(len(G.GEX_GRID))]}
    if nb["surface"] is not None and day in nb["surface"].index:
        row.update({k: nb["surface"].at[day, k] for k in ("pin", "call_vol_rel")})
    return row


def _own_session(bundle: dict, ticker: str, nb: dict, day: pd.Timestamp, row: dict):
    """The own features' inputs from the notebook's own context grids,
    earnings dates and session list -- what `goodmoments_context.own_day`
    builds from the app's caches."""
    need = bundle["needs"]
    if not need.any:
        return None
    kept = pd.DatetimeIndex(nb["sessions"].index[nb["sessions"]["kept"]])
    own = C.OwnDay(ticker=ticker, day=day)
    for symbol in need.context:
        m = _context_minutes(nb, symbol)
        own.context[symbol] = C.context_day(symbol, m, G.daily_candles(m), kept, day)
    if "SMH" in own.context and ticker in C.OWN_FEATURES:
        daily = nb["daily"]
        dly = pd.concat([daily[daily.index < day], pd.DataFrame(index=pd.DatetimeIndex([day], name="date"))])
        own.values.update(C.smh_daily_values(dly, own.context["SMH"].daily, day))
    if need.earnings:
        # The notebook counts sessions on its own data's list (it ends 16 Oct 2026).
        sessions = pd.DatetimeIndex(nb["sessions"].index).as_unit("ns")
        own.values.update(C.earnings_clock(C.report_sessions(nb["earnings"], sessions), sessions, day))
    stats = {s: C.context_stats(own.context[s], G._tape_of(_context_minutes(nb, s), day), None, 0)
             for s in need.context}
    return C.OwnSession(own=own, stats=stats, position=row)


# Per ticker: AAPL's original six; AMD's first modelled month, results mornings
# (2025-02-05, 2025-11-05, 2026-08-05 -- reported the evening before), a Monday
# and the last session of the fit.
DAYS = {
    "AAPL": ["2024-06-03", "2025-03-14", "2025-11-20", "2026-04-07", "2026-09-30", "2026-10-16"],
    "AMD": ["2024-03-15", "2025-02-05", "2025-11-05", "2026-04-06", "2026-08-05", "2026-10-16"],
}
OPTION_DAYS = {
    "AAPL": ["2024-09-16", "2025-06-02", "2026-03-10", "2026-08-14"],
    "AMD": ["2024-09-16", "2025-06-02", "2026-03-10", "2026-10-16"],
}


class TestAgainstTheNotebook:
    @pytest.mark.parametrize("ticker", sorted(DAYS))
    def test_the_cleaning_and_the_daily_candles_are_the_notebooks(self, ticker):
        nb = notebook(ticker)
        raw = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(nb["folder"] / "raw" / f"{ticker}_*.parquet")))])
        grid, report = G.clean_sessions(raw.sort_index()[G.BAR_COLS])
        cols = ["date", "minute", *G.BAR_COLS]
        assert np.array_equal(grid[cols[2:]].to_numpy(float), nb["minutes"][cols[2:]].to_numpy(float))
        assert report["kept"].equals(nb["sessions"]["kept"])
        daily = pd.read_parquet(nb["folder"] / "daily.parquet")
        assert np.array_equal(nb["daily"].to_numpy(), daily[nb["daily"].columns].to_numpy(),
                              equal_nan=True)

    @pytest.mark.parametrize("ticker,day", [(t, d) for t, days in DAYS.items() for d in days])
    def test_every_feature_and_forecast_is_the_notebooks(self, ticker, day):
        bundle = G.load_bundle(ticker)
        if bundle is None:
            pytest.skip(f"the GoodMoments {ticker} bundle is not installed")
        nb = notebook(ticker)
        d = pd.Timestamp(day)
        drow, norms = G.day_inputs(nb["minutes"], nb["sessions"], nb["daily"], d)
        row = _position_row(nb, d)
        f = G.session_features(
            G._tape_of(nb["minutes"], d), None, d, drow, norms, row, nb["events"], seam=0,
            own=_own_session(bundle, ticker, nb, d, row),
        )
        ref = nb["panel"][nb["panel"]["date"] == d].set_index("minute")
        mine = f.loc[ref.index]
        # Every shared column the panel has (AAPL's and AMD's predate the two
        # volatility clocks), and every column the model reads.
        cols = [c for c in G.FEATURE_COLS if c in ref] + [c for c in bundle["features"] if c not in G.FEATURE_COLS]
        assert set(bundle["features"]) <= set(cols)
        for col in cols:
            np.testing.assert_array_equal(mine[col].to_numpy(float), ref[col].to_numpy(float), err_msg=col)
        for model, cols in (("model", "features"), ("timevol", "timevol_features")):
            np.testing.assert_array_equal(
                bundle[model].predict(mine[bundle[cols]]), bundle[model].predict(ref[bundle[cols]])
            )

    @pytest.mark.parametrize("ticker,day", [(t, d) for t, days in OPTION_DAYS.items() for d in days])
    def test_the_option_row_is_the_notebooks(self, ticker, day):
        """`options.positioning` on the notebook's own option bars."""
        nb = notebook(ticker)
        options = nb["folder"] / "options"
        contracts = pd.read_parquet(options / "contracts.parquet")
        # AMD's notebook saved two expiries with no bars at all, typed object:
        # in pandas 3 they would turn the whole volume column into objects.
        frames = [pd.read_parquet(f) for f in sorted(glob.glob(str(options / "bars" / "*.parquet")))]
        bars = pd.concat([f for f in frames if len(f)], ignore_index=True)
        bars = bars.merge(contracts[["symbol", "type", "strike", "expiry"]], on="symbol")
        d = pd.Timestamp(day)
        sessions = pd.DatetimeIndex(nb["daily"].index)
        prev = sessions[sessions < d].max()
        row = G.positioning_row(bars[bars["date"] <= prev], nb["daily"], sessions, prev, d)
        ref = nb["positioning"].loc[d]
        assert row["spot"] == ref["spot"]
        assert row["call_wall"] == ref["call_wall"]
        assert row["put_wall"] == ref["put_wall"]
        assert row["iv_30d"] == ref["iv_30d"]
        assert row["pc_volume"] == ref["pc_volume"]
        gex = np.array([ref[f"gex_{k}"] for k in range(len(G.GEX_GRID))])
        np.testing.assert_allclose(row["gex"], gex, rtol=1e-10)
        if nb["surface"] is not None:
            # `nvda.surface`'s pin and call-volume norm, on the same bars.
            surface = C.surface_row(bars[bars["date"] <= prev], nb["daily"], sessions, prev, d, G.HALF_LIFE)
            assert surface["pin"] == nb["surface"].at[d, "pin"]
            assert surface["call_vol_rel"] == nb["surface"].at[d, "call_vol_rel"]

    def test_amd_reads_the_call_wall_and_the_hour(self):
        bundle = G.load_bundle("AMD")
        if bundle is None:
            pytest.skip("the GoodMoments AMD bundle is not installed")
        assert bundle["ticker"] == "AMD" and bundle["reads_walls"]
        assert {"d_call_wall", "mom_1h", "bvc_1h", "rel_vol", "vol_regime"} <= set(bundle["features"])
        assert bundle["timevol_features"] == ["frac_left", "p_reach"]

    def test_nvda_and_mu_read_what_their_notebooks_saved(self):
        nvda, mu = G.load_bundle("NVDA"), G.load_bundle("MU")
        if nvda is None or mu is None:
            pytest.skip("the GoodMoments NVDA or MU bundle is not installed")
        assert set(C.OWN_FEATURES["NVDA"]) <= set(nvda["features"]) and len(nvda["features"]) == 38
        assert nvda["option_keys"] == ("call_vol_rel", "pin", "put_wall") and nvda["reads_analysts"]
        assert mu["features"] == ["frac_left", "p_reach_60m", "p_reach_smh"]
        assert mu["needs"] == C.Needs(context=("SMH",)) and not mu["reads_options"] and not mu["reads_analysts"]
