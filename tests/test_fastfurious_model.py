"""The Fast&Furious mirror: the live tape's two cut lines, the cleaning, the bundle,
and -- where the notebook's data is on this machine -- the features, the forecasts
and the entry threshold against what the notebook's own package built.

The live path's whole claim is that a minute's forecast reads SIP through 15
bars back and IEX through the bar itself, and nothing else of the session. The
synthetic tests pin where the fetches are cut; the notebook-backed ones pin
that a session forecast whole (a replay) and one forecast minute by minute from
tapes cut at those lines (live) are the same numbers, and that both are the
notebook's.
"""

from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

F = pytest.importorskip("agent_stonks.fastfurious_model")

NOTEBOOK = Path("/Users/aleksandra/Documents/playground/Code/FinNotebooks/Fast&Furious/data/AAPL")
TZ = "America/New_York"


# --- synthetic tapes ------------------------------------------------------------


def _bars(day: str, minutes, price: float = 100.0) -> pd.DataFrame:
    """Regular-session minute bars of one day at the given minutes."""
    stamps = pd.DatetimeIndex(
        [pd.Timestamp(f"{day} 09:30", tz=TZ) + pd.Timedelta(minutes=int(m)) for m in minutes], name="ts"
    )
    n = len(stamps)
    close = price + np.arange(n) * 0.01
    return pd.DataFrame({
        "open": close, "high": close + 0.02, "low": close - 0.02, "close": close,
        "volume": np.full(n, 1000.0), "vwap": close,
    }, index=stamps)


class TestCleaning:
    def test_a_full_session_is_kept_and_a_short_or_late_one_is_not(self):
        full = _bars("2026-09-15", range(390))
        short = _bars("2026-09-16", range(380))
        late = _bars("2026-09-17", range(1, 390))
        tapes = F.process(pd.concat([full, short, late]), pd.concat([full, short, late]))
        report = tapes.report
        assert bool(report.loc["2026-09-15", "kept"])
        assert report.loc["2026-09-16", "reason"] == "short"
        assert report.loc["2026-09-17", "reason"] == "no_open_bar"
        assert list(tapes.kept) == [pd.Timestamp("2026-09-15")]
        # Every session SIP printed has a daily candle, kept or not: "yesterday"
        # is always the previous session.
        assert len(tapes.daily) == 3
        assert set(tapes.iex["date"]) == {pd.Timestamp("2026-09-15")}

    def test_a_half_day_is_not_kept_and_its_candle_ends_at_13_00(self):
        half = _bars("2026-11-27", range(390))  # the day after Thanksgiving
        tapes = F.process(half, half)
        assert tapes.report.loc["2026-11-27", "reason"] == "half_day"
        assert not len(tapes.kept)
        assert tapes.daily.loc["2026-11-27", "close"] == pytest.approx(half["close"].iloc[209])

    def test_a_vwap_outside_its_bar_is_replaced_by_the_close(self):
        bars = _bars("2026-09-15", range(390))
        bars.iloc[5, bars.columns.get_loc("vwap")] = 1e6
        clean = F.clean(bars)
        assert clean["vwap"].iloc[5] == clean["close"].iloc[5]

    def test_a_holiday_print_is_not_a_session(self):
        tapes = F.process(_bars("2026-09-07", range(390)), _bars("2026-09-07", range(390)))  # Labor Day
        assert not len(tapes.report)

    def test_bars_frame_keeps_the_regular_session_only(self):
        rows = [
            {"t": "2026-09-15T13:29:00Z", "o": 1, "h": 1, "l": 1, "c": 1, "v": 1, "vw": 1},
            {"t": "2026-09-15T13:30:00Z", "o": 2, "h": 2, "l": 2, "c": 2, "v": 2, "vw": 2},
            {"t": "2026-09-15T19:59:00Z", "o": 3, "h": 3, "l": 3, "c": 3, "v": 3, "vw": 3},
            {"t": "2026-09-15T20:00:00Z", "o": 4, "h": 4, "l": 4, "c": 4, "v": 4, "vw": 4},
        ]
        frame = F.bars_frame(rows)
        assert list(frame["close"]) == [2.0, 3.0]
        assert list(frame.columns) == F.BAR_COLUMNS


class TestLiveTapes:
    """Where today's two fetches are cut: SIP through bar t-15, IEX through t."""

    def _capture(self, monkeypatch, iex_last: int):
        asked = []

        def fake_fetch(symbol, start, end, key, secret, feed):
            asked.append((feed, start, end))
            return _bars("2026-10-13", range(iex_last + 1 if feed == "iex" else 0))

        monkeypatch.setattr(F, "_fetch", fake_fetch)
        return asked

    def test_sip_ends_fifteen_bars_back_and_iex_at_the_bar(self, monkeypatch):
        asked = self._capture(monkeypatch, iex_last=60)
        sip, iex, settled = F.live_tapes("AAPL", date(2026, 10, 13), 60, "k", "s")
        ends = {feed: end for feed, _, end in asked}
        assert ends["sip"].strftime("%H:%M") == "10:15"   # bar 45 = 60 - 15
        assert ends["iex"].strftime("%H:%M") == "10:30"   # bar 60
        assert settled

    def test_before_bar_15_there_is_no_sip_request(self, monkeypatch):
        asked = self._capture(monkeypatch, iex_last=10)
        F.live_tapes("AAPL", date(2026, 10, 13), 10, "k", "s")
        assert [feed for feed, _, _ in asked] == ["iex"]

    def test_a_late_iex_bar_is_asked_for_again_while_fresh(self, monkeypatch):
        asked = self._capture(monkeypatch, iex_last=59)  # bar 60 never arrives
        monkeypatch.setattr(F.time, "sleep", lambda s: None)
        now = datetime(2026, 10, 13, 14, 31, 5, tzinfo=timezone.utc)  # 10:31:05 ET

        class Clock(datetime):
            @classmethod
            def now(cls, tz=None):
                return now if tz is None else now.astimezone(tz)

        monkeypatch.setattr(F, "datetime", Clock)
        _, _, settled = F.live_tapes("AAPL", date(2026, 10, 13), 60, "k", "s")
        assert not settled
        assert [feed for feed, _, _ in asked].count("iex") == F._LIVE_RETRIES + 1
        asked.clear()
        F.live_tapes("AAPL", date(2026, 10, 13), 60, "k", "s", retries=0)
        assert [feed for feed, _, _ in asked].count("iex") == 1

    def test_no_credentials_is_refused(self, monkeypatch):
        monkeypatch.delenv("ALPACA_API_KEY", raising=False)
        monkeypatch.delenv("ALPACA_SECRET", raising=False)
        with pytest.raises(ValueError, match="credentials"):
            F.live_tapes("AAPL", date(2026, 10, 13), 60)


class TestHistoryCache:
    """The minute history grows a day at a time and is never thrown away for a
    request that merely does not overlap it -- a replayed session asks for its
    200 days and then for itself, one day past them."""

    def _fake(self, monkeypatch, tmp_path, scale: "dict | None" = None):
        monkeypatch.setattr(F, "HISTORY_DIR", tmp_path)
        calls = []

        def fetch(symbol, first, before, key, secret):
            calls.append((first, before))
            days = pd.bdate_range(first, before - timedelta(days=1))
            parts = [_bars(str(d.date()), range(390), (scale or {}).get(str(d.date()), 100.0)) for d in days]
            sip = pd.concat(parts) if parts else F._empty_bars()
            return {"sip": sip, "iex": sip.copy()}

        monkeypatch.setattr(F, "_fetch_stretch", fetch)
        return calls

    def test_the_session_after_the_history_fetches_only_from_the_seam(self, monkeypatch, tmp_path):
        calls = self._fake(monkeypatch, tmp_path)
        F.raw_history("AAPL", date(2026, 3, 1), date(2026, 3, 20), "k", "s")
        F.raw_history("AAPL", date(2026, 3, 20), date(2026, 3, 21), "k", "s")
        # The last cached session (Thursday 19 March) is the seam.
        assert calls == [(date(2026, 3, 1), date(2026, 3, 20)), (date(2026, 3, 19), date(2026, 3, 21))]
        window = F.raw_history("AAPL", date(2026, 3, 2), date(2026, 3, 21), "k", "s")
        assert len(calls) == 2
        days = set(window["sip"].index.strftime("%Y-%m-%d"))
        assert {"2026-03-02", "2026-03-19", "2026-03-20"} <= days

    def test_a_request_past_a_gap_fetches_the_gap_too(self, monkeypatch, tmp_path):
        calls = self._fake(monkeypatch, tmp_path)
        F.raw_history("AAPL", date(2026, 3, 1), date(2026, 3, 10), "k", "s")
        F.raw_history("AAPL", date(2026, 4, 1), date(2026, 4, 2), "k", "s")
        window = F.raw_history("AAPL", date(2026, 3, 1), date(2026, 4, 2), "k", "s")
        assert len(calls) == 2
        assert "2026-03-20" in set(window["sip"].index.strftime("%Y-%m-%d"))

    def test_a_split_at_the_seam_rebuilds_from_the_request(self, monkeypatch, tmp_path):
        calls = self._fake(monkeypatch, tmp_path)
        F.raw_history("AAPL", date(2026, 3, 1), date(2026, 3, 10), "k", "s")
        # The tape is now split-adjusted: every earlier close has moved.
        monkeypatch.setattr(F, "_fetch_stretch", lambda s, a, b, k, x: (
            calls.append((a, b)),
            {"sip": pd.concat([_bars(str(d.date()), range(390), 50.0) for d in pd.bdate_range(a, b - timedelta(days=1))]),
             "iex": pd.concat([_bars(str(d.date()), range(390), 50.0) for d in pd.bdate_range(a, b - timedelta(days=1))])},
        )[1])
        window = F.raw_history("AAPL", date(2026, 3, 1), date(2026, 3, 13), "k", "s")
        assert calls[-1] == (date(2026, 3, 1), date(2026, 3, 13))
        assert window["sip"]["close"].iloc[0] == pytest.approx(50.0)


@lru_cache(maxsize=None)
def _recipe(name: str) -> "dict | None":
    """A saved recipe's bundle: "amended" (what AAPL runs) or "original"
    (the notebook's own `fastfurious_AAPL`, which the app no longer reads)."""
    if name == "amended":
        return F.load_bundle("AAPL")
    return F._build_bundle(F.model_store.MODEL_DIR / "fastfurious_AAPL.joblib")


class TestBundle:
    def test_aapl_runs_the_amended_rerun_by_name(self, monkeypatch):
        monkeypatch.delenv("APPLE_FASTFURIOUS_MODEL", raising=False)
        monkeypatch.delenv("APPLE_FASTFURIOUS_MODEL_AAPL", raising=False)
        assert F.model_path("AAPL").name == "fastfurious_AAPL_amended.joblib"
        assert F.model_path("NVDA").name == "fastfurious_NVDA.joblib"
        monkeypatch.setenv("APPLE_FASTFURIOUS_MODEL_AAPL", "/elsewhere/ff.joblib")
        assert str(F.model_path("AAPL")) == "/elsewhere/ff.joblib"

    def test_the_shipped_bundle_reads_only_mirrored_columns(self):
        bundle = _recipe("amended")
        if bundle is None:
            pytest.skip("no Fast&Furious bundle on this machine")
        model = bundle["model"]
        assert bundle["kind"] == "fastfurious"
        assert set(model.momentum.cols) <= set(F.FEATURE_COLS) | F.ANALYST_COLS
        assert set(model.range.cols) <= set(F.FEATURE_COLS)
        assert bundle["reads_options"] and bundle["momentum_reads_options"] and bundle["reads_analysts"]
        assert model.momentum.cols == F.AMENDED_MOMENTUM_COLS
        assert model.range.cols == F.RANGE_COLS

    def test_the_original_recipe_still_loads(self):
        bundle = _recipe("original")
        if bundle is None:
            pytest.skip("no original Fast&Furious bundle on this machine")
        assert bundle["model"].momentum.cols == F.MOMENTUM_COLS
        assert bundle["reads_options"]
        assert not bundle["momentum_reads_options"] and not bundle["reads_analysts"]

    def test_a_half_day_has_no_forecast(self):
        bundle = F.load_bundle("AAPL") or {"path": "x", "model": None}
        with pytest.raises(ValueError, match="13:00"):
            F.session_forecasts(bundle, "AAPL", "2026-11-27")


class TestOptionCloses:
    """Which nights' option rows a session asks for."""

    def _day(self):
        kept = pd.DatetimeIndex(pd.bdate_range("2026-06-01", "2026-07-31"))
        report = pd.DataFrame({"kept": True}, index=kept)
        report.loc[pd.Timestamp("2026-07-15"), "kept"] = False  # a short session: a close, not kept
        daily = pd.DataFrame(index=kept)
        hist = F.Tapes(sip=None, iex=None, daily=daily, report=report)
        return F._Day(day=pd.Timestamp("2026-08-03"), history=hist, daily=daily, prev_close=kept[-1])

    def test_a_range_only_reader_asks_for_last_night_only(self):
        ctx_day = self._day()
        assert F._positioning_closes({"momentum_reads_options": False}, ctx_day) == [ctx_day.prev_close]

    def test_a_momentum_reader_asks_for_the_night_before_each_threshold_session(self):
        ctx_day = self._day()
        closes = F._positioning_closes({"momentum_reads_options": True}, ctx_day)
        kept = ctx_day.history.kept
        first = kept[-F.THRESHOLD_SESSIONS]
        assert closes[0] == ctx_day.daily.index[ctx_day.daily.index < first][-1]
        assert closes[-1] == ctx_day.prev_close
        assert pd.Timestamp("2026-07-15") in closes  # every close, the dropped sessions' too

    def test_a_momentum_reader_without_the_table_waits(self, monkeypatch):
        ctx_day = self._day()

        def failed(bundle, ctx, symbol, key, secret):
            ctx.positioning_problem = "Alpaca is down"

        monkeypatch.setattr(F, "_positioning", failed)
        with pytest.raises(F.InputsPending, match="Alpaca is down"):
            F._momentum_options({"momentum_reads_options": True}, ctx_day, "AAPL", None, None)
        # The original recipe forecasts without it (only its stop needs it).
        F._momentum_options({"reads_options": True, "momentum_reads_options": False}, ctx_day, "AAPL", None, None)


class TestAnalysts:
    RAW = pd.DataFrame(
        {"Firm": ["A", "B", "A", "C"], "ToGrade": ["Buy", "Hold", "Buy", "Sell"],
         "FromGrade": ["Hold", "Buy", "Buy", "Hold"], "Action": ["up", "down", "main", "down"],
         "priceTargetAction": ["Raises", "Lowers", "Raises", "Lowers"],
         "currentPriceTarget": [300.0, 250.0, 320.0, 0.0], "priorPriceTarget": [280.0, 260.0, 300.0, 0.0]},
        # Yahoo's stamps: naive UTC, in seconds since 2026.
        index=pd.DatetimeIndex(["2026-07-01 12:00:00", "2026-07-10 13:00:00", "2026-07-15 14:00:00",
                                "2026-07-21 14:00:00"], name="GradeDate").as_unit("s"),
    )

    def test_cleaning_stamps_in_ns_new_york_time(self):
        a = F.clean_analysts(self.RAW)
        assert a.index.unit == "ns" and str(a.index.tz) == TZ
        assert a.index[0] == pd.Timestamp("2026-07-01 08:00", tz=TZ)
        assert a["target_chg"].iloc[0] == pytest.approx(300 / 280 - 1)

    def test_days_since_a_change_and_the_consensus_read_only_notes_before(self):
        a = F.clean_analysts(self.RAW)
        days = pd.DatetimeIndex(["2026-07-21"]).as_unit("ns")
        T = F.origin_times(days)
        f = F.analyst_group(a, days, T, np.full((1, F.M), 300.0))
        # C's downgrade is stamped 10:00:00: the forecast made at 10:00 (t=29)
        # counts only notes stamped strictly before it, the one at 10:01 sees it.
        b_down = pd.Timestamp("2026-07-10 09:00", tz=TZ).value
        assert f["an_days_chg"][0, 29] == pytest.approx((T[0, 29] - b_down) / 86_400e9)
        assert f["an_days_chg"][0, 30] == pytest.approx(1 / 1440)
        # The consensus at 09:30: A's latest 320 and B's 250 (C priced nothing), median 285.
        assert f["an_target_dist"][0, 0] == pytest.approx(np.log(285 / 300))

    def test_live_refreshes_hourly_and_a_replay_reads_a_later_copy(self, monkeypatch, tmp_path):
        monkeypatch.setattr(F, "HISTORY_DIR", tmp_path)
        F.reset_caches()
        calls = []

        def download(symbol):
            calls.append(symbol)
            return self.RAW

        monkeypatch.setattr(F, "_download_analysts", download)
        close = pd.Timestamp("2026-07-21 16:00", tz=TZ)
        F.analyst_actions("AAPL", close, live=False)
        F.analyst_actions("AAPL", close, live=False)
        F.analyst_actions("AAPL", close, live=True)
        assert calls == ["AAPL"]
        monkeypatch.setattr(F, "ANALYST_REFRESH_SEC", -1.0)  # an hour has passed
        F.analyst_actions("AAPL", close, live=True)
        assert calls == ["AAPL", "AAPL"]
        F.reset_caches()

    def test_a_failed_fetch_keeps_the_cached_copy_and_none_waits(self, monkeypatch, tmp_path):
        monkeypatch.setattr(F, "HISTORY_DIR", tmp_path)
        F.reset_caches()

        def down(symbol):
            raise OSError("Yahoo is down")

        monkeypatch.setattr(F, "_download_analysts", down)
        with pytest.raises(F.InputsPending, match="Yahoo is down"):
            F.analyst_actions("AAPL", pd.Timestamp("2026-07-21 16:00", tz=TZ), live=True)
        self.RAW.to_parquet(tmp_path / "AAPL_analysts.parquet")
        F.reset_caches()
        assert len(F.analyst_actions("AAPL", pd.Timestamp("2026-07-21 16:00", tz=TZ), live=True)) == 4
        F.reset_caches()


# --- against the notebook ----------------------------------------------------------

needs_notebook = pytest.mark.skipif(
    not (NOTEBOOK / "panel_all.parquet").exists() or _recipe("amended") is None
    or _recipe("original") is None,
    reason="the Fast&Furious notebook data or bundles are not on this machine",
)


@lru_cache(maxsize=1)
def _raw() -> "dict[str, pd.DataFrame]":
    out = {}
    for feed, suffix in (("sip", ""), ("iex", "_iex")):
        parts = [pd.read_parquet(NOTEBOOK / f"raw/AAPL_{y}{suffix}.parquet") for y in (2024, 2025, 2026)]
        out[feed] = pd.concat(parts).sort_index()[F.BAR_COLUMNS]
    return out


def _slice(first, before) -> "dict[str, pd.DataFrame]":
    out = {}
    for feed, bars in _raw().items():
        stamp = bars.index.normalize().tz_localize(None)
        out[feed] = bars[(stamp >= pd.Timestamp(first)) & (stamp < pd.Timestamp(before))]
    return out


@pytest.fixture
def notebook_tapes(monkeypatch):
    """The module reading the notebook's raw minute files, option table, gamma
    profile and analyst notes instead of Alpaca and Yahoo, with today after
    the notebook's last session."""
    monkeypatch.setattr(F, "raw_history", lambda symbol, first, before, key=None, secret=None: _slice(first, before))
    positioning = pd.read_parquet(NOTEBOOK / "options/positioning.parquet")
    positioning.index = pd.DatetimeIndex(positioning.index, name="date").as_unit("ns")
    profile = pd.read_parquet(NOTEBOOK / "options/gex_profile.parquet")

    def options(bundle, ctx_day, symbol, key, secret):
        ctx_day.positioning = positioning[positioning.index <= ctx_day.prev_close]
        ctx_day.profile = profile[profile["date"] <= ctx_day.prev_close]

    monkeypatch.setattr(F, "_positioning", options)
    actions = F.clean_analysts(pd.read_parquet(NOTEBOOK / "analysts_raw.parquet"))
    monkeypatch.setattr(F, "analyst_actions", lambda symbol, as_of, live: actions)
    monkeypatch.setattr(F, "_today_et", lambda: date(2026, 10, 13))
    F.reset_caches()
    yield actions
    F.reset_caches()


@lru_cache(maxsize=1)
def _panel() -> pd.DataFrame:
    return pd.read_parquet(NOTEBOOK / "panel_all.parquet")


# A 2025 session, the first sessions after two half days, a 2026 one, the first
# test session and the notebook's three fresh ones.
SESSIONS = ["2025-03-03", "2025-07-07", "2025-12-01", "2026-03-02", "2026-09-08", "2026-10-07",
            "2026-10-09"]


@needs_notebook
class TestAgainstTheNotebook:
    @pytest.mark.parametrize("day", SESSIONS)
    def test_every_feature_is_the_notebooks(self, notebook_tapes, day):
        bundle = _recipe("amended")
        stamp = F._normal_day(day)
        ctx_day = F._session_day(bundle, "AAPL", stamp, None, None)
        today = _slice(stamp, stamp + pd.Timedelta(days=1))
        F._positioning(bundle, ctx_day, "AAPL", None, None)
        gs, gi, ds, ctx = F._panel_inputs(
            ctx_day.history, ctx_day.daily, F.clean(today["sip"]), F.clean(today["iex"]), stamp
        )
        row = slice(len(gs.days) - 1, len(gs.days))
        feats = F.features(F._grid_rows(gs, row), F._grid_rows(gi, row), ds.iloc[row],
                           F._rows(ctx, row), ctx_day.positioning, profile=ctx_day.profile,
                           actions=notebook_tapes)
        mine = F._frame(feats, 0, np.arange(F.LAST_ORIGIN + 1)).dropna(subset=["p", "s30", "d_ma28"])
        ref = _panel()[_panel()["date"] == stamp].set_index("minute", drop=False)
        assert list(mine.index) == list(ref.index)
        for col in F.FEATURE_COLS + sorted(F.ANALYST_COLS):
            np.testing.assert_array_equal(mine[col].to_numpy(float), ref[col].to_numpy(float), err_msg=col)
        np.testing.assert_allclose(mine["s30"], ref["s30"], rtol=1e-14, atol=0)

    @pytest.mark.parametrize("recipe", ["amended", "original"])
    @pytest.mark.parametrize("day", SESSIONS)
    def test_every_forecast_is_the_notebooks(self, notebook_tapes, recipe, day):
        bundle = _recipe(recipe)
        ref = _panel()[_panel()["date"] == F._normal_day(day)].set_index("minute", drop=False)
        forecast = F.session_forecasts(bundle, "AAPL", day)
        expected = F._predict(bundle, ref, True)
        np.testing.assert_array_equal(forecast["y_hat"].to_numpy(), expected["y_hat"].to_numpy())
        np.testing.assert_array_equal(forecast["down_hat"].to_numpy(), expected["down_hat"].to_numpy())

    @pytest.mark.parametrize("recipe", ["amended", "original"])
    def test_a_minute_forecast_live_is_the_replays(self, notebook_tapes, monkeypatch, recipe):
        """The same session forecast whole and forecast at cut-off minutes from
        tapes cut at SIP t-15 and IEX t -- the 15-minute seam included."""
        bundle = _recipe(recipe)
        day = "2026-10-08"
        whole = F.session_forecasts(bundle, "AAPL", day)

        def cut(symbol, d, through, key=None, secret=None, retries=None):
            bars = _slice(d, d + timedelta(days=1))
            minute = lambda b: b.index.hour * 60 + b.index.minute - F.OPEN_MINUTE  # noqa: E731
            return (bars["sip"][minute(bars["sip"]) <= through - F.L],
                    bars["iex"][minute(bars["iex"]) <= through], True)

        monkeypatch.setattr(F, "live_tapes", cut)
        monkeypatch.setattr(F, "_today_et", lambda: date(2026, 10, 8))
        for t in (0, 14, 15, 16, 29, 44, 45, 200, 359):
            live = F.session_forecasts(bundle, "AAPL", day, through=t)
            np.testing.assert_array_equal(
                live[["y_hat", "down_hat", "up_hat"]].to_numpy(),
                whole.loc[whole.index <= t, ["y_hat", "down_hat", "up_hat"]].to_numpy(),
            )

    def test_the_amended_threshold_is_the_notebooks_rule(self, notebook_tapes):
        """`evaluate_amended.py`'s threshold on 9 Oct 2026: the q-quantile of
        |y_hat| over the 20 sessions before it (11 Sep - 8 Oct, all in the test
        window, where the notebook read the saved model's own forecasts),
        here with each session's own option rows and analyst notes."""
        bundle = _recipe("amended")
        days = np.sort(_panel()["date"].unique())
        i = list(days).index(np.datetime64("2026-10-09"))
        rows = _panel()[_panel()["date"].isin(days[i - 20:i])]
        reference = np.abs(bundle["model"].momentum.predict(rows))
        for q in (0.9, 0.95):
            theta, sessions = F.threshold(bundle, "AAPL", "2026-10-09", q)
            assert sessions == 20
            assert theta == np.quantile(reference, q)
        assert F.threshold(bundle, "AAPL", "2026-10-09", 0.9)[0] == pytest.approx(0.034119, abs=5e-7)
        reading = F.forecast_at(bundle, "AAPL", "2026-10-09", 359, 0.9)
        assert reading["y_hat"] == pytest.approx(-0.045383, abs=5e-7)
        assert reading["signal"] == -1 and reading["range_problem"] == ""

    def test_the_threshold_and_a_reading_are_notebook_6s(self, notebook_tapes):
        """Notebook 06 section 6.8, the original recipe: 9 Oct 2026 at 15:30,
        -0.0148 bps a minute, |y_hat| 0.0099 against a threshold of 0.0425."""
        bundle = _recipe("original")
        theta, sessions = F.threshold(bundle, "AAPL", "2026-10-09", 0.95)
        assert sessions == 20
        assert theta == pytest.approx(0.0425, abs=5e-5)
        reading = F.forecast_at(bundle, "AAPL", "2026-10-09", 359, 0.95)
        assert reading["mom_bps"] == pytest.approx(-0.0148, abs=5e-5)
        assert abs(reading["y_hat"]) == pytest.approx(0.0099, abs=5e-5)
        assert reading["signal"] == 0
        assert reading["range_problem"] == ""
