"""The HighLow2 mirror: the history cache, this morning's inputs, and the bundle.

The tests that need neither the saved model nor the notebook pin the seams --
what the app has to get right to hand the model correct inputs: which days the
cache fetches and when, a split being noticed, the 9:35 line every pre-market
and opening input is cut at, a short history being refused.

The ones that need both pin the mirror itself: the notebook's own raw minute
files (SIP and IEX, regular and extended hours), summarised by the live path's
code, must reproduce what the notebook's `highlow2` package predicts.
"""

from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agent_stonks import apple_models

H = pytest.importorskip("agent_stonks.highlow2_model")

NOTEBOOK = Path("/Users/aleksandra/Documents/playground/Code/FinNotebooks/HighLow2_5m")

# What the notebook's own `highlow2` package predicts from its own panel
# (`pipeline.build`, then `models.load_bundle(...).predict_prices`) with the
# AAPL bundle saved 2026-10-04: (pred_high, pred_low, adr14_usd). The test
# window (8-18 Sep 2026) and the traded week (21-25 Sep), plus shock days the
# model was fitted without (2024-11-01, 2025-03-03, 2025-07-07) and the
# sessions after a half day (2025-07-07, 2025-12-01, 2025-12-26), whose
# after-hours and pre-market rows straddle a session SIP dropped.
AAPL_FORECASTS = {
    "2024-11-01": (224.42990920563565, 220.12577468283314, 3.415635714285713),
    "2025-03-03": (245.58525246343277, 241.03141745076212, 5.208178571428576),
    "2025-07-07": (216.97999957320107, 212.2083624829336, 3.788707142857141),
    "2025-12-01": (279.6486259598362, 275.5029910283409, 5.790285714285716),
    "2025-12-26": (276.20039604303486, 273.1638371323617, 4.327450000000007),
    "2026-03-02": (265.1439923112559, 259.69249734502415, 6.819571428571438),
    "2026-09-08": (320.7851716481907, 314.2385080741127, 7.313878571428567),
    "2026-09-09": (317.78416739452535, 313.28665356314286, 7.317449999999996),
    "2026-09-10": (323.4289014960368, 315.74399304217303, 7.286742857142855),
    "2026-09-11": (336.49074514444203, 326.57, 7.329599999999999),
    "2026-09-14": (335.7043698751707, 329.6215447982723, 7.6546),
    "2026-09-15": (332.86472018163533, 327.5092396165973, 7.709600000000003),
    "2026-09-16": (336.70953519290657, 331.39792641662274, 7.570314285714285),
    "2026-09-17": (335.39, 330.13341035789784, 7.438178571428572),
    "2026-09-18": (339.35768525820515, 333.89373783501696, 7.592235714285716),
    "2026-09-21": (337.203944225059, 331.9863296812984, 7.523692857142862),
    "2026-09-22": (346.1086343675032, 339.3082680504676, 7.3919071428571455),
    "2026-09-23": (342.9608444473288, 337.62954551126455, 6.9647642857142875),
    "2026-09-24": (338.8920928271278, 334.1129883422683, 7.066907142857145),
    "2026-09-25": (337.29399002103725, 332.5079589178143, 6.917621428571432),
}


# --- synthetic sessions -------------------------------------------------------

TZ = "America/New_York"


def _clock(day: date, hhmm: str) -> pd.Timestamp:
    return pd.Timestamp(f"{day} {hhmm}").tz_localize(TZ)


def _bars(times: pd.DatetimeIndex, base: float, rng, scale: float = 1.0) -> "tuple[list[dict], float]":
    closes = base * np.exp(np.cumsum(rng.normal(0, 0.0008, len(times))))
    opens = np.r_[base, closes[:-1]]
    highs = np.maximum(opens, closes) * 1.0004
    lows = np.minimum(opens, closes) * 0.9996
    volumes = 1000 + rng.integers(0, 500, len(times))
    stamps = times.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    out = [
        {"t": t, "o": o * scale, "h": h * scale, "l": lo * scale, "c": c * scale, "v": int(v)}
        for t, o, h, lo, c, v in zip(stamps, opens, highs, lows, closes, volumes)
    ]
    return out, float(closes[-1])


# A sparse pre-market (04:00-09:29, every 7 minutes), every regular minute, and
# a sparse after-hours, as offsets in minutes from midnight.
_PRE = [4 * 60 + 7 * k for k in range(48)]
_RTH = [9 * 60 + 30 + i for i in range(390)]
_POST = [16 * 60 + 10 * k for k in range(24)]
# IEX: the regular session and the pre-market bars from 09:20 on.
_IEX = [i for i, m in enumerate(_PRE + _RTH + _POST) if 9 * 60 + 20 <= m < 16 * 60]


def _day(day: date, base: float, rng, scale: float = 1.0) -> "tuple[list[dict], list[dict], float]":
    """One weekday in Alpaca's bar-dict shape: SIP every regular minute plus a
    sparse pre-market and after-hours; IEX the regular session and SIP's
    pre-market bars from 09:20, at a fraction of the volume."""
    midnight = pd.Timestamp(day).tz_localize(TZ)
    times = midnight + pd.to_timedelta(_PRE + _RTH + _POST, unit="min")
    sip, base = _bars(pd.DatetimeIndex(times), base, rng, scale)
    iex = [{**sip[i], "v": sip[i]["v"] // 20} for i in _IEX]
    return sip, iex, base


@lru_cache(maxsize=None)
def _tape(first: date, last: date, scale: float = 1.0) -> "dict[date, tuple[list, list]]":
    """Deterministic (SIP, IEX) bars for every weekday in [first, last].
    Shared between tests: treat it as read-only."""
    rng = np.random.default_rng(5)
    out, day, base = {}, first, 100.0
    while day <= last:
        if day.weekday() < 5:
            sip, iex, base = _day(day, base, rng, scale)
            out[day] = (sip, iex)
        day += timedelta(days=1)
    return out


def _frames(tape, days) -> "tuple[pd.DataFrame, pd.DataFrame]":
    sip = [b for d in days for b in tape[d][0]]
    iex = [b for d in days for b in tape[d][1]]
    return H.bars_frame(sip), H.bars_frame(iex)


class FakeAlpaca:
    """Stands in for `_fetch_stretch` and `fetch_morning`, counting what is asked."""

    def __init__(self, tape):
        self.tape = tape
        self.stretches: "list[tuple[date, date]]" = []
        self.mornings: "list[date]" = []

    def stretch(self, symbol, first, before, key, secret, min_bars=H.MIN_BARS_PER_SESSION):
        self.stretches.append((first, before))
        sip, iex = _frames(self.tape, [d for d in self.tape if first <= d < before])
        return H.stretch_from(sip, iex, min_bars)

    def morning(self, symbol, session_date, n_minutes=H.OPENING_MINUTES, key=None, secret=None):
        day = pd.Timestamp(session_date).date()
        self.mornings.append(day)
        sip, iex = _frames(self.tape, [day])
        return H.morning_from_bars(sip, iex, day, n_minutes)


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(H, "HISTORY_DIR", tmp_path / "highlow2")
    return tmp_path / "highlow2"


@pytest.fixture
def alpaca(cache_dir, monkeypatch):
    fake = FakeAlpaca(_tape(date(2025, 12, 1), date(2026, 9, 30)))
    monkeypatch.setattr(H, "_fetch_stretch", fake.stretch)
    monkeypatch.setattr(H, "fetch_morning", fake.morning)
    monkeypatch.setattr(H, "_forecast_cache", {})
    return fake


class Constant:
    """A shipped candidate that answers one excursion pair whatever it is shown,
    and remembers the rows it was shown."""

    feature_cols = H.FEATURE_COLS

    def __init__(self):
        self.rows = []

    def predict(self, X):
        self.rows.append(X)
        return np.tile([1.0, 0.5], (len(X), 1))


def _bundle(path="synthetic"):
    candidate = Constant()
    model = H.HighLowModel({"c": candidate}, {"c": 1.0}, list(H.FEATURE_COLS))
    return {"kind": "highlow2", "model": model, "opening_minutes": 5, "path": path}, candidate


# --- the registry ---------------------------------------------------------------

class TestRegistry:
    def test_highlow2_is_offered_where_a_bundle_was_saved(self):
        assert apple_models.HIGHLOW2_KEY in apple_models.keys_for("AAPL")
        assert apple_models.HIGHLOW2_KEY not in apple_models.keys_for("INTC")

    def test_it_drives_the_day_range_rules(self):
        assert apple_models.strategy(apple_models.HIGHLOW2_KEY) == apple_models.STRATEGY_DAYRANGE


# --- the pre-market ---------------------------------------------------------------

class TestPremarket:
    def _pieces(self, tape, days):
        sip, iex = _frames(tape, days)
        return H.part_summaries(H.extended_bars(sip, "pre"), H.extended_bars(iex, "pre"),
                                H.extended_bars(sip, "post"))

    def test_summarising_in_pieces_matches_summarising_at_once(self):
        """What lets the cache keep per-date pieces and join them later."""
        tape = _tape(date(2026, 9, 7), date(2026, 9, 18))
        days = sorted(tape)
        whole = self._pieces(tape, days)
        halves = [self._pieces(tape, days[:5]), self._pieces(tape, days[5:])]
        for part in H.PARTS:
            pd.testing.assert_frame_equal(
                whole[part], pd.concat([h[part] for h in halves]), check_dtype=False,
            )

    def test_only_what_is_public_at_935_is_read(self):
        """SIP to its 09:19 bar, IEX's pre-market from 09:20 to 09:29."""
        tape = _tape(date(2026, 9, 14), date(2026, 9, 14))
        pieces = self._pieces(tape, sorted(tape))
        sip, iex = _frames(tape, sorted(tape))
        pre = sip.between_time("04:00", "09:19")
        assert pieces["pm"]["pm_high"].iloc[0] == pre["high"].max()
        assert pieces["pm"]["pm_bars"].iloc[0] == len(pre)
        tail = iex.between_time("09:20", "09:29")
        assert pieces["iexpm"]["iexpm_bars"].iloc[0] == len(tail)
        assert pieces["iexpm"]["iexpm_last"].iloc[0] == tail["close"].iloc[-1]

    def test_an_evening_belongs_to_the_next_date_with_a_premarket(self):
        """Friday's after-hours is Monday's `ah_*`, not Saturday's."""
        tape = _tape(date(2026, 9, 10), date(2026, 9, 14))
        pieces = self._pieces(tape, sorted(tape))
        summary = H.session_summary_from(pieces["pm"], pieces["pml"], pieces["iexpm"], pieces["ah"])
        friday = pieces["ah"].loc[pd.Timestamp("2026-09-11")]
        assert summary.loc[pd.Timestamp("2026-09-14"), "ah_high"] == friday["ah_high"]
        assert pd.isna(summary.loc[pd.Timestamp("2026-09-10"), "ah_high"])  # no evening before it here


# --- this morning ------------------------------------------------------------------

class TestMorning:
    DAY = date(2026, 9, 14)

    def test_bars_past_the_935_line_cannot_move_the_morning(self):
        tape = _tape(self.DAY, self.DAY)
        sip, iex = _frames(tape, [self.DAY])
        cut = H.morning_from_bars(sip[sip.index < _clock(self.DAY, "09:20")],
                                  iex[iex.index < _clock(self.DAY, "09:35")], self.DAY)
        whole = H.morning_from_bars(sip, iex, self.DAY)
        pd.testing.assert_frame_equal(cut.opening, whole.opening)
        for part in H.MORNING_PARTS:
            pd.testing.assert_frame_equal(cut.parts[part], whole.parts[part])

    def test_the_opening_is_iexs(self):
        tape = _tape(self.DAY, self.DAY)
        sip, iex = _frames(tape, [self.DAY])
        iex = iex.copy()
        first = _clock(self.DAY, "09:30")
        iex.loc[first, "open"] = iex.loc[first, "high"]  # IEX's open only, still a sane bar
        assert iex.loc[first, "open"] != sip.loc[first, "open"]
        morning = H.morning_from_bars(sip, iex, self.DAY)
        assert morning.opening["open5"].iloc[0] == iex.loc[first, "open"]

    def test_a_short_iex_window_is_kept_and_says_so(self):
        """IEX misses quiet minutes; the notebook keeps such a window."""
        tape = _tape(self.DAY, self.DAY)
        sip, iex = _frames(tape, [self.DAY])
        iex = iex.drop([_clock(self.DAY, "09:31"), _clock(self.DAY, "09:33")])
        morning = H.morning_from_bars(sip, iex, self.DAY)
        assert morning.opening["or_bars_frac"].iloc[0] == pytest.approx(0.6)

    def test_an_empty_iex_window_is_refused(self):
        tape = _tape(self.DAY, self.DAY)
        sip, iex = _frames(tape, [self.DAY])
        with pytest.raises(ValueError, match="IEX printed nothing"):
            H.morning_from_bars(sip, iex[iex.index < _clock(self.DAY, "09:30")], self.DAY)


# --- the history cache -----------------------------------------------------------

class TestHistoryCache:
    def test_a_cold_cache_fetches_the_whole_window_once(self, alpaca):
        day = date(2026, 9, 14)
        history = H.history_inputs("AAPL", day, "k", "s")
        assert alpaca.stretches == [(day - timedelta(days=H.HISTORY_CALENDAR_DAYS), day)]
        assert history.sessions.index.max() < pd.Timestamp(day)
        assert len(history.sessions) >= H.MIN_PRIOR_SESSIONS
        for part in H.PARTS:
            assert len(history.parts[part]) and history.parts[part].index.max() < pd.Timestamp(day)

        H.history_inputs("AAPL", day, "k", "s")
        assert len(alpaca.stretches) == 1  # served from disk

    def test_the_next_morning_fetches_only_the_new_session(self, alpaca):
        H.history_inputs("AAPL", date(2026, 9, 14), "k", "s")
        history = H.history_inputs("AAPL", date(2026, 9, 15), "k", "s")
        # From the newest cached session (Friday the 11th, the seam), whole, up
        # to the session being forecast -- so Monday's pre-market and Friday's
        # evening are both in.
        assert alpaca.stretches[-1] == (date(2026, 9, 11), date(2026, 9, 15))
        assert history.sessions.index[-1] == pd.Timestamp("2026-09-14")
        assert history.parts["ah"].index[-1] == pd.Timestamp("2026-09-14")

    def test_a_split_rebuilds_the_cache_rather_than_mixing_scales(self, alpaca, monkeypatch):
        H.history_inputs("AAPL", date(2026, 9, 14), "k", "s")
        after_split = FakeAlpaca(_tape(date(2025, 12, 1), date(2026, 9, 30), scale=0.25))
        monkeypatch.setattr(H, "_fetch_stretch", after_split.stretch)
        history = H.history_inputs("AAPL", date(2026, 9, 15), "k", "s")
        assert after_split.stretches[-1] == (
            date(2026, 9, 15) - timedelta(days=H.HISTORY_CALENDAR_DAYS), date(2026, 9, 15)
        )
        assert history.sessions["close"].max() < 50  # every row on the new scale
        assert history.parts["pm"]["pm_high"].max() < 50

    def test_a_cache_in_another_layout_is_rebuilt(self, alpaca, cache_dir):
        H.history_inputs("AAPL", date(2026, 9, 14), "k", "s")
        path = H.history_path("AAPL")
        path.write_text(path.read_text().replace(f'"layout": {H.CACHE_LAYOUT}', '"layout": 0'))
        H.history_inputs("AAPL", date(2026, 9, 14), "k", "s")
        assert len(alpaca.stretches) == 2

    def test_highlows_cache_is_not_this_ones(self):
        """HighLow's files hold SIP openings and no extended hours."""
        from agent_stonks import model_catalogue

        assert H.HISTORY_DIR != model_catalogue.HIGHLOW_HISTORY_DIR

    def test_no_credentials_is_a_refusal_not_a_guess(self, cache_dir, monkeypatch):
        monkeypatch.delenv("ALPACA_API_KEY", raising=False)
        monkeypatch.delenv("ALPACA_SECRET", raising=False)
        with pytest.raises(ValueError, match="credentials"):
            H.history_inputs("AAPL", date(2026, 9, 14))


# --- one session's forecast ---------------------------------------------------------

class TestForecast:
    DAY = date(2026, 9, 15)

    def test_live_the_morning_is_fetched_and_the_answer_memoised(self, alpaca):
        bundle, candidate = _bundle()
        opening = pd.DataFrame({"open": [1.0] * 5})
        first = H.forecast_session(bundle, "AAPL", opening, self.DAY, "k", "s")
        assert alpaca.mornings == [self.DAY]
        assert H.forecast_session(bundle, "AAPL", opening, self.DAY, "k", "s") == first
        assert len(candidate.rows) == 1
        # (up, down) = (1.0, 0.5) ADRs from the IEX close5, clipped to its range.
        row = candidate.rows[0]
        adr, close5 = float(row["adr14"].iloc[0]), float(row["close5"].iloc[0])
        assert first["pred_high"] == pytest.approx(max(close5 * np.exp(adr), row["high5"].iloc[0]))
        assert first["pred_low"] == pytest.approx(min(close5 * np.exp(-0.5 * adr), row["low5"].iloc[0]))

    def test_a_replay_of_a_day_the_cache_covers_reads_its_morning_from_there(self, alpaca):
        """The cached pieces are cut at the same 9:35 line by the same code, so
        the forecast is the same and nothing is fetched for the morning."""
        bundle, _ = _bundle()
        opening = pd.DataFrame({"open": [1.0] * 5})
        fetched = H.forecast_session(bundle, "AAPL", opening, self.DAY, "k", "s")
        H.warm_history(bundle, "AAPL", self.DAY + timedelta(days=7), "k", "s")
        H._forecast_cache.clear()
        cached = H.forecast_session(bundle, "AAPL", opening, self.DAY, "k", "s")
        assert alpaca.mornings == [self.DAY]  # only the first, live-like ask
        assert cached == pytest.approx(fetched)

    def test_an_unsettled_live_window_is_not_memoised(self, alpaca, monkeypatch):
        bundle, candidate = _bundle()
        opening = pd.DataFrame({"open": [1.0] * 5})

        def unsettled(*a, **k):
            morning = alpaca.morning(*a, **k)
            morning.settled = False
            return morning

        monkeypatch.setattr(H, "fetch_morning", unsettled)
        H.forecast_session(bundle, "AAPL", opening, self.DAY, "k", "s")
        H.forecast_session(bundle, "AAPL", opening, self.DAY, "k", "s")
        assert len(candidate.rows) == 2

    def test_a_morning_with_no_premarket_still_forecasts(self, alpaca):
        """The pre-market columns may be missing (`NAN_OK`); LightGBM was fitted so."""
        bundle, candidate = _bundle()
        sip, iex = _frames(alpaca.tape, [self.DAY])
        morning = H.morning_from_bars(sip.iloc[:0], iex[iex.index >= _clock(self.DAY, "09:30")], self.DAY)
        history = H.history_inputs("AAPL", self.DAY, "k", "s")
        H.forecast_from(bundle, history, morning, self.DAY)
        row = candidate.rows[-1]
        assert pd.isna(row["pm_range_adr"].iloc[0]) and pd.isna(row["open_vs_pre_adr"].iloc[0])

    def test_a_short_history_is_refused(self, alpaca):
        bundle, _ = _bundle()
        history = H.history_inputs("AAPL", self.DAY, "k", "s")
        short = H.History(history.sessions.iloc[-60:], history.parts)
        with pytest.raises(ValueError, match="sessions of history"):
            H.forecast_from(bundle, short, alpaca.morning("AAPL", self.DAY), self.DAY)

    def test_too_few_opening_bars_is_refused(self, alpaca):
        bundle, _ = _bundle()
        with pytest.raises(ValueError, match="first 5 minutes"):
            H.forecast_session(bundle, "AAPL", pd.DataFrame({"open": [1.0] * 3}), self.DAY)


# --- the bundle -------------------------------------------------------------------

def _installed_bundle():
    bundle = H.load_bundle("AAPL")
    if bundle is None:
        pytest.skip("the HighLow2 AAPL bundle is not installed")
    return bundle


class TestBundle:
    def test_the_shipped_blend_is_three_lightgbm_seeds(self):
        bundle = _installed_bundle()
        assert bundle["kind"] == "highlow2"
        assert bundle["ticker"] == "AAPL"
        assert bundle["daily_models"] == ["lgbm_11", "lgbm_23", "lgbm_7"]
        assert bundle["opening_feed"] == "iex" and bundle["target"] == "close5/adr14"
        assert set(bundle["model"].feature_cols) == set(H.FEATURE_COLS)

    def _resaved(self, tmp_path, **changes):
        import joblib

        H._register_unpickle_alias()
        src = H.model_path("AAPL")
        if not src.exists():
            pytest.skip("the HighLow2 AAPL bundle is not installed")
        blob = {**joblib.load(src), **changes}
        target = tmp_path / src.name
        joblib.dump(blob, target)
        (tmp_path / src.with_suffix(".json").name).write_bytes(src.with_suffix(".json").read_bytes())
        return blob, target

    def test_a_weighted_network_is_refused(self, tmp_path):
        """A net ships as a `.pt` beside the joblib; this module mirrors none."""
        import joblib

        blob, target = self._resaved(tmp_path)
        joblib.dump({**blob, "weights": {**blob["weights"], "nbeats": 0.5}}, target)
        assert H._build_bundle(target) is None
        joblib.dump({**blob, "weights": {**blob["weights"], "nbeats": 0.0}}, target)
        assert H._build_bundle(target) is not None  # at weight 0 it is never asked

    def test_a_candidate_reading_an_unmirrored_group_is_refused(self, tmp_path):
        blob, target = self._resaved(tmp_path)
        import joblib

        joblib.dump({**blob, "feature_cols": blob["feature_cols"] + ["vix"]}, target)
        assert H._build_bundle(target) is None

    def test_the_option_scaled_target_is_refused(self, tmp_path):
        _, target = self._resaved(tmp_path, target={"anchor": "close5", "scale": "geo"})
        assert H._build_bundle(target) is None


# --- the mirror, against the notebook ------------------------------------------------

_NOTEBOOK_INPUTS: dict = {}


def notebook_inputs():
    """The bundle, and the notebook's *raw* minute files -- SIP and IEX, the
    regular session and both extended windows -- summarised by the live path's
    code, so the session cleaning, the half days and the 9:35 cuts are all part
    of what is checked."""
    if _NOTEBOOK_INPUTS:
        return _NOTEBOOK_INPUTS["inputs"]
    folder = NOTEBOOK / "data" / "AAPL" / "raw"
    if not sorted(folder.glob("AAPL_20??_pre_sip.parquet")):
        pytest.skip("the HighLow2_5m AAPL notebook data is not on this machine")
    bundle = _installed_bundle()

    def tape(*patterns):
        raw = pd.concat(pd.read_parquet(f) for p in patterns for f in sorted(folder.glob(p)))
        return raw.sort_index()[H.OHLCV]

    sip = tape("AAPL_20??.parquet", "AAPL_20??_pre_sip.parquet", "AAPL_20??_post_sip.parquet")
    iex = tape("AAPL_20??_iex.parquet", "AAPL_20??_pre_iex.parquet", "AAPL_20??_post_iex.parquet")
    _NOTEBOOK_INPUTS["inputs"] = (bundle, sip, iex, H.stretch_from(sip, iex))
    return _NOTEBOOK_INPUTS["inputs"]


def _one_day(frame, stamp):
    return frame[frame.index.normalize().tz_localize(None) == stamp]


class TestAgainstTheNotebook:
    """The mirror contract: the notebook's tapes, summarised by the live path's
    code, reproduce the notebook package's forecasts. LightGBM alone and the
    same library version, so to the last digit rather than within float noise."""

    @pytest.mark.parametrize("day", sorted(AAPL_FORECASTS))
    def test_reproduces_the_notebook_forecast(self, day):
        bundle, sip, iex, history = notebook_inputs()
        stamp = pd.Timestamp(day)
        morning = H.morning_from_bars(_one_day(sip, stamp), _one_day(iex, stamp), stamp)
        out = H.forecast_from(bundle, history, morning, stamp)
        high, low, adr = AAPL_FORECASTS[day]
        assert out["pred_high"] == pytest.approx(high, abs=1e-8)
        assert out["pred_low"] == pytest.approx(low, abs=1e-8)
        assert out["adr14_abs"] == pytest.approx(adr, abs=1e-9)

    def test_the_cached_morning_is_the_fetched_one(self, cache_dir):
        """`_cached_morning` against `morning_from_bars` on real tape."""
        _, sip, iex, history = notebook_inputs()
        cache = H._empty_cache()
        H._merge(cache, history, None)
        cache["from"], cache["through"] = "2024-01-02", "2026-09-25"
        H._write_cache("AAPL", cache)
        for day in ("2026-09-10", "2026-09-21"):  # with and without IEX's 09:20-09:29
            stamp = pd.Timestamp(day)
            cached = H._cached_morning("AAPL", stamp.date())
            fetched = H.morning_from_bars(_one_day(sip, stamp), _one_day(iex, stamp), stamp)
            pd.testing.assert_frame_equal(
                cached.opening, fetched.opening, check_dtype=False, check_index_type=False,
            )
            for part in H.MORNING_PARTS:
                pd.testing.assert_frame_equal(
                    cached.parts[part], fetched.parts[part], check_dtype=False,
                    check_index_type=False,
                )
