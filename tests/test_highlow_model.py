"""The HighLow mirror: the SIP history cache, the forecast's seams, and the bundle.

The tests that need neither the saved model nor the notebook pin the seams --
what the app has to get right to hand the model correct inputs: which sessions
the history cache fetches and when, a split being noticed, a short history
being refused.

The one that needs both pins the mirror itself: the notebook's own minute tape,
rolled up the way the live path rolls up SIP bars, must reproduce what the
notebook's `highlow` package predicts for the held-out week.
"""

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agent_stonks import apple_models

H = pytest.importorskip("agent_stonks.highlow_model")

NOTEBOOK = Path("/Users/aleksandra/Documents/playground/Code/FinNotebooks/HighLow_5m")

# What the notebook's own `highlow` package predicts from its own panel for the
# held-out week of 31 Aug 2026 plus one later day and one test-window day
# (`highlow.models.load_bundle(...).predict_prices`), per ticker's bundle:
# AAPL saved 2026-09-15, INTC 2026-09-21. (pred_high, pred_low, adr14_usd)
AAPL_FORECASTS = {
    "2026-03-02": (265.63175058123767, 259.76960582407247, 6.819571428571438),
    "2026-08-31": (321.235, 315.78224457142767, 5.9200214285714265),
    "2026-09-01": (317.8488540510188, 313.09447158886843, 6.009664285714284),
    "2026-09-02": (328.7640088736563, 323.0533824041481, 6.543949999999995),
    "2026-09-03": (328.4214399925502, 323.1039373069979, 6.609664285714282),
    "2026-09-04": (329.76991231125703, 324.7885355428754, 6.860378571428567),
    "2026-09-11": (336.3560774590717, 326.3, 7.329599999999999),
}
INTC_FORECASTS = {
    "2026-03-02": (45.88399158706168, 43.88328505541922, 2.331371428571429),
    "2026-08-31": (92.52259005688143, 88.91520912956068, 4.119050000000001),
    "2026-09-01": (88.20672619422156, 85.05890732577451, 4.1104785714285725),
    "2026-09-02": (90.92706444298031, 87.5004722200887, 4.157135714285715),
    "2026-09-03": (90.08890242862724, 87.19492111916887, 3.8492785714285733),
    "2026-09-04": (95.81487950584402, 91.81971111056774, 3.8367785714285736),
    "2026-09-11": (104.88463667029384, 101.07, 3.8160571428571433),
}
NOTEBOOK_FORECASTS = {"AAPL": AAPL_FORECASTS, "INTC": INTC_FORECASTS}
# The candidates each bundle ships with non-zero weight.
SHIPPED = {"AAPL": ["lgbm", "nbeats"], "INTC": ["nhits"]}


# --- synthetic sessions -------------------------------------------------------

def _session_bars(day: date, base: float, rng) -> "list[dict]":
    """A full 390-bar regular session in Alpaca's bar-dict shape."""
    start = pd.Timestamp(day).tz_localize("America/New_York") + pd.Timedelta(hours=9, minutes=30)
    closes = base * np.exp(np.cumsum(rng.normal(0, 0.0008, 390)))
    bars = []
    prev = base
    for i, close in enumerate(closes):
        hi, lo = max(prev, close) * 1.0004, min(prev, close) * 0.9996
        bars.append({
            "t": (start + pd.Timedelta(minutes=i)).tz_convert("UTC").isoformat(),
            "o": prev, "h": hi, "l": lo, "c": close, "v": 1000 + int(rng.integers(0, 500)),
        })
        prev = close
    return bars


def _tape(first: date, last: date, scale: float = 1.0) -> "dict[date, list[dict]]":
    """Deterministic sessions for every weekday in [first, last]."""
    rng = np.random.default_rng(3)
    out, day, base = {}, first, 100.0
    while day <= last:
        if day.weekday() < 5:
            bars = _session_bars(day, base, rng)
            base = bars[-1]["c"]
            out[day] = [
                {**b, "o": b["o"] * scale, "h": b["h"] * scale, "l": b["l"] * scale, "c": b["c"] * scale}
                for b in bars
            ]
        day += timedelta(days=1)
    return out


class FakeSip:
    """Stands in for `highlow_model._fetch_rollups`, counting what is asked."""

    def __init__(self, tape):
        self.tape = tape
        self.calls: "list[tuple[date, date]]" = []

    def __call__(self, symbol, first, before, key, secret):
        self.calls.append((first, before))
        bars = [b for d, day_bars in self.tape.items() if first <= d < before for b in day_bars]
        return H.session_rollups(H.minute_frame_from_bars(bars))


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(H, "HISTORY_DIR", tmp_path / "highlow")
    return tmp_path / "highlow"


class TestRegistry:
    def test_highlow_is_offered_where_a_bundle_was_saved(self):
        for symbol in ("AAPL", "INTC"):
            assert apple_models.HIGHLOW_KEY in apple_models.keys_for(symbol)
        assert apple_models.HIGHLOW_KEY not in apple_models.keys_for("GOOGL")

    def test_it_drives_the_day_range_rules(self):
        assert apple_models.strategy(apple_models.HIGHLOW_KEY) == apple_models.STRATEGY_DAYRANGE


class TestRollups:
    def test_a_half_day_is_dropped_like_the_notebook_drops_it(self):
        tape = _tape(date(2026, 3, 2), date(2026, 3, 3))
        half = [b for b in tape[date(2026, 3, 3)]][:210]  # 13:00 close
        rollups, dropped = H.session_rollups(
            H.minute_frame_from_bars(tape[date(2026, 3, 2)] + half)
        )
        assert list(rollups.index) == [pd.Timestamp("2026-03-02")]
        assert [pd.Timestamp(d) for d in dropped] == [pd.Timestamp("2026-03-03")]

    def test_rolling_up_in_chunks_matches_rolling_up_at_once(self):
        tape = _tape(date(2026, 3, 2), date(2026, 3, 13))
        bars = [b for day in tape.values() for b in day]
        whole, _ = H.session_rollups(H.minute_frame_from_bars(bars))
        days = sorted(tape)
        halves = [
            H.session_rollups(H.minute_frame_from_bars([b for d in part for b in tape[d]]))[0]
            for part in (days[:4], days[4:])
        ]
        pd.testing.assert_frame_equal(pd.concat(halves), whole)


class TestHalfDays:
    def test_the_rule_reproduces_the_notebooks_list(self):
        """`config.EARLY_CLOSE_DATES` in HighLow_5m, which ends in 2025."""
        listed = {
            "2023-07-03", "2023-11-24", "2024-07-03", "2024-11-29", "2024-12-24",
            "2025-07-03", "2025-11-28", "2025-12-24",
        }
        days = pd.bdate_range("2023-01-01", "2025-12-31")
        assert set(days[H.early_close(days)].strftime("%Y-%m-%d")) == listed

    def test_a_half_day_with_a_full_bar_count_is_still_dropped(self):
        """INTC's 28 Nov 2025: SIP printed past the 13:00 close, so the session
        passed the bar count. The date is what drops it."""
        tape = _tape(date(2025, 11, 27), date(2025, 11, 28))
        rollups, dropped = H.session_rollups(H.minute_frame_from_bars(tape[date(2025, 11, 28)]))
        assert rollups.empty
        assert [pd.Timestamp(d) for d in dropped] == [pd.Timestamp("2025-11-28")]


class TestHistoryCache:
    def test_a_cold_cache_fetches_the_whole_window_once(self, cache_dir, monkeypatch):
        sip = FakeSip(_tape(date(2025, 12, 1), date(2026, 9, 30)))
        monkeypatch.setattr(H, "_fetch_rollups", sip)
        day = date(2026, 9, 14)

        frame = H.history_frame("AAPL", day, "k", "s")
        assert sip.calls == [(day - timedelta(days=H.HISTORY_CALENDAR_DAYS), day)]
        assert frame.index.max() < pd.Timestamp(day)
        assert len(frame) >= H.MIN_PRIOR_SESSIONS

        H.history_frame("AAPL", day, "k", "s")
        assert len(sip.calls) == 1  # served from disk

    def test_the_next_morning_fetches_only_the_new_session(self, cache_dir, monkeypatch):
        sip = FakeSip(_tape(date(2025, 12, 1), date(2026, 9, 30)))
        monkeypatch.setattr(H, "_fetch_rollups", sip)
        H.history_frame("AAPL", date(2026, 9, 14), "k", "s")

        frame = H.history_frame("AAPL", date(2026, 9, 15), "k", "s")
        # From the newest cached session (Friday the 11th, the seam) up to the
        # session being forecast.
        assert sip.calls[-1] == (date(2026, 9, 11), date(2026, 9, 15))
        assert frame.index[-1] == pd.Timestamp("2026-09-14")

    def test_a_replay_of_an_earlier_day_fetches_only_the_older_edge(
        self, cache_dir, monkeypatch
    ):
        sip = FakeSip(_tape(date(2025, 12, 1), date(2026, 9, 30)))
        monkeypatch.setattr(H, "_fetch_rollups", sip)
        H.history_frame("AAPL", date(2026, 9, 14), "k", "s")
        oldest = min(H._read_cache("AAPL")["sessions"])

        frame = H.history_frame("AAPL", date(2026, 9, 1), "k", "s")
        # Only the thirteen days its window reaches further back were missing,
        # read up to and including the oldest cached session (the seam).
        assert sip.calls[1:] == [(
            date(2026, 9, 1) - timedelta(days=H.HISTORY_CALENDAR_DAYS),
            date.fromisoformat(oldest) + timedelta(days=1),
        )]
        assert frame.index.max() < pd.Timestamp("2026-09-01")

        H.history_frame("AAPL", date(2026, 8, 20), "k", "s")
        assert len(sip.calls) == 3  # a little further back again, nothing forward

    def test_a_split_rebuilds_the_cache_rather_than_mixing_scales(self, cache_dir, monkeypatch):
        before_split = FakeSip(_tape(date(2025, 12, 1), date(2026, 9, 30)))
        monkeypatch.setattr(H, "_fetch_rollups", before_split)
        H.history_frame("AAPL", date(2026, 9, 14), "k", "s")

        after_split = FakeSip(_tape(date(2025, 12, 1), date(2026, 9, 30), scale=0.25))
        monkeypatch.setattr(H, "_fetch_rollups", after_split)
        frame = H.history_frame("AAPL", date(2026, 9, 15), "k", "s")

        # The seam session came back at a quarter of its cached close: refetched whole.
        assert after_split.calls[-1] == (date(2026, 9, 15) - timedelta(days=H.HISTORY_CALENDAR_DAYS),
                                         date(2026, 9, 15))
        assert frame["close"].max() < 50  # every row on the new scale

    def test_no_credentials_is_a_refusal_not_a_guess(self, cache_dir, monkeypatch):
        monkeypatch.delenv("ALPACA_API_KEY", raising=False)
        monkeypatch.delenv("ALPACA_SECRET", raising=False)
        with pytest.raises(ValueError, match="credentials"):
            H.history_frame("AAPL", date(2026, 9, 14))


class TestForecastSeams:
    def test_a_short_history_is_refused(self):
        tape = _tape(date(2026, 6, 1), date(2026, 7, 31))
        history, _ = H.session_rollups(
            H.minute_frame_from_bars([b for d in sorted(tape)[:-1] for b in tape[d]])
        )
        last = sorted(tape)[-1]
        opening = H.minute_frame_from_bars(tape[last][:5])
        with pytest.raises(ValueError, match="sessions of history"):
            H.forecast_from({"model": None, "opening_minutes": 5}, history, opening, last)

    def test_too_few_opening_bars_is_refused(self):
        with pytest.raises(ValueError, match="first 5 minutes"):
            H.forecast_from(
                {"model": None, "opening_minutes": 5}, pd.DataFrame(),
                H.minute_frame_from_bars(_tape(date(2026, 7, 1), date(2026, 7, 1))[date(2026, 7, 1)][:3]),
                date(2026, 7, 1),
            )


_NOTEBOOK_INPUTS: dict = {}


def notebook_inputs(ticker: str):
    """The bundle, and the notebook's *raw* SIP files rolled up by the live
    path's code -- raw rather than the cleaned `minute.parquet`, so the session
    cleaning (half days included) is part of what is checked."""
    if ticker in _NOTEBOOK_INPUTS:
        return _NOTEBOOK_INPUTS[ticker]
    files = sorted((NOTEBOOK / "data" / ticker / "raw").glob("*.parquet"))
    if not files:
        pytest.skip(f"the HighLow_5m {ticker} notebook data is not on this machine")
    bundle = H.load_bundle(ticker)
    if bundle is None:
        pytest.skip(f"the HighLow {ticker} bundle is not installed")
    raw = pd.concat(pd.read_parquet(f) for f in files).sort_index()
    raw = raw[~raw.index.duplicated()].loc["2023-01-01":"2026-09-11 23:59"]
    raw = raw.tz_convert("America/New_York").between_time("09:30", "15:59")[H.OHLCV]
    history, _ = H.session_rollups(raw)
    _NOTEBOOK_INPUTS[ticker] = (bundle, raw, history)
    return _NOTEBOOK_INPUTS[ticker]


class TestAgainstTheNotebook:
    """The mirror contract: the notebook's minute tape, rolled up by the live
    path's code, reproduces the notebook package's forecasts. Not bit-exact --
    the N-BEATS half runs in float32 and the two venvs carry different pandas
    -- but within a millionth of a dollar, far inside anything a level reads."""

    @pytest.mark.parametrize(
        "ticker,day",
        [(t, d) for t, days in NOTEBOOK_FORECASTS.items() for d in sorted(days)],
    )
    def test_reproduces_the_notebook_forecast(self, ticker, day):
        bundle, raw, history = notebook_inputs(ticker)
        stamp = pd.Timestamp(day)
        opening = raw[raw.index.normalize().tz_localize(None) == stamp].iloc[:5]
        out = H.forecast_from(bundle, history, opening, stamp)
        high, low, adr = NOTEBOOK_FORECASTS[ticker][day]
        assert out["pred_high"] == pytest.approx(high, abs=1e-5)
        assert out["pred_low"] == pytest.approx(low, abs=1e-5)
        assert out["adr14_abs"] == pytest.approx(adr, abs=1e-9)

    @pytest.mark.parametrize("ticker", sorted(SHIPPED))
    def test_only_the_weighted_candidates_are_loaded(self, ticker):
        bundle, _, _ = notebook_inputs(ticker)
        assert bundle["kind"] == "highlow"
        assert bundle["ticker"] == ticker  # the file that loaded is the right one
        assert sorted(bundle["daily_models"]) == SHIPPED[ticker]

    def test_a_bundle_whose_shipped_candidate_reads_a_custom_group_is_refused(
        self, tmp_path, monkeypatch
    ):
        """INTC's linear median reads microstructure columns this module never
        builds; at weight 0 that is harmless, weighted it must not load."""
        import joblib

        src = H.model_path("INTC")
        if not src.exists():
            pytest.skip("the HighLow INTC bundle is not installed")
        H._register_unpickle_alias()
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            blob = joblib.load(src)
        blob["weights"] = {**blob["weights"], "linear": 1.0}
        target = tmp_path / src.name
        joblib.dump(blob, target)
        for extra in (src.with_name(f"{src.stem}_nhits.pt"), src.with_suffix(".json")):
            (tmp_path / extra.name).write_bytes(extra.read_bytes())
        assert H._build_bundle(target) is None
