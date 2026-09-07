"""The TimeToChange delta-momentum mirror: momentum, features, and the seams.

Two kinds of test here, and the distinction matters.

The ones that need neither the saved model nor the notebooks pin the *seams* --
what the app has to get right to hand the model correct inputs: the bar shape
becoming momlib's frame, a short history being refused rather than imputed
around, the previous sessions being fetched through the one call SimLab
patches, and the newest bar's regime coming from the minute before it.

The one that needs both pins the *mirror itself*, against momlib run on the
same bars in the same process. That is the test that would fail if
`momlib/features.py` changed and this copy did not, and it is exact to the last
bit rather than approximate -- the same contract `tests/test_dayrange_model.py`
holds with `dayrange` and `tests/test_persistence_model.py` with `mshift`.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agent_stonks import apple_models, clock, historical
from agent_stonks import momentum_change_model as M

NOTEBOOK = Path("/Users/aleksandra/Documents/playground/Code/FinNotebooks/TimeToChange")

# Every ticker this model is wired up for, from the registry rather than a list
# here -- adding one there should extend the mirror check rather than leave it
# silently covering the old set.
TICKERS = list(apple_models.MOMENTUM_CHANGE_TICKERS)

MARKET_TZ = "America/New_York"


# --------------------------------------------------------------- synthetic tape

def synthetic_bars(
    sessions: int = 7,
    bars_per_session: int = 390,
    first_day: str = "2026-07-13",
    seed: int = 11,
) -> "list[dict]":
    """`sessions` full regular sessions of app-shaped minute bars, UTC ISO.

    Deterministic and gently noisy, so every trailing window fills and the
    regime pipeline produces a mixture of all three regimes. These tests are
    about plumbing, not about whether the numbers are any good.
    """
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(first_day, periods=sessions)
    price = 100.0
    bars: "list[dict]" = []
    for day in days:
        start = pd.Timestamp(f"{day.date()} 09:30", tz=MARKET_TZ)
        for i in range(bars_per_session):
            price *= float(np.exp(rng.normal(0.0, 0.0004)))
            ts = (start + pd.Timedelta(minutes=i)).tz_convert("UTC")
            bars.append(
                {
                    "t": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "o": price, "h": price * 1.0002, "l": price * 0.9998,
                    "c": price, "v": 1000.0 + i,
                }
            )
    return bars


def session_date(bars: "list[dict]") -> pd.Timestamp:
    """The exchange-local date of the last bar in a bar list."""
    ts = pd.Timestamp(bars[-1]["t"]).tz_convert(MARKET_TZ)
    return pd.Timestamp(ts.date())


class FakeSymState:
    """The one thing `session_frame` reads off symbol state: locked bars."""

    def __init__(self, bars: "list[dict]") -> None:
        self.bars = list(bars)

    @property
    def lock(self):
        import threading

        return threading.Lock()


# ----------------------------------------------------------------- the bar shape

class TestFrameFromBars:
    def test_it_produces_momlibs_column_names(self):
        frame = M.frame_from_bars(synthetic_bars(sessions=1, bars_per_session=5))
        assert list(frame.columns) == ["Open", "High", "Low", "Close", "Volume", "day"]
        assert str(frame.index.tz) == MARKET_TZ

    def test_premarket_and_after_hours_bars_are_dropped(self):
        """momlib's own loaders keep 09:30-15:59 only, and every per-day
        aggregate -- `minute_of_day`, the day's open, the VWAP -- is defined
        against that grid. One 09:00 bar would shift all of them."""
        bars = synthetic_bars(sessions=1, bars_per_session=5)
        early = dict(bars[0])
        early["t"] = "2026-07-13T12:00:00Z"  # 08:00 ET
        late = dict(bars[0])
        late["t"] = "2026-07-13T21:30:00Z"  # 17:30 ET
        frame = M.frame_from_bars([early, *bars, late])
        assert len(frame) == 5

    def test_duplicate_timestamps_keep_the_last(self):
        bars = synthetic_bars(sessions=1, bars_per_session=3)
        revised = dict(bars[1])
        revised["c"] = 999.0
        frame = M.frame_from_bars([*bars, revised])
        assert len(frame) == 3
        assert frame["Close"].iloc[1] == 999.0

    def test_an_empty_list_still_has_the_columns(self):
        frame = M.frame_from_bars([])
        assert list(frame.columns) == ["Open", "High", "Low", "Close", "Volume", "day"]
        assert not len(frame)


# --------------------------------------------------------------- the history seam

class TestHistory:
    def _dated(self, monkeypatch, per_day: "dict[str, list[dict]]") -> list:
        """Patch the one fetch this model reads days through, and record the
        dates it asked for."""
        asked: list = []

        def fake(symbol, date, interval="1m"):
            asked.append(date)
            return per_day.get(date, [])

        monkeypatch.setattr(historical, "fetch_intraday_bars_for_date", fake)
        M.reset_history_cache()
        return asked

    def _day_bars(self, day: str) -> "list[dict]":
        bars = synthetic_bars(sessions=1, first_day=day)
        return bars

    def test_it_walks_back_until_it_has_six_sessions(self, monkeypatch):
        days = ["2026-07-20", "2026-07-21", "2026-07-22", "2026-07-23",
                "2026-07-24", "2026-07-27"]
        per_day = {d: self._day_bars(d) for d in days}
        asked = self._dated(monkeypatch, per_day)

        bars = M.history_bars("GOOGL", pd.Timestamp("2026-07-28"))
        frame = M.frame_from_bars(bars)
        assert sorted({str(d) for d in frame["day"]}) == days
        # Weekends cost a step of the walk and nothing else.
        assert "2026-07-25" in asked and "2026-07-26" in asked

    def test_a_short_session_is_skipped_rather_than_used(self, monkeypatch):
        """A half day has its own volatility and its own closing print, and
        `theta` is built from exactly those. momlib dropped such days from the
        training frame; letting one in here would move every regime."""
        half = self._day_bars("2026-07-27")[:200]
        per_day = {"2026-07-27": half}
        for d in ("2026-07-20", "2026-07-21", "2026-07-22", "2026-07-23",
                  "2026-07-24", "2026-07-17"):
            per_day[d] = self._day_bars(d)
        self._dated(monkeypatch, per_day)

        frame = M.frame_from_bars(M.history_bars("GOOGL", pd.Timestamp("2026-07-28")))
        kept = {str(d) for d in frame["day"]}
        assert "2026-07-27" not in kept
        assert "2026-07-17" in kept  # ...and the walk went one session further back

    def test_the_answer_is_fetched_once_per_session(self, monkeypatch):
        per_day = {
            d: self._day_bars(d)
            for d in ("2026-07-20", "2026-07-21", "2026-07-22", "2026-07-23",
                      "2026-07-24", "2026-07-27")
        }
        asked = self._dated(monkeypatch, per_day)

        for _ in range(3):
            M.history_bars("GOOGL", pd.Timestamp("2026-07-28"))
        assert len(asked) == len(set(asked))

    def test_two_symbols_do_not_evict_each_other(self, monkeypatch):
        """The cache expires on the date, not the key: evicting per key would
        make two streamed symbols re-download each other's history every bar."""
        per_day = {
            d: self._day_bars(d)
            for d in ("2026-07-20", "2026-07-21", "2026-07-22", "2026-07-23",
                      "2026-07-24", "2026-07-27")
        }
        self._dated(monkeypatch, per_day)
        day = pd.Timestamp("2026-07-28")

        M.history_bars("GOOGL", day)
        M.history_bars("INTC", day)
        assert ("GOOGL", day.date()) in M._history_cache
        assert ("INTC", day.date()) in M._history_cache

        # ...and rolling to the next session drops both.
        M.history_bars("GOOGL", pd.Timestamp("2026-07-29"))
        assert ("GOOGL", day.date()) not in M._history_cache

    def test_a_failing_fetch_costs_a_session_rather_than_the_run(self, monkeypatch):
        def fake(symbol, date, interval="1m"):
            raise RuntimeError("yfinance is down")

        monkeypatch.setattr(historical, "fetch_intraday_bars_for_date", fake)
        M.reset_history_cache()
        assert M.history_bars("GOOGL", pd.Timestamp("2026-07-28")) == []


class TestRequireHistory:
    def test_a_short_history_is_refused(self):
        frame = M.frame_from_bars(synthetic_bars(sessions=3))
        problem = M.require_history(frame, session_date(synthetic_bars(sessions=3)))
        assert problem is not None
        assert "2 of the 6" in problem

    def test_six_previous_sessions_are_enough(self):
        bars = synthetic_bars(sessions=7)
        frame = M.frame_from_bars(bars)
        assert M.require_history(frame, session_date(bars)) is None

    def test_it_counts_completed_sessions_not_bars(self):
        """One very long day is not six days: `theta` needs yesterday and
        `ret_5d` needs the session six back, and neither is a bar count."""
        frame = M.frame_from_bars(synthetic_bars(sessions=1, bars_per_session=390))
        problem = M.require_history(frame, session_date(synthetic_bars(sessions=1)))
        assert problem is not None and "0 of the 6" in problem


class TestSessionFrame:
    def test_history_is_concatenated_behind_today(self, monkeypatch):
        bars = synthetic_bars(sessions=7)
        today = session_date(bars)
        live = [b for b in bars if pd.Timestamp(b["t"]).tz_convert(MARKET_TZ).date()
                == today.date()]
        history = [b for b in bars if b not in live]

        monkeypatch.setattr(M, "history_bars", lambda *a, **k: history)
        clock.set_simulated(
            pd.Timestamp(f"{today.date()} 14:30", tz=MARKET_TZ).tz_convert("UTC")
            .to_pydatetime()
        )
        try:
            frame = M.session_frame(FakeSymState(live), "GOOGL")
        finally:
            clock.clear()

        assert len(frame) == len(bars)
        assert frame.index.is_monotonic_increasing
        assert M.sessions_before(frame, today) == 6

    def test_bars_from_other_days_in_the_buffer_are_not_taken_for_today(
        self, monkeypatch
    ):
        """The live buffer is trusted for today only; anything else in it would
        be counted twice against the fetched history."""
        bars = synthetic_bars(sessions=7)
        today = session_date(bars)
        monkeypatch.setattr(M, "history_bars", lambda *a, **k: [])
        clock.set_simulated(
            pd.Timestamp(f"{today.date()} 14:30", tz=MARKET_TZ).tz_convert("UTC")
            .to_pydatetime()
        )
        try:
            frame = M.session_frame(FakeSymState(bars), "GOOGL")
        finally:
            clock.clear()
        assert set(str(d) for d in frame["day"]) == {str(today.date())}


# ------------------------------------------------------------------- the bundle

class TestBundle:
    def test_a_missing_file_is_unavailable_rather_than_an_error(self, monkeypatch, tmp_path):
        monkeypatch.setenv("APPLE_MOMENTUM_CHANGE_MODEL_GOOGL", str(tmp_path / "nope.joblib"))
        M.reset_bundle_cache()
        assert M.load_bundle("GOOGL") is None
        M.reset_bundle_cache()

    def test_a_joblib_that_is_not_this_bundle_is_refused(self, monkeypatch, tmp_path):
        """Some other project's file sitting at this path would otherwise be
        asked for a prediction it cannot make."""
        joblib = pytest.importorskip("joblib")
        path = tmp_path / "stranger.joblib"
        joblib.dump({"something": "else"}, path)
        monkeypatch.setenv("APPLE_MOMENTUM_CHANGE_MODEL_GOOGL", str(path))
        M.reset_bundle_cache()
        assert M.load_bundle("GOOGL") is None
        M.reset_bundle_cache()

    def test_the_bare_override_only_answers_for_the_default_ticker(self, monkeypatch):
        """One file cannot be two models. Letting the bare variable answer for
        every symbol would hand an INTC run the GOOGL model in silence."""
        monkeypatch.setenv("APPLE_MOMENTUM_CHANGE_MODEL", "/tmp/whatever.joblib")
        assert M.model_path(M.DEFAULT_TICKER) == Path("/tmp/whatever.joblib")
        assert M.model_path("INTC").name == "momentum_change_INTC.joblib"

    def test_pipeline_params_fall_back_to_the_published_defaults(self):
        assert M.pipeline_params(None)["persist"] == 15
        assert M.pipeline_params({"pipeline_params": {"persist": 20}})["persist"] == 20


# ------------------------------------------------------------------- read_latest

@pytest.fixture
def scoring_bundle():
    """A bundle whose estimator answers with the mean of the row, so the
    plumbing can be checked without the saved file."""

    class Mean:
        def predict(self, X):
            return np.nan_to_num(X.to_numpy(float)).mean(axis=1)

    return {
        "estimator": Mean(),
        "feature_cols": list(M.FEATURE_COLS),
        "pipeline_params": dict(M.PIPELINE_DEFAULTS),
        "model_name": "Mean",
    }


class TestReadLatest:
    def _frame(self, sessions: int = 7):
        return M.frame_from_bars(synthetic_bars(sessions=sessions))

    def test_the_first_bars_of_a_session_are_unscorable(self, scoring_bundle):
        """`mom_15` needs fifteen bars and `mom_slope` five more, so the first
        twenty minutes have no prediction rather than an imputed one."""
        frame = self._frame()
        day = frame["day"].iloc[-1]
        early = frame[(frame["day"] < day) | (frame.groupby("day").cumcount() < 10)]
        read = M.read_latest(scoring_bundle, early)
        assert read["warming_up"] is True and read["pred"] is None

    def test_a_full_session_scores(self, scoring_bundle):
        read = M.read_latest(scoring_bundle, self._frame())
        assert read["warming_up"] is False
        assert read["pred"] is not None and np.isfinite(read["pred"])
        assert read["regime"] in (-1, 0, 1)

    def test_regime_before_is_the_previous_minute_of_the_same_day(self, scoring_bundle):
        """What the rules gate on. The bar's own regime already contains the
        move the model is being asked to predict."""
        frame = self._frame()
        scored, _ = M.prepare(frame, M.PIPELINE_DEFAULTS)
        read = M.read_latest(scoring_bundle, frame)
        assert read["regime_before"] == int(scored["regime"].iloc[-2])

    def test_scoring_one_bar_equals_scoring_the_frame(self, scoring_bundle):
        """Not an approximation: the estimator is a per-row function, so the
        trader reading one bar and an overlay scoring the day agree exactly."""
        frame = self._frame()
        scored, events = M.prepare(frame, M.PIPELINE_DEFAULTS)
        batch = M.score_bars(scoring_bundle, scored, events)
        read = M.read_latest(scoring_bundle, frame)
        assert read["pred"] == pytest.approx(float(batch.iloc[-1]), abs=0, rel=0)

    def test_an_empty_frame_reads_as_nothing(self, scoring_bundle):
        assert M.read_latest(scoring_bundle, M.frame_from_bars([])) is None


# ------------------------------------------------------------- against the notebook

def _momlib():
    """momlib, importable only where the notebooks are checked out."""
    if str(NOTEBOOK) not in sys.path:
        sys.path.insert(0, str(NOTEBOOK))
    return pytest.importorskip("momlib")


@pytest.mark.skipif(
    not NOTEBOOK.exists(), reason="the TimeToChange notebooks are not on this machine"
)
@pytest.mark.parametrize("ticker", TICKERS)
class TestAgainstMomlib:
    """The mirror contract, run on the archive momlib itself was trained on.

    Exact rather than approximate, and over every bar rather than a sample:
    the saved estimator is a function of these column definitions, and a
    feature that drifts by a rounding error still produces a confident number
    off the wrong input.
    """

    def _bars(self, ml, ticker: str):
        try:
            src = ml.load_bars_csv(ticker)
        except FileNotFoundError:
            pytest.skip(f"the weekly CSV archive for {ticker} is not on this machine")
        idx = src.index
        utc = (idx.tz_localize(MARKET_TZ) if idx.tz is None else idx).tz_convert("UTC")
        bars = [
            {"t": ts.strftime("%Y-%m-%dT%H:%M:%SZ"), "o": r.Open, "h": r.High,
             "l": r.Low, "c": r.Close, "v": r.Volume}
            for ts, r in zip(utc, src.itertuples(index=False))
        ]
        return src, bars

    def test_the_frame_the_app_builds_is_the_frame_momlib_loads(self, ticker):
        ml = _momlib()
        src, bars = self._bars(ml, ticker)
        mine = M.frame_from_bars(bars)
        assert len(mine) == len(src)
        assert (mine.index.tz_convert("UTC") == src.index.tz_convert("UTC")).all()
        for column in ("Open", "High", "Low", "Close", "Volume"):
            assert np.array_equal(mine[column].to_numpy(), src[column].to_numpy())

    def test_every_feature_matches_bit_for_bit(self, ticker):
        ml = _momlib()
        src, bars = self._bars(ml, ticker)
        params = M.pipeline_params(M.load_bundle(ticker))

        their_df, their_events, _ = ml.prepare(src, params)
        my_df, my_events = M.prepare(M.frame_from_bars(bars), params)
        assert len(my_events) == len(their_events)

        theirs = ml.build_live_features(
            their_df, their_events, n_prev_changes=params["n_prev_changes"]
        )
        mine = M.build_live_features(
            my_df, my_events, n_prev_changes=params["n_prev_changes"]
        )
        assert list(mine.columns) == list(theirs.columns)
        a, b = theirs.to_numpy(float), mine.to_numpy(float)
        assert np.array_equal(np.isnan(a), np.isnan(b))
        assert np.nanmax(np.abs(a - b)) == 0.0

    def test_the_saved_model_predicts_the_same_number(self, ticker):
        ml = _momlib()
        bundle = M.load_bundle(ticker)
        if bundle is None:
            pytest.skip(f"the {ticker} bundle is not installed")
        src, bars = self._bars(ml, ticker)
        params = M.pipeline_params(bundle)

        their_df, their_events, _ = ml.prepare(src, params)
        my_df, my_events = M.prepare(M.frame_from_bars(bars), params)
        theirs = ml.score_bars(bundle, their_df, their_events)
        mine = M.score_bars(bundle, my_df, my_events)
        assert int(mine.notna().sum()) == int(theirs.notna().sum()) > 0
        assert np.nanmax(np.abs(theirs.to_numpy(float) - mine.to_numpy(float))) == 0.0


@pytest.mark.skipif(
    not NOTEBOOK.exists(), reason="the TimeToChange notebooks are not on this machine"
)
@pytest.mark.parametrize("ticker", TICKERS)
def test_scoring_bar_by_bar_is_momlibs_confirm_lag_of_persist_minus_one(ticker):
    """The live event lag, pinned as an identity -- **and pinned off by one**.

    A frame ending at bar t contains an event at bar t - persist and nothing
    younger, because a change is only persistent once the new regime has held
    that long. momlib's switch for the same idea is `confirm_lag`, but
    `confirm_lag=N` re-stamps an event at the bar N later and the history
    lookup takes events *strictly* before the current bar -- so N models a lag
    of N + 1, and the live equivalent is `persist - 1`.

    Both halves are asserted, because the off-by-one is the thing that hides:
    on the two tree bundles `confirm_lag=persist` still agrees with the live
    path on most bars (a RandomForest quantises a small feature difference
    away), and it was only AAPL's Ridge that made it visible. A future reader
    "correcting" `live_confirm_lag` back to `persist` has to fail a test.

    The gap against the *training* view (`confirm_lag=0`) is the real
    training/live difference and cannot be closed; it is asserted to be
    non-zero so this stays a statement about the tape rather than about
    floating point.
    """
    ml = _momlib()
    bundle = M.load_bundle(ticker)
    if bundle is None:
        pytest.skip(f"the {ticker} bundle is not installed")
    try:
        src = ml.load_bars_csv(ticker)
    except FileNotFoundError:
        pytest.skip("the weekly CSV archive is not on this machine")

    idx = src.index
    utc = (idx.tz_localize(MARKET_TZ) if idx.tz is None else idx).tz_convert("UTC")
    frame = M.frame_from_bars(
        [
            {"t": ts.strftime("%Y-%m-%dT%H:%M:%SZ"), "o": r.Open, "h": r.High,
             "l": r.Low, "c": r.Close, "v": r.Volume}
            for ts, r in zip(utc, src.itertuples(index=False))
        ]
    )
    window = set(sorted(pd.unique(frame["day"]))[-(M.HISTORY_SESSIONS + 1):])
    sub = frame[[d in window for d in frame["day"]]]
    today = max(window)

    params = M.pipeline_params(bundle)
    scored, events = M.prepare(sub, params)
    lag = M.live_confirm_lag(bundle)
    assert lag == params["persist"] - 1

    curves = {
        name: M.score_bars(bundle, scored, events, confirm_lag=value).to_numpy(float)
        for name, value in
        (("live", lag), ("one_too_slow", params["persist"]), ("training", 0))
    }

    # Every seventh bar, not every seventeenth: the disagreement shows only on
    # the exact bar a change becomes visible, and a coarse sample walks past it.
    positions = [i for i, d in enumerate(sub["day"]) if d == today][::7]
    live = []
    kept = []
    for i in positions:
        read = M.read_latest(bundle, sub.iloc[: i + 1])
        if read["pred"] is None:
            continue
        live.append(read["pred"])
        kept.append(i)

    assert len(live) > 20
    live = np.array(live)

    def gap(name: str) -> float:
        return float(np.max(np.abs(live - curves[name][kept])))

    assert gap("live") == pytest.approx(0.0, abs=1e-12)
    assert gap("one_too_slow") > 1e-6
    assert gap("training") > 1e-6


@pytest.mark.skipif(
    not NOTEBOOK.exists(), reason="the TimeToChange notebooks are not on this machine"
)
@pytest.mark.parametrize("ticker", TICKERS)
def test_the_rules_fire_on_the_same_bars_as_the_notebook(ticker):
    """`MomentumChangeTrader`'s rules against `momlib/sim.py`'s, over the holdout week.

    The mirror tests above pin the model; this pins the strategy written on it.
    Same predictions, same thresholds, same momentum floor and stop -- so every
    entry and exit should land on the same *minute*. What deliberately differs
    is only the fill: the notebook reads a signal off bar t and fills at the
    open of t+1, while this ledger market-orders on bar t itself. So the
    comparison is signal bars, not prices, and the notebook's entry timestamps
    are shifted back one bar to get them.

    Note the trade counts (32 on GOOGL, 77 on INTC over five sessions): that is
    the momentum floor at -2 x theta churning one-minute round trips, which the
    notebook's own sweep documents and this reproduces rather than quietly
    fixes.
    """
    ml = _momlib()
    bundle = M.load_bundle(ticker)
    if bundle is None:
        pytest.skip(f"the {ticker} bundle is not installed")
    try:
        src = ml.load_bars_csv(ticker)
    except FileNotFoundError:
        pytest.skip("the weekly CSV archive is not on this machine")

    from agent_stonks.apple_trader import AppleTraderConfig, MomentumChangeTrader

    # The clock has to be pinned, and not as a formality: `_exit_reason` ends
    # with the closing-flatten rule, which reads the *wall* clock rather than
    # the bar. Run for real inside the last five minutes of a session it fires
    # on every held bar and the trader churns -- 82 entries against the
    # notebook's 32. Nothing here is about that rule (the notebook's `eod` exit
    # is a bar-index rule, excluded below), so mid-session is the honest pin.
    clock.set_simulated(datetime(2026, 8, 26, 15, 0, tzinfo=timezone.utc))

    params = M.pipeline_params(bundle)
    frame, events, _ = ml.prepare(src, params)
    # `live_confirm_lag`, not `persist` -- see the test above for the off-by-one.
    pred = ml.score_bars(bundle, frame, events, confirm_lag=M.live_confirm_lag(bundle))
    days = sorted(frame["day"].unique())[-5:]
    sessions = {d: ml.prepare_session(frame, pred, d) for d in days}
    notebook = ml.simulate_days(sessions, ticker=ticker)
    assert len(notebook) > 0

    trader = MomentumChangeTrader(AppleTraderConfig(model_key="momentum_change", ticker=ticker))
    entries, exits = [], []
    for day in days:
        session = sessions[day]
        theta = float(session["theta"].iloc[0])
        trader.entry = None
        for ts, row in session.iterrows():
            read = {
                "ts": ts,
                "price": float(row["Close"]),
                "pred": None if pd.isna(row["pred"]) else float(row["pred"]),
                "mom": None if pd.isna(row["mom"]) else float(row["mom"]),
                "theta": theta,
                "regime": None if pd.isna(row["regime"]) else int(row["regime"]),
                "regime_before": (
                    None if pd.isna(row["regime_before"]) else int(row["regime_before"])
                ),
            }
            if trader.entry is None:
                if trader._entry_signal(read):
                    trader.entry = {"price": read["price"], "bars": 0}
                    entries.append(ts)
            elif trader._exit_reason(read) is not None:
                trader.entry = None
                exits.append(ts)
        trader.entry = None

    clock.clear()
    one_bar = pd.Timedelta(minutes=1)
    assert entries == list(notebook["entry_time"] - one_bar)
    # The notebook's last exit of a session can be its `eod` rule, which fires
    # on the last actionable bar rather than on a rule this trader has (the
    # closing flatten is a clock rule and there is no clock here), so compare
    # the exits it does share.
    theirs = [t for t in (notebook["exit_time"] - one_bar) if t in set(exits)]
    assert len(theirs) >= len(notebook) - len(days)


@pytest.mark.skipif(
    not NOTEBOOK.exists(), reason="the TimeToChange notebooks are not on this machine"
)
def test_the_wired_up_tickers_are_the_ones_with_a_bundle():
    """A model listed in the registry with no file behind it is an installation
    the app reports as broken; catching it here says which retrain is missing.
    """
    missing = [t for t in TICKERS if not M.model_path(t).exists()]
    assert not missing, (
        f"no bundle for {', '.join(missing)} — re-run TimeToChange's "
        "scripts/train_ticker.py with this project's interpreter and "
        "--model-dir ../../Models"
    )
