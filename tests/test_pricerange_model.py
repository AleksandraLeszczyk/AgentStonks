"""The PriceRange2 mirror: features, assembly, refusals, and the bundle.

Three kinds of test here, and the distinction matters.

The ones that need neither the saved model nor the notebooks pin the **seams**
-- what this app has to get right to hand the model correct inputs. Today's row
carrying no outcome columns, a short history refused rather than extrapolated,
the cross-asset block refused rather than filled with NaNs, the opening-volume
walk stopping when it has enough.

The ones that need the bundle pin the **loader**: what makes a file usable, and
what makes it refuse rather than degrade.

The ones that need the notebooks too pin the **mirror itself**, and they are the
reason this file exists. `test_reproduces_the_shipped_panel_exactly` rebuilds
all 156 feature columns from PriceRange2's own session table and compares them
against the panel that project trained on -- every column, every one of its
1,164 sessions, on all three tickers, to 0.00e+00. If `pricerange/features.py`
ever changes and this copy does not, that is the test that fails.

`test_reproduces_the_notebook_forecast_exactly` closes the loop: it feeds
`forecast_session` the inputs a live 09:35 would give it and checks the
predicted levels *and* the two quantile edges the trading rule reads against
`pricerange.modeling`'s own answer for the same session.
"""

import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agent_stonks import apple_models, market_hours

P = pytest.importorskip("agent_stonks.pricerange_model")

NOTEBOOK = Path("/Users/aleksandra/Documents/playground/Code/FinNotebooks/PriceRange2")

# Every ticker PriceRange2 was run for, from the registry rather than a list
# here -- adding one there should extend the mirror check rather than leave it
# silently covering the old set.
TICKERS = list(apple_models.PRICERANGE_TICKERS)

# The session columns `build_features` reads. PriceRange2's panel carries them
# alongside the features, which is what makes the mirror check possible without
# re-running its data pipeline.
SESSION_COLUMNS = [
    "open", "high", "low", "close", "volume",
    "open_open", "open_high", "open_low", "ref", "open_volume", "open_rv", "open_ret",
]

needs_model = pytest.mark.skipif(
    not P.model_path().exists(), reason="the PriceRange2 bundle is not installed"
)


def _panel(ticker: str) -> "pd.DataFrame | None":
    path = NOTEBOOK / "data" / f"panel_{ticker.upper()}.parquet"
    return pd.read_parquet(path) if path.exists() else None


def _blocks(ticker: str) -> "dict[str, list[str]] | None":
    import json

    path = NOTEBOOK / "data" / f"panel_blocks_{ticker.upper()}.json"
    if not path.exists():
        return None
    return {k: v.split(",") for k, v in json.loads(path.read_text()).items()}


def _notebook_cross() -> "pd.DataFrame | None":
    """The cross-asset frame from the notebooks' own cache, via their code.

    Imported lazily and only for the mirror tests: `pricerange.sources.market`
    reads a cache this repo does not own, so every test that touches it skips
    cleanly on a machine without the notebooks.
    """
    if not NOTEBOOK.exists():
        return None
    if str(NOTEBOOK) not in sys.path:
        sys.path.insert(0, str(NOTEBOOK))
    try:
        from pricerange.sources import market  # type: ignore

        return market.cross_asset_daily("2022-01-03", "2026-09-09", cache=True)
    except Exception:
        return None


def synthetic_daily(n: int = 400, start: str = "2025-01-02") -> pd.DataFrame:
    """A daily series long enough to clear every trailing window.

    Deterministic and gently trending, so the feature table's last row is
    finite everywhere -- these tests are about plumbing, not about whether the
    numbers are any good.
    """
    dates = pd.bdate_range(start, periods=n, name="date")
    rng = np.random.default_rng(11)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, n)))
    frame = pd.DataFrame(
        {
            "open": close * (1 + rng.normal(0, 0.002, n)),
            "high": close * (1 + np.abs(rng.normal(0.006, 0.003, n))),
            "low": close * (1 - np.abs(rng.normal(0.006, 0.003, n))),
            "close": close,
            "volume": rng.uniform(4e7, 6e7, n),
        },
        index=dates,
    )
    frame["high"] = frame[["open", "high", "close"]].max(axis=1)
    frame["low"] = frame[["open", "low", "close"]].min(axis=1)
    return frame


def opening_bars(day: str = "2026-06-15", n: int = 5, base: float = 120.0):
    """`n` one-minute bars from 09:30 of `day`, in the app's frame shape."""
    start = pd.Timestamp(f"{day} 09:30", tz=market_hours.MARKET_TZ)
    index = pd.DatetimeIndex([start + pd.Timedelta(minutes=i) for i in range(n)])
    return pd.DataFrame(
        {
            "open": [base + 0.10 * i for i in range(n)],
            "high": [base + 0.10 * i + 0.15 for i in range(n)],
            "low": [base + 0.10 * i - 0.12 for i in range(n)],
            "close": [base + 0.10 * i + 0.05 for i in range(n)],
            "volume": [50_000.0 + 1_000 * i for i in range(n)],
            "minutes_from_open": [float(i) for i in range(n)],
        },
        index=index,
    )


def opening_volume(day: str = "2026-06-15", sessions: int = 21) -> pd.Series:
    """`sessions` days of opening volume ending the business day before `day`."""
    end = pd.Timestamp(day) - pd.tseries.offsets.BDay(1)
    index = pd.bdate_range(end=end, periods=sessions, name="date")
    return pd.Series(np.linspace(40_000, 60_000, sessions), index=index)


def cross_frame(day: str = "2026-06-15", n: int = 400, drop: "set[str] | None" = None,
                without_opens: "set[str] | None" = None) -> pd.DataFrame:
    """A complete 17-series cross-asset frame, minus anything named."""
    drop, without_opens = drop or set(), without_opens or set()
    bars_by_name, opens = {}, {}
    dates = pd.bdate_range(end=pd.Timestamp(day) - pd.tseries.offsets.BDay(1), periods=n)
    for i, name in enumerate(P.CROSS_SYMBOLS.values()):
        if name in drop:
            continue
        rng = np.random.default_rng(100 + i)
        close = 50 * np.exp(np.cumsum(rng.normal(0.0002, 0.008, n)))
        bars_by_name[name] = [
            {"t": d.date().isoformat(), "o": float(c * 0.999), "h": float(c * 1.004),
             "l": float(c * 0.996), "c": float(c), "v": 1e6}
            for d, c in zip(dates, close)
        ]
        if name not in without_opens:
            opens[name] = float(close[-1] * 1.001)
    return P.cross_frame_from_bars(bars_by_name, day, opens)


# ---------------------------------------------------------------- the mirror


class TestAgainstTheNotebook:
    """The contract that makes the saved model usable at all.

    PriceRange2's panel carries both the raw session columns and the 156
    features built from them, so the mirror can be checked column by column
    without re-running that project's data pipeline.
    """

    @pytest.mark.parametrize("ticker", TICKERS)
    def test_reproduces_the_shipped_panel_exactly(self, ticker):
        panel = _panel(ticker)
        blocks = _blocks(ticker)
        cross = _notebook_cross()
        if panel is None or blocks is None or cross is None:
            pytest.skip(f"the PriceRange2 {ticker} panel is not on this machine")

        built = P.build_features(panel[SESSION_COLUMNS], cross)
        wanted = [c for b in ("history", "opening", "cross") for c in blocks[b]]
        assert len(wanted) == 156

        worst, offender = 0.0, ""
        for column in wanted:
            assert column in built.columns, f"the mirror does not produce {column}"
            theirs = panel[column].astype("float64")
            ours = built[column].astype("float64")
            # A column that is NaN in one and not the other is a mismatch even
            # where the finite values agree -- LightGBM treats missing as its
            # own branch, so the two are different inputs.
            assert (theirs.isna() == ours.isna()).all(), f"{column} disagrees on NaNs"
            both = theirs.notna()
            if both.any():
                gap = float((theirs[both] - ours[both]).abs().max())
                if gap > worst:
                    worst, offender = gap, column
        assert worst == 0.0, f"{offender} differs by {worst:.3e}"

    @pytest.mark.parametrize("ticker", TICKERS)
    @needs_model
    def test_reproduces_the_notebook_forecast_exactly(self, ticker):
        """`forecast_session` against `pricerange.modeling` on the same day.

        Fed the way a live 09:35 would feed it -- daily bars, five minute bars,
        the cross frame -- rather than the assembled panel row, so the assembly
        seams are inside the comparison rather than assumed correct.
        """
        panel = _panel(ticker)
        blocks = _blocks(ticker)
        cross_long = _notebook_cross()
        minutes_path = NOTEBOOK / "data" / "cache" / f"{ticker}_1min_2022-01-03_2026-09-09.parquet"
        if panel is None or blocks is None or cross_long is None or not minutes_path.exists():
            pytest.skip(f"the PriceRange2 {ticker} notebook data is not on this machine")
        bundle = P.load_bundle(ticker)
        if bundle is None:
            pytest.skip(f"the PriceRange2 {ticker} bundle is not installed")

        if str(NOTEBOOK) not in sys.path:
            sys.path.insert(0, str(NOTEBOOK))
        from pricerange import modeling as M  # type: ignore

        models = bundle["models"]
        feats = bundle["feature_cols"]
        point = {e: models[e] for e in M.EDGES}
        qmodels = {
            (e, q): models[f"q{int(q * 100)}_{e}"]
            for e in M.EDGES for q in M.QUANTILES
            if f"q{int(q * 100)}_{e}" in models
        }

        minutes = pd.read_parquet(minutes_path).between_time("09:30", "15:59")
        daily_bars = [
            {"t": d.date().isoformat(), "o": r.open, "h": r.high,
             "l": r.low, "c": r.close, "v": r.volume}
            for d, r in panel[["open", "high", "low", "close", "volume"]].iterrows()
        ]
        cross_bars = {
            symbol: [
                {"t": d.date().isoformat(), "o": r.open, "h": r.high,
                 "l": r.low, "c": r.close, "v": r.volume}
                for d, r in group.set_index("date")[
                    ["open", "high", "low", "close", "volume"]
                ].iterrows()
            ]
            for symbol, group in cross_long.groupby("symbol")
        }

        for day in panel.index[-3:]:
            row = panel.loc[[day]]
            theirs = M.to_levels(M.predict(point, feats, row), row).iloc[0]
            quantiles = M.predict_quantiles(qmodels, feats, row).iloc[0]
            ref = float(row["ref"].iloc[0])

            iso = day.date().isoformat()
            ours = P.forecast_session(
                bundle,
                P.daily_frame_from_bars([b for b in daily_bars if b["t"] < iso]),
                minutes[
                    (minutes.index.normalize().tz_localize(None) == day)
                    & (minutes.index.time < pd.Timestamp("09:35").time())
                ],
                day,
                cross=P.cross_frame_from_bars(
                    cross_bars,
                    day,
                    opens={
                        symbol: float(group.loc[group["date"] == day, "open"].iloc[0])
                        for symbol, group in cross_long.groupby("symbol")
                        if (group["date"] == day).any()
                    },
                ),
                open_price=float(panel.loc[day, "open"]),
                opening_volume=panel.loc[panel.index < day, "open_volume"].tail(21),
            )
            assert ours["pred_high"] == pytest.approx(theirs["pred_high"], abs=0)
            assert ours["pred_low"] == pytest.approx(theirs["pred_low"], abs=0)
            # The two the trading rule actually rests orders at.
            assert ours["buy_edge"] == pytest.approx(
                ref * np.exp(quantiles["y_low_q75"]), abs=0
            )
            assert ours["sell_edge"] == pytest.approx(
                ref * np.exp(quantiles["y_high_q25"]), abs=0
            )


# ---------------------------------------------------------------- the seams


class TestSessionFrame:
    def test_todays_row_carries_no_outcome_columns(self):
        """The day's high, low, close and volume do not exist at 09:35.

        Filling them from the opening window would look harmless -- nothing
        reads them for *today's* forecast, since every rolled statistic is
        lagged -- and would corrupt tomorrow's if the frame were reused.
        """
        frame = P.session_frame(
            synthetic_daily(), opening_bars(), "2026-06-15", open_price=121.5,
            opening_volume=opening_volume(),
        )
        today = frame.loc[pd.Timestamp("2026-06-15")]
        for column in ("high", "low", "close", "volume"):
            assert pd.isna(today[column]), f"{column} should be unknown at 09:35"

    def test_todays_row_carries_what_is_known_by_0935(self):
        window = opening_bars()
        frame = P.session_frame(
            synthetic_daily(), window, "2026-06-15", open_price=121.5,
            opening_volume=opening_volume(),
        )
        today = frame.loc[pd.Timestamp("2026-06-15")]
        assert today["open"] == 121.5          # the official auction print
        assert today["open_open"] == 121.5
        assert today["open_high"] == window["high"].max()
        assert today["open_low"] == window["low"].min()
        assert today["ref"] == window["close"].iloc[-1]
        assert today["open_volume"] == window["volume"].sum()
        assert today["open_ret"] == pytest.approx(np.log(today["ref"] / 121.5))

    def test_the_first_minute_bar_stands_in_for_a_missing_open(self):
        window = opening_bars()
        frame = P.session_frame(
            synthetic_daily(), window, "2026-06-15", opening_volume=opening_volume()
        )
        today = frame.loc[pd.Timestamp("2026-06-15")]
        assert today["open"] == window["open"].iloc[0]
        assert today["open_open"] == window["open"].iloc[0]

    def test_prior_rows_take_their_opening_volume_from_the_history(self):
        """The one opening column the past needs, for `open_vol_vs_21`."""
        volumes = opening_volume()
        frame = P.session_frame(
            synthetic_daily(), opening_bars(), "2026-06-15",
            opening_volume=volumes,
        )
        prior = frame[frame.index < pd.Timestamp("2026-06-15")]
        assert prior["open_volume"].notna().sum() == len(volumes)
        for day, value in volumes.items():
            assert prior.loc[day, "open_volume"] == value

    def test_a_session_at_or_after_today_is_dropped_from_the_history(self):
        daily = synthetic_daily()
        day = daily.index[-1]
        frame = P.session_frame(
            daily, opening_bars(day.date().isoformat()), day,
            opening_volume=opening_volume(day.date().isoformat()),
        )
        # The stored row for `day` is replaced by the partial one, not kept
        # alongside it -- otherwise the index would carry it twice.
        assert (frame.index == day).sum() == 1
        assert pd.isna(frame.loc[day, "close"])


class TestRequirements:
    def test_a_short_history_is_refused_rather_than_extrapolated(self):
        short = synthetic_daily(n=P.MIN_DAILY_SESSIONS - 1)
        problem = P.require_history(short)
        assert problem is not None
        assert str(P.MIN_DAILY_SESSIONS) in problem

    def test_a_long_enough_history_passes(self):
        assert P.require_history(synthetic_daily(n=P.MIN_DAILY_SESSIONS)) is None

    def test_the_partial_row_does_not_count_towards_the_history(self):
        """`session_frame` appends a row whose close is NaN; it is not a session.

        Counting it would let a frame one short of the warm-up through, and the
        252-day windows would then be built on 251 days of data.
        """
        daily = synthetic_daily(n=P.MIN_DAILY_SESSIONS - 1)
        frame = P.session_frame(
            daily, opening_bars(), "2026-06-15", opening_volume=opening_volume()
        )
        assert len(frame) == P.MIN_DAILY_SESSIONS
        assert P.require_history(frame) is not None

    def test_a_complete_cross_frame_passes(self):
        assert P.require_cross(cross_frame(), "2026-06-15") is None

    def test_an_empty_cross_frame_is_refused(self):
        problem = P.require_cross(pd.DataFrame(), "2026-06-15")
        assert problem is not None
        assert "17" in problem and "110" in problem

    def test_a_missing_series_is_refused_and_named(self):
        problem = P.require_cross(cross_frame(drop={"vix", "TLT"}), "2026-06-15")
        assert problem is not None
        assert "TLT" in problem and "vix" in problem

    def test_a_missing_opening_print_is_refused_and_named(self):
        """`{sym}_gap` is the block's only same-day column, and it matters.

        `UNG_gap` is the second most important feature on AAPL by gain, so a
        cross series present but without today's open is not "nearly there".
        """
        problem = P.require_cross(cross_frame(without_opens={"UNG"}), "2026-06-15")
        assert problem is not None
        assert "UNG" in problem and "gap" in problem

    def test_twenty_sessions_of_opening_volume_is_refused(self):
        """One short is refused, because `rolling(21)` returns NaN, not "nearly".

        This is the exact case the live yfinance path always lands in: 30
        calendar days of 1-minute history is 20 trading sessions.
        """
        problem = P.require_opening_volume(
            opening_volume(sessions=P.OPENING_HISTORY_SESSIONS - 1), "2026-06-15"
        )
        assert problem is not None
        assert "20" in problem and "21" in problem

    def test_twenty_one_sessions_passes(self):
        assert P.require_opening_volume(opening_volume(), "2026-06-15") is None

    def test_sessions_on_or_after_today_do_not_count(self):
        volumes = opening_volume(sessions=P.OPENING_HISTORY_SESSIONS)
        volumes[pd.Timestamp("2026-06-15")] = 55_000.0
        # 22 rows, but one of them is today -- still one short of the window.
        assert P.require_opening_volume(volumes.iloc[1:], "2026-06-15") is not None

    def test_nothing_at_all_is_refused(self):
        assert P.require_opening_volume(None, "2026-06-15") is not None


class TestCrossFrame:
    def test_todays_row_carries_only_the_open(self):
        """Everything else in the block is lagged, so today's OHLC is not read.

        Guessing at it would be inventing data the model would then split on.
        """
        frame = cross_frame()
        today = frame[frame["date"] == pd.Timestamp("2026-06-15")]
        assert len(today) == len(P.CROSS_SYMBOLS)
        assert today["open"].notna().all()
        for column in ("high", "low", "close", "volume"):
            assert today[column].isna().all()

    def test_stored_rows_on_or_after_the_session_are_dropped(self):
        """A replay must not see a cross-asset bar from after the day it draws."""
        bars = [
            {"t": "2026-06-12", "o": 1.0, "h": 1.1, "l": 0.9, "c": 1.0, "v": 1.0},
            {"t": "2026-06-15", "o": 2.0, "h": 2.1, "l": 1.9, "c": 2.0, "v": 1.0},
            {"t": "2026-06-16", "o": 3.0, "h": 3.1, "l": 2.9, "c": 3.0, "v": 1.0},
        ]
        frame = P.cross_frame_from_bars({"SPY": bars}, "2026-06-15", opens={"SPY": 9.0})
        assert sorted(str(d.date()) for d in frame["date"]) == ["2026-06-12", "2026-06-15"]
        # ...and the surviving same-day row is the *open* that was passed in,
        # not the stored bar's outcome.
        today = frame[frame["date"] == pd.Timestamp("2026-06-15")].iloc[0]
        assert today["open"] == 9.0
        assert pd.isna(today["close"])

    def test_an_empty_input_gives_an_empty_frame_with_the_right_columns(self):
        frame = P.cross_frame_from_bars({}, "2026-06-15")
        assert frame.empty
        assert list(frame.columns) == [
            "date", "symbol", "open", "high", "low", "close", "volume"
        ]

    def test_the_index_names_are_the_models_column_names(self):
        """`vix`, not `^VIX` -- the feature is called `vix_level`."""
        assert P.CROSS_SYMBOLS["^VIX"] == "vix"
        assert P.CROSS_SYMBOLS["DX-Y.NYB"] == "dxy"
        assert P.CROSS_SYMBOLS["SPY"] == "SPY"
        assert len(P.CROSS_SYMBOLS) == 17


class TestFetchCrossFrame:
    """The live fetch, with `historical` stubbed."""

    def _stub(self, monkeypatch, today="2026-06-15"):
        from agent_stonks import clock, historical

        bars = [
            {"t": d.date().isoformat(), "o": 1.0, "h": 1.1, "l": 0.9, "c": 1.0, "v": 1.0}
            for d in pd.bdate_range(end=pd.Timestamp(today) - pd.tseries.offsets.BDay(1),
                                    periods=30)
        ]
        monkeypatch.setattr(historical, "fetch_daily_ohlc_bars", lambda s, days=0: bars)
        monkeypatch.setattr(historical, "fetch_session_open", lambda s, ttl_sec=0: 7.0)
        monkeypatch.setattr(
            clock, "now",
            lambda: datetime.fromisoformat(f"{today}T14:00:00+00:00"),
        )

    def test_todays_call_attaches_the_opening_prints(self, monkeypatch):
        self._stub(monkeypatch)
        frame = P.fetch_cross_frame("2026-06-15")
        today = frame[frame["date"] == pd.Timestamp("2026-06-15")]
        assert len(today) == len(P.CROSS_SYMBOLS)
        assert (today["open"] == 7.0).all()
        assert P.require_cross(frame, "2026-06-15") is None

    def test_a_past_session_gets_no_opens_rather_than_todays(self, monkeypatch):
        """The seam this guard exists for.

        `historical.fetch_session_open` has no date parameter -- it answers
        about today whatever it is asked. Stapling this morning's SPY open onto
        a session from last week would produce a `SPY_gap` computed from two
        days that never met: a plausible number with nothing behind it, on the
        block's only same-day column. Refusing is the honest outcome.
        """
        self._stub(monkeypatch, today="2026-06-15")
        frame = P.fetch_cross_frame("2026-06-08")
        assert frame[frame["date"] == pd.Timestamp("2026-06-08")].empty
        problem = P.require_cross(frame, "2026-06-08")
        assert problem is not None and "opening print" in problem


class TestOpeningVolumeHistory:
    """The walk that fetches 21 opening windows, with the REST call stubbed."""

    def _stub(self, monkeypatch, *, sessions_with_bars=None, record=None):
        from agent_stonks import agent as agent_mod

        def fake(symbol, timeframe, start, end, key, secret, feed, limit=200):
            day = pd.Timestamp(start).tz_convert(market_hours.MARKET_TZ).normalize()
            if record is not None:
                record.append((symbol, day.date(), timeframe, limit))
            if sessions_with_bars is not None and day.date() not in sessions_with_bars:
                return []
            return [{"t": start.isoformat(), "v": 1000.0}] * 5

        monkeypatch.setattr(agent_mod, "fetch_bars_window", fake)

    def test_it_collects_the_window_it_needs_and_stops(self, monkeypatch):
        calls: list = []
        self._stub(monkeypatch, record=calls)
        P.reset_opening_volume_cache()
        volumes = P.fetch_opening_volume_history("AAPL", "2026-06-15", "k", "s")
        assert len(volumes) == P.OPENING_HISTORY_SESSIONS
        assert (volumes == 5000.0).all()
        # Every request is for one session's opening window, not a month of bars.
        assert all(timeframe == "1Min" for _, _, timeframe, _ in calls)
        assert all(limit <= 2 * P.OPENING_MINUTES for *_, limit in calls)
        # ...and it never looks forward.
        assert all(day < date(2026, 6, 15) for _, day, _, _ in calls)

    def test_weekends_cost_nothing(self, monkeypatch):
        calls: list = []
        self._stub(monkeypatch, record=calls)
        P.reset_opening_volume_cache()
        P.fetch_opening_volume_history("AAPL", "2026-06-15", "k", "s")
        assert all(day.weekday() < 5 for _, day, _, _ in calls)

    def test_a_holiday_is_skipped_rather_than_recorded_as_zero(self, monkeypatch):
        """A zero would go straight into a 21-session mean."""
        wanted = {
            d.date() for d in pd.bdate_range(end="2026-06-12", periods=30)
        } - {date(2026, 6, 11)}
        self._stub(monkeypatch, sessions_with_bars=wanted)
        P.reset_opening_volume_cache()
        volumes = P.fetch_opening_volume_history("AAPL", "2026-06-15", "k", "s")
        assert date(2026, 6, 11) not in {d.date() for d in volumes.index}
        assert len(volumes) == P.OPENING_HISTORY_SESSIONS

    def test_without_credentials_it_returns_nothing_rather_than_guessing(
        self, monkeypatch
    ):
        calls: list = []
        self._stub(monkeypatch, record=calls)
        P.reset_opening_volume_cache()
        volumes = P.fetch_opening_volume_history("AAPL", "2026-06-15", None, None)
        assert volumes.empty
        assert calls == []
        assert P.require_opening_volume(volumes, "2026-06-15") is not None

    def test_the_answer_is_cached_for_the_session(self, monkeypatch):
        calls: list = []
        self._stub(monkeypatch, record=calls)
        P.reset_opening_volume_cache()
        P.fetch_opening_volume_history("AAPL", "2026-06-15", "k", "s")
        first = len(calls)
        P.fetch_opening_volume_history("AAPL", "2026-06-15", "k", "s")
        assert len(calls) == first

    def test_a_new_session_evicts_the_old_one(self, monkeypatch):
        self._stub(monkeypatch)
        P.reset_opening_volume_cache()
        P.fetch_opening_volume_history("AAPL", "2026-06-15", "k", "s")
        P.fetch_opening_volume_history("AAPL", "2026-06-16", "k", "s")
        assert all(key[1] == date(2026, 6, 16) for key in P._opening_volume_cache)


class TestVolumeScaleWarning:
    def test_iex_is_warned_about(self):
        warning = P.volume_scale_warning("iex")
        assert warning is not None
        assert "open_vol_share" in warning

    def test_the_same_feed_ratio_is_named_as_unaffected(self):
        """The precise claim: one of the two volume features is fine on IEX.

        `open_vol_vs_21` divides today's opening volume by the mean of 21
        sessions of opening volume from the same tape, so the feed's scale
        cancels. Saying both were broken would be easier and wrong.
        """
        assert "open_vol_vs_21" in P.volume_scale_warning("iex")

    @pytest.mark.parametrize("feed", ["sip", "yfinance", "SIP", None, ""])
    def test_consolidated_tapes_are_not_warned_about(self, feed):
        assert P.volume_scale_warning(feed) is None


class TestForecastRefusals:
    """Every way `forecast_session` declines, rather than answering off NaNs."""

    def _call(self, **overrides):
        kwargs = dict(
            history=synthetic_daily(),
            opening_bars=opening_bars(),
            session_date="2026-06-15",
            cross=cross_frame(),
            open_price=121.5,
            opening_volume=opening_volume(),
        )
        kwargs.update(overrides)
        bundle = {"opening_minutes": 5, "feature_cols": [], "models": {}, "quantiles": []}
        return P.forecast_session(bundle, **kwargs)

    def test_a_short_opening_window_is_refused(self):
        with pytest.raises(ValueError, match="only 3 bars"):
            self._call(opening_bars=opening_bars(n=3))

    def test_a_short_history_is_refused(self):
        with pytest.raises(ValueError, match="252-day"):
            self._call(history=synthetic_daily(n=100))

    def test_missing_opening_volume_is_refused(self):
        with pytest.raises(ValueError, match="opening volume"):
            self._call(opening_volume=None)

    def test_missing_cross_data_is_refused(self):
        with pytest.raises(ValueError, match="cross-asset"):
            self._call(cross=None)

    def test_a_drifted_mirror_is_refused_by_name(self):
        """A model column the feature code no longer produces."""
        bundle = {
            "opening_minutes": 5,
            "feature_cols": ["open_range", "a_column_that_moved_away"],
            "models": {},
            "quantiles": [],
        }
        with pytest.raises(ValueError, match="a_column_that_moved_away"):
            P.forecast_session(
                bundle, synthetic_daily(), opening_bars(), "2026-06-15",
                cross=cross_frame(), opening_volume=opening_volume(),
            )


# ---------------------------------------------------------------- the bundle


class TestBundle:
    @needs_model
    def test_it_loads_with_both_edges_and_all_six_quantiles(self):
        bundle = P.load_bundle()
        assert bundle is not None
        assert set(P.EDGES) <= set(bundle["models"])
        assert bundle["quantiles"] == [0.25, 0.5, 0.75]
        assert P.has_quantiles(bundle)
        assert len(bundle["feature_cols"]) == 156

    @needs_model
    def test_it_records_the_symbol_the_file_itself_claims(self):
        """Not the ticker that asked -- a mismatch should be visible."""
        assert P.load_bundle("INTC")["ticker"] == "INTC"

    def test_a_missing_file_is_none_rather_than_an_exception(self, tmp_path):
        assert P._build_bundle(tmp_path / "nothing.joblib") is None

    def test_a_bundle_without_the_point_estimators_is_refused(self, tmp_path):
        """`y_high` and `y_low` *are* the forecast; quantiles alone are not it."""
        joblib = pytest.importorskip("joblib")
        path = tmp_path / "pricerange2_test.joblib"
        joblib.dump({"models": {"q75_y_low": object()}, "feature_cols": ["a"]}, path)
        assert P._build_bundle(path) is None

    def test_a_feature_list_the_metadata_disowns_is_refused(self, tmp_path):
        """A half-written file or a mismatched retrain, caught before it predicts."""
        joblib = pytest.importorskip("joblib")
        path = tmp_path / "pricerange2_test.joblib"
        joblib.dump(
            {"models": {"y_high": object(), "y_low": object()}, "feature_cols": ["a", "b"]},
            path,
        )
        path.with_suffix(".json").write_text('{"n_features": 156}')
        assert P._build_bundle(path) is None

    def test_missing_quantiles_are_recorded_rather_than_fatal(self, tmp_path):
        """The point forecast is what was scored, so it still loads.

        The trader checks `has_quantiles` before arming, because the *levels*
        it rests orders at are the quantiles -- but the chart overlay and the
        catalogue can both use a point-only bundle.
        """
        joblib = pytest.importorskip("joblib")
        path = tmp_path / "pricerange2_test.joblib"
        joblib.dump(
            {"models": {"y_high": object(), "y_low": object()}, "feature_cols": ["a"]},
            path,
        )
        bundle = P._build_bundle(path)
        assert bundle is not None
        assert bundle["quantiles"] == []
        assert not P.has_quantiles(bundle)


class TestModelPath:
    def test_the_filename_is_the_notebooks_lower_case_one(self):
        """The mirror contract reaches the filename too: PriceRange2 writes
        `pricerange2_aapl.joblib`, so that is what the store looks for."""
        assert P.model_path("AAPL").name == "pricerange2_aapl.joblib"
        assert P.model_path("INTC").name == "pricerange2_intc.joblib"
        assert P.metadata_path(P.model_path("GOOG")).name == "pricerange2_goog.json"

    def test_the_environment_override_keeps_the_upper_case_symbol(self, monkeypatch):
        """A file is named by a notebook; an env var is typed by a person."""
        monkeypatch.setenv("APPLE_PRICERANGE_MODEL_INTC", "/tmp/elsewhere.joblib")
        P.reset_bundle_cache()
        assert P.model_path("INTC") == Path("/tmp/elsewhere.joblib")
        # ...and the bare key answers for the default ticker only.
        monkeypatch.setenv("APPLE_PRICERANGE_MODEL", "/tmp/default.joblib")
        assert P.model_path("AAPL") == Path("/tmp/default.joblib")
        assert P.model_path("GOOG").name == "pricerange2_goog.joblib"
        P.reset_bundle_cache()
