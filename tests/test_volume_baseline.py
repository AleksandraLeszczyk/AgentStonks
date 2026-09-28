"""The volume panel's "usual volume" reference."""

import statistics
from datetime import date, datetime, timedelta

import pandas as pd
import pytest

from agent_stonks import historical

from agent_stonks.volume_baseline import (
    DEFAULT_VOLUME_BASELINE,
    VOLUME_BASELINE_WINDOWS,
    WEEK_SESSIONS,
    lookback_days,
    minute_volume_baseline,
    volume_band,
)

OPEN_UTC = "T13:30:00+00:00"  # 09:30 ET


def bars(date: str, minutes, volume, price: float = 100.0) -> list[dict]:
    """Minute bars for one ET session, `minutes` past the 09:30 open."""
    base = datetime.fromisoformat(date + OPEN_UTC)
    vol = volume if callable(volume) else (lambda _m: volume)
    return [
        {
            "t": (base + timedelta(minutes=m)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "o": price, "h": price + 1, "l": price - 1, "c": price,
            "v": float(vol(m)),
        }
        for m in minutes
    ]


WEEK = ["2026-09-14", "2026-09-15", "2026-09-16", "2026-09-17", "2026-09-18"]


@pytest.fixture
def history() -> list[dict]:
    """Five full sessions: a busy first ten minutes, a quiet rest."""
    out: list[dict] = []
    for day in WEEK:
        out += bars(day, range(390), lambda m: 5000 if m < 10 else 1000)
    return out


@pytest.fixture
def today() -> list[dict]:
    return bars("2026-09-21", range(30), lambda m: 4000 if m < 10 else 900)


class TestTheWindowChoice:
    def test_off_draws_nothing(self, today, history):
        assert minute_volume_baseline("off", today, history) is None

    def test_an_unknown_window_draws_nothing(self, today, history):
        assert minute_volume_baseline("fortnight", today, history) is None

    def test_every_window_is_labelled(self):
        assert all(VOLUME_BASELINE_WINDOWS.values())

    def test_the_default_is_a_real_window(self):
        assert DEFAULT_VOLUME_BASELINE in VOLUME_BASELINE_WINDOWS
        assert DEFAULT_VOLUME_BASELINE != "off"

    def test_only_prior_windows_need_a_fetch(self):
        assert lookback_days("off") == 0
        assert lookback_days("session") == 0
        assert lookback_days("yesterday") > 0
        assert lookback_days("week") > lookback_days("yesterday")


class TestThisSession:
    def test_mean_is_todays_own_average(self, today, history):
        result = minute_volume_baseline("session", today, history)
        expected = (10 * 4000 + 20 * 900) / 30
        assert result["mean_per_minute"] == pytest.approx(expected)

    def test_has_no_shape_of_its_own(self, today, history):
        # It would be a tracing of the very bars it is drawn over.
        assert minute_volume_baseline("session", today, history)["per_minute"] == {}

    def test_ignores_the_history(self, today, history):
        with_history = minute_volume_baseline("session", today, history)
        alone = minute_volume_baseline("session", today, [])
        assert with_history["mean_per_minute"] == alone["mean_per_minute"]

    def test_no_bars_yet_means_no_baseline(self):
        assert minute_volume_baseline("session", [], []) is None


class TestYesterday:
    def test_reads_only_the_last_completed_session(self, today, history):
        result = minute_volume_baseline("yesterday", today, history, today="2026-09-21")
        assert result["sessions"] == 1
        assert result["dates"] == ["2026-09-18"]

    def test_averages_each_clock_minute(self, today, history):
        result = minute_volume_baseline("yesterday", today, history, today="2026-09-21")
        assert result["per_minute"][9 * 60 + 30] == pytest.approx(5000)
        assert result["per_minute"][12 * 60] == pytest.approx(1000)


class TestLastTradingWeek:
    def test_spans_five_sessions(self, today, history):
        result = minute_volume_baseline("week", today, history, today="2026-09-21")
        assert result["sessions"] == WEEK_SESSIONS
        assert result["dates"] == WEEK

    def test_keeps_the_newest_sessions_when_more_are_offered(self, today, history):
        older = bars("2026-09-09", range(390), 99999) + history
        result = minute_volume_baseline("week", today, older, today="2026-09-21")
        assert "2026-09-09" not in result["dates"]
        assert result["sessions"] == WEEK_SESSIONS

    def test_a_short_week_averages_the_days_that_traded(self, today):
        # Three sessions, not five diluted by two that never happened.
        short = []
        for day in WEEK[:3]:
            short += bars(day, range(390), 2000)
        result = minute_volume_baseline("week", today, short, today="2026-09-21")
        assert result["sessions"] == 3
        assert result["mean_per_minute"] == pytest.approx(2000)

    def test_mean_is_the_per_minute_average(self, today, history):
        result = minute_volume_baseline("week", today, history, today="2026-09-21")
        expected = (10 * 5000 + 380 * 1000) / 390
        assert result["mean_per_minute"] == pytest.approx(expected)

    def test_the_open_stands_above_the_mean(self, today, history):
        result = minute_volume_baseline("week", today, history, today="2026-09-21")
        # The whole point of the per-minute shape: 09:31 is not a typical minute.
        assert result["per_minute"][9 * 60 + 31] > result["mean_per_minute"] * 4

    def test_a_thin_minute_is_not_averaged_down(self, today):
        # One session prints 09:45, the others do not. The minute is thin, not
        # a minute that trades a fifth of what it does.
        history = []
        for i, day in enumerate(WEEK):
            minutes = [0, 15] if i == 0 else [0]
            history += bars(day, minutes, 3000)
        result = minute_volume_baseline("week", today, history, today="2026-09-21")
        assert result["per_minute"][9 * 60 + 45] == pytest.approx(3000)

    def test_todays_bars_never_leak_into_the_baseline(self, today, history):
        # Otherwise the reference is partly a copy of what it references.
        polluted = history + bars("2026-09-21", range(390), 1_000_000)
        result = minute_volume_baseline("week", today, polluted, today="2026-09-21")
        assert "2026-09-21" not in result["dates"]
        assert result["mean_per_minute"] < 10_000

    def test_measures_back_from_the_chart_not_from_now(self, today, history):
        # A chart of an earlier session compares against the days before it.
        result = minute_volume_baseline("week", today, history, today="2026-09-16")
        assert result["dates"] == ["2026-09-14", "2026-09-15"]

    def test_no_prior_sessions_means_no_baseline(self, today, history):
        assert minute_volume_baseline("week", today, history, today="2026-09-14") is None

    def test_infers_today_from_the_session_bars(self, today, history):
        assert (
            minute_volume_baseline("week", today, history)["dates"]
            == minute_volume_baseline("week", today, history, today="2026-09-21")["dates"]
        )


class TestBadBars:
    def test_a_bar_without_a_volume_is_skipped(self, today):
        history = bars("2026-09-18", range(3), 1000)
        history[1]["v"] = None
        result = minute_volume_baseline("yesterday", today, history, today="2026-09-21")
        assert set(result["per_minute"]) == {9 * 60 + 30, 9 * 60 + 32}

    def test_a_bar_without_a_timestamp_is_skipped(self, today):
        history = bars("2026-09-18", range(3), 1000)
        history[0]["t"] = None
        result = minute_volume_baseline("yesterday", today, history, today="2026-09-21")
        assert len(result["per_minute"]) == 2

    def test_all_bars_unusable_means_no_baseline(self, today):
        history = bars("2026-09-18", range(3), 1000)
        for bar in history:
            bar["v"] = "n/a"
        assert minute_volume_baseline("yesterday", today, history, today="2026-09-21") is None


class TestFetchingTheHistory:
    """`fetch_intraday_history_bars`, around yfinance's per-request limits."""

    @pytest.fixture(autouse=True)
    def clear_cache(self):
        historical._intraday_history_cache.clear()
        yield
        historical._intraday_history_cache.clear()

    def frame(self, start: str, rows: int) -> pd.DataFrame:
        index = pd.date_range(start + " 09:30", periods=rows, freq="1min", tz="America/New_York")
        return pd.DataFrame(
            {"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.5, "Volume": 1000.0},
            index=index,
        )

    def test_a_long_window_is_split_into_requests_yfinance_accepts(self, monkeypatch):
        windows: list[tuple[str, str]] = []

        def fake(symbol, start, end, **kwargs):
            windows.append((start, end))
            return self.frame(start, 2)

        monkeypatch.setattr(historical.yf, "download", fake)
        historical.fetch_intraday_history_bars("AAPL", 12)
        assert len(windows) > 1
        for start, end in windows:
            span = (date.fromisoformat(end) - date.fromisoformat(start)).days
            assert span <= 8  # yfinance refuses more than 8 days of 1m bars

    def test_the_whole_window_is_covered(self, monkeypatch):
        windows: list[tuple[str, str]] = []
        monkeypatch.setattr(
            historical.yf, "download",
            lambda symbol, start, end, **kw: windows.append((start, end)) or self.frame(start, 1),
        )
        historical.fetch_intraday_history_bars("AAPL", 12)
        assert windows == sorted(windows)
        for (_, end), (start, _) in zip(windows, windows[1:]):
            assert start == end  # no gap between chunks

    def test_bars_come_back_oldest_first_without_duplicates(self, monkeypatch):
        # Both chunks report the same day, as a seam can.
        monkeypatch.setattr(
            historical.yf, "download",
            lambda symbol, start, end, **kw: self.frame("2026-09-14", 3),
        )
        bars = historical.fetch_intraday_history_bars("AAPL", 12)
        stamps = [bar["t"] for bar in bars]
        assert stamps == sorted(stamps)
        assert len(stamps) == len(set(stamps)) == 3

    def test_one_failed_chunk_does_not_sink_the_read(self, monkeypatch):
        calls = {"n": 0}

        def flaky(symbol, start, end, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("yahoo said no")
            return self.frame(start, 2)

        monkeypatch.setattr(historical.yf, "download", flaky)
        assert historical.fetch_intraday_history_bars("AAPL", 12)

    def test_a_total_failure_returns_nothing_rather_than_raising(self, monkeypatch):
        monkeypatch.setattr(
            historical.yf, "download",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("down")),
        )
        assert historical.fetch_intraday_history_bars("AAPL", 12) == []

    def test_a_failure_is_not_retried_on_every_call(self, monkeypatch):
        # The caller is a chart fragment rerunning every 30 seconds.
        calls = {"n": 0}

        def failing(*a, **kw):
            calls["n"] += 1
            raise RuntimeError("down")

        monkeypatch.setattr(historical.yf, "download", failing)
        historical.fetch_intraday_history_bars("AAPL", 12)
        first = calls["n"]
        historical.fetch_intraday_history_bars("AAPL", 12)
        assert calls["n"] == first

    def test_a_good_read_is_cached(self, monkeypatch):
        calls = {"n": 0}

        def counted(symbol, start, end, **kwargs):
            calls["n"] += 1
            return self.frame(start, 2)

        monkeypatch.setattr(historical.yf, "download", counted)
        historical.fetch_intraday_history_bars("AAPL", 12)
        first = calls["n"]
        historical.fetch_intraday_history_bars("AAPL", 12)
        assert calls["n"] == first

    def test_bars_carry_the_shape_the_rest_of_the_app_uses(self, monkeypatch):
        monkeypatch.setattr(
            historical.yf, "download",
            lambda symbol, start, end, **kw: self.frame("2026-09-14", 1),
        )
        (bar,) = historical.fetch_intraday_history_bars("AAPL", 7)
        assert set(bar) == {"t", "o", "h", "l", "c", "v"}
        assert bar["t"] == "2026-09-14T13:30:00Z"  # 09:30 ET, in UTC


def band_level(samples: "list[float]") -> float:
    return statistics.fmean(samples) + statistics.stdev(samples)


class TestVolumeBand:
    """`volume_band`: mean + 1 sigma per clock bucket over last week."""

    NOON = 150  # minutes past the open: 12:00 ET
    TODAY = "2026-09-21"

    def test_the_default_window_is_the_band(self):
        assert DEFAULT_VOLUME_BASELINE == "week_band"
        assert lookback_days("week_band") == lookback_days("week")
        # The band is drawn by `volume_band`, not the mean/shape builder.
        assert minute_volume_baseline("week_band", [], []) is None

    def test_mean_plus_sigma_of_the_buckets_five_minutes_either_side(self):
        # Session i trades 1000 * (i + 1) every minute: the noon bucket pools
        # 11 minutes x 5 sessions of those.
        history = [
            b for i, day in enumerate(WEEK) for b in bars(day, range(390), 1000.0 * (i + 1))
        ]
        result = volume_band(history, self.TODAY)
        expected = band_level([1000.0 * (i + 1) for i in range(5) for _ in range(11)])
        assert result["per_minute"][570 + self.NOON] == pytest.approx(expected)
        assert result["dates"] == WEEK
        assert result["sessions"] == 5

    def test_today_and_older_sessions_are_left_out(self):
        older = bars("2026-09-11", range(390), 1e9)
        week = [b for day in WEEK for b in bars(day, range(390), 1000.0)]
        todays = bars(self.TODAY, range(390), 1e9)
        result = volume_band(older + week + todays, self.TODAY)
        assert result["per_minute"][570 + self.NOON] == pytest.approx(1000.0)

    def test_the_auction_minutes_are_not_pooled(self):
        # A huge 09:30 print; 09:31-09:40 quiet, varying by session.
        history = []
        for i, day in enumerate(WEEK):
            history += bars(day, [0], 500_000.0 + 1000 * i)
            history += bars(day, range(1, 20), 1000.0 + 10 * i)
        per_minute = volume_band(history, self.TODAY)["per_minute"]
        assert per_minute[570] == pytest.approx(band_level([500_000.0 + 1000 * i for i in range(5)]))
        # 09:31 pools 09:31-09:36 (09:26-09:30 are other stretches of the day).
        assert per_minute[571] == pytest.approx(
            band_level([1000.0 + 10 * i for i in range(5) for _ in range(6)])
        )

    def test_the_pre_market_is_not_pooled_with_the_session(self):
        history = []
        for day in WEEK:
            history += bars(day, range(-10, 0), 100.0)
            history += bars(day, range(1, 20), 50_000.0)
        per_minute = volume_band(history, self.TODAY)["per_minute"]
        assert per_minute[569] == pytest.approx(100.0)

    def test_coarser_bars_sum_each_sessions_bucket_first(self):
        # Five-minute buckets of five 1,000-share minutes: 5,000 a bucket.
        history = [b for day in WEEK for b in bars(day, range(390), 1000.0)]
        per_minute = volume_band(history, self.TODAY, span=5)["per_minute"]
        assert per_minute[570 + self.NOON] == pytest.approx(5000.0)
        assert all(m % 5 == 0 for m in per_minute)

    def test_nothing_to_measure(self):
        assert volume_band([], self.TODAY) is None
        assert volume_band(bars(self.TODAY, range(10), 100.0), self.TODAY) is None
