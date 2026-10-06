import threading
import time
from datetime import date

import numpy as np
import pytest

from agent_stonks import prior_profile


def _bar(t: str, low: float, high: float, volume: float) -> dict:
    return {"t": t, "o": low, "h": high, "l": low, "c": high, "v": volume}


def _session(levels: "list[tuple[float, int, float]]") -> "list[dict]":
    """Minute bars a cent wide: `count` minutes at each (price, count, volume)."""
    bars = []
    for price, count, volume in levels:
        bars += [_bar("2026-10-05T14:00:00Z", price, price + 0.01, volume)] * count
    return bars


@pytest.fixture(autouse=True)
def _fresh_cache(monkeypatch):
    monkeypatch.setattr(prior_profile, "_kept", {})
    monkeypatch.setattr(prior_profile, "_running", set())
    monkeypatch.setattr(prior_profile, "_tried", {})


class TestVolumeAtPrice:
    def test_a_bars_volume_is_spread_over_its_range(self):
        bars = [_bar("t", 100.0, 101.0, 1000.0), _bar("t", 100.0, 100.5, 500.0)]
        centers, volume = prior_profile.volume_at_price(bars, n_bins=4)
        assert centers.tolist() == pytest.approx([100.125, 100.375, 100.625, 100.875])
        # 250 a slice from the first bar, 250 more in the two the second covers.
        assert volume.tolist() == pytest.approx([500.0, 500.0, 250.0, 250.0])

    def test_a_flat_bar_lands_in_one_slice(self):
        bars = [_bar("t", 100.0, 101.0, 0.0), _bar("t", 100.9, 100.9, 300.0)]
        _, volume = prior_profile.volume_at_price(bars, n_bins=4)
        assert volume.tolist() == [0.0, 0.0, 0.0, 300.0]

    def test_none_without_range_or_volume(self):
        assert prior_profile.volume_at_price([]) is None
        assert prior_profile.volume_at_price([_bar("t", 100.0, 100.0, 50.0)]) is None
        assert prior_profile.volume_at_price([_bar("t", 100.0, 101.0, 0.0)]) is None


class TestProfileLevels:
    def test_poc_and_a_separated_second_peak(self):
        # Heavy trade at 100, a second shelf at 103, little between.
        bars = _session([(100.0, 60, 5000.0), (101.5, 5, 200.0), (103.0, 30, 4000.0)])
        levels = prior_profile.profile_levels(bars)
        assert levels["poc"] == pytest.approx(100.0, abs=0.05)
        assert len(levels["peaks"]) == 1
        assert levels["peaks"][0] == pytest.approx(103.0, abs=0.05)

    def test_a_ripple_is_not_a_peak(self):
        # The bump at 102 stands well under 15% of the POC's height above its trough.
        bars = _session([(100.0, 60, 5000.0), (101.0, 10, 300.0), (102.0, 10, 400.0), (103.0, 10, 300.0)])
        levels = prior_profile.profile_levels(bars)
        assert levels["poc"] == pytest.approx(100.0, abs=0.05)
        assert levels["peaks"] == []

    def test_peaks_ascending_and_without_the_poc(self):
        bars = _session([(98.0, 30, 3000.0), (100.0, 60, 5000.0), (102.0, 30, 3000.0)])
        levels = prior_profile.profile_levels(bars)
        assert levels["poc"] == pytest.approx(100.0, abs=0.05)
        assert levels["peaks"] == sorted(levels["peaks"])
        assert [round(p) for p in levels["peaks"]] == [98, 102]

    def test_prominence_against_the_higher_trough(self):
        values = np.array([0.0, 10.0, 2.0, 6.0, 4.0, 5.0, 0.0])
        found = dict(prior_profile._prominences(values))
        assert found == {1: 10.0, 3: 4.0, 5: 1.0}

    def test_beyond_the_ends_is_zero(self):
        # Volume piled at the day's low and high: both are peaks.
        values = np.array([7.0, 1.0, 10.0, 1.0, 3.0])
        assert dict(prior_profile._prominences(values)) == {0: 6.0, 2: 10.0, 4: 2.0}


class TestSessions:
    def test_previous_session_from_daily_bars(self):
        daily = [{"t": "2026-10-01T04:00:00Z"}, {"t": "2026-10-02T04:00:00Z"}, {"t": "2026-10-05T04:00:00Z"}]
        # Monday's chart reads Friday; Tuesday's reads Monday, today's bar ignored.
        assert prior_profile.previous_session(date(2026, 10, 5), daily) == date(2026, 10, 2)
        assert prior_profile.previous_session(date(2026, 10, 6), daily) == date(2026, 10, 5)
        assert prior_profile.previous_session(date(2026, 10, 6), []) is None

    def test_chart_day_is_the_latest_bars_et_day(self):
        # 01:30 UTC on the 6th is 21:30 ET on the 5th.
        bars = [{"t": "2026-10-05T19:59:00Z"}, {"t": "2026-10-06T01:30:00Z"}]
        assert prior_profile.chart_day(bars) == date(2026, 10, 5)

    def test_fetch_reads_the_daily_bars_session(self, monkeypatch):
        asked = []

        def fetch(symbol, day):
            asked.append(day)
            return _session([(100.0, 60, 5000.0), (103.0, 30, 4000.0)])

        monkeypatch.setattr(prior_profile, "fetch_intraday_bars_for_date", fetch)
        result = prior_profile._fetch("AAPL", date(2026, 10, 6), [{"t": "2026-10-02T04:00:00Z"}])
        assert asked == ["2026-10-02"]
        assert result["date"] == "2026-10-02"
        assert result["poc"] == pytest.approx(100.0, abs=0.05)

    def test_fetch_without_daily_bars_skips_a_weekend_and_a_holiday(self, monkeypatch):
        asked = []

        def fetch(symbol, day):
            asked.append(day)
            return [] if day == "2026-10-02" else _session([(100.0, 60, 5000.0)])

        monkeypatch.setattr(prior_profile, "fetch_intraday_bars_for_date", fetch)
        result = prior_profile._fetch("AAPL", date(2026, 10, 5), [])
        assert asked == ["2026-10-02", "2026-10-01"]
        assert result["date"] == "2026-10-01"

    def test_fetch_raises_when_the_known_session_comes_back_empty(self, monkeypatch):
        monkeypatch.setattr(prior_profile, "fetch_intraday_bars_for_date", lambda symbol, day: [])
        with pytest.raises(LookupError):
            prior_profile._fetch("AAPL", date(2026, 10, 6), [{"t": "2026-10-05T04:00:00Z"}])


class TestLevels:
    DAILY = [{"t": "2026-10-05T04:00:00Z"}]

    def _wait(self, symbol, day):
        deadline = time.monotonic() + 2
        while (symbol, day.isoformat()) in prior_profile._running and time.monotonic() < deadline:
            time.sleep(0.01)

    def test_fetched_once_in_the_background_then_kept(self, monkeypatch):
        calls, release = [], threading.Event()

        def fetch(symbol, day):
            calls.append(day)
            release.wait(2)
            return _session([(100.0, 60, 5000.0), (103.0, 30, 4000.0)])

        monkeypatch.setattr(prior_profile, "fetch_intraday_bars_for_date", fetch)
        day = date(2026, 10, 6)
        assert prior_profile.levels("AAPL", day, self.DAILY) is None
        release.set()
        self._wait("AAPL", day)
        levels = prior_profile.levels("AAPL", day, self.DAILY)
        assert levels["date"] == "2026-10-05"
        assert levels["poc"] == pytest.approx(100.0, abs=0.05)
        assert prior_profile.levels("AAPL", day, self.DAILY) is levels
        assert calls == ["2026-10-05"]

    def test_a_failure_is_retried_only_after_the_interval(self, monkeypatch):
        calls = []

        def fetch(symbol, day):
            calls.append(day)
            raise RuntimeError("yfinance down")

        monkeypatch.setattr(prior_profile, "fetch_intraday_bars_for_date", fetch)
        day = date(2026, 10, 6)
        assert prior_profile.levels("AAPL", day, self.DAILY) is None
        self._wait("AAPL", day)
        assert prior_profile.levels("AAPL", day, self.DAILY) is None
        assert calls == ["2026-10-05"]
        prior_profile._tried[("AAPL", day.isoformat())] -= prior_profile.RETRY_SEC
        prior_profile.levels("AAPL", day, self.DAILY)
        self._wait("AAPL", day)
        assert calls == ["2026-10-05", "2026-10-05"]
