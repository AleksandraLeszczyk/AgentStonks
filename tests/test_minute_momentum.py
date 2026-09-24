"""`abs_mean_minute_momentum`: last week's mean absolute one-minute move."""

from datetime import datetime, timedelta

import pytest

from agent_stonks import historical, minute_momentum
from agent_stonks.minute_momentum import compute, load_or_compute, refresh

OPEN_UTC = "T13:30:00+00:00"  # 09:30 ET


def bars(date: str, closes: list[float]) -> list[dict]:
    """Consecutive minute bars from the 09:30 ET open, one per close."""
    base = datetime.fromisoformat(date + OPEN_UTC)
    return [
        {
            "t": (base + timedelta(minutes=m)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "o": c, "h": c, "l": c, "c": c, "v": 100.0,
        }
        for m, c in enumerate(closes)
    ]


WEEK = ["2026-09-14", "2026-09-15", "2026-09-16", "2026-09-17", "2026-09-18"]


class TestCompute:
    def test_mean_of_absolute_close_to_close_changes(self):
        # Changes +0.10, -0.30, +0.20 -> |.| mean 0.20.
        result = compute(bars(WEEK[0], [100.0, 100.1, 99.8, 100.0]), "2026-09-21")
        assert result["abs_mean_minute_momentum"] == pytest.approx(0.20)
        assert result["changes"] == 3
        assert result["dates"] == [WEEK[0]]

    def test_the_overnight_gap_is_not_a_minute(self):
        history = bars(WEEK[0], [100.0, 100.1]) + bars(WEEK[1], [110.0, 110.1])
        result = compute(history, "2026-09-21")
        assert result["abs_mean_minute_momentum"] == pytest.approx(0.10)
        assert result["changes"] == 2

    def test_only_the_last_five_sessions_before_today(self):
        older = bars("2026-09-11", [100.0, 105.0])  # a sixth, older session
        week = [b for d in WEEK for b in bars(d, [100.0, 100.5])]
        today = bars("2026-09-21", [100.0, 150.0])  # today's own bars
        result = compute(older + week + today, "2026-09-21")
        assert result["abs_mean_minute_momentum"] == pytest.approx(0.5)
        assert result["dates"] == WEEK

    def test_unordered_bars_are_sorted_within_a_session(self):
        result = compute(list(reversed(bars(WEEK[0], [100.0, 101.0, 100.0]))), "2026-09-21")
        assert result["abs_mean_minute_momentum"] == pytest.approx(1.0)

    def test_nothing_to_measure(self):
        assert compute([], "2026-09-21") is None
        assert compute(bars(WEEK[0], [100.0]), "2026-09-21") is None


class TestOncePerDay:
    @pytest.fixture(autouse=True)
    def cache_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(minute_momentum, "CACHE_DIR", tmp_path)
        self.fetches = []

        def fake(symbol, days, **kwargs):
            self.fetches.append((symbol, days))
            return bars(WEEK[-1], [100.0, 100.25])

        monkeypatch.setattr(historical, "fetch_intraday_history_bars", fake)
        return tmp_path

    def test_second_call_the_same_day_reads_the_stored_value(self):
        first = load_or_compute("AAPL", today="2026-09-21")
        second = load_or_compute("AAPL", today="2026-09-21")
        assert first["abs_mean_minute_momentum"] == pytest.approx(0.25)
        assert second == first
        assert len(self.fetches) == 1

    def test_a_new_day_computes_again(self):
        load_or_compute("AAPL", today="2026-09-21")
        load_or_compute("AAPL", today="2026-09-22")
        assert len(self.fetches) == 2

    def test_a_failed_fetch_is_not_stored(self, monkeypatch):
        monkeypatch.setattr(historical, "fetch_intraday_history_bars", lambda *a, **k: [])
        assert load_or_compute("AAPL", today="2026-09-21") is None
        assert not list(minute_momentum.CACHE_DIR.iterdir())

    def test_refresh_sets_the_state_field(self):
        class State:
            symbol = "AAPL"
            abs_mean_minute_momentum = None

        state = State()
        assert refresh(state) == pytest.approx(0.25)
        assert state.abs_mean_minute_momentum == pytest.approx(0.25)
