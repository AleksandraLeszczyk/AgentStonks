"""The live chart's net gamma panel: a bar's value is taken once, at its close."""
import json

import pandas as pd
import pytest

from agent_stonks import gamma_history
from agent_stonks.options import net_gamma_exposure


def chain(calls_oi=(10.0, 50.0, 400.0), iv=0.3):
    return {
        "strikes": [95.0, 100.0, 105.0],
        "calls_oi": list(calls_oi),
        "puts_oi": [300.0, 40.0, 5.0],
        "calls_iv": [iv] * 3,
        "puts_iv": [iv] * 3,
        "t_years": 10 / 365,
    }


# Another fetch a minute later: new open interest and IVs.
CHAIN_A = chain()
CHAIN_B = chain(calls_oi=(900.0, 50.0, 10.0), iv=0.45)

# 13:40 ET on 2026-10-01 (EDT, UTC-4).
T0 = pd.Timestamp("2026-10-01T17:40:00Z")


def bars(closes, start=T0, minutes=1):
    return [
        {"t": (start + pd.Timedelta(minutes=minutes * i)).isoformat(), "c": c}
        for i, c in enumerate(closes)
    ]


def at(minutes, seconds=0):
    """The wall clock `minutes` after 13:40 ET."""
    return (T0 + pd.Timedelta(minutes=minutes, seconds=seconds)).to_pydatetime()


def value(series, t):
    return dict(zip(series["t"], series["value"]))[t]


def priced(c, data):
    return float(net_gamma_exposure(data, [c])[0])


class TestTakenOnce:
    def test_a_closed_bar_keeps_its_value_when_the_chain_changes(self):
        # 13:41:30: the 13:40 bar has closed, 13:41 is forming.
        history = bars([100.0, 101.0])
        first = gamma_history.series("AAPL", "1Min", history, CHAIN_A, now=at(1, 30))
        t1340 = history[0]["t"]
        assert value(first, t1340) == pytest.approx(priced(100.0, CHAIN_A))

        # The chain is refetched and more minutes print: 13:40 does not move.
        later = bars([100.0, 101.0, 102.0, 99.0])
        again = gamma_history.series("AAPL", "1Min", later, CHAIN_B, now=at(3, 30))
        assert value(again, t1340) == pytest.approx(priced(100.0, CHAIN_A))
        assert priced(100.0, CHAIN_B) != pytest.approx(priced(100.0, CHAIN_A))

    def test_the_forming_bar_follows_its_close_and_the_chain(self):
        history = bars([100.0, 101.0])
        t1341 = history[1]["t"]
        first = gamma_history.series("AAPL", "1Min", history, CHAIN_A, now=at(1, 10))
        assert value(first, t1341) == pytest.approx(priced(101.0, CHAIN_A))

        history[1]["c"] = 101.5
        second = gamma_history.series("AAPL", "1Min", history, CHAIN_B, now=at(1, 40))
        assert value(second, t1341) == pytest.approx(priced(101.5, CHAIN_B))

    def test_the_forming_bar_is_kept_once_a_later_bar_prints(self):
        history = bars([100.0, 101.0])
        t1341 = history[1]["t"]
        gamma_history.series("AAPL", "1Min", history, CHAIN_A, now=at(1, 50))
        history.append({"t": (T0 + pd.Timedelta(minutes=2)).isoformat(), "c": 102.0})
        gamma_history.series("AAPL", "1Min", history, CHAIN_A, now=at(2, 0))
        later = gamma_history.series("AAPL", "1Min", history, CHAIN_B, now=at(2, 30))
        assert value(later, t1341) == pytest.approx(priced(101.0, CHAIN_A))

    def test_the_last_bar_is_kept_once_its_period_has_ended(self):
        """After the close no later bar comes; the last one still stops moving."""
        history = bars([100.0, 101.0])
        t1341 = history[1]["t"]
        gamma_history.series("AAPL", "1Min", history, CHAIN_A, now=at(2, 5))
        later = gamma_history.series("AAPL", "1Min", history, CHAIN_B, now=at(30))
        assert value(later, t1341) == pytest.approx(priced(101.0, CHAIN_A))

    def test_a_revised_close_does_not_move_a_kept_bar(self):
        history = bars([100.0, 101.0])
        gamma_history.series("AAPL", "1Min", history, CHAIN_A, now=at(1, 30))
        history[0]["c"] = 100.7  # the consolidated tape replaces the provisional bar
        again = gamma_history.series("AAPL", "1Min", history, CHAIN_A, now=at(1, 40))
        assert value(again, history[0]["t"]) == pytest.approx(priced(100.0, CHAIN_A))

    def test_a_five_minute_bar_is_forming_for_five_minutes(self):
        history = bars([100.0], minutes=5)
        gamma_history.series("AAPL", "5Min", history, CHAIN_A, now=at(4))
        history[0]["c"] = 103.0
        forming = gamma_history.series("AAPL", "5Min", history, CHAIN_A, now=at(4, 50))
        assert forming["value"] == pytest.approx([priced(103.0, CHAIN_A)])
        gamma_history.series("AAPL", "5Min", history, CHAIN_A, now=at(5, 1))
        kept = gamma_history.series("AAPL", "5Min", history, CHAIN_B, now=at(9))
        assert kept["value"] == pytest.approx([priced(103.0, CHAIN_A)])

    def test_timestamp_and_string_bar_times_are_the_same_bar(self):
        history = bars([100.0, 101.0])
        gamma_history.series("AAPL", "1Min", history, CHAIN_A, now=at(1, 30))
        history[0]["t"] = pd.Timestamp(history[0]["t"]).tz_convert("America/New_York")
        again = gamma_history.series("AAPL", "1Min", history, CHAIN_B, now=at(1, 40))
        assert again["value"][0] == pytest.approx(priced(100.0, CHAIN_A))


class TestBeforeTheChain:
    def test_no_chain_yet_says_so(self):
        series = gamma_history.series("AAPL", "1Min", bars([100.0, 101.0]), None, now=at(1, 30))
        assert series["t"] == [] and series["value"] == []
        assert "options chain" in series["note"]

    def test_a_chain_without_ivs_waits_for_the_next_refresh(self):
        old = {k: v for k, v in CHAIN_A.items() if k not in ("calls_iv", "puts_iv")}
        series = gamma_history.series("AAPL", "1Min", bars([100.0, 101.0]), old, now=at(1, 30))
        assert series["value"] == [] and "refresh" in series["note"]

    def test_bars_closed_before_the_first_chain_are_priced_with_it_and_kept(self):
        history = bars([100.0, 101.0, 102.0])
        gamma_history.series("AAPL", "1Min", history, None, now=at(2, 30))
        gamma_history.series("AAPL", "1Min", history, CHAIN_A, now=at(2, 40))
        later = gamma_history.series("AAPL", "1Min", history, CHAIN_B, now=at(2, 50))
        assert later["value"][:2] == pytest.approx(
            [priced(100.0, CHAIN_A), priced(101.0, CHAIN_A)]
        )
        assert later["value"][2] == pytest.approx(priced(102.0, CHAIN_B))  # forming

    def test_kept_bars_are_drawn_while_the_chain_is_missing(self):
        history = bars([100.0, 101.0])
        gamma_history.series("AAPL", "1Min", history, CHAIN_A, now=at(1, 30))
        series = gamma_history.series("AAPL", "1Min", history, None, now=at(1, 40))
        assert series["t"] == [history[0]["t"]]
        assert series["note"] == ""


class TestKeptAcrossARestart:
    def test_values_come_back_from_disk(self, monkeypatch):
        history = bars([100.0, 101.0])
        gamma_history.series("AAPL", "1Min", history, CHAIN_A, now=at(1, 30))
        monkeypatch.setattr(gamma_history, "_kept", {})  # a new process
        again = gamma_history.series("AAPL", "1Min", history, CHAIN_B, now=at(1, 40))
        assert again["value"][0] == pytest.approx(priced(100.0, CHAIN_A))

    def test_one_file_per_symbol_timeframe_and_day_without_the_forming_bar(self):
        history = bars([100.0, 101.0])
        gamma_history.series("AAPL", "1Min", history, CHAIN_A, now=at(1, 30))
        path = gamma_history.CACHE_DIR / "AAPL_1Min_2026-10-01.json"
        saved = json.loads(path.read_text())
        assert list(saved) == [pd.Timestamp(history[0]["t"]).isoformat()]

    def test_an_unreadable_file_is_ignored(self):
        gamma_history.CACHE_DIR.mkdir(parents=True)
        (gamma_history.CACHE_DIR / "AAPL_1Min_2026-10-01.json").write_text("{not json")
        series = gamma_history.series("AAPL", "1Min", bars([100.0, 101.0]), CHAIN_A, now=at(1, 30))
        assert series["value"][0] == pytest.approx(priced(100.0, CHAIN_A))


class TestOnlyTheLatestDay:
    def test_yesterdays_bars_are_left_out(self):
        yesterday = bars([90.0, 91.0], start=T0 - pd.Timedelta(days=1))
        today = bars([100.0, 101.0])
        series = gamma_history.series("AAPL", "1Min", yesterday + today, CHAIN_A, now=at(1, 30))
        assert series["t"] == [b["t"] for b in today]

    def test_symbols_are_kept_apart(self):
        history = bars([100.0, 101.0])
        gamma_history.series("AAPL", "1Min", history, CHAIN_A, now=at(1, 30))
        other = gamma_history.series("MSFT", "1Min", history, CHAIN_B, now=at(1, 40))
        assert other["value"][0] == pytest.approx(priced(100.0, CHAIN_B))
