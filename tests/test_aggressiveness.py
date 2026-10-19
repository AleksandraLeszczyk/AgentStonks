"""Transaction aggressiveness: the per-trade side against the prevailing quote,
the per-minute sums and per-bar ratio, the SIP/IEX tiers, and the background
fetch and its store."""
import json
from datetime import datetime, timezone

import pandas as pd
import pytest

from agent_stonks import aggressiveness as ag


def q(t, bp, ap):
    return {"t": t, "bp": bp, "ap": ap}


def tr(t, p, s=100, c=("@",)):
    return {"t": t, "p": p, "s": s, "c": list(c)}


T0 = "2026-10-19T14:00:00.000000000Z"


class TestClassify:
    def test_ask_bid_and_midpoint(self):
        quotes = [q(T0, 100.00, 100.02)]
        trades = [
            tr("2026-10-19T14:00:01Z", 100.02),   # at the ask
            tr("2026-10-19T14:00:02Z", 100.015),  # inside, nearer the ask
            tr("2026-10-19T14:00:03Z", 100.00),   # at the bid
            tr("2026-10-19T14:00:04Z", 100.01),   # the midpoint
            tr("2026-10-19T14:00:05Z", 100.05),   # through the offer
        ]
        assert list(ag.classify(trades, quotes)["side"]) == [1, 1, -1, 0, 1]

    def test_the_quote_at_the_trades_own_instant_is_not_the_prevailing_one(self):
        """A buy that takes out the 100.02 offer prints together with the
        quote it leaves behind -- bid 100.02 / ask 100.03 -- and against that
        quote it would read as a sale."""
        quotes = [q(T0, 100.00, 100.02), q("2026-10-19T14:00:01.5Z", 100.02, 100.03)]
        trades = [tr("2026-10-19T14:00:01.5Z", 100.02)]
        assert list(ag.classify(trades, quotes)["side"]) == [1]

    def test_value_is_price_times_size(self):
        out = ag.classify([tr("2026-10-19T14:00:01Z", 100.02, s=250)], [q(T0, 100.0, 100.02)])
        assert out["value"].iloc[0] == pytest.approx(25005.0)

    def test_non_book_prints_are_left_out(self):
        quotes = [q(T0, 100.0, 100.02)]
        trades = [
            tr("2026-10-19T14:00:01Z", 100.02, c=("@", "O")),       # opening cross
            tr("2026-10-19T14:00:02Z", 100.02, c=("@", "Z")),       # out of sequence
            tr("2026-10-19T14:00:03Z", 100.02, c=("@", "4")),       # derivatively priced
            tr("2026-10-19T14:00:04Z", 100.02, c=("@", "F", "I")),  # sweep, odd lot: kept
        ]
        out = ag.classify(trades, quotes)
        assert list(out["t"]) == [pd.Timestamp("2026-10-19T14:00:04Z")]

    def test_no_quote_crossed_or_one_sided_quote_is_left_out(self):
        quotes = [
            q("2026-10-19T14:00:10Z", 100.03, 100.02),  # crossed
            q("2026-10-19T14:00:20Z", 0.0, 100.02),     # one-sided
            q("2026-10-19T14:01:00Z", 100.0, 100.02),
        ]
        trades = [
            tr("2026-10-19T14:00:05Z", 100.02),  # before any quote
            tr("2026-10-19T14:00:15Z", 100.02),
            tr("2026-10-19T14:00:25Z", 100.02),
            tr("2026-10-19T14:02:00Z", 100.02),  # the quote is over QUOTE_LEAD_SEC old
            tr("2026-10-19T14:01:01Z", 100.02),
        ]
        out = ag.classify(trades, quotes)
        assert list(out["t"]) == [pd.Timestamp("2026-10-19T14:01:01Z")]

    def test_nothing_in(self):
        assert ag.classify([], [q(T0, 1, 2)]).empty
        assert ag.classify([tr(T0, 1)], []).empty


class TestSums:
    def test_formula_per_minute(self):
        quotes = [q(T0, 100.0, 100.02), q("2026-10-19T14:01:00Z", 101.0, 101.02)]
        trades = [
            tr("2026-10-19T14:00:01Z", 100.02, s=300),  # ask $30,006
            tr("2026-10-19T14:00:02Z", 100.00, s=100),  # bid $10,000
            tr("2026-10-19T14:00:03Z", 100.01, s=100),  # mid $10,001
            tr("2026-10-19T14:01:01Z", 101.00, s=50),   # bid $5,050
        ]
        sums = ag.minute_sums(trades, quotes)
        first = sums["2026-10-19T14:00:00+00:00"]
        assert first == pytest.approx([30006.0, 10000.0, 50007.0, 3])
        # (ask - bid) / total, the midpoint print in the total only.
        assert ag.bar_value([first]) == pytest.approx(20006.0 / 50007.0)
        assert ag.bar_value([sums["2026-10-19T14:01:00+00:00"]]) == -1.0
        # A bar of both minutes is the sum of their sums, not a mean of ratios.
        both = ag.bar_value(list(sums.values()))
        assert both == pytest.approx((30006.0 - 15050.0) / 55057.0)

    def test_nothing_traded_has_no_value(self):
        assert ag.bar_value([]) is None
        assert ag.bar_value([[0.0, 0.0, 0.0, 0]]) is None


def ts(text):
    return pd.Timestamp(text)


class TestPlan:
    REAL_NOW = datetime(2026, 10, 19, 18, 0, tzinfo=timezone.utc)

    def test_sip_once_old_enough_iex_before(self):
        minutes = [ts("2026-10-19T17:40:00Z"), ts("2026-10-19T17:43:00Z"),
                   ts("2026-10-19T17:44:00Z"), ts("2026-10-19T17:59:00Z")]
        iex, sip = ag._plan(minutes, {}, self.REAL_NOW, self.REAL_NOW)
        # 17:43 ends 17:44, exactly SIP_DELAY_MIN (16) back.
        assert sip == [ts("2026-10-19T17:43:00Z"), ts("2026-10-19T17:40:00Z")]
        assert iex == [ts("2026-10-19T17:44:00Z"), ts("2026-10-19T17:59:00Z")]

    def test_kept_minutes_are_not_fetched_again(self):
        kept = {
            "2026-10-19T17:40:00+00:00": [1, 0, 1, 1, "sip"],
            "2026-10-19T17:41:00+00:00": [1, 0, 1, 1, "iex-final"],   # SIP due now
            "2026-10-19T17:50:00+00:00": [1, 0, 1, 1, "iex-final"],
            "2026-10-19T17:58:00+00:00": [1, 0, 1, 1, "iex"],         # fetched while forming
        }
        minutes = [ts(t) for t in kept]
        iex, sip = ag._plan(minutes, kept, self.REAL_NOW, self.REAL_NOW)
        assert sip == [ts("2026-10-19T17:41:00Z")]
        assert iex == [ts("2026-10-19T17:58:00Z")]

    def test_a_replays_forming_minute_is_not_settled(self):
        """Long on the delayed SIP tape by the wall clock, but not over by the
        replay's: fetched as far as it has got, and again."""
        replay_now = datetime(2026, 10, 16, 15, 0, 20, tzinfo=timezone.utc)
        minutes = [ts("2026-10-16T14:59:00Z"), ts("2026-10-16T15:00:00Z")]
        iex, sip = ag._plan(minutes, {}, replay_now, self.REAL_NOW)
        assert sip == [ts("2026-10-16T14:59:00Z")]
        assert iex == [ts("2026-10-16T15:00:00Z")]

    def test_not_started_yet(self):
        iex, sip = ag._plan([ts("2026-10-19T18:00:00Z")], {}, self.REAL_NOW, self.REAL_NOW)
        assert iex == sip == []

    def test_chunks_are_runs_newest_first(self):
        minutes = sorted(
            [ts("2026-10-19T14:00:00Z") + pd.Timedelta(minutes=i) for i in range(25)]
            + [ts("2026-10-19T13:40:00Z")],
            reverse=True,
        )
        chunks = ag._chunks(minutes)
        assert [len(c) for c in chunks] == [10, 10, 5, 1]
        assert chunks[0][0] == ts("2026-10-19T14:24:00Z")


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A clean store on a scratch disk, no pauses, and the worker run inline."""
    monkeypatch.setattr(ag, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(ag, "REQUEST_GAP_SEC", 0.0)
    for name in ("_kept", "_wanted", "_iex_at", "_retry_at", "_errors"):
        monkeypatch.setattr(ag, name, {})
    monkeypatch.setattr(ag, "_running", set())

    class Inline:
        def __init__(self, target, args=(), **_):
            self.target, self.args = target, args

        def start(self):
            self.target(*self.args)

    monkeypatch.setattr(ag.threading, "Thread", Inline)
    return tmp_path


def bars_from(start, n, step_min=1):
    t0 = ts(start)
    return [{"t": (t0 + pd.Timedelta(minutes=i * step_min)).isoformat(),
             "o": 1, "h": 1, "l": 1, "c": 1, "v": 1} for i in range(n)]


class FakeTape:
    """Alpaca REST stand-in: per minute one buy at the ask of $ (minute + 1) x
    100 and one sell at the bid of $100, every quote 100.00 / 100.02."""

    def __init__(self):
        self.calls = []

    def __call__(self, kind, symbol, start, end, key, secret, feed):
        self.calls.append((kind, feed, start, end))
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        out = []
        m = start.floor("min")
        while m < end:
            if kind == "quotes":
                out.append(q(m.isoformat(), 100.0, 100.02))
            elif m >= start:
                buy = (m.minute + 1) * 100 / 100.02
                out.append(tr((m + pd.Timedelta(seconds=10)).isoformat(), 100.02, s=buy))
                out.append(tr((m + pd.Timedelta(seconds=20)).isoformat(), 100.00, s=1))
            m += pd.Timedelta(minutes=1)
        return out


class TestSeries:
    NOW = datetime(2026, 10, 19, 14, 1, 30, tzinfo=timezone.utc)   # 10:01:30 ET

    def test_fetches_settles_and_keeps(self, store, monkeypatch):
        tape = FakeTape()
        monkeypatch.setattr(ag, "fetch_ticks", tape)
        monkeypatch.setattr(ag.clock, "now", lambda: self.NOW)
        bars = bars_from("2026-10-19T13:25:00Z", 36)   # 09:25 .. 10:00 ET
        first = ag.series("AAPL", "1Min", bars, "k", "s", now=self.NOW)
        assert first["t"] == []                        # fetched in the background
        out = ag.series("AAPL", "1Min", bars, "k", "s", now=self.NOW)
        # The regular-session bars only, 09:30 .. 10:00.
        assert out["t"][0] == "2026-10-19T13:30:00+00:00"
        assert len(out["t"]) == 31
        # 09:31: a $3,200 buy against a $100 sale.
        assert out["value"][1] == pytest.approx((3200.0 - 100.0) / 3300.0)
        # Older than SIP_DELAY_MIN against the real clock, which is long past.
        assert all(out["settled"])
        assert {feed for _, feed, _, _ in tape.calls} == {"sip"}
        kept = json.loads((store / "AAPL_2026-10-19.json").read_text())
        assert len(kept) == 31

        # A reload draws them from disk, without a fetch.
        monkeypatch.setattr(ag, "_kept", {})
        tape.calls.clear()
        assert ag.series("AAPL", "1Min", bars, "k", "s", now=self.NOW)["value"] == out["value"]
        assert tape.calls == []

    def test_recent_minutes_are_iex_until_sip(self, store, monkeypatch):
        tape = FakeTape()
        monkeypatch.setattr(ag, "fetch_ticks", tape)
        real = datetime(2026, 10, 19, 14, 10, 30, tzinfo=timezone.utc)
        monkeypatch.setattr(ag, "_wall_now", lambda: real)
        monkeypatch.setattr(ag.clock, "now", lambda: real)
        bars = bars_from("2026-10-19T13:30:00Z", 41)   # 09:30 .. 10:10 ET
        ag.series("AAPL", "1Min", bars, "k", "s", now=real)
        out = ag.series("AAPL", "1Min", bars, "k", "s", now=real)
        settled = dict(zip(out["t"], out["settled"]))
        assert settled["2026-10-19T13:53:00+00:00"] is True    # ends 13:54, 16 min back
        assert settled["2026-10-19T13:54:00+00:00"] is False
        assert settled["2026-10-19T14:10:00+00:00"] is False   # the forming minute
        # Only the settled minutes on disk.
        kept = json.loads((store / "AAPL_2026-10-19.json").read_text())
        assert max(kept) == "2026-10-19T13:53:00+00:00"

    def test_five_minute_bars_sum_their_minutes(self, store, monkeypatch):
        monkeypatch.setattr(ag, "fetch_ticks", FakeTape())
        monkeypatch.setattr(ag.clock, "now", lambda: self.NOW)
        bars = bars_from("2026-10-19T13:30:00Z", 6, step_min=5)
        ag.series("AAPL", "5Min", bars, "k", "s", now=self.NOW)
        out = ag.series("AAPL", "5Min", bars, "k", "s", now=self.NOW)
        # 09:30-09:34: buys of $3,100..$3,500, five $100 sales.
        assert out["value"][0] == pytest.approx((16500.0 - 500.0) / 17000.0)
        assert out["traded"][0] == pytest.approx(17000.0)

    def test_a_failed_fetch_says_so_and_waits(self, store, monkeypatch):
        def broken(*_):
            raise RuntimeError("429 Too Many Requests")
        monkeypatch.setattr(ag, "fetch_ticks", broken)
        monkeypatch.setattr(ag.clock, "now", lambda: self.NOW)
        bars = bars_from("2026-10-19T13:30:00Z", 10)
        ag.series("AAPL", "1Min", bars, "k", "s", now=self.NOW)
        out = ag.series("AAPL", "1Min", bars, "k", "s", now=self.NOW)
        assert out["t"] == [] and "429" in out["note"]
        assert "AAPL" in ag._retry_at and "AAPL" not in ag._running

    def test_notes(self, store, monkeypatch):
        monkeypatch.setattr(ag, "fetch_ticks", FakeTape())
        pre = ag.series("AAPL", "1Min", bars_from("2026-10-19T12:00:00Z", 30), "k", "s",
                        now=self.NOW)
        assert "09:30" in pre["note"]
        no_key = ag.series("AAPL", "1Min", bars_from("2026-10-19T13:30:00Z", 5), "", "",
                           now=self.NOW)
        assert "Alpaca key" in no_key["note"]
        assert ag.series("AAPL", "1Min", [], "k", "s")["note"] == "No bars yet"
