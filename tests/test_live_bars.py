"""The live buffer's tiered backfill: which source each bar comes from, and
which bars a later backfill may replace (`bar_history.fetch_live_bars`,
`stream_common.merge_live_bars`)."""
from datetime import datetime, timedelta, timezone

from agent_stonks import bar_history, finnhub_stream, stream, stream_common
from agent_stonks.state import AppState

# 14:00 ET on a Friday: regular session, so the settled window holds both
# regular-session and (much earlier) pre-market minutes.
NOW = datetime(2026, 9, 25, 18, 0, 30, tzinfo=timezone.utc)


def _bar(ts: str, v: float = 100.0, c: float = 1.0) -> dict:
    return {"t": ts, "o": c, "h": c, "l": c, "c": c, "v": v}


def _state():
    app = AppState()
    app.set_symbols(["AAPL"])
    return app.sym("AAPL")


def _key(ts: str) -> str:
    return bar_history.bar_key(ts)


class TestFetchLiveBars:
    def _patch(self, monkeypatch, base, yf, recent, calls=None, base_feed="sip_delayed"):
        calls = calls if calls is not None else {}

        def _history(symbol, timeframe, key, secret, feed, **kw):
            return base, bar_history.SOURCE_LABELS[base_feed], []

        def _yf(symbol, interval, start, end):
            calls["yf_end"] = end
            return yf

        def _recent(symbol, timeframe, start, end, key, secret, feed):
            calls["recent"] = (start, end, feed)
            return recent

        monkeypatch.setattr(bar_history, "fetch_history_bars", _history)
        monkeypatch.setattr(bar_history, "_yfinance_window", _yf)
        monkeypatch.setattr(bar_history, "_fetch_recent_bars", _recent)
        return calls

    def test_each_age_comes_from_its_own_source(self, monkeypatch):
        premarket = _bar("2026-09-25T12:00:00Z", v=437)       # 08:00 ET, SIP
        regular_sip = _bar("2026-09-25T17:00:00Z", v=59805)   # 13:00 ET, SIP
        young_iex = _bar("2026-09-25T17:55:00Z", v=160)       # 5 min old, IEX
        calls = self._patch(
            monkeypatch,
            base=[premarket, regular_sip],
            yf=[_bar("2026-09-25T17:00:00Z", v=59813)],
            recent=[young_iex],
        )

        live = bar_history.fetch_live_bars("AAPL", "1Min", "k", "s", "sip_delayed", now=NOW)

        by_t = {_key(b["t"]): b for b in live.bars}
        assert by_t[_key(premarket["t"])]["v"] == 437
        assert by_t[_key(regular_sip["t"])]["v"] == 59805    # SIP, not yfinance
        assert by_t[_key(young_iex["t"])]["v"] == 160
        assert live.provisional == {_key(young_iex["t"])}
        # The SIP bars are settled: they may replace a streamed candle.
        assert live.settled == {_key(premarket["t"]), _key(regular_sip["t"])}
        # Each bar records the tape it came off.
        assert by_t[_key(premarket["t"])]["src"] == "sip"
        assert by_t[_key(regular_sip["t"])]["src"] == "sip"
        assert by_t[_key(young_iex["t"])]["src"] == "iex"
        assert "src" not in young_iex  # the fetchers' own (cached) bars are untouched
        # With SIP in hand yfinance is not asked.
        assert "yf_end" not in calls
        # The young window stops short of the minute in progress, on IEX.
        start, end, feed = calls["recent"]
        assert end == datetime(2026, 9, 25, 18, 0, tzinfo=timezone.utc)
        assert feed == "iex"

    def test_without_sip_yfinance_serves_the_settled_regular_session(self, monkeypatch):
        premarket_iex = _bar("2026-09-25T12:00:00Z", v=20)
        regular_iex = _bar("2026-09-25T17:00:00Z", v=2200)
        calls = self._patch(
            monkeypatch,
            base=[premarket_iex, regular_iex],
            yf=[_bar("2026-09-25T17:00:00Z", v=59813), _bar("2026-09-25T12:00:00Z", v=0)],
            recent=[],
            base_feed="iex",
        )

        live = bar_history.fetch_live_bars("AAPL", "1Min", "k", "s", "iex", now=NOW)

        by_t = {_key(b["t"]): b for b in live.bars}
        assert by_t[_key(regular_iex["t"])]["v"] == 59813
        assert by_t[_key(regular_iex["t"])]["src"] == "yfinance"
        assert by_t[_key(premarket_iex["t"])]["v"] == 20     # yfinance's 0 never used
        assert live.provisional == {_key(premarket_iex["t"])}
        assert live.settled == {_key(regular_iex["t"])}
        assert calls["yf_end"] == datetime(2026, 9, 25, 17, 45, 30, tzinfo=timezone.utc)

    def test_a_bar_still_in_progress_is_not_settled(self, monkeypatch):
        # 15Min at 14:00:30 ET: the 13:30 ET bar ended 13:45, before the
        # 13:45:30 cut; the 13:45 bar started before the cut but runs to 14:00.
        done = _bar("2026-09-25T17:30:00Z", v=900000)
        running = _bar("2026-09-25T17:45:00Z", v=400000)
        self._patch(monkeypatch, base=[done, running], yf=[], recent=[])

        live = bar_history.fetch_live_bars("AAPL", "15Min", "k", "s", "sip_delayed", now=NOW)

        assert live.settled == {_key(done["t"])}

    def test_the_resolved_feed_never_supplies_young_bars(self, monkeypatch):
        young = _bar("2026-09-25T17:58:00Z", v=99999)
        self._patch(monkeypatch, base=[young], yf=[], recent=[])

        live = bar_history.fetch_live_bars("AAPL", "1Min", "k", "s", "sip_delayed", now=NOW)

        assert live.bars == []

    def test_a_realtime_sip_key_serves_the_young_window_as_settled(self, monkeypatch):
        young = _bar("2026-09-25T17:55:00Z", v=27000)
        calls = self._patch(monkeypatch, base=[], yf=[], recent=[young])

        live = bar_history.fetch_live_bars("AAPL", "1Min", "k", "s", "sip", now=NOW)

        assert calls["recent"][2] == "sip"
        assert live.provisional == set()

    def test_raises_only_when_nothing_answered(self, monkeypatch):
        def _boom(*a, **k):
            raise RuntimeError("down")

        monkeypatch.setattr(bar_history, "fetch_history_bars", _boom)
        monkeypatch.setattr(bar_history, "_yfinance_window", _boom)
        monkeypatch.setattr(bar_history, "_fetch_recent_bars", _boom)
        try:
            bar_history.fetch_live_bars("AAPL", "1Min", "k", "s", "sip_delayed", now=NOW)
        except RuntimeError:
            return
        raise AssertionError("expected RuntimeError")


class TestMergeLiveBars:
    def test_settled_bar_replaces_a_provisional_one(self):
        state = _state()
        state.bars.append(_bar("2026-09-25T17:40:00Z", v=1000))
        state.provisional_bar_keys.add(_key("2026-09-25T17:40:00Z"))

        added, replaced = stream_common.merge_live_bars(
            state, [_bar("2026-09-25T17:40:00Z", v=30000)], set()
        )

        assert (added, replaced) == (0, 1)
        assert state.bars[0]["v"] == 30000
        assert state.provisional_bar_keys == set()

    def test_young_streamed_bars_are_never_replaced(self):
        state = _state()
        state.bars.append({**_bar("2026-09-25T17:55:00Z", v=25000), "src": "finnhub"})

        added, replaced = stream_common.merge_live_bars(
            state, [{**_bar("2026-09-25T17:55:00Z", v=30000), "src": "sip"}], set()
        )

        assert (added, replaced) == (0, 0)
        assert state.bars[0]["v"] == 25000

    def test_a_settled_sip_bar_replaces_an_older_streamed_one(self):
        state = _state()
        state.bars.append({**_bar("2026-09-25T17:40:00Z", v=2500), "src": "finnhub"})
        state.bars.append(_bar("2026-09-25T17:41:00Z", v=2600))  # untagged, also replaced

        added, replaced = stream_common.merge_live_bars(
            state,
            [{**_bar("2026-09-25T17:40:00Z", v=30000), "src": "sip"},
             {**_bar("2026-09-25T17:41:00Z", v=31000), "src": "sip"}],
            set(),
            {_key("2026-09-25T17:40:00Z"), _key("2026-09-25T17:41:00Z")},
        )

        assert (added, replaced) == (0, 2)
        assert [(b["v"], b["src"]) for b in state.bars] == [(30000, "sip"), (31000, "sip")]

    def test_sip_replaces_yfinance_and_never_the_other_way(self):
        state = _state()
        state.bars.append({**_bar("2026-09-25T17:00:00Z", v=59813), "src": "yfinance"})
        state.bars.append({**_bar("2026-09-25T17:01:00Z", v=50000), "src": "sip"})
        settled = {_key("2026-09-25T17:00:00Z"), _key("2026-09-25T17:01:00Z")}

        added, replaced = stream_common.merge_live_bars(
            state,
            [{**_bar("2026-09-25T17:00:00Z", v=59805), "src": "sip"},
             {**_bar("2026-09-25T17:01:00Z", v=50500), "src": "yfinance"}],
            set(),
            settled,
        )

        assert (added, replaced) == (0, 1)
        assert [(b["v"], b["src"]) for b in state.bars] == [(59805, "sip"), (50000, "sip")]

    def test_iex_never_replaces_anything(self):
        state = _state()
        state.bars.append(_bar("2026-09-25T17:40:00Z", v=900))
        state.provisional_bar_keys.add(_key("2026-09-25T17:40:00Z"))

        added, replaced = stream_common.merge_live_bars(
            state,
            [_bar("2026-09-25T17:40:00Z", v=800)],
            {_key("2026-09-25T17:40:00Z")},
        )

        assert (added, replaced) == (0, 0)
        assert state.bars[0]["v"] == 900

    def test_iex_fills_a_gap_and_is_remembered_as_provisional(self):
        state = _state()
        state.bars.append(_bar("2026-09-25T17:40:00Z"))

        added, _ = stream_common.merge_live_bars(
            state,
            [_bar("2026-09-25T17:55:00Z", v=160)],
            {_key("2026-09-25T17:55:00Z")},
        )

        assert added == 1
        assert [b["t"] for b in state.bars] == ["2026-09-25T17:40:00Z", "2026-09-25T17:55:00Z"]
        assert state.provisional_bar_keys == {_key("2026-09-25T17:55:00Z")}


class TestStopStartEdges:
    def test_restart_flags_the_minute_the_stop_cut_short(self):
        state = _state()
        state.bars.extend([_bar("2026-09-25T17:39:00Z"), _bar("2026-09-25T17:40:00Z", v=12)])

        stream_common.reset_symbol_for_new_stream(state)

        assert state.provisional_bar_keys == {_key("2026-09-25T17:40:00Z")}

    def test_finnhubs_first_candle_is_provisional_and_later_ones_are_not(self):
        state = _state()
        builder = finnhub_stream.CandleBuilder(state, 1)

        builder.add_trade(1.0, 10, "2026-09-25T17:50:40Z")
        builder.add_trade(1.0, 10, "2026-09-25T17:51:05Z")

        assert state.provisional_bar_keys == {_key("2026-09-25T17:50:00Z")}

    def test_backfill_after_restart_settles_old_bars_and_fills_the_gap(self, monkeypatch):
        state = _state()
        streamed = {**_bar("2026-09-25T17:00:00Z", v=2000), "src": "finnhub"}
        cut_short = {**_bar("2026-09-25T17:01:00Z", v=500), "src": "finnhub"}
        young = {**_bar("2026-09-25T17:55:00Z", v=1800), "src": "finnhub"}
        state.bars.extend([streamed, cut_short, young])
        stream_common.reset_symbol_for_new_stream(state)  # flags `young`, the newest

        def _history(*a, **k):
            return (
                [_bar("2026-09-25T17:00:00Z", v=59000), _bar("2026-09-25T17:01:00Z", v=44000),
                 _bar("2026-09-25T17:02:00Z", v=53000)],
                bar_history.SOURCE_LABELS["sip_delayed"],
                [],
            )

        monkeypatch.setattr(bar_history, "fetch_history_bars", _history)
        monkeypatch.setattr(bar_history, "datetime", _FrozenDatetime)

        changed, _source = stream.backfill_bars("AAPL", "k", "s", "sip_delayed", state, "1Min")

        # 17:00 and 17:01 settled to SIP, 17:02 added; the young candle stays,
        # still provisional until IEX or SIP serves its minute.
        assert changed == 3
        assert [(b["v"], b["src"]) for b in state.bars] == [
            (59000, "sip"), (44000, "sip"), (53000, "sip"), (1800, "finnhub"),
        ]
        assert state.provisional_bar_keys == {_key("2026-09-25T17:55:00Z")}


class _FrozenDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW


class TestWeekMinuteBars:
    """`fetch_week_minute_bars`: a week of one feed's minutes, once a day."""

    def _patch(self, monkeypatch, result):
        calls = []

        def _range(symbol, timeframe, start, end, key, secret, feed):
            calls.append((symbol, feed, start, end))
            if isinstance(result, Exception):
                raise result
            return result

        monkeypatch.setattr(bar_history, "fetch_bars_range", _range)
        monkeypatch.setattr(bar_history, "_week_minute_cache", {})
        return calls

    def test_fetched_once_per_feed_per_day(self, monkeypatch):
        calls = self._patch(monkeypatch, [_bar("2026-09-24T14:00:00Z")])
        for _ in range(3):
            assert len(bar_history.fetch_week_minute_bars("AAPL", "iex", "k", "s", 12, now=NOW)) == 1
        bar_history.fetch_week_minute_bars("AAPL", "sip", "k", "s", 12, now=NOW)
        assert [c[1] for c in calls] == ["iex", "sip"]
        # Ends SIP_DELAY_MIN back, which delayed SIP allows.
        assert calls[0][3] == NOW - timedelta(minutes=bar_history.SIP_DELAY_MIN)

    def test_a_refusal_is_not_retried_every_rerun(self, monkeypatch):
        calls = self._patch(monkeypatch, RuntimeError("403"))
        assert bar_history.fetch_week_minute_bars("AAPL", "sip", "k", "s", 12, now=NOW) == []
        assert bar_history.fetch_week_minute_bars("AAPL", "sip", "k", "s", 12, now=NOW) == []
        assert len(calls) == 1

    def test_no_key_no_request(self, monkeypatch):
        calls = self._patch(monkeypatch, [])
        assert bar_history.fetch_week_minute_bars("AAPL", "sip", "", "", 12, now=NOW) == []
        assert not calls
