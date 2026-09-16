import pytest

from agent_stonks import bar_history


def _bar(ts: str, v: float = 100.0) -> dict:
    return {"t": ts, "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": v}


class _Boom(Exception):
    pass


def _raiser(*_a, **_k):
    raise _Boom("unavailable")


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Nothing in this module may reach the network: every source is stubbed by
    default, and each test overrides only the ones it is about."""
    monkeypatch.setattr(bar_history, "fetch_bars", _raiser)
    monkeypatch.setattr(bar_history, "fetch_bars_window", _raiser)
    monkeypatch.setattr(bar_history, "fetch_daily_bars", _raiser)
    monkeypatch.setattr(bar_history, "fetch_intraday_bars", _raiser)


class TestResolveHistoryFeed:
    def test_auto_prefers_sip_when_the_key_is_entitled(self, monkeypatch):
        monkeypatch.setattr(bar_history, "fetch_bars", lambda *a, **k: [_bar("t")])
        assert bar_history.resolve_history_feed("auto", "AAPL", "k", "s") == "sip"

    def test_auto_falls_back_to_yfinance_not_iex_when_sip_is_denied(self, monkeypatch):
        # The whole point: an unentitled key must not silently land on IEX while
        # a consolidated free source is available.
        monkeypatch.setattr(bar_history, "fetch_bars", _raiser)
        assert bar_history.resolve_history_feed("auto", "AAPL", "k", "s") == "yfinance"

    def test_auto_reaches_iex_only_when_yfinance_cannot_serve_the_timeframe(self, monkeypatch):
        monkeypatch.setattr(bar_history, "fetch_bars", _raiser)
        assert bar_history.resolve_history_feed("auto", "AAPL", "k", "s", "1Day") == "iex"

    def test_auto_prefers_delayed_sip_over_yfinance(self, monkeypatch):
        # Alpaca's free/basic plans refuse only the trailing 15 minutes of SIP.
        # That is still the consolidated tape, and still the right backfill
        # source -- it must not be mistaken for "no SIP" and demoted to Yahoo.
        monkeypatch.setattr(bar_history, "fetch_bars", _raiser)
        monkeypatch.setattr(bar_history, "fetch_bars_window", lambda *a, **k: [_bar("t")])
        assert bar_history.resolve_history_feed("auto", "AAPL", "k", "s") == "sip_delayed"

    def test_explicit_sip_resolves_to_the_delayed_tier_rather_than_403ing(self, monkeypatch):
        monkeypatch.setattr(bar_history, "fetch_bars", _raiser)
        monkeypatch.setattr(bar_history, "fetch_bars_window", lambda *a, **k: [_bar("t")])
        assert bar_history.resolve_history_feed("sip", "AAPL", "k", "s") == "sip_delayed"

    def test_explicit_sip_falls_back_when_there_is_no_sip_at_all(self, monkeypatch):
        assert bar_history.resolve_history_feed("sip", "AAPL", "k", "s") == "yfinance"

    def test_probes_sip_specifically(self, monkeypatch):
        seen = {}

        def _probe(symbol, timeframe, limit, key, secret, feed, **kw):
            seen["feed"] = feed
            return [_bar("t")]

        monkeypatch.setattr(bar_history, "fetch_bars", _probe)
        bar_history.resolve_history_feed("auto", "AAPL", "k", "s")
        assert seen["feed"] == "sip"

    def test_an_explicit_choice_is_honoured_without_probing(self, monkeypatch):
        monkeypatch.setattr(bar_history, "fetch_bars", _raiser)
        assert bar_history.resolve_history_feed("iex", "AAPL", "k", "s") == "iex"
        assert bar_history.resolve_history_feed("yfinance", "AAPL", "k", "s") == "yfinance"

    def test_explicit_yfinance_degrades_for_a_timeframe_it_cannot_serve(self, monkeypatch):
        monkeypatch.setattr(bar_history, "fetch_bars", lambda *a, **k: [_bar("t")])
        assert bar_history.resolve_history_feed("yfinance", "AAPL", "k", "s", "1Day") == "sip"

    def test_an_unknown_choice_falls_back_to_the_default(self, monkeypatch):
        monkeypatch.setattr(bar_history, "fetch_bars", lambda *a, **k: [_bar("t")])
        assert bar_history.resolve_history_feed("nonsense", "AAPL", "k", "s") == "sip"


class TestFetchHistoryBars:
    def test_uses_the_resolved_feed(self, monkeypatch):
        seen = {}

        def _fetch(symbol, timeframe, limit, key, secret, feed, **kw):
            seen["feed"] = feed
            return [_bar("2024-01-01T14:00:00Z")]

        monkeypatch.setattr(bar_history, "fetch_bars", _fetch)
        bars, source, failures = bar_history.fetch_history_bars("AAPL", "1Min", "k", "s", "sip")

        assert seen["feed"] == "sip"
        assert len(bars) == 1
        assert source == bar_history.SOURCE_LABELS["sip"]
        assert failures == []

    def test_never_passes_the_unresolved_auto_to_alpaca(self, monkeypatch):
        # "auto" is a choice, not a feed -- reaching Alpaca as feed=auto would
        # be rejected on every call.
        seen = []

        def _fetch(symbol, timeframe, limit, key, secret, feed, **kw):
            seen.append(feed)
            return [_bar("2024-01-01T14:00:00Z")]

        monkeypatch.setattr(bar_history, "fetch_bars", _fetch)
        bar_history.fetch_history_bars("AAPL", "1Min", "k", "s", "auto")

        assert "auto" not in seen
        assert seen == ["sip"]

    def test_sip_failure_falls_through_to_yfinance_before_iex(self, monkeypatch):
        tried = []

        def _fetch(symbol, timeframe, limit, key, secret, feed, **kw):
            tried.append(feed)
            raise _Boom(f"{feed} down")

        monkeypatch.setattr(bar_history, "fetch_bars", _fetch)
        monkeypatch.setattr(
            bar_history, "fetch_intraday_bars",
            lambda symbol, interval="1m": [_bar("2024-01-01T14:00:00Z")],
        )

        bars, source, failures = bar_history.fetch_history_bars("AAPL", "1Min", "k", "s", "sip")

        # yfinance answered before IEX was reached; IEX is never tried early
        assert tried == ["sip"]
        assert source == bar_history.SOURCE_LABELS["yfinance"]
        assert [f[0] for f in failures] == [
            bar_history.SOURCE_LABELS["sip"], bar_history.SOURCE_LABELS["sip_delayed"]
        ]

    def test_delayed_sip_holds_the_window_back_and_outranks_yfinance(self, monkeypatch):
        import datetime as _dt

        seen = {}

        def _window(symbol, timeframe, start, end, key, secret, feed="iex", limit=200,
                    keep="oldest"):
            seen["feed"] = feed
            seen["end"] = end
            seen["keep"] = keep
            return [_bar("2024-01-01T14:00:00Z")]

        monkeypatch.setattr(bar_history, "fetch_bars_window", _window)
        monkeypatch.setattr(
            bar_history, "fetch_intraday_bars",
            lambda *a, **k: pytest.fail("sip_delayed must outrank yfinance"),
        )

        bars, source, _ = bar_history.fetch_history_bars(
            "AAPL", "1Min", "k", "s", "sip_delayed"
        )

        assert source == bar_history.SOURCE_LABELS["sip_delayed"]
        assert seen["feed"] == "sip"  # it is the SIP feed, just held back
        behind = _dt.datetime.now(_dt.timezone.utc) - seen["end"]
        assert behind >= _dt.timedelta(minutes=bar_history.SIP_DELAY_MIN - 1)
        # A 16-hour window of minute bars overruns the limit for most of a
        # session, and the end of it is the half a backfill needs.
        assert seen["keep"] == "newest"

    def test_iex_is_the_last_resort_when_yfinance_also_fails(self, monkeypatch):
        tried = []

        def _fetch(symbol, timeframe, limit, key, secret, feed, **kw):
            tried.append(feed)
            if feed == "iex":
                return [_bar("2024-01-01T14:00:00Z")]
            raise _Boom(f"{feed} down")

        monkeypatch.setattr(bar_history, "fetch_bars", _fetch)
        monkeypatch.setattr(bar_history, "fetch_intraday_bars", _raiser)

        bars, source, failures = bar_history.fetch_history_bars("AAPL", "1Min", "k", "s", "sip")

        assert tried == ["sip", "iex"]
        assert source == bar_history.SOURCE_LABELS["iex"]
        assert len(failures) == 3  # sip, sip_delayed, yfinance

    def test_passes_the_matching_yfinance_interval(self, monkeypatch):
        seen = {}

        def _yf(symbol, interval="1m"):
            seen["interval"] = interval
            return [_bar("2024-01-01T14:00:00Z")]

        monkeypatch.setattr(bar_history, "fetch_intraday_bars", _yf)
        bar_history.fetch_history_bars("AAPL", "15Min", "k", "s", "yfinance")
        assert seen["interval"] == "15m"

    def test_skips_yfinance_for_a_timeframe_it_cannot_serve(self, monkeypatch):
        tried = []

        def _fetch(symbol, timeframe, limit, key, secret, feed, **kw):
            tried.append(feed)
            if feed == "iex":
                return [_bar("2024-01-01T14:00:00Z")]
            raise _Boom("down")

        monkeypatch.setattr(bar_history, "fetch_bars", _fetch)
        monkeypatch.setattr(
            bar_history, "fetch_intraday_bars",
            lambda *a, **k: pytest.fail("yfinance cannot serve 1Day"),
        )

        _, source, _ = bar_history.fetch_history_bars("AAPL", "1Day", "k", "s", "sip")
        assert source == bar_history.SOURCE_LABELS["iex"]
        assert tried == ["sip", "iex"]

    def test_raises_only_when_every_source_failed(self, monkeypatch):
        monkeypatch.setattr(bar_history, "fetch_bars", _raiser)
        monkeypatch.setattr(bar_history, "fetch_intraday_bars", _raiser)

        with pytest.raises(RuntimeError):
            bar_history.fetch_history_bars("AAPL", "1Min", "k", "s", "sip")


class TestSessionConsistency:
    """One buffer, one set of volume units: the initial load, the backfill and
    the fallback poll must not each pick their own feed."""

    def test_launch_stream_reuses_the_feed_the_session_resolved(self, monkeypatch):
        from agent_stonks import stream
        from agent_stonks.state import AppState

        app = AppState()
        app.set_symbols(["AAPL"])
        app.history_feed_resolved = "sip"
        monkeypatch.setattr(
            bar_history, "resolve_history_feed",
            lambda *a, **k: pytest.fail("must not re-resolve an already-resolved session"),
        )
        monkeypatch.setattr(stream.finnhub_stream, "launch", lambda *a, **k: None)

        passed = {}

        class FakeThread:
            def __init__(self, target=None, args=(), **k):
                passed[getattr(target, "__name__", "?")] = args

            def start(self):
                pass

        monkeypatch.setattr(stream.threading, "Thread", FakeThread)
        stream.launch_stream(
            ["AAPL"], "k", "s", "iex", app, "1Min",
            data_source="finnhub", finnhub_token="tok",
        )

        assert app.history_feed_resolved == "sip"
        # _fallback_bars_loop's last arg is the history feed it will backfill from
        assert passed["_fallback_bars_loop"][-1] == "sip"

    def test_launch_stream_resolves_once_when_the_session_has_not(self, monkeypatch):
        from agent_stonks import stream
        from agent_stonks.state import AppState

        app = AppState()
        app.set_symbols(["AAPL"])
        calls = []
        monkeypatch.setattr(
            bar_history, "resolve_history_feed",
            lambda *a, **k: (calls.append(a) or "yfinance"),
        )
        monkeypatch.setattr(stream.finnhub_stream, "launch", lambda *a, **k: None)
        monkeypatch.setattr(
            stream.threading, "Thread",
            lambda **k: type("T", (), {"start": lambda self: None})(),
        )

        stream.launch_stream(
            ["AAPL"], "k", "s", "iex", app, "1Min",
            data_source="finnhub", finnhub_token="tok",
        )

        assert len(calls) == 1
        assert app.history_feed_resolved == "yfinance"

    def test_backfill_uses_the_history_feed_not_the_stream_feed(self, monkeypatch):
        from agent_stonks import stream
        from agent_stonks.state import AppState

        app = AppState()
        app.set_symbols(["AAPL"])
        state = app.sym("AAPL")
        state.bars.append(_bar("2024-01-01T14:00:00Z"))
        seen = {}

        def _fetch(symbol, timeframe, limit, key, secret, feed, **kw):
            seen["feed"] = feed
            return [_bar("2024-01-01T14:01:00Z")]

        monkeypatch.setattr(bar_history, "fetch_bars", _fetch)
        # stream feed is iex; history feed is sip -- the backfill must use sip
        added, source = stream.backfill_bars("AAPL", "k", "s", "sip", state, "1Min")

        assert seen["feed"] == "sip"
        assert added == 1


class TestDailyBaseline:
    """The daily series is the denominator of every volume comparison -- the
    high-volume alert, the volume_ratio/rvol_pace alert fields, and the
    briefing's relative-volume pace. It has to come off the same tape as the
    intraday bars or all three are wrong by the feeds' ratio."""

    def test_uses_the_sessions_feed(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(
            bar_history, "fetch_daily_bars",
            lambda symbol, key, secret, feed, **kw: seen.update(feed=feed) or [_bar("t")],
        )
        bars, source = bar_history.fetch_daily("AAPL", "k", "s", "iex")
        assert seen["feed"] == "iex"
        assert source == bar_history.SOURCE_LABELS["iex"]

    def test_delayed_sip_holds_the_window_back(self, monkeypatch):
        import datetime as _dt

        seen = {}

        def _window(symbol, timeframe, start, end, key, secret, feed="iex", limit=200):
            seen.update(timeframe=timeframe, feed=feed, end=end)
            return [_bar("t")]

        monkeypatch.setattr(bar_history, "fetch_bars_window", _window)
        _, source = bar_history.fetch_daily("AAPL", "k", "s", "sip_delayed")

        assert seen["timeframe"] == "1Day"
        assert seen["feed"] == "sip"
        assert _dt.datetime.now(_dt.timezone.utc) - seen["end"] >= _dt.timedelta(
            minutes=bar_history.SIP_DELAY_MIN - 1
        )
        assert source == bar_history.SOURCE_LABELS["sip_delayed"]

    def test_falls_back_to_iex_rather_than_leaving_no_baseline(self, monkeypatch):
        monkeypatch.setattr(
            bar_history, "fetch_daily_bars",
            lambda symbol, key, secret, feed, **kw: (
                [_bar("t")] if feed == "iex" else _raiser()
            ),
        )
        _, source = bar_history.fetch_daily("AAPL", "k", "s", "sip")
        assert source == bar_history.SOURCE_LABELS["iex"]

    def test_raises_when_every_source_failed(self, monkeypatch):
        with pytest.raises(RuntimeError):
            bar_history.fetch_daily("AAPL", "k", "s", "sip")


class TestFeedRanking:
    """The one ranking of tapes: `feed_order` and `is_consolidated`. Every path
    that picks a source reads them, so the preference is stated once."""

    def test_the_chosen_feed_comes_first_and_the_rest_follow_in_quality_order(self):
        assert bar_history.feed_order("yfinance") == [
            "yfinance", "sip", "sip_delayed", "iex"
        ]

    def test_iex_is_always_last_when_it_was_not_asked_for(self):
        for feed in ("auto", "sip", "sip_delayed", "yfinance", "nonsense", ""):
            assert bar_history.feed_order(feed)[-1] == "iex"

    def test_an_unresolved_choice_is_not_passed_through_as_a_feed(self):
        """"auto" is a choice, not a tape -- Alpaca would take it literally."""
        assert bar_history.feed_order("auto") == list(bar_history.CONCRETE_FEEDS)

    def test_every_feed_appears_exactly_once(self):
        order = bar_history.feed_order("iex")
        assert sorted(order) == sorted(bar_history.CONCRETE_FEEDS)

    @pytest.mark.parametrize("feed", ["sip", "sip_delayed", "yfinance", "finnhub"])
    def test_the_consolidated_tapes(self, feed):
        assert bar_history.is_consolidated(feed)

    @pytest.mark.parametrize("feed", ["iex", "IEX", "", None])
    def test_iex_is_not_consolidated_and_neither_is_an_unknown_tape(self, feed):
        assert not bar_history.is_consolidated(feed)
