from marketview import stream
from marketview.state import AppState


def _app(*symbols: str):
    """AppState streaming `symbols` (default AAPL) plus its first SymbolState."""
    app = AppState()
    app.set_symbols(list(symbols) or ["AAPL"])
    return app, app.sym(app.symbols[0])


class _StopAfter:
    """Fake stop_event that lets the fallback loop body run `n` times, then ends it.

    Mirrors threading.Event's `.wait()` interface (the only method the loops use):
    returns False (don't stop) for the first `n` calls, True (stop) after that.
    """

    def __init__(self, n: int) -> None:
        self.n = n
        self.calls = 0

    def wait(self, timeout: float | None = None) -> bool:
        self.calls += 1
        return self.calls > self.n


def test_fallback_bars_loop_updates_state_from_rest_when_disconnected(monkeypatch):
    app, state = _app()
    app.bars_connected = False
    bars = [{"t": "2024-01-01T14:00:00Z", "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 100}]

    monkeypatch.setattr(stream, "fetch_bars", lambda *a, **k: bars)
    monkeypatch.setattr(stream, "fetch_trades", lambda *a, **k: [{"p": 1.6}])

    stream._fallback_bars_loop(["AAPL"], "k", "s", "iex", app, "1Min", _StopAfter(1))

    assert list(state.bars) == bars
    assert state.last_price == 1.6
    assert state.previous_minute_high == 2
    assert state.previous_minute_low == 0.5
    assert state.day_volume == 100
    assert "Fallback" in app.status
    assert "Alpaca REST" in app.status


def test_fallback_bars_loop_polls_every_symbol(monkeypatch):
    app, _ = _app("AAPL", "TSLA")
    app.bars_connected = False
    bars = [{"t": "2024-01-01T14:00:00Z", "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 100}]
    polled = []

    def _fetch_bars(symbol, *a, **k):
        polled.append(symbol)
        return bars

    monkeypatch.setattr(stream, "fetch_bars", _fetch_bars)
    monkeypatch.setattr(stream, "fetch_trades", lambda *a, **k: [{"p": 1.6}])

    stream._fallback_bars_loop(["AAPL", "TSLA"], "k", "s", "iex", app, "1Min", _StopAfter(1))

    assert polled == ["AAPL", "TSLA"]
    assert list(app.sym("AAPL").bars) == bars
    assert list(app.sym("TSLA").bars) == bars


def test_fallback_bars_loop_refreshes_quote_when_disconnected(monkeypatch):
    app, state = _app()
    app.bars_connected = False
    bars = [{"t": "2024-01-01T14:00:00Z", "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 100}]

    monkeypatch.setattr(stream, "fetch_bars", lambda *a, **k: bars)
    monkeypatch.setattr(stream, "fetch_trades", lambda *a, **k: [{"p": 1.6}])
    monkeypatch.setattr(
        stream, "fetch_latest_quote", lambda *a, **k: {"bp": 1.55, "bs": 10, "ap": 1.57, "as": 20}
    )

    stream._fallback_bars_loop(["AAPL"], "k", "s", "iex", app, "1Min", _StopAfter(1))

    assert state.bid_price == 1.55
    assert state.bid_size == 10
    assert state.ask_price == 1.57
    assert state.ask_size == 20


def test_apply_quote_treats_zero_price_as_no_quote():
    """Alpaca reports a one-sided book as bp/ap = 0 -- that side must become None,
    not a 0.0 that would corrupt the spread and bid/ask alerts."""
    _, state = _app()
    state.bid_price = 1.5
    ts = "2026-07-02T14:51:59.118379395Z"
    stream._apply_quote(state, {"bp": 0, "bs": 0, "ap": 1.57, "as": 20, "t": ts})

    assert state.bid_price is None
    assert state.ask_price == 1.57
    assert state.quote_ts == ts


def test_fallback_bars_loop_keeps_last_quote_when_quote_fetch_fails(monkeypatch):
    app, state = _app()
    app.bars_connected = False
    state.bid_price = 1.5
    state.ask_price = 1.6
    bars = [{"t": "2024-01-01T14:00:00Z", "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 100}]

    monkeypatch.setattr(stream, "fetch_bars", lambda *a, **k: bars)
    monkeypatch.setattr(stream, "fetch_trades", lambda *a, **k: [{"p": 1.6}])

    def _raise(*a, **k):
        raise RuntimeError("quotes endpoint down")

    monkeypatch.setattr(stream, "fetch_latest_quote", _raise)

    stream._fallback_bars_loop(["AAPL"], "k", "s", "iex", app, "1Min", _StopAfter(1))

    assert state.bid_price == 1.5
    assert state.ask_price == 1.6


def test_fallback_bars_loop_backfills_instead_of_polling_when_stream_connected(monkeypatch):
    """While the WS is healthy the loop must not snapshot-replace state, but it
    does run a periodic backfill that merges only-missing bars."""
    app, state = _app()
    app.bars_connected = True
    app.status = "✅ Streaming AAPL (IEX)"
    live_bar = {"t": "2024-01-01T14:00:00Z", "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 100}
    missed_bar = {"t": "2024-01-01T14:01:00Z", "o": 1.5, "h": 1.6, "l": 1.4, "c": 1.6, "v": 50}
    state.bars.append(live_bar)
    monkeypatch.setattr(stream, "fetch_bars", lambda *a, **k: [dict(live_bar, c=9.9), missed_bar])

    stream._fallback_bars_loop(["AAPL"], "k", "s", "iex", app, "1Min", _StopAfter(2))

    assert list(state.bars) == [live_bar, missed_bar]  # hole filled, live bar untouched
    assert app.status == "✅ Streaming AAPL (IEX)"  # no fallback warning
    assert state.last_price is None  # snapshot-replace path did not run


def test_fallback_bars_loop_falls_back_to_yfinance_when_rest_fails(monkeypatch):
    app, state = _app()
    app.bars_connected = False
    bars = [{"t": "2024-01-01T14:00:00Z", "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 100}]

    def _raise(*a, **k):
        raise RuntimeError("Alpaca down")

    monkeypatch.setattr(stream, "fetch_bars", _raise)
    monkeypatch.setattr(stream, "fetch_intraday_bars", lambda *a, **k: bars)

    stream._fallback_bars_loop(["AAPL"], "k", "s", "iex", app, "1Min", _StopAfter(1))

    assert list(state.bars) == bars
    assert "yfinance" in app.status


def test_fallback_news_loop_appends_new_articles_when_disconnected(monkeypatch):
    app, state = _app()
    app.news_connected = False
    state.news = [{"id": "1", "headline": "old"}]
    fresh = [{"id": "1", "headline": "old"}, {"id": "2", "headline": "new"}]

    monkeypatch.setattr(stream, "fetch_news_with_fallback", lambda *a, **k: fresh)

    stream._fallback_news_loop(["AAPL"], "k", "s", "wn-key", app, _StopAfter(1))

    assert [a["id"] for a in state.news] == ["1", "2"]
    assert app.agent_wake_event.is_set()
    assert "Fallback" in app.news_status


def test_fallback_news_loop_skips_polling_when_stream_connected(monkeypatch):
    app, _ = _app()
    app.news_connected = True
    called = []
    monkeypatch.setattr(stream, "fetch_news_with_fallback", lambda *a, **k: called.append(1))

    stream._fallback_news_loop(["AAPL"], "k", "s", "wn-key", app, _StopAfter(2))

    assert called == []


def test_fire_due_alerts_wakes_on_price_field():
    import time
    app, state = _app()
    state.last_price = 151.0
    state.recent_prices.append((time.monotonic(), 151.0))
    state.alerts = [{"symbol": "AAPL", "field": "last_price", "condition": "above", "value": 150.0}]

    stream._fire_due_alerts(state)

    assert state.alerts == []  # cleared once fired
    assert app.agent_wake_event.is_set()
    assert "last_price above 150" in app.agent_wake_reason


def test_fire_due_alerts_checks_other_symbols_alerts_too():
    """A tick on one symbol re-checks every symbol's pending alerts (the shared
    portfolio value, for instance, moves on any symbol's trade)."""
    app, aapl = _app("AAPL", "TSLA")
    tsla = app.sym("TSLA")
    tsla.day_volume = 6_000_000
    tsla.alerts = [{"symbol": "TSLA", "field": "day_volume", "condition": "above", "value": 5_000_000}]

    stream._fire_due_alerts(aapl)  # tick arrived on AAPL

    assert tsla.alerts == []
    assert app.agent_wake_event.is_set()
    assert "TSLA" in app.agent_wake_reason


def test_fire_due_alerts_wakes_on_non_price_field():
    """Alerts on continuously-updated, non-price fields (here cumulative day volume)
    fire too -- not just price levels."""
    app, state = _app()
    state.day_volume = 6_000_000
    state.alerts = [{"symbol": "AAPL", "field": "day_volume", "condition": "above", "value": 5_000_000}]

    stream._fire_due_alerts(state)

    assert state.alerts == []
    assert app.agent_wake_event.is_set()


def test_fire_due_alerts_does_not_wake_when_unmet():
    import time
    app, state = _app()
    state.last_price = 149.0
    state.recent_prices.append((time.monotonic(), 149.0))
    alert = {"symbol": "AAPL", "field": "last_price", "condition": "above", "value": 150.0}
    state.alerts = [alert]

    stream._fire_due_alerts(state)

    assert state.alerts == [alert]
    assert not app.agent_wake_event.is_set()


def test_fire_due_alerts_wakes_on_prior_price_in_window():
    """A last_price alert fires if any price within the last minute crossed the
    threshold, even if the current last_price has since reverted below it."""
    import time
    app, state = _app()
    now = time.monotonic()
    # Spike to 152 happened 30 s ago; current price is back at 148.
    state.recent_prices.append((now - 30, 152.0))
    state.recent_prices.append((now, 148.0))
    state.last_price = 148.0
    state.alerts = [{"symbol": "AAPL", "field": "last_price", "condition": "above", "value": 150.0}]

    stream._fire_due_alerts(state)

    assert state.alerts == []
    assert app.agent_wake_event.is_set()


def test_fire_due_alerts_ignores_stale_price_outside_window():
    """Prices older than 60 s do not satisfy a last_price alert."""
    import time
    app, state = _app()
    now = time.monotonic()
    # Spike happened 90 s ago -- outside the 60 s window.
    state.recent_prices.append((now - 90, 155.0))
    state.recent_prices.append((now, 148.0))
    state.last_price = 148.0
    alert = {"symbol": "AAPL", "field": "last_price", "condition": "above", "value": 150.0}
    state.alerts = [alert]

    stream._fire_due_alerts(state)

    assert state.alerts == [alert]
    assert not app.agent_wake_event.is_set()


def test_fallback_bars_loop_fires_volume_alert(monkeypatch):
    """A day_volume alert fires from the REST fallback path, not just the live stream."""
    app, state = _app()
    app.bars_connected = False
    state.alerts = [{"symbol": "AAPL", "field": "day_volume", "condition": "above", "value": 50.0}]
    bars = [{"t": "2024-01-01T14:00:00Z", "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 100}]

    monkeypatch.setattr(stream, "fetch_bars", lambda *a, **k: bars)
    monkeypatch.setattr(stream, "fetch_trades", lambda *a, **k: [{"p": 1.6}])

    stream._fallback_bars_loop(["AAPL"], "k", "s", "iex", app, "1Min", _StopAfter(1))

    assert state.alerts == []
    assert app.agent_wake_event.is_set()


def _bar(ts: str, c: float = 1.0, v: float = 100.0) -> dict:
    return {"t": ts, "o": c, "h": c, "l": c, "c": c, "v": v}


class TestMergeMissingBars:
    def test_inserts_only_missing_timestamps_in_order(self):
        _, state = _app()
        state.bars.extend([_bar("2024-01-01T14:00:00Z"), _bar("2024-01-01T14:03:00Z")])

        added = stream.merge_missing_bars(
            state,
            [
                _bar("2024-01-01T14:00:00Z", c=9.9),  # collision: streamed bar wins
                _bar("2024-01-01T14:01:00Z"),
                _bar("2024-01-01T14:02:00Z"),
            ],
        )

        assert added == 2
        ts = [b["t"] for b in state.bars]
        assert ts == [
            "2024-01-01T14:00:00Z",
            "2024-01-01T14:01:00Z",
            "2024-01-01T14:02:00Z",
            "2024-01-01T14:03:00Z",
        ]
        assert state.bars[0]["c"] == 1.0  # not clobbered by the REST copy

    def test_treats_z_and_offset_timestamps_as_equal(self):
        _, state = _app()
        state.bars.append(_bar("2024-01-01T14:00:00+00:00"))
        added = stream.merge_missing_bars(state, [_bar("2024-01-01T14:00:00Z")])
        assert added == 0
        assert len(state.bars) == 1

    def test_empty_fetch_is_a_noop(self):
        _, state = _app()
        assert stream.merge_missing_bars(state, []) == 0


class TestBackfillBars:
    def test_uses_rest_and_reports_added_count(self, monkeypatch):
        _, state = _app()
        state.bars.append(_bar("2024-01-01T14:00:00Z"))
        monkeypatch.setattr(
            stream, "fetch_bars", lambda *a, **k: [_bar("2024-01-01T14:01:00Z")]
        )

        added, source = stream.backfill_bars("AAPL", "k", "s", "iex", state, "1Min")

        assert added == 1
        assert source == "Alpaca REST"

    def test_falls_back_to_yfinance_when_rest_fails(self, monkeypatch):
        _, state = _app()

        def _boom(*a, **k):
            raise RuntimeError("REST down")

        monkeypatch.setattr(stream, "fetch_bars", _boom)
        seen = {}

        def _yf(symbol, interval="1m"):
            seen["interval"] = interval
            return [_bar("2024-01-01T14:00:00Z")]

        monkeypatch.setattr(stream, "fetch_intraday_bars", _yf)

        added, source = stream.backfill_bars("AAPL", "k", "s", "iex", state, "5Min")

        assert added == 1
        assert source == "yfinance (delayed)"
        assert seen["interval"] == "5m"


def test_floor_ts_emits_alpaca_z_format():
    # Must match REST bar timestamps so buckets dedupe across sources.
    assert stream._floor_ts("2024-01-01T14:03:27.5Z", 5) == "2024-01-01T14:00:00Z"
