from marketview import stream
from marketview.state import AppState


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
    state = AppState()
    state.bars_connected = False
    bars = [{"t": "2024-01-01T14:00:00Z", "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 100}]

    monkeypatch.setattr(stream, "fetch_bars", lambda *a, **k: bars)
    monkeypatch.setattr(stream, "fetch_trades", lambda *a, **k: [{"p": 1.6}])

    stream._fallback_bars_loop("AAPL", "k", "s", "iex", state, "1Min", _StopAfter(1))

    assert list(state.bars) == bars
    assert state.last_price == 1.6
    assert state.day_high == 2
    assert state.day_low == 0.5
    assert state.day_volume == 100
    assert "Fallback" in state.status
    assert "Alpaca REST" in state.status


def test_fallback_bars_loop_refreshes_quote_when_disconnected(monkeypatch):
    state = AppState()
    state.bars_connected = False
    bars = [{"t": "2024-01-01T14:00:00Z", "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 100}]

    monkeypatch.setattr(stream, "fetch_bars", lambda *a, **k: bars)
    monkeypatch.setattr(stream, "fetch_trades", lambda *a, **k: [{"p": 1.6}])
    monkeypatch.setattr(
        stream, "fetch_latest_quote", lambda *a, **k: {"bp": 1.55, "bs": 10, "ap": 1.57, "as": 20}
    )

    stream._fallback_bars_loop("AAPL", "k", "s", "iex", state, "1Min", _StopAfter(1))

    assert state.bid_price == 1.55
    assert state.bid_size == 10
    assert state.ask_price == 1.57
    assert state.ask_size == 20


def test_fallback_bars_loop_keeps_last_quote_when_quote_fetch_fails(monkeypatch):
    state = AppState()
    state.bars_connected = False
    state.bid_price = 1.5
    state.ask_price = 1.6
    bars = [{"t": "2024-01-01T14:00:00Z", "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 100}]

    monkeypatch.setattr(stream, "fetch_bars", lambda *a, **k: bars)
    monkeypatch.setattr(stream, "fetch_trades", lambda *a, **k: [{"p": 1.6}])

    def _raise(*a, **k):
        raise RuntimeError("quotes endpoint down")

    monkeypatch.setattr(stream, "fetch_latest_quote", _raise)

    stream._fallback_bars_loop("AAPL", "k", "s", "iex", state, "1Min", _StopAfter(1))

    assert state.bid_price == 1.5
    assert state.ask_price == 1.6


def test_fallback_bars_loop_skips_polling_when_stream_connected(monkeypatch):
    state = AppState()
    state.bars_connected = True
    state.status = "✅ Streaming AAPL (IEX)"
    called = []
    monkeypatch.setattr(stream, "fetch_bars", lambda *a, **k: called.append(1))

    stream._fallback_bars_loop("AAPL", "k", "s", "iex", state, "1Min", _StopAfter(2))

    assert called == []
    assert state.status == "✅ Streaming AAPL (IEX)"


def test_fallback_bars_loop_falls_back_to_yfinance_when_rest_fails(monkeypatch):
    state = AppState()
    state.bars_connected = False
    bars = [{"t": "2024-01-01T14:00:00Z", "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 100}]

    def _raise(*a, **k):
        raise RuntimeError("Alpaca down")

    monkeypatch.setattr(stream, "fetch_bars", _raise)
    monkeypatch.setattr(stream, "fetch_intraday_bars", lambda *a, **k: bars)

    stream._fallback_bars_loop("AAPL", "k", "s", "iex", state, "1Min", _StopAfter(1))

    assert list(state.bars) == bars
    assert "yfinance" in state.status


def test_fallback_news_loop_appends_new_articles_when_disconnected(monkeypatch):
    state = AppState()
    state.news_connected = False
    state.news = [{"id": "1", "headline": "old"}]
    fresh = [{"id": "1", "headline": "old"}, {"id": "2", "headline": "new"}]

    monkeypatch.setattr(stream, "fetch_news_with_fallback", lambda *a, **k: fresh)

    stream._fallback_news_loop("AAPL", "k", "s", "wn-key", state, _StopAfter(1))

    assert [a["id"] for a in state.news] == ["1", "2"]
    assert state.agent_wake_event.is_set()
    assert "Fallback" in state.news_status


def test_fallback_news_loop_skips_polling_when_stream_connected(monkeypatch):
    state = AppState()
    state.news_connected = True
    called = []
    monkeypatch.setattr(stream, "fetch_news_with_fallback", lambda *a, **k: called.append(1))

    stream._fallback_news_loop("AAPL", "k", "s", "wn-key", state, _StopAfter(2))

    assert called == []


def test_fire_due_alerts_wakes_on_price_field():
    import time
    state = AppState()
    state.last_price = 151.0
    state.recent_prices.append((time.monotonic(), 151.0))
    state.alerts = [{"field": "last_price", "condition": "above", "value": 150.0}]

    stream._fire_due_alerts(state)

    assert state.alerts == []  # cleared once fired
    assert state.agent_wake_event.is_set()
    assert "last_price above 150" in state.agent_wake_reason


def test_fire_due_alerts_wakes_on_non_price_field():
    """Alerts on continuously-updated, non-price fields (here cumulative day volume)
    fire too -- not just price levels."""
    state = AppState()
    state.day_volume = 6_000_000
    state.alerts = [{"field": "day_volume", "condition": "above", "value": 5_000_000}]

    stream._fire_due_alerts(state)

    assert state.alerts == []
    assert state.agent_wake_event.is_set()


def test_fire_due_alerts_does_not_wake_when_unmet():
    import time
    state = AppState()
    state.last_price = 149.0
    state.recent_prices.append((time.monotonic(), 149.0))
    state.alerts = [{"field": "last_price", "condition": "above", "value": 150.0}]

    stream._fire_due_alerts(state)

    assert state.alerts == [{"field": "last_price", "condition": "above", "value": 150.0}]
    assert not state.agent_wake_event.is_set()


def test_fire_due_alerts_wakes_on_prior_price_in_window():
    """A last_price alert fires if any price within the last minute crossed the
    threshold, even if the current last_price has since reverted below it."""
    import time
    state = AppState()
    now = time.monotonic()
    # Spike to 152 happened 30 s ago; current price is back at 148.
    state.recent_prices.append((now - 30, 152.0))
    state.recent_prices.append((now, 148.0))
    state.last_price = 148.0
    state.alerts = [{"field": "last_price", "condition": "above", "value": 150.0}]

    stream._fire_due_alerts(state)

    assert state.alerts == []
    assert state.agent_wake_event.is_set()


def test_fire_due_alerts_ignores_stale_price_outside_window():
    """Prices older than 60 s do not satisfy a last_price alert."""
    import time
    state = AppState()
    now = time.monotonic()
    # Spike happened 90 s ago -- outside the 60 s window.
    state.recent_prices.append((now - 90, 155.0))
    state.recent_prices.append((now, 148.0))
    state.last_price = 148.0
    state.alerts = [{"field": "last_price", "condition": "above", "value": 150.0}]

    stream._fire_due_alerts(state)

    assert state.alerts == [{"field": "last_price", "condition": "above", "value": 150.0}]
    assert not state.agent_wake_event.is_set()


def test_fallback_bars_loop_fires_volume_alert(monkeypatch):
    """A day_volume alert fires from the REST fallback path, not just the live stream."""
    state = AppState()
    state.bars_connected = False
    state.alerts = [{"field": "day_volume", "condition": "above", "value": 50.0}]
    bars = [{"t": "2024-01-01T14:00:00Z", "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 100}]

    monkeypatch.setattr(stream, "fetch_bars", lambda *a, **k: bars)
    monkeypatch.setattr(stream, "fetch_trades", lambda *a, **k: [{"p": 1.6}])

    stream._fallback_bars_loop("AAPL", "k", "s", "iex", state, "1Min", _StopAfter(1))

    assert state.alerts == []
    assert state.agent_wake_event.is_set()
