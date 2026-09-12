import json

from agent_stonks import finnhub_stream, stream
from agent_stonks.state import AppState


def _app(*symbols: str):
    """AppState streaming `symbols` (default AAPL) plus its first SymbolState."""
    app = AppState()
    app.set_symbols(list(symbols) or ["AAPL"])
    return app, app.sym(app.symbols[0])


def _builder(state, tf_minutes: int = 1) -> finnhub_stream.CandleBuilder:
    return finnhub_stream.CandleBuilder(state, tf_minutes)


class TestNormalizeTrade:
    def test_swaps_finnhubs_symbol_and_size_into_the_alpaca_spelling(self):
        # Finnhub's "s" is the symbol and "v" the size; Alpaca's "s" is the size.
        trade = finnhub_stream.normalize_trade(
            {"s": "AAPL", "p": 226.1, "t": 1704117600123, "v": 100, "c": ["12"]}
        )
        assert trade == {
            "S": "AAPL",
            "p": 226.1,
            "s": 100.0,
            "t": "2024-01-01T14:00:00.123Z",
            "c": ["12"],
        }

    def test_converts_epoch_millis_to_the_alpaca_z_format(self):
        trade = finnhub_stream.normalize_trade({"s": "AAPL", "p": 1.0, "t": 1704117661500})
        assert trade["t"] == "2024-01-01T14:01:01.500Z"

    def test_drops_a_record_without_a_price(self):
        assert finnhub_stream.normalize_trade({"s": "AAPL", "t": 1704117600123}) is None

    def test_drops_a_record_without_a_symbol(self):
        assert finnhub_stream.normalize_trade({"p": 226.1}) is None

    def test_omits_absent_optional_fields(self):
        assert finnhub_stream.normalize_trade({"s": "AAPL", "p": 226.1}) == {
            "S": "AAPL", "p": 226.1
        }


class TestCandleBuilder:
    def test_first_trade_opens_a_bar_at_its_floored_bucket(self):
        _, state = _app()
        builder = _builder(state)

        assert builder.add_trade(10.0, 50.0, "2024-01-01T14:00:31Z") is None

        assert list(state.bars) == [
            {"t": "2024-01-01T14:00:00Z", "o": 10.0, "h": 10.0, "l": 10.0,
             "c": 10.0, "v": 50.0, "vw": 10.0, "n": 1}
        ]

    def test_later_trades_in_the_same_minute_extend_the_open_bar(self):
        _, state = _app()
        builder = _builder(state)

        builder.add_trade(10.0, 50.0, "2024-01-01T14:00:01Z")
        builder.add_trade(12.0, 50.0, "2024-01-01T14:00:20Z")
        assert builder.add_trade(9.0, 100.0, "2024-01-01T14:00:59Z") is None

        (bar,) = list(state.bars)
        assert (bar["o"], bar["h"], bar["l"], bar["c"]) == (10.0, 12.0, 9.0, 9.0)
        assert bar["v"] == 200.0
        assert bar["n"] == 3
        # vw is the size-weighted mean: (10*50 + 12*50 + 9*100) / 200
        assert bar["vw"] == (10.0 * 50 + 12.0 * 50 + 9.0 * 100) / 200

    def test_a_trade_in_the_next_minute_opens_a_bar_and_returns_the_closed_one(self):
        _, state = _app()
        builder = _builder(state)
        builder.add_trade(10.0, 50.0, "2024-01-01T14:00:01Z")

        closed = builder.add_trade(11.0, 20.0, "2024-01-01T14:01:03Z")

        assert closed is not None
        assert closed["t"] == "2024-01-01T14:00:00Z"
        assert closed["c"] == 10.0
        assert [b["t"] for b in state.bars] == [
            "2024-01-01T14:00:00Z", "2024-01-01T14:01:00Z"
        ]
        assert state.bars[-1]["o"] == 11.0

    def test_buckets_by_the_chosen_timeframe(self):
        _, state = _app()
        builder = _builder(state, tf_minutes=5)

        builder.add_trade(10.0, 1.0, "2024-01-01T14:02:00Z")
        builder.add_trade(11.0, 1.0, "2024-01-01T14:04:59Z")
        builder.add_trade(12.0, 1.0, "2024-01-01T14:05:01Z")

        assert [b["t"] for b in state.bars] == [
            "2024-01-01T14:00:00Z", "2024-01-01T14:05:00Z"
        ]

    def test_an_out_of_order_trade_never_rewrites_a_settled_bar(self):
        _, state = _app()
        builder = _builder(state)
        builder.add_trade(10.0, 50.0, "2024-01-01T14:00:01Z")
        builder.add_trade(11.0, 20.0, "2024-01-01T14:01:03Z")

        # A late print stamped back in the already-closed 14:00 minute.
        assert builder.add_trade(99.0, 10.0, "2024-01-01T14:00:59Z") is None

        assert [b["t"] for b in state.bars] == [
            "2024-01-01T14:00:00Z", "2024-01-01T14:01:00Z"
        ]
        assert state.bars[0]["h"] == 10.0  # untouched by the 99.0 print
        assert state.bars[1]["h"] == 11.0

    def test_does_not_duplicate_a_bucket_the_seeded_history_already_holds(self):
        # state.bars starts out full of backfilled Alpaca REST bars; the first
        # Finnhub trade may still be stamped inside the newest of them.
        _, state = _app()
        state.bars.append(
            {"t": "2024-01-01T14:00:00Z", "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 100}
        )
        builder = _builder(state)

        assert builder.add_trade(9.0, 10.0, "2024-01-01T14:00:30Z") is None

        assert len(state.bars) == 1
        assert state.bars[0]["c"] == 1.5

    def test_appends_after_the_seeded_history_without_reporting_a_close(self):
        _, state = _app()
        state.bars.append(
            {"t": "2024-01-01T14:00:00Z", "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 100}
        )
        builder = _builder(state)

        # The backfilled bar is already closed and already published, so opening
        # the next bucket must not re-publish it.
        assert builder.add_trade(9.0, 10.0, "2024-01-01T14:01:30Z") is None
        assert [b["t"] for b in state.bars] == [
            "2024-01-01T14:00:00Z", "2024-01-01T14:01:00Z"
        ]


class TestCloseIfElapsed:
    def _ts(self, iso: str) -> float:
        from agent_stonks import clock

        return clock.parse_iso_strict(iso).timestamp()

    def test_closes_the_bar_once_its_minute_has_ended(self):
        _, state = _app()
        builder = _builder(state)
        builder.add_trade(10.0, 50.0, "2024-01-01T14:00:30Z")

        closed = builder.close_if_elapsed(now=self._ts("2024-01-01T14:01:01Z"))

        assert closed is not None and closed["t"] == "2024-01-01T14:00:00Z"
        assert builder.open_bucket is None

    def test_leaves_a_bar_whose_minute_is_still_running(self):
        _, state = _app()
        builder = _builder(state)
        builder.add_trade(10.0, 50.0, "2024-01-01T14:00:30Z")

        assert builder.close_if_elapsed(now=self._ts("2024-01-01T14:00:45Z")) is None
        assert builder.open_bucket == "2024-01-01T14:00:00Z"

    def test_closes_only_once_so_a_repeating_timer_does_not_republish(self):
        _, state = _app()
        builder = _builder(state)
        builder.add_trade(10.0, 50.0, "2024-01-01T14:00:30Z")
        later = self._ts("2024-01-01T14:05:00Z")

        assert builder.close_if_elapsed(now=later) is not None
        assert builder.close_if_elapsed(now=later) is None

    def test_never_closes_a_bar_it_did_not_build(self):
        # Backfilled REST bars are already closed; the timer must ignore them.
        _, state = _app()
        state.bars.append(
            {"t": "2024-01-01T14:00:00Z", "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 100}
        )
        builder = _builder(state)

        assert builder.close_if_elapsed(now=self._ts("2024-01-01T18:00:00Z")) is None

    def test_an_elapsed_bar_does_not_get_a_replacement_bucket(self):
        _, state = _app()
        builder = _builder(state)
        builder.add_trade(10.0, 50.0, "2024-01-01T14:00:30Z")

        builder.close_if_elapsed(now=self._ts("2024-01-01T14:09:00Z"))

        assert [b["t"] for b in state.bars] == ["2024-01-01T14:00:00Z"]


class TestApplyTrade:
    def test_updates_the_live_scalars(self):
        app, state = _app()
        state.day_volume = 1000.0
        builder = _builder(state)

        finnhub_stream.apply_trade(
            state, builder, {"S": "AAPL", "p": 10.0, "s": 50.0, "t": "2024-01-01T14:00:30Z"}
        )

        assert state.last_price == 10.0
        assert state.day_volume == 1050.0
        assert state.trades == [{"p": 10.0, "s": 50.0, "t": "2024-01-01T14:00:30Z"}]
        assert list(state.recent_prices)[-1][1] == 10.0

    def test_publishes_previous_minute_fields_only_when_a_bar_closes(self):
        app, state = _app()
        builder = _builder(state)

        finnhub_stream.apply_trade(
            state, builder, {"S": "AAPL", "p": 10.0, "s": 1.0, "t": "2024-01-01T14:00:10Z"}
        )
        finnhub_stream.apply_trade(
            state, builder, {"S": "AAPL", "p": 12.0, "s": 1.0, "t": "2024-01-01T14:00:50Z"}
        )
        # The in-progress bar must not move the "last completed bar" fields.
        assert state.previous_minute_close is None

        finnhub_stream.apply_trade(
            state, builder, {"S": "AAPL", "p": 11.0, "s": 1.0, "t": "2024-01-01T14:01:05Z"}
        )

        assert state.previous_minute_high == 12.0
        assert state.previous_minute_low == 10.0
        assert state.previous_minute_close == 12.0

    def test_fires_a_pending_price_alert(self):
        app, state = _app()
        state.alerts = [
            {"symbol": "AAPL", "field": "last_price", "condition": "above", "value": 9.0}
        ]
        builder = _builder(state)

        finnhub_stream.apply_trade(
            state, builder, {"S": "AAPL", "p": 10.0, "s": 1.0, "t": "2024-01-01T14:00:10Z"}
        )

        assert app.agent_wake_event.is_set()
        assert state.alerts == []

    def test_fires_the_high_volume_alert(self):
        app, state = _app()
        app.volume_alert_enabled = True
        app.volume_alert_multiplier = 1.5
        state.daily_bars = [{"t": "2023-12-29T05:00:00Z", "v": 100}]
        state.day_volume = 140.0
        builder = _builder(state)

        finnhub_stream.apply_trade(
            state, builder, {"S": "AAPL", "p": 10.0, "s": 20.0, "t": "2024-01-01T14:00:10Z"}
        )

        assert state.volume_alert_triggered
        assert app.agent_wake_event.is_set()

    def test_a_trade_without_a_size_still_moves_the_price(self):
        app, state = _app()
        state.day_volume = 500.0
        builder = _builder(state)

        finnhub_stream.apply_trade(
            state, builder, {"S": "AAPL", "p": 10.0, "t": "2024-01-01T14:00:10Z"}
        )

        assert state.last_price == 10.0
        assert state.day_volume == 500.0


class TestSocketMessages:
    def _stream(self, app, symbols, monkeypatch, timeframe="1Min"):
        """Run start_stream against a fake WebSocketApp, returning its handlers."""
        captured: dict = {}

        class FakeWS:
            def __init__(self, url, **handlers):
                captured["url"] = url
                captured.update(handlers)
                self.sent: list[str] = []

            def send(self, payload):
                self.sent.append(payload)

            def run_forever(self, **_):
                captured["ran"] = True

        monkeypatch.setattr(finnhub_stream.websocket, "WebSocketApp", FakeWS)
        builders = {s: _builder(app.sym(s)) for s in symbols}
        finnhub_stream.start_stream(symbols, "tok", app, timeframe, builders)
        return captured, app.ws

    def test_authenticates_in_the_url_and_subscribes_per_symbol(self, monkeypatch):
        app, _ = _app("AAPL", "TSLA")
        captured, ws = self._stream(app, ["AAPL", "TSLA"], monkeypatch)

        captured["on_open"](ws)

        assert captured["url"] == "wss://ws.finnhub.io?token=tok"
        assert [json.loads(m) for m in ws.sent] == [
            {"type": "subscribe", "symbol": "AAPL"},
            {"type": "subscribe", "symbol": "TSLA"},
        ]
        assert app.bars_connected is True

    def test_a_trade_message_lands_in_the_right_symbols_bars(self, monkeypatch):
        app, _ = _app("AAPL", "TSLA")
        captured, ws = self._stream(app, ["AAPL", "TSLA"], monkeypatch)

        captured["on_message"](
            ws,
            json.dumps(
                {"type": "trade", "data": [
                    {"s": "TSLA", "p": 250.5, "t": 1704117600000, "v": 30},
                    {"s": "AAPL", "p": 190.0, "t": 1704117600000, "v": 10},
                ]},
            ),
        )

        assert app.sym("TSLA").last_price == 250.5
        assert app.sym("AAPL").last_price == 190.0
        assert app.sym("TSLA").bars[-1]["v"] == 30.0

    def test_ignores_a_trade_for_an_unsubscribed_symbol(self, monkeypatch):
        app, state = _app("AAPL")
        captured, ws = self._stream(app, ["AAPL"], monkeypatch)

        captured["on_message"](
            ws,
            json.dumps({"type": "trade", "data": [{"s": "NVDA", "p": 1.0, "v": 1}]}),
        )

        assert state.last_price is None

    def test_ping_and_malformed_frames_are_no_ops(self, monkeypatch):
        app, state = _app()
        captured, ws = self._stream(app, ["AAPL"], monkeypatch)
        app.bars_connected = True

        captured["on_message"](ws, json.dumps({"type": "ping"}))
        captured["on_message"](ws, "not json")
        captured["on_message"](ws, json.dumps(["unexpected list"]))

        assert app.bars_connected is True
        assert state.last_price is None

    def test_an_error_frame_marks_the_stream_disconnected(self, monkeypatch):
        app, _ = _app()
        captured, ws = self._stream(app, ["AAPL"], monkeypatch)
        app.bars_connected = True

        captured["on_message"](ws, json.dumps({"type": "error", "msg": "Invalid token"}))

        assert app.bars_connected is False
        assert "Invalid token" in app.status

    def test_close_marks_the_stream_disconnected(self, monkeypatch):
        app, _ = _app()
        captured, ws = self._stream(app, ["AAPL"], monkeypatch)
        captured["on_open"](ws)

        captured["on_close"](ws)

        assert app.bars_connected is False
        assert app.status == "Stream closed"


class TestSourceSelection:
    def test_finnhub_is_the_default(self):
        assert stream.resolve_data_source("finnhub", "tok") == "finnhub"
        assert AppState().data_source == "finnhub"

    def test_falls_back_to_alpaca_without_a_token(self):
        assert stream.resolve_data_source("finnhub", "") == "alpaca"

    def test_an_explicit_alpaca_choice_is_honoured(self):
        assert stream.resolve_data_source("alpaca", "tok") == "alpaca"

    def test_launch_stream_starts_the_finnhub_socket(self, monkeypatch):
        app, _ = _app()
        launched: dict = {}
        monkeypatch.setattr(
            stream.finnhub_stream, "launch",
            lambda symbols, token, a, tf, ev: launched.update(
                symbols=symbols, token=token, timeframe=tf
            ),
        )
        monkeypatch.setattr(stream.threading, "Thread", lambda **k: type(
            "T", (), {"start": lambda self: None}
        )())

        stream.launch_stream(
            ["AAPL"], "k", "s", "iex", app, "1Min",
            data_source="finnhub", finnhub_token="tok",
        )

        assert launched == {"symbols": ["AAPL"], "token": "tok", "timeframe": "1Min"}
        assert app.data_source == "finnhub"

    def test_launch_stream_starts_the_alpaca_socket_without_a_token(self, monkeypatch):
        app, _ = _app()
        started: list = []
        monkeypatch.setattr(
            stream.finnhub_stream, "launch",
            lambda *a, **k: started.append("finnhub"),
        )

        class FakeThread:
            def __init__(self, target=None, args=(), **k):
                started.append(getattr(target, "__name__", str(target)))

            def start(self):
                pass

        monkeypatch.setattr(stream.threading, "Thread", FakeThread)

        stream.launch_stream(
            ["AAPL"], "k", "s", "iex", app, "1Min", data_source="finnhub", finnhub_token=""
        )

        assert "finnhub" not in started
        assert "_start_stream" in started
        assert app.data_source == "alpaca"


class TestQuotePollUnderFinnhub:
    def test_refreshes_bid_ask_while_the_finnhub_socket_is_connected(self, monkeypatch):
        from tests.test_stream import _StopAfter

        app, state = _app()
        app.bars_connected = True
        monkeypatch.setattr(
            stream, "fetch_latest_quote",
            lambda *a, **k: {"bp": 1.55, "bs": 10, "ap": 1.57, "as": 20},
        )
        monkeypatch.setattr(stream, "_backfill_all_quietly", lambda *a, **k: None)

        stream._fallback_bars_loop(
            ["AAPL"], "k", "s", "iex", app, "1Min", _StopAfter(1), "finnhub"
        )

        assert state.bid_price == 1.55
        assert state.ask_price == 1.57

    def test_does_not_poll_quotes_on_the_alpaca_source(self, monkeypatch):
        from tests.test_stream import _StopAfter

        app, state = _app()
        app.bars_connected = True
        calls: list = []
        monkeypatch.setattr(
            stream, "fetch_latest_quote", lambda *a, **k: calls.append(a) or {}
        )
        monkeypatch.setattr(stream, "_backfill_all_quietly", lambda *a, **k: None)

        stream._fallback_bars_loop(
            ["AAPL"], "k", "s", "iex", app, "1Min", _StopAfter(1), "alpaca"
        )

        assert calls == []
        assert state.bid_price is None


class TestBufferOwnership:
    """The bar buffer is shared: the REST backfill merges into it and re-sorts,
    and the stream-down fallback replaces it wholesale. The builder must notice
    when the bar it was filling is no longer the one at the end."""

    def test_starts_a_fresh_bar_when_the_backfill_displaced_its_own(self):
        from agent_stonks import stream

        _, state = _app()
        builder = _builder(state)
        builder.add_trade(10.0, 5.0, "2024-01-01T14:01:30Z")

        # A backfill lands a bar for a *later* minute, so ours is no longer last.
        stream.merge_missing_bars(
            state, [{"t": "2024-01-01T14:02:00Z", "o": 1, "h": 1, "l": 1, "c": 1, "v": 9}]
        )
        builder.add_trade(11.0, 5.0, "2024-01-01T14:01:40Z")

        # The displaced 14:01 bar keeps the volume it had; nothing wrote into
        # the backfilled 14:02 bar.
        assert [b["t"] for b in state.bars] == [
            "2024-01-01T14:01:00Z", "2024-01-01T14:02:00Z"
        ]
        assert state.bars[0]["v"] == 5.0
        assert state.bars[1]["v"] == 9

    def test_does_not_close_a_bar_the_fallback_poll_wiped(self):
        _, state = _app()
        builder = _builder(state)
        builder.add_trade(10.0, 5.0, "2024-01-01T14:01:30Z")

        # _poll_symbol_via_rest replaces the whole buffer while the WS is down.
        state.bars.clear()
        state.bars.append(
            {"t": "2024-01-01T14:03:00Z", "o": 2, "h": 2, "l": 2, "c": 2, "v": 7}
        )

        from agent_stonks import clock

        elapsed = clock.parse_iso_strict("2024-01-01T14:09:00Z").timestamp()
        assert builder.close_if_elapsed(now=elapsed) is None
        assert state.bars[-1]["c"] == 2
