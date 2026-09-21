import itertools
import logging
import threading
import time

import pytest
from websocket import WebSocketBadStatusException

from agent_stonks import ws_reconnect
from agent_stonks.ws_reconnect import ReconnectingSocket, rate_limit_wait


def _refused_429(reset: object = None) -> WebSocketBadStatusException:
    headers = {"x-ratelimit-limit": "5", "x-ratelimit-remaining": "0"}
    if reset is not None:
        headers["x-ratelimit-reset"] = str(reset)
    return WebSocketBadStatusException(
        "Handshake status 429 Too Many Requests", 429, resp_headers=headers
    )


class TestRateLimitWait:
    def test_waits_until_just_past_the_servers_reset_time(self):
        assert rate_limit_wait(_refused_429(reset=1_000_004), now=1_000_000.0) == 5.0

    def test_a_reset_already_past_still_waits_a_second(self):
        assert rate_limit_wait(_refused_429(reset=999_990), now=1_000_000.0) == 1.0

    def test_no_reset_header_falls_back_to_a_long_wait(self):
        assert rate_limit_wait(_refused_429()) == ws_reconnect.RATE_LIMIT_FALLBACK_SEC

    def test_anything_but_a_429_is_not_rate_limiting(self):
        assert rate_limit_wait(None) is None
        assert rate_limit_wait(ConnectionResetError("lost")) is None
        assert rate_limit_wait(
            WebSocketBadStatusException("Handshake status 502", 502, resp_headers={})
        ) is None


class _ScriptedWS:
    """A WebSocketApp whose every run_forever plays the next scripted outcome:
    "open" connects then drops, an exception is a refused handshake."""

    def __init__(self, url, **handlers):
        self.handlers = handlers
        self.script: list = []
        self.runs = 0

    def run_forever(self, **kwargs):
        assert kwargs["reconnect"] == 0  # the library's own loop stays off
        self.runs += 1
        outcome = self.script.pop(0) if self.script else "open"
        if outcome == "open":
            self.handlers["on_open"](self)
        else:
            self.handlers["on_error"](self, outcome)
        if self.handlers.get("on_close"):
            self.handlers["on_close"](self)

    def send(self, payload):
        pass

    def close(self):
        pass


@pytest.fixture
def scripted(monkeypatch):
    monkeypatch.setattr(ws_reconnect.websocket, "WebSocketApp", _ScriptedWS)
    monkeypatch.setattr(ws_reconnect, "_jittered", lambda s: s)


def _run(sock: ReconnectingSocket, attempts: int) -> list[float]:
    """Run the loop for `attempts` connection attempts, returning the wait that
    followed each one."""
    waits: list[float] = []

    def fake_wait(seconds):
        waits.append(seconds)
        if len(waits) >= attempts:
            sock._stop.set()
        return sock._stop.is_set()

    sock._stop.wait = fake_wait
    sock.run_forever()
    return waits


class TestReconnectLoop:
    def test_every_connection_runs_on_open(self, scripted):
        opened: list = []
        sock = ReconnectingSocket(
            "test", "wss://x", on_open=opened.append, on_message=lambda *a: None
        )

        _run(sock, attempts=3)

        assert len(opened) == 3

    def test_the_wait_doubles_up_to_the_cap(self, scripted):
        sock = ReconnectingSocket("test", "wss://x", on_open=lambda ws: None,
                                  on_message=lambda *a: None)
        sock.ws.script = [ConnectionResetError("lost")] * 10

        waits = _run(sock, attempts=8)

        assert waits == [5.0, 10.0, 20.0, 40.0, 80.0, 120.0, 120.0, 120.0]

    def test_a_429_waits_for_the_reset_time_when_that_is_longer(self, scripted, monkeypatch):
        monkeypatch.setattr(ws_reconnect.time, "time", lambda: 1_000_000.0)
        sock = ReconnectingSocket("test", "wss://x", on_open=lambda ws: None,
                                  on_message=lambda *a: None)
        sock.ws.script = [_refused_429(reset=1_000_030), _refused_429(reset=1_000_002)]

        waits = _run(sock, attempts=2)

        assert waits == [31.0, 10.0]

    def test_a_connection_that_stayed_up_resets_the_wait(self, scripted, monkeypatch):
        clock = itertools.chain([0.0], itertools.repeat(1000.0))
        monkeypatch.setattr(ws_reconnect.time, "monotonic", lambda: next(clock))
        sock = ReconnectingSocket("test", "wss://x", on_open=lambda ws: None,
                                  on_message=lambda *a: None)
        # A refused attempt grows the wait, then a long-lived connection resets it.
        sock.ws.script = [ConnectionResetError("lost"), "open"]

        waits = _run(sock, attempts=2)

        assert waits == [5.0, 5.0]

    def test_close_ends_the_loop_while_it_is_waiting(self, scripted, monkeypatch):
        monkeypatch.setattr(ws_reconnect, "RECONNECT_BASE_SEC", 60.0)
        sock = ReconnectingSocket("test", "wss://x", on_open=lambda ws: None,
                                  on_message=lambda *a: None)
        thread = threading.Thread(target=sock.run_forever, daemon=True)
        thread.start()
        time.sleep(0.1)

        sock.close()
        thread.join(timeout=2)

        assert not thread.is_alive()
        assert sock.ws.runs == 1

    def test_logs_name_the_socket(self, scripted, caplog):
        sock = ReconnectingSocket("Finnhub stream", "wss://x", on_open=lambda ws: None,
                                  on_message=lambda *a: None)
        sock.ws.script = [_refused_429(reset=0)]

        with caplog.at_level(logging.INFO, logger="agent_stonks.ws_reconnect"):
            _run(sock, attempts=2)

        messages = [r.getMessage() for r in caplog.records]
        assert any(m.startswith("Finnhub stream: handshake refused, HTTP 429") for m in messages)
        assert "Finnhub stream: connected" in messages

    def test_the_librarys_goodbye_line_is_dropped(self, caplog):
        with caplog.at_level(logging.INFO, logger="websocket"):
            logging.getLogger("websocket").error("Connection to remote host was lost. - goodbye")
            logging.getLogger("websocket").info("Websocket connected")

        assert [r.getMessage() for r in caplog.records] == ["Websocket connected"]


class TestAgainstARealServer:
    """End to end over a local socket with the real websocket-client, because
    the loop reuses one WebSocketApp across run_forever calls."""

    @pytest.fixture
    def server(self):
        from websockets.sync.server import serve

        received: list[str] = []
        refuse = {"left": 0}

        def handler(conn):
            # Take the subscribe frame, then drop the connection like a network blip.
            received.append(conn.recv(timeout=5))
            conn.close()

        def process_request(conn, request):
            if refuse["left"]:
                refuse["left"] -= 1
                response = conn.respond(429, '{"error":"API limit reached."}')
                response.headers["x-ratelimit-reset"] = str(int(time.time()))
                return response
            return None

        with serve(handler, "127.0.0.1", 0, process_request=process_request) as srv:
            threading.Thread(target=srv.serve_forever, daemon=True).start()
            port = srv.socket.getsockname()[1]
            yield f"ws://127.0.0.1:{port}", received, refuse
            srv.shutdown()

    def _wait_for(self, predicate, timeout=10.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(0.02)
        return False

    def test_resubscribes_after_every_drop_and_rides_out_a_429(self, server, monkeypatch):
        url, received, refuse = server
        monkeypatch.setattr(ws_reconnect, "RECONNECT_BASE_SEC", 0.05)
        monkeypatch.setattr(ws_reconnect, "RECONNECT_MAX_SEC", 0.2)
        refuse["left"] = 1
        errors: list = []

        sock = ReconnectingSocket(
            "test",
            url,
            on_open=lambda ws: ws.send('{"type":"subscribe","symbol":"AAPL"}'),
            on_message=lambda *a: None,
            on_error=lambda ws, err: errors.append(err),
        )
        thread = threading.Thread(target=sock.run_forever, daemon=True)
        thread.start()
        try:
            assert self._wait_for(lambda: len(received) >= 3)
        finally:
            sock.close()
            thread.join(timeout=5)

        assert set(received) == {'{"type":"subscribe","symbol":"AAPL"}'}
        assert any(getattr(e, "status_code", None) == 429 for e in errors)
        assert not thread.is_alive()
