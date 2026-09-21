"""The reconnect loop every live WebSocket runs in.

websocket-client can reconnect by itself (`run_forever(reconnect=N)`), and all
three sockets used to rely on that. It has three faults, which together caused
the Finnhub `429 Too Many Requests` handshake storm seen on 2026-09-18:

* **A reconnected socket never re-authenticates or re-subscribes.** After a
  reconnect the library calls `on_reconnect`, not `on_open`, and nothing set
  `on_reconnect`. So after the first network blip the Finnhub socket was up but
  subscribed to nothing, and Alpaca's sockets never sent `auth`, got closed for
  it, and reconnected again every few seconds for good.
* **It retries at a fixed interval whatever went wrong.** A handshake refused
  with 429 was retried 5 s later, back into the same limit, so the limit never
  recovered. Finnhub allows 5 handshakes per window, separate from the
  60-calls-a-minute REST limit, and reports when the window resets in
  `x-ratelimit-reset`.
* **Its log lines do not say which socket they are about.** The bars, news and
  Finnhub sockets all log as `[websocket]`, which is why that log could not tell
  Alpaca from Finnhub.

`ReconnectingSocket` runs `run_forever()` with the library's reconnect off, in a
loop of its own. Every attempt is a fresh connection, so `on_open` runs every
time. The wait doubles from `RECONNECT_BASE_SEC` up to `RECONNECT_MAX_SEC`, with
jitter so several sockets that dropped together do not retry together. It goes
back to the base wait once a connection has stayed up for
`RECONNECT_STABLE_SEC`. After a 429 it waits at least until the server's reset
time. `close()` ends the loop, including when it is waiting between attempts.
"""
from __future__ import annotations

import logging
import random
import threading
import time
from typing import Any, Callable

import websocket

from .stream_common import keepalive_sockopt

logger = logging.getLogger(__name__)

RECONNECT_BASE_SEC = 5.0
RECONNECT_MAX_SEC = 120.0
# How long a connection has to stay up before it counts as healthy and the next
# drop starts again from RECONNECT_BASE_SEC rather than the grown wait.
RECONNECT_STABLE_SEC = 60.0
# How long to wait after a 429 when the server gave no parsable reset time.
RATE_LIMIT_FALLBACK_SEC = 60.0

PING_INTERVAL_SEC = 20
PING_TIMEOUT_SEC = 10


class _DropLibraryGoodbye(logging.Filter):
    """With the library's reconnect off, it logs every disconnect as
    `ERROR ... - goodbye`, which reads as if the stream had ended for good.
    `ReconnectingSocket` logs what actually happens next, with the socket's
    name, so the library's line is dropped."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not record.getMessage().endswith(" - goodbye")


logging.getLogger("websocket").addFilter(_DropLibraryGoodbye())


def rate_limit_wait(err: object, now: float | None = None) -> float | None:
    """Seconds to wait before the next handshake if `err` is a 429 refusal,
    else None.

    websocket-client raises `WebSocketBadStatusException` for a refused
    handshake and keeps the response headers, lower-cased, on it.
    `x-ratelimit-reset` is an epoch second, so one second is added to land
    safely past it.
    """
    if getattr(err, "status_code", None) != 429:
        return None
    headers = getattr(err, "resp_headers", None) or {}
    reset = headers.get("x-ratelimit-reset") if hasattr(headers, "get") else None
    try:
        return max(0.0, float(reset) - (time.time() if now is None else now)) + 1.0
    except (TypeError, ValueError):
        return RATE_LIMIT_FALLBACK_SEC


def _jittered(seconds: float) -> float:
    return seconds * random.uniform(0.8, 1.2)


class ReconnectingSocket:
    """A `websocket.WebSocketApp` plus the reconnect loop described above.

    Construct it, keep it where `close()` can reach it (`app.ws` /
    `app.ws_news`), then call `run_forever()` on a background thread. The
    handlers have websocket-client's usual signatures.
    """

    def __init__(
        self,
        label: str,
        url: str,
        *,
        on_open: Callable[[Any], None],
        on_message: Callable[[Any, str], None],
        on_error: Callable[[Any, Exception], None] | None = None,
        on_close: Callable[..., None] | None = None,
    ) -> None:
        self.label = label
        self._on_open_cb = on_open
        self._on_error_cb = on_error
        self._stop = threading.Event()
        self._last_error: Exception | None = None
        self._connected_at: float | None = None
        self.ws = websocket.WebSocketApp(
            url,
            on_open=self._on_open,
            # Belt and braces: the library's own reconnect is off, so this never
            # fires, but if `reconnect=` is ever reintroduced the socket would
            # still re-authenticate and re-subscribe.
            on_reconnect=self._on_open,
            on_message=on_message,
            on_error=self._on_error,
            on_close=on_close,
        )

    @property
    def closed(self) -> bool:
        return self._stop.is_set()

    def _on_open(self, ws: Any) -> None:
        self._connected_at = time.monotonic()
        logger.info("%s: connected", self.label)
        self._on_open_cb(ws)

    def _on_error(self, ws: Any, err: Exception) -> None:
        self._last_error = err
        if self._on_error_cb is not None:
            self._on_error_cb(ws, err)

    def send(self, payload: str) -> bool:
        """Send on the live connection. Returns False, and sends nothing, when
        no connection is up. The next `on_open` sends the full state again."""
        try:
            self.ws.send(payload)
            return True
        except Exception:  # noqa: BLE001 - closed/closing socket; on_open resends
            return False

    def close(self) -> None:
        """Stop for good: close the connection and end `run_forever`."""
        self._stop.set()
        try:
            self.ws.close()
        except Exception:  # noqa: BLE001 - already closed
            pass

    def run_forever(self) -> None:
        """Connect, and reconnect with backoff, until `close()`. Blocks."""
        delay = RECONNECT_BASE_SEC
        while not self._stop.is_set():
            self._last_error = None
            self._connected_at = None
            try:
                self.ws.run_forever(
                    ping_interval=PING_INTERVAL_SEC,
                    ping_timeout=PING_TIMEOUT_SEC,
                    reconnect=0,
                    sockopt=keepalive_sockopt(),
                )
            except Exception as exc:  # noqa: BLE001 - never let the loop die
                self._last_error = exc
            if self._stop.is_set():
                break

            if (
                self._connected_at is not None
                and time.monotonic() - self._connected_at >= RECONNECT_STABLE_SEC
            ):
                delay = RECONNECT_BASE_SEC
            wait = _jittered(delay)
            limited = rate_limit_wait(self._last_error)
            if limited is not None:
                wait = max(wait, limited)
                logger.warning(
                    "%s: handshake refused, HTTP 429 (rate limit); reconnecting in %.0fs",
                    self.label, wait,
                )
            else:
                logger.info(
                    "%s: disconnected (%s); reconnecting in %.0fs",
                    self.label, self._last_error or "closed by server", wait,
                )
            delay = min(delay * 2, RECONNECT_MAX_SEC)
            if self._stop.wait(wait):
                break
        logger.info("%s: stopped", self.label)
