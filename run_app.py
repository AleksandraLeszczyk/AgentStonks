"""Run the live app under a supervisor that brings it back when it goes down.

    python run_app.py                         # main.py on port 8501
    python run_app.py --server.port 8533      # streamlit options pass through
    python run_app.py sim_main.py             # another script

``streamlit run main.py`` on its own has no one watching it: when the server
dies or stops answering, the browser shows "Connection error" and nothing
happens until it is restarted by hand. This does that restart.

What it watches for
-------------------
* **The process ending** -- a crash, a segfault in a native library, the OS
  killing it. Any exit the supervisor did not ask for is restarted.
* **The process hanging** -- alive, but no longer answering. ``/_stcore/health``
  is polled every CHECK_EVERY_SEC; after FAIL_AFTER failures in a row the
  server is asked for every thread's stack (SIGUSR1, registered in main.py),
  so the log says where it was stuck, then stopped (SIGTERM, then SIGKILL)
  and started again.

Restarts back off (2 s, 4 s, ... 60 s) and give up after MAX_RESTARTS within
RESTART_WINDOW_SEC: a crash loop is reported, not hammered.

Everything the app prints is also written to ``data/logs/<script>.log``
(rotated), with the supervisor's own lines among it, so the cause of a restart
can be read afterwards -- the terminal's scrollback was all there was before.

What the app does once it is back is the app's business: the open browser tab
reconnects by itself, and the first session restores today's ledger and starts
again the stream and agent that were running (`ui._resume_after_restart`).

Ctrl-C (or closing the terminal) stops the app and the supervisor together.
"""
from __future__ import annotations

import logging
import logging.handlers
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "data" / "logs"
LOG_MAX_BYTES = 10 * 1024 * 1024
LOG_BACKUPS = 5

CHECK_EVERY_SEC = 10.0
CHECK_TIMEOUT_SEC = 5.0
# Six missed checks: a minute without an answer. A busy server still answers --
# the health route runs on the event loop, not the script threads.
FAIL_AFTER = 6
# Importing the models (torch, LightGBM) makes the first answer slow.
STARTUP_GRACE_SEC = 180.0
MAX_RESTARTS = 5
RESTART_WINDOW_SEC = 15 * 60.0
BACKOFF_MIN_SEC = 2.0
BACKOFF_MAX_SEC = 60.0
STOP_TIMEOUT_SEC = 15.0


def _option(args: "list[str]", name: str) -> "str | None":
    """The value of a streamlit `--name value` / `--name=value` option."""
    for i, arg in enumerate(args):
        if arg == name and i + 1 < len(args):
            return args[i + 1]
        if arg.startswith(name + "="):
            return arg.split("=", 1)[1]
    return None


def _notify(message: str) -> None:
    """A desktop notification on macOS; nothing elsewhere. Best effort."""
    if sys.platform != "darwin":
        return
    text = message.replace("\\", "").replace('"', "'")
    try:
        subprocess.run(
            ["osascript", "-e", f'display notification "{text}" with title "Agent Stonks"'],
            timeout=5, check=False, capture_output=True,
        )
    except (OSError, subprocess.SubprocessError):
        pass


def _port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        return sock.connect_ex((host, port)) == 0


class Supervisor:
    """Start `cmd`, watch it, and start it again when it dies or hangs.

    `restart_cmd` is what a restart runs instead, when it differs (no browser
    tab opened on a restart). `log` receives every line, the app's and the
    supervisor's. Timings are parameters so tests can run the whole cycle in
    a second."""

    def __init__(
        self,
        cmd: "list[str]",
        health_url: str,
        log,
        *,
        restart_cmd: "list[str] | None" = None,
        env: "dict | None" = None,
        check_every: float = CHECK_EVERY_SEC,
        check_timeout: float = CHECK_TIMEOUT_SEC,
        fail_after: int = FAIL_AFTER,
        startup_grace: float = STARTUP_GRACE_SEC,
        max_restarts: int = MAX_RESTARTS,
        restart_window: float = RESTART_WINDOW_SEC,
        backoff: "tuple[float, float]" = (BACKOFF_MIN_SEC, BACKOFF_MAX_SEC),
        stop_timeout: float = STOP_TIMEOUT_SEC,
        notify=_notify,
    ) -> None:
        self.cmd = cmd
        self.restart_cmd = restart_cmd or cmd
        self.health_url = health_url
        self.log = log
        self.env = env
        self.check_every = check_every
        self.check_timeout = check_timeout
        self.fail_after = fail_after
        self.startup_grace = startup_grace
        self.max_restarts = max_restarts
        self.restart_window = restart_window
        self.backoff = backoff
        self.stop_timeout = stop_timeout
        self.notify = notify
        self.proc: "subprocess.Popen | None" = None
        self.starts = 0
        self.restarts: list[float] = []
        self._stop = threading.Event()
        self._reader: "threading.Thread | None" = None

    def say(self, message: str) -> None:
        self.log(f"{datetime.now():%Y-%m-%d %H:%M:%S} SUPERVISOR {message}")

    def stop(self) -> None:
        """Ask `run` to stop the app and return (any thread, or a signal)."""
        self._stop.set()

    # --- the child ------------------------------------------------------

    def _start(self) -> None:
        cmd = self.cmd if self.starts == 0 else self.restart_cmd
        self.starts += 1
        self.proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=self.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            # Its own process group: a terminal Ctrl-C reaches the supervisor
            # only, which then stops the app the way it chooses.
            start_new_session=True,
        )
        self.say(f"started the app (pid {self.proc.pid}): {' '.join(cmd)}")
        self._reader = threading.Thread(
            target=self._copy_output, args=(self.proc,), daemon=True
        )
        self._reader.start()

    def _copy_output(self, proc: subprocess.Popen) -> None:
        for raw in iter(proc.stdout.readline, b""):
            self.log(raw.decode("utf-8", errors="replace").rstrip("\n"))
        proc.stdout.close()

    def _healthy(self) -> bool:
        try:
            with urllib.request.urlopen(self.health_url, timeout=self.check_timeout) as resp:
                return resp.status == 200
        except Exception:
            return False

    def _end(self, *, hung: bool) -> None:
        """Stop the app: a stack dump first when it hung, then SIGTERM, then
        SIGKILL if it does not go."""
        proc = self.proc
        if proc is None or proc.poll() is not None:
            return
        if hung and hasattr(signal, "SIGUSR1"):
            self.say("asking the hung app for a stack dump of every thread")
            try:
                proc.send_signal(signal.SIGUSR1)
                time.sleep(min(2.0, self.stop_timeout))
            except OSError:
                pass
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(self.stop_timeout)
            except subprocess.TimeoutExpired:
                self.say(f"the app did not stop in {self.stop_timeout:g} s; killing it")
                proc.kill()
                proc.wait()
        self._join_reader()

    def _join_reader(self) -> None:
        if self._reader is not None:
            self._reader.join(timeout=5)

    # --- the loop ---------------------------------------------------------

    def _watch(self) -> str:
        """Wait until the app needs a restart; say why. "stop" when asked to."""
        started = time.monotonic()
        answered = False
        failures = 0
        while not self._stop.wait(self.check_every):
            code = self.proc.poll()
            if code is not None:
                self._join_reader()
                return f"the app exited (code {code})"
            if self._healthy():
                if not answered:
                    self.say("the app is up and answering")
                elif failures:
                    self.say("the app is answering again")
                answered, failures = True, 0
                continue
            if not answered:
                if time.monotonic() - started > self.startup_grace:
                    return f"the app never answered in {self.startup_grace:g} s"
                continue
            failures += 1
            self.say(f"health check failed ({failures}/{self.fail_after})")
            if failures >= self.fail_after:
                return "hung"
        return "stop"

    def run(self) -> int:
        """Supervise until stopped (0) or the app keeps failing (1)."""
        while True:
            self._start()
            why = self._watch()
            if why == "stop":
                self.say("stopping the app")
                self._end(hung=False)
                return 0
            if why == "hung":
                why = (
                    f"the app stopped answering ({self.fail_after} health checks "
                    f"in a row, {self.check_every:g} s apart)"
                )
                self.say(why)
                self._end(hung=True)
            else:
                self.say(why)
                self._end(hung=False)
            now = time.monotonic()
            self.restarts = [t for t in self.restarts if now - t < self.restart_window]
            self.restarts.append(now)
            if len(self.restarts) > self.max_restarts:
                message = (
                    f"giving up: {len(self.restarts)} restarts in "
                    f"{self.restart_window / 60:g} min. Last: {why}. See {LOG_DIR}."
                )
                self.say(message)
                self.notify(f"Agent Stonks is down — {message}")
                return 1
            low, high = self.backoff
            delay = min(high, low * 2 ** (len(self.restarts) - 1))
            self.say(f"restarting in {delay:g} s (restart {len(self.restarts)})")
            self.notify(f"Agent Stonks restarted: {why}")
            if self._stop.wait(delay):
                return 0


def _file_log(script: str):
    """A writer of one line to the terminal and to data/logs/<script>.log."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        LOG_DIR / f"{Path(script).stem}.log",
        maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUPS, encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    file_log = logging.getLogger("run_app.file")
    file_log.propagate = False
    file_log.setLevel(logging.INFO)
    file_log.addHandler(handler)
    lock = threading.Lock()

    def log(line: str) -> None:
        with lock:
            print(line, flush=True)
            file_log.info(line)

    return log


def main(argv: "list[str]") -> int:
    script = "main.py"
    if argv and argv[0].endswith(".py"):
        script, argv = argv[0], argv[1:]
    port = int(_option(argv, "--server.port") or 8501)
    address = _option(argv, "--server.address") or "127.0.0.1"
    host = "127.0.0.1" if address in ("0.0.0.0", "::", "localhost") else address
    base = (_option(argv, "--server.baseUrlPath") or "").strip("/")
    health_url = f"http://{host}:{port}/{base + '/' if base else ''}_stcore/health"

    if _port_in_use(host, port):
        print(
            f"Port {port} is already in use — is the app already running? Stop it "
            "first, or pass --server.port with a free one.",
            file=sys.stderr,
        )
        return 1

    cmd = [sys.executable, "-m", "streamlit", "run", script, *argv]
    # A restart must not open another browser tab: the open one reconnects.
    restart_cmd = cmd if _option(argv, "--server.headless") else [
        *cmd, "--server.headless", "true",
    ]
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        # A fatal crash (segfault, abort) dumps every thread's stack too.
        "PYTHONFAULTHANDLER": "1",
    }
    supervisor = Supervisor(cmd, health_url, _file_log(script), restart_cmd=restart_cmd, env=env)

    def on_signal(signum, _frame) -> None:
        supervisor.stop()

    for name in ("SIGINT", "SIGTERM", "SIGHUP"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), on_signal)
    supervisor.say(
        f"supervising {script} on port {port}; logging to {LOG_DIR / (Path(script).stem + '.log')}"
    )
    return supervisor.run()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
