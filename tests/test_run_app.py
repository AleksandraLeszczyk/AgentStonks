"""run_app.py: the supervisor that restarts the app when it dies or hangs.

Driven against a stand-in server -- a few lines of http.server answering the
health route -- that crashes, hangs or behaves as each test tells it, run by
run, with every timing shrunk so a whole restart takes a fraction of a second.
"""
import socket
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import run_app  # noqa: E402

FAKE_SERVER = r'''
import faulthandler, os, signal, sys, time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

port = int(sys.argv[sys.argv.index("--server.port") + 1])
runs = Path(os.environ["FAKE_RUNS"])
with runs.open("a") as f:
    f.write(" ".join(sys.argv[1:]) + "\n")
run_no = len(runs.read_text().splitlines())
plan = os.environ["FAKE_PLAN"].split(",")      # one per run, the last repeats
mode = plan[min(run_no, len(plan)) - 1]
faulthandler.register(signal.SIGUSR1, all_threads=True)
print(f"fake run {run_no}: {mode}", flush=True)
if mode == "crash":
    sys.exit(3)
if mode == "silent":                           # up, but never listening
    time.sleep(600)
answered = 0

class Health(BaseHTTPRequestHandler):
    def do_GET(self):
        global answered
        answered += 1
        if mode == "hang" and answered > 2:
            time.sleep(60)
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *args):
        pass

HTTPServer(("127.0.0.1", port), Health).serve_forever()
'''


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class Harness:
    def __init__(self, tmp_path: Path, plan: str, **overrides):
        self.port = _free_port()
        script = tmp_path / "fake_server.py"
        script.write_text(FAKE_SERVER)
        self.runs = tmp_path / "runs.txt"
        self.lines: list[str] = []
        self.notices: list[str] = []
        cmd = [sys.executable, str(script), "--server.port", str(self.port)]
        options = dict(
            restart_cmd=[*cmd, "--restarted"],
            env={"FAKE_RUNS": str(self.runs), "FAKE_PLAN": plan, "PATH": ""},
            check_every=0.05, check_timeout=0.3, fail_after=3, startup_grace=10.0,
            max_restarts=3, restart_window=60.0, backoff=(0.01, 0.05),
            stop_timeout=3.0, notify=self.notices.append,
        )
        options.update(overrides)
        self.supervisor = run_app.Supervisor(
            cmd, f"http://127.0.0.1:{self.port}/_stcore/health", self.lines.append, **options
        )
        self.result: "int | None" = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        self.result = self.supervisor.run()

    def start(self):
        self.thread.start()
        return self

    def wait_for(self, text: str, count: int = 1, timeout: float = 20.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if sum(text in line for line in list(self.lines)) >= count:
                return
            time.sleep(0.02)
        raise AssertionError(f"never saw {text!r} x{count}:\n" + "\n".join(self.lines))

    def stop(self) -> int:
        self.supervisor.stop()
        self.thread.join(20)
        assert not self.thread.is_alive()
        return self.result

    def run_args(self) -> "list[str]":
        return self.runs.read_text().splitlines()


@pytest.fixture
def harness(tmp_path):
    made: list[Harness] = []

    def make(plan: str, **overrides) -> Harness:
        made.append(Harness(tmp_path, plan, **overrides))
        return made[-1]

    yield make
    for h in made:
        h.supervisor.stop()
        h.thread.join(20)
        proc = h.supervisor.proc
        if proc is not None and proc.poll() is None:
            proc.kill()


def test_a_crash_is_restarted(harness):
    h = harness("crash,ok").start()
    h.wait_for("the app is up and answering")

    assert h.stop() == 0
    assert any("the app exited (code 3)" in line for line in h.lines)
    assert any("fake run 1: crash" in line for line in h.lines)  # the app's own output
    assert len(h.run_args()) == 2
    assert h.notices and "restarted" in h.notices[0]


def test_a_restart_does_not_open_another_browser_tab(harness):
    h = harness("crash,ok").start()
    h.wait_for("the app is up and answering")
    h.stop()

    first, second = h.run_args()
    assert "--restarted" not in first and "--restarted" in second


def test_a_hang_is_dumped_stopped_and_restarted(harness):
    h = harness("hang,ok").start()
    h.wait_for("the app stopped answering")
    h.wait_for("the app is up and answering", count=2)

    assert h.stop() == 0
    assert any("asking the hung app for a stack dump" in line for line in h.lines)
    # faulthandler's dump of the hung server, copied from its stderr.
    assert any("most recent call first" in line for line in h.lines)
    assert len(h.run_args()) == 2


def test_a_crash_loop_is_given_up_on(harness):
    h = harness("crash").start()
    h.thread.join(20)

    assert h.result == 1
    assert len(h.run_args()) == 4  # the first start and max_restarts=3 more
    assert any("giving up" in line for line in h.lines)
    assert "giving up" in h.notices[-1]


def test_a_server_that_never_answers_is_restarted(harness):
    h = harness("silent,ok", startup_grace=0.3).start()
    h.wait_for("the app is up and answering")

    assert h.stop() == 0
    assert any("the app never answered" in line for line in h.lines)
    assert len(h.run_args()) == 2


def test_stopping_stops_the_app(harness):
    h = harness("ok").start()
    h.wait_for("the app is up and answering")
    proc = h.supervisor.proc

    assert h.stop() == 0
    assert proc.poll() is not None
    assert any("stopping the app" in line for line in h.lines)


def test_streamlit_options_are_read_both_ways():
    args = ["--server.port", "8533", "--server.address=0.0.0.0"]
    assert run_app._option(args, "--server.port") == "8533"
    assert run_app._option(args, "--server.address") == "0.0.0.0"
    assert run_app._option(args, "--server.headless") is None


def test_it_refuses_a_port_already_taken(capsys):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        port = sock.getsockname()[1]
        assert run_app.main(["--server.port", str(port)]) == 1
    assert "already in use" in capsys.readouterr().err
