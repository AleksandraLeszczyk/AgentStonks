import faulthandler
import logging
import signal

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

# run_app.py answers a hang with SIGUSR1 before it restarts the app: every
# thread's stack goes to stderr, and so into data/logs/, saying where it was
# stuck. Re-registering on every rerun is harmless.
if hasattr(signal, "SIGUSR1"):
    faulthandler.register(signal.SIGUSR1, all_threads=True)

from agent_stonks.ui import build_ui

build_ui()
