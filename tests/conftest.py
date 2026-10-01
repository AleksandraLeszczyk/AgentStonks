"""Suite-wide guards.

The important one: no test may reach a real Alpaca trading account.

That is not hypothetical. `simlab/runner.py` calls `load_dotenv()` at import
time, so the moment any test imports it the developer's real `.env` — live
market-data keys, and a paper trading key that Alpaca will happily authenticate
— is in `os.environ` for the rest of the session. Tests that read credentials
from the environment then quietly stop being hermetic: they pass or fail
depending on whose machine they run on, and they authenticate against a real
brokerage account to do it.

Stripping the variables for every test kills the whole class of problem at the
root. A test that wants credentials sets them itself with `monkeypatch.setenv`,
which makes the dependency visible in the test that has it, and keeps the value
fake.
"""
import pytest

# Every variable that could name a tradable Alpaca account, including the
# market-data pair, which `trading_mode` accepts as the paper fallback.
_ALPACA_ENV_VARS = (
    "ALPACA_API_KEY",
    "ALPACA_SECRET",
    "ALPACA_PAPER_API_KEY",
    "ALPACA_PAPER_SECRET",
    "ALPACA_LIVE_API_KEY",
    "ALPACA_LIVE_SECRET",
    "ALPACA_ENABLE_LIVE_TRADING",
)


@pytest.fixture(autouse=True)
def _no_real_alpaca_credentials(monkeypatch):
    """Remove every Alpaca credential from the environment for each test."""
    for var in _ALPACA_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(autouse=True)
def _no_live_bar_tiers(monkeypatch):
    """Stub the network tiers `bar_history.fetch_live_bars` adds on top of the
    resolved feed -- the yfinance regular-session window and the young IEX
    window -- so a test that fakes `fetch_bars` does not also download real
    yfinance bars. Tests of those tiers patch them back in themselves."""
    from agent_stonks import bar_history

    monkeypatch.setattr(bar_history, "_yfinance_window", lambda *a, **k: [])
    monkeypatch.setattr(bar_history, "_fetch_recent_bars", lambda *a, **k: [])


@pytest.fixture(autouse=True)
def _session_files_in_tmp(monkeypatch, tmp_path):
    """Keep `session_store`'s day files out of the real data/sessions/, and
    start every test with no state owning a day and no run resumed."""
    from agent_stonks import session_store

    monkeypatch.setattr(session_store, "SESSION_DIR", tmp_path / "sessions")
    monkeypatch.setattr(session_store, "_owners", {})
    monkeypatch.setattr(session_store, "_resumed", set())


@pytest.fixture(autouse=True)
def _net_gamma_in_tmp(monkeypatch, tmp_path):
    """Keep the live chart's kept net gamma values out of the real
    data/net_gamma/, and start every test with none kept in memory."""
    from agent_stonks import gamma_history

    monkeypatch.setattr(gamma_history, "CACHE_DIR", tmp_path / "net_gamma")
    monkeypatch.setattr(gamma_history, "_kept", {})


@pytest.fixture(autouse=True)
def _last_setup_in_tmp(monkeypatch, tmp_path):
    """Keep the dashboard's remembered setup out of the real
    data/last_setup.json."""
    from agent_stonks import last_setup

    monkeypatch.setattr(last_setup, "SETUP_PATH", tmp_path / "last_setup.json")
