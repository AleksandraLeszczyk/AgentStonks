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
