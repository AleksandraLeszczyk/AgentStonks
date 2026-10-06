from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from agent_stonks.options import (
    _bs_gamma,
    _select_expiry,
    fetch_option_chain,
    gamma_flip,
    net_gamma_exposure,
)


def _future_date(days: int) -> str:
    return (datetime.now(timezone.utc).date() + timedelta(days=days)).isoformat()


class FakeTicker:
    def __init__(self, options, calls_df, puts_df, spot):
        self.options = options
        self._calls_df = calls_df
        self._puts_df = puts_df
        self._spot = spot

    @property
    def fast_info(self):
        return {"lastPrice": self._spot}

    def option_chain(self, expiry):
        return SimpleNamespace(calls=self._calls_df, puts=self._puts_df)

    def history(self, period="1d"):
        return pd.DataFrame({"Close": [self._spot]})


def _chain_frames():
    calls = pd.DataFrame(
        {"strike": [95.0, 100.0, 105.0], "openInterest": [50, 100, 800], "impliedVolatility": [0.3, 0.3, 0.3]}
    )
    puts = pd.DataFrame(
        {"strike": [95.0, 100.0, 105.0], "openInterest": [700, 100, 30], "impliedVolatility": [0.3, 0.3, 0.3]}
    )
    return calls, puts


class TestSelectExpiry:
    def test_prefers_expiry_within_window(self):
        expirations = [_future_date(5), _future_date(60)]
        assert _select_expiry(expirations, max_dte=45) == expirations[0]

    def test_falls_back_to_nearest_when_all_beyond_window(self):
        expirations = [_future_date(90), _future_date(60)]
        assert _select_expiry(expirations, max_dte=45) == expirations[1]

    def test_raises_on_no_expirations(self):
        with pytest.raises(ValueError):
            _select_expiry([])


class TestBsGamma:
    def test_zero_for_zero_volatility(self):
        assert _bs_gamma(100.0, 100.0, 0.1, 0.0) == 0.0

    def test_zero_for_zero_time(self):
        assert _bs_gamma(100.0, 100.0, 0.0, 0.3) == 0.0

    def test_positive_for_normal_inputs(self):
        assert _bs_gamma(100.0, 100.0, 0.25, 0.3) > 0.0


class TestFetchOptionChain:
    def test_builds_strikes_oi_and_gamma_exposure(self, monkeypatch):
        calls, puts = _chain_frames()
        expiry = _future_date(30)
        ticker = FakeTicker([expiry], calls, puts, spot=100.0)
        monkeypatch.setattr("agent_stonks.options.yf.Ticker", lambda symbol: ticker)

        data = fetch_option_chain("AAPL")

        assert data["expiry"] == expiry
        assert data["spot"] == 100.0
        assert data["strikes"] == [95.0, 100.0, 105.0]
        assert data["calls_oi"] == [50.0, 100.0, 800.0]
        assert data["puts_oi"] == [700.0, 100.0, 30.0]
        # Calls contribute positive gamma exposure, puts negative.
        assert all(v >= 0 for v in data["calls_gamma_exposure"])
        assert all(v <= 0 for v in data["puts_gamma_exposure"])

    def test_uses_explicit_spot_over_fast_info(self, monkeypatch):
        calls, puts = _chain_frames()
        expiry = _future_date(30)
        ticker = FakeTicker([expiry], calls, puts, spot=100.0)
        monkeypatch.setattr("agent_stonks.options.yf.Ticker", lambda symbol: ticker)

        data = fetch_option_chain("AAPL", spot=123.0)
        assert data["spot"] == 123.0

    def test_retries_once_when_yahoo_returns_no_contracts(self, monkeypatch):
        calls, puts = _chain_frames()
        ticker = FakeTicker([_future_date(30)], calls, puts, spot=100.0)
        answers = [SimpleNamespace(calls=None, puts=None), SimpleNamespace(calls=calls, puts=puts)]
        ticker.option_chain = lambda expiry: answers.pop(0)
        monkeypatch.setattr("agent_stonks.options.yf.Ticker", lambda symbol: ticker)

        data = fetch_option_chain("AAPL")
        assert data["strikes"] == [95.0, 100.0, 105.0]

    def test_no_contracts_twice_raises_a_readable_error(self, monkeypatch):
        expiry = _future_date(30)
        ticker = FakeTicker([expiry], None, None, spot=100.0)
        monkeypatch.setattr("agent_stonks.options.yf.Ticker", lambda symbol: ticker)

        with pytest.raises(ValueError, match=f"no contracts for expiry {expiry}"):
            fetch_option_chain("AAPL")


class TestNetGammaExposure:
    def _chain(self, monkeypatch):
        calls, puts = _chain_frames()
        ticker = FakeTicker([_future_date(30)], calls, puts, spot=100.0)
        monkeypatch.setattr("agent_stonks.options.yf.Ticker", lambda symbol: ticker)
        return fetch_option_chain("AAPL")

    def test_at_the_fetch_spot_it_is_the_chains_own_total(self, monkeypatch):
        data = self._chain(monkeypatch)
        total = sum(data["calls_gamma_exposure"]) + sum(data["puts_gamma_exposure"])
        assert net_gamma_exposure(data, [100.0])[0] == pytest.approx(total)

    def test_repriced_at_other_spots_matches_a_fresh_fetch_there(self, monkeypatch):
        data = self._chain(monkeypatch)
        spots = [97.0, 103.5]
        values = net_gamma_exposure(data, spots)
        for spot, value in zip(spots, values):
            there = fetch_option_chain("AAPL", spot=spot)
            total = sum(there["calls_gamma_exposure"]) + sum(there["puts_gamma_exposure"])
            assert value == pytest.approx(total)

    def test_a_chain_without_ivs_gives_none(self, monkeypatch):
        data = self._chain(monkeypatch)
        del data["calls_iv"]
        assert net_gamma_exposure(data, [100.0]) is None


def _flip_chain(strikes, calls_oi, puts_oi, spot=100.0):
    """A chain as `fetch_option_chain` shapes it, five days from expiry."""
    n = len(strikes)
    return {
        "strikes": strikes, "calls_oi": calls_oi, "puts_oi": puts_oi,
        "calls_iv": [0.3] * n, "puts_iv": [0.3] * n, "t_years": 5 / 365, "spot": spot,
    }


class TestGammaFlip:
    # Puts under the price, calls over it: dealers short gamma below, long above.
    ONE_FLIP = _flip_chain([95.0, 100.0, 105.0], [0, 0, 800], [800, 0, 0])

    def test_where_net_gamma_changes_sign(self):
        flip = gamma_flip(self.ONE_FLIP)
        assert 99.0 < flip < 101.0
        below, above = net_gamma_exposure(self.ONE_FLIP, [flip - 0.01, flip + 0.01])
        assert below < 0 < above

    def test_nearest_the_given_price_when_there_are_two(self):
        chain = _flip_chain([90.0, 100.0, 110.0], [0, 800, 0], [800, 0, 800])
        low, high = gamma_flip(chain, near=93.0), gamma_flip(chain, near=108.0)
        assert 93.0 < low < 97.0
        assert 103.0 < high < 107.0
        # The chain's own spot when no price is given.
        assert gamma_flip(chain) == pytest.approx(high, abs=0.01)

    def test_none_when_the_sign_never_changes(self):
        calls_only = _flip_chain([95.0, 100.0, 105.0], [100, 100, 100], [0, 0, 0])
        assert gamma_flip(calls_only) is None

    def test_none_when_the_flip_is_outside_the_window(self):
        # The flip near 100 is more than 25% (log) below 200.
        assert gamma_flip(self.ONE_FLIP, near=200.0) is None

    def test_none_without_ivs_or_strikes(self):
        no_ivs = dict(self.ONE_FLIP)
        del no_ivs["calls_iv"]
        assert gamma_flip(no_ivs) is None
        assert gamma_flip({}) is None
        assert gamma_flip(None) is None
