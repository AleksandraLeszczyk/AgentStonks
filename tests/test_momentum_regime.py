"""The momentum/regime pipeline mirrored from mshift.

These pin the definitions Apple Trader 2's momentum signals are computed with.
The numbers themselves were checked bar-for-bar against mshift's own output on
the notebooks' cached AAPL history; what is pinned here is the behaviour that
check would catch changing.
"""

import numpy as np
import pandas as pd

from agent_stonks import momentum_regime as mr

PARAMS = mr.MOMENTUM_DEFAULTS


def _bars(day: str, closes, start: str = "09:30", seed: int = 0) -> list[dict]:
    """Minute bars in the {"t","o","h","l","c","v"} shape the streams deliver,
    each with a real body, a wick and its own volume."""
    rng = np.random.default_rng(seed)
    closes = np.asarray(closes, dtype=float)
    opens = np.concatenate([closes[:1], closes[:-1]])
    wick = closes * 3e-4
    highs = np.maximum(opens, closes) + wick
    lows = np.minimum(opens, closes) - wick
    volumes = 1000.0 + rng.integers(0, 500, size=len(closes))
    idx = pd.date_range(
        f"{day} {start}", periods=len(closes), freq="1min", tz="America/New_York"
    ).tz_convert("UTC")
    return [
        {"t": ts.strftime("%Y-%m-%dT%H:%M:%SZ"), "o": o, "h": h, "l": lo, "c": c, "v": v}
        for ts, o, h, lo, c, v in zip(idx, opens, highs, lows, closes, volumes)
    ]


def _flat_then_trend(n_flat: int, n_trend: int, step: float, start: float = 100.0, seed: int = 0):
    """Closes that oscillate with no net drift, then trend at `step` per minute.

    The flat half alternates rather than wandering: a random walk normalised by
    its own realised volatility crosses the +-0.90 entry band by chance often
    enough to make "did a regime change happen" a coin flip in a test.
    """
    rng = np.random.default_rng(seed)
    flat = np.resize([1e-4, -1e-4], n_flat) if n_flat else np.zeros(0)
    trend = np.full(n_trend, step) + rng.normal(0, 5e-5, size=n_trend)
    return start * np.exp(np.cumsum(np.concatenate([flat, trend])))


def _session(day: str = "2026-07-21", n_flat: int = 90, n_trend: int = 90, step: float = 3e-4):
    return mr.frame_from_bars(_bars(day, _flat_then_trend(n_flat, n_trend, step)))


class TestFrame:
    def test_keeps_only_the_regular_session_and_stamps_the_bookkeeping(self):
        frame = mr.frame_from_bars(_bars("2026-07-21", [100.0] * 180, start="08:00"))
        assert frame.index.min().strftime("%H:%M") == "09:30"
        assert frame.index.max().strftime("%H:%M") == "10:59"
        assert list(frame.columns) == [
            "open", "high", "low", "close", "volume",
            "session", "bar_of_day", "minutes_from_open",
        ]
        assert frame["bar_of_day"].iloc[0] == 0
        assert frame["minutes_from_open"].iloc[0] == 0.0
        assert frame["minutes_from_open"].iloc[-1] == 89.0

    def test_the_live_frame_is_todays_session_only(self, monkeypatch):
        """Nothing in the pipeline crosses the overnight gap, so yesterday's
        bars would only slow it down -- and they are dropped."""
        import threading
        from types import SimpleNamespace

        monkeypatch.setattr(mr.clock, "now", lambda: pd.Timestamp("2026-07-21 15:00", tz="UTC"))
        sym_state = SimpleNamespace(
            symbol="AAPL",
            lock=threading.Lock(),
            bars=_bars("2026-07-20", [100.0] * 390) + _bars("2026-07-21", [111.0] * 30),
        )
        frame = mr.minute_frame(sym_state)
        assert len(frame) == 30
        assert frame["close"].eq(111.0).all()
        assert frame["bar_of_day"].iloc[-1] == 29


class TestMomentum:
    def test_momentum_is_drift_in_units_of_its_own_random_walk_scale(self):
        frame = mr.compute_momentum(_session(), PARAMS)
        # The trailing return needs `horizon` bars and sigma needs 10, so the
        # first bars carry no momentum rather than a fabricated one.
        assert frame["mom_raw"].iloc[:10].isna().all()
        quiet, trending = frame["mom"].iloc[80], frame["mom"].iloc[-1]
        assert abs(quiet) < 0.9  # noise alone stays inside the entry band
        assert trending > 3.0  # a steady 4 bps/min drift is many sigmas of it

    def test_momentum_does_not_reach_across_the_overnight_gap(self):
        frame = mr.frame_from_bars(
            _bars("2026-07-20", [100.0] * 60) + _bars("2026-07-21", [130.0] * 60)
        )
        scored = mr.compute_momentum(frame, PARAMS)
        day2 = scored[scored["session"] == scored["session"].iloc[-1]]
        # The 30% overnight jump is invisible: day 2 opens with no return at all
        # and its own trailing windows have to fill from scratch.
        assert pd.isna(day2["ret_1"].iloc[0])
        assert day2["mom_raw"].iloc[:10].isna().all()


class TestRegimes:
    def test_the_schmitt_trigger_enters_high_and_leaves_low(self):
        """Entering a directional regime takes |mom| > 0.90; leaving it only
        needs a fall back below 0.40. The gap is what stops a score hovering on
        the line from emitting a burst of fake changes."""
        mom = np.array([0.0, 0.5, 0.89, 0.91, 0.60, 0.41, 0.39, -0.5, -0.95, -0.5, -0.39])
        starts = np.zeros(len(mom), dtype=bool)
        regimes = mr._hysteresis_regimes(mom, starts, enter=0.90, exit_=0.40)
        assert list(regimes) == [0, 0, 0, 1, 1, 1, 0, 0, -1, -1, 0]

    def test_each_session_starts_flat(self):
        mom = np.array([2.0, 2.0, 2.0, 2.0])
        starts = np.array([False, False, True, False])
        assert list(mr._hysteresis_regimes(mom, starts, 0.90, 0.40)) == [1, 1, 1, 1]
        # ...and a session that opens below the entry band starts balanced even
        # though the previous session ended in a regime.
        mom = np.array([2.0, 2.0, 0.5, 0.5])
        assert list(mr._hysteresis_regimes(mom, starts, 0.90, 0.40)) == [1, 1, 0, 0]

    def test_a_trend_produces_one_change_into_positive(self):
        scored = mr.add_momentum_regimes(_session(), PARAMS)
        changes = scored[scored["regime_change"]]
        to_positive = changes[changes["regime"] == 1]
        assert len(to_positive) == 1
        # It is exactly the bar where the regime column first turns positive.
        assert to_positive.index[0] == scored.index[scored["regime"] == 1][0]
        assert to_positive["prev_regime"].iloc[0] == 0

    def test_pre_dwell_is_how_long_the_old_regime_had_held(self):
        scored = mr.add_momentum_regimes(_session(), PARAMS)
        change = scored[scored["regime_change"]].iloc[0]
        pos = scored.index.get_loc(scored[scored["regime_change"]].index[0])
        assert change["pre_dwell"] == pos  # the balanced run started at bar 0
        assert scored["pre_dwell"][~scored["regime_change"]].isna().all()
