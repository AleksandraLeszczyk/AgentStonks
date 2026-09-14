"""The intraday momentum score and its regime, computed from the tape.

A session-local momentum score and a Schmitt-trigger regime over it, defined
the way TimeToChange2's `mshift/momentum.py` defines them. No model is
involved: every number here is a function of the bars alone, so it means the
same thing on any symbol and needs nothing installed beyond pandas.

momentum   the `horizon`-bar log return divided by its own random-walk scale
           (sigma * sqrt(horizon)), then EWM-smoothed -- a dimensionless
           "sigmas of drift" score, comparable across quiet and busy parts of
           the day. Computed inside a trading session only.
regime     a Schmitt trigger over that score: ENTER a directional regime at
           |mom| > `enter_threshold`, leave it only once |mom| falls back below
           `exit_threshold`. The hysteresis is what stops a score hovering near
           the line from emitting a burst of fake changes.

It also owns the live minute grid the rule agents and the chart overlays read
(`minute_frame`, `frame_from_bars`): today's regular-session bars with the
per-day bookkeeping everything else groups on.

Everything is session-local: nothing crosses the overnight gap, so the live
frame is today's streamed bars and nothing else. One consequence is worth
naming: the volatility floor (`sigma_floor`) is a per-session median, so live
it is the median over the bars seen *so far* rather than the whole day. It only
ever acts as a clip on a near-flat window, so it rarely binds at all.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import clock, market_hours, model_store

REGIME_NAME = {-1: "negative", 0: "balanced", 1: "positive"}

# The regular-session window the pipeline runs on; everything outside it is
# dropped before anything is computed.
RTH_START = model_store.RTH_START
RTH_END = model_store.RTH_END

# mshift.config.MomentumParams.
MOMENTUM_DEFAULTS = {
    "horizon": 15,
    "vol_window": 30,
    "smooth_span": 7,
    "enter_threshold": 0.90,
    "exit_threshold": 0.40,
    "warmup_bars": 30,
}


# --- momentum / regimes (mirrors mshift.momentum) ---------------------------

def compute_momentum(bars: pd.DataFrame, params: "dict | None" = None) -> pd.DataFrame:
    """Add `ret_1`, `sigma`, `mom_raw` and `mom`, per session.

    `mom` is the `horizon`-bar log return in units of its own random-walk
    standard deviation, EWM-smoothed: +1 means "price drifted up by about one
    sigma more than a coin flip would".
    """
    p = {**MOMENTUM_DEFAULTS, **(params or {})}
    out = bars.copy()
    logp = np.log(out["close"])
    log_grp = logp.groupby(out["session"], sort=False)

    out["ret_1"] = log_grp.diff()
    out["sigma"] = (
        out["ret_1"]
        .groupby(out["session"], sort=False)
        .transform(lambda s: s.rolling(p["vol_window"], min_periods=10).std())
    )
    # A dead-flat window would divide by ~0; floor sigma at a tiny fraction of
    # the session's own typical volatility.
    sigma_floor = out.groupby("session", sort=False)["sigma"].transform("median") * 0.10
    sigma = out["sigma"].clip(lower=sigma_floor).replace(0.0, np.nan)

    horizon_return = log_grp.transform(lambda s: s - s.shift(p["horizon"]))
    out["mom_raw"] = horizon_return / (sigma * np.sqrt(p["horizon"]))
    out["mom"] = (
        out["mom_raw"]
        .groupby(out["session"], sort=False)
        .transform(lambda s: s.ewm(span=p["smooth_span"], adjust=False).mean())
    )
    return out


def _hysteresis_regimes(
    mom: np.ndarray, session_start: np.ndarray, enter: float, exit_: float
) -> np.ndarray:
    """Schmitt-trigger classification of a momentum series.

    Entering a directional regime needs `|mom| > enter`; leaving it needs
    `|mom|` to fall back below `exit_`. Each session starts flat.
    """
    n = mom.shape[0]
    regimes = np.zeros(n, dtype=np.int8)
    state = 0
    for i in range(n):
        if session_start[i]:
            state = 0
        m = mom[i]
        if np.isnan(m):
            regimes[i] = 0
            state = 0
            continue
        if state == 1:
            if m < exit_:
                state = -1 if m < -enter else 0
        elif state == -1:
            if m > -exit_:
                state = 1 if m > enter else 0
        else:
            if m > enter:
                state = 1
            elif m < -enter:
                state = -1
        regimes[i] = state
    return regimes


def assign_regimes(bars: pd.DataFrame, params: "dict | None" = None) -> pd.DataFrame:
    """Add `regime`, `prev_regime`, `regime_change` and the dwell counters."""
    p = {**MOMENTUM_DEFAULTS, **(params or {})}
    out = bars.copy()

    out["regime"] = _hysteresis_regimes(
        out["mom"].to_numpy(dtype=float),
        out["bar_of_day"].to_numpy() == 0,
        p["enter_threshold"],
        p["exit_threshold"],
    )
    prev = out.groupby("session", sort=False)["regime"].shift(1)
    out["prev_regime"] = prev
    out["regime_change"] = (out["regime"] != prev) & prev.notna()

    # run_id increments on every change, so the dwell counter is a plain
    # cumulative count inside each run.
    out["run_id"] = out["regime_change"].cumsum() + out["session"].factorize()[0] * 10**6
    out["bars_in_regime"] = out.groupby("run_id", sort=False).cumcount() + 1
    # How long the OLD regime had run, on a change bar.
    prev_dwell = out["bars_in_regime"].groupby(out["session"], sort=False).shift(1)
    out["pre_dwell"] = prev_dwell.where(out["regime_change"])
    return out


def add_momentum_regimes(bars: pd.DataFrame, params: "dict | None" = None) -> pd.DataFrame:
    """Momentum + regimes in one call."""
    return assign_regimes(compute_momentum(bars, params), params)


# --- the live minute grid ---------------------------------------------------

def frame_from_bars(bars: "list[dict]") -> pd.DataFrame:
    """Alpaca/yfinance {"t","o","h","l","c","v"} bars -> the frame everything
    here runs on: an exchange-local index, lower-case OHLCV, and the session
    bookkeeping.

    `minute_frame` is the normal way in and reads the live buffer; this is for
    the callers that have bars from somewhere else -- a REST window recovering
    an opening range the buffer no longer covers, a stored session, or a test's
    fixture -- and want them shaped the same way.
    """
    columns = [
        "open", "high", "low", "close", "volume",
        "session", "bar_of_day", "minutes_from_open",
    ]
    if not bars:
        return pd.DataFrame(columns=columns)
    idx = pd.to_datetime([b["t"] for b in bars], utc=True, format="mixed").tz_convert(
        market_hours.MARKET_TZ
    )
    df = pd.DataFrame(
        {
            "open": [float(b["o"]) for b in bars],
            "high": [float(b["h"]) for b in bars],
            "low": [float(b["l"]) for b in bars],
            "close": [float(b["c"]) for b in bars],
            "volume": [float(b.get("v") or 0.0) for b in bars],
        },
        index=idx,
    )
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.between_time(RTH_START, RTH_END)
    df = df[df["close"] > 0]
    return add_session_columns(df)


def add_session_columns(bars: pd.DataFrame) -> pd.DataFrame:
    """Attach the per-day bookkeeping the rest of the pipeline groups on."""
    out = bars.copy()
    out["session"] = out.index.normalize()
    out["bar_of_day"] = out.groupby("session").cumcount()
    open_ts = out.index.normalize() + pd.Timedelta(f"{RTH_START}:00")
    out["minutes_from_open"] = (out.index - open_ts).total_seconds() / 60.0
    return out


def minute_frame(sym_state) -> pd.DataFrame:
    """The frame the pipeline runs on: today's session, live.

    Nothing here crosses the overnight gap, so it needs no history behind today
    at all.
    """
    today = clock.now().astimezone(market_hours.MARKET_TZ).date()
    with sym_state.lock:
        live_bars = list(sym_state.bars)
    live = frame_from_bars(live_bars)
    if not len(live):
        return live
    return live[live.index.date == today]


def regime_name(regime: "int | None") -> str:
    return REGIME_NAME.get(regime, "unknown")
