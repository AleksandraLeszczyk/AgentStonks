"""The average momentum of the next 30 minutes, at every minute (Fast&Furious).

FinNotebooks' `Fast&Furious` project writes `fastfurious_<TICKER>.*` in
`Code/Models`. Every other model Apple Trader runs on answers one question once
a morning; this one answers its question at the close of **every** minute bar
from 09:31 to 15:30:

    m30  = log(close[t+30] / c0) / 30            the average 1-minute momentum
    s30  = adr14 * sqrt(the share of the day's variance bars t+1 .. t+30 carry)
    y    = log(close[t+30] / c0) / s30           what the model learns

    forecast m30 (bps per minute) = y_hat * s30 / 30 * 1e4

`s30` is the expected size of a 30-minute move at that time of day in that
volatility regime, so 09:35 and 12:30 share one footing. A second model (a
LightGBM pair, Next2h's recipe) forecasts the next two hours' high and low in
units of `s120`; the trading system hangs its stop off that.

The live tape
-------------
On the basic Alpaca plan SIP runs 15 minutes late. At the close of bar t a
trader has SIP up to bar t-15 and **IEX for the last 15 minutes**, and every
feature reads exactly that (the notebook's scramble test checks it). The
`--- tape` block below builds every window statistic from those two parts; this
module hands it the same two tapes -- SIP through t-15 and IEX through t,
fetched from Alpaca -- live, and cuts a finished session's tapes at the same
two lines in a replay. Because every statistic reads only within those lines,
one pass over a session's grid gives each minute's features exactly as that
minute would have seen them, which is what lets a replay forecast a whole day
at once and a live run forecast every minute so far from the tape it has.

The mirror contract
-------------------
The `--- tape`, `--- features` and `--- models` blocks are verbatim copies of
the parts of `fastfurious/tape.py`, `features.py` and `models.py` the shipped
model reads, with the cuts each block names; the cleaning is `data.py`'s. If
`fastfurious` changes, retrain **and** update this module.
`tests/test_fastfurious_model.py` rebuilds the notebook's own panel from its raw
minute files and pins the features and the forecasts to it.

`build_panel` is restructured, not copied: the per-day quantities read off
earlier sessions (the variance profile, the same minute's past outcomes, the
IEX volume scale, last month's volume curve, the daily statistics) are
`context`, and the session's own statistics are `features`. Same arithmetic,
in the same order.

The option inputs of the stop's range model (the 1-day implied move, the 7-day
put wall, the put gamma wall) come from last night's positioning table, which
`highlow3m_model` already rebuilds from Alpaca's option bars -- the two
notebooks' tables are identical on all 686 closes both cover (same contracts,
same open-interest calibration, same dividend yield), so the rows are shared
and cached once under `data/highlow3m/`.

What the live path has to supply
--------------------------------
* ~200 calendar days of SIP and IEX minute bars before the session (cached in
  `HISTORY_DIR`, extended a day at a time): 126 sessions for the 126-day high,
  and 100 kept sessions for the threshold -- each of the 20 sessions it reads
  needs the same minute's outcomes of the 20 before it, and each of those the
  60-session variance profile before that.
* This session's SIP bars through t-15 and IEX bars through t.
* Last night's option positioning row, for the stop.

History is read strictly before the session being forecast, from the session
date the caller names rather than the wall clock, so a SimLab replay reads what
a live run on that day would have had.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import threading
import time
import types
import warnings
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# LightGBM brings its own OpenMP runtime (see `dayrange_model`); loaded up front
# as the other LightGBM-backed model modules do.
os.environ.setdefault("OMP_NUM_THREADS", "1")

try:  # pragma: no cover - depends on which optional extras are installed
    importlib.import_module("lightgbm")
except ImportError:
    pass

from . import market_hours, model_store  # noqa: E402
from .highlow3m_model import NYSE_HOLIDAYS  # noqa: E402
from .model_store import RTH_END, RTH_START, ModelStore  # noqa: E402
from .newsimpact_model import early_close  # noqa: E402

MODEL_PATH_ENV = "APPLE_FASTFURIOUS_MODEL"
DEFAULT_TICKER = model_store.DEFAULT_TICKER
# Which symbols have a bundle: `apple_models.FASTFURIOUS_TICKERS`.

# --- the question (fastfurious.config) ------------------------------------------

M = SESSION_MINUTES = 390
HORIZON = 30          # the momentum is averaged over the next 30 bars
RANGE_HORIZON = 120   # the stop's high and low: the next two hours
L = SIP_DELAY = 15    # SIP is 15 minutes late; the last 15 bars come from IEX
LAST_ORIGIN = 359     # 15:30, the last origin with a full 30-minute window
MIN_BARS_PER_SESSION = 385
OPEN_MINUTE = 9 * 60 + 30

# --- the trading system's threshold (fastfurious.trading.thresholds) ---------------

# An entry needs |y_hat| above the q-quantile of the forecasts of the previous
# 20 sessions, every minute of them; fewer than 5 such sessions and there is no
# threshold at all.
THRESHOLD_SESSIONS = 20
THRESHOLD_MIN_SESSIONS = 5

# Kept sessions the forecast needs before the session: the 126-day high reads
# 126 daily candles, and the threshold's 20 sessions each read the same
# minute's outcomes over the 20 before them, each of those the 60-session
# variance profile before that -- 100. The notebook had two years behind every
# row; a history this long reproduces it exactly, a shorter one only roughly.
MIN_PRIOR_SESSIONS = 127
# Asked for, in calendar days: comfortably more than MIN_PRIOR_SESSIONS.
HISTORY_CALENDAR_DAYS = 200

# SIP and IEX minute bars of the regular session, one parquet file per symbol
# and feed, and the calendar days they cover beside them.
HISTORY_DIR = Path(__file__).resolve().parent.parent / "data" / "fastfurious"
CACHE_LAYOUT = 1
FEEDS = ("sip", "iex")
BAR_COLUMNS = ["open", "high", "low", "close", "volume", "vwap"]


# --- tape (mirrors fastfurious.tape) -----------------------------------------------
#
# Cut: `live_close_ago` (the recent-bars and market groups read it, the shipped
# model reads neither).

FIELDS = ("open", "high", "low", "close", "volume", "vwap")


@dataclass
class Grid:
    """One feed's bars as D x 390 arrays; closes forward-filled within the day."""

    days: pd.DatetimeIndex
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    vwap: np.ndarray
    close_ff: np.ndarray
    ret: np.ndarray       # 1-minute log return of close_ff; bar 0 against its own open


def to_grid(minute: pd.DataFrame, days: pd.DatetimeIndex) -> Grid:
    di = days.get_indexer(minute["date"])
    ok = di >= 0
    rows, cols = di[ok], minute["minute"].to_numpy()[ok]
    arrays = {}
    for f in FIELDS:
        a = np.full((len(days), M), np.nan)
        a[rows, cols] = minute[f].to_numpy(float)[ok]
        arrays[f] = a
    arrays["volume"] = np.nan_to_num(arrays["volume"])
    close_ff = ffill(arrays["close"])
    with np.errstate(invalid="ignore", divide="ignore"):
        ret = np.diff(np.log(close_ff), axis=1, prepend=np.nan)
        ret[:, 0] = np.log(arrays["close"][:, 0] / arrays["open"][:, 0])
    return Grid(days, **arrays, close_ff=close_ff, ret=ret)


def ffill(a: np.ndarray) -> np.ndarray:
    return pd.DataFrame(a.T).ffill().to_numpy().T


def shift(a: np.ndarray, k: int) -> np.ndarray:
    """a[:, t - k] at t (NaN before the session); negative k looks ahead."""
    out = np.full_like(a, np.nan, dtype=float)
    if k > 0:
        out[:, k:] = a[:, :-k]
    elif k < 0:
        out[:, :k] = a[:, -k:]
    else:
        out[:] = a
    return out


def wsum(a: np.ndarray, w: int) -> np.ndarray:
    """Sum over bars t-w+1 .. t, NaN as 0, cut at the session start."""
    c = np.cumsum(np.nan_to_num(a), axis=1)
    out = c.copy()
    out[:, w:] -= c[:, :-w]
    return out


def wext(a: np.ndarray, w: int, how: str) -> np.ndarray:
    """Max or min over bars t-w+1 .. t, ignoring NaN, cut at the session start."""
    r = pd.DataFrame(a.T).rolling(w, min_periods=1)
    return (r.max() if how == "max" else r.min()).to_numpy().T


def forward_ext(a: np.ndarray, h: int, how: str) -> np.ndarray:
    """Max or min over bars t+1 .. t+h, cut at the session end."""
    return shift(wext(a[:, ::-1], h, how)[:, ::-1], -1)


def price(iex: Grid) -> np.ndarray:
    """The live price at the close of bar t: the last IEX trade."""
    return iex.close_ff


def live_open(sip: Grid, iex: Grid) -> np.ndarray:
    """Today's open as known at t: SIP's 09:30 bar once published, IEX's before."""
    t = np.arange(M)[None, :]
    return np.where(t >= L, sip.open[:, :1], iex.open[:, :1])


def live_ext(sip: Grid, iex: Grid, w: "int | None", how: str) -> np.ndarray:
    """High (max) or low (min) over the last w bars; w=None: the session so far."""
    field_ = "high" if how == "max" else "low"
    w = M if w is None else w
    recent = wext(getattr(iex, field_), min(w, L), how)
    if w <= L:
        return recent
    older = shift(wext(getattr(sip, field_), w - L, how), L)
    return np.fmax(recent, older) if how == "max" else np.fmin(recent, older)


def live_rsum(sip: Grid, iex: Grid, w: "int | None", power: int = 2) -> np.ndarray:
    """Sum of |1-minute returns|**power over the last w bars (None: the session so far).

    Returns of bars t-13 .. t are IEX's, the return into bar t-14 joins IEX to SIP,
    the rest are SIP's.
    """
    w = M if w is None else w
    f = lambda r: np.abs(r) ** power  # noqa: E731
    if w < L:
        return wsum(f(iex.ret), w)
    joint = f(np.log(shift(iex.close_ff, L - 1)) - np.log(shift(sip.close_ff, L)))
    out = wsum(f(iex.ret), L - 1) + np.nan_to_num(joint)
    if w > L:
        out = out + shift(wsum(f(sip.ret), w - L), L)
        out = np.where(np.isnan(out), wsum(f(iex.ret), L - 1), out)  # before t=15 there is no SIP part
    return out


def iex_scale(sip: Grid, iex: Grid, window: int = 20) -> np.ndarray:
    """Per day, the trailing SIP/IEX volume ratio over the previous `window` sessions."""
    s = pd.Series(sip.volume.sum(1)).rolling(window, min_periods=5).sum()
    i = pd.Series(iex.volume.sum(1)).rolling(window, min_periods=5).sum()
    return (s / i.replace(0, np.nan)).shift(1).bfill().to_numpy()


def live_volume(sip: Grid, iex: Grid, k: np.ndarray, w: "int | None", signed: bool = False) -> np.ndarray:
    """Volume over the last w bars (None: session), IEX's last 15 minutes scaled by k.
    `signed`: each bar's volume carries the sign of its return (buying minus selling pressure)."""
    w = M if w is None else w
    vs, vi = sip.volume, iex.volume
    if signed:
        vs, vi = vs * np.sign(np.nan_to_num(sip.ret)), vi * np.sign(np.nan_to_num(iex.ret))
    recent = k[:, None] * wsum(vi, min(w, L))
    if w <= L:
        return recent
    return recent + np.nan_to_num(shift(wsum(vs, w - L), L))


def live_vwap(sip: Grid, iex: Grid, k: np.ndarray) -> np.ndarray:
    """Session VWAP on the live tape."""
    pv_s, pv_i = np.nan_to_num(sip.vwap * sip.volume), np.nan_to_num(iex.vwap * iex.volume)
    v = np.nan_to_num(shift(wsum(sip.volume, M), L)) + k[:, None] * wsum(iex.volume, L)
    pv = np.nan_to_num(shift(wsum(pv_s, M), L)) + k[:, None] * wsum(pv_i, L)
    with np.errstate(invalid="ignore", divide="ignore"):
        return pv / np.where(v > 0, v, np.nan)


def past_mean(a: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    """Per minute, the mean over the previous `window` sessions (today excluded)."""
    return pd.DataFrame(a).rolling(window, min_periods=min_periods).mean().shift(1).to_numpy()


def variance_share(sip: Grid, window: int = 60) -> np.ndarray:
    """Each minute's expected share of the session's variance, from past sessions: the mean
    absolute 1-minute return per minute over the previous `window` sessions, smoothed over
    five minutes, squared, normalised to sum to one."""
    prof = past_mean(np.abs(sip.ret), window, 20)
    prof = pd.DataFrame(prof.T).rolling(5, center=True, min_periods=1).mean().to_numpy().T
    q = prof**2
    return q / q.sum(axis=1, keepdims=True)


# --- features (mirrors fastfurious.features) ------------------------------------------
#
# Cut: every group the shipped model does not read -- recent bars (but `rvx_15`,
# which the range model reads), yesterday (but `prev_rv`), the 7/14/28-day
# averages, long-term momentum (but `d_high126`), the point of control, the
# options group (but the three columns below), analysts, market, events.

EPS = 1e-12

MOMENTUM_COLS = [
    "minute", "dow", "win_share", "y_hist20", "sess_ret", "gap", "sess_hi", "sess_lo",
    "sess_pos", "sess_rngx", "sess_rvx", "vwap_dist", "rvol_sess", "sess_ofi", "or_hi", "or_lo",
]
RANGE_COLS = [
    "minute", "win_share120", "sess_rvx", "sess_rngx", "rvx_15", "rvol_sess", "or_hi", "iv1d_u",
    "prev_rv", "pw7_dist", "pgw_dist", "d_high126",
]
FEATURE_COLS = list(dict.fromkeys(MOMENTUM_COLS + RANGE_COLS))
OPTION_COLS = {"iv1d_u", "pw7_dist", "pgw_dist"}
# A row of the notebook's panel needs these (`build_panel`'s `need`): the
# anchors, the targets and four weeks of daily history.
NEED = ["c0", "p", "s30", "y", "up", "down", "d_ma28"]


def _log(a, b):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.log(a / b)


def _div(a, b):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(np.abs(b) > EPS, a / np.where(np.abs(b) > EPS, b, 1.0), np.nan)


def _col(v) -> np.ndarray:
    """A per-day series as a D x 1 column, to broadcast over minutes."""
    return np.asarray(v, dtype=float)[:, None]


def previous_row(table: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
    """For each day, the last row of `table` dated strictly before it."""
    left = pd.DataFrame({"day": days})
    right = table.sort_index().rename_axis("asof").reset_index()
    return pd.merge_asof(left, right, left_on="day", right_on="asof", allow_exact_matches=False).set_index("day")


def window_share(cshare: np.ndarray, h: int) -> "tuple[np.ndarray, np.ndarray]":
    """Expected variance share of bars t+1 .. t+H, H = min(h, bars left); and H."""
    t = np.arange(M)[None, :]
    H = np.minimum(h, M - 1 - t)
    end = np.take_along_axis(cshare, np.broadcast_to(np.minimum(t + H, M - 1), cshare.shape), axis=1)
    return end - cshare, np.broadcast_to(H, cshare.shape)


def daily_stats(daily: pd.DataFrame) -> pd.DataFrame:
    """What is known about past sessions at day d's open (everything shifted a day).
    Averages use the open and close, as asked; ranges use the full candles."""
    o, h, l, c, v = (daily[k] for k in ["open", "high", "low", "close", "volume"])
    mid = (o + c) / 2
    cc = np.log(c / c.shift(1))
    rng = np.log(h / l)
    lv = np.log(v.where(v > 0))

    f = pd.DataFrame(index=daily.index)
    for k in ["open", "high", "low", "close"]:
        f[f"prev_{k}"] = daily[k].shift(1)
    f["prev_rng"], f["prev_body"], f["prev_ret"] = rng.shift(1), np.log(c / o).shift(1), cc.shift(1)
    f["prev_rv"] = daily["rv"].shift(1) if "rv" in daily else np.nan
    f["prev_close_pos"] = ((c - l) / (h - l).where(h > l)).shift(1)
    f["prev_vol_z"] = ((lv - lv.rolling(28).mean()) / lv.rolling(28).std()).shift(1)
    for w in (7, 14, 28):
        f[f"ma{w}"] = mid.rolling(w).mean().shift(1)
        f[f"ccvol{w}"] = cc.rolling(w).std().shift(1)
        f[f"adr{w}"] = rng.rolling(w).mean().shift(1)
    f["ccvol63"] = cc.rolling(63).std().shift(1)
    f["adr126"] = rng.rolling(126).mean().shift(1)
    rv = daily["rv"] if "rv" in daily else pd.Series(np.nan, index=daily.index)
    f["daily_var14"] = (rv**2).rolling(14, min_periods=10).mean().shift(1)
    f["daily_var14"] = f["daily_var14"].fillna((0.5 * f["adr14"]) ** 2)  # before minute data: from the range
    for w in (5, 21, 63, 126, 252):
        f[f"mom{w}"] = np.log(c / c.shift(w)).shift(1)
    for w in (126, 252):
        f[f"high{w}"] = h.rolling(w).max().shift(1)
        f[f"low{w}"] = l.rolling(w).min().shift(1)
    f["adr14_usd"] = (h - l).rolling(14).mean().shift(1)
    f["dollar_adv20"] = (c * v).rolling(20).mean().shift(1)
    return f


@dataclass
class Context:
    """What each day's features read off the sessions before it, D rows each:
    the variance profile (cumulated), the same minute's past outcomes, the
    SIP/IEX volume scale and last month's cumulative volume curve -- plus the
    day's own targets, which only later days' `y_hist20` reads."""

    cshare: np.ndarray
    y: np.ndarray
    y_hist20: np.ndarray
    k: np.ndarray
    vol_norm: np.ndarray


def context(sip: Grid, iex: Grid, ds: pd.DataFrame) -> Context:
    """`build_panel`'s cross-session half: every row reads only the rows before it."""
    u = _col(ds["adr14"])
    cshare = np.cumsum(variance_share(sip), axis=1)
    w30, _ = window_share(cshare, HORIZON)
    s30 = u * np.sqrt(w30)
    c0 = sip.close_ff
    r30 = _log(shift(c0, -HORIZON), c0)
    with np.errstate(invalid="ignore", divide="ignore"):
        y = r30 / s30
    # the same minute's outcome over the last 20 sessions, smoothed over 15 minutes
    yh = pd.DataFrame(np.clip(y, -4, 4).T).rolling(15, center=True, min_periods=5).mean().to_numpy().T
    return Context(
        cshare=cshare,
        y=y,
        y_hist20=past_mean(yh, 20, 10),
        k=iex_scale(sip, iex),
        vol_norm=past_mean(np.cumsum(sip.volume, axis=1), 20, 10),
    )


def _rows(ctx: Context, rows: slice) -> Context:
    return Context(*(getattr(ctx, f)[rows] for f in ("cshare", "y", "y_hist20", "k", "vol_norm")))


def _grid_rows(g: Grid, rows: slice) -> Grid:
    return Grid(g.days[rows], *(getattr(g, f)[rows] for f in (*FIELDS, "close_ff", "ret")))


def features(sip: Grid, iex: Grid, ds: pd.DataFrame, ctx: Context,
             positioning: "pd.DataFrame | None" = None, targets: bool = False) -> "dict[str, np.ndarray]":
    """`build_panel`'s per-session half, D x 390 per column: the anchors and
    scales, the shipped models' features, and (`targets`) the outcomes a
    panel row needs to exist. `ds` and `ctx` are the rows of the same days."""
    days = sip.days
    D = len(days)
    t = np.broadcast_to(np.arange(M)[None, :], (D, M))
    f: "dict[str, np.ndarray]" = {}

    # --- the anchor, the scales ---------------------------------------------------
    p = price(iex)
    c0 = sip.close_ff
    u = _col(ds["adr14"])
    cshare = ctx.cshare
    w30, _ = window_share(cshare, HORIZON)
    w120, _ = window_share(cshare, RANGE_HORIZON)
    s30, s120 = u * np.sqrt(w30), u * np.sqrt(w120)
    f.update(c0=c0, p=p, u=np.broadcast_to(u, (D, M)), s30=s30, s120=s120)
    if targets:
        fut_hi = forward_ext(sip.high, RANGE_HORIZON, "max")
        fut_lo = forward_ext(sip.low, RANGE_HORIZON, "min")
        f.update(y=ctx.y, up=_log(fut_hi, c0) / s120, down=_log(c0, fut_lo) / s120)

    # --- time ------------------------------------------------------------------------
    f["minute"], f["dow"] = t.astype(float), _col(days.dayofweek)
    f["win_share"], f["win_share120"] = w30, w120
    f["y_hist20"] = ctx.y_hist20

    # --- recent bars: the 15-minute realised variance against its norm ---------------
    k = ctx.k
    dvar = _col(ds["daily_var14"])
    w = 15
    rv2 = live_rsum(sip, iex, w, 2)
    exp = np.where(t >= w, cshare - shift(cshare, w), cshare)
    f[f"rvx_{w}"] = 0.5 * _log(rv2, exp * dvar)

    # --- today so far (today's open, range, VWAP, volume) ------------------------------
    op = live_open(sip, iex)
    sh, sl = live_ext(sip, iex, None, "max"), live_ext(sip, iex, None, "min")
    f["sess_ret"] = _log(p, op) / u
    f["gap"] = _log(op, _col(ds["prev_close"])) / u
    f["sess_hi"], f["sess_lo"] = _log(sh, p) / u, _log(p, sl) / u
    f["sess_pos"] = np.clip(_div(_log(p, sl), _log(sh, sl)), 0, 1)
    f["sess_rngx"] = _log(_log(sh, sl), u * np.sqrt(cshare))
    f["sess_rvx"] = 0.5 * _log(live_rsum(sip, iex, None, 2), cshare * dvar)
    f["vwap_dist"] = _log(p, live_vwap(sip, iex, k)) / u
    vol_sess = live_volume(sip, iex, k, None)
    f["rvol_sess"] = _log(vol_sess + 1, ctx.vol_norm + 1)
    f["sess_ofi"] = _div(live_volume(sip, iex, k, None, signed=True), vol_sess)
    # the first 30 minutes' range: SIP once published, IEX from 10:00 to 10:15
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        or_s = (np.nanmax(sip.high[:, :30], axis=1), np.nanmin(sip.low[:, :30], axis=1))
        or_i = (np.nanmax(iex.high[:, :30], axis=1), np.nanmin(iex.low[:, :30], axis=1))
    or_hi = np.where(t >= 30 + L - 1, _col(or_s[0]), np.where(t >= 29, _col(or_i[0]), np.nan))
    or_lo = np.where(t >= 30 + L - 1, _col(or_s[1]), np.where(t >= 29, _col(or_i[1]), np.nan))
    f["or_hi"], f["or_lo"] = _log(or_hi, p) / u, _log(p, or_lo) / u

    # --- yesterday, the 28-day average, the 126-day high ------------------------------
    f["prev_rv"] = _col(ds["prev_rv"]) / u
    f["d_ma28"] = _log(p, _col(ds["ma28"])) / u
    f["d_high126"] = _log(p, _col(ds["high126"])) / u

    # --- options: last night's table, measured from the live price ----------------------
    if positioning is not None and len(positioning):
        r = previous_row(positioning, days)
        for name, col, side in (("pw7", "put_wall_7d", -1), ("pgw", "put_gamma_wall", -1)):
            f[f"{name}_dist"] = side * _log(_col(r[col]), p) / u
        f["iv1d_u"] = _col(r["iv_1d"] / np.sqrt(252)) / u
    else:
        for name in OPTION_COLS:
            f[name] = np.full((D, M), np.nan)

    out = {}
    for name, a in f.items():
        a = np.broadcast_to(np.asarray(a, dtype=float), (D, M))
        out[name] = np.where(np.isinf(a), np.nan, a)
    return out


# --- models (mirrors the inference half of fastfurious.models) ----------------------------


def _X(df: pd.DataFrame, cols: "list[str]") -> np.ndarray:
    return df[cols].to_numpy(np.float32)


@dataclass
class MomentumModel:
    """LightGBM on y, averaged over seeds (unpickled)."""

    cols: list
    params: dict = field(default_factory=dict)
    seeds: tuple = (0,)
    clip: float = 4.0
    models: list = field(default_factory=list, repr=False)
    calib: tuple = (0.0, 1.0)

    @property
    def is_classifier(self) -> bool:
        return self.params.get("objective") == "binary"

    def _raw(self, models, X) -> np.ndarray:
        a = _X(X, self.cols)
        if self.is_classifier:
            return np.mean([m.predict_proba(a)[:, 1] for m in models], axis=0) - 0.5
        return np.mean([m.predict(a) for m in models], axis=0)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raw = self._raw(self.models, X)
        if self.is_classifier:
            a, b = self.calib
            return a + b * raw + 1e-6 * raw  # keeps the ranking if the line is flat
        return raw


@dataclass
class RangeModel:
    """LightGBM pair on the next two hours' excursions in s120 units (unpickled)."""

    cols: list
    params: dict = field(default_factory=dict)
    models: dict = field(default_factory=dict, repr=False)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Excursions in s120 units, floored at 0.05."""
        return pd.DataFrame({f"{t}_hat": np.maximum(m.predict(_X(X, self.cols)), 0.05) for t, m in self.models.items()},
                            index=X.index)


@dataclass
class Bundle:
    """The momentum model, the range model and the notebook's metadata (unpickled)."""

    momentum: MomentumModel
    range: RangeModel
    meta: dict


# --- the saved bundle --------------------------------------------------------------


def _register_unpickle_alias() -> None:
    """Make `fastfurious.models.{Bundle,MomentumModel,RangeModel}` resolvable
    for joblib.

    The real package pulls the notebook's data paths, requests and yfinance in,
    so a stub pointing at the mirrors above stands in -- unless the real package
    is genuinely importable, in which case it wins.
    """
    if "fastfurious.models" in sys.modules:
        return
    try:
        __import__("fastfurious.models")
        return
    except Exception:
        sys.modules.pop("fastfurious", None)
    package = types.ModuleType("fastfurious")
    package.__path__ = []
    module = types.ModuleType("fastfurious.models")
    module.Bundle = Bundle
    module.MomentumModel = MomentumModel
    module.RangeModel = RangeModel
    package.models = module
    sys.modules["fastfurious"] = package
    sys.modules["fastfurious.models"] = module


_STORE = ModelStore(
    env_key=MODEL_PATH_ENV,
    filename="fastfurious_{ticker}.joblib",
    build=lambda path: _build_bundle(path),
)

model_path = _STORE.path
metadata_path = _STORE.metadata_path


def _build_bundle(path: Path) -> "dict | None":
    """One ticker's saved model plus its metadata, or None when it cannot be
    assembled.

    Refuses rather than degrades when the shipped models read a column this
    module does not mirror, or the momentum model is not the regressor (or the
    calibrated classifier) the notebook's `MomentumModel` can be.
    """
    try:
        import joblib
    except ImportError:
        return None
    _register_unpickle_alias()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = joblib.load(path)
    except (OSError, ValueError, KeyError, ModuleNotFoundError, AttributeError, ImportError, EOFError):
        return None
    momentum, ranges = getattr(model, "momentum", None), getattr(model, "range", None)
    if momentum is None or ranges is None or not getattr(momentum, "models", None):
        return None
    if set(momentum.cols) - set(FEATURE_COLS) or set(ranges.cols) - set(FEATURE_COLS):
        return None
    if set(getattr(ranges, "models", {}) or {}) != {"up", "down"}:
        return None
    meta = dict(getattr(model, "meta", None) or {})
    meta_file = metadata_path(path)
    try:
        if meta_file.exists():
            meta = {**meta, **json.loads(meta_file.read_text())}
    except (OSError, ValueError):
        pass
    return {
        "kind": "fastfurious",
        "model": model,
        "metadata": meta,
        "reads_options": bool(set(ranges.cols) & OPTION_COLS),
        "trained_at": f"through {meta['fit_end']}" if meta.get("fit_end") else None,
        "path": str(path),
        "ticker": str(meta.get("ticker") or "").upper() or None,
    }


load_bundle = _STORE.load
reset_bundle_cache = _STORE.reset


# --- the tape's cleaning (mirrors fastfurious.data) ------------------------------------


def nyse_sessions(dates) -> pd.DatetimeIndex:
    """Weekdays that are not NYSE holidays (half days included), among `dates`."""
    days = pd.DatetimeIndex(dates)
    holidays = pd.DatetimeIndex(pd.to_datetime(list(NYSE_HOLIDAYS)))
    return days[(days.dayofweek < 5) & ~days.isin(holidays)]


def add_session_columns(bars: pd.DataFrame) -> pd.DataFrame:
    """`date` (naive) and `minute` (0 is the 09:30 bar)."""
    out = bars.copy()
    out["date"] = out.index.normalize().tz_localize(None).as_unit("ns")
    out["minute"] = (out.index.hour * 60 + out.index.minute - OPEN_MINUTE).astype("int16")
    return out


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    out = add_session_columns(raw)
    px = out[["open", "high", "low", "close"]]
    sane = ((px > 0).all(axis=1) & (out["high"] >= out["low"])
            & (out["high"] >= px[["open", "close"]].max(axis=1) - 1e-9)
            & (out["low"] <= px[["open", "close"]].min(axis=1) + 1e-9))
    out = out[sane & out["minute"].between(0, SESSION_MINUTES - 1)].copy()
    out["volume"] = out["volume"].astype("int64")
    out["vwap"] = out["vwap"].where(out["vwap"].between(out["low"], out["high"]), out["close"])
    return out


def session_report(minute: pd.DataFrame) -> pd.DataFrame:
    """One row per session: kept, or the first rule it failed. (The notebook
    lists its 13:00 closes, to 2025; `early_close` is the rule that generates
    the list, identical on every date it covers.)"""
    rep = minute.groupby("date").agg(n_bars=("close", "size"), first_minute=("minute", "min"),
                                     last_minute=("minute", "max"))
    rules = {
        "half_day": early_close(pd.DatetimeIndex(rep.index)),
        "short": rep["n_bars"] < MIN_BARS_PER_SESSION,
        "no_open_bar": rep["first_minute"] > 0,
        "early_end": rep["last_minute"] < SESSION_MINUTES - 5,
    }
    rep["reason"] = ""
    for name, failed in reversed(rules.items()):
        rep.loc[np.asarray(failed), "reason"] = name
    rep["kept"] = rep["reason"] == ""
    return rep


def daily_bars(minute: pd.DataFrame) -> pd.DataFrame:
    """Minute bars rolled up per session; `rv` is the root of summed squared 1-minute log returns."""
    g = minute.groupby("date")
    daily = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                          "close": g["close"].last(), "volume": g["volume"].sum()})
    r = np.log(minute["close"]).groupby(minute["date"]).diff()
    daily["rv"] = np.sqrt((r**2).groupby(minute["date"]).sum())
    daily.index.name = "date"
    return daily


@dataclass
class Tapes:
    """A stretch of sessions as the notebook's `data.process` leaves them:
    SIP and IEX minute bars of the kept sessions, every session's daily candle
    (half days to 13:00, short sessions too) and why the rest were dropped."""

    sip: pd.DataFrame
    iex: pd.DataFrame
    daily: pd.DataFrame
    report: pd.DataFrame

    @property
    def kept(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.report.index[self.report["kept"]])


def process(raw_sip: pd.DataFrame, raw_iex: pd.DataFrame) -> Tapes:
    """`data.process` on tapes already in hand (regular-session bars, exchange-local)."""
    sip = clean(raw_sip)
    sip = sip[sip["date"].isin(nyse_sessions(sip["date"].unique()))]
    report = session_report(sip)
    kept = report.index[report["kept"]]
    iex = clean(raw_iex)
    half = pd.Series(early_close(pd.DatetimeIndex(sip["date"])), index=sip.index)
    daily = daily_bars(sip[~half | (sip["minute"] < 13 * 60 - OPEN_MINUTE)])
    return Tapes(sip[sip["date"].isin(kept)], iex[iex["date"].isin(kept)], daily, report)


# --- the minute history: SIP and IEX, regular session ---------------------------------------

_history_lock = threading.Lock()


def _tape_path(symbol: str, feed: str) -> Path:
    return HISTORY_DIR / f"{symbol.upper()}_{feed}.parquet"


def _index_path(symbol: str) -> Path:
    return HISTORY_DIR / f"{symbol.upper()}_tape.json"


def _empty_bars() -> pd.DataFrame:
    return pd.DataFrame(
        columns=BAR_COLUMNS, index=pd.DatetimeIndex([], tz=market_hours.MARKET_TZ, name="ts"), dtype=float
    )


def bars_frame(bars: "list[dict]") -> pd.DataFrame:
    """Alpaca `{"t","o","h","l","c","v","vw"}` bars -> an exchange-local frame
    of the regular session, sorted, de-duplicated (the notebook's `_trim`)."""
    if not bars:
        return _empty_bars()
    idx = pd.to_datetime([b["t"] for b in bars], utc=True, format="ISO8601").tz_convert(
        market_hours.MARKET_TZ
    )
    df = pd.DataFrame(
        {
            "open": [float(b["o"]) for b in bars],
            "high": [float(b["h"]) for b in bars],
            "low": [float(b["l"]) for b in bars],
            "close": [float(b["c"]) for b in bars],
            "volume": [float(b.get("v") or 0.0) for b in bars],
            "vwap": [float(b.get("vw", b["c"])) for b in bars],
        },
        index=pd.DatetimeIndex(idx, name="ts"),
    )
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df.between_time(RTH_START, RTH_END)


def _credentials(key: "str | None", secret: "str | None") -> "tuple[str, str]":
    """The caller's Alpaca keys, else the environment's (SimLab has no sidebar)."""
    key = key or os.getenv("ALPACA_API_KEY", "")
    secret = secret or os.getenv("ALPACA_SECRET", "")
    if not (key and secret):
        raise ValueError(
            "the Fast&Furious forecast needs Alpaca SIP and IEX minute bars and no Alpaca "
            "credentials are available (sidebar connection, or ALPACA_API_KEY / "
            "ALPACA_SECRET in the environment)."
        )
    return key, secret


def _fetch(symbol: str, start: datetime, end: datetime, key: str, secret: str, feed: str) -> pd.DataFrame:
    """One feed's regular-session minute bars for [start, end], split-adjusted."""
    from .rest import fetch_bars_range

    return bars_frame(fetch_bars_range(symbol, "1Min", start, end, key, secret,
                                       feed=feed, adjustment="split"))


def _fetch_stretch(symbol: str, first: date, before: date, key: str, secret: str) -> "dict[str, pd.DataFrame]":
    """Both feeds for the calendar days in [first, before)."""
    from .datalog import log_fetch

    tz = market_hours.MARKET_TZ
    start = datetime.combine(first, datetime.min.time(), tzinfo=tz)
    end = datetime.combine(before, datetime.min.time(), tzinfo=tz)
    out = {feed: _fetch(symbol, start, end, key, secret, feed) for feed in FEEDS}
    log_fetch(
        "minute bars (Fast&Furious history)",
        "Alpaca REST (SIP + IEX, regular session, split-adjusted)",
        symbol=symbol,
        detail=f"{len(out['sip'])} SIP + {len(out['iex'])} IEX bars from {first} to {before}",
    )
    return out


def _read_tapes(symbol: str) -> "tuple[dict, dict[str, pd.DataFrame]]":
    """The symbol's cached coverage and bars, or empty ones when unreadable or
    in another layout."""
    try:
        index = json.loads(_index_path(symbol).read_text())
    except (OSError, ValueError):
        index = {}
    if index.get("layout") != CACHE_LAYOUT or not index.get("from") or not index.get("through"):
        return {}, {feed: _empty_bars() for feed in FEEDS}
    bars = {}
    for feed in FEEDS:
        try:
            bars[feed] = pd.read_parquet(_tape_path(symbol, feed))
        except Exception:  # noqa: BLE001 -- a broken file is an empty cache
            return {}, {feed: _empty_bars() for feed in FEEDS}
    return index, bars


def _write_tapes(symbol: str, index: dict, bars: "dict[str, pd.DataFrame]") -> None:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    for feed in FEEDS:
        path = _tape_path(symbol, feed)
        tmp = path.with_suffix(".tmp")
        bars[feed].to_parquet(tmp)
        tmp.replace(path)
    path = _index_path(symbol)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"layout": CACHE_LAYOUT, **index}))
    tmp.replace(path)


def _session_close(bars: pd.DataFrame, day: str) -> "float | None":
    on = bars[bars.index.strftime("%Y-%m-%d") == day]
    return float(on["close"].iloc[-1]) if len(on) else None


def _join(old: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
    both = pd.concat([old, fresh]).sort_index()
    return both[~both.index.duplicated(keep="last")]


def raw_history(symbol: str, first: date, before: date,
                key: "str | None" = None, secret: "str | None" = None) -> "dict[str, pd.DataFrame]":
    """Both feeds' regular-session bars for the calendar days in [first, before).

    `highlow3m_model.history_inputs`' scheme: the cache remembers which calendar
    days it has looked at, and only the days outside that stretch are fetched --
    so a live session fetches one day each morning and a replay inside the
    stretch fetches nothing. Unlike that one's, the stretch only ever grows,
    and grows contiguously (a request past either end fetches the gap too):
    the forecast asks for a session's 200-day history and then for the session
    itself, one day past it, and a cache that restarted on every request that
    did not overlap it would fetch the 200 days again for every replayed day.
    Each extension re-reads the cached session at its seam: if that completed
    session's SIP close has moved, a split has rescaled the tape, and the cache
    is rebuilt from the request rather than mixed.
    """
    symbol = symbol.upper()
    last_wanted = before - timedelta(days=1)
    with _history_lock:
        index, bars = _read_tapes(symbol)
        lo = date.fromisoformat(index["from"]) if index.get("from") else None
        hi = date.fromisoformat(index["through"]) if index.get("through") else None
        if lo is None or hi is None or not len(bars["sip"]):
            lo = hi = None
        if lo is None or lo > first or hi < last_wanted:
            key, secret = _credentials(key, secret)

            def fetch(a: date, b: date) -> "dict[str, pd.DataFrame]":
                return _fetch_stretch(symbol, a, b, key, secret)

            def merge(fresh: dict, seam: str) -> bool:
                was, now = _session_close(bars["sip"], seam), _session_close(fresh["sip"], seam)
                if was is not None and now is not None and abs(now / was - 1.0) > 1e-6:
                    return False
                for feed in FEEDS:
                    bars[feed] = _join(bars[feed], fresh[feed])
                return True

            ok = True
            if lo is None:
                bars = fetch(first, before)
                lo, hi = first, last_wanted
            else:
                if lo > first:
                    seam = bars["sip"].index.min().strftime("%Y-%m-%d")
                    ok = merge(fetch(first, date.fromisoformat(seam) + timedelta(days=1)), seam)
                    lo = first
                if ok and hi < last_wanted:
                    seam = bars["sip"].index.max().strftime("%Y-%m-%d")
                    ok = merge(fetch(date.fromisoformat(seam), before), seam)
                    hi = last_wanted
            if not ok:
                bars = fetch(first, before)
                lo, hi = first, last_wanted
            _write_tapes(symbol, {"from": lo.isoformat(), "through": hi.isoformat()}, bars)
    window = {}
    for feed in FEEDS:
        b = bars[feed]
        stamp = b.index.tz_convert(market_hours.MARKET_TZ).normalize().tz_localize(None)
        window[feed] = b[(stamp >= pd.Timestamp(first)) & (stamp < pd.Timestamp(before))]
    return window


def history(symbol: str, before, key: "str | None" = None, secret: "str | None" = None) -> Tapes:
    """The sessions strictly before `before`, reaching HISTORY_CALENDAR_DAYS back."""
    before = pd.Timestamp(before).date()
    raw = raw_history(symbol, before - timedelta(days=HISTORY_CALENDAR_DAYS), before, key, secret)
    return process(raw["sip"], raw["iex"])


# --- this session's tape -----------------------------------------------------------


def _today_et() -> date:
    return datetime.now(market_hours.MARKET_TZ).date()


def _bar_time(day: date, minute: int) -> datetime:
    """The exchange-local timestamp of bar `minute` (0 = 09:30) on `day`."""
    return datetime(day.year, day.month, day.day, 9, 30, tzinfo=market_hours.MARKET_TZ) + timedelta(
        minutes=int(minute)
    )


# Live, IEX's bar t is asked for the moment the caller's bar t has closed, and
# Alpaca can publish it a moment later (`highlow3m_model.fetch_opening`'s retry).
_LIVE_RETRIES = 2
_LIVE_RETRY_SEC = 1.5
_LIVE_FRESH_SEC = 45


def live_tapes(symbol: str, day: date, through: int,
               key: "str | None" = None, secret: "str | None" = None,
               retries: int = None) -> "tuple[pd.DataFrame, pd.DataFrame, bool]":
    """Today's SIP bars through `through - 15` and IEX bars through `through`,
    straight from Alpaca, and whether IEX's bar `through` was in (False for a
    fetch made before it was published -- forecast from, not memoised).

    The SIP window ends 15 bars back, the line the basic plan's delay draws:
    a request reaching into the trailing 15 minutes is refused. `retries`
    (default `_LIVE_RETRIES`) is how often a missing IEX bar is asked for
    again, `_LIVE_RETRY_SEC` apart; a chart passes 0 rather than wait."""
    retries = _LIVE_RETRIES if retries is None else int(retries)
    key, secret = _credentials(key, secret)
    start = _bar_time(day, 0)
    sip = _empty_bars()
    if through - L >= 0:
        sip = _fetch(symbol, start, _bar_time(day, through - L), key, secret, "sip")
    target = _bar_time(day, through)
    for attempt in range(retries + 1):
        iex = _fetch(symbol, start, target, key, secret, "iex")
        settled = bool(len(iex)) and iex.index[-1] >= target
        fresh = datetime.now(timezone.utc) < target + timedelta(minutes=1, seconds=_LIVE_FRESH_SEC)
        if settled or not fresh or attempt == retries:
            break
        time.sleep(_LIVE_RETRY_SEC)
    return sip, iex, settled or not fresh


# --- one session's forecasts ---------------------------------------------------------


@dataclass
class _Day:
    """What a session's forecasts read from before it, built once per session."""

    day: pd.Timestamp
    history: Tapes
    daily: pd.DataFrame          # every session's candle before the day, plus a row for it
    prev_close: pd.Timestamp     # the last session before it: last night's option table
    positioning: "pd.DataFrame | None" = None
    positioning_problem: str = ""
    positioning_tried: float = 0.0          # monotonic time of the last failed attempt
    theta_abs: "np.ndarray | None" = None   # |y_hat| over the threshold's sessions
    theta_sessions: int = 0


_day_lock = threading.Lock()
_days: "dict[tuple, _Day]" = {}
_DAYS_MAX = 32
_forecast_memo: "dict[tuple, pd.DataFrame]" = {}
_FORECAST_MEMO_MAX = 256


def _day_key(bundle: dict, symbol: str, day: pd.Timestamp) -> tuple:
    return (bundle.get("path"), symbol, day)


def _normal_day(session_date) -> pd.Timestamp:
    day = pd.Timestamp(session_date).normalize()
    if day.tzinfo is not None:
        day = day.tz_localize(None)
    return day.as_unit("ns")


def _session_day(bundle: dict, symbol: str, day: pd.Timestamp,
                 key: "str | None", secret: "str | None") -> _Day:
    """The session's history, read (and memoised) once."""
    memo = _day_key(bundle, symbol, day)
    with _day_lock:
        found = _days.get(memo)
    if found is not None:
        return found
    hist = history(symbol, day, key, secret)
    if len(hist.kept) < MIN_PRIOR_SESSIONS:
        raise ValueError(
            f"only {len(hist.kept)} complete SIP sessions of history before {day.date()}; "
            f"the Fast&Furious forecast needs {MIN_PRIOR_SESSIONS} (the 126-day high, and the "
            "entry threshold's 20 sessions with their own history)."
        )
    if not len(hist.daily):
        raise ValueError(f"no SIP session before {day.date()}.")
    # Today's candle is a placeholder: every daily statistic is shifted a day.
    daily = pd.concat([hist.daily, pd.DataFrame(index=pd.DatetimeIndex([day], name="date"))])
    built = _Day(day=day, history=hist, daily=daily, prev_close=hist.daily.index.max())
    with _day_lock:
        if len(_days) >= _DAYS_MAX:
            _days.clear()
        _days.setdefault(memo, built)
        return _days[memo]


def _panel_inputs(hist: Tapes, daily: pd.DataFrame, sip_today: "pd.DataFrame | None",
                  iex_today: "pd.DataFrame | None", day: "pd.Timestamp | None"):
    """Grids, daily statistics and context over the history's kept sessions,
    plus `day` when it is given (with whatever of its tape is in hand)."""
    days = hist.kept
    sip, iex = hist.sip, hist.iex
    if day is not None:
        days = days.append(pd.DatetimeIndex([day]))
        sip = pd.concat([sip, sip_today]) if sip_today is not None and len(sip_today) else sip
        iex = pd.concat([iex, iex_today]) if iex_today is not None and len(iex_today) else iex
    days = pd.DatetimeIndex(days, name="date").as_unit("ns")
    gs, gi = to_grid(sip, days), to_grid(iex, days)
    ds = daily_stats(daily).reindex(days)
    return gs, gi, ds, context(gs, gi, ds)


def _frame(feats: "dict[str, np.ndarray]", row: int, minutes: np.ndarray) -> pd.DataFrame:
    """One session's rows of the feature arrays, as the panel holds them:
    features in float32, the anchors and scales in float64."""
    out = {"minute": minutes.astype("int16")}
    for name, a in feats.items():
        if name == "minute":
            continue
        values = a[row, minutes]
        out[name] = values if name in ("c0", "p", "u", "s30", "s120", "y", "up", "down") else values.astype("float32")
    return pd.DataFrame(out, index=pd.Index(minutes, name="minute"))


def _threshold_inputs(bundle: dict, ctx_day: _Day) -> None:
    """|y_hat| over every panel row of the 20 kept sessions before the day --
    what `trading.thresholds` takes its quantile of."""
    if ctx_day.theta_abs is not None:
        return
    hist = ctx_day.history
    gs, gi, ds, ctx = _panel_inputs(hist, ctx_day.daily.iloc[:-1], None, None, None)
    rows = slice(max(0, len(gs.days) - THRESHOLD_SESSIONS), len(gs.days))
    feats = features(_grid_rows(gs, rows), _grid_rows(gi, rows), ds.iloc[rows], _rows(ctx, rows), targets=True)
    minutes = np.arange(LAST_ORIGIN + 1)
    parts, sessions = [], 0
    for r in range(feats["p"].shape[0]):
        frame = _frame(feats, r, minutes)
        frame = frame.dropna(subset=NEED)
        if not len(frame):
            continue
        sessions += 1
        parts.append(np.abs(bundle["model"].momentum.predict(frame)))
    ctx_day.theta_abs = np.concatenate(parts) if parts else np.array([])
    ctx_day.theta_sessions = sessions


def threshold(bundle: dict, ticker: str, session_date, q: float,
              key: "str | None" = None, secret: "str | None" = None) -> "tuple[float, int]":
    """`(theta, sessions)`: the q-quantile of |y_hat| over the forecasts of the
    previous 20 sessions, every minute 09:31-15:30 of them, and how many
    sessions that was. NaN with fewer than 5 (`trading.thresholds`).

    The notebook's test window read the walk-forward models' forecasts of the
    sessions before it; live there is one model, so these are the saved model's
    own -- out of sample for every session after its fit (4 Sep 2026)."""
    symbol = str(ticker).upper()
    ctx_day = _session_day(bundle, symbol, _normal_day(session_date), key, secret)
    _threshold_inputs(bundle, ctx_day)
    if ctx_day.theta_sessions < THRESHOLD_MIN_SESSIONS:
        return float("nan"), ctx_day.theta_sessions
    return float(np.quantile(ctx_day.theta_abs, q)), ctx_day.theta_sessions


# A failed option-table build is tried again after this long, not every minute:
# a cold build fetches months of option bars.
_POSITIONING_RETRY_SEC = 300.0


def _positioning(ctx_day: _Day, symbol: str, key, secret) -> None:
    """Last night's option positioning row, from HighLow_3m's tables (shared)."""
    if ctx_day.positioning is not None:
        return
    if ctx_day.positioning_problem and time.monotonic() - ctx_day.positioning_tried < _POSITIONING_RETRY_SEC:
        return
    from . import highlow3m_model

    try:
        rows = highlow3m_model.positioning_rows(symbol, [ctx_day.prev_close], key, secret)
        table, _ = highlow3m_model.positioning_frames(rows)
        if not len(table):
            raise ValueError(f"no option positioning for the {ctx_day.prev_close.date()} close.")
        table.index = pd.DatetimeIndex(table.index, name="date").as_unit("ns")
        ctx_day.positioning = table
        ctx_day.positioning_problem = ""
    except Exception as exc:  # noqa: BLE001 -- the momentum forecast stands without it
        ctx_day.positioning_problem = str(exc)
        ctx_day.positioning_tried = time.monotonic()


def _predict(bundle: dict, frame: pd.DataFrame, with_range: bool) -> pd.DataFrame:
    """`Bundle.predict` on a session's rows, plus the price the forecast was
    read at (the live IEX price, not the SIP close the notebook anchors on,
    which is 15 minutes away live)."""
    model = bundle["model"]
    out = pd.DataFrame(index=frame.index)
    out["y_hat"] = model.momentum.predict(frame)
    out["mom_bps"] = out["y_hat"] * frame["s30"] / 30 * 1e4
    out["p"] = frame["p"]
    out["s30"] = frame["s30"]
    out["s120"] = frame["s120"]
    if with_range:
        rng = model.range.predict(frame)
        out = out.join(rng)
        out["pred_high"] = frame["p"] * np.exp(rng["up_hat"] * frame["s120"])
        out["pred_low"] = frame["p"] * np.exp(-rng["down_hat"] * frame["s120"])
    else:
        for col in ("up_hat", "down_hat", "pred_high", "pred_low"):
            out[col] = np.nan
    return out


def is_half_day(session_date) -> bool:
    return bool(early_close(pd.DatetimeIndex([_normal_day(session_date)]))[0])


def session_forecasts(
    bundle: dict,
    ticker: str,
    session_date,
    through: "int | None" = None,
    key: "str | None" = None,
    secret: "str | None" = None,
    wait: bool = True,
) -> pd.DataFrame:
    """Every minute's forecast of the session through bar `through` (0 =
    09:30; all of 09:31-15:30 when None or past it), one row per minute.

    Columns: `y_hat` (s30 units), `mom_bps` (bps per minute over the next 30),
    `p` (the live price it was read at), `s30`, `s120`, `up_hat` / `down_hat`
    (the next two hours' excursions in s120 units, floored at 0.05) and
    `pred_high` / `pred_low` (those as prices off `p`). Rows the notebook's
    panel would not have -- no IEX print yet, too little daily history -- are
    left out. `attrs["range_problem"]` says why the range columns are NaN when
    last night's option table could not be read.

    A finished session is read whole from the history cache and memoised, so a
    SimLab grid replaying it pays once; today's is fetched live, SIP through
    `through - 15` and IEX through `through` -- waiting a few seconds for an
    IEX bar not yet published unless `wait` is off (a chart, which draws what
    is there). Either way a minute's row reads nothing that was not public at
    its close.
    """
    symbol = str(ticker).upper()
    day = _normal_day(session_date)
    if is_half_day(day):
        raise ValueError(
            f"{day.date()} closes at 13:00, and the notebook keeps no such session: there is no "
            "Fast&Furious forecast for it."
        )
    last = LAST_ORIGIN if through is None else min(int(through), LAST_ORIGIN)
    if last < 0:
        raise ValueError("no bar of the session has closed yet.")
    live = day.date() >= _today_et()
    # A finished session is memoised whole, today's per minute.
    memo = (bundle.get("path"), symbol, day, last if live else None)
    found = _forecast_memo.get(memo)
    if found is not None:
        return found[found.index <= last].copy()

    ctx_day = _session_day(bundle, symbol, day, key, secret)
    settled = True
    if live:
        raw_sip, raw_iex, settled = live_tapes(
            symbol, day.date(), last, key, secret, retries=None if wait else 0
        )
        sip_today, iex_today = clean(raw_sip), clean(raw_iex)
    else:
        raw = raw_history(symbol, day.date(), day.date() + timedelta(days=1), key, secret)
        sip_today, iex_today = clean(raw["sip"]), clean(raw["iex"])
        if not len(sip_today):
            raise ValueError(f"SIP printed nothing on {day.date()}.")
        report = session_report(sip_today)
        if not bool(report["kept"].iloc[0]):
            raise ValueError(
                f"{day.date()} is not a session the notebook keeps "
                f"({report['reason'].iloc[0].replace('_', ' ')}), so there is no forecast for it."
            )
    reads_options = bundle.get("reads_options", True)
    if reads_options:
        _positioning(ctx_day, symbol, key, secret)
    gs, gi, ds, ctx = _panel_inputs(ctx_day.history, ctx_day.daily, sip_today, iex_today, day)
    row = slice(len(gs.days) - 1, len(gs.days))
    feats = features(_grid_rows(gs, row), _grid_rows(gi, row), ds.iloc[row], _rows(ctx, row),
                     ctx_day.positioning)
    minutes = np.arange(last + 1 if live else LAST_ORIGIN + 1)
    frame = _frame(feats, 0, minutes)
    # The panel's own rows: a minute the notebook keeps needs the live price,
    # its scale and four weeks of daily history (the targets and the SIP
    # anchor are there for every minute of a kept session).
    frame = frame.dropna(subset=["p", "s30", "d_ma28"])
    with_range = not reads_options or ctx_day.positioning is not None
    out = _predict(bundle, frame, with_range)
    out.attrs["range_problem"] = "" if with_range else ctx_day.positioning_problem
    # Not memoised without the range a later try may still read.
    if settled and with_range:
        if len(_forecast_memo) >= _FORECAST_MEMO_MAX:
            _forecast_memo.clear()
        _forecast_memo[memo] = out
    return out[out.index <= last].copy()


def forecast_at(
    bundle: dict,
    ticker: str,
    session_date,
    minute: int,
    q: float,
    key: "str | None" = None,
    secret: "str | None" = None,
) -> dict:
    """The forecast at the close of bar `minute`, with the entry threshold.

    Keys: `minute`, `y_hat`, `mom_bps`, `price`, `theta` (y units),
    `theta_bps` (the threshold in bps per minute at this minute's scale),
    `theta_sessions`, `signal` (+1 at or above the threshold, -1 at or below
    its negative, 0 between), `down_hat`, `up_hat`, `s30`, `s120`,
    `pred_high`, `pred_low` and `range_problem`. Raises ValueError when there is no
    forecast for that minute (none after 15:30, none before IEX has printed).
    """
    minute = int(minute)
    if minute > LAST_ORIGIN:
        raise ValueError("the last forecast is made at 15:30 (the 15:29 bar's close).")
    fc = session_forecasts(bundle, ticker, session_date, minute, key, secret)
    if minute not in fc.index:
        raise ValueError(f"no forecast at bar {minute}: IEX has not printed yet, or the history is short.")
    row = fc.loc[minute]
    theta, sessions = threshold(bundle, ticker, session_date, q, key, secret)
    y_hat = float(row["y_hat"])
    signal = 0
    if np.isfinite(theta):
        signal = 1 if y_hat >= theta else -1 if y_hat <= -theta else 0
    return {
        "minute": minute,
        "y_hat": y_hat,
        "mom_bps": float(row["mom_bps"]),
        "price": float(row["p"]),
        "theta": theta,
        "theta_bps": theta * float(row["s30"]) / 30 * 1e4 if np.isfinite(theta) else float("nan"),
        "theta_sessions": sessions,
        "signal": signal,
        "up_hat": float(row["up_hat"]),
        "down_hat": float(row["down_hat"]),
        "s30": float(row["s30"]),
        "s120": float(row["s120"]),
        "pred_high": float(row["pred_high"]),
        "pred_low": float(row["pred_low"]),
        "range_problem": fc.attrs.get("range_problem", ""),
    }


def warm(bundle: dict, ticker: str, session_date,
         key: "str | None" = None, secret: "str | None" = None) -> None:
    """Fill what the session's first forecast reads, ahead of it: the minute
    history, the threshold's forecasts and last night's option table."""
    symbol = str(ticker).upper()
    day = _normal_day(session_date)
    ctx_day = _session_day(bundle, symbol, day, key, secret)
    _threshold_inputs(bundle, ctx_day)
    if bundle.get("reads_options", True):
        _positioning(ctx_day, symbol, key, secret)


def reset_caches() -> None:
    """Forget every memoised session (tests; a refit bundle)."""
    with _day_lock:
        _days.clear()
    _forecast_memo.clear()
