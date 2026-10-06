"""Where the rest of the session's high and low will land, called at 9:33 (HighLow_3m).

FinNotebooks' `HighLow_3m` project writes `highlow3m_<TICKER>.*` in
`Code/Models`. Its question is not quite HighLow's or HighLow2's:

    up   = log(rest_high / close3) / adr14     how far above the 9:33 price
    down = log(close3 / rest_low)  / adr14     how far below it

    pred_high = close3 * exp(up   * adr14)
    pred_low  = close3 * exp(-down * adr14)

`close3` is the close of the 9:32 bar **on IEX**, the only live tape at 9:33 on
a basic Alpaca plan; `rest_high` and `rest_low` are the extremes of the **SIP
bars from 9:33 to 15:59** -- the part of the day a 9:33 order can still reach.
On 47% of AAPL sessions the day's high or low prints in the first three
minutes, so this is a different number from the whole session's extreme, and
it is not clipped to contain the opening range (HighLow's and HighLow2's are).
The forecast says so (`range_after_opening`), and
`apple_trader.DayRangeTrader._move_range` then measures breaches on the bars
after the window rather than on the whole session.

What else is new, and why it is a module of its own:

* **Three opening minutes**, not five, all IEX's (open, extremes, volume).
* **The options market.** Seven of the 21 inputs read last night's option
  positioning: the near-term call and put walls, the call gamma wall, dealer
  gamma re-read at the 9:33 price against normal turnover, the one-session
  implied move, implied against realised vol, and put/call volume. Alpaca
  keeps no history of open interest, so the notebook rebuilds it per contract
  as the running sum of daily volume weighted by the root of the average trade
  size, scaled to contracts by a calibration on expired contracts
  (`NOTEBOOK_OPTIONS`). This module rebuilds it the same way, from Alpaca's
  daily option bars, cached per expiration (`_options_dir`).
* **LightGBM (three seeds) 0.8 + a linear median regression 0.2.** The linear
  part is a scikit-learn pipeline pickled under 1.7.2, which 1.9 cannot run
  (`SimpleImputer` lost an attribute), so it is evaluated here from its fitted
  arrays -- the same arithmetic, verified to the last digit.

The mirror contract
-------------------
The `--- features`, `--- options` and `--- models` blocks are verbatim copies
of the parts of `highlow3m/features.py`, `options.py` and `models.py` the
shipped model reads, with the cuts each block names. The SIP/IEX cleaning
(`highlow3m/data.py`) is line for line `highlow2.data`'s, so it is imported
from `highlow2_model` rather than copied twice. If `highlow3m` changes, retrain
**and** update this module. `tests/test_highlow3m_model.py` re-forecasts the
notebook's own sessions from its raw minute files and its option bars and pins
the result.

Not mirrored, and refused by `_build_bundle` rather than predicted around: the
analysts, market and premarket groups (built in the notebook, pruned before
shipping), `or_vwap_dist` and `opt_volume_rel`.

What the live path has to supply
--------------------------------
* Per past session (cached in `HISTORY_DIR`, fetched only where the cache
  falls short, `highlow2_model.history_inputs`' scheme): the SIP rollup, the
  SIP extremes after 9:33 (the targets, which `down_hist14` averages), the IEX
  3-minute opening, and the SIP close of *every* session -- half days and
  dropped sessions too, which the option tables price off.
* Per previous close (cached in `_positioning_path`): the option positioning
  row. Built from the contract list (Alpaca's trading API), every listed
  expiration's daily bars over its last 120 days, and the 13-week T-bill yield
  (Yahoo ^IRX) for the risk-free rate. A row for day d depends on nothing
  after d, so once built it is kept.
* At 9:33: IEX's 09:30-09:32 bars (`fetch_opening`).

History is read strictly *before* the session being forecast, from the
session date the caller names rather than the wall clock, so a SimLab replay
reads the same honest history a live run on that day would have.
"""

from __future__ import annotations

import importlib
import json
import math
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
from .highlow2_model import (  # noqa: E402
    OHLCV,
    _fetch_tape,
    add_session_columns,
    bars_frame,  # noqa: F401 -- re-exported: Alpaca bar dicts -> this module's frames
    clean_minute,
    daily_bars,
    sane_bars,
    trim_to_sessions,
)
from .model_store import RTH_END, RTH_START, ModelStore  # noqa: E402
from .newsimpact_model import early_close  # noqa: E402

MODEL_PATH_ENV = "APPLE_HIGHLOW3M_MODEL"
DEFAULT_TICKER = model_store.DEFAULT_TICKER
# Which symbols have a bundle: `apple_models.HIGHLOW3M_TICKERS`.

# The forecast is made at 9:33, once the 9:30-9:32 bars have closed, from IEX
# (`config.OPENING_MINUTES`, `config.OPENING_FEED`).
OPENING_MINUTES = 3
OPENING_FEED = "iex"
SESSION_MINUTES = 390
OPEN_MINUTE = 9 * 60 + 30
# `config.MIN_BARS_PER_SESSION`.
MIN_BARS_PER_SESSION = 385

# Prior sessions the feature row needs: 126-day momentum reads
# `mid.shift(126)` of a series that is itself shifted a day. The notebook lets
# it be missing in 2024 (its data starts then); live a history that short is a
# fetch gone wrong.
MIN_PRIOR_SESSIONS = 127
# Asked for, in calendar days: comfortably more than MIN_PRIOR_SESSIONS, and
# more than the option tables' 120 days of contract history (their spot is the
# SIP close).
HISTORY_CALENDAR_DAYS = 220

# Per-session summaries of the SIP and IEX tapes, one JSON file per symbol, the
# option positioning rows beside them, and the option chain's raw daily bars
# under `options/`.
HISTORY_DIR = Path(__file__).resolve().parent.parent / "data" / "highlow3m"
CACHE_LAYOUT = 1


# --- calendar (config.NYSE_HOLIDAYS) -------------------------------------------

# NYSE full-day closures, published years ahead: the notebook's list (to 2027),
# with 2028 added for expiries the notebook's window never reached. Read by the
# option tables' session clock (`sessions_between`) and the monthly expiry, and
# to keep a stray print on a holiday out of the session closes.
NYSE_HOLIDAYS: "tuple[str, ...]" = (
    "2024-01-01", "2024-01-15", "2024-02-19", "2024-03-29", "2024-05-27", "2024-06-19",
    "2024-07-04", "2024-09-02", "2024-11-28", "2024-12-25",
    "2025-01-01", "2025-01-09", "2025-01-20", "2025-02-17", "2025-04-18", "2025-05-26",
    "2025-06-19", "2025-07-04", "2025-09-01", "2025-11-27", "2025-12-25",
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25", "2026-06-19",
    "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
    "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31", "2027-06-18",
    "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24",
    "2028-01-17", "2028-02-21", "2028-04-14", "2028-05-29", "2028-06-19", "2028-07-04",
    "2028-09-04", "2028-11-23", "2028-12-25",
)


# --- features (mirrors highlow3m.features) ---------------------------------------
#
# Cut: the analysts, market and premarket groups, `or_vwap_dist` (the app's
# bars carry no VWAP) and `opt_volume_rel` (it rolls over 20 sessions of option
# tables, where this module builds the one it needs). None is shipped.

N = OPENING_MINUTES
SHORT_WINDOWS = (7, 14, 28)
LONG_WINDOWS = (63, 126)
TARGETS = ["up", "down"]
SQRT_YEAR = np.sqrt(252.0)
EPS = 1e-12


def _pos_in_range(value, low, high):
    """Where `value` sits inside [low, high]; 0.5 for an empty range."""
    span = high - low
    return ((value - low) / span.where(span > EPS)).fillna(0.5)


def previous_row(table: pd.DataFrame, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """For each session, the last row of `table` dated strictly before it, and its date as `asof`."""
    left = pd.DataFrame({"session": pd.DatetimeIndex(sessions)}).sort_values("session")
    right = table.sort_index().rename_axis("asof").reset_index()
    out = pd.merge_asof(left, right, left_on="session", right_on="asof", allow_exact_matches=False)
    return out.set_index("session").rename_axis("date")


def daily_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Everything known about past sessions by the open of day t (all shifted a day).

    The averages use the open/close mid price, (open + close) / 2, as asked; ranges
    and realised volatility use the full SIP candles.
    """
    o, h, l, c, v, rv = (daily[k] for k in ["open", "high", "low", "close", "volume", "rv"])
    mid = (o + c) / 2
    cc = np.log(c / c.shift(1))
    rng = np.log(h / l)
    body = np.log(c / o)
    logv = np.log(v)

    f = pd.DataFrame(index=daily.index)
    f["prev_open"], f["prev_close"], f["prev_mid"] = o.shift(1), c.shift(1), mid.shift(1)
    f["prev_range"] = rng.shift(1)
    f["prev_rv"] = rv.shift(1)
    f["prev_body"] = body.shift(1)
    f["prev_ret"] = cc.shift(1)
    f["prev_up"] = np.log(h / o).shift(1)
    f["prev_down"] = np.log(o / l).shift(1)
    f["prev_close_pos"] = _pos_in_range(c, l, h).shift(1)
    f["prev_volume_z"] = ((logv - logv.rolling(28).mean()) / logv.rolling(28).std()).shift(1)

    for w in SHORT_WINDOWS:
        f[f"ma{w}"] = mid.rolling(w).mean().shift(1)
        f[f"vol{w}"] = cc.rolling(w).std().shift(1)            # close-to-close volatility
        f[f"body{w}"] = body.abs().rolling(w).mean().shift(1)  # open-to-close move
        f[f"adr{w}"] = rng.rolling(w).mean().shift(1)          # average daily range
        f[f"rv{w}"] = rv.rolling(w).mean().shift(1)            # intraday volatility

    # long-term momentum. The data starts in 2024, so the 126-day numbers exist from
    # mid-2024 and stay missing before that.
    f["mom28"] = np.log(mid / mid.shift(28)).shift(1)
    for w in LONG_WINDOWS:
        f[f"mom{w}"] = np.log(mid / mid.shift(w)).shift(1)
        f[f"vol{w}"] = cc.rolling(w, min_periods=w // 2).std().shift(1)
    f["adr126"] = rng.rolling(126, min_periods=60).mean().shift(1)
    f["dist_high126"] = np.log(c / h.rolling(126, min_periods=60).max()).shift(1)
    f["dist_low126"] = np.log(c / l.rolling(126, min_periods=60).min()).shift(1)

    # dollar ADR for the trading rule, dollar volume for sizing dealer gamma; not model inputs
    f["adr14_usd"] = (h - l).rolling(14).mean().shift(1)
    f["dollar_adv20"] = (c * v).rolling(20).mean().shift(1)
    return f


def opening_features(iex: pd.DataFrame, n: int = N) -> pd.DataFrame:
    """The first `n` one-minute bars of each session, from IEX. (Less `vwap3`
    and `or_vwap_dist`, which no shipped model reads.)"""
    first = iex[iex["minute"] < n]
    g = first.groupby("date")
    bar_ret = np.log(first["close"] / first["open"])
    f = pd.DataFrame({
        "open3": g["open"].first(), "high3": g["high"].max(), "low3": g["low"].min(),
        "close3": g["close"].last(), "volume3": g["volume"].sum(), "bars3": g["close"].size(),
    })
    f["or_up"] = np.log(f["high3"] / f["open3"])
    f["or_down"] = np.log(f["open3"] / f["low3"])
    f["or_ret"] = np.log(f["close3"] / f["open3"])
    f["or_range"] = np.log(f["high3"] / f["low3"])
    f["or_close_pos"] = _pos_in_range(f["close3"], f["low3"], f["high3"])
    f["or_bar_std"] = bar_ret.groupby(first["date"]).std().fillna(0.0)
    f["or_last_ret"] = bar_ret.groupby(first["date"]).last()   # the move into 9:33
    return f


def rest_of_session(sip: pd.DataFrame, n: int = N) -> pd.DataFrame:
    """The high, low and close of the SIP bars from 9:33 on: what the forecast is about."""
    g = sip[sip["minute"] >= n].groupby("date")
    return pd.DataFrame({"rest_high": g["high"].max(), "rest_low": g["low"].min(), "rest_close": g["close"].last()})


def _gex_at(profile: pd.DataFrame, asof: pd.Series, m: pd.Series) -> pd.DataFrame:
    """Net and gross dealer gamma from each session's `asof` profile, read at log-move `m`."""
    out = pd.DataFrame(np.nan, index=m.index, columns=["net", "gross"])
    if profile is None or profile.empty:
        return out
    net = profile.pivot(index="date", columns="m", values="net_gex")
    gross = profile.pivot(index="date", columns="m", values="gross_gex")
    grid = net.columns.to_numpy(float)
    row = net.index.get_indexer(pd.DatetimeIndex(asof))
    for i, (r, x) in enumerate(zip(row, m.to_numpy())):
        if r >= 0 and np.isfinite(x):
            out.iloc[i] = np.interp(x, grid, net.iloc[r].to_numpy()), np.interp(x, grid, gross.iloc[r].to_numpy())
    return out


def options_features(positioning: pd.DataFrame, profile: pd.DataFrame, f: pd.DataFrame) -> pd.DataFrame:
    """Last night's option positioning, measured from the 9:33 price in ADRs."""
    p = previous_row(positioning, f.index)
    close3, adr = f["close3"], f["adr14"]
    o = pd.DataFrame(index=f.index)
    # walls: where open interest is stacked above and below the price
    o["call_wall_dist"] = np.log(p["call_wall"] / close3) / adr
    o["put_wall_dist"] = np.log(close3 / p["put_wall"]) / adr
    o["call_wall7_dist"] = np.log(p["call_wall_7d"] / close3) / adr
    o["put_wall7_dist"] = np.log(close3 / p["put_wall_7d"]) / adr
    o["call_gwall_dist"] = np.log(p["call_gamma_wall"] / close3) / adr
    o["put_gwall_dist"] = np.log(close3 / p["put_gamma_wall"]) / adr
    o["wall_width"] = np.log(p["call_wall"] / p["put_wall"]) / adr
    o["flip_dist"] = np.log(close3 / p["gamma_flip"]) / adr
    # dealer gamma: as a share of all gamma (the proxy's level drifts, the ratio does not),
    # at last night's close and re-read at the 9:33 price, and against normal turnover
    at_open = _gex_at(profile, p["asof"], np.log(close3 / p["spot"]))
    o["gex_ratio_close"] = p["gex_ratio"]
    o["gex_ratio_open"] = at_open["net"] / at_open["gross"].where(at_open["gross"] > 0)
    o["gex_open_adv"] = at_open["net"] / f["dollar_adv20"]
    o["gex_negative"] = (at_open["net"] < 0).astype(float).where(at_open["net"].notna())
    # the options market's own guess at today's move, against the stock's recent range
    o["iv1d_move"] = p["iv_1d"] / SQRT_YEAR / adr
    o["iv_rv_ratio"] = p["iv_30d"] / (f["vol28"] * SQRT_YEAR)
    o["iv_term"] = np.log(p["iv_1d"] / p["iv_30d"])
    # activity
    o["pcr_volume"] = np.log((p["put_volume"] + 1) / (p["call_volume"] + 1))
    o["pcr_oi"] = np.log((p["put_oi"] + 1) / (p["call_oi"] + 1))
    o["opex_today"] = p["next_is_opex"].astype(float)
    return o


GROUPS: "dict[str, list[str]]" = {
    "opening": [
        "or_up", "or_down", "or_ret", "or_range", "or_close_pos", "or_bar_std", "or_last_ret",
        "or_range_vs_rv", "or_volume_rel",
    ],
    "prior": [
        "gap", "open_vs_mid", "prev_range", "prev_rv", "prev_body", "prev_ret", "prev_up", "prev_down",
        "prev_close_pos", "prev_volume_z",
    ],
    "averages": [
        "ma7_dist", "ma14_dist", "ma28_dist", "vol7", "vol14", "vol28", "body7", "body14", "body28",
        "adr7_rel", "adr28_rel", "rv7", "rv14", "rv28", "up_hist14", "down_hist14",
    ],
    "momentum": [
        "mom28", "mom63", "mom126", "dist_high126", "dist_low126", "vol63", "vol_regime", "dow",
    ],
    "options": [
        "call_wall_dist", "put_wall_dist", "call_wall7_dist", "put_wall7_dist", "call_gwall_dist",
        "put_gwall_dist", "wall_width", "flip_dist", "gex_ratio_close", "gex_ratio_open", "gex_open_adv",
        "gex_negative", "iv1d_move", "iv_rv_ratio", "iv_term", "pcr_volume", "pcr_oi", "opex_today",
    ],
}

# divided by adr14, so a split learned in a calm year still applies in a wild one
_PER_ADR = [
    "or_up", "or_down", "or_ret", "or_range", "or_bar_std", "or_last_ret",
    "gap", "open_vs_mid", "prev_range", "prev_rv", "prev_body", "prev_ret", "prev_up", "prev_down",
    "vol7", "vol14", "vol28", "body7", "body14", "body28", "rv7", "rv14", "rv28", "vol63",
]


def feature_cols(groups=tuple(GROUPS)) -> "list[str]":
    cols: "list[str]" = []
    for g in groups:
        cols += [c for c in GROUPS[g] if c not in cols]
    return cols


# Every column this module can build: what a shipped component may read.
FEATURE_COLS = feature_cols()
OPTION_COLS = set(GROUPS["options"])

# Columns a shipped row may be missing and still be forecast, as the notebook's
# panel lets them be (its `dropna` asks only for the anchor, the scale and four
# weeks of history; the trees and the linear part's imputer cope with the rest).
# A wall is missing on a day no open interest sits on that side of the price
# within reach -- twice in AAPL's 2025-26, for the 7-day walls; the gamma flip
# whenever net gamma keeps one sign over the grid. Everything else missing
# means a fetch went wrong, and the forecast is refused rather than imputed.
NAN_OK = {
    "call_wall_dist", "put_wall_dist", "call_wall7_dist", "put_wall7_dist", "call_gwall_dist",
    "put_gwall_dist", "wall_width", "flip_dist",
}


def panel_from(daily: pd.DataFrame, opening: pd.DataFrame, rest: pd.DataFrame,
               positioning: "pd.DataFrame | None" = None,
               profile: "pd.DataFrame | None" = None) -> pd.DataFrame:
    """`highlow3m.features.build_panel` for the groups the model reads.

    The notebook's `build_panel(inputs)` is `daily_features(daily_bars(sip))`
    joined to `opening_features(iex)` and `rest_of_session(sip)`, then this
    body, verbatim -- split at the joins because the live path caches
    per-session summaries rather than re-reading months of minute bars. What
    differs: nothing is dropped at the end (at 9:33 today's row has no targets,
    and the caller checks the row itself), and the option tables are handed in
    rather than loaded.
    """
    f = daily_features(daily).join(opening, how="inner").join(rest)

    f["gap"] = np.log(f["open3"] / f["prev_close"])
    f["open_vs_mid"] = np.log(f["open3"] / f["prev_mid"])
    f["or_range_vs_rv"] = f["or_range"] / f["rv14"]
    lv3 = np.log(f["volume3"].clip(lower=1))
    # IEX's own history: it carries a few percent of the tape, so SIP volume is no yardstick
    f["or_volume_rel"] = lv3 - np.log(f["volume3"].rolling(14, min_periods=7).mean().shift(1))
    for w in SHORT_WINDOWS:
        f[f"ma{w}_dist"] = np.log(f["close3"] / f[f"ma{w}"]) / f["adr14"]
    f["adr7_rel"] = np.log(f["adr7"] / f["adr14"])
    f["adr28_rel"] = np.log(f["adr28"] / f["adr14"])
    f["vol_regime"] = np.log(f["adr14"] / f["adr126"])
    f["dow"] = f.index.dayofweek

    # targets, and how far the rest of the session went over the last 14 sessions
    f["up"] = np.log(f["rest_high"] / f["close3"]) / f["adr14"]
    f["down"] = np.log(f["close3"] / f["rest_low"]) / f["adr14"]
    f["up_hist14"] = f["up"].rolling(14).mean().shift(1)
    f["down_hist14"] = f["down"].rolling(14).mean().shift(1)

    if positioning is not None:
        f = f.join(options_features(positioning, profile, f))

    # last, so every group above could read the unscaled values
    for col in _PER_ADR:
        f[col] = f[col] / f["adr14"]
    return f


# --- options (mirrors highlow3m.options) -------------------------------------------
#
# Cut: the downloaders (replaced by the cache below), the calibration search
# (its answer is `NOTEBOOK_OPTIONS`) and the warm-up blanking of the first 40
# sessions of option history, which only the notebook's start of data has --
# this cache always reaches a contract's whole 120-day life.

OPTION_MAX_DTE = 120             # contracts are followed for their last 120 days
OPTION_STRIKE_BAND = (0.6, 1.5)  # strikes kept, as multiples of the underlying price
OPTION_HISTORY_START = "2024-01-01"  # config.HISTORY_START: no contract day before it
POSITIONING_MAX_DTE = 60     # calendar days; further out adds little gamma
SHORT_DTE = 7                # the near-term walls
GEX_GRID = np.round(np.arange(-0.08, 0.08 + 1e-9, 0.0025), 4)  # log moves of spot for the profile
IV_BOUNDS = (0.03, 3.0)
SMILE_MAX_K = 0.25           # |log-moneyness| used to fit a smile
SMILE_MIN_VOLUME = 5         # contracts traded for a point to count
SIZE_CAP = 100  # contracts per trade, for the size-weighted flow

# What the notebook settled per ticker, which no saved file records:
# `config._TICKERS` (the dividend yield the options are priced with, and the
# first date the chain can be trusted after a split) and the open-interest
# proxy `options.calibrate` chose on the expirations to the end of 2025
# (`data/<T>/options/oi_calibration.parquet`, the `chosen` row). `scale` turns
# the proxy into contracts, which dealer gamma against turnover
# (`gex_open_adv`) reads -- so it matters to the forecast, to the last digit.
# A ticker is forecast only with an entry here.
NOTEBOOK_OPTIONS: "dict[str, dict]" = {
    "AAPL": {
        "dividend_yield": 0.004,
        "options_start": None,
        "half_life": math.inf,
        "weighting": "size",
        "scale": 0.1077413779174071,
    },
}


def _holidays() -> np.ndarray:
    return np.array(NYSE_HOLIDAYS, dtype="datetime64[D]")


def _d64(dates) -> np.ndarray:
    return pd.DatetimeIndex(dates).to_numpy().astype("datetime64[D]")


def sessions_between(start, end) -> np.ndarray:
    """NYSE sessions in (start, end]. From a Friday, a Monday expiry is one session away.
    (The notebook adds a bare 1 to the day dates, which NumPy deprecates; one
    day, said in days, is the same number.)"""
    day = np.timedelta64(1, "D")
    return np.busday_count(_d64(start) + day, _d64(end) + day, holidays=_holidays())


def bar_window(expiration, through=None) -> "tuple[pd.Timestamp, pd.Timestamp]":
    """Dates a contract is followed for: its last OPTION_MAX_DTE days, up to
    `through` (the notebook's HISTORY_END; here, the last close asked about)."""
    expiration = pd.Timestamp(expiration)
    start = max(pd.Timestamp(OPTION_HISTORY_START), expiration - pd.Timedelta(OPTION_MAX_DTE, "D"))
    return start, expiration if through is None else min(expiration, pd.Timestamp(through))


def load_bars(bars: pd.DataFrame, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """Every option bar dated on a session (Alpaca stamps a few on closed days)."""
    bars = bars[bars["date"].isin(sessions)]
    return bars.astype({"volume": "int64", "trades": "int64"}).reset_index(drop=True)


def bsm_price(spot, strike, t, r, q, vol, is_call):
    from scipy.special import ndtr

    d1 = (np.log(spot / strike) + (r - q + 0.5 * vol**2) * t) / (vol * np.sqrt(t))
    d2 = d1 - vol * np.sqrt(t)
    call = spot * np.exp(-q * t) * ndtr(d1) - strike * np.exp(-r * t) * ndtr(d2)
    put = strike * np.exp(-r * t) * ndtr(-d2) - spot * np.exp(-q * t) * ndtr(-d1)
    return np.where(is_call, call, put)


def bsm_gamma(spot, strike, t, r, q, vol):
    """Gamma per share; the same for a call and a put."""
    d1 = (np.log(spot / strike) + (r - q + 0.5 * vol**2) * t) / (vol * np.sqrt(t))
    return np.exp(-q * t - 0.5 * d1**2) / (np.sqrt(2 * np.pi) * spot * vol * np.sqrt(t))


def implied_vol(price, spot, strike, t, r, q, is_call, iters: int = 40):
    """Bisection on the Black-Scholes-Merton price; NaN where no vol inside IV_BOUNDS fits."""
    lo = np.full(len(price), IV_BOUNDS[0])
    hi = np.full(len(price), IV_BOUNDS[1])
    fits = (bsm_price(spot, strike, t, r, q, lo, is_call) <= price) & (price <= bsm_price(spot, strike, t, r, q, hi, is_call))
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        above = bsm_price(spot, strike, t, r, q, mid, is_call) > price
        hi, lo = np.where(above, mid, hi), np.where(above, lo, mid)
    return np.where(fits, 0.5 * (lo + hi), np.nan)


def risk_free(irx: pd.Series, dates: pd.DatetimeIndex) -> pd.Series:
    """13-week T-bill yield as a decimal, as of each date's close. `irx` is
    Yahoo's ^IRX close, in percent (the notebook reads it inside)."""
    irx = irx / 100
    return irx.reindex(irx.index.union(dates)).ffill().bfill().reindex(dates)


def contract_days(contracts: pd.DataFrame, bars: pd.DataFrame, closes: pd.Series, q: float,
                  rates: pd.Series) -> pd.DataFrame:
    """One row per contract and session, from its first trade to expiry (or the last session).

    Days without a bar have volume 0 and no price. `t` is the year fraction to the
    expiry close and `k` the log-moneyness against the forward. (`rates` is
    `risk_free(days)`, handed in rather than read inside.)
    """
    days = closes.index
    first = bars.groupby("symbol")["date"].min()
    meta = contracts.set_index("symbol").reindex(first.index)
    first, meta = first[meta["type"].notna()], meta[meta["type"].notna()]
    start = days.searchsorted(first.to_numpy())
    end = days.searchsorted(np.minimum(meta["expiration"].to_numpy(), days[-1].to_datetime64()), side="right") - 1
    length = np.maximum(end - start + 1, 0)
    which = np.repeat(np.arange(len(first)), length)
    n = np.repeat(start, length) + np.arange(length.sum()) - np.repeat(np.cumsum(length) - length, length)

    p = pd.DataFrame({
        "symbol": first.index.to_numpy()[which], "date": days[n], "n": n,
        "type": meta["type"].to_numpy()[which], "strike": meta["strike"].to_numpy()[which],
        "expiration": meta["expiration"].to_numpy()[which],
    })
    p = p.merge(bars[["symbol", "date", "close", "volume", "trades"]], on=["symbol", "date"], how="left")
    p[["volume", "trades"]] = p[["volume", "trades"]].fillna(0).astype("int64")
    p["spot"] = closes.to_numpy()[n]
    p["rate"] = rates.to_numpy()[n]
    p["q"] = q
    p["dte"] = (p["expiration"] - p["date"]).dt.days
    p["t"] = p["dte"] / 365
    p["k"] = np.log(p["strike"] / (p["spot"] * np.exp((p["rate"] - q) * p["t"])))

    # implied vol where the contract traded and its price is worth more than intrinsic
    is_call = (p["type"] == "call").to_numpy()
    intrinsic = np.where(is_call, p["spot"] * np.exp(-q * p["t"]) - p["strike"] * np.exp(-p["rate"] * p["t"]),
                         p["strike"] * np.exp(-p["rate"] * p["t"]) - p["spot"] * np.exp(-q * p["t"])).clip(min=0)
    ok = ((p["volume"] > 0) & (p["t"] > 0) & (p["close"] > intrinsic + 0.01)).to_numpy()
    p["iv"] = np.nan
    cols = [p[c].to_numpy()[ok] for c in ("close", "spot", "strike", "t", "rate")]
    p.loc[ok, "iv"] = implied_vol(*cols, q, is_call[ok])
    return p.sort_values(["symbol", "date"]).reset_index(drop=True)


def fit_smiles(panel: pd.DataFrame) -> pd.DataFrame:
    """Implied vol against log-moneyness for each (date, expiration), from out-of-the-money trades.

    In-the-money options are left out: American puts carry an early-exercise premium
    and deep in-the-money prints are often stale. Weighted by the square root of
    volume. Quadratic with six points and two on each side, else linear, else flat.
    """
    otm = np.where(panel["type"] == "call", panel["k"] >= 0, panel["k"] <= 0)
    pts = panel[otm & panel["iv"].notna() & (panel["volume"] >= SMILE_MIN_VOLUME)
                & (panel["k"].abs() <= SMILE_MAX_K)]
    rows = []
    for (date_, expiration), g in pts.groupby(["date", "expiration"], sort=False):
        k, iv = g["k"].to_numpy(), g["iv"].to_numpy()
        w = g["volume"].to_numpy(float) ** 0.25  # polyfit squares the weight
        if len(g) >= 6 and (k < 0).sum() >= 2 and (k > 0).sum() >= 2:
            c, b, a = np.polyfit(k, iv, 2, w=w)
        elif len(g) >= 2 and np.ptp(k) >= 0.01:
            (b, a), c = np.polyfit(k, iv, 1, w=w), 0.0
        else:
            a, b, c = np.average(iv, weights=w**2), 0.0, 0.0
        rows.append((date_, expiration, a, b, c, k.min(), k.max(), len(g)))
    smiles = pd.DataFrame(rows, columns=["date", "expiration", "a", "b", "c", "kmin", "kmax", "n_points"])
    smiles["dte"] = (smiles["expiration"] - smiles["date"]).dt.days
    return smiles.sort_values(["date", "expiration"]).reset_index(drop=True)


def smile_iv(smile: pd.DataFrame, k) -> np.ndarray:
    """Fitted vol at log-moneyness k, flat beyond the range the smile was fitted on."""
    kc = np.clip(k, smile["kmin"].to_numpy(), smile["kmax"].to_numpy())
    return np.clip(smile["a"].to_numpy() + smile["b"].to_numpy() * kc + smile["c"].to_numpy() * kc**2, 0.05, 3.0)


def add_fitted_iv(panel: pd.DataFrame, smiles: pd.DataFrame) -> pd.DataFrame:
    """`iv_fit` for every live contract-day. An expiration with no smile that day borrows
    the smile of the expiration nearest in days to expiry."""
    params = ["a", "b", "c", "kmin", "kmax"]
    live = panel.loc[panel["t"] > 0, ["date", "expiration", "dte"]].drop_duplicates()
    own = live.merge(smiles[["date", "expiration", *params]], on=["date", "expiration"], how="left")
    missing = own[own["a"].isna()].drop(columns=params).sort_values("dte")
    donors = smiles[["date", "dte", *params]].sort_values("dte")
    borrowed = pd.merge_asof(missing, donors, on="dte", by="date", direction="nearest")
    assigned = pd.concat([own[own["a"].notna()], borrowed], ignore_index=True).drop(columns="dte")
    out = panel.merge(assigned, on=["date", "expiration"], how="left")
    out["iv_fit"] = np.where(out["a"].notna(), smile_iv(out, out["k"].to_numpy()), np.nan)
    return out.drop(columns=params)


def daily_flow(panel: pd.DataFrame, weighting: str) -> np.ndarray:
    """What one day adds to a contract's open interest.

    "volume" counts contracts traded. "size" weights them by the square root of the
    average trade size, capped: big prints tend to open positions that stay on, single
    lots are more often closed the same day.
    """
    volume = panel["volume"].to_numpy(float)
    if weighting == "volume":
        return volume
    trades = panel["trades"].to_numpy(float)
    size = np.divide(volume, trades, out=np.zeros_like(volume), where=trades > 0)
    return volume * np.sqrt(np.minimum(size, SIZE_CAP))


def oi_proxy(panel: pd.DataFrame, half_life: float, weighting: str) -> np.ndarray:
    """Per contract, the sum of past flow fading with `half_life` sessions. Panel sorted by symbol, date."""
    flow = pd.Series(daily_flow(panel, weighting), index=panel.index)
    if np.isinf(half_life):
        return flow.groupby(panel["symbol"]).cumsum().to_numpy()
    # rescaled by each contract's own age so the running sum cannot overflow
    age = (panel["n"] - panel.groupby("symbol")["n"].transform("first")).to_numpy()
    decay = 0.5 ** (1 / half_life)
    return (flow * decay ** (-age)).groupby(panel["symbol"]).cumsum().to_numpy() * decay**age


def scaled_oi(panel: pd.DataFrame, calibration: dict) -> np.ndarray:
    """The chosen proxy, in contracts of open interest."""
    return calibration["scale"] * oi_proxy(panel, calibration["half_life"], calibration["weighting"])


def dollar_gamma(df: pd.DataFrame, spot: np.ndarray) -> np.ndarray:
    """Dealer dollar gamma for a 1% move at `spot`: + for calls, - for puts."""
    g = bsm_gamma(spot, df["strike"].to_numpy(), df["t"].to_numpy(), df["rate"].to_numpy(),
                  df["q"].to_numpy(), df["iv_fit"].to_numpy())
    sign = np.where(df["type"] == "call", 1.0, -1.0)
    return np.nan_to_num(sign * g * df["oi"].to_numpy() * 100 * spot**2 * 0.01)


def wall(uni: pd.DataFrame, value: str, kind: str, max_dte: "int | None" = None) -> pd.Series:
    """Per date, the strike with the most `value`: calls above spot, or puts below it."""
    side = uni[uni["type"] == kind]
    side = side[side["strike"] > side["spot"]] if kind == "call" else side[side["strike"] < side["spot"]]
    if max_dte is not None:
        side = side[side["dte"] <= max_dte]
    by_strike = side.groupby(["date", "strike"])[value].sum().abs()
    by_strike = by_strike[by_strike > 0].reset_index()
    return by_strike.loc[by_strike.groupby("date")[value].idxmax()].set_index("date")["strike"]


def gex_profile(uni: pd.DataFrame) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """Net and gross dealer gamma per date if spot moved by exp(m), vols and time held fixed."""
    codes, dates = pd.factorize(uni["date"], sort=True)
    spot = uni["spot"].to_numpy()
    net = np.empty((len(dates), len(GEX_GRID)))
    gross = np.empty_like(net)
    for j, m in enumerate(GEX_GRID):
        dg = dollar_gamma(uni, spot * np.exp(m))
        net[:, j] = np.bincount(codes, weights=dg, minlength=len(dates))
        gross[:, j] = np.bincount(codes, weights=np.abs(dg), minlength=len(dates))
    index = pd.DatetimeIndex(dates, name="date")
    return pd.DataFrame(net, index=index, columns=GEX_GRID), pd.DataFrame(gross, index=index, columns=GEX_GRID)


def gamma_flip(net: pd.DataFrame, spot: pd.Series) -> pd.Series:
    """The spot level nearest the close where net dealer gamma changes sign; NaN if none on the grid."""
    v, m = net.to_numpy(), net.columns.to_numpy(float)
    left, right = v[:, :-1], v[:, 1:]
    with np.errstate(invalid="ignore", divide="ignore"):
        cross = m[:-1] - left * (m[1:] - m[:-1]) / (right - left)
    cross = np.where(left * right < 0, cross, np.nan)
    nearest = np.full(len(v), np.nan)
    has = ~np.isnan(cross).all(axis=1)
    if has.any():
        pick = np.nanargmin(np.abs(cross[has]), axis=1)
        nearest[has] = cross[has][np.arange(has.sum()), pick]
    return pd.Series(spot.reindex(net.index).to_numpy() * np.exp(nearest), index=net.index)


def _interp_vol(x: np.ndarray, vol: np.ndarray, target: float) -> float:
    """Vol at time `target`, linear in total variance between expirations, flat outside them."""
    order = np.argsort(x)
    x, vol = x[order], vol[order]
    if target <= x[0]:
        return vol[0]
    if target >= x[-1]:
        return vol[-1]
    return float(np.sqrt(np.interp(target, x, vol**2 * x) / target))


def implied_moves(smiles: pd.DataFrame) -> pd.DataFrame:
    """At-the-money vol one session ahead and 30 calendar days ahead.

    `iv_1d` is measured in session time: iv_1d / sqrt(252) is the options market's
    expected size of the next session's move. Fixed tenors rather than "the nearest
    expiry", because the listed expiries changed over the window (AAPL added Monday
    and Wednesday expiries in 2026).
    """
    s = smiles[smiles["dte"] > 0].copy()
    s["atm"] = smile_iv(s, 0.0)
    s["x_sessions"] = sessions_between(s["date"], s["expiration"]) / 252
    # the vol per year of sessions that gives the same total variance
    s["atm_sessions"] = s["atm"] * np.sqrt((s["dte"] / 365) / s["x_sessions"].where(s["x_sessions"] > 0))
    rows = {}
    for date_, g in s.groupby("date"):
        g1 = g[g["x_sessions"] > 0]
        rows[date_] = {
            "iv_1d": _interp_vol(g1["x_sessions"].to_numpy(), g1["atm_sessions"].to_numpy(), 1 / 252) if len(g1) else np.nan,
            "iv_30d": _interp_vol(g["dte"].to_numpy() / 365, g["atm"].to_numpy(), 30 / 365),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def next_is_monthly_opex(sessions: pd.DatetimeIndex) -> pd.Series:
    """Whether the session after each date is the monthly expiry (the third Friday, or the
    session before it when that Friday is a holiday). Calendar knowledge, known years ahead."""
    following = np.busday_offset(_d64(sessions), 1, roll="forward", holidays=_holidays())
    months = pd.date_range(sessions.min().replace(day=1), sessions.max() + pd.Timedelta(40, "D"), freq="MS")
    third_friday = months + pd.to_timedelta((4 - months.dayofweek) % 7 + 14, unit="D")
    opex = np.busday_offset(_d64(third_friday), 0, roll="backward", holidays=_holidays())
    return pd.Series(np.isin(following, opex), index=sessions)


def positioning_tables(panel: pd.DataFrame, smiles: pd.DataFrame, closes: pd.Series,
                       options_start: "str | None" = None) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """One row per close with walls, gamma, the flip, implied moves and activity; and the gamma profile.

    `panel` must carry `oi` and `iv_fit`. Row d is the close of d. (Less the
    notebook's warm-up blanking -- see the block comment.)
    """
    days = closes.index
    # A day without a single option bar (2 Feb 2024 at Alpaca) would read as zero gamma;
    # it stays empty instead.
    traded = panel.loc[panel["volume"] > 0, "date"].unique()
    # The download kept strikes by the price range over each contract's life, which runs
    # past d. Re-banding on the day's own close makes the set point-in-time.
    lo, hi = OPTION_STRIKE_BAND
    in_band = panel["strike"].between(lo * panel["spot"], hi * panel["spot"])
    live = (panel["dte"] > 0) & (panel["dte"] <= POSITIONING_MAX_DTE)
    uni = panel[in_band & live & panel["date"].isin(traded)].copy()
    uni["dgamma"] = dollar_gamma(uni, uni["spot"].to_numpy())
    is_call = uni["type"] == "call"

    out = pd.DataFrame(index=pd.DatetimeIndex(days, name="date"))
    out["spot"] = closes
    out["call_wall"] = wall(uni, "oi", "call")
    out["put_wall"] = wall(uni, "oi", "put")
    out["call_wall_7d"] = wall(uni, "oi", "call", SHORT_DTE)
    out["put_wall_7d"] = wall(uni, "oi", "put", SHORT_DTE)
    out["call_gamma_wall"] = wall(uni, "dgamma", "call")
    out["put_gamma_wall"] = wall(uni, "dgamma", "put")
    out["net_gex"] = uni.groupby("date")["dgamma"].sum()
    out["gross_gex"] = uni["dgamma"].abs().groupby(uni["date"]).sum()
    out["gex_ratio"] = out["net_gex"] / out["gross_gex"].replace(0, np.nan)
    net, gross = gex_profile(uni)
    out["gamma_flip"] = gamma_flip(net, out["spot"])
    out = out.join(implied_moves(smiles))

    # activity counts every in-band contract, those expiring that day included
    traded_today = panel[in_band]
    out["call_volume"] = traded_today.loc[traded_today["type"] == "call"].groupby("date")["volume"].sum()
    out["put_volume"] = traded_today.loc[traded_today["type"] == "put"].groupby("date")["volume"].sum()
    out["call_oi"] = uni.loc[is_call].groupby("date")["oi"].sum()
    out["put_oi"] = uni.loc[~is_call].groupby("date")["oi"].sum()
    out["next_is_opex"] = next_is_monthly_opex(days)

    young = pd.Series(False, index=out.index)
    if options_start is not None:
        young |= out.index < pd.Timestamp(options_start)
    cols = [c for c in out.columns if c not in ("spot", "next_is_opex")]
    out.loc[young, cols] = np.nan

    profile = pd.concat({"net_gex": net.stack(), "gross_gex": gross.stack()}, axis=1)
    profile = profile.rename_axis(["date", "m"]).reset_index()
    profile = profile[~profile["date"].isin(out.index[young])]
    return out, profile.reset_index(drop=True)


# --- session closes (mirrors highlow3m.data.session_closes) ---------------------------

def _clean_regular(raw: pd.DataFrame) -> pd.DataFrame:
    """`highlow3m.data.clean_regular`: sane bars of the regular session, with session columns."""
    framed = add_session_columns(raw)
    framed = framed[sane_bars(framed) & framed["minute"].between(0, SESSION_MINUTES - 1)].copy()
    framed["volume"] = framed["volume"].astype("int64")
    return framed


def session_closes(raw_sip: pd.DataFrame) -> pd.Series:
    """The underlying's close on every NYSE session in the tape, half days included.

    Options trade on half days even though the models leave them out, so the option
    tables need their close: the 12:59 bar, which ends at the 13:00 close. (The
    notebook's half-day list ends with 2025; `early_close` is the rule that
    generates it, identical on every date the list covers.)
    """
    bars = _clean_regular(raw_sip)
    holidays = pd.DatetimeIndex(pd.to_datetime(list(NYSE_HOLIDAYS)))
    bars = bars[~bars["date"].isin(holidays) & (bars.index.dayofweek < 5)]
    half = early_close(pd.DatetimeIndex(bars["date"]))
    bars = bars[~half | (bars["minute"] <= 12 * 60 + 59 - OPEN_MINUTE)]
    close = bars.groupby("date")["close"].last()
    close.index.name = "date"
    return close.rename("close")


# --- models (mirrors the inference half of highlow3m.models) ----------------------------

def to_prices(pred: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    """Predicted excursions (ADRs) to the predicted high and low after 9:33, in dollars."""
    scale = frame["adr14"].to_numpy()
    close3 = frame["close3"].to_numpy()
    return pd.DataFrame({
        "pred_high": close3 * np.exp(pred["up"].to_numpy() * scale),
        "pred_low": close3 * np.exp(-pred["down"].to_numpy() * scale),
    }, index=frame.index)


class LGBMPair:
    """One LightGBM per target and seed; predictions averaged over seeds (unpickled)."""

    kind = "lgbm"

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        out = {t: np.mean([self.models[t, s].predict(X[self.cols]) for s in self.seeds], axis=0) for t in TARGETS}
        return pd.DataFrame(out, index=X.index)


class LinearMedian:
    """Median regression on standardised features, one per target (unpickled).

    The notebook's `predict` is `self.models[t].predict(X[self.cols])` on a
    `SimpleImputer(median) -> StandardScaler -> QuantileRegressor` pipeline
    pickled by scikit-learn 1.7.2, which 1.9's `SimpleImputer.transform` cannot
    run. The same three steps from their fitted arrays instead: fill a missing
    value with the training median, standardise, take the linear median.
    """

    kind = "linear"

    def _arrays(self, target: str) -> tuple:
        cached = self.__dict__.setdefault("_fitted", {})
        if target not in cached:
            cached[target] = linear_arrays(self.models[target])
        return cached[target]

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        x = X[self.cols].to_numpy(dtype=float)
        out = {}
        for t in TARGETS:
            fill, mean, scale, coef, intercept = self._arrays(t)
            z = (np.where(np.isnan(x), fill, x) - mean) / scale
            out[t] = z @ coef + intercept
        return pd.DataFrame(out, index=X.index)


def linear_arrays(pipeline) -> tuple:
    """(fill, mean, scale, coef, intercept) out of a fitted imputer -> scaler ->
    quantile-regression pipeline, or ValueError for any other shape of it."""
    steps = [step for _, step in getattr(pipeline, "steps", [])]
    names = [type(step).__name__ for step in steps]
    if names != ["SimpleImputer", "StandardScaler", "QuantileRegressor"]:
        raise ValueError(f"unexpected linear pipeline {names}")
    imputer, scaler, regressor = steps
    if (getattr(imputer, "strategy", None) != "median" or getattr(imputer, "add_indicator", False)
            or not (isinstance(imputer.missing_values, float) and math.isnan(imputer.missing_values))):
        raise ValueError("the imputer is not a plain median fill of NaN")
    if not (scaler.with_mean and scaler.with_std):
        raise ValueError("the scaler does not centre and scale")
    fill = np.asarray(imputer.statistics_, dtype=float)
    if np.isnan(fill).any():
        raise ValueError("the imputer has no median for some column")
    return (fill, np.asarray(scaler.mean_, dtype=float), np.asarray(scaler.scale_, dtype=float),
            np.asarray(regressor.coef_, dtype=float), float(regressor.intercept_))


class HighLowModel:
    """A weighted blend of fitted components, and everything needed to use it (unpickled)."""

    def predict_excursions(self, frame: pd.DataFrame) -> pd.DataFrame:
        parts = [self.weights[k] * m.predict(frame) for k, m in self.components.items() if self.weights[k] > 0]
        return sum(parts) / sum(w for w in self.weights.values() if w > 0)

    def predict_prices(self, frame: pd.DataFrame) -> pd.DataFrame:
        return to_prices(self.predict_excursions(frame), frame)


# --- the saved bundle --------------------------------------------------------------

def _register_unpickle_alias() -> None:
    """Make `highlow3m.models.{HighLowModel,LGBMPair,LinearMedian}` resolvable
    for joblib.

    The real package pulls requests, yfinance and the notebook's data paths in,
    so a stub pointing at the mirrors above stands in -- unless the real package
    is genuinely importable, in which case it wins.
    """
    if "highlow3m.models" in sys.modules:
        return
    try:
        __import__("highlow3m.models")
        return
    except Exception:
        sys.modules.pop("highlow3m", None)
    package = types.ModuleType("highlow3m")
    package.__path__ = []
    module = types.ModuleType("highlow3m.models")
    module.HighLowModel = HighLowModel
    module.LGBMPair = LGBMPair
    module.LinearMedian = LinearMedian
    package.models = module
    sys.modules["highlow3m"] = package
    sys.modules["highlow3m.models"] = module


_STORE = ModelStore(
    env_key=MODEL_PATH_ENV,
    filename="highlow3m_{ticker}.joblib",
    build=lambda path: _build_bundle(path),
)

model_path = _STORE.path
metadata_path = _STORE.metadata_path


def _build_bundle(path: Path) -> "dict | None":
    """One ticker's saved model plus its metadata, or None when it cannot be
    assembled.

    Refuses rather than degrades when the shipped blend needs something this
    module does not mirror: a component other than the LightGBM pair and the
    linear median, a column outside `FEATURE_COLS`, an opening other than
    IEX's first three minutes, an option-reading model on a ticker whose
    notebook settings are not in `NOTEBOOK_OPTIONS`, or a linear part whose
    fitted arrays cannot be read.
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
    if not all(hasattr(model, a) for a in ("components", "weights", "feature_cols")):
        return None

    weights = {k: float(w) for k, w in dict(model.weights).items()}
    shipped = {k: m for k, m in dict(model.components).items() if weights.get(k, 0.0) > 0}
    if not shipped:
        return None
    read = set(model.feature_cols)
    for m in shipped.values():
        if getattr(m, "kind", None) not in ("lgbm", "linear"):
            return None
        read |= set(getattr(m, "cols", None) or ())
        if m.kind == "linear":
            try:
                for t in TARGETS:
                    m._arrays(t)
            except (AttributeError, KeyError, TypeError, ValueError):
                return None
    if read - set(FEATURE_COLS):
        return None

    meta_file = metadata_path(path)
    try:
        metadata = json.loads(meta_file.read_text()) if meta_file.exists() else {}
    except (OSError, ValueError):
        metadata = {}
    if int(metadata.get("opening_minutes") or OPENING_MINUTES) != OPENING_MINUTES:
        return None
    if str(metadata.get("opening_feed") or OPENING_FEED) != OPENING_FEED:
        return None
    ticker = str(metadata.get("ticker") or "").upper() or None
    if read & OPTION_COLS and (ticker or DEFAULT_TICKER) not in NOTEBOOK_OPTIONS:
        return None
    # Only the shipped components, so a weight-0 one is never asked to predict.
    model.components = shipped
    model.weights = {k: weights[k] for k in shipped}
    return {
        "kind": "highlow3m",
        "model": model,
        "metadata": metadata,
        "daily_models": sorted(shipped),
        "opening_minutes": OPENING_MINUTES,
        "opening_feed": OPENING_FEED,
        "min_bars": MIN_BARS_PER_SESSION,
        "reads_options": bool(read & OPTION_COLS),
        "trained_at": metadata.get("created"),
        "path": str(path),
        "ticker": ticker,
    }


load_bundle = _STORE.load
reset_bundle_cache = _STORE.reset


def opening_minutes(bundle: "dict | None" = None) -> int:
    try:
        return int((bundle or {})["opening_minutes"])
    except (KeyError, TypeError, ValueError):
        return OPENING_MINUTES


def opening_feed(bundle: "dict | None" = None) -> str:
    """The tape the bundle reads this morning from -- IEX."""
    return str((bundle or {}).get("opening_feed") or OPENING_FEED)


def min_bars(bundle: "dict | None" = None) -> int:
    try:
        return int((bundle or {})["min_bars"])
    except (KeyError, TypeError, ValueError):
        return MIN_BARS_PER_SESSION


# --- the history: SIP sessions, IEX openings, session closes ------------------------------

# A kept session's row: `daily_bars`' columns, the SIP extremes after 9:33 (the
# targets `up_hist14` / `down_hist14` average), and the IEX opening summary
# (missing on a session IEX printed nothing in the window of).
_ROLLUP_COLS = ["open", "high", "low", "close", "volume", "rv"]
_REST_COLS = ["rest_high", "rest_low"]
_OPENING_COLS = [
    "open3", "high3", "low3", "close3", "volume3", "bars3",
    "or_up", "or_down", "or_ret", "or_range", "or_close_pos", "or_bar_std", "or_last_ret",
]
_SESSION_COLS = _ROLLUP_COLS + _REST_COLS + _OPENING_COLS


@dataclass
class History:
    """What the cache knows about a stretch of days.

    `sessions` is one row per session SIP kept (`_SESSION_COLS`); `closes` the
    SIP close of every session it printed, dropped ones and half days included
    (what the option tables price off); `dropped` the sessions SIP refused.
    """

    sessions: pd.DataFrame
    closes: pd.Series
    dropped: list = field(default_factory=list)

    def before(self, day) -> "History":
        """The same, cut to the dates strictly before `day`."""
        day = pd.Timestamp(day)
        return History(
            self.sessions[self.sessions.index < day],
            self.closes[self.closes.index < day],
            [d for d in self.dropped if pd.Timestamp(d) < day],
        )


def _dates(values) -> pd.DatetimeIndex:
    """ISO dates read back from a cache as a session index. In nanoseconds,
    as the tapes' own session dates are: pandas 3 parses strings to
    microseconds, and `merge_asof` refuses to join the two."""
    return pd.DatetimeIndex(pd.to_datetime(list(values)), name="date").as_unit("ns")


def _empty(cols: "list[str]") -> pd.DataFrame:
    return pd.DataFrame(columns=cols, index=pd.DatetimeIndex([], name="date"), dtype=float)


def _empty_closes() -> pd.Series:
    return pd.Series([], index=pd.DatetimeIndex([], name="date"), dtype=float, name="close")


def stretch_from(sip: pd.DataFrame, iex: pd.DataFrame,
                 min_bars: int = MIN_BARS_PER_SESSION) -> History:
    """Everything the cache keeps about a stretch of days, from that stretch's
    SIP and IEX tapes (exchange-local OHLCV; extended hours are ignored).

    Every step groups by date, so a stretch summarised in pieces gives the same
    rows as the whole of it at once -- which is what lets the cache grow a day
    at a time.
    """
    sessions, closes, dropped = _empty(_SESSION_COLS), _empty_closes(), []
    rth = sip.between_time(RTH_START, RTH_END) if len(sip) else sip
    if len(rth):
        closes = session_closes(rth[OHLCV])
        minute, report = clean_minute(rth[OHLCV], min_bars)
        dropped = list(report.index[~report["kept"]])
        if len(minute):
            daily = daily_bars(minute)
            rest = rest_of_session(minute)
            iex_rth = iex.between_time(RTH_START, RTH_END) if len(iex) else iex
            thin = trim_to_sessions(iex_rth[OHLCV], daily.index) if len(iex_rth) else None
            opening = (
                opening_features(thin)[_OPENING_COLS] if thin is not None and len(thin)
                else _empty(_OPENING_COLS)
            )
            sessions = daily[_ROLLUP_COLS].join(rest[_REST_COLS]).join(opening, how="left")
    return History(sessions, closes, dropped)


_history_lock = threading.Lock()


def history_path(symbol: str) -> Path:
    return HISTORY_DIR / f"{symbol.upper()}_sessions.json"


def _empty_cache(min_bars: int = MIN_BARS_PER_SESSION) -> dict:
    return {"layout": CACHE_LAYOUT, "min_bars": min_bars, "from": None, "through": None,
            "sessions": {}, "closes": {}, "dropped": []}


def _read_json(path: Path) -> "dict | None":
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(path)


def _read_cache(symbol: str, min_bars: int = MIN_BARS_PER_SESSION) -> dict:
    """The symbol's cache, or an empty one if it is unreadable, in another
    layout, or cleaned at another bar count."""
    payload = _read_json(history_path(symbol))
    if not payload or payload.get("layout") != CACHE_LAYOUT or int(payload.get("min_bars", -1)) != min_bars:
        return _empty_cache(min_bars)
    for key in ("sessions", "closes"):
        payload.setdefault(key, {})
    payload.setdefault("dropped", [])
    return payload


def _write_cache(symbol: str, payload: dict) -> None:
    _write_json(history_path(symbol), payload)


def _credentials(key: "str | None", secret: "str | None") -> "tuple[str, str]":
    """The caller's Alpaca keys, else the environment's (SimLab has no sidebar)."""
    key = key or os.getenv("ALPACA_API_KEY", "")
    secret = secret or os.getenv("ALPACA_SECRET", "")
    if not (key and secret):
        raise ValueError(
            "the HighLow_3m forecast needs Alpaca SIP, IEX and option history and no Alpaca "
            "credentials are available (sidebar connection, or ALPACA_API_KEY / "
            "ALPACA_SECRET in the environment)."
        )
    return key, secret


def _fetch_stretch(symbol: str, first: date, before: date, key: str, secret: str,
                   min_bars: int = MIN_BARS_PER_SESSION) -> History:
    """`stretch_from` for the calendar days in [first, before), straight from
    Alpaca: SIP and IEX, split-adjusted."""
    from .datalog import log_fetch

    tz = market_hours.MARKET_TZ
    start = datetime.combine(first, datetime.min.time(), tzinfo=tz)
    end = datetime.combine(before, datetime.min.time(), tzinfo=tz)
    sip = _fetch_tape(symbol, start, end, key, secret, "sip")
    iex = _fetch_tape(symbol, start, end, key, secret, OPENING_FEED)
    out = stretch_from(sip, iex, min_bars)
    log_fetch(
        "minute bars (HighLow_3m history)",
        "Alpaca REST (SIP + IEX, split-adjusted)",
        symbol=symbol,
        detail=f"{len(sip)} SIP + {len(iex)} IEX bars, {len(out.sessions)} sessions from {first}",
    )
    return out


def _rows(frame: pd.DataFrame) -> dict:
    return {
        d.strftime("%Y-%m-%d"): {c: float(frame.at[d, c]) for c in frame.columns}
        for d in frame.index
    }


def _frame(rows: dict, cols: "list[str]", lo: str, hi: str) -> pd.DataFrame:
    """Cached rows dated in [lo, hi) as a frame, oldest first."""
    rows = {d: v for d, v in rows.items() if lo <= d < hi}
    if not rows:
        return _empty(cols)
    frame = pd.DataFrame.from_dict(rows, orient="index")
    frame.index = _dates(frame.index)
    return frame.sort_index().reindex(columns=cols)


def _series(values: dict, lo: str, hi: str) -> pd.Series:
    values = {d: v for d, v in values.items() if lo <= d < hi}
    if not values:
        return _empty_closes()
    out = pd.Series(values, dtype=float, name="close")
    out.index = _dates(out.index)
    return out.sort_index()


def _merge(cache: dict, fresh: History, seam: "str | None") -> bool:
    """Fold a fetch into the cache. False (and nothing merged) if the session
    both hold -- `seam` -- disagrees, which means a split rescaled the tape."""
    sessions = _rows(fresh.sessions)
    if seam and seam in cache["sessions"] and seam in sessions:
        if abs(sessions[seam]["close"] / cache["sessions"][seam]["close"] - 1.0) > 1e-6:
            return False
    cache["sessions"].update(sessions)
    cache["closes"].update({d.strftime("%Y-%m-%d"): float(v) for d, v in fresh.closes.items()})
    cache["dropped"] = sorted(
        set(cache["dropped"]) | {pd.Timestamp(d).strftime("%Y-%m-%d") for d in fresh.dropped}
    )
    return True


def _history_from_cache(cache: dict, lo: str, hi: str) -> History:
    return History(
        _frame(cache["sessions"], _SESSION_COLS, lo, hi),
        _series(cache["closes"], lo, hi),
        [d for d in cache["dropped"] if lo <= d < hi],
    )


def history_inputs(
    symbol: str,
    before,
    key: "str | None" = None,
    secret: "str | None" = None,
    min_bars: int = MIN_BARS_PER_SESSION,
) -> History:
    """Everything the forecast reads about the days strictly before `before`,
    reaching HISTORY_CALENDAR_DAYS back.

    `highlow2_model.history_inputs`' scheme: the cache remembers which calendar
    days it has looked at (`from` .. `through`), and only the days outside that
    stretch are fetched -- so a live session fetches one day each morning and a
    replay of a day inside the stretch fetches nothing. Each extension re-reads
    the cached session at its seam, whole: if that completed session's close
    has moved, a split has rescaled the tape, and the cache is rebuilt rather
    than mixed.
    """
    symbol = symbol.upper()
    before = pd.Timestamp(before).date()
    first_wanted = before - timedelta(days=HISTORY_CALENDAR_DAYS)
    last_wanted = before - timedelta(days=1)
    with _history_lock:
        cache = _read_cache(symbol, min_bars)
        lo = date.fromisoformat(cache["from"]) if cache.get("from") else None
        hi = date.fromisoformat(cache["through"]) if cache.get("through") else None
        if lo is None or hi is None or not cache["sessions"]:
            cache, lo, hi = _empty_cache(min_bars), None, None

        if lo is None or lo > first_wanted or hi < last_wanted:
            key, secret = _credentials(key, secret)

            def fetch(first: date, upto: date) -> History:
                return _fetch_stretch(symbol, first, upto, key, secret, min_bars)

            ok = True
            if lo is None or hi < first_wanted or lo > last_wanted:
                # Nothing usable overlaps: fetch the whole window.
                cache = _empty_cache(min_bars)
                _merge(cache, fetch(first_wanted, before), None)
                lo, hi = first_wanted, last_wanted
            else:
                if lo > first_wanted:
                    seam = min(cache["sessions"])
                    upto = date.fromisoformat(seam) + timedelta(days=1)
                    ok = _merge(cache, fetch(first_wanted, upto), seam)
                    lo = first_wanted
                if ok and hi < last_wanted:
                    seam = max(cache["sessions"])
                    ok = _merge(cache, fetch(date.fromisoformat(seam), before), seam)
                    hi = last_wanted
            if not ok:
                cache = _empty_cache(min_bars)
                _merge(cache, fetch(first_wanted, before), None)
                lo, hi = first_wanted, last_wanted
            cache["from"], cache["through"] = lo.isoformat(), hi.isoformat()
            _write_cache(symbol, cache)

    return _history_from_cache(cache, first_wanted.isoformat(), before.isoformat())


# --- this morning ---------------------------------------------------------------

def opening_from_bars(iex: pd.DataFrame, session_date, n_minutes: int = OPENING_MINUTES) -> pd.DataFrame:
    """Today's one-row IEX opening summary from today's IEX bars, cut at the
    9:33 line: bars past it are ignored, so handing it more of the day cannot
    leak the session."""
    day = pd.Timestamp(session_date).normalize()
    if day.tzinfo is not None:
        day = day.tz_localize(None)
    iex_rth = iex.between_time(RTH_START, RTH_END) if len(iex) else iex
    thin = trim_to_sessions(iex_rth[OHLCV], [day]) if len(iex_rth) else None
    if thin is not None:
        thin = thin[thin["minute"] < n_minutes]
    if thin is None or not len(thin):
        raise ValueError(
            f"IEX printed nothing in {day.date()}'s first {n_minutes} minutes, and this "
            "model reads the opening from IEX only; the notebook has no row for such a session."
        )
    return opening_features(thin, n_minutes)[_OPENING_COLS]


def _cached_opening(symbol: str, day: date, min_bars: int = MIN_BARS_PER_SESSION) -> "pd.DataFrame | None":
    """Today's opening out of the cache, when it already covers the day --
    true only of a replay (live, the cache ends yesterday). The same function
    over the same bars as `opening_from_bars`, so this is the fetch without the
    request. None -- fetch it -- for a day the cache has not looked at, or one
    SIP later dropped (the cache keeps no opening for those)."""
    iso = day.isoformat()
    with _history_lock:
        cache = _read_cache(symbol, min_bars)
    if not (cache.get("from") and cache.get("through")) or not cache["from"] <= iso <= cache["through"]:
        return None
    row = cache["sessions"].get(iso)
    if row is None or not np.isfinite(float(row.get("open3", np.nan))):
        return None
    return _frame({iso: row}, _OPENING_COLS, iso, (day + timedelta(days=1)).isoformat())


# Live, the IEX window is fetched the moment the caller's 09:32 bar has closed,
# and Alpaca can publish that minute a moment later (`highlow_model`'s retry).
_WINDOW_RETRIES = 3
_WINDOW_RETRY_SEC = 2.0
_WINDOW_FRESH_SEC = 90


def fetch_opening(symbol: str, session_date, n_minutes: int = OPENING_MINUTES,
                  key: "str | None" = None, secret: "str | None" = None) -> "tuple[pd.DataFrame, bool]":
    """Today's IEX opening summary straight from Alpaca, and whether it is
    settled: False for a window fetched before IEX had published its last
    minute, which is still forecast from (the notebook keeps a short window)
    but not memoised, so a later ask reads the whole one."""
    key, secret = _credentials(key, secret)
    day = pd.Timestamp(session_date).date()
    tz = market_hours.MARKET_TZ
    start = datetime(day.year, day.month, day.day, 9, 30, tzinfo=tz)
    window_end = start + timedelta(minutes=n_minutes)
    for attempt in range(_WINDOW_RETRIES):
        iex = _fetch_tape(symbol, start, window_end, key, secret, OPENING_FEED)
        complete = len(iex) and iex.index[-1] >= window_end - timedelta(minutes=1)
        fresh = datetime.now(timezone.utc) < window_end + timedelta(seconds=_WINDOW_FRESH_SEC)
        if complete or not fresh or attempt == _WINDOW_RETRIES - 1:
            break
        time.sleep(_WINDOW_RETRY_SEC)
    return opening_from_bars(iex, day, n_minutes), bool(complete) or not fresh


# --- the option chain's history: contracts, daily bars, the T-bill yield ------------------

OPTIONS_LAYOUT = 1
_options_lock = threading.Lock()

_OPTION_BARS_URL = "/v1beta1/options/bars"
_CONTRACTS_PATH = "/v2/options/contracts"
# Contracts per bars request (Alpaca's cap on `symbols`), and the pace kept
# between requests: a cold window is ~100 of them, and the basic plan allows
# about 200 a minute for the whole app.
_SYMBOLS_PER_REQUEST = 100
_REQUEST_GAP_SEC = 0.3
_REQUEST_ATTEMPTS = 5
_BAR_COLUMNS = ["date", "symbol", "open", "high", "low", "close", "volume", "trades", "vwap"]
# How far past today the contract list is fetched: past the 120 days any close
# up to today reads.
_CONTRACTS_REACH_DAYS = 130
_CONTRACTS_BACK_DAYS = 31
_CONTRACT_COLUMNS = ["symbol", "type", "strike", "expiration"]

_last_request = [0.0]


def _options_dir(symbol: str) -> Path:
    return HISTORY_DIR / "options" / symbol.upper()


def _get_json(url: str, params: dict, key: str, secret: str) -> dict:
    """One GET, paced, with backoff on throttling, server errors and a dropped
    connection -- a cold build is ~100 requests, and one lost to a blip would
    cost the session its forecast."""
    import requests

    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    for attempt in range(_REQUEST_ATTEMPTS):
        wait = _REQUEST_GAP_SEC - (time.monotonic() - _last_request[0])
        if wait > 0:
            time.sleep(wait)
        _last_request[0] = time.monotonic()
        try:
            r = requests.get(url, headers=headers, params=params, timeout=60)
        except (requests.ConnectionError, requests.Timeout):
            if attempt == _REQUEST_ATTEMPTS - 1:
                raise
            time.sleep(min(30.0, 2.0 * 2**attempt))
            continue
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503, 504) and attempt < _REQUEST_ATTEMPTS - 1:
            time.sleep(min(30.0, 2.0 * 2**attempt))
            continue
        r.raise_for_status()
    return {}


def _get_paged(url: str, params: dict, key: str, secret: str) -> "list[dict]":
    params, pages = dict(params), []
    while True:
        payload = _get_json(url, params, key, secret)
        pages.append(payload)
        token = payload.get("next_page_token")
        if not token:
            return pages
        params["page_token"] = token


def _tidy_contracts(raw: "list[dict]", underlying: str) -> pd.DataFrame:
    """`options._tidy_contracts`, less the columns nothing here reads: adjusted
    contracts (after splits or special dividends) get another root or size."""
    if not raw:
        return pd.DataFrame(columns=_CONTRACT_COLUMNS)
    raw = pd.DataFrame(raw)
    raw = raw[(raw["root_symbol"] == underlying) & (raw["size"].astype(str) == "100")]
    out = pd.DataFrame({
        "symbol": raw["symbol"], "type": raw["type"],
        "strike": pd.to_numeric(raw["strike_price"]),
        "expiration": pd.to_datetime(raw["expiration_date"]),
    })
    return out.drop_duplicates("symbol").sort_values(["expiration", "type", "strike"]).reset_index(drop=True)


def _download_contracts(underlying: str, lo: date, hi: date, key: str, secret: str) -> pd.DataFrame:
    """Every contract, active or expired, expiring in [lo, hi]. The list lives
    on the trading API; the paper host answers paper keys, the live host live
    ones."""
    import requests

    from .config import TRADING_REST_LIVE, TRADING_REST_PAPER

    rows: "list[dict]" = []
    for host in (TRADING_REST_PAPER, TRADING_REST_LIVE):
        try:
            for status in ("active", "inactive"):
                params = dict(underlying_symbols=underlying, status=status, limit=10000,
                              expiration_date_gte=lo.isoformat(), expiration_date_lte=hi.isoformat())
                for page in _get_paged(f"{host}{_CONTRACTS_PATH}", params, key, secret):
                    rows += page.get("option_contracts") or []
            break
        except requests.HTTPError as exc:
            if getattr(exc.response, "status_code", None) not in (401, 403) or host == TRADING_REST_LIVE:
                raise
            rows = []
    return _tidy_contracts(rows, underlying)


def _download_option_bars(symbols: "list[str]", start: pd.Timestamp, end: pd.Timestamp,
                          key: str, secret: str) -> pd.DataFrame:
    """Daily bars of `symbols` dated in [start, end] (Alpaca's date bounds are
    inclusive), in `options._download_expiration`'s frame."""
    from .config import DATA_REST

    rows = []
    for i in range(0, len(symbols), _SYMBOLS_PER_REQUEST):
        params = dict(symbols=",".join(symbols[i:i + _SYMBOLS_PER_REQUEST]), timeframe="1Day",
                      limit=10000, start=f"{start:%Y-%m-%d}", end=f"{end:%Y-%m-%d}")
        for page in _get_paged(f"{DATA_REST}{_OPTION_BARS_URL}", params, key, secret):
            for symbol, bars in (page.get("bars") or {}).items():
                rows += [(b["t"], symbol, b["o"], b["h"], b["l"], b["c"], b["v"], b["n"], b["vw"]) for b in bars]
    out = pd.DataFrame(rows, columns=["t", *_BAR_COLUMNS[1:]])
    # "t" is midnight New York time, written in UTC
    stamps = pd.to_datetime(out.pop("t"), utc=True).dt.tz_convert(market_hours.MARKET_TZ)
    out.insert(0, "date", stamps.dt.tz_localize(None).dt.normalize())
    return out


def _options_index(symbol: str) -> dict:
    payload = _read_json(_options_dir(symbol) / "index.json")
    if not payload or payload.get("layout") != OPTIONS_LAYOUT:
        return {"layout": OPTIONS_LAYOUT, "contracts": None, "expirations": {}}
    payload.setdefault("expirations", {})
    return payload


def _contracts(symbol: str, index: dict, lo: pd.Timestamp, hi: pd.Timestamp, after: pd.Timestamp,
               key: str, secret: str) -> pd.DataFrame:
    """The contracts expiring in [lo, hi], from a list fetched on a day later
    than `after` -- a contract listed on a session trades on it, so a list from
    that morning or before could miss one."""
    folder = _options_dir(symbol)
    path = folder / "contracts.parquet"
    meta = index.get("contracts") or {}
    first = lo
    cached = None
    if path.exists():
        try:
            cached = pd.read_parquet(path)
        except (OSError, ValueError):
            cached = None
    fresh = (
        cached is not None and meta.get("fetched", "") > f"{after:%Y-%m-%d}"
        and meta.get("lo", "9999") <= f"{lo:%Y-%m-%d}" and meta.get("hi", "") >= f"{hi:%Y-%m-%d}"
    )
    if not fresh:
        from .datalog import log_fetch

        # Out to everything listed, so the next close of a replayed week -- whose
        # window reaches a day further -- is covered by this same list; a month
        # back, for a replay stepping back; and over what the last list covered,
        # so the record of what is covered stays one stretch fetched on one day.
        reach = max(hi, pd.Timestamp(_today_et()) + pd.Timedelta(_CONTRACTS_REACH_DAYS, "D"))
        lo = lo - pd.Timedelta(_CONTRACTS_BACK_DAYS, "D")
        if cached is not None and meta.get("lo"):
            lo = min(lo, pd.Timestamp(meta["lo"]))
        got = _download_contracts(symbol, lo.date(), reach.date(), key, secret)
        merged = got if cached is None else pd.concat([got, cached]).drop_duplicates("symbol")
        folder.mkdir(parents=True, exist_ok=True)
        merged.sort_values(["expiration", "type", "strike"]).reset_index(drop=True).to_parquet(path)
        index["contracts"] = {"fetched": _today_et().isoformat(), "lo": f"{lo:%Y-%m-%d}", "hi": f"{reach:%Y-%m-%d}"}
        _write_json(folder / "index.json", index)
        log_fetch("option contracts (HighLow_3m)", "Alpaca trading API", symbol=symbol,
                  detail=f"{len(got)} contracts expiring {lo:%Y-%m-%d} – {reach:%Y-%m-%d}")
        cached = merged
    expiry = cached["expiration"]
    return cached[(expiry >= first) & (expiry <= hi)].reset_index(drop=True)


def _bars_path(symbol: str, expiration: pd.Timestamp) -> Path:
    return _options_dir(symbol) / "bars" / f"{expiration:%Y-%m-%d}.parquet"


def _ensure_option_bars(symbol: str, index: dict, contracts: pd.DataFrame, closes: pd.Series,
                        through: pd.Timestamp, key: str, secret: str) -> pd.DataFrame:
    """Every listed expiration's daily bars over its last 120 days, up to
    `through`, fetched where the cache falls short; returns them all.

    Which contracts: `options.fetch_bars`' band, strikes within 0.6-1.5 times
    the lowest and highest close over the contract's window -- by the closes
    known so far. The band only ever widens as the window grows, and a strike
    entering it is fetched over its whole window, so the cache holds a superset
    of what any close up to `through` needs: the tables re-band on each day's
    own close, and nothing outside 0.6-1.5 times it is read.
    """
    from .datalog import log_fetch

    lo, hi = OPTION_STRIKE_BAND
    jobs: "dict[tuple[pd.Timestamp, pd.Timestamp], list[str]]" = {}
    plans: "dict[pd.Timestamp, dict]" = {}
    for expiration, group in contracts.groupby("expiration"):
        start, end = bar_window(expiration, through)
        path_px = closes.loc[start:end]
        if path_px.empty:
            continue
        band = set(group.loc[group["strike"].between(lo * path_px.min(), hi * path_px.max()), "symbol"])
        entry = index["expirations"].get(f"{expiration:%Y-%m-%d}") or {}
        have = set(entry.get("symbols") or [])
        upto = pd.Timestamp(entry["through"]) if entry.get("through") else None
        new = sorted(band - have) if upto is not None else sorted(band)
        if new:
            # As far as the rest of the expiration's cache reaches, so it stays
            # one `through` for every contract in it.
            jobs.setdefault((start, max(end, upto) if upto is not None else end), []).extend(new)
        if upto is not None and upto < end and have:
            jobs.setdefault((upto + pd.Timedelta(1, "D"), end), []).extend(sorted(have))
        plans[expiration] = {"symbols": sorted(have | band),
                             "through": max(end, upto) if upto is not None else end}

    fetched = []
    if jobs:
        key, secret = _credentials(key, secret)
        for (start, end), symbols in sorted(jobs.items()):
            if start <= end:
                fetched.append(_download_option_bars(symbols, start, end, key, secret))
        got = pd.concat(fetched, ignore_index=True) if fetched else pd.DataFrame(columns=_BAR_COLUMNS)
        log_fetch("option daily bars (HighLow_3m)", "Alpaca REST", symbol=symbol,
                  detail=f"{len(got)} bars of {sum(len(s) for s in jobs.values())} contracts "
                         f"through {through:%Y-%m-%d}")
        expiry_of = contracts.set_index("symbol")["expiration"]
        got_by_exp = dict(tuple(got.groupby(got["symbol"].map(expiry_of)))) if len(got) else {}
    else:
        got_by_exp = {}

    frames = []
    for expiration, plan in plans.items():
        path = _bars_path(symbol, expiration)
        old = None
        if path.exists():
            try:
                old = pd.read_parquet(path)
            except (OSError, ValueError):
                old = None
        add = got_by_exp.get(expiration)
        if add is not None and len(add):
            both = add if old is None else pd.concat([old, add], ignore_index=True)
            both = both.drop_duplicates(["symbol", "date"], keep="last").sort_values(["symbol", "date"])
            path.parent.mkdir(parents=True, exist_ok=True)
            both[_BAR_COLUMNS].reset_index(drop=True).to_parquet(path)
            old = both
        index["expirations"][f"{expiration:%Y-%m-%d}"] = {
            "symbols": plan["symbols"], "through": f"{plan['through']:%Y-%m-%d}",
        }
        if old is not None and len(old):
            frames.append(old[_BAR_COLUMNS])
    if jobs:
        _write_json(_options_dir(symbol) / "index.json", index)
    if not frames:
        return pd.DataFrame(columns=_BAR_COLUMNS)
    bars = pd.concat(frames, ignore_index=True)
    return bars[bars["date"] <= through].reset_index(drop=True)


def _irx_path() -> Path:
    return HISTORY_DIR / "irx.json"


def _download_irx(start: date, end: date) -> "dict[str, float]":
    """Yahoo's ^IRX daily close over [start, end), as `yahoo.fetch_daily` reads it."""
    import yfinance as yf

    raw = yf.download("^IRX", start=start.isoformat(), end=end.isoformat(),
                      auto_adjust=False, progress=False)
    if raw is None or not len(raw):
        return {}
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    close = raw.rename(columns=str.lower)["close"].dropna()
    idx = pd.DatetimeIndex(close.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return {d.strftime("%Y-%m-%d"): float(v) for d, v in zip(idx.normalize(), close.to_numpy())}


def treasury_yield(first: pd.Timestamp, last: pd.Timestamp) -> pd.Series:
    """^IRX closes (percent) from at least ten days before `first` through
    `last`, cached in `irx.json`.

    Only completed days are kept: a close dated today would be the morning's
    quote, and tomorrow's tables would read it as today's close. Bond-market
    holidays the NYSE trades through (Columbus Day, Veterans Day) have no
    close, and the tables carry the last one forward, as the notebook's do.
    """
    today = _today_et().isoformat()
    want_from = (first - pd.Timedelta(10, "D")).strftime("%Y-%m-%d")
    want_to = f"{last:%Y-%m-%d}"
    cache = _read_json(_irx_path()) or {}
    closes = dict(cache.get("closes") or {})
    covered = (cache.get("from", "9999") <= want_from
               and cache.get("checked", "") > want_to)
    if not covered:
        start = min(want_from, cache.get("from", want_from))
        got = _download_irx(date.fromisoformat(start), _today_et())
        closes.update({d: v for d, v in got.items() if d < today})
        _write_json(_irx_path(), {"from": start, "checked": today, "closes": closes})
    out = pd.Series({d: v for d, v in closes.items()}, dtype=float)
    out.index = _dates(out.index)
    return out.sort_index()


def _today_et() -> date:
    return datetime.now(market_hours.MARKET_TZ).date()


# --- positioning rows, one per previous close -------------------------------------------

POSITIONING_LAYOUT = 1
_ROW_FIELDS = [
    "spot", "call_wall", "put_wall", "call_wall_7d", "put_wall_7d", "call_gamma_wall",
    "put_gamma_wall", "net_gex", "gross_gex", "gex_ratio", "gamma_flip", "iv_1d", "iv_30d",
    "call_volume", "put_volume", "call_oi", "put_oi", "next_is_opex",
]


def _positioning_path(symbol: str) -> Path:
    return HISTORY_DIR / f"{symbol.upper()}_positioning.json"


def _calibration(symbol: str) -> dict:
    settings = NOTEBOOK_OPTIONS.get(symbol.upper())
    if settings is None:
        raise ValueError(
            f"no notebook option settings for {symbol} (`highlow3m_model.NOTEBOOK_OPTIONS`): "
            "its dividend yield and open-interest calibration are what the notebook priced with."
        )
    return settings


def _signature(settings: dict) -> dict:
    return {k: (str(v) if isinstance(v, float) and math.isinf(v) else v) for k, v in settings.items()}


def positioning_from(contracts: pd.DataFrame, bars: pd.DataFrame, closes: pd.Series,
                     irx: pd.Series, settings: dict) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """`options.build`'s steps 3-6 on bars already in hand: the positioning
    table and the gamma profile, one row (or grid) per close in `closes`.

    `closes` must reach back over every contract's window -- a row for day d is
    exact once every contract alive on d has its bars from 120 days before its
    expiry -- and end at the last close wanted: nothing after it is read.
    """
    bars = load_bars(bars, closes.index)
    if not len(bars):
        raise ValueError("Alpaca returned no option bars for the window.")
    rates = risk_free(irx, closes.index)
    panel = contract_days(contracts, bars, closes, settings["dividend_yield"], rates)
    smiles = fit_smiles(panel)
    panel = add_fitted_iv(panel, smiles)
    panel["oi"] = scaled_oi(panel, settings)
    return positioning_tables(panel, smiles, closes, settings.get("options_start"))


def _row_payload(table: pd.DataFrame, profile: pd.DataFrame, day: pd.Timestamp) -> dict:
    row = {c: float(table.at[day, c]) for c in _ROW_FIELDS}
    grid = profile[profile["date"] == day].set_index("m")
    row["profile_net"] = [float(v) for v in grid["net_gex"].reindex(GEX_GRID).to_numpy()]
    row["profile_gross"] = [float(v) for v in grid["gross_gex"].reindex(GEX_GRID).to_numpy()]
    return row


def positioning_frames(rows: "dict[str, dict]") -> "tuple[pd.DataFrame, pd.DataFrame]":
    """Cached rows -> the notebook's two tables: positioning (one row per
    close) and the long gamma profile `features._gex_at` pivots."""
    if not rows:
        return (pd.DataFrame(columns=_ROW_FIELDS, index=pd.DatetimeIndex([], name="date")),
                pd.DataFrame(columns=["date", "m", "net_gex", "gross_gex"]))
    dates = sorted(rows)
    index = _dates(dates)
    table = pd.DataFrame([{c: rows[d][c] for c in _ROW_FIELDS} for d in dates], index=index)
    long = []
    for d, stamp in zip(dates, index):
        net, gross = rows[d].get("profile_net"), rows[d].get("profile_gross")
        if not net:
            continue
        long.append(pd.DataFrame({"date": stamp, "m": GEX_GRID, "net_gex": net, "gross_gex": gross}))
    profile = pd.concat(long, ignore_index=True) if long else pd.DataFrame(
        columns=["date", "m", "net_gex", "gross_gex"])
    profile = profile.dropna(subset=["net_gex"])
    return table, profile


def positioning_rows(
    symbol: str,
    days,
    key: "str | None" = None,
    secret: "str | None" = None,
    min_bars: int = MIN_BARS_PER_SESSION,
) -> "dict[str, dict]":
    """The option positioning row of each close in `days` (session dates), from
    the cache or built.

    A row for day d reads the contracts expiring within 120 days of d, each
    over its last 120 days up to d, d's SIP close and the T-bill yield on d --
    nothing later, so a built row is kept for good. It is only built for a day
    before today (Alpaca's daily option bars of a session still trading are not
    its close) and only kept once ^IRX has a close dated d or later (before
    that, d's rate may be a stale one carried forward). Building one row builds
    every close in between too, at no extra fetch.
    """
    symbol = symbol.upper()
    settings = _calibration(symbol)
    wanted = sorted({pd.Timestamp(d).normalize() for d in days})
    if not wanted:
        return {}
    today = pd.Timestamp(_today_et())
    if wanted[-1] >= today:
        raise ValueError(
            f"the option tables of {wanted[-1].date()} are not final until that session is over."
        )
    with _options_lock:
        cache = _read_json(_positioning_path(symbol)) or {}
        if cache.get("layout") != POSITIONING_LAYOUT or cache.get("settings") != _signature(settings):
            cache = {"layout": POSITIONING_LAYOUT, "settings": _signature(settings), "rows": {}}
        rows = cache["rows"]
        missing = [d for d in wanted if f"{d:%Y-%m-%d}" not in rows]
        out = {}
        for lo, hi in _spans(missing):
            built = _build_rows(symbol, lo, hi, settings, key, secret, min_bars)
            for iso, (row, final) in built.items():
                if final:
                    rows[iso] = row
            out.update({iso: row for iso, (row, _) in built.items()})
        if missing:
            _write_json(_positioning_path(symbol), cache)
        out.update({iso: rows[iso] for iso in rows})
    return {f"{d:%Y-%m-%d}": out[f"{d:%Y-%m-%d}"] for d in wanted if f"{d:%Y-%m-%d}" in out}


# Closes further apart than this are built separately: one build reads the
# minute history around both ends, and a span much wider than that history
# would leave a hole in the closes between them.
_SPAN_GAP_DAYS = 60


def _spans(days: "list[pd.Timestamp]") -> "list[tuple[pd.Timestamp, pd.Timestamp]]":
    """Sorted days -> (first, last) runs no wider apart inside than _SPAN_GAP_DAYS."""
    spans: "list[list[pd.Timestamp]]" = []
    for day in days:
        if spans and (day - spans[-1][1]).days <= _SPAN_GAP_DAYS:
            spans[-1][1] = day
        else:
            spans.append([day, day])
    return [(a, b) for a, b in spans]


def _build_rows(symbol: str, lo: pd.Timestamp, hi: pd.Timestamp, settings: dict,
                key: "str | None", secret: "str | None", min_bars: int) -> "dict[str, tuple[dict, bool]]":
    """Every close in [lo, hi] -> (row, whether it may be kept)."""
    key, secret = _credentials(key, secret)
    history = history_inputs(symbol, hi + pd.Timedelta(1, "D"), key, secret, min_bars)
    closes = history.closes
    reach = lo - pd.Timedelta(OPTION_MAX_DTE + 5, "D")
    if not len(closes) or closes.index.min() > reach:
        older = history_inputs(symbol, lo, key, secret, min_bars).closes
        closes = pd.concat([older, closes])
        closes = closes[~closes.index.duplicated(keep="last")].sort_index()
    closes = closes[closes.index <= hi]
    days = closes.index[(closes.index >= lo) & (closes.index <= hi)]
    if not len(days):
        return {}

    index = _options_index(symbol)
    contracts = _contracts(symbol, index, lo, hi + pd.Timedelta(OPTION_MAX_DTE, "D"), hi, key, secret)
    if not len(contracts):
        raise ValueError(f"Alpaca lists no {symbol} option contracts expiring after {lo.date()}.")
    bars = _ensure_option_bars(symbol, index, contracts, closes, hi, key, secret)
    irx = treasury_yield(closes.index.min(), hi)
    if not len(irx):
        raise ValueError("no 13-week T-bill yield (Yahoo ^IRX) to price the options with.")
    table, profile = positioning_from(contracts, bars, closes, irx, settings)
    last_rate = irx.index.max()
    return {
        f"{d:%Y-%m-%d}": (_row_payload(table, profile, d), last_rate >= d)
        for d in days
    }


def previous_close(history: History, day) -> pd.Timestamp:
    """The last session SIP printed before `day`: the close the option row is
    read from (half days and dropped sessions included, as the notebook's
    table has a row for each)."""
    closes = history.closes[history.closes.index < pd.Timestamp(day)]
    if not len(closes):
        raise ValueError(f"no SIP session before {pd.Timestamp(day).date()} to read the option chain at.")
    return closes.index.max()


# --- one session's forecast --------------------------------------------------

_forecast_cache: "dict[tuple, dict]" = {}
_FORECAST_CACHE_MAX = 512


def forecast_from(bundle: dict, history: History, opening: pd.DataFrame, session_date,
                  positioning: "pd.DataFrame | None" = None,
                  profile: "pd.DataFrame | None" = None) -> dict:
    """The predicted high and low after 9:33 from the history (`history_inputs`),
    the IEX opening (`fetch_opening`) and last night's option tables
    (`positioning_frames`).

    Returns the keys `dayrange_model.forecast_session` returns --
    `{"pred_high", "pred_low", "prev_avg", "adr14_abs", "or_high", "or_low"}`
    in dollars -- plus `range_after_opening`: the range is the one *after* the
    window, so the opening's own extremes are not a breach of it.
    Raises ValueError when the inputs cannot support a forecast.
    """
    day = pd.Timestamp(session_date).normalize()
    if day.tzinfo is not None:
        day = day.tz_localize(None)
    history = history.before(day)
    if len(history.sessions) < MIN_PRIOR_SESSIONS:
        raise ValueError(
            f"only {len(history.sessions)} complete SIP sessions of history before {day.date()}; "
            f"the HighLow_3m forecast needs {MIN_PRIOR_SESSIONS} for its 126-day windows."
        )
    model = bundle["model"]
    if bundle.get("reads_options", True) and (positioning is None or not len(positioning)):
        raise ValueError("no option positioning for the previous close.")

    opening_today = opening.copy()
    opening_today.index = pd.DatetimeIndex([day], name="date").as_unit("ns")
    # Today's daily row, which `daily_features` reads nothing out of but its
    # date: every statistic there is shifted a day, and the gap reads IEX's
    # `open3` rather than this row's open. Filled from the opening window so
    # the frame holds nothing the session has not printed by 9:33.
    o = opening_today.iloc[0]
    today_daily = pd.DataFrame(
        {"open": o["open3"], "high": o["high3"], "low": o["low3"], "close": o["close3"],
         "volume": o["volume3"], "rv": np.nan},
        index=opening_today.index,
    )

    def stacked(frames: "list[pd.DataFrame]") -> pd.DataFrame:
        return pd.concat([f for f in frames if len(f)] or frames[:1])

    sessions = history.sessions
    daily = stacked([sessions[_ROLLUP_COLS], today_daily])
    # A past session IEX printed nothing in keeps its daily row but leaves the
    # panel, as the notebook's inner join leaves it out.
    opening_all = stacked([sessions.loc[sessions["open3"].notna(), _OPENING_COLS], opening_today])
    panel = panel_from(daily, opening_all, sessions[_REST_COLS], positioning, profile)
    row = panel.loc[[day]]

    missing = [
        c for c in model.feature_cols
        if c not in NAN_OK and not np.isfinite(float(row[c].iloc[0]))
    ]
    if missing:
        raise ValueError(
            "the feature row is incomplete "
            f"({', '.join(missing[:4])}{'…' if len(missing) > 4 else ''}); "
            "the SIP, IEX or option history is too short or has gaps."
        )
    pred = model.predict_prices(row).iloc[0]
    return {
        "pred_high": float(pred["pred_high"]),
        "pred_low": float(pred["pred_low"]),
        "prev_avg": float(row["prev_mid"].iloc[0]),
        "adr14_abs": float(row["adr14_usd"].iloc[0]),
        "or_high": float(row["high3"].iloc[0]),
        "or_low": float(row["low3"].iloc[0]),
        "range_after_opening": True,
    }


def warm_history(
    bundle: dict, ticker: str, before, key: "str | None" = None, secret: "str | None" = None,
) -> None:
    """Stretch the caches `forecast_session` reads over the sessions before
    `before`: the minute history, and the option tables of the last close
    before it (when that close is over)."""
    history = history_inputs(ticker, before, key, secret, min_bars(bundle))
    if bundle.get("reads_options", True):
        last = previous_close(history, before)
        if last < pd.Timestamp(_today_et()):
            positioning_rows(ticker, [last], key, secret, min_bars(bundle))


def warm_span(
    bundle: dict, ticker: str, first, last, key: "str | None" = None, secret: "str | None" = None,
) -> None:
    """`warm_history` for every session in [first, last] at once: the minute
    history reaching both ends, and the option tables of every close the
    sessions read -- one build for the lot rather than one per session."""
    warm_history(bundle, ticker, first, key, secret)
    history = history_inputs(ticker, last, key, secret, min_bars(bundle))
    if not bundle.get("reads_options", True):
        return
    start = previous_close(history_inputs(ticker, first, key, secret, min_bars(bundle)), first)
    closes = history.closes
    closes = closes[(closes.index >= start) & (closes.index < pd.Timestamp(_today_et()))]
    if len(closes):
        positioning_rows(ticker, list(closes.index), key, secret, min_bars(bundle))


def forecast_session(
    bundle: dict,
    ticker: str,
    opening_bars: pd.DataFrame,
    session_date,
    key: "str | None" = None,
    secret: "str | None" = None,
) -> dict:
    """`forecast_from` with the history, this morning and the option tables
    fetched for `ticker`.

    `opening_bars` -- the caller's first bars, on whatever tape it trades --
    only has to show that the opening window has closed: every input is read
    from Alpaca (or the caches of it), so a replay of any dataset forecasts
    from what a live run would have seen at 9:33.

    Memoised per (bundle, ticker, session): a SimLab tuning grid replays the
    same session under dozens of configurations, and the forecast depends on
    none of them.
    """
    want = opening_minutes(bundle)
    if len(opening_bars) < want:
        raise ValueError(
            f"the forecast is built on the first {want} minutes and only "
            f"{len(opening_bars)} bars have closed."
        )
    day = pd.Timestamp(session_date).date()
    symbol = str(ticker).upper()
    memo = (bundle.get("path"), symbol, day)
    cached = _forecast_cache.get(memo)
    if cached is not None:
        return dict(cached)
    history = history_inputs(symbol, day, key, secret, min_bars(bundle))
    settled = True
    opening = _cached_opening(symbol, day, min_bars(bundle)) if want == OPENING_MINUTES else None
    if opening is None:
        opening, settled = fetch_opening(symbol, day, want, key, secret)
    positioning = profile = None
    if bundle.get("reads_options", True):
        last = previous_close(history, day)
        rows = positioning_rows(symbol, [last], key, secret, min_bars(bundle))
        positioning, profile = positioning_frames(rows)
    out = forecast_from(bundle, history, opening, day, positioning, profile)
    if settled:
        if len(_forecast_cache) >= _FORECAST_CACHE_MAX:
            _forecast_cache.clear()
        _forecast_cache[memo] = dict(out)
    return out
