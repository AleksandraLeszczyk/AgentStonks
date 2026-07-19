"""Predicted price profile at the open (LevelsML density model).

Loads the pack trained by Models/train_open_profile.py: one LightGBM booster
per volume-quantile of the day's price profile, predicting where today's
volume will trade (in bps of log-return vs the 9:30 open) from daily-bar
features plus the opening print. The predicted quantile function is turned
into a smooth density with a monotone cubic (PCHIP) CDF — where quantiles
bunch, density is high — and drawn next to the realized profile in the live
chart, where the Gaussian/Cauchy mixture fit can target it.

Feature definitions here must mirror levelsml/features.py exactly; the pack's
"features" list is the contract. Requires the optional `lightgbm` dependency;
everything degrades to None without it (or without the pack file).
"""

from __future__ import annotations

import gzip
import json
import os
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import clock
from .state import completed_daily_bars, today_daily_bar

# Default: sibling "Models" directory next to the AgentStonks repo checkout.
MODEL_PATH_ENV = "OPEN_PROFILE_MODEL"
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "Models" / "open_profile_lgbm.json.gz"

# Fewer than this many completed daily bars and the 20d rolling features are
# all-NaN — a prediction would be climatology at best, so refuse instead.
MIN_DAILY_BARS = 21

_lock = threading.Lock()
_cache: dict = {"path": None, "pack": None, "boosters": None}


def _model_path() -> Path:
    return Path(os.environ.get(MODEL_PATH_ENV) or DEFAULT_MODEL_PATH)


def load_pack() -> "dict | None":
    """The model pack with instantiated boosters, or None if unavailable.

    Cached after the first successful load; a missing file or missing
    lightgbm install is also cached (per path) so the chart poll doesn't
    retry the filesystem every few seconds.
    """
    path = _model_path()
    with _lock:
        if _cache["path"] == path:
            return _cache["pack"]
        _cache.update(path=path, pack=None, boosters=None)
        try:
            import lightgbm as lgb
        except ImportError:
            return None
        try:
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                raw = json.load(fh)
            boosters = {
                q: lgb.Booster(model_str=s) for q, s in raw["boosters"].items()
            }
        except (OSError, KeyError, ValueError):
            return None
        raw["_boosters"] = boosters
        _cache["pack"] = raw
        return raw


def compute_features(
    daily_bars: list[dict],
    today_open: float,
    today: "str | None" = None,
) -> "dict | None":
    """The pack's feature vector for today, from Alpaca daily bars + the open.

    Replicates levelsml/features.py by appending a stub row for today (open
    only) to the completed daily history and evaluating the same shifted /
    rolling expressions, so every feature is point-in-time correct.
    """
    completed = completed_daily_bars(daily_bars, today)
    if len(completed) < MIN_DAILY_BARS or not today_open or today_open <= 0:
        return None

    try:
        daily = pd.DataFrame(
            {
                "open": [float(b["o"]) for b in completed],
                "high": [float(b["h"]) for b in completed],
                "low": [float(b["l"]) for b in completed],
                "close": [float(b["c"]) for b in completed],
                "volume": [float(b["v"]) for b in completed],
            },
            index=pd.to_datetime([str(b["t"])[:10] for b in completed]),
        )
    except (KeyError, TypeError, ValueError):
        return None

    today_ts = pd.Timestamp((today or clock.now().strftime("%Y-%m-%d"))[:10])
    stub = pd.DataFrame(
        {"open": [today_open], "high": [np.nan], "low": [np.nan],
         "close": [np.nan], "volume": [np.nan]},
        index=[today_ts],
    )
    df = pd.concat([daily, stub])

    prev_close = df["close"].shift(1)
    df["prev_close"] = prev_close
    df["prev_high"] = df["high"].shift(1)
    df["prev_low"] = df["low"].shift(1)
    df["prev_range"] = (df["prev_high"] - df["prev_low"]) / df["prev_close"]
    df["prev_ret"] = np.log(df["close"] / df["open"]).shift(1)
    df["prev_volume_z"] = (
        (df["volume"] - df["volume"].rolling(20).mean()) / df["volume"].rolling(20).std()
    ).shift(1)
    df["prev_close_pos_in_range"] = (
        (df["close"] - df["low"]) / (df["high"] - df["low"])
    ).shift(1)

    close_y = df["close"].shift(1)
    df["ret_5d"] = np.log(close_y / close_y.shift(5))
    df["ret_20d"] = np.log(close_y / close_y.shift(20))
    df["vol_5d"] = np.log(df["close"] / df["close"].shift(1)).shift(1).rolling(5).std()
    df["vol_20d"] = np.log(df["close"] / df["close"].shift(1)).shift(1).rolling(20).std()
    tr = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs(),
        ),
    )
    df["atr_14"] = (tr.rolling(14).mean() / df["close"]).shift(1)

    df["dist_high_20d"] = np.log(close_y / df["high"].shift(1).rolling(20).max())
    df["dist_low_20d"] = np.log(close_y / df["low"].shift(1).rolling(20).min())
    df["dist_high_252d"] = np.log(
        close_y / df["high"].shift(1).rolling(252, min_periods=60).max()
    )
    df["dist_low_252d"] = np.log(
        close_y / df["low"].shift(1).rolling(252, min_periods=60).min()
    )
    df["trend_5d"] = close_y.rolling(5).apply(_slope, raw=True)
    df["trend_20d"] = close_y.rolling(20).apply(_slope, raw=True)

    df["day_of_week"] = df.index.dayofweek
    df["open_gap"] = np.log(df["open"] / df["prev_close"])

    row = df.iloc[-1]
    return {c: float(row[c]) if pd.notna(row[c]) else np.nan for c in df.columns}


def _slope(x: np.ndarray) -> float:
    """OLS slope of log-price vs time, in return-per-day units (levelsml)."""
    if np.isnan(x).any():
        return np.nan
    t = np.arange(len(x))
    return float(np.polyfit(t, np.log(x), 1)[0])


def predict_quantiles(pack: dict, features: dict) -> "np.ndarray | None":
    """Predicted volume-quantiles of today's profile, bps vs the open, sorted."""
    cols = pack["features"]
    if any(c not in features for c in cols):
        return None
    X = pd.DataFrame([[features[c] for c in cols]], columns=cols)
    try:
        q = np.array([
            float(pack["_boosters"][f"q{p}"].predict(X)[0]) for p in pack["p_levels"]
        ])
    except (KeyError, ValueError):
        return None
    return np.sort(q)  # enforce monotone quantiles (independent per-q models)


# --- monotone cubic (PCHIP) density, pure numpy ----------------------------
# Same construction as LevelsML notebook 11: a Fritsch–Carlson monotone cubic
# CDF through the (padded) quantile points, differentiated on a grid. A
# piecewise-linear CDF would have a staircase derivative that fakes modes.

def _pchip_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    h = np.diff(x)
    delta = np.diff(y) / h
    d = np.zeros_like(y)
    for i in range(1, len(x) - 1):
        if delta[i - 1] * delta[i] <= 0:
            d[i] = 0.0
        else:
            w1 = 2 * h[i] + h[i - 1]
            w2 = h[i] + 2 * h[i - 1]
            d[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])
    for i, (ha, hb, da, db) in ((0, (h[0], h[1], delta[0], delta[1])),
                                (-1, (h[-1], h[-2], delta[-1], delta[-2]))):
        dd = ((2 * ha + hb) * da - ha * db) / (ha + hb)
        if np.sign(dd) != np.sign(da):
            dd = 0.0
        elif np.sign(da) != np.sign(db) and abs(dd) > 3 * abs(da):
            dd = 3 * da
        d[i] = dd
    return d


def _pchip_derivative(x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """d/dg of the PCHIP interpolant through (x, y), zero outside [x0, xn]."""
    d = _pchip_slopes(x, y)
    out = np.zeros_like(grid, dtype=float)
    inside = (grid >= x[0]) & (grid <= x[-1])
    g = grid[inside]
    idx = np.clip(np.searchsorted(x, g, side="right") - 1, 0, len(x) - 2)
    h = x[idx + 1] - x[idx]
    t = (g - x[idx]) / h
    # derivatives of the cubic Hermite basis
    out[inside] = (
        (6 * t**2 - 6 * t) * (y[idx] - y[idx + 1]) / h
        + (3 * t**2 - 4 * t + 1) * d[idx]
        + (3 * t**2 - 2 * t) * d[idx + 1]
    )
    return np.maximum(out, 0.0)


def density_from_quantiles(
    q_bps: np.ndarray, p_levels: list, n_grid: int = 241, min_gap_bps: float = 5.0
) -> tuple[np.ndarray, np.ndarray]:
    """(grid_bps, density) from quantile positions; density integrates to ~1.

    Adjacent quantiles are spread to at least `min_gap_bps` apart — the
    training profiles bin volume in 5 bps slices, so any sharper bunching in
    the independent per-quantile predictions is resolution noise that would
    otherwise become a knife-edge spike in the density.
    """
    x = np.asarray(q_bps, dtype=float).copy()
    for i in range(1, len(x)):
        x[i] = max(x[i], x[i - 1] + min_gap_bps)
    x -= (x - np.asarray(q_bps)).mean()  # keep the spread mass-centered
    p = np.asarray(p_levels, dtype=float) / 100
    span = max(x[-1] - x[0], 1.0)
    x_ext = np.concatenate([[x[0] - 0.4 * span], x, [x[-1] + 0.4 * span]])
    p_ext = np.concatenate([[0.0], p, [1.0]])
    grid = np.linspace(x_ext[0], x_ext[-1], n_grid)
    return grid, _pchip_derivative(x_ext, p_ext, grid)


def _today_open(daily_bars: list[dict], bars: list[dict], today: str) -> "float | None":
    """Today's opening print: today's still-forming daily bar, else the first
    of today's intraday bars (pre-open / lagging daily feed)."""
    bar = today_daily_bar(daily_bars, today)
    if bar is not None:
        try:
            return float(bar["o"])
        except (KeyError, TypeError, ValueError):
            pass
    todays = [b for b in bars if str(b.get("t", "")).startswith(today)]
    if todays:
        try:
            return float(todays[0]["o"])
        except (KeyError, TypeError, ValueError):
            pass
    return None


def predicted_open_profile(sym_state, bars: "list[dict] | None" = None) -> "dict | None":
    """Today's predicted price profile for one symbol, ready for the chart.

    Returns {"prices", "density", "quantiles_bps", "open", "poc_price"} with
    density normalized to peak 1.0, or None when the pack, daily history, or
    the opening print is unavailable. Cached on the SymbolState per (day, open).
    """
    pack = load_pack()
    if pack is None:
        return None
    today = clock.now().strftime("%Y-%m-%d")
    open_px = _today_open(sym_state.daily_bars, bars or [], today)
    if not open_px:
        return None

    key = (today, round(open_px, 4))
    cached = getattr(sym_state, "predicted_profile_cache", None)
    if cached and cached.get("key") == key:
        return cached["profile"]

    feats = compute_features(sym_state.daily_bars, open_px, today)
    profile = None
    if feats is not None:
        q = predict_quantiles(pack, feats)
        if q is not None:
            grid_bps, dens = density_from_quantiles(q, pack["p_levels"])
            if dens.max() > 0:
                profile = {
                    "prices": open_px * np.exp(grid_bps / 1e4),
                    "density": dens / dens.max(),
                    "quantiles_bps": q,
                    "open": open_px,
                    "poc_price": float(open_px * np.exp(grid_bps[int(np.argmax(dens))] / 1e4)),
                }
    sym_state.predicted_profile_cache = {"key": key, "profile": profile}
    return profile
