"""When in the session a stock moves, and how far it moves in a day (IntradayVolatility).

FinNotebooks/IntradayVolatility asked whether the extremes of a session print
near the open, and found they do because volatility itself is L-shaped: the
09:30 five minutes carry ~10x the day's average variance, it decays like a
power law to a flat midday, and ramps up again in the last quarter hour. This
module carries two of that study's components, exported by
`scripts/export_app_model.py` into `Code/Models/intravol_<TICKER>.json`:

shape      the time-of-day volatility profile, notebook 03's recommended closed
           form (profile R² ~0.97 with five parameters):
               vol(t) = a + b (1+t)^-alpha + c exp(-(390-t)/kappa)
day_range  a HAR forecast of the day's log high-low range from daily bars:
           yesterday's range, the week's and the month's, and today's gap.

Together they answer the question the chart overlay asks -- how wide should the
price be allowed to swing *at this time of day* -- in two ways that differ only
in where the day's extremes come from (`envelope`): this model's own day-range
forecast, or TimeToChange3's predicted high and low.

What it is not
--------------
The notebooks' best forecasts update bin by bin (a CARR on today's realised
ranges, then LightGBM); none of that is here. Everything this module computes is
known at the 09:30 open and does not change during the session, which is what
makes it a picture of the day rather than a signal.

The day-range HAR is **not** the notebooks' HAR. Theirs predicts a per-bin
Parkinson level from lagged minute statistics, which the app's daily history
(yfinance daily bars) cannot supply, so the exporter refits it on daily-bar
features and validates it walk-forward the way notebook 04 does. The file's
`day_range.walk_forward` carries the per-year result; read it before trusting
the standalone band's width.

The mirror contract
-------------------
`day_range_features`, `relative_volatility` and `volatility_shape` are copies of
the exporter's functions of the same names. The exported file carries a `check`
block -- one session's daily bars, open and forecast, and the shape at a few
minutes -- which `tests/test_intraday_vol_model.py` reproduces exactly. Change
one side and re-export.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .model_store import ModelStore

MODEL_PATH_ENV = "INTRAVOL_MODEL"

# The symbols the study was run on; one exported file each.
TICKERS = ("AAPL", "GOOGL", "INTC")

SESSION_MINUTES = 390
FEATURES = ("lr_d", "lr_w", "lr_m", "abs_gap")
WEEK, MONTH = 5, 22
MIN_RANGE = 5e-5

_SHAPE_PARAMS = {"a", "b", "alpha", "c", "kappa"}


def _build(path: Path) -> "dict | None":
    """The exported model, or None when the file is missing or not this shape."""
    try:
        raw = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    shape, day_range = raw.get("shape") or {}, raw.get("day_range") or {}
    if _SHAPE_PARAMS - set(shape.get("params") or {}) or len(shape.get("t_domain") or []) != 2:
        return None
    if {"const", *FEATURES} - set(day_range.get("coef") or {}):
        return None
    return raw


_STORE = ModelStore(
    env_key=MODEL_PATH_ENV,
    filename="intravol_{ticker}.json",
    build=_build,
)

model_path = _STORE.path
load = _STORE.load
reset_cache = _STORE.reset


def covers(ticker: "str | None") -> bool:
    return (ticker or "").upper() in TICKERS


# --- the shape --------------------------------------------------------------


def relative_volatility(model: dict, minutes) -> np.ndarray:
    """The fitted volatility curve at minute `m` of the session (1 = day average).

    Evaluated at the minute's midpoint and held flat outside the bin centres
    the curve was fitted on, so it is never extrapolated into the auction
    minute.
    """
    shape = model["shape"]
    p = shape["params"]
    lo, hi = shape["t_domain"]
    t = np.clip(np.asarray(minutes, float) + 0.5, lo, hi)
    return (
        p["a"]
        + p["b"] * (1.0 + t) ** (-p["alpha"])
        + p["c"] * np.exp(-(SESSION_MINUTES - t) / p["kappa"])
    )


def volatility_shape(model: dict, minutes=None) -> np.ndarray:
    """Relative volatility scaled so the session's maximum is 1.

    For these stocks the maximum is the open, so the curve is 1 at 09:30,
    falls to roughly a fifth of that by midday and turns up into the close.
    """
    grid = np.arange(SESSION_MINUTES + 1)
    peak = float(relative_volatility(model, grid).max())
    return relative_volatility(model, grid if minutes is None else minutes) / peak


# --- the day's range --------------------------------------------------------


def day_range_features(
    daily_bars: "list[dict]", session_date, open_price: float
) -> "dict[str, float]":
    """The HAR inputs for one session, from completed daily bars and its open.

    Only bars dated strictly before the session count, whatever the caller
    passes, so a store holding the day's own bar cannot leak its range. Raises
    ValueError when there is too little history rather than forecasting off a
    shorter window than the model was fitted on.
    """
    day = pd.Timestamp(session_date).normalize()
    by_date: "dict[pd.Timestamp, dict]" = {}
    for bar in daily_bars or []:
        stamp = pd.Timestamp(str(bar.get("t", ""))[:10])
        if stamp < day:
            by_date[stamp] = bar
    prior = [by_date[k] for k in sorted(by_date)]
    if len(prior) < MONTH:
        raise ValueError(
            f"needs {MONTH} completed daily bars before {day.date()} for the monthly "
            f"range; got {len(prior)}."
        )
    if not open_price or open_price <= 0:
        raise ValueError("no opening price for the session.")

    ranges = [
        max(math.log(float(b["h"]) / float(b["l"])), MIN_RANGE) for b in prior[-MONTH:]
    ]
    return {
        "lr_d": math.log(ranges[-1]),
        "lr_w": math.log(float(np.mean(ranges[-WEEK:]))),
        "lr_m": math.log(float(np.mean(ranges))),
        "abs_gap": abs(math.log(float(open_price) / float(prior[-1]["c"]))),
    }


def predict_log_range(model: dict, features: "dict[str, float]") -> float:
    """log of the day's ln(high/low) -- the HAR's point forecast."""
    coef = model["day_range"]["coef"]
    return float(coef["const"] + sum(coef[k] * float(features[k]) for k in FEATURES))


def predicted_extremes(
    model: dict, daily_bars: "list[dict]", session_date, open_price: float
) -> "tuple[float, float]":
    """The day's high and low this model alone implies: the forecast range,
    split evenly in log space either side of the open -- volatility has no
    direction."""
    log_range = predict_log_range(
        model, day_range_features(daily_bars, session_date, open_price)
    )
    half = math.exp(log_range) / 2.0
    return float(open_price) * math.exp(half), float(open_price) * math.exp(-half)


# --- the band ---------------------------------------------------------------


def envelope(
    model: dict, open_price: float, high: float, low: float, minutes=None
) -> "tuple[np.ndarray, np.ndarray]":
    """Upper and lower price curves that follow the volatility shape.

        upper(t) = open + (high - open) * w(t)
        lower(t) = open - (open - low)  * w(t)

    with `w` the volatility shape scaled to peak at 1. The upper curve's
    maximum is therefore exactly `high` and the lower curve's minimum exactly
    `low`, at the time of day volatility peaks, and the band between them is as
    wide as the stock usually moves at that time of day relative to the open.
    Each side keeps its own distance, so an asymmetric forecast stays
    asymmetric.

    The centre is clamped into [low, high]: an official open a tick outside a
    forecast that contains the opening range would otherwise flip a side.
    """
    centre = min(max(float(open_price), float(low)), float(high))
    w = volatility_shape(model, minutes)
    return centre + (float(high) - centre) * w, centre - (centre - float(low)) * w
