"""Where today's high and low will land, called at 09:35 (PriceRange2).

The app's other session-scale forecast, `dayrange_model`, asks the same shaped
question and answers it differently enough that the two are separate models
rather than variants. The differences are the whole reason this one exists:

                      dayrange (TimeToChange3)      pricerange (PriceRange2)
    reference price   (prev_open + prev_close)/2    the traded price at 09:35
    decision made     from the first 5 minutes      from the first 5 minutes
    predictor         LGBM + N-BEATS + N-HiTS       LightGBM alone, L1
                      blend, plus an opening ridge
    trained on        21 sessions of minute data    1,164 sessions of Alpaca
                                                    SIP minute data
    sees              its own daily history         its own history *and* 17
                                                    other markets
    also ships        --                            six quantile models

`ref` is the interesting change. Measuring against a price that has already
printed this morning means the model is never asked to re-learn the overnight
gap, and it is the price a trader can actually deal at when the decision is
made. Everything below is in logs against it::

    y_high = log(day high / ref)
    y_low  = log(day low  / ref)
    y_range = y_high - y_low

Predicting the two edges rather than the width costs nothing -- the range is
their difference -- and it is what a rule needs, because a rule that buys near
the low and sells near the high needs levels, not a width.

What the numbers are
--------------------
Walk-forward over 659 sessions, against a 21-day trailing mean of the edges
(a genuinely hard baseline: range is strongly autocorrelated):

    skill vs baseline    AAPL +14.8%    GOOG +16.8%    INTC +9.4%
    Spearman on range    0.47 (0.33)    0.49 (0.25)    0.63 (0.52)

INTC's lower skill is a *better baseline*, not a worse model -- its range
repeats more, so the trailing mean starts from a Spearman of 0.52.

**One bias is documented rather than patched, and it matters to any consumer.**
The L1 objective predicts the median and the range distribution is
right-skewed, so the point forecast runs about 0.3 log points narrow (0.9 on
INTC, whose right tail is heavier). PriceRange2's answer is the quantile
models, not a constant correction, which would improve MAE while misleading
any rule that needs the levels. `forecast_session` therefore returns the
quantile levels beside the point forecast, and `PriceRangeTrader` trades the
quantiles.

The mirror contract
-------------------
Everything below `--- features` and `--- inference` is a verbatim copy of
`pricerange/features.py` and the inference half of `pricerange/modeling.py`
from `FinNotebooks/PriceRange2`. It has to be: the saved model is a function of
those exact column definitions, and a mirror that drifts produces confident
numbers off a different feature. If `pricerange` changes, retrain **and**
update this module -- the same contract `persistence_model` has with `mshift`,
`profile_model` has with `levelsml` and `dayrange_model` has with `dayrange`.

`tests/test_pricerange_model.py` pins it: the mirrored feature functions
reproduce all 156 columns of PriceRange2's own shipped panel to 0.00e+00 on
all three tickers, over every one of its 1,164 sessions. That is the test to
re-run after any change here.

Unlike `dayrange_model` there is **no unpickle alias to install**. PriceRange2
saves plain `lightgbm.LGBMRegressor` estimators in a dict, so nothing in the
joblib is stamped with a module that has to be resolvable -- which is also why
this module needs no PyTorch and stays cheap to import.

What the live path has to supply, and where it can go wrong
-----------------------------------------------------------
Three inputs, and the third is the one that makes this model more expensive to
run than any other here.

1. **A year of the stock's own daily bars.** The longest windows are 252-day,
   and every rolled statistic is then shifted a day, so 253 completed sessions
   is the exact warm-up -- confirmed against the shipped panels, whose first
   fully-dense row is number 253. `require_history` refuses a short history
   rather than predicting off a table full of NaNs.

2. **This morning's first five 1-minute bars**, which give `ref`, the opening
   range and the opening volume.

3. **The first five minutes of the previous 21 sessions** -- their volume, and
   nothing else. Exactly one feature needs this (`open_vol_vs_21`, today's
   opening volume against its own 21-day normal) and it is the only reason the
   daily bars are not enough. `fetch_opening_volume_history` gets it, one small
   `[09:30, 09:35)` window per session, through `rest.fetch_bars_window` --
   which SimLab patches to the dataset's stored bars.

   It goes to **Alpaca rather than yfinance deliberately**, and not for
   fidelity: yfinance serves 1-minute history for the last 30 calendar days,
   which is **20 trading sessions**. The feature needs 21. That is not bad luck
   on a given morning, it is one short every morning, so the yfinance path
   cannot build this column at all and there is no point offering it.

4. **17 other markets** -- 13 ETFs plus VIX, VVIX, the 10-year yield and the
   dollar index -- each needing daily history *and* today's opening print,
   because `{sym}_gap` reads it. That is 110 of the 156 columns, and they are
   not decoration: `UNG_gap` is the second most important feature on AAPL by
   gain. `fetch_cross_frame` is the one seam that gets them.

   In SimLab this is where the model usually stops: a dataset carries the
   symbols it was recorded for, so unless SPY, QQQ and the rest were recorded
   alongside, `require_cross` refuses the session. That is the right answer --
   LightGBM would happily predict off 110 NaNs and produce a number with
   nothing behind it.

Two seams worth knowing about, both measured rather than assumed:

* **The cross-asset tape is a different vendor's here.** PriceRange2 pulls the
  13 ETFs from Alpaca (raw) and the 4 indices from Yahoo; this module pulls all
  17 from yfinance through `historical.fetch_daily_ohlc_bars`, unadjusted, for
  the same reason `dayrange_model` does. Measured over 173 recent AAPL
  sessions the swap moves the predicted edges by a **median $0.004 and at most
  $0.10**, against the model's own $1.44 mean range error -- 0.3% of it
  typically and 7% at worst. Small, one-sided in no particular direction, and
  worth knowing before comparing a live forecast to a notebook one.

* **The daily bars are the day's, not the minute tape's.** PriceRange2 rebuilds
  its daily rows from regular-hours minute bars, so its `close` is the 15:59
  bar and its `volume` is regular-hours only. yfinance's daily bar carries the
  closing auction and a slightly different volume. This is the same seam
  `dayrange_model` documents and accepts, and it reaches only the lagged
  history block.

Nothing here does I/O except the two `fetch_*` helpers, and nothing here knows
about Streamlit: the trader, the chart overlay and SimLab all call
`forecast_session` with inputs they assembled themselves.
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import market_hours
from .model_store import ModelStore

# Both LightGBM and PyTorch wheels bundle their own OpenMP runtime and this
# environment has no system `libomp`, which makes loading one in a process that
# already initialised the other a segfault with no traceback. Capping the
# thread count before either is loaded is what defuses it, and it is what
# `dayrange_model` and the notebooks' own `__init__` do for the same reason.
#
# This module never touches torch, so it only has the "LightGBM after torch"
# edge of that trap, and it cannot close it alone: a process that imported
# `nbeats_model` first has already loaded torch's runtime before this line
# runs. Importing LightGBM eagerly here is the part that *is* in reach --
# it happens at `import pricerange_model`, which any caller can order, rather
# than in the middle of a `joblib.load` deep inside a trading loop.
os.environ.setdefault("OMP_NUM_THREADS", "1")

try:  # pragma: no cover - depends on which optional extras are installed
    importlib.import_module("lightgbm")
except ImportError:
    pass


MODEL_PATH_ENV = "APPLE_PRICERANGE_MODEL"
DEFAULT_TICKER = "AAPL"

# The decision minute, and the window behind it. The window is [09:30, 09:35):
# five bars stamped 09:30..09:34. The bar stamped 09:35 covers 09:35-09:36 and
# has not finished at the decision instant, so taking it would be a minute of
# hindsight -- `pricerange.data.opening_window` says exactly this.
OPENING_MINUTES = 5

# Calendar days of daily history to ask for, to clear `MIN_DAILY_SESSIONS`
# trading days with room for holidays. Same figure `dayrange_model` uses.
DAILY_HISTORY_DAYS = 420

# The exact warm-up: 252-day rolling windows, then a one-day lag. Checked
# against the shipped panels rather than reasoned about -- their first row with
# all 156 features finite is number 253 on all three tickers.
MIN_DAILY_SESSIONS = 253

# `open_vol_vs_21` divides today's opening volume by its own 21-session mean,
# and `rolling(21)` needs all 21 -- 20 gives NaN, not a slightly worse number.
OPENING_HISTORY_SESSIONS = 21

# Calendar days to walk back looking for them. 21 sessions is a shade over four
# weeks; 40 clears a fortnight of holidays without ever running long, since the
# walk stops as soon as it has enough.
OPENING_HISTORY_LOOKBACK_DAYS = 40

EPS = 1e-12
SHORT_WINDOWS = (5, 10, 21)
LONG_WINDOWS = (63, 126, 252)

EDGES = ["y_high", "y_low"]
QUANTILES = (0.25, 0.5, 0.75)

# The 13 tradeable ETFs, which PriceRange2 takes from Alpaca and this module
# takes from yfinance (see the module docstring for what that costs).
CROSS_ETFS = (
    "SPY", "QQQ", "XLK", "IWM", "GLD", "USO", "SLV",
    "DBC", "UNG", "TLT", "HYG", "UUP", "VXX",
)

# The four that are indices rather than ETFs, so no vendor carries them as
# tradeable symbols. Mapped feed symbol -> the name the model's columns use,
# which is `config.YAHOO_EXTRAS` in PriceRange2 and is load-bearing: the
# feature is called `vix_level`, not `^VIX_level`.
CROSS_INDICES = {"^VIX": "vix", "^VVIX": "vvix", "^TNX": "us10y", "DX-Y.NYB": "dxy"}

#: Every cross-asset series the shipped feature set needs, as
#: {feed symbol -> column name}. 17 of them, 110 of the 156 columns.
CROSS_SYMBOLS: "dict[str, str]" = {
    **{symbol: symbol for symbol in CROSS_ETFS},
    **CROSS_INDICES,
}


# =========================================================================
# --- features
#
# Verbatim from `pricerange/features.py`. See the mirror contract above; do
# not "improve" anything below without retraining.
# =========================================================================

def _safe_div(a, b):
    return a / b.replace(0, np.nan)


def _lag(df: pd.DataFrame) -> pd.DataFrame:
    """Shift a frame one session, so day t sees only day t-1 and earlier."""
    return df.shift(1)


def history_features(sess: pd.DataFrame) -> pd.DataFrame:
    """Rolled daily statistics, lagged a day. The backbone of the model.

    Range begets range: a stock that has been swinging 3% a day is the single
    best clue that today will swing too. Most of this block is that idea at
    several horizons, plus where the price sits in its own recent range.
    """
    d = sess.copy()
    rng = np.log(_safe_div(d["high"], d["low"]))
    ret = np.log(_safe_div(d["close"], d["close"].shift(1)))
    gap_hist = np.log(_safe_div(d["open"], d["close"].shift(1)))
    body = (d["close"] - d["open"]).abs() / d["close"]

    f = pd.DataFrame(index=d.index)
    f["prev_range"] = rng
    f["prev_ret"] = ret
    f["prev_absret"] = ret.abs()
    f["prev_gap"] = gap_hist.abs()
    f["prev_body"] = body

    for w in SHORT_WINDOWS + LONG_WINDOWS:
        f[f"range{w}"] = rng.rolling(w).mean()
        f[f"vol{w}"] = ret.rolling(w).std()
    for w in SHORT_WINDOWS:
        f[f"range{w}_sd"] = rng.rolling(w).std()
        f[f"absret{w}"] = ret.abs().rolling(w).mean()

    # Volatility regime: is the short window running hot against the long one?
    f["range_ratio_5_63"] = _safe_div(f["range5"], f["range63"])
    f["range_ratio_21_252"] = _safe_div(f["range21"], f["range252"])
    f["vol_ratio_5_63"] = _safe_div(f["vol5"], f["vol63"])

    # Trend and location, which say something about how jumpy the tape is.
    for w in (21, 63, 126):
        f[f"mom{w}"] = np.log(_safe_div(d["close"], d["close"].shift(w)))
    hi252, lo252 = d["high"].rolling(252).max(), d["low"].rolling(252).min()
    f["pos_252"] = (
        (d["close"] - lo252) / (hi252 - lo252).where(lambda s: s > EPS)
    ).fillna(0.5)
    f["dist_hi252"] = np.log(_safe_div(d["close"], hi252))

    # Volume, in units of its own recent normal.
    f["vol_vs_21"] = _safe_div(d["volume"], d["volume"].rolling(21).mean())

    # Calendar. Range has a weekday shape and a month-end shape.
    out = _lag(f)
    out["dow"] = d.index.dayofweek
    out["month"] = d.index.month
    out["is_month_end"] = (
        d.index.to_period("M") != (d.index + pd.Timedelta(days=3)).to_period("M")
    ).astype(int)
    return out


def opening_features(sess: pd.DataFrame) -> pd.DataFrame:
    """The overnight gap and the first five minutes - the only same-day inputs.

    The opening range is the most direct evidence available about today in
    particular: a wide, heavily traded first five minutes says the day has
    something to argue about.
    """
    d = sess.copy()
    prev_close = d["close"].shift(1)
    prev_range = np.log(_safe_div(d["high"], d["low"])).shift(1)
    adr21 = np.log(_safe_div(d["high"], d["low"])).rolling(21).mean().shift(1)

    f = pd.DataFrame(index=d.index)
    f["gap"] = np.log(_safe_div(d["open_open"], prev_close))
    f["gap_abs"] = f["gap"].abs()
    f["gap_vs_adr"] = _safe_div(f["gap_abs"], adr21)

    f["open_range"] = np.log(_safe_div(d["open_high"], d["open_low"]))
    f["open_range_vs_adr"] = _safe_div(f["open_range"], adr21)
    f["open_range_vs_prev"] = _safe_div(f["open_range"], prev_range)
    f["open_ret_5m"] = d["open_ret"]
    f["open_rv_5m"] = d["open_rv"]

    # Where the 09:35 price sits inside the first five minutes: at the top of
    # the opening range, at the bottom, or in the middle.
    span = d["open_high"] - d["open_low"]
    f["open_pos"] = ((d["ref"] - d["open_low"]) / span.where(span > EPS)).fillna(0.5)

    # Opening volume against its own recent normal - a proxy for how much the
    # market cares about today.
    f["open_vol_vs_21"] = _safe_div(
        d["open_volume"], d["open_volume"].rolling(21).mean().shift(1)
    )
    f["open_vol_share"] = _safe_div(
        d["open_volume"], d["volume"].rolling(21).mean().shift(1)
    )
    return f


def cross_asset_features(sess: pd.DataFrame, cross: pd.DataFrame) -> pd.DataFrame:
    """What the rest of the market did yesterday, and its gap this morning.

    Two different claims are being tested here. The lagged part asks whether
    yesterday's macro tape (VIX, credit, oil, the dollar) predicts today's
    single-stock range. The gap part asks whether *this morning's* move in SPY
    and QQQ - which is known at 09:35, because they open at 09:30 too - carries
    information beyond the stock's own open.
    """
    wide = cross.pivot(index="date", columns="symbol")
    idx = sess.index
    cols: "dict[str, pd.Series]" = {}

    for sym in cross["symbol"].unique():
        try:
            c = wide[("close", sym)].reindex(idx).ffill()
            h = wide[("high", sym)].reindex(idx).ffill()
            l = wide[("low", sym)].reindex(idx).ffill()
            o = wide[("open", sym)].reindex(idx).ffill()
        except KeyError:
            continue

        ret = np.log(_safe_div(c, c.shift(1)))
        rng = np.log(_safe_div(h, l))

        # Lagged: strictly yesterday and earlier.
        cols[f"{sym}_ret"] = ret.shift(1)
        cols[f"{sym}_absret"] = ret.abs().shift(1)
        cols[f"{sym}_range"] = rng.shift(1)
        cols[f"{sym}_range21"] = rng.rolling(21).mean().shift(1)
        cols[f"{sym}_range_ratio"] = _safe_div(rng, rng.rolling(21).mean()).shift(1)

        # This morning's gap. The open prints at 09:30, five minutes before the
        # decision, so it is legitimately available.
        cols[f"{sym}_gap"] = np.log(_safe_div(o, c.shift(1)))

    # Levels, not just changes, for the fear gauges - VIX at 30 is a different
    # world from VIX at 12 regardless of which way it moved yesterday.
    for name in ("vix", "vvix"):
        if ("close", name) in wide.columns:
            lvl = wide[("close", name)].reindex(idx).ffill()
            cols[f"{name}_level"] = lvl.shift(1)
            cols[f"{name}_chg"] = np.log(_safe_div(lvl, lvl.shift(1))).shift(1)
            cols[f"{name}_vs_21"] = _safe_div(lvl, lvl.rolling(21).mean()).shift(1)

    # The stock against its two natural benchmarks: is it moving on its own
    # account?
    own = np.log(_safe_div(sess["close"], sess["close"].shift(1))).shift(1)
    for bench in ("QQQ", "XLK"):
        if f"{bench}_ret" in cols:
            cols[f"excess_vs_{bench}"] = own - cols[f"{bench}_ret"]

    return pd.DataFrame(cols, index=idx)


def build_features(sess: pd.DataFrame, cross: pd.DataFrame) -> pd.DataFrame:
    """The three shipped blocks, side by side, indexed by session date.

    `pricerange.features.build_panel` also attaches the targets and the two
    blocks that were measured and left out (news, world events). Neither is
    needed to *use* the model, and the targets are the day's outcome, so this
    is the inference-time half of it.
    """
    return pd.concat(
        [
            history_features(sess),
            opening_features(sess),
            cross_asset_features(sess, cross),
        ],
        axis=1,
    )


# =========================================================================
# --- inference
#
# Verbatim from `pricerange/modeling.py`, minus everything about fitting.
# =========================================================================

def clip_to_opening(pred: pd.DataFrame, sess: pd.DataFrame) -> pd.DataFrame:
    """Force the prediction to contain the opening range.

    By 09:35 the day has already printed a high and a low. A predicted high
    below the one already on the tape is not a forecast, it is a contradiction.
    Free accuracy -- no model, no fitting, just arithmetic the market has
    already done -- and it matters most on exactly the volatile mornings where
    the model is least sure.
    """
    out = pred.copy()
    ref = sess.loc[out.index, "ref"]
    open_hi = np.log(sess.loc[out.index, "open_high"] / ref)
    open_lo = np.log(sess.loc[out.index, "open_low"] / ref)
    out["y_high"] = np.maximum(out["y_high"], open_hi)
    out["y_low"] = np.minimum(out["y_low"], open_lo)
    return out


def predict(models: dict, feature_cols: "list[str]", sess: pd.DataFrame,
            feats: pd.DataFrame, clip: bool = True) -> pd.DataFrame:
    """Predicted edges for every row, in log units against `ref`."""
    X = feats[feature_cols]
    out = pd.DataFrame(
        {e: models[e].predict(X) for e in EDGES}, index=feats.index
    )
    return clip_to_opening(out, sess) if clip else out


def predict_quantiles(models: dict, feature_cols: "list[str]", sess: pd.DataFrame,
                      feats: pd.DataFrame, clip: bool = True) -> pd.DataFrame:
    """Predicted edges at every fitted quantile, as columns `y_high_q25` etc.

    Note the asymmetry in how a rule uses them. For a *buy near the low*, a
    higher quantile of `y_low` is the more conservative level -- it sits closer
    to the 09:35 price, so it fills more often. For a *sell near the high*, a
    lower quantile of `y_high` does the same job.
    """
    X = feats[feature_cols]
    cols = {}
    for e in EDGES:
        for q in QUANTILES:
            key = f"q{int(q * 100)}_{e}"
            if key in models:
                cols[f"{e}_q{int(q * 100)}"] = models[key].predict(X)
    out = pd.DataFrame(cols, index=feats.index)

    if clip and len(out.columns):
        ref = sess.loc[out.index, "ref"]
        open_hi = np.log(sess.loc[out.index, "open_high"] / ref)
        open_lo = np.log(sess.loc[out.index, "open_low"] / ref)
        for c in out.columns:
            if c.startswith("y_high"):
                out[c] = np.maximum(out[c], open_hi)
            else:
                out[c] = np.minimum(out[c], open_lo)
    return out


def to_levels(pred: pd.DataFrame, sess: pd.DataFrame) -> pd.DataFrame:
    """Turn log-edge predictions into dollar price levels."""
    ref = sess.loc[pred.index, "ref"]
    return pd.DataFrame(
        {
            "ref": ref,
            "pred_high": ref * np.exp(pred["y_high"]),
            "pred_low": ref * np.exp(pred["y_low"]),
            "pred_range_pct": np.expm1(pred["y_high"] - pred["y_low"]),
        },
        index=pred.index,
    )


# =========================================================================
# --- the saved bundle
# =========================================================================

# PriceRange2 fits one model per ticker and makes no claim that one transfers,
# so `pricerange2_intc.joblib` is a different model rather than the same one
# pointed elsewhere. The lower-case filename is the notebook's; see
# `ModelStore.lowercase_file`.
_STORE = ModelStore(
    env_key=MODEL_PATH_ENV,
    filename="pricerange2_{ticker}.joblib",
    build=lambda path: _build_bundle(path),
    lowercase_file=True,
)

model_path = _STORE.path
metadata_path = _STORE.metadata_path


def _build_bundle(path: Path) -> "dict | None":
    """One ticker's saved model plus its metadata, or None when it cannot be
    assembled.

    Refuses rather than degrades in three cases, each of which would otherwise
    produce a forecast that looks fine and is not the model that was measured:

    * a missing joblib, or a joblib that is not this bundle's shape;
    * a missing point estimator -- `y_high` and `y_low` *are* the forecast;
    * a `feature_cols` list that is not the 156 the metadata claims, which is
      what a half-written file or a mismatched retrain looks like from here.

    A missing *quantile* model is not fatal, because the point forecast is the
    part that was scored -- but it is recorded in `quantiles`, and the trader
    checks it before arming, since the quantiles are the levels it trades.
    """
    try:
        import joblib
    except ImportError:
        return None
    try:
        saved = joblib.load(path)
    except (OSError, ValueError, KeyError, ModuleNotFoundError, AttributeError):
        return None
    if not isinstance(saved, dict) or {"models", "feature_cols"} - set(saved):
        return None

    models = saved["models"]
    feature_cols = list(saved["feature_cols"])
    if not isinstance(models, dict) or any(e not in models for e in EDGES):
        return None
    if not feature_cols:
        return None

    meta_file = metadata_path(path)
    try:
        metadata = json.loads(meta_file.read_text()) if meta_file.exists() else {}
    except (OSError, ValueError):
        metadata = {}

    claimed = metadata.get("n_features")
    if claimed is not None and int(claimed) != len(feature_cols):
        return None

    quantiles = sorted(
        q for q in QUANTILES
        if all(f"q{int(q * 100)}_{e}" in models for e in EDGES)
    )
    return {
        "models": models,
        "feature_cols": feature_cols,
        "metadata": metadata,
        "quantiles": quantiles,
        "opening_minutes": OPENING_MINUTES,
        "trained_at": metadata.get("created"),
        "path": str(path),
        # Which symbol this bundle was fitted on, as the file itself recorded
        # it -- not the ticker that asked for it. A mismatch is worth being
        # able to see rather than inferring from the filename.
        "ticker": str(metadata.get("ticker") or "").upper() or None,
    }


load_bundle = _STORE.load
reset_bundle_cache = _STORE.reset


def opening_minutes(bundle: "dict | None" = None) -> int:
    """The opening window this bundle's features were built on."""
    try:
        return int((bundle or {})["opening_minutes"])
    except (KeyError, TypeError, ValueError):
        return OPENING_MINUTES


def has_quantiles(bundle: "dict | None") -> bool:
    """Whether both trading levels can be had from this bundle.

    0.25 and 0.75 are the two the rule reads -- a 75th-percentile low to buy at
    and a 25th-percentile high to sell at. The 0.5 pair is the point forecast
    by another name and is not needed for it.
    """
    return {0.25, 0.75} <= set((bundle or {}).get("quantiles") or ())


# =========================================================================
# --- assembling one session's inputs
# =========================================================================

def daily_frame_from_bars(bars: "list[dict]") -> pd.DataFrame:
    """`{"t","o","h","l","c","v"}` daily bars -> the notebook's session frame:
    a tz-naive midnight index named `date`, lower-case OHLCV.

    Deliberately a copy of `dayrange_model.daily_frame_from_bars` rather than
    an import of it: that module pulls PyTorch in at import, and this model
    needs none of it. Twenty lines is a cheaper price than a 200 MB dependency
    on every process that wants a range forecast.
    """
    columns = ["open", "high", "low", "close", "volume"]
    if not bars:
        return pd.DataFrame(columns=columns, index=pd.DatetimeIndex([], name="date"))
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime([str(b.get("t", ""))[:10] for b in bars]),
            "open": [float(b["o"]) for b in bars],
            "high": [float(b["h"]) for b in bars],
            "low": [float(b["l"]) for b in bars],
            "close": [float(b["c"]) for b in bars],
            "volume": [float(b.get("v") or 0.0) for b in bars],
        }
    )
    frame = frame.dropna(subset=["date"]).drop_duplicates("date", keep="last")
    return frame.sort_values("date").set_index("date")[columns]


def session_frame(
    history: pd.DataFrame,
    opening_bars: pd.DataFrame,
    session_date,
    open_price: "float | None" = None,
    opening_volume: "pd.Series | None" = None,
) -> pd.DataFrame:
    """`history` with today appended as a *partially known* row.

    This mirrors `predict_today._partial_session`, and the part that matters is
    what is left out. Today's `high`, `low`, `close` and `volume` are the day's
    outcome, they do not exist at 09:35, and they are set to NaN here rather
    than filled in with the opening window's values.

    That is not caution, it is correctness. `history_features` lags every
    rolled statistic by a day, so today's row never reaches today's own
    features -- but `opening_features` reads `close.shift(1)` and
    `cross_asset_features` reads `close.pct_change().shift(1)`, and a row
    carrying five minutes' worth of high/low/close where the model was fitted
    on a whole session's would quietly corrupt *tomorrow's* forecast if this
    frame were ever reused. NaN cannot.

    The columns today's row *does* carry are the ones known by 09:35: the
    official open, and everything the opening window produced.

    `open_price` is today's official opening print when the caller could get
    one (`historical.fetch_session_open`). Without it the first regular-session
    minute bar's open stands in -- the same number on about half of all
    sessions and a few basis points off otherwise, the auction print against
    the first trade the feed happened to see.

    `opening_volume` is the one opening-window column the *prior* rows need: a
    Series of past sessions' first-five-minute volume, indexed by date, from
    `fetch_opening_volume_history`. Without it `open_vol_vs_21` is NaN and
    `forecast_session` refuses. Every other opening feature reads only today's
    window and the daily history, which is why this is the single extra input
    rather than a whole second table.
    """
    day = pd.Timestamp(session_date).normalize()
    prior = history[history.index < day].copy()
    if opening_volume is not None and len(opening_volume):
        volume = pd.Series(opening_volume, dtype="float64")
        volume.index = pd.to_datetime(volume.index).normalize()
        prior["open_volume"] = volume.reindex(prior.index)

    window = opening_bars
    first_open = float(window["open"].iloc[0])
    ref = float(window["close"].iloc[-1])
    r = np.log(window["close"].to_numpy(dtype=float) / window["open"].to_numpy(dtype=float))
    open_open = float(open_price) if open_price else first_open

    today = pd.DataFrame(
        {
            # `open` is the day's official open and feeds `prev_gap` tomorrow;
            # `open_open` is the opening window's first print and feeds `gap`
            # today. PriceRange2 keeps them separate and so must this.
            "open": [open_open],
            "high": [np.nan],
            "low": [np.nan],
            "close": [np.nan],
            "volume": [np.nan],
            "open_open": [open_open],
            "open_high": [float(window["high"].max())],
            "open_low": [float(window["low"].min())],
            "ref": [ref],
            "open_volume": [float(window["volume"].sum())],
            "open_rv": [float(np.sqrt((r ** 2).sum()))],
            "open_ret": [float(np.log(ref / open_open))],
        },
        index=pd.DatetimeIndex([day], name="date"),
    )
    return pd.concat([prior, today])


def cross_frame_from_bars(
    bars_by_name: "dict[str, list[dict]]",
    session_date,
    opens: "dict[str, float] | None" = None,
) -> pd.DataFrame:
    """The long cross-asset frame `cross_asset_features` reads.

    Columns: date, symbol, open, high, low, close, volume -- long rather than
    wide because the feature builder applies the same transform per symbol.

    Today's row per symbol carries **only the open**, for the same reason the
    stock's does: `{sym}_gap` is the one same-day cross column and it reads the
    open alone. Everything else in the block is `.shift(1)`, so today's
    high/low/close are never read and are left NaN rather than guessed at.
    """
    day = pd.Timestamp(session_date).normalize()
    opens = opens or {}
    frames = []
    for name, bars in bars_by_name.items():
        daily = daily_frame_from_bars(bars)
        daily = daily[daily.index < day]
        if not len(daily):
            continue
        frame = daily.reset_index()
        today_open = opens.get(name)
        if today_open is not None and np.isfinite(today_open):
            frame = pd.concat(
                [
                    frame,
                    pd.DataFrame(
                        {
                            "date": [day], "open": [float(today_open)],
                            "high": [np.nan], "low": [np.nan],
                            "close": [np.nan], "volume": [np.nan],
                        }
                    ),
                ],
                ignore_index=True,
            )
        frame["symbol"] = name
        frames.append(frame)

    columns = ["date", "symbol", "open", "high", "low", "close", "volume"]
    if not frames:
        return pd.DataFrame(columns=columns)
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["symbol", "date"]).reset_index(drop=True)[columns]


def fetch_cross_frame(session_date, days: int = DAILY_HISTORY_DAYS) -> pd.DataFrame:
    """Pull all 17 cross-asset series, and today's opens, into one frame.

    The only network I/O in this module. `historical` is imported inside the
    call so that SimLab's patched `fetch_daily_ohlc_bars` / `fetch_session_open`
    are the ones used -- `simlab.patches` replaces those attributes on the
    module, and binding them at import time would freeze the live versions in.

    34 fetches sounds heavy and is not, in practice: `historical` caches daily
    bars for an hour and today's open for five minutes, and the forecast is
    made once a session. A symbol that cannot be had is simply absent from the
    frame, which `require_cross` then reports by name rather than letting it
    become 6 silent NaN columns.

    **This is a live-only helper, and `session_date` is checked rather than
    trusted.** `historical.fetch_session_open` answers about *today* whatever
    it is asked -- it has no date parameter, because the live app never wants
    another day's. So a call naming a past session would otherwise staple this
    morning's SPY open onto that session's row and compute `SPY_gap` from two
    days that never met: a plausible-looking number with nothing behind it, on
    the one same-day column the block has. When `session_date` is not today the
    opens are left off, and `require_cross` then refuses by name. A replay that
    genuinely wants a past morning's opens has them in its own store and builds
    the frame with `cross_frame_from_bars` -- which is what SimLab does.
    """
    from . import historical
    from .clock import now as _now

    day = pd.Timestamp(session_date).normalize()
    today = pd.Timestamp(_now().astimezone(market_hours.MARKET_TZ).date())

    bars_by_name: "dict[str, list[dict]]" = {}
    opens: "dict[str, float]" = {}
    for symbol, name in CROSS_SYMBOLS.items():
        bars = historical.fetch_daily_ohlc_bars(symbol, days=days)
        if bars:
            bars_by_name[name] = bars
        if day != today:
            continue
        today_open = historical.fetch_session_open(symbol)
        if today_open is not None:
            opens[name] = float(today_open)
    return cross_frame_from_bars(bars_by_name, day, opens)


_opening_volume_cache: "dict[tuple, pd.Series]" = {}


def fetch_opening_volume_history(
    ticker: str,
    session_date,
    key: "str | None",
    secret: "str | None",
    feed: str = "iex",
    sessions: int = OPENING_HISTORY_SESSIONS,
    lookback_days: int = OPENING_HISTORY_LOOKBACK_DAYS,
) -> pd.Series:
    """Opening-window volume for the `sessions` completed sessions before
    `session_date`, indexed by date.

    Walks backwards a calendar day at a time asking for that day's
    `[09:30, 09:35)` alone -- five bars, not a session -- through
    `rest.fetch_bars_window`, which SimLab patches to the dataset's stored bars
    so a replay reads the history the live agent would have had. A weekend or
    holiday returns nothing and costs one step of the walk.

    Twenty-one tiny requests rather than one big one on purpose:
    `fetch_bars_window` does not paginate, and a month of 1-minute bars is
    about 12,000 -- past Alpaca's 10,000-row page and so silently truncated at
    the wrong end. Five bars a day is never near it.

    Cached per (symbol, session, feed) since the answer cannot change during
    the session, and evicted a session at a time for the reason
    `momentum_change_model.history_bars` gives: it is the *date* that goes
    stale, and evicting per symbol would make two streamed symbols re-download
    each other's history.

    Returns an empty Series when credentials are missing or nothing came back;
    `require_opening_volume` is what turns that into a refusal.
    """
    from . import agent as agent_mod

    symbol = str(ticker or DEFAULT_TICKER).upper()
    day0 = pd.Timestamp(session_date).normalize()
    cache_key = (symbol, day0.date(), str(feed or "").lower(), int(sessions))
    cached = _opening_volume_cache.get(cache_key)
    if cached is not None:
        return cached

    collected: "dict[pd.Timestamp, float]" = {}
    if key and secret:
        tz = market_hours.MARKET_TZ
        day = day0
        for _ in range(lookback_days):
            day = day - pd.Timedelta(days=1)
            if day.dayofweek >= 5:  # free: the walk need not ask about weekends
                continue
            start = pd.Timestamp(day).tz_localize(tz).replace(
                hour=market_hours.MARKET_OPEN.hour,
                minute=market_hours.MARKET_OPEN.minute,
            )
            try:
                bars = agent_mod.fetch_bars_window(
                    symbol, "1Min", start.to_pydatetime(),
                    (start + pd.Timedelta(minutes=OPENING_MINUTES)).to_pydatetime(),
                    key, secret, feed, limit=OPENING_MINUTES * 2,
                )
            except Exception:
                bars = []
            volume = sum(float(b.get("v") or 0.0) for b in bars)
            # A session that printed no opening volume is not a session -- it
            # is a holiday, a half day that opened late, or a gap in the feed.
            # Recording a 0 would put a zero into a 21-day mean.
            if bars and volume > 0:
                collected[day] = volume
            if len(collected) >= sessions:
                break

    out = pd.Series(collected, dtype="float64").sort_index()
    out.index = pd.DatetimeIndex(out.index, name="date")
    for stale in [k for k in _opening_volume_cache if k[1] != cache_key[1]]:
        _opening_volume_cache.pop(stale, None)
    _opening_volume_cache[cache_key] = out
    return out


def reset_opening_volume_cache() -> None:
    """Forget the fetched opening volumes -- for tests, and for a long-lived
    process that has crossed midnight."""
    _opening_volume_cache.clear()


def require_opening_volume(
    opening_volume: "pd.Series | None",
    session_date,
    sessions: int = OPENING_HISTORY_SESSIONS,
) -> "str | None":
    """Why `open_vol_vs_21` cannot be built, or None if it can.

    Twenty of the twenty-one is not "nearly": `rolling(21)` returns NaN, and a
    NaN reaches LightGBM as a missing value it was never trained to see in that
    column, so the refusal has to happen here.
    """
    day = pd.Timestamp(session_date).normalize()
    have = 0
    if opening_volume is not None and len(opening_volume):
        index = pd.DatetimeIndex(pd.to_datetime(opening_volume.index)).normalize()
        have = int((index < day).sum())
    if have >= sessions:
        return None
    return (
        f"only {have} of the {sessions} previous sessions' opening volume could be "
        "fetched; `open_vol_vs_21` compares this morning's first five minutes "
        "against their 21-session mean and cannot be built from fewer. "
        "Alpaca credentials are needed for it — yfinance serves 1-minute history "
        "for 30 calendar days, which is 20 sessions, one short every day."
    )


def require_history(daily: pd.DataFrame) -> "str | None":
    """Why this daily frame cannot support a forecast, or None if it can.

    Checked before anything is computed, because the failure it prevents is the
    quiet one: LightGBM eats NaNs, so a short history produces a number rather
    than an error -- one built off a 60-day extreme where the model was fitted
    on a 252-day one.
    """
    completed = len(daily.dropna(subset=["close"]))
    if completed < MIN_DAILY_SESSIONS:
        return (
            f"only {completed} completed daily sessions of history; the forecast "
            f"needs {MIN_DAILY_SESSIONS} for its 252-day windows to warm up."
        )
    return None


def require_cross(cross: pd.DataFrame, session_date) -> "str | None":
    """Why the cross-asset block cannot be built, or None if it can.

    Two failures, and both have to be refusals rather than warnings. The
    cross block is 110 of the 156 columns and LightGBM will predict off a
    row of NaNs without complaint, so a missing series does not announce
    itself -- it just quietly moves the answer.

    The second check is the one SimLab hits: a dataset recorded for AAPL alone
    has the stock's bars and none of the market's, so the history is there and
    the opens are not.
    """
    day = pd.Timestamp(session_date).normalize()
    wanted = set(CROSS_SYMBOLS.values())
    if cross.empty:
        return (
            "no cross-asset data at all; the forecast reads 17 other markets "
            "(110 of its 156 features) and cannot be made without them."
        )

    present = set(cross["symbol"].unique())
    missing = sorted(wanted - present)
    if missing:
        return (
            f"{len(missing)} of the 17 cross-asset series are missing "
            f"({', '.join(missing[:5])}{'…' if len(missing) > 5 else ''}); "
            "each contributes 6 features the model was fitted with."
        )

    today = cross[cross["date"] == day]
    have_open = set(today.loc[today["open"].notna(), "symbol"].unique())
    no_open = sorted(wanted - have_open)
    if no_open:
        return (
            f"no opening print today for {len(no_open)} of the 17 cross-asset "
            f"series ({', '.join(no_open[:5])}{'…' if len(no_open) > 5 else ''}); "
            "their `_gap` columns are the block's only same-day features and "
            "`UNG_gap` alone is the second most important feature on AAPL."
        )
    return None


def volume_scale_warning(feed: "str | None") -> "str | None":
    """Whether this tape's minute volume is on the scale the model was fitted
    on.

    Two features read minute volume, and **only one of them is at risk**, which
    is worth stating precisely because the obvious reading is wrong:

    * `open_vol_vs_21` divides today's opening volume by the mean of the last
      21 sessions' opening volume, and both now come from the same feed --
      `fetch_opening_volume_history` walks the same tape the live buffer is on.
      A ratio of two quantities on the same scale is scale-free, so this one is
      fine on IEX.
    * `open_vol_share` divides today's opening volume by the mean of the last
      21 sessions' *daily* volume, which arrives from yfinance and is
      consolidated. That one genuinely mixes scales: on IEX the numerator is
      roughly 4% of what the model was fitted with, i.e. about log(0.04) = -3.2
      out in log space.

    So the warning names the one feature that is actually wrong rather than
    both. This is also a softer failure than the equivalent in `dayrange_model`,
    where the feature feeds a ridge that extrapolates linearly: a tree splits on
    thresholds, so an out-of-range value lands in the outermost leaf and stays
    there. It is still the wrong leaf, on every bar.
    """
    if str(feed or "").lower() != "iex":
        return None
    return (
        "the IEX feed carries about 4% of consolidated minute volume. "
        "`open_vol_share` measures this morning's opening volume against a "
        "consolidated 21-day daily average, so on IEX it lands far below anything "
        "the model saw in training (`open_vol_vs_21` is a same-feed ratio and is "
        "unaffected). Run on the SIP or yfinance tape."
    )


def forecast_session(
    bundle: dict,
    history: pd.DataFrame,
    opening_bars: pd.DataFrame,
    session_date,
    cross: "pd.DataFrame | None" = None,
    open_price: "float | None" = None,
    opening_volume: "pd.Series | None" = None,
) -> dict:
    """The day's predicted high and low, plus the levels a rule needs.

    `history` is completed daily bars for the stock (today's row is ignored if
    present), `opening_bars` the first `opening_minutes` regular-session
    1-minute bars of `session_date`, `cross` the long cross-asset frame from
    `fetch_cross_frame`, `opening_volume` the previous 21 sessions' opening
    volume from `fetch_opening_volume_history`, and `open_price` today's
    official opening print if one could be had.

    Returns dollars throughout::

        pred_high, pred_low   the point forecast, clipped to contain the
                              opening range
        pred_range_pct        their difference, as a fraction of `ref`
        ref                   the 09:35 price everything is measured against
        open_high, open_low   the opening range, for a caller that wants to
                              show what the clip was against
        buy_edge, sell_edge   the q75 low and q25 high -- the fill-friendlier
                              levels PriceRange2's sweep found were the only
                              ones that ever made money. Absent (None) on a
                              bundle without quantile models.

    Raises ValueError when the inputs cannot support a forecast, rather than
    returning a number built off NaNs -- the caller turns that into a logged
    refusal to trade.
    """
    want = opening_minutes(bundle)
    if len(opening_bars) < want:
        raise ValueError(
            f"the forecast is built on the first {want} minutes and only "
            f"{len(opening_bars)} bars have closed."
        )
    opening_bars = opening_bars.iloc[:want]

    day = pd.Timestamp(session_date).normalize()
    sess = session_frame(
        history, opening_bars, session_date, open_price, opening_volume
    )
    for problem in (
        require_history(sess),
        require_opening_volume(opening_volume, day),
        require_cross(cross if cross is not None else pd.DataFrame(), day),
    ):
        if problem is not None:
            raise ValueError(problem)
    cross = cross if cross is not None else pd.DataFrame()

    feats = build_features(sess, cross)
    feature_cols = bundle["feature_cols"]
    unknown = [c for c in feature_cols if c not in feats.columns]
    if unknown:
        raise ValueError(
            f"the feature table is missing {len(unknown)} of the model's columns "
            f"({', '.join(unknown[:4])}{'…' if len(unknown) > 4 else ''}); the "
            "mirrored feature code and the saved model have drifted apart."
        )

    row = feats.loc[[day], feature_cols]
    missing = [c for c in feature_cols if not np.isfinite(row[c].iloc[0])]
    if missing:
        raise ValueError(
            "the feature row is incomplete "
            f"({', '.join(missing[:4])}{'…' if len(missing) > 4 else ''}); "
            "the daily history is too short or the cross-asset tape has gaps."
        )

    models = bundle["models"]
    point = predict(models, feature_cols, sess, feats.loc[[day]])
    levels = to_levels(point, sess).iloc[0]

    buy_edge = sell_edge = None
    if has_quantiles(bundle):
        q = predict_quantiles(models, feature_cols, sess, feats.loc[[day]]).iloc[0]
        ref = float(levels["ref"])
        buy_edge = float(ref * np.exp(q["y_low_q75"]))
        sell_edge = float(ref * np.exp(q["y_high_q25"]))

    return {
        "pred_high": float(levels["pred_high"]),
        "pred_low": float(levels["pred_low"]),
        "pred_range_pct": float(levels["pred_range_pct"]),
        "ref": float(levels["ref"]),
        "open_high": float(sess.loc[day, "open_high"]),
        "open_low": float(sess.loc[day, "open_low"]),
        "buy_edge": buy_edge,
        "sell_edge": sell_edge,
    }


def session_bars(frame: pd.DataFrame, session_date) -> pd.DataFrame:
    """Regular-session minute bars belonging to one date."""
    day = pd.Timestamp(session_date).date()
    if not len(frame):
        return frame
    return frame[frame.index.date == day]


def market_date() -> "pd.Timestamp":
    """Today, in the exchange's timezone -- the session a forecast belongs to."""
    from .clock import now as _now

    return pd.Timestamp(_now().astimezone(market_hours.MARKET_TZ).date())
