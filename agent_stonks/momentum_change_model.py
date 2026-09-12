"""How far momentum is about to move, in bps/min (TimeToChange / momlib).

The other three models here answer a yes/no question about a regime change --
will this one hold, is one about to happen, where will the day's range land.
This one answers a *quantity*: given the minute that just closed, how much will
the smoothed 15-minute momentum score differ fifteen bars from now?

    mom_delta = mom[t + 15] - mom[t - 1]      # bps per minute

That is a signed number rather than a probability, which is the whole reason it
is a separate strategy: a positive prediction on a bar whose regime is
**negative** is the model calling a turn upwards, and a negative one on a
**positive** bar is it calling the move over. `apple_trader.MomentumChangeTrader`
buys the first and sells the second, exactly as `momlib/sim.py` does.

One bundle per ticker, and the estimator is chosen per ticker
--------------------------------------------------------------
TimeToChange fits *and selects* per ticker on that ticker's own validation
days, so a bundle is a symbol's model rather than a shared one pointed
elsewhere -- and the selection really does differ, which is the argument for
keeping them separate:

    AAPL    Ridge                R2 0.66 validation / 0.48 holdout week,
                                 sign right on 94% of holdout changes
    GOOGL   RandomForest         R2 0.48 / 0.19, sign right on 91%
    INTC    HistGradientBoosting R2 0.60 / 0.43, sign right on 86%

All three beat the predict-zero baseline on days used neither for fitting nor
for selection, which is the claim worth making about them.

Read the holdout column, not the validation one. GOOGL's is the weakest of the
three and it is also the one that falls furthest from its own validation score
-- a 0.48 that becomes 0.19 is the split doing its job, and the reason the
estimator is chosen on validation days and then reported on a week neither step
touched.

**GOOGL has a bundle again but is still not wired up.** It was withdrawn once,
when there was no file behind it; the refit on the full archive put the file
back, so `Code/Models/momentum_change_GOOGL.joblib` loads and the numbers above
are real. `apple_models.MOMENTUM_CHANGE_TICKERS` deliberately still omits it.
The rule that motivated the withdrawal only runs one way -- a ticker in the
registry with no bundle is an agent that reports itself broken whenever
somebody picks it, whereas a bundle no entry points at costs nothing. Re-adding
it is one entry there, plus the three test modules that currently use
`momentum_change` on GOOGL as their "this pairing was never fitted" fixture.

**Every bundle is fitted on the whole weekly CSV archive**, not on the ~30-day
yfinance window the notebooks started from. AAPL gets 36 sessions (its archive
reaches a week further back), GOOGL and INTC 31 each; after the reserved week
and the 70/30 day split that is 22 training days for AAPL and 19 for the other
two. AAPL's Ridge in particular is a different model from the RandomForest the
demo fitted on 14 days of yfinance bars, and it is the archive, not a change of
method, that moved it.

Note the AAPL bundle here is **not** notebook 03's. That one is the project's
demo -- a month of yfinance bars, no reserved holdout week, metrics from the
same days its estimator was chosen on -- and it still sits in
`FinNotebooks/Models` because notebooks 04 and 05 were executed against it.
The bundle this module loads is trained by `scripts/train_ticker.py` on the
weekly CSV archive and under the held-out-week protocol, the same as the other
two, so all three carry the same metrics schema and mean the same thing.

What "the direction holds, the timing does not" means
-----------------------------------------------------
TimeToChange's own notebook 04 is blunt about the limit, and it is the thing to
keep in mind before reading a prediction as a trade: over *sampled stable
minutes* the absolute prediction separates "near a change" from "quiet" with
AUC ~0.68, but run bar by bar over every minute of an unseen day that falls to
~0.53. The sign is the reliable half. So the rules built on it gate on a regime
the tape has already printed and use the model for direction, never as a change
detector on its own.

Why this bundle is re-fitted rather than copied
------------------------------------------------
Every other model here is the notebook's own file, copied. This one cannot be:
the notebooks' venv pickles with scikit-learn 1.7.2 and this app runs 1.9.0, in
which a 1.7.2 `SimpleImputer` raises on `transform` and a 1.7.2
`HistGradientBoosting*` will not unpickle at all. So the bundle in
`Code/Models` is produced by running TimeToChange's own training script with
*this* project's interpreter:

    cd FinNotebooks/TimeToChange && \\
      ../../AgentStonks/.venv/bin/python scripts/train_ticker.py AAPL GOOGL INTC \\
      --model-dir ../../Models

Same data (the Data Collection weekly CSV archive), same code, same split, same
estimator chosen for each ticker; the metrics move in the third decimal. The
sidecar records which stack fitted it, and `load_bundle` refuses a bundle it
cannot load rather than degrading.

The mirror contract
-------------------
Everything under `--- momentum` and `--- features` is a verbatim copy of
`momlib/momentum.py` and `momlib/features.py`, and `score_bars` of the
inference half of `momlib/model.py`. It has to be: the saved estimator is a
function of those exact column definitions, and a mirror that drifts produces
confident numbers off a different feature. If `momlib` changes, retrain **and**
update this module -- the same contract `persistence_model` has with `mshift`,
`dayrange_model` with `dayrange` and `profile_model` with `levelsml`.
`tests/test_momentum_change_model.py` pins it against momlib itself.

Note the frame here keeps momlib's capitalised `Open/High/Low/Close/Volume` and
its `day` column rather than the lower-case shape the rest of the app uses.
That is deliberate: it keeps the copied block a copy, and `frame_from_bars` is
the one place the translation happens.

What the live path has to supply, and what it costs
---------------------------------------------------
Unlike the momentum-persistence model, **nothing here fits inside today's
tape**. Four features reach across the overnight gap and one of them decides
what a regime even is:

* `theta`, the regime threshold, is `0.4 * sigma_prev_day / sqrt(15)` where
  sigma is the standard deviation of the *previous session's 1-minute log
  returns*. Not of its daily bar -- of its minutes.
* `dist_prev_close`, `dist_sma5d`, `ret_prev_day` and `ret_5d` are built from
  the last close of each of the previous six sessions, again taken from the
  minute grid (the 15:59 bar) rather than from a daily bar, because that is
  what the training frame was.
* `dist_change_1..3` and `bars_since_change` count backwards through the
  persistent changes of the concatenated session grid, so the changes of
  previous days are in scope.

So `session_frame` fetches `HISTORY_SESSIONS` previous sessions of minute bars
through `historical.fetch_intraday_bars_for_date` (patched in SimLab to the
dataset's stored bars) and concatenates them behind today's live buffer. One
consequence worth stating plainly: **a one-day SimLab dataset cannot run this
model.** `require_history` refuses rather than predicting off a table of NaNs.

And one seam that cannot be closed, only named: **live event history is
fifteen bars stale, by construction.** A persistent change is only persistent
once the new regime has held `persist` minutes, so `find_persistent_changes`
cannot see a change younger than that -- an event at bar q first reaches the
features at bar q + `persist`. Training used no lag at all.

The notebooks' switch for this is `confirm_lag`, and **the live equivalent is
`persist - 1`, not `persist`** -- see `live_confirm_lag`, which is the number
to pass when comparing anything here against momlib. `confirm_lag=N` re-stamps
an event at the bar N later and `_event_history_features` then takes events
*strictly* before the current bar, so N models a lag of N+1. Off by one, and it
hides easily: on a tree bundle the wrong value still agrees with the live path
on most bars, because a tree quantises a small feature difference away -- it
was invisible on both of the tree-based tickers fitted at the time (GOOGL's
RandomForest and INTC's HistGradientBoosting). AAPL's Ridge does not quantise,
which is how it was caught, and is why the Ridge is the bundle to re-check this
against.
`tests/test_momentum_change_model.py` now pins both halves -- that `persist-1`
matches exactly, and that `persist` does not.

The lag itself moves `bars_since_change` and the `dist_change_*` distances in
the minutes right after a turn, which is exactly when this model is being
asked; against the no-lag training view the readings differ by up to ~0.29
bps/min on a holdout session. There is nothing to fix, because the honest
number is the lagged one. Worth knowing in the other direction too: any future
short cut that scored a cached whole-day frame instead of the bars in hand
would quietly become `confirm_lag=0`, and would look *better*.

The volume features are the one thing that does *not* need a consolidated tape:
`rel_volume_15` and `vol_accel` are both ratios of minute volume to minute
volume inside the same session, so a thinner feed cancels out of them. That is
the opposite of `dayrange_model.or_volume_share`, and the reason there is no
feed warning here.

Requires the optional `scikit-learn` / `joblib` dependencies (the estimator is
a pickled sklearn model); everything degrades to None without them or without
the bundle, exactly like the other saved models.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import clock, historical, market_hours, model_store
from .model_store import ModelStore

BPS = 1e4

REGIME_NAME = {-1: "negative", 0: "balanced", 1: "positive"}

# Default: the shared model store next to the AgentStonks checkout
# (Code/Models), where every other saved bundle lives.
MODEL_PATH_ENV = "APPLE_MOMENTUM_CHANGE_MODEL"
# The symbol the bare env override and the zero-argument calls mean, matching
# every other model module and `apple_models.DEFAULT_TICKER`.
DEFAULT_TICKER = model_store.DEFAULT_TICKER

# Regular-session window the model was trained on -- `momlib.data` drops
# everything outside it before computing anything, so a premarket bar reaching
# the frame would shift every `minute_of_day` and every per-day aggregate.
RTH_START = model_store.RTH_START
RTH_END = model_store.RTH_END

# `momlib.model.PIPELINE_PARAMS` -- the fallbacks if a bundle carries none.
PIPELINE_DEFAULTS = {
    "window": 15,
    "smooth_halflife": 8.0,
    "c": 0.4,
    "persist": 15,
    "stride": 10,
    "n_prev_changes": 3,
}

# Previous sessions of minute bars to put behind today. Six, because that is
# what the longest cross-day feature needs: `ret_5d` at today's bars is
# `log(close[d-1] / close[d-6])`. Everything else is shorter -- the 5-day SMA
# reaches back five sessions and `theta` exactly one.
HISTORY_SESSIONS = 6

# Calendar days to walk back looking for those sessions. Six trading days span
# eight calendar days in the ordinary case; the slack covers a long weekend
# plus a holiday, and bounds the walk so a symbol with no history at all fails
# in a second rather than a minute.
HISTORY_LOOKBACK_DAYS = 16

# A past session with fewer bars than this is skipped rather than used, exactly
# as `momlib.data.load_bars_csv` dropped it from the training frame: a half
# session has its own volatility and its own closing print, and letting one in
# would move `theta` and the daily closes away from anything the model saw.
# Today is exempt -- it is partial by definition.
MIN_BARS_PER_DAY = 300


# Today's history, keyed by (symbol, session date). Six dated fetches is six
# yfinance downloads live; the answer cannot change during the session, so it
# is assembled once per morning rather than once per bar.
_history_cache: "dict[tuple[str, object], list[dict]]" = {}


# --- momentum / regimes (mirrors momlib.momentum) ----------------------------

def add_momentum(
    df: pd.DataFrame, window: int = 15, smooth_halflife: float = 8.0
) -> pd.DataFrame:
    """Add log returns and trailing momentum (bps/min), computed per day.

    `mom_raw` is the plain trailing mean; `mom` is its trailing EMA
    (both use only past data, so they are point-in-time safe).
    """
    out = df.copy()
    grp = out.groupby("day", sort=False)
    out["log_ret"] = grp["Close"].transform(lambda s: np.log(s).diff()) * BPS
    out["mom_raw"] = grp["log_ret"].transform(
        lambda s: s.rolling(window, min_periods=window).mean()
    )
    out["mom"] = out.groupby("day", sort=False)["mom_raw"].transform(
        lambda s: s.ewm(halflife=smooth_halflife, min_periods=1).mean()
    )
    return out


def _daily_sigma(df: pd.DataFrame) -> pd.Series:
    """Std of 1-minute log returns (bps) for each day."""
    return df.groupby("day", sort=False)["log_ret"].std()


def add_regimes(df: pd.DataFrame, window: int = 15, c: float = 0.4) -> pd.DataFrame:
    """Classify each minute into a momentum regime with a per-day threshold.

    The threshold for day d uses the volatility of day d-1 so that it is known
    at the start of day d (the first day falls back to its own volatility).
    """
    out = df.copy()
    sigma = _daily_sigma(out)
    prev_sigma = sigma.shift(1)
    prev_sigma.iloc[0] = sigma.iloc[0]
    theta_by_day = c * prev_sigma / np.sqrt(window)
    out["theta"] = pd.Series(out["day"], index=out.index).map(theta_by_day).astype(float)
    regime = np.select(
        [out["mom"] > out["theta"], out["mom"] < -out["theta"]], [1, -1], default=0
    ).astype(float)
    regime[out["mom"].isna().to_numpy()] = np.nan
    out["regime"] = regime
    return out


def find_persistent_changes(df: pd.DataFrame, persist: int = 15) -> pd.DataFrame:
    """Find persistent regime changes.

    Returns one row per change with the change timestamp (first minute of the
    new regime), previous/new regime, momentum before/after, and the position
    of the bar within its day.

    Note what the `n - persist` bound means for a live frame: a change is only
    detectable once the new regime has held `persist` minutes, so the newest
    `persist` bars can never contain one. That is the source of the live event
    lag named in the module docstring, and it is the definition rather than a
    limitation of this implementation.
    """
    events = []
    for day, g in df.groupby("day", sort=True):
        reg = g["regime"].to_numpy()
        n = len(g)
        for t in range(persist, n - persist):
            r_new, r_prev = reg[t], reg[t - 1]
            if np.isnan(r_new) or np.isnan(r_prev) or r_new == r_prev:
                continue
            before = reg[t - persist : t]
            after = reg[t : t + persist]
            if np.isnan(before).any() or np.isnan(after).any():
                continue
            if (before == r_prev).all() and (after == r_new).all():
                ts = g.index[t]
                events.append(
                    {
                        "timestamp": ts,
                        "day": day,
                        "bar_of_day": t,
                        "price": g["Close"].iloc[t],
                        "prev_regime": int(r_prev),
                        "new_regime": int(r_new),
                        "mom_before": g["mom"].iloc[t - 1],
                        "mom_after": g["mom"].iloc[t + persist - 1],
                        "theta": g["theta"].iloc[t],
                    }
                )
    ev = pd.DataFrame(events)
    if len(ev):
        ev["transition"] = (
            ev["prev_regime"].map(REGIME_NAME) + " -> " + ev["new_regime"].map(REGIME_NAME)
        )
        ev["mom_delta"] = ev["mom_after"] - ev["mom_before"]
        ev = ev.set_index("timestamp").sort_index()
    return ev


# `momlib.momentum.find_stable_points` is deliberately absent: it samples the
# negative class for *training* and nothing at inference time has any use for
# it. Its absence is not drift -- there is no feature it feeds.


# --- features (mirrors momlib.features) --------------------------------------

def _per_bar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Trailing per-bar features on the full minute grid."""
    out = pd.DataFrame(index=df.index)
    g = df.groupby("day", sort=False)

    # Momentum at several horizons, z-scored by the day's regime threshold so
    # that +-1 corresponds to the regime boundary.
    for w in (5, 15, 60):
        m = g["log_ret"].transform(lambda s, w=w: s.rolling(w, min_periods=w).mean())
        out[f"mom_{w}"] = m
        out[f"mom_z_{w}"] = m / df["theta"]
    out["mom_slope"] = out["mom_15"] - out["mom_15"].groupby(df["day"].to_numpy()).shift(5)
    # The smoothed momentum that defines the regimes, z-scored by theta:
    # +-1 is exactly the regime boundary.
    out["mom_smooth_z"] = df["mom"] / df["theta"]

    # Volume behaviour.
    vol = df["Volume"].astype(float)
    v15 = vol.groupby(df["day"]).transform(lambda s: s.rolling(15, min_periods=15).mean())
    v60_prior = vol.groupby(df["day"]).transform(
        lambda s: s.rolling(60, min_periods=30).mean().shift(15)
    )
    out["rel_volume_15"] = v15 / v60_prior
    v5 = vol.groupby(df["day"]).transform(lambda s: s.rolling(5, min_periods=5).mean())
    out["vol_accel"] = v5 / v15

    # Volatility.
    s15 = g["log_ret"].transform(lambda s: s.rolling(15, min_periods=15).std())
    s60 = g["log_ret"].transform(lambda s: s.rolling(60, min_periods=30).std())
    out["volat_15"] = s15
    out["volat_ratio"] = s15 / s60

    # Position vs characteristic prices (log distance, bps).
    close = df["Close"]
    day_open = g["Open"].transform("first")
    out["dist_day_open"] = np.log(close / day_open) * BPS
    day_high = g["High"].transform(lambda s: s.cummax())
    day_low = g["Low"].transform(lambda s: s.cummin())
    rng = (day_high - day_low).replace(0.0, np.nan)
    out["range_pos_day"] = (close - day_low) / rng

    pv = (close * vol).groupby(df["day"]).transform("cumsum")
    cv = vol.groupby(df["day"]).transform("cumsum").replace(0.0, np.nan)
    out["dist_vwap"] = np.log(close / (pv / cv)) * BPS

    # Previous-day / multi-day context (known before the day starts).
    daily_close = df.groupby("day", sort=True)["Close"].last()
    prev_close = daily_close.shift(1)
    sma5 = daily_close.rolling(5, min_periods=3).mean().shift(1)
    daily_ret = np.log(daily_close / prev_close) * BPS  # return of each day, known at its close
    day_ser = pd.Series(df["day"], index=df.index)
    out["dist_prev_close"] = np.log(close / day_ser.map(prev_close).astype(float)) * BPS
    out["dist_sma5d"] = np.log(close / day_ser.map(sma5).astype(float)) * BPS
    out["ret_prev_day"] = day_ser.map(daily_ret.shift(1)).astype(float)
    ret_5d = (np.log(daily_close / daily_close.shift(5)) * BPS).shift(1)
    out["ret_5d"] = day_ser.map(ret_5d).astype(float)

    out["minute_of_day"] = g.cumcount()
    out["regime_now"] = df["regime"]
    out["theta"] = df["theta"]
    return out


def _event_history_features(
    points: pd.DataFrame, events: pd.DataFrame, n_prev: int = 3
) -> pd.DataFrame:
    """Distance to the prices of the last `n_prev` persistent changes strictly
    before each point, plus time since the last change (in trading minutes,
    approximated by bar count within the concatenated RTH grid)."""
    ev = events.sort_index()
    ev_ts = ev.index.to_numpy()
    ev_price = ev["price"].to_numpy(dtype=float)
    ev_pos = ev["global_pos"].to_numpy()

    rows = []
    for ts, row in points.iterrows():
        k = np.searchsorted(ev_ts, ts)  # events strictly before ts (left side)
        feat = {}
        for j in range(1, n_prev + 1):
            i = k - j
            if i >= 0:
                feat[f"dist_change_{j}"] = float(np.log(row["price"] / ev_price[i]) * BPS)
            else:
                feat[f"dist_change_{j}"] = np.nan
        feat["bars_since_change"] = (
            float(row["global_pos"] - ev_pos[k - 1]) if k >= 1 else np.nan
        )
        rows.append(feat)
    return pd.DataFrame(rows, index=points.index)


def build_live_features(
    df: pd.DataFrame,
    events: pd.DataFrame,
    n_prev_changes: int = 3,
    confirm_lag: int = 0,
) -> pd.DataFrame:
    """Features for *every* bar of `df`, for scoring new data with a saved model.

    Same definitions as the training-time `build_feature_table`, but evaluated
    on the full minute grid instead of on labelled points, and without any
    target column. Rows whose trailing windows are not yet filled contain NaNs
    (the model's imputer handles them; the first ~60 minutes of each day are
    best ignored).

    `events` supplies the change history used by the `dist_change_*` /
    `bars_since_change` features; pass an empty frame if no history is known.
    `confirm_lag` delays each event's availability by that many bars, which is
    what a live system would have to do (a change is only *known* to be
    persistent 15 bars after it happened). Training used `confirm_lag=0`, so
    leave it at 0 to reproduce training conditions exactly -- and see the
    module docstring for why a live frame is already lagged whatever this says.
    """
    bar_feats = _per_bar_features(df)
    pos = pd.Series(np.arange(len(df)), index=df.index, name="global_pos")
    points = pd.DataFrame({"price": df["Close"].astype(float)}).join(pos)

    if len(events):
        ev = events.join(pos)
        if confirm_lag:
            avail = np.minimum(ev["global_pos"].to_numpy() + confirm_lag, len(df) - 1)
            ev = ev.set_index(df.index[avail]).sort_index()
        hist = _event_history_features(points, ev, n_prev=n_prev_changes)
    else:
        cols = [f"dist_change_{j}" for j in range(1, n_prev_changes + 1)]
        hist = pd.DataFrame(np.nan, index=points.index, columns=cols + ["bars_since_change"])

    out = pd.concat([bar_feats.drop(columns=["regime_now"]), hist], axis=1)
    # regime of the previous minute, within the same day.
    out["regime_before"] = df.groupby("day", sort=False)["regime"].shift(1)
    return out[FEATURE_COLS]


FEATURE_COLS = [
    "mom_5", "mom_15", "mom_60",
    "mom_z_5", "mom_z_15", "mom_z_60", "mom_smooth_z",
    "mom_slope",
    "rel_volume_15", "vol_accel",
    "volat_15", "volat_ratio",
    "dist_day_open", "range_pos_day", "dist_vwap",
    "dist_prev_close", "dist_sma5d", "ret_prev_day", "ret_5d",
    "minute_of_day", "theta",
    "dist_change_1", "dist_change_2", "dist_change_3", "bars_since_change",
    "regime_before",
]

# The three features whose absence makes a prediction meaningless rather than
# merely imputed -- `momlib.model.score_bars`'s readiness gate. All three are
# NaN until the day's twentieth bar (`mom_15` needs fifteen, `mom_slope` five
# more), which is what makes the first twenty minutes of a session unscorable.
READY_COLS = ["mom_15", "mom_slope", "mom_smooth_z"]


# --- scoring (mirrors momlib.model) ------------------------------------------

def prepare(df: pd.DataFrame, params: "dict | None" = None):
    """Run the full detection pipeline: momentum -> regimes -> changes.

    Returns `(df_with_momentum, events)`. `momlib.prepare` returns a third
    element, the sampled stable points; nothing at inference time reads them.
    """
    p = {**PIPELINE_DEFAULTS, **(params or {})}
    df = add_momentum(df, window=p["window"], smooth_halflife=p["smooth_halflife"])
    df = add_regimes(df, window=p["window"], c=p["c"])
    events = find_persistent_changes(df, persist=p["persist"])
    return df, events


def score_bars(
    bundle: dict,
    df: pd.DataFrame,
    events: pd.DataFrame,
    confirm_lag: int = 0,
) -> pd.Series:
    """Predicted delta momentum for every bar of `df` (NaN where features are
    not ready).

    `df` must already carry momentum/regime columns (see `prepare`), and
    `events` the persistent changes known for those bars.
    """
    X = build_live_features(
        df,
        events,
        n_prev_changes=pipeline_params(bundle)["n_prev_changes"],
        confirm_lag=confirm_lag,
    )[bundle["feature_cols"]]
    # A prediction before the trailing windows are filled is meaningless: the
    # momentum features that carry the signal are still NaN there.
    ready = X[READY_COLS].notna().all(axis=1)
    pred = pd.Series(np.nan, index=df.index, name="pred_mom_delta")
    if ready.any():
        pred[ready] = bundle["estimator"].predict(X[ready])
    return pred


# --- the saved bundle --------------------------------------------------------

def _build_bundle(path: Path) -> "dict | None":
    """One ticker's saved model plus its metadata, or None when it cannot be
    loaded.

    A bundle that unpickles but is missing `estimator` or `feature_cols` is
    refused like a missing file: it is some other project's joblib sitting at
    this path, and reading a prediction out of it would be worse than having
    none. So is a bundle whose estimator was pickled by a scikit-learn this
    process cannot restore -- the failure mode this model exists to avoid (see
    the module docstring), and the reason the retrain command is in the error.
    """
    if not path.exists():
        return None
    try:
        import joblib
    except ImportError:
        return None
    try:
        bundle = joblib.load(path)
    except Exception:
        # Includes the cross-version unpickle failures: a scikit-learn that
        # moved a private module, or an estimator whose attributes this version
        # no longer understands. Either way there is no model here.
        return None
    if not isinstance(bundle, dict):
        return None
    if bundle.get("estimator") is None or not bundle.get("feature_cols"):
        return None
    return bundle


# One file per ticker, because TimeToChange fits and *selects* per ticker --
# AAPL's Ridge and INTC's HistGradientBoosting are not the same model with
# different weights.
#
# The file name is `momlib.model.model_path`'s, deliberately: the model this
# mirrors is the one the notebooks save, and the two stores agreeing means a
# retrain lands where the app already looks. `dayrange_model` has the same
# property for the same reason -- TimeToChange3 writes
# `timetochange3_dayrange_<TICKER>` itself.
_STORE = ModelStore(
    env_key=MODEL_PATH_ENV,
    filename="momentum_change_{ticker}.joblib",
    build=_build_bundle,
)

model_path = _STORE.path
# The readable JSON sidecar `momlib.model.save_model` writes beside the bundle.
metadata_path = _STORE.metadata_path
load_bundle = _STORE.load
reset_bundle_cache = _STORE.reset


def pipeline_params(bundle: "dict | None" = None) -> dict:
    """The momentum/regime parameters the bundle was fitted under.

    Read from the bundle rather than assumed, because the features and the
    regimes the model reasons about are only meaningful together with them --
    which is exactly why `momlib.model.save_model` stores them.
    """
    saved = (bundle or {}).get("pipeline_params") or {}
    return {**PIPELINE_DEFAULTS, **saved}


def model_name(bundle: "dict | None" = None) -> str:
    """Which estimator this ticker's selection picked, for a log line."""
    return str((bundle or {}).get("model_name") or "model")


def live_confirm_lag(bundle: "dict | None" = None) -> int:
    """The `confirm_lag` that reproduces this module's live view in momlib.

    `persist - 1`, and the minus one is the whole point of the function
    existing. A live frame ending at bar t contains an event at bar t - persist
    and not one bar younger; `confirm_lag=N` re-stamps an event at the bar N
    later and the history lookup is *strictly* before the current bar, so N
    models a lag of N + 1 bars.

    Anything that compares this module against momlib -- a notebook check, a
    trade-log comparison -- has to pass this rather than `persist`, or it is
    scoring the live path against a view one bar staler than the live path.
    """
    return int(pipeline_params(bundle)["persist"]) - 1


# --- the live minute grid ----------------------------------------------------

def frame_from_bars(bars: "list[dict]") -> pd.DataFrame:
    """Alpaca/yfinance `{"t","o","h","l","c","v"}` bars -> the momlib frame.

    The one place the app's bar shape becomes the notebooks': an
    exchange-local index, capitalised OHLCV, a `day` column of dates, and the
    regular session only. Everything above this line is a copy of momlib and
    reads exactly these names.
    """
    columns = ["Open", "High", "Low", "Close", "Volume", "day"]
    if not bars:
        return pd.DataFrame(columns=columns)
    idx = pd.to_datetime([b["t"] for b in bars], utc=True, format="mixed").tz_convert(
        market_hours.MARKET_TZ
    )
    df = pd.DataFrame(
        {
            "Open": [float(b["o"]) for b in bars],
            "High": [float(b["h"]) for b in bars],
            "Low": [float(b["l"]) for b in bars],
            "Close": [float(b["c"]) for b in bars],
            "Volume": [float(b.get("v") or 0.0) for b in bars],
        },
        index=idx,
    )
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.between_time(RTH_START, RTH_END)
    df = df[df["Close"] > 0]
    df["day"] = df.index.date
    return df


def history_bars(
    ticker: str,
    session_date,
    sessions: int = HISTORY_SESSIONS,
    lookback_days: int = HISTORY_LOOKBACK_DAYS,
) -> "list[dict]":
    """Minute bars of the `sessions` completed sessions before `session_date`.

    Walks backwards a calendar day at a time through
    `historical.fetch_intraday_bars_for_date` -- which SimLab patches to the
    dataset's stored bars, so a replay reads the same history the live agent
    would and no wall-clock day leaks in. A weekend or holiday returns nothing
    and simply costs one step of the walk.

    Short sessions are skipped rather than used: see `MIN_BARS_PER_DAY`.
    Cached for the session, since the answer cannot change during it.
    """
    symbol = str(ticker or DEFAULT_TICKER).upper()
    key = (symbol, pd.Timestamp(session_date).date())
    cached = _history_cache.get(key)
    if cached is not None:
        return cached

    collected: "list[list[dict]]" = []
    day = pd.Timestamp(session_date).date()
    for _ in range(lookback_days):
        day = day - pd.Timedelta(days=1)
        try:
            bars = historical.fetch_intraday_bars_for_date(symbol, day.isoformat())
        except Exception:
            bars = []
        # The fetch is not restricted to the regular session; count what the
        # frame will actually keep rather than what arrived.
        kept = frame_from_bars(bars)
        if len(kept) >= MIN_BARS_PER_DAY:
            collected.append(bars)
        if len(collected) >= sessions:
            break
    out = [bar for day_bars in reversed(collected) for bar in day_bars]
    # One session at a time, all symbols of it: yesterday's entry is dead
    # weight the moment the date rolls, and a process left running for a month
    # would otherwise hold a month of minute bars it can never use again. The
    # date is what expires, not the symbol -- evicting per key would make two
    # streamed symbols re-download each other's history every bar.
    for stale in [k for k in _history_cache if k[1] != key[1]]:
        _history_cache.pop(stale, None)
    _history_cache[key] = out
    return out


def reset_history_cache() -> None:
    """Forget the fetched history -- for tests, and for a long-lived process
    that has crossed midnight."""
    _history_cache.clear()


def session_frame(sym_state, ticker: str = DEFAULT_TICKER) -> pd.DataFrame:
    """The frame the pipeline runs on: six previous sessions, then today live.

    Unlike `persistence_model.minute_frame` this cannot be today alone -- the
    regime threshold itself is yesterday's volatility. The history is fetched
    once per session and the live buffer is re-read every call, so the cost per
    bar is the concatenation and not the download.
    """
    symbol = str(ticker or DEFAULT_TICKER).upper()
    today = market_date().date()
    with sym_state.lock:
        live_bars = list(sym_state.bars)
    live = frame_from_bars(live_bars)
    if len(live):
        live = live[live.index.date == today]
    history = frame_from_bars(history_bars(symbol, today))
    if len(history):
        history = history[history.index.date < today]
    if not len(history):
        return live
    if not len(live):
        return history
    return pd.concat([history, live])


def sessions_before(frame: pd.DataFrame, session_date) -> int:
    """How many distinct completed sessions sit behind `session_date`."""
    if not len(frame):
        return 0
    day = pd.Timestamp(session_date).date()
    return sum(1 for d in pd.unique(frame["day"]) if d < day)


def require_history(frame: pd.DataFrame, session_date) -> "str | None":
    """Why this frame cannot support a prediction, or None if it can.

    Checked before anything is computed, because the failure it prevents is the
    quiet one: a missing previous session leaves `theta` as the day's *own*
    volatility (`add_regimes` falls back for the first day of a frame), which
    silently redefines every regime and every z-scored feature; and a missing
    sixth leaves `ret_5d` NaN where the model was fitted on a number. Neither
    raises anything on its own.
    """
    have = sessions_before(frame, session_date)
    if have >= HISTORY_SESSIONS:
        return None
    return (
        f"only {have} of the {HISTORY_SESSIONS} previous sessions of minute bars are "
        "available; the regime threshold is yesterday's volatility and the multi-day "
        "features reach back six sessions, so they cannot be built. In SimLab, add "
        "those days to the dataset."
    )


def read_latest(bundle: dict, frame: pd.DataFrame) -> "dict | None":
    """Score the newest bar of `frame`, or None when the frame is empty.

    Returns `{"ts", "price", "pred", "mom", "theta", "regime", "regime_before",
    "bars_today", "warming_up"}`. `pred` is the model's delta-momentum forecast
    in bps/min and is None while the day's trailing windows are still filling
    (`warming_up`); `regime_before` is the previous minute's regime, which is
    what the rules gate on -- the bar's own regime already contains the move
    the model is being asked to predict.

    Only the last row is predicted. That is not an approximation: the estimator
    is a per-row function, so scoring one bar and scoring the whole frame give
    the same number for that bar -- unlike `nbeats_model`, whose bootstrap draws
    make the two differ.
    """
    if not len(frame):
        return None
    params = pipeline_params(bundle)
    scored, events = prepare(frame, params)
    features = build_live_features(
        scored, events, n_prev_changes=params["n_prev_changes"]
    )[bundle["feature_cols"]]

    row = features.iloc[[-1]]
    last = scored.iloc[-1]
    ts = scored.index[-1]
    ready = bool(row[READY_COLS].notna().all(axis=1).iloc[0])
    pred = float(bundle["estimator"].predict(row)[0]) if ready else None

    regime = last.get("regime")
    day_mask = scored["day"] == last["day"]
    prev = scored.loc[day_mask, "regime"]
    regime_before = prev.iloc[-2] if len(prev) >= 2 else np.nan
    return {
        "ts": ts,
        "price": float(last["Close"]),
        "pred": pred,
        "mom": None if pd.isna(last.get("mom")) else float(last["mom"]),
        "theta": None if pd.isna(last.get("theta")) else float(last["theta"]),
        "regime": None if pd.isna(regime) else int(regime),
        "regime_before": None if pd.isna(regime_before) else int(regime_before),
        "bars_today": int(day_mask.sum()),
        "warming_up": not ready,
    }


def regime_name(regime: "int | None") -> str:
    """"positive" / "balanced" / "negative", or "unknown" before there is one."""
    return REGIME_NAME.get(regime, "unknown")


def market_date() -> "pd.Timestamp":
    """Today, in the exchange's timezone -- the session being traded.

    Through `clock` rather than `datetime.now`, so a simulated day is the
    simulated day (see `agent_stonks.clock`).
    """
    return pd.Timestamp(clock.now().astimezone(market_hours.MARKET_TZ).date())
