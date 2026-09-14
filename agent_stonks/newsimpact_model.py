"""Where a news release leaves the next 15 minutes' momentum (NewsImpact).

The LLM scorer in `agent_stonks.news` reads an article and guesses which way
it pushes the stock. This model answers a narrower, measurable question about
the *price*: for an article released inside the regular session, what state
will the stock's 15-minute momentum be in over the 15 minutes after it --
positive, flat or negative?

    before = L[P0] - L[P0 - 15]      P0 = close of the last bar completed before the release
    z      = return / expected volatility over those minutes
    state  = pos if z > 1.0, neg if z < -1.0, else neu

`L` is the ticker's cumulative return **net of beta x SPY** (beta from the
previous 20 sessions), and the expected volatility is a time-of-day profile
from the previous 20 sessions, so a move at 09:50 and one at 13:00 are in the
same units. The window, threshold and basis are not assumed here: notebook 02
chose them on development data and the notebook writes them into the model's
JSON sidecar (`label`), which is where `_build_bundle` reads them from.

What the saved model reads, and what it does not
------------------------------------------------
`newsimpact_trajectory_<TICKER>.joblib` is the family the project's selection
rule chose, `prestate`: momentum state, z and slope t-stat over the 15 minutes
**before** the release, the number of the ticker's articles in the previous 30
minutes, and the hour. **It does not read the article's text.** The notebook's
finding is that almost all predictability of the change comes from the
momentum before the release; the best news-reading family
(`newsimpact_trajectory_news_<TICKER>`) was +0.004 log-loss skill better, inside
the tie tolerance. That bundle's pipeline pickles NewsImpact's own text
preprocessor and phrase selector, which the stub below cannot stand in for, so
`_build_bundle` refuses any bundle whose inputs are not in `SUPPORTED_INPUTS`
and the app falls back to the LLM for that symbol.

From seven class probabilities to a badge
-----------------------------------------
The classes are (state before -> state after). `post_state_probs` folds them
into P(after = pos / neu / neg), and the badge is whichever of the three is
most likely. Checked on the 9,162 labelled AAPL articles, with the saved model:

    badge      share   mean excess return over the next 15 min
    negative    2.6%   -14.4 bps
    neutral    96.9%    -0.4 bps
    positive    0.5%    +4.9 bps

So nearly everything is neutral, which is the model's honest answer (the state
after a release is neu 70% of the time), and the rare non-neutral badges do sort
the move that follows. The alternative the notebook uses for its confusion
matrix -- argmax of p / class prior -- spreads badges across all three (43%
positive) but its "positive" rows average -0.2 bps, and it would need the
training prior, which the bundle does not carry.

Only intraday releases can be scored
------------------------------------
A premarket, after-hours or weekend article has no "15 minutes before" inside a
session, and neither has one from the first 15 minutes after the open. Those
are reported as not scorable (the badge stays "unknown") rather than
approximated. More than half of AAPL's articles are out of session.

A release from *today* needs its 15 minutes of bars to have arrived; until then
it is `pending` and re-scored on the next refresh.

The mirror contract
-------------------
Everything under `--- trajectory` and `--- sessions` is a verbatim copy of
`newsimpact/trajectory.py` and `newsimpact/sessions.py`, and `proba_full` /
`post_state_probs` of `newsimpact/modeling.py`, with three adaptations marked
where they happen: the calendar never drops *today* for being short (the
notebook only ever saw completed sessions), today's close is the bell, not its
last bar so far, and two `to_numpy()` calls take `copy=True` because this venv's
pandas 3 hands back read-only views where the notebooks' pandas 2 copied. If
NewsImpact's windows, volatility profile or beta
change, retrain AND update this module -- the contract `dayrange_model` has
with `dayrange`.
`tests/test_newsimpact_model.py` pins it against the notebook's own events.

The pickle was written by scikit-learn 1.7.2 and this app runs 1.9.0. It
crosses the gap cleanly -- a
OneHotEncoder, a StandardScaler and a LogisticRegression -- and its
probabilities were checked identical (max abs difference 0.0) to the
notebooks' venv on sampled rows, so the version warnings are silenced at load.

What the live path has to supply
--------------------------------
Nothing in the features fits inside today's tape. The volatility profile is a
trailing mean over the previous 20 sessions of *market-adjusted* returns, and
each of those sessions is adjusted by its own beta -- itself a trailing mean
over the 20 sessions before it. So a release needs **40 completed sessions** of
minute bars behind it, of the ticker *and* of SPY, for its z to be the
notebook's; with 25 the notebook's own code moves z by up to ~10%
(`history_sessions`). Those come from Alpaca SIP, split-adjusted -- the tape the
model was fitted on -- with no fallback (see `fetch_history`). Today's bars come
through `bar_history.fetch_history_bars` on the session's resolved feed. Both
are cached: history per trading day, today's bars for `TODAY_TTL_SEC`.

Adding a ticker
---------------
Nothing to register. Drop `newsimpact_trajectory_<TICKER>.joblib` and its
`.json` into `Code/Models` (NewsImpact's `settings_for("GOOGL")` notebooks
write there) and `uses_model` answers True for it; `NEWS_IMPACT_MODEL_<TICKER>`
relocates one file.
"""
from __future__ import annotations

import json
import logging
import sys
import threading
import time as _time
import types
import warnings
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from . import bar_history, clock
from .datalog import log_fetch, log_fetch_failure
from .model_store import RTH_END, RTH_START, ModelStore
from .rest import fetch_bars_range

logger = logging.getLogger(__name__)

MARKET_TZ = "America/New_York"

MODEL_PATH_ENV = "NEWS_IMPACT_MODEL"

# The inputs this module can build. The prestate family reads exactly these;
# a bundle wanting anything else (article text, author, tagged symbols) is
# refused rather than fed a guess.
SUPPORTED_INPUTS = frozenset({"state_pre", "z_pre_c", "t_pre_c", "n_prior_news_30m", "hour"})

# TrajectoryParams fields the pre-release momentum depends on, with
# NewsImpact's defaults; the sidecar's `settings.trajectory` overrides them.
TRAJECTORY_DEFAULTS = {
    "vol_lookback_sessions": 20,
    "vol_smooth_minutes": 15,
    "beta_lookback_sessions": 20,
}

# An article older than this is not worth a longer history download: the
# initial news load is the 15 latest articles, which is days, not weeks.
MAX_ARTICLE_AGE_DAYS = 10

# Today's bars are re-fetched at most this often; a pending article waits at
# most this long past the minute its 15-minute window closes.
TODAY_TTL_SEC = 60

# After a refresh fails (no bars from any source, say), the next attempt waits
# this long instead of retrying on every panel poll.
RETRY_AFTER_FAILURE_SEC = 120

# Impact methods the News tab offers. "auto" is the default: a symbol with its
# own model uses it, every other symbol the LLM.
IMPACT_METHOD_AUTO = "auto"
IMPACT_METHOD_LLM = "llm"
IMPACT_METHODS: dict[str, str] = {
    IMPACT_METHOD_AUTO: "News-impact model where fitted, LLM otherwise",
    IMPACT_METHOD_LLM: "LLM for every symbol",
}

STATUS_SCORED = "scored"
STATUS_PENDING = "pending"
STATUS_NOT_SCORABLE = "not_scorable"
SOURCE_MODEL = "model"

_STATE_WORDS = {"pos": "positive", "neu": "flat", "neg": "negative"}


# --- config (newsimpact/config.py) --------------------------------------------

MAX_BARS = 390

STATES = ("neg", "neu", "pos")
NO_CHANGE = "no_change"
# The seven classes: a change from one momentum state to another, or no change.
CLASSES = tuple(f"{a}->{b}" for a in STATES for b in STATES if a != b) + (NO_CHANGE,)

EPS = 1e-6


@dataclass(frozen=True)
class _Trajectory:
    vol_lookback_sessions: int
    vol_smooth_minutes: int
    beta_lookback_sessions: int


@dataclass(frozen=True)
class _Settings:
    """The slice of `newsimpact.config.Settings` the copied functions read, so
    their bodies stay copies (`st.trajectory.vol_lookback_sessions`)."""

    trajectory: _Trajectory


# --- sessions (newsimpact/sessions.py) ----------------------------------------

def early_close(dates: pd.DatetimeIndex) -> np.ndarray:
    """NYSE 13:00 closes: the day after Thanksgiving, and July 3 / December 24
    when they fall Monday-Thursday. (When July 3 or December 24 is a Friday the
    exchange is closed for the observed holiday, so there are no bars to label.)"""
    d = pd.DatetimeIndex(dates)
    after_thanksgiving = (d.month == 11) & (d.weekday == 4) & (d.day >= 23) & (d.day <= 29)
    eve = ((d.month == 7) & (d.day == 3)) | ((d.month == 12) & (d.day == 24))
    return np.asarray(after_thanksgiving | (eve & (d.weekday <= 3)))


def session_calendar(
    bars: pd.DataFrame, min_bars_per_session: int, today: "pd.Timestamp | None" = None
) -> pd.DataFrame:
    """One row per session: open, close, bar count, span in minutes.

    `sessions.session_calendar`, less the afternoon-volume column it keeps only
    for inspection, and with the two live adaptations: `today` is kept however
    few bars it has so far, and its close is the bell (13:00 on a half day)
    rather than its latest bar -- the notebook's "last bar + 1 minute" only
    means the close once the session is over.
    """
    columns = ["open_ts", "half_day", "close_ts", "n_bars", "span_bars"]
    if not len(bars):
        return pd.DataFrame(columns=columns, index=pd.DatetimeIndex([], name="date"))
    dates = pd.DatetimeIndex(sorted(bars["date"].unique()), name="date")

    cal = pd.DataFrame(index=dates)
    cal["open_ts"] = dates.tz_localize(MARKET_TZ) + pd.Timedelta(hours=9, minutes=30)
    last_bar = pd.Series(bars.index, index=bars.index).groupby(bars["date"]).max().reindex(dates)
    cal["half_day"] = early_close(dates)
    half_close = cal["open_ts"] + pd.Timedelta(hours=3, minutes=30)
    close = (last_bar + pd.Timedelta(minutes=1)).where(~cal["half_day"], half_close)
    if today is not None and today in dates:
        bell = cal.loc[today, "open_ts"] + pd.Timedelta(hours=6, minutes=30)
        close.loc[today] = half_close.loc[today] if cal.loc[today, "half_day"] else bell
    cal["close_ts"] = close
    in_session = bars.index < bars["date"].map(cal["close_ts"]).to_numpy()
    cal["n_bars"] = bars[in_session].groupby("date").size().reindex(dates).fillna(0).astype(int)
    cal["span_bars"] = ((cal["close_ts"] - cal["open_ts"]) / pd.Timedelta(minutes=1)).round().astype(int)
    keep = cal["n_bars"] >= min_bars_per_session
    if today is not None:
        keep |= cal.index == today
    return cal[keep]


# --- trajectory (newsimpact/trajectory.py) ------------------------------------

@dataclass
class Grid:
    dates: pd.DatetimeIndex        # sessions, ascending
    n_bars: np.ndarray             # (S,) bars spanned, last index + 1
    L: dict[str, np.ndarray]       # basis -> (S, 390) log path
    V: dict[str, np.ndarray]       # basis -> (S, 390) cumulative expected variance
    beta: np.ndarray               # (S,) beta to the market used for "excess"

    def session_index(self, dates: pd.Series) -> np.ndarray:
        idx = self.dates.get_indexer(pd.DatetimeIndex(dates))
        return idx


def _close_matrix(bars: pd.DataFrame, dates: pd.DatetimeIndex,
                  span: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(S, 390) closes, forward-filled inside each session; NaN after its last bar.
    `span` (minutes from the open to the close, per session) cuts half days at 13:00."""
    b = bars[bars["date"].isin(dates)]
    s = dates.get_indexer(b["date"])
    minute = ((b.index.hour * 60 + b.index.minute) - 570).to_numpy()
    keep = (minute >= 0) & (minute < MAX_BARS) & (s >= 0)
    if span is not None:
        keep &= minute < np.asarray(span)[np.clip(s, 0, None)]
    C = np.full((len(dates), MAX_BARS), np.nan)
    C[s[keep], minute[keep]] = b["close"].to_numpy()[keep]
    n_bars = np.zeros(len(dates), dtype=int)
    np.maximum.at(n_bars, s[keep], minute[keep] + 1)
    # copy=True: the one departure from the copy. pandas 3 (this venv) returns
    # a read-only view under copy-on-write; pandas 2 (the notebooks') a copy.
    C = pd.DataFrame(C).ffill(axis=1).to_numpy(copy=True)
    C[np.arange(MAX_BARS)[None, :] >= n_bars[:, None]] = np.nan
    return C, n_bars


def _trailing_mean(X: np.ndarray, lookback: int) -> np.ndarray:
    """Row s = nanmean of rows s-lookback .. s-1 (strictly earlier sessions)."""
    valid = np.isfinite(X)
    cs = np.vstack([np.zeros((1, X.shape[1])), np.cumsum(np.where(valid, X, 0.0), axis=0)])
    cn = np.vstack([np.zeros((1, X.shape[1])), np.cumsum(valid, axis=0)])
    S = X.shape[0]
    hi = np.arange(S)
    lo = np.maximum(hi - lookback, 0)
    tot = cs[hi] - cs[lo]
    cnt = cn[hi] - cn[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = tot / cnt
    out[cnt < max(3, lookback // 2)] = np.nan
    return out


def _profile_variance(R: np.ndarray, st: _Settings) -> np.ndarray:
    """(S, 390) cumulative sum over minutes of the expected squared 1-min return."""
    tp = st.trajectory
    ms = _trailing_mean(R ** 2, tp.vol_lookback_sessions)
    # copy=True for pandas 3's read-only view, as in `_close_matrix`.
    ms = pd.DataFrame(ms).T.rolling(tp.vol_smooth_minutes, center=True, min_periods=1).mean().T.to_numpy(copy=True)
    ms[:, 0] = 0.0                       # no return "into" the first bar
    ms = np.where(np.isfinite(ms), ms, np.nan)
    return np.nancumsum(ms, axis=1) + np.where(np.isnan(ms).all(axis=1, keepdims=True), np.nan, 0.0)


def build_grid(bars: pd.DataFrame, market: pd.DataFrame, cal: pd.DataFrame, st: _Settings) -> Grid:
    dates = pd.DatetimeIndex(cal.index)
    span = cal["span_bars"].to_numpy()
    C, n_bars = _close_matrix(bars, dates, span)
    Cm, _ = _close_matrix(market, dates, span)

    La = np.log(C)
    Ra = np.diff(La, axis=1, prepend=np.nan)
    Rm = np.diff(np.log(Cm), axis=1, prepend=np.nan)

    # beta from the previous sessions' pooled 1-minute returns
    both = np.isfinite(Ra) & np.isfinite(Rm)
    xy = np.where(both, Ra * Rm, np.nan)
    xx = np.where(both, Rm * Rm, np.nan)
    num = _trailing_mean(np.nanmean(xy, axis=1, keepdims=True), st.trajectory.beta_lookback_sessions)[:, 0]
    den = _trailing_mean(np.nanmean(xx, axis=1, keepdims=True), st.trajectory.beta_lookback_sessions)[:, 0]
    beta = num / den

    Re = Ra - beta[:, None] * Rm
    Le = np.where(np.isfinite(La), np.nancumsum(np.where(np.isfinite(Re), Re, 0.0), axis=1), np.nan)
    Le[~np.isfinite(beta)] = np.nan

    L = {"raw": La, "excess": Le}
    V = {"raw": _profile_variance(Ra, st), "excess": _profile_variance(Re, st)}
    return Grid(dates=dates, n_bars=n_bars, L=L, V=V, beta=beta)


def release_minute(ts: pd.Series) -> np.ndarray:
    """Index of the bar containing each timestamp (0 = the 09:30 bar)."""
    return ((ts.dt.hour * 60 + ts.dt.minute) - 570).to_numpy()


def slope_tstat(grid: Grid, s: np.ndarray, k: np.ndarray, pre: int, post: int, basis: str) -> pd.DataFrame:
    """OLS slope t-statistic of the log path over each window - a second momentum
    measure that reads the whole path, not only its end points."""
    L = grid.L[basis]
    out = {}
    for name, lo_off, hi_off in (("t_pre", -1 - pre, -1), ("t_post", -1, -1 + post)):
        n = hi_off - lo_off + 1
        idx = (np.asarray(k)[:, None] + np.arange(lo_off, hi_off + 1)[None, :])
        valid = (idx >= 0).all(axis=1) & (idx < MAX_BARS).all(axis=1) & (np.asarray(s) >= 0)
        idx = np.clip(idx, 0, MAX_BARS - 1)
        Y = L[np.where(valid, s, 0)[:, None], idx]
        x = np.arange(n) - (n - 1) / 2
        ym = Y.mean(axis=1, keepdims=True)
        b = ((Y - ym) * x).sum(axis=1) / (x ** 2).sum()
        resid = Y - ym - b[:, None] * x
        se = np.sqrt((resid ** 2).sum(axis=1) / (n - 2) / (x ** 2).sum())
        with np.errstate(invalid="ignore", divide="ignore"):
            t = b / se
        out[name] = np.where(valid & np.isfinite(t), t, np.nan)
    return pd.DataFrame(out)


def states(z: np.ndarray | pd.Series, threshold: float) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    out = np.where(z > threshold, "pos", np.where(z < -threshold, "neg", "neu")).astype(object)
    out[~np.isfinite(z)] = None
    return out


def pre_momentum(grid: Grid, s: np.ndarray, k: np.ndarray, pre: int, threshold: float,
                 basis: str) -> pd.DataFrame:
    """Momentum before the release only - what a live model can know at `t`.

    Needs bars up to k-1 and nothing after, so it is defined for items near the
    close too (where `window_stats` marks the event unlabelled).
    """
    L, V = grid.L[basis], grid.V[basis]
    s, k = np.asarray(s), np.asarray(k)
    a, i0 = k - 1 - pre, k - 1
    ok = (s >= 0) & (a >= 0) & (i0 < grid.n_bars[np.clip(s, 0, None)])
    ss = np.where(ok, s, 0)
    ia, ib = np.clip(a, 0, MAX_BARS - 1), np.clip(i0, 0, MAX_BARS - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (L[ss, ib] - L[ss, ia]) / np.sqrt(V[ss, ib] - V[ss, ia])
    z = np.where(ok & np.isfinite(z), z, np.nan)
    t = slope_tstat(grid, np.where(ok, s, -1), k, pre, 0, basis)["t_pre"].to_numpy()
    return pd.DataFrame({"z_pre": z, "t_pre": t, "state_pre": states(z, threshold)})


# --- modeling (newsimpact/modeling.py) ----------------------------------------

def proba_full(pipe, X: pd.DataFrame) -> np.ndarray:
    """Probabilities over all CLASSES in fixed order, floored and renormalised."""
    p = pipe.predict_proba(X)
    out = np.zeros((len(X), len(CLASSES)))
    for j, c in enumerate(pipe.classes_):
        out[:, CLASSES.index(c)] = p[:, j]
    out = np.clip(out, EPS, None)
    return out / out.sum(axis=1, keepdims=True)


def post_state_probs(P: pd.DataFrame, state_pre: pd.Series) -> pd.DataFrame:
    """P(momentum after is pos / neg) implied by the class probabilities."""
    sp_ = state_pre.reindex(P.index)
    nc = P["no_change"]
    p_pos = P["neg->pos"] + P["neu->pos"] + nc * (sp_ == "pos")
    p_neg = P["neu->neg"] + P["pos->neg"] + nc * (sp_ == "neg")
    return pd.DataFrame({"p_post_pos": p_pos, "p_post_neg": p_neg, "direction_score": p_pos - p_neg})


@dataclass
class Predictor:
    """Stand-in for `newsimpact.modeling.Predictor`, the class the bundle was
    pickled as. Only its fields matter for unpickling; the real class's
    `predict_proba` is `proba_full` above."""

    pipeline: object
    family: str
    classes: tuple
    inputs: list


def _register_unpickle_alias() -> None:
    """Make `newsimpact.modeling.Predictor` resolvable for `joblib.load`.

    Importing the real package would need NewsImpact installed in this venv
    (plus its plotting stack), for a dataclass of four fields, so a stub module
    pointing at the class above stands in -- unless the real package is
    genuinely importable, in which case it wins.
    """
    if "newsimpact.modeling" in sys.modules:
        return
    try:
        __import__("newsimpact.modeling")
        return
    except Exception:
        for name in [m for m in sys.modules if m == "newsimpact" or m.startswith("newsimpact.")]:
            sys.modules.pop(name, None)
    package = types.ModuleType("newsimpact")
    package.__path__ = []  # a package, so "newsimpact.modeling" is a legal submodule
    module = types.ModuleType("newsimpact.modeling")
    module.Predictor = Predictor
    package.modeling = module
    sys.modules["newsimpact"] = package
    sys.modules["newsimpact.modeling"] = module


# --- the saved bundle --------------------------------------------------------

def _build_bundle(path: Path) -> "dict | None":
    """One ticker's classifier plus the label definition it was trained on, or
    None when it cannot be used.

    The sidecar is required, not decorative: the window, threshold and basis
    live only there (`label`), and a model scored with a different window is a
    different model. A bundle whose inputs this module cannot build, whose
    classes are not the seven, or that does not unpickle, is refused like a
    missing file.
    """
    if not path.exists():
        return None
    try:
        meta = json.loads(path.with_suffix(".json").read_text())
    except (OSError, ValueError):
        return None
    label = meta.get("label") or {}
    if {"pre", "post", "threshold", "basis"} - set(label):
        return None
    _register_unpickle_alias()
    try:
        import joblib

        with warnings.catch_warnings():
            try:
                from sklearn.exceptions import InconsistentVersionWarning

                warnings.simplefilter("ignore", InconsistentVersionWarning)
            except ImportError:
                pass
            predictor = joblib.load(path)
    except Exception:
        return None
    pipeline = getattr(predictor, "pipeline", None)
    inputs = list(getattr(predictor, "inputs", None) or [])
    classes = tuple(getattr(predictor, "classes", None) or ())
    if pipeline is None or not inputs or set(inputs) - SUPPORTED_INPUTS:
        return None
    if set(classes) != set(CLASSES):
        return None
    settings = meta.get("settings") or {}
    saved_trajectory = settings.get("trajectory") or {}
    trajectory = {k: int(saved_trajectory.get(k, v)) for k, v in TRAJECTORY_DEFAULTS.items()}
    data = settings.get("data") or {}
    return {
        "pipeline": pipeline,
        "family": str(getattr(predictor, "family", meta.get("family", ""))),
        "inputs": inputs,
        "pre": int(label["pre"]),
        "post": int(label["post"]),
        "threshold": float(label["threshold"]),
        "basis": str(label["basis"]),
        "trajectory": trajectory,
        "market_symbol": str(data.get("market_symbol") or "SPY"),
        "min_bars_per_session": int(data.get("min_bars_per_session", 180)),
        "trained_on": meta.get("trained_on") or {},
    }


# The file name is NewsImpact's `Settings.model_name`, so a notebook run for a
# new ticker lands where the app already looks.
_STORE = ModelStore(
    env_key=MODEL_PATH_ENV,
    filename="newsimpact_trajectory_{ticker}.joblib",
    build=_build_bundle,
)

model_path = _STORE.path
load_bundle = _STORE.load
reset_bundle_cache = _STORE.reset


def has_model(ticker: "str | None") -> bool:
    """Whether `ticker` has its own usable news-impact model."""
    return bool(ticker) and load_bundle(ticker) is not None


def uses_model(ticker: "str | None", method: "str | None") -> bool:
    """Whether `ticker`'s news is scored by the model under the chosen method."""
    return method != IMPACT_METHOD_LLM and has_model(ticker)


def history_sessions(bundle: dict) -> int:
    """Completed sessions a release needs behind it for its z to be the notebook's.

    On the market-adjusted basis this is the *sum* of both windows, not the
    longer one: the volatility profile averages the previous
    `vol_lookback_sessions` of excess returns, and each of those sessions is
    net of its own beta, a trailing mean over the `beta_lookback_sessions`
    before it. With only 25 sessions behind a release, the notebook's own code
    moves z by up to ~10% (0.998 -> 1.017 on 2026-03-04, across the 1.0
    threshold) while beta for the day itself still agrees to 1e-14.
    """
    tp = bundle["trajectory"]
    if bundle["basis"] == "excess":
        return tp["vol_lookback_sessions"] + tp["beta_lookback_sessions"]
    return tp["vol_lookback_sessions"]


def _settings(bundle: dict) -> _Settings:
    return _Settings(trajectory=_Trajectory(**bundle["trajectory"]))


# --- scoring -----------------------------------------------------------------

def bars_frame(bars: "list[dict]") -> pd.DataFrame:
    """Alpaca/yfinance `{"t","o","h","l","c","v"}` bars -> NewsImpact's bar frame.

    Regular-hours bars stamped in market time, positive prices only, and a naive
    `date` column -- the shape `data.load_bars` hands the notebooks.
    """
    empty = pd.DataFrame(
        {"close": pd.Series(dtype=float), "date": pd.Series(dtype="datetime64[ns]")},
        index=pd.DatetimeIndex([], tz=MARKET_TZ, name="ts"),
    )
    if not bars:
        return empty
    idx = pd.to_datetime([b["t"] for b in bars], utc=True, format="mixed").tz_convert(MARKET_TZ)
    df = pd.DataFrame(
        {
            "open": [float(b["o"]) for b in bars],
            "high": [float(b["h"]) for b in bars],
            "low": [float(b["l"]) for b in bars],
            "close": [float(b["c"]) for b in bars],
        },
        index=idx.rename("ts"),
    )
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.between_time(RTH_START, RTH_END)
    df = df[df[["open", "high", "low", "close"]].gt(0).all(axis=1)]
    if not len(df):
        return empty
    out = df[["close"]].copy()
    out["date"] = out.index.tz_localize(None).normalize()
    return out


def release_ts(item: dict) -> "pd.Timestamp | None":
    """An article's `created_at` in market time; naive stamps are read as UTC."""
    try:
        ts = pd.Timestamp(item.get("created_at"))
    except (TypeError, ValueError):
        return None
    if ts is pd.NaT:
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert(MARKET_TZ)


def impact_label(p_up: float, p_flat: float, p_down: float) -> str:
    """The most likely momentum state after the release, as a badge label.
    A tie goes to neutral."""
    return max(
        ((p_flat, "neutral"), (p_up, "positive"), (p_down, "negative")),
        key=lambda pair: pair[0],
    )[1]


def _unscored(status: str, short: str, reason: str) -> dict:
    return {
        "label": "unknown",
        "source": SOURCE_MODEL,
        "status": status,
        "short": short,
        "reason": reason,
    }


def _placement(ts, cal: pd.DataFrame, grid: "Grid | None", today: pd.Timestamp,
               now: pd.Timestamp, bundle: dict) -> "dict | tuple[int, int]":
    """(session index, release minute) for a scorable article, else its verdict."""
    pre = bundle["pre"]
    only_intraday = "The model only reads releases inside the regular session."
    if ts is None:
        return _unscored(STATUS_NOT_SCORABLE, "no timestamp", "The article has no usable timestamp.")
    if ts > now:
        return _unscored(STATUS_PENDING, "pending", "Stamped later than the current time.")
    day = ts.tz_localize(None).normalize()
    in_hours = ts.weekday() < 5 and RTH_START <= ts.strftime("%H:%M") <= RTH_END
    if day not in cal.index:
        if day == today and in_hours:
            return _unscored(STATUS_PENDING, "pending", "Today's minute bars have not arrived yet.")
        if len(cal) and day < cal.index[0]:
            return _unscored(STATUS_NOT_SCORABLE, "too old", "Older than the minute history fetched for scoring.")
        return _unscored(STATUS_NOT_SCORABLE, "outside session", f"Not released on a trading session. {only_intraday}")
    row = cal.loc[day]
    if ts < row["open_ts"]:
        return _unscored(STATUS_NOT_SCORABLE, "premarket", f"Released before the open. {only_intraday}")
    if ts >= row["close_ts"]:
        return _unscored(STATUS_NOT_SCORABLE, "after hours", f"Released after the close. {only_intraday}")
    k = int(release_minute(pd.Series([ts]))[0])
    if k - 1 - pre < 0:
        return _unscored(
            STATUS_NOT_SCORABLE,
            f"first {pre} min",
            f"Released in the first {pre} minutes; the model needs the {pre} minutes of bars before it.",
        )
    s = int(cal.index.get_loc(day))
    lookback = history_sessions(bundle)
    if s < lookback:
        return _unscored(
            STATUS_NOT_SCORABLE,
            "no history",
            f"Only {s} of the {lookback} earlier sessions of minute bars are available; "
            "the volatility profile and beta are trailing means over them.",
        )
    if grid is None or grid.n_bars[s] < k:
        if day == today:
            return _unscored(
                STATUS_PENDING, "pending", f"Waiting for the {pre} minutes of bars before the release."
            )
        return _unscored(STATUS_NOT_SCORABLE, "no bars", "The bars before the release are missing.")
    return s, k


def score_articles(
    bundle: dict,
    articles: "list[dict]",
    bars: pd.DataFrame,
    market: pd.DataFrame,
    now,
) -> "dict[str, dict]":
    """{news_id: result} for every article with an id.

    `bars` and `market` are `bars_frame`s of the ticker and the market ETF
    covering the articles' sessions and `history_sessions(bundle)` completed
    sessions before the oldest; `now` bounds what is "today". A result carries
    the badge `label`, `status` (scored / pending / not_scorable), a `short`
    and a `reason` for the tooltip, and for a scored article the three
    post-release probabilities and the momentum before it.

    The article list is also the news count's population
    (`n_prior_news_30m`), as the notebook's was the ticker's whole feed.
    """
    now = pd.Timestamp(now).tz_convert(MARKET_TZ)
    today = now.tz_localize(None).normalize()
    cal = session_calendar(bars, bundle["min_bars_per_session"], today)
    grid = None
    if len(cal):
        with warnings.catch_warnings():
            # nanmean over a session with no aligned minutes (today, early on).
            warnings.simplefilter("ignore", RuntimeWarning)
            grid = build_grid(bars, market, cal, _settings(bundle))

    stamped = [
        (str(item.get("id") or ""), release_ts(item)) for item in articles
    ]
    stamped = [(news_id, ts) for news_id, ts in stamped if news_id]
    known = np.sort(np.array([ts.value for _, ts in stamped if ts is not None], dtype="int64"))

    out: "dict[str, dict]" = {}
    rows: "list[tuple[str, pd.Timestamp, int, int]]" = []
    for news_id, ts in stamped:
        placed = _placement(ts, cal, grid, today, now, bundle)
        if isinstance(placed, dict):
            out[news_id] = placed
        else:
            rows.append((news_id, ts, *placed))
    if not rows:
        return out

    s = np.array([r[2] for r in rows])
    k = np.array([r[3] for r in rows])
    pm = pre_momentum(grid, s, k, bundle["pre"], bundle["threshold"], bundle["basis"])
    t_ns = np.array([r[1].value for r in rows], dtype="int64")
    frame = pd.DataFrame(
        {
            "state_pre": pm["state_pre"].to_numpy(),
            "z_pre_c": pm["z_pre"].clip(-5, 5).to_numpy(),
            "t_pre_c": pm["t_pre"].clip(-10, 10).fillna(0.0).to_numpy(),
            "n_prior_news_30m": (
                np.searchsorted(known, t_ns, side="left")
                - np.searchsorted(known, t_ns - 30 * 60 * 10**9, side="left")
            ).astype(float),
            "hour": [str(r[1].hour) for r in rows],
        }
    )
    ok = np.isfinite(pm["z_pre"].to_numpy())
    for i in np.flatnonzero(~ok):
        out[rows[i][0]] = _unscored(
            STATUS_NOT_SCORABLE,
            "no profile",
            "No volatility profile or beta for that session (missing bars of the ticker or "
            f"{bundle['market_symbol']}).",
        )
    if not ok.any():
        return out

    X = frame[ok].reset_index(drop=True)
    P = pd.DataFrame(proba_full(bundle["pipeline"], X[bundle["inputs"]]), columns=list(CLASSES))
    post = post_state_probs(P, X["state_pre"])
    basis = "market-adjusted" if bundle["basis"] == "excess" else "raw"
    for j, i in enumerate(np.flatnonzero(ok)):
        p_up = float(post["p_post_pos"].iloc[j])
        p_down = float(post["p_post_neg"].iloc[j])
        p_flat = max(0.0, 1.0 - p_up - p_down)
        state_pre = str(X["state_pre"].iloc[j])
        z_pre = float(pm["z_pre"].iloc[i])
        out[rows[i][0]] = {
            "label": impact_label(p_up, p_flat, p_down),
            "source": SOURCE_MODEL,
            "status": STATUS_SCORED,
            "short": "model",
            "reason": (
                f"{bundle['post']}-min momentum after the release: up {p_up:.0%} · "
                f"flat {p_flat:.0%} · down {p_down:.0%}. Before it: "
                f"{_STATE_WORDS.get(state_pre, state_pre)} (z {z_pre:+.2f}, {basis})."
            ),
            "p_up": p_up,
            "p_flat": p_flat,
            "p_down": p_down,
            "state_pre": state_pre,
            "z_pre": z_pre,
        }
    return out


# --- bars for the live path --------------------------------------------------

_history_cache: "dict[tuple[str, date, date], pd.DataFrame]" = {}
_today_cache: "dict[str, tuple[float, date, pd.DataFrame]]" = {}
_cache_lock = threading.Lock()


def _calendar_days(sessions: int) -> int:
    """Calendar days certain to hold `sessions` trading sessions, holidays included."""
    return int(sessions * 7 / 5) + 10


def fetch_history(symbol: str, first_day: date, today: date, key: str, secret: str) -> pd.DataFrame:
    """Completed sessions' minute bars from `first_day` up to (not including) `today`.

    Alpaca SIP, split-adjusted: the tape NewsImpact fitted on, and deep enough.
    There is deliberately no yfinance fallback -- it serves about 30 days of
    minute bars, which cannot reach the 40 sessions `history_sessions` needs,
    so it could only ever produce "no history" verdicts or a drifted z. A
    failure raises instead, and the News tab shows it. Cached per trading day.
    """
    cache_key = (symbol, first_day, today)
    with _cache_lock:
        cached = _history_cache.get(cache_key)
    if cached is not None:
        return cached

    tz = pd.Timestamp(first_day).tz_localize(MARKET_TZ).tzinfo
    start = datetime.combine(first_day, datetime.min.time(), tzinfo=tz)
    end = datetime.combine(today, datetime.min.time(), tzinfo=tz)
    source = "Alpaca REST (SIP, split-adjusted)"
    try:
        bars = fetch_bars_range(symbol, "1Min", start, end, key, secret, feed="sip", adjustment="split")
    except Exception as exc:
        log_fetch_failure(
            "minute bars (news-impact history)", [(source, exc)], symbol=symbol,
            consequence="news-impact model cannot score this symbol",
        )
        raise RuntimeError(f"no SIP minute history for {symbol}: {exc}") from exc
    frame = bars_frame(bars)
    frame = frame[frame["date"] < pd.Timestamp(today)]
    log_fetch(
        "minute bars (news-impact history)", source, symbol=symbol,
        detail=f"{len(frame)} bars from {first_day}",
    )
    with _cache_lock:
        for stale in [k for k in _history_cache if k[0] == symbol and k[2] != today]:
            _history_cache.pop(stale, None)
        _history_cache[cache_key] = frame
    return frame


def fetch_today(symbol: str, key: str, secret: str, feed: str, now: pd.Timestamp) -> pd.DataFrame:
    """Today's regular-session minute bars on the session's resolved feed."""
    today = now.date()
    with _cache_lock:
        cached = _today_cache.get(symbol)
    if cached is not None and cached[1] == today and _time.monotonic() - cached[0] < TODAY_TTL_SEC:
        return cached[2]
    open_ts = pd.Timestamp(today).tz_localize(MARKET_TZ) + pd.Timedelta(hours=9, minutes=30)
    frame = bars_frame([])
    if now > open_ts:
        hours = int((now - open_ts) / pd.Timedelta(hours=1)) + 2
        bars, _source, _failures = bar_history.fetch_history_bars(
            symbol, "1Min", key, secret, feed, limit=10000, lookback_hours=hours,
            what="minute bars (news impact)",
        )
        frame = bars_frame(bars)
        frame = frame[frame["date"] == pd.Timestamp(today)]
    with _cache_lock:
        _today_cache[symbol] = (_time.monotonic(), today, frame)
    return frame


def reset_bar_caches() -> None:
    """Forget fetched bars -- for tests."""
    with _cache_lock:
        _history_cache.clear()
        _today_cache.clear()


def score_symbol_news(
    symbol: str,
    articles: "list[dict]",
    key: str,
    secret: str,
    feed: str,
    now=None,
) -> "dict[str, dict]":
    """Fetch what the articles need and score them with `symbol`'s model."""
    bundle = load_bundle(symbol)
    if bundle is None:
        raise LookupError(f"no news-impact model for {symbol}")
    now = pd.Timestamp(now or clock.now()).tz_convert(MARKET_TZ)
    today = now.date()
    days = [ts.date() for ts in map(release_ts, articles) if ts is not None and ts <= now]
    oldest = max(min(days, default=today), today - timedelta(days=MAX_ARTICLE_AGE_DAYS))
    first_day = oldest - timedelta(days=_calendar_days(history_sessions(bundle)))

    frames = {}
    for sym in (symbol, bundle["market_symbol"]):
        past = fetch_history(sym, first_day, today, key, secret)
        try:
            live = fetch_today(sym, key, secret, feed, now)
        except Exception:
            live = bars_frame([])  # today's articles stay pending
        frames[sym] = pd.concat([f for f in (past, live) if len(f)]) if len(past) or len(live) else past
    return score_articles(bundle, articles, frames[symbol], frames[bundle["market_symbol"]], now)


# --- the News tab's background refresh ---------------------------------------

_launch_lock = threading.Lock()


def needs_refresh(sym_state) -> bool:
    """Whether any article lacks a settled model verdict."""
    with sym_state.lock:
        news = list(sym_state.news)
        details = sym_state.news_impact_details
        for item in news:
            news_id = str(item.get("id") or "")
            if not news_id:
                continue
            detail = details.get(news_id)
            if detail is None or detail.get("source") != SOURCE_MODEL or detail.get("status") == STATUS_PENDING:
                return True
    return False


def refresh_impacts(sym_state, key: str, secret: str, feed: str, now=None) -> None:
    """Score `sym_state`'s news with its model and publish the labels."""
    with sym_state.lock:
        articles = list(sym_state.news)
    results = score_symbol_news(sym_state.symbol, articles, key, secret, feed, now)
    with sym_state.lock:
        impacts = dict(sym_state.news_impacts)
        details = dict(sym_state.news_impact_details)
        for news_id, result in results.items():
            impacts[news_id] = result["label"]
            details[news_id] = result
        # Swapped whole, so a reader never sees a half-updated pair.
        sym_state.news_impacts = impacts
        sym_state.news_impact_details = details
        sym_state.news_impact_error = None


def clear_model_impacts(sym_state) -> None:
    """Drop the model's labels, e.g. once the symbol is switched to the LLM."""
    with sym_state.lock:
        details = sym_state.news_impact_details
        mine = {k for k, v in details.items() if v.get("source") == SOURCE_MODEL}
        if not mine:
            return
        sym_state.news_impacts = {k: v for k, v in sym_state.news_impacts.items() if k not in mine}
        sym_state.news_impact_details = {k: v for k, v in details.items() if k not in mine}


def launch_refresh(sym_state, key: str, secret: str, feed: str, force: bool = False) -> bool:
    """Re-score `sym_state`'s news on a background thread, if there is anything
    to settle and no refresh is already running. Returns whether one started.

    Cheap to call on every panel poll: history is cached for the day and
    today's bars for `TODAY_TTL_SEC`, so a refresh that only settles pending
    articles is a model call, not a download.
    """
    if not force and not needs_refresh(sym_state):
        return False
    with _launch_lock:
        if sym_state.news_impact_scoring:
            return False
        if not force and _time.monotonic() < sym_state.news_impact_retry_at:
            return False
        sym_state.news_impact_scoring = True

    def run() -> None:
        try:
            refresh_impacts(sym_state, key, secret, feed)
        except Exception as exc:
            logger.warning("News-impact model scoring failed for %s: %s", sym_state.symbol, exc)
            sym_state.news_impact_error = str(exc)
            sym_state.news_impact_retry_at = _time.monotonic() + RETRY_AFTER_FAILURE_SEC
        finally:
            sym_state.news_impact_scoring = False

    threading.Thread(target=run, name=f"news-impact-{sym_state.symbol}", daemon=True).start()
    return True
