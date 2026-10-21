"""The chance of a +0.2 ADR rise before the close, at every minute (GoodMoments).

FinNotebooks' `GoodMoments` project writes `goodmoments_<TICKER>.*` in
`Code/Models`. At the close of every 1-minute bar of the regular session it
answers one question:

    ADR = mean log(high / low) over the previous 14 sessions
    y_t = 1  if a later bar of the session trades at close[t] * exp(0.2 ADR)

    p      the saved LightGBM's probability of y_t (AAPL's 27 features,
           AMD's 23 -- each notebook run picks its own from the same 55)
    p_tv   the same trees on the clock and volatility alone (`frac_left`,
           `p_reach`) -- notebook 3's "time and volatility" baseline
    edge   p - p_tv: what the references, the order flow, the options and the
           analysts add to what the clock and today's volatility say

Notebook 4's chosen system (design 2) trades the **edge**: a high probability on
its own mostly means "early in a volatile session", which makes a fall as
likely as the rise. The notebook saves only the first model; the second is
refitted with the same frozen recipe by `GoodMoments/scripts/export_timevol.py`
(it checks that the recipe reproduces the saved model's trees exactly first)
and lives beside it as `goodmoments_<TICKER>_timevol.joblib`.

The live tape
-------------
The notebook's features read the SIP bar at its close; on Alpaca's basic plan
SIP runs 15 minutes late. Here today's session is read as Fast&Furious reads it
(`fastfurious_model`): at the close of bar t, SIP through bar t-15 and **IEX
for the last 15 bars**, IEX's volume put on SIP's scale by the trailing SIP/IEX
volume ratio of the last 20 sessions. Every session statistic -- the high and
low so far, VWAP, the realised variance, the 15-minute volume, the signed-flow
proxies, the developing POC -- is the SIP part through t-15 joined to the IEX
part after it, and the price is the last IEX trade. That is the one place this
module departs from the notebook, on purpose: a replay cuts a finished
session's tapes at the same two lines, so SimLab trades what a live run would
have seen. With the seam at 0 (`session_features(..., seam=0)`) every feature
is the notebook's own, which is what `tests/test_goodmoments_model.py` pins
against its panel.

The mirror contract
-------------------
The `--- features` and `--- options` blocks are copies of the parts of
`goodmoments/data.py`, `features.py`, `options.py` and `analysts.py` the shipped
model reads, restructured per session rather than per panel: the daily
statistics and the same-minute norms come from the sessions before the day,
the session statistics from the day's own tape. Same arithmetic, in the same
order. If `goodmoments` changes, retrain **and** update this module.

What the live path has to supply
--------------------------------
* ~130 calendar days of SIP and IEX minute bars before the session (shared with
  Fast&Furious: `fastfurious_model.raw_history`, cached under
  `data/fastfurious/`): 61 sessions for the 60-session momentum, 20 for the
  same-minute norms, and the option contracts' 120-day windows.
* Last night's option positioning (put/call volume, the 30-day implied move,
  dealer gamma on a price grid), built from the shared cached contract list and
  daily option bars (`option_positioning.option_history`) with this notebook's own
  recipe (open interest = cumulative volume, a flat 4.5% rate), cached per day.
* Yahoo's analyst price targets (`yfinance`), cached and refreshed hourly live.
* This session's SIP bars through t-15 and IEX bars through t.

History is read strictly before the session being forecast, from the session
date the caller names rather than the wall clock, so a SimLab replay reads what
a live run on that day would have had.
"""

from __future__ import annotations

import importlib
import json
import os
import threading
import time
import warnings
from dataclasses import dataclass
from datetime import date, datetime, timedelta
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

from scipy.signal import lfilter  # noqa: E402
from scipy.special import ndtr  # noqa: E402

from . import market_hours, model_store  # noqa: E402
from .model_store import ModelStore  # noqa: E402
from .newsimpact_model import early_close  # noqa: E402

MODEL_PATH_ENV = "APPLE_GOODMOMENTS_MODEL"
DEFAULT_TICKER = model_store.DEFAULT_TICKER
# Which symbols have a bundle: `apple_models.GOODMOMENTS_TICKERS`.

TZ = "America/New_York"

# --- the question (goodmoments.config) ------------------------------------------

SESSION_MINUTES = 390
MIN_SESSION_BARS = 380            # fewer: a half day or a broken session
HALF_DAY_MINUTES = 210            # 09:30 .. 12:59
ADR_WINDOW = 14
TARGET_ADR = 0.2
# A row needs a later bar to reach the target: the 15:58 bar's close is the
# last forecast (`build_panel` keeps `y.notna()`).
LAST_FORECAST = SESSION_MINUTES - 2

OPTION_MAX_DTE = 60               # expiries used for dealer gamma
OPTION_HISTORY_DAYS = 120         # each contract's bars, over its last 120 days
WALL_BAND = 0.15                  # the call wall: most open interest within +-15% of spot
OPTIONS_START = "2024-02-01"
RISK_FREE = 0.045                 # flat rate; gamma barely depends on it
DIV_YIELD = 0.004
# The open-interest calibration's pick (`options.calibrate_half_life`, written
# into the notebook's positioning table): no decay, i.e. cumulative volume.
HALF_LIFE = 1e9
GEX_GRID = np.linspace(-0.10, 0.10, 81)   # log-moneyness grid, 0.25% steps
STRIKE_BAND = (0.6, 1.5)          # strikes fetched, and kept, around spot

POC_BIN = 0.0005                  # volume-profile bins: 5 bps of log price
NORM_SESSIONS = 20                # sessions behind the volume and volatility norms
ANALYST_DAYS = 90                 # the price-target mean's look-back

# --- the live tape ---------------------------------------------------------------

L = SIP_DELAY = 15                # SIP is 15 minutes late; the last 15 bars come from IEX
IEX_SCALE_SESSIONS = 20

# Sessions before the day the features read: 61 for the 60-session momentum
# (the close 61 sessions back), and nothing else reaches further.
MIN_PRIOR_SESSIONS = 62
# Asked for, in calendar days: the sessions above, and every option contract
# alive on the day followed over its last 120 days.
HISTORY_CALENDAR_DAYS = 130

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "goodmoments"
POSITIONING_LAYOUT = 1
ANALYSTS_LAYOUT = 1
# Live, the analyst table is fetched again when older than this.
ANALYST_REFRESH_SEC = 3600.0

# --- the shipped models' features -------------------------------------------------
#
# What AAPL's (27) and AMD's (23) saved models and their time-and-volatility
# twins read -- AMD's adds the weekday, the range so far, the 1-hour, the
# 20-session momentum, the volatility regime, the 14-day open average, session
# relative volume, the call wall and the 1-hour signed flow.
# Cut: every feature neither reads -- gap, position in the range, the 1- to
# 15-minute and 2-hour momentum, the 7-day close and 28-day open averages,
# realised volatility as a feature, the put wall and the gamma flip, the analyst
# counts, the 5- and 15-minute signed-flow proxies.

FEATURE_COLS = [
    "frac_left", "p_reach", "dow",
    "d_high", "d_low", "d_vwap", "range_sofar", "mom_today", "mom_1h",
    "mom_1d", "mom_20d", "mom_60d", "y_range", "y_rv", "vol_regime",
    "d_y_open", "d_y_close", "d_y_high", "d_y_low",
    "d_sma_c14", "d_sma_c28", "d_sma_o7", "d_sma_o14",
    "rel_vol", "rel_vol_15m",
    "d_poc", "d_poc_y", "d_poc_5d",
    "d_call_wall", "gex", "iv_adr", "pc_volume",
    "pt_gap",
    "bvc_1h", "bvc_sess", "clv_sess",
]
OPTION_COLS = {"d_call_wall", "gex", "iv_adr", "pc_volume"}
ANALYST_COLS = {"pt_gap"}


# --- the saved bundle ---------------------------------------------------------------

_STORE = ModelStore(
    env_key=MODEL_PATH_ENV,
    filename="goodmoments_{ticker}.joblib",
    build=lambda path: _build_bundle(path),
)

model_path = _STORE.path
metadata_path = _STORE.metadata_path
TIMEVOL_SUFFIX = "_timevol.joblib"


def timevol_path(path: "Path | None" = None) -> Path:
    """The time-and-volatility model beside the saved one."""
    return _STORE.sidecar_path(TIMEVOL_SUFFIX, path)


def _read_joblib(path: Path):
    import joblib

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return joblib.load(path)


def _booster_ok(saved, cols) -> bool:
    model = saved.get("model") if isinstance(saved, dict) else None
    if model is None or not cols or not hasattr(model, "predict"):
        return False
    if set(cols) - set(FEATURE_COLS):
        return False
    names = list(getattr(model, "feature_name", lambda: [])() or [])
    return not names or names == list(cols)


def _build_bundle(path: Path) -> "dict | None":
    """The saved model, its time-and-volatility twin and the metadata, or None
    when they cannot be assembled.

    Refuses rather than degrades when either model reads a column this module
    does not mirror, or the twin is missing: without it there is no edge, and
    the rules trade the edge."""
    try:
        import joblib  # noqa: F401
        import lightgbm  # noqa: F401
    except ImportError:
        return None
    try:
        saved = _read_joblib(path)
        twin = _read_joblib(timevol_path(path))
    except (OSError, ValueError, KeyError, EOFError, ModuleNotFoundError, AttributeError, ImportError):
        return None
    cols = list((saved or {}).get("features") or [])
    tv_cols = list((twin or {}).get("features") or [])
    if not (_booster_ok(saved, cols) and _booster_ok(twin, tv_cols)):
        return None
    meta: dict = {}
    try:
        meta_file = metadata_path(path)
        if meta_file.exists():
            meta = json.loads(meta_file.read_text())
    except (OSError, ValueError):
        meta = {}
    window = meta.get("fit_window") or []
    return {
        "kind": "goodmoments",
        "model": saved["model"],
        "features": cols,
        "timevol": twin["model"],
        "timevol_features": tv_cols,
        "metadata": meta,
        "reads_options": bool(set(cols) & OPTION_COLS),
        "reads_walls": "d_call_wall" in cols,
        "reads_analysts": bool(set(cols) & ANALYST_COLS),
        "trained_at": f"through {window[-1]}" if window else None,
        "path": str(path),
        "ticker": str(meta.get("ticker") or "").upper() or None,
    }


load_bundle = _STORE.load
reset_bundle_cache = _STORE.reset


def unavailable_reason(ticker: str = DEFAULT_TICKER) -> str:
    """Which of the two files is missing, in terms a user can act on."""
    path = model_path(ticker)
    if not path.exists():
        return (
            f"No GoodMoments model at {path} (or LightGBM, SciPy and joblib are not "
            "installed)."
        )
    if not timevol_path(path).exists():
        return (
            f"The GoodMoments model is at {path}, but its time-and-volatility twin "
            f"({timevol_path(path).name}) is not — the rules trade the model's edge over it. "
            "Write it with FinNotebooks/GoodMoments/scripts/export_timevol.py."
        )
    return (
        f"The GoodMoments files at {path} could not be read, or read a feature this app "
        "does not compute (or LightGBM, SciPy and joblib are not installed)."
    )


# --- minute bars (mirrors goodmoments.data) -------------------------------------------

BAR_COLS = ["open", "high", "low", "close", "volume", "vwap"]


def clean_sessions(raw: pd.DataFrame) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """Regular-session bars on a full 390-minute grid, plus a per-session report
    (`data.clean_sessions`, less the trade counts nothing here reads).

    Missing minutes take the previous close and zero volume. Sessions with fewer
    than MIN_SESSION_BARS real bars (the 13:00 half days) are `kept = False`
    and cut at 13:00."""
    if not len(raw):
        empty = pd.DataFrame(columns=["date", "minute", *BAR_COLS, "filled"])
        return empty, pd.DataFrame(columns=["bars", "kept", "last_minute", "filled"])
    ts = raw.index
    minute = (ts.hour * 60 + ts.minute) - (9 * 60 + 30)
    reg = raw[(minute >= 0) & (minute < SESSION_MINUTES)].copy()
    reg["date"] = reg.index.tz_localize(None).normalize()
    reg["minute"] = minute[(minute >= 0) & (minute < SESSION_MINUTES)]

    report = reg.groupby("date").agg(bars=("close", "size"))
    report["kept"] = report["bars"] >= MIN_SESSION_BARS
    report["last_minute"] = np.where(report["kept"], SESSION_MINUTES - 1, HALF_DAY_MINUTES - 1)
    reg = reg[reg["minute"].to_numpy() <= report["last_minute"].reindex(reg["date"]).to_numpy()]
    report["bars"] = reg.groupby("date")["close"].size()

    grid = pd.MultiIndex.from_product([report.index, range(SESSION_MINUTES)], names=["date", "minute"])
    full = reg.set_index(["date", "minute"])[BAR_COLS].reindex(grid)
    full["filled"] = full["close"].isna()
    full[["volume"]] = full[["volume"]].fillna(0)
    close = full["close"].groupby(level="date").ffill()
    close = close.fillna(full["open"].groupby(level="date").bfill())
    for col in ("open", "high", "low", "close", "vwap"):
        full[col] = full[col].fillna(close)

    last = report["last_minute"].reindex(full.index.get_level_values("date")).to_numpy()
    full = full[full.index.get_level_values("minute") <= last]
    report["filled"] = full["filled"].groupby(level="date").sum()
    return full.reset_index(), report


def daily_candles(minutes: pd.DataFrame) -> pd.DataFrame:
    """One regular-session candle per session (half days included), with realised volatility."""
    m = minutes.sort_values(["date", "minute"])
    r = np.log(m["close"]).groupby(m["date"]).diff()
    g = m.assign(r2=r**2, pv=m["vwap"] * m["volume"]).groupby("date")
    return pd.DataFrame({
        "open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(), "close": g["close"].last(),
        "volume": g["volume"].sum(), "vwap": g["pv"].sum() / g["volume"].sum(),
        "rv": np.sqrt(g["r2"].sum()),
    })


@dataclass
class Tape:
    """One feed's bars of one session on the 390-minute grid, filled as
    `clean_sessions` fills a kept session -- whatever part of it is in hand."""

    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    vwap: np.ndarray

    @property
    def ret(self) -> np.ndarray:
        """1-minute log returns of the close; bar 0 against its own open."""
        prev = np.concatenate([[np.nan], self.close[:-1]])
        prev[0] = self.open[0]
        return np.log(self.close / prev)


def session_tape(raw: pd.DataFrame) -> "Tape | None":
    """One session's raw bars -> its Tape, or None when it printed nothing."""
    ts = raw.index
    minute = (ts.hour * 60 + ts.minute) - (9 * 60 + 30)
    reg = raw[(minute >= 0) & (minute < SESSION_MINUTES)]
    if not len(reg):
        return None
    m = minute[(minute >= 0) & (minute < SESSION_MINUTES)]
    full = pd.DataFrame(reg[BAR_COLS].to_numpy(float), index=np.asarray(m), columns=BAR_COLS)
    full = full[~full.index.duplicated(keep="last")].reindex(range(SESSION_MINUTES))
    full["volume"] = full["volume"].fillna(0)
    close = full["close"].ffill().fillna(full["open"].bfill())
    for col in ("open", "high", "low", "close", "vwap"):
        full[col] = full[col].fillna(close)
    return Tape(*(full[c].to_numpy(float) for c in BAR_COLS))


def _tape_of(minutes: pd.DataFrame, day: pd.Timestamp) -> Tape:
    rows = minutes[minutes["date"] == day].sort_values("minute")
    return Tape(*(rows[c].to_numpy(float) for c in BAR_COLS))


# --- features (mirrors goodmoments.features) -----------------------------------------


def _gsum(x: np.ndarray) -> np.ndarray:
    """A session's running sum, as `groupby(day).cumsum()` computes it."""
    return pd.Series(x).groupby(np.zeros(len(x), dtype=int)).cumsum().to_numpy()


def _window_sum(x: np.ndarray, k: int) -> np.ndarray:
    """Sum over the last k bars of the session (`features._window_sum`)."""
    cs = pd.Series(_gsum(x))
    return (cs - cs.shift(k).fillna(0.0)).to_numpy()


def _norm_rows(wide: pd.DataFrame) -> pd.DataFrame:
    """`_norm_by_minute`'s rolling mean, one row per kept session: the mean of
    the same minute over the NORM_SESSIONS kept sessions before it."""
    return wide.rolling(NORM_SESSIONS, min_periods=10).mean().shift(1)


def _session_profile(low, high, vol):
    """Each bar's volume spread evenly over the 5 bp bins of its range; the base bin and a bars x bins matrix."""
    lo_b = np.floor(np.log(low) / POC_BIN).astype(np.int64)
    hi_b = np.floor(np.log(high) / POC_BIN).astype(np.int64)
    n = hi_b - lo_b + 1
    base, nb = lo_b.min(), hi_b.max() - lo_b.min() + 1
    rows = np.repeat(np.arange(len(low)), n)
    cols = np.repeat(lo_b - base, n) + (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n))
    a = np.zeros((len(low), nb))
    np.add.at(a, (rows, cols), np.repeat(vol / n, n))
    return base, a


def _bin_price(b):
    return np.exp((b + 0.5) * POC_BIN)


def day_pocs(minutes: pd.DataFrame, extra_day: "pd.Timestamp | None" = None) -> pd.DataFrame:
    """Per session, the full-day POC and the POC of the five sessions before it
    combined (`volume_profiles`' daily half); `extra_day` adds a row for a
    session not in `minutes` (today), whose five-session POC reads the last five."""
    profiles, day_poc = {}, {}
    for day, idx in minutes.groupby("date").indices.items():
        s = minutes.iloc[idx]
        base, a = _session_profile(s["low"].to_numpy(), s["high"].to_numpy(), s["volume"].to_numpy(float))
        total = a.sum(axis=0)
        day_poc[day] = _bin_price(base + total.argmax())
        profiles[day] = pd.Series(total, index=np.arange(base, base + len(total)))
    days = sorted(profiles)
    if extra_day is not None and extra_day not in profiles:
        days = days + [extra_day]
    poc_5d = {}
    for i, day in enumerate(days):
        if i >= 5:
            combo = pd.concat([profiles[d] for d in days[i - 5:i]]).groupby(level=0).sum()
            poc_5d[day] = _bin_price(combo.idxmax())
    return pd.DataFrame({"poc": pd.Series(day_poc, dtype=float), "poc_5d_prior": pd.Series(poc_5d, dtype=float)})


def daily_features(daily: pd.DataFrame, day_poc: pd.DataFrame) -> pd.DataFrame:
    """Known before the session opens (`features.daily_features`, the columns read)."""
    d = daily.join(day_poc)
    lr = np.log(d["high"] / d["low"])
    adr = lr.rolling(ADR_WINDOW).mean().shift(1)
    rv_norm = d["rv"].rolling(NORM_SESSIONS).mean().shift(1)
    out = pd.DataFrame(index=d.index)
    out["adr"] = adr
    out["vol_regime"] = adr / lr.rolling(60).mean().shift(1)
    out["y_range"] = lr.shift(1) / adr
    out["y_rv"] = d["rv"].shift(1) / d["rv"].rolling(NORM_SESSIONS).mean().shift(2)
    out["mom_1d"] = np.log(d["close"] / d["close"].shift(1)).shift(1) / adr
    for n in (20, 60):
        out[f"mom_{n}d"] = np.log(d["close"].shift(1) / d["close"].shift(n + 1)) / (adr * np.sqrt(n))
    for col in ("open", "close", "high", "low"):
        out[f"y_{col}"] = d[col].shift(1)
    for n in (14, 28):
        out[f"sma_c{n}"] = d["close"].rolling(n).mean().shift(1)
    for n in (7, 14):
        out[f"sma_o{n}"] = d["open"].rolling(n).mean().shift(1)
    out["poc_y"] = d["poc"].shift(1)
    out["poc_5d"] = d["poc_5d_prior"]
    out["rv_norm"] = rv_norm
    out["adv_dollar"] = (d["volume"] * d["vwap"]).rolling(NORM_SESSIONS).mean().shift(1)
    out["dow"] = d.index.dayofweek
    return out


def _session_sums(t: Tape) -> "dict[str, np.ndarray]":
    """The per-minute quantities the same-minute norms average, on one session."""
    r = t.ret
    return {
        "cum_vol": _gsum(t.volume),
        "vol_15": _window_sum(t.volume, 15),
        "cum_rv": np.sqrt(_gsum(r**2)),
        "cum_var": _gsum(r**2),
    }


def minute_norms(minutes: pd.DataFrame, report: pd.DataFrame, day: pd.Timestamp) -> "dict[str, np.ndarray]":
    """Each norm's row for `day`: the mean of the same minute over the 20 kept
    sessions before it (fewer than 10: NaN) -- `_norm_by_minute` for a kept day."""
    kept = [d for d in report.index[report["kept"]] if d < day]
    rows: "dict[str, list]" = {"cum_vol": [], "vol_15": [], "cum_rv": [], "cum_var": []}
    for d in kept:
        for name, a in _session_sums(_tape_of(minutes, d)).items():
            rows[name].append(a)
    index = pd.DatetimeIndex([*kept, day])
    out = {}
    for name, stack in rows.items():
        wide = pd.DataFrame(np.vstack([*stack, np.full(SESSION_MINUTES, np.nan)]), index=index)
        out[name] = _norm_rows(wide).iloc[-1].to_numpy(float)
    return out


def iex_scale(raw_sip: pd.DataFrame, raw_iex: pd.DataFrame, report: pd.DataFrame, day: pd.Timestamp) -> float:
    """SIP over IEX regular-session volume across the 20 kept sessions before
    `day`: what puts the IEX part of the live tape on SIP's scale."""
    kept = {d.date() for d in report.index[report["kept"]] if d < day}
    kept = set(sorted(kept)[-IEX_SCALE_SESSIONS:])

    def total(raw: pd.DataFrame) -> float:
        if not len(raw):
            return 0.0
        dates = raw.index.tz_convert(TZ).date
        return float(raw["volume"].to_numpy(float)[np.isin(dates, list(kept))].sum())

    s, i = total(raw_sip), total(raw_iex)
    return s / i if i > 0 else float("nan")


def _at(a: np.ndarray, j: np.ndarray) -> np.ndarray:
    """a[j], 0 where j < 0 (a running sum before the session)."""
    out = np.zeros(len(j))
    ok = j >= 0
    out[ok] = a[j[ok]]
    return out


def session_stats(sip: Tape, iex: "Tape | None", k: float, sigma: float, seam: int = L) -> "dict[str, np.ndarray]":
    """Per minute t, the session so far as known at the close of bar t: SIP
    through bar j = t - seam, IEX's bars j+1 .. t after it (volumes x k).

    `seam` 0 is the notebook's tape (SIP to the bar itself) and is the
    notebook's arithmetic verbatim; `seam` > 0 needs `iex`. On the joined tape
    the return into bar j+1 is IEX's close over SIP's, every later one IEX's
    own, and before any SIP bar (t < seam) the session is IEX's alone."""
    n = SESSION_MINUTES
    c_s, r_s, v_s = sip.close, sip.ret, sip.volume
    hi_s, lo_s = np.maximum.accumulate(sip.high), np.minimum.accumulate(sip.low)
    pv_s, vol_s, r2_s = _gsum(sip.vwap * v_s), _gsum(v_s), _gsum(r_s**2)

    def bvc_term(r, v):
        return v * (2 * ndtr(r / sigma) - 1)

    def clv_term(t: Tape, v):
        wide = t.high > t.low
        return v * np.where(wide, (2 * t.close - t.high - t.low) / np.where(wide, t.high - t.low, 1.0), 0.0)

    bvc_s = bvc_term(r_s, v_s)
    b_s, cl_s = _gsum(bvc_s), _gsum(clv_term(sip, v_s))
    base_s, a_s = _session_profile(sip.low, sip.high, v_s)
    if seam <= 0:
        with np.errstate(invalid="ignore", divide="ignore"):
            vol_sess = pd.Series(vol_s).replace(0, np.nan).to_numpy()
            vol_1h = pd.Series(_window_sum(v_s, 60)).replace(0, np.nan).to_numpy()
            return {
                "c": c_s, "s_open": np.full(n, sip.open[0]), "hi": hi_s, "lo": lo_s,
                "ref_1h": pd.Series(c_s).shift(60).fillna(sip.open[0]).to_numpy(),
                "vwap": pd.Series(pv_s / vol_s).fillna(pd.Series(c_s)).to_numpy(),
                "cum_vol": vol_s, "cum_var": r2_s, "vol_15": _window_sum(v_s, 15),
                "bvc_1h": _window_sum(bvc_s, 60) / vol_1h,
                "bvc_sess": b_s / vol_sess, "clv_sess": cl_s / vol_sess,
                "dev_poc": _bin_price(base_s + np.cumsum(a_s, axis=0).argmax(axis=1)),
            }
    if seam > 60:
        raise ValueError("the hour-long windows below assume the seam lies inside the hour")

    t = np.arange(n)
    j = t - seam                      # the last SIP bar; < 0 before bar `seam`
    on = j >= 0
    jj = np.maximum(j, 0)
    v_i = iex.volume * k
    r_i = iex.ret

    def joined(sip_sum: np.ndarray, iex_terms: np.ndarray, first_term: "np.ndarray | None" = None) -> np.ndarray:
        """SIP's running sum through j plus IEX's terms over j+1 .. t --
        `first_term` replacing IEX's own at bar j+1 (the joint return)."""
        tail = np.cumsum(iex_terms)
        if first_term is None:
            return np.where(on, sip_sum[jj] + tail - tail[jj], tail)
        rest = tail - tail[np.minimum(jj + 1, n - 1)]
        return np.where(on, sip_sum[jj] + first_term + rest, tail)

    c = iex.close
    s_open = np.where(on, sip.open[0], iex.open[0])
    hi_i = pd.Series(iex.high).rolling(seam, min_periods=1).max().to_numpy()
    lo_i = pd.Series(iex.low).rolling(seam, min_periods=1).min().to_numpy()
    hi = np.where(on, np.fmax(hi_s[jj], hi_i), hi_i)
    lo = np.where(on, np.fmin(lo_s[jj], lo_i), lo_i)

    vol = joined(vol_s, v_i)
    pv = joined(pv_s, iex.vwap * v_i)
    with np.errstate(invalid="ignore", divide="ignore"):
        vwap = np.where(vol > 0, pv / np.where(vol > 0, vol, 1.0), c)

    nxt = np.minimum(jj + 1, n - 1)
    joint = np.log(iex.close[nxt] / c_s[jj])
    cum_var = joined(r2_s, r_i**2, joint**2)
    bvc = joined(b_s, bvc_term(r_i, v_i), bvc_term(joint, v_i[nxt]))
    clv = joined(cl_s, clv_term(iex, v_i))

    # The 15-bar volume: SIP's bars t-14 .. j, IEX's after them.
    cs_v = np.cumsum(v_i)
    sip_15 = np.where(j >= t - 14, vol_s[jj] - _at(vol_s, t - 15), 0.0)
    vol_15 = sip_15 + cs_v - _at(cs_v, np.maximum(j, t - 15))

    # The hour: with the seam inside it, bar t-60 is SIP's, so the hour's
    # sums are the joined running sums less SIP's own through t-60, and its
    # first close is SIP's (the session's open before an hour has passed).
    ref_1h = np.where(t >= 60, c_s[np.maximum(t - 60, 0)], s_open)
    vol_1h = vol - _at(vol_s, t - 60)
    bvc_1h = bvc - _at(b_s, t - 60)

    # The developing POC: SIP's profile through j plus IEX's (x k) after it, on
    # bins covering both tapes.
    base_i, a_i = _session_profile(iex.low, iex.high, v_i)
    lo_b = min(base_s, base_i)
    width = max(base_s + a_s.shape[1], base_i + a_i.shape[1]) - lo_b
    cs_s, cs_i = np.zeros((n, width)), np.zeros((n, width))
    cs_s[:, base_s - lo_b: base_s - lo_b + a_s.shape[1]] = np.cumsum(a_s, axis=0)
    cs_i[:, base_i - lo_b: base_i - lo_b + a_i.shape[1]] = np.cumsum(a_i, axis=0)
    prof = np.where(on[:, None], cs_s[jj] + cs_i - cs_i[jj], cs_i)
    dev = _bin_price(lo_b + prof.argmax(axis=1))

    with np.errstate(invalid="ignore", divide="ignore"):
        per = np.where(vol > 0, vol, np.nan)
        return {
            "c": c, "s_open": s_open, "hi": hi, "lo": lo, "ref_1h": ref_1h, "vwap": vwap,
            "cum_vol": vol, "cum_var": cum_var, "vol_15": vol_15,
            "bvc_1h": bvc_1h / np.where(vol_1h > 0, vol_1h, np.nan),
            "bvc_sess": bvc / per, "clv_sess": clv / per, "dev_poc": dev,
        }


def analyst_events(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per action: when it is usable and the new price target (`analysts.events`)."""
    df = raw.reset_index().rename(columns={"GradeDate": "time"})
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(TZ, ambiguous="NaT", nonexistent="shift_forward")
    out = pd.DataFrame({
        "time": df["time"],
        "price_target": df["currentPriceTarget"].where(df["currentPriceTarget"] > 0),
    })
    return out.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)


def pt_gap(ts: pd.Series, price: np.ndarray, events: pd.DataFrame) -> np.ndarray:
    """The mean price target of the last 90 days' notes stamped at or before
    each bar's close, over the price, in logs (`analyst_features`' `pt_gap`)."""
    # Nanoseconds on both sides, as the notebook's pandas stored them (pandas 3
    # infers coarser units, and the windows below are counted in ns).
    et = events["time"].dt.tz_convert(TZ).dt.as_unit("ns").astype("int64").to_numpy()
    t = ts.dt.as_unit("ns").astype("int64").to_numpy()

    def trailing(values, days):
        cs = np.concatenate([[0.0], np.cumsum(values)])
        hi = np.searchsorted(et, t, side="right")
        lo = np.searchsorted(et, t - int(days * 86400e9), side="right")
        return cs[hi] - cs[lo]

    pt = events["price_target"]
    n_pt = trailing(pt.notna().to_numpy(float), ANALYST_DAYS)
    sum_pt = trailing(pt.fillna(0).to_numpy(float), ANALYST_DAYS)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.log(np.where(n_pt > 0, sum_pt / np.maximum(n_pt, 1), np.nan) / price)


def bar_close_times(day: pd.Timestamp, minutes: np.ndarray) -> pd.Series:
    """Each bar's close, exchange-local (`build_panel`'s `ts`)."""
    stamps = pd.Series(pd.Timestamp(day)).repeat(len(minutes)).reset_index(drop=True)
    return stamps.dt.tz_localize(TZ) + pd.to_timedelta(9 * 60 + 31 + pd.Series(minutes), unit="min")


def session_features(
    sip: Tape,
    iex: "Tape | None",
    day: pd.Timestamp,
    drow: pd.Series,
    norms: "dict[str, np.ndarray]",
    position: "dict | None",
    events: "pd.DataFrame | None",
    k: float = float("nan"),
    seam: int = L,
) -> pd.DataFrame:
    """Every minute's features of one session, 390 rows (`build_panel`'s
    columns the shipped models read).

    `drow` is the day's `daily_features` row, `norms` its same-minute norms
    (`minute_norms`), `position` last night's option row (`positioning_row`;
    None leaves the option columns NaN) and `events` the analyst table (None:
    `pt_gap` NaN)."""
    n = SESSION_MINUTES
    minute = np.arange(n)
    adr = float(drow["adr"])
    sigma = float(drow["rv_norm"]) / np.sqrt(SESSION_MINUTES)
    s = session_stats(sip, iex, k, sigma, seam)
    c = s["c"]

    def dist(ref):
        return np.log(c / ref) / adr

    f = pd.DataFrame(index=pd.Index(minute, name="minute"))
    f["price"] = c
    f["adr"] = adr
    f["frac_left"] = (SESSION_MINUTES - 1 - minute) / SESSION_MINUTES
    f["dow"] = int(drow["dow"])
    f["d_high"], f["d_low"] = dist(s["hi"]), dist(s["lo"])
    f["range_sofar"] = np.log(s["hi"] / s["lo"]) / adr
    f["d_vwap"] = dist(s["vwap"])
    f["mom_1h"] = np.log(c / s["ref_1h"]) / (adr * np.sqrt(60 / SESSION_MINUTES))
    f["mom_today"] = dist(s["s_open"])
    for col in ("mom_1d", "mom_20d", "mom_60d", "y_range", "y_rv", "vol_regime"):
        f[col] = float(drow[col])
    for col in ("open", "close", "high", "low"):
        f[f"d_y_{col}"] = dist(float(drow[f"y_{col}"]))
    f["d_sma_c14"], f["d_sma_c28"] = dist(float(drow["sma_c14"])), dist(float(drow["sma_c28"]))
    f["d_sma_o7"], f["d_sma_o14"] = dist(float(drow["sma_o7"])), dist(float(drow["sma_o14"]))

    cum_rv = np.sqrt(s["cum_var"])
    f["rel_vol"] = s["cum_vol"] / norms["cum_vol"]
    f["rel_vol_15m"] = s["vol_15"] / norms["vol_15"]
    rv_sofar = pd.Series(cum_rv / norms["cum_rv"], index=f.index)
    var_done = pd.Series(norms["cum_var"], index=f.index)
    var_day = float(norms["cum_var"][-1])
    sigma_left = np.sqrt((var_day - var_done).clip(lower=1e-12)) * rv_sofar.clip(0.5, 3.0)
    f["p_reach"] = 2 * (1 - ndtr(TARGET_ADR * adr / sigma_left))

    f["d_poc"] = dist(s["dev_poc"])
    f["d_poc_y"], f["d_poc_5d"] = dist(float(drow["poc_y"])), dist(float(drow["poc_5d"]))

    if position is not None:
        f["d_call_wall"] = dist(float(position.get("call_wall", np.nan)))
        grid = np.asarray(position["gex"], dtype=float)
        g0, dg = GEX_GRID[0], GEX_GRID[1] - GEX_GRID[0]
        x = np.clip((np.log(c / position["spot"]) - g0) / dg, 0, len(GEX_GRID) - 1 - 1e-9)
        i = np.nan_to_num(np.floor(x)).astype(int)
        w = x - i
        f["gex"] = (grid[i] * (1 - w) + grid[i + 1] * w) / float(drow["adv_dollar"])
        f["iv_adr"] = position["iv_30d"] / np.sqrt(252) / adr
        f["pc_volume"] = np.log(position["pc_volume"])
    else:
        f["d_call_wall"] = f["gex"] = f["iv_adr"] = f["pc_volume"] = np.nan

    f["pt_gap"] = pt_gap(bar_close_times(day, minute), c, events) if events is not None and len(events) else np.nan
    f["bvc_1h"], f["bvc_sess"], f["clv_sess"] = s["bvc_1h"], s["bvc_sess"], s["clv_sess"]
    return f.replace([np.inf, -np.inf], np.nan)


# --- option positioning (mirrors goodmoments.options) ------------------------------------


def _d1(S, K, T, sigma, r=RISK_FREE, q=DIV_YIELD):
    return (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def bs_price(S, K, T, sigma, is_call, r=RISK_FREE, q=DIV_YIELD):
    d1 = _d1(S, K, T, sigma, r, q)
    d2 = d1 - sigma * np.sqrt(T)
    call = S * np.exp(-q * T) * ndtr(d1) - K * np.exp(-r * T) * ndtr(d2)
    put = K * np.exp(-r * T) * ndtr(-d2) - S * np.exp(-q * T) * ndtr(-d1)
    return np.where(is_call, call, put)


def bs_gamma(S, K, T, sigma, q=DIV_YIELD):
    d1 = _d1(S, K, T, sigma, q=q)
    return np.exp(-q * T) * np.exp(-0.5 * d1**2) / np.sqrt(2 * np.pi) / (S * sigma * np.sqrt(T))


def implied_vol(price, S, K, T, is_call, lo=0.01, hi=3.0, iters=60):
    """Vectorised bisection; NaN where the price is outside the no-arbitrage range."""
    lo, hi = np.full_like(price, lo), np.full_like(price, hi)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        above = bs_price(S, K, T, mid, is_call) > price
        hi, lo = np.where(above, mid, hi), np.where(above, lo, mid)
    iv = 0.5 * (lo + hi)
    bad = (bs_price(S, K, T, np.full_like(price, 0.01), is_call) > price) | (
        bs_price(S, K, T, np.full_like(price, 3.0), is_call) < price)
    return np.where(bad, np.nan, iv)


def atm_vols(bars: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Median implied vol of out-of-the-money contracts within 5% of spot, per date and expiry."""
    b = bars[bars["volume"] > 0].merge(daily["close"].rename("spot"), left_on="date", right_index=True)
    b = b.assign(T=((b["expiry"] - b["date"]).dt.days.clip(lower=1)) / 365.0, is_call=b["type"] == "call")
    otm = np.where(b["is_call"], b["strike"] >= b["spot"], b["strike"] <= b["spot"])
    b = b[otm & (np.abs(np.log(b["strike"] / b["spot"])) < 0.05) & (b["expiry"] > b["date"])]
    b = b.assign(iv=implied_vol(b["close"].to_numpy(float), b["spot"].to_numpy(float), b["strike"].to_numpy(float),
                                b["T"].to_numpy(float), b["is_call"].to_numpy()))
    return b.groupby(["date", "expiry"])["iv"].median().dropna().rename("iv").reset_index()


def in_band(bars: pd.DataFrame, closes: pd.Series, through: pd.Timestamp) -> pd.DataFrame:
    """The bars `options.fetch_bars` would have fetched with the closes known
    through `through`: per expiry, strikes within 0.6-1.5 times the lowest and
    highest close over the contract's last 120 days."""
    keep = []
    for expiry, group in bars.groupby("expiry"):
        start = max(expiry - pd.Timedelta(days=OPTION_HISTORY_DAYS), pd.Timestamp(OPTIONS_START))
        spot = closes.loc[start:min(expiry, pd.Timestamp(through))]
        if spot.empty:
            continue
        lo, hi = STRIKE_BAND[0] * spot.min(), STRIKE_BAND[1] * spot.max()
        keep.append(group[group["strike"].between(lo, hi)])
    return pd.concat(keep, ignore_index=True) if keep else bars.iloc[:0]


def positioning_row(bars: pd.DataFrame, daily: pd.DataFrame, sessions: pd.DatetimeIndex,
                    prev: pd.Timestamp, day: pd.Timestamp) -> "dict | None":
    """`options.positioning`'s row for `day` as of `prev`'s close: spot, the
    call wall, the 30-day implied vol, put/call volume and dealer gamma on
    GEX_GRID. None when no contract alive over the next 60 days has open
    interest.

    `bars` are daily option bars with each contract's `type`, `strike` and
    `expiry`, through `prev`; `sessions` every session over their span."""
    meta = bars.drop_duplicates("symbol").set_index("symbol")[["type", "strike", "expiry"]]
    vol = bars.pivot_table(index="date", columns="symbol", values="volume", aggfunc="sum")
    vol = vol.reindex(sessions[(sessions >= vol.index.min()) & (sessions <= vol.index.max())]).fillna(0.0)
    if prev not in vol.index:
        return None
    decay = 0.5 ** (1.0 / HALF_LIFE)
    oi = pd.DataFrame(lfilter([1.0], [1.0, -decay], vol.to_numpy(), axis=0), index=vol.index, columns=vol.columns)
    vols = atm_vols(bars[bars["date"] == prev], daily)
    day_vol = bars[bars["date"] == prev].groupby(["date", "type"])["volume"].sum().unstack()

    spot = daily.at[prev, "close"]
    alive = meta[(meta["expiry"] >= day) & (meta["expiry"] <= day + pd.Timedelta(days=OPTION_MAX_DTE))]
    o = oi.loc[prev].reindex(alive.index).fillna(0.0)
    c = alive.assign(oi=o.to_numpy())
    c = c[(c["oi"] > 0) & c["strike"].between(spot * STRIKE_BAND[0], spot * STRIKE_BAND[1])]
    if c.empty:
        return None
    band = c[c["strike"].between(spot * (1 - WALL_BAND), spot * (1 + WALL_BAND))]
    by_strike = band.groupby(["type", "strike"])["oi"].sum()
    call_wall = by_strike.loc["call"].idxmax() if "call" in by_strike.index.get_level_values(0) else np.nan
    v = vols[vols["date"] == prev].set_index("expiry")["iv"]
    sigma = c["expiry"].map(v).fillna(v.median() if len(v) else 0.3).to_numpy()
    T = (((c["expiry"] - day).dt.days + 0.25) / 365.0).to_numpy()
    S = spot * np.exp(GEX_GRID)[:, None]
    sign = np.where(c["type"] == "call", 1.0, -1.0)
    g = bs_gamma(S, c["strike"].to_numpy()[None, :], T[None, :], sigma[None, :])
    gex = (g * (sign * c["oi"].to_numpy() * 100)[None, :]).sum(axis=1) * S[:, 0] ** 2 * 0.01
    dte = (v.index - day).days
    near = v[dte >= 1]
    has = prev in day_vol.index and {"put", "call"} <= set(day_vol.columns)
    return {
        "spot": float(spot),
        "call_wall": float(call_wall),
        "iv_30d": float(np.interp(30, dte[dte >= 1], near.to_numpy())) if len(near) else float("nan"),
        "pc_volume": float(day_vol.at[prev, "put"] / day_vol.at[prev, "call"]) if has else float("nan"),
        "gex": [float(x) for x in gex],
    }


_positioning_lock = threading.Lock()


def _positioning_path(symbol: str) -> Path:
    return CACHE_DIR / f"{symbol.upper()}_positioning.json"


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


def _today_et() -> date:
    return datetime.now(market_hours.MARKET_TZ).date()


def option_bars(symbol: str, prev: pd.Timestamp, closes: pd.Series,
                key: "str | None", secret: "str | None") -> pd.DataFrame:
    """Every contract's daily bars through `prev` that a row for the session
    after it reads: the contracts expiring within 120 days of `prev`, over
    their last 120 days, from the shared option cache (`option_positioning`),
    kept to this notebook's strike band and its standard 100-share contracts."""
    import re

    from . import option_positioning

    contracts, bars = option_positioning.option_history(
        symbol, prev, prev + pd.Timedelta(days=OPTION_HISTORY_DAYS), closes, prev, key, secret
    )
    if not len(contracts) or not len(bars):
        return pd.DataFrame(columns=["symbol", "date", "close", "volume", "type", "strike", "expiry"])
    pattern = re.compile(rf"^{re.escape(symbol.upper())}\d{{6}}[CP]\d{{8}}$")
    contracts = contracts[contracts["symbol"].map(lambda s: bool(pattern.match(str(s))))]
    meta = contracts.rename(columns={"expiration": "expiry"})[["symbol", "type", "strike", "expiry"]]
    out = bars[["symbol", "date", "close", "volume"]].merge(meta, on="symbol")
    out = out[out["date"] <= prev]
    return in_band(out, closes, prev).reset_index(drop=True)


def positioning(symbol: str, prev: pd.Timestamp, day: pd.Timestamp, daily: pd.DataFrame,
                sessions: pd.DatetimeIndex, key: "str | None" = None,
                secret: "str | None" = None, walls: bool = False) -> dict:
    """Last night's option row for `day` (from `prev`'s close), from the cache
    or built. Raises ValueError when there is none. Built only once `prev` is
    over: Alpaca's daily option bars of a session still trading are not its
    close. A row cached before the call wall was kept (AAPL's, to 2026-10-21)
    is built again only for a model that reads the wall (`walls`)."""
    iso = f"{pd.Timestamp(day):%Y-%m-%d}"
    with _positioning_lock:
        cache = _read_json(_positioning_path(symbol)) or {}
        if cache.get("layout") != POSITIONING_LAYOUT:
            cache = {"layout": POSITIONING_LAYOUT, "rows": {}}
        found = cache["rows"].get(iso)
        if (found is not None and found.get("prev") == f"{prev:%Y-%m-%d}"
                and (not walls or "call_wall" in found)):
            return found
        if pd.Timestamp(prev) >= pd.Timestamp(_today_et()):
            raise ValueError(f"the option bars of {prev.date()} are not final until that session is over.")
        closes = daily["close"].dropna()
        closes = closes[closes.index <= prev]
        bars = option_bars(symbol, prev, closes, key, secret)
        if not len(bars):
            raise ValueError(f"no {symbol} option bars through {prev.date()}.")
        row = positioning_row(bars, daily, sessions, prev, day)
        if row is None:
            raise ValueError(f"no {symbol} option contract with open interest at the {prev.date()} close.")
        if not all(np.isfinite([row["iv_30d"], row["pc_volume"]])):
            raise ValueError(f"the {prev.date()} option table has no 30-day implied vol or put/call volume.")
        row["prev"] = f"{prev:%Y-%m-%d}"
        cache["rows"][iso] = row
        _write_json(_positioning_path(symbol), cache)
        return row


# --- the analyst table -----------------------------------------------------------------

_analyst_lock = threading.Lock()
_analyst_memo: "dict[str, tuple[float, pd.DataFrame]]" = {}


def _analysts_path(symbol: str) -> Path:
    return CACHE_DIR / f"{symbol.upper()}_analysts.parquet"


def _download_analysts(symbol: str) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.Ticker(symbol).upgrades_downgrades
    if raw is None or not len(raw) or "currentPriceTarget" not in raw:
        raise ValueError(f"Yahoo returned no analyst actions for {symbol}.")
    return raw


def analysts(symbol: str, as_of: pd.Timestamp, live: bool) -> pd.DataFrame:
    """The analyst events, from a table fetched after `as_of` (a replayed
    day's close), or live from one at most an hour old; a failed fetch falls
    back to the cached table, and only a missing one raises."""
    symbol = symbol.upper()
    path = _analysts_path(symbol)
    meta_path = path.with_suffix(".json")
    with _analyst_lock:
        memo = _analyst_memo.get(symbol)
        meta = _read_json(meta_path) or {}
        fetched = float(meta.get("fetched", 0.0)) if meta.get("layout") == ANALYSTS_LAYOUT else 0.0
        fresh = (time.time() - fetched < ANALYST_REFRESH_SEC) if live else (
            fetched >= pd.Timestamp(as_of).timestamp())
        if fresh and memo is not None and memo[0] == fetched:
            return memo[1]
        raw = None
        if fresh and path.exists():
            try:
                raw = pd.read_parquet(path)
            except (OSError, ValueError):
                raw = None
        if raw is None:
            try:
                raw = _download_analysts(symbol)
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                raw.to_parquet(path)
                fetched = time.time()
                _write_json(meta_path, {"layout": ANALYSTS_LAYOUT, "fetched": fetched})
                from .datalog import log_fetch

                log_fetch("analyst actions (GoodMoments)", "Yahoo Finance (yfinance)", symbol=symbol,
                          detail=f"{len(raw)} rating and price-target actions")
            except Exception as exc:  # noqa: BLE001 -- the cached table stands in
                if not path.exists():
                    raise ValueError(f"no analyst price targets for {symbol} (Yahoo: {exc}).") from exc
                raw = pd.read_parquet(path)
        events = analyst_events(raw)
        _analyst_memo[symbol] = (fetched, events)
        return events


# --- one session's forecasts ---------------------------------------------------------------


def _normal_day(session_date) -> pd.Timestamp:
    day = pd.Timestamp(session_date).normalize()
    if day.tzinfo is not None:
        day = day.tz_localize(None)
    return day.as_unit("ns")


def is_half_day(session_date) -> bool:
    return bool(early_close(pd.DatetimeIndex([_normal_day(session_date)]))[0])


def is_session(session_date) -> bool:
    """A weekday that is not an NYSE holiday."""
    from .option_positioning import NYSE_HOLIDAYS

    day = _normal_day(session_date)
    return day.dayofweek < 5 and f"{day:%Y-%m-%d}" not in set(NYSE_HOLIDAYS)


@dataclass
class _Day:
    """What a session's forecasts read from before it, built once per session."""

    day: pd.Timestamp
    prev: pd.Timestamp
    minutes: pd.DataFrame
    report: pd.DataFrame
    daily: pd.DataFrame
    drow: pd.Series
    norms: dict
    k: float
    position: "dict | None" = None
    position_problem: str = ""
    position_tried: float = 0.0


_day_lock = threading.Lock()
_days: "dict[tuple, _Day]" = {}
_DAYS_MAX = 32
_forecast_memo: "dict[tuple, pd.DataFrame]" = {}
_FORECAST_MEMO_MAX = 256
# A failed option-table build is tried again after this long, not every minute.
_POSITIONING_RETRY_SEC = 300.0


def _ff():
    from . import fastfurious_model

    return fastfurious_model


def history_frames(raw_sip: pd.DataFrame):
    """The cleaned SIP grid, its report and its daily candles, from
    regular-session bars."""
    minutes, report = clean_sessions(raw_sip)
    return minutes, report, daily_candles(minutes)


def day_inputs(minutes: pd.DataFrame, report: pd.DataFrame, daily: pd.DataFrame,
               day: pd.Timestamp) -> "tuple[pd.Series, dict]":
    """The day's daily-features row and its same-minute norms, from the
    sessions strictly before it."""
    before = minutes[minutes["date"] < day]
    pocs = day_pocs(before, extra_day=day)
    dly = pd.concat([daily[daily.index < day], pd.DataFrame(index=pd.DatetimeIndex([day], name="date"))])
    drow = daily_features(dly, pocs).loc[day]
    return drow, minute_norms(before, report[report.index < day], day)


def _session_day(bundle: dict, symbol: str, day: pd.Timestamp,
                 key: "str | None", secret: "str | None") -> _Day:
    memo = (bundle.get("path"), symbol, day)
    with _day_lock:
        found = _days.get(memo)
    if found is not None:
        return found
    before = day.date()
    raw = _ff().raw_history(symbol, before - timedelta(days=HISTORY_CALENDAR_DAYS), before, key, secret)
    minutes, report, daily = history_frames(raw["sip"])
    if len(daily) < MIN_PRIOR_SESSIONS:
        raise ValueError(
            f"only {len(daily)} SIP sessions of history before {day.date()}; the GoodMoments "
            f"forecast needs {MIN_PRIOR_SESSIONS} (the 60-session momentum)."
        )
    drow, norms = day_inputs(minutes, report, daily, day)
    missing = [c for c in ("adr", "mom_60d", "sma_c28", "y_rv", "poc_5d", "adv_dollar", "rv_norm")
               if not np.isfinite(float(drow[c]))]
    if missing or not np.isfinite(norms["cum_var"]).all():
        raise ValueError(
            f"the history before {day.date()} has gaps ({', '.join(missing) or 'same-minute norms'})."
        )
    built = _Day(day=day, prev=daily.index.max(), minutes=minutes, report=report, daily=daily,
                 drow=drow, norms=norms, k=iex_scale(raw["sip"], raw["iex"], report, day))
    with _day_lock:
        if len(_days) >= _DAYS_MAX:
            _days.clear()
        _days.setdefault(memo, built)
        return _days[memo]


def _positioning(ctx: _Day, bundle: dict, symbol: str, key, secret) -> None:
    if ctx.position is not None:
        return
    if ctx.position_problem and time.monotonic() - ctx.position_tried < _POSITIONING_RETRY_SEC:
        return
    sessions = pd.DatetimeIndex(ctx.daily.index)
    try:
        ctx.position = positioning(symbol, ctx.prev, ctx.day, ctx.daily, sessions, key, secret,
                                   walls=bool(bundle.get("reads_walls")))
        ctx.position_problem = ""
    except Exception as exc:  # noqa: BLE001 -- reported with the forecast that needed it
        ctx.position_problem = str(exc)
        ctx.position_tried = time.monotonic()


def _predict(bundle: dict, frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    out["p"] = bundle["model"].predict(frame[bundle["features"]])
    out["p_tv"] = bundle["timevol"].predict(frame[bundle["timevol_features"]])
    out["edge"] = out["p"] - out["p_tv"]
    for col in ("price", "adr", "p_reach", "frac_left"):
        out[col] = frame[col]
    return out


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
    09:30; all of 09:30-15:58 when None or past it), one row per minute.

    Columns: `p` (the model's probability of a +0.2 ADR rise before the close),
    `p_tv` (the time-and-volatility model's), `edge` (`p - p_tv`), `price` (the
    live price it was read at), `adr` (today's ADR, a fraction of the price),
    `p_reach` and `frac_left`.

    A finished session is read whole from the history cache and memoised, so a
    SimLab grid replaying it pays once; today's is fetched live, SIP through
    `through - 15` and IEX through `through` -- waiting a few seconds for an IEX
    bar not yet published unless `wait` is off (a chart). Raises ValueError
    when the session cannot be forecast (a half day, too little history, no
    option table or analyst table)."""
    symbol = str(ticker).upper()
    day = _normal_day(session_date)
    if is_half_day(day):
        raise ValueError(
            f"{day.date()} closes at 13:00, and the notebook models full sessions only: there is "
            "no GoodMoments forecast for it."
        )
    last = LAST_FORECAST if through is None else min(int(through), LAST_FORECAST)
    if last < 0:
        raise ValueError("no bar of the session has closed yet.")
    live = day.date() >= _today_et()
    memo = (bundle.get("path"), symbol, day, last if live else None)
    found = _forecast_memo.get(memo)
    if found is not None:
        return found[found.index <= last].copy()

    ctx = _session_day(bundle, symbol, day, key, secret)
    if bundle.get("reads_options", True):
        _positioning(ctx, bundle, symbol, key, secret)
        if ctx.position is None:
            raise ValueError(f"no option positioning for the {ctx.prev.date()} close: {ctx.position_problem}")
    events = None
    if bundle.get("reads_analysts", True):
        as_of = pd.Timestamp(day).tz_localize(TZ) + pd.Timedelta(hours=16)
        events = analysts(symbol, as_of, live)

    settled = True
    if live:
        raw_sip, raw_iex, settled = _ff().live_tapes(
            symbol, day.date(), last, key, secret, retries=None if wait else 0
        )
    else:
        raw = _ff().raw_history(symbol, day.date(), day.date() + timedelta(days=1), key, secret)
        raw_sip, raw_iex = raw["sip"], raw["iex"]
        if len(raw_sip) < MIN_SESSION_BARS:
            raise ValueError(
                f"SIP printed {len(raw_sip)} regular-session bars on {day.date()}: not a session the "
                "notebook keeps, so there is no forecast for it."
            )
    iex = session_tape(raw_iex)
    if iex is None:
        raise ValueError(f"IEX has printed nothing for {symbol} on {day.date()} yet.")
    # Before bar 15 there is no SIP bar to read, and `session_stats` reads none:
    # the IEX tape stands in for the empty one.
    sip = session_tape(raw_sip) or iex
    frame = session_features(sip, iex, day, ctx.drow, ctx.norms, ctx.position, events, ctx.k, L)
    upto = last if live else LAST_FORECAST
    if live:
        # The grid is filled past IEX's last print; nothing after it is a bar.
        stamps = raw_iex.index.tz_convert(TZ)
        upto = min(upto, int((stamps.hour * 60 + stamps.minute).max()) - (9 * 60 + 30))
    frame = frame.loc[:upto].dropna(subset=["price", "p_reach"])
    out = _predict(bundle, frame)
    if settled:
        if len(_forecast_memo) >= _FORECAST_MEMO_MAX:
            _forecast_memo.clear()
        _forecast_memo[memo] = out
    return out[out.index <= last].copy()


def forecast_at(bundle: dict, ticker: str, session_date, minute: int,
                key: "str | None" = None, secret: "str | None" = None) -> dict:
    """The forecast at the close of bar `minute`: `minute`, `p`, `p_tv`,
    `edge`, `price`, `adr`, `p_reach`. Raises ValueError when there is none
    (after 15:58, before IEX has printed, a session that cannot be forecast)."""
    minute = int(minute)
    if minute > LAST_FORECAST:
        raise ValueError("the last forecast is made at the 15:58 bar's close.")
    fc = session_forecasts(bundle, ticker, session_date, minute, key, secret)
    if minute not in fc.index:
        raise ValueError(f"no forecast at bar {minute}: IEX has not printed it yet.")
    row = fc.loc[minute]
    return {"minute": minute, **{c: float(row[c]) for c in ("p", "p_tv", "edge", "price", "adr", "p_reach")}}


def session_adr(bundle: dict, ticker: str, session_date,
                key: "str | None" = None, secret: "str | None" = None) -> float:
    """Today's ADR -- the mean log(high / low) of the 14 sessions before it, a
    fraction of the price -- that the question and the rules' take and stop are
    counted in. Raises ValueError when the history cannot give one."""
    ctx = _session_day(bundle, str(ticker).upper(), _normal_day(session_date), key, secret)
    return float(ctx.drow["adr"])


def warm(bundle: dict, ticker: str, session_date,
         key: "str | None" = None, secret: "str | None" = None) -> None:
    """Fill what the session's first forecast reads, ahead of it: the minute
    history, last night's option table and the analyst table."""
    symbol = str(ticker).upper()
    day = _normal_day(session_date)
    ctx = _session_day(bundle, symbol, day, key, secret)
    if bundle.get("reads_options", True):
        _positioning(ctx, bundle, symbol, key, secret)
    if bundle.get("reads_analysts", True):
        analysts(symbol, pd.Timestamp(day).tz_localize(TZ) + pd.Timedelta(hours=16), day.date() >= _today_et())


def reset_caches() -> None:
    """Forget every memoised session (tests; a refit bundle)."""
    with _day_lock:
        _days.clear()
    _forecast_memo.clear()
    with _analyst_lock:
        _analyst_memo.clear()
