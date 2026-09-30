"""Where the day's high and low will land, called at 9:35 (HighLow).

The same question TimeToChange3's day-range model answers (`dayrange_model`),
asked of a much larger dataset and anchored differently. FinNotebooks'
`HighLow_5m` project writes it as `highlow15m_<TICKER>.*` in `Code/Models`:

    up   = log(high  / close5) / adr14     how far above the 9:35 price the high lands
    down = log(close5 / low)   / adr14     how far below it the low lands

    pred_high = max(close5 * exp(up   * adr14), high5)
    pred_low  = min(close5 * exp(-down * adr14), low5)

`close5`, `high5`, `low5` come from the first five 1-minute bars; `adr14` is the
mean `log(high/low)` of the previous 14 sessions. The blend is picked per ticker
on validation and the bundle records it: AAPL ships 0.5 LightGBM + 0.5 N-BEATS,
INTC ships N-HiTS alone, MU 0.25 LightGBM + 0.5 N-BEATS + 0.25 N-HiTS, BE 0.75
N-BEATS + 0.25 N-HiTS; every other candidate is in the bundle at weight 0 and
is not loaded. On the 129-session test window AAPL scores 0.0050 mean absolute
log error per extreme ($1.37) against TimeToChange3's 0.0077, INTC 0.0150
($1.22) against 0.0216.

INTC's notebook also built three custom feature groups (market via SPY,
regime, microstructure). Validation kept only microstructure, and only for the
linear median, which ships at weight 0 -- so the shipped forecast reads the base
48 features alone and none of the groups is mirrored here. A retrain that gives
a candidate reading them a weight would need them copied first;
`_build_bundle` refuses such a bundle rather than predicting off missing columns.
The one group that is mirrored is BE's "theme" (below).

Half days: the notebook drops an explicit list of NYSE 13:00 closes (a bar
count misses some -- INTC's 28 Nov 2025 passed it). That list ends in 2025, so
`session_report` uses the rule that generates it (`newsimpact_model.early_close`).

Apple Trader uses it *only* for the forecast. Everything downstream -- the two
resting levels, the breach update, the managed exit, the circuit breaker -- is
`apple_trader.DayRangeTrader`, unchanged, reading `pred_high`/`pred_low` from
here instead of from `dayrange_model`. `forecast_session` returns the same keys
for exactly that reason.

The mirror contract
-------------------
The `--- features` and `--- models` blocks are verbatim copies of
`highlow/features.py` and the inference half of `highlow/models.py`. The saved
model is a function of those exact column definitions; if `highlow` changes,
retrain **and** update this module. `tests/test_highlow_model.py` re-forecasts
the notebook's own panel rows from the bundle and pins the result.

The pickle is stamped `highlow.models`, so `_register_unpickle_alias` installs
a stub pointing at the mirrors below (the real package wins if importable).

What the live path has to supply
--------------------------------
This model's daily bars are **rolled up from 1-minute SIP bars**, not taken
from a daily feed: the day's high and low are the minute extremes, the close is
the 15:59 bar's close, volume is regular-hours only, `rv` is realised
volatility from 1-minute log returns, and half days are dropped outright. The
longest window (126-day momentum and volatility) needs 127 prior sessions of
that, and `or_volume_z` needs the opening five minutes' volume of 28 of them.
No daily feed can supply `rv` or the opening volumes, and yfinance serves about
30 days of minute bars, so the history comes from Alpaca SIP, split-adjusted --
the tape the notebook fitted on -- and there is no fallback.

That is ~140 sessions of minute bars, so the per-session rollups are cached on
disk (`HISTORY_DIR`) and only missing sessions are fetched. Completed sessions
never change, except that a split rewrites all of them: every incremental fetch
re-reads the cached session at its seam and rebuilds the cache if it moved.

History is always read strictly *before* the session being forecast, from the
session date the caller names rather than from the wall clock, so a SimLab
replay reads the same honest history a live run on that day would have.

Today's open is the first minute bar's, not the official auction print that
`dayrange_model` asks for: this model's daily frame is rolled up from minute
bars, so that is the open it was fitted on.

Minute volume is on the consolidated scale here too: `or_volume_rel` and
`or_volume_z` read today's opening five minutes against SIP history, so an IEX
opening window biases both. `dayrange_model.volume_scale_warning` applies as is,
except to a bundle fitted on IEX openings (below), which is on IEX's scale.

IEX openings (MU)
-----------------
A bundle whose sidecar says `"opening_feed": "iex"` (MU, saved 2026-09-30) was
fitted with everything about *this morning* read from IEX, because SIP is 15
minutes behind on the notebook's Alpaca plan and IEX is the only tape there is
at 9:35: the opening summary, the `close5` anchor, the `high5`/`low5` clip, and
the 28-session `or_volume_z` history of opening volumes. The daily rollups (and
so every other feature and the targets) stay on SIP. For such a bundle:

* the history cache holds SIP rollups joined to *IEX* opening summaries (the
  cache records which, and is rebuilt if asked for the other), IEX bars trimmed
  to the sessions SIP kept with no bar count of their own -- IEX is thin, and
  the notebook keeps a window of three or four bars, or one missing 09:30;
* a session IEX printed nothing in during the window (MU 2025-03-10) keeps its
  SIP rollup but has no opening row, so it drops out of the panel as it does in
  the notebook's inner join;
* `forecast_session` fetches today's IEX window itself, whatever tape the
  caller's bars are on, so a SIP replay forecasts from IEX exactly as live does.

Today's *open* (`gap_adr`, `open_vs_avg_adr`) is still the caller's first bar,
as for every bundle: the notebook read it from SIP. A replay on a SIP dataset is
therefore the notebook exactly; live at 9:35 the only 09:30 bar is IEX's, a
median 6 bp (p90 22 bp) off SIP's on MU, about 0.01-0.05 ADR on those two features.

Theme peers (BE)
----------------
BE's bundle (saved 2026-09-30) is the first whose shipped candidates read a
custom column: both nets (N-BEATS 0.75 + N-HiTS 0.25) take `lead_or_ret_adr`,
the lead peer's (VST's) opening five minutes in VST's own ADR, falling back to
the mean over all three peers (VST, PLUG, XLU) on a morning VST did not print --
13 of the notebook's 767 sessions, the latest 2025-12-31. So the "theme" group
is mirrored (`theme_from`, verbatim but for reading cached opening summaries)
and every peer is fed to it. Each peer is read as the notebook's `load_peers`
reads it: SIP rollups cleaned at the standard 385 bars, openings from IEX
(`PEER_OPENING_FEED`), cached like any other symbol, plus today's IEX window
fetched at 9:35. The group's other columns are unshipped and match the
notebook to ~1e-11, except `theme_gap`: it reads the peer's open today, which
is IEX's first bar here and SIP's in the notebook (0.02-0.04 ADR apart on the
days checked). A retrain that weights `theme_gap` inherits that seam, the
same one `gap_adr` has.

A peer whose SIP session today turns out incomplete would have no row in the
notebook, so its `lead_or_ret_adr` would fall back. At 9:35 nobody knows that
yet, so the mirror counts it. VST on 2025-12-31 (380 bars) is one: $0.0009 on
BE's predicted high. The mirror test hands the peers the windows the notebook
kept, so it pins the code and not this seam.

BE also keeps sessions of 370 bars or more rather than 385 (the sidecar's
`min_bars_per_session`, 44 quiet sessions). The own history is cleaned at that
threshold, and the cache records the threshold it was cleaned at; at 385,
2025-03-20's forecast would be $0.08 off.
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
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# The OpenMP trap `dayrange_model` documents applies unchanged: this bundle
# holds LightGBM and a torch network in one process. Same three measures, same
# reasons -- see that module's header.
os.environ.setdefault("OMP_NUM_THREADS", "1")

try:  # pragma: no cover - depends on which optional extras are installed
    importlib.import_module("lightgbm")
except ImportError:
    pass

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

torch.set_num_threads(1)

from . import market_hours, model_store  # noqa: E402
from .model_store import ModelStore  # noqa: E402
from .newsimpact_model import early_close  # noqa: E402

MODEL_PATH_ENV = "APPLE_HIGHLOW_MODEL"
DEFAULT_TICKER = model_store.DEFAULT_TICKER
# Which symbols have a bundle: `apple_models.HIGHLOW_TICKERS`.

# How much of the open the forecast may look at (`config.OPENING_MINUTES`).
OPENING_MINUTES = 5
# Which tape the opening window is read from (`Settings.opening_feed`); a
# sidecar without the key predates it and read SIP.
OPENING_FEED_SIP = "sip"
OPENING_FEED_IEX = "iex"
# The feed a theme peer's opening is read from (`highlow.data.load_peers`'
# default, whatever the modelled ticker's own opening feed).
PEER_OPENING_FEED = OPENING_FEED_IEX
SESSION_MINUTES = 390
# A full session has 390 bars. Anything under this is a half day or a feed gap
# (`config.MIN_BARS_PER_SESSION`), and the notebook drops those sessions. A
# thin name's sidecar can lower it (`min_bars_per_session`, BE's 370).
MIN_BARS_PER_SESSION = 385

# Prior sessions the feature row needs: 126-day momentum and volatility read
# `mid.shift(126)` of a series that is itself shifted a day.
MIN_PRIOR_SESSIONS = 127
# Asked for, in calendar days: comfortably more than MIN_PRIOR_SESSIONS once
# weekends, holidays and dropped half days are taken out.
HISTORY_CALENDAR_DAYS = 220

# Per-session rollups of the SIP history, one JSON file per symbol. Under the
# repo's `data/`, beside SimLab's store.
HISTORY_DIR = Path(__file__).resolve().parent.parent / "data" / "highlow"


# --- features (mirrors highlow.features) ------------------------------------

SHORT_WINDOWS = (7, 14, 28)
LONG_WINDOWS = (63, 126)
TARGETS = ["up", "down"]
EPS = 1e-12


def _pos_in_range(value, low, high):
    """Where `value` sits inside [low, high]; 0.5 for an empty range."""
    span = high - low
    return ((value - low) / span.where(span > EPS)).fillna(0.5)


def daily_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Everything known about the past by the open of day t."""
    o, h, l, c, v, rv = (daily[k] for k in ["open", "high", "low", "close", "volume", "rv"])
    mid = (o + c) / 2  # the day's average price, from open and close
    ret = np.log(mid / mid.shift(1))
    rng = np.log(h / l)
    logv = np.log(v)

    f = pd.DataFrame(index=daily.index)
    f["prev_avg"] = mid.shift(1)

    # --- yesterday -------------------------------------------------------
    f["prev_range"] = rng.shift(1)
    f["prev_rv"] = rv.shift(1)
    f["prev_body"] = np.log(c / o).shift(1)
    f["prev_ret"] = ret.shift(1)
    f["prev_close_pos"] = _pos_in_range(c, l, h).shift(1)
    f["prev_up"] = np.log(h / o).shift(1)
    f["prev_down"] = np.log(o / l).shift(1)
    f["prev_volume_z"] = ((logv - logv.rolling(28).mean()) / logv.rolling(28).std()).shift(1)

    # --- 7 / 14 / 28-day windows -----------------------------------------
    for w in SHORT_WINDOWS:
        f[f"avg{w}_dist"] = np.log(mid / mid.rolling(w).mean()).shift(1)  # price vs its average
        f[f"vol{w}"] = ret.rolling(w).std().shift(1)                      # close-to-close volatility
        f[f"adr{w}"] = rng.rolling(w).mean().shift(1)                     # average daily range
        f[f"rv{w}"] = rv.rolling(w).mean().shift(1)                       # intraday volatility
        f[f"up{w}"] = np.log(h / o).rolling(w).mean().shift(1)
        f[f"down{w}"] = np.log(o / l).rolling(w).mean().shift(1)

    # --- long-term momentum ----------------------------------------------
    f["mom28"] = np.log(mid / mid.shift(28)).shift(1)
    for w in LONG_WINDOWS:
        f[f"mom{w}"] = np.log(mid / mid.shift(w)).shift(1)
        f[f"vol{w}"] = ret.rolling(w).std().shift(1)
    f["dist_high126"] = np.log(mid / h.rolling(126, min_periods=60).max()).shift(1)
    f["dist_low126"] = np.log(mid / l.rolling(126, min_periods=60).min()).shift(1)

    # --- today at 9:30 ---------------------------------------------------
    f["gap"] = np.log(o / c.shift(1))
    f["open_vs_avg"] = np.log(o / f["prev_avg"])
    f["dow"] = daily.index.dayofweek

    # TTC3's baseline: the 14-day average of each extreme against the previous mid
    f["high_off14"] = np.log(h / mid.shift(1)).rolling(14).mean().shift(1)
    f["low_off14"] = np.log(l / mid.shift(1)).rolling(14).mean().shift(1)

    # dollar ADR for the trading rule and the volume yardstick; not model inputs
    f["adr14_usd"] = (h - l).rolling(14).mean().shift(1)
    f["advol14"] = v.rolling(14).mean().shift(1)
    return f


def opening_features(minute: pd.DataFrame, n_minutes: int = OPENING_MINUTES) -> pd.DataFrame:
    """The first `n_minutes` bars of each session, summarised."""
    first = minute[minute["minute"] < n_minutes]
    g = first.groupby("date")
    bar_ret = np.log(first["close"] / first["open"])

    f = pd.DataFrame({
        "open5": g["open"].first(), "high5": g["high"].max(), "low5": g["low"].min(),
        "close5": g["close"].last(), "volume5": g["volume"].sum(),
    })
    f["or_up"] = np.log(f["high5"] / f["open5"])
    f["or_down"] = np.log(f["open5"] / f["low5"])
    f["or_ret"] = np.log(f["close5"] / f["open5"])
    f["or_range"] = np.log(f["high5"] / f["low5"])
    f["or_close_pos"] = _pos_in_range(f["close5"], f["low5"], f["high5"])
    f["or_bar_std"] = bar_ret.groupby(first["date"]).std()
    f["or_up_bars"] = (bar_ret > 0).groupby(first["date"]).mean()
    return f


_SCALE_BY_ADR = [
    "or_up", "or_down", "or_ret", "or_range", "or_bar_std", "gap", "open_vs_avg",
    "prev_range", "prev_rv", "prev_up", "prev_down", "prev_body", "prev_ret",
    "adr7", "adr28", "rv7", "rv14", "rv28", "vol7", "vol14", "vol28",
    "up7", "down7", "up14", "down14", "up28", "down28",
]


def panel_from(daily: pd.DataFrame, opening: pd.DataFrame, n_minutes: int = OPENING_MINUTES,
               peers: "tuple[dict, dict] | None" = None) -> pd.DataFrame:
    """`highlow.features.build_panel` from its two inputs' summaries.

    The notebook's `build_panel(minute, daily)` is `daily_features(daily)`
    joined to `opening_features(minute)` and then this body, verbatim, except
    that the target columns are not computed and nothing is dropped: at 9:35
    today's row has no targets, and the caller checks the feature row itself.
    Split at the join because the live path caches per-session opening
    summaries rather than re-reading months of minute bars.

    `peers` is `({ticker: opening summaries}, {ticker: daily bars})`: given, the
    "theme" group is joined as `build_panel` joins it. No other custom group is
    mirrored.
    """
    f = daily_features(daily).join(opening, how="inner")

    for col in _SCALE_BY_ADR:
        f[f"{col}_adr"] = f[col] / f["adr14"]
    f["or_range_vs_rv"] = f["or_range"] / f["rv14"]
    lv5 = np.log(f["volume5"])
    # how busy the open was against a typical 5 minutes of the last 14 days
    f["or_volume_rel"] = lv5 - np.log(f["advol14"] * n_minutes / SESSION_MINUTES)
    f["or_volume_z"] = (lv5 - lv5.rolling(28).mean().shift(1)) / lv5.rolling(28).std().shift(1)
    f["adr_trend"] = np.log(f["adr7"] / f["adr28"])  # range expanding or drying up
    f["vol_trend"] = np.log(f["vol7"] / f["vol28"])

    if peers is not None:
        f = f.join(theme_from(peers[0], peers[1], daily))
        f["own_vs_theme"] = f["or_ret"] / f["adr14"] - f["theme_or_ret"]
    return f


FEATURE_COLS: "list[str]" = [
    # the first five minutes
    "or_up_adr", "or_down_adr", "or_ret_adr", "or_range_adr", "or_bar_std_adr",
    "or_close_pos", "or_up_bars", "or_range_vs_rv", "or_volume_rel", "or_volume_z",
    # the open against yesterday
    "gap_adr", "open_vs_avg_adr",
    # yesterday
    "prev_range_adr", "prev_rv_adr", "prev_up_adr", "prev_down_adr", "prev_body_adr",
    "prev_ret_adr", "prev_close_pos", "prev_volume_z",
    # 7/14/28-day averages and volatility
    "avg7_dist", "avg14_dist", "avg28_dist",
    "adr7_adr", "adr28_adr", "rv7_adr", "rv14_adr", "rv28_adr",
    "vol7_adr", "vol14_adr", "vol28_adr",
    "up7_adr", "down7_adr", "up14_adr", "down14_adr", "up28_adr", "down28_adr",
    "adr_trend", "vol_trend",
    # the volatility level itself, and long-term momentum
    "adr14", "vol63", "vol126", "mom28", "mom63", "mom126", "dist_high126", "dist_low126",
    "dow",
]


# --- the "theme" group (mirrors highlow.features.theme_features) -------------

THEME_COLS: "list[str]" = [
    "theme_or_ret", "theme_gap", "theme_range", "theme_dispersion", "lead_or_ret_adr",
    "beta_theme63", "own_vs_theme",
]


def theme_from(peer_openings: "dict[str, pd.DataFrame]", peer_dailies: "dict[str, pd.DataFrame]",
               own_daily: pd.DataFrame) -> pd.DataFrame:
    """`highlow.features.theme_features`, verbatim, except that each peer's
    opening arrives already summarised (`opening_features`' columns, one row per
    session, NaN where the feed printed nothing) rather than as minute bars --
    the same split as `panel_from`. Peers in the notebook's order: the first is
    the lead."""
    f = pd.DataFrame(index=own_daily.index)
    rets, gaps, ranges = [], [], []
    peer_returns = []
    for i, (name, po) in enumerate(peer_openings.items()):
        pd_daily = peer_dailies[name]
        pf = daily_features(pd_daily)
        ret = (po["or_ret"] / pf["adr14"]).reindex(f.index)
        rets.append(ret)
        gaps.append((pf["gap"] / pf["adr14"]).reindex(f.index))
        ranges.append((po["or_range"] / pf["adr14"]).reindex(f.index))
        peer_returns.append(np.log(pd_daily["close"]).diff())
        if i == 0:  # the closest peer keeps its own column
            f["lead_or_ret_adr"] = ret

    f["theme_or_ret"] = pd.concat(rets, axis=1).mean(axis=1)
    f["theme_gap"] = pd.concat(gaps, axis=1).mean(axis=1)
    f["theme_range"] = pd.concat(ranges, axis=1).mean(axis=1)
    # all peers moving the same way is a theme day; disagreement is stock-specific news
    f["theme_dispersion"] = pd.concat(rets, axis=1).std(axis=1).fillna(0.0)
    # a peer that did not print this morning falls back to the rest of the group
    f["lead_or_ret_adr"] = f["lead_or_ret_adr"].fillna(f["theme_or_ret"])

    # sort=True is what the notebook's pandas does unasked (and 3.x warns about)
    theme_ret = pd.concat(peer_returns, axis=1, sort=True).mean(axis=1)
    both = own_daily.index.intersection(theme_ret.dropna().index)
    r_own = np.log(own_daily.loc[both, "close"]).diff()
    r_theme = theme_ret.loc[both]
    f["beta_theme63"] = (r_own.rolling(63).cov(r_theme) / r_theme.rolling(63).var()).shift(1).reindex(f.index)
    return f


def to_prices(pred: np.ndarray, frame: pd.DataFrame) -> pd.DataFrame:
    """Turn (up, down) predictions back into dollar highs and lows, clipped to
    contain the opening range."""
    pred = np.asarray(pred)
    scale = frame["adr14"].to_numpy()
    close5 = frame["close5"].to_numpy()
    high = np.maximum(close5 * np.exp(pred[:, 0] * scale), frame["high5"].to_numpy())
    low = np.minimum(close5 * np.exp(-pred[:, 1] * scale), frame["low5"].to_numpy())
    return pd.DataFrame({"pred_high": high, "pred_low": low}, index=frame.index)


# --- minute bars -> daily rollup (mirrors highlow.data) -----------------------

OHLCV = ["open", "high", "low", "close", "volume"]


def add_session_columns(minute: pd.DataFrame) -> pd.DataFrame:
    out = minute.copy()
    ts = out.index
    out["date"] = ts.normalize().tz_localize(None)
    out["minute"] = (ts.hour * 60 + ts.minute - (9 * 60 + 30)).astype("int16")
    return out


def sane_bars(minute: pd.DataFrame) -> pd.Series:
    """Bars with positive prices and an OHLC that makes sense."""
    px = minute[["open", "high", "low", "close"]]
    return (
        (px > 0).all(axis=1)
        & (minute["high"] >= minute["low"])
        & (minute["high"] >= px[["open", "close"]].max(axis=1) - 1e-9)
        & (minute["low"] <= px[["open", "close"]].min(axis=1) + 1e-9)
    )


def session_report(minute: pd.DataFrame, min_bars: int = MIN_BARS_PER_SESSION) -> pd.DataFrame:
    """One row per session: bar count, first and last minute, and whether it is kept."""
    rep = minute.groupby("date").agg(
        n_bars=("close", "size"), first_minute=("minute", "min"),
        last_minute=("minute", "max"), volume=("volume", "sum"),
    )
    # The notebook lists NYSE's 13:00 closes (`config.EARLY_CLOSE_DATES`) because
    # a bar count alone misses some: SIP can print after a 13:00 close, and
    # INTC's 28 Nov 2025 passed the count. The list stops at 2025, so the rule
    # that generates it -- identical on every date it covers -- stands in here.
    rep["half_day"] = early_close(pd.DatetimeIndex(rep.index))
    rep["short"] = rep["n_bars"] < min_bars
    rep["no_open_bar"] = rep["first_minute"] != 0
    rep["early_end"] = rep["last_minute"] < SESSION_MINUTES - 5
    rep["kept"] = ~(rep["half_day"] | rep["short"] | rep["no_open_bar"] | rep["early_end"])
    return rep


def clean_minute(raw: pd.DataFrame, min_bars: int = MIN_BARS_PER_SESSION) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """Drop broken bars and incomplete sessions (half days, feed gaps)."""
    minute = raw[sane_bars(raw)]
    minute = add_session_columns(minute[OHLCV])
    minute["volume"] = minute["volume"].astype("int64")
    rep = session_report(minute, min_bars)
    kept = rep.index[rep["kept"]]
    return minute[minute["date"].isin(kept)], rep


def daily_bars(minute: pd.DataFrame) -> pd.DataFrame:
    """Roll clean minute bars up to one row per session, with realised volatility."""
    g = minute.groupby("date")
    daily = pd.DataFrame({
        "open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
        "close": g["close"].last(), "volume": g["volume"].sum(),
    })
    r = np.log(minute["close"]).groupby(minute["date"]).diff()
    daily["rv"] = np.sqrt((r**2).groupby(minute["date"]).sum())
    daily.index.name = "date"
    return daily


def minute_frame_from_bars(bars: "list[dict]") -> pd.DataFrame:
    """Alpaca `{"t","o","h","l","c","v"}` bars -> the notebook's raw minute frame:
    regular hours (09:30-15:59 ET), exchange-local index, lower-case OHLCV."""
    if not bars:
        return pd.DataFrame(columns=OHLCV, index=pd.DatetimeIndex([], tz=market_hours.MARKET_TZ))
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
        },
        index=idx,
    )
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df.between_time(model_store.RTH_START, model_store.RTH_END)


# The per-session summary the live path caches: `daily_bars`' row plus
# `opening_features`' row, which is everything `panel_from` reads about a past
# session.
_ROLLUP_COLS = ["open", "high", "low", "close", "volume", "rv"]
_OPENING_COLS = [
    "open5", "high5", "low5", "close5", "volume5",
    "or_up", "or_down", "or_ret", "or_range", "or_close_pos", "or_bar_std", "or_up_bars",
]


def opening_minute_frame(raw: pd.DataFrame, keep_dates) -> pd.DataFrame:
    """A thin feed's minute bars as the notebook reads its opening from them:
    sane bars on the sessions the SIP tape kept, and no bar count of their own
    (notebook 1's `trim`)."""
    frame = add_session_columns(raw[sane_bars(raw)][OHLCV])
    return frame[frame["date"].isin(keep_dates)]


def session_rollups(
    raw: pd.DataFrame, opening_raw: "pd.DataFrame | None" = None,
    min_bars: int = MIN_BARS_PER_SESSION,
) -> "tuple[pd.DataFrame, list[pd.Timestamp]]":
    """Raw regular-hours minute bars -> `(rollups, dropped_dates)`.

    `rollups` has one row per kept session with `_ROLLUP_COLS` and the opening
    summary; `dropped_dates` are the sessions `clean_minute` refused (half days,
    feed gaps), remembered so the cache knows it has already looked at them.
    Every step groups by date, so rolling up in chunks gives the same rows as
    rolling up the whole history at once.

    `opening_raw` is another feed's bars for the same stretch (an IEX-opening
    bundle's): the opening summary is then read from it, and a kept session it
    printed nothing in has NaN there rather than no row.
    """
    if raw.empty:
        return pd.DataFrame(columns=_ROLLUP_COLS + _OPENING_COLS), []
    minute, rep = clean_minute(raw, min_bars)
    dropped = list(rep.index[~rep["kept"]])
    if minute.empty:
        return pd.DataFrame(columns=_ROLLUP_COLS + _OPENING_COLS), dropped
    daily = daily_bars(minute)
    if opening_raw is None:
        opening = opening_features(minute)[_OPENING_COLS]
        return daily.join(opening, how="inner"), dropped
    thin = opening_minute_frame(opening_raw, daily.index)
    opening = opening_features(thin)[_OPENING_COLS] if len(thin) else None
    return daily.join(
        opening if opening is not None else pd.DataFrame(columns=_OPENING_COLS, dtype=float),
        how="left",
    ), dropped


# --- models (mirrors the inference half of highlow.models) --------------------

SEED = 7
LOOKBACK = 32

SEQ_CHANNELS = [
    "ch_ret", "ch_range", "ch_rv", "ch_gap", "ch_body",
    "ch_up", "ch_down", "ch_close_pos", "ch_volume_z",
]
# Channels in log-price units. Divided by the forecast day's adr14.
_PRICE_CHANNELS = ["ch_ret", "ch_range", "ch_rv", "ch_gap", "ch_body", "ch_up", "ch_down"]


def channel_frame(daily: pd.DataFrame) -> pd.DataFrame:
    """Per-day channels. Each one is known once that day has closed."""
    o, h, l, c, v, rv = (daily[k] for k in ["open", "high", "low", "close", "volume", "rv"])
    mid = (o + c) / 2
    logv = np.log(v)
    ch = pd.DataFrame(index=daily.index)
    ch["ch_ret"] = np.log(mid / mid.shift(1))
    ch["ch_range"] = np.log(h / l)
    ch["ch_rv"] = rv
    ch["ch_gap"] = np.log(o / c.shift(1))
    ch["ch_body"] = np.log(c / o)
    ch["ch_up"] = np.log(h / o)
    ch["ch_down"] = np.log(o / l)
    ch["ch_close_pos"] = _pos_in_range(c, l, h)
    ch["ch_volume_z"] = (logv - logv.rolling(28).mean()) / logv.rolling(28).std()
    return ch[SEQ_CHANNELS]


def make_sequences(ch: pd.DataFrame, panel: pd.DataFrame, lookback: int = LOOKBACK) -> np.ndarray:
    """(n, lookback, channels) windows ending the day *before* each panel row."""
    pos = {d: i for i, d in enumerate(ch.index)}
    values = ch.to_numpy(dtype="float32")
    scaled = np.array([c in _PRICE_CHANNELS for c in ch.columns])
    scale = panel["adr14"].to_numpy(dtype="float32")

    out = np.empty((len(panel), lookback, values.shape[1]), dtype="float32")
    for k, d in enumerate(panel.index):
        i = pos[d]
        if i < lookback:
            raise ValueError(f"{d.date()} has only {i} prior sessions, need {lookback}")
        window = values[i - lookback : i].copy()  # strictly the past
        window[:, scaled] /= scale[k]
        out[k] = window
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _mlp(sizes: "list[int]", dropout: float = 0.0) -> nn.Sequential:
    layers: "list[nn.Module]" = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), nn.ReLU()]
        if dropout:
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class NBeatsBlock(nn.Module):
    """Shared trunk; one head explains the input (backcast), one forecasts."""

    def __init__(self, input_dim: int, hidden: int, horizon: int, theta_dim: int = 16,
                 n_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.trunk = _mlp([input_dim] + [hidden] * n_layers, dropout)
        self.to_backcast = nn.Sequential(nn.Linear(hidden, theta_dim), nn.Linear(theta_dim, input_dim))
        self.to_forecast = nn.Sequential(nn.Linear(hidden, theta_dim), nn.Linear(theta_dim, horizon))

    def forward(self, x):
        h = self.trunk(x)
        return self.to_backcast(h), self.to_forecast(h)


class NHitsBlock(nn.Module):
    """Max-pool the lookback, learn on the coarse view, interpolate the backcast back."""

    def __init__(self, lookback: int, n_channels: int, hidden: int, pool: int, horizon: int,
                 n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lookback, self.n_channels = lookback, n_channels
        self.pool = nn.MaxPool1d(pool, stride=pool, ceil_mode=True)
        pooled_dim = int(np.ceil(lookback / pool)) * n_channels
        self.trunk = _mlp([pooled_dim] + [hidden] * n_layers, dropout)
        self.to_backcast = nn.Linear(hidden, pooled_dim)
        self.to_forecast = nn.Linear(hidden, horizon)

    def forward(self, x):
        b = x.shape[0]
        seq = x.view(b, self.lookback, self.n_channels).transpose(1, 2)
        h = self.trunk(self.pool(seq).flatten(1))
        coarse = self.to_backcast(h).view(b, self.n_channels, -1)
        backcast = nn.functional.interpolate(coarse, size=self.lookback, mode="linear", align_corners=False)
        return backcast.transpose(1, 2).reshape(b, -1), self.to_forecast(h)


class SeqNet(nn.Module):
    """Residual block stack over the lookback, plus an MLP over the tabular features."""

    def __init__(self, kind: str, lookback: int, n_channels: int, n_exo: int, horizon: int = 2,
                 hidden: int = 64, n_blocks: int = 3, dropout: float = 0.1):
        super().__init__()
        self.kind, self.lookback, self.n_channels, self.n_exo = kind, lookback, n_channels, n_exo
        if kind == "nbeats":
            blocks = [NBeatsBlock(lookback * n_channels, hidden, horizon, dropout=dropout) for _ in range(n_blocks)]
        elif kind == "nhits":
            pools = [8, 4, 1, 1, 1][:n_blocks]
            blocks = [NHitsBlock(lookback, n_channels, hidden, p, horizon, dropout=dropout) for p in pools]
        else:
            raise ValueError(f"unknown kind {kind!r}")
        self.blocks = nn.ModuleList(blocks)
        self.exo = _mlp([n_exo, hidden, hidden], dropout)
        self.exo_head = nn.Linear(hidden, horizon)

    def forward(self, seq, exo):
        residual, forecast = seq.flatten(1), 0.0
        for block in self.blocks:
            backcast, block_forecast = block(residual)
            residual = residual - backcast
            forecast = forecast + block_forecast
        return forecast + self.exo_head(self.exo(exo))


class SeqRegressor:
    """Inference half of the notebook's SeqRegressor: scalers plus a SeqNet."""

    def __init__(self, kind: str, lookback: int = LOOKBACK, hidden: int = 32, n_blocks: int = 3,
                 dropout: float = 0.1, lr: float = 1e-4, weight_decay: float = 1e-2,
                 max_epochs: int = 500, patience: int = 60, batch_size: int = 64, seed: int = SEED):
        self.params = dict(kind=kind, lookback=lookback, hidden=hidden, n_blocks=n_blocks,
                           dropout=dropout, lr=lr, weight_decay=weight_decay, max_epochs=max_epochs,
                           patience=patience, batch_size=batch_size, seed=seed)
        self.net: "SeqNet | None" = None

    def _tensors(self, seq, exo):
        seq = (np.asarray(seq, "float32") - self.seq_mean) / self.seq_std
        exo = (np.asarray(exo, "float32") - self.exo_mean) / self.exo_std
        return torch.tensor(seq, dtype=torch.float32), torch.tensor(exo, dtype=torch.float32)

    def predict(self, seq, exo) -> np.ndarray:
        xs, xe = self._tensors(seq, exo)
        with torch.no_grad():
            out = self.net(xs, xe).numpy()
        return out * self.y_std + self.y_mean


class LGBMExcursion:
    """One LightGBM per target (unpickled; the notebook fitted it)."""

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.column_stack([self.models[t].predict(X[self.feature_cols]) for t in TARGETS])


class LinearMedian:
    """Median regression per target (unpickled; weight 0 in the shipped blend)."""

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.column_stack([self.models[t].predict(X[self.feature_cols]) for t in TARGETS])


class HighLowModel:
    """A weighted blend of candidates, from panel rows to dollar highs and lows."""

    def __init__(self, models: dict, weights: dict, lookback: int, feature_cols: "list[str]",
                 metadata: "dict | None" = None):
        self.models = models
        self.weights = weights
        self.lookback = lookback
        self.feature_cols = feature_cols
        self.metadata = metadata or {}

    def predict_excursions(self, panel: pd.DataFrame, seq: "np.ndarray | None" = None) -> np.ndarray:
        total, wsum = 0.0, 0.0
        for name, model in self.models.items():
            w = self.weights.get(name, 0.0)
            if w == 0:
                continue
            if isinstance(model, SeqRegressor):
                p = model.predict(seq, panel[self.feature_cols].to_numpy())
            else:
                p = model.predict(panel)
            total, wsum = total + w * p, wsum + w
        return total / wsum

    def predict_prices(self, panel: pd.DataFrame, seq: "np.ndarray | None" = None) -> pd.DataFrame:
        return to_prices(self.predict_excursions(panel, seq), panel)


# --- the saved bundle --------------------------------------------------------

def _register_unpickle_alias() -> None:
    """Make `highlow.models.{LGBMExcursion,LinearMedian}` resolvable for joblib.

    The real package would pull matplotlib and requests in through its
    `__init__`, so a stub module pointing at the mirrors above stands in --
    unless the real package is genuinely importable, in which case it wins.
    """
    if "highlow.models" in sys.modules:
        return
    try:
        __import__("highlow.models")
        return
    except Exception:
        sys.modules.pop("highlow", None)
    package = types.ModuleType("highlow")
    package.__path__ = []
    module = types.ModuleType("highlow.models")
    module.LGBMExcursion = LGBMExcursion
    module.LinearMedian = LinearMedian
    module.SeqRegressor = SeqRegressor
    module.SeqNet = SeqNet
    package.models = module
    sys.modules["highlow"] = package
    sys.modules["highlow.models"] = module


_STORE = ModelStore(
    env_key=MODEL_PATH_ENV,
    filename="highlow15m_{ticker}.joblib",
    build=lambda path: _build_bundle(path),
)

model_path = _STORE.path
metadata_path = _STORE.metadata_path


def checkpoint_path(kind: str, path: "Path | None" = None) -> Path:
    """Where one sequence model's weights sit, beside the joblib."""
    return _STORE.sidecar_path(f"_{kind}.pt", path)


def _restore_seq(path: Path) -> SeqRegressor:
    saved = torch.load(path, weights_only=False)
    p = saved["params"]
    reg = SeqRegressor(**p)
    reg.net = SeqNet(p["kind"], p["lookback"], saved["n_channels"], saved["n_exo"], 2,
                     p["hidden"], p["n_blocks"], p["dropout"])
    reg.net.load_state_dict(saved["state_dict"])
    reg.net.eval()
    for k in ("seq_mean", "seq_std", "exo_mean", "exo_std", "y_mean", "y_std"):
        setattr(reg, k, saved[k])
    return reg


def _build_bundle(path: Path) -> "dict | None":
    """One ticker's saved model plus its metadata, or None when it cannot be
    assembled.

    Only the candidates the blend actually weights are required: a missing
    N-HiTS checkpoint is harmless at weight 0, a missing N-BEATS one is not.
    Refuses rather than degrades when a weighted voter is missing or its
    weights will not load onto these definitions (a drifted mirror).
    """
    try:
        import joblib
    except ImportError:
        return None
    _register_unpickle_alias()
    try:
        # The pickle holds the linear median (weight 0, dropped below), whose
        # scikit-learn pipeline warns about the notebook's older version. It is
        # never called, so the warning describes nothing this app runs.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            blob = joblib.load(path)
    except (OSError, ValueError, KeyError, ModuleNotFoundError, AttributeError, ImportError):
        return None
    if not isinstance(blob, dict) or {"models", "weights", "lookback", "feature_cols"} - set(blob):
        return None

    weights = {k: float(w) for k, w in blob["weights"].items()}
    models = {k: m for k, m in dict(blob["models"]).items() if weights.get(k, 0.0) > 0}
    try:
        for kind, w in weights.items():
            if w > 0 and kind not in models:
                models[kind] = _restore_seq(checkpoint_path(kind, path))
    except (OSError, KeyError, RuntimeError, ValueError, AttributeError):
        return None
    if not models:
        return None

    meta_file = metadata_path(path)
    try:
        metadata = json.loads(meta_file.read_text()) if meta_file.exists() else {}
    except (OSError, ValueError):
        metadata = {}
    peers = tuple(str(t).upper() for t in (metadata.get("peers") or ()))

    # The base feature set is mirrored, and the "theme" group when the sidecar
    # names the peers it reads (BE). A weighted candidate reading any other
    # custom group (INTC's market / regime / microstructure) would hit columns
    # this module never builds, so the bundle is refused here rather than
    # failing -- or worse, predicting -- at 9:35.
    read = set(blob["feature_cols"])
    for m in models.values():
        read |= set(getattr(m, "feature_cols", None) or ())
    if read - set(FEATURE_COLS) - (set(THEME_COLS) if peers else set()):
        return None
    if not read & set(THEME_COLS):
        peers = ()  # built for the notebook's search, read by nothing shipped

    model = HighLowModel(
        models=models, weights=weights, lookback=int(blob["lookback"]),
        feature_cols=list(blob["feature_cols"]), metadata=metadata,
    )
    return {
        "kind": "highlow",
        "model": model,
        "metadata": metadata,
        "daily_models": list(models),
        "opening_minutes": int(metadata.get("opening_minutes") or OPENING_MINUTES),
        "opening_feed": str(metadata.get("opening_feed") or OPENING_FEED_SIP).lower(),
        "min_bars": int(metadata.get("min_bars_per_session") or MIN_BARS_PER_SESSION),
        "peers": peers,
        "lookback": model.lookback,
        "trained_at": metadata.get("created"),
        "path": str(path),
        "ticker": str(metadata.get("ticker") or "").upper() or None,
    }


load_bundle = _STORE.load
reset_bundle_cache = _STORE.reset


def opening_minutes(bundle: "dict | None" = None) -> int:
    try:
        return int((bundle or {})["opening_minutes"])
    except (KeyError, TypeError, ValueError):
        return OPENING_MINUTES


def opening_feed(bundle: "dict | None" = None) -> str:
    """The tape the bundle read this morning from: "sip", or "iex" (MU)."""
    return str((bundle or {}).get("opening_feed") or OPENING_FEED_SIP)


def min_bars(bundle: "dict | None" = None) -> int:
    """The bar count a session needs to be kept: 385, or the sidecar's (BE 370)."""
    try:
        return int((bundle or {})["min_bars"])
    except (KeyError, TypeError, ValueError):
        return MIN_BARS_PER_SESSION


def peers(bundle: "dict | None" = None) -> "tuple[str, ...]":
    """The theme peers the shipped blend reads, lead first; () for most bundles."""
    return tuple((bundle or {}).get("peers") or ())


# --- the SIP history, cached per session -------------------------------------

_history_lock = threading.Lock()


def history_path(symbol: str) -> Path:
    return HISTORY_DIR / f"{symbol.upper()}_sip_sessions.json"


def _read_cache(
    symbol: str, opening_feed: str = OPENING_FEED_SIP, min_bars: int = MIN_BARS_PER_SESSION
) -> dict:
    """The symbol's cache, or an empty one if it holds another feed's openings
    or sessions cleaned at another bar count (a file written before either key
    existed holds SIP's, at 385)."""
    try:
        payload = json.loads(history_path(symbol).read_text())
    except (OSError, ValueError):
        return _empty_cache(opening_feed, min_bars)
    if (payload.get("opening_feed", OPENING_FEED_SIP) != opening_feed
            or int(payload.get("min_bars", MIN_BARS_PER_SESSION)) != min_bars):
        return _empty_cache(opening_feed, min_bars)
    payload.setdefault("sessions", {})
    payload.setdefault("dropped", [])
    payload["opening_feed"] = opening_feed
    payload["min_bars"] = min_bars
    return payload


def _write_cache(symbol: str, payload: dict) -> None:
    path = history_path(symbol)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(path)


def _credentials(key: "str | None", secret: "str | None") -> "tuple[str, str]":
    """The caller's Alpaca keys, else the environment's (SimLab has no sidebar)."""
    key = key or os.getenv("ALPACA_API_KEY", "")
    secret = secret or os.getenv("ALPACA_SECRET", "")
    if not (key and secret):
        raise ValueError(
            "the HighLow forecast needs Alpaca SIP minute history and no Alpaca "
            "credentials are available (sidebar connection, or ALPACA_API_KEY / "
            "ALPACA_SECRET in the environment)."
        )
    return key, secret


def _fetch_rollups(
    symbol: str, first: date, before: date, key: str, secret: str,
    opening_feed: str = OPENING_FEED_SIP, min_bars: int = MIN_BARS_PER_SESSION,
):
    """Rollups for the sessions in [first, before), straight from Alpaca SIP --
    with the opening summaries from `opening_feed` when that is not SIP."""
    from .datalog import log_fetch
    from .rest import fetch_bars_range

    tz = market_hours.MARKET_TZ
    start = datetime.combine(first, datetime.min.time(), tzinfo=tz)
    end = datetime.combine(before, datetime.min.time(), tzinfo=tz)
    bars = fetch_bars_range(symbol, "1Min", start, end, key, secret, feed="sip", adjustment="split")
    opening_raw = None
    if opening_feed != OPENING_FEED_SIP:
        opening_raw = minute_frame_from_bars(fetch_bars_range(
            symbol, "1Min", start, end, key, secret, feed=opening_feed, adjustment="split",
        ))
    rollups, dropped = session_rollups(minute_frame_from_bars(bars), opening_raw, min_bars)
    log_fetch(
        "minute bars (HighLow history)",
        "Alpaca REST (SIP, split-adjusted)" if opening_raw is None
        else f"Alpaca REST (SIP + {opening_feed.upper()} openings, split-adjusted)",
        symbol=symbol, detail=f"{len(bars)} bars, {len(rollups)} sessions from {first}",
    )
    return rollups, dropped


def _rows(frame: pd.DataFrame) -> dict:
    return {
        d.strftime("%Y-%m-%d"): {c: float(frame.at[d, c]) for c in frame.columns}
        for d in frame.index
    }


def _empty_cache(opening_feed: str = OPENING_FEED_SIP, min_bars: int = MIN_BARS_PER_SESSION) -> dict:
    return {"sessions": {}, "dropped": [], "from": None, "through": None,
            "opening_feed": opening_feed, "min_bars": min_bars}


def _merge(cache: dict, rollups: pd.DataFrame, dropped, seam: "str | None") -> bool:
    """Fold a fetch into the cache. False (and nothing merged) if the session
    both hold -- `seam` -- disagrees, which means a split rescaled the tape."""
    fresh = _rows(rollups)
    sessions = cache["sessions"]
    if seam and seam in sessions and seam in fresh:
        if abs(fresh[seam]["close"] / sessions[seam]["close"] - 1.0) > 1e-6:
            return False
    sessions.update(fresh)
    cache["dropped"] = sorted(
        set(cache["dropped"]) | {pd.Timestamp(d).strftime("%Y-%m-%d") for d in dropped}
    )
    return True


def history_frame(
    symbol: str,
    before,
    key: "str | None" = None,
    secret: "str | None" = None,
    opening_feed: str = OPENING_FEED_SIP,
    min_bars: int = MIN_BARS_PER_SESSION,
) -> pd.DataFrame:
    """Per-session rollups of the SIP tape for the sessions strictly before
    `before`, oldest first: `_ROLLUP_COLS` plus the opening summary, read from
    `opening_feed` (NaN on a session that feed printed nothing in), sessions
    kept at `min_bars`.

    The cache remembers which calendar days it has looked at (`from` ..
    `through`), and only the days outside that stretch are fetched -- so a live
    session fetches one day each morning and a replay of any day inside the
    stretch fetches nothing. Each extension re-reads the cached session at its
    seam: if that completed session's close has moved, a split has rescaled
    the tape, and the cache is dropped and rebuilt rather than mixed.
    """
    symbol = symbol.upper()
    before = pd.Timestamp(before).date()
    first_wanted = before - timedelta(days=HISTORY_CALENDAR_DAYS)
    last_wanted = before - timedelta(days=1)
    with _history_lock:
        cache = _read_cache(symbol, opening_feed, min_bars)
        lo = date.fromisoformat(cache["from"]) if cache.get("from") else None
        hi = date.fromisoformat(cache["through"]) if cache.get("through") else None
        if lo is None or hi is None or not cache["sessions"]:
            cache, lo, hi = _empty_cache(opening_feed, min_bars), None, None

        if lo is None or lo > first_wanted or hi < last_wanted:
            key, secret = _credentials(key, secret)

            def fetch(first: date, upto: date):
                return _fetch_rollups(symbol, first, upto, key, secret, opening_feed, min_bars)

            ok = True
            if lo is None or hi < first_wanted or lo > last_wanted:
                # Nothing usable overlaps: fetch the whole window.
                cache = _empty_cache(opening_feed, min_bars)
                _merge(cache, *fetch(first_wanted, before), None)
                lo, hi = first_wanted, last_wanted
            else:
                if lo > first_wanted:
                    seam = min(cache["sessions"])
                    upto = date.fromisoformat(seam) + timedelta(days=1)
                    ok = _merge(cache, *fetch(first_wanted, upto), seam)
                    lo = first_wanted
                if ok and hi < last_wanted:
                    seam = max(cache["sessions"])
                    ok = _merge(cache, *fetch(date.fromisoformat(seam), before), seam)
                    hi = last_wanted
            if not ok:
                cache = _empty_cache(opening_feed, min_bars)
                _merge(cache, *fetch(first_wanted, before), None)
                lo, hi = first_wanted, last_wanted
            cache["from"], cache["through"] = lo.isoformat(), hi.isoformat()
            _write_cache(symbol, cache)

    lo_s, hi_s = first_wanted.isoformat(), before.isoformat()
    rows = {d: v for d, v in cache["sessions"].items() if lo_s <= d < hi_s}
    if not rows:
        return pd.DataFrame(columns=_ROLLUP_COLS + _OPENING_COLS, index=pd.DatetimeIndex([], name="date"))
    frame = pd.DataFrame.from_dict(rows, orient="index")
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index), name="date")
    return frame.sort_index()[_ROLLUP_COLS + _OPENING_COLS]


# --- one session's forecast --------------------------------------------------

_forecast_cache: "dict[tuple, dict]" = {}
_FORECAST_CACHE_MAX = 512

# Live, the window is fetched the moment the caller's 09:34 bar has closed, and
# Alpaca can publish that minute a moment later: re-asked this many times, this
# far apart, while the window is this fresh. Never in a replay (the window is
# long past on the real clock).
_WINDOW_RETRIES = 3
_WINDOW_RETRY_SEC = 2.0
_WINDOW_FRESH_SEC = 90


def _market_indexed(frame: pd.DataFrame) -> pd.DataFrame:
    """`frame` with its index in exchange-local time (naive read as local)."""
    out = frame.copy()
    idx = pd.DatetimeIndex(out.index)
    out.index = (
        idx.tz_localize(market_hours.MARKET_TZ) if idx.tz is None
        else idx.tz_convert(market_hours.MARKET_TZ)
    )
    return out


def fetch_opening_window(
    symbol: str, session_date, want: int, feed: str,
    key: "str | None" = None, secret: "str | None" = None,
) -> pd.DataFrame:
    """The session's first `want` minutes on `feed`, straight from Alpaca, in
    the notebook's raw minute shape. Possibly short: a thin feed misses minutes."""
    from .rest import fetch_bars_range

    key, secret = _credentials(key, secret)
    day = pd.Timestamp(session_date).date()
    start = datetime(day.year, day.month, day.day, 9, 30, tzinfo=market_hours.MARKET_TZ)
    end = start + timedelta(minutes=want)
    for attempt in range(_WINDOW_RETRIES):
        bars = fetch_bars_range(symbol, "1Min", start, end, key, secret, feed=feed, adjustment="split")
        frame = minute_frame_from_bars(bars)
        complete = len(frame) and frame.index[-1] >= end - timedelta(minutes=1)
        fresh = datetime.now(timezone.utc) < end + timedelta(seconds=_WINDOW_FRESH_SEC)
        if complete or not fresh or attempt == _WINDOW_RETRIES - 1:
            return frame
        time.sleep(_WINDOW_RETRY_SEC)
    return frame


def _peer_inputs(
    peer_data: "dict[str, tuple[pd.DataFrame, pd.DataFrame | None]]", day: pd.Timestamp, want: int,
) -> "tuple[dict, dict]":
    """`theme_from`'s `(openings, dailies)` from each peer's cached history and
    today's IEX window, in the peers' order.

    Today's row is what 9:35 knows: the window's opening summary, and a daily
    row rolled up from the window, whose only column the features read is the
    open (every other statistic is shifted a day). A peer IEX printed nothing
    for gets no row today, and so falls back as in the notebook.
    """
    openings, dailies = {}, {}
    for name, (history, window) in peer_data.items():
        history = history[history.index < day]
        today_open = today_daily = None
        if window is not None and len(window):
            thin = opening_minute_frame(_market_indexed(window[OHLCV]), [day])
            thin = thin[thin["minute"] < want]
            if len(thin):
                today_open = opening_features(thin, want)[_OPENING_COLS]
                today_daily = daily_bars(thin)[_ROLLUP_COLS]
        openings[name] = pd.concat(
            [history.loc[history["open5"].notna(), _OPENING_COLS]]
            + ([today_open] if today_open is not None else [])
        )
        dailies[name] = pd.concat(
            [history[_ROLLUP_COLS]] + ([today_daily] if today_daily is not None else [])
        )
    return openings, dailies


def forecast_from(
    bundle: dict,
    history: pd.DataFrame,
    opening_bars: pd.DataFrame,
    session_date,
    opening_window: "pd.DataFrame | None" = None,
    peer_data: "dict[str, tuple[pd.DataFrame, pd.DataFrame | None]] | None" = None,
) -> dict:
    """The day's predicted high and low from a history of session rollups
    (`history_frame`) and today's first five minute bars.

    `opening_window` is today's window on the bundle's opening feed when that
    is not SIP (`opening_feed`): the opening summary, anchor and clip are read
    from it, and `opening_bars` then supplies only today's open. It may be
    short -- the notebook keeps a window IEX printed three bars in -- but not
    empty.

    `peer_data` is `{peer: (history, today's IEX window)}` for a bundle that
    reads the theme group (`peers`, BE): each peer's `history_frame` on
    `PEER_OPENING_FEED`, and its window as `fetch_opening_window` returns it
    (None or empty when IEX printed nothing).

    Returns the keys `dayrange_model.forecast_session` returns --
    `{"pred_high", "pred_low", "prev_avg", "adr14_abs", "or_high", "or_low"}`
    in dollars -- so `apple_trader.DayRangeTrader` runs on it unchanged.
    Raises ValueError when the inputs cannot support a forecast.
    """
    want = opening_minutes(bundle)
    if len(opening_bars) < want:
        raise ValueError(
            f"the forecast is built on the first {want} minutes and only "
            f"{len(opening_bars)} bars have closed."
        )
    day = pd.Timestamp(session_date).normalize()
    if day.tzinfo is not None:
        day = day.tz_localize(None)
    history = history[history.index < day]
    if len(history) < MIN_PRIOR_SESSIONS:
        raise ValueError(
            f"only {len(history)} complete SIP sessions of history before {day.date()}; "
            f"the HighLow forecast needs {MIN_PRIOR_SESSIONS} for its 126-day windows."
        )

    # Today, as the notebook's minute frame would hold it at 9:35: the first
    # `want` bars and nothing else. `daily_features` reads only today's open
    # from the daily row (every other statistic is shifted a day), so the
    # partial row cannot leak the session's outcome.
    minute = add_session_columns(_market_indexed(opening_bars.iloc[:want][OHLCV]))
    if int(minute["minute"].iloc[0]) != 0:
        raise ValueError("the opening window does not start at 09:30.")
    minute = minute[minute["date"] == day]
    if len(minute) < want:
        raise ValueError(f"the opening window is not {day.date()}'s first {want} minutes.")
    today_daily = daily_bars(minute)
    feed = opening_feed(bundle)
    if feed == OPENING_FEED_SIP:
        today_opening = opening_features(minute, want)[_OPENING_COLS]
    else:
        thin = pd.DataFrame(columns=OHLCV) if opening_window is None else opening_window[OHLCV]
        thin = opening_minute_frame(_market_indexed(thin), [day]) if len(thin) else thin
        if not len(thin) or not (thin["minute"] < want).any():
            raise ValueError(
                f"{feed.upper()} printed nothing in {day.date()}'s first {want} minutes, and "
                f"this model reads the open from {feed.upper()} only; the notebook drops "
                "such a session."
            )
        today_opening = opening_features(thin, want)[_OPENING_COLS]

    theme = None
    if peers(bundle):
        missing_peers = [t for t in peers(bundle) if t not in (peer_data or {})]
        if missing_peers:
            raise ValueError(f"this model reads its theme peers and {', '.join(missing_peers)} "
                             "was not supplied.")
        theme = _peer_inputs({t: peer_data[t] for t in peers(bundle)}, day, want)

    daily = pd.concat([history[_ROLLUP_COLS], today_daily[_ROLLUP_COLS]])
    # A past session the opening feed printed nothing in keeps its daily row but
    # leaves the panel, as the notebook's inner join leaves it out.
    opening = pd.concat([history.loc[history["open5"].notna(), _OPENING_COLS], today_opening])
    panel = panel_from(daily, opening, want, theme)
    row = panel.loc[[day]]

    model = bundle["model"]
    missing = [c for c in model.feature_cols if not np.isfinite(float(row[c].iloc[0]))]
    if missing:
        raise ValueError(
            "the feature row is incomplete "
            f"({', '.join(missing[:4])}{'…' if len(missing) > 4 else ''}); "
            + ("the SIP history is too short or has gaps"
               if not set(missing) <= set(THEME_COLS)
               else f"none of the theme peers ({', '.join(peers(bundle))}) has an IEX opening "
                    "and 14 sessions of history this morning")
            + "."
        )
    seq = make_sequences(channel_frame(daily), row, model.lookback)
    pred = model.predict_prices(row, seq).iloc[0]
    return {
        "pred_high": float(pred["pred_high"]),
        "pred_low": float(pred["pred_low"]),
        "prev_avg": float(row["prev_avg"].iloc[0]),
        "adr14_abs": float(row["adr14_usd"].iloc[0]),
        "or_high": float(row["high5"].iloc[0]),
        "or_low": float(row["low5"].iloc[0]),
    }


def warm_history(
    bundle: dict, ticker: str, before, key: "str | None" = None, secret: "str | None" = None,
) -> None:
    """Stretch the history caches `forecast_session` reads -- the ticker's and
    each theme peer's -- over the sessions before `before`."""
    history_frame(ticker, before, key, secret, opening_feed(bundle), min_bars(bundle))
    for peer in peers(bundle):
        history_frame(peer, before, key, secret, PEER_OPENING_FEED)


def forecast_session(
    bundle: dict,
    ticker: str,
    opening_bars: pd.DataFrame,
    session_date,
    key: "str | None" = None,
    secret: "str | None" = None,
) -> dict:
    """`forecast_from` with the history fetched (and cached) for `ticker`.

    Memoised on the inputs: a SimLab tuning grid replays the same session under
    dozens of configurations, and the forecast depends on none of them.

    A bundle that reads the open from IEX (`opening_feed`) is handed today's
    IEX window fetched here, whatever tape `opening_bars` is on, so a replay of
    a SIP dataset forecasts from what a live run would have seen at 9:35. So is
    a bundle that reads theme peers (BE), for each peer, beside its history.
    """
    want = opening_minutes(bundle)
    feed = opening_feed(bundle)
    window = opening_bars.iloc[:want]
    memo = (
        bundle.get("path"), str(ticker).upper(), pd.Timestamp(session_date).date(), feed,
        tuple(map(str, window.index)),
        tuple(np.round(window[OHLCV].to_numpy(dtype=float), 6).ravel()),
    )
    cached = _forecast_cache.get(memo)
    if cached is not None:
        return dict(cached)
    history = history_frame(ticker, session_date, key, secret, feed, min_bars(bundle))
    thin = None
    if feed != OPENING_FEED_SIP:
        thin = fetch_opening_window(ticker, session_date, want, feed, key, secret)
    peer_data = None
    if peers(bundle):
        peer_data = {
            peer: (
                history_frame(peer, session_date, key, secret, PEER_OPENING_FEED),
                fetch_opening_window(peer, session_date, want, PEER_OPENING_FEED, key, secret),
            )
            for peer in peers(bundle)
        }
    out = forecast_from(bundle, history, opening_bars, session_date, thin, peer_data)
    if len(_forecast_cache) >= _FORECAST_CACHE_MAX:
        _forecast_cache.clear()
    _forecast_cache[memo] = dict(out)
    return out
