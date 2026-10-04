"""Where the day's high and low will land, called at 9:35 (HighLow2).

FinNotebooks' `HighLow2_5m` project asks HighLow_5m's question (`highlow_model`)
under stricter rules and with more inputs, and writes `highlow2_5m_<TICKER>.*`
in `Code/Models`. The target is HighLow's:

    up   = log(high  / close5) / adr14     how far above the 9:35 price the high lands
    down = log(close5 / low)   / adr14     how far below it the low lands

    pred_high = max(close5 * exp(up   * adr14), high5)
    pred_low  = min(close5 * exp(-down * adr14), low5)

What is different, and why it is a model of its own rather than another HighLow
bundle:

* **Everything about this morning is IEX's** -- `open5`, `high5`, `low5`,
  `close5`, the opening volume, *and today's open*: the gap is read off `open5`,
  not off a SIP daily open. On the notebook's Alpaca plan SIP is 15 minutes
  behind, so at 9:35 IEX is the only tape there is. The daily history and the
  targets are SIP. Unlike HighLow's IEX bundles (MU, BE) there is no seam on the
  open: a replay and a live run read the same numbers.
* **The pre-market.** Ten features read what traded before the open, as public
  at 9:35: SIP from 04:00 to its 09:19 bar (the last one out on a 15-minute
  delay), IEX's 09:20-09:29, and last evening's SIP after-hours.
* **Opening volume against its own feed's history** (`or_volume_rel` against
  the last 14 IEX openings rather than SIP daily volume), plus `or_bars_frac`,
  how much of the window IEX printed.
* **LightGBM alone**: one L1 pair per seed, three seeds averaged. No network,
  so this module never imports torch.
* Fitted from 2024 only, with the five other mega-caps pooled in (each in its
  own ADR units) and **shock days left out** of training and scoring: the day
  after earnings, NFP days, and geopolitical shocks. Neither changes inference
  -- the pool is training rows and the calendar only decides what was fitted on
  -- so a shock day is forecast like any other here. The model has never seen
  one; the notebook's trading week (notebook 6) sits out the ones known before
  the open (`events.skip_dates`). Its calendar is hand-curated and ends with the
  traded week, so nothing here reads it.

On its held-out window (8-18 Sep 2026, 9 sessions) it misses each extreme by
0.0036 log units ($1.17), 15.8% of the 14-day range -- level with HighLow_5m,
slightly ahead (the notebook's README: parity with a small edge, not a clear
win).

Apple Trader uses it only for the forecast, as it uses HighLow:
`forecast_session` returns the keys `dayrange_model.forecast_session` returns
and `apple_trader.DayRangeTrader` runs on them unchanged.

The mirror contract
-------------------
The `--- data`, `--- pre-market`, `--- features` and `--- models` blocks are
verbatim copies of the parts of `highlow2/data.py`, `premarket.py`,
`features.py` and `models.py` the shipped model reads: the base and pre-market
feature groups, and LightGBM / linear-median inference. If `highlow2` changes,
retrain **and** update this module. `tests/test_highlow2_model.py` re-forecasts
the notebook's own sessions from its raw minute files and pins the result.

Not mirrored, and refused by `_build_bundle` rather than predicted around: the
options, analysts, market and calendar groups (built and validated in the
notebook, not shipped), the N-BEATS / N-HiTS candidates (a `.pt` beside the
joblib) and the `geo` target scale, which reads yesterday's option chain.

The pickle is stamped `highlow2.models`, so `_register_unpickle_alias` installs
a stub pointing at the mirrors below (the real package wins if importable).

What the live path has to supply
--------------------------------
Per past session, cached on disk (`HISTORY_DIR`) and fetched only where the
cache falls short -- the same scheme as `highlow_model.history_frame`:

* the SIP regular-session rollup (minute high and low, 15:59 close, `rv` from
  1-minute returns; half days, short sessions and feed gaps dropped);
* the IEX opening summary, on the sessions SIP kept (none of IEX's own: it is
  thin, and the notebook keeps a window of one bar);
* the per-date pre-market pieces -- SIP to 09:19, SIP from 08:00 to 09:19, IEX
  09:20-09:29 -- and the SIP after-hours, on *every* date that printed one, half
  days included: `pm_volume_z` rolls over the notebook's summary rows, and that
  summary has a row for every date with a pre-market.

Alpaca's minute bars carry the extended hours, so one SIP and one IEX request
per stretch give all of it.

At 9:35 (`fetch_morning`): today's SIP bars from 04:00 to the 09:19 bar --
waited for, if the SIP delay has not quite passed -- and today's IEX bars from
09:20 to 09:34, re-asked briefly if the 09:34 minute is not out yet. A replay
of a day the cache already covers reads that morning from the cache instead
(`_cached_morning`): every piece of it is cut at the same 9:35 line by the same
code, so they are the same numbers without the requests.

History is always read strictly *before* the session being forecast, from the
session date the caller names rather than from the wall clock, so a SimLab
replay reads the same honest history a live run on that day would have.
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

# LightGBM brings its own OpenMP runtime, which the torch-backed model modules
# have to be loaded after (see `dayrange_model`). This one never imports torch,
# but loads LightGBM up front for the same reason they do.
os.environ.setdefault("OMP_NUM_THREADS", "1")

try:  # pragma: no cover - depends on which optional extras are installed
    importlib.import_module("lightgbm")
except ImportError:
    pass

from . import market_hours, model_store  # noqa: E402
from .model_store import RTH_END, RTH_START, ModelStore  # noqa: E402
from .newsimpact_model import early_close  # noqa: E402

MODEL_PATH_ENV = "APPLE_HIGHLOW2_MODEL"
DEFAULT_TICKER = model_store.DEFAULT_TICKER
# Which symbols have a bundle: `apple_models.HIGHLOW2_TICKERS`.

# How much of the open the forecast may look at, and from which feed
# (`config.OPENING_MINUTES`, `config.OPENING_FEED`). Every ticker the notebook
# runs reads its opening from IEX.
OPENING_MINUTES = 5
OPENING_FEED = "iex"
SESSION_MINUTES = 390
# A full session has 390 bars on SIP. Anything under this is a half day or a
# feed gap (`config.MIN_BARS_PER_SESSION`).
MIN_BARS_PER_SESSION = 385

# Prior sessions the feature row needs: 126-day momentum and volatility read
# `mid.shift(126)` of a series that is itself shifted a day. The notebook lets
# those two be missing (its data starts in 2024), but live a history that short
# is a fetch gone wrong, not a young listing.
MIN_PRIOR_SESSIONS = 127
# Asked for, in calendar days: comfortably more than MIN_PRIOR_SESSIONS once
# weekends, holidays and dropped half days are taken out.
HISTORY_CALENDAR_DAYS = 220

# Per-session and per-date summaries of the SIP and IEX tapes, one JSON file per
# symbol. Beside HighLow's `data/highlow`, not inside it: these hold IEX
# openings and the extended hours, which HighLow's files do not.
HISTORY_DIR = Path(__file__).resolve().parent.parent / "data" / "highlow2"
# The cache's layout. A file written in another one is rebuilt, not read.
CACHE_LAYOUT = 1


# --- data (mirrors highlow2.data) ---------------------------------------------

OHLCV = ["open", "high", "low", "close", "volume"]
OPEN_MINUTE = 9 * 60 + 30


def add_session_columns(bars: pd.DataFrame) -> pd.DataFrame:
    """`date` (naive) and `minute` (0 = the 9:30 bar), from the local-time index."""
    out = bars.copy()
    ts = out.index
    out["date"] = ts.normalize().tz_localize(None)
    out["minute"] = (ts.hour * 60 + ts.minute - OPEN_MINUTE).astype("int16")
    return out


def sane_bars(bars: pd.DataFrame) -> pd.Series:
    """Bars with positive prices and an OHLC that holds together."""
    px = bars[["open", "high", "low", "close"]]
    body_hi = px[["open", "close"]].max(axis=1)
    body_lo = px[["open", "close"]].min(axis=1)
    return ((px > 0).all(axis=1) & (bars["high"] >= bars["low"])
            & (bars["high"] >= body_hi - 1e-9) & (bars["low"] <= body_lo + 1e-9))


def session_report(minute: pd.DataFrame, bad_bars: "pd.Series | None" = None,
                   min_bars: int = MIN_BARS_PER_SESSION) -> pd.DataFrame:
    """One row per session the tape printed: bar count, first and last minute, and the verdict.

    A session is dropped when it is a 13:00 half day, has fewer than `min_bars` bars,
    has no 9:30 bar, or stops more than five minutes before the close. `reason` names
    the first rule that failed.
    """
    rep = minute.groupby("date").agg(
        n_bars=("close", "size"), first_minute=("minute", "min"),
        last_minute=("minute", "max"), volume=("volume", "sum"),
    )
    rep["bad_bars"] = 0 if bad_bars is None else bad_bars.reindex(rep.index, fill_value=0).astype(int)
    # The notebook lists NYSE's 13:00 closes in its window (`config.EARLY_CLOSE_DATES`),
    # and the list stops with 2025. The rule that generates it -- identical on every
    # date it covers -- stands in, as in `highlow_model`.
    rep["half_day"] = early_close(pd.DatetimeIndex(rep.index))
    rep["short"] = rep["n_bars"] < min_bars
    rep["no_open_bar"] = rep["first_minute"] > 0
    rep["early_end"] = rep["last_minute"] < SESSION_MINUTES - 5
    rules = ["half_day", "short", "no_open_bar", "early_end"]
    rep["kept"] = ~rep[rules].any(axis=1)
    rep["reason"] = ""
    for rule in reversed(rules):  # reversed, so the first failing rule wins
        rep.loc[rep[rule], "reason"] = rule
    return rep


def clean_minute(raw: pd.DataFrame, min_bars: int = MIN_BARS_PER_SESSION) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """SIP bars of complete sessions only, and the report that says why the rest went."""
    framed = add_session_columns(raw)
    ok = sane_bars(framed)
    bad = (~ok).groupby(framed["date"]).sum()
    minute = framed[ok & framed["minute"].between(0, SESSION_MINUTES - 1)].copy()
    minute["volume"] = minute["volume"].astype("int64")
    report = session_report(minute, bad, min_bars)
    kept = report.index[report["kept"]]
    return minute[minute["date"].isin(kept)], report


def trim_to_sessions(raw: pd.DataFrame, sessions) -> pd.DataFrame:
    """Another feed's bars, cleaned the same way, on the sessions SIP kept.

    Used for IEX, which skips quiet minutes. A session is never dropped for that.
    """
    framed = add_session_columns(raw)
    framed = framed[sane_bars(framed) & framed["minute"].between(0, SESSION_MINUTES - 1)]
    out = framed[framed["date"].isin(sessions)].copy()
    out["volume"] = out["volume"].astype("int64")
    return out


def daily_bars(minute: pd.DataFrame) -> pd.DataFrame:
    """Clean SIP minute bars rolled up to one row per session.

    `rv` is the day's realised volatility, the root of summed squared 1-minute log
    returns. (The notebook's `vwap` column is left out: no feature reads it.)
    """
    g = minute.groupby("date")
    daily = pd.DataFrame({
        "open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
        "close": g["close"].last(), "volume": g["volume"].sum(),
    })
    r = np.log(minute["close"]).groupby(minute["date"]).diff()
    daily["rv"] = np.sqrt((r**2).groupby(minute["date"]).sum())
    daily.index.name = "date"
    return daily


# --- pre-market (mirrors highlow2.premarket) ----------------------------------

WINDOWS = {"pre": ("04:00", "09:29"), "post": ("16:00", "19:59")}
# the last bar of each feed that is public at 09:35
LAST_PUBLIC = {"sip": "09:19", "iex": "09:29"}
# the "late" pre-market, when most of the real pre-open trading happens
LATE_START = "08:00"
# The four per-date pieces `session_summary` joins, each `_summarise`d on its
# own -- which is what lets the live path cache them per date.
PARTS = ("pm", "pml", "iexpm", "ah")
# Today's pieces at 9:35; last evening's after-hours is history by then.
MORNING_PARTS = ("pm", "pml", "iexpm")
_SUMMARY_FIELDS = ["open", "high", "low", "last", "volume", "bars", "rv"]


def extended_bars(bars: pd.DataFrame, part: str) -> pd.DataFrame:
    """One extended-hours window of a feed's bars as the notebook caches and
    loads it (`premarket._trim`, then `premarket.load`): sorted, de-duplicated,
    sane, with `date` and `clock` (minutes after midnight)."""
    bars = bars.sort_index()
    bars = bars[~bars.index.duplicated(keep="first")].between_time(*WINDOWS[part])
    bars = bars[sane_bars(bars)].copy()
    bars["date"] = bars.index.normalize().tz_localize(None)
    bars["clock"] = (bars.index.hour * 60 + bars.index.minute).astype("int16")
    return bars


def _clock(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def _summarise(bars: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """High, low, first and last price, volume, bar count and realised vol per date."""
    g = bars.groupby("date")
    r = np.log(bars["close"]).groupby(bars["date"]).diff()
    out = pd.DataFrame({
        "open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
        "last": g["close"].last(), "volume": g["volume"].sum(), "bars": g["close"].size(),
        "rv": np.sqrt((r**2).groupby(bars["date"]).sum()),
    })
    return out.add_prefix(prefix)


def public_bars(bars: pd.DataFrame, feed: str) -> pd.DataFrame:
    """Pre-market bars that are already published at 09:35 for that feed."""
    return bars[bars["clock"] <= _clock(LAST_PUBLIC[feed])]


def part_summaries(sip_pre: pd.DataFrame, iex_pre: pd.DataFrame,
                   sip_post: pd.DataFrame) -> "dict[str, pd.DataFrame]":
    """`session_summary`'s four pieces, before it joins them: the first half of
    its body, verbatim. Each is one row per date that printed in that window."""
    sip = public_bars(sip_pre, "sip")
    iex = public_bars(iex_pre, "iex")
    iex_tail = iex[iex["clock"] > _clock(LAST_PUBLIC["sip"])]
    return {
        "pm": _summarise(sip, "pm_"),
        "pml": _summarise(sip[sip["clock"] >= _clock(LATE_START)], "pml_"),
        "iexpm": _summarise(iex_tail, "iexpm_"),
        "ah": _summarise(sip_post, "ah_"),
    }


def session_summary_from(pm: pd.DataFrame, pml: pd.DataFrame, iexpm: pd.DataFrame,
                         ah: pd.DataFrame) -> pd.DataFrame:
    """`highlow2.premarket.session_summary` from its four pieces: the second half
    of its body, verbatim.

    Columns (prices in dollars, volumes in shares):
      pm_*      SIP 04:00-09:19
      pml_*     SIP 08:00-09:19, the busier late pre-market
      iexpm_*   IEX 09:20-09:29, the ten minutes SIP has not shown yet
      pre_high, pre_low, pre_last   both feeds together: the full pre-open as seen at 09:35
      ah_*      SIP 16:00-19:59 of the previous date that had after-hours bars

    The one departure: an empty after-hours piece is still joined (as missing
    columns) where the notebook would skip it and then fail on the missing
    `ah_*` columns. The notebook's tape always has one.
    """
    out = pm
    out = out.join(pml, how="left")
    out = out.join(iexpm, how="outer")
    out["pre_high"] = out[["pm_high", "iexpm_high"]].max(axis=1)
    out["pre_low"] = out[["pm_low", "iexpm_low"]].min(axis=1)
    # IEX's 09:20-09:29 is later than anything SIP shows, so it wins when it printed
    out["pre_last"] = out["iexpm_last"].fillna(out["pm_last"])

    # the evening of date d belongs to the next date that has a pre-market
    nxt = out.index.searchsorted(ah.index, side="right")
    keep = nxt < len(out.index)
    ah = ah[keep]
    ah.index = out.index[nxt[keep]]
    ah = ah[~ah.index.duplicated(keep="last")]
    out = out.join(ah, how="left")
    out.index.name = "date"
    return out


# --- features (mirrors highlow2.features: the base and pre-market groups) -----

SHORT_WINDOWS = (7, 14, 28)
LONG_WINDOWS = (63, 126)
TARGETS = ["up", "down"]
EPS = 1e-12


def _pos_in_range(value, low, high):
    """Where `value` sits inside [low, high]; 0.5 for an empty range."""
    span = high - low
    return ((value - low) / span.where(span > EPS)).fillna(0.5)


def daily_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Everything known about the past by the open of day t.

    Raw levels never enter a model - only returns, ratios and z-scores, which
    mean the same thing in 2024 and 2026. The averages use the day's mid price
    (open + close) / 2, as the brief asks.
    """
    o, h, l, c, v, rv = (daily[k] for k in ["open", "high", "low", "close", "volume", "rv"])
    mid = (o + c) / 2
    ret = np.log(mid / mid.shift(1))
    rng = np.log(h / l)
    logv = np.log(v)

    f = pd.DataFrame(index=daily.index)
    f["prev_avg"] = mid.shift(1)
    f["prev_close"] = c.shift(1)

    # yesterday
    f["prev_range"] = rng.shift(1)
    f["prev_rv"] = rv.shift(1)
    f["prev_body"] = np.log(c / o).shift(1)
    f["prev_ret"] = ret.shift(1)
    f["prev_close_pos"] = _pos_in_range(c, l, h).shift(1)
    f["prev_up"] = np.log(h / o).shift(1)
    f["prev_down"] = np.log(o / l).shift(1)
    f["prev_volume_z"] = ((logv - logv.rolling(28).mean()) / logv.rolling(28).std()).shift(1)

    # 7 / 14 / 28-day windows
    for w in SHORT_WINDOWS:
        f[f"avg{w}_dist"] = np.log(mid / mid.rolling(w).mean()).shift(1)  # price vs its average
        f[f"vol{w}"] = ret.rolling(w).std().shift(1)                      # close-to-close volatility
        f[f"adr{w}"] = rng.rolling(w).mean().shift(1)                     # average daily range
        f[f"rv{w}"] = rv.rolling(w).mean().shift(1)                       # intraday volatility
        f[f"up{w}"] = np.log(h / o).rolling(w).mean().shift(1)
        f[f"down{w}"] = np.log(o / l).rolling(w).mean().shift(1)

    # long-term momentum. The data starts in 2024, so the 126-day numbers only
    # exist from mid-2024 and stay missing before that (trees handle it).
    f["mom28"] = np.log(mid / mid.shift(28)).shift(1)
    for w in LONG_WINDOWS:
        f[f"mom{w}"] = np.log(mid / mid.shift(w)).shift(1)
        f[f"vol{w}"] = ret.rolling(w).std().shift(1)
    f["dist_high126"] = np.log(mid / h.rolling(126, min_periods=60).max()).shift(1)
    f["dist_low126"] = np.log(mid / l.rolling(126, min_periods=60).min()).shift(1)
    f["dow"] = daily.index.dayofweek

    # dollar ADR for the trading rule and dollar volume for scaling gamma; not model inputs
    f["adr14_usd"] = (h - l).rolling(14).mean().shift(1)
    f["dollar_adv20"] = (c * v).rolling(20).mean().shift(1)
    return f


def opening_features(opening: pd.DataFrame, n_minutes: int = OPENING_MINUTES) -> pd.DataFrame:
    """The first `n_minutes` bars of each session, from the opening feed (IEX)."""
    first = opening[opening["minute"] < n_minutes]
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
    # a one-bar window has no bar-to-bar variation rather than an unknown amount
    f["or_bar_std"] = bar_ret.groupby(first["date"]).std().fillna(0.0)
    f["or_up_bars"] = (bar_ret > 0).groupby(first["date"]).mean()
    # IEX can miss a minute; how much of the window it printed
    f["or_bars_frac"] = g["close"].size() / n_minutes
    return f


# divided by adr14, so a tree split learned in a calm year still applies in a wild one
_SCALE_BY_ADR = [
    "or_up", "or_down", "or_ret", "or_range", "or_bar_std", "gap", "open_vs_avg",
    "prev_range", "prev_rv", "prev_up", "prev_down", "prev_body", "prev_ret",
    "adr7", "adr28", "rv7", "rv14", "rv28", "vol7", "vol14", "vol28",
    "up7", "down7", "up14", "down14", "up28", "down28",
]


def premarket_features(summary: pd.DataFrame, core: pd.DataFrame) -> pd.DataFrame:
    """What traded before the open, as seen at 9:35 (see `premarket.session_summary`).

    SIP is read up to its 09:19 bar, the last one out at 9:35 on a 15-minute delay;
    IEX's 09:20-09:29 is live. Last evening's after-hours is hours old.
    """
    s = summary.reindex(core.index)
    adr = core["adr14"]
    f = pd.DataFrame(index=core.index)
    f["pm_range_adr"] = np.log(s["pm_high"] / s["pm_low"]) / adr
    f["pml_range_adr"] = np.log(s["pml_high"] / s["pml_low"]) / adr
    f["pm_rv_adr"] = s["pm_rv"] / adr
    f["pre_range_adr"] = np.log(s["pre_high"] / s["pre_low"]) / adr
    f["iexpm_range_adr"] = np.log(s["iexpm_high"] / s["iexpm_low"]) / adr
    f["pm_ret_adr"] = np.log(s["pre_last"] / core["prev_close"]) / adr
    # the jump from the last pre-market print to the open, and where the open sits in the pre-open range
    f["open_vs_pre_adr"] = np.log(core["open5"] / s["pre_last"]) / adr
    f["open_pos_pre"] = _pos_in_range(core["open5"], s["pre_low"], s["pre_high"])
    f["ah_range_adr"] = np.log(s["ah_high"] / s["ah_low"]) / adr
    # pre-market volume against its own last 20 sessions
    lv = np.log(summary["pm_volume"].clip(lower=1))
    z = (lv - lv.rolling(20, min_periods=10).mean().shift(1)) / lv.rolling(20, min_periods=10).std().shift(1)
    f["pm_volume_z"] = z.reindex(core.index)
    return f


GROUPS: "dict[str, list[str]]" = {
    "base": [
        # the first five minutes (IEX)
        "or_up_adr", "or_down_adr", "or_ret_adr", "or_range_adr", "or_bar_std_adr",
        "or_close_pos", "or_up_bars", "or_range_vs_rv", "or_volume_rel", "or_volume_z", "or_bars_frac",
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
    ],
    "premarket": [
        "pm_range_adr", "pml_range_adr", "pm_rv_adr", "pre_range_adr", "iexpm_range_adr", "pm_ret_adr",
        "open_vs_pre_adr", "open_pos_pre", "ah_range_adr", "pm_volume_z",
    ],
}

# Columns allowed to be missing in a row the models still use. Everything else
# must be present, or the row is dropped from the panel. (The notebook's set,
# less the columns of groups not mirrored here.)
NAN_OK = {
    "vol126", "mom126",
    # a quiet day can have no pre-market or after-hours print on one feed
    "pm_range_adr", "pml_range_adr", "pm_rv_adr", "pre_range_adr", "iexpm_range_adr", "pm_ret_adr",
    "open_vs_pre_adr", "open_pos_pre", "ah_range_adr", "pm_volume_z",
}


def feature_cols(groups=("base",)) -> "list[str]":
    cols: "list[str]" = []
    for g in groups:
        cols += [c for c in GROUPS[g] if c not in cols]
    return cols


# Every column this module can build: what a shipped candidate may read.
FEATURE_COLS = feature_cols(("base", "premarket"))


def panel_from(daily: pd.DataFrame, opening: pd.DataFrame,
               premarket: "pd.DataFrame | None" = None) -> pd.DataFrame:
    """`highlow2.features.build_panel` for the two groups the model reads.

    The notebook's `build_panel(daily, opening, premarket=...)` is
    `daily_features(daily)` joined to `opening_features(opening)` and then this
    body, verbatim, except that the targets are not computed and nothing is
    dropped: at 9:35 today's row has no targets, and the caller checks the row
    itself. Split at the join, as `highlow_model.panel_from` is, because the
    live path caches per-session opening summaries rather than re-reading
    months of minute bars. `premarket` is `session_summary`'s frame.
    """
    f = daily_features(daily).join(opening, how="inner")
    f["gap"] = np.log(f["open5"] / f["prev_close"])
    f["open_vs_avg"] = np.log(f["open5"] / f["prev_avg"])
    for col in _SCALE_BY_ADR:
        f[f"{col}_adr"] = f[col] / f["adr14"]
    f["or_range_vs_rv"] = f["or_range"] / f["rv14"]
    # opening volume against the same feed's own history: IEX carries a few percent
    # of the tape, so it must not be compared with SIP's daily volume
    lv5 = np.log(f["volume5"].clip(lower=1))
    f["or_volume_rel"] = lv5 - np.log(f["volume5"].rolling(14).mean().shift(1))
    f["or_volume_z"] = (lv5 - lv5.rolling(28).mean().shift(1)) / lv5.rolling(28).std().shift(1)
    f["adr_trend"] = np.log(f["adr7"] / f["adr28"])
    f["vol_trend"] = np.log(f["vol7"] / f["vol28"])

    if premarket is not None:
        f = f.join(premarket_features(premarket, f))
    return f


# --- models (mirrors the inference half of highlow2.models) -------------------

class LGBMExcursion:
    """One LightGBM per target (unpickled; the notebook fitted it)."""

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.column_stack([self.models[t].predict(X[self.feature_cols]) for t in TARGETS])


class LinearMedian:
    """Median regression per target on standardised features (unpickled)."""

    def _x(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.feature_cols].fillna(self.fill)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.column_stack([self.models[t].predict(self._x(X)) for t in TARGETS])


@dataclass(frozen=True)
class Target:
    """How the day's high and low are turned into the two numbers a model learns.

    `anchor` is where the excursions are measured from:
      close5    the 9:35 price (HighLow_5m's choice): up = log(high / close5)
      extremes  the opening range itself: up = log(high / high5), down = log(low5 / low).
    `scale` is what they are divided by: adr14, the 14-day average range. (The
    notebook's `geo` scale reads yesterday's implied move, from an option chain
    this module does not rebuild; `_build_bundle` refuses it.)
    """

    anchor: str = "close5"
    scale: str = "adr14"

    def scale_of(self, frame: pd.DataFrame) -> np.ndarray:
        adr = frame["adr14"].to_numpy(dtype=float)
        if self.scale == "adr14":
            return adr
        raise ValueError(f"unknown scale {self.scale!r}")

    def _bases(self, frame: pd.DataFrame) -> "tuple[np.ndarray, np.ndarray]":
        if self.anchor == "close5":
            c5 = np.log(frame["close5"].to_numpy(dtype=float))
            return c5, c5
        if self.anchor == "extremes":
            return np.log(frame["high5"].to_numpy(dtype=float)), np.log(frame["low5"].to_numpy(dtype=float))
        raise ValueError(f"unknown anchor {self.anchor!r}")

    def decode(self, pred: np.ndarray, frame: pd.DataFrame) -> pd.DataFrame:
        """(up, down) back to dollar highs and lows, never inside the opening range."""
        pred = np.asarray(pred, dtype=float)
        hb, lb = self._bases(frame)
        s = self.scale_of(frame)
        high = np.maximum(np.exp(hb + pred[:, 0] * s), frame["high5"].to_numpy())
        low = np.minimum(np.exp(lb - pred[:, 1] * s), frame["low5"].to_numpy())
        return pd.DataFrame({"pred_high": high, "pred_low": low}, index=frame.index)

    @property
    def label(self) -> str:
        return f"{self.anchor}/{self.scale}"


# The anchors and scales `Target` above can decode.
_ANCHORS = ("close5", "extremes")
_SCALES = ("adr14",)


class HighLowModel:
    """A weighted blend of tabular candidates, from panel rows to dollar highs and lows."""

    def __init__(self, models: dict, weights: dict, feature_cols: "list[str]",
                 metadata: "dict | None" = None, target: Target = Target()):
        self.models = models
        self.weights = weights
        self.feature_cols = feature_cols
        self.metadata = metadata or {}
        self.target = target

    def predict_excursions(self, panel: pd.DataFrame) -> np.ndarray:
        total, wsum = 0.0, 0.0
        for name, model in self.models.items():
            w = self.weights.get(name, 0.0)
            if w == 0:
                continue
            p = model.predict(panel)
            total, wsum = total + w * p, wsum + w
        return total / wsum

    def predict_prices(self, panel: pd.DataFrame) -> pd.DataFrame:
        return self.target.decode(self.predict_excursions(panel), panel)


# --- the saved bundle --------------------------------------------------------

def _register_unpickle_alias() -> None:
    """Make `highlow2.models.{LGBMExcursion,LinearMedian}` resolvable for joblib.

    The real package pulls torch, scikit-learn and requests in through its
    modules, so a stub pointing at the mirrors above stands in -- unless the
    real package is genuinely importable, in which case it wins.
    """
    if "highlow2.models" in sys.modules:
        return
    try:
        __import__("highlow2.models")
        return
    except Exception:
        sys.modules.pop("highlow2", None)
    package = types.ModuleType("highlow2")
    package.__path__ = []
    module = types.ModuleType("highlow2.models")
    module.LGBMExcursion = LGBMExcursion
    module.LinearMedian = LinearMedian
    package.models = module
    sys.modules["highlow2"] = package
    sys.modules["highlow2.models"] = module


_STORE = ModelStore(
    env_key=MODEL_PATH_ENV,
    filename="highlow2_5m_{ticker}.joblib",
    build=lambda path: _build_bundle(path),
)

model_path = _STORE.path
metadata_path = _STORE.metadata_path


def _build_bundle(path: Path) -> "dict | None":
    """One ticker's saved model plus its metadata, or None when it cannot be
    assembled.

    Refuses rather than degrades when the shipped blend needs something this
    module does not mirror: a weighted network (`models.save_bundle` writes
    those to a `.pt` beside the joblib, so the joblib does not hold them), a
    candidate reading a feature group other than base and pre-market, or a
    target this module cannot decode.
    """
    try:
        import joblib
    except ImportError:
        return None
    _register_unpickle_alias()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            blob = joblib.load(path)
    except (OSError, ValueError, KeyError, ModuleNotFoundError, AttributeError, ImportError, EOFError):
        return None
    if not isinstance(blob, dict) or {"models", "weights", "feature_cols"} - set(blob):
        return None

    weights = {k: float(w) for k, w in blob["weights"].items()}
    shipped = {k for k, w in weights.items() if w > 0}
    models = {k: m for k, m in dict(blob["models"]).items() if k in shipped}
    if not models or shipped - set(models):
        return None
    try:
        target = Target(**(blob.get("target") or {}))
    except TypeError:
        return None
    if target.anchor not in _ANCHORS or target.scale not in _SCALES:
        return None
    read = set(blob["feature_cols"])
    for m in models.values():
        read |= set(getattr(m, "feature_cols", None) or ())
    if read - set(FEATURE_COLS):
        return None

    meta_file = metadata_path(path)
    try:
        metadata = json.loads(meta_file.read_text()) if meta_file.exists() else {}
    except (OSError, ValueError):
        metadata = {}
    model = HighLowModel(
        models=models, weights=weights, feature_cols=list(blob["feature_cols"]),
        metadata=metadata, target=target,
    )
    return {
        "kind": "highlow2",
        "model": model,
        "metadata": metadata,
        "daily_models": sorted(models),
        "opening_minutes": OPENING_MINUTES,
        "opening_feed": OPENING_FEED,
        "min_bars": MIN_BARS_PER_SESSION,
        "target": target.label,
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
    """The tape the bundle reads this morning from -- IEX for every one."""
    return str((bundle or {}).get("opening_feed") or OPENING_FEED)


def min_bars(bundle: "dict | None" = None) -> int:
    try:
        return int((bundle or {})["min_bars"])
    except (KeyError, TypeError, ValueError):
        return MIN_BARS_PER_SESSION


# --- the history: SIP sessions, IEX openings, the extended hours ---------------

# A kept session's row: `daily_bars`' columns plus the IEX opening summary
# (missing on a session IEX printed nothing in the window of), which is
# everything `panel_from` reads about a past session.
_ROLLUP_COLS = ["open", "high", "low", "close", "volume", "rv"]
_OPENING_COLS = [
    "open5", "high5", "low5", "close5", "volume5",
    "or_up", "or_down", "or_ret", "or_range", "or_close_pos", "or_bar_std", "or_up_bars",
    "or_bars_frac",
]
_SESSION_COLS = _ROLLUP_COLS + _OPENING_COLS


def _part_cols(part: str) -> "list[str]":
    return [f"{part}_{c}" for c in _SUMMARY_FIELDS]


@dataclass
class History:
    """What the cache knows about a stretch of days.

    `sessions` is one row per session SIP kept (`_SESSION_COLS`); `parts` is
    `PARTS` -> one row per date that printed in that window, half days and
    dropped sessions included; `dropped` the sessions SIP refused.
    """

    sessions: pd.DataFrame
    parts: "dict[str, pd.DataFrame]"
    dropped: list = field(default_factory=list)

    def before(self, day) -> "History":
        """The same, cut to the dates strictly before `day`."""
        day = pd.Timestamp(day)
        return History(
            self.sessions[self.sessions.index < day],
            {p: frame[frame.index < day] for p, frame in self.parts.items()},
            [d for d in self.dropped if pd.Timestamp(d) < day],
        )


def _empty(cols: "list[str]") -> pd.DataFrame:
    return pd.DataFrame(columns=cols, index=pd.DatetimeIndex([], name="date"), dtype=float)


def bars_frame(bars: "list[dict]") -> pd.DataFrame:
    """Alpaca `{"t","o","h","l","c","v"}` bars -> an exchange-local OHLCV frame,
    every hour the feed printed, sorted and de-duplicated."""
    if not bars:
        return pd.DataFrame(columns=OHLCV, index=pd.DatetimeIndex([], tz=market_hours.MARKET_TZ), dtype=float)
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
    return df[~df.index.duplicated(keep="first")].sort_index()


def stretch_from(sip: pd.DataFrame, iex: pd.DataFrame,
                 min_bars: int = MIN_BARS_PER_SESSION) -> History:
    """Everything the cache keeps about a stretch of days, from that stretch's
    SIP and IEX tapes (every hour, exchange-local OHLCV).

    Every step groups by date, so a stretch summarised in pieces gives the same
    rows as the whole of it at once -- which is what lets the cache grow a day
    at a time.
    """
    sessions, dropped = _empty(_SESSION_COLS), []
    rth = sip.between_time(RTH_START, RTH_END) if len(sip) else sip
    if len(rth):
        minute, report = clean_minute(rth[OHLCV], min_bars)
        dropped = list(report.index[~report["kept"]])
        if len(minute):
            daily = daily_bars(minute)
            iex_rth = iex.between_time(RTH_START, RTH_END) if len(iex) else iex
            thin = trim_to_sessions(iex_rth[OHLCV], daily.index) if len(iex_rth) else None
            opening = (
                opening_features(thin)[_OPENING_COLS] if thin is not None and len(thin)
                else _empty(_OPENING_COLS)
            )
            sessions = daily[_ROLLUP_COLS].join(opening, how="left")
    parts = part_summaries(
        extended_bars(sip[OHLCV], "pre"), extended_bars(iex[OHLCV], "pre"),
        extended_bars(sip[OHLCV], "post"),
    )
    return History(sessions, parts, dropped)


_history_lock = threading.Lock()


def history_path(symbol: str) -> Path:
    return HISTORY_DIR / f"{symbol.upper()}_sessions.json"


def _empty_cache(min_bars: int = MIN_BARS_PER_SESSION) -> dict:
    return {"layout": CACHE_LAYOUT, "min_bars": min_bars, "from": None, "through": None,
            "sessions": {}, "dropped": [], **{part: {} for part in PARTS}}


def _read_cache(symbol: str, min_bars: int = MIN_BARS_PER_SESSION) -> dict:
    """The symbol's cache, or an empty one if it is unreadable, in another
    layout, or cleaned at another bar count."""
    try:
        payload = json.loads(history_path(symbol).read_text())
    except (OSError, ValueError):
        return _empty_cache(min_bars)
    if payload.get("layout") != CACHE_LAYOUT or int(payload.get("min_bars", -1)) != min_bars:
        return _empty_cache(min_bars)
    for key in ("sessions", *PARTS):
        payload.setdefault(key, {})
    payload.setdefault("dropped", [])
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
            "the HighLow2 forecast needs Alpaca SIP and IEX minute history and no Alpaca "
            "credentials are available (sidebar connection, or ALPACA_API_KEY / "
            "ALPACA_SECRET in the environment)."
        )
    return key, secret


def _fetch_tape(symbol: str, start: datetime, end: datetime, key: str, secret: str,
                feed: str) -> pd.DataFrame:
    from .rest import fetch_bars_range

    return bars_frame(fetch_bars_range(symbol, "1Min", start, end, key, secret,
                                       feed=feed, adjustment="split"))


def _fetch_stretch(symbol: str, first: date, before: date, key: str, secret: str,
                   min_bars: int = MIN_BARS_PER_SESSION) -> History:
    """`stretch_from` for the calendar days in [first, before), straight from
    Alpaca: SIP and IEX, every hour, split-adjusted."""
    from .datalog import log_fetch

    tz = market_hours.MARKET_TZ
    start = datetime.combine(first, datetime.min.time(), tzinfo=tz)
    end = datetime.combine(before, datetime.min.time(), tzinfo=tz)
    sip = _fetch_tape(symbol, start, end, key, secret, "sip")
    iex = _fetch_tape(symbol, start, end, key, secret, OPENING_FEED)
    out = stretch_from(sip, iex, min_bars)
    log_fetch(
        "minute bars (HighLow2 history)",
        "Alpaca REST (SIP + IEX, extended hours, split-adjusted)",
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
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index), name="date")
    return frame.sort_index().reindex(columns=cols)


def _merge(cache: dict, fresh: History, seam: "str | None") -> bool:
    """Fold a fetch into the cache. False (and nothing merged) if the session
    both hold -- `seam` -- disagrees, which means a split rescaled the tape."""
    sessions = _rows(fresh.sessions)
    if seam and seam in cache["sessions"] and seam in sessions:
        if abs(sessions[seam]["close"] / cache["sessions"][seam]["close"] - 1.0) > 1e-6:
            return False
    cache["sessions"].update(sessions)
    for part in PARTS:
        cache[part].update(_rows(fresh.parts[part]))
    cache["dropped"] = sorted(
        set(cache["dropped"]) | {pd.Timestamp(d).strftime("%Y-%m-%d") for d in fresh.dropped}
    )
    return True


def _history_from_cache(cache: dict, lo: str, hi: str) -> History:
    return History(
        _frame(cache["sessions"], _SESSION_COLS, lo, hi),
        {part: _frame(cache[part], _part_cols(part), lo, hi) for part in PARTS},
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

    `highlow_model.history_frame`'s scheme: the cache remembers which calendar
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


# --- this morning --------------------------------------------------------------

@dataclass
class Morning:
    """What 9:35 knows about today: the IEX opening summary (one row), and
    today's pre-market pieces (`MORNING_PARTS`, a row each where that window
    printed).

    `settled` is False for a live window fetched before IEX had published its
    09:34 minute: the forecast is still made from it -- the notebook keeps a
    short window -- but not memoised, so a later ask reads the whole one.
    """

    opening: pd.DataFrame
    parts: "dict[str, pd.DataFrame]"
    settled: bool = True


def morning_from_bars(sip: pd.DataFrame, iex: pd.DataFrame, session_date,
                      n_minutes: int = OPENING_MINUTES) -> Morning:
    """Today's `Morning` from today's SIP and IEX bars, each cut at the 9:35
    line by the notebook's own rules: SIP to its 09:19 bar, IEX's pre-market
    from 09:20, and IEX's first `n_minutes` minutes. Bars past those lines are
    ignored, so handing it more of the day cannot leak the session."""
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
            "model reads the open from IEX only; the notebook has no row for such a session."
        )
    sip_pre = extended_bars(sip[OHLCV], "pre")
    iex_pre = extended_bars(iex[OHLCV], "pre")
    parts = part_summaries(sip_pre[sip_pre["date"] == day], iex_pre[iex_pre["date"] == day],
                           sip_pre.iloc[:0])
    return Morning(opening_features(thin, n_minutes)[_OPENING_COLS],
                   {part: parts[part] for part in MORNING_PARTS})


def _cached_morning(symbol: str, day: date, min_bars: int = MIN_BARS_PER_SESSION) -> "Morning | None":
    """Today's `Morning` out of the cache, when it already covers the day.

    True only of a replay (live, the cache ends yesterday). The cached pieces
    are the same functions over the same bars as `morning_from_bars`, cut at
    the same lines, so this is the fetch without the requests. None -- fetch
    it -- for a day the cache has not looked at, or one SIP later dropped (the
    cache keeps no opening for those).
    """
    iso = day.isoformat()
    with _history_lock:
        cache = _read_cache(symbol, min_bars)
    if not (cache.get("from") and cache.get("through")) or not cache["from"] <= iso <= cache["through"]:
        return None
    row = cache["sessions"].get(iso)
    if row is None or not np.isfinite(float(row.get("open5", np.nan))):
        return None
    hi = (day + timedelta(days=1)).isoformat()
    opening = _frame({iso: row}, _OPENING_COLS, iso, hi)
    return Morning(opening, {
        part: _frame(cache[part], _part_cols(part), iso, hi) for part in MORNING_PARTS
    })


# On a basic Alpaca plan SIP refuses the trailing 15 minutes, which is why the
# notebook reads it only to the 09:19 bar (`LAST_PUBLIC`): the 09:20 boundary is
# just old enough at 9:35. A forecast asked in that first moment waits out the
# remainder (a few seconds) rather than spend a request on a refusal, and a
# refusal anyway is re-asked a couple of times. Never in a replay: the morning is
# long past on the real clock.
_SIP_DELAY = timedelta(minutes=15, seconds=3)
_SIP_WAIT_MAX_SEC = 90
_SIP_RETRIES = 3
_SIP_RETRY_SEC = 5.0
# Live, the IEX window is fetched the moment the caller's 09:34 bar has closed,
# and Alpaca can publish that minute a moment later (`highlow_model`'s retry).
_WINDOW_RETRIES = 3
_WINDOW_RETRY_SEC = 2.0
_WINDOW_FRESH_SEC = 90


def _fetch_sip_premarket(symbol: str, start: datetime, end: datetime, key: str, secret: str) -> pd.DataFrame:
    import requests

    wait = (end + _SIP_DELAY - datetime.now(timezone.utc)).total_seconds()
    if 0 < wait <= _SIP_WAIT_MAX_SEC:
        time.sleep(wait)
    for attempt in range(_SIP_RETRIES):
        try:
            return _fetch_tape(symbol, start, end, key, secret, "sip")
        except requests.HTTPError as exc:
            refused = getattr(exc.response, "status_code", None) == 403
            fresh = datetime.now(timezone.utc) < end + _SIP_DELAY + timedelta(seconds=_SIP_WAIT_MAX_SEC)
            if not (refused and fresh) or attempt == _SIP_RETRIES - 1:
                raise
            time.sleep(_SIP_RETRY_SEC)
    raise AssertionError("unreachable")


def fetch_morning(symbol: str, session_date, n_minutes: int = OPENING_MINUTES,
                  key: "str | None" = None, secret: "str | None" = None) -> Morning:
    """Today's `Morning`, straight from Alpaca: SIP from 04:00 to the 09:19 bar
    and IEX from 09:20 to the end of the opening window."""
    key, secret = _credentials(key, secret)
    day = pd.Timestamp(session_date).date()
    tz = market_hours.MARKET_TZ
    pre_start = datetime(day.year, day.month, day.day, 4, 0, tzinfo=tz)
    sip_end = datetime(day.year, day.month, day.day, 9, 20, tzinfo=tz)
    window_end = datetime(day.year, day.month, day.day, 9, 30, tzinfo=tz) + timedelta(minutes=n_minutes)
    sip = _fetch_sip_premarket(symbol, pre_start, sip_end, key, secret)
    for attempt in range(_WINDOW_RETRIES):
        iex = _fetch_tape(symbol, sip_end, window_end, key, secret, OPENING_FEED)
        complete = len(iex) and iex.index[-1] >= window_end - timedelta(minutes=1)
        fresh = datetime.now(timezone.utc) < window_end + timedelta(seconds=_WINDOW_FRESH_SEC)
        if complete or not fresh or attempt == _WINDOW_RETRIES - 1:
            break
        time.sleep(_WINDOW_RETRY_SEC)
    morning = morning_from_bars(sip, iex, day, n_minutes)
    morning.settled = bool(complete) or not fresh
    return morning


# --- one session's forecast --------------------------------------------------

_forecast_cache: "dict[tuple, dict]" = {}
_FORECAST_CACHE_MAX = 512


def forecast_from(bundle: dict, history: History, morning: Morning, session_date) -> dict:
    """The day's predicted high and low from the history (`history_inputs`)
    and this morning (`fetch_morning`).

    Returns the keys `dayrange_model.forecast_session` returns --
    `{"pred_high", "pred_low", "prev_avg", "adr14_abs", "or_high", "or_low"}`
    in dollars -- so `apple_trader.DayRangeTrader` runs on it unchanged.
    Raises ValueError when the inputs cannot support a forecast.
    """
    day = pd.Timestamp(session_date).normalize()
    if day.tzinfo is not None:
        day = day.tz_localize(None)
    history = history.before(day)
    if len(history.sessions) < MIN_PRIOR_SESSIONS:
        raise ValueError(
            f"only {len(history.sessions)} complete SIP sessions of history before {day.date()}; "
            f"the HighLow2 forecast needs {MIN_PRIOR_SESSIONS} for its 126-day windows."
        )
    opening_today = morning.opening.copy()
    opening_today.index = pd.DatetimeIndex([day], name="date")
    # Today's daily row, which `daily_features` reads nothing out of but its
    # date: every statistic there is shifted a day, and the gap reads the IEX
    # `open5` rather than this row's open. Filled from the opening window so
    # the frame holds nothing the session has not printed by 9:35.
    o = opening_today.iloc[0]
    today_daily = pd.DataFrame(
        {"open": o["open5"], "high": o["high5"], "low": o["low5"], "close": o["close5"],
         "volume": o["volume5"], "rv": np.nan},
        index=opening_today.index,
    )

    def stacked(frames: "list[pd.DataFrame]") -> pd.DataFrame:
        return pd.concat([f for f in frames if len(f)] or frames[:1])

    sessions = history.sessions
    daily = stacked([sessions[_ROLLUP_COLS], today_daily])
    # A past session IEX printed nothing in keeps its daily row but leaves the
    # panel, as the notebook's inner join leaves it out.
    opening = stacked([sessions.loc[sessions["open5"].notna(), _OPENING_COLS], opening_today])
    parts = {
        part: stacked([history.parts[part]] + ([morning.parts[part]] if part in morning.parts else []))
        for part in PARTS
    }
    summary = session_summary_from(parts["pm"], parts["pml"], parts["iexpm"], parts["ah"])
    panel = panel_from(daily, opening, summary)
    row = panel.loc[[day]]

    model = bundle["model"]
    missing = [
        c for c in model.feature_cols
        if c not in NAN_OK and not np.isfinite(float(row[c].iloc[0]))
    ]
    if missing:
        raise ValueError(
            "the feature row is incomplete "
            f"({', '.join(missing[:4])}{'…' if len(missing) > 4 else ''}); "
            "the SIP or IEX history is too short or has gaps."
        )
    pred = model.predict_prices(row).iloc[0]
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
    """Stretch the history cache `forecast_session` reads over the sessions
    before `before`."""
    history_inputs(ticker, before, key, secret, min_bars(bundle))


def forecast_session(
    bundle: dict,
    ticker: str,
    opening_bars: pd.DataFrame,
    session_date,
    key: "str | None" = None,
    secret: "str | None" = None,
) -> dict:
    """`forecast_from` with the history and this morning fetched for `ticker`.

    `opening_bars` -- the caller's first bars, on whatever tape it trades --
    only has to show that the opening window has closed: every input is read
    from Alpaca's SIP and IEX tapes here (or the cache of them), so a replay of
    any dataset forecasts from what a live run would have seen at 9:35.

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
    memo = (bundle.get("path"), str(ticker).upper(), day)
    cached = _forecast_cache.get(memo)
    if cached is not None:
        return dict(cached)
    history = history_inputs(ticker, day, key, secret, min_bars(bundle))
    morning = None
    if want == OPENING_MINUTES:
        morning = _cached_morning(str(ticker).upper(), day, min_bars(bundle))
    if morning is None:
        morning = fetch_morning(ticker, day, want, key, secret)
    out = forecast_from(bundle, history, morning, day)
    if morning.settled:
        if len(_forecast_cache) >= _FORECAST_CACHE_MAX:
            _forecast_cache.clear()
        _forecast_cache[memo] = dict(out)
    return out
