"""Where the high and low of what is left of the window will land (HighLow_1h, HighLow_4h).

Two FinNotebooks projects ask HighLow2_5m's question again later in the day,
every five minutes, about the stretch still to come:

`1h`  HighLow_1h: from 10:30 (one hour of tape seen) to 13:25, the high and low
      of the SIP bars from now to 13:30 (`highlow_1h_<TICKER>.*`).
`4h`  HighLow_4h: from 13:30 (four hours seen) to 15:55, the high and low from
      now to the 16:00 close (`highlow_4h_<TICKER>.*`).

Both forecast the excursions from the live price in units of what the time
left usually allows:

    scale = adr14 * sqrt(minutes_left / 390)
    up    = log(win_high / price) / scale
    down  = log(price / win_low)  / scale

    pred_high = price * exp(max(up_hat, 0) * scale)
    pred_low  = price * exp(-max(down_hat, 0) * scale)

with `price` the last print a trader on a basic Alpaca plan can see at minute
m: SIP is 15 minutes late, so bars older than that are SIP's and the last 15
IEX's (the "live tape"). LightGBM, an L1 pair, three seeds averaged. The 4h
model's trees learn a correction to the model-free forecast (the median past
excursion at this minute, `offset`), and read last night's option table
(`highlow2_options`) and, on an FOMC afternoon, the volatility past FOMC
afternoons brought (`rem_vol_hist`).

On their 2026 holdouts both are better than HighLow2_5m read for the window
(-74% / -75% error) and a little better than the model-free forecast (-2.6% /
-2.7%); the 4h model is 1.9% better than the 1h model asked about the
afternoon. Their own trading rules did not survive their holdouts.

`highlow_chain_model` strings them behind HighLow2's 9:35 forecast for Apple
Trader.

The mirror contract
-------------------
The `--- data`, `--- features` and `--- models` blocks are verbatim copies of
the parts of `highlow1h` / `highlow4h`'s `data.py`, `features.py` and
`models.py` the shipped models read, with the cuts each names: the feature
groups neither shipped model reads (volume, profile, flow, analysts, market,
calendar) and the training half of `models.py`. If either package changes,
retrain **and** update this module. `tests/test_highlow_chain.py` re-forecasts
the notebooks' own sessions from HighLow2_5m's processed files and pins the
result. `_build_bundle` refuses a bundle reading a feature this module does
not compute.

The pickles are stamped `highlow1h.models` / `highlow4h.models`, so
`_register_unpickle_alias` installs stubs (the real packages win if
importable).

What the live path has to supply
--------------------------------
* the SIP daily rollups of the ~150 sessions before the day -- HighLow2_5m's
  daily table, which `highlow2_model.history_inputs` keeps;
* SIP and IEX regular-session minute bars of the `HIST_SESSIONS` sessions
  before it (the 60-session median excursions, the 14-session volatility
  curve) -- the tape cache `fastfurious_model.raw_history` keeps for
  Fast&Furious and GoodMoments;
* today's: in a replay the whole session from that cache, each decision cut at
  SIP < m - 15 and IEX < m by the notebook's own code; live, SIP to 15 minutes
  ago and IEX to the last closed bar (`fastfurious_model.live_tapes`);
* today's SIP pre-market, for the 1h model's two-hour momentum before 11:31;
* for the 4h model, last night's HighLow2_5m option table, and on an FOMC day
  every FOMC session since 2024 (`_event_history`, cached for good).

Everything is read strictly before the session date the caller names, so a
replay reads what a live run that day would have.
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

# LightGBM loads its own OpenMP runtime first (see `highlow2_model`).
os.environ.setdefault("OMP_NUM_THREADS", "1")

try:  # pragma: no cover - depends on which optional extras are installed
    importlib.import_module("lightgbm")
except ImportError:
    pass

from . import market_hours, model_store  # noqa: E402
from .model_store import ModelStore  # noqa: E402
from .newsimpact_model import early_close  # noqa: E402

DEFAULT_TICKER = model_store.DEFAULT_TICKER

# --- the questions (highlow1h.config, highlow4h.config) ------------------------------

SESSION_MINUTES = 390
SIP_DELAY = 15        # at minute m the trader sees SIP bars < m - 15 and IEX bars < m
DECISION_STEP = 5
MARKET_TZ = market_hours.MARKET_TZ


@dataclass(frozen=True)
class Head:
    """One of the two window models."""

    key: str
    label: str            # the notebook's name
    first: int            # first decision, minutes after 9:30
    end: int              # the window's end: bars m .. end - 1
    filename: str
    env_key: str
    package: str          # what the pickle is stamped with

    @property
    def decisions(self) -> "tuple[int, ...]":
        return tuple(range(self.first, self.end, DECISION_STEP))


HEADS: "dict[str, Head]" = {
    "1h": Head("1h", "HighLow_1h", 60, 240, "highlow_1h_{ticker}.joblib",
               "APPLE_HIGHLOW_1H_MODEL", "highlow1h"),
    "4h": Head("4h", "HighLow_4h", 240, 390, "highlow_4h_{ticker}.joblib",
               "APPLE_HIGHLOW_4H_MODEL", "highlow4h"),
}


def head_at(minute: int) -> "Head | None":
    """The model whose decision grid holds `minute` (minutes after 9:30)."""
    for head in HEADS.values():
        if minute in head.decisions:
            return head
    return None


# Sessions of minute tape read before the day: the 60-session median past
# excursions (`_past_excursions`) and the 14-session volatility curve
# (`_history`), with room to spare.
HIST_SESSIONS = 70
HIST_CALENDAR_DAYS = 120
# Daily history before the day: 126-day momentum reads `mid.shift(126)` of a
# series itself shifted a day.
MIN_PRIOR_SESSIONS = 127

# What the 4h model reads on an FOMC day (`event_history`): every previous
# FOMC session in the notebook's history, which starts here.
EVENT_HISTORY_START = "2024-01-01"
# The FOMC statements the calendar (`event_days.CALENDAR_PATH`, the notebook's
# copy) does not list yet: it ends with the notebook's traded week. The Fed's
# published 2026 schedule; the calendar file is left alone because tuning
# fingerprints it.
FOMC_AFTER_CALENDAR = ("2026-10-28", "2026-12-09")

HISTORY_DIR = Path(__file__).resolve().parent.parent / "data" / "highlow_window"
CACHE_LAYOUT = 1


# --- data (mirrors highlow1h.data / highlow4h.data) --------------------------------------
#
# Cut: the loaders (replaced by the caches below) and the vwap and trade-count
# arrays, which only unshipped groups read.

PRE_MINUTES = 330  # 04:00 - 09:29


@dataclass
class Tape:
    """One feed's regular-hours bars as (n_days, 390) arrays; NaN where no bar printed."""

    dates: pd.DatetimeIndex
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    close_ff: np.ndarray = field(init=False)   # last print at or before each minute
    last_idx: np.ndarray = field(init=False)   # minute of that print, -1 before the first

    def __post_init__(self):
        idx = np.where(np.isnan(self.close), -1, np.arange(self.close.shape[1])[None, :])
        self.last_idx = np.maximum.accumulate(idx, axis=1)
        rows = np.arange(len(self.dates))[:, None]
        self.close_ff = np.where(self.last_idx >= 0, self.close[rows, np.maximum(self.last_idx, 0)], np.nan)


def to_tape(minute: pd.DataFrame, dates: pd.DatetimeIndex, n_minutes: int = SESSION_MINUTES) -> Tape:
    m = minute[(minute["minute"] >= 0) & (minute["minute"] < n_minutes)]
    row = dates.get_indexer(pd.DatetimeIndex(m["date"]))
    keep = row >= 0
    row, col = row[keep], m["minute"].to_numpy()[keep].astype(int)

    def grid(values) -> np.ndarray:
        out = np.full((len(dates), n_minutes), np.nan)
        out[row, col] = np.asarray(values, dtype=float)[keep]
        return out

    return Tape(dates, grid(m["open"]), grid(m["high"]), grid(m["low"]), grid(m["close"]))


def premarket_closes(pre: "pd.DataFrame | None", dates: pd.DatetimeIndex) -> np.ndarray:
    """Pre-market SIP closes forward-filled over 04:00-09:29, (n_days, 330)."""
    out = np.full((len(dates), PRE_MINUTES), np.nan)
    if pre is None or pre.empty:
        return out
    row = dates.get_indexer(pd.DatetimeIndex(pre["date"]))
    keep = row >= 0
    out[row[keep], pre["pm"].to_numpy()[keep]] = pre["close"].to_numpy()[keep]
    return pd.DataFrame(out).ffill(axis=1).to_numpy()


def premarket_frame(bars: pd.DataFrame) -> pd.DataFrame:
    """`load_premarket`'s frame from exchange-local SIP bars: 04:00-09:29, `pm` =
    minutes since 04:00."""
    pre = bars.copy()
    ts = pre.index.tz_convert(MARKET_TZ)
    pre["date"] = ts.tz_localize(None).normalize()
    pre["pm"] = (ts.hour * 60 + ts.minute - 240).astype(np.int16)
    return pre[(pre["pm"] >= 0) & (pre["pm"] < PRE_MINUTES)]


# --- features (mirrors highlow1h.features / highlow4h.features) ---------------------------
#
# Cut: the volume, profile, flow, analysts, market and calendar groups, the
# options group's columns neither model reads (`gex_at_price`, the skew, the
# volume norm, the expiry flags and pinning), the morning-window training rows
# and the leakage checks. 1h and 4h differ in `end` (240 / 390) and in the
# volatility still to come: the 4h model's reads past FOMC afternoons on an
# FOMC day (`event_history`).

EPS = 1e-12
SQRT_YEAR = np.sqrt(252.0)
RANGE_PER_SD = 1.6                 # expected daily range / daily sd for a random walk (Parkinson)
MOM_MINUTES = (1, 5, 15, 60, 120)  # intraday momentum horizons
HIST_DAYS = 14                     # "typical for this time of day" = mean of the last 14 sessions
RECENT = 30                        # minutes in the "recent" SIP window
HIST_EXCURSION_DAYS = 60           # sessions behind the median past excursions
MIN_EVENTS = 3                     # past event sessions needed before they replace the 14-day history
TARGETS = ["up", "down"]


def _pos(value, low, high):
    """Where `value` sits inside [low, high]; 0.5 for an empty range."""
    span = high - low
    return np.where(span > EPS, (value - low) / np.where(span > EPS, span, 1.0), 0.5)


def _coalesce(*arrays):
    out = np.array(arrays[0], dtype=float, copy=True)
    for a in arrays[1:]:
        out = np.where(np.isnan(out), a, out)
    return out


def _ratio(num, den):
    return np.where(np.abs(den) > EPS, num / np.where(np.abs(den) > EPS, den, 1.0), np.nan)


def _window(cum: np.ndarray, a: int, b: int) -> np.ndarray:
    """Sum over minutes [a, b) from a cumulative-sum array."""
    return cum[:, b - 1] - (cum[:, a - 1] if a > 0 else 0.0)


def _history(cum: np.ndarray, dates: pd.DatetimeIndex) -> np.ndarray:
    """Mean of the previous HIST_DAYS sessions, minute by minute (today excluded)."""
    return pd.DataFrame(cum, index=dates).rolling(HIST_DAYS, min_periods=5).mean().shift(1).to_numpy()


def daily_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Levels and statistics from sessions before d. Averages use (open + close) / 2."""
    o, h, l, c, v, rv = (daily[k] for k in ["open", "high", "low", "close", "volume", "rv"])
    mid = (o + c) / 2
    rng = np.log(h / l)
    f = pd.DataFrame(index=daily.index)
    f["prev_open"], f["prev_high"], f["prev_low"], f["prev_close"] = o.shift(), h.shift(), l.shift(), c.shift()
    f["prev_range"] = rng.shift()
    f["prev_rv"] = rv.shift()
    f["prev_ret"] = np.log(c / c.shift()).shift()
    ret = np.log(mid / mid.shift())
    for w in (7, 14, 28, 63):
        f[f"avg{w}"] = mid.rolling(w).mean().shift()
        f[f"adr{w}"] = rng.rolling(w).mean().shift()
        f[f"rv{w}"] = rv.rolling(w).mean().shift()
        f[f"vol{w}"] = ret.rolling(w).std().shift()
    for w in (5, 28, 63, 126):
        f[f"mom{w}"] = np.log(mid / mid.shift(w)).shift()
    f["hi126"] = h.rolling(126, min_periods=40).max().shift()
    f["lo126"] = l.rolling(126, min_periods=40).min().shift()
    f["adr14_usd"] = (h - l).rolling(14).mean().shift()
    return f


@dataclass
class LiveTape:
    """SIP + IEX + pre-market for one ticker, with the arrays every minute reuses."""

    sip: Tape
    iex: Tape
    pre: np.ndarray           # pre-market closes, forward filled
    prev_close: np.ndarray

    def __post_init__(self):
        s, x = self.sip, self.iex
        self.sip_hi = np.fmax.accumulate(s.high, axis=1)
        self.sip_lo = np.fmin.accumulate(s.low, axis=1)
        self.c_var = np.cumsum(self._returns(s) ** 2, axis=1)
        self.ci_var = np.cumsum(self._returns(x) ** 2, axis=1)

    @staticmethod
    def _returns(t: Tape) -> np.ndarray:
        """1-minute log returns on the forward-filled closes (`_flows`' first half)."""
        lc = np.log(t.close_ff)
        r = np.diff(lc, axis=1, prepend=np.nan)
        r[:, 0] = np.log(t.close[:, 0] / t.open[:, 0])
        return np.nan_to_num(r)

    # ---- what the trader sees at minute m ----
    def seen_close(self, m: int, j: int) -> np.ndarray:
        """Latest visible price at the end of bar j, as seen at minute m (j < m)."""
        k = m - SIP_DELAY
        fallback = _coalesce(self.pre[:, -1], self.prev_close)
        if j >= k:
            sip_px = _coalesce(self.sip.close_ff[:, k - 1], fallback)
            return np.where(self.iex.last_idx[:, j] >= k, self.iex.close_ff[:, j], sip_px)
        if j >= 0:
            return _coalesce(self.sip.close_ff[:, j], fallback)
        pm = self.pre[:, PRE_MINUTES + j] if PRE_MINUTES + j >= 0 else np.full(len(self.prev_close), np.nan)
        return _coalesce(pm, self.prev_close)

    def extremes_so_far(self, m: int, price: np.ndarray):
        k = m - SIP_DELAY
        hi = np.fmax(self.sip_hi[:, k - 1], np.fmax.reduce(self.iex.high[:, k:m], axis=1))
        lo = np.fmin(self.sip_lo[:, k - 1], np.fmin.reduce(self.iex.low[:, k:m], axis=1))
        return np.fmax(hi, price), np.fmin(lo, price)

    def _seen_bars(self, m: int, a: int = 0):
        """Highs and lows of bars a .. m-1 as seen live: SIP before m-15, IEX after."""
        k = m - SIP_DELAY
        hi = np.concatenate([self.sip.high[:, a:k], self.iex.high[:, k:m]], axis=1)
        lo = np.concatenate([self.sip.low[:, a:k], self.iex.low[:, k:m]], axis=1)
        return hi, lo

    def extreme_ages(self, m: int):
        """Minutes since today's high and low so far printed, as a share of the minutes traded."""
        hi, lo = self._seen_bars(m)
        i_hi = np.argmax(np.where(np.isnan(hi), -np.inf, hi), axis=1)
        i_lo = np.argmin(np.where(np.isnan(lo), np.inf, lo), axis=1)
        return (m - i_hi) / m, (m - i_lo) / m


def event_history(c_var: np.ndarray, hist: np.ndarray, adr: np.ndarray, dates: pd.DatetimeIndex,
                  events: pd.DatetimeIndex) -> np.ndarray:
    """`hist`, except on event sessions: the mean cumulative variance of the previous event sessions.

    Each past event session is measured in its own ADRs and rescaled to today's, so a
    calm-regime FOMC day and a volatile-regime one are comparable. Only sessions that
    finished before d enter d's row; with fewer than MIN_EVENTS of them `hist` stays.
    """
    out = hist.copy()
    norm = c_var / adr[:, None] ** 2
    idx = np.flatnonzero(dates.isin(events))
    for j, i in enumerate(idx):
        prev = idx[:j]
        prev = prev[np.isfinite(norm[prev, -1])]
        if len(prev) >= MIN_EVENTS and np.isfinite(adr[i]):
            out[i] = norm[prev].mean(axis=0) * adr[i] ** 2
    return out


@dataclass
class _Context:
    dates: pd.DatetimeIndex
    day: pd.DataFrame
    tape: LiveTape
    win: "dict[int, tuple[np.ndarray, np.ndarray]]"   # window end -> (win_hi, win_lo)
    hist_var: np.ndarray
    hist_var_event: np.ndarray
    opt: "pd.DataFrame | None"   # last night's positioning row per session (today's only)


def _context(daily: pd.DataFrame, sip: pd.DataFrame, iex: pd.DataFrame, pre: "pd.DataFrame | None",
             dates: pd.DatetimeIndex, events: "tuple[pd.DatetimeIndex, np.ndarray, np.ndarray] | None",
             opt: "pd.DataFrame | None") -> _Context:
    """The notebook's `_context` over `dates` (sessions with their tape, the day
    last). `daily` is the daily table over every session before them too.

    `events` is the event history the notebook's `event_history` reads -- every
    FOMC session since EVENT_HISTORY_START, `(dates, c_var rows, adr)` -- of
    which only the ones before the last date matter: earlier sessions' rows are
    read for their excursions, never for the event curve.
    """
    day = daily_features(daily).reindex(dates)
    tape = LiveTape(to_tape(sip, dates), to_tape(iex, dates),
                    premarket_closes(pre, dates), day["prev_close"].to_numpy())
    s = tape.sip
    win = {}
    for end in sorted({h.end for h in HEADS.values()}):
        win[end] = (np.fmax.accumulate(s.high[:, :end][:, ::-1], axis=1)[:, ::-1],
                    np.fmin.accumulate(s.low[:, :end][:, ::-1], axis=1)[:, ::-1])
    hv = _history(tape.c_var, dates)
    hve = hv
    if events is not None:
        # Only the last session's row is forecast. The notebook's arrays reach
        # back to 2024; here the event sessions before it are stacked in front
        # of its own row, in date order, and its row is read back off the end.
        ev_dates, ev_cvar, ev_adr = events
        stacked = event_history(
            np.vstack([ev_cvar, tape.c_var[-1:]]),
            np.vstack([np.full_like(ev_cvar, np.nan), hv[-1:]]),
            np.concatenate([ev_adr, day["adr14"].to_numpy()[-1:]]),
            ev_dates.append(dates[-1:]),
            ev_dates.append(dates[-1:]),
        )
        hve = hv.copy()
        hve[-1] = stacked[-1]
    # One option row per session, as `previous_row` gives the notebook's; the
    # earlier sessions are read for their excursions only, which need none.
    return _Context(dates, day, tape, win, hv, hve, None if opt is None else opt.reindex(dates))


def _mom(t: LiveTape, m: int, price: np.ndarray, adr: np.ndarray, h: int) -> np.ndarray:
    ref = t.seen_close(m, m - 1 - h)
    return np.log(price / ref) / (adr * np.sqrt(h / SESSION_MINUTES))


def minute_frame(ctx: _Context, m: int, head: Head) -> pd.DataFrame:
    """Every feature the head reads, its targets and reference prices for a
    forecast at minute m, one row per session."""
    t, d = ctx.tape, ctx.day
    k = m - SIP_DELAY
    price = t.seen_close(m, m - 1)
    adr = d["adr14"].to_numpy()
    hi, lo = t.extremes_so_far(m, price)
    end = head.end
    left = end - m
    win_hi, win_lo = ctx.win[end]
    f: "dict[str, np.ndarray]" = {}

    def dist(level, above=False):
        """log distance from price to a level, in ADRs; positive when the level is above (above=True)."""
        x = np.log(level / price) if above else np.log(price / level)
        return x / adr

    # --- meta and targets ---
    scale = adr * np.sqrt(left / SESSION_MINUTES)
    up_log, down_log = np.log(win_hi[:, m] / price), np.log(price / win_lo[:, m])
    f.update(price=price, adr=adr, adr_usd=d["adr14_usd"].to_numpy(), scale=scale,
             hi_sofar=hi, lo_sofar=lo, win_high=win_hi[:, m], win_low=win_lo[:, m],
             minutes_left=np.full(len(price), float(left)))
    f["session_progress"] = np.full(len(price), m / SESSION_MINUTES)
    f["up"] = up_log / scale
    f["down"] = down_log / scale

    # --- core ---
    f["dist_prev_close"] = dist(d["prev_close"])
    f["dist_prev_open"] = dist(d["prev_open"])
    f["dist_prev_high"] = dist(d["prev_high"], above=True)
    f["dist_prev_low"] = dist(d["prev_low"])
    for w in (7, 14, 28):
        f[f"dist_avg{w}"] = dist(d[f"avg{w}"])
    f["dist_hi126"] = dist(d["hi126"], above=True)
    f["dist_lo126"] = dist(d["lo126"])
    f["dist_hi"] = dist(hi, above=True)
    f["range_sofar"] = np.log(hi / lo) / adr
    f["range_vs_expected"] = np.log(hi / lo) / (adr * np.sqrt(m / SESSION_MINUTES))
    f["pos_range"] = _pos(price, lo, hi)
    f["hi_age"], _ = t.extreme_ages(m)
    # the nearest references again, in units of the move still possible (the target's own scale)
    to_t = adr / scale
    for c in ("dist_hi", "dist_prev_close", "dist_prev_high", "dist_prev_low"):
        f[f"{c}_t"] = f[c] * to_t
    for h in MOM_MINUTES:
        f[f"mom_{h}m"] = _mom(t, m, price, adr, h)
    for w in (5, 28, 126):
        f[f"mom{w}d"] = d[f"mom{w}"] / (adr * np.sqrt(w))
    f["prev_ret"] = d["prev_ret"] / adr
    f["prev_range"] = d["prev_range"] / adr
    f["prev_rv"] = d["prev_rv"] / adr
    f["adr14"] = adr
    f["adr63_ratio"] = np.log(adr / d["adr63"])
    f["rv14_adr"] = d["rv14"] / adr
    f["vol14_adr"] = d["vol14"] / adr
    f["rv_trend"] = np.log(d["rv7"] / d["rv28"])
    hv = ctx.hist_var
    # today's realised volatility against the same minutes of the last 14 sessions
    f["rv_today_rel"] = np.sqrt(_ratio(t.c_var[:, k - 1], hv[:, k - 1]))
    f["rv_recent_rel"] = np.sqrt(_ratio(_window(t.c_var, k - RECENT, k), _window(hv, k - RECENT, k)))
    if k - 120 >= 0:   # the 4h model's; at the 1h model's minutes it does not exist
        f["rv_2h_rel"] = np.sqrt(_ratio(_window(t.c_var, k - 120, k), _window(hv, k - 120, k)))
    f["rv_live_rel"] = np.sqrt(_ratio(_window(t.ci_var, k, m), _window(hv, k, m)))
    # typical volatility still to come: last 14 sessions, or (4h) past event sessions on an event day
    hve = ctx.hist_var_event if head.key == "4h" else hv
    f["rem_vol_hist"] = np.sqrt(_window(hve, m, end)) / scale

    # --- options: yesterday's positioning, measured from the live price ---
    n = len(price)
    if ctx.opt is not None:
        p = ctx.opt
        f["call_wall"] = dist(p["call_wall_oi"], above=True)
        f["put_wall"] = dist(p["put_wall_oi"])
        f["call_wall7"] = dist(p["call_wall_oi_7"], above=True)
        f["put_wall7"] = dist(p["put_wall_oi_7"])
        f["call_gwall"] = dist(p["call_wall_gex"], above=True)
        f["put_gwall"] = dist(p["put_wall_gex"])
        f["gamma_flip"] = dist(p["gamma_flip"])
        f["max_pain"] = dist(p["max_pain_std"])
        f["gex_ratio_close"] = p["gex_ratio"].to_numpy()
        f["iv1d_adr"] = p["iv_1d"] / SQRT_YEAR / adr
        f["iv5d_adr"] = p["iv_5d"] / SQRT_YEAR / adr
        # the day's implied range not yet used up, in ADRs
        f["iv_left"] = RANGE_PER_SD * f["iv1d_adr"] - f["range_sofar"]
        f["iv_term"] = np.log(p["iv_1d"] / p["atm_iv_30"])
        f["pcr_oi"] = np.log((p["put_oi"] + 1) / (p["call_oi"] + 1))
        f["pcr_volume"] = np.log((p["put_volume"] + 1) / (p["call_volume"] + 1))
        f["days_to_opex"] = p["days_to_monthly_opex"].astype(float)
        thin = (p["opt_volume_coverage"] < 0.4).to_numpy()   # partial option tape: activity numbers unusable
        for col in ("pcr_volume", "gex_ratio_close"):
            f[col] = np.where(thin, np.nan, np.asarray(f[col], dtype=float))

    nan = np.full(n, np.nan)
    cols = {k_: np.asarray(v, dtype=float) for k_, v in f.items()}
    cols.update({c: nan for c in FEATURE_COLS if c not in cols})   # no option table for this session
    cols["minute"] = np.full(n, m)
    return pd.DataFrame(cols, index=ctx.dates)


def _past_excursions(panel: pd.DataFrame, lookback: int = HIST_EXCURSION_DAYS) -> pd.DataFrame:
    """Median up / down at the same minute over the previous `lookback` sessions."""
    out = {}
    for t in TARGETS:
        wide = panel[t].unstack("minute")
        out[f"hist_{t}"] = wide.rolling(lookback, min_periods=20).median().shift(1).stack()
    return pd.DataFrame(out)


# Every feature this module computes -- the union of what the two shipped
# models read. `_build_bundle` refuses a bundle reading anything else.
OPTION_COLS = [
    "call_wall", "put_wall", "call_wall7", "put_wall7", "call_gwall", "put_gwall", "gamma_flip",
    "max_pain", "gex_ratio_close", "iv1d_adr", "iv5d_adr", "iv_left", "iv_term", "pcr_oi",
    "pcr_volume", "days_to_opex",
]
FEATURE_COLS = [
    "minutes_left", "session_progress", "dist_prev_close", "dist_prev_open", "dist_prev_high",
    "dist_prev_low", "dist_avg7", "dist_avg14", "dist_avg28", "dist_hi126", "dist_lo126", "dist_hi",
    "range_sofar", "range_vs_expected", "pos_range", "hi_age", "dist_hi_t", "dist_prev_close_t",
    "dist_prev_high_t", "dist_prev_low_t", "mom_1m", "mom_5m", "mom_15m", "mom_60m", "mom_120m",
    "mom5d", "mom28d", "mom126d", "prev_ret", "prev_range", "prev_rv", "adr14", "adr63_ratio",
    "rv14_adr", "vol14_adr", "rv_trend", "rv_today_rel", "rv_recent_rel", "rv_2h_rel",
    "rv_live_rel", "rem_vol_hist", "hist_up", "hist_down", *OPTION_COLS,
]


def build_panel(ctx: _Context, head: Head, decisions) -> pd.DataFrame:
    """`build_panel` over the context's sessions: one row per (session, decision
    minute), features in float32. (Cut: the filter on rows with a target, which
    the day being forecast has none of yet, and the shock flags.)"""
    # The other head's columns are computed too, and some of them divide by
    # nothing at this head's minutes; neither model reads those.
    with np.errstate(invalid="ignore", divide="ignore"):
        frames = [minute_frame(ctx, m, head) for m in decisions]
    panel = pd.concat(frames).rename_axis("date").set_index("minute", append=True).sort_index()
    panel = panel.drop(columns=[c for c in ("hist_up", "hist_down") if c in panel]).join(_past_excursions(panel))
    return panel.astype({c: np.float32 for c in FEATURE_COLS})


# --- models (mirrors the inference half of highlow1h.models / highlow4h.models) ---------------


def to_prices(up: np.ndarray, down: np.ndarray, frame: pd.DataFrame, scale_col: str = "scale") -> pd.DataFrame:
    """(up, down) in scale units -> dollar high and low, never on the wrong side of the price."""
    price, s = frame["price"].to_numpy(), frame[scale_col].to_numpy()
    return pd.DataFrame({"pred_high": price * np.exp(np.clip(up, 0, None) * s),
                         "pred_low": price * np.exp(-np.clip(down, 0, None) * s)}, index=frame.index)


class WindowModel:
    """One LightGBM per target and seed, averaged (unpickled). The 1h model has
    no `scale_col` / `offset`: its scale is `scale` and its trees the excursion
    itself."""

    feature_cols: list
    models: dict

    def _base(self, frame: pd.DataFrame, t: str) -> np.ndarray:
        """The model-free part of the forecast, in this model's scale (0 without offset)."""
        if not getattr(self, "offset", False):
            return np.zeros(len(frame))
        b = (frame[f"hist_{t}"] * frame["scale"] / frame[self.scale_col]).to_numpy(dtype=float)
        return np.where(np.isfinite(b), b, self.offset_fill.get(t, np.nan))

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """(up, down) in this model's scale."""
        X = frame[self.feature_cols]
        return np.column_stack([np.mean([m.predict(X) for m in self.models[t]], axis=0) + self._base(frame, t)
                                for t in TARGETS])

    def predict_prices(self, frame: pd.DataFrame) -> pd.DataFrame:
        pred = self.predict(frame)
        return to_prices(pred[:, 0], pred[:, 1], frame, getattr(self, "scale_col", "scale"))


class Bundle:
    """The model and the notebook's metadata (unpickled)."""

    model: WindowModel
    metadata: dict


# --- the saved bundles -----------------------------------------------------------


def _register_unpickle_alias(head: Head) -> None:
    """Make `<package>.models.{Bundle,WindowModel}` resolvable for joblib.

    The real packages pull the notebooks' data paths and SciPy in, so a stub
    pointing at the mirrors above stands in -- unless the real package is
    genuinely importable, in which case it wins.
    """
    name = f"{head.package}.models"
    if name in sys.modules:
        return
    try:
        __import__(name)
        return
    except Exception:
        sys.modules.pop(head.package, None)
    package = types.ModuleType(head.package)
    package.__path__ = []
    module = types.ModuleType(name)
    module.Bundle = Bundle
    module.WindowModel = WindowModel
    package.models = module
    sys.modules[head.package] = package
    sys.modules[name] = module


def _build_bundle(path: Path, head: Head) -> "dict | None":
    """One ticker's saved window model plus its metadata, or None when it
    cannot be assembled -- including a model reading a feature this module
    does not compute, or a scale it cannot decode."""
    try:
        import joblib
    except ImportError:
        return None
    _register_unpickle_alias(head)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            blob = joblib.load(path)
    except (OSError, ValueError, KeyError, ModuleNotFoundError, AttributeError, ImportError, EOFError):
        return None
    model = getattr(blob, "model", None)
    cols = list(getattr(model, "feature_cols", None) or [])
    if not cols or set(cols) - set(FEATURE_COLS) or set(getattr(model, "models", {})) != set(TARGETS):
        return None
    if getattr(model, "scale_col", "scale") != "scale":
        return None
    meta_file = path.with_suffix(".json")
    try:
        metadata = json.loads(meta_file.read_text()) if meta_file.exists() else {}
    except (OSError, ValueError):
        metadata = getattr(blob, "metadata", None) or {}
    return {
        "kind": "highlow_window",
        "head": head.key,
        "model": model,
        "metadata": metadata,
        "feature_cols": cols,
        "reads_options": bool(set(cols) & set(OPTION_COLS)),
        "path": str(path),
        "ticker": str(metadata.get("ticker") or "").upper() or None,
        "fit_through": metadata.get("fit_through"),
    }


_STORES = {
    key: ModelStore(env_key=head.env_key, filename=head.filename,
                    build=lambda path, head=head: _build_bundle(path, head))
    for key, head in HEADS.items()
}


def model_path(ticker: "str | None" = None, head: str = "1h") -> Path:
    return _STORES[head].path(ticker)


def load_bundle(ticker: "str | None" = None, head: str = "1h") -> "dict | None":
    return _STORES[head].load(ticker)


def reset_bundle_cache() -> None:
    for store in _STORES.values():
        store.reset()


# --- the inputs --------------------------------------------------------------------


def _hl2():
    from . import highlow2_model

    return highlow2_model


def _ff():
    from . import fastfurious_model

    return fastfurious_model


def _credentials(key: "str | None", secret: "str | None") -> "tuple[str, str]":
    """The caller's Alpaca keys, else the environment's (SimLab has no sidebar)."""
    key = key or os.getenv("ALPACA_API_KEY", "")
    secret = secret or os.getenv("ALPACA_SECRET", "")
    if not (key and secret):
        raise ValueError(
            "the HighLow_1h / HighLow_4h forecasts need Alpaca SIP and IEX minute bars and no "
            "Alpaca credentials are available (sidebar connection, or ALPACA_API_KEY / "
            "ALPACA_SECRET in the environment)."
        )
    return key, secret


def _today_et() -> date:
    return datetime.now(MARKET_TZ).date()


def _minute_frame_of(raw: pd.DataFrame, sessions) -> pd.DataFrame:
    """Regular-session bars -> the notebook's minute table (`date`, `minute`),
    sane bars only, on `sessions` (HighLow2_5m's `trim_to_sessions`, which is
    what its SIP and IEX minute files hold for a kept session)."""
    hl2 = _hl2()
    if not len(raw):
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "date", "minute"])
    out = hl2.trim_to_sessions(raw[hl2.OHLCV], pd.DatetimeIndex(sessions).as_unit("ns"))
    out["date"] = pd.DatetimeIndex(out["date"]).as_unit("ns")
    return out


def fomc_dates() -> pd.DatetimeIndex:
    """The FOMC statement days the notebook's `event_dates` reads: the
    calendar's `fomc` rows known before the open, plus `FOMC_AFTER_CALENDAR`."""
    from .event_days import CALENDAR_PATH

    try:
        cal = pd.read_csv(CALENDAR_PATH, parse_dates=["date"])
        cal["known_before_open"] = cal["known_before_open"].astype(str).str.lower().eq("true")
        rows = cal[cal["scope"].isin(["ALL"]) & cal["category"].isin(["fomc"]) & cal["known_before_open"]]
        listed = list(rows["date"].dt.strftime("%Y-%m-%d"))
    except (OSError, ValueError, KeyError):
        listed = []
    days = sorted(set(listed) | set(FOMC_AFTER_CALENDAR))
    return pd.DatetimeIndex(pd.to_datetime(days), name="date").as_unit("ns")


# --- the event history: every FOMC session since 2024, kept for good -----------------

_event_lock = threading.Lock()
# Calendar days of SIP read before an event session for its 14-session ADR.
_EVENT_REACH_DAYS = 35


def _events_path(symbol: str) -> Path:
    return HISTORY_DIR / f"{symbol.upper()}_events.json"


def _event_row(symbol: str, day: pd.Timestamp, key: str, secret: str) -> "dict | None":
    """One FOMC session's cumulative variance curve and 14-session ADR, from
    Alpaca's SIP tape -- or None for a session HighLow2_5m does not keep."""
    hl2 = _hl2()
    tz = MARKET_TZ
    start = datetime.combine((day - pd.Timedelta(_EVENT_REACH_DAYS, "D")).date(), datetime.min.time(), tzinfo=tz)
    end = datetime.combine((day + pd.Timedelta(1, "D")).date(), datetime.min.time(), tzinfo=tz)
    sip = hl2._fetch_tape(symbol, start, end, key, secret, "sip")
    rth = sip.between_time(model_store.RTH_START, model_store.RTH_END) if len(sip) else sip
    if not len(rth):
        return None
    minute, report = hl2.clean_minute(rth[hl2.OHLCV])
    daily = hl2.daily_bars(minute)
    if day not in daily.index:
        return None
    adr = float(daily_features(daily).at[day, "adr14"])
    dates = pd.DatetimeIndex([day])
    tape = LiveTape(to_tape(minute[minute["date"] == day], dates), to_tape(minute.iloc[:0], dates),
                    np.full((1, PRE_MINUTES), np.nan), np.full(1, np.nan))
    return {"adr14": adr, "c_var": [float(v) for v in tape.c_var[0]]}


def _event_history(symbol: str, before: pd.Timestamp, key: "str | None",
                   secret: "str | None") -> "tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]":
    """`(dates, c_var rows, adr14)` of every FOMC session from
    EVENT_HISTORY_START to before `before`, fetched once each and kept: a
    finished session never changes. Its variance is in its own ADRs
    (`event_history` divides them out), so a split does not move it."""
    wanted = [d for d in fomc_dates() if pd.Timestamp(EVENT_HISTORY_START) <= d < before]
    with _event_lock:
        try:
            cache = json.loads(_events_path(symbol).read_text())
        except (OSError, ValueError):
            cache = {}
        if cache.get("layout") != CACHE_LAYOUT:
            cache = {"layout": CACHE_LAYOUT, "rows": {}}
        missing = [d for d in wanted if f"{d:%Y-%m-%d}" not in cache["rows"]]
        if missing:
            key, secret = _credentials(key, secret)
            for d in missing:
                cache["rows"][f"{d:%Y-%m-%d}"] = _event_row(symbol, d, key, secret)
            HISTORY_DIR.mkdir(parents=True, exist_ok=True)
            path = _events_path(symbol)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(cache))
            tmp.replace(path)
    rows = [(d, cache["rows"].get(f"{d:%Y-%m-%d}")) for d in wanted]
    rows = [(d, r) for d, r in rows if r]
    if not rows:
        return pd.DatetimeIndex([], name="date").as_unit("ns"), np.zeros((0, SESSION_MINUTES)), np.zeros(0)
    dates = pd.DatetimeIndex([d for d, _ in rows], name="date").as_unit("ns")
    return (dates, np.array([r["c_var"] for _, r in rows], dtype=float),
            np.array([r["adr14"] for _, r in rows], dtype=float))


# --- one session's forecasts ---------------------------------------------------------


@dataclass
class _Day:
    """What a session's forecasts read from before it, built once per session."""

    day: pd.Timestamp
    daily: pd.DataFrame         # every session's rollup before the day, plus a row for it
    dates: pd.DatetimeIndex     # the HIST_SESSIONS sessions with tape, plus the day
    sip: pd.DataFrame           # their minute tables
    iex: pd.DataFrame
    events: "tuple | None" = None
    opt: "pd.DataFrame | None" = None
    pre: "pd.DataFrame | None" = None
    problems: "dict[str, str]" = field(default_factory=dict)
    options_tried: float = 0.0   # monotonic time of the last option-table read


_day_lock = threading.Lock()
_days: "dict[tuple, _Day]" = {}
_DAYS_MAX = 32
_rows_memo: "dict[tuple, dict[int, dict]]" = {}
_problems_memo: "dict[tuple, dict[int, str]]" = {}
_ROWS_MEMO_MAX = 256
_OPTIONS_PROBLEM = "last night's option table"
# How long a failed option-table read waits before it is tried again.
_OPTIONS_RETRY_SEC = 300.0
# Option-table cells the notebook's own table leaves empty on ordinary days (no
# gamma flip on the grid, a thin option tape): the trees were fitted with them
# missing. Anything else missing is a fetch gone wrong, and refused.
_NAN_OK = set(OPTION_COLS)


def _normal_day(session_date) -> pd.Timestamp:
    day = pd.Timestamp(session_date).normalize()
    if day.tzinfo is not None:
        day = day.tz_localize(None)
    return day.as_unit("ns")


def is_half_day(session_date) -> bool:
    return bool(early_close(pd.DatetimeIndex([_normal_day(session_date)]))[0])


def _session_day(symbol: str, day: pd.Timestamp, heads: "tuple[str, ...]",
                 key: "str | None", secret: "str | None") -> _Day:
    """The session's history, read (and memoised) once."""
    memo = (symbol, day, heads)
    with _day_lock:
        found = _days.get(memo)
    if found is not None:
        return found
    hl2, ff = _hl2(), _ff()
    history = hl2.history_inputs(symbol, day, key, secret)
    sessions = history.sessions[hl2._ROLLUP_COLS]
    sessions = sessions[sessions.index < day]
    if len(sessions) < MIN_PRIOR_SESSIONS:
        raise ValueError(
            f"only {len(sessions)} complete SIP sessions of history before {day.date()}; "
            f"the HighLow_1h / HighLow_4h forecasts need {MIN_PRIOR_SESSIONS} for their 126-day "
            "windows."
        )
    sessions.index = pd.DatetimeIndex(sessions.index, name="date").as_unit("ns")
    # Today's row is a placeholder: every daily statistic is shifted a day.
    daily = pd.concat([sessions, pd.DataFrame(index=pd.DatetimeIndex([day], name="date"))])
    past = sessions.index[-HIST_SESSIONS:]
    raw = ff.raw_history(symbol, (day - pd.Timedelta(HIST_CALENDAR_DAYS, "D")).date(), day.date(), key, secret)
    sip = _minute_frame_of(raw["sip"], past)
    iex = _minute_frame_of(raw["iex"], past)
    have = pd.DatetimeIndex(sip["date"].unique())
    if not past[-HIST_EXCURSION_DAYS:].isin(have).all():
        missing = past[-HIST_EXCURSION_DAYS:][~past[-HIST_EXCURSION_DAYS:].isin(have)]
        raise ValueError(
            f"the SIP minute tape of {len(missing)} of the {HIST_EXCURSION_DAYS} sessions before "
            f"{day.date()} is missing ({', '.join(f'{d:%Y-%m-%d}' for d in missing[:3])}); the "
            "median past excursions read every one."
        )
    built = _Day(day=day, daily=daily, dates=past.append(pd.DatetimeIndex([day])).rename("date"),
                 sip=sip, iex=iex)
    with _day_lock:
        if len(_days) >= _DAYS_MAX:
            _days.clear()
        _days.setdefault(memo, built)
        return _days[memo]


def _options_row(ctx_day: _Day, symbol: str, key, secret) -> None:
    """Last night's option table, once a session (4h model). A failure is kept
    as the reason the 4h forecasts are refused -- never made without it -- and
    tried again after `_OPTIONS_RETRY_SEC`."""
    if ctx_day.opt is not None:
        return
    if "options" in ctx_day.problems and time.monotonic() - ctx_day.options_tried < _OPTIONS_RETRY_SEC:
        return
    from . import highlow2_options

    ctx_day.options_tried = time.monotonic()
    prev = highlow2_options.previous_session(ctx_day.day)
    try:
        rows = highlow2_options.positioning_rows(symbol, [prev], key, secret)
    except Exception as exc:  # noqa: BLE001 -- said in the forecast's refusal
        ctx_day.problems["options"] = f"{_OPTIONS_PROBLEM} could not be read: {exc}"
        return
    if f"{prev:%Y-%m-%d}" not in rows:
        ctx_day.problems["options"] = f"{_OPTIONS_PROBLEM} (the {prev:%Y-%m-%d} close) is not built yet."
        return
    ctx_day.problems.pop("options", None)
    table = highlow2_options.frame(rows)
    ctx_day.opt = pd.DataFrame([table.iloc[-1].to_dict()], index=pd.DatetimeIndex([ctx_day.day], name="date"))


def _premarket(ctx_day: _Day, symbol: str, key, secret) -> None:
    """Today's SIP pre-market, once a session (the 1h model's two-hour momentum
    reads it until 11:30)."""
    if ctx_day.pre is not None:
        return
    hl2 = _hl2()
    day = ctx_day.day.date()
    start = datetime(day.year, day.month, day.day, 4, 0, tzinfo=MARKET_TZ)
    end = datetime(day.year, day.month, day.day, 9, 29, tzinfo=MARKET_TZ)
    key, secret = _credentials(key, secret)
    bars = hl2._fetch_tape(symbol, start, end, key, secret, "sip")
    ctx_day.pre = premarket_frame(bars) if len(bars) else pd.DataFrame(columns=["close", "date", "pm"])


def _today_tapes(symbol: str, day: pd.Timestamp, through: int, live: bool,
                 key, secret, wait: bool) -> "tuple[pd.DataFrame, pd.DataFrame, bool]":
    """Today's SIP and IEX minute tables: the whole session from the tape cache
    in a replay, and live SIP to 15 minutes ago and IEX through bar `through`.
    The third value is False for a live fetch made before IEX's bar `through`
    was out (forecast from, not memoised)."""
    ff = _ff()
    if live:
        raw_sip, raw_iex, settled = ff.live_tapes(
            symbol, day.date(), through, key, secret, retries=None if wait else 0
        )
    else:
        raw = ff.raw_history(symbol, day.date(), (day + pd.Timedelta(1, "D")).date(), key, secret)
        raw_sip, raw_iex, settled = raw["sip"], raw["iex"], True
    sip, iex = _minute_frame_of(raw_sip, [day]), _minute_frame_of(raw_iex, [day])
    if not live:
        hl2 = _hl2()
        if not len(sip):
            raise ValueError(f"SIP printed nothing on {day.date()}.")
        report = hl2.session_report(sip)
        if not bool(report["kept"].iloc[0]):
            raise ValueError(
                f"{day.date()} is not a session the notebooks keep "
                f"({report['reason'].iloc[0].replace('_', ' ')}), so there is no window forecast for it."
            )
    return sip, iex, settled


def session_forecasts(
    bundles: "dict[str, dict]",
    ticker: str,
    session_date,
    through: "int | None" = None,
    key: "str | None" = None,
    secret: "str | None" = None,
    wait: bool = True,
) -> "tuple[dict[int, dict], dict[int, str]]":
    """The window forecasts of the session made by the close of bar `through`
    (0 = 09:30; all of them when None), keyed by decision minute, and why any
    decision in that span has none.

    `bundles` maps a head (`"1h"`, `"4h"`) to its bundle; a decision minute is
    forecast by the head whose grid holds it, and the forecast at minute m is
    made once bar m - 1 has closed. Each forecast: `head`, `minute`,
    `pred_high`, `pred_low`, `price` (the live price it was read at), `scale`,
    `window_end` (minutes after 9:30) and `adr14_abs`. A decision whose inputs
    are missing -- no option table, a feature the history cannot supply -- has
    a reason instead.

    A finished session is computed whole from the tape cache and memoised, so
    a SimLab grid replaying it pays once; today's is fetched live, SIP to 15
    minutes back and IEX through `through`, each decision cut by the notebook's
    own live-tape code. Either way a decision reads nothing that was not public
    at its minute.
    """
    symbol = str(ticker).upper()
    day = _normal_day(session_date)
    if is_half_day(day):
        raise ValueError(
            f"{day.date()} closes at 13:00, and the notebooks keep no such session: there are no "
            "HighLow_1h / HighLow_4h forecasts for it."
        )
    heads = tuple(sorted(h for h in bundles if h in HEADS and bundles[h] is not None))
    every = sorted(m for h in heads for m in HEADS[h].decisions)
    wanted = every if through is None else [m for m in every if m - 1 <= int(through)]
    if not wanted:
        return {}, {}
    live = day.date() >= _today_et()
    memo = (tuple(bundles[h].get("path") for h in heads), symbol, day)
    with _day_lock:
        known = dict(_rows_memo.get(memo) or {})
        refused = dict(_problems_memo.get(memo) or {})
    # A finished session is computed whole; today's as far as it has got.
    todo = [m for m in (wanted if live else every) if m not in known and m not in refused]
    if todo:
        rows, problems, settled = _forecast(bundles, heads, symbol, day, todo, live, key, secret, wait)
        if settled:
            with _day_lock:
                if len(_rows_memo) >= _ROWS_MEMO_MAX:
                    _rows_memo.clear()
                    _problems_memo.clear()
                _rows_memo.setdefault(memo, {}).update(rows)
                # A missing option table may be there on a later try; the rest
                # is the session's for good.
                _problems_memo.setdefault(memo, {}).update(
                    {m: why for m, why in problems.items() if not why.startswith(_OPTIONS_PROBLEM)}
                )
        known.update(rows)
        refused.update(problems)
    return ({m: known[m] for m in wanted if m in known},
            {m: refused[m] for m in wanted if m in refused})


def _forecast(bundles: "dict[str, dict]", heads: "tuple[str, ...]", symbol: str, day: pd.Timestamp,
              todo: "list[int]", live: bool, key, secret,
              wait: bool) -> "tuple[dict[int, dict], dict[int, str], bool]":
    """`session_forecasts`' decisions `todo`, made: (forecasts, refusals,
    whether the tape they read was settled)."""
    ctx_day = _session_day(symbol, day, heads, key, secret)
    sip_today, iex_today, settled = _today_tapes(symbol, day, max(todo) - 1, live, key, secret, wait)
    if "1h" in heads and any(m in HEADS["1h"].decisions and m - 1 - 120 < 0 for m in todo):
        _premarket(ctx_day, symbol, key, secret)
    if "4h" in heads and any(m in HEADS["4h"].decisions for m in todo):
        if bundles["4h"].get("reads_options"):
            _options_row(ctx_day, symbol, key, secret)
        if ctx_day.events is None and day in fomc_dates():
            ctx_day.events = _event_history(symbol, day, key, secret)

    sip = pd.concat([ctx_day.sip, sip_today]) if len(sip_today) else ctx_day.sip
    iex = pd.concat([ctx_day.iex, iex_today]) if len(iex_today) else ctx_day.iex
    ctx = _context(ctx_day.daily, sip, iex, ctx_day.pre, ctx_day.dates, ctx_day.events, ctx_day.opt)

    out: "dict[int, dict]" = {}
    problems: "dict[int, str]" = {}
    for name in heads:
        head, bundle = HEADS[name], bundles[name]
        minutes = [m for m in todo if m in head.decisions]
        if not minutes:
            continue
        if bundle.get("reads_options") and ctx.opt is None:
            why = ctx_day.problems.get("options", f"{_OPTIONS_PROBLEM} is missing.")
            problems.update({m: why for m in minutes})
            continue
        rows = build_panel(ctx, head, minutes).xs(day, level="date").loc[minutes]
        model = bundle["model"]
        missing = rows[model.feature_cols].isna()
        bad = missing[[c for c in model.feature_cols if c not in _NAN_OK]].any(axis=1)
        for m in rows.index[bad]:
            cols = [c for c in model.feature_cols if c not in _NAN_OK and bool(missing.at[m, c])]
            problems[int(m)] = (
                f"the feature row is incomplete ({', '.join(cols[:4])}"
                f"{'…' if len(cols) > 4 else ''}); the SIP or IEX tape is too short or has gaps."
            )
        good = rows[~bad]
        if not len(good):
            continue
        prices = model.predict_prices(good)
        for m in good.index:
            out[int(m)] = {
                "head": head.key,
                "minute": int(m),
                "pred_high": float(prices.at[m, "pred_high"]),
                "pred_low": float(prices.at[m, "pred_low"]),
                "price": float(good.at[m, "price"]),
                "scale": float(good.at[m, "scale"]),
                "window_end": head.end,
                "adr14_abs": float(good.at[m, "adr_usd"]),
            }
    return out, problems, settled


def warm(ticker: str, session_date, heads=("1h", "4h"),
         key: "str | None" = None, secret: "str | None" = None) -> None:
    """Fill what the session's window forecasts read, ahead of 10:30: the daily
    history, the minute tape, last night's option table and the event history."""
    symbol = str(ticker).upper()
    day = _normal_day(session_date)
    heads = tuple(sorted(heads))
    ctx_day = _session_day(symbol, day, heads, key, secret)
    if "4h" in heads:
        _options_row(ctx_day, symbol, key, secret)
        if ctx_day.events is None and day in fomc_dates():
            ctx_day.events = _event_history(symbol, day, key, secret)


def reset_caches() -> None:
    """Forget every memoised session (tests; a refit bundle)."""
    with _day_lock:
        _days.clear()
        _rows_memo.clear()
        _problems_memo.clear()
