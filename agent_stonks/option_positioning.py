"""Last night's option positioning, rebuilt from Alpaca's daily option bars.

Alpaca keeps no history of open interest, and the option inputs two models read
-- the near-term walls, dealer gamma, the implied move, put/call volume -- are
functions of it. FinNotebooks' `HighLow_3m` project rebuilt it per contract as
the running sum of daily volume weighted by the root of the average trade size,
scaled to contracts by a calibration on expired contracts (`NOTEBOOK_OPTIONS`),
and priced every contract off the underlying's SIP close and the 13-week T-bill
yield. This module does the same, from Alpaca's daily option bars cached per
expiration (`_options_dir`).

It was HighLow_3m's (`highlow3m_model`) until that model was removed from the
app on 2026-10-20. What it kept is what the two models still reading the chain
need:

* `fastfurious_model` -- the stop's range model reads the 1-day implied move,
  the 7-day put wall and the put gamma wall off `positioning_rows`. Its
  notebook's tables are identical to this one's on all 686 closes both cover
  (same contracts, same open-interest calibration, same dividend yield).
* `goodmoments_model` -- prices the same contracts its own notebook's way, from
  the cached contract list and daily bars (`option_history`).

The mirror contract
-------------------
The `--- options` and `--- session closes` blocks are verbatim copies of the
parts of `highlow3m/options.py` and `highlow3m/data.py` the tables read, with
the cuts each block names. `tests/test_option_positioning.py` rebuilds the
notebook's own tables from its option bars and pins the result.

What the live path has to supply
--------------------------------
* The SIP close of *every* session over the contracts' 120-day windows -- half
  days and sessions the models drop too, which the options are priced off
  (cached in `_closes_path`, fetched only where the cache falls short).
* The contract list (Alpaca's trading API), every listed expiration's daily bars
  over its last 120 days, and Yahoo ^IRX for the risk-free rate.

A positioning row for close d depends on nothing after d, so once built it is
kept (`_positioning_path`). It is read from the session date the caller names
rather than the wall clock, so a SimLab replay reads the same honest table a
live run on that day would have.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from . import market_hours
from .highlow2_model import OHLCV, _fetch_tape, add_session_columns, sane_bars
from .model_store import RTH_END, RTH_START
from .newsimpact_model import early_close

SESSION_MINUTES = 390
OPEN_MINUTE = 9 * 60 + 30

# The session closes asked for, in calendar days before the latest close a row
# is built for: more than a contract's 120-day window, with room for a replay
# stepping back a few months without refetching.
HISTORY_CALENDAR_DAYS = 220

# The SIP session closes, one JSON file per symbol, the option positioning rows
# beside them, the ^IRX closes, and the option chain's raw daily bars under
# `options/`.
HISTORY_DIR = Path(__file__).resolve().parent.parent / "data" / "option_positioning"
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


# --- the session closes' cache ---------------------------------------------------------

_closes_lock = threading.Lock()


def _dates(values) -> pd.DatetimeIndex:
    """ISO dates read back from a cache as a session index. In nanoseconds,
    as the tapes' own session dates are: pandas 3 parses strings to
    microseconds, and `merge_asof` refuses to join the two."""
    return pd.DatetimeIndex(pd.to_datetime(list(values)), name="date").as_unit("ns")


def _empty_closes() -> pd.Series:
    return pd.Series([], index=pd.DatetimeIndex([], name="date"), dtype=float, name="close")


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


def _credentials(key: "str | None", secret: "str | None") -> "tuple[str, str]":
    """The caller's Alpaca keys, else the environment's (SimLab has no sidebar)."""
    key = key or os.getenv("ALPACA_API_KEY", "")
    secret = secret or os.getenv("ALPACA_SECRET", "")
    if not (key and secret):
        raise ValueError(
            "the option tables need Alpaca SIP and option history and no Alpaca "
            "credentials are available (sidebar connection, or ALPACA_API_KEY / "
            "ALPACA_SECRET in the environment)."
        )
    return key, secret


def _closes_path(symbol: str) -> Path:
    return HISTORY_DIR / f"{symbol.upper()}_closes.json"


def _empty_cache() -> dict:
    return {"layout": CACHE_LAYOUT, "from": None, "through": None, "closes": {}}


def _read_cache(symbol: str) -> dict:
    """The symbol's cache, or an empty one if it is unreadable or in another layout."""
    payload = _read_json(_closes_path(symbol))
    if not payload or payload.get("layout") != CACHE_LAYOUT:
        return _empty_cache()
    payload.setdefault("closes", {})
    return payload


def _fetch_closes(symbol: str, first: date, before: date, key: str, secret: str) -> pd.Series:
    """`session_closes` for the calendar days in [first, before), straight
    from Alpaca's SIP tape, split-adjusted."""
    from .datalog import log_fetch

    tz = market_hours.MARKET_TZ
    start = datetime.combine(first, datetime.min.time(), tzinfo=tz)
    end = datetime.combine(before, datetime.min.time(), tzinfo=tz)
    sip = _fetch_tape(symbol, start, end, key, secret, "sip")
    rth = sip.between_time(RTH_START, RTH_END) if len(sip) else sip
    out = session_closes(rth[OHLCV]) if len(rth) else _empty_closes()
    log_fetch(
        "minute bars (option tables' session closes)",
        "Alpaca REST (SIP, split-adjusted)",
        symbol=symbol,
        detail=f"{len(sip)} SIP bars, {len(out)} session closes from {first}",
    )
    return out


def _merge(cache: dict, fresh: pd.Series, seam: "str | None") -> bool:
    """Fold a fetch into the cache. False (and nothing merged) if the session
    both hold -- `seam` -- closed at another price, which means a split
    rescaled the tape."""
    closes = {d.strftime("%Y-%m-%d"): float(v) for d, v in fresh.items()}
    if seam and seam in cache["closes"] and seam in closes:
        if abs(closes[seam] / cache["closes"][seam] - 1.0) > 1e-6:
            return False
    cache["closes"].update(closes)
    return True


def _series(values: dict, lo: str, hi: str) -> pd.Series:
    values = {d: v for d, v in values.items() if lo <= d < hi}
    if not values:
        return _empty_closes()
    out = pd.Series(values, dtype=float, name="close")
    out.index = _dates(out.index)
    return out.sort_index()


def sip_closes(
    symbol: str,
    before,
    key: "str | None" = None,
    secret: "str | None" = None,
) -> pd.Series:
    """SIP's close of every session strictly before `before`, reaching
    HISTORY_CALENDAR_DAYS back (`session_closes`).

    `highlow2_model.history_inputs`' scheme: the cache remembers which calendar
    days it has looked at (`from` .. `through`), and only the days outside that
    stretch are fetched -- so a live session fetches one day each morning and a
    replay of a day inside the stretch fetches nothing. Each extension re-reads
    the cached session at its seam: if that completed session's close has
    moved, a split has rescaled the tape, and the cache is rebuilt rather than
    mixed.
    """
    symbol = symbol.upper()
    before = pd.Timestamp(before).date()
    first_wanted = before - timedelta(days=HISTORY_CALENDAR_DAYS)
    last_wanted = before - timedelta(days=1)
    with _closes_lock:
        cache = _read_cache(symbol)
        lo = date.fromisoformat(cache["from"]) if cache.get("from") else None
        hi = date.fromisoformat(cache["through"]) if cache.get("through") else None
        if lo is None or hi is None or not cache["closes"]:
            cache, lo, hi = _empty_cache(), None, None

        if lo is None or lo > first_wanted or hi < last_wanted:
            key, secret = _credentials(key, secret)

            def fetch(first: date, upto: date) -> pd.Series:
                return _fetch_closes(symbol, first, upto, key, secret)

            ok = True
            if lo is None or hi < first_wanted or lo > last_wanted:
                # Nothing usable overlaps: fetch the whole window.
                cache = _empty_cache()
                _merge(cache, fetch(first_wanted, before), None)
                lo, hi = first_wanted, last_wanted
            else:
                if lo > first_wanted:
                    seam = min(cache["closes"])
                    upto = date.fromisoformat(seam) + timedelta(days=1)
                    ok = _merge(cache, fetch(first_wanted, upto), seam)
                    lo = first_wanted
                if ok and hi < last_wanted:
                    seam = max(cache["closes"])
                    ok = _merge(cache, fetch(date.fromisoformat(seam), before), seam)
                    hi = last_wanted
            if not ok:
                cache = _empty_cache()
                _merge(cache, fetch(first_wanted, before), None)
                lo, hi = first_wanted, last_wanted
            cache["from"], cache["through"] = lo.isoformat(), hi.isoformat()
            _write_json(_closes_path(symbol), cache)

    return _series(cache["closes"], first_wanted.isoformat(), before.isoformat())


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
        log_fetch("option contracts (option tables)", "Alpaca trading API", symbol=symbol,
                  detail=f"{len(got)} contracts expiring {lo:%Y-%m-%d} – {reach:%Y-%m-%d}")
        cached = merged
    expiry = cached["expiration"]
    return cached[(expiry >= first) & (expiry <= hi)].reset_index(drop=True)


def _bars_path(symbol: str, expiration: pd.Timestamp) -> Path:
    return _options_dir(symbol) / "bars" / f"{expiration:%Y-%m-%d}.parquet"


def _read_bars(path: Path) -> "pd.DataFrame | None":
    """An expiration's cached bars, or None when the file is missing or unreadable."""
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except (OSError, ValueError):
        return None


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
    olds: "dict[pd.Timestamp, pd.DataFrame | None]" = {}
    for expiration, group in contracts.groupby("expiration"):
        start, end = bar_window(expiration, through)
        path_px = closes.loc[start:end]
        if path_px.empty:
            continue
        band = set(group.loc[group["strike"].between(lo * path_px.min(), hi * path_px.max()), "symbol"])
        entry = index["expirations"].get(f"{expiration:%Y-%m-%d}") or {}
        olds[expiration] = _read_bars(_bars_path(symbol, expiration))
        if olds[expiration] is None:
            # The index remembers bars the file no longer holds (missing, or
            # unreadable): fetched again over the whole window, rather than
            # extended -- an extension alone would leave the file the newest
            # days only (the 2026-10-09 expiry, found 2026-10-20).
            entry = {}
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
        log_fetch("option daily bars (option tables)", "Alpaca REST", symbol=symbol,
                  detail=f"{len(got)} bars of {sum(len(s) for s in jobs.values())} contracts "
                         f"through {through:%Y-%m-%d}")
        expiry_of = contracts.set_index("symbol")["expiration"]
        got_by_exp = dict(tuple(got.groupby(got["symbol"].map(expiry_of)))) if len(got) else {}
    else:
        got_by_exp = {}

    frames = []
    for expiration, plan in plans.items():
        path = _bars_path(symbol, expiration)
        old = olds.get(expiration)
        add = got_by_exp.get(expiration)
        if add is not None and len(add):
            both = add if old is None else pd.concat([old, add], ignore_index=True)
            both = both.drop_duplicates(["symbol", "date"], keep="last").sort_values(["symbol", "date"])
            path.parent.mkdir(parents=True, exist_ok=True)
            # Written aside and moved into place: SimLab and the live app share
            # this cache, and a reader catching a half-written file would take
            # it for a missing one.
            tmp = path.with_suffix(".tmp")
            both[_BAR_COLUMNS].reset_index(drop=True).to_parquet(tmp)
            tmp.replace(path)
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
            f"no notebook option settings for {symbol} (`option_positioning.NOTEBOOK_OPTIONS`): "
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
            built = _build_rows(symbol, lo, hi, settings, key, secret)
            for iso, (row, final) in built.items():
                if final:
                    rows[iso] = row
            out.update({iso: row for iso, (row, _) in built.items()})
        if missing:
            _write_json(_positioning_path(symbol), cache)
        out.update({iso: rows[iso] for iso in rows})
    return {f"{d:%Y-%m-%d}": out[f"{d:%Y-%m-%d}"] for d in wanted if f"{d:%Y-%m-%d}" in out}


# Closes further apart than this are built separately: one build reads the
# session closes around both ends, and a span much wider than they reach
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
                key: "str | None", secret: "str | None") -> "dict[str, tuple[dict, bool]]":
    """Every close in [lo, hi] -> (row, whether it may be kept)."""
    key, secret = _credentials(key, secret)
    closes = sip_closes(symbol, hi + pd.Timedelta(1, "D"), key, secret)
    reach = lo - pd.Timedelta(OPTION_MAX_DTE + 5, "D")
    if not len(closes) or closes.index.min() > reach:
        older = sip_closes(symbol, lo, key, secret)
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


def option_history(
    symbol: str,
    first_expiry: pd.Timestamp,
    last_expiry: pd.Timestamp,
    closes: pd.Series,
    through: pd.Timestamp,
    key: "str | None" = None,
    secret: "str | None" = None,
) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """The contracts expiring in [first_expiry, last_expiry] and their daily
    bars up to `through`, from this module's option cache (fetched where it
    falls short) -- for another model that prices the same chain its own way
    (`goodmoments_model`). `closes` are SIP session closes over the contracts'
    windows: they decide which strikes are fetched (`_ensure_option_bars`)."""
    symbol = symbol.upper()
    key, secret = _credentials(key, secret)
    with _options_lock:
        index = _options_index(symbol)
        contracts = _contracts(symbol, index, pd.Timestamp(first_expiry), pd.Timestamp(last_expiry),
                               pd.Timestamp(through), key, secret)
        if not len(contracts):
            return contracts, pd.DataFrame(columns=_BAR_COLUMNS)
        bars = _ensure_option_bars(symbol, index, contracts, closes, pd.Timestamp(through), key, secret)
    return contracts, bars

