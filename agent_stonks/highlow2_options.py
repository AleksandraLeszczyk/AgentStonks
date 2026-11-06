"""Last night's option positioning the way HighLow2_5m built it.

FinNotebooks' `HighLow2_5m` rebuilt the option chain's positioning per close in
its own `highlow2/options.py`, and `HighLow_4h` (`highlow_window_model`) reads
that table as it is: the walls by open interest and by dealer gamma, the gamma
flip, the max pain of the next standard expiry, the 1- and 5-session implied
moves, the 30-day at-the-money vol, put/call open interest and volume, and the
sessions to the monthly expiry.

It is the same idea as HighLow_3m's tables (`option_positioning`) -- open
interest rebuilt from daily option bars, every contract priced off the SIP
close and the 13-week T-bill yield -- and it reads the same raw material, so
this module reuses that one's caches (the SIP session closes, the contract
list, each expiration's daily bars, ^IRX) and its pricing functions, which are
the same code. What differs, and is mirrored here:

* the smile is fitted from moment sums (`_weighted_polyfit`) rather than
  `np.polyfit` -- the same least squares, other rounding;
* the open-interest proxy's calibration (`NOTEBOOK_OPTIONS`: HighLow2_5m's own
  pick, a different scale from HighLow_3m's);
* a wall's tie goes to the higher strike;
* the 30-day vol reads expirations at least two days out, and the 5-session
  implied move, max pain, the expiry calendar and the volume coverage are
  HighLow2_5m's alone.

The mirror contract
-------------------
The `--- options` block is a verbatim copy of the parts of `highlow2/options.py`
the shipped HighLow_4h model reads, with the columns no feature reads cut from
`positioning_tables`. `tests/test_highlow_chain.py` rebuilds the notebook's own
positioning table from its cached option bars and pins the result.

A row for close d depends on nothing after d, so once built it is kept
(`_positioning_path`); it is read from the session date the caller names rather
than the wall clock, so a SimLab replay reads the same table a live run on that
day would have.
"""

from __future__ import annotations

import math
import threading

import numpy as np
import pandas as pd

from . import option_positioning as op
from .option_positioning import (
    GEX_GRID,
    OPTION_MAX_DTE,
    OPTION_STRIKE_BAND,
    SMILE_MAX_K,
    SMILE_MIN_VOLUME,
    _d64,
    _holidays,
    dollar_gamma,
    gamma_flip,
    gex_profile,
    implied_vol,
    risk_free,
    sessions_between,
)

# What the notebook settled per ticker, which no saved file but its calibration
# table records: the dividend yield the options are priced with
# (`config._TICKERS`) and the open-interest proxy `options.calibrate` chose
# (`data/<T>/options/oi_proxy_calibration.parquet`, the `chosen` row: a plain
# running sum of size-weighted volume, the trade size capped at 100 contracts).
# A ticker is forecast only with an entry here.
NOTEBOOK_OPTIONS: "dict[str, dict]" = {
    "AAPL": {
        "dividend_yield": 0.004,
        "h1": math.inf,
        "h2": math.nan,
        "weighting": "size",
        "cap": 100.0,
        "scale": 0.1223216296453345,
    },
}


# --- options (mirrors highlow2.options) -------------------------------------------
#
# Cut: the downloaders and the strike pre-selection (the shared cache in
# `option_positioning` holds a superset, and the tables re-band on each day's
# own close), the calibration search (its answer is `NOTEBOOK_OPTIONS`), the
# front expiry's straddle, and every positioning column no HighLow_4h feature
# reads. `bsm_*`, `implied_vol`, `risk_free`, `dollar_gamma`, `gex_profile` and
# `gamma_flip` are `option_positioning`'s, the same code as the notebook's.

SMILE_BOUNDS = (0.05, 3.0)    # fitted vols clipped to this range
K_SCALE = 0.1                 # k is fitted in units of 0.1 to keep the normal equations well scaled
POSITIONING_MAX_DTE = 60   # calendar days; further out adds little gamma
SHORT_WALL_DTE = 7
FIXED_TENORS = {"iv_1d": 1, "iv_5d": 5}   # sessions ahead
VOLUME_LOOKBACK = 20                      # sessions behind the coverage median


def is_standard_expiry(expirations) -> np.ndarray:
    """Fridays, and Thursdays standing in for a Friday holiday. Monday and Wednesday
    expiries, and their own holiday shifts, are not standard."""
    e = pd.DatetimeIndex(expirations)
    friday_off = np.isin(_d64(e + pd.Timedelta(days=1)), _holidays())
    return np.asarray((e.dayofweek == 4) | ((e.dayofweek == 3) & friday_off))


def contract_days(contracts: pd.DataFrame, bars: pd.DataFrame, close: pd.Series,
                  dividend_yield: float, rates: pd.Series) -> pd.DataFrame:
    """One row per contract and option session, from its first trade to expiry or the last session.

    `close` is the underlying's close per session (`session_closes`). Days without
    a bar have volume 0 and no price. `n` is the session index, `t` the year
    fraction to the 16:00 close on expiry, `k` log-moneyness against the forward.
    (`rates` is `risk_free(days)`, handed in rather than read inside; the last
    session is `close`'s, the notebook's HISTORY_END.)
    """
    days = close.index
    first = bars.groupby("symbol")["date"].min()
    meta = contracts.set_index("symbol").reindex(first.index)
    first, meta = first[meta["type"].notna()], meta[meta["type"].notna()]
    start = days.searchsorted(first.to_numpy())
    last_day = np.minimum(meta["expiration"].to_numpy(), days[-1].to_datetime64())
    end = days.searchsorted(last_day, side="right") - 1
    lengths = np.maximum(end - start + 1, 0)

    which = np.repeat(np.arange(len(first)), lengths)
    offset = np.arange(lengths.sum()) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    n = np.repeat(start, lengths) + offset
    panel = pd.DataFrame({
        "symbol": first.index.to_numpy()[which],
        "date": days[n],
        "n": n,
        "type": meta["type"].to_numpy()[which],
        "strike": meta["strike"].to_numpy()[which],
        "expiration": meta["expiration"].to_numpy()[which],
    })
    panel = panel.merge(bars[["symbol", "date", "close", "volume", "trades"]], on=["symbol", "date"], how="left")
    panel[["volume", "trades"]] = panel[["volume", "trades"]].fillna(0).astype("int64")

    panel["spot"] = close.to_numpy()[n]
    panel["rate"] = rates.to_numpy()[n]
    panel["q"] = dividend_yield
    panel["dte"] = (panel["expiration"] - panel["date"]).dt.days
    panel["t"] = panel["dte"] / 365
    forward = panel["spot"] * np.exp((panel["rate"] - panel["q"]) * panel["t"])
    panel["k"] = np.log(panel["strike"] / forward)
    panel["iv"] = _contract_iv(panel)
    return panel.sort_values(["symbol", "date"]).reset_index(drop=True)


def _contract_iv(panel: pd.DataFrame) -> np.ndarray:
    """IV from the option close where it traded and is worth more than intrinsic."""
    p = panel
    is_call = (p["type"] == "call").to_numpy()
    spot_pv = p["spot"] * np.exp(-p["q"] * p["t"])
    strike_pv = p["strike"] * np.exp(-p["rate"] * p["t"])
    intrinsic = np.where(is_call, spot_pv - strike_pv, strike_pv - spot_pv).clip(min=0)
    ok = ((p["volume"] > 0) & (p["t"] > 0) & (p["close"] > intrinsic + 0.01)).to_numpy()
    iv = np.full(len(p), np.nan)
    cols = [p[c].to_numpy()[ok] for c in ("close", "spot", "strike", "t", "rate", "q")]
    iv[ok] = implied_vol(*cols, is_call[ok])
    return iv


def _weighted_polyfit(sums: pd.DataFrame, degree: int) -> np.ndarray:
    """Weighted least squares per group from moment sums; columns a, b, c in units of u = k / K_SCALE."""
    size = degree + 1
    moments = np.stack([[sums[f"m{i + j}"].to_numpy() for j in range(size)] for i in range(size)])
    moments = moments.transpose(2, 0, 1)
    rhs = np.stack([sums[f"r{i}"].to_numpy() for i in range(size)], axis=1)[..., None]
    coef = (np.linalg.pinv(moments) @ rhs)[..., 0]
    return np.pad(coef, ((0, 0), (0, 3 - size)))


def fit_smiles(panel: pd.DataFrame) -> pd.DataFrame:
    """IV against log-moneyness per (date, expiration), from out-of-the-money contracts.

    Weighted by sqrt(volume). Quadratic with at least 6 points and 2 on each side
    of the money, else linear, else a constant. Only out-of-the-money options are
    used: in-the-money ones carry the early-exercise premium of American puts and
    their last trade is often stale.
    """
    p = panel
    otm = np.where(p["type"] == "call", p["k"] >= 0, p["k"] <= 0)
    pts = p[p["iv"].notna() & (p["volume"] >= SMILE_MIN_VOLUME) & (p["k"].abs() <= SMILE_MAX_K) & otm]
    u = pts["k"] / K_SCALE
    w = np.sqrt(pts["volume"])
    cols = {f"m{i}": w * u**i for i in range(5)}
    cols.update({f"r{i}": w * u**i * pts["iv"] for i in range(3)})
    cols.update(below=(pts["k"] < 0).astype(int), above=(pts["k"] > 0).astype(int), k=pts["k"])
    frame = pd.DataFrame(cols).assign(date=pts["date"], expiration=pts["expiration"])
    sums = frame.groupby(["date", "expiration"]).agg(
        **{c: (c, "sum") for c in [*(f"m{i}" for i in range(5)), "r0", "r1", "r2", "below", "above"]},
        n_points=("k", "size"), kmin=("k", "min"), kmax=("k", "max"),
    )
    quad = (sums["n_points"] >= 6) & (sums["below"] >= 2) & (sums["above"] >= 2)
    linear = ~quad & (sums["n_points"] >= 2) & (sums["kmax"] - sums["kmin"] >= 0.01)
    coef = np.column_stack([sums["r0"] / sums["m0"], np.zeros(len(sums)), np.zeros(len(sums))])
    for mask, degree in ((quad, 2), (linear, 1)):
        if mask.any():
            coef[mask.to_numpy()] = _weighted_polyfit(sums[mask], degree)
    smiles = sums[["n_points", "kmin", "kmax"]].copy()
    smiles["a"] = coef[:, 0]
    smiles["b"] = coef[:, 1] / K_SCALE
    smiles["c"] = coef[:, 2] / K_SCALE**2
    smiles["kind"] = np.select([quad, linear], ["quadratic", "linear"], "constant")
    smiles = smiles.reset_index()
    smiles["dte"] = (smiles["expiration"] - smiles["date"]).dt.days
    return smiles


def smile_iv(smile: pd.DataFrame, k) -> np.ndarray:
    """Fitted IV at log-moneyness k, flat beyond the fitted range."""
    kc = np.clip(k, smile["kmin"].to_numpy(), smile["kmax"].to_numpy())
    iv = smile["a"].to_numpy() + smile["b"].to_numpy() * kc + smile["c"].to_numpy() * kc**2
    return np.clip(iv, *SMILE_BOUNDS)


def assign_smiles(panel: pd.DataFrame, smiles: pd.DataFrame) -> pd.DataFrame:
    """Smile for every live (date, expiration). Expirations without one borrow the nearest by DTE that day."""
    params = ["a", "b", "c", "kmin", "kmax"]
    live = panel.loc[panel["t"] > 0, ["date", "expiration", "dte"]].drop_duplicates()
    own = live.merge(smiles[["date", "expiration", *params]], on=["date", "expiration"], how="left")
    own["smile_dte"] = np.where(own["a"].notna(), own["dte"], np.nan)
    missing = own[own["a"].isna()].drop(columns=[*params, "smile_dte"])
    donors = smiles[["date", "dte", *params]].assign(smile_dte=smiles["dte"])
    borrowed = pd.merge_asof(missing.sort_values(["dte", "date"], kind="stable"),
                             donors.sort_values(["dte", "date"], kind="stable"),
                             on="dte", by="date", direction="nearest")
    out = pd.concat([own[own["a"].notna()], borrowed], ignore_index=True)
    out["borrowed"] = out["smile_dte"] != out["dte"]
    return out.sort_values(["date", "expiration"], kind="stable").reset_index(drop=True)


def add_fitted_iv(panel: pd.DataFrame, assigned: pd.DataFrame) -> pd.DataFrame:
    """iv_fit for every live contract-day, traded or not. `assigned` comes from assign_smiles."""
    merged = panel.merge(assigned.drop(columns="dte"), on=["date", "expiration"], how="left")
    merged["iv_fit"] = np.where(merged["a"].notna(), smile_iv(merged, merged["k"].to_numpy()), np.nan)
    return merged.drop(columns=["a", "b", "c", "kmin", "kmax"])


def daily_flow(panel: pd.DataFrame, weighting: str = "volume", cap: float = np.inf) -> np.ndarray:
    """What one day adds to a contract's open-interest proxy."""
    volume = panel["volume"].to_numpy(float)
    if weighting == "volume":
        return volume
    if weighting == "size":
        trades = panel["trades"].to_numpy(float)
        size = np.divide(volume, trades, out=np.zeros_like(volume), where=trades > 0)
        return volume * np.sqrt(np.minimum(size, cap))
    raise ValueError(f"unknown weighting {weighting!r}")


def decayed_volume(panel: pd.DataFrame, half_life: float, weighting: str = "volume",
                   cap: float = np.inf) -> np.ndarray:
    """Per contract: sum over s <= d of flow(s) * 0.5 ** ((n(d) - n(s)) / half_life).

    `panel` must be sorted by symbol and date. half_life=inf is the plain running sum.
    """
    flow = pd.Series(daily_flow(panel, weighting, cap), index=panel.index)
    if np.isinf(half_life):
        return flow.groupby(panel["symbol"]).cumsum().to_numpy()
    # rescale by the contract's own age so the running sum never overflows
    age = (panel["n"] - panel.groupby("symbol")["n"].transform("first")).to_numpy()
    decay = 0.5 ** (1 / half_life)
    grown = flow * decay ** (-age)
    return grown.groupby(panel["symbol"]).cumsum().to_numpy() * decay**age


def oi_proxy(panel: pd.DataFrame, h1: float, h2: float = np.nan, weighting: str = "volume",
             cap: float = np.inf) -> np.ndarray:
    """Decayed flow with half-life h1, or the equal blend of h1 and h2. Not scaled to real OI."""
    if np.isnan(h2):
        return decayed_volume(panel, h1, weighting, cap)
    return 0.5 * (decayed_volume(panel, h1, weighting, cap) + decayed_volume(panel, h2, weighting, cap))


def scaled_oi(panel: pd.DataFrame, settings: dict) -> np.ndarray:
    """The picked proxy on every panel row, in contracts of real open interest.
    (`settings` is the calibration's `chosen` row, `NOTEBOOK_OPTIONS`.)"""
    return settings["scale"] * oi_proxy(panel, settings["h1"], settings["h2"], settings["weighting"],
                                        settings["cap"])


def _wall(uni: pd.DataFrame, value: str, kind: str, max_dte: "int | None" = None) -> pd.Series:
    """Strike with the largest summed `value`: calls above spot, or puts below spot.
    A tie goes to the higher strike."""
    side = uni[uni["type"] == kind]
    side = side[side["strike"] > side["spot"]] if kind == "call" else side[side["strike"] < side["spot"]]
    if max_dte is not None:
        side = side[side["dte"] <= max_dte]
    by_strike = side.groupby(["date", "strike"])[value].sum().abs().reset_index()
    by_strike = by_strike[by_strike[value] > 0]
    top = by_strike.sort_values([value, "strike"], kind="stable").drop_duplicates("date", keep="last")
    return top.set_index("date")["strike"]


def _term_iv(t: np.ndarray, iv: np.ndarray, target: float) -> float:
    """IV at year fraction `target`, linear in total variance between expirations, flat outside."""
    order = np.argsort(t, kind="stable")
    t, iv = t[order], iv[order]
    if target <= t[0]:
        return iv[0]
    if target >= t[-1]:
        return iv[-1]
    return np.sqrt(np.interp(target, t, iv**2 * t) / target)


def term_structure(smiles: pd.DataFrame) -> pd.DataFrame:
    """ATM vol at 30 calendar days, from expirations with their own smile.
    (Cut: the 7-day vol, the skew and the slope, which no shipped feature reads.)"""
    rows = {}
    for date, g in smiles[smiles["dte"] >= 2].groupby("date"):
        t = g["dte"].to_numpy() / 365
        atm = smile_iv(g, 0.0)
        rows[date] = dict(atm_iv_30=_term_iv(t, atm, 30 / 365))
    return pd.DataFrame.from_dict(rows, orient="index")


def fixed_tenor_iv(smiles: pd.DataFrame) -> pd.DataFrame:
    """ATM vol 1 and 5 sessions ahead, from every expiration with its own smile.

    Each expiration prices a total variance, atm_iv^2 * dte / 365. Laid out against
    sessions to expiry / 252 and interpolated linearly, it gives a vol per year of
    sessions, so iv_1d / sqrt(252) is the expected size of one session's move.
    Flat vol before the first expiration and after the last. Which weekdays carry
    an expiry changes the knots, not what is measured.
    """
    s = smiles[smiles["dte"] > 0]
    x = sessions_between(s["date"], s["expiration"]) / 252
    w = smile_iv(s, 0.0) ** 2 * s["dte"].to_numpy() / 365
    keep = x > 0
    curve = pd.DataFrame({"date": s["date"].to_numpy()[keep], "x": x[keep],
                          "iv": np.sqrt(w[keep] / x[keep])})
    rows = {}
    for date, g in curve.groupby("date"):
        rows[date] = {name: _term_iv(g["x"].to_numpy(), g["iv"].to_numpy(), n / 252)
                      for name, n in FIXED_TENORS.items()}
    return pd.DataFrame.from_dict(rows, orient="index")


def max_pain(chain: pd.DataFrame) -> pd.Series:
    """Per date, the settlement strike that pays option holders least. One expiration per date in `chain`."""
    out = {}
    for date, g in chain.groupby("date"):
        if g["oi"].sum() <= 0:
            continue
        strikes = np.unique(g["strike"])
        calls, puts = g[g["type"] == "call"], g[g["type"] == "put"]
        pain = (np.clip(strikes[:, None] - calls["strike"].to_numpy(), 0, None) * calls["oi"].to_numpy()).sum(1)
        pain += (np.clip(puts["strike"].to_numpy() - strikes[:, None], 0, None) * puts["oi"].to_numpy()).sum(1)
        out[date] = strikes[np.argmin(pain)]  # a tie goes to the lower strike
    return pd.Series(out, dtype=float)


def expiry_calendar(sessions: pd.DatetimeIndex, expirations) -> pd.DataFrame:
    """Per session: the next session, the next monthly opex, and the nearest
    standard expiry after the session (`std_front`). (Cut: what expires on the
    next session.)"""
    holidays = _holidays()
    d64 = _d64(sessions)
    next_session = np.busday_offset(d64, 1, roll="forward", holidays=holidays)

    months = pd.date_range(sessions.min().replace(day=1), sessions.max() + pd.Timedelta(days=70), freq="MS")
    third_friday = months + pd.to_timedelta((4 - months.dayofweek) % 7 + 14, unit="D")
    opex = np.busday_offset(_d64(third_friday), 0, roll="backward", holidays=holidays)
    upcoming = opex[np.searchsorted(opex, next_session)]

    expiries = np.unique(_d64(expirations))
    standard = expiries[is_standard_expiry(expiries)]
    std_front = standard[np.searchsorted(standard, d64, side="right")]
    return pd.DataFrame({
        "next_session": pd.DatetimeIndex(next_session).as_unit("ns"),
        "days_to_monthly_opex": np.busday_count(next_session, upcoming, holidays=holidays),
        "std_front": pd.DatetimeIndex(std_front).as_unit("ns"),
    }, index=sessions)


def positioning_tables(panel: pd.DataFrame, smiles: pd.DataFrame, contracts: pd.DataFrame,
                       close: pd.Series, rates: pd.Series) -> pd.DataFrame:
    """One row per option session with walls, gamma, vol and expiry features.

    `panel` needs `oi` (the scaled proxy) and `iv_fit`; `close` is the underlying's
    close per session. Not shifted: row d is the close of d. (Cut: the front
    expiry, the skew and slope, the next session's expiry flags and open
    interest shares, the off-cycle expiries, the contract counts and the
    long-format gamma profile -- none read by a shipped feature.)
    """
    days = close.index
    # A day without a single option bar (2024-02-02 at Alpaca) has no vols, so
    # gamma there would read as zero. Such days stay empty instead.
    traded = panel.loc[panel["volume"] > 0, "date"].unique()
    # The download kept strikes by the price range over each contract's whole life,
    # which runs past d. Re-banding on the day's own close makes the set point-in-time:
    # every strike the download dropped lies outside this band on every date.
    lo_mult, hi_mult = OPTION_STRIKE_BAND
    in_band = panel["strike"].between(lo_mult * panel["spot"], hi_mult * panel["spot"])
    near = (panel["dte"] > 0) & (panel["dte"] <= POSITIONING_MAX_DTE) & in_band
    uni = panel[near & panel["date"].isin(traded)].copy()
    uni["dgamma"] = dollar_gamma(uni, uni["spot"].to_numpy())
    is_call = uni["type"] == "call"

    def by_date(col: str, mask: pd.Series) -> pd.Series:
        return uni[col].where(mask, 0).groupby(uni["date"]).sum()

    out = pd.DataFrame(index=pd.DatetimeIndex(days, name="date"))
    out["spot"] = close
    out["rate"] = rates
    out["call_wall_oi"] = _wall(uni, "oi", "call")
    out["put_wall_oi"] = _wall(uni, "oi", "put")
    out["call_wall_gex"] = _wall(uni, "dgamma", "call")
    out["put_wall_gex"] = _wall(uni, "dgamma", "put")
    out["call_wall_oi_7"] = _wall(uni, "oi", "call", SHORT_WALL_DTE)
    out["put_wall_oi_7"] = _wall(uni, "oi", "put", SHORT_WALL_DTE)
    # the proxy's level drifts against real OI over the years; the ratio does not care
    out["net_gex"] = uni.groupby("date")["dgamma"].sum()
    out["gross_gex"] = uni["dgamma"].abs().groupby(uni["date"]).sum()
    out["gex_ratio"] = out["net_gex"] / out["gross_gex"].replace(0, np.nan)

    net_profile, _ = gex_profile(uni)
    out["gamma_flip"] = gamma_flip(net_profile, out["spot"])
    out = out.join(term_structure(smiles))
    out = out.join(fixed_tenor_iv(smiles))

    out["call_volume"] = by_date("volume", is_call)
    out["put_volume"] = by_date("volume", ~is_call)
    # Every in-band contract, expiring ones included: leaving out the day's own
    # expiry would make Mondays and Wednesdays look thin from 2026 on. A day with
    # no bars at all counts as zero, so a partial tape stands out against the past
    # month's median.
    flow = panel.loc[in_band, "volume"].groupby(panel.loc[in_band, "date"]).sum().reindex(days)
    past = flow.shift(1).rolling(VOLUME_LOOKBACK, min_periods=VOLUME_LOOKBACK // 2).median()
    out["opt_volume_coverage"] = flow / past.replace(0, np.nan)
    out["call_oi"] = by_date("oi", is_call)
    out["put_oi"] = by_date("oi", ~is_call)

    calendar = expiry_calendar(days, contracts["expiration"].unique())
    in_std_front = uni["expiration"].to_numpy() == calendar["std_front"].reindex(uni["date"]).to_numpy()
    out["max_pain_std"] = max_pain(uni[in_std_front])
    out = out.join(calendar[["days_to_monthly_opex"]])
    return out


def positioning_from(contracts: pd.DataFrame, bars: pd.DataFrame, closes: pd.Series,
                     irx: pd.Series, settings: dict) -> pd.DataFrame:
    """`options.build_all`'s steps 4-6 on bars already in hand: the positioning
    table, one row per close in `closes`.

    `closes` must reach back over every contract's window -- a row for day d is
    exact once every contract alive on d has its bars from 120 days before its
    expiry -- and end at the last close wanted: nothing after it is read.
    """
    bars = op.load_bars(bars, closes.index)
    if not len(bars):
        raise ValueError("Alpaca returned no option bars for the window.")
    rates = risk_free(irx, closes.index)
    panel = contract_days(contracts, bars, closes, settings["dividend_yield"], rates)
    smiles = fit_smiles(panel)
    assigned = assign_smiles(panel, smiles)
    panel = add_fitted_iv(panel, assigned)
    panel["oi"] = scaled_oi(panel, settings)
    return positioning_tables(panel, smiles, contracts, closes, rates)


# --- positioning rows, one per previous close -------------------------------------------

POSITIONING_LAYOUT = 1
ROW_FIELDS = [
    "spot", "call_wall_oi", "put_wall_oi", "call_wall_gex", "put_wall_gex", "call_wall_oi_7",
    "put_wall_oi_7", "net_gex", "gross_gex", "gex_ratio", "gamma_flip", "atm_iv_30", "iv_1d",
    "iv_5d", "call_volume", "put_volume", "opt_volume_coverage", "call_oi", "put_oi",
    "max_pain_std", "days_to_monthly_opex",
]
# The contracts that expired this many calendar days before a span's first
# close are still read: the volume coverage takes the median of the 20
# sessions before each close, and a contract counts on every day it traded.
_COVERAGE_REACH_DAYS = 45
# Closes further apart than this are built separately: the session closes are
# read 220 calendar days back from a span's last close, and they must reach
# over the windows of the contracts read for its first.
_SPAN_DAYS = 40
_lock = threading.Lock()


def _positioning_path(symbol: str):
    return op.HISTORY_DIR / f"{symbol.upper()}_hl2_positioning.json"


def _settings(symbol: str) -> dict:
    settings = NOTEBOOK_OPTIONS.get(symbol.upper())
    if settings is None:
        raise ValueError(
            f"no HighLow2_5m option settings for {symbol} (`highlow2_options.NOTEBOOK_OPTIONS`): "
            "its dividend yield and open-interest calibration are what the notebook priced with."
        )
    return settings


def _signature(settings: dict) -> dict:
    return {k: (str(v) if isinstance(v, float) and not math.isfinite(v) else v)
            for k, v in settings.items()}


def _spans(days: "list[pd.Timestamp]") -> "list[tuple[pd.Timestamp, pd.Timestamp]]":
    """Sorted days -> (first, last) runs no wider than _SPAN_DAYS."""
    spans: "list[list[pd.Timestamp]]" = []
    for day in days:
        if spans and (day - spans[-1][0]).days <= _SPAN_DAYS:
            spans[-1][1] = day
        else:
            spans.append([day, day])
    return [(a, b) for a, b in spans]


def _row(table: pd.DataFrame, day: pd.Timestamp) -> dict:
    return {c: float(table.at[day, c]) for c in ROW_FIELDS}


def _build_rows(symbol: str, lo: pd.Timestamp, hi: pd.Timestamp, settings: dict,
                key: "str | None", secret: "str | None") -> "dict[str, tuple[dict, bool]]":
    """Every close in [lo, hi] -> (row, whether it may be kept)."""
    key, secret = op._credentials(key, secret)
    closes = op.sip_closes(symbol, hi + pd.Timedelta(1, "D"), key, secret)
    closes = closes[closes.index <= hi]
    days = closes.index[(closes.index >= lo) & (closes.index <= hi)]
    if not len(days):
        return {}
    contracts, bars = op.option_history(
        symbol, lo - pd.Timedelta(_COVERAGE_REACH_DAYS, "D"), hi + pd.Timedelta(OPTION_MAX_DTE, "D"),
        closes, hi, key, secret,
    )
    if not len(contracts):
        raise ValueError(f"Alpaca lists no {symbol} option contracts expiring after {lo.date()}.")
    irx = op.treasury_yield(closes.index.min(), hi)
    if not len(irx):
        raise ValueError("no 13-week T-bill yield (Yahoo ^IRX) to price the options with.")
    table = positioning_from(contracts, bars, closes, irx, settings)
    last_rate = irx.index.max()
    return {f"{d:%Y-%m-%d}": (_row(table, d), last_rate >= d) for d in days}


def positioning_rows(
    symbol: str,
    days,
    key: "str | None" = None,
    secret: "str | None" = None,
) -> "dict[str, dict]":
    """HighLow2_5m's positioning row of each close in `days` (option sessions,
    half days included), from the cache or built.

    A row for day d reads the contracts alive on d and over the 20 sessions
    before it, each over its last 120 days up to d, d's SIP close and the
    T-bill yield on d -- nothing later, so a built row is kept for good. It is
    only built for a day before today (a session still trading has no close)
    and only kept once ^IRX has a close dated d or later.
    """
    symbol = symbol.upper()
    settings = _settings(symbol)
    wanted = sorted({pd.Timestamp(d).normalize() for d in days})
    if not wanted:
        return {}
    today = pd.Timestamp(op._today_et())
    if wanted[-1] >= today:
        raise ValueError(
            f"the option table of {wanted[-1].date()} is not final until that session is over."
        )
    with _lock:
        cache = op._read_json(_positioning_path(symbol)) or {}
        if cache.get("layout") != POSITIONING_LAYOUT or cache.get("settings") != _signature(settings):
            cache = {"layout": POSITIONING_LAYOUT, "settings": _signature(settings), "rows": {}}
        rows = cache["rows"]
        missing = [d for d in wanted if f"{d:%Y-%m-%d}" not in rows]
        out = {}
        for lo, hi in _spans(missing):
            for iso, (row, final) in _build_rows(symbol, lo, hi, settings, key, secret).items():
                if final:
                    rows[iso] = row
                out[iso] = row
        if missing:
            op._write_json(_positioning_path(symbol), cache)
        out.update({iso: rows[iso] for iso in rows if iso not in out})
    return {f"{d:%Y-%m-%d}": out[f"{d:%Y-%m-%d}"] for d in wanted if f"{d:%Y-%m-%d}" in out}


def previous_session(day) -> pd.Timestamp:
    """The option session before `day`: the last NYSE session, half days
    included -- whose table `features.previous_row` hands a session."""
    d64 = np.datetime64(pd.Timestamp(day).date(), "D")
    return pd.Timestamp(np.busday_offset(d64, -1, roll="forward", holidays=_holidays()))


def frame(rows: "dict[str, dict]") -> pd.DataFrame:
    """Cached rows -> the notebook's positioning table (one row per close)."""
    if not rows:
        return pd.DataFrame(columns=ROW_FIELDS, index=pd.DatetimeIndex([], name="date"), dtype=float)
    dates = sorted(rows)
    return pd.DataFrame([{c: rows[d][c] for c in ROW_FIELDS} for d in dates], index=op._dates(dates))
