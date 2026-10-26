"""What GoodMoments' NVDA and MU models read beyond their own stock's bars.

The notebook's second passes (`goodmoments/nvda.py`, `goodmoments/mu.py`)
added features of each stock's own to the 55 shared ones, and notebook 5 made
MU's two volatility clocks shared code (`features.vol_clocks`). The saved
models read only some of them:

    NVDA (38 features, 12 its own)
        market   qqq_today, qqq_rv_sofar, attention     QQQ's session so far,
                                                         NVDA's volume against QQQ's
        semis    resid_smh_today, rs_smh_20d, peers_today
                                                         SMH's session and NVDA's 20-day
                                                         strength against it; the day of
                                                         AMD, AVGO, TSM and MU
        earnings days_to_earn, days_since_earn          sessions to the next report, since
                                                         the last
        surface  call_vol_rel, d_pin                    last night's call volume against
                                                         its norm, the nearest expiry's pin
        levels   to_5_up                                the next $5 level above
        participation trade_size_15m                    the 15-minute average trade size
                                                         against its norm
    MU (3 features)
        frac_left and the two clocks: `p_reach` with its volatility multiplier
        read off MU's last hour (`p_reach_60m`) or off SMH's session so far
        (`p_reach_smh`) -- the first is MU's own tape, in `goodmoments_model`;
        the second reads SMH.

This module holds the inputs those need: the other symbols' tapes (history and
today's, cut at the same SIP/IEX seam as the stock's own -- each context
symbol's IEX volume put on its own SIP scale), the earnings clock, the
nearest expiry's pin and the call-volume norm (folded into last night's option
row, `goodmoments_model.positioning`), and each bar's trade count (Alpaca's
bars carry it, the shared tape cache drops it, so SIP's is cached here).

The live tape departs from the notebook here as for the stock's own features:
QQQ's, SMH's and the peers' last 15 minutes are IEX's. A context symbol that
has not printed on IEX in the bar being forecast carries its last print (the
notebook's fill of a minute with no trade). The 15-minute trade size is the
one exception to the joined tape: it is SIP's last 15 bars, 15 minutes back,
read against that minute's norm (`goodmoments_model.session_stats` says why),
and missing before 09:45.

The mirror contract is `goodmoments_model`'s: same arithmetic, in the same
order, pinned against the notebook's panel at seam 0 by
`tests/test_goodmoments_model.py`.
"""

from __future__ import annotations

import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from . import clock, market_hours

TZ = "America/New_York"

# --- which features need what -------------------------------------------------------

# Each ticker's own features this module computes (the shipped models' picks
# from `goodmoments.nvda.FEATURES`; MU's two clocks are shared columns).
OWN_FEATURES = {
    "NVDA": (
        "qqq_today", "qqq_rv_sofar", "attention", "resid_smh_today", "rs_smh_20d", "peers_today",
        "days_to_earn", "days_since_earn", "call_vol_rel", "d_pin", "to_5_up", "trade_size_15m",
    ),
}
# NVDA's peers (`nvda.PEERS`).
PEERS = {"NVDA": ("AMD", "AVGO", "TSM", "MU")}
# Which context symbol each feature reads.
_READS = {
    "p_reach_smh": ("SMH",),
    "qqq_today": ("QQQ",), "qqq_rv_sofar": ("QQQ",), "attention": ("QQQ",),
    "resid_smh_today": ("SMH",), "rs_smh_20d": ("SMH",),
}
EARNINGS_COLS = {"days_to_earn", "days_since_earn"}
SURFACE_COLS = {"call_vol_rel", "d_pin"}
TRADES_COLS = {"trade_size_15m"}

# How a context symbol's sessions were cleaned: NVDA's notebook kept only full
# sessions of its context series (`nvda.load_context`, the shared rule); MU's
# keeps every session's 390 minutes however thin (`data.context_minutes`,
# since LRCX went minutes without a trade before its split).
CONTEXT_MIN_BARS = {"NVDA": 380}

EARN_HORIZON = 15             # sessions; further reports count as 15
EARN_SINCE_CAP = 65           # about a quarter
# Asked of the calendar past the session: more than EARN_HORIZON sessions.
EARN_LOOKAHEAD_DAYS = 45
# Reached back for the call-volume norm's 20 sessions: the contracts that
# expired over them traded on those days too.
CALL_NORM_DAYS = 45

# NVDA's 10-for-1 split of 10 June 2024 (`nvda.SPLIT_DATE` ...): option tables
# and trade-count norms are not comparable across it. Nothing the app replays
# reaches back that far; kept so the mirror is the notebook's on every day.
NVDA_SPLIT_DATE = pd.Timestamp("2024-06-10")
NVDA_OPTIONS_FROM = pd.Timestamp("2024-06-11")
NVDA_OI_FROM = pd.Timestamp("2024-07-10")
NVDA_NORM_FROM = pd.Timestamp("2024-07-09")


@dataclass(frozen=True)
class Needs:
    """What a bundle's features read beyond the stock's own tape and options."""

    context: "tuple[str, ...]" = ()
    earnings: bool = False
    surface: bool = False
    trades: bool = False

    @property
    def any(self) -> bool:
        return bool(self.context or self.earnings or self.surface or self.trades)


def needs(ticker: str, cols) -> Needs:
    """What the features `cols` of `ticker`'s bundle read here."""
    cols = set(cols)
    symbols: "list[str]" = []
    for col in sorted(cols):
        for symbol in _READS.get(col, ()):
            if symbol not in symbols:
                symbols.append(symbol)
    if "peers_today" in cols:
        symbols += [p for p in PEERS.get(ticker, ()) if p not in symbols]
    return Needs(
        context=tuple(symbols),
        earnings=bool(cols & EARNINGS_COLS),
        surface=bool(cols & SURFACE_COLS),
        trades=bool(cols & TRADES_COLS),
    )


def _gm():
    from . import goodmoments_model

    return goodmoments_model


def _adr(daily: pd.DataFrame) -> pd.Series:
    """`nvda._adr`: the mean log range of the 14 sessions before each."""
    return np.log(daily["high"] / daily["low"]).rolling(14).mean().shift(1)


# --- a context symbol's history ----------------------------------------------------------


@dataclass
class ContextDay:
    """One context symbol as the session's forecasts read it from before it."""

    symbol: str
    daily: pd.DataFrame          # its candles before the day, plus a row for it
    adr: float                   # its own ADR on the day
    norms: dict                  # same-minute norms over the stock's kept sessions
    k: float                     # its SIP/IEX volume ratio


def context_day(symbol: str, minutes: pd.DataFrame, daily: pd.DataFrame, kept: pd.DatetimeIndex,
                day: pd.Timestamp, k: float = float("nan")) -> ContextDay:
    """A context symbol's day inputs, from its cleaned grid and candles before
    `day`; `kept` are the stock's own kept sessions, the norms' sessions
    (`_norm_by_minute(x, context, kept)` in the notebook)."""
    G = _gm()
    before = minutes[minutes["date"] < day]
    dates = pd.DatetimeIndex(sorted(before["date"].unique()))
    report = pd.DataFrame({"kept": dates.isin(kept)}, index=dates)
    norms = G.minute_norms(before, report, day)
    dly = pd.concat([daily[daily.index < day], pd.DataFrame(index=pd.DatetimeIndex([day], name="date"))])
    return ContextDay(symbol=symbol, daily=dly, adr=float(_adr(dly).loc[day]), norms=norms, k=k)


def context_stats(ctx: ContextDay, sip, iex, seam: int) -> dict:
    """The context symbol's session so far at every minute, on the tape the
    stock's own is read on (`goodmoments_model.session_stats`)."""
    return _gm().session_stats(sip, iex, ctx.k, 1.0, seam)


# --- the day-level values ------------------------------------------------------------------


def smh_daily_values(daily: pd.DataFrame, smh: pd.DataFrame, day: pd.Timestamp) -> dict:
    """NVDA's 60-session beta on SMH and its 20-session strength against it,
    as of the close before `day` (`nvda.build`'s `beta_smh`, `rs_smh_20d`).
    Both frames carry a row for `day`."""
    adr_d = _adr(daily)
    rn = np.log(daily["close"]).diff()
    rs = np.log(smh["close"]).diff().reindex(daily.index)
    beta = (rn.rolling(60).cov(rs) / rs.rolling(60).var()).shift(1)
    n = 20
    rel = np.log(daily["close"] / daily["close"].shift(n)) - np.log(
        smh["close"] / smh["close"].shift(n)).reindex(daily.index)
    return {"beta_smh": float(beta.loc[day]), "rs_smh_20d": float((rel.shift(1) / (adr_d * np.sqrt(n))).loc[day])}


def sessions_around(day: pd.Timestamp, back_days: int = 140, ahead_days: int = EARN_LOOKAHEAD_DAYS) -> pd.DatetimeIndex:
    """NYSE sessions (half days included) from `back_days` before `day` to
    `ahead_days` after it: the session clock the earnings features count on."""
    from .fastfurious_model import nyse_sessions

    span = pd.date_range(day - pd.Timedelta(days=back_days), day + pd.Timedelta(days=ahead_days), freq="D")
    return pd.DatetimeIndex(nyse_sessions(span)).as_unit("ns")


def report_sessions(stamps, sessions: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """The session that first trades on each report: the next one for a
    report at or after 16:00 (`nvda.earnings`)."""
    if stamps is None or not len(stamps):
        return pd.DatetimeIndex([])
    t = pd.Series(pd.to_datetime(list(stamps), utc=True)).dt.tz_convert(TZ)
    day = t.dt.tz_localize(None).dt.normalize().dt.as_unit("ns")
    after_close = t.dt.hour * 60 + t.dt.minute >= 16 * 60
    keep = (day >= sessions[0]).to_numpy()
    pos = np.searchsorted(sessions.to_numpy(), day.to_numpy()[keep], side="left") + after_close.to_numpy()[keep]
    pos = pos[pos < len(sessions)]
    return pd.DatetimeIndex(sorted(set(sessions[pos])))


def earnings_clock(earn: pd.DatetimeIndex, sessions: pd.DatetimeIndex, day: pd.Timestamp) -> dict:
    """Sessions from `day` to the next report session (0 on it, at most 15)
    and since the last (at most 65), on the session list (`nvda.build`)."""
    idx = sessions
    pos = np.array([idx.get_loc(day)])
    e_pos = np.searchsorted(idx.to_numpy(), pd.DatetimeIndex(earn).as_unit("ns").to_numpy())
    e_pos = e_pos[(e_pos < len(idx))]
    nxt = np.searchsorted(e_pos, pos, side="left")
    to = np.where(nxt < len(e_pos), e_pos[np.minimum(nxt, len(e_pos) - 1)] - pos, EARN_HORIZON) if len(e_pos) \
        else np.array([EARN_HORIZON])
    prv = np.searchsorted(e_pos, pos, side="right") - 1
    since = np.where(prv >= 0, pos - e_pos[np.maximum(prv, 0)], EARN_SINCE_CAP) if len(e_pos) \
        else np.array([EARN_SINCE_CAP])
    return {"days_to_earn": float(np.minimum(to, EARN_HORIZON)[0]),
            "days_since_earn": float(np.minimum(since, EARN_SINCE_CAP)[0])}


# --- the option surface (mirrors goodmoments.nvda.surface) ----------------------------------


def surface_row(bars: pd.DataFrame, daily: pd.DataFrame, sessions: pd.DatetimeIndex,
                prev: pd.Timestamp, day: pd.Timestamp, half_life: float) -> dict:
    """`nvda.surface`'s row for `day`, from the option bars through `prev`:
    the nearest expiry's pin (its strike of most open interest) and the call
    volume of `prev` against its 20-session mean, in logs. Both NaN where the
    notebook has no row (no expiry on or after the day with open interest)."""
    from scipy.signal import lfilter

    out = {"pin": float("nan"), "call_vol_rel": float("nan")}
    b = bars.astype({"close": float, "volume": float}).merge(
        daily["close"].rename("spot"), left_on="date", right_index=True)
    is_call = (b["type"] == "call").to_numpy()
    calls = b[is_call]
    cv = calls.groupby("date")["volume"].sum()
    call_vol_rel = np.log(cv / cv.rolling(20, min_periods=10).mean().shift(1))

    vol = b.pivot_table(index="date", columns="symbol", values="volume", aggfunc="sum")
    vol = vol.reindex(sessions[(sessions >= vol.index.min()) & (sessions <= vol.index.max())]).fillna(0.0)
    if prev not in vol.index:
        return out
    decay = 0.5 ** (1.0 / half_life)
    oi = pd.DataFrame(lfilter([1.0], [1.0, -decay], vol.to_numpy(), axis=0), index=vol.index, columns=vol.columns)
    meta = b.drop_duplicates("symbol").set_index("symbol")[["type", "strike", "expiry"]].reindex(oi.columns)
    expiries = np.sort(meta["expiry"].unique())
    i = np.searchsorted(expiries, np.datetime64(day))
    if i == len(expiries):
        return out
    e = pd.Timestamp(expiries[i])
    c = meta[meta["expiry"] == e].assign(oi=oi.loc[prev, meta.index[meta["expiry"] == e]].to_numpy())
    c = c[c["oi"] > 0]
    if c.empty:
        return out
    out["pin"] = float(c.groupby("strike")["oi"].sum().idxmax())
    out["call_vol_rel"] = float(call_vol_rel.get(prev, np.nan))
    return out


# --- trade counts ------------------------------------------------------------------------------
#
# Only SIP's: the live 15-bar trade size is read off SIP 15 bars back
# (`goodmoments_model.session_stats`), so IEX's counts are never needed.

TRADES_LAYOUT = 1
_trades_lock = threading.Lock()


def _trades_dir() -> Path:
    return _gm().CACHE_DIR / "trades"


def _trades_path(symbol: str) -> Path:
    return _trades_dir() / f"{symbol.upper()}_sip.parquet"


def _trades_index_path(symbol: str) -> Path:
    return _trades_dir() / f"{symbol.upper()}.json"


def _empty_trades() -> pd.Series:
    return pd.Series([], index=pd.DatetimeIndex([], tz=market_hours.MARKET_TZ, name="ts"),
                     dtype=float, name="trades")


def _fetch_trades(symbol: str, start: datetime, end: datetime, key: str, secret: str) -> pd.Series:
    """SIP's trade count per regular-session minute bar in [start, end]."""
    from .rest import fetch_bars_range

    bars = fetch_bars_range(symbol, "1Min", start, end, key, secret, feed="sip", adjustment="split")
    if not bars:
        return _empty_trades()
    idx = pd.to_datetime([b["t"] for b in bars], utc=True, format="ISO8601").tz_convert(market_hours.MARKET_TZ)
    out = pd.Series([float(b.get("n") or 0.0) for b in bars], index=pd.DatetimeIndex(idx, name="ts"), name="trades")
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    minute = out.index.hour * 60 + out.index.minute - (9 * 60 + 30)
    return out[(minute >= 0) & (minute < 390)]


def trade_history(symbol: str, first: date, before: date,
                  key: "str | None" = None, secret: "str | None" = None) -> pd.Series:
    """SIP's trade counts per minute for the calendar days in [first,
    before), cached under `data/goodmoments/trades/`. The cache keeps one
    contiguous stretch and fetches only what lies outside it, as the shared
    tape cache does (`fastfurious_model.raw_history`); a split does not change
    a count, so nothing is ever rebuilt."""
    from .fastfurious_model import _credentials

    symbol = symbol.upper()
    last_wanted = before - timedelta(days=1)
    tz = market_hours.MARKET_TZ
    with _trades_lock:
        index = _gm()._read_json(_trades_index_path(symbol)) or {}
        series = None
        if index.get("layout") == TRADES_LAYOUT and index.get("from") and index.get("through"):
            try:
                series = pd.read_parquet(_trades_path(symbol))["trades"]
            except Exception:  # noqa: BLE001 -- a broken file is an empty cache
                series = None
        lo = date.fromisoformat(index["from"]) if series is not None else None
        hi = date.fromisoformat(index["through"]) if series is not None else None
        if lo is None or lo > first or hi < last_wanted:
            key, secret = _credentials(key, secret)

            def fetch(a: date, b: date) -> pd.Series:
                start = datetime.combine(a, datetime.min.time(), tzinfo=tz)
                end = datetime.combine(b, datetime.min.time(), tzinfo=tz)
                return _fetch_trades(symbol, start, end, key, secret)

            if lo is None:
                series, lo, hi = fetch(first, before), first, last_wanted
            else:
                parts = [series]
                if lo > first:
                    parts.append(fetch(first, lo))
                if hi < last_wanted:
                    parts.append(fetch(hi + timedelta(days=1), before))
                series = pd.concat(parts).sort_index()
                series = series[~series.index.duplicated(keep="last")]
                lo, hi = min(lo, first), max(hi, last_wanted)
            _trades_dir().mkdir(parents=True, exist_ok=True)
            path = _trades_path(symbol)
            tmp = path.with_suffix(".tmp")
            series.to_frame("trades").to_parquet(tmp)
            tmp.replace(path)
            _gm()._write_json(_trades_index_path(symbol),
                              {"layout": TRADES_LAYOUT, "from": lo.isoformat(), "through": hi.isoformat()})
            from .datalog import log_fetch

            log_fetch("trade counts (GoodMoments)", "Alpaca REST (SIP minute bars, n)", symbol=symbol,
                      detail=f"cached {lo} to {hi}")
    stamp = series.index.tz_convert(tz).normalize().tz_localize(None)
    return series[(stamp >= pd.Timestamp(first)) & (stamp < pd.Timestamp(before))]


def live_trades(symbol: str, day: date, through: int,
                key: "str | None" = None, secret: "str | None" = None) -> pd.Series:
    """Today's SIP trade counts through bar `through - 15`, the SIP part of
    the live tape (none before bar 15)."""
    from .fastfurious_model import L, _bar_time, _credentials

    if through - L < 0:
        return _empty_trades()
    key, secret = _credentials(key, secret)
    return _fetch_trades(symbol, _bar_time(day, 0), _bar_time(day, through - L), key, secret)


def with_trades(raw: pd.DataFrame, trades: "pd.Series | None") -> pd.DataFrame:
    """Raw bars with their trade count beside them (0 where none was cached)."""
    if trades is None:
        return raw
    out = raw.copy()
    out["trades"] = trades.reindex(out.index).fillna(0.0).to_numpy(float)
    return out


# --- the earnings dates ------------------------------------------------------------------------

_earnings_lock = threading.Lock()


def _earnings_path(symbol: str) -> Path:
    return _gm().CACHE_DIR / f"{symbol.upper()}_earnings.json"


def earnings_stamps(symbol: str, day: date) -> list:
    """Yahoo's report times for `symbol` as known on `day`
    (`event_days.fetch_earnings_stamps`, the seam SimLab patches to its kept
    copy), the last good read kept on disk for a day Yahoo fails. Raises
    ValueError when there is neither."""
    from . import event_days

    path = _earnings_path(symbol)
    with _earnings_lock:
        try:
            stamps = event_days.fetch_earnings_stamps(symbol.upper(), day)
        except Exception:  # noqa: BLE001 -- the kept copy stands in
            stamps = None
        if stamps:
            iso = [pd.Timestamp(s).isoformat() for s in stamps]
            _gm()._write_json(path, {"symbol": symbol.upper(), "dates": iso, "read_on": day.isoformat()})
            return iso
        kept = _gm()._read_json(path) or {}
        if kept.get("dates"):
            return list(kept["dates"])
    raise ValueError(f"no earnings dates for {symbol.upper()} (Yahoo), and none kept from an earlier read.")


# --- one session's own features -----------------------------------------------------------------


@dataclass
class OwnDay:
    """What the stock's own features read from before the session."""

    ticker: str
    day: pd.Timestamp
    context: "dict[str, ContextDay]" = field(default_factory=dict)
    values: dict = field(default_factory=dict)    # day-level: beta, strength, earnings clock


@dataclass
class OwnSession:
    """The own features' inputs for one session's tape: the day's inputs, the
    context symbols' session stats and last night's surface (pin, call
    volume) from the option row."""

    own: OwnDay
    stats: "dict[str, dict]" = field(default_factory=dict)
    position: "dict | None" = None

    @property
    def smh_rv(self) -> "np.ndarray | None":
        """SMH's realised volatility so far against the same minutes of the
        stock's recent sessions: `p_reach_smh`'s multiplier."""
        ctx, s = self.own.context.get("SMH"), self.stats.get("SMH")
        if ctx is None or s is None:
            return None
        return np.sqrt(s["cum_var"]) / ctx.norms["cum_rv"]

    def _today(self, symbol: str) -> np.ndarray:
        s, ctx = self.stats[symbol], self.own.context[symbol]
        return np.log(s["c"] / s["s_open"]) / ctx.adr

    def add(self, f: pd.DataFrame, s: dict, norms: dict) -> None:
        """Write the stock's own features into `f` (`goodmoments_model.session_features`'
        frame), from its session stats `s` and same-minute `norms`. A context
        symbol whose tape could not be read leaves its features NaN."""
        if self.own.ticker != "NVDA":
            return
        nan = np.full(len(f), np.nan)
        adr = float(f["adr"].iloc[0])
        c = s["c"]

        def dist(ref):
            return np.log(c / ref) / adr

        ret_today = np.log(c / s["s_open"])
        v = self.own.values
        f["qqq_today"] = f["qqq_rv_sofar"] = f["attention"] = nan
        if "QQQ" in self.stats:
            q, qc = self.stats["QQQ"], self.own.context["QQQ"]
            f["qqq_today"] = self._today("QQQ")
            f["qqq_rv_sofar"] = np.sqrt(q["cum_var"]) / qc.norms["cum_rv"]
            with np.errstate(invalid="ignore", divide="ignore"):
                f["attention"] = np.log(f["rel_vol"].to_numpy() / (q["cum_vol"] / qc.norms["cum_vol"]))
        f["resid_smh_today"] = nan
        if "SMH" in self.stats:
            sm = self.stats["SMH"]
            f["resid_smh_today"] = (ret_today - v.get("beta_smh", np.nan) * np.log(sm["c"] / sm["s_open"])) / adr
        f["rs_smh_20d"] = v.get("rs_smh_20d", np.nan)
        peers = [self._today(p) for p in PEERS["NVDA"] if p in self.stats]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)       # every peer missing: NaN
            f["peers_today"] = np.nanmean(peers, axis=0) if peers else nan
        f["days_to_earn"] = v.get("days_to_earn", np.nan)
        f["days_since_earn"] = v.get("days_since_earn", np.nan)
        pos = self.position or {}
        f["call_vol_rel"] = float(pos.get("call_vol_rel", np.nan))
        f["d_pin"] = dist(float(pos.get("pin", np.nan)))
        f["to_5_up"] = np.log((np.floor(c / 5) + 1) * 5 / c) / adr
        f["trade_size_15m"] = nan
        if "size_15" in s and "size_15" in norms:
            at = s["size_at"]
            norm = np.where(at >= 0, norms["size_15"][np.maximum(at, 0)], np.nan)
            with np.errstate(invalid="ignore", divide="ignore"):
                f["trade_size_15m"] = s["size_15"] / norm

        # The split: option tables, open interest and trade-count norms not yet comparable.
        day = self.own.day
        if day < NVDA_OPTIONS_FROM:
            for col in ("call_vol_rel", "d_pin", "d_call_wall", "d_put_wall", "gex", "iv_adr", "pc_volume"):
                f[col] = np.nan
        if day < NVDA_OI_FROM:
            for col in ("d_pin", "d_call_wall", "d_put_wall", "gex"):
                f[col] = np.nan
        if NVDA_SPLIT_DATE <= day < NVDA_NORM_FROM:
            f["trade_size_15m"] = f["call_vol_rel"] = np.nan


# --- building them for the app --------------------------------------------------------------------


def _context_history(symbol: str, ticker: str, day: pd.Timestamp, kept: pd.DatetimeIndex,
                     key, secret) -> ContextDay:
    G = _gm()
    before = day.date()
    raw = G._ff().raw_history(symbol, before - timedelta(days=G.HISTORY_CALENDAR_DAYS), before, key, secret)
    minutes, report = G.clean_sessions(raw["sip"], min_bars=CONTEXT_MIN_BARS.get(ticker, 0))
    if not len(minutes):
        raise ValueError(f"no SIP history for {symbol} before {day.date()} (a {ticker} GoodMoments input).")
    # IEX's volume on SIP's scale, over the stock's own kept sessions.
    scale_report = pd.DataFrame({"kept": report.index.isin(kept)}, index=report.index)
    k = G.iex_scale(raw["sip"], raw["iex"], scale_report, day)
    return context_day(symbol, minutes, G.daily_candles(minutes), kept, day, k)


def own_day(need: Needs, symbol: str, day: pd.Timestamp, report: pd.DataFrame,
            daily: pd.DataFrame, key, secret) -> "OwnDay | None":
    """The day's inputs for a bundle's own features (None when it reads
    none): every context symbol's history and the day-level values. The
    trade counts are joined to the stock's own history (`trade_history`)."""
    if not (need.context or need.earnings):
        return None
    kept = pd.DatetimeIndex(report.index[report["kept"] & (report.index < day)])
    own = OwnDay(ticker=symbol, day=day)
    if need.context:
        with ThreadPoolExecutor(max_workers=min(6, len(need.context))) as pool:
            work = clock.inherit(lambda sym: (sym, _context_history(sym, symbol, day, kept, key, secret)))
            own.context = dict(pool.map(work, need.context))
    if "SMH" in own.context and symbol in OWN_FEATURES:
        dly = pd.concat([daily[daily.index < day], pd.DataFrame(index=pd.DatetimeIndex([day], name="date"))])
        own.values.update(smh_daily_values(dly, own.context["SMH"].daily, day))
    if need.earnings:
        sessions = sessions_around(day)
        own.values.update(earnings_clock(report_sessions(earnings_stamps(symbol, day.date()), sessions),
                                         sessions, day))
    return own


def session_context(own: "OwnDay | None", day: pd.Timestamp, last: int, live: bool,
                    position: "dict | None", key, secret, seam: int) -> "OwnSession | None":
    """The context symbols' tapes for the session through bar `last` (live:
    fetched now, without waiting for a bar IEX has not printed yet; a replay:
    the whole day from the shared cache), cut at the seam."""
    if own is None:
        return None
    G = _gm()
    ff = G._ff()

    def one(sym):
        if live:
            raw_sip, raw_iex, _ = ff.live_tapes(sym, day.date(), last, key, secret, retries=0)
        else:
            raw = ff.raw_history(sym, day.date(), day.date() + timedelta(days=1), key, secret)
            raw_sip, raw_iex = raw["sip"], raw["iex"]
        iex = G.session_tape(raw_iex)
        if iex is None:
            return sym, None
        sip = G.session_tape(raw_sip) or iex
        return sym, context_stats(own.context[sym], sip, iex, seam)

    stats = {}
    if own.context:
        with ThreadPoolExecutor(max_workers=min(6, len(own.context))) as pool:
            for sym, found in pool.map(clock.inherit(one), list(own.context)):
                if found is not None:
                    stats[sym] = found
    return OwnSession(own=own, stats=stats, position=position)
