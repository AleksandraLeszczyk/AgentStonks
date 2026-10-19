"""Transaction aggressiveness per bar: which side of the spread the money
traded on.

    aggressiveness = (value traded closer to the ask - value traded closer to the bid)
                     / value of every trade

A trade's value is its price times its size, and "closer" is measured against
the quote prevailing when it printed: above the midpoint is closer to the ask
(a buyer paid up), below it closer to the bid (a seller gave in). +1 is a bar
where every dollar lifted the offer, -1 one where every dollar hit the bid, 0
a balanced bar -- or one that traded at the midpoint.

- **The prevailing quote** is the latest quote stamped strictly before the
  trade. A trade that takes out the ask moves the quote at the same instant,
  and that new quote -- the ask a tick up, the bid where the trade printed --
  would read the buy as a sale. A trade with no quote in the QUOTE_LEAD_SEC
  before it, or under a one-sided or crossed quote, is left out.
- **Trades at the midpoint** go to neither side but count in the total.
- **Prints that did not trade against the book at that moment are left out of
  all three sums** (`SKIP_CONDITIONS`): auction crosses, late and out-of-
  sequence reports, average-price and derivatively priced prints, official
  open/close reports. The opening cross alone would otherwise outweigh the
  first minute, at a quote from before the open.
- **Where the ticks come from.** Alpaca REST, trades and quotes of the same
  feed for the same minutes: the consolidated tape (SIP) once a minute is
  SIP_DELAY_MIN old -- the free plan's SIP stops that far back -- and IEX for
  the minutes since, each replaced by SIP as it ages, like the bars
  (`bar_history`). Not from the live socket: Finnhub's carries no quotes, and
  the app polls bid/ask every FALLBACK_POLL_SEC there.
- **Kept per minute,** as the three sums, so a bar of any size is the sum of
  its minutes. The SIP minutes are written under ``data/aggressiveness/`` (one
  file per symbol and ET day): a reload or a restart draws them again without
  a fetch. IEX minutes are only held in memory -- they are replaced anyway.
- **Regular session only,** like the momentum panels it sits under, and only
  the latest bar's ET day.
- **Fetched only while the panel is drawn,** in the background: `series` reads
  what is kept and starts one worker per symbol for what is missing. Newest
  minutes first, so after a mid-day start the bars nearest now fill in first.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import clock
from .config import SIP_DELAY_MIN
from .market_hours import MARKET_CLOSE, MARKET_OPEN, MARKET_TZ
from .rest import fetch_ticks
from .stream_common import TF_MINUTES

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "aggressiveness"

# Trade conditions (CTS and UTDF codes, as Alpaca passes them through) of
# prints that are not an execution against the book at the time they print.
SKIP_CONDITIONS = frozenset({
    "B",  # CTS: average price trade / UTDF: bunched trade (reported late)
    "C",  # cash sale
    "G",  # bunched sold trade
    "H",  # price variation trade
    "L",  # sold last (late reporting)
    "M",  # market center official close (a report, not a trade)
    "N",  # next day
    "O",  # opening prints (the auction cross)
    "P",  # prior reference price
    "Q",  # market center official open (a report, not a trade)
    "R",  # seller
    "U",  # extended hours, sold out of sequence
    "V",  # contingent trade
    "W",  # average price trade
    "X",  # cross trade
    "Z",  # sold out of sequence
    "4",  # derivatively priced
    "5",  # re-opening prints (the auction after a halt)
    "6",  # closing prints (the auction cross)
    "7",  # qualified contingent trade
    "9",  # corrected consolidated close
})

# How far before a window its quotes are fetched from, so the window's first
# trades have a quote before them.
QUOTE_LEAD_SEC = 30
# A minute is fetched as complete this long after it ends, giving the REST
# tape time to hold all of its prints.
SETTLE_SEC = 5
# How often the IEX minutes -- the newest SIP_DELAY_MIN -- are re-fetched.
REFRESH_SEC = 25
# SIP minutes are fetched this many at a time, newest chunk first.
CHUNK_MIN = 10
# A pause between requests: a mid-day start backfills a few hundred minutes,
# and the market-data rate limit is shared with the rest of the app.
REQUEST_GAP_SEC = 0.3
# After a failed fetch, wait this long before trying again.
RETRY_SEC = 60

_lock = threading.Lock()
# (symbol, ET date) -> {minute start, UTC ISO: [ask $, bid $, total $, trades, tape]}.
_kept: dict[tuple[str, str], dict[str, list]] = {}
# (symbol, ET date) -> the regular-session minutes the chart draws, newest last.
_wanted: dict[tuple[str, str], list[pd.Timestamp]] = {}
_running: set[str] = set()
# symbol -> monotonic time of the last IEX refresh, of the next try after a failure.
_iex_at: dict[str, float] = {}
_retry_at: dict[str, float] = {}
_errors: dict[str, str] = {}


def _utc(t) -> pd.Timestamp:
    stamp = pd.Timestamp(t)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _is_rth(minute: pd.Timestamp) -> bool:
    et = minute.tz_convert(MARKET_TZ).time()
    return MARKET_OPEN <= et < MARKET_CLOSE


def _stamps(ticks: "list[dict]") -> pd.DatetimeIndex:
    """The ticks' times, UTC, at nanoseconds whatever their strings' precision
    -- `merge_asof` refuses keys of two resolutions."""
    return pd.to_datetime([x["t"] for x in ticks], utc=True, format="ISO8601").as_unit("ns")


def classify(trades: "list[dict]", quotes: "list[dict]") -> pd.DataFrame:
    """Each usable trade with its side against the prevailing quote: a frame
    of `t` (UTC), `value` ($) and `side` (+1 closer to the ask, -1 closer to
    the bid, 0 at the midpoint). Alpaca REST trade ({t, p, s, c}) and quote
    ({t, bp, ap}) dicts, in any order."""
    empty = pd.DataFrame({"t": pd.Series(dtype="datetime64[ns, UTC]"),
                          "value": pd.Series(dtype=float), "side": pd.Series(dtype=int)})
    if not trades or not quotes:
        return empty
    tr = pd.DataFrame({
        "t": _stamps(trades),
        "p": [float(x.get("p") or 0.0) for x in trades],
        "s": [float(x.get("s") or 0.0) for x in trades],
        "skip": [bool(SKIP_CONDITIONS.intersection(x.get("c") or ())) for x in trades],
    })
    tr = tr[~tr["skip"] & (tr["p"] > 0) & (tr["s"] > 0)].sort_values("t", kind="stable")
    qt = pd.DataFrame({
        "t": _stamps(quotes),
        "bp": [float(x.get("bp") or 0.0) for x in quotes],
        "ap": [float(x.get("ap") or 0.0) for x in quotes],
    }).sort_values("t", kind="stable")
    qt["quote_t"] = qt["t"]
    if tr.empty:
        return empty
    # The last quote stamped before each trade -- not at the same instant,
    # which is the quote the trade itself made.
    m = pd.merge_asof(tr, qt, on="t", direction="backward", allow_exact_matches=False)
    usable = (
        m["quote_t"].notna()
        & (m["t"] - m["quote_t"] <= pd.Timedelta(seconds=QUOTE_LEAD_SEC))
        & (m["bp"] > 0) & (m["ap"] > 0) & (m["ap"] >= m["bp"])
    )
    m = m[usable].reset_index(drop=True)
    mid = (m["bp"] + m["ap"]) / 2.0
    at_mid = np.isclose(m["p"], mid, rtol=0.0, atol=1e-6)
    side = np.where(at_mid, 0, np.where(m["p"] > mid, 1, -1))
    return pd.DataFrame({"t": m["t"], "value": m["p"] * m["s"], "side": side})


def minute_sums(trades: "list[dict]", quotes: "list[dict]") -> dict[str, list[float]]:
    """{minute start, UTC ISO: [ask $, bid $, total $, trades]} -- the sums
    `classify`'s trades add up to in each minute they printed in."""
    sides = classify(trades, quotes)
    out: dict[str, list[float]] = {}
    if sides.empty:
        return out
    minute = sides["t"].dt.floor("min")
    for stamp, group in sides.groupby(minute):
        ask = float(group.loc[group["side"] > 0, "value"].sum())
        bid = float(group.loc[group["side"] < 0, "value"].sum())
        out[_utc(stamp).isoformat()] = [ask, bid, float(group["value"].sum()), int(len(group))]
    return out


def bar_value(sums: "list[list[float]]") -> "float | None":
    """A bar's aggressiveness from its minutes' [ask, bid, total, ...] sums,
    None with nothing traded."""
    ask = sum(s[0] for s in sums)
    bid = sum(s[1] for s in sums)
    total = sum(s[2] for s in sums)
    return (ask - bid) / total if total > 0 else None


# --- the store -------------------------------------------------------------

def _path(symbol: str, day: str) -> Path:
    return CACHE_DIR / f"{symbol}_{day}.json"


def _load(symbol: str, day: str) -> dict[str, list]:
    path = _path(symbol, day)
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return {str(t): [float(a), float(b), float(v), int(n), "sip"] for t, (a, b, v, n) in raw.items()}
    except (OSError, ValueError, TypeError) as exc:
        logger.warning("Kept aggressiveness %s not read: %s", path, exc)
        return {}


def _save(symbol: str, day: str, kept: dict[str, list]) -> None:
    settled = {t: s[:4] for t, s in kept.items() if s[4] == "sip"}
    path = _path(symbol, day)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(settled), encoding="utf-8")
        os.replace(tmp, path)
    except OSError as exc:  # the panel still draws from memory
        logger.warning("Kept aggressiveness %s not written: %s", path, exc)


def _store(symbol: str, day: str) -> dict[str, list]:
    """The day's kept minutes, loaded from disk on first use. Caller holds
    `_lock`."""
    key = (symbol, day)
    kept = _kept.get(key)
    if kept is None:
        # A new day: the symbol's earlier days are not drawn any more.
        for old in [k for k in _kept if k[0] == symbol]:
            del _kept[old]
        kept = _kept[key] = _load(symbol, day)
    return kept


# --- what to fetch ---------------------------------------------------------

def _wall_now() -> datetime:
    """The wall clock, also under a replay: SIP's delay runs in real time."""
    return datetime.now(timezone.utc)


def _sip_from(real_now: datetime) -> pd.Timestamp:
    """Minutes ending at or before this are on the delayed SIP tape."""
    return _utc(real_now) - pd.Timedelta(minutes=SIP_DELAY_MIN)


def _plan(
    minutes: "list[pd.Timestamp]", kept: dict[str, list], now: datetime, real_now: datetime,
) -> "tuple[list[pd.Timestamp], list[pd.Timestamp]]":
    """(minutes due from IEX, minutes due from SIP, newest first) of the
    chart's `minutes` that have started by `now`.

    SIP for a minute that is over by `now` and on the delayed tape by
    `real_now`, once; IEX for the rest, once over and until then on every
    refresh. The two clocks differ under a replay, whose forming minute is
    long on SIP but not over yet."""
    sip_from = _sip_from(real_now)
    now = _utc(now)
    one = pd.Timedelta(minutes=1)
    settle = pd.Timedelta(seconds=SETTLE_SEC)
    iex, sip = [], []
    for m in minutes:
        if m >= now:
            continue
        entry = kept.get(m.isoformat())
        if m + one + settle <= now and m + one <= sip_from:
            if entry is None or entry[4] != "sip":
                sip.append(m)
        elif entry is None or entry[4] == "iex":
            iex.append(m)
    return iex, sorted(sip, reverse=True)


def _chunks(minutes: "list[pd.Timestamp]") -> "list[list[pd.Timestamp]]":
    """Newest-first minutes as runs of consecutive minutes, at most CHUNK_MIN
    each, newest run first."""
    out: list[list[pd.Timestamp]] = []
    for m in minutes:
        run = out[-1] if out else None
        if run and len(run) < CHUNK_MIN and run[-1] - m == pd.Timedelta(minutes=1):
            run.append(m)
        else:
            out.append([m])
    return out


def _fetch(symbol: str, minutes: "list[pd.Timestamp]", feed: str, key: str, secret: str,
           now: datetime) -> dict[str, list[float]]:
    """The minute sums over `minutes` (consecutive or not) from one feed, every
    fetched minute present -- a minute with no usable trade as zeros."""
    start = min(minutes)
    end = min(max(minutes) + pd.Timedelta(minutes=1), _utc(now))
    trades = fetch_ticks("trades", symbol, start.to_pydatetime(), end.to_pydatetime(),
                         key, secret, feed)
    time.sleep(REQUEST_GAP_SEC)
    quotes = fetch_ticks("quotes", symbol,
                         (start - pd.Timedelta(seconds=QUOTE_LEAD_SEC)).to_pydatetime(),
                         end.to_pydatetime(), key, secret, feed)
    sums = minute_sums(trades, quotes)
    return {m.isoformat(): sums.get(m.isoformat(), [0.0, 0.0, 0.0, 0]) for m in minutes}


def _work(symbol: str, day: str, key: str, secret: str) -> None:
    """Fetch what the chart is missing until nothing is due: the IEX minutes
    whenever REFRESH_SEC has passed, the SIP minutes a chunk at a time between
    them. Re-reads the chart's minutes after every fetch, so minutes that
    appear meanwhile are picked up."""
    try:
        while True:
            now = clock.now()
            real_now = _wall_now()
            with _lock:
                kept = _kept.get((symbol, day))
                if kept is None:
                    return  # the chart has moved on to another day
                minutes = list(_wanted.get((symbol, day), []))
                iex, sip = _plan(minutes, kept, now, real_now)
                iex_due = bool(iex) and time.monotonic() - _iex_at.get(symbol, -1e9) >= REFRESH_SEC
            if iex_due:
                sums = _fetch(symbol, iex, "iex", key, secret, now)
                done = _utc(now) - pd.Timedelta(minutes=1, seconds=SETTLE_SEC)
                with _lock:
                    for t, s in sums.items():
                        # A minute fetched before it was over is fetched again.
                        kept[t] = [*s, "iex-final" if _utc(t) <= done else "iex"]
                    _iex_at[symbol] = time.monotonic()
                    _errors.pop(symbol, None)
            elif sip:
                sums = _fetch(symbol, _chunks(sip)[0], "sip", key, secret, now)
                with _lock:
                    for t, s in sums.items():
                        kept[t] = [*s, "sip"]
                    _errors.pop(symbol, None)
                    snapshot = dict(kept)
                _save(symbol, day, snapshot)
            else:
                return
            time.sleep(REQUEST_GAP_SEC)
    except Exception as exc:
        logger.warning("Aggressiveness for %s not fetched: %s", symbol, exc)
        with _lock:
            _errors[symbol] = str(exc)
            _retry_at[symbol] = time.monotonic() + RETRY_SEC
    finally:
        with _lock:
            _running.discard(symbol)


# --- the panel --------------------------------------------------------------

def _rth_minutes(stamps: "list[pd.Timestamp]", period: pd.Timedelta) -> "list[pd.Timestamp]":
    """The regular-session minutes the bars at `stamps` cover."""
    out = []
    for stamp in stamps:
        for i in range(int(period / pd.Timedelta(minutes=1))):
            m = stamp + pd.Timedelta(minutes=i)
            if _is_rth(m):
                out.append(m)
    return out


def series(
    symbol: str,
    timeframe: str,
    bars: "list[dict]",
    key: str,
    secret: str,
    now: "datetime | None" = None,
) -> dict:
    """The aggressiveness panel for `bars`: ``{"t", "value", "settled",
    "traded", "note"}`` -- per regular-session bar of the latest bar's ET day
    with anything traded, its aggressiveness, whether every minute of it is
    on the settled SIP tape, and the dollar value it is measured over.

    Starts the background fetch of the minutes not kept yet; `note` says why
    the panel is empty while nothing is."""
    if not bars:
        return {"t": [], "value": [], "settled": [], "traded": [], "note": "No bars yet"}
    now = now or clock.now()
    period = pd.Timedelta(minutes=TF_MINUTES.get(timeframe, 1))
    stamps = [_utc(b["t"]) for b in bars]
    day = stamps[-1].tz_convert(MARKET_TZ).date().isoformat()
    todays = sorted({s for s in stamps if s.tz_convert(MARKET_TZ).date().isoformat() == day})
    minutes = _rth_minutes(todays, period)

    with _lock:
        kept = _store(symbol, day)
        _wanted[(symbol, day)] = minutes
        iex, sip = _plan(minutes, kept, now, _wall_now())
        due = bool(sip) or (
            bool(iex) and time.monotonic() - _iex_at.get(symbol, -1e9) >= REFRESH_SEC
        )
        start = (
            due and bool(key) and symbol not in _running
            and time.monotonic() >= _retry_at.get(symbol, 0.0)
        )
        if start:
            _running.add(symbol)
        rows = []
        for stamp in todays:
            covered = [m.isoformat() for m in _rth_minutes([stamp], period) if m < _utc(now)]
            sums = [kept[m] for m in covered if m in kept]
            value = bar_value(sums)
            if value is None:
                continue
            # Every minute of it in, from the consolidated tape: it won't change.
            settled = len(sums) == len(covered) and all(s[4] == "sip" for s in sums)
            rows.append((stamp.isoformat(), value, settled, sum(s[2] for s in sums)))
        error = _errors.get(symbol)
        fetching = symbol in _running

    if start:
        threading.Thread(
            target=clock.inherit(_work), args=(symbol, day, key, secret),
            name=f"aggressiveness-{symbol}", daemon=True,
        ).start()

    if rows:
        note = ""
    elif not minutes:
        note = "Aggressiveness starts at the 09:30 open (regular session only)"
    elif not key:
        note = "Aggressiveness needs an Alpaca key for its trades and quotes"
    elif error:
        note = f"Trades and quotes not fetched ({error}) — retrying in a minute"
    elif fetching or start:
        note = "Fetching trades and quotes…"
    else:
        note = "No trades with a quote yet"
    return {
        "t": [r[0] for r in rows],
        "value": [r[1] for r in rows],
        "settled": [r[2] for r in rows],
        "traded": [r[3] for r in rows],
        "note": note,
    }
