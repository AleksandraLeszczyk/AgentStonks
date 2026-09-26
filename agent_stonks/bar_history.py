"""Where the app's historical minute bars come from -- and why not IEX.

Four separate paths pour REST bars into the same `SymbolState.bars` buffer that
the live socket is filling: the initial history load, the timeframe reload, the
periodic hole-repairing backfill, and the stream-down fallback poll. They have
to agree on a source, because the buffer is read as one series -- a volume
profile, an rvol_pace, a VWAP, a model's volume feature all sum across it
without knowing which bar came from where.

That makes the feed choice a data-quality decision rather than a plumbing
detail, and on volume the feeds are not close. Measured on AAPL, the regular
session of 2026-09-11, 390 aligned one-minute bars:

    IEX        1,560,682     3.75% of consolidated
    Alpaca SIP 41,595,644    the consolidated tape
    yfinance   42,223,381    101.5% of SIP, per-minute correlation 0.964

IEX is one venue's slice of the tape -- under 4% of it -- so a buffer that mixes
IEX history with a consolidated live stream has a ~26x volume step at the seam.
Every volume-derived read in the app crosses that seam: relative volume compares
today's running total against a daily average, the volume profile bins the whole
session, and the momentum models take volume features over a lookback window.

So the default here is "auto", which means: **the consolidated tape if this key
can have it, and never IEX while a consolidated alternative exists.**

    sip          real-time consolidated, needs a paid Alpaca plan
    sip_delayed  the same consolidated tape on a free/basic plan, which refuses
                 only the trailing 15 minutes -- fine for a backfill, whose job
                 is repairing holes the live stream already flew past
    yfinance     consolidated too (within 1.5% of SIP), free, but delayed ~15
                 minutes and capped at about 7 days of one-minute history
    iex          last resort -- correct prices, unusable volume

The Finnhub live source streams the consolidated tape, so "auto" resolving to
sip or yfinance is what makes the backfilled half of the buffer comparable with
the streamed half. It matters on the Alpaca IEX stream too, just in the other
direction: there the *history* becomes the odd one out instead.
"""
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from . import clock
from .config import (
    DEFAULT_HISTORY_FEED,
    HISTORY_FEEDS,
    MAX_BARS,
    SETTLED_BAR_AGE_MIN,
    SIP_DELAY_MIN,
)
from .datalog import log_fetch, log_fetch_failure
from .historical import fetch_intraday_bars, fetch_intraday_bars_window
from .market_hours import MARKET_CLOSE, MARKET_OPEN, MARKET_TZ
from .rest import KEEP_NEWEST, fetch_bars, fetch_bars_window, fetch_daily_bars

logger = logging.getLogger(__name__)

# yfinance equivalents of Alpaca timeframes. A timeframe absent here (1Day) has
# no yfinance intraday equivalent, so yfinance cannot stand in for it.
YF_INTERVALS: dict[str, str] = {
    "1Min": "1m", "5Min": "5m", "15Min": "15m", "30Min": "30m", "1Hour": "60m",
}

SOURCE_LABELS: dict[str, str] = {
    "sip": "Alpaca REST (SIP, consolidated)",
    "sip_delayed": f"Alpaca REST (SIP, consolidated, {SIP_DELAY_MIN}min delayed)",
    "iex": "Alpaca REST (IEX, single venue)",
    "yfinance": "yfinance (delayed, consolidated)",
}

# Concrete sources `fetch_history_bars` knows how to read, best first. "auto" is
# a *choice* and never appears here -- resolve_history_feed turns it into one of
# these. sip_delayed outranks yfinance because it is the same consolidated tape
# Finnhub streams rather than a third party's rendering of it, it carries the
# auction prints as their own bars, and it is not capped at ~7 days of history.
CONCRETE_FEEDS: tuple[str, ...] = ("sip", "sip_delayed", "yfinance", "iex")


def is_consolidated(feed: str) -> bool:
    """Whether `feed` names a tape that carries the whole market's volume.

    Everything here but IEX does: SIP (both tiers) is the consolidated tape,
    yfinance is a rendering of it within 1.5%, and the Finnhub live source
    streams it too -- which is why "finnhub" answers True despite never being a
    REST feed. IEX is one venue at under 4%, so a number fitted on consolidated
    volume is off its scale there.
    """
    return str(feed or "").lower() not in ("iex", "")


def feed_order(feed: str) -> "list[str]":
    """`feed` first, then every other concrete source, consolidated before IEX.

    The one ranking of tapes in the app: a consolidated source is replaced by
    another consolidated source wherever possible and IEX is reached only when
    nothing else answers. `feed` is filtered against the concrete set because
    callers may still be holding the unresolved "auto" -- a choice, not a feed,
    which would reach Alpaca as a literal `feed=auto`.
    """
    return ([feed] if feed in CONCRETE_FEEDS else []) + [
        f for f in CONCRETE_FEEDS if f != feed
    ]


def _sip_window(lookback_hours: int) -> "tuple[datetime, datetime]":
    """The [start, end) a delayed-SIP key is allowed to ask for, ending
    SIP_DELAY_MIN behind now."""
    end = datetime.now(timezone.utc) - timedelta(minutes=SIP_DELAY_MIN)
    return end - timedelta(hours=lookback_hours), end


def _probe_sip(symbol: str, key: str, secret: str) -> "tuple[str, object | None]":
    """What kind of SIP access this key has: "sip", "sip_delayed", or "".

    There are three tiers, and they are not distinguishable without asking.
    Alpaca's paid plans serve SIP up to the current second; the free and basic
    plans serve the same tape but refuse any window reaching into the trailing
    15 minutes, with 403 "subscription does not permit querying recent SIP
    data"; an account with no SIP at all is refused outright.

    So the probe asks twice: once ending now, and, if that is refused for any
    reason, once ending SIP_DELAY_MIN back. It does not try to tell the two
    refusals apart from the exception -- `raise_for_status` puts only the status
    line in the message, not Alpaca's explanatory body -- and it does not need
    to: the second request answers the question directly. A key with no SIP at
    all fails both and the caller moves to the next consolidated source.
    """
    try:
        fetch_bars(symbol, "1Min", 1, key, secret, "sip", lookback_hours=48)
        return "sip", None
    except Exception as exc:
        realtime_error = exc
    start, end = _sip_window(48)
    try:
        fetch_bars_window(symbol, "1Min", start, end, key, secret, "sip", limit=1)
    except Exception:
        # Both refused: report the real-time error, which is the informative one
        # (a plain 403 with no SIP entitlement at all).
        return "", realtime_error
    return "sip_delayed", None


def resolve_history_feed(
    choice: str, symbol: str, key: str, secret: str, timeframe: str = "1Min"
) -> str:
    """Turn the configured choice into the concrete feed this session will use.

    Called once per session so every REST bar path uses the same source -- the
    point of the exercise is a buffer that does not change volume units halfway
    along, and re-resolving per call could produce exactly that.

    "auto" prefers SIP -- real-time if the plan allows it, otherwise the same
    tape held back by SIP_DELAY_MIN, which is still the right source for a
    backfill that repairs holes. It falls back to yfinance where the timeframe
    has an intraday equivalent, and only then to IEX.

    An explicit "sip" is resolved the same way, so a delayed-SIP key asking for
    SIP gets SIP rather than a 403 on every fetch. An explicit yfinance for a
    timeframe it cannot serve (1Day) falls back rather than failing every fetch.
    """
    if choice not in HISTORY_FEEDS:
        choice = DEFAULT_HISTORY_FEED
    if choice == "yfinance" and timeframe not in YF_INTERVALS:
        logger.info(
            "History feed: yfinance has no %s equivalent, using Alpaca SIP/IEX instead", timeframe
        )
        choice = "auto"
    if choice not in ("auto", "sip"):
        return choice

    tier, exc = _probe_sip(symbol, key, secret)
    if tier == "sip":
        logger.info("History feed: SIP (consolidated tape, real-time)")
        return "sip"
    if tier == "sip_delayed":
        logger.info(
            "History feed: SIP delayed %dmin (consolidated tape; this plan refuses the "
            "trailing 15min, which the live stream covers anyway)", SIP_DELAY_MIN
        )
        return "sip_delayed"
    if choice == "sip":
        logger.warning("History feed: SIP unavailable (%s), falling back", exc)
    if timeframe in YF_INTERVALS:
        logger.info(
            "History feed: yfinance (consolidated, ~15min delayed) — SIP unavailable (%s)", exc
        )
        return "yfinance"
    logger.warning(
        "History feed: IEX — SIP unavailable (%s) and yfinance has no %s equivalent. "
        "IEX carries under 4%% of consolidated volume, so volume-derived reads "
        "(relative volume, volume profile, model volume features) will understate "
        "badly against live streamed bars.",
        exc,
        timeframe,
    )
    return "iex"


def fetch_history_bars(
    symbol: str,
    timeframe: str,
    key: str,
    secret: str,
    feed: str,
    limit: int = MAX_BARS,
    lookback_hours: int = 16,
    what: str = "bars",
) -> "tuple[list[dict], str, list[tuple[str, object]]]":
    """Historical bars for one symbol from the resolved feed, with fallbacks.

    Returns (bars, source_label, failures) where `failures` lists the
    (source, error) pairs tried before the one that worked, for the caller's
    log line. Raises only when every source failed.

    The fallback order preserves the module's premise: a consolidated source is
    replaced by another consolidated source where possible, and IEX is reached
    only when nothing else answers.

    `limit` is a cap on how many bars come back, and `lookback_hours` of minute
    bars is routinely more than that -- 16 hours spans a premarket, a session
    and the previous afternoon, which is over 600 one-minute bars on a liquid
    symbol by the close against a 420-bar cap. What survives the cap is
    therefore a real decision and it is made the same way on every Alpaca path:
    the **newest** bars in the window. A page starting at the window's beginning
    would stop hours short of now, which is where the holes a backfill exists to
    repair actually are.
    """
    failures: list[tuple[str, object]] = []
    yf_interval = YF_INTERVALS.get(timeframe)

    for candidate in feed_order(feed):
        label = SOURCE_LABELS.get(candidate, candidate)
        try:
            if candidate == "yfinance":
                if yf_interval is None:
                    continue
                bars = fetch_intraday_bars(symbol, interval=yf_interval)
            elif candidate == "sip_delayed":
                start, end = _sip_window(lookback_hours)
                # The newest end of the window, not the oldest: `lookback_hours`
                # of minute bars is more than `limit` of them for most of a
                # session, and the half worth having is the recent half.
                bars = fetch_bars_window(
                    symbol, timeframe, start, end, key, secret, "sip",
                    limit=limit, keep=KEEP_NEWEST,
                )
            else:
                bars = fetch_bars(
                    symbol, timeframe, limit, key, secret, candidate,
                    lookback_hours=lookback_hours,
                )
        except Exception as exc:
            failures.append((label, exc))
            continue
        return bars, label, failures

    log_fetch_failure(
        what, failures, symbol=symbol, consequence="no bar source available this cycle"
    )
    raise RuntimeError(f"every bar source failed for {symbol}: {failures}")


def fetch_daily(
    symbol: str,
    key: str,
    secret: str,
    feed: str,
    lookback_days: int = 365,
) -> "tuple[list[dict], str]":
    """Daily bars from the session's resolved feed, with the same fallbacks.

    The daily series is the *baseline* every volume comparison divides by -- the
    high-volume alert, the `volume_ratio` and `rvol_pace` alert fields, and the
    briefing's relative-volume pace all measure today's running total against
    it. So it has to come from the same tape as the intraday bars: pairing a
    consolidated intraday series with an IEX daily average divides ~41M shares
    by a ~1.3M baseline and reports a perfectly ordinary session as 30-40x
    normal participation.

    Returns (bars, source_label). Falls back to IEX rather than failing, since
    a wrong baseline is still better than no daily history at all -- but the
    caller logs which feed answered so the mismatch is visible.
    """
    failures: list[tuple[str, object]] = []
    for candidate in feed_order(feed):
        # yfinance's intraday helper has no daily equivalent wired up here, and
        # Alpaca serves the whole daily history under one call either way.
        if candidate == "yfinance":
            continue
        try:
            if candidate == "sip_delayed":
                start, end = _sip_window(lookback_days * 24)
                bars = fetch_bars_window(
                    symbol, "1Day", start, end, key, secret, "sip", limit=lookback_days + 10
                )
            else:
                bars = fetch_daily_bars(
                    symbol, key, secret, candidate, lookback_days=lookback_days
                )
        except Exception as exc:
            failures.append((SOURCE_LABELS.get(candidate, candidate), exc))
            continue
        label = SOURCE_LABELS.get(candidate, candidate)
        log_fetch(
            "daily bars (initial load)", label, symbol=symbol,
            detail=f"{len(bars)} daily bars", failures=failures,
        )
        return bars, label
    log_fetch_failure(
        "daily bars (initial load)", failures, symbol=symbol,
        consequence="no volume baseline; relative-volume reads and the high-volume alert are off",
    )
    raise RuntimeError(f"every daily bar source failed for {symbol}: {failures}")


def fetch_and_log(
    symbol: str,
    timeframe: str,
    key: str,
    secret: str,
    feed: str,
    limit: int = MAX_BARS,
    lookback_hours: int = 16,
    what: str = "bars",
    detail_suffix: str = "",
) -> "tuple[list[dict], str]":
    """`fetch_history_bars` plus the standard success log line."""
    bars, source, failures = fetch_history_bars(
        symbol, timeframe, key, secret, feed,
        limit=limit, lookback_hours=lookback_hours, what=what,
    )
    log_fetch(
        what,
        source,
        symbol=symbol,
        detail=f"{len(bars)} {timeframe} bars{detail_suffix}",
        failures=failures,
    )
    return bars, source


# -- the live buffer's tiered fetch --------------------------------------------
#
# The buffer the live socket fills gets its REST bars by age, because no single
# source is right for all of them (measured on AAPL, 2026-09-25):
#
#   older than SETTLED_BAR_AGE_MIN, regular session   yfinance
#       within 0.5% of SIP minute by minute, and the source asked for.
#   older than SETTLED_BAR_AGE_MIN, pre/post-market   the resolved feed
#       yfinance reports these minutes with volume 0 (and 3,565 shares for the
#       16:00 ET minute that carries SIP's 5.1M-share closing auction), so it
#       cannot stand in for them. Delayed SIP can.
#   younger, up to the minute in progress             Alpaca IEX, provisional
#       nothing consolidated is free that recently, and IEX is ~3-4% of the
#       volume -- so these bars are flagged and replaced once they settle.
#   the minute in progress                            never fetched
#       the live socket owns it. A REST snapshot of a half-finished minute
#       would sit in the buffer and make the socket refuse that minute's trades.

# yfinance downloads are ~1 s each and the stream-down poll asks every 15 s, so
# a window is reused for this long. Keyed by (symbol, interval).
_YF_WINDOW_TTL_SEC = 30.0
_yf_window_cache: "dict[tuple[str, str], tuple[float, datetime, list[dict]]]" = {}


@dataclass
class LiveBars:
    """What `fetch_live_bars` found: the bars, which of them are provisional
    (by `bar_key`), a label naming every source that contributed, and the
    (source, error) pairs that failed along the way."""

    bars: list[dict]
    provisional: "set[str]" = field(default_factory=set)
    source: str = ""
    failures: "list[tuple[str, object]]" = field(default_factory=list)


def bar_key(ts: object) -> str:
    """One spelling per bar timestamp ('Z' and '+00:00' agree). The same key
    `stream_common.bar_ts_key` produces, repeated here to keep this module free
    of the tick path's imports."""
    return clock.parse_iso_strict(ts).isoformat()


def _in_regular_session(ts: object) -> bool:
    et = clock.parse_iso_strict(ts).astimezone(MARKET_TZ)
    return et.weekday() < 5 and MARKET_OPEN <= et.time() < MARKET_CLOSE


def _bucket_start(now: datetime, timeframe: str) -> datetime:
    """Start of the `timeframe` bucket holding `now` -- the bar in progress."""
    minutes = {"5Min": 5, "15Min": 15, "30Min": 30, "1Hour": 60}.get(timeframe, 1)
    total = now.hour * 60 + now.minute
    floored = (total // minutes) * minutes
    return now.replace(hour=floored // 60, minute=floored % 60, second=0, microsecond=0)


def _yfinance_window(symbol: str, interval: str, start: datetime, end: datetime) -> "list[dict]":
    """yfinance bars in [start, end), through a short cache."""
    cache_key = (symbol, interval)
    hit = _yf_window_cache.get(cache_key)
    if hit is not None and time.monotonic() - hit[0] < _YF_WINDOW_TTL_SEC and hit[1] <= start:
        bars = hit[2]
    else:
        bars = fetch_intraday_bars_window(symbol, start, end, interval=interval)
        _yf_window_cache[cache_key] = (time.monotonic(), start, bars)
    return [b for b in bars if start <= clock.parse_iso_strict(b["t"]) < end]


def _fetch_recent_bars(
    symbol: str, timeframe: str, start: datetime, end: datetime,
    key: str, secret: str, feed: str,
) -> "list[dict]":
    """The young tier: Alpaca bars in [start, end) off `feed`."""
    return fetch_bars_window(
        symbol, timeframe, start, end, key, secret, feed, limit=200, keep=KEEP_NEWEST,
    )


def fetch_live_bars(
    symbol: str,
    timeframe: str,
    key: str,
    secret: str,
    feed: str,
    limit: int = MAX_BARS,
    lookback_hours: int = 16,
    what: str = "bars",
    now: "datetime | None" = None,
) -> LiveBars:
    """REST bars for the live buffer, each from the source its age calls for.

    `feed` is the session's resolved history feed; it serves the settled bars
    yfinance cannot (pre/post-market, the 1Day timeframe, a yfinance outage).
    Bars that came off IEX are listed in `provisional` so the buffer knows to
    replace them. A real-time SIP key serves the young tier too, and then
    nothing is provisional.

    Raises only when no tier returned anything and something failed.
    """
    now = now or datetime.now(timezone.utc)
    settled_cut = now - timedelta(minutes=SETTLED_BAR_AGE_MIN)
    current = _bucket_start(now, timeframe)
    failures: list[tuple[str, object]] = []
    by_key: dict[str, dict] = {}
    provisional: set[str] = set()
    sources: list[str] = []

    # Settled, from the resolved feed. Anything it returns from the young window
    # is dropped: that window has its own tier.
    try:
        base, base_label, base_failures = fetch_history_bars(
            symbol, timeframe, key, secret, feed,
            limit=limit, lookback_hours=lookback_hours, what=what,
        )
    except Exception:
        base, base_label, base_failures = [], "", []
    failures.extend(base_failures)
    base_is_iex = base_label == SOURCE_LABELS["iex"]
    kept = 0
    for bar in base:
        if "t" in bar and clock.parse_iso_strict(bar["t"]) < settled_cut:
            by_key[bar_key(bar["t"])] = bar
            kept += 1
            if base_is_iex:
                provisional.add(bar_key(bar["t"]))
    if kept:
        sources.append(base_label)

    # Settled regular-session minutes from yfinance, over whatever the resolved
    # feed said for them.
    yf_interval = YF_INTERVALS.get(timeframe)
    if yf_interval is not None and base_label != SOURCE_LABELS["yfinance"]:
        start = now - timedelta(hours=lookback_hours)
        try:
            yf_bars = _yfinance_window(symbol, yf_interval, start, settled_cut)
        except Exception as exc:
            failures.append((SOURCE_LABELS["yfinance"], exc))
            yf_bars = []
        replaced = 0
        for bar in yf_bars:
            if "t" in bar and _in_regular_session(bar["t"]):
                k = bar_key(bar["t"])
                by_key[k] = bar
                provisional.discard(k)
                replaced += 1
        if replaced:
            sources.append(f"{SOURCE_LABELS['yfinance']} regular session")

    # The young tier, up to (not including) the bar in progress.
    recent_feed = "sip" if feed == "sip" else "iex"
    recent_start = min(settled_cut, now - timedelta(minutes=SIP_DELAY_MIN))
    if current > recent_start:
        try:
            recent = _fetch_recent_bars(
                symbol, timeframe, recent_start, current, key, secret, recent_feed
            )
        except Exception as exc:
            failures.append((SOURCE_LABELS[recent_feed], exc))
            recent = []
        added = 0
        for bar in recent:
            if "t" not in bar:
                continue
            k = bar_key(bar["t"])
            if k in by_key:
                continue
            by_key[k] = bar
            added += 1
            if recent_feed == "iex":
                provisional.add(k)
        if added:
            sources.append(
                f"{SOURCE_LABELS[recent_feed]} last {SETTLED_BAR_AGE_MIN}min"
                + (", provisional" if recent_feed == "iex" else "")
            )

    if not by_key and failures:
        log_fetch_failure(
            what, failures, symbol=symbol, consequence="no bar source available this cycle"
        )
        raise RuntimeError(f"every bar source failed for {symbol}: {failures}")
    bars = [by_key[k] for k in sorted(by_key)][-limit:]
    present = {bar_key(b["t"]) for b in bars}
    return LiveBars(
        bars=bars,
        provisional=provisional & present,
        source=" + ".join(sources) or "no bars",
        failures=failures,
    )
