"""Price levels an agent can hang a plan on.

The opening range, prior-session structure, swing highs and lows, the volume
profile and the floor pivots. All of them return the same shape of answer: a
set of named prices, each with how far spot sits from it.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from .. import clock, market_hours

_ET = market_hours.MARKET_TZ

from ._shared import _news_datetimes, _news_near

from .indicators import atr


# How late (after 09:30 ET) the earliest retained session bar may be before the
# buffer is judged not to reach back to the open. Generous enough for a thin
# symbol whose first prints trickle in, tight enough that a stream started
# mid-morning can never pass off its start time as the opening bell.
_OPENING_RANGE_COVERAGE_GRACE_MIN = 3

def compute_opening_range(
    bars: list[dict],
    minutes: int = 15,
    now: "datetime | None" = None,
    assume_coverage: bool = False,
) -> dict:
    """Measure today's opening range from bar timestamps: the high/low printed
    in the 09:30 ET to 09:30+`minutes` window of the latest bar's session.

    Unlike a first-N-bars slice, this refuses to fabricate a range when the bar
    history does not actually reach back to the opening bell (stream started
    mid-session, or the buffer has evicted the morning) -- it returns a dict
    with only a `note` explaining why. `assume_coverage` skips that check for
    callers that fetched the 09:30 window explicitly (the REST fallback).
    Returns {date, minutes, high, low, bar_count, avg_volume, complete} on
    success; `complete` is False while the window is still forming.
    """
    if not bars:
        return {"note": "no intraday bars available yet"}

    stamped = [(b, dt) for b in bars if (dt := clock.bar_dt(b)) is not None]
    if not stamped:
        return {"note": "intraday bars carry no usable timestamps -- cannot anchor the 09:30 ET opening range"}

    session_date = stamped[-1][1].astimezone(_ET).date()
    open_et = datetime.combine(session_date, market_hours.MARKET_OPEN, tzinfo=_ET)
    window_end = open_et + timedelta(minutes=max(1, minutes))

    session = [(b, dt) for b, dt in stamped if dt.astimezone(_ET).date() == session_date and dt >= open_et]
    if not session:
        return {"note": "no bars from today's regular session yet -- the opening range has not formed"}

    if not assume_coverage:
        earliest = session[0][1]
        if earliest > open_et + timedelta(minutes=_OPENING_RANGE_COVERAGE_GRACE_MIN):
            return {
                "note": (
                    "opening range unavailable: bar history only reaches back to "
                    f"{earliest.astimezone(_ET).strftime('%H:%M')} ET, not today's 09:30 ET open "
                    "-- refusing to fabricate an opening range from a mid-session window"
                )
            }

    opening = [(b, dt) for b, dt in session if dt < window_end]
    if not opening:
        return {"note": "no bars printed inside the opening window yet"}

    now = now or clock.now()
    complete = max(now, session[-1][1]) >= window_end
    volumes = [float(b.get("v") or 0.0) for b, _ in opening]
    return {
        "date": session_date.isoformat(),
        "minutes": max(1, minutes),
        "high": round(max(float(b["h"]) for b, _ in opening), 4),
        "low": round(min(float(b["l"]) for b, _ in opening), 4),
        "bar_count": len(opening),
        "avg_volume": (sum(volumes) / len(volumes)) if volumes else 0.0,
        "complete": complete,
    }

def analyze_opening_range(
    bars: list[dict],
    minutes: int = 15,
    opening_range: "dict | None" = None,
    now: "datetime | None" = None,
) -> dict:
    """Opening Range Breakout (ORB) read: the high/low set by the 09:30 ET +
    `minutes` window of today's session, and whether price has since broken out.

    The range itself comes from `opening_range` when given (a cached/pre-fetched
    result of :func:`compute_opening_range`, so the read survives buffer
    eviction and mid-session starts); otherwise it is measured from `bars`.
    When the range cannot be established honestly, the result carries only a
    `note` -- never a fabricated range.
    """
    rng = opening_range if opening_range is not None else compute_opening_range(bars, minutes, now=now)
    if "high" not in rng:
        return rng

    stamped = [(b, dt) for b in bars if (dt := clock.bar_dt(b)) is not None]
    session_date = rng.get("date")
    open_et = None
    if session_date:
        open_et = datetime.combine(
            datetime.fromisoformat(session_date).date(), market_hours.MARKET_OPEN, tzinfo=_ET
        )
    window_end = open_et + timedelta(minutes=rng["minutes"]) if open_et else None

    session = [
        (b, dt)
        for b, dt in stamped
        if open_et is not None and dt >= open_et and dt.astimezone(_ET).date().isoformat() == session_date
    ]
    after_window = [(b, dt) for b, dt in session if window_end is not None and dt >= window_end]

    or_high, or_low = float(rng["high"]), float(rng["low"])
    price = float(session[-1][0]["c"]) if session else (float(bars[-1]["c"]) if bars else None)

    if not rng.get("complete", True):
        status = "still forming"
    elif price is None:
        status = "unknown"
    elif price > or_high:
        status = "broken out above"
    elif price < or_low:
        status = "broken out below"
    else:
        status = "inside_range"

    opening_avg_volume = float(rng.get("avg_volume") or 0.0)
    breakout_bars = [b for b, _ in after_window][-3:]
    breakout_avg_volume = (
        sum(float(b.get("v") or 0.0) for b in breakout_bars) / len(breakout_bars)
        if breakout_bars
        else None
    )
    volume_ratio = (
        breakout_avg_volume / opening_avg_volume
        if breakout_avg_volume is not None and opening_avg_volume
        else None
    )

    summary_parts = [
        f"Opening range (09:30 ET + {rng['minutes']} min): {or_low:.2f}-{or_high:.2f}."
        + (f" Price {price:.2f} is {status.replace('_', ' ')}." if price is not None else "")
    ]
    if volume_ratio is not None:
        summary_parts.append(
            f"Recent volume is {volume_ratio:.1f}x the opening-range average"
            f" ({'confirms' if volume_ratio >= 1.5 else 'does not confirm'} a breakout)."
        )

    return {
        "opening_range_minutes": rng["minutes"],
        "opening_range_date": session_date,
        "opening_range_high": or_high,
        "opening_range_low": or_low,
        "current_price": round(price, 4) if price is not None else None,
        "status": status,
        "volume_ratio_vs_opening_range": round(volume_ratio, 2) if volume_ratio is not None else None,
        "summary": " ".join(summary_parts),
    }

def key_levels(
    intraday_bars: list[dict],
    daily_bars: "list[dict] | None" = None,
    spot: "float | None" = None,
    opening_range_minutes: int = 15,
    opening_range: "dict | None" = None,
) -> dict:
    """Session-structure support/resistance map: the concrete price levels an
    intraday trader anchors entries, stops, and targets to.

    Collects the most-watched structural levels -- the prior day's high/low/
    close (from the last completed daily bar), today's premarket high/low,
    the opening-range high/low, and the session high/low so far -- then splits
    them around `spot` into overhead resistance (nearest first) and support
    below (nearest first). The nearest overhead level is the natural first
    target and "room to run" cap for a long entry; the nearest support anchors
    the stop. An empty overhead list means blue-sky territory: price is above
    every tracked level.
    """
    if not intraday_bars and not daily_bars:
        return {"note": "no bar data available yet"}

    # An empty `today` (timestamp-less synthetic bars) matches every bar, so
    # they all count as today's session; with no intraday bars at all, fall
    # back to the calendar date so the prior-day filter still works.
    today = str(intraday_bars[-1].get("t", ""))[:10] if intraday_bars else ""
    if not intraday_bars:
        today = clock.now().strftime("%Y-%m-%d")

    levels: dict[str, float] = {}

    prior = [
        b
        for b in (daily_bars or [])
        if str(b.get("t", ""))[:10] and (not today or str(b.get("t", ""))[:10] < today)
    ]
    if prior:
        prior_day = prior[-1]
        levels["prior_day_high"] = float(prior_day["h"])
        levels["prior_day_low"] = float(prior_day["l"])
        levels["prior_day_close"] = float(prior_day["c"])

    # Split today's intraday bars at the 9:30 ET bell; bars without a parseable
    # timestamp count as regular-session bars (matching _session_bars' fallback).
    premarket: list[dict] = []
    session: list[dict] = []
    for b in intraday_bars:
        ts_raw = str(b.get("t", ""))
        if ts_raw[:10] != today:
            continue
        ts = clock.parse_iso(ts_raw)
        if ts is None:
            session.append(b)
            continue
        et = ts.astimezone(_ET)
        if et.hour * 60 + et.minute < 9 * 60 + 30:
            premarket.append(b)
        else:
            session.append(b)

    if premarket:
        levels["premarket_high"] = max(float(b["h"]) for b in premarket)
        levels["premarket_low"] = min(float(b["l"]) for b in premarket)
    if session:
        levels["session_high"] = max(float(b["h"]) for b in session)
        levels["session_low"] = min(float(b["l"]) for b in session)
    # Opening-range levels come from the timestamp-anchored measurement (or a
    # caller-supplied cached range), never from a first-N-bars slice: when the
    # buffer doesn't reach back to the 09:30 ET open the levels are simply
    # omitted rather than fabricated from a mid-session window.
    rng = (
        opening_range
        if opening_range is not None
        else compute_opening_range(intraday_bars, opening_range_minutes)
    )
    if "high" in rng:
        levels["opening_range_high"] = float(rng["high"])
        levels["opening_range_low"] = float(rng["low"])

    if spot is None:
        if intraday_bars:
            spot = float(intraday_bars[-1]["c"])
        elif "prior_day_close" in levels:
            spot = levels["prior_day_close"]
    if spot is None or not levels:
        return {"note": "not enough data to build key levels"}

    def _entry(name: str) -> dict:
        level = levels[name]
        return {
            "name": name,
            "level": round(level, 4),
            "distance_pct": round((level / spot - 1) * 100, 2),
        }

    resistance = sorted(
        (_entry(name) for name, value in levels.items() if value > spot),
        key=lambda e: e["level"],
    )
    support = sorted(
        (_entry(name) for name, value in levels.items() if value <= spot),
        key=lambda e: -e["level"],
    )
    nearest_resistance = resistance[0] if resistance else None
    nearest_support = support[0] if support else None

    summary_parts = [f"Spot {spot:.2f}."]
    if nearest_resistance is not None:
        summary_parts.append(
            f"Nearest overhead resistance: {nearest_resistance['name']} at "
            f"{nearest_resistance['level']:.2f} ({nearest_resistance['distance_pct']:+.2f}%)"
            + (
                f"; {len(resistance) - 1} more level(s) above."
                if len(resistance) > 1
                else "; nothing tracked above it."
            )
        )
    else:
        summary_parts.append(
            "No overhead level -- price is in blue-sky territory above the session, "
            "premarket, and prior-day highs."
        )
    if nearest_support is not None:
        summary_parts.append(
            f"Nearest support: {nearest_support['name']} at {nearest_support['level']:.2f} "
            f"({nearest_support['distance_pct']:+.2f}%)."
        )
    else:
        summary_parts.append("No structural support below -- price is under every tracked level.")

    return {
        "spot": round(spot, 4),
        "levels": {name: round(value, 4) for name, value in levels.items()},
        "resistance_above": resistance,
        "support_below": support,
        "nearest_resistance": nearest_resistance,
        "nearest_support": nearest_support,
        "summary": " ".join(summary_parts),
    }

def swing_levels(bars: list[dict], swing: int = 3, max_levels: int = 6, spot: "float | None" = None) -> dict:
    """Clustered swing-point (fractal) support/resistance.

    A swing high is a bar whose high is the highest of the `swing` bars on
    each side (swing low mirrored) -- a confirmed local extreme, not a
    still-forming one. Nearby pivots are then clustered within a tolerance of
    0.25 ATR (falling back to 0.1% of price), because a level retested several
    times is far stronger evidence of defended supply/demand than any single
    extreme print. Clusters are ranked by touch count, then recency; each
    carries its mean level, touches, and how many bars ago it was last tested.
    """
    n = len(bars)
    if n < 2 * swing + 1:
        return {"note": "not enough bars to locate swing points", "levels": []}

    h = [float(b["h"]) for b in bars]
    l = [float(b["l"]) for b in bars]
    if spot is None:
        spot = float(bars[-1]["c"])

    pivots: list[tuple[float, int]] = []
    for i in range(swing, n - swing):
        if h[i] == max(h[i - swing : i + swing + 1]):
            pivots.append((h[i], i))
        if l[i] == min(l[i - swing : i + swing + 1]):
            pivots.append((l[i], i))
    if not pivots:
        return {"note": "no confirmed swing points in the window", "levels": []}

    atr_value = atr(bars, period=min(14, n - 1))
    tolerance = 0.25 * atr_value if atr_value else spot * 0.001

    clusters: list[dict] = []
    for price, idx in sorted(pivots):
        if clusters and abs(price - clusters[-1]["_sum"] / clusters[-1]["touches"]) <= tolerance:
            cluster = clusters[-1]
            cluster["_sum"] += price
            cluster["touches"] += 1
            cluster["last_index"] = max(cluster["last_index"], idx)
        else:
            clusters.append({"_sum": price, "touches": 1, "last_index": idx})

    levels = []
    for cluster in clusters:
        level = cluster["_sum"] / cluster["touches"]
        levels.append(
            {
                "level": round(level, 4),
                "touches": cluster["touches"],
                "last_test_bars_ago": n - 1 - cluster["last_index"],
                "type": "resistance" if level > spot else "support",
            }
        )
    levels.sort(key=lambda e: (-e["touches"], e["last_test_bars_ago"]))
    levels = levels[:max_levels]

    resistance = [e for e in levels if e["type"] == "resistance"]
    support = [e for e in levels if e["type"] == "support"]
    nearest_resistance = min(resistance, key=lambda e: e["level"]) if resistance else None
    nearest_support = max(support, key=lambda e: e["level"]) if support else None

    summary_parts = [
        f"{len(levels)} clustered swing level(s) over {n} bars "
        f"(cluster tolerance {tolerance:.4f}); spot {spot:.2f}."
    ]
    if nearest_resistance is not None:
        summary_parts.append(
            f"Nearest swing resistance {nearest_resistance['level']:.2f} "
            f"({nearest_resistance['touches']} touch(es), last {nearest_resistance['last_test_bars_ago']} bars ago)."
        )
    else:
        summary_parts.append("No swing resistance above spot -- price is above every confirmed swing high.")
    if nearest_support is not None:
        summary_parts.append(
            f"Nearest swing support {nearest_support['level']:.2f} "
            f"({nearest_support['touches']} touch(es), last {nearest_support['last_test_bars_ago']} bars ago)."
        )
    else:
        summary_parts.append("No swing support below spot.")

    return {
        "spot": round(spot, 4),
        "cluster_tolerance": round(tolerance, 4),
        "levels": levels,
        "nearest_resistance": nearest_resistance,
        "nearest_support": nearest_support,
        "summary": " ".join(summary_parts),
    }

def volume_profile_levels(bars: list[dict], bins: int = 24, spot: "float | None" = None) -> dict:
    """Volume-by-price profile: POC, value area, and high/low-volume nodes.

    Buckets each bar's volume at its typical price ((H+L+C)/3) across `bins`
    equal price slices of the window's range. The Point of Control (POC) is
    the price with the most transacted volume -- a magnet/defended level; the
    value area is the price band around the POC covering 70% of volume.
    High-volume nodes (HVNs, >=1.5x the mean bin volume) act as
    support/resistance where positions were actually built; low-volume nodes
    (LVNs, <=0.5x) are air pockets price tends to travel through quickly --
    an LVN just above an entry improves the odds of a fast run to the next
    HVN.
    """
    if len(bars) < 10:
        return {"note": "not enough bars for a volume profile"}

    lo = min(float(b["l"]) for b in bars)
    hi = max(float(b["h"]) for b in bars)
    if hi <= lo:
        return {"note": "no price range in the window; cannot build a profile"}
    if spot is None:
        spot = float(bars[-1]["c"])

    width = (hi - lo) / bins
    volume_by_bin = [0.0] * bins
    for b in bars:
        typical = (float(b["h"]) + float(b["l"]) + float(b["c"])) / 3.0
        idx = min(bins - 1, max(0, int((typical - lo) / width)))
        volume_by_bin[idx] += float(b.get("v") or 0.0)
    total_volume = sum(volume_by_bin)
    if total_volume <= 0:
        return {"note": "no traded volume in the window; cannot build a profile"}

    def _bin_center(idx: int) -> float:
        return lo + (idx + 0.5) * width

    poc_idx = max(range(bins), key=lambda i: volume_by_bin[i])
    poc = _bin_center(poc_idx)

    # Value area: expand from the POC toward whichever neighbor bin holds more
    # volume until 70% of the total is covered.
    covered = volume_by_bin[poc_idx]
    low_idx = high_idx = poc_idx
    while covered < 0.70 * total_volume and (low_idx > 0 or high_idx < bins - 1):
        below = volume_by_bin[low_idx - 1] if low_idx > 0 else -1.0
        above = volume_by_bin[high_idx + 1] if high_idx < bins - 1 else -1.0
        if above >= below:
            high_idx += 1
            covered += volume_by_bin[high_idx]
        else:
            low_idx -= 1
            covered += volume_by_bin[low_idx]
    value_area_low = lo + low_idx * width
    value_area_high = lo + (high_idx + 1) * width

    mean_volume = total_volume / bins

    def _nodes(predicate) -> list[dict]:
        """Merge contiguous qualifying bins into volume-weighted nodes."""
        nodes: list[dict] = []
        run: list[int] = []
        for i in range(bins + 1):
            if i < bins and predicate(volume_by_bin[i]):
                run.append(i)
                continue
            if run:
                run_volume = sum(volume_by_bin[j] for j in run)
                center = (
                    sum(_bin_center(j) * volume_by_bin[j] for j in run) / run_volume
                    if run_volume
                    else _bin_center(run[len(run) // 2])
                )
                nodes.append(
                    {
                        "price": round(center, 4),
                        "volume_pct": round(run_volume / total_volume * 100, 1),
                    }
                )
                run = []
        return nodes

    hvns = _nodes(lambda v: v >= 1.5 * mean_volume)
    lvns = _nodes(lambda v: 0 < v <= 0.5 * mean_volume)

    spot_idx = min(bins - 1, max(0, int((spot - lo) / width)))
    spot_in_lvn = 0 < volume_by_bin[spot_idx] <= 0.5 * mean_volume

    hvns_above = [nd for nd in hvns if nd["price"] > spot]
    hvns_below = [nd for nd in hvns if nd["price"] <= spot]
    nearest_hvn_above = min(hvns_above, key=lambda nd: nd["price"]) if hvns_above else None
    nearest_hvn_below = max(hvns_below, key=lambda nd: nd["price"]) if hvns_below else None

    summary_parts = [
        f"Volume profile over {len(bars)} bars ({lo:.2f}-{hi:.2f}, {bins} bins): "
        f"POC {poc:.2f}, value area {value_area_low:.2f}-{value_area_high:.2f}; spot {spot:.2f}."
    ]
    if nearest_hvn_above is not None:
        summary_parts.append(
            f"Nearest high-volume node above: {nearest_hvn_above['price']:.2f} "
            f"({nearest_hvn_above['volume_pct']:.0f}% of volume) -- likely resistance/magnet."
        )
    else:
        summary_parts.append("No high-volume node above spot.")
    if nearest_hvn_below is not None:
        summary_parts.append(
            f"Nearest high-volume node below: {nearest_hvn_below['price']:.2f} -- likely support."
        )
    if spot_in_lvn:
        summary_parts.append(
            "Spot sits in a low-volume node (air pocket) -- expect fast travel to the next high-volume node."
        )

    return {
        "spot": round(spot, 4),
        "range_low": round(lo, 4),
        "range_high": round(hi, 4),
        "poc": round(poc, 4),
        "value_area_low": round(value_area_low, 4),
        "value_area_high": round(value_area_high, 4),
        "high_volume_nodes": hvns,
        "low_volume_nodes": lvns,
        "nearest_hvn_above": nearest_hvn_above,
        "nearest_hvn_below": nearest_hvn_below,
        "spot_in_low_volume_node": spot_in_lvn,
        "summary": " ".join(summary_parts),
    }

def analyze_volume_profile_2(
    bars: list[dict],
    news_times: "list[str] | None" = None,
    date: "str | None" = None,
    spot: "float | None" = None,
    warmup_min: int = 15,
    spike_mult: float = 3.0,
    momentum_window: int = 5,
    news_window_min: int = 15,
    price_bins: int = 24,
    scattered_mult: float = 1.5,
    flat_bps: float = 8.0,
) -> dict:
    """Recent support/resistance from one session's intraday activity.

    Two complementary views of the same 1-minute yfinance bars:

    * The **volume profile** here is the minute-by-minute volume series (volume
      transacted in each 1-min bin). Its intraday *spikes* -- local maxima whose
      volume runs `spike_mult`x the session's median minute, past the opening
      `warmup_min` where volume is always heavy -- mark prices where size
      actually changed hands. Each spike is classed by how price momentum
      shifted across it: a swing from up into flat/down is a **supply** line
      (distribution -> resistance); down into flat/up is a **demand** line
      (accumulation -> support); a catalyst printing within `news_window_min`
      flags it **news_driven**; anything else is **unsure**.

    * The **price profile** is a volume-weighted histogram of price. Rebuilt
      *after* the spike bars' volume is removed, its remaining peaks are
      **scattered** levels -- prices that accumulated real size drip-by-drip
      rather than in one burst, hidden activity the spike scan alone misses.

    News resets the picture: once a news-driven spike is found, every peak that
    formed *before the most recent* one is dropped as stale (nothing is dropped
    when no spike was news-driven). Returns the surviving peaks
    (`{price, time, date, rel_vol, vol, type, news_driven}`) plus the nearest
    surviving support below and resistance above `spot`.
    """
    stamped = sorted(
        ((b, dt) for b in bars if (dt := clock.bar_dt(b)) is not None),
        key=lambda pair: pair[1],
    )
    if not stamped:
        return {"note": "intraday bars carry no usable timestamps"}

    session_date = (
        datetime.strptime(date, "%Y-%m-%d").date()
        if date
        else stamped[-1][1].astimezone(_ET).date()
    )
    open_et = datetime.combine(session_date, market_hours.MARKET_OPEN, tzinfo=_ET)
    close_et = datetime.combine(session_date, market_hours.MARKET_CLOSE, tzinfo=_ET)
    session = [
        (b, dt)
        for b, dt in stamped
        if open_et <= dt.astimezone(_ET) < close_et
        and dt.astimezone(_ET).date() == session_date
    ]
    if len(session) < warmup_min + 2 * momentum_window:
        return {"note": f"not enough regular-session bars for {session_date.isoformat()} to scan for spikes"}

    dts = [dt for _, dt in session]
    closes = [float(b["c"]) for b, _ in session]
    typicals = [(float(b["h"]) + float(b["l"]) + float(b["c"])) / 3.0 for b, _ in session]
    vols = [float(b.get("v") or 0.0) for b, _ in session]
    n = len(session)

    warmup_end = open_et + timedelta(minutes=warmup_min)
    eligible = [i for i in range(n) if dts[i] >= warmup_end]
    post_warmup_vols = [vols[i] for i in eligible]
    baseline = float(np.median([v for v in post_warmup_vols if v > 0])) if post_warmup_vols else 0.0
    if baseline <= 0:
        return {"note": "no post-warmup volume to baseline against; cannot scan for spikes"}

    news_dts = _news_datetimes(news_times)

    if spot is None:
        spot = closes[-1]

    def _et_hhmm(dt: datetime) -> str:
        return dt.astimezone(_ET).strftime("%H:%M")

    def _sign(bps: float) -> int:
        return 1 if bps > flat_bps else (-1 if bps < -flat_bps else 0)

    # --- Volume-profile spikes: local maxima >= spike_mult x baseline, with
    # non-maximum suppression so a single burst yields one peak, not a cluster.
    candidates = sorted(
        (i for i in eligible if vols[i] >= spike_mult * baseline),
        key=lambda i: vols[i],
        reverse=True,
    )
    spike_idxs: list[int] = []
    for i in candidates:
        if all(abs(i - j) > momentum_window for j in spike_idxs):
            spike_idxs.append(i)
    spike_idxs.sort()

    peaks: list[dict] = []
    for i in spike_idxs:
        w = momentum_window
        before = closes[i] - closes[max(0, i - w)]
        after = closes[min(n - 1, i + w)] - closes[i]
        before_bps = before / closes[max(0, i - w)] * 1e4 if closes[max(0, i - w)] else 0.0
        after_bps = after / closes[i] * 1e4 if closes[i] else 0.0
        bs, as_ = _sign(before_bps), _sign(after_bps)
        if bs > 0 and as_ <= 0:
            kind = "supply"
        elif bs < 0 and as_ >= 0:
            kind = "demand"
        else:
            kind = "unsure"
        news_driven = _news_near(news_dts, dts[i], news_window_min)
        peaks.append(
            {
                "price": round(typicals[i], 4),
                "time": _et_hhmm(dts[i]),
                "date": session_date.isoformat(),
                "rel_vol": round(vols[i] / baseline, 2),
                "vol": int(round(vols[i])),
                "type": kind,
                "news_driven": news_driven,
                "_dt": dts[i],
            }
        )

    # --- Price profile with the spike bars' volume removed, to surface levels
    # built up gradually ("scattered") rather than in the bursts above.
    lo = min(float(b["l"]) for b, _ in session)
    hi = max(float(b["h"]) for b, _ in session)
    spike_set = set(spike_idxs)
    half_bin = (hi - lo) / price_bins / 2.0 if hi > lo else 0.0
    if hi > lo:
        width = (hi - lo) / price_bins
        resid_vol = [0.0] * price_bins
        # Per bin, the non-spike minute that transacted the most volume -- a
        # representative timestamp for a level with no single moment of its own.
        bin_best: list["tuple[float, int] | None"] = [None] * price_bins
        for i in range(n):
            if i in spike_set:
                continue
            idx = min(price_bins - 1, max(0, int((typicals[i] - lo) / width)))
            resid_vol[idx] += vols[i]
            if bin_best[idx] is None or vols[i] > bin_best[idx][0]:
                bin_best[idx] = (vols[i], i)
        mean_bin = sum(resid_vol) / price_bins if price_bins else 0.0
        spike_prices = [p["price"] for p in peaks]
        for idx in range(price_bins):
            v = resid_vol[idx]
            if mean_bin <= 0 or v < scattered_mult * mean_bin:
                continue
            # Keep only local maxima so a broad node yields one scattered peak.
            left = resid_vol[idx - 1] if idx > 0 else -1.0
            right = resid_vol[idx + 1] if idx < price_bins - 1 else -1.0
            if v < left or v < right:
                continue
            center = lo + (idx + 0.5) * width
            # Skip a level a spike already owns (same price bin) -- it is not
            # hidden activity, just the residue of that spike's own bar.
            if any(abs(center - sp) <= half_bin for sp in spike_prices):
                continue
            best = bin_best[idx]
            when = dts[best[1]] if best else dts[-1]
            peaks.append(
                {
                    "price": round(center, 4),
                    "time": _et_hhmm(when),
                    "date": session_date.isoformat(),
                    "rel_vol": round(v / baseline, 2),
                    "vol": int(round(v)),
                    "type": "scattered",
                    "news_driven": False,
                    "_dt": when,
                }
            )

    peaks.sort(key=lambda p: p["_dt"])

    # --- Drop everything before the freshest catalyst: levels that formed ahead
    # of the most recent news-driven spike are stale once the news re-priced it.
    news_spike_dts = [p["_dt"] for p in peaks if p["news_driven"]]
    removed = 0
    if news_spike_dts:
        cutoff = max(news_spike_dts)
        kept = [p for p in peaks if p["_dt"] >= cutoff]
        removed = len(peaks) - len(kept)
        peaks = kept

    for p in peaks:
        p.pop("_dt", None)

    supports = [p for p in peaks if p["price"] < spot]
    resistances = [p for p in peaks if p["price"] > spot]
    nearest_support = max(supports, key=lambda p: p["price"]) if supports else None
    nearest_resistance = min(resistances, key=lambda p: p["price"]) if resistances else None

    by_type: dict[str, int] = {}
    for p in peaks:
        by_type[p["type"]] = by_type.get(p["type"], 0) + 1
    summary_parts = [
        f"{len(peaks)} support/resistance level(s) for {session_date.isoformat()} "
        f"from {n} regular-session bars (spot {spot:.2f}): "
        + (", ".join(f"{c} {t}" for t, c in sorted(by_type.items())) or "none")
        + "."
    ]
    if removed:
        summary_parts.append(f"Dropped {removed} pre-catalyst level(s) ahead of the latest news-driven spike.")
    if nearest_resistance is not None:
        summary_parts.append(
            f"Nearest resistance above: {nearest_resistance['price']:.2f} "
            f"({nearest_resistance['type']}, {nearest_resistance['rel_vol']}x)."
        )
    if nearest_support is not None:
        summary_parts.append(
            f"Nearest support below: {nearest_support['price']:.2f} "
            f"({nearest_support['type']}, {nearest_support['rel_vol']}x)."
        )

    # A profile whose price range doesn't even contain spot cannot be
    # describing the tape being traded (wrong session, stale fetch, bad data
    # source) -- surface that loudly instead of returning a normal-looking map
    # whose every level silently sits on one side of the price. The 1% slack
    # absorbs the source's ~15-minute delay, so a genuine breakout just past
    # the mapped range does not read as a corrupt map.
    slack = spot * 0.01
    brackets_spot = bool(lo - slack <= spot <= hi + slack)
    warning = None
    if not brackets_spot:
        warning = (
            f"WARNING: spot {spot:.2f} lies well outside this profile's price range "
            f"({lo:.2f}-{hi:.2f}) -- the map does not describe the current tape "
            "(wrong or stale session data). Treat every level here as invalid "
            "until a re-run brackets spot."
        )
        summary_parts.append(warning)

    return {
        "date": session_date.isoformat(),
        "spot": round(spot, 4),
        "baseline_minute_volume": int(round(baseline)),
        "range_low": round(lo, 4),
        "range_high": round(hi, 4),
        "brackets_spot": brackets_spot,
        "warning": warning,
        "peaks": peaks,
        "nearest_support": nearest_support,
        "nearest_resistance": nearest_resistance,
        "removed_pre_news": removed,
        "summary": " ".join(summary_parts),
    }

def floor_pivots(daily_bars: "list[dict] | None", spot: "float | None" = None, today: "str | None" = None) -> dict:
    """Classic floor-trader pivot levels from the last completed session.

    P = (H+L+C)/3 from the prior day's bar, with the standard R1-R3 above and
    S1-S3 below. These are formula levels rather than structure, but they are
    watched widely enough to act as intraday reaction points -- and a pivot
    that coincides with a structural level (session high, swing cluster, HVN)
    is reinforced. Splits the levels around `spot` like key_levels does.
    """
    today = today or clock.now().strftime("%Y-%m-%d")
    prior = [
        b for b in (daily_bars or []) if str(b.get("t", ""))[:10] and str(b.get("t", ""))[:10] < today
    ]
    if not prior:
        return {"note": "no completed prior-day daily bar available for pivot levels"}

    bar = prior[-1]
    high, low, close = float(bar["h"]), float(bar["l"]), float(bar["c"])
    pivot = (high + low + close) / 3.0
    levels = {
        "r3": high + 2 * (pivot - low),
        "r2": pivot + (high - low),
        "r1": 2 * pivot - low,
        "pivot": pivot,
        "s1": 2 * pivot - high,
        "s2": pivot - (high - low),
        "s3": low - 2 * (high - pivot),
    }

    result: dict = {
        "prior_day_date": str(bar.get("t", ""))[:10],
        "levels": {name: round(value, 4) for name, value in levels.items()},
    }
    summary_parts = [
        f"Floor pivots from {result['prior_day_date']} (H {high:.2f} / L {low:.2f} / C {close:.2f}): "
        f"P {pivot:.2f}, R1 {levels['r1']:.2f}, R2 {levels['r2']:.2f}, S1 {levels['s1']:.2f}, S2 {levels['s2']:.2f}."
    ]

    if spot is not None:
        def _entry(name: str) -> dict:
            return {
                "name": name,
                "level": round(levels[name], 4),
                "distance_pct": round((levels[name] / spot - 1) * 100, 2),
            }

        resistance = sorted(
            (_entry(name) for name, value in levels.items() if value > spot),
            key=lambda e: e["level"],
        )
        support = sorted(
            (_entry(name) for name, value in levels.items() if value <= spot),
            key=lambda e: -e["level"],
        )
        result["spot"] = round(spot, 4)
        result["resistance_above"] = resistance
        result["support_below"] = support
        result["nearest_resistance"] = resistance[0] if resistance else None
        result["nearest_support"] = support[0] if support else None
        if resistance:
            summary_parts.append(
                f"Spot {spot:.2f}: nearest pivot resistance {resistance[0]['name'].upper()} "
                f"at {resistance[0]['level']:.2f} ({resistance[0]['distance_pct']:+.2f}%)."
            )
        else:
            summary_parts.append(f"Spot {spot:.2f} is above every pivot level (including R3).")

    result["summary"] = " ".join(summary_parts)
    return result
