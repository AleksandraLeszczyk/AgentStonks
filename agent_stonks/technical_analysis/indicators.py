"""Classic indicators, and the reads built directly on them.

The base layer of the package: every other module here imports from this one
and nothing here imports from them. Trend, momentum, volatility, VWAP bands,
volume participation and consolidation -- the arithmetic a human technical
analyst would do first, returned as labeled values plus a one-line summary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import clock, market_hours

_ET = market_hours.MARKET_TZ


def _closes(bars: list[dict]) -> pd.Series:
    return pd.Series([b["c"] for b in bars], dtype=float)

def sma(series: pd.Series, period: int) -> "float | None":
    if len(series) < period:
        return None
    return float(series.tail(period).mean())

def rsi(series: pd.Series, period: int = 14) -> "float | None":
    """Wilder's RSI over the trailing `period` bars."""
    if len(series) < period + 1:
        return None
    delta = series.diff().dropna()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean().iloc[-1]
    avg_loss = loss.rolling(period).mean().iloc[-1]
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100 - (100 / (1 + rs)))

def _rsi_label(value: float) -> str:
    if value >= 70:
        return "overbought"
    if value <= 30:
        return "oversold"
    return "neutral"

def atr(bars: list[dict], period: int = 14) -> "float | None":
    """Average True Range over the trailing `period` bars."""
    if len(bars) < period + 1:
        return None
    df = pd.DataFrame(bars)
    prev_close = df["c"].shift(1)
    true_range = pd.concat(
        [
            df["h"] - df["l"],
            (df["h"] - prev_close).abs(),
            (df["l"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    value = true_range.rolling(period).mean().iloc[-1]
    return float(value) if pd.notna(value) else None

def adx(bars: list[dict], period: int = 14) -> "float | None":
    """Wilder's Average Directional Index over the trailing bars.

    ADX measures *trend strength* irrespective of direction. A reading below
    ~20 marks a rangebound, non-trending tape -- the regime VWAP mean-reversion
    needs, where price oscillates around VWAP. Above ~25 a real trend is under
    way and VWAP becomes a trend line rather than a mean, so fading stretches
    away from it stops working. Uses simple rolling means for the directional
    smoothing, matching this module's ATR convention.
    """
    if len(bars) < 2 * period:
        return None
    df = pd.DataFrame(bars)
    high, low, close = df["h"], df["l"], df["c"]

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)

    atr_ = true_range.rolling(period).mean()
    plus_di = 100 * plus_dm.rolling(period).mean() / atr_
    minus_di = 100 * minus_dm.rolling(period).mean() / atr_
    di_sum = (plus_di + minus_di).replace(0, pd.NA)
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    value = dx.rolling(period).mean().iloc[-1]
    return float(value) if pd.notna(value) else None

def _adx_label(value: float) -> str:
    if value < 20:
        return "ranging (no trend)"
    if value < 25:
        return "weak / developing trend"
    return "trending"

def obv_trend(bars: list[dict], window: int = 10) -> "str | None":
    """Direction of on-balance volume over the trailing `window` bars."""
    if len(bars) < window + 1:
        return None
    df = pd.DataFrame(bars)
    direction = df["c"].diff().apply(lambda d: 1 if d > 0 else (-1 if d < 0 else 0))
    obv = (direction * df["v"]).cumsum()
    slope = obv.iloc[-1] - obv.iloc[-window]
    if slope > 0:
        return "rising"
    if slope < 0:
        return "falling"
    return "flat"

def _ma_alignment(price: float, sma20: "float | None", sma50: "float | None", sma200: "float | None") -> str:
    values = [("price", price)]
    for name, val in (("sma20", sma20), ("sma50", sma50), ("sma200", sma200)):
        if val is not None:
            values.append((name, val))
    if len(values) < 2:
        return "insufficient data for moving average alignment"

    nums = [v for _, v in values]
    if all(nums[i] >= nums[i + 1] for i in range(len(nums) - 1)):
        return "bullish stack (" + " > ".join(n for n, _ in values) + ")"
    if all(nums[i] <= nums[i + 1] for i in range(len(nums) - 1)):
        return "bearish stack (" + " < ".join(n for n, _ in values) + ")"
    return "mixed, no clear stack (" + ", ".join(f"{n}={v:.2f}" for n, v in values) + ")"

def support_resistance(bars: list[dict], lookback: int = 20) -> dict:
    window = bars[-lookback:] if len(bars) >= lookback else bars
    return {
        "support": min(b["l"] for b in window),
        "resistance": max(b["h"] for b in window),
        "lookback_bars": len(window),
    }

def analyze_trend(bars: list[dict]) -> dict:
    """Medium/long-term regime read for daily bars: direction, strength, key levels."""
    if len(bars) < 5:
        return {"note": "not enough bars for trend analysis"}

    closes = _closes(bars)
    price = float(closes.iloc[-1])
    period_start = float(closes.iloc[0])
    pct_change = (price / period_start - 1) * 100 if period_start else 0.0

    period_high = max(b["h"] for b in bars)
    period_low = min(b["l"] for b in bars)
    range_pct = (price - period_low) / (period_high - period_low) * 100 if period_high > period_low else 50.0

    sma20 = sma(closes, 20)
    sma50 = sma(closes, 50)
    sma200 = sma(closes, 200)
    rsi14 = rsi(closes, 14)

    def _vote(a: float, b: float) -> int:
        return 1 if a > b else -1 if a < b else 0

    score = 0
    if sma20 is not None:
        score += _vote(price, sma20)
    if sma20 is not None and sma50 is not None:
        score += _vote(sma20, sma50)
    if sma200 is not None:
        score += _vote(price, sma200)
    if pct_change > 1:
        score += 1
    elif pct_change < -1:
        score -= 1

    if score >= 2:
        regime = "bullish"
    elif score <= -2:
        regime = "bearish"
    else:
        regime = "neutral"
    strength = "strong" if abs(score) >= 3 else "moderate" if abs(score) == 2 else "weak"

    levels = support_resistance(bars, lookback=min(20, len(bars)))
    alignment = _ma_alignment(price, sma20, sma50, sma200)

    summary_parts = [
        f"{regime.capitalize()} regime ({strength}): price {pct_change:+.1f}% over {len(bars)} bars, "
        f"at {range_pct:.0f}% of the {period_low:.2f}-{period_high:.2f} period range.",
        f"Moving averages: {alignment}.",
    ]
    if rsi14 is not None:
        summary_parts.append(f"RSI(14) {rsi14:.0f} ({_rsi_label(rsi14)}).")
    summary_parts.append(f"Recent support {levels['support']:.2f}, resistance {levels['resistance']:.2f}.")

    return {
        "regime": regime,
        "trend_strength": strength,
        "pct_change_over_period": round(pct_change, 2),
        "price_position_in_range_pct": round(range_pct, 1),
        "sma20": sma20,
        "sma50": sma50,
        "sma200": sma200,
        "moving_average_alignment": alignment,
        "rsi_14": rsi14,
        "rsi_label": _rsi_label(rsi14) if rsi14 is not None else None,
        "support": levels["support"],
        "resistance": levels["resistance"],
        "summary": " ".join(summary_parts),
    }

def _hl_momentum_pattern(bars: list[dict]) -> str:
    """Classify a run of bars as higher-highs/higher-lows (uptrend), lower-highs/
    lower-lows (downtrend), or choppy, by comparing the most recent swing window
    against the one before it. This is the same read for any session -- today's
    window, yesterday's, or a full day -- so it lives as a shared helper."""
    if len(bars) < 4:
        return "not enough bars to classify high/low pattern"
    swing = min(10, len(bars) // 2)
    recent_highs = [b["h"] for b in bars[-swing:]]
    recent_lows = [b["l"] for b in bars[-swing:]]
    prior_highs = [b["h"] for b in bars[-2 * swing : -swing]] if len(bars) >= 2 * swing else []
    prior_lows = [b["l"] for b in bars[-2 * swing : -swing]] if len(bars) >= 2 * swing else []
    if not (prior_highs and prior_lows):
        return "not enough bars to classify high/low pattern"
    higher_highs = max(recent_highs) > max(prior_highs)
    higher_lows = min(recent_lows) > min(prior_lows)
    lower_highs = max(recent_highs) < max(prior_highs)
    lower_lows = min(recent_lows) < min(prior_lows)
    if higher_highs and higher_lows:
        return "making higher highs and higher lows (uptrend)"
    if lower_highs and lower_lows:
        return "making lower highs and lower lows (downtrend)"
    return "choppy / no consistent higher-high or lower-low pattern"

def _session_momentum_block(bars: "list[dict] | None") -> "dict | None":
    """Compact momentum read over a whole session (or any bar run): net move
    open->close plus the higher-high/lower-low pattern. Returns None when there
    aren't enough bars to say anything."""
    if not bars or len(bars) < 2:
        return None
    closes = _closes(bars)
    start = float(closes.iloc[0])
    end = float(closes.iloc[-1])
    pct = (end / start - 1) * 100 if start else 0.0
    return {
        "pct_change": round(pct, 2),
        "momentum_pattern": _hl_momentum_pattern(bars),
        "bars": len(bars),
    }

def _segment_sse(y: np.ndarray, a: int, b: int) -> float:
    """Sum of squared residuals of the least-squares line fit to y[a:b+1]."""
    if b - a < 2:
        return 0.0
    xs = np.arange(a, b + 1, dtype=float)
    ys = y[a : b + 1]
    coeffs = np.polyfit(xs, ys, 1)
    resid = ys - np.polyval(coeffs, xs)
    return float(np.dot(resid, resid))

def piecewise_regimes(
    closes: pd.Series,
    tol_pct: float = 0.15,
    flat_pct: float = 0.08,
) -> list[dict]:
    """Segment a close-price series into consecutive momentum regimes with
    bottom-up piecewise-linear regression.

    Starting from the finest partition, adjacent segments are greedily merged as
    long as the merged segment's per-bar RMSE stays under `tol_pct`% of price --
    so a straight run of bars collapses into one leg and a genuine turn stays a
    boundary. Each regime carries its slope, net % move, and an up/down/flat
    direction (flat when the leg's total move is under `flat_pct`% of price).
    Returned oldest-first; the last entry is the leg in force right now.
    """
    y = np.asarray(closes, dtype=float)
    n = len(y)
    if n < 3:
        return []

    price_scale = float(np.median(np.abs(y))) or 1.0
    tol = tol_pct / 100.0 * price_scale  # RMSE ceiling, in price units

    # Finest partition: contiguous 2-bar segments as [start, end] index pairs.
    segs: list[list[int]] = [[i, min(i + 1, n - 1)] for i in range(0, n - 1, 2)]
    if segs[-1][1] < n - 1:
        segs[-1][1] = n - 1

    while len(segs) > 1:
        best_j = 0
        best_rmse = float("inf")
        for j in range(len(segs) - 1):
            a, b = segs[j][0], segs[j + 1][1]
            rmse = (_segment_sse(y, a, b) / (b - a + 1)) ** 0.5
            if rmse < best_rmse:
                best_rmse = rmse
                best_j = j
        if best_rmse > tol:
            break
        segs[best_j] = [segs[best_j][0], segs[best_j + 1][1]]
        del segs[best_j + 1]

    flat_move = flat_pct / 100.0 * price_scale
    regimes: list[dict] = []
    for a, b in segs:
        xs = np.arange(a, b + 1, dtype=float)
        slope = float(np.polyfit(xs, y[a : b + 1], 1)[0]) if b > a else 0.0
        net = float(y[b] - y[a])
        if abs(net) < flat_move:
            direction = "flat"
        elif net > 0:
            direction = "up"
        else:
            direction = "down"
        base = float(y[a])
        regimes.append(
            {
                "start_index": a,
                "end_index": b,
                "bars": b - a + 1,
                "direction": direction,
                "slope_per_bar": round(slope, 6),
                "pct_move": round((net / base) * 100, 2) if base else 0.0,
            }
        )
    return regimes

def analyze_intraday(
    bars: list[dict],
    *,
    prev_session_bars: "list[dict] | None" = None,
    full_session_bars: "list[dict] | None" = None,
    market_return_pct: "float | None" = None,
    beta: "float | None" = None,
    beta_window_days: "int | None" = None,
) -> dict:
    """Short-term price-action read for intraday bars: momentum, position vs VWAP, volatility.

    Optional context enriches the read:
      * `prev_session_bars` -- yesterday's intraday bars, for yesterday's momentum.
      * `full_session_bars` -- today's complete session (the `bars` argument may be
        only a recent slice), for today's total momentum and for anchoring the
        piecewise regime detection that measures how long the current leg has run.
      * `market_return_pct` + `beta` -- the market's (SPY) % move over the same
        recent window and the ticker's long-term beta to it, used to strip the
        broad-market component out and report the idiosyncratic momentum.
    """
    if len(bars) < 5:
        return {"note": "not enough bars for intraday analysis"}

    closes = _closes(bars)
    price = float(closes.iloc[-1])
    session_start = float(closes.iloc[0])
    pct_change = (price / session_start - 1) * 100 if session_start else 0.0

    momentum = _hl_momentum_pattern(bars)

    vwap_note = None
    if "vw" in bars[-1] and bars[-1]["vw"]:
        vwap = float(bars[-1]["vw"])
        vwap_diff_pct = (price - vwap) / vwap * 100 if vwap else 0.0
        vwap_note = f"price is {abs(vwap_diff_pct):.2f}% {'above' if vwap_diff_pct >= 0 else 'below'} session VWAP"

    atr_value = atr(bars, period=min(14, len(bars) - 1))
    volatility_pct = (atr_value / price * 100) if (atr_value is not None and price) else None

    # Yesterday's momentum: the same higher-high/lower-low read over the prior
    # session, so the agent can tell whether today continues or reverses it.
    yesterday_block = _session_momentum_block(prev_session_bars)

    # Today's total momentum: measured over the full session, not just the
    # recent `bars` window, so a fade off the highs doesn't hide a big up day.
    session_bars = full_session_bars if (full_session_bars and len(full_session_bars) >= 2) else bars
    today_block = _session_momentum_block(session_bars)

    # How long the current momentum has lasted: piecewise-linear regime
    # detection over the session, reporting the leg in force right now.
    current_leg = None
    regimes = piecewise_regimes(_closes(session_bars)) if len(session_bars) >= 3 else []
    if regimes:
        leg = regimes[-1]
        current_leg = {
            "direction": leg["direction"],
            "bars": leg["bars"],
            "pct_move": leg["pct_move"],
            "slope_per_bar": leg["slope_per_bar"],
            "regimes_in_session": len(regimes),
        }
        start_t = clock.bar_dt(session_bars[leg["start_index"]])
        end_t = clock.bar_dt(session_bars[leg["end_index"]])
        if start_t and end_t:
            current_leg["duration_minutes"] = round((end_t - start_t).total_seconds() / 60.0, 1)
            current_leg["started_at"] = start_t.astimezone(_ET).strftime("%Y-%m-%d %H:%M ET")

    # Market-neutral momentum: strip the broad market's move (beta-scaled) out of
    # this ticker's move so what's left is the stock's own, idiosyncratic push.
    market_neutral = None
    if market_return_pct is not None and beta is not None:
        residual = pct_change - beta * market_return_pct
        market_component = beta * market_return_pct
        if abs(residual) < 0.02:
            mn_dir = "flat"
        elif residual > 0:
            mn_dir = "up"
        else:
            mn_dir = "down"
        market_neutral = {
            "residual_pct": round(residual, 3),
            "direction": mn_dir,
            "beta": round(beta, 3),
            "market_return_pct": round(market_return_pct, 3),
            "market_component_pct": round(market_component, 3),
            "beta_window_days": beta_window_days,
            "note": (
                "Momentum with the broad-market move removed: this ticker moved "
                f"{pct_change:+.2f}% while beta*market accounts for "
                f"{market_component:+.2f}%, leaving {residual:+.2f}% idiosyncratic."
            ),
        }

    summary_parts = [
        f"Price {pct_change:+.2f}% since the start of this window, {momentum}.",
    ]
    if vwap_note:
        summary_parts.append(vwap_note.capitalize() + ".")
    if volatility_pct is not None:
        summary_parts.append(f"ATR-based volatility ~{volatility_pct:.2f}% of price.")
    if today_block:
        summary_parts.append(f"Today total {today_block['pct_change']:+.2f}%.")
    if yesterday_block:
        summary_parts.append(f"Yesterday {yesterday_block['pct_change']:+.2f}%.")
    if current_leg:
        dur = (
            f"{current_leg['duration_minutes']:.0f}m"
            if "duration_minutes" in current_leg
            else f"{current_leg['bars']} bars"
        )
        summary_parts.append(
            f"Current {current_leg['direction']} leg has run {dur} ({current_leg['pct_move']:+.2f}%)."
        )
    if market_neutral:
        summary_parts.append(
            f"Market-neutral (beta {market_neutral['beta']}): {market_neutral['residual_pct']:+.2f}%."
        )

    return {
        "pct_change_in_window": round(pct_change, 2),
        "momentum_pattern": momentum,
        "vwap_position": vwap_note,
        "atr": atr_value,
        "volatility_pct_of_price": round(volatility_pct, 2) if volatility_pct is not None else None,
        "yesterday_momentum": yesterday_block,
        "today_total_momentum": today_block,
        "current_momentum_duration": current_leg,
        "market_neutral_momentum": market_neutral,
        "summary": " ".join(summary_parts),
    }

def _session_bars(bars: list[dict]) -> list[dict]:
    """The subset of bars belonging to the latest bar's calendar day.

    VWAP is session-anchored -- it resets each trading day -- so the bands are
    only meaningful over today's bars. Falls back to all bars when timestamps
    are missing (e.g. synthetic/test bars) so the math still runs.
    """
    today = str(bars[-1].get("t", ""))[:10]
    if not today:
        return bars
    day_bars = [b for b in bars if str(b.get("t", ""))[:10] == today]
    return day_bars or bars

def _rejection_candle(bar: dict) -> "str | None":
    """Classify a bar as a bullish/bearish rejection (long-tail) candle.

    A long lower wick with a close in the upper part of the range is buyers
    rejecting lower prices (bullish); a long upper wick with a close in the
    lower part is sellers rejecting higher prices (bearish). These are the
    confirmation candles a mean-reversion trader wants to see right at a band.
    """
    o, h, l, c = float(bar["o"]), float(bar["h"]), float(bar["l"]), float(bar["c"])
    rng = h - l
    if rng <= 0:
        return None
    body = abs(c - o)
    lower_wick = min(o, c) - l
    upper_wick = h - max(o, c)
    # A genuine rejection: one tail dominates the bar and the body is small.
    if lower_wick >= rng * 0.5 and lower_wick > body and upper_wick < lower_wick:
        return "bullish_rejection"
    if upper_wick >= rng * 0.5 and upper_wick > body and lower_wick < upper_wick:
        return "bearish_rejection"
    return None

def analyze_vwap_bands(bars: list[dict], num_std: float = 2.0) -> dict:
    """VWAP mean-reversion read: session VWAP, standard-deviation bands, and the
    ranging-vs-trending regime that decides whether fading a stretch is valid.

    Computes the session-anchored VWAP and the volume-weighted standard
    deviation of price around it, then expresses where price sits as a signed
    z-score (number of std devs from VWAP). A setup only exists when ADX
    confirms a range (below 20) AND price has stretched at least `num_std`
    std devs from VWAP -- long below, short above -- ideally with a rejection
    candle at the band. In a trending tape (ADX rising through 25) VWAP is a
    trend line, not a mean, and stretches are not faded.
    """
    if len(bars) < 5:
        return {"note": "not enough bars for VWAP band analysis"}

    session = _session_bars(bars)
    if len(session) < 5:
        return {"note": "not enough bars in today's session yet for VWAP bands"}

    df = pd.DataFrame(session)
    typical = (df["h"] + df["l"] + df["c"]) / 3.0
    vol = df["v"].astype(float)
    cum_vol = vol.cumsum()
    if cum_vol.iloc[-1] <= 0:
        return {"note": "no traded volume in session bars; cannot compute VWAP"}

    vwap_series = (typical * vol).cumsum() / cum_vol
    variance = ((typical - vwap_series) ** 2 * vol).cumsum() / cum_vol
    std = float(variance.iloc[-1]) ** 0.5
    vwap = float(vwap_series.iloc[-1])
    price = float(df["c"].iloc[-1])

    z = (price - vwap) / std if std > 0 else 0.0

    adx_value = adx(session, period=min(14, len(session) // 2))
    is_ranging = adx_value is not None and adx_value < 20
    rejection = _rejection_candle(session[-1])

    if is_ranging and z <= -num_std:
        signal = "long_setup"
    elif is_ranging and z >= num_std:
        signal = "short_setup"
    elif adx_value is not None and adx_value >= 25 and abs(z) >= num_std:
        # Stretched, but the tape is trending -- this is exactly the failure mode
        # where VWAP becomes a trend line and fading it bleeds.
        signal = "no_setup_trending"
    else:
        signal = "no_setup"

    def _bands(k: float) -> "tuple[float, float]":
        return round(vwap + k * std, 4), round(vwap - k * std, 4)

    upper1, lower1 = _bands(1.0)
    upper2, lower2 = _bands(2.0)
    upper3, lower3 = _bands(3.0)

    summary_parts = [
        f"Price {price:.2f} is {abs(z):.1f} std devs "
        f"{'above' if z >= 0 else 'below'} session VWAP {vwap:.2f} (1σ={std:.3f}).",
    ]
    if adx_value is not None:
        summary_parts.append(f"ADX {adx_value:.0f} -- {_adx_label(adx_value)}.")
    else:
        summary_parts.append("ADX unavailable (too few bars) -- range not confirmed.")
    if signal == "long_setup":
        summary_parts.append(
            f"Long mean-reversion setup: oversold ≥{num_std}σ below VWAP in a confirmed range, "
            f"target VWAP {vwap:.2f}, stop below {lower3:.2f}."
        )
    elif signal == "short_setup":
        summary_parts.append(
            f"Short mean-reversion setup: overbought ≥{num_std}σ above VWAP in a confirmed range, "
            f"target VWAP {vwap:.2f} (long-only accounts trim/exit here rather than short)."
        )
    elif signal == "no_setup_trending":
        summary_parts.append("Stretched from VWAP but ADX shows a trend -- do not fade; VWAP is acting as a trend line.")
    else:
        summary_parts.append("No setup: price is not stretched far enough from VWAP, or the range is unconfirmed.")
    if rejection:
        summary_parts.append(f"Latest bar is a {rejection.replace('_', ' ')} candle.")

    return {
        "vwap": round(vwap, 4),
        "price": round(price, 4),
        "std_dev": round(std, 4),
        "z_score": round(z, 2),
        "num_std_trigger": num_std,
        "upper_band_1sd": upper1,
        "lower_band_1sd": lower1,
        "upper_band_2sd": upper2,
        "lower_band_2sd": lower2,
        "upper_band_3sd": upper3,
        "lower_band_3sd": lower3,
        "adx": round(adx_value, 1) if adx_value is not None else None,
        "adx_label": _adx_label(adx_value) if adx_value is not None else None,
        "is_ranging": is_ranging,
        "rejection_candle": rejection,
        "signal": signal,
        "session_bars": len(session),
        "summary": " ".join(summary_parts),
    }

# Participation gate: session pace AND local participation must both clear.
# Pace alone passed at every losing tactic entry in the SimLab run store --
# it stays elevated all session once a name is busy, so it cannot tell a move
# buyers are still in from one they have already left.
PARTICIPATION_MIN_PACE = 2.0

PARTICIPATION_MIN_LOCAL = 1.2

# Bars behind `volume_burst` -- short enough to catch volume arriving now.
VOLUME_BURST_BARS = 3

def analyze_volume(
    bars: list[dict],
    rvol_pace: "float | None" = None,
    partial_volume_feed: bool = False,
) -> dict:
    """Volume confirmation read: is participation backing the recent price move?

    `rvol_pace` is the time-of-day-adjusted relative volume (today's cumulative
    volume vs an average day's cumulative at this minute -- see
    state.rvol_pace); when provided it is surfaced as the primary participation
    gauge, since the local 10-bar-vs-10-bar `relative_volume` only measures the
    last few minutes against the few minutes before them. `partial_volume_feed`
    marks single-venue (IEX) volume, a small sample of the consolidated tape.

    The two gauges measure genuinely different things and pass at different
    times: session pace stays elevated all day once a name is busy, so on its
    own it waves through moves buyers have already left. `participation_ok`
    resolves the pair into the single answer callers actually want -- pace
    elevated AND local participation rising -- and `volume_burst` adds the
    shortest-horizon read, the last few bars against the 10-bar baseline.
    """
    if not bars:
        return {"note": "no intraday bars available yet"}

    volumes = [b.get("v", 0) for b in bars]
    recent = volumes[-10:]
    prior = volumes[-20:-10] if len(volumes) >= 20 else volumes[:-10]
    recent_avg = sum(recent) / len(recent) if recent else 0.0
    prior_avg = sum(prior) / len(prior) if prior else 0.0

    if prior_avg > 0:
        relative_volume = recent_avg / prior_avg
    else:
        relative_volume = None

    # Is volume arriving RIGHT NOW? The last few bars against the 10-bar
    # baseline, on a window short enough to catch a burst that the 10-vs-10
    # ratio still has mostly ahead of it.
    burst_window = volumes[-VOLUME_BURST_BARS:]
    burst_avg = sum(burst_window) / len(burst_window) if burst_window else 0.0
    volume_burst = burst_avg / recent_avg if recent_avg > 0 else None
    volume_trend = (
        "increasing" if recent_avg > prior_avg else "decreasing" if recent_avg < prior_avg else "flat"
    )

    closes = _closes(bars)
    price_pct_change = 0.0
    if len(closes) >= 10:
        price_pct_change = (float(closes.iloc[-1]) / float(closes.iloc[-10]) - 1) * 100

    flow_trend = obv_trend(bars, window=min(10, len(bars) - 1))

    price_up = price_pct_change > 0.1
    price_down = price_pct_change < -0.1
    volume_up = volume_trend == "increasing"
    if (price_up and volume_up) or (price_down and volume_up):
        confirmation = "confirming"
    elif (price_up or price_down) and volume_trend == "decreasing":
        confirmation = "diverging (move not backed by rising volume)"
    else:
        confirmation = "inconclusive"

    # The two gauges answer different questions and are routinely confused for
    # each other, so settle it here rather than leaving the caller to combine
    # them: session pace elevated AND local participation rising right now.
    if rvol_pace is None or relative_volume is None:
        participation_ok = None
    else:
        participation_ok = (
            rvol_pace >= PARTICIPATION_MIN_PACE
            and relative_volume >= PARTICIPATION_MIN_LOCAL
        )

    summary_parts = []
    if rvol_pace is not None:
        summary_parts.append(
            f"Time-of-day-adjusted relative volume (rvol_pace) is {rvol_pace:.2f}x an average "
            f"day's pace ({'clearly elevated' if rvol_pace >= 1.5 else 'not elevated'} participation)."
        )
    summary_parts.append(
        f"Local volume is {volume_trend} (last 10 bars avg {recent_avg:,.0f} vs prior 10 avg {prior_avg:,.0f}"
        + (f", {relative_volume:.1f}x" if relative_volume is not None else "")
        + ")."
    )
    if volume_burst is not None:
        summary_parts.append(
            f"Last {VOLUME_BURST_BARS} bars are running {volume_burst:.2f}x the 10-bar average "
            f"({'volume arriving now' if volume_burst >= 1.2 else 'no fresh volume'})."
        )
    if participation_ok is not None:
        summary_parts.append(
            f"Participation gate ({PARTICIPATION_MIN_PACE:.1f}x pace AND "
            f"{PARTICIPATION_MIN_LOCAL:.1f}x local): "
            f"{'PASS' if participation_ok else 'FAIL'}."
        )
    if flow_trend:
        summary_parts.append(f"On-balance volume is {flow_trend}.")
    summary_parts.append(f"Volume is {confirmation} relative to the {price_pct_change:+.2f}% price move.")
    if partial_volume_feed:
        summary_parts.append(
            "Note: volumes are from the IEX feed only (a few percent of the consolidated "
            "tape) -- treat absolute volume levels and small-sample ratios as directional, "
            "not precise."
        )

    return {
        "bar_count": len(volumes),
        "rvol_pace": round(rvol_pace, 2) if rvol_pace is not None else None,
        "recent_10bar_avg_volume": recent_avg,
        "prior_10bar_avg_volume": prior_avg,
        "relative_volume": round(relative_volume, 2) if relative_volume is not None else None,
        "volume_burst": round(volume_burst, 2) if volume_burst is not None else None,
        "participation_ok": participation_ok,
        "volume_trend": volume_trend,
        "obv_trend": flow_trend,
        "price_pct_change_10bar": round(price_pct_change, 2),
        "confirmation": confirmation,
        "partial_volume_feed": partial_volume_feed or None,
        "summary": " ".join(summary_parts),
    }

def analyze_consolidation(bars: list[dict], base_bars: int = 10, prior_bars: int = 20) -> dict:
    """Tight-base / coiling read for breakout setups.

    Splits the window into the candidate base (the most recent `base_bars`)
    and the window before it, then checks the three things a breakout trader
    wants before a level break is worth trusting: the base's range has
    contracted vs what came before, volume inside the base is declining (not
    rising), and the base's high/low have been tested more than once (more
    touches = more energy coiled under the level). `base_height` (the base's
    high minus low) is returned for projecting targets after a breakout.
    """
    if len(bars) < base_bars + 5:
        return {"note": "not enough bars to assess a base/consolidation"}

    base = bars[-base_bars:]
    prior_window = (
        bars[-(base_bars + prior_bars) : -base_bars] if len(bars) >= base_bars + prior_bars else bars[: -base_bars]
    )

    base_high = max(b["h"] for b in base)
    base_low = min(b["l"] for b in base)
    base_height = base_high - base_low
    last_price = float(base[-1]["c"])
    base_height_pct = (base_height / last_price * 100) if last_price else 0.0

    prior_high = max(b["h"] for b in prior_window) if prior_window else base_high
    prior_low = min(b["l"] for b in prior_window) if prior_window else base_low
    prior_range = prior_high - prior_low
    range_contraction_pct = ((prior_range - base_height) / prior_range * 100) if prior_range else 0.0

    base_avg_volume = sum(b.get("v", 0) for b in base) / len(base)
    prior_avg_volume = sum(b.get("v", 0) for b in prior_window) / len(prior_window) if prior_window else base_avg_volume
    if prior_avg_volume:
        if base_avg_volume < prior_avg_volume * 0.9:
            volume_trend_in_base = "declining"
        elif base_avg_volume > prior_avg_volume * 1.1:
            volume_trend_in_base = "rising"
        else:
            volume_trend_in_base = "flat"
    else:
        volume_trend_in_base = "unknown"

    touch_tolerance = max(base_height * 0.15, last_price * 0.001) if base_height else last_price * 0.001
    touches_at_resistance = sum(1 for b in base if base_high - b["h"] <= touch_tolerance)
    touches_at_support = sum(1 for b in base if b["l"] - base_low <= touch_tolerance)
    well_tested = touches_at_resistance >= 2 or touches_at_support >= 2

    is_coiling = range_contraction_pct > 10 and volume_trend_in_base in ("declining", "flat")

    summary_parts = [
        f"Base over the last {len(base)} bars: {base_low:.2f}-{base_high:.2f} "
        f"(height {base_height:.2f}, {base_height_pct:.1f}% of price)."
    ]
    summary_parts.append(
        f"Range has {'contracted' if range_contraction_pct > 0 else 'expanded'} "
        f"{abs(range_contraction_pct):.0f}% vs the prior {len(prior_window)} bars, "
        f"with {volume_trend_in_base} volume inside the base."
    )
    summary_parts.append(
        f"Resistance tested {touches_at_resistance}x, support tested {touches_at_support}x "
        f"({'well-tested level' if well_tested else 'not yet well-tested'})."
    )
    if is_coiling:
        summary_parts.append("This reads as a genuine tight base/coil -- a break of either edge carries weight.")
    else:
        summary_parts.append("This does not yet read as a tight, coiling base -- be skeptical of a break either way.")

    return {
        "base_bars": len(base),
        "base_high": round(base_high, 4),
        "base_low": round(base_low, 4),
        "base_height": round(base_height, 4),
        "base_height_pct": round(base_height_pct, 2),
        "range_contraction_pct": round(range_contraction_pct, 1),
        "volume_trend_in_base": volume_trend_in_base,
        "touches_at_resistance": touches_at_resistance,
        "touches_at_support": touches_at_support,
        "well_tested": well_tested,
        "is_coiling": is_coiling,
        "summary": " ".join(summary_parts),
    }
