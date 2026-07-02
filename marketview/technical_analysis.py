"""
Human-readable technical analysis over OHLCV bars.

Raw bars (Alpaca format: keys o/h/l/c/v/t/vw) are fine for charting but a
flat array of numbers isn't a signal -- an LLM agent has to redo the same
arithmetic every cycle to notice anything. This module does that arithmetic
once and returns the kind of read a human technical analyst would give:
trend regime, momentum, volatility, support/resistance, and volume
confirmation, each as a labeled value plus a one-line summary the agent can
reason over directly.
"""
from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd

_ET = ZoneInfo("America/New_York")


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


def analyze_intraday(bars: list[dict]) -> dict:
    """Short-term price-action read for intraday bars: momentum, position vs VWAP, volatility."""
    if len(bars) < 5:
        return {"note": "not enough bars for intraday analysis"}

    closes = _closes(bars)
    price = float(closes.iloc[-1])
    session_start = float(closes.iloc[0])
    pct_change = (price / session_start - 1) * 100 if session_start else 0.0

    swing = min(10, len(bars) // 2)
    recent_highs = [b["h"] for b in bars[-swing:]]
    recent_lows = [b["l"] for b in bars[-swing:]]
    prior_highs = [b["h"] for b in bars[-2 * swing : -swing]] if len(bars) >= 2 * swing else []
    prior_lows = [b["l"] for b in bars[-2 * swing : -swing]] if len(bars) >= 2 * swing else []

    if prior_highs and prior_lows:
        higher_highs = max(recent_highs) > max(prior_highs)
        higher_lows = min(recent_lows) > min(prior_lows)
        lower_highs = max(recent_highs) < max(prior_highs)
        lower_lows = min(recent_lows) < min(prior_lows)
        if higher_highs and higher_lows:
            momentum = "making higher highs and higher lows (uptrend)"
        elif lower_highs and lower_lows:
            momentum = "making lower highs and lower lows (downtrend)"
        else:
            momentum = "choppy / no consistent higher-high or lower-low pattern"
    else:
        momentum = "not enough bars to classify high/low pattern"

    vwap_note = None
    if "vw" in bars[-1] and bars[-1]["vw"]:
        vwap = float(bars[-1]["vw"])
        vwap_diff_pct = (price - vwap) / vwap * 100 if vwap else 0.0
        vwap_note = f"price is {abs(vwap_diff_pct):.2f}% {'above' if vwap_diff_pct >= 0 else 'below'} session VWAP"

    atr_value = atr(bars, period=min(14, len(bars) - 1))
    volatility_pct = (atr_value / price * 100) if (atr_value is not None and price) else None

    summary_parts = [
        f"Price {pct_change:+.2f}% since the start of this window, {momentum}.",
    ]
    if vwap_note:
        summary_parts.append(vwap_note.capitalize() + ".")
    if volatility_pct is not None:
        summary_parts.append(f"ATR-based volatility ~{volatility_pct:.2f}% of price.")

    return {
        "pct_change_in_window": round(pct_change, 2),
        "momentum_pattern": momentum,
        "vwap_position": vwap_note,
        "atr": atr_value,
        "volatility_pct_of_price": round(volatility_pct, 2) if volatility_pct is not None else None,
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


def analyze_opening_range(bars: list[dict], minutes: int = 15) -> dict:
    """Opening Range Breakout (ORB) read: the high/low set by the first `minutes`
    of today's bars, and whether price has since broken out of that range.

    Assumes 1-minute bars and that the earliest bar for today's date is the
    first bar of the session (true if streaming started at/near the open).
    """
    if not bars:
        return {"note": "no intraday bars available yet"}

    today = str(bars[-1].get("t", ""))[:10]
    day_bars = [b for b in bars if str(b.get("t", ""))[:10] == today]
    if len(day_bars) < 2:
        return {"note": "not enough bars in today's session yet to establish an opening range"}

    opening_bars = day_bars[: max(1, minutes)]
    or_high = max(b["h"] for b in opening_bars)
    or_low = min(b["l"] for b in opening_bars)
    price = float(day_bars[-1]["c"])

    if len(opening_bars) < minutes:
        status = "still forming"
    elif price > or_high:
        status = "broken out above"
    elif price < or_low:
        status = "broken out below"
    else:
        status = "inside_range"

    opening_avg_volume = sum(b.get("v", 0) for b in opening_bars) / len(opening_bars)
    breakout_bars = day_bars[len(opening_bars) :][-3:]
    breakout_avg_volume = (
        sum(b.get("v", 0) for b in breakout_bars) / len(breakout_bars) if breakout_bars else None
    )
    volume_ratio = (
        breakout_avg_volume / opening_avg_volume
        if breakout_avg_volume is not None and opening_avg_volume
        else None
    )

    summary_parts = [
        f"Opening range (first {len(opening_bars)} min): {or_low:.2f}-{or_high:.2f}. "
        f"Price {price:.2f} is {status.replace('_', ' ')}."
    ]
    if volume_ratio is not None:
        summary_parts.append(
            f"Recent volume is {volume_ratio:.1f}x the opening-range average"
            f" ({'confirms' if volume_ratio >= 1.5 else 'does not confirm'} a breakout)."
        )

    return {
        "opening_range_minutes": len(opening_bars),
        "opening_range_high": round(or_high, 4),
        "opening_range_low": round(or_low, 4),
        "current_price": round(price, 4),
        "status": status,
        "volume_ratio_vs_opening_range": round(volume_ratio, 2) if volume_ratio is not None else None,
        "summary": " ".join(summary_parts),
    }


def _vix_label(value: float) -> str:
    if value < 13:
        return "very low (complacent)"
    if value < 17:
        return "low (calm)"
    if value < 20:
        return "normal"
    if value < 26:
        return "elevated"
    if value < 35:
        return "high (fear)"
    return "extreme (panic)"


def _drawdown_from_high_pct(series: pd.Series) -> float:
    """Percent the last value sits below the series' peak (<= 0)."""
    peak = float(series.max())
    last = float(series.iloc[-1])
    if peak <= 0:
        return 0.0
    return (last / peak - 1) * 100


def analyze_market(
    vix_close: "pd.Series | None" = None,
    spy_close: "pd.Series | None" = None,
    vix3m_close: "pd.Series | None" = None,
) -> dict:
    """Broad-market conditions read from the best-known regime gauges.

    The per-ticker analyzers answer "what is this stock doing?"; this answers
    "what is the overall market doing, and how much risk should I take?" using:

    - the VIX fear level and its short-term trend,
    - the VIX term structure (near-term vs 3-month implied vol) -- inversion
      flags acute stress,
    - SPY's primary trend (vs its 50/200-day averages), drawdown from its high,
      and RSI.

    Each marker votes on a risk score (positive = risk-on, negative =
    risk-off). Returns the labeled markers, a `risk_environment` classification,
    a list of actionable `insights`, and a one-line `summary`.
    """
    markers: dict = {}
    insights: list[str] = []
    score = 0

    have_vix = vix_close is not None and len(vix_close) > 0
    have_spy = spy_close is not None and len(spy_close) >= 2
    if not have_vix and not have_spy:
        return {"note": "no market indicator data available"}

    vix = None
    if have_vix:
        vix = float(vix_close.iloc[-1])
        label = _vix_label(vix)
        markers["vix"] = round(vix, 2)
        markers["vix_label"] = label

        if vix < 17:
            score += 2
            insights.append(
                f"VIX {vix:.1f} ({label}): low implied volatility — a calm tape supports "
                "trend-following and normal sizing, though complacency can precede sharp reversals."
            )
        elif vix < 20:
            score += 1
            insights.append(f"VIX {vix:.1f} ({label}): volatility is contained — no broad risk warning.")
        elif vix < 26:
            insights.append(
                f"VIX {vix:.1f} ({label}): volatility is picking up — tighten risk and favor higher-conviction setups."
            )
        elif vix < 35:
            score -= 2
            insights.append(
                f"VIX {vix:.1f} ({label}): elevated tail risk — cut position size, widen stops, don't chase strength."
            )
        else:
            score -= 3
            insights.append(
                f"VIX {vix:.1f} ({label}): crisis-level fear — capital preservation first; "
                "only high-conviction trades with well-defined risk."
            )

        if len(vix_close) >= 6:
            prior = float(vix_close.iloc[-6])
            if prior:
                vix_chg = (vix / prior - 1) * 100
                markers["vix_5d_change_pct"] = round(vix_chg, 1)
                if vix_chg > 10:
                    score -= 1
                    insights.append(
                        f"VIX rising fast (+{vix_chg:.0f}% over 5 sessions): fear is building, momentum favors caution."
                    )
                elif vix_chg < -10:
                    score += 1
                    insights.append(
                        f"VIX falling (-{abs(vix_chg):.0f}% over 5 sessions): fear is subsiding, supportive of risk assets."
                    )

    if have_vix and vix3m_close is not None and len(vix3m_close) > 0:
        vix3m = float(vix3m_close.iloc[-1])
        markers["vix3m"] = round(vix3m, 2)
        if vix3m < vix:
            score -= 2
            markers["vix_term_structure"] = "backwardation"
            insights.append(
                f"VIX term structure inverted (VIX {vix:.1f} > VIX3M {vix3m:.1f}): the market is pricing "
                "acute near-term stress — historically a defensive signal."
            )
        else:
            markers["vix_term_structure"] = "contango"
            insights.append(
                f"VIX term structure normal (contango, VIX3M {vix3m:.1f} ≥ VIX {vix:.1f}): no acute near-term stress priced in."
            )

    if have_spy:
        price = float(spy_close.iloc[-1])
        markers["spy"] = round(price, 2)
        sma50 = sma(spy_close, 50)
        sma200 = sma(spy_close, 200)
        rsi14 = rsi(spy_close, 14)
        drawdown = _drawdown_from_high_pct(spy_close)
        markers["spy_drawdown_from_high_pct"] = round(drawdown, 1)

        if sma200 is not None:
            if price >= sma200:
                score += 1
                insights.append(
                    "S&P 500 (SPY) above its 200-day average: the primary trend is up — broad backdrop supports long exposure."
                )
            else:
                score -= 1
                insights.append(
                    "S&P 500 (SPY) below its 200-day average: the primary trend is down — a headwind for new long positions."
                )
        if sma50 is not None:
            score += 1 if price >= sma50 else -1
        if sma50 is not None and sma200 is not None:
            if sma50 >= sma200:
                markers["spy_ma_cross"] = "golden (50d above 200d)"
            else:
                markers["spy_ma_cross"] = "death (50d below 200d)"
                insights.append(
                    "SPY's 50-day average is below its 200-day (death-cross posture): medium-term momentum is negative."
                )

        if rsi14 is not None:
            markers["spy_rsi_14"] = round(rsi14, 1)
            if rsi14 >= 70:
                insights.append(
                    f"SPY RSI(14) {rsi14:.0f} (overbought): the market is extended — pullback risk is up, be selective adding longs."
                )
            elif rsi14 <= 30:
                insights.append(
                    f"SPY RSI(14) {rsi14:.0f} (oversold): the broad selloff may be stretched — watch for mean-reversion bounces."
                )

        if drawdown <= -20:
            score -= 2
            insights.append(
                f"SPY {drawdown:.0f}% off its high (bear-market territory): a structurally defensive backdrop."
            )
        elif drawdown <= -10:
            score -= 1
            insights.append(
                f"SPY {drawdown:.0f}% off its high (correction territory): broad weakness — favor defense over offense."
            )

    if score >= 3:
        environment = "risk-on"
    elif score <= -3:
        environment = "risk-off"
    else:
        environment = "neutral / mixed"

    summary_bits = [f"Market environment: {environment} (risk score {score:+d})."]
    if "vix" in markers:
        summary_bits.append(f"VIX {markers['vix']:.1f} ({markers['vix_label']}).")
    if "vix_term_structure" in markers:
        summary_bits.append(f"Term structure: {markers['vix_term_structure']}.")
    if "spy_drawdown_from_high_pct" in markers:
        summary_bits.append(f"SPY {markers['spy_drawdown_from_high_pct']:+.1f}% from its recent high.")

    return {
        "risk_environment": environment,
        "risk_score": score,
        **markers,
        "insights": insights,
        "summary": " ".join(summary_bits),
    }


def _gamma_flip_strike(strikes: list[float], net_gamma_by_strike: list[float]) -> "float | None":
    """Approximate zero-gamma level: the strike where cumulative net dealer gamma,
    scanned from the lowest strike up, crosses from negative to positive.

    Below this level dealer hedging tends to amplify moves (negative gamma);
    above it, dampen them (positive gamma). Linearly interpolated between the two
    bracketing strikes.
    """
    cumulative = 0.0
    prev_strike: "float | None" = None
    prev_cumulative: "float | None" = None
    for strike, gamma in zip(strikes, net_gamma_by_strike):
        cumulative += gamma
        if prev_cumulative is not None and prev_cumulative < 0 <= cumulative:
            span = cumulative - prev_cumulative
            if span:
                frac = -prev_cumulative / span
                return prev_strike + frac * (strike - prev_strike)
            return strike
        prev_strike, prev_cumulative = strike, cumulative
    return None


def _wall_trend(history: "list[dict] | None", key: str) -> "str | None":
    """Direction of `key` (call_wall/put_wall) across recorded snapshots, oldest to newest."""
    if not history or len(history) < 2:
        return None
    first = history[0].get(key)
    last = history[-1].get(key)
    if first is None or last is None:
        return None
    if last > first:
        return "rising"
    if last < first:
        return "falling"
    return "flat"


def get_put_call_walls_and_gamma(
    strikes: list[float],
    calls_oi: list[float],
    puts_oi: list[float],
    calls_gamma_exposure: list[float],
    puts_gamma_exposure: list[float],
    spot: float,
    wall_history: "list[dict] | None" = None,
) -> dict:
    """Options-derived support/resistance and dealer-gamma regime.

    The Call Wall (strike with the most call open interest) marks likely
    resistance; the Put Wall (most put open interest) marks likely support.
    Net dealer gamma across strikes says whether hedging flows should dampen
    price action (positive gamma) or amplify it (negative gamma) -- most
    dangerous right around a wall breach. `wall_history` (oldest-first
    `{"call_wall", "put_wall"}` snapshots) lets a rising call wall / falling put
    wall read as a bullish/bearish momentum tell, independent of current price.
    """
    if not strikes:
        return {"note": "no options chain data available"}

    call_wall_idx = max(range(len(strikes)), key=lambda i: calls_oi[i])
    put_wall_idx = max(range(len(strikes)), key=lambda i: puts_oi[i])
    call_wall = float(strikes[call_wall_idx])
    put_wall = float(strikes[put_wall_idx])

    net_gamma_by_strike = [c + p for c, p in zip(calls_gamma_exposure, puts_gamma_exposure)]
    total_net_gamma = sum(net_gamma_by_strike)
    gamma_regime = "positive (dampening)" if total_net_gamma >= 0 else "negative (amplifying)"
    gamma_flip = _gamma_flip_strike(strikes, net_gamma_by_strike)

    in_range = put_wall <= spot <= call_wall
    range_width = call_wall - put_wall
    range_position_pct = (spot - put_wall) / range_width * 100 if range_width > 0 else None

    near_call_wall = call_wall > 0 and abs(spot - call_wall) / call_wall < 0.01
    near_put_wall = put_wall > 0 and abs(spot - put_wall) / put_wall < 0.01

    insights: list[str] = []
    if in_range:
        insights.append(
            f"Spot {spot:.2f} sits inside the {put_wall:.2f}-{call_wall:.2f} put/call wall range"
            + (f" ({range_position_pct:.0f}% of the way from put wall to call wall)." if range_position_pct is not None else ".")
        )
    elif spot > call_wall:
        insights.append(
            f"Spot {spot:.2f} is already above the call wall ({call_wall:.2f}) -- resistance has "
            "been breached; a held break above often runs further as call gamma overhead thins out."
        )
    else:
        insights.append(
            f"Spot {spot:.2f} is already below the put wall ({put_wall:.2f}) -- support has given way; "
            "a held break below often accelerates as put gamma underneath thins out."
        )

    if near_call_wall:
        insights.append(f"Price is within 1% of the call wall ({call_wall:.2f}) -- watch for a stall or reversal here.")
    if near_put_wall:
        insights.append(f"Price is within 1% of the put wall ({put_wall:.2f}) -- watch for a bounce here.")

    if total_net_gamma >= 0:
        insights.append(
            "Net dealer gamma is positive: hedging flows tend to dampen moves (sell rallies, buy dips), "
            "favoring range-bound, mean-reverting price action between the walls."
        )
    else:
        insights.append(
            "Net dealer gamma is negative: hedging flows tend to amplify moves in the prevailing direction "
            "(buy strength, sell weakness), raising breakout/breakdown risk if a wall gives way."
        )

    call_wall_trend = _wall_trend(wall_history, "call_wall")
    put_wall_trend = _wall_trend(wall_history, "put_wall")
    if call_wall_trend == "rising":
        insights.append("Call wall has been rising -- bullish sentiment, resistance migrating higher.")
    elif call_wall_trend == "falling":
        insights.append("Call wall has been falling -- resistance compressing lower.")
    if put_wall_trend == "falling":
        insights.append("Put wall has been falling -- bearish sentiment, support migrating lower.")
    elif put_wall_trend == "rising":
        insights.append("Put wall has been rising -- support migrating higher, a bullish tell.")

    summary_parts = [
        f"Call wall {call_wall:.2f} (resistance), put wall {put_wall:.2f} (support), net gamma {gamma_regime}.",
    ]
    if gamma_flip is not None:
        summary_parts.append(f"Gamma flip near {gamma_flip:.2f}.")
    summary_parts.append(insights[0])

    return {
        "call_wall": call_wall,
        "put_wall": put_wall,
        "spot": spot,
        "gamma_flip": gamma_flip,
        "net_gamma": total_net_gamma,
        "gamma_regime": gamma_regime,
        "in_range": in_range,
        "range_position_pct": round(range_position_pct, 1) if range_position_pct is not None else None,
        "call_wall_trend": call_wall_trend,
        "put_wall_trend": put_wall_trend,
        "insights": insights,
        "summary": " ".join(summary_parts),
    }


def analyze_volume(bars: list[dict]) -> dict:
    """Volume confirmation read: is participation backing the recent price move?"""
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

    summary_parts = [
        f"Volume is {volume_trend} (last 10 bars avg {recent_avg:,.0f} vs prior 10 avg {prior_avg:,.0f}"
        + (f", {relative_volume:.1f}x" if relative_volume is not None else "")
        + ")."
    ]
    if flow_trend:
        summary_parts.append(f"On-balance volume is {flow_trend}.")
    summary_parts.append(f"Volume is {confirmation} relative to the {price_pct_change:+.2f}% price move.")

    return {
        "bar_count": len(volumes),
        "recent_10bar_avg_volume": recent_avg,
        "prior_10bar_avg_volume": prior_avg,
        "relative_volume": round(relative_volume, 2) if relative_volume is not None else None,
        "volume_trend": volume_trend,
        "obv_trend": flow_trend,
        "price_pct_change_10bar": round(price_pct_change, 2),
        "confirmation": confirmation,
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


def session_time_window(latest_bar_ts: "str | None" = None) -> dict:
    """Classify the current point in the trading day for breakout timing discipline.

    Breakouts in the first 90 minutes or the final hour of the regular session
    are historically the most reliable; the 12:00-14:00 ET stretch is a
    notorious fakeout zone. Uses the timestamp of the latest bar (Alpaca bars
    are UTC ISO strings) when given, otherwise the current time.
    """
    if latest_bar_ts:
        try:
            dt = datetime.fromisoformat(latest_bar_ts.replace("Z", "+00:00"))
        except ValueError:
            dt = datetime.now(timezone.utc)
    else:
        dt = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    et = dt.astimezone(_ET)
    minutes = et.hour * 60 + et.minute

    open_ = 9 * 60 + 30
    morning_end = 11 * 60
    dead_start = 12 * 60
    dead_end = 14 * 60
    power_hour_start = 15 * 60
    close_ = 16 * 60

    if minutes < open_ or minutes >= close_:
        window, favorable = "outside_regular_hours", False
        note = "Outside the 9:30-16:00 ET regular session."
    elif minutes < morning_end:
        window, favorable = "opening_window", True
        note = "First 90 minutes of the session -- historically the most reliable window for breakouts."
    elif dead_start <= minutes < dead_end:
        window, favorable = "midday_dead_zone", False
        note = "12:00-14:00 ET dead zone -- breakouts here are notoriously prone to fakeouts; demand stronger confirmation or stand aside."
    elif minutes >= power_hour_start:
        window, favorable = "power_hour", True
        note = "Final hour of the session -- a favorable window for breakouts."
    else:
        window, favorable = "other_session_hours", True
        note = "Mid-morning/early-afternoon -- acceptable but not the highest-conviction window."

    return {
        "et_time": et.strftime("%H:%M"),
        "window": window,
        "favorable_for_breakouts": favorable,
        "summary": f"{et.strftime('%H:%M')} ET -- {window.replace('_', ' ')}. {note}",
    }


def breakout_trade_geometry(
    entry: float, stop: float, base_height: "float | None" = None, atr: "float | None" = None
) -> dict:
    """Mechanical entry/stop/target math for a long breakout trade.

    Projects targets by adding 1x and 2x the base height (the classic
    "measured move") and/or 1x and 2x ATR above entry, then expresses each as
    a reward-to-risk multiple of the entry-to-stop distance. `meets_min_reward_risk`
    flags whether the best available target clears the 2:1 minimum breakout
    traders require before taking the trade.
    """
    if entry <= 0 or stop <= 0:
        return {"note": "entry and stop must be positive prices"}
    risk_per_share = entry - stop
    if risk_per_share <= 0:
        return {"note": "stop must be below entry for a long breakout setup"}

    targets: dict[str, float] = {}
    if base_height is not None and base_height > 0:
        targets["target1_base_height"] = round(entry + base_height, 4)
        targets["target2_base_height"] = round(entry + 2 * base_height, 4)
    if atr is not None and atr > 0:
        targets["target1_atr"] = round(entry + atr, 4)
        targets["target2_atr"] = round(entry + 2 * atr, 4)

    reward_risk: dict[str, float] = {}
    for key, target in targets.items():
        reward = target - entry
        reward_risk[key.replace("target", "rr")] = round(reward / risk_per_share, 2)

    best_rr = max(reward_risk.values()) if reward_risk else None
    meets_min_rr = best_rr is not None and best_rr >= 2.0

    summary_parts = [f"Risk per share {risk_per_share:.2f} (entry {entry:.2f}, stop {stop:.2f})."]
    if targets:
        labelled = ", ".join(
            f"{k}={v:.2f} (R:R {reward_risk[k.replace('target', 'rr')]:.1f})" for k, v in targets.items()
        )
        summary_parts.append(labelled + ".")
        summary_parts.append(
            "Meets the 2:1 minimum reward-to-risk."
            if meets_min_rr
            else "Does NOT meet the 2:1 minimum reward-to-risk -- skip, or wait for a tighter stop/better entry."
        )
    else:
        summary_parts.append("No base_height or atr given -- cannot project a target.")

    return {
        "risk_per_share": round(risk_per_share, 4),
        **targets,
        **reward_risk,
        "best_reward_risk_ratio": best_rr,
        "meets_min_reward_risk": meets_min_rr,
        "summary": " ".join(summary_parts),
    }


def vwap_reversion_geometry(
    entry: float,
    vwap: float,
    std_dev: float,
    side: str = "long",
    min_reward_risk: float = 1.5,
) -> dict:
    """Mechanical entry/stop/target math for a VWAP mean-reversion trade.

    The target is always VWAP (the mean price is expected to revert to). The
    stop sits one standard deviation beyond the entry -- i.e. past the next
    band -- so a long entered near the -2σ band stops out below -3σ, a short
    entered near +2σ stops out above +3σ. Returns the reward-to-risk ratio and
    whether it clears the mean-reversion minimum (1.5:1 by default; these are
    tighter-R:R, higher-win-rate trades than breakouts).
    """
    if entry <= 0 or vwap <= 0 or std_dev <= 0:
        return {"note": "entry, vwap, and std_dev must be positive"}
    side = side.lower()
    if side not in ("long", "short"):
        return {"note": "side must be 'long' or 'short'"}

    if side == "long":
        if entry >= vwap:
            return {"note": "for a long reversion, entry must be below VWAP (price stretched down to the band)"}
        stop = entry - std_dev
        target = vwap
        reward = target - entry
        risk = entry - stop
    else:  # short
        if entry <= vwap:
            return {"note": "for a short reversion, entry must be above VWAP (price stretched up to the band)"}
        stop = entry + std_dev
        target = vwap
        reward = entry - target
        risk = stop - entry

    reward_risk = reward / risk if risk > 0 else None
    meets_min_rr = reward_risk is not None and reward_risk >= min_reward_risk

    summary = (
        f"{side.capitalize()} reversion: entry {entry:.2f}, stop {stop:.2f} "
        f"(1σ={std_dev:.3f} beyond entry), target VWAP {target:.2f}. "
        f"Reward/risk {reward_risk:.2f}:1. "
        + (
            f"Meets the {min_reward_risk:.1f}:1 minimum."
            if meets_min_rr
            else f"Below the {min_reward_risk:.1f}:1 minimum -- skip or wait for a deeper stretch / closer entry."
        )
    )

    return {
        "side": side,
        "entry": round(entry, 4),
        "stop": round(stop, 4),
        "target": round(target, 4),
        "risk_per_share": round(risk, 4),
        "reward_per_share": round(reward, 4),
        "reward_risk_ratio": round(reward_risk, 2) if reward_risk is not None else None,
        "meets_min_reward_risk": meets_min_rr,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Smart Money Concepts (SMC): order blocks, fair value gaps, and the composite
# institutional setup that combines a higher-timeframe demand zone with an
# intraday confirmation signal (rejection candle, FVG fill, or breaker/BOS).
# ---------------------------------------------------------------------------


def _order_block(bars: list[dict], i: int, kind: str, highs: list[float], lows: list[float]) -> dict:
    """One order-block zone descriptor anchored on the origin candle at index `i`.

    The zone is the full high-low range of that candle -- the price band
    institutions are presumed to defend on a return.
    """
    top = highs[i]
    bottom = lows[i]
    return {
        "type": kind,
        "index": i,
        "top": round(top, 4),
        "bottom": round(bottom, 4),
        "mid": round((top + bottom) / 2.0, 4),
        "timestamp": bars[i].get("t"),
    }


def find_order_blocks(bars: list[dict], swing: int = 5, lookahead: int = 3, max_blocks: int = 6) -> dict:
    """Locate institutional order blocks: the last opposing candle before a
    displacement move that breaks structure.

    A *bullish* order block is the last down-close candle before an up-move that
    takes out the prior `swing`-bar high (a bullish break of structure) -- the
    footprint of institutions absorbing supply before driving price up, and a
    zone they tend to defend on a return. A *bearish* order block is the mirror:
    the last up-close candle before a down-move that breaks the prior swing low.
    Each block's zone is the full high-low range of its origin candle. A block is
    "mitigated" once price has traded back into its zone after forming (its first
    defence has already been tested), which makes a *fresh, unmitigated* block the
    higher-quality one to trade a return into.
    """
    n = len(bars)
    if n < swing + lookahead + 1:
        return {"note": "not enough bars to locate order blocks", "order_blocks": []}

    o = [float(b["o"]) for b in bars]
    h = [float(b["h"]) for b in bars]
    l = [float(b["l"]) for b in bars]
    c = [float(b["c"]) for b in bars]

    blocks: list[dict] = []
    for i in range(swing, n - lookahead):
        swing_high = max(h[i - swing : i])
        swing_low = min(l[i - swing : i])
        impulse_high = max(h[i + 1 : i + 1 + lookahead])
        impulse_low = min(l[i + 1 : i + 1 + lookahead])
        if c[i] < o[i] and impulse_high > swing_high:
            blocks.append(_order_block(bars, i, "bullish", h, l))
        elif c[i] > o[i] and impulse_low < swing_low:
            blocks.append(_order_block(bars, i, "bearish", h, l))

    for blk in blocks:
        idx = blk["index"]
        # Mitigated if any bar *after* the displacement window traded back into the zone.
        blk["mitigated"] = any(
            l[j] <= blk["top"] and h[j] >= blk["bottom"] for j in range(idx + lookahead + 1, n)
        )
        blk["bars_ago"] = n - 1 - idx

    return {"order_blocks": blocks[-max_blocks:], "bar_count": n}


def _nearest_bullish_demand(blocks: list[dict], spot: float) -> "dict | None":
    """The bullish order block closest below (or containing) `spot` -- the demand
    zone price would return *down* into. Highest such zone wins (nearest support)."""
    candidates = [b for b in blocks if b["type"] == "bullish" and b["bottom"] <= spot]
    if not candidates:
        return None
    return max(candidates, key=lambda b: b["top"])


def _nearest_bearish_supply(blocks: list[dict], spot: float) -> "dict | None":
    """The bearish order block closest above `spot` -- the supply zone that makes
    a natural upside target. Lowest such zone wins (nearest overhead resistance)."""
    candidates = [b for b in blocks if b["type"] == "bearish" and b["top"] >= spot]
    if not candidates:
        return None
    return min(candidates, key=lambda b: b["bottom"])


def analyze_order_blocks(bars: list[dict], spot: "float | None" = None) -> dict:
    """Order-block read: every detected block plus the nearest bullish demand zone
    at/below price and the nearest bearish supply zone above it.

    The demand zone is the candidate entry on a return; the supply zone is a
    natural structural target. Returns labeled values plus a one-line summary.
    """
    found = find_order_blocks(bars)
    if "note" in found:
        return found
    blocks = found["order_blocks"]
    if spot is None:
        spot = float(bars[-1]["c"])

    demand = _nearest_bullish_demand(blocks, spot)
    supply = _nearest_bearish_supply(blocks, spot)

    summary_parts = [f"Found {len(blocks)} order block(s) over {found['bar_count']} bars; spot {spot:.2f}."]
    if demand is not None:
        state = "unmitigated" if not demand["mitigated"] else "mitigated"
        summary_parts.append(
            f"Nearest bullish demand block {demand['bottom']:.2f}-{demand['top']:.2f} "
            f"({state}, {demand['bars_ago']} bars ago)."
        )
    else:
        summary_parts.append("No bullish demand block at/below price.")
    if supply is not None:
        summary_parts.append(f"Nearest bearish supply block {supply['bottom']:.2f}-{supply['top']:.2f} (target).")

    return {
        "spot": round(spot, 4),
        "order_blocks": blocks,
        "nearest_bullish_ob": demand,
        "nearest_bearish_ob": supply,
        "summary": " ".join(summary_parts),
    }


def find_fair_value_gaps(bars: list[dict], max_gaps: int = 6) -> dict:
    """Locate fair value gaps (FVGs): three-candle price imbalances institutions
    tend to revisit.

    A *bullish* FVG forms when a strong up-candle leaves a gap between the high of
    the candle before it and the low of the candle after it (`low[i+1] > high[i-1]`);
    the gap zone `(high[i-1], low[i+1])` is an unfilled imbalance that often acts as
    support on a pullback. A *bearish* FVG is the mirror (`high[i+1] < low[i-1]`).
    A gap is "filled" once a later bar trades back through the zone.
    """
    n = len(bars)
    if n < 3:
        return {"note": "not enough bars to locate fair value gaps", "fair_value_gaps": []}

    h = [float(b["h"]) for b in bars]
    l = [float(b["l"]) for b in bars]

    gaps: list[dict] = []
    for i in range(1, n - 1):
        if l[i + 1] > h[i - 1]:
            gaps.append({"type": "bullish", "index": i, "bottom": round(h[i - 1], 4), "top": round(l[i + 1], 4)})
        elif h[i + 1] < l[i - 1]:
            gaps.append({"type": "bearish", "index": i, "bottom": round(h[i + 1], 4), "top": round(l[i - 1], 4)})

    for g in gaps:
        idx = g["index"]
        g["filled"] = any(l[j] <= g["top"] and h[j] >= g["bottom"] for j in range(idx + 2, n))
        g["bars_ago"] = n - 1 - idx

    return {"fair_value_gaps": gaps[-max_gaps:], "bar_count": n}


def analyze_fair_value_gaps(bars: list[dict], spot: "float | None" = None) -> dict:
    """Fair-value-gap read: detected gaps plus the nearest bullish FVG at/below
    price (a support imbalance price may be filling now). Returns a one-line summary."""
    found = find_fair_value_gaps(bars)
    if "note" in found:
        return found
    gaps = found["fair_value_gaps"]
    if spot is None:
        spot = float(bars[-1]["c"])

    bullish_below = [g for g in gaps if g["type"] == "bullish" and g["bottom"] <= spot * 1.001]
    nearest = max(bullish_below, key=lambda g: g["top"]) if bullish_below else None

    summary_parts = [f"Found {len(gaps)} fair value gap(s); spot {spot:.2f}."]
    if nearest is not None:
        state = "filled" if nearest["filled"] else "unfilled"
        summary_parts.append(f"Nearest bullish FVG {nearest['bottom']:.2f}-{nearest['top']:.2f} ({state}).")
    else:
        summary_parts.append("No bullish FVG at/below price.")

    return {
        "spot": round(spot, 4),
        "fair_value_gaps": gaps,
        "nearest_bullish_fvg": nearest,
        "summary": " ".join(summary_parts),
    }


def _intraday_break_of_structure(bars: list[dict], swing: int = 5) -> bool:
    """Whether intraday price has made a bullish break of structure and is holding it:
    the recent `swing` bars took out the prior `swing`-bar high and price still sits
    above that broken level. This is the lightweight 'breaker / structure flip'
    confirmation -- old resistance reclaimed as support."""
    if len(bars) < 2 * swing:
        return False
    h = [float(b["h"]) for b in bars]
    c = [float(b["c"]) for b in bars]
    prior_high = max(h[-2 * swing : -swing])
    recent_high = max(h[-swing:])
    return recent_high > prior_high and c[-1] > prior_high


def smart_money_trade_geometry(
    entry: float, stop: float, target: float, min_reward_risk: float = 3.0
) -> dict:
    """Mechanical entry/stop/target math for a long Smart Money setup.

    Entry is at the higher-timeframe demand (order block) on a return, the stop
    sits just beyond the block, and the target is the next opposing structural
    level. `meets_min_reward_risk` flags whether the reward-to-risk clears the
    3:1 minimum the Smart Money setup demands (it typically runs 3:1 to 5:1).
    """
    if entry <= 0 or stop <= 0 or target <= 0:
        return {"note": "entry, stop, and target must be positive prices"}
    if not (stop < entry < target):
        return {"note": "a long smart-money setup needs stop < entry < target"}

    risk = entry - stop
    reward = target - entry
    reward_risk = reward / risk if risk > 0 else None
    meets_min_rr = reward_risk is not None and reward_risk >= min_reward_risk

    summary = (
        f"Long smart-money setup: entry {entry:.2f}, stop {stop:.2f} (just beyond the block), "
        f"target {target:.2f}. Reward/risk {reward_risk:.2f}:1. "
        + (
            f"Meets the {min_reward_risk:.0f}:1 minimum."
            if meets_min_rr
            else f"Below the {min_reward_risk:.0f}:1 minimum -- skip, or wait for a deeper return into the block."
        )
    )

    return {
        "entry": round(entry, 4),
        "stop": round(stop, 4),
        "target": round(target, 4),
        "risk_per_share": round(risk, 4),
        "reward_per_share": round(reward, 4),
        "reward_risk_ratio": round(reward_risk, 2) if reward_risk is not None else None,
        "meets_min_reward_risk": meets_min_rr,
        "summary": summary,
    }


def analyze_premium_discount(
    bars: list[dict], lookback: int = 20, spot: "float | None" = None
) -> dict:
    """Premium / discount read over the recent dealing range.

    Smart Money buys in *discount* (below the range midpoint, "equilibrium") and
    sells in *premium* (above it). The dealing range is the highest high and
    lowest low over the last `lookback` bars; its midpoint is equilibrium. A
    return into a bullish demand block that *also* sits in discount is a
    higher-quality long than the same block sitting in premium -- price is cheap
    relative to where institutions accumulated. The deep-discount "OTE" zone is
    the 0.618-0.79 retracement down from the range high, where institutional
    longs are statistically filled.
    """
    if not bars or len(bars) < 3:
        return {"note": "not enough bars for a premium/discount read"}

    window = bars[-lookback:] if len(bars) >= lookback else bars
    range_high = max(float(b["h"]) for b in window)
    range_low = min(float(b["l"]) for b in window)
    rng = range_high - range_low
    if rng <= 0:
        return {"note": "flat dealing range; premium/discount undefined"}

    if spot is None:
        spot = float(bars[-1]["c"])
    equilibrium = (range_high + range_low) / 2.0
    # 0.0 = range low, 1.0 = range high.
    position = (spot - range_low) / rng

    if position < 0.45:
        zone = "discount"
    elif position > 0.55:
        zone = "premium"
    else:
        zone = "equilibrium"

    # Optimal Trade Entry: the 0.618-0.79 retracement down from the high.
    ote_top = round(range_high - 0.618 * rng, 4)
    ote_bottom = round(range_high - 0.79 * rng, 4)
    in_ote = ote_bottom <= spot <= ote_top

    summary = (
        f"Dealing range {range_low:.2f}-{range_high:.2f}, equilibrium {equilibrium:.2f}; "
        f"spot {spot:.2f} is in the {zone} zone ({position * 100:.0f}% of range). "
        + (
            "Inside the deep-discount OTE zone -- prime institutional long area."
            if in_ote
            else ("Below equilibrium -- favourable for longs." if zone == "discount"
                  else "At/above equilibrium -- longs are buying retail-expensive prices.")
        )
    )

    return {
        "spot": round(spot, 4),
        "range_high": round(range_high, 4),
        "range_low": round(range_low, 4),
        "equilibrium": round(equilibrium, 4),
        "range_position": round(position, 3),
        "zone": zone,
        "in_discount": zone == "discount",
        "ote_zone": {"bottom": ote_bottom, "top": ote_top},
        "in_ote_zone": in_ote,
        "summary": summary,
    }


def _cluster_levels(levels: list[float], tol_pct: float) -> list[dict]:
    """Group near-equal price levels into liquidity pools.

    Levels within `tol_pct` of a running cluster anchor are merged; a pool with
    two or more members is an *equal-highs/lows* cluster -- a stronger resting
    pool of liquidity (more stops bunched at one price)."""
    pools: list[dict] = []
    for price in sorted(levels):
        if pools and price <= pools[-1]["anchor"] * (1 + tol_pct):
            pool = pools[-1]
            pool["members"].append(price)
            pool["price"] = sum(pool["members"]) / len(pool["members"])
        else:
            pools.append({"anchor": price, "price": price, "members": [price]})
    return [
        {"price": round(p["price"], 4), "count": len(p["members"]), "equal": len(p["members"]) >= 2}
        for p in pools
    ]


def analyze_liquidity(
    bars: list[dict], swing: int = 3, tol_pct: float = 0.0015, recent: int = 5, spot: "float | None" = None
) -> dict:
    """Liquidity pools and recent sweeps -- the core Smart Money 'stop hunt' read.

    Liquidity rests where retail stops cluster: just above swing highs (buy-side
    liquidity, BSL) and just below swing lows (sell-side liquidity, SSL).
    Institutions push price through these pools to fill large orders, then
    reverse -- a *liquidity sweep* / stop run. A bullish SSL sweep (price
    undercuts a prior swing low then closes back above it) is exactly the trap
    that precedes an institutional markup, and one of the strongest
    confirmations for a long off a demand block. Near-equal highs/lows within
    `tol_pct` are merged into a single, stronger resting-liquidity pool.
    """
    n = len(bars)
    if n < 2 * swing + 2:
        return {"note": "not enough bars to locate liquidity", "buy_side_liquidity": [], "sell_side_liquidity": []}

    h = [float(b["h"]) for b in bars]
    l = [float(b["l"]) for b in bars]
    c = [float(b["c"]) for b in bars]

    swing_highs: list[tuple[int, float]] = []
    swing_lows: list[tuple[int, float]] = []
    for i in range(swing, n - swing):
        if h[i] == max(h[i - swing : i + swing + 1]):
            swing_highs.append((i, h[i]))
        if l[i] == min(l[i - swing : i + swing + 1]):
            swing_lows.append((i, l[i]))

    if spot is None:
        spot = float(bars[-1]["c"])

    bsl = _cluster_levels([p for _, p in swing_highs], tol_pct)  # buy-side (above)
    ssl = _cluster_levels([p for _, p in swing_lows], tol_pct)  # sell-side (below)

    bsl_above = [pool for pool in bsl if pool["price"] >= spot]
    ssl_below = [pool for pool in ssl if pool["price"] <= spot]
    nearest_bsl = min(bsl_above, key=lambda p: p["price"]) if bsl_above else None
    nearest_ssl = max(ssl_below, key=lambda p: p["price"]) if ssl_below else None

    # Recent sweep: a bar in the last `recent` that pierced a *prior* swing level
    # and closed back on the other side of it (the reversal that defines a sweep).
    recent_sweep: "dict | None" = None
    for j in range(max(swing, n - recent), n):
        prior_lows = [pl for k, pl in swing_lows if k <= j - 1]
        for pl in prior_lows:
            if l[j] < pl and c[j] > pl:
                recent_sweep = {"type": "bullish", "level": round(pl, 4), "bars_ago": n - 1 - j}
                break
        if recent_sweep is not None:
            continue
        prior_highs = [ph for k, ph in swing_highs if k <= j - 1]
        for ph in prior_highs:
            if h[j] > ph and c[j] < ph:
                recent_sweep = {"type": "bearish", "level": round(ph, 4), "bars_ago": n - 1 - j}
                break

    bullish_sweep = recent_sweep is not None and recent_sweep["type"] == "bullish"

    parts = [f"{len(bsl)} buy-side and {len(ssl)} sell-side liquidity pool(s); spot {spot:.2f}."]
    if nearest_bsl is not None:
        eq = " (equal highs)" if nearest_bsl["equal"] else ""
        parts.append(f"Nearest overhead BSL {nearest_bsl['price']:.2f}{eq} -- liquidity/target above.")
    if nearest_ssl is not None:
        eq = " (equal lows)" if nearest_ssl["equal"] else ""
        parts.append(f"Nearest SSL {nearest_ssl['price']:.2f}{eq} below -- stops resting there.")
    if recent_sweep is not None:
        parts.append(
            f"Recent {recent_sweep['type']} sweep of {recent_sweep['level']:.2f} "
            f"({recent_sweep['bars_ago']} bars ago)"
            + (" -- bullish stop-run, supports a long." if bullish_sweep else ".")
        )
    else:
        parts.append("No recent sweep.")

    return {
        "spot": round(spot, 4),
        "buy_side_liquidity": bsl,
        "sell_side_liquidity": ssl,
        "nearest_bsl_above": nearest_bsl,
        "nearest_ssl_below": nearest_ssl,
        "recent_sweep": recent_sweep,
        "bullish_sweep": bullish_sweep,
        "summary": " ".join(parts),
    }


def analyze_smart_money_setup(
    daily_bars: list[dict],
    intraday_bars: "list[dict] | None" = None,
    spot: "float | None" = None,
    min_reward_risk: float = 3.0,
    stop_buffer: float = 0.1,
) -> dict:
    """The composite Smart Money setup: a higher-timeframe bullish order block that
    price is returning into, confirmed by an intraday signal.

    Combines the higher-timeframe structure (daily order blocks + trend regime)
    with intraday confirmation (a bullish rejection candle, a bullish FVG that price
    has filled, or an intraday break-of-structure/breaker). A `long_setup` requires
    all of: a bullish demand block at/below price, a non-bearish daily regime, price
    actually inside that block, at least one intraday confirmation, and a target
    (the next opposing structural level) that clears the 3:1 reward-to-risk minimum.
    Anything short of that with a valid block reads as `watching`; no qualifying
    block at all is `no_setup`.
    """
    if not daily_bars or len(daily_bars) < 8:
        return {"note": "not enough daily bars for a smart-money structural read", "signal": "no_setup"}

    if spot is None:
        ref = intraday_bars if intraday_bars else daily_bars
        spot = float(ref[-1]["c"])

    ob = analyze_order_blocks(daily_bars, spot=spot)
    demand = ob.get("nearest_bullish_ob")
    supply = ob.get("nearest_bearish_ob")
    regime = analyze_trend(daily_bars).get("regime", "neutral")

    price_in_ob = demand is not None and demand["bottom"] <= spot <= demand["top"]

    # Premium/discount context: smart money buys discount (below equilibrium).
    pd_read = analyze_premium_discount(daily_bars, spot=spot)
    in_discount = bool(pd_read.get("in_discount"))

    # Intraday confirmation signals (any one qualifies; more = higher quality).
    confirmations: list[str] = []
    nearest_fvg = None
    liquidity = None
    if intraday_bars and len(intraday_bars) >= 3:
        if _rejection_candle(intraday_bars[-1]) == "bullish_rejection":
            confirmations.append("rejection_candle")
        fvg = analyze_fair_value_gaps(intraday_bars, spot=spot)
        nearest_fvg = fvg.get("nearest_bullish_fvg")
        if nearest_fvg is not None and nearest_fvg.get("filled"):
            confirmations.append("fvg_fill")
        if _intraday_break_of_structure(intraday_bars):
            confirmations.append("breaker")
        liquidity = analyze_liquidity(intraday_bars, spot=spot)
        if liquidity.get("bullish_sweep"):
            confirmations.append("liquidity_sweep")

    # Entry, stop, and target geometry.
    entry = round(spot if price_in_ob else (demand["top"] if demand else spot), 4)
    suggested_stop = None
    structural_target = None
    geometry = None
    if demand is not None:
        zone_height = demand["top"] - demand["bottom"]
        suggested_stop = round(demand["bottom"] - stop_buffer * max(zone_height, entry * 0.001), 4)
        if supply is not None and supply["bottom"] > entry:
            structural_target = supply["bottom"]
        else:
            recent_high = max(b["h"] for b in daily_bars[-20:])
            structural_target = round(recent_high, 4) if recent_high > entry else None
        if structural_target is not None:
            geometry = smart_money_trade_geometry(entry, suggested_stop, structural_target, min_reward_risk)

    meets_rr = bool(geometry and geometry.get("meets_min_reward_risk"))

    if demand is None or regime == "bearish":
        signal = "no_setup"
    elif price_in_ob and confirmations and meets_rr:
        signal = "long_setup"
    else:
        signal = "watching"

    if signal == "long_setup" and len(confirmations) >= 2 and regime == "bullish" and in_discount and (
        geometry and geometry["reward_risk_ratio"] >= 4.0
    ):
        quality = "A+"
    elif signal == "long_setup":
        quality = "B"
    else:
        quality = "C"

    summary_parts = [f"HTF regime {regime}; spot {spot:.2f}."]
    if demand is not None:
        loc = "inside" if price_in_ob else ("above" if spot > demand["top"] else "below")
        summary_parts.append(
            f"Bullish demand block {demand['bottom']:.2f}-{demand['top']:.2f} "
            f"({'unmitigated' if not demand['mitigated'] else 'mitigated'}); price is {loc} it."
        )
    else:
        summary_parts.append("No bullish demand block at/below price.")
    summary_parts.append(f"Price in {pd_read.get('zone', 'n/a')} zone (eq {pd_read.get('equilibrium', float('nan')):.2f}).")
    summary_parts.append(
        f"Intraday confirmation: {', '.join(confirmations) if confirmations else 'none'}."
    )
    if geometry is not None and "reward_risk_ratio" in geometry:
        summary_parts.append(
            f"Geometry entry {entry:.2f} / stop {suggested_stop:.2f} / target {structural_target:.2f} "
            f"= {geometry['reward_risk_ratio']:.1f}:1."
        )
    summary_parts.append(
        {
            "long_setup": f"LONG setup ({quality}): return into demand with confirmation and ≥{min_reward_risk:.0f}:1 target.",
            "watching": "Watching: a valid demand block exists but price/confirmation/RR isn't all there yet.",
            "no_setup": "No setup: no bullish demand block at/below price, or the daily regime is bearish.",
        }[signal]
    )

    return {
        "signal": signal,
        "quality": quality,
        "htf_regime": regime,
        "spot": round(spot, 4),
        "order_block": demand,
        "supply_block": supply,
        "price_in_order_block": price_in_ob,
        "premium_discount_zone": pd_read.get("zone"),
        "in_discount": in_discount,
        "equilibrium": pd_read.get("equilibrium"),
        "intraday_confirmation": confirmations,
        "confirmed": bool(confirmations),
        "recent_sweep": liquidity.get("recent_sweep") if liquidity else None,
        "nearest_bullish_fvg": nearest_fvg,
        "suggested_entry": entry if demand is not None else None,
        "suggested_stop": suggested_stop,
        "structural_target": structural_target,
        "reward_risk_to_target": geometry["reward_risk_ratio"] if geometry and "reward_risk_ratio" in geometry else None,
        "meets_min_reward_risk": meets_rr,
        "summary": " ".join(summary_parts),
    }
