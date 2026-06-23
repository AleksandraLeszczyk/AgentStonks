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

import pandas as pd


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
