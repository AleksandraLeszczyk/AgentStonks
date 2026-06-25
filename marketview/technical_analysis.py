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
