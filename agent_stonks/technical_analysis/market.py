"""The broad-market backdrop one ticker is being traded against."""

from __future__ import annotations

import pandas as pd


from .indicators import rsi, sma


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
