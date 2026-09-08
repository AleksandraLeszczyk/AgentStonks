"""Has the TRAJECTORY changed?

A support/resistance map says where price was defended in the past; it says
nothing about the tape turning while a plan sits armed at those levels. This
is the read that answers that question instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import clock, market_hours

_ET = market_hours.MARKET_TZ

from ._shared import _news_datetimes, _news_near

from .indicators import _closes, _session_bars, piecewise_regimes


# --------------------------------------------------------------------------
# Regime-shift detection: has the TRAJECTORY changed?
#
# A support/resistance map says where price was defended in the past; it says
# nothing about the tape turning while a plan sits armed at those levels. That
# is the failure mode this read exists for -- a demand line bid into a tape that
# has already rolled over is a falling-knife bid, and a level-only agent finds
# out about it at its stop. The constants below define what "the trajectory
# changed" means quantitatively.
# --------------------------------------------------------------------------

# Bars in the "right now" velocity window compared against the window before it.
_REGIME_FAST_WINDOW = 5

# Fast-window move this fraction (or less) of the previous window's counts as
# the move losing speed rather than merely pausing.
_REGIME_STALL_RATIO = 0.5

# Volume behind the turn (vs the session's median minute) at or above this makes
# the turn participated -- a real handover, not a drift.
_REGIME_TURN_VOLUME_MULT = 1.5

# News this recent (minutes, relative to the latest bar) is a live catalyst.
_REGIME_NEWS_WINDOW_MIN = 30

def _swing_extremes(bars: list[dict], swing: int) -> "tuple[dict | None, dict | None]":
    """The most recent CONFIRMED swing high and swing low -- a bar whose high
    (low) is the extreme of the `swing` bars on each side, so it has been
    validated by the bars that followed it. Each is {"price", "bars_ago"};
    either is None when no such pivot exists yet."""
    n = len(bars)
    highs = [float(b["h"]) for b in bars]
    lows = [float(b["l"]) for b in bars]
    high = low = None
    for i in range(n - swing - 1, swing - 1, -1):
        if high is None and highs[i] == max(highs[i - swing : i + swing + 1]):
            high = {"price": round(highs[i], 4), "bars_ago": n - 1 - i}
        if low is None and lows[i] == min(lows[i - swing : i + swing + 1]):
            low = {"price": round(lows[i], 4), "bars_ago": n - 1 - i}
        if high and low:
            break
    return high, low

def _last_vwap_cross(closes: list[float], vwaps: list[float]) -> "dict | None":
    """The most recent session-VWAP cross: which side price is on now, how many
    bars ago it flipped, and at what VWAP. None while price has stayed on one
    side for the whole session (nothing crossed)."""
    side = [1 if c >= v else -1 for c, v in zip(closes, vwaps)]
    now = side[-1]
    for i in range(len(side) - 2, -1, -1):
        if side[i] != now:
            return {
                "direction": "reclaimed_vwap" if now > 0 else "lost_vwap",
                "bars_ago": len(side) - 1 - (i + 1),
                "vwap_at_cross": round(vwaps[i + 1], 4),
            }
    return None

def detect_regime_shift(
    bars: list[dict],
    news_times: "list[str] | None" = None,
    spot: "float | None" = None,
    swing: int = 3,
    fast_window: int = _REGIME_FAST_WINDOW,
    news_window_min: int = _REGIME_NEWS_WINDOW_MIN,
) -> dict:
    """Detect whether today's intraday trajectory has TURNED, and how convincingly.

    Six independent reads of the same session bars, because any one of them
    alone is noise:

    * **Legs.** `piecewise_regimes` segments the session into momentum legs; the
      last is the leg in force now, the one before it what it replaced. Their
      transition is classified as a reversal (up->down or down->up), a stall
      (a directional leg giving way to a flat one), a resumption (flat into a
      directional leg), or none.
    * **Giveback.** How much of the prior leg the current one has already
      retraced -- a 20% pullback is a pause, 60% is a reversal.
    * **Velocity.** The last `fast_window` bars against the `fast_window` before
      them: accelerating, decelerating, or already reversing right now. This is
      the earliest of the six to fire.
    * **VWAP.** The session-anchored VWAP, which side price sits on, and how
      many bars ago that flipped.
    * **Structure.** Whether the last confirmed swing low has been broken to the
      downside (bearish break of structure) or the last swing high to the upside.
    * **Participation and catalyst.** Volume behind the bars since the turn
      versus the session's median minute, plus whether news landed near the turn
      or is otherwise fresh. A participated or news-backed turn re-prices the
      stock, which is what makes an existing level map stale.

    Volume here is whatever feed the bars came from -- for a live buffer that is
    the single-venue Alpaca tape, so `turn_volume_rel` is a like-for-like ratio
    within the session, not a consolidated-tape figure.

    Returns `shift_detected`, a `regime` label, the long-side `bias`,
    `levels_stale`, the component reads, reference levels for arming tactics,
    insights, and a one-line summary.
    """
    if len(bars) < 10:
        return {"note": "not enough bars for a regime read"}
    session = _session_bars(bars)
    if len(session) < 10:
        return {"note": "not enough bars in today's session yet for a regime read"}

    closes = [float(b["c"]) for b in session]
    price = float(spot) if spot else closes[-1]
    n = len(session)

    regimes = piecewise_regimes(_closes(session))
    if not regimes:
        return {"note": "could not segment the session into momentum legs"}
    current, prior = regimes[-1], (regimes[-2] if len(regimes) > 1 else None)

    def _leg(leg: "dict | None") -> "dict | None":
        if leg is None:
            return None
        out = {
            "direction": leg["direction"],
            "bars": leg["bars"],
            "pct_move": leg["pct_move"],
            "slope_per_bar": leg["slope_per_bar"],
        }
        start_t, end_t = clock.bar_dt(session[leg["start_index"]]), clock.bar_dt(session[leg["end_index"]])
        if start_t and end_t:
            out["duration_minutes"] = round((end_t - start_t).total_seconds() / 60.0, 1)
            out["started_at"] = start_t.astimezone(_ET).strftime("%H:%M ET")
        return out

    # --- The turn between the two legs.
    turn_index = current["start_index"]
    turn_dt = clock.bar_dt(session[turn_index])
    last_dt = clock.bar_dt(session[-1])
    transition = "none"
    if prior is not None and prior["direction"] != current["direction"]:
        if prior["direction"] in ("up", "down") and current["direction"] in ("up", "down"):
            transition = "reversal_down" if current["direction"] == "down" else "reversal_up"
        elif current["direction"] == "flat":
            transition = f"stall_from_{prior['direction']}"
        else:
            transition = f"resumption_{current['direction']}"
    turn = {
        "transition": transition,
        "price": round(closes[turn_index], 4),
        "bars_ago": n - 1 - turn_index,
    }
    if turn_dt:
        turn["at"] = turn_dt.astimezone(_ET).strftime("%H:%M ET")
        if last_dt:
            turn["minutes_ago"] = round((last_dt - turn_dt).total_seconds() / 60.0, 1)

    # --- Giveback: how much of the prior leg the current one has retraced.
    giveback_pct = None
    if prior is not None and prior["direction"] in ("up", "down"):
        prior_move = closes[prior["end_index"]] - closes[prior["start_index"]]
        retraced = closes[prior["end_index"]] - price
        if abs(prior_move) > 0:
            # Positive = giving back the prior leg (whichever way it ran).
            giveback_pct = round(retraced / prior_move * 100, 1)

    # --- Velocity right now vs the window before it.
    velocity = None
    if n >= 2 * fast_window + 1:
        base_fast, base_prior = closes[-1 - fast_window], closes[-1 - 2 * fast_window]
        fast_pct = (closes[-1] / base_fast - 1) * 100 if base_fast else 0.0
        prior_pct = (base_fast / base_prior - 1) * 100 if base_prior else 0.0
        if fast_pct * prior_pct < 0:
            state = "reversing"
        elif abs(fast_pct) <= abs(prior_pct) * _REGIME_STALL_RATIO:
            state = "decelerating"
        elif abs(fast_pct) > abs(prior_pct):
            state = "accelerating"
        else:
            state = "steady"
        velocity = {
            "state": state,
            "window_bars": fast_window,
            "recent_pct": round(fast_pct, 3),
            "previous_pct": round(prior_pct, 3),
        }

    # --- Session-anchored VWAP and the last cross of it.
    df = pd.DataFrame(session)
    typical = (df["h"] + df["l"] + df["c"]) / 3.0
    vol = df["v"].astype(float)
    cum_vol = vol.cumsum()
    vwap = vwap_cross = None
    if float(cum_vol.iloc[-1]) > 0:
        vwap_series = ((typical * vol).cumsum() / cum_vol).tolist()
        vwap = round(float(vwap_series[-1]), 4)
        vwap_cross = _last_vwap_cross(closes, vwap_series)

    # --- Break of structure against the last confirmed swings.
    swing_high, swing_low = _swing_extremes(session, swing)
    structure_break = None
    if swing_low and price < swing_low["price"]:
        structure_break = "bearish"
    elif swing_high and price > swing_high["price"]:
        structure_break = "bullish"

    # --- Participation behind the turn.
    session_vols = [float(b.get("v") or 0.0) for b in session]
    positive = [v for v in session_vols if v > 0]
    baseline = float(np.median(positive)) if positive else 0.0
    since_turn = session_vols[turn_index:]
    turn_volume_rel = (
        round((sum(since_turn) / len(since_turn)) / baseline, 2) if baseline > 0 and since_turn else None
    )

    # --- Catalyst: news at the turn, or fresh news since.
    news_dts = _news_datetimes(news_times)
    news_at_turn = bool(turn_dt and _news_near(news_dts, turn_dt, news_window_min))
    minutes_since_news = None
    if last_dt:
        elapsed = [(last_dt - nd).total_seconds() / 60.0 for nd in news_dts if nd <= last_dt]
        if elapsed:
            minutes_since_news = round(min(elapsed), 1)
    fresh_news = minutes_since_news is not None and minutes_since_news <= news_window_min

    # --- Verdict. Count the bearish and bullish evidence, then label the tape.
    bearish = [
        transition in ("reversal_down", "stall_from_up", "resumption_down"),
        giveback_pct is not None and prior is not None and prior["direction"] == "up" and giveback_pct >= 50,
        velocity is not None and velocity["state"] == "reversing" and velocity["recent_pct"] < 0,
        vwap_cross is not None and vwap_cross["direction"] == "lost_vwap",
        structure_break == "bearish",
    ]
    bullish = [
        transition in ("reversal_up", "stall_from_down", "resumption_up"),
        giveback_pct is not None and prior is not None and prior["direction"] == "down" and giveback_pct >= 50,
        velocity is not None and velocity["state"] == "reversing" and velocity["recent_pct"] > 0,
        vwap_cross is not None and vwap_cross["direction"] == "reclaimed_vwap",
        structure_break == "bullish",
    ]
    bearish_signals, bullish_signals = sum(bearish), sum(bullish)
    shift_detected = max(bearish_signals, bullish_signals) >= 2
    participated = turn_volume_rel is not None and turn_volume_rel >= _REGIME_TURN_VOLUME_MULT

    if shift_detected and bearish_signals > bullish_signals:
        direction, regime = "bearish", "turning_down"
    elif shift_detected and bullish_signals > bearish_signals:
        direction, regime = "bullish", "turning_up"
    else:
        direction = None
        regime = {
            "up": "trending_up",
            "down": "trending_down",
            "flat": "range_bound",
        }[current["direction"]]

    # Long-only account: the bias is about defending or committing capital.
    if direction == "bearish":
        bias = "exit_or_stand_aside"
    elif regime == "trending_down":
        bias = "stand_aside"
    elif direction == "bullish":
        bias = "re_engage"
    elif regime == "trending_up":
        bias = "trend_intact"
    else:
        bias = "wait_for_direction"

    # The level map is built from the session's past; a participated or
    # news-backed turn is exactly what invalidates it.
    levels_stale = bool((shift_detected and participated) or news_at_turn or fresh_news)

    insights: list[str] = []
    if transition != "none":
        insights.append(
            f"Momentum leg turned {transition.replace('_', ' ')} at {turn['price']:.2f}"
            + (f" ({turn['at']}, {turn['bars_ago']} bars ago)" if "at" in turn else "")
            + "."
        )
    if giveback_pct is not None and giveback_pct >= 50:
        insights.append(
            f"{giveback_pct:.0f}% of the prior {prior['direction']} leg has been given back -- "
            "a retrace this deep is a reversal, not a pause."
        )
    if velocity is not None and velocity["state"] in ("reversing", "decelerating"):
        insights.append(
            f"Velocity {velocity['state']}: {velocity['recent_pct']:+.2f}% over the last "
            f"{fast_window} bars against {velocity['previous_pct']:+.2f}% before them."
        )
    if vwap_cross is not None and vwap_cross["bars_ago"] <= 20:
        insights.append(
            f"Price {vwap_cross['direction'].replace('_', ' ')} {vwap_cross['vwap_at_cross']:.2f} "
            f"{vwap_cross['bars_ago']} bars ago; session VWAP now {vwap:.2f}."
        )
    if structure_break == "bearish":
        insights.append(
            f"Bearish break of structure: price {price:.2f} is below the last confirmed swing low "
            f"{swing_low['price']:.2f}. Support below is untested in this leg."
        )
    elif structure_break == "bullish":
        insights.append(
            f"Bullish break of structure: price {price:.2f} is above the last confirmed swing high "
            f"{swing_high['price']:.2f}."
        )
    if turn_volume_rel is not None:
        insights.append(
            f"Volume since the turn is {turn_volume_rel:.2f}x the session's median minute -- "
            + ("participated, so the handover is real." if participated else "thin, so treat the turn as provisional.")
        )
    if news_at_turn:
        insights.append("News landed at the turn: the move is catalyst-driven and old levels are re-priced.")
    elif fresh_news:
        insights.append(f"News {minutes_since_news:.0f} minutes old is still live on the tape.")
    if levels_stale:
        insights.append("Re-run the level map: the levels behind any armed plan predate this turn.")

    summary = (
        f"Regime {regime.replace('_', ' ')} (bias: {bias.replace('_', ' ')}). Current "
        f"{current['direction']} leg {current['pct_move']:+.2f}% over {current['bars']} bars"
        + (f", following a {prior['direction']} leg {prior['pct_move']:+.2f}%" if prior else "")
        + f". {bearish_signals} bearish / {bullish_signals} bullish shift signal(s)"
        + ("; level map is stale." if levels_stale else ".")
    )

    return {
        "regime": regime,
        "shift_detected": shift_detected,
        "shift_direction": direction,
        "bias": bias,
        "levels_stale": levels_stale,
        "bearish_signals": bearish_signals,
        "bullish_signals": bullish_signals,
        "current_leg": _leg(current),
        "prior_leg": _leg(prior),
        "legs_in_session": len(regimes),
        "turn": turn,
        "giveback_pct_of_prior_leg": giveback_pct,
        "velocity": velocity,
        "vwap": vwap,
        "vwap_position": (
            None if vwap is None else ("above" if price >= vwap else "below")
        ),
        "last_vwap_cross": vwap_cross,
        "structure_break": structure_break,
        "last_swing_high": swing_high,
        "last_swing_low": swing_low,
        "turn_volume_rel": turn_volume_rel,
        "news_at_turn": news_at_turn,
        "minutes_since_news": minutes_since_news,
        "reference_levels": {
            "price": round(price, 4),
            "session_vwap": vwap,
            "turn_price": turn["price"],
            "last_swing_high": swing_high["price"] if swing_high else None,
            "last_swing_low": swing_low["price"] if swing_low else None,
            "session_high": round(max(float(b["h"]) for b in session), 4),
            "session_low": round(min(float(b["l"]) for b in session), 4),
        },
        "insights": insights,
        "session_bars": n,
        "summary": summary,
    }
