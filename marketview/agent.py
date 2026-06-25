"""
LLM trading agent.

The agent reads data that the app has already fetched (intraday bars, daily
bars, news, quotes — all living on `AppState`) via tool calls, reasons about
the trading regime and a fitting strategy, then finalizes each cycle with
exactly one `submit_decision` tool call (buy / sell / sleep). The decision is
handed to a `DecisionTracker`, which independently fetches the fill price —
the agent never gets to pick its own fill price.

Trading is paper-only: `DecisionTracker` defaults to `PaperBroker`. Swapping
in a live broker later only requires implementing `Broker` and passing it to
`DecisionTracker` — this module doesn't need to change.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable

from . import historical
from . import observability as obs
from . import technical_analysis as ta
from .config import AGENT_MAX_TOOL_ITERS
from .llm import DEFAULT_AGENT_MODELS, get_agent_client
from .state import alert_triggered

if TYPE_CHECKING:
    from .decisions import DecisionTracker
    from .state import AppState


AGENT_SYSTEM_PROMPT = """\
You are an autonomous trading research agent for a single equity ticker, \
operating in a paper-trading sandbox -- no real orders are ever placed, so \
reason as if real capital is on the line.

Work through this process every cycle. The analysis tools don't just return a \
verdict -- they return specific numbers (support/resistance, RSI, ATR, moving \
averages, call/put walls). Cite and use those numbers, don't just paraphrase \
the labels.

1. ESTABLISH THE REGIME. Call analyze_daily_trend first. Use `regime` and \
`trend_strength` to classify bullish/bearish/neutral, but don't stop there -- \
note the actual `support` and `resistance` levels (these are your candidate \
alert/stop levels later), the `moving_average_alignment` (a clean bullish or \
bearish stack is higher conviction than a mixed one), and `rsi_label` \
(overbought caps how aggressively you chase a bullish regime; oversold caps \
how aggressively you press a bearish one -- in either case prefer waiting for \
a pullback/bounce over chasing). Don't skip this step -- it determines which \
strategy applies below. Then call analyze_market to read the broad-market \
backdrop (VIX level/trend, VIX term structure, S&P 500 trend and drawdown). \
A risk-off market (high or rising VIX, inverted term structure, S&P below its \
200-day average or in a correction) is a reason to demand more conviction and \
size smaller even when the ticker's own trend looks bullish; a risk-on \
backdrop is a tailwind. Use its `risk_score` and `insights` to scale size and \
stop width, not just to color the narrative.

2. CHECK INTRADAY CONFIRMATION. Call analyze_intraday_momentum and \
analyze_volume. Look for whether short-term price action and volume confirm \
or contradict the regime (e.g. a bullish regime with breaking-down intraday \
price and weak volume is a warning sign, not a buy signal). Read \
`volatility_pct_of_price` (from ATR) as a direct input to position size and \
stop distance -- wider ATR means a wider stop is needed to avoid noise \
stop-outs, which means a smaller share count for the same dollar risk. Read \
`confirmation` from analyze_volume literally: "diverging" volume against the \
move is a reason to downgrade conviction even if price action looks right. \
If get_put_call_walls has data, treat the Call Wall/Put Wall as concrete \
nearby resistance/support levels -- combine them with the daily \
support/resistance from step 1 (when they cluster near the same price, that \
level is higher-confidence) -- and use the net gamma regime to gauge whether \
a move toward a wall is likely to stall (positive gamma) or accelerate \
through it (negative gamma).

3. CHECK NEWS. Call get_news. Treat clearly negative news (or negative-for- \
symbol competitor news) as a reason to be more conservative even in a \
bullish regime, and vice versa.

4. CHOOSE A STRATEGY THAT FITS THE REGIME:
   - Bullish + confirming volume/momentum -> trend-following: look to buy on \
strength or on a shallow pullback, with the put wall / nearest support as \
your downside reference and the call wall / resistance as your target or \
the level beyond which the move likely needs fresh conviction.
   - Bearish + confirming volume -> defensive: avoid new buys, consider \
selling existing exposure.
   - Neutral/choppy or conflicting signals -> mean-reversion or stand aside: \
prefer sleep unless there is a clear, well-confirmed edge.
   When signals conflict or conviction is low, the correct decision is \
sleep. Trading is optional; capital preservation matters more than being in \
a position.

5. SIZE THE TRADE. Call get_position to see current cash and share count \
before deciding quantity. Never request a sell quantity larger than the \
current position. Size buys conservatively relative to cash available -- \
this is one ticker in what should be a diversified book, not the whole \
account. Let the volatility read from step 2 (`volatility_pct_of_price`) and \
the market risk score from step 1 scale size down together: high ATR and/or \
a risk-off market both argue for a smaller, not larger, position for the \
same conviction level.

6. FINALIZE. Call submit_decision exactly once, with the regime you \
established, the action (buy/sell/sleep/alert), a quantity (omit or 0 for \
sleep/alert), and reasoning that ties together the regime, the strategy, \
and why this specific action follows from it -- reference the actual support/ \
resistance, RSI, ATR, or wall levels you used, not just the regime label. Do \
not call submit_decision more than once, and do not stop without calling it.

Be decisive but not reckless: sleep is a valid and often correct decision. \
If you'd otherwise sleep but there's a specific price level (or a bracket of \
two -- a downside level and an upside level, e.g. a stop-loss below and a \
breakout level above) that would change your mind before the next scheduled \
cycle, use action "alert" instead of "sleep" and set alert_low_price (wakes \
you when price falls to/below it) and/or alert_high_price (wakes you when \
price rises to/above it). Ground these levels in what the tools already gave \
you -- the daily support/resistance, the call/put walls, or a recent \
intraday swing high/low -- rather than picking an arbitrary distance from the \
current price. Set just one for a single level, or both to watch a range \
from both sides in one cycle. This wakes you up early -- as soon as either \
level is crossed -- instead of waiting out the full fixed cycle interval \
blind to what happens in between. Use plain "sleep" when no specific level \
is worth watching.

Regardless of which action you choose, you will also be woken up early -- \
before the next scheduled cycle -- if fresh news for the ticker arrives \
while you're waiting. You don't need to do anything to enable this; it \
happens automatically so a sleep/alert decision is never blind to breaking \
news.
"""

MOMENTUM_SYSTEM_PROMPT = """\
You are an autonomous momentum-trading agent for a single equity ticker, \
operating in a paper-trading sandbox -- no real orders are ever placed, so \
reason as if real capital is on the line.

Core idea: stocks in motion tend to stay in motion. You are not predicting a \
new move -- you are jumping on a move already in progress, riding it, and \
getting out before it reverses. Most of the day there is nothing to do; only \
take A+ setups and sleep the rest of the time.

Work through this process every cycle, citing the actual numbers the tools \
return (levels, ratios, RSI, ATR), not just their labels:

1. SCREEN FOR A MOMENTUM CONDITION. Call get_quote (price vs prior close -- a \
5-20% gap is the sweet spot; bigger than that is often already parabolic and \
late) and analyze_volume (relative volume -- you want it clearly elevated, \
2x+ is the kind of move worth your attention; flat/declining volume means \
there's no real participation behind the move). Call get_news to find the \
catalyst -- earnings beat, upgrade, FDA news, M&A. A move with no catalyst \
and no volume is noise, not momentum; default to sleep.

2. IDENTIFY THE SETUP. Call analyze_intraday_momentum for the higher-highs/ \
higher-lows pattern, VWAP position, and ATR-based volatility. Match what you \
see to one of:
   - Bull flag: a sharp move (the flagpole) followed by a tight, low-volume \
consolidation, then a fresh breakout on rising volume.
   - VWAP reclaim: price dipped to/through session VWAP, found buyers, and is \
reclaiming it -- analyze_intraday_momentum's vwap_position tells you which \
side of VWAP price is on right now.
   If neither is present, there is no trade -- sleep.

3. ENTRY DISCIPLINE. Never chase. Require: a recognizable setup from step 2, \
a clear breakout/reclaim level (the flag's high, or VWAP), and volume \
confirmation of at least 1.5x average -- analyze_volume's `relative_volume` \
speaks to this directly. Know your stop before you size the trade: for a \
bull flag, the stop sits just below the consolidation low; for a VWAP \
reclaim, just below VWAP.

4. SIZE THE TRADE. Call get_position for current cash and share count. Risk \
a small, fixed slice of the account on the distance between entry and your \
stop -- momentum trades move fast and wrong setups should cost little. Wider \
ATR (from analyze_intraday_momentum) means a wider stop, which means a \
smaller share count for the same dollar risk. Never request a sell quantity \
larger than the current position.

5. EXIT DISCIPLINE (when you already hold a position). Sell or tighten the \
stop when: volume dries up (analyze_volume showing decreasing/diverging \
volume) with no fresh buyers, price breaks back below VWAP, intraday \
momentum has rolled into lower-highs/lower-lows or a reversal candle near \
resistance, or it's drifted into the 12:00-14:00 dead zone without strength \
(check the bar timestamps) -- unless the stock is exceptionally strong. Once \
the position is up roughly 1R (one stop-distance) move your effective stop \
to breakeven via the alert mechanism in step 6, and otherwise let winners \
run rather than booking small gains out of fear. Cut losers immediately if \
the setup fails -- don't wait to see.

6. FINALIZE. Call submit_decision exactly once: action (buy/sell/sleep/ \
alert), quantity (omit or 0 for sleep/alert), the regime, and reasoning that \
names the setup, the breakout/stop levels, and the volume confirmation you \
used. If you'd otherwise sleep but there's a specific trigger level worth \
watching (a breakout level above, a stop level below, or both), use action \
"alert" with alert_low_price and/or alert_high_price instead -- this wakes \
you the instant price crosses it rather than waiting out the full cycle \
blind. Do not call submit_decision more than once, and do not stop without \
calling it.

Emotional discipline matters more than any single setup: sitting on your \
hands through a quiet, no-edge stretch is correct and far more common than \
trading. Sleep is a valid and often correct decision.

Regardless of which action you choose, you will also be woken up early -- \
before the next scheduled cycle -- if fresh news for the ticker arrives, or \
if a high-volume alert fires intraday. You don't need to do anything to \
enable this; it happens automatically.
"""

BREAKOUT_SYSTEM_PROMPT = """\
You are an autonomous breakout-trading agent for a single equity ticker, \
operating in a paper-trading sandbox -- no real orders are ever placed, so \
reason as if real capital is on the line.

Core idea: the first part of the session sets a level -- the opening range -- \
and when price finally clears it on a surge in volume, trapped sellers get \
stopped out and new buyers rush in, creating a self-reinforcing move. You are \
not predicting the break -- you are waiting for it to actually happen, with \
volume proving real buying pressure is behind it, and only then acting. Most \
cycles there is nothing to do; only take A+ setups and sleep the rest of the \
time.

Work through this process every cycle, citing the actual numbers the tools \
return (levels, ratios, ATR), not just their labels:

1. WAIT FOR THE OPENING-RANGE BREAK. Call analyze_opening_range for today's \
opening-range high/low and whether `status` shows price has broken out above \
or below it yet. This range is your level -- don't anticipate the break, \
wait for `status` to actually show it.

2. DEMAND VOLUME. Call analyze_volume for relative volume and confirmation. A \
breakout is only valid with volume at least 1.5x average -- ideally 2-3x \
(`analyze_opening_range`'s `volume_ratio_vs_opening_range` and \
analyze_volume's `relative_volume` and `confirmation` both speak to this \
directly). No volume spike means no trade, full stop, regardless of how \
clean the range break looks.

3. RULE OUT A FALSE BREAKOUT. A break that closes back inside the range, on \
weak volume, or with a long wick rejecting the level, is a fakeout, not a \
breakout -- it often reverses sharply as the trapped longs (or shorts) bail \
out. If you see those tells, do not buy the break; consider whether the \
reversal itself is the trade (a fade back through the level), or simply \
sleep/alert and wait for a cleaner signal.

4. CHECK FOR A CATALYST. Call get_news. A breakout with a real catalyst \
behind it (earnings, guidance, upgrade, macro data) is more likely to follow \
through than one on no news -- demand a cleaner setup and smaller size when \
there's no catalyst.

5. ENTRY DISCIPLINE -- DON'T CHASE. Prefer the close of the breakout bar, or \
better, a pullback/retest of the range high/low (resistance-turned-support) \
for a better risk/reward. If price is already extended well beyond the range \
(it ran 5-8%+ past it with no pullback), it's too late -- this is chasing, \
not breakout trading; sleep and wait for the next range to form instead of \
buying the extension.

6. SIZE THE TRADE WITH ATR-BASED TARGETS. Your stop sits just below the \
opening-range low (never at a round number -- nudge it just under the \
structure). Call analyze_intraday_momentum for the current `atr`, then call \
breakout_trade_geometry with your entry, that stop, and the `atr` to get \
projected targets and reward/risk ratios. Require `meets_min_reward_risk` to \
be true (at least 2:1) -- if it isn't, do not take the trade; either it's a \
bad entry or the stop is too wide. Call get_position for current cash/shares \
before sizing, and risk only a small, fixed slice of the account on the \
entry-to-stop distance. Never request a sell quantity larger than the \
current position.

7. EXIT DISCIPLINE (when you already hold a position from a prior breakout). \
Sell or tighten the stop when volume dries up with no fresh buyers \
(analyze_volume showing decreasing/diverging volume), price closes back \
inside the broken range, or momentum rolls over (analyze_intraday_momentum \
showing lower highs/lower lows). Once price has reached roughly 1x ATR \
beyond entry, consider moving the stop to breakeven via the alert mechanism \
in step 8 rather than risking a full round-trip back to the original stop.

8. FINALIZE. Call submit_decision exactly once: action (buy/sell/sleep/ \
alert), quantity (omit or 0 for sleep/alert), the regime, and reasoning that \
names the range level, the volume confirmation, the entry/stop/target \
geometry, and why this action follows from it. If you'd otherwise sleep but \
there's a specific level worth watching (the range high above, the range low \
below, or both), use action "alert" with alert_low_price and/or \
alert_high_price instead -- this wakes you the instant price crosses it \
rather than waiting out the full cycle blind. Do not call submit_decision \
more than once, and do not stop without calling it.

Patience is the edge here: passing on setups with no clean range break or no \
volume confirmation is correct and far more common than trading. Sleep is a \
valid and often correct decision.

Regardless of which action you choose, you will also be woken up early -- \
before the next scheduled cycle -- if fresh news for the ticker arrives, or \
if a high-volume alert fires intraday. You don't need to do anything to \
enable this; it happens automatically.
"""

AGENT_PERSONALITIES: dict[str, dict[str, str]] = {
    "swing": {"label": "Swing / Position Trader", "system_prompt": AGENT_SYSTEM_PROMPT},
    "momentum": {"label": "Momentum Trader", "system_prompt": MOMENTUM_SYSTEM_PROMPT},
    "breakout": {"label": "Breakout Trader", "system_prompt": BREAKOUT_SYSTEM_PROMPT},
}
DEFAULT_PERSONALITY = "swing"

_TOOL_GET_QUOTE = {
    "type": "function",
    "function": {
        "name": "get_quote",
        "description": "Get the latest streamed quote and trade price for the ticker.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

_TOOL_ANALYZE_INTRADAY_MOMENTUM = {
    "type": "function",
    "function": {
        "name": "analyze_intraday_momentum",
        "description": (
            "Analyze recent intraday price action for the ticker: momentum pattern "
            "(higher highs/lows vs lower highs/lows), position relative to session VWAP, "
            "and ATR-based volatility. Returns labeled values plus a one-line summary."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Number of most recent bars to analyze (default 50, max 300).",
                }
            },
            "required": [],
        },
    },
}

_TOOL_ANALYZE_DAILY_TREND = {
    "type": "function",
    "function": {
        "name": "analyze_daily_trend",
        "description": (
            "Analyze daily bars (up to ~1 year) to establish the medium-term trading regime: "
            "bullish/bearish/neutral with strength, moving-average alignment, RSI, and recent "
            "support/resistance. Returns labeled values plus a one-line summary."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Number of most recent daily bars (default 60).",
                }
            },
            "required": [],
        },
    },
}

_TOOL_ANALYZE_MARKET = {
    "type": "function",
    "function": {
        "name": "analyze_market",
        "description": (
            "Analyze broad-market conditions (independent of the ticker) using the best-known "
            "regime gauges: the VIX fear level and its trend, the VIX term structure "
            "(near-term vs 3-month implied vol), and the S&P 500's primary trend, drawdown, and "
            "RSI. Returns a risk-on/neutral/risk-off classification, labeled markers, and a list "
            "of actionable insights. Use it to set the overall risk backdrop before sizing trades."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

_TOOL_ANALYZE_VOLUME = {
    "type": "function",
    "function": {
        "name": "analyze_volume",
        "description": (
            "Analyze recent trade volume to gauge participation: relative volume vs the prior "
            "window, on-balance-volume trend, and whether volume confirms or diverges from the "
            "recent price move. Returns labeled values plus a one-line summary."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

_TOOL_ANALYZE_OPENING_RANGE = {
    "type": "function",
    "function": {
        "name": "analyze_opening_range",
        "description": (
            "Analyze today's Opening Range Breakout (ORB) setup: the high/low set by the "
            "first N minutes of today's session, whether price has since broken out above "
            "or below that range, and whether recent volume confirms the breakout. Returns "
            "labeled values plus a one-line summary."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "minutes": {
                    "type": "integer",
                    "description": "Length of the opening range in minutes (default 15).",
                }
            },
            "required": [],
        },
    },
}

_TOOL_GET_PUT_CALL_WALLS = {
    "type": "function",
    "function": {
        "name": "get_put_call_walls",
        "description": (
            "Read the Call Wall (resistance, from peak call open interest) and Put Wall "
            "(support, from peak put open interest) for the ticker, plus the net dealer-gamma "
            "regime (positive = dampening, negative = amplifying) and whether those walls have "
            "been rising/falling recently. Uses the options chain most recently fetched in the "
            "background -- does not fetch fresh data itself. Returns labeled values, actionable "
            "insights, and a one-line summary."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

_TOOL_GET_NEWS = {
    "type": "function",
    "function": {
        "name": "get_news",
        "description": "Get recent news headlines/summaries for the ticker, with impact labels where available.",
        "parameters": {
            "type": "object",
            "properties": {"limit": {"type": "integer", "description": "Max number of articles (default 10)."}},
            "required": [],
        },
    },
}

_TOOL_GET_POSITION = {
    "type": "function",
    "function": {
        "name": "get_position",
        "description": (
            "Get the current paper trading position size, cash balance, and total "
            "portfolio value (cash + position marked to the latest price)."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

_TOOL_SUBMIT_DECISION = {
    "type": "function",
    "function": {
        "name": "submit_decision",
        "description": (
            "Finalize this trading cycle with exactly one decision: buy, sell, sleep, or "
            "alert. Must be called exactly once, after analysis is complete."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["buy", "sell", "sleep", "alert"]},
                "quantity": {
                    "type": "number",
                    "description": "Shares to buy/sell. Ignored for sleep/alert. Must be > 0 for buy/sell.",
                },
                "regime": {
                    "type": "string",
                    "enum": ["bullish", "bearish", "neutral"],
                    "description": "The trading regime established during analysis.",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Concise justification covering regime, strategy, and why this action follows from it.",
                },
                "alert_low_price": {
                    "type": "number",
                    "description": (
                        "When action is 'alert': wake the agent early if price falls to or "
                        "below this level. Optional -- set this, alert_high_price, or both."
                    ),
                },
                "alert_high_price": {
                    "type": "number",
                    "description": (
                        "When action is 'alert': wake the agent early if price rises to or "
                        "above this level. Optional -- set this, alert_low_price, or both."
                    ),
                },
            },
            "required": ["action", "reasoning"],
        },
    },
}

_TOOL_BREAKOUT_TRADE_GEOMETRY = {
    "type": "function",
    "function": {
        "name": "breakout_trade_geometry",
        "description": (
            "Compute the mechanical entry/stop/target math for a long breakout trade: targets "
            "projected from ATR (1x and 2x), the resulting reward-to-risk ratio for each, and "
            "whether the best one clears the 2:1 minimum. Use this instead of doing the "
            "arithmetic yourself before sizing a trade."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "entry": {"type": "number", "description": "Planned entry price."},
                "stop": {"type": "number", "description": "Planned stop-loss price, below entry."},
                "atr": {
                    "type": "number",
                    "description": "ATR (from analyze_intraday_momentum), for an ATR-multiple target.",
                },
            },
            "required": ["entry", "stop"],
        },
    },
}

# Swing/position trader: the full kitchen sink -- medium-term regime, broad-market
# backdrop, and options positioning all matter on this slower timescale.
BASE_TOOLS: list[dict] = [
    _TOOL_GET_QUOTE,
    _TOOL_ANALYZE_INTRADAY_MOMENTUM,
    _TOOL_ANALYZE_DAILY_TREND,
    _TOOL_ANALYZE_MARKET,
    _TOOL_ANALYZE_VOLUME,
    _TOOL_ANALYZE_OPENING_RANGE,
    _TOOL_GET_PUT_CALL_WALLS,
    _TOOL_GET_NEWS,
    _TOOL_GET_POSITION,
    _TOOL_SUBMIT_DECISION,
]

# Momentum trader: RVOL + price action/VWAP + news + price -- no medium-term regime
# or broad-market backdrop, the whole point is reacting fast to what's happening now.
MOMENTUM_TOOLS: list[dict] = [
    _TOOL_GET_QUOTE,
    _TOOL_ANALYZE_INTRADAY_MOMENTUM,
    _TOOL_ANALYZE_VOLUME,
    _TOOL_GET_NEWS,
    _TOOL_GET_POSITION,
    _TOOL_SUBMIT_DECISION,
]

# Breakout trader: ORB + volume + ATR-based targets + news + price.
BREAKOUT_TOOLS: list[dict] = [
    _TOOL_GET_QUOTE,
    _TOOL_ANALYZE_OPENING_RANGE,
    _TOOL_ANALYZE_VOLUME,
    _TOOL_ANALYZE_INTRADAY_MOMENTUM,
    _TOOL_BREAKOUT_TRADE_GEOMETRY,
    _TOOL_GET_NEWS,
    _TOOL_GET_POSITION,
    _TOOL_SUBMIT_DECISION,
]

PERSONALITY_TOOLS: dict[str, list[dict]] = {
    "swing": BASE_TOOLS,
    "momentum": MOMENTUM_TOOLS,
    "breakout": BREAKOUT_TOOLS,
}


def _log(state: "AppState", entry: dict) -> None:
    entry = {"ts": datetime.now(timezone.utc).isoformat(), **entry}
    with state.lock:
        state.agent_log.append(entry)


def _tool_get_quote(state: "AppState") -> dict:
    with state.lock:
        return {
            "last_price": state.last_price,
            "prev_close": state.prev_close,
            "bid_price": state.bid_price,
            "bid_size": state.bid_size,
            "ask_price": state.ask_price,
            "ask_size": state.ask_size,
        }


def _tool_analyze_intraday_momentum(state: "AppState", limit: object = None) -> dict:
    n = max(1, min(int(limit or 50), 300))
    with state.lock:
        bars = list(state.bars)[-n:]
    if not bars:
        return {"note": "no intraday bars available yet"}
    return ta.analyze_intraday(bars)


def _tool_analyze_daily_trend(state: "AppState", limit: object = None) -> dict:
    n = max(1, min(int(limit or 60), 365))
    bars = list(state.daily_bars)[-n:]
    if not bars:
        return {"note": "no daily bars available yet"}
    return ta.analyze_trend(bars)


def _tool_analyze_opening_range(state: "AppState", minutes: object = None) -> dict:
    n = max(1, min(int(minutes or 15), 120))
    with state.lock:
        bars = list(state.bars)
    if not bars:
        return {"note": "no intraday bars available yet"}
    return ta.analyze_opening_range(bars, minutes=n)


def _tool_analyze_market(state: "AppState") -> dict:
    data = historical.fetch_market_indicators()
    return ta.analyze_market(
        vix_close=data.get("vix"),
        spy_close=data.get("spy"),
        vix3m_close=data.get("vix3m"),
    )


def _tool_analyze_volume(state: "AppState") -> dict:
    with state.lock:
        bars = list(state.bars)
    if not bars:
        return {"note": "no intraday bars available yet"}
    return ta.analyze_volume(bars)


def _tool_get_put_call_walls(state: "AppState") -> dict:
    with state.lock:
        data = state.options_chain
        history = list(state.options_wall_history)
    if not data:
        return {"note": "no options chain data available yet"}
    return ta.get_put_call_walls_and_gamma(
        strikes=data["strikes"],
        calls_oi=data["calls_oi"],
        puts_oi=data["puts_oi"],
        calls_gamma_exposure=data["calls_gamma_exposure"],
        puts_gamma_exposure=data["puts_gamma_exposure"],
        spot=data["spot"],
        wall_history=history,
    )


def _tool_get_news(state: "AppState", limit: object = None) -> dict:
    n = max(1, min(int(limit or 10), 30))
    with state.lock:
        news = list(state.news)[:n]
        impacts = dict(state.news_impacts)
    return {
        "articles": [
            {
                "headline": item.get("headline"),
                "summary": item.get("summary"),
                "created_at": item.get("created_at"),
                "source": item.get("source"),
                "impact": impacts.get(str(item.get("id", "")), "unknown"),
            }
            for item in news
        ]
    }


def _tool_get_position(state: "AppState", tracker: "DecisionTracker") -> dict:
    snap = tracker.snapshot()
    return {
        "cash": snap["cash"],
        "position": snap["position"],
        # Kept fresh independently by the price stream, not fetched here.
        "portfolio_value": state.portfolio_value,
        "decisions_so_far": len(snap["decisions"]),
    }


def _tool_breakout_trade_geometry(state: "AppState", entry: object, stop: object, atr: object = None) -> dict:
    return ta.breakout_trade_geometry(
        float(entry),
        float(stop),
        atr=float(atr) if atr is not None else None,
    )


_DISPATCH: dict[str, Callable[[dict, "AppState", "DecisionTracker"], dict]] = {
    "get_quote": lambda args, state, tracker: _tool_get_quote(state),
    "analyze_intraday_momentum": lambda args, state, tracker: _tool_analyze_intraday_momentum(state, args.get("limit")),
    "analyze_daily_trend": lambda args, state, tracker: _tool_analyze_daily_trend(state, args.get("limit")),
    "analyze_opening_range": lambda args, state, tracker: _tool_analyze_opening_range(state, args.get("minutes")),
    "analyze_market": lambda args, state, tracker: _tool_analyze_market(state),
    "analyze_volume": lambda args, state, tracker: _tool_analyze_volume(state),
    "breakout_trade_geometry": lambda args, state, tracker: _tool_breakout_trade_geometry(
        state, args.get("entry"), args.get("stop"), args.get("atr")
    ),
    "get_put_call_walls": lambda args, state, tracker: _tool_get_put_call_walls(state),
    "get_news": lambda args, state, tracker: _tool_get_news(state, args.get("limit")),
    "get_position": lambda args, state, tracker: _tool_get_position(state, tracker),
}


def _dispatch_tool(name: str, args: dict, state: "AppState", tracker: "DecisionTracker") -> dict:
    handler = _DISPATCH.get(name)
    if handler is None:
        return {"error": f"unknown tool {name}"}
    try:
        return handler(args, state, tracker)
    except Exception as exc:
        return {"error": str(exc)}


def _preview(text: str, width: int = 200) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"


@obs.observe(name="agent-cycle")
def run_agent_cycle(
    client: Any,
    model: str,
    symbol: str,
    state: "AppState",
    tracker: "DecisionTracker",
    max_iters: int = AGENT_MAX_TOOL_ITERS,
    personality: str = DEFAULT_PERSONALITY,
) -> None:
    """Run one analyze-then-decide cycle. Always ends with exactly one recorded decision.

    When Langfuse is configured, the whole cycle is one trace: every LLM turn
    nests under it as a generation, so per-cycle latency, token usage, and cost
    roll up automatically (see `marketview.observability`).
    """
    obs.update_trace(
        name=f"agent-cycle:{symbol}", input=symbol, metadata={"model": model, "symbol": symbol, "personality": personality}
    )
    system_prompt = AGENT_PERSONALITIES.get(personality, AGENT_PERSONALITIES[DEFAULT_PERSONALITY])["system_prompt"]
    tools = PERSONALITY_TOOLS.get(personality, BASE_TOOLS)
    state.price_alerts = []
    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Ticker: {symbol}. Run your analysis process and finish by calling submit_decision.",
        },
    ]
    _log(state, {"type": "cycle_start", "text": f"Starting analysis cycle for {symbol}"})

    decision_made = False
    for _ in range(max_iters):
        try:
            response = client.chat.completions.create(
                model=model, messages=messages, tools=tools, tool_choice="auto"
            )
        except Exception as exc:
            _log(state, {"type": "error", "text": f"LLM call failed: {exc}"})
            break

        msg = response.choices[0].message
        if msg.content:
            _log(state, {"type": "analysis", "text": msg.content})

        assistant_msg: dict = {"role": "assistant", "content": msg.content}
        tool_calls = getattr(msg, "tool_calls", None) or []
        if tool_calls:
            calls = []
            for tc in tool_calls:
                call = {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                # Gemini 3+ "thinking" models attach a thought_signature here that must be
                # echoed back verbatim on the next turn, or the API rejects the request with
                # "Function call ... is missing a thought_signature".
                extra_content = getattr(tc, "extra_content", None)
                if extra_content:
                    call["extra_content"] = extra_content
                calls.append(call)
            assistant_msg["tool_calls"] = calls
        messages.append(assistant_msg)

        if not tool_calls:
            messages.append(
                {"role": "user", "content": "Please finalize this cycle by calling submit_decision now."}
            )
            continue

        for tc in tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            if name == "submit_decision":
                action = args.get("action", "sleep")
                quantity = float(args.get("quantity") or 0)
                reasoning = args.get("reasoning", "")
                regime = args.get("regime", "unknown")
                alert_low_price = args.get("alert_low_price")
                alert_high_price = args.get("alert_high_price")
                alerts = []
                if alert_low_price is not None:
                    alerts.append({"price": float(alert_low_price), "condition": "below"})
                if alert_high_price is not None:
                    alerts.append({"price": float(alert_high_price), "condition": "above"})

                if action == "alert" and not alerts:
                    # Some models (small/cheap ones especially) pick action="alert" but
                    # forget the optional price fields. Reject and let the model retry
                    # instead of silently demoting to sleep -- it never sees that
                    # happen otherwise, so it can't course-correct.
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps(
                                {
                                    "error": (
                                        "action 'alert' requires alert_low_price and/or "
                                        "alert_high_price. Call submit_decision again with at "
                                        "least one price level, or use action 'sleep' if no "
                                        "level is worth watching."
                                    )
                                }
                            ),
                        }
                    )
                    continue

                if action in ("buy", "sell") and quantity > 0:
                    decision = tracker.record_trade(
                        symbol, action, quantity, reasoning, state.api_key, state.api_secret, state.feed
                    )
                elif action == "alert":
                    decision = tracker.record_alert(symbol, alerts, reasoning)
                    state.price_alerts = alerts
                else:
                    decision = tracker.record_sleep(symbol, reasoning)
                _log(
                    state,
                    {
                        "type": "decision",
                        "action": decision.action,
                        "status": decision.status,
                        "price": decision.price,
                        "quantity": decision.filled_quantity,
                        "reasoning": reasoning,
                        "regime": regime,
                        "alerts": decision.alerts,
                    },
                )
                result_content = json.dumps(
                    {
                        "status": decision.status,
                        "filled_quantity": decision.filled_quantity,
                        "price": decision.price,
                        "cash_after": decision.cash_after,
                        "position_after": decision.position_after,
                    }
                )
                decision_made = True
                obs.update_trace(
                    output={"action": decision.action, "regime": regime, "reasoning": reasoning}
                )
            else:
                result = _dispatch_tool(name, args, state, tracker)
                result_content = json.dumps(result)
                _log(state, {"type": "tool_call", "name": name, "args": args, "result_preview": _preview(result_content)})

            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_content})

        if decision_made:
            break

    if not decision_made:
        forced = tracker.record_sleep(
            symbol, "Max reasoning iterations reached without a finalized decision; defaulting to sleep."
        )
        obs.update_trace(output={"action": forced.action, "regime": "unknown", "reasoning": forced.reasoning})
        _log(
            state,
            {
                "type": "decision",
                "action": forced.action,
                "status": forced.status,
                "price": forced.price,
                "quantity": forced.filled_quantity,
                "reasoning": forced.reasoning,
                "regime": "unknown",
            },
        )


def _wait_for_next_cycle(state: "AppState", stop_event: threading.Event, cycle_sec: int) -> None:
    """Block until the next cycle is actually due.

    With no active alert, this is a plain `cycle_sec` timer (woken early only
    by fresh news). With an active alert, the fixed timer is disabled --
    the agent committed to "nothing changes until price reaches that level or
    news arrives", so it should wait indefinitely for `state.agent_wake_event`
    rather than also waking on the next scheduled tick. The price/news stream
    threads set that event directly the moment a price alert condition is met
    or fresh news arrives -- never on a timer just to check state.
    """
    state.agent_wake_event.clear()
    state.agent_wake_reason = None
    if stop_event.is_set():
        return

    # The stream only signals on the *next* tick, so an alert level that's
    # already satisfied by the current price (the instant it's set) would
    # otherwise wait for a tick that may not come. Catch that once, up front.
    alerts = state.price_alerts
    if alerts:
        with state.lock:
            price = state.last_price
        if price is not None:
            hit = next((a for a in alerts if alert_triggered(price, a)), None)
            if hit is not None:
                state.price_alerts = []
                _log(
                    state,
                    {"type": "status", "text": f"Price alert hit at {price} ({hit['condition']} {hit['price']}); waking early."},
                )
                return

    # An active alert means the agent should sleep until that condition
    # fires or news arrives -- not get woken by the regular cycle timer too.
    timeout = None if alerts else cycle_sec
    woke_early = state.agent_wake_event.wait(timeout=timeout)
    if stop_event.is_set():
        return
    if woke_early and state.agent_wake_reason:
        _log(state, {"type": "status", "text": f"{state.agent_wake_reason} Waking early."})
    state.agent_wake_event.clear()
    state.agent_wake_reason = None


def _agent_loop(
    state: "AppState",
    tracker: "DecisionTracker",
    symbol: str,
    provider: str,
    api_key: str,
    model: str,
    cycle_sec: int,
    stop_event: threading.Event,
    personality: str = DEFAULT_PERSONALITY,
) -> None:
    client = get_agent_client(provider, api_key)
    while not stop_event.is_set():
        try:
            run_agent_cycle(client, model, symbol, state, tracker, personality=personality)
        except Exception as exc:
            _log(state, {"type": "error", "text": f"Agent cycle failed: {exc}"})
        _wait_for_next_cycle(state, stop_event, cycle_sec)
    state.agent_running = False
    _log(state, {"type": "status", "text": "Agent stopped"})


def launch_agent(
    state: "AppState",
    tracker: "DecisionTracker",
    symbol: str,
    api_key: str,
    provider: str = "gemini",
    model: "str | None" = None,
    cycle_sec: int = 60,
    personality: str = DEFAULT_PERSONALITY,
) -> None:
    """Stop any running agent for this state, then start a new background cycle loop."""
    model = model or DEFAULT_AGENT_MODELS[provider]
    stop_agent(state)
    stop_event = threading.Event()
    state.agent_stop_event = stop_event
    state.agent_running = True
    threading.Thread(
        target=_agent_loop,
        args=(state, tracker, symbol, provider, api_key, model, cycle_sec, stop_event, personality),
        daemon=True,
    ).start()


def stop_agent(state: "AppState") -> None:
    if state.agent_stop_event:
        state.agent_stop_event.set()
    state.agent_running = False
    # Interrupt a blocked _wait_for_next_cycle immediately instead of letting
    # it sit until the timeout expires.
    state.agent_wake_event.set()
    # Push any buffered traces from the cycle(s) that just ran to Langfuse
    # before the background flusher would otherwise get to them.
    obs.flush()
