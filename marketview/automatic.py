"""
Automatic orchestrator agent.

This is a meta-agent that sits above the individual strategy agents in
`marketview.agent`. Each round it runs a *regime-detection* cycle -- reading the
same analysis tools the strategies use (daily trend, broad-market backdrop,
intraday momentum, volume, VWAP/ADX range read, opening range, order blocks,
options walls, news) -- and finishes by calling `select_strategy` to activate the
single strategy best suited to current conditions.

Once a strategy is activated, the orchestrator goes to sleep: it simply hands the
loop to `run_agent_cycle(..., under_automatic=True)`, which runs that strategy's
normal observe-and-decide cycles. The strategy trades on its own until it decides
its edge has faded and calls `stand_down` (see `AUTOMATIC_MODE_ADDENDUM` in
`agent.py`). That wakes the orchestrator, which re-assesses the regime and may
activate a different strategy.

Lifecycle integrates with the existing controls: `launch_automatic` uses the same
`agent_stop_event` / `agent_running` / `agent_wake_event` plumbing as
`launch_agent`, so `stop_agent` stops it too.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from typing import Any

from . import agent
from . import market_hours
from . import observability as obs
from .agent import (
    AGENT_PERSONALITIES,
    PREMARKET_PERSONALITY,
    _TOOL_ANALYZE_DAILY_TREND,
    _TOOL_ANALYZE_INTRADAY_MOMENTUM,
    _TOOL_ANALYZE_MARKET,
    _TOOL_ANALYZE_OPENING_RANGE,
    _TOOL_ANALYZE_ORDER_BLOCKS,
    _TOOL_ANALYZE_VOLUME,
    _TOOL_ANALYZE_VWAP_BANDS,
    _TOOL_GET_NEWS,
    _TOOL_GET_PUT_CALL_WALLS,
    _TOOL_GET_QUOTE,
    _dispatch_tool,
    _log,
    _preview,
    _reject,
    _wait_for_next_cycle,
    run_agent_cycle,
    run_premarket_session,
)
from .config import AGENT_CYCLE_SEC, AGENT_MAX_TOOL_ITERS
from .decisions import DecisionTracker
from .llm import DEFAULT_AGENT_MODELS, get_agent_client
from .state import AppState

AUTOMATIC_KEY = "automatic"
AUTOMATIC_LABEL = "Automatic (regime-adaptive orchestrator)"
AUTOMATIC_AVATAR = "Multiavatar-18fd00dfa76e2785b7.png"

# Strategies the regime cycle can choose between -- every tradeable intraday
# personality. The Premarket Analyst is excluded: it is not a regime call, the
# orchestrator activates it deterministically whenever the session hasn't
# started (see `_automatic_loop`).
SELECTABLE_STRATEGIES: list[str] = [
    key for key in AGENT_PERSONALITIES if key != PREMARKET_PERSONALITY
]

# Regime vocabulary the orchestrator classifies into. Free-text reasoning carries
# the nuance; this enum just anchors the headline read.
REGIMES: list[str] = [
    "bullish_trend",
    "bearish_trend",
    "ranging",
    "volatile",
    "breakout_pending",
    "quiet",
]


AUTOMATIC_SYSTEM_PROMPT = f"""\
You are the Automatic orchestrator for a single equity ticker, operating in a \
paper-trading sandbox. You do NOT place trades yourself. Your one job each round \
is to read the current market regime and activate the ONE strategy agent best \
suited to it. That agent then trades on its own until its edge fades, at which \
point control returns to you and you re-assess.

Work through this every round, citing the actual numbers the tools return \
(trend strength, RSI, ATR, ADX, relative volume, support/resistance, VIX), not \
just their labels:

1. READ THE TICKER'S OWN STRUCTURE. Call analyze_daily_trend (medium-term \
regime: bullish/bearish/neutral, MA alignment, RSI, support/resistance) and \
analyze_order_blocks (institutional demand/supply zones at/below price).

2. READ THE BROAD-MARKET BACKDROP. Call analyze_market for the VIX level/trend, \
term structure, and the S&P's trend and drawdown -- a risk-off backdrop argues \
for more defensive/selective strategies and smaller risk.

3. READ TODAY'S INTRADAY CHARACTER. Call analyze_intraday_momentum (higher-highs \
vs lower-lows, VWAP position, ATR), analyze_volume (relative volume -- is there \
real participation?), analyze_vwap_bands (the ADX read is the key range-vs-trend \
gate: ADX below 20 = ranging, 25+ = trending), and analyze_opening_range (is an \
opening-range break setting up?). Optionally get_put_call_walls and get_news for \
positioning and catalysts.

4. MATCH THE REGIME TO A STRATEGY. Pick exactly one:
   - momentum -> a fresh, news-driven directional move ALREADY in progress on \
clearly elevated relative volume (a 5-20% gap with a catalyst). Best early in a \
strong, high-participation move.
   - breakout -> price is coiled against a clear opening range / level and a \
volume-backed break looks imminent or just happened. Best when a level is being \
tested with rising volume but no trend has resolved yet.
   - reversal -> a confirmed RANGE (ADX below 20, no catalyst, large-cap quiet \
tape) where price is stretched from VWAP. Best in the quiet middle of the \
session with no trend. Do NOT pick this when ADX shows a real trend.
   - smart_money -> price is returning to a higher-timeframe bullish demand \
order block in a non-bearish regime -- the highest-edge, most all-conditions \
setup when such a zone exists at/below price. When none of the above fits \
cleanly, default to momentum as the broadest-purpose intraday read.

5. FINALIZE. Call select_strategy exactly once with: the chosen strategy, the \
headline regime (one of: {", ".join(REGIMES)}), and reasoning that ties the \
specific numbers you read to why this strategy fits NOW and the others don't. \
Do not call select_strategy more than once, and do not stop without calling it.

You will be re-invoked when the activated strategy stands down (it judged its \
edge gone) -- so prefer the strategy that fits CURRENT conditions over hedging; \
if conditions change, the strategy will hand control back to you.
"""

_TOOL_SELECT_STRATEGY = {
    "type": "function",
    "function": {
        "name": "select_strategy",
        "description": (
            "Finalize this orchestration round by activating exactly one strategy "
            "agent to trade the current regime. Must be called exactly once, after "
            "the regime analysis is complete."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "strategy": {
                    "type": "string",
                    "enum": SELECTABLE_STRATEGIES,
                    "description": "Which strategy agent to activate for current conditions.",
                },
                "regime": {
                    "type": "string",
                    "enum": REGIMES,
                    "description": "The headline market regime you classified.",
                },
                "reasoning": {
                    "type": "string",
                    "description": (
                        "Why this strategy fits the current regime and the others don't -- "
                        "reference the actual trend/ADX/volume/VIX numbers you read."
                    ),
                },
            },
            "required": ["strategy", "reasoning"],
        },
    },
}

# Read-only analysis tools the orchestrator uses to classify the regime, plus the
# terminal select_strategy. No trading tools -- the orchestrator never trades.
REGIME_TOOLS: list[dict] = [
    _TOOL_GET_QUOTE,
    _TOOL_ANALYZE_DAILY_TREND,
    _TOOL_ANALYZE_MARKET,
    _TOOL_ANALYZE_INTRADAY_MOMENTUM,
    _TOOL_ANALYZE_VOLUME,
    _TOOL_ANALYZE_VWAP_BANDS,
    _TOOL_ANALYZE_OPENING_RANGE,
    _TOOL_ANALYZE_ORDER_BLOCKS,
    _TOOL_GET_PUT_CALL_WALLS,
    _TOOL_GET_NEWS,
    _TOOL_SELECT_STRATEGY,
]


def _strategy_label(key: str) -> str:
    entry = AGENT_PERSONALITIES.get(key)
    return entry["label"] if entry else key


@obs.observe(name="regime-cycle")
def run_regime_cycle(
    client: Any,
    model: str,
    symbol: str,
    state: AppState,
    tracker: DecisionTracker,
    max_iters: int = AGENT_MAX_TOOL_ITERS,
) -> "dict | None":
    """Run one regime-assessment round. Returns the selection dict
    {"strategy", "regime", "reasoning"} the orchestrator should activate, or None
    if the model failed to produce a valid selection."""
    obs.update_trace(
        name=f"regime-cycle:{symbol}", input=symbol, metadata={"model": model, "symbol": symbol}
    )
    messages: list[dict] = [
        {"role": "system", "content": AUTOMATIC_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Ticker: {symbol}. Assess the current market regime and finish by "
                "calling select_strategy with the best-fitting strategy."
            ),
        },
    ]
    _log(state, {"type": "cycle_start", "text": f"Automatic: assessing regime for {symbol}"})

    selection: "dict | None" = None
    for _ in range(max_iters):
        try:
            response = client.chat.completions.create(
                model=model, messages=messages, tools=REGIME_TOOLS, tool_choice="auto"
            )
        except Exception as exc:
            _log(state, {"type": "error", "text": f"Regime LLM call failed: {exc}"})
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
                extra_content = getattr(tc, "extra_content", None)
                if extra_content:
                    call["extra_content"] = extra_content
                calls.append(call)
            assistant_msg["tool_calls"] = calls
        messages.append(assistant_msg)

        if not tool_calls:
            messages.append(
                {"role": "user", "content": "Please finalize by calling select_strategy now."}
            )
            continue

        for tc in tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            if name == "select_strategy":
                strategy = args.get("strategy", "")
                regime = args.get("regime", "unknown")
                reasoning = args.get("reasoning", "")
                if strategy not in SELECTABLE_STRATEGIES:
                    _reject(
                        messages,
                        tc.id,
                        "strategy must be one of: "
                        f"{', '.join(SELECTABLE_STRATEGIES)}. Call select_strategy again "
                        "with a valid strategy.",
                    )
                    continue
                selection = {"strategy": strategy, "regime": regime, "reasoning": reasoning}
                _log(
                    state,
                    {
                        "type": "regime_select",
                        "strategy": strategy,
                        "label": _strategy_label(strategy),
                        "regime": regime,
                        "reasoning": reasoning,
                    },
                )
                obs.update_trace(output={"strategy": strategy, "regime": regime, "reasoning": reasoning})
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": json.dumps({"status": "activated"})}
                )
                break
            else:
                result = _dispatch_tool(name, args, state, tracker)
                result_content = json.dumps(result)
                _log(state, {"type": "tool_call", "name": name, "args": args, "result_preview": _preview(result_content)})
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_content})

        if selection is not None:
            break

    return selection


def _automatic_loop(
    state: AppState,
    tracker: DecisionTracker,
    symbol: str,
    provider: str,
    api_key: str,
    model: str,
    cycle_sec: int,
    stop_event: threading.Event,
) -> None:
    client = get_agent_client(provider, api_key)
    while not stop_event.is_set():
        # 0. Before the session starts there is no intraday regime to read --
        #    the Premarket Analyst runs instead. It holds until ~2 minutes
        #    before the bell, arms opening tactics, and retires once one
        #    executes; only then does the normal regime loop take over.
        if not market_hours.is_market_open():
            state.automatic_active_strategy = PREMARKET_PERSONALITY
            state.automatic_regime = "premarket"
            state.automatic_reason = (
                "Session hasn't started -- the Premarket Analyst prepares opening tactics."
            )
            _log(
                state,
                {
                    "type": "status",
                    "text": (
                        "Automatic: session hasn't started; activating "
                        f"{_strategy_label(PREMARKET_PERSONALITY)}."
                    ),
                },
            )
            outcome = run_premarket_session(client, model, symbol, state, tracker, stop_event)
            state.automatic_active_strategy = None
            if stop_event.is_set():
                break
            _log(
                state,
                {
                    "type": "status",
                    "text": (
                        "Premarket analyst "
                        + ("executed its opening tactics" if outcome == "executed" else "finished")
                        + "; Automatic assessing the regime."
                    ),
                },
            )
            continue

        # 1. Assess the regime and pick a strategy.
        state.automatic_active_strategy = None
        selection = None
        try:
            selection = run_regime_cycle(client, model, symbol, state, tracker)
        except Exception as exc:
            _log(state, {"type": "error", "text": f"Regime assessment failed: {exc}"})

        if stop_event.is_set():
            break

        if not selection:
            _log(
                state,
                {"type": "status", "text": "Automatic: no strategy selected this round; retrying."},
            )
            _wait_for_next_cycle(state, stop_event, cycle_sec)
            continue

        strategy = selection["strategy"]
        state.automatic_active_strategy = strategy
        state.automatic_regime = selection.get("regime")
        state.automatic_reason = selection.get("reasoning")
        _log(
            state,
            {
                "type": "status",
                "text": (
                    f"Automatic activated {_strategy_label(strategy)} "
                    f"[{selection.get('regime')}]: {selection.get('reasoning')}"
                ),
            },
        )

        # 2. Hand control to the chosen strategy until it stands down (or we stop).
        #    The orchestrator is "asleep" for the duration of this inner loop.
        while not stop_event.is_set():
            try:
                status = run_agent_cycle(
                    client, model, symbol, state, tracker,
                    personality=strategy, under_automatic=True,
                )
            except Exception as exc:
                _log(state, {"type": "error", "text": f"Strategy cycle failed: {exc}"})
                status = "decided"

            if status == "stand_down":
                _log(
                    state,
                    {
                        "type": "status",
                        "text": (
                            f"{_strategy_label(strategy)} stood down; "
                            "Automatic re-assessing the regime."
                        ),
                    },
                )
                break
            if stop_event.is_set():
                break
            _wait_for_next_cycle(state, stop_event, cycle_sec)

    state.agent_running = False
    state.automatic_active_strategy = None
    _log(state, {"type": "status", "text": "Automatic orchestrator stopped"})
    obs.flush()


def launch_automatic(
    state: AppState,
    tracker: DecisionTracker,
    symbol: str,
    api_key: str,
    provider: str = "openai",
    model: "str | None" = None,
    cycle_sec: int = AGENT_CYCLE_SEC,
) -> None:
    """Stop any running agent for this state, then start the Automatic orchestrator
    loop in the background. Uses the same stop/wake plumbing as `launch_agent`, so
    `stop_agent` halts it too."""
    model = model or DEFAULT_AGENT_MODELS[provider]
    agent.stop_agent(state)
    stop_event = threading.Event()
    state.agent_stop_event = stop_event
    state.agent_running = True
    agent.start_tactics_executor(state, tracker)
    state.automatic_active_strategy = None
    state.automatic_regime = None
    state.automatic_reason = None
    threading.Thread(
        target=_automatic_loop,
        args=(state, tracker, symbol, provider, api_key, model, cycle_sec, stop_event),
        daemon=True,
    ).start()
