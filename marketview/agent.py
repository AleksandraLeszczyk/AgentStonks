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
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable

from .config import AGENT_ALERT_POLL_SEC, AGENT_MAX_TOOL_ITERS
from .llm import DEFAULT_AGENT_MODELS, get_agent_client

if TYPE_CHECKING:
    from .decisions import DecisionTracker
    from .state import AppState


AGENT_SYSTEM_PROMPT = """\
You are an autonomous trading research agent for a single equity ticker, \
operating in a paper-trading sandbox -- no real orders are ever placed, so \
reason as if real capital is on the line.

Work through this process every cycle:

1. ESTABLISH THE REGIME. Call get_daily_bars first. Use the medium-term \
trend (price relative to its recent range, direction over the last 20-60 \
days, and any acceleration/deceleration) to classify the regime as bullish, \
bearish, or neutral/choppy. Don't skip this step -- it determines which \
strategy applies below.

2. CHECK INTRADAY CONFIRMATION. Call get_intraday_bars and get_volume_stats. \
Look for whether short-term price action and volume confirm or contradict \
the regime (e.g. a bullish regime with breaking-down intraday price and weak \
volume is a warning sign, not a buy signal).

3. CHECK NEWS. Call get_news. Treat clearly negative news (or negative-for- \
symbol competitor news) as a reason to be more conservative even in a \
bullish regime, and vice versa.

4. CHOOSE A STRATEGY THAT FITS THE REGIME:
   - Bullish + confirming volume/momentum -> trend-following: look to buy on \
strength or on a shallow pullback.
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
account.

6. FINALIZE. Call submit_decision exactly once, with the regime you \
established, the action (buy/sell/sleep/alert), a quantity (omit or 0 for \
sleep/alert), and reasoning that ties together the regime, the strategy, \
and why this specific action follows from it. Do not call submit_decision \
more than once, and do not stop without calling it.

Be decisive but not reckless: sleep is a valid and often correct decision. \
If you'd otherwise sleep but there's a specific price level (or a bracket of \
two -- a downside level and an upside level, e.g. a stop-loss below and a \
breakout level above) that would change your mind before the next scheduled \
cycle, use action "alert" instead of "sleep" and set alert_low_price (wakes \
you when price falls to/below it) and/or alert_high_price (wakes you when \
price rises to/above it). Set just one for a single level, or both to watch \
a range from both sides in one cycle. This wakes you up early -- as soon as \
either level is crossed -- instead of waiting out the full fixed cycle \
interval blind to what happens in between. Use plain "sleep" when no \
specific level is worth watching.

Regardless of which action you choose, you will also be woken up early -- \
before the next scheduled cycle -- if fresh news for the ticker arrives \
while you're waiting. You don't need to do anything to enable this; it \
happens automatically so a sleep/alert decision is never blind to breaking \
news.
"""

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_quote",
            "description": "Get the latest streamed quote and trade price for the ticker.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_intraday_bars",
            "description": "Get recent intraday OHLCV bars for the ticker, most recent last.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Number of most recent bars to return (default 50, max 300).",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_daily_bars",
            "description": (
                "Get daily OHLCV bars (up to ~1 year) for the ticker, used to establish the "
                "medium-term trading regime (bullish/bearish/choppy)."
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
    },
    {
        "type": "function",
        "function": {
            "name": "get_volume_stats",
            "description": "Get recent trade volume statistics to gauge participation and momentum confirmation.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_news",
            "description": "Get recent news headlines/summaries for the ticker, with impact labels where available.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Max number of articles (default 10)."}
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_position",
            "description": "Get the current paper trading position size and cash balance.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
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
    },
]


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


def _tool_get_intraday_bars(state: "AppState", limit: object = None) -> dict:
    n = max(1, min(int(limit or 50), 300))
    with state.lock:
        bars = list(state.bars)[-n:]
    return {"bars": bars, "count": len(bars)}


def _tool_get_daily_bars(state: "AppState", limit: object = None) -> dict:
    n = max(1, min(int(limit or 60), 365))
    bars = list(state.daily_bars)[-n:]
    return {"bars": bars, "count": len(bars)}


def _tool_get_volume_stats(state: "AppState") -> dict:
    with state.lock:
        bars = list(state.bars)
    if not bars:
        return {"note": "no intraday bars available yet"}
    volumes = [b.get("v", 0) for b in bars]
    recent = volumes[-10:]
    prior = volumes[-20:-10] if len(volumes) >= 20 else volumes[:-10]
    recent_avg = sum(recent) / len(recent) if recent else 0
    prior_avg = sum(prior) / len(prior) if prior else 0
    trend = "increasing" if recent_avg > prior_avg else "decreasing" if recent_avg < prior_avg else "flat"
    return {
        "bar_count": len(volumes),
        "total_volume": sum(volumes),
        "recent_10bar_avg_volume": recent_avg,
        "prior_10bar_avg_volume": prior_avg,
        "volume_trend": trend,
    }


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


def _tool_get_position(tracker: "DecisionTracker") -> dict:
    snap = tracker.snapshot()
    return {"cash": snap["cash"], "position": snap["position"], "decisions_so_far": len(snap["decisions"])}


_DISPATCH: dict[str, Callable[[dict, "AppState", "DecisionTracker"], dict]] = {
    "get_quote": lambda args, state, tracker: _tool_get_quote(state),
    "get_intraday_bars": lambda args, state, tracker: _tool_get_intraday_bars(state, args.get("limit")),
    "get_daily_bars": lambda args, state, tracker: _tool_get_daily_bars(state, args.get("limit")),
    "get_volume_stats": lambda args, state, tracker: _tool_get_volume_stats(state),
    "get_news": lambda args, state, tracker: _tool_get_news(state, args.get("limit")),
    "get_position": lambda args, state, tracker: _tool_get_position(tracker),
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


def run_agent_cycle(
    client: Any,
    model: str,
    symbol: str,
    state: "AppState",
    tracker: "DecisionTracker",
    max_iters: int = AGENT_MAX_TOOL_ITERS,
) -> None:
    """Run one analyze-then-decide cycle. Always ends with exactly one recorded decision."""
    state.price_alerts = []
    messages: list[dict] = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
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
                model=model, messages=messages, tools=TOOLS, tool_choice="auto"
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
                if action in ("buy", "sell") and quantity > 0:
                    decision = tracker.record_trade(
                        symbol, action, quantity, reasoning, state.api_key, state.api_secret, state.feed
                    )
                elif action == "alert" and alerts:
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


def _alert_triggered(price: float, alert: dict) -> bool:
    target = alert.get("price")
    condition = alert.get("condition")
    if target is None:
        return False
    if condition == "above":
        return price >= target
    if condition == "below":
        return price <= target
    return False


def _wait_for_next_cycle(state: "AppState", stop_event: threading.Event, cycle_sec: int) -> None:
    """Wait up to `cycle_sec`, but wake early if a price alert fires or fresh news arrives."""
    deadline = time.monotonic() + cycle_sec
    with state.lock:
        news_baseline = len(state.news)
    while not stop_event.is_set():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        alerts = state.price_alerts
        if alerts:
            with state.lock:
                price = state.last_price
            if price is not None:
                hit = next((a for a in alerts if _alert_triggered(price, a)), None)
                if hit is not None:
                    _log(
                        state,
                        {"type": "status", "text": f"Price alert hit at {price} ({hit['condition']} {hit['price']}); waking early."},
                    )
                    state.price_alerts = []
                    return
        with state.lock:
            news_count = len(state.news)
            new_articles = list(state.news)[news_baseline:] if news_count > news_baseline else []
        if new_articles:
            headline = new_articles[-1].get("headline", "")
            text = f"{len(new_articles)} new news item(s) for the ticker; waking early."
            if headline:
                text += f" Latest: {headline}"
            _log(state, {"type": "news_alert", "text": text})
            return
        stop_event.wait(min(AGENT_ALERT_POLL_SEC, remaining))


def _agent_loop(
    state: "AppState",
    tracker: "DecisionTracker",
    symbol: str,
    provider: str,
    api_key: str,
    model: str,
    cycle_sec: int,
    stop_event: threading.Event,
) -> None:
    client = get_agent_client(provider, api_key)
    while not stop_event.is_set():
        try:
            run_agent_cycle(client, model, symbol, state, tracker)
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
) -> None:
    """Stop any running agent for this state, then start a new background cycle loop."""
    model = model or DEFAULT_AGENT_MODELS[provider]
    stop_agent(state)
    stop_event = threading.Event()
    state.agent_stop_event = stop_event
    state.agent_running = True
    threading.Thread(
        target=_agent_loop,
        args=(state, tracker, symbol, provider, api_key, model, cycle_sec, stop_event),
        daemon=True,
    ).start()


def stop_agent(state: "AppState") -> None:
    if state.agent_stop_event:
        state.agent_stop_event.set()
    state.agent_running = False
