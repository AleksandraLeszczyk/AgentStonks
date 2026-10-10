"""
The Strategist (called Automatic until 2026-10-10).

This is a meta-agent that sits above the individual strategy agents in
`agent_stonks.agent`. Each round it runs a *regime-detection* cycle -- reading the
same analysis tools the strategies use (daily trend, broad-market backdrop,
intraday momentum, volume, VWAP/ADX range read, opening range, order blocks,
options walls, news, incoming corporate actions) -- and calls `select_strategy`
once per ticker to assign each one the strategy best suited to IT.

It starts each round from the day's research briefing (`agent_stonks.premarket`,
written when the stream starts): the catalyst, bias, shock flag and risks per
ticker are its first guess at which kind of day each ticker is having, which the
live tools then confirm or overturn. Before the bell it does nothing but wait for
it -- there is no intraday regime to read yet. It used to hand the pre-open to a
Premarket Analyst that armed opening tactics; that agent duplicated the
briefing's read and the Strategist's job, and was removed.

Assignment is per ticker because tickers are not in the same state as each
other. Measured live on 2026-09-11, the same round gave AAPL `reversal` (ADX
16.9, sitting on VWAP) and KO `momentum` (ADX 36.2, lower highs and lower lows)
under one shared `bullish_trend` market backdrop -- a single basket-wide pick
would have traded one of them with the wrong strategy. So each assignment weighs
two things: the broad market, which is shared and sets how much risk is sensible
at all, and that ticker's own structure, which decides which setup is present.

Tickers sharing a strategy are traded together by one agent -- `run_agent_cycle`
has always taken a basket -- so a round costs one cycle per distinct strategy,
not one per ticker. The groups run sequentially rather than in parallel: they
share a single cash balance, and two agents sizing against the same money at the
same moment would each ignore what the other is committing.

A strategy trades its tickers until it decides its edge has faded and calls
`stand_down` (see `STRATEGIST_MODE_ADDENDUM` in `agent_prompts.py`). Only the
tickers it held are then re-assessed; the rest keep trading under their own
assignments.

Lifecycle integrates with the existing controls: `launch_strategist` uses the same
`agent_stop_event` / `agent_running` / `agent_wake_event` plumbing as
`launch_agent`, so `stop_agent` stops it too.
"""
from __future__ import annotations

import json
import threading
import time
from typing import Any

from . import agent
from . import clock
from . import market_hours
from . import observability as obs
from . import scoring
from .agent import (
    AGENT_PERSONALITIES,
    _dispatch_tool,
    _log,
    _reject,
    _wait_for_next_cycle,
    breakout_preconditions,
    premarket_briefing_addendum,
    run_agent_cycle,
    schedule_retry,
    selectable_personalities,
)
# The Strategist's own regime-detection cycle is a personality in everything
# but name, so it assembles a tool list the same way the others do -- from the
# shared schemas rather than from a copy of them.
from .agent_tools import (
    _TOOL_ANALYZE_DAILY_TREND,
    _TOOL_ANALYZE_INTRADAY_MOMENTUM,
    _TOOL_ANALYZE_MARKET,
    _TOOL_ANALYZE_OPENING_RANGE,
    _TOOL_ANALYZE_ORDER_BLOCKS,
    _TOOL_ANALYZE_VOLUME,
    _TOOL_ANALYZE_VWAP_BANDS,
    _TOOL_GET_CORPORATE_ACTIONS,
    _TOOL_GET_NEWS,
    _TOOL_GET_PUT_CALL_WALLS,
    _TOOL_GET_QUOTE,
    _TOOL_GET_SESSION_CLOCK,
)
from .config import (
    AGENT_MAX_TOOL_ITERS,
    STRATEGIST_BRIEFING_POLL_SEC,
    STRATEGIST_BRIEFING_WAIT_SEC,
    STRATEGIST_OPEN_POLL_SEC,
)
from .decisions import DecisionTracker
from .llm import DEFAULT_AGENT_MODELS, get_agent_client
from .state import AppState

# The key keeps the Strategist's old name. It is what every stored run says
# it used -- `data/last_setup.json`, the day files' `run_spec` (which a restart
# launches again) and the scoring journal's `mode` -- so renaming it would
# orphan all of them for a word nobody sees: the screen shows the label.
STRATEGIST_KEY = "automatic"
STRATEGIST_LABEL = "Strategist (picks the agent for each ticker)"
STRATEGIST_AVATAR = "Multiavatar-18fd00dfa76e2785b7.png"

# Strategies the regime cycle can choose between -- every enabled tradeable
# intraday personality.
SELECTABLE_STRATEGIES: list[str] = list(selectable_personalities())

# Regime vocabulary the Strategist classifies into. Free-text reasoning carries
# the nuance; this enum just anchors the headline read.
REGIMES: list[str] = [
    "bullish_trend",
    "bearish_trend",
    "ranging",
    "volatile",
    "breakout_pending",
    "quiet",
]


STRATEGIST_SYSTEM_PROMPT = f"""\
You are the Strategist for a basket of equity tickers. You do NOT \
place trades yourself. Your job each round is to assign EACH TICKER the ONE \
strategy agent best suited to that ticker right now. Those agents then trade on \
their own until their edge fades, at which point control over those tickers \
returns to you and you re-assess them.

Different tickers can be in genuinely different states on the same day -- one \
gapping on news while another sits dead in a range -- so a single basket-wide \
pick is usually wrong for most of the basket. Assign per ticker. Tickers you \
give the same strategy are traded together by one agent, so matching two tickers \
that really are in the same state is good; forcing a third into it is not.

Every assignment is a judgement about TWO things at once:

  * the BROAD MARKET BACKDROP, which is shared by every ticker and sets how \
much risk is sensible at all;
  * that TICKER'S OWN STRUCTURE AND TAPE, which decides which specific setup is \
present in it.

A ticker in a clean bullish trend inside a risk-off, high-VIX market is not the \
same opportunity as the same chart in a calm market, and you should say so.

Work through this every round, citing the actual numbers the tools return \
(trend strength, RSI, ATR, ADX, relative volume, support/resistance, VIX), not \
just their labels:

0. START FROM THE RESEARCH BRIEFING, when one is reproduced at the end of these \
instructions. It is the day's research per ticker, usually written before the \
bell -- the overnight news and the catalyst behind any gap, a directional bias \
with its confidence, \
the levels and risks to watch, and whether today is a market-wide shock day -- \
and it is your first guess at which kind of day each ticker is having:
   - a real catalyst behind a confident bullish/bearish bias -> a momentum \
candidate, if the tape is actually moving on volume;
   - a neutral bias with no catalyst -> a range candidate (reversal), once the \
ADX confirms there is no trend;
   - a ticker pressing a level the briefing says to watch for a break -> a \
breakout candidate, if its opening range is measured and the break is setting \
up there;
   - a shock day, or a thesis outweighed by its risk list -> the most selective \
strategy that still fits, and only for an unmistakable setup.
   A guess is all it is: the briefing was written before the tape you are about \
to read, and the live tools decide. A ticker with no briefing starts from the \
tools alone.

1. READ THE BROAD-MARKET BACKDROP ONCE. Call analyze_market for the VIX \
level/trend, term structure, and the S&P's trend and drawdown -- a risk-off \
backdrop argues for more defensive/selective strategies and smaller risk across \
every ticker. Also call get_session_clock (which part of the session is this -- \
the opening window, the fakeout-prone midday dead zone, power hour?). Both are \
shared context: read them once, apply them to every ticker.

2. THEN READ EACH TICKER SEPARATELY. Every per-ticker tool takes a `symbol` \
argument, and you must actually read EVERY ticker you are assigning -- do not \
infer one ticker's state from another's. Per ticker call analyze_daily_trend \
(medium-term regime, MA alignment, RSI, support/resistance), \
analyze_intraday_momentum (higher-highs vs lower-lows, VWAP position, ATR), \
analyze_volume (rvol_pace -- is participation genuinely elevated for this time \
of day?), analyze_vwap_bands (the ADX read is the key range-vs-trend gate: ADX \
below 20 = ranging, 25+ = trending), analyze_opening_range (is an opening-range \
break setting up?) and analyze_order_blocks (institutional demand/supply zones \
at/below price). Optionally get_put_call_walls and get_news for positioning and \
catalysts (news the briefing already covered need not be re-read -- look for \
what broke since it was written), and get_corporate_actions for incoming corporate actions \
(ex-dividend dates, splits, mergers, spin-offs) -- these are scheduled \
mechanical catalysts that can distort a ticker's tape: an ex-dividend gap-down \
is not a bearish trend and a split resets every level, so don't let them \
masquerade as an organic regime.

3. MATCH EACH TICKER TO A STRATEGY. For each ticker pick exactly one:
   - momentum -> a fresh, news-driven directional move ALREADY in progress on \
clearly elevated relative volume (a 5-20% gap with a catalyst). Best early in a \
strong, high-participation move.
   - breakout -> price is coiled against a clear, MEASURED opening range and a \
volume-backed break looks imminent or just happened. Best when a level is being \
tested with rising volume but no trend has resolved yet. Only selectable when \
that ticker's analyze_opening_range returns a real range (a `note`-only result \
means there is no valid range -- do not pick breakout for it then) and \
get_session_clock shows a favorable window (never the 12:00-14:00 ET dead \
zone); selecting it otherwise is rejected and you must re-pick for that ticker.
   - reversal -> a confirmed RANGE (ADX below 20, no catalyst, large-cap quiet \
tape) where price is stretched from VWAP. Best in the quiet middle of the \
session with no trend. Do NOT pick this when ADX shows a real trend.
   - smart_money -> price is returning to a higher-timeframe bullish demand \
order block in a non-bearish regime -- the highest-edge, most all-conditions \
setup when such a zone exists at/below price. When none of the above fits \
cleanly for a ticker, default to momentum as the broadest-purpose intraday read.

4. FINALIZE. Call select_strategy ONCE PER TICKER -- every ticker you were given \
must get exactly one call, and you are not finished until all of them do. Each \
call takes: that ticker's `symbol`, the chosen `strategy`, that ticker's own \
`regime` (one of: {", ".join(REGIMES)}), the shared `market_regime` (the same \
backdrop value on every call -- it describes the market, not the ticker), and \
`reasoning` that cites the specific numbers you read for THAT ticker, says \
how the market backdrop shaped the choice, and -- when there was a briefing on \
it -- whether the live read confirmed its thesis or overturned it.

You will be re-invoked for a ticker when the strategy trading it stands down (it \
judged its edge gone) -- so prefer the strategy that fits CURRENT conditions over \
hedging; if conditions change, control over that ticker comes back to you.
"""

# The day's briefing as the Strategist reads it (see `premarket_briefing_addendum`,
# which renders it and works out its age). Framed for picking a strategy rather
# than for trading: what a briefing gets right for hours is the KIND of day --
# catalyst or none, bias, shock -- while its prices go stale with the first
# bars, and the trading framing would spend its words on the prices.
STRATEGIST_BRIEFING_ADDENDUM = """

--- RESEARCH BRIEFING FOR TODAY ({age}) ---
The day's research briefing on these tickers was written {age} \
({phase_note}). It is reproduced below as the starting point for step 0.

Read it for what lasts: the catalyst behind each ticker (or the absence of one), \
the directional bias and how confident the analyst was, whether it is a shock \
day, and the risks it named. Those say what kind of day a ticker is having and \
tend to hold for hours. Its exact price levels last least -- the tape may \
already have taken them out.

Where the briefing and your live tool output disagree, THE LIVE TOOLS WIN: a \
"bullish, high confidence" thesis on a ticker whose ADX reads 14 while it sits \
on VWAP is a range day, not a momentum day. Say so in your reasoning.

{briefings}
"""

_TOOL_SELECT_STRATEGY = {
    "type": "function",
    "function": {
        "name": "select_strategy",
        "description": (
            "Assign ONE ticker the strategy agent that should trade it. Call once "
            "per ticker in the basket, after analysing that ticker and the broad "
            "market. The round finishes when every ticker has an assignment."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "The ticker this assignment is for.",
                },
                "strategy": {
                    "type": "string",
                    "enum": SELECTABLE_STRATEGIES,
                    "description": "Which strategy agent should trade this ticker.",
                },
                "regime": {
                    "type": "string",
                    "enum": REGIMES,
                    "description": "THIS TICKER's own regime, not the market's.",
                },
                "market_regime": {
                    "type": "string",
                    "enum": REGIMES,
                    "description": (
                        "The shared broad-market backdrop. Describes the market, not "
                        "the ticker, so it is the SAME value on every call this round."
                    ),
                },
                "reasoning": {
                    "type": "string",
                    "description": (
                        "Why this strategy fits THIS ticker now and the others don't -- "
                        "cite the actual trend/ADX/volume numbers you read for it, "
                        "say how the market backdrop (VIX, S&P trend) shaped the choice, "
                        "and whether the live read confirmed or overturned the "
                        "briefing's thesis on it."
                    ),
                },
            },
            "required": ["symbol", "strategy", "reasoning"],
        },
    },
}

# Read-only analysis tools the Strategist uses to classify the regime, plus the
# terminal select_strategy. No trading tools -- the Strategist never trades.
REGIME_TOOLS: list[dict] = [
    _TOOL_GET_QUOTE,
    _TOOL_GET_SESSION_CLOCK,
    _TOOL_ANALYZE_DAILY_TREND,
    _TOOL_ANALYZE_MARKET,
    _TOOL_ANALYZE_INTRADAY_MOMENTUM,
    _TOOL_ANALYZE_VOLUME,
    _TOOL_ANALYZE_VWAP_BANDS,
    _TOOL_ANALYZE_OPENING_RANGE,
    _TOOL_ANALYZE_ORDER_BLOCKS,
    _TOOL_GET_PUT_CALL_WALLS,
    _TOOL_GET_NEWS,
    _TOOL_GET_CORPORATE_ACTIONS,
    _TOOL_SELECT_STRATEGY,
]


def _strategy_label(key: str) -> str:
    entry = AGENT_PERSONALITIES.get(key)
    return entry["label"] if entry else key


def briefing_addendum(state: AppState, symbols: list[str]) -> str:
    """The day's briefing on `symbols`, framed for the Strategist -- '' when
    there is none on them.

    Also '' when the briefing on hand was written on an earlier ET day. A
    Strategist left running overnight still holds the briefing from when its
    stream started, and that one describes a session already traded -- the
    trading agents get it with its age attached, but here it would anchor
    today's first assignments to yesterday's catalysts.
    """
    generated_at = state.premarket_generated_at
    if generated_at is not None:
        written = generated_at.astimezone(market_hours.MARKET_TZ).date()
        today = clock.now().astimezone(market_hours.MARKET_TZ).date()
        if written != today:
            return ""
    return premarket_briefing_addendum(
        state.premarket_briefings,
        symbols,
        generated_at=generated_at,
        phase=state.premarket_phase,
        template=STRATEGIST_BRIEFING_ADDENDUM,
    )


@obs.observe(name="regime-cycle")
def run_regime_cycle(
    client: Any,
    model: str,
    symbols: list[str],
    state: AppState,
    tracker: DecisionTracker,
    max_iters: int = AGENT_MAX_TOOL_ITERS,
) -> dict:
    """Assess the market and each ticker, and assign every ticker a strategy.

    Returns `{symbol: {"strategy", "regime", "market_regime", "reasoning"}}`,
    covering as many of `symbols` as the model managed to assign -- an empty
    dict when it produced nothing usable. Partial results are deliberately kept:
    a basket where three of four tickers were assigned should trade those three
    rather than discard the round.
    """
    symbols_label = ", ".join(symbols)
    obs.update_trace(
        name=f"regime-cycle:{symbols_label}",
        input=symbols_label,
        metadata={"model": model, "symbols": symbols_label},
    )
    briefing = briefing_addendum(state, symbols)
    messages: list[dict] = [
        {"role": "system", "content": STRATEGIST_SYSTEM_PROMPT + briefing},
        {
            "role": "user",
            "content": (
                f"Tickers: {symbols_label}. "
                + ("Start from the research briefing, then read " if briefing else "Read ")
                + "the broad market once, assess each ticker separately, and finish by "
                "calling select_strategy once for EVERY ticker listed."
            ),
        },
    ]
    _log(state, {
        "type": "cycle_start",
        "text": (
            f"Strategist: assessing regime for {symbols_label}"
            + (" from the pre-market briefing and the live tools" if briefing else "")
        ),
    })

    wanted = {s.upper() for s in symbols}
    assignments: dict[str, dict] = {}
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
            missing = sorted(wanted - set(assignments))
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Still unassigned: {', '.join(missing)}. Call select_strategy for "
                        "each of them now."
                    )
                    if missing
                    else "Please finalize by calling select_strategy now.",
                }
            )
            continue

        for tc in tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            if name == "select_strategy":
                symbol = str(args.get("symbol", "")).upper()
                strategy = args.get("strategy", "")
                regime = args.get("regime", "unknown")
                market_regime = args.get("market_regime", "unknown")
                reasoning = args.get("reasoning", "")
                if symbol not in wanted:
                    _reject(
                        messages,
                        tc.id,
                        f"symbol must be one of the tickers you were given: "
                        f"{', '.join(sorted(wanted))}. Call select_strategy again.",
                    )
                    continue
                if strategy not in SELECTABLE_STRATEGIES:
                    _reject(
                        messages,
                        tc.id,
                        "strategy must be one of: "
                        f"{', '.join(SELECTABLE_STRATEGIES)}. Call select_strategy again "
                        "with a valid strategy.",
                    )
                    continue
                if strategy == "breakout":
                    # Deterministic gate: the ORB specialist needs a real,
                    # measurable opening range and a favorable session window
                    # -- never deploy it into the midday dead zone or onto a
                    # session whose 09:30 window cannot be established. Checked
                    # for THIS ticker only: one symbol lacking a measurable
                    # opening range says nothing about another's.
                    blocked = breakout_preconditions(state, [symbol])
                    if blocked:
                        _reject(
                            messages,
                            tc.id,
                            f"{blocked}. Call select_strategy again for {symbol} with a "
                            "different strategy.",
                        )
                        continue
                assignments[symbol] = {
                    "strategy": strategy,
                    "regime": regime,
                    "market_regime": market_regime,
                    "reasoning": reasoning,
                }
                _log(
                    state,
                    {
                        "type": "regime_select",
                        "symbol": symbol,
                        "strategy": strategy,
                        "label": _strategy_label(strategy),
                        "regime": regime,
                        "market_regime": market_regime,
                        "reasoning": reasoning,
                    },
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps({"status": "assigned", "symbol": symbol}),
                    }
                )
            else:
                result = _dispatch_tool(name, args, state, tracker)
                result_content = json.dumps(result)
                _log(state, {"type": "tool_call", "name": name, "args": args, "result": result})
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_content})

        if wanted <= set(assignments):
            break

    if assignments:
        obs.update_trace(
            output={s: a["strategy"] for s, a in assignments.items()}
        )

    # The select_strategy reasoning must cite real tool numbers -- audit it
    # like any strategy cycle (see agent_stonks.scoring).
    scoring.record_cycle_grounding(state, messages, STRATEGIST_KEY)

    unassigned = sorted(wanted - set(assignments))
    if unassigned:
        _log(
            state,
            {
                "type": "status",
                "text": (
                    "Strategist: no strategy assigned for "
                    f"{', '.join(unassigned)}; they will be re-assessed next round."
                ),
            },
        )
    return assignments


def group_by_strategy(assignments: dict[str, dict]) -> list[tuple[str, list[str]]]:
    """Assignments collapsed into the units that actually run.

    Tickers sharing a strategy are traded by one agent over all of them, which
    is what `run_agent_cycle` already expects -- it has always taken a basket.
    So a round is one cycle per distinct strategy, not one per ticker: three
    tickers all reading as momentum cost one cycle, not three.

    Ordering is stable (first assignment wins) so the same conditions produce
    the same sequence of cycles rather than a set-iteration shuffle.
    """
    groups: dict[str, list[str]] = {}
    for symbol, entry in assignments.items():
        groups.setdefault(entry["strategy"], []).append(symbol)
    return [(strategy, syms) for strategy, syms in groups.items()]


def _publish_assignments(state: AppState, assignments: dict[str, dict]) -> None:
    """Mirror the live assignments onto the app state for the UI.

    `strategist_assignments` is the real answer now that each ticker has its own
    strategy. The three older single-valued fields are kept in step because the
    report, the avatar card and SimLab all read them; they hold the *dominant*
    assignment (the strategy covering the most tickers), which is the honest
    summary when one number has to stand for several.
    """
    state.strategist_assignments = dict(assignments)
    if not assignments:
        state.strategist_active_strategy = None
        state.strategist_regime = None
        state.strategist_reason = None
        return
    groups = group_by_strategy(assignments)
    strategy, syms = max(groups, key=lambda pair: len(pair[1]))
    entry = assignments[syms[0]]
    state.strategist_active_strategy = strategy
    state.strategist_regime = entry.get("market_regime") or entry.get("regime")
    if len(groups) == 1:
        state.strategist_reason = entry.get("reasoning")
    else:
        state.strategist_reason = "; ".join(
            f"{sym} → {_strategy_label(a['strategy'])}" for sym, a in assignments.items()
        )


def _wait_for_open(state: AppState, stop_event: threading.Event) -> bool:
    """Hold until the regular session is running. False when stopped first.

    Before the bell there is nothing for the Strategist to read -- no intraday
    trend, volume or opening range -- and so nothing to assign. The briefing is
    meanwhile being written (or already done) and is read at the open.
    """
    if market_hours.is_market_open():
        return True
    open_et = market_hours.next_market_open().astimezone(market_hours.MARKET_TZ)
    _log(
        state,
        {
            "type": "status",
            "text": (
                "Strategist: the session hasn't started; holding until the bell "
                f"({open_et.strftime('%Y-%m-%d %H:%M')} ET), then assigning from the "
                "pre-market briefing and the live tape."
            ),
        },
    )
    state.strategist_waiting = "the opening bell"
    try:
        while not stop_event.is_set():
            if market_hours.is_market_open():
                return True
            stop_event.wait(
                min(market_hours.seconds_until_next_open(), STRATEGIST_OPEN_POLL_SEC)
            )
        return False
    finally:
        state.strategist_waiting = None


def _await_briefing(state: AppState, symbols: list[str], stop_event: threading.Event) -> None:
    """Hold a round while the briefing on `symbols` is still being written.

    The briefing starts with the stream, on its own thread, at several seconds
    of LLM work per ticker -- so an agent started with the stream (mid-session,
    or by a restart) reaches its first assessment before it lands. Going ahead
    would make the first round, the one that sets the whole basket up, the one
    round that never saw the day's research. Bounded, because a stuck briefing
    must not keep the basket out of the market.
    """

    def still_writing() -> list[str]:
        pending = {s.upper() for s in (state.premarket_pending or [])}
        return [s for s in symbols if s.upper() in pending]

    waiting = still_writing()
    if not waiting:
        return
    _log(
        state,
        {
            "type": "status",
            "text": (
                f"Strategist: waiting for the pre-market briefing on {', '.join(waiting)} "
                "before assigning strategies."
            ),
        },
    )
    state.strategist_waiting = "the pre-market briefing"
    deadline = time.monotonic() + STRATEGIST_BRIEFING_WAIT_SEC
    try:
        while not stop_event.is_set():
            waiting = still_writing()
            if not waiting:
                return
            left = deadline - time.monotonic()
            if left <= 0:
                _log(
                    state,
                    {
                        "type": "status",
                        "text": (
                            f"Strategist: the briefing on {', '.join(waiting)} isn't ready "
                            f"after {STRATEGIST_BRIEFING_WAIT_SEC / 60:g} min; assessing "
                            "without it."
                        ),
                    },
                )
                return
            stop_event.wait(min(left, STRATEGIST_BRIEFING_POLL_SEC))
    finally:
        state.strategist_waiting = None


def _strategist_loop(
    state: AppState,
    tracker: DecisionTracker,
    symbols: list[str],
    provider: str,
    api_key: str,
    model: str,
    stop_event: threading.Event,
) -> None:
    client = get_agent_client(provider, api_key)
    # symbol -> {strategy, regime, market_regime, reasoning}. Survives rounds:
    # only tickers whose strategy stood down are removed and re-assessed.
    assignments: dict[str, dict] = {}
    # Whether this round follows a wake. A wake is not routed to the ticker it
    # was about (one shared wake event), so after one every group runs. The
    # round straight after a stand-down follows no wake: only the groups that
    # took a freshly assigned ticker run, the rest are still waiting on what
    # they armed.
    woken = True
    while not stop_event.is_set():
        # 0. Outside the session there is no regime to read: hold for the
        #    bell. A session that ended under assignments ends their scoring
        #    windows with it, and the next one is assessed from scratch.
        if not market_hours.is_market_open():
            if assignments:
                scoring.record_activation_end(state)
                assignments = {}
            _publish_assignments(state, assignments)
            if not _wait_for_open(state, stop_event):
                break
            woken = True
            continue

        # 1. Assess anything currently unassigned. On the first round that is
        #    the whole basket; later it is only the tickers whose strategy stood
        #    down, so a ticker that is trading well is left alone rather than
        #    being re-picked because a different ticker's edge faded.
        scoring.maybe_score_day(state, tracker)
        pending = [s for s in symbols if s.upper() not in assignments]
        fresh: dict[str, dict] = {}
        if pending:
            _await_briefing(state, pending, stop_event)
            if stop_event.is_set():
                break
            try:
                fresh = run_regime_cycle(client, model, pending, state, tracker)
            except Exception as exc:
                _log(state, {"type": "error", "text": f"Regime assessment failed: {exc}"})
                fresh = {}
            for symbol, entry in fresh.items():
                assignments[symbol] = entry
                # One scoring window per (strategy, ticker) assignment. Windows
                # for different tickers now overlap in time, so `scoring`
                # attributes decisions to them by symbol rather than by clock
                # alone -- see `record_activation_start`.
                scoring.record_activation_start(
                    state, entry["strategy"], entry.get("regime"), symbols=[symbol]
                )
                _log(
                    state,
                    {
                        "type": "status",
                        "text": (
                            f"Strategist assigned {symbol} to {_strategy_label(entry['strategy'])} "
                            f"[{entry.get('regime')} ticker / {entry.get('market_regime')} market]: "
                            f"{entry.get('reasoning')}"
                        ),
                    },
                )
            _publish_assignments(state, assignments)
            if len(fresh) < len(pending):
                # The unassigned are re-assessed on the next wake; if every
                # other ticker is asleep on far-away alerts that could be
                # hours, so a retry is scheduled for them.
                schedule_retry(state)

        if stop_event.is_set():
            break

        if not assignments:
            _log(
                state,
                {"type": "status", "text": "Strategist: no strategy selected this round; retrying."},
            )
            _wait_for_next_cycle(state, stop_event)
            woken = True
            continue

        # 2. One cycle per distinct strategy, over the tickers assigned to it.
        #    Sequential rather than parallel: the tickers share one cash
        #    balance, and two agents deciding how to spend it at the same
        #    moment would each size against money the other is already
        #    committing. Sequential cycles see each other's fills.
        stood_down = False
        for strategy, group_symbols in group_by_strategy(assignments):
            if stop_event.is_set():
                break
            if not woken and not any(s in fresh for s in group_symbols):
                continue
            try:
                status = run_agent_cycle(
                    client, model, group_symbols, state, tracker,
                    personality=strategy, under_strategist=True,
                )
            except Exception as exc:
                _log(state, {"type": "error", "text": f"Strategy cycle failed: {exc}"})
                schedule_retry(state)
                status = "decided"

            if status == "stand_down":
                _log(
                    state,
                    {
                        "type": "status",
                        "text": (
                            f"{_strategy_label(strategy)} stood down on "
                            f"{', '.join(group_symbols)}; the Strategist will re-assess "
                            "those tickers."
                        ),
                    },
                )
                for symbol in group_symbols:
                    assignments.pop(symbol, None)
                    scoring.record_activation_end(state, symbols=[symbol])
                _publish_assignments(state, assignments)
                stood_down = True

        if stop_event.is_set():
            break
        if stood_down:
            # Control over those tickers is back with the Strategist: re-assess
            # them now rather than after whatever the other groups armed.
            woken = False
            continue
        _wait_for_next_cycle(state, stop_event)
        woken = True

    scoring.end_session(state, tracker)
    state.agent_running = False
    _publish_assignments(state, {})
    _log(state, {"type": "status", "text": "Strategist stopped"})
    obs.flush()


def launch_strategist(
    state: AppState,
    tracker: DecisionTracker,
    symbols: list[str],
    api_key: str,
    provider: str = "openai",
    model: "str | None" = None,
) -> None:
    """Stop any running agent for this state, then start the Strategist loop
    (over the whole symbol basket) in the background. Uses the same stop/wake
    plumbing as `launch_agent`, so `stop_agent` halts it too."""
    model = model or DEFAULT_AGENT_MODELS[provider]
    agent.stop_agent(state)
    stop_event = threading.Event()
    state.agent_stop_event = stop_event
    state.agent_running = True
    scoring.begin_session(state, STRATEGIST_KEY, symbols)
    agent.start_tactics_executor(state, tracker)
    _publish_assignments(state, {})
    state.strategist_waiting = None
    threading.Thread(
        target=clock.inherit(_strategist_loop),
        args=(state, tracker, symbols, provider, api_key, model, stop_event),
        daemon=True,
    ).start()
