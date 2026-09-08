"""What every rule-based (non-LLM) agent does the same way.

Four state machines live in `apple_trader` and `apple_trader2`, and they trade
on four unrelated ideas: a momentum-persistence probability, a day-range
forecast, a delta-momentum regressor, and a user-written rule list. What they
do *around* the idea is identical, and was written out four times -- the same
pre-flight guards, the same position sizing, the same order and log calls, the
same clock-aligned loop, the same launcher.

That shared half is here. The split is deliberate and narrow: this module owns
nothing about *when to trade*. It never inspects a bar, a probability or a
level. A subclass that overrode nothing would place no orders at all, because
`run_cycle` is not implemented here -- the four cycles read different frames,
roll their sessions on different clocks and ask different questions, and one
template method over them would have been an abstraction over a coincidence.

What is shared, and why each piece is
-------------------------------------
* **the pre-flight**. An agent whose symbol is not streamed, or a session that
  is closed, is the same non-event for all four; both return before anything
  reads a bar.
* **the flatten window**. Intraday features do not survive the overnight gap,
  so every strategy here closes before the bell. It is one rule and one
  setting.
* **buy / sell / log**. The paper ledger is the same ledger. All four size a
  position the same way, place the same market order through the same
  `DecisionTracker`, and write the same decision entry to the same log -- the
  differences were a price to read, a sentence to log, and whether the position
  record carries a trailing peak.
* **the loop**. One cycle per closed bar, aligned to the clock rather than
  sleeping a flat interval from wherever the last cycle finished, so scoring
  that takes ten seconds does not walk the cadence forward.
"""
from __future__ import annotations

import math
import threading
from typing import TYPE_CHECKING

from . import market_hours
from . import observability as obs
from . import scoring
from .clock import now as _now
from .config import APPLE_TRADER_BAR_LAG_SEC, APPLE_TRADER_CYCLE_SEC
from .state import append_agent_log as _log

if TYPE_CHECKING:
    from .decisions import DecisionTracker
    from .state import AppState, SymbolState  # noqa: F401  (quoted annotation)


# The outcome tags `run_cycle` returns. SimLab counts them and the live loop
# ignores them, but naming them keeps the four implementations answering with
# the same vocabulary.
BOUGHT = "bought"
SOLD = "sold"
HOLD = "hold"
WARMING_UP = "warming_up"
CLOSED = "closed"
NO_DATA = "no_data"


def order_quantity(cash: float, price: float, position_pct: float) -> float:
    """Shares that `position_pct` of `cash` buys at `price`, rounded down.

    How much to deploy is a property of the agent, not of the rule that decided
    to deploy it, so every strategy sizes through here.
    """
    if price <= 0:
        return 0.0
    budget = cash * position_pct / 100.0
    return math.floor(budget / price * 1e4) / 1e4


class BaseTrader:
    """The half of a rule agent that is not its rules.

    Subclasses implement `run_cycle` and `_entry_reasoning`, and call the
    helpers here to act on what they decide.
    """

    #: Shown in the market-closed line, so the log names which agent is idle.
    AGENT_NAME = "Apple Trader"

    #: Leads the "cash buys nothing" line -- the four describe their own
    #: trigger there ("Entry signal confirmed", "Buy level touched").
    ENTRY_TRIGGER_TEXT = "Entry signal confirmed"

    #: Whether an open position tracks the highest price seen since entry.
    #: Only the momentum rules trail a stop; the others exit on a level, a
    #: forecast or a fixed stop measured from the entry price itself.
    TRAILS_PEAK = False

    def __init__(self, config) -> None:
        self.config = config
        # Read off the config once: every log line, order and state lookup is
        # about this one symbol, and the config is not re-read mid-run.
        self.ticker = config.ticker
        # The open long, or None while flat.
        self.entry: "dict | None" = None
        # Timestamp of the last bar acted on, so a cycle that re-reads the same
        # bar (slow fetch, market data lag) cannot act on it twice.
        self.last_bar_ts = None
        # Why today cannot be traded, when the reason is permanent for the
        # session. Kept so the failure is logged once rather than every minute
        # for six and a half hours.
        self.blocked: "dict | None" = None

    # --- pre-flight -------------------------------------------------------

    def preflight(self, state: "AppState") -> "tuple[SymbolState | None, str | None]":
        """The symbol's state, or (None, outcome-tag) when the cycle cannot run.

        Two ways to have nothing to do, and both are reported rather than
        raised: a symbol that is not streamed (a misconfiguration -- logged as
        an error) and a closed session (routine -- logged as status).
        """
        sym_state = state.sym(self.ticker)
        if sym_state is None:
            _log(
                state,
                {
                    "type": "error",
                    "text": f"{self.ticker} is not being streamed; nothing to trade.",
                },
            )
            return None, NO_DATA

        if not market_hours.is_market_open():
            _log(
                state,
                {
                    "type": "status",
                    "text": f"Market closed -- {self.AGENT_NAME} is not watching bars.",
                },
            )
            return None, CLOSED

        return sym_state, None

    def closing_soon(self) -> bool:
        """Whether the flatten-before-close rule is already in force."""
        to_close = market_hours.seconds_to_close()
        return to_close is not None and to_close <= self.config.flatten_before_close_min * 60

    # --- the ledger -------------------------------------------------------

    def buy(
        self,
        state: "AppState",
        tracker: "DecisionTracker",
        price: float,
        reasoning: str,
        log_extra: "dict | None" = None,
    ) -> bool:
        """Deploy `position_pct` of the cash balance; False if it buys nothing."""
        cash = tracker.snapshot()["cash"]
        quantity = order_quantity(cash, price, self.config.position_pct)
        if quantity <= 0:
            _log(
                state,
                {
                    "type": "status",
                    "text": (
                        f"{self.ENTRY_TRIGGER_TEXT} but ${cash:,.2f} cash buys no "
                        f"{self.ticker}."
                    ),
                },
            )
            return False

        decision = tracker.record_trade(
            self.ticker, "buy", quantity, reasoning,
            state.api_key, state.api_secret, state.feed,
        )
        self.log_decision(state, decision, log_extra)
        if decision.status == "filled":
            self.entry = {"price": decision.price, "bars": 0}
            if self.TRAILS_PEAK:
                # The peak starts at the fill, not at the triggering bar's
                # high: the stop trails the highest price seen since the
                # position existed, and that high happened before it did.
                self.entry["peak"] = decision.price
        return decision.status == "filled"

    def sell(
        self,
        state: "AppState",
        tracker: "DecisionTracker",
        quantity: float,
        reasoning: str,
        log_extra: "dict | None" = None,
    ) -> None:
        decision = tracker.record_trade(
            self.ticker, "sell", quantity, reasoning,
            state.api_key, state.api_secret, state.feed,
        )
        self.log_decision(state, decision, log_extra)
        if decision.status == "filled":
            self.entry = None

    def log_decision(
        self, state: "AppState", decision, extra: "dict | None" = None
    ) -> None:
        _log(
            state,
            {
                "type": "decision",
                "action": decision.action,
                "symbol": decision.symbol,
                "status": decision.status,
                "price": decision.price,
                "quantity": decision.filled_quantity,
                "reasoning": decision.reasoning,
                **(extra or {}),
            },
        )


# --- the loop ---------------------------------------------------------------

def seconds_to_next_bar(
    cycle_sec: int, lag: float = APPLE_TRADER_BAR_LAG_SEC
) -> float:
    """Seconds to wait so the next cycle lands just after a bar closes.

    Aligning to the clock (rather than sleeping a flat `cycle_sec` from wherever
    the last cycle finished) keeps one cycle per closed bar however long the
    scoring itself took.
    """
    ts = _now().timestamp()
    return (math.floor(ts / cycle_sec) + 1) * cycle_sec + lag - ts


def run_loop(
    state: "AppState",
    tracker: "DecisionTracker",
    run_cycle,
    stop_event: threading.Event,
    cycle_sec: int = APPLE_TRADER_CYCLE_SEC,
    agent_name: str = "Apple Trader",
) -> None:
    """Call `run_cycle()` once per closed bar until stopped, then wind down.

    A cycle that raises is logged and the loop continues: one bad bar (a fetch
    that failed, a frame that came back short) is not a reason to stop trading
    for the day, and a stopped agent that still looks running is worse than a
    logged error.
    """
    while not stop_event.is_set():
        try:
            run_cycle()
        except Exception as exc:
            _log(state, {"type": "error", "text": f"{agent_name} cycle failed: {exc}"})
        scoring.maybe_score_day(state, tracker)
        stop_event.wait(seconds_to_next_bar(cycle_sec))

    end_session(state, tracker, f"{agent_name} stopped")


def end_session(state: "AppState", tracker: "DecisionTracker", text: "str | None") -> None:
    """Close out a run: score the session, clear the running flag, say so.

    Reached both from the end of the loop and from every start-up refusal
    before it, which is why it is a function rather than the loop's tail: an
    agent that never started still has a scoring session open.
    """
    scoring.end_session(state, tracker)
    state.agent_running = False
    if text:
        _log(state, {"type": "status", "text": text})
    obs.flush()


def launch(
    state: "AppState",
    tracker: "DecisionTracker",
    agent_key: str,
    ticker: str,
    target,
    args: tuple,
    stop_agent,
) -> None:
    """Stop whatever agent is running for this state, then start `target`.

    `stop_agent` is passed in rather than imported to keep this module off
    `agent.py`, which imports enough of the app that a rule agent should not
    have to.
    """
    stop_agent(state)
    stop_event = threading.Event()
    state.agent_stop_event = stop_event
    state.agent_running = True
    scoring.begin_session(state, agent_key, [ticker])
    threading.Thread(
        target=target, args=(*args, stop_event), daemon=True
    ).start()
