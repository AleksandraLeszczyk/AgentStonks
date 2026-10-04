"""Orchestra: Apple Trader's rules over several (ticker, model) pairs at once.

Apple Trader trades one symbol on one model. Orchestra is its own agent: it
runs one `DayRangeTrader` per pair -- AAPL on the day-range model and AAPL on
HighLow are two pairs -- over one shared ledger, and lets exactly one of them
hold a position at a time. The pairs race for it:

* **Racing.** Every pair forecasts its own session at 9:35 and rests its own
  levels, exactly as Apple Trader would on it alone. Each bar the pairs are
  read in Orchestra's order, and the first whose buy *fills* takes the
  position. A buy level touched is not enough, and neither is a limit order
  the price had already left: those leave the race open.
* **Following.** While a pair holds, it is the only one that may trade. The
  others are still read every bar -- their forecasts move with the session
  and their levels stay current on the chart -- but their entries are closed
  (`BaseTrader.entry_gate`), and a pair on the holder's own symbol sees the
  symbol as flat, so it never adopts a position that is not its own.
* **Reopening.** Once the holder's position is closed -- at the target, by the
  momentum take and its runner, or by the circuit breaker -- the race is open
  again, to every pair that is still armed. The holder's own rules decide
  whether *it* is: a breaker stand-down keeps that pair out for the day and
  leaves the rest racing. (The user's choice, 2026-10-02.)
* **Picking.** With a selection (`OrchestraConfig.selection`, on by default),
  Orchestra first narrows its pairs to the day's candidates, once, at 09:34 --
  four opening minutes in, one bar before the forecasts -- from the
  pre-market briefing, the earnings calendar and the opening gap
  (`agent_stonks.candidates`). The pairs left out sit the day out: they are
  not read and never forecast. The choice is published and saved with the
  session, so a restart keeps it rather than picking again.
* **Stopping.** A stop-out ends the whole run, as it ends an Apple Trader one:
  the live loop stops the agent and only ▶ Start re-arms it. A replay has no
  ▶ Start, so there every pair's entries stay closed for the rest of that
  session, while their levels are still read bar by bar.

Nothing about a pair's own rules changes. With one pair Orchestra *is* an
Apple Trader run, decision for decision (`tests/test_orchestra.py`), which is
what lets it share every tuned level and every stored SimLab record with it.
In the code a pair Orchestra is running is a "racer".

Where the racers keep their state
---------------------------------
Apple Trader keeps its form and its record in `AppState.apple_trader_config`
and `apple_trader_levels`. Orchestra gives each racer a `RacerSlot` instead,
keyed by `racer_key` ("INTC:dayrange"), in `AppState.orchestra_configs` and
`orchestra_levels`; Orchestra as a whole -- order, holder, the board -- is
`AppState.orchestra`. The chart of a symbol draws the racer holding it, else
the first racer on it (`model_overlays.live_trader_view`).
"""
from __future__ import annotations

import re
import threading
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Optional

from . import apple_models, candidates, event_days, market_hours, momentum_regime, rule_agent
from . import apple_trader as at
from .agent import stop_agent
from .apple_trader import AppleTraderConfig, TraderSlot
from .config import APPLE_TRADER_CYCLE_SEC, APPLE_TRADER_TUNED_LEVELS, LEVEL_UNIT_TOKENS
from .decisions import DecisionTracker
from .state import AppState, agent_log_tag
from .state import append_agent_log as _log

# The agent's identity: its personality key (also every SimLab run's), what
# the pickers call it and its face.
ORCHESTRA_KEY = "orchestra"
ORCHESTRA_LABEL = "Orchestra (rule-based, no LLM)"
ORCHESTRA_AVATAR = "Multiavatar-2d21d8b739482562eb.png"

# The fields every racer must share. Orchestra is one rule set over several
# instruments, not several rule sets: the pair (ticker, model) and the numbers
# tuned per pair -- the two distances and the circuit breaker -- are the racer's
# own, and everything else is Orchestra's. `level_source` follows the model.
PER_RACER_FIELDS = ("ticker", "model_key", "buy_k", "sell_k", "min_win_k", "level_source")


def racer_key(config: AppleTraderConfig) -> str:
    """A racer's identity: its symbol and its model. Stable for a whole run --
    a racer never switches model, since the model is half of what it is."""
    return f"{config.ticker}:{config.model_key}"


def split_key(key: str) -> "tuple[str, str]":
    ticker, _, model_key = str(key).partition(":")
    return ticker.upper(), model_key


# What a racer's model is called beside its symbol. The registry's labels are
# written for a picker ("Day-range forecast (TimeToChange3)") and are too long
# to repeat on every log line and board row.
SHORT_MODEL_NAMES = {
    "dayrange": "Day Range",
    "dayrange_intraday": "Day Range × IV",
    "highlow": "HighLow",
    "highlow2": "HighLow2",
}


def racer_label(config_or_key) -> str:
    """"INTC · Day Range": what the log, the board and the form call a racer."""
    if isinstance(config_or_key, AppleTraderConfig):
        ticker, model_key = config_or_key.ticker, config_or_key.model_key
    else:
        ticker, model_key = split_key(config_or_key)
    name = SHORT_MODEL_NAMES.get(model_key) or apple_models.get(model_key).label
    return f"{ticker} · {name}"


def all_pairs() -> "list[str]":
    """Every (ticker, model) pair Orchestra can be given, as racer keys: each
    symbol some model covers, with each model it has, in picker order."""
    return [
        f"{ticker}:{key}"
        for ticker in apple_models.tickers()
        for key in apple_models.keys_for(ticker)
    ]


def default_pairs() -> "list[str]":
    """The pairs Orchestra starts from: every pair SimLab's tuning has picked
    levels for (`config.APPLE_TRADER_TUNED_LEVELS`), in picker order. A pair
    nobody tuned is offered, not chosen."""
    tuned = {f"{ticker}:{model}" for model, ticker in APPLE_TRADER_TUNED_LEVELS}
    return [key for key in all_pairs() if key in tuned]


# --- the configuration --------------------------------------------------------


@dataclass
class OrchestraConfig:
    """The racers, in the order ties are broken in: when two buys fill on the
    same bar the earlier racer is read first and takes the position."""

    racers: "list[AppleTraderConfig]" = field(default_factory=list)
    # How the day's candidates are picked at 09:34, or None to race every pair
    # all day -- which is what every record made before the selection means.
    selection: "candidates.SelectionRules | None" = None

    def __post_init__(self) -> None:
        self.racers = [
            r if isinstance(r, AppleTraderConfig) else AppleTraderConfig(**r)
            for r in self.racers
        ]
        if isinstance(self.selection, dict):
            known = {f.name for f in fields(candidates.SelectionRules)}
            self.selection = candidates.SelectionRules(
                **{k: v for k, v in self.selection.items() if k in known}
            )
        if not self.racers:
            raise ValueError("Orchestra needs at least one (ticker, model) pair")
        keys = [racer_key(r) for r in self.racers]
        twice = sorted({k for k in keys if keys.count(k) > 1})
        if twice:
            raise ValueError(f"{', '.join(twice)} is in Orchestra more than once")
        first = _shared(self.racers[0])
        for racer in self.racers[1:]:
            if _shared(racer) != first:
                differ = sorted(k for k in first if first[k] != _shared(racer)[k])
                raise ValueError(
                    f"every pair shares Orchestra's rules, but {racer_key(racer)} differs "
                    f"from {keys[0]} in {', '.join(differ)}"
                )

    @property
    def keys(self) -> "list[str]":
        return [racer_key(r) for r in self.racers]

    @property
    def tickers(self) -> "list[str]":
        return list(dict.fromkeys(r.ticker for r in self.racers))


def _shared(config: AppleTraderConfig) -> dict:
    return {k: v for k, v in asdict(config).items() if k not in PER_RACER_FIELDS}


def build_orchestra_config(
    pairs: "list[str]",
    base: AppleTraderConfig,
    levels: "dict | None" = None,
    selection: "candidates.SelectionRules | None" = None,
) -> OrchestraConfig:
    """One racer per pair, each `base` with its own pair and the numbers tuned
    for it: `levels[key]` = `(buy_k, sell_k, min_win_k)` where given, the
    pair's shipped defaults otherwise. `selection` as `OrchestraConfig`'s."""
    levels = levels or {}
    racers = []
    for key in pairs:
        ticker, model_key = split_key(key)
        buy_k, sell_k = at.dayrange_levels(ticker, model_key)
        min_win_k = at.min_win_for(ticker)
        if key in levels:
            buy_k, sell_k, min_win_k = levels[key]
        racers.append(replace(
            base, ticker=ticker, model_key=model_key,
            buy_k=float(buy_k), sell_k=float(sell_k), min_win_k=float(min_win_k),
            level_source=apple_models.get(model_key).level_source,
        ))
    return OrchestraConfig(racers, selection=selection)


def orchestra_signature(race: OrchestraConfig) -> str:
    """Compact identity of an Orchestra setup, standing in for a model name in SimLab.

    The racers in order -- each its model, symbol, two distances and circuit
    breaker, since those are what the pair tunes -- then the shared rules once,
    read off the first racer's own signature with its per-pair part removed, so
    a rule signs here exactly as it does in an Apple Trader run.
    """
    first = race.racers[0]
    unit = LEVEL_UNIT_TOKENS[first.level_unit]
    pairs = ",".join(
        f"{r.model_key}_{r.ticker}@{r.buy_k:g}/{r.sell_k:g}{unit}"
        + (f"/mw{r.min_win_k:g}" if r.min_win_k else "")
        for r in race.racers
    )
    single = at.config_signature(first)
    rules = single[single.index("size="):-1]
    rules = re.sub(r",min_win=[^,)]*", "", rules)
    return f"orchestra[{pairs}]({rules}{candidates.rules_signature(race.selection)})"


# --- one racer's view of the shared state --------------------------------------


class RacerSlot(TraderSlot):
    """A racer's own form and record, keyed by its racer key (`TraderSlot`)."""

    def __init__(self, key: str) -> None:
        self.key = key

    def form(self, state: AppState) -> "AppleTraderConfig | None":
        return (getattr(state, "orchestra_configs", None) or {}).get(self.key)

    def levels(self, state: AppState) -> "dict | None":
        return (getattr(state, "orchestra_levels", None) or {}).get(self.key)

    def publish(self, state: AppState, record: dict) -> None:
        # Replaced rather than mutated, so a reader on another thread (the
        # chart, the autosave) never sees a dict change size under it.
        state.orchestra_levels = {**(getattr(state, "orchestra_levels", None) or {}), self.key: record}


class _RacerLedger:
    """The shared ledger as one racer sees it.

    The holder sees the ledger as it is. Every other racer sees no position at
    all -- a racer on the holder's own symbol would otherwise take that
    position for its own and manage it with its own stop. Fills are reported
    to Orchestra, which is how it learns who holds and when the position is
    closed. Everything else (cash, the snapshot) is the ledger's.
    """

    def __init__(self, race: "Orchestra", key: str, tracker: DecisionTracker) -> None:
        self._race = race
        self._key = key
        self._tracker = tracker

    def position_for(self, symbol: str) -> float:
        if self._race.holder != self._key:
            return 0.0
        return self._tracker.position_for(symbol)

    def record_trade(self, symbol, action, quantity, reasoning, *args, **kwargs):
        decision = self._tracker.record_trade(symbol, action, quantity, reasoning, *args, **kwargs)
        if decision.status == "filled":
            if action == "buy":
                self._race._took(self._key, decision)
            elif self._tracker.position_for(symbol) <= 0:
                self._race._released(self._key)
        return decision

    def __getattr__(self, name):
        return getattr(self._tracker, name)


# --- the coordinator ----------------------------------------------------------


@dataclass
class Racer:
    key: str
    label: str
    config: AppleTraderConfig
    trader: object
    bundle: dict
    outcome: Optional[str] = None


class Orchestra:
    """The coordinator: one trader per racer, one position between them.

    `run_cycle(state, tracker)` is the whole interface -- the live loop calls
    it once a closed bar, SimLab's rule day loop once a replayed one.
    """

    def __init__(
        self,
        racers: "list[Racer]",
        selection: "candidates.SelectionRules | None" = None,
        sources=None,
    ) -> None:
        if not racers:
            raise ValueError("Orchestra needs at least one racer")
        self.racers = racers
        self.by_key = {r.key: r for r in racers}
        # The 09:34 candidate selection, and where its briefings and earnings
        # calendar come from (`candidates.LiveSources`, or a replay's).
        self.selection = selection
        self.sources = sources or candidates.LIVE_SOURCES
        # The day the selection was made for, its rows, and the pairs it left
        # out -- each sits the day out, with the reason.
        self._selected_day = None
        self.selection_rows: "list[dict]" = []
        self.benched: "dict[str, str]" = {}
        # The racer holding the position, or None while the race is open.
        self.holder: "str | None" = None
        self.holder_since = None
        # Set by a stop-out: the live loop stops the agent on it; a replay
        # closes every racer's entries for the rest of `self._stopped_day`.
        self.halt: "str | None" = None
        self._stopped_day = None
        self._day = None
        self._adopted = False
        self._last_line = None
        self._event = None
        for racer in racers:
            racer.trader.slot = RacerSlot(racer.key)
            racer.trader.entry_gate = (lambda key=racer.key: self._gate(key))

    @property
    def keys(self) -> "list[str]":
        return [r.key for r in self.racers]

    # --- who holds -----------------------------------------------------

    def _gate(self, key: str) -> "str | None":
        if self._stopped_day is not None and self._stopped_day == self._day:
            return f"Orchestra stopped for the day when {self.halt}"
        if self.holder is None or self.holder == key:
            return None
        since = f", bought {self.holder_since:%H:%M}" if self.holder_since is not None else ""
        return f"Orchestra is following {self.by_key[self.holder].label}{since}"

    def _took(self, key: str, decision) -> None:
        if self.holder == key:
            return
        self.holder = key
        self.holder_since = at.clock.now().astimezone(market_hours.MARKET_TZ)
        self._event = ("took", key)

    def _released(self, key: str) -> None:
        if self.holder != key:
            return
        self.holder = None
        self.holder_since = None
        self._event = ("released", key)

    def _adopt(self, state: AppState, tracker: DecisionTracker) -> None:
        """Who holds when Orchestra starts onto a ledger that is not flat: the
        racer an earlier run today recorded as holding, while the ledger still
        has its symbol; else the first racer on a symbol the ledger holds."""
        self._adopted = True
        prior = getattr(state, "orchestra", None) or {}
        held = [r for r in self.racers if tracker.position_for(r.trader.ticker) > 0]
        if not held:
            return
        recorded = prior.get("holder")
        racer = next((r for r in held if r.key == recorded), held[0])
        self.holder = racer.key
        _log(state, {"type": "status", "text": (
            f"The ledger holds {tracker.position_for(racer.trader.ticker):g} "
            f"{racer.trader.ticker}; Orchestra follows {racer.label} until it is sold."
        )})

    # --- one cycle -----------------------------------------------------

    def run_cycle(self, state: AppState, tracker: DecisionTracker) -> str:
        today = at._dayrange().market_date()
        if self._day != today:
            # A new session: nobody holds overnight (every racer flattens
            # before the close), and a stop-out is only about its own day.
            self._day = today
            self.halt = None
            self._last_line = None
            self.benched = {}
            self.selection_rows = []
            # A trader keeps its stop-out flag across the night (nothing reads
            # it in a single replay); here it would stop the next session too.
            for racer in self.racers:
                racer.trader.halt = None
        if not market_hours.is_market_open():
            # Said once for Orchestra rather than once per racer.
            _log(state, {"type": "status", "text": (
                "Market closed -- Orchestra is not watching bars."
            )})
            return rule_agent.CLOSED
        if not self._adopted:
            self._adopt(state, tracker)
        if self.selection is not None and self._selected_day != today:
            if not self._restore_selection(state, today):
                if not self._selection_due(state):
                    # Before 09:34 there is nothing to read: no racer can
                    # forecast before 09:35, and which ones will is not decided.
                    self._publish(state)
                    return rule_agent.WARMING_UP
                self._select(state, today)

        alone = len(self.racers) == 1
        # The holder first: its exit decides whether the race is open on this
        # very bar, and a racer read before it would be refused for a
        # position that is about to close. A pair the selection left out is
        # not read at all -- unless it holds a position from before it.
        order = sorted(
            (r for r in self.racers if r.key not in self.benched or r.key == self.holder),
            key=lambda r: r.key != self.holder,
        )
        for racer in self.racers:
            if racer.key in self.benched and racer.key != self.holder:
                racer.outcome = rule_agent.HOLD
        for racer in order:
            racer.trader.quiet_read = not alone and racer.key != self.holder
            ledger = _RacerLedger(self, racer.key, tracker)
            with agent_log_tag(racer=racer.label):
                racer.outcome = racer.trader.run_cycle(racer.bundle, state, ledger)
            if self._event is not None:
                self._announce(state, tracker)
            halt = getattr(racer.trader, "halt", None)
            if halt and self._stopped_day != today:
                # Every racer is still read for the rest of the session -- their
                # levels stay current -- but none may buy (`_gate`).
                self.halt = f"{racer.label} {halt}"
                self._stopped_day = today
        if not alone:
            self._race_line(state)
        self._publish(state)
        outcomes = [r.outcome for r in self.racers]
        for tag in (rule_agent.BOUGHT, rule_agent.SOLD):
            if tag in outcomes:
                return tag
        if self.holder is not None:
            return self.by_key[self.holder].outcome or rule_agent.HOLD
        for tag in (rule_agent.HOLD, rule_agent.WARMING_UP, rule_agent.NO_DATA):
            if tag in outcomes:
                return tag
        return outcomes[0] or rule_agent.HOLD

    # --- the 09:34 selection --------------------------------------------

    def _selection_due(self, state: AppState) -> bool:
        """Whether 09:30 to 09:33 have closed on any racer's symbol: the
        selection's moment, one bar before the forecasts. A run started later
        is past it on its first bar, and picks at once."""
        for ticker in dict.fromkeys(r.trader.ticker for r in self.racers):
            sym_state = state.sym(ticker)
            if sym_state is None:
                continue
            if len(momentum_regime.minute_frame(sym_state)) >= candidates.SELECT_AFTER_BARS:
                return True
        return False

    def _select(self, state: AppState, today) -> None:
        facts = [
            candidates.gather_facts(r.config, r.label, state, today.date(), self.sources)
            for r in self.racers
        ]
        chosen = candidates.select_candidates(facts, self.selection)
        self._apply_selection(today, candidates.to_rows(chosen))
        _log(state, {"type": "analysis", "text": candidates.summary(chosen)})

    def _restore_selection(self, state: AppState, today) -> bool:
        """Take back today's selection from an earlier run (a restart, or ▶ Stop
        and ▶ Start): it was made once, at 09:34, and is not made again."""
        prior = (getattr(state, "orchestra", None) or {}).get("selection") or {}
        rows = prior.get("rows") or []
        if prior.get("date") != str(today.date()) or {r.get("key") for r in rows} != set(self.keys):
            return False
        self._apply_selection(today, rows)
        chosen = [r["label"] for r in rows if r.get("selected")]
        _log(state, {"type": "status", "text": (
            "Keeping today's candidates, picked at 09:34 by the earlier run: "
            + (", ".join(chosen) if chosen else "none") + "."
        )})
        return True

    def _apply_selection(self, today, rows: "list[dict]") -> None:
        self._selected_day = today
        self.selection_rows = [dict(r) for r in rows]
        self.benched = {r["key"]: r.get("reason") or "" for r in rows if not r.get("selected")}

    def _announce(self, state: AppState, tracker: DecisionTracker) -> None:
        kind, key = self._event
        self._event = None
        if len(self.racers) == 1:
            return
        racer = self.by_key[key]
        if kind == "took":
            others = [r.label for r in self.racers if r.key != key]
            text = (
                f"{racer.label} filled first: Orchestra follows it alone until its "
                f"position is closed. Waiting: {', '.join(others)}."
            )
        else:
            armed = [r.label for r in self.racers if self._armed(r) and not self._benched(r)]
            text = (
                f"{racer.label} is flat again, so the race is open: "
                + (f"{', '.join(armed)} can buy." if armed else "no pair is still armed today.")
            )
        _log(state, {"type": "analysis", "text": text})

    # --- what Orchestra looks like -------------------------------------

    @staticmethod
    def _armed(racer: Racer) -> bool:
        trader = racer.trader
        plan = getattr(trader, "plan", None)
        return (
            trader.blocked is None
            and plan is not None
            and not plan.get("stand_down")
        )

    def _benched(self, racer: Racer) -> bool:
        return racer.key in self.benched and racer.key != self.holder

    def board(self) -> "list[dict]":
        """One row per racer: where it stands against its buy level, in its
        own level unit, which is what makes racers on different symbols and
        different prices comparable."""
        rows = []
        for racer in self.racers:
            trader = racer.trader
            plan = trader.plan or {}
            close = trader.last_close
            unit = float(plan.get("level_unit") or 0.0)
            buy = plan.get("buy_level")
            if racer.key == self.holder:
                status = "holding"
            elif self._benched(racer):
                status = f"sits out today ({self.benched[racer.key]})"
            elif self.selection is not None and self._selected_day != self._day:
                status = "waiting for 09:34"
            elif (trader.blocked or {}).get("sit_out"):
                status = "sits out today (" + ", ".join(
                    event_days.LABELS[c] for c in trader.blocked["sit_out"]
                ) + ")"
            elif trader.blocked is not None:
                status = "cannot forecast"
            elif not plan:
                status = "waiting for the forecast"
            elif plan.get("stand_down"):
                status = f"stood down ({plan['stand_down']})"
            elif self.holder is not None:
                status = "waiting for the holder"
            else:
                status = "racing"
            rows.append({
                "key": racer.key,
                "label": racer.label,
                "ticker": trader.ticker,
                "status": status,
                "close": close,
                "buy": None if buy is None else float(buy),
                "sell": None if plan.get("sell_level") is None else float(plan["sell_level"]),
                "to_buy": (
                    (close - float(buy)) / unit
                    if close is not None and buy is not None and unit > 0 else None
                ),
                "unit": unit or None,
            })
        return rows

    def _race_line(self, state: AppState) -> None:
        """One log line per bar for every racer not holding -- in place of the
        read line each would write on its own -- nearest to its buy first."""
        if self.holder is not None:
            return
        stamps = [r.trader.last_bar_ts for r in self.racers if r.trader.last_bar_ts is not None]
        if not stamps:
            return
        newest = max(stamps)
        if newest == self._last_line:
            return
        self._last_line = newest
        racing = [row for row in self.board() if row["status"] == "racing" and row["to_buy"] is not None]
        if not racing:
            return
        racing.sort(key=lambda row: row["to_buy"])
        parts = [
            f"{row['label']} ${row['close']:,.2f} vs buy ${row['buy']:,.2f} ({row['to_buy']:+.2f} u)"
            for row in racing
        ]
        _log(state, {"type": "analysis", "text": (
            f"Orchestra at {newest:%H:%M}, flat — nearest to a buy first, in each pair's "
            f"level unit: {' · '.join(parts)}"
        )})

    def _publish(self, state: AppState) -> None:
        state.orchestra = {
            "date": str(self._day.date()) if self._day is not None else None,
            "order": self.keys,
            "labels": {r.key: r.label for r in self.racers},
            "holder": self.holder,
            "board": self.board(),
            "running": True,
            "selection": (
                None if self.selection is None or self._selected_day is None
                else {
                    "date": str(self._selected_day.date()),
                    "rows": self.selection_rows,
                    "rules": asdict(self.selection),
                }
            ),
        }

    def publish_memory(self, state: AppState) -> None:
        for racer in self.racers:
            racer.trader.publish_memory(state)

    def activity(self, outcome: str, tracker: DecisionTracker) -> "tuple[str, str]":
        """The status line: the holder's own, else how the race stands."""
        if outcome == rule_agent.CLOSED:
            return rule_agent.WAITING, "Agent waiting for session start"
        if self.holder is not None:
            racer = self.by_key[self.holder]
            dot, phrase = racer.trader.activity(racer.outcome or rule_agent.HOLD, _RacerLedger(self, racer.key, tracker))
            return dot, f"{phrase} ({racer.label})"
        board = self.board()
        if self.selection is not None and self._selected_day != self._day:
            return rule_agent.WAITING, "Agent waiting to pick its candidates at 09:34"
        racing = [row for row in board if row["status"] == "racing"]
        if racing:
            near = [row for row in racing if row["to_buy"] is not None]
            if near:
                best = min(near, key=lambda row: row["to_buy"])
                return rule_agent.TRADING, (
                    f"Agent watching {len(racing)} pairs — nearest: {best['label']}, "
                    f"{best['to_buy']:+.2f} × its unit from its buy level"
                )
            return rule_agent.TRADING, f"Agent watching {len(racing)} pairs"
        if any(row["status"] == "waiting for the forecast" for row in board):
            return rule_agent.WAITING, "Agent waiting for the opening forecasts"
        if all(row["status"] == "cannot forecast" for row in board):
            return rule_agent.FAILED, "Agent cannot trade today — see the log"
        if all(row["status"].startswith("sits out") for row in board):
            return rule_agent.WAITING, "Agent found no candidates today"
        return rule_agent.WAITING, "Agent stood down for the day"


def build_orchestra(
    race: OrchestraConfig, bundles: "dict[str, dict]", sources=None,
) -> Orchestra:
    """An `Orchestra` over `race`'s racers, each built the way an Apple Trader
    run is (`apple_trader.build_trader`) on its loaded bundle. `sources` serves
    the selection its briefings and earnings calendar (live by default)."""
    return Orchestra([
        Racer(
            key=racer_key(config),
            label=racer_label(config),
            config=config,
            trader=at.build_trader(config, bundles[racer_key(config)]),
            bundle=bundles[racer_key(config)],
        )
        for config in race.racers
    ], selection=race.selection, sources=sources)


def load_racers(race: OrchestraConfig) -> "tuple[dict[str, dict], dict[str, str]]":
    """`(bundles, refusals)` per racer key: each racer's model loaded and
    checked exactly as an Apple Trader run checks its own (`_apple_trader_loop`)."""
    bundles: "dict[str, dict]" = {}
    refusals: "dict[str, str]" = {}
    for config in race.racers:
        key = racer_key(config)
        refusal = at.model_ticker_error(config)
        bundle = None
        if refusal is None:
            bundle = apple_models.load(config.model_key, config.ticker)
            if bundle is None:
                refusal = apple_models.unavailable_reason(config.model_key, config.ticker)
            else:
                refusal = at.config_error(config, bundle)
        if refusal is None:
            bundles[key] = bundle
        else:
            refusals[key] = refusal
    return bundles, refusals


# --- the loop -------------------------------------------------------------------


def _orchestra_loop(
    state: AppState,
    tracker: DecisionTracker,
    race_config: OrchestraConfig,
    cycle_sec: int,
    stop_event: threading.Event,
) -> None:
    bundles, refusals = load_racers(race_config)
    for key, refusal in refusals.items():
        _log(state, {"type": "error", "text": f"{racer_label(key)} is left out of Orchestra: {refusal}"})
    kept = [c for c in race_config.racers if racer_key(c) in bundles]
    if not kept:
        _log(state, {"type": "error", "text": "No pair can run, so Orchestra cannot start."})
        rule_agent.end_session(state, tracker, None)
        return
    race_config = OrchestraConfig(kept, selection=race_config.selection)
    race = build_orchestra(race_config, bundles)
    if len(kept) > 1:
        _log(state, {"type": "status", "text": (
            f"Orchestra watching {len(kept)} pairs: "
            + ", ".join(
                f"{racer_label(c)} (buy {c.buy_k:g} / sell {c.sell_k:g} × {c.unit_phrase})"
                for c in kept
            )
            + ". Each forecasts its own session at 9:35 and rests its own levels; the first "
            "whose buy fills is the only one that trades until its position is closed, then "
            "the race is open again. Ties go to the earlier pair in this list. A stop-out "
            "stops the agent."
            + (
                " At 09:34 it first picks the day's candidates (keeping "
                + (f"the best {race_config.selection.max_pairs}" if race_config.selection.max_pairs else "every eligible pair")
                + "); the rest sit the day out."
                if race_config.selection is not None else ""
            )
        )})
    for config in kept:
        key = racer_key(config)
        with agent_log_tag(racer=racer_label(config)):
            _log(state, {"type": "status", "text": at._armed_summary(
                config, apple_models.get(config.model_key), bundles[key]
            )})

    def cycle() -> str:
        outcome = race.run_cycle(state, tracker)
        state.agent_activity = race.activity(outcome, tracker)
        race.publish_memory(state)
        if race.halt and not stop_event.is_set():
            _log(state, {"type": "status", "text": (
                f"{race.halt} — Orchestra stops. Press ▶ Start Agent to trade again today."
            )})
            stop_event.set()
        return outcome

    try:
        rule_agent.run_loop(state, tracker, cycle, stop_event, cycle_sec, "Orchestra")
    finally:
        # Unless a newer run has already replaced this one (▶ Start while
        # running stops this loop and starts another before this line).
        if getattr(state, "orchestra", None) and state.agent_stop_event is stop_event:
            state.orchestra = {**state.orchestra, "running": False}


def launch_orchestra(
    state: AppState,
    tracker: DecisionTracker,
    race_config: OrchestraConfig,
    cycle_sec: int = APPLE_TRADER_CYCLE_SEC,
) -> None:
    """Stop any running agent for this state, then start Orchestra's loop.
    Every pair's symbol must already be streamed."""
    prior = getattr(state, "orchestra", None) or {}
    state.orchestra = {
        "date": None,
        "order": race_config.keys,
        "labels": {racer_key(c): racer_label(c) for c in race_config.racers},
        "holder": prior.get("holder"),
        "board": [],
        "running": True,
        # Today's 09:34 selection, if an earlier run made it: kept, not re-made.
        "selection": prior.get("selection"),
    }
    rule_agent.launch(
        state, tracker, ORCHESTRA_KEY, race_config.tickers,
        target=_orchestra_loop,
        args=(state, tracker, race_config, cycle_sec),
        stop_agent=stop_agent,
    )
