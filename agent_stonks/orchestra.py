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

Split: every pair at once, each on its own share
-------------------------------------------------
All of the above is the **race** (`MODE_RACE`, the default). In the other
mode, **split** (`MODE_SPLIT`, the user's request of 2026-10-12), nobody
races: each pair is given a share of the budget (`OrchestraConfig.budget_pct`,
in percent, adding up to at most 100) and trades it on its own, all pairs at
once.

* **The budget** is what the ledger is worth at the session's first open bar
  -- its cash, plus any shares of a pair's symbol it already holds. Each pair's
  account starts the session at its share of it.
* **A pair's account** (`PairAccount`) is that share less what its open
  shares cost, plus what its sales brought back, fees included -- booked from
  the ledger's own fills, at the ledger's own price. A pair sizes every buy,
  its ladder too, off its account exactly as Apple Trader sizes off the
  ledger's cash (`position_pct` of it), and never sees more cash than the
  ledger has. Its position is the shares *it* bought, so two pairs on one
  symbol each manage only their own.
* **No selection.** The 09:34 candidate pick narrows a race; a split trades
  every pair it is given.
* **A stop-out stands down that pair** for the rest of the day -- through a
  restart too (`DayRangeTrader.rearm_after_stop`) -- and the others keep
  trading. Stopping the agent would leave their positions unmanaged.

With one pair at 100% a split is the same Apple Trader run as a race of one.
The day's budget and every account are published with Orchestra's state, so a
restart the same day continues them rather than re-reading the budget off a
ledger that is already invested.

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

# How Orchestra shares the budget between its pairs (see the module docstring).
MODE_RACE = "race"
MODE_SPLIT = "split"
MODE_LABELS = {
    MODE_RACE: "Race — the first buy to fill takes the budget",
    MODE_SPLIT: "Split — each pair trades its own share, all at once",
}

# The fields every racer must share. Orchestra is one rule set over several
# instruments, not several rule sets: the pair (ticker, model) and the numbers
# tuned per pair -- the two distances and the circuit breaker -- are the racer's
# own, and everything else is Orchestra's. `level_source` follows the model.
PER_RACER_FIELDS = (
    "ticker", "model_key", "buy_k", "sell_k", "min_win_k", "level_source",
)


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
    "fastfurious": "Fast&Furious",
    "goodmoments": "GoodMoments",
    "ff_highlow": "F&F + HighLow",
}


def racer_label(config_or_key) -> str:
    """"INTC · Day Range": what the log, the board and the form call a racer."""
    # A key is the string; anything else is a config, of whichever class --
    # see `apple_trader.as_config`.
    if isinstance(config_or_key, str):
        ticker, model_key = split_key(config_or_key)
    else:
        ticker, model_key = config_or_key.ticker, config_or_key.model_key
    name = SHORT_MODEL_NAMES.get(model_key) or apple_models.get(model_key).label
    return f"{ticker} · {name}"


def all_pairs() -> "list[str]":
    """Every (ticker, model) pair Orchestra can be given, as racer keys: each
    symbol some model covers, with each day-range model it has, in picker
    order. Fast&Furious drives its own rules (a per-minute signal, no levels),
    which neither the race nor the 09:34 selection is written for -- and so
    does Fast&Furious + HighLow before 9:35, whose window would buy ahead of
    the selection."""
    return [
        f"{ticker}:{key}"
        for ticker in apple_models.tickers()
        for key in apple_models.keys_for_strategy(
            ticker, apple_models.STRATEGY_DAYRANGE, windows=False
        )
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
    # Race or split (`MODE_LABELS`); every record made before the split existed
    # is a race.
    mode: str = MODE_RACE
    # Under a split: each pair's share of the day's budget, in percent, by
    # racer key -- every pair's above 0, all of them together at most 100. The
    # rest of the budget stays in cash. Empty in a race.
    budget_pct: "dict[str, float]" = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.racers = [at.as_config(r) for r in self.racers]
        if isinstance(self.selection, dict):
            known = {f.name for f in fields(candidates.SelectionRules)}
            self.selection = candidates.SelectionRules(
                **{k: v for k, v in self.selection.items() if k in known}
            )
        if not self.racers:
            raise ValueError("Orchestra needs at least one (ticker, model) pair")
        if self.mode not in MODE_LABELS:
            raise ValueError(f"Orchestra has no mode {self.mode!r}")
        other = [r for r in self.racers if r.strategy != apple_models.STRATEGY_DAYRANGE]
        if other:
            raise ValueError(
                f"Orchestra races the day-range rules, and {racer_label(other[0])} runs its "
                "own: run it as Apple Trader"
            )
        windowed = [r for r in self.racers if apple_models.window_key(r.model_key)]
        if windowed:
            raise ValueError(
                f"Orchestra does not trade an opening window, and {racer_label(windowed[0])} "
                "buys before 9:35: its pairs are picked at 09:34 and race from their own "
                "forecasts. Run it as Apple Trader"
            )
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
        if self.mode != MODE_SPLIT:
            self.budget_pct = {}
            return
        if self.selection is not None:
            raise ValueError(
                "a split Orchestra trades every pair it is given; the 09:34 candidate "
                "pick only narrows a race"
            )
        # Only the pairs it has: a pair left out of the run (a model that did
        # not load) takes its share with it, into cash.
        given = {str(k): float(v) for k, v in (self.budget_pct or {}).items()}
        self.budget_pct = {key: given.get(key, 0.0) for key in keys}
        none = [racer_label(k) for k, pct in self.budget_pct.items() if not pct > 0]
        if none:
            raise ValueError(f"{', '.join(none)} {'has' if len(none) == 1 else 'have'} no share of the budget")
        total = sum(self.budget_pct.values())
        if total > 100.0 + 1e-6:
            raise ValueError(
                f"the pairs' shares add up to {total:g}% of the budget; they can be at most 100%"
            )

    @property
    def keys(self) -> "list[str]":
        return [racer_key(r) for r in self.racers]

    @property
    def tickers(self) -> "list[str]":
        return list(dict.fromkeys(r.ticker for r in self.racers))

    @property
    def split(self) -> bool:
        return self.mode == MODE_SPLIT


def _shared(config: AppleTraderConfig) -> dict:
    return {k: v for k, v in asdict(config).items() if k not in PER_RACER_FIELDS}


def build_orchestra_config(
    pairs: "list[str]",
    base: AppleTraderConfig,
    levels: "dict | None" = None,
    selection: "candidates.SelectionRules | None" = None,
    mode: str = MODE_RACE,
    budget_pct: "dict[str, float] | None" = None,
) -> OrchestraConfig:
    """One racer per pair, each `base` with its own pair and the numbers tuned
    for it: `levels[key]` = `(buy_k, sell_k, min_win_k)` where given, the
    pair's shipped defaults otherwise. `selection`, `mode` and `budget_pct`
    as `OrchestraConfig`'s."""
    levels = levels or {}
    racers = []
    for key in pairs:
        ticker, model_key = split_key(key)
        buy_k, sell_k = at.dayrange_levels(ticker, model_key)
        min_win_k = at.min_win_for(ticker, model_key)
        if key in levels:
            buy_k, sell_k, min_win_k = levels[key]
        racers.append(replace(
            base, ticker=ticker, model_key=model_key,
            buy_k=float(buy_k), sell_k=float(sell_k), min_win_k=float(min_win_k),
            level_source=apple_models.get(model_key).level_source,
        ))
    return OrchestraConfig(racers, selection=selection, mode=mode, budget_pct=budget_pct or {})


def orchestra_signature(race: OrchestraConfig) -> str:
    """Compact identity of an Orchestra setup, standing in for a model name in SimLab.

    The racers in order -- each its model, symbol, two distances and circuit
    breaker, since those are what the pair tunes -- then the shared rules once,
    read off the first racer's own signature with its per-pair part removed, so
    a rule signs here exactly as it does in an Apple Trader run. A split says
    so up front and adds each pair's share of the budget ("=40%"); a race signs
    as it always has.
    """
    first = race.racers[0]
    unit = LEVEL_UNIT_TOKENS[first.level_unit]
    pairs = ",".join(
        f"{r.model_key}_{r.ticker}@{r.buy_k:g}/{r.sell_k:g}{unit}"
        + (f"/mw{r.min_win_k:g}" if r.min_win_k else "")
        + (f"={race.budget_pct[racer_key(r)]:g}%" if race.split else "")
        for r in race.racers
    )
    head = "split:" if race.split else ""
    single = at.config_signature(first)
    rules = single[single.index("size="):-1]
    rules = re.sub(r",min_win=[^,)]*", "", rules)
    return f"orchestra[{head}{pairs}]({rules}{candidates.rules_signature(race.selection)})"


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


@dataclass
class PairAccount:
    """One pair's part of the ledger under a split: its share of the day's
    budget (`pct`), the cash that share has left, and the shares it holds."""

    pct: float
    cash: float
    shares: float = 0.0

    def settle(self, decision) -> None:
        """Book a fill exactly as the ledger books it -- the same price, the
        same fee, the same arithmetic -- so a pair given the whole budget has
        the ledger's own cash to the cent."""
        filled = float(decision.filled_quantity or 0.0)
        price = float(decision.price or 0.0)
        fee = float(getattr(decision, "fee", 0.0) or 0.0)
        if decision.action == "buy":
            self.cash -= filled * price + fee
            self.shares += filled
        else:
            self.cash += filled * price - fee
            self.shares = max(0.0, self.shares - filled)


class _SplitLedger:
    """The shared ledger as one pair of a split sees it: its own account.

    Its cash is the account's -- never more than the ledger itself has, and
    never below nothing -- and its position is the shares it bought and has not
    sold, never more than the ledger holds (a sale made outside the app is a
    sale of them). Fills go through the ledger and are booked on the account.
    Everything else is the ledger's.
    """

    def __init__(self, race: "Orchestra", key: str, tracker: DecisionTracker) -> None:
        self._race = race
        self._key = key
        self._tracker = tracker

    def _account(self) -> "PairAccount | None":
        return self._race.accounts.get(self._key)

    def position_for(self, symbol: str) -> float:
        account = self._account()
        if account is None or symbol != self._race.by_key[self._key].trader.ticker:
            return 0.0
        return min(account.shares, self._tracker.position_for(symbol))

    def snapshot(self) -> dict:
        snap = self._tracker.snapshot()
        account = self._account()
        cash = 0.0 if account is None else max(0.0, min(account.cash, snap["cash"]))
        return {**snap, "cash": cash}

    def record_trade(self, symbol, action, quantity, reasoning, *args, **kwargs):
        decision = self._tracker.record_trade(symbol, action, quantity, reasoning, *args, **kwargs)
        account = self._account()
        if decision.status == "filled" and account is not None:
            account.settle(decision)
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
    """The coordinator: one trader per racer, and in a race one position
    between them -- in a split, one account each (`PairAccount`).

    `run_cycle(state, tracker)` is the whole interface -- the live loop calls
    it once a closed bar, SimLab's rule day loop once a replayed one.
    """

    def __init__(
        self,
        racers: "list[Racer]",
        selection: "candidates.SelectionRules | None" = None,
        sources=None,
        mode: str = MODE_RACE,
        budget_pct: "dict[str, float] | None" = None,
    ) -> None:
        if not racers:
            raise ValueError("Orchestra needs at least one racer")
        self.racers = racers
        self.by_key = {r.key: r for r in racers}
        self.mode = mode
        # Under a split: each pair's share of the day's budget, the budget the
        # shares are of (the ledger's worth at the session's first open bar),
        # the day it is for, and each pair's account.
        self.budget_pct = dict(budget_pct or {}) if mode == MODE_SPLIT else {}
        self.budget_base: "float | None" = None
        self._budget_day = None
        self.accounts: "dict[str, PairAccount]" = {}
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
            if self.split:
                # Nobody waits for anybody. A stop-out stands the pair down
                # for the day, and a restart must not undo that: the agent
                # goes on running, so no ▶ Start is the say-so to re-arm it.
                racer.trader.rearm_after_stop = False
            else:
                racer.trader.entry_gate = (lambda key=racer.key: self._gate(key))

    @property
    def keys(self) -> "list[str]":
        return [r.key for r in self.racers]

    @property
    def split(self) -> bool:
        return self.mode == MODE_SPLIT

    def holds(self, key: str) -> bool:
        """Whether this racer has a position: the race's holder, or under a
        split a pair whose account holds shares."""
        if self.split:
            account = self.accounts.get(key)
            return account is not None and account.shares > 0
        return key == self.holder

    def holders(self) -> "list[str]":
        return [key for key in self.keys if self.holds(key)]

    def _ledger(self, key: str, tracker: DecisionTracker):
        if self.split:
            return _SplitLedger(self, key, tracker)
        return _RacerLedger(self, key, tracker)

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

    # --- a split's accounts ----------------------------------------------

    def _open_accounts(self, state: AppState, tracker: DecisionTracker, today) -> None:
        """Each pair's account for this session, on its first open bar.

        An earlier run today (a restart, or ▶ Stop and ▶ Start) left the day's
        budget and every account in `AppState.orchestra`: those go on, since
        the ledger they are read off is by now partly invested. A share changed
        in between moves the difference into or out of the pair's cash, and a
        pair new to the run starts on its share.

        Otherwise the budget is what the ledger is worth now: its cash, plus any
        shares of a pair's symbol it holds already (at their last price), which
        are booked to the first pair on that symbol, the way a race hands them
        to the first racer on it (`_adopt`).
        """
        self._budget_day = today
        alone = len(self.racers) == 1
        prior = (getattr(state, "orchestra", None) or {}).get("budget") or {}
        if prior.get("date") == str(today.date()) and prior.get("base") is not None:
            base = float(prior["base"])
            saved = prior.get("accounts") or {}
            self.budget_base = base
            self.accounts = {}
            for key, pct in self.budget_pct.items():
                was = saved.get(key)
                if was is None:
                    self.accounts[key] = PairAccount(pct, pct / 100.0 * base)
                    continue
                cash = float(was["cash"])
                if float(was["pct"]) != pct:
                    cash += (pct - float(was["pct"])) / 100.0 * base
                self.accounts[key] = PairAccount(pct, cash, float(was.get("shares") or 0.0))
            self._reconcile(tracker)
            self._book_unclaimed(state, self._unclaimed(state, tracker))
            if not alone:
                _log(state, {"type": "status", "text": (
                    f"Continuing today's split of the ${base:,.2f} budget: " + ", ".join(
                        f"{self.by_key[key].label} {account.pct:g}% "
                        f"(${account.cash:,.2f} cash, {account.shares:g} sh)"
                        for key, account in self.accounts.items()
                    ) + "."
                )})
            return

        self.accounts = {}
        held = self._unclaimed(state, tracker)
        base = float(tracker.snapshot()["cash"])
        for shares, price in held.values():
            base += shares * price
        self.budget_base = base
        self.accounts = {
            key: PairAccount(pct, pct / 100.0 * base) for key, pct in self.budget_pct.items()
        }
        self._book_unclaimed(state, held)
        if not alone:
            unused = 100.0 - sum(self.budget_pct.values())
            _log(state, {"type": "status", "text": (
                f"Today's budget is ${base:,.2f}: " + ", ".join(
                    f"{self.by_key[key].label} {account.pct:g}% (${account.pct / 100.0 * base:,.2f})"
                    for key, account in self.accounts.items()
                ) + (f"; {unused:g}% stays in cash." if unused > 1e-9 else ".")
            )})

    def _tickers(self) -> "list[str]":
        return list(dict.fromkeys(r.trader.ticker for r in self.racers))

    def _unclaimed(self, state: AppState, tracker: DecisionTracker) -> "dict[str, tuple[float, float]]":
        """`{ticker: (shares, last price)}` the ledger holds of a pair's symbol
        beyond what the pairs' accounts hold of it."""
        found = {}
        for ticker in self._tickers():
            claimed = sum(
                a.shares for k, a in self.accounts.items() if self.by_key[k].trader.ticker == ticker
            )
            extra = tracker.position_for(ticker) - claimed
            if extra > 0:
                found[ticker] = (extra, state.mark_price(ticker) or 0.0)
        return found

    def _book_unclaimed(self, state: AppState, found: "dict[str, tuple[float, float]]") -> None:
        """Book shares nobody's account holds to the first pair on their symbol,
        at their last price, so they are managed rather than left behind."""
        for ticker, (shares, price) in found.items():
            racer = next(r for r in self.racers if r.trader.ticker == ticker)
            account = self.accounts[racer.key]
            account.shares += shares
            account.cash -= shares * price
            _log(state, {"type": "status", "text": (
                f"The ledger holds {shares:g} {ticker} no pair bought; Orchestra books them "
                f"to {racer.label}'s share of the budget at ${price:,.2f}, and it manages them."
            )})

    def _reconcile(self, tracker: DecisionTracker) -> None:
        """Take shares the ledger no longer has (sold outside the app) off the
        accounts, the last pair on a symbol first, so no pair goes on counting
        a position that is gone. The cash they brought is the ledger's, not
        the pair's: nobody knows what they were sold for."""
        for ticker in self._tickers():
            on_it = [a for k, a in self.accounts.items() if self.by_key[k].trader.ticker == ticker]
            excess = sum(a.shares for a in on_it) - tracker.position_for(ticker)
            for account in reversed(on_it):
                if excess <= 0:
                    break
                cut = min(account.shares, excess)
                account.shares -= cut
                excess -= cut

    def _budget_record(self) -> "dict | None":
        """The day's budget and every account, for `AppState.orchestra` -- what
        a restart continues (`_open_accounts`) and the board shows."""
        if not self.split or self._budget_day is None:
            return None
        return {
            "date": str(self._budget_day.date()),
            "base": self.budget_base,
            "accounts": {key: asdict(account) for key, account in self.accounts.items()},
        }

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
        if self.split:
            if self._budget_day != today:
                self._open_accounts(state, tracker, today)
            self._reconcile(tracker)
        elif not self._adopted:
            self._adopt(state, tracker)
        if self.selection is not None and self._selected_day != today:
            if not self._restore_selection(state, today):
                if not self._selection_due(state):
                    # Before 09:34 there is nothing to read: which racers will
                    # forecast is not decided.
                    self._publish(state)
                    return rule_agent.WARMING_UP
                self._select(state, today)

        alone = len(self.racers) == 1
        # The holder first: its exit decides whether the race is open on this
        # very bar, and a racer read before it would be refused for a
        # position that is about to close. A pair the selection left out is
        # not read at all -- unless it holds a position from before it. (A
        # split has neither: every pair is read, in order.)
        order = sorted(
            (r for r in self.racers if r.key not in self.benched or r.key == self.holder),
            key=lambda r: r.key != self.holder,
        )
        for racer in self.racers:
            if racer.key in self.benched and racer.key != self.holder:
                racer.outcome = rule_agent.HOLD
        for racer in order:
            # A pair with a position writes its own read line; the rest share
            # one (`_race_line`).
            racer.trader.quiet_read = not alone and not self.holds(racer.key)
            ledger = self._ledger(racer.key, tracker)
            with agent_log_tag(racer=racer.label):
                racer.outcome = racer.trader.run_cycle(racer.bundle, state, ledger)
            if self._event is not None:
                self._announce(state, tracker)
            halt = getattr(racer.trader, "halt", None)
            if halt and self.split:
                # That pair's own stand-down already keeps it out today; the
                # others go on, each with its own position to manage.
                racer.trader.halt = None
                if not alone:
                    with agent_log_tag(racer=racer.label):
                        _log(state, {"type": "analysis", "text": (
                            f"{racer.label} {halt}: it sits out the rest of today. The "
                            "other pairs keep trading their own shares of the budget."
                        )})
            elif halt and self._stopped_day != today:
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
        different prices comparable -- and under a split, its account."""
        rows = []
        for racer in self.racers:
            trader = racer.trader
            plan = trader.plan or {}
            close = trader.last_close
            unit = float(plan.get("level_unit") or 0.0)
            buy = plan.get("buy_level")
            account = self.accounts.get(racer.key)
            if self.holds(racer.key):
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
                status = "watching" if self.split else "racing"
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
                "budget_pct": self.budget_pct.get(racer.key),
                "cash": None if account is None else account.cash,
                "shares": None if account is None else account.shares,
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
        racing = [
            row for row in self.board()
            if row["status"] in ("racing", "watching") and row["to_buy"] is not None
        ]
        if not racing:
            return
        racing.sort(key=lambda row: row["to_buy"])
        parts = [
            f"{row['label']} ${row['close']:,.2f} vs buy ${row['buy']:,.2f} ({row['to_buy']:+.2f} u)"
            for row in racing
        ]
        head = "the pairs without a position" if self.split else "flat"
        _log(state, {"type": "analysis", "text": (
            f"Orchestra at {newest:%H:%M}, {head} — nearest to a buy first, in each pair's "
            f"level unit: {' · '.join(parts)}"
        )})

    def _publish(self, state: AppState) -> None:
        state.orchestra = {
            "date": str(self._day.date()) if self._day is not None else None,
            "mode": self.mode,
            "order": self.keys,
            "labels": {r.key: r.label for r in self.racers},
            "holder": self.holder,
            "holders": self.holders(),
            "budget": self._budget_record(),
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
        holding = [row["label"] for row in board if row["status"] == "holding"]
        racing = [row for row in board if row["status"] in ("racing", "watching")]
        if holding:
            # Only a split gets here with a position: a race has its holder.
            more = f" — watching {len(racing)} more" if racing else ""
            return rule_agent.TRADING, f"Agent holding {', '.join(holding)}{more}"
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
    ], selection=race.selection, sources=sources, mode=race.mode, budget_pct=race.budget_pct)


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
    race_config = OrchestraConfig(
        kept, selection=race_config.selection,
        mode=race_config.mode, budget_pct=race_config.budget_pct,
    )
    race = build_orchestra(race_config, bundles)
    if len(kept) > 1 and race_config.split:
        unused = 100.0 - sum(race_config.budget_pct.values())
        _log(state, {"type": "status", "text": (
            f"Orchestra trading {len(kept)} pairs at once, each on its own share of the "
            "budget: "
            + ", ".join(
                f"{racer_label(c)} {race_config.budget_pct[racer_key(c)]:g}% "
                f"(buy {c.buy_k:g} / sell {c.sell_k:g} × {c.unit_phrase})"
                for c in kept
            )
            + (f"; {unused:g}% stays in cash" if unused > 1e-9 else "")
            + ". Each forecasts its own session at 9:35, rests its own levels and sizes "
            f"every buy at {kept[0].position_pct:g}% of the cash its share has left. The "
            "budget is what the ledger is worth at the first open bar. A stop-out stands "
            "that pair down for the day; the others keep trading."
        )})
    elif len(kept) > 1:
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
        "mode": race_config.mode,
        "order": race_config.keys,
        "labels": {racer_key(c): racer_label(c) for c in race_config.racers},
        "holder": prior.get("holder"),
        "holders": [],
        # Today's budget and each pair's account, if an earlier split run
        # today kept them: continued, not re-read off an invested ledger.
        "budget": prior.get("budget"),
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
