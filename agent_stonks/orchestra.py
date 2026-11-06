"""Orchestra: Apple Trader's rules over several (ticker, model) pairs at once.

Apple Trader trades one symbol on one model. Orchestra is its own agent: it
runs one Apple Trader state machine per pair -- AAPL on HighLow and AAPL on
HighLow2 are two pairs -- over one shared ledger, and the pairs race for its
cash:

* **Racing.** Every pair trades on its own rules, exactly as Apple Trader
  would on it alone: a day-range pair (HighLow, HighLow2) forecasts its
  session at 9:35 and rests its own levels; a GoodMoments pair (GoodMoments,
  HighLow2 + GoodMoments, HighLow + GoodMoments -- the probability rules,
  since 2026-10-25) reads its edge at the close of every bar. Each bar the pairs are read in Orchestra's
  order -- the most profitable a session in SimLab's tuning first
  (`tuning_profits`, `rank_pairs`) -- and the first whose buy *fills* takes
  its share of the cash: `position_pct` of it, as Apple Trader sizes. A buy
  level touched is not enough, and neither is a limit order the price had
  already left: those leave the race open.
* **Holding several.** While a pair holds, the others are still read every
  bar, and one whose buy fills too takes `position_pct` of the cash that is
  left -- whenever a share of it fits in what is left, on the same bar or
  later. So when two pairs buy on one bar the more profitable one gets the
  cash, and the next one the rest. Only one pair may hold a symbol: a pair on
  a held symbol sees it as flat, so it never adopts a position that is not
  its own, and its entries are closed (`BaseTrader.entry_gate`) until the
  holder sells, as they are while the cash left buys none of its shares.
  (For a few hours on 2026-10-28 a share also had to be cheaper than every
  held one; the user went back on it. No SimLab run was stored under it.)
  (The user's request of 2026-10-28. Until then the first pair to fill was
  the only one to trade until its position closed --
  `OrchestraConfig.one_holder`, which SimLab records from before replay
  with.)
* **Reopening.** Once a holder's position is closed -- at the target, by the
  momentum take and its runner, by the circuit breaker or by its stop -- its
  cash is back in the race, open to every pair that is still armed. The
  holder's own rules decide whether *it* is: a breaker stand-down or a
  stop-out keeps that pair out for the day and leaves the rest racing. (The
  user's choices, 2026-10-02 for the breaker, 2026-10-27 for the stop.)
* **Picking.** With a selection (`OrchestraConfig.selection`, on by default),
  Orchestra first narrows its pairs to the day's candidates, once, at 09:34 --
  four opening minutes in, one bar before the forecasts -- from the
  pre-market briefing, the earnings calendar and the opening gap
  (`agent_stonks.candidates`). The pairs left out sit the day out: they are
  not read and never forecast. The choice is published and saved with the
  session, so a restart keeps it rather than picking again. Nothing is read
  before the pick, so a GoodMoments pair -- which could buy from 09:30 on
  its own -- trades from the 09:34 bar.
* **Stopping.** A day-range stop-out stands that pair down for the rest of
  the day -- through a restart too (`DayRangeTrader.rearm_after_stop`), since
  the agent runs on and no ▶ Start is the say-so to re-arm it -- and the race
  reopens to the others. Only a race of one stops, as an Apple Trader run
  does: the live loop stops the agent and only ▶ Start re-arms it. A
  GoodMoments stop does not stand anything down -- its own rules pause that
  pair's entries for 15 bars, as they do on Apple Trader.

  Until 2026-10-27 any pair's stop-out ended the whole run, and in a replay
  closed every pair's entries for the rest of that session
  (`OrchestraConfig.halt_on_stop`, which SimLab records from before then
  replay with).

Nothing about a pair's own rules changes. With one pair Orchestra *is* an
Apple Trader run, decision for decision (`tests/test_orchestra.py`), which is
what lets it share every tuned level and every stored SimLab record with it.
In the code a pair Orchestra is running is a "racer".

What a pair is given, and what it shares
----------------------------------------
Every pair shares Orchestra's sizing, its closing flatten, its time limit, its
days off and its no-buy rule on a falling, negative-gamma day
(`SHARED_BY_ALL`). The day-range pairs also share every other rule of theirs
-- the unit, the ladder, the exits, the breach policy, the circuit breaker --
and each keeps its two distances, its stop and its momentum confirmation
period (`DAYRANGE_PAIR_FIELDS`). A GoodMoments pair keeps its own configuration --
its pair's base, or the shipped defaults -- and Orchestra sets only the
numbers that matter most (`SIGNAL_PAIR_FIELDS`): the edge a buy needs, the
take, the stop and its momentum confirmation, and on the hybrid HighLow2's
buy distance. Fast&Furious is not offered: neither the race nor the 09:34
selection is written for its rules.

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
  trading. Stopping the agent would leave their positions unmanaged. (A
  GoodMoments pair pauses for 15 bars instead, by its own rules.)

With one pair at 100% a split is the same Apple Trader run as a race of one.
The day's budget and every account are published with Orchestra's state, so a
restart the same day continues them rather than re-reading the budget off a
ledger that is already invested.

Where the racers keep their state
---------------------------------
Apple Trader keeps its form and its record in `AppState.apple_trader_config`
and `apple_trader_levels`. Orchestra gives each racer a `RacerSlot` instead,
keyed by `racer_key` ("INTC:highlow"), in `AppState.orchestra_configs` and
`orchestra_levels`; Orchestra as a whole -- order, holder, the board -- is
`AppState.orchestra`. The chart of a symbol draws the racer holding it, else
the first racer on it (`model_overlays.live_trader_view`).
"""
from __future__ import annotations

import logging
import re
import threading
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Optional

from . import apple_models, candidates, event_days, market_hours, momentum_regime, rule_agent
from . import apple_trader as at
from . import orchestra_defaults
from .agent import stop_agent
from .apple_trader import AppleTraderConfig, TraderSlot
from .config import (
    APPLE_TRADER_CYCLE_SEC,
    APPLE_TRADER_GM_COOLDOWN_BARS,
    APPLE_TRADER_GM_LAST_ENTRY_MINUTE,
    APPLE_TRADER_TUNED_LEVELS,
    LEVEL_UNIT_TOKENS,
)
from .decisions import DecisionTracker
from .state import AppState, agent_log_tag
from .state import append_agent_log as _log

logger = logging.getLogger(__name__)

# The agent's identity: its personality key (also every SimLab run's), what
# the pickers call it and its face.
ORCHESTRA_KEY = "orchestra"
ORCHESTRA_LABEL = "Orchestra (rule-based, no LLM)"
ORCHESTRA_AVATAR = "Multiavatar-2d21d8b739482562eb.png"

# How Orchestra shares the budget between its pairs (see the module docstring).
MODE_RACE = "race"
MODE_SPLIT = "split"
MODE_LABELS = {
    MODE_RACE: "Race — the most profitable buy first, the rest with what is left",
    MODE_SPLIT: "Split — each pair trades its own share, all at once",
}

# The rule sets Orchestra runs pairs on: the day-range rules (two resting
# levels) and GoodMoments' probability rules (an edge read every bar).
STRATEGIES = (apple_models.STRATEGY_DAYRANGE, apple_models.STRATEGY_PROBABILITY)

# What Orchestra sets on each day-range racer, over the rules they share: the
# two distances, the stop and the momentum confirmation period, tuned per pair
# (the confirmation and the stop per pair since 2026-10-26, the user's
# requests; they were shared).
DAYRANGE_PAIR_FIELDS = ("buy_k", "sell_k", "stop_gain_fraction", "momentum_confirmation_bars")
# The fields a day-range racer need not share with the others. Orchestra is
# one rule set over several instruments, not several rule sets: the pair
# (ticker, model) and its own numbers are the racer's, and everything else is
# Orchestra's. The circuit breaker is one setting for every pair since
# 2026-10-26 (the user's request), but a record from before it has one per
# pair, which still replays.
PER_RACER_FIELDS = ("ticker", "model_key", "min_win_k") + DAYRANGE_PAIR_FIELDS
# The fields every racer must share, whatever its rule set: Orchestra's sizing,
# its closing flatten, its time limit (one for every pair, the user's request
# of 2026-11-02), the sessions it sits out and the no-buy rule on a falling,
# negative-gamma day (2026-11-05).
SHARED_BY_ALL = (
    "position_pct", "flatten_before_close_min", "time_limit_minutes", "skip_events",
    "no_buy_down_negative_gamma",
)
# What Orchestra sets on a GoodMoments racer, over its pair's own configuration
# (`build_orchestra_config`): the edge a buy needs, the take, the stop and the
# momentum confirmation -- and on a hybrid the buy distance under its levels
# model's predicted high.
SIGNAL_PAIR_FIELDS = ("gm_enter", "gm_take_k", "gm_stop_k", "gm_confirm_bars")
HYBRID_PAIR_FIELDS = SIGNAL_PAIR_FIELDS + ("buy_k",)
# The pair numbers counted in bars rather than in units.
_BAR_FIELDS = ("momentum_confirmation_bars", "gm_confirm_bars")


def is_signal(config_or_key) -> bool:
    """Whether a racer trades GoodMoments' rules (an edge read every bar)
    rather than the day-range rules (two resting levels)."""
    model_key = (
        split_key(config_or_key)[1] if isinstance(config_or_key, str) else config_or_key.model_key
    )
    return (
        model_key in apple_models.MODELS
        and apple_models.strategy(model_key) == apple_models.STRATEGY_PROBABILITY
    )


def pair_fields(model_key: str) -> "tuple[str, ...]":
    """The numbers a racer is given by Orchestra: a day-range one's, or a
    GoodMoments one's (the hybrid's buy distance too)."""
    if apple_models.strategy(model_key) != apple_models.STRATEGY_PROBABILITY:
        return DAYRANGE_PAIR_FIELDS
    return HYBRID_PAIR_FIELDS if apple_models.levels_of(model_key) else SIGNAL_PAIR_FIELDS


def _pair_numbers(model_key: str, numbers: "dict | None") -> dict:
    """The pair numbers in `numbers` this model's racer is given, typed."""
    return {
        name: int(value) if name in _BAR_FIELDS else float(value)
        for name, value in (numbers or {}).items() if name in pair_fields(model_key)
    }


def racer_key(config: AppleTraderConfig) -> str:
    """A racer's identity: its symbol and its model. Stable for a whole run --
    a racer never switches model, since the model is half of what it is."""
    return f"{config.ticker}:{config.model_key}"


def split_key(key: str) -> "tuple[str, str]":
    ticker, _, model_key = str(key).partition(":")
    return ticker.upper(), model_key


# What a racer's model is called beside its symbol. The registry's labels are
# written for a picker ("HighLow2 model") and are too long to repeat on every
# log line and board row.
SHORT_MODEL_NAMES = {
    "highlow": "HighLow",
    "highlow2": "HighLow2",
    "highlow_chain": "HighLow Chain",
    "medianmove": "MedianMove",
    "fastfurious": "Fast&Furious",
    "goodmoments": "GoodMoments",
    "highlow2_goodmoments": "HighLow2 + GoodMoments",
    "highlow_goodmoments": "HighLow + GoodMoments",
}


def racer_label(config_or_key) -> str:
    """"INTC · HighLow": what the log, the board and the form call a racer."""
    # A key is the string; anything else is a config, of whichever class --
    # see `apple_trader.as_config`.
    if isinstance(config_or_key, str):
        ticker, model_key = split_key(config_or_key)
    else:
        ticker, model_key = config_or_key.ticker, config_or_key.model_key
    # A removed model keeps its key rather than borrowing the default's label.
    name = SHORT_MODEL_NAMES.get(model_key) or (
        apple_models.MODELS[model_key].label if model_key in apple_models.MODELS else model_key
    )
    return f"{ticker} · {name}"


def all_pairs() -> "list[str]":
    """Every (ticker, model) pair Orchestra can be given, as racer keys: each
    symbol some model covers, with each day-range and GoodMoments model it
    has, in picker order. Fast&Furious drives its own rules (a forecast of the
    next half hour's momentum and a clock), which neither the race nor the
    09:34 selection is written for."""
    return [
        f"{ticker}:{key}"
        for ticker in apple_models.tickers()
        for key in apple_models.keys_for(ticker)
        if apple_models.strategy(key) in STRATEGIES
    ]


def default_pairs() -> "list[str]":
    """The pairs Orchestra starts from, in picker order: the ones SimLab's ML
    summary set (`orchestra_defaults`), else every pair SimLab's tuning has
    picked levels for (`config.APPLE_TRADER_TUNED_LEVELS`). A pair nobody
    tuned is offered, not chosen; a stored pair Orchestra no longer offers (a
    removed model) is dropped, and with none left the tuned pairs are back."""
    offered = all_pairs()
    stored = set(orchestra_defaults.pairs() or ())
    chosen = [key for key in offered if key in stored]
    if chosen:
        return chosen
    tuned = {f"{ticker}:{model}" for model, ticker in APPLE_TRADER_TUNED_LEVELS}
    return [key for key in offered if key in tuned]


def pairs_widget() -> str:
    """The pair picker's widget name in a form (`FormCopy.key`): carries the
    stored default's digest (`orchestra_defaults.key_suffix`), so a new
    default re-seeds the picker of every open page."""
    return f"pairs{orchestra_defaults.key_suffix()}"


# --- the order buys are offered in ---------------------------------------------

# `tuning_profits`' last answer, and the state of the stores it was read off.
_profits_cache: "tuple[tuple, dict[str, float]] | None" = None


def _profits_stamp(tuning_dir) -> tuple:
    """What `tuning_profits` depends on, cheaply: every tuning job file's name
    and mtime, and the base configurations' store's."""
    from . import base_configs

    def mtime(path) -> "int | None":
        try:
            return path.stat().st_mtime_ns
        except OSError:
            return None

    jobs = tuple(sorted((p.name, mtime(p)) for p in tuning_dir.glob("*.json")))
    return jobs, mtime(base_configs.BASE_PATH)


def tuning_profits() -> "dict[str, float]":
    """Each pair's average profit a session in SimLab's tuning, by racer key:
    what SimLab's ML summary reads for it over all its tuning data
    (`simlab.ml_summary.pair_metrics`) -- every replay of the pair's current
    configuration (the one "Set as current configuration" stored), every
    session counted, traded or not; for a pair with none stored, its latest
    job's base. A pair no tuning job has replayed is not in it.

    Read again only when a job file or the stored configurations change, so a
    form can ask on every rerun. Empty, never an error, if the store cannot be
    read: the order buys are offered in is not worth failing a start over."""
    global _profits_cache
    try:
        from simlab import ml_summary
        from simlab import tuning as sim_tuning

        stamp = _profits_stamp(sim_tuning.TUNING_DIR)
        if _profits_cache is not None and _profits_cache[0] == stamp:
            return dict(_profits_cache[1])
        found: "dict[str, float]" = {}
        for series in ml_summary.base_series(sim_tuning.list_jobs()):
            days = sorted(series.daily)
            row = ml_summary.pair_metrics(series, (days[0], days[-1])) if days else None
            if row is not None:
                found[f"{series.ticker}:{series.model_key}"] = float(row["per_session"])
    except Exception as exc:  # noqa: BLE001 -- see the docstring
        logger.warning("Could not read the pairs' tuning profits: %s", exc)
        return {}
    _profits_cache = (stamp, found)
    return dict(found)


def rank_pairs(pairs: "list[str]", profits: "dict[str, float]") -> "list[str]":
    """`pairs` (racer keys) the most profitable a session first; a pair with no
    tuning profit after every one with one, in the order given."""
    return sorted(pairs, key=lambda key: (key not in profits, -profits.get(key, 0.0)))


# --- the configuration --------------------------------------------------------


@dataclass
class OrchestraConfig:
    """The racers, in the order they are read every bar -- in a race the order
    buys are offered in: when two fill on the same bar the earlier racer takes
    its share of the cash first, and the next one what is left.
    `build_orchestra_config` puts a race's in the order of their tuning profit
    a session (`profit_per_session`)."""

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
    # In a race of several pairs: whether one pair's stop-out ends every
    # pair's buying for the day -- the live agent stops -- rather than standing
    # only that pair down. What every record made before 2026-10-27 means,
    # which SimLab decodes so (`simlab.rule_agents`); off since, the user's
    # request. A race of one stops on its stop-out either way, as an Apple
    # Trader run does, and a split never does.
    halt_on_stop: bool = False
    # In a race: whether only one pair may hold at a time -- the first to fill,
    # until its position closes -- rather than every pair whose buy fills while
    # a share of it fits in the cash left (one pair a symbol). What every
    # record made before 2026-10-28 means, which SimLab decodes so; off since,
    # the user's request.
    one_holder: bool = False
    # In a race: each pair's average profit a session in SimLab's tuning
    # (`tuning_profits`), by racer key, as the racers were put in order by
    # when the race was built -- what the form and the board show beside each.
    # A pair without one is not in it. Empty in a split, and in a record made
    # before 2026-10-28 (raced in the order its pairs were picked in).
    profit_per_session: "dict[str, float]" = field(default_factory=dict)

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
        other = [r for r in self.racers if r.strategy not in STRATEGIES]
        if other:
            raise ValueError(
                f"Orchestra races the day-range and GoodMoments rules, and "
                f"{racer_label(other[0])} runs its own: run it as Apple Trader"
            )
        keys = [racer_key(r) for r in self.racers]
        twice = sorted({k for k in keys if keys.count(k) > 1})
        if twice:
            raise ValueError(f"{', '.join(twice)} is in Orchestra more than once")
        # What every pair shares, then what the day-range pairs share among
        # themselves: a GoodMoments pair keeps the rest of its own rules.
        ranged = [r for r in self.racers if not is_signal(r)]
        for group, shared in ((self.racers, _common), (ranged, _shared)):
            if not group:
                continue
            first = shared(group[0])
            for racer in group[1:]:
                if shared(racer) != first:
                    differ = sorted(k for k in first if first[k] != shared(racer)[k])
                    raise ValueError(
                        f"every pair shares Orchestra's rules, but {racer_key(racer)} differs "
                        f"from {racer_key(group[0])} in {', '.join(differ)}"
                    )
        if self.mode != MODE_SPLIT:
            self.budget_pct = {}
            given = self.profit_per_session or {}
            self.profit_per_session = {
                key: float(given[key]) for key in keys if given.get(key) is not None
            }
            return
        self.profit_per_session = {}
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


def _common(config: AppleTraderConfig) -> dict:
    return {k: getattr(config, k) for k in SHARED_BY_ALL}


def signal_racer(
    ticker: str, model_key: str, base: AppleTraderConfig, numbers: "dict | None" = None,
) -> AppleTraderConfig:
    """A GoodMoments racer: its pair's own configuration (`default_config`:
    the pair's base, else the shipped defaults; a hybrid's from both its
    models), with Orchestra's sizing, flatten, days off, time limit and
    negative-gamma rule from `base` (`SHARED_BY_ALL`) and the numbers Orchestra sets on it (`pair_fields`)
    from `numbers`, where given.

    A take that follows the intraday volatility curve keeps its quietest-minute
    take unless the take set here is under it, when it is cut to the take --
    flat, that is, until the take is raised again. A take of 0 is off, and so
    is its curve."""
    pair = at.default_config(ticker, model_key)
    given = _pair_numbers(model_key, numbers)
    take = given.get("gm_take_k", float(pair.gm_take_k))
    take_min = pair.gm_take_min_k
    if take_min is not None:
        take_min = min(float(take_min), take) if take else None
    return replace(
        pair, ticker=ticker, model_key=model_key,
        **{name: getattr(base, name) for name in SHARED_BY_ALL},
        **given, gm_take_min_k=take_min,
    )


def build_orchestra_config(
    pairs: "list[str]",
    base: AppleTraderConfig,
    levels: "dict | None" = None,
    selection: "candidates.SelectionRules | None" = None,
    mode: str = MODE_RACE,
    budget_pct: "dict[str, float] | None" = None,
    profits: "dict[str, float] | None" = None,
) -> OrchestraConfig:
    """One racer per pair, `levels[key]` = `{field: value}` of the numbers
    Orchestra sets on it (`pair_fields`). A day-range pair is `base` -- its
    circuit breaker too, one for every pair -- with its own two distances,
    stop and momentum confirmation period: those given, the pair's own for the
    rest (its base configuration's, else the shipped ones). A GoodMoments pair is
    its own configuration with the numbers given (`signal_racer`).
    `selection`, `mode` and `budget_pct` as `OrchestraConfig`'s.

    A race's pairs are put in the order of `profits` -- each pair's tuning
    profit a session (`tuning_profits`), the most profitable first
    (`rank_pairs`) -- which is the order its buys are offered in; a split's
    keep the order given, where nobody waits for anybody."""
    levels = levels or {}
    profits = {k: float(v) for k, v in (profits or {}).items() if v is not None}
    if mode != MODE_SPLIT:
        pairs = rank_pairs(list(pairs), profits)
    racers = []
    for key in pairs:
        ticker, model_key = split_key(key)
        if is_signal(key):
            racers.append(signal_racer(ticker, model_key, base, levels.get(key)))
            continue
        buy_k, sell_k = at.dayrange_levels(ticker, model_key)
        own = {
            "buy_k": float(buy_k), "sell_k": float(sell_k),
            "stop_gain_fraction": at.stop_for(ticker, model_key),
            "momentum_confirmation_bars": at.confirmation_for(ticker, model_key),
        }
        own.update(_pair_numbers(model_key, levels.get(key)))
        racers.append(replace(base, ticker=ticker, model_key=model_key, **own))
    return OrchestraConfig(
        racers, selection=selection, mode=mode, budget_pct=budget_pct or {},
        profit_per_session=profits,
    )


def orchestra_signature(race: OrchestraConfig) -> str:
    """Compact identity of an Orchestra setup, standing in for a model name in SimLab.

    The racers in order -- a day-range pair as its model, symbol, two distances
    and circuit breaker, since those are what the pair tunes; a GoodMoments
    pair as its whole rule set, as an Apple Trader run on it signs it up to
    the size -- then the shared rules once, read off the first day-range
    racer's own signature (the first racer's, with none) with its per-pair
    part removed, so a rule signs here exactly as it does in an Apple Trader
    run. A split says so up front and adds each pair's share of the budget
    ("=40%"); a race signs as it always has.

    The day-range pairs' momentum confirmation and stop are each signed with
    the shared rules (`confirm=3b`, `stop=E-0.5G`) while every pair has the
    same one -- every record from before they were set per pair -- and
    otherwise on each pair that has one ("/conf3b", "/stop0.5G"). The rules
    are then read off the first pair that has the ones that differ, so what
    they drive -- the take, the stop under the fill on a ladder -- is in them.

    A race of several pairs where a stop-out stands only that pair down adds
    ",stop_out=pair" -- when a day-range pair has a stop at all -- so it is
    not confused with a record from before, when a stop-out ended every
    pair's day (`OrchestraConfig.halt_on_stop`), which signs as it did.

    A race of pairs on more than one symbol where several may hold at once adds
    ",hold=several" after that, for the same reason: a record from before
    2026-10-28 let only the first to fill hold (`OrchestraConfig.one_holder`),
    and signs as it did. The order of the pairs is the order buys are offered
    in, so it is signed already.
    """
    ranged = [r for r in race.racers if not is_signal(r)]
    own_confirmation = len({r.momentum_confirmation_bars for r in ranged}) > 1
    own_stop = len({(r.stop_gain_fraction, r.stop_k) for r in ranged}) > 1

    def pair(r: AppleTraderConfig) -> str:
        if is_signal(r):
            single = at.config_signature(r)
            token = single[:single.index(",size=")] + ")"
        else:
            token = (
                f"{r.model_key}_{r.ticker}@{r.buy_k:g}/{r.sell_k:g}"
                f"{LEVEL_UNIT_TOKENS[r.level_unit]}"
                + (f"/mw{r.min_win_k:g}" if r.min_win_k else "")
                + (
                    f"/conf{r.momentum_confirmation_bars}b"
                    if own_confirmation and r.momentum_confirmation_bars else ""
                )
                + (_stop_token(r) if own_stop else "")
            )
        return token + (f"={race.budget_pct[racer_key(r)]:g}%" if race.split else "")

    pairs = ",".join(pair(r) for r in race.racers)
    head = "split:" if race.split else ""
    # The first pair with the most of what differs; with nothing differing,
    # the first pair.
    first = max(
        ranged,
        key=lambda r: (own_confirmation and bool(r.momentum_confirmation_bars))
        + (own_stop and r.has_stop),
        default=race.racers[0],
    )
    single = at.config_signature(first)
    rules = single[single.index("size="):-1]
    rules = re.sub(r",min_win=[^,)]*", "", rules)
    if own_confirmation:
        rules = re.sub(r",confirm=\d+b", "", rules)
    if own_stop:
        rules = re.sub(r",stop=E-[^,)]*", "", rules)
    if (
        not race.split and len(race.racers) > 1 and not race.halt_on_stop
        and any(r.has_stop for r in ranged)
    ):
        rules += ",stop_out=pair"
    if not race.split and not race.one_holder and len(race.tickers) > 1:
        rules += ",hold=several"
    return f"orchestra[{head}{pairs}]({rules}{candidates.rules_signature(race.selection)})"


def _stop_token(r: AppleTraderConfig) -> str:
    """A day-range pair's own stop in its signature, as the rules would sign
    it: a share of the predicted gain, or a legacy record's ADRs. Nothing
    without one."""
    if r.stop_gain_fraction:
        return f"/stop{r.stop_gain_fraction:g}G"
    return f"/stop{r.stop_k:g}A" if r.stop_k else ""


def racer_terms(config: AppleTraderConfig) -> str:
    """A racer's own numbers, as the log's opening line lists them."""
    if not is_signal(config):
        return (
            f"buy {config.buy_k:g} / sell {config.sell_k:g} × {config.unit_phrase}, "
            + (
                f"stop {config.stop_gain_fraction:g} × the predicted gain"
                if config.stop_gain_fraction else "no stop"
            )
            + ", " + _confirmation_terms(config.momentum_confirmation_bars)
        )
    take = f"{config.gm_take_k:g}" if config.gm_take_k else "off"
    if config.gm_take_k and config.gm_take_min_k is not None:
        take = f"{config.gm_take_k:g}..{config.gm_take_min_k:g} with volatility"
    text = (
        f"edge {config.gm_enter:+g}, take {take} / stop "
        + (f"{config.gm_stop_k:g}" if config.gm_stop_k else "off") + " ADR"
    )
    if config.gm_confirm_bars:
        text += ", " + _confirmation_terms(config.gm_confirm_bars)
    parent = apple_models.levels_of(config.model_key)
    if parent is not None:
        text += (
            f", under {SHORT_MODEL_NAMES.get(parent, parent)}'s buy {config.buy_k:g} × "
            f"{config.unit_phrase} after the opening window"
        )
    return text


def _confirmation_terms(bars: int) -> str:
    """A racer's momentum confirmation period, as `racer_terms` lists it."""
    if not bars:
        return "no momentum confirmation"
    return f"momentum confirmed over {bars} bar{'' if bars == 1 else 's'}"


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
    """The shared ledger as one racer of a race sees it.

    A holder sees its symbol's position as it is -- only one racer holds a
    symbol. Every other racer sees no position at all -- a racer on a
    holder's symbol would otherwise take that position for its own and manage
    it with its own stop. Fills are reported to Orchestra, which is how it
    learns who holds and when a position is closed. Everything else (cash --
    what is left of it, while others hold -- and the snapshot) is the ledger's.
    """

    def __init__(self, race: "Orchestra", key: str, tracker: DecisionTracker) -> None:
        self._race = race
        self._key = key
        self._tracker = tracker

    def position_for(self, symbol: str) -> float:
        if not self._race.holds(self._key):
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
    """The coordinator: one trader per racer, and in a race one cash balance
    between them, one holder a symbol -- in a split, one account each
    (`PairAccount`).

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
        halt_on_stop: bool = False,
        one_holder: bool = False,
        profit_per_session: "dict[str, float] | None" = None,
    ) -> None:
        if not racers:
            raise ValueError("Orchestra needs at least one racer")
        self.racers = racers
        self.by_key = {r.key: r for r in racers}
        self.mode = mode
        # Whether a stop-out ends the run (`OrchestraConfig.halt_on_stop`): a
        # race of one always, as an Apple Trader run; otherwise it stands only
        # that pair down.
        self.stops = mode != MODE_SPLIT and (halt_on_stop or len(racers) == 1)
        # In a race, whether the first pair to fill holds alone
        # (`OrchestraConfig.one_holder`); and each pair's tuning profit a
        # session, for the board.
        self.one_holder = mode != MODE_SPLIT and one_holder
        self.profit_per_session = dict(profit_per_session or {})
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
        # In a race, the racers holding a position, in the order they bought,
        # each with when (None for one adopted from an earlier run).
        self.held: "dict[str, object]" = {}
        # The ledger this cycle runs on, for the entry gate's cash check.
        self._tracker: "DecisionTracker | None" = None
        # Set by a stop-out that ends the run (`self.stops`): the live loop
        # stops the agent on it; a replay closes every racer's entries for the
        # rest of `self._stopped_day`.
        self.halt: "str | None" = None
        self._stopped_day = None
        self._day = None
        self._adopted = False
        self._last_line = None
        self._events: "list[tuple[str, str]]" = []
        # Whether this bar changed what a race holds while others still hold:
        # what the cash left may buy is then said once the bar is read
        # (`_cash_left_text`).
        self._cash_note = False
        for racer in racers:
            racer.trader.slot = RacerSlot(racer.key)
            if not self.stops:
                # A stop-out stands the pair down for the day, and a restart
                # must not undo that: the agent goes on running, so no ▶ Start
                # is the say-so to re-arm it.
                racer.trader.rearm_after_stop = False
            if not self.split:
                # In a split nobody waits for anybody.
                racer.trader.entry_gate = (lambda key=racer.key: self._gate(key))

    @property
    def keys(self) -> "list[str]":
        return [r.key for r in self.racers]

    @property
    def split(self) -> bool:
        return self.mode == MODE_SPLIT

    @property
    def holder(self) -> "str | None":
        """In a race, the racer that has held longest -- the only one, with
        `one_holder` -- or None while nobody holds."""
        return next(iter(self.held), None)

    def holds(self, key: str) -> bool:
        """Whether this racer has a position: one of the race's holders, or
        under a split a pair whose account holds shares."""
        if self.split:
            account = self.accounts.get(key)
            return account is not None and account.shares > 0
        return key in self.held

    def holders(self) -> "list[str]":
        return [key for key in self.keys if self.holds(key)]

    def _ledger(self, key: str, tracker: DecisionTracker):
        if self.split:
            return _SplitLedger(self, key, tracker)
        return _RacerLedger(self, key, tracker)

    # --- who holds -----------------------------------------------------

    def _waits_on(self, key: str) -> "tuple[str, object] | None":
        """What keeps a racer of a race from buying while others hold:
        ("holder", key) for the racer holding its symbol -- with `one_holder`,
        the one holding at all -- or ("cash", cash) for cash left that buys
        none of its shares at its last close, sized as its buy would be
        (`rule_agent.order_quantity`). None while it may buy, or holds."""
        if self.split or not self.held or key in self.held:
            return None
        racer = self.by_key[key]
        if self.one_holder:
            return "holder", self.holder
        ticker = racer.trader.ticker
        same = next((k for k in self.held if self.by_key[k].trader.ticker == ticker), None)
        if same is not None:
            return "holder", same
        price = racer.trader.last_close
        if self._tracker is not None and price:
            cash = float(self._tracker.snapshot()["cash"])
            if rule_agent.order_quantity(cash, float(price), racer.config.position_pct) <= 0:
                return "cash", cash
        return None

    def _gate(self, key: str) -> "str | None":
        if self._stopped_day is not None and self._stopped_day == self._day:
            return f"Orchestra stopped for the day when {self.halt}"
        waits = self._waits_on(key)
        if waits is None:
            return None
        kind, what = waits
        if kind == "cash":
            holding = ", ".join(self.by_key[k].label for k in self.held)
            return f"the ${what:,.2f} cash left while Orchestra holds {holding} buys not one share"
        holder = self.by_key[what]
        if not self.one_holder:
            return f"{holder.label} holds {holder.trader.ticker} already"
        since = self.held.get(what)
        since = f", bought {since:%H:%M}" if since is not None else ""
        return f"Orchestra is following {holder.label}{since}"

    def _took(self, key: str, decision) -> None:
        if key in self.held:
            return
        self.held[key] = at.clock.now().astimezone(market_hours.MARKET_TZ)
        self._events.append(("took", key))

    def _released(self, key: str) -> None:
        if key not in self.held:
            return
        del self.held[key]
        self._events.append(("released", key))

    def _adopt(self, state: AppState, tracker: DecisionTracker) -> None:
        """Who holds when Orchestra starts onto a ledger that is not flat: on
        each symbol the ledger holds, the racer an earlier run today recorded
        as holding it, else the first racer on it -- with `one_holder`, on the
        first such symbol only."""
        self._adopted = True
        prior = getattr(state, "orchestra", None) or {}
        recorded = list(prior.get("holders") or []) + [prior.get("holder")]
        held = [r for r in self.racers if tracker.position_for(r.trader.ticker) > 0]
        adopted: "dict[str, Racer]" = {}
        if self.one_holder and held:
            first = next((r for r in held if r.key in recorded), held[0])
            adopted[first.trader.ticker] = first
        elif held:
            for ticker in dict.fromkeys(r.trader.ticker for r in held):
                on_it = [r for r in held if r.trader.ticker == ticker]
                adopted[ticker] = next((r for r in on_it if r.key in recorded), on_it[0])
        for ticker, racer in adopted.items():
            self.held[racer.key] = None
            _log(state, {"type": "status", "text": (
                f"The ledger holds {tracker.position_for(ticker):g} "
                f"{ticker}; Orchestra follows {racer.label} until it is sold."
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
        today = at.day_range.market_date()
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
        for racer in self.racers:
            # A GoodMoments pair fills its first forecast's caches in the
            # background, before the bell as on Apple Trader -- here, where
            # Orchestra would otherwise not read it until the market is open
            # (and, with a selection, until 09:34). Live only, once a day.
            prewarm = getattr(racer.trader, "_prewarm", None)
            if prewarm is not None and racer.key not in self.benched:
                prewarm(racer.bundle, state)
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
        # The ledger the entry gate reads the cash left off (`_waits_on`).
        self._tracker = tracker
        # The holders first: their exits decide how much cash is in the race
        # on this very bar, and a racer read before them would be refused, or
        # sized, for a position that is about to close. Then the rest in
        # Orchestra's order, the most profitable first. A pair the selection
        # left out is not read at all -- unless it holds a position from
        # before it. (A split has neither: every pair is read, in order.)
        order = sorted(
            (r for r in self.racers if r.key not in self.benched or r.key in self.held),
            key=lambda r: r.key not in self.held,
        )
        for racer in self.racers:
            if racer.key in self.benched and racer.key not in self.held:
                racer.outcome = rule_agent.HOLD
        for racer in order:
            # A pair with a position writes its own read line; the rest share
            # one (`_race_line`).
            racer.trader.quiet_read = not alone and not self.holds(racer.key)
            ledger = self._ledger(racer.key, tracker)
            with agent_log_tag(racer=racer.label):
                racer.outcome = racer.trader.run_cycle(racer.bundle, state, ledger)
            halt = getattr(racer.trader, "halt", None)
            if halt and not self.stops:
                # That pair's own stand-down already keeps it out today; the
                # others go on -- in a split each with its own position to
                # manage, in a race to the cash this one just gave back.
                racer.trader.halt = None
                if not alone:
                    with agent_log_tag(racer=racer.label):
                        _log(state, {"type": "analysis", "text": (
                            f"{racer.label} {halt}: it sits out the rest of today. "
                            + (
                                "The other pairs keep trading their own shares of the budget."
                                if self.split else "The other pairs race on without it."
                            )
                        )})
            elif halt and self._stopped_day != today:
                # Every racer is still read for the rest of the session -- their
                # levels stay current -- but none may buy (`_gate`).
                self.halt = f"{racer.label} {halt}"
                self._stopped_day = today
            if self._events:
                self._announce(state, tracker)
        if self._cash_note:
            self._cash_note = False
            text = self._cash_left_text(tracker)
            if text:
                _log(state, {"type": "analysis", "text": text})
        if not alone:
            self._race_line(state)
        self._publish(state)
        outcomes = [r.outcome for r in self.racers]
        for tag in (rule_agent.BOUGHT, rule_agent.SOLD):
            if tag in outcomes:
                return tag
        if self.held:
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
        events, self._events = self._events, []
        if len(self.racers) == 1:
            return
        for kind, key in events:
            _log(state, {"type": "analysis", "text": self._event_text(kind, key, tracker)})

    def _event_text(self, kind: str, key: str, tracker: DecisionTracker) -> str:
        """A racer bought (`kind` "took") or sold out ("released"), in words:
        who holds now -- or, with nobody left holding, who may buy. Who may
        still buy beside a holding is said once the whole bar is read
        (`_cash_left_text`): a pair read after this one may not have its
        levels yet, or this bar's price."""
        racer = self.by_key[key]
        if kind == "took" and self.one_holder:
            others = [r.label for r in self.racers if r.key != key]
            return (
                f"{racer.label} filled first: Orchestra follows it alone until its "
                f"position is closed. Waiting: {', '.join(others)}."
            )
        held = [self.by_key[k].label for k in self.held if k != key]
        if kind != "took" and not held:
            armed = [r.label for r in self.racers if self._armed(r) and not self._benched(r)]
            return (
                f"{racer.label} is flat again, so the race is open: "
                + (f"{', '.join(armed)} can buy." if armed else "no pair is still armed today.")
            )
        self._cash_note = True
        if kind == "took":
            beside = f" beside {', '.join(held)}" if held else ""
            return f"{racer.label} filled: Orchestra holds it{beside}."
        return f"{racer.label} is flat again; Orchestra still holds {', '.join(held)}."

    def _cash_left_text(self, tracker: DecisionTracker) -> "str | None":
        """Who may still buy with the cash left beside what Orchestra holds,
        in the order buys go in -- or why no pair may. None with nobody
        holding, or no pair left to buy."""
        others = [r for r in self.racers if r.key not in self.held and not self._benched(r)]
        if not self.held or not others:
            return None
        head = f"The ${float(tracker.snapshot()['cash']):,.2f} cash left"
        armed = [r for r in others if self._armed(r)]
        free = [r.label for r in armed if self._waits_on(r.key) is None]
        if free:
            return f"{head} is open to {', '.join(free)}, in that order."
        if not armed:
            return f"{head} waits: no other pair is armed to buy now."
        return f"{head} goes to no other pair — " + "; ".join(
            self._why_not(r) for r in armed
        ) + "."

    def _why_not(self, racer: Racer) -> str:
        """Why an armed racer may not buy beside a holding, for
        `_cash_left_text`."""
        kind, what = self._waits_on(racer.key)
        if kind == "cash":
            return f"{racer.label}: a share costs more than that"
        return f"{racer.label}: {self.by_key[what].label} holds {racer.trader.ticker}"

    # --- what Orchestra looks like -------------------------------------

    @staticmethod
    def _armed(racer: Racer) -> bool:
        trader = racer.trader
        if is_signal(racer.config):
            # Reading its edge, and not past the last minute it may buy in.
            minute = getattr(trader, "last_minute", None)
            return (
                trader.blocked is None
                and getattr(trader, "session", None) is not None
                and (minute is None or minute <= APPLE_TRADER_GM_LAST_ENTRY_MINUTE)
            )
        plan = getattr(trader, "plan", None)
        return (
            trader.blocked is None
            and plan is not None
            and not plan.get("stand_down")
            and not plan.get("paused")
        )

    def _benched(self, racer: Racer) -> bool:
        return racer.key in self.benched and racer.key not in self.held

    def board(self) -> "list[dict]":
        """One row per racer: where it stands against its buy level, in its
        own level unit, which is what makes racers on different symbols and
        different prices comparable -- for a GoodMoments pair, where its edge
        stands against the edge a buy needs (and on the hybrid its levels
        model's buy level too) -- in a race, its tuning profit a session, and
        under a split, its account."""
        rows = []
        for racer in self.racers:
            trader = racer.trader
            signal = is_signal(racer.config)
            account = self.accounts.get(racer.key)
            if signal:
                plan = getattr(getattr(trader, "levels", None), "plan", None) or {}
                forecast = getattr(trader, "forecast", None) or {}
                edge = forecast.get("edge")
                entry = trader.enter_at() if getattr(trader, "session", None) is not None else None
            else:
                plan = trader.plan or {}
                edge = entry = None
            close = trader.last_close
            unit = float(plan.get("level_unit") or 0.0)
            buy = plan.get("buy_level")
            rows.append({
                "key": racer.key,
                "label": racer.label,
                "ticker": trader.ticker,
                "kind": "signal" if signal else "levels",
                "status": self._status(racer),
                "close": close,
                "buy": None if buy is None else float(buy),
                "sell": None if plan.get("sell_level") is None or signal else float(plan["sell_level"]),
                "to_buy": (
                    (close - float(buy)) / unit
                    if close is not None and buy is not None and unit > 0 else None
                ),
                "unit": unit or None,
                "edge": None if edge is None else float(edge),
                "entry": entry,
                "profit_per_session": self.profit_per_session.get(racer.key),
                "budget_pct": self.budget_pct.get(racer.key),
                "cash": None if account is None else account.cash,
                "shares": None if account is None else account.shares,
            })
        return rows

    def _status(self, racer: Racer) -> str:
        """Where one racer stands, in the board's words."""
        trader = racer.trader
        if self.holds(racer.key):
            return "holding"
        if self._benched(racer):
            return f"sits out today ({self.benched[racer.key]})"
        if self.selection is not None and self._selected_day != self._day:
            return "waiting for 09:34"
        if (trader.blocked or {}).get("sit_out"):
            return "sits out today (" + ", ".join(
                event_days.LABELS[c] for c in trader.blocked["sit_out"]
            ) + ")"
        if trader.blocked is not None:
            return "cannot forecast"
        if is_signal(racer.config):
            session = getattr(trader, "session", None)
            minute = getattr(trader, "last_minute", None)
            if session is None:
                return "waiting for the forecast"
            if minute is not None and minute < int(session.get("cool_until") or 0):
                return f"paused after a stop, until {at._session_clock(session['cool_until'])}"
            if minute is not None and minute > APPLE_TRADER_GM_LAST_ENTRY_MINUTE:
                return f"no new entries after {at._session_clock(APPLE_TRADER_GM_LAST_ENTRY_MINUTE)}"
        else:
            plan = trader.plan or {}
            if not plan:
                return "waiting for the forecast"
            if plan.get("stand_down"):
                return f"stood down ({plan['stand_down']})"
            if plan.get("paused"):
                return f"paused until the forecast is revised ({plan['paused']['headline']})"
        waits = self._waits_on(racer.key)
        if waits is not None:
            kind, what = waits
            if kind == "cash":
                return "waiting for cash"
            if self.one_holder:
                return "waiting for the holder"
            return f"waiting for {self.by_key[what].label} to sell"
        return "watching" if self.split else "racing"

    @staticmethod
    def _signal_words(row: dict) -> str:
        """A GoodMoments pair on the race line: its edge against the entry --
        and on the hybrid its close against its levels model's buy level."""
        text = f"{row['label']} edge {row['edge']:+.3f} vs entry {row['entry']:+.3f}"
        if row.get("buy") is not None and row.get("close") is not None:
            text += f", ${row['close']:,.2f} vs buy ${row['buy']:,.2f}"
        return text

    def _race_line(self, state: AppState) -> None:
        """One log line per bar for every racer not holding that may still buy
        -- in place of the read line each would write on its own -- nearest to
        its buy first: the day-range pairs by their distance to the buy level,
        the GoodMoments pairs by how far their edge is from the entry."""
        if self.held and self.one_holder:
            return
        stamps = [r.trader.last_bar_ts for r in self.racers if r.trader.last_bar_ts is not None]
        if not stamps:
            return
        newest = max(stamps)
        if newest == self._last_line:
            return
        self._last_line = newest
        racing = [row for row in self.board() if row["status"] in ("racing", "watching")]
        levels = sorted(
            (row for row in racing if row["kind"] == "levels" and row["to_buy"] is not None),
            key=lambda row: row["to_buy"],
        )
        signals = sorted(
            (row for row in racing if row["kind"] == "signal" and row["edge"] is not None),
            key=lambda row: row["entry"] - row["edge"],
        )
        parts = []
        if levels:
            parts.append("nearest to a buy first, in each pair's level unit: " + " · ".join(
                f"{row['label']} ${row['close']:,.2f} vs buy ${row['buy']:,.2f} ({row['to_buy']:+.2f} u)"
                for row in levels
            ))
        if signals:
            parts.append(
                "GoodMoments' edge, nearest to its entry first: "
                + " · ".join(self._signal_words(row) for row in signals)
            )
        if not parts:
            return
        head = "the pairs without a position" if self.split or self.held else "flat"
        _log(state, {"type": "analysis", "text": (
            f"Orchestra at {newest:%H:%M}, {head} — {'; '.join(parts)}"
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
        """The status line: a lone holder's own, else how the race stands."""
        if outcome == rule_agent.CLOSED:
            return rule_agent.WAITING, "Agent waiting for session start"
        if len(self.held) == 1:
            racer = self.by_key[self.holder]
            dot, phrase = racer.trader.activity(racer.outcome or rule_agent.HOLD, _RacerLedger(self, racer.key, tracker))
            return dot, f"{phrase} ({racer.label})"
        board = self.board()
        if self.selection is not None and self._selected_day != self._day:
            return rule_agent.WAITING, "Agent waiting to pick its candidates at 09:34"
        holding = [row["label"] for row in board if row["status"] == "holding"]
        racing = [row for row in board if row["status"] in ("racing", "watching")]
        if holding:
            # A split, or a race with several holders.
            more = f" — watching {len(racing)} more" if racing else ""
            return rule_agent.TRADING, f"Agent holding {', '.join(holding)}{more}"
        if racing:
            near = [row for row in racing if row["kind"] == "levels" and row["to_buy"] is not None]
            if near:
                best = min(near, key=lambda row: row["to_buy"])
                return rule_agent.TRADING, (
                    f"Agent watching {len(racing)} pairs — nearest: {best['label']}, "
                    f"{best['to_buy']:+.2f} × its unit from its buy level"
                )
            edges = [row for row in racing if row["kind"] == "signal" and row["edge"] is not None]
            if edges:
                best = min(edges, key=lambda row: row["entry"] - row["edge"])
                return rule_agent.TRADING, (
                    f"Agent watching {len(racing)} pairs — nearest: {best['label']}, edge "
                    f"{best['edge'] - best['entry']:+.3f} from its entry"
                )
            return rule_agent.TRADING, f"Agent watching {len(racing)} pairs"
        if any(row["status"] == "waiting for the forecast" for row in board):
            return rule_agent.WAITING, "Agent waiting for the opening forecasts"
        if any(row["status"].startswith("paused after a stop") for row in board):
            return rule_agent.WAITING, "Agent paused after a stop"
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
    ], selection=race.selection, sources=sources, mode=race.mode, budget_pct=race.budget_pct,
        halt_on_stop=race.halt_on_stop, one_holder=race.one_holder,
        profit_per_session=race.profit_per_session)


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
        halt_on_stop=race_config.halt_on_stop, one_holder=race_config.one_holder,
        profit_per_session=race_config.profit_per_session,
    )
    race = build_orchestra(race_config, bundles)
    if len(kept) > 1 and race_config.split:
        unused = 100.0 - sum(race_config.budget_pct.values())
        _log(state, {"type": "status", "text": (
            f"Orchestra trading {len(kept)} pairs at once, each on its own share of the "
            "budget: "
            + ", ".join(
                f"{racer_label(c)} {race_config.budget_pct[racer_key(c)]:g}% ({racer_terms(c)})"
                for c in kept
            )
            + (f"; {unused:g}% stays in cash" if unused > 1e-9 else "")
            + f". {_how_pairs_read(kept, split=True)} and sizes "
            f"every buy at {kept[0].position_pct:g}% of the cash its share has left. The "
            "budget is what the ledger is worth at the first open bar. A stop-out stands "
            "that pair down for the day; the others keep trading."
        )})
    elif len(kept) > 1:
        _log(state, {"type": "status", "text": (
            f"Orchestra watching {len(kept)} pairs: "
            + ", ".join(
                f"{racer_label(c)} ({racer_terms(c)}{_profit_terms(race_config, c)})" for c in kept
            )
            + f". {_how_pairs_read(kept)}; "
            + (
                "the first whose buy fills is the only one that trades until its position is "
                "closed, then the race is open again. Ties go to the earlier pair in this list."
                if race_config.one_holder else
                f"the first whose buy fills buys {kept[0].position_pct:g}% of the cash, and "
                "any other whose buy fills while a share fits in what is left buys "
                f"{kept[0].position_pct:g}% of that, one pair a symbol. Buys on the same bar "
                "go in this list's order"
                + (
                    ", the most profitable a session in SimLab's tuning first."
                    if race_config.profit_per_session else "."
                )
            )
            + " A stop-out "
            + (
                "stops the agent" if race.stops
                else "stands that pair down for the day, and the others race on"
            )
            + (
                " (a GoodMoments stop pauses only that pair, for "
                f"{APPLE_TRADER_GM_COOLDOWN_BARS} bars)."
                if any(is_signal(c) for c in kept) else "."
            )
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
                config, apple_models.get(config.model_key), bundles[key],
                stops_agent=race.stops,
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


def _profit_terms(race: OrchestraConfig, config: AppleTraderConfig) -> str:
    """A racer's tuning profit a session, for the opening line's list."""
    made = race.profit_per_session.get(racer_key(config))
    return "" if made is None else f"; ${made:,.2f} a session in tuning"


def _how_pairs_read(kept: "list[AppleTraderConfig]", split: bool = False) -> str:
    """How the pairs find their buys, for the loop's opening line: each
    day-range pair's levels from its 9:35 forecast, each GoodMoments pair's
    edge at every bar."""
    signals = [c for c in kept if is_signal(c)]
    if not signals:
        return (
            "Each forecasts its own session at 9:35, rests its own levels"
            if split else "Each forecasts its own session at 9:35 and rests its own levels"
        )
    if len(signals) == len(kept):
        return "Each reads its own edge at the close of every bar"
    return (
        "Each day-range pair forecasts its own session at 9:35 and rests its own levels, "
        "each GoodMoments pair reads its own edge at the close of every bar"
        + ("," if split else "")
    )


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
        # Who held when an earlier run today stopped: whom a race follows on
        # a ledger that still holds their symbols (`Orchestra._adopt`).
        "holder": prior.get("holder"),
        "holders": list(prior.get("holders") or []),
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
