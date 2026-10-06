"""The rule-based (non-LLM) agents SimLab can replay, behind one interface.

Every other agent SimLab runs is a prompt handed to a model, so the engine can
treat them as one thing. A rule agent is instead a hand-written state machine
with its own tunables, its own dependencies and -- in Apple Trader's case -- a
saved model to load first. This module is where those differences are
absorbed, so the engine, the runner and the UI branch on "is this rule-based?"
and never on *which* rule agent it is.

A `RuleAgent` supplies everything the rest of SimLab needs about one of them:
what to call it, the one ticker a given config trades, how its config survives a
round trip through the JSON experiment record, how to name that config in
Results, and how to build a trader exposing a uniform
``run_cycle(state, tracker)`` -- which is what hides Apple Trader's extra
``bundle`` argument from the day loop.

The ticker is asked of the *config* rather than of the agent, because the agent
picks its instrument -- Apple Trader from the symbols its chosen model was
fitted on. Apple Trader is single-symbol per run; Orchestra
(`agent_stonks.orchestra`) trades several, which is why the dataset check asks
for `tickers` -- every symbol a config may trade -- rather than the one.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Callable, Optional

from agent_stonks import apple_models
from simlab.session_context import ReplaySources
from agent_stonks.config import (
    APPLE_TRADER_BUY_K,
    APPLE_TRADER_SELL_K,
    BREACH_OFF,
    LEVELS_DAYRANGE,
    UNIT_ADR,
)
from agent_stonks.apple_trader import (
    APPLE_TRADER_AVATAR,
    APPLE_TRADER_KEY,
    APPLE_TRADER_LABEL,
    AppleTraderConfig,
    build_trader,
    config_error,
    model_ticker_error,
)
from agent_stonks.apple_trader import DEFAULT_TICKER as APPLE_TRADER_TICKER
from agent_stonks.apple_trader import config_signature as apple_config_signature
from agent_stonks.candidates import SelectionRules
from agent_stonks.orchestra import (
    ORCHESTRA_AVATAR,
    ORCHESTRA_KEY,
    ORCHESTRA_LABEL,
    OrchestraConfig,
    build_orchestra,
    build_orchestra_config,
    default_pairs,
    load_racers,
    orchestra_signature,
    racer_label,
)


@dataclass(frozen=True)
class RuleAgent:
    """One rule-based agent as the rest of SimLab sees it."""

    key: str
    label: str
    avatar: str
    # config -> the symbol that config trades (the first, for Orchestra). A
    # dataset without it is a configuration mistake worth catching before the
    # run produces an empty ledger.
    ticker: Callable[[Any], str]
    # The symbol a config that has not been built yet would trade -- what a
    # description page names, and the fallback wherever there is no config in
    # hand.
    default_ticker: str
    # config -> a trader exposing `run_cycle(state, tracker)`. May raise
    # RuntimeError when something the agent depends on is not installed.
    build: Callable[[Any], Any]
    # config -> the compact identity Results groups and de-duplicates runs on,
    # standing in for `provider/model` on the LLM agents.
    signature: Callable[[Any], str]
    # config <-> the JSON-ready dict stored on the experiment record.
    to_record: Callable[[Any], dict]
    from_record: Callable[[Optional[dict]], Any]
    # config -> every symbol that config may trade, in order: what a dataset
    # must carry and what the run is handed. One for a single-symbol agent.
    tickers: Optional[Callable[[Any], "list[str]"]] = None

    def symbols(self, config) -> "list[str]":
        return list(self.tickers(config)) if self.tickers is not None else [self.ticker(config)]


class _BundleBound:
    """Apple Trader plus the loaded bundle its cycle takes as an argument.

    Exists only so the engine's day loop can call `run_cycle(state, tracker)`
    on any rule agent without knowing which one it is holding.
    """

    def __init__(self, trader: Any, bundle: dict) -> None:
        self.trader = trader
        self.bundle = bundle

    def run_cycle(self, state, tracker) -> str:
        return self.trader.run_cycle(self.bundle, state, tracker)


# ------------------------------------------------------------- Apple Trader


def _build_apple(config: AppleTraderConfig) -> _BundleBound:
    """The Apple Trader state machine plus the model its config names, or a
    clear failure.

    Which state machine is `build_trader`'s decision, not this module's: the
    model a config names decides which rules run. A missing bundle would otherwise surface as a run that
    simply never trades, which reads like a strategy result rather than the
    installation problem it is.
    """
    # The model/instrument pairing is checked first: a model that was never
    # fitted on this symbol is a different problem from one whose file is
    # missing, and reporting it as the latter sends the reader looking for a
    # file that was never meant to exist.
    pairing = model_ticker_error(config)
    if pairing is not None:
        raise RuntimeError(pairing)
    bundle = apple_models.load(config.model_key, config.ticker)
    if bundle is None:
        raise RuntimeError(
            apple_models.unavailable_reason(config.model_key, config.ticker)
        )
    mismatch = config_error(config, bundle)
    if mismatch is not None:
        raise RuntimeError(mismatch)
    return _BundleBound(build_trader(config, bundle), bundle)


# What Apple Trader's rule set meant before a field existed to say otherwise.
# A stored record is a description of a run that already happened, so a missing
# key has to decode to the behaviour of the day it was written -- not to
# today's default, which would silently replay a record under a different
# strategy and file it in Results beside the original as though it matched.
_APPLE_LEGACY = {
    # Before this key existed Apple Trader only ran the persistence classifier.
    # That model has since been removed, and a record naming it is refused
    # (`model_ticker_error`) rather than replayed on the model that is left.
    "model_key": "persistence",
    # Before the day-range levels were per instrument every run used the
    # notebook's pair, so a record without them replays at that pair rather
    # than at today's per-ticker default.
    "buy_k": APPLE_TRADER_BUY_K,
    "sell_k": APPLE_TRADER_SELL_K,
    # Before the managed exit existed a position left only at the sell level or
    # the closing flatten. Zero switches the stop and the momentum take off, so
    # such a record replays -- and signs -- exactly as it was run.
    "stop_k": 0.0,
    # And before the stop was written against the predicted gain it was written
    # in ADRs (`stop_k`), which every record made until then carries. Zero here
    # leaves that one to speak: a record saying 0.2 ADR under the fill replays
    # at 0.2 ADR under the fill and signs `stop=E-0.2A`, rather than being
    # re-read as 0.2 of a gain it never mentioned -- a different stop, filed in
    # Results beside the original as though it matched.
    "stop_gain_fraction": 0.0,
    "momentum_drop": 0.0,
    # And before the take read a positive-to-balanced turn over N bars it read
    # a fall of `momentum_drop` sigmas from the peak, which every record made
    # until then carries. Zero here leaves that one to speak, exactly as for
    # the stop above: a record saying 1σ replays the 1σ rule and signs
    # `@mom-1`, rather than picking up today's 15-bar turn beside it.
    "momentum_fade_bars": 0,
    # And before the take read the N-bar $ momentum staying negative for a
    # streak it read that positive-to-balanced turn, which every record made
    # 2026-09-21 to -23 carries as `momentum_fade_bars`. Zero here leaves that
    # one to speak too: such a record replays the turn and signs `@fade15b`.
    "negative_momentum_bars": 0,
    # Before the intraday update existed the 9:35 forecast stood all day and
    # the two levels never moved. "off" is that rule, and it is left out of the
    # signature, so such a record replays and files exactly where it did.
    "breach_update": BREACH_OFF,
    # And before the reference was a choice, the levels always hung off the
    # predicted high -- which is still the default, so this entry changes
    # nothing today and is here so that it keeps meaning the same thing if the
    # default ever moves.
    "level_source": LEVELS_DAYRANGE,
    # And before the unit was a choice, a k was always an ADR -- which is no
    # longer the default, so unlike `level_source` above this entry is doing
    # work today. Every stored record's distances, its stop as a fraction of
    # the predicted gain, its runner threshold and its circuit breaker were all
    # counted in ADRs, and re-reading those same numbers as predicted ranges
    # would replay a different strategy under the signature of the original.
    "level_unit": UNIT_ADR,
    # Before this existed the 9:35 forecast was only ever widened by the breach
    # policy, so under "off" a predicted high the tape had traded clean through
    # stayed the number every level was measured from. False is that rule, and
    # it is left out of the signature.
    "contain_range": False,
    # And before a breach kept the unit's width (2026-09-25) "shift" kept the
    # width the range last had and "brownian" moved only the breached side.
    # False is that rule, and it is left out of the signature.
    "keep_width": False,
    # And a breach moved the forecast rather than closing anything: under
    # "brownian" the sell level was carried past the bar that breached and the
    # position rode on. False replays that, so such a record's ledger and its
    # signature are both the ones it was written with.
    "breach_exit": False,
    # Before the circuit breaker existed a session kept re-arming its levels
    # however badly the last trade went, stopping only on a stop. 0 is that
    # rule, and it is left out of the signature.
    "min_win_k": 0.0,
    # Before the ladder existed a position was bought once and nothing more was
    # bought until it had closed. False is that rule, and it is left out of the
    # signature.
    "scale_in": False,
    # And while it existed without this key (2026-09-23 to -28) the stop sat
    # under the next rung rather than under the fill. True is that rule, and
    # such a record keeps the `adds=half` signature it was filed under.
    "stop_under_next_buy": True,
    # And an add filled on any bar that reached the rung, even one closing at
    # or above the last fill (before 2026-09-28). False is that rule, and it is
    # left out of the signature.
    "add_under_fill": False,
    # And each add rested half-way between the last buy and the bottom of the
    # range rather than a fixed step under it (before 2026-10-01). 0 is that
    # rule, and such a record keeps the `adds=half` signature it was filed under.
    "buy_step_k": 0.0,
    # And an entry was never refused for the speed of the fall that reached the
    # buy level. 0 is that rule, and it is left out of the signature.
    "max_fall_k": 0.0,
    # Before the momentum confirmation (2026-09-24) nothing read the behaviour
    # table: the buy and the target sold on a touch, and the take was whichever
    # of the rules above the record carries. 0 is that, and it is left out of
    # the signature, so every stored run keeps the identity it was filed under.
    "momentum_confirmation_bars": 0,
    # And before 2026-09-29 the take fired on any profit at all, however small.
    # 0 is that rule, and it is left out of the signature. No new config sets it
    # either (the time below replaced it); a record from 09-29 to 10-02 carries
    # its 0.2 and replays that.
    "take_min_gain_fraction": 0.0,
    # And before 2026-10-02 the take fired from the first bar after the fill.
    # 0 is that rule, and it is left out of the signature.
    "take_after_minutes": 0,
    # And before 2026-10-06 the take fired in profit only, leaving a losing
    # position to the stop. False is that rule, and it is left out of the
    # signature.
    "take_in_loss": False,
    # And every buy was a market order, filling near the close of the bar that
    # reached the level even when that was above it (before 2026-09-30). False
    # is that rule, and it is left out of the signature.
    "limit_entry": False,
    # And no session was sat out for an earnings report, a CPI or jobs release
    # or a shock (before 2026-10-04). Empty is that rule, and it is left out of
    # the signature.
    "skip_events": (),
    # And nothing was traded before the run's own forecast: HighLow_3m's 9:33
    # window (2026-10-06) did not exist. False is that rule -- also today's
    # default, so this entry is here to keep meaning the same if it moves --
    # and it is left out of the signature.
    "use_3m": False,
}


# A record may still carry fields of a strategy that has been removed -- an
# entry mode, a trailing stop, a delta-momentum threshold. They describe a run
# that already happened and mean nothing to today's config, so they are dropped
# on the way in; the model key they came with is what gets the run refused.
_APPLE_FIELDS = {f.name for f in fields(AppleTraderConfig)}


def _apple_from_record(raw: "dict | None") -> AppleTraderConfig:
    merged = {**_APPLE_LEGACY, **(raw or {})}
    return AppleTraderConfig(**{k: v for k, v in merged.items() if k in _APPLE_FIELDS})


# ---------------------------------------------------------------- Orchestra


def _build_orchestra(config: OrchestraConfig):
    """Orchestra over the config's pairs, each with its model loaded and
    checked as Apple Trader's is (`_build_apple`). Any pair that cannot run
    fails the whole replay: an Orchestra quietly short of a pair is a different
    configuration from the one its signature names."""
    bundles, refusals = load_racers(config)
    if refusals:
        raise RuntimeError("; ".join(
            f"{racer_label(key)}: {reason}" for key, reason in refusals.items()
        ))
    # The 09:34 selection reads the briefings cached for each replayed day and
    # never asks for a new one (`session_context.ReplaySources`).
    selection = config.selection
    sources = (
        None if selection is None
        else ReplaySources(selection.briefing_provider, selection.briefing_model)
    )
    return build_orchestra(config, bundles, sources=sources)


def _orchestra_from_record(raw: "dict | None") -> OrchestraConfig:
    """Each racer decoded as a single Apple Trader record is -- the same
    legacy defaults for a field a later version added. No record at all is
    the default Orchestra: every tuned pair on today's rules."""
    racers = (raw or {}).get("racers") or []
    if not racers:
        return build_orchestra_config(
            default_pairs(), AppleTraderConfig(), selection=SelectionRules(),
        )
    # A record without `selection` was made before the 09:34 selection existed
    # and raced every pair all day; None replays it -- and signs it -- so.
    return OrchestraConfig(
        [_apple_from_record(r) for r in racers], selection=(raw or {}).get("selection"),
    )


# ------------------------------------------------------------------ registry

RULE_AGENTS: dict[str, RuleAgent] = {
    APPLE_TRADER_KEY: RuleAgent(
        key=APPLE_TRADER_KEY,
        label=APPLE_TRADER_LABEL,
        avatar=APPLE_TRADER_AVATAR,
        # Constrained rather than free: the symbols this agent can trade are
        # the ones its chosen model was fitted on, which `model_ticker_error`
        # enforces and the pickers offer.
        ticker=lambda config: config.ticker,
        default_ticker=APPLE_TRADER_TICKER,
        build=_build_apple,
        signature=apple_config_signature,
        to_record=asdict,
        from_record=_apple_from_record,
    ),
    ORCHESTRA_KEY: RuleAgent(
        key=ORCHESTRA_KEY,
        label=ORCHESTRA_LABEL,
        avatar=ORCHESTRA_AVATAR,
        ticker=lambda config: config.racers[0].ticker,
        tickers=lambda config: config.tickers,
        default_ticker=APPLE_TRADER_TICKER,
        build=_build_orchestra,
        signature=orchestra_signature,
        to_record=asdict,
        from_record=_orchestra_from_record,
    ),
}


def is_rule_based(personality: str) -> bool:
    """Whether this agent runs SimLab's rule day loop instead of the LLM one."""
    return personality in RULE_AGENTS


def rule_agent(personality: str) -> RuleAgent:
    agent = RULE_AGENTS.get(personality)
    if agent is None:
        raise RuntimeError(f"{personality} is not a rule-based agent.")
    return agent
