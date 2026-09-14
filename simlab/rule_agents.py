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

The ticker is asked of the *config* rather than of the agent, because both of
them pick their instrument -- Apple Trader from the symbols its chosen model was
fitted on, Apple Trader 2 from anything the dataset carries. Both are still
single-symbol per run, which is what the dataset check depends on.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Optional

from agent_stonks import apple_models
from agent_stonks.config import APPLE_TRADER_BUY_K, APPLE_TRADER_SELL_K
from agent_stonks.apple_trader import (
    APPLE_TRADER_AVATAR,
    APPLE_TRADER_KEY,
    APPLE_TRADER_LABEL,
    ENTRY_CONFIRM,
    AppleTraderConfig,
    build_trader,
    config_error,
    model_ticker_error,
)
from agent_stonks.apple_trader import DEFAULT_TICKER as APPLE_TRADER_TICKER
from agent_stonks.apple_trader import config_signature as apple_config_signature
from agent_stonks.apple_trader2 import (
    APPLE_TRADER2_AVATAR,
    APPLE_TRADER2_KEY,
    APPLE_TRADER2_LABEL,
    AppleTrader2Config,
)
from agent_stonks.apple_trader2 import DEFAULT_TICKER as APPLE_TRADER2_TICKER
from agent_stonks.apple_trader2 import build_trader as build_apple2
from agent_stonks.apple_trader2 import config_error as apple2_config_error
from agent_stonks.apple_trader2 import config_signature as apple2_config_signature
from agent_stonks.apple_trader2 import load_bundles as apple2_load_bundles


@dataclass(frozen=True)
class RuleAgent:
    """One rule-based agent as the rest of SimLab sees it."""

    key: str
    label: str
    avatar: str
    # config -> the single symbol that config trades. A rule agent is
    # single-symbol per run by design, and a dataset without that symbol is a
    # configuration mistake worth catching before the run produces an empty
    # ledger.
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
    model a config names decides whether the run is the momentum rules or the
    day-range ones. A missing bundle would otherwise surface as a run that
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


def _apple_signature(config: AppleTraderConfig) -> str:
    # `prob_threshold=None` means "whatever cut-off the model chose", so the
    # signature needs that model to name the configuration it actually ran --
    # but only where a cut-off is a thing that exists. The day-range rules have
    # no threshold, and asking for one would load a 200 MB bundle to answer a
    # question its signature never asks.
    # The ticker has to be passed: since TimeToChange2 was re-run per symbol
    # each bundle picks its own cut-off, so asking without one would sign a
    # GOOGL run with AAPL's threshold -- while the trader, which reads the
    # bundle it actually loaded, ran on GOOGL's.
    threshold = (
        apple_models.threshold(config.model_key, ticker=config.ticker)
        if apple_models.is_momentum(config.model_key)
        else None
    )
    return apple_config_signature(config, model_threshold=threshold)


# What Apple Trader's rule set meant before a field existed to say otherwise.
# A stored record is a description of a run that already happened, so a missing
# key has to decode to the behaviour of the day it was written -- not to
# today's default, which would silently replay a record under a different
# strategy and file it in Results beside the original as though it matched.
_APPLE_LEGACY = {
    "model_key": "persistence",
    "entry_mode": ENTRY_CONFIRM,
    # Before this key existed the trailing stop was the only exit, so a record
    # without it describes a run that did not sell on a forecast reversal --
    # replaying it under today's default would file a different strategy in
    # Results beside the original.
    "reversal_threshold": None,
    # Before the day-range levels were per instrument every run used the
    # notebook's pair, so a record without them replays at that pair rather
    # than at today's per-ticker default.
    "buy_k": APPLE_TRADER_BUY_K,
    "sell_k": APPLE_TRADER_SELL_K,
}


def _apple_from_record(raw: "dict | None") -> AppleTraderConfig:
    return AppleTraderConfig(**{**_APPLE_LEGACY, **(raw or {})})


# ---------------------------------------------------------- Apple Trader 2


def _build_apple2(config: AppleTrader2Config):
    """Apple Trader 2 plus whichever bundles its rules name, or a clear failure.

    Same shape as `_build_apple` and for the same reason -- a missing model
    would otherwise produce a run that never trades, which reads like a strategy
    result rather than an installation problem. The difference is which bundles
    get loaded: here it is whatever the *rules* name, which for a set written
    out of price and momentum alone is none at all.

    No `_BundleBound` wrapper: this trader's `run_cycle` already takes only
    (state, tracker), since the bundles it needs are decided by the rules and
    held on the trader.
    """
    bundles = apple2_load_bundles(config)
    mismatch = apple2_config_error(config, bundles)
    if mismatch is not None:
        raise RuntimeError(mismatch)
    return build_apple2(config, bundles)


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
        signature=_apple_signature,
        to_record=asdict,
        from_record=_apple_from_record,
    ),
    APPLE_TRADER2_KEY: RuleAgent(
        key=APPLE_TRADER2_KEY,
        label=APPLE_TRADER2_LABEL,
        avatar=APPLE_TRADER2_AVATAR,
        ticker=lambda config: config.ticker,
        default_ticker=APPLE_TRADER2_TICKER,
        build=_build_apple2,
        signature=apple2_config_signature,
        # The config owns its own JSON shape here rather than falling out of
        # `asdict`: it nests (a rule set of action items of conditions), and the
        # decoder has to rebuild those dataclasses rather than hand the trader
        # dicts. There is no legacy-defaults map to go with it, because no
        # record of this agent predates any of its fields.
        to_record=AppleTrader2Config.to_record,
        from_record=AppleTrader2Config.from_record,
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
