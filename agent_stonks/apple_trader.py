"""Apple Trader -- a rule-based agent with no LLM in the loop.

Every other personality in `agent_stonks.agent` is a system prompt handed to a
model that reasons its way to a decision. This one is a plain loop: once a
minute it looks at the minute bar that just closed and applies fixed rules
built on a saved model from FinNotebooks. Same paper ledger, same fill path,
same log -- only the decision-making is deterministic, so the same tape always
produces the same trades.

Two rule sets. The range models drive TimeToChange3's day-range rules
(`DayRangeTrader`): one forecast at 9:35 of where the session's high and low
will land, then two resting levels derived from it and nothing more asked of
the model all day. Fast&Furious drives the momentum rules (`FastFuriousTrader`):
a forecast of the next 30 minutes' momentum at every bar, bought when it is
extreme. GoodMoments drives the probability rules (`GoodMomentsTrader`): at
every bar the chance of a +0.2 ADR rise before the close, bought when the model
sees it likelier than the clock and volatility alone do, sold at +0.2 ADR, a
stop or the close. `AppleTraderConfig.model_key` names the model and
`build_trader` turns it into a state machine -- see `agent_stonks.apple_models`,
which owns that mapping.

Which symbol, and why it is a setting rather than a name
--------------------------------------------------------
`AppleTraderConfig.ticker` names the one symbol a run trades. Everything this
agent does is a saved model's output, so the instrument is not free the way it
is for a rule set written on the tape: a model exists for a symbol or it does
not, and `apple_models` owns that fact. The pairing is checked before the loop
starts (`config_error`) rather than discovered as a bundle that would not load,
and the picker offers only the models a symbol has.

The agent keeps its name. It is the loop that is Apple Trader, not the symbol.
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, fields, is_dataclass, replace
from datetime import timedelta, timezone
from typing import Optional

import pandas as pd

from . import agent as agent_mod
from . import state as state_mod
from . import (
    apple_models, bar_history, base_configs, candle_patterns, clock, event_days, historical,
    intraday_vol_model, market_hours, momentum_regime, rule_agent,
)
from .agent import stop_agent
from .rule_agent import BaseTrader
from .state import append_agent_log as _log
from .config import (
    APPLE_TRADER_BREACH_UPDATE,
    APPLE_TRADER_BRIEFING_WAIT_BARS as BRIEFING_WAIT_BARS,
    APPLE_TRADER_CONTAIN_RANGE,
    APPLE_TRADER_KEEP_WIDTH,
    APPLE_TRADER_BREACH_EXIT,
    APPLE_TRADER_BUY_K,
    APPLE_TRADER_CYCLE_SEC,
    APPLE_TRADER_BUY_STEP_K,
    APPLE_TRADER_DAYRANGE_LEVELS,
    APPLE_TRADER_FF_ENTRY_QUANTILE,
    APPLE_TRADER_FF_EXIT_ON_SELL_SIGNAL,
    APPLE_TRADER_FF_HOLD_MINUTES,
    APPLE_TRADER_FF_STOP_K,
    APPLE_TRADER_GM_COOLDOWN_BARS,
    APPLE_TRADER_GM_CONFIRM_BARS,
    APPLE_TRADER_GM_ENTER,
    APPLE_TRADER_GM_EXIT,
    APPLE_TRADER_GM_LAST_ENTRY_MINUTE,
    APPLE_TRADER_GM_STOP_K,
    APPLE_TRADER_GM_TAKE_K,
    APPLE_TRADER_GM_TAKE_MIN_K,
    APPLE_TRADER_FLATTEN_BEFORE_CLOSE_MIN,
    APPLE_TRADER_HOLD_MIN_GAIN_K,
    APPLE_TRADER_LEVEL_SOURCE,
    APPLE_TRADER_MIN_WIN,
    APPLE_TRADER_MIN_WIN_K,
    APPLE_TRADER_MODEL,
    APPLE_TRADER_MOMENTUM_CONFIRMATION_BARS,
    APPLE_TRADER_NEGATIVE_FOR_BARS,
    APPLE_TRADER_NEGATIVE_MOMENTUM_BARS,
    APPLE_TRADER_POSITION_PCT,
    MOMENTUM_NEUTRAL_FRACTION,
    APPLE_TRADER_SCALE_IN,
    APPLE_TRADER_SELL_K,
    APPLE_TRADER_SKIP_EVENTS,
    APPLE_TRADER_STOP_GAIN_FRACTION,
    APPLE_TRADER_TAKE_FRACTION,
    APPLE_TRADER_TAKE_AFTER_MINUTES,
    APPLE_TRADER_TAKE_IN_LOSS,
    APPLE_TRADER_TAKE_TRIGGER,
    APPLE_TRADER_TUNED_LEVELS,
    APPLE_TRADER_TUNED_STOP,
    BREACH_BROWNIAN,
    BREACH_LABELS,
    BREACH_OFF,
    BREACH_POLICIES,
    BREACH_SHIFT,
    LEVELS_DAYRANGE,
    LEVEL_SOURCES,
    LEVEL_UNITS,
    LEVEL_UNIT_LABELS,
    LEVEL_UNIT_PHRASES,
    LEVEL_UNIT_TOKENS,
    APPLE_TRADER_LEVEL_UNIT,
    APPLE_TRADER_ZONE_MIN_BARS,
    APPLE_TRADER_ZONE_WIDTH,
    TAKE_MOMENTUM,
    TAKE_TRIGGERS,
    TAKE_ZONE,
    UNIT_ADR,
    UNIT_PRED_RANGE,
    SIP_DELAY_MIN,
)
from .decisions import DecisionTracker, whole_shares
from .state import AppState

APPLE_TRADER_KEY = "apple_trader"
APPLE_TRADER_LABEL = "Apple Trader (rule-based, no LLM)"
APPLE_TRADER_AVATAR = "Multiavatar-4bcbffe68af819e050.png"

# Stands where a provider name goes for the other agents (SimLab run records,
# result grouping): this one has no LLM behind it, its rules are the "model".
RULE_PROVIDER = "rules"

# The symbol a run trades unless its config names another -- what a record
# written before the instrument was configurable means, and the one symbol
# every model here covers. See `apple_models.DEFAULT_TICKER`, the authority.
DEFAULT_TICKER = apple_models.DEFAULT_TICKER


class TraderSlot:
    """Where one trader keeps its published record and reads its sidebar form.

    A run of one trader uses the two fields `AppState` has always had for it:
    `apple_trader_config`, which the form writes, and `apple_trader_levels`,
    which the trader writes once a cycle and the chart, the session file and a
    restart read back. A race (`orchestra`) runs several traders over one
    state -- two of them on the same symbol, under different models -- so each
    gets its own slot there instead, and none overwrites another's levels, its
    open position's memory or the form it adopts edits from.

    This one is the single run's, and stateless: every trader shares it unless
    a race hands it another.
    """

    def form(self, state: AppState) -> "AppleTraderConfig | None":
        return getattr(state, "apple_trader_config", None)

    def levels(self, state: AppState) -> "dict | None":
        return getattr(state, "apple_trader_levels", None)

    def publish(self, state: AppState, record: dict) -> None:
        state.apple_trader_levels = record


SINGLE_SLOT = TraderSlot()


def base_record(model_key: "str | None", ticker: str) -> "tuple[dict | None, str]":
    """The pair's base configuration as a record, and the model it was set for,
    or `(None, "")` when it has none."""
    model_key = str(model_key or APPLE_TRADER_MODEL)
    symbol = (ticker or DEFAULT_TICKER).strip().upper()
    record = base_configs.stored(model_key, symbol)
    if record is None:
        return None, ""
    return record, model_key


def _base_number(model_key: "str | None", ticker: str, name: str) -> "float | None":
    """One number of the pair's base configuration (`base_record`), if it has
    one and the number is set there."""
    record, _ = base_record(model_key, ticker)
    value = (record or {}).get(name)
    return None if value is None else float(value)


def _tuned(table: dict, model_key: "str | None", symbol: str):
    """The pair's entry in a tuned table, or None."""
    return table.get((str(model_key or APPLE_TRADER_MODEL), symbol))


def dayrange_levels(
    ticker: str, model_key: "str | None" = None, use_base: bool = True
) -> "tuple[float, float]":
    """The `(buy_k, sell_k)` a run of this model on this symbol starts from.

    Per model and instrument, since the same distances are different prices
    under a different forecast. The pair's base configuration first -- a tuning
    result made its default from SimLab (`base_configs`) -- unless `use_base`
    is off; then SimLab's tuning pick for the pair as shipped
    (`config.APPLE_TRADER_TUNED_LEVELS`, with how far each deserves trust).
    A pair never tuned falls back to the instrument's notebook-05 sweep
    (`config.APPLE_TRADER_DAYRANGE_LEVELS`), and a symbol never swept at all
    to the notebook's specified pair. A hybrid with no base of its own takes
    its levels model's buy distance (`apple_models.levels_of`) and a sell of 0:
    it rests no sell -- its exits are its signal model's -- and 0 sits under
    every buy distance a form or a grid can reach.
    """
    symbol = (ticker or DEFAULT_TICKER).strip().upper()
    if use_base:
        buy_k = _base_number(model_key, symbol, "buy_k")
        sell_k = _base_number(model_key, symbol, "sell_k")
        if buy_k is not None and sell_k is not None:
            return buy_k, sell_k
    parent = apple_models.levels_of(model_key)
    if parent is not None:
        return dayrange_levels(symbol, parent, use_base)[0], 0.0
    tuned = _tuned(APPLE_TRADER_TUNED_LEVELS, model_key, symbol)
    if tuned is not None:
        return tuned
    return APPLE_TRADER_DAYRANGE_LEVELS.get(symbol, (APPLE_TRADER_BUY_K, APPLE_TRADER_SELL_K))


def stop_for(ticker: str, model_key: "str | None" = None, use_base: bool = True) -> float:
    """The stop a run of this model on this symbol starts from, as a share of
    the predicted gain.

    Per model and instrument, like the levels it is a share of: the pair's base
    configuration (`base_configs`) unless `use_base` is off, then SimLab's
    tuning pick for the pair as shipped (`config.APPLE_TRADER_TUNED_STOP`),
    else the one shared default.
    """
    symbol = (ticker or DEFAULT_TICKER).strip().upper()
    if use_base:
        stop = _base_number(model_key, symbol, "stop_gain_fraction")
        if stop is not None:
            return stop
    tuned = _tuned(APPLE_TRADER_TUNED_STOP, model_key, symbol)
    return APPLE_TRADER_STOP_GAIN_FRACTION if tuned is None else tuned


def min_win_for(
    ticker: str, model_key: "str | None" = None, use_base: bool = True
) -> float:
    """The circuit breaker a run on this symbol starts from, in ADRs a share.

    Per instrument because the number is only readable against that symbol's own
    `buy_k - sell_k`: the same 0.20 that lets GOOGL stand down on a bad trade
    would stand AAPL down on its best one. See `config.APPLE_TRADER_MIN_WIN`.
    With a `model_key`, the pair's base configuration (`base_configs`) answers
    first unless `use_base` is off -- a base is per model, the table per symbol.
    """
    symbol = (ticker or DEFAULT_TICKER).strip().upper()
    if use_base and model_key:
        min_win = _base_number(model_key, symbol, "min_win_k")
        if min_win is not None:
            return min_win
    return APPLE_TRADER_MIN_WIN.get(symbol, APPLE_TRADER_MIN_WIN_K)


# Fields kept only so a stored record replays as it was run (see each one's
# comment on `AppleTraderConfig`): the form has no knob for them, and next to
# the knob that replaced them they make a config the dataclass refuses. A
# configuration still using one cannot be a pair's base -- every new run of the
# pair would start from a rule nobody can set up any more. `stop_under_next_buy`
# is the one whose "on" is the legacy value.
_LEGACY_ONLY = (
    "stop_k", "negative_momentum_bars", "momentum_fade_bars", "momentum_drop",
    "max_fall_k", "take_min_gain_fraction", "stop_under_next_buy",
)


def base_error(config: "AppleTraderConfig") -> "str | None":
    """Why `config` cannot be its pair's base configuration, or None if it can."""
    used = [name for name in _LEGACY_ONLY if getattr(config, name)]
    if used:
        return (
            f"It uses {', '.join(used)} — rules kept only so old records replay, which "
            "no new run can be set up with."
        )
    return model_ticker_error(config)


def shipped_config(ticker: str, model_key: "str | None" = None) -> "AppleTraderConfig":
    """The pair's configuration as the code ships it, whatever base is stored."""
    model_key = str(model_key or APPLE_TRADER_MODEL)
    buy_k, sell_k = dayrange_levels(ticker, model_key, use_base=False)
    return AppleTraderConfig(
        model_key=model_key, ticker=ticker, buy_k=buy_k, sell_k=sell_k,
        stop_gain_fraction=stop_for(ticker, model_key, use_base=False),
        min_win_k=min_win_for(ticker, model_key, use_base=False),
    )


def default_config(ticker: str, model_key: "str | None" = None) -> "AppleTraderConfig":
    """What a new run of this model on this symbol starts from: the pair's base
    configuration (`base_record`) when one is stored, the shipped defaults
    otherwise -- and for a hybrid with no base of its own, each half's
    (`_hybrid_default`).

    A stored base today's config refuses -- a field since narrowed, a model
    since removed -- is logged and passed over rather than taking a form down.
    """
    model_key = str(model_key or APPLE_TRADER_MODEL)
    symbol = (ticker or DEFAULT_TICKER).strip().upper()
    record, _ = base_record(model_key, symbol)
    if record is None and apple_models.levels_of(model_key) is not None:
        return _hybrid_default(symbol, model_key)
    if record is not None:
        known = {f.name for f in fields(AppleTraderConfig)}
        try:
            return AppleTraderConfig(**{k: v for k, v in record.items() if k in known})
        except (TypeError, ValueError) as exc:
            logging.getLogger(__name__).warning(
                "The base configuration of %s/%s is refused: %s", model_key, symbol, exc
            )
    return AppleTraderConfig(ticker=symbol, model_key=model_key)


# What a hybrid's buy level is made of, and so what it takes from its levels
# model when it has no base of its own: the distance, its unit, and how a
# breach moves the forecast it hangs off.
_HYBRID_LEVEL_FIELDS = ("buy_k", "level_unit", "breach_update", "keep_width", "contain_range")


def _hybrid_default(symbol: str, model_key: str) -> "AppleTraderConfig":
    """A hybrid's starting point when it has no base of its own: its signal
    model's configuration as that pair would start (thresholds, exits, size,
    days off -- its base, if SimLab set one) with its levels model's buy level
    in place, so "follow GoodMoments, and HighLow2 agrees" opens on the rules
    each half is run on today. The sell is 0, as `dayrange_levels` says."""
    signal = default_config(symbol, apple_models.signal_of(model_key))
    levels = default_config(symbol, apple_models.levels_of(model_key))
    return replace(
        signal, model_key=model_key, sell_k=0.0,
        **{name: getattr(levels, name) for name in _HYBRID_LEVEL_FIELDS},
    )


@dataclass
class AppleTraderConfig:
    """Tunables of the loop.

    `ticker` and `model_key` come first because they constrain each other: a
    model exists for a symbol or it does not (`apple_models.keys_for`), so the
    pair is validated together by `model_ticker_error`. `buy_k` and `sell_k`
    are the two that change what the agent does; `position_pct` and
    `flatten_before_close_min` are sizing and housekeeping.
    """

    # Which saved model the agent runs on -- a key of `apple_models.MODELS`.
    # Its `strategy` decides the rules; see `build_trader`.
    model_key: str = APPLE_TRADER_MODEL
    # The one symbol this run trades. Not free: it has to be one the chosen
    # model was fitted on, which is why the two are checked together.
    ticker: str = DEFAULT_TICKER
    # The day-range strategy's two levels, in average daily ranges below the
    # predicted high. See `DayRangeTrader`. None -> the instrument's own swept
    # pair (`dayrange_levels`), filled in by `__post_init__`, so after
    # construction both are always floats.
    buy_k: Optional[float] = None
    sell_k: Optional[float] = None
    position_pct: float = APPLE_TRADER_POSITION_PCT
    flatten_before_close_min: int = APPLE_TRADER_FLATTEN_BEFORE_CLOSE_MIN
    # The managed exit on top of the sell level and the flatten -- see
    # `DayRangeTrader._exit`. Distances are in the same average daily range the
    # levels are written in, but measured from the fill rather than from H.
    #
    # How far under the fill a bar's low may reach before the trade is taken to
    # have gone the wrong way, as a share of what the trade is playing for --
    # the predicted gain, `(buy_k - sell_k) x ADR`, which is the gap between the
    # two levels at every minute of the session. 0.5 risks one dollar for every
    # two the target is worth. 0 switches the stop off. None -> the pair's own
    # tuned stop (`stop_for`), filled in by `__post_init__`.
    stop_gain_fraction: Optional[float] = None
    # The same stop in the units it used to be written in: ADRs under the fill,
    # with no reference to what the trade was playing for. Kept only so that a
    # stored record replays and signs exactly as the run it describes -- nothing
    # configures it any more, and a new config leaves it at 0. The two are
    # mutually exclusive (`__post_init__`); `stop_distance` reads whichever is
    # set.
    stop_k: float = 0.0
    # The momentum confirmation: the one look-back, in bars, over which both
    # sides read the user's behaviour table -- momentum as the average per-bar
    # move, its change as the average change of the 1-bar momentum, each
    # positive, neutral (under MOMENTUM_NEUTRAL_FRACTION of the ticker's
    # `abs_mean_minute_momentum`) or negative. It gates the buy at the buy
    # level, the sell at the sell level and the breach exit, and is the
    # momentum take below the sell level (`take_fraction`, `take_in_loss`).
    # Never the stop, the breakeven or the flatten. See `_momentum_read`. 0
    # switches it off, and with it the take, the runner and its breakeven.
    momentum_confirmation_bars: int = APPLE_TRADER_MOMENTUM_CONFIRMATION_BARS
    # The momentum take as it was 2026-09-23 to -24, kept only so a stored
    # record replays and signs exactly as the run it describes: gains taken
    # short of the sell level once `close - close[N bars ago]` had been
    # negative for `negative_for_bars` bars in a row since the entry. A new
    # config leaves it at 0; it cannot be set beside the confirmation.
    negative_momentum_bars: int = 0
    # How many bars in a row that momentum has to stay under zero: "negative
    # for long enough". Only read while the legacy take above is on.
    negative_for_bars: int = APPLE_TRADER_NEGATIVE_FOR_BARS
    # The two earlier forms of the take, kept only so a stored record replays
    # and signs exactly as the run it describes -- nothing configures either
    # any more, and a new config leaves both at 0. `momentum_fade_bars` is the
    # positive-to-balanced turn of the sigma score over N bars (2026-09-21 to
    # -23), `momentum_drop` the fall in sigmas of the smoothed 15-bar score
    # from its best since the entry (before that). At most one of the three
    # takes may be set (`__post_init__`), like `stop_k` with
    # `stop_gain_fraction`.
    momentum_fade_bars: int = 0
    momentum_drop: float = 0.0
    # The share of the position that take sells when a runner is kept.
    take_fraction: float = APPLE_TRADER_TAKE_FRACTION
    # How many minutes after the last fill the take may fire at all, measured
    # between bar timestamps so a replay reads it as the live loop does. 0 is
    # the take from the first bar, the rule before it existed.
    take_after_minutes: int = APPLE_TRADER_TAKE_AFTER_MINUTES
    # The gate the time above replaced (2026-09-29 to 2026-10-02), kept only so
    # a stored record replays and signs exactly as the run it describes: no take
    # until the close is this share of the predicted gain (`target_gain_k`
    # level units, like the stop) above the fill. A new config leaves it at 0,
    # any profit.
    take_min_gain_fraction: float = 0.0
    # Whether the take also fires on a position at or under its fill, once the
    # time above has passed. False is the rule before 2026-10-06 -- gains only,
    # a loss left to the stop -- which a stored record replays as. A runner kept
    # from a take at a loss has no breakeven (`_exit`).
    take_in_loss: bool = APPLE_TRADER_TAKE_IN_LOSS
    # What fires the take -- one of `config.TAKE_TRIGGERS`. "momentum" is the
    # confirmation table's take row (or whichever legacy rule a record
    # carries); "zone" a stall zone forming under the position
    # (`DayRangeTrader._zone_formed`): `zone_min_bars` bars whose closes all sit
    # inside a band `zone_width` x the ticker's typical minute move wide, that
    # length reached after the last fill -- the chart's grey "forming" box at the
    # same settings. Under "zone" the take is on whatever the confirmation
    # period, which still gates the buy and the target; the share, the runner,
    # the wait and the loss rule are the same. "momentum" is every record made
    # before 2026-10-19, and the two zone settings are read only under "zone".
    take_trigger: str = APPLE_TRADER_TAKE_TRIGGER
    zone_min_bars: int = APPLE_TRADER_ZONE_MIN_BARS
    zone_width: float = APPLE_TRADER_ZONE_WIDTH
    # The gain still left to the sell level, in ADRs above the fill, that is
    # worth keeping a runner for. Short of it the take sells everything.
    hold_min_gain_k: float = APPLE_TRADER_HOLD_MIN_GAIN_K
    # What to do when the session trades through the forecast the two levels
    # are built on -- one of `dayrange_model.BREACH_POLICIES`. "off" is the
    # notebook's rule (one forecast, held all day); "shift" moves both sides by
    # the breach, "brownian" past it by the expected excursion (both sides
    # with `keep_width`, the breached one without), "extreme" the breached side
    # only -- and the levels with it. See `_update_range`.
    breach_update: str = APPLE_TRADER_BREACH_UPDATE
    # Whether the forecast is always widened to hold what the session has
    # printed, whatever `breach_update` says (`dayrange_model.contain_session`).
    # With it on, "off" no longer keeps a predicted high the tape has traded
    # through -- and therefore agrees with "extreme" on a breached side.
    contain_range: bool = APPLE_TRADER_CONTAIN_RANGE
    # Whether "shift" and "brownian" keep the range at the level unit's width
    # through a breach -- the 9:35 predicted range, or the ADR -- widening it
    # only to hold what the session has printed (`breach_width`). Off, "shift"
    # keeps the width the range last had and "brownian" moves one side only,
    # which is what a stored record replays.
    keep_width: bool = APPLE_TRADER_KEEP_WIDTH
    # Whether a bar trading through the predicted high closes an open position.
    # Tested against the high as it stood when the bar opened, and after the
    # sell level, so a breach that also reaches the target logs as the target.
    breach_exit: bool = APPLE_TRADER_BREACH_EXIT
    # Which number the two distances above are measured below: "dayrange", the
    # predicted high itself. "intraday" (IntradayVolatility's time-of-day shape,
    # removed 2026-10-20) is only ever a stored record's, which
    # `level_source_error` refuses. See `config.LEVEL_SOURCES`.
    level_source: str = APPLE_TRADER_LEVEL_SOURCE
    # What one k is worth in dollars -- one of `config.LEVEL_UNITS`. "adr" is
    # the trailing 14-day average daily range, a fixed number for the session
    # and no part of the model's output; "pred_range" is `pred_high -
    # pred_low`, so the forecast that decides where the levels sit decides how
    # far apart they are too. Everything measured against the gap between the
    # levels -- the stop as a fraction of the predicted gain, the runner
    # threshold, the circuit breaker -- is counted in this same unit, so that
    # those comparisons stay comparisons. See `level_unit`.
    level_unit: str = APPLE_TRADER_LEVEL_UNIT
    # The session circuit breaker: a trade that closes for no more than this
    # many ADRs per share stands the agent down for the rest of the day. 0
    # switches it off. None -> the instrument's own default (`min_win_for`),
    # filled in by `__post_init__`, so after construction it is always a float.
    # See `_close_out`.
    min_win_k: Optional[float] = None
    # Whether an open position is added to at a lower level when the cash left
    # after the first buy can pay for more -- see `DayRangeTrader._add` and
    # `can_scale_in`. Where the next add rests is `buy_step_k`.
    scale_in: bool = APPLE_TRADER_SCALE_IN
    # How many level units further under the reference each buy moves the next
    # one: after `n` fills the buy distance is `buy_k + n x buy_step_k`, back to
    # `buy_k` once the position is sold out. 0 is the rule before 2026-10-01,
    # each add half-way between the last buy and the bottom of the range
    # (`reference - 1 x unit`), which a stored record replays. See `_rung_k`.
    buy_step_k: float = APPLE_TRADER_BUY_STEP_K
    # Where the stop hangs while the ladder can still add. False (every new
    # config, since 2026-09-28): under the actual fill, like a position that
    # cannot be added to -- the price paid is what the forecast *and* the
    # momentum confirmation settled on, the buy level only the forecast -- and
    # an add is placed only while its rung is above that stop. True: under the
    # next rung, as it was 2026-09-23 to -28, kept only so a record made then
    # replays and signs exactly as it was run. See `_after_fill`.
    stop_under_next_buy: bool = False
    # Whether an add must fill under the last fill: "buy again lower" taken
    # literally. The rung is a forecast level, so a first buy the momentum
    # confirmation held back until the price was already under the next rung
    # would otherwise add on the very next bar at no better a price. True for
    # every new config since 2026-09-28; False, what every record made before
    # then replays as, adds on any bar that reaches the rung.
    add_under_fill: bool = True
    # No buy -- first entry or add -- while the price has fallen more than this
    # many level units over the last `fall_bars` bars. The entry gate as it was
    # 2026-09-23 to -24, kept only so a stored record replays: a new config
    # leaves it at 0, and it cannot be set beside the momentum confirmation,
    # which decides entries now. See `DayRangeTrader._falling`.
    max_fall_k: float = 0.0
    # Whether a buy -- first entry or add -- is a limit order at the buy level
    # rather than a market order. The trigger is a bar whose low reached the
    # level, but the order goes out after that bar has closed, and on a bar
    # that dipped and recovered a market order pays the recovery: above the
    # price the rule said to buy at. With the limit it pays the level at most,
    # or buys nothing and the level stays armed for the next touch. True for
    # every new config since 2026-09-30; False, what every record made before
    # then replays as, is the market order.
    limit_entry: bool = True
    # The sessions this run sits out -- `event_days.CATEGORIES`: the day after
    # earnings, CPI and jobs-report days, market and geopolitical shocks. On
    # such a day the agent makes no forecast and rests no order
    # (`DayRangeTrader._sit_out`). Empty, what every record made before
    # 2026-10-04 replays as, trades every session.
    skip_events: "tuple[str, ...]" = APPLE_TRADER_SKIP_EVENTS
    # Fast&Furious's rules (`FastFuriousTrader`, the momentum strategy), read by
    # no other model. Every day-range field above is ignored under it, and
    # these are ignored under the day-range models. See the block in `config`.
    #
    # The entry threshold: the quantile of |forecast| over the previous 20
    # sessions' every minute the forecast has to reach.
    ff_entry_quantile: float = APPLE_TRADER_FF_ENTRY_QUANTILE
    # The stop, in multiples of the forecast 2-hour drop from the signal bar's
    # price (the range model's `down_hat`), hung under the fill. 0 switches it
    # off -- and with it the need for last night's option table.
    ff_stop_k: float = APPLE_TRADER_FF_STOP_K
    # How long a position is held after the last buy signal; a fresh one
    # restarts the clock. The notebook's 30 is its forecast horizon.
    ff_hold_minutes: int = APPLE_TRADER_FF_HOLD_MINUTES
    # Whether a forecast at or under minus the threshold closes a long -- the
    # notebook system's reversal into a short, which this long-only ledger
    # reads as a close. Off is its long-only variant.
    ff_exit_on_sell_signal: bool = APPLE_TRADER_FF_EXIT_ON_SELL_SIGNAL
    # GoodMoments' rules (`GoodMomentsTrader`, the probability strategy), read
    # by no other model -- notebook 4's design 2 by default. See the block in
    # `config`.
    #
    # The entry: the model's probability of a +0.2 ADR rise before the close
    # minus the time-and-volatility model's, at or above this.
    gm_enter: float = APPLE_TRADER_GM_ENTER
    # Sell when the edge falls under this; None (the notebook's pick) never
    # sells on the signal.
    gm_exit: Optional[float] = APPLE_TRADER_GM_EXIT
    # The take-profit over the fill and the stop under it, in ADRs (today's
    # 14-session mean log range). 0 switches either off.
    gm_take_k: float = APPLE_TRADER_GM_TAKE_K
    gm_stop_k: float = APPLE_TRADER_GM_STOP_K
    # With a number here the take follows the intraday volatility curve:
    # `gm_take_k` at the session's most volatile minute, this at its quietest,
    # in proportion between, moved every bar (`gm_take_k_at`). None (the
    # notebook's rule) keeps `gm_take_k` all day.
    gm_take_min_k: Optional[float] = APPLE_TRADER_GM_TAKE_MIN_K
    # The momentum confirmation on GoodMoments' rules: the look-back, in bars,
    # of the day-range rules' behaviour table (`MOMENTUM_TABLE`, the same
    # neutral band). An entry needs the table's "buy"; a reached take sells
    # only once momentum is no longer positive -- until then it is held, and
    # sold on the first bar that is, wherever the price is. Never the stop, the
    # edge exit or the flatten. 0 (the notebook's rule) switches it off. Its
    # own field rather than `momentum_confirmation_bars`, which every stored
    # GoodMoments record carries at the day-range default.
    gm_confirm_bars: int = APPLE_TRADER_GM_CONFIRM_BARS

    def __post_init__(self) -> None:
        # Resolved before the checks below, which need numbers -- and before
        # `dayrange_levels`, because all three read the same normalised ticker.
        self.ticker = (self.ticker or DEFAULT_TICKER).strip().upper()
        if self.min_win_k is None:
            self.min_win_k = min_win_for(self.ticker, self.model_key)
        if self.stop_gain_fraction is None:
            self.stop_gain_fraction = stop_for(self.ticker, self.model_key)
        for name in (
            "stop_k", "stop_gain_fraction", "momentum_drop", "momentum_fade_bars",
            "negative_momentum_bars", "hold_min_gain_k", "min_win_k", "max_fall_k",
            "momentum_confirmation_bars", "take_min_gain_fraction", "buy_step_k",
            "take_after_minutes",
        ):
            if getattr(self, name) < 0:
                raise ValueError(
                    f"{name} {getattr(self, name)!r} is a distance and cannot be negative "
                    "(0 is how the stop or the momentum take is switched off)"
                )
        # One stop, in one set of units. Both set is not a wider stop or a
        # narrower one, it is a config that does not say which rule it means --
        # and the only way to reach it is by hand, since `stop_k` is a legacy
        # record's field and nothing writes both.
        if self.stop_k and self.stop_gain_fraction:
            raise ValueError(
                f"stop_k {self.stop_k!r} and stop_gain_fraction "
                f"{self.stop_gain_fraction!r} are two ways of writing the same stop and "
                "only one may be set; stop_k is the legacy unit a stored record replays "
                "under, new configurations use stop_gain_fraction"
            )
        # Bar counts, whatever a form or a JSON record handed over.
        for name in (
            "negative_momentum_bars", "negative_for_bars", "momentum_fade_bars",
            "momentum_confirmation_bars",
        ):
            if float(getattr(self, name)) != int(getattr(self, name)):
                raise ValueError(
                    f"{name} {getattr(self, name)!r} is a number of bars and must be whole"
                )
            setattr(self, name, int(getattr(self, name)))
        if float(self.take_after_minutes) != int(self.take_after_minutes):
            raise ValueError(
                f"take_after_minutes {self.take_after_minutes!r} is a number of minutes "
                "and must be whole"
            )
        self.take_after_minutes = int(self.take_after_minutes)
        # A tuple in canonical order, whatever a form (a list) or a JSON record
        # handed over, so the same days always sign the same way.
        self.skip_events = event_days.normalise(self.skip_events)
        # A streak of no bars would fire on the first bar in profit, which is
        # not "negative for long enough" but no rule at all.
        if self.negative_momentum_bars and self.negative_for_bars < 1:
            raise ValueError(
                f"negative_for_bars {self.negative_for_bars!r} must be at least 1 bar "
                "while the momentum take is on"
            )
        # The same reason as the stop: two takes is not a stricter take, it is
        # a config that does not say which rule it means.
        takes = [
            name
            for name in (
                "momentum_confirmation_bars", "negative_momentum_bars",
                "momentum_fade_bars", "momentum_drop",
            )
            if getattr(self, name)
        ]
        if len(takes) > 1:
            raise ValueError(
                f"{' and '.join(takes)} are different ways of writing the momentum take "
                "and only one may be set; negative_momentum_bars, momentum_fade_bars and "
                "momentum_drop are the legacy rules a stored record replays under, new "
                "configurations use momentum_confirmation_bars"
            )
        # The same for the entry: the confirmation's table decides what a buy
        # at the buy level needs, and a second gate beside it would be a rule
        # nobody configured.
        if self.momentum_confirmation_bars and self.max_fall_k:
            raise ValueError(
                f"max_fall_k {self.max_fall_k!r} is the legacy entry gate a stored record "
                "replays under and cannot be set beside momentum_confirmation_bars "
                f"{self.momentum_confirmation_bars!r}, which decides entries itself"
            )
        if not 0 < self.take_fraction <= 1:
            raise ValueError(
                f"take_fraction {self.take_fraction!r} must be a share of the position, "
                "above 0 and at most 1"
            )
        self.take_trigger = str(self.take_trigger or TAKE_MOMENTUM)
        if self.take_trigger not in TAKE_TRIGGERS:
            raise ValueError(
                f"take_trigger {self.take_trigger!r} is not one of {', '.join(TAKE_TRIGGERS)}"
            )
        # The legacy takes are momentum rules: a zone beside one would be two
        # takes, the same refusal as two momentum takes above.
        if self.take_trigger == TAKE_ZONE and takes and takes != ["momentum_confirmation_bars"]:
            raise ValueError(
                f"take_trigger {TAKE_ZONE!r} cannot be set beside {' and '.join(takes)}, the "
                "legacy momentum take a stored record replays under"
            )
        if (
            float(self.zone_min_bars) != int(self.zone_min_bars)
            or int(self.zone_min_bars) < 2
        ):
            raise ValueError(
                f"zone_min_bars {self.zone_min_bars!r} is a whole number of bars, at least 2"
            )
        self.zone_min_bars = int(self.zone_min_bars)
        self.zone_width = float(self.zone_width)
        if not self.zone_width > 0:
            raise ValueError(
                f"zone_width {self.zone_width!r} is a band in typical minute moves and must "
                "be above 0"
            )
        # Refused rather than read as "off", unlike `updated_range`'s own
        # tolerance: that one is reached with a policy some record already
        # carries, this one is a config being built, and the earliest place a
        # typo can be reported is the best one.
        self.breach_update = str(self.breach_update or BREACH_OFF)
        if self.breach_update not in BREACH_POLICIES:
            raise ValueError(
                f"breach_update {self.breach_update!r} is not one of "
                f"{', '.join(BREACH_POLICIES)}"
            )
        self.level_source = str(self.level_source or LEVELS_DAYRANGE)
        if self.level_source not in LEVEL_SOURCES:
            raise ValueError(
                f"level_source {self.level_source!r} is not one of "
                f"{', '.join(LEVEL_SOURCES)}"
            )
        self.level_unit = str(self.level_unit or UNIT_ADR)
        if self.level_unit not in LEVEL_UNITS:
            raise ValueError(
                f"level_unit {self.level_unit!r} is not one of "
                f"{', '.join(LEVEL_UNITS)}"
            )
        # Resolved per field, so a config that names only one level still
        # gets the instrument's default for the other.
        default_buy, default_sell = dayrange_levels(self.ticker, self.model_key)
        if self.buy_k is None:
            self.buy_k = default_buy
        if self.sell_k is None:
            self.sell_k = default_sell
        # The ordering is what makes the rule a rule (buy below, sell above),
        # and catching it here means a UI cannot hand the loop a pair that
        # would buy and sell on the same bar forever.
        if self.sell_k >= self.buy_k:
            raise ValueError(
                f"sell_k {self.sell_k!r} must sit above the buy level, i.e. strictly "
                f"below buy_k {self.buy_k!r} — both are distances *below* the predicted "
                "high, so the smaller number is the higher price"
            )
        # Fast&Furious's rules, checked whatever the model.
        self.ff_entry_quantile = float(self.ff_entry_quantile)
        if not 0.0 < self.ff_entry_quantile < 1.0:
            raise ValueError(
                f"ff_entry_quantile {self.ff_entry_quantile!r} is a quantile of the last 20 "
                "sessions' forecasts and must be between 0 and 1"
            )
        self.ff_stop_k = float(self.ff_stop_k)
        if self.ff_stop_k < 0:
            raise ValueError(
                f"ff_stop_k {self.ff_stop_k!r} is a distance under the fill and cannot be "
                "negative (0 switches the stop off)"
            )
        if float(self.ff_hold_minutes) != int(self.ff_hold_minutes) or int(self.ff_hold_minutes) < 1:
            raise ValueError(
                f"ff_hold_minutes {self.ff_hold_minutes!r} is a whole number of minutes, at "
                "least 1"
            )
        self.ff_hold_minutes = int(self.ff_hold_minutes)
        self.ff_exit_on_sell_signal = bool(self.ff_exit_on_sell_signal)
        # GoodMoments' rules, checked whatever the model, like Fast&Furious's.
        self.gm_enter = float(self.gm_enter)
        if not -1.0 < self.gm_enter < 1.0:
            raise ValueError(
                f"gm_enter {self.gm_enter!r} is a difference of two probabilities and must be "
                "between -1 and 1"
            )
        if self.gm_exit is not None:
            self.gm_exit = float(self.gm_exit)
            if self.gm_exit >= self.gm_enter:
                raise ValueError(
                    f"gm_exit {self.gm_exit!r} must sit below the entry gm_enter "
                    f"{self.gm_enter!r}, or every buy would be sold on the next bar"
                )
        for name in ("gm_take_k", "gm_stop_k"):
            setattr(self, name, float(getattr(self, name)))
            if getattr(self, name) < 0:
                raise ValueError(
                    f"{name} {getattr(self, name)!r} is a distance from the fill in ADRs and "
                    "cannot be negative (0 switches it off)"
                )
        if float(self.gm_confirm_bars) != int(self.gm_confirm_bars) or int(self.gm_confirm_bars) < 0:
            raise ValueError(
                f"gm_confirm_bars {self.gm_confirm_bars!r} is a whole number of bars, 0 or "
                "more (0 switches the confirmation off)"
            )
        self.gm_confirm_bars = int(self.gm_confirm_bars)
        if self.gm_take_min_k is not None:
            self.gm_take_min_k = float(self.gm_take_min_k)
            if not 0 <= self.gm_take_min_k <= self.gm_take_k:
                raise ValueError(
                    f"gm_take_min_k {self.gm_take_min_k!r} is the take at the session's quietest "
                    "minute and must sit between 0 and the take at its most volatile, gm_take_k "
                    f"{self.gm_take_k!r}"
                )

    @property
    def strategy(self) -> str:
        """Which rule set this configuration runs."""
        return apple_models.strategy(self.model_key)

    @property
    def target_gain_k(self) -> float:
        """What a target exit is playing for, in level units a share.

        `buy_level` and `sell_level` are both `reference - k x unit` off the
        same reference, so the gap between them is this many units whatever the
        reference does -- a breach moving the forecast moves both levels
        together.

        In *units* it is therefore a property of the configuration. In dollars
        it is only fixed while the unit is: under `level_unit = "adr"` all
        session, under `"pred_range"` only until a breach widens the forecast.
        The stop still becomes a fixed price at the fill because `_buy` converts it
        to dollars once, there and then, and `_risk` reads that back rather
        than re-deriving it.
        """
        return float(self.buy_k) - float(self.sell_k)

    @property
    def unit_phrase(self) -> str:
        """What a log line calls one k -- "ADR" or "predicted range".

        Every line that prints a k reads this rather than writing "ADR", so a
        run under one unit cannot be read back in the other's terms.
        """
        return LEVEL_UNIT_PHRASES[self.level_unit]

    @property
    def has_stop(self) -> bool:
        """Whether this configuration stops out at all, in either unit.

        Separate from `stop_distance` because the run opens its log before
        there is a session, and so before there is an ADR to measure the stop
        against -- "is there a stop" is answerable then and "how far" is not.
        """
        return bool(self.stop_gain_fraction or self.stop_k)

    @property
    def can_scale_in(self) -> bool:
        """Whether a position can ever be added to under this configuration.

        `scale_in` asks for it; a position size under 100% is what leaves cash
        to do it with. At 100% the first buy spends the balance and the setting
        cannot change a trade, so the signature leaves it out there too.
        """
        return bool(self.scale_in) and self.position_pct < 100

    @property
    def ladder_has_room(self) -> bool:
        """Whether there is anywhere lower for an add to rest.

        Always under a fixed step. Under the half-way rule only while `buy_k`
        is above the bottom of the range, since every rung is half-way to it.
        """
        return bool(self.buy_step_k) or float(self.buy_k) < 1.0

    @property
    def fall_bars(self) -> int:
        """The look-back `max_fall_k` measures the fall over, in bars.

        The momentum take's, so "falling too fast to buy" and "the move has
        faded" read the same stretch of tape. With the take off there is no look-back
        of the run's own, so the default one.
        """
        return (
            int(self.negative_momentum_bars)
            or int(self.momentum_fade_bars)
            or APPLE_TRADER_NEGATIVE_MOMENTUM_BARS
        )

    @property
    def has_take(self) -> bool:
        """Whether the take is on, in any of its forms -- and with it the
        runner and the breakeven."""
        return bool(
            self.take_trigger == TAKE_ZONE
            or self.momentum_confirmation_bars
            or self.negative_momentum_bars
            or self.momentum_fade_bars
            or self.momentum_drop
        )


def as_config(value) -> AppleTraderConfig:
    """An `AppleTraderConfig` from one, from a record's dict, or from an
    instance of the class as it was before Streamlit reloaded this module.

    Saving any of the app's files makes Streamlit unload every one of its
    modules, so the next rerun builds a new `AppleTraderConfig` class while an
    agent started before the save keeps publishing configs of the old one into
    the state -- which `isinstance` refuses and `**` cannot unpack. Rebuilt by
    field name; a field the class no longer has is dropped and one it gained
    takes its default.
    """
    if isinstance(value, AppleTraderConfig):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        known = {f.name for f in fields(AppleTraderConfig)}
        value = {f.name: getattr(value, f.name) for f in fields(value) if f.name in known}
    return AppleTraderConfig(**value)


def level_unit(config: AppleTraderConfig, plan: "dict") -> float:
    """What one k is worth in dollars for this session, given the unit chosen.

    The single place `level_unit` is read. Everything counted in ks -- the two
    distances, the stop as a fraction of the predicted gain, the runner
    threshold, the circuit breaker -- goes through here, so there is no way for
    two of them to end up measuring against different yardsticks and comparing
    the results anyway.

    Under "adr" this is `adr14_abs`, fixed for the session. Under "pred_range"
    it is the current `pred_high - pred_low`, which is *not* fixed: with
    `keep_width` a breach keeps it until the session's own range is wider than
    the forecast, and without it a breach can widen it on every move
    (`_update_range`), so the unit -- and with it the dollar gap between the two
    levels -- widens as the day outgrows its forecast. Callers that need a number frozen at a moment (the stop, once
    there is a fill) must hold the dollars rather than re-reading the k.

    Falls back to the ADR when the forecast has no usable width, which keeps a
    half-populated forecast from collapsing both levels onto the reference --
    the failure mode is silent (two levels at the same price trade as a single
    one) where a fallback is merely wrong about the unit.
    """
    adr = float(plan.get("adr14_abs") or 0.0)
    if config.level_unit != UNIT_PRED_RANGE:
        return adr
    width = float(plan.get("pred_high") or 0.0) - float(plan.get("pred_low") or 0.0)
    return width if width > 0 else adr


def breach_width(config: AppleTraderConfig, plan: "dict") -> "float | None":
    """The width a breach keeps the forecast at, in dollars -- None for none.

    The level unit as it stood at 9:35: the ADR under "adr", the forecast's own
    `pred_high - pred_low` under "pred_range" -- read from `forecast_width`,
    which the plan keeps from before any breach, not from `level_unit`, which
    follows the range once containment has widened it. None when `keep_width`
    is off or the policy moves only one side (or nothing), which leaves
    `updated_range` doing what a stored record was run under.
    """
    if not config.keep_width or config.breach_update not in (BREACH_SHIFT, BREACH_BROWNIAN):
        return None
    adr = float(plan.get("adr14_abs") or 0.0)
    if config.level_unit != UNIT_PRED_RANGE:
        return adr or None
    width = float(plan.get("forecast_width") or 0.0)
    return width if width > 0 else (adr or None)


def stop_unit(config: AppleTraderConfig, plan: "dict") -> float:
    """The yardstick `stop_distance` should be handed for this config and session.

    The gain-fraction stop is a share of the predicted gain and follows the
    level unit; the legacy `stop_k` stop is in ADRs and stays there. Split out
    so that the four call sites cannot disagree about which.
    """
    if config.stop_gain_fraction:
        return level_unit(config, plan)
    return float(plan.get("adr14_abs") or 0.0)


def stop_distance(config: AppleTraderConfig, unit: float) -> float:
    """How far under the fill the stop sits, in dollars -- 0.0 when there is none.

    The one place the two parameterisations meet. A configuration made today
    carries `stop_gain_fraction`, a share of the predicted gain; a record
    written before that carries `stop_k`, a multiple of the ADR with no
    reference to what the trade was playing for. They are mutually exclusive
    on the config, so this is a choice between exactly one of them and nothing.

    `unit` is `level_unit` for the gain-fraction form, because a fraction of
    the predicted gain has to be counted in whatever the gain is counted in.
    For `stop_k` it is the ADR whatever the config's unit says: that form only
    ever appears on records written before the unit was a choice, and re-reading
    a stored "0.2 ADR under the fill" as 0.2 of something else would file a
    different stop beside the original as though it matched. `stop_unit` is
    what picks the right one.

    Module-level rather than a method on the config because the chart draws
    this line too and a second reading of the same settings is how a picture
    and a trade stop agreeing.
    """
    if config.stop_gain_fraction:
        return config.stop_gain_fraction * config.target_gain_k * float(unit)
    return config.stop_k * float(unit)


def stop_phrase(config: AppleTraderConfig) -> str:
    """How a log line names the stop's distance, in the units it was written in.

    A run reads back in the terms it was configured in: "0.5 x the predicted
    gain" for a configuration, "0.2 x ADR" for a record replayed from before
    the stop was written that way. Saying either in the other's units would
    make the log disagree with the form that produced it.
    """
    if config.stop_gain_fraction:
        return f"{config.stop_gain_fraction:g} × the predicted gain"
    return f"{config.stop_k:g} × ADR"


def fade_phrase(config: AppleTraderConfig) -> str:
    """The take's trigger in words, in whichever form it is written."""
    if config.take_trigger == TAKE_ZONE:
        return (
            f"a stall zone forming after the fill: {config.zone_min_bars} bars with every "
            f"close inside a band {config.zone_width:g} × the typical minute move wide"
        )
    if config.momentum_confirmation_bars:
        return (
            f"momentum over the last {config.momentum_confirmation_bars} bars negative "
            "and not recovering — its change neutral or negative"
        )
    if config.negative_momentum_bars:
        return (
            f"the {config.negative_momentum_bars}-bar momentum negative for "
            f"{config.negative_for_bars} bars in a row"
        )
    if config.momentum_fade_bars:
        return (
            f"the {config.momentum_fade_bars}-bar momentum turning from positive to "
            "balanced or negative"
        )
    return f"the momentum score {config.momentum_drop:g}σ off its peak"


def config_signature(config: "AppleTraderConfig | None" = None) -> str:
    """Compact identity of one rule set, standing in for a model name.

    SimLab groups and de-duplicates runs on this string exactly as it does on
    `provider/model` for the LLM agents, so two runs that differ in a level,
    the size, the model or the symbol are two configurations to test rather
    than a repeat. The symbol is in it because the same levels over GOOGL are a
    different experiment from the same levels over AAPL.

    The model key is written as stored rather than looked up through
    `apple_models.get`, which falls back to the default for a key it does not
    know -- a record naming a removed model should sign as that model, not as
    the one that is left.

    The managed exit is written only while switched on -- the stop when it has
    a distance, the take and its runner threshold when the take is on -- so a config with both off signs exactly as a run recorded before the
    exit existed, and `take_fraction` never splits two runs that cannot differ.

    The stop is written in the units it was configured in: `E-0.5G` is half the
    predicted **G**ain under the entry, `E-0.2A` the legacy 0.2 **A**DR. Not
    converted to one or the other, because a stored record's signature is its
    identity everywhere downstream -- rewriting an old run's would move it to a
    different row in Results and hide it from the already-tested check.
    The intraday update and the session circuit breaker follow the same rule for
    the same reason: each appears only when it is switched on, so every record
    written before it existed (which replays as off) keeps the signature it was
    filed under.
    """
    c = config or AppleTraderConfig()
    if c.strategy == apple_models.STRATEGY_MOMENTUM:
        return _momentum_signature(c)
    if c.strategy == apple_models.STRATEGY_PROBABILITY:
        return _probability_signature(c)
    # "A" for the ADR, "R" for the predicted range. The token rides on every k
    # in the signature rather than appearing once at the end, so that a run is
    # never read as the same strategy at a different size -- the same 0.4 is a
    # different distance under each unit. Records written before the unit was a
    # choice replay as "adr" and keep the "A" they were filed under.
    unit = LEVEL_UNIT_TOKENS[c.level_unit]
    exits = ""
    if c.stop_gain_fraction:
        exits += f",stop=E-{c.stop_gain_fraction:g}G"
    elif c.stop_k:
        exits += f",stop=E-{c.stop_k:g}A"
    # The take is signed in the form it was configured in, like the stop:
    # `@conf` is the momentum confirmation's table (its look-back is signed with
    # the entry gate, `confirm=5b`, since one number drives both);
    # `@neg15b/5b` is the 15-bar momentum negative for 5 bars in a row;
    # `@fade15b` the legacy positive-to-balanced turn over 15 bars and `@mom-1`
    # the older 1σ fall from the peak, which stored records keep. `@zone20b/3u`
    # is the forming-zone take: 20 bars inside a band 3 typical minute moves
    # wide, whatever the confirmation's look-back.
    if c.take_trigger == TAKE_ZONE:
        take = f"zone{c.zone_min_bars}b/{c.zone_width:g}u"
    elif c.momentum_confirmation_bars:
        take = "conf"
    elif c.negative_momentum_bars:
        take = f"neg{c.negative_momentum_bars}b/{c.negative_for_bars}b"
    elif c.momentum_fade_bars:
        take = f"fade{c.momentum_fade_bars}b"
    elif c.momentum_drop:
        take = f"mom-{c.momentum_drop:g}"
    else:
        take = ""
    if take:
        exits += f",take={c.take_fraction * 100:g}%@{take}"
        # The two gates, each only while set, so every record written before
        # it existed (which replays at 0) keeps its filed signature: the
        # legacy realised-gain gate, then the minutes since the fill.
        if c.take_min_gain_fraction:
            exits += f">={c.take_min_gain_fraction:g}G"
        if c.take_after_minutes:
            exits += f">={c.take_after_minutes}m"
        # Only while on, so every record written before it existed (which
        # replays gains-only) keeps its filed signature.
        if c.take_in_loss:
            exits += "+loss"
        exits += f",runner>={c.hold_min_gain_k:g}{unit}"
    if c.min_win_k:
        exits += f",min_win={c.min_win_k:g}{unit}"
    # Only where it can change a trade (`can_scale_in`), so a run at 100% signs
    # the same with the setting on or off -- and every record written before it
    # existed, which replays with it off, keeps the signature it was filed under.
    # The step is written as the step; the half-way rule it replaced keeps the
    # token every record made before it was filed under.
    if not c.can_scale_in:
        adds = ""
    elif c.buy_step_k:
        adds = f",adds=+{c.buy_step_k:g}{unit}"
    else:
        adds = ",adds=half"
    # The stop under the fill rather than under the next rung. Only where the
    # two can differ -- a ladder and a stop -- and only in today's form, so a
    # record from before it keeps the `adds=half` it was filed under.
    if c.can_scale_in and c.has_stop and not c.stop_under_next_buy:
        adds += ",stop@fill"
    if c.can_scale_in and c.add_under_fill:
        adds += ",add<fill"
    # Only while on, so every record written before it existed (which replays
    # with market buys) keeps the signature it was filed under.
    if c.limit_entry:
        adds += ",limit"
    # Only while on, like every rule added since the notebook's. The look-back
    # rides along because it is not otherwise in the signature when the take
    # is off, and the same limit over 5 bars and over 30 is not one rule.
    if c.max_fall_k:
        adds += f",nofall={c.max_fall_k:g}{unit}/{c.fall_bars}b"
    # The confirmation's look-back, written once for both sides. Only while on,
    # so every record written before it existed keeps its filed signature.
    if c.momentum_confirmation_bars:
        adds += f",confirm={c.momentum_confirmation_bars}b"
    breach = "" if c.breach_update == BREACH_OFF else f",breach={c.breach_update}"
    # Both appear only while switched on, like every rule added since the
    # notebook's, so a record written before either existed keeps the signature
    # it was filed under. They are separate tokens because they are separate
    # rules: one is about what the forecast is allowed to say, the other about
    # what closes a position.
    # Only where it can change a trade (`breach_width`), so a run under "off"
    # signs the same either way -- and every record written before it existed,
    # which replays with it off, keeps the signature it was filed under.
    if c.keep_width and c.breach_update in (BREACH_SHIFT, BREACH_BROWNIAN):
        breach += ",keep_width"
    if c.contain_range:
        breach += ",contain"
    if c.breach_exit:
        breach += ",breach_exit"
    # Only while some day is sat out, so every record written before the rule
    # existed (which replays with none) keeps the signature it was filed under.
    if c.skip_events:
        breach += ",skip=" + "+".join(event_days.TOKENS[e] for e in c.skip_events)
    # Written only when the reference is not the predicted high: a record from
    # when it could be IntradayVolatility's curve keeps the identity it was
    # filed under (and is refused, `level_source_error`).
    levels = "" if c.level_source == LEVELS_DAYRANGE else f",levels={c.level_source}"
    # "H" in the two distances is whatever `level_source` says it is, which is
    # why that token is next to them rather than at the end.
    return (
        f"{c.model_key}_{c.ticker}(buy=H-{c.buy_k:g}{unit},sell=H-{c.sell_k:g}{unit}{levels},"
        f"size={c.position_pct:g}%{adds}{exits}{breach})"
    )


def _momentum_signature(c: AppleTraderConfig) -> str:
    """`config_signature` for Fast&Furious's rules: only the fields they read.

    `q` is the entry quantile, `stop=1.5dn` the stop in forecast 2-hour drops
    under the fill (`stop=off` without one), `hold` the clock a signal starts,
    `sellexit` the close on a sell signal. The day-range fields are not in it:
    nothing under this model reads them, so two runs differing only there are
    one configuration.
    """
    skip = ",skip=" + "+".join(event_days.TOKENS[e] for e in c.skip_events) if c.skip_events else ""
    return f"{c.model_key}_{c.ticker}({_ff_rules(c)},size={c.position_pct:g}%{skip})"


def _probability_signature(c: AppleTraderConfig) -> str:
    """`config_signature` for GoodMoments' rules: only the fields they read.

    `enter=+0.02` is the edge a buy needs, `exit` the edge under which a long
    is sold (`off` without one), `take` and `stop` the two distances from the
    fill in **A**DRs (`off` at 0). A take that follows the intraday volatility
    curve signs as `take=0.4..0.1A@vol` -- at the most volatile minute, then at
    the quietest; one that does not keeps the form every record before it was
    filed under. The momentum confirmation signs as `confirm=3b`, only while
    it is on. A hybrid adds the buy level its entries also need after the
    opening window, in the day-range signature's tokens: `buy<=H-0.6R` and how
    a breach moves it (`_hybrid_levels_signature`)."""
    exit_ = f"{c.gm_exit:+g}" if c.gm_exit is not None else "off"
    take = f"{c.gm_take_k:g}A" if c.gm_take_k else "off"
    if c.gm_take_k and c.gm_take_min_k is not None:
        take = f"{c.gm_take_k:g}..{c.gm_take_min_k:g}A@vol"
    stop = f"{c.gm_stop_k:g}A" if c.gm_stop_k else "off"
    confirm = f",confirm={c.gm_confirm_bars}b" if c.gm_confirm_bars else ""
    skip = ",skip=" + "+".join(event_days.TOKENS[e] for e in c.skip_events) if c.skip_events else ""
    levels = _hybrid_levels_signature(c) if apple_models.levels_of(c.model_key) else ""
    return (
        f"{c.model_key}_{c.ticker}(enter={c.gm_enter:+g},exit={exit_},take={take},stop={stop}"
        f"{confirm}{levels},size={c.position_pct:g}%{skip})"
    )


def _hybrid_levels_signature(c: AppleTraderConfig) -> str:
    """A hybrid's buy level in signature form: its distance under the
    predicted high and, each only while it can move the level, the breach
    policy, the kept width and the containment -- the tokens the day-range
    signature writes for them."""
    unit = LEVEL_UNIT_TOKENS[c.level_unit]
    out = f",buy<=H-{c.buy_k:g}{unit}"
    if c.breach_update != BREACH_OFF:
        out += f",breach={c.breach_update}"
    if c.keep_width and c.breach_update in (BREACH_SHIFT, BREACH_BROWNIAN):
        out += ",keep_width"
    if c.contain_range:
        out += ",contain"
    return out


def _ff_rules(c: AppleTraderConfig) -> str:
    """Fast&Furious's four rules in signature form, wherever they are traded."""
    stop = f"{c.ff_stop_k:g}dn" if c.ff_stop_k else "off"
    rules = f"q={c.ff_entry_quantile:g},stop={stop},hold={c.ff_hold_minutes}m"
    if c.ff_exit_on_sell_signal:
        rules += ",sellexit"
    return rules


def model_ticker_error(config: AppleTraderConfig) -> "str | None":
    """Why this model cannot trade this symbol, or None if it can.

    The one check here that needs no bundle, because it is about a model that
    was never fitted -- or no longer exists -- rather than one that failed to
    load, and those are different problems with different fixes. Left to the
    loader it would surface as "no model at <path>", sending the reader to look
    for a file that was never meant to exist.
    """
    if config.model_key not in apple_models.MODELS:
        available = ", ".join(m.label for m in apple_models.MODELS.values())
        return (
            f"'{config.model_key}' is not a model this app can run: it has been "
            "removed, and a run configured on it is not replayed on a different "
            f"model. Apple Trader runs on {available}."
        )
    if apple_models.covers(config.model_key, config.ticker):
        return None
    model = apple_models.get(config.model_key)
    alternatives = ", ".join(
        apple_models.get(key).label for key in apple_models.keys_for(config.ticker)
    )
    remedy = (
        f"Pick one of the models {config.ticker} has ({alternatives})"
        if alternatives
        else f"Nothing here was fitted on {config.ticker}, so pick another instrument"
    )
    return (
        f"{model.label} was fitted on {', '.join(model.tickers)} only and nothing "
        f"claims it transfers, so it cannot trade {config.ticker}. {remedy}."
    )


def config_error(config: AppleTraderConfig, bundle: "dict | None" = None) -> "str | None":
    """The first reason this rule set cannot run on this bundle, or None.

    One call for every caller that is about to start a run, so a rule added
    later is checked everywhere it needs to be rather than in whichever launch
    path was remembered.
    """
    return (
        model_ticker_error(config) or level_source_error(config)
    )


def level_source_error(config: AppleTraderConfig) -> "str | None":
    """Why a record measuring its levels below something other than the
    predicted high cannot run, or None.

    The only other reference there was is IntradayVolatility's time-of-day
    curve stretched to the day-range forecast ("Day Range × Intraday
    Volatility", removed 2026-10-20). A stored SimLab record made with it is
    refused rather than replayed on the flat high, which would be a different
    strategy filed beside the original as though it matched.
    """
    if config.level_source == LEVELS_DAYRANGE:
        return None
    return (
        f"levels measured below '{config.level_source}' are not something this app can "
        "run: levels under IntradayVolatility's time-of-day curve were removed with the "
        "Day Range × Intraday Volatility model (2026-10-20), and a run configured on them "
        "is not replayed on the flat predicted high."
    )


def _dayrange():
    """`agent_stonks.dayrange_model`, imported on first use.

    Kept out of this module's imports because it pulls PyTorch and LightGBM in
    (in that order, deliberately -- see its docstring), and a process that only
    lists the agents or renders the form must not pay for a 200 MB dependency
    it does not use. `sys.modules` makes the repeat calls free.
    """
    from . import dayrange_model

    return dayrange_model


def _highlow():
    """`agent_stonks.highlow_model`, imported on first use (same reason as
    `_dayrange`)."""
    from . import highlow_model

    return highlow_model


def _highlow2():
    """`agent_stonks.highlow2_model`, imported on first use (LightGBM)."""
    from . import highlow2_model

    return highlow2_model


def _fastfurious():
    """`agent_stonks.fastfurious_model`, imported on first use (LightGBM)."""
    from . import fastfurious_model

    return fastfurious_model


def session_forecast(
    bundle: dict,
    ticker: str,
    opening,
    today,
    key: "str | None" = None,
    secret: "str | None" = None,
) -> "tuple[dict, float | None]":
    """Today's `(forecast, open_price)` from whichever model `bundle` is.

    The one place the strategy learns which model it is running on, and the
    only thing that differs between them: both return the same forecast keys
    (`pred_high`, `pred_low`, `adr14_abs`, ...), and every level, exit and
    breach rule downstream reads those and nothing else.

    `open_price` is the official opening print the day-range model is fed. The
    HighLow models roll their daily
    bars up from minute bars, so they read the first bar's open instead (IEX's,
    for HighLow2) and need no print; None then makes the plan
    fall back to the caller's first bar.
    """
    if bundle.get("kind") == "highlow":
        return (
            _highlow().forecast_session(bundle, ticker, opening, today, key, secret),
            None,
        )
    if bundle.get("kind") == "highlow2":
        return (
            _highlow2().forecast_session(bundle, ticker, opening, today, key, secret),
            None,
        )
    dayrange = _dayrange()
    history = dayrange.daily_frame_from_bars(
        historical.fetch_daily_ohlc_bars(ticker, days=dayrange.DAILY_HISTORY_DAYS)
    )
    # Returned too, so the plan (and the record's seed) carries the print the
    # forecast was built from.
    open_price = historical.fetch_session_open(ticker)
    forecast = dayrange.forecast_session(
        bundle, history, opening, today, open_price=open_price,
    )
    return forecast, open_price


def _window_bars(
    state: AppState, ticker: str, tape: str, start, end
) -> "list[dict]":
    """One tape's bars for [start, end), or an exception saying why not.

    The Alpaca tapes go through `agent_mod.fetch_bars_window` -- the same
    (simulation-patched) call `agent._opening_range_for` recovers an opening
    range with, so a replay reads its dataset here rather than the network.
    yfinance serves today's session only, which is all this is asked for, and
    needs no credentials, so it is the consolidated source a free Alpaca key
    still has.

    `sip_delayed` is SIP asked for a window a free/basic plan will actually
    answer: those plans refuse anything reaching into the trailing
    SIP_DELAY_MIN minutes, so a window that recent is declined here instead of
    spending a request on a certain 403.
    """
    if tape == "yfinance":
        bars = historical.fetch_intraday_bars(ticker, interval="1m")
        return [b for b in bars if start <= pd.to_datetime(b["t"], utc=True) < end]
    if tape == "sip_delayed":
        if end > clock.now().astimezone(timezone.utc) - timedelta(minutes=SIP_DELAY_MIN):
            raise ValueError(
                f"delayed SIP does not serve the trailing {SIP_DELAY_MIN} minutes"
            )
        tape = "sip"
    if not (state.api_key and state.api_secret):
        raise ValueError("no Alpaca credentials")
    return agent_mod.fetch_bars_window(
        ticker, "1Min", start, end, state.api_key, state.api_secret, tape,
    )


def _refetch_opening_window(
    state: AppState, ticker: str, want: int, consolidated_only: bool = False
):
    """The session's first `want` minutes from the best tape that will serve
    them: `(frame, tape)`, or `(None, "")` when none of them can.

    Ordered by `bar_history.feed_order` from the session's resolved history
    feed, so this asks the consolidated tape first and reaches IEX only when
    SIP, delayed SIP and yfinance have all declined -- the same ranking the bar
    buffer itself is filled by, for the same reason. `consolidated_only` drops
    IEX from that list, for the caller who already holds the right five minutes
    on the IEX scale and is here to improve on them: a REST IEX window would
    return the same numbers for the price of a round trip.

    A tape that answers with a short window, or one that does not start at
    09:30, has not answered: the forecast is a claim about the opening five
    minutes and half of them is not a cheaper version of it.
    """
    open_utc = market_hours.session_open()
    if open_utc is None:
        return None, ""
    end = open_utc + timedelta(minutes=want)
    resolved = state.history_feed_resolved or state.history_feed
    for tape in bar_history.feed_order(resolved):
        if consolidated_only and not bar_history.is_consolidated(tape):
            continue
        try:
            bars = _window_bars(state, ticker, tape, open_utc, end)
        except Exception:
            continue
        recovered = momentum_regime.frame_from_bars(bars)
        if len(recovered) >= want and float(recovered["minutes_from_open"].iloc[0]) < 1.0:
            return recovered.iloc[:want], tape
    return None, ""


def fetch_opening_window(state: AppState, frame, want: int, ticker: str = DEFAULT_TICKER):
    """The first `want` regular-session bars of today and the tape they are on.

    Returns `(frame, tape)`. The tape is part of the answer because the forecast
    reads minute *volume* out of these bars (`or_volume_share`), and the ridge
    that reads it was fitted on consolidated volume -- IEX carries under 4% of
    it, which is a bias rather than a rounding error. The caller quotes the tape
    in the caveat it logs, so the caveat describes the bars the forecast was
    actually built on rather than whichever Alpaca feed the sidebar is holding.

    Two ways the window is wrong if taken straight off the buffer, and both are
    repaired the same way -- by re-fetching the 09:30 window from the best tape
    that will serve it (`_refetch_opening_window`):

    * an agent started after 9:35 has a buffer that begins wherever the stream
      did, and `frame.iloc[:want]` would hand the model five bars from the middle
      of the day as though they were the open;
    * a buffer filled from IEX has the right five minutes on the wrong volume
      scale. A consolidated tape is preferred over it even when the bars are
      right there, and the IEX bars are kept (with the caveat) only when nothing
      consolidated will answer -- which is the case before 9:50 on a free key,
      where every consolidated source is 15 minutes behind.

    A replay has none of that choice: a simulation has exactly one tape, the one
    its dataset was downloaded on (`AppState.bar_tape_override`), and every
    source below is patched back to those same stored bars. So there is nothing
    to shop for, and shopping would be worse than useless -- it would hand back
    IEX bars under the name of whichever consolidated feed was asked first, and
    the caveat that should have been logged would not be.

    `ticker` is the symbol the caller trades, and only matters on the re-fetch
    path, since `frame` is already the right symbol's bars.
    """
    buffered = frame.iloc[:want]
    covers_open = float(buffered["minutes_from_open"].iloc[0]) < 1.0
    tape = state_mod.bar_tape(state)
    replayed = str(getattr(state, "bar_tape_override", "") or "")
    if covers_open and (replayed or bar_history.is_consolidated(tape)):
        return buffered, tape

    recovered, source = _refetch_opening_window(
        state, ticker, want, consolidated_only=covers_open
    )
    if recovered is not None:
        return recovered, replayed or source
    if covers_open:
        # IEX bars of the right five minutes. Worse than a consolidated tape,
        # better than no forecast, and the caveat says which one this was.
        return buffered, tape
    raise ValueError(
        f"the first {want} minutes of the session are not in the bar buffer "
        f"(it starts at {frame.index[0]:%H:%M}) and could not be re-fetched; "
        "the forecast is built on the 09:30 window and cannot be made without it."
    )


# --- the momentum confirmation's behaviour table ---------------------------
#
# The user's table (2026-09-24), keyed by (momentum, momentum change), each
# "positive", "neutral" or "negative". What it reads the tape as, and what each
# side may do on that bar:
#
#   buy          -- a buy at the buy level (first entry or an add) is allowed;
#   take         -- below the sell level (in profit, or at a loss with
#                   `take_in_loss`): sell `take_fraction`;
#   sell_target  -- at or above the sell level: the target (and the breach
#                   exit) may sell.
#
# The price column of the table (down / flat / up) is the momentum's own sign,
# so it is not a key of its own. The stop, the breakeven and the flatten never
# read this.
POSITIVE, NEUTRAL, NEGATIVE = "positive", "neutral", "negative"


@dataclass(frozen=True)
class MomentumRow:
    prediction: str
    buy: bool
    take: bool
    sell_target: bool


_GOING_UP = MomentumRow("the price is going up", buy=True, take=False, sell_target=False)
MOMENTUM_TABLE: "dict[tuple[str, str], MomentumRow]" = {
    (NEGATIVE, POSITIVE): MomentumRow("the drop is slowing down", False, False, True),
    (NEGATIVE, NEUTRAL): MomentumRow("the price keeps dropping", False, True, True),
    (NEGATIVE, NEGATIVE): MomentumRow("the drop is accelerating", False, True, True),
    (NEUTRAL, POSITIVE): MomentumRow("the price is about to start rising", True, False, True),
    # Allowed since 2026-09-24 (the user's change): a flat tape at the buy
    # level is bought, only one about to start dropping is not.
    (NEUTRAL, NEUTRAL): MomentumRow("the price is still flat", True, False, True),
    (NEUTRAL, NEGATIVE): MomentumRow("the price is about to start dropping", False, False, True),
    (POSITIVE, POSITIVE): _GOING_UP,
    (POSITIVE, NEUTRAL): _GOING_UP,
    (POSITIVE, NEGATIVE): _GOING_UP,
}


def momentum_class(value: float, band: float) -> str:
    """Positive, neutral or negative: neutral while `|value| < band`."""
    if abs(value) < band:
        return NEUTRAL
    return POSITIVE if value > 0 else NEGATIVE


def momentum_read(closes, n: int, minute_move: "float | None") -> "dict | None":
    """The confirmation's read of a session's closes, or None while it cannot be made.

    `closes` are this session's closed bars, oldest first. Over the last `n`
    bars:

    * momentum -- the average per-bar move, `(close - close[n bars ago]) / n`;
    * change -- the average bar-to-bar change of the 1-bar momentum,
      `(m1 - m1[n bars ago]) / n`, where `m1 = close - previous close`.

    Both are per bar, the scale of `minute_move` (the ticker's
    `abs_mean_minute_momentum`), and neutral while their size is under
    `MOMENTUM_NEUTRAL_FRACTION` of it. None without a look-back, without that
    measure, or before the session has the `n + 2` bars the change needs.
    """
    n = int(n or 0)
    if n < 1 or minute_move is None or not float(minute_move) > 0:
        return None
    c = [float(x) for x in closes]
    if len(c) < n + 2:
        return None
    mom = (c[-1] - c[-1 - n]) / n
    change = ((c[-1] - c[-2]) - (c[-1 - n] - c[-2 - n])) / n
    band = MOMENTUM_NEUTRAL_FRACTION * float(minute_move)
    key = (momentum_class(mom, band), momentum_class(change, band))
    return {
        "n": n, "mom": mom, "change": change, "band": band,
        "mom_class": key[0], "change_class": key[1], "row": MOMENTUM_TABLE[key],
    }


def momentum_unknown(ticker: str, n: int, minute_move: "float | None", bars: int) -> str:
    """Why `momentum_read` cannot read this bar -- for the log line that says
    a rule wanted to act and the confirmation held it."""
    if minute_move is None or not float(minute_move) > 0:
        return (
            f"{ticker}'s mean one-minute move over last week is not known yet, "
            "so momentum has no neutral band to be read against"
        )
    return f"momentum over {n} bars needs {n + 2} bars of the session and there are {bars}"


def momentum_words(read: dict) -> str:
    """One read in a log line's words."""
    return (
        f"momentum over the last {read['n']} bars is {read['mom']:+.3f} $/bar "
        f"({read['mom_class']}) and its change {read['change']:+.3f} $/bar "
        f"({read['change_class']}; neutral inside ±{read['band']:.3f}) — "
        f"{read['row'].prediction}"
    )


class _SitsOut:
    """The days-off check (`skip_events`), the same for every rule set: it is
    a statement about the day, not about how it is traded. Expects `config`,
    `ticker`, `blocked` and a `_sit_out_noted` dict on the trader."""

    def _sit_out(self, state: AppState, frame, today, want: int) -> bool:
        """Whether this session is one the configuration sits out
        (`skip_events`, see `event_days`) -- or, while the morning's briefing is
        still being written, cannot say yet. True means no forecast this cycle;
        a session sat out for good is recorded in `self.blocked`, like a
        forecast that failed, and lasts until the next session.

        Asked when the opening window has closed, the moment the forecast would
        be made: by then the 09:25 briefing has normally landed. A briefing
        still being written is waited for up to BRIEFING_WAIT_BARS bars, after
        which the session trades on the calendar's word alone.
        """
        if not self.config.skip_events:
            return False
        noted = self._sit_out_noted if self._sit_out_noted.get("date") == today else {}
        self._sit_out_noted = noted
        noted["date"] = today
        pending = self.ticker in (getattr(state, "premarket_pending", None) or [])
        try:
            found = event_days.check(self.ticker, today, self.config.skip_events, pending)
        except Exception as exc:  # noqa: BLE001 -- a calendar read must not stop the day
            if not noted.get("failed"):
                noted["failed"] = True
                _log(state, {"type": "error", "text": (
                    f"Could not check whether today is a day to sit out ({exc}); trading it."
                )})
            return False
        if found.waiting and len(frame) - want < BRIEFING_WAIT_BARS:
            if not noted.get("waiting"):
                noted["waiting"] = True
                _log(state, {"type": "status", "text": (
                    f"Waiting for {self.ticker}'s pre-market briefing to say whether today is a "
                    "market or geopolitical shock before forecasting."
                )})
            return True
        if found.events:
            self.blocked = {
                "date": today, "sit_out": found.categories,
                "reason": f"sits the session out: {event_days.sit_out_phrase(found.events)}",
            }
            _log(state, {"type": "analysis", "text": (
                f"Apple Trader sits out {self.ticker}'s session today — "
                f"{event_days.sit_out_phrase(found.events)}. No forecast, no orders; the "
                "forecast is not built for a day set by that news."
            )})
            return True
        if found.notes and not noted.get("notes"):
            noted["notes"] = True
            _log(state, {"type": "status", "text": (
                "Days-off check: " + "; ".join(found.notes) + "."
            )})
        return False


class DayRangeTrader(_SitsOut, BaseTrader):
    """The day-range rules: one forecast at 9:35, then two resting levels.

    TimeToChange3's model says where the session's high and low will land, and
    it says it exactly once -- from the daily history up to yesterday plus the
    first five minutes of this morning. The model is never re-run, so the rule
    built on it cannot be a per-bar signal. It is two price levels, set at 9:35,
    from notebook 05:

        buy_level  = H - buy_k  * A
        sell_level = H - sell_k * A

    with `H` the predicted high and `A` the 14-day average daily range in
    dollars. Buy when the price comes down to the buy level, sell when it comes
    back up to the sell level, repeat as often as the day allows, and flatten
    what is still open before the close.

    One setting makes `H` less of a constant than the notebook's:
    `breach_update` -- a session that trades *through* the predicted high has
    falsified it, and the levels hanging off it with it, so the breached side
    moves. One-way: the forecast only ever widens (`_update_range`). With it
    `off` this is the notebook's rule exactly.

    It is a mean-reversion bet, and the reason it is shaped that way is in the
    model's own results: what TimeToChange3 forecasts well is the *width* of
    the session, not its direction. So the rule never takes a view on where the
    day is going -- it buys well below where the day is expected to top out and
    sells just under it, and on a day that never dips that far it simply does
    nothing.

    The entry is those two levels and nothing else -- no regime, no
    probability. The exit has more to it (`_exit`): on top of the sell level
    and the flatten there is a stop under the fill, a momentum take that banks
    most of a fading gain short of the target, and a breakeven on the runner
    that take leaves. That half is not the notebook's and has not been measured
    against it; a stop and a momentum take of 0 switch it off and give notebook
    05's rule back.

    Nor is the ladder (`scale_in`): under a position size below 100% the cash
    the first buy leaves can buy more on the way down, each add resting
    `buy_step_k` units under the last buy's level and the whole ladder back at
    `buy_k` once the position is sold out (`_add`, `_after_fill`, `_rung_k`).
    The stop stays under the last actual fill, so a rung is only bought while
    it sits above that stop. Switched off it is the
    notebook's one buy per position.

    Against the notebook
    --------------------
    The trigger is the notebook's, bar for bar: a buy fires on a bar whose
    **low** touched the buy level and a sell on a bar whose **high** touched
    the sell level, checked in that order and never both on one bar.
    `tests/test_dayrange_model.py` pins the forecast itself to notebook 05's
    recorded number for 2026-08-07.

    Three differences remain, and only the first is small:

    * **the fill**. The notebook rests limit orders and fills a buy at
      `min(bar_open, buy_level)` -- a touch fills *at* the level. Live there is
      no resting order in this ledger: the loop sees the bar after it closed
      and only then sends the order. It is a limit at the buy level
      (`limit_entry`), so it never pays more than the level -- but on a bar
      that dipped to the level and recovered it buys nothing, where the
      notebook's resting order would have filled at the level; the level stays
      armed for the next touch. A paper or SimLab buy fills at the quote if
      that is at or under the limit, a real one is a day limit order cancelled
      if unfilled after `ORDER_FILL_TIMEOUT_SEC`. Records made before
      2026-09-30 replay the old market order, which filled near the close,
      above the level on such a bar. Read a SimLab result against the
      notebook's with either in mind.
    * **the flatten**. The notebook closes at the 15:59 bar, so the whole
      session is available. Here `flatten_before_close_min` applies as it does
      to every other agent -- at the default of 5 the position is closed around
      15:55, giving up the last few minutes. Set it to 1 for the notebook's
      behaviour.
    * **no entry inside the flatten window**. The notebook has no such rule
      because it never needs one; here a long the next rule is about to shut is
      two commissions.

    And what the rule is worth, honestly: notebook 05 ran it across all 21
    sessions that had minute data, 15 of which traded, 10 profitable, $949
    total on $10,000 a day. Twenty-one sessions is a sanity check, not an edge,
    and there are no commissions or slippage in that number.
    """

    # Its entry fires on a resting level rather than on a signal, and the log
    # line says so.
    ENTRY_TRIGGER_TEXT = "Buy level touched"

    #: The stand-down a stop leaves. Unlike the breaker's, it ends the live run
    #: (`halt`) and is not carried into the next ▶ Start (`_resume`).
    STOPPED_OUT = "stopped out"

    def __init__(self, config: "AppleTraderConfig | None" = None) -> None:
        super().__init__(config or AppleTraderConfig())
        # The session's forecast and the two levels derived from it, or None
        # before 9:35. Keyed by date so a multi-day run re-forecasts each
        # morning rather than trading Tuesday off Monday's levels.
        self.plan: "dict | None" = None
        # The last sidebar edit refused for its unit, so it is reported once
        # rather than every minute (`_adopt_form_levels`).
        self._refused_form = None
        # The bundle of a model switched to from the sidebar mid-run
        # (`_adopt_form_model`), used in place of the one the loop was started
        # with. None while the run is still on the model it started on.
        self._bundle: "dict | None" = None
        # The last model switch refused, so it is reported once.
        self._refused_model = None
        # The ticker's `abs_mean_minute_momentum`, read off its state each
        # cycle: what the momentum confirmation's neutral band is a share of.
        self.minute_move: "float | None" = None
        # Why this bar did not sell at the sell level, for `run_cycle` to log
        # (`_exit` only returns what to sell).
        self._held_note: "str | None" = None
        # The session `_sit_out` last wrote about, so its notes and its wait for
        # the briefing are each logged once a day rather than every minute.
        self._sit_out_noted: "dict" = {}
        # Set by a stop-out: the live loop stops the agent on it, and only a
        # fresh ▶ Start trades again. SimLab never reads it -- a replay has no
        # ▶ Start, so there the stand-down simply lasts the session.
        self.halt: "str | None" = None
        # Whether a run started again today re-arms a session that was stopped
        # out (`_resume`). A single run stops on a stop-out, so being started
        # again is the say-so; a split Orchestra (`orchestra`) runs on past one
        # pair's stop-out, so for its pairs a restart is not.
        self.rearm_after_stop = True
        # Where the published record and the sidebar form live (`TraderSlot`).
        self.slot: TraderSlot = SINGLE_SLOT
        # Whether the per-bar read line goes to the log. A race silences the
        # racers that hold nothing and writes one line for all of them instead.
        self.quiet_read = False
        # The close and the time of the last bar this trader read, for whoever
        # watches several traders at once (`orchestra`'s board).
        self.last_close: "float | None" = None

    def activity(self, outcome: str, tracker: DecisionTracker) -> "tuple[str, str]":
        """The base's phrases, plus the three this strategy's day has: before the
        forecast, a session sat out, and after a stand-down."""
        if outcome != rule_agent.CLOSED and (self.blocked or {}).get("sit_out"):
            labels = ", ".join(event_days.LABELS[c] for c in self.blocked["sit_out"])
            return rule_agent.WAITING, f"Agent sits today out ({labels})"
        if outcome == rule_agent.WARMING_UP and self.plan is None and self.blocked is None:
            return rule_agent.WAITING, "Agent waiting for the opening forecast"
        if (
            outcome != rule_agent.CLOSED
            and self.plan is not None
            and self.plan.get("stand_down")
            and tracker.position_for(self.ticker) <= 0
        ):
            return rule_agent.WAITING, "Agent stood down for the day"
        return super().activity(outcome, tracker)

    # --- one cycle --------------------------------------------------------

    def run_cycle(self, bundle: dict, state: AppState, tracker: DecisionTracker) -> str:
        """Read the newest closed bar and act on it. Returns a short outcome
        tag ("bought", "sold", "hold", "warming_up", "closed", "no_data")."""
        sym_state, refused = self.preflight(state)
        if refused is not None:
            return refused

        today = _dayrange().market_date()
        self._roll_session(today)

        frame = momentum_regime.minute_frame(sym_state)
        if not len(frame):
            _log(
                state,
                {"type": "status", "text": f"No {self.ticker} bars yet today."},
            )
            return "no_data"
        self.minute_move = getattr(sym_state, "abs_mean_minute_momentum", None)

        # Before anything reads the levels, so an edit made in the sidebar
        # during the last minute is what this bar is judged against. The model
        # first: a switch brings its own pair of distances with it.
        replanned = self._adopt_form_model(state, frame, today)
        self._adopt_form_levels(state, frame.index[-1])
        bundle = self._bundle or bundle

        want = _dayrange().opening_minutes(bundle)
        if self.plan is None:
            if self.blocked is not None:
                return "no_data"
            if len(frame) >= want and self._sit_out(state, frame, today, want):
                return "no_data" if self.blocked is not None else "warming_up"
            if len(frame) < want:
                _log(
                    state,
                    {
                        "type": "status",
                        "text": (
                            f"{len(frame)} of the first {want} {self.ticker} minutes are in; "
                            "the "
                            "day's high cannot be forecast until the opening window closes."
                        ),
                    },
                )
                return "warming_up"
            if not self._plan_session(bundle, state, frame, today, want):
                return "no_data"

        last = frame.iloc[-1]
        ts = frame.index[-1]
        self.last_close = float(last["close"])
        fresh_bar = ts != self.last_bar_ts
        if fresh_bar:
            self.last_bar_ts = ts

        position = tracker.position_for(self.ticker)
        if position > 0 and self.entry is None:
            # A position without a remembered entry (agent restarted onto an
            # existing ledger): adopt it at this bar. The real fill is not
            # known, so the stop, the breakeven and the momentum peak are all
            # measured from here -- an approximation, but a stop measured from
            # nothing would be worse.
            self.entry = {
                "price": float(last["close"]),
                "bars": 0,
                "ts": ts,
                "risk": stop_distance(self.config, stop_unit(self.config, self.plan))
                if self.plan
                else None,
            }
        if position <= 0:
            self.entry = None
        if fresh_bar and self.entry is not None:
            self.entry["bars"] += 1

        # Before the read, so the line below quotes the levels this bar is
        # actually about to be measured against rather than last bar's. The
        # forecast moves first and the levels are then rebuilt from it at this
        # minute, which is also what re-reads a reference that follows the clock.
        # A forecast just re-made for a switched model is the 9:35 one and has
        # not yet seen this bar, so it is moved even when the bar is not new.
        if (fresh_bar or replanned) and self.plan is not None:
            # The high this bar is judged a breach of, recorded before the bar
            # is allowed to move it. Testing against the updated number would
            # be testing the bar against a level it had just pushed out of its
            # own way, which under "brownian" is exactly what used to keep a
            # resolved position open. See `_exit`.
            self.plan["high_at_bar"] = float(self.plan["pred_high"])
            self._update_range(state, frame, ts)
            self._set_levels(ts)

        # Recorded from the same plan the line below prints, in the same cycle,
        # so the chart and the log cannot quote two different levels.
        self._record_levels(state, ts)
        if not self.quiet_read:
            _log(state, {"type": "analysis", "text": self._read_summary(last, ts, position, frame)})

        # Trading starts after the opening window, since the forecast does not
        # exist before it -- the notebook's `start_after`. The bar the plan was
        # built on is the last bar *of* that window, so it is never traded.
        if ts <= self.plan["opening_end"]:
            return "warming_up"

        if position > 0:
            exit_ = self._exit(frame, position)
            if exit_ is not None:
                quantity, reason, kind = exit_
                self._sell(state, tracker, quantity, last, reason, kind)
                return "sold"
            if self._held_note and fresh_bar:
                _log(state, {"type": "status", "text": self._held_note})
            # After the exits, so a bar wide enough to reach both the next rung
            # and the stop is the stop's. Never inside the flatten window: `_exit`
            # has already sold everything there.
            if (
                fresh_bar
                and self._can_add()
                and float(last["low"]) <= self.plan["buy_level"]
            ):
                if self._above_last_fill(state, last):
                    return "hold"
                if self._refuse_entry(state, frame, ts):
                    return "hold"
                return "bought" if self._add(state, tracker, last, position) else "hold"
            return "hold"

        # The session has been stood down: a stop, or a trade that closed for
        # too little (`_close_out`). Either way the levels have already been
        # tried today and found not to work, and re-arming them on the same tape
        # is how one bad trade becomes five.
        if self.plan.get("stand_down"):
            return "hold"

        if fresh_bar and float(last["low"]) <= self.plan["buy_level"]:
            if self.entry_refused(state, ts):
                return "hold"
            if self.closing_soon():
                _log(
                    state,
                    {
                        "type": "status",
                        "text": (
                            f"The {ts:%H:%M} bar traded down to the buy level, but the session "
                            f"is inside its last {self.config.flatten_before_close_min} min and "
                            "any position would be flattened straight back out. Standing down."
                        ),
                    },
                )
                return "hold"
            if self._refuse_entry(state, frame, ts):
                return "hold"
            return "bought" if self._buy(state, tracker, last) else "hold"
        return "hold"

    def _roll_session(self, today) -> None:
        """Forget yesterday's forecast at the start of a new session."""
        stale_plan = self.plan is not None and self.plan["date"] != today
        stale_block = self.blocked is not None and self.blocked["date"] != today
        if stale_plan:
            self.plan = None
            self.entry = None
            self.last_bar_ts = None
        if stale_block:
            self.blocked = None

    # --- the forecast ------------------------------------------------------

    def _plan_session(
        self, bundle: dict, state: AppState, frame, today, want: int,
        switching: bool = False,
    ) -> bool:
        """Forecast the day and set the two levels. False if it cannot be done.

        Every failure here is fatal for the session rather than for the bar --
        a daily history that is too short at 9:35 is still too short at 14:00
        -- so it is recorded in `self.blocked` and reported once. Except for a
        model switched to mid-run (`switching`): the failure is raised, and the
        run stays on the model it had (`_adopt_form_model`).
        """
        try:
            opening, tape = self._opening_window(state, frame, want)
            # A replay's state carries placeholder keys (every *dataset* read is
            # patched), so HighLow's SIP history falls back to the environment's
            # there -- the one read no dataset can serve.
            replayed = bool(getattr(state, "bar_tape_override", ""))
            forecast, open_price = session_forecast(
                bundle, self.ticker, opening, today,
                key=None if replayed else state.api_key,
                secret=None if replayed else state.api_secret,
            )
        except Exception as exc:
            if switching:
                raise
            self.blocked = {"date": today, "reason": str(exc)}
            _log(
                state,
                {
                    "type": "error",
                    "text": (
                        f"Apple Trader cannot forecast today's {self.ticker} range, so it "
                        "will not "
                        f"trade this session: {exc}"
                    ),
                },
            )
            return False

        self.plan = {
            "date": today,
            "opening_end": opening.index[-1],
            # The official print when there is one, else the 09:30 bar's open --
            # the same fallback `model_overlays._session_open_price` makes.
            "open_price": float(open_price or opening["open"].iloc[0]),
            **forecast,
        }
        # What the forecast started from, published with the record so the
        # chart can carry the prediction on past the last bar this run read --
        # after ▶ Stop -- without running the model again
        # (`model_overlays._after_record`).
        self.plan["seed"] = {
            "forecast": dict(forecast),
            "opening_end": opening.index[-1],
            "open_price": self.plan["open_price"],
        }
        self._set_levels(opening.index[-1])
        prior = self.slot.levels(state) or {}
        if (
            prior.get("rows")
            and prior.get("ticker") == self.ticker
            and pd.Timestamp(prior.get("date")) == pd.Timestamp(today)
            and _record_strategy(prior) == self.config.strategy
        ):
            # A restart mid-session: what the earlier run rested is history and
            # stays on the chart. This run's levels start at the bar it first
            # reads, not at 09:35 -- it rested nothing before it existed.
            self.plan["history"] = list(prior["rows"])
            if not switching:
                self._resume(state, prior.get("memory"))
        else:
            self._record_levels(state, opening.index[-1])

        # A HighLow bundle fitted on IEX openings (MU, every HighLow2 one)
        # fetched its own IEX window and was fitted on IEX volume, so the IEX
        # caveat does not describe it.
        warning = (
            None if (bundle.get("opening_feed") or "sip") != "sip"
            else _dayrange().volume_scale_warning(tape)
        )
        if warning:
            _log(state, {"type": "status", "text": f"Forecast caveat: {warning}"})
        _log(state, {"type": "analysis", "text": self._plan_summary()})
        self._catch_up(state, frame)
        return True

    def _catch_up(self, state: AppState, frame) -> None:
        """Walk the bars between the forecast and the newest one, one at a time.

        A run started (or restarted, or switched to another model) after 9:35
        makes the 9:35 forecast in `_plan_session`, and without this its first
        cycle would move the range once against everything the session has
        printed: a breach at 10:15 would be dated at the minute the agent
        started, and under "brownian" extended by the reach left at *that*
        minute rather than at 10:15. Walking each bar as `session_levels` does
        puts every revision at the bar that caused it, and leaves the plan where
        a run that had been up since 9:35 would have it. The newest bar is left
        to `run_cycle`, which reads it as its fresh bar.

        Bars the record does not have yet get a row, so the chart draws each
        revision where it happened; bars an earlier run today already recorded
        keep what it recorded. No orders and no stop on those rows -- nothing
        was running to rest one.
        """
        plan = self.plan
        missed = [ts for ts in frame.index[:-1] if ts > plan["opening_end"]]
        if not missed:
            return
        history = plan.setdefault("history", [])
        recorded_to = pd.Timestamp(history[-1]["t"]) if history else None
        revised, added = [], False
        for ts in missed:
            if self._move_range(frame[frame.index <= ts], ts) is not None:
                revised.append(ts)
            self._set_levels(ts)
            if recorded_to is None or ts > recorded_to:
                history.append(_levels_row(self.config, plan, ts))
                added = True
        if added:
            # Re-writes the last caught-up row (with an open position's stop, if
            # a restart resumed one) and publishes the record for the chart.
            self._record_levels(state, missed[-1])
        if revised:
            times = ", ".join(f"{ts:%H:%M}" for ts in revised[:8])
            if len(revised) > 8:
                times += f" and {len(revised) - 8} more"
            _log(
                state,
                {
                    "type": "analysis",
                    "text": (
                        f"{self.ticker}: caught up on the {len(missed)} bars since "
                        f"{plan['opening_end']:%H:%M}. The forecast was revised at the "
                        f"bars that traded outside it ({times}); predicted range is now "
                        f"${float(plan['pred_low']):,.2f} – ${float(plan['pred_high']):,.2f}, "
                        f"buy ${plan['buy_level']:,.2f}, sell ${plan['sell_level']:,.2f}."
                    ),
                },
            )

    def _opening_window(self, state: AppState, frame, want: int):
        return fetch_opening_window(state, frame, want, ticker=self.ticker)

    def _adopt_form_model(self, state: AppState, frame, today) -> bool:
        """Switch to the model the sidebar now names, if it changed.

        Returns True when the session's forecast was re-made, which `run_cycle`
        needs to know: the new forecast is the new model's 9:35 one and has not
        yet been moved by what the session has printed since.

        The model is what the levels hang off, so a switch re-forecasts the day
        with it -- from the same opening window -- and rebuilds both levels.
        The distances come with it: the form re-seeds them with the new model's
        tuned pair, so the old model's numbers under the new forecast would be
        a pair nobody picked. Everything else stays, as for a distance
        edit (`_adopt_form_levels`): the unit, the exits and the breach rules
        still need ▶ Start, and a form in another unit is refused.

        What the session has already decided carries over: an open position
        keeps its fill and its stop, a stood-down session stays stood down, and
        the chart's record keeps the old model's levels up to this minute (the
        re-forecast continues it, as a restart does). A new model that cannot
        forecast -- a missing file, a pairing it was never fitted on, a history
        it cannot read -- is refused and logged once, and the run carries on
        with the model it had.
        """
        form = self.slot.form(state)
        config = self.config
        if (
            form is None
            or (form.ticker or "").upper() != self.ticker.upper()
            or form.model_key == config.model_key
        ):
            # Back on the run's own model: a later switch is news again.
            self._refused_model = None
            return False
        ts = frame.index[-1]
        new, bundle, refusal = None, None, None
        if apple_models.strategy(form.model_key) != config.strategy:
            # Another rule set entirely (Fast&Furious trades a per-minute
            # signal, not levels): not something a running day-range agent can
            # turn into mid-session.
            refusal = (
                f"{apple_models.get(form.model_key).label} runs its own rules rather than the "
                "day-range levels; switching to it needs ▶ Stop and ▶ Start"
            )
        elif form.level_unit != config.level_unit:
            refusal = (
                f"its distances are counted in {form.unit_phrase} and this run counts in "
                f"{config.unit_phrase}; a new unit needs ▶ Start"
            )
        else:
            try:
                new = replace(
                    config, model_key=form.model_key,
                    buy_k=float(form.buy_k), sell_k=float(form.sell_k),
                )
            except ValueError as exc:
                refusal = str(exc)
            else:
                refusal = config_error(new)
                if refusal is None:
                    bundle = apple_models.load(new.model_key, self.ticker)
                    if bundle is None:
                        refusal = apple_models.unavailable_reason(new.model_key, self.ticker)

        was, now = apple_models.get(config.model_key).label, apple_models.get(form.model_key).label
        if refusal is None and self.plan is not None:
            before = (float(self.plan["buy_level"]), float(self.plan["sell_level"]))
            old = (self.config, self._bundle, self.plan)
            kept = {k: self.plan[k] for k in ("stand_down",) if k in self.plan}
            self.config, self._bundle = new, bundle
            try:
                self._plan_session(
                    bundle, state, frame, today, _dayrange().opening_minutes(bundle),
                    switching=True,
                )
            except Exception as exc:
                self.config, self._bundle, self.plan = old
                refusal = f"it cannot forecast today's {self.ticker} range: {exc}"
            else:
                self.plan.update(kept)

        if refusal is not None:
            if self._refused_model != form.model_key:
                self._refused_model = form.model_key
                _log(state, {"type": "error", "text": (
                    f"The sidebar names {now}, but the run stays on {was}: {refusal}."
                )})
            return False

        self._refused_model = None
        text = (
            f"{self.ticker} model switched from the sidebar at {pd.Timestamp(ts):%H:%M}: "
            f"{was} → {now}, buy {config.buy_k:g} → {new.buy_k:g}, sell "
            f"{config.sell_k:g} → {new.sell_k:g} × {new.unit_phrase}."
        )
        if self.plan is None:
            # Before 9:35 there is nothing to re-make: the forecast will simply
            # be the new model's. A model that could not forecast is not this
            # one, so its refusal does not stand either.
            self.config, self._bundle, self.blocked = new, bundle, None
            _log(state, {"type": "analysis", "text": text})
            return False
        text += (
            f" Re-forecast from the opening window: buy ${self.plan['buy_level']:,.2f} "
            f"(was ${before[0]:,.2f}), sell ${self.plan['sell_level']:,.2f} "
            f"(was ${before[1]:,.2f}), before this bar moves the range."
        )
        if self.entry is not None:
            text += " The open position keeps its fill and its stop."
        _log(state, {"type": "analysis", "text": text})
        return True

    def _adopt_form_levels(self, state: AppState, ts) -> None:
        """Take the buy and sell distances the sidebar now holds, if they changed.

        The one part of the configuration a run can be steered by mid-session:
        the two distances say where the orders rest, which is exactly what a
        person watching the tape wants to nudge, and changing them touches no
        position -- an open trade keeps the stop it was filled with (`_risk`).
        Everything else still needs ▶ Start, which re-reads the whole form.

        Only a form for this run's instrument and model is read, and only when
        its distances are counted in the same unit: 0.8 × ADR and 0.8 × the
        predicted range are different orders, and adopting the number without
        the unit would rest them somewhere nobody asked for.
        """
        form = self.slot.form(state)
        config = self.config
        if (
            form is None
            or (form.ticker or "").upper() != self.ticker.upper()
            or form.model_key != config.model_key
        ):
            return
        new = (float(form.buy_k), float(form.sell_k))
        old = (float(config.buy_k), float(config.sell_k))
        if new == old:
            return
        if form.level_unit != config.level_unit:
            refused = (new, form.level_unit)
            if self._refused_form != refused:
                self._refused_form = refused
                _log(
                    state,
                    {
                        "type": "status",
                        "text": (
                            f"The sidebar's buy/sell distances are counted in "
                            f"{form.unit_phrase} but this run counts in "
                            f"{config.unit_phrase}; a new unit needs ▶ Start, so the "
                            f"levels stay at {old[0]:g} / {old[1]:g} × "
                            f"{config.unit_phrase}."
                        ),
                    },
                )
            return
        try:
            self.config = replace(config, buy_k=new[0], sell_k=new[1])
        except ValueError as exc:  # the form repairs a crossed pair; belt and braces
            _log(state, {"type": "error", "text": f"Sidebar levels not applied: {exc}"})
            return
        text = (
            f"{self.ticker} levels changed from the sidebar at {pd.Timestamp(ts):%H:%M}: "
            f"buy {old[0]:g} → {new[0]:g}, sell {old[1]:g} → {new[1]:g} × "
            f"{config.unit_phrase} below the predicted high."
        )
        if self.plan is not None:
            was = (float(self.plan["buy_level"]), float(self.plan["sell_level"]))
            self._set_levels(ts)
            text += (
                f" Buy ${self.plan['buy_level']:,.2f} (was ${was[0]:,.2f}), sell "
                f"${self.plan['sell_level']:,.2f} (was ${was[1]:,.2f})."
            )
        if self.entry is not None:
            text += " The open position keeps the stop it was filled with."
        _log(state, {"type": "analysis", "text": text})

    def _record_levels(self, state: AppState, ts) -> None:
        """Add this cycle's levels to the session's record, and publish it.

        The chart draws this record while the run is live
        (`model_overlays.live_overlays`) rather than re-deriving the levels,
        because only the run knows what it actually rested: the forecast it
        made from its own opening window, each sidebar edit at the minute it
        was adopted, and a restart mid-session. One row per cycle, keyed by the
        bar the cycle read; a second write for the same bar replaces the first.

        `stop` is the position's own stop while long -- the number the log line
        quotes -- and None while flat: it is measured from the actual fill, so
        before a buy there is no stop to draw.
        """
        plan = self.plan
        row = _levels_row(self.config, plan, ts)
        if self.entry is not None:
            if self.entry.get("runner") and not self.entry.get("loss_runner"):
                row["stop"] = float(self.entry["price"])
            else:
                row["stop"] = self._stop_price()
        history = plan.setdefault("history", [])
        if history and history[-1]["t"] == row["t"]:
            history[-1] = row
        else:
            history.append(row)
        self.slot.publish(state, {
            "ticker": self.ticker,
            "date": plan["date"],
            "config": self.config,
            "rows": history,
            "memory": self.memory(),
            "seed": plan.get("seed"),
        })

    def memory(self) -> dict:
        """What this run knows about the day that the ledger does not: the open
        position's own record (its fill, stop, ladder, what a partial exit has
        banked) and a session stand-down. Published beside the levels and saved
        with the session (`session_store`), so a restart the same day picks the
        position up where it was rather than re-adopting it at the last close,
        and does not re-arm a session that had stood down."""
        return {
            "entry": dict(self.entry) if self.entry is not None else None,
            "stand_down": (self.plan or {}).get("stand_down"),
        }

    def publish_memory(self, state: AppState) -> None:
        """Refresh `memory` on the published levels after a cycle -- an order
        placed after this cycle's levels were recorded would otherwise reach the
        session file only a cycle later."""
        levels = self.slot.levels(state)
        day = (self.plan or {}).get("date")
        if (
            not levels
            or day is None
            or levels.get("ticker") != self.ticker
            or pd.Timestamp(levels.get("date")) != pd.Timestamp(day)
        ):
            return
        self.slot.publish(state, {**levels, "memory": self.memory()})

    def _resume(self, state: AppState, memory: "dict | None") -> None:
        """Take back what an earlier run today knew (`memory`), on a restart.

        The entry is only a candidate: `run_cycle` drops it at once if the
        ledger it continues is flat, and keeps it only while shares are held.

        A stop-out's stand-down is the one thing not taken back: the stop ended
        that run, and pressing ▶ Start again is the say-so to trade again today
        -- unless `rearm_after_stop` is off (a split Orchestra's pairs), when it
        holds like any other. The breaker's stand-down (`_close_out`) did not
        stop the run, so it still holds."""
        if not memory:
            return
        stand_down = memory.get("stand_down")
        if stand_down == self.STOPPED_OUT and self.rearm_after_stop:
            _log(
                state,
                {
                    "type": "status",
                    "text": (
                        f"{self.ticker}: today's earlier run was stopped out; started "
                        "again, so the levels are armed again."
                    ),
                },
            )
        elif stand_down and not self.plan.get("stand_down"):
            self.plan["stand_down"] = stand_down
            _log(
                state,
                {
                    "type": "status",
                    "text": (
                        f"{self.ticker}: continuing today's run, which had stood down "
                        f"({stand_down}) — still no new entries today."
                    ),
                },
            )
        entry = memory.get("entry")
        if entry and self.entry is None:
            self.entry = dict(entry)

    # --- the levels, and the forecast they hang off ------------------------

    def _reference(self, ts) -> float:
        """The number the two distances are measured below, on this bar: the
        predicted high, as `_update_range` has left it -- one number, the same
        at 09:36 and at 15:50 until the session breaches it, and the notebook's
        rule."""
        return float(self.plan["pred_high"])

    def _set_levels(self, ts) -> None:
        """Rebuild the two resting levels from the reference this bar has.

        Called when the plan is made, every time `_update_range` moves the
        forecast, and once per closed bar. That is the whole of "the levels follow the
        forecast": they are never stored independently of it, so there is no way
        for the two to disagree.

        `sell_k < buy_k` is enforced on the config, and both distances are
        subtracted from the same reference in the same unit, so the sell level
        sits above the buy level at every minute however either moves.

        The unit is re-read here rather than cached with the plan because under
        `level_unit = "pred_range"` it is a function of the forecast, and the
        forecast is exactly what `_update_range` moves. That is the intended
        behaviour -- a day the tape has shown to be wider than predicted gets
        wider distances, not just higher ones -- and it means the dollar gap
        between the two levels grows over such a session, where under "adr" it
        is the same all day. The plan carries the current unit for the log
        lines and the chart, which must not compute a second one.
        """
        plan, config = self.plan, self.config
        unit = level_unit(config, plan)
        reference = self._reference(ts)
        plan["reference"] = reference
        plan["level_unit"] = unit
        plan["buy_level"] = reference - self._buy_k() * unit
        plan["sell_level"] = reference - config.sell_k * unit

    # --- adding to a position on the way down --------------------------------

    def _buy_k(self) -> float:
        """How far under the reference the buy level rests on this bar, in units.

        `buy_k` while flat. While long with another add still possible it is the
        next rung of the ladder (`_rung_k`); otherwise `buy_k` again, the level
        a position that cannot be added to has no use for but the chart and the
        log have always shown.
        """
        if self._can_add():
            return self._rung_k(int(self.entry["fills"]))
        return float(self.config.buy_k)

    def _rung_k(self, fills: int) -> float:
        """The distance of the buy that would follow `fills` buys, in units.

        Each buy moves the next `buy_step_k` units further down: 0.55, 0.65,
        0.75 for AAPL's 0.55 at a 0.1 step. Counted from the position's fills,
        which go with it, so selling it out is what puts the buy back at `buy_k`;
        a partial sell keeps the count.

        Under a step of 0, the rule a record from before 2026-10-01 replays,
        each add rests half-way between the last buy and the bottom of the
        range, `reference - 1 x unit`: the predicted low under "pred_range",
        one ADR under the reference under "adr". So the gap left to the bottom
        halves with every fill -- 0.40, 0.70, 0.85, 0.925 for a 0.40 -- and no
        add is ever placed under it.

        Counted from `buy_k` rather than stored per rung, so a sidebar edit to
        the buy distance moves the whole ladder with it, the same way it moves
        the first buy.
        """
        buy_k = float(self.config.buy_k)
        step = float(self.config.buy_step_k)
        if step:
            return buy_k + step * fills
        return 1.0 - (1.0 - buy_k) / 2 ** fills

    def _above_last_fill(self, state: AppState, bar) -> bool:
        """Whether this bar's add is refused for not being lower (`add_under_fill`).

        The add fills at the close, so that is what has to be under the last
        fill -- a bar that dipped to the rung and closed back above the price
        already paid would buy more of the same position dearer, not lower.
        Logged, because a rung touched and not bought otherwise looks like a
        missed order. The next bar is judged again.
        """
        if not self.config.add_under_fill:
            return False
        close, fill = float(bar["close"]), float(self.entry["last_fill"])
        if close < fill:
            return False
        _log(
            state,
            {
                "type": "status",
                "text": (
                    f"The {bar.name:%H:%M} bar reached the ${self.plan['buy_level']:,.2f} "
                    f"next buy, but closed at ${close:,.2f}, not under the "
                    f"${fill:,.2f} last fill. Not buying again at no better a price; "
                    "the next bar is judged again."
                ),
            },
        )
        return True

    def _can_add(self) -> bool:
        """Whether the open position may still be added to.

        `can_add` is decided at each fill, from the cash that fill left
        (`_after_fill`). A runner is never added to -- the momentum take already
        decided to take money off this position -- and under the half-way rule
        a buy distance at or under the bottom of the range leaves nowhere lower
        to go (`ladder_has_room`).
        """
        entry = self.entry or {}
        return bool(
            self.config.can_scale_in
            and entry.get("can_add")
            and entry.get("fills")
            and not entry.get("runner")
            and self.config.ladder_has_room
        )

    def _momentum_read(self, frame) -> "dict | None":
        """This bar's read of the behaviour table (`momentum_read`), or None."""
        return momentum_read(
            frame["close"], self.config.momentum_confirmation_bars, self.minute_move
        )

    def _momentum_unknown(self, frame) -> str:
        """Why the confirmation cannot read this bar -- for the log line that
        says a level was reached and nothing was done about it."""
        return momentum_unknown(
            self.ticker, int(self.config.momentum_confirmation_bars), self.minute_move, len(frame)
        )

    def _refuse_entry(self, state: AppState, frame, ts) -> bool:
        """Log and refuse a buy at the buy level the tape does not support.

        With the momentum confirmation on, the behaviour table decides: a buy
        needs positive momentum, or neutral momentum whose change is not
        negative -- flat, or about to rise.
        Without it, the legacy fall gate a stored record replays under
        (`_refuse_falling`). Either way this bar only: the next is judged again.
        """
        if not self.config.momentum_confirmation_bars:
            return self._refuse_falling(state, frame, ts)
        read = self._momentum_read(frame)
        if read is not None and read["row"].buy:
            return False
        why = momentum_words(read) if read is not None else self._momentum_unknown(frame)
        _log(
            state,
            {
                "type": "status",
                "text": (
                    f"The {ts:%H:%M} bar traded down to the "
                    f"${self.plan['buy_level']:,.2f} buy level, but {why}. Not buying "
                    "yet; the next bar is judged again."
                ),
            },
        )
        return True

    def _hold_at_target(self, frame) -> "str | None":
        """Why the sell level is not sold on this bar, or None to sell it.

        Only with the momentum confirmation on: the table holds past the target
        while momentum is still positive -- the move has not ended -- and the
        rule holds whenever it cannot read the bar at all.
        """
        if not self.config.momentum_confirmation_bars:
            return None
        read = self._momentum_read(frame)
        if read is None:
            return self._momentum_unknown(frame)
        return None if read["row"].sell_target else momentum_words(read)

    def _falling(self, frame) -> "str | None":
        """Why the price is falling too fast to buy right now, or None if it is not.

        The fall is `close - close[fall_bars bars ago]` in dollars -- the
        absolute momentum the live chart's panel draws -- against `max_fall_k`
        level units. Early in the session, with fewer bars than the look-back,
        it is the change since the first regular-session bar: the whole of the
        move so far is the fall there is to judge.
        """
        limit_k = float(self.config.max_fall_k)
        if not limit_k or self.plan is None or not len(frame):
            return None
        closes = frame["close"]
        n = self.config.fall_bars
        before = float(closes.iloc[-1 - n]) if len(closes) > n else float(closes.iloc[0])
        change = float(closes.iloc[-1]) - before
        unit = level_unit(self.config, self.plan)
        if not unit or change >= -limit_k * unit:
            return None
        span = f"{n} bars" if len(closes) > n else f"{len(closes) - 1} bars since the open"
        return (
            f"the price has fallen ${-change:,.2f} over the last {span} "
            f"({change / unit:+.2f} × the {self.config.unit_phrase}), steeper than the "
            f"{limit_k:g} × {self.config.unit_phrase} (${limit_k * unit:,.2f}) an entry "
            "is allowed into"
        )

    def _refuse_falling(self, state: AppState, frame, ts) -> bool:
        """Log and refuse a buy the price reached by falling too fast.

        Refuses this bar only: the next one is judged afresh, so a fall that
        eases with the price still at the buy level is bought then.
        """
        why = self._falling(frame)
        if why is None:
            return False
        _log(
            state,
            {
                "type": "status",
                "text": (
                    f"The {ts:%H:%M} bar traded down to the "
                    f"${self.plan['buy_level']:,.2f} buy level, but {why}. Not buying "
                    "into the fall; the next bar is judged again."
                ),
            },
        )
        return True

    def _stop_price(self) -> "float | None":
        """Where this position's stop sits, in dollars -- or None with no stop.

        Frozen at each fill (`_after_fill`) for the reason `_risk` is: a stop
        must not move under a position on its own. The usual distance under
        the last actual fill -- or, for a record from before 2026-09-28
        (`stop_under_next_buy`), under the next buy while cash could still pay
        for it.

        A position adopted on a restart has no recorded stop and falls back to
        the old reading, `risk` under the fill it was adopted at.
        """
        entry = self.entry or {}
        if "stop" in entry:
            return entry["stop"]
        risk = self._risk()
        if not risk or not entry.get("price"):
            return None
        return float(entry["price"]) - risk

    def _after_fill(self, state: AppState, tracker: DecisionTracker, bar) -> None:
        """Settle what a buy leaves behind: the stop, and the next rung above it.

        Called after the first buy and after every add. The stop is frozen
        `risk` under this fill -- the price actually paid, which the momentum
        confirmation had as much say in as the forecast, rather than the buy
        level the forecast alone put there. Whether another add is possible is
        decided here, once, from the cash this fill left and the price of the
        next rung -- the same whole-share sizing the buy itself uses -- and from
        where that rung sits against the stop: a bar that reaches a rung under
        the stop has been through the stop first, and `_exit` reads the stop
        first. The levels are then rebuilt, so the plan and the line below
        quote the rung the position is now waiting on.

        A record from before 2026-09-28 (`stop_under_next_buy`) replays the old
        rule instead: the stop under the next rung while one is affordable.
        """
        config, plan, entry = self.config, self.plan, self.entry
        ts = bar.name
        risk = self._risk()
        fill = float(entry["last_fill"])
        rung_k = self._rung_k(int(entry["fills"]))
        rung = float(plan["reference"]) - rung_k * level_unit(config, plan)
        cash = float(tracker.snapshot()["cash"])
        shares = rule_agent.order_quantity(cash, rung, config.position_pct)
        affordable = bool(
            config.can_scale_in and config.ladder_has_room and shares > 0
        )
        if config.stop_under_next_buy:
            entry["can_add"] = affordable
            anchor = rung if affordable else fill
        else:
            entry["can_add"] = affordable and not (risk and rung <= fill - risk)
            anchor = fill
        entry["stop"] = anchor - risk if risk else None
        entry["stop_under"] = (
            "next buy" if config.stop_under_next_buy and affordable else "fill"
        )
        self._set_levels(ts)
        if not config.can_scale_in:
            return
        position = tracker.position_for(self.ticker)
        held = (
            f"{self.ticker} holds {position:g} sh at an average ${entry['price']:,.2f} "
            f"after {int(entry['fills'])} buy(s)."
        )
        if entry["can_add"]:
            if not risk:
                stop = ""
            elif entry["stop_under"] == "next buy":
                stop = (
                    f" The stop sits {stop_phrase(config)} (${risk:,.2f}) under that, at "
                    f"${entry['stop']:,.2f}, so the price can reach the next buy before it."
                )
            else:
                stop = (
                    f" The stop sits {stop_phrase(config)} (${risk:,.2f}) under the "
                    f"${fill:,.2f} fill, at ${entry['stop']:,.2f}, below the next buy."
                )
            if config.buy_step_k:
                where = f"{config.buy_step_k:g} lower than the last buy's"
            else:
                where = (
                    f"half-way to the "
                    f"${float(plan['reference']) - level_unit(config, plan):,.2f} bottom "
                    "of the range"
                )
            text = (
                f"{held} Next buy at ${plan['buy_level']:,.2f} ({rung_k:g} × "
                f"{config.unit_phrase} under the predicted high, {where}): ${cash:,.2f} "
                f"cash pays for {shares:g} more share(s).{stop}"
            )
        else:
            if not config.ladder_has_room:
                why = "the buy distance leaves nowhere lower to go"
            elif not affordable:
                why = f"${cash:,.2f} cash buys no share at the ${rung:,.2f} next rung"
            else:
                why = (
                    f"the ${rung:,.2f} next rung is at or under the stop, so a bar "
                    "reaching it would be stopped out first"
                )
            stop = (
                f" The stop sits {stop_phrase(config)} (${risk:,.2f}) under the last "
                f"${fill:,.2f} fill, at ${entry['stop']:,.2f}."
                if risk else ""
            )
            text = f"{held} No further buys: {why}.{stop}"
        _log(state, {"type": "analysis", "text": text})

    def _add(
        self, state: AppState, tracker: DecisionTracker, bar, position: float
    ) -> bool:
        """Buy more of the open position at the next rung of the ladder.

        Sized like any entry -- `position_pct` of the cash that is left -- and
        folded into the one position the exits manage: the entry price becomes
        the average cost, which the target, the momentum take, the breakeven
        and the circuit breaker all measure from, and the momentum take starts
        looking for its bounce from this fill rather than from the first. What
        the position has banked and how many bars it has been open carry over.
        """
        before = dict(self.entry)
        rung_k = self._buy_k()
        if self.config.buy_step_k:
            where = f"{self.config.buy_step_k:g} lower than the last buy's"
        else:
            where = "half-way from the last buy to the bottom of the range"
        reasoning = (
            f"The bar traded down to ${float(bar['low']):,.2f}, at or through the "
            f"${self.plan['buy_level']:,.2f} next buy level — {rung_k:g} × the "
            f"{self.config.unit_phrase} below the predicted high, {where}. Adding to the "
            f"{position:g} sh bought at an average ${float(before['price']):,.2f}."
        )
        if not self.buy(
            state, tracker, float(bar["close"]), reasoning,
            limit_price=self._buy_limit(),
        ):
            if self._limit_missed():
                # The price was back above the rung when the order went out.
                # Not a reason to end the ladder: the rung is still where it
                # was, and the next bar that reaches it tries again.
                self._note_limit_miss(state, "next buy")
                return False
            # Nothing bought -- the cash at this close did not stretch to a share,
            # or the order was refused. The ladder ends here; the stop stays
            # where the last fill put it, since it must not move on its own.
            self.entry = {**before, "can_add": False}
            self._set_levels(bar.name)
            return False
        fill = float(self.entry["price"])
        held = tracker.position_for(self.ticker)
        added = held - position
        average = (float(before["price"]) * position + fill * added) / held
        self.entry = {
            **before,
            "price": average,
            "ts": bar.name,
            "fills": int(before["fills"]) + 1,
            "last_fill": fill,
        }
        self._after_fill(state, tracker, bar)
        return True

    def _update_range(self, state: AppState, frame, ts) -> None:
        """Move the forecast the session has traded through, and the levels with it.

        The forecast is a statement about the width of the day, made at 9:35 off
        five minutes of tape. When the session prints a price outside it that
        statement has been falsified in one direction, and going on measuring
        against it -- the position's target in particular -- means trading
        against a number the tape has already disproved. So the breached side is
        moved (`dayrange_model.updated_range` decides how far; `breach_update`
        picks the policy) and both levels are rebuilt from the new high.

        Deliberately once per closed bar and only outside the opening window:
        the bar the plan was built on is the last bar of that window, and its
        extremes are already in the forecast through `apply_open_constraint`.

        What this does to a run in flight, in both directions. An open position's
        sell level moves *up*, so a day that is running further than forecast is
        held for more of it rather than being handed the old target. And the buy
        level moves up with it -- which on the bar of a breach can put it above
        where the bar traded, so a strategy that had stood aside all morning can
        enter on the bar that moved the level. That is the intended reading (the
        dip is measured from where the day is *now* expected to top out, not from
        a number it has outgrown) but it is a real change in when the agent
        trades, and it is why the stop matters more under these policies than
        under "off": a day that ratchets the high all afternoon will keep
        re-arming the entry until a stop ends the session's trading.
        """
        before = self._move_range(frame, ts)
        if before is not None:
            _log(state, {"type": "analysis", "text": self._range_summary(ts, before)})

    def _move_range(self, frame, ts) -> "dict | None":
        """`_update_range` without the log line: the levels as they were, or None.

        Split out because the chart overlay walks a session through exactly this
        (`session_levels`) and must not invent a second reading of the same
        settings -- but it has no agent log to write to, and a decoration that
        logged would be a decoration with side effects.
        """
        config, plan = self.config, self.plan
        if plan is None:
            return None
        # "off" with containment on still has work to do -- it does not lead
        # the tape, but it does not keep a high the tape has passed either --
        # so the early return is about having *neither*, not about the policy.
        if config.breach_update == BREACH_OFF and not config.contain_range:
            return None
        if ts <= plan["opening_end"]:
            return None

        dayrange = _dayrange()
        # Re-based to this bar's minute first, so that under a reference which
        # moves with the clock the "was" in the log line is this minute's level
        # without the breach rather than last minute's with it -- the breach's
        # own effect, which is what the line is about.
        self._set_levels(ts)
        # The width the forecast was made at, kept before anything can move it
        # -- this is the first bar past the opening window that gets here.
        plan.setdefault(
            "forecast_width", float(plan["pred_high"]) - float(plan["pred_low"])
        )
        before = {k: float(plan[k]) for k in
                  ("pred_high", "pred_low", "buy_level", "sell_level")}
        # A forecast of the range *after* the opening window (HighLow2's rest
        # head) says nothing about the window itself, so its own extremes are not a
        # breach of it: the tape is read from the first bar after the window.
        # Every other forecast covers the whole session and is clipped to
        # contain the window -- the window *as the forecast read it*, which is
        # what `or_high`/`or_low` are. Not re-read from the frame: the live
        # buffer swaps the window's bars for SIP's once they are 15 minutes old
        # (`bar_history.SETTLED_BAR_AGE_MIN`), and SIP's, with every venue's
        # prints, can reach a few cents past the Finnhub/IEX bar the forecast
        # was made from. Re-read, that revision was a "breach" no bar after the
        # forecast had traded (AAPL 2026-10-12: the 09:30 low went $337.63 →
        # $337.60 at 09:46, against a predicted low of $337.63, and both sides
        # of the range moved).
        traded = frame[frame.index > plan["opening_end"]]
        session_high = float(traded["high"].max())
        session_low = float(traded["low"].min())
        if not plan.get("range_after_opening"):
            window = frame[frame.index <= plan["opening_end"]]
            or_high = plan.get("or_high")
            or_low = plan.get("or_low")
            session_high = max(session_high, float(
                or_high if or_high is not None else window["high"].max()
            ))
            session_low = min(session_low, float(
                or_low if or_low is not None else window["low"].min()
            ))
        high, low = dayrange.updated_range(
            plan,
            session_high=session_high,
            session_low=session_low,
            minutes_left=dayrange.minutes_left_at(ts),
            policy=config.breach_update,
            contain=config.contain_range,
            width=breach_width(config, plan),
        )
        if high == before["pred_high"] and low == before["pred_low"]:
            return None

        plan["pred_high"], plan["pred_low"] = high, low
        plan["range_updates"] = int(plan.get("range_updates", 0)) + 1
        self._set_levels(ts)
        # Which side the tape went through, as distinct from which side moved:
        # under "shift" the other side follows without having been breached.
        before["breached_high"] = session_high > before["pred_high"]
        before["breached_low"] = session_low < before["pred_low"]
        return before

    def _range_summary(self, ts, before: dict) -> str:
        """The one line an update writes: what the tape did, and what moved.

        Whether a breach of the *low* moves anything depends on the unit: under
        "adr" the levels are built from the predicted high alone, so it moves
        the forecast and nothing else; under "pred_range" the low is half the
        unit, so both levels move with it. The line is written from the levels
        themselves rather than from the policy for exactly that reason -- a log
        that reported levels that had not changed (or missed ones that had)
        would be the reader's problem on every grinding session, which is
        exactly when it matters.
        """
        plan, config = self.plan, self.config
        dayrange = _dayrange()
        moved = []
        breached_high = before.get("breached_high", plan["pred_high"] != before["pred_high"])
        breached_low = before.get("breached_low", plan["pred_low"] != before["pred_low"])
        if breached_high:
            moved.append(
                f"the session has traded up through the ${before['pred_high']:,.2f} "
                "predicted high"
            )
        if breached_low:
            moved.append(
                f"it has traded down through the ${before['pred_low']:,.2f} predicted low"
            )
        width = breach_width(config, plan)
        if config.breach_update == BREACH_BROWNIAN:
            left = dayrange.minutes_left_at(ts)
            reach = dayrange.brownian_reach(plan["adr14_abs"], left)
            how = (
                f"the extreme so far, extended by the ${reach:,.2f} a driftless walk with "
                f"this ADR's volatility is still expected to add over the {left:.0f} min left"
            )
            if width and breached_high != breached_low:
                how += f", and the other side ${width:,.2f} from it"
        elif width and breached_high != breached_low:
            how = (
                f"the breached side moved to the extreme so far and the other ${width:,.2f} "
                "from it, no further than the session has printed"
            )
        elif config.breach_update == BREACH_SHIFT and breached_high != breached_low:
            how = (
                "the breached side moved to the extreme so far and the other with it, "
                "no further than the session has printed"
            )
        else:
            how = "the extreme so far"
        if not moved:
            # Containment alone, with nothing breached this bar, cannot move a
            # side -- but say something true rather than nothing if it ever does.
            moved.append("the forecast was widened to hold the session so far")
        if plan["buy_level"] == before["buy_level"]:
            # Only the low moved. Nothing this strategy rests on hangs off it.
            levels = (
                f"The buy and sell levels are built from the predicted high, so they stay "
                f"at ${plan['buy_level']:,.2f} and ${plan['sell_level']:,.2f}."
            )
        else:
            levels = (
                f"Both levels are rebuilt from it: buy ${plan['buy_level']:,.2f} (was "
                f"${before['buy_level']:,.2f}), sell ${plan['sell_level']:,.2f} (was "
                f"${before['sell_level']:,.2f})."
            )
        return (
            f"{self.ticker} forecast updated at {ts:%H:%M}: {', and '.join(moved)}. "
            f"Predicted range is now ${float(plan['pred_low']):,.2f} – "
            f"${float(plan['pred_high']):,.2f} ({how}). {levels}"
        )

    # --- the check on an open position -------------------------------------

    # Which rule closed (or trimmed) a position -- carried on the log entry.
    EXIT_BREAKEVEN = "breakeven"
    EXIT_STOP = "stop"
    EXIT_TARGET = "target"
    EXIT_BREACH = "breach"
    EXIT_FLATTEN = "flatten"
    EXIT_TAKE = "momentum_take"

    def _exit(self, frame, position: float) -> "tuple[float, str, str] | None":
        """What to sell on this bar, why, and under which rule -- or None to hold.

        Returns `(quantity, reasoning, kind)`. The checks run in a fixed order
        and the first that applies takes the bar:

        1. **breakeven** -- a runner (what a momentum take left behind) is sold
           once a bar's low comes back to the fill. It was kept to wait for the
           sell level, not to hand back the gain the take already banked. Not
           a runner kept from a take at or under the fill (`take_in_loss`):
           it is under the fill already, so it is left to the stop.
        2. **stop** -- a bar's low at `fill - stop_distance`, which is a share
           of what the trade is playing for (`stop_gain_fraction`): the day went
           the other way from the forecast. Everything is sold, and `run_cycle`
           takes no new entry for the rest of the session.
        3. **target** -- a bar's high at the sell level. Everything. With the
           momentum confirmation on, held past the level while momentum is
           still positive (or cannot be read yet); the breach exit likewise.
        4. **flatten** -- the closing bell. Everything.
        5. **momentum take** -- see `_momentum_take`.

        The price rules trigger on a touch -- the low for the two stops, the
        high for the target -- like the resting levels the entry already
        models; the momentum take reads the close. The stops come before the
        target because a bar wide enough to touch both says nothing about which
        came first, so it is read the careful way.

        With the stop and the momentum take both 0 only the target and the
        flatten are left, which is notebook 05's rule as specified.
        """
        config, plan = self.config, self.plan
        bar = frame.iloc[-1]
        entry = self.entry or {}
        entry_price = entry.get("price") or 0.0
        price, low, high = float(bar["close"]), float(bar["low"]), float(bar["high"])
        pnl_pct = (price / entry_price - 1) * 100 if entry_price else 0.0
        self._held_note = None

        if entry.get("runner") and not entry.get("loss_runner") and low <= entry_price:
            return position, (
                f"Breakeven: the runner the take left traded back down to "
                f"${low:,.2f}, at or under the ${entry_price:,.2f} fill. It was kept for the "
                f"${plan['sell_level']:,.2f} sell level, not to give the gain back, so the "
                f"rest is sold at market ({pnl_pct:+.2f}%)."
            ), self.EXIT_BREAKEVEN

        risk = self._risk()
        stop = self._stop_price()
        if stop is not None and entry_price:
            if low <= stop:
                under = (
                    f"${stop + risk:,.2f} next buy"
                    if entry.get("stop_under") == "next buy"
                    else f"${float(entry.get('last_fill') or entry_price):,.2f} fill"
                )
                return position, (
                    f"Stop loss: the bar traded down to ${low:,.2f}, at or through the "
                    f"${stop:,.2f} stop ({stop_phrase(config)}, ${risk:,.2f}, under the "
                    f"{under}). The day is not going the way the forecast "
                    f"said, so the position is closed at market ({pnl_pct:+.2f}%) and "
                    "nothing more is bought this session."
                ), self.EXIT_STOP

        # Reached the sell level, or traded through the forecast: both are the
        # table's "price above the sell target", so the confirmation may hold
        # them past this bar. Computed only when one of them is in reach.
        breached = plan.get("high_at_bar", plan["pred_high"])
        in_reach = high >= plan["sell_level"] or (config.breach_exit and high > breached)
        held = self._hold_at_target(frame) if in_reach else None
        if held is not None:
            self._held_note = (
                f"The bar traded up to ${high:,.2f}, "
                + (
                    f"at or through the ${plan['sell_level']:,.2f} sell level"
                    if high >= plan["sell_level"]
                    else f"through the ${breached:,.2f} predicted high"
                )
                + f", but {held}. Holding on; the next bar is judged again."
            )

        if held is None and high >= plan["sell_level"]:
            return position, (
                f"Target: the bar traded up to ${high:,.2f}, at or through the "
                f"${plan['sell_level']:,.2f} sell level "
                f"({config.sell_k:g} × {config.unit_phrase} under the predicted high, "
                f"${plan['reference']:,.2f}). Selling at market ({pnl_pct:+.2f}%)."
            ), self.EXIT_TARGET

        # After the target, so a breach that also reached the sell level is
        # logged as the target exit it is -- which is every breach under "off"
        # and "extreme", where the sell level sits `sell_k` units under the
        # high the bar just traded through. What is left for this rule is the
        # case it exists for: "brownian", where the forecast leads the tape and
        # the target is carried past the bar that settled the bet.
        if held is None and config.breach_exit and high > breached:
            return position, (
                f"Breach exit: the bar traded up to ${high:,.2f}, through the "
                f"${breached:,.2f} predicted high the levels were resting under. The "
                f"position was a bet that the day tops out around there and the tape has "
                f"just settled it, at a better price than the ${plan['sell_level']:,.2f} "
                f"sell level was offering, so it is sold at market ({pnl_pct:+.2f}%) "
                "rather than held against a forecast that has moved."
            ), self.EXIT_BREACH

        if self.closing_soon():
            to_close = market_hours.seconds_to_close() or 0.0
            return position, (
                f"Session ends in {to_close / 60:.0f} min and the day never came back up to "
                f"${plan['sell_level']:,.2f}. The forecast is a statement about today "
                f"only, so the position is flattened rather than carried overnight "
                f"({pnl_pct:+.2f}%)."
            ), self.EXIT_FLATTEN

        # A bar held at the sell level is above the target, where the take (a
        # sale *short* of it) does not apply.
        if in_reach:
            return None
        return self._momentum_take(frame, position, entry_price, price)

    def _momentum_take(
        self, frame, position: float, entry_price: float, price: float
    ) -> "tuple[float, str, str] | None":
        """Bank gains short of the target once momentum has been negative for long enough.

        Fires when the position is in profit (or at all, with `take_in_loss`),
        `take_after_minutes` or more have
        passed since the last fill (`entry["ts"]`, which an add moves) -- and,
        on a legacy record, the close is `take_min_gain_fraction` of the
        predicted gain over the fill -- and the momentum confirmation's
        table says take (`MOMENTUM_TABLE`): momentum over the look-back is
        negative and its change neutral or negative -- the price is still
        dropping, or dropping faster. A drop that is slowing (negative momentum,
        positive change) is left alone. Under `take_trigger = "zone"` the
        trigger is a stall zone formed since the fill instead (`_zone_formed`),
        and everything else here is the same. A legacy record reads the rule it was
        run under instead: `negative_momentum_bars` the N-bar momentum negative
        for `negative_for_bars` bars in a row since the entry
        (`_momentum_negative`),
        `momentum_fade_bars` the positive-to-balanced turn (`_fade_turned`),
        `momentum_drop` the fall from the peak (`_fade_dropped`).

        What it sells depends on how much the forecast still promises. If the
        sell level is `hold_min_gain_k` level units or more above the fill,
        `take_fraction` of the shares go and the rest is kept as a runner --
        left to the sell level, the flatten, or the breakeven. Short of that the
        target is too close to be worth the wait and everything goes. Once per
        position: a runner is never trimmed again. The split is the same at a
        loss (the user's choice); that runner just has no breakeven (`_sell`
        marks it `loss_runner`), so it rides on the stop.

        The momentum is recomputed over the session on each call rather than
        tracked bar by bar, so a cycle that missed a bar still sees the turn.
        """
        config, plan = self.config, self.plan
        entry = self.entry or {}
        if (
            not config.has_take
            or entry.get("runner")
            or not entry_price
            or (price <= entry_price and not config.take_in_loss)
        ):
            return None
        # Not before the trade has had its time: until then a negative read is
        # noise around the fill, which the stop is there for. Bar time, not
        # the wall clock, so a replay waits exactly as the live loop does.
        since = entry.get("ts", frame.index[-1])
        if frame.index[-1] - pd.Timestamp(since) < timedelta(
            minutes=config.take_after_minutes
        ):
            return None
        # The legacy gate: not before the trade has banked its share of what
        # it is playing for. Read at the current unit, like the runner test.
        unit = level_unit(config, plan)
        if (
            config.take_min_gain_fraction
            and price - entry_price < config.take_min_gain_fraction * config.target_gain_k * unit
        ):
            return None

        if config.take_trigger == TAKE_ZONE:
            why = self._zone_formed(frame, since)
        elif config.momentum_confirmation_bars:
            read = self._momentum_read(frame)
            why = momentum_words(read) if read is not None and read["row"].take else None
        elif config.negative_momentum_bars:
            why = self._momentum_negative(frame, since)
        elif config.momentum_fade_bars:
            why = self._fade_turned(frame, since)
        else:
            why = self._fade_dropped(frame, since)
        if why is None:
            return None

        left = plan["sell_level"] - entry_price
        pnl_pct = (price / entry_price - 1) * 100
        in_loss = price <= entry_price
        fade = (
            f"{'Zone take' if config.take_trigger == TAKE_ZONE else 'Momentum take'}: "
            f"{why}, with the price at ${price:,.2f} — "
            + (
                f"at or under the ${entry_price:,.2f} fill"
                if in_loss
                else f"above the ${entry_price:,.2f} fill but short of the "
                f"${plan['sell_level']:,.2f} sell level"
            )
        )
        to_target = f"${left:,.2f} ({left / unit:.2f} × {config.unit_phrase})"

        if left < config.hold_min_gain_k * unit:
            return position, (
                f"{fade}. The sell level is only {to_target} above the fill, under the "
                f"{config.hold_min_gain_k:g} × {config.unit_phrase} worth keeping a runner for, so the whole "
                f"position is sold at market ({pnl_pct:+.2f}%)."
            ), self.EXIT_TAKE

        quantity = min(position, max(1.0, whole_shares(position * config.take_fraction)))
        if quantity >= position:
            return position, (
                f"{fade}. The sell level is still {to_target} above the fill, but "
                f"{config.take_fraction:.0%} of {position:g} shares leaves no whole share to "
                f"keep, so the whole position is sold at market ({pnl_pct:+.2f}%)."
            ), self.EXIT_TAKE
        stop = self._stop_price()
        return quantity, (
            f"{fade}. {'Selling' if in_loss else 'Banking'} {quantity:g} of {position:g} "
            f"shares at market ({pnl_pct:+.2f}%). The sell level is still {to_target} above "
            f"the fill — at least the {config.hold_min_gain_k:g} × {config.unit_phrase} worth "
            f"waiting for — so the other {position - quantity:g} ride on to it or the "
            "closing flatten, and "
            + (
                (
                    f"are sold at the ${stop:,.2f} stop."
                    if stop is not None
                    else "have no stop under them."
                )
                if in_loss
                else "are sold if the price comes back to the fill."
            )
        ), self.EXIT_TAKE

    def _zone_formed(self, frame, since) -> "str | None":
        """Whether a stall zone has formed under the position -- and if so, how
        to say so.

        The zone is the one the chart's Stall-zone levels pattern draws grey at
        the same settings (`candle_patterns.forming_zone`): price still inside
        a run of at least `zone_min_bars` bars whose closes all sit within
        `zone_width` typical minute moves of each other. Only a zone that
        reached that length after `since`, the last fill, counts: a position
        bought into a stretch that was already a zone was bought knowing the
        price was going nowhere, and selling it at the first bar the wait
        allows would turn every flat-tape entry into a round trip for nothing.
        Recomputed over the session on each call, like the momentum reads, so
        a backfilled bar is read as it now stands.
        """
        config = self.config
        zone = candle_patterns.forming_zone(
            frame["close"], self.minute_move, config.zone_min_bars, config.zone_width
        )
        if zone is None:
            return None
        formed = frame.index[zone["confirmed"]]
        if formed <= pd.Timestamp(since):
            return None
        return (
            f"the price has stalled — the last {zone['bars']} bars all closed between "
            f"${zone['low']:,.2f} and ${zone['high']:,.2f}, inside a "
            f"${zone['width']:,.2f} band ({config.zone_width:g} × the "
            f"${zone['unit']:,.3f} typical minute move), a zone since {formed:%H:%M}"
        )

    def _momentum_negative(self, frame, since) -> "str | None":
        """Whether the `negative_momentum_bars`-bar momentum has been negative
        on each of the last `negative_for_bars` bars after `since` -- and if
        so, how to say so.

        Momentum is `close - close[N bars ago]` in dollars, and the take fires
        once that has sat under zero for the streak. The first N bars of the
        session have nothing to compare against and break a streak rather than
        extend it. Recomputed over the session on each call,
        so a cycle that missed a bar still counts it.
        """
        n, needed = self.config.negative_momentum_bars, self.config.negative_for_bars
        closes = frame["close"]
        mom = closes - closes.shift(n)
        after = mom[frame.index > since].to_numpy()
        streak = 0
        for value in after[::-1]:
            if not value < 0:  # NaN (not known yet) or at/above zero
                break
            streak += 1
        if streak < needed:
            return None
        return (
            f"the {n}-bar momentum has been negative for the last {streak} bars since "
            f"the entry (now ${float(mom.iloc[-1]):+,.2f} against {n} bars ago; "
            f"{needed} in a row is the trigger)"
        )

    def _fade_turned(self, frame, since) -> "str | None":
        """The take a record from 2026-09-21 to -23 replays: whether the
        `momentum_fade_bars`-bar momentum has turned from positive to balanced
        or negative since `since` -- and if so, how to say so.

        "Total momentum over the last N bars" is the N-bar log return in units
        of its own random-walk scale, smoothed (`compute_momentum` at `horizon
        = N`), and positive / balanced / negative are the Schmitt-trigger regime
        over it (`assign_regimes`) -- the same definitions the chart's momentum
        panel drew with then, at a look-back of the user's choosing. The hysteresis
        is what keeps a score hovering at the line from counting as a turn: it
        becomes positive above 0.9σ and stops being positive under 0.4σ.
        """
        n = self.config.momentum_fade_bars
        scored = momentum_regime.add_momentum_regimes(frame, {"horizon": n})
        now = scored.iloc[-1]
        after = scored[scored.index >= since]
        if not len(after) or math.isnan(float(now["mom"])):
            return None
        if now["regime"] == 1 or not (after["regime"] == 1).any():
            return None
        peak = float(after["mom"].max())
        return (
            f"the {n}-bar momentum was positive since the entry (up to {peak:+.2f}σ) and "
            f"has turned {momentum_regime.regime_name(int(now['regime']))} at "
            f"{float(now['mom']):+.2f}σ"
        )

    def _fade_dropped(self, frame, since) -> "str | None":
        """The legacy take a stored record replays: the smoothed 15-bar score
        fallen `momentum_drop` sigmas from its best since `since`."""
        drop = self.config.momentum_drop
        mom = momentum_regime.compute_momentum(frame)["mom"]
        now = float(mom.iloc[-1])
        after = mom[frame.index >= since].dropna()
        if math.isnan(now) or not len(after):
            return None
        peak = float(after.max())
        if peak - now < drop:
            return None
        return (
            f"the momentum score has fallen from {peak:+.2f}σ, its best since the entry, "
            f"to {now:+.2f}σ ({drop:g}σ is the trigger)"
        )

    def _risk(self) -> float:
        """How far under the fill this position's stop sits, in dollars.

        Frozen at the fill (`_buy`) rather than recomputed each bar, because
        under `level_unit = "pred_range"` the unit is a function of a forecast
        that `_update_range` moves: re-reading it would *widen* the stop under
        an open position every time the day breached its predicted high, which
        is the one direction a stop must never move on its own. Under "adr" the
        two readings are identical, the unit being fixed for the session.

        Falls back to a fresh reading for a position adopted before there was a
        plan, and for an entry recorded before this was stored -- both are the
        old behaviour, which is correct under the unit those runs used.
        """
        entry = self.entry or {}
        if entry.get("risk") is not None:
            return float(entry["risk"])
        return stop_distance(self.config, stop_unit(self.config, self.plan or {}))

    # --- orders ------------------------------------------------------------

    def _buy(self, state: AppState, tracker: DecisionTracker, bar) -> bool:
        bought = self.buy(
            state, tracker, float(bar["close"]), self._entry_reasoning(bar),
            limit_price=self._buy_limit(),
        )
        if not bought:
            self._note_limit_miss(state, "buy level")
        else:
            # Where the momentum take starts counting for this position.
            self.entry["ts"] = bar.name
            # And how far under it the stop sits, in dollars, decided once here.
            # See `_risk`.
            self.entry["risk"] = stop_distance(
                self.config, stop_unit(self.config, self.plan)
            )
            # The first rung of the ladder `_add` climbs down.
            self.entry["fills"] = 1
            self.entry["last_fill"] = float(self.entry["price"])
            self._after_fill(state, tracker, bar)
        return bought

    def _buy_limit(self) -> "float | None":
        """The most a share may cost on this buy: the level that triggered it,
        when buys are limit orders (`limit_entry`)."""
        if not self.config.limit_entry:
            return None
        return float(self.plan["buy_level"])

    def _limit_missed(self) -> bool:
        """Whether the last buy went unfilled for its limit alone."""
        return bool(self.last_buy is not None and self.last_buy.limit_missed)

    def _note_limit_miss(self, state: AppState, what: str) -> None:
        """Say that a missed limit leaves the level armed, since the rejected
        decision alone reads as the end of it."""
        if not self._limit_missed():
            return
        _log(
            state,
            {
                "type": "status",
                "text": (
                    f"Not bought: the price had moved back above the "
                    f"${self.last_buy.limit_price:,.2f} {what} by the time the order "
                    "went out. The level stays armed; the next bar that reaches it "
                    "tries again."
                ),
            },
        )

    def _entry_reasoning(self, bar) -> str:
        plan, config = self.plan, self.config
        exits = [f"a resting sell at ${plan['sell_level']:,.2f}"]
        risk = stop_distance(config, stop_unit(config, plan))
        if risk:
            exits.append(
                f"a stop {stop_phrase(config)} (${risk:,.2f}) under the fill"
            )
        if config.has_take:
            exits.append(
                f"a take if the price stalls short of it ({fade_phrase(config)})"
                if config.take_trigger == TAKE_ZONE
                else f"a momentum take if the move gives out short of it ({fade_phrase(config)})"
            )
        return (
            f"The bar traded down to ${float(bar['low']):,.2f}, at or through the "
            f"${plan['buy_level']:,.2f} buy level — {config.buy_k:g} × the "
            f"{config.unit_phrase} (${plan['level_unit']:,.2f}) below the predicted high at "
            f"${plan['reference']:,.2f}. Buying the dip below where "
            f"the day is expected to top out; the exit is {', '.join(exits)}, or the "
            "closing bell."
        )

    def _sell(
        self,
        state: AppState,
        tracker: DecisionTracker,
        quantity: float,
        bar,
        reasoning: str,
        kind: str,
    ) -> None:
        entry = self.entry or {}
        decision = self.sell(state, tracker, quantity, reasoning, log_extra={"exit": kind})
        if decision.status != "filled":
            return
        if kind == self.EXIT_STOP:
            self._stand_down(
                state,
                self.STOPPED_OUT,
                "The day did not go the way the forecast said, and the buy level is by now "
                "usually just above the price — re-arming it would buy the same slide "
                "again, one stop lower each time.",
            )
            self.halt = self.STOPPED_OUT

        # What this position has realised so far, carried across a partial exit
        # so that a trade taken off in two pieces is judged on the whole of it
        # rather than on whichever piece happened to close it.
        banked = float(entry.get("banked") or 0.0)
        shares = float(entry.get("banked_shares") or 0.0)
        filled = float(decision.filled_quantity or 0.0)
        fill = float(decision.price or 0.0)
        if filled and entry.get("price"):
            banked += (fill - float(entry["price"])) * filled
            shares += filled

        if self.entry is None and tracker.position_for(self.ticker) > 0:
            # `sell` forgets the entry on any fill, but what is left is still this
            # position -- same fill, same peak. After a momentum take it is the
            # runner; after any other exit that did not fill in full it is
            # whatever it was.
            self.entry = {
                **entry,
                "runner": entry.get("runner") or kind == self.EXIT_TAKE,
                # A take at or under the fill (`take_in_loss`): the runner is
                # under the fill already, so no breakeven -- the stop is its
                # floor. The close the take judged, not the fill, so the rule
                # and the message it logged agree.
                "loss_runner": entry.get("loss_runner") or (
                    kind == self.EXIT_TAKE
                    and float(bar["close"]) <= float(entry.get("price") or 0.0)
                ),
                "banked": banked,
                "banked_shares": shares,
            }
        elif shares:
            # Flat: the trade is over and can be judged.
            self._close_out(state, banked, shares)

    def _stand_down(self, state: AppState, headline: str, why: str) -> None:
        """Take no further entry this session, and say once why.

        One flag for both rules that can end a session's trading, because from
        the entry's point of view they are the same instruction and a second
        reason arriving later must not re-announce it.
        """
        if self.plan.get("stand_down"):
            return
        self.plan["stand_down"] = headline
        _log(
            state,
            {
                "type": "analysis",
                "text": (
                    f"{self.ticker}: {headline} — no further entries today. {why}"
                ),
            },
        )

    def _close_out(self, state: AppState, banked: float, shares: float) -> None:
        """Judge a finished trade, and stand the session down if it barely paid.

        `min_win_k` is the bar, in ADRs **per share** -- the same unit the levels
        and the stop are written in, so it can be read against them. A round trip
        that closed for no more than that is taken as evidence the setup was not
        there today: the forecast said the day would be wide enough for the dip
        to be worth buying, and the trade that came out of it says otherwise.

        Measured over the whole position, including a momentum take's piece and
        the runner it left, so a trade exited twice is judged once and on all of
        it. Worth reading against `buy_k - sell_k`, which is the most a target
        exit can net: where `min_win_k` is the larger of the two, *every*
        completed trade stands the session down, which is a one-trade-a-day rule
        rather than a circuit breaker. The form says so when they disagree.
        """
        config, plan = self.config, self.plan
        if not config.min_win_k or shares <= 0 or plan.get("stand_down"):
            return
        unit = level_unit(config, plan)
        per_share = banked / shares
        if per_share > config.min_win_k * unit:
            return
        self._stand_down(
            state,
            f"closed for {per_share / unit:+.2f} × {config.unit_phrase}",
            f"The round trip netted ${per_share:,.2f} a share ({per_share / unit:+.2f} × "
            f"{config.unit_phrase} over {shares:g} share(s)), at or under the "
            f"{config.min_win_k:g} × {config.unit_phrase} "
            f"(${config.min_win_k * unit:,.2f}) this configuration treats as worth "
            "continuing for. A day whose first trade barely paid is not a day to keep "
            "buying the same levels on.",
        )

    # --- logging -----------------------------------------------------------

    def _plan_summary(self) -> str:
        plan = self.plan
        held = (
            "Both are held all day."
            if self.config.breach_update == BREACH_OFF
            else (
                f"If the session trades outside that range it moves {_breach_side(self.config)} "
                f"({BREACH_LABELS[self.config.breach_update].lower()}) and both levels "
                "follow it."
            )
        )
        rests = "The levels rest under the predicted high, which is flat all session."
        # A forecast of the range after the opening window (HighLow2's rest
        # head) is not the session's, and says so; it also carries the whole
        # day's pair, which nothing here trades on.
        span = "the session"
        if plan.get("range_after_opening"):
            span = f"the rest of the session after {plan['opening_end']:%H:%M}"
        day = (
            f"; the whole day's: high ${float(plan['day_high']):,.2f}, low "
            f"${float(plan['day_low']):,.2f}"
            if plan.get("day_high") is not None and plan.get("day_low") is not None else ""
        )
        return (
            f"{self.ticker} forecast for {span}, from the first "
            f"{plan['opening_end']:%H:%M} minutes: high ${plan['pred_high']:,.2f}, low "
            f"${plan['pred_low']:,.2f}{day} (yesterday's average ${plan['prev_avg']:,.2f}, "
            f"14-day average range ${plan['adr14_abs']:,.2f}). {rests} Buy at "
            f"${plan['buy_level']:,.2f} (ref − {self.config.buy_k:g} × {self.config.unit_phrase}), sell at "
            f"${plan['sell_level']:,.2f} (ref − {self.config.sell_k:g} × {self.config.unit_phrase}). {held}"
        )

    def _read_summary(self, bar, ts, position: float, frame=None) -> str:
        price = float(bar["close"])
        plan = self.plan
        parts = [
            f"{self.ticker} {ts:%H:%M} ${price:,.2f}",
            f"buy ${plan['buy_level']:,.2f} ({price - plan['buy_level']:+.2f})",
            f"sell ${plan['sell_level']:,.2f} ({price - plan['sell_level']:+.2f})",
        ]
        updates = int(plan.get("range_updates") or 0)
        if updates:
            # So a level read off this line is never mistaken for the 9:35 one.
            parts.append(
                f"H ${plan['pred_high']:,.2f} / L ${plan['pred_low']:,.2f}, "
                f"updated ×{updates}"
            )
        if self.config.momentum_confirmation_bars and frame is not None:
            read = self._momentum_read(frame)
            if read is not None:
                parts.append(
                    f"mom {read['mom']:+.3f} ({read['mom_class']}), "
                    f"Δ {read['change']:+.3f} ({read['change_class']})"
                )
        if position > 0 and self.entry:
            entry_price = self.entry["price"]
            pnl = (price / entry_price - 1) * 100 if entry_price else 0.0
            fills = int(self.entry.get("fills") or 0)
            average = f" avg of {fills} buys" if fills > 1 else ""
            parts.append(
                f"long {position:g} sh @ ${entry_price:,.2f}{average} ({pnl:+.2f}%), "
                f"{self.entry['bars']} bars"
            )
            if (
                self.config.take_trigger == TAKE_ZONE
                and frame is not None
                and not self.entry.get("runner")
            ):
                zone = candle_patterns.forming_zone(
                    frame["close"], self.minute_move,
                    self.config.zone_min_bars, self.config.zone_width,
                )
                if zone is not None:
                    parts.append(
                        f"zone {zone['bars']} bars ${zone['low']:,.2f}–${zone['high']:,.2f}"
                    )
            if self.entry.get("runner") and not self.entry.get("loss_runner"):
                parts.append(f"runner, out at ${entry_price:,.2f}")
            else:
                if self.entry.get("runner"):
                    parts.append("runner")
                if self._stop_price() is not None:
                    parts.append(f"stop ${self._stop_price():,.2f}")
        elif plan.get("stand_down"):
            parts.append(f"{plan['stand_down']}, no new entries today")
        return " · ".join(parts)


def _record_strategy(record: dict) -> str:
    """Which rule set wrote a published record (`TraderSlot`).

    A run continues an earlier run's record of the same day only when the same
    rules wrote it: a day-range record's open position has no momentum clock,
    and a Fast&Furious one has no levels, so either taken up by the other
    would be read wrong -- a position the other rules then hold is adopted at
    the bar instead, as one left by a run that kept no record. A record whose
    configuration did not survive a restore is a day-range one, as every
    record was before there was another rule set."""
    config = record.get("config")
    if config is None:
        return apple_models.STRATEGY_DAYRANGE
    return apple_models.strategy(getattr(config, "model_key", None))


def _levels_row(config: AppleTraderConfig, plan: dict, ts) -> dict:
    """One bar's levels off a plan: what `session_levels` walks and a live run
    records (`DayRangeTrader._record_levels`), in the one shape the chart reads.

    `stop` is None: a stop hangs under the actual fill, so before a buy there
    is none to draw. `_record_levels` fills it in while a position is open.
    `model_key` is the model the levels hang off, which a run can switch
    mid-session (`_adopt_form_model`)."""
    return {
        "t": ts,
        "model_key": config.model_key,
        "buy": float(plan["buy_level"]),
        "sell": float(plan["sell_level"]),
        "stop": None,
        "reference": float(plan["reference"]),
        "pred_high": float(plan["pred_high"]),
        "pred_low": float(plan["pred_low"]),
    }


def session_levels(
    config: AppleTraderConfig,
    forecast: dict,
    session,
    opening_end,
    open_price: "float | None" = None,
) -> "list[dict]":
    """The buy and sell levels this configuration would rest, bar by bar.

    For drawing, not for trading -- and for a session no live run recorded:
    while one is running the chart draws its own record instead
    (`DayRangeTrader._record_levels`), which knows about sidebar edits and
    restarts that a walk under one configuration cannot. `model_overlays` puts the two levels beside
    the candles that tested them, and the only honest way to do that is to ask
    the agent -- so this walks a real `DayRangeTrader` through the session and
    reads its plan, rather than re-deriving `reference - k x unit` somewhere the
    two could drift apart. Every setting that moves a level is therefore
    accounted for by construction: the breach update ratchets the forecast, and
    a change to it shows up in the picture the same day it shows up in the
    trades.

    `forecast` is `dayrange_model.forecast_session`'s dict, `session` the day's
    minute bars and `opening_end` the last bar of the window the forecast was
    built on -- bars at or before it are skipped, exactly as the loop skips
    trading them. Returns one
    `{"t", "model_key", "buy", "sell", "stop", "reference", "pred_high", "pred_low"}` for
    `opening_end` itself -- the levels the forecast alone rests -- and then one
    per bar after that, in order, each holding the levels once that bar has
    been read. Empty while no bar has closed after the forecast.

    `stop` is always None here: the stop hangs under the actual fill, and a
    walk places no orders, so there is no fill for it to hang under.

    The forecast is in each row as well as the levels because the two must be
    drawn from the same walk: the levels hang off the predicted high, so a
    chart that took the levels from here and the forecast from the 9:35 dict
    would draw a breached session with the buy and sell stepping up and the
    line they are measured under standing still.

    No orders, no log and no ledger: nothing here touches `run_cycle`. The
    circuit breaker and the managed exit are deliberately not modelled -- they
    are about a position, and this is about where the levels sat.
    """
    trader = DayRangeTrader(config)
    trader.plan = {
        "date": pd.Timestamp(opening_end).normalize(),
        "opening_end": opening_end,
        "open_price": float(
            open_price if open_price else session["open"].iloc[0]
        ),
        **forecast,
    }
    def row(ts) -> dict:
        return _levels_row(config, trader.plan, ts)

    after = [ts for ts in session.index if ts > opening_end]
    if not after:
        return []
    # The levels as the forecast alone sets them, before any bar has had a
    # chance to move them. Without this row a breach on the first bar after
    # the forecast is already in every row, the walk looks flat, and the chart
    # draws the revised levels as if they had stood since 09:35.
    trader._set_levels(opening_end)
    out: "list[dict]" = [row(opening_end)]
    for ts in after:
        trader._move_range(session[session.index <= ts], ts)
        trader._set_levels(ts)
        out.append(row(ts))
    return out


# --- Fast&Furious's rules (`FastFuriousTrader`) -------------------------------


def ff_credentials(state: AppState) -> "tuple[str | None, str | None]":
    """The sidebar's Alpaca keys live; the environment's in a replay, whose
    state carries placeholders (every dataset read is patched, Fast&Furious's
    history is not)."""
    if getattr(state, "bar_tape_override", ""):
        return None, None
    return state.api_key or None, state.api_secret or None


def ff_signal(config: AppleTraderConfig, forecast: "dict | None") -> int:
    """A bar's signal as the rules read it: +1 buy, -1 sell (when a sell signal
    closes a long, `ff_exit_on_sell_signal`), 0 nothing -- or no forecast."""
    if forecast is None:
        return 0
    signal = int(forecast["signal"])
    if signal < 0 and not config.ff_exit_on_sell_signal:
        return 0
    return signal


def ff_stop(config: AppleTraderConfig, forecast: "dict | None", fill: float) -> "float | None":
    """The stop under a fill: `ff_stop_k` x the forecast 2-hour drop, in the
    forecast's own log units. None without a stop, or without the range
    forecast to size one."""
    if not config.ff_stop_k or forecast is None:
        return None
    down, s120 = forecast.get("down_hat"), forecast.get("s120")
    if down is None or s120 is None or not (math.isfinite(down) and math.isfinite(s120)):
        return None
    return float(fill * math.exp(-config.ff_stop_k * float(down) * float(s120)))


def ff_threshold_words(
    config: AppleTraderConfig, theta: float, sessions: int, forecast: "dict | None"
) -> str:
    """The session's entry threshold in a log line's words."""
    bps = (forecast or {}).get("theta_bps")
    scale = f", {bps:.3f} bps a minute at this minute's scale" if bps is not None else ""
    return (
        f"entry threshold (y ±{theta:.4f}, the {config.ff_entry_quantile:.0%} quantile of "
        f"the last {sessions} sessions' forecasts{scale})"
    )


class FastFuriousTrader(_SitsOut, BaseTrader):
    """Fast&Furious's rules: a forecast of the next 30 minutes at every bar.

    The model answers one question at the close of every bar from 09:31 to
    15:30 -- what will the average 1-minute momentum be over the next 30
    minutes? -- and notebook 06's rules act on the answer directly:

    * **enter** when the forecast `y_hat` is at or above the threshold: the
      `ff_entry_quantile` (0.95) of |y_hat| over every minute of the previous
      20 sessions. A threshold rather than a number because the model's output
      is shrunk toward zero and only its rank carries information, so "the top
      5% of what it has been saying" is the claim;
    * **stop**: `ff_stop_k` (1.5) x the forecast 2-hour drop from the signal
      bar's price (`down_hat` x `s120`), hung under the fill. After a stop the
      next buy waits for the forecast to leave the buy zone first, so a stop is
      not undone by buying the next minute;
    * **out** `ff_hold_minutes` (30) after the last buy signal -- a fresh one
      restarts the clock -- at once on a sell signal (a forecast at or under
      minus the threshold, `ff_exit_on_sell_signal`), at the flatten at the
      latest.

    In that order on every bar, as the notebook's `trading._session` does: the
    stop against the bar, then the bar's own signal.

    Against the notebook
    --------------------
    * **Long only.** The notebook's chosen rules trade both sides; this ledger
      holds longs, so a sell signal closes rather than reverses. Its long
      trades are these trades exactly, on the same forecasts and fills.
    * **The threshold** reads the saved model's own forecasts of the previous
      20 sessions. The notebook's test month read the walk-forward models'
      for the sessions before it; for every session from 2 October 2026 on the
      two are the same forecasts.
    * **Fills.** The notebook fills an entry or a time exit at the next bar's
      open and a stop at the stop (or a worse open); here every order is a
      market order sent once the bar has closed, so a stop fills wherever the
      price is after the bar that touched it -- usually worse.
    * **The flatten** closes `flatten_before_close_min` before the bell rather
      than at the 15:59 close.
    * **No stand-down.** A stop does not stop the run or the session, unlike
      the day-range rules': at around one trade a day and a stop on a quarter
      of them, it is part of the system rather than a verdict on the day.
    """

    ENTRY_TRIGGER_TEXT = "Momentum forecast over the entry threshold"

    #: Which rule closed a position -- carried on the log entry.
    EXIT_STOP = "stop"
    EXIT_TIME = "time"
    EXIT_SELL_SIGNAL = "sell_signal"
    EXIT_FLATTEN = "flatten"

    def __init__(self, config: "AppleTraderConfig | None" = None) -> None:
        super().__init__(config or AppleTraderConfig(model_key=apple_models.FASTFURIOUS_KEY))
        # The session's threshold, whether the buy side is blocked after a
        # stop, and the chart's record of what was read each bar; None before
        # the first forecast.
        self.session: "dict | None" = None
        # The newest bar's forecast (`fastfurious_model.forecast_at`), or None.
        self.forecast: "dict | None" = None
        self._sit_out_noted: "dict" = {}
        # Never set: a stop is not a reason to stop (see the docstring). Kept
        # because the live loop reads it off every trader.
        self.halt: "str | None" = None
        self.rearm_after_stop = True
        self.slot: TraderSlot = SINGLE_SLOT
        self.quiet_read = False
        self.last_close: "float | None" = None
        # The last forecast failure reported, so a feed that is down is said
        # once rather than every minute.
        self._forecast_note: "str | None" = None
        self._prewarmed = None

    def activity(self, outcome: str, tracker: DecisionTracker) -> "tuple[str, str]":
        if outcome != rule_agent.CLOSED and (self.blocked or {}).get("sit_out"):
            labels = ", ".join(event_days.LABELS[c] for c in self.blocked["sit_out"])
            return rule_agent.WAITING, f"Agent sits today out ({labels})"
        if outcome == rule_agent.WARMING_UP and self.session is None and self.blocked is None:
            return rule_agent.WAITING, "Agent waiting for the first forecast"
        return super().activity(outcome, tracker)

    # --- one cycle --------------------------------------------------------

    def run_cycle(self, bundle: dict, state: AppState, tracker: DecisionTracker) -> str:
        """Read the newest closed bar's forecast and act on it."""
        self._prewarm(bundle, state)
        sym_state, refused = self.preflight(state)
        if refused is not None:
            return refused
        today = pd.Timestamp(clock.now().astimezone(market_hours.MARKET_TZ).date())
        self._roll_session(today)

        frame = momentum_regime.minute_frame(sym_state)
        if not len(frame):
            _log(state, {"type": "status", "text": f"No {self.ticker} bars yet today."})
            return rule_agent.NO_DATA
        if self.blocked is not None:
            return rule_agent.NO_DATA
        if self.session is None:
            if self._sit_out(state, frame, today, 1):
                return rule_agent.NO_DATA if self.blocked is not None else rule_agent.WARMING_UP
            if not self._plan_session(bundle, state, today):
                return rule_agent.NO_DATA

        last = frame.iloc[-1]
        ts = frame.index[-1]
        self.last_close = float(last["close"])
        minute = int(round(float(last["minutes_from_open"])))
        position = tracker.position_for(self.ticker)
        if position > 0 and self.entry is None:
            # A position without a remembered entry (a restart onto an existing
            # ledger): adopted at this bar, with no stop -- the forecast it was
            # bought on is not known -- and a fresh clock.
            self.entry = {"price": float(last["close"]), "bars": 0, "ts": ts,
                          "due": minute + self.config.ff_hold_minutes, "stop": None}
        if position <= 0:
            self.entry = None
        if ts == self.last_bar_ts:
            return rule_agent.HOLD
        self.last_bar_ts = ts
        if self.entry is not None:
            self.entry["bars"] = int(self.entry.get("bars") or 0) + 1

        self.forecast = self._forecast_for(bundle, state, today, minute)
        self._record(state, ts, last)
        if not self.quiet_read:
            _log(state, {"type": "analysis", "text": self._read_summary(last, ts, position)})

        want = self._want()
        # 1. The stop, against this bar: it rested from the fill on.
        stop = (self.entry or {}).get("stop")
        if position > 0 and stop is not None and float(last["low"]) <= float(stop):
            self._sell(state, tracker, position, self.EXIT_STOP, (
                f"The {ts:%H:%M} bar traded down to ${float(last['low']):,.2f}, through the "
                f"${float(stop):,.2f} stop ({self.config.ff_stop_k:g} × the forecast 2-hour drop "
                "under the fill). The next buy waits for the forecast to leave the buy zone."
            ))
            self.session["blocked"] = True
            self._unblock(want)
            return rule_agent.SOLD
        # 2. The flatten: nothing is held into the close, and nothing bought.
        if self.closing_soon():
            if position > 0:
                self._sell(state, tracker, position, self.EXIT_FLATTEN, (
                    f"Inside the last {self.config.flatten_before_close_min} min of the "
                    "session: flattening before the close."
                ))
                return rule_agent.SOLD
            return rule_agent.HOLD
        # 3. The bar's own signal (`trading._session`'s close-of-bar step).
        want = self._unblock(want)
        if position > 0:
            if want == 1:
                self.entry["due"] = minute + self.config.ff_hold_minutes
                return rule_agent.HOLD
            if want == -1:
                self._sell(state, tracker, position, self.EXIT_SELL_SIGNAL, (
                    f"The forecast turned: {self.forecast['mom_bps']:+.3f} bps a minute over the "
                    f"next 30, at or under minus the {self._threshold_words()} — a sell signal "
                    "closes the long."
                ))
                return rule_agent.SOLD
            if minute >= int(self.entry.get("due") or 0):
                self._sell(state, tracker, position, self.EXIT_TIME, (
                    f"{self.config.ff_hold_minutes} minutes since the last buy signal and no "
                    "fresh one: the forecast the position was bought on has run its course."
                ))
                return rule_agent.SOLD
            return rule_agent.HOLD
        if want != 1:
            return rule_agent.HOLD
        if self.entry_refused(state, ts):
            return rule_agent.HOLD
        return rule_agent.BOUGHT if self._buy(state, tracker, last, ts, minute) else rule_agent.HOLD

    def _want(self) -> int:
        """The bar's signal as the rules read it (`ff_signal`)."""
        return ff_signal(self.config, self.forecast)

    def _unblock(self, want: int) -> int:
        """The notebook's re-arm after a stop: the buy side stays blocked while
        the forecast is still in the buy zone, and clears as soon as it is not.
        Returns the signal the rules may act on."""
        if not self.session.get("blocked"):
            return want
        if want != 1:
            self.session["blocked"] = False
            return want
        return 0

    # --- the session ------------------------------------------------------

    def _roll_session(self, today) -> None:
        if self.session is not None and self.session["date"] != today:
            self.session = None
            self.entry = None
            self.last_bar_ts = None
            self.forecast = None
            self._forecast_note = None
        if self.blocked is not None and self.blocked["date"] != today:
            self.blocked = None

    def _credentials(self, state: AppState) -> "tuple[str | None, str | None]":
        return ff_credentials(state)

    def _prewarm(self, bundle: dict, state: AppState) -> None:
        """Fill the first forecast's caches in the background, once a session:
        ~200 days of SIP and IEX minute bars, the threshold's 20 sessions of
        forecasts and last night's option table -- tens of seconds cold, which
        would otherwise hold the 09:31 forecast. Live only; a failure is not
        reported, the first forecast makes the same reads and reports its own."""
        if getattr(state, "bar_tape_override", ""):
            return
        today = pd.Timestamp(clock.now().astimezone(market_hours.MARKET_TZ).date())
        if self._prewarmed == today or not len(_fastfurious().nyse_sessions([today])):
            return
        self._prewarmed = today
        module, ticker = _fastfurious(), self.ticker
        key, secret = self._credentials(state)

        def work() -> None:
            try:
                module.warm(bundle, ticker, today, key, secret)
            except Exception:  # noqa: BLE001 -- the first forecast retries and reports
                pass

        threading.Thread(target=clock.inherit(work), name=f"fastfurious-warm-{ticker}", daemon=True).start()

    def _plan_session(self, bundle: dict, state: AppState, today) -> bool:
        """The session's entry threshold, once. False -- and the session
        blocked, reported once -- when there is none to be had."""
        module = _fastfurious()
        key, secret = self._credentials(state)
        try:
            if module.is_half_day(today):
                raise ValueError(
                    "today closes at 13:00, and the model was fitted on full sessions only"
                )
            theta, sessions = module.threshold(
                bundle, self.ticker, today, self.config.ff_entry_quantile, key, secret
            )
            if not math.isfinite(theta):
                raise ValueError(
                    f"only {sessions} earlier sessions have forecasts; the entry threshold "
                    f"needs at least {module.THRESHOLD_MIN_SESSIONS}"
                )
        except Exception as exc:  # noqa: BLE001 -- fatal for the session, not the run
            self.blocked = {"date": today, "reason": str(exc)}
            _log(state, {"type": "error", "text": (
                f"Apple Trader cannot read Fast&Furious's forecasts for {self.ticker} today, so "
                f"it will not trade this session: {exc}"
            )})
            return False
        self.session = {"date": today, "theta": float(theta), "sessions": int(sessions),
                        "blocked": False, "history": []}
        prior = self.slot.levels(state) or {}
        if (
            prior.get("rows")
            and prior.get("ticker") == self.ticker
            and pd.Timestamp(prior.get("date")) == pd.Timestamp(today)
            and _record_strategy(prior) == self.config.strategy
        ):
            # A restart mid-session: the earlier run's record stays on the
            # chart, and its open position's stop and clock are taken back.
            self.session["history"] = list(prior["rows"])
            self._resume(state, prior.get("memory"))
        _log(state, {"type": "analysis", "text": (
            f"{self.ticker} entry threshold for today: a forecast y of ±{theta:.4f} — the "
            f"{self.config.ff_entry_quantile:.0%} quantile of |y| over every minute of the last "
            f"{sessions} sessions' forecasts (y is the next 30 minutes' log return in units of "
            "its expected size at that minute). Buy at or above it"
            + (", sell a long at or under minus it" if self.config.ff_exit_on_sell_signal else "")
            + "."
        )})
        return True

    def _forecast_for(self, bundle: dict, state: AppState, today, minute: int) -> "dict | None":
        """This bar's forecast, or None (after 15:30, or when it cannot be
        made -- said once per distinct reason)."""
        if minute > _fastfurious().LAST_ORIGIN:
            return None
        key, secret = self._credentials(state)
        try:
            found = _fastfurious().forecast_at(
                bundle, self.ticker, today, minute, self.config.ff_entry_quantile, key, secret
            )
        except Exception as exc:  # noqa: BLE001 -- one bar without a forecast is not the day
            note = str(exc)
            if note != self._forecast_note:
                self._forecast_note = note
                _log(state, {"type": "error", "text": (
                    f"No Fast&Furious forecast for {self.ticker} at bar {minute}: {note}"
                )})
            return None
        self._forecast_note = None
        return found

    # --- orders -----------------------------------------------------------

    def _stop_for(self, fill: float) -> "float | None":
        """The stop under a fill (`ff_stop`)."""
        return ff_stop(self.config, self.forecast, fill)

    def _buy(self, state: AppState, tracker: DecisionTracker, bar, ts, minute: int) -> bool:
        fc = self.forecast
        if self.config.ff_stop_k and not math.isfinite(fc.get("down_hat", float("nan"))):
            note = fc.get("range_problem") or "the 2-hour range forecast is missing"
            _log(state, {"type": "status", "text": (
                f"{self.ticker}: a buy signal at {ts:%H:%M}, but there is no 2-hour range "
                f"forecast to size the stop with ({note}), so it is not taken."
            )})
            return False
        drop = 1 - math.exp(-self.config.ff_stop_k * fc["down_hat"] * fc["s120"]) if self.config.ff_stop_k else 0.0
        exits = [f"out at {(ts + pd.Timedelta(minutes=self.config.ff_hold_minutes + 1)):%H:%M} unless a "
                 "fresh signal renews it"]
        if self.config.ff_stop_k:
            exits.insert(0, f"a stop {drop:.2%} under the fill ({self.config.ff_stop_k:g} × the "
                            "forecast 2-hour drop)")
        if self.config.ff_exit_on_sell_signal:
            exits.append("at once on a sell signal")
        reasoning = (
            f"The forecast of the next 30 minutes' average momentum is {fc['mom_bps']:+.3f} bps a "
            f"minute (y {fc['y_hat']:+.4f}), at or above the {self._threshold_words()}. Buying; "
            f"{', '.join(exits)}, or the closing flatten."
        )
        bought = self.buy(state, tracker, float(bar["close"]), reasoning)
        if bought:
            fill = float(self.entry["price"])
            self.entry.update({
                "ts": ts,
                "signal_minute": minute,
                "due": minute + self.config.ff_hold_minutes,
                "stop": self._stop_for(fill),
                "y_hat": float(fc["y_hat"]),
            })
            self._record(state, ts, bar)
        return bought

    def _sell(self, state: AppState, tracker: DecisionTracker, quantity: float, kind: str,
              reasoning: str) -> None:
        entry = dict(self.entry or {})
        decision = self.sell(state, tracker, quantity, reasoning, log_extra={"exit": kind})
        if decision.status == "filled" and self.entry is None and tracker.position_for(self.ticker) > 0:
            # Partly filled: what is left is the same position, same stop.
            self.entry = entry

    def _resume(self, state: AppState, memory: "dict | None") -> None:
        """Take back an earlier run's open position (its stop and its clock) and
        a blocked buy side, on a restart the same day. The entry is only a
        candidate: `run_cycle` drops it if the ledger is flat."""
        if not memory:
            return
        self.session["blocked"] = bool(memory.get("blocked"))
        entry = memory.get("entry")
        if entry and self.entry is None:
            self.entry = dict(entry)
            _log(state, {"type": "status", "text": (
                f"{self.ticker}: continuing today's earlier run's position — "
                + (f"stop ${float(entry['stop']):,.2f}, " if entry.get("stop") else "")
                + f"out at bar {entry.get('due')} unless renewed."
            )})

    # --- the record and the log -------------------------------------------

    def memory(self) -> dict:
        return {
            "entry": dict(self.entry) if self.entry is not None else None,
            "blocked": bool((self.session or {}).get("blocked")),
        }

    def publish_memory(self, state: AppState) -> None:
        levels = self.slot.levels(state)
        if (
            not levels
            or self.session is None
            or levels.get("ticker") != self.ticker
            or pd.Timestamp(levels.get("date")) != pd.Timestamp(self.session["date"])
        ):
            return
        self.slot.publish(state, {**levels, "memory": self.memory()})

    def _record(self, state: AppState, ts, bar) -> None:
        """This bar's forecast and the position's stop, added to the session's
        record and published: what the chart draws while the run is live."""
        fc = self.forecast or {}
        row = {
            "t": ts,
            "model_key": self.config.model_key,
            "price": float(bar["close"]),
            "mom_bps": fc.get("mom_bps"),
            "theta_bps": fc.get("theta_bps"),
            "signal": fc.get("signal"),
            "pred_high": fc.get("pred_high"),
            "pred_low": fc.get("pred_low"),
            "stop": (self.entry or {}).get("stop"),
        }
        history = self.session.setdefault("history", [])
        if history and history[-1]["t"] == row["t"]:
            history[-1] = row
        else:
            history.append(row)
        self.slot.publish(state, {
            "ticker": self.ticker,
            "date": self.session["date"],
            "config": self.config,
            "rows": history,
            "memory": self.memory(),
            "seed": None,
        })

    def _threshold_words(self) -> str:
        return ff_threshold_words(
            self.config, self.session["theta"], self.session["sessions"], self.forecast
        )

    def _read_summary(self, bar, ts, position: float) -> str:
        price = float(bar["close"])
        parts = [f"{self.ticker} {ts:%H:%M} ${price:,.2f}"]
        fc = self.forecast
        if fc is not None:
            share = fc["y_hat"] / self.session["theta"] if self.session["theta"] else 0.0
            parts.append(
                f"next 30 min {fc['mom_bps']:+.3f} bps/min ({share:+.0%} of the entry threshold)"
            )
            if fc["signal"]:
                parts.append("buy signal" if fc["signal"] > 0 else "sell signal")
        else:
            parts.append("no forecast")
        if position > 0 and self.entry:
            entry_price = float(self.entry["price"])
            pnl = (price / entry_price - 1) * 100 if entry_price else 0.0
            parts.append(f"long {position:g} sh @ ${entry_price:,.2f} ({pnl:+.2f}%)")
            if self.entry.get("stop"):
                parts.append(f"stop ${float(self.entry['stop']):,.2f}")
            due = self.entry.get("due")
            if due is not None:
                out = pd.Timestamp(ts).normalize() + pd.Timedelta(hours=9, minutes=30 + int(due) + 1)
                parts.append(f"out {out:%H:%M} unless renewed")
        elif (self.session or {}).get("blocked"):
            parts.append("stopped out — buys wait for the forecast to leave the buy zone")
        return " · ".join(parts)


# --- GoodMoments' rules ------------------------------------------------------


def _goodmoments():
    """`agent_stonks.goodmoments_model`, imported on first use (LightGBM, SciPy)."""
    from . import goodmoments_model

    return goodmoments_model


def gm_take_k_at(config: AppleTraderConfig, scale: "float | None") -> float:
    """The take's distance from the fill in ADRs at a minute whose volatility
    sits `scale` of the way from the session's quietest minute (0) to its most
    volatile (1) -- `intraday_vol_model.volatility_scale`: `gm_take_k` at the
    peak, `gm_take_min_k` at the trough, in proportion between. `gm_take_k`
    itself when the take does not follow the curve."""
    if config.gm_take_min_k is None or scale is None:
        return config.gm_take_k
    return config.gm_take_min_k + (config.gm_take_k - config.gm_take_min_k) * float(scale)


def gm_exits(config: AppleTraderConfig, fill: float, adr: "float | None",
             scale: "float | None" = None) -> dict:
    """The take and the stop over and under a fill, in prices: the take's
    distance at this minute (`gm_take_k_at`) and `gm_stop_k` ADRs in log terms,
    as the notebook rests them. None for one switched off, or with no ADR to
    size it."""
    usable = adr is not None and math.isfinite(adr) and adr > 0
    take_k = gm_take_k_at(config, scale)
    return {
        "take": float(fill * math.exp(take_k * adr)) if usable and config.gm_take_k else None,
        "stop": float(fill * math.exp(-config.gm_stop_k * adr)) if usable and config.gm_stop_k else None,
    }


def _session_clock(minute: int) -> str:
    """Minute `minute` of the regular session as an ET wall clock, "15:05"."""
    total = 9 * 60 + 30 + int(minute)
    return f"{total // 60:02d}:{total % 60:02d}"


class GoodMomentsTrader(_SitsOut, BaseTrader):
    """GoodMoments' rules: the chance of a +0.2 ADR rise, at every bar.

    At the close of every bar the model says how likely it is that a later bar
    of the session trades 0.2 ADR above this one's close (`p`), and the same
    trees on the clock and today's volatility alone say it too (`p_tv`).
    Notebook 4's chosen system acts on the difference -- the edge -- because a
    high probability on its own mostly means "early in a volatile session",
    where a fall of 0.2 ADR is as likely as the rise:

    * **enter** at market when the edge is at least `gm_enter` (0.02), up to the
      15:30 bar and never within 15 bars of a stop;
    * **take** `gm_take_k` (0.2) ADR over the fill -- the question's own target
      -- and **stop** `gm_stop_k` (0.75) ADR under it, both resting from the
      fill on and checked against every bar after it (a bar touching both is a
      stop). With `gm_take_min_k` set the take follows the time of day's
      volatility instead (IntradayVolatility's fitted curve): `gm_take_k` at
      the session's most volatile minute, `gm_take_min_k` at its quietest, and
      moved to the bar's own minute before each check -- still measured from
      the fill, so a position bought at the open asks less of the price as the
      session quietens;
    * **sell on the signal** when the edge falls under `gm_exit` -- off in the
      notebook's pick, which holds what neither level closes to the close;
    * **confirm on momentum** with `gm_confirm_bars` -- the day-range rules'
      behaviour table over that many bars (`MOMENTUM_TABLE`): an entry the
      edge calls for also needs the table's "buy" (momentum not negative, and
      not flat with a falling change), and a take that is reached is held
      while momentum is still positive, then sold on the first bar it is not,
      wherever the price is by then. Until the table can be read (N + 2 bars,
      last week's mean minute move) neither happens. Off in the notebook.

    In that order on every bar, as the notebook's `trading._session` does: the
    stop and the take against the bar, then the bar's own forecast. After a
    take the same bar's forecast can buy again, as in the notebook.

    Against the notebook
    --------------------
    * **The tape.** The notebook reads SIP at the bar's close; a basic plan
      has SIP 15 minutes late, so the session so far is SIP to 15 bars back and
      IEX since (`goodmoments_model`), and the forecasts differ a little.
    * **Fills.** The notebook buys or sells on a signal at the next bar's open,
      and fills a take or a stop at its level (or a worse open); here every
      order is a market order sent once the bar has closed, so a take fills
      wherever the price is after the bar that touched it -- usually worse --
      and so does a stop.
    * **The flatten** closes `flatten_before_close_min` before the bell rather
      than at the 15:59 close.
    * **No stand-down.** A stop pauses entries for 15 bars, as the notebook's
      rules do, rather than ending the run as the day-range rules' stop does.
    """

    ENTRY_TRIGGER_TEXT = "Edge at the entry threshold"

    #: Which rule closed a position -- carried on the log entry.
    EXIT_STOP = "stop"
    EXIT_TAKE = "take"
    EXIT_SIGNAL = "signal"
    EXIT_FLATTEN = "flatten"

    def __init__(self, config: "AppleTraderConfig | None" = None) -> None:
        super().__init__(config or AppleTraderConfig(model_key=apple_models.GOODMOMENTS_KEY))
        # The session's ADR, the bar entries wait for after a stop, and the
        # chart's record of what was read each bar; None before the first bar.
        self.session: "dict | None" = None
        # The newest bar's forecast (`goodmoments_model.forecast_at`), or None.
        self.forecast: "dict | None" = None
        self._sit_out_noted: "dict" = {}
        # Never set: a stop is not a reason to stop (see the docstring). Kept
        # because the live loop reads it off every trader.
        self.halt: "str | None" = None
        self.slot: TraderSlot = SINGLE_SLOT
        self.quiet_read = False
        self.last_close: "float | None" = None
        # The last forecast failure reported, so a feed that is down is said
        # once rather than every minute.
        self._forecast_note: "str | None" = None
        self._prewarmed = None
        # The ticker's `abs_mean_minute_momentum`, read off its state each
        # cycle: what the momentum confirmation's neutral band is a share of.
        self.minute_move: "float | None" = None
        # The last reason the confirmation refused an entry, so a stretch of
        # refused bars is logged once.
        self._momentum_note: "str | None" = None

    def activity(self, outcome: str, tracker: DecisionTracker) -> "tuple[str, str]":
        if outcome != rule_agent.CLOSED and (self.blocked or {}).get("sit_out"):
            labels = ", ".join(event_days.LABELS[c] for c in self.blocked["sit_out"])
            return rule_agent.WAITING, f"Agent sits today out ({labels})"
        if outcome == rule_agent.WARMING_UP and self.session is None and self.blocked is None:
            return rule_agent.WAITING, "Agent waiting for the first forecast"
        return super().activity(outcome, tracker)

    # --- one cycle --------------------------------------------------------

    def run_cycle(self, bundle: dict, state: AppState, tracker: DecisionTracker) -> str:
        """Read the newest closed bar's forecast and act on it."""
        self._prewarm(bundle, state)
        sym_state, refused = self.preflight(state)
        if refused is not None:
            return refused
        today = pd.Timestamp(clock.now().astimezone(market_hours.MARKET_TZ).date())
        self._roll_session(today)

        frame = momentum_regime.minute_frame(sym_state)
        if not len(frame):
            _log(state, {"type": "status", "text": f"No {self.ticker} bars yet today."})
            return rule_agent.NO_DATA
        self.minute_move = getattr(sym_state, "abs_mean_minute_momentum", None)
        if self.blocked is not None:
            return rule_agent.NO_DATA
        if self.session is None:
            if self._sit_out(state, frame, today, 1):
                return rule_agent.NO_DATA if self.blocked is not None else rule_agent.WARMING_UP
            if not self._plan_session(bundle, state, today):
                return rule_agent.NO_DATA

        last = frame.iloc[-1]
        ts = frame.index[-1]
        self.last_close = float(last["close"])
        minute = int(round(float(last["minutes_from_open"])))
        position = tracker.position_for(self.ticker)
        if position > 0 and self.entry is None:
            # A position without a remembered entry (a restart onto an existing
            # ledger): adopted at this bar, its take and stop measured from it.
            close = float(last["close"])
            self.entry = {"price": close, "bars": 0, "ts": ts, **self._exits(close, minute)}
        if position <= 0:
            self.entry = None
        if ts == self.last_bar_ts:
            return rule_agent.HOLD
        self.last_bar_ts = ts
        if self.entry is not None:
            self.entry["bars"] = int(self.entry.get("bars") or 0) + 1
            self._move_take(minute)

        self.forecast = self._forecast_for(bundle, state, today, minute)
        self._after_forecast(bundle, state, frame, today, ts)
        self._record(state, ts, last)
        if not self.quiet_read:
            _log(state, {"type": "analysis", "text": self._read_summary(last, ts, position, minute, frame)})

        # 1. The stop and the take, against this bar: both rested from the fill
        # on, and a bar that touched both is a stop. With the confirmation on, a
        # take reached on this bar or an earlier one sells only once momentum
        # is no longer positive.
        took = False
        if position > 0:
            stop, take = self.entry.get("stop"), self.entry.get("take")
            if stop is not None and float(last["low"]) <= float(stop):
                self._sell(state, tracker, position, self.EXIT_STOP, (
                    f"The {ts:%H:%M} bar traded down to ${float(last['low']):,.2f}, through the "
                    f"${float(stop):,.2f} stop ({self.config.gm_stop_k:g} ADR under the fill). "
                    f"No new entry for {APPLE_TRADER_GM_COOLDOWN_BARS} bars."
                ))
                self.session["cool_until"] = minute + APPLE_TRADER_GM_COOLDOWN_BARS
                return rule_agent.SOLD
            reached_before = bool(self.entry.get("take_reached"))
            reached = take is not None and (reached_before or float(last["high"]) >= float(take))
            held = self._hold_past_take(frame) if reached else None
            if held is not None and not reached_before:
                # Held, not sold: the flatten and the edge exit below still apply.
                self.entry["take_reached"] = True
                _log(state, {"type": "status", "text": (
                    f"The {ts:%H:%M} bar traded up to ${float(last['high']):,.2f}, through "
                    f"the ${float(take):,.2f} take, but {held}. Holding on; sold on the "
                    "first bar momentum is no longer positive."
                )})
            if reached and held is None:
                if reached_before:
                    # No read only when a restart brought a held take onto a
                    # run with the confirmation off.
                    read = self._momentum_read(frame)
                    reason = (
                        f"The take was reached earlier and held while momentum rose; now "
                        f"{momentum_words(read) if read is not None else 'the confirmation is off'}. "
                        f"Selling at ${float(last['close']):,.2f}."
                    )
                else:
                    reason = (
                        f"The {ts:%H:%M} bar traded up to ${float(last['high']):,.2f}, through the "
                        f"${float(take):,.2f} take ({self._take_phrase(minute)}) — "
                        + ("the rise the model was asked about happened."
                           if self.config.gm_take_min_k is None
                           else "as far as this time of day's volatility asks for.")
                    )
                    if self.config.gm_confirm_bars:
                        reason += f" Momentum does not hold it: {momentum_words(self._momentum_read(frame))}."
                self._sell(state, tracker, position, self.EXIT_TAKE, reason)
                took = True
                position = tracker.position_for(self.ticker)
                if position > 0:
                    return rule_agent.SOLD
        sold = rule_agent.SOLD if took else rule_agent.HOLD
        # 2. The flatten: nothing is held into the close, and nothing bought.
        if self.closing_soon():
            if position > 0:
                self._sell(state, tracker, position, self.EXIT_FLATTEN, (
                    f"Inside the last {self.config.flatten_before_close_min} min of the "
                    "session: flattening before the close."
                ))
                return rule_agent.SOLD
            return sold
        fc = self.forecast
        # 3. The bar's own forecast (`trading._session`'s close-of-bar step).
        if position > 0:
            if self.config.gm_exit is not None and fc is not None and fc["edge"] < self.config.gm_exit:
                self._sell(state, tracker, position, self.EXIT_SIGNAL, (
                    f"The edge fell to {fc['edge']:+.3f}, under the {self.config.gm_exit:+g} exit: "
                    "the model no longer sees the rise as likelier than the clock and volatility do."
                ))
                return rule_agent.SOLD
            return rule_agent.HOLD
        if fc is None or fc["edge"] < self.config.gm_enter:
            return sold
        if minute > APPLE_TRADER_GM_LAST_ENTRY_MINUTE:
            return sold
        if minute < int(self.session.get("cool_until") or 0):
            return sold
        if self._refuse_on_levels(state, frame, ts, last, minute):
            return sold
        if self._refuse_on_momentum(state, frame, ts):
            return sold
        if self.entry_refused(state, ts):
            return sold
        return rule_agent.BOUGHT if self._buy(state, tracker, last, ts, minute) else sold

    # --- what a hybrid adds (`HighLow2GoodMomentsTrader`) ---------------------

    def _after_forecast(self, bundle: dict, state: AppState, frame, today, ts) -> None:
        """What else a new bar is read for once its forecast is in, before it
        is recorded. Nothing, on GoodMoments alone."""

    def _refuse_on_levels(self, state: AppState, frame, ts, bar, minute: int) -> bool:
        """Whether a second model refuses an entry the edge calls for. Never,
        on GoodMoments alone."""
        return False

    def _entry_note(self, bar, minute: int) -> str:
        """What a buy's reasoning says about a second model, as a sentence
        ending in a space. Nothing, on GoodMoments alone."""
        return ""

    # --- the momentum confirmation ------------------------------------------

    def _momentum_read(self, frame) -> "dict | None":
        """This bar's read of the behaviour table (`momentum_read`), or None."""
        return momentum_read(frame["close"], self.config.gm_confirm_bars, self.minute_move)

    def _hold_past_take(self, frame) -> "str | None":
        """Why a reached take is not sold on this bar, or None to sell it: with
        the confirmation on, momentum still positive -- the rise has not ended
        -- or no read at all."""
        if not self.config.gm_confirm_bars:
            return None
        read = self._momentum_read(frame)
        if read is None:
            return momentum_unknown(self.ticker, self.config.gm_confirm_bars, self.minute_move, len(frame))
        return None if read["row"].sell_target else momentum_words(read)

    def _refuse_on_momentum(self, state: AppState, frame, ts) -> bool:
        """Whether the confirmation refuses an entry the edge calls for: the
        table's "buy" needs positive momentum, or neutral momentum whose change
        is not negative. Said once per stretch of refusals for one reason; the
        next bar is judged again."""
        if not self.config.gm_confirm_bars:
            return False
        read = self._momentum_read(frame)
        if read is not None and read["row"].buy:
            self._momentum_note = None
            return False
        if read is None:
            why = momentum_unknown(self.ticker, self.config.gm_confirm_bars, self.minute_move, len(frame))
            key = "unknown"
        else:
            why, key = momentum_words(read), f"{read['mom_class']}/{read['change_class']}"
        if key != self._momentum_note:
            self._momentum_note = key
            _log(state, {"type": "status", "text": (
                f"The edge calls for a buy at {ts:%H:%M}, but {why}. Not buying yet; each "
                "bar is judged again."
            )})
        return True

    # --- the session ------------------------------------------------------

    def _roll_session(self, today) -> None:
        if self.session is not None and self.session["date"] != today:
            self.session = None
            self.entry = None
            self.last_bar_ts = None
            self.forecast = None
            self._forecast_note = None
        if self.blocked is not None and self.blocked["date"] != today:
            self.blocked = None

    def _credentials(self, state: AppState) -> "tuple[str | None, str | None]":
        return ff_credentials(state)

    def _prewarm(self, bundle: dict, state: AppState) -> None:
        """Fill the first forecast's caches in the background, once a session:
        ~130 days of SIP and IEX minute bars, last night's option table and the
        analyst table -- tens of seconds cold, which would otherwise hold the
        09:31 forecast. Live only; a failure is not reported, the first
        forecast makes the same reads and reports its own."""
        if getattr(state, "bar_tape_override", ""):
            return
        today = pd.Timestamp(clock.now().astimezone(market_hours.MARKET_TZ).date())
        if self._prewarmed == today or not _goodmoments().is_session(today):
            return
        self._prewarmed = today
        module, ticker = _goodmoments(), self.ticker
        key, secret = self._credentials(state)

        def work() -> None:
            try:
                module.warm(bundle, ticker, today, key, secret)
            except Exception:  # noqa: BLE001 -- the first forecast retries and reports
                pass

        threading.Thread(target=clock.inherit(work), name=f"goodmoments-warm-{ticker}", daemon=True).start()

    def _plan_session(self, bundle: dict, state: AppState, today) -> bool:
        """The session's ADR, once. False -- and the session blocked, reported
        once -- when there is none to be had."""
        module = _goodmoments()
        key, secret = self._credentials(state)
        try:
            if module.is_half_day(today):
                raise ValueError(
                    "today closes at 13:00, and the model was fitted on full sessions only"
                )
            adr = module.session_adr(bundle, self.ticker, today, key, secret)
            if not (math.isfinite(adr) and adr > 0):
                raise ValueError("the history gives no ADR for today")
            vol_scale = self._vol_scale_curve()
        except Exception as exc:  # noqa: BLE001 -- fatal for the session, not the run
            self.blocked = {"date": today, "reason": str(exc)}
            _log(state, {"type": "error", "text": (
                f"Apple Trader cannot read GoodMoments' forecasts for {self.ticker} today, so it "
                f"will not trade this session: {exc}"
            )})
            return False
        self.session = {"date": today, "adr": float(adr), "cool_until": 0, "history": [],
                        "vol_scale": vol_scale}
        prior = self.slot.levels(state) or {}
        if (
            prior.get("rows")
            and prior.get("ticker") == self.ticker
            and pd.Timestamp(prior.get("date")) == pd.Timestamp(today)
            and _record_strategy(prior) == self.config.strategy
        ):
            # A restart mid-session: the earlier run's record stays on the
            # chart, and its open position's take and stop are taken back.
            self.session["history"] = list(prior["rows"])
            self._resume(state, prior.get("memory"))
        exits = []
        if self.config.gm_take_k and vol_scale is not None:
            peak, trough = int(vol_scale.argmax()), int(vol_scale.argmin())
            exits.append(
                f"a take over the fill that follows the intraday volatility curve, "
                f"{self.config.gm_take_k:g} ADR ({math.exp(self.config.gm_take_k * adr) - 1:.2%}) at "
                f"its most volatile minute ({_session_clock(peak)}) and "
                f"{self.config.gm_take_min_k:g} ADR ({math.exp(self.config.gm_take_min_k * adr) - 1:.2%}) "
                f"at its quietest ({_session_clock(trough)}), moved every bar"
            )
        elif self.config.gm_take_k:
            exits.append(f"a take {self.config.gm_take_k:g} ADR "
                         f"({math.exp(self.config.gm_take_k * adr) - 1:.2%}) over the fill")
        if self.config.gm_stop_k:
            exits.append(f"a stop {self.config.gm_stop_k:g} ADR "
                         f"({1 - math.exp(-self.config.gm_stop_k * adr):.2%}) under it")
        _log(state, {"type": "analysis", "text": (
            f"{self.ticker} today: ADR {adr:.2%} (the 14-session mean high-to-low range), so the "
            f"question is a rise of {math.exp(_goodmoments().TARGET_ADR * adr) - 1:.2%} before the close. Buy "
            f"when the model's probability of it is at least {self.config.gm_enter:+g} over the "
            "time-and-volatility model's"
            + (f"; {', '.join(exits)}" if exits else "")
            + (f"; sell when the edge falls under {self.config.gm_exit:+g}"
               if self.config.gm_exit is not None else "")
            + "."
            + (f" Momentum over the last {self.config.gm_confirm_bars} bars confirms both: no buy "
               "while it is negative (or flat and turning down), and a reached take is held "
               "while it is still positive."
               if self.config.gm_confirm_bars else "")
        )})
        return True

    def _vol_scale_curve(self):
        """The session's volatility scale, one value per bar (09:30 first,
        `intraday_vol_model.volatility_scale`), when the take follows it; None
        with a flat take. Raises when the take should follow a curve this
        symbol has none of -- trading a flat take instead would be a rule
        nobody chose."""
        if self.config.gm_take_min_k is None or not self.config.gm_take_k:
            return None
        curve = intraday_vol_model.load(self.ticker)
        if curve is None:
            raise ValueError(
                f"the take is set to follow the intraday volatility curve, and there is none "
                f"for {self.ticker} ({intraday_vol_model.model_path(self.ticker)})"
            )
        return intraday_vol_model.volatility_scale(curve)

    def _scale_at(self, minute: int) -> "float | None":
        """This bar's place on the session's volatility scale, or None with a
        flat take."""
        scale = (self.session or {}).get("vol_scale")
        if scale is None:
            return None
        return float(scale[min(max(int(minute), 0), len(scale) - 1)])

    def _exits(self, fill: float, minute: int) -> dict:
        """The take and the stop for a fill, as they stand at this minute."""
        return gm_exits(self.config, fill, self.session["adr"], self._scale_at(minute))

    def _move_take(self, minute: int) -> None:
        """The volatility-following take, moved to this bar's minute before it
        is checked: the same distance from the fill this time of day's
        volatility gives any new position. The stop does not move."""
        if self.entry is None or self._scale_at(minute) is None or not self.entry.get("price"):
            return
        self.entry["take"] = self._exits(float(self.entry["price"]), minute)["take"]

    def _take_phrase(self, minute: int) -> str:
        """How far over the fill the take is at this minute, in words."""
        take_k = gm_take_k_at(self.config, self._scale_at(minute))
        if self._scale_at(minute) is None:
            return f"{take_k:g} ADR over the fill"
        return (
            f"{take_k:.2f} ADR over the fill at {_session_clock(minute)}'s volatility — "
            f"{self.config.gm_take_k:g} at the session's most volatile minute, "
            f"{self.config.gm_take_min_k:g} at its quietest"
        )

    def _forecast_for(self, bundle: dict, state: AppState, today, minute: int) -> "dict | None":
        """This bar's forecast, or None (after 15:58, or when it cannot be made
        -- said once per distinct reason)."""
        if minute > _goodmoments().LAST_FORECAST:
            return None
        key, secret = self._credentials(state)
        try:
            found = _goodmoments().forecast_at(bundle, self.ticker, today, minute, key, secret)
        except Exception as exc:  # noqa: BLE001 -- one bar without a forecast is not the day
            note = str(exc)
            if note != self._forecast_note:
                self._forecast_note = note
                _log(state, {"type": "error", "text": (
                    f"No GoodMoments forecast for {self.ticker} at bar {minute}: {note}"
                )})
            return None
        self._forecast_note = None
        return found

    # --- orders -----------------------------------------------------------

    def _buy(self, state: AppState, tracker: DecisionTracker, bar, ts, minute: int) -> bool:
        fc = self.forecast
        adr = self.session["adr"]
        exits = []
        if self.config.gm_take_k:
            exits.append(f"a take {self._take_phrase(minute)}"
                         + (", moved every bar" if self._scale_at(minute) is not None else ""))
        if self.config.gm_stop_k:
            exits.append(f"a stop {self.config.gm_stop_k:g} ADR under it")
        if self.config.gm_exit is not None:
            exits.append(f"a sale once the edge is under {self.config.gm_exit:+g}")
        reasoning = (
            f"The model puts the chance of a {math.exp(_goodmoments().TARGET_ADR * adr) - 1:.2%} rise before the "
            f"close at {fc['p']:.1%}, against {fc['p_tv']:.1%} from the clock and today's volatility "
            f"alone: an edge of {fc['edge']:+.3f}, at or above the {self.config.gm_enter:+g} entry. "
            f"{self._entry_note(bar, minute)}"
            f"Buying; {', '.join(exits + ['the closing flatten'])}."
        )
        bought = self.buy(state, tracker, float(bar["close"]), reasoning)
        if bought:
            fill = float(self.entry["price"])
            self.entry.update({"ts": ts, "edge": float(fc["edge"]), **self._exits(fill, minute)})
            self._record(state, ts, bar)
        return bought

    def _sell(self, state: AppState, tracker: DecisionTracker, quantity: float, kind: str,
              reasoning: str) -> None:
        entry = dict(self.entry or {})
        decision = self.sell(state, tracker, quantity, reasoning, log_extra={"exit": kind})
        if decision.status == "filled" and self.entry is None and tracker.position_for(self.ticker) > 0:
            # Partly filled: what is left is the same position, same levels.
            self.entry = entry

    def _resume(self, state: AppState, memory: "dict | None") -> None:
        """Take back an earlier run's open position (its take and stop) and its
        pause after a stop, on a restart the same day. The entry is only a
        candidate: `run_cycle` drops it if the ledger is flat."""
        if not memory:
            return
        self.session["cool_until"] = int(memory.get("cool_until") or 0)
        entry = memory.get("entry")
        if entry and self.entry is None:
            self.entry = dict(entry)
            levels = [f"{name} ${float(entry[name]):,.2f}" for name in ("take", "stop") if entry.get(name)]
            _log(state, {"type": "status", "text": (
                f"{self.ticker}: continuing today's earlier run's position"
                + (f" — {', '.join(levels)}" if levels else "") + "."
            )})

    # --- the record and the log -------------------------------------------

    def memory(self) -> dict:
        return {
            "entry": dict(self.entry) if self.entry is not None else None,
            "cool_until": int((self.session or {}).get("cool_until") or 0),
        }

    def publish_memory(self, state: AppState) -> None:
        levels = self.slot.levels(state)
        if (
            not levels
            or self.session is None
            or levels.get("ticker") != self.ticker
            or pd.Timestamp(levels.get("date")) != pd.Timestamp(self.session["date"])
        ):
            return
        self.slot.publish(state, {**levels, "memory": self.memory()})

    def _record(self, state: AppState, ts, bar) -> None:
        """This bar's forecast and the position's take and stop, added to the
        session's record and published: what the chart draws while the run is
        live."""
        row = self._row(ts, bar)
        history = self.session.setdefault("history", [])
        if history and history[-1]["t"] == row["t"]:
            history[-1] = row
        else:
            history.append(row)
        self.slot.publish(state, {
            "ticker": self.ticker,
            "date": self.session["date"],
            "config": self.config,
            "rows": history,
            "memory": self.memory(),
            "seed": None,
        })

    def _row(self, ts, bar) -> dict:
        """One bar of the record: the forecast, and the open position's take
        and stop."""
        fc = self.forecast or {}
        return {
            "t": ts,
            "model_key": self.config.model_key,
            "price": float(bar["close"]),
            "p": fc.get("p"),
            "p_tv": fc.get("p_tv"),
            "edge": fc.get("edge"),
            "take": (self.entry or {}).get("take"),
            "stop": (self.entry or {}).get("stop"),
        }

    def _read_summary(self, bar, ts, position: float, minute: int, frame=None) -> str:
        price = float(bar["close"])
        parts = [f"{self.ticker} {ts:%H:%M} ${price:,.2f}"]
        fc = self.forecast
        if fc is not None:
            parts.append(
                f"P(+{_goodmoments().TARGET_ADR:g} ADR by the close) {fc['p']:.1%} vs {fc['p_tv']:.1%} on time and volatility "
                f"(edge {fc['edge']:+.3f}, entry {self.config.gm_enter:+g})"
            )
        else:
            parts.append("no forecast")
        read = self._momentum_read(frame) if frame is not None and self.config.gm_confirm_bars else None
        if read is not None:
            parts.append(
                f"{read['n']}-bar momentum {read['mom']:+.3f} $/bar ({read['mom_class']}), "
                f"change {read['change_class']}"
            )
        if position > 0 and self.entry:
            entry_price = float(self.entry["price"])
            pnl = (price / entry_price - 1) * 100 if entry_price else 0.0
            parts.append(f"long {position:g} sh @ ${entry_price:,.2f} ({pnl:+.2f}%)")
            for name in ("take", "stop"):
                if self.entry.get(name):
                    parts.append(f"{name} ${float(self.entry[name]):,.2f}")
            if self.entry.get("take_reached"):
                parts.append("past the take — held while momentum stays positive")
        elif self.session and minute < int(self.session.get("cool_until") or 0):
            parts.append("stopped out — no entry until the pause is over")
        return " · ".join(parts)


# --- HighLow2 + GoodMoments --------------------------------------------------


class HighLow2GoodMomentsTrader(GoodMomentsTrader):
    """GoodMoments' rules, with HighLow2's buy level as a second vote.

    The user's design (2026-10-21):

    * **the opening window** (09:30-09:34, the bars HighLow2's forecast is made
      from, before it exists): a buy is GoodMoments' alone -- the edge at the
      entry, as `GoodMomentsTrader` buys;
    * **from the 09:35 bar on**: a buy needs both -- the edge at the entry *and*
      the bar's close at or under HighLow2's buy level, `H - buy_k x unit`
      under its predicted high, the price a HighLow2 run would rest its buy
      at. The close rather than the bar's low because the order here is sent
      once the bar has closed, at the market, and HighLow2's own buy is a limit
      at its level: a bar that dipped to the level and closed back above it
      would not have filled HighLow2 either;
    * **every exit is GoodMoments'**: the take and the stop in ADRs from the
      fill (and the take's fall with the intraday volatility curve), the pause
      after a stop, the optional sale on a fading edge and the momentum hold,
      the flatten.

    HighLow2's level is the one a HighLow2 run would rest, walked by the same
    code: a `DayRangeTrader` whose plan is made from the same opening window by
    `session_forecast`, moved through a breach under this configuration's
    `breach_update`, `keep_width` and `contain_range` (`_move_range`). It only
    votes. It rests nothing, and nothing of HighLow2's exit applies -- the sell
    level, its stop, the momentum take, the breaker and the ladder are the
    day-range rules', not HighLow2's model, and this trades GoodMoments'.

    A session HighLow2 cannot forecast is traded on GoodMoments alone through
    the opening window and buys nothing after it -- there is no second vote to
    agree -- which is logged once; a position held then keeps its exits.
    """

    def __init__(self, config: "AppleTraderConfig | None" = None) -> None:
        super().__init__(
            config or AppleTraderConfig(model_key=apple_models.HIGHLOW2_GOODMOMENTS_KEY)
        )
        # HighLow2's levels as a HighLow2 run would rest them, the plan set at
        # the close of the opening window; None before it, or with no forecast.
        self.levels: "DayRangeTrader | None" = None
        # Why HighLow2 has no forecast today, once it is known.
        self.levels_problem: "str | None" = None
        self._levels_day = None
        # How many bars the opening window is: read off HighLow2's bundle at
        # the first bar, the 5 every HighLow2 bundle carries until then.
        self.window_minutes = 5
        # The bar the vote last refused a buy on, so a run of refusals is
        # logged once.
        self._levels_refused_at = None

    # --- HighLow2's levels -------------------------------------------------

    def _after_forecast(self, bundle: dict, state: AppState, frame, today, ts) -> None:
        """Make HighLow2's forecast once its window has closed, and move its
        levels with every bar after it."""
        part = bundle.get(apple_models.HIGHLOW2_KEY)
        self.window_minutes = _dayrange().opening_minutes(part)
        if self._levels_day != today:
            self.levels, self.levels_problem, self._levels_day = None, None, today
        if self.levels is None:
            if self.levels_problem is None and len(frame) >= self.window_minutes:
                self._plan_levels(part, state, frame, today)
            return
        if ts > self.levels.plan["opening_end"]:
            self._move_levels(state, frame, ts)

    def _plan_levels(self, part: "dict | None", state: AppState, frame, today) -> None:
        """HighLow2's 9:35 forecast and its buy level, walked on through every
        bar since -- a run started late reads the session as one up since
        9:35 would have. A failure is the day's: logged once, and every buy
        after the window refused (`_refuse_on_levels`)."""
        try:
            if part is None:
                raise ValueError("the bundle carries no HighLow2 model")
            opening, _tape = fetch_opening_window(state, frame, self.window_minutes, ticker=self.ticker)
            # A replay's state carries placeholder keys, as `DayRangeTrader` says.
            replayed = bool(getattr(state, "bar_tape_override", ""))
            forecast, open_price = session_forecast(
                part, self.ticker, opening, today,
                key=None if replayed else state.api_key,
                secret=None if replayed else state.api_secret,
            )
        except Exception as exc:  # noqa: BLE001 -- the day's second vote, not the run
            self.levels_problem = str(exc)
            _log(state, {"type": "error", "text": (
                f"HighLow2 cannot forecast today's {self.ticker} range ({exc}), so after the "
                "opening window nothing is bought — the two models have to agree there. "
                "Positions keep GoodMoments' exits."
            )})
            return
        walker = DayRangeTrader(self.config)
        opening_end = opening.index[-1]
        walker.plan = {
            "date": today,
            "opening_end": opening_end,
            "open_price": float(open_price or opening["open"].iloc[0]),
            **forecast,
        }
        walker._set_levels(opening_end)
        for ts in frame.index:
            if ts > opening_end:
                walker._move_range(frame[frame.index <= ts], ts)
                walker._set_levels(ts)
        self.levels = walker
        plan = walker.plan
        _log(state, {"type": "analysis", "text": (
            f"HighLow2's forecast for {self.ticker} at the {opening_end:%H:%M} bar: high "
            f"${float(plan['pred_high']):,.2f}, low ${float(plan['pred_low']):,.2f}"
            + (f" (moved since: buy level now ${plan['buy_level']:,.2f})"
               if plan.get("range_updates") else "")
            + f". From the bar after it a buy also needs the close at or under its buy level, "
            f"{self.config.buy_k:g} × the {self.config.unit_phrase} under the predicted high: "
            f"${plan['buy_level']:,.2f} now"
            + ("" if self.config.breach_update == BREACH_OFF
               else ", moved with the forecast whenever the session trades outside it")
            + "."
        )})

    def _move_levels(self, state: AppState, frame, ts) -> None:
        """This bar's breach, if it made one: the forecast and the buy level
        moved as a HighLow2 run would move them, said in one line."""
        before = self.levels._move_range(frame, ts)
        self.levels._set_levels(ts)
        if before is None:
            return
        plan = self.levels.plan
        _log(state, {"type": "analysis", "text": (
            f"HighLow2's forecast for {self.ticker} moved at {ts:%H:%M}, the session having "
            f"traded outside it: range now ${float(plan['pred_low']):,.2f} – "
            f"${float(plan['pred_high']):,.2f}, buy level ${plan['buy_level']:,.2f} "
            f"(was ${before['buy_level']:,.2f})."
        )})

    def _buy_level(self) -> "float | None":
        plan = self.levels.plan if self.levels is not None else None
        return float(plan["buy_level"]) if plan else None

    # --- the vote ------------------------------------------------------------

    def _refuse_on_levels(self, state: AppState, frame, ts, bar, minute: int) -> bool:
        """Whether HighLow2 refuses the buy the edge calls for: never inside
        the opening window; after it, unless the bar closed at or under its buy
        level. Said once per run of refused bars; the next bar is judged
        again."""
        if minute < self.window_minutes:
            return False
        level, close = self._buy_level(), float(bar["close"])
        if level is not None and close <= level:
            self._levels_refused_at = None
            return False
        if level is None:
            why = (
                f"HighLow2 has no forecast today ({self.levels_problem})"
                if self.levels_problem else
                f"HighLow2's forecast is not made yet (it needs the first {self.window_minutes} "
                "bars of the session)"
            )
        else:
            plan = self.levels.plan
            why = (
                f"${close:,.2f} is above HighLow2's ${level:,.2f} buy level "
                f"({self.config.buy_k:g} × the {self.config.unit_phrase} under its predicted high "
                f"${float(plan['pred_high']):,.2f})"
            )
        previous = frame.index[-2] if len(frame) > 1 else None
        if self._levels_refused_at is None or self._levels_refused_at != previous:
            _log(state, {"type": "status", "text": (
                f"The edge calls for a buy at {ts:%H:%M}, but {why}. Not buying: after the "
                "opening window both models have to agree; each bar is judged again."
            )})
        self._levels_refused_at = ts
        return True

    def _entry_note(self, bar, minute: int) -> str:
        if minute < self.window_minutes:
            return (
                "Inside the opening window, before HighLow2's forecast: GoodMoments' "
                "signal alone. "
            )
        return (
            f"HighLow2 agrees: the ${float(bar['close']):,.2f} close is at or under its "
            f"${self._buy_level():,.2f} buy level. "
        )

    # --- the session, the record and the log -------------------------------

    def _plan_session(self, bundle: dict, state: AppState, today) -> bool:
        planned = super()._plan_session(bundle, state, today)
        if planned:
            _log(state, {"type": "analysis", "text": (
                f"Hybrid with HighLow2: through the opening window (its first "
                f"{self.window_minutes} bars) a buy is GoodMoments' alone; from the bar after "
                f"it, it also needs the close at or under HighLow2's buy level, "
                f"{self.config.buy_k:g} × the {self.config.unit_phrase} under its predicted "
                "high. Every exit is GoodMoments'."
            )})
        return planned

    def _row(self, ts, bar) -> dict:
        """GoodMoments' row, with HighLow2's levels from the bar after its
        window: what the chart draws the buy level from."""
        row = super()._row(ts, bar)
        plan = self.levels.plan if self.levels is not None else None
        after = plan is not None and ts > plan["opening_end"]
        row.update({
            "buy": float(plan["buy_level"]) if after else None,
            "pred_high": float(plan["pred_high"]) if after else None,
            "pred_low": float(plan["pred_low"]) if after else None,
        })
        return row

    def _read_summary(self, bar, ts, position: float, minute: int, frame=None) -> str:
        text = super()._read_summary(bar, ts, position, minute, frame)
        level = self._buy_level()
        if minute < self.window_minutes:
            return text + " · HighLow2: opening window, GoodMoments alone"
        if level is None:
            return text + " · no HighLow2 buy level"
        side = "at or under" if float(bar["close"]) <= level else "above"
        return text + f" · HighLow2 buy ${level:,.2f} (close {side})"


def build_trader(config: AppleTraderConfig, bundle: "dict | None" = None):
    """The state machine this configuration's model calls for.

    The one place a model's strategy turns into an object. Every launch path --
    the live loop below, SimLab's `rule_agents._build_apple` -- goes through
    here, so a second strategy is added in one place rather than in whichever
    entry points were remembered. The returned object exposes
    `run_cycle(bundle, state, tracker)` and nothing else a caller needs.
    """
    if config.strategy == apple_models.STRATEGY_MOMENTUM:
        return FastFuriousTrader(config)
    if config.model_key == apple_models.HIGHLOW2_GOODMOMENTS_KEY:
        return HighLow2GoodMomentsTrader(config)
    if config.strategy == apple_models.STRATEGY_PROBABILITY:
        return GoodMomentsTrader(config)
    return DayRangeTrader(config)


# --- the loop ---------------------------------------------------------------

def _breach_side(config: AppleTraderConfig) -> str:
    """What a breach moves under this policy, for the log lines that say so."""
    if config.breach_update == BREACH_SHIFT or (
        config.keep_width and config.breach_update == BREACH_BROWNIAN
    ):
        return "both sides of it"
    return "the breached side"


def _momentum_armed_summary(config: AppleTraderConfig, model, bundle: dict) -> str:
    """`_armed_summary` for Fast&Furious's rules."""
    scores = ((bundle.get("metadata") or {}).get("walk_forward_scores") or {}).get("2026") or {}
    quality = f", rank correlation {scores['ic']:.3f} on its 2026 walk-forward" if scores.get("ic") else ""
    exits = [
        f"{config.ff_hold_minutes} minutes after the last buy signal (a fresh one restarts the clock)"
    ]
    if config.ff_exit_on_sell_signal:
        exits.append("at once on a forecast at or under minus that threshold")
    if config.ff_stop_k:
        exits.append(
            f"at a stop {config.ff_stop_k:g} × the forecast 2-hour drop under the fill, after which "
            "it buys again only once the forecast has left the buy zone"
        )
    exits.append("at the closing flatten")
    return (
        f"Apple Trader armed on {model.label} (fitted {bundle.get('trained_at') or 'unknown'}"
        f"{quality}): at the close of every bar from 09:31 to 15:30 it forecasts {config.ticker}'s "
        "average momentum over the next 30 minutes, and buys when that forecast is at or above "
        f"the {config.ff_entry_quantile:.0%} quantile of the last 20 sessions' forecasts. It "
        f"sells {', '.join(exits[:-1])}, or {exits[-1]}."
    )


def _probability_armed_summary(config: AppleTraderConfig, model, bundle: dict) -> str:
    """`_armed_summary` for GoodMoments' rules."""
    scores = (bundle.get("metadata") or {}).get("test_scores") or {}
    quality = (
        f", test AUC {scores['auc']:.3f} ({scores['auc_30min']:.3f} within the half hour)"
        if scores.get("auc") and scores.get("auc_30min") else ""
    )
    exits = []
    if config.gm_take_k and config.gm_take_min_k is not None:
        exits.append(
            f"at a take over the fill that follows the intraday volatility curve — "
            f"{config.gm_take_k:g} ADR at the session's most volatile minute, "
            f"{config.gm_take_min_k:g} ADR at its quietest, moved every bar"
        )
    elif config.gm_take_k:
        exits.append(f"at a take {config.gm_take_k:g} ADR over the fill")
    if config.gm_stop_k:
        exits.append(
            f"at a stop {config.gm_stop_k:g} ADR under it, after which it buys nothing for "
            f"{APPLE_TRADER_GM_COOLDOWN_BARS} bars"
        )
    if config.gm_exit is not None:
        exits.append(f"once the edge is under {config.gm_exit:+g}")
    exits.append("at the closing flatten")
    target = _goodmoments().TARGET_ADR
    confirm = (
        f" Momentum over the last {config.gm_confirm_bars} bars confirms both sides: no buy while "
        "it is negative or flat and turning down, and a reached take is held while it is still "
        "positive, sold on the first bar it is not."
        if config.gm_confirm_bars else ""
    )
    levels_model = apple_models.levels_of(config.model_key)
    if levels_model is not None:
        part = bundle.get(levels_model) or {}
        made = 30 + _dayrange().opening_minutes(part)
        confirm += (
            f" From 9:{made:02d}, when {apple_models.get(levels_model).label} forecasts today's "
            f"high and low, a buy also needs the bar's close at or under its buy level, "
            f"{config.buy_k:g} × the {config.unit_phrase} under the predicted high; before "
            "then the edge alone buys."
        )
    return (
        f"Apple Trader armed on {model.label} (fitted {bundle.get('trained_at') or 'unknown'}"
        f"{quality}): at the close of every bar it forecasts the chance that {config.ticker} trades "
        f"{target:g} ADR above that bar's close before the session ends, beside the same trees' "
        "chance on the clock and today's volatility alone, and buys when the first is at least "
        f"{config.gm_enter:+g} over the second (no new entry after 15:30). It sells "
        + (f"{', '.join(exits[:-1])}, or {exits[-1]}." if len(exits) > 1 else f"{exits[0]}.")
        + confirm
    )


def _armed_summary(config: AppleTraderConfig, model, bundle: dict) -> str:
    """The one line the log opens a run with: which model, and what it will do."""
    if config.strategy == apple_models.STRATEGY_MOMENTUM:
        return _momentum_armed_summary(config, model, bundle)
    if config.strategy == apple_models.STRATEGY_PROBABILITY:
        return _probability_armed_summary(config, model, bundle)
    metadata = bundle.get("metadata") or {}
    # TimeToChange3 files its held-out score as `test_metrics_ensemble`, HighLow
    # and HighLow2 as `test_metrics`; the dollar error is `mae_usd_mean` in all.
    scores = metadata.get("test_metrics_ensemble") or metadata.get("test_metrics") or {}
    mae = scores.get("mae_usd_mean")
    quality = f", held-out mean error ${mae:.2f}" if mae else ""
    exits = []
    if config.has_stop:
        exits.append(
            f"a stop {stop_phrase(config)} under the fill, after which the agent stops "
            "until ▶ Start Agent is pressed again"
        )
    if config.has_take:
        exits.append(
            f"a {config.take_fraction:.0%} take "
            + ("in profit or at a loss" if config.take_in_loss else "in profit")
            + (
                f" (at least {config.take_min_gain_fraction:g} × the predicted gain)"
                if config.take_min_gain_fraction else ""
            )
            + (
                f" (not before {config.take_after_minutes} min after the fill)"
                if config.take_after_minutes else ""
            )
            + (
                " once the price stalls "
                if config.take_trigger == TAKE_ZONE
                else " once momentum gives out "
            )
            + f"({fade_phrase(config)}), the rest kept for the sell level only if "
            f"it is {config.hold_min_gain_k:g} × {config.unit_phrase} or more above the fill and sold if the "
            "price comes back to it"
            + (" (a take at a loss leaves that to the stop)" if config.take_in_loss else "")
        )
    managed = f" The exit adds {'; and '.join(exits)}." if exits else ""
    if config.momentum_confirmation_bars:
        no_fall = (
            f" Both levels wait for momentum over the last "
            f"{config.momentum_confirmation_bars} bars: it buys at the buy level only "
            "while momentum is positive, or neutral and not turning down, and holds past the "
            "sell level while momentum is still positive."
        )
    elif config.max_fall_k:
        no_fall = (
            f" No buy while the price has fallen more than {config.max_fall_k:g} × "
            f"{config.unit_phrase} over the last {config.fall_bars} bars."
        )
    else:
        no_fall = ""
    breaker = (
        ""
        if not config.min_win_k
        else (
            f" A trade that closes for no more than {config.min_win_k:g} × {config.unit_phrase} a share "
            "stands the agent down for the rest of the session."
        )
    )
    breach = (
        ""
        if config.breach_update == BREACH_OFF
        else (
            f" A session that trades outside the forecast moves {_breach_side(config)} "
            f"({BREACH_LABELS[config.breach_update].lower()}) and both levels with it."
        )
    )
    reference = "the predicted high"
    # When the forecast is made, and of what: the window the bundle reads, and
    # the whole day's extremes or (a HighLow2 bundle with a rest head) those
    # still to come after it.
    made = 30 + _dayrange().opening_minutes(bundle)
    extremes = (
        f"the rest of today's {config.ticker} high and low"
        if bundle.get("rest_head")
        else f"today's {config.ticker} high and low"
    )
    return (
        f"Apple Trader armed on {model.label} (fitted "
        f"{bundle.get('trained_at', 'unknown')}{quality}): at 9:{made:02d} it forecasts where "
        f"{extremes} will land, then rests a buy "
        f"{config.buy_k:g} × the {config.unit_phrase} below {reference} and a sell "
        f"{config.sell_k:g} below it, until the closing flatten.{breach}{no_fall}"
        f"{managed}{breaker}"
    )


def _apple_trader_loop(
    state: AppState,
    tracker: DecisionTracker,
    config: AppleTraderConfig,
    cycle_sec: int,
    stop_event: threading.Event,
) -> None:
    model = apple_models.get(config.model_key)
    # Three ways the run is over before it starts, each reported and none
    # raised. The pairing check runs before the load, so a model that was never
    # fitted on this symbol is never reported as a file that failed to appear.
    bundle = None
    refusal = model_ticker_error(config)
    if refusal is None:
        bundle = apple_models.load(config.model_key, config.ticker)
        if bundle is None:
            refusal = (
                f"{apple_models.unavailable_reason(config.model_key, config.ticker)} "
                f"Apple Trader cannot run without it."
            )
        else:
            refusal = config_error(config, bundle)
    if refusal is not None:
        _log(state, {"type": "error", "text": refusal})
        rule_agent.end_session(state, tracker, None)
        return

    trader = build_trader(config, bundle)
    _log(state, {"type": "status", "text": _armed_summary(config, model, bundle)})

    def cycle() -> str:
        outcome = trader.run_cycle(bundle, state, tracker)
        activity = getattr(trader, "activity", None)
        if activity is not None:
            state.agent_activity = activity(outcome, tracker)
        publish = getattr(trader, "publish_memory", None)
        if publish is not None:
            publish(state)
        halt = getattr(trader, "halt", None)
        if halt and not stop_event.is_set():
            # A stop-out says the strategy is wrong about today: the run ends
            # here, as if ▶ Stop had been pressed, and buys again only after
            # ▶ Start Agent.
            _log(state, {"type": "status", "text": (
                f"{config.ticker} {halt} — Apple Trader stops. Press ▶ Start Agent to "
                "trade again today."
            )})
            stop_event.set()
        return outcome

    rule_agent.run_loop(
        state, tracker, cycle, stop_event, cycle_sec, "Apple Trader",
    )


def launch_apple_trader(
    state: AppState,
    tracker: DecisionTracker,
    config: "AppleTraderConfig | None" = None,
    cycle_sec: int = APPLE_TRADER_CYCLE_SEC,
) -> None:
    """Stop any running agent for this state, then start the Apple Trader loop.

    It trades only the one symbol its config names, which must already be
    streamed. No LLM client, no tools and no tactics are involved -- the loop
    places its own orders through the same `DecisionTracker` as every other
    personality.
    """
    config = config or AppleTraderConfig()
    rule_agent.launch(
        state, tracker, APPLE_TRADER_KEY, config.ticker,
        target=_apple_trader_loop,
        args=(state, tracker, config, cycle_sec),
        stop_agent=stop_agent,
    )
