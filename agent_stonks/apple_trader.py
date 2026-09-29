"""Apple Trader -- a rule-based agent with no LLM in the loop.

Every other personality in `agent_stonks.agent` is a system prompt handed to a
model that reasons its way to a decision. This one is a plain loop: once a
minute it looks at the minute bar that just closed and applies fixed rules
built on a saved model from FinNotebooks. Same paper ledger, same fill path,
same log -- only the decision-making is deterministic, so the same tape always
produces the same trades.

The rules are TimeToChange3's day-range rules (`DayRangeTrader`): one forecast
at 9:35 of where the session's high and low will land, then two resting levels
derived from it and nothing more asked of the model all day.
`AppleTraderConfig.model_key` names the model and `build_trader` turns it into
a state machine -- see `agent_stonks.apple_models`, which owns that mapping --
so a second strategy is a registry entry and a class rather than a rewrite.

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

import math
import threading
from dataclasses import dataclass, replace
from datetime import timedelta, timezone
from typing import Optional

import pandas as pd

from . import agent as agent_mod
from . import state as state_mod
from . import (
    apple_models, bar_history, clock, historical, intraday_vol_model, market_hours,
    momentum_regime, rule_agent,
)
from .agent import stop_agent
from .rule_agent import BaseTrader
from .state import append_agent_log as _log
from .config import (
    APPLE_TRADER_BREACH_UPDATE,
    APPLE_TRADER_CONTAIN_RANGE,
    APPLE_TRADER_KEEP_WIDTH,
    APPLE_TRADER_BREACH_EXIT,
    APPLE_TRADER_BUY_K,
    APPLE_TRADER_CYCLE_SEC,
    APPLE_TRADER_DAYRANGE_LEVELS,
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
    APPLE_TRADER_STOP_GAIN_FRACTION,
    APPLE_TRADER_TAKE_FRACTION,
    APPLE_TRADER_TAKE_MIN_GAIN_FRACTION,
    APPLE_TRADER_TUNED_LEVELS,
    BREACH_BROWNIAN,
    BREACH_LABELS,
    BREACH_OFF,
    BREACH_POLICIES,
    BREACH_SHIFT,
    LEVELS_DAYRANGE,
    LEVELS_INTRADAY,
    LEVEL_SOURCES,
    LEVEL_SOURCE_LABELS,
    LEVEL_UNITS,
    LEVEL_UNIT_LABELS,
    LEVEL_UNIT_PHRASES,
    LEVEL_UNIT_TOKENS,
    APPLE_TRADER_LEVEL_UNIT,
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


def dayrange_levels(ticker: str, model_key: "str | None" = None) -> "tuple[float, float]":
    """The `(buy_k, sell_k)` a run of this model on this symbol starts from.

    Per model and instrument, since the same distances are different prices
    under a different forecast: SimLab's tuning pick for the pair
    (`config.APPLE_TRADER_TUNED_LEVELS`, with how far each deserves trust).
    A pair never tuned falls back to the instrument's notebook-05 sweep
    (`config.APPLE_TRADER_DAYRANGE_LEVELS`), and a symbol never swept at all
    to the notebook's specified pair.
    """
    symbol = (ticker or DEFAULT_TICKER).strip().upper()
    tuned = APPLE_TRADER_TUNED_LEVELS.get((str(model_key or APPLE_TRADER_MODEL), symbol))
    if tuned is not None:
        return tuned
    return APPLE_TRADER_DAYRANGE_LEVELS.get(symbol, (APPLE_TRADER_BUY_K, APPLE_TRADER_SELL_K))


def min_win_for(ticker: str) -> float:
    """The circuit breaker a run on this symbol starts from, in ADRs a share.

    Per instrument because the number is only readable against that symbol's own
    `buy_k - sell_k`: the same 0.20 that lets GOOGL stand down on a bad trade
    would stand AAPL down on its best one. See `config.APPLE_TRADER_MIN_WIN`.
    """
    return APPLE_TRADER_MIN_WIN.get(
        (ticker or DEFAULT_TICKER).strip().upper(), APPLE_TRADER_MIN_WIN_K
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
    # two the target is worth. 0 switches the stop off.
    stop_gain_fraction: float = APPLE_TRADER_STOP_GAIN_FRACTION
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
    # momentum take below the sell level (`take_fraction`, in profit only).
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
    # How much of the predicted gain the close has to be above the fill before
    # the take may fire at all -- a share of `target_gain_k` level units, like
    # the stop. 0 is any profit, the rule before it existed.
    take_min_gain_fraction: float = APPLE_TRADER_TAKE_MIN_GAIN_FRACTION
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
    # Which number the two distances above are measured below -- one of
    # `config.LEVEL_SOURCES`. "dayrange" is the predicted high itself;
    # "intraday" is that forecast read through IntradayVolatility's
    # time-of-day shape, so the reference moves with the clock. See
    # `_set_levels`, and `config_error` for what it requires.
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
    # `can_scale_in`. The next add rests half-way between the last buy and the
    # bottom of the range, `reference - 1 x unit`.
    scale_in: bool = APPLE_TRADER_SCALE_IN
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

    def __post_init__(self) -> None:
        # Resolved before the checks below, which need numbers -- and before
        # `dayrange_levels`, because all three read the same normalised ticker.
        self.ticker = (self.ticker or DEFAULT_TICKER).strip().upper()
        if self.min_win_k is None:
            self.min_win_k = min_win_for(self.ticker)
        for name in (
            "stop_k", "stop_gain_fraction", "momentum_drop", "momentum_fade_bars",
            "negative_momentum_bars", "hold_min_gain_k", "min_win_k", "max_fall_k",
            "momentum_confirmation_bars", "take_min_gain_fraction",
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
        # A model that names a reference decides it, because that is the whole
        # difference between the two: "Day Range × Intraday Volatility" *is*
        # the day-range forecast read through the intraday shape, so a config
        # naming it and measuring below the flat high would be that model in
        # name only. Applied after validation so an impossible value is still
        # reported as one, and only for a model that declares a reference --
        # which leaves a record written when this was a separate choice saying
        # what it said. See `apple_models.AppleModel.level_source`.
        named = apple_models.MODELS.get(self.model_key)
        if named is not None and named.level_source != LEVELS_DAYRANGE:
            self.level_source = named.level_source
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

    @property
    def strategy(self) -> str:
        """Which rule set this configuration runs."""
        return apple_models.strategy(self.model_key)

    @property
    def target_gain_k(self) -> float:
        """What a target exit is playing for, in level units a share.

        `buy_level` and `sell_level` are both `reference - k x unit` off the
        same reference, so the gap between them is this many units whatever the
        reference does -- a breach moving the forecast, or an intraday curve
        moving it every minute, move both levels together.

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
        """Whether the momentum take is on, in any of its forms -- and with
        it the runner and the breakeven."""
        return bool(
            self.momentum_confirmation_bars
            or self.negative_momentum_bars
            or self.momentum_fade_bars
            or self.momentum_drop
        )


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
    """The momentum take's trigger in words, in whichever form it is written."""
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
    # the older 1σ fall from the peak, which stored records keep.
    if c.momentum_confirmation_bars:
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
        # The realised-gain gate, only while set, so every record written
        # before it existed (which replays at 0) keeps its filed signature.
        if c.take_min_gain_fraction:
            exits += f">={c.take_min_gain_fraction:g}G"
        exits += f",runner>={c.hold_min_gain_k:g}{unit}"
    if c.min_win_k:
        exits += f",min_win={c.min_win_k:g}{unit}"
    # Only where it can change a trade (`can_scale_in`), so a run at 100% signs
    # the same with the setting on or off -- and every record written before it
    # existed, which replays with it off, keeps the signature it was filed under.
    adds = ",adds=half" if c.can_scale_in else ""
    # The stop under the fill rather than under the next rung. Only where the
    # two can differ -- a ladder and a stop -- and only in today's form, so a
    # record from before it keeps the `adds=half` it was filed under.
    if c.can_scale_in and c.has_stop and not c.stop_under_next_buy:
        adds += ",stop@fill"
    if c.can_scale_in and c.add_under_fill:
        adds += ",add<fill"
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
    # Written only when the reference is *not* the one this model implies. For
    # "Day Range × Intraday Volatility" the model key already says it, and a
    # second token would be the same fact twice; for a record written when the
    # reference was a separate choice the model implies the flat high, so the
    # token appears exactly as it did and the record keeps its filed identity.
    implied = apple_models.MODELS.get(c.model_key)
    implied = implied.level_source if implied is not None else LEVELS_DAYRANGE
    levels = "" if c.level_source == implied else f",levels={c.level_source}"
    # "H" in the two distances is whatever `level_source` says it is, which is
    # why that token is next to them rather than at the end.
    return (
        f"{c.model_key}_{c.ticker}(buy=H-{c.buy_k:g}{unit},sell=H-{c.sell_k:g}{unit}{levels},"
        f"size={c.position_pct:g}%{adds}{exits}{breach})"
    )


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
    return model_ticker_error(config) or level_source_error(config)


def level_source_error(config: AppleTraderConfig) -> "str | None":
    """Why the chosen reference cannot be computed for this symbol, or None.

    Only the intraday shape can fail: it is a *second* model on top of the
    day-range bundle, fitted on its own set of symbols and shipped as its own
    file, so a configuration can name a pairing that the day-range checks above
    are perfectly happy with and this one is not. Checked before a run starts
    rather than discovered at 9:35, when the answer would be an agent that
    forecasts the day and then cannot say where to rest an order.
    """
    if config.level_source != LEVELS_INTRADAY:
        return None
    label = LEVEL_SOURCE_LABELS[LEVELS_INTRADAY]
    if not intraday_vol_model.covers(config.ticker):
        return (
            f"'{label}' reads IntradayVolatility's time-of-day shape, which was fitted on "
            f"{', '.join(intraday_vol_model.TICKERS)} only, so it cannot be used on "
            f"{config.ticker}. Run the flat day-range model, or pick another instrument."
        )
    if intraday_vol_model.load(config.ticker) is None:
        return (
            f"'{label}' needs the IntradayVolatility export for {config.ticker} at "
            f"{intraday_vol_model.model_path(config.ticker)}, which is missing or "
            "unreadable. Write it with FinNotebooks/IntradayVolatility/scripts/"
            "export_app_model.py, or run the flat day-range model."
        )
    return None


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

    `open_price` is the official opening print the day-range model is fed and
    the intraday envelope is centred on. The HighLow model rolls its daily bars
    up from minute bars, so it reads the first bar's open instead and needs no
    print; None then makes the plan fall back to that same bar.
    """
    if bundle.get("kind") == "highlow":
        return (
            _highlow().forecast_session(bundle, ticker, opening, today, key, secret),
            None,
        )
    dayrange = _dayrange()
    history = dayrange.daily_frame_from_bars(
        historical.fetch_daily_ohlc_bars(ticker, days=dayrange.DAILY_HISTORY_DAYS)
    )
    # Kept, not just passed on: under `intraday` the envelope is centred on the
    # session's open, so the same print the forecast was built from is needed
    # again on every bar of the day.
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

    Module-level rather than a method because both day-range consumers need it
    and they are not related by inheritance: `DayRangeTrader` here, and Apple
    Trader 2's `SessionForecaster`, which makes the same forecast only when some
    rule asks for it. `ticker` is the symbol the caller trades, and only matters on
    the re-fetch path, since `frame` is already the right symbol's bars.
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
#   take         -- below the sell level, in profit: sell `take_fraction`;
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


def momentum_words(read: dict) -> str:
    """One read in a log line's words."""
    return (
        f"momentum over the last {read['n']} bars is {read['mom']:+.3f} $/bar "
        f"({read['mom_class']}) and its change {read['change']:+.3f} $/bar "
        f"({read['change_class']}; neutral inside ±{read['band']:.3f}) — "
        f"{read['row'].prediction}"
    )


class DayRangeTrader(BaseTrader):
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

    Two settings make `H` less of a constant than the notebook's, and they are
    separate because they say different things:

    * `breach_update` -- a session that trades *through* the predicted high has
      falsified it, and the levels hanging off it with it, so the breached side
      moves. One-way: the forecast only ever widens (`_update_range`).
    * `level_source` -- what `H` is in the first place. `dayrange` is the
      predicted high itself. `intraday` reads the same forecast through
      IntradayVolatility's time-of-day shape, so the reference is the upper
      curve of the "predicted intraday range x day range" band at this minute:
      widest at the open, pulled in at midday, widening into the close
      (`_reference`). Not one-way -- that is the point of it, and the caveat.

    With both at their defaults for `off` and `dayrange` this is the notebook's
    rule exactly.

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
    half-way between the last buy and the bottom of the range (`_add`,
    `_after_fill`). The stop stays under the last actual fill, so a rung is
    only bought while it sits above that stop. Switched off it is the
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
      and sends a market order, which fills near that bar's close. On a bar
      that dipped to the level and recovered, the notebook buys at the level
      and this buys higher. That is a real cost and it runs one way, against
      the strategy; it is the price of the paper ledger being market-order
      only, not a modelling choice. Read a SimLab result against the notebook's
      with that in mind.
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
            # the same fallback `model_overlays._session_open_price` makes, so
            # the level and the band drawn behind it are centred alike. Only
            # read under `intraday`.
            "open_price": float(open_price or opening["open"].iloc[0]),
            "vol_shape": self._vol_shape(state),
            **forecast,
        }
        self._set_levels(opening.index[-1])
        prior = getattr(state, "apple_trader_levels", None) or {}
        if (
            prior.get("rows")
            and prior.get("ticker") == self.ticker
            and pd.Timestamp(prior.get("date")) == pd.Timestamp(today)
        ):
            # A restart mid-session: what the earlier run rested is history and
            # stays on the chart. This run's levels start at the bar it first
            # reads, not at 09:35 -- it rested nothing before it existed.
            self.plan["history"] = list(prior["rows"])
            if not switching:
                self._resume(state, prior.get("memory"))
        else:
            self._record_levels(state, opening.index[-1])

        warning = _dayrange().volume_scale_warning(tape)
        if warning:
            _log(state, {"type": "status", "text": f"Forecast caveat: {warning}"})
        _log(state, {"type": "analysis", "text": self._plan_summary()})
        return True

    def _opening_window(self, state: AppState, frame, want: int):
        return fetch_opening_window(state, frame, want, ticker=self.ticker)

    def _vol_shape(self, state: AppState) -> "dict | None":
        """IntradayVolatility's export for this symbol, once per session, or None.

        None whenever the reference does not read it, so a `dayrange` run never
        touches the file. A run that does need it has already been refused at
        launch if the file is missing (`level_source_error`); this reports the
        case where it disappeared between the two, and the reference then falls
        back to the flat predicted high rather than to nothing.
        """
        if self.config.level_source != LEVELS_INTRADAY:
            return None
        shape = intraday_vol_model.load(self.ticker)
        if shape is None:
            _log(
                state,
                {
                    "type": "error",
                    "text": (
                        f"The IntradayVolatility shape for {self.ticker} could not be "
                        "loaded, so today's levels rest on the flat predicted high "
                        "instead of following the time of day."
                    ),
                },
            )
        return shape

    # --- the sidebar, and the record the chart draws ------------------------

    def _adopt_form_model(self, state: AppState, frame, today) -> bool:
        """Switch to the model the sidebar now names, if it changed.

        Returns True when the session's forecast was re-made, which `run_cycle`
        needs to know: the new forecast is the new model's 9:35 one and has not
        yet been moved by what the session has printed since.

        The model is what the levels hang off, so a switch re-forecasts the day
        with it -- from the same opening window -- and rebuilds both levels.
        The distances come with it: the form re-seeds them with the new model's
        tuned pair, so the old model's numbers under the new forecast would be
        a pair nobody picked. So does the reference the model implies (the flat
        high, or the intraday curve). Everything else stays, as for a distance
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
        form = getattr(state, "apple_trader_config", None)
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
        if form.level_unit != config.level_unit:
            refusal = (
                f"its distances are counted in {form.unit_phrase} and this run counts in "
                f"{config.unit_phrase}; a new unit needs ▶ Start"
            )
        else:
            try:
                new = replace(
                    config, model_key=form.model_key, level_source=form.level_source,
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
        form = getattr(state, "apple_trader_config", None)
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
            f"{config.unit_phrase} below {self._ref_name}."
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
            if self.entry.get("runner"):
                row["stop"] = float(self.entry["price"])
            else:
                row["stop"] = self._stop_price()
        history = plan.setdefault("history", [])
        if history and history[-1]["t"] == row["t"]:
            history[-1] = row
        else:
            history.append(row)
        state.apple_trader_levels = {
            "ticker": self.ticker,
            "date": plan["date"],
            "config": self.config,
            "rows": history,
            "memory": self.memory(),
        }

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
        levels = getattr(state, "apple_trader_levels", None)
        if (
            not levels
            or self.plan is None
            or levels.get("ticker") != self.ticker
            or pd.Timestamp(levels.get("date")) != pd.Timestamp(self.plan["date"])
        ):
            return
        state.apple_trader_levels = {**levels, "memory": self.memory()}

    def _resume(self, state: AppState, memory: "dict | None") -> None:
        """Take back what an earlier run today knew (`memory`), on a restart.

        The entry is only a candidate: `run_cycle` drops it at once if the
        ledger it continues is flat, and keeps it only while shares are held."""
        if not memory:
            return
        stand_down = memory.get("stand_down")
        if stand_down and not self.plan.get("stand_down"):
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
        """The number the two distances are measured below, on this bar.

        Under `dayrange` it is the predicted high: one number, the same at 09:36
        and at 15:50, and the notebook's rule.

        Under `intraday` it is the upper curve of `intraday_vol_model.envelope`
        -- the same forecast stretched by IntradayVolatility's time-of-day shape,
        which is the "predicted intraday range x day range" overlay read as a
        level instead of as a decoration. The shape peaks at the open, so the
        reference *is* the predicted high at 09:30 and pulls in towards the
        opening price through the morning, bottoming out around a fifth of that
        distance at midday before opening back up into the close.

        What that means for the strategy, stated plainly because it is a real
        change and not a refinement:

        * The whole ladder descends through the morning and rises again into
          the last half hour. An entry needs a deeper dip to fill as the day
          quiets -- and the *target* descends too, so a position opened in the
          morning can be closed by a sell level that came down to it rather
          than by a price that went up to it. That is the claim the model makes
          (the day is no longer moving enough to reach the morning's target),
          and it is the opposite of the one-way ratchet `_update_range`
          applies, which is why the two are separate settings.
        * The band is anchored on the session's **open**, not on the price. By
          midday the reference sits within a fifth of the forecast distance
          from the opening print, so a day that has trended well away from it
          leaves the levels behind: a trending-down day keeps the buy level
          above the price and re-arms the entry until the stop ends the
          session's trading. That is the same dependence on the stop the breach
          policies have, for a different reason.
        * It also damps `breach_update`. A dollar added to the predicted high
          moves this reference by the shape at that minute -- about a fifth of
          a dollar midday -- so the two settings are much less than additive.

        Neither of these has been swept, and the shipped buy/sell distances were
        swept against a reference that does not move.

        Falls back to the predicted high if the envelope cannot be built -- the
        shape or the session's open missing. `config_error` refuses a run whose
        symbol has no shape at all, so this covers the narrower case of a
        session with no opening print, where the alternative would be no levels.
        """
        plan = self.plan
        if self.config.level_source != LEVELS_INTRADAY:
            return float(plan["pred_high"])
        shape, open_price = plan.get("vol_shape"), plan.get("open_price")
        if shape is None or not open_price:
            return float(plan["pred_high"])
        upper, _ = intraday_vol_model.envelope_at(
            shape, open_price, plan["pred_high"], plan["pred_low"],
            self._minutes_from_open(ts),
        )
        return upper

    @property
    def _ref_name(self) -> str:
        """What the log calls the reference -- "H" is ambiguous once it moves."""
        return (
            "the predicted high"
            if self.config.level_source != LEVELS_INTRADAY
            else "the intraday range's upper curve"
        )

    @staticmethod
    def _minutes_from_open(ts) -> float:
        """Minutes from 09:30 to this bar, the index the shape is a function of."""
        stamp = pd.Timestamp(ts)
        open_ts = stamp.normalize() + pd.Timedelta(
            hours=market_hours.MARKET_OPEN.hour, minutes=market_hours.MARKET_OPEN.minute
        )
        return (stamp - open_ts).total_seconds() / 60.0

    def _set_levels(self, ts) -> None:
        """Rebuild the two resting levels from the reference this bar has.

        Called when the plan is made, every time `_update_range` moves the
        forecast, and -- because the reference can be a function of the clock --
        once per closed bar. That is the whole of "the levels follow the
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

        Each add rests half-way between the last buy and the bottom of the
        range, `reference - 1 x unit`: the predicted low under "pred_range",
        one ADR under the reference under "adr". So the gap left to the bottom
        halves with every fill -- 0.40, 0.70, 0.85, 0.925 for AAPL's 0.40 --
        and no add is ever placed under it.

        Counted from `buy_k` rather than stored per rung, so a sidebar edit to
        the buy distance moves the whole ladder with it, the same way it moves
        the first buy.
        """
        buy_k = float(self.config.buy_k)
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
        decided to take money off this position -- and a buy distance at or
        under the bottom of the range leaves nowhere lower to go.
        """
        entry = self.entry or {}
        return bool(
            self.config.can_scale_in
            and entry.get("can_add")
            and entry.get("fills")
            and not entry.get("runner")
            and float(self.config.buy_k) < 1.0
        )

    def _momentum_read(self, frame) -> "dict | None":
        """This bar's read of the behaviour table (`momentum_read`), or None."""
        return momentum_read(
            frame["close"], self.config.momentum_confirmation_bars, self.minute_move
        )

    def _momentum_unknown(self, frame) -> str:
        """Why the confirmation cannot read this bar -- for the log line that
        says a level was reached and nothing was done about it."""
        n = int(self.config.momentum_confirmation_bars)
        if self.minute_move is None or not float(self.minute_move) > 0:
            return (
                f"{self.ticker}'s mean one-minute move over last week is not known yet, "
                "so momentum has no neutral band to be read against"
            )
        return (
            f"momentum over {n} bars needs {n + 2} bars of the session and there "
            f"are {len(frame)}"
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
            config.can_scale_in and float(config.buy_k) < 1.0 and shares > 0
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
            text = (
                f"{held} Next buy at ${plan['buy_level']:,.2f} ({rung_k:g} × "
                f"{config.unit_phrase} under {self._ref_name}, half-way to the "
                f"${float(plan['reference']) - level_unit(config, plan):,.2f} bottom of "
                f"the range): ${cash:,.2f} cash pays for {shares:g} more share(s).{stop}"
            )
        else:
            if float(config.buy_k) >= 1.0:
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
        reasoning = (
            f"The bar traded down to ${float(bar['low']):,.2f}, at or through the "
            f"${self.plan['buy_level']:,.2f} next buy level — {rung_k:g} × the "
            f"{self.config.unit_phrase} below {self._ref_name}, half-way from the last "
            f"buy to the bottom of the range. Adding to the {position:g} sh bought at an "
            f"average ${float(before['price']):,.2f}."
        )
        if not self.buy(state, tracker, float(bar["close"]), reasoning):
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
        session_high = float(frame["high"].max())
        session_low = float(frame["low"].min())
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
           sell level, not to hand back the gain the take already banked.
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

        if entry.get("runner") and low <= entry_price:
            return position, (
                f"Breakeven: the runner the momentum take left traded back down to "
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
                f"({config.sell_k:g} × {config.unit_phrase} under {self._ref_name}, "
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

        Fires when the position is in profit by at least
        `take_min_gain_fraction` of the predicted gain (`target_gain_k` level
        units, the same yardstick as the stop) and the momentum confirmation's
        table says take (`MOMENTUM_TABLE`): momentum over the look-back is
        negative and its change neutral or negative -- the price is still
        dropping, or dropping faster. A drop that is slowing (negative momentum,
        positive change) is left alone. A legacy record reads the rule it was
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
        position: a runner is never trimmed again.

        The momentum is recomputed over the session on each call rather than
        tracked bar by bar, so a cycle that missed a bar still sees the turn.
        Only reached with a profitable, untrimmed position and the rule on.
        """
        config, plan = self.config, self.plan
        entry = self.entry or {}
        if (
            not config.has_take
            or entry.get("runner")
            or not entry_price
            or price <= entry_price
        ):
            return None
        # Not before the trade has banked its share of what it is playing for:
        # short of that a negative read is noise around the fill, which the
        # stop is there for. Read at the current unit, like the runner test.
        unit = level_unit(config, plan)
        if price - entry_price < config.take_min_gain_fraction * config.target_gain_k * unit:
            return None

        since = entry.get("ts", frame.index[-1])
        if config.momentum_confirmation_bars:
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
        fade = (
            f"Momentum take: {why}, with the "
            f"price at ${price:,.2f} — above the ${entry_price:,.2f} fill but short of the "
            f"${plan['sell_level']:,.2f} sell level"
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
        return quantity, (
            f"{fade}. Banking {quantity:g} of {position:g} shares at market "
            f"({pnl_pct:+.2f}%). The sell level is still {to_target} above the fill — at "
            f"least the {config.hold_min_gain_k:g} × {config.unit_phrase} worth waiting for — so the other "
            f"{position - quantity:g} ride on to it or the closing flatten, and are sold if "
            "the price comes back to the fill."
        ), self.EXIT_TAKE

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
            state, tracker, float(bar["close"]), self._entry_reasoning(bar)
        )
        if bought:
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

    def _entry_reasoning(self, bar) -> str:
        plan, config = self.plan, self.config
        exits = [f"a resting sell at ${plan['sell_level']:,.2f}"]
        risk = stop_distance(config, stop_unit(config, plan))
        if risk:
            exits.append(
                f"a stop {stop_phrase(config)} (${risk:,.2f}) under the fill"
            )
        if config.has_take:
            exits.append(f"a momentum take if the move gives out short of it ({fade_phrase(config)})")
        return (
            f"The bar traded down to ${float(bar['low']):,.2f}, at or through the "
            f"${plan['buy_level']:,.2f} buy level — {config.buy_k:g} × the "
            f"{config.unit_phrase} (${plan['level_unit']:,.2f}) below {self._ref_name} at "
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
                "stopped out",
                "The day did not go the way the forecast said, and the buy level is by now "
                "usually just above the price — re-arming it would buy the same slide "
                "again, one stop lower each time.",
            )

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
        if self.config.level_source == LEVELS_INTRADAY:
            rests = (
                f"The levels rest under the intraday range's upper curve — that forecast "
                f"stretched by IntradayVolatility's time-of-day shape around the "
                f"${plan['open_price']:,.2f} open — so they follow the clock: "
                f"${plan['reference']:,.2f} now, pulling in towards the open through the "
                "morning and widening again into the close."
            )
        else:
            rests = "The levels rest under the predicted high, which is flat all session."
        return (
            f"{self.ticker} forecast for the session, from the first "
            f"{plan['opening_end']:%H:%M} minutes: high ${plan['pred_high']:,.2f}, low "
            f"${plan['pred_low']:,.2f} (yesterday's average ${plan['prev_avg']:,.2f}, "
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
        if self.config.level_source == LEVELS_INTRADAY:
            # The two levels above are this minute's, not the day's, and without
            # the reference there is nothing in the line that says so.
            parts.append(f"ref ${plan['reference']:,.2f} (intraday)")
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
            if self.entry.get("runner"):
                parts.append(f"runner, out at ${entry_price:,.2f}")
            elif self._stop_price() is not None:
                parts.append(f"stop ${self._stop_price():,.2f}")
        elif plan.get("stand_down"):
            parts.append(f"{plan['stand_down']}, no new entries today")
        return " · ".join(parts)


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
    accounted for by construction: the breach update ratchets the forecast, the
    intraday source re-reads it each minute, and a change to either shows up in
    the picture the same day it shows up in the trades.

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
        "vol_shape": (
            intraday_vol_model.load(config.ticker)
            if config.level_source == LEVELS_INTRADAY
            else None
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


def build_trader(config: AppleTraderConfig, bundle: "dict | None" = None):
    """The state machine this configuration's model calls for.

    The one place a model's strategy turns into an object. Every launch path --
    the live loop below, SimLab's `rule_agents._build_apple` -- goes through
    here, so a second strategy is added in one place rather than in whichever
    entry points were remembered. The returned object exposes
    `run_cycle(bundle, state, tracker)` and nothing else a caller needs.
    """
    return DayRangeTrader(config)


# --- the loop ---------------------------------------------------------------

def _breach_side(config: AppleTraderConfig) -> str:
    """What a breach moves under this policy, for the log lines that say so."""
    if config.breach_update == BREACH_SHIFT or (
        config.keep_width and config.breach_update == BREACH_BROWNIAN
    ):
        return "both sides of it"
    return "the breached side"


def _armed_summary(config: AppleTraderConfig, model, bundle: dict) -> str:
    """The one line the log opens a run with: which model, and what it will do."""
    metadata = bundle.get("metadata") or {}
    # TimeToChange3 files its held-out score as `test_metrics_ensemble`, HighLow
    # as `test_metrics`; the dollar error is `mae_usd_mean` in both.
    scores = metadata.get("test_metrics_ensemble") or metadata.get("test_metrics") or {}
    mae = scores.get("mae_usd_mean")
    quality = f", held-out mean error ${mae:.2f}" if mae else ""
    exits = []
    if config.has_stop:
        exits.append(
            f"a stop {stop_phrase(config)} under the fill, after which it buys nothing "
            "more that day"
        )
    if config.has_take:
        exits.append(
            f"a {config.take_fraction:.0%} take in profit"
            + (
                f" (at least {config.take_min_gain_fraction:g} × the predicted gain)"
                if config.take_min_gain_fraction else ""
            )
            + " once momentum gives out "
            f"({fade_phrase(config)}), the rest kept for the sell level only if "
            f"it is {config.hold_min_gain_k:g} × {config.unit_phrase} or more above the fill and sold if the "
            "price comes back to it"
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
    reference = (
        "the predicted high"
        if config.level_source != LEVELS_INTRADAY
        else (
            "the upper curve of that range stretched by IntradayVolatility's time-of-day "
            "shape, so both levels move with the clock"
        )
    )
    return (
        f"Apple Trader armed on {model.label} (fitted "
        f"{bundle.get('trained_at', 'unknown')}{quality}): at 9:35 it forecasts where "
        f"today's {config.ticker} high and low will land, then rests a buy "
        f"{config.buy_k:g} × the {config.unit_phrase} below {reference} and a sell "
        f"{config.sell_k:g} below it, until the closing flatten.{breach}{no_fall}{managed}{breaker}"
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
        publish = getattr(trader, "publish_memory", None)
        if publish is not None:
            publish(state)
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
