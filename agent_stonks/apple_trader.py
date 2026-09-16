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
from dataclasses import dataclass
from datetime import timedelta
from typing import Optional

import pandas as pd

from . import agent as agent_mod
from . import (
    apple_models, historical, intraday_vol_model, market_hours, momentum_regime,
    rule_agent,
)
from .agent import stop_agent
from .rule_agent import BaseTrader
from .state import append_agent_log as _log
from .config import (
    APPLE_TRADER_BREACH_UPDATE,
    APPLE_TRADER_BUY_K,
    APPLE_TRADER_CYCLE_SEC,
    APPLE_TRADER_DAYRANGE_LEVELS,
    APPLE_TRADER_FLATTEN_BEFORE_CLOSE_MIN,
    APPLE_TRADER_HOLD_MIN_GAIN_K,
    APPLE_TRADER_LEVEL_SOURCE,
    APPLE_TRADER_MODEL,
    APPLE_TRADER_MOMENTUM_DROP,
    APPLE_TRADER_POSITION_PCT,
    APPLE_TRADER_SELL_K,
    APPLE_TRADER_STOP_K,
    APPLE_TRADER_TAKE_FRACTION,
    BREACH_BROWNIAN,
    BREACH_LABELS,
    BREACH_OFF,
    BREACH_POLICIES,
    LEVELS_DAYRANGE,
    LEVELS_INTRADAY,
    LEVEL_SOURCES,
    LEVEL_SOURCE_LABELS,
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


def dayrange_levels(ticker: str) -> "tuple[float, float]":
    """The day-range `(buy_k, sell_k)` a run on this symbol starts from.

    Per instrument, from re-running notebook 05's grid over every session with
    a forecast -- `config.APPLE_TRADER_DAYRANGE_LEVELS` carries the table and
    how far each pair deserves trust. A symbol never swept gets the notebook's
    specified pair.
    """
    return APPLE_TRADER_DAYRANGE_LEVELS.get(
        (ticker or DEFAULT_TICKER).strip().upper(),
        (APPLE_TRADER_BUY_K, APPLE_TRADER_SELL_K),
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
    # have gone the wrong way. 0 switches the stop off.
    stop_k: float = APPLE_TRADER_STOP_K
    # How far (in momentum sigmas) the score has to fall from its best since the
    # entry, with the position in profit, to take gains short of the sell level.
    # 0 switches the take off, and with it the runner and its breakeven.
    momentum_drop: float = APPLE_TRADER_MOMENTUM_DROP
    # The share of the position that take sells when a runner is kept.
    take_fraction: float = APPLE_TRADER_TAKE_FRACTION
    # The gain still left to the sell level, in ADRs above the fill, that is
    # worth keeping a runner for. Short of it the take sells everything.
    hold_min_gain_k: float = APPLE_TRADER_HOLD_MIN_GAIN_K
    # What to do when the session trades through the forecast the two levels
    # are built on -- one of `dayrange_model.BREACH_POLICIES`. "off" is the
    # notebook's rule (one forecast, held all day); the other two move the
    # breached side and the levels with it. See `_update_range`.
    breach_update: str = APPLE_TRADER_BREACH_UPDATE
    # Which number the two distances above are measured below -- one of
    # `config.LEVEL_SOURCES`. "dayrange" is the predicted high itself;
    # "intraday" is that forecast read through IntradayVolatility's
    # time-of-day shape, so the reference moves with the clock. See
    # `_set_levels`, and `config_error` for what it requires.
    level_source: str = APPLE_TRADER_LEVEL_SOURCE

    def __post_init__(self) -> None:
        for name in ("stop_k", "momentum_drop", "hold_min_gain_k"):
            if getattr(self, name) < 0:
                raise ValueError(
                    f"{name} {getattr(self, name)!r} is a distance and cannot be negative "
                    "(0 is how the stop or the momentum take is switched off)"
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
        self.ticker = (self.ticker or DEFAULT_TICKER).strip().upper()
        # Resolved per field, so a config that names only one level still
        # gets the instrument's default for the other.
        default_buy, default_sell = dayrange_levels(self.ticker)
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

    The managed exit is written only while switched on -- the stop when
    `stop_k` is set, the take and its runner threshold when `momentum_drop` is
    -- so a config with both off signs exactly as a run recorded before the
    exit existed, and `take_fraction` never splits two runs that cannot differ.
    The intraday update follows the same rule for the same reason: it appears
    only when it is not "off", so every record written before it existed (which
    replays as "off") keeps the signature it was filed under.
    """
    c = config or AppleTraderConfig()
    exits = ""
    if c.stop_k:
        exits += f",stop=E-{c.stop_k:g}A"
    if c.momentum_drop:
        exits += (
            f",take={c.take_fraction * 100:g}%@mom-{c.momentum_drop:g},"
            f"runner>={c.hold_min_gain_k:g}A"
        )
    breach = "" if c.breach_update == BREACH_OFF else f",breach={c.breach_update}"
    levels = "" if c.level_source == LEVELS_DAYRANGE else f",levels={c.level_source}"
    # "H" in the two distances is whatever `level_source` says it is, which is
    # why that token is next to them rather than at the end.
    return (
        f"{c.model_key}_{c.ticker}(buy=H-{c.buy_k:g}A,sell=H-{c.sell_k:g}A{levels},"
        f"size={c.position_pct:g}%{exits}{breach})"
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
            f"{config.ticker}. Use the predicted high, or pick another instrument."
        )
    if intraday_vol_model.load(config.ticker) is None:
        return (
            f"'{label}' needs the IntradayVolatility export for {config.ticker} at "
            f"{intraday_vol_model.model_path(config.ticker)}, which is missing or "
            "unreadable. Write it with FinNotebooks/IntradayVolatility/scripts/"
            "export_app_model.py, or use the predicted high."
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


def fetch_opening_window(state: AppState, frame, want: int, ticker: str = DEFAULT_TICKER):
    """The first `want` regular-session bars of today, or a clear failure.

    The live buffer normally holds them -- it keeps the whole session -- but an
    agent started after 9:35 has a buffer that begins wherever the stream did,
    and `frame.iloc[:want]` would then hand the model five bars from the middle
    of the day as though they were the open. Rather than produce a confident
    forecast off the wrong five minutes, the 09:30 window is re-fetched, exactly
    as `agent._opening_range_for` does for the same reason and through the same
    (simulation-patched) call.

    Module-level rather than a method because both day-range consumers need it
    and they are not related by inheritance: `DayRangeTrader` here, and Apple
    Trader 2's `SessionForecaster`, which makes the same forecast only when some
    rule asks for it. `ticker` is the symbol the caller trades, and only matters on
    the re-fetch path, since `frame` is already the right symbol's bars.
    """
    first = frame.iloc[:want]
    if float(first["minutes_from_open"].iloc[0]) < 1.0:
        return first

    open_utc = market_hours.session_open()
    window = []
    if open_utc is not None and state.api_key and state.api_secret:
        try:
            window = agent_mod.fetch_bars_window(
                ticker, "1Min", open_utc,
                open_utc + timedelta(minutes=want),
                state.api_key, state.api_secret, state.feed,
            )
        except Exception:
            window = []
    recovered = momentum_regime.frame_from_bars(window)
    if len(recovered) < want or float(recovered["minutes_from_open"].iloc[0]) >= 1.0:
        raise ValueError(
            f"the first {want} minutes of the session are not in the bar buffer "
            f"(it starts at {frame.index[0]:%H:%M}) and could not be re-fetched; "
            "the forecast is built on the 09:30 window and cannot be made without it."
        )
    return recovered.iloc[:want]


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
    against it; `stop_k = momentum_drop = 0` switches it off and gives notebook
    05's rule back.

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
            self.entry = {"price": float(last["close"]), "bars": 0, "ts": ts}
        if position <= 0:
            self.entry = None
        if fresh_bar and self.entry is not None:
            self.entry["bars"] += 1

        # Before the read, so the line below quotes the levels this bar is
        # actually about to be measured against rather than last bar's. The
        # forecast moves first and the levels are then rebuilt from it at this
        # minute, which is also what re-reads a reference that follows the clock.
        if fresh_bar and self.plan is not None:
            self._update_range(state, frame, ts)
            self._set_levels(ts)

        _log(state, {"type": "analysis", "text": self._read_summary(last, ts, position)})

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
            return "hold"

        # A stop means the day did not go the way the forecast said, and the
        # buy level is by then usually just above the price -- re-arming it
        # would buy the same slide again, one stop lower each time.
        if self.plan.get("stopped_out"):
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
        self, bundle: dict, state: AppState, frame, today, want: int
    ) -> bool:
        """Forecast the day and set the two levels. False if it cannot be done.

        Every failure here is fatal for the session rather than for the bar --
        a daily history that is too short at 9:35 is still too short at 14:00
        -- so it is recorded in `self.blocked` and reported once.
        """
        try:
            opening = self._opening_window(state, frame, want)
            dayrange = _dayrange()
            history = dayrange.daily_frame_from_bars(
                historical.fetch_daily_ohlc_bars(
                    self.ticker, days=dayrange.DAILY_HISTORY_DAYS
                )
            )
            # Kept, not just passed on: under `intraday` the envelope is centred
            # on the session's open, so the same print the forecast was built
            # from is needed again on every bar of the day.
            open_price = historical.fetch_session_open(self.ticker)
            forecast = dayrange.forecast_session(
                bundle, history, opening, today, open_price=open_price,
            )
        except Exception as exc:
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

        warning = _dayrange().volume_scale_warning(getattr(state, "feed", None))
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
          above the price and re-arms the entry until `stop_k` ends the
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
        subtracted from the same reference, so the sell level sits above the buy
        level at every minute however the reference moves.
        """
        plan, config = self.plan, self.config
        adr = plan["adr14_abs"]
        reference = self._reference(ts)
        plan["reference"] = reference
        plan["buy_level"] = reference - config.buy_k * adr
        plan["sell_level"] = reference - config.sell_k * adr

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
        trades, and it is why `stop_k` matters more under these policies than
        under "off": a day that ratchets the high all afternoon will keep
        re-arming the entry until a stop ends the session's trading.
        """
        config, plan = self.config, self.plan
        if config.breach_update == BREACH_OFF or plan is None:
            return
        if ts <= plan["opening_end"]:
            return

        dayrange = _dayrange()
        # Re-based to this bar's minute first, so that under a reference which
        # moves with the clock the "was" in the log line is this minute's level
        # without the breach rather than last minute's with it -- the breach's
        # own effect, which is what the line is about.
        self._set_levels(ts)
        before = {k: float(plan[k]) for k in
                  ("pred_high", "pred_low", "buy_level", "sell_level")}
        high, low = dayrange.updated_range(
            plan,
            session_high=float(frame["high"].max()),
            session_low=float(frame["low"].min()),
            minutes_left=dayrange.minutes_left_at(ts),
            policy=config.breach_update,
        )
        if high == before["pred_high"] and low == before["pred_low"]:
            return

        plan["pred_high"], plan["pred_low"] = high, low
        plan["range_updates"] = int(plan.get("range_updates", 0)) + 1
        self._set_levels(ts)
        _log(state, {"type": "analysis", "text": self._range_summary(ts, before)})

    def _range_summary(self, ts, before: dict) -> str:
        """The one line an update writes: what the tape did, and what moved.

        The two levels are built from the predicted *high* alone, so a breach of
        the low moves the forecast and nothing else. The line says which it was
        rather than claiming a rebuild either way -- a log that reported levels
        that had not changed would be the reader's problem on every grinding
        session, which is exactly when it matters.
        """
        plan, config = self.plan, self.config
        dayrange = _dayrange()
        moved = []
        if plan["pred_high"] != before["pred_high"]:
            moved.append(
                f"the session has traded up through the ${before['pred_high']:,.2f} "
                "predicted high"
            )
        if plan["pred_low"] != before["pred_low"]:
            moved.append(
                f"it has traded down through the ${before['pred_low']:,.2f} predicted low"
            )
        if config.breach_update == BREACH_BROWNIAN:
            left = dayrange.minutes_left_at(ts)
            reach = dayrange.brownian_reach(plan["adr14_abs"], left)
            how = (
                f"the extreme so far, extended by the ${reach:,.2f} a driftless walk with "
                f"this ADR's volatility is still expected to add over the {left:.0f} min left"
            )
        else:
            how = "the extreme so far"
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
    EXIT_FLATTEN = "flatten"
    EXIT_TAKE = "momentum_take"

    def _exit(self, frame, position: float) -> "tuple[float, str, str] | None":
        """What to sell on this bar, why, and under which rule -- or None to hold.

        Returns `(quantity, reasoning, kind)`. The checks run in a fixed order
        and the first that applies takes the bar:

        1. **breakeven** -- a runner (what a momentum take left behind) is sold
           once a bar's low comes back to the fill. It was kept to wait for the
           sell level, not to hand back the gain the take already banked.
        2. **stop** -- a bar's low at `fill - stop_k x ADR`: the day went the
           other way from the forecast. Everything is sold, and `run_cycle`
           takes no new entry for the rest of the session.
        3. **target** -- a bar's high at the sell level. Everything.
        4. **flatten** -- the closing bell. Everything.
        5. **momentum take** -- see `_momentum_take`.

        The price rules trigger on a touch -- the low for the two stops, the
        high for the target -- like the resting levels the entry already
        models; the momentum take reads the close. The stops come before the
        target because a bar wide enough to touch both says nothing about which
        came first, so it is read the careful way.

        With `stop_k` and `momentum_drop` both 0 only the target and the
        flatten are left, which is notebook 05's rule as specified.
        """
        config, plan = self.config, self.plan
        bar = frame.iloc[-1]
        entry = self.entry or {}
        entry_price = entry.get("price") or 0.0
        price, low, high = float(bar["close"]), float(bar["low"]), float(bar["high"])
        pnl_pct = (price / entry_price - 1) * 100 if entry_price else 0.0

        if entry.get("runner") and low <= entry_price:
            return position, (
                f"Breakeven: the runner the momentum take left traded back down to "
                f"${low:,.2f}, at or under the ${entry_price:,.2f} fill. It was kept for the "
                f"${plan['sell_level']:,.2f} sell level, not to give the gain back, so the "
                f"rest is sold at market ({pnl_pct:+.2f}%)."
            ), self.EXIT_BREAKEVEN

        if config.stop_k and entry_price:
            stop = entry_price - config.stop_k * plan["adr14_abs"]
            if low <= stop:
                return position, (
                    f"Stop loss: the bar traded down to ${low:,.2f}, at or through the "
                    f"${stop:,.2f} stop ({config.stop_k:g} × ADR under the "
                    f"${entry_price:,.2f} fill). The day is not going the way the forecast "
                    f"said, so the position is closed at market ({pnl_pct:+.2f}%) and "
                    "nothing more is bought this session."
                ), self.EXIT_STOP

        if high >= plan["sell_level"]:
            return position, (
                f"Target: the bar traded up to ${high:,.2f}, at or through the "
                f"${plan['sell_level']:,.2f} sell level "
                f"({config.sell_k:g} × ADR under {self._ref_name}, "
                f"${plan['reference']:,.2f}). Selling at market ({pnl_pct:+.2f}%)."
            ), self.EXIT_TARGET

        if self.closing_soon():
            to_close = market_hours.seconds_to_close() or 0.0
            return position, (
                f"Session ends in {to_close / 60:.0f} min and the day never came back up to "
                f"${plan['sell_level']:,.2f}. The forecast is a statement about today "
                f"only, so the position is flattened rather than carried overnight "
                f"({pnl_pct:+.2f}%)."
            ), self.EXIT_FLATTEN

        return self._momentum_take(frame, position, entry_price, price)

    def _momentum_take(
        self, frame, position: float, entry_price: float, price: float
    ) -> "tuple[float, str, str] | None":
        """Bank gains short of the target when the move carrying them fades.

        Fires when the position is in profit and the momentum score has fallen
        `momentum_drop` sigmas from its best since the entry bar. The peak is
        taken over this position's bars only: a morning surge before the entry
        is not a move this trade was riding.

        What it sells depends on how much the forecast still promises. If the
        sell level is `hold_min_gain_k x ADR` or more above the fill,
        `take_fraction` of the shares go and the rest is kept as a runner --
        left to the sell level, the flatten, or the breakeven. Short of that the
        target is too close to be worth the wait and everything goes. Once per
        position: a runner is never trimmed again.

        The score is recomputed over the session on each call rather than
        tracked bar by bar, so a cycle that missed a bar still sees its peak.
        Only reached with a profitable, untrimmed position and the rule on.
        """
        config, plan = self.config, self.plan
        entry = self.entry or {}
        if (
            not config.momentum_drop
            or entry.get("runner")
            or not entry_price
            or price <= entry_price
        ):
            return None

        mom = momentum_regime.compute_momentum(frame)["mom"]
        now = float(mom.iloc[-1])
        since = mom[frame.index >= entry.get("ts", frame.index[-1])].dropna()
        if math.isnan(now) or not len(since):
            return None
        peak = float(since.max())
        if peak - now < config.momentum_drop:
            return None

        adr = plan["adr14_abs"]
        left = plan["sell_level"] - entry_price
        pnl_pct = (price / entry_price - 1) * 100
        fade = (
            f"Momentum take: the momentum score has fallen from {peak:+.2f}σ, its best since "
            f"the entry, to {now:+.2f}σ ({config.momentum_drop:g}σ is the trigger), with the "
            f"price at ${price:,.2f} — above the ${entry_price:,.2f} fill but short of the "
            f"${plan['sell_level']:,.2f} sell level"
        )
        to_target = f"${left:,.2f} ({left / adr:.2f} × ADR)"

        if left < config.hold_min_gain_k * adr:
            return position, (
                f"{fade}. The sell level is only {to_target} above the fill, under the "
                f"{config.hold_min_gain_k:g} × ADR worth keeping a runner for, so the whole "
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
            f"least the {config.hold_min_gain_k:g} × ADR worth waiting for — so the other "
            f"{position - quantity:g} ride on to it or the closing flatten, and are sold if "
            "the price comes back to the fill."
        ), self.EXIT_TAKE

    # --- orders ------------------------------------------------------------

    def _buy(self, state: AppState, tracker: DecisionTracker, bar) -> bool:
        bought = self.buy(
            state, tracker, float(bar["close"]), self._entry_reasoning(bar)
        )
        if bought:
            # Where the momentum take starts looking for this position's peak.
            self.entry["ts"] = bar.name
        return bought

    def _entry_reasoning(self, bar) -> str:
        plan, config = self.plan, self.config
        exits = [f"a resting sell at ${plan['sell_level']:,.2f}"]
        if config.stop_k:
            exits.append(
                f"a stop {config.stop_k:g} × ADR (${config.stop_k * plan['adr14_abs']:,.2f}) "
                "under the fill"
            )
        if config.momentum_drop:
            exits.append(
                f"a momentum take if the move fades {config.momentum_drop:g}σ short of it"
            )
        return (
            f"The bar traded down to ${float(bar['low']):,.2f}, at or through the "
            f"${plan['buy_level']:,.2f} buy level — {config.buy_k:g} average daily "
            f"ranges (${plan['adr14_abs']:,.2f} each) below {self._ref_name} at "
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
        entry = self.entry
        decision = self.sell(state, tracker, quantity, reasoning, log_extra={"exit": kind})
        if decision.status != "filled":
            return
        if kind == self.EXIT_STOP:
            self.plan["stopped_out"] = True
        if entry is not None and tracker.position_for(self.ticker) > 0:
            # `sell` forgets the entry on any fill, but what is left is still this
            # position -- same fill, same peak. After a momentum take it is the
            # runner; after any other exit that did not fill in full it is
            # whatever it was.
            self.entry = {**entry, "runner": entry.get("runner") or kind == self.EXIT_TAKE}

    # --- logging -----------------------------------------------------------

    def _plan_summary(self) -> str:
        plan = self.plan
        held = (
            "Both are held all day."
            if self.config.breach_update == BREACH_OFF
            else (
                f"If the session trades outside that range the breached side is moved "
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
            f"${plan['buy_level']:,.2f} (ref − {self.config.buy_k:g} × ADR), sell at "
            f"${plan['sell_level']:,.2f} (ref − {self.config.sell_k:g} × ADR). {held}"
        )

    def _read_summary(self, bar, ts, position: float) -> str:
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
        if position > 0 and self.entry:
            entry_price = self.entry["price"]
            pnl = (price / entry_price - 1) * 100 if entry_price else 0.0
            parts.append(
                f"long {position:g} sh @ ${entry_price:,.2f} ({pnl:+.2f}%), "
                f"{self.entry['bars']} bars"
            )
            if self.entry.get("runner"):
                parts.append(f"runner, out at ${entry_price:,.2f}")
            elif self.config.stop_k:
                stop = entry_price - self.config.stop_k * plan["adr14_abs"]
                parts.append(f"stop ${stop:,.2f}")
        elif plan.get("stopped_out"):
            parts.append("stopped out, no new entries today")
        return " · ".join(parts)


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

def _armed_summary(config: AppleTraderConfig, model, bundle: dict) -> str:
    """The one line the log opens a run with: which model, and what it will do."""
    metadata = bundle.get("metadata") or {}
    mae = (metadata.get("test_metrics_ensemble") or {}).get("mae_usd_mean")
    quality = f", held-out mean error ${mae:.2f}" if mae else ""
    exits = []
    if config.stop_k:
        exits.append(
            f"a stop {config.stop_k:g} ADR under the fill, after which it buys nothing more "
            "that day"
        )
    if config.momentum_drop:
        exits.append(
            f"a {config.take_fraction:.0%} take once momentum fades "
            f"{config.momentum_drop:g}σ in profit, the rest kept for the sell level only if "
            f"it is {config.hold_min_gain_k:g} ADR or more above the fill and sold if the "
            "price comes back to it"
        )
    managed = f" The exit adds {'; and '.join(exits)}." if exits else ""
    breach = (
        ""
        if config.breach_update == BREACH_OFF
        else (
            f" A session that trades outside the forecast moves the breached side "
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
        f"{config.buy_k:g} average daily ranges below {reference} and a sell "
        f"{config.sell_k:g} below it, until the closing flatten.{breach}{managed}"
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
    rule_agent.run_loop(
        state, tracker,
        lambda: trader.run_cycle(bundle, state, tracker),
        stop_event, cycle_sec, "Apple Trader",
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
