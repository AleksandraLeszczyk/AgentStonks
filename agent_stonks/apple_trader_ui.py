"""The Streamlit form Apple Trader is configured in, for both apps.

One module rather than one per app, for the reason `apple_rules_ui` gives about
the rule builder: the live dashboard and SimLab both offer this agent, and two
copies of a form drift the first time a range is widened or a knob is added.
They had already been forked once -- `ui.py` and `simlab/app.py` each carried
their own copy of these functions -- and the copies still agreed on every
range, step and format, which is exactly the state in which merging them is
cheap.

Structure here, wording from the caller
---------------------------------------
What is shared is the *shape* of the form: which knobs exist, what they are
allowed to be, how the widget keys are built, and the config that comes out.
Drift there is a bug -- a range SimLab will sweep and the live app will refuse.

What is not shared is the prose, because the two audiences differ and the
difference is deliberate. SimLab's help text says which knob is worth sweeping
and how a setting lands in Results; the dashboard's says what the setting will
do to a run that is about to start with real money's worth of paper behind it.
Flattening those into one voice would lose something both apps were written to
say, so each passes its own `FormCopy` and this module never invents a sentence
of its own.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import streamlit as st

from . import apple_models
from .apple_trader import AppleTraderConfig, dayrange_levels, min_win_for
from .config import (
    BREACH_EXTREME,
    BREACH_LABELS,
    BREACH_OFF,
    BREACH_POLICIES,
    LEVEL_UNITS,
    LEVEL_UNIT_LABELS,
    UNIT_ADR,
    UNIT_PRED_RANGE,
)


@dataclass(frozen=True)
class FormCopy:
    """One app's half of the form: its widget-key namespace and its wording.

    `help` and `intro` are keyed by field/strategy name rather than being
    separate attributes so that adding a knob needs no change here -- a missing
    key renders a knob with no help text, which is a thin form rather than a
    crash.
    """

    #: Namespaces every widget key, so both apps can render this in one
    #: process (and SimLab can render several setups) without colliding.
    prefix: str
    #: Appended to an instrument the app cannot currently run -- "not streamed"
    #: live, "not in the datasets" in SimLab.
    unavailable_suffix: str
    instrument_help: str
    model_help: str
    #: Leading caption per strategy key, and the trailing note under it.
    intro: "dict[str, str]" = field(default_factory=dict)
    outro: "dict[str, str]" = field(default_factory=dict)
    #: Help text per knob.
    help: "dict[str, str]" = field(default_factory=dict)

    def key(self, name: str) -> str:
        return f"{self.prefix}_{name}"


def params(symbols: "list[str] | None", copy: FormCopy) -> AppleTraderConfig:
    """Apple Trader's instrument and tunables, as one config.

    The instrument comes first because it decides which models exist, and the
    model then decides which rules apply.
    """
    defaults = AppleTraderConfig()
    ticker = instrument_row(defaults, symbols, copy)
    keys = apple_models.keys_for(ticker)
    model_key = str(
        st.selectbox(
            "Model",
            keys,
            index=keys.index(defaults.model_key) if defaults.model_key in keys else 0,
            format_func=model_label,
            # Scoped to the instrument: the models on offer change with it, and
            # a widget holding one that is no longer an option would be a stale
            # selection rather than a choice.
            key=f"{copy.prefix}_model_{ticker}",
            help=copy.model_help,
        )
    )
    model = apple_models.get(model_key)
    bundle = apple_models.load(model.key, ticker)
    st.caption(model.summary)
    # What this app has to say about *this* model on top of the registry's own
    # summary -- what it will do to a run here, in the voice this app uses.
    # Keyed by model rather than by strategy: the two day-range models share a
    # rule set and differ only in what the levels are measured below, which is
    # exactly the thing a reader picking between them needs spelled out.
    _caption(copy.outro.get(f"model_{model_key}"))
    if bundle is None:
        st.error(apple_models.unavailable_reason(model_key, ticker))
    return dayrange_params(defaults, model_key, ticker, copy)


def instrument_row(
    defaults: AppleTraderConfig, symbols: "list[str] | None", copy: FormCopy
) -> str:
    """The symbol this run trades, out of the ones a model exists for.

    Not a free-text field, unlike Apple Trader 2's: there every signal but the
    model forecasts is computed from the tape, so any symbol is a working
    configuration. Here the model *is* the strategy, so the list is exactly
    `apple_models.tickers()`.
    """
    options = apple_models.tickers()
    available = {str(s).strip().upper() for s in (symbols or [])}
    current = st.session_state.get(copy.key("ticker")) or defaults.ticker
    ticker = str(
        st.selectbox(
            "Instrument",
            options,
            index=options.index(current) if current in options else 0,
            format_func=(
                lambda t: t if not available or t in available
                else f"{t} ({copy.unavailable_suffix})"
            ),
            key=copy.key("ticker"),
            help=copy.instrument_help,
        )
    )
    labels = ", ".join(apple_models.get(k).label for k in apple_models.keys_for(ticker))
    st.caption(f":material/model_training: Models fitted on {ticker}: {labels}.")
    return ticker


def model_label(key: str) -> str:
    """One picker entry: the model's name."""
    return apple_models.get(key).label


def dayrange_params(
    defaults: AppleTraderConfig, model_key: str, ticker: str, copy: FormCopy
) -> AppleTraderConfig:
    """The day-range rules: two resting levels below the predicted high.

    Both start from the instrument's own swept pair, and the widget keys carry
    the ticker so that switching instrument re-seeds them with that symbol's
    pair rather than carrying the last symbol's numbers across.
    """
    _caption(copy.intro.get("dayrange"))
    default_buy, default_sell = dayrange_levels(ticker)
    levels = dict(ticker=ticker, buy_k=f"{default_buy:g}", sell_k=f"{default_sell:g}")
    # Before the two distances, because it decides what they are counted in and
    # a label that named the wrong unit would be worse than no label at all.
    level_unit = level_unit_param(defaults, copy)
    unit_label = UNIT_FORM_LABELS[level_unit]
    col_a, col_b = st.columns(2)
    buy_k = col_a.number_input(
        f"Buy distance (× {unit_label} below H)",
        min_value=0.05, max_value=3.0, value=default_buy, step=0.05, format="%.2f",
        key=copy.key(f"buy_k_{ticker}"),
        help=copy.help.get("buy_k", "").format(**levels),
    )
    sell_k = col_b.number_input(
        f"Sell distance (× {unit_label} below H)",
        min_value=0.0, max_value=3.0, value=default_sell, step=0.05, format="%.2f",
        key=copy.key(f"sell_k_{ticker}"),
        help=copy.help.get("sell_k", "").format(**levels),
    )
    position_pct = col_a.number_input(
        "Position size (% of cash)",
        min_value=1.0, max_value=100.0, value=defaults.position_pct, step=5.0,
        key=copy.key("dayrange_size"),
    )
    # A pair the wrong way round is not a strategy -- it would sell at a price
    # below the one it bought at, on every bar. The config refuses it outright,
    # which here would take the whole page down mid-render, so the pair is
    # repaired and the repair is stated rather than applied quietly.
    if sell_k >= buy_k:
        sell_k = round(max(0.0, buy_k - 0.05), 2)
        st.error(
            "The sell level has to sit *above* the buy level, so its distance below the "
            f"predicted high must be the smaller of the two — using {sell_k:g} until the "
            "buy distance is raised."
        )
    breach_update = breach_param(defaults, copy)
    contain_range, breach_exit = containment_params(defaults, breach_update, copy)
    stop_gain_fraction, momentum_drop, take_fraction, hold_min_gain_k = exit_params(
        defaults, float(buy_k), float(sell_k), copy
    )
    min_win_k = min_win_param(ticker, float(buy_k), float(sell_k), copy)
    _caption(copy.outro.get("dayrange"))
    return AppleTraderConfig(
        model_key=model_key,
        ticker=ticker,
        buy_k=float(buy_k),
        sell_k=float(sell_k),
        position_pct=float(position_pct),
        level_unit=level_unit,
        breach_update=breach_update,
        contain_range=contain_range,
        breach_exit=breach_exit,
        stop_gain_fraction=stop_gain_fraction,
        momentum_drop=momentum_drop,
        take_fraction=take_fraction,
        hold_min_gain_k=hold_min_gain_k,
        min_win_k=min_win_k,
    )


def min_win_param(
    ticker: str, buy_k: float, sell_k: float, copy: FormCopy
) -> float:
    """The session circuit breaker, and the one thing worth checking it against.

    The most a target exit can net is `buy_k - sell_k` ADRs a share, so a
    threshold at or above that stands the session down after *every* completed
    trade however well it went. That is a legitimate setting -- one trade a day
    unless it runs past the target -- but it is not what "stop after a bad
    trade" sounds like, so the two are compared here rather than left for a
    Results row to explain. Stated, not repaired: unlike an inverted buy/sell
    pair this configuration works, it just means something else.

    Keyed by ticker, like the levels and for the same reason: the default is per
    instrument because it only means something against that symbol's own pair,
    so switching symbol must re-seed it rather than carry the last one across.
    """
    _caption(copy.intro.get("dayrange_breaker"))
    min_win_k = st.number_input(
        "Stand down after a trade under (× ADR a share)",
        min_value=0.0, max_value=3.0, value=min_win_for(ticker), step=0.05, format="%.2f",
        key=copy.key(f"min_win_k_{ticker}"),
        help=copy.help.get("min_win_k", "").format(
            ticker=ticker, min_win_k=f"{min_win_for(ticker):g}"
        ),
    )
    target_gain = buy_k - sell_k
    if min_win_k and min_win_k >= target_gain:
        st.warning(
            f"The buy and sell levels are {target_gain:.2f} × ADR apart, so a trade that "
            f"runs all the way to the sell level nets at most that — under the "
            f"{min_win_k:.2f} × ADR above. Every completed trade will stand the session "
            "down, whatever it made: this is a one-trade-a-day rule rather than a circuit "
            f"breaker. Set it below {target_gain:.2f} to have it fire only on the weak ones.",
            icon=":material/info:",
        )
    return float(min_win_k)


# What the two number inputs call the unit in their own labels. Shorter than
# `LEVEL_UNIT_LABELS`, which has room to explain itself in a dropdown and none
# to sit inside "Buy distance (× ... below H)".
UNIT_FORM_LABELS = {
    UNIT_ADR: "ADR",
    UNIT_PRED_RANGE: "Predicted Range",
}


def level_unit_param(defaults: AppleTraderConfig, copy: FormCopy) -> str:
    """What one k is worth: the ADR, or the model's own predicted range.

    Not keyed by ticker, unlike the levels and the reference: it is a statement
    about how the strategy is parameterised rather than about a symbol, and it
    is available for every instrument -- both numbers come from the day-range
    forecast every run already loads.

    The warning fires on the combination rather than on either setting, because
    neither is wrong on its own: a moving reference is the point of the breach
    policies, and a forecast-derived unit is the point of this one. Together
    they mean the gap between the two levels grows over a breaching session,
    which is a real change to what a position is playing for and is not
    something the swept pairs were chosen under.
    """
    choice = str(
        st.selectbox(
            "Distances counted in",
            LEVEL_UNITS,
            index=(
                LEVEL_UNITS.index(defaults.level_unit)
                if defaults.level_unit in LEVEL_UNITS
                else 0
            ),
            format_func=lambda key: LEVEL_UNIT_LABELS[key],
            key=copy.key("level_unit"),
            help=copy.help.get("level_unit"),
        )
    )
    if choice == UNIT_PRED_RANGE:
        st.caption(
            ":material/info: The shipped distances were swept in ADRs, so under the "
            "predicted range they are starting points rather than swept ones."
        )
    return choice


def breach_param(defaults: AppleTraderConfig, copy: FormCopy) -> str:
    """What happens when the session trades outside the forecast.

    Not keyed by ticker: it is a rule about the forecast rather than a number
    swept per instrument, so switching symbol keeps the choice — the same reason
    the exit knobs below are not keyed either.
    """
    _caption(copy.intro.get("dayrange_breach"))
    options = list(BREACH_POLICIES)
    choice = st.selectbox(
        "If the session trades outside the forecast",
        options,
        index=options.index(defaults.breach_update),
        format_func=lambda key: BREACH_LABELS[key],
        key=copy.key("breach_update"),
        help=copy.help.get("breach_update"),
    )
    _caption(copy.outro.get(f"dayrange_breach_{choice}"))
    return str(choice)


def containment_params(
    defaults: AppleTraderConfig, breach_update: str, copy: FormCopy
) -> "tuple[bool, bool]":
    """The two rules about data the session has already printed.

    Rendered under the breach policy because both are about the same thing from
    the other side: that one says how far the forecast may *lead* the tape, and
    these say that it may not argue with it, and that a bet the tape has settled
    is banked rather than re-forecast.

    Neither is keyed by ticker: they are rules rather than swept numbers.
    """
    col_a, col_b = st.columns(2)
    contain_range = bool(
        col_a.checkbox(
            "Forecast must contain the session so far",
            value=defaults.contain_range,
            key=copy.key("contain_range"),
            help=copy.help.get("contain_range"),
        )
    )
    breach_exit = bool(
        col_b.checkbox(
            "A breach of the predicted high sells",
            value=defaults.breach_exit,
            key=copy.key("breach_exit"),
            help=copy.help.get("breach_exit"),
        )
    )
    # Said where the two settings meet rather than on either alone: neither is
    # wrong, but together they leave `breach_update` with two live options
    # instead of three, and a form that let someone pick between two identical
    # ones without saying so would be the form's fault.
    if contain_range and breach_update in (BREACH_OFF, BREACH_EXTREME):
        st.caption(
            ":material/info: With containment on, *hold the forecast* and *move to the "
            "extreme* are the same rule — the forecast is pulled out to the session's "
            "extreme either way. *Lead the tape* is the only policy that still differs."
        )
    return contain_range, breach_exit


def exit_params(
    defaults: AppleTraderConfig, buy_k: float, sell_k: float, copy: FormCopy
) -> "tuple[float, float, float, float]":
    """The managed exit: a stop under the fill, a momentum take, and a runner.

    Not keyed by ticker, unlike the levels: none of these was swept per
    instrument, so there is no per-symbol default for a switch to re-seed. The
    two knobs that only mean something once the take is on are greyed out while
    it is off rather than hidden, so turning it back on finds them where they were.

    The stop is the one that takes the levels as an argument, because it is
    written as a share of what they are playing for rather than as a distance
    of its own. The number on screen therefore means a different stop on every
    instrument, and what it comes to in ADRs is said underneath rather than
    left to be worked out -- the whole point of the reparameterisation is that
    the *fraction* travels between symbols and the distance does not.
    """
    _caption(copy.intro.get("dayrange_exits"))
    col_a, col_b = st.columns(2)
    stop_gain_fraction = col_a.number_input(
        "Stop loss (× the predicted gain, below the fill)",
        min_value=0.0, max_value=3.0, value=defaults.stop_gain_fraction, step=0.05,
        format="%.2f",
        key=copy.key("stop_gain_fraction"),
        help=copy.help.get("stop_gain_fraction", "").format(
            stop_gain_fraction=f"{defaults.stop_gain_fraction:g}"
        ),
    )
    stop_caption(col_a, float(stop_gain_fraction), float(buy_k), float(sell_k))
    momentum_drop = col_b.number_input(
        "Momentum fade to take gains (σ off its peak)",
        min_value=0.0, max_value=5.0, value=defaults.momentum_drop, step=0.1, format="%.1f",
        key=copy.key("momentum_drop"),
        help=copy.help.get("momentum_drop"),
    )
    take_pct = col_a.number_input(
        "Take on a fade (% of shares)",
        min_value=1.0, max_value=100.0, value=defaults.take_fraction * 100, step=5.0,
        key=copy.key("take_pct"),
        help=copy.help.get("take_fraction"),
        disabled=not momentum_drop,
    )
    hold_min_gain_k = col_b.number_input(
        "Keep a runner if the target is ≥ (× ADR above the fill)",
        min_value=0.0, max_value=3.0, value=defaults.hold_min_gain_k, step=0.05,
        format="%.2f",
        key=copy.key("hold_min_gain_k"),
        help=copy.help.get("hold_min_gain_k"),
        disabled=not momentum_drop,
    )
    return (
        float(stop_gain_fraction), float(momentum_drop),
        float(take_pct) / 100.0, float(hold_min_gain_k),
    )


def stop_caption(
    column, stop_gain_fraction: float, buy_k: float, sell_k: float
) -> None:
    """What the stop fraction comes to against these levels, in ADRs.

    A fraction of the predicted gain is the right thing to *set* and the wrong
    thing to compare against the other exit knobs, which are all ADR distances
    -- so the conversion is on screen rather than in the reader's head. It also
    catches the configuration that looks cautious and is not: a stop wider than
    the gain risks more than a target exit can ever pay, which is a legal bet
    and rarely the intended one.

    Written into `column` rather than into the page, or it would render under
    *both* columns of the exit grid instead of under the box it is about.

    The dollar signs are escaped for the same reason `tuning.METRICS` has none:
    this is Streamlit markdown, where a bare "$" opens a LaTeX formula and the
    text between two of them silently disappears.
    """
    if not stop_gain_fraction:
        column.caption(
            ":material/info: No stop — a position leaves at the sell level, on a "
            "momentum take, or at the closing flatten."
        )
        return
    gain = buy_k - sell_k
    column.caption(
        f"{stop_gain_fraction:g} × the {gain:.2f} × ADR the two levels are apart = "
        f"**{stop_gain_fraction * gain:.3f} × ADR** under the fill, risking "
        f"\${stop_gain_fraction:.2f} for every \$1.00 a target exit pays."
    )
    if stop_gain_fraction > 1:
        column.warning(
            f"The stop is {stop_gain_fraction:g} × the predicted gain, so this risks more "
            "than a target exit can pay. That is a legitimate bet on a rule that also "
            "exits on momentum and at the close, but it is not what a stop usually "
            "means — under 1.00 risks less than the trade is playing for.",
            icon=":material/info:",
        )


def _caption(text: "str | None") -> None:
    """Render a caption only if the calling app wrote one for this slot."""
    if text:
        st.caption(text)
