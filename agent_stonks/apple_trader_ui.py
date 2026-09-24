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
of its own -- beyond the few that are numbers worked out from the form itself.

Nothing is written into the panel
---------------------------------
Every explanation lives behind a `?`: on a widget's own help, or on the small
section headings that group the knobs. The panel itself is only the controls,
plus a warning where a combination is legal but almost certainly not what was
meant. Prose written into the page between the widgets made the form hard to
scan; in a tooltip it is there for whoever asks.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import streamlit as st

from . import apple_models
from .apple_trader import AppleTraderConfig, dayrange_levels, min_win_for
from .config import (
    BREACH_LABELS,
    BREACH_OFFERED,
    LEVEL_UNITS,
    LEVEL_UNIT_LABELS,
    UNIT_ADR,
    UNIT_PRED_RANGE,
)


@dataclass(frozen=True)
class FormCopy:
    """One app's half of the form: its widget-key namespace and its wording.

    `help` and `sections` are keyed by name rather than being separate
    attributes so that adding a knob needs no change here -- a missing key
    renders a knob (or a heading) with no `?`, which is a thin form rather than
    a crash.
    """

    #: Namespaces every widget key, so both apps can render this in one
    #: process (and SimLab can render several setups) without colliding.
    prefix: str
    #: Appended to an instrument the app cannot currently run -- "not streamed"
    #: live, "not in the datasets" in SimLab.
    unavailable_suffix: str
    instrument_help: str
    model_help: str
    #: The `?` on each section heading: "dayrange" (the levels),
    #: "dayrange_breach", "dayrange_exits", "dayrange_breaker".
    sections: "dict[str, str]" = field(default_factory=dict)
    #: Help text per knob, and per model as `model_<key>` -- the latter is
    #: appended to the model picker's own help for the model selected.
    help: "dict[str, str]" = field(default_factory=dict)

    def key(self, name: str) -> str:
        return f"{self.prefix}_{name}"


def params(
    symbols: "list[str] | None",
    copy: FormCopy,
    seed: "AppleTraderConfig | None" = None,
) -> AppleTraderConfig:
    """Apple Trader's instrument and tunables, as one config.

    The instrument comes first because it decides which models exist, and the
    model then decides which rules apply.

    `seed` is a running agent's configuration. The buy and sell distances open
    on its numbers rather than the instrument's shipped pair whenever the two
    widgets are drawn fresh -- which Streamlit does after they have been off
    screen for a run. A running agent adopts what these two widgets hold
    (`DayRangeTrader._adopt_form_levels`), so re-seeding them from the shipped
    pair would quietly move its orders.
    """
    defaults = AppleTraderConfig()
    ticker = instrument_row(defaults, symbols, copy)
    keys = apple_models.keys_for(ticker)
    default_key = defaults.model_key if defaults.model_key in keys else keys[0]
    # Scoped to the instrument: the models on offer change with it, and a
    # widget holding one that is no longer an option would be a stale
    # selection rather than a choice.
    model_widget = f"{copy.prefix}_model_{ticker}"
    shown = st.session_state.get(model_widget, default_key)
    model_key = str(
        st.selectbox(
            "Model",
            keys,
            index=keys.index(default_key),
            format_func=model_label,
            key=model_widget,
            help=model_help(shown if shown in keys else default_key, copy),
        )
    )
    bundle = apple_models.load(model_key, ticker)
    if bundle is None:
        st.error(apple_models.unavailable_reason(model_key, ticker))
    return dayrange_params(defaults, model_key, ticker, copy, seed)


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
            help=instrument_help(options, copy),
        )
    )
    return ticker


def instrument_help(options: "list[str]", copy: FormCopy) -> str:
    """The instrument picker's `?`, ending with which models each symbol has."""
    fitted = "\n".join(
        f"- **{t}**: "
        + ", ".join(apple_models.get(k).label for k in apple_models.keys_for(t))
        for t in options
    )
    return f"{copy.instrument_help}\n\n**Models fitted per instrument**\n\n{fitted}"


def model_help(key: str, copy: FormCopy) -> str:
    """The model picker's `?`: what picking a model means, then the one picked.

    The registry's summary says what the model is; the app's own note, keyed by
    model rather than by strategy, says what it does to a run here -- the two
    day-range models share a rule set and differ only in what the levels are
    measured below, which is exactly what a reader choosing between them needs.
    The help is built before the widget renders, so it describes the selection
    the widget is holding: the one on screen.
    """
    model = apple_models.get(key)
    parts = [copy.model_help, f"**{model.label}**", model.summary]
    note = copy.help.get(f"model_{key}")
    if note:
        parts.append(note)
    return "\n\n".join(p for p in parts if p)


def model_label(key: str) -> str:
    """One picker entry: the model's name."""
    return apple_models.get(key).label


def dayrange_params(
    defaults: AppleTraderConfig, model_key: str, ticker: str, copy: FormCopy,
    seed: "AppleTraderConfig | None" = None,
) -> AppleTraderConfig:
    """The day-range rules: two resting levels below the predicted high.

    Both start from the instrument's own swept pair, and the widget keys carry
    the ticker so that switching instrument re-seeds them with that symbol's
    pair rather than carrying the last symbol's numbers across.
    """
    section("Levels", copy.sections.get("dayrange"))
    default_buy, default_sell = dayrange_levels(ticker)
    levels = dict(ticker=ticker, buy_k=f"{default_buy:g}", sell_k=f"{default_sell:g}")
    # Before the two distances, because it decides what they are counted in and
    # a label that named the wrong unit would be worse than no label at all.
    level_unit = level_unit_param(defaults, copy)
    unit_label = UNIT_FORM_LABELS[level_unit]
    start_buy, start_sell = default_buy, default_sell
    if (
        seed is not None
        and (seed.ticker or "").upper() == ticker
        and seed.model_key == model_key
        and seed.level_unit == level_unit
    ):
        start_buy, start_sell = float(seed.buy_k), float(seed.sell_k)
    col_a, col_b = st.columns(2)
    buy_k = col_a.number_input(
        f"Buy distance (× {unit_label} below H)",
        min_value=0.05, max_value=3.0, value=start_buy, step=0.05, format="%.2f",
        key=copy.key(f"buy_k_{ticker}"),
        help=copy.help.get("buy_k", "").format(**levels),
    )
    sell_k = col_b.number_input(
        f"Sell distance (× {unit_label} below H)",
        min_value=0.0, max_value=3.0, value=start_sell, step=0.05, format="%.2f",
        key=copy.key(f"sell_k_{ticker}"),
        help=copy.help.get("sell_k", "").format(**levels),
    )
    position_pct = col_a.number_input(
        "Position size (% of cash)",
        min_value=1.0, max_value=100.0, value=defaults.position_pct, step=5.0,
        key=copy.key("dayrange_size"),
        help=copy.help.get("position_pct"),
    )
    # Disabled rather than hidden at 100%: the setting is still what the run
    # would do with spare cash, there just is none -- and the config ignores it
    # there too (`can_scale_in`), so the two cannot disagree.
    scale_in = st.checkbox(
        "Buy again lower while cash allows",
        value=defaults.scale_in,
        key=copy.key("dayrange_scale_in"),
        disabled=float(position_pct) >= 100,
        help=copy.help.get("scale_in"),
    )
    # One look-back for both sides of the behaviour table: the buy at the buy
    # level, and the sells at and short of the sell level (the Exits below).
    momentum_confirmation_bars = col_b.number_input(
        "Momentum confirmation period (bars)",
        min_value=0, max_value=60, value=int(defaults.momentum_confirmation_bars), step=1,
        key=copy.key("momentum_confirmation_bars"),
        help=copy.help.get("momentum_confirmation_bars"),
    )
    # A pair the wrong way round is not a strategy -- it would sell at a price
    # below the one it bought at, on every bar. The config refuses it outright,
    # which here would take the whole page down mid-render, so the pair is
    # repaired and the repair is stated rather than applied quietly.
    if sell_k >= buy_k:
        sell_k = round(max(0.0, buy_k - 0.05), 2)
        st.error(
            f"The sell distance must be smaller than the buy distance — using "
            f"{sell_k:g} until the buy distance is raised.",
            icon=":material/error:",
        )
    breach_update = breach_param(defaults, copy)
    contain_range, breach_exit = containment_params(defaults, breach_update, copy)
    stop_gain_fraction, take_fraction, hold_min_gain_k = exit_params(
        defaults, float(buy_k), float(sell_k), unit_label, copy,
        take_on=bool(momentum_confirmation_bars),
    )
    min_win_k = min_win_param(ticker, float(buy_k), float(sell_k), unit_label, copy)
    return AppleTraderConfig(
        model_key=model_key,
        ticker=ticker,
        buy_k=float(buy_k),
        sell_k=float(sell_k),
        position_pct=float(position_pct),
        scale_in=bool(scale_in),
        momentum_confirmation_bars=int(momentum_confirmation_bars),
        level_unit=level_unit,
        breach_update=breach_update,
        contain_range=contain_range,
        breach_exit=breach_exit,
        stop_gain_fraction=stop_gain_fraction,
        take_fraction=take_fraction,
        hold_min_gain_k=hold_min_gain_k,
        min_win_k=min_win_k,
    )


def min_win_param(
    ticker: str, buy_k: float, sell_k: float, unit_label: str, copy: FormCopy
) -> float:
    """The session circuit breaker, and the one thing worth checking it against.

    The most a target exit can net is `buy_k - sell_k` level units a share, so a
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
    section("Circuit breaker", copy.sections.get("dayrange_breaker"))
    min_win_k = st.number_input(
        f"Stand down after a trade under (× {unit_label} a share)",
        min_value=0.0, max_value=3.0, value=min_win_for(ticker), step=0.05, format="%.2f",
        key=copy.key(f"min_win_k_{ticker}"),
        help=copy.help.get("min_win_k", "").format(
            ticker=ticker, min_win_k=f"{min_win_for(ticker):g}", unit=unit_label
        ),
    )
    target_gain = buy_k - sell_k
    if min_win_k and min_win_k >= target_gain:
        st.warning(
            f"At or above the {target_gain:.2f} the levels are apart, every trade stands "
            f"the session down — one trade a day. Set it below {target_gain:.2f} to stop "
            "only after weak trades.",
            icon=":material/warning:",
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
    return choice


def breach_param(defaults: AppleTraderConfig, copy: FormCopy) -> str:
    """What happens when the session trades outside the forecast.

    Not keyed by ticker: it is a rule about the forecast rather than a number
    swept per instrument, so switching symbol keeps the choice — the same reason
    the exit knobs below are not keyed either.
    """
    section("Forecast breach", copy.sections.get("dayrange_breach"))
    # The rules offered, plus the one this form was seeded with if it is an
    # earlier rule no longer offered, so reopening on it does not change it.
    options = list(BREACH_OFFERED)
    if defaults.breach_update not in options:
        options.append(defaults.breach_update)
    choice = st.selectbox(
        "If the session trades outside the forecast",
        options,
        index=options.index(defaults.breach_update),
        format_func=lambda key: BREACH_LABELS[key],
        key=copy.key("breach_update"),
        help=copy.help.get("breach_update"),
    )
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
    # How containment and the breach policy overlap is said in the
    # `contain_range` help of each app rather than under the boxes.
    return contain_range, breach_exit


def exit_params(
    defaults: AppleTraderConfig,
    buy_k: float,
    sell_k: float,
    unit_label: str,
    copy: FormCopy,
    take_on: bool = True,
) -> "tuple[float, float, float]":
    """The managed exit: a stop under the fill, a momentum take, and a runner.

    The take has no trigger of its own here: it is the behaviour table's sell
    short of the sell level, read over the momentum confirmation period set
    beside the levels. `take_on` is whether that period is on, and greys out
    the two knobs that only mean something while it is.

    Not keyed by ticker, unlike the levels: none of these was swept per
    instrument, so there is no per-symbol default for a switch to re-seed. The
    knobs that only mean something once the take is on are greyed out while
    it is off rather than hidden, so turning it back on finds them where they were.

    The stop is the one that takes the levels as an argument, because it is
    written as a share of what they are playing for rather than as a distance
    of its own. The number on screen therefore means a different stop on every
    instrument, and what it comes to in level units is said in its `?` rather than
    left to be worked out -- the whole point of the reparameterisation is that
    the *fraction* travels between symbols and the distance does not.
    """
    section("Exits", copy.sections.get("dayrange_exits"))
    col_a, col_b = st.columns(2)
    stop_key = copy.key("stop_gain_fraction")
    shown = float(st.session_state.get(stop_key, defaults.stop_gain_fraction))
    stop_gain_fraction = col_a.number_input(
        "Stop loss (× the predicted gain, below the fill)",
        min_value=0.0, max_value=3.0, value=defaults.stop_gain_fraction, step=0.05,
        format="%.2f",
        key=stop_key,
        help="\n\n".join(p for p in (
            copy.help.get("stop_gain_fraction", "").format(
                stop_gain_fraction=f"{defaults.stop_gain_fraction:g}"
            ),
            stop_note(shown, float(buy_k), float(sell_k), unit_label),
        ) if p),
    )
    stop_warning(col_a, float(stop_gain_fraction))
    take_pct = col_b.number_input(
        "Take on negative momentum (% of shares)",
        min_value=1.0, max_value=100.0, value=defaults.take_fraction * 100, step=5.0,
        key=copy.key("take_pct"),
        help=copy.help.get("take_fraction"),
        disabled=not take_on,
    )
    hold_min_gain_k = col_a.number_input(
        f"Keep a runner if the target is ≥ (× {unit_label} above the fill)",
        min_value=0.0, max_value=3.0, value=defaults.hold_min_gain_k, step=0.05,
        format="%.2f",
        key=copy.key("hold_min_gain_k"),
        help=copy.help.get("hold_min_gain_k", "").format(unit=unit_label) or None,
        disabled=not take_on,
    )
    return float(stop_gain_fraction), float(take_pct) / 100.0, float(hold_min_gain_k)


def stop_note(
    stop_gain_fraction: float, buy_k: float, sell_k: float, unit_label: str
) -> str:
    """What the stop fraction comes to against these levels, in their unit.

    A fraction of the predicted gain is the right thing to *set* and the wrong
    thing to compare against the other exit knobs, which are all level-unit distances
    -- so the conversion is in the stop's `?` rather than in the reader's head.
    The help is built before the widget renders, so it reads the value the
    widget is holding: the one on screen.

    The dollar signs are escaped for the same reason `tuning.METRICS` has none:
    this is Streamlit markdown, where a bare "$" opens a LaTeX formula and the
    text between two of them silently disappears.
    """
    if not stop_gain_fraction:
        return (
            "**At the current setting:** no stop — a position leaves at the sell level, "
            "on a momentum take, or at the closing flatten."
        )
    gain = buy_k - sell_k
    return (
        f"**At the current levels:** {stop_gain_fraction:g} × the {gain:.2f} × "
        f"{unit_label} between them = **{stop_gain_fraction * gain:.3f} × {unit_label}** "
        "under the fill, "
        f"risking \\${stop_gain_fraction:.2f} for every \\$1.00 a target exit pays."
    )


def stop_warning(column, stop_gain_fraction: float) -> None:
    """Flag a stop wider than the predicted gain: legal, rarely what is meant.

    Written into `column` rather than into the page, or it would render under
    *both* columns of the exit grid instead of under the box it is about.
    """
    if stop_gain_fraction > 1:
        column.warning(
            "The stop is wider than the predicted gain — it risks more than a target "
            "exit can pay.",
            icon=":material/warning:",
        )


def section(title: str, help: "str | None") -> None:
    """A small heading for a group of knobs, with the group's explanation on its `?`."""
    st.markdown(f"**{title}**", help=help or None)
