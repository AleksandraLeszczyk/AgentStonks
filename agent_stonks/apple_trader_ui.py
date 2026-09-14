"""The Streamlit form Apple Trader is configured in, for both apps.

One module rather than one per app, for the reason `apple_rules_ui` gives about
the rule builder: the live dashboard and SimLab both offer this agent, and two
copies of a form drift the first time a range is widened or a knob is added.
They had already been forked once -- `ui.py` and `simlab/app.py` each carried
their own copy of these four functions -- and the copies still agreed on every
range, step and format, which is exactly the state in which merging them is
cheap.

Structure here, wording from the caller
---------------------------------------
What is shared is the *shape* of the form: which knobs exist, what they are
allowed to be, how the widget keys are built, which strategy renders which set,
and the config that comes out. Drift there is a bug -- a range SimLab will
sweep and the live app will refuse.

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

from . import apple_models, persistence_model
from .apple_trader import (
    ENTRY_ANTICIPATE,
    ENTRY_MODE_LABEL,
    ENTRY_MODE_PROB_LABEL,
    ENTRY_MODE_SUMMARY,
    ENTRY_MODES,
    AppleTraderConfig,
    dayrange_levels,
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
    model then decides which of the rest do: the momentum models are asked a
    question on every bar, the day-range model one at 9:35, and the
    delta-momentum regressor a third thing entirely, so the rule sets share no
    knob but position size. Rather than grey out five inputs that mean nothing,
    each strategy renders its own.
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
    if bundle is None:
        st.error(apple_models.unavailable_reason(model_key, ticker))

    if model.strategy == apple_models.STRATEGY_DAYRANGE:
        return dayrange_params(defaults, model_key, ticker, copy)
    if model.strategy == apple_models.STRATEGY_MOMENTUM_CHANGE:
        return momentum_change_params(defaults, model_key, ticker, copy)
    return momentum_params(defaults, model, bundle, ticker, copy)


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
    """One picker entry: the model's name, plus the caveat a picker can carry.

    Only the momentum models get the anticipation note -- on the other
    strategies there is no entry mode to be unable to run.
    """
    model = apple_models.get(key)
    if model.strategy != apple_models.STRATEGY_MOMENTUM:
        return model.label
    return model.label + ("" if model.anticipates else " — cannot anticipate")


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
    col_a, col_b = st.columns(2)
    buy_k = col_a.number_input(
        "Buy distance (× ADR below H)",
        min_value=0.05, max_value=3.0, value=default_buy, step=0.05, format="%.2f",
        key=copy.key(f"buy_k_{ticker}"),
        help=copy.help.get("buy_k", "").format(**levels),
    )
    sell_k = col_b.number_input(
        "Sell distance (× ADR below H)",
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
    _caption(copy.outro.get("dayrange"))
    return AppleTraderConfig(
        model_key=model_key,
        ticker=ticker,
        buy_k=float(buy_k),
        sell_k=float(sell_k),
        position_pct=float(position_pct),
    )


def momentum_change_params(
    defaults: AppleTraderConfig, model_key: str, ticker: str, copy: FormCopy
) -> AppleTraderConfig:
    """The delta-momentum rules: two thresholds on a bps/min forecast, plus two
    risk exits."""
    _caption(copy.intro.get("momentum_change"))
    col_a, col_b = st.columns(2)
    buy_thr = col_a.number_input(
        "Buy above (Δ momentum, bps/min)",
        min_value=0.0, max_value=3.0, value=float(defaults.buy_thr), step=0.05,
        format="%.2f", key=copy.key("buy_thr"), help=copy.help.get("buy_thr"),
    )
    sell_thr = col_b.number_input(
        "Sell below (−Δ momentum, bps/min)",
        min_value=0.0, max_value=3.0, value=float(defaults.sell_thr), step=0.05,
        format="%.2f", key=copy.key("sell_thr"), help=copy.help.get("sell_thr"),
    )
    m1_mult = col_a.number_input(
        "Momentum floor (× θ)",
        min_value=-6.0, max_value=0.0, value=float(defaults.m1_mult), step=0.5,
        format="%.1f", key=copy.key("m1_mult"), help=copy.help.get("m1_mult"),
    )
    stop_pct = col_b.number_input(
        "Stop below entry (%)",
        min_value=0.05, max_value=10.0, value=float(defaults.stop_pct), step=0.05,
        format="%.2f", key=copy.key("stop_pct"), help=copy.help.get("stop_pct"),
    )
    position_pct = col_a.number_input(
        "Position size (% of cash)",
        min_value=1.0, max_value=100.0, value=defaults.position_pct, step=5.0,
        key=copy.key("momentum_change_size"),
    )
    _caption(copy.outro.get("momentum_change"))
    return AppleTraderConfig(
        model_key=model_key,
        ticker=ticker,
        buy_thr=float(buy_thr),
        sell_thr=float(sell_thr),
        m1_mult=float(m1_mult),
        stop_pct=float(stop_pct),
        position_pct=float(position_pct),
    )


def momentum_params(
    defaults: AppleTraderConfig,
    model,
    bundle: "dict | None",
    ticker: str,
    copy: FormCopy,
) -> AppleTraderConfig:
    """The momentum rules: when the model is asked about a regime change, how
    sure it has to be, how much of the run to give back, and whether the model
    gets a say in the exit too."""
    model_key = model.key
    _caption(copy.intro.get("momentum"))
    entry_mode = st.segmented_control(
        "Entry",
        ENTRY_MODES,
        default=defaults.entry_mode,
        format_func=lambda mode: ENTRY_MODE_LABEL.get(mode, mode),
        key=copy.key("entry_mode"),
        help=copy.help.get("entry_mode"),
    ) or defaults.entry_mode
    st.caption(ENTRY_MODE_SUMMARY[entry_mode])

    if entry_mode == ENTRY_ANTICIPATE and not model.anticipates:
        st.error(copy.help.get("anticipate_error", "").format(label=model.label))
    bundle_threshold = persistence_model.model_threshold(bundle)

    col_a, col_b = st.columns(2)
    prob_threshold = col_a.number_input(
        ENTRY_MODE_PROB_LABEL[entry_mode],
        min_value=0.0, max_value=1.0,
        value=float(defaults.prob_threshold or bundle_threshold),
        step=0.01, format="%.2f",
        # The model key is part of the widget key so switching models re-seeds
        # this input with that model's own cut-off. The two probabilities are
        # not on a shared scale, so carrying a number across the switch would
        # silently change the strategy.
        key=f"{copy.prefix}_prob_{model_key}",
        help=copy.help.get("prob_threshold", "").format(threshold=f"{bundle_threshold:g}"),
    )
    trail_pct = col_b.number_input(
        "Trailing stop (%)",
        min_value=0.05, max_value=10.0, value=defaults.trail_pct, step=0.05,
        key=copy.key("trail"), help=copy.help.get("trail_pct"),
    )
    position_pct = col_a.number_input(
        "Position size (% of cash)",
        min_value=1.0, max_value=100.0, value=defaults.position_pct, step=5.0,
        key=copy.key("size"),
    )

    # The second exit. Off for a model that cannot forecast, since the question
    # is about bars that are not regime changes -- the same reason such a model
    # cannot anticipate.
    sells_on_reversal = col_b.checkbox(
        "Also sell on a forecast reversal",
        value=defaults.sells_on_reversal and model.anticipates,
        disabled=not model.anticipates,
        key=f"{copy.prefix}_reversal_on_{model_key}",
        help=copy.help.get("sells_on_reversal"),
    )
    reversal_threshold = col_b.number_input(
        "Reversal probability to sell",
        min_value=0.0, max_value=1.0,
        value=float(defaults.reversal_threshold or 0.30), step=0.05, format="%.2f",
        disabled=not sells_on_reversal,
        key=copy.key("reversal"), help=copy.help.get("reversal_threshold"),
    )
    if not model.anticipates:
        st.caption(
            f"{model.label} cannot forecast the breakdown of a regime, so the "
            "trailing stop is the only exit available to it."
        )
    return AppleTraderConfig(
        model_key=str(model_key),
        ticker=ticker,
        entry_mode=str(entry_mode),
        prob_threshold=float(prob_threshold),
        trail_pct=float(trail_pct),
        reversal_threshold=float(reversal_threshold) if sells_on_reversal else None,
        position_pct=float(position_pct),
    )


def _caption(text: "str | None") -> None:
    """Render a caption only if the calling app wrote one for this slot."""
    if text:
        st.caption(text)
