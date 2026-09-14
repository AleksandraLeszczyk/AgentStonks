"""The models Apple Trader can run on, behind one interface.

*Which* saved model the agent trades on is a choice, and this module is where
that choice lives, so the trader, the loop, SimLab and the UI ask for "the model
named X" and never branch on which one they got. Today there is one:

`dayrange`     TimeToChange3's blend of LightGBM, N-BEATS and N-HiTS, plus an
               opening ridge, forecasting where the *whole session's* high and
               low will land, once, from the first five minutes. 30% of a
               14-day rolling baseline's error removed over 129 test sessions.

The persistence classifier, the N-BEATS persistence forecaster and the
delta-momentum regressor used to sit beside it and have been removed. Their
bundles may still be in `Code/Models`, but nothing here reads them, and a
stored SimLab record naming one is refused rather than replayed on another
model (`apple_trader.model_ticker_error`).

Which symbols a model exists for
--------------------------------
A model is fitted on one ticker and the notebooks make no claim that any of
them transfers, so "which model" and "which instrument" are one question rather
than two. `AppleModel.tickers` is the answer, and it is deliberately a property
of the model rather than of the app.

Everything downstream reads `keys_for(ticker)` instead of `MODELS`, which is
what makes an instrument with no model at all a supported choice rather than a
broken one: a rule set written on the tape, the momentum regime, the position
and the clock needs no model. Adding a ticker to a model here (plus its bundle
in `Code/Models`) is the whole change needed to offer it -- ORCL, for instance,
is trained in `FinNotebooks/Models` and is one entry away.

`AppleModel.strategy` names which rule set a model drives, and
`apple_trader.build_trader` turns it into a state machine. With one model there
is one strategy, but the seam is kept: a second model is a registry entry and a
trader class, not an edit to every caller.

Every model is optional: a missing file or a missing dependency makes it
*unavailable*, reported as such, rather than an agent that silently never
trades.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .config import APPLE_TRADER_MODEL


# The rule sets a model can drive.
STRATEGY_DAYRANGE = "dayrange"

# The symbol everything here defaults to, and what a config or a stored record
# arriving without one means.
DEFAULT_TICKER = "AAPL"

# Which symbols the day-range model was fitted on. Each really is its own
# model: TimeToChange3's pipeline was run per ticker, and each run produced its
# own bundle.
DAYRANGE_TICKERS = (DEFAULT_TICKER, "GOOGL", "INTC")


@dataclass(frozen=True)
class AppleModel:
    """One model Apple Trader can be pointed at."""

    key: str
    label: str
    # One line for a picker: what the model is, not how well it scores.
    summary: str
    # What has to be installed for `load` to be able to return anything.
    requires: str
    # Which rule set this model drives -- `apple_trader.build_trader` turns it
    # into a state machine.
    strategy: str
    # The symbols this model was fitted on. A model is not available for
    # anything else -- see the module docstring -- and `load`/`path` are only
    # ever called with one of these.
    tickers: "tuple[str, ...]"
    # Registry data, not the entry point: everything loads through
    # `load(key, ticker)` below, so there is one seam for tests to replace and
    # one place a caller can reach a model from.
    load: Callable[[str], "dict | None"]
    path: Callable[[str], Path]

    def covers(self, ticker: "str | None") -> bool:
        return (ticker or DEFAULT_TICKER).upper() in self.tickers


def _load_dayrange(ticker: str = DEFAULT_TICKER) -> "dict | None":
    """The TimeToChange3 bundle for one ticker, or None if torch/LightGBM are
    not installed.

    Imported here rather than at module scope: the module pulls LightGBM in
    ahead of torch on purpose, and doing that at `import apple_models` time
    would impose a 200 MB dependency, and the ordering, on every process that
    only wanted to list the model names.
    """
    try:
        from . import dayrange_model
    except ImportError:
        return None
    return dayrange_model.load_bundle(ticker)


def _dayrange_path(ticker: str = DEFAULT_TICKER) -> Path:
    try:
        from . import dayrange_model
    except ImportError:
        return Path(f"timetochange3_dayrange_{(ticker or DEFAULT_TICKER).upper()}.joblib")
    return dayrange_model.model_path(ticker)


DAYRANGE_KEY = "dayrange"

MODELS: "dict[str, AppleModel]" = {
    DAYRANGE_KEY: AppleModel(
        key=DAYRANGE_KEY,
        label="Day-range forecast (TimeToChange3)",
        summary=(
            "Forecasts where the whole session's high and low will land, once, at 9:35 "
            "— an equal blend of LightGBM, N-BEATS and N-HiTS over a year of daily "
            "history, corrected by a ridge on the first five minutes and clipped to "
            "contain the opening range. It removes 30% of a 14-day rolling baseline's "
            "error over 129 test sessions. What it predicts well is the *width* of the "
            "day, not its direction, which is why the rules around it buy well below "
            "the predicted high and sell just under it."
        ),
        requires="PyTorch, LightGBM, scikit-learn and joblib, plus both .pt checkpoints",
        strategy=STRATEGY_DAYRANGE,
        tickers=DAYRANGE_TICKERS,
        load=_load_dayrange,
        path=_dayrange_path,
    ),
}

DEFAULT_MODEL = APPLE_TRADER_MODEL if APPLE_TRADER_MODEL in MODELS else DAYRANGE_KEY


def keys() -> "list[str]":
    """Every model key, in the order a picker should offer them."""
    return list(MODELS)


def keys_for(ticker: "str | None") -> "list[str]":
    """The model keys that exist for one symbol, in picker order.

    Empty for a symbol nothing was fitted on, which is a supported answer: the
    rules written on the tape, the momentum regime, the position and the clock
    need no model at all.
    """
    return [key for key, model in MODELS.items() if model.covers(ticker)]


def tickers() -> "list[str]":
    """Every symbol some model covers, with the default first.

    What a picker offers as the *known* instruments. It is not a whitelist --
    anything that is streamed can be traded on model-free rules -- so callers
    that offer a free-text choice should union this with what they have.
    """
    seen: "list[str]" = []
    for model in MODELS.values():
        for symbol in model.tickers:
            if symbol not in seen:
                seen.append(symbol)
    return sorted(seen, key=lambda s: (s != DEFAULT_TICKER, s))


def covers(key: "str | None", ticker: "str | None") -> bool:
    """Whether the named model exists for this symbol at all."""
    return get(key).covers(ticker)


def get(key: "str | None") -> AppleModel:
    """The named model, falling back to the default for an unknown key.

    Unknown keys reach here from experiment records written before a model
    existed or after one was removed; the Results page should still render
    rather than crash. Anything about to *run* a config checks membership in
    `MODELS` first (`apple_trader.model_ticker_error`), so the fallback never
    turns a removed model into a different one.
    """
    return MODELS.get(key or DEFAULT_MODEL) or MODELS[DEFAULT_MODEL]


def load(key: "str | None", ticker: "str | None" = None) -> "dict | None":
    """The named model's bundle for one symbol, or None when it cannot be
    assembled.

    A symbol the model was never fitted on is refused here rather than in each
    loader, so "no such model for this ticker" and "the file is missing" are one
    answer to the caller and `unavailable_reason` can tell them apart.
    """
    symbol = (ticker or DEFAULT_TICKER).upper()
    model = get(key)
    if not model.covers(symbol):
        return None
    return model.load(symbol)


def unavailable_reason(key: "str | None", ticker: "str | None" = None) -> str:
    """Why `load` returned None, in the terms a user can act on."""
    symbol = (ticker or DEFAULT_TICKER).upper()
    model = get(key)
    if not model.covers(symbol):
        return (
            f"There is no {model.label} model for {symbol} — it was fitted on "
            f"{', '.join(model.tickers)} only, and nothing claims it transfers."
        )
    return (
        f"No {model.label} model at {model.path(symbol)} "
        f"(or {model.requires} are not installed)."
    )


def strategy(key: "str | None") -> str:
    """Which rule set the named model drives."""
    return get(key).strategy
