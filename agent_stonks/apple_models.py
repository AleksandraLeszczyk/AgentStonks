"""The models Apple Trader can run on, behind one interface.

*Which* saved model the agent trades on is a choice, and this module is where
that choice lives, so the trader, the loop, SimLab and the UI ask for "the model
named X" and never branch on which one they got. Today there are five, and they
share a rule set:

`dayrange`           TimeToChange3's blend of LightGBM, N-BEATS and N-HiTS, plus
                     an opening ridge, forecasting where the *whole session's*
                     high and low will land, once, from the first five minutes.
                     30% of a 14-day rolling baseline's error removed over 129
                     test sessions. The levels rest under the predicted high,
                     flat all day.
`dayrange_intraday`  the same forecast read through IntradayVolatility's
                     time-of-day shape, so the reference the levels hang off
                     moves with the clock. Two saved files rather than one, and
                     unavailable wherever either is missing.
`highlow`            HighLow_5m's forecast of the same two numbers, anchored on
                     the 9:35 price and fitted on every session since 2023 (35%
                     less error than TimeToChange3 on the same test window).
                     Only the forecast differs: its bundle carries
                     `kind="highlow"` and `apple_trader.DayRangeTrader` asks
                     `highlow_model` for today's range instead.
`highlow2`           HighLow2_5m's forecast of the same two numbers under
                     stricter rules: this morning read entirely from IEX, the
                     pre-market as an input, shock days out of training, five
                     mega-caps pooled in. Level with HighLow on its test window,
                     slightly ahead. Its bundle carries `kind="highlow2"` and
                     the forecast comes from `highlow2_model`. INTC's bundle
                     also forecasts the high and low *after* 9:35, and that
                     pair is the one the trader gets -- marked, like
                     HighLow_3m's, as starting after the opening window.
`highlow3m`          HighLow_3m's forecast of a different pair of numbers: the
                     high and low of the rest of the session *after* 9:33,
                     called at 9:33 from IEX's first three minutes, the daily
                     history and last night's option positioning (walls,
                     dealer gamma, the implied move). Its bundle carries
                     `kind="highlow3m"` and the forecast comes from
                     `highlow3m_model`; the range is marked as starting after
                     the opening window, so the trader measures breaches on
                     the bars after it.

That second entry is why `AppleModel` carries `level_source`: what the levels
are measured below used to be a separate setting beside the model picker, which
asked the reader to pair two choices that only make sense in two combinations.
It is a property of the model now, and `apple_trader.AppleTraderConfig` reads it
from here. A record written while it was a separate choice still carries its own
`level_source` and replays on that, keeping the signature it was filed under.

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

from .config import APPLE_TRADER_MODEL, LEVELS_DAYRANGE, LEVELS_INTRADAY


# The rule sets a model can drive.
STRATEGY_DAYRANGE = "dayrange"

# The symbol everything here defaults to, and what a config or a stored record
# arriving without one means.
DEFAULT_TICKER = "AAPL"

# Which symbols the day-range model was fitted on. Each really is its own
# model: TimeToChange3's pipeline was run per ticker, and each run produced its
# own bundle.
DAYRANGE_TICKERS = (DEFAULT_TICKER, "GOOGL", "INTC")

# Which symbols HighLow_5m has saved a bundle for (`highlow15m_<TICKER>.*`).
# AAPL (saved 2026-09-15), INTC (2026-09-21), MU (2026-09-30, its opening
# read from IEX -- see `highlow_model`), BE (2026-09-30, IEX opening plus its
# theme peers VST, PLUG and XLU), AVGO (2026-10-04, base features, sessions
# kept from 320 bars) and NVDA (2026-10-04, reads SPY, the pre-market, the
# previous evening, the opening's shape and the volatility regime); GOOGL is
# in the notebook's config but has not been run through it. Adding one is this
# tuple plus its files in `Code/Models` -- and a check that the bundle's shipped
# candidates read only the base features and the custom groups `highlow_model`
# mirrors (`GROUP_COLS`; `_build_bundle` refuses anything else).
HIGHLOW_TICKERS = (DEFAULT_TICKER, "INTC", "MU", "BE", "AVGO", "NVDA")

# Which symbols HighLow2_5m has saved a bundle for (`highlow2_5m_<TICKER>.*`):
# AAPL (saved 2026-10-04) and INTC (2026-10-06, AAPL's recipe with ten stocks
# pooled in, plus a rest-of-session head: the high and low from 9:35 to the
# close, which `highlow2_model.forecast_from` hands the trader in place of the
# day's). The notebook has processed GOOGL, MSFT, AMZN, META, NVDA, AMD, MU,
# AVGO and QCOM as training pools, but fitted no model of their own. Adding one
# is this tuple plus its two files in `Code/Models` -- `_build_bundle` refuses a
# bundle whose shipped blend (either head) reads anything but the base and
# pre-market groups.
HIGHLOW2_TICKERS = (DEFAULT_TICKER, "INTC")

# Which symbols HighLow_3m has saved a bundle for (`highlow3m_<TICKER>.*`):
# AAPL (saved 2026-10-05). GOOGL, MSFT, AMZN, META and NVDA are processed as its
# training pool, with their own option tables, but have no model. Adding one is
# this tuple plus its two files in `Code/Models` and, if its blend reads the
# option group, the notebook's per-ticker option settings in
# `highlow3m_model.NOTEBOOK_OPTIONS` (`_build_bundle` refuses it without them).
HIGHLOW3M_TICKERS = (DEFAULT_TICKER,)


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
    # Why `load` returned None, for a model whose answer is more specific than
    # "no file at <path>". A model built on two saved files has two ways to be
    # unavailable and the reader has to be told which one, so this overrides
    # `unavailable_reason`'s default sentence when it is set.
    unavailable: "Callable[[str], str] | None" = None
    # What the two levels are measured below, for a model that decides it.
    # `apple_trader.AppleTraderConfig` reads this rather than offering the
    # reference as a separate choice: which curve the levels hang off is what
    # distinguishes these two models, so it is the model picker's answer.
    level_source: str = LEVELS_DAYRANGE

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


def _intraday():
    """IntradayVolatility's loader, imported late like the day-range one.

    Cheap by comparison -- a JSON file and no ML stack -- but kept lazy so that
    listing the model names still imports nothing but this module.
    """
    from . import intraday_vol_model

    return intraday_vol_model


def _load_dayrange_intraday(ticker: str = DEFAULT_TICKER) -> "dict | None":
    """The day-range bundle, but only when the shape that reads it is there too.

    Returns the *day-range* bundle because that is what the rules forecast
    from; the shape is loaded where it is used (`apple_trader._reference`).
    What this adds is the refusal: a configuration naming this model with no
    shape on disk would otherwise run as the flat model under this model's
    name, which is the one outcome worth failing for.
    """
    bundle = _load_dayrange(ticker)
    if bundle is None or _intraday().load(ticker) is None:
        return None
    return bundle


def _intraday_path(ticker: str = DEFAULT_TICKER) -> Path:
    return _intraday().model_path(ticker)


def _dayrange_intraday_unavailable(ticker: str = DEFAULT_TICKER) -> str:
    """Which of the two files is missing, rather than a guess at one of them."""
    symbol = (ticker or DEFAULT_TICKER).upper()
    if _load_dayrange(symbol) is None:
        return (
            f"No day-range bundle at {_dayrange_path(symbol)} (or PyTorch, LightGBM, "
            "scikit-learn and joblib are not installed). This model reads that "
            "forecast before it reads the intraday shape."
        )
    return (
        f"The day-range bundle for {symbol} loaded, but the IntradayVolatility export "
        f"at {_intraday_path(symbol)} is missing or unreadable — write it with "
        "FinNotebooks/IntradayVolatility/scripts/export_app_model.py, or run the "
        "flat day-range model instead."
    )


def _load_highlow(ticker: str = DEFAULT_TICKER) -> "dict | None":
    """The HighLow bundle for one ticker, imported late for the same reason as
    the day-range one (torch and LightGBM, in that order)."""
    try:
        from . import highlow_model
    except ImportError:
        return None
    return highlow_model.load_bundle(ticker)


def _highlow_path(ticker: str = DEFAULT_TICKER) -> Path:
    try:
        from . import highlow_model
    except ImportError:
        return Path(f"highlow15m_{(ticker or DEFAULT_TICKER).upper()}.joblib")
    return highlow_model.model_path(ticker)


def _load_highlow2(ticker: str = DEFAULT_TICKER) -> "dict | None":
    """The HighLow2 bundle for one ticker, imported late: LightGBM only, no
    torch, but still not something listing the model names should load."""
    try:
        from . import highlow2_model
    except ImportError:
        return None
    return highlow2_model.load_bundle(ticker)


def _highlow2_path(ticker: str = DEFAULT_TICKER) -> Path:
    try:
        from . import highlow2_model
    except ImportError:
        return Path(f"highlow2_5m_{(ticker or DEFAULT_TICKER).upper()}.joblib")
    return highlow2_model.model_path(ticker)


def _load_highlow3m(ticker: str = DEFAULT_TICKER) -> "dict | None":
    """The HighLow_3m bundle for one ticker, imported late (LightGBM, and
    SciPy for the option pricing)."""
    try:
        from . import highlow3m_model
    except ImportError:
        return None
    return highlow3m_model.load_bundle(ticker)


def _highlow3m_path(ticker: str = DEFAULT_TICKER) -> Path:
    try:
        from . import highlow3m_model
    except ImportError:
        return Path(f"highlow3m_{(ticker or DEFAULT_TICKER).upper()}.joblib")
    return highlow3m_model.model_path(ticker)


DAYRANGE_KEY = "dayrange"
DAYRANGE_INTRADAY_KEY = "dayrange_intraday"
HIGHLOW_KEY = "highlow"
HIGHLOW2_KEY = "highlow2"
HIGHLOW3M_KEY = "highlow3m"
# Both saved models have to exist for the pairing, so the symbols it covers are
# the symbols both were fitted on.
DAYRANGE_INTRADAY_TICKERS = tuple(
    t for t in DAYRANGE_TICKERS if t in _intraday().TICKERS
)

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
    DAYRANGE_INTRADAY_KEY: AppleModel(
        key=DAYRANGE_INTRADAY_KEY,
        label="Day Range × Intraday Volatility",
        summary=(
            "The same 9:35 day-range forecast, read through IntradayVolatility's "
            "time-of-day shape instead of flat. The predicted high and low are "
            "stretched by a curve that peaks at the open, decays to a flat midday and "
            "opens back up into the close, and the levels are measured below that "
            "curve's upper edge at each minute — so they answer \"how far is this "
            "stock reaching *right now*\" rather than \"how far will it reach today\". "
            "Two saved models rather than one, and the shape is a second claim on top "
            "of the forecast: opt in and measure it in SimLab rather than assuming it "
            "improves on the flat high."
        ),
        requires=(
            "PyTorch, LightGBM, scikit-learn and joblib, plus both .pt checkpoints "
            "and the IntradayVolatility export"
        ),
        strategy=STRATEGY_DAYRANGE,
        # Both files have to exist, so this is the intersection rather than
        # either model's own list -- a symbol with a day-range bundle and no
        # shape is not a symbol this model was fitted on.
        tickers=DAYRANGE_INTRADAY_TICKERS,
        load=_load_dayrange_intraday,
        path=_intraday_path,
        unavailable=_dayrange_intraday_unavailable,
        level_source=LEVELS_INTRADAY,
    ),
    HIGHLOW_KEY: AppleModel(
        key=HIGHLOW_KEY,
        label="HighLow model",
        summary=(
            "FinNotebooks' HighLow_5m forecast of where the session's high and low "
            "will land, made once at 9:35 -- measured from the 9:35 price in units "
            "of the 14-day average range, by a blend picked per ticker on validation "
            "(AAPL and AVGO: LightGBM + N-BEATS; INTC: N-HiTS alone; MU: LightGBM + "
            "N-BEATS + N-HiTS; BE: N-BEATS + N-HiTS; NVDA: LightGBM + N-HiTS), fitted on "
            "every session since 2023 with its first five minutes (MU's and BE's read "
            "from IEX, the only tape there is at 9:35 on a basic plan; BE's also with "
            "VST's opening move, its lead AI-power peer; NVDA's with SPY's opening, the "
            "pre-market to 9:19 and the previous evening's after-hours). On the "
            "129-session test window its error is 35% below "
            "TimeToChange3's on AAPL ($1.37 per extreme against $2.11) and 31% below on "
            "INTC; MU, BE and AVGO have no TimeToChange3 bundle, and beat that approach "
            "retrained on them by 25%, 33% and 34%. "
            "Only the forecast changes: the "
            "levels, exits and breach rules are the day-range strategy's, hung off "
            "this predicted high and predicted range."
        ),
        requires=(
            "PyTorch, LightGBM, scikit-learn and joblib, the weighted network "
            "checkpoints, and Alpaca credentials for ~150 sessions of SIP minute history "
            "(plus IEX openings on MU and BE, BE's peers VST, PLUG and XLU, and on NVDA "
            "SPY's history and this morning's SIP pre-market)"
        ),
        strategy=STRATEGY_DAYRANGE,
        tickers=HIGHLOW_TICKERS,
        load=_load_highlow,
        path=_highlow_path,
    ),
    HIGHLOW2_KEY: AppleModel(
        key=HIGHLOW2_KEY,
        label="HighLow2 model",
        summary=(
            "FinNotebooks' HighLow2_5m forecast of where the session's high and low "
            "will land, made once at 9:35 and measured like HighLow's -- from the 9:35 "
            "price in 14-day average ranges -- under stricter rules: everything about "
            "this morning (the first five minutes, the open, the opening volume) read "
            "from IEX, the only live tape at 9:35 on a basic plan; the pre-market as an "
            "input (SIP to 09:19, IEX to 09:29, last evening's after-hours); shock days "
            "(the day after earnings, jobs reports, geopolitical shocks) left out of "
            "training; and similar stocks pooled in as training rows (five mega-caps for "
            "AAPL, those plus AAPL and four chip stocks for INTC). LightGBM alone, three "
            "seeds. On AAPL's 9-session test window it misses each extreme by $1.17 "
            "(15.8% of the 14-day range), level with HighLow and slightly ahead. INTC's "
            "bundle also forecasts the high and low still to come from 9:35 to the close "
            "(on about a quarter of INTC's sessions one of the day's extremes is already "
            "in by then), and on INTC that is the pair the levels are built on: the "
            "reference high and the predicted range are the rest of the session's, and a "
            "breach is read on the bars after 9:35. Only the forecast changes: the "
            "levels, exits and breach rules are the day-range strategy's."
        ),
        requires=(
            "LightGBM and joblib, and Alpaca credentials for ~150 sessions of SIP and IEX "
            "minute history, extended hours included"
        ),
        strategy=STRATEGY_DAYRANGE,
        tickers=HIGHLOW2_TICKERS,
        load=_load_highlow2,
        path=_highlow2_path,
    ),
    HIGHLOW3M_KEY: AppleModel(
        key=HIGHLOW3M_KEY,
        label="HighLow_3m model",
        summary=(
            "FinNotebooks' HighLow_3m forecast of how high and how low the price will "
            "trade from 9:33 to the close -- the part of the day a 9:33 order can still "
            "reach, not the whole session's extremes (on about half of AAPL's sessions "
            "one of those prints in the first three minutes). Called once at 9:33 from "
            "IEX's first three minutes, the SIP daily history and last night's option "
            "positioning: the near-term call and put walls, the call gamma wall, dealer "
            "gamma re-read at the 9:33 price, the options market's implied move and "
            "put/call volume, with open interest rebuilt from Alpaca's daily option bars. "
            "LightGBM (three seeds) 0.8 + a linear median regression 0.2, the five other "
            "mega-caps pooled in. On its 2026 holdout it misses each extreme by 30% of a "
            "typical day's range, 8.7% less than a 14-day baseline in every month. Only "
            "the forecast changes: the levels, exits and breach rules are the day-range "
            "strategy's, and a breach is measured on the bars after 9:33."
        ),
        requires=(
            "LightGBM, SciPy and joblib, and Alpaca credentials for ~150 sessions of SIP "
            "and IEX minute history, the option contract list and ~4 months of daily "
            "option bars, plus Yahoo's 13-week T-bill yield"
        ),
        strategy=STRATEGY_DAYRANGE,
        tickers=HIGHLOW3M_TICKERS,
        load=_load_highlow3m,
        path=_highlow3m_path,
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
    if model.unavailable is not None:
        return model.unavailable(symbol)
    return (
        f"No {model.label} model at {model.path(symbol)} "
        f"(or {model.requires} are not installed)."
    )


def strategy(key: "str | None") -> str:
    """Which rule set the named model drives."""
    return get(key).strategy
