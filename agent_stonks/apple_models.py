"""The models Apple Trader can run on, behind one interface.

*Which* saved model the agent trades on is a choice, and this module is where
that choice lives, so the trader, the loop, SimLab and the UI ask for "the model
named X" and never branch on which one they got. Today there are five saved
models, the first three of which share a rule set, and one hybrid of two of
them:

`dayrange`           TimeToChange3's blend of LightGBM, N-BEATS and N-HiTS, plus
                     an opening ridge, forecasting where the *whole session's*
                     high and low will land, once, from the first five minutes.
                     30% of a 14-day rolling baseline's error removed over 129
                     test sessions. The levels rest under the predicted high,
                     flat all day.
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
                     the forecast comes from `highlow2_model`. A bundle with
                     a rest-of-session head (INTC's had one; none ships now)
                     hands the trader the high and low *after* 9:35 instead,
                     marked as starting after the opening window, so the
                     trader measures breaches on the bars after it.
`fastfurious`        Fast&Furious's forecast of a different kind of number:
                     the average momentum of the next 30 minutes, made at the
                     close of every minute from 09:31 to 15:30, with a forecast
                     of the next two hours' high and low beside it. It drives
                     its own rule set (`STRATEGY_MOMENTUM`): buy when the
                     forecast is in the top 5% of the last 20 sessions', a
                     stop under the forecast low, out 30 minutes after the
                     last signal. Its bundle carries `kind="fastfurious"` and
                     the forecasts come from `fastfurious_model`.
`goodmoments`        GoodMoments' forecast of another kind of number again: at
                     the close of every minute, the probability that a later
                     bar of the session trades 0.2 ADR above this one's close,
                     beside the same trees' probability on the clock and
                     today's volatility alone. It drives its own rule set
                     (`STRATEGY_PROBABILITY`): buy when the model's *edge*
                     over that baseline is 2 points or more, a take-profit at
                     +0.2 ADR, a stop at -0.75 ADR, otherwise held to the
                     close. Its bundle carries `kind="goodmoments"` and the
                     forecasts come from `goodmoments_model`.
`highlow2_goodmoments`
                     A hybrid of the two (2026-10-21): GoodMoments' rules,
                     the edge alone buying in the opening window, and from
                     09:35 a buy also needing the bar's close at or under
                     HighLow2's buy level (`AppleModel.levels_of`). Its
                     bundle is GoodMoments' with HighLow2's under
                     `HIGHLOW2_KEY`; `apple_trader.HighLow2GoodMomentsTrader`
                     trades it.

The persistence classifier, the N-BEATS persistence forecaster and the
delta-momentum regressor used to sit beside them and have been removed, and so
have HighLow_3m, Day Range × Intraday Volatility and Fast&Furious + HighLow
(2026-10-20). Their bundles may still be in `Code/Models`, but nothing here
reads them, and a stored SimLab record naming one is refused rather than
replayed on another model (`apple_trader.model_ticker_error`).

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
`apple_trader.build_trader` turns it into a state machine: the three range
models drive the day-range rules (two resting levels), Fast&Furious the
momentum rules (a per-minute signal) and GoodMoments the probability rules (a
per-minute edge, a take and a stop in ADRs). A caller written for one of them
-- Orchestra, the day-range levels on the chart, a tuning axis -- asks
`strategy(key)` rather than assuming.

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
STRATEGY_MOMENTUM = "momentum"
STRATEGY_PROBABILITY = "probability"

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
# kept from 320 bars), NVDA (2026-10-04, reads SPY, the pre-market, the
# previous evening, the opening's shape and the volatility regime), and QCOM,
# ORCL and META (2026-10-13, NVDA's recipe; all three read the opening's trade
# counts and VWAP, and ORCL also SPY and its SIP theme peers MSFT, AMZN and
# GOOGL), and AMGN (2026-10-19, the same recipe outside tech: SPY, the
# pre-market, the previous evening, the opening's trade counts and its biotech
# peers GILD, REGN and VRTX, sessions kept from 310 bars), and TMO (2026-10-20,
# the same recipe, of which the blend kept only its life-science tools peers
# DHR, A and WAT; sessions kept from 260 bars, the peers' with first bars as
# late as 09:37), and AMD (2026-10-21, the same recipe: SPY, the previous
# evening, the opening's shape, the volatility regime and its chip peers NVDA,
# INTC and AVGO -- AVGO's history cleaned at AVGO's own 320 bars); GOOGL is
# in the notebook's config but has not been run through it.
# Adding one is this tuple plus its files in `Code/Models` -- and a check that
# the bundle's shipped candidates read only the base features and the custom
# groups `highlow_model` mirrors (`GROUP_COLS`; `_build_bundle` refuses
# anything else).
HIGHLOW_TICKERS = (
    DEFAULT_TICKER, "INTC", "MU", "BE", "AVGO", "NVDA", "QCOM", "ORCL", "META", "AMGN", "TMO", "AMD",
)

# Which symbols HighLow2_5m is wired up for (`highlow2_5m_<TICKER>.*`): AAPL
# (saved 2026-10-04). INTC's bundle (2026-10-06, with a rest-of-session head:
# the high and low from 9:35 to the close, which `highlow2_model.forecast_from`
# hands the trader in place of the day's) was wired up until 2026-10-20 and
# removed at the user's request; its files stay in `Code/Models`. The notebook
# has processed GOOGL, MSFT, AMZN, META, NVDA, AMD, MU, AVGO and QCOM as training
# pools, but fitted no model of their own. Adding one is this tuple plus its two
# files in `Code/Models` -- `_build_bundle` refuses a bundle whose shipped blend
# (either head) reads anything but the base and pre-market groups.
HIGHLOW2_TICKERS = (DEFAULT_TICKER,)

# Which symbols Fast&Furious has saved a bundle for (`fastfurious_<TICKER>.*`):
# AAPL (fitted through 2026-09-04, saved 2026-10-12). GOOGL, MSFT, AMZN, META
# and NVDA were processed as a training pool and made the model worse, so none
# has a model. Adding one is this tuple plus its two files in `Code/Models`,
# and -- since the stop's range model reads the option tables -- the ticker's
# entry in `option_positioning.NOTEBOOK_OPTIONS`.
FASTFURIOUS_TICKERS = (DEFAULT_TICKER,)

# Which symbols GoodMoments has saved a bundle for (`goodmoments_<TICKER>.*`,
# plus the `_timevol` twin its rules read the edge against): AAPL (fitted
# through 2026-10-16, saved 2026-10-19; the twin written 2026-10-20 by
# `GoodMoments/scripts/export_timevol.py`). The notebook is AAPL-only. Adding
# one is this tuple plus its four files in `Code/Models`.
GOODMOMENTS_TICKERS = (DEFAULT_TICKER,)

# HighLow2 + GoodMoments: wherever both are fitted.
HIGHLOW2_GOODMOMENTS_TICKERS = tuple(t for t in GOODMOMENTS_TICKERS if t in HIGHLOW2_TICKERS)


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
    # For a hybrid of two models: `signal_of` is the model whose per-minute
    # signal and exits it trades (its rule set is that model's), `levels_of`
    # the day-range model whose buy level its entries also have to be under
    # once that model's opening window has closed. A hybrid with no base
    # configuration of its own starts from both (`apple_trader.default_config`),
    # and its bundle is the signal model's with the levels model's under
    # `levels_of`'s key. None for every model that is its own forecast.
    signal_of: "str | None" = None
    levels_of: "str | None" = None

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


def _load_fastfurious(ticker: str = DEFAULT_TICKER) -> "dict | None":
    """The Fast&Furious bundle for one ticker, imported late (LightGBM)."""
    try:
        from . import fastfurious_model
    except ImportError:
        return None
    return fastfurious_model.load_bundle(ticker)


def _fastfurious_path(ticker: str = DEFAULT_TICKER) -> Path:
    try:
        from . import fastfurious_model
    except ImportError:
        return Path(f"fastfurious_{(ticker or DEFAULT_TICKER).upper()}.joblib")
    return fastfurious_model.model_path(ticker)


def _load_goodmoments(ticker: str = DEFAULT_TICKER) -> "dict | None":
    """The GoodMoments bundle for one ticker (the saved model and its
    time-and-volatility twin), imported late (LightGBM, SciPy)."""
    try:
        from . import goodmoments_model
    except ImportError:
        return None
    return goodmoments_model.load_bundle(ticker)


def _goodmoments_path(ticker: str = DEFAULT_TICKER) -> Path:
    try:
        from . import goodmoments_model
    except ImportError:
        return Path(f"goodmoments_{(ticker or DEFAULT_TICKER).upper()}.joblib")
    return goodmoments_model.model_path(ticker)


def _goodmoments_unavailable(ticker: str = DEFAULT_TICKER) -> str:
    """Which of the two files is missing -- the twin has its own fix."""
    try:
        from . import goodmoments_model
    except ImportError:
        return "GoodMoments needs LightGBM, SciPy and joblib, which are not installed."
    return goodmoments_model.unavailable_reason(ticker)


def _load_highlow2_goodmoments(ticker: str = DEFAULT_TICKER) -> "dict | None":
    """GoodMoments' bundle with HighLow2's beside it under `HIGHLOW2_KEY`, or
    None when either cannot be assembled. GoodMoments' own keys are left as
    they are -- its forecasts, caches and provenance read them -- so the
    hybrid shares GoodMoments' memoised sessions."""
    goodmoments = _load_goodmoments(ticker)
    highlow2 = _load_highlow2(ticker)
    if goodmoments is None or highlow2 is None:
        return None
    return {**goodmoments, HIGHLOW2_KEY: highlow2}


def _highlow2_goodmoments_unavailable(ticker: str = DEFAULT_TICKER) -> str:
    """Which half of the hybrid is missing, in that model's own words."""
    missing = [
        unavailable_reason(key, ticker)
        for key, load_part in ((GOODMOMENTS_KEY, _load_goodmoments), (HIGHLOW2_KEY, _load_highlow2))
        if load_part(ticker) is None
    ]
    return " ".join(missing) or "Both halves loaded; try again."


DAYRANGE_KEY = "dayrange"
HIGHLOW_KEY = "highlow"
HIGHLOW2_KEY = "highlow2"
FASTFURIOUS_KEY = "fastfurious"
GOODMOMENTS_KEY = "goodmoments"
HIGHLOW2_GOODMOMENTS_KEY = "highlow2_goodmoments"

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
    HIGHLOW_KEY: AppleModel(
        key=HIGHLOW_KEY,
        label="HighLow model",
        summary=(
            "FinNotebooks' HighLow_5m forecast of where the session's high and low "
            "will land, made once at 9:35 -- measured from the 9:35 price in units "
            "of the 14-day average range, by a blend picked per ticker on validation "
            "(AAPL and AVGO: LightGBM + N-BEATS; INTC: N-HiTS alone; MU: LightGBM + "
            "N-BEATS + N-HiTS; BE: N-BEATS + N-HiTS; NVDA: LightGBM + N-HiTS; QCOM, "
            "ORCL, TMO and AMD: LightGBM + N-BEATS + N-HiTS; META and AMGN: N-BEATS alone), fitted on "
            "every session since 2023 with its first five minutes (MU's and BE's read "
            "from IEX, the only tape there is at 9:35 on a basic plan; BE's also with "
            "VST's opening move, its lead AI-power peer; NVDA's with SPY's opening, the "
            "pre-market to 9:19 and the previous evening's after-hours; QCOM's, ORCL's "
            "and META's also with the opening's trade counts and VWAP, and ORCL's with "
            "SPY and its cloud peers MSFT, AMZN and GOOGL; AMGN's with SPY, the "
            "pre-market, the previous evening, the opening's trade counts and its "
            "biotech peers GILD, REGN and VRTX; TMO's with its life-science tools peers "
            "DHR, A and WAT; AMD's with SPY, the previous evening and its chip peers "
            "NVDA, INTC and AVGO). On the "
            "129-session test window its error is 35% below "
            "TimeToChange3's on AAPL ($1.37 per extreme against $2.11) and 31% below on "
            "INTC; MU, BE and AVGO have no TimeToChange3 bundle, and beat that approach "
            "retrained on them by 25%, 33% and 34%, QCOM, ORCL and META by 43%, 31% "
            "and 29%, AMGN by 34%, TMO by 42% (AMD's notebook ran no such comparison). "
            "Only the forecast changes: the "
            "levels, exits and breach rules are the day-range strategy's, hung off "
            "this predicted high and predicted range."
        ),
        requires=(
            "PyTorch, LightGBM, scikit-learn and joblib, the weighted network "
            "checkpoints, and Alpaca credentials for ~150 sessions of SIP minute history "
            "(plus IEX openings on MU and BE, BE's peers VST, PLUG and XLU, on NVDA, "
            "ORCL, AMGN and AMD SPY's history, on ORCL its peers MSFT, AMZN and GOOGL, on "
            "AMGN GILD, REGN and VRTX, on TMO DHR, A and WAT, on AMD NVDA, INTC and "
            "AVGO, and on NVDA, QCOM, ORCL, "
            "META and AMGN this "
            "morning's SIP pre-market)"
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
            "training; and five similar mega-caps pooled in as training rows. LightGBM "
            "alone, three seeds. On AAPL's 9-session test window it misses each extreme by "
            "$1.17 (15.8% of the 14-day range), level with HighLow and slightly ahead. "
            "Only the forecast changes: the levels, exits and breach rules are the "
            "day-range strategy's."
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
    FASTFURIOUS_KEY: AppleModel(
        key=FASTFURIOUS_KEY,
        label="Fast&Furious model",
        summary=(
            "FinNotebooks' Fast&Furious forecast of the average momentum of the next 30 "
            "minutes, made at the close of every minute from 09:31 to 15:30 -- not a range "
            "made once in the morning. It reads only today's session as a live trader on a "
            "basic plan sees it (SIP to 15 minutes ago, IEX since): the return since the "
            "open, the gap, the distance to the session's high, low and VWAP, the opening "
            "range, realised volatility and volume against their norms, the signed-volume "
            "share, the time of day and this minute's recent history. LightGBM, Huber loss, "
            "three seeds; a second LightGBM pair forecasts the next two hours' high and low "
            "(with last night's option walls and implied move) for the stop. Its ranking "
            "edge is small -- a rank correlation of 0.03 with the next half hour on 2025 and "
            "on 2026, and -0.03 on its 24-session test month. It drives its own rules: buy "
            "when the forecast is in the top 5% of the last 20 sessions', a stop 1.5 × the "
            "forecast 2-hour drop under the fill, out 30 minutes after the last signal or "
            "at once on a sell signal. The notebook's verdict on those rules: out of sample "
            "no edge that luck does not also produce."
        ),
        requires=(
            "LightGBM, SciPy and joblib, Alpaca credentials for ~200 days of SIP and IEX "
            "minute bars and today's live ones, and (for the stop) last night's option "
            "tables: the contract list, ~4 months of daily option bars and Yahoo's ^IRX"
        ),
        strategy=STRATEGY_MOMENTUM,
        tickers=FASTFURIOUS_TICKERS,
        load=_load_fastfurious,
        path=_fastfurious_path,
    ),
    GOODMOMENTS_KEY: AppleModel(
        key=GOODMOMENTS_KEY,
        label="GoodMoments model",
        summary=(
            "FinNotebooks' GoodMoments forecast, made at the close of every minute: the "
            "probability that a later bar of the session trades at least 0.2 ADR above this "
            "bar's close. LightGBM on 27 features -- the clock and the variance still to come, "
            "where the price sits against today's high, low, VWAP and open, yesterday's candle, "
            "the 7- to 28-day averages and three volume-profile POCs, the session's order flow "
            "(signed-volume and close-location proxies), last night's put/call volume, implied "
            "move and dealer gamma, and the analysts' mean price target. On its 2026 test it "
            "ranks rises at an AUC of 0.685 (0.585 within the same half hour), ahead of every "
            "baseline, by a little. The same trees on the clock and today's volatility alone "
            "give a second probability, and the rules trade the difference -- the edge: buy when "
            "the model sees a rise 2 points likelier than time and volatility do, take +0.2 ADR, "
            "stop -0.75 ADR, otherwise hold to the close (notebook 4's design 2; its first "
            "design, on the probability itself, lost out of sample). The notebook's verdict on "
            "those rules: +$496 on 199 test sessions at $10,000 a trade, inside the placebo "
            "spread and below holding AAPL -- no edge that luck does not also produce."
        ),
        requires=(
            "LightGBM, SciPy and joblib, the saved model and its time-and-volatility twin, "
            "Alpaca credentials for ~130 days of SIP and IEX minute bars and today's live ones, "
            "the shared option cache (the contract list and daily option bars) and Yahoo's "
            "analyst price targets"
        ),
        strategy=STRATEGY_PROBABILITY,
        tickers=GOODMOMENTS_TICKERS,
        load=_load_goodmoments,
        path=_goodmoments_path,
        unavailable=_goodmoments_unavailable,
    ),
    HIGHLOW2_GOODMOMENTS_KEY: AppleModel(
        key=HIGHLOW2_GOODMOMENTS_KEY,
        label="HighLow2 + GoodMoments",
        summary=(
            "GoodMoments' rules with HighLow2 as a second vote on every buy after the opening "
            "window. In the session's first five minutes -- the bars HighLow2's forecast is made "
            "from, before it exists -- a buy is GoodMoments' alone: its edge over the "
            "time-and-volatility model at the entry threshold. From the 09:35 bar on a buy needs "
            "both: the edge, and the bar's close at or under the buy level HighLow2's day-range "
            "rules would rest under its predicted high, moved with that forecast whenever the "
            "session trades outside it. Every exit is GoodMoments': the take and the stop in "
            "ADRs from the fill, the pause after a stop, the optional sale on a fading edge, the "
            "closing flatten. Never measured as a pair: each model's own record is in its card."
        ),
        requires=(
            "everything GoodMoments needs (LightGBM, SciPy, both its saved files, ~130 days of "
            "Alpaca SIP and IEX minute bars, the option cache, Yahoo's analyst targets) and "
            "everything HighLow2 needs (its bundle, ~150 sessions of SIP and IEX history with "
            "extended hours)"
        ),
        strategy=STRATEGY_PROBABILITY,
        tickers=HIGHLOW2_GOODMOMENTS_TICKERS,
        load=_load_highlow2_goodmoments,
        path=_goodmoments_path,
        unavailable=_highlow2_goodmoments_unavailable,
        signal_of=GOODMOMENTS_KEY,
        levels_of=HIGHLOW2_KEY,
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


def levels_of(key: "str | None") -> "str | None":
    """The day-range model whose buy level the named hybrid's entries also
    need (`AppleModel.levels_of`), or None for a model that is its own
    forecast."""
    return get(key).levels_of


def signal_of(key: "str | None") -> "str | None":
    """The model whose signal and exits the named hybrid trades
    (`AppleModel.signal_of`), or None for a model that is its own forecast."""
    return get(key).signal_of


def keys_for_strategy(ticker: "str | None", rule_set: str) -> "list[str]":
    """`keys_for(ticker)` narrowed to the models that drive one rule set --
    what a caller written for one of them offers (Orchestra races day-range
    pairs only)."""
    return [key for key in keys_for(ticker) if MODELS[key].strategy == rule_set]
