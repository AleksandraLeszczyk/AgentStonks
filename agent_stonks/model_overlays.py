"""What the saved models predict, in the shape a price chart can draw.

The ML surface of this app is a set of models that answer questions about a
session -- where its volume will trade, how wide it will be. Each of them already has a consumer (`profile_model`
feeds the profile curve, `apple_models` feeds the traders), but until now the
only way to see what a model said was to read a trader's log. This module is
the other way round: it asks each model its question about a session and
returns the answer as **drawing instructions**, so the live chart and SimLab's
replay chart can show the prediction beside the tape that tested it.

Ten overlays
------------
`day_range`          TimeToChange3's forecast of where the session's high and
                     low will land, made once from the first five minutes -- the
                     model's claim about the *width* of the day. Two price levels
                     and the band between them while the tape stays inside that
                     claim; once it trades outside, the forecast is revised for
                     the rest of the session and the pair is drawn stepped, off
                     the same walk `trader_levels` uses. See `dayrange_model`.
`profile_range`      the LevelsML density model's predicted price profile,
                     reduced to the three numbers a price axis can carry: its
                     outer quantiles and its point of control. The curve itself
                     is drawn separately by `charts._plot_price_distribution`;
                     this is the same prediction as horizontal levels, so it can
                     be read against the candles rather than only against the
                     histogram.
`intraday_range`     IntradayVolatility's time-of-day volatility curve as a
                     price envelope around the open, scaled to that model's own
                     daily-bar forecast of the day's range. See
                     `intraday_vol_model`, including why that forecast is weak.
`intraday_dayrange`  the same curve stretched so its peak is TimeToChange3's
                     predicted high and its trough the predicted low. The one
                     forecast is shared with `day_range` within a call.
`highlow_range`      HighLow's forecast of the same two numbers, drawn exactly
                     like `day_range` (flat, or stepped once the session
                     breaks it). See `highlow_model`.
`highlow2_range`     HighLow2's, drawn the same way. See `highlow2_model`.
`highlow3m_range`    HighLow_3m's forecast of the range *after* 9:33, drawn the
                     same way from the 09:32 bar on; the first three minutes'
                     own extremes are not a breach of it. See `highlow3m_model`.
`fastfurious_signal` Fast&Furious's forecast of the next 30 minutes' average
                     momentum, read at the close of every bar: a line in bps
                     per minute on the momentum panel, drawn 30 minutes
                     later over the bars it forecast, with the entry
                     threshold either side of zero, and a marker on the
                     candles at the bar where it crosses one (▲ a buy
                     signal, ▼ a sell signal). See `fastfurious_model`.
`fastfurious_range`  The next two hours' predicted high and low from each
                     bar's price, as a band: the second model in the bundle,
                     which only sizes the stop. Its own key so that it is off
                     unless asked for.
`goodmoments_signal` GoodMoments' edge, read at the close of every bar: its
                     probability that a later bar of the session trades 0.2 ADR
                     above this bar's close, less the same trees' probability
                     on the clock and volatility alone -- a line in percentage
                     points on the momentum panel with the entry threshold,
                     and a ▲ on the candles where it first reaches it. Drawn
                     at the bar it was made at: its horizon is the close. See
                     `goodmoments_model`.
`trader_levels`      where Apple Trader would rest its buy and its sell, from
                     the forecast of the model its configuration names. The odd one out: every other overlay
                     draws what a *model* said, this one draws what an *agent*
                     would do about it, so it takes a configuration and asks
                     `apple_trader.session_levels` rather than deriving the
                     levels itself. Under Fast&Furious, which rests no levels,
                     it is the running agent's stop; under GoodMoments its take
                     and its stop.

Both envelopes are widest at 09:30, narrow to roughly a fifth of that by
midday and open again into the close. They are a picture of *how far the day
usually swings at this time*, scaled to a forecast's extremes -- not a coverage
band, and price routinely leaves the midday part of it.

The four item kinds
-------------------
A prediction is about a price, a moment, a stretch of time, or a range that
changes through the day, and the chart draws each one differently. So
`compute` returns a flat list of items, each of which is one of:

    level   a price with no time extent      -> a horizontal line, drawn in the
                                                candle chart AND in the price
                                                profile beside it, since both
                                                share the price axis.
    event   a moment with no price extent    -> a vertical line plus an icon.
    span    a stretch of time, optionally    -> a semi-transparent background,
            bounded in price                    behind the candles.
    band    an upper and a lower price per   -> two edges with the range
            timestamp                           between them tinted, behind the
                                                candles; no profile mirror, as
                                                a curve has no single price.
    series  a value that is not a price,     -> a line on a value axis of its
            per timestamp, in a unit of its     own over the momentum panel; not
            own                                 drawn where there is none.

Nothing here knows about plotly: `charts.add_model_overlays` is the only
renderer, and SimLab draws the same items into a different figure. Nothing here
knows about Streamlit either, so the same call serves the live buffer and a
replay of a day that ended months ago -- the caller supplies the bars.

Point-in-time honesty
---------------------
An overlay drawn on a past session must show what the model *could have said
that morning*, not what it would say knowing how the day ended. Both models
that see daily history are given completed days only, and today's row is
assembled from the opening window exactly as the traders assemble it
(`dayrange_model.session_daily_frame` is the shared seam).

Every model is optional in the same way it is everywhere else: a missing file
or a missing dependency produces a note explaining why an overlay is empty,
never an exception and never a fabricated line.
"""

from __future__ import annotations

import importlib
from dataclasses import astuple, dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from . import (
    apple_models, clock, event_days, historical, intraday_vol_model, market_hours,
    momentum_regime, profile_model,
)
from .config import MODEL_OVERLAY_COLORS


# --- the catalogue ----------------------------------------------------------

DAY_RANGE_KEY = "day_range"
PROFILE_RANGE_KEY = "profile_range"
INTRADAY_RANGE_KEY = "intraday_range"
INTRADAY_DAYRANGE_KEY = "intraday_dayrange"
TRADER_LEVELS_KEY = "trader_levels"
HIGHLOW_RANGE_KEY = "highlow_range"
HIGHLOW2_RANGE_KEY = "highlow2_range"
HIGHLOW3M_RANGE_KEY = "highlow3m_range"
FASTFURIOUS_SIGNAL_KEY = "fastfurious_signal"
FASTFURIOUS_RANGE_KEY = "fastfurious_range"
GOODMOMENTS_SIGNAL_KEY = "goodmoments_signal"


@dataclass(frozen=True)
class ModelOverlay:
    """One model's prediction, as something a chart can offer to draw."""

    key: str
    label: str
    # Just the model's name, for the live chart's picker. `label` also says
    # what is drawn, and is what legends and messages show.
    name: str
    # One line for a picker: what the overlay shows, not how well it scores.
    summary: str
    # What has to be installed (or trained) for it to produce anything.
    requires: str
    # The symbols the underlying model was fitted on, or None for one that
    # generalises. `profile_range` is the only None: LevelsML's pack was
    # deliberately trained without ticker dummies so it transfers.
    tickers: "tuple[str, ...] | None"
    # The `apple_models` keys whose predictions this overlay draws -- the seam
    # that lets a caller holding a run's configuration ask "show me what *that*
    # model said" without knowing how overlays are carved up. Empty for
    # `profile_range`, whose pack is not in that registry and drives no agent:
    # nothing can name it, so nothing auto-selects it.
    models: "tuple[str, ...]" = ()

    def covers(self, ticker: "str | None") -> bool:
        if self.tickers is None:
            return True
        return (ticker or "").upper() in self.tickers

    def draws(self, model_key: "str | None") -> bool:
        """Whether this overlay is a picture of what the named model predicts."""
        return (model_key or "") in self.models


OVERLAYS: "dict[str, ModelOverlay]" = {
    DAY_RANGE_KEY: ModelOverlay(
        key=DAY_RANGE_KEY,
        label="Predicted day range (TimeToChange3)",
        name="TimeToChange3",
        summary=(
            "TimeToChange3's forecast of the session's high and low, made once from "
            "the first five minutes. Two levels and the band between them while the "
            "session stays inside it, and a stepped pair once it trades outside and "
            "the forecast is revised."
        ),
        requires="PyTorch, LightGBM and the day-range bundle",
        tickers=apple_models.DAYRANGE_TICKERS,
        # Both models forecast the day range; the pairing then reads it through
        # the intraday shape, which is the other overlay below.
        models=(apple_models.DAYRANGE_KEY, apple_models.DAYRANGE_INTRADAY_KEY),
    ),
    PROFILE_RANGE_KEY: ModelOverlay(
        key=PROFILE_RANGE_KEY,
        label="Predicted price profile range (LevelsML)",
        name="LevelsML",
        summary=(
            "The LevelsML density model's outer quantiles and point of control, as "
            "price levels. The same prediction the profile curve draws, on the "
            "price axis."
        ),
        requires="LightGBM and the open-profile pack",
        tickers=None,
    ),
    INTRADAY_RANGE_KEY: ModelOverlay(
        key=INTRADAY_RANGE_KEY,
        label="Predicted intraday range (IntradayVolatility)",
        name="IntradayVolatility",
        summary=(
            "IntradayVolatility's time-of-day volatility curve around the open, scaled "
            "to its own forecast of the day's range, as a shaded envelope: widest at "
            "09:30, narrowest at midday, opening up again into the close. Known at the "
            "open."
        ),
        requires="the IntradayVolatility export (intravol_<TICKER>.json)",
        tickers=intraday_vol_model.TICKERS,
    ),
    INTRADAY_DAYRANGE_KEY: ModelOverlay(
        key=INTRADAY_DAYRANGE_KEY,
        label="Predicted intraday range × day range (IntradayVolatility × TimeToChange3)",
        name="IntradayVolatility × TimeToChange3",
        summary=(
            "The same time-of-day curve, stretched so it tops out at TimeToChange3's "
            "predicted high and bottoms out at its predicted low. Made at 09:35."
        ),
        requires=(
            "the IntradayVolatility export plus PyTorch, LightGBM and the day-range bundle"
        ),
        tickers=apple_models.DAYRANGE_INTRADAY_TICKERS,
        # This curve *is* what "Day Range × Intraday Volatility" measures its
        # levels below, so a run on that model opens showing it.
        models=(apple_models.DAYRANGE_INTRADAY_KEY,),
    ),
    HIGHLOW_RANGE_KEY: ModelOverlay(
        key=HIGHLOW_RANGE_KEY,
        label="Predicted day range (HighLow)",
        name="HighLow",
        summary=(
            "HighLow's forecast of the session's high and low, made once from the "
            "first five minutes and anchored on the 9:35 price. Drawn like the "
            "TimeToChange3 range: flat while the session stays inside it, stepped "
            "once it trades outside."
        ),
        requires="PyTorch, LightGBM, the HighLow bundle and Alpaca SIP minute history",
        tickers=apple_models.HIGHLOW_TICKERS,
        # Fast&Furious + HighLow hangs its levels off this forecast from 9:35.
        models=(apple_models.HIGHLOW_KEY, apple_models.FF_HIGHLOW_KEY),
    ),
    HIGHLOW2_RANGE_KEY: ModelOverlay(
        key=HIGHLOW2_RANGE_KEY,
        label="Predicted day range (HighLow2)",
        name="HighLow2",
        summary=(
            "HighLow2's forecast of the session's high and low, made once at 9:35 from "
            "IEX's first five minutes and the pre-market. Drawn like the other ranges: "
            "flat while the session stays inside it, stepped once it trades outside. "
            "On INTC it is the forecast of the high and low *after* 9:35 -- the pair "
            "Apple Trader's levels hang off there -- so the first five minutes can sit "
            "outside it and are not a breach of it."
        ),
        requires=(
            "LightGBM, the HighLow2 bundle and Alpaca SIP and IEX minute history "
            "(extended hours)"
        ),
        tickers=apple_models.HIGHLOW2_TICKERS,
        models=(apple_models.HIGHLOW2_KEY,),
    ),
    HIGHLOW3M_RANGE_KEY: ModelOverlay(
        key=HIGHLOW3M_RANGE_KEY,
        label="Predicted rest-of-day range (HighLow_3m)",
        name="HighLow_3m",
        summary=(
            "HighLow_3m's forecast of how high and how low the price trades from 9:33 to "
            "the close, made once at 9:33 from IEX's first three minutes and last night's "
            "option positioning. Not the whole day's range: the first three minutes can "
            "sit outside it. Drawn like the other ranges, stepped once the tape after "
            "9:33 trades outside."
        ),
        requires=(
            "LightGBM, SciPy, the HighLow_3m bundle, Alpaca SIP and IEX minute history, "
            "Alpaca's option contracts and daily option bars, and Yahoo's ^IRX"
        ),
        tickers=apple_models.HIGHLOW3M_TICKERS,
        models=(apple_models.HIGHLOW3M_KEY,),
    ),
    FASTFURIOUS_SIGNAL_KEY: ModelOverlay(
        key=FASTFURIOUS_SIGNAL_KEY,
        label="Fast&Furious 30-min momentum forecast (buy/sell signals)",
        name="Fast&Furious",
        summary=(
            "Fast&Furious's forecast of the next 30 minutes' average momentum, made at the "
            "close of every bar from 09:31 to 15:30: a yellow line in bps per minute on the "
            "momentum panel (right axis), drawn 30 minutes after it was made — over the bars it "
            "forecast — with the entry threshold above and below zero, and a marker on the "
            "candles at the bar where it crosses one — ▲ a buy signal, ▼ a sell signal."
        ),
        requires="LightGBM, the Fast&Furious bundle, and Alpaca SIP and IEX minute bars",
        tickers=apple_models.FASTFURIOUS_TICKERS,
        # Fast&Furious + HighLow buys on these signals before 9:35.
        models=(apple_models.FASTFURIOUS_KEY, apple_models.FF_HIGHLOW_KEY),
    ),
    FASTFURIOUS_RANGE_KEY: ModelOverlay(
        key=FASTFURIOUS_RANGE_KEY,
        label="Fast&Furious 2-hour range (the stop's forecast)",
        name="Fast&Furious 2-hour range",
        summary=(
            "The second model in the Fast&Furious bundle: the next two hours' high and low "
            "from each bar's price, as a band. Not the momentum forecast the trades are made "
            "on — the agent only hangs its stop off the predicted low."
        ),
        requires=(
            "LightGBM, the Fast&Furious bundle, Alpaca SIP and IEX minute bars, and "
            "HighLow_3m's option tables"
        ),
        tickers=apple_models.FASTFURIOUS_TICKERS,
        # Deliberately not `models=(FASTFURIOUS_KEY,)`: `for_models` pre-selects
        # what a run's model said, and drawn by default this band reads as the
        # model's forecast when it is only the stop's input.
        models=(),
    ),
    GOODMOMENTS_SIGNAL_KEY: ModelOverlay(
        key=GOODMOMENTS_SIGNAL_KEY,
        label="GoodMoments edge (+0.2 ADR before the close; buy signals)",
        name="GoodMoments",
        summary=(
            "GoodMoments' probability, made at the close of every bar, that a later bar of the "
            "session trades 0.2 ADR above that bar's close, less the same trees' probability on "
            "the clock and today's volatility alone: the edge, in percentage points, as a lime "
            "line on the momentum panel (right axis) with the entry threshold, and a ▲ on the "
            "candles at the bar where it first reaches it. The hover shows both probabilities."
        ),
        requires=(
            "LightGBM, SciPy, the GoodMoments bundle and its time-and-volatility twin, Alpaca SIP "
            "and IEX minute bars, HighLow_3m's option cache and Yahoo's analyst targets"
        ),
        tickers=apple_models.GOODMOMENTS_TICKERS,
        models=(apple_models.GOODMOMENTS_KEY,),
    ),
    TRADER_LEVELS_KEY: ModelOverlay(
        key=TRADER_LEVELS_KEY,
        label="Apple Trader buy/sell levels (agent orders, not a forecast)",
        name="Apple Trader",
        summary=(
            "Where the configured Apple Trader rests its two orders: the buy and the "
            "sell, each a distance in ADRs under the day-range forecast, following "
            "whatever settings the Apple Trader form is holding. The only overlay here "
            "that draws an agent's orders rather than a model's answer."
        ),
        requires="PyTorch, LightGBM and the forecast bundle of the model the agent runs",
        # Every symbol some Apple Trader model covers: the levels hang off
        # whichever forecast the configured model makes (HighLow alone on MU).
        tickers=tuple(apple_models.tickers()),
        # Deliberately not `models=(DAYRANGE_KEY,)`, though it is built on that
        # forecast: `for_models` pre-selects what a *model* said, and these are
        # one agent's orders, not a forecast of the day.
        models=(),
    ),
}


def keys() -> "list[str]":
    """Every overlay key, in the order a picker should offer them."""
    return list(OVERLAYS)


def keys_for(ticker: "str | None") -> "list[str]":
    """The overlays that exist for one symbol, in picker order.

    A symbol nothing was fitted on still gets `profile_range`, which is the
    only model here that claims to transfer.
    """
    return [key for key, overlay in OVERLAYS.items() if overlay.covers(ticker)]


def get(key: "str | None") -> "ModelOverlay | None":
    return OVERLAYS.get(key or "")


def label(key: "str | None") -> str:
    overlay = get(key)
    return overlay.label if overlay else str(key)


def name(key: "str | None") -> str:
    overlay = get(key)
    return overlay.name if overlay else str(key)


def for_models(
    model_keys: "list[str] | tuple[str, ...] | None", ticker: "str | None" = None
) -> dict:
    """The overlay selection that draws what these saved models predicted.

    The question a chart of a *past* run asks is not "which overlays exist"
    but "what did the model that made these trades say about this tape" --
    so a caller that knows which `apple_models` bundles a run loaded (SimLab
    reads it off the stored configuration; see `simlab.results.ml_models`) can
    turn that into a selection here instead of hard-coding the mapping at the
    call site. Which model drove which overlay is this catalogue's business.

    Returns `{"keys": [...], "unmatched": [...]}`:

    `keys`         overlays to draw, in picker order. Named a `ticker` they are
                   filtered to the ones that exist for it -- a run's model
                   always covers the symbol it traded, but the caller may be
                   looking at another tab of the same run, where the picker has
                   no such option to select. `None` asks the question without a
                   symbol and filters nothing.
    `unmatched`    models the run used that no overlay draws. Not an error and
                   not silence either: a model whose answer is neither a level,
                   a moment nor a span has nothing honest to draw -- and a
                   caller that pre-selected nothing should be able to say why.
    """
    named = [str(k) for k in (model_keys or []) if k]
    keys = [
        key for key, overlay in OVERLAYS.items()
        if any(overlay.draws(m) for m in named)
        and (ticker is None or overlay.covers(ticker))
    ]
    unmatched = [
        m for m in named
        if not any(overlay.draws(m) for overlay in OVERLAYS.values())
    ]
    return {"keys": keys, "unmatched": unmatched}


# --- item constructors ------------------------------------------------------
#
# The renderer's contract, in one place. Every item carries the overlay `key`
# it came from so a chart can group or filter them, and a `color` so the
# renderer never has to know which model produced what.


def _level(key: str, label_: str, value: float, color: str, dash: str = "dash",
           note: str = "", x0=None, x1=None) -> dict:
    """A predicted price.

    `x0`/`x1` bound the line to the session it was predicted for. They are
    optional because on a single-day chart the answer is the whole axis and
    saying so is noise -- but SimLab replays several days into one figure, and
    five days of predicted highs drawn across all five would each claim to be
    about days they were not.
    """
    return {
        "kind": "level",
        "key": key,
        "label": label_,
        "value": float(value),
        "color": color,
        "dash": dash,
        "note": note,
        "x0": None if x0 is None else _iso(x0),
        "x1": None if x1 is None else _iso(x1),
    }


def _event(key: str, label_: str, ts, color: str, icon: str = "◆",
           price: "float | None" = None, dash: str = "dot", note: str = "",
           forward: bool = False, line: bool = True) -> dict:
    """A moment worth marking.

    `line` decides whether the moment also gets a vertical rule through the
    plot, and it is what separates the model's claims from their context. A
    regime change that has already happened is a fact about the tape and is
    marked with its icon alone; a change the model says will *hold*, or a turn
    it says is coming, is a prediction and gets the rule -- which on a chart
    covering several sessions is the difference between a readable picture and
    forty vertical lines.
    """
    return {
        "kind": "event",
        "key": key,
        "label": label_,
        # What the legend calls the whole set. Each event's own `label` is
        # about that one moment ("-> positive 74%"), which is the wrong thing
        # to name a legend entry that hides or shows every mark at once.
        "group": OVERLAYS[key].label if key in OVERLAYS else key,
        "ts": _iso(ts),
        "color": color,
        "icon": icon,
        "price": None if price is None else float(price),
        "dash": dash,
        "note": note,
        "forward": forward,
        "line": line,
    }


def _span(key: str, label_: str, x0, x1, color: str,
          y0: "float | None" = None, y1: "float | None" = None,
          note: str = "", forward: bool = False) -> dict:
    """A stretch of time the prediction covers.

    `y0`/`y1` bound it in price as well -- a predicted *range* over a period is
    a rectangle, and a predicted *moment lasting n bars* is a full-height
    column. Both are drawn semi-transparent and behind the candles.

    `forward` marks a span whose point is that it reaches past the newest bar:
    a window the model says the next n bars will do something. The renderer
    widens the time axis for those and clamps every other span to the tape,
    so a claim about 16:00 made at 09:35 does not squash a two-hour chart into
    a corner.
    """
    return {
        "kind": "span",
        "key": key,
        "label": label_,
        "x0": _iso(x0),
        "x1": _iso(x1),
        "y0": None if y0 is None else float(y0),
        "y1": None if y1 is None else float(y1),
        "color": color,
        "note": note,
        "forward": forward,
    }


def _band(key: str, label_: str, ts, lower, upper, color: str,
          dash: str = "dot", note: str = "",
          lower_label: str = "", upper_label: str = "",
          step: bool = False) -> dict:
    """A price range that changes through the session.

    Where a `span` is one rectangle, a band is a pair of curves over the same
    timestamps -- a prediction of how wide the price can swing *at each time of
    day*. It is drawn as two edges with the range between them tinted, behind
    the candles, and like any non-forward span it is clipped to the bars in
    hand. There is no profile mirror: a curve through time has no single price
    to put on that axis.

    `lower_label` and `upper_label` name the two edges for the hover, and are
    what a band that *replaces* a pair of levels owes its reader: the flat
    shape says "Pred. high" and "Pred. low" on the two lines, and a moving
    shape that could only say "upper" and "lower" would have lost something in
    the switch. They default to that generic pair, which is right for a band
    whose edges have no names of their own -- a volatility envelope's do not.

    `step` draws each value as holding until the next timestamp instead of
    sloping into it. It is for a forecast or a level that is *revised* at a
    bar rather than one that varies continuously: the revision happened at
    that candle, and a one-minute diagonal into it would blur exactly the
    moment the reader is looking for.
    """
    stamps = pd.DatetimeIndex(ts)
    if stamps.tz is None:
        stamps = stamps.tz_localize(market_hours.MARKET_TZ)
    return {
        "kind": "band",
        "key": key,
        "label": label_,
        "group": OVERLAYS[key].label if key in OVERLAYS else key,
        "t": [stamp.isoformat() for stamp in stamps.tz_convert("UTC")],
        "lower": [float(v) for v in lower],
        "upper": [float(v) for v in upper],
        "lower_label": lower_label or f"{label_} lower",
        "upper_label": upper_label or f"{label_} upper",
        "color": color,
        "dash": dash,
        "note": note,
        "step": step,
        "forward": False,
    }


def _path(key: str, label_: str, ts, values, color: str,
          dash: str = "dot", note: str = "", step: bool = False) -> dict:
    """A single price that changes through the session.

    A `level` that moves: one curve over the timestamps, with nothing tinted
    either side of it. For a line that belongs beside a band rather than inside
    it -- Apple Trader's stop follows its buy level, but the gap between the two
    is not a range anyone predicted, and tinting it would say it was. Clipped
    like a band, and shares its overlay's legend entry. `step` as for a band.
    A None value is a gap: the line is not drawn there.
    """
    stamps = pd.DatetimeIndex(ts)
    if stamps.tz is None:
        stamps = stamps.tz_localize(market_hours.MARKET_TZ)
    return {
        "kind": "path",
        "key": key,
        "label": label_,
        "group": OVERLAYS[key].label if key in OVERLAYS else key,
        "t": [stamp.isoformat() for stamp in stamps.tz_convert("UTC")],
        "values": [float("nan") if v is None else float(v) for v in values],
        "color": color,
        "dash": dash,
        "note": note,
        "step": step,
        "forward": False,
    }


def _series(key: str, label_: str, ts, values, color: str, unit: str,
            dash: str = "solid", width: float = 1.8, note: str = "",
            text: "list[str] | None" = None) -> dict:
    """A forecast that is not a price, through the session.

    Drawn as a line on the momentum panel against a value axis of its own,
    labelled `unit`, rather than on the price axis: Fast&Furious's momentum
    forecast is a few hundredths of a bp per minute, which on the candles' axis
    -- or the momentum bars' dollars -- is a flat line at zero. The renderer
    centres that axis on zero with the bars', so above zero is above zero on
    both. Every series of a chart shares the one axis and the first one's
    `unit`. `text` is a hover line per timestamp. A chart without a momentum
    panel does not draw it.
    """
    stamps = pd.DatetimeIndex(ts)
    if stamps.tz is None:
        stamps = stamps.tz_localize(market_hours.MARKET_TZ)
    return {
        "kind": "series",
        "key": key,
        "label": label_,
        "group": OVERLAYS[key].label if key in OVERLAYS else key,
        "t": [stamp.isoformat() for stamp in stamps.tz_convert("UTC")],
        "values": [float("nan") if v is None else float(v) for v in values],
        "unit": unit,
        "color": color,
        "dash": dash,
        "width": width,
        "note": note,
        "text": None if text is None else [str(t) for t in text],
        "forward": False,
    }


def _iso(ts) -> str:
    """A timestamp in the one form every consumer here accepts.

    UTC ISO-8601, because the chart's x axis is built from `pd.to_datetime(...,
    utc=True)` and SimLab stores bar timestamps as strings. Naive inputs are
    read as exchange-local, which is what the momentum frame's index is.
    """
    stamp = pd.Timestamp(ts)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize(market_hours.MARKET_TZ)
    return stamp.tz_convert("UTC").isoformat()


# --- the entry point --------------------------------------------------------


def compute(
    overlay_keys: "list[str] | tuple[str, ...] | None",
    ticker: str,
    bars: "list[dict]",
    daily_bars: "list[dict] | None" = None,
    session_date=None,
    open_price: "float | None" = None,
    dayrange_daily_bars: "list[dict] | None" = None,
    trader_config=None,
    credentials: "tuple[str, str] | None" = None,
    trader_history: "dict | None" = None,
) -> dict:
    """Draw instructions for the requested overlays, plus why any are empty.

    `bars` are the session's 1-minute bars (`{"t","o","h","l","c","v"}`, the
    shape both the live buffer and SimLab's store carry) and `daily_bars` the
    daily history behind it -- rows dated on or after the session are ignored
    by the models that read it, so a caller may pass the whole store.

    `session_date` names the day being drawn; it defaults to the last bar's,
    which is what a live caller wants and what a single-day replay wants too.
    `open_price` is the official opening print when the caller has one.
    `dayrange_daily_bars` is the day-range forecast's own daily history, for a
    caller whose `daily_bars` cannot serve it (see `live_overlays`); it
    defaults to `daily_bars`.

    `trader_config` is the `AppleTraderConfig` whose orders `trader_levels`
    draws. None means the shipped configuration for this instrument, which is
    the honest default: a chart with nothing configured should show what the
    agent would do if started now, not the last symbol's numbers. A caller with
    a real one -- the live form's, or a stored run's -- passes it, and the
    levels drawn are then that run's rather than a plausible set. The levels
    hang off the forecast of the model that configuration names, so a HighLow
    run draws its levels under HighLow's range.

    `credentials` are the Alpaca key and secret the HighLow forecast reads its
    SIP history with; None falls back to the environment's.

    `trader_history` is a live Apple Trader's own record of what it rested
    (`AppState.apple_trader_levels`). When it is for this symbol and this day,
    `trader_levels` and the day-range overlay of the model it runs draw it
    instead of walking the session under `trader_config`: the record carries
    the forecast the run actually made, every sidebar edit at the minute it was
    adopted, and any restart, none of which a walk can know. It is what makes
    the chart quote the same levels as the agent's log.

    Returns `{"items": [...], "notes": [...]}`. A note is a sentence naming an
    overlay that produced nothing and saying what would fix it; there is never
    a partial answer presented as a whole one.
    """
    wanted = [k for k in (overlay_keys or []) if k in OVERLAYS]
    items: "list[dict]" = []
    notes: "list[str]" = []
    if not wanted or not bars:
        return {"items": items, "notes": notes}

    symbol = (ticker or "").upper()
    day, session, problem = _session_of(bars, session_date)
    if session is None:
        return {"items": items, "notes": [problem]}

    # Asked for at most once per call, however many overlays draw it.
    memo: dict = {}
    dayrange_history = (
        (daily_bars or []) if dayrange_daily_bars is None else dayrange_daily_bars
    )

    def day_range_forecast() -> dict:
        if "result" not in memo:
            memo["result"] = _day_range_forecast(
                symbol, session, dayrange_history, day, open_price
            )
        return memo["result"]

    def highlow_forecast(model_key: str = apple_models.HIGHLOW_KEY) -> dict:
        if model_key not in memo:
            memo[model_key] = _highlow_forecast(symbol, session, day, credentials, model_key)
        return memo[model_key]

    def fastfurious_forecast() -> dict:
        if "fastfurious" not in memo:
            memo["fastfurious"] = _fastfurious_forecast(
                symbol, session, day, credentials, trader_config
            )
        return memo["fastfurious"]

    def goodmoments_forecast() -> dict:
        if "goodmoments" not in memo:
            memo["goodmoments"] = _goodmoments_forecast(
                symbol, session, day, credentials, trader_config
            )
        return memo["goodmoments"]

    def trader_forecast() -> dict:
        model_key = apple_models.forecast_key(trader_model(trader_config, symbol))
        if model_key in _HIGHLOW_MODULES:
            return highlow_forecast(model_key)
        return day_range_forecast()

    recorded = _recorded_levels(trader_history, symbol, day)

    def range_items(key: str, forecast) -> "tuple[list[dict], str]":
        # Only the minutes the run spent on this overlay's model: a run switched
        # from one model to another mid-session drew each range for its part.
        own = [] if recorded is None else [
            row for row in recorded[0]
            if OVERLAYS[key].draws(row.get("model_key") or recorded[1].model_key)
        ]
        if own and OVERLAYS[key].draws(recorded[1].model_key):
            walked = own + _after_record(own, trader_history, session, recorded[1])
            return _day_range_items(_recorded_result(own), day, symbol, key=key, walked=walked)
        return _day_range_items(
            forecast(), day, symbol, session,
            open_price if key == DAY_RANGE_KEY else None, trader_config, key=key,
        )

    def trader_items() -> "tuple[list[dict], str]":
        model_key = trader_model(trader_config, symbol)
        rules = apple_models.strategy(model_key)
        if rules != apple_models.STRATEGY_DAYRANGE:
            # Only a record the same rules wrote: a day-range run's earlier
            # today is not this configuration's stop.
            own = recorded is not None and apple_models.strategy(
                getattr(recorded[1], "model_key", None)
            ) == rules
            return _signal_trader_items(symbol, recorded if own else None, model_key)
        if recorded is not None:
            rows, config = recorded
            return _trader_levels_items(
                symbol, session, day, open_price, _recorded_result(rows), config,
                levels=rows,
            )
        sat_out = _sat_out(symbol, day, trader_config)
        if sat_out:
            return [], sat_out
        return _trader_levels_items(
            symbol, session, day, open_price, trader_forecast(), trader_config
        )

    builders = {
        DAY_RANGE_KEY: lambda: range_items(DAY_RANGE_KEY, day_range_forecast),
        PROFILE_RANGE_KEY: lambda: _profile_range_items(
            session, daily_bars or [], day
        ),
        INTRADAY_RANGE_KEY: lambda: _intraday_range_items(
            symbol, session, daily_bars or [], day, open_price
        ),
        INTRADAY_DAYRANGE_KEY: lambda: _intraday_dayrange_items(
            symbol, session, day, open_price, day_range_forecast()
        ),
        HIGHLOW_RANGE_KEY: lambda: range_items(HIGHLOW_RANGE_KEY, highlow_forecast),
        HIGHLOW2_RANGE_KEY: lambda: range_items(
            HIGHLOW2_RANGE_KEY, lambda: highlow_forecast(apple_models.HIGHLOW2_KEY)
        ),
        HIGHLOW3M_RANGE_KEY: lambda: range_items(
            HIGHLOW3M_RANGE_KEY, lambda: highlow_forecast(apple_models.HIGHLOW3M_KEY)
        ),
        FASTFURIOUS_SIGNAL_KEY: lambda: _fastfurious_signal_items(fastfurious_forecast()),
        FASTFURIOUS_RANGE_KEY: lambda: _fastfurious_range_items(fastfurious_forecast()),
        GOODMOMENTS_SIGNAL_KEY: lambda: _goodmoments_signal_items(goodmoments_forecast()),
        TRADER_LEVELS_KEY: trader_items,
    }

    for key in wanted:
        overlay = OVERLAYS[key]
        if not overlay.covers(symbol):
            notes.append(
                f"{overlay.label}: not available for {symbol} — the model was fitted "
                f"on {', '.join(overlay.tickers or ())} only."
            )
            continue
        try:
            produced, why = builders[key]()
        except Exception as exc:  # a decoration must never take the chart down
            produced, why = [], f"{overlay.label}: {exc}"
        items.extend(produced)
        if why:
            notes.append(why)

    return {"items": items, "notes": notes}


def _session_of(
    bars: "list[dict]", session_date=None
) -> "tuple[pd.Timestamp | None, pd.DataFrame | None, str]":
    """`(day, session frame, "")` for the regular session of `session_date` in
    `bars` -- the last bar's day when None -- or `(day, None, why not)`."""
    frame = momentum_regime.frame_from_bars(bars)
    if not len(frame):
        return None, None, "No regular-session bars to draw on."
    day = pd.Timestamp(session_date).normalize() if session_date is not None else None
    if day is None:
        day = pd.Timestamp(frame.index[-1]).normalize().tz_localize(None)
    session = frame[frame.index.normalize().tz_localize(None) == day]
    if not len(session):
        return day, None, f"No bars for {day.date()}."
    return day, session, ""


# The range overlay that draws each model's forecast; every other model (the
# two TimeToChange3 ones) forecasts through TimeToChange3's.
_RANGE_OVERLAYS = {
    apple_models.HIGHLOW_KEY: HIGHLOW_RANGE_KEY,
    apple_models.HIGHLOW2_KEY: HIGHLOW2_RANGE_KEY,
    apple_models.HIGHLOW3M_KEY: HIGHLOW3M_RANGE_KEY,
}


def session_forecast(
    model_key: "str | None",
    ticker: str,
    bars: "list[dict]",
    daily_bars: "list[dict] | None" = None,
    session_date=None,
    open_price: "float | None" = None,
    credentials: "tuple[str, str] | None" = None,
) -> dict:
    """The morning forecast of the session's high and low that an Apple Trader
    on `model_key` hangs its levels off, as numbers rather than drawing items.

    The same forecast `trader_levels` draws under, off the same inputs as
    `compute` (whose arguments these are), for a chart that wants the two
    numbers themselves -- SimLab's Tuning tab draws one per session over daily
    candles. Returns `{"forecast", "made_at", "problem", "overlay"}`: the first
    three as `_day_range_forecast` returns them, `overlay` the key of the range
    overlay that draws this model's forecast, for its label and colour.
    """
    symbol = (ticker or "").upper()
    # A combined model's levels hang off another model's forecast.
    model_key = apple_models.forecast_key(model_key) or model_key
    overlay = _RANGE_OVERLAYS.get(model_key or "", DAY_RANGE_KEY)
    day, session, problem = _session_of(bars or [], session_date)
    if apple_models.strategy(model_key) != apple_models.STRATEGY_DAYRANGE:
        # Its forecast is made every minute and is not a range of the day.
        label = apple_models.get(model_key).label
        what = (
            "the next 30 minutes" if apple_models.strategy(model_key) == apple_models.STRATEGY_MOMENTUM
            else "the chance of a +0.2 ADR rise before the close"
        )
        result = {"forecast": None, "made_at": None, "problem": (
            f"{label} makes no morning forecast of the day's range: it forecasts {what} at "
            "every bar."
        )}
        overlay = _SIGNAL_OVERLAYS.get(model_key, FASTFURIOUS_SIGNAL_KEY)
    elif session is None:
        result = {"forecast": None, "made_at": None, "problem": problem}
    elif overlay == DAY_RANGE_KEY:
        result = _day_range_forecast(symbol, session, daily_bars or [], day, open_price)
    else:
        result = _highlow_forecast(symbol, session, day, credentials, model_key)
    return {**result, "overlay": overlay}


# --- day range (TimeToChange3) ----------------------------------------------


def _session_close(day: pd.Timestamp) -> pd.Timestamp:
    """The 16:00 bell of `day`, exchange-local.

    A day-range forecast is a claim about the whole session, so its band runs
    to the close rather than to the last bar in hand -- which is the point on a
    live chart, where the last bar is 11:04 and the claim is about 16:00.
    """
    return pd.Timestamp(day).tz_localize(market_hours.MARKET_TZ) + timedelta(
        hours=market_hours.MARKET_CLOSE.hour, minutes=market_hours.MARKET_CLOSE.minute
    )


def _day_range_forecast(
    symbol: str,
    session: pd.DataFrame,
    daily_bars: "list[dict]",
    day: pd.Timestamp,
    open_price: "float | None",
) -> dict:
    """TimeToChange3's forecast for the session, or why there is none.

    Returns `{"forecast": dict | None, "made_at": Timestamp | None, "problem":
    str}`. Two overlays draw this one forecast -- the day range itself, and the
    intraday envelope stretched between its high and low -- so `compute` asks
    for it once per call and each overlay names itself in the note. Selecting
    both must not run three networks twice.
    """
    def failed(problem: str) -> dict:
        return {"forecast": None, "made_at": None, "problem": problem}

    try:
        bundle = apple_models.load(apple_models.DAYRANGE_KEY, symbol)
        if bundle is None:
            return failed(apple_models.unavailable_reason(apple_models.DAYRANGE_KEY, symbol))

        from . import dayrange_model  # heavy (torch + LightGBM); only once it is needed

        want = dayrange_model.opening_minutes(bundle)
        if len(session) < want:
            return failed(
                f"the forecast is built on the first {want} minutes and only "
                f"{len(session)} bars have closed."
            )
        opening = session.iloc[:want]
        if float(opening["minutes_from_open"].iloc[0]) >= 1.0:
            return failed(
                f"these bars start at {session.index[0]:%H:%M}, not the 09:30 open, so "
                "the first five minutes the forecast needs are not here."
            )
        history = dayrange_model.daily_frame_from_bars(daily_bars)
        forecast = dayrange_model.forecast_session(
            bundle, history, opening, day, open_price=open_price
        )
    except Exception as exc:  # a decoration must never take the chart down
        return failed(str(exc))
    return {"forecast": forecast, "made_at": opening.index[-1], "problem": ""}


# The module behind each HighLow model, imported only once its forecast is
# drawn (torch and LightGBM for HighLow, LightGBM for HighLow2 and HighLow_3m).
# All expose the same `opening_minutes` / `forecast_session` pair.
_HIGHLOW_MODULES = {
    apple_models.HIGHLOW_KEY: "highlow_model",
    apple_models.HIGHLOW2_KEY: "highlow2_model",
    apple_models.HIGHLOW3M_KEY: "highlow3m_model",
}
# What the trader-levels note calls the forecast its levels hang off, when it
# is not TimeToChange3's.
_HIGHLOW_NAMES = {
    apple_models.HIGHLOW_KEY: OVERLAYS[HIGHLOW_RANGE_KEY].name,
    apple_models.HIGHLOW2_KEY: OVERLAYS[HIGHLOW2_RANGE_KEY].name,
    apple_models.HIGHLOW3M_KEY: OVERLAYS[HIGHLOW3M_RANGE_KEY].name,
}


def _highlow_forecast(
    symbol: str,
    session: pd.DataFrame,
    day: pd.Timestamp,
    credentials: "tuple[str, str] | None",
    model_key: str = apple_models.HIGHLOW_KEY,
) -> dict:
    """HighLow's (or HighLow2's, `model_key`) forecast for the session, or why
    there is none -- the same `{"forecast", "made_at", "problem"}` shape
    `_day_range_forecast` returns.

    Its history is minute bars strictly before `day`, read from the model's
    own cache (`highlow_model.history_frame`, `highlow2_model.history_inputs`,
    `highlow3m_model.history_inputs` and its option tables), so a past session
    is drawn with what the model could have said that morning.
    """
    def failed(problem: str) -> dict:
        return {"forecast": None, "made_at": None, "problem": problem}

    try:
        bundle = apple_models.load(model_key, symbol)
        if bundle is None:
            return failed(apple_models.unavailable_reason(model_key, symbol))

        # heavy (LightGBM, and torch for HighLow); only once it is needed
        module = importlib.import_module(f".{_HIGHLOW_MODULES[model_key]}", __package__)

        want = module.opening_minutes(bundle)
        if len(session) < want:
            return failed(
                f"the forecast is built on the first {want} minutes and only "
                f"{len(session)} bars have closed."
            )
        opening = session.iloc[:want]
        if float(opening["minutes_from_open"].iloc[0]) >= 1.0:
            return failed(
                f"these bars start at {session.index[0]:%H:%M}, not the 09:30 open, so "
                f"the first {want} minutes the forecast needs are not here."
            )
        key, secret = credentials or (None, None)
        forecast = module.forecast_session(bundle, symbol, opening, day, key, secret)
    except Exception as exc:  # a decoration must never take the chart down
        return failed(str(exc))
    return {"forecast": forecast, "made_at": opening.index[-1], "problem": ""}


def _day_range_items(
    result: dict,
    day: pd.Timestamp,
    symbol: str = "",
    session: "pd.DataFrame | None" = None,
    open_price: "float | None" = None,
    config=None,
    key: str = DAY_RANGE_KEY,
    walked: "list[dict] | None" = None,
) -> "tuple[list[dict], str]":
    """The predicted high and low, and the range between them.

    Flat while the session stays inside the forecast, and a pair of stepped
    curves once it does not -- the same shape `trader_levels` draws, for the
    same reason and off the same walk. A forecast the tape has traded through
    is revised during the session (`apple_trader.DayRangeTrader._update_range`,
    `dayrange_model.contain_session`), and the buy and sell levels are rebuilt
    from the revision every time it moves. Drawing the 9:35 numbers flat while
    the levels beneath them step up would put the two halves of one picture in
    disagreement, and the half that was wrong would be the one labelled with
    the model's name.

    Falls back to the flat 9:35 forecast when there is no session to walk,
    which is what a caller with bars but no usable window has.

    `key` is the overlay being drawn: `day_range` (TimeToChange3),
    `highlow_range` or `highlow2_range`, which differ only in whose forecast
    `result` holds.

    `walked` is a live run's own record (`_recorded_levels`), drawn as given in
    place of the walk.
    """
    overlay = OVERLAYS[key]
    forecast = result["forecast"]
    if forecast is None:
        return [], f"{overlay.label}: {result['problem']}"

    color = MODEL_OVERLAY_COLORS[key]
    x0 = result["made_at"]
    x1 = _session_close(day)
    high, low = float(forecast["pred_high"]), float(forecast["pred_low"])
    made_at = f"forecast at {pd.Timestamp(x0):%H:%M}"
    # A forecast of what is left after the opening window (HighLow_3m's,
    # HighLow2's rest head) is not the session's, and its notes say so.
    rest = bool(forecast.get("range_after_opening"))
    high_is, low_is = (
        (f"predicted high after {pd.Timestamp(x0):%H:%M}", f"predicted low after {pd.Timestamp(x0):%H:%M}")
        if rest else ("predicted session high", "predicted session low")
    )

    if walked is None:
        walked = _day_range_walk(symbol, session, x0, open_price, forecast, config)
    highs = [row["pred_high"] for row in walked]
    lows = [row["pred_low"] for row in walked]
    moves = bool(walked) and (min(highs) != max(highs) or min(lows) != max(lows))
    if not moves:
        return (
            [
                _span(
                    key, "Predicted range after the open" if rest else "Predicted day range",
                    x0, x1, color, y0=low, y1=high,
                    note=f"{low:.2f} – {high:.2f} ({made_at})",
                ),
                _level(key, "Pred. high", high, color,
                       note=f"{high_is} ({made_at})", x0=x0, x1=x1),
                _level(key, "Pred. low", low, color,
                       note=f"{low_is} ({made_at})", x0=x0, x1=x1),
            ],
            "",
        )
    # One band rather than a span plus two levels: the span's whole job was to
    # tint between the two lines, which is what a band already does, and three
    # items that must agree bar by bar are three chances to disagree.
    return (
        [
            _band(
                key, "Pred. high / low",
                [row["t"] for row in walked], lows, highs, color,
                lower_label="Pred. low", upper_label="Pred. high", step=True,
                note=(
                    f"the {made_at}, revised where the session traded outside it "
                    f"({low:.2f} – {high:.2f} at {pd.Timestamp(x0):%H:%M}, "
                    f"{lows[-1]:.2f} – {highs[-1]:.2f} now)"
                ),
            )
        ],
        "",
    )


def _day_range_walk(
    symbol: str,
    session: "pd.DataFrame | None",
    made_at,
    open_price: "float | None",
    forecast: dict,
    config,
) -> "list[dict]":
    """The forecast bar by bar, as the configured agent maintains it.

    Shares `session_levels` with `trader_levels` rather than re-walking the
    session, so the predicted high drawn here is by construction the one the
    buy and sell levels are measured under -- the drift this is meant to
    prevent could not be prevented by two walks that merely agree today.

    An empty list means "nothing to walk", not "nothing moved": the caller
    reads it as the flat 9:35 forecast, which is what it is.
    """
    if session is None or not len(session):
        return []
    from .apple_trader import AppleTraderConfig, session_levels  # heavy-ish, and only here

    if config is None or (config.ticker or "").upper() != (symbol or "").upper():
        # Same rule as `trader_levels`: another symbol's configuration is not
        # this chart's, and the shipped one is what an agent started now would
        # maintain the forecast with.
        config = AppleTraderConfig(ticker=symbol)
    return session_levels(config, forecast, session, made_at, open_price=open_price)


def _recorded_levels(
    history: "dict | None", symbol: str, day: pd.Timestamp
) -> "tuple[list[dict], object] | None":
    """A live run's record of its levels, if it is about this chart's session.

    `(rows, config)` with each row's `t` as an exchange-local timestamp, or
    None when there is no record, it is another symbol's, or another day's --
    yesterday's run must not draw over this morning's candles.
    """
    if not history or not history.get("rows") or history.get("config") is None:
        return None
    if (history.get("ticker") or "").upper() != symbol:
        return None
    if pd.Timestamp(history.get("date")).normalize() != pd.Timestamp(day).normalize():
        return None
    rows = []
    for row in list(history["rows"]):  # the agent's thread appends to it
        stamp = pd.Timestamp(row["t"])
        if stamp.tzinfo is None:
            stamp = stamp.tz_localize(market_hours.MARKET_TZ)
        rows.append({**row, "t": stamp.tz_convert(market_hours.MARKET_TZ)})
    return rows, history["config"]


def _after_record(rows: "list[dict]", history: dict, session: pd.DataFrame, config) -> "list[dict]":
    """The forecast carried on past the last bar a run recorded, or [].

    A run that has been stopped leaves a record ending at the minute it stopped,
    and the prediction is about the whole session so far. So the bars after it
    are walked (`session_levels`) from the forecast the run itself started
    from -- its `seed`, not a fresh model call -- under its configuration, which
    reproduces the run's own path to the bar and then carries it on. [] when
    the record predates the seed, or nothing has closed after it.
    """
    seed = (history or {}).get("seed")
    if not seed or not rows or not len(session):
        return []
    last = pd.Timestamp(rows[-1]["t"])
    if session.index[-1] <= last:
        return []
    from .apple_trader import session_levels  # heavy-ish, and only here

    opening_end = pd.Timestamp(seed["opening_end"])
    if opening_end.tzinfo is None:
        opening_end = opening_end.tz_localize(market_hours.MARKET_TZ)
    walk = session_levels(
        config, dict(seed["forecast"]), session,
        opening_end.tz_convert(session.index.tz), open_price=seed.get("open_price"),
    )
    return [row for row in walk if pd.Timestamp(row["t"]) > last]


def _recorded_result(rows: "list[dict]") -> dict:
    """The forecast a recorded run started from, in `_day_range_forecast`'s shape."""
    first = rows[0]
    return {
        "forecast": {"pred_high": first["pred_high"], "pred_low": first["pred_low"]},
        "made_at": first["t"],
        "problem": "",
    }


# --- intraday range (IntradayVolatility) ------------------------------------


def _intraday_model(symbol: str, label_: str) -> "tuple[dict | None, str]":
    model = intraday_vol_model.load(symbol)
    if model is None:
        return None, (
            f"{label_}: no IntradayVolatility model at "
            f"{intraday_vol_model.model_path(symbol)} — export it with "
            "FinNotebooks/IntradayVolatility/scripts/export_app_model.py."
        )
    return model, ""


def _session_open_price(
    session: pd.DataFrame, open_price: "float | None"
) -> "tuple[float | None, str]":
    """The price the envelope is centred on: the official print, else the
    first bar's open -- but only if that bar really is 09:30's."""
    if open_price:
        return float(open_price), ""
    if float(session["minutes_from_open"].iloc[0]) >= 1.0:
        return None, (
            f"these bars start at {session.index[0]:%H:%M}, not the 09:30 open, and no "
            "opening print was supplied to centre the range on."
        )
    return float(session["open"].iloc[0]), ""


def _session_band(
    key: str, label_: str, model: dict, day: pd.Timestamp,
    open_px: float, high: float, low: float, note: str,
) -> dict:
    """The envelope for one session, one point per minute from 09:30 to 16:00.

    Drawn over the whole session, the first minutes included, even when the
    extremes came from a forecast made at 09:35: it is a claim about the shape
    of the day, and its maximum is at the open.
    """
    minutes = np.arange(intraday_vol_model.SESSION_MINUTES + 1)
    upper, lower = intraday_vol_model.envelope(model, open_px, high, low, minutes)
    start = pd.Timestamp(day).tz_localize(market_hours.MARKET_TZ) + timedelta(
        hours=market_hours.MARKET_OPEN.hour, minutes=market_hours.MARKET_OPEN.minute
    )
    stamps = start + pd.to_timedelta(minutes, unit="min")
    return _band(key, label_, stamps, lower, upper, MODEL_OVERLAY_COLORS[key], note=note)


def _intraday_range_items(
    symbol: str,
    session: pd.DataFrame,
    daily_bars: "list[dict]",
    day: pd.Timestamp,
    open_price: "float | None",
) -> "tuple[list[dict], str]":
    """IntradayVolatility alone: its day-range forecast, shaped by time of day."""
    overlay = OVERLAYS[INTRADAY_RANGE_KEY]
    model, why = _intraday_model(symbol, overlay.label)
    if model is None:
        return [], why
    open_px, why = _session_open_price(session, open_price)
    if open_px is None:
        return [], f"{overlay.label}: {why}"
    try:
        high, low = intraday_vol_model.predicted_extremes(model, daily_bars, day, open_px)
    except ValueError as exc:
        return [], f"{overlay.label}: {exc}"
    note = (
        f"{100 * (high / low - 1):.2f}% day range forecast at the open "
        f"({low:.2f} – {high:.2f})"
    )
    return [
        _session_band(INTRADAY_RANGE_KEY, "Pred. intraday range", model, day,
                      open_px, high, low, note)
    ], ""


def _intraday_dayrange_items(
    symbol: str,
    session: pd.DataFrame,
    day: pd.Timestamp,
    open_price: "float | None",
    result: dict,
) -> "tuple[list[dict], str]":
    """The time-of-day curve stretched to TimeToChange3's high and low."""
    overlay = OVERLAYS[INTRADAY_DAYRANGE_KEY]
    model, why = _intraday_model(symbol, overlay.label)
    if model is None:
        return [], why
    forecast = result["forecast"]
    if forecast is None:
        return [], f"{overlay.label}: {result['problem']}"
    open_px, why = _session_open_price(session, open_price)
    if open_px is None:
        return [], f"{overlay.label}: {why}"
    high, low = forecast["pred_high"], forecast["pred_low"]
    note = (
        f"between the predicted high {high:.2f} and low {low:.2f} "
        f"(forecast at {pd.Timestamp(result['made_at']):%H:%M})"
    )
    return [
        _session_band(INTRADAY_DAYRANGE_KEY, "Pred. intraday × day range", model, day,
                      open_px, high, low, note)
    ], ""


# --- Apple Trader's resting orders ------------------------------------------


def trader_model(config, symbol: str) -> str:
    """The model whose forecast `trader_levels` hangs the levels off on `symbol`.

    The configured one when the configuration is for this symbol and that model
    covers it; otherwise the default model if it covers the symbol, else the
    first that does -- so a symbol only HighLow was fitted on (MU) draws
    HighLow's levels rather than asking for a day-range bundle it never had.
    """
    named = getattr(config, "model_key", None)
    if (
        named
        and (getattr(config, "ticker", "") or "").upper() == symbol
        and apple_models.covers(named, symbol)
    ):
        return named
    if apple_models.covers(apple_models.DEFAULT_MODEL, symbol):
        return apple_models.DEFAULT_MODEL
    return next(iter(apple_models.keys_for(symbol)), apple_models.DEFAULT_MODEL)


def _sat_out(symbol: str, day: pd.Timestamp, config=None) -> "str | None":
    """Why the configured agent rests nothing on this session, when its
    days-off rule sits it out (`event_days`) -- the note drawn in place of
    levels it would never have rested. Asked before the forecast, which such a
    session never needs. None when it trades, or when the check cannot be read
    (the walk is drawn then, as before the rule)."""
    from .apple_trader import default_config  # heavy-ish, and only here

    if config is None or (config.ticker or "").upper() != symbol:
        config = default_config(symbol, trader_model(None, symbol))
    if not config.skip_events:
        return None
    try:
        found = event_days.check(symbol, pd.Timestamp(day).date(), config.skip_events)
    except Exception:  # a decoration must never take the chart down
        return None
    if not found.events:
        return None
    return (
        f"{OVERLAYS[TRADER_LEVELS_KEY].label}: the agent sits this session out — "
        f"{event_days.sit_out_phrase(found.events)}."
    )


def _trader_levels_items(
    symbol: str,
    session: pd.DataFrame,
    day: pd.Timestamp,
    open_price: "float | None",
    result: dict,
    config=None,
    levels: "list[dict] | None" = None,
) -> "tuple[list[dict], str]":
    """The buy, the sell and the stop, as the configured agent would rest them.

    Flat under the notebook's settings and a moving pair under the others, so
    the shape of the drawing follows the shape of the strategy: two levels when
    neither moves all session -- which also mirrors them into the price profile,
    where a resting order is exactly the kind of thing to read against traded
    volume -- and a band between them when they do.

    `levels` is a live run's own record (`_recorded_levels`) with `config` the
    run's configuration; it is drawn as given, sidebar edits and all, in place
    of `session_levels`' walk under `config`.
    """
    from .apple_trader import (  # heavy-ish, and only here
        default_config, session_levels, stop_phrase,
    )

    overlay = OVERLAYS[TRADER_LEVELS_KEY]
    forecast = result["forecast"]
    if forecast is None:
        return [], f"{overlay.label}: {result['problem']}"
    if config is None or (config.ticker or "").upper() != symbol:
        # A configuration for another symbol is not this chart's strategy, and
        # its distances were swept on that symbol's tape. The pair's defaults
        # are: its base configuration, else the shipped ones.
        config = default_config(symbol, trader_model(None, symbol))

    made_at = result["made_at"]
    recorded = levels is not None
    if levels is None:
        levels = session_levels(config, forecast, session, made_at, open_price=open_price)
    if not levels:
        return [], (
            f"{overlay.label}: the session has no bar after the {pd.Timestamp(made_at):%H:%M} "
            "forecast yet, so there is nothing resting."
        )

    color = MODEL_OVERLAY_COLORS[TRADER_LEVELS_KEY]
    buys = [row["buy"] for row in levels]
    sells = [row["sell"] for row in levels]
    forecast_of = apple_models.forecast_key(config.model_key)
    how = (
        f"buy {config.buy_k:g} × and sell {config.sell_k:g} × the {config.unit_phrase} under "
        + ("the predicted high" if config.level_source != "intraday"
           else "the intraday band's upper curve")
        + (f" ({_HIGHLOW_NAMES[forecast_of]})" if forecast_of in _HIGHLOW_NAMES else "")
    )
    # A run that traded HighLow_3m's 9:33 window (`use_3m`) recorded those two
    # bars on that model's levels before its own: never one flat pair.
    early = config.model_key != apple_models.HIGHLOW3M_KEY and any(
        row.get("model_key") == apple_models.HIGHLOW3M_KEY for row in levels
    )
    if early:
        how += (
            f"; before its forecast, HighLow_3m's 9:33 window: buy "
            f"{getattr(config, 'buy_3m_k', 0.0):g} × and sell "
            f"{getattr(config, 'sell_3m_k', 0.0):g} × under that model's predicted high"
        )
    # Only while a position is open: the stop hangs under the actual fill, so
    # before a buy there is none, and a walk (which fills nothing) never has one.
    stops = [row.get("stop") for row in levels]
    stop_path = (
        [_path(TRADER_LEVELS_KEY, "Stop level", [row["t"] for row in levels], stops,
               MODEL_OVERLAY_COLORS["trader_stop"], step=True,
               note=f"Apple Trader's stop, {stop_phrase(config)} under the actual fill, "
                    "while a position is open")]
        if recorded and any(v is not None for v in stops) else []
    )
    moves = min(buys) != max(buys) or min(sells) != max(sells) or early
    if not moves:
        x0, x1 = made_at, _session_close(day)
        items = [
            _level(TRADER_LEVELS_KEY, "Buy level", buys[0], color,
                   note=f"Apple Trader's resting buy — {how}", x0=x0, x1=x1),
            _level(TRADER_LEVELS_KEY, "Sell level", sells[0], color, dash="dashdot",
                   note=f"Apple Trader's resting sell — {how}", x0=x0, x1=x1),
        ]
        return items + stop_path, ""
    return (
        [
            _band(
                TRADER_LEVELS_KEY, "Buy/sell levels",
                [row["t"] for row in levels], buys, sells, color,
                lower_label="Buy level", upper_label="Sell level", step=True,
                note=(
                    f"{how}; they move through the session "
                    f"({buys[0]:.2f} – {sells[0]:.2f} at the forecast, "
                    f"{buys[-1]:.2f} – {sells[-1]:.2f} now)"
                ),
            ),
            *stop_path,
        ],
        "",
    )


# --- Fast&Furious -----------------------------------------------------------


def _fastfurious_forecast(
    symbol: str,
    session: pd.DataFrame,
    day: pd.Timestamp,
    credentials: "tuple[str, str] | None",
    config=None,
) -> dict:
    """Every minute's Fast&Furious forecast of the session so far, for the two
    overlays that draw it.

    Each minute's forecast is the one made at that bar's close, read from the
    model's own tape (SIP to 15 minutes back, IEX since), so a past session is
    drawn with what the model said at each minute and a live one with what it
    has said so far. The threshold is the configured run's quantile when the
    chart's configuration is a Fast&Furious one (or trades its opening window),
    the notebook's otherwise.

    Returns `{"fc", "stamps", "horizon", "theta", "sessions", "quantile",
    "problem", "threshold_problem"}`: `stamps` are the bars each forecast was
    made at, `horizon` how many minutes the momentum forecast looks ahead,
    `problem` says why there is no forecast at all, `threshold_problem` why
    there is no threshold -- which only the signals need, so the range is still
    drawn without one.
    """
    model_key = apple_models.FASTFURIOUS_KEY
    from .config import APPLE_TRADER_FF_ENTRY_QUANTILE

    quantile = APPLE_TRADER_FF_ENTRY_QUANTILE
    named = getattr(config, "model_key", None) if config is not None else None
    if named == model_key or apple_models.window_key(named) == model_key:
        # A run on Fast&Furious, or one trading its opening window: that run's own.
        quantile = float(config.ff_entry_quantile)
    out = {"fc": None, "stamps": [], "horizon": 0, "theta": float("nan"), "sessions": 0,
           "quantile": quantile, "problem": "", "threshold_problem": ""}
    try:
        bundle = apple_models.load(model_key, symbol)
        if bundle is None:
            return {**out, "problem": apple_models.unavailable_reason(model_key, symbol)}
        module = importlib.import_module(".fastfurious_model", __package__)
        # Closed bars only: a live buffer's newest row can be the minute still
        # trading, whose forecast does not exist yet.
        closed = session[session.index + pd.Timedelta(minutes=1) <= pd.Timestamp(clock.now())]
        if not len(closed):
            return {**out, "problem": "no bar of the session has closed yet."}
        last = int(round(float(closed["minutes_from_open"].iloc[-1])))
        key, secret = credentials or (None, None)
        fc = module.session_forecasts(bundle, symbol, day, last, key, secret, wait=False)
    except Exception as exc:  # a decoration must never take the chart down
        return {**out, "problem": str(exc)}
    if not len(fc):
        return {**out, "problem": "no forecast yet — IEX has not printed this session."}
    try:
        theta, sessions = module.threshold(bundle, symbol, day, quantile, key, secret)
        problem = "" if np.isfinite(theta) else f"no entry threshold ({sessions} earlier sessions)."
    except Exception as exc:
        theta, sessions, problem = float("nan"), 0, f"no entry threshold ({exc})."
    open_ts = pd.Timestamp(day).tz_localize(market_hours.MARKET_TZ) + pd.Timedelta(
        hours=market_hours.MARKET_OPEN.hour, minutes=market_hours.MARKET_OPEN.minute
    )
    return {
        **out,
        "fc": fc,
        "stamps": [open_ts + pd.Timedelta(minutes=int(m)) for m in fc.index],
        "horizon": int(module.HORIZON),
        "theta": float(theta),
        "sessions": int(sessions),
        "threshold_problem": problem,
    }


def _fastfurious_signal_items(result: dict) -> "tuple[list[dict], str]":
    """The momentum forecast the trades are made on, and its signals.

    The forecast is drawn in bps per minute (`mom_bps`) on the momentum panel,
    with the entry threshold either side of zero in the same unit. The model
    compares `y_hat` with a fixed threshold in units of `s30`, the usual size
    of a 30-minute move at that time of day, so in bps per minute the threshold
    is a curve: wide at the open and into the close, narrow at midday. The
    line crosses it exactly where `y_hat` crosses the fixed one, which is where
    a ▲ or ▼ goes on the candles.

    The line and the thresholds are drawn `horizon` (30) minutes after the bar
    each forecast was made at -- at the end of the window it is about -- so it
    sits over the momentum bars it was a forecast of, and the newest half hour
    of forecasts is not drawn until that time comes. The markers stay at the
    bar the forecast was made at, which is when the trader acts on it, so a
    marker is `horizon` minutes left of where the line crosses a threshold.
    """
    overlay = OVERLAYS[FASTFURIOUS_SIGNAL_KEY]
    if result["problem"]:
        return [], f"{overlay.label}: {result['problem']}"
    fc, stamps = result["fc"], result["stamps"]
    theta, sessions, quantile = result["theta"], result["sessions"], result["quantile"]
    horizon = pd.Timedelta(minutes=result["horizon"])
    due = [stamp + horizon for stamp in stamps]
    color = MODEL_OVERLAY_COLORS["fastfurious"]
    sell_color = MODEL_OVERLAY_COLORS["trader_stop"]
    has_theta = np.isfinite(theta)
    # bps per minute for one unit of y: what turns y_hat into `mom_bps`.
    per_y = fc["s30"] / 30 * 1e4
    text = [f"made at the {stamp:%H:%M} bar · y {y:+.4f}" for stamp, y in zip(stamps, fc["y_hat"])]
    if has_theta:
        text = [f"{t} · {y / theta:+.0%} of the threshold" for t, y in zip(text, fc["y_hat"])]
    items: "list[dict]" = [_series(
        FASTFURIOUS_SIGNAL_KEY, "30-min momentum forecast", due, fc["mom_bps"],
        MODEL_OVERLAY_COLORS["fastfurious_forecast"], unit="bps/min",
        note="the average 1-minute move forecast at a bar's close for the next 30 minutes, "
             "drawn at the end of them",
        text=text,
    )]
    if not has_theta:
        return items, f"{overlay.label}: {result['threshold_problem']}"
    edge = (theta * per_y).tolist()
    where = (
        f"±{theta:.4f} in y, the {quantile:.0%} quantile of the last {sessions} sessions' "
        "forecasts"
    )
    items += [
        _series(FASTFURIOUS_SIGNAL_KEY, "Buy threshold", due, edge, color,
                unit="bps/min", dash="dash", width=1.0, note=where),
        _series(FASTFURIOUS_SIGNAL_KEY, "Sell threshold", due, [-v for v in edge],
                sell_color, unit="bps/min", dash="dash", width=1.0, note=where),
    ]
    sign = np.where(fc["y_hat"] >= theta, 1, np.where(fc["y_hat"] <= -theta, -1, 0))
    # The first bar of each run of the same signal: the trader acts on
    # every one of them, but forty markers in a row say nothing more.
    previous = np.r_[0, sign[:-1]]
    for stamp, (minute, row), s, before in zip(stamps, fc.iterrows(), sign, previous):
        if not s or s == before:
            continue
        items.append(_event(
            FASTFURIOUS_SIGNAL_KEY,
            f"{'Buy' if s > 0 else 'Sell'} signal: {row['mom_bps']:+.3f} bps/min",
            stamp,
            color if s > 0 else sell_color,
            icon="▲" if s > 0 else "▼",
            price=float(row["p"]),
            line=False,
            note=(
                f"forecast y {row['y_hat']:+.4f} against a threshold of ±{theta:.4f} (the "
                f"{quantile:.0%} quantile of the last {sessions} sessions' forecasts)"
            ),
        ))
    return items, ""


def _fastfurious_range_items(result: dict) -> "tuple[list[dict], str]":
    """The next two hours' predicted high and low from each bar's price -- the
    range model the stop hangs off, not the forecast the trades are made on."""
    overlay = OVERLAYS[FASTFURIOUS_RANGE_KEY]
    if result["problem"]:
        return [], f"{overlay.label}: {result['problem']}"
    fc, stamps = result["fc"], result["stamps"]
    ranged = fc["pred_high"].notna()
    if not ranged.any():
        problem = fc.attrs.get("range_problem") or "no option table"
        return [], f"{overlay.label}: no 2-hour range ({problem})."
    return [_band(
        FASTFURIOUS_RANGE_KEY, "2-hour range",
        [s for s, ok in zip(stamps, ranged) if ok],
        fc.loc[ranged, "pred_low"], fc.loc[ranged, "pred_high"],
        MODEL_OVERLAY_COLORS["fastfurious"],
        lower_label="Forecast 2-hour low", upper_label="Forecast 2-hour high",
        note="where the next two hours' high and low are forecast to land, from each "
             "bar's price at its close",
    )], ""


# The overlay that draws each per-minute model's signals.
_SIGNAL_OVERLAYS = {
    apple_models.FASTFURIOUS_KEY: FASTFURIOUS_SIGNAL_KEY,
    apple_models.GOODMOMENTS_KEY: GOODMOMENTS_SIGNAL_KEY,
}


def _signal_trader_items(
    symbol: str, recorded: "tuple[list[dict], object] | None", model_key: str,
) -> "tuple[list[dict], str]":
    """`trader_levels` under a per-minute model (Fast&Furious, GoodMoments):
    the running agent's stop -- and GoodMoments' take -- while it holds a
    position. It rests no buy or sell; its entries are the signals its own
    overlay draws."""
    overlay = OVERLAYS[TRADER_LEVELS_KEY]
    model = apple_models.get(model_key)
    if recorded is None:
        signals = OVERLAYS[_SIGNAL_OVERLAYS.get(model.key, FASTFURIOUS_SIGNAL_KEY)].name
        drawn = "take and stop are" if model.key == apple_models.GOODMOMENTS_KEY else "stop is"
        return [], (
            f"{overlay.label}: {model.label.replace(' model', '')} rests no levels — its buys "
            f"are signals (the {signals} overlay); a running agent's {drawn} drawn here."
        )
    rows, config = recorded
    stamps = [row["t"] for row in rows]
    items = []
    if model.key == apple_models.GOODMOMENTS_KEY:
        takes = [row.get("take") for row in rows]
        if any(v is not None for v in takes):
            items.append(_path(
                TRADER_LEVELS_KEY, "Take-profit", stamps, takes,
                MODEL_OVERLAY_COLORS["trader_levels"], step=True,
                note=f"Apple Trader's take, {getattr(config, 'gm_take_k', 0):g} ADR over the fill",
            ))
        stop_note = f"Apple Trader's stop, {getattr(config, 'gm_stop_k', 0):g} ADR under the fill"
    else:
        stop_note = (
            f"Apple Trader's stop, {getattr(config, 'ff_stop_k', 0):g} × the forecast 2-hour "
            "drop under the fill"
        )
    stops = [row.get("stop") for row in rows]
    if any(v is not None for v in stops):
        items.append(_path(
            TRADER_LEVELS_KEY, "Stop level", stamps, stops,
            MODEL_OVERLAY_COLORS["trader_stop"], step=True,
            note=f"{stop_note}, while a position is open",
        ))
    return items, ""


# --- GoodMoments ------------------------------------------------------------


def _goodmoments_forecast(
    symbol: str,
    session: pd.DataFrame,
    day: pd.Timestamp,
    credentials: "tuple[str, str] | None",
    config=None,
) -> dict:
    """Every minute's GoodMoments forecast of the session so far.

    Each minute's is the one made at that bar's close from the model's own tape
    (SIP to 15 minutes back, IEX since), so a past session is drawn with what
    the model said at each minute and a live one with what it has said so far.
    The thresholds are the configured run's when the chart's configuration is
    a GoodMoments one, the notebook's otherwise.

    Returns `{"fc", "stamps", "enter", "exit", "problem"}`.
    """
    from .config import APPLE_TRADER_GM_ENTER, APPLE_TRADER_GM_EXIT

    model_key = apple_models.GOODMOMENTS_KEY
    enter, exit_ = APPLE_TRADER_GM_ENTER, APPLE_TRADER_GM_EXIT
    if config is not None and getattr(config, "model_key", None) == model_key:
        enter, exit_ = float(config.gm_enter), config.gm_exit
    out = {"fc": None, "stamps": [], "enter": enter, "exit": exit_, "problem": ""}
    try:
        bundle = apple_models.load(model_key, symbol)
        if bundle is None:
            return {**out, "problem": apple_models.unavailable_reason(model_key, symbol)}
        module = importlib.import_module(".goodmoments_model", __package__)
        # Closed bars only: a live buffer's newest row can be the minute still
        # trading, whose forecast does not exist yet.
        closed = session[session.index + pd.Timedelta(minutes=1) <= pd.Timestamp(clock.now())]
        if not len(closed):
            return {**out, "problem": "no bar of the session has closed yet."}
        last = int(round(float(closed["minutes_from_open"].iloc[-1])))
        key, secret = credentials or (None, None)
        fc = module.session_forecasts(bundle, symbol, day, last, key, secret, wait=False)
    except Exception as exc:  # a decoration must never take the chart down
        return {**out, "problem": str(exc)}
    if not len(fc):
        return {**out, "problem": "no forecast yet — IEX has not printed this session."}
    open_ts = pd.Timestamp(day).tz_localize(market_hours.MARKET_TZ) + pd.Timedelta(
        hours=market_hours.MARKET_OPEN.hour, minutes=market_hours.MARKET_OPEN.minute
    )
    return {**out, "fc": fc, "stamps": [open_ts + pd.Timedelta(minutes=int(m)) for m in fc.index]}


def _goodmoments_signal_items(result: dict) -> "tuple[list[dict], str]":
    """The edge the trades are made on, and its buy signals.

    The edge is drawn in percentage points on the momentum panel at the bar it
    was made at -- its horizon is the close, so there is no later moment it
    belongs to -- with the entry threshold (and the exit, when one is set).
    A ▲ marks the first bar of each run at or over the entry up to 15:30, the
    last bar the rules buy on."""
    from .config import APPLE_TRADER_GM_LAST_ENTRY_MINUTE

    overlay = OVERLAYS[GOODMOMENTS_SIGNAL_KEY]
    if result["problem"]:
        return [], f"{overlay.label}: {result['problem']}"
    fc, stamps = result["fc"], result["stamps"]
    enter, exit_ = result["enter"], result["exit"]
    color = MODEL_OVERLAY_COLORS["goodmoments"]
    text = [
        f"made at the {stamp:%H:%M} bar · P(+0.2 ADR by the close) {p:.1%} vs {tv:.1%} on time "
        "and volatility"
        for stamp, p, tv in zip(stamps, fc["p"], fc["p_tv"])
    ]
    items: "list[dict]" = [
        _series(
            GOODMOMENTS_SIGNAL_KEY, "Edge over time & volatility", stamps, (fc["edge"] * 100).tolist(),
            color, unit="pts",
            note="the model's probability of a +0.2 ADR rise before the close, less the "
                 "time-and-volatility model's, in percentage points",
            text=text,
        ),
        _series(
            GOODMOMENTS_SIGNAL_KEY, "Entry threshold", stamps, [enter * 100] * len(stamps), color,
            unit="pts", dash="dash", width=1.0, note=f"buy at an edge of {enter:+g} or more",
        ),
    ]
    if exit_ is not None:
        items.append(_series(
            GOODMOMENTS_SIGNAL_KEY, "Exit threshold", stamps, [exit_ * 100] * len(stamps),
            MODEL_OVERLAY_COLORS["trader_stop"], unit="pts", dash="dash", width=1.0,
            note=f"sell a long once the edge is under {exit_:+g}",
        ))
    on = (fc["edge"] >= enter).to_numpy() & (fc.index.to_numpy() <= APPLE_TRADER_GM_LAST_ENTRY_MINUTE)
    before = np.r_[False, on[:-1]]
    for stamp, (minute, row), now, was in zip(stamps, fc.iterrows(), on, before):
        if not now or was:
            continue
        items.append(_event(
            GOODMOMENTS_SIGNAL_KEY,
            f"Buy signal: edge {row['edge']:+.3f}",
            stamp, color, icon="▲", price=float(row["price"]), line=False,
            note=(
                f"P(+0.2 ADR by the close) {row['p']:.1%} against {row['p_tv']:.1%} on time and "
                f"volatility — at or over the {enter:+g} entry"
            ),
        ))
    return items, ""


# --- predicted profile range (LevelsML) -------------------------------------


def _profile_range_items(
    session: pd.DataFrame, daily_bars: "list[dict]", day: pd.Timestamp
) -> "tuple[list[dict], str]":
    """The predicted profile's outer quantiles and point of control.

    The curve is already drawn in the histogram beside the candles; these are
    the same numbers as levels, which is what makes the prediction readable
    against price action rather than only against volume.
    """
    overlay = OVERLAYS[PROFILE_RANGE_KEY]
    pack = profile_model.load_pack()
    if pack is None:
        return [], (
            f"{overlay.label}: no open-profile pack at "
            f"{profile_model._model_path()} (or LightGBM is not installed)."
        )

    today = str(day.date())
    open_px = float(session["open"].iloc[0])
    feats = profile_model.compute_features(daily_bars, open_px, today)
    if feats is None:
        return [], (
            f"{overlay.label}: needs at least {profile_model.MIN_DAILY_BARS} completed "
            f"daily bars before {today}; got "
            f"{len(profile_model.completed_daily_bars(daily_bars, today))}."
        )
    quantiles = profile_model.predict_quantiles(pack, feats)
    if quantiles is None:
        return [], f"{overlay.label}: the pack could not score today's features."

    levels = list(pack["p_levels"])
    grid_bps, density = profile_model.density_from_quantiles(quantiles, levels)
    if not density.max() > 0:
        return [], f"{overlay.label}: the predicted density is empty."

    def price(bps: float) -> float:
        return float(open_px * np.exp(bps / 1e4))

    lo, hi = price(quantiles[0]), price(quantiles[-1])
    poc = price(float(grid_bps[int(np.argmax(density))]))
    color = MODEL_OVERLAY_COLORS[PROFILE_RANGE_KEY]
    poc_color = MODEL_OVERLAY_COLORS["profile_poc"]
    x0, x1 = session.index[0], _session_close(day)
    return (
        [
            _span(
                PROFILE_RANGE_KEY,
                f"Predicted profile q{levels[0]:g}–q{levels[-1]:g}",
                x0, x1, color, y0=lo, y1=hi,
                note=f"{lo:.2f} – {hi:.2f} of predicted volume",
            ),
            _level(PROFILE_RANGE_KEY, f"Pred. q{levels[-1]:g}", hi, color,
                   note=f"{levels[-1]:g}th volume-quantile of the predicted profile",
                   x0=x0, x1=x1),
            _level(PROFILE_RANGE_KEY, f"Pred. q{levels[0]:g}", lo, color,
                   note=f"{levels[0]:g}th volume-quantile of the predicted profile",
                   x0=x0, x1=x1),
            _level(PROFILE_RANGE_KEY, "Pred. POC", poc, poc_color, dash="dashdot",
                   note="densest predicted price", x0=x0, x1=x1),
        ],
        "",
    )


# --- callers' helpers -------------------------------------------------------


def session_date_of(bars: "list[dict]") -> "datetime | None":
    """The exchange-local date the last of these bars belongs to."""
    if not bars:
        return None
    ts = pd.to_datetime(bars[-1]["t"], utc=True).tz_convert(market_hours.MARKET_TZ)
    return ts.normalize().tz_localize(None).to_pydatetime()


def live_overlays(
    sym_state,
    bars: "list[dict]",
    overlay_keys: "list[str] | None",
) -> dict:
    """`compute` for the live app, cached on the SymbolState until a new bar.

    The chart fragment re-runs every few seconds and nothing about these
    answers changes between bars: a day-range forecast is made once at 09:35
    and never updated, and the profile range is a function of the closed bars.
    Recomputing them on every poll would cost seconds for a picture that did
    not move, so the result is held against the
    newest bar's timestamp and the selection that produced it.
    """
    wanted = [k for k in (overlay_keys or []) if k in OVERLAYS]
    if not wanted or not bars:
        return {"items": [], "notes": []}

    # The agent's configuration is in the key because `trader_levels` is a
    # picture of it: moving a distance in the sidebar has to move the lines on
    # the next rerun, not on the next bar. And the running agent's record,
    # which grows once a cycle and changes the moment a sidebar edit is
    # adopted -- which can be mid-bar.
    config, history = live_trader_view(getattr(sym_state, "app", None), sym_state.symbol)
    rows = (history or {}).get("rows") or []
    key = (
        bars[-1].get("t"), tuple(wanted), len(bars),
        None if config is None else astuple(config),
        len(rows), tuple(rows[-1].values()) if rows else None,
    )
    cached = getattr(sym_state, "model_overlay_cache", None)
    if cached and cached.get("key") == key:
        return cached["result"]

    session_date = session_date_of(bars)
    app = getattr(sym_state, "app", None)
    result = compute(
        wanted,
        sym_state.symbol,
        bars,
        daily_bars=list(sym_state.daily_bars or []),
        session_date=session_date,
        trader_config=config,
        trader_history=history,
        credentials=(getattr(app, "api_key", "") or None, getattr(app, "api_secret", "") or None),
        **_live_dayrange_inputs(sym_state.symbol, wanted, session_date),
    )
    sym_state.model_overlay_cache = {"key": key, "result": result}
    return result


def live_trader_view(app, symbol: str) -> "tuple[object, dict | None]":
    """`(trader_config, trader_history)` for the live chart of `symbol`.

    Outside a race that is the single run's: the form's configuration and the
    running agent's record (`AppState.apple_trader_config`/`apple_trader_levels`).

    A race (`orchestra`) keeps one of each per racer, and races both models of
    a symbol when it has two, so the chart of a symbol has to pick: the racer
    holding the position when it trades this symbol (under a split, the first
    pair on it with a position), otherwise the first racer on it in the race's
    order. Its own record when it has one, otherwise the
    form's configuration for it, walked like any configuration without a record.
    A symbol the race does not trade gets neither, and draws the instrument's
    shipped configuration.
    """
    if app is None:
        return None, None
    race = getattr(app, "orchestra", None) or {}
    configs = getattr(app, "orchestra_configs", None) or {}
    if not configs and not race.get("running"):
        return (
            getattr(app, "apple_trader_config", None),
            getattr(app, "apple_trader_levels", None),
        )
    symbol = (symbol or "").upper()
    records = getattr(app, "orchestra_levels", None) or {}
    order = list(race.get("order") or []) + [k for k in configs if k not in (race.get("order") or [])]
    on_symbol = [
        key for key in order
        if str((records.get(key) or {}).get("ticker") or getattr(configs.get(key), "ticker", "")).upper()
        == symbol
    ]
    if not on_symbol:
        return None, None
    holders = list(race.get("holders") or []) or [race.get("holder")]
    key = next((k for k in on_symbol if k in holders), on_symbol[0])
    record = records.get(key)
    if record:
        return record.get("config") or configs.get(key), record
    return configs.get(key), None


# The overlays that run TimeToChange3's forecast, and so need its inputs.
_DAYRANGE_DRIVEN = (DAY_RANGE_KEY, INTRADAY_DAYRANGE_KEY)


def _live_dayrange_inputs(
    symbol: str, wanted: "list[str]", session_date: "datetime | None"
) -> dict:
    """The day-range forecast's inputs, fetched the way Apple Trader fetches them.

    The live `SymbolState.daily_bars` is the volume baseline: 365 calendar days
    off the stream's feed, which is about 250 completed sessions -- short of the
    253 `dayrange_model.require_history` demands, so every live forecast was
    refused and the overlay drew nothing. It is also not the unadjusted yfinance
    series the model was fitted on. So the forecast gets the trader's history
    (`fetch_daily_ohlc_bars`' 420-day default is `dayrange_model.DAILY_HISTORY_DAYS`)
    and its official opening print; the other overlays keep the buffer's bars.

    Fetched only when an overlay that needs it is selected for a symbol it
    covers, and both reads are cached in `historical`. The opening print is
    today's, so a chart of an earlier session falls back to its first bar.
    """
    if not any(k in _DAYRANGE_DRIVEN and OVERLAYS[k].covers(symbol) for k in wanted):
        return {}
    today = clock.now().astimezone(market_hours.MARKET_TZ).date()
    is_today = session_date is not None and session_date.date() == today
    return {
        "dayrange_daily_bars": historical.fetch_daily_ohlc_bars(symbol),
        "open_price": historical.fetch_session_open(symbol) if is_today else None,
    }
