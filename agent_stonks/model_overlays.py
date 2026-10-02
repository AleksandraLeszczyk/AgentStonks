"""What the saved models predict, in the shape a price chart can draw.

The ML surface of this app is a set of models that answer questions about a
session -- where its volume will trade, how wide it will be. Each of them already has a consumer (`profile_model`
feeds the profile curve, `apple_models` feeds the traders), but until now the
only way to see what a model said was to read a trader's log. This module is
the other way round: it asks each model its question about a session and
returns the answer as **drawing instructions**, so the live chart and SimLab's
replay chart can show the prediction beside the tape that tested it.

Five overlays
-------------
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
`trader_levels`      where Apple Trader would rest its buy and its sell, from
                     the forecast of the model its configuration names. The odd one out: every other overlay
                     draws what a *model* said, this one draws what an *agent*
                     would do about it, so it takes a configuration and asks
                     `apple_trader.session_levels` rather than deriving the
                     levels itself.

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

from dataclasses import astuple, dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from . import (
    apple_models, historical, intraday_vol_model, market_hours, momentum_regime, profile_model,
)
from .config import MODEL_OVERLAY_COLORS


# --- the catalogue ----------------------------------------------------------

DAY_RANGE_KEY = "day_range"
PROFILE_RANGE_KEY = "profile_range"
INTRADAY_RANGE_KEY = "intraday_range"
INTRADAY_DAYRANGE_KEY = "intraday_dayrange"
TRADER_LEVELS_KEY = "trader_levels"
HIGHLOW_RANGE_KEY = "highlow_range"


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
            "to its own forecast of the day's range: widest at 09:30, narrowest at "
            "midday, opening up again into the close. Known at the open."
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
        models=(apple_models.HIGHLOW_KEY,),
    ),
    TRADER_LEVELS_KEY: ModelOverlay(
        key=TRADER_LEVELS_KEY,
        label="Apple Trader buy/sell levels (agent orders, not a forecast)",
        name="Apple Trader",
        summary=(
            "Where the configured Apple Trader rests its two orders: the buy and the "
            "sell, each a distance in ADRs under the day-range forecast. The only "
            "overlay here that draws an agent's orders rather than a model's answer."
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
    frame = momentum_regime.frame_from_bars(bars)
    if not len(frame):
        return {"items": items, "notes": ["No regular-session bars to draw on."]}

    day = pd.Timestamp(session_date).normalize() if session_date is not None else None
    if day is None:
        day = pd.Timestamp(frame.index[-1]).normalize().tz_localize(None)
    session = frame[frame.index.normalize().tz_localize(None) == day]
    if not len(session):
        return {"items": items, "notes": [f"No bars for {day.date()}."]}

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

    def highlow_forecast() -> dict:
        if "highlow" not in memo:
            memo["highlow"] = _highlow_forecast(symbol, session, day, credentials)
        return memo["highlow"]

    def trader_forecast() -> dict:
        if trader_model(trader_config, symbol) == apple_models.HIGHLOW_KEY:
            return highlow_forecast()
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
        if recorded is not None:
            rows, config = recorded
            return _trader_levels_items(
                symbol, session, day, open_price, _recorded_result(rows), config,
                levels=rows,
            )
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


def _highlow_forecast(
    symbol: str,
    session: pd.DataFrame,
    day: pd.Timestamp,
    credentials: "tuple[str, str] | None",
) -> dict:
    """HighLow's forecast for the session, or why there is none -- the same
    `{"forecast", "made_at", "problem"}` shape `_day_range_forecast` returns.

    Its history is SIP minute bars strictly before `day`, read from the
    model's own cache (`highlow_model.history_frame`), so a past session is
    drawn with what the model could have said that morning.
    """
    def failed(problem: str) -> dict:
        return {"forecast": None, "made_at": None, "problem": problem}

    try:
        bundle = apple_models.load(apple_models.HIGHLOW_KEY, symbol)
        if bundle is None:
            return failed(apple_models.unavailable_reason(apple_models.HIGHLOW_KEY, symbol))

        from . import highlow_model  # heavy (torch + LightGBM); only once it is needed

        want = highlow_model.opening_minutes(bundle)
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
        key, secret = credentials or (None, None)
        forecast = highlow_model.forecast_session(bundle, symbol, opening, day, key, secret)
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

    `key` is the overlay being drawn: `day_range` (TimeToChange3) or
    `highlow_range`, which differ only in whose forecast `result` holds.

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

    if walked is None:
        walked = _day_range_walk(symbol, session, x0, open_price, forecast, config)
    highs = [row["pred_high"] for row in walked]
    lows = [row["pred_low"] for row in walked]
    moves = bool(walked) and (min(highs) != max(highs) or min(lows) != max(lows))
    if not moves:
        return (
            [
                _span(
                    key, "Predicted day range", x0, x1, color,
                    y0=low, y1=high,
                    note=f"{low:.2f} – {high:.2f} ({made_at})",
                ),
                _level(key, "Pred. high", high, color,
                       note=f"predicted session high ({made_at})", x0=x0, x1=x1),
                _level(key, "Pred. low", low, color,
                       note=f"predicted session low ({made_at})", x0=x0, x1=x1),
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
        AppleTraderConfig, session_levels, stop_phrase,
    )

    overlay = OVERLAYS[TRADER_LEVELS_KEY]
    forecast = result["forecast"]
    if forecast is None:
        return [], f"{overlay.label}: {result['problem']}"
    if config is None or (config.ticker or "").upper() != symbol:
        # A configuration for another symbol is not this chart's strategy, and
        # its distances were swept on that symbol's tape. The shipped ones are.
        config = AppleTraderConfig(ticker=symbol, model_key=trader_model(None, symbol))

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
    how = (
        f"buy {config.buy_k:g} × and sell {config.sell_k:g} × the {config.unit_phrase} under "
        + ("the predicted high" if config.level_source != "intraday"
           else "the intraday band's upper curve")
        + (" (HighLow)" if config.model_key == apple_models.HIGHLOW_KEY else "")
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
    moves = min(buys) != max(buys) or min(sells) != max(sells)
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
    holding the position when it trades this symbol, otherwise the first racer
    on it in the race's order. Its own record when it has one, otherwise the
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
    holder = race.get("holder")
    key = holder if holder in on_symbol else on_symbol[0]
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
    today = datetime.now(timezone.utc).astimezone(market_hours.MARKET_TZ).date()
    is_today = session_date is not None and session_date.date() == today
    return {
        "dayrange_daily_bars": historical.fetch_daily_ohlc_bars(symbol),
        "open_price": historical.fetch_session_open(symbol) if is_today else None,
    }
