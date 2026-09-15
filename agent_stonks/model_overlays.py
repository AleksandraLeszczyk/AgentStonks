"""What the saved models predict, in the shape a price chart can draw.

The ML surface of this app is a set of models that answer questions about a
session -- where its volume will trade, how wide it will be. Each of them already has a consumer (`profile_model`
feeds the profile curve, `apple_models` feeds the traders), but until now the
only way to see what a model said was to read a trader's log. This module is
the other way round: it asks each model its question about a session and
returns the answer as **drawing instructions**, so the live chart and SimLab's
replay chart can show the prediction beside the tape that tested it.

Four overlays
-------------
`day_range`          TimeToChange3's forecast of where the session's high and
                     low will land, made once from the first five minutes. Two
                     price levels and the band between them -- the model's
                     claim about the *width* of the day. See `dayrange_model`.
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

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from . import apple_models, intraday_vol_model, market_hours, momentum_regime, profile_model
from .config import MODEL_OVERLAY_COLORS


# --- the catalogue ----------------------------------------------------------

DAY_RANGE_KEY = "day_range"
PROFILE_RANGE_KEY = "profile_range"
INTRADAY_RANGE_KEY = "intraday_range"
INTRADAY_DAYRANGE_KEY = "intraday_dayrange"


@dataclass(frozen=True)
class ModelOverlay:
    """One model's prediction, as something a chart can offer to draw."""

    key: str
    label: str
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
        label="Predicted day range",
        summary=(
            "TimeToChange3's forecast of the session's high and low, made once from "
            "the first five minutes. Drawn as two levels and the band between them."
        ),
        requires="PyTorch, LightGBM and the day-range bundle",
        tickers=apple_models.DAYRANGE_TICKERS,
        models=(apple_models.DAYRANGE_KEY,),
    ),
    PROFILE_RANGE_KEY: ModelOverlay(
        key=PROFILE_RANGE_KEY,
        label="Predicted price profile range",
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
        label="Predicted intraday range",
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
        label="Predicted intraday range × day range",
        summary=(
            "The same time-of-day curve, stretched so it tops out at TimeToChange3's "
            "predicted high and bottoms out at its predicted low. Made at 09:35."
        ),
        requires=(
            "the IntradayVolatility export plus PyTorch, LightGBM and the day-range bundle"
        ),
        tickers=tuple(
            t for t in apple_models.DAYRANGE_TICKERS if intraday_vol_model.covers(t)
        ),
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
          dash: str = "dot", note: str = "") -> dict:
    """A price range that changes through the session.

    Where a `span` is one rectangle, a band is a pair of curves over the same
    timestamps -- a prediction of how wide the price can swing *at each time of
    day*. It is drawn as two edges with the range between them tinted, behind
    the candles, and like any non-forward span it is clipped to the bars in
    hand. There is no profile mirror: a curve through time has no single price
    to put on that axis.
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
        "color": color,
        "dash": dash,
        "note": note,
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
) -> dict:
    """Draw instructions for the requested overlays, plus why any are empty.

    `bars` are the session's 1-minute bars (`{"t","o","h","l","c","v"}`, the
    shape both the live buffer and SimLab's store carry) and `daily_bars` the
    daily history behind it -- rows dated on or after the session are ignored
    by the models that read it, so a caller may pass the whole store.

    `session_date` names the day being drawn; it defaults to the last bar's,
    which is what a live caller wants and what a single-day replay wants too.
    `open_price` is the official opening print when the caller has one.

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

    def day_range_forecast() -> dict:
        if "result" not in memo:
            memo["result"] = _day_range_forecast(
                symbol, session, daily_bars or [], day, open_price
            )
        return memo["result"]

    builders = {
        DAY_RANGE_KEY: lambda: _day_range_items(day_range_forecast(), day),
        PROFILE_RANGE_KEY: lambda: _profile_range_items(
            session, daily_bars or [], day
        ),
        INTRADAY_RANGE_KEY: lambda: _intraday_range_items(
            symbol, session, daily_bars or [], day, open_price
        ),
        INTRADAY_DAYRANGE_KEY: lambda: _intraday_dayrange_items(
            symbol, session, day, open_price, day_range_forecast()
        ),
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


def _day_range_items(result: dict, day: pd.Timestamp) -> "tuple[list[dict], str]":
    """The predicted high and low, and the band between them."""
    overlay = OVERLAYS[DAY_RANGE_KEY]
    forecast = result["forecast"]
    if forecast is None:
        return [], f"{overlay.label}: {result['problem']}"

    color = MODEL_OVERLAY_COLORS[DAY_RANGE_KEY]
    x0 = result["made_at"]
    x1 = _session_close(day)
    high, low = forecast["pred_high"], forecast["pred_low"]
    made_at = f"forecast at {pd.Timestamp(x0):%H:%M}"
    return (
        [
            _span(
                DAY_RANGE_KEY, "Predicted day range", x0, x1, color,
                y0=low, y1=high,
                note=f"{low:.2f} – {high:.2f} ({made_at})",
            ),
            _level(DAY_RANGE_KEY, "Pred. high", high, color,
                   note=f"predicted session high ({made_at})", x0=x0, x1=x1),
            _level(DAY_RANGE_KEY, "Pred. low", low, color,
                   note=f"predicted session low ({made_at})", x0=x0, x1=x1),
        ],
        "",
    )


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

    key = (bars[-1].get("t"), tuple(wanted), len(bars))
    cached = getattr(sym_state, "model_overlay_cache", None)
    if cached and cached.get("key") == key:
        return cached["result"]

    result = compute(
        wanted,
        sym_state.symbol,
        bars,
        daily_bars=list(sym_state.daily_bars or []),
        session_date=session_date_of(bars),
    )
    sym_state.model_overlay_cache = {"key": key, "result": result}
    return result
