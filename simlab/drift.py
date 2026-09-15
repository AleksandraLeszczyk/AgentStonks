"""How a saved model's accuracy moves over time, against where its training data ended.

A model's metadata carries one number for how good it is -- a held-out MAE, a
walk-forward EMD -- measured once, on a window that ended before the model was
saved. This module asks the question that number cannot answer: on the
sessions SimLab has stored since, is the model still that good? It scores each
stored session the way a live run would have seen it and compares the answer
with what the day then did, so the Drift tab can plot the metric session by
session or week by week beside the dates the training data stopped.

What each model is scored on
----------------------------
`dayrange`      TimeToChange3's predicted session high and low against the
                day's actual high and low (the stored daily bar), in the log
                units its test window reports. Needs the opening minutes, so
                only sessions with stored minute bars.
`intraday_vol`  IntradayVolatility's daily-bar forecast of the log day range
                against the day's ln(high/low) -- daily bars only, so the whole
                stored daily history -- plus how well each session's 5-minute
                ranges follow the fitted time-of-day curve, where minute bars
                exist.
`open_profile`  the LevelsML density model's predicted volume quantiles against
                the session's realised profile: the earth mover's distance
                LevelsML scores it with, and the share of volume inside the
                predicted outer quantiles.

Point in time
-------------
A session is scored from daily bars dated strictly before it, its official
opening print and -- for the models that read them -- its own opening minutes.
The day's high, low and volume are read only to score the answer, never to
make it. A forecast built with the outcome in reach would measure nothing.

Training cutoffs
----------------
Each model's saved metadata says where its training data ended, and every model
here has more than one stage, so there can be more than one cutoff. They are
read, not assumed: see `dayrange_training_from_metadata` and its siblings. A
session on or before the last cutoff was seen in training by at least one
stage, and the tab shades that stretch -- a model doing well there is fitting,
not generalising.
"""
from __future__ import annotations

import gzip
import json
import math
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Callable

import numpy as np

from agent_stonks import (
    apple_models,
    intraday_vol_model,
    model_catalogue,
    momentum_regime,
    profile_model,
)

from . import data as sim_data

DAY = "day"
WEEK = "week"


@dataclass(frozen=True)
class Metric:
    """One per-session number a model is tracked by."""

    key: str
    label: str
    unit: str = ""
    fmt: str = "%.4f"     # printf, for tables and cards
    hover: str = ".4f"    # d3, for plotly hovers
    better: str = "lower"  # "lower", "higher" or "zero"
    help: str = ""


@dataclass(frozen=True)
class Cutoff:
    """The last session one training stage saw."""

    date: str
    label: str
    note: str = ""


@dataclass(frozen=True)
class Reference:
    """A number the model's own evaluation reported, in a metric's units."""

    metric: str
    value: float
    label: str
    note: str = ""


@dataclass(frozen=True)
class DriftModel:
    key: str
    label: str
    summary: str
    # The symbols a model exists for, or None for one fitted to transfer.
    tickers: "tuple[str, ...] | None"
    # Whether a session can only be scored where its minute bars are stored.
    needs_minute_bars: bool
    metrics: "tuple[Metric, ...]"
    # (ticker, feed) -> {"rows": [{"date": iso, <metric>: value, ...}], "notes": [...]}
    evaluate: Callable[..., dict]
    # ticker -> {"cutoffs": [Cutoff], "references": [Reference]}
    training: Callable[[str], dict]


# --- the store --------------------------------------------------------------


def _minute_files(symbol: str, feed: str) -> "dict[date, Path]":
    """Every stored minute-bar file for one symbol on one tape, by day.

    `iex` also reads the pre-feed flat layout, as `simlab.data` does; a
    feed-scoped file wins over a legacy one for the same day.
    """
    symbol = symbol.upper()
    anchor = date(2000, 1, 3)
    dirs = [sim_data.bars_path(symbol, anchor, feed).parent]
    if feed == sim_data.LEGACY_FEED:
        dirs.insert(0, sim_data._legacy_bars_path(symbol, anchor).parent)
    found: "dict[date, Path]" = {}
    for directory in dirs:
        for path in directory.glob("*.json.gz") if directory.exists() else ():
            try:
                found[date.fromisoformat(path.name[:10])] = path
            except ValueError:
                continue
    return found


def stored_minute_days(symbol: str, feed: str) -> "list[date]":
    """Days with stored minute bars -- an empty file is a holiday, not a session."""
    return sorted(
        day for day in _minute_files(symbol, feed)
        if sim_data.load_day_bars(symbol.upper(), day, feed)
    )


def stored_symbols(feed: str) -> "list[str]":
    """Symbols with minute or daily bars stored on this tape."""
    store = sim_data.STORE_DIR
    found: "set[str]" = set()
    bars_dir, daily_dir = store / "bars" / feed, store / "daily" / feed
    if bars_dir.exists():
        found |= {p.name.upper() for p in bars_dir.iterdir() if p.is_dir()}
    if daily_dir.exists():
        found |= {p.name.split(".")[0].upper() for p in daily_dir.glob("*.json.gz")}
    if feed == sim_data.LEGACY_FEED:
        for p in (store / "bars").iterdir() if (store / "bars").exists() else ():
            if p.is_dir() and p.name not in sim_data.FEEDS:
                found.add(p.name.upper())
        for p in (store / "daily").glob("*.json.gz") if (store / "daily").exists() else ():
            found.add(p.name.split(".")[0].upper())
    return sorted(found)


def store_signature(symbol: str, feed: str) -> tuple:
    """What changes when a day is added or re-downloaded -- a cache key."""
    files = sorted(_minute_files(symbol, feed).values())
    files.append(sim_data.stored_daily_path(symbol.upper(), feed))
    out = []
    for path in files:
        try:
            stat = path.stat()
        except OSError:
            continue
        out.append((path.name, stat.st_size, stat.st_mtime_ns))
    return tuple(out)


def _daily(symbol: str, feed: str) -> "list[dict]":
    by_date = {str(b.get("t", ""))[:10]: b for b in sim_data.load_daily_bars(symbol.upper(), feed)}
    return [by_date[k] for k in sorted(by_date)]


def _session_frame(symbol: str, day: date, feed: str):
    """The day's regular-session minute frame."""
    frame = momentum_regime.frame_from_bars(sim_data.load_day_bars(symbol.upper(), day, feed))
    return frame[frame.index.date == day] if len(frame) else frame


def _failures_to_notes(failures: "dict[str, list[str]]") -> "list[str]":
    return [
        f"{len(days)} session{'s' if len(days) != 1 else ''} not scored — {reason} "
        f"({', '.join(days[:6])}{'…' if len(days) > 6 else ''})."
        for reason, days in failures.items()
    ]


# --- grouping ---------------------------------------------------------------


def aggregate(rows: "list[dict]", metric: str, by: str) -> "list[dict]":
    """Scored sessions grouped by day or by Monday-to-Sunday week, in time order.

    Each group: `start` (the day, or the week's Monday), `end` (its last
    session), `label`, `n`, `mean`, `min`, `max` and its `sessions`.
    """
    scored = sorted((r for r in rows if r.get(metric) is not None), key=lambda r: r["date"])
    groups: "dict[str, list[dict]]" = {}
    for row in scored:
        day = date.fromisoformat(row["date"])
        start = day if by == DAY else day - timedelta(days=day.weekday())
        groups.setdefault(start.isoformat(), []).append(row)
    out = []
    for start, members in groups.items():
        values = [float(m[metric]) for m in members]
        first = date.fromisoformat(start)
        label = (
            f"{first:%a} {first.day} {first:%b %Y}" if by == DAY
            else f"Week of {first.day} {first:%b %Y}"
        )
        out.append({
            "start": start,
            "end": members[-1]["date"],
            "label": label,
            "n": len(values),
            "mean": float(np.mean(values)),
            "min": min(values),
            "max": max(values),
            "sessions": [m["date"] for m in members],
        })
    return out


def in_training(day: str, cutoffs: "list[Cutoff]") -> bool:
    """Whether a session is on or before the last cutoff -- seen by some stage."""
    return bool(cutoffs) and day <= max(c.date for c in cutoffs)


def split_by_cutoff(rows: "list[dict]", metric: str, cutoffs: "list[Cutoff]") -> dict:
    """The metric's mean inside the training data and after its last cutoff."""
    parts: "dict[str, list[float]]" = {"inside": [], "after": []}
    for row in rows:
        if row.get(metric) is None:
            continue
        parts["inside" if in_training(row["date"], cutoffs) else "after"].append(float(row[metric]))
    return {
        name: {"mean": float(np.mean(values)) if values else None, "n": len(values)}
        for name, values in parts.items()
    }


def _previous_weekday(day: date) -> date:
    day -= timedelta(days=1)
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return day


def _read_json(path: "Path | None") -> dict:
    try:
        return json.loads(Path(path).read_text()) if path else {}
    except (OSError, ValueError):
        return {}


# --- day range (TimeToChange3) ----------------------------------------------


DAYRANGE_METRICS = (
    Metric("mae", "Mean absolute error of the high and low", "(log units)",
           help="|log(predicted ÷ actual)| for the session high and for the low, averaged — "
                "the units TimeToChange3 reports its held-out test error in."),
    Metric("mae_usd", "Mean absolute error in dollars", "(USD)", "%.2f", ".2f",
           help="The same error in price: how far the predicted high and low were from the "
                "day's, averaged. Tracks the share price as much as the model."),
    Metric("abs_err_high", "Absolute error of the high", "(log units)"),
    Metric("abs_err_low", "Absolute error of the low", "(log units)"),
    Metric("bias_high", "Bias of the high (predicted − actual)", "(log units)", better="zero",
           help="Above zero: the model expected a higher high than the day printed."),
    Metric("bias_low", "Bias of the low (predicted − actual)", "(log units)", better="zero",
           help="Above zero: the model expected the day to hold up higher than it did."),
)


def dayrange_training_from_metadata(meta: dict) -> dict:
    """Cutoffs and references out of a day-range bundle's JSON sidecar.

    TimeToChange3 (notebook 04) fits two stages on two different windows: the
    daily models on every session before the minute data starts
    (`daily_fit_through`), and -- where the bundle has one -- the opening ridge
    on the minute sessions before its held-out simulation day (`held_out`),
    which is where its minute data stopped counting as training.
    """
    cutoffs: "list[Cutoff]" = []
    references: "list[Reference]" = []
    if not meta:
        return {"cutoffs": cutoffs, "references": references}
    if meta.get("daily_fit_through"):
        cutoffs.append(Cutoff(
            str(meta["daily_fit_through"])[:10], "Daily models trained through",
            "LightGBM, N-BEATS and N-HiTS on daily bars up to the start of the minute data",
        ))
    if meta.get("opening_correction") and meta.get("held_out"):
        held_out = date.fromisoformat(str(meta["held_out"])[:10])
        cutoffs.append(Cutoff(
            _previous_weekday(held_out).isoformat(), "Opening ridge trained through",
            f"fitted on {meta.get('opening_fit_sessions', '?')} minute sessions before the "
            f"held-out day {held_out}",
        ))
    test = meta.get("test_metrics_ensemble") or {}
    for metric, field in (
        ("mae", "mae_mean"), ("mae_usd", "mae_usd_mean"),
        ("abs_err_high", "mae_y_high"), ("abs_err_low", "mae_y_low"),
        ("bias_high", "bias_y_high"), ("bias_low", "bias_y_low"),
    ):
        if test.get(field) is not None:
            references.append(Reference(
                metric, float(test[field]), "Held-out test window",
                "TimeToChange3's ~130-session test window, which ends before the minute data",
            ))
    return {"cutoffs": cutoffs, "references": references}


def dayrange_training(ticker: str) -> dict:
    spec = model_catalogue.spec(apple_models.DAYRANGE_KEY, ticker)
    path = next((f.path for f in spec.files if f.role == "metadata"), None) if spec else None
    return dayrange_training_from_metadata(_read_json(path))


def evaluate_dayrange(ticker: str, feed: str) -> dict:
    """Every stored minute session's forecast against its actual high and low."""
    symbol = ticker.upper()
    bundle = apple_models.load(apple_models.DAYRANGE_KEY, symbol)
    if bundle is None:
        return {"rows": [], "notes": [
            apple_models.unavailable_reason(apple_models.DAYRANGE_KEY, symbol)
        ]}
    from agent_stonks import dayrange_model  # torch + LightGBM, only once needed

    daily = _daily(symbol, feed)
    by_date = {str(b["t"])[:10]: b for b in daily}
    want = dayrange_model.opening_minutes(bundle)
    rows: "list[dict]" = []
    failures: "dict[str, list[str]]" = {}
    for day in stored_minute_days(symbol, feed):
        iso = day.isoformat()
        outcome = by_date.get(iso)
        if outcome is None:
            failures.setdefault("no stored daily bar to score the day against", []).append(iso)
            continue
        session = _session_frame(symbol, day, feed)
        if len(session) < want or float(session["minutes_from_open"].iloc[0]) >= 1.0:
            failures.setdefault("the opening minutes are not stored", []).append(iso)
            continue
        prior = [b for b in daily if str(b["t"])[:10] < iso]
        try:
            forecast = dayrange_model.forecast_session(
                bundle, dayrange_model.daily_frame_from_bars(prior), session.iloc[:want],
                day, open_price=float(outcome["o"]),
            )
        except ValueError as exc:
            failures.setdefault(str(exc).rstrip("."), []).append(iso)
            continue
        high, low = float(outcome["h"]), float(outcome["l"])
        err_high = math.log(forecast["pred_high"] / high)
        err_low = math.log(forecast["pred_low"] / low)
        rows.append({
            "date": iso,
            "mae": (abs(err_high) + abs(err_low)) / 2,
            "mae_usd": (abs(forecast["pred_high"] - high) + abs(forecast["pred_low"] - low)) / 2,
            "abs_err_high": abs(err_high),
            "abs_err_low": abs(err_low),
            "bias_high": err_high,
            "bias_low": err_low,
            "pred_high": forecast["pred_high"],
            "pred_low": forecast["pred_low"],
            "actual_high": high,
            "actual_low": low,
        })
    return {"rows": rows, "notes": _failures_to_notes(failures)}


# --- intraday volatility (IntradayVolatility) -------------------------------


INTRADAY_VOL_METRICS = (
    Metric("abs_log_range_err", "Absolute error of the log day range", "(log units)",
           "%.3f", ".3f",
           help="|log predicted − log actual| of the day's ln(high ÷ low). Daily bars only, "
                "so every stored daily session with a month of history behind it is scored."),
    Metric("log_range_bias", "Bias of the log day range (predicted − actual)", "(log units)",
           "%.3f", ".3f", better="zero",
           help="Above zero: the model expected a wider day than it got."),
    Metric("shape_corr", "Time-of-day shape correlation", "", "%.2f", ".2f", better="higher",
           help="Correlation between the session's log 5-minute high-low ranges and the fitted "
                "time-of-day curve. Scored only on sessions with stored minute bars."),
)


def intraday_vol_training_from_model(model: dict) -> dict:
    """Cutoffs from the export's two samples; the in-sample error as a reference."""
    day_range = (model or {}).get("day_range") or {}
    shape = (model or {}).get("shape") or {}
    cutoffs: "list[Cutoff]" = []
    references: "list[Reference]" = []
    if len(day_range.get("sample") or []) == 2:
        start, end = day_range["sample"]
        cutoffs.append(Cutoff(str(end)[:10], "Day-range HAR trained through",
                              f"refitted on daily ranges from {str(start)[:10]}"))
    if len(shape.get("sample") or []) == 2:
        start, end = shape["sample"]
        cutoffs.append(Cutoff(str(end)[:10], "Time-of-day curve fitted through",
                              f"on the diurnal profile from {str(start)[:10]}"))
    if day_range.get("residual_sd") is not None:
        references.append(Reference(
            "abs_log_range_err", float(day_range["residual_sd"]) * math.sqrt(2 / math.pi),
            "In-sample fit", "the mean absolute error a normal residual of the fitted spread gives",
        ))
    return {"cutoffs": cutoffs, "references": references}


def intraday_vol_training(ticker: str) -> dict:
    return intraday_vol_training_from_model(intraday_vol_model.load(ticker) or {})


def shape_correlation(model: dict, bars: "list[dict]") -> "float | None":
    """How closely one session's 5-minute ranges follow the fitted curve (log-log)."""
    frame = momentum_regime.frame_from_bars(bars)
    if len(frame) < 30:
        return None
    bins = (frame["minutes_from_open"] // 5).astype(int)
    grouped = frame.groupby(bins).agg(high=("high", "max"), low=("low", "min"))
    ranges = np.log(grouped["high"] / grouped["low"]).clip(lower=intraday_vol_model.MIN_RANGE)
    if len(ranges) < 3:
        return None
    realised = np.log(ranges.to_numpy(float))
    fitted = np.log(intraday_vol_model.relative_volatility(model, grouped.index.to_numpy() * 5 + 2))
    if np.std(realised) == 0 or np.std(fitted) == 0:
        return None
    return float(np.corrcoef(realised, fitted)[0, 1])


def evaluate_intraday_vol(ticker: str, feed: str) -> dict:
    symbol = ticker.upper()
    model = intraday_vol_model.load(symbol)
    if model is None:
        return {"rows": [], "notes": [
            f"No IntradayVolatility model at {intraday_vol_model.model_path(symbol)} — export "
            "it with FinNotebooks/IntradayVolatility/scripts/export_app_model.py."
        ]}
    daily = _daily(symbol, feed)
    minute_days = {d.isoformat() for d in stored_minute_days(symbol, feed)}
    rows: "list[dict]" = []
    for i, bar in enumerate(daily):
        iso = str(bar["t"])[:10]
        if i < intraday_vol_model.MONTH:
            continue
        try:
            features = intraday_vol_model.day_range_features(daily[:i], iso, float(bar["o"]))
            high, low = float(bar["h"]), float(bar["l"])
        except (KeyError, TypeError, ValueError):
            continue
        predicted = intraday_vol_model.predict_log_range(model, features)
        actual = math.log(max(math.log(high / low), intraday_vol_model.MIN_RANGE))
        rows.append({
            "date": iso,
            "abs_log_range_err": abs(predicted - actual),
            "log_range_bias": predicted - actual,
            "pred_range_pct": 100.0 * (math.exp(math.exp(predicted)) - 1.0),
            "actual_range_pct": 100.0 * (high / low - 1.0),
            "shape_corr": (
                shape_correlation(model, sim_data.load_day_bars(symbol, date.fromisoformat(iso), feed))
                if iso in minute_days else None
            ),
        })
    notes = []
    training = intraday_vol_training_from_model(model)
    if rows and training["cutoffs"] and all(in_training(r["date"], training["cutoffs"]) for r in rows):
        last = max(c.date for c in training["cutoffs"])
        notes.append(
            f"Every scored session is inside this model's training sample (through {last}), so "
            "this shows how well it fits, not how it has drifted, until later sessions are stored."
        )
    return {"rows": rows, "notes": notes}


# --- open price profile (LevelsML) ------------------------------------------


OPEN_PROFILE_METRICS = (
    Metric("emd_bps", "Earth mover's distance to the realised profile", "(bps)", "%.1f", ".1f",
           help="The volume-quantile weighted |predicted − realised| LevelsML scores the model "
                "with, in bps of the open. The realised profile here is built from stored minute "
                "bars; LevelsML built its from 1-hour bars, so its walk-forward number is a close "
                "reference, not an identical one."),
    Metric("inside_band", "Volume inside the predicted outer quantiles", "(%)", "%.1f", ".1f",
           better="higher",
           help="Share of the session's regular-hours volume that traded between the predicted "
                "lowest and highest volume-quantile prices. A calibrated band holds the nominal "
                "share."),
)


def emd_weights(levels) -> np.ndarray:
    """Trapezoid weights over the quantile levels, as `train_open_profile.py` builds them."""
    levels = np.asarray(levels, float)
    edges = np.concatenate([[0.0], (levels[:-1] + levels[1:]) / 2, [100.0]])
    return np.diff(edges) / 100.0


def _typical_prices_and_volume(bars: "list[dict]"):
    frame = momentum_regime.frame_from_bars(bars)
    if not len(frame):
        return None, None
    price = ((frame["high"] + frame["low"] + frame["close"]) / 3.0).to_numpy(float)
    volume = frame["volume"].to_numpy(float)
    return (price, volume) if volume.sum() > 0 else (None, None)


def realised_quantiles(bars: "list[dict]", open_price: float, levels) -> "np.ndarray | None":
    """The session's volume quantiles in bps of the open, from minute bars.

    Each regular-session bar's volume sits at its typical price; the quantile
    function is interpolated through the cumulative volume share exactly as
    `train_open_profile.day_quantiles` does it.
    """
    price, volume = _typical_prices_and_volume(bars)
    if price is None:
        return None
    rel = np.log(price / open_price) * 1e4
    order = np.argsort(rel)
    rel, volume = rel[order], volume[order]
    cumulative = np.cumsum(volume) / volume.sum()
    return np.interp(np.asarray(levels, float) / 100.0, cumulative, rel)


def volume_share_inside(bars: "list[dict]", low: float, high: float) -> "float | None":
    price, volume = _typical_prices_and_volume(bars)
    if price is None:
        return None
    inside = (price >= low) & (price <= high)
    return float(volume[inside].sum() / volume.sum())


def open_profile_training_from_pack(pack: dict) -> dict:
    meta = (pack or {}).get("metadata") or {}
    cutoffs: "list[Cutoff]" = []
    references: "list[Reference]" = []
    if len(meta.get("date_range") or []) == 2:
        start, end = meta["date_range"]
        cutoffs.append(Cutoff(
            str(end)[:10], "Trained through",
            f"sessions from {str(start)[:10]} across {', '.join(meta.get('universe') or []) or '?'}",
        ))
    walk_forward = meta.get("walk_forward_emd_bps") or {}
    if walk_forward.get("lgbm live-features") is not None:
        references.append(Reference(
            "emd_bps", float(walk_forward["lgbm live-features"]), "Walk-forward EMD",
            "LevelsML's walk-forward score for this feature set, on 1-hour-bar profiles",
        ))
    if walk_forward.get("ATR climatology") is not None:
        references.append(Reference(
            "emd_bps", float(walk_forward["ATR climatology"]), "ATR climatology",
            "the baseline the model was measured against",
        ))
    levels = (pack or {}).get("p_levels") or []
    if len(levels) >= 2:
        references.append(Reference(
            "inside_band", float(levels[-1]) - float(levels[0]), "Nominal",
            f"the share between the q{levels[0]:g} and q{levels[-1]:g} predictions if calibrated",
        ))
    return {"cutoffs": cutoffs, "references": references}


def open_profile_training(ticker: str) -> dict:
    spec = model_catalogue.spec(model_catalogue.OPEN_PROFILE_KEY)
    path = spec.primary_path if spec else None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            pack = json.load(fh)
    except (OSError, TypeError, ValueError):
        pack = {}
    return open_profile_training_from_pack(pack)


def evaluate_open_profile(ticker: str, feed: str) -> dict:
    symbol = ticker.upper()
    pack = profile_model.load_pack()
    if pack is None:
        return {"rows": [], "notes": [
            "No open-profile pack (or LightGBM is not installed), so nothing can be scored."
        ]}
    levels = np.asarray(pack["p_levels"], float)
    weights = emd_weights(levels)
    daily = _daily(symbol, feed)
    by_date = {str(b["t"])[:10]: b for b in daily}
    rows: "list[dict]" = []
    failures: "dict[str, list[str]]" = {}
    for day in stored_minute_days(symbol, feed):
        iso = day.isoformat()
        bar = by_date.get(iso)
        if bar is None:
            failures.setdefault("no stored daily bar for the opening print", []).append(iso)
            continue
        open_price = float(bar["o"])
        prior = [b for b in daily if str(b["t"])[:10] < iso]
        features = profile_model.compute_features(prior, open_price, iso)
        if features is None:
            failures.setdefault(
                f"fewer than {profile_model.MIN_DAILY_BARS} daily bars of history", []
            ).append(iso)
            continue
        predicted = profile_model.predict_quantiles(pack, features)
        bars = sim_data.load_day_bars(symbol, day, feed)
        realised = realised_quantiles(bars, open_price, levels)
        if predicted is None or realised is None:
            failures.setdefault("no regular-session volume stored", []).append(iso)
            continue
        predicted = np.sort(np.asarray(predicted, float))
        inside = volume_share_inside(
            bars, open_price * math.exp(predicted[0] / 1e4), open_price * math.exp(predicted[-1] / 1e4)
        )
        rows.append({
            "date": iso,
            "emd_bps": float((np.abs(predicted - realised) * weights).sum()),
            "inside_band": 100.0 * inside if inside is not None else None,
        })
    return {"rows": rows, "notes": _failures_to_notes(failures)}


# --- the catalogue ----------------------------------------------------------


MODELS: "dict[str, DriftModel]" = {
    apple_models.DAYRANGE_KEY: DriftModel(
        key=apple_models.DAYRANGE_KEY,
        label=apple_models.get(apple_models.DAYRANGE_KEY).label,
        summary=(
            "The predicted session high and low against the day's actual ones, on every stored "
            "session with its opening minutes. Its daily models stopped learning at the start "
            "of the minute data and its opening ridge at the held-out day, so the sessions after "
            "both are the ones it never saw."
        ),
        tickers=apple_models.DAYRANGE_TICKERS,
        needs_minute_bars=True,
        metrics=DAYRANGE_METRICS,
        evaluate=evaluate_dayrange,
        training=dayrange_training,
    ),
    "intraday_vol": DriftModel(
        key="intraday_vol",
        label="Intraday volatility (IntradayVolatility)",
        summary=(
            "The daily-bar forecast of how wide each day will be, over the whole stored daily "
            "history, and how well each session's intraday volatility follows the fitted "
            "time-of-day curve where minute bars exist."
        ),
        tickers=intraday_vol_model.TICKERS,
        needs_minute_bars=False,
        metrics=INTRADAY_VOL_METRICS,
        evaluate=evaluate_intraday_vol,
        training=intraday_vol_training,
    ),
    model_catalogue.OPEN_PROFILE_KEY: DriftModel(
        key=model_catalogue.OPEN_PROFILE_KEY,
        label="Open price profile (LevelsML)",
        summary=(
            "Where the session's volume was predicted to trade, at the open, against where it "
            "did — on every stored session with minute bars. Fitted to transfer, so any stored "
            "symbol can be scored."
        ),
        tickers=None,
        needs_minute_bars=True,
        metrics=OPEN_PROFILE_METRICS,
        evaluate=evaluate_open_profile,
        training=open_profile_training,
    ),
}
