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

Which metric stands for a model
-------------------------------
Each model tracks several numbers, but one of them is the model's: the
session-level version of what the ML Models tab prints in its headline column,
named by `DriftModel.headline`. For the day range and the open profile that is
literally the same quantity the saved file was graded on -- a MAE in log units,
an EMD in bps -- so the reference line on the chart is that grade and the series
is the same measurement taken later. `intraday_vol` is graded by a walk-forward
R², which is a statistic of a window rather than of a session, so what is
tracked is the error term inside it; `DriftModel.catalogue_metric` says so on
screen rather than letting the two pages look interchangeable when they are not.

Has it actually moved?
----------------------
Twenty noisy sessions will always look like they are going somewhere, so the
answer is a test rather than a slope: `trend_test` (Mann-Kendall over the scored
sessions) and, where the store straddles a cutoff, `split_test` (Mann-Whitney U
either side of it). Both are rank tests on per-session values -- see the comment
above `ALPHA` for why rank, why per session, and why both.
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
    # The name for a table cell, units and all. Set rather than clipped off
    # `label`: these run to half a sentence, and a column narrow enough for
    # nine of them beside two verdicts truncates every one mid-word.
    short: str = ""

    @property
    def name(self) -> str:
        return self.short or f"{self.label} {self.unit}".strip()


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
    # The one metric this model is summarised by -- the session-level version of
    # the number the ML Models tab puts in its headline column. See
    # `catalogue_metric` for how exactly the two line up.
    headline: str = ""
    # What the ML Models tab calls that number, and whether the two are the same
    # quantity or only the same subject. `model_catalogue` is the source of the
    # value; this says what it can be compared with.
    catalogue_metric: str = ""

    @property
    def headline_metric(self) -> Metric:
        return next(m for m in self.metrics if m.key == (self.headline or self.metrics[0].key))


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


def model_signature(model_key: str, ticker: str) -> tuple:
    """What changes when a model is retrained -- the other half of a cache key.

    Stat calls only, and read through `model_catalogue` because that module is
    deliberately a reader: asking the model modules where their files are would
    import PyTorch to answer a question about an mtime.
    """
    spec = model_catalogue.spec(model_key, ticker)
    out = []
    for entry in spec.files if spec else ():
        try:
            stat = entry.path.stat()
        except OSError:
            continue
        out.append((entry.path.name, stat.st_size, stat.st_mtime_ns))
    return tuple(out)


def signature(model_key: str, ticker: str, feed: str) -> tuple:
    """Everything one scoring depends on: the stored bars and the saved model.

    Both halves, so a re-downloaded day and a retrained bundle each invalidate
    the answer -- the second is why the tab needs no "the model file changed"
    button of its own.
    """
    return (store_signature(ticker, feed), model_signature(model_key, ticker))


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


# --- is it drifting? --------------------------------------------------------
#
# A metric that wanders is not the same as a metric that has moved, and the eye
# is bad at telling them apart on twenty noisy sessions. Two tests, because the
# two ways a saved model goes wrong look different on a chart and because
# neither test can be run on every model:
#
# * a **trend** over the scored sessions (Mann-Kendall), which is the only one
#   that can always be run -- `intraday_vol` has every stored session inside its
#   training sample and `open_profile` has none of them, so a before/after split
#   is undefined for two of the three models;
# * a **split** at the last training cutoff (Mann-Whitney U), for the model
#   where the store straddles it, which is the sharper question when it can be
#   asked: is the model worse on the sessions it never saw?
#
# Both are rank tests on the per-session values. Rank tests because these
# metrics are absolute errors -- bounded below, long-tailed above -- so one wild
# session moves a mean and a t-test far more than it should. Per session, not
# per displayed group: a session is the observation, and a weekly grouping of
# six points has no power to reject anything (Mann-Kendall needs |tau| > 0.85
# at n=6). Grouping is a way to read the chart, not a way to run the test.
#
# The normal approximations below are the standard ones, with tie corrections,
# and are used rather than SciPy's exact versions because this module otherwise
# needs nothing but numpy -- and because at these sample sizes the difference is
# in the third decimal of a p-value nobody should be reading that closely.

ALPHA = 0.05
# Below these the normal approximation is not worth printing a p-value for.
MIN_TREND_N = 8
MIN_SPLIT_N = 5

TREND = "trend"
SPLIT = "split"


@dataclass(frozen=True)
class TestResult:
    """One answer to "has this metric changed?", with what it could not answer."""

    kind: str
    label: str
    # Rank statistic: Kendall's tau-b for the trend, rank-biserial for the split.
    # Both run -1..1 and both are positive when the metric is *rising*.
    statistic: "float | None"
    p: "float | None"
    # The change in the metric's own units: per 30 days for the trend
    # (Theil-Sen), after minus inside for the split (medians).
    effect: "float | None"
    effect_label: str
    n: int
    # Why there is no p-value, when there is none.
    note: str = ""

    @property
    def significant(self) -> bool:
        return self.p is not None and self.p < ALPHA

    @property
    def direction(self) -> str:
        """"up", "down" or "flat" -- of the metric, not of the model's health."""
        if self.statistic is None or not self.significant:
            return "flat"
        return "up" if self.statistic > 0 else "down"


def _normal_sf(z: float) -> float:
    """P(Z > z) for a standard normal, to the precision `erfc` gives."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _two_sided_p(z: float) -> float:
    return min(1.0, 2.0 * _normal_sf(abs(z)))


def _ranks(values: np.ndarray) -> np.ndarray:
    """1-based ranks with ties averaged, as both tests below need them."""
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    ranks = np.empty(len(values), float)
    start = 0
    while start < len(values):
        stop = start
        while stop + 1 < len(values) and ordered[stop + 1] == ordered[start]:
            stop += 1
        ranks[order[start:stop + 1]] = (start + stop) / 2.0 + 1.0
        start = stop + 1
    return ranks


def _tie_groups(values: np.ndarray) -> "list[int]":
    _, counts = np.unique(values, return_counts=True)
    return [int(c) for c in counts if c > 1]


def _series(rows: "list[dict]", metric: str) -> "tuple[list[date], np.ndarray]":
    """The scored sessions in time order, as dates and values."""
    scored = sorted(
        (r for r in rows if r.get(metric) is not None), key=lambda r: r["date"]
    )
    return (
        [date.fromisoformat(r["date"]) for r in scored],
        np.array([float(r[metric]) for r in scored]),
    )


def _theil_sen(days: "list[date]", values: np.ndarray) -> "float | None":
    """Median pairwise slope, in the metric's units per 30 days.

    The robust slope rather than a least-squares one, for the same reason the
    test is a rank test: a single 3-sigma session should not set the trend.
    """
    slopes = [
        (values[j] - values[i]) / (days[j] - days[i]).days
        for i in range(len(values)) for j in range(i + 1, len(values))
        if days[j] != days[i]
    ]
    return float(np.median(slopes)) * 30.0 if slopes else None


def trend_test(rows: "list[dict]", metric: str) -> TestResult:
    """Mann-Kendall on the per-session values: is the metric going anywhere?

    S counts how many later sessions are above earlier ones minus how many are
    below; under "no trend" it is symmetric about zero with the variance below,
    tie-corrected. Kendall's tau-b normalises it to -1..1, so it reads as an
    effect size rather than as a count that grows with the sample.
    """
    days, values = _series(rows, metric)
    n = len(values)
    if n < MIN_TREND_N:
        return TestResult(
            TREND, "Trend over the scored sessions", None, None, None,
            "per 30 days", n,
            f"fewer than {MIN_TREND_N} scored sessions ({n})",
        )
    signs = np.sign(values[None, :] - values[:, None])
    s = float(np.triu(signs, k=1).sum())
    ties = _tie_groups(values)
    variance = (
        n * (n - 1) * (2 * n + 5) - sum(t * (t - 1) * (2 * t + 5) for t in ties)
    ) / 18.0
    pairs_total = n * (n - 1) / 2.0
    # tau-b's denominator, which collapses to zero exactly when every session
    # ties every other one -- a flat series, where there is nothing to test.
    untied = pairs_total * (pairs_total - sum(t * (t - 1) / 2.0 for t in ties))
    if untied <= 0 or variance <= 0:
        return TestResult(
            TREND, "Trend over the scored sessions", None, None, None,
            "per 30 days", n, "every scored session has the same value",
        )
    tau = s / math.sqrt(untied)
    # The continuity correction pulls S one step toward zero before it is
    # standardised, which is what makes the approximation usable at n ~ 20.
    z = (s - math.copysign(1.0, s)) / math.sqrt(variance) if s else 0.0
    return TestResult(
        TREND, "Trend over the scored sessions", float(tau), _two_sided_p(z),
        _theil_sen(days, values), "per 30 days", n,
    )


def split_test(rows: "list[dict]", metric: str, cutoffs: "list[Cutoff]") -> TestResult:
    """Mann-Whitney U across the last training cutoff: is it worse where it is blind?

    The positive direction is *after* the cutoff, so a positive statistic always
    means the metric is larger on the sessions the model never saw. The effect
    is the difference of medians, in the metric's own units.
    """
    label = "After the last cutoff vs inside the training data"
    inside, after = [], []
    for row in rows:
        if row.get(metric) is None:
            continue
        (inside if in_training(row["date"], cutoffs) else after).append(float(row[metric]))
    if not cutoffs:
        return TestResult(
            SPLIT, label, None, None, None, "difference of medians", len(after),
            "no training cutoff in this model's file to split on",
        )
    if len(inside) < MIN_SPLIT_N or len(after) < MIN_SPLIT_N:
        total = len(inside) + len(after)
        if not inside or not after:
            # The usual case rather than an edge one: two of the three models
            # have the whole store on one side of their cutoff.
            note = f"all {total} sessions {'after' if inside == [] else 'inside'} the cutoff"
        else:
            note = f"{len(inside)} inside, {len(after)} after — needs {MIN_SPLIT_N} each side"
        return TestResult(
            SPLIT, label, None, None, None, "difference of medians", total, note,
        )
    a, b = np.array(after), np.array(inside)
    combined = np.concatenate([a, b])
    ranks = _ranks(combined)
    n1, n2 = len(a), len(b)
    u = float(ranks[:n1].sum()) - n1 * (n1 + 1) / 2.0
    mean_u = n1 * n2 / 2.0
    ties = _tie_groups(combined)
    total = n1 + n2
    correction = sum(t ** 3 - t for t in ties) / (total * (total - 1)) if total > 1 else 0.0
    variance = n1 * n2 / 12.0 * ((total + 1) - correction)
    if variance <= 0:
        return TestResult(
            SPLIT, label, None, None, None, "difference of medians", total,
            "every scored session has the same value",
        )
    z = (u - mean_u - math.copysign(0.5, u - mean_u)) / math.sqrt(variance)
    # Rank-biserial: the chance a session after the cutoff scores above one
    # inside it, rescaled to -1..1, which is a plain-language effect size.
    return TestResult(
        SPLIT, label, 2.0 * u / (n1 * n2) - 1.0, _two_sided_p(z),
        float(np.median(a) - np.median(b)), "difference of medians", total,
    )


def worsened(metric: Metric, statistic: "float | None") -> "bool | None":
    """Whether a rise in this metric is the model getting worse.

    `None` for a bias, where the sign says which way it leans and neither
    direction is the good one -- only zero is.
    """
    if statistic is None or metric.better == "zero":
        return None
    return (statistic > 0) if metric.better == "lower" else (statistic < 0)


def assess(rows: "list[dict]", metric: Metric, cutoffs: "list[Cutoff]") -> dict:
    """Both tests plus the cutoff split, for one model on one instrument."""
    return {
        "trend": trend_test(rows, metric.key),
        "split": split_test(rows, metric.key, cutoffs),
        "parts": split_by_cutoff(rows, metric.key, cutoffs),
        "n": sum(1 for r in rows if r.get(metric.key) is not None),
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
           short="MAE (log units)",
           help="|log(predicted ÷ actual)| for the session high and for the low, averaged — "
                "the units TimeToChange3 reports its held-out test error in."),
    Metric("mae_usd", "Mean absolute error in dollars", "(USD)", "%.2f", ".2f",
           short="MAE (USD)",
           help="The same error in price: how far the predicted high and low were from the "
                "day's, averaged. Tracks the share price as much as the model."),
    Metric("abs_err_high", "Absolute error of the high", "(log units)",
           short="|error| high (log)"),
    Metric("abs_err_low", "Absolute error of the low", "(log units)",
           short="|error| low (log)"),
    Metric("bias_high", "Bias of the high (predicted − actual)", "(log units)", better="zero",
           short="Bias, high (log)",
           help="Above zero: the model expected a higher high than the day printed."),
    Metric("bias_low", "Bias of the low (predicted − actual)", "(log units)", better="zero",
           short="Bias, low (log)",
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
           "%.3f", ".3f", short="|error| log range",
           help="|log predicted − log actual| of the day's ln(high ÷ low). Daily bars only, "
                "so every stored daily session with a month of history behind it is scored."),
    Metric("log_range_bias", "Bias of the log day range (predicted − actual)", "(log units)",
           "%.3f", ".3f", better="zero", short="Bias, log range",
           help="Above zero: the model expected a wider day than it got."),
    Metric("shape_corr", "Time-of-day shape correlation", "", "%.2f", ".2f", better="higher",
           short="Shape correlation",
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
           short="EMD (bps)",
           help="The volume-quantile weighted |predicted − realised| LevelsML scores the model "
                "with, in bps of the open. The realised profile here is built from stored minute "
                "bars; LevelsML built its from 1-hour bars, so its walk-forward number is a close "
                "reference, not an identical one."),
    Metric("inside_band", "Volume inside the predicted outer quantiles", "(%)", "%.1f", ".1f",
           better="higher", short="Volume inside band (%)",
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
        headline="mae",
        catalogue_metric=(
            "MAE (log units) — the same quantity the ML Models tab reports, on later "
            "sessions instead of the held-out test window"
        ),
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
        headline="abs_log_range_err",
        catalogue_metric=(
            "day range walk-forward R² — the ML Models tab's number for the same "
            "forecast, but an R² is a statistic of a whole window and has no value on "
            "a single session, so what is tracked here is that window's error term: "
            "the absolute error in log day range, session by session"
        ),
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
        headline="emd_bps",
        catalogue_metric=(
            "walk-forward EMD (bps) — the same quantity the ML Models tab reports, on "
            "profiles rebuilt from stored minute bars rather than LevelsML's 1-hour ones"
        ),
    ),
}


# --- every model at once ----------------------------------------------------


def scorable_symbols(feed: str, model: DriftModel) -> "list[str]":
    """The stored symbols one model can be scored on, on this tape."""
    return [
        s for s in stored_symbols(feed)
        if model.tickers is None or s in model.tickers
    ]


def default_symbols(feed: str) -> "list[str]":
    """The symbols worth scoring everything on without being asked.

    The ones a per-instrument model was actually fitted for. `open_profile`
    transfers to any stored symbol, so taking *its* list would quietly turn one
    click into a dozen scorings of the model that has the least to say about a
    symbol it never saw.
    """
    fitted: "set[str]" = set()
    for model in MODELS.values():
        if model.tickers is not None:
            fitted |= set(model.tickers)
    stored = stored_symbols(feed)
    return [s for s in stored if s in fitted] or stored[:1]


def pairs(feed: str, symbols: "list[str] | None" = None) -> "list[tuple[str, str]]":
    """Every (model, instrument) the store can score, in catalogue order.

    Model-major, so the Drift tab's sections come out in the order the ML
    Models tab lists them and an instrument that only one model exists for
    simply appears once.
    """
    stored = set(stored_symbols(feed))
    wanted = [s for s in (symbols if symbols is not None else stored_symbols(feed)) if s in stored]
    return [
        (key, symbol)
        for key, model in MODELS.items()
        for symbol in wanted
        if model.tickers is None or symbol in model.tickers
    ]
