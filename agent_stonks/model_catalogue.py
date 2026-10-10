"""Every trained model in this app, described from its own saved file.

The ML models here are each loaded by their own module, each for their own
consumer -- `apple_models` feeds the traders, `profile_model` feeds the price
profile, `model_overlays` feeds both charts. What none of them offer is an
answer to "what is actually installed, fitted on what, and how well does it
score?", which is the question the ML Models tab asks. This module is that
answer, and it is deliberately a *reader* rather than another model: everything
below comes out of the bundles and sidecars on disk, so a retrain changes the
tab without anyone editing it.

Read the file, not the model
----------------------------
A spec is assembled without importing PyTorch or LightGBM. That is the whole
design constraint, and it is not fussiness: `apple_models` goes out of its way
to keep a 200 MB dependency off the default start-up path, and a tab that lists
model names would undo that if it had to instantiate them to do it. So:

* `dayrange` is read from its **JSON sidecar**, which carries the metrics, the
  settings and the whole feature list. Its `model_path` helper lives in a
  module that imports torch at module scope, so the path is mirrored here
  instead -- see `_saved_path`.
* `open_profile` is a gzipped JSON pack, so its metadata is readable with the
  standard library alone even though scoring it needs LightGBM.

Availability is answered the same cheap way: `importlib.util.find_spec` for the
dependency plus `Path.exists` for the file. A model reported ready here has
still only been *found*, not loaded -- for the torch-backed day-range bundle
that is a weaker claim than it sounds.
The tab says so.

The mirror contract, again
--------------------------
`_saved_path` reproduces the `MODEL_DIR` / `<ENV>_<TICKER>` / `<ENV>` lookup
that each model module implements for itself. It is a copy, kept because the
alternative imports torch, and it carries the obligation every mirror does: `tests/test_model_catalogue.py`
pins every mirrored path against the real `model_path` it stands in for, so a
renamed file or a new env override fails a test rather than showing a user a
path nothing reads.
"""

from __future__ import annotations

import gzip
import json
import math
import os
import re
from dataclasses import dataclass, field
from importlib.util import find_spec
from pathlib import Path

from . import apple_models, codenames, intraday_vol_model, model_store

# The shared store beside the AgentStonks checkout, where every model family
# resolves its files from (see `model_store.ModelStore`).
MODEL_DIR = model_store.MODEL_DIR

OPEN_PROFILE_KEY = "open_profile"
# Not an `apple_models` key: it drives no agent, only the chart overlays.
INTRADAY_VOL_KEY = "intraday_vol"


@dataclass(frozen=True)
class ModelFile:
    """One file a model is assembled from, and whether it is there."""

    role: str
    path: Path

    @property
    def exists(self) -> bool:
        return self.path.exists()

    @property
    def size_mb(self) -> "float | None":
        try:
            return self.path.stat().st_size / 1e6
        except OSError:
            return None


@dataclass(frozen=True)
class ModelSpec:
    """What one trained model is, for one instrument.

    A spec is per (model, ticker) rather than per model: each symbol's bundle
    is fitted and measured on that symbol's own days, so one row per model
    would have to pick one of them to be true about.
    """

    key: str
    label: str
    # The instrument, short enough for a table cell: a symbol, or "any" for the
    # one pack fitted to transfer. What it was fitted on, when that is a
    # different thing, goes in `ticker_note`.
    ticker: str
    # The project the model came out of, which is where its notebooks are.
    project: str
    # The picker line -- what the model is, not how well it scores. The four
    # registry models take `AppleModel.summary` verbatim, so a model reads the
    # same here as it does where it is chosen.
    summary: str
    # What it predicts, at length, and the target in symbols where there is
    # one worth writing down.
    predicts: str
    target: str
    # Which tape the model saw, when that is not simply `ticker`. Only the
    # LevelsML pack has one: it was fitted across a small universe and
    # deliberately without ticker dummies, so the symbol it runs on and the
    # symbols it learned from are genuinely two different facts.
    ticker_note: str
    # The estimator, as specifically as the saved file says it.
    algorithm: str
    # The same thing in two or three words, for a table cell. Set explicitly
    # rather than clipped off `algorithm`: the saved files describe themselves
    # at wildly different lengths -- "Ridge" against a whole blend-plus-
    # correction sentence -- and truncating the long ones lands mid-phrase.
    family: str
    # What reads it: an Apple Trader strategy, a chart overlay, or both.
    consumers: "tuple[str, ...]"
    # The input side: named features, plus the window they are taken over.
    features: "tuple[str, ...]"
    inputs: str
    # Metric name -> value, in the order they should be read. Values stay
    # numeric where they are numeric so the UI can format them.
    metrics: "dict[str, object]" = field(default_factory=dict)
    # The metric worth putting in a summary row, as (name, formatted value).
    headline: "tuple[str, str] | None" = None
    files: "tuple[ModelFile, ...]" = ()
    trained_at: str = ""
    # Which sessions were held out, when the file says.
    data_note: str = ""
    # Library versions the file was written with, when it records them.
    versions: "dict[str, str]" = field(default_factory=dict)
    # The decision cut-off, for the models that have one.
    threshold: "float | None" = None
    requires: str = ""
    available: bool = False
    unavailable_reason: str = ""
    # The honest limitation -- what the headline number does not say.
    caveat: str = ""

    @property
    def n_features(self) -> int:
        return len(self.features)

    @property
    def primary_path(self) -> "Path | None":
        return self.files[0].path if self.files else None


# --- reading the files ------------------------------------------------------

def _saved_path(env_prefix: str, pattern: str, ticker: str) -> Path:
    """Where one ticker's file lives, mirroring the model modules' own lookup.

    `<ENV>_<TICKER>` relocates one ticker's file; the bare `<ENV>` names a
    single file and so can only answer for the default ticker. Pinned against
    the real `model_path` functions by `tests/test_model_catalogue.py`.
    """
    symbol = (ticker or apple_models.DEFAULT_TICKER).upper()
    override = os.environ.get(f"{env_prefix}_{symbol}")
    if not override and symbol == apple_models.DEFAULT_TICKER:
        override = os.environ.get(env_prefix)
    return Path(override or MODEL_DIR / pattern.format(ticker=symbol))


def _read_json(path: Path) -> dict:
    """A sidecar's contents, or `{}` when it is missing or malformed.

    Missing metadata makes a model *less described*, never unavailable: the
    estimator is in the bundle beside it and the trader would still run.
    """
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def _read_gzip_json(path: Path) -> dict:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def _installed(*modules: str) -> "list[str]":
    """Which of the named importable modules are missing (no import happens)."""
    missing = []
    for name in modules:
        try:
            if find_spec(name) is None:
                missing.append(name)
        except (ImportError, ValueError):
            missing.append(name)
    return missing


def _availability(files: "tuple[ModelFile, ...]", *modules: str) -> "tuple[bool, str]":
    """Whether a spec's files and dependencies are all present, and what is not.

    Deliberately not a load: `find_spec` and `exists` cost nothing, and the
    point of the tab is to be openable without pulling torch into the process.
    """
    missing_files = [f for f in files if not f.exists]
    missing_deps = _installed(*modules)
    if not missing_files and not missing_deps:
        return True, ""
    parts = []
    if missing_files:
        parts.append(
            "missing " + ", ".join(f"{f.role} ({f.path.name})" for f in missing_files)
        )
    if missing_deps:
        parts.append("not installed: " + ", ".join(missing_deps))
    return False, "; ".join(parts)


def format_metric(value: "object | None") -> str:
    """One metric as a string, with the precision its magnitude deserves.

    The numbers on this page span six orders of magnitude -- an AUC of 0.84, a
    day-range MAE of 0.0077 log units, an EMD of 76.7 bps, a Wilcoxon p of 2e-05
    -- so a single format string makes at least one of them unreadable.
    """
    if value is None:
        return "—"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, str):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != number:  # NaN: a metric the training run did not compute
        return "—"
    if number == int(number) and abs(number) < 1e6:
        return f"{int(number):,}"
    magnitude = abs(number)
    if magnitude >= 10:
        return f"{number:,.2f}"
    if magnitude >= 0.01:
        return f"{number:.3f}"
    if magnitude >= 1e-4:
        return f"{number:.5f}"
    return f"{number:.2e}"


def dayrange_error_pct_adr(
    meta: dict, sessions: "dict | None" = None, ticker: str = ""
) -> "float | None":
    """A day-range bundle's held-out MAE as a percentage of a typical day's range.

    The sidecar grades the model in log units, which is a *fraction of the
    day's reference price*; dividing that by the average daily range expressed
    as a fraction of the same price gives how much of a normal day the model
    typically misses by. Both sides are dimensionless, so the price cancels and
    the answer means the same thing on a $300 stock and a $25 one -- and it is
    the unit the rules downstream are already written in, where the buy and
    sell levels and the stop are all multiples of ADR (`apple_trader`).

    The denominator is the ADR the saved file carries: `adr14_abs` on the
    bundle's own simulation day, over that day's reference price. One day's
    reading standing in for the test window's, so this is an estimate to a few
    points rather than a measurement -- `simlab.drift` computes the same ratio
    per session against each session's own ADR, which is the exact version.

    A refit sidecar can lack that simulation day (INTC's does). Then the ADR
    is taken the way `highlow_error_pct_adr` takes it -- from the SIP rollups
    in `data/highlow`, averaged over the sidecar's `test_window` -- or, with no
    rollups covering the window, there is no percentage.
    """
    test = meta.get("test_metrics_ensemble") or {}
    sim = meta.get("sim_date_forecast") or {}
    try:
        mae = float(test["mae_mean"])
    except (KeyError, TypeError, ValueError):
        return None
    if mae != mae:
        return None
    if sim:
        try:
            adr_rel = float(sim["adr14_abs"]) / float(sim["prev_avg"])
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            return None
    else:
        match = re.match(
            r"\s*(\d{4}-\d{2}-\d{2})\s*\.\.\s*(\d{4}-\d{2}-\d{2})",
            str(meta.get("test_window") or ""),
        )
        if match is None:
            return None
        if sessions is None:
            sessions = _highlow_sessions(ticker or str(meta.get("ticker") or ""))
        adr_rel, _ = _window_adr(sessions, match.group(1), match.group(2))
    if adr_rel != adr_rel or adr_rel <= 0:
        return None
    return 100.0 * mae / adr_rel


# Where `highlow_model` caches its per-session SIP rollups. Mirrored rather
# than imported for the same reason as `_saved_path`: that module imports
# torch. Pinned against `highlow_model.history_path` by the tests.
HIGHLOW_HISTORY_DIR = Path(__file__).resolve().parent.parent / "data" / "highlow"


def _highlow_sessions(ticker: str) -> dict:
    """The cached `{date: {high, low, ...}}` rollups for one ticker, or `{}`."""
    path = HIGHLOW_HISTORY_DIR / f"{ticker.upper()}_sip_sessions.json"
    return (_read_json(path).get("sessions") or {}) if path.exists() else {}


def _window_adr(sessions: dict, start: str, end: str) -> "tuple[float, int]":
    """The mean `adr14` -- mean `log(high / low)` of the previous 14 sessions --
    over the cached sessions dated `start..end`, and how many that is.
    (0.0, 0) when the cache holds none."""
    ranges: "list[tuple[str, float]]" = []
    for day in sorted(sessions):
        try:
            high, low = float(sessions[day]["high"]), float(sessions[day]["low"])
            ranges.append((day, math.log(high / low)))
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            continue
    adrs = [
        sum(r for _, r in ranges[i - 14:i]) / 14
        for i in range(14, len(ranges))
        if start <= ranges[i][0] <= end
    ]
    return (sum(adrs) / len(adrs) if adrs else 0.0), len(adrs)


def highlow_error_pct_adr(
    meta: dict, sessions: "dict | None" = None, ticker: str = ""
) -> "tuple[float, int] | None":
    """A HighLow bundle's test-window MAE as a percentage of its ADR, and how
    many sessions that ADR was averaged over -- or None.

    The sidecar grades the model in log units, as TimeToChange3's does, but
    unlike TimeToChange3's it records no ADR at all, so the denominator comes
    from the SIP rollups the model itself reads (`data/highlow`). It is the
    model's own `adr14` -- the mean `log(high / low)` of the previous 14
    sessions -- averaged over the test-window sessions the cache holds. Also
    a fraction of the price, so the price cancels as in
    `dayrange_error_pct_adr`.

    An estimate: the sidecar has only the window's mean error, so this is
    mean error over mean ADR rather than the mean of the per-session ratios,
    and the cache may not reach back to the start of the window (it is
    fetched for trading, not for grading). None when there is nothing in the
    window to measure -- a percentage of an assumed range would be worse than
    the raw error.
    """
    test = meta.get("test_metrics") or {}
    window = (meta.get("splits") or {}).get("test") or []
    try:
        mae = float(test["mae_mean"])
        start, end = str(window[0]), str(window[-1])
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    if sessions is None:
        sessions = _highlow_sessions(ticker or str(meta.get("ticker") or ""))
    adr, n_sessions = _window_adr(sessions, start, end)
    if mae != mae or not adr > 0:
        return None
    return 100.0 * mae / adr, n_sessions


# --- one builder per model --------------------------------------------------

def _dayrange_spec(ticker: str) -> ModelSpec:
    path = _saved_path(
        "APPLE_DAYRANGE_MODEL", "timetochange3_dayrange_{ticker}.joblib", ticker
    )
    files = (
        ModelFile("bundle", path),
        ModelFile("N-BEATS weights", path.with_name(f"{path.stem}_nbeats.pt")),
        ModelFile("N-HiTS weights", path.with_name(f"{path.stem}_nhits.pt")),
        ModelFile("metadata", path.with_suffix(".json")),
    )
    available, reason = _availability(files, "torch", "lightgbm", "sklearn", "joblib")
    meta = _read_json(path.with_suffix(".json"))
    test = meta.get("test_metrics_ensemble") or {}
    metrics = {k: v for k, v in test.items()}
    if meta.get("opening_correction_loo_gain") is not None:
        metrics["opening-stage LOO gain"] = meta["opening_correction_loo_gain"]
    if meta.get("opening_stage_loo_mae") is not None:
        metrics["opening-stage LOO MAE"] = meta["opening_stage_loo_mae"]
    error_pct = dayrange_error_pct_adr(meta, ticker=ticker)
    if error_pct is not None:
        # First, not last: it is the row's headline, and the log-unit MAE it is
        # derived from is directly above it in `test_metrics_ensemble`.
        metrics = {"MAE (% of ADR)": error_pct, **metrics}
    features = tuple(
        list(meta.get("daily_features") or [])
        + list(meta.get("opening_features") or [])
        + list(meta.get("sequence_channels") or [])
    )
    return ModelSpec(
        key=apple_models.DAYRANGE_KEY,
        label=apple_models.get(apple_models.DAYRANGE_KEY).label,
        summary=apple_models.get(apple_models.DAYRANGE_KEY).summary,
        ticker=ticker,
        ticker_note="",
        project=codenames.DAYRANGE,
        predicts=(
            "Where the **whole session's** high and low will land, called once at "
            "9:35 from the daily history plus the first five 1-minute bars, then "
            "never updated. What it predicts well is the *width* of the day; "
            "notebook 3 showed the direction is not predictable here at all."
        ),
        target=(
            meta.get("target_definition")
            or "log(extreme_today / ((prev_open + prev_close) / 2))"
        )
        + "  →  y_high, y_low",
        algorithm=(
            "Equal-weight blend of "
            + ", ".join(meta.get("daily_models") or ["lgbm", "nbeats", "nhits"])
            + ", + an opening ridge on the residuals, clipped to contain the "
            "observed 5-minute range"
        ),
        family="LightGBM + N-BEATS + N-HiTS blend",
        consumers=("Apple Trader — day-range strategy", f"Chart overlay — {codenames.DAYRANGE}"),
        features=features,
        inputs=(
            f"~252 sessions of unadjusted daily bars (a {meta.get('lookback', 32)}-day "
            f"× {len(meta.get('sequence_channels') or []) or 8}-channel sequence inside "
            f"it) + the first {meta.get('opening_minutes', 5)} minutes of today"
        ),
        metrics=metrics,
        headline=(
            ("MAE (% of ADR)", f"{error_pct:.1f}%")
            if error_pct is not None
            # No simulation day in the sidecar means no ADR to divide by, and a
            # percentage of an assumed range would be worse than the raw error.
            else ("MAE (log units)", format_metric(test.get("mae_mean")))
        ),
        files=files,
        trained_at=str(meta.get("created") or ""),
        data_note=(
            f"Daily stage fitted through {meta.get('daily_fit_through', '?')} · "
            f"opening ridge on {meta.get('opening_fit_sessions', '?')} sessions · "
            f"held out {meta.get('held_out', '?')} · best single model on validation: "
            f"{meta.get('best_on_validation', '?')}"
            if meta
            else ""
        ),
        versions={},
        threshold=None,
        requires=apple_models.get(apple_models.DAYRANGE_KEY).requires,
        available=available,
        unavailable_reason=reason,
        caveat=(
            f"MAE {format_metric(test.get('mae_mean'))} log units"
            + (
                f" (~${float(test['mae_usd_mean']):,.2f})"
                if test.get("mae_usd_mean") is not None
                else ""
            )
            + " on the held-out test window"
            + (
                f" — about {error_pct:.0f}% of a typical day's range"
                if error_pct is not None
                else ""
            )
            + (
                f", and {float(test['skill vs rolling 14d']):.0%} of a 14-day rolling "
                "baseline's error removed"
                if test.get("skill vs rolling 14d") is not None
                else ""
            )
            + ". "
            + (
                "The range that percentage is measured against is the 14-day average "
                "on the bundle's own simulation day — the only one the file records — "
                "so read it to the nearest few points, and see SimLab's Drift tab for "
                "the same ratio against each session's own ADR. "
                if error_pct is not None
                else ""
            )
            + "It is a statement about the "
            "day's width, not its direction, so the rules on it are a mean-reversion "
            "bet. Missing PyTorch makes the bundle unavailable rather than degrading "
            "to LightGBM alone: two of three voters gone is a different predictor, not "
            "a smaller install."
        ),
    )


def _open_profile_spec() -> ModelSpec:
    path = Path(
        os.environ.get("OPEN_PROFILE_MODEL") or MODEL_DIR / "open_profile_lgbm.json.gz"
    )
    files = (ModelFile("pack", path),)
    available, reason = _availability(files, "lightgbm")
    pack = _read_gzip_json(path) if path.exists() else {}
    meta = pack.get("metadata") or {}
    levels = pack.get("p_levels") or []
    walk_forward = meta.get("walk_forward_emd_bps") or {}
    metrics: "dict[str, object]" = {
        f"walk-forward EMD (bps) · {name}": value for name, value in walk_forward.items()
    }
    if meta.get("wilcoxon_p_live_vs_atr_climatology") is not None:
        metrics["Wilcoxon p vs ATR climatology"] = meta[
            "wilcoxon_p_live_vs_atr_climatology"
        ]
    if meta.get("n_rows") is not None:
        metrics["training rows"] = meta["n_rows"]
    return ModelSpec(
        key=OPEN_PROFILE_KEY,
        label=codenames.OPEN_PROFILE,
        summary=(
            "A density model rather than a point forecast: one LightGBM booster per "
            "quantile of the day's volume-weighted price profile, fitted across a "
            "small universe deliberately without ticker dummies so it transfers. The "
            "only model here that is not tied to one symbol, and the only one whose "
            "consumer is a chart rather than a trader."
        ),
        # The only model here fitted across a universe rather than on one
        # symbol -- deliberately without ticker dummies, so it transfers.
        ticker="any",
        ticker_note=(
            "fitted on " + ", ".join(meta.get("universe") or [])
            if meta.get("universe")
            else "fitted to transfer across symbols"
        ),
        project=f"{codenames.OPEN_PROFILE} (notebook 11 workflow)",
        predicts=(
            "**Where today's volume will trade** — one LightGBM booster per "
            f"volume-quantile of the day's price profile ({len(levels)} of them: "
            f"{', '.join(f'p{p}' for p in levels)}). The predicted quantile function "
            "is turned into a smooth density with a monotone cubic (PCHIP) CDF."
        ),
        target=str(pack.get("target") or "session volume quantiles, bps vs the 9:30 open"),
        algorithm=str(pack.get("model") or "LightGBM, per-quantile L1 (EMD)"),
        family=f"LightGBM quantile boosters (×{len(levels) or 11})",
        consumers=("Live chart — price distribution curve", f"Chart overlay — {codenames.OPEN_PROFILE}"),
        features=tuple(pack.get("features") or ()),
        inputs=(
            f"≥{21} completed daily bars (the 20-day rolling features) plus today's "
            "9:30 opening print"
        ),
        metrics=metrics,
        headline=(
            "walk-forward EMD (bps)",
            format_metric(walk_forward.get("lgbm live-features")),
        ),
        files=files,
        trained_at=str(meta.get("trained_at") or ""),
        data_note=(
            f"{meta.get('n_rows', '?')} sessions over "
            f"{' → '.join(meta.get('date_range') or ['?', '?'])}, across "
            f"{', '.join(meta.get('universe') or [])}"
            if meta
            else ""
        ),
        versions=(
            {"lightgbm": str(meta["lightgbm_version"])}
            if meta.get("lightgbm_version")
            else {}
        ),
        threshold=None,
        requires="LightGBM",
        available=available,
        unavailable_reason=reason,
        caveat=(
            "76.7 bps walk-forward EMD against 77.8 for an ATR climatology and 84.1 "
            "for a point mass at the open — a small edge over climatology (Wilcoxon "
            "p ≈ 2e-05), not a large one. The full-feature variant scores 75.4; this "
            "pack uses the subset available live."
        ),
    )


def _intraday_vol_spec(ticker: str) -> ModelSpec:
    """IntradayVolatility's export: a time-of-day curve plus a day-range HAR.

    Read straight from the model module's path -- `intraday_vol_model` is pure
    JSON and numpy, so unlike the day-range bundle there is no torch import to
    mirror around.
    """
    path = intraday_vol_model.model_path(ticker)
    files = (ModelFile("model", path),)
    available, reason = _availability(files)
    raw = _read_json(path) if path.exists() else {}
    shape = raw.get("shape") or {}
    day_range = raw.get("day_range") or {}
    params = shape.get("params") or {}
    pooled = day_range.get("walk_forward_pooled") or {}

    metrics: "dict[str, object]" = {}
    if shape.get("profile_r2") is not None:
        metrics["time-of-day profile R²"] = shape["profile_r2"]
    for name, field_ in (
        ("day range walk-forward R² (log)", "r2"),
        ("22-day mean walk-forward R²", "r2_mean22"),
        ("skill vs 22-day mean (MSE)", "skill_vs_mean22"),
    ):
        if pooled.get(field_) is not None:
            metrics[name] = pooled[field_]
    if day_range.get("r2_in_sample") is not None:
        metrics["day range in-sample R²"] = day_range["r2_in_sample"]
    for year, row in (day_range.get("walk_forward") or {}).items():
        metrics[f"walk-forward R² · {year}"] = (row or {}).get("r2")

    def _sample(block: dict) -> str:
        return " → ".join(block.get("sample") or ["?", "?"])

    return ModelSpec(
        key=INTRADAY_VOL_KEY,
        label=codenames.INTRADAY_VOL,
        summary=(
            "How volatile each minute of the session usually is — a five-parameter "
            "power-law curve, widest at the open, flat through midday, with a short "
            "ramp into the close — plus a daily-bar forecast of how wide the whole day "
            "will be. Drawn on the charts as a time-of-day price envelope, on its own "
            f"or stretched to {codenames.DAYRANGE}'s high and low."
        ),
        ticker=ticker,
        ticker_note="",
        project=f"{codenames.INTRADAY_VOL}, exported by `scripts/export_app_model.py`",
        predicts=(
            "**The shape of volatility through the session** — relative volatility at "
            "each minute, fitted to the 5-minute diurnal variance factor — and **the "
            "day's log high-low range**, from yesterday's, the week's and the month's "
            "ranges plus today's gap. Everything is known at the open and fixed for "
            "the day."
        ),
        target="√s(t) = a + b(1+t)^−α + c·e^−(390−t)/κ  ·  log ln(H/L)",
        algorithm=(
            f"power decay + close ramp (α {params.get('alpha', float('nan')):.2f}, "
            f"κ {params.get('kappa', float('nan')):.1f} min) by least squares; "
            "log-HAR on daily ranges by OLS"
        ),
        family="Power-law profile + HAR",
        consumers=(
            f"Chart overlay — {codenames.INTRADAY_VOL}",
            f"Chart overlay — {codenames.DAYRANGE_INTRADAY}",
        ),
        features=tuple((day_range.get("features") or {}).keys()),
        inputs="22 completed daily bars plus today's opening print",
        metrics=metrics,
        headline=("day range walk-forward R²", format_metric(pooled.get("r2"))),
        files=files,
        trained_at=str(raw.get("created") or ""),
        data_note=(
            f"Profile fitted on {shape.get('sessions', '?')} sessions, {_sample(shape)}; "
            f"day-range HAR on {day_range.get('sessions', '?')} sessions, "
            f"{_sample(day_range)}, walk-forward over "
            f"{', '.join(day_range.get('walk_forward') or {}) or '?'}"
            if raw
            else ""
        ),
        versions={},
        threshold=None,
        requires="nothing beyond numpy",
        available=available,
        unavailable_reason=reason,
        caveat=(
            f"The time-of-day shape is the strong half (profile R² "
            f"{format_metric(shape.get('profile_r2'))}, and notebook 03 found it stable "
            "across years). The day-range forecast is the weak half: refitted on daily "
            "bars because the app keeps no minute history, it explains "
            f"{format_metric(pooled.get('r2'))} of a test year's variance in log range "
            f"and removes {format_metric(pooled.get('skill_vs_mean22'))} of a 22-day "
            f"mean's error — so for the band's width, prefer {codenames.DAYRANGE}'s high "
            "and low. Nothing here updates during the session."
        ),
    )


# The network checkpoints a HighLow bundle may ship beside its joblib.
_HIGHLOW_NETS = {"nbeats": "N-BEATS", "nhits": "N-HiTS"}


def _highlow_spec(ticker: str) -> ModelSpec:
    path = _saved_path("APPLE_HIGHLOW_MODEL", "highlow15m_{ticker}.joblib", ticker)
    meta = _read_json(path.with_suffix(".json"))
    # Only the networks the blend weights have to be on disk (INTC ships
    # N-HiTS alone); a sidecar without weights predates the choice: N-BEATS.
    shipped = [k for k, w in (meta.get("weights") or {"nbeats": 1}).items() if w and k in _HIGHLOW_NETS]
    files = (
        ModelFile("bundle", path),
        *(ModelFile(f"{_HIGHLOW_NETS[k]} weights", path.with_name(f"{path.stem}_{k}.pt")) for k in shipped),
        ModelFile("metadata", path.with_suffix(".json")),
    )
    available, reason = _availability(files, "torch", "lightgbm", "sklearn", "joblib")
    test = meta.get("test_metrics") or {}
    metrics = {k: v for k, v in test.items()}
    if meta.get("walk_forward_mae_mean") is not None:
        metrics["walk-forward MAE"] = meta["walk_forward_mae_mean"]
    # Sidecars from 2026-10-03 on (AVGO, NVDA) carry the notebook's own score in
    # ADR units, per session; older ones only the log MAE, put over the cached ADR.
    exact = test.get("mae_adr_mean")
    measured = None if exact is not None else highlow_error_pct_adr(meta, ticker=ticker)
    error_pct = float(exact) if exact is not None else measured[0] if measured else None
    if error_pct is not None:
        # First, as on TimeToChange3's row: it is the headline.
        metrics = {"MAE (% of ADR)": error_pct, **metrics}
    weights = {k: w for k, w in (meta.get("weights") or {}).items() if w}
    data = meta.get("data") or {}
    model = apple_models.get(apple_models.HIGHLOW_KEY)
    return ModelSpec(
        key=apple_models.HIGHLOW_KEY,
        label=model.label,
        summary=model.summary,
        ticker=ticker,
        ticker_note="",
        project=codenames.HIGHLOW,
        predicts=(
            "Where the **whole session's** high and low will land, called once at "
            "9:35 and measured from the 9:35 price in 14-day average ranges. Like "
            f"{codenames.DAYRANGE} it forecasts the *width* of the day well and its centre "
            "only roughly."
        ),
        target="up = log(high / close5) / adr14, down = log(close5 / low) / adr14",
        algorithm=(
            "Weighted blend "
            + " + ".join(f"{w:g} {k}" for k, w in weights.items())
            + ", L1 loss, clipped to contain the observed 5-minute range"
        ),
        family="LightGBM + N-BEATS blend",
        consumers=("Apple Trader — day-range strategy", f"Chart overlay — {codenames.HIGHLOW}"),
        features=tuple(list(meta.get("features") or []) + list(meta.get("sequence_channels") or [])),
        inputs=(
            f"~127 sessions of SIP minute bars rolled up to daily (a "
            f"{meta.get('lookback', 32)}-day sequence inside it) + the first "
            f"{meta.get('opening_minutes', 5)} minutes of today"
            + (
                f", read from {str(meta['opening_feed']).upper()}"
                if str(meta.get("opening_feed") or "sip").lower() != "sip" else ""
            )
            + (
                # BE: the theme group, whose lead-peer column its nets read
                f"; the same first minutes of its theme peers {', '.join(meta['peers'])} "
                "(from IEX), with their SIP history"
                if meta.get("peers") and "lead_or_ret_adr" in (meta.get("custom_features_kept") or ())
                else ""
            )
            + (
                # NVDA: the market, pre-market and after-hours groups
                f"; {meta.get('market_proxy') or 'SPY'}'s first minutes and SIP history, this "
                "morning's SIP pre-market to 09:19 and the previous evening's after-hours"
                if {"mkt_or_ret_adr", "pm_gap_extend", "ah_ret_adr"}
                & set(meta.get("custom_features_kept") or ())
                else ""
            )
        ),
        metrics=metrics,
        headline=(
            ("MAE (% of ADR)", f"{error_pct:.1f}%")
            if error_pct is not None
            # No SIP history cached yet (it is fetched on the first forecast):
            # the dollar error, then the raw one, rather than an assumed range.
            else ("MAE ($ per extreme)", f"${float(test['mae_usd_mean']):,.2f}")
            if test.get("mae_usd_mean") is not None
            else ("MAE (log units)", format_metric(test.get("mae_mean")))
        ),
        files=files,
        trained_at=str(meta.get("created") or ""),
        data_note=(
            f"Fitted through {data.get('fit_through', '?')} on {data.get('sessions', '?')} "
            f"sessions · held out {' – '.join(meta.get('held_out_week') or []) or '?'}"
            if meta
            else ""
        ),
        versions={},
        threshold=None,
        requires=model.requires,
        available=available,
        unavailable_reason=reason,
        caveat=(
            (
                f"Misses each extreme by about {error_pct:.0f}% of a typical day's "
                f"range — the ADR averaged over the {measured[1]} test-window sessions "
                "in the local SIP cache, so an estimate. "
                if measured
                else f"Misses each extreme by {error_pct:.0f}% of that day's 14-day range "
                "on average (the notebook's own score). "
                if error_pct is not None
                else ""
            )
            + f"MAE {format_metric(test.get('mae_mean'))} log units on the "
            f"{', '.join((meta.get('splits') or {}).get('test') or []) or 'test'} window"
            + (
                f", against {codenames.DAYRANGE}'s published "
                f"{format_metric(meta.get('ttc3_published_test_mae_mean'))}"
                if meta.get("ttc3_published_test_mae_mean") is not None
                else ""
            )
            + f". The shipped trading distances were swept on {codenames.DAYRANGE}'s forecast, "
            "not this one."
        ),
    )


def _highlow2_spec(ticker: str) -> ModelSpec:
    path = _saved_path("APPLE_HIGHLOW2_MODEL", "highlow2_5m_{ticker}.joblib", ticker)
    files = (
        ModelFile("bundle", path),
        ModelFile("metadata", path.with_suffix(".json")),
    )
    available, reason = _availability(files, "lightgbm", "joblib")
    meta = _read_json(path.with_suffix(".json"))
    # The notebook's sidecar carries no score; the copy in Code/Models has the
    # notebook's own `models.score` of its test window added (`test_metrics_note`).
    test = meta.get("test_metrics") or {}
    metrics = {k: v for k, v in test.items()}
    error_pct = test.get("mae_adr_mean")
    if error_pct is not None:
        # First, as on the other day-range rows: it is the headline. Exact here
        # rather than estimated -- the notebook divides each session's dollar
        # miss by that session's own 14-day range.
        metrics = {"MAE (% of ADR)": error_pct, **metrics}
    for window, change in (meta.get("walk_forward_vs_highlow5m") or {}).items():
        metrics[f"walk-forward vs {codenames.HIGHLOW} · {window}"] = change
    # A bundle with a rest-of-session head (INTC's) hands the trader that head's
    # forecast, so its `test_metrics` are the rest head's, against the extremes
    # after 9:35; the day head's sit beside them.
    rest = bool(meta.get("rest_head"))
    day_pct = (meta.get("test_metrics_day") or {}).get("mae_adr_mean")
    for k, v in (meta.get("test_metrics_day") or {}).items():
        metrics[f"day head · {k}"] = v
    for window, change in ((meta.get("rest_head") or {}).get("walk_forward_vs_day_head_as_rest") or {}).items():
        metrics[f"rest head vs day head read as the rest · {window}"] = change
    weights = {k: w for k, w in (meta.get("weights") or {}).items() if w}
    data = meta.get("data") or {}
    rows = meta.get("training_rows") or {}
    test_window = " – ".join(meta.get("test_window") or []) or "?"
    model = apple_models.get(apple_models.HIGHLOW2_KEY)
    return ModelSpec(
        key=apple_models.HIGHLOW2_KEY,
        label=model.label,
        summary=model.summary,
        ticker=ticker,
        ticker_note=(
            f"Trained on {ticker} with {', '.join(meta['training_pool'])} pooled in, each in its "
            "own 14-day-range units"
            if meta.get("training_pool") else ""
        ),
        project=codenames.HIGHLOW2,
        predicts=(
            (
                "Where the high and low **after 9:35** will land — from the 9:35 bar to the "
                "close, what a 9:35 order can still reach — and, from a second head, the "
                "whole session's. Apple Trader's levels hang off the first pair. "
                if rest else "Where the **whole session's** high and low will land. "
            )
            + "Called once at 9:35 from IEX's first five minutes, the pre-market and the "
            "SIP daily history, measured from the 9:35 price in 14-day average ranges. "
            "Fitted with shock days left out, so it forecasts an ordinary day's width — on "
            "the day after earnings it has nothing to say about the news."
        ),
        target=(
            "rest_up = log(rest_high / close5) / adr14, rest_down = log(close5 / rest_low) / adr14 "
            "(rest = 9:35 bar to the close); day head: up = log(high / close5) / adr14, "
            "down = log(close5 / low) / adr14"
            if rest else "up = log(high / close5) / adr14, down = log(close5 / low) / adr14"
        ),
        algorithm=(
            f"LightGBM, L1 loss, averaged over {len(weights) or '?'} seeds "
            f"({', '.join(str(s) for s in meta.get('seeds') or []) or '?'}), "
            + (
                "one pair per head; the rest head kept on the right side of the 9:35 price "
                "and inside the day forecast, the day head clipped to contain the observed "
                "5-minute range"
                if rest else "clipped to contain the observed 5-minute range"
            )
        ),
        family="LightGBM (3 seeds)",
        consumers=(
            "Apple Trader — day-range strategy",
            f"Chart overlay — {codenames.HIGHLOW2}",
        ),
        features=tuple(meta.get("features") or []),
        inputs=(
            "~127 sessions of SIP minute bars rolled up to daily, the IEX openings of the "
            "last 28, and the pre-market of the last 20; today's first 5 minutes from IEX, "
            "SIP's pre-market to 09:19, IEX's 09:20–09:29 and last evening's after-hours"
        ),
        metrics=metrics,
        headline=(
            ("MAE (% of ADR)", f"{float(error_pct):.1f}%")
            if error_pct is not None
            else ("MAE (log units)", format_metric(test.get("mae_mean")))
        ),
        files=files,
        trained_at=str(meta.get("created") or ""),
        data_note=(
            f"Fitted through {data.get('fit_through', '?')} on {sum(rows.values()) or '?'} sessions "
            f"({rows.get(ticker, '?')} {ticker}) from {data.get('first_row', '?')}, shock days "
            f"out · tested {test_window} · traded {' – '.join(meta.get('traded_week') or []) or '?'}"
            if meta
            else ""
        ),
        versions={},
        threshold=None,
        requires=model.requires,
        available=available,
        unavailable_reason=reason,
        caveat=(
            (
                f"Misses each extreme by about {float(error_pct):.0f}% of a typical day's range "
                f"on its {test.get('n', '?')}-session test window ({test_window}). "
                if error_pct is not None
                else ""
            )
            + (
                "That is the rest of the session's forecast against the extremes after 9:35, "
                "a harder question than the day's (the opening's extremes are not given to "
                "it), so it does not compare with the day-range rows"
                + (
                    f"; the day head misses by {float(day_pct):.0f}%. "
                    if day_pct is not None else ". "
                )
                + "The notebook found the rest head no better than reading the day forecast as "
                "the rest, and ships it because a rest-of-session forecast was asked for. "
                if rest else ""
            )
            + f"Level with {codenames.HIGHLOW} over the notebook's walk-forward and slightly ahead — "
            "its README calls it parity with a small edge, not a clear win. The shipped "
            "trading distances were never swept on this forecast."
        ),
    )


# HighLow_3m's sidecar scores four windows (`scores`); what each is called here.
_HIGHLOW3M_WINDOWS = (
    ("holdout_2026", "2026 holdout"),
    ("test", "test window"),
    ("baseline_test", "14-day baseline, test window"),
    ("selection_2025", "2025 selection"),
)
_HIGHLOW3M_SCORES = ("mae_pct_adr", "mae_usd", "mae", "range_corr", "high_inside", "low_inside", "n")


def _highlow3m_spec(ticker: str) -> ModelSpec:
    path = _saved_path("APPLE_HIGHLOW3M_MODEL", "highlow3m_{ticker}.joblib", ticker)
    files = (
        ModelFile("bundle", path),
        ModelFile("metadata", path.with_suffix(".json")),
    )
    available, reason = _availability(files, "lightgbm", "scipy", "joblib")
    meta = _read_json(path.with_suffix(".json"))
    scores = meta.get("scores") or {}
    # The headline is the 2026 holdout, not the 9-session test window: walk-forward
    # months refitted on everything before them, every one out of sample -- the
    # notebook README's "honest number". The test window is the shipped model's own
    # score, on nine sessions.
    holdout = scores.get("holdout_2026") or {}
    test = scores.get("test") or {}
    error_pct = holdout.get("mae_pct_adr")
    metrics: "dict[str, object]" = {}
    if error_pct is not None:
        metrics["MAE (% of ADR)"] = error_pct
    for window, label in _HIGHLOW3M_WINDOWS:
        for name in _HIGHLOW3M_SCORES:
            value = (scores.get(window) or {}).get(name)
            if value is not None:
                metrics[f"{label} · {name}"] = value
    weights = {k: w for k, w in (meta.get("weights") or {}).items() if w}
    data = meta.get("data") or {}
    recipe = meta.get("recipe") or {}
    windows = meta.get("windows") or {}
    test_window = " – ".join(windows.get("test") or []) or "?"
    model = apple_models.get(apple_models.HIGHLOW3M_KEY)
    blend = " + ".join(f"{k} {w:g}" for k, w in weights.items()) or "?"
    return ModelSpec(
        key=apple_models.HIGHLOW3M_KEY,
        label=model.label,
        summary=model.summary,
        ticker=ticker,
        ticker_note=(
            f"Trained on {ticker} with {', '.join(data['pool'])} pooled in, each in its "
            "own 14-day-range units"
            if data.get("pool") else ""
        ),
        project=codenames.HIGHLOW3M,
        predicts=(
            "How high and how low the price trades **from 9:33 to the close** — not the whole "
            "session's extremes, which print in the first three minutes on about half of "
            "AAPL's sessions — called once at 9:33 from IEX's first three minutes, the SIP "
            "daily history and last night's option positioning, measured from the 9:33 "
            "price (IEX's 9:32 close) in 14-day average ranges. Not clipped to the opening "
            "range."
        ),
        target="up = log(rest_high / close3) / adr14, down = log(close3 / rest_low) / adr14",
        algorithm=(
            f"Blend {blend}: LightGBM, L1 loss, {recipe.get('lgbm_params', {}).get('num_leaves', '?')} "
            f"leaves, averaged over seeds {', '.join(str(x) for x in recipe.get('seeds') or []) or '?'}; "
            f"linear median (quantile) regression on standardised features, alpha "
            f"{recipe.get('linear_alpha', '?')}"
        ),
        family="LightGBM (3 seeds) + linear median",
        consumers=(
            "Apple Trader — day-range strategy",
            f"Chart overlay — {codenames.HIGHLOW3M}",
        ),
        features=tuple(meta.get("feature_cols") or []),
        inputs=(
            "~127 sessions of SIP minute bars rolled up to daily, the SIP extremes after 9:33 "
            "of the last 14 and the IEX openings of the last 14; today's first 3 minutes from "
            "IEX; last night's option positioning, rebuilt from the contract list and every "
            "expiration's daily bars over its last 120 days (open interest as the running sum "
            "of size-weighted volume), priced with Yahoo's 13-week T-bill yield"
        ),
        metrics=metrics,
        headline=(
            ("MAE (% of ADR)", f"{float(error_pct):.1f}%")
            if error_pct is not None
            else ("MAE (log units)", format_metric(test.get("mae")))
        ),
        files=files,
        trained_at=str(meta.get("created") or ""),
        data_note=(
            f"Fitted through {data.get('fit_through', '?')} on {data.get('training_rows', '?')} "
            f"sessions ({data.get('aapl_rows', '?')} {ticker}) from {data.get('history_start', '?')} · "
            f"tested {test_window} · traded {' – '.join(windows.get('traded_week') or []) or '?'}"
            if meta
            else ""
        ),
        versions={},
        threshold=None,
        requires=model.requires,
        available=available,
        unavailable_reason=reason,
        caveat=(
            (
                f"Misses each extreme by about {float(error_pct):.0f}% of a typical day's range "
                f"over the {holdout.get('n', '?')}-session 2026 holdout"
                + (
                    f" ({float(test['mae_pct_adr']):.0f}% on the {test.get('n', '?')}-session "
                    f"test window, {test_window})"
                    if test.get("mae_pct_adr") is not None
                    else ""
                )
                + ". "
                if error_pct is not None
                else ""
            )
            + "Not comparable with the other day-range rows: its target is the range after "
            "9:33, which no opening-range clip can help. 8.7% better than a 14-day baseline "
            "in every 2026 month — a forecast of width, not direction. The shipped trading "
            "distances were never swept on it."
        ),
    )


_BUILDERS = {
    apple_models.DAYRANGE_KEY: _dayrange_spec,
    apple_models.HIGHLOW_KEY: _highlow_spec,
    apple_models.HIGHLOW2_KEY: _highlow2_spec,
    apple_models.HIGHLOW3M_KEY: _highlow3m_spec,
    INTRADAY_VOL_KEY: _intraday_vol_spec,
}


# --- the catalogue ----------------------------------------------------------

def spec(key: str, ticker: "str | None" = None) -> "ModelSpec | None":
    """One model's spec for one instrument, read from its saved file."""
    if key == OPEN_PROFILE_KEY:
        return _open_profile_spec()
    builder = _BUILDERS.get(key)
    if builder is None:
        return None
    return builder((ticker or apple_models.DEFAULT_TICKER).upper())


def specs() -> "list[ModelSpec]":
    """Every (model, instrument) pair this app can read, in registry order.

    The per-ticker models come first, one spec each, because that is the unit a
    model actually exists in -- then the one pack that was fitted to transfer.
    """
    out: "list[ModelSpec]" = []
    for key, model in apple_models.MODELS.items():
        for ticker in model.tickers:
            found = spec(key, ticker)
            if found is not None:
                out.append(found)
    for ticker in intraday_vol_model.TICKERS:
        out.append(_intraday_vol_spec(ticker))
    out.append(_open_profile_spec())
    return out


def specs_by_model() -> "dict[str, list[ModelSpec]]":
    """The same specs grouped by model key, which is how the tab lays them out."""
    grouped: "dict[str, list[ModelSpec]]" = {}
    for found in specs():
        grouped.setdefault(found.key, []).append(found)
    return grouped
