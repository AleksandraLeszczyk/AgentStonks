"""Every trained model in this app, described from its own saved file.

The five ML models here are each loaded by their own module, each for their own
consumer -- `apple_models` feeds the traders, `profile_model` feeds the price
profile, `model_overlays` feeds both charts. What none of them offer is an
answer to "what is actually installed, fitted on what, and how well does it
score?", which is the question the ML Models tab asks. This module is that
answer, and it is deliberately a *reader* rather than a sixth model: everything
below comes out of the bundles and sidecars on disk, so a retrain changes the
tab without anyone editing it.

Read the file, not the model
----------------------------
A spec is assembled without importing PyTorch or LightGBM. That is the whole
design constraint, and it is not fussiness: `apple_models` goes out of its way
to keep a 200 MB dependency off the default start-up path, and a tab that lists
model names would undo that if it had to instantiate them to do it. So:

* `persistence` and `momentum_change` are read through their own modules --
  scikit-learn and joblib are already imported by the time the UI runs.
* `nbeats` and `dayrange` are read from their **JSON sidecars**, which carry the
  metrics, the settings and (for day-range) the whole feature list. Their
  `model_path` helpers live in modules that import torch at module scope, so
  the paths are mirrored here instead -- see `_saved_path`.
* `open_profile` is a gzipped JSON pack, so its metadata is readable with the
  standard library alone even though scoring it needs LightGBM.

Availability is answered the same cheap way: `importlib.util.find_spec` for the
dependency plus `Path.exists` for the file. A model reported ready here has
still only been *found*, not loaded -- for the two that unpickle an estimator
that is nearly the same thing, and for the two torch ones it is a weaker claim.
The tab says so.

The mirror contract, again
--------------------------
`_saved_path` reproduces the `MODEL_DIR` / `<ENV>_<TICKER>` / `<ENV>` lookup
that each model module implements for itself. It is the same kind of copy
`persistence_model` keeps of `mshift` and for the same reason -- the alternative
imports torch -- and it carries the same obligation: `tests/test_model_catalogue.py`
pins every mirrored path against the real `model_path` it stands in for, so a
renamed file or a new env override fails a test rather than showing a user a
path nothing reads.
"""

from __future__ import annotations

import gzip
import json
import os
from dataclasses import dataclass, field
from importlib.util import find_spec
from pathlib import Path

from . import apple_models, momentum_change_model, persistence_model

# The shared store beside the AgentStonks checkout -- the same constant each
# model module defines for itself (`persistence_model.MODEL_DIR` and friends).
MODEL_DIR = persistence_model.MODEL_DIR

OPEN_PROFILE_KEY = "open_profile"


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

    A spec is per (model, ticker) rather than per model: TimeToChange fits and
    *selects* per symbol, so AAPL's delta-momentum regressor is a Ridge and
    INTC's is a HistGradientBoosting -- one row per model would have to pick
    one of them to be true about.
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


# --- one builder per model --------------------------------------------------

def _persistence_spec(ticker: str) -> ModelSpec:
    path = persistence_model.model_path(ticker)
    files = (ModelFile("bundle", path),)
    available, reason = _availability(files, "sklearn", "joblib")
    bundle = persistence_model.load_bundle(ticker) if available else None
    metrics = dict((bundle or {}).get("metrics") or {})
    hard = metrics.pop("hard_half", None)
    if isinstance(hard, dict):
        metrics["hard_half n"] = hard.get("n")
        metrics["hard_half ROC AUC"] = hard.get("roc_auc")
    settings = (bundle or {}).get("settings") or {}
    momentum = settings.get("momentum") or {}
    estimator = (bundle or {}).get("pipeline")
    clf = getattr(estimator, "steps", [(None, None)])[-1][1] if estimator else None
    return ModelSpec(
        key=apple_models.PERSISTENCE_KEY,
        label=apple_models.get(apple_models.PERSISTENCE_KEY).label,
        summary=apple_models.get(apple_models.PERSISTENCE_KEY).summary,
        ticker=ticker,
        ticker_note="",
        project="TimeToChange2 (notebook 04 / 08), mirrors `mshift`",
        predicts=(
            "Whether a momentum-regime change that has just printed will **hold** — "
            "the old regime ran ≥15 bars and the new one survives ≥15 bars, or price "
            "moves ≥1% its way within 10 bars."
        ),
        target="P(persistent | the 20 bars ending at the change), binary",
        algorithm=(
            f"Sequence summariser → {type(clf).__name__}"
            if clf is not None
            else "Sequence summariser → HistGradientBoostingClassifier"
        ),
        family="Gradient-boosted classifier",
        consumers=("Apple Trader — momentum strategy", "Chart overlay — momentum"),
        features=tuple((bundle or {}).get("feature_columns") or ()),
        inputs=(
            f"{(bundle or {}).get('seq_len', 20)} × 1-minute bars ending at the change "
            f"bar, session-local (momentum horizon {momentum.get('horizon', 15)}, "
            f"enter {momentum.get('enter_threshold', 0.9)} / exit "
            f"{momentum.get('exit_threshold', 0.4)} Schmitt trigger)"
        ),
        metrics=metrics,
        headline=("ROC AUC", format_metric(metrics.get("roc_auc"))),
        files=files,
        trained_at=str((bundle or {}).get("trained_at") or ""),
        data_note=(
            f"Sessions from {(bundle or {}).get('excluded_sessions_from')} excluded "
            "from fitting."
            if (bundle or {}).get("excluded_sessions_from")
            else ""
        ),
        versions=(
            {"scikit-learn": str((bundle or {}).get("sklearn_version"))}
            if (bundle or {}).get("sklearn_version")
            else {}
        ),
        threshold=(bundle or {}).get("threshold"),
        requires=apple_models.get(apple_models.PERSISTENCE_KEY).requires,
        available=available and bundle is not None,
        unavailable_reason=reason or ("bundle could not be unpickled" if available else ""),
        caveat=(
            "The headline AUC is over **all** regime changes. Over the ones that "
            "already pass the observable pre-dwell ≥ 15 pre-condition — the half that "
            "matters — it is a coin flip. Use it to reject changes that cannot hold, "
            "not to rank the ones that can."
        ),
    )


def _nbeats_spec(ticker: str) -> ModelSpec:
    path = _saved_path("APPLE_NBEATS_MODEL", "timetochange2_nbeats_{ticker}.pt", ticker)
    files = (
        ModelFile("checkpoint", path),
        ModelFile("residual sidecar", path.with_name(f"{path.stem}_residuals.npz")),
        ModelFile("metadata", path.with_suffix(".json")),
    )
    available, reason = _availability(files, "torch")
    meta = _read_json(path.with_suffix(".json"))
    all_metrics = meta.get("metrics") or {}
    persistence = dict(all_metrics.get("persistence") or {})
    forecast = dict(all_metrics.get("forecast") or {})
    hard = all_metrics.get("hard_half") or {}
    metrics = {f"persistence · {k}": v for k, v in persistence.items() if k not in ("model", "family")}
    if hard:
        metrics["hard half · n"] = hard.get("n")
        metrics["hard half · ROC AUC"] = hard.get("roc_auc")
    metrics.update({f"forecast · {k}": v for k, v in forecast.items() if k not in ("family", "seeds")})
    momentum = ((meta.get("trained_on") or {}).get("momentum")) or {}
    # The feature side is `persistence_model`'s, unchanged and by contract --
    # both models are handed the identical (20, 25) block (see nbeats_model).
    shared = persistence_model.load_bundle(ticker) or {}
    return ModelSpec(
        key=apple_models.NBEATS_KEY,
        label=apple_models.get(apple_models.NBEATS_KEY).label,
        summary=apple_models.get(apple_models.NBEATS_KEY).summary,
        ticker=ticker,
        ticker_note="",
        project="TimeToChange2 (notebooks 06–08), mirrors `mshift`",
        predicts=(
            "The same question as the classifier, by a longer route: "
            f"{meta.get('n_seeds', 5)} seeds of N-BEATS forecast the next "
            f"{momentum.get('horizon', 15)} bars of the momentum score, 500 sampled "
            "futures are replayed through the real Schmitt trigger, and the fraction "
            "that survive is the probability. Also answers the *anticipation* "
            "question — will a change happen next bar — which the classifier cannot."
        ),
        target="P(post-dwell ≥ 15) × the observable pre-dwell gate, by Monte Carlo",
        algorithm=(
            f"{meta.get('n_seeds', 5)}-seed N-BEATS ensemble (mean) + row-wise "
            "residual bootstrap → 500 paths → trigger replay"
        ),
        family=f"N-BEATS ensemble ({meta.get('n_seeds', 5)} seeds)",
        consumers=("Apple Trader — momentum strategy", "Chart overlay — momentum"),
        features=tuple(shared.get("feature_columns") or ()),
        inputs=(
            f"{((meta.get('trained_on') or {}).get('dataset') or {}).get('seq_len', 20)}"
            " × 1-minute bars — the identical block the classifier is handed "
            "(same features, same window, same momentum pipeline)"
        ),
        metrics=metrics,
        headline=(
            "hard-half ROC AUC" if hard else "ROC AUC",
            format_metric(hard.get("roc_auc") if hard else persistence.get("roc_auc")),
        ),
        files=files,
        trained_at=", ".join(meta.get("train_sessions") or []),
        data_note=(
            f"Validation {', '.join(meta.get('valid_sessions') or []) or '—'} · "
            f"test {', '.join(meta.get('test_sessions') or []) or '—'} · "
            f"excluded from {meta.get('excluded_sessions_from', '?')}"
            if meta
            else ""
        ),
        versions={"torch": str(meta["torch_version"])} if meta.get("torch_version") else {},
        threshold=persistence.get("threshold"),
        requires=apple_models.get(apple_models.NBEATS_KEY).requires,
        available=available,
        unavailable_reason=reason,
        caveat=(
            "0.67 ± 0.07 on the hard half across four walk-forward folds — a real but "
            "small effect measured on ~35 events, and the only entrant above chance on "
            "all four. The probability is a Monte-Carlo estimate (~0.015 SE at 500 "
            "paths), so a candidate sitting on the threshold can move between runs."
        ),
    )


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
        project="TimeToChange3, mirrors `dayrange`",
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
        consumers=("Apple Trader — day-range strategy", "Chart overlay — predicted day range"),
        features=features,
        inputs=(
            f"~252 sessions of unadjusted daily bars (a {meta.get('lookback', 32)}-day "
            f"× {len(meta.get('sequence_channels') or []) or 8}-channel sequence inside "
            f"it) + the first {meta.get('opening_minutes', 5)} minutes of today"
        ),
        metrics=metrics,
        headline=("MAE (log units)", format_metric(test.get("mae_mean"))),
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
            "MAE 0.0077 log units (~$2.12) over 129 test sessions against 0.0110 for a "
            "14-day rolling baseline — 30% of the baseline's error removed, and the "
            "ordering holds across all four walk-forward refits. It is a statement "
            "about the day's width, not its direction, so the rules on it are a "
            "mean-reversion bet. Missing PyTorch makes the bundle unavailable rather "
            "than degrading to LightGBM alone: two of three voters gone is a different "
            "predictor, not a smaller install."
        ),
    )


def _momentum_change_spec(ticker: str) -> ModelSpec:
    path = momentum_change_model.model_path(ticker)
    files = (
        ModelFile("bundle", path),
        ModelFile("metadata", momentum_change_model.metadata_path(path)),
    )
    available, reason = _availability(files, "sklearn", "joblib")
    meta = _read_json(momentum_change_model.metadata_path(path))
    params = meta.get("pipeline_params") or {}
    train_days = meta.get("train_days") or []
    versions = {
        name: str(meta[key])
        for name, key in (
            ("scikit-learn", "sklearn_version"),
            ("pandas", "pandas_version"),
            ("numpy", "numpy_version"),
        )
        if meta.get(key)
    }
    metrics = dict(meta.get("metrics") or {})
    return ModelSpec(
        key=apple_models.MOMENTUM_CHANGE_KEY,
        label=apple_models.get(apple_models.MOMENTUM_CHANGE_KEY).label,
        summary=apple_models.get(apple_models.MOMENTUM_CHANGE_KEY).summary,
        ticker=ticker,
        ticker_note="",
        project="TimeToChange, mirrors `momlib`",
        predicts=(
            "**How far** the smoothed momentum score will move over the next 15 bars "
            "— a signed quantity, not a probability. A positive prediction on a bar "
            "whose regime is negative is the model calling a turn upwards; a negative "
            "one on a positive bar is it calling the move over."
        ),
        target=(
            meta.get("target", "mom_delta")
            + " = mom[t+15] − mom[t−1], in "
            + str(meta.get("target_units", "bps/min")).split("(")[0].strip()
        ),
        algorithm=(
            f"{meta.get('model_name', '?')} — selected per ticker on that ticker's "
            "own validation days"
        ),
        # The family really is per ticker here: TimeToChange *selects* the
        # estimator on each symbol's own validation days, and the three
        # answers differ.
        family=f"{meta.get('model_name', 'regressor')} regressor",
        consumers=("Apple Trader — delta-momentum strategy",),
        features=tuple(meta.get("feature_cols") or ()),
        inputs=(
            "The minute that just closed, off six sessions of 1-minute history "
            f"(momentum window {params.get('window', 15)}, smoothing half-life "
            f"{params.get('smooth_halflife', 8.0)}, adaptive threshold c="
            f"{params.get('c', 0.4)}, {params.get('n_prev_changes', 3)} previous changes)"
        ),
        metrics=metrics,
        headline=("R² (holdout week)", format_metric(metrics.get("holdout_r2"))),
        files=files,
        trained_at=str(meta.get("saved_at") or ""),
        data_note=(
            f"{len(train_days)} training days "
            f"({train_days[0]} … {train_days[-1]}); the last 5 sessions are reserved "
            "as a holdout week, used for neither fitting nor selection. `r2_test` is "
            "the validation days, `holdout_*` the reserved week."
            if train_days
            else str(meta.get("notes") or "")
        ),
        versions=versions,
        threshold=None,
        requires=apple_models.get(apple_models.MOMENTUM_CHANGE_KEY).requires,
        available=available,
        unavailable_reason=reason,
        caveat=(
            "The **sign** is the reliable half. Over sampled stable minutes the "
            "absolute prediction separates 'near a change' from 'quiet' at AUC ~0.68, "
            "but bar by bar over an unseen day that falls to ~0.53 — so the rules gate "
            "on a regime the tape has already printed and use the model for direction "
            "only, never as a change detector."
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
        label="Open price profile (LevelsML)",
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
        project="LevelsML (notebook 11 workflow), mirrors `levelsml/features.py`",
        predicts=(
            "**Where today's volume will trade** — one LightGBM booster per "
            f"volume-quantile of the day's price profile ({len(levels)} of them: "
            f"{', '.join(f'p{p}' for p in levels)}). The predicted quantile function "
            "is turned into a smooth density with a monotone cubic (PCHIP) CDF."
        ),
        target=str(pack.get("target") or "session volume quantiles, bps vs the 9:30 open"),
        algorithm=str(pack.get("model") or "LightGBM, per-quantile L1 (EMD)"),
        family=f"LightGBM quantile boosters (×{len(levels) or 11})",
        consumers=("Live chart — price distribution curve", "Chart overlay — profile range"),
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


_BUILDERS = {
    apple_models.PERSISTENCE_KEY: _persistence_spec,
    apple_models.NBEATS_KEY: _nbeats_spec,
    apple_models.DAYRANGE_KEY: _dayrange_spec,
    apple_models.MOMENTUM_CHANGE_KEY: _momentum_change_spec,
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
    out.append(_open_profile_spec())
    return out


def specs_by_model() -> "dict[str, list[ModelSpec]]":
    """The same specs grouped by model key, which is how the tab lays them out."""
    grouped: "dict[str, list[ModelSpec]]" = {}
    for found in specs():
        grouped.setdefault(found.key, []).append(found)
    return grouped
