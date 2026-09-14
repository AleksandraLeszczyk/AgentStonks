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
import os
from dataclasses import dataclass, field
from importlib.util import find_spec
from pathlib import Path

from . import apple_models, model_store

# The shared store beside the AgentStonks checkout, where every model family
# resolves its files from (see `model_store.ModelStore`).
MODEL_DIR = model_store.MODEL_DIR

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
    apple_models.DAYRANGE_KEY: _dayrange_spec,
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
