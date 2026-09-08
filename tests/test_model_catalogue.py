"""Tests for the model catalogue (agent_stonks/model_catalogue.py).

Three things are worth pinning, and they are the three ways this module can be
wrong without anyone noticing.

**The mirrored paths.** `model_catalogue` reads the N-BEATS and day-range
sidecars without importing the modules that own them, because those modules
import PyTorch at module scope and a tab that lists model names must not drag a
200 MB dependency into the process. The price of that is a copy of their
`model_path` lookup, and the copy has to keep agreeing -- a renamed file or a
new env override would otherwise show a user a path nothing reads. These tests
import the real modules (torch and all) and compare, so the drift fails here
rather than on the page. They skip when torch is not installed, which is the
same condition under which the models themselves are unavailable.

**The reading itself.** A spec is assembled off files on disk, so the tests
that exercise it build their own: a bundle stub for the joblib models and a
sidecar JSON for the torch ones, pointed at with the env overrides. What is
pinned is that the metrics, features and versions come out where the UI expects
them, and that a missing file or dependency becomes an *unavailable* spec with
a reason instead of an exception.

**The import weight.** `specs()` must not pull torch or lightgbm in. That is
the whole design of the module, and it is one assertion.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from agent_stonks import apple_models, model_catalogue as mc


# --- the mirrored paths -----------------------------------------------------

@pytest.mark.parametrize("ticker", ["AAPL", "GOOGL", "INTC"])
def test_nbeats_path_mirrors_the_real_one(ticker):
    torch_model = pytest.importorskip("agent_stonks.nbeats_model")
    assert mc._saved_path(
        "APPLE_NBEATS_MODEL", "timetochange2_nbeats_{ticker}.pt", ticker
    ) == torch_model.model_path(ticker)


@pytest.mark.parametrize("ticker", ["AAPL", "GOOGL", "INTC"])
def test_dayrange_path_mirrors_the_real_one(ticker):
    dayrange = pytest.importorskip("agent_stonks.dayrange_model")
    assert mc._saved_path(
        "APPLE_DAYRANGE_MODEL", "timetochange3_dayrange_{ticker}.joblib", ticker
    ) == dayrange.model_path(ticker)


@pytest.mark.parametrize("ticker", ["AAPL", "GOOG", "INTC"])
def test_pricerange_path_mirrors_the_real_one(ticker):
    """Including the case of the filename, which is the notebook's, not ours.

    PriceRange2 builds its name from `config.MODEL_NAME` and so writes
    `pricerange2_aapl.joblib`. The catalogue mirrors the lookup rather than
    importing the module, so this is where the two can drift apart -- and an
    upper-cased mirror would show a user a path nothing reads.
    """
    pricerange = pytest.importorskip("agent_stonks.pricerange_model")
    mirrored = mc._saved_path(
        "APPLE_PRICERANGE_MODEL", "pricerange2_{ticker}.joblib", ticker, lowercase=True
    )
    assert mirrored == pricerange.model_path(ticker)
    assert mirrored.name == f"pricerange2_{ticker.lower()}.joblib"


def test_pricerange_sidecar_path_mirrors_the_real_one():
    pricerange = pytest.importorskip("agent_stonks.pricerange_model")
    spec = mc.spec(apple_models.PRICERANGE_KEY, "AAPL")
    by_role = {f.role: f.path for f in spec.files}
    real = pricerange.model_path("AAPL")
    assert by_role["bundle"] == real
    assert by_role["metadata"] == pricerange.metadata_path(real)


def test_nbeats_sidecar_paths_mirror_the_real_ones():
    torch_model = pytest.importorskip("agent_stonks.nbeats_model")
    spec = mc.spec(apple_models.NBEATS_KEY, "AAPL")
    by_role = {f.role: f.path for f in spec.files}
    real = torch_model.model_path("AAPL")
    assert by_role["checkpoint"] == real
    assert by_role["residual sidecar"] == torch_model.residuals_path(real)
    assert by_role["metadata"] == torch_model.metadata_path(real)


def test_dayrange_sidecar_paths_mirror_the_real_ones():
    dayrange = pytest.importorskip("agent_stonks.dayrange_model")
    spec = mc.spec(apple_models.DAYRANGE_KEY, "AAPL")
    by_role = {f.role: f.path for f in spec.files}
    real = dayrange.model_path("AAPL")
    assert by_role["bundle"] == real
    assert by_role["N-BEATS weights"] == dayrange.checkpoint_path("nbeats", real)
    assert by_role["N-HiTS weights"] == dayrange.checkpoint_path("nhits", real)
    assert by_role["metadata"] == dayrange.metadata_path(real)


def test_per_ticker_env_override_does_not_answer_for_other_symbols(monkeypatch, tmp_path):
    """The distinction each model module is careful about: the bare env var names
    one file, so it can only mean the default ticker's."""
    monkeypatch.setenv("APPLE_NBEATS_MODEL", str(tmp_path / "only_aapl.pt"))
    pattern = "timetochange2_nbeats_{ticker}.pt"
    assert mc._saved_path("APPLE_NBEATS_MODEL", pattern, "AAPL").name == "only_aapl.pt"
    assert mc._saved_path("APPLE_NBEATS_MODEL", pattern, "GOOGL").name == (
        "timetochange2_nbeats_GOOGL.pt"
    )
    monkeypatch.setenv("APPLE_NBEATS_MODEL_GOOGL", str(tmp_path / "googl.pt"))
    assert mc._saved_path("APPLE_NBEATS_MODEL", pattern, "GOOGL").name == "googl.pt"


# --- reading a spec off the files -------------------------------------------

NBEATS_SIDECAR = {
    "model": "nbeats",
    "n_seeds": 5,
    "trained_on": {
        "momentum": {"horizon": 15, "enter_threshold": 0.9, "exit_threshold": 0.4},
        "dataset": {"seq_len": 20},
    },
    "excluded_sessions_from": "2026-08-24",
    "train_sessions": ["2026-07-13", "2026-07-14"],
    "valid_sessions": ["2026-08-10"],
    "test_sessions": ["2026-08-18"],
    "metrics": {
        "persistence": {"threshold": 0.41, "roc_auc": 0.85, "brier": 0.147},
        "forecast": {"mae": 0.42, "mase": 0.696},
        "hard_half": {"n": 38.0, "roc_auc": 0.666},
    },
    "torch_version": "2.13.0",
}


def _write_nbeats(tmp_path, monkeypatch, *, sidecar=True, residuals=True):
    checkpoint = tmp_path / "nbeats_TEST.pt"
    checkpoint.write_bytes(b"not a real checkpoint")
    if residuals:
        (tmp_path / "nbeats_TEST_residuals.npz").write_bytes(b"residuals")
    if sidecar:
        (tmp_path / "nbeats_TEST.json").write_text(json.dumps(NBEATS_SIDECAR))
    monkeypatch.setenv("APPLE_NBEATS_MODEL_AAPL", str(checkpoint))
    return checkpoint


def test_nbeats_spec_reads_the_sidecar(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    _write_nbeats(tmp_path, monkeypatch)
    spec = mc.spec(apple_models.NBEATS_KEY, "AAPL")

    assert spec.available and not spec.unavailable_reason
    assert spec.threshold == pytest.approx(0.41)
    assert spec.versions == {"torch": "2.13.0"}
    assert spec.trained_at == "2026-07-13, 2026-07-14"
    # The hard half is the number worth quoting, and it wins the headline
    # whenever the training run recorded one.
    assert spec.headline == ("hard-half ROC AUC", "0.666")
    assert spec.metrics["persistence · roc_auc"] == 0.85
    assert spec.metrics["forecast · mae"] == 0.42
    assert spec.metrics["hard half · ROC AUC"] == 0.666
    assert "2026-08-24" in spec.data_note


def test_nbeats_spec_without_its_residual_sidecar_is_unavailable(tmp_path, monkeypatch):
    """A checkpoint alone is not a model: `nbeats_model` refuses to assemble one
    without the residuals its sampling step needs."""
    pytest.importorskip("torch")
    _write_nbeats(tmp_path, monkeypatch, residuals=False)
    spec = mc.spec(apple_models.NBEATS_KEY, "AAPL")

    assert not spec.available
    assert "residual sidecar" in spec.unavailable_reason


def test_missing_metadata_leaves_a_spec_described_but_not_broken(tmp_path, monkeypatch):
    """Metadata is descriptive, so losing it costs the metrics and nothing else --
    the estimator beside it would still trade."""
    pytest.importorskip("torch")
    _write_nbeats(tmp_path, monkeypatch, sidecar=False)
    spec = mc.spec(apple_models.NBEATS_KEY, "AAPL")

    assert not spec.available  # the metadata file is one of the model's files
    assert spec.metrics == {}
    assert spec.label and spec.predicts and spec.consumers


def test_momentum_change_spec_reads_its_json_sidecar(tmp_path, monkeypatch):
    bundle = tmp_path / "momentum_change_TEST.joblib"
    bundle.write_bytes(b"stub")
    (tmp_path / "momentum_change_TEST.json").write_text(
        json.dumps(
            {
                "ticker": "AAPL",
                "target": "mom_delta",
                "target_units": "bps/min (momentum 15 bars after)",
                "model_name": "Ridge",
                "feature_cols": ["mom_5", "mom_15", "theta"],
                "pipeline_params": {"window": 15, "smooth_halflife": 8.0, "c": 0.4},
                "train_days": ["2026-07-10", "2026-08-10"],
                "metrics": {"r2_test": 0.65, "holdout_r2": 0.478},
                "saved_at": "2026-09-07T21:34:50",
                "sklearn_version": "1.9.0",
            }
        )
    )
    monkeypatch.setenv("APPLE_MOMENTUM_CHANGE_MODEL_AAPL", str(bundle))
    spec = mc.spec(apple_models.MOMENTUM_CHANGE_KEY, "AAPL")

    assert spec.available
    assert spec.algorithm.startswith("Ridge")
    assert spec.features == ("mom_5", "mom_15", "theta")
    assert spec.headline == ("R² (holdout week)", "0.478")
    assert "2 training days" in spec.data_note
    assert spec.versions["scikit-learn"] == "1.9.0"


def test_open_profile_spec_reads_the_gzipped_pack():
    """The one model fitted across a universe rather than on a symbol, so its
    'instrument' says which tape it was trained on rather than which one it is
    for."""
    spec = mc.spec(mc.OPEN_PROFILE_KEY)
    assert spec.key == mc.OPEN_PROFILE_KEY
    # The symbol it runs on and the symbols it learned from are two facts, and
    # the overview table has room for only the first.
    assert spec.ticker == "any"
    assert "fitted on" in spec.ticker_note
    assert spec.consumers


def test_missing_file_is_a_reason_not_an_exception(tmp_path, monkeypatch):
    monkeypatch.setenv("OPEN_PROFILE_MODEL", str(tmp_path / "nope.json.gz"))
    spec = mc.spec(mc.OPEN_PROFILE_KEY)
    assert not spec.available
    assert "nope.json.gz" in spec.unavailable_reason
    assert spec.metrics == {}


# --- the catalogue as a whole -----------------------------------------------

def test_specs_cover_every_registered_model_and_ticker():
    """One spec per (model, instrument) that exists, plus the transferring pack.

    `apple_models.MODELS` is the registry; a model added there without a
    catalogue entry would silently vanish from the tab.
    """
    expected = {
        (key, ticker)
        for key, model in apple_models.MODELS.items()
        for ticker in model.tickers
    }
    found = {(s.key, s.ticker) for s in mc.specs() if s.key != mc.OPEN_PROFILE_KEY}
    assert found == expected
    assert any(s.key == mc.OPEN_PROFILE_KEY for s in mc.specs())


def test_every_spec_is_fully_described():
    """The fields the tab renders unconditionally must never come back empty --
    an unavailable model still has to say what it would predict and why it is
    missing."""
    for spec in mc.specs():
        assert spec.label and spec.predicts and spec.target and spec.algorithm
        # `family` is what the overview table renders, and it is short by
        # design -- a clipped `algorithm` would land mid-phrase.
        assert spec.family and len(spec.family) <= 40
        assert spec.consumers and spec.inputs and spec.project and spec.caveat
        # The section intro. Distinct from `predicts`, which the card renders
        # right below it -- one model reading the same line twice is a bug.
        assert spec.summary and spec.summary != spec.predicts
        assert spec.files
        assert spec.available or spec.unavailable_reason


def test_format_metric_spans_the_magnitudes_on_the_page():
    assert mc.format_metric(0.841) == "0.841"
    assert mc.format_metric(0.0077353) == "0.00774"
    assert mc.format_metric(76.7) == "76.70"
    assert mc.format_metric(2.1887e-05) == "2.19e-05"
    assert mc.format_metric(3350) == "3,350"
    assert mc.format_metric(float("nan")) == "—"
    assert mc.format_metric(None) == "—"


def test_both_apps_render_the_same_panel():
    """One page, two apps. A copy in each would drift -- one of them still
    quoting a metric the other had stopped showing -- so both import the same
    entry point and this pins that they do."""
    from agent_stonks import model_catalogue_ui

    assert callable(model_catalogue_ui.model_catalogue_panel)
    root = Path(__file__).resolve().parents[1]
    for app in ("agent_stonks/ui.py", "simlab/app.py"):
        source = (root / app).read_text()
        assert "model_catalogue_panel()" in source, app


def test_building_the_catalogue_does_not_import_torch_or_lightgbm():
    """The design constraint of the whole module: `apple_models` keeps a 200 MB
    dependency off the default start-up path, and a page that merely *lists*
    models must not undo that."""
    script = (
        "import sys;"
        "from agent_stonks import model_catalogue as mc;"
        "from agent_stonks import model_catalogue_ui;"
        "specs = mc.specs();"
        "assert len(specs) > 1;"
        "assert 'torch' not in sys.modules, 'torch was imported';"
        "assert 'lightgbm' not in sys.modules, 'lightgbm was imported'"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
