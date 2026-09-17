"""Tests for the model catalogue (agent_stonks/model_catalogue.py).

Three things are worth pinning, and they are the three ways this module can be
wrong without anyone noticing.

**The mirrored paths.** `model_catalogue` reads the day-range sidecar
without importing the module that owns it, because that module imports
PyTorch at module scope and a tab that lists model names must not drag a
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
def test_dayrange_path_mirrors_the_real_one(ticker):
    dayrange = pytest.importorskip("agent_stonks.dayrange_model")
    assert mc._saved_path(
        "APPLE_DAYRANGE_MODEL", "timetochange3_dayrange_{ticker}.joblib", ticker
    ) == dayrange.model_path(ticker)


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
    monkeypatch.setenv("APPLE_DAYRANGE_MODEL", str(tmp_path / "only_aapl.joblib"))
    pattern = "timetochange3_dayrange_{ticker}.joblib"
    assert mc._saved_path("APPLE_DAYRANGE_MODEL", pattern, "AAPL").name == "only_aapl.joblib"
    assert mc._saved_path("APPLE_DAYRANGE_MODEL", pattern, "GOOGL").name == (
        "timetochange3_dayrange_GOOGL.joblib"
    )
    monkeypatch.setenv("APPLE_DAYRANGE_MODEL_GOOGL", str(tmp_path / "googl.joblib"))
    assert mc._saved_path("APPLE_DAYRANGE_MODEL", pattern, "GOOGL").name == "googl.joblib"


# --- reading a spec off the files -------------------------------------------

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


# --- the day-range headline -------------------------------------------------

DAYRANGE_SIDECAR = {
    "created": "2026-09-09T16:09:02",
    "daily_fit_through": "2026-07-09",
    "opening_correction": True,
    "held_out": "2026-08-28",
    "test_metrics_ensemble": {"mae_mean": 0.008, "mae_usd_mean": 2.4},
    # prev_avg 200, adr14_abs 4 -> a normal day is 2% of the price, so an
    # error of 0.008 log units is 40% of one.
    "sim_date_forecast": {"date": "2026-08-28", "prev_avg": 200.0, "adr14_abs": 4.0},
}


def _dayrange_spec_from(meta, tmp_path, monkeypatch, ticker="AAPL"):
    """A spec built off a sidecar written here, with no bundle beside it.

    The bundle's absence makes the spec *unavailable*, which is orthogonal to
    what these tests read: the headline is assembled from the JSON either way.
    """
    bundle = tmp_path / "timetochange3_dayrange_TEST.joblib"
    if meta is not None:
        bundle.with_suffix(".json").write_text(json.dumps(meta))
    monkeypatch.setenv(f"APPLE_DAYRANGE_MODEL_{ticker}", str(bundle))
    return mc.spec(apple_models.DAYRANGE_KEY, ticker)


class TestDayRangeErrorPctAdr:
    """The headline is a ratio of two things the sidecar records separately.

    Both sides are fractions of a price -- the MAE because the target is a log
    ratio, the ADR because it is divided by the day it was measured on -- so
    the price cancels and what is left is "how much of a normal day the model
    misses by", which is the unit the trading rules are written in.
    """

    def test_the_ratio_is_the_log_mae_over_the_relative_adr(self):
        assert mc.dayrange_error_pct_adr(DAYRANGE_SIDECAR) == pytest.approx(40.0)

    def test_the_spec_leads_with_it(self, tmp_path, monkeypatch):
        spec = _dayrange_spec_from(DAYRANGE_SIDECAR, tmp_path, monkeypatch)
        assert spec.headline == ("MAE (% of ADR)", "40.0%")
        # First in the metrics table too, and numeric so the table formats it.
        assert next(iter(spec.metrics)) == "MAE (% of ADR)"
        assert spec.metrics["MAE (% of ADR)"] == pytest.approx(40.0)
        assert "40% of a typical day's range" in spec.caveat

    @pytest.mark.parametrize("meta", [
        {k: v for k, v in DAYRANGE_SIDECAR.items() if k != "sim_date_forecast"},
        {**DAYRANGE_SIDECAR, "sim_date_forecast": {"prev_avg": 200.0, "adr14_abs": 0.0}},
        {**DAYRANGE_SIDECAR, "test_metrics_ensemble": {}},
        {},
    ])
    def test_no_adr_to_divide_by_is_no_percentage(self, meta):
        """A percentage of an assumed range would be worse than the raw error."""
        assert mc.dayrange_error_pct_adr(meta) is None

    def test_the_headline_falls_back_to_the_log_mae(self, tmp_path, monkeypatch):
        meta = {k: v for k, v in DAYRANGE_SIDECAR.items() if k != "sim_date_forecast"}
        spec = _dayrange_spec_from(meta, tmp_path, monkeypatch)
        assert spec.headline == ("MAE (log units)", "0.00800")
        assert "MAE (% of ADR)" not in spec.metrics

    def test_an_unreadable_sidecar_still_builds_a_spec(self, tmp_path, monkeypatch):
        spec = _dayrange_spec_from(None, tmp_path, monkeypatch)
        assert spec.headline == ("MAE (log units)", "—")
        assert not spec.available


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
    } | {
        # Not a registry model -- it drives only the chart overlays -- but it is
        # per ticker too, and the tab lists one card per exported file.
        (mc.INTRADAY_VOL_KEY, ticker) for ticker in mc.intraday_vol_model.TICKERS
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
