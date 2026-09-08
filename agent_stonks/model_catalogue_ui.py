"""The ML Models page, rendered the same way in both apps.

Both Streamlit apps want this page and neither owns it. The live dashboard
shows it beside its Agent tab because the models are what those agents read;
SimLab shows it beside its Agents tab because a replay is only interpretable
if you know what the model behind it was fitted on. A page that drifted between
the two -- one app quoting a metric the other had already stopped showing --
would be worse than no page, so there is one of it, here, and both apps call
`model_catalogue_panel()`. The same reason `apple_rules_ui` is a shared module
rather than a copy in each app.

Everything rendered comes from `model_catalogue`, which reads the saved bundles
and their JSON sidecars and deliberately imports neither PyTorch nor LightGBM.
Nothing in this module loads a model either -- opening the tab in either app
costs a handful of `stat` calls and two small JSON reads.
"""

from __future__ import annotations

import html

import streamlit as st

from . import apple_models, model_catalogue
from .config import PALETTE


# The two tables on the ML Models tab are hand-rolled HTML rather than
# `st.dataframe`, for the reason SimLab's breakdown table is: a canvas grid
# clips rather than wraps, and both of these carry a column of prose (what a
# model predicts, what a metric is called) that has to wrap to be readable.
_MODEL_TABLE_CSS = f"""
<style>
table.model-catalogue {{
    width: 100%; border-collapse: collapse; font-size: 0.85rem;
    table-layout: fixed;
}}
table.model-catalogue th, table.model-catalogue td {{
    padding: 0.35rem 0.6rem; text-align: left; vertical-align: top;
    border-bottom: 1px solid {PALETTE["grid"]};
}}
table.model-catalogue th {{ font-weight: 600; opacity: 0.75; white-space: nowrap; }}
table.model-catalogue td.num {{
    text-align: right; white-space: nowrap; font-family: monospace;
}}
table.model-catalogue td.name {{ font-family: monospace; font-size: 0.8rem; }}
table.model-catalogue .ready {{ color: {PALETTE["up"]}; white-space: nowrap; }}
table.model-catalogue .missing {{ color: {PALETTE["down"]}; white-space: nowrap; }}
table.model-catalogue .muted {{ color: {PALETTE["muted"]}; }}
/* Column widths rather than the browser's guess: left to itself the overview
   gives the estimator names the room and squeezes the prose column into a
   word-per-line ribbon -- and that is the column that has to be readable. */
table.model-catalogue.overview th:nth-child(1) {{ width: 15%; }}
table.model-catalogue.overview th:nth-child(2) {{ width: 10%; }}
table.model-catalogue.overview th:nth-child(3) {{ width: 17%; }}
table.model-catalogue.overview th:nth-child(4) {{ width: 35%; }}
table.model-catalogue.overview th:nth-child(5) {{ width: 12%; }}
table.model-catalogue.overview th:nth-child(6) {{ width: 11%; }}
/* The metrics table is two columns of very different weight: a name that has
   to wrap, and a right-aligned number that never does. */
table.model-catalogue.metrics th:nth-child(1) {{ width: 76%; }}
table.model-catalogue.metrics th:nth-child(2) {{ width: 24%; }}
</style>
"""


def _model_overview_html(grouped: "dict[str, list[model_catalogue.ModelSpec]]") -> str:
    """Every (model, instrument) pair in one table.

    One row per pair rather than per model, because that is the unit that was
    fitted: AAPL's delta-momentum regressor is a Ridge and INTC's is a
    HistGradientBoosting, and a single row would have to pick one to be true
    about.
    """
    rows = []
    for specs in grouped.values():
        for spec in specs:
            status = (
                '<span class="ready">● Ready</span>'
                if spec.available
                else '<span class="missing">● Unavailable</span>'
            )
            metric = (
                f"{html.escape(spec.headline[1])}"
                f'<br><span class="muted">{html.escape(spec.headline[0])}</span>'
                if spec.headline
                else '<span class="muted">—</span>'
            )
            note = (
                f' title="{html.escape(spec.ticker_note, quote=True)}"'
                if spec.ticker_note
                else ""
            )
            rows.append(
                "<tr>"
                f"<td>{html.escape(spec.label)}</td>"
                f'<td class="name"{note}>{html.escape(spec.ticker)}</td>'
                f"<td>{html.escape(spec.family)}</td>"
                f"<td>{html.escape(_model_one_liner(spec))}</td>"
                f'<td class="num">{metric}</td>'
                f"<td>{status}</td>"
                "</tr>"
            )
    head = "".join(
        f"<th>{label}</th>"
        for label in (
            "Model", "Instrument", "Type", "Predicts", "Headline metric", "Status"
        )
    )
    return (
        _MODEL_TABLE_CSS
        + f'<table class="model-catalogue overview"><thead><tr>{head}</tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _model_one_liner(spec: model_catalogue.ModelSpec) -> str:
    """The first sentence of what a model predicts, with the markdown stripped."""
    first = spec.predicts.replace("**", "").split(". ")[0].rstrip(".")
    return first if len(first) <= 120 else first[:117] + "…"


def _model_metrics_html(spec: model_catalogue.ModelSpec) -> str:
    """One model's metrics, exactly as its file recorded them.

    Every metric rather than a chosen few: which numbers matter depends on what
    the reader is asking, and a page that silently dropped the ones that
    flatter a model least would be the wrong page.
    """
    rows = "".join(
        f'<tr><td class="name">{html.escape(name)}</td>'
        f'<td class="num">{html.escape(model_catalogue.format_metric(value))}</td></tr>'
        for name, value in spec.metrics.items()
    )
    return (
        _MODEL_TABLE_CSS
        + '<table class="model-catalogue metrics"><thead><tr><th>Metric</th>'
        f"<th>Value</th></tr></thead><tbody>{rows}</tbody></table>"
    )


def _model_status_html(spec: model_catalogue.ModelSpec) -> str:
    """The one-line verdict at the top of a model card.

    "Ready" is a claim about files and dependencies being *present*, not about
    the estimator having been unpickled -- `model_catalogue` deliberately does
    not load a torch checkpoint to fill this page in. The wording says so.
    """
    if spec.available:
        return (
            f"<span style='color:{PALETTE['up']}'>● Ready</span>"
            f"<span style='color:{PALETTE['muted']}'> — files and dependencies "
            f"present</span>"
        )
    return (
        f"<span style='color:{PALETTE['down']}'>● Unavailable</span>"
        f"<span style='color:{PALETTE['muted']}'> — "
        f"{html.escape(spec.unavailable_reason or 'not found')}</span>"
    )


def _model_files_html(spec: model_catalogue.ModelSpec) -> str:
    """Every file the model is assembled from, with its size and whether it is
    there. More than one for three of the five: a checkpoint without its
    residual sidecar, or a day-range bundle without its two `.pt` files, is not
    a model that loads."""
    rows = []
    for entry in spec.files:
        size = entry.size_mb
        mark = (
            f"<span style='color:{PALETTE['up']}'>✓</span>"
            if entry.exists
            else f"<span style='color:{PALETTE['down']}'>✗</span>"
        )
        size_txt = f" · {size:.1f} MB" if size is not None else ""
        rows.append(
            f"<div style='font-family:monospace;font-size:11px;margin:2px 0'>"
            f"{mark} <span style='color:{PALETTE['muted']}'>{html.escape(entry.role)}</span> "
            f"{html.escape(str(entry.path))}"
            f"<span style='color:{PALETTE['muted']}'>{size_txt}</span></div>"
        )
    return "".join(rows)


def _model_feature_html(spec: model_catalogue.ModelSpec) -> str:
    """The feature names as monospace chips.

    The whole list rather than a count: the feature set *is* the contract each
    of these modules keeps with the notebook package it mirrors, so a reader
    checking whether the app and the training code still agree needs to see the
    names, not a number.
    """
    if not spec.features:
        return (
            f"<span style='color:{PALETTE['muted']}'>Feature list unavailable — it "
            f"lives in the saved file, which could not be read.</span>"
        )
    chips = " ".join(
        f"<span style='background:{PALETTE['panel']};border:1px solid {PALETTE['grid']};"
        f"border-radius:4px;padding:1px 5px;margin:2px 2px 0 0;display:inline-block;"
        f"font-family:monospace;font-size:11px'>{html.escape(name)}</span>"
        for name in spec.features
    )
    return f"<div style='line-height:2'>{chips}</div>"


def _model_spec_card(spec: model_catalogue.ModelSpec) -> None:
    """One (model, instrument) pair in full: what it predicts, off what, how
    well, and where it lives."""
    st.markdown(_model_status_html(spec), unsafe_allow_html=True)
    st.caption(
        f":material/finance: Instrument — **{spec.ticker}**"
        + (f" ({spec.ticker_note})" if spec.ticker_note else "")
    )
    st.markdown(f"**Predicts** — {spec.predicts}")
    st.markdown(
        f"<div style='font-family:monospace;font-size:11px;color:{PALETTE['muted']};"
        f"margin-bottom:8px'>target: {html.escape(spec.target)}</div>",
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)
    with left:
        st.markdown("**Algorithm**")
        st.markdown(f"{spec.algorithm}")
        if spec.threshold is not None:
            st.markdown(
                f"Decision threshold **{spec.threshold:.2f}** — chosen on this "
                "symbol's own validation events, and not comparable across models "
                "or across symbols."
            )
        st.markdown("**Read by**")
        st.markdown("\n".join(f"- {c}" for c in spec.consumers))
    with right:
        st.markdown("**Inputs**")
        st.markdown(spec.inputs)
        st.markdown(f"**Features ({spec.n_features})**")
        st.markdown(_model_feature_html(spec), unsafe_allow_html=True)

    if spec.metrics:
        st.markdown("**Metrics**")
        st.markdown(_model_metrics_html(spec), unsafe_allow_html=True)
    if spec.caveat:
        st.info(spec.caveat, icon=":material/priority_high:")

    st.markdown("**Metadata**")
    st.markdown(_model_files_html(spec), unsafe_allow_html=True)
    meta_bits = []
    if spec.trained_at:
        meta_bits.append(f"**Trained on** {spec.trained_at}")
    if spec.data_note:
        meta_bits.append(spec.data_note)
    if spec.versions:
        meta_bits.append(
            "**Fitted with** "
            + ", ".join(f"{lib} {ver}" for lib, ver in spec.versions.items())
        )
    if spec.requires:
        meta_bits.append(f"**Requires** {spec.requires}")
    meta_bits.append(f"**Project** {spec.project}")
    for bit in meta_bits:
        st.caption(bit)


def model_catalogue_panel() -> None:
    """Every trained model this app can read, with its specification.

    Grouped by model rather than by instrument, because a model is the unit
    that was fitted and graded -- and then split per instrument inside, because
    that is the unit that actually exists on disk. Nothing here loads a model:
    the page is assembled from the saved bundles and their JSON sidecars, so
    opening the tab does not pull PyTorch into the process.
    """
    st.caption(
        "The saved models behind this app's non-LLM decisions. Each was fitted in its "
        "own notebook project and is mirrored here feature-for-feature — the app "
        "recomputes the training code's exact columns from the live tape, so a drift "
        "between the two is the classic way a saved model goes silently wrong. Every "
        "number below is read from the model's own file, so a retrain updates this page "
        "on its own. All of them are optional: a missing file or a missing dependency "
        "makes a model *unavailable* and says so, rather than an agent that quietly "
        "never trades."
    )

    grouped = model_catalogue.specs_by_model()
    st.markdown(_model_overview_html(grouped), unsafe_allow_html=True)
    st.caption(
        ":material/info: The headline metrics are **not comparable across rows** — a "
        "persistence AUC, a day-range MAE in log units, an R² on bps/min and an EMD in "
        "bps answer different questions on different data. Read each model's own "
        "section for what its number does and does not say."
    )

    for key, specs in grouped.items():
        registry = (
            apple_models.MODELS.get(key)
            if key in apple_models.MODELS
            else None
        )
        st.markdown(f"### {specs[0].label}")
        st.markdown(specs[0].summary)
        if registry is not None:
            st.caption(
                f":material/rule: Drives the **{registry.strategy}** strategy · "
                + (
                    "can be asked about a change that has not happened yet"
                    if registry.anticipates
                    else "answers only about a change the tape has printed"
                    if registry.strategy == apple_models.STRATEGY_MOMENTUM
                    else "not a per-bar momentum signal"
                )
            )
        else:
            st.caption(
                ":material/rule: Drives no trading strategy — it is read by the "
                "charts, which draw the predicted profile beside the tape."
            )
        if len(specs) == 1:
            _model_spec_card(specs[0])
            continue
        for tab, spec in zip(st.tabs([s.ticker for s in specs]), specs):
            with tab:
                _model_spec_card(spec)
