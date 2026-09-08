"""SimLab Streamlit UI: agents / datasets / simulate / summary / results.

Run with ``streamlit run sim_main.py``. Kept separate from the live dashboard
(``main.py``) -- this app never opens a stream or touches the live tape; it
only reads the local dataset store and replays agents against it.
"""
from __future__ import annotations

import json
import os
import textwrap
import uuid
from datetime import date, datetime, time, timedelta, timezone
from html import escape
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from agent_stonks import (
    apple_models,
    clock,
    model_overlays,
    momentum_change_model,
    persistence_model,
)
from agent_stonks import observability as obs
from agent_stonks.agent import (
    AGENT_PERSONALITIES,
    PERSONALITY_TOOLS,
    PREMARKET_PERSONALITY,
    _dispatch_tool,
    selectable_personalities,
)
from agent_stonks.apple_trader import (
    APPLE_TRADER_KEY,
    RULE_PROVIDER,
    AppleTraderConfig,
)
from agent_stonks import apple_trader_ui
from agent_stonks.apple_rules_ui import rules_panel, signal_catalogue
from agent_stonks.apple_trader2 import APPLE_TRADER2_KEY, AppleTrader2Config
from agent_stonks.charts import (
    add_model_overlays,
    add_session_markers,
    overlay_x_max,
    session_rangebreaks,
)
from agent_stonks.config import PALETTE
from agent_stonks.model_catalogue_ui import model_catalogue_panel
from agent_stonks.llm import DEFAULT_AGENT_MODELS, ENV_KEYS, PROVIDERS, models_for
from agent_stonks.market_hours import MARKET_TZ

from . import data as sim_data
from . import experiments as sim_experiments
from . import prompts as sim_prompts
from . import results as sim_results
from .engine import SimulationConfig, SimulationEngine
from .market import SimMarket
from .patches import simulation_context
from .rule_agents import RULE_AGENTS, rule_agent

AVATAR_DIR = Path(__file__).resolve().parent.parent / "data" / "avatars"

# submit_decision / set_tactics mutate the ledger and need a full cycle around
# them -- the hand-tester exposes only the read/analysis tools.
_UNTESTABLE_TOOLS = {"submit_decision", "set_tactics", "stand_down"}

def _testable_agents() -> list[str]:
    """Every agent SimLab can replay, in picker order: the LLM personalities
    first, the rule-based ones last.

    The Premarket Analyst is hidden from SimLab only -- it stays wired
    everywhere else (the app, the Automatic pre-open handoff, and the engine's
    own premarket day loop), and past premarket runs still resolve their label,
    avatar and prompt. Re-offering it is a one-line change here."""
    return [
        *(p for p in selectable_personalities() if p != PREMARKET_PERSONALITY),
        *RULE_AGENTS,
    ]


def _agent_label(key: "str | None") -> str:
    """Rule agents have no entry in AGENT_PERSONALITIES, so their label comes
    from the registry instead."""
    if key in RULE_AGENTS:
        return RULE_AGENTS[key].label
    return AGENT_PERSONALITIES.get(key or "", {}).get("label", key or "?")


def _agent_avatar(key: str) -> Path:
    if key in RULE_AGENTS:
        return AVATAR_DIR / RULE_AGENTS[key].avatar
    return AVATAR_DIR / AGENT_PERSONALITIES.get(key, {}).get("avatar", "")


def _env_key(provider: str) -> str:
    return os.getenv(ENV_KEYS.get(provider, ""), "")


@st.cache_data(show_spinner=False)
def _load_runs(signature: tuple) -> list[dict]:
    # `signature` is unused on purpose: it is the cache key, so the store is
    # re-read exactly when a run file is added, changed, or deleted. It must
    # not be underscore-prefixed -- Streamlit excludes those from the key.
    return sim_results.list_runs()


def _runs() -> list[dict]:
    """Stored run records, re-read only when the run store actually changes.
    Full records carry decisions and agent logs, so parsing every one of them
    on every rerun adds up."""
    return _load_runs(sim_results.store_signature())


def _chart_layout(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=40, r=20, t=30, b=30),
        paper_bgcolor=PALETTE["bg"],
        plot_bgcolor=PALETTE["panel"],
        font=dict(color=PALETTE["text"]),
        xaxis=dict(gridcolor=PALETTE["grid"]),
        yaxis=dict(gridcolor=PALETTE["grid"]),
        showlegend=True,
    )
    return fig


# ---------------------------------------------------------------------------
# Tab 1 — agents
# ---------------------------------------------------------------------------

def _render_tool_tester(personality: str) -> None:
    st.markdown("##### Try a tool by hand")
    datasets = sim_data.list_datasets()
    if not datasets:
        st.info("Download a dataset first (Datasets tab) to test tools against stored data.")
        return

    ds_names = [d.name for d in datasets]
    col_ds, col_sym, col_day = st.columns(3)
    ds = sim_data.get_dataset(col_ds.selectbox("Dataset", ds_names, key="tt_ds"))
    symbol = col_sym.selectbox("Symbol", ds.symbols, key="tt_sym")
    day = date.fromisoformat(col_day.selectbox("Day", ds.days or [ds.start], key="tt_day"))
    probe_time = st.slider(
        "Moment (ET)",
        min_value=time(4, 30),
        max_value=time(20, 0),
        value=time(10, 30),
        step=timedelta(minutes=5),
        key="tt_time",
    )
    at = datetime.combine(day, probe_time, tzinfo=MARKET_TZ).astimezone(timezone.utc)

    tools = [
        t["function"] for t in PERSONALITY_TOOLS[personality]
        if t["function"]["name"] not in _UNTESTABLE_TOOLS
    ]
    tool = st.selectbox(
        "Tool", tools, format_func=lambda t: t["name"], key="tt_tool"
    )
    with st.expander("What this tool does"):
        st.write(tool["description"])

    args: dict = {}
    props = (tool.get("parameters") or {}).get("properties", {})
    extra = {k: v for k, v in props.items() if k != "symbol"}
    if extra:
        cols = st.columns(min(3, len(extra)))
        for i, (name, spec) in enumerate(extra.items()):
            raw = cols[i % len(cols)].text_input(
                name, key=f"tt_arg_{tool['name']}_{name}",
                help=spec.get("description", ""), placeholder="default",
            )
            if raw.strip():
                try:
                    args[name] = json.loads(raw)
                except json.JSONDecodeError:
                    args[name] = raw
    args["symbol"] = symbol

    if st.button("Run tool", icon=":material/play_arrow:", type="primary", key="tt_run"):
        market = SimMarket(ds.symbols, [day], ds.feed)
        config = SimulationConfig(
            personality=personality, provider="openai", model="-", api_key="",
            symbols=ds.symbols, days=[day], feed=ds.feed,
        )
        engine = SimulationEngine(market, config)
        with simulation_context(market):
            engine.seed_until(at)
            clock.set_simulated(at)
            result = _dispatch_tool(tool["name"], args, engine.app, engine.tracker)
        st.caption(f"`{tool['name']}` at {at.astimezone(MARKET_TZ).strftime('%Y-%m-%d %H:%M ET')}")
        st.json(result)


def _render_rule_agent(personality: str) -> None:
    """A rule agent's "prompt": its rules, and whatever they depend on.

    There is nothing to edit here the way a system prompt is edited -- the
    thresholds (Apple Trader) and the rule list (Apple Trader 2) are
    per-simulation settings, picked in the Simulate tab -- so this is a
    read-only description of what the loop does.
    """
    st.subheader(_agent_label(personality))
    st.caption(
        ":material/function: Rule-based — no LLM, no prompt, no tools. The same tape "
        "always produces the same trades."
    )
    if personality == APPLE_TRADER2_KEY:
        _render_apple2_rules()
        return
    _render_apple_rules()


def _render_apple2_rules() -> None:
    """What Apple Trader 2 is: a vocabulary, and whatever list is written in it.

    Nothing to describe as *the* strategy, because there isn't one -- so what
    this page owes the reader instead is the vocabulary itself, the rules the
    engine applies around whatever list it is handed, and which of the shipped
    strategies can be reproduced in it (all of them).
    """
    st.markdown(
        "The same fixed loop over one symbol's minute bars as Apple Trader, with the "
        "strategy taken out of the code. A run is configured with an **instrument** and "
        "a **list of action items**, each one a buy or a sell, a size, and the "
        "conditions that arm it — written in the **Simulate** tab and carried on the "
        "experiment record, the way a prompt is for an LLM agent. Like Apple Trader it "
        "states no reasoning of its own, so the judge never scores it: profit, profit "
        "efficiency and the oracle ceiling are the whole verdict."
    )

    st.markdown("##### What one rule is")
    st.markdown(
        "- **An action** — buy or sell.\n"
        "- **A size** — a percentage (of cash on a buy, of the position on a sell), a "
        "dollar amount, or a share count. Every mode is clipped to what the ledger can "
        "do, so a rule set is portable across starting balances.\n"
        "- **Conditions** — one or more, each a signal against a number, joined by "
        "**AND** or **OR**. One joiner per rule: `A and B or C` has no meaning without "
        "precedence rules, and a genuine mix is two rules.\n"
        "- Optionally a **cooldown**: bars the rule sits out after firing, so a "
        "condition that stays true ladders in only if that was the intent."
    )

    st.markdown("##### What the engine adds")
    st.markdown(
        "- **At most one action per closed bar**, and the **first matching rule wins** — "
        "the list is a priority order. A rule that matches but cannot transact (a sell "
        "with the book flat, a buy with no cash) is passed over rather than eating the "
        "bar, so an exit written above an entry does not block it.\n"
        "- **An absent signal never matches** — not in an AND and not in an OR. A model "
        "asked about a bar it was not fitted for, a day-range forecast before 9:35, a "
        "P&L with no position: all read as nothing, and nothing fires a rule.\n"
        "- **The flatten before the close is not a rule** and cannot be deleted. Every "
        "signal here is intraday and none survives the overnight gap.\n"
        "- **Nothing is computed that no rule reads.** The bundles loaded are the ones "
        "the conditions name, the day-range forecast is made only if something asks, and "
        "conditions short-circuit within a rule."
    )

    st.markdown("##### The signals a condition can read")
    st.caption(
        "The models are here as *signals*, not as strategies: `nbeats.turn_proba` and "
        "`persistence.proba` are two models answering the same question and a rule set "
        "may name both. What each number is worth is the same open question it is under "
        "Apple Trader — see that agent's page, and read the AUC caveats there before "
        "building a rule on one. This is the full catalogue, which is what "
        f"{apple_models.DEFAULT_TICKER} offers; every other instrument gets the subset "
        "its models cover — "
        + "; ".join(
            f"**{symbol}**: "
            + ", ".join(
                apple_models.get(k).label for k in apple_models.keys_for(symbol)
            )
            for symbol in apple_models.tickers()
        )
        + ". Anything else reads the tape, the momentum regime, the position and the "
        "clock only."
    )
    signal_catalogue()

    st.info(
        ":material/lightbulb: Apple Trader's two strategies are both expressible here, "
        "and ship as presets — so a rule set can be compared against the thing it was "
        "meant to improve on rather than against an intuition. What the vocabulary adds "
        "beyond them is partial exits, scaled entries, and conditions from one model "
        "crossed with another's."
    )


def _render_apple_rules() -> None:
    """What Apple Trader does, plus the provenance of the bundles it needs.

    Two strategies, and the model picked per simulation decides which runs.
    They are described separately because they have nothing in common: one
    asks a question on every bar, the other asks one at 9:35 and then works
    two price levels.
    """
    ticker = rule_agent(APPLE_TRADER_KEY).default_ticker
    st.markdown(
        "A fixed loop over one symbol's minute bars with no LLM anywhere in it. Which "
        "rules it runs is decided by which saved model it is pointed at, and which "
        "**instrument** it can be pointed at is decided by the same choice — both are "
        "picked per simulation in the **Simulate** tab. It is never scored by the LLM "
        "judge — it states no reasoning of its own to judge, so profit, profit "
        "efficiency and the oracle ceiling are the whole verdict."
    )
    st.caption(
        ":material/model_training: Every rule here is a saved model's output, so the "
        "instrument is only as free as the models are — "
        + "; ".join(
            f"**{symbol}**: "
            + ", ".join(apple_models.get(k).label for k in apple_models.keys_for(symbol))
            for symbol in apple_models.tickers()
        )
        + ". TimeToChange2 was only ever fitted on "
        f"{ticker}; TimeToChange3 was run per ticker."
    )

    st.markdown("##### The momentum rules — `persistence`, `nbeats`")
    st.markdown(
        f"Once a minute it reads the bar that just closed (**{ticker}** only — see "
        "above) and "
        "tracks the momentum regime — a Schmitt trigger over a volatility-normalised "
        "momentum score, so a value hovering near the line cannot emit a burst of fake "
        "changes:\n"
        "- **Buy** on **the entry mode**'s question, given the 20 bars leading into that "
        "bar. On *Anticipate* (the default) the regime is still negative or balanced and "
        "the model forecasts that it turns positive on the next bar; on *Confirm* the "
        "change has already printed and the model rates it likely to hold. The second is "
        "the notebook's rule, and it is structurally late — the momentum score has "
        "already crossed its threshold by then, so the fill lands after the move that "
        "produced the signal.\n"
        "- **Sell** on either of two triggers. Price falling **the trailing stop** below "
        "the highest price seen since the entry — the peak only ratchets up, so the rule "
        "starts as a stop under the entry and becomes a profit lock as the move runs. Or, "
        "if it is armed, **the forecast reversal**: the model putting the positive regime "
        "at the configured probability or better of flipping negative, which can close the "
        "position while price is still at its high. The stop waits for the give-back; the "
        "reversal acts before it.\n"
        "- Nothing else closes the position but the closing bell: every feature the model "
        "uses is intraday, so the book is flattened before the close rather than carried "
        "overnight."
    )

    st.markdown("##### The day-range rules — `dayrange`")
    st.markdown(
        "One question, asked once. At **9:35** the model forecasts where the session's "
        "high **H** and low will land, from a year of daily history plus the first five "
        "minutes, and the rest of the day is two levels derived from it and held fixed "
        "(TimeToChange3 notebook 05), with **A** the trailing 14-day average daily range "
        "in dollars:\n"
        "- **Buy** when a bar's low reaches `H − buy × A`, well under where the day is "
        "expected to top out.\n"
        "- **Sell** when a bar's high reaches `H − sell × A`, just under it. Then it can "
        "buy again, as often as the day allows.\n"
        "- Anything still open is flattened before the close. There is no stop: the "
        "forecast is a statement about where today tops out, so leaving early on weakness "
        "would be a second, unmeasured strategy on top of this one.\n\n"
        "It is a mean-reversion bet, and deliberately so — what the model forecasts well "
        "is the *width* of the day, not its direction. On a day that never dips to the buy "
        "level it does nothing at all."
    )
    st.caption(
        ":material/compare_arrows: **Against the notebook**, one difference matters and it "
        "runs against the strategy: the notebook rests limit orders and fills a touch *at* "
        "the level, while this ledger is market-order only and buys near the close of the "
        "bar that touched it. A bar that dipped to the level and recovered fills worse "
        "here than there. Read the two side by side with that in mind."
    )

    st.markdown("##### Models")
    st.caption(
        "The momentum question can be put to either of two saved TimeToChange2 models. "
        "Everything before the question — bars, momentum, regimes, all 25 features, the "
        "20-bar window — is identical for both, so a dataset run through each on the "
        "*Confirm* entry is a comparison of the models and nothing else. *Anticipate* "
        "asks about a bar that is not a regime change, which only a forecaster can "
        "answer. The day-range model is not on that scale and its numbers below are not "
        "comparable to theirs."
    )
    for model in (apple_models.get(key) for key in apple_models.keys()):
        with st.expander(model.label, expanded=model.key == AppleTraderConfig().model_key):
            st.markdown(model.summary)
            st.caption(
                ":material/candlestick_chart: Fitted on "
                + ", ".join(model.tickers)
                + (
                    " — one bundle per symbol, each fitted, selected and measured on "
                    "that symbol's own days."
                    if len(model.tickers) > 1
                    else ". Nothing claims it transfers, so it is the only instrument "
                         "this model can be run on."
                )
            )
            if model.strategy == apple_models.STRATEGY_MOMENTUM and not model.anticipates:
                st.caption(
                    ":material/block: Fitted on regime-change bars only, so it runs the "
                    "*Confirm* entry and not *Anticipate*."
                )
            # One provenance block per symbol: the numbers below are that
            # bundle's own, and quoting AAPL's for a GOOGL run would be a
            # different model's held-out error under the right heading.
            for symbol in model.tickers:
                if len(model.tickers) > 1:
                    st.markdown(f"**{symbol}**")
                bundle = apple_models.load(model.key, symbol)
                if bundle is None:
                    st.error(
                        f"{apple_models.unavailable_reason(model.key, symbol)} Simulations "
                        f"naming this model on {symbol} will fail until it is available."
                    )
                    continue
                if model.strategy == apple_models.STRATEGY_DAYRANGE:
                    _render_dayrange_bundle(bundle)
                    continue
                if model.strategy == apple_models.STRATEGY_MOMENTUM_CHANGE:
                    _render_momentum_change_bundle(bundle)
                    continue
                metrics = bundle.get("metrics") or {}
                cols = st.columns(4)
                cols[0].metric("Sequence", f"{bundle['seq_len']} bars")
                cols[1].metric("Features", len(bundle.get("feature_columns") or []))
                cols[2].metric("Held-out AUC", f"{metrics.get('roc_auc', float('nan')):.2f}")
                cols[3].metric(
                    "Own threshold", f"{persistence_model.model_threshold(bundle):g}"
                )
                st.caption(
                    f"Fitted {bundle.get('trained_at', '?')}, excluding sessions from "
                    f"{bundle.get('excluded_sessions_from', '?')} onwards. "
                    f"{bundle.get('notes', '')}"
                )
                st.json(
                    {"settings": bundle.get("settings"), "metrics": metrics},
                    expanded=False,
                )
    st.warning(
        ":material/warning: Those AUCs cover *all* regime changes, and roughly half of "
        "them are decided by one observable boolean — the old regime had already held 15 "
        "bars. On the changes that pass it the classifier scores ~0.50 and N-BEATS ~0.67, "
        "and the interval on that 0.67 only just excludes chance. Either momentum entry is "
        "best read as a change the model did not veto, which is why the exit does not "
        "consult it at all."
    )


def _render_dayrange_bundle(bundle: dict) -> None:
    """The day-range bundle's provenance, in its own units.

    Nothing here shares a scale with the momentum models: the error is dollars
    of misprediction on a price, not an AUC on a label, and there is no
    threshold at all.
    """
    metadata = bundle.get("metadata") or {}
    test = metadata.get("test_metrics_ensemble") or {}
    cols = st.columns(4)
    cols[0].metric("Blend", " + ".join(bundle.get("daily_models") or []))
    cols[1].metric("Built from", f"first {bundle.get('opening_minutes', 5)} min")
    cols[2].metric("Held-out error", f"${test.get('mae_usd_mean', float('nan')):.2f}")
    cols[3].metric(
        "vs 14-day baseline", f"{test.get('skill vs rolling 14d', float('nan')):.0%}"
    )
    st.caption(
        f"Fitted {bundle.get('trained_at', '?')} on daily bars through "
        f"{metadata.get('daily_fit_through', '?')}; the opening ridge on "
        f"{metadata.get('opening_fit_sessions', '?')} sessions after that, with "
        f"{metadata.get('held_out', '?')} held out of everything. The error is the mean "
        "absolute miss on the day's high and low over a 129-session test window — a "
        "number about prices, not about a label, so it cannot be compared to the AUCs "
        "above."
    )
    st.json(
        {"test_metrics": test, "constraint": metadata.get("constraint"),
         "opening_correction_gain": metadata.get("opening_correction_loo_gain")},
        expanded=False,
    )


def _render_momentum_change_bundle(bundle: dict) -> None:
    """The delta-momentum bundle's provenance, in its own units.

    Two things this panel is careful about. The headline numbers are the
    **holdout week's** -- five sessions removed before the estimator was fitted
    *or* chosen -- rather than the validation days the selection ran on, because
    the latter are only out-of-sample for fitting. And the R2 is on a bps/min
    quantity, so it shares no scale with the AUCs above or the dollars below.
    """
    metrics = bundle.get("metrics") or {}
    days = bundle.get("train_days") or []
    cols = st.columns(4)
    cols[0].metric("Estimator", bundle.get("model_name", "?"))
    cols[1].metric("Holdout R²", f"{metrics.get('holdout_r2', float('nan')):.2f}")
    cols[2].metric(
        "Sign on changes", f"{metrics.get('holdout_sign_hit_rate_on_changes', float('nan')):.0%}"
    )
    cols[3].metric(
        "MAE vs predict-zero",
        f"{metrics.get('holdout_mae', float('nan')):.2f}",
        delta=f"{metrics.get('holdout_mae', 0) - metrics.get('holdout_mae_predict_zero', 0):+.2f}",
        delta_color="inverse",
    )
    st.caption(
        f"{bundle.get('model_name', '?')} chosen on validation days and fitted "
        f"{bundle.get('saved_at', '?')} on {len(days)} training days "
        f"({days[0] if days else '?'} … {days[-1] if days else '?'}). The figures above "
        f"are from the {int(metrics.get('n_holdout_days', 0))} reserved sessions, used "
        "neither for fitting nor for selection. Target is Δ momentum in bps/min, so the "
        "R² is not comparable to the AUCs above or the dollar errors below. "
        f"{bundle.get('notes', '')}"
    )
    st.json({"metrics": metrics, "pipeline_params": bundle.get("pipeline_params")},
            expanded=False)


def render_agents_tab() -> None:
    keys = _testable_agents()
    personality = st.session_state.get("agents_selected", keys[0])
    if personality not in keys:
        personality = keys[0]
    cols = st.columns(len(keys))
    for col, key in zip(cols, keys):
        with col, st.container(border=True):
            avatar = _agent_avatar(key)
            if avatar.exists():
                st.image(str(avatar), width=72)
            st.caption(_agent_label(key))
            if st.button(
                "Selected" if key == personality else "Open",
                key=f"agent_pick_{key}",
                type="primary" if key == personality else "secondary",
            ):
                st.session_state["agents_selected"] = key
                st.rerun()

    if not sim_prompts.has_prompt(personality):
        _render_rule_agent(personality)
        return

    meta = AGENT_PERSONALITIES[personality]
    st.subheader(meta["label"])
    overridden = sim_prompts.has_override(personality)
    if overridden:
        st.caption(
            ":material/edit: Using a **modified** prompt (simulations launched here use it; "
            "the live app keeps the built-in)."
        )
    else:
        st.caption(":material/lock: Using the built-in prompt.")

    prompt_text = st.text_area(
        "System prompt",
        value=sim_prompts.get_prompt(personality),
        height=420,
        key=f"prompt_editor_{personality}",
    )
    with st.container(horizontal=True):
        if st.button("Save prompt", icon=":material/save:", type="primary"):
            sim_prompts.save_override(personality, prompt_text)
            st.rerun()
        if overridden and st.button("Reset to built-in", icon=":material/restart_alt:"):
            sim_prompts.reset_override(personality)
            st.rerun()

    st.divider()
    st.markdown(f"##### Tools ({len(PERSONALITY_TOOLS[personality])})")
    st.caption(
        "The exact tool set this agent gets in a cycle. In simulation each tool reads "
        "the stored tape as of the simulated moment."
    )
    _render_tool_tester(personality)


# ---------------------------------------------------------------------------
# Tab 2 — datasets
# ---------------------------------------------------------------------------

def render_datasets_tab() -> None:
    st.caption(
        "Datasets are named bundles of symbols + a date range + a **feed**. Minute bars "
        "(04:00–20:00 ET), daily history, news, and SPY/VIX context are stored locally, "
        "deduplicated per (feed, symbol, day) — overlapping datasets never re-download a day."
    )
    st.info(
        ":material/info: **The feed is part of the data, not a download setting.** "
        "`yfinance` (the default) is the consolidated tape — every venue — free, and the "
        "same source the live volume tools read. `iex` is one venue, roughly 4% of "
        "consolidated volume on a large cap, so its bars carry different closes, far "
        "smaller volumes and occasionally an extra or missing minute; any agent whose "
        "rules are thresholds over those bars can trade a different day on each tape. "
        "The same day on two feeds is stored twice, on purpose."
    )
    st.warning(
        f":material/schedule: **yfinance reaches back {sim_data.YF_MINUTE_WINDOW_DAYS} days.** "
        "Yahoo serves 1-minute history for the last "
        f"{sim_data.YF_MINUTE_WINDOW_DAYS} days only, so a yfinance dataset cannot start "
        f"before **{date.today() - timedelta(days=sim_data.YF_MINUTE_WINDOW_DAYS)}** — "
        "earlier days download empty. Use `sip` (needs a paid Alpaca data subscription) "
        "for an older window. yfinance bars also carry no per-bar VWAP."
    )
    with st.form("dataset_form"):
        name = st.text_input("Dataset name", placeholder="e.g. nvda-earnings-week")
        symbols_raw = st.text_input("Symbols (comma-separated)", placeholder="NVDA, AAPL")
        col_start, col_end, col_feed = st.columns(3)
        start = col_start.date_input("Start", value=date.today() - timedelta(days=7))
        end = col_end.date_input("End", value=date.today() - timedelta(days=1))
        feed = col_feed.selectbox("Feed", list(sim_data.FEEDS))
        col_key, col_secret = st.columns(2)
        api_key = col_key.text_input(
            "Alpaca API key", value=os.getenv("ALPACA_API_KEY", ""), type="password"
        )
        api_secret = col_secret.text_input(
            "Alpaca secret", value=os.getenv("ALPACA_SECRET", ""), type="password"
        )
        st.caption(
            "Alpaca credentials are required for the `iex`/`sip` feeds. On `yfinance` "
            "they are optional and fetch news only — without them the dataset has no news."
        )
        submitted = st.form_submit_button("Download dataset", icon=":material/download:", type="primary")

    if submitted:
        symbols = [s.strip().upper() for s in symbols_raw.split(",") if s.strip()]
        if not name.strip() or not symbols:
            st.error("A dataset needs a name and at least one symbol.")
        elif feed != "yfinance" and not (api_key and api_secret):
            st.error(f"Alpaca credentials are required for the `{feed}` feed.")
        else:
            with st.status(f"Downloading '{name}'…", expanded=True) as status:
                try:
                    ds = sim_data.create_dataset(
                        name.strip(), symbols, start, end, api_key, api_secret, feed,
                        progress=st.write,
                    )
                    status.update(
                        label=f"Dataset '{ds.name}' ready — {len(ds.days)} trading day(s)",
                        state="complete",
                    )
                except Exception as exc:
                    status.update(label=f"Download failed: {exc}", state="error")

    datasets = sim_data.list_datasets()
    st.divider()
    if not datasets:
        st.info("No datasets yet.")
        return
    size_mb = sim_data.store_size_bytes() / 1e6
    st.markdown(f"##### Stored datasets — shared store {size_mb:.1f} MB")
    for ds in datasets:
        with st.container(border=True, horizontal=True, vertical_alignment="center"):
            st.markdown(
                f"**{ds.name}** — {', '.join(ds.symbols)} · {ds.start} → {ds.end} "
                f"· {len(ds.days)} trading day(s) · `{ds.feed}`"
            )
            if st.button("Delete", key=f"del_ds_{ds.name}", icon=":material/delete:"):
                sim_data.delete_dataset(ds.name)
                st.rerun()


# ---------------------------------------------------------------------------
# Tab 3 — simulate
# ---------------------------------------------------------------------------

def _equity_chart(equity: list[dict], starting_cash: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[p["ts"] for p in equity],
            y=[p["value"] for p in equity],
            mode="lines",
            name="portfolio value",
            line=dict(color=PALETTE["accent"], width=2),
        )
    )
    fig.add_hline(y=starting_cash, line_dash="dot", line_color=PALETTE["muted"])
    return _chart_layout(fig, height=300)


# The hover card's shape. Wide enough that a sentence is not shredded, short
# enough that the label does not cover the candles it explains.
_HOVER_WIDTH = 64
_HOVER_MAX_LINES = 12


def _decision_hover(decision: dict) -> str:
    """One fill's hover card: what was traded, and why.

    The "why" is the reason the trader wrote at decision time, which already
    names the condition that fired -- a trailing stop names the give-back it
    passed, a model entry names the probability and the threshold it cleared,
    an Apple Trader 2 fill names the rule and every condition with the value it
    held. None of that was reachable from the chart before; it was in the
    Decisions table, several scrolls from the marker that raised the question.

    `results.decision_trigger` supplies the headline above it, so the kind of
    event is legible before the sentence is read. An unrecognised reason simply
    has no headline -- the prose is what matters and it is always shown.
    """
    action = str(decision.get("action") or "").upper()
    quantity = decision.get("filled_quantity") or decision.get("requested_quantity") or 0
    # Fills are fractional and the fraction is not noise -- it is what the
    # position-sizing rule chose -- so the trailing zeros go and nothing else.
    quantity_text = f"{float(quantity):,.4f}".rstrip("0").rstrip(".") or "0"
    price = decision.get("price")
    price_text = f"${float(price):,.2f}" if price is not None else "—"
    # The axis is in the frame the stored timestamps are in, so the hover reads
    # the same clock as the candle under it.
    stamp = str(decision.get("ts") or "").replace("T", " ")[:16]

    lines = [
        f"<b>{escape(action)}</b> {escape(quantity_text)} sh @ {escape(price_text)}",
        f"<span style='font-size:11px'>{escape(stamp)}</span>",
    ]
    trigger = sim_results.decision_trigger(decision)
    if trigger:
        lines.append(f"<b>{escape(trigger)}</b>")
    reason = str(decision.get("reasoning") or "").strip()
    if reason:
        # Wrapped rather than left to the browser: plotly sizes a hover label
        # to its longest line, and a 400-character reason on one line produces
        # a tooltip wider than the chart.
        wrapped = textwrap.wrap(reason, width=_HOVER_WIDTH)
        # An LLM agent can write a dozen paragraphs of justification, and a
        # tooltip taller than the plot covers the tape it is explaining. The
        # rule agents never come close to this; when it does bite, the
        # Decisions table below the chart still holds the whole thing.
        if len(wrapped) > _HOVER_MAX_LINES:
            wrapped = wrapped[:_HOVER_MAX_LINES] + ["… (full reason in Decisions below)"]
        lines.append("<br>".join(escape(line) for line in wrapped))
    else:
        lines.append("<i>No reason recorded.</i>")
    return "<br>".join(lines)


def _price_chart(
    symbol: str,
    bars: list[dict],
    decisions: list[dict],
    overlays: "list[dict] | None" = None,
) -> go.Figure:
    """The replayed day(s)' candles, the fills, and what the models predicted.

    `overlays` are `model_overlays.compute` items, drawn by the same renderer
    the live chart uses. There is no price-profile column here, so the
    predicted levels are drawn once instead of mirrored -- everything else is
    identical, which is the point of the items being data rather than plotly
    calls.

    A run usually covers several days, and the exchange is shut for two thirds
    of each one. Those stretches are removed from the time axis rather than
    drawn as blank space (`charts.session_rangebreaks`), and the day boundary
    they used to provide is put back as a rule at 09:30 and 16:00
    (`charts.add_session_markers`).
    """
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=[b["t"] for b in bars],
            open=[b["o"] for b in bars],
            high=[b["h"] for b in bars],
            low=[b["l"] for b in bars],
            close=[b["c"] for b in bars],
            name=symbol,
            increasing_line_color=PALETTE["up"],
            decreasing_line_color=PALETTE["down"],
        )
    )
    for action, color, symbol_marker in (("buy", PALETTE["up"], "triangle-up"), ("sell", PALETTE["down"], "triangle-down")):
        fills = [
            d for d in decisions
            if d.get("symbol") == symbol and d.get("action") == action and d.get("status") == "filled"
        ]
        if fills:
            fig.add_trace(
                go.Scatter(
                    x=[d["ts"] for d in fills],
                    y=[d["price"] for d in fills],
                    mode="markers",
                    name=action,
                    marker=dict(color=color, size=13, symbol=symbol_marker,
                                line=dict(width=1, color=PALETTE["text"])),
                    hovertext=[_decision_hover(d) for d in fills],
                    # `<extra></extra>` drops plotly's trace-name box, which
                    # would repeat "buy" beside a card that already says it.
                    hovertemplate="%{hovertext}<extra></extra>",
                    hoverlabel=dict(
                        align="left",
                        bgcolor=PALETTE["panel"],
                        bordercolor=color,
                        font=dict(color=PALETTE["text"], size=12),
                    ),
                )
            )
    if bars:
        add_session_markers(fig, bars)
        fig.update_xaxes(rangebreaks=session_rangebreaks(bars))
    if overlays and bars:
        x0, x1 = pd.Timestamp(bars[0]["t"]), pd.Timestamp(bars[-1]["t"])
        add_model_overlays(overlays, fig, x0, x1, row=None, col=None)
        fig.update_xaxes(range=[x0, overlay_x_max(overlays, x1)])
    fig.update_layout(xaxis_rangeslider_visible=False)
    return _chart_layout(fig, height=420)



def _run_overlay_controls(
    record: dict, market: "SimMarket", symbol: str, days: "list[date]"
) -> dict:
    """Pick which model predictions to draw over a replayed day, and compute them.

    Deliberately available on *every* run, not only the ones a model drove: the
    interesting question in Results is usually what a model would have said
    about a day, and an LLM agent's tape is as good a place to ask it as a rule
    agent's. Nothing here reads the run's own decisions.

    Each day is scored on its own, from the state of the world at its 9:31 --
    completed daily bars strictly before it (`SimMarket.completed_daily_bars`)
    and the stored opening print -- which is the same point-in-time view the
    agent had. A replay chart that showed a forecast built on the day's own
    outcome would be worse than no forecast at all.
    """
    available = model_overlays.keys_for(symbol)
    if not available:
        return {"items": [], "notes": []}

    run_id = record.get("run_id") or "run"
    selected = st.multiselect(
        "Model predictions",
        available,
        format_func=model_overlays.label,
        key=f"sim_overlays_{run_id}_{symbol}",
        help="What the trained models predicted for this session, drawn over "
        "the replayed tape: predicted ranges as horizontal lines, momentum "
        "changes as marked moments, time-spanning predictions as a shaded "
        "background.",
    )
    if not selected:
        return {"items": [], "notes": []}

    momentum_model = None
    if model_overlays.MOMENTUM_KEY in selected:
        momentum_keys = [k for k in apple_models.keys() if apple_models.is_momentum(k)]
        momentum_model = st.selectbox(
            "Momentum model",
            momentum_keys,
            format_func=lambda key: apple_models.get(key).label,
            key=f"sim_overlay_model_{run_id}_{symbol}",
        )

    items: list[dict] = []
    notes: list[str] = []
    bars = market.series[symbol].minute_bars
    for day in days:
        t = market.session_open(day) + timedelta(minutes=1)
        result = model_overlays.compute(
            selected,
            symbol,
            bars,
            daily_bars=market.completed_daily_bars(symbol, t),
            session_date=day,
            open_price=market.session_open_price(symbol, t),
            momentum_model=momentum_model,
        )
        items.extend(result["items"])
        for note in result["notes"]:
            entry = f"{day}: {note}" if len(days) > 1 else note
            if entry not in notes:
                notes.append(entry)
    return {"items": items, "notes": notes}


def _render_judge_report(judge_report: dict) -> None:
    st.markdown("##### :material/gavel: LLM judge")
    if judge_report.get("judge_model"):
        st.caption(f":material/robot_2: Judged by {judge_report['judge_model']}")
    cols = st.columns(3)
    cols[0].metric("Overall score", f"{judge_report.get('overall_score', '—')}/10")
    cols[1].metric("Strategy adherence", f"{judge_report.get('strategy_adherence', '—')}/10")
    avg = judge_report.get("avg_entry_score")
    cols[2].metric("Avg entry score", f"{avg}/10" if avg is not None else "—")
    if judge_report.get("summary"):
        st.write(judge_report["summary"])
    for item in judge_report.get("top_improvements") or []:
        st.markdown(f"- {item}")
    if judge_report.get("error"):
        st.warning(f"Overall judgment failed: {judge_report['error']}")
    for entry in judge_report.get("entries", []):
        header = f"{entry.get('ts', '')[:16]} · {entry.get('symbol')} @ {entry.get('price')}"
        score = entry.get("score")
        label = f"{header} — {score}/10 ({entry.get('verdict', '?')})" if score is not None else header
        with st.expander(label):
            if entry.get("error"):
                st.warning(entry["error"])
                continue
            st.markdown(f"**Reasoning quality:** {entry.get('reasoning_quality')}")
            st.markdown(f"**What went well:** {entry.get('what_went_well')}")
            st.markdown(f"**To improve:** {entry.get('what_to_improve')}")


def _render_run(record: dict) -> None:
    summary = record.get("summary") or {}
    config = record.get("config_summary") or {}
    if record.get("error"):
        st.error(f"Simulation ended with an error (partial results below): {record['error']}")

    cols = st.columns(6)
    profit = summary.get("profit", 0.0)
    cols[0].metric("Final value", f"${summary.get('final_value', 0):,.0f}",
                   delta=f"{summary.get('return_pct', 0):+.2f}%")
    cols[1].metric("Profit", f"${profit:,.2f}")
    cols[2].metric("Oracle ceiling", f"{summary.get('oracle_ceiling_pct', 0):.2f}%",
                   help="Best single round trip an oracle could have made on this tape.")
    eff = summary.get("profit_efficiency")
    cols[3].metric("Profit efficiency", f"{eff:.1%}" if eff is not None else "—",
                   help="Session return ÷ oracle ceiling.")
    cols[4].metric("Trades filled", summary.get("trades_filled", 0))
    rule_based = bool(config.get("rule_based"))
    cols[5].metric(
        "Bars scored" if rule_based else "LLM cycles",
        summary.get("cycles_run", record.get("cycles_run", 0)),
    )
    if rule_based:
        st.caption(
            f":material/function: Rule-based run — {_agent_label(config.get('personality'))}, "
            f"no LLM and no judge. Rules: `{config.get('model')}`"
        )
    # The tape is part of what produced these numbers: identical rules on
    # `yfinance`, `iex` and `sip` are different bars and can be different
    # trades. A record from before the feed was tracked ran on `iex`.
    feed = config.get("feed") or sim_data.LEGACY_FEED
    st.caption(f":material/database: Tape: `{feed}`")

    equity = record.get("equity") or []
    if equity:
        st.plotly_chart(_equity_chart(equity, record.get("starting_cash", 0.0)))

    symbols = config.get("symbols") or []
    days = [date.fromisoformat(d) for d in config.get("days") or []]
    decisions = record.get("decisions") or []
    if symbols and days:
        try:
            market = SimMarket(symbols, days, feed)
            tabs = st.tabs(symbols)
            for tab, sym in zip(tabs, symbols):
                with tab:
                    bars = market.series[sym].minute_bars
                    if bars:
                        overlays = _run_overlay_controls(record, market, sym, days)
                        st.plotly_chart(
                            _price_chart(sym, bars, decisions, overlays["items"])
                        )
                        for note in overlays["notes"]:
                            st.caption(f":material/info: {note}")
                    else:
                        st.info("No stored bars for this symbol/day.")
        except Exception as exc:
            st.caption(f"Price charts unavailable ({exc}).")

    if record.get("judge"):
        _render_judge_report(record["judge"])

    with st.expander(f"Decisions ({len(decisions)})"):
        st.dataframe(
            [
                {k: d.get(k) for k in ("ts", "symbol", "action", "status", "price",
                                       "filled_quantity", "cash_after", "reasoning")}
                for d in decisions
            ],
            height=300,
        )
    log = record.get("agent_log") or []
    with st.expander(f"Agent log ({len(log)} entries)"):
        for entry in log[-400:]:
            ts = str(entry.get("ts", ""))[11:19]
            kind = entry.get("type", "")
            text = entry.get("text") or entry.get("reasoning") or entry.get("name") or ""
            st.markdown(f"`{ts}` **{kind}** {text}")


# The pipeline: queued experiments, a parallelism limit, and worker processes.
# One process = one simulation (module-global sim clock), so parallel runs are
# separate `python -m simlab.runner` subprocesses scheduled by
# `sim_experiments.tick()`; this section is both the status bar and the tick.

MAX_PARALLEL_KEY = "pipeline_max_parallel"

_STATUS_MARKUP = {
    sim_experiments.WAITING: ":orange[:material/hourglass_top: waiting]",
    sim_experiments.RUNNING: ":blue[:material/sync: running]",
    sim_experiments.FINISHED: ":green[:material/check_circle: finished]",
    sim_experiments.FAILED: ":red[:material/error: failed]",
}


def _experiment_label(exp: dict) -> str:
    config = exp.get("config") or {}
    personality = config.get("personality") or "?"
    agent = _agent_label(personality)
    return (
        f"**{agent}** · {config.get('provider')}/{config.get('model')} · "
        f"{exp.get('dataset')} · {len(config.get('days') or [])} day(s)"
    )


def _past_experiment_rows(past: list[dict]) -> list[dict]:
    rows = []
    for exp in past:
        config = exp.get("config") or {}
        result = exp.get("result_summary") or {}
        personality = config.get("personality") or "?"
        rows.append({
            "queued": (exp.get("created_at") or "")[:16].replace("T", " "),
            "agent": _agent_label(personality),
            "model": f"{config.get('provider')}/{config.get('model')}",
            "dataset": exp.get("dataset"),
            "days": len(config.get("days") or []),
            "status": exp.get("status"),
            "return_pct": result.get("return_pct"),
            "profit_efficiency": result.get("profit_efficiency"),
            "judge": result.get("judge_overall"),
            "run_id": exp.get("run_id"),
            "error": exp.get("error"),
        })
    return rows


def _render_pipeline_body(auto_refresh: bool) -> None:
    col_head, col_workers = st.columns([4, 1], vertical_alignment="bottom")
    col_head.markdown("##### :material/stacks: Experiment pipeline")
    max_parallel = col_workers.number_input(
        "Parallel workers", min_value=1, max_value=8, value=2, key=MAX_PARALLEL_KEY,
        help="Experiments beyond this limit wait in the queue. Each experiment "
             "runs in its own worker process.",
    )
    sim_experiments.tick(max_parallel)
    exps = sim_experiments.list_experiments()
    active = [e for e in exps if e["status"] in sim_experiments.ACTIVE_STATUSES]
    if auto_refresh and not active:
        st.rerun()  # pipeline just drained -- refresh the whole app once

    counts = {
        status: sum(1 for e in exps if e["status"] == status)
        for status in (sim_experiments.WAITING, sim_experiments.RUNNING,
                       sim_experiments.FINISHED, sim_experiments.FAILED)
    }
    cols = st.columns(4)
    cols[0].metric("Waiting", counts[sim_experiments.WAITING])
    cols[1].metric("Running", counts[sim_experiments.RUNNING])
    cols[2].metric("Finished", counts[sim_experiments.FINISHED])
    cols[3].metric("Failed", counts[sim_experiments.FAILED])

    for exp in active:
        with st.container(border=True, horizontal=True, vertical_alignment="center"):
            st.markdown(_experiment_label(exp))
            st.markdown(_STATUS_MARKUP[exp["status"]])
            if exp["status"] == sim_experiments.RUNNING:
                line = sim_experiments.last_log_line(exp["experiment_id"])
                if line:
                    st.caption(line)
                if st.button(
                    "Stop", key=f"exp_stop_{exp['experiment_id']}",
                    icon=":material/stop_circle:",
                    help="Kill the worker now. The cycles run so far are lost — "
                         "no run record is saved.",
                ):
                    sim_experiments.stop(exp["experiment_id"])
                    st.toast("Experiment stopped", icon=":material/stop_circle:")
                    st.rerun()
            elif st.button(
                "Remove", key=f"exp_rm_{exp['experiment_id']}", icon=":material/close:"
            ):
                sim_experiments.delete_experiment(exp["experiment_id"])
                st.rerun()

    past = [e for e in exps
            if e["status"] in (sim_experiments.FINISHED, sim_experiments.FAILED)]
    if past:
        with st.expander(f"Past experiments ({len(past)})"):
            st.dataframe(
                pd.DataFrame(_past_experiment_rows(past)),
                hide_index=True,
                column_config={
                    "queued": "Queued (UTC)",
                    "agent": "Agent",
                    "model": "Model",
                    "dataset": "Dataset",
                    "days": "Days",
                    "status": "Status",
                    "return_pct": st.column_config.NumberColumn("Return", format="%+.2f%%"),
                    "profit_efficiency": st.column_config.NumberColumn(
                        "Profit eff.", format="percent"),
                    "judge": st.column_config.NumberColumn("Judge", format="%.1f"),
                    "run_id": "Run",
                    "error": "Error",
                },
            )
            if st.button("Clear history", icon=":material/delete_sweep:",
                         help="Drops the experiment records; stored runs are kept."):
                sim_experiments.clear_finished()
                st.rerun()


def _render_pipeline() -> None:
    # While experiments are active the section refreshes itself (which also
    # ticks the scheduler); once idle it renders statically until the next
    # full rerun.
    auto_refresh = sim_experiments.has_active()
    st.fragment(run_every=2.5 if auto_refresh else None)(
        lambda: _render_pipeline_body(auto_refresh)
    )()


def _prior_run_line(record: dict, selected_days: list[str]) -> str:
    summary = record.get("summary") or {}
    judge = record.get("judge") or {}
    run_days = (record.get("config_summary") or {}).get("days") or []
    parts = [
        f"`{record['run_id']}`",
        ", ".join(run_days) or "—",
        f"{summary.get('return_pct', 0):+.2f}%",
    ]
    eff = summary.get("profit_efficiency")
    if eff is not None:
        parts.append(f"eff {eff:.1%}")
    if judge.get("overall_score") is not None:
        parts.append(f"judge {judge['overall_score']}/10")
    line = " · ".join(parts)
    if selected_days and set(run_days) == set(selected_days):
        line += " — :orange[**same trading day(s)**]"
    return line


def _combo_label(combo: tuple[str, str, str, str]) -> str:
    personality, provider, model, dataset = combo
    return (
        f"**{_agent_label(personality)}** · "
        f"{provider}/{model} · {dataset}"
    )


def _render_already_tested(
    combos: list[tuple[str, str, str, str]], days_by_dataset: dict[str, list[str]]
) -> set[tuple[str, str, str, str]]:
    """Flag agent/model/dataset combinations that are already queued or have
    already been tested cleanly, so the same experiment isn't paid for twice.
    Runs whose cycles hit LLM errors don't count as tested -- those are exactly
    the ones worth running again. Returns the cleanly tested combinations."""
    pending: dict[tuple[str, str, str, str], list[dict]] = {}
    for exp in sim_experiments.list_experiments():
        if exp["status"] not in sim_experiments.ACTIVE_STATUSES:
            continue
        config = exp.get("config") or {}
        key = (config.get("personality"), config.get("provider"),
               config.get("model"), exp.get("dataset"))
        pending.setdefault(key, []).append(exp)

    runs = _runs()
    queued_lines, tested, tested_lines, degraded_lines = [], set(), [], []
    for combo in combos:
        personality, provider, model, dataset = combo
        if combo in pending:
            statuses = ", ".join(sorted({e["status"] for e in pending[combo]}))
            queued_lines.append(
                f"- {_combo_label(combo)} — {len(pending[combo])} experiment(s): {statuses}"
            )
        prior = sim_results.find_prior_runs(runs, personality, provider, model, dataset)
        clean, degraded = prior["clean"], prior["degraded"]
        if clean:
            tested.add(combo)
            days = days_by_dataset.get(dataset) or []
            runs_text = "\n".join(
                f"    - {_prior_run_line(r, days)}" for r in clean[:5]
            )
            more = f"\n    - …and {len(clean) - 5} more" if len(clean) > 5 else ""
            tested_lines.append(
                f"- {_combo_label(combo)} — {len(clean)} clean run(s)\n{runs_text}{more}"
            )
        elif degraded:
            errors = sum(sim_results.cycle_error_count(r) for r in degraded)
            degraded_lines.append(
                f"- {_combo_label(combo)} — run {len(degraded)} time(s) before, but "
                f"{errors} cycle(s) failed (LLM errors)"
            )

    if queued_lines:
        st.warning(
            f":material/schedule: {len(queued_lines)} combination(s) already in the "
            "pipeline:\n" + "\n".join(queued_lines)
        )
    if tested_lines:
        st.warning(
            f":material/history: {len(tested_lines)} combination(s) have **already been "
            "tested** cleanly — results are in the Results tab. Re-run only if you "
            "changed the prompt or the settings below.\n" + "\n".join(tested_lines)
        )
    if degraded_lines:
        st.info(
            ":material/error_outline: These combinations ran before but hit LLM errors, "
            "so they aren't a clean test:\n" + "\n".join(degraded_lines)
        )
    return tested


def _configured_rule_tickers(personalities: list[str]) -> dict[str, list[str]]:
    """The symbol each selected rule agent's setups currently trade.

    Read out of `st.session_state` rather than from the rendered setups,
    because the dataset scope is drawn *above* them and needs the answer first.
    That is safe here in a way it would not be in a plain script: Streamlit
    reruns the whole page on every widget change, and a widget's stored value
    is already the new one by the time the rerun starts -- so this is the
    current selection, not the previous one. `_render_agent_setups` leans on
    the same property for the signature it shows on a collapsed setup.

    A slot whose instrument widget has never rendered has no stored value; the
    agent's `default_ticker` is what that widget will come up with, so it is
    what this reports.
    """
    tickers: dict[str, list[str]] = {}
    for personality in personalities:
        agent = RULE_AGENTS.get(personality)
        if agent is None:
            continue
        found = []
        for slot in _rule_slots(personality):
            stored = st.session_state.get(f"sim_rule_{personality}_{slot}_ticker")
            symbol = str(stored or agent.default_ticker).strip().upper()
            if symbol and symbol not in found:
                found.append(symbol)
        tickers[personality] = found
    return tickers


def _render_dataset_scope(
    datasets: list, rule_tickers: dict[str, list[str]], has_llm: bool
) -> dict[str, dict]:
    """Per-dataset trading days and symbols -- each selected dataset carries its
    own, since days and symbols differ from dataset to dataset.

    The symbols are chosen for the user rather than by them, because which
    instruments a batch trades is already decided elsewhere on this page:

    * a **rule agent** names its own instrument in its setup and is
      single-symbol by design, so its runs are queued with exactly that symbol
      and there is nothing here to pick. Ticking a symbol never made one of
      those runs trade it, and forgetting to tick the right one made the run
      fail in the engine -- which is the manual step this replaces.
    * an **LLM agent** is handed the whole basket and decides for itself, so
      *which* basket is a real experimental choice and stays editable. It
      defaults to everything the dataset carries.

    With no LLM agent selected the multiselect disappears and the derived
    symbols are shown instead, so the page still says what will be traded.
    """
    traded = sorted({t for tickers in rule_tickers.values() for t in tickers})
    scope: dict[str, dict] = {}
    for ds in datasets:
        with st.container(border=True):
            st.markdown(f"**{ds.name}** — {ds.start} → {ds.end} · tape `{ds.feed}`")
            col_days, col_symbols = st.columns(2)
            day_options = ds.days or []
            days = col_days.multiselect(
                "Trading day(s)", day_options, default=day_options[:1],
                key=f"sim_days_{ds.name}",
            )
            if has_llm:
                symbols = col_symbols.multiselect(
                    "Symbols (LLM basket)", ds.symbols, default=ds.symbols,
                    key=f"sim_syms_{ds.name}",
                    help="The basket handed to the LLM agents, which trade across all "
                         "of it. Rule agents ignore this — each is queued with the one "
                         "symbol its setup names.",
                )
            else:
                # Nothing reads a basket here, so there is no widget to get
                # wrong. What the rule setups name is the whole answer.
                symbols = [s for s in ds.symbols if s in traded]
                with col_symbols:
                    st.markdown("**Symbols**")
                    if symbols:
                        st.markdown(
                            " ".join(f"`{s}`" for s in symbols)
                            + f"<span style='color:{PALETTE['muted']}'> — selected "
                            "automatically from the agent setups below.</span>",
                            unsafe_allow_html=True,
                        )
                    elif not traded:
                        # No agent picked yet. Saying the setups are wrong
                        # would be describing a mistake nobody has made.
                        st.caption(
                            ":material/info: Chosen from the agent setups below — "
                            "pick an agent to fill this in."
                        )
                    else:
                        st.caption(
                            ":material/warning: This dataset carries "
                            f"{', '.join(ds.symbols) or 'nothing'}, none of which "
                            f"the selected setups trade ({', '.join(traded)})."
                        )
            scope[ds.name] = {"days": days, "symbols": symbols}
    return scope


def _render_model_picker() -> tuple[list[tuple[str, str]], dict[str, str]]:
    """Models to test, across providers: returns [(provider, model), …] and the
    API key per selected provider."""
    providers = st.pills(
        "Providers", list(PROVIDERS), selection_mode="multi",
        default=[PROVIDERS[0]], key="sim_providers",
    ) or []
    model_choices: list[tuple[str, str]] = []
    api_keys: dict[str, str] = {}
    for provider in providers:
        with st.container(border=True):
            col_models, col_key = st.columns([2, 1])
            picked = col_models.multiselect(
                f"{provider} models",
                models_for(provider, default=DEFAULT_AGENT_MODELS[provider]),
                default=[DEFAULT_AGENT_MODELS[provider]],
                key=f"sim_models_{provider}",
            )
            api_keys[provider] = col_key.text_input(
                f"{provider} API key", value=_env_key(provider), type="password",
                key=f"sim_key_{provider}",
            )
            model_choices += [(provider, model) for model in picked]
    return model_choices, api_keys


def _rule_combinations(
    rule_setups: dict, dataset_names: list[str]
) -> "tuple[list[tuple], dict[tuple, object]]":
    """(combinations, combination -> the settings that produced it).

    A rule agent has no model dimension, so its own settings stand in for one:
    each setup is queued against each dataset as its own experiment, and lands
    in Results as its own configuration. That is what makes a threshold sweep
    or a two-instrument comparison one batch rather than several trips through
    this tab.

    The combination is keyed on the agent's **signature**, which is what makes
    two setups that sign the same collapse into one experiment rather than
    running the same thing twice. Signature rather than settings on purpose:
    the signature is the identity the whole pipeline uses -- it is the run
    record's `model` field, what `_render_already_tested` matches on and what
    Results groups by -- so two setups sharing one would produce two runs shown
    as a single row whatever this did. Not every field is in it (neither
    agent's carries the closing flatten), so setups can differ on screen and
    still be one configuration; `_render_agent_setups` says so when they do.
    """
    combos: list[tuple] = []
    by_combo: dict[tuple, object] = {}
    for name in dataset_names:
        for personality, configs in rule_setups.items():
            agent = rule_agent(personality)
            for config in configs:
                combo = (personality, RULE_PROVIDER, agent.signature(config), name)
                if combo not in by_combo:
                    combos.append(combo)
                by_combo[combo] = config
    return combos, by_combo


def _experiment_symbols(
    personality: str, rule_settings, basket: list[str]
) -> list[str]:
    """The symbols one queued experiment actually needs.

    A rule agent is single-symbol by design -- its setup names the instrument
    and the engine refuses to build it against any other -- so the run needs
    that symbol and nothing else. Handing it the rest of the basket loaded bars
    no rule would ever read, and made the symbol boxes a manual step whose only
    possible outcomes were "the same run" and "a run that fails in the engine".

    An LLM agent is the opposite: it is told what it is holding and picks
    across it, so the basket *is* the configuration and stays the user's.

    One consequence beyond the tab, worth knowing before comparing numbers:
    `results.summarize_run` takes the oracle ceiling as the *best* round trip
    over every symbol in the market, so a rule run against a seven-symbol
    basket was divided by whichever of the seven moved most -- a ceiling it
    could not have reached, since it only ever traded one of them. Its
    `profit_efficiency` is now measured against its own instrument, which is
    the honest denominator and **not comparable with rule runs stored before
    this change**. Nothing rewrites those records; they are what they were.
    """
    if personality in RULE_AGENTS:
        return [rule_agent(personality).ticker(rule_settings)]
    return list(basket)


def _rule_agents_missing_ticker(
    rule_setups: dict, datasets_by_name: dict, selected_names: list[str]
) -> list[tuple[str, str, list[str]]]:
    """(agent, symbol, datasets that do not carry it) for every setup that
    cannot be replayed.

    A dataset without the symbol is not a strategy result, it is a run that
    cannot trade, so it is worth catching before the experiments are queued.
    Grouped by *symbol* rather than by agent, because one agent can now be
    queued several times over different instruments and only some of them may
    be missing.

    Checked against what a dataset **carries**, not against what is ticked on
    the page. A rule run is queued with its own symbol whatever the basket
    selection says, so the only thing that can defeat it is the dataset not
    holding those bars -- and that is a download to redo, not a box to tick.
    """
    missing: list[tuple[str, str, list[str]]] = []
    for personality, configs in rule_setups.items():
        agent = rule_agent(personality)
        for ticker in dict.fromkeys(agent.ticker(config) for config in configs):
            names = [
                name for name in selected_names
                if ticker not in (datasets_by_name[name].symbols or [])
            ]
            if names:
                missing.append((personality, ticker, names))
    return missing


# Which rule-agent setups the Simulate tab is currently holding, per agent:
# ``{personality: [slot_id, ...]}``. A slot is an opaque id that owns one set of
# widget keys, so removing the second of three setups leaves the other two's
# inputs exactly where they were -- which indexing by position would not.
_RULE_SLOTS_KEY = "sim_rule_slots"


def _rule_slots(personality: str) -> list[str]:
    """This agent's setup slots, creating the first one on demand."""
    slots = st.session_state.setdefault(_RULE_SLOTS_KEY, {})
    if not slots.get(personality):
        slots[personality] = [uuid.uuid4().hex[:8]]
    return slots[personality]


def _add_rule_slot(personality: str) -> None:
    _rule_slots(personality).append(uuid.uuid4().hex[:8])


def _drop_rule_slot(personality: str, slot: str) -> None:
    slots = _rule_slots(personality)
    if len(slots) > 1 and slot in slots:
        slots.remove(slot)


def _render_rule_params(
    personalities: list[str], symbols: list[str]
) -> dict[str, list]:
    """Every rule-agent setup this batch will queue, per agent.

    A *list* per agent rather than one configuration, because a rule agent's
    settings are what an LLM agent's model is: the thing under test. One prompt
    applies to every model in the grid, but a rule set does not apply to
    anything -- it *is* the entry -- so sweeping a threshold or comparing two
    instruments should be one batch rather than three trips through this tab.
    Each setup queues its own experiment per dataset, and lands in Results as
    its own configuration, since the signature is a function of the settings.

    `symbols` is every symbol the selected datasets carry, offered to the agents
    that pick their own instrument.
    """
    renderers = {
        APPLE_TRADER_KEY: _render_apple_params,
        APPLE_TRADER2_KEY: _render_apple2_params,
    }
    return {
        key: _render_agent_setups(key, symbols, renderers[key])
        for key in personalities
        if key in renderers
    }


def _render_agent_setups(personality: str, symbols: list[str], renderer) -> list:
    """One agent's setups: an editor each, plus add and remove.

    The newest setup is the one left open and the older ones collapse to their
    **signature** -- the same string Results groups runs by, so a glance at the
    collapsed titles answers the only question that matters here, which is
    whether these are actually different configurations. The signature shown is
    the one built on the previous rerun (an expander's label is fixed before its
    body runs); every widget change reruns the page, so it trails an edit by
    nothing a user can perceive.
    """
    slots = _rule_slots(personality)
    label = _agent_label(personality)
    st.markdown(f"**{label}** — {len(slots)} setup(s)")
    configs = []
    for index, slot in enumerate(slots):
        prefix = f"sim_rule_{personality}_{slot}"
        title = f"Setup {index + 1}"
        remembered = st.session_state.get(f"{prefix}__signature")
        if remembered and index != len(slots) - 1:
            title += f" — {remembered}"
        with st.expander(title, expanded=index == len(slots) - 1):
            config = renderer(symbols, prefix)
            configs.append(config)
            signature = rule_agent(personality).signature(config)
            st.session_state[f"{prefix}__signature"] = signature
            with st.container(horizontal=True, vertical_alignment="center"):
                st.button(
                    "Remove this setup", icon=":material/delete:",
                    key=f"{prefix}_remove", disabled=len(slots) == 1,
                    on_click=_drop_rule_slot, args=(personality, slot),
                    help=None if len(slots) > 1
                    else "The last setup cannot be removed — deselect the agent instead.",
                )
                st.caption(f"`{signature}`")
    st.button(
        f"Add another {label} setup", icon=":material/add:",
        key=f"sim_rule_add_{personality}", on_click=_add_rule_slot, args=(personality,),
        help="Queues a second configuration of the same agent over the same "
             "datasets — a threshold sweep, two instruments, or one rule switched "
             "on and off, run side by side and compared in Results.",
    )
    duplicates = len(configs) - len({
        rule_agent(personality).signature(config) for config in configs
    })
    if duplicates:
        # Deliberately about the *signature* rather than about the settings:
        # not every field is in it (the closing flatten is not, on either
        # agent), so two setups can differ visibly here and still be one
        # configuration everywhere downstream.
        st.caption(
            f":material/info: {duplicates} setup(s) sign the same as another and will "
            "be queued once. Results identifies a configuration by its signature, so "
            "two runs of one would be a single row — to test them apart, change "
            "something the signature carries (it is shown under each setup)."
        )
    return configs


def _render_apple2_params(symbols: list[str], prefix: str) -> AppleTrader2Config:
    """One Apple Trader 2 setup.

    The same builder the live dashboard renders, under its own widget prefix --
    the two apps run in separate processes, but the prefix is what keeps a rule
    set edited here from being confused with one edited there if they ever do
    not.

    There is no model picker: which bundles a run loads falls out of which
    signals the rules read, so a rule set written on price and momentum queues
    without needing any saved artifact at all. There *is* an instrument picker,
    seeded with the symbols the selected datasets carry -- a run reads one
    symbol's bars and a dataset without it cannot be replayed, which is what
    `_rule_agents_missing_ticker` checks before anything is queued.
    """
    st.caption(
        "One list of buy/sell rules, checked in order on every closed minute bar. "
        "Each rule is an action, a size and the conditions that arm it, joined by "
        "AND or OR; the first rule that matches *and* can transact takes the bar. "
        "The instrument and the rule set together are the configuration Results "
        "groups these runs by, so moving one number queues a new configuration to "
        "compare rather than a repeat."
    )
    return rules_panel(prefix, symbols=symbols)


# SimLab's half of the Apple Trader form (see `agent_stonks.apple_trader_ui`).
# Its wording is about what is worth sweeping and how a setting lands in
# Results; the dashboard's is about the run that is about to start.
_APPLE_TRADER_COPY_FIELDS = dict(
    unavailable_suffix="not in the datasets",
    instrument_help=(
        "The one symbol the run trades. Only symbols a saved model covers are "
        "listed — the models are the strategy here, so the instrument and the "
        "model constrain each other. The same rules over two symbols are two "
        "configurations in Results, never one averaged row."
    ),
    model_help=(
        "On the confirm entry the two momentum models are handed the same "
        "20 bars on the same tape and return one probability, so running a "
        "dataset through both is a straight comparison. Neither the day-range "
        "forecast nor the delta-momentum regressor is on that scale, and "
        "neither is comparable to them by number — each is a different "
        "strategy on the same symbol, and the way to compare them is to run "
        "the same dataset through each. Only the models fitted on the "
        "instrument above are listed."
    ),
    intro={
        "dayrange": (
            "Two knobs, and they are the whole strategy. At 9:35 the model forecasts where "
            "today's high **H** will land; the buy rests `buy × ADR` below it and the sell "
            "`sell × ADR` below it, with ADR the trailing 14-day average daily range in "
            "dollars. Each distinct pair is its own configuration in Results, so sweeping "
            "them here is the intended use — notebook 05 only ever swept five sessions."
        ),
        "momentum_change": (
            "The model predicts how far the momentum score moves over the next 15 bars, "
            "in **bps/min**. The rules read that as a direction call on a regime the tape "
            "has already printed: buy a *negative* minute the model expects to turn up, "
            "sell a *positive* one it expects to turn down, and cut on either risk exit. "
            "The notebook's own ablation says the exits carry the P&L — sweeping all four "
            "here is the intended use."
        ),
        "momentum": (
            "Four knobs decide everything: **when** the saved model is asked about a "
            "regime change, how sure it has to be, how much of the run the trade gives "
            "back before selling, and whether the model also gets to call the exit. Each "
            "distinct rule set is tracked as its own configuration in Results, so moving "
            "the entry, retuning the stop or arming the reversal exit is a new test "
            "rather than a repeat of one already run."
        ),
    },
    outro={
        "dayrange": (
            ":material/info: No entry mode, no probability, no trailing stop — none of them "
            "mean anything to a forecast of the day's range, and the run's signature leaves "
            "them out so a day-range result is never filed beside a momentum one."
        ),
        "momentum_change": (
            ":material/warning: This is the only model here that reads days *before* the "
            f"one being simulated: {momentum_change_model.HISTORY_SESSIONS} previous sessions of "
            "minute bars, because the regime threshold is yesterday's volatility. A dataset "
            "whose first days have nothing behind them will log a refusal to trade for "
            "those sessions rather than trading them blind."
        ),
    },
    help={
        "buy_k": (
            "The notebook's 0.75 was specified, not fitted, and its own sweep says "
            "why not to trust a peak: over five sessions the week total climbs "
            "steadily from $70 at 0.30 to $339 at 0.85 as deeper entries fill better, "
            "then turns erratic past 0.90 as whole days stop trading. Through all of "
            "it the count of profitable sessions is flat at three in five. The levels "
            "change the price paid on the same winning days, not how often the rule "
            "is right — and the best cell of a 195-cell grid on five sessions is "
            "mostly selection noise."
        ),
        "sell_k": (
            "Where the exit rests below the same predicted high — the smaller of the "
            "two numbers, since it is the higher price. A day that never reaches it "
            "is held to the closing flatten."
        ),
        "buy_thr": (
            "How large an upward move the model has to predict before a negative "
            "regime is bought. 0.30 is the notebook's, specified rather than fitted; "
            "0 buys every negative minute the model does not call down, which is the "
            "\"no model entry filter\" ablation."
        ),
        "sell_thr": (
            "Stated positive and compared against its negation: at 0.30 a held "
            "position is sold when the model predicts −0.30 bps/min or worse on a "
            "positive minute. This is the exit the ablation says does the work on "
            "both tickers."
        ),
        "m1_mult": (
            "A hard exit when momentum falls below this multiple of the day's regime "
            "threshold θ. Entries only happen while momentum is below −θ, so anything "
            "above −1 is already breached at entry and churns one-minute round trips; "
            "the notebook's sweep runs through that region deliberately."
        ),
        "stop_pct": (
            "A fixed stop measured from the entry price — not a trailing one. The "
            "momentum rules' trailing stop is a different strategy's knob and is not "
            "read here."
        ),
        "entry_mode": (
            "The setting that moves the fill most. On the 2026-07-27 SIP tape "
            "“Confirm” bought 337.45 / 338.67 / 336.35 and “Anticipate” bought the "
            "same three episodes at 336.56 / 338.20 / 335.99 — one to six bars "
            "earlier, while the regime was still balanced, taking the session from "
            "−0.41% to +0.08%. That is three trades on one day: a check that the "
            "wiring works, not a measurement of the edge."
        ),
        "anticipate_error": (
            "{label} was fitted on regime-change bars only, so it cannot "
            "forecast a change that has not happened yet. Pick a forecasting model "
            "or switch the entry to \u201cConfirm the turn\u201d; this pairing stops the run "
            "rather than producing an empty ledger."
        ),
        "prob_threshold": (
            "Default {threshold} is the cut-off this model chose on its own "
            "validation block — on the *confirm* question. On “Anticipate” it is a "
            "starting point rather than a tuned setting, and it is the first thing "
            "worth sweeping here: it decides how early in the build-up the entry "
            "fires."
        ),
        "trail_pct": (
            "Sell once price is this far below the highest price seen since the "
            "entry. The peak only ratchets up, so this starts as a stop under the "
            "entry and becomes a profit lock as the move runs."
        ),
        "sells_on_reversal": (
            "Closes the position when the model puts the positive regime at the "
            "probability below or better of flipping negative — while price may "
            "still be at its high, rather than waiting for the trailing stop's "
            "give-back. Only a forecasting model can be asked."
        ),
        "reversal_threshold": (
            "Over five AAPL sessions this separates bars within three of a positive "
            "run's end from bars with 8+ to go at 0.89 AUC, and the cut-off picks "
            "where to sit on it: 0.20 fires on 11% of held bars, 0.30 on 2.6%, 0.40 "
            "on 0.9%, with about half of each landing near the end against a 15% "
            "base rate. It fires in the right places; whether that pays is untested "
            "— A/B-ing those same sessions moved them +0.14%→+0.04% and "
            "−0.58%→−0.64%, which is noise on 6 and 12 round trips. No notebook ever "
            "tuned an exit, so this is the thing most worth sweeping here."
        ),
    },
)


def _render_apple_params(symbols: list[str], prefix: str) -> AppleTraderConfig:
    """One Apple Trader setup: its instrument and rules.

    `symbols` is what the selected datasets carry, used only to mark the
    instruments that can actually be replayed -- the list itself comes from
    what the models were fitted on, since without a model this agent has no
    rules at all. Whether the datasets carry the chosen one is checked before
    anything is queued (see `_rule_agents_missing_ticker`).

    The prefix is per setup, not per app: SimLab renders several of these on
    one page and their widgets must not collide.
    """
    return apple_trader_ui.params(
        symbols, apple_trader_ui.FormCopy(prefix=prefix, **_APPLE_TRADER_COPY_FIELDS)
    )


_apple_model_label = apple_trader_ui.model_label


def render_simulate_tab() -> None:
    _render_pipeline()
    st.divider()

    datasets = sim_data.list_datasets()
    if not datasets:
        st.info("Download a dataset first (Datasets tab).")
        return

    st.caption(
        "Pick several agents, models, and datasets — every combination is queued as "
        "its own experiment. A rule-based agent has no model to vary, so its "
        "**setups** take that place: add it as many times as you have "
        "configurations to compare, and each one is queued against every dataset."
    )
    names = [d.name for d in datasets]
    selected_names = st.multiselect(
        "Datasets", names, default=names[:1], key="sim_datasets"
    )
    by_name = {d.name: d for d in datasets}
    # The agent selection is read before it is drawn: the dataset scope below
    # derives its symbols from what the agents trade, and it is rendered first.
    # Streamlit hands a widget's stored value back already updated at the top of
    # the rerun a change causes, so this is the current selection.
    chosen = st.session_state.get("sim_agents")
    prior = list(chosen) if chosen is not None else _testable_agents()[:1]
    rule_tickers = _configured_rule_tickers(
        [p for p in prior if not sim_prompts.has_prompt(p)]
    )
    dataset_scope = _render_dataset_scope(
        [by_name[name] for name in selected_names],
        rule_tickers,
        has_llm=any(sim_prompts.has_prompt(p) for p in prior),
    )
    personalities = st.multiselect(
        "Agents", _testable_agents(),
        default=_testable_agents()[:1],
        format_func=_agent_label,
        key="sim_agents",
    )
    llm_personalities = [p for p in personalities if sim_prompts.has_prompt(p)]
    rule_personalities = [p for p in personalities if not sim_prompts.has_prompt(p)]
    # Every symbol the selected datasets *carry*, not what is ticked: this only
    # annotates the instrument pickers, and a rule agent's own symbol is no
    # longer gated by a tick, so narrowing it to the selection would mark
    # perfectly replayable instruments as unavailable.
    dataset_symbols = sorted(
        {symbol for name in selected_names for symbol in (by_name[name].symbols or [])}
    )
    rule_setups = _render_rule_params(rule_personalities, dataset_symbols)
    # The model picker only sizes the LLM grid: a rule agent runs the same
    # way whatever is selected there, so it is queued once per dataset instead.
    model_choices, api_keys = (
        _render_model_picker() if llm_personalities else ([], {})
    )

    with st.expander("Simulation settings"):
        col_cash, col_cycle, col_max = st.columns(3)
        starting_cash = col_cash.number_input("Starting cash", value=100_000.0, step=10_000.0)
        cycle_minutes = col_cycle.number_input(
            "Cycle interval (min)", value=5, min_value=1, max_value=60,
            help="Re-cycle cadence while nothing is armed. With alerts/tactics armed the "
                 "agent sleeps until a condition fires, exactly as live.",
        )
        max_cycles = col_max.number_input("Max LLM cycles per day", value=40, min_value=1, max_value=200)
        run_judge = st.checkbox(
            "Judge the run with an LLM after the simulation", value=True,
            help="Grades every entry on the information available at entry time, plus an "
                 "overall strategy-adherence review.",
        )
        if run_judge and rule_personalities:
            st.caption(
                ":material/gavel: "
                + ", ".join(_agent_label(p) for p in rule_personalities)
                + (" are" if len(rule_personalities) > 1 else " is")
                + " never judged — they state no reasoning of their own, so profit, "
                "profit efficiency and the oracle ceiling are the whole scorecard."
            )
        # Judging with the agent's own model is per-combination; a single
        # explicit judge is shared by every experiment in the grid.
        judge_override = run_judge and bool(llm_personalities) and st.checkbox(
            "Judge with a different LLM than the agent", value=False,
            help="By default each agent's own provider/model grades its run. Pick a "
                 "separate judge to avoid an agent marking its own homework.",
        )
        judge_provider = judge_model = judge_api_key = None
        if judge_override:
            col_jprov, col_jmodel, col_jkey = st.columns(3)
            judge_provider = col_jprov.selectbox("Judge provider", list(PROVIDERS))
            judge_model = col_jmodel.selectbox(
                "Judge model",
                models_for(judge_provider, default=DEFAULT_AGENT_MODELS[judge_provider]),
            )
            judge_api_key = col_jkey.text_input(
                "Judge API key", value=_env_key(judge_provider), type="password"
            )

    # One combination per (agent, model, dataset) for the LLM agents; a rule
    # agent has no model dimension, so its own rule set stands in for one.
    rule_combos, rule_by_combo = _rule_combinations(rule_setups, selected_names)
    combos = [
        (personality, provider, model, name)
        for name in selected_names
        for personality in llm_personalities
        for provider, model in model_choices
    ] + rule_combos
    days_by_dataset = {name: scope["days"] for name, scope in dataset_scope.items()}
    tested = _render_already_tested(combos, days_by_dataset)
    skip_tested = bool(tested) and st.checkbox(
        f"Skip the {len(tested)} combination(s) already tested cleanly", value=True,
        key="sim_skip_tested",
        help="Uncheck to run them again anyway — e.g. after editing a prompt.",
    )

    overridden = [p for p in personalities if sim_prompts.has_override(p)]
    if overridden:
        labels = ", ".join(_agent_label(p) for p in overridden)
        st.caption(f":material/edit: Runs with a **modified** prompt (Agents tab): {labels}.")
    missing_ticker = _rule_agents_missing_ticker(rule_setups, by_name, selected_names)
    for personality, ticker, names_missing in missing_ticker:
        st.error(
            f":material/error: {_agent_label(personality)} has a setup trading "
            f"{ticker}, which is not in: {', '.join(names_missing)}. Download that "
            "symbol into the dataset, or point the setup at another instrument."
        )
    st.caption(
        ":material/monitoring: Langfuse export: "
        + ("enabled — cycles are traced and run scores registered." if obs.is_enabled()
           else "disabled (set LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY to enable).")
    )

    to_queue = [c for c in combos if not (skip_tested and c in tested)]
    with st.container(horizontal=True, vertical_alignment="center"):
        run = st.button(
            f"Run {len(to_queue)} experiment(s)", icon=":material/play_arrow:",
            type="primary", disabled=not to_queue,
            help="Adds every combination to the pipeline; each starts as soon "
                 "as a worker slot is free.",
        )
        if combos:
            parts = []
            if llm_personalities:
                parts.append(
                    f"{len(llm_personalities)} LLM agent(s) × {len(model_choices)} model(s)"
                )
            if rule_personalities:
                setups = sum(len(rule_setups.get(p, [])) for p in rule_personalities)
                parts.append(
                    f"{setups} rule-based setup(s) "
                    f"over {len(rule_personalities)} agent(s)"
                )
            st.caption(
                f"({' + '.join(parts)}) × {len(selected_names)} dataset(s) = "
                f"{len(combos)} combination(s)"
                + (f", {len(combos) - len(to_queue)} skipped" if skip_tested else "")
            )
        elif personalities and not llm_personalities:
            st.caption("Pick at least one dataset.")
        else:
            st.caption("Pick at least one dataset, agent, and model.")

    if run:
        # Symbols only have to be non-empty where something reads them: an LLM
        # run is handed the basket, a rule run brings its own symbol and
        # `missing_ticker` above is what catches a dataset that cannot serve it.
        empty = [
            name for name in selected_names
            if not dataset_scope[name]["days"]
            or (llm_personalities and not dataset_scope[name]["symbols"])
        ]
        missing_keys = sorted({provider for provider, _ in model_choices
                               if not api_keys.get(provider)})
        if empty:
            st.error(
                (
                    "Pick at least one trading day and one symbol for: "
                    if llm_personalities
                    else "Pick at least one trading day for: "
                )
                + ", ".join(empty)
            )
        elif missing_ticker:
            personality, ticker, names_missing = missing_ticker[0]
            st.error(
                f"{', '.join(names_missing)} does not carry {ticker}. Download it "
                f"into the dataset, or change the {_agent_label(personality)} setup "
                "that trades it."
            )
        elif missing_keys:
            st.error(f"An API key is required for: {', '.join(missing_keys)}.")
        elif run_judge and judge_override and not judge_api_key:
            st.error(f"An API key for the judge provider ({judge_provider}) is required.")
        else:
            for personality, provider, model, name in to_queue:
                scope = dataset_scope[name]
                rule_based = personality in rule_personalities
                rule_settings = (
                    rule_by_combo[(personality, provider, model, name)]
                    if rule_based else None
                )
                sim_experiments.submit(name, {
                    "personality": personality,
                    "provider": provider,
                    "model": model,
                    "api_key": "" if rule_based else api_keys[provider],
                    "symbols": _experiment_symbols(
                        personality, rule_settings, scope["symbols"]
                    ),
                    "days": scope["days"],
                    "starting_cash": float(starting_cash),
                    "cycle_minutes": int(cycle_minutes),
                    "max_cycles_per_day": int(max_cycles),
                    "feed": by_name[name].feed,
                    "system_prompt_override": sim_prompts.get_override(personality),
                    "rule_config": (
                        rule_agent(personality).to_record(rule_settings)
                        if rule_based else None
                    ),
                    # A rule agent is never judged: no reasoning of its own to
                    # grade, so the profit metrics are the whole scorecard.
                    "run_judge": bool(run_judge) and not rule_based,
                    "judge_provider": judge_provider or provider,
                    "judge_model": judge_model or model,
                    "judge_api_key": judge_api_key or api_keys.get(provider, ""),
                })
            sim_experiments.tick(int(st.session_state.get(MAX_PARALLEL_KEY, 2)))
            st.toast(f"{len(to_queue)} experiment(s) queued",
                     icon=":material/rocket_launch:")
            st.rerun()


# ---------------------------------------------------------------------------
# Tabs 4 & 5 — summary (aggregates) and results (one run at a time)
# ---------------------------------------------------------------------------

# Agent first and by default: it is the thing under test. "Model" covers both
# the LLM behind a personality and the rule set behind the rule-based agent --
# both are what varies while agent and dataset are held fixed. "Instrument" is
# the symbol a run traded, which is a dimension in its own right here: the same
# rule set over AAPL and over INTC is two configurations, and whether a result
# is the strategy or the tape is exactly what comparing them answers.
# "LLM / rules" is what used to be called "Model". It was renamed when "ML
# model" arrived beside it: the old name covered the LLM behind a personality
# *and* the rule set behind a rule agent, which is a different axis entirely
# from which saved artifact out of `Code/Models` produced the decisions, and
# two adjacent buttons both reading "Model" would have made the page unusable.
_BREAKDOWN_DIMENSIONS = {
    "Agent": "agent", "LLM / rules": "model", "Dataset": "dataset",
    "Instrument": "instrument", "ML model": "ml_model",
}


def _ml_model_label(key: str) -> str:
    """One ML-model breakdown row, named for a reader rather than for the store.

    Three shapes, because `results.ml_model_key` produces three: a provider for
    an LLM run, a set of `apple_models` keys for a rule run, and the sentinels
    for a rule set that names none. Model names come from the registry, so
    renaming a model there moves the row label with it.

    The names are shortened to what identifies the model -- the breakdown table
    does not wrap, and a rule set naming two models would otherwise put a
    hundred characters in the first column.
    """
    if key.startswith(sim_results.LLM_MODEL_PREFIX):
        return f"{key[len(sim_results.LLM_MODEL_PREFIX):]} (LLM)"
    if not key or key in (sim_results.NO_ML_MODEL, sim_results.UNKNOWN_INSTRUMENT):
        return key or sim_results.UNKNOWN_INSTRUMENT

    def _short(model_key: str) -> str:
        # Membership, not `apple_models.get`: that deliberately falls back to
        # the default model for an unknown key so a stored run still replays,
        # which here would print a real model's name over a key that is not
        # one. A key this does not recognise is shown as itself.
        model = apple_models.MODELS.get(model_key)
        if model is None:
            return model_key
        label = model.label
        for cut in (" (", " →"):
            label = label.split(cut)[0]
        return label.strip() or model_key

    return " + ".join(_short(k) for k in key.split(sim_results.ML_MODEL_JOIN))

# Ranking metrics for the top-runs cards, mapped to their `summary` keys.
_TOP_RUN_METRICS = {"Best return": "return_pct", "Profit efficiency": "profit_efficiency"}


# The breakdown table is hand-rolled HTML rather than st.dataframe: only a real
# `title` attribute gives the best-return cell a hover tooltip naming the run
# behind it (st.column_config's `help` only tooltips the column header).
_BREAKDOWN_CSS = f"""
<style>
table.simlab-breakdown {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
table.simlab-breakdown th, table.simlab-breakdown td {{
    padding: 0.4rem 0.6rem; text-align: right; white-space: nowrap;
    border-bottom: 1px solid rgba(128, 128, 128, 0.25);
}}
table.simlab-breakdown th {{ font-weight: 600; opacity: 0.75; }}
table.simlab-breakdown th:first-child, table.simlab-breakdown td:first-child {{ text-align: left; }}
table.simlab-breakdown th[title] {{ cursor: help; text-decoration: underline dotted; }}
table.simlab-breakdown .best {{ cursor: help; text-decoration: underline dotted; }}
table.simlab-breakdown .up {{ color: {PALETTE["up"]}; }}
table.simlab-breakdown .down {{ color: {PALETTE["down"]}; }}
table.simlab-breakdown .none {{ opacity: 0.45; }}
</style>
"""


def _best_run_tooltip(best: "dict | None") -> str:
    """Which model / agent / dataset produced a group's best return."""
    if not best:
        return ""
    provider, model = best.get("provider"), best.get("model")
    lines = [
        f"Model: {'/'.join(p for p in (provider, model) if p) or '?'}",
        "Agent: " + _agent_label(best.get("personality")),
        f"Dataset: {best.get('dataset') or '(no dataset)'}",
        # Named here for the same reason the other three are: in a breakdown
        # along any one dimension the rest are invisible, and which symbol
        # produced a number is not a detail on this page.
        f"Instrument: {best.get('instrument') or '?'}",
        "ML model: " + _ml_model_label(best.get("ml_model") or ""),
    ]
    if best.get("run_id"):
        lines.append(f"Run: {best['run_id']}")
    return "\n".join(lines)


def _render_breakdown_table(rows: list[dict], dim_label: str) -> None:
    headers = [
        (dim_label, ""),
        ("Runs", ""),
        ("Avg return", ""),
        ("Best return", "Best single run in this group — hover the value for its "
                        "model, agent, and dataset."),
        ("Avg profit efficiency",
         "Session return ÷ oracle best-round-trip ceiling, averaged over runs."),
        ("Avg judge score", "LLM judge overall score (0–10), averaged over judged runs."),
    ]
    head = "".join(
        f'<th title="{escape(help_text, quote=True)}">{escape(label)}</th>'
        if help_text else f"<th>{escape(label)}</th>"
        for label, help_text in headers
    )
    missing = '<span class="none">—</span>'

    def _pct(value: "float | None") -> str:
        if value is None:
            return missing
        return f'<span class="{"up" if value >= 0 else "down"}">{value:+.2f}%</span>'

    body = []
    for row in rows:
        best = _pct(row["best_return_pct"])
        tooltip = _best_run_tooltip(row.get("best_run"))
        if tooltip and row["best_return_pct"] is not None:
            # &#10; keeps the tooltip multi-line without a raw newline inside the
            # attribute, which markdown would treat as a block break.
            title = escape(tooltip, quote=True).replace("\n", "&#10;")
            best = f'<span class="best" title="{title}">{best}</span>'
        efficiency = row["avg_profit_efficiency"]
        score = row["avg_judge_score"]
        body.append(
            "<tr>"
            f"<td>{escape(str(row['group']))}</td>"
            f"<td>{row['runs']}</td>"
            f"<td>{_pct(row['avg_return_pct'])}</td>"
            f"<td>{best}</td>"
            f"<td>{f'{efficiency:.1%}' if efficiency is not None else missing}</td>"
            f"<td>{f'{score:.1f}' if score is not None else missing}</td>"
            "</tr>"
        )
    st.markdown(
        _BREAKDOWN_CSS
        + f'<table class="simlab-breakdown"><thead><tr>{head}</tr></thead>'
        + f"<tbody>{''.join(body)}</tbody></table>",
        unsafe_allow_html=True,
    )
    st.caption("Hover a best return to see the run behind it.")


def _breakdown_bar(labels: list[str], values: list[float], title: str, color: str,
                   tickformat: "str | None" = None) -> go.Figure:
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=color))
    fig = _chart_layout(fig, height=320)
    fig.update_layout(title=title, showlegend=False)
    if tickformat:
        fig.update_yaxes(tickformat=tickformat)
    return fig


def _render_run_filters(runs: list[dict], key_prefix: str) -> list[dict]:
    """Dataset / model filters over the stored runs. Selecting nothing in a
    filter leaves that dimension unrestricted, so the default view is all runs.
    Everything below (breakdown, charts, run picker) works off the result.
    Summary and Results each render their own copy -- hence the key prefix --
    so filtering one tab doesn't silently reshape the other."""
    options = sim_results.filter_options(runs)
    col_datasets, col_models = st.columns(2)
    datasets = col_datasets.multiselect(
        "Datasets", options["datasets"], key=f"{key_prefix}_filter_datasets",
        placeholder="All datasets",
    )
    models = col_models.multiselect(
        "Models", options["models"], key=f"{key_prefix}_filter_models",
        placeholder="All models",
    )
    filtered = sim_results.filter_runs(runs, datasets=datasets, models=models)
    if datasets or models:
        st.caption(f"Showing {len(filtered)} of {len(runs)} runs.")
    return filtered


def _short_model(key: str, limit: int = 46) -> str:
    """Rule agents encode their entire rule set in the model string, which would
    otherwise stretch one card far past the others. The full string is still in
    the breakdown table and on the run itself."""
    return key if len(key) <= limit else key[: limit - 1].rstrip() + "…"


def _render_top_runs(runs: list[dict]) -> None:
    """The three best single runs under the filters, on whichever metric is
    picked. Return and profit efficiency disagree often -- a big return on an
    easy tape can be a worse trade than a small one on a flat tape -- so both
    are always shown, only the ranking changes."""
    st.markdown("##### Top runs")
    metric_label = st.segmented_control(
        "Rank by", list(_TOP_RUN_METRICS), default="Best return",
        key="summary_top_metric",
    ) or "Best return"
    metric = _TOP_RUN_METRICS[metric_label]
    top = sim_results.top_runs(runs, by=metric, limit=3)
    if not top:
        st.caption(f"No runs scored on {metric_label.lower()} yet.")
        return
    for rank, (column, row) in enumerate(zip(st.columns(len(top)), top), start=1):
        efficiency = row["profit_efficiency"]
        return_pct = row["return_pct"]
        headline = (
            f"{return_pct:+.2f}%" if metric == "return_pct"
            else f"{efficiency:.1%}"
        )
        if metric == "return_pct":
            secondary = (
                f"Profit efficiency {efficiency:.1%}" if efficiency is not None
                else "Profit efficiency —"
            )
        else:
            secondary = (
                f"Return {return_pct:+.2f}%" if return_pct is not None
                else "Return —"
            )
        with column.container(border=True):
            st.metric(f"#{rank} · {_agent_label(row['personality'])}", headline)
            st.caption(
                f"{secondary}  \n{_short_model(row['model'])} · {row['dataset']}"
                f"  \n`{row['run_id']}`"
            )


@st.dialog("Delete all runs?")
def _confirm_delete_all_runs(total: int) -> None:
    """Wiping the store is irreversible and the filters make it easy to forget
    how much is actually in there, so the count is spelled out before it goes."""
    st.markdown(
        f"This permanently deletes **all {total} stored run"
        f"{'' if total == 1 else 's'}**, including any hidden by the current "
        "filters. Scores already exported to Langfuse are kept."
    )
    with st.container(horizontal=True):
        if st.button("Delete them", type="primary", icon=":material/delete_forever:"):
            removed = sim_results.delete_all_runs()
            for key in (
                "last_run_id",
                "results_filter_datasets", "results_filter_models",
                "summary_filter_datasets", "summary_filter_models",
            ):
                st.session_state.pop(key, None)
            st.toast(f"Deleted {removed} run{'' if removed == 1 else 's'}",
                     icon=":material/delete_sweep:")
            st.rerun()
        if st.button("Cancel", icon=":material/close:"):
            st.rerun()


def render_summary_tab() -> None:
    all_runs = _runs()
    if not all_runs:
        st.info("No stored runs yet — queue an experiment in the Simulate tab.")
        return

    runs = _render_run_filters(all_runs, "summary")
    if not runs:
        st.info("No runs match the current filters.")
        return

    _render_top_runs(runs)

    st.divider()
    st.markdown("##### Breakdown")
    dim_label = st.segmented_control(
        "Break down by", list(_BREAKDOWN_DIMENSIONS), default="Agent",
        # Keyed apart from the pre-"ML model" control: that one stored the
        # chosen *label*, and "Model" is no longer one of the options.
        key="results_breakdown_dim_v2",
    ) or "Agent"
    dimension = _BREAKDOWN_DIMENSIONS[dim_label]
    rows = sim_results.breakdown(runs, dimension)
    # Both of these group on a stable key and render a human label, so a
    # renamed agent or model moves the row without rewriting stored runs.
    relabel = {"agent": _agent_label, "ml_model": _ml_model_label}.get(dimension)
    if relabel:
        for row in rows:
            row["group"] = relabel(row["group"])
    if dimension == "ml_model":
        st.caption(
            ":material/info: What the run's decisions actually came out of. Apple "
            "Trader names one model and that model *is* its strategy; Apple Trader 2 "
            "names them per condition, so a rule set reads none, one or several and "
            "the set is the row. An LLM agent loads no saved model at all, so those "
            "runs are grouped by **provider** — the rows are comparable as "
            "*approaches*, not as one model against another."
        )
    _render_breakdown_table(rows, dim_label)
    col_eff, col_score = st.columns(2)
    eff_rows = [r for r in rows if r["avg_profit_efficiency"] is not None]
    if eff_rows:
        col_eff.plotly_chart(_breakdown_bar(
            [r["group"] for r in eff_rows],
            [r["avg_profit_efficiency"] for r in eff_rows],
            "Avg profit efficiency", PALETTE["accent"], tickformat=".0%",
        ))
    score_rows = [r for r in rows if r["avg_judge_score"] is not None]
    if score_rows:
        col_score.plotly_chart(_breakdown_bar(
            [r["group"] for r in score_rows],
            [r["avg_judge_score"] for r in score_rows],
            "Avg judge score (0–10)", PALETTE["up"],
        ))


def render_results_tab() -> None:
    all_runs = _runs()
    if not all_runs:
        st.info("No stored runs yet — queue an experiment in the Simulate tab.")
        return

    runs = _render_run_filters(all_runs, "results")
    if not runs:
        st.info("No runs match the current filters.")
        return

    labels = {
        r["run_id"]: (
            f"{r['run_id']} · {r.get('config_summary', {}).get('personality')} · "
            f"{r.get('config_summary', {}).get('model')} · {r.get('dataset')} · "
            f"{r.get('summary', {}).get('return_pct', 0):+.2f}%"
        )
        for r in runs
    }
    default_id = st.session_state.get("last_run_id", runs[0]["run_id"])
    ids = list(labels)
    selected = st.selectbox(
        "Run", ids,
        index=ids.index(default_id) if default_id in ids else 0,
        format_func=lambda rid: labels[rid],
    )
    record = next(r for r in runs if r["run_id"] == selected)
    _render_run(record)
    with st.container(horizontal=True):
        if st.button("Delete this run", icon=":material/delete:"):
            sim_results.delete_run(selected)
            st.session_state.pop("last_run_id", None)
            st.rerun()
        if st.button(
            "Delete all runs", icon=":material/delete_sweep:",
            help="Clears the whole run store, not just the filtered runs.",
        ):
            _confirm_delete_all_runs(len(all_runs))


# ---------------------------------------------------------------------------

def build_ui() -> None:
    st.set_page_config(page_title="AgentStonks SimLab", page_icon="🧪", layout="wide")
    st.title("SimLab — strategy testing")
    st.caption(
        "Replay the trading agents against stored historical sessions: same prompts, same "
        "tools, same execution path as live — hours of tape in minutes of simulation."
    )
    # ML Models sits beside Agents rather than near Results: it describes what
    # an agent *is* before a run, not what one did afterwards.
    (
        tab_agents, tab_models, tab_datasets, tab_sim, tab_summary, tab_results,
    ) = st.tabs(
        [":material/smart_toy: Agents", ":material/neurology: ML Models",
         ":material/database: Datasets",
         ":material/play_circle: Simulate", ":material/leaderboard: Summary",
         ":material/insights: Results"]
    )
    with tab_agents:
        render_agents_tab()
    with tab_models:
        model_catalogue_panel()
    with tab_datasets:
        render_datasets_tab()
    with tab_sim:
        render_simulate_tab()
    with tab_summary:
        render_summary_tab()
    with tab_results:
        render_results_tab()
