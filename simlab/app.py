"""SimLab Streamlit UI: agents / datasets / simulate / summary / results.

Run with ``streamlit run sim_main.py``. Kept separate from the live dashboard
(``main.py``) -- this app never opens a stream or touches the live tape; it
only reads the local dataset store and replays agents against it.
"""
from __future__ import annotations

import contextlib
import json
import os
import textwrap
import threading
import uuid
from collections.abc import Callable
from datetime import date, datetime, time, timedelta, timezone
from html import escape
from itertools import combinations
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from agent_stonks import (
    apple_models,
    candle_patterns,
    clock,
    minute_momentum,
    model_overlays,
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
from agent_stonks.orchestra import ORCHESTRA_KEY
from agent_stonks.orchestra import default_pairs as default_orchestra_pairs
from agent_stonks.charts import (
    _rgba,
    add_candle_patterns,
    add_model_overlays,
    add_session_markers,
    overlay_x_max,
    session_rangebreaks,
)
from agent_stonks.config import MODEL_OVERLAY_COLORS, PALETTE
from agent_stonks.model_catalogue_ui import model_catalogue_panel
from agent_stonks.llm import DEFAULT_AGENT_MODELS, ENV_KEYS, PROVIDERS, models_for
from agent_stonks.market_hours import MARKET_TZ, is_market_open

from . import data as sim_data
from . import drift as sim_drift
from . import experiments as sim_experiments
from . import prompts as sim_prompts
from . import results as sim_results
from . import session_context as sim_session
from . import tuning as sim_tuning
from .engine import SimulationConfig, SimulationEngine
from .market import SimMarket
from .patches import simulation_context
from dataclasses import asdict

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
    thresholds are per-simulation settings, picked in the Simulate tab -- so
    this is a read-only description of what the loop does.
    """
    st.subheader(_agent_label(personality))
    st.caption(
        ":material/function: Rule-based — no LLM, no prompt, no tools. The same tape "
        "always produces the same trades."
    )
    if personality == ORCHESTRA_KEY:
        st.markdown(
            "Apple Trader's rules on several **(ticker, model) pairs** at once, from one "
            "cash balance. Every pair forecasts its own session at 9:35 and rests its own "
            "levels; the first whose buy fills is the only one that trades until its "
            "position is closed, and then the others may buy again. A stop-out stands "
            "every pair down for the rest of that session. Each pair's own rules are "
            "Apple Trader's, unchanged — Orchestra with one pair replays exactly as "
            "Apple Trader."
        )
        st.caption(
            ":material/database: A dataset must carry every pair's symbol, and the "
            "oracle ceiling is the best round trip on any of them."
        )
        return
    _render_apple_rules()


def _render_apple_rules() -> None:
    """What Apple Trader does, plus the provenance of the bundles it needs."""
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
        + ". TimeToChange3 was run per ticker."
    )

    st.markdown("##### The day-range rules — `dayrange`")
    st.markdown(
        "One question, asked once. At **9:35** the model forecasts where the session's "
        "high **H** and low will land, from a year of daily history plus the first five "
        "minutes, and the rest of the day is two levels derived from it "
        "(TimeToChange3 notebook 05), with **A** the trailing 14-day average daily range "
        "in dollars:\n"
        "- **Buy** when a bar's low reaches `H − buy × A`, well under where the day is "
        "expected to top out.\n"
        "- **Sell** when a bar's high reaches `H − sell × A`, just under it. Then it can "
        "buy again, as often as the day allows.\n"
        "- Anything still open is flattened before the close.\n"
        "- A **managed exit** sits on top, every distance measured from the fill: a "
        "**stop** when a bar's low falls `stop × A` under it (after which the session takes "
        "no new entry); a **take** of part of the position, while the trade is in profit, "
        "once the momentum over the last `N` bars has been positive since the entry and "
        "turns balanced or negative; and the rest "
        "kept as a **runner** only if the sell level is still `hold × A` above the fill, "
        "sold if the price comes back to the fill. It is not in the notebook's numbers, "
        "and a stored run from before it existed replays with it off.\n"
        "- **Two things end a session's buying**, not one: the stop above, and a trade "
        "that closes for no more than `min win × A` a share — measured over the whole "
        "position, so a momentum take and the runner it left are judged together. The "
        "reasoning is that a round trip which barely paid is evidence the setup was not "
        "there today. Note that the most a target exit can net is `(buy − sell) × A`, so a "
        "threshold at or above that stands the session down after every completed trade.\n"
        "- **H** need not be the predicted high. The *levels measured below* setting can "
        "point the two distances at the upper curve of the **predicted intraday range × "
        "day range** band instead — the same forecast stretched by IntradayVolatility's "
        "time-of-day shape around the session's open — so the reference is the predicted "
        "high at 09:30, a fifth of that distance by midday, and wider again into the "
        "close. The whole ladder moves with it, targets included. It needs that symbol's "
        "own `intravol_<TICKER>.json`, and the shipped buy/sell distances were swept "
        "against a reference that does not move.\n"
        "- **H** itself can also move. The model runs once and cannot be re-run, but a session "
        "that trades *through* the predicted high has falsified it, so an **intraday "
        "update** moves the breached side — to the extreme so far, or past it by the "
        "excursion a driftless walk with ADR-implied volatility would still be expected to "
        "make — and both levels are rebuilt from it, including under an open position. "
        "Only outward, never back towards the price. Off gives the notebook's fixed "
        "levels, which is what a record written before the setting existed replays as.\n\n"
        "It is a mean-reversion bet, and deliberately so — what the model forecasts well "
        "is the *width* of the day, not its direction. On a day that never dips to the buy "
        "level it does nothing at all."
    )
    st.caption(
        ":material/compare_arrows: **Against the notebook**, one difference matters and it "
        "runs against the strategy: the notebook rests limit orders and fills a touch *at* "
        "the level, while this ledger sends its order only after the bar that touched it "
        "has closed. That order is a limit at the buy level, so it never pays more than "
        "the level — but a bar that dipped to the level and recovered buys nothing here, "
        "where the notebook bought. (Runs recorded before 2026-09-30 used a market order "
        "and bought such a bar near its close, above the level.) Read the two side by "
        "side with that in mind."
    )

    st.markdown("##### Models")
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
                _render_dayrange_bundle(bundle)


def _render_dayrange_bundle(bundle: dict) -> None:
    """The day-range bundle's provenance, in its own units.

    The error is dollars of misprediction on a price, not an AUC on a label,
    and there is no threshold at all.
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
        "absolute miss on the day's high and low over a 129-session test window."
    )
    st.json(
        {"test_metrics": test, "constraint": metadata.get("constraint"),
         "opening_correction_gain": metadata.get("opening_correction_loo_gain")},
        expanded=False,
    )


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

def _incomplete_note(item: "sim_data.IncompleteFile") -> str:
    """What happens next to a stored day that is not whole."""
    if datetime.now(timezone.utc) < sim_data.day_final_at(item.day):
        return "re-downloaded once the day is over"
    if item.fixable:
        return "re-download pending (retried automatically)"
    return "no longer downloadable on this feed"


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
    # What the automatic repair (`_repair_incomplete_store`) could not fix yet:
    # a day still trading, or one Yahoo no longer serves.
    incomplete = [f for f in sim_data.incomplete_files() if f.kind == "bars"]
    for ds in datasets:
        with st.container(border=True, horizontal=True, vertical_alignment="center"):
            st.markdown(
                f"**{ds.name}** — {', '.join(ds.symbols)} · {ds.start} → {ds.end} "
                f"· {len(ds.days)} trading day(s) · `{ds.feed}`"
            )
            if st.button("Delete", key=f"del_ds_{ds.name}", icon=":material/delete:"):
                sim_data.delete_dataset(ds.name)
                st.rerun()
        lo, hi = ds.date_range()
        reach = set(sim_data.weekdays(lo, hi)) | set(minute_momentum.prior_week_days(lo))
        partial = [
            f for f in incomplete
            if f.feed == ds.feed and f.symbol in ds.symbols and f.day in reach
        ]
        if partial:
            groups: "dict[tuple[str, str], list[date]]" = {}
            for item in partial:
                groups.setdefault((item.symbol, _incomplete_note(item)), []).append(item.day)
            week_before = sum(1 for item in partial if item.day < lo)
            st.caption(
                f":material/warning: {len(partial)} stored day(s) are not whole"
                + (f" ({week_before} in the week before the first session, read only for "
                   "`abs_mean_minute_momentum`)" if week_before else "")
                + " — "
                + "; ".join(
                    f"{symbol} {days[0]}"
                    + (f" → {days[-1]} ({len(days)} days)" if len(days) > 1 else "")
                    + f": {note}"
                    for (symbol, note), days in groups.items()
                ),
                help="\n\n".join(f"{item.symbol} {item.day}: {item.reason}" for item in partial),
            )


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
    passed, an Apple Trader fill names the level it reached. None of that was reachable from the chart before; it was in the
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


def _regular_session_bars(bars: list[dict]) -> list[dict]:
    """The bars stamped 09:30-15:59 ET on a weekday -- the session the
    engine trades in (`market_hours.is_market_open`, as `_run_rule_day` asks
    it). Pre-market, after-hours and overnight bars are dropped."""
    out = []
    for bar in bars:
        ts = clock.parse_iso(bar.get("t"))
        if ts is not None and is_market_open(ts):
            out.append(bar)
    return out


def _price_chart(
    symbol: str,
    bars: list[dict],
    decisions: list[dict],
    overlays: "list[dict] | None" = None,
    patterns: "list[dict] | None" = None,
) -> go.Figure:
    """The replayed day(s)' candles, the fills, and what the models predicted.

    `patterns` are `candle_patterns.compute` items (fair value gaps), drawn by
    the live chart's renderer too.

    `overlays` are `model_overlays.compute` items, drawn by the same renderer
    the live chart uses. There is no price-profile column here, so the
    predicted levels are drawn once instead of mirrored -- everything else is
    identical, which is the point of the items being data rather than plotly
    calls.

    Only the regular session is drawn (`_regular_session_bars`): the agents
    trade 09:30-16:00, and the pre-, post- and overnight tape the store keeps
    was never traded on. A run usually covers several days, so the time from
    one day's close to the next day's open is removed from the axis rather
    than drawn as blank space (`charts.session_rangebreaks`), and each day's
    open is put back as a dated rule (`charts.add_session_markers`).
    """
    bars = _regular_session_bars(bars)
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
        x0, x1 = pd.Timestamp(bars[0]["t"]), pd.Timestamp(bars[-1]["t"])
        # Pinned to the session bars, so a fill outside the regular session
        # (the pre-market agent can fill before the bell) cannot stretch the
        # axis back over the hours this chart leaves out. It is still in the
        # Decisions table.
        fig.update_xaxes(
            rangebreaks=session_rangebreaks(bars),
            range=[x0, overlay_x_max(overlays, x1)],
        )
    if overlays and bars:
        add_model_overlays(overlays, fig, x0, x1, row=None, col=None)
    if patterns and bars:
        add_candle_patterns(
            patterns, fig, pd.Timestamp(bars[0]["t"]), pd.Timestamp(bars[-1]["t"]),
            row=None, col=None,
        )
    fig.update_layout(xaxis_rangeslider_visible=False)
    return _chart_layout(fig, height=420)



@st.cache_data(show_spinner=False)
def _overlay_day(
    _market: "SimMarket",
    feed: str,
    symbol: str,
    day: date,
    selected: tuple,
    trader_config: tuple = (),
) -> dict:
    """One replayed day's overlay items, computed once per tape and selection.

    Now that the run's own model is pre-selected, this runs on merely *opening*
    a run rather than on asking for it, and it runs again on every rerun the
    page does -- a filter change, a tab, the delete button. Forecasting five
    sessions through the day-range bundle each time would make Results feel
    broken, and the answer cannot move: a stored day's bars and a trained
    model are both fixed. Same reasoning as `model_overlays.live_overlays`'
    per-bar cache, a different lifetime.

    The key is the tape (feed + symbol + day) plus `trader_config`, and that
    second half is the exception that proves the first: every *model* overlay is
    about the session and not about who traded it, so two runs over one tape get
    one answer -- but `trader_levels` draws an agent's resting orders, which are
    exactly a property of the run. It arrives as a flat tuple of
    `(field, value)` pairs rather than the dataclass because this is
    `st.cache_data` and a key has to hash.
    """
    t = _market.session_open(day) + timedelta(minutes=1)
    return model_overlays.compute(
        list(selected),
        symbol,
        _market.series[symbol].minute_bars,
        daily_bars=_market.completed_daily_bars(symbol, t),
        session_date=day,
        open_price=_market.session_open_price(symbol, t),
        trader_config=_apple_config_from(trader_config),
    )


def _apple_config_from(pairs: tuple) -> "AppleTraderConfig | None":
    """Rebuild the configuration `_apple_config_key` flattened, or None.

    The pairs came from an already-decoded config, so this is the plain
    constructor -- the record's own reading (legacy fields, removed ones) was
    done on the way in, where it belongs.
    """
    if not pairs:
        return None
    try:
        return AppleTraderConfig(**dict(pairs))
    except Exception:
        return None


def _apple_config_key(record: dict, symbol: str) -> tuple:
    """The stored run's Apple Trader configuration, as a hashable cache key.

    Decoded through `rule_agents` rather than read field by field: a record
    predates fields that exist now and carries fields that no longer do, and
    `from_record` is the one place that knows what each absence meant. Reading
    `rule_config["ticker"]` directly would also miss the oldest records, which
    describe an AAPL run by not naming a symbol at all.

    Empty for every other agent, for a configuration that no longer decodes, and
    for a symbol tab this run did not trade -- drawing one run's levels over
    another symbol's candles would be a caption that reads right over a picture
    that is wrong. The overlay then falls back to the shipped configuration,
    which is what it does everywhere else it is not told otherwise.
    """
    summary = record.get("config_summary") or {}
    personality = summary.get("personality")
    if personality not in (APPLE_TRADER_KEY, ORCHESTRA_KEY):
        return ()
    try:
        config = rule_agent(personality).from_record(summary.get("rule_config") or {})
    except Exception:
        return ()
    if personality == ORCHESTRA_KEY:
        # Orchestra trades several symbols, two pairs possibly on one: this
        # symbol's tab draws the first racer on it, as the live chart does.
        config = next(
            (r for r in config.racers if r.ticker == (symbol or "").upper()), None
        )
        if config is None:
            return ()
    if config.ticker != (symbol or "").upper():
        return ()
    return tuple(sorted(asdict(config).items()))


def _run_overlay_controls(
    record: dict, market: "SimMarket", symbol: str, days: "list[date]"
) -> dict:
    """Pick which model predictions to draw over a replayed day, and compute them.

    **The run's own model is pre-selected.** Reading a result means asking
    whether the model was right, and that question is one chart away only if
    the chart already shows what the model said -- so a run driven by the
    day-range forecast opens with that forecast over its candles
    (`simlab.results.ml_models` -> `model_overlays.for_models`). This is the
    one place in the page that reads the run's configuration to decide what to
    draw; everything below still just draws what is selected.

    The picker stays, and stays available on *every* run, including the ones no
    model drove: "what would the day-range model have said about this LLM
    agent's tape" is the other question worth asking here, and an empty default
    is the honest starting point when nothing in the run can answer it.

    Each day is scored on its own, from the state of the world at its 9:31 --
    completed daily bars strictly before it (`SimMarket.completed_daily_bars`)
    and the stored opening print -- which is the same point-in-time view the
    agent had. A replay chart that showed a forecast built on the day's own
    outcome would be worse than no forecast at all.
    """
    available = model_overlays.keys_for(symbol)
    if not available:
        return {"items": [], "notes": []}

    # What the run itself loaded, in the terms the overlay catalogue uses. A
    # symbol tab the run's model was never fitted on drops out here rather than
    # being offered as a default the picker has no option for.
    ran = model_overlays.for_models(sim_results.ml_models(record) or [], symbol)

    run_id = record.get("run_id") or "run"
    selected = st.multiselect(
        "Model predictions",
        available,
        default=ran["keys"],
        format_func=model_overlays.label,
        key=f"sim_overlays_{run_id}_{symbol}",
        help="What the trained models predicted for this session, drawn over "
        "the replayed tape: predicted ranges as horizontal lines, time-of-day "
        "ranges as a shaded envelope that follows the session's volatility, "
        "time-spanning predictions as a shaded background. The model this run traded on is selected for you; add or "
        "remove any of the others.",
    )
    notes: list[str] = []
    for key in ran["unmatched"]:
        # Named off the registry rather than through `apple_models.get`, which
        # answers an unknown key with the default model -- and a record can
        # name a model that has since been renamed or retired. Saying the key
        # back is honest; calling a retired model "Persistence classifier" is
        # not.
        model = apple_models.MODELS.get(key)
        notes.append(
            f"This run traded on {model.label if model else key}, and no overlay "
            "draws what it predicts — so the chart shows the tape it traded, not "
            "its forecasts."
        )
    if not selected:
        return {"items": [], "notes": notes}

    items: list[dict] = []
    for day in days:
        result = _overlay_day(
            market, market.feed, symbol, day, tuple(selected),
            _apple_config_key(record, symbol),
        )
        items.extend(result["items"])
        for note in result["notes"]:
            entry = f"{day}: {note}" if len(days) > 1 else note
            if entry not in notes:
                notes.append(entry)
    return {"items": items, "notes": notes}


def _run_pattern_controls(record: dict, symbol: str, bars: "list[dict]") -> "list[dict]":
    """Pick which candle patterns to draw over a replayed tape, and find them.

    Nothing is pre-selected: unlike a model overlay, no run traded on a
    pattern. Each replayed day is read on its own (`candle_patterns.compute`).
    """
    run_id = record.get("run_id") or "run"
    c1, c2, c3 = st.columns([2, 1.2, 0.8], vertical_alignment="bottom")
    with c1:
        selected = st.multiselect(
            "Candle patterns",
            candle_patterns.keys(),
            format_func=candle_patterns.label,
            key=f"sim_patterns_{run_id}_{symbol}",
            help="Shapes read off the replayed candles themselves. A fair value "
            "gap is three candles where candle 1's wick and candle 3's wick "
            "don't overlap: boxed from candle 1, green up / red down, faded once "
            "price trades through it. Regular session only.",
        )
    if candle_patterns.FVG_KEY not in selected:
        return candle_patterns.compute(selected, bars)
    with c2:
        min_size = st.slider(
            "Min FVG size (× avg bar range)",
            min_value=0.0, max_value=2.0, step=0.25,
            value=candle_patterns.DEFAULT_FVG_MIN_SIZE,
            key=f"sim_fvg_min_{run_id}_{symbol}",
            help="Drops gaps narrower than this many times the average high-low "
            "range of the 14 bars before them; 0 draws every one.",
        )
    with c3:
        hide_filled = st.checkbox(
            "Hide filled", key=f"sim_fvg_hide_{run_id}_{symbol}",
            help="Only draw gaps price had not traded through by the end of the day.",
        )
    return candle_patterns.compute(
        selected, bars, min_size=min_size, hide_filled=hide_filled
    )


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
                        patterns = _run_pattern_controls(record, sym, bars)
                        st.plotly_chart(
                            _price_chart(sym, bars, decisions, overlays["items"], patterns)
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
            if personality == ORCHESTRA_KEY:
                pairs = st.session_state.get(f"sim_rule_{personality}_{slot}_pairs")
                symbols = [
                    str(p).partition(":")[0].upper()
                    for p in (default_orchestra_pairs() if pairs is None else pairs)
                ]
            else:
                stored = st.session_state.get(f"sim_rule_{personality}_{slot}_ticker")
                symbols = [str(stored or agent.default_ticker).strip().upper()]
            for symbol in symbols:
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
        return rule_agent(personality).symbols(rule_settings)
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
        traded = [t for config in configs for t in agent.symbols(config)]
        for ticker in dict.fromkeys(traded):
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
    Each configuration queues its own experiment per dataset, and lands in
    Results as its own row, since the signature is a function of the settings.

    Flattened across setups on the way out: a setup is one configuration until
    it is swept (`_render_apple_sweep`) and a grid of them after, and nothing
    downstream needs to know which setup a configuration came from.

    `symbols` is every symbol the selected datasets carry, offered to the
    instrument pickers.
    """
    renderers = {
        APPLE_TRADER_KEY: _render_apple_setup,
        ORCHESTRA_KEY: _render_orchestra_setup,
    }
    return {
        key: _render_agent_setups(key, symbols, renderers[key])
        for key in personalities
        if key in renderers
    }


def _setup_title(signatures: "list[str]") -> str:
    """What a collapsed setup is called: its signature, or how many it stands for.

    A setup is one configuration until it is swept, and then it is a grid --
    and a grid's title cannot be a signature, because it has several. The count
    is the useful thing at that point: which cell is which is a question for
    Results, where each one is its own row.
    """
    if len(signatures) == 1:
        return signatures[0]
    return f"{len(signatures)} configurations"


def _setup_signatures(signatures: "list[str]", show: int = 3) -> str:
    """The signatures under an open setup, truncated -- a 40-cell grid's worth
    of them would bury the controls that produced it."""
    if not signatures:
        return "No configuration — this setup queues nothing."
    shown = " · ".join(f"`{s}`" for s in signatures[:show])
    rest = len(signatures) - show
    return shown + (f" · …and {rest} more" if rest > 0 else "")


def _render_agent_setups(personality: str, symbols: list[str], renderer) -> list:
    """One agent's setups: an editor each, plus add and remove.

    `renderer` hands back a *list* of configurations per setup -- one for a
    plain setup, one per cell for a swept one -- and they are flattened here,
    because a configuration is the unit everything downstream queues, signs and
    groups by.

    The newest setup is the one left open and the older ones collapse to their
    **signature** -- the same string Results groups runs by, so a glance at the
    collapsed titles answers the only question that matters here, which is
    whether these are actually different configurations. A swept setup has no
    single signature, so it collapses to its count instead (`_setup_title`). The
    title shown is the one built on the previous rerun (an expander's label is
    fixed before its body runs); every widget change reruns the page, so it
    trails an edit by nothing a user can perceive.
    """
    slots = _rule_slots(personality)
    agent = rule_agent(personality)
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
            setup = renderer(symbols, prefix)
            configs.extend(setup)
            signatures = [agent.signature(config) for config in setup]
            st.session_state[f"{prefix}__signature"] = _setup_title(signatures)
            with st.container(horizontal=True, vertical_alignment="center"):
                st.button(
                    "Remove this setup", icon=":material/delete:",
                    key=f"{prefix}_remove", disabled=len(slots) == 1,
                    on_click=_drop_rule_slot, args=(personality, slot),
                    help=None if len(slots) > 1
                    else "The last setup cannot be removed — deselect the agent instead.",
                )
                st.caption(_setup_signatures(signatures))
    st.button(
        f"Add another {label} setup", icon=":material/add:",
        key=f"sim_rule_add_{personality}", on_click=_add_rule_slot, args=(personality,),
        help="Queues a second configuration of the same agent over the same "
             "datasets — a threshold sweep, two instruments, or one rule switched "
             "on and off, run side by side and compared in Results.",
    )
    duplicates = len(configs) - len({agent.signature(config) for config in configs})
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


# SimLab's half of the Apple Trader form (see `agent_stonks.apple_trader_ui`).
# Its wording is about what is worth sweeping and how a setting lands in
# Results; the dashboard's is about the run that is about to start.
_APPLE_TRADER_COPY_FIELDS = dict(
    unavailable_suffix="not in the datasets",
    instrument_help=(
        "The one symbol the run trades.\n\n"
        "- Only symbols a saved model covers are listed — the models are the strategy "
        "here, so the instrument and the model constrain each other.\n"
        "- The same rules over two symbols are two configurations in Results, never "
        "one averaged row."
    ),
    model_help=(
        "Which saved model the run trades on, and with it which rules. Only the "
        "models fitted on the instrument above are listed."
    ),
    sections={
        "dayrange": (
            "Two knobs, and they are the whole strategy.\n\n"
            "- At 9:35 the model forecasts where today's high **H** will land.\n"
            "- The buy rests `buy × unit` below it, the sell `sell × unit` below it.\n"
            "- Each distinct pair is its own configuration in Results — sweeping them is "
            "the intended use.\n"
            "- The defaults are each instrument's best plateau over the notebook's "
            "sessions: still only a month or two of days.\n\n"
            "No entry mode and no probability — neither means anything to a forecast of "
            "the day's range, and the signature leaves them out so a day-range result is "
            "never filed beside a momentum one."
        ),
        "dayrange_breach": (
            "What a session that trades outside the forecast does to it.\n\n"
            "- Each choice is its own configuration in Results.\n"
            "- Queue the same levels three times and let the datasets say whether either "
            "update is worth anything — neither has ever been swept."
        ),
        "dayrange_exits": (
            "The managed exit, all of it measured from the fill.\n\n"
            "- **Stop** — a share of what the trade is playing for.\n"
            "- **Runner threshold** — in the same unit as the levels.\n"
            "- Each switched-on knob is in the signature, so sweeping the stop or the "
            "take queues its own configuration.\n"
            "- 0 switches the stop or the take off. With both off a run files beside the "
            "records made before the exit existed, which replay that way."
        ),
        "dayrange_skip": (
            "Sessions the replay does not trade: no forecast, no order. In the signature as "
            "`skip=...`; a record from before the rule replays with none."
        ),
        "dayrange_breaker": (
            "When to stop for the day.\n\n"
            "- Its own axis on the Tuning tab, and worth one: it changes how many trades "
            "a session takes at all — the number most of the other knobs only nudge."
        ),
    },
    help={
        "model_dayrange": (
            "The predicted high, flat for the session — what the shipped buy/sell "
            "distances were swept against, and what every stored record replays as."
        ),
        "model_highlow": (
            "The day-range rules on HighLow's predicted high and range. The shipped "
            "distances were swept on TimeToChange3's forecast, so sweep them again on "
            "the Tuning tab.\n\n"
            "- Reads Alpaca SIP minute history *before* each replayed day (ALPACA_API_KEY "
            "/ ALPACA_SECRET), cached under `data/highlow/` — the first replay fetches it."
        ),
        "skip_events": (
            "What a replay can know of each, as of that morning:\n\n"
            "- **Day after earnings** — Yahoo's earnings dates, kept under "
            "`data/simlab/session_context/` like the Orchestra selection's.\n"
            "- **CPI**, **jobs report** — the calendar (`agent_stonks/calendars/shock_days.csv`).\n"
            "- **Market / geopolitical shock** — the calendar's hand-marked rows (to 25 Sep "
            "2026, only those known before the open), then the shock flags the live app's "
            "pre-market briefings recorded before 09:35. A day after the calendar that the "
            "live app never briefed counts as an ordinary day."
        ),
        "model_highlow2": (
            "The day-range rules on HighLow2's predicted high and range — on INTC the range "
            "*after 9:35* (its rest-of-session head), so breaches are read on the bars after "
            "it. No distances were ever swept on it, so sweep them on the Tuning tab.\n\n"
            "- Reads every input from Alpaca, whatever the dataset's tape: SIP and IEX "
            "minute history before each replayed day, and that morning's IEX opening and "
            "pre-market (ALPACA_API_KEY / ALPACA_SECRET), cached under `data/highlow2/` — "
            "the first replay fetches it.\n"
            "- Fitted without shock days, so a replay through earnings scores a forecast "
            "of an ordinary day."
        ),
        "model_highlow3m": (
            "The day-range rules on HighLow_3m's predicted high and range — the range "
            "*after 9:33*, forecast at 9:33, so breaches are read on the bars after it. "
            "No distances were ever swept on it, so sweep them on the Tuning tab.\n\n"
            "- Reads every input from Alpaca, whatever the dataset's tape: SIP and IEX "
            "minute history before each replayed day, that morning's IEX opening, and the "
            "option chain's daily bars up to the previous close, plus Yahoo's ^IRX "
            "(ALPACA_API_KEY / ALPACA_SECRET), cached under `data/highlow3m/` — the first "
            "replay fetches it (about a minute)."
        ),
        "model_dayrange_intraday": (
            "The levels rest under the top of the intraday band, so the whole ladder "
            "descends through the morning and rises into the close.\n\n"
            "- Expect fewer fills midday and **shorter holds** — the target descends too, "
            "so a morning position can be closed by a sell level that came down to it.\n"
            "- The swept distances are a poor starting point here (they were picked "
            "against a reference that does not move): sweep them again on the Tuning tab "
            "before reading a first result as the strategy's worth.\n"
            "- It damps the intraday update, since the band is a fraction of the forecast "
            "distance most of the day — expect the two settings to be much less than additive."
        ),
        "buy_k": (
            "Starts at **{buy_k}** on {ticker}, for this model.\n\n"
            "- The pick of this model and instrument's Tuning-tab job: the best total "
            "profit of a buy × sell grid summed over two or three weeks of tape.\n"
            "- A pair never tuned (the intraday-volatility model) starts from notebook 05's "
            "per-ticker sweep instead.\n"
            "- In-sample, and swept under the exit rules of the day each job ran (2026-09-23 "
            "for the day-range forecast on AAPL and GOOGL, 2026-10-04 — with the stop — for "
            "the rest) — "
            "`config.APPLE_TRADER_TUNED_LEVELS` records each pick's weeks and caveats."
        ),
        "sell_k": (
            "Where the exit rests below the same predicted high.\n\n"
            "- **{sell_k}** on {ticker} for this model, from the same tuning job.\n"
            "- The smaller of the two numbers, since it is the higher price.\n"
            "- A day that never reaches it is held to the closing flatten."
        ),
        "level_unit": (
            "The yardstick `buy` and `sell` are counted in.\n\n"
            "- **ADR** — this symbol's trailing 14-day average daily range.\n"
            "- **Predicted range** — `predicted high − predicted low` from the forecast itself.\n"
            "- Sweep it as a pair with the two distances: a k is a different number of "
            "dollars under each, so stored pairs are not comparable across it.\n"
            "- In the signature either way (`A` or `R` on every k).\n"
            "- Everything read against `buy − sell` — the stop, the runner threshold, the "
            "circuit breaker — follows it.\n\n"
            "The shipped distances were swept in ADRs."
        ),
        "breach_update": (
            "Whether the predicted high is held all session or moved when the tape trades "
            "through it — both levels are rebuilt from it each time it moves.\n\n"
            "- **Hold the 9:35 forecast** — it stands all day. What the levels were swept "
            "under, and what every record from before this setting replays as.\n"
            "- **Move to the extreme so far** — the breached side moves to the session's "
            "own extreme and the other side follows by the same amount (never past what "
            "the session printed), so the range keeps its width. Expect more trading "
            "days than holding: the buy level chases the high up. Signs `breach=shift`; "
            "a record signed `breach=extreme` moved only the breached side and replays "
            "that.\n"
            "- **Brownian extension** — past the extreme by ADR × √(session left) ÷ 2. "
            "Wider than the extreme early in the day, converging on it by the close.\n\n"
            "Both moving policies keep the range at the unit's width — the 9:35 predicted "
            "range, or the ADR — and widen it only to hold what the session has printed "
            "(`keep_width` in the signature). A record without it replays as run: *shift* "
            "keeping the width it last had, *Brownian* moving the breached side only.\n\n"
            "In the signature unless held, so each queues as its own configuration — "
            "the only way to find out what it is worth, since the levels were swept with "
            "the forecast fixed."
        ),
        "contain_range": (
            "Whether the forecast is always widened to hold what the session has printed "
            "— `apply_open_constraint`'s 9:35 clip, kept true all day.\n\n"
            "- In the signature as `contain` when on.\n"
            "- Pair it with the breach policy rather than sweeping it alone: with it on, "
            "*hold* moves a breached side to the extreme too, so it only differs from "
            "*move to the extreme* by that policy moving the other side with it."
        ),
        "breach_exit": (
            "Whether a bar through the predicted high closes an open position instead of "
            "moving the forecast.\n\n"
            "- In the signature as `breach_exit` when on.\n"
            "- Under *hold* and *move to the extreme* the sell level is reached on the same "
            "bar anyway, "
            "so this only changes *Brownian* ledgers — banking a settled bet versus riding "
            "on against a target that moved out of the bar's way."
        ),
        "stop_gain_fraction": (
            "Starts at **{stop_gain_fraction}**: the stop tuned with this model and "
            "instrument's levels, or the shared default for a pair never tuned "
            "(Orchestra's pairs share one). A share of the **predicted gain** "
            "(`buy − sell`, what a target exit pays), not a distance of its own.\n\n"
            "- 0.5 risks \\$0.50 for every \\$1.00 the trade plays for, on every "
            "instrument. In ADRs it would not travel: 0.20 ADR was a third of AAPL's "
            "target and a third of GOOGL's — different bets.\n"
            "- Swept against `buy − sell` it is a grid over reward and risk:reward, not "
            "over two distances that happen to interact.\n"
            "- Signs as `stop=E-0.5G`; a record from before carries `stop=E-0.2A` and "
            "replays in its own units.\n"
            "- 0 switches the stop off."
        ),
        "momentum_confirmation_bars": (
            "One look-back, in bars, over which both the buy and the sells read the "
            "momentum behaviour table.\n\n"
            "- **Momentum** — the average move per bar over the period, "
            "`(close − close N bars ago) / N`. **Change** — the average bar-to-bar "
            "change of the 1-bar momentum, `(m1 − m1 N bars ago) / N`.\n"
            "- Each is **neutral** while its size is under 0.1 × the ticker's mean "
            "absolute one-minute move over last week (`abs_mean_minute_momentum`), "
            "else positive or negative.\n"
            "- **Buy** at the buy level only on positive momentum, or neutral momentum "
            "whose change is not negative (flat, or about to rise).\n"
            "- **At or above the sell level**, sell unless momentum is still positive "
            "— then hold on while the price keeps rising. The breach exit likewise.\n"
            "- **Below the sell level**, in profit: sell the *Take on negative "
            "momentum* share while momentum is negative and its change neutral or "
            "negative (still dropping, or dropping faster).\n"
            "- Never gates the stop loss, the runner's breakeven or the closing "
            "flatten.\n"
            "- Until it can be read (the first N + 2 bars, or before last week's "
            "mean move is known) nothing is bought or sold at the levels.\n"
            "- Signs as `confirm=5b` with the take as `take=70%@conf`; a record from "
            "before it replays with it off, under the take it carries (`@neg15b/5b`, "
            "`@fade15b`, `@mom-1`) and any `nofall=` gate.\n"
            "- SimLab measures last week's mean move from the 5 stored sessions "
            "before each replayed day.\n"
            "- Sweepable; 0 switches it off, and with it the take and the runner."
        ),
        "position_pct": (
            "The share of available cash each entry spends. In the signature as `size=`."
        ),
        "take_fraction": (
            "How much of the position a momentum take sells when a runner is kept.\n\n"
            "- In the signature only while the take is on.\n"
            "- Whole shares, rounded down, and at least one."
        ),
        "take_after_minutes": (
            "The momentum take fires only once this many minutes of bars have passed "
            "since the last buy; a buy lower down starts the wait again.\n\n"
            "- In the signature as `take=70%@conf>=15m` while the take is on; a record "
            "from before it replays at 0 (from the first bar).\n"
            "- A record from 2026-09-29 to 10-02 replays the gain gate it was run "
            "under instead, `>=0.2G`: no take until 0.2 of the predicted gain is banked.\n"
            "- 0 takes from the first bar after the buy."
        ),
        "hold_min_gain_k": (
            "Keep a runner only if the sell level is still this far above the fill; "
            "otherwise the take sells everything.\n\n"
            "- The gap is at most `buy − sell`, so on AAPL's default pair 0.30 never "
            "keeps one.\n"
            "- In the signature as `runner>=` while the take is on."
        ),
        "scale_in": (
            "Adds to an open position on the way down while the cash left can pay for "
            "a share: each next buy rests *Next buy step* units lower than the last "
            "— placed only while it is above the stop, which sits under the last "
            "actual fill, and filled only under the last fill. Selling the whole "
            "position resets it to the buy distance.\n\n"
            "- Needs a position size under 100%.\n"
            "- In the signature as `adds=+0.1A` (the step) while it can change a "
            "trade; `adds=half` is the step of 0."
        ),
        "buy_step_k": (
            "How many units lower each buy moves the next one; the whole ladder goes "
            "back to the buy distance once the position is sold out.\n\n"
            "- **0** — half-way to the bottom of the range (reference − 1 unit), the "
            "rule every run before 2026-10-01 replays."
        ),
        "min_win_k": (
            "The session circuit breaker: after a trade closes for no more than this many "
            "× {unit} a share, the run buys nothing else that day.\n\n"
            "- {ticker} starts at **{min_win_k}** — per instrument, since it is only "
            "readable against that symbol's own `buy − sell`.\n"
            "- Judged over the whole position: a momentum take and its runner count as "
            "one trade.\n"
            "- At or over `buy − sell`, every completed trade stands the session down — a "
            "different experiment (one trade a day).\n"
            "- In the signature while it is on, and sweepable."
        ),
    },
)


def _sweep_axes(base: AppleTraderConfig, prefix: str) -> "list[dict]":
    """The axes the user has asked this setup to vary, as `tuning.grid` takes them.

    One control per selected field, and which control depends on what the field
    is. A number gets from/to/step -- the same three boxes the Tuning tab uses,
    so a range means the same thing in both places and `tuning.axis_values`
    decides what it expands to. A rule gets a multiselect of its options, since
    there is no range to walk: "hold the forecast, or move to the extreme" is a
    set, not an interval.

    Each axis says how many values it came to and what the setup above it holds,
    because the single most useful thing to know while building a grid is
    whether the configuration you started from is still in it.
    """
    names = st.multiselect(
        "Vary these settings",
        list(sim_tuning.SWEEPABLE),
        default=[],
        format_func=sim_tuning.sweep_label,
        key=f"{prefix}_sweep_axes",
        help="Each one selected multiplies the setup out: every combination of the "
             "values below is queued as its own experiment and lands in Results as "
             "its own configuration. Leave empty to queue this setup as it stands.",
    )
    axes: list[dict] = []
    for name in names:
        choice = sim_tuning.CHOICES.get(name)
        if choice is not None:
            values = st.multiselect(
                choice.label, list(choice.pickable), default=list(choice.pickable),
                format_func=lambda key, c=choice: c.labels.get(key, key),
                key=f"{prefix}_sweep_{name}",
            )
        else:
            values = _sweep_range(name, prefix)
        if not values:
            st.caption(
                f":material/info: {sim_tuning.sweep_label(name)} has no values selected, "
                "so it is not varied."
            )
            continue
        axes.append({"name": name, "values": values})
        st.caption(
            f"{len(values)} value(s): "
            + ", ".join(sim_tuning.value_label(name, v) for v in values)
            + f" — this setup has {sim_tuning.value_label(name, getattr(base, name))}."
        )
    return axes


def _sweep_range(name: str, prefix: str) -> list:
    """One numeric axis's values from a from/to/step row, or [] if it is invalid.

    The reason it can be invalid at all -- rather than clamped into something
    that runs -- is that a backwards range is a typo and a silently corrected
    typo queues the wrong grid. `tuning.axis_values` raises and the message it
    raises with is the one shown.
    """
    tunable = sim_tuning.TUNABLES[name]
    cast = int if tunable.integer else float
    start0, stop0, step0 = tunable.default_range
    bounds = dict(
        min_value=cast(tunable.minimum), max_value=cast(tunable.maximum),
        step=cast(tunable.step), format=tunable.fmt,
    )
    col_from, col_to, col_step = st.columns(3)
    start = col_from.number_input(
        f"{tunable.label}: from", value=cast(start0),
        key=f"{prefix}_sweep_{name}_from", **bounds
    )
    stop = col_to.number_input(
        "to", value=cast(stop0), key=f"{prefix}_sweep_{name}_to", **bounds
    )
    step = col_step.number_input(
        "step", min_value=cast(tunable.step), max_value=cast(tunable.maximum),
        value=cast(step0), step=cast(tunable.step), format=tunable.fmt,
        key=f"{prefix}_sweep_{name}_step",
    )
    try:
        return sim_tuning.axis_values(name, start, stop, step)
    except ValueError as exc:
        st.error(f":material/error: {exc}")
        return []


def _render_apple_sweep(
    base: AppleTraderConfig, prefix: str
) -> "list[AppleTraderConfig]":
    """One Apple Trader setup expanded over the settings it is asked to vary.

    The setup form above supplies the configuration every combination starts
    from; this crosses it with the axes and hands back one config per cell, each
    of which the tab queues as its own experiment against every selected
    dataset. With nothing varied it is the base configuration alone, which is
    what a setup was before this existed.

    Three things a grid does that a single setup cannot, all reported rather
    than absorbed: a cell can be refused outright (`AppleTraderConfig` says
    why), several cells can collapse into one configuration because the run
    signature does not separate them, and the whole thing can be far larger
    than anyone meant. `tuning.expand` decides the first two; the size guard is
    here because it is a statement about the queue rather than about the grid,
    and it is measured on the grid rather than on what survives it -- a refusal
    has to be cheaper than the thing it refuses.
    """
    axes = _sweep_axes(base, prefix)
    if not axes:
        return [base]

    # Counted before it is built: the guard exists to stop a grid nobody meant,
    # and building one to measure it is the thing being guarded against.
    cells = sim_tuning.cell_count(axes)
    if cells > sim_tuning.MAX_SWEEP_CONFIGS:
        st.error(
            f":material/error: {cells} combinations is over the "
            f"{sim_tuning.MAX_SWEEP_CONFIGS} a single setup may queue — each one is its "
            "own worker process, its own run record and its own row in Results, not a "
            "cell in a heatmap. **This setup is queued as the base configuration alone** "
            "until the grid is narrowed; drop an axis or widen a step. To sweep a "
            "numeric pair this finely, use the Tuning tab."
        )
        return [base]

    configs, refused = sim_tuning.expand(base, axes)
    for overrides, reason in refused[:3]:
        st.caption(
            ":material/block: not a strategy, so it is not queued — "
            + ", ".join(
                f"{sim_tuning.sweep_label(n)} {sim_tuning.value_label(n, v)}"
                for n, v in overrides.items()
            )
            + f": {reason}"
        )
    if len(refused) > 3:
        st.caption(f":material/block: …and {len(refused) - 3} more refused combination(s).")

    collapsed = cells - len(configs) - len(refused)
    if collapsed > 0:
        # The same fact `_render_agent_setups` reports across setups: Results
        # identifies a configuration by its signature, and a field switched off
        # by another (a take fraction with the momentum take at 0) is not in it.
        st.caption(
            f":material/merge: {collapsed} combination(s) sign the same as another and "
            "are queued once — a setting the signature does not carry was varied while "
            "something else had switched it off."
        )
    st.caption(
        f":material/grid_view: **{' × '.join(str(len(a['values'])) for a in axes)} = "
        f"{cells} combination(s)** → {len(configs)} configuration(s), each queued "
        "against every selected dataset."
    )
    return configs


def _render_apple_setup(
    symbols: list[str], prefix: str
) -> "list[AppleTraderConfig]":
    """One Apple Trader entry in the Simulate tab: a base configuration and the
    grid it is swept over.

    Split from `_render_apple_params` because the Tuning tab renders that same
    form as the base of its own grid and must keep getting one configuration
    back; a sweep inside a sweep is not a thing.
    """
    return _render_apple_sweep(_render_apple_params(symbols, prefix), prefix)


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


def _render_orchestra_setup(symbols: list[str], prefix: str) -> list:
    """One Orchestra setup in the Simulate tab: its pairs, each pair's own
    numbers and the rules they share -- the live form's, under SimLab's copy.
    Not swept: an Orchestra is compared against its pairs' Apple Trader runs and
    other Orchestras as whole configurations, each its own setup."""
    fields = dict(_APPLE_TRADER_COPY_FIELDS)
    fields["help"] = {
        **fields.get("help", {}),
        "pairs": (
            "The (ticker, model) pairs Orchestra watches. Every pair's symbol must be "
            "in the datasets. The order is the tie-break when two buys fill on one bar. "
            "Each Orchestra setup is one row in Results; compare it with Apple Trader "
            "setups on the same pairs."
        ),
    }
    fields["sections"] = {
        **fields.get("sections", {}),
        "selection": (
            "Replays Orchestra's 09:34 pick of the day's candidates from what the "
            "morning knew: each symbol's **cached** pre-market briefing (written as of "
            "09:25 that day), its earnings calendar and its opening gap. A replay never "
            "writes a briefing — an LLM's answer is not reproducible — so a day without "
            "one is picked without its bias. Brief the missing days below first."
        ),
    }
    race = apple_trader_ui.orchestra_params(
        symbols, apple_trader_ui.FormCopy(prefix=prefix, **fields),
        # Always at least the default, so a setup made before anything is cached
        # names the briefings it will read once they are written.
        briefing_choices=sim_session.cached_briefing_models() or [sim_session.DEFAULT_BRIEFING],
    )
    if race is not None and race.selection is not None:
        _render_briefing_coverage(race, prefix)
    return [race] if race is not None else []


def _selected_days() -> "dict[str, list[str]]":
    """{feed: [day, ...]} over the datasets and days the Simulate tab has picked."""
    days: "dict[str, list[str]]" = {}
    for name in st.session_state.get("sim_datasets") or []:
        ds = sim_data.get_dataset(name)
        if ds is None:
            continue
        for day in st.session_state.get(f"sim_days_{name}") or (ds.days or [])[:1]:
            bucket = days.setdefault(ds.feed, [])
            if str(day) not in bucket:
                bucket.append(str(day))
    return days


def _render_briefing_coverage(race, prefix: str) -> None:
    """How many of the briefings this setup's replays will read are cached, and
    a button to write the missing ones -- an LLM call per symbol and day, made
    only when asked for."""
    selection = race.selection
    provider, model = selection.briefing_provider, selection.briefing_model
    if not (provider and model):
        return
    days_by_feed = _selected_days()
    days = sorted({d for ds in days_by_feed.values() for d in ds})
    if not days:
        return
    cached, missing = sim_session.briefing_coverage(race.tickers, days, provider, model)
    total = cached + len(missing)
    if not missing:
        st.caption(
            f":material/check_circle: All {total} briefings these replays read are cached "
            f"({provider} / {model})."
        )
        return
    st.caption(
        f":material/info: {cached} of {total} briefings these replays read are cached "
        f"({provider} / {model}); the other {len(missing)} days are picked without a "
        "bias: " + ", ".join(f"{sym} {day}" for sym, day in missing[:8])
        + (" …" if len(missing) > 8 else "")
    )
    api_key = _env_key(provider)
    if st.button(
        f"Brief the {len(missing)} missing session(s) with {provider} / {model}",
        key=f"{prefix}_brief_missing",
        disabled=not api_key,
        help=(
            "One LLM call per symbol and day, as of 09:25 that day, cached and read by "
            "every replay from then on."
            if api_key else f"{ENV_KEYS.get(provider, provider)} is not set."
        ),
    ):
        errors: "dict[str, str]" = {}
        with st.spinner(f"Briefing {len(missing)} session(s)…"):
            for symbol in sorted({sym for sym, _ in missing}):
                want = {
                    feed: [d for d in ds if (symbol, d) in set(missing)]
                    for feed, ds in days_by_feed.items()
                }
                _, failed = sim_session.session_biases(
                    symbol, {f: d for f, d in want.items() if d}, provider, model, api_key,
                    os.getenv("ALPACA_API_KEY", ""), os.getenv("ALPACA_SECRET", ""),
                )
                errors.update({f"{symbol} {day}": err for day, err in failed.items()})
        if errors:
            st.warning("Not briefed: " + "; ".join(f"{k}: {v}" for k, v in errors.items()))
        st.rerun()


def render_simulate_tab() -> None:
    datasets = sim_data.list_datasets()
    if not datasets:
        st.info("Download a dataset first (Datasets tab).")
        return

    st.caption(
        "Pick several agents, models, and datasets — every combination is queued as "
        "its own experiment. A rule-based agent has no model to vary, so its "
        "**setups** take that place: add it as many times as you have "
        "configurations to compare, and each one is queued against every dataset. "
        "Apple Trader can also **vary settings inside one setup** — pick the levels, "
        "the exits or the forecast rules to sweep and the setup multiplies out into "
        "one experiment per combination."
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
                # Configurations rather than setups: one Apple Trader setup with
                # a sweep on it is several, and the multiplication below only
                # adds up if this is what it counts.
                configured = sum(len(rule_setups.get(p, [])) for p in rule_personalities)
                parts.append(
                    f"{configured} rule-based configuration(s) "
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
    an LLM run, an `apple_models` key for a rule run, and the sentinel for a
    rule run that names none. Model names come from the registry, so renaming a
    model there moves the row label with it.

    The names are shortened to what identifies the model -- the breakdown table
    does not wrap.
    """
    if key.startswith(sim_results.LLM_MODEL_PREFIX):
        return f"{key[len(sim_results.LLM_MODEL_PREFIX):]} (LLM)"
    if not key or key == sim_results.UNKNOWN_INSTRUMENT:
        return key or sim_results.UNKNOWN_INSTRUMENT
    # Membership, not `apple_models.get`: that deliberately falls back to the
    # default model for an unknown key so a stored run still replays, which
    # here would print a real model's name over a key that is not one. A key
    # this does not recognise is shown as itself.
    model = apple_models.MODELS.get(key)
    if model is None:
        return key
    label = model.label
    for cut in (" (", " →"):
        label = label.split(cut)[0]
    return label.strip() or key

# Ranking metrics for the top-runs cards, mapped to their `summary` keys.
# "Return" rather than "Best return": which end of the ranking is shown is a
# separate control now, and a label claiming "best" while the worst three are
# on screen would be the one thing on the card that lies.
_TOP_RUN_METRICS = {"Return": "return_pct", "Profit efficiency": "profit_efficiency"}
# Which end of that ranking the cards show.
_TOP_RUN_BEST = "Best"
_TOP_RUN_WORST = "Worst"


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


# The filters, in the order they are laid out, each as
# (filter_options key / filter_runs keyword, label, placeholder, label maker).
#
# The five match the five breakdown dimensions one for one, and deliberately
# use the same names: a filter and the table row it narrows to should not be
# called different things. "LLM / rules" is what the filter used to call
# "Models" -- renamed for the same reason the breakdown control was, since
# "Models" sitting next to "ML model" reads as the same axis twice and is not.
#
# There is no strategy filter: a strategy is not a stored dimension of a run.
# Which one an Apple Trader run used *is* the model it names, so the ML model
# filter already selects it, and adding a control that only restates another
# would give the page a knob that never narrows anything on its own.
_RUN_FILTERS: "tuple[tuple[str, str, str, object], ...]" = (
    ("datasets", "Datasets", "All datasets", None),
    ("agents", "Agents", "All agents", _agent_label),
    ("instruments", "Instruments", "All instruments", None),
    ("ml_models", "ML models", "All ML models", _ml_model_label),
    # No shortening here, unlike the top-run cards: a rule agent's whole rule
    # set is its model string, and two sets can agree for a hundred characters
    # before they differ. An abbreviated option list would offer the reader two
    # identical-looking choices that are not the same runs.
    ("models", "LLM / rules", "All models", None),
)


def _render_run_filters(runs: list[dict], key_prefix: str) -> list[dict]:
    """Dataset / agent / instrument / ML model / LLM filters over the stored
    runs. Selecting nothing in a filter leaves that dimension unrestricted, so
    the default view is all runs, and the dimensions combine with AND.
    Everything below (breakdown, charts, run picker) works off the result.
    Summary and Results each render their own copy -- hence the key prefix --
    so filtering one tab doesn't silently reshape the other.

    Options are the values actually present in the stored runs, so a filter
    never offers a choice that would empty the page on its own. They are stored
    keys rendered through the breakdown's own labellers, which is what lets a
    renamed agent or model move an option without stranding a stored run.
    """
    options = sim_results.filter_options(runs)
    selections: dict[str, list[str]] = {}
    # Three then two rather than one row of five: a multiselect a fifth of the
    # page wide truncates its own chips.
    columns = list(st.columns(3)) + list(st.columns(2))
    for column, (dimension, label, placeholder, label_of) in zip(columns, _RUN_FILTERS):
        selections[dimension] = column.multiselect(
            label, options[dimension], key=f"{key_prefix}_filter_{dimension}",
            placeholder=placeholder,
            **({"format_func": label_of} if label_of else {}),
        )
    filtered = sim_results.filter_runs(runs, **selections)
    if any(selections.values()):
        st.caption(f"Showing {len(filtered)} of {len(runs)} runs.")
    return filtered


def _short_model(key: str, limit: int = 46) -> str:
    """Rule agents encode their entire rule set in the model string, which would
    otherwise stretch one card far past the others. The full string is still in
    the breakdown table and on the run itself."""
    return key if len(key) <= limit else key[: limit - 1].rstrip() + "…"


def _open_run_in_results(run_id: str) -> None:
    """Point the Results tab at `run_id` and switch to it.

    A callback rather than a branch after the button, for two reasons that both
    come down to *when* it runs. The active tab lives in the tab widget's own
    session-state key, and Streamlit refuses a write to a widget's key once that
    widget has been instantiated -- which, by the time a card inside a tab is
    drawn, it has. `on_click` runs before the script re-executes, while the key
    is still writable.

    The filters are cleared for the same reason they exist: Results keeps its
    own set, separate from Summary's, so a run Summary can see may be one
    Results has filtered out -- and its picker would then quietly open a
    different run. Clearing them is the reading of "open this run" that cannot
    be wrong.
    """
    st.session_state["last_run_id"] = run_id
    for dimension, *_ in _RUN_FILTERS:
        st.session_state[f"results_filter_{dimension}"] = []
    st.session_state[TAB_STATE_KEY] = TAB_RESULTS


def _render_top_runs(runs: list[dict]) -> None:
    """The three best -- or worst -- single runs under the filters, on whichever
    metric is picked.

    Return and profit efficiency disagree often (a big return on an easy tape
    can be a worse trade than a small one on a flat tape), so both are always
    shown and only the ranking changes. The worst end is worth the same glance
    as the best: a configuration that loses reliably is as much a finding as one
    that wins, and it is the end nobody scrolls a table to find.

    Each card opens its run in the Results tab, which is where the equity curve,
    the decision ledger and the judge's report are -- a card is a pointer, not a
    destination.
    """
    direction = st.segmented_control(
        "Show", [_TOP_RUN_BEST, _TOP_RUN_WORST], default=_TOP_RUN_BEST,
        key="summary_top_direction",
    ) or _TOP_RUN_BEST
    worst = direction == _TOP_RUN_WORST
    st.markdown(f"##### {'Worst' if worst else 'Top'} runs")
    metric_label = st.segmented_control(
        # Keyed apart from the control this replaces, whose stored value is a
        # label ("Best return") that is no longer one of the options.
        "Rank by", list(_TOP_RUN_METRICS), default="Return",
        key="summary_top_metric_v2",
    ) or "Return"
    metric = _TOP_RUN_METRICS[metric_label]
    top = sim_results.top_runs(runs, by=metric, limit=3, worst=worst)
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
            st.button(
                "Open in Results", icon=":material/open_in_new:",
                key=f"summary_open_{row['run_id']}", width="stretch",
                on_click=_open_run_in_results, args=(row["run_id"],),
                help="Switches to the Results tab on this run, clearing the "
                     "filters there so it is reachable.",
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
            for key in ["last_run_id"] + [
                f"{prefix}_filter_{dimension}"
                for prefix in ("results", "summary")
                for dimension, *_ in _RUN_FILTERS
            ]:
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
            "Trader names one model and that model *is* its strategy. An LLM agent loads no saved model at all, so those "
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
# Tab — drift
# ---------------------------------------------------------------------------

_DRIFT_GROUP_LABELS = {sim_drift.DAY: "Day", sim_drift.WEEK: "Week"}
# One colour per instrument, assigned from the chosen list rather than per
# chart, so a symbol keeps its colour all the way down the tab and the eye can
# follow it from one model's chart into the next.
_DRIFT_SERIES_COLORS = (
    PALETTE["accent"], PALETTE["orange"], PALETTE["up"],
    "#c084fc", "#f472b6", "#facc15",
)
# What one press of Update covered: the tape, the instruments, and the stored
# bars behind them. Any of the three changing would make the numbers on screen
# stale, and recomputing them means loading PyTorch and LightGBM -- so the tab
# asks for the button again rather than doing that on a rerun nobody asked for.
_DRIFT_SCOPE_KEY = "drift_scope"
_DRIFT_UPDATED_KEY = "drift_updated_at"
# Above this many sessions in one series, the dots go and the line stays --
# `intraday_vol` is scored on the whole daily history, which is years of it.
_DRIFT_MARKER_LIMIT = 60


@st.cache_data(show_spinner=False)
def _drift_result(model_key: str, ticker: str, feed: str, signature: tuple) -> dict:
    """One model's per-session metrics on one instrument's stored tape.

    Keyed on the stored bars and the saved model file (`drift.signature`), so a
    new download or a retrain recomputes and nothing else does -- which is what
    makes Update cheap the second time: the pairs where neither has moved come
    straight back out. Scoring loads the model, PyTorch for the day-range
    bundle, which is why the tab computes on request rather than whenever
    SimLab reruns.
    """
    return sim_drift.MODELS[model_key].evaluate(ticker, feed)


def _drift_color(ticker: str, tickers: "list[str]") -> str:
    index = tickers.index(ticker) if ticker in tickers else 0
    return _DRIFT_SERIES_COLORS[index % len(_DRIFT_SERIES_COLORS)]


def _drift_scope(feed: str, symbols: "list[str]") -> tuple:
    """Everything an update would be computed from, as one comparable value.

    The saved model files are in it as well as the stored bars, so retraining a
    bundle marks the tab stale on its own -- the case the old single-model view
    needed an explicit "Recompute" button for.
    """
    return (feed, tuple(symbols), tuple(
        sim_drift.signature(model_key, ticker, feed)
        for model_key, ticker in sim_drift.pairs(feed, symbols)
    ))


def _drift_update(feed: str, symbols: "list[str]") -> None:
    """Score every model on every chosen instrument, with the progress visible.

    Every pair goes through `_drift_result`, so this is one cache fill rather
    than a separate code path -- a pair already scored on unchanged bars costs
    a dictionary lookup, and only the new work shows on the bar.
    """
    todo = sim_drift.pairs(feed, symbols)
    progress = st.progress(0.0, text="Scoring…")
    for done, (model_key, ticker) in enumerate(todo):
        model = sim_drift.MODELS[model_key]
        progress.progress(done / len(todo), text=f"{model.label} · {ticker}")
        _drift_result(model_key, ticker, feed, sim_drift.signature(model_key, ticker, feed))
    progress.empty()
    st.session_state[_DRIFT_SCOPE_KEY] = _drift_scope(feed, symbols)
    st.session_state[_DRIFT_UPDATED_KEY] = datetime.now().strftime("%H:%M:%S")


def _drift_scoreboard(feed: str, symbols: "list[str]") -> "list[dict]":
    """Every (model, instrument) pair, with its rows, its cutoffs and its notes.

    Read out of the cache Update filled, so this runs on every rerun of the tab
    without rescoring anything.
    """
    board = []
    for model_key, ticker in sim_drift.pairs(feed, symbols):
        model = sim_drift.MODELS[model_key]
        result = _drift_result(
            model_key, ticker, feed, sim_drift.signature(model_key, ticker, feed)
        )
        training = model.training(ticker)
        board.append({
            "model": model,
            "ticker": ticker,
            "rows": result.get("rows") or [],
            "notes": result.get("notes") or [],
            "cutoffs": training.get("cutoffs") or [],
            "references": training.get("references") or [],
        })
    return board


def _drift_signed(metric, value: "float | None") -> str:
    """A change in a metric's own units, always with its sign."""
    return "—" if value is None else metric.fmt.replace("%", "%+", 1) % value


def _drift_verdict(metric, test) -> str:
    """One test as a sentence: whether it moved, which way, and by how much.

    "Worse" rather than "up" wherever the metric has a good direction, because
    up is bad for an error and good for a correlation and nobody should have to
    remember which column they are in. A bias has no good direction -- only
    zero is -- so it only ever *changed*.
    """
    if test.p is None:
        return f"Not tested — {test.note}"
    if not test.significant:
        return f"No significant change (p {test.p:.2f})"
    worse = sim_drift.worsened(metric, test.statistic)
    headline = {True: "Worse", False: "Better", None: "Changed"}[worse]
    moved = "rising" if test.statistic > 0 else "falling"
    effect = (
        f", {_drift_signed(metric, test.effect)} {test.effect_label}"
        if test.effect is not None else ""
    )
    return f"{headline} — {moved}{effect} (p {test.p:.3f})"


def _drift_row(entry: dict, metric) -> dict:
    """One (model, instrument) pair as a table row, for one metric.

    The two means share a cell as `inside → after`, which is the comparison
    anyone reads them for and costs one column instead of three; where a store
    sits entirely on one side of a cutoff -- which two of the three models do --
    the empty half says so at a glance.
    """
    assessment = sim_drift.assess(entry["rows"], metric, entry["cutoffs"])
    inside, after = assessment["parts"]["inside"], assessment["parts"]["after"]

    def mean(part: dict) -> str:
        return metric.fmt % part["mean"] if part["n"] else "—"

    return {
        # Without the project in brackets: the section below says which model
        # this is at length, and here it costs the column its readable width.
        "Model": entry["model"].label.split(" (")[0],
        "Instrument": entry["ticker"],
        "Metric": metric.name,
        "Sessions": assessment["n"],
        "In training → after": f"{mean(inside)} → {mean(after)}",
        "Trend over time": _drift_verdict(metric, assessment["trend"]),
        "Across the cutoff": _drift_verdict(metric, assessment["split"]),
    }


def _drift_table(rows: "list[dict]", drop: "tuple[str, ...]" = ()) -> None:
    """The scoreboard, with the two verdict columns given the room to be read.

    Widths set rather than left to the grid: the verdicts are sentences and the
    numbers are four characters, and the browser's guess gives them equal thirds
    and truncates the only two columns anyone opened the tab for.
    """
    text = st.column_config.TextColumn
    st.dataframe(
        pd.DataFrame([{k: v for k, v in r.items() if k not in drop} for r in rows]),
        hide_index=True, width="stretch", column_config={
            "Model": text("Model", width="medium"),
            "Instrument": text("Instrument", width="small"),
            "Metric": text("Metric", width="medium"),
            "Sessions": st.column_config.NumberColumn("Sessions", width="small"),
            "In training → after": text(
                "In training → after", width="medium",
                help="The metric's mean on the sessions the model saw in training and on "
                     "the ones after its last cutoff. One side empty means the whole "
                     "store falls on the other.",
            ),
            "Trend over time": text(
                "Trend over time", width="large",
                help="Mann–Kendall on the per-session values: is the metric going "
                     "anywhere over the whole scored stretch?",
            ),
            "Across the cutoff": text(
                "Across the cutoff", width="large",
                help="Mann–Whitney U on the sessions after the model's last training "
                     "cutoff against the ones inside it: is it worse where it is blind?",
            ),
        },
    )


def render_drift_tab() -> None:
    st.caption(
        "How every saved model's accuracy moves over time on the sessions SimLab has "
        "stored, session by session or week by week, against the dates its training data "
        "ended. Each session is scored the way a live run would have seen it — daily "
        "history strictly before the day, that day's opening print and minutes — and "
        "compared with what the day actually did. A model is only really being tested to "
        "the right of its last cutoff."
    )

    col_feed, col_symbols, col_group = st.columns([1, 3, 1], vertical_alignment="bottom")
    feed = col_feed.selectbox(
        "Tape", list(sim_data.FEEDS), key="drift_feed",
        help="Which stored bars to score on. The same day on two tapes is two sets of bars.",
    )
    stored = sim_drift.stored_symbols(feed)
    if not stored:
        st.info(
            f"Nothing stored on the `{feed}` tape. Download a dataset in the Datasets tab."
        )
        return
    symbols = col_symbols.multiselect(
        "Instruments", stored, default=sim_drift.default_symbols(feed),
        key=f"drift_symbols_{feed}",
        help="Every model is scored on every instrument it exists for. The default is the "
             "symbols a per-instrument model was actually fitted for; the open-profile "
             "pack transfers, so it can be scored on any of them.",
    )
    by = col_group.segmented_control(
        "Group by", list(_DRIFT_GROUP_LABELS), format_func=_DRIFT_GROUP_LABELS.get,
        default=sim_drift.DAY, key="drift_group",
    ) or sim_drift.DAY
    if not symbols:
        st.info("Choose at least one instrument.")
        return

    todo = sim_drift.pairs(feed, symbols)
    scope = _drift_scope(feed, symbols)
    current = st.session_state.get(_DRIFT_SCOPE_KEY) == scope
    col_button, col_status = st.columns([1, 4], vertical_alignment="bottom")
    if col_button.button(
        "Update", type="primary" if not current else "secondary",
        icon=":material/refresh:", key="drift_update", width="stretch",
        help="Score every model on every chosen instrument. Pairs whose stored bars have "
             "not changed and whose model file has not been retrained come back out of "
             "the cache.",
    ):
        _drift_update(feed, symbols)
        st.rerun()
    if not current:
        col_status.caption(
            f":material/info: {len(todo)} model–instrument pair"
            f"{'s' if len(todo) != 1 else ''} to score across {len(sim_drift.MODELS)} models"
            + (
                " — the stored bars or a model file have changed since the last update."
                if st.session_state.get(_DRIFT_SCOPE_KEY) is not None
                else ". Scoring loads the models, so it runs when asked."
            )
        )
        return
    col_status.caption(
        f":material/check_circle: Updated at {st.session_state.get(_DRIFT_UPDATED_KEY, '—')} · "
        f"{len(todo)} model–instrument pair{'s' if len(todo) != 1 else ''} · cached until the "
        "stored bars or a model file change"
    )

    board = _drift_scoreboard(feed, symbols)
    st.markdown("#### Overview")
    st.caption(
        "Each model by the one number the ML Models tab grades it on, measured again on "
        "the stored sessions. Two rank tests on the per-session values ask whether it has "
        "actually moved: a Mann–Kendall trend over the whole stretch, and a Mann–Whitney U "
        "either side of the model's last training cutoff. Significant means p < "
        f"{sim_drift.ALPHA:g}. The metrics are **not comparable across rows** — a MAE as "
        "a share of ADR and an EMD in bps answer different questions."
    )
    scored = [
        _drift_row(entry, entry["model"].headline_metric)
        for entry in board if entry["rows"]
    ]
    if scored:
        _drift_table(scored)
    else:
        st.warning(
            f"Nothing on the `{feed}` tape could be scored. Each model says why in its "
            "section below — usually a missing model file or a missing dependency."
        )

    for model_key, model in sim_drift.MODELS.items():
        entries = [e for e in board if e["model"].key == model_key]
        st.markdown(f"#### {model.label}")
        st.caption(model.summary)
        st.caption(f":material/neurology: On the ML Models tab — {model.catalogue_metric}")
        if not entries:
            # Headed and explained rather than dropped: a model quietly missing
            # from the tab reads as a model that has nothing wrong with it.
            st.info(
                f"None of the chosen instruments has a {model.label} model"
                + (f" — it exists for {', '.join(model.tickers)}." if model.tickers else ".")
            )
            continue
        metrics = {m.key: m for m in model.metrics}
        metric = metrics[st.selectbox(
            "Metric", list(metrics), format_func=lambda k: metrics[k].label,
            key=f"drift_metric_{model_key}",
        )]
        if metric.help:
            st.caption(metric.help)
        charted = [e for e in entries if any(r.get(metric.key) is not None for r in e["rows"])]
        if not charted:
            st.warning(
                f"No stored session on `{feed}` could be scored for this metric."
                + ("".join(f" {n}" for e in entries for n in e["notes"]))
            )
            continue
        st.plotly_chart(
            _drift_chart(charted, metric, by, symbols),
            key=f"drift_chart_{model_key}_{metric.key}_{by}_{feed}",
        )
        _drift_references(charted, metric)
        if metric.key != model.headline:
            _drift_table([_drift_row(e, metric) for e in charted], drop=("Model",))
        with st.expander(f"{model.label} — session detail"):
            _drift_detail(charted, metric, by)


def _drift_references(entries: "list[dict]", metric) -> None:
    """What the model's own evaluation said, in the metric's units.

    The chart draws these as dotted lines and the line cannot carry the
    sentence explaining what window the number came off, which is the part that
    decides whether it is a fair thing to be measured against.
    """
    # Grouped on the label and the sentence, not on the value: each ticker's
    # day-range bundle was graded on its own test window, so three references
    # differ only in their number and repeating the sentence three times is how
    # a useful caption becomes an unread one.
    seen: "dict[tuple, list]" = {}
    for entry in entries:
        for ref in (r for r in entry["references"] if r.metric == metric.key):
            seen.setdefault((ref.label, ref.note), []).append((entry["ticker"], ref.value))
    if not seen:
        return
    parts = []
    for (label, note), found in seen.items():
        shared = len({value for _, value in found}) == 1 and len(found) == len(entries)
        values = (
            metric.fmt % found[0][1] if shared
            else ", ".join(f"{ticker} {metric.fmt % value}" for ticker, value in found)
        )
        parts.append(f"**{label}** {values}" + (f" — {note}" if note else ""))
    st.caption(":material/flag: " + " · ".join(parts))


def _drift_detail(entries: "list[dict]", metric, by: str) -> None:
    """Per-instrument: what was scored, what could not be, and the period means."""
    for entry in entries:
        st.markdown(f"**{entry['ticker']}**")
        for note in entry["notes"]:
            st.caption(f":material/info: {note}")
        if entry["cutoffs"]:
            st.caption(" · ".join(
                f"**{c.label}** {c.date}" + (f" — {c.note}" if c.note else "")
                for c in sorted(entry["cutoffs"], key=lambda c: c.date)
            ))
        groups = sim_drift.aggregate(entry["rows"], metric.key, by)
        number = st.column_config.NumberColumn
        st.dataframe(
            pd.DataFrame([{
                "period": g["label"],
                "sessions": g["n"],
                "mean": g["mean"],
                "min": g["min"],
                "max": g["max"],
                "data": (
                    "in training" if sim_drift.in_training(g["end"], entry["cutoffs"])
                    else "after cutoff"
                ),
            } for g in groups]),
            hide_index=True, width="stretch", column_config={
                "period": _DRIFT_GROUP_LABELS[by],
                "sessions": "Sessions",
                "mean": number(f"Mean {metric.unit}".strip(), format=metric.fmt),
                "min": number("Min", format=metric.fmt),
                "max": number("Max", format=metric.fmt),
                "data": "Data",
            },
        )


def _drift_chart(entries: "list[dict]", metric, by: str, tickers: "list[str]") -> go.Figure:
    """One model's metric over time, one instrument per colour.

    A chart per model rather than one for everything: within a model the
    instruments share units and are worth reading against each other, and across
    models they are a MAE as a share of ADR and an EMD in bps on the same axis,
    which would be a chart that lies.

    Two things the single-instrument version did not have to think about.
    Instruments that stopped learning on the same day -- which is every one of
    them for the two models fitted on a shared window -- get *one* cutoff line
    naming them all, because three dashed lines on the same date and three
    labels in the same place read as neither. And the training stretch is shaded
    only when there is one instrument on the chart, since overlapping
    rectangles shade nothing in particular.
    """
    fig = go.Figure()
    single = len(entries) == 1
    dates: "list[str]" = []
    # Both collected rather than drawn in the loop, and both keyed on the thing
    # itself: the open-profile pack is one model graded once, so three
    # instruments carry the same two reference numbers and the same cutoff, and
    # drawing them per instrument means six lines where there are two facts.
    marks: "dict[str, dict]" = {}          # cutoff date -> instruments, label
    refs: "dict[tuple, dict]" = {}         # (label, value) -> instruments
    for entry in entries:
        ticker, colour = entry["ticker"], _drift_color(entry["ticker"], tickers)
        rows = sorted(
            (r for r in entry["rows"] if r.get(metric.key) is not None),
            key=lambda r: r["date"],
        )
        own = [r["date"] for r in rows]
        dates += own
        weekly = by == sim_drift.WEEK
        # Three years of daily bars with a dot on every one of them is a
        # hairball, and the dots carry nothing the line does not.
        crowded = len(own) > _DRIFT_MARKER_LIMIT
        fig.add_trace(go.Scatter(
            x=own, y=[r[metric.key] for r in rows],
            mode="markers" if weekly else ("lines" if crowded else "lines+markers"),
            name=f"{ticker} · session" if weekly else ticker, legendgroup=ticker,
            opacity=0.3 if weekly else (0.75 if crowded else 1.0),
            marker=dict(size=4 if crowded else 6, color=colour),
            line=dict(color=colour, width=1.0 if crowded else 1.5),
            hovertemplate=(
                f"<b>{ticker}</b> %{{x|%a %d %b %Y}}<br>{metric.label}: "
                f"%{{y:{metric.hover}}}<extra></extra>"
            ),
        ))
        if weekly:
            groups = sim_drift.aggregate(rows, metric.key, by)
            fig.add_trace(go.Scatter(
                x=[g["start"] for g in groups], y=[g["mean"] for g in groups],
                mode="lines+markers", name=f"{ticker} · weekly mean", legendgroup=ticker,
                marker=dict(size=8, color=colour), line=dict(color=colour, width=2.5),
                customdata=[[g["label"], g["n"]] for g in groups],
                hovertemplate=(
                    f"<b>{ticker}</b> %{{customdata[0]}}<br>{metric.label}: "
                    f"%{{y:{metric.hover}}}<br>%{{customdata[1]}} sessions<extra></extra>"
                ),
            ))
        for ref in (r for r in entry["references"] if r.metric == metric.key):
            key = (ref.label, round(ref.value, 9))
            refs.setdefault(key, {"value": ref.value, "label": ref.label, "tickers": []})
            refs[key]["tickers"].append(ticker)
        cutoffs = sorted(entry["cutoffs"], key=lambda c: c.date)
        if single and cutoffs and own:
            fig.add_shape(
                type="rect", xref="x", yref="paper", y0=0, y1=1, line_width=0,
                x0=min([*own, cutoffs[0].date]), x1=cutoffs[-1].date,
                fillcolor="rgba(148,163,184,0.10)", layer="below",
            )
        for cutoff in (cutoffs if single else cutoffs[-1:]):
            mark = marks.setdefault(cutoff.date, {"tickers": [], "label": cutoff.label})
            mark["tickers"].append(ticker)
    first = min(dates)
    for i, (day, mark) in enumerate(sorted(marks.items())):
        named = mark["tickers"]
        colour = (
            _drift_color(named[0], tickers) if not single and len(named) == 1
            else PALETTE["down"]
        )
        fig.add_shape(
            type="line", xref="x", yref="paper", x0=day, x1=day, y0=0, y1=1,
            line=dict(color=colour, dash="dash", width=1.5),
        )
        fig.add_annotation(
            xref="x", yref="paper", x=day, y=1.0 - 0.07 * i, showarrow=False,
            # A cutoff before the first scored session sits on the left edge,
            # where a right-anchored label runs off the chart. Hang it off the
            # inside of its line instead.
            xanchor="left" if day <= first else "right",
            yanchor="top", font=dict(color=colour, size=11),
            text=(
                f" {mark['label']} {day} " if single
                else f" {', '.join(named)} trained through {day} "
            ),
        )
    for i, ref in enumerate(sorted(refs.values(), key=lambda r: r["value"])):
        named = ref["tickers"]
        shared = single or len(named) == len(entries)
        colour = (
            PALETTE["muted"] if len(named) > 1 and not single
            else _drift_color(named[0], tickers)
        )
        fig.add_hline(y=ref["value"], line=dict(color=colour, dash="dot", width=1.5))
        fig.add_annotation(
            # Staggered leftwards rather than all at x=1: two references within
            # a rounding error of each other land on the same pixel otherwise,
            # and a model graded against a baseline always has two.
            xref="paper", x=1.0 - 0.15 * i, y=ref["value"], showarrow=False,
            xanchor="right", yanchor="bottom", font=dict(color=colour, size=11),
            text=(ref["label"] if shared else f"{', '.join(named)} · {ref['label']}")
                 + " " + metric.fmt % ref["value"],
        )
    # Fitted to what is drawn: autorange pads a date axis by weeks, which put an
    # empty month in front of the first cutoff.
    pad = timedelta(days=3)
    fig.update_xaxes(
        type="date", rangebreaks=[dict(bounds=["sat", "mon"])],
        range=[
            (date.fromisoformat(min([*dates, *marks])) - pad).isoformat(),
            (date.fromisoformat(max(dates)) + pad).isoformat(),
        ],
    )
    fig.update_yaxes(title=f"{metric.label} {metric.unit}".strip())
    fig.update_layout(title=(
        f"{entries[0]['ticker']} · " if single else ""
    ) + f"{metric.label} by {_DRIFT_GROUP_LABELS[by].lower()}")
    return _chart_layout(fig, height=440)


# ---------------------------------------------------------------------------
# Tab — tuning
# ---------------------------------------------------------------------------

_TUNE_PREFIX = "tune"
_PICK_LABELS = {
    sim_tuning.PICK_MAX: "Highest single cell",
    sim_tuning.PICK_PLATEAU: "Middle of the best plateau",
}


def _tuning_param_label(name: str) -> str:
    """An axis's name without its unit, for tables, titles and captions."""
    return sim_tuning.axis_title(name)


def _tuning_value(name: str, value) -> str:
    """One value on an axis: a formatted number, or a rule under its own name."""
    return sim_tuning.value_label(name, value)


def _tuning_values_text(values: "dict | None") -> str:
    """One cell's settings, for the summary table and the pick's caption.

    In the chart's tick form rather than the form's: both of the places this
    appears sit next to a heatmap axis or a `config_signature`, and a cell that
    the grid calls `brownian` and the table calls "Brownian extension,
    volatility implied by ADR" reads as two different things.
    """
    if not values:
        return "—"
    return ", ".join(
        f"{_tuning_param_label(n)} {sim_tuning.axis_tick(n, v)}" for n, v in values.items()
    )


def _tuning_metric_text(metric: str, value) -> str:
    if value is None:
        return ""
    if metric in ("profit", "worst_day"):
        return f"{value:+,.0f}"
    if metric == "return_pct":
        return f"{value:+.2f}%"
    return f"{value:g}"


# Two settings a derived grid can differ by that are not numbers, in the short
# form `config_signature` writes them as — `breach=brownian`, `levels=intraday`
# — since that is what they are called everywhere else in this app.
_SHORT_FIELD_LABELS = {
    "breach_update": "breach",
    "level_source": "levels",
    "level_unit": "unit",
    "contain_range": "contain",
    "breach_exit": "breach exit",
}


def _field_text(name: str, value) -> str:
    """One configuration field as `label value`, compactly enough for a label."""
    if name in sim_tuning.TUNABLES:
        return f"{_tuning_param_label(name)} {_tuning_value(name, value)}"
    return f"{_SHORT_FIELD_LABELS.get(name, name)} {value}"


def _derived_discriminators(jobs: list[dict]) -> dict[str, str]:
    """For each derived grid, the shared settings that tell it from its look-alikes.

    Several grids over the same instrument, axes and dataset are common and
    read identically otherwise — the run store holds one momentum sweep per
    breach policy, say. What separates them is a setting none of them sweeps,
    so it appears nowhere else in the label; this diffs the base configurations
    within each look-alike group and names only the fields that actually move.

    The base configurations are diffed rather than their signatures because a
    signature omits a setting that is switched off, and "off" is exactly the
    variant that would otherwise have nothing to show.
    """
    groups: dict[tuple, list[dict]] = {}
    for job in jobs:
        if not sim_tuning.is_derived(job):
            continue
        spec = job["spec"]
        key = (
            spec["base"]["ticker"],
            tuple(a["name"] for a in spec["axes"]),
            tuple(sim_tuning.dataset_names(spec)),
        )
        groups.setdefault(key, []).append(job)
    text: dict[str, str] = {}
    for group in groups.values():
        if len(group) < 2:
            continue
        axes = {a["name"] for a in group[0]["spec"]["axes"]}
        moving = [
            name
            for name in group[0]["spec"]["base"]
            if name not in axes
            and len({json.dumps(j["spec"]["base"].get(name), default=str) for j in group}) > 1
        ]
        for job in group:
            text[job["job_id"]] = ", ".join(
                _field_text(name, job["spec"]["base"].get(name)) for name in moving
            )
    return text


def _tuning_job_label(job: dict, markdown: bool = True, detail: str = "") -> str:
    spec = job["spec"]
    params = " × ".join(_tuning_param_label(a["name"]) for a in spec["axes"])
    ticker = spec["base"]["ticker"]
    # The model is part of what a grid is about -- the same levels mean
    # different prices under the day-range and the high/low forecast -- and the
    # picker otherwise shows two jobs of one ticker as the same line.
    model = _apple_model_label(spec["base"].get("model_key", ""))
    head = f"{f'**{ticker}**' if markdown else ticker} · {model} · {params}"
    started = (job.get("created_at") or "")[:16].replace("T", " ")
    # No dataset names: a job's datasets change as they are added and dropped,
    # and the results below list them anyway.
    if sim_tuning.is_derived(job):
        filled = len(job["cells"].get(sim_tuning.dataset_names(spec)[0]) or ())
        total = len(sim_tuning.grid(spec["axes"]))
        route = f"{filled}/{total} cells from stored runs"
        return " · ".join(
            part for part in (head, route, detail, f"newest run {started} UTC") if part
        )
    return f"{head} · {started} UTC"


def render_tuning_tab() -> None:
    st.caption(
        "Sweep a grid of Apple Trader's parameters over one or more datasets — a heatmap "
        "per dataset, then their sum — and pick the combination with the best total "
        "profit. Every cell is a full replay through the same engine and trader as "
        "Simulate — market fills, the managed exit, the flatten — so a cell's number is "
        "what a Simulate run of that configuration would show. A finished job takes a "
        "new dataset (a new week, say) with the same settings, and datasets can be "
        "dropped again."
    )
    _render_tuning_jobs()
    submitted = sim_tuning.list_jobs()
    with st.expander("New tuning job", icon=":material/tune:", expanded=not submitted):
        _render_tuning_form()
    # Derived grids are offered beside the submitted jobs rather than in their
    # own section: they answer the same question in the same shape, and the
    # useful comparison is between a grid somebody swept and a grid the store
    # already held -- which is hard to make across two lists.
    jobs = submitted + _derived_jobs(sim_results.store_signature())
    if jobs:
        st.divider()
        _render_tuning_results(jobs)


@st.cache_data(show_spinner=False)
def _derived_jobs(signature: tuple) -> list[dict]:
    # `signature` is the cache key and is deliberately unused -- see `_load_runs`.
    # Cached because assembling these scores every run that lands on a grid, and
    # the tab re-renders on every widget touch.
    return sim_tuning.derived_jobs(_load_runs(signature))


def _render_tuning_jobs() -> None:
    # Refreshes itself while a job runs, like the experiment pipeline, and
    # renders statically once nothing is running.
    auto_refresh = any(j["status"] == sim_tuning.RUNNING for j in sim_tuning.list_jobs())
    st.fragment(run_every=3.0 if auto_refresh else None)(
        lambda: _render_tuning_jobs_body(auto_refresh)
    )()


def _render_tuning_jobs_body(auto_refresh: bool) -> None:
    active = [j for j in sim_tuning.list_jobs() if j["status"] == sim_tuning.RUNNING]
    if auto_refresh and not active:
        st.rerun()  # the last job just finished -- redraw its results below
    if not active:
        return
    st.markdown("##### :material/grid_on: Running")
    for job in sorted(active, key=lambda j: _tuning_job_label(j, markdown=False).casefold()):
        with st.container(border=True):
            st.markdown(_tuning_job_label(job))
            done = int(job["progress"]["done"])
            total = max(int(job["progress"]["total"]), 1)
            reused = int(job["progress"].get("reused") or 0)
            st.progress(
                min(done / total, 1.0),
                text=f"{done} of {total} replays"
                + (f" ({reused} reused from stored runs)" if reused else ""),
            )
            with st.container(horizontal=True, vertical_alignment="center"):
                line = sim_tuning.last_log_line(job["job_id"])
                if line:
                    st.caption(line)
                if st.button(
                    "Stop", key=f"tune_stop_{job['job_id']}", icon=":material/stop_circle:",
                    help="Kill the worker and its pool. The cells finished so far are kept.",
                ):
                    sim_tuning.stop(job["job_id"])
                    st.toast("Tuning job stopped", icon=":material/stop_circle:")
                    st.rerun()


def _tuning_dataset_spec(dataset) -> dict:
    return {
        "name": dataset.name,
        "days": list(dataset.days),
        "feed": dataset.feed,
        "symbols": [str(s).upper() for s in (dataset.symbols or [])],
    }


def _tuning_describe_dataset(dataset) -> str:
    """One dataset for a picker: name, size, span and tape."""
    return (
        f"{dataset.name} · {len(dataset.days)} sessions · {dataset.start} → {dataset.end} "
        f"· {dataset.feed}"
    )


def _render_tuning_overlaps(datasets: "list[dict]") -> None:
    """Warn about datasets that would be summed over the same sessions or tapes.

    A session two datasets share is replayed on both and so counted twice in
    the sum -- it weighs double in the pick. Mixed tapes are allowed but said
    out loud: fills differ between them, so part of any difference between
    two heatmaps is the tape.
    """
    for first, second in combinations(datasets, 2):
        shared = sim_tuning.overlapping_days(first["days"], second["days"])
        if shared:
            st.warning(
                f"`{first['name']}` and `{second['name']}` share {len(shared)} session"
                f"{'s' if len(shared) != 1 else ''} ({', '.join(shared[:5])}"
                f"{'…' if len(shared) > 5 else ''}). The sum counts those twice, so they "
                "weigh double in the pick."
            )
    feeds = sorted({d["feed"] for d in datasets})
    if len(feeds) > 1:
        st.info(
            f"These datasets read different tapes ({', '.join(feeds)}). Fills differ "
            "between tapes, so part of any difference between their heatmaps is the tape."
        )


def _tuning_range(name: str) -> "tuple[list, str]":
    """One numeric axis from a from/to/step row: (values, "") or ([], why not).

    The Simulate tab's sweep asks the same three boxes (`_sweep_range`) and
    both hand the answer to `tuning.axis_values`, so a range means the same
    thing in both places. The problem comes back rather than being shown here
    because the Tuning tab reports one problem at a time, above the button it
    disables -- a red box under each bad row and a live button is the arrangement
    that gets a broken grid queued.
    """
    tunable = sim_tuning.TUNABLES[name]
    cast = int if tunable.integer else float
    start0, stop0, step0 = tunable.default_range
    bounds = dict(
        min_value=cast(tunable.minimum), max_value=cast(tunable.maximum),
        step=cast(tunable.step), format=tunable.fmt,
    )
    col_from, col_to, col_step = st.columns(3)
    start = col_from.number_input(
        f"{tunable.label}: from", value=cast(start0), key=f"tune_{name}_from", **bounds
    )
    stop = col_to.number_input("to", value=cast(stop0), key=f"tune_{name}_to", **bounds)
    step = col_step.number_input(
        "step", min_value=cast(tunable.step), max_value=cast(tunable.maximum),
        value=cast(step0), step=cast(tunable.step), format=tunable.fmt,
        key=f"tune_{name}_step",
    )
    try:
        return sim_tuning.axis_values(name, start, stop, step), ""
    except ValueError as exc:
        return [], str(exc)


def _render_tuning_form() -> None:
    datasets = sim_data.list_datasets()
    if not datasets:
        st.info("Download a dataset first (Datasets tab).")
        return
    symbols = sorted({str(s).upper() for d in datasets for s in (d.symbols or [])})

    st.markdown("**Base configuration**")
    st.caption(
        "Every parameter the grid does not sweep comes from here, and the grid is "
        "compared against this configuration exactly as it stands."
    )
    base = _render_apple_params(symbols, _TUNE_PREFIX)

    st.markdown("**Parameters to tune**")
    names = st.multiselect(
        "Tune", [a for a in sim_tuning.AXES if a not in sim_tuning.LEGACY_AXES],
        default=["buy_k", "sell_k"],
        max_selections=sim_tuning.MAX_AXES,
        format_func=sim_tuning.sweep_label, key="tune_axes",
        help="One parameter draws a bar chart, two a heatmap — the first one is its rows. "
             "A third is shown over that heatmap one value at a time, or collapsed: each "
             "square at its best value of the third, or averaged over all of them. "
             "A number is swept over a range; a rule is swept over the options you tick, "
             "so a grid can put the levels under each forecast policy side by side.",
    )
    axes: list[dict] = []
    problems: list[str] = []
    for name in names:
        choice = sim_tuning.CHOICES.get(name)
        if choice is not None:
            values = st.multiselect(
                choice.label, list(choice.pickable), default=list(choice.pickable),
                format_func=lambda key, c=choice: c.labels.get(key, key),
                key=f"tune_{name}_options",
                help="Each ticked option is a row (or a column) of the grid: the same "
                     "sweep replayed under that rule.",
            )
            if not values:
                problems.append(f"{choice.label}: no values to try.")
                continue
        else:
            values, problem = _tuning_range(name)
            if problem:
                problems.append(problem)
                continue
        axes.append({"name": name, "values": values})
        st.caption(
            f"{len(values)} values: {', '.join(_tuning_value(name, v) for v in values)} — "
            f"the base configuration has {_tuning_value(name, getattr(base, name))}."
        )

    st.markdown("**Datasets**")
    carrying = [
        d for d in datasets
        if base.ticker in {str(s).upper() for s in (d.symbols or [])}
    ]
    if not carrying:
        st.warning(f"No stored dataset carries {base.ticker}. Download one in the Datasets tab.")
        return
    # Oldest first, the order the results show them in.
    stored = {d.name: d for d in carrying}
    by_name = {
        d["name"]: stored[d["name"]]
        for d in sim_tuning.chronological([_tuning_dataset_spec(d) for d in carrying])
    }
    newest = max(carrying, key=lambda d: str(d.end)).name
    chosen = st.multiselect(
        "Sweep on", list(by_name), default=[newest], format_func=lambda n: _tuning_describe_dataset(by_name[n]),
        key=f"tune_datasets_{base.ticker}",
        help="The grid is swept on every dataset picked here, each drawn as its own "
        "heatmap, and the pick is the best total profit over all of them. A finished job "
        "takes more datasets later with the same settings.",
    )
    picked = [by_name[name] for name in chosen]
    _render_tuning_overlaps([_tuning_dataset_spec(d) for d in picked])

    st.markdown("**How the pick is chosen**")
    st.caption("On the sum of every dataset's heatmap: the best total profit.")
    col_rule, col_share, col_cash = st.columns(3)
    rule = col_rule.selectbox(
        "Pick", list(_PICK_LABELS), format_func=_PICK_LABELS.get, key="tune_rule",
        help="The highest cell of a small grid is usually the luckiest. The plateau pick "
        "takes the cell whose neighbourhood — itself and every adjacent cell — scores "
        "best on average, which is how the shipped per-ticker levels were chosen.",
    )
    min_share = col_share.slider(
        "Must trade on at least", 0, 100, 50, step=5, format="%d%% of days",
        key="tune_min_traded",
        help="Counted over every session of every dataset. A deep entry that filled on "
        "one lucky day cannot be the pick.",
    ) / 100.0
    starting_cash = col_cash.number_input(
        "Starting cash", value=100_000.0, step=10_000.0, key="tune_cash",
        help="Each dataset is replayed from this cash, so a cell's summed return is its "
        "total profit as a share of it.",
    )
    col_reuse, col_workers = st.columns([3, 1], vertical_alignment="bottom")
    reuse_runs = col_reuse.checkbox(
        "Reuse stored runs", value=True, key="tune_reuse_runs",
        help="A cell is the same replay a Simulate run of that configuration is, so a "
        "matching run already in the store answers it — same rules, same sessions, same "
        "tape, same starting cash. Those cells fill in below before anything is queued "
        "and the job only replays the rest.",
    )
    cpus = os.cpu_count() or 2
    workers = col_workers.number_input(
        "Parallel workers", min_value=1, max_value=max(1, cpus),
        value=min(4, max(1, cpus - 1)), key="tune_workers",
        help="Replays run in separate processes, each loading the day-range model once.",
    )

    spec = {
        "base": rule_agent(APPLE_TRADER_KEY).to_record(base),
        "axes": axes,
        "datasets": [_tuning_dataset_spec(d) for d in picked],
        "starting_cash": float(starting_cash),
        "rule": rule,
        "min_traded_share": float(min_share),
        "reuse_runs": bool(reuse_runs),
        "workers": int(workers),
    }
    problem = problems[0] if problems else sim_tuning.validate(spec)
    if problem:
        st.error(problem)
    else:
        cells = sim_tuning.grid(axes)
        refused = 0
        for overrides in cells:
            try:
                sim_tuning.make_config(spec["base"], overrides)
            except (TypeError, ValueError):
                refused += 1
        runs = _runs()
        prior = sim_tuning.prior_cells(spec, runs)
        reused = sum(len(found) for found in prior.values())
        minutes = sim_tuning.estimated_seconds(spec, prior) / 60.0
        duration = "under a minute" if minutes < 1 else f"roughly {minutes:.0f} min"
        st.caption(
            f"{len(cells)} combinations"
            + (f" ({refused} refused by the configuration, left blank)" if refused else "")
            + f" × {len(picked)} dataset{'s' if len(picked) != 1 else ''}"
            + f" · {sim_tuning.total_replays(spec)} replays"
            + (f", {reused} of them already in the run store" if reused else "")
            + f" · {duration} with {workers} worker{'s' if workers != 1 else ''}."
        )
        _render_tuning_prior(spec, axes, prior, runs)
    if st.button(
        "Run tuning", type="primary", icon=":material/play_arrow:",
        disabled=bool(problem), key="tune_run",
    ):
        record = sim_tuning.submit(spec, runs=_runs())
        st.session_state["tune_selected_job"] = record["job_id"]
        st.toast("Tuning job started", icon=":material/tune:")
        st.rerun()


def _tuning_base_marks(spec: dict, axes: list[dict]) -> dict:
    """Where the base configuration sits on the grid, for a heatmap's outline."""
    return {"Base configuration": {a["name"]: spec["base"][a["name"]] for a in axes}}


def _render_tuning_prior(
    spec: dict, axes: list[dict], prior: dict, runs: list[dict]
) -> None:
    """What the run store already says about this grid, before anything is queued.

    A tuning cell and a Simulate run of the same configuration are the same
    replay, so a grid is rarely a blank sheet: anything swept from the Simulate
    tab, or left behind by an earlier tuning job, already answers part of it.
    Drawing that first does two things — it says what the job is actually going
    to cost, and it often answers the question before the job is queued at all.

    Only the cells with a stored run are filled; the rest of the surface is left
    transparent, which is the honest picture — an empty square here means "not
    tried", never "tried and flat". One heatmap per dataset that has any.
    """
    base = spec["base"]
    ticker, model = base["ticker"], base.get("model_key", "")
    pair = f"{ticker} on {_apple_model_label(model)}"
    stored = sim_tuning.runs_for_pair(runs, ticker, model)
    stale = sum(1 for record in stored if sim_tuning.run_is_stale(record))
    aged = (
        f" {stale} stored {pair} run{'s were' if stale != 1 else ' was'} saved before the "
        "data it read was last refreshed — a different daily history means a different "
        "forecast, so those have to be replayed rather than reused."
        if stale else ""
    )
    base_key = sim_tuning.overrides_key({})
    swept = {
        name: [cell for key, cell in found.items() if key != base_key]
        for name, found in prior.items()
    }
    swept = {name: cells for name, cells in swept.items() if cells}
    if not swept:
        if not stored:
            why = f"nothing has traded {pair} yet"
        else:
            why = (
                f"none of the {len(stored)} stored {pair} run"
                f"{'s' if len(stored) != 1 else ''} answers a cell of it — a run has to "
                "share every setting the grid does not sweep, plus the sessions, the "
                "tape and the starting cash."
                + aged
            )
        st.caption(f":material/history: No stored run covers this grid ({why})")
        return
    total = len(sim_tuning.grid(axes)) * len(prior)
    st.markdown(
        f"**Already run** — {sum(len(c) for c in swept.values())} of {total} "
        "combinations over the datasets picked"
    )
    marks = _tuning_base_marks(spec, axes)
    view = _tuning_view_picker(axes, marks, key="tune_prior_view")
    _render_tuning_charts([
        (
            f"tune_prior_heat_{index}",
            _tuning_view_heatmap(cells, axes, marks, f"From stored runs · {name}", view),
        )
        for index, (name, cells) in enumerate(swept.items())
    ])
    st.caption(
        f"Stored {pair} runs matching this base configuration in everything the grid "
        "does not sweep, and still current — same sessions, same tape, same starting "
        "cash, and nothing they read has changed since. Blank squares are combinations "
        "nobody has replayed, and they are what the job would run." + aged
    )


def _render_tuning_charts(
    charts: "list[tuple[str, go.Figure]]", job_id: "str | None" = None
) -> None:
    """Heatmaps two to a row; a lone one gets the whole width.

    With a `job_id`, a click on a square shows that configuration in the job's
    results (`_tuning_on_click`).
    """
    def draw(target, key: str, fig: go.Figure) -> None:
        if job_id is None:
            target.plotly_chart(fig, key=key)
            return
        target.plotly_chart(
            fig, key=key, selection_mode="points",
            on_select=lambda: _tuning_on_click(job_id, key),
        )

    if len(charts) == 1:
        draw(st, *charts[0])
        return
    for start in range(0, len(charts), 2):
        for column, (key, fig) in zip(st.columns(2), charts[start:start + 2]):
            draw(column, key, fig)


def _tuning_chosen_key(job_id: str) -> str:
    return f"tune_chosen_{job_id}"


def _tuning_on_click(job_id: str, chart_key: str) -> None:
    """Remember the configuration of the square just clicked, for the results.

    The configuration rides in the clicked point's customdata
    (`_tuning_heatmap`'s `config_of`). It is kept under its own key rather than
    read off the chart: the chart's selection is lost whenever its figure
    changes, and the figure changes on the very next run -- it outlines the
    square that was clicked. A click on a square with nothing to show (blank,
    refused) or a double-click, which clears the chart's selection, keeps
    whatever was shown before.
    """
    state = st.session_state.get(chart_key) or {}
    for point in (state.get("selection") or {}).get("points") or ():
        custom = point.get("customdata")
        if isinstance(custom, (list, tuple)) and len(custom) > 3 and custom[3]:
            st.session_state[_tuning_chosen_key(job_id)] = json.loads(custom[3])
            return


def _tuning_chosen(job: dict) -> "dict | None":
    """The configuration clicked on one of the job's heatmaps; None shows the pick.

    A click on the pick's own square is the pick, so it reads as one; a
    configuration no longer on the job's grid is forgotten.
    """
    chosen = st.session_state.get(_tuning_chosen_key(job["job_id"]))
    if not chosen:
        return None
    key = sim_tuning.overrides_key(chosen)
    best = job.get("best")
    if best and key == sim_tuning.overrides_key(best["overrides"]):
        return None
    if key not in {sim_tuning.overrides_key(o) for o in sim_tuning.grid(job["spec"]["axes"])}:
        return None
    return chosen


def _tuning_shown_cell(job: dict, name: str, chosen: "dict | None") -> "dict | None":
    """The shown configuration's cell on one dataset: the clicked one, else the pick's."""
    if chosen is None:
        return sim_tuning.pick_cell(job, name)
    return sim_tuning.cell_at(job, name, chosen)


# The two ways a three-axis grid's heatmap collapses its third parameter, as
# the switch beside its values names them.
_SLICE_LABELS = {
    sim_tuning.SLICE_BEST: "Best of all",
    sim_tuning.SLICE_MEAN: "Average",
}

# The clicked configuration's mark: a hue the heatmaps' red-yellow-green scale
# never uses, so it shows on any square.
_TUNING_SELECTED_COLOR = "#e040fb"


def _tuning_view_picker(axes: list[dict], marks: dict, key: str) -> "int | str | None":
    """Which view of the third parameter the heatmaps show; None with fewer axes.

    An index into the third axis's values draws the grid at that value;
    `SLICE_BEST` and `SLICE_MEAN` collapse the axis (`tuning.slice_cells`). It
    starts on the slice the first of `marks` sits on -- the pick, on a job's
    results -- so the heatmap opens on the outline it is about.

    The key carries the axis, so a form whose third parameter changes does not
    hand a new axis an index into the old one's values.
    """
    if len(axes) < 3:
        return None
    name, values = axes[2]["name"], axes[2]["values"]
    param = _tuning_param_label(name)
    default = next(
        (values.index(o[name]) for o in marks.values() if o and o.get(name) in values), 0
    )
    view = st.segmented_control(
        param, [*range(len(values)), *_SLICE_LABELS], default=default, required=True,
        format_func=lambda o: _SLICE_LABELS.get(o) or sim_tuning.axis_tick(name, values[o]),
        key=f"{key}_{name}_{len(values)}",
        help=f"The heatmaps show the first two parameters. A value of {param} draws the "
        f"grid at that value. **Best of all** draws each square at whichever {param} "
        "made the most there — the hover names it. **Average** draws each square's "
        f"profit averaged over every {param} the configuration accepts.",
    )
    return default if view is None else view


def _tuning_view_heatmap(
    cells: list[dict], axes: list[dict], marks: dict, title: str, view,
    clickable: bool = False,
) -> go.Figure:
    """`_tuning_heatmap` for a grid of any size, a third axis drawn as `view` says.

    On a slice, a mark is outlined only if it sits on that slice. Collapsed,
    every mark is outlined where it sits on the first two axes; its own value
    of the third is in the caption above the charts.

    `clickable` lets a click name the square's configuration: on a slice, the
    square at that value of the third; on Best of all, the value it won at.
    An Average square is several configurations, so it names none.
    """
    if len(axes) < 3:
        config_of = (lambda cell: cell["overrides"]) if clickable else None
        return _tuning_heatmap(cells, axes, marks, title, config_of=config_of)
    name, values = axes[2]["name"], axes[2]["values"]
    param = _tuning_param_label(name)
    metric_label = None
    if view == sim_tuning.SLICE_BEST:
        title, shown = f"{title} · best {param}", marks
        config_of = lambda cell: {**cell["overrides"], name: cell["best_of"]["value"]}  # noqa: E731
    elif view == sim_tuning.SLICE_MEAN:
        title, shown = f"{title} · average over {param}", marks
        metric_label = f"Average {sim_tuning.METRICS[sim_tuning.PICK_METRIC].lower()}"
        config_of = None
    else:
        value = values[int(view)]
        title = f"{title} · {param} {sim_tuning.axis_tick(name, value)}"
        shown = {label: o for label, o in marks.items() if (o or {}).get(name) == value}
        config_of = lambda cell: {**cell["overrides"], name: value}  # noqa: E731
    return _tuning_heatmap(
        sim_tuning.slice_cells(cells, axes, view), axes[:2], shown, title,
        metric_label=metric_label, config_of=config_of if clickable else None,
    )


def _tuning_hover_data(cell: "dict | None") -> list:
    """[return %, a line on days and datasets, a line on the third parameter]
    for one cell's hover.

    The third line is empty except on a collapsed three-axis view
    (`tuning.slice_cells`): which value of the third parameter a square's best
    is at, or what its average is over.

    Formatted here rather than in the hovertemplate: plotly ignores a format
    spec on a heatmap's nested `customdata`.
    """
    if not sim_tuning.is_scored(cell):
        return ["", "", ""]
    best_of, mean_of = cell.get("best_of"), cell.get("mean_of")
    if mean_of:
        note = f"{cell['days_traded']:g}/{cell['days']} days traded on average"
    else:
        note = f"{cell['days_traded']}/{cell['days']} days traded"
    if "datasets" in cell:
        note += f" · in profit on {cell['datasets_up']}/{cell['datasets']} datasets"
    third = ""
    if best_of:
        third = (
            f"Best at {_tuning_param_label(best_of['name'])} "
            f"{sim_tuning.axis_tick(best_of['name'], best_of['value'])} "
            f"(of {best_of['count']} values)"
        )
    elif mean_of:
        metric = sim_tuning.PICK_METRIC
        third = (
            f"Average of {mean_of['count']} {_tuning_param_label(mean_of['name'])} values: "
            f"{_tuning_metric_text(metric, mean_of['low'])} to "
            f"{_tuning_metric_text(metric, mean_of['high'])}"
        )
        if mean_of["of"] > mean_of["count"]:
            third += f" ({mean_of['of'] - mean_of['count']} refused or errored, left out)"
    return [f"{cell['return_pct']:+.2f}%", note, third]


def _tuning_heatmap(
    cells: list[dict], axes: list[dict], marks: dict, title: str, metric: str = "profit",
    metric_label: "str | None" = None,
    config_of: "Callable[[dict], dict] | None" = None,
) -> go.Figure:
    """The grid coloured by `metric`; `marks` outlines named cells (the pick, the
    base, the square clicked).

    The hover carries the return beside the dollar profit: the profit is what
    the cells are ranked on, the return is what it means against the cash.
    A three-axis grid comes here already sliced to its first two axes
    (`_tuning_view_heatmap`), and the hover then says what was done with the
    third.

    With `config_of`, a square can be clicked: each scored one carries its
    configuration (`config_of(cell)`, as JSON) as a fourth customdata entry,
    which `_tuning_on_click` reads. Plotly selects points of scatter-like
    traces only, never a heatmap's squares, so a clickable heatmap gets an
    invisible marker at each square's centre, and the hover moves onto those
    markers. With no limit on the hover distance the nearest centre answers,
    and on a grid that is the square under the cursor.
    """
    by_key = {sim_tuning.overrides_key(c["overrides"]): c for c in cells}

    def cell_at(overrides: dict) -> "dict | None":
        return by_key.get(sim_tuning.overrides_key(overrides))

    def custom_of(cell: "dict | None") -> list:
        data = _tuning_hover_data(cell)
        if config_of is not None and sim_tuning.is_scored(cell):
            data.append(json.dumps(config_of(cell)))
        return data

    metric_label = metric_label or sim_tuning.METRICS[metric]
    collapsed = any(c.get("best_of") or c.get("mean_of") for c in cells)
    hover_tail = (
        f"{metric_label}: %{{text}}<br>Return: %{{customdata[0]}}<br>%{{customdata[1]}}"
        + ("<br>%{customdata[2]}" if collapsed else "") + "<extra></extra>"
    )
    # A bar chart's stars: (colour, size).
    star_styles = {
        "Pick": (PALETTE["text"], 16),
        "Base configuration": (PALETTE["muted"], 16),
        "Selected": (_TUNING_SELECTED_COLOR, 12),
    }
    # A heatmap's outlines: (colour, size). The pick's and the base's are the
    # colours they always showed, which Streamlit's colorway used to hand out
    # by trace order. The clicked square's sits inside the others, so it still
    # shows on the pick's or the base's square.
    outline_styles = {
        "Pick": ("#0068c9", 30),
        "Base configuration": ("#ffabab", 30),
        "Selected": (_TUNING_SELECTED_COLOR, 22),
    }
    # A click selects a point in plotly, which dims every other point of every
    # selectable trace until the next run redraws the chart; the outlines and
    # bars keep their look.
    undimmed = dict(marker=dict(opacity=1))
    fig = go.Figure()
    if len(axes) == 1:
        axis = axes[0]
        xs = [sim_tuning.axis_tick(axis["name"], v) for v in axis["values"]]
        found = [cell_at({axis["name"]: value}) for value in axis["values"]]
        ys = [cell[metric] if sim_tuning.is_scored(cell) else None for cell in found]
        fig.add_trace(go.Bar(
            x=xs, y=ys, name=metric_label,
            marker_color=[PALETTE["up"] if (y or 0) >= 0 else PALETTE["down"] for y in ys],
            text=[_tuning_metric_text(metric, y) for y in ys], textposition="outside",
            customdata=[custom_of(cell) for cell in found], unselected=undimmed,
            hovertemplate=f"{_tuning_param_label(axis['name'])} %{{x}}<br>" + hover_tail,
        ))
        for label, overrides in marks.items():
            value = (overrides or {}).get(axis["name"])
            if value in axis["values"]:
                y = ys[axis["values"].index(value)]
                color, size = star_styles.get(label, (PALETTE["text"], 16))
                fig.add_trace(go.Scatter(
                    x=[sim_tuning.axis_tick(axis["name"], value)], y=[y or 0], mode="markers",
                    name=label, hoverinfo="skip", unselected=undimmed,
                    marker=dict(symbol="star", size=size, color=color),
                ))
        fig.update_xaxes(title=_tuning_param_label(axis["name"]), type="category")
        fig.update_yaxes(title=metric_label)
    else:
        row_axis, col_axis = axes
        xs = [sim_tuning.axis_tick(col_axis["name"], v) for v in col_axis["values"]]
        ys = [sim_tuning.axis_tick(row_axis["name"], v) for v in row_axis["values"]]
        position = (
            f"{_tuning_param_label(row_axis['name'])} %{{y}}<br>"
            f"{_tuning_param_label(col_axis['name'])} %{{x}}<br>"
        )
        z, text, custom = [], [], []
        # The clickable overlay's markers, one per square, flat.
        point_x, point_y, point_text, point_custom, point_hover = [], [], [], [], []
        for row_value, y in zip(row_axis["values"], ys):
            z_row, text_row, custom_row = [], [], []
            for col_value, x in zip(col_axis["values"], xs):
                cell = cell_at({row_axis["name"]: row_value, col_axis["name"]: col_value})
                if sim_tuning.is_scored(cell):
                    z_row.append(cell[metric])
                    text_row.append(_tuning_metric_text(metric, cell[metric]))
                    hover = hover_tail
                else:
                    z_row.append(None)
                    text_row.append("×" if cell and "invalid" in cell else "")
                    hover = (
                        "Not replayed yet" if cell is None
                        else "Refused by the configuration" if "invalid" in cell
                        else "The replay ended with an error"
                    ) + "<extra></extra>"
                custom_row.append(custom_of(cell))
                point_x.append(x)
                point_y.append(y)
                point_text.append(text_row[-1])
                point_custom.append(custom_row[-1])
                point_hover.append(position + hover)
            z.append(z_row)
            text.append(text_row)
            custom.append(custom_row)
        centred = metric in ("profit", "return_pct", "worst_day")
        fig.add_trace(go.Heatmap(
            z=z, x=xs, y=ys, text=text, texttemplate="%{text}", colorscale="RdYlGn",
            zmid=0 if centred else None, hoverongaps=False, customdata=custom,
            colorbar=dict(title=metric_label),
            hovertemplate=position + hover_tail,
            hoverinfo="skip" if config_of is not None else None,
        ))
        if config_of is not None:
            fig.add_trace(go.Scatter(
                x=point_x, y=point_y, mode="markers", name="", showlegend=False,
                marker=dict(size=1, opacity=0), unselected=dict(marker=dict(opacity=0)),
                text=point_text, customdata=point_custom, hovertemplate=point_hover,
            ))
            fig.update_layout(hoverdistance=-1, dragmode="zoom")
        for label, overrides in marks.items():
            overrides = overrides or {}
            row_value, col_value = overrides.get(row_axis["name"]), overrides.get(col_axis["name"])
            if row_value in row_axis["values"] and col_value in col_axis["values"]:
                # An open symbol is stroked in the marker's own colour; left
                # unset, Streamlit's colorway gives it one by trace order.
                color, size = outline_styles.get(label, (PALETTE["text"], 30))
                fig.add_trace(go.Scatter(
                    x=[sim_tuning.axis_tick(col_axis["name"], col_value)],
                    y=[sim_tuning.axis_tick(row_axis["name"], row_value)],
                    mode="markers", name=label, hoverinfo="skip", unselected=undimmed,
                    marker=dict(symbol="square-open", size=size, color=color, line=dict(
                        width=3, color=color)),
                ))
        # The whole grid, in grid order, even while cells are still missing --
        # a category axis otherwise shows only the values something was drawn at.
        fig.update_xaxes(
            title=_tuning_param_label(col_axis["name"]), type="category",
            categoryorder="array", categoryarray=xs,
        )
        fig.update_yaxes(
            title=_tuning_param_label(row_axis["name"]), type="category",
            categoryorder="array", categoryarray=ys,
        )
    fig.update_layout(title=title, legend=dict(orientation="h", y=-0.25))
    return _chart_layout(fig, height=440)


def _tuning_summary_rows(job: dict, chosen: "dict | None" = None) -> list[dict]:
    """The base configuration and the shown one on each dataset, then summed.

    The shown configuration is the pick, or the square clicked on a heatmap
    (`chosen`); its columns are `shown_*` either way.
    """
    spec = job["spec"]
    datasets = sim_tuning.chronological(spec["datasets"])

    def row(label: str, sessions: int, base: "dict | None", shown: "dict | None") -> dict:
        out = {"dataset": label, "sessions": sessions}
        for prefix, cell in (("base", base), ("shown", shown)):
            scored = sim_tuning.is_scored(cell)
            out[f"{prefix}_profit"] = cell["profit"] if scored else None
            out[f"{prefix}_return"] = cell["return_pct"] if scored else None
        scored = sim_tuning.is_scored(shown)
        out["shown_days"] = (
            f"{shown['days_up']} up · {shown['days_traded']} traded / {shown['days']}"
            if scored else "—"
        )
        out["shown_win_rate"] = sim_tuning.win_rate(shown) if scored else None
        out["shown_round_trips"] = (
            f"{shown['wins']} of {shown['round_trips']}"
            if scored and shown.get("round_trips") is not None else "—"
        )
        out["shown_worst"] = shown["worst_day"] if scored else None
        return out

    rows = [
        row(d["name"], len(d["days"]), job["baseline"].get(d["name"]),
            _tuning_shown_cell(job, d["name"], chosen))
        for d in datasets
    ]
    if len(datasets) > 1:
        rows.append(row(
            f"Sum of {len(datasets)}", sum(len(d["days"]) for d in datasets),
            sim_tuning.summed_baseline(job),
            job.get("best") if chosen is None else sim_tuning.summed_cell(job, chosen),
        ))
    return rows


def _tuning_usd(value: float) -> str:
    """A signed dollar amount for markdown: `+$134`, `−$517`.

    The `$` is escaped because Streamlit's markdown reads a pair of bare dollar
    signs as a formula and typesets everything between them.
    """
    return f"{'+' if value >= 0 else '−'}\\${abs(value):,.0f}"


def _tuning_verdict(job: dict) -> "str | None":
    """One line: how the pick did in total, and against the base configuration.

    Every dataset is in sample -- the pick is read off their sum -- so the
    robustness read is how many of them it made money on, not one held-out week.
    """
    spec = job["spec"]
    names = sim_tuning.dataset_names(spec)
    best, base = job.get("best"), sim_tuning.summed_baseline(job)
    if not sim_tuning.is_scored(best):
        if job["status"] == sim_tuning.FINISHED:
            return (
                ":orange[**No pick.**] No combination traded on enough of the days to be "
                "eligible — lower the minimum share of days, or widen the grid."
            )
        return None
    where = f"summed over {len(names)} datasets" if len(names) > 1 else f"on `{names[0]}`"
    detail = (
        f"The pick made **{_tuning_usd(best['profit'])}** ({best['return_pct']:+.2f}%) {where}"
    )
    if len(names) > 1:
        detail += f", in profit on {best['datasets_up']} of them"
    if sim_tuning.is_scored(base):
        detail += (
            f"; the base configuration made {_tuning_usd(base['profit'])} "
            f"({base['return_pct']:+.2f}%)"
        )
    beat_base = not sim_tuning.is_scored(base) or best["profit"] >= base["profit"]
    if best["profit"] > 0 and beat_base:
        head = ":green[**Beats the base configuration.**]"
    elif best["profit"] > 0:
        head = ":orange[**Profitable, but no better than the base configuration.**]"
    else:
        head = ":red[**The pick lost money.**]"
    return f"{head} {detail}."


_TUNING_WIN_RATE_HELP = (
    "Share of round trips that made money, fees included. A round trip runs from the "
    "buy that opens a flat book to the sell that flattens it, however many rungs and "
    "partial takes it had. Blank for a cell replayed before round trips were counted."
)


def _tuning_cell_rows(job: dict) -> list[dict]:
    """Every summed combination, best total first, with each dataset's share."""
    names = sim_tuning.dataset_names(job["spec"], by_date=True)
    per_dataset = [
        {sim_tuning.overrides_key(c["overrides"]): c for c in job["cells"].get(name) or ()}
        for name in names
    ]
    rows = []
    for cell in sim_tuning.summed_cells(job):
        if not sim_tuning.is_scored(cell):
            continue
        key = sim_tuning.overrides_key(cell["overrides"])
        row = {
            _tuning_param_label(name): value for name, value in cell["overrides"].items()
        }
        row.update(
            profit=cell["profit"], return_pct=cell["return_pct"], trades=cell["trades"],
            win_rate=sim_tuning.win_rate(cell),
            days_up=cell["days_up"], days_traded=cell["days_traded"],
            worst_day=cell["worst_day"],
        )
        for index, found in enumerate(per_dataset):
            part = found.get(key)
            row[f"ds_{index}"] = part["profit"] if sim_tuning.is_scored(part) else None
        row["source"] = (
            f"{cell['reused']} of {cell['datasets']} from stored runs" if cell["reused"]
            else "swept"
        )
        rows.append(row)
    rows.sort(key=lambda r: r["profit"], reverse=True)
    return rows


def _tuning_session_bars(symbol: str, feed: str, days: list[str]) -> dict[str, dict]:
    """Each session's stored daily bar (o, h, l, c), keyed by ET date.

    Daily bars are stamped at midnight exchange time in UTC (04:00Z or 05:00Z),
    so the session a bar belongs to is its date in ET, not in UTC. A day with no
    stored daily bar is left out rather than drawn as an empty candle.
    """
    wanted = set(days)
    bars = {}
    for bar in sim_data.load_daily_bars(symbol, feed):
        day = pd.Timestamp(bar["t"]).tz_convert(MARKET_TZ).date().isoformat()
        if day in wanted:
            bars[day] = {k: bar[k] for k in ("o", "h", "l", "c")}
    return bars


_BIAS_MARK = {
    "bullish": ("▲", PALETTE["up"]),
    "neutral": ("→", PALETTE["muted"]),
    "bearish": ("▼", PALETTE["down"]),
}


def _tuning_color(index: int) -> str:
    """One dataset's colour, in the order the job shows them (oldest first)."""
    return _DRIFT_SERIES_COLORS[index % len(_DRIFT_SERIES_COLORS)]


def _tuning_shown_days(job: dict, chosen: "dict | None" = None) -> dict[str, list[str]]:
    """{feed: sessions the shown configuration was scored on} across every
    dataset of the job -- the pick's, or `chosen`'s, the square clicked."""
    by_feed: dict[str, list[str]] = {}
    for dataset in job["spec"]["datasets"]:
        cell = _tuning_shown_cell(job, dataset["name"], chosen)
        if sim_tuning.is_scored(cell):
            by_feed.setdefault(dataset["feed"], []).extend(cell["daily"])
    return by_feed


def _tuning_session_context(job: dict, chosen: "dict | None" = None) -> dict:
    """The VIX open and the pre-market bias of every session the shown
    configuration was scored on (the pick's, or `chosen`'s).

    Briefings are LLM calls, made once per session and cached on disk
    (`session_context`); a day that failed is remembered for this browser
    session so a rerun does not ask again, and "Retry" clears that.
    """
    symbol = job["spec"]["base"]["ticker"]
    by_feed = _tuning_shown_days(job, chosen)
    days = sorted({d for ds in by_feed.values() for d in ds})
    context = {"vix": sim_session.vix_opens(days), "biases": {}, "errors": {}, "provider": None}
    briefer = sim_session.briefing_provider()
    if briefer is None:
        return context
    provider, model, api_key = briefer
    context["provider"] = f"{provider} · {model}"
    failed = st.session_state.setdefault("tune_briefing_failed", {})
    wanted = {
        feed: [d for d in ds if (symbol, d, provider, model) not in failed]
        for feed, ds in by_feed.items()
    }
    uncached = sum(
        1 for ds in wanted.values() for d in ds
        if sim_session.cached_briefing(symbol, d, provider, model) is None
    )
    with st.spinner(
        f"Writing {uncached} pre-market briefing{'s' if uncached != 1 else ''} "
        f"({provider} · {model})…",
        show_time=True,
    ) if uncached else contextlib.nullcontext():
        biases, errors = sim_session.session_biases(
            symbol, wanted, provider, model, api_key,
            os.getenv("ALPACA_API_KEY", ""), os.getenv("ALPACA_SECRET", ""),
        )
    for day, error in errors.items():
        failed[(symbol, day, provider, model)] = error
    context["biases"] = biases
    context["errors"] = {
        day: error for (sym, day, prov, mod), error in failed.items()
        if sym == symbol and prov == provider and mod == model and day in days
    }
    return context


@st.cache_data(show_spinner="Forecasting each session's day range…")
def _tuning_forecast(symbol: str, model_key: str, feed: str, day: str, signature: tuple) -> dict:
    """One session's morning forecast of its high and low, as the replay made it.

    The inputs `_overlay_day` hands the overlays -- the stored tape's opening
    minutes, completed daily bars strictly before the day and the stored
    opening print -- so the range is the one every cell's levels hung off at
    09:35, whatever the configuration did with it afterwards. `signature`
    (`drift.signature`) is the half of the key that moves when the store or
    the model does; a day with no forecast is cached with its reason, as
    `_overlay_day` caches its notes.
    """
    session = date.fromisoformat(day)
    market = SimMarket([symbol], [session], feed)
    t = market.session_open(session) + timedelta(minutes=1)
    result = model_overlays.session_forecast(
        model_key, symbol, market.series[symbol].minute_bars,
        daily_bars=market.completed_daily_bars(symbol, t),
        session_date=session,
        open_price=market.session_open_price(symbol, t),
    )
    forecast = result["forecast"]
    return {
        "high": float(forecast["pred_high"]) if forecast else None,
        "low": float(forecast["pred_low"]) if forecast else None,
        "made_at": f"{pd.Timestamp(result['made_at']):%H:%M}" if forecast else None,
        "problem": result["problem"],
        "overlay": result["overlay"],
    }


def _tuning_forecasts(job: dict, chosen: "dict | None" = None) -> dict:
    """{"days": {day: forecast}, "overlay", "notes"} for every session the
    shown configuration was scored on (the pick's, or `chosen`'s), from the
    model that configuration trades on."""
    spec = job["spec"]
    symbol = spec["base"]["ticker"]
    shown = chosen if chosen is not None else (job.get("best") or {}).get("overrides") or {}
    model_key = model_overlays.trader_model(sim_tuning.make_config(spec["base"], shown), symbol)
    out = {"days": {}, "overlay": None, "notes": []}
    missing: dict[str, str] = {}
    for feed, days in _tuning_shown_days(job, chosen).items():
        signature = sim_drift.signature(model_key, symbol, feed)
        for day in days:
            found = _tuning_forecast(symbol, model_key, feed, day, signature)
            out["overlay"] = found["overlay"]
            if found["high"] is None:
                missing[day] = found["problem"]
            else:
                out["days"][day] = found
    if missing:
        first = min(missing)
        out["notes"].append(
            f"No predicted range for {len(missing)} session(s) — {first}: {missing[first]}"
        )
    return out


def _tuning_day_label(day: str, context: dict) -> str:
    lines = [day]
    record = context["biases"].get(day)
    if record:
        icon, color = _BIAS_MARK.get(record["bias"], ("?", PALETTE["muted"]))
        lines.append(f"<span style='color:{color}'>{icon} {record['bias']}</span>")
    else:
        lines.append(f"<span style='color:{PALETTE['muted']}'>no bias</span>")
    vix = context["vix"].get(day)
    lines.append(f"VIX {vix:.1f}" if vix is not None else "VIX –")
    return "<br>".join(lines)


def _tuning_day_hover(day: str, context: dict) -> str:
    record = context["biases"].get(day)
    vix = context["vix"].get(day)
    bias = f"{record['bias']} ({record['confidence']} confidence)" if record else "no briefing"
    return f"pre-market: {bias}<br>VIX open: " + (f"{vix:.2f}" if vix is not None else "–")


def _tuning_forecast_hover(forecast: dict, bar: "dict | None") -> str:
    lines = [
        f"predicted {forecast['low']:.2f} – {forecast['high']:.2f} "
        f"(forecast at {forecast['made_at']})"
    ]
    if bar is not None:
        lines.append(f"actual {bar['l']:.2f} – {bar['h']:.2f}")
        lines.append(
            f"actual − predicted: high {bar['h'] - forecast['high']:+.2f}, "
            f"low {bar['l'] - forecast['low']:+.2f}"
        )
    return "<br>".join(lines)


def _tuning_daily_chart(
    job: dict,
    context: "dict | None" = None,
    chosen: "dict | None" = None,
    forecasts: "dict | None" = None,
) -> go.Figure:
    """The pick's profit per session over the daily candles -- or, with
    `chosen`, the profit of the square clicked on a heatmap. `forecasts`
    (`_tuning_forecasts`) boxes each session's predicted high and low behind
    its candle."""
    spec = job["spec"]
    symbol = spec["base"]["ticker"]
    context = context or {"vix": {}, "biases": {}}
    who = "Pick" if chosen is None else "Selected"
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        row_heights=[0.6, 0.4],
    )
    bars: dict[str, dict] = {}
    for index, dataset in enumerate(sim_tuning.chronological(spec["datasets"])):
        cell = _tuning_shown_cell(job, dataset["name"], chosen)
        if not sim_tuning.is_scored(cell):
            continue
        days = list(cell["daily"])
        fig.add_trace(go.Bar(
            x=days, y=[cell["daily"][d] for d in days],
            name=f"{who} on {dataset['name']}", marker_color=_tuning_color(index),
            hovertext=[_tuning_day_hover(d, context) for d in days],
        ), row=1, col=1)
        bars.update(_tuning_session_bars(symbol, dataset["feed"], days))
    predicted = (forecasts or {}).get("days") or {}
    if predicted:
        # A box from the predicted low to the predicted high, behind the candle:
        # a wick through its lid or floor is the session trading outside the
        # forecast. Added before the candles so they draw over it.
        overlay = forecasts["overlay"]
        color = MODEL_OVERLAY_COLORS[overlay]
        days = sorted(predicted)
        fig.add_trace(go.Bar(
            x=days,
            base=[predicted[d]["low"] for d in days],
            y=[predicted[d]["high"] - predicted[d]["low"] for d in days],
            width=0.7,
            name=model_overlays.label(overlay),
            marker=dict(color=_rgba(color, 0.18), line=dict(color=color, width=1.5)),
            customdata=[_tuning_forecast_hover(predicted[d], bars.get(d)) for d in days],
            hovertemplate="%{customdata}<extra></extra>",
        ), row=2, col=1)
    if bars:
        days = sorted(bars)
        fig.add_trace(go.Candlestick(
            x=days,
            open=[bars[d]["o"] for d in days],
            high=[bars[d]["h"] for d in days],
            low=[bars[d]["l"] for d in days],
            close=[bars[d]["c"] for d in days],
            name=f"{symbol} daily",
            increasing_line_color=PALETTE["up"],
            decreasing_line_color=PALETTE["down"],
        ), row=2, col=1)
    fig.update_xaxes(
        type="category", categoryorder="category ascending", gridcolor=PALETTE["grid"],
        rangeslider_visible=False,
    )
    # Each session's label carries what the morning looked like: the pre-market
    # briefing's call and the VIX's opening print. Set on the bottom axis only,
    # the one a shared x shows.
    shown_days = _tuning_shown_days(job, chosen).values()
    all_days = sorted({d for ds in shown_days for d in ds} | set(bars))
    fig.update_xaxes(
        tickvals=all_days, ticktext=[_tuning_day_label(d, context) for d in all_days],
        row=2, col=1,
    )
    fig.update_yaxes(title="Profit ($)", row=1, col=1)
    fig.update_yaxes(title=f"{symbol} ($)", row=2, col=1)
    fig.update_yaxes(gridcolor=PALETTE["grid"])
    title = (
        "The pick, session by session" if chosen is None
        else f"Selected — {_tuning_values_text(chosen)}, session by session"
    )
    fig.update_layout(title=title, barmode="group")
    _chart_layout(fig, height=520)
    fig.update_layout(margin=dict(b=80))
    return fig


def _render_tuning_session_context(
    job: dict, context: dict, chosen: "dict | None" = None
) -> None:
    """Under the chart: where the labels come from, and each morning's briefing,
    beside the shown configuration's profit that day (the pick's, or `chosen`'s)."""
    if context["provider"] is None:
        st.caption(
            ":material/info: No LLM key in the environment (GEMINI_API_KEY, OPENAI_API_KEY "
            "or ANTHROPIC_API_KEY), so the sessions carry no pre-market bias."
        )
    if context["errors"]:
        first_day = min(context["errors"])
        col_msg, col_retry = st.columns([5, 1], vertical_alignment="center")
        col_msg.warning(
            f"No pre-market briefing for {len(context['errors'])} session(s) — "
            f"{first_day}: {context['errors'][first_day]}"
        )
        if col_retry.button("Retry", key=f"tune_briefing_retry_{job['job_id']}"):
            st.session_state["tune_briefing_failed"] = {}
            st.rerun()
    if not context["biases"]:
        return
    profit = {}
    for name in sim_tuning.dataset_names(job["spec"]):
        cell = _tuning_shown_cell(job, name, chosen)
        if sim_tuning.is_scored(cell):
            profit.update(cell["daily"])
    rows = [
        {
            "day": day,
            "bias": record["bias"],
            "confidence": record["confidence"],
            "vix_open": context["vix"].get(day),
            "profit": profit.get(day),
            "summary": record["summary"],
        }
        for day, record in sorted(context["biases"].items())
    ]
    with st.expander(
        f"Pre-market briefings ({context['provider']})", icon=":material/wb_twilight:"
    ):
        st.caption(
            "Written as of 09:25 ET each session from what was known then — the daily "
            "closes and SPY/VIX/VIX3M closes before that day, the week's news up to "
            "09:25, and the earnings schedule; no fundamentals, analyst targets or "
            "alternative data, which have no point-in-time history. One LLM call per "
            "session, cached: the first answer is the one kept. VIX open is ^VIX's "
            "opening print that day."
        )
        st.dataframe(pd.DataFrame(rows), hide_index=True, column_config={
            "day": "Session",
            "bias": "Bias",
            "confidence": "Confidence",
            "vix_open": st.column_config.NumberColumn("VIX open", format="%.2f"),
            "profit": st.column_config.NumberColumn(
                f"{'Pick' if chosen is None else 'Selected'} profit ($)", format="%+.2f"
            ),
            "summary": st.column_config.TextColumn("Summary", width="large"),
        })


def _render_tuning_notes(job: dict) -> None:
    for dataset in sim_tuning.chronological(job["spec"]["datasets"]):
        name = dataset["name"]
        cells = [*(job["cells"].get(name) or ()), job["baseline"].get(name)]
        errors = [c for c in cells if c and c.get("error")]
        if errors:
            st.warning(
                f"{len(errors)} replay(s) on `{name}` ended with an error — "
                f"the first: {errors[0]['error']}"
            )
        base = job["baseline"].get(name)
        missing = (base or {}).get("no_forecast_days") or []
        if missing:
            st.warning(
                f"On `{name}` the day-range model could not forecast "
                f"{len(missing)} of {base['days']} sessions ({', '.join(missing)}), so no "
                "configuration traded them. The usual cause is too little daily history "
                "before the dataset's first day."
            )
        sat_out = (base or {}).get("sat_out_days") or []
        if sat_out:
            st.caption(
                f":material/event_busy: On `{name}` the base configuration sat out "
                f"{len(sat_out)} of {base['days']} sessions ({', '.join(sat_out)}) under its "
                "days-off rule — the day after earnings, a CPI or jobs release, or a shock."
            )
    reused = 0 if sim_tuning.is_derived(job) else sim_tuning.reused_count(job)
    if reused:
        st.caption(
            f":material/history: {reused} of this job's {job['progress']['total']} replays "
            "came out of the run store — a stored simulation of the same configuration "
            "over the same sessions, tape and starting cash — rather than being swept "
            "again. Which cells is in the table below."
        )
    refused = sum(1 for c in sim_tuning.summed_cells(job) if "invalid" in c)
    if refused:
        st.caption(
            f":material/block: {refused} combination{'s' if refused != 1 else ''} the "
            "configuration refuses — a sell distance at or above the buy distance, say — "
            "are marked × and were not replayed."
        )


def _render_derived_header(job: dict) -> None:
    """What a derived grid is, and the offer to finish it.

    Nobody submitted this one: it is the stored runs that happen to agree on
    everything except its axes, read as the grid they describe. So it comes
    with the caveats a submitted job does not have — one dataset, no base
    configuration anyone nominated, no pick rule — and the one action worth
    offering is to turn it into a real job, which replays the holes and reuses
    every cell already on screen. Once it is a job, more datasets can be added.
    """
    spec = job["spec"]
    axes = spec["axes"]
    [name] = sim_tuning.dataset_names(spec)
    cells = sim_tuning.grid(axes)
    filled = len(job["cells"].get(name) or ())
    missing = len(cells) - filled
    st.info(
        f"**Assembled from {filled} stored run{'s' if filled != 1 else ''}**, not swept as a "
        "job. They agree on every setting except "
        + " and ".join(f"**{_tuning_param_label(a['name'])}**" for a in axes)
        + f", so they read as a grid of {' × '.join(str(len(a['values'])) for a in axes)} "
        f"over `{name}` — "
        + (
            f"{missing} of its {len(cells)} combinations nobody has run, left blank below."
            if missing else "and every combination has been run."
        ),
        icon=":material/auto_awesome_mosaic:",
    )
    st.caption(
        "Incomplete by construction: there is only the one dataset, the base configuration "
        "marked on the grid is simply the most recent of these runs, and the pick is the "
        "highest cell with no minimum share of days traded — nobody chose any of that. "
        "Read it as what has been tried, not as a tuned answer."
    )
    if not missing:
        return
    workers = min(4, max(1, (os.cpu_count() or 2) - 1))
    spec = {**spec, "workers": workers}
    problem = sim_tuning.validate(spec)
    if problem:
        st.caption(f":material/block: This grid cannot be submitted as a job: {problem}")
        return
    # What the job would reuse is this grid's own cells -- it was assembled by
    # the same match `prior_cells` makes -- so the estimate reads them off the
    # job rather than scoring the whole store again.
    filled_cells = {
        name: {
            sim_tuning.overrides_key(c["overrides"]): c
            for c in [*job["cells"][name], job["baseline"][name]] if c
        },
    }
    minutes = sim_tuning.estimated_seconds(spec, filled_cells) / 60.0
    duration = "under a minute" if minutes < 1 else f"roughly {minutes:.0f} min"
    with st.container(horizontal=True, vertical_alignment="center"):
        if st.button(
            f"Sweep the {missing} missing combination{'s' if missing != 1 else ''}",
            icon=":material/play_arrow:", type="primary", key=f"tune_fill_{job['job_id']}",
            help="Queues a real tuning job over this grid. The cells above are reused as "
            "they are, so only the blanks are replayed.",
        ):
            record = sim_tuning.submit(spec, runs=_runs())
            st.session_state["tune_selected_job"] = record["job_id"]
            st.toast("Tuning job started", icon=":material/tune:")
            st.rerun()
        st.caption(
            f"{duration} with {workers} workers. Once it is a job, more datasets can be "
            "added to it; for a wider range, copy these settings into **New tuning job**."
        )


def _render_tuning_dataset_controls(job: dict) -> None:
    """Add a dataset to a finished job, finish a stopped one, or drop a dataset.

    The weekly chore this tab is for: a new week of tape arrives, it is swept
    with the job's own settings, and the sum and the pick are re-read with it.
    """
    spec = job["spec"]
    job_id = job["job_id"]
    ticker = spec["base"]["ticker"]
    names = sim_tuning.dataset_names(spec, by_date=True)
    addable = sorted(
        (
            d for d in sim_data.list_datasets()
            if ticker in {str(s).upper() for s in (d.symbols or [])} and d.name not in names
        ),
        key=lambda d: str(d.end), reverse=True,
    )
    missing = sim_tuning.missing_replays(job)
    col_add, col_actions = st.columns([3, 2], vertical_alignment="bottom")
    chosen = None
    if addable:
        by_name = {d.name: d for d in addable}
        chosen = by_name[col_add.selectbox(
            "Add a dataset", list(by_name),
            format_func=lambda n: _tuning_describe_dataset(by_name[n]),
            key=f"tune_add_{job_id}",
            help="Swept with this job's base configuration, grid, cash and pick rule, then "
            "summed with the others.",
        )]
    else:
        col_add.caption(f"Every stored dataset that carries {ticker} is already in this job.")
    with col_actions.container(horizontal=True, vertical_alignment="bottom"):
        if chosen is not None and st.button(
            "Add and sweep", icon=":material/add:", type="primary",
            key=f"tune_add_go_{job_id}",
        ):
            try:
                sim_tuning.add_dataset(job_id, _tuning_dataset_spec(chosen), runs=_runs())
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.toast(f"Sweeping {chosen.name}", icon=":material/tune:")
                st.rerun()
        if missing and st.button(
            f"Sweep {missing} missing", icon=":material/play_arrow:",
            key=f"tune_resume_{job_id}",
            help="Replays only what this job has no answer for yet — the rest of a stopped "
            "job, say. Stored runs are reused where they match.",
        ):
            sim_tuning.resume(job_id, runs=_runs())
            st.toast("Tuning job resumed", icon=":material/tune:")
            st.rerun()
        if len(names) > 1:
            with st.popover("Remove a dataset", icon=":material/remove:"):
                dropped = st.selectbox("Dataset", names, key=f"tune_remove_{job_id}")
                st.caption(
                    "Its cells are dropped and the sum and the pick are re-read from the "
                    "rest; nothing is replayed. Adding it back sweeps it again, except "
                    "where the run store answers a cell."
                )
                if st.button(
                    "Remove", icon=":material/delete:", key=f"tune_remove_go_{job_id}",
                ):
                    sim_tuning.remove_dataset(job_id, dropped)
                    st.toast(f"Removed {dropped}", icon=":material/remove:")
                    st.rerun()
    if chosen is not None:
        added = _tuning_dataset_spec(chosen)
        _render_tuning_overlaps(sim_tuning.chronological([*spec["datasets"], added]))
        prior = sim_tuning.prior_cells(spec, _runs(), [added])
        minutes = sim_tuning.estimated_seconds(spec, prior, [added]) / 60.0
        found = len(prior[added["name"]])
        st.caption(
            f"Adding `{chosen.name}` is {len(sim_tuning.grid(spec['axes'])) + 1} replays"
            + (f", {found} of them already in the run store" if found else "")
            + " · " + ("under a minute" if minutes < 1 else f"roughly {minutes:.0f} min")
            + f" with {spec.get('workers') or 1} worker"
            + ("s" if int(spec.get("workers") or 1) != 1 else "") + "."
        )


def _render_tuning_results(jobs: list[dict]) -> None:
    st.markdown("##### :material/grid_on: Tuning results")
    wanted = st.session_state.get("tune_selected_job")
    detail = _derived_discriminators(jobs)
    labels = {
        j["job_id"]: _tuning_job_label(j, markdown=False, detail=detail.get(j["job_id"], ""))
        for j in jobs
    }
    ids = sorted(labels, key=lambda i: labels[i].casefold())
    job_id = st.selectbox(
        "Tuning job", ids, index=ids.index(wanted) if wanted in ids else 0,
        format_func=labels.get,
    )
    job = next(j for j in jobs if j["job_id"] == job_id)
    spec = job["spec"]
    axes = spec["axes"]
    names = sim_tuning.dataset_names(spec, by_date=True)

    if job["status"] == sim_tuning.RUNNING:
        st.info("Still running — the heatmaps fill in as cells finish.",
                icon=":material/hourglass_top:")
    elif job["status"] == sim_tuning.FAILED:
        st.error(
            f"This job did not finish ({job.get('error')}). What was swept before it "
            "stopped is shown below; **Sweep missing** replays the rest."
        )
    elif sim_tuning.is_derived(job):
        _render_derived_header(job)

    signature = rule_agent(APPLE_TRADER_KEY).signature(sim_tuning.make_config(spec["base"], {}))
    st.caption(
        f"Base `{signature}` · pick: best total profit"
        + (f" over {len(names)} datasets" if len(names) > 1 else "")
        + f", {_PICK_LABELS.get(spec.get('rule'), spec.get('rule')).lower()}, trading on at "
        f"least {float(spec.get('min_traded_share') or 0):.0%} of days · starting cash "
        f"\\${float(spec['starting_cash']):,.0f} per dataset."
    )
    if job["status"] not in (sim_tuning.RUNNING, sim_tuning.DERIVED):
        _render_tuning_dataset_controls(job)

    verdict = _tuning_verdict(job)
    if verdict:
        st.markdown(verdict)

    best = job.get("best")
    # A square clicked on any of the heatmaps below replaces the pick in the
    # table, the session-by-session chart and the configuration at the bottom;
    # the verdict above stays about the pick.
    chosen = _tuning_chosen(job)
    who = "Pick" if chosen is None else "Selected"
    base_values = {a["name"]: spec["base"][a["name"]] for a in axes}
    st.caption(
        f"Base configuration: {_tuning_values_text(base_values)}"
        + (f" · Pick: {_tuning_values_text(best['overrides'])}" if best else "")
        + (f" · **Selected: {_tuning_values_text(chosen)}**" if chosen else "")
    )
    if chosen is not None:
        with st.container(horizontal=True, vertical_alignment="center"):
            st.caption(
                ":material/ads_click: Showing the square clicked on a heatmap rather than "
                "the pick, here and in the session-by-session chart below."
            )
            st.button(
                "Back to the pick", icon=":material/undo:", key=f"tune_unchoose_{job_id}",
                on_click=st.session_state.pop, args=(_tuning_chosen_key(job_id), None),
            )
    money = st.column_config.NumberColumn
    st.dataframe(
        pd.DataFrame(_tuning_summary_rows(job, chosen)), hide_index=True, column_config={
            "dataset": "Dataset",
            "sessions": "Sessions",
            "base_profit": money("Base profit ($)", format="%+.2f"),
            "base_return": money("Base return", format="%+.2f%%"),
            "shown_profit": money(f"{who} profit ($)", format="%+.2f"),
            "shown_return": money(f"{who} return", format="%+.2f%%"),
            "shown_days": f"{who} days",
            "shown_win_rate": money(
                f"{who} win rate", format="%.0f%%", help=_TUNING_WIN_RATE_HELP
            ),
            "shown_round_trips": st.column_config.TextColumn(
                f"{who} wins",
                help=f"Winning round trips of all the {who.lower()}'s round trips.",
            ),
            "shown_worst": money(f"{who} worst day ($)", format="%+.2f"),
        },
    )

    marks = {"Pick": (best or {}).get("overrides"), **_tuning_base_marks(spec, axes)}
    if chosen is not None:
        marks["Selected"] = chosen
    prefix = "From stored runs · " if sim_tuning.is_derived(job) else ""
    view = _tuning_view_picker(axes, marks, key=f"tune_view_{job_id}")
    # An Average square is several configurations at once, so it cannot be one.
    clickable = view != sim_tuning.SLICE_MEAN
    _render_tuning_charts([
        (
            f"tune_heat_{job_id}_{index}",
            _tuning_view_heatmap(
                job["cells"].get(name) or [], axes, marks, f"{prefix}{name}", view,
                clickable=clickable,
            ),
        )
        for index, name in enumerate(names)
    ], job_id=job_id if clickable else None)
    if len(names) > 1:
        _render_tuning_charts([(
            f"tune_heat_sum_{job_id}",
            _tuning_view_heatmap(
                sim_tuning.summed_cells(job), axes, marks,
                f"Sum of all {len(names)} datasets", view, clickable=clickable,
            ),
        )], job_id=job_id if clickable else None)
        st.caption(
            "A combination is summed once every dataset has replayed it. The pick is the "
            "best cell of this sum, so every dataset above is in sample: a pick worth "
            "keeping is in profit on most of them, not carried by one."
        )
    if len(axes) > 2:
        third = _tuning_param_label(axes[2]["name"])
        st.caption(
            f"The heatmaps are the first two parameters; the switch above picks what to "
            f"show of {third}. On one of its values the outlines mark the pick and the "
            "base configuration only if they sit on it. Best of all and Average outline "
            "them where they sit on the first two, whatever their own value of "
            f"{third}. A collapsed square fills in once every {third} has been replayed "
            "there, and refused combinations are left out of its average."
        )
    st.caption(
        f":material/ads_click: Click a {'bar' if len(axes) == 1 else 'square'} to show "
        "that configuration in the table above and the session-by-session chart below; "
        "it is marked in magenta"
        + (
            f" — on Best of all, at the {_tuning_param_label(axes[2]['name'])} it won at. "
            "An Average square is several configurations, so it cannot be clicked."
            if len(axes) > 2 else "."
        )
    )
    _render_tuning_notes(job)

    rows = _tuning_cell_rows(job)
    if rows:
        per_dataset = {
            f"ds_{index}": money(f"{name} ($)", format="%+.2f")
            for index, name in enumerate(names)
        }
        with st.expander(f"Every combination ({len(rows)})", icon=":material/table_rows:"):
            st.dataframe(pd.DataFrame(rows), hide_index=True, column_config={
                "profit": money("Total profit ($)", format="%+.2f"),
                "return_pct": money("Total return", format="%+.2f%%"),
                "trades": "Trades",
                "win_rate": money("Win rate", format="%.0f%%", help=_TUNING_WIN_RATE_HELP),
                "days_up": "Days up",
                "days_traded": "Days traded",
                "worst_day": money("Worst day ($)", format="%+.2f"),
                **per_dataset,
                "source": st.column_config.TextColumn(
                    "Where from",
                    help="`swept` — replayed by this job. Otherwise how many of the "
                    "datasets' replays were taken from a stored simulation run of the same "
                    "configuration over the same sessions.",
                ),
            })

    shown = chosen if chosen is not None else (best or {}).get("overrides")
    if _tuning_shown_days(job, chosen) and shown is not None:
        context = _tuning_session_context(job, chosen)
        forecasts = _tuning_forecasts(job, chosen)
        st.plotly_chart(
            _tuning_daily_chart(job, context, chosen, forecasts), key=f"tune_daily_{job_id}"
        )
        for note in forecasts["notes"]:
            st.caption(f":material/info: {note}")
        _render_tuning_session_context(job, context, chosen)
        picked = sim_tuning.make_config(spec["base"], shown)
        st.markdown(
            "**The pick as a configuration**" if chosen is None
            else "**The selected square as a configuration**"
        )
        st.code(rule_agent(APPLE_TRADER_KEY).signature(picked), language=None)
        st.caption(
            f"{_tuning_values_text(shown)}. Set these in an Apple Trader setup on the "
            f"Simulate tab to replay {'the pick' if chosen is None else 'it'} with its "
            "full ledger, charts and decisions."
        )

    # A derived grid has no record to delete -- it is the runs themselves, and
    # they are Results' to remove.
    if job["status"] not in (sim_tuning.RUNNING, sim_tuning.DERIVED) and st.button(
        "Delete this tuning job", icon=":material/delete:", key=f"tune_delete_{job_id}"
    ):
        sim_tuning.delete_job(job_id)
        st.session_state.pop("tune_selected_job", None)
        st.rerun()


# ---------------------------------------------------------------------------

# The tab bar, as labels rather than as a literal list at the call site: the
# active tab is session state now (`TAB_STATE_KEY`), and the value stored there
# is a label, so anything that switches tabs has to name the exact string.
#
# Left to right they follow the order the work happens in. ML Models sits beside
# Agents rather than near Results: it describes what an agent *is* before a run,
# not what one did afterwards. Tuning follows it because it answers the same
# question one step on -- which settings that agent should run -- and Drift
# follows both, being about the models again, as they have held up rather than
# as they were saved. Then the datasets, the runs, and what the runs came to:
# Results one at a time and Summary across all of them, which is the last thing
# there is to look at.
TAB_AGENTS = ":material/smart_toy: Agents"
TAB_MODELS = ":material/neurology: ML Models"
TAB_TUNING = ":material/tune: Tuning"
TAB_DRIFT = ":material/monitoring: Drift"
TAB_DATASETS = ":material/database: Datasets"
TAB_SIMULATE = ":material/play_circle: Simulate"
TAB_RESULTS = ":material/insights: Results"
TAB_SUMMARY = ":material/leaderboard: Summary"
SIMLAB_TABS = [
    TAB_AGENTS, TAB_MODELS, TAB_TUNING, TAB_DRIFT,
    TAB_DATASETS, TAB_SIMULATE, TAB_RESULTS, TAB_SUMMARY,
]
TAB_STATE_KEY = "simlab_tab"

# How often the app re-downloads stored files that were fetched before their
# day was over and can now be had whole (`sim_data.repair_incomplete`). Once per
# process at start, then at most this often -- a day that finishes while the
# app is open (today's, after 20:30 ET) is picked up on a later rerun.
_REPAIR_EVERY = timedelta(minutes=10)
_repair_lock = threading.Lock()
_last_repair: "dict[str, datetime]" = {}


def _repair_incomplete_store() -> None:
    """Re-download what the store holds only part of, before anything reads it.

    A day fetched mid-session (2026-09-09 stopped at 10:29) used to stay that
    way: a replay ran out of tape before the flatten. The scan is a stat per
    file; downloads happen only when something is both incomplete and now
    whole at the source. Never lets a failure reach the page.
    """
    now = datetime.now(timezone.utc)
    if now - _last_repair.get("at", datetime.min.replace(tzinfo=timezone.utc)) < _REPAIR_EVERY:
        return
    if not _repair_lock.acquire(blocking=False):
        return  # another session's rerun is already at it
    try:
        _last_repair["at"] = now
        pending = [f for f in sim_data.incomplete_files() if f.fixable]
        if pending:
            with st.status(
                f"Re-downloading {len(pending)} stored file(s) fetched before their day was over…",
                expanded=False,
            ) as status:
                repaired = sim_data.repair_incomplete(progress=st.write)
                status.update(
                    label=f"Re-downloaded {len(repaired)} of {len(pending)} incomplete stored file(s)",
                    state="complete" if len(repaired) == len(pending) else "error",
                )
        sim_data.reconcile_dataset_days()
    except Exception as exc:  # the store repair must never take the app down
        st.warning(f"Checking the stored data for incomplete days failed: {exc}")
    finally:
        _repair_lock.release()


def build_ui() -> None:
    st.set_page_config(page_title="AgentStonks SimLab", page_icon="🧪", layout="wide")
    st.title("SimLab — strategy testing")
    st.caption(
        "Replay the trading agents against stored historical sessions: same prompts, same "
        "tools, same execution path as live — hours of tape in minutes of simulation."
    )
    # Above the tabs, not inside Simulate, because the queue is the app's and
    # not that tab's: `on_change="rerun"` below makes the tabs lazy, and a
    # pipeline that only renders while Simulate is open is a scheduler that only
    # ticks while Simulate is open -- so a batch would stall the moment anyone
    # went to read a result. It is also the honest place for it: experiments run
    # whatever is on screen.
    _repair_incomplete_store()
    _render_pipeline()
    st.divider()
    (
        tab_agents, tab_models, tab_tuning, tab_drift, tab_datasets, tab_sim,
        tab_results, tab_summary,
    ) = st.tabs(
        SIMLAB_TABS, key=TAB_STATE_KEY,
        # Two things at once, and both are wanted. The active tab becomes
        # readable and *writable* through `st.session_state[TAB_STATE_KEY]`,
        # which is how a Summary card opens its run in Results
        # (`_open_run_in_results`) -- without it the write is accepted and the
        # frontend ignores it. And tab bodies stop running when they are not
        # open, so opening SimLab no longer scores every drift model and parses
        # every stored run before drawing the first tab.
        on_change="rerun",
    )
    with tab_agents:
        render_agents_tab()
    with tab_models:
        model_catalogue_panel()
    with tab_tuning:
        render_tuning_tab()
    with tab_drift:
        render_drift_tab()
    with tab_datasets:
        render_datasets_tab()
    with tab_sim:
        render_simulate_tab()
    with tab_results:
        render_results_tab()
    with tab_summary:
        render_summary_tab()
