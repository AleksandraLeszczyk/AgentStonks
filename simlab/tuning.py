"""Parameter tuning for Apple Trader: sweep a grid, pick on one dataset, test on another.

TimeToChange3's notebook 05 sweeps the buy and sell distances over its sessions
and reads a profit heatmap; `scripts/sweep_levels.py` turned that into the
per-ticker defaults this app ships. This module is the same exercise inside
SimLab, with two differences that are the point of doing it here:

* **every cell is a real replay.** Each combination runs through
  `SimulationEngine` with the real `DayRangeTrader` -- market fills near the
  bar's close, the managed exit, the flatten window -- not a re-implementation
  of the rule. What a cell scores is what that configuration would have done in
  a Simulate run, down to the fill. The notebook's limit fills at the level are
  kinder, so expect lower numbers than its heatmap.
* **the pick is tested out of sample.** A grid always has a best cell, and on a
  handful of sessions it is mostly the luckiest one. So the grid is swept on a
  *tuning* dataset, one cell is picked there, and that cell -- next to the
  untuned base configuration -- is replayed on a separate *test* dataset.
  Whether its profit survives is the actual finding; the tuning heatmap alone
  is not. The whole grid can be swept on the test dataset too, which shows
  whether the profitable region moved rather than just whether one cell held.

What a job is
-------------
One JSON record under `data/simlab/tuning/`, plus a sidecar log. The work runs
in a detached worker (`python -m simlab.tuning <job_id>`) for the same reason
experiments do: the simulation clock and `simulation_context` are process
globals, so one process can host one replay at a time. Cells fan out over a
spawn-context process pool inside that worker, and the record is rewritten
after every finished cell, which is what the Tuning tab's progress bar reads.

Up to two parameters are tuned at once -- two is what a heatmap can show. Every
other field comes from the base configuration. A combination the config refuses
(a sell level at or under the buy level, a take fraction of 0) is recorded as
an *invalid* cell rather than replayed or silently dropped, so the heatmap
shows the hole where it is.
"""
from __future__ import annotations

import json
import math
import multiprocessing
import os
import signal
import subprocess
import sys
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Optional

from agent_stonks.apple_trader import (
    APPLE_TRADER_KEY,
    RULE_PROVIDER,
    AppleTraderConfig,
    config_signature,
    model_ticker_error,
)
from agent_stonks.config import (
    BREACH_LABELS,
    BREACH_POLICIES,
    LEVEL_SOURCES,
    LEVEL_SOURCE_LABELS,
)
from agent_stonks.market_hours import MARKET_TZ

TUNING_DIR = Path(__file__).resolve().parent.parent / "data" / "simlab" / "tuning"
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

RUNNING = "running"
FINISHED = "finished"
FAILED = "failed"
STOPPED_ERROR = "stopped by user"

TUNE = "tune"
TEST = "test"
BASELINE = "baseline"

# How many parameters one job sweeps: two is what a heatmap can show.
MAX_AXES = 2
# A guard against a slip of the step size queueing hours of replays.
MAX_CELLS = 400
# The same guard for the Simulate tab's per-setup sweep, and much lower for a
# reason: a tuning cell is a replay on a pooled worker inside one job, while a
# swept configuration there is a queued *experiment* -- its own subprocess, its
# own run record, its own row in Results. A hundred of those is an afternoon
# and a Results page nobody reads.
MAX_SWEEP_CONFIGS = 64

# How the best cell is chosen on the tuning dataset.
PICK_MAX = "max"
PICK_PLATEAU = "plateau"

# What a cell is scored by. Every metric is "higher is better". The labels
# carry no "$": they reach Streamlit markdown, where a bare dollar sign opens a
# formula.
METRICS = {
    "profit": "Total profit",
    "return_pct": "Return",
    "days_up": "Profitable days",
    "worst_day": "Worst day",
}

# A day's profit below this many dollars in either direction is a flat day,
# not a win or a loss -- float noise on a ledger that never traded.
_FLAT_DAY_USD = 0.005


@dataclass(frozen=True)
class Tunable:
    """One `AppleTraderConfig` field a grid can sweep, and its legal range.

    `default_range` is the (from, to, step) the form starts on: coarse enough
    that a two-parameter grid stays a few dozen replays, wide enough to reach
    past every shipped default.
    """

    name: str
    label: str
    minimum: float
    maximum: float
    step: float
    default_range: "tuple[float, float, float]"
    fmt: str = "%.2f"
    integer: bool = False


TUNABLES: "dict[str, Tunable]" = {
    t.name: t
    for t in (
        Tunable("buy_k", "Buy distance (× ADR below H)", 0.05, 3.0, 0.05, (0.30, 0.90, 0.10)),
        Tunable("sell_k", "Sell distance (× ADR below H)", 0.0, 3.0, 0.05, (0.05, 0.45, 0.10)),
        Tunable(
            "stop_gain_fraction", "Stop loss (× the predicted gain)", 0.0, 3.0, 0.05,
            (0.0, 1.0, 0.25),
        ),
        Tunable(
            "momentum_drop", "Momentum fade (σ off its peak)", 0.0, 5.0, 0.1,
            (0.0, 2.0, 0.5), "%.1f",
        ),
        Tunable("take_fraction", "Share taken on a fade", 0.05, 1.0, 0.05, (0.30, 1.0, 0.10)),
        Tunable(
            "hold_min_gain_k", "Keep a runner if target ≥ (× ADR)", 0.0, 3.0, 0.05,
            (0.0, 0.60, 0.10),
        ),
        Tunable(
            "min_win_k", "Stand down under (× ADR a share)", 0.0, 3.0, 0.05,
            (0.0, 0.40, 0.10),
        ),
        Tunable(
            "position_pct", "Position size (% of cash)", 1.0, 100.0, 5.0,
            (25.0, 100.0, 25.0), "%.0f",
        ),
        Tunable(
            "flatten_before_close_min", "Flatten before close (min)", 1, 60, 1,
            (1, 15, 2), "%d", integer=True,
        ),
    )
}

@dataclass(frozen=True)
class Choice:
    """One `AppleTraderConfig` field varied over a fixed set of rules.

    The counterpart of `Tunable` for the settings that are not numbers: which
    reference the levels hang off, and what happens when the session trades
    through the forecast. They have no range and no step -- a sweep over one is
    a set of named alternatives, and the whole set is the useful default.

    Absent from the Tuning tab, which draws a heatmap of two numeric axes and
    picks a cell on it; these belong to the Simulate tab's per-setup sweep,
    where the output is several queued runs to compare rather than a surface.
    """

    name: str
    label: str
    options: "tuple[str, ...]"
    #: option -> what the form and the captions call it.
    labels: "dict[str, str]"


CHOICES: "dict[str, Choice]" = {
    c.name: c
    for c in (
        Choice(
            "level_source", "Levels measured below",
            tuple(LEVEL_SOURCES), dict(LEVEL_SOURCE_LABELS),
        ),
        Choice(
            "breach_update", "If the session trades outside the forecast",
            tuple(BREACH_POLICIES), dict(BREACH_LABELS),
        ),
    )
}

# What the Simulate tab's per-setup sweep offers, in the order the form lists
# it: the two levels, then the rules about what they hang off, then the exits
# and the sizing -- the same reading order as the setup form above it.
#
# `flatten_before_close_min` is a `Tunable` and is deliberately not here. The
# run signature does not carry it (`config_signature`), and Results identifies
# a configuration by that signature, so varying it would queue several runs
# that collapse into one row -- a grid that looks like a comparison and is not.
SWEEPABLE: "tuple[str, ...]" = (
    "buy_k",
    "sell_k",
    "level_source",
    "breach_update",
    "stop_gain_fraction",
    "momentum_drop",
    "take_fraction",
    "hold_min_gain_k",
    "min_win_k",
    "position_pct",
)


def sweep_label(name: str) -> str:
    """One sweepable field's name, whether it is numeric or a set of rules."""
    field = TUNABLES.get(name) or CHOICES.get(name)
    return field.label if field else name


def value_label(name: str, value) -> str:
    """One value of one sweepable field, as the form and the captions show it."""
    choice = CHOICES.get(name)
    if choice is not None:
        return choice.labels.get(value, str(value))
    tunable = TUNABLES.get(name)
    try:
        return tunable.fmt % value if tunable else str(value)
    except TypeError:
        return str(value)


# Wall-clock seconds one worker spends replaying one session, for the form's
# estimate. Measured on a stored AAPL week (a 4-session replay in ~0.6-0.8 s
# once the market index and the replay minute frame were in); each pool worker
# also pays a few seconds up front to import torch and load the bundle. A rough
# guide, not a promise.
SECONDS_PER_SESSION = 0.25


def estimated_seconds(spec: dict, prior: "dict | None" = None) -> float:
    """Roughly how long a job takes, from its replay count and session counts.

    `prior` is `prior_cells`' answer: those replays are already on disk and the
    job skips them, so they cost nothing but still count towards its total.
    """
    cells = len(grid(spec["axes"]))
    prior = prior or {}
    sessions = (cells + 1 - len(prior.get(TUNE) or ())) * len(spec["tune_dataset"]["days"])
    test = spec.get("test_dataset")
    if test:
        planned = (cells if spec.get("sweep_test_grid") else 1) + 1
        sessions += (planned - len(prior.get(TEST) or ())) * len(test["days"])
    workers = max(1, min(int(spec.get("workers") or 1), total_replays(spec)))
    return SECONDS_PER_SESSION * max(sessions, 0) / workers


# --- the grid ---------------------------------------------------------------


def axis_values(name: str, start: float, stop: float, step: float) -> list:
    """`start, start+step, ... <= stop` for one parameter, rounded to clean numbers."""
    tunable = TUNABLES[name]
    if step <= 0:
        raise ValueError(f"{tunable.label}: the step must be positive.")
    if stop < start:
        raise ValueError(f"{tunable.label}: the end is below the start.")
    if start < tunable.minimum or stop > tunable.maximum:
        raise ValueError(
            f"{tunable.label}: allowed range is {tunable.minimum:g} to {tunable.maximum:g}."
        )
    count = int(math.floor((stop - start) / step + 1e-9)) + 1
    values = [round(start + i * step, 6) for i in range(count)]
    return [int(round(v)) for v in values] if tunable.integer else values


def cell_count(axes: "list[dict]") -> int:
    """How many combinations `grid` would produce, without producing them.

    A size guard has to run *before* the grid exists: seven axes of ten values
    is ten million override dicts, and a form that builds them to find out it
    should not have is a hung page rather than a refusal.
    """
    count = 1
    for axis in axes:
        count *= len(axis["values"])
    return count


def grid(axes: "list[dict]") -> "list[dict]":
    """Every combination of the axes' values, as override dicts, row by row."""
    if not axes:
        return [{}]
    names = [axis["name"] for axis in axes]
    return [dict(zip(names, combo)) for combo in product(*(axis["values"] for axis in axes))]


def make_config(base: dict, overrides: dict) -> AppleTraderConfig:
    """The base configuration with one cell's values on top (may raise ValueError).

    Decoded through `rule_agents`, not straight into the dataclass, because a
    job's `base` is a *stored record* and outlives the fields it was written
    with. A key that did not exist when the job was saved has to decode to what
    its absence meant then (`_APPLE_LEGACY`) rather than to today's default --
    otherwise re-opening a job silently re-signs and re-runs it as a strategy it
    was never tuned under, and the heatmap already on screen belongs to a
    different configuration from the one its caption names.
    """
    from .rule_agents import rule_agent

    return rule_agent(APPLE_TRADER_KEY).from_record({**base, **overrides})


def expand(
    base: AppleTraderConfig, axes: "list[dict]"
) -> "tuple[list[AppleTraderConfig], list[tuple[dict, str]]]":
    """`base` crossed with every combination of `axes`: (configurations, refused).

    The Simulate tab's sweep, and the one place that decides what a grid of
    settings actually queues. `axes` is the same `[{"name", "values"}]` shape
    `grid` takes, so a sweep and a tuning job describe a grid identically.

    Two kinds of cell never become a run, and they are different:

    * **refused** -- the configuration is not a strategy at all, and
      `AppleTraderConfig` says why (a sell level at or under the buy level, a
      take fraction of nothing). Returned with its reason rather than dropped,
      because a grid quietly one row short is worse than one that explains
      itself.
    * **collapsed** -- the configuration is real but signs the same as one
      already in the list, which happens whenever a varied field is switched
      off by another (`take_fraction` means nothing with `momentum_drop` at 0,
      and the signature leaves it out). The whole pipeline identifies a run by
      its signature, so queueing both would be one Results row run twice.
      Silently collapsed here; the caller compares against `cell_count(axes)`
      to say how many.

    Order is the grid's, so the first configuration is every axis at its first
    value and the list reads like the form above it.
    """
    record = asdict(base)
    configs: "list[AppleTraderConfig]" = []
    refused: "list[tuple[dict, str]]" = []
    seen: "set[str]" = set()
    for overrides in grid(axes):
        try:
            config = make_config(record, overrides)
        except (TypeError, ValueError) as exc:
            refused.append((overrides, str(exc)))
            continue
        signature = config_signature(config)
        if signature in seen:
            continue
        seen.add(signature)
        configs.append(config)
    return configs, refused


# --- one cell ---------------------------------------------------------------


def evaluate(dataset: dict, base: dict, overrides: dict, starting_cash: float) -> dict:
    """Replay Apple Trader over one dataset with `overrides` applied to `base`.

    Module-level and dependency-light so a spawn-context pool worker can import
    and call it. The market is loaded per call: it is a few JSON files and
    costs milliseconds, where holding one per process would make the result
    depend on which worker happened to run the cell.
    """
    from .engine import SimulationConfig, SimulationEngine
    from .market import SimMarket

    try:
        config = make_config(base, overrides)
    except (TypeError, ValueError) as exc:
        return {"overrides": overrides, "invalid": str(exc)}

    days = [date.fromisoformat(d) for d in dataset["days"]]
    market = SimMarket([config.ticker], days, dataset["feed"])
    sim = SimulationConfig(
        personality=APPLE_TRADER_KEY,
        provider=RULE_PROVIDER,
        model=config_signature(config),
        api_key="",
        symbols=[config.ticker],
        days=days,
        starting_cash=float(starting_cash),
        rule_config=asdict(config),
        feed=dataset["feed"],
    )
    result = SimulationEngine(market, sim).run()
    return {"overrides": overrides, **score(result, days)}


def score(result, days: "list[date]") -> dict:
    """A replay's result as the numbers a grid is compared on.

    The account runs across the whole dataset, as in a Simulate run, and a
    day's profit is the change in marked equity from the previous day's last
    step to this day's -- the trader flattens before every close, so each day
    is close to an independent sample.
    """
    last_value: "dict[date, float]" = {}
    for point in result.equity:
        stamp = datetime.fromisoformat(str(point["ts"]))
        last_value[stamp.astimezone(MARKET_TZ).date()] = float(point["value"])

    daily: "dict[str, float]" = {}
    previous = float(result.starting_cash)
    for day in days:
        value = last_value.get(day, previous)
        daily[day.isoformat()] = round(value - previous, 2)
        previous = value

    buys_by_day: "dict[str, int]" = {}
    fills = [d for d in result.decisions if d.get("status") == "filled"]
    for decision in fills:
        if decision.get("action") != "buy":
            continue
        day = datetime.fromisoformat(str(decision["ts"])).astimezone(MARKET_TZ).date()
        buys_by_day[day.isoformat()] = buys_by_day.get(day.isoformat(), 0) + 1

    no_forecast = {
        datetime.fromisoformat(str(e["ts"])).astimezone(MARKET_TZ).date().isoformat()
        for e in result.agent_log
        if e.get("type") == "error" and "cannot forecast" in str(e.get("text", ""))
    }
    profits = list(daily.values())
    profit = float(result.final_value) - float(result.starting_cash)
    return {
        "profit": round(profit, 2),
        "return_pct": round(100.0 * profit / result.starting_cash, 4)
        if result.starting_cash else 0.0,
        "trades": sum(buys_by_day.values()),
        "sells": sum(1 for d in fills if d.get("action") == "sell"),
        "days": len(days),
        "days_traded": len(buys_by_day),
        "days_up": sum(1 for p in profits if p > _FLAT_DAY_USD),
        "days_down": sum(1 for p in profits if p < -_FLAT_DAY_USD),
        "worst_day": min(profits) if profits else 0.0,
        "best_day": max(profits) if profits else 0.0,
        "daily": daily,
        "no_forecast_days": sorted(no_forecast),
        "error": result.error,
    }


def is_scored(cell: "dict | None") -> bool:
    return bool(cell) and "invalid" not in cell and not cell.get("error")


# --- cells a stored run already answers -------------------------------------
#
# A tuning cell and a Simulate run are the same thing: `evaluate` builds the
# identical `SimulationConfig` the Simulate tab does and hands it to the same
# engine. So a cell whose configuration, sessions, tape and starting cash match
# a run already in the store has an answer on disk, and replaying it would
# produce that answer again -- the replay is deterministic, which is what
# `test_a_cell_is_exactly_a_simulate_run` pins.
#
# Two uses follow from that. The Tuning form draws the grid a job *would*
# sweep filled in wherever the store already covers it, before anything is
# queued; and the job itself is seeded with those cells and only replays the
# holes.
#
# What has to match is every field of the decoded configuration -- not the
# signature, which deliberately leaves out `flatten_before_close_min` and
# writes a switched-off exit as nothing, so two runs that share one are not
# necessarily the same replay. Records are decoded through `make_config`
# first, so a run stored before a field existed matches the cell that means
# what it meant then rather than falling out for lack of a key.


def overrides_key(overrides: "dict | None") -> str:
    """One cell's overrides as a dict key, independent of the order they were
    written in."""
    return json.dumps(overrides or {}, sort_keys=True)


def _days_key(days) -> "tuple[str, ...]":
    """A dataset's sessions as an order-independent key -- the tape a replay
    reads, not the order a form listed it in."""
    return tuple(sorted(str(d)[:10] for d in days or ()))


def _config_key(config: AppleTraderConfig) -> str:
    return json.dumps(asdict(config), sort_keys=True, default=str)


def _replay_key(config: AppleTraderConfig, days, feed, starting_cash) -> tuple:
    """Everything that decides what a replay of `config` produces.

    The cash is in it because every metric a grid is read on is a dollar
    amount, and the tape is because fills differ between feeds -- the same
    reason the Tuning form warns when the two datasets disagree about it.
    """
    return (
        _config_key(config), str(feed or ""), _days_key(days),
        round(float(starting_cash or 0.0), 2),
    )


def _replay_inputs(config: AppleTraderConfig, days, feed) -> "list[Path]":
    """Every file a replay of this configuration reads.

    A replay is a pure function of these: the dataset's minute bars and news
    for each session, the symbol's daily history, the market indicators, and
    the saved model's bundle with its sidecar checkpoints.

    The last three are the ones that move. The daily history and the
    indicators are a rolling store shared by every dataset, not part of the
    dataset, and refreshing them changes what the day-range model forecasts for
    sessions that were downloaded months ago -- which moves both levels, and so
    every fill.
    """
    from . import data as sim_data
    from agent_stonks import apple_models

    symbol = config.ticker.upper()
    paths = [sim_data.market_path(), sim_data.stored_daily_path(symbol, feed)]
    for day in days or ():
        stamp = date.fromisoformat(str(day)[:10])
        paths.append(sim_data.stored_bars_path(symbol, stamp, feed))
        paths.append(sim_data.news_path(symbol, stamp))
    bundle = apple_models.get(config.model_key).path(symbol)
    paths.extend(sorted(bundle.parent.glob(f"{bundle.stem}*")))
    return paths


def _last_changed(paths: "list[Path]") -> "datetime | None":
    """When any of these files was last written, or None if none exists."""
    newest = None
    for path in paths:
        try:
            stamp = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
        except OSError:
            continue
        if newest is None or stamp > newest:
            newest = stamp
    return newest


def run_is_stale(record: dict) -> bool:
    """Whether a stored run was made before something it read last changed.

    The reason a match on the configuration and the sessions is not enough. A
    replay is deterministic given its inputs, but its inputs are files on disk
    that outlive the run: re-downloading a symbol's daily history rewrites what
    the day-range model sees, and a run saved before that describes a forecast
    the same configuration would no longer produce. Seen in the wild -- a run
    of 2026-09-09 armed its buy at 315.82 off a predicted high of 318.19, and
    the same replay after the daily file was refreshed on 2026-09-12 arms it at
    316.00 off 318.40.

    Modification times, not content hashes: a stored record says when it was
    written and nothing about what it read, so this is the only signal an
    already-saved run offers. It errs the safe way -- a re-download that wrote
    identical bytes makes a run look stale and it is replayed again, which
    costs seconds and cannot produce a wrong number.

    It cannot see a change in *code*, which no timestamp on a data file
    reports. A rework of the trader's rules therefore wants the affected runs
    deleted, or reuse switched off for the job.
    """
    try:
        saved = datetime.fromisoformat(str(record.get("created_at")))
        config = make_config((record.get("config_summary") or {}).get("rule_config") or {}, {})
    except (TypeError, ValueError):
        return True
    if saved.tzinfo is None:
        return True
    summary = record.get("config_summary") or {}
    changed = _last_changed(_replay_inputs(config, summary.get("days"), summary.get("feed")))
    return changed is not None and changed > saved


def run_replay_key(record: dict) -> "tuple | None":
    """Which replay a stored simulation run *is*, or None if it is not one.

    None covers every run that cannot stand in for a cell: an LLM run, a
    different rule agent, a run that errored or was interrupted before the end,
    a rule set this build can no longer decode or would refuse to run (a record
    naming a removed model -- see `rule_agents._APPLE_LEGACY`), and a run whose
    inputs have moved on under it (`run_is_stale`).

    A run whose day-range model could not forecast some sessions is *not*
    excluded: those days are an ordinary part of a replay and `score` reports
    them (`no_forecast_days`) exactly as it would for a freshly swept cell.
    """
    config = record.get("config_summary") or {}
    if config.get("personality") != APPLE_TRADER_KEY or not config.get("rule_based"):
        return None
    if record.get("error") or record.get("interrupted") or not (record.get("cycles_run") or 0):
        return None
    try:
        decoded = make_config(config.get("rule_config") or {}, {})
    except (TypeError, ValueError):
        return None
    if model_ticker_error(decoded) is not None:
        return None
    if run_is_stale(record):
        return None
    return _replay_key(
        decoded, config.get("days"), config.get("feed"), config.get("starting_cash")
    )


def score_run(record: dict, days: "list[date]") -> dict:
    """A stored run scored as a grid cell.

    A saved record carries the engine's own result at its top level
    (`results.save_run` spreads `asdict(result)` into it), so this is the same
    `score` call `evaluate` makes on a fresh replay, over the same fields.
    """
    return score(SimpleNamespace(**record), days)


def runs_for_pair(runs: "list[dict]", ticker: str, model_key: str) -> "list[dict]":
    """Stored Apple Trader runs for one instrument and one saved model.

    The pair a tuning grid is about: the model decides which rules exist and
    the symbol is part of the model's identity, so no run outside the pair can
    ever answer one of its cells. Used for the Tuning form's count of what the
    store holds -- how much of it lands on *this* grid is `prior_cells`.
    """
    wanted = str(ticker or "").strip().upper()
    found = []
    for record in runs:
        config = record.get("config_summary") or {}
        if config.get("personality") != APPLE_TRADER_KEY or not config.get("rule_based"):
            continue
        rule_config = config.get("rule_config") or {}
        if str(rule_config.get("ticker") or "").strip().upper() != wanted:
            continue
        if str(rule_config.get("model_key") or "") == str(model_key or ""):
            found.append(record)
    return found


def index_runs(runs: "list[dict]") -> "dict[tuple, dict]":
    """Stored runs keyed by the replay each one is, newest first wins.

    Newest rather than best, deliberately: two runs with the same key are the
    same replay of the same configuration over the same tape, so they agree,
    and where a code change has made them disagree the later one is the one
    this build would produce.
    """
    index: "dict[tuple, dict]" = {}
    for record in runs:
        key = run_replay_key(record)
        if key is not None:
            index.setdefault(key, record)
    return index


def prior_cell(
    index: "dict[tuple, dict]", spec: dict, dataset: dict, overrides: dict
) -> "dict | None":
    """One grid cell answered out of the run index, or None if nothing matches.

    An invalid combination is never looked up: it is a hole in the grid, and
    the caller marks it as one.
    """
    try:
        config = make_config(spec["base"], overrides)
    except (TypeError, ValueError):
        return None
    record = index.get(
        _replay_key(config, dataset.get("days"), dataset.get("feed"), spec["starting_cash"])
    )
    if record is None:
        return None
    days = [date.fromisoformat(str(d)[:10]) for d in dataset["days"]]
    return {
        "overrides": dict(overrides),
        "run_id": record.get("run_id") or "",
        "run_dataset": record.get("dataset") or "",
        **score_run(record, days),
    }


def prior_cells(
    spec: dict, runs: "list[dict] | None" = None
) -> "dict[str, dict[str, dict]]":
    """Every cell of this job's grid the run store already answers.

    Returns `{role: {overrides_key: cell}}` for the tuning and test datasets,
    with the baseline under `overrides_key({})`. The test grid is only looked
    up when the job would sweep it -- a job that does not sweep it is not
    asking about those cells, and half-filling a heatmap it never draws would
    put rows in Results tables for a comparison nobody asked for. Its baseline
    is looked up either way, since every job with a test dataset replays that.

    Nothing here runs a replay, so this is cheap enough for the form to call on
    every rerun -- it parses the store (cached upstream) and scores the records
    that land on the grid.
    """
    found: "dict[str, dict[str, dict]]" = {TUNE: {}, TEST: {}}
    if not spec.get("reuse_runs", True):
        return found
    if runs is None:
        from .results import list_runs

        runs = list_runs()
    index = index_runs(runs)
    if not index:
        return found
    cells = grid(spec["axes"])
    for role, dataset, swept in (
        (TUNE, spec.get("tune_dataset"), True),
        (TEST, spec.get("test_dataset"), bool(spec.get("sweep_test_grid"))),
    ):
        if not dataset:
            continue
        for overrides in ([*cells, {}] if swept else [{}]):
            cell = prior_cell(index, spec, dataset, overrides)
            if cell is not None:
                found[role][overrides_key(overrides)] = cell
    return found


def reused_count(record: dict) -> int:
    """How many of a job's replays came out of the run store rather than a
    fresh sweep -- countable against `progress["total"]`.

    `best_test` is only its own replay when the test grid was not swept; where
    it was, the pick's test cell is already one of `cells[TEST]` and counting
    it again would claim more replays than the job has.
    """
    cells = [
        *record["cells"][TUNE], *record["cells"][TEST],
        record["baseline"].get(TUNE), record["baseline"].get(TEST),
    ]
    if not record["spec"].get("sweep_test_grid"):
        cells.append(record.get("best_test"))
    return sum(1 for c in cells if c and c.get("run_id"))


# --- picking ----------------------------------------------------------------


def _position(cell: dict, axes: "list[dict]") -> "tuple[int, ...]":
    return tuple(axis["values"].index(cell["overrides"][axis["name"]]) for axis in axes)


def pick_best(
    cells: "list[dict]",
    axes: "list[dict]",
    metric: str = "profit",
    rule: str = PICK_MAX,
    min_traded_share: float = 0.0,
) -> "dict | None":
    """The cell to carry to the test dataset, with the `pick_score` it won on.

    `PICK_MAX` takes the highest metric. `PICK_PLATEAU` takes the cell whose
    neighbourhood -- itself and every adjacent cell on the grid, diagonals
    included -- scores best on average, which is how `sweep_levels.py` chose
    the shipped defaults: the middle of a profitable region rather than its
    sharpest point, which on a few sessions is usually a fluke.

    A cell is eligible only if it traded on at least `min_traded_share` of the
    days, so a deep entry that filled once cannot win on that one fill.
    Invalid cells and cells whose replay errored are never picked and never
    count as a neighbour. Ties go to the earlier cell in grid order.
    """
    scored = [c for c in cells if is_scored(c)]
    if not scored:
        return None
    by_position = {_position(c, axes): c for c in scored}

    def plateau(cell: dict) -> float:
        here = _position(cell, axes)
        values = [
            float(other[metric])
            for offset in product((-1, 0, 1), repeat=len(axes))
            if (other := by_position.get(tuple(p + o for p, o in zip(here, offset))))
        ]
        return sum(values) / len(values)

    eligible = [
        c for c in scored if c["days_traded"] >= min_traded_share * max(c["days"], 1)
    ]
    if not eligible:
        return None
    rank = plateau if rule == PICK_PLATEAU else (lambda c: float(c[metric]))
    best = max(eligible, key=rank)  # max() keeps the first of equal keys
    return {**best, "pick_score": round(rank(best), 4)}


def overlapping_days(first: "list[str]", second: "list[str]") -> "list[str]":
    """Sessions two datasets share -- a test on them is not out of sample."""
    return sorted(set(first) & set(second))


# --- the job store ----------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _path(job_id: str) -> Path:
    return TUNING_DIR / f"{job_id}.json"


def log_path(job_id: str) -> Path:
    return TUNING_DIR / f"{job_id}.log"


def _write(record: dict) -> None:
    """Temp file + rename, so the UI never reads a half-written record."""
    TUNING_DIR.mkdir(parents=True, exist_ok=True)
    path = _path(record["job_id"])
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=2, default=str))
    tmp.replace(path)


def get_job(job_id: str) -> "dict | None":
    try:
        return json.loads(_path(job_id).read_text())
    except (OSError, json.JSONDecodeError):
        return None


def list_jobs() -> "list[dict]":
    """Every job record, newest first, with dead workers marked failed."""
    if not TUNING_DIR.exists():
        return []
    jobs = []
    for path in sorted(TUNING_DIR.glob("*.json"), reverse=True):
        try:
            jobs.append(_reap(json.loads(path.read_text())))
        except (OSError, json.JSONDecodeError):
            continue
    return jobs


def delete_job(job_id: str) -> None:
    for path in (_path(job_id), log_path(job_id)):
        if path.exists():
            path.unlink()


def total_replays(spec: dict) -> int:
    """How many replays a job runs: the grid on the tuning dataset plus the
    baseline, then on the test dataset either the grid or just the pick, plus
    the baseline again."""
    cells = len(grid(spec["axes"]))
    total = cells + 1
    if spec.get("test_dataset"):
        total += (cells if spec.get("sweep_test_grid") else 1) + 1
    return total


def validate(spec: dict) -> "str | None":
    """Why this job cannot run, or None."""
    axes = spec.get("axes") or []
    if not 1 <= len(axes) <= MAX_AXES:
        return f"Pick one or two parameters to tune (got {len(axes)})."
    if len({a["name"] for a in axes}) != len(axes):
        return "The same parameter is tuned twice."
    for axis in axes:
        if axis["name"] not in TUNABLES:
            return f"{axis['name']} is not a tunable parameter."
        if not axis.get("values"):
            return f"{TUNABLES[axis['name']].label}: no values to try."
    cells = len(grid(axes))
    if cells > MAX_CELLS:
        return f"{cells} combinations is more than the {MAX_CELLS} one job may sweep."
    ticker = spec["base"]["ticker"]
    for role in ("tune_dataset", "test_dataset"):
        dataset = spec.get(role)
        if dataset is None:
            continue
        if ticker not in (dataset.get("symbols") or []):
            return f"Dataset {dataset['name']} does not carry {ticker}."
    if not spec.get("tune_dataset"):
        return "Pick a dataset to tune on."
    return None


def submit(spec: dict, launch: bool = True, runs: "list[dict] | None" = None) -> dict:
    """Store a job and start its worker. Raises ValueError for an invalid spec.

    The record is seeded with every cell the run store already answers
    (`prior_cells`), so the heatmap is partly drawn the moment the job appears
    and the worker only replays the holes. Those cells count as done against
    the job's full total, which stays what the grid asks for -- a job that
    reused half its grid ran the whole grid, it just did not have to replay it.

    `runs` is the already-parsed store when the caller has one (the UI caches
    it); left out, the store is read here.
    """
    problem = validate(spec)
    if problem:
        raise ValueError(problem)
    job_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    prior = prior_cells(spec, runs)
    base_key = overrides_key({})
    reused = sum(len(found) for found in prior.values())
    record = {
        "job_id": job_id,
        "created_at": _now_iso(),
        "finished_at": None,
        "status": RUNNING,
        "pid": None,
        "error": None,
        "spec": spec,
        "progress": {"done": reused, "total": total_replays(spec), "reused": reused},
        "cells": {
            role: [cell for key, cell in prior[role].items() if key != base_key]
            for role in (TUNE, TEST)
        },
        "baseline": {role: prior[role].get(base_key) for role in (TUNE, TEST)},
        "best": None,
        "best_test": None,
    }
    _write(record)
    if launch:
        TUNING_DIR.mkdir(parents=True, exist_ok=True)
        log = open(log_path(job_id), "ab")
        proc = subprocess.Popen(
            [sys.executable, "-m", "simlab.tuning", job_id],
            cwd=_PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        record["pid"] = proc.pid
        _write(record)
    return record


def _pid_alive(pid) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return False
    return True


def _reap(record: dict) -> dict:
    """A running record whose worker is gone is a failed job, not a stuck one."""
    if record.get("status") == RUNNING and record.get("pid") and not _pid_alive(record["pid"]):
        record.update(status=FAILED, error="worker process died", finished_at=_now_iso())
        _write(record)
    return record


def stop(job_id: str) -> "dict | None":
    """Kill the worker and its pool (one process group) and mark the job failed.

    The cells finished so far stay in the record: unlike an experiment, a
    half-swept grid is still a picture of the half that was swept.
    """
    record = get_job(job_id)
    if record is None or record["status"] != RUNNING:
        return record
    if _pid_alive(record.get("pid")):
        try:
            os.killpg(os.getpgid(int(record["pid"])), signal.SIGTERM)
        except (OSError, ValueError):
            pass
    record = get_job(job_id) or record
    if record["status"] == RUNNING:
        record.update(status=FAILED, error=STOPPED_ERROR, finished_at=_now_iso())
        _write(record)
    return record


def last_log_line(job_id: str) -> str:
    try:
        lines = log_path(job_id).read_text().strip().splitlines()
        return lines[-1] if lines else ""
    except OSError:
        return ""


# --- the worker -------------------------------------------------------------


def _run_tasks(
    tasks: "list[tuple[str, str, dict]]",
    spec: dict,
    workers: int,
    on_done: "Callable[[str, str, dict], None]",
) -> None:
    """Evaluate (role, kind, overrides) tasks, calling `on_done` as each lands.

    One worker runs them in this process, in order -- which is what the tests
    use. More fan out over a spawn-context pool: spawn rather than fork,
    because every child ends up importing torch through the day-range model and
    a forked OpenMP runtime is not something to rely on.
    """
    def dataset_for(role: str) -> dict:
        return spec["tune_dataset"] if role == TUNE else spec["test_dataset"]

    cash = spec["starting_cash"]
    if workers <= 1:
        for role, kind, overrides in tasks:
            on_done(role, kind, evaluate(dataset_for(role), spec["base"], overrides, cash))
        return
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        futures = {
            pool.submit(evaluate, dataset_for(role), spec["base"], overrides, cash): (role, kind)
            for role, kind, overrides in tasks
        }
        for future in as_completed(futures):
            role, kind = futures[future]
            on_done(role, kind, future.result())


def _stored_cell(spec: dict, dataset: dict, overrides: dict) -> "dict | None":
    """One cell answered out of the run store, reading the store on the spot.

    `prior_cells` covers everything a job knows about at submit; this is for
    the one replay it cannot -- the pick's, on a test dataset whose grid is not
    being swept, since which cell wins is only settled once the sweep is done.
    """
    if not spec.get("reuse_runs", True):
        return None
    from .results import list_runs

    return prior_cell(index_runs(list_runs()), spec, dataset, overrides)


def run_job(job_id: str, progress: "Callable[[str], None]" = print) -> dict:
    """Sweep, pick, test -- the whole job, writing the record as it goes.

    Only the replays the record does not already hold are run: `submit` seeds
    it with every cell the run store answers, and what is left is the holes.
    """
    record = get_job(job_id)
    if record is None:
        raise RuntimeError(f"unknown tuning job {job_id}")
    spec = record["spec"]
    axes = spec["axes"]
    cells = grid(axes)
    workers = max(1, int(spec.get("workers") or 1))
    has_test = bool(spec.get("test_dataset"))

    def on_done(role: str, kind: str, result: dict) -> None:
        if kind == BASELINE:
            record["baseline"][role] = result
        elif kind == "pick":
            record["best_test"] = result
        else:
            record["cells"][role].append(result)
        record["progress"]["done"] += 1
        _write(record)
        detail = result.get("invalid") or result.get("error") or f"${result.get('profit', 0):+,.2f}"
        progress(
            f"[{record['progress']['done']}/{record['progress']['total']}] {role} "
            f"{kind} {result['overrides'] or 'base'}: {detail}"
        )

    def missing(role: str) -> "list[dict]":
        """The grid's cells this record has no answer for yet, in grid order."""
        have = {overrides_key(c["overrides"]) for c in record["cells"][role]}
        return [c for c in cells if overrides_key(c) not in have]

    first = [(TUNE, "cell", c) for c in missing(TUNE)]
    if record["baseline"][TUNE] is None:
        first.append((TUNE, BASELINE, {}))
    if has_test:
        if record["baseline"][TEST] is None:
            first.append((TEST, BASELINE, {}))
        if spec.get("sweep_test_grid"):
            first += [(TEST, "cell", c) for c in missing(TEST)]
    _run_tasks(first, spec, workers, on_done)

    record["cells"][TUNE].sort(key=lambda c: cells.index(c["overrides"]))
    record["cells"][TEST].sort(key=lambda c: cells.index(c["overrides"]))
    best = pick_best(
        record["cells"][TUNE], axes, spec.get("metric", "profit"),
        spec.get("rule", PICK_MAX), float(spec.get("min_traded_share") or 0.0),
    )
    record["best"] = best
    _write(record)

    if has_test and spec.get("sweep_test_grid"):
        # The pick was already replayed on the test dataset as part of its grid.
        if best is not None:
            record["best_test"] = next(
                (c for c in record["cells"][TEST] if c["overrides"] == best["overrides"]), None
            )
    elif has_test:
        if best is None:
            record["progress"]["done"] += 1  # the pick's replay: nothing was eligible
        else:
            # The pick is only known now, so its test replay could not be
            # seeded at submit -- but the store may still answer it.
            stored = _stored_cell(spec, spec["test_dataset"], best["overrides"])
            if stored is None:
                _run_tasks([(TEST, "pick", best["overrides"])], spec, 1, on_done)
            else:
                record["best_test"] = stored
                record["progress"]["done"] += 1
                record["progress"]["reused"] = record["progress"].get("reused", 0) + 1
                _write(record)
                progress(
                    f"[{record['progress']['done']}/{record['progress']['total']}] {TEST} "
                    f"pick {best['overrides']}: reused run {stored['run_id']}"
                )

    record.update(status=FINISHED, finished_at=_now_iso())
    _write(record)
    progress(f"Tuning job {job_id} finished.")
    return record


def main(argv: "list[str]") -> None:
    if len(argv) != 2:
        raise SystemExit("usage: python -m simlab.tuning <job_id>")
    job_id = argv[1]
    try:
        run_job(job_id, progress=lambda m: print(m, flush=True))
    except Exception as exc:
        record = get_job(job_id)
        if record is not None:
            record.update(status=FAILED, error=str(exc), finished_at=_now_iso())
            _write(record)
        raise


if __name__ == "__main__":
    main(sys.argv)
