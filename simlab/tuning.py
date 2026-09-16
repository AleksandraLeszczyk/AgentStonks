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
from typing import Callable, Optional

from agent_stonks.apple_trader import (
    APPLE_TRADER_KEY,
    RULE_PROVIDER,
    AppleTraderConfig,
    config_signature,
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


def estimated_seconds(spec: dict) -> float:
    """Roughly how long a job takes, from its replay count and session counts."""
    cells = len(grid(spec["axes"]))
    sessions = (cells + 1) * len(spec["tune_dataset"]["days"])
    test = spec.get("test_dataset")
    if test:
        sessions += ((cells if spec.get("sweep_test_grid") else 1) + 1) * len(test["days"])
    workers = max(1, min(int(spec.get("workers") or 1), total_replays(spec)))
    return SECONDS_PER_SESSION * sessions / workers


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


def submit(spec: dict, launch: bool = True) -> dict:
    """Store a job and start its worker. Raises ValueError for an invalid spec."""
    problem = validate(spec)
    if problem:
        raise ValueError(problem)
    job_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    record = {
        "job_id": job_id,
        "created_at": _now_iso(),
        "finished_at": None,
        "status": RUNNING,
        "pid": None,
        "error": None,
        "spec": spec,
        "progress": {"done": 0, "total": total_replays(spec)},
        "cells": {TUNE: [], TEST: []},
        "baseline": {TUNE: None, TEST: None},
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


def run_job(job_id: str, progress: "Callable[[str], None]" = print) -> dict:
    """Sweep, pick, test -- the whole job, writing the record as it goes."""
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

    first = [(TUNE, "cell", c) for c in cells] + [(TUNE, BASELINE, {})]
    if has_test:
        first.append((TEST, BASELINE, {}))
        if spec.get("sweep_test_grid"):
            first += [(TEST, "cell", c) for c in cells]
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
        if best is not None:
            _run_tasks([(TEST, "pick", best["overrides"])], spec, 1, on_done)
        else:
            record["progress"]["done"] += 1  # the pick's replay: nothing was eligible

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
