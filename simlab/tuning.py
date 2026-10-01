"""Parameter tuning for Apple Trader: sweep one grid over several datasets and sum them.

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
* **the grid is swept on every dataset of the job, and the pick is read off
  their sum.** A grid always has a best cell, and on a handful of sessions it
  is mostly the luckiest one. One week's heatmap says little; the same grid
  over several weeks, heatmap by heatmap and then summed, shows which region
  keeps paying. The pick is the best total profit on that sum.

A job grows. Tuning is a weekly chore -- a new week of tape arrives, and the
question is whether the pick still holds with it -- so a finished job takes a
new dataset (`add_dataset`): the same base configuration and axes are swept
over it, and the sum and the pick are re-read. A dataset can be dropped again
(`remove_dataset`), which costs no replay at all.

Jobs written before this (2026-09-26) had exactly two datasets, a *tune* and a
*test* one, under `spec.tune_dataset` / `spec.test_dataset`. `_upgrade` reads
them as a job over those two datasets, so they take part in the new workflow
like any other.

What a job is
-------------
One JSON record under `data/simlab/tuning/`, plus a sidecar log. Cells are kept
per dataset, keyed by the dataset's name (`record["cells"][name]`, likewise
`record["baseline"][name]`). The work runs in a detached worker
(`python -m simlab.tuning <job_id>`) for the same reason experiments do: the
simulation clock and `simulation_context` are process globals, so one process
can host one replay at a time. Cells fan out over a spawn-context process pool
inside that worker, and the record is rewritten after every finished cell,
which is what the Tuning tab's progress bar reads. The worker replays whatever
the record has no answer for yet, which is what makes adding a dataset or
resuming a stopped job the same operation as running a new one.

Up to two parameters are tuned at once -- two is what a heatmap can show. Every
other field comes from the base configuration. A combination the config refuses
(a sell level at or under the buy level, a take fraction of 0) is recorded as
an *invalid* cell rather than replayed or silently dropped, so the heatmap
shows the hole where it is.
"""
from __future__ import annotations

import hashlib
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
from itertools import combinations, product
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
    BREACH_OFFERED,
    BREACH_POLICIES,
    LEVEL_SOURCES,
    LEVEL_SOURCE_LABELS,
)
from agent_stonks import minute_momentum
from agent_stonks.market_hours import MARKET_TZ

TUNING_DIR = Path(__file__).resolve().parent.parent / "data" / "simlab" / "tuning"
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

RUNNING = "running"
FINISHED = "finished"
FAILED = "failed"
STOPPED_ERROR = "stopped by user"

# The two roles a job's datasets had before a job could hold any number of
# them. Only `_upgrade` reads them now.
TUNE = "tune"
TEST = "test"
BASELINE = "baseline"

#: What the pick is chosen on: the total profit over every dataset of a job.
PICK_METRIC = "profit"

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
        # "unit" rather than "ADR": which yardstick a k is counted in is the
        # run's own `level_unit`, and a grid swept under one is not comparable
        # with a grid swept under the other. The setup form names it; a column
        # header has no room to and must not name the wrong one.
        Tunable("buy_k", "Buy distance (× unit below H)", 0.05, 3.0, 0.05, (0.30, 0.90, 0.10)),
        Tunable("sell_k", "Sell distance (× unit below H)", 0.0, 3.0, 0.05, (0.05, 0.45, 0.10)),
        Tunable(
            "stop_gain_fraction", "Stop loss (× the predicted gain)", 0.0, 3.0, 0.05,
            (0.0, 1.0, 0.25),
        ),
        Tunable(
            "momentum_confirmation_bars", "Momentum confirmation period (bars)", 0, 60, 1,
            (2, 15, 1), "%d", integer=True,
        ),
        # The take and the entry gate as they were 2026-09-23 to -24. Still
        # tunables so a stored sweep over one reads back as the grid it was;
        # not offered for a new one (`LEGACY_AXES`), since neither can be set
        # beside the confirmation.
        Tunable(
            "negative_momentum_bars", "Negative momentum look-back, legacy (bars)", 0, 120, 1,
            (5, 30, 5), "%d", integer=True,
        ),
        Tunable(
            "negative_for_bars", "Negative for, legacy (bars in a row)", 1, 60, 1,
            (1, 10, 1), "%d", integer=True,
        ),
        # The take as it used to be written (`AppleTraderConfig.momentum_fade_bars`
        # and, before that, `momentum_drop`). Still tunables so a stored sweep
        # over either is still read back as the grid it was (`derived_jobs`);
        # not offered for a new one (`LEGACY_AXES`), since neither can be set
        # beside the negative-momentum look-back.
        Tunable(
            "momentum_fade_bars", "Momentum fade look-back, legacy (bars)", 0, 120, 1,
            (5, 30, 5), "%d", integer=True,
        ),
        Tunable(
            "momentum_drop", "Momentum fade, legacy (σ off its peak)", 0.0, 5.0, 0.1,
            (0.0, 2.0, 0.5), "%.1f",
        ),
        Tunable("take_fraction", "Share taken on negative momentum", 0.05, 1.0, 0.05, (0.30, 1.0, 0.10)),
        Tunable(
            "take_min_gain_fraction", "Take only after gaining (× predicted gain)",
            0.0, 1.0, 0.05, (0.0, 0.50, 0.10),
        ),
        Tunable(
            "hold_min_gain_k", "Keep a runner if target ≥ (× unit)", 0.0, 3.0, 0.05,
            (0.0, 0.60, 0.10),
        ),
        Tunable(
            "min_win_k", "Stand down under (× unit a share)", 0.0, 3.0, 0.05,
            (0.0, 0.40, 0.10),
        ),
        Tunable(
            "max_fall_k", "No buy into a fall over, legacy (× unit)", 0.0, 3.0, 0.05,
            (0.0, 0.50, 0.10),
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

    A grid axis like any other, in the Tuning tab as well as the Simulate
    tab's per-setup sweep: a heatmap with a row per rule reads exactly as one
    with a row per level, and a sweep of the levels under each forecast policy
    is the comparison the policies have never had.

    What a set of rules does not have is a *neighbourhood*. Two adjacent buy
    distances are nearly the same strategy, which is what makes the plateau
    pick mean something; two forecast policies are not near each other in any
    sense, so `pick_best` reads a plateau along the numeric axes only.
    """

    name: str
    label: str
    options: "tuple[str, ...]"
    #: option -> what the form and the captions call it.
    labels: "dict[str, str]"
    #: The same field named for a chart axis or a job label, where `label` is a
    #: whole sentence and what it sits next to is two words.
    short: str = ""
    #: What a picker offers for a new sweep, when that is not all of `options`:
    #: an earlier rule stays a valid option so a stored job still reads back,
    #: but is not offered again.
    offered: "tuple[str, ...]" = ()

    @property
    def pickable(self) -> "tuple[str, ...]":
        """The options a picker lists for a new sweep."""
        return self.offered or self.options


CHOICES: "dict[str, Choice]" = {
    c.name: c
    for c in (
        Choice(
            "level_source", "Levels measured below",
            tuple(LEVEL_SOURCES), dict(LEVEL_SOURCE_LABELS), short="Level source",
        ),
        Choice(
            "breach_update", "If the session trades outside the forecast",
            tuple(BREACH_POLICIES), dict(BREACH_LABELS), short="Forecast breach",
            offered=tuple(BREACH_OFFERED),
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
    "momentum_confirmation_bars",
    "take_fraction",
    "take_min_gain_fraction",
    "hold_min_gain_k",
    "min_win_k",
    "position_pct",
)


# Everything a grid axis may be, numbers first: the order the Tuning tab and
# the Simulate tab's sweep both list their pickers in.
AXES: "tuple[str, ...]" = tuple(TUNABLES) + tuple(CHOICES)

# Axes a stored grid may have but a new one is not offered: a field kept only
# so old records replay.
LEGACY_AXES: "tuple[str, ...]" = (
    "momentum_fade_bars", "momentum_drop",
    "negative_momentum_bars", "negative_for_bars", "max_fall_k",
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


def axis_title(name: str) -> str:
    """An axis's name for a chart, a table or a job label: no unit, no sentence.

    `Tunable.label` carries the unit a k is counted in, which a form needs and
    an axis title next to its own tick values does not; `Choice.label` is a
    sentence for the same reason. Both are trimmed to the field's name here so
    a heatmap's two titles are the two parameters and not a paragraph.
    """
    choice = CHOICES.get(name)
    if choice is not None:
        return choice.short or choice.label
    return sweep_label(name).split(" (")[0]


def axis_tick(name: str, value) -> str:
    """One axis value, short enough to be a chart tick.

    A rule comes out as its own key -- `extreme`, `brownian` -- which is what
    `config_signature` writes and what a caption elsewhere in the app calls it.
    The sentence in `Choice.labels` is the axis *title*'s job; repeated down
    the side of a heatmap it is most of the chart.
    """
    return str(value) if name in CHOICES else value_label(name, value)


# Wall-clock seconds one worker spends replaying one session, for the form's
# estimate. Measured on a stored AAPL week (a 4-session replay in ~0.6-0.8 s
# once the market index and the replay minute frame were in); each pool worker
# also pays a few seconds up front to import torch and load the bundle. A rough
# guide, not a promise.
SECONDS_PER_SESSION = 0.25


def estimated_seconds(
    spec: dict, prior: "dict | None" = None, datasets: "list[dict] | None" = None
) -> float:
    """Roughly how long sweeping `datasets` (default: all of the job's) takes.

    `prior` is `prior_cells`' answer, `{dataset name: {cell key: cell}}`: those
    replays are already on disk and the job skips them, so they cost nothing
    but still count towards its total.
    """
    cells = len(grid(spec["axes"]))
    prior = prior or {}
    datasets = spec["datasets"] if datasets is None else datasets
    sessions = sum(
        (cells + 1 - len(prior.get(d["name"]) or ())) * len(d["days"]) for d in datasets
    )
    workers = max(1, min(int(spec.get("workers") or 1), (cells + 1) * max(len(datasets), 1)))
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
      off by another (`take_fraction` means nothing with `negative_momentum_bars` at 0,
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
    round_trips, wins = _round_trips(fills, float(result.starting_cash))
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
        "round_trips": round_trips,
        "wins": wins,
        "worst_day": min(profits) if profits else 0.0,
        "best_day": max(profits) if profits else 0.0,
        "daily": daily,
        "no_forecast_days": sorted(no_forecast),
        "error": result.error,
    }


def _round_trips(
    fills: "list[dict]", starting_cash: float
) -> "tuple[int | None, int | None]":
    """(round trips, winning ones) over a replay's fills, in the order made.

    A round trip runs from the buy that opens a flat book to the fill that
    flattens it again, however many rungs were added or partial takes sold in
    between; it won if the cash it ended on beats the cash it started from,
    fees included. A position still open at the end of the replay is not a
    round trip -- the trader flattens before every close, so that is a replay
    cut short rather than a trade. Apple Trader holds one ticker, so the book
    is flat when that ticker's position is. (None, None) when a fill does not
    say what it left behind.
    """
    if any("cash_after" not in f or "position_after" not in f for f in fills):
        return None, None
    trips = wins = 0
    cash = starting_cash
    opened_with = None
    for fill in fills:
        if opened_with is None and fill.get("action") == "buy":
            opened_with = cash
        cash = float(fill["cash_after"])
        if opened_with is not None and float(fill.get("position_after") or 0.0) <= 0.0:
            trips += 1
            if cash - opened_with > _FLAT_DAY_USD:
                wins += 1
            opened_with = None
    return trips, wins


def win_rate(cell: "dict | None") -> "float | None":
    """Winning round trips as a percentage, or None with none to count.

    None too for a cell scored before round trips were counted: its replay
    kept no fills, so there is nothing to count them from.
    """
    if not cell or not cell.get("round_trips"):
        return None
    return round(100.0 * cell["wins"] / cell["round_trips"], 1)


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
        # The week before each session, which `abs_mean_minute_momentum` (and
        # so the momentum confirmation's neutral band) is measured from.
        for prior in minute_momentum.prior_week_days(stamp):
            paths.append(sim_data.stored_bars_path(symbol, prior, feed))
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
    spec: dict, runs: "list[dict] | None" = None, datasets: "list[dict] | None" = None
) -> "dict[str, dict[str, dict]]":
    """Every cell of this job's grid the run store already answers.

    Returns `{dataset name: {overrides_key: cell}}` for `datasets` (default:
    every dataset of the job), with the baseline under `overrides_key({})`.

    Nothing here runs a replay, so this is cheap enough for the form to call on
    every rerun -- it parses the store (cached upstream) and scores the records
    that land on the grid.
    """
    datasets = (spec.get("datasets") or []) if datasets is None else datasets
    found: "dict[str, dict[str, dict]]" = {d["name"]: {} for d in datasets}
    if not spec.get("reuse_runs", True) or not datasets:
        return found
    if runs is None:
        from .results import list_runs

        runs = list_runs()
    index = index_runs(runs)
    if not index:
        return found
    cells = grid(spec["axes"])
    for dataset in datasets:
        for overrides in [*cells, {}]:
            cell = prior_cell(index, spec, dataset, overrides)
            if cell is not None:
                found[dataset["name"]][overrides_key(overrides)] = cell
    return found


def reused_count(record: dict) -> int:
    """How many of a job's replays came out of the run store rather than a
    fresh sweep -- countable against `progress["total"]`."""
    cells = [
        cell
        for name in dataset_names(record["spec"])
        for cell in [*(record["cells"].get(name) or ()), record["baseline"].get(name)]
    ]
    return sum(1 for c in cells if c and c.get("run_id"))


# --- grids the run store already holds --------------------------------------
#
# A tuning job is a grid of replays over one dataset. Nothing says that grid
# has to have been *asked for*: a Simulate sweep of the two levels, or a
# handful of setups queued one afternoon to compare a stop, leaves the same
# thing behind -- several runs of one configuration with one or two fields
# moved. Those are a tuning job that was run without ever being submitted, and
# they are worth reading as one rather than as rows in Results.
#
# So the Tuning tab also offers *derived* jobs: job records assembled out of
# the run store, with the cells the runs answer filled and the rest of the
# lattice left as holes. They are incomplete by construction -- nobody chose a
# base configuration, a second dataset or a pick rule, and the grid is only as
# dense as the runs happen to be -- and the tab offers to finish one, which
# submits a real job that reuses every filled cell and replays only the holes.


#: A job nobody submitted: assembled from stored runs rather than swept.
DERIVED = "from stored runs"

#: The fewest filled cells a derived grid is worth offering. Two runs that
#: happen to differ in one field are a pair, not a surface.
MIN_DERIVED_CELLS = 3

#: How many derived grids the tab offers, densest first. A store of a few
#: hundred runs holds a long tail of small ones and the picker has to stay
#: readable.
MAX_DERIVED = 20


def _axis_candidates() -> "list[tuple[str, ...]]":
    """Every set of one or two tunables a derived grid could be over.

    In `TUNABLES` order, so a two-axis grid's rows and columns come out in the
    same reading order the form lists them in.
    """
    names = list(TUNABLES)
    return [(n,) for n in names] + [tuple(pair) for pair in combinations(names, 2)]


def _axis_lattice(name: str, values: "list") -> "list":
    """The axis a set of swept values implies, holes included.

    Observed values alone can never leave a hole on a one-axis grid -- every
    value on the axis is there because some run had it -- so a sweep that is
    missing a step would read as complete. Where the values sit on a regular
    step, that step is the axis and the gaps in it are holes.

    Only where that is unambiguous, because guessing a finer axis than anyone
    swept would invent holes rather than find them. Every gap has to be a whole
    number of the smallest one, that step has to be a multiple of the field's
    own granularity, and the filled axis may not come out more than twice as
    long as what was actually run: 0.40, 0.45 and 0.75 are three settings
    somebody tried, not a 0.05 sweep with five holes in it.

    Values a run actually used are carried through as they were stored, not as
    the arithmetic reproduces them -- a cell is looked up by its overrides, and
    0.7 + 0.1 is not 0.8.
    """
    tunable = TUNABLES[name]
    if len(values) < 3:
        return list(values)
    gaps = [later - earlier for earlier, later in zip(values, values[1:])]
    step = min(gaps)
    if step <= 0:
        return list(values)
    def off_grid(value: float, unit: float) -> bool:
        return abs(value / unit - round(value / unit)) > 1e-6
    if any(off_grid(gap, step) for gap in gaps) or off_grid(step, tunable.step):
        return list(values)
    count = int(round((values[-1] - values[0]) / step)) + 1
    if count > 2 * len(values) or count > MAX_CELLS:
        return list(values)
    lattice = []
    for index in range(count):
        value = round(values[0] + index * step, 6)
        if tunable.integer:
            value = int(round(value))
        stored = next((v for v in values if abs(v - value) <= 1e-6), None)
        lattice.append(value if stored is None else stored)
    return lattice


def _derived_id(names: "tuple[str, ...]", bucket: tuple) -> str:
    """A stable id for one derived grid, so a selection survives a rerun.

    Hashed from what defines the grid -- the axes and the configuration,
    dataset and cash every cell shares -- and not from the runs in it, so a
    grid keeps its id when a new run fills one of its holes.
    """
    blob = json.dumps([list(names), list(bucket)], sort_keys=True, default=str)
    return "runs-" + hashlib.sha1(blob.encode()).hexdigest()[:10]


def _derived_job(names: "tuple[str, ...]", bucket: tuple, found: dict) -> dict:
    """One derived grid as a job record the Tuning tab can render unchanged.

    `found` maps a cell's `overrides_key` to the (record, config, overrides) of
    the newest run answering it, newest cell first.

    The base configuration is the newest of those runs. A derived grid has no
    untuned base to compare against -- nobody nominated one -- and the most
    recent run is the honest stand-in: it is a configuration that was actually
    chosen, its cell is on the grid by construction, and the heatmap's outline
    then marks where the last thing anyone ran sits on the surface.
    """
    _, feed, days, cash = bucket
    newest, base_config, _ = next(iter(found.values()))
    # A dataset's name is what a job keys its cells by, so it cannot be empty
    # even for a run that was stored without one.
    name = newest.get("dataset") or f"{days[0]} → {days[-1]}"
    axes = [
        {
            "name": name,
            "values": _axis_lattice(
                name, sorted({o[name] for _, _, o in found.values()})
            ),
        }
        for name in names
    ]
    spec = {
        "base": asdict(base_config),
        "axes": axes,
        "datasets": [{
            "name": name,
            "days": list(days),
            "feed": feed,
            # What the replay needs, rather than whatever basket the runs were
            # handed: a rule agent trades one symbol and `validate` only asks
            # that the dataset carries it.
            "symbols": [base_config.ticker],
        }],
        "starting_cash": cash,
        # Nobody chose these. The least-assuming pair -- the highest cell, no
        # minimum share of days traded -- so the marker says "the best anyone
        # has run" and nothing more; the form is where a real pick is set up.
        "rule": PICK_MAX,
        "min_traded_share": 0.0,
        "reuse_runs": True,
        "workers": 1,
    }
    sessions = [date.fromisoformat(day) for day in days]
    cells = [
        {
            "overrides": dict(overrides),
            "run_id": record.get("run_id") or "",
            "run_dataset": record.get("dataset") or "",
            **score_run(record, sessions),
        }
        for record, _, overrides in found.values()
    ]
    order = [overrides_key(o) for o in grid(axes)]
    cells.sort(key=lambda cell: order.index(overrides_key(cell["overrides"])))
    baseline = next(
        (
            {**cell, "overrides": {}}
            for cell in cells
            if cell["run_id"] == (newest.get("run_id") or "")
        ),
        None,
    )
    record = {
        "job_id": _derived_id(names, bucket),
        "created_at": newest.get("created_at") or "",
        "finished_at": newest.get("created_at") or "",
        "status": DERIVED,
        "pid": None,
        "error": None,
        "spec": spec,
        # Against what a submitted job would have replayed, so "6 of 30" reads
        # as how much of the grid the store actually covers.
        "progress": {
            "done": len(cells) + 1, "total": len(order) + 1, "reused": len(cells) + 1,
        },
        "cells": {name: cells},
        "baseline": {name: baseline},
        "best": None,
    }
    return refresh_pick(record)


def derived_jobs(runs: "list[dict] | None" = None) -> "list[dict]":
    """Every tuning grid the run store already holds, densest first.

    A grid is a set of stored runs that agree on everything except one or two
    tunable fields, and on the sessions, the tape and the starting cash -- the
    same match `prior_cells` makes, read the other way round: instead of asking
    which runs answer a grid somebody described, this asks which grids the runs
    describe by themselves. Only runs that could stand in for a cell at all are
    considered (`run_replay_key`), so a stale or unfinished run never invents a
    grid.

    Every axis must take at least two values, or it is not an axis; a grid of
    fewer than `MIN_DERIVED_CELLS` runs is a coincidence rather than a sweep.
    One run belongs to as many grids as it fits, but a grid whose runs are all
    inside a denser one is dropped -- a two-axis sweep would otherwise also be
    offered as each of its rows and each of its columns.
    """
    if runs is None:
        from .results import list_runs

        runs = list_runs()
    usable = []
    for record in runs:  # newest first, and every dict below keeps that order
        key = run_replay_key(record)
        if key is None:
            continue
        config = make_config(record["config_summary"]["rule_config"], {})
        usable.append((record, config, asdict(config), key))
    if not usable:
        return []

    candidates = []
    for names in _axis_candidates():
        buckets: "dict[tuple, dict]" = {}
        for record, config, fields, (_, feed, days, cash) in usable:
            shared = json.dumps(
                {k: v for k, v in fields.items() if k not in names},
                sort_keys=True, default=str,
            )
            overrides = {name: fields[name] for name in names}
            buckets.setdefault((shared, feed, days, cash), {}).setdefault(
                overrides_key(overrides), (record, config, overrides)
            )
        for bucket, found in buckets.items():
            if len(found) < MIN_DERIVED_CELLS:
                continue
            if any(
                len({o[name] for _, _, o in found.values()}) < 2 for name in names
            ):
                continue  # a field that never moved is not an axis
            candidates.append((names, bucket, found))

    jobs, covered = [], []
    for names, bucket, found in sorted(candidates, key=lambda c: -len(c[2])):
        ids = {record.get("run_id") for record, _, _ in found.values()}
        if any(ids <= seen for seen in covered):
            continue
        covered.append(ids)
        jobs.append(_derived_job(names, bucket, found))
        if len(jobs) >= MAX_DERIVED:
            break
    return jobs


def is_derived(job: dict) -> bool:
    return job.get("status") == DERIVED


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
    """The best of `cells`, with the `pick_score` it won on.

    A job hands it the grid summed over all of its datasets (`summed_cells`).

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

    # A neighbour only means something where a step along the axis is a small
    # change in the same setting. It is on a numeric axis -- 0.35 is nearly
    # 0.30 -- and it is not on a set of rules, where the order is only the one
    # the form lists them in. So a choice axis offers no offsets and the
    # plateau is read within each rule rather than smeared across them.
    offsets = list(product(*(
        (-1, 0, 1) if axis["name"] in TUNABLES else (0,) for axis in axes
    )))

    def plateau(cell: dict) -> float:
        here = _position(cell, axes)
        values = [
            float(other[metric])
            for offset in offsets
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
    """Sessions two datasets share -- summed, those sessions count twice."""
    return sorted(set(first) & set(second))


# --- the sum over datasets --------------------------------------------------


def dataset_names(spec: dict) -> "list[str]":
    """A job's datasets by name, in the order they were added."""
    return [d["name"] for d in spec.get("datasets") or ()]


def _cells_by_key(record: dict, name: str) -> "dict[str, dict]":
    return {overrides_key(c["overrides"]): c for c in record["cells"].get(name) or ()}


def combine(overrides: dict, parts: "list[dict | None]") -> "dict | None":
    """One grid cell summed over datasets, or None while a dataset has no answer.

    Each dataset is its own replay starting from the same cash, so profits add,
    and so do returns: the summed return is the total profit as a share of the
    starting cash. Counts add too; the worst and best day are the extremes over
    every session.

    A cell the configuration refuses is refused on every dataset, so one
    `invalid` part makes the sum invalid. A replay that errored makes the sum
    unscored rather than a number missing one dataset's share. A dataset that
    has not been swept yet leaves the sum a hole -- a partial total would read
    as a bad cell rather than an unfinished one.
    """
    if not parts:
        return None
    refused = next((p for p in parts if p and "invalid" in p), None)
    if refused is not None:
        return {"overrides": dict(overrides), "invalid": refused["invalid"]}
    if any(p is None for p in parts):
        return None
    errored = next((p for p in parts if p.get("error")), None)
    if errored is not None:
        return {"overrides": dict(overrides), "error": errored["error"]}
    total = lambda field: sum(p.get(field) or 0 for p in parts)  # noqa: E731
    # A dataset scored before round trips were counted leaves the sum without
    # them, rather than a rate over the other datasets' trades only.
    counted = all(p.get("round_trips") is not None for p in parts)
    return {
        "overrides": dict(overrides),
        "profit": round(total("profit"), 2),
        "return_pct": round(total("return_pct"), 4),
        "trades": total("trades"),
        "sells": total("sells"),
        "days": total("days"),
        "days_traded": total("days_traded"),
        "days_up": total("days_up"),
        "days_down": total("days_down"),
        "round_trips": total("round_trips") if counted else None,
        "wins": total("wins") if counted else None,
        "worst_day": min(p["worst_day"] for p in parts),
        "best_day": max(p["best_day"] for p in parts),
        "datasets": len(parts),
        "datasets_up": sum(1 for p in parts if p["profit"] > _FLAT_DAY_USD),
        "reused": sum(1 for p in parts if p.get("run_id")),
        "error": None,
    }


def summed_cells(record: dict) -> "list[dict]":
    """The grid summed over every dataset of the job, in grid order.

    Only cells every dataset has answered are summed (plus refused ones), so
    while a job runs the sum fills in as each combination completes everywhere.
    """
    names = dataset_names(record["spec"])
    by_name = [_cells_by_key(record, name) for name in names]
    out = []
    for overrides in grid(record["spec"]["axes"]):
        key = overrides_key(overrides)
        cell = combine(overrides, [cells.get(key) for cells in by_name])
        if cell is not None:
            out.append(cell)
    return out


def summed_baseline(record: dict) -> "dict | None":
    """The base configuration summed over every dataset of the job."""
    names = dataset_names(record["spec"])
    return combine({}, [(record.get("baseline") or {}).get(name) for name in names])


def pick_cell(record: dict, name: str) -> "dict | None":
    """The pick's own cell on one dataset: its share of the summed total."""
    best = record.get("best")
    if not best:
        return None
    return _cells_by_key(record, name).get(overrides_key(best["overrides"]))


def refresh_pick(record: dict) -> dict:
    """Re-read the pick off the summed grid; returns the record.

    The best total profit, under the job's pick rule and minimum share of days
    traded -- counted over every session of every dataset.
    """
    spec = record["spec"]
    record["best"] = pick_best(
        summed_cells(record), spec["axes"], PICK_METRIC,
        spec.get("rule", PICK_MAX), float(spec.get("min_traded_share") or 0.0),
    )
    return record


def missing_replays(record: dict) -> int:
    """How many replays the record still has no answer for, over all datasets."""
    keys = [overrides_key(c) for c in grid(record["spec"]["axes"])]
    missing = 0
    for name in dataset_names(record["spec"]):
        have = _cells_by_key(record, name)
        missing += sum(1 for key in keys if key not in have)
        missing += (record.get("baseline") or {}).get(name) is None
    return missing


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
        return _upgrade(json.loads(_path(job_id).read_text()))
    except (OSError, json.JSONDecodeError):
        return None


def list_jobs() -> "list[dict]":
    """Every job record, newest first, with dead workers marked failed."""
    if not TUNING_DIR.exists():
        return []
    jobs = []
    for path in sorted(TUNING_DIR.glob("*.json"), reverse=True):
        try:
            jobs.append(_upgrade(_reap(json.loads(path.read_text()))))
        except (OSError, json.JSONDecodeError):
            continue
    return jobs


def _answered(record: dict) -> int:
    """How many replays the record holds an answer for, over all datasets."""
    return sum(
        len(record["cells"].get(name) or ()) + (record["baseline"].get(name) is not None)
        for name in dataset_names(record["spec"])
    )


def _upgrade(record: dict) -> dict:
    """A job stored with a tune and a test dataset, read as a job over both.

    Before 2026-09-26 a job had exactly two roles: the grid was swept on the
    *tune* dataset, the pick chosen there, and either the whole grid or only
    the pick (plus the baseline) replayed on the *test* dataset. Both become
    ordinary datasets of the job, in that order. A test dataset whose grid was
    not swept keeps only the replays it had -- the pick and the baseline -- so
    the rest of its grid is holes that resuming the job fills.

    The pick is re-read off the sum, which is what a pick means now, so an old
    job's marker moves from the tuning week's best cell to the best total over
    both weeks. The stored metric goes with it: the sum is ranked on profit.

    In memory only: the file keeps its old shape until something rewrites it
    (adding or removing a dataset, resuming), and a worker still running under
    the old code keeps writing the shape it knows.
    """
    spec = record.get("spec") or {}
    if "datasets" in spec:
        return record
    datasets: "list[dict]" = []
    cells: "dict[str, list[dict]]" = {}
    baseline: "dict[str, dict | None]" = {}
    for role, key in ((TUNE, "tune_dataset"), (TEST, "test_dataset")):
        dataset = spec.get(key)
        if not dataset or dataset["name"] in cells:
            continue
        found = list((record.get("cells") or {}).get(role) or ())
        if role == TEST and not spec.get("sweep_test_grid"):
            found = [c for c in (record.get("best_test"),) if c]
        datasets.append(dataset)
        cells[dataset["name"]] = found
        baseline[dataset["name"]] = (record.get("baseline") or {}).get(role)
    upgraded = {
        k: v for k, v in record.items() if k not in ("best_test",)
    }
    upgraded["spec"] = {
        **{k: v for k, v in spec.items()
           if k not in ("tune_dataset", "test_dataset", "sweep_test_grid", "metric")},
        "datasets": datasets,
    }
    upgraded["cells"] = cells
    upgraded["baseline"] = baseline
    upgraded["progress"] = {
        **(record.get("progress") or {}),
        "done": _answered(upgraded),
        "total": total_replays(upgraded["spec"]),
    }
    upgraded["progress"]["reused"] = reused_count(upgraded)
    if upgraded.get("status") != RUNNING:
        refresh_pick(upgraded)
    return upgraded


def delete_job(job_id: str) -> None:
    for path in (_path(job_id), log_path(job_id)):
        if path.exists():
            path.unlink()


def total_replays(spec: dict) -> int:
    """How many replays a job runs: the grid plus the baseline, on every dataset."""
    return (len(grid(spec["axes"])) + 1) * len(spec.get("datasets") or ())


def validate(spec: dict) -> "str | None":
    """Why this job cannot run, or None."""
    axes = spec.get("axes") or []
    if not 1 <= len(axes) <= MAX_AXES:
        return f"Pick one or two parameters to tune (got {len(axes)})."
    if len({a["name"] for a in axes}) != len(axes):
        return "The same parameter is tuned twice."
    for axis in axes:
        if axis["name"] not in AXES:
            return f"{axis['name']} is not a tunable parameter."
        if not axis.get("values"):
            return f"{sweep_label(axis['name'])}: no values to try."
        choice = CHOICES.get(axis["name"])
        # A stored job outlives the rules it was written with, and a spec is
        # rewritten by hand often enough while debugging. An option that no
        # longer exists would replay as the config's default and be reported
        # under the name of a rule it never ran.
        if choice is not None:
            unknown = [v for v in axis["values"] if v not in choice.options]
            if unknown:
                return f"{choice.label}: {', '.join(map(str, unknown))} is not an option."
    cells = len(grid(axes))
    if cells > MAX_CELLS:
        return f"{cells} combinations is more than the {MAX_CELLS} one job may sweep."
    datasets = spec.get("datasets") or []
    if not datasets:
        return "Pick at least one dataset."
    names = [d["name"] for d in datasets]
    # Cells are kept per dataset name, so a name twice would be one set of
    # cells counted twice in the sum.
    if len(set(names)) != len(names):
        return "The same dataset is in the job twice."
    ticker = spec["base"]["ticker"]
    for dataset in datasets:
        if ticker not in (dataset.get("symbols") or []):
            return f"Dataset {dataset['name']} does not carry {ticker}."
    return None


def _seed(record: dict, runs: "list[dict] | None", datasets: "list[dict]") -> int:
    """Fill the record's holes on `datasets` from the run store; how many it filled.

    A cell or baseline the record already holds is never replaced -- it is the
    same replay either way, and the one on the record is the one on screen.
    """
    prior = prior_cells(record["spec"], runs, datasets)
    base_key = overrides_key({})
    filled = 0
    for dataset in datasets:
        name = dataset["name"]
        have = _cells_by_key(record, name)
        cells = record["cells"].setdefault(name, [])
        for key, cell in prior[name].items():
            if key == base_key:
                if record["baseline"].get(name) is None:
                    record["baseline"][name] = cell
                    filled += 1
            elif key not in have:
                cells.append(cell)
                filled += 1
    return filled


def _launch(record: dict) -> dict:
    """Start the detached worker for this record, and write it."""
    TUNING_DIR.mkdir(parents=True, exist_ok=True)
    log = open(log_path(record["job_id"]), "ab")
    proc = subprocess.Popen(
        [sys.executable, "-m", "simlab.tuning", record["job_id"]],
        cwd=_PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    record["pid"] = proc.pid
    _write(record)
    return record


def submit(spec: dict, launch: bool = True, runs: "list[dict] | None" = None) -> dict:
    """Store a job and start its worker. Raises ValueError for an invalid spec.

    The record is seeded with every cell the run store already answers
    (`prior_cells`), so the heatmaps are partly drawn the moment the job
    appears and the worker only replays the holes. Those cells count as done
    against the job's full total, which stays what the grid asks for -- a job
    that reused half its grid ran the whole grid, it just did not have to
    replay it.

    `runs` is the already-parsed store when the caller has one (the UI caches
    it); left out, the store is read here.
    """
    problem = validate(spec)
    if problem:
        raise ValueError(problem)
    job_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    names = dataset_names(spec)
    record = {
        "job_id": job_id,
        "created_at": _now_iso(),
        "finished_at": None,
        "status": RUNNING,
        "pid": None,
        "error": None,
        "spec": spec,
        "progress": {"done": 0, "total": total_replays(spec), "reused": 0},
        "cells": {name: [] for name in names},
        "baseline": {name: None for name in names},
        "best": None,
    }
    reused = _seed(record, runs, spec["datasets"])
    record["progress"].update(done=reused, reused=reused)
    _write(record)
    return _launch(record) if launch else record


def _editable(job_id: str) -> dict:
    """A stored job that may be changed now -- not running, not derived."""
    record = get_job(job_id)
    if record is None:
        raise ValueError(f"No tuning job {job_id}.")
    if record.get("status") == RUNNING:
        raise ValueError("The job is still running — wait for it to finish, or stop it.")
    return record


def _restart(record: dict, reused: int, launch: bool) -> dict:
    """Set the record running again over whatever it still lacks."""
    record["progress"] = {
        "done": _answered(record),
        "total": total_replays(record["spec"]),
        "reused": int((record.get("progress") or {}).get("reused") or 0) + reused,
    }
    record.update(status=RUNNING, error=None, finished_at=None, best=None, pid=None)
    _write(record)
    return _launch(record) if launch else record


def add_dataset(
    job_id: str, dataset: dict, runs: "list[dict] | None" = None, launch: bool = True
) -> dict:
    """Sweep the job's grid over one more dataset, then re-read the sum and the pick.

    The base configuration, the axes, the cash and the pick rule are the job's
    own, so the new heatmap is directly comparable with the ones already there.
    Cells the run store answers are filled in first, as at submit.
    """
    record = _editable(job_id)
    spec = {**record["spec"], "datasets": [*record["spec"]["datasets"], dataset]}
    problem = validate(spec)
    if problem:
        raise ValueError(problem)
    record["spec"] = spec
    record["cells"][dataset["name"]] = []
    record["baseline"][dataset["name"]] = None
    return _restart(record, _seed(record, runs, [dataset]), launch)


def resume(job_id: str, runs: "list[dict] | None" = None, launch: bool = True) -> dict:
    """Replay whatever a stopped or failed job, or an upgraded one, still lacks."""
    record = _editable(job_id)
    return _restart(record, _seed(record, runs, record["spec"]["datasets"]), launch)


def remove_dataset(job_id: str, name: str) -> dict:
    """Drop one dataset and its cells from a job, and re-read the sum and the pick.

    Nothing is replayed. The dropped cells are gone with it -- adding the
    dataset back sweeps it again, except where the run store answers a cell.
    A job keeps at least one dataset; deleting the job is the way to lose that.
    """
    record = _editable(job_id)
    names = dataset_names(record["spec"])
    if name not in names:
        raise ValueError(f"{name} is not a dataset of this job.")
    if len(names) == 1:
        raise ValueError("A job needs at least one dataset — delete the job instead.")
    record["spec"] = {
        **record["spec"],
        "datasets": [d for d in record["spec"]["datasets"] if d["name"] != name],
    }
    record["cells"].pop(name, None)
    record["baseline"].pop(name, None)
    record["progress"] = {
        "done": _answered(record),
        "total": total_replays(record["spec"]),
        "reused": reused_count(record),
    }
    # A job that stopped part-way through the dataset just dropped may have
    # nothing left to replay: it is then as finished as any other.
    if record.get("status") == FAILED and not missing_replays(record):
        record.update(status=FINISHED, error=None, finished_at=_now_iso())
    refresh_pick(record)
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
    half-swept grid is still a picture of the half that was swept, and
    `resume` replays only the rest.
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
        refresh_pick(record)
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
    """Evaluate (dataset name, kind, overrides) tasks, calling `on_done` as each lands.

    One worker runs them in this process, in order -- which is what the tests
    use. More fan out over a spawn-context pool: spawn rather than fork,
    because every child ends up importing torch through the day-range model and
    a forked OpenMP runtime is not something to rely on.
    """
    by_name = {d["name"]: d for d in spec["datasets"]}
    cash = spec["starting_cash"]
    if workers <= 1:
        for name, kind, overrides in tasks:
            on_done(name, kind, evaluate(by_name[name], spec["base"], overrides, cash))
        return
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        futures = {
            pool.submit(evaluate, by_name[name], spec["base"], overrides, cash): (name, kind)
            for name, kind, overrides in tasks
        }
        for future in as_completed(futures):
            name, kind = futures[future]
            on_done(name, kind, future.result())


def run_job(job_id: str, progress: "Callable[[str], None]" = print) -> dict:
    """Sweep every dataset, sum, pick -- writing the record as it goes.

    Only the replays the record does not already hold are run: `submit`,
    `add_dataset` and `resume` seed it with every cell the run store answers,
    and what is left is the holes -- on a new job the whole grid, on an added
    dataset that dataset's grid, on a resumed job whatever was not reached.
    """
    record = get_job(job_id)
    if record is None:
        raise RuntimeError(f"unknown tuning job {job_id}")
    spec = record["spec"]
    cells = grid(spec["axes"])
    workers = max(1, int(spec.get("workers") or 1))

    def on_done(name: str, kind: str, result: dict) -> None:
        if kind == BASELINE:
            record["baseline"][name] = result
        else:
            record["cells"][name].append(result)
        record["progress"]["done"] += 1
        _write(record)
        detail = result.get("invalid") or result.get("error") or f"${result.get('profit', 0):+,.2f}"
        progress(
            f"[{record['progress']['done']}/{record['progress']['total']}] {name} "
            f"{kind} {result['overrides'] or 'base'}: {detail}"
        )

    tasks = []
    for name in dataset_names(spec):
        record["cells"].setdefault(name, [])
        record["baseline"].setdefault(name, None)
        have = _cells_by_key(record, name)
        tasks += [(name, "cell", c) for c in cells if overrides_key(c) not in have]
        if record["baseline"][name] is None:
            tasks.append((name, BASELINE, {}))
    _run_tasks(tasks, spec, workers, on_done)

    order = [overrides_key(c) for c in cells]
    for name in dataset_names(spec):
        record["cells"][name].sort(key=lambda c: order.index(overrides_key(c["overrides"])))
    refresh_pick(record)
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
