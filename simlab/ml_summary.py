"""The ML summary: how each model's base configuration has done lately, and
how it did on event days, read off the tuning jobs.

Every cell of a tuning job -- its base configuration's on each dataset (the
"Base profit" column of its results) and every square of its grid -- is a
replay of one fixed configuration with a profit per session (`daily`), so the
store of jobs already holds, per (model, instrument), the weeks they were swept
on session by session. Nothing here replays anything: it reads those cells and
lines them up.

Which replays a pair is read from follows its base configuration -- the one
"Set as base configuration" stored (`agent_stonks.base_configs`), which is what
every new run of the pair starts from: every cell of every one of the pair's
jobs that replayed exactly that configuration, a job's own base or a square of
its grid, newest job first where two answer the same session. Set another base
and the next render reads the pair off the new one's replays. A pair with no
base stored, or whose base no job has replayed, falls back on the base of the
job reaching its latest session (the most recently started on a tie); jobs'
bases differ -- AMD's sit out different days, say -- so two are never added
together. A pair the app can trade with no base stored is not tuned -- no job
has scored a replay of it, or none of its jobs' picks was set as its base --
and is listed apart (`untuned_pairs`) rather than left to go missing.

Event days are classified after the fact (`events_by_day`), not as a trader
would have had to that morning: the calendar's every row, known before the open
or not, the day after a report from the earnings dates SimLab keeps, and a
geopolitical or market shock any symbol's pre-market briefing flagged -- read
as market-wide, since both categories are. A session the base sat out by its
own `skip_events` made nothing by rule and is left out of the comparison
rather than counted as a flat day.
"""
from __future__ import annotations

import csv
import statistics
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

from agent_stonks import candidates, event_days

from . import tuning

WEEK = "week"
MONTH = "month"
ALL = "all"
# Calendar days back from the latest session replayed, that session included.
WINDOW_DAYS = {WEEK: 7, MONTH: 30, ALL: None}

# What the calendar calls its categories, in `event_days`' names where it has
# one; the rest -- FOMC days, product events -- are kept under their own, since
# a day's impact is worth seeing whether or not a trader can sit it out.
_CALENDAR_CATEGORIES = {"earnings_next": event_days.EARNINGS}
EXTRA_LABELS = {"fomc": "FOMC decision", "product_event": "Product event"}
_MARKET_WIDE_SHOCKS = set(event_days.SHOCKS.values())


def event_label(category: str) -> str:
    return event_days.LABELS.get(category) or EXTRA_LABELS.get(category) or category


# Where a pair's replays were read from (`BaseSeries.source`).
FROM_BASE = "base"
FROM_JOB = "job"


@dataclass
class BaseSeries:
    """One pair's configuration, session by session: its stored base's
    replays (`FROM_BASE`), or one job's base (`FROM_JOB`)."""

    model_key: str
    ticker: str
    signature: str
    skip_events: "tuple[str, ...]"
    starting_cash: float
    source: str = FROM_JOB
    # The jobs read, newest first.
    job_ids: "list[str]" = field(default_factory=list)
    created_at: str = ""
    # When the stored base was set; with `FROM_JOB`, set only when the pair has
    # a base no job has replayed, so the fallback can say so.
    base_set_at: str = ""
    daily: "dict[str, float]" = field(default_factory=dict)
    sat_out: "set[str]" = field(default_factory=set)
    no_forecast: "set[str]" = field(default_factory=set)

    @property
    def last_day(self) -> str:
        return max(self.daily) if self.daily else ""

    def counted(self, day: str) -> bool:
        """Whether the base was in the market's hands that session: not sat out
        by its own rule, and with a forecast to trade on."""
        return day not in self.sat_out and day not in self.no_forecast

    def add(self, cell: dict) -> None:
        """One replay's sessions, over whatever an earlier one said of them."""
        self.daily.update({str(d): float(p) for d, p in (cell.get("daily") or {}).items()})
        self.sat_out |= {str(d) for d in cell.get("sat_out_days") or ()}
        self.no_forecast |= {str(d) for d in cell.get("no_forecast_days") or ()}


def _new_series(config, job: dict, source: str) -> BaseSeries:
    return BaseSeries(
        model_key=config.model_key, ticker=config.ticker,
        signature=tuning.config_signature(config), skip_events=tuple(config.skip_events),
        starting_cash=float(job["spec"].get("starting_cash") or 0.0), source=source,
        job_ids=[job["job_id"]], created_at=str(job.get("created_at") or ""),
    )


def _job_base(job: dict) -> "BaseSeries | None":
    """A job's own base, every dataset it was replayed on."""
    spec = job["spec"]
    try:
        config = tuning.make_config(spec["base"], {})
    except (TypeError, ValueError):
        return None  # a removed model, or a field since narrowed
    out = _new_series(config, job, FROM_JOB)
    for name in tuning.dataset_names(spec):
        cell = (job.get("baseline") or {}).get(name)
        if tuning.is_scored(cell):
            out.add(cell)
    return out if out.daily else None


def _replays(job: dict):
    """(configuration, cell) for every scored cell of a job -- its base's on
    each dataset and its grid's -- decoded once per combination."""
    spec = job["spec"]
    decoded: "dict[str, object]" = {}
    for name in tuning.dataset_names(spec):
        for cell in [(job.get("baseline") or {}).get(name), *((job.get("cells") or {}).get(name) or ())]:
            if not tuning.is_scored(cell):
                continue
            key = tuning.overrides_key(cell.get("overrides"))
            if key not in decoded:
                try:
                    decoded[key] = tuning.make_config(spec["base"], cell.get("overrides") or {})
                except (TypeError, ValueError):
                    decoded[key] = None
            if decoded[key] is not None:
                yield decoded[key], cell


def _stored_base(jobs: "list[dict]", config, set_at: str) -> "BaseSeries | None":
    """Every replay of `config` in `jobs`, the newest job's on a shared session."""
    out = None
    for job in sorted(jobs, key=lambda j: str(j.get("created_at") or "")):
        matched = False
        for replayed, cell in _replays(job):
            if replayed != config:
                continue
            if out is None:
                out = _new_series(config, job, FROM_BASE)
                out.job_ids = []
            out.add(cell)
            matched = True
        if matched:
            out.job_ids.insert(0, job["job_id"])
            out.created_at = str(job.get("created_at") or "")
            out.starting_cash = float(job["spec"].get("starting_cash") or 0.0)
    if out is not None:
        out.base_set_at = set_at
    return out


def stored_bases() -> "dict[tuple[str, str], tuple[object, str]]":
    """{(model, ticker): (configuration, set at)} for every stored base,
    decoded as a tuning job's base is (`tuning.make_config`) -- a field a record
    lacks reads as what its absence meant then -- so the two compare alike."""
    from agent_stonks import base_configs

    out = {}
    for entry in base_configs.load().values():
        try:
            config = tuning.make_config(entry["config"], {})
        except (KeyError, TypeError, ValueError):
            continue  # a removed model, or a field since narrowed
        out[(config.model_key, config.ticker)] = (config, str(entry.get("set_at") or ""))
    return out


def base_series(jobs: "list[dict]", bases: "dict | None" = None) -> "list[BaseSeries]":
    """Every pair the tuning jobs cover, by instrument then model: its stored
    base's replays (`bases`, `stored_bases()` by default), else the base of the
    job reaching its latest session (the most recently started on a tie).
    Derived grids are not jobs anybody swept, and are skipped."""
    bases = stored_bases() if bases is None else bases
    by_pair: "dict[tuple[str, str], list[dict]]" = {}
    fallback: "dict[tuple[str, str], BaseSeries]" = {}
    for job in jobs:
        if tuning.is_derived(job):
            continue
        series = _job_base(job)
        if series is None:
            continue
        key = (series.model_key, series.ticker)
        by_pair.setdefault(key, []).append(job)
        current = fallback.get(key)
        if current is None or (series.last_day, series.created_at) > (
            current.last_day, current.created_at
        ):
            fallback[key] = series
    out = []
    for key, pair_jobs in by_pair.items():
        stored = bases.get(key)
        series = _stored_base(pair_jobs, *stored) if stored else None
        if series is None:
            series = fallback[key]
            series.base_set_at = stored[1] if stored else ""
        out.append(series)
    return sorted(out, key=lambda s: (s.ticker, s.model_key))


def registry_pairs() -> "list[tuple[str, str]]":
    """(model, ticker) for every model on every instrument it was fitted on
    (`apple_models.MODELS`), in the registry's order: every pair the app can
    trade."""
    from agent_stonks import apple_models

    return [(key, ticker) for key, model in apple_models.MODELS.items() for ticker in model.tickers]


# Why a pair is untuned (`untuned_pairs`).
NO_JOB = "no job"
JOB_RUNNING = "running"
NOT_SCORED = "not scored"
NO_BASE = "no base"


def untuned_pairs(
    series: "list[BaseSeries]",
    jobs: "list[dict]",
    datasets: "list",
    pairs: "list[tuple[str, str]] | None" = None,
) -> "list[dict]":
    """Every pair the app can trade (`pairs`, `registry_pairs()` by default)
    with no tuned base configuration, by instrument, then the registry's order.
    Each with why: no job at all, one still running, only jobs whose base
    never scored (failed, stopped, every replay an error), or jobs that scored
    but none of whose picks was set as its base (`NO_BASE`: `series` reads it
    off a job's base, and a new run starts from `apple_trader.default_config`'s
    fallback); the pair's jobs
    newest first; and how many `datasets` (`simlab.data.Dataset`) carry the
    instrument, since a job needs one. A pair whose stored base no job has
    replayed has a base, and is not listed."""
    pairs = registry_pairs() if pairs is None else pairs
    replayed = {(s.model_key, s.ticker): s for s in series}
    tuned = {
        key for key, s in replayed.items() if s.source == FROM_BASE or s.base_set_at
    }
    by_pair: "dict[tuple[str, str], list[dict]]" = {}
    for job in sorted(jobs, key=lambda j: str(j.get("created_at") or ""), reverse=True):
        if tuning.is_derived(job):
            continue
        try:
            config = tuning.make_config(job["spec"]["base"], {})
        except (KeyError, TypeError, ValueError):
            continue  # a removed model, or a field since narrowed
        by_pair.setdefault((config.model_key, config.ticker), []).append(job)
    order = {pair: index for index, pair in enumerate(pairs)}
    out = []
    for model_key, ticker in sorted(
        (p for p in pairs if p not in tuned), key=lambda p: (p[1], order[p])
    ):
        pair_jobs = by_pair.get((model_key, ticker), [])
        if (model_key, ticker) in replayed:
            status = NO_BASE
        elif any(j.get("status") == tuning.RUNNING for j in pair_jobs):
            status = JOB_RUNNING
        else:
            status = NOT_SCORED if pair_jobs else NO_JOB
        out.append({
            "model_key": model_key, "ticker": ticker, "status": status,
            "job_ids": [j["job_id"] for j in pair_jobs],
            "datasets": sum(
                1 for d in datasets if ticker in {str(s).upper() for s in d.symbols or ()}
            ),
        })
    return out


def window(series: "list[BaseSeries]", name: str) -> "tuple[str, str] | None":
    """(first, last) ISO session dates of a window: `WINDOW_DAYS[name]`
    calendar days back from the latest session any pair has, or every session
    for `ALL`. None with nothing replayed."""
    days = [d for s in series for d in s.daily]
    if not days:
        return None
    last = max(days)
    span = WINDOW_DAYS[name]
    if span is None:
        return min(days), last
    return (date.fromisoformat(last) - timedelta(days=span - 1)).isoformat(), last


def _days(s: BaseSeries, span: "tuple[str, str]") -> "list[str]":
    return sorted(d for d in s.daily if span[0] <= d <= span[1])


def pair_metrics(s: BaseSeries, span: "tuple[str, str]") -> "dict | None":
    """One pair over a window, every session counted, traded or not -- the
    same per-session rate the Tuning tab's job names show. None with no session
    in it."""
    days = _days(s, span)
    if not days:
        return None
    profits = [s.daily[d] for d in days]
    total = sum(profits)
    per_session = total / len(days)
    flat = tuning._FLAT_DAY_USD
    return {
        "model_key": s.model_key, "ticker": s.ticker, "job_ids": s.job_ids,
        "signature": s.signature, "source": s.source, "base_set_at": s.base_set_at,
        "sessions": len(days),
        "profit": round(total, 2),
        "per_session": round(per_session, 2),
        "return_pct": round(100.0 * per_session / s.starting_cash, 4) if s.starting_cash else None,
        "days_up": sum(1 for p in profits if p > flat),
        "days_down": sum(1 for p in profits if p < -flat),
        "best_day": max(profits),
        "worst_day": min(profits),
        "sat_out": sum(1 for d in days if d in s.sat_out),
        "skip_events": s.skip_events,
    }


def model_metrics(rows: "list[dict]") -> "list[dict]":
    """`pair_metrics` rows by model: the mean over its instruments -- each its
    own account, so weighted alike -- of the profit a session and of the
    window's profit, how many made money, and the best and worst of them. Best
    model first."""
    by_model: "dict[str, list[dict]]" = {}
    for row in rows:
        by_model.setdefault(row["model_key"], []).append(row)
    out = []
    for model_key, group in by_model.items():
        ranked = sorted(group, key=lambda r: r["per_session"], reverse=True)
        out.append({
            "model_key": model_key,
            "instruments": len(group),
            "per_session": round(statistics.fmean(r["per_session"] for r in group), 2),
            "profit": round(statistics.fmean(r["profit"] for r in group), 2),
            "in_profit": sum(1 for r in group if r["profit"] > tuning._FLAT_DAY_USD),
            "best": ranked[0]["ticker"],
            "worst": ranked[-1]["ticker"],
        })
    return sorted(out, key=lambda r: r["per_session"], reverse=True)


def best_per_instrument(rows: "list[dict]") -> "list[dict]":
    """`pair_metrics` rows by instrument: the model whose base made the most a
    session over the window, the runner-up and by how much the winner leads it,
    and how many models the instrument was compared on. A session rate rather
    than the window's profit, since two models of one instrument may have been
    replayed on different sessions. Best instrument first."""
    by_ticker: "dict[str, list[dict]]" = {}
    for row in rows:
        by_ticker.setdefault(row["ticker"], []).append(row)
    out = []
    for ticker, group in by_ticker.items():
        best, *rest = sorted(group, key=lambda r: r["per_session"], reverse=True)
        second = rest[0] if rest else None
        out.append({
            "ticker": ticker,
            "model_key": best["model_key"],
            "per_session": best["per_session"],
            "profit": best["profit"],
            "sessions": best["sessions"],
            "days_up": best["days_up"],
            "days_down": best["days_down"],
            "models": len(group),
            "runner_up": second["model_key"] if second else None,
            "runner_up_per_session": second["per_session"] if second else None,
            "lead": round(best["per_session"] - second["per_session"], 2) if second else None,
        })
    return sorted(out, key=lambda r: r["per_session"], reverse=True)


def weekly(series: "list[BaseSeries]", span: "tuple[str, str]") -> "list[dict]":
    """Per pair and week (its Monday), the per-session profit in the window."""
    out = []
    for s in series:
        weeks: "dict[str, list[float]]" = {}
        for d in _days(s, span):
            day = date.fromisoformat(d)
            weeks.setdefault((day - timedelta(days=day.weekday())).isoformat(), []).append(
                s.daily[d]
            )
        out += [
            {"model_key": s.model_key, "ticker": s.ticker, "week": week,
             "per_session": round(statistics.fmean(values), 2), "sessions": len(values)}
            for week, values in sorted(weeks.items())
        ]
    return out


# --- event days ----------------------------------------------------------------

@dataclass(frozen=True)
class DayEvent:
    category: str
    what: str
    # The instrument it is about, or "" for the whole market.
    scope: str = ""

    @property
    def label(self) -> str:
        return event_label(self.category)


def calendar_rows(path: "Path | None" = None) -> "list[dict]":
    """Every row of the shock calendar, whatever its category and whether or
    not it was known before the open: what happened, not what could be sat out."""
    path = path or event_days.CALENDAR_PATH
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError:
        return []
    return [
        {
            "date": row["date"],
            "category": _CALENDAR_CATEGORIES.get(row.get("category", ""), row.get("category", "")),
            "scope": str(row.get("scope") or "ALL").upper(),
            "event": row.get("event", ""),
        }
        for row in rows
        if row.get("date") and row.get("category")
    ]


def _kept_earnings(ticker: str) -> "list | None":
    from .session_context import earnings_dates

    # The copy SimLab keeps, never a fetch: a summary tab must not wait on Yahoo.
    return earnings_dates(ticker, fetch=lambda _symbol: [])


def _verdict(ticker: str, day: str) -> "dict | None":
    return event_days.recorded_verdict(ticker, date.fromisoformat(day))


def events_by_day(
    tickers: "list[str]",
    days: "list[str]",
    rows: "list[dict] | None" = None,
    earnings=_kept_earnings,
    verdict=_verdict,
) -> "dict[tuple[str, str], list[DayEvent]]":
    """{(ticker, ISO day): events} for every session of every ticker that has
    one. `days` is every session replayed, in any order: the previous one
    bounds the earnings window (`candidates.earnings_in_window`)."""
    rows = calendar_rows() if rows is None else rows
    days = sorted(set(days))
    stamps = {t: earnings(t) or [] for t in tickers}
    shocks: "dict[str, DayEvent]" = {}
    for day in days:
        for t in tickers:
            found = event_days.SHOCKS.get(str((verdict(t, day) or {}).get("shock")))
            if found in _MARKET_WIDE_SHOCKS and day not in shocks:
                reason = (verdict(t, day) or {}).get("reason") or "flagged by the pre-market briefing"
                shocks[day] = DayEvent(found, reason)
    out: "dict[tuple[str, str], list[DayEvent]]" = {}
    for index, day in enumerate(days):
        prev = date.fromisoformat(days[index - 1]) if index else None
        for t in tickers:
            found: "dict[str, DayEvent]" = {}
            for row in rows:
                if row["date"] == day and row["scope"] in (t, "ALL"):
                    scope = "" if row["scope"] == "ALL" else t
                    found.setdefault(row["category"], DayEvent(row["category"], row["event"], scope))
            if event_days.EARNINGS not in found:
                when = candidates.earnings_in_window(stamps[t], date.fromisoformat(day), prev)
                if when:
                    found[event_days.EARNINGS] = DayEvent(event_days.EARNINGS, f"report {when}", t)
            if day in shocks and shocks[day].category not in found:
                found[shocks[day].category] = shocks[day]
            if found:
                out[(t, day)] = list(found.values())
    return out


def event_impact(
    series: "list[BaseSeries]",
    events: "dict[tuple[str, str], list[DayEvent]]",
    span: "tuple[str, str]",
) -> "tuple[list[dict], list[dict]]":
    """(event-day rows, per-pair normal days) over a window.

    A row is one pair on one event day: what the base made, what it made on an
    average normal day of the window (no event, not sat out, a forecast to
    trade on), the difference, and that difference in the normal days' spread
    (None with fewer than two). A day the base sat out or had no forecast is a
    row with `profit` None, so a reader sees it was not counted.
    """
    rows: "list[dict]" = []
    normals: "list[dict]" = []
    for s in series:
        days = _days(s, span)
        normal = [s.daily[d] for d in days if s.counted(d) and not events.get((s.ticker, d))]
        mean = statistics.fmean(normal) if normal else None
        spread = statistics.pstdev(normal) if len(normal) > 1 else None
        normals.append({
            "model_key": s.model_key, "ticker": s.ticker,
            "normal_days": len(normal), "normal_mean": mean, "normal_spread": spread,
        })
        for d in days:
            day_events = events.get((s.ticker, d))
            if not day_events:
                continue
            counted = s.counted(d) and mean is not None
            profit = s.daily[d] if counted else None
            delta = profit - mean if counted else None
            rows.append({
                "model_key": s.model_key, "ticker": s.ticker, "day": d,
                "events": day_events,
                "profit": profit,
                "normal_mean": mean,
                "delta": delta,
                "sigma": delta / spread if counted and spread else None,
                "sat_out": d in s.sat_out,
            })
    return rows, normals


def event_summaries(rows: "list[dict]") -> "list[dict]":
    """Per event day and what happened on it, newest first: the counted rows
    (pairs that traded it), how many did better and worse than a normal day,
    and the best and worst of them by the difference."""
    groups: "dict[tuple, list[dict]]" = {}
    for row in rows:
        key = (row["day"], tuple(sorted((e.category, e.scope) for e in row["events"])))
        groups.setdefault(key, []).append(row)
    out = []
    for (day, _), group in groups.items():
        counted = sorted(
            (r for r in group if r["delta"] is not None), key=lambda r: r["delta"], reverse=True
        )
        out.append({
            "day": day,
            "events": group[0]["events"],
            "rows": counted,
            "skipped": [r for r in group if r["delta"] is None],
            "better": sum(1 for r in counted if r["delta"] > tuning._FLAT_DAY_USD),
            "worse": sum(1 for r in counted if r["delta"] < -tuning._FLAT_DAY_USD),
            "best": counted[0] if counted else None,
            "worst": counted[-1] if counted else None,
        })
    return sorted(out, key=lambda g: g["day"], reverse=True)


def instrument_impact(rows: "list[dict]") -> "list[dict]":
    """Per instrument, over every event day and model it was counted on: the
    mean difference from a normal day, how many of those sessions beat one,
    and the event-day and normal-day means. Best first."""
    by_ticker: "dict[str, list[dict]]" = {}
    for row in rows:
        if row["delta"] is not None:
            by_ticker.setdefault(row["ticker"], []).append(row)
    out = [
        {
            "ticker": ticker,
            "sessions": len(group),
            "event_days": len({r["day"] for r in group}),
            "models": len({r["model_key"] for r in group}),
            "delta": statistics.fmean(r["delta"] for r in group),
            "event_mean": statistics.fmean(r["profit"] for r in group),
            "normal_mean": statistics.fmean(r["normal_mean"] for r in group),
            "better": sum(1 for r in group if r["delta"] > tuning._FLAT_DAY_USD),
        }
        for ticker, group in by_ticker.items()
    ]
    return sorted(out, key=lambda r: r["delta"], reverse=True)
