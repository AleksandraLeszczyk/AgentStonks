"""The ML summary: how each model's base configuration has done lately, and
how it did on event days, read off the tuning jobs.

A tuning job replays its base configuration on every dataset beside the grid
(the "Base profit" column of its results), so the store of jobs already holds,
per (model, instrument), a session-by-session P&L of one fixed configuration
over the weeks it was swept on. Nothing here replays anything: it reads those
baseline cells (`daily`, one profit per session) and lines them up.

One job per pair. A pair often has several jobs whose bases differ -- AMD's
sit out different days, say -- and adding their sessions together would mix
configurations, so a pair is read from the job that reaches the latest session,
the most recently started one on a tie (`base_series`).

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


@dataclass
class BaseSeries:
    """One pair's base configuration, session by session, from one job."""

    model_key: str
    ticker: str
    job_id: str
    created_at: str
    signature: str
    skip_events: "tuple[str, ...]"
    starting_cash: float
    daily: "dict[str, float]" = field(default_factory=dict)
    sat_out: "set[str]" = field(default_factory=set)
    no_forecast: "set[str]" = field(default_factory=set)
    # How many jobs the pair has, this one included.
    jobs: int = 1

    @property
    def last_day(self) -> str:
        return max(self.daily) if self.daily else ""

    def counted(self, day: str) -> bool:
        """Whether the base was in the market's hands that session: not sat out
        by its own rule, and with a forecast to trade on."""
        return day not in self.sat_out and day not in self.no_forecast


def _series(job: dict) -> "BaseSeries | None":
    spec = job["spec"]
    try:
        config = tuning.make_config(spec["base"], {})
    except (TypeError, ValueError):
        return None  # a removed model, or a field since narrowed
    out = BaseSeries(
        model_key=config.model_key, ticker=config.ticker, job_id=job["job_id"],
        created_at=str(job.get("created_at") or ""),
        signature=tuning.config_signature(config),
        skip_events=tuple(config.skip_events),
        starting_cash=float(spec.get("starting_cash") or 0.0),
    )
    for name in tuning.dataset_names(spec):
        cell = (job.get("baseline") or {}).get(name)
        if not tuning.is_scored(cell):
            continue
        out.daily.update({str(d): float(p) for d, p in (cell.get("daily") or {}).items()})
        out.sat_out |= {str(d) for d in cell.get("sat_out_days") or ()}
        out.no_forecast |= {str(d) for d in cell.get("no_forecast_days") or ()}
    return out if out.daily else None


def base_series(jobs: "list[dict]") -> "list[BaseSeries]":
    """Every pair's base configuration from the job that reaches its latest
    session (the most recently started on a tie), by instrument then model.
    Derived grids are not jobs anybody swept a base in, and are skipped."""
    chosen: "dict[tuple[str, str], BaseSeries]" = {}
    counts: "dict[tuple[str, str], int]" = {}
    for job in jobs:
        if tuning.is_derived(job):
            continue
        series = _series(job)
        if series is None:
            continue
        key = (series.model_key, series.ticker)
        counts[key] = counts.get(key, 0) + 1
        current = chosen.get(key)
        if current is None or (series.last_day, series.created_at) > (
            current.last_day, current.created_at
        ):
            chosen[key] = series
    for key, series in chosen.items():
        series.jobs = counts[key]
    return sorted(chosen.values(), key=lambda s: (s.ticker, s.model_key))


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
        "model_key": s.model_key, "ticker": s.ticker, "job_id": s.job_id,
        "signature": s.signature, "jobs": s.jobs,
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
    """`pair_metrics` rows by model: the mean per-session profit over its
    instruments -- each its own account, so weighted alike -- their summed
    profit, and the best and worst of them. Best model first."""
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
            "profit": round(sum(r["profit"] for r in group), 2),
            "in_profit": sum(1 for r in group if r["profit"] > tuning._FLAT_DAY_USD),
            "best": ranked[0]["ticker"],
            "worst": ranked[-1]["ticker"],
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
