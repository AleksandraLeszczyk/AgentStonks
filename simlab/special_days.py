"""A tuning job's special sessions: earnings, CPI, jobs reports, shocks.

Which sessions, what the market expected of each, what came, and how far off
the expectation was -- the Tuning tab's special-days table, whose switches read
a job as if its configurations had sat a kind of session out
(`tuning.without_days`).

Which sessions are special is what a replay's days-off check
(`event_days.check`) finds for the job's instrument, whatever the job's own
days-off rule says: the calendar's rows known before the open, the first
session after an earnings report on SimLab's kept earnings dates
(`session_context.earnings_dates`), and the pre-market briefing's shock verdicts
made before 09:35. So a kind of session left out on the tab is exactly the set
of sessions a configuration sitting that kind out would not have traded.

What was expected and what came
-------------------------------
* **Earnings** -- Yahoo's earnings history: the EPS estimate, the reported EPS
  and its surprise, kept per symbol under `RESULTS_DIR`. Not in the
  earnings-dates file a replay reads (`session_context.EARNINGS_DIR`): a write
  there moves its mtime, and `tuning.run_is_stale` would retire every stored run
  with a days-off rule.
* **CPI and jobs reports** -- no free source keeps economists' consensus (Yahoo's
  economic calendar has no expectations and misses releases; Finnhub's is paid),
  so Claude looks each release up on the web (`lookup_release`): once, by hand
  from the tab, kept per release under `RELEASES_DIR` with the pages it read. A
  release is market-wide, so one lookup serves every job and instrument.
* **Shocks** -- nothing was expected; the row says what happened.
"""
from __future__ import annotations

import json
import math
import os
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from agent_stonks import candidates, event_days
from agent_stonks import observability as obs
from agent_stonks.market_hours import MARKET_TZ

from . import data as sim_data
from .session_context import CONTEXT_DIR, earnings_dates

RESULTS_DIR = CONTEXT_DIR / "earnings_results"
RELEASES_DIR = CONTEXT_DIR / "releases"

# What a CPI or jobs-report lookup asks for, in the order the table shows them:
# the first is the one its expected / actual / surprise columns carry.
RELEASES = {
    event_days.CPI: (
        "the US BLS Consumer Price Index release",
        (("CPI m/m", "%"), ("CPI y/y", "%"), ("Core CPI m/m", "%")),
    ),
    event_days.NFP: (
        "the US BLS Employment Situation (jobs report) release",
        (("Nonfarm payrolls", "k"), ("Unemployment rate", "%")),
    ),
}

#: The model that looks releases up, and how hard it may search.
LOOKUP_MODEL = "claude-opus-5-5"
_LOOKUP_SEARCHES = 5

_RELEASE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["period", "measures", "sources", "note"],
    "properties": {
        "period": {"type": "string"},
        "measures": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["name", "unit", "expected", "actual"],
            "properties": {
                "name": {"type": "string"},
                "unit": {"type": "string"},
                "expected": {"type": ["number", "null"]},
                "actual": {"type": ["number", "null"]},
            },
        }},
        "sources": {"type": "array", "items": {"type": "string"}},
        "note": {"type": "string"},
    },
}


# --- which sessions -----------------------------------------------------------


def _sessions(symbol: str, feed: str) -> "list[str]":
    """Every session the stored daily history of (symbol, feed) has, as ET dates."""
    return sorted({
        pd.Timestamp(bar["t"]).tz_convert(MARKET_TZ).date().isoformat()
        for bar in sim_data.load_daily_bars(symbol, feed)
    })


def _previous(sessions: "list[str]", day: str) -> "date | None":
    """The last session before `day`, or None (`earnings_in_window` then guesses
    from the weekday, as a replay with no daily history would)."""
    earlier = [s for s in sessions if s < day]
    return date.fromisoformat(earlier[-1]) if earlier else None


def special_days(spec: dict, earnings=None) -> "list[dict]":
    """Every (session, kind) of the job's datasets with an event, oldest first.

    One row per session and kind -- a CPI morning that is also the day after
    earnings is two rows -- with `datasets` naming every dataset of the job the
    session is in, and `report`, on an earnings row, the report it follows
    ("YYYY-MM-DD HH:MM ET"). `earnings` reads the symbol's report times
    (SimLab's kept copy, fetched the first time, by default).
    """
    ticker = str(spec["base"]["ticker"]).upper()
    calendar = event_days.load_calendar()
    stamps = (earnings or earnings_dates)(ticker) or []
    # Every dataset day is a session too, whether or not the stored daily
    # history had caught up with it when it was last refreshed.
    replayed = {d for dataset in spec["datasets"] for d in dataset["days"]}
    found: "dict[tuple[str, str], dict]" = {}
    for dataset in spec["datasets"]:
        sessions = sorted(set(_sessions(ticker, dataset["feed"])) | replayed)
        for iso in sorted(dataset["days"]):
            day = date.fromisoformat(iso)
            events = {
                e.category: e
                for e in event_days.calendar_events(ticker, day, event_days.CATEGORIES, calendar)
            }
            report = candidates.earnings_in_window(stamps, day, _previous(sessions, iso))
            if report and event_days.EARNINGS not in events:
                events[event_days.EARNINGS] = event_days.Event(
                    event_days.EARNINGS, f"report {report}", "Yahoo earnings dates"
                )
            shock = event_days.verdict_event(
                event_days.recorded_verdict(ticker, day, before_cutoff=True),
                event_days.CATEGORIES,
            )
            if shock is not None and shock.category not in events:
                events[shock.category] = shock
            for category, event in events.items():
                row = found.get((iso, category))
                if row is not None:
                    row["datasets"].append(dataset["name"])
                    continue
                found[(iso, category)] = {
                    "day": iso, "category": category, "what": event.what,
                    "source": event.source, "datasets": [dataset["name"]],
                    "report": report if category == event_days.EARNINGS else None,
                }
    return sorted(
        found.values(), key=lambda r: (r["day"], event_days.CATEGORIES.index(r["category"]))
    )


# --- earnings -----------------------------------------------------------------


def _number(value) -> "float | None":
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def _fetch_earnings_results(symbol: str) -> "list[dict]":
    from agent_stonks import historical

    frame = historical.fetch_earnings_dates(symbol, days=3650)
    if frame is None or frame.empty:
        return []
    return [
        {
            "time": pd.Timestamp(stamp).isoformat(),
            "estimate": _number(row.get("EPS Estimate")),
            "reported": _number(row.get("Reported EPS")),
            "surprise_pct": _number(row.get("Surprise(%)")),
        }
        for stamp, row in frame.iterrows()
    ]


def earnings_results(symbol: str, reports: "list[str]", fetch=None) -> "dict[str, dict]":
    """{report: {estimate, reported, surprise_pct}} for each of `reports` Yahoo
    has, matched on the report's ET date; `{"unlisted": True}` for one Yahoo no
    longer lists, as an estimated date that later moved does (SimLab keeps the
    earnings dates it first read, `session_context.earnings_dates`).

    Read from the copy kept under `RESULTS_DIR`, fetched again only when a
    report there has no reported EPS yet and the copy is older than the day
    after it -- a copy fetched before a report cannot have its result.
    """
    fetch = fetch or _fetch_earnings_results
    path = RESULTS_DIR / f"{symbol.upper()}.json"
    try:
        kept = json.loads(path.read_text())
    except (OSError, ValueError):
        kept = None

    def matched(rows: "list[dict]") -> "dict[str, dict]":
        by_day = {
            pd.Timestamp(r["time"]).tz_convert(MARKET_TZ).date().isoformat(): r for r in rows
        }
        return {r: by_day.get(r[:10]) or {"unlisted": True} for r in reports if rows}

    if kept is not None:
        found = matched(kept["rows"])
        fetched = datetime.fromisoformat(kept["fetched_at"]).astimezone(MARKET_TZ).date()
        stale = [
            r for r in reports
            if found.get(r, {}).get("reported") is None
            and fetched <= date.fromisoformat(r[:10])
        ]
        if not stale:
            return found
    try:
        rows = fetch(symbol.upper())
    except Exception:  # noqa: BLE001 -- a table without the numbers, not a broken tab
        rows = []
    if not rows:
        return matched(kept["rows"]) if kept is not None else {}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "symbol": symbol.upper(), "rows": rows,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }, indent=1))
    return matched(rows)


# --- CPI and jobs reports -------------------------------------------------------


def release_path(category: str, day: str) -> Path:
    return RELEASES_DIR / f"{day}_{category}.json"


def release_outcome(category: str, day: str) -> "dict | None":
    """The lookup kept for one release, or None before it was looked up."""
    try:
        return json.loads(release_path(category, day).read_text())
    except (OSError, ValueError):
        return None


def _release_question(category: str, day: str) -> str:
    what, measures = RELEASES[category]
    wanted = ", ".join(f"{name} ({'%' if unit == '%' else 'thousands'})" for name, unit in measures)
    return (
        f"Find the economists' consensus forecast and the actual figures of {what} "
        f"published on {day} at 08:30 ET. Measures, in this order: {wanted}. The consensus "
        "is the one reported before the release (Reuters, Bloomberg, Dow Jones/WSJ or a "
        "similar poll). Give each measure's unit as written here: % for a percentage, k "
        "for thousands of jobs. Use null where you cannot find a figure; never estimate "
        "one. List the pages you read under sources, and say in the note whose consensus "
        "you used and anything the sources disagreed on."
    )


def lookup_release(category: str, day: str, client=None) -> dict:
    """Look one CPI or jobs-report release up on the web and keep the answer.

    One Claude call with web search, about half a minute; raises when there is
    no key, the model declines, or the answer is not the JSON asked for.
    """
    import anthropic

    if client is None:
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        client = anthropic.Anthropic(api_key=key)
    question = _release_question(category, day)
    messages = [{"role": "user", "content": question}]
    with obs.anthropic_generation(
        name="release-lookup", model=LOOKUP_MODEL, input=question
    ) as generation:
        # A long search can pause the turn; it is resumed by sending it back.
        for _ in range(3):
            response = client.beta.messages.create(
                model=LOOKUP_MODEL, max_tokens=16000,
                betas=["server-side-fallback-2026-07-01"], fallbacks="default",
                output_config={
                    "effort": "medium",
                    "format": {"type": "json_schema", "schema": _RELEASE_SCHEMA},
                },
                tools=[{
                    "type": "web_search_20260209", "name": "web_search",
                    "max_uses": _LOOKUP_SEARCHES,
                }],
                messages=messages,
            )
            if response.stop_reason != "pause_turn":
                break
            messages = [*messages, {"role": "assistant", "content": response.content}]
        texts = [b.text for b in response.content if b.type == "text"]
        if response.stop_reason in ("refusal", "max_tokens", "pause_turn") or not texts:
            obs.record_anthropic_usage(generation, response, None)
            raise RuntimeError(f"no answer ({response.stop_reason})")
        answer = json.loads(texts[-1])
        obs.record_anthropic_usage(generation, response, answer)
    record = {
        "category": category, "day": day, **answer,
        "model": getattr(response, "model", LOOKUP_MODEL),
        "looked_up_at": datetime.now(timezone.utc).isoformat(),
    }
    path = release_path(category, day)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1))
    return record


# --- the table's cells ------------------------------------------------------------


def _figure(value: "float | None", unit: str) -> str:
    if value is None:
        return "—"
    if unit == "k":
        return f"{value:,.0f}k"
    return f"{value:g}{unit}"


def _miss(expected: "float | None", actual: "float | None", unit: str) -> str:
    if expected is None or actual is None:
        return "—"
    diff = actual - expected
    if unit == "%":
        return f"{diff:+.1f} pp"
    if unit == "k":
        return f"{diff:+,.0f}k"
    return f"{diff:+g}{unit}"


def outcome_cells(row: dict, earnings: "dict[str, dict] | None" = None) -> dict:
    """{measure, expected, actual, surprise, more} as the table writes them for
    one row.

    Earnings: the EPS estimate, the reported EPS and Yahoo's surprise. CPI and
    jobs reports: the first measure looked up, the others under `more`. A shock
    expected nothing; its row says what happened under the event already.
    """
    blank = {"measure": "", "expected": "", "actual": "", "surprise": "", "more": ""}
    category = row["category"]
    if category == event_days.EARNINGS:
        result = (earnings or {}).get(row.get("report") or "")
        if not result:
            return blank
        if result.get("unlisted"):
            return {**blank, "actual": "Yahoo lists no report then"}
        estimate, reported = result.get("estimate"), result.get("reported")
        surprise = result.get("surprise_pct")
        if surprise is None and estimate and reported is not None:
            surprise = 100.0 * (reported - estimate) / abs(estimate)
        return {
            "measure": "EPS",
            "expected": "" if estimate is None else f"{estimate:.2f}",
            "actual": "not reported yet" if reported is None else f"{reported:.2f}",
            "surprise": "" if surprise is None else f"{surprise:+.1f}%",
            "more": "",
        }
    if category not in RELEASES:
        return blank
    found = release_outcome(category, row["day"])
    if not found or not found.get("measures"):
        return blank
    first, *rest = found["measures"]
    unit = first.get("unit") or ""
    return {
        "measure": first["name"],
        "expected": _figure(first.get("expected"), unit),
        "actual": _figure(first.get("actual"), unit),
        "surprise": _miss(first.get("expected"), first.get("actual"), unit),
        "more": "; ".join(
            f"{m['name']} {_figure(m.get('actual'), m.get('unit') or '')} vs "
            f"{_figure(m.get('expected'), m.get('unit') or '')} expected "
            f"({_miss(m.get('expected'), m.get('actual'), m.get('unit') or '')})"
            for m in rest
        ),
    }


def missing_releases(rows: "list[dict]") -> "list[tuple[str, str]]":
    """(kind, day) of every CPI or jobs-report row not looked up yet."""
    return sorted({
        (r["category"], r["day"]) for r in rows
        if r["category"] in RELEASES and release_outcome(r["category"], r["day"]) is None
    })
