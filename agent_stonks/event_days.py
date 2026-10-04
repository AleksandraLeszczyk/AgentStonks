"""The days the day-range traders sit out: earnings, CPI, jobs reports, shocks.

HighLow2_5m (`highlow2_model`) was fitted with its shock days left out -- the
day after earnings, jobs-report days, geopolitical shocks -- because their
highs and lows are set by news a 9:35 forecast cannot see, and the notebook's
trading week sits out the ones known before the open. Apple Trader does the same
on every model since 2026-10-04 (the user's call, with CPI days and market
shocks added to the notebook's three): `AppleTraderConfig.skip_events` names
the categories, and on a day with one the agent makes no forecast and rests no
order (`apple_trader.DayRangeTrader._sit_out`). Five categories:

`earnings`      the first session after the symbol's quarterly report: one
                between the previous session's close and this session's
                (`candidates.earnings_in_window`, the rule Orchestra's 09:34
                selection already uses).
`cpi`           a BLS Consumer Price Index release, 08:30 ET.
`nfp`           a BLS Employment Situation (jobs report) release, 08:30 ET.
`market_shock`  an unscheduled market-wide shock that is not geopolitical: the
                yen carry-trade unwind, DeepSeek, an election result.
`geo`           war and military strikes, tariffs and sanctions announced or
                taking effect.

Where each comes from
---------------------
* **The calendar**, `calendars/shock_days.csv`: HighLow2_5m's hand-curated
  calendar copied verbatim -- one row per session and category, each with its
  source -- plus BLS's schedule for the CPI and jobs reports after it ends
  (25 Sep 2026). Only rows `known_before_open` count: a shock that broke
  during the session (Iran's missiles on 1 Oct 2024) is one a trader could not
  have sat out, and a replay that did would be peeking. Scheduled releases
  have to be added as BLS publishes them; `check` says when a schedule has run
  out.
* **Yahoo's earnings dates** (`fetch_earnings_stamps`), for every symbol, read
  once a day; in SimLab, the copy it keeps (`session_context.earnings_dates`).
* **The pre-market briefing** for geo and market shocks from here on: the
  calendar's shock rows were marked after the fact, so nothing says a morning
  is one but the news. The briefing (`premarket.PremarketBriefing.shock`) is
  written when the stream starts, and each verdict -- shock or not -- is kept
  per (symbol, day) under `VERDICTS_DIR`, one file per day (`record_verdict`),
  so a SimLab run depends on the files of its own days only. That one verdict is
  what the trader, a restarted trader and the chart all read, so a regenerated
  briefing cannot flip a session the agent has already decided. A verdict made
  before the opening window closes may be replaced by a later one that is too
  (the 9:20 briefing knows more than the 8:00 one); after that it stands.
  A replay reads only verdicts made before the opening window closed, which is
  what a live run on that morning could have read.
"""
from __future__ import annotations

import csv
import json
import threading
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path

from . import candidates, historical
from .market_hours import MARKET_TZ

EARNINGS = "earnings"
CPI = "cpi"
NFP = "nfp"
MARKET_SHOCK = "market_shock"
GEO = "geo"
# In the order the form offers them and a signature writes them.
CATEGORIES = (EARNINGS, CPI, NFP, MARKET_SHOCK, GEO)
LABELS = {
    EARNINGS: "Day after earnings",
    CPI: "CPI release",
    NFP: "Jobs report (NFP)",
    MARKET_SHOCK: "Market shock",
    GEO: "Geopolitical shock",
}
# What `config_signature` writes for each.
TOKENS = {EARNINGS: "earn", CPI: "cpi", NFP: "nfp", MARKET_SHOCK: "mkt", GEO: "geo"}
# The two a briefing can flag, by the value of `PremarketBriefing.shock`.
SHOCKS = {"geo": GEO, "market": MARKET_SHOCK}

# The notebook's categories that are ours (it also keeps FOMC days and Apple's
# keynotes, which nothing here sits out).
_CALENDAR_CATEGORIES = {
    "earnings_next": EARNINGS, "cpi": CPI, "nfp": NFP, "market_shock": MARKET_SHOCK, "geo": GEO,
}
# The calendar's categories that only come from a published schedule, and how
# long after the last one a gap means the schedule has run out rather than
# that the month has no release yet.
_SCHEDULED = {CPI: 40, NFP: 40}

CALENDAR_PATH = Path(__file__).resolve().parent / "calendars" / "shock_days.csv"
VERDICTS_DIR = Path(__file__).resolve().parent.parent / "data" / "event_days" / "verdicts"

# When the trader decides: the opening window the forecast is built on closes
# at 09:35. A verdict made before it is one a replay may read.
VERDICT_CUTOFF = time(9, 35)


def as_date(day) -> date:
    """A session date from a `date`, a `datetime` or a pandas Timestamp (whose
    `isoformat` would carry a time and match no calendar row)."""
    return day.date() if isinstance(day, datetime) else day


def normalise(categories) -> "tuple[str, ...]":
    """`categories` in `CATEGORIES` order, each once; ValueError on one that is not."""
    given = [str(c) for c in (categories or ())]
    unknown = sorted(set(given) - set(CATEGORIES))
    if unknown:
        raise ValueError(
            f"skip_events {unknown!r}: not one of {', '.join(CATEGORIES)}"
        )
    return tuple(c for c in CATEGORIES if c in given)


@dataclass(frozen=True)
class Event:
    """One reason to sit a session out."""

    category: str
    what: str
    source: str

    @property
    def phrase(self) -> str:
        return f"{LABELS[self.category]} ({self.what})"


@dataclass
class DayCheck:
    """What `check` found: the events, anything it could not read, and whether
    the morning's briefing is still being written."""

    events: "list[Event]" = field(default_factory=list)
    notes: "list[str]" = field(default_factory=list)
    waiting: bool = False

    @property
    def categories(self) -> "list[str]":
        return [e.category for e in self.events]


# --- the calendar ------------------------------------------------------------

_calendar_lock = threading.Lock()
_calendar_cache: "dict[tuple, list[dict]]" = {}


def load_calendar(path: "Path | None" = None) -> "list[dict]":
    """The calendar's rows that count: our categories, known before the open.
    Re-read when the file changes, so a row added by hand applies at once."""
    path = path or CALENDAR_PATH
    try:
        stamp = (str(path), path.stat().st_mtime_ns)
    except OSError:
        return []
    with _calendar_lock:
        if stamp not in _calendar_cache:
            rows = []
            with path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    category = _CALENDAR_CATEGORIES.get(row.get("category", ""))
                    if category is None or str(row.get("known_before_open", "")).lower() != "true":
                        continue
                    rows.append({
                        "date": row["date"], "category": category,
                        "scope": str(row.get("scope") or "ALL").upper(),
                        "event": row.get("event", ""), "source": row.get("source", ""),
                    })
            _calendar_cache.clear()
            _calendar_cache[stamp] = rows
        return _calendar_cache[stamp]


def calendar_events(ticker: str, day: date, categories, rows: "list[dict] | None" = None) -> "list[Event]":
    """The calendar's events for `ticker` (or the whole market) on `day`."""
    iso = as_date(day).isoformat()
    symbol = ticker.upper()
    return [
        Event(row["category"], row["event"], "calendar")
        for row in (load_calendar() if rows is None else rows)
        if row["date"] == iso and row["category"] in categories and row["scope"] in (symbol, "ALL")
    ]


def schedule_notes(day: date, categories, rows: "list[dict] | None" = None) -> "list[str]":
    """A sentence per scheduled category whose dates in the calendar end well
    before `day`: that schedule needs BLS's next dates added."""
    rows = load_calendar() if rows is None else rows
    notes = []
    for category, grace in _SCHEDULED.items():
        if category not in categories:
            continue
        dates = [row["date"] for row in rows if row["category"] == category]
        last = max(dates) if dates else None
        if last is None or date.fromisoformat(last) + timedelta(days=grace) < day:
            notes.append(
                f"the calendar's {LABELS[category]} dates end {last or 'nowhere'}; add BLS's "
                f"next ones to {CALENDAR_PATH.name}, or those days will not be sat out"
            )
    return notes


# --- earnings ----------------------------------------------------------------

def fetch_earnings_stamps(ticker: str, day: date) -> "list | None":
    """Yahoo's report times for `ticker`, read once a day; None when they cannot
    be read. Patched in SimLab to the copy it keeps (`simlab.patches`)."""
    return candidates.LIVE_SOURCES.earnings(ticker.upper(), day)


def previous_session(ticker: str, day: date) -> "date | None":
    """The last session before `day`, from the daily bars (completed days only,
    live and in SimLab), so a holiday is not mistaken for one -- and a chart of
    a past session is measured from that session's eve, not from yesterday."""
    try:
        bars = [
            b for b in historical.fetch_daily_ohlc_bars(ticker)
            if str(b.get("t", ""))[:10] < day.isoformat()
        ]
        return candidates.daily_stats(bars)[2]
    except Exception:  # noqa: BLE001 -- a guess from the weekday is the fallback
        return None


def earnings_event(ticker: str, day: date, stamps, prev_day: "date | None" = None) -> "Event | None":
    """The report this session is the first after, as an Event, or None."""
    when = candidates.earnings_in_window(stamps or [], day, prev_day)
    if when is None:
        return None
    return Event(EARNINGS, f"report {when}", "Yahoo earnings dates")


# --- the briefing's verdict ---------------------------------------------------

_verdict_lock = threading.Lock()


def verdicts_path(day) -> Path:
    """The file holding every symbol's verdict for one ET session date."""
    return VERDICTS_DIR / f"{as_date(day).isoformat()}.json"


def _read_verdicts(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _made_before_cutoff(made_at: datetime) -> bool:
    local = made_at.astimezone(MARKET_TZ)
    return local.time() < VERDICT_CUTOFF


def record_verdict(symbol: str, shock: str, reason: str, made_at: datetime) -> bool:
    """Keep one briefing's shock verdict for `symbol` on the ET day it was made.
    True if it was kept. See the module docstring for which verdict wins."""
    shock = shock if shock in SHOCKS else "none"
    path = verdicts_path(made_at.astimezone(MARKET_TZ).date())
    symbol = symbol.upper()
    with _verdict_lock:
        verdicts = _read_verdicts(path)
        prior = verdicts.get(symbol)
        if prior is not None:
            prior_at = datetime.fromisoformat(prior["made_at"])
            if not (_made_before_cutoff(prior_at) and _made_before_cutoff(made_at)
                    and made_at > prior_at):
                return False
        verdicts[symbol] = {
            "shock": shock, "reason": str(reason or "").strip(),
            "made_at": made_at.astimezone(MARKET_TZ).isoformat(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(verdicts, indent=1, sort_keys=True), encoding="utf-8")
        tmp.replace(path)
    return True


def recorded_verdict(symbol: str, day: date, before_cutoff: bool = False) -> "dict | None":
    """The verdict kept for `symbol` on `day`, or None. `before_cutoff` keeps
    only one made before the opening window closed -- a replay's view."""
    with _verdict_lock:
        verdict = _read_verdicts(verdicts_path(day)).get(symbol.upper())
    if verdict is None:
        return None
    if before_cutoff and not _made_before_cutoff(datetime.fromisoformat(verdict["made_at"])):
        return None
    return verdict


def briefing_verdict(symbol: str, day: date) -> "dict | None":
    """The verdict the trader reads: whatever was kept for the day. Patched in
    SimLab to `recorded_verdict(..., before_cutoff=True)`."""
    return recorded_verdict(symbol, day)


def verdict_event(verdict: "dict | None", categories) -> "Event | None":
    category = SHOCKS.get(str((verdict or {}).get("shock")))
    if category is None or category not in categories:
        return None
    made = datetime.fromisoformat(verdict["made_at"]).astimezone(MARKET_TZ)
    return Event(
        category, verdict.get("reason") or "flagged by the pre-market briefing",
        f"pre-market briefing, {made:%H:%M} ET",
    )


# --- one session ---------------------------------------------------------------

def check(ticker: str, day: date, categories, briefing_pending: bool = False) -> DayCheck:
    """Every reason to sit `ticker`'s session on `day` out, among `categories`.

    `briefing_pending` is whether the morning's briefing for the symbol is still
    being written; with no verdict kept yet, the check then says to wait.
    """
    categories = normalise(categories)
    day = as_date(day)
    out = DayCheck()
    if not categories:
        return out
    rows = load_calendar()
    found = {e.category: e for e in calendar_events(ticker, day, categories, rows)}
    out.notes += schedule_notes(day, categories, rows)

    if EARNINGS in categories and EARNINGS not in found:
        stamps = fetch_earnings_stamps(ticker, day)
        if stamps is None:
            out.notes.append(f"{ticker}'s earnings dates could not be read")
        elif stamps:
            event = earnings_event(ticker, day, stamps, previous_session(ticker, day))
            if event is not None:
                found[EARNINGS] = event

    if set(SHOCKS.values()) & set(categories):
        verdict = briefing_verdict(ticker, day)
        event = verdict_event(verdict, categories)
        if event is not None and event.category not in found:
            found[event.category] = event
        elif verdict is None:
            if briefing_pending:
                out.waiting = True
            else:
                out.notes.append(
                    f"no pre-market briefing for {ticker} today, so only the calendar says "
                    "whether it is a market or geopolitical shock"
                )
    out.events = [found[c] for c in categories if c in found]
    return out


def sit_out_phrase(events: "list[Event]") -> str:
    """'CPI release (...) and Geopolitical shock (...)': what a log line quotes."""
    phrases = [e.phrase for e in events]
    return phrases[0] if len(phrases) == 1 else ", ".join(phrases[:-1]) + " and " + phrases[-1]
