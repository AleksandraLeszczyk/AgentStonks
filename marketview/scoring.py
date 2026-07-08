"""Daily agent-accuracy scoring.

Collection vs. scoring are deliberately split:

- **Collection** is cheap, deterministic, and always on. While an agent runs,
  a per-session :class:`Scorecard` (on ``AppState.scorecard``) accumulates
  grounding results per cycle, set_tactics validation rejections, tool
  errors / unreliable-quote warnings, and (under Automatic) the strategy
  activation windows. When the session ends the scorecard is flattened into
  one journal record (``data/scoring/journal.jsonl``).

- **Scoring** runs at most once per UTC calendar day. :func:`maybe_score_day`
  aggregates the current day's journal records (plus the still-running
  session, if any) into a single report file (``data/scoring/day-YYYY-MM-DD.json``)
  whose existence is the "this day already had a scoring session" gate. It
  refuses to score until the day has accumulated at least
  ``SCORING_MIN_TOTAL_RUNTIME_SEC`` (one hour) of agent runtime, so a couple
  of short experiments never produce a (statistically meaningless) report.
  When Langfuse is configured the report is also registered there as a
  ``daily-grounding`` score (see :func:`_register_langfuse_score`).

The grounding score is a deterministic faithfulness check, not an LLM judge:
every number the model emits in a finalizing tool call (submit_decision,
set_tactics, select_strategy, stand_down) must trace back to a number it was
actually shown earlier in that cycle's context (system/user prompts and tool
results -- i.e. the post-``_round_prices_for_llm`` strings). See
:func:`grounding_from_messages`.

Every hook called from the agent loops is wrapped to never raise: a scoring
bug must not take down a trading session.
"""
from __future__ import annotations

import functools
import json
import logging
import re
import threading
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Callable, TypeVar

from . import observability as obs
from .config import SCORING_MIN_TOTAL_RUNTIME_SEC

if TYPE_CHECKING:
    from .decisions import DecisionTracker
    from .state import AppState

logger = logging.getLogger(__name__)

# Overridable for tests. Lives next to data/avatars.
SCORING_DIR = Path(__file__).resolve().parent.parent / "data" / "scoring"
JOURNAL_NAME = "journal.jsonl"

# Tool calls that finalize (part of) a cycle: the numbers inside their
# arguments are the agent's outward-facing claims, so they are what the
# grounding check audits.
_DECISION_TOOLS = frozenset({"submit_decision", "set_tactics", "select_strategy", "stand_down"})

# Argument keys whose numbers are the agent's own free choices (share counts,
# self-declared quiet windows), not restatements of market data -- exempt from
# grounding.
_GROUNDING_EXEMPT_KEYS = frozenset({"quantity", "expected_quiet_minutes"})

# Small integers (counts, bar limits, "3 attempts") are overwhelmingly the
# agent's own bookkeeping, not market data; auditing them would only add noise.
_GROUNDING_MIN_ABS = 10.0

_NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")

F = TypeVar("F", bound=Callable[..., object])


def _never_raise(func: F) -> F:
    """Scoring is bookkeeping around a live trading loop -- log and swallow."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            logger.exception("scoring hook %s failed", func.__name__)
            return None

    return wrapper  # type: ignore[return-value]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _day_key(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).date().isoformat()


def _day_key_of_iso(ts: str) -> "str | None":
    try:
        return _day_key(datetime.fromisoformat(ts.replace("Z", "+00:00")))
    except ValueError:
        return None


# --------------------------------------------------------------------------
# Grounding: do the numbers the agent emits trace back to numbers it was shown?
# --------------------------------------------------------------------------

def _extract_numbers(text: str) -> set[float]:
    out: set[float] = set()
    for match in _NUMBER_RE.finditer(text):
        try:
            out.add(float(match.group().replace(",", "")))
        except ValueError:
            continue
    return out


def _collect_decision_numbers(args: object, out: list[float]) -> None:
    """Numeric leaves of a decision-call's arguments, plus numbers quoted
    inside its free-text fields (reasoning). Exempt keys are skipped."""
    if isinstance(args, dict):
        for key, value in args.items():
            if key in _GROUNDING_EXEMPT_KEYS:
                continue
            _collect_decision_numbers(value, out)
    elif isinstance(args, list):
        for value in args:
            _collect_decision_numbers(value, out)
    elif isinstance(args, bool):
        pass
    elif isinstance(args, (int, float)):
        out.append(float(args))
    elif isinstance(args, str):
        out.extend(_extract_numbers(args))


def _is_grounded(n: float, seen: set[float]) -> bool:
    # Absolute slack of 0.51 above $100 mirrors _round_prices_for_llm (the
    # model only ever saw whole dollars there); the 1%-of-source slack accepts
    # values legitimately derived from a shown level (entry +/- ATR, midpoints).
    slack = 0.51 if abs(n) >= 100 else 0.01
    return any(abs(n - t) <= max(slack, 0.01 * abs(t)) for t in seen)


def grounding_from_messages(messages: list[dict]) -> "dict | None":
    """Score one cycle's numeric faithfulness from its full LLM transcript.

    Walks `messages` in order, accumulating every number the model was shown
    (system/user/tool contents -- exactly the strings that entered its
    context) and, at each finalizing tool call, checking the numbers the model
    emitted against what it had seen *up to that point*. Returns
    ``{"total", "grounded", "score", "ungrounded"}``, or None when the cycle
    emitted no auditable numbers.
    """
    seen: set[float] = set()
    total = 0
    ungrounded: list[float] = []
    for msg in messages:
        role = msg.get("role")
        if role in ("system", "user", "tool"):
            content = msg.get("content")
            if isinstance(content, str):
                seen |= _extract_numbers(content)
            continue
        if role != "assistant":
            continue
        for call in msg.get("tool_calls") or []:
            function = call.get("function") or {}
            if function.get("name") not in _DECISION_TOOLS:
                continue
            try:
                args = json.loads(function.get("arguments") or "{}")
            except (TypeError, json.JSONDecodeError):
                continue
            numbers: list[float] = []
            _collect_decision_numbers(args, numbers)
            for n in numbers:
                if abs(n) < _GROUNDING_MIN_ABS:
                    continue
                total += 1
                if not _is_grounded(n, seen):
                    ungrounded.append(n)
    if total == 0:
        return None
    grounded = total - len(ungrounded)
    return {
        "total": total,
        "grounded": grounded,
        "score": grounded / total,
        "ungrounded": sorted({round(n, 4) for n in ungrounded})[:20],
    }


# --------------------------------------------------------------------------
# Per-session collection
# --------------------------------------------------------------------------

class Scorecard:
    """Accumulates one agent session's scoring inputs. Created by
    :func:`begin_session`, flattened into a journal record by
    :func:`end_session`."""

    def __init__(self, mode: str, symbols: list[str], start_value: "float | None") -> None:
        self.lock = threading.Lock()
        self.mode = mode
        self.symbols = list(symbols)
        self.start_value = start_value
        self.started_at = _utcnow().isoformat()
        self.cycles_run = 0
        self.grounding: list[dict] = []
        self.tactics_attempts = 0
        self.tactics_rejections = 0
        self.tool_calls = 0
        self.tool_errors = 0
        self.quote_calls = 0
        self.quote_warnings = 0
        # Closed Automatic activation windows: {strategy, regime, started_at, ended_at}.
        self.activations: list[dict] = []
        self._open_activation: "dict | None" = None

    def runtime_sec(self, now: "datetime | None" = None) -> float:
        started = datetime.fromisoformat(self.started_at)
        return max(0.0, ((now or _utcnow()) - started).total_seconds())

    def close_activation(self, ended_at: "str | None" = None) -> None:
        with self.lock:
            if self._open_activation is not None:
                self._open_activation["ended_at"] = ended_at or _utcnow().isoformat()
                self.activations.append(self._open_activation)
                self._open_activation = None


@_never_raise
def begin_session(state: "AppState", mode: str, symbols: list[str]) -> None:
    """Attach a fresh scorecard for a launching agent session. `mode` is the
    personality key, or "automatic" for the orchestrator."""
    start_value = state.mark_to_market()
    if start_value is None:
        start_value = state.starting_budget
    state.scorecard = Scorecard(mode, symbols, start_value)


@_never_raise
def record_cycle_grounding(state: "AppState", messages: list[dict], personality: str) -> None:
    """Score one finished cycle's transcript and add it to the session card."""
    card: "Scorecard | None" = state.scorecard
    if card is None:
        return
    result = grounding_from_messages(messages)
    with card.lock:
        card.cycles_run += 1
        if result is not None:
            card.grounding.append(
                {"ts": _utcnow().isoformat(), "personality": personality, **result}
            )


@_never_raise
def record_tactics_call(state: "AppState", ok: bool) -> None:
    card: "Scorecard | None" = state.scorecard
    if card is None:
        return
    with card.lock:
        card.tactics_attempts += 1
        if not ok:
            card.tactics_rejections += 1


@_never_raise
def record_tool_call(state: "AppState", name: str, result: dict) -> None:
    """Count every dispatched tool call; flag errors and unreliable quotes
    (stale / placeholder-wide, per the warning get_quote attaches)."""
    card: "Scorecard | None" = state.scorecard
    if card is None:
        return
    is_error = isinstance(result, dict) and "error" in result
    is_quote = name == "get_quote"
    has_warning = is_quote and isinstance(result, dict) and bool(result.get("warning"))
    with card.lock:
        card.tool_calls += 1
        if is_error:
            card.tool_errors += 1
        if is_quote:
            card.quote_calls += 1
            if has_warning:
                card.quote_warnings += 1


@_never_raise
def record_activation_start(state: "AppState", strategy: str, regime: "str | None") -> None:
    """Automatic activated `strategy`; a window opens until the strategy stands
    down (or the orchestrator stops)."""
    card: "Scorecard | None" = state.scorecard
    if card is None:
        return
    card.close_activation()
    with card.lock:
        card._open_activation = {
            "strategy": strategy,
            "regime": regime,
            "started_at": _utcnow().isoformat(),
        }


@_never_raise
def record_activation_end(state: "AppState") -> None:
    card: "Scorecard | None" = state.scorecard
    if card is None:
        return
    card.close_activation()


# --------------------------------------------------------------------------
# Session finalization -> journal record
# --------------------------------------------------------------------------

def _decisions_in_window(decisions: list[dict], start: str, end: str) -> list[dict]:
    return [d for d in decisions if start <= str(d.get("ts") or "") <= end]


def _decision_quality(
    decisions: list[dict], start_value: "float | None", end_value: "float | None"
) -> dict:
    counts = {"buy_sell_filled": 0, "buy_sell_rejected": 0, "tactics_armed": 0,
              "alerts": 0, "forced_sleeps": 0}
    for d in decisions:
        action, status = d.get("action"), d.get("status")
        if action in ("buy", "sell"):
            counts["buy_sell_filled" if status == "filled" else "buy_sell_rejected"] += 1
        elif action == "tactics":
            counts["tactics_armed"] += 1
        elif action == "alert":
            counts["alerts"] += 1
        elif action == "sleep":
            counts["forced_sleeps"] += 1
    total = len(decisions)
    active = counts["buy_sell_filled"] + counts["tactics_armed"]
    return_pct = None
    if start_value and end_value is not None:
        return_pct = (end_value / start_value - 1.0) * 100.0
    return {
        "total": total,
        **counts,
        "active": active,
        "active_rate": (active / total) if total else None,
        "start_value": start_value,
        "end_value": end_value,
        "return_pct": return_pct,
    }


def _activation_outcomes(activations: list[dict], decisions: list[dict]) -> list[dict]:
    """Attribute the session's decisions to each Automatic activation window
    and judge it: a strategy that produced no filled trade and armed no tactics
    -- only alarms, or nothing at all -- was not a good fit for that day."""
    out = []
    for window in activations:
        in_window = _decisions_in_window(
            decisions, window["started_at"], window.get("ended_at") or "9999"
        )
        active = sum(
            1
            for d in in_window
            if (d.get("action") in ("buy", "sell") and d.get("status") == "filled")
            or d.get("action") == "tactics"
        )
        alerts = sum(1 for d in in_window if d.get("action") == "alert")
        out.append(
            {
                **window,
                "decisions": len(in_window),
                "active_decisions": active,
                "alert_decisions": alerts,
                "effective": active > 0,
            }
        )
    return out


def _grounding_summary(cycle_results: list[dict]) -> "dict | None":
    if not cycle_results:
        return None
    scores = [c["score"] for c in cycle_results]
    ungrounded = sorted({n for c in cycle_results for n in c.get("ungrounded", [])})
    return {
        "scored_cycles": len(cycle_results),
        "mean_score": sum(scores) / len(scores),
        "min_score": min(scores),
        "ungrounded": ungrounded[:20],
    }


def _session_record(
    card: Scorecard,
    tracker: "DecisionTracker | None",
    state: "AppState | None",
    ended_at: "str | None" = None,
) -> dict:
    now_iso = ended_at or _utcnow().isoformat()
    decisions: list[dict] = []
    if tracker is not None:
        decisions = [asdict(d) for d in tracker.snapshot()["decisions"]]
    decisions = _decisions_in_window(decisions, card.started_at, now_iso)
    end_value = state.mark_to_market() if state is not None else None
    with card.lock:
        activations = list(card.activations)
        if card._open_activation is not None:  # still-running Automatic window
            activations.append({**card._open_activation, "ended_at": now_iso})
        record = {
            "started_at": card.started_at,
            "ended_at": now_iso,
            "runtime_sec": round(card.runtime_sec(datetime.fromisoformat(now_iso)), 1),
            "mode": card.mode,
            "symbols": card.symbols,
            "cycles": card.cycles_run,
            "grounding": _grounding_summary(card.grounding),
            "tactics": {"attempts": card.tactics_attempts, "rejections": card.tactics_rejections},
            "tools": {
                "calls": card.tool_calls,
                "errors": card.tool_errors,
                "quote_calls": card.quote_calls,
                "quote_warnings": card.quote_warnings,
            },
            "decisions": _decision_quality(decisions, card.start_value, end_value),
            "activations": _activation_outcomes(activations, decisions),
        }
    return record


@_never_raise
def end_session(state: "AppState", tracker: "DecisionTracker | None") -> None:
    """Flatten the finished session's scorecard into the journal, then give
    the daily scorer a chance to run."""
    card: "Scorecard | None" = state.scorecard
    if card is None:
        return
    state.scorecard = None
    record = _session_record(card, tracker, state)
    _append_journal(record)
    maybe_score_day(state, tracker)


def _append_journal(record: dict) -> None:
    SCORING_DIR.mkdir(parents=True, exist_ok=True)
    with (SCORING_DIR / JOURNAL_NAME).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def _read_journal() -> list[dict]:
    path = SCORING_DIR / JOURNAL_NAME
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


# --------------------------------------------------------------------------
# Daily scoring session
# --------------------------------------------------------------------------

_score_lock = threading.Lock()


def day_report_path(day: str) -> Path:
    return SCORING_DIR / f"day-{day}.json"


def _merge_strategy_stats(records: list[dict]) -> dict:
    strategies: dict[str, dict] = {}
    for record in records:
        for activation in record.get("activations") or []:
            stats = strategies.setdefault(
                activation["strategy"],
                {"activations": 0, "effective": 0, "alert_only": 0, "regimes": []},
            )
            stats["activations"] += 1
            if activation.get("effective"):
                stats["effective"] += 1
            else:
                stats["alert_only"] += 1
            regime = activation.get("regime")
            if regime and regime not in stats["regimes"]:
                stats["regimes"].append(regime)
    for stats in strategies.values():
        stats["effectiveness"] = stats["effective"] / stats["activations"]
    return strategies


def _build_day_report(day: str, records: list[dict]) -> dict:
    grounding_cycles = [
        {"score": g["mean_score"], "min": g["min_score"], "n": g["scored_cycles"],
         "ungrounded": g.get("ungrounded", [])}
        for g in (r.get("grounding") for r in records)
        if g
    ]
    scored_n = sum(g["n"] for g in grounding_cycles)
    grounding = None
    if scored_n:
        grounding = {
            "scored_cycles": scored_n,
            "mean_score": sum(g["score"] * g["n"] for g in grounding_cycles) / scored_n,
            "min_score": min(g["min"] for g in grounding_cycles),
            "ungrounded": sorted({n for g in grounding_cycles for n in g["ungrounded"]})[:20],
        }

    tactics_attempts = sum(r["tactics"]["attempts"] for r in records)
    tactics_rejections = sum(r["tactics"]["rejections"] for r in records)
    tool_calls = sum(r["tools"]["calls"] for r in records)
    tool_errors = sum(r["tools"]["errors"] for r in records)
    quote_calls = sum(r["tools"]["quote_calls"] for r in records)
    quote_warnings = sum(r["tools"]["quote_warnings"] for r in records)

    decision_totals = {
        key: sum(r["decisions"][key] for r in records)
        for key in ("total", "buy_sell_filled", "buy_sell_rejected", "tactics_armed",
                    "alerts", "forced_sleeps", "active")
    }
    returns = [r["decisions"]["return_pct"] for r in records
               if r["decisions"].get("return_pct") is not None]

    return {
        "day": day,
        "scored_at": _utcnow().isoformat(),
        "sessions": len(records),
        "total_runtime_sec": round(sum(r.get("runtime_sec") or 0.0 for r in records), 1),
        "grounding": grounding,
        "tactics_validation": {
            "attempts": tactics_attempts,
            "rejections": tactics_rejections,
            "rejection_rate": (tactics_rejections / tactics_attempts) if tactics_attempts else None,
        },
        "tools": {
            "calls": tool_calls,
            "errors": tool_errors,
            "error_rate": (tool_errors / tool_calls) if tool_calls else None,
            "quote_calls": quote_calls,
            "quote_warnings": quote_warnings,
            "quote_warning_rate": (quote_warnings / quote_calls) if quote_calls else None,
        },
        "decision_quality": {
            **decision_totals,
            "active_rate": (
                decision_totals["active"] / decision_totals["total"]
                if decision_totals["total"] else None
            ),
            "mean_session_return_pct": (sum(returns) / len(returns)) if returns else None,
        },
        "automatic": {"strategies": _merge_strategy_stats(records)},
        "session_index": [
            {"started_at": r["started_at"], "mode": r["mode"],
             "runtime_sec": r.get("runtime_sec"),
             "return_pct": r["decisions"].get("return_pct")}
            for r in records
        ],
    }


@_never_raise
def _register_langfuse_score(report: dict) -> None:
    """Register the daily report in Langfuse (no-op when unconfigured).

    The report rides along as the trace input; the score value is the day's
    mean grounding score. Days whose cycles emitted no auditable numbers have
    no grounding summary and register nothing.
    """
    grounding = report.get("grounding")
    if not grounding:
        return
    obs.record_score(
        trace_name=f"daily-scoring-{report['day']}",
        name="daily-grounding",
        value=grounding["mean_score"],
        comment=(
            f"{report['sessions']} session(s), "
            f"{report['total_runtime_sec'] / 3600:.1f} h agent runtime, "
            f"min cycle score {grounding['min_score']:.2f}"
        ),
        input=report,
    )


def _log_status(state: "AppState | None", text: str) -> None:
    if state is None:
        return
    entry = {"ts": _utcnow().isoformat(), "type": "status", "text": text}
    with state.lock:
        state.agent_log.append(entry)


@_never_raise
def maybe_score_day(
    state: "AppState | None" = None,
    tracker: "DecisionTracker | None" = None,
    now: "datetime | None" = None,
) -> "dict | None":
    """Run the daily scoring session if it is due.

    Skips (returning None) when this UTC day already has a report file, or
    when the day's journaled agent runtime -- including the still-running
    session, if `state` carries a live scorecard -- totals less than
    ``SCORING_MIN_TOTAL_RUNTIME_SEC``. Otherwise writes and returns the report.
    Cheap to call every cycle: the already-scored check is one stat() call.
    """
    now = now or _utcnow()
    day = _day_key(now)
    with _score_lock:
        path = day_report_path(day)
        if path.exists():
            return None

        records = [r for r in _read_journal() if _day_key_of_iso(r.get("started_at", "")) == day]
        card: "Scorecard | None" = state.scorecard if state is not None else None
        if card is not None and _day_key_of_iso(card.started_at) == day:
            records.append(_session_record(card, tracker, state, ended_at=now.isoformat()))

        total_runtime = sum(r.get("runtime_sec") or 0.0 for r in records)
        if total_runtime < SCORING_MIN_TOTAL_RUNTIME_SEC:
            return None

        report = _build_day_report(day, records)
        SCORING_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    _register_langfuse_score(report)

    grounding = report.get("grounding")
    grounding_note = (
        f"grounding {grounding['mean_score']:.2f}" if grounding else "no scored cycles"
    )
    _log_status(
        state,
        f"Daily scoring session {day}: {report['sessions']} session(s), {grounding_note}",
    )
    logger.info("daily scoring report written: %s", path)
    return report
