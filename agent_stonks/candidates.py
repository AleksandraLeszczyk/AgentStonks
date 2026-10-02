"""Which of Orchestra's pairs are worth racing today: the candidate selection.

Orchestra (`agent_stonks.orchestra`) races a fixed list of (ticker, model)
pairs. This module narrows that list once a session -- at **09:34 ET**, with
four of the opening minutes closed and one before the opening window the
forecasts are built on is complete -- so the race only runs on the pairs the
morning gives a reason to trade.

What is known at 09:34, and what the selection reads of it:

* **the pre-market briefing** (`agent_stonks.premarket`): its bias and its
  confidence. The strategy buys dips and only goes long, so a confidently
  bearish morning is the one most likely to walk through the buy level into
  the stop. Live it is the app's own briefing; a SimLab replay reads the one
  `simlab.session_context` cached for that symbol and day, written as of 09:25.
* **the earnings calendar**: a report since the last close, or due before
  this one, makes the day a different kind of day from the ones the levels
  were tuned on.
* **the daily bars before today**: the previous close and the 14-session
  average daily range (ADR), which put the opening gap and the first four
  minutes' move on a scale that compares across symbols.
* **the pair's own levels**: what a target exit pays, `buy_k - sell_k`, read in
  ADRs and then as a share of the price -- which is what a trade earns on the
  cash, the thing that matters when only one pair is held at a time.

The rules are `SelectionRules`, and every one of them is switchable: exclude a
pair on earnings, on a bearish briefing (confident only, or any), on a gap
wider than a limit, then rank what is left by the target as a share of the
price and keep the best `max_pairs`. `select_candidates` is a pure function of
the facts and the rules, so a replay of the same morning makes the same choice.

None of this has been shown to add anything yet. It is written so SimLab can
measure it: the same Orchestra with and without the selection, over the same
sessions.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Optional

import pandas as pd

from . import historical, momentum_regime
from .market_hours import MARKET_TZ

# "Just before the fifth minute": the selection is made once the bars of
# 09:30 to 09:33 have closed -- one bar before the five-minute opening window
# the forecasts are built on -- so only the pairs it keeps forecast at all.
SELECT_AFTER_BARS = 4

# How many completed sessions the average daily range is taken over, as the
# day-range model's own ADR is.
ADR_SESSIONS = 14

BEARISH_OFF = "off"
BEARISH_HIGH = "high"
BEARISH_ANY = "any"
BEARISH_RULES = (BEARISH_OFF, BEARISH_HIGH, BEARISH_ANY)
BEARISH_LABELS = {
    BEARISH_OFF: "Never",
    BEARISH_HIGH: "Bearish with high confidence",
    BEARISH_ANY: "Any bearish briefing",
}


@dataclass
class SelectionRules:
    """How Orchestra narrows its pairs at 09:34. See the module docstring."""

    # Keep at most this many pairs, best first; 0 keeps every eligible pair.
    max_pairs: int = 3
    # Leave out a pair whose symbol reports between the last close and this one.
    exclude_earnings: bool = True
    # Leave out a pair whose briefing is bearish: never, only with high
    # confidence, or at any confidence.
    exclude_bearish: str = BEARISH_HIGH
    # Leave out a pair whose symbol opened more than this many ADRs from the
    # previous close, either way. 0 is no limit.
    max_gap_adr: float = 0.0
    # Which cached briefings a SimLab replay reads: `simlab.session_context`
    # keeps one per (symbol, day, provider, model). A live run reads the app's
    # own briefing and ignores both.
    briefing_provider: str = ""
    briefing_model: str = ""

    def __post_init__(self) -> None:
        if float(self.max_pairs) != int(self.max_pairs) or int(self.max_pairs) < 0:
            raise ValueError(f"max_pairs {self.max_pairs!r} must be a whole number, 0 or more")
        self.max_pairs = int(self.max_pairs)
        if self.exclude_bearish not in BEARISH_RULES:
            raise ValueError(
                f"exclude_bearish {self.exclude_bearish!r} is not one of {', '.join(BEARISH_RULES)}"
            )
        if self.max_gap_adr < 0:
            raise ValueError(f"max_gap_adr {self.max_gap_adr!r} cannot be negative (0 is no limit)")


def rules_signature(rules: "SelectionRules | None") -> str:
    """The selection's part of an Orchestra signature: empty when there is no
    selection, so every record made before it existed keeps its identity."""
    if rules is None:
        return ""
    parts = [f"select={rules.max_pairs or 'all'}"]
    if rules.exclude_earnings:
        parts.append("earn")
    if rules.exclude_bearish != BEARISH_OFF:
        parts.append(f"bear={rules.exclude_bearish}")
    if rules.max_gap_adr:
        parts.append(f"gap<={rules.max_gap_adr:g}A")
    if rules.briefing_provider and rules.briefing_model:
        parts.append(f"brief={rules.briefing_provider}/{rules.briefing_model}")
    return "," + ",".join(parts)


# --- the facts ------------------------------------------------------------------


@dataclass
class PairFacts:
    """What is known about one pair at 09:34."""

    key: str
    label: str
    ticker: str
    # `buy_k - sell_k`: what a target exit pays, in level units (read as ADRs).
    target_k: float
    prev_close: Optional[float] = None
    adr: Optional[float] = None
    open: Optional[float] = None
    last: Optional[float] = None
    bias: Optional[str] = None
    confidence: Optional[str] = None
    # Why there is no bias to read, when there is none.
    briefing_note: str = ""
    # The report that falls in today's window, or None; `earnings_known` is
    # False when the calendar could not be read at all.
    earnings: Optional[str] = None
    earnings_known: bool = True

    @property
    def gap_adr(self) -> Optional[float]:
        if self.open is None or self.prev_close is None or not self.adr:
            return None
        return (self.open - self.prev_close) / self.adr

    @property
    def move_adr(self) -> Optional[float]:
        """The first minutes' move from the open, in ADRs."""
        if self.open is None or self.last is None or not self.adr:
            return None
        return (self.last - self.open) / self.adr

    @property
    def target_pct(self) -> Optional[float]:
        """What a target exit pays, as a percentage of the price."""
        price = self.last or self.open or self.prev_close
        if not price or not self.adr:
            return None
        return 100.0 * self.target_k * self.adr / price


@dataclass
class Candidate:
    facts: PairFacts
    selected: bool
    # Why: the rule that left it out, or where it ranked.
    reason: str
    rank: Optional[int] = None


def select_candidates(facts: "list[PairFacts]", rules: SelectionRules) -> "list[Candidate]":
    """Which pairs race today, in the order given. Pure: the same facts and
    rules always make the same choice."""
    excluded: "dict[str, str]" = {}
    for f in facts:
        if f.adr is None or f.target_pct is None:
            excluded[f.key] = "no daily history to measure the day against"
        elif rules.exclude_earnings and f.earnings:
            excluded[f.key] = f"earnings {f.earnings}"
        elif (
            rules.exclude_bearish != BEARISH_OFF
            and f.bias == "bearish"
            and (rules.exclude_bearish == BEARISH_ANY or f.confidence == "high")
        ):
            excluded[f.key] = f"bearish briefing, {f.confidence or 'unknown'} confidence"
        elif rules.max_gap_adr and f.gap_adr is not None and abs(f.gap_adr) > rules.max_gap_adr:
            excluded[f.key] = (
                f"opened {f.gap_adr:+.2f} ADR from the close, beyond ±{rules.max_gap_adr:g}"
            )
    eligible = [f for f in facts if f.key not in excluded]
    # Best target first; `sorted` is stable, so ties keep Orchestra's order.
    ranked = sorted(eligible, key=lambda f: -float(f.target_pct))
    keep = len(ranked) if not rules.max_pairs else rules.max_pairs
    rank_of = {f.key: i + 1 for i, f in enumerate(ranked)}
    out = []
    for f in facts:
        if f.key in excluded:
            out.append(Candidate(f, False, excluded[f.key]))
            continue
        rank = rank_of[f.key]
        if rank <= keep:
            reason = f"#{rank} of {len(ranked)} by target ({f.target_pct:.2f}% of the price)"
            out.append(Candidate(f, True, reason, rank))
        else:
            out.append(Candidate(
                f, False, f"ranked #{rank} of {len(ranked)}; keeps the best {keep}", rank,
            ))
    return out


def summary(candidates: "list[Candidate]") -> str:
    """One log line: who races and why the rest do not."""
    chosen = [c for c in candidates if c.selected]
    left = [c for c in candidates if not c.selected]
    text = "Candidates at 09:34: " + (
        ", ".join(f"{c.facts.label} ({c.facts.target_pct:.2f}%)" for c in chosen)
        if chosen else "none"
    )
    if left:
        text += ". Left out: " + "; ".join(f"{c.facts.label} — {c.reason}" for c in left)
    notes = sorted({f"{c.facts.ticker}: {c.facts.briefing_note}" for c in candidates if c.facts.briefing_note})
    if notes:
        text += ". Briefings: " + "; ".join(notes)
    return text + "."


def to_rows(candidates: "list[Candidate]") -> "list[dict]":
    """The selection as plain rows: what the board, the Candidates tab and
    the session file keep."""
    rows = []
    for c in candidates:
        f = c.facts
        rows.append({
            "key": f.key, "label": f.label, "ticker": f.ticker,
            "selected": c.selected, "reason": c.reason, "rank": c.rank,
            "bias": f.bias, "confidence": f.confidence, "briefing_note": f.briefing_note,
            "earnings": f.earnings, "earnings_known": f.earnings_known,
            "prev_close": f.prev_close, "adr": f.adr, "open": f.open, "last": f.last,
            "gap_adr": f.gap_adr, "move_adr": f.move_adr, "target_pct": f.target_pct,
        })
    return rows


# --- where the facts come from ----------------------------------------------------


def daily_stats(bars: "list[dict]") -> "tuple[float | None, float | None, date | None]":
    """`(previous close, ADR, previous session's date)` from completed daily
    bars, oldest first, in the app's {"t","o","h","l","c"} shape."""
    usable = [b for b in bars if b.get("h") is not None and b.get("l") is not None]
    if not usable:
        return None, None, None
    tail = usable[-ADR_SESSIONS:]
    adr = sum(float(b["h"]) - float(b["l"]) for b in tail) / len(tail)
    last = usable[-1]
    try:
        prev_day = pd.Timestamp(str(last["t"])[:10]).date()
    except (TypeError, ValueError):
        prev_day = None
    close = last.get("c")
    return (None if close is None else float(close)), (adr if adr > 0 else None), prev_day


def earnings_in_window(
    stamps: "list", day: date, prev_day: "date | None" = None
) -> "str | None":
    """The earnings report that falls between the previous session's close and
    this session's close, as "YYYY-MM-DD HH:MM ET", or None.

    A report after yesterday's bell gaps today's open; one before today's is
    the same day seen from the other side; one after today's bell is a
    position nobody holds over it, since every pair flattens before the close.
    """
    prev_day = prev_day or (day - timedelta(days=3 if day.weekday() == 0 else 1))
    start = datetime.combine(prev_day, time(16, 0), tzinfo=MARKET_TZ)
    end = datetime.combine(day, time(16, 0), tzinfo=MARKET_TZ)
    for stamp in sorted(pd.Timestamp(s) for s in stamps):
        stamp = stamp.tz_localize(MARKET_TZ) if stamp.tzinfo is None else stamp.tz_convert(MARKET_TZ)
        if start < stamp.to_pydatetime() <= end:
            return stamp.strftime("%Y-%m-%d %H:%M ET")
    return None


class LiveSources:
    """The briefing and the earnings calendar as the live app has them."""

    def __init__(self) -> None:
        self._earnings: "dict[tuple[str, date], list | None]" = {}
        self._lock = threading.Lock()

    def briefing(self, ticker: str, day: date, state) -> "tuple[dict | None, str]":
        briefing = (getattr(state, "premarket_briefings", None) or {}).get(ticker)
        if briefing is None:
            pending = ticker in (getattr(state, "premarket_pending", None) or [])
            return None, "still being written" if pending else "no briefing"
        made = getattr(state, "premarket_generated_at", None)
        if made is not None and made.astimezone(MARKET_TZ).date() != day:
            return None, f"the briefing is from {made.astimezone(MARKET_TZ):%Y-%m-%d}"
        return {"bias": briefing.overall_bias, "confidence": briefing.confidence}, ""

    def earnings(self, ticker: str, day: date) -> "list | None":
        """Report timestamps, read once a day per symbol; None when unreadable."""
        with self._lock:
            if (ticker, day) in self._earnings:
                return self._earnings[(ticker, day)]
        try:
            frame = historical.fetch_earnings_dates(ticker, days=400)
            stamps = list(frame.index) if frame is not None else []
        except Exception:
            stamps = None
        with self._lock:
            self._earnings[(ticker, day)] = stamps
        return stamps


LIVE_SOURCES = LiveSources()


def gather_facts(config, label: str, state, day: date, sources) -> PairFacts:
    """What is known about one pair now, from the bars on `state` and the
    briefing and calendar `sources` serves (`LiveSources`, or SimLab's)."""
    ticker = config.ticker
    facts = PairFacts(
        key=f"{ticker}:{config.model_key}", label=label, ticker=ticker,
        target_k=float(config.buy_k) - float(config.sell_k),
    )
    facts.prev_close, facts.adr, prev_day = daily_stats(
        historical.fetch_daily_ohlc_bars(ticker)
    )
    sym_state = state.sym(ticker) if hasattr(state, "sym") else None
    frame = momentum_regime.minute_frame(sym_state) if sym_state is not None else None
    if frame is not None and len(frame):
        opening = frame.iloc[:SELECT_AFTER_BARS]
        facts.last = float(opening["close"].iloc[-1])
        facts.open = float(opening["open"].iloc[0])
    official = historical.fetch_session_open(ticker)
    if official:
        facts.open = float(official)
    briefing, note = sources.briefing(ticker, day, state)
    if briefing is not None:
        facts.bias = briefing.get("bias")
        facts.confidence = briefing.get("confidence")
    facts.briefing_note = note
    stamps = sources.earnings(ticker, day)
    if stamps is None:
        facts.earnings_known = False
    else:
        facts.earnings = earnings_in_window(stamps, day, prev_day)
    return facts

