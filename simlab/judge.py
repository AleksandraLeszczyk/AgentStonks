"""LLM-as-judge: were the simulated entries reasonable?

Profit alone is a noisy verdict on one session -- a bad entry can luck into a
gain and a textbook entry can stop out. The judge grades each *entry* on the
information the agent had at that moment (its stated reasoning, the tape
leading in) against what the tape did next (excursions, the exit), then rolls
the trade grades plus the run stats into an overall assessment.

Runs on the same provider-agnostic ``llm.parse_structured`` path as the rest
of the app, so any configured provider can judge any provider's run.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Optional

from pydantic import BaseModel, Field

from agent_stonks.llm import parse_structured

from .market import SimMarket, parse_ts

# Tape shown to the judge around each entry.
CONTEXT_BARS_BEFORE = 30
OUTCOME_WINDOW_MIN = 60

JUDGE_SYSTEM = """\
You are a strict but fair trading coach reviewing a single simulated intraday \
trade entry made by an autonomous LLM trading agent. Judge ONLY what the agent \
could know at entry time: does its stated reasoning match the tape leading in, \
was the entry level/timing sound for the strategy it claims to follow, and was \
the risk framing (stop/target, sizing rationale) coherent? The outcome that \
followed is shown to calibrate severity (a good process with a bad outcome is \
still a good process), not to reward luck. Score 0-10: 8-10 disciplined A-grade \
process, 5-7 defensible with flaws, 2-4 weak/chased/contradicted by the tape, \
0-1 reckless or ungrounded in the data cited."""

RUN_SYSTEM = """\
You are a trading-desk reviewer summarizing an autonomous LLM trading agent's \
simulated session. You are given the strategy brief the agent was running, the \
session statistics (including an oracle ceiling: the best single round trip \
available on the tape), and per-entry judgments already produced by a trade \
reviewer. Assess how well the agent executed ITS OWN strategy: selectivity \
(trading only its A+ setups vs overtrading or freezing), entry quality, exit \
and risk management, and honest use of the data. Be concrete and reference the \
numbers. Score 0-10 overall; list the most important improvements."""


class EntryJudgment(BaseModel):
    score: int = Field(ge=0, le=10, description="Entry process quality, 0-10.")
    verdict: str = Field(description="One of: good_entry, acceptable, poor_entry.")
    reasoning_quality: str = Field(
        description="Did the agent's cited evidence actually support the entry? 1-3 sentences."
    )
    what_went_well: str
    what_to_improve: str


class RunJudgment(BaseModel):
    overall_score: int = Field(ge=0, le=10, description="Overall execution quality, 0-10.")
    strategy_adherence: int = Field(
        ge=0, le=10, description="How faithfully the agent followed its own strategy brief, 0-10."
    )
    summary: str = Field(description="3-6 sentence assessment of the session.")
    top_improvements: list[str] = Field(description="The 2-4 highest-impact improvements.")


def _fmt_bar(bar: dict) -> str:
    return (
        f"{str(bar.get('t'))[11:16]} o={bar.get('o')} h={bar.get('h')} "
        f"l={bar.get('l')} c={bar.get('c')} v={bar.get('v')}"
    )


def _entry_context(entry: dict, exit_decision: "dict | None", market: SimMarket) -> "str | None":
    """The user prompt for one entry judgment, or None when the tape around
    the entry can't be reconstructed."""
    ts = parse_ts(entry.get("ts"))
    symbol = str(entry.get("symbol") or "")
    if ts is None or symbol not in market.series:
        return None
    before = market.completed_bars(symbol, ts)[-CONTEXT_BARS_BEFORE:]
    series = market.series[symbol]
    horizon = ts + timedelta(minutes=OUTCOME_WINDOW_MIN)
    after = [
        bar
        for bar, bts in zip(series.minute_bars, series.minute_ts)
        if ts < bts <= horizon
    ]
    price = float(entry.get("price") or 0.0)
    lines = [
        f"ENTRY: buy {entry.get('filled_quantity')} sh {symbol} @ {price} at {entry.get('ts')}",
        f"AGENT'S STATED REASONING: {entry.get('reasoning')}",
        "",
        f"TAPE BEFORE ENTRY (last {len(before)} one-minute bars):",
        *[_fmt_bar(b) for b in before],
    ]
    if after:
        highs = [float(b["h"]) for b in after]
        lows = [float(b["l"]) for b in after]
        mfe = (max(highs) / price - 1.0) * 100.0 if price else 0.0
        mae = (min(lows) / price - 1.0) * 100.0 if price else 0.0
        lines += [
            "",
            f"OUTCOME (next {OUTCOME_WINDOW_MIN} min): max favorable excursion "
            f"{mfe:+.2f}%, max adverse excursion {mae:+.2f}%, "
            f"close of window {float(after[-1]['c'])}",
        ]
    if exit_decision is not None:
        exit_price = exit_decision.get("price")
        pnl = ((float(exit_price) / price - 1.0) * 100.0) if price and exit_price else None
        lines.append(
            f"EXIT: sold {exit_decision.get('filled_quantity')} sh @ {exit_price} at "
            f"{exit_decision.get('ts')}"
            + (f" ({pnl:+.2f}% vs this entry)" if pnl is not None else "")
        )
        lines.append(f"EXIT REASONING: {exit_decision.get('reasoning')}")
    else:
        lines.append("EXIT: position still open at session end (marked to close).")
    return "\n".join(lines)


def _first_exit_after(entry: dict, decisions: list[dict]) -> "dict | None":
    """First filled sell in the same symbol recorded after `entry` -- by ledger
    order, since a simulated fill and its exit can share the same bar minute."""
    try:
        start = decisions.index(entry) + 1
    except ValueError:
        start = 0
    for d in decisions[start:]:
        if (
            d.get("action") == "sell"
            and d.get("status") == "filled"
            and d.get("symbol") == entry.get("symbol")
        ):
            return d
    return None


def judge_run(
    decisions: list[dict],
    summary: dict,
    strategy_prompt: str,
    market: SimMarket,
    provider: str,
    api_key: str,
    model: str,
    progress=lambda msg: None,
) -> dict:
    """Judge every filled entry plus the run overall. Returns a JSON-ready
    report; individual judgment failures degrade to error notes, never raise."""
    entries = [
        d for d in decisions if d.get("action") == "buy" and d.get("status") == "filled"
    ]
    entry_reports: list[dict] = []
    for i, entry in enumerate(entries, 1):
        progress(f"judging entry {i}/{len(entries)}")
        context = _entry_context(entry, _first_exit_after(entry, decisions), market)
        report: dict = {
            "ts": entry.get("ts"),
            "symbol": entry.get("symbol"),
            "price": entry.get("price"),
            "quantity": entry.get("filled_quantity"),
        }
        if context is None:
            report["error"] = "could not reconstruct tape context"
            entry_reports.append(report)
            continue
        try:
            judgment: Optional[EntryJudgment] = parse_structured(
                provider, api_key, model, JUDGE_SYSTEM, context, EntryJudgment
            )
            if judgment is None:
                report["error"] = "judge returned no structured result"
            else:
                report.update(judgment.model_dump())
        except Exception as exc:
            report["error"] = str(exc)
        entry_reports.append(report)

    progress("judging the run overall")
    scored = [r for r in entry_reports if "score" in r]
    run_context = "\n".join(
        [
            "STRATEGY BRIEF THE AGENT WAS RUNNING:",
            strategy_prompt.strip(),
            "",
            f"SESSION STATS: {summary}",
            "",
            f"PER-ENTRY JUDGMENTS ({len(scored)} of {len(entry_reports)} entries scored):",
            *(
                [
                    f"- {r['ts']} {r['symbol']} @ {r['price']}: score {r.get('score')} "
                    f"({r.get('verdict')}) -- {r.get('what_to_improve')}"
                    for r in scored
                ]
                or ["(no entries were made this session)"]
            ),
        ]
    )
    overall: dict = {}
    try:
        run_judgment: Optional[RunJudgment] = parse_structured(
            provider, api_key, model, RUN_SYSTEM, run_context, RunJudgment
        )
        if run_judgment is not None:
            overall = run_judgment.model_dump()
    except Exception as exc:
        overall = {"error": str(exc)}

    avg = round(sum(r["score"] for r in scored) / len(scored), 2) if scored else None
    return {
        "judge_model": f"{provider}/{model}",
        "entries": entry_reports,
        "avg_entry_score": avg,
        **overall,
    }
