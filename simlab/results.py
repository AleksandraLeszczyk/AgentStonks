"""Simulation run records: scoring, persistence, and optional Langfuse export.

A finished simulation is summarized (profit, trade counts, an oracle
best-round-trip ceiling and the profit efficiency against it -- the same
oracle the live daily scoring uses), optionally judged by an LLM
(``simlab.judge``), and saved as one JSON file under ``data/simlab/runs/``.

When Langfuse is configured the headline metrics are also registered there as
scores on a ``simlab-run`` trace, so runs line up next to the live agent's
traces and daily scores in the same project. The local JSON stays the source
of truth -- Langfuse is an export target, never a dependency.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from agent_stonks import observability as obs

from .engine import SimulationResult
from .market import SimMarket

RUNS_DIR = Path(__file__).resolve().parent.parent / "data" / "simlab" / "runs"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def oracle_best_round_trip(closes: list[float]) -> float:
    """Max single-round-trip profit percentage over an ordered close series
    (buy at the running minimum, sell at the best later price). 0 when the
    series never rises."""
    best = 0.0
    low: Optional[float] = None
    for price in closes:
        if price <= 0:
            continue
        if low is None or price < low:
            low = price
            continue
        best = max(best, (price / low - 1.0) * 100.0)
    return best


def summarize_run(result: SimulationResult, market: SimMarket) -> dict:
    """Headline stats: return, fees, trade counts, and the oracle ceiling."""
    decisions = result.decisions
    fills = [d for d in decisions if d.get("status") == "filled" and d.get("action") in ("buy", "sell")]
    per_symbol_oracle = {}
    for symbol, series in market.series.items():
        closes = [float(b["c"]) for b in series.minute_bars if b.get("c") is not None]
        per_symbol_oracle[symbol] = round(oracle_best_round_trip(closes), 3)
    oracle_ceiling = max(per_symbol_oracle.values(), default=0.0)
    return_pct = (
        (result.final_value / result.starting_cash - 1.0) * 100.0 if result.starting_cash else 0.0
    )
    efficiency = (return_pct / oracle_ceiling) if oracle_ceiling > 0 else None
    return {
        "starting_cash": result.starting_cash,
        "final_value": round(result.final_value, 2),
        "profit": round(result.final_value - result.starting_cash, 2),
        "return_pct": round(return_pct, 4),
        "total_fees": round(sum(d.get("fee") or 0.0 for d in decisions), 2),
        "cycles_run": result.cycles_run,
        "trades_filled": len(fills),
        "buys": sum(1 for d in fills if d["action"] == "buy"),
        "sells": sum(1 for d in fills if d["action"] == "sell"),
        "tactics_armed": sum(1 for d in decisions if d.get("action") == "tactics"),
        "alerts_set": sum(1 for d in decisions if d.get("action") == "alert"),
        "oracle_best_round_trip_pct": per_symbol_oracle,
        "oracle_ceiling_pct": round(oracle_ceiling, 3),
        "profit_efficiency": round(efficiency, 4) if efficiency is not None else None,
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_run(
    result: SimulationResult,
    summary: dict,
    judge_report: "dict | None" = None,
    dataset_name: str = "",
) -> dict:
    """Persist one finished run; returns the full stored record."""
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    record = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset_name,
        **asdict(result),
        "summary": summary,
        "judge": judge_report,
    }
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    (RUNS_DIR / f"{run_id}.json").write_text(json.dumps(record, indent=2, default=str))
    _export_to_langfuse(record)
    return record


def list_runs() -> list[dict]:
    """Stored run records, newest first (full records -- they're small)."""
    if not RUNS_DIR.exists():
        return []
    records = []
    for path in sorted(RUNS_DIR.glob("*.json"), reverse=True):
        try:
            records.append(json.loads(path.read_text()))
        except (OSError, json.JSONDecodeError):
            continue
    return records


def delete_run(run_id: str) -> None:
    path = RUNS_DIR / f"{run_id}.json"
    if path.exists():
        path.unlink()


# ---------------------------------------------------------------------------
# Langfuse export (no-op when unconfigured, like the rest of observability)
# ---------------------------------------------------------------------------

def _export_to_langfuse(record: dict) -> None:
    if not obs.is_enabled():
        return
    summary = record.get("summary") or {}
    config = record.get("config_summary") or {}
    label = (
        f"simlab-run:{config.get('personality')}:{record.get('dataset') or ','.join(config.get('symbols', []))}"
    )
    comment = json.dumps(
        {"run_id": record["run_id"], **{k: config.get(k) for k in ("personality", "model", "days")}}
    )
    scores = {
        "sim-return-pct": summary.get("return_pct"),
        "sim-profit-efficiency": summary.get("profit_efficiency"),
    }
    judge = record.get("judge") or {}
    if judge.get("overall_score") is not None:
        scores["sim-judge-overall"] = judge["overall_score"]
    for name, value in scores.items():
        if value is None:
            continue
        obs.record_score(
            trace_name=label, name=name, value=float(value), comment=comment, input=summary
        )
