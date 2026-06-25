"""
Builds a single self-contained HTML file documenting an agent run: starting
conditions, the Live/Historical/Agent charts as they looked at save time, and
the full history of agent decisions and activity.
"""
from __future__ import annotations

import html
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import plotly.graph_objects as go

from .config import PALETTE

_CSS = f"""
body {{
    background: {PALETTE['bg']};
    color: {PALETTE['text']};
    font-family: Inter, -apple-system, sans-serif;
    margin: 0;
    padding: 24px 32px 48px;
}}
h1 {{ font-size: 22px; margin: 0 0 4px; }}
h2 {{
    font-size: 16px;
    color: {PALETTE['accent']};
    border-bottom: 1px solid {PALETTE['grid']};
    padding-bottom: 6px;
    margin: 32px 0 14px;
}}
.subtitle {{ color: {PALETTE['muted']}; font-size: 13px; margin: 0 0 24px; }}
.conditions {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: 10px;
    margin-bottom: 8px;
}}
.cond-item {{
    background: {PALETTE['panel']};
    border: 1px solid {PALETTE['grid']};
    border-radius: 8px;
    padding: 10px 14px;
}}
.cond-label {{
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: {PALETTE['muted']};
    margin-bottom: 4px;
}}
.cond-value {{ font-size: 15px; font-weight: 600; }}
table.report-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
}}
table.report-table th, table.report-table td {{
    border-bottom: 1px solid {PALETTE['grid']};
    padding: 6px 10px;
    text-align: left;
    vertical-align: top;
}}
table.report-table th {{
    color: {PALETTE['muted']};
    text-transform: uppercase;
    font-size: 10px;
    letter-spacing: 0.04em;
}}
tr.action-buy td:nth-child(2) {{ color: {PALETTE['up']}; font-weight: 600; }}
tr.action-sell td:nth-child(2) {{ color: {PALETTE['down']}; font-weight: 600; }}
tr.action-alert td:nth-child(2) {{ color: {PALETTE['orange']}; font-weight: 600; }}
.empty-note {{ color: {PALETTE['muted']}; font-style: italic; }}
.log-card {{
    background: {PALETTE['panel']};
    border-radius: 8px;
    padding: 10px 14px;
    border: 1px solid {PALETTE['grid']};
    margin-bottom: 8px;
    font-size: 12px;
}}
.log-head {{
    display: flex;
    justify-content: space-between;
    color: {PALETTE['muted']};
    font-size: 11px;
    margin-bottom: 4px;
}}
"""

_AGENT_ENTRY_COLORS: dict[str, str] = {
    "buy": PALETTE["up"],
    "sell": PALETTE["down"],
    "alert": PALETTE["accent"],
    "sleep": PALETTE["muted"],
}


def _cond_item(label: str, value: str) -> str:
    return (
        f'<div class="cond-item"><div class="cond-label">{html.escape(label)}</div>'
        f'<div class="cond-value">{html.escape(value)}</div></div>'
    )


def _starting_conditions_html(conditions: dict[str, str]) -> str:
    items = "".join(_cond_item(label, value) for label, value in conditions.items())
    return f'<div class="conditions">{items}</div>'


def _fig_html(fig: Optional[go.Figure], include_plotlyjs: bool, empty_msg: str) -> str:
    if fig is None:
        return f'<p class="empty-note">{html.escape(empty_msg)}</p>'
    return fig.to_html(full_html=False, include_plotlyjs="cdn" if include_plotlyjs else False)


def _decisions_table_html(decisions: list[dict]) -> str:
    if not decisions:
        return '<p class="empty-note">No decisions recorded.</p>'
    rows = []
    for d in decisions:
        action = str(d.get("action", ""))
        price = d.get("price")
        price_str = f"${price:,.4f}" if price is not None else "—"
        qty = d.get("filled_quantity") or 0.0
        fee = d.get("fee") or 0.0
        cash_after = d.get("cash_after")
        cash_str = f"${cash_after:,.2f}" if cash_after is not None else "—"
        position_after = d.get("position_after") or 0.0
        reasoning = html.escape(d.get("reasoning", ""))
        if d.get("action") == "alert" and d.get("alerts"):
            reasoning += " " + " · ".join(
                f"[wake {a.get('condition')} ${a.get('price'):,.4f}]" for a in d["alerts"]
            )
        try:
            ts_fmt = pd.to_datetime(d.get("ts", "")).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            ts_fmt = str(d.get("ts", ""))
        rows.append(
            f'<tr class="action-{html.escape(action)}">'
            f"<td>{ts_fmt}</td><td>{html.escape(action)}</td>"
            f"<td>{html.escape(str(d.get('status', '')))}</td>"
            f"<td>{qty:.4f}</td><td>{price_str}</td><td>${fee:.2f}</td>"
            f"<td>{cash_str}</td><td>{position_after:.4f}</td>"
            f"<td>{reasoning}</td></tr>"
        )
    return (
        "<table class='report-table'><thead><tr>"
        "<th>Time (UTC)</th><th>Action</th><th>Status</th><th>Qty</th><th>Price</th>"
        "<th>Fee</th><th>Cash after</th><th>Position after</th><th>Reasoning</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def _agent_log_html(log: list[dict]) -> str:
    if not log:
        return '<p class="empty-note">No agent activity recorded.</p>'
    cards = []
    for entry in log:
        etype = entry.get("type", "status")
        label = etype
        color = PALETTE["muted"]
        if etype == "decision":
            action = entry.get("action", "sleep")
            label = action.upper()
            color = _AGENT_ENTRY_COLORS.get(action, PALETTE["muted"])
            price = entry.get("price")
            price_str = f"${price:,.4f}" if price is not None else "—"
            qty = entry.get("quantity") or 0
            regime = html.escape(str(entry.get("regime", "unknown")))
            reasoning = html.escape(entry.get("reasoning", ""))
            body = (
                f"<div>Regime: <b>{regime}</b> · Qty: <b>{qty:.2f}</b> · "
                f"Price: <b>{price_str}</b></div>"
                f"<div style='color:{PALETTE['muted']}'>{reasoning}</div>"
            )
        elif etype == "tool_call":
            label = str(entry.get("name", "tool"))
            color = PALETTE["accent"]
            result = html.escape(entry.get("result_preview", ""))
            body = f"<div>{result}</div>"
        else:
            color = PALETTE["down"] if etype == "error" else PALETTE["text"]
            body = f"<div>{html.escape(entry.get('text', ''))}</div>"
        try:
            ts_fmt = pd.to_datetime(entry.get("ts", "")).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            ts_fmt = str(entry.get("ts", ""))
        cards.append(
            f'<div class="log-card" style="border-left:3px solid {color}">'
            f'<div class="log-head"><span style="color:{color};font-weight:600">'
            f"{html.escape(str(label))}</span><span>{ts_fmt}</span></div>"
            f"{body}</div>"
        )
    return "".join(cards)


def build_report_html(
    *,
    symbol: str,
    feed: str,
    timeframe: str,
    session_start: datetime,
    starting_budget: float,
    trade_fixed_cost: float,
    llm_provider: str,
    llm_model: str,
    llm_personality: str,
    agent_running: bool,
    live_fig: Optional[go.Figure],
    historical_fig: Optional[go.Figure],
    historical_period_label: Optional[str],
    performance_fig: Optional[go.Figure],
    performance_stats: Optional[dict],
    decisions: list[dict],
    agent_log: list[dict],
) -> str:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    conditions = {
        "Symbol": symbol or "—",
        "Feed": feed,
        "Timeframe": timeframe,
        "Session start": session_start.strftime("%Y-%m-%d %H:%M UTC"),
        "Starting budget": f"${starting_budget:,.2f}",
        "Fee per trade": f"${trade_fixed_cost:.2f}",
        "LLM provider": llm_provider,
        "LLM model": llm_model or "(default)",
        "Agent personality": llm_personality,
        "Agent status": "running" if agent_running else "stopped",
    }

    perf_summary = ""
    if performance_stats:
        perf_summary = _starting_conditions_html(
            {
                "Portfolio value": f"${performance_stats['current_value']:,.2f}",
                "Return": f"{performance_stats['return_pct']:+.2f}%",
                "Total fees paid": f"${performance_stats['total_fees']:,.2f}",
            }
        )

    hist_title = f"Historical — {historical_period_label}" if historical_period_label else "Historical"

    body = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Agent run report — {html.escape(symbol or '')}</title>
<style>{_CSS}</style>
</head>
<body>
<h1>Agent run report — {html.escape(symbol or 'Unknown')}</h1>
<p class="subtitle">Generated {generated_at}</p>

<h2>Starting conditions</h2>
{_starting_conditions_html(conditions)}

<h2>Live chart</h2>
{_fig_html(live_fig, include_plotlyjs=True, empty_msg="No live chart data available.")}

<h2>{html.escape(hist_title)}</h2>
{_fig_html(historical_fig, include_plotlyjs=False, empty_msg="No historical chart data available.")}

<h2>Agent performance</h2>
{perf_summary}
{_fig_html(performance_fig, include_plotlyjs=False, empty_msg="No agent performance data available.")}

<h2>Decision history</h2>
{_decisions_table_html(decisions)}

<h2>Agent activity log</h2>
{_agent_log_html(agent_log)}

</body>
</html>"""
    return body
