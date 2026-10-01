from datetime import datetime, timezone

from agent_stonks.report import build_report_html

SESSION_START = datetime(2024, 1, 15, 13, 20, tzinfo=timezone.utc)

DECISION = {
    "ts": "2024-01-15T14:30:00Z",
    "action": "buy",
    "status": "filled",
    "filled_quantity": 10.0,
    "price": 150.25,
    "fee": 1.15,
    "cash_after": 98500.0,
    "position_after": 10.0,
    "reasoning": "Strong momentum",
}

ALERT_DECISION = {
    "ts": "2024-01-15T14:35:00Z",
    "action": "alert",
    "status": "noop",
    "filled_quantity": 0.0,
    "price": None,
    "fee": 0.0,
    "cash_after": 98500.0,
    "position_after": 10.0,
    "reasoning": "Watching for breakout",
    "alerts": [{"field": "last_price", "condition": "above", "value": 160.0}],
}

LOG = [
    {"type": "cycle_start", "ts": "2024-01-15T14:30:00Z", "text": "cycle 1"},
    {
        "type": "decision",
        "ts": "2024-01-15T14:30:05Z",
        "action": "buy",
        "price": 150.25,
        "quantity": 10,
        "regime": "trending",
        "reasoning": "Strong momentum",
    },
    {"type": "error", "ts": "2024-01-15T14:31:00Z", "text": "boom"},
]


def _base_kwargs(**overrides) -> dict:
    kwargs = dict(
        symbols=["AAPL"],
        feed="iex",
        timeframe="1Min",
        session_start=SESSION_START,
        starting_budget=100_000.0,
        trade_fixed_cost=1.15,
        llm_provider="openai",
        llm_model="",
        llm_personality="Swing / Position Trader",
        agent_running=True,
        live_figs=[],
        performance_fig=None,
        performance_stats=None,
        decisions=[],
        agent_log=[],
    )
    kwargs.update(overrides)
    return kwargs


class TestBuildReportHtml:
    def test_produces_valid_html_document(self):
        result = build_report_html(**_base_kwargs())
        assert result.startswith("<!DOCTYPE html>")
        assert "<html" in result and "</html>" in result

    def test_includes_symbol_and_starting_conditions(self):
        result = build_report_html(**_base_kwargs(symbols=["AAPL", "TSLA"], starting_budget=50_000.0))
        assert "AAPL, TSLA" in result
        assert "$50,000.00" in result

    def test_empty_state_shows_placeholders(self):
        result = build_report_html(**_base_kwargs())
        assert "No live chart data available." in result
        assert "No decisions recorded." in result
        assert "No agent activity recorded." in result

    def test_renders_decision_table_with_reasoning(self):
        result = build_report_html(**_base_kwargs(decisions=[DECISION]))
        assert "Strong momentum" in result
        assert "$150.2500" in result
        assert "action-buy" in result

    def test_renders_alert_decision_with_levels(self):
        result = build_report_html(**_base_kwargs(decisions=[ALERT_DECISION]))
        assert "wake when last_price above 160" in result

    def test_renders_agent_log_entries(self):
        result = build_report_html(**_base_kwargs(agent_log=LOG))
        assert "cycle 1" in result
        assert "BUY" in result
        assert "boom" in result

    def test_includes_performance_summary_when_present(self):
        stats = {"current_value": 105_000.0, "return_pct": 5.0, "total_fees": 2.30}
        result = build_report_html(**_base_kwargs(performance_stats=stats))
        assert "$105,000.00" in result
        assert "+5.00%" in result

    def test_briefing_and_news_cards_are_included(self):
        result = build_report_html(
            **_base_kwargs(
                briefing_title="Intraday Situation Briefing",
                briefing_note="Generated 2024-01-15 10:05 ET",
                briefing_cards=["<div>BRIEFING-CARD</div>"],
                news_cards=["<div>NEWS-CARD</div>"],
            )
        )
        assert "Intraday Situation Briefing" in result
        assert "Generated 2024-01-15 10:05 ET" in result
        assert "BRIEFING-CARD" in result and "NEWS-CARD" in result
        # Context comes before the charts.
        assert result.index("BRIEFING-CARD") < result.index("NEWS-CARD") < result.index("Live chart")

    def test_missing_briefing_news_and_walls_show_placeholders(self):
        result = build_report_html(**_base_kwargs())
        assert "No briefing was generated for this run." in result
        assert "No news was loaded for this run." in result
        assert "No options chain data available." in result

    def test_option_walls_section(self):
        import plotly.graph_objects as go

        analysis = {
            "call_wall": 105.0,
            "put_wall": 95.0,
            "call_wall_trend": "rising",
            "put_wall_trend": None,
            "gamma_regime": "negative (amplifying)",
            "summary": "Call wall 105.00 (resistance), put wall 95.00 (support).",
            "insights": ["Spot inside the range.", "Net dealer gamma is negative."],
        }
        walls = [
            {
                "symbol": "AAPL",
                "fig": go.Figure(),
                "expiry": "2024-01-19",
                "fetched_at": "2024-01-15T14:00:00+00:00",
                "analysis": analysis,
            }
        ]
        result = build_report_html(**_base_kwargs(option_walls=walls))
        assert "Put/Call walls — AAPL" in result
        assert "Expiry 2024-01-19" in result
        assert "$105.00 (rising)" in result
        assert "$95.00" in result
        assert "Negative" in result
        assert "Net dealer gamma is negative." in result
        # With no live chart above it, the walls chart is the one that loads plotly.js.
        assert result.count("cdn.plot.ly") == 1
