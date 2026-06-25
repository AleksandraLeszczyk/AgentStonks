from datetime import datetime, timezone

from marketview.report import build_report_html

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
    "alerts": [{"price": 160.0, "condition": "above"}],
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
        symbol="AAPL",
        feed="iex",
        timeframe="1Min",
        session_start=SESSION_START,
        starting_budget=100_000.0,
        trade_fixed_cost=1.15,
        llm_provider="gemini",
        llm_model="",
        llm_personality="Swing / Position Trader",
        agent_running=True,
        live_fig=None,
        historical_fig=None,
        historical_period_label=None,
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
        result = build_report_html(**_base_kwargs(symbol="AAPL", starting_budget=50_000.0))
        assert "AAPL" in result
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
        assert "wake above $160.0000" in result

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

    def test_historical_period_label_in_heading(self):
        result = build_report_html(**_base_kwargs(historical_period_label="1 Year"))
        assert "Historical — 1 Year" in result
