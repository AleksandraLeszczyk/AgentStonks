import json
from datetime import datetime, timedelta, timezone

import pytest

from marketview import scoring
from marketview.scoring import (
    Scorecard,
    begin_session,
    end_session,
    grounding_from_messages,
    maybe_score_week,
    record_activation_end,
    record_activation_start,
    record_cycle_grounding,
    record_tactics_call,
    record_tool_call,
    week_report_path,
)
from marketview.state import AppState


def _decision_call(name: str, args: dict) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
        ],
    }


def _tool_msg(content: dict) -> dict:
    return {"role": "tool", "tool_call_id": "t1", "content": json.dumps(content)}


class TestGroundingFromMessages:
    def test_numbers_seen_in_tool_results_are_grounded(self):
        messages = [
            {"role": "system", "content": "You trade stocks."},
            _tool_msg({"last_price": 587.0, "vwap": 585.25}),
            _decision_call(
                "submit_decision",
                {"action": "alert", "reasoning": "price 587 above vwap 585.25",
                 "alerts": [{"symbol": "SPY", "field": "last_price",
                             "condition": "above", "value": 587.0}]},
            ),
        ]
        result = grounding_from_messages(messages)
        assert result["score"] == 1.0
        assert result["ungrounded"] == []

    def test_invented_number_is_ungrounded(self):
        messages = [
            _tool_msg({"last_price": 587.0}),
            _decision_call(
                "submit_decision",
                {"action": "alert", "reasoning": "resistance at 612.5",
                 "alerts": [{"symbol": "SPY", "field": "last_price",
                             "condition": "above", "value": 612.5}]},
            ),
        ]
        result = grounding_from_messages(messages)
        assert result["score"] == 0.0
        assert 612.5 in result["ungrounded"]

    def test_number_shown_only_after_the_decision_does_not_ground_it(self):
        messages = [
            _tool_msg({"last_price": 587.0}),
            _decision_call("set_tactics", {"symbol": "SPY", "reasoning": "stop 55.5"}),
            _tool_msg({"revealed_later": 55.5}),
        ]
        result = grounding_from_messages(messages)
        assert result["score"] == 0.0

    def test_quantity_and_small_numbers_are_exempt(self):
        messages = [
            _tool_msg({"last_price": 587.0}),
            _decision_call(
                "submit_decision",
                {"action": "buy", "symbol": "SPY", "quantity": 42,
                 "reasoning": "buying 3 lots at 587"},
            ),
        ]
        result = grounding_from_messages(messages)
        # Only 587 is auditable: quantity=42 is exempt by key, 3 is below the floor.
        assert result["total"] == 1
        assert result["score"] == 1.0

    def test_whole_dollar_rounding_slack_above_100(self):
        # The model saw 587 (rounded); echoing 587.0 vs a raw 586.8 elsewhere is fine,
        # and derived values within 1% of a shown level also count.
        messages = [
            _tool_msg({"support": 587}),
            _decision_call("set_tactics", {"reasoning": "entry near 586.6"}),
        ]
        assert grounding_from_messages(messages)["score"] == 1.0

    def test_no_auditable_numbers_returns_none(self):
        messages = [
            _tool_msg({"note": "no data"}),
            _decision_call("submit_decision", {"action": "alert", "reasoning": "quiet"}),
        ]
        assert grounding_from_messages(messages) is None

    def test_prompt_numbers_count_as_seen(self):
        messages = [
            {"role": "system", "content": "ADX below 20 means ranging"},
            _decision_call("select_strategy", {"strategy": "reversal",
                                               "reasoning": "ADX at 20, ranging"}),
        ]
        assert grounding_from_messages(messages)["score"] == 1.0


class TestScorecardRecording:
    def _state(self) -> AppState:
        state = AppState()
        begin_session(state, "momentum", ["SPY"])
        return state

    def test_begin_session_attaches_scorecard(self):
        state = self._state()
        assert isinstance(state.scorecard, Scorecard)
        assert state.scorecard.mode == "momentum"
        assert state.scorecard.start_value == state.starting_budget

    def test_tactics_rejections_counted(self):
        state = self._state()
        record_tactics_call(state, ok=True)
        record_tactics_call(state, ok=False)
        record_tactics_call(state, ok=False)
        assert state.scorecard.tactics_attempts == 3
        assert state.scorecard.tactics_rejections == 2

    def test_tool_errors_and_quote_warnings_counted(self):
        state = self._state()
        record_tool_call(state, "analyze_volume", {"relative_volume": 1.2})
        record_tool_call(state, "analyze_volume", {"error": "boom"})
        record_tool_call(state, "get_quote", {"last_price": 10.0})
        record_tool_call(state, "get_quote", {"last_price": 10.0, "warning": "quote is 90 min old"})
        card = state.scorecard
        assert card.tool_calls == 4
        assert card.tool_errors == 1
        assert card.quote_calls == 2
        assert card.quote_warnings == 1

    def test_cycle_grounding_recorded(self):
        state = self._state()
        messages = [
            _tool_msg({"last_price": 587.0}),
            _decision_call("submit_decision", {"action": "alert", "reasoning": "hold at 587"}),
        ]
        record_cycle_grounding(state, messages, "momentum")
        record_cycle_grounding(state, [], "momentum")  # no numbers -> counted, not scored
        card = state.scorecard
        assert card.cycles_run == 2
        assert len(card.grounding) == 1
        assert card.grounding[0]["score"] == 1.0

    def test_hooks_are_noops_without_scorecard(self):
        state = AppState()
        record_tactics_call(state, ok=False)
        record_tool_call(state, "get_quote", {"error": "x"})
        record_cycle_grounding(state, [], "momentum")
        record_activation_start(state, "momentum", "quiet")
        record_activation_end(state)
        assert state.scorecard is None

    def test_activation_windows_open_and_close(self):
        state = self._state()
        record_activation_start(state, "momentum", "bullish_trend")
        record_activation_end(state)
        record_activation_start(state, "reversal", "ranging")
        record_activation_start(state, "breakout", "breakout_pending")  # implicitly closes reversal
        card = state.scorecard
        assert [a["strategy"] for a in card.activations] == ["momentum", "reversal"]
        assert card._open_activation["strategy"] == "breakout"


class TestDecisionQuality:
    def test_counts_and_active_rate(self):
        decisions = [
            {"action": "buy", "status": "filled"},
            {"action": "sell", "status": "rejected"},
            {"action": "tactics", "status": "armed"},
            {"action": "alert", "status": "noop"},
            {"action": "sleep", "status": "noop"},
        ]
        quality = scoring._decision_quality(decisions, 100_000.0, 101_000.0)
        assert quality["buy_sell_filled"] == 1
        assert quality["buy_sell_rejected"] == 1
        assert quality["tactics_armed"] == 1
        assert quality["alerts"] == 1
        assert quality["forced_sleeps"] == 1
        assert quality["active"] == 2
        assert quality["active_rate"] == pytest.approx(0.4)
        assert quality["return_pct"] == pytest.approx(1.0)


class TestActivationOutcomes:
    def test_alerts_only_window_is_not_effective(self):
        activations = [
            {"strategy": "momentum", "regime": "quiet",
             "started_at": "2026-07-06T10:00:00+00:00", "ended_at": "2026-07-06T12:00:00+00:00"},
            {"strategy": "breakout", "regime": "breakout_pending",
             "started_at": "2026-07-06T12:00:00+00:00", "ended_at": "2026-07-06T14:00:00+00:00"},
        ]
        decisions = [
            {"ts": "2026-07-06T10:30:00+00:00", "action": "alert", "status": "noop"},
            {"ts": "2026-07-06T11:30:00+00:00", "action": "alert", "status": "noop"},
            {"ts": "2026-07-06T12:30:00+00:00", "action": "tactics", "status": "armed"},
            {"ts": "2026-07-06T13:00:00+00:00", "action": "buy", "status": "filled"},
        ]
        outcomes = scoring._activation_outcomes(activations, decisions)
        momentum, breakout = outcomes
        assert momentum["effective"] is False  # only alarms set: bad fit that day
        assert momentum["alert_decisions"] == 2
        assert breakout["effective"] is True
        assert breakout["active_decisions"] == 2


class TestWeeklyScoring:
    @pytest.fixture(autouse=True)
    def _tmp_scoring_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(scoring, "SCORING_DIR", tmp_path / "scoring")
        self.dir = tmp_path / "scoring"

    def _journal_session(self, started_at: datetime, runtime_sec: float, **overrides) -> dict:
        record = {
            "started_at": started_at.isoformat(),
            "ended_at": (started_at + timedelta(seconds=runtime_sec)).isoformat(),
            "runtime_sec": runtime_sec,
            "mode": "momentum",
            "symbols": ["SPY"],
            "cycles": 5,
            "grounding": {"scored_cycles": 4, "mean_score": 0.9, "min_score": 0.75,
                          "ungrounded": [612.5]},
            "tactics": {"attempts": 3, "rejections": 1},
            "tools": {"calls": 20, "errors": 2, "quote_calls": 5, "quote_warnings": 1},
            "decisions": {"total": 5, "buy_sell_filled": 1, "buy_sell_rejected": 0,
                          "tactics_armed": 1, "alerts": 3, "forced_sleeps": 0,
                          "active": 2, "active_rate": 0.4,
                          "start_value": 100_000.0, "end_value": 100_500.0,
                          "return_pct": 0.5},
            "activations": [],
        }
        record.update(overrides)
        scoring._append_journal(record)
        return record

    def test_skips_below_one_hour_total(self):
        now = datetime(2026, 7, 8, 15, 0, tzinfo=timezone.utc)
        self._journal_session(now - timedelta(hours=2), runtime_sec=1200)
        self._journal_session(now - timedelta(hours=1), runtime_sec=1800)
        assert maybe_score_week(now=now) is None
        assert not week_report_path("2026-W28").exists()

    def test_short_sessions_accumulate_across_the_week(self):
        now = datetime(2026, 7, 8, 15, 0, tzinfo=timezone.utc)
        self._journal_session(now - timedelta(days=1), runtime_sec=2000)
        self._journal_session(now - timedelta(hours=2), runtime_sec=2000)
        report = maybe_score_week(now=now)
        assert report is not None
        assert report["sessions"] == 2
        assert report["total_runtime_sec"] == 4000
        assert week_report_path("2026-W28").exists()

    def test_scored_week_registers_langfuse_score(self, monkeypatch):
        pushed = []
        monkeypatch.setattr(
            scoring.obs, "record_score", lambda **kwargs: pushed.append(kwargs)
        )
        now = datetime(2026, 7, 8, 15, 0, tzinfo=timezone.utc)
        self._journal_session(now - timedelta(hours=3), runtime_sec=4000)
        assert maybe_score_week(now=now) is not None
        assert len(pushed) == 1
        assert pushed[0]["name"] == "weekly-grounding"
        assert pushed[0]["value"] == 0.9
        assert pushed[0]["input"]["week"] == "2026-W28"
        # already scored -> no second registration
        assert maybe_score_week(now=now) is None
        assert len(pushed) == 1

    def test_no_grounding_week_registers_no_score(self, monkeypatch):
        pushed = []
        monkeypatch.setattr(
            scoring.obs, "record_score", lambda **kwargs: pushed.append(kwargs)
        )
        now = datetime(2026, 7, 8, 15, 0, tzinfo=timezone.utc)
        self._journal_session(now - timedelta(hours=3), runtime_sec=4000, grounding=None)
        assert maybe_score_week(now=now) is not None
        assert pushed == []

    def test_runs_at_most_once_per_week(self):
        now = datetime(2026, 7, 8, 15, 0, tzinfo=timezone.utc)
        self._journal_session(now - timedelta(hours=3), runtime_sec=4000)
        assert maybe_score_week(now=now) is not None
        self._journal_session(now - timedelta(hours=1), runtime_sec=4000)
        assert maybe_score_week(now=now) is None  # this week already has a session

    def test_other_weeks_records_are_excluded(self):
        now = datetime(2026, 7, 8, 15, 0, tzinfo=timezone.utc)
        self._journal_session(now - timedelta(days=10), runtime_sec=90_000)  # last week
        assert maybe_score_week(now=now) is None

    def test_report_aggregates_metrics(self):
        now = datetime(2026, 7, 8, 15, 0, tzinfo=timezone.utc)
        self._journal_session(
            now - timedelta(hours=3), runtime_sec=4000,
            activations=[
                {"strategy": "momentum", "regime": "quiet",
                 "started_at": "2026-07-08T10:00:00+00:00",
                 "ended_at": "2026-07-08T11:00:00+00:00",
                 "decisions": 2, "active_decisions": 0, "alert_decisions": 2,
                 "effective": False},
                {"strategy": "momentum", "regime": "bullish_trend",
                 "started_at": "2026-07-08T11:00:00+00:00",
                 "ended_at": "2026-07-08T12:00:00+00:00",
                 "decisions": 2, "active_decisions": 1, "alert_decisions": 1,
                 "effective": True},
            ],
        )
        report = maybe_score_week(now=now)
        assert report["grounding"]["mean_score"] == pytest.approx(0.9)
        assert report["tactics_validation"]["rejection_rate"] == pytest.approx(1 / 3)
        assert report["tools"]["error_rate"] == pytest.approx(0.1)
        assert report["tools"]["quote_warning_rate"] == pytest.approx(0.2)
        assert report["decision_quality"]["active_rate"] == pytest.approx(0.4)
        momentum = report["automatic"]["strategies"]["momentum"]
        assert momentum["activations"] == 2
        assert momentum["alert_only"] == 1
        assert momentum["effectiveness"] == pytest.approx(0.5)

    def test_live_session_counts_toward_runtime(self):
        now = datetime(2026, 7, 8, 15, 0, tzinfo=timezone.utc)
        state = AppState()
        begin_session(state, "momentum", ["SPY"])
        state.scorecard.started_at = (now - timedelta(hours=2)).isoformat()
        report = maybe_score_week(state, tracker=None, now=now)
        assert report is not None
        assert report["sessions"] == 1
        assert report["total_runtime_sec"] == pytest.approx(7200, abs=1)

    def test_end_session_journals_and_clears_scorecard(self):
        state = AppState()
        begin_session(state, "momentum", ["SPY"])
        end_session(state, tracker=None)
        assert state.scorecard is None
        records = scoring._read_journal()
        assert len(records) == 1
        assert records[0]["mode"] == "momentum"
