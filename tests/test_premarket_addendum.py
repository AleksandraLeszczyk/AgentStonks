from datetime import datetime, timedelta, timezone

import pytest

from agent_stonks.agent_prompts import premarket_briefing_addendum
from agent_stonks.premarket import (
    Catalyst,
    PremarketBriefing,
    TechnicalLevel,
    briefing_to_prompt_text,
)

NOW = datetime(2026, 9, 11, 18, 0, tzinfo=timezone.utc)


def _briefing(**kw) -> PremarketBriefing:
    base = dict(
        overall_bias="bullish",
        confidence="medium",
        summary="Extended above the consensus mean after a 5.75% move.",
        catalysts=[
            Catalyst(headline="Foldable iPhone launch", impact="positive", relevance="demand"),
            Catalyst(headline="Price increase holds", impact="positive", relevance="margin"),
        ],
        technical_levels=[
            TechnicalLevel(level=336.22, role="session high / resistance", note="n"),
            TechnicalLevel(level=327.5, role="opening print / support", note="n"),
        ],
        risk_factors=["Extended above the 324 consensus mean", "Rate-hike odds rising"],
        macro_context="SPY soft over five days, VIX moderate.",
        key_levels_to_watch=["Acceptance above 336.22", "Loss of 327.50 shifts control"],
    )
    base.update(kw)
    return PremarketBriefing(**base)


class TestBriefingToPromptText:
    def test_renders_prose_not_json(self):
        text = briefing_to_prompt_text(_briefing(), "AAPL")
        assert "{" not in text and '"overall_bias"' not in text
        assert text.startswith("AAPL: bullish bias, medium confidence.")
        assert "Thesis:" in text

    def test_includes_each_section_that_has_content(self):
        text = briefing_to_prompt_text(_briefing(), "AAPL")
        assert "What drove it: Foldable iPhone launch (positive)" in text
        assert "336.22 as session high / resistance" in text
        assert "Risks it named: Extended above the 324 consensus mean" in text
        assert "It said to watch:" in text
        assert "Macro read:" in text

    def test_omits_sections_with_nothing_in_them(self):
        text = briefing_to_prompt_text(
            _briefing(catalysts=[], technical_levels=[], risk_factors=[],
                      key_levels_to_watch=[], macro_context=""),
            "AAPL",
        )
        assert "What drove it" not in text
        assert "Levels it identified" not in text
        assert "Macro read" not in text
        assert "Thesis:" in text

    def test_caps_each_section_so_a_basket_does_not_crowd_out_live_data(self):
        text = briefing_to_prompt_text(
            _briefing(
                catalysts=[Catalyst(headline=f"C{i}", impact="positive", relevance="r")
                           for i in range(9)],
                risk_factors=[f"R{i}" for i in range(9)],
            ),
            "AAPL",
        )
        assert "C2" in text and "C3" not in text   # 3 catalysts
        assert "R2" in text and "R3" not in text   # 3 risks

    def test_formats_levels_without_trailing_zeros(self):
        text = briefing_to_prompt_text(
            _briefing(technical_levels=[TechnicalLevel(level=327.50, role="support", note="")]),
            "AAPL",
        )
        assert "327.5 as support" in text


class TestAddendum:
    def test_no_briefings_means_no_addendum(self):
        # The prompt must be byte-identical to before when nothing was generated.
        assert premarket_briefing_addendum({}, ["AAPL"], now=NOW) == ""
        assert premarket_briefing_addendum(None, ["AAPL"], now=NOW) == ""

    def test_only_includes_tickers_this_agent_trades(self):
        briefings = {"AAPL": _briefing(), "TSLA": _briefing(), "NVDA": _briefing()}
        text = premarket_briefing_addendum(
            briefings, ["AAPL", "NVDA"], generated_at=NOW, now=NOW
        )
        assert "AAPL:" in text and "NVDA:" in text
        assert "TSLA:" not in text

    def test_nothing_when_no_traded_ticker_has_a_briefing(self):
        assert premarket_briefing_addendum({"TSLA": _briefing()}, ["AAPL"], now=NOW) == ""

    def test_frames_the_briefing_as_a_prior_not_an_instruction(self):
        # Without this the agent follows a stale thesis against its own live
        # data, which is worse than having no briefing at all.
        text = premarket_briefing_addendum({"AAPL": _briefing()}, ["AAPL"], now=NOW)
        assert "PRIOR, not as instructions" in text
        assert "YOUR TOOL OUTPUT WINS" in text
        assert "Do NOT trade on the briefing alone" in text

    def test_warns_that_levels_may_already_be_gone(self):
        text = premarket_briefing_addendum({"AAPL": _briefing()}, ["AAPL"], now=NOW)
        assert "already have been taken out" in text

    @pytest.mark.parametrize("delta, expected", [
        (timedelta(0), "just now"),
        (timedelta(minutes=12), "12 minutes ago"),
        (timedelta(minutes=89), "89 minutes ago"),
        (timedelta(hours=6), "6.0 hours ago"),
    ])
    def test_reports_age_at_cycle_time_not_briefing_time(self, delta, expected):
        # A long-running agent must see the gap grow rather than be told a
        # wall-clock time it has to reason about.
        text = premarket_briefing_addendum(
            {"AAPL": _briefing()}, ["AAPL"], generated_at=NOW - delta, now=NOW
        )
        assert expected in text

    def test_a_missing_timestamp_degrades_to_vague_rather_than_wrong(self):
        text = premarket_briefing_addendum({"AAPL": _briefing()}, ["AAPL"], now=NOW)
        assert "earlier" in text

    @pytest.mark.parametrize("phase, marker", [
        ("premarket", "before today's opening bell"),
        ("open", "has moved on since"),
        ("after_hours", "after the close"),
        ("weekend", "market closed"),
        ("", "unrecorded point"),
    ])
    def test_says_which_session_phase_it_was_written_in(self, phase, marker):
        text = premarket_briefing_addendum(
            {"AAPL": _briefing()}, ["AAPL"], generated_at=NOW, phase=phase, now=NOW
        )
        assert marker in text

    def test_renders_every_included_ticker(self):
        briefings = {"AAPL": _briefing(), "TSLA": _briefing(overall_bias="bearish")}
        text = premarket_briefing_addendum(
            briefings, ["AAPL", "TSLA"], generated_at=NOW, now=NOW
        )
        assert "AAPL: bullish bias" in text
        assert "TSLA: bearish bias" in text


class TestCycleWiring:
    def test_the_addendum_reaches_the_system_prompt(self, monkeypatch):
        """The briefing has to land in the system message the model actually sees."""
        from agent_stonks import agent
        from agent_stonks.decisions import DecisionTracker
        from agent_stonks.state import AppState

        app = AppState()
        app.set_symbols(["AAPL"])
        app.premarket_briefings = {"AAPL": _briefing()}
        app.premarket_generated_at = datetime.now(timezone.utc)
        app.premarket_phase = "premarket"

        captured = {}

        class FakeClient:
            class chat:
                class completions:
                    @staticmethod
                    def create(model, messages, tools, tool_choice):
                        captured["system"] = messages[0]["content"]
                        raise RuntimeError("stop after capturing the prompt")

        agent.run_agent_cycle(
            FakeClient(), "m", ["AAPL"], app, DecisionTracker(), max_iters=1
        )

        assert "RESEARCH BRIEFING FOR TODAY" in captured["system"]
        assert "AAPL: bullish bias, medium confidence." in captured["system"]
        assert "YOUR TOOL OUTPUT WINS" in captured["system"]

    def test_prompt_is_unchanged_when_no_briefing_exists(self, monkeypatch):
        from agent_stonks import agent
        from agent_stonks.decisions import DecisionTracker
        from agent_stonks.state import AppState

        app = AppState()
        app.set_symbols(["AAPL"])
        captured = {}

        class FakeClient:
            class chat:
                class completions:
                    @staticmethod
                    def create(model, messages, tools, tool_choice):
                        captured["system"] = messages[0]["content"]
                        raise RuntimeError("stop")

        agent.run_agent_cycle(
            FakeClient(), "m", ["AAPL"], app, DecisionTracker(), max_iters=1
        )
        assert "RESEARCH BRIEFING" not in captured["system"]


class TestExecutionVenueInPrompt:
    """Every personality used to open with "no real orders are ever placed".
    Alpaca routing made that false; an agent that believes its orders are inert
    reasons differently from one spending real money."""

    def test_no_personality_still_claims_orders_are_never_placed(self):
        from agent_stonks import agent_prompts

        for key, entry in agent_prompts.AGENT_PERSONALITIES.items():
            assert "no real orders are ever placed" not in entry["system_prompt"], key

    @pytest.mark.parametrize("mode, marker", [
        ("local", "No order reaches a broker"),
        ("alpaca_paper", "REAL ORDERS sent to a brokerage paper account"),
        ("alpaca_live", "REAL MONEY"),
    ])
    def test_states_the_actual_venue(self, mode, marker):
        from agent_stonks.agent_prompts import execution_venue_addendum

        assert marker in execution_venue_addendum(mode)

    def test_an_unknown_mode_falls_back_to_the_safe_description(self):
        from agent_stonks.agent_prompts import execution_venue_addendum

        assert "LOCAL SIMULATION" in execution_venue_addendum("nonsense")

    def test_live_mode_reaches_the_system_prompt(self):
        from agent_stonks import agent
        from agent_stonks.decisions import DecisionTracker
        from agent_stonks.state import AppState

        app = AppState()
        app.set_symbols(["AAPL"])
        app.trading_mode = "alpaca_live"
        captured = {}

        class FakeClient:
            class chat:
                class completions:
                    @staticmethod
                    def create(model, messages, tools, tool_choice):
                        captured["system"] = messages[0]["content"]
                        raise RuntimeError("stop")

        agent.run_agent_cycle(
            FakeClient(), "m", ["AAPL"], app, DecisionTracker(), max_iters=1
        )
        assert "REAL MONEY" in captured["system"]
        assert "losses are real and permanent" in captured["system"]

    def test_default_state_describes_local_simulation(self):
        from agent_stonks import agent
        from agent_stonks.decisions import DecisionTracker
        from agent_stonks.state import AppState

        app = AppState()
        app.set_symbols(["AAPL"])
        captured = {}

        class FakeClient:
            class chat:
                class completions:
                    @staticmethod
                    def create(model, messages, tools, tool_choice):
                        captured["system"] = messages[0]["content"]
                        raise RuntimeError("stop")

        agent.run_agent_cycle(
            FakeClient(), "m", ["AAPL"], app, DecisionTracker(), max_iters=1
        )
        assert "LOCAL SIMULATION" in captured["system"]
