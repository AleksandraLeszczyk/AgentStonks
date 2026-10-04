"""The days-off check (agent_stonks/event_days.py): which sessions Apple Trader
sits out, and where each reason comes from.

Three sources, each pinned on its own: the calendar file (read as shipped --
its rows are the notebook's, with sources), Yahoo's earnings dates (stubbed),
and the briefing verdicts (kept in a temporary file by the suite's
`_days_off_hermetic` fixture). Then `check`, which puts them together.
"""

import json
from datetime import date, datetime, timezone

import pandas as pd
import pytest

from agent_stonks import event_days as E
from agent_stonks.market_hours import MARKET_TZ

ALL = E.CATEGORIES


def et(day: str, hhmm: str) -> datetime:
    return pd.Timestamp(f"{day} {hhmm}").tz_localize(MARKET_TZ).to_pydatetime()


class TestCategories:
    def test_categories_come_back_in_one_order(self):
        assert E.normalise(["geo", "earnings", "cpi", "geo"]) == ("earnings", "cpi", "geo")

    def test_an_unknown_category_is_refused(self):
        with pytest.raises(ValueError, match="fomc"):
            E.normalise(["cpi", "fomc"])


class TestCalendar:
    def test_a_cpi_and_a_jobs_report_day_are_on_it(self):
        cpi = E.calendar_events("AAPL", date(2026, 9, 11), ALL)
        nfp = E.calendar_events("INTC", date(2026, 8, 7), ALL)
        assert [e.category for e in cpi] == ["cpi"] and [e.category for e in nfp] == ["nfp"]

    def test_bls_dates_after_the_notebooks_calendar_are_on_it(self):
        assert [e.category for e in E.calendar_events("AAPL", date(2026, 10, 14), ALL)] == ["cpi"]
        assert [e.category for e in E.calendar_events("AAPL", date(2026, 12, 4), ALL)] == ["nfp"]

    def test_a_shock_that_broke_during_the_session_does_not_count(self):
        """Iran's missiles on 1 Oct 2024 came mid-session: no trader could have
        sat it out, and a replay that did would be peeking."""
        assert E.calendar_events("AAPL", date(2024, 10, 1), ALL) == []

    def test_a_symbols_own_rows_apply_to_it_only(self):
        """Apple's results, and the smartphone tariff exemption, are AAPL's."""
        assert [e.category for e in E.calendar_events("AAPL", date(2026, 7, 31), ALL)] == ["earnings"]
        assert E.calendar_events("MU", date(2026, 7, 31), ALL) == []
        assert E.calendar_events("AAPL", date(2025, 4, 14), ALL)
        assert E.calendar_events("INTC", date(2025, 4, 14), ALL) == []

    def test_only_the_categories_asked_for(self):
        assert E.calendar_events("AAPL", date(2026, 9, 11), ("nfp", "geo")) == []

    def test_a_pandas_timestamp_is_the_same_day(self):
        assert E.calendar_events("AAPL", pd.Timestamp("2026-09-11"), ALL)

    def test_a_schedule_that_has_run_out_is_named(self):
        assert E.schedule_notes(date(2026, 12, 15), ALL) == []
        notes = E.schedule_notes(date(2027, 2, 1), ALL)
        assert len(notes) == 2 and "2026-12-10" in notes[0] and "2026-12-04" in notes[1]
        assert E.schedule_notes(date(2027, 2, 1), ("earnings", "geo")) == []


class TestEarnings:
    STAMPS = [pd.Timestamp("2026-10-29 16:30", tz=MARKET_TZ)]

    def test_the_first_session_after_a_report_after_the_close(self):
        event = E.earnings_event("AAPL", date(2026, 10, 30), self.STAMPS, date(2026, 10, 29))
        assert event.category == "earnings" and "2026-10-29 16:30" in event.what

    def test_the_report_day_itself_trades(self):
        """The position is flat before 16:30, so the report is the next day's news."""
        assert E.earnings_event("AAPL", date(2026, 10, 29), self.STAMPS, date(2026, 10, 28)) is None

    def test_yahoos_after_the_close_stamp_is_the_next_sessions_news(self):
        """Yahoo stamps an after-the-close report 16:00 sharp (AAPL's of
        2026-07-30, whose reaction day the notebook marks as 31 Jul)."""
        stamps = [pd.Timestamp("2026-07-30 16:00", tz=MARKET_TZ)]
        assert E.earnings_event("AAPL", date(2026, 7, 30), stamps, date(2026, 7, 29)) is None
        assert E.earnings_event("AAPL", date(2026, 7, 31), stamps, date(2026, 7, 30)) is not None

    def test_a_holiday_in_between_is_read_off_the_daily_bars(self, monkeypatch):
        """Friday's report is Tuesday's news when Monday is a holiday; the
        weekday alone would look for it after Monday's close."""
        friday = [pd.Timestamp("2026-09-04 16:30", tz=MARKET_TZ)]
        bars = [{"t": d, "o": 1, "h": 1, "l": 1, "c": 1} for d in ("2026-09-03", "2026-09-04")]
        monkeypatch.setattr(E.historical, "fetch_daily_ohlc_bars", lambda *a, **k: bars)
        monkeypatch.setattr(E, "fetch_earnings_stamps", lambda *a, **k: friday)
        found = E.check("AAPL", date(2026, 9, 8), ("earnings",))
        assert found.categories == ["earnings"]

    def test_the_previous_session_is_the_checked_days_eve(self, monkeypatch):
        """A chart of a past session must not measure from yesterday."""
        bars = [{"t": d, "o": 1, "h": 1, "l": 1, "c": 1}
                for d in ("2026-09-03", "2026-09-04", "2026-10-02")]
        monkeypatch.setattr(E.historical, "fetch_daily_ohlc_bars", lambda *a, **k: bars)
        assert E.previous_session("AAPL", date(2026, 9, 8)) == date(2026, 9, 4)

    def test_unreadable_earnings_dates_are_a_note_not_a_skip(self, monkeypatch):
        monkeypatch.setattr(E, "fetch_earnings_stamps", lambda *a, **k: None)
        found = E.check("AAPL", date(2026, 9, 8), ("earnings",))
        assert found.events == [] and "could not be read" in found.notes[0]


class TestVerdicts:
    def test_a_verdict_is_kept_for_the_day_it_was_made(self):
        assert E.record_verdict("aapl", "geo", "strikes on Iran", et("2026-10-05", "08:10"))
        kept = E.recorded_verdict("AAPL", date(2026, 10, 5))
        assert kept["shock"] == "geo" and kept["reason"] == "strikes on Iran"

    def test_a_later_morning_briefing_replaces_an_earlier_one(self):
        """The 9:20 briefing has read news the 8:00 one had not."""
        E.record_verdict("AAPL", "none", "", et("2026-10-05", "08:00"))
        assert E.record_verdict("AAPL", "market", "crash", et("2026-10-05", "09:20"))
        assert E.recorded_verdict("AAPL", date(2026, 10, 5))["shock"] == "market"

    def test_once_the_session_is_decided_the_verdict_stands(self):
        """A stream restarted at 11:00 must not flip what 9:35 decided."""
        E.record_verdict("AAPL", "none", "", et("2026-10-05", "09:10"))
        assert not E.record_verdict("AAPL", "geo", "intraday", et("2026-10-05", "11:00"))
        assert E.recorded_verdict("AAPL", date(2026, 10, 5))["shock"] == "none"

    def test_a_late_first_verdict_is_kept_but_a_replay_does_not_read_it(self):
        E.record_verdict("AAPL", "geo", "late", et("2026-10-05", "10:30"))
        assert E.recorded_verdict("AAPL", date(2026, 10, 5))["shock"] == "geo"
        assert E.recorded_verdict("AAPL", date(2026, 10, 5), before_cutoff=True) is None

    def test_an_unknown_flag_is_read_as_none(self):
        E.record_verdict("AAPL", "weird", "", et("2026-10-05", "08:00"))
        assert E.recorded_verdict("AAPL", date(2026, 10, 5))["shock"] == "none"

    def test_symbols_and_days_are_kept_apart(self):
        E.record_verdict("AAPL", "geo", "x", et("2026-10-05", "08:00"))
        assert E.recorded_verdict("MU", date(2026, 10, 5)) is None
        assert E.recorded_verdict("AAPL", date(2026, 10, 6)) is None


class TestCheck:
    DAY = date(2026, 10, 5)  # an ordinary Monday, after the calendar's shock rows end

    def test_a_flagged_shock_is_a_reason(self):
        E.record_verdict("AAPL", "geo", "strikes on Iran", et("2026-10-05", "08:00"))
        found = E.check("AAPL", self.DAY, ALL)
        assert found.categories == ["geo"]
        assert found.events[0].source.startswith("pre-market briefing")
        assert "strikes on Iran" in E.sit_out_phrase(found.events)

    def test_a_shock_category_not_asked_for_is_no_reason(self):
        E.record_verdict("AAPL", "geo", "x", et("2026-10-05", "08:00"))
        assert E.check("AAPL", self.DAY, ("cpi", "market_shock")).events == []

    def test_a_none_verdict_is_an_ordinary_day_with_nothing_to_say(self):
        E.record_verdict("AAPL", "none", "", et("2026-10-05", "08:00"))
        found = E.check("AAPL", self.DAY, ALL)
        assert found.events == [] and found.notes == [] and not found.waiting

    def test_a_briefing_still_being_written_is_waited_for(self):
        assert E.check("AAPL", self.DAY, ALL, briefing_pending=True).waiting

    def test_no_briefing_at_all_is_a_note(self):
        found = E.check("AAPL", self.DAY, ALL)
        assert not found.waiting and "no pre-market briefing" in found.notes[0]

    def test_without_the_shock_categories_no_briefing_is_needed(self):
        found = E.check("AAPL", self.DAY, ("earnings", "cpi", "nfp"), briefing_pending=True)
        assert not found.waiting and found.notes == []

    def test_several_reasons_are_all_named_in_category_order(self):
        E.record_verdict("AAPL", "market", "selloff", et("2026-10-14", "08:00"))
        found = E.check("AAPL", date(2026, 10, 14), ALL)
        assert found.categories == ["cpi", "market_shock"]

    def test_nothing_asked_for_checks_nothing(self, monkeypatch):
        monkeypatch.setattr(E, "load_calendar", lambda *a, **k: pytest.fail("read"))
        assert E.check("AAPL", date(2026, 9, 11), ()).events == []


class TestBriefingFlag:
    def test_the_schema_the_model_sees_asks_for_the_flag_with_no_defaults(self):
        from openai.lib._pydantic import to_strict_json_schema

        from agent_stonks.premarket import PremarketBriefing

        schema = to_strict_json_schema(PremarketBriefing)
        assert {"shock", "shock_reason"} <= set(schema["required"])
        assert "default" not in json.dumps(schema)

    def test_a_briefing_without_the_flag_is_an_ordinary_day(self):
        from agent_stonks.premarket import PremarketBriefing

        briefing = PremarketBriefing(
            overall_bias="neutral", confidence="low", summary="s", catalysts=[],
            technical_levels=[], risk_factors=[], macro_context="m", key_levels_to_watch=[],
        )
        assert briefing.shock == "none"

    @pytest.mark.parametrize("phase,kept", [
        ("premarket", True), ("open", True), ("after_hours", False), ("weekend", False),
    ])
    def test_only_a_briefing_about_todays_session_is_kept(self, phase, kept, monkeypatch):
        from agent_stonks import premarket
        from agent_stonks.premarket import PremarketBriefing

        briefing = PremarketBriefing(
            overall_bias="bearish", confidence="high", summary="s", catalysts=[],
            technical_levels=[], risk_factors=[], macro_context="m", key_levels_to_watch=[],
            shock="geo", shock_reason="strikes",
        )
        seen = []
        monkeypatch.setattr(premarket.event_days, "record_verdict", lambda *a, **k: seen.append(a))
        premarket._record_shock_verdict("AAPL", briefing, phase)
        assert bool(seen) == kept
        if kept:
            assert seen[0][:3] == ("AAPL", "geo", "strikes")


class TestReplay:
    """Inside a simulation the check reads SimLab's kept earnings dates and only
    the verdicts recorded before the replayed session's opening window closed."""

    def test_the_replay_reads_only_what_that_morning_could_have(self, monkeypatch, tmp_path):
        from simlab import session_context
        from simlab.market import SimMarket
        from simlab.patches import simulation_context

        E.record_verdict("AAPL", "geo", "late", et("2026-10-05", "10:30"))
        monkeypatch.setattr(session_context, "earnings_dates",
                            lambda ticker: ["2026-10-02T16:30:00-04:00"])
        with simulation_context(SimMarket([], [])):
            assert E.briefing_verdict("AAPL", date(2026, 10, 5)) is None
            assert E.fetch_earnings_stamps("AAPL", date(2026, 10, 5)) == ["2026-10-02T16:30:00-04:00"]
        assert E.briefing_verdict("AAPL", date(2026, 10, 5))["shock"] == "geo"
