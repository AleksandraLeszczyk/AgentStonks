"""Orchestra's 09:34 candidate selection: the rules, and the facts they read."""
from datetime import date, datetime, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from agent_stonks import candidates as cd
from agent_stonks.candidates import PairFacts, SelectionRules, select_candidates


def facts(key="AAPL:dayrange", target_k=0.5, adr=4.0, price=200.0, **kwargs):
    ticker = key.partition(":")[0]
    base = dict(
        key=key, label=key, ticker=ticker, target_k=target_k,
        prev_close=price, adr=adr, open=price, last=price,
        bias="neutral", confidence="medium",
    )
    base.update(kwargs)
    return PairFacts(**base)


def picked(result):
    return [c.facts.key for c in result if c.selected]


class TestTheRules:
    def test_ranks_by_what_a_target_pays_on_the_price_and_keeps_the_best(self):
        result = select_candidates(
            [
                facts("AAPL:dayrange", target_k=0.5, adr=4.0, price=200.0),   # 1.00%
                facts("INTC:dayrange", target_k=0.6, adr=1.0, price=25.0),    # 2.40%
                facts("BE:highlow", target_k=0.1, adr=6.0, price=40.0),       # 1.50%
                facts("MU:highlow", target_k=0.2, adr=3.0, price=150.0),      # 0.40%
            ],
            SelectionRules(max_pairs=2),
        )
        assert picked(result) == ["INTC:dayrange", "BE:highlow"]
        # In the order given, each with its rank.
        assert [c.facts.key for c in result] == [
            "AAPL:dayrange", "INTC:dayrange", "BE:highlow", "MU:highlow",
        ]
        assert result[0].reason == "ranked #3 of 4; keeps the best 2"
        assert result[1].reason.startswith("#1 of 4 by target (2.40%")

    def test_ties_keep_orchestras_order(self):
        result = select_candidates(
            [facts("AAPL:highlow"), facts("AAPL:dayrange")], SelectionRules(max_pairs=1),
        )
        assert picked(result) == ["AAPL:highlow"]

    def test_zero_keeps_every_eligible_pair(self):
        result = select_candidates(
            [facts("AAPL:dayrange"), facts("INTC:dayrange"), facts("MU:highlow")],
            SelectionRules(max_pairs=0),
        )
        assert len(picked(result)) == 3

    def test_earnings_leave_a_pair_out_unless_switched_off(self):
        f = [facts("AAPL:dayrange", earnings="2026-10-29 16:30 ET"), facts("INTC:dayrange")]
        assert picked(select_candidates(f, SelectionRules())) == ["INTC:dayrange"]
        assert select_candidates(f, SelectionRules())[0].reason == "earnings 2026-10-29 16:30 ET"
        assert len(picked(select_candidates(f, SelectionRules(exclude_earnings=False)))) == 2

    @pytest.mark.parametrize("rule, confidence, out", [
        ("high", "high", True),
        ("high", "medium", False),
        ("any", "low", True),
        ("off", "high", False),
    ])
    def test_a_bearish_briefing(self, rule, confidence, out):
        f = [facts("AAPL:dayrange", bias="bearish", confidence=confidence)]
        result = select_candidates(f, SelectionRules(exclude_bearish=rule))
        assert (not result[0].selected) == out

    def test_a_pair_without_a_briefing_is_not_left_out_for_it(self):
        f = [facts("AAPL:dayrange", bias=None, confidence=None, briefing_note="no briefing")]
        assert picked(select_candidates(f, SelectionRules(exclude_bearish="any"))) == ["AAPL:dayrange"]

    def test_the_gap_limit_reads_both_ways(self):
        f = [
            facts("AAPL:dayrange", open=206.0),   # +1.5 ADR
            facts("INTC:dayrange", open=194.0),   # -1.5 ADR
            facts("MU:highlow", open=202.0),      # +0.5 ADR
        ]
        result = select_candidates(f, SelectionRules(max_gap_adr=1.0, max_pairs=0))
        assert picked(result) == ["MU:highlow"]
        assert "opened +1.50 ADR" in result[0].reason

    def test_no_daily_history_is_no_candidate(self):
        result = select_candidates([facts(adr=None)], SelectionRules())
        assert not result[0].selected and "no daily history" in result[0].reason

    def test_rules_refuse_nonsense(self):
        with pytest.raises(ValueError):
            SelectionRules(max_pairs=-1)
        with pytest.raises(ValueError):
            SelectionRules(exclude_bearish="sometimes")
        with pytest.raises(ValueError):
            SelectionRules(max_gap_adr=-0.5)

    def test_signature_names_each_rule_that_is_on(self):
        assert cd.rules_signature(None) == ""
        assert cd.rules_signature(SelectionRules()) == ",select=3,earn,bear=high"
        sig = cd.rules_signature(SelectionRules(
            max_pairs=0, exclude_earnings=False, exclude_bearish="off", max_gap_adr=1.0,
            briefing_provider="gemini", briefing_model="gemini-3.5-flash",
        ))
        assert sig == ",select=all,gap<=1A,brief=gemini/gemini-3.5-flash"


class TestTheFacts:
    def test_daily_stats_take_the_last_14_sessions(self):
        bars = [{"t": f"2026-09-{d:02d}", "h": 10.0 + (d % 2), "l": 8.0, "c": 9.0} for d in range(1, 21)]
        close, adr, prev_day = cd.daily_stats(bars)
        assert close == 9.0 and prev_day == date(2026, 9, 20)
        assert adr == pytest.approx(sum(10.0 + (d % 2) - 8.0 for d in range(7, 21)) / 14)

    @pytest.mark.parametrize("stamp, counts", [
        ("2026-09-14 16:30-04:00", True),    # after the previous close
        ("2026-09-15 07:00-04:00", True),    # before today's open
        ("2026-09-15 16:30-04:00", False),   # after today's close: flat by then
        ("2026-09-14 08:00-04:00", False),   # yesterday's morning: already traded
        # Yahoo's stamp for an after-the-close report is 16:00 sharp: the next
        # session's news, not the report day's.
        ("2026-09-14 16:00-04:00", True),
        ("2026-09-15 16:00-04:00", False),
    ])
    def test_the_earnings_window(self, stamp, counts):
        found = cd.earnings_in_window([pd.Timestamp(stamp)], date(2026, 9, 15), date(2026, 9, 14))
        assert (found is not None) == counts

    def test_a_monday_looks_back_to_friday(self):
        found = cd.earnings_in_window([pd.Timestamp("2026-09-11 16:05-04:00")], date(2026, 9, 14))
        assert found == "2026-09-11 16:05 ET"

    def test_gather_reads_the_first_four_minutes_and_the_official_open(self, monkeypatch):
        daily = [{"t": f"2026-09-{d:02d}", "h": 102.0, "l": 98.0, "c": 100.0} for d in range(1, 15)]
        monkeypatch.setattr(cd.historical, "fetch_daily_ohlc_bars", lambda *a, **k: daily)
        monkeypatch.setattr(cd.historical, "fetch_session_open", lambda *a, **k: 101.0)
        index = pd.date_range("2026-09-15 09:30", periods=6, freq="1min", tz="America/New_York")
        frame = pd.DataFrame(
            {"open": [101.5, 102, 103, 104, 105, 106], "close": [102, 103, 104, 103.5, 105, 106]},
            index=index,
        )
        monkeypatch.setattr(cd.momentum_regime, "minute_frame", lambda ss: frame)
        state = SimpleNamespace(sym=lambda t: object())

        class Sources:
            def briefing(self, ticker, day, state):
                return {"bias": "bullish", "confidence": "low"}, ""

            def earnings(self, ticker, day):
                return [pd.Timestamp("2026-09-14 16:30-04:00")]

        config = SimpleNamespace(ticker="AAPL", model_key="dayrange", buy_k=0.55, sell_k=0.05)
        f = cd.gather_facts(config, "AAPL · Day Range", state, date(2026, 9, 15), Sources())
        assert (f.prev_close, f.adr, f.open, f.last) == (100.0, 4.0, 101.0, 103.5)
        assert f.gap_adr == pytest.approx(0.25) and f.move_adr == pytest.approx(0.625)
        assert f.target_pct == pytest.approx(100 * 0.5 * 4.0 / 103.5)
        assert (f.bias, f.confidence) == ("bullish", "low")
        assert f.earnings == "2026-09-14 16:30 ET"


class TestLiveBriefings:
    def _state(self, made, briefings=None, pending=()):
        return SimpleNamespace(
            premarket_briefings=briefings or {}, premarket_pending=list(pending),
            premarket_generated_at=made,
        )

    def test_reads_todays_briefing(self):
        briefing = SimpleNamespace(overall_bias="bearish", confidence="high")
        state = self._state(datetime(2026, 9, 15, 12, 0, tzinfo=timezone.utc), {"AAPL": briefing})
        got, note = cd.LiveSources().briefing("AAPL", date(2026, 9, 15), state)
        assert got == {"bias": "bearish", "confidence": "high"} and note == ""

    def test_yesterdays_briefing_is_none(self):
        briefing = SimpleNamespace(overall_bias="bearish", confidence="high")
        state = self._state(datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc), {"AAPL": briefing})
        got, note = cd.LiveSources().briefing("AAPL", date(2026, 9, 15), state)
        assert got is None and "2026-09-14" in note

    def test_one_still_being_written_says_so(self):
        state = self._state(None, pending=["AAPL"])
        assert cd.LiveSources().briefing("AAPL", date(2026, 9, 15), state) == (None, "still being written")
