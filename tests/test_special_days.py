"""A tuning job's special sessions: which, what was expected, what came
(simlab/special_days.py)."""
import json
from datetime import date, datetime, timezone
from types import SimpleNamespace

import pytest

from agent_stonks import event_days
from simlab import special_days as sd


@pytest.fixture(autouse=True)
def _caches_in_tmp(monkeypatch, tmp_path):
    monkeypatch.setattr(sd, "RESULTS_DIR", tmp_path / "earnings_results")
    monkeypatch.setattr(sd, "RELEASES_DIR", tmp_path / "releases")


def daily_bar(day: str) -> dict:
    """A stored daily bar, stamped at midnight ET in UTC as the store keeps them."""
    return {"t": f"{day}T04:00:00Z", "o": 1, "h": 1, "l": 1, "c": 1, "v": 1}


@pytest.fixture
def week(monkeypatch):
    """INTC's week of 2026-10-19: a jobs report on the calendar, a report after
    the close on the 21st, a geopolitical shock flagged on the 23rd."""
    calendar = [
        {"date": "2026-10-20", "category": event_days.NFP, "scope": "ALL",
         "event": "BLS Employment Situation release, 08:30 ET", "source": "bls"},
        {"date": "2026-10-20", "category": event_days.GEO, "scope": "MSFT",
         "event": "someone else's", "source": "x"},
    ]
    monkeypatch.setattr(event_days, "load_calendar", lambda *a, **k: calendar)
    monkeypatch.setattr(sd, "earnings_dates", lambda symbol: ["2026-10-21T16:00:00-04:00"])
    monkeypatch.setattr(sd.sim_data, "load_daily_bars", lambda symbol, feed: [
        daily_bar(d) for d in ("2026-10-16", "2026-10-19", "2026-10-20", "2026-10-21")
    ])
    event_days.record_verdict(
        "INTC", "geo", "strikes overnight",
        datetime(2026, 10, 23, 8, 0, tzinfo=event_days.MARKET_TZ),
    )
    # Made after the opening window: a replay could not have read it.
    event_days.record_verdict(
        "INTC", "market", "late news",
        datetime(2026, 10, 22, 10, 0, tzinfo=event_days.MARKET_TZ),
    )
    days = ["2026-10-19", "2026-10-20", "2026-10-21", "2026-10-22", "2026-10-23"]
    return {
        "base": {"ticker": "INTC"},
        "datasets": [
            {"name": "w1", "days": days, "feed": "live"},
            {"name": "w2", "days": days[2:], "feed": "live"},
        ],
    }


class TestWhichSessions:
    def test_the_sessions_a_replays_days_off_check_would_find(self, week):
        rows = sd.special_days(week)
        assert [(r["day"], r["category"]) for r in rows] == [
            ("2026-10-20", event_days.NFP),
            ("2026-10-22", event_days.EARNINGS),
            ("2026-10-23", event_days.GEO),
        ]
        earnings = rows[1]
        assert earnings["report"] == "2026-10-21 16:00 ET"
        assert earnings["what"] == "report 2026-10-21 16:00 ET"
        assert rows[2]["what"] == "strikes overnight"

    def test_a_session_in_two_datasets_is_one_row_naming_both(self, week):
        rows = {r["day"]: r for r in sd.special_days(week)}
        assert rows["2026-10-20"]["datasets"] == ["w1"]
        assert rows["2026-10-22"]["datasets"] == ["w1", "w2"]


class TestEarnings:
    ROWS = [
        {"time": "2026-10-21T16:00:00-04:00", "estimate": 0.4, "reported": 0.5,
         "surprise_pct": 25.0},
        {"time": "2026-07-23T16:00:00-04:00", "estimate": 0.2, "reported": 0.4,
         "surprise_pct": 100.0},
    ]

    def test_a_report_is_matched_on_its_et_date_and_kept(self):
        calls = []

        def fetch(symbol):
            calls.append(symbol)
            return self.ROWS

        found = sd.earnings_results("intc", ["2026-10-21 16:00 ET"], fetch=fetch)
        assert found["2026-10-21 16:00 ET"]["reported"] == 0.5
        assert sd.earnings_results("INTC", ["2026-10-21 16:00 ET"], fetch=fetch) == found
        assert calls == ["INTC"]

    def test_a_copy_fetched_before_the_report_is_fetched_again(self):
        before = [{**self.ROWS[0], "reported": None, "surprise_pct": None}, self.ROWS[1]]
        path = sd.RESULTS_DIR / "INTC.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({
            "symbol": "INTC", "rows": before,
            "fetched_at": datetime(2026, 10, 2, tzinfo=timezone.utc).isoformat(),
        }))
        found = sd.earnings_results("INTC", ["2026-10-21 16:00 ET"], fetch=lambda s: self.ROWS)
        assert found["2026-10-21 16:00 ET"]["reported"] == 0.5

    def test_a_report_yahoo_no_longer_lists_says_so(self):
        """SimLab keeps the dates it first read; one that later moved is not a report."""
        found = sd.earnings_results("INTC", ["2026-10-22 16:00 ET"], fetch=lambda s: self.ROWS)
        assert found == {"2026-10-22 16:00 ET": {"unlisted": True}}
        row = {"category": event_days.EARNINGS, "day": "2026-10-23", "report": "2026-10-22 16:00 ET"}
        assert sd.outcome_cells(row, found)["actual"] == "Yahoo lists no report then"

    def test_a_failed_fetch_is_no_numbers_not_an_error(self):
        def fetch(symbol):
            raise OSError("offline")

        assert sd.earnings_results("INTC", ["2026-10-21 16:00 ET"], fetch=fetch) == {}

    def test_the_cells_are_the_estimate_the_report_and_the_surprise(self):
        row = {"category": event_days.EARNINGS, "day": "2026-10-22", "report": "2026-10-21 16:00 ET"}
        found = sd.earnings_results("INTC", [row["report"]], fetch=lambda s: self.ROWS)
        assert sd.outcome_cells(row, found) == {
            "measure": "EPS", "expected": "0.40", "actual": "0.50", "surprise": "+25.0%",
            "more": "",
        }


def release(category="cpi", day="2026-10-14", **fields) -> dict:
    record = {
        "category": category, "day": day, "period": "September 2026",
        "measures": [
            {"name": "CPI m/m", "unit": "%", "expected": 0.2, "actual": 0.4},
            {"name": "CPI y/y", "unit": "%", "expected": 3.6, "actual": 3.7},
        ],
        "sources": ["https://example.com/cpi"], "note": "Dow Jones poll", **fields,
    }
    path = sd.release_path(category, day)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))
    return record


class TestReleases:
    def test_the_first_measure_fills_the_columns_and_the_rest_go_under_also(self):
        release()
        cells = sd.outcome_cells({"category": "cpi", "day": "2026-10-14"})
        assert cells == {
            "measure": "CPI m/m", "expected": "0.2%", "actual": "0.4%", "surprise": "+0.2 pp",
            "more": "CPI y/y 3.7% vs 3.6% expected (+0.1 pp)",
        }

    def test_payrolls_are_in_thousands_and_a_missing_figure_is_a_dash(self):
        release("nfp", "2026-10-02", measures=[
            {"name": "Nonfarm payrolls", "unit": "k", "expected": 150, "actual": 1254},
            {"name": "Unemployment rate", "unit": "%", "expected": None, "actual": 4.3},
        ])
        cells = sd.outcome_cells({"category": "nfp", "day": "2026-10-02"})
        assert cells["actual"] == "1,254k"
        assert cells["surprise"] == "+1,104k"
        assert cells["more"] == "Unemployment rate 4.3% vs — expected (—)"

    def test_a_release_not_looked_up_is_blank_and_listed_as_missing(self):
        rows = [{"category": "cpi", "day": "2026-10-14"}, {"category": "cpi", "day": "2026-10-14"},
                {"category": "nfp", "day": "2026-10-02"}, {"category": "geo", "day": "2026-10-03"}]
        assert sd.outcome_cells(rows[0])["expected"] == ""
        assert sd.missing_releases(rows) == [("cpi", "2026-10-14"), ("nfp", "2026-10-02")]
        release()
        assert sd.missing_releases(rows) == [("nfp", "2026-10-02")]

    def test_a_shock_expected_nothing(self):
        assert sd.outcome_cells({"category": "geo", "day": "2026-10-03"}) == {
            "measure": "", "expected": "", "actual": "", "surprise": "", "more": "",
        }


class FakeMessages:
    """`client.beta.messages` answering with `responses` in turn."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def response(stop_reason, text=None):
    content = [SimpleNamespace(type="server_tool_use")]
    if text is not None:
        content.append(SimpleNamespace(type="text", text=text))
    return SimpleNamespace(stop_reason=stop_reason, content=content, model="claude-opus-5-5",
                           usage=None)


class TestLookup:
    ANSWER = {
        "period": "September 2026",
        "measures": [{"name": "CPI m/m", "unit": "%", "expected": 0.3, "actual": 0.4}],
        "sources": ["https://example.com"], "note": "",
    }

    def client(self, *responses):
        messages = FakeMessages(responses)
        return SimpleNamespace(beta=SimpleNamespace(messages=messages)), messages

    def test_the_answer_is_kept_and_a_paused_search_resumed(self):
        client, messages = self.client(
            response("pause_turn"), response("end_turn", json.dumps(self.ANSWER))
        )
        record = sd.lookup_release("cpi", "2026-10-14", client=client)
        assert record["measures"] == self.ANSWER["measures"]
        assert sd.release_outcome("cpi", "2026-10-14")["period"] == "September 2026"
        first, second = messages.calls
        assert first["model"] == sd.LOOKUP_MODEL
        assert first["tools"][0]["type"] == "web_search_20260209"
        assert "2026-10-14" in first["messages"][0]["content"]
        # The paused turn goes back as the assistant's, to be carried on.
        assert second["messages"][-1]["role"] == "assistant"

    def test_a_refusal_keeps_nothing(self):
        client, _ = self.client(response("refusal"))
        with pytest.raises(RuntimeError):
            sd.lookup_release("nfp", "2026-10-02", client=client)
        assert sd.release_outcome("nfp", "2026-10-02") is None

    def test_the_question_names_the_release_and_its_measures(self):
        question = sd._release_question("nfp", "2026-10-02")
        assert "Employment Situation" in question and "2026-10-02" in question
        assert "Nonfarm payrolls (thousands)" in question


# --- the switches on a job's results (simlab/app.py) ---------------------------


DAYS = ["2026-06-15", "2026-06-16"]


def cell(buy: float, profits: "list[float]") -> dict:
    return {
        "overrides": {"buy_k": buy}, "profit": sum(profits), "return_pct": sum(profits) / 100.0,
        "trades": 2, "sells": 2, "days": 2, "days_traded": 2,
        "days_up": sum(1 for p in profits if p > 0), "days_down": sum(1 for p in profits if p < 0),
        "worst_day": min(profits), "best_day": max(profits),
        "daily": dict(zip(DAYS, profits)), "no_forecast_days": [], "sat_out_days": [],
        "error": None,
    }


def job(skip_events=()) -> dict:
    from dataclasses import asdict

    from agent_stonks.apple_trader import AppleTraderConfig
    from simlab import tuning as tu

    base = {**asdict(AppleTraderConfig(ticker="AAPL")), "skip_events": list(skip_events)}
    record = {
        "job_id": "j", "status": tu.FINISHED,
        "spec": {
            "base": base, "axes": [{"name": "buy_k", "values": [0.3, 0.5]}],
            "datasets": [{"name": "w1", "days": DAYS, "feed": "live", "symbols": ["AAPL"]}],
            "starting_cash": 10_000.0, "rule": "max", "min_traded_share": 0.0,
        },
        # 0.3 made its money on the CPI morning; 0.5 on the ordinary day.
        "cells": {"w1": [cell(0.3, [900.0, -100.0]), cell(0.5, [-50.0, 400.0])]},
        "baseline": {"w1": cell(0.4, [100.0, 0.0])},
        "best": None,
    }
    return tu.refresh_pick(record)


def _switches_app(record: dict) -> None:
    import json

    import streamlit as st

    from simlab import app as sim_app

    read = sim_app._render_tuning_special_days(record)
    st.text(json.dumps({
        "pick": read["best"]["overrides"], "skip_events": read["spec"]["base"]["skip_events"],
    }))


class TestTuningSwitches:
    ROWS = [{"day": "2026-06-15", "category": "cpi", "what": "BLS CPI release",
             "source": "calendar", "datasets": ["w1"], "report": None}]

    def app(self, monkeypatch, record):
        from streamlit.testing.v1 import AppTest

        from simlab import app as sim_app

        monkeypatch.setattr(sim_app, "_tuning_special_rows", lambda spec_json: (self.ROWS, {}))
        app = AppTest.from_function(_switches_app, args=(record,), default_timeout=30)
        app.run()
        assert not app.exception, app.exception
        return app

    def read(self, app) -> dict:
        return json.loads(app.text[-1].value)

    def test_a_kind_unticked_leaves_its_sessions_out_of_the_pick_and_into_the_rule(
        self, monkeypatch
    ):
        app = self.app(monkeypatch, job())
        assert self.read(app) == {"pick": {"buy_k": 0.3}, "skip_events": []}
        table = app.dataframe[0].value
        assert list(table["kind"]) == ["CPI release"] and list(table["shown"]) == [900.0]
        assert list(table["base"]) == [100.0] and list(table["status"]) == ["counted"]
        app.checkbox(key="tune_special_j_cpi").uncheck().run()
        assert self.read(app) == {"pick": {"buy_k": 0.5}, "skip_events": ["cpi"]}
        assert list(app.dataframe[0].value["status"]) == ["left out"]
        assert any("Leaving out 1 special session" in c.value for c in app.caption)

    def test_a_kind_the_base_sat_out_cannot_be_put_back(self, monkeypatch):
        app = self.app(monkeypatch, job(skip_events=["cpi"]))
        [box] = app.checkbox
        assert box.disabled and box.value is False
        assert list(app.dataframe[0].value["status"]) == ["sat out by the base"]
        assert self.read(app)["skip_events"] == ["cpi"]
