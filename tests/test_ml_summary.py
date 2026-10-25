"""The ML summary tab's arithmetic (`simlab.ml_summary`): one base series per
pair read off the tuning jobs, the week/month windows, and event days set
against a pair's normal days."""
from __future__ import annotations

from dataclasses import asdict

import pytest

from agent_stonks import event_days
from agent_stonks.apple_trader import AppleTraderConfig
from simlab import ml_summary as ms
from simlab import tuning as tu


def baseline(daily: dict, sat_out=(), no_forecast=()) -> dict:
    profits = list(daily.values())
    return {
        "overrides": {}, "profit": sum(profits), "return_pct": 0.0, "trades": 1, "sells": 1,
        "days": len(profits), "days_traded": len(profits), "days_up": 0, "days_down": 0,
        "worst_day": min(profits), "best_day": max(profits), "daily": dict(daily),
        "no_forecast_days": list(no_forecast), "sat_out_days": list(sat_out), "error": None,
    }


def job(job_id: str, daily: dict, ticker="AMD", model="highlow", created="2026-10-20T10:00",
        skip=(), **cell) -> dict:
    """A finished job whose base replayed `daily` on one dataset."""
    base = asdict(AppleTraderConfig(ticker=ticker, model_key=model, skip_events=tuple(skip)))
    return {
        "job_id": job_id, "status": tu.FINISHED, "created_at": created,
        "spec": {
            "base": base, "axes": [{"name": "buy_k", "values": [0.5]}],
            "datasets": [{"name": "w", "days": sorted(daily), "feed": "live", "symbols": [ticker]}],
            "starting_cash": 100_000.0,
        },
        "baseline": {"w": baseline(daily, **cell)},
        "cells": {"w": []},
    }


WEEK1 = {"2026-10-05": 100.0, "2026-10-06": -50.0, "2026-10-07": 0.0}
WEEK2 = {"2026-10-12": 300.0, "2026-10-13": 100.0, "2026-10-14": -200.0}


class TestBaseSeries:
    def test_a_pair_is_read_from_the_job_reaching_its_latest_session(self):
        old_but_longer = job("a", {**WEEK1, **WEEK2}, created="2026-10-01T00:00")
        newer_but_shorter = job("b", WEEK1, created="2026-10-20T00:00")
        [series] = ms.base_series([newer_but_shorter, old_but_longer])
        assert series.job_id == "a" and series.jobs == 2
        assert series.last_day == "2026-10-14"

    def test_on_a_tie_the_newest_job_wins_and_pairs_stay_apart(self):
        found = ms.base_series([
            job("old", WEEK1, created="2026-10-01T00:00"),
            job("new", WEEK1, created="2026-10-02T00:00"),
            job("mu", WEEK1, ticker="MU"),
        ])
        assert [(s.ticker, s.job_id) for s in found] == [("AMD", "new"), ("MU", "mu")]

    def test_derived_grids_and_unscored_bases_are_skipped(self):
        derived = {**job("d", WEEK1), "status": tu.DERIVED}
        broken = job("x", WEEK1)
        broken["baseline"]["w"]["error"] = "boom"
        assert ms.base_series([derived, broken]) == []


class TestWindows:
    def test_a_week_counts_back_from_the_latest_session_of_any_pair(self):
        series = ms.base_series([job("a", {**WEEK1, **WEEK2}), job("m", WEEK1, ticker="MU")])
        assert ms.window(series, ms.WEEK) == ("2026-10-08", "2026-10-14")
        assert ms.window(series, ms.ALL) == ("2026-10-05", "2026-10-14")
        mu = next(s for s in series if s.ticker == "MU")
        assert ms.pair_metrics(mu, ms.window(series, ms.WEEK)) is None

    def test_every_session_counts_traded_or_not(self):
        [series] = ms.base_series([job("a", {**WEEK1, **WEEK2})])
        week = ms.pair_metrics(series, ms.window([series], ms.WEEK))
        assert (week["sessions"], week["profit"], week["per_session"]) == (3, 200.0, 66.67)
        assert (week["days_up"], week["days_down"]) == (2, 1)
        assert week["return_pct"] == pytest.approx(0.0667)
        whole = ms.pair_metrics(series, ms.window([series], ms.ALL))
        assert whole["sessions"] == 6 and whole["worst_day"] == -200.0

    def test_a_model_is_the_mean_of_its_instruments(self):
        rows = [
            {"model_key": "highlow", "ticker": "AMD", "per_session": 100.0, "profit": 500.0},
            {"model_key": "highlow", "ticker": "MU", "per_session": -20.0, "profit": -100.0},
            {"model_key": "goodmoments", "ticker": "AAPL", "per_session": 50.0, "profit": 250.0},
        ]
        best, second = ms.model_metrics(rows)
        assert (best["model_key"], best["per_session"], best["profit"]) == ("goodmoments", 50.0, 250.0)
        assert (second["per_session"], second["in_profit"], second["best"], second["worst"]) == (
            40.0, 1, "AMD", "MU"
        )

    def test_weeks_start_on_monday(self):
        [series] = ms.base_series([job("a", {**WEEK1, **WEEK2})])
        weeks = ms.weekly([series], ms.window([series], ms.ALL))
        assert [(w["week"], w["per_session"]) for w in weeks] == [
            ("2026-10-05", 16.67), ("2026-10-12", 66.67)
        ]


CPI_ROW = {"date": "2026-10-14", "category": event_days.CPI, "scope": "ALL", "event": "CPI"}


class TestEventsByDay:
    def test_a_market_wide_row_is_every_instruments_and_a_scoped_one_its_own(self):
        rows = [CPI_ROW, {"date": "2026-10-13", "category": "product_event", "scope": "AAPL",
                          "event": "keynote"}]
        found = ms.events_by_day(
            ["AAPL", "AMD"], ["2026-10-13", "2026-10-14"], rows=rows,
            earnings=lambda t: [], verdict=lambda t, d: None,
        )
        assert set(found) == {("AAPL", "2026-10-13"), ("AAPL", "2026-10-14"), ("AMD", "2026-10-14")}
        [keynote] = found[("AAPL", "2026-10-13")]
        assert (keynote.label, keynote.scope) == ("Product event", "AAPL")
        assert found[("AMD", "2026-10-14")][0].scope == ""

    def test_the_day_after_a_report_from_the_kept_earnings_dates(self):
        found = ms.events_by_day(
            ["INTC"], ["2026-10-22", "2026-10-23"], rows=[],
            earnings=lambda t: ["2026-10-22T16:00:00-04:00"], verdict=lambda t, d: None,
        )
        [event] = found[("INTC", "2026-10-23")]
        assert event.category == event_days.EARNINGS and event.scope == "INTC"
        assert ("INTC", "2026-10-22") not in found

    def test_a_shock_any_briefing_flagged_is_the_whole_markets(self):
        def verdict(ticker, day):
            return {"shock": "geo", "reason": "strikes"} if (ticker, day) == ("AAPL", "2026-10-13") else None

        found = ms.events_by_day(
            ["AAPL", "TMO"], ["2026-10-13"], rows=[], earnings=lambda t: [], verdict=verdict,
        )
        assert [e.category for e in found[("TMO", "2026-10-13")]] == [event_days.GEO]


class TestEventImpact:
    EVENTS = {("AMD", "2026-10-14"): [ms.DayEvent(event_days.CPI, "CPI")]}

    def test_an_event_day_is_set_against_the_pairs_normal_days(self):
        [series] = ms.base_series([job("a", {**WEEK1, **WEEK2})])
        rows, normals = ms.event_impact([series], self.EVENTS, ms.window([series], ms.ALL))
        [row] = rows
        assert normals[0]["normal_days"] == 5
        assert row["normal_mean"] == pytest.approx(90.0)
        assert row["delta"] == pytest.approx(-290.0)
        assert row["sigma"] == pytest.approx(-290.0 / normals[0]["normal_spread"])

    def test_a_session_the_base_sat_out_is_shown_but_not_counted(self):
        [series] = ms.base_series([job("a", {**WEEK1, **WEEK2}, sat_out=["2026-10-14"],
                                       no_forecast=["2026-10-06"])])
        rows, normals = ms.event_impact([series], self.EVENTS, ms.window([series], ms.ALL))
        assert rows[0]["profit"] is None and rows[0]["sat_out"]
        assert normals[0]["normal_days"] == 4  # the day without a forecast is left out too
        summary = ms.event_summaries(rows)[0]
        assert summary["rows"] == [] and len(summary["skipped"]) == 1

    def test_who_did_better_and_worse_by_instrument(self):
        series = ms.base_series([
            job("a", {**WEEK1, **WEEK2}),
            job("m", {"2026-10-12": 0.0, "2026-10-13": 0.0, "2026-10-14": 500.0}, ticker="MU"),
        ])
        events = {**self.EVENTS, ("MU", "2026-10-14"): [ms.DayEvent(event_days.CPI, "CPI")]}
        rows, _ = ms.event_impact(series, events, ms.window(series, ms.ALL))
        [day] = ms.event_summaries(rows)
        assert (day["better"], day["worse"]) == (1, 1)
        assert (day["best"]["ticker"], day["worst"]["ticker"]) == ("MU", "AMD")
        assert [i["ticker"] for i in ms.instrument_impact(rows)] == ["MU", "AMD"]
