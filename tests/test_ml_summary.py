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
from simlab.data import Dataset


def baseline(daily: dict, sat_out=(), no_forecast=()) -> dict:
    profits = list(daily.values())
    return {
        "overrides": {}, "profit": sum(profits), "return_pct": 0.0, "trades": 1, "sells": 1,
        "days": len(profits), "days_traded": len(profits), "days_up": 0, "days_down": 0,
        "worst_day": min(profits), "best_day": max(profits), "daily": dict(daily),
        "no_forecast_days": list(no_forecast), "sat_out_days": list(sat_out), "error": None,
    }


def job(job_id: str, daily: dict, ticker="AMD", model="highlow", created="2026-10-20T10:00",
        skip=(), grid=None, **cell) -> dict:
    """A finished job whose base replayed `daily` on one dataset, and each
    `grid` buy_k its own `daily`."""
    base = asdict(AppleTraderConfig(ticker=ticker, model_key=model, skip_events=tuple(skip)))
    return {
        "job_id": job_id, "status": tu.FINISHED, "created_at": created,
        "spec": {
            "base": base, "axes": [{"name": "buy_k", "values": [0.5]}],
            "datasets": [{"name": "w", "days": sorted(daily), "feed": "live", "symbols": [ticker]}],
            "starting_cash": 100_000.0,
        },
        "baseline": {"w": baseline(daily, **cell)},
        "cells": {"w": [
            {**baseline(days), "overrides": {"buy_k": buy_k}}
            for buy_k, days in (grid or {}).items()
        ]},
    }


def stored(buy_k: float, ticker="AMD", set_at="2026-10-25T10:00") -> dict:
    """`bases` holding the pair's base at `buy_k`, as `stored_bases` decodes one."""
    record = asdict(AppleTraderConfig(
        ticker=ticker, model_key="highlow", buy_k=buy_k, skip_events=()
    ))
    config = tu.make_config(record, {})
    return {(config.model_key, config.ticker): (config, set_at)}


WEEK1 = {"2026-10-05": 100.0, "2026-10-06": -50.0, "2026-10-07": 0.0}
WEEK2 = {"2026-10-12": 300.0, "2026-10-13": 100.0, "2026-10-14": -200.0}


class TestBaseSeries:
    """Without a stored base, a pair is read from one job's own base."""

    def test_a_pair_is_read_from_the_job_reaching_its_latest_session(self):
        old_but_longer = job("a", {**WEEK1, **WEEK2}, created="2026-10-01T00:00")
        newer_but_shorter = job("b", WEEK1, created="2026-10-20T00:00")
        [series] = ms.base_series([newer_but_shorter, old_but_longer], bases={})
        assert series.job_ids == ["a"] and series.source == ms.FROM_JOB
        assert series.last_day == "2026-10-14"

    def test_on_a_tie_the_newest_job_wins_and_pairs_stay_apart(self):
        found = ms.base_series([
            job("old", WEEK1, created="2026-10-01T00:00"),
            job("new", WEEK1, created="2026-10-02T00:00"),
            job("mu", WEEK1, ticker="MU"),
        ], bases={})
        assert [(s.ticker, s.job_ids) for s in found] == [("AMD", ["new"]), ("MU", ["mu"])]

    def test_derived_grids_and_unscored_bases_are_skipped(self):
        derived = {**job("d", WEEK1), "status": tu.DERIVED}
        broken = job("x", WEEK1)
        broken["baseline"]["w"]["error"] = "boom"
        assert ms.base_series([derived, broken], bases={}) == []


class TestStoredBase:
    """With a stored base, a pair is read from that configuration's replays."""

    def test_a_square_of_a_grid_is_the_bases_replay(self):
        jobs = [job("a", WEEK1, grid={0.5: {"2026-10-05": 7.0}})]
        [series] = ms.base_series(jobs, bases=stored(0.5))
        assert series.source == ms.FROM_BASE and series.base_set_at == "2026-10-25T10:00"
        assert series.daily == {"2026-10-05": 7.0} and series.signature.startswith("highlow_AMD(buy=H-0.5")

    def test_setting_another_base_reads_its_own_replays(self):
        jobs = [job("a", WEEK1, grid={0.5: {"2026-10-05": 7.0}, 0.6: {"2026-10-05": -3.0}})]
        assert ms.base_series(jobs, bases=stored(0.6))[0].daily == {"2026-10-05": -3.0}

    def test_every_job_that_replayed_it_adds_its_sessions_the_newest_winning(self):
        jobs = [
            job("old", WEEK1, created="2026-10-01T00:00",
                grid={0.5: {"2026-10-05": 1.0, "2026-10-06": 2.0}}),
            job("new", WEEK2, created="2026-10-20T00:00",
                grid={0.5: {"2026-10-06": 5.0, "2026-10-12": 9.0}}),
        ]
        [series] = ms.base_series(jobs, bases=stored(0.5))
        assert series.daily == {"2026-10-05": 1.0, "2026-10-06": 5.0, "2026-10-12": 9.0}
        assert series.job_ids == ["new", "old"]

    def test_a_base_no_job_replayed_falls_back_and_says_so(self):
        [series] = ms.base_series([job("a", WEEK1)], bases=stored(0.55))
        assert series.source == ms.FROM_JOB and series.job_ids == ["a"]
        assert series.base_set_at == "2026-10-25T10:00"
        assert series.daily == WEEK1


def dataset(*symbols: str) -> Dataset:
    return Dataset(name="w", symbols=list(symbols), start="2026-10-05", end="2026-10-09")


class TestUntunedPairs:
    """Every pair the app can trade that no job has scored a replay of."""

    PAIRS = [("highlow", "AMD"), ("highlow", "MU"), ("highlow2", "AAPL"), ("highlow2", "GOOG")]

    def untuned(self, jobs, datasets=(), bases=None):
        series = ms.base_series(jobs, bases=bases or {})
        return ms.untuned_pairs(series, jobs, list(datasets), pairs=self.PAIRS)

    def test_a_pair_with_its_base_replayed_is_tuned(self):
        found = self.untuned([job("a", WEEK1, grid={0.5: WEEK1})], bases=stored(0.5))
        assert [(p["ticker"], p["model_key"]) for p in found] == [
            ("AAPL", "highlow2"), ("GOOG", "highlow2"), ("MU", "highlow")
        ]
        assert all(p["status"] == ms.NO_JOB and p["job_ids"] == [] for p in found)

    def test_a_tuned_pair_with_no_base_set_is_listed(self):
        amd = next(p for p in self.untuned([job("a", WEEK1)]) if p["ticker"] == "AMD")
        assert (amd["status"], amd["job_ids"]) == (ms.NO_BASE, ["a"])

    def test_a_base_set_but_never_replayed_is_a_base(self):
        found = self.untuned([job("a", WEEK1)], bases=stored(0.55))
        assert "AMD" not in {p["ticker"] for p in found}

    def test_why_a_pair_with_jobs_is_still_untuned(self):
        running = {**job("r", WEEK1, ticker="MU"), "status": tu.RUNNING}
        running["baseline"] = {}
        failed = job("f", WEEK1, ticker="AAPL", model="highlow2", created="2026-10-01T00:00")
        failed["status"] = tu.FAILED
        failed["baseline"]["w"]["error"] = "boom"
        stopped = {**job("s", WEEK1, ticker="AAPL", model="highlow2"), "status": tu.FAILED}
        stopped["baseline"] = {}
        found = {(p["ticker"], p["model_key"]): p for p in self.untuned([running, failed, stopped])}
        assert found[("MU", "highlow")]["status"] == ms.JOB_RUNNING
        assert found[("AAPL", "highlow2")]["status"] == ms.NOT_SCORED
        assert found[("AAPL", "highlow2")]["job_ids"] == ["s", "f"]

    def test_a_running_job_that_scored_its_base_has_tuned_the_pair(self):
        running = {**job("r", WEEK1, ticker="MU"), "status": tu.RUNNING}
        mu = next(p for p in self.untuned([running]) if p["ticker"] == "MU")
        assert mu["status"] == ms.NO_BASE

    def test_how_many_datasets_carry_the_instrument(self):
        found = self.untuned([], datasets=[dataset("AMD", "MU"), dataset("mu", "GOOGL")])
        assert {(p["ticker"], p["datasets"]) for p in found} == {
            ("AMD", 1), ("MU", 2), ("AAPL", 0), ("GOOG", 0)
        }

    def test_the_registry_by_default(self):
        from agent_stonks import apple_models

        pairs = ms.registry_pairs()
        assert ("highlow", apple_models.DEFAULT_TICKER) in pairs
        assert all(t in apple_models.MODELS[k].tickers for k, t in pairs)


class TestWindows:
    def test_a_week_counts_back_from_the_latest_session_of_any_pair(self):
        series = ms.base_series(
            [job("a", {**WEEK1, **WEEK2}), job("m", WEEK1, ticker="MU")], bases={}
        )
        assert ms.window(series, ms.WEEK) == ("2026-10-08", "2026-10-14")
        assert ms.window(series, ms.ALL) == ("2026-10-05", "2026-10-14")
        mu = next(s for s in series if s.ticker == "MU")
        assert ms.pair_metrics(mu, ms.window(series, ms.WEEK)) is None

    def test_every_session_counts_traded_or_not(self):
        [series] = ms.base_series([job("a", {**WEEK1, **WEEK2})], bases={})
        week = ms.pair_metrics(series, ms.window([series], ms.WEEK))
        assert (week["sessions"], week["profit"], week["per_session"]) == (3, 200.0, 66.67)
        assert (week["days_up"], week["days_down"]) == (2, 1)
        assert week["return_pct"] == pytest.approx(0.0667)
        assert week["session_std"] == pytest.approx(251.66, abs=0.005)  # 300, 100, -200
        whole = ms.pair_metrics(series, ms.window([series], ms.ALL))
        assert whole["sessions"] == 6 and whole["worst_day"] == -200.0

    def test_a_model_is_the_mean_of_its_instruments(self):
        """Profit a session, its spread and the window's profit alike: per
        instrument, the spread over the instruments that have one."""
        rows = [
            {"model_key": "highlow", "ticker": "AMD", "per_session": 100.0, "profit": 500.0,
             "session_std": 300.0},
            {"model_key": "highlow", "ticker": "MU", "per_session": -20.0, "profit": -100.0,
             "session_std": 100.0},
            {"model_key": "goodmoments", "ticker": "AAPL", "per_session": 50.0, "profit": 250.0,
             "session_std": None},
        ]
        best, second = ms.model_metrics(rows)
        assert (best["model_key"], best["per_session"], best["profit"]) == ("goodmoments", 50.0, 250.0)
        assert best["session_std"] is None
        assert (second["per_session"], second["profit"], second["in_profit"]) == (40.0, 200.0, 1)
        assert second["session_std"] == 200.0
        assert (second["best"], second["worst"]) == ("AMD", "MU")

    def test_each_instrument_names_its_best_model_and_the_runner_up(self):
        def row(model, ticker, per_session):
            return {"model_key": model, "ticker": ticker, "per_session": per_session,
                    "session_std": abs(per_session) * 2,
                    "profit": per_session * 5, "sessions": 5, "days_up": 3, "days_down": 2}

        found = ms.best_per_instrument([
            row("highlow", "AAPL", 40.0), row("goodmoments", "AAPL", 90.0),
            row("highlow2", "AAPL", -10.0), row("highlow", "MU", 120.0),
        ])
        assert [(b["ticker"], b["model_key"]) for b in found] == [
            ("MU", "highlow"), ("AAPL", "goodmoments")
        ]
        aapl = found[1]
        assert (aapl["runner_up"], aapl["lead"], aapl["models"]) == ("highlow", 50.0, 3)
        assert aapl["session_std"] == 180.0  # GoodMoments', the best model's
        assert found[0]["runner_up"] is None and found[0]["lead"] is None

    def test_weeks_start_on_monday(self):
        [series] = ms.base_series([job("a", {**WEEK1, **WEEK2})], bases={})
        weeks = ms.weekly([series], ms.window([series], ms.ALL))
        assert [(w["week"], w["per_session"]) for w in weeks] == [
            ("2026-10-05", 16.67), ("2026-10-12", 66.67)
        ]


def _best_table() -> None:
    import streamlit as st

    from simlab import app as sim_app

    sim_app._render_ml_best(st.session_state["best"], "month")


class TestSetOrchestra:
    """The best-model table's boxes and its Set Orchestra button
    (simlab/app.py, `orchestra_defaults`)."""

    @staticmethod
    def best(ticker: str, model: str) -> dict:
        return {"ticker": ticker, "model_key": model, "per_session": 10.0,
                "session_std": 40.0, "profit": 50.0,
                "sessions": 5, "days_up": 3, "days_down": 2, "models": 1,
                "runner_up": None, "runner_up_per_session": None, "lead": None}

    def run(self, rows):
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_function(_best_table, default_timeout=60)
        app.session_state["best"] = rows
        app.run()
        assert not app.exception, app.exception
        return app

    def test_the_boxes_open_on_orchestras_pairs_and_the_button_sets_them(self):
        from agent_stonks import orchestra as ar
        from agent_stonks import orchestra_defaults

        rows = [self.best("AAPL", "highlow"), self.best("AMD", "goodmoments")]
        assert "AAPL:highlow" in ar.default_pairs() and "AMD:goodmoments" not in ar.default_pairs()
        app = self.run(rows)
        app.button(key="ml_set_orchestra").click().run()
        assert orchestra_defaults.pairs() == ["AAPL:highlow"]
        assert ar.default_pairs() == ["AAPL:highlow"]
        assert "now opens on AAPL · HighLow" in app.success[0].value

    def test_restore_brings_the_tuned_pairs_back(self):
        from agent_stonks import orchestra_defaults

        orchestra_defaults.set_pairs(["AMD:goodmoments"])
        app = self.run([self.best("AMD", "goodmoments")])
        app.button(key="ml_restore_orchestra").click().run()
        assert orchestra_defaults.pairs() is None
        assert "tuned pairs again" in app.success[0].value

    def test_nothing_ticked_sets_nothing(self):
        app = self.run([self.best("AMD", "goodmoments")])
        assert app.button(key="ml_set_orchestra").disabled


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
        [series] = ms.base_series([job("a", {**WEEK1, **WEEK2})], bases={})
        rows, normals = ms.event_impact([series], self.EVENTS, ms.window([series], ms.ALL))
        [row] = rows
        assert normals[0]["normal_days"] == 5
        assert row["normal_mean"] == pytest.approx(90.0)
        assert row["delta"] == pytest.approx(-290.0)
        assert row["sigma"] == pytest.approx(-290.0 / normals[0]["normal_spread"])

    def test_a_session_the_base_sat_out_is_shown_but_not_counted(self):
        [series] = ms.base_series([job("a", {**WEEK1, **WEEK2}, sat_out=["2026-10-14"],
                                       no_forecast=["2026-10-06"])], bases={})
        rows, normals = ms.event_impact([series], self.EVENTS, ms.window([series], ms.ALL))
        assert rows[0]["profit"] is None and rows[0]["sat_out"]
        assert normals[0]["normal_days"] == 4  # the day without a forecast is left out too
        summary = ms.event_summaries(rows)[0]
        assert summary["rows"] == [] and len(summary["skipped"]) == 1

    def test_who_did_better_and_worse_by_instrument(self):
        series = ms.base_series([
            job("a", {**WEEK1, **WEEK2}),
            job("m", {"2026-10-12": 0.0, "2026-10-13": 0.0, "2026-10-14": 500.0}, ticker="MU"),
        ], bases={})
        events = {**self.EVENTS, ("MU", "2026-10-14"): [ms.DayEvent(event_days.CPI, "CPI")]}
        rows, _ = ms.event_impact(series, events, ms.window(series, ms.ALL))
        [day] = ms.event_summaries(rows)
        assert (day["better"], day["worse"]) == (1, 1)
        assert (day["best"]["ticker"], day["worst"]["ticker"]) == ("MU", "AMD")
        assert [i["ticker"] for i in ms.instrument_impact(rows)] == ["MU", "AMD"]
