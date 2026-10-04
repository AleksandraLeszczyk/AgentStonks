"""SimLab's parameter tuning for Apple Trader (simlab/tuning.py).

Four parts. The grid and the pick are pure functions, pinned on hand-made
cells. Scoring is pinned on a hand-made replay result. One job runs end to end
on a synthetic store with the day-range forecast stubbed, in process. And the
two replay speed-ups the grid depends on -- the market's per-day index and the
replay-backed minute frame -- are pinned against the scans they replaced,
at every step of a stored session, because a faster replay that answered
differently would make every tuned number wrong.
"""
import os
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from agent_stonks import momentum_regime
from agent_stonks.apple_trader import APPLE_TRADER_KEY, AppleTraderConfig
from agent_stonks.config import UNIT_ADR
from agent_stonks.market_hours import MARKET_TZ
from dataclasses import asdict, fields
from simlab import data as sim_data
from simlab import tuning as tu
from simlab.engine import SimulationConfig, SimulationEngine
from simlab.market import BAR_SEC, SimMarket
from simlab.patches import simulation_context

TUNE_DAY = date(2026, 6, 15)   # a Monday
TEST_DAY = date(2026, 6, 16)
TICKER = "AAPL"


# --- the grid ---------------------------------------------------------------


class TestGrid:
    def test_axis_values_are_clean_numbers(self):
        assert tu.axis_values("buy_k", 0.3, 0.9, 0.1) == [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    def test_an_integer_parameter_stays_integer(self):
        assert tu.axis_values("flatten_before_close_min", 1, 7, 2) == [1, 3, 5, 7]

    @pytest.mark.parametrize("start,stop,step", [(0.3, 0.9, 0.0), (0.9, 0.3, 0.1), (0.0, 0.5, 0.1)])
    def test_an_impossible_axis_is_refused(self, start, stop, step):
        with pytest.raises(ValueError):
            tu.axis_values("buy_k", start, stop, step)  # buy_k's minimum is 0.05

    def test_the_grid_is_every_combination_row_by_row(self):
        axes = [{"name": "buy_k", "values": [0.5, 0.7]}, {"name": "sell_k", "values": [0.1, 0.2]}]
        assert tu.grid(axes) == [
            {"buy_k": 0.5, "sell_k": 0.1}, {"buy_k": 0.5, "sell_k": 0.2},
            {"buy_k": 0.7, "sell_k": 0.1}, {"buy_k": 0.7, "sell_k": 0.2},
        ]

    def test_every_tunable_is_a_config_field_with_a_legal_default_range(self):
        fields = set(asdict(AppleTraderConfig()))
        for tunable in tu.TUNABLES.values():
            assert tunable.name in fields
            assert tu.axis_values(tunable.name, *tunable.default_range)


def dataset(name: str = "d", day: date = TUNE_DAY, **overrides) -> dict:
    return {"name": name, "days": [str(day)], "feed": sim_data.DEFAULT_FEED,
            "symbols": [TICKER], **overrides}


def spec(**overrides):
    base = {
        "base": asdict(AppleTraderConfig(ticker=TICKER)),
        "axes": [{"name": "buy_k", "values": [0.5, 0.7]}],
        "datasets": [dataset("d", TUNE_DAY), dataset("e", TEST_DAY)],
        "starting_cash": 10_000.0,
    }
    return {**base, **overrides}


class TestBaseConfiguration:
    """A job's `base` is a stored record, so it decodes like one.

    A job saved before a field existed has to keep describing the run it
    actually tuned -- re-opening it must not re-sign or re-run it under
    whatever the default has since become.
    """

    def test_overrides_land_on_top_of_the_base(self):
        config = tu.make_config(spec()["base"], {"buy_k": 0.55})
        assert config.buy_k == 0.55 and config.ticker == TICKER

    def test_a_field_the_job_predates_decodes_to_what_it_meant_then(self):
        base = {k: v for k, v in spec()["base"].items() if k != "breach_update"}
        assert tu.make_config(base, {}).breach_update == "off"
        assert tu.make_config(spec()["base"], {}).breach_update == (
            AppleTraderConfig().breach_update
        )

    def test_a_field_of_a_removed_strategy_is_dropped_rather_than_raising(self):
        config = tu.make_config({**spec()["base"], "reversal_threshold": 0.3}, {})
        assert config.ticker == TICKER

    def test_an_impossible_cell_still_raises_for_the_caller_to_mark(self):
        with pytest.raises(ValueError):
            tu.make_config(spec()["base"], {"buy_k": 0.1, "sell_k": 0.5})


class TestSweepVocabulary:
    """What the Simulate tab's per-setup sweep is allowed to vary.

    Shared with the Tuning tab's numeric axes on purpose -- one table saying
    what a setting is and what it may be, rather than two that drift the first
    time a range is widened.
    """

    def test_every_sweepable_field_is_a_real_config_field(self):
        names = {f.name for f in fields(AppleTraderConfig)}
        assert set(tu.SWEEPABLE) <= names

    def test_every_sweepable_field_has_a_range_or_a_set_of_options(self):
        for name in tu.SWEEPABLE:
            assert name in tu.TUNABLES or name in tu.CHOICES

    def test_the_two_vocabularies_do_not_overlap(self):
        assert not set(tu.TUNABLES) & set(tu.CHOICES)

    def test_housekeeping_the_signature_ignores_is_not_offered(self):
        """Varying it would queue several runs that Results shows as one row."""
        assert "flatten_before_close_min" in tu.TUNABLES
        assert "flatten_before_close_min" not in tu.SWEEPABLE

    def test_a_choices_options_are_all_configurations_the_agent_accepts(self):
        for name, choice in tu.CHOICES.items():
            for option in choice.options:
                assert getattr(AppleTraderConfig(**{name: option}), name) == option

    def test_every_option_has_a_label(self):
        for choice in tu.CHOICES.values():
            assert all(choice.labels.get(o) for o in choice.options)

    def test_a_number_is_labelled_by_its_format_and_a_rule_by_its_name(self):
        assert tu.value_label("buy_k", 0.5) == "0.50"
        assert tu.value_label("breach_update", "off") == tu.CHOICES[
            "breach_update"
        ].labels["off"]

    def test_an_unknown_field_falls_back_to_its_name_and_value(self):
        assert tu.sweep_label("nonsense") == "nonsense"
        assert tu.value_label("nonsense", 3) == "3"

    def test_a_grid_axis_may_be_a_number_or_a_rule(self):
        """Both vocabularies, numbers first -- the order both forms list them in."""
        assert tu.AXES == tuple(tu.TUNABLES) + tuple(tu.CHOICES)
        assert "breach_update" in tu.AXES

    def test_an_axis_is_titled_without_its_unit_or_its_sentence(self):
        """What a heatmap axis and a job label carry, next to their own values."""
        assert tu.axis_title("buy_k") == "Buy distance"
        assert tu.axis_title("breach_update") == "Forecast breach"
        assert tu.axis_title("nonsense") == "nonsense"

    def test_a_rule_ticks_as_its_own_key(self):
        """`extreme`, not the sentence -- the axis title already says what it is,
        and this is what `config_signature` writes it as."""
        assert tu.axis_tick("breach_update", "extreme") == "extreme"
        assert tu.axis_tick("buy_k", 0.5) == "0.50"


class TestCellCount:
    """The size guard runs before the grid exists, so it cannot build one."""

    def test_it_is_the_product_of_the_axes(self):
        axes = [{"name": "buy_k", "values": [1, 2, 3]},
                {"name": "sell_k", "values": [1, 2]}]
        assert tu.cell_count(axes) == len(tu.grid(axes)) == 6

    def test_no_axes_is_the_one_base_configuration(self):
        assert tu.cell_count([]) == len(tu.grid([])) == 1

    def test_a_grid_far_too_large_to_build_is_still_counted(self):
        axes = [{"name": f"a{i}", "values": list(range(10))} for i in range(7)]
        assert tu.cell_count(axes) == 10_000_000


class TestExpand:
    """One setup crossed with its axes: what actually gets queued."""

    def base(self, **kwargs) -> AppleTraderConfig:
        kwargs.setdefault("ticker", TICKER)
        return AppleTraderConfig(**kwargs)

    def test_no_axes_is_the_base_configuration_alone(self):
        base = self.base()
        configs, refused = tu.expand(base, [])
        assert configs == [base] and refused == []

    def test_one_axis_is_one_configuration_per_value(self):
        configs, _ = tu.expand(
            self.base(), [{"name": "buy_k", "values": [0.5, 0.6, 0.7]}]
        )
        assert [c.buy_k for c in configs] == [0.5, 0.6, 0.7]

    def test_two_axes_are_their_product(self):
        configs, _ = tu.expand(self.base(), [
            {"name": "buy_k", "values": [0.5, 0.7]},
            {"name": "breach_update", "values": ["off", "extreme"]},
        ])
        assert len(configs) == 4
        assert {(c.buy_k, c.breach_update) for c in configs} == {
            (0.5, "off"), (0.5, "extreme"), (0.7, "off"), (0.7, "extreme"),
        }

    def test_a_numeric_and_a_rule_axis_mix(self):
        configs, _ = tu.expand(self.base(), [
            {"name": "stop_gain_fraction", "values": [0.0, 0.2]},
            {"name": "level_source", "values": ["dayrange"]},
        ])
        assert all(c.level_source == "dayrange" for c in configs)
        assert sorted(c.stop_gain_fraction for c in configs) == [0.0, 0.2]

    def test_everything_the_axes_do_not_name_comes_from_the_base(self):
        base = self.base(position_pct=40.0, min_win_k=0.15)
        configs, _ = tu.expand(base, [{"name": "buy_k", "values": [0.5, 0.6]}])
        assert all(c.position_pct == 40.0 and c.min_win_k == 0.15 for c in configs)

    def test_a_cell_that_is_not_a_strategy_is_refused_with_its_reason(self):
        """Not dropped: a grid quietly one row short is worse than one that
        explains itself."""
        configs, refused = tu.expand(
            self.base(sell_k=0.25), [{"name": "buy_k", "values": [0.1, 0.5]}]
        )
        assert [c.buy_k for c in configs] == [0.5]
        assert len(refused) == 1
        overrides, reason = refused[0]
        assert overrides == {"buy_k": 0.1} and "must sit above the buy level" in reason

    def test_cells_that_sign_the_same_are_queued_once(self):
        """`take_fraction` is not in the signature while the momentum take is
        off, so varying it there is one configuration however many values."""
        configs, refused = tu.expand(
            self.base(momentum_confirmation_bars=0),
            [{"name": "take_fraction", "values": [0.3, 0.6, 1.0]}],
        )
        assert len(configs) == 1 and refused == []

    def test_the_collapse_is_visible_against_the_grid(self):
        axes = [{"name": "take_fraction", "values": [0.3, 0.6, 1.0]}]
        configs, refused = tu.expand(self.base(momentum_confirmation_bars=0), axes)
        assert tu.cell_count(axes) - len(configs) - len(refused) == 2

    def test_the_order_is_the_grids(self):
        axes = [
            {"name": "buy_k", "values": [0.5, 0.7]},
            {"name": "stop_gain_fraction", "values": [0.0, 0.2]},
        ]
        configs, _ = tu.expand(self.base(), axes)
        assert [(c.buy_k, c.stop_gain_fraction) for c in configs] == [
            (0.5, 0.0), (0.5, 0.2), (0.7, 0.0), (0.7, 0.2)
        ]

    def test_every_configuration_is_one_the_engine_would_accept(self):
        configs, _ = tu.expand(self.base(), [
            {"name": "buy_k", "values": tu.axis_values("buy_k", 0.3, 0.9, 0.1)},
            {"name": "level_source", "values": list(tu.CHOICES["level_source"].options)},
        ])
        assert configs and all(isinstance(c, AppleTraderConfig) for c in configs)
        assert len({tu.config_signature(c) for c in configs}) == len(configs)


class TestValidation:
    def test_a_sound_spec_passes(self):
        assert tu.validate(spec()) is None

    def test_at_most_three_parameters(self):
        axes = [
            {"name": n, "values": [0.5]}
            for n in ("buy_k", "sell_k", "stop_gain_fraction", "take_fraction")
        ]
        assert "one to 3" in tu.validate(spec(axes=axes))
        assert tu.validate(spec(axes=axes[:3])) is None

    def test_the_same_parameter_twice_is_refused(self):
        axes = [{"name": "buy_k", "values": [0.5]}, {"name": "buy_k", "values": [0.7]}]
        assert "twice" in tu.validate(spec(axes=axes))

    def test_a_dataset_without_the_instrument_is_refused(self):
        other = dataset("f", symbols=["GOOGL"])
        assert "does not carry AAPL" in tu.validate(spec(datasets=[dataset(), other]))

    def test_at_least_one_dataset(self):
        assert "at least one dataset" in tu.validate(spec(datasets=[]))

    def test_the_same_dataset_twice_is_refused(self):
        """Cells are kept by dataset name: twice would be one sweep summed twice."""
        assert "twice" in tu.validate(spec(datasets=[dataset(), dataset()]))

    def test_a_rule_is_a_parameter_a_job_may_tune(self):
        """The levels under each forecast policy, which is the comparison the
        policies have never had: neither shipped level was swept under them."""
        axes = [{"name": "buy_k", "values": [0.5, 0.7]},
                {"name": "breach_update", "values": ["off", "extreme", "brownian"]}]
        assert tu.validate(spec(axes=axes)) is None
        assert len(tu.grid(axes)) == 6

    def test_a_rule_axis_with_nothing_ticked_is_refused(self):
        axes = [{"name": "breach_update", "values": []}]
        assert "no values to try" in tu.validate(spec(axes=axes))

    def test_an_option_that_is_not_one_is_refused(self):
        """A stored job outlives its rules. An option this build no longer has
        would quietly replay as the config's default and be filed under the
        name of a policy it never ran."""
        axes = [{"name": "breach_update", "values": ["extreme", "telepathy"]}]
        assert "telepathy" in tu.validate(spec(axes=axes))

    def test_too_many_cells_is_refused(self):
        axes = [{"name": "buy_k", "values": list(range(30))},
                {"name": "sell_k", "values": list(range(30))}]
        assert "more than" in tu.validate(spec(axes=axes))

    def test_the_replay_count_is_the_grid_and_the_baseline_on_every_dataset(self):
        assert tu.total_replays(spec()) == (2 + 1) * 2
        assert tu.total_replays(spec(datasets=[dataset()])) == 2 + 1


# --- the pick ---------------------------------------------------------------

AXES = [{"name": "buy_k", "values": [0.3, 0.5, 0.7]},
        {"name": "sell_k", "values": [0.1, 0.2, 0.3]}]


def cells_from(profits, traded=None):
    """A 3x3 grid of scored cells from a row-major list of profits."""
    out = []
    for i, overrides in enumerate(tu.grid(AXES)):
        if profits[i] is None:
            out.append({"overrides": overrides, "invalid": "sell_k must sit above"})
            continue
        out.append({"overrides": overrides, "profit": profits[i], "days": 10,
                    "days_traded": 10 if traded is None else traded[i]})
    return out


class TestPick:
    # A lone spike at (0.3, 0.1) against a broad profitable region around (0.7, 0.2).
    PROFITS = [900, -400, -400,
               -300, 300, 350,
               250, 400, 300]

    def test_max_takes_the_highest_cell(self):
        best = tu.pick_best(cells_from(self.PROFITS), AXES, rule=tu.PICK_MAX)
        assert best["overrides"] == {"buy_k": 0.3, "sell_k": 0.1}
        assert best["pick_score"] == 900

    def test_plateau_takes_the_cell_with_the_best_neighbourhood(self):
        """Not the spike: the corner of the profitable region, whose neighbourhood
        -- itself and the three cells around it that the grid has -- averages
        best. An edge cell averages over fewer neighbours, as `sweep_levels.py`'s
        clipped window does."""
        best = tu.pick_best(cells_from(self.PROFITS), AXES, rule=tu.PICK_PLATEAU)
        assert best["overrides"] == {"buy_k": 0.7, "sell_k": 0.3}
        assert best["pick_score"] == pytest.approx((300 + 350 + 400 + 300) / 4)

    def test_a_cell_that_rarely_trades_is_not_eligible(self):
        traded = [1] + [10] * 8
        best = tu.pick_best(cells_from(self.PROFITS, traded), AXES, min_traded_share=0.5)
        assert best["overrides"] != {"buy_k": 0.3, "sell_k": 0.1}

    def test_invalid_cells_are_never_picked_nor_counted_as_neighbours(self):
        profits = list(self.PROFITS)
        profits[0] = None
        best = tu.pick_best(cells_from(profits), AXES, rule=tu.PICK_PLATEAU)
        assert best["overrides"] == {"buy_k": 0.7, "sell_k": 0.3}
        # (0.5, 0.2) now averages over its seven scored neighbours, not eight.
        middle = tu.pick_best(
            [c for c in cells_from(profits) if c["overrides"] == {"buy_k": 0.5, "sell_k": 0.2}]
            + cells_from(profits), AXES, rule=tu.PICK_PLATEAU,
        )
        assert middle["overrides"] == {"buy_k": 0.7, "sell_k": 0.3}
        spike = tu.pick_best(cells_from(profits), AXES, rule=tu.PICK_MAX)
        assert spike["overrides"] == {"buy_k": 0.7, "sell_k": 0.2}

    def test_nothing_scored_is_no_pick(self):
        assert tu.pick_best(cells_from([None] * 9), AXES) is None

    def test_ties_go_to_the_first_cell(self):
        best = tu.pick_best(cells_from([5] * 9), AXES)
        assert best["overrides"] == {"buy_k": 0.3, "sell_k": 0.1}

    def test_a_rule_axis_has_no_neighbours(self):
        """A plateau is robustness to a small change in the same setting, and
        there is no such thing as a small change of forecast policy. So the
        neighbourhood is read within each rule: here the middle of the run of
        profitable cells under `off`, not a cell smeared across two policies.
        """
        axes = [{"name": "breach_update", "values": ["off", "extreme"]},
                {"name": "buy_k", "values": [0.3, 0.5, 0.7]}]
        profits = {
            ("off", 0.3): 300, ("off", 0.5): 320, ("off", 0.7): 310,
            ("extreme", 0.3): 900, ("extreme", 0.5): -500, ("extreme", 0.7): -500,
        }
        cells = [
            {"overrides": o, "profit": profits[(o["breach_update"], o["buy_k"])],
             "days": 10, "days_traded": 10}
            for o in tu.grid(axes)
        ]
        best = tu.pick_best(cells, axes, rule=tu.PICK_PLATEAU)
        assert best["overrides"] == {"breach_update": "off", "buy_k": 0.7}
        # Its neighbourhood is the `off` row alone -- itself and 0.5. Were the
        # two policies adjacent, every `off` cell would be averaged against the
        # losses under `extreme` and the 900 spike would have carried the pick.
        assert best["pick_score"] == pytest.approx((320 + 310) / 2)

    def test_a_numeric_axis_still_averages_across_both(self):
        """The change is only about rules: a grid of two numbers is unmoved."""
        best = tu.pick_best(cells_from(self.PROFITS), AXES, rule=tu.PICK_PLATEAU)
        assert best["pick_score"] == pytest.approx((300 + 350 + 400 + 300) / 4)

    def test_overlapping_days_are_named(self):
        assert tu.overlapping_days(["2026-06-15", "2026-06-16"], ["2026-06-16"]) == ["2026-06-16"]


class TestChronological:
    """Datasets are shown oldest first, whatever order they were added in."""

    SPEC = {"datasets": [
        {"name": "late", "days": ["2026-09-22", "2026-09-21"]},
        {"name": "b_early", "days": ["2026-09-14", "2026-09-18"]},
        {"name": "a_early", "days": ["2026-09-14", "2026-09-18"]},
        {"name": "early_short", "days": ["2026-09-14", "2026-09-15"]},
    ]}

    def test_by_first_session_then_last_then_name(self):
        assert tu.dataset_names(self.SPEC, by_date=True) == [
            "early_short", "a_early", "b_early", "late",
        ]

    def test_the_record_keeps_the_order_they_were_added_in(self):
        assert tu.dataset_names(self.SPEC) == ["late", "b_early", "a_early", "early_short"]
        tu.chronological(self.SPEC["datasets"])
        assert self.SPEC["datasets"][0]["name"] == "late"


# --- the sum over datasets --------------------------------------------------


def scored(overrides, profit, *, days=5, traded=5, worst=-10.0, best=20.0, run_id=None):
    cell = {"overrides": overrides, "profit": profit, "return_pct": profit / 100.0,
            "trades": traded, "sells": traded, "days": days, "days_traded": traded,
            "days_up": 1 if profit > 0 else 0, "days_down": 0 if profit > 0 else 1,
            "worst_day": worst, "best_day": best, "daily": {}, "no_forecast_days": [],
            "error": None}
    if run_id:
        cell["run_id"] = run_id
    return cell


def job_record(datasets: "dict[str, list[dict]]", axes=None, **spec_overrides) -> dict:
    """A job record over `{dataset name: cells}`, as the store keeps one."""
    axes = axes or [{"name": "buy_k", "values": [0.3, 0.5]}]
    return {
        "job_id": "j", "status": tu.FINISHED,
        "spec": {**spec(axes=axes, datasets=[dataset(n) for n in datasets]), **spec_overrides},
        "cells": {n: list(cells) for n, cells in datasets.items()},
        "baseline": {n: scored({}, 1.0) for n in datasets},
        "best": None,
    }


class TestSum:
    """The heatmap a job is picked on: every dataset's grid, added cell by cell."""

    def test_profits_returns_and_counts_add_and_the_extremes_are_kept(self):
        total = tu.combine({"buy_k": 0.3}, [
            scored({"buy_k": 0.3}, 300.0, worst=-50.0, best=90.0, traded=4),
            scored({"buy_k": 0.3}, -100.0, worst=-80.0, best=40.0, traded=2, run_id="r"),
        ])
        assert total["profit"] == 200.0 and total["return_pct"] == pytest.approx(2.0)
        assert (total["days"], total["days_traded"], total["trades"]) == (10, 6, 6)
        assert (total["worst_day"], total["best_day"]) == (-80.0, 90.0)
        assert (total["datasets"], total["datasets_up"], total["reused"]) == (2, 1, 1)
        assert tu.is_scored(total)

    def test_round_trips_and_wins_add_into_one_win_rate(self):
        total = tu.combine({"buy_k": 0.3}, [
            {**scored({"buy_k": 0.3}, 300.0), "round_trips": 3, "wins": 2},
            {**scored({"buy_k": 0.3}, -100.0), "round_trips": 1, "wins": 0},
        ])
        assert (total["round_trips"], total["wins"]) == (4, 2)
        assert tu.win_rate(total) == 50.0

    def test_a_dataset_scored_before_round_trips_leaves_the_sum_without_a_rate(self):
        """Not a rate over the other datasets' trades only."""
        total = tu.combine({"buy_k": 0.3}, [
            {**scored({"buy_k": 0.3}, 300.0), "round_trips": 3, "wins": 2},
            scored({"buy_k": 0.3}, -100.0),
        ])
        assert total["round_trips"] is None and tu.win_rate(total) is None

    def test_a_dataset_not_yet_swept_leaves_a_hole(self):
        """A partial total would read as a bad cell rather than an unfinished one."""
        assert tu.combine({"buy_k": 0.3}, [scored({"buy_k": 0.3}, 300.0), None]) is None

    def test_a_refused_cell_is_refused_in_the_sum(self):
        total = tu.combine({"sell_k": 0.9}, [{"overrides": {"sell_k": 0.9}, "invalid": "no"}, None])
        assert total == {"overrides": {"sell_k": 0.9}, "invalid": "no"}

    def test_an_errored_replay_leaves_the_sum_unscored(self):
        broken = {**scored({"buy_k": 0.3}, 0.0), "error": "boom"}
        total = tu.combine({"buy_k": 0.3}, [scored({"buy_k": 0.3}, 300.0), broken])
        assert total["error"] == "boom" and not tu.is_scored(total)

    def test_the_pick_is_the_best_total_not_the_best_week(self):
        """0.3 wins the first week by a mile and loses the second; 0.5 is
        steady and wins the sum."""
        record = job_record({
            "w1": [scored({"buy_k": 0.3}, 900.0), scored({"buy_k": 0.5}, 400.0)],
            "w2": [scored({"buy_k": 0.3}, -700.0), scored({"buy_k": 0.5}, 350.0)],
        })
        tu.refresh_pick(record)
        assert record["best"]["overrides"] == {"buy_k": 0.5}
        assert record["best"]["profit"] == 750.0
        assert tu.pick_cell(record, "w2")["profit"] == 350.0
        assert tu.summed_baseline(record)["profit"] == 2.0

    def test_missing_replays_count_every_dataset(self):
        record = job_record({"w1": [scored({"buy_k": 0.3}, 1.0)], "w2": []})
        record["baseline"]["w2"] = None
        assert tu.missing_replays(record) == 1 + 3


# --- a third axis -----------------------------------------------------------


AXES3 = [{"name": "buy_k", "values": [0.3, 0.5]},
         {"name": "sell_k", "values": [0.1, 0.2]},
         {"name": "take_fraction", "values": [0.5, 0.75, 1.0]}]


def cells3(profits: dict) -> "list[dict]":
    """A 2x2x3 grid from {(buy, sell, take): profit}; None is a refused cell."""
    out = []
    for overrides in tu.grid(AXES3):
        profit = profits[tuple(overrides.values())]
        if profit is None:
            out.append({"overrides": overrides, "invalid": "refused"})
        else:
            out.append(scored(overrides, profit, traded=profit % 5))
    return out


class TestThirdAxis:
    """A three-axis grid drawn as a heatmap of its first two: one value of the
    third at a time, or the third collapsed to each square's best or average."""

    PROFITS = {
        (0.3, 0.1, 0.5): 100, (0.3, 0.1, 0.75): 400, (0.3, 0.1, 1.0): 100,
        (0.3, 0.2, 0.5): -60, (0.3, 0.2, 0.75): -30, (0.3, 0.2, 1.0): -90,
        (0.5, 0.1, 0.5): 50, (0.5, 0.1, 0.75): 50, (0.5, 0.1, 1.0): 20,
        (0.5, 0.2, 0.5): None, (0.5, 0.2, 0.75): 70, (0.5, 0.2, 1.0): 10,
    }

    def by_square(self, view, cells=None) -> dict:
        out = tu.slice_cells(cells or cells3(self.PROFITS), AXES3, view)
        return {(c["overrides"]["buy_k"], c["overrides"]["sell_k"]): c for c in out}

    def test_a_slice_is_the_grid_at_one_value_of_the_third(self):
        found = self.by_square(1)
        assert {k: c["profit"] for k, c in found.items()} == {
            (0.3, 0.1): 400, (0.3, 0.2): -30, (0.5, 0.1): 50, (0.5, 0.2): 70,
        }
        # Only the first two axes are left, so it draws as any two-axis grid.
        assert all(set(c["overrides"]) == {"buy_k", "sell_k"} for c in found.values())

    def test_a_refused_cell_stays_refused_on_its_slice(self):
        assert "invalid" in self.by_square(0)[(0.5, 0.2)]

    def test_best_takes_each_squares_best_value_and_names_it(self):
        found = self.by_square(tu.SLICE_BEST)
        assert {k: c["profit"] for k, c in found.items()} == {
            (0.3, 0.1): 400, (0.3, 0.2): -30, (0.5, 0.1): 50, (0.5, 0.2): 70,
        }
        assert found[(0.3, 0.1)]["best_of"] == {
            "name": "take_fraction", "value": 0.75, "count": 3,
        }
        # The winning cell as it is, not a re-scored one.
        assert found[(0.3, 0.1)]["days_traded"] == 400 % 5

    def test_best_ties_go_to_the_earlier_value(self):
        assert self.by_square(tu.SLICE_BEST)[(0.5, 0.1)]["best_of"]["value"] == 0.5

    def test_best_skips_a_refused_value(self):
        assert self.by_square(tu.SLICE_BEST)[(0.5, 0.2)]["best_of"]["count"] == 2

    def test_the_average_is_over_every_value_the_configuration_accepts(self):
        found = self.by_square(tu.SLICE_MEAN)
        assert found[(0.3, 0.1)]["profit"] == pytest.approx(200.0)
        assert found[(0.3, 0.1)]["return_pct"] == pytest.approx(2.0)
        assert found[(0.3, 0.1)]["mean_of"] == {
            "name": "take_fraction", "count": 3, "of": 3, "low": 100.0, "high": 400.0,
        }
        # The refused value is left out, not counted as zero.
        assert found[(0.5, 0.2)]["profit"] == pytest.approx(40.0)
        assert (found[(0.5, 0.2)]["mean_of"]["count"], found[(0.5, 0.2)]["mean_of"]["of"]) == (2, 3)
        assert all(tu.is_scored(c) for c in found.values())

    def test_a_square_collapses_only_once_every_value_is_in(self):
        """The best or the average of the values swept so far would read as a
        finished square."""
        cells = [c for c in cells3(self.PROFITS)
                 if c["overrides"] != {"buy_k": 0.3, "sell_k": 0.1, "take_fraction": 1.0}]
        for view in (tu.SLICE_BEST, tu.SLICE_MEAN):
            found = self.by_square(view, cells)
            assert (0.3, 0.1) not in found and len(found) == 3
        # A slice that does not need the missing cell is unaffected.
        assert (0.3, 0.1) in self.by_square(0, cells)

    def test_a_square_with_every_value_refused_is_refused(self):
        profits = {**self.PROFITS, (0.5, 0.2, 0.75): None, (0.5, 0.2, 1.0): None}
        for view in (tu.SLICE_BEST, tu.SLICE_MEAN):
            found = self.by_square(view, cells3(profits))
            assert found[(0.5, 0.2)] == {"overrides": {"buy_k": 0.5, "sell_k": 0.2},
                                         "invalid": "refused"}

    def test_an_errored_square_with_nothing_scored_is_errored(self):
        cells = cells3({**self.PROFITS, (0.5, 0.2, 0.75): None, (0.5, 0.2, 1.0): 0})
        for cell in cells:
            if cell["overrides"] == {"buy_k": 0.5, "sell_k": 0.2, "take_fraction": 1.0}:
                cell["error"] = "boom"
        found = self.by_square(tu.SLICE_MEAN, cells)
        assert found[(0.5, 0.2)]["error"] == "boom" and not tu.is_scored(found[(0.5, 0.2)])

    def test_a_grid_of_two_axes_is_returned_as_it_is(self):
        cells = cells_from(TestPick.PROFITS)
        assert tu.slice_cells(cells, AXES, tu.SLICE_BEST) == cells

    def test_the_sum_slices_like_any_grid(self):
        """Best of all on the sum is each square's best *total*, which need not
        be either dataset's own best."""
        w1 = cells3(self.PROFITS)
        w2 = [
            {**c, "profit": -c["profit"] + (500 if c["overrides"]["take_fraction"] == 1.0 else 0)}
            if tu.is_scored(c) else c
            for c in cells3(self.PROFITS)
        ]
        record = job_record({"w1": w1, "w2": w2}, axes=AXES3)
        found = {
            (c["overrides"]["buy_k"], c["overrides"]["sell_k"]): c
            for c in tu.slice_cells(tu.summed_cells(record), AXES3, tu.SLICE_BEST)
        }
        assert found[(0.3, 0.1)]["best_of"]["value"] == 1.0
        assert found[(0.3, 0.1)]["profit"] == 500.0
        assert found[(0.3, 0.1)]["datasets"] == 2

    def test_the_pick_reads_a_plateau_along_all_three_axes(self):
        """On a 2x2 every square neighbours every other, so the plateau is
        decided along the third axis: take 0.5's neighbourhood (itself and
        0.75, the refused cell left out) averages best, and the first of its
        cells in grid order takes the tie -- not the 400 spike at 0.75."""
        best = tu.pick_best(cells3(self.PROFITS), AXES3, rule=tu.PICK_PLATEAU)
        assert best["overrides"] == {"buy_k": 0.3, "sell_k": 0.1, "take_fraction": 0.5}
        assert best["pick_score"] == pytest.approx((100 - 60 + 50 + 400 - 30 + 50 + 70) / 7)
        assert tu.pick_best(cells3(self.PROFITS), AXES3)["profit"] == 400


# --- scoring ----------------------------------------------------------------


def et(day: date, hhmm: str) -> str:
    hour, minute = map(int, hhmm.split(":"))
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=MARKET_TZ).isoformat()


class TestScore:
    def result(self):
        return SimpleNamespace(
            starting_cash=10_000.0,
            final_value=10_150.0,
            equity=[
                {"ts": et(TUNE_DAY, "09:31"), "value": 10_000.0},
                {"ts": et(TUNE_DAY, "15:59"), "value": 10_200.0},
                {"ts": et(TEST_DAY, "15:59"), "value": 10_150.0},
            ],
            decisions=[
                {"action": "buy", "status": "filled", "ts": et(TUNE_DAY, "10:00")},
                {"action": "sell", "status": "filled", "ts": et(TUNE_DAY, "11:00")},
                {"action": "buy", "status": "filled", "ts": et(TEST_DAY, "10:00")},
                {"action": "buy", "status": "rejected", "ts": et(TEST_DAY, "10:05")},
            ],
            agent_log=[
                {"type": "error", "ts": et(TEST_DAY, "09:35"),
                 "text": "Apple Trader cannot forecast today's AAPL range"},
            ],
            error=None,
        )

    def test_daily_profit_is_the_change_in_marked_equity(self):
        scored = tu.score(self.result(), [TUNE_DAY, TEST_DAY, date(2026, 6, 17)])
        assert scored["daily"] == {
            "2026-06-15": 200.0, "2026-06-16": -50.0, "2026-06-17": 0.0,
        }
        assert (scored["days_up"], scored["days_down"], scored["days"]) == (1, 1, 3)
        assert (scored["worst_day"], scored["best_day"]) == (-50.0, 200.0)

    def test_trades_count_filled_entries_and_the_days_they_were_on(self):
        scored = tu.score(self.result(), [TUNE_DAY, TEST_DAY])
        assert (scored["trades"], scored["sells"], scored["days_traded"]) == (2, 1, 2)

    def test_profit_return_and_unforecastable_days(self):
        scored = tu.score(self.result(), [TUNE_DAY, TEST_DAY])
        assert scored["profit"] == 150.0 and scored["return_pct"] == pytest.approx(1.5)
        assert scored["no_forecast_days"] == ["2026-06-16"]

    def test_a_session_sat_out_is_named_apart_from_one_not_forecast(self):
        result = self.result()
        result.agent_log.append({
            "type": "analysis", "ts": et(TUNE_DAY, "09:35"),
            "text": "Apple Trader sits out AAPL's session today — CPI release (...).",
        })
        scored = tu.score(result, [TUNE_DAY, TEST_DAY])
        assert scored["sat_out_days"] == ["2026-06-15"]
        assert scored["no_forecast_days"] == ["2026-06-16"]

    def test_fills_without_cash_or_position_leave_round_trips_uncounted(self):
        scored = tu.score(self.result(), [TUNE_DAY, TEST_DAY])
        assert scored["round_trips"] is None and tu.win_rate(scored) is None


class TestReplayInputs:
    def test_days_off_make_a_run_read_the_calendar_and_its_own_days_verdicts(self):
        from agent_stonks import event_days

        days = [str(TUNE_DAY), str(TEST_DAY)]
        with_days = tu._replay_inputs(AppleTraderConfig(ticker=TICKER), days, "yfinance")
        without = tu._replay_inputs(AppleTraderConfig(ticker=TICKER, skip_events=()), days, "yfinance")
        added = set(with_days) - set(without)
        assert event_days.CALENDAR_PATH in added
        assert {event_days.verdicts_path(TUNE_DAY), event_days.verdicts_path(TEST_DAY)} <= added
        # Only its own sessions': a verdict written this morning stales no old run.
        assert event_days.verdicts_path(date(2026, 10, 5)) not in added


class TestRoundTrips:
    """A round trip is flat to flat, fees included, however many fills it took."""

    @staticmethod
    def fill(action, cash_after, position_after):
        return {"action": action, "status": "filled",
                "cash_after": cash_after, "position_after": position_after}

    def test_a_laddered_entry_and_a_partial_take_are_one_round_trip(self):
        fills = [
            self.fill("buy", 5_000.0, 50),     # opens from 10,000
            self.fill("buy", 3_000.0, 70),     # a rung lower
            self.fill("sell", 6_100.0, 40),    # partial take
            self.fill("sell", 10_050.0, 0),    # flat: +50
        ]
        assert tu._round_trips(fills, 10_000.0) == (1, 1)

    def test_each_trip_is_measured_from_the_cash_it_opened_with(self):
        fills = [
            self.fill("buy", 5_000.0, 50),
            self.fill("sell", 9_900.0, 0),     # −100
            self.fill("buy", 4_900.0, 50),
            self.fill("sell", 9_920.0, 0),     # +20 on the 9,900 it opened with
            self.fill("buy", 4_920.0, 50),
            self.fill("sell", 9_919.0, 0),     # −1, fees ate it
        ]
        assert tu._round_trips(fills, 10_000.0) == (3, 1)

    def test_a_position_still_open_at_the_end_is_not_a_round_trip(self):
        fills = [
            self.fill("buy", 5_000.0, 50),
            self.fill("sell", 10_100.0, 0),
            self.fill("buy", 5_100.0, 50),
        ]
        assert tu._round_trips(fills, 10_000.0) == (1, 1)

    def test_no_round_trips_is_no_win_rate_not_zero(self):
        assert tu._round_trips([], 10_000.0) == (0, 0)
        assert tu.win_rate({"round_trips": 0, "wins": 0}) is None
        assert tu.win_rate({"round_trips": 3, "wins": 2}) == pytest.approx(66.7)


# --- the job store ----------------------------------------------------------


@pytest.fixture()
def tuning_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(tu, "TUNING_DIR", tmp_path / "tuning")
    return tmp_path / "tuning"


class TestStore:
    def test_a_job_whose_worker_is_gone_is_failed_not_stuck(self, tuning_dir):
        record = tu.submit(spec(), launch=False)
        record["pid"] = 2 ** 22 + 12345  # no such process
        tu._write(record)
        [job] = tu.list_jobs()
        assert job["status"] == tu.FAILED and "died" in job["error"]

    def test_an_invalid_spec_is_never_stored(self, tuning_dir):
        with pytest.raises(ValueError):
            tu.submit(spec(axes=[]), launch=False)
        assert tu.list_jobs() == []

    def test_a_dataset_is_removed_with_its_cells_and_the_pick_re_read(self, tuning_dir):
        record = job_record({
            "w1": [scored({"buy_k": 0.3}, 900.0), scored({"buy_k": 0.5}, 400.0)],
            "w2": [scored({"buy_k": 0.3}, -700.0), scored({"buy_k": 0.5}, 350.0)],
        })
        tu._write(tu.refresh_pick(record))
        assert record["best"]["overrides"] == {"buy_k": 0.5}
        after = tu.remove_dataset("j", "w2")
        assert tu.dataset_names(after["spec"]) == ["w1"] and set(after["cells"]) == {"w1"}
        assert after["best"]["overrides"] == {"buy_k": 0.3}
        assert after["progress"] == {"done": 3, "total": 3, "reused": 0}
        assert tu.get_job("j")["best"]["overrides"] == {"buy_k": 0.3}

    def test_the_last_dataset_is_not_removed(self, tuning_dir):
        tu._write(job_record({"w1": [scored({"buy_k": 0.3}, 1.0)]}))
        with pytest.raises(ValueError, match="at least one dataset"):
            tu.remove_dataset("j", "w1")

    def test_a_running_job_is_not_changed(self, tuning_dir):
        record = job_record({"w1": [], "w2": []})
        record.update(status=tu.RUNNING, pid=os.getpid())
        tu._write(record)
        with pytest.raises(ValueError, match="still running"):
            tu.remove_dataset("j", "w2")
        with pytest.raises(ValueError, match="still running"):
            tu.add_dataset("j", dataset("w3"), runs=[], launch=False)

    def test_removing_what_a_stopped_job_never_finished_finishes_it(self, tuning_dir):
        record = job_record({
            "w1": [scored({"buy_k": 0.3}, 1.0), scored({"buy_k": 0.5}, 2.0)],
            "w2": [scored({"buy_k": 0.3}, 1.0)],
        })
        record.update(status=tu.FAILED, error=tu.STOPPED_ERROR)
        tu._write(record)
        after = tu.remove_dataset("j", "w2")
        assert after["status"] == tu.FINISHED and after["error"] is None

    def test_a_dataset_is_added_with_the_jobs_own_settings(self, tuning_dir):
        tu._write(tu.refresh_pick(job_record({
            "w1": [scored({"buy_k": 0.3}, 1.0), scored({"buy_k": 0.5}, 2.0)],
        })))
        record = tu.add_dataset("j", dataset("w2", TEST_DAY), runs=[], launch=False)
        assert tu.dataset_names(record["spec"]) == ["w1", "w2"]
        assert record["spec"]["axes"] == [{"name": "buy_k", "values": [0.3, 0.5]}]
        assert record["status"] == tu.RUNNING and record["best"] is None
        assert record["progress"] == {"done": 3, "total": 6, "reused": 0}
        assert tu.missing_replays(record) == 3
        with pytest.raises(ValueError, match="twice"):
            record.update(status=tu.FINISHED)
            tu._write(record)
            tu.add_dataset("j", dataset("w2", TEST_DAY), runs=[], launch=False)


class TestUpgrade:
    """Jobs stored with a tune and a test dataset read as a job over both."""

    def old_record(self, sweep_test_grid=True):
        tune = [scored({"buy_k": 0.3}, 900.0), scored({"buy_k": 0.5}, 400.0)]
        test = [scored({"buy_k": 0.3}, -700.0), scored({"buy_k": 0.5}, 350.0)]
        return {
            "job_id": "old", "status": tu.FINISHED, "pid": None, "error": None,
            "created_at": "2026-09-23T18:19:47+00:00", "finished_at": None,
            "spec": {
                **{k: v for k, v in spec(axes=[{"name": "buy_k", "values": [0.3, 0.5]}]).items()
                   if k != "datasets"},
                "tune_dataset": dataset("w1"), "test_dataset": dataset("w2", TEST_DAY),
                "sweep_test_grid": sweep_test_grid, "metric": "worst_day",
            },
            "progress": {"done": 6, "total": 6},
            "cells": {tu.TUNE: tune, tu.TEST: test if sweep_test_grid else []},
            "baseline": {tu.TUNE: scored({}, 1.0), tu.TEST: scored({}, 2.0)},
            "best": tune[0],
            "best_test": test[0],
        }

    def test_both_datasets_become_the_jobs_and_the_pick_is_read_off_the_sum(self, tuning_dir):
        tu._write(self.old_record())
        [job] = tu.list_jobs()
        assert tu.dataset_names(job["spec"]) == ["w1", "w2"]
        assert "tune_dataset" not in job["spec"] and "best_test" not in job
        assert len(job["cells"]["w2"]) == 2 and job["baseline"]["w2"]["profit"] == 2.0
        assert job["best"]["overrides"] == {"buy_k": 0.5}  # was 0.3, the tuning week's best
        assert job["progress"]["total"] == 6 and tu.missing_replays(job) == 0

    def test_an_unswept_test_grid_keeps_the_picks_replay_and_leaves_holes(self, tuning_dir):
        tu._write(self.old_record(sweep_test_grid=False))
        job = tu.get_job("old")
        assert [c["overrides"] for c in job["cells"]["w2"]] == [{"buy_k": 0.3}]
        assert tu.missing_replays(job) == 1
        # Only 0.3 is summed so far, so it is the pick until the rest is swept.
        assert job["best"]["overrides"] == {"buy_k": 0.3}

    def test_the_new_shape_is_left_alone(self):
        record = job_record({"w1": []})
        assert tu._upgrade(record) is record


# --- a whole job, end to end ------------------------------------------------

OPEN_UTC = {day: datetime(day.year, day.month, day.day, 13, 30, tzinfo=timezone.utc)
            for day in (TUNE_DAY, TEST_DAY)}
# 105.50 predicted high on a $4 ADR: buy_k 0.75 -> 102.50, sell_k 0.10 -> 105.10.
FORECAST = {"pred_high": 105.5, "pred_low": 99.0, "prev_avg": 102.0,
            "adr14_abs": 4.0, "or_high": 104.1, "or_low": 103.9}


def _bar(ts: datetime, close: float) -> dict:
    return {"t": ts.strftime("%Y-%m-%dT%H:%M:%SZ"), "o": close - 0.05,
            "h": close + 0.1, "l": close - 0.1, "c": close, "v": 1000.0}


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """Two sessions that dip to ~102.4 and recover to ~105.6, plus daily history."""
    monkeypatch.setattr(sim_data, "STORE_DIR", tmp_path / "store")
    monkeypatch.setattr(sim_data, "MANIFEST_PATH", tmp_path / "datasets.json")
    prices = (
        [104.0] * 5
        + [104.0 - 0.16 * (i + 1) for i in range(10)]
        + [102.4 + 0.20 * (i + 1) for i in range(16)]
        + [105.0] * 5
    )
    for day in (TUNE_DAY, TEST_DAY):
        bars = [_bar(OPEN_UTC[day] + timedelta(minutes=i), p) for i, p in enumerate(prices)]
        sim_data._write_gz(sim_data.bars_path(TICKER, day), bars)
    daily = [_bar(datetime(2026, 6, 15, tzinfo=timezone.utc) - timedelta(days=i), 99.0)
             for i in range(30, 0, -1)]
    sim_data._write_gz(sim_data.daily_path(TICKER), {
        "symbol": TICKER, "start": "2026-05-16", "end": "2026-06-16", "bars": daily,
    })
    return tmp_path


@pytest.fixture()
def stub_model(monkeypatch):
    dayrange = pytest.importorskip("agent_stonks.dayrange_model")
    bundle = {"model": None, "metadata": {}, "daily_models": [], "opening_minutes": 5,
              "lookback": 32, "trained_at": "2026-08-10"}
    monkeypatch.setattr(dayrange, "load_bundle", lambda ticker=None: bundle)
    monkeypatch.setattr(dayrange, "forecast_session", lambda *a, **k: dict(FORECAST))


class TestJob:
    def job_spec(self, **overrides):
        # The managed exit and the momentum confirmation off, so the grid is
        # about the two levels alone (the tape dives straight to the deeper
        # level), and the ADR unit because the tape below and the numbers
        # asserted on it are that arithmetic. No days off: TUNE_DAY is a
        # geopolitical-shock day in the calendar, which a default run sits out.
        base = AppleTraderConfig(ticker=TICKER, buy_k=0.75, sell_k=0.10,
                                 stop_gain_fraction=0.0, momentum_confirmation_bars=0,
                                 level_unit=UNIT_ADR, skip_events=())
        return spec(
            base=asdict(base),
            axes=[{"name": "buy_k", "values": [0.5, 0.75, 1.0]},
                  {"name": "sell_k", "values": [0.10, 0.60]}],
            workers=1,
            **overrides,
        )

    def test_sweeps_every_dataset_sums_and_picks(self, store, stub_model, tuning_dir):
        record = tu.submit(self.job_spec(), launch=False)
        done = tu.run_job(record["job_id"], progress=lambda m: None)

        assert done["status"] == tu.FINISHED
        assert done["progress"]["done"] == done["progress"]["total"] == (6 + 1) * 2
        cells = {tuple(c["overrides"].values()): c for c in done["cells"]["d"]}
        # A sell level at or below the buy level is a hole in the grid, not a replay.
        assert "invalid" in cells[(0.5, 0.60)]
        # A buy level the day never dips to never trades.
        assert cells[(1.0, 0.10)]["trades"] == 0 and cells[(1.0, 0.10)]["profit"] == 0.0
        # The deeper entry buys the same recovery lower, so it wins.
        assert cells[(0.75, 0.10)]["profit"] > cells[(0.5, 0.10)]["profit"] > 0
        assert len(done["cells"]["e"]) == 6
        assert all(tu.is_scored(done["baseline"][name]) for name in ("d", "e"))
        # The two sessions are the same tape, so the sum is each cell twice.
        best = done["best"]
        assert best["overrides"] == {"buy_k": 0.75, "sell_k": 0.10}
        assert best["profit"] == pytest.approx(2 * cells[(0.75, 0.10)]["profit"])
        assert best["datasets"] == best["datasets_up"] == 2

    def test_an_added_dataset_is_swept_alone(self, store, stub_model, tuning_dir):
        record = tu.submit(self.job_spec(datasets=[dataset("d")]), launch=False)
        first = tu.run_job(record["job_id"], progress=lambda m: None)
        swept_d = first["cells"]["d"]
        tu.add_dataset(record["job_id"], dataset("e", TEST_DAY), runs=[], launch=False)
        lines = []
        done = tu.run_job(record["job_id"], progress=lines.append)
        assert len(lines) == 6 + 1 + 1  # e's grid and baseline, then "finished"
        assert all(" e " in line for line in lines[:-1])
        assert done["cells"]["d"] == swept_d
        assert done["status"] == tu.FINISHED and tu.missing_replays(done) == 0
        assert done["progress"]["done"] == done["progress"]["total"] == 14
        assert done["best"]["datasets"] == 2

    def test_a_stopped_job_resumes_where_it_stopped(self, store, stub_model, tuning_dir):
        record = tu.submit(self.job_spec(), launch=False)
        done = tu.run_job(record["job_id"], progress=lambda m: None)
        dropped = done["cells"]["e"].pop()
        done.update(status=tu.FAILED, error=tu.STOPPED_ERROR)
        tu._write(done)
        tu.resume(record["job_id"], runs=[], launch=False)
        again = tu.run_job(record["job_id"], progress=lambda m: None)
        assert again["cells"]["e"][-1]["overrides"] == dropped["overrides"]
        assert again["status"] == tu.FINISHED and tu.missing_replays(again) == 0

    def test_a_cell_is_exactly_a_simulate_run(self, store, stub_model):
        """What the grid scores is what Simulate would have done with that config."""
        dataset = self.job_spec()["datasets"][0]
        base = self.job_spec()["base"]
        cell = tu.evaluate(dataset, base, {"buy_k": 0.75}, 10_000.0)

        config = AppleTraderConfig(**{**base, "buy_k": 0.75})
        market = SimMarket([TICKER], [TUNE_DAY], dataset["feed"])
        result = SimulationEngine(market, SimulationConfig(
            personality=APPLE_TRADER_KEY, provider="rules", model="x", api_key="",
            symbols=[TICKER], days=[TUNE_DAY], starting_cash=10_000.0,
            rule_config=asdict(config), feed=dataset["feed"],
        )).run()
        assert cell["profit"] == round(result.final_value - 10_000.0, 2)


# --- cells a stored run already answers -------------------------------------


def stored_run(
    config: AppleTraderConfig,
    days: "list[date]",
    *,
    feed: str = sim_data.DEFAULT_FEED,
    cash: float = 10_000.0,
    run_id: str = "20260616-120000-aaaaaa",
    rule_config: "dict | None" = None,
    created_at: "str | None" = None,
    **fields,
) -> dict:
    """A saved run record shaped exactly as `results.save_run` writes one:
    the engine's result spread over the top level, plus the run's identity.

    Stamped *now* unless told otherwise, so it is newer than every file a
    replay of it reads -- a run saved before one of those changed is stale and
    deliberately not reusable (`run_is_stale`).
    """
    profit_per_day = 100.0
    record = {
        "run_id": run_id,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "dataset": "d",
        "config_summary": {
            "personality": APPLE_TRADER_KEY,
            "provider": "rules",
            "model": "x",
            "symbols": [config.ticker],
            "days": [d.isoformat() for d in days],
            "feed": feed,
            "starting_cash": cash,
            "cycle_minutes": 5,
            "rule_based": True,
            "rule_config": asdict(config) if rule_config is None else rule_config,
        },
        "decisions": [
            {"action": "buy", "status": "filled", "ts": et(day, "10:00")} for day in days
        ],
        "agent_log": [],
        "equity": [
            {"ts": et(day, "15:59"), "value": cash + profit_per_day * (i + 1)}
            for i, day in enumerate(days)
        ],
        "cycles_run": len(days),
        "final_value": cash + profit_per_day * len(days),
        "starting_cash": cash,
        "error": None,
        "interrupted": False,
        "summary": {},
        "judge": None,
    }
    record.update(fields)
    return record


class TestPriorRuns:
    """Which stored simulation runs may stand in for a cell of a grid.

    A cell *is* a Simulate run of that configuration -- same engine, same
    trader, same tape -- so a matching stored run answers it and replaying it
    would only produce the same number again. What has to match is every field
    of the decoded configuration plus the sessions, the tape and the starting
    cash; anything looser would put a different replay's number in the cell.
    """

    def job(self, **overrides):
        base = AppleTraderConfig(ticker=TICKER, buy_k=0.75, sell_k=0.10)
        return spec(**{
            "base": asdict(base),
            "axes": [{"name": "buy_k", "values": [0.5, 0.75]}],
            "datasets": [dataset("d")],
            **overrides,
        })

    def cell(self, job, overrides, runs):
        dataset = job["datasets"][0]
        return tu.prior_cell(tu.index_runs(runs), job, dataset, overrides)

    def config(self, job, **overrides):
        return tu.make_config(job["base"], overrides)

    def test_a_run_of_the_same_replay_answers_the_cell(self):
        job = self.job()
        run = stored_run(self.config(job, buy_k=0.5), [TUNE_DAY])
        cell = self.cell(job, {"buy_k": 0.5}, [run])
        assert cell["overrides"] == {"buy_k": 0.5}
        assert cell["run_id"] == run["run_id"] and cell["profit"] == 100.0

    def test_a_run_of_another_cell_does_not_answer_this_one(self):
        job = self.job()
        run = stored_run(self.config(job, buy_k=0.5), [TUNE_DAY])
        assert self.cell(job, {"buy_k": 0.75}, [run]) is None

    @pytest.mark.parametrize("field,value", [
        ("position_pct", 50.0), ("sell_k", 0.2), ("min_win_k", 0.3),
        # The signature leaves this one out (`SWEEPABLE` says why), so two runs
        # that differ in it share a Results row -- but they are not the same
        # replay, and a cell must not be answered by the other one.
        ("flatten_before_close_min", 9),
    ])
    def test_a_setting_the_grid_does_not_sweep_must_still_match(self, field, value):
        job = self.job()
        run = stored_run(self.config(job, buy_k=0.5, **{field: value}), [TUNE_DAY])
        assert self.cell(job, {"buy_k": 0.5}, [run]) is None

    @pytest.mark.parametrize("other", [
        {"days": [str(TEST_DAY)]},                    # other sessions
        {"days": [str(TUNE_DAY), str(TEST_DAY)]},     # more sessions
        {"feed": "iex"},                              # another tape
    ])
    def test_a_run_over_a_different_dataset_is_a_different_replay(self, other):
        job = self.job()
        config = self.config(job, buy_k=0.5)
        days = [date.fromisoformat(d) for d in other.get("days", [str(TUNE_DAY)])]
        run = stored_run(config, days, feed=other.get("feed", sim_data.DEFAULT_FEED))
        assert self.cell(job, {"buy_k": 0.5}, [run]) is None

    def test_a_run_on_other_starting_cash_is_a_different_replay(self):
        job = self.job()
        run = stored_run(self.config(job, buy_k=0.5), [TUNE_DAY], cash=50_000.0)
        assert self.cell(job, {"buy_k": 0.5}, [run]) is None

    def test_the_session_order_is_not_part_of_the_dataset(self):
        """The tape a replay reads, not the order a form happened to list it in."""
        job = self.job(datasets=[dataset("d", days=[str(TEST_DAY), str(TUNE_DAY)])])
        run = stored_run(self.config(job, buy_k=0.5), [TUNE_DAY, TEST_DAY])
        assert self.cell(job, {"buy_k": 0.5}, [run])["run_id"] == run["run_id"]

    def test_a_run_stored_before_a_field_existed_matches_what_it_meant_then(self):
        """A record without `breach_update` describes a run whose levels never
        moved intraday, so it answers the cell that has the update off -- and
        not the one at today's default, which is a different strategy."""
        job = self.job()
        run = stored_run(
            self.config(job, buy_k=0.5), [TUNE_DAY],
            rule_config={k: v for k, v in {**job["base"], "buy_k": 0.5}.items()
                         if k != "breach_update"},
        )
        switched_off = self.job(base={**job["base"], "buy_k": 0.5, "breach_update": "off"})
        assert self.cell(switched_off, {}, [run])["run_id"] == run["run_id"]
        assert AppleTraderConfig().breach_update != "off"  # today's default is not that
        assert self.cell(job, {"buy_k": 0.5}, [run]) is None

    @pytest.mark.parametrize("damage", [
        {"error": "no simulated tape price"},
        {"interrupted": True},
        {"cycles_run": 0},
    ])
    def test_a_run_that_did_not_finish_is_never_reused(self, damage):
        job = self.job()
        run = stored_run(self.config(job, buy_k=0.5), [TUNE_DAY], **damage)
        assert self.cell(job, {"buy_k": 0.5}, [run]) is None

    def test_an_llm_run_is_never_reused(self):
        job = self.job()
        run = stored_run(self.config(job, buy_k=0.5), [TUNE_DAY])
        run["config_summary"].update(rule_based=False, rule_config=None)
        assert self.cell(job, {"buy_k": 0.5}, [run]) is None

    def test_a_rule_set_this_build_cannot_decode_is_never_reused(self):
        """A record naming a removed model is refused rather than replayed on
        the model that is left, so it cannot answer a cell either."""
        job = self.job()
        config = self.config(job, buy_k=0.5)
        run = stored_run(
            config, [TUNE_DAY], rule_config={**asdict(config), "buy_k": 0.1, "sell_k": 0.5},
        )
        assert self.cell(job, {"buy_k": 0.5}, [run]) is None

    def test_the_newest_of_two_identical_replays_wins(self):
        job = self.job()
        config = self.config(job, buy_k=0.5)
        newer = stored_run(config, [TUNE_DAY], run_id="new")
        older = stored_run(config, [TUNE_DAY], run_id="old")
        assert self.cell(job, {"buy_k": 0.5}, [newer, older])["run_id"] == "new"

    def test_an_invalid_combination_is_a_hole_not_a_lookup(self):
        job = self.job(axes=[{"name": "sell_k", "values": [0.9]}])
        run = stored_run(self.config(job), [TUNE_DAY])
        assert self.cell(job, {"sell_k": 0.9}, [run]) is None

    def test_the_grid_and_the_baseline_are_both_looked_up(self):
        job = self.job()
        runs = [
            stored_run(self.config(job, buy_k=0.5), [TUNE_DAY], run_id="cell"),
            stored_run(self.config(job), [TUNE_DAY], run_id="base"),
        ]
        found = tu.prior_cells(job, runs)["d"]
        assert found[tu.overrides_key({"buy_k": 0.5})]["run_id"] == "cell"
        assert found[tu.overrides_key({})]["run_id"] == "base"
        # buy_k 0.75 is the base's own value, so the baseline run answers that
        # cell too -- same configuration, same sessions, same replay.
        assert found[tu.overrides_key({"buy_k": 0.75})]["run_id"] == "base"

    def test_every_dataset_is_looked_up_on_its_own_sessions(self):
        job = self.job(datasets=[dataset("d"), dataset("e", TEST_DAY)])
        runs = [
            stored_run(self.config(job, buy_k=0.5), [TEST_DAY], run_id="cell"),
            stored_run(self.config(job), [TEST_DAY], run_id="base"),
        ]
        found = tu.prior_cells(job, runs)
        assert found["d"] == {}
        assert found["e"][tu.overrides_key({"buy_k": 0.5})]["run_id"] == "cell"
        assert found["e"][tu.overrides_key({})]["run_id"] == "base"
        # Or only the dataset about to be added.
        assert set(tu.prior_cells(job, runs, [job["datasets"][1]])) == {"e"}

    def test_reuse_can_be_switched_off(self):
        job = self.job(reuse_runs=False)
        run = stored_run(self.config(job, buy_k=0.5), [TUNE_DAY])
        assert tu.prior_cells(job, [run]) == {"d": {}}

    def test_a_run_older_than_the_data_it_read_is_never_reused(self, store):
        """The `store` fixture writes this session's bars and daily history
        now, so a run stamped before them describes a replay off data that has
        since been rewritten -- a different forecast, different levels,
        different fills."""
        job = self.job()
        config = self.config(job, buy_k=0.5)
        fresh = stored_run(config, [TUNE_DAY])
        stale = stored_run(config, [TUNE_DAY], created_at="2026-06-01T12:00:00+00:00")
        assert tu.run_is_stale(stale) and not tu.run_is_stale(fresh)
        assert self.cell(job, {"buy_k": 0.5}, [stale]) is None
        assert self.cell(job, {"buy_k": 0.5}, [fresh])["run_id"] == fresh["run_id"]

    def test_a_run_with_no_usable_timestamp_is_never_reused(self):
        job = self.job()
        run = stored_run(self.config(job, buy_k=0.5), [TUNE_DAY], created_at="whenever")
        assert self.cell(job, {"buy_k": 0.5}, [run]) is None

    def test_a_run_of_a_model_this_build_cannot_run_is_never_reused(self):
        """A record naming a removed model is refused rather than replayed on
        the model that is left, so it cannot answer a cell either."""
        job = self.job()
        config = self.config(job, buy_k=0.5)
        run = stored_run(
            config, [TUNE_DAY], rule_config={**asdict(config), "model_key": "persistence"},
        )
        assert tu.run_replay_key(run) is None

    def test_runs_for_pair_is_one_symbol_and_one_saved_model(self):
        job = self.job()
        config = self.config(job)
        mine = stored_run(config, [TUNE_DAY], run_id="mine")
        other_symbol = stored_run(config, [TUNE_DAY], run_id="sym")
        other_symbol["config_summary"]["rule_config"]["ticker"] = "GOOGL"
        other_model = stored_run(config, [TUNE_DAY], run_id="mod")
        other_model["config_summary"]["rule_config"]["model_key"] = "something-else"
        llm = stored_run(config, [TUNE_DAY], run_id="llm")
        llm["config_summary"].update(rule_based=False)
        found = tu.runs_for_pair(
            [mine, other_symbol, other_model, llm], TICKER, config.model_key
        )
        assert [r["run_id"] for r in found] == ["mine"]


class TestDerivedGrids:
    """Tuning grids nobody submitted, read out of the run store.

    The mirror of `TestPriorRuns`: that asks which runs answer a grid somebody
    described, this asks which grids the runs describe by themselves -- runs
    agreeing on everything but one or two tunable fields, over the same
    sessions, tape and starting cash.
    """

    def base(self, **overrides) -> AppleTraderConfig:
        return AppleTraderConfig(**{
            "ticker": TICKER, "buy_k": 0.75, "sell_k": 0.10,
            # A legacy record's take, which still reads back as the grid it was.
            "momentum_drop": 1.0, "momentum_confirmation_bars": 0,
            **overrides,
        })

    def runs(self, configs, days=(TUNE_DAY,), **kwargs) -> "list[dict]":
        """Newest first, as `results.list_runs` hands them over."""
        return [
            stored_run(config, list(days), run_id=f"r{i}", **kwargs)
            for i, config in enumerate(configs)
        ]

    def test_runs_differing_in_one_field_are_a_one_axis_grid(self):
        runs = self.runs([self.base(buy_k=k) for k in (0.9, 0.8, 0.7)])
        [job] = tu.derived_jobs(runs)
        assert job["status"] == tu.DERIVED and tu.is_derived(job)
        assert job["spec"]["axes"] == [{"name": "buy_k", "values": [0.7, 0.8, 0.9]}]
        assert [c["overrides"]["buy_k"] for c in job["cells"]["d"]] == [0.7, 0.8, 0.9]
        assert all(c["run_id"] for c in job["cells"]["d"])

    def test_a_two_axis_grid_keeps_its_holes(self):
        """The point of the whole thing: the lattice is the product of the
        values seen, so a combination nobody ran is a blank square."""
        runs = self.runs([
            self.base(buy_k=b, sell_k=s)
            for b, s in ((0.7, 0.1), (0.7, 0.2), (0.9, 0.1))
        ])
        [job] = tu.derived_jobs(runs)
        assert [a["values"] for a in job["spec"]["axes"]] == [[0.7, 0.9], [0.1, 0.2]]
        assert len(job["cells"]["d"]) == 3
        assert len(tu.grid(job["spec"]["axes"])) == 4
        assert job["progress"] == {"done": 4, "total": 5, "reused": 4}

    def test_a_regular_sweep_with_a_step_missing_shows_the_hole(self):
        """Observed values alone can never leave a gap on one axis — every
        value is there because a run had it — so the swept step is inferred."""
        runs = self.runs([
            self.base(momentum_drop=m) for m in (0.0, 0.4, 0.8, 1.6, 2.0)
        ])
        [job] = tu.derived_jobs(runs)
        assert job["spec"]["axes"][0]["values"] == [0.0, 0.4, 0.8, 1.2, 1.6, 2.0]
        assert len(job["cells"]["d"]) == 5
        assert job["progress"] == {"done": 6, "total": 7, "reused": 6}

    def test_settings_that_are_not_a_sweep_are_left_alone(self):
        """Three values somebody tried are three settings, not a fine sweep
        riddled with holes."""
        runs = self.runs([self.base(buy_k=k) for k in (0.40, 0.45, 0.75)])
        [job] = tu.derived_jobs(runs)
        assert job["spec"]["axes"][0]["values"] == [0.40, 0.45, 0.75]

    def test_a_filled_value_is_carried_through_as_it_was_stored(self):
        """A cell is looked up by its overrides and 0.7 + 0.1 is not 0.8, so
        the inferred axis has to hand back the stored floats."""
        values = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        [job] = tu.derived_jobs(self.runs([self.base(buy_k=k) for k in values]))
        axis = job["spec"]["axes"][0]
        assert axis["values"] == values
        assert all(c["overrides"]["buy_k"] in axis["values"] for c in job["cells"]["d"])
        assert job["best"] is not None  # pick_best indexes the axis by value

    def test_an_integer_axis_stays_integer(self):
        runs = self.runs([
            self.base(flatten_before_close_min=m) for m in (2, 4, 8, 10)
        ])
        [job] = tu.derived_jobs(runs)
        assert job["spec"]["axes"][0]["values"] == [2, 4, 6, 8, 10]
        assert all(isinstance(v, int) for v in job["spec"]["axes"][0]["values"])

    def test_a_field_that_never_moved_is_not_an_axis(self):
        runs = self.runs([self.base(buy_k=k) for k in (0.7, 0.8, 0.9)])
        [job] = tu.derived_jobs(runs)
        assert [a["name"] for a in job["spec"]["axes"]] == ["buy_k"]

    def test_runs_over_different_sessions_are_different_grids(self):
        configs = [self.base(buy_k=k) for k in (0.7, 0.8, 0.9)]
        runs = self.runs(configs) + [
            stored_run(c, [TEST_DAY], run_id=f"t{i}") for i, c in enumerate(configs)
        ]
        jobs = tu.derived_jobs(runs)
        assert len(jobs) == 2
        assert {j["spec"]["datasets"][0]["days"][0] for j in jobs} == {
            str(TUNE_DAY), str(TEST_DAY)
        }

    def test_runs_at_different_starting_cash_are_different_grids(self):
        configs = [self.base(buy_k=k) for k in (0.7, 0.8, 0.9)]
        runs = self.runs(configs) + [
            stored_run(c, [TUNE_DAY], run_id=f"c{i}", cash=50_000.0)
            for i, c in enumerate(configs)
        ]
        assert len(tu.derived_jobs(runs)) == 2

    def test_a_setting_outside_the_axes_splits_the_grid(self):
        """Two momentum sweeps at two breach policies are two grids, not one
        surface with each cell run twice."""
        runs = self.runs([
            self.base(momentum_drop=m, breach_update=b)
            for b in ("off", "extreme") for m in (0.0, 1.0, 2.0)
        ])
        jobs = tu.derived_jobs(runs)
        assert len(jobs) == 2
        assert {j["spec"]["base"]["breach_update"] for j in jobs} == {"off", "extreme"}
        assert all([a["name"] for a in j["spec"]["axes"]] == ["momentum_drop"] for j in jobs)

    def test_a_grid_inside_a_denser_one_is_not_offered_as_well(self):
        """A full two-axis sweep would otherwise also appear as each of its
        rows and each of its columns."""
        runs = self.runs([
            self.base(buy_k=b, sell_k=s)
            for b in (0.7, 0.8, 0.9) for s in (0.1, 0.2, 0.3)
        ])
        [job] = tu.derived_jobs(runs)
        assert [a["name"] for a in job["spec"]["axes"]] == ["buy_k", "sell_k"]
        assert len(job["cells"]["d"]) == 9

    def test_too_few_runs_to_be_a_sweep(self):
        runs = self.runs([self.base(buy_k=k) for k in (0.7, 0.9)])
        assert tu.MIN_DERIVED_CELLS == 3 and tu.derived_jobs(runs) == []

    def test_runs_that_could_not_answer_a_cell_never_invent_a_grid(self):
        """Stale, unfinished and undecodable runs are refused as cells
        (`run_replay_key`), so they cannot make a grid either."""
        configs = [self.base(buy_k=k) for k in (0.7, 0.8, 0.9)]
        runs = self.runs(configs)
        runs[0]["created_at"] = "2026-06-01T12:00:00+00:00"  # older than its inputs
        runs[1]["interrupted"] = True
        assert tu.derived_jobs(runs) == []

    def test_the_newest_run_is_the_base_and_the_baseline(self):
        runs = self.runs([self.base(buy_k=k) for k in (0.9, 0.8, 0.7)])  # r0 is newest
        [job] = tu.derived_jobs(runs)
        assert job["spec"]["base"]["buy_k"] == 0.9
        assert job["baseline"]["d"]["run_id"] == "r0"
        # The baseline reads as a job's does -- the configuration with nothing
        # overridden -- whichever cell of the grid it happens to sit on.
        assert job["baseline"]["d"]["overrides"] == {}
        assert tu.dataset_names(job["spec"]) == ["d"] and set(job["cells"]) == {"d"}

    def test_the_best_cell_is_the_highest_profit(self):
        runs = self.runs([self.base(buy_k=k) for k in (0.7, 0.8, 0.9)])
        # `stored_run` pays 100 a session, so every cell ties; the pick then
        # goes to the first in grid order, as `pick_best` promises.
        [job] = tu.derived_jobs(runs)
        assert job["best"]["overrides"] == {"buy_k": 0.7}

    def test_an_id_survives_a_new_run_filling_a_hole(self):
        """The selection has to hold across a rerun, so the id is hashed from
        what defines the grid rather than from the runs in it."""
        configs = [self.base(buy_k=b, sell_k=s)
                   for b, s in ((0.7, 0.1), (0.7, 0.2), (0.9, 0.1))]
        [before] = tu.derived_jobs(self.runs(configs))
        filled = self.runs([*configs, self.base(buy_k=0.9, sell_k=0.2)])
        [after] = tu.derived_jobs(filled)
        assert before["job_id"] == after["job_id"]
        assert len(after["cells"]["d"]) == 4

    def test_the_grid_is_submittable_as_a_real_job(self, tuning_dir):
        """The one action a derived grid offers: it becomes a job that reuses
        every cell already on it and replays only the holes."""
        runs = self.runs([
            self.base(buy_k=b, sell_k=s)
            for b, s in ((0.7, 0.1), (0.7, 0.2), (0.9, 0.1))
        ])
        [job] = tu.derived_jobs(runs)
        assert tu.validate(job["spec"]) is None
        record = tu.submit(job["spec"], launch=False, runs=runs)
        assert record["progress"]["reused"] == 4  # 3 cells and the baseline
        assert record["progress"]["total"] == 5   # the 4-cell grid and the baseline


class TestSeededJob:
    """A job starts with what the run store already answers and replays the rest."""

    def job(self, **overrides):
        base = AppleTraderConfig(ticker=TICKER, buy_k=0.75, sell_k=0.10)
        return spec(**{
            "base": asdict(base),
            "axes": [{"name": "buy_k", "values": [0.5, 0.75]}],
            "datasets": [dataset("d")],
            **overrides,
        })

    def test_a_seeded_cell_is_stored_and_counted_as_done(self, tuning_dir):
        job = self.job()
        run = stored_run(tu.make_config(job["base"], {"buy_k": 0.5}), [TUNE_DAY])
        record = tu.submit(job, launch=False, runs=[run])
        [cell] = record["cells"]["d"]
        assert cell["overrides"] == {"buy_k": 0.5} and cell["run_id"] == run["run_id"]
        # The total is still the whole grid: the job ran it, it just did not
        # have to replay this one.
        assert record["progress"] == {"done": 1, "total": 3, "reused": 1}

    def test_the_worker_replays_only_the_holes(self, store, stub_model, tuning_dir):
        job = self.job()
        run = stored_run(tu.make_config(job["base"], {"buy_k": 0.5}), [TUNE_DAY])
        record = tu.submit(job, launch=False, runs=[run])
        done = tu.run_job(record["job_id"], progress=lambda m: None)

        assert done["status"] == tu.FINISHED
        assert done["progress"]["done"] == done["progress"]["total"] == 3
        by_cell = {tuple(c["overrides"].values()): c for c in done["cells"]["d"]}
        assert by_cell[(0.5,)]["run_id"] == run["run_id"]   # never replayed
        assert "run_id" not in by_cell[(0.75,)]             # swept here
        assert tu.reused_count(done) == 1

    def test_a_job_the_store_answers_entirely_replays_nothing(
        self, store, stub_model, tuning_dir
    ):
        job = self.job(datasets=[dataset("d"), dataset("e", TEST_DAY)])
        runs = [stored_run(tu.make_config(job["base"], o), days)
                for o in ({"buy_k": 0.5}, {"buy_k": 0.75}, {})
                for days in ([TUNE_DAY], [TEST_DAY])]
        record = tu.submit(job, launch=False, runs=runs)
        lines = []
        done = tu.run_job(record["job_id"], progress=lines.append)
        assert lines == [f"Tuning job {record['job_id']} finished."]
        assert done["progress"] == {"done": 6, "total": 6, "reused": 6}
        assert tu.reused_count(done) == 6
        assert done["best"]["reused"] == 2

    def test_an_added_dataset_is_seeded_from_the_store(self, tuning_dir):
        job = self.job()
        tu.submit(job, launch=False, runs=[])
        [record] = tu.list_jobs()
        record.update(status=tu.FINISHED)
        tu._write(record)
        run = stored_run(tu.make_config(job["base"], {"buy_k": 0.5}), [TEST_DAY])
        added = tu.add_dataset(record["job_id"], dataset("e", TEST_DAY), runs=[run], launch=False)
        assert [c["run_id"] for c in added["cells"]["e"]] == [run["run_id"]]
        assert added["progress"] == {"done": 1, "total": 6, "reused": 1}

    def test_nothing_stored_leaves_the_job_exactly_as_it_was(
        self, store, stub_model, tuning_dir
    ):
        record = tu.submit(self.job(), launch=False, runs=[])
        done = tu.run_job(record["job_id"], progress=lambda m: None)
        assert done["progress"] == {"done": 3, "total": 3, "reused": 0}
        assert tu.reused_count(done) == 0
        assert len(done["cells"]["d"]) == 2 and tu.is_scored(done["baseline"]["d"])

    def test_a_reused_cell_is_what_the_sweep_would_have_produced(
        self, store, stub_model, tuning_dir, monkeypatch, tmp_path
    ):
        """The claim the whole feature rests on: a stored run of a cell's
        configuration scores as that cell, down to the session."""
        from simlab import results as sim_results

        monkeypatch.setattr(sim_results, "RUNS_DIR", tmp_path / "runs")
        job = self.job()
        overrides = {"buy_k": 0.5}
        swept = tu.evaluate(job["datasets"][0], job["base"], overrides, job["starting_cash"])

        config = tu.make_config(job["base"], overrides)
        market = SimMarket([TICKER], [TUNE_DAY], job["datasets"][0]["feed"])
        result = SimulationEngine(market, SimulationConfig(
            personality=APPLE_TRADER_KEY, provider="rules", model="x", api_key="",
            symbols=[TICKER], days=[TUNE_DAY], starting_cash=job["starting_cash"],
            rule_config=asdict(config), feed=job["datasets"][0]["feed"],
        )).run()
        saved = sim_results.save_run(result, {}, dataset_name="d")

        reused = tu.prior_cell(
            tu.index_runs([saved]), job, job["datasets"][0], overrides
        )
        assert reused["run_id"] == saved["run_id"]
        for field in ("profit", "return_pct", "trades", "sells", "days_up", "worst_day",
                      "daily", "no_forecast_days"):
            assert reused[field] == swept[field], field


# --- the replay speed-ups ---------------------------------------------------


def naive_daily_bars_at(market, symbol, t):
    """`SimMarket.daily_bars_at` as it was before the per-day index."""
    today = t.astimezone(MARKET_TZ).date().isoformat()
    series = market.series[symbol]
    out = [b for b in series.daily_bars if str(b.get("t", ""))[:10] < today]
    todays = [b for b, ts in zip(series.minute_bars, series.minute_ts)
              if ts.astimezone(MARKET_TZ).date().isoformat() == today
              and ts + timedelta(seconds=BAR_SEC) <= t]
    if todays:
        out.append({"t": f"{today}T05:00:00Z", "o": float(todays[0]["o"]),
                    "h": max(float(b["h"]) for b in todays),
                    "l": min(float(b["l"]) for b in todays),
                    "c": float(todays[-1]["c"]),
                    "v": sum(float(b.get("v") or 0.0) for b in todays)})
    return out


def naive_day_volume(market, symbol, t):
    """`SimulationEngine._day_volume` as it was before the per-day index."""
    series = market.series[symbol]
    today = t.astimezone(MARKET_TZ).date()
    total = 0.0
    for bar, ts in zip(series.minute_bars, series.minute_ts):
        if ts + timedelta(seconds=BAR_SEC) > t:
            break
        if ts.astimezone(MARKET_TZ).date() == today:
            total += float(bar.get("v") or 0.0)
    return total


class TestReplaySpeedups:
    def market(self):
        return SimMarket([TICKER], [TUNE_DAY, TEST_DAY])

    def test_the_market_index_answers_exactly_as_the_scans_did(self, store):
        market = self.market()
        for day in (TUNE_DAY, TEST_DAY):
            steps = market.step_times(day)
            for t in [steps[0] - timedelta(minutes=1), *steps, steps[-1] + timedelta(hours=3)]:
                assert market.daily_bars_at(TICKER, t) == naive_daily_bars_at(market, TICKER, t)
                assert market.day_volume(TICKER, t) == naive_day_volume(market, TICKER, t)
                prior = [b for b in market.series[TICKER].daily_bars
                         if str(b["t"])[:10] < t.astimezone(MARKET_TZ).date().isoformat()]
                assert market.completed_daily_bars(TICKER, t) == prior
                assert market.prev_close(TICKER, t) == (float(prior[-1]["c"]) if prior else None)

    def test_the_market_hands_out_copies_of_its_cache(self, store):
        market = self.market()
        t = market.step_times(TEST_DAY)[3]
        market.completed_daily_bars(TICKER, t).append({"t": "junk"})
        assert market.completed_daily_bars(TICKER, t)[-1]["t"] != "junk"

    def test_the_replay_minute_frame_is_the_buffers_frame(self, store):
        """At every step the engine takes, row for row and column for column."""
        market = self.market()
        original = momentum_regime.minute_frame
        engine = SimulationEngine(market, SimulationConfig(
            personality=APPLE_TRADER_KEY, provider="rules", model="x", api_key="",
            symbols=[TICKER], days=market.days,
        ))
        sym_state = engine.app.sym(TICKER)
        with simulation_context(market):
            assert momentum_regime.minute_frame is not original
            for day in market.days:
                for t in market.step_times(day):
                    engine._apply_step(t)
                    fast = momentum_regime.minute_frame(sym_state)
                    slow = original(sym_state)
                    if len(slow):
                        pd.testing.assert_frame_equal(fast, slow)
                    else:
                        assert len(fast) == 0
        assert momentum_regime.minute_frame is original
