"""SimLab's parameter tuning for Apple Trader (simlab/tuning.py).

Four parts. The grid and the pick are pure functions, pinned on hand-made
cells. Scoring is pinned on a hand-made replay result. One job runs end to end
on a synthetic store with the day-range forecast stubbed, in process. And the
two replay speed-ups the grid depends on -- the market's per-day index and the
replay-backed minute frame -- are pinned against the scans they replaced,
at every step of a stored session, because a faster replay that answered
differently would make every tuned number wrong.
"""
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from agent_stonks import momentum_regime
from agent_stonks.apple_trader import APPLE_TRADER_KEY, AppleTraderConfig
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


def spec(**overrides):
    dataset = {"name": "d", "days": [str(TUNE_DAY)], "feed": sim_data.DEFAULT_FEED,
               "symbols": [TICKER]}
    base = {
        "base": asdict(AppleTraderConfig(ticker=TICKER)),
        "axes": [{"name": "buy_k", "values": [0.5, 0.7]}],
        "tune_dataset": dataset,
        "test_dataset": {**dataset, "name": "e", "days": [str(TEST_DAY)]},
        "starting_cash": 10_000.0,
        "sweep_test_grid": False,
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
            self.base(momentum_drop=0.0),
            [{"name": "take_fraction", "values": [0.3, 0.6, 1.0]}],
        )
        assert len(configs) == 1 and refused == []

    def test_the_collapse_is_visible_against_the_grid(self):
        axes = [{"name": "take_fraction", "values": [0.3, 0.6, 1.0]}]
        configs, refused = tu.expand(self.base(momentum_drop=0.0), axes)
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

    def test_at_most_two_parameters(self):
        axes = [
            {"name": n, "values": [1.0]}
            for n in ("buy_k", "sell_k", "stop_gain_fraction")
        ]
        assert "one or two" in tu.validate(spec(axes=axes))

    def test_the_same_parameter_twice_is_refused(self):
        axes = [{"name": "buy_k", "values": [0.5]}, {"name": "buy_k", "values": [0.7]}]
        assert "twice" in tu.validate(spec(axes=axes))

    def test_a_dataset_without_the_instrument_is_refused(self):
        other = {**spec()["tune_dataset"], "symbols": ["GOOGL"]}
        assert "does not carry AAPL" in tu.validate(spec(tune_dataset=other))

    def test_too_many_cells_is_refused(self):
        axes = [{"name": "buy_k", "values": list(range(30))},
                {"name": "sell_k", "values": list(range(30))}]
        assert "more than" in tu.validate(spec(axes=axes))

    def test_the_replay_count_follows_what_the_test_set_sweeps(self):
        assert tu.total_replays(spec()) == 2 + 1 + 1 + 1
        assert tu.total_replays(spec(sweep_test_grid=True)) == 2 + 1 + 2 + 1
        assert tu.total_replays(spec(test_dataset=None)) == 2 + 1


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

    def test_overlapping_days_are_named(self):
        assert tu.overlapping_days(["2026-06-15", "2026-06-16"], ["2026-06-16"]) == ["2026-06-16"]


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
        # The managed exit off, so the grid is about the two levels alone.
        base = AppleTraderConfig(ticker=TICKER, buy_k=0.75, sell_k=0.10,
                                 stop_gain_fraction=0.0, momentum_drop=0.0)
        return spec(
            base=asdict(base),
            axes=[{"name": "buy_k", "values": [0.5, 0.75, 1.0]},
                  {"name": "sell_k", "values": [0.10, 0.60]}],
            workers=1,
            **overrides,
        )

    def test_sweeps_picks_and_tests(self, store, stub_model, tuning_dir):
        record = tu.submit(self.job_spec(), launch=False)
        done = tu.run_job(record["job_id"], progress=lambda m: None)

        assert done["status"] == tu.FINISHED
        assert done["progress"]["done"] == done["progress"]["total"] == 6 + 1 + 1 + 1
        cells = {tuple(c["overrides"].values()): c for c in done["cells"][tu.TUNE]}
        # A sell level at or below the buy level is a hole in the grid, not a replay.
        assert "invalid" in cells[(0.5, 0.60)]
        # A buy level the day never dips to never trades.
        assert cells[(1.0, 0.10)]["trades"] == 0 and cells[(1.0, 0.10)]["profit"] == 0.0
        # The deeper entry buys the same recovery lower, so it wins.
        assert cells[(0.75, 0.10)]["profit"] > cells[(0.5, 0.10)]["profit"] > 0
        assert done["best"]["overrides"] == {"buy_k": 0.75, "sell_k": 0.10}
        assert tu.is_scored(done["best_test"]) and done["best_test"]["profit"] > 0
        assert tu.is_scored(done["baseline"][tu.TUNE])
        assert tu.is_scored(done["baseline"][tu.TEST])
        assert done["cells"][tu.TEST] == []

    def test_the_test_grid_is_swept_when_asked(self, store, stub_model, tuning_dir):
        record = tu.submit(self.job_spec(sweep_test_grid=True), launch=False)
        done = tu.run_job(record["job_id"], progress=lambda m: None)
        assert len(done["cells"][tu.TEST]) == 6
        assert done["best_test"]["overrides"] == done["best"]["overrides"]
        assert done["progress"]["done"] == done["progress"]["total"]

    def test_a_cell_is_exactly_a_simulate_run(self, store, stub_model):
        """What the grid scores is what Simulate would have done with that config."""
        dataset = self.job_spec()["tune_dataset"]
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
            "test_dataset": None,
            **overrides,
        })

    def cell(self, job, overrides, runs):
        dataset = job["tune_dataset"]
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

    @pytest.mark.parametrize("dataset", [
        {"days": [str(TEST_DAY)]},                    # other sessions
        {"days": [str(TUNE_DAY), str(TEST_DAY)]},     # more sessions
        {"feed": "iex"},                              # another tape
    ])
    def test_a_run_over_a_different_dataset_is_a_different_replay(self, dataset):
        job = self.job()
        config = self.config(job, buy_k=0.5)
        days = [date.fromisoformat(d) for d in dataset.get("days", [str(TUNE_DAY)])]
        run = stored_run(config, days, feed=dataset.get("feed", sim_data.DEFAULT_FEED))
        assert self.cell(job, {"buy_k": 0.5}, [run]) is None

    def test_a_run_on_other_starting_cash_is_a_different_replay(self):
        job = self.job()
        run = stored_run(self.config(job, buy_k=0.5), [TUNE_DAY], cash=50_000.0)
        assert self.cell(job, {"buy_k": 0.5}, [run]) is None

    def test_the_session_order_is_not_part_of_the_dataset(self):
        """The tape a replay reads, not the order a form happened to list it in."""
        job = self.job(tune_dataset={
            "name": "d", "days": [str(TEST_DAY), str(TUNE_DAY)],
            "feed": sim_data.DEFAULT_FEED, "symbols": [TICKER],
        })
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
        found = tu.prior_cells(job, runs)[tu.TUNE]
        assert found[tu.overrides_key({"buy_k": 0.5})]["run_id"] == "cell"
        assert found[tu.overrides_key({})]["run_id"] == "base"
        # buy_k 0.75 is the base's own value, so the baseline run answers that
        # cell too -- same configuration, same sessions, same replay.
        assert found[tu.overrides_key({"buy_k": 0.75})]["run_id"] == "base"

    def test_the_test_grid_is_only_looked_up_when_it_is_swept(self):
        dataset = {"name": "e", "days": [str(TEST_DAY)], "feed": sim_data.DEFAULT_FEED,
                   "symbols": [TICKER]}
        job = self.job(test_dataset=dataset)
        runs = [
            stored_run(self.config(job, buy_k=0.5), [TEST_DAY], run_id="cell"),
            stored_run(self.config(job), [TEST_DAY], run_id="base"),
        ]
        # The baseline is replayed either way, so it is always looked up.
        assert set(tu.prior_cells(job, runs)[tu.TEST]) == {tu.overrides_key({})}
        swept = tu.prior_cells({**job, "sweep_test_grid": True}, runs)[tu.TEST]
        assert swept[tu.overrides_key({"buy_k": 0.5})]["run_id"] == "cell"

    def test_reuse_can_be_switched_off(self):
        job = self.job(reuse_runs=False)
        run = stored_run(self.config(job, buy_k=0.5), [TUNE_DAY])
        assert tu.prior_cells(job, [run]) == {tu.TUNE: {}, tu.TEST: {}}

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


class TestSeededJob:
    """A job starts with what the run store already answers and replays the rest."""

    def job(self, **overrides):
        base = AppleTraderConfig(ticker=TICKER, buy_k=0.75, sell_k=0.10)
        return spec(**{
            "base": asdict(base),
            "axes": [{"name": "buy_k", "values": [0.5, 0.75]}],
            "test_dataset": None,
            **overrides,
        })

    def test_a_seeded_cell_is_stored_and_counted_as_done(self, tuning_dir):
        job = self.job()
        run = stored_run(tu.make_config(job["base"], {"buy_k": 0.5}), [TUNE_DAY])
        record = tu.submit(job, launch=False, runs=[run])
        [cell] = record["cells"][tu.TUNE]
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
        by_cell = {tuple(c["overrides"].values()): c for c in done["cells"][tu.TUNE]}
        assert by_cell[(0.5,)]["run_id"] == run["run_id"]   # never replayed
        assert "run_id" not in by_cell[(0.75,)]             # swept here
        assert tu.reused_count(done) == 1

    def test_the_picks_reused_test_replay_is_counted_once(self, store, stub_model, tuning_dir):
        """A swept test grid already holds the pick's test cell, so counting
        `best_test` again would claim more replays than the job has."""
        dataset = {"name": "e", "days": [str(TEST_DAY)], "feed": sim_data.DEFAULT_FEED,
                   "symbols": [TICKER]}
        job = self.job(test_dataset=dataset, sweep_test_grid=True)
        runs = [stored_run(tu.make_config(job["base"], o), days)
                for o in ({"buy_k": 0.5}, {"buy_k": 0.75}, {})
                for days in ([TUNE_DAY], [TEST_DAY])]
        record = tu.submit(job, launch=False, runs=runs)
        done = tu.run_job(record["job_id"], progress=lambda m: None)
        assert done["progress"] == {"done": 6, "total": 6, "reused": 6}
        assert tu.reused_count(done) == 6
        assert done["best_test"]["run_id"]

    def test_nothing_stored_leaves_the_job_exactly_as_it_was(
        self, store, stub_model, tuning_dir
    ):
        record = tu.submit(self.job(), launch=False, runs=[])
        done = tu.run_job(record["job_id"], progress=lambda m: None)
        assert done["progress"] == {"done": 3, "total": 3, "reused": 0}
        assert tu.reused_count(done) == 0
        assert len(done["cells"][tu.TUNE]) == 2 and tu.is_scored(done["baseline"][tu.TUNE])

    def test_a_reused_cell_is_what_the_sweep_would_have_produced(
        self, store, stub_model, tuning_dir, monkeypatch, tmp_path
    ):
        """The claim the whole feature rests on: a stored run of a cell's
        configuration scores as that cell, down to the session."""
        from simlab import results as sim_results

        monkeypatch.setattr(sim_results, "RUNS_DIR", tmp_path / "runs")
        job = self.job()
        overrides = {"buy_k": 0.5}
        swept = tu.evaluate(job["tune_dataset"], job["base"], overrides, job["starting_cash"])

        config = tu.make_config(job["base"], overrides)
        market = SimMarket([TICKER], [TUNE_DAY], job["tune_dataset"]["feed"])
        result = SimulationEngine(market, SimulationConfig(
            personality=APPLE_TRADER_KEY, provider="rules", model="x", api_key="",
            symbols=[TICKER], days=[TUNE_DAY], starting_cash=job["starting_cash"],
            rule_config=asdict(config), feed=job["tune_dataset"]["feed"],
        )).run()
        saved = sim_results.save_run(result, {}, dataset_name="d")

        reused = tu.prior_cell(
            tu.index_runs([saved]), job, job["tune_dataset"], overrides
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
