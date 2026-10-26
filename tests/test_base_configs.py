"""A pair's base configuration: a tuning result made the default for its
(model, ticker) pair in both apps (`agent_stonks.base_configs`).

The conftest points the store at a temporary file, so every test starts with
no base set: the shipped defaults.
"""
from dataclasses import asdict, replace

import pytest
from streamlit.testing.v1 import AppTest

from agent_stonks import apple_models, base_configs
from agent_stonks import apple_trader as at
from agent_stonks.apple_trader import AppleTraderConfig


def _base(**changes) -> AppleTraderConfig:
    """A HighLow AAPL configuration unlike the shipped one in the levels, the
    stop, the breaker, a shared knob and a field the form has no knob for."""
    config = AppleTraderConfig(ticker="AAPL", model_key="highlow")
    return replace(config, **{
        "buy_k": 0.85, "sell_k": 0.15, "stop_gain_fraction": 0.7, "min_win_k": 0.05,
        "momentum_confirmation_bars": 7, "limit_entry": False, **changes,
    })


class TestStore:
    def test_nothing_stored_is_nothing_to_find(self):
        assert base_configs.entry("highlow", "AAPL") is None
        assert base_configs.stored("highlow", "AAPL") is None
        assert base_configs.key_suffix("highlow", "AAPL") == ""

    def test_a_set_base_reads_back_per_pair(self):
        record = asdict(_base())
        base_configs.set_base(record, source={"job_id": "j1", "overrides": {"buy_k": 0.85}})
        found = base_configs.entry("highlow", "aapl")
        assert found["config"]["buy_k"] == 0.85
        assert found["source"]["job_id"] == "j1"
        assert found["set_at"]
        # Other pairs, the same symbol under another model included, have none.
        assert base_configs.entry("highlow2", "AAPL") is None
        assert base_configs.entry("highlow", "INTC") is None

    def test_the_key_suffix_follows_the_configuration(self):
        base_configs.set_base(asdict(_base()))
        first = base_configs.key_suffix("highlow", "AAPL")
        base_configs.set_base(asdict(_base()))  # the same again: same keys
        assert base_configs.key_suffix("highlow", "AAPL") == first
        base_configs.set_base(asdict(_base(buy_k=0.9)))
        assert base_configs.key_suffix("highlow", "AAPL") not in ("", first)

    def test_clear_goes_back_to_nothing(self):
        base_configs.set_base(asdict(_base()))
        assert base_configs.clear("highlow", "AAPL") is True
        assert base_configs.entry("highlow", "AAPL") is None
        assert base_configs.clear("highlow", "AAPL") is False

    def test_a_write_from_another_process_is_seen(self):
        """The live app reads what SimLab wrote: the cache is keyed on the file."""
        assert base_configs.stored("highlow", "AAPL") is None
        record = asdict(_base())
        other = {"version": 1, "pairs": {"highlow/AAPL": {"config": record}}}
        import json

        base_configs.BASE_PATH.write_text(json.dumps(other), encoding="utf-8")
        assert base_configs.stored("highlow", "AAPL")["buy_k"] == 0.85

    def test_an_unreadable_file_is_no_base(self):
        base_configs.BASE_PATH.write_text("{not json", encoding="utf-8")
        assert base_configs.entry("highlow", "AAPL") is None


class TestDefaults:
    def test_the_pairs_numbers_come_from_its_base(self):
        shipped = (at.dayrange_levels("AAPL", "highlow"), at.stop_for("AAPL", "highlow"))
        base_configs.set_base(asdict(_base()))
        assert at.dayrange_levels("AAPL", "highlow") == (0.85, 0.15)
        assert at.stop_for("AAPL", "highlow") == 0.7
        assert at.min_win_for("AAPL", "highlow") == 0.05
        # ... but not the shipped answer, nor another model's on the symbol.
        assert (
            at.dayrange_levels("AAPL", "highlow", use_base=False),
            at.stop_for("AAPL", "highlow", use_base=False),
        ) == shipped
        assert at.dayrange_levels("AAPL", "highlow2") != (0.85, 0.15)
        # A breaker asked for without a model is the symbol's table.
        assert at.min_win_for("AAPL") == at.APPLE_TRADER_MIN_WIN.get("AAPL", at.APPLE_TRADER_MIN_WIN_K)

    def test_a_config_left_to_its_defaults_resolves_them_from_the_base(self):
        base_configs.set_base(asdict(_base()))
        config = AppleTraderConfig(ticker="AAPL", model_key="highlow")
        assert (config.buy_k, config.sell_k) == (0.85, 0.15)
        assert config.stop_gain_fraction == 0.7
        assert config.min_win_k == 0.05

    def test_default_config_is_the_whole_base(self):
        assert at.default_config("AAPL", "highlow") == AppleTraderConfig(
            ticker="AAPL", model_key="highlow"
        )
        base_configs.set_base(asdict(_base()))
        assert at.default_config("AAPL", "highlow") == _base()
        assert at.shipped_config("AAPL", "highlow") != _base()

    def test_a_base_todays_config_refuses_falls_back_to_the_shipped_one(self):
        record = asdict(_base())
        record["breach_update"] = "no such policy"
        base_configs.set_base(record)
        assert at.default_config("AAPL", "highlow").breach_update != "no such policy"

    def test_a_stored_record_replays_as_it_was_run(self):
        """A record carries its numbers: setting a base moves no past run."""
        from simlab.rule_agents import rule_agent

        agent = rule_agent(at.APPLE_TRADER_KEY)
        run = asdict(AppleTraderConfig(ticker="AAPL", model_key="highlow"))
        before = agent.from_record(run)
        base_configs.set_base(asdict(_base()))
        assert agent.from_record(run) == before

    def test_orchestra_pairs_start_from_their_base(self):
        from agent_stonks.orchestra import build_orchestra_config

        base_configs.set_base(asdict(_base()))
        shared = AppleTraderConfig(min_win_k=0.2)
        race = build_orchestra_config(["AAPL:highlow", "INTC:highlow"], shared)
        aapl, intc = race.racers
        assert (aapl.buy_k, aapl.sell_k, aapl.momentum_confirmation_bars) == (0.85, 0.15, 7)
        assert (intc.buy_k, intc.sell_k) == at.dayrange_levels("INTC", "highlow", use_base=False)
        assert intc.momentum_confirmation_bars == at.APPLE_TRADER_MOMENTUM_CONFIRMATION_BARS
        # The circuit breaker is Orchestra's, one for every pair.
        assert aapl.min_win_k == intc.min_win_k == 0.2


class TestBaseError:
    def test_todays_configuration_can_be_a_base(self):
        assert at.base_error(_base()) is None

    @pytest.mark.parametrize("legacy", [
        {"stop_k": 0.2, "stop_gain_fraction": 0.0},
        {"momentum_fade_bars": 15, "momentum_confirmation_bars": 0},
        {"take_min_gain_fraction": 0.2},
        {"stop_under_next_buy": True},
    ])
    def test_a_legacy_rule_cannot(self, legacy):
        assert "old records" in at.base_error(_base(**legacy))


# --- the form both apps render -------------------------------------------------


def _form() -> None:
    import streamlit as st

    from agent_stonks import apple_trader_ui

    copy = apple_trader_ui.FormCopy(
        prefix="t", unavailable_suffix="", instrument_help="", model_help=""
    )
    st.session_state["out"] = apple_trader_ui.params(["AAPL"], copy)


@pytest.fixture
def form(monkeypatch):
    # No bundle is loaded to draw a form.
    monkeypatch.setattr(apple_models, "load", lambda *a, **k: {"model": "stub"})

    def run(app=None):
        app = app or AppTest.from_function(_form, default_timeout=30)
        app.run()
        assert not app.exception, app.exception
        return app

    return run


class TestForm:
    def _pick_highlow(self, form):
        app = form()
        app.selectbox(key="t_model_AAPL").select("highlow").run()
        assert not app.exception
        return app

    def test_without_a_base_the_keys_and_values_are_the_shipped_ones(self, form):
        app = self._pick_highlow(form)
        assert app.number_input(key="t_buy_k_highlow_AAPL").value == pytest.approx(
            at.dayrange_levels("AAPL", "highlow")[0]
        )
        assert app.session_state["out"] == AppleTraderConfig(ticker="AAPL", model_key="highlow")

    def test_the_conditional_extension_brings_its_two_numbers(self, form):
        """Only under "conditional", the one policy that reads them."""
        app = self._pick_highlow(form)
        assert not [n for n in app.number_input if n.key.startswith("t_breach_hold")]
        app.selectbox(key="t_breach_update_AAPL").select("conditional").run()
        assert not app.exception
        assert app.number_input(key="t_breach_hold_bars").value == 2
        assert app.number_input(key="t_breach_hold_k").value == pytest.approx(0.05)
        app.number_input(key="t_breach_hold_bars").set_value(3).run()
        out = app.session_state["out"]
        assert (out.breach_update, out.breach_hold_bars, out.breach_hold_k) == (
            "conditional", 3, pytest.approx(0.05)
        )

    def test_the_form_opens_on_the_base_and_keeps_what_it_has_no_knob_for(self, form):
        base_configs.set_base(asdict(_base()))
        app = self._pick_highlow(form)
        suffix = base_configs.key_suffix("highlow", "AAPL")
        assert app.number_input(key=f"t_buy_k_highlow_AAPL{suffix}").value == pytest.approx(0.85)
        assert app.number_input(
            key=f"t_momentum_confirmation_bars{suffix}"
        ).value == 7
        assert app.session_state["out"] == _base()

    def test_a_new_base_reseeds_an_open_form(self, form):
        app = self._pick_highlow(form)
        app.number_input(key="t_momentum_confirmation_bars").set_value(9).run()
        assert app.session_state["out"].momentum_confirmation_bars == 9
        # SimLab sets a base while this page is open: its next rerun opens on it.
        base_configs.set_base(asdict(_base()))
        app.run()
        assert not app.exception
        assert app.session_state["out"] == _base()

    def test_other_pairs_keep_the_form_as_it_was(self, form):
        base_configs.set_base(asdict(_base()))
        app = form()
        app.selectbox(key="t_model_AAPL").select("highlow2").run()  # a model with no base
        assert not app.exception
        assert app.number_input(key="t_buy_k_highlow2_AAPL").value == pytest.approx(
            at.dayrange_levels("AAPL", "highlow2")[0]
        )
        assert app.session_state["out"].limit_entry is True

    def test_the_model_help_says_where_the_values_come_from(self):
        from agent_stonks import apple_trader_ui

        assert apple_trader_ui.base_help("highlow", "AAPL") == ""
        base_configs.set_base(asdict(_base()), source={"job_id": "20261014-x"})
        text = apple_trader_ui.base_help("highlow", "AAPL")
        assert "20261014-x" in text
        assert "`limit_entry` False (shipped True)" in text
        assert "`buy_k` 0.85" in text


def _two_symbol_form() -> None:
    import streamlit as st

    from agent_stonks import apple_trader_ui

    copy = apple_trader_ui.FormCopy(
        prefix="t", unavailable_suffix="", instrument_help="", model_help=""
    )
    st.session_state["out"] = apple_trader_ui.params(["AAPL", "AMGN"], copy)


class TestBreachPerInstrument:
    """The breach policy's default can follow the instrument (per instrument
    from 2026-10-25; one default for all since 10-26, the table kept), so
    switching symbol re-seeds it -- where no base digest scopes the key already."""

    def test_switching_symbol_reseeds_the_policy(self, form, monkeypatch):
        monkeypatch.setitem(at.APPLE_TRADER_BREACH_UPDATE_BY_TICKER, "AMGN", "brownian")
        app = form(AppTest.from_function(_two_symbol_form, default_timeout=30))
        assert app.session_state["out"].breach_update == "shift"
        app.selectbox(key="t_ticker").select("AMGN").run()
        assert not app.exception
        assert app.selectbox(key="t_breach_update_AMGN").value == "brownian"
        assert app.session_state["out"].breach_update == "brownian"
        app.selectbox(key="t_ticker").select("AAPL").run()
        assert app.session_state["out"].breach_update == "shift"

    def test_a_pairs_base_keeps_its_key_and_its_policy(self, form):
        base_configs.set_base(asdict(_base(breach_update="shift")))
        app = form()
        suffix = base_configs.key_suffix("highlow", "AAPL")
        assert app.selectbox(key=f"t_breach_update{suffix}").value == "shift"
        assert app.session_state["out"].breach_update == "shift"


class TestBreakerUntilRevisedBox:
    def test_off_by_default_and_ticked_it_is_on(self, form):
        app = form()
        assert app.checkbox(key="t_breaker_until_revised").value is False
        assert app.session_state["out"].breaker_until_revised is False
        app.checkbox(key="t_breaker_until_revised").check().run()
        assert app.session_state["out"].breaker_until_revised is True


class TestCircuitBreakerBox:
    """The breaker is switched off by its box; the threshold is kept, greyed out."""

    KEY, BOX = "t_min_win_k_AAPL", "t_min_win_on_AAPL"

    def test_on_by_default_at_the_symbols_threshold(self, form):
        app = form()
        assert app.checkbox(key=self.BOX).value is True
        assert not app.number_input(key=self.KEY).disabled
        assert app.session_state["out"].min_win_k == pytest.approx(at.min_win_for("AAPL"))

    def test_unticked_it_is_off_and_the_threshold_waits(self, form):
        app = form()
        app.number_input(key=self.KEY).set_value(0.07).run()
        app.checkbox(key=self.BOX).uncheck().run()
        assert app.session_state["out"].min_win_k == 0.0
        assert app.number_input(key=self.KEY).disabled
        assert not app.warning
        app.checkbox(key=self.BOX).check().run()
        assert app.session_state["out"].min_win_k == pytest.approx(0.07)

    def test_a_kept_zero_opens_it_unticked(self, form):
        """A 0 remembered from before the box (`last_setup`) meant "off"."""
        form()  # for its stubbed bundle; a fresh AppTest is falsy, so seed our own
        app = AppTest.from_function(_form, default_timeout=30)
        app.session_state[self.KEY] = 0.0
        app.run()
        assert not app.exception, app.exception
        assert app.checkbox(key=self.BOX).value is False
        assert app.number_input(key=self.KEY).value == pytest.approx(at.min_win_for("AAPL"))
        assert app.session_state["out"].min_win_k == 0.0

    def test_a_base_without_a_breaker_opens_it_unticked(self, form):
        base_configs.set_base(asdict(_base(min_win_k=0.0)))
        app = form()
        app.selectbox(key="t_model_AAPL").select("highlow").run()
        suffix = base_configs.key_suffix("highlow", "AAPL")
        assert app.checkbox(key=f"{self.BOX}{suffix}").value is False
        assert app.session_state["out"] == _base(min_win_k=0.0)
