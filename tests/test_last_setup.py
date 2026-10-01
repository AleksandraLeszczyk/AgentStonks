import json

from streamlit.testing.v1 import AppTest

from agent_stonks import last_setup


def _stored() -> dict:
    return last_setup.load()


def test_a_changed_setting_is_written_and_an_untouched_default_is_not():
    session = {"sidebar_symbols": "AAPL", "agent_llm_personality": "momentum"}
    assert last_setup.remember(session) is False  # first sight: the defaults
    session["sidebar_symbols"] = "MU"
    assert last_setup.remember(session) is True
    assert _stored() == {"sidebar_symbols": "MU"}


def test_only_kept_keys_are_written():
    session = {"sidebar_symbols": "AAPL", "agent_start": False, "some_password": "x"}
    last_setup.remember(session)
    session.update(sidebar_symbols="MU", agent_start=True, some_password="secret")
    last_setup.remember(session)
    assert _stored() == {"sidebar_symbols": "MU"}


def test_keys_built_from_the_instrument_or_provider_are_kept():
    assert last_setup.is_kept("apple_trader_buy_k_highlow_AAPL")
    assert last_setup.is_kept("agent_llm_model_anthropic")
    assert last_setup.is_kept("premarket_model_openai")
    assert not last_setup.is_kept("agent_continue_today")
    assert not last_setup.is_kept("live_chart_AAPL")


def test_restore_fills_only_what_is_missing():
    last_setup._write(
        {"sidebar_symbols": "MU", "agent_trading_mode": "alpaca_live"},
        last_setup.SETUP_PATH,
    )
    session = {"sidebar_symbols": "TSLA"}
    assert last_setup.restore(session) == 1
    assert session["sidebar_symbols"] == "TSLA"
    assert session["agent_trading_mode"] == "alpaca_live"
    # Seeded values are what the session "first saw": nothing to write back.
    assert last_setup.remember(session) is False


def test_a_tab_that_only_looks_does_not_undo_another_tabs_change():
    last_setup._write({"apple_trader_dayrange_size": 50.0}, last_setup.SETUP_PATH)
    tab_a, tab_b = {}, {}
    last_setup.restore(tab_a)
    last_setup.restore(tab_b)
    tab_a["apple_trader_dayrange_size"] = 95.0
    last_setup.remember(tab_a)
    assert last_setup.remember(tab_b) is False
    assert _stored()["apple_trader_dayrange_size"] == 95.0


def test_a_dropped_widget_comes_back_on_its_last_value():
    session = {"apple_trader_take_pct": 70.0}
    last_setup.remember(session)
    session["apple_trader_take_pct"] = 50.0
    last_setup.remember(session)
    del session["apple_trader_take_pct"]  # another personality was picked
    last_setup.remember(session)
    assert _stored()["apple_trader_take_pct"] == 50.0
    last_setup.restore(session)
    assert session["apple_trader_take_pct"] == 50.0


def test_an_unreadable_or_foreign_file_restores_nothing():
    path = last_setup.SETUP_PATH
    path.write_text("{not json", encoding="utf-8")
    assert last_setup.load() == {}
    path.write_text(json.dumps({"version": 99, "values": {"sidebar_symbols": "MU"}}))
    assert last_setup.load() == {}
    session = {}
    assert last_setup.restore(session) == 0


def _app() -> None:
    import streamlit as st

    from agent_stonks import last_setup

    last_setup.restore()
    st.text_input("Symbols", value="AAPL", key="sidebar_symbols")
    st.selectbox(
        "Personality", ["momentum", "apple_trader"], index=0, key="agent_llm_personality"
    )
    st.number_input(
        "Size", min_value=1.0, max_value=100.0, value=95.0,
        key="apple_trader_dayrange_size",
    )
    last_setup.remember()


def test_a_new_session_opens_on_the_last_setup():
    first = AppTest.from_function(_app).run()
    first.text_input(key="sidebar_symbols").input("MU").run()
    first.selectbox(key="agent_llm_personality").select("apple_trader").run()
    first.number_input(key="apple_trader_dayrange_size").set_value(50.0).run()
    assert not first.exception

    # A restart, or a reconnect as a new session: a fresh session state.
    second = AppTest.from_function(_app).run()
    assert not second.exception
    assert second.text_input(key="sidebar_symbols").value == "MU"
    assert second.selectbox(key="agent_llm_personality").value == "apple_trader"
    assert second.number_input(key="apple_trader_dayrange_size").value == 50.0


def test_a_stored_value_a_widget_can_no_longer_take_falls_back_to_its_default():
    last_setup._write(
        {"agent_llm_personality": "retired_personality", "apple_trader_dayrange_size": 500.0},
        last_setup.SETUP_PATH,
    )
    app = AppTest.from_function(_app).run()
    assert not app.exception
    assert app.selectbox(key="agent_llm_personality").value == "momentum"
    assert app.number_input(key="apple_trader_dayrange_size").value == 95.0
