import logging

import pytest

from agent_stonks import datalog


@pytest.fixture(autouse=True)
def _reset_dedup():
    datalog._last_outcome.clear()
    yield
    datalog._last_outcome.clear()


def test_log_fetch_basic_message(caplog):
    with caplog.at_level(logging.INFO, logger="agent_stonks.data"):
        datalog.log_fetch("ask/bid price", "Alpaca REST /quotes/latest", symbol="AAPL", detail="bid=1.0, ask=1.1")
    assert caplog.records[0].levelno == logging.INFO
    assert caplog.records[0].message == (
        "AAPL: ask/bid price fetched from Alpaca REST /quotes/latest (bid=1.0, ask=1.1)"
    )


def test_log_fetch_lists_failed_sources(caplog):
    with caplog.at_level(logging.INFO, logger="agent_stonks.data"):
        datalog.log_fetch(
            "bars",
            "yfinance (delayed)",
            symbol="AAPL",
            failures=[("Alpaca REST", "403 Forbidden"), ("Alpaca WS", "timeout")],
        )
    assert caplog.records[0].message == (
        "AAPL: bars fetched from yfinance (delayed) "
        "(fetching from Alpaca REST, Alpaca WS failed: 403 Forbidden; timeout)"
    )


def test_log_fetch_failure_message_and_level(caplog):
    with caplog.at_level(logging.INFO, logger="agent_stonks.data"):
        datalog.log_fetch_failure(
            "ask/bid price",
            [("Alpaca REST /quotes/latest", "timeout")],
            symbol="AAPL",
            consequence="keeping last known bid/ask",
        )
    assert caplog.records[0].levelno == logging.WARNING
    assert caplog.records[0].message == (
        "AAPL: ask/bid price fetch failed — tried Alpaca REST /quotes/latest: timeout"
        " — keeping last known bid/ask"
    )


def test_repeat_outcome_demotes_to_debug(caplog):
    with caplog.at_level(logging.DEBUG, logger="agent_stonks.data"):
        datalog.log_fetch("last price", "Alpaca WS", symbol="AAPL", detail="price=1")
        datalog.log_fetch("last price", "Alpaca WS", symbol="AAPL", detail="price=2")
    assert [r.levelno for r in caplog.records] == [logging.INFO, logging.DEBUG]


def test_source_change_logs_at_info_again(caplog):
    with caplog.at_level(logging.DEBUG, logger="agent_stonks.data"):
        datalog.log_fetch("last price", "Alpaca WS", symbol="AAPL")
        datalog.log_fetch("last price", "Alpaca REST", symbol="AAPL")
        datalog.log_fetch("last price", "Alpaca WS", symbol="AAPL")
    assert [r.levelno for r in caplog.records] == [logging.INFO] * 3


def test_failure_then_recovery_both_logged(caplog):
    with caplog.at_level(logging.DEBUG, logger="agent_stonks.data"):
        datalog.log_fetch_failure("ask/bid price", [("Alpaca REST", "boom")], symbol="AAPL")
        datalog.log_fetch_failure("ask/bid price", [("Alpaca REST", "boom")], symbol="AAPL")
        datalog.log_fetch("ask/bid price", "Alpaca REST", symbol="AAPL")
    assert [r.levelno for r in caplog.records] == [
        logging.WARNING,
        logging.DEBUG,
        logging.INFO,
    ]


def test_symbols_and_kinds_tracked_independently(caplog):
    with caplog.at_level(logging.DEBUG, logger="agent_stonks.data"):
        datalog.log_fetch("bars", "Alpaca WS", symbol="AAPL")
        datalog.log_fetch("bars", "Alpaca WS", symbol="TSLA")
        datalog.log_fetch("last price", "Alpaca WS", symbol="AAPL")
    assert [r.levelno for r in caplog.records] == [logging.INFO] * 3


def test_no_symbol_message(caplog):
    with caplog.at_level(logging.INFO, logger="agent_stonks.data"):
        datalog.log_fetch("market indicators", "yfinance")
    assert caplog.records[0].message == "market indicators fetched from yfinance"
