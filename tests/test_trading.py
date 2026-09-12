"""Order routing: the Trading API client, AlpacaBroker, venue selection, and
the ledger's live-trade path.

Every order here is mocked. Nothing in this file may reach a real Alpaca
account, paper or otherwise.
"""
import pytest

from agent_stonks import trading_mode, trading_rest
from agent_stonks.broker import AlpacaBroker, PaperBroker
from agent_stonks.config import LIVE_TRADING_CONFIRM_PHRASE, LIVE_TRADING_ENV_FLAG
from agent_stonks.decisions import DecisionTracker
from agent_stonks.trading_rest import TradingCredentials, TradingError

PAPER = TradingCredentials(key="k", secret="s", live=False)
LIVE = TradingCredentials(key="k", secret="s", live=True)
PAPER_URL = "https://paper-api.alpaca.markets"
LIVE_URL = "https://api.alpaca.markets"


# Credentials are stripped for every test by tests/conftest.py, so a test that
# wants one sets it itself and the value is always fake.


class TestCredentials:
    def test_paper_and_live_point_at_different_hosts(self):
        assert PAPER.base_url == PAPER_URL
        assert LIVE.base_url == LIVE_URL
        assert "LIVE" in LIVE.venue and "paper" in PAPER.venue

    def test_unconfigured_credentials_refuse_to_call(self):
        with pytest.raises(TradingError):
            trading_rest.get_account(TradingCredentials(key="", secret=""))


class TestTradingRest:
    def test_surfaces_alpacas_error_message_not_just_the_status(self, requests_mock):
        # A bare "403 Forbidden" hides the only useful part of a rejection.
        requests_mock.post(
            f"{PAPER_URL}/v2/orders",
            status_code=403,
            json={"message": "insufficient buying power"},
        )
        with pytest.raises(TradingError, match="insufficient buying power"):
            trading_rest.submit_order("AAPL", "buy", 1, PAPER)

    def test_sends_credentials_in_alpaca_headers(self, requests_mock):
        requests_mock.get(f"{PAPER_URL}/v2/account", json={"cash": "1000"})
        trading_rest.get_account(PAPER)
        headers = requests_mock.last_request.headers
        assert headers["APCA-API-KEY-ID"] == "k"
        assert headers["APCA-API-SECRET-KEY"] == "s"

    def test_formats_a_fractional_quantity_alpaca_accepts(self, requests_mock):
        requests_mock.post(f"{PAPER_URL}/v2/orders", json={"id": "o1"})
        trading_rest.submit_order("AAPL", "buy", 12.837100000, PAPER)
        assert requests_mock.last_request.json()["qty"] == "12.8371"

    def test_does_not_send_scientific_notation(self, requests_mock):
        requests_mock.post(f"{PAPER_URL}/v2/orders", json={"id": "o1"})
        trading_rest.submit_order("AAPL", "buy", 0.00001, PAPER)
        assert requests_mock.last_request.json()["qty"] == "0.00001"

    def test_rejects_a_bad_side_before_reaching_the_network(self):
        with pytest.raises(ValueError):
            trading_rest.submit_order("AAPL", "hold", 1, PAPER)

    def test_rejects_a_non_positive_quantity(self):
        with pytest.raises(ValueError):
            trading_rest.submit_order("AAPL", "buy", 0, PAPER)

    def test_positions_come_back_keyed_by_symbol(self, requests_mock):
        requests_mock.get(
            f"{PAPER_URL}/v2/positions",
            json=[{"symbol": "AAPL", "qty": "10"}, {"symbol": "TSLA", "qty": "-4"}],
        )
        assert trading_rest.positions_by_symbol(PAPER) == {"AAPL": 10.0, "TSLA": -4.0}

    def test_wait_for_fill_polls_until_terminal(self, requests_mock):
        requests_mock.get(
            f"{PAPER_URL}/v2/orders/o1",
            [
                {"json": {"id": "o1", "status": "new", "filled_qty": "0"}},
                {"json": {"id": "o1", "status": "partially_filled", "filled_qty": "3"}},
                {"json": {"id": "o1", "status": "filled", "filled_qty": "10",
                          "filled_avg_price": "101.5"}},
            ],
        )
        order = trading_rest.wait_for_fill("o1", PAPER, timeout_sec=5, sleep=lambda _: None)
        assert order["status"] == "filled"
        assert order["filled_qty"] == "10"

    def test_wait_for_fill_gives_up_without_cancelling(self, requests_mock):
        # Cancelling a working order is a trading decision, not a timeout.
        requests_mock.get(
            f"{PAPER_URL}/v2/orders/o1", json={"id": "o1", "status": "new", "filled_qty": "0"}
        )
        order = trading_rest.wait_for_fill("o1", PAPER, timeout_sec=0, sleep=lambda _: None)
        assert order["status"] == "new"
        assert not any(r.method == "DELETE" for r in requests_mock.request_history)


def _account(requests_mock, cash="10000", buying_power="10000", positions=None, **extra):
    requests_mock.get(
        f"{PAPER_URL}/v2/account",
        json={"cash": cash, "buying_power": buying_power, "equity": cash,
              "status": "ACTIVE", **extra},
    )
    requests_mock.get(f"{PAPER_URL}/v2/positions", json=positions or [])


class TestAlpacaBroker:
    def _broker(self) -> AlpacaBroker:
        return AlpacaBroker(PAPER, fill_timeout_sec=5, sleep=lambda _: None)

    def test_is_not_simulated(self):
        assert self._broker().is_simulated is False
        assert PaperBroker().is_simulated is True

    def test_account_snapshot_reads_cash_and_positions(self, requests_mock):
        _account(requests_mock, cash="2500.5", positions=[{"symbol": "AAPL", "qty": "7"}])
        snap = self._broker().account_snapshot()
        assert snap["cash"] == 2500.5
        assert snap["positions"] == {"AAPL": 7.0}

    def test_a_failed_account_read_returns_none_not_an_empty_account(self, requests_mock):
        # None means "don't know"; an empty dict would zero the ledger's cash
        # and wipe every position on a transient network blip.
        requests_mock.get(f"{PAPER_URL}/v2/account", status_code=500, json={})
        assert self._broker().account_snapshot() is None

    def test_flags_a_blocked_account(self, requests_mock):
        _account(requests_mock, trading_blocked=True)
        assert self._broker().account_snapshot()["blocked"] is True

    def test_floors_a_fractional_quantity_on_a_whole_share_symbol(self, requests_mock):
        requests_mock.get(f"{PAPER_URL}/v2/assets/BRK.A", json={"fractionable": False})
        # Rounding up would spend money the sizing never allowed for.
        assert self._broker().normalize_quantity("BRK.A", 12.9) == 12.0

    def test_keeps_a_fractional_quantity_where_the_symbol_allows_it(self, requests_mock):
        requests_mock.get(f"{PAPER_URL}/v2/assets/AAPL", json={"fractionable": True})
        assert self._broker().normalize_quantity("AAPL", 12.9) == 12.9

    def test_an_unknown_asset_is_treated_as_whole_share_only(self, requests_mock):
        requests_mock.get(f"{PAPER_URL}/v2/assets/ZZZZ", status_code=404, json={})
        assert self._broker().normalize_quantity("ZZZZ", 12.9) == 12.0

    def test_rejects_rather_than_ordering_when_the_quantity_rounds_to_zero(self, requests_mock):
        requests_mock.get(f"{PAPER_URL}/v2/assets/BRK.A", json={"fractionable": False})
        posted = requests_mock.post(f"{PAPER_URL}/v2/orders", json={"id": "o1"})
        report = self._broker().submit_order("BRK.A", "buy", 0.4, 100.0)
        assert report["status"] == "rejected"
        assert "rounds down to zero" in report["reason"]
        assert posted.call_count == 0

    def test_reports_the_brokers_average_fill_not_the_quoted_price(self, requests_mock):
        # Slippage between the two is the thing routing a real order reveals.
        requests_mock.get(f"{PAPER_URL}/v2/assets/AAPL", json={"fractionable": True})
        requests_mock.post(f"{PAPER_URL}/v2/orders", json={"id": "o1"})
        requests_mock.get(
            f"{PAPER_URL}/v2/orders/o1",
            json={"id": "o1", "status": "filled", "filled_qty": "10",
                  "filled_avg_price": "101.73"},
        )
        report = self._broker().submit_order("AAPL", "buy", 10, price=100.0)
        assert report["status"] == "filled"
        assert report["filled_qty"] == 10.0
        assert report["filled_price"] == 101.73

    def test_reports_a_partial_fill_as_what_it_is(self, requests_mock):
        requests_mock.get(f"{PAPER_URL}/v2/assets/AAPL", json={"fractionable": True})
        requests_mock.post(f"{PAPER_URL}/v2/orders", json={"id": "o1"})
        requests_mock.get(
            f"{PAPER_URL}/v2/orders/o1",
            json={"id": "o1", "status": "done_for_day", "filled_qty": "4",
                  "filled_avg_price": "100"},
        )
        report = self._broker().submit_order("AAPL", "buy", 10, price=100.0)
        assert report["filled_qty"] == 4.0
        assert "partial fill: 4 of 10" in report["reason"]

    def test_a_rejected_order_is_a_result_not_an_exception(self, requests_mock):
        requests_mock.get(f"{PAPER_URL}/v2/assets/AAPL", json={"fractionable": True})
        requests_mock.post(
            f"{PAPER_URL}/v2/orders", status_code=403,
            json={"message": "insufficient buying power"},
        )
        report = self._broker().submit_order("AAPL", "buy", 10, price=100.0)
        assert report["status"] == "rejected"
        assert report["filled_qty"] == 0.0
        assert "insufficient buying power" in report["reason"]

    def test_an_order_still_working_reports_no_fill_and_says_so(self, requests_mock):
        requests_mock.get(f"{PAPER_URL}/v2/assets/AAPL", json={"fractionable": True})
        requests_mock.post(f"{PAPER_URL}/v2/orders", json={"id": "o1"})
        requests_mock.get(
            f"{PAPER_URL}/v2/orders/o1", json={"id": "o1", "status": "new", "filled_qty": "0"}
        )
        broker = AlpacaBroker(PAPER, fill_timeout_sec=0, sleep=lambda _: None)
        report = broker.submit_order("AAPL", "buy", 10, price=100.0)
        assert report["status"] == "rejected"
        assert "still working" in report["reason"]
        assert "NOT cancelled" in report["reason"]

    def test_max_quantity_uses_buying_power_not_the_local_ledger(self, requests_mock):
        _account(requests_mock, cash="10000", buying_power="500")
        assert self._broker().max_quantity("AAPL", "buy", 100.0) == 5.0

    def test_max_quantity_for_a_sell_is_the_brokers_position(self, requests_mock):
        _account(requests_mock, positions=[{"symbol": "AAPL", "qty": "3"}])
        assert self._broker().max_quantity("AAPL", "sell", 100.0) == 3.0


class TestResolveBroker:
    def test_local_never_touches_the_network(self):
        broker, mode, _ = trading_mode.resolve_broker("local")
        assert mode == "local" and broker.is_simulated

    def test_live_refused_without_the_environment_flag(self, monkeypatch):
        monkeypatch.setenv("ALPACA_LIVE_API_KEY", "k")
        monkeypatch.setenv("ALPACA_LIVE_SECRET", "s")
        broker, mode, msg = trading_mode.resolve_broker(
            "alpaca_live", LIVE_TRADING_CONFIRM_PHRASE
        )
        assert mode == "local" and broker.is_simulated
        assert LIVE_TRADING_ENV_FLAG in msg

    def test_live_refused_without_the_typed_confirmation(self, monkeypatch):
        monkeypatch.setenv(LIVE_TRADING_ENV_FLAG, "true")
        monkeypatch.setenv("ALPACA_LIVE_API_KEY", "k")
        monkeypatch.setenv("ALPACA_LIVE_SECRET", "s")
        broker, mode, msg = trading_mode.resolve_broker("alpaca_live", "yes please")
        assert mode == "local" and broker.is_simulated
        assert LIVE_TRADING_CONFIRM_PHRASE in msg

    def test_live_needs_both_gates_together(self, monkeypatch, requests_mock):
        monkeypatch.setenv(LIVE_TRADING_ENV_FLAG, "true")
        monkeypatch.setenv("ALPACA_LIVE_API_KEY", "k")
        monkeypatch.setenv("ALPACA_LIVE_SECRET", "s")
        requests_mock.get(
            f"{LIVE_URL}/v2/account",
            json={"cash": "500", "equity": "500", "status": "ACTIVE"},
        )
        broker, mode, _ = trading_mode.resolve_broker(
            "alpaca_live", LIVE_TRADING_CONFIRM_PHRASE
        )
        assert mode == "alpaca_live"
        assert broker.is_simulated is False and broker.is_live is True

    def test_paper_falls_back_when_its_keys_are_missing(self):
        broker, mode, msg = trading_mode.resolve_broker("alpaca_paper")
        assert mode == "local" and broker.is_simulated
        assert "ALPACA_PAPER_API_KEY" in msg

    def test_paper_may_fall_back_to_the_market_data_keys(self, monkeypatch, requests_mock):
        # An Alpaca paper key is usually the one already in ALPACA_API_KEY;
        # making the user paste it twice buys nothing on an account with no
        # real money in it.
        monkeypatch.setenv("ALPACA_API_KEY", "k")
        monkeypatch.setenv("ALPACA_SECRET", "s")
        requests_mock.get(
            f"{PAPER_URL}/v2/account",
            json={"cash": "100000", "equity": "100000", "status": "ACTIVE"},
        )
        _, mode, _ = trading_mode.resolve_broker("alpaca_paper")
        assert mode == "alpaca_paper"

    def test_live_NEVER_borrows_the_market_data_keys(self, monkeypatch, requests_mock):
        # The one inheritance that must not exist: if ALPACA_API_KEY happened to
        # be a live key, letting live fall back to it would make the difference
        # between simulated and real money depend on which variable was set.
        monkeypatch.setenv(LIVE_TRADING_ENV_FLAG, "true")
        monkeypatch.setenv("ALPACA_API_KEY", "k")
        monkeypatch.setenv("ALPACA_SECRET", "s")
        account = requests_mock.get(f"{LIVE_URL}/v2/account", json={"status": "ACTIVE"})

        broker, mode, msg = trading_mode.resolve_broker(
            "alpaca_live", LIVE_TRADING_CONFIRM_PHRASE
        )

        assert mode == "local" and broker.is_simulated
        assert "ALPACA_LIVE_API_KEY" in msg
        assert account.call_count == 0  # the live account was never even contacted

    def test_a_blocked_account_is_refused_before_any_order(self, monkeypatch, requests_mock):
        monkeypatch.setenv("ALPACA_PAPER_API_KEY", "k")
        monkeypatch.setenv("ALPACA_PAPER_SECRET", "s")
        requests_mock.get(
            f"{PAPER_URL}/v2/account",
            json={"cash": "0", "equity": "0", "status": "ACCOUNT_CLOSED",
                  "trading_blocked": True},
        )
        broker, mode, msg = trading_mode.resolve_broker("alpaca_paper")
        assert mode == "local" and broker.is_simulated
        assert "blocked" in msg

    def test_bad_credentials_fall_back_rather_than_raising(self, monkeypatch, requests_mock):
        monkeypatch.setenv("ALPACA_PAPER_API_KEY", "k")
        monkeypatch.setenv("ALPACA_PAPER_SECRET", "s")
        requests_mock.get(f"{PAPER_URL}/v2/account", status_code=401, json={"message": "auth"})
        _, mode, _ = trading_mode.resolve_broker("alpaca_paper")
        assert mode == "local"

    def test_an_unknown_mode_falls_back_to_the_default(self):
        # Default is alpaca_paper; with no keys at all in this test env it
        # degrades once more to local rather than raising.
        _, mode, _ = trading_mode.resolve_broker("bogus")
        assert mode == "local"


class _StubBroker(PaperBroker):
    """A non-simulated broker with scripted answers, so the ledger's live path
    can be exercised without any HTTP at all."""

    def __init__(self, report, snapshot, ceiling=None, price=100.0):
        self.report = report
        self.snapshot = snapshot
        self.ceiling = ceiling
        self.price = price
        self.orders: list[tuple] = []

    @property
    def is_simulated(self) -> bool:
        return False

    @property
    def venue(self) -> str:
        return "stub venue"

    def get_current_price(self, symbol, key, secret, feed="iex") -> float:
        return self.price

    def max_quantity(self, symbol, side, price):
        return self.ceiling

    def account_snapshot(self):
        return self.snapshot

    def submit_order(self, symbol, side, quantity, price) -> dict:
        self.orders.append((symbol, side, quantity, price))
        return self.report


class TestLedgerOnALiveBroker:
    def test_cash_and_positions_come_from_the_account_not_arithmetic(self):
        # The venue is the truth: a fill at a different price, or a position
        # moved outside this app, must not leave the ledger disagreeing.
        broker = _StubBroker(
            report={"status": "filled", "filled_qty": 10.0, "filled_price": 101.5},
            snapshot={"cash": 8985.0, "positions": {"AAPL": 10.0}, "buying_power": 8985.0},
        )
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)

        decision = tracker.record_trade("AAPL", "buy", 10, "entry", "k", "s")

        assert decision.status == "filled"
        assert decision.filled_quantity == 10.0
        assert decision.price == 101.5
        assert tracker.cash == 8985.0
        assert tracker.positions == {"AAPL": 10.0}

    def test_records_the_partial_fill_the_venue_reported(self):
        broker = _StubBroker(
            report={"status": "filled", "filled_qty": 4.0, "filled_price": 100.0,
                    "reason": "partial fill: 4 of 10 shares"},
            snapshot={"cash": 9600.0, "positions": {"AAPL": 4.0}, "buying_power": 9600.0},
        )
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)

        decision = tracker.record_trade("AAPL", "buy", 10, "entry", "k", "s")

        assert decision.requested_quantity == 10
        assert decision.filled_quantity == 4.0
        assert "partial fill" in decision.reasoning
        assert tracker.positions == {"AAPL": 4.0}

    def test_a_rejection_is_recorded_with_the_brokers_reason(self):
        broker = _StubBroker(
            report={"status": "rejected", "filled_qty": 0.0, "filled_price": 100.0,
                    "reason": "insufficient buying power"},
            snapshot={"cash": 10_000.0, "positions": {}, "buying_power": 10_000.0},
        )
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)

        decision = tracker.record_trade("AAPL", "buy", 10, "entry", "k", "s")

        assert decision.status == "rejected"
        assert decision.filled_quantity == 0.0
        assert "insufficient buying power" in decision.reasoning
        assert tracker.cash == 10_000.0

    def test_clamps_the_request_to_what_the_venue_allows(self):
        broker = _StubBroker(
            report={"status": "filled", "filled_qty": 5.0, "filled_price": 100.0},
            snapshot={"cash": 0.0, "positions": {"AAPL": 5.0}, "buying_power": 0.0},
            ceiling=5.0,
        )
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)

        tracker.record_trade("AAPL", "buy", 100, "entry", "k", "s")

        assert broker.orders == [("AAPL", "buy", 5.0, 100.0)]

    def test_no_order_is_sent_when_the_venue_allows_nothing(self):
        broker = _StubBroker(
            report={"status": "filled", "filled_qty": 1.0, "filled_price": 100.0},
            snapshot={"cash": 0.0, "positions": {}, "buying_power": 0.0},
            ceiling=0.0,
        )
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)

        decision = tracker.record_trade("AAPL", "buy", 10, "entry", "k", "s")

        assert broker.orders == []
        assert decision.status == "rejected"
        assert "no buying power" in decision.reasoning

    def test_charges_no_modelled_fee_on_a_real_venue(self):
        # Alpaca takes no commission and the real fees are already inside the
        # cash the account reports -- adding TRADE_FIXED_COST would double-count.
        broker = _StubBroker(
            report={"status": "filled", "filled_qty": 10.0, "filled_price": 100.0},
            snapshot={"cash": 9000.0, "positions": {"AAPL": 10.0}, "buying_power": 9000.0},
        )
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker, trade_cost=1.15)

        decision = tracker.record_trade("AAPL", "buy", 10, "entry", "k", "s")

        assert decision.fee == 0.0

    def test_falls_back_to_local_arithmetic_when_the_account_read_fails(self):
        # An unreachable broker should leave a stale-but-plausible ledger,
        # not a zeroed one.
        broker = _StubBroker(
            report={"status": "filled", "filled_qty": 10.0, "filled_price": 100.0},
            snapshot=None,
        )
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)

        tracker.record_trade("AAPL", "buy", 10, "entry", "k", "s")

        assert tracker.cash == 9000.0
        assert tracker.positions == {"AAPL": 10.0}

    def test_sync_adopts_the_accounts_balance_and_holdings(self):
        broker = _StubBroker(
            report={}, snapshot={"cash": 4321.0, "positions": {"TSLA": 3.0}},
        )
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)

        assert tracker.sync_from_broker() is True
        assert tracker.cash == 4321.0
        assert tracker.positions == {"TSLA": 3.0}

    def test_sync_reports_failure_without_touching_the_ledger(self):
        broker = _StubBroker(report={}, snapshot=None)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=broker)

        assert tracker.sync_from_broker() is False
        assert tracker.cash == 10_000.0

    def test_the_simulated_path_is_untouched(self):
        # The local ledger must keep behaving exactly as it always has, fee
        # included -- simlab and every strategy test depend on it.
        tracker = DecisionTracker(
            starting_cash=1000.0, broker=_SimPriced(100.0), trade_cost=1.0
        )
        decision = tracker.record_trade("AAPL", "buy", 5, "entry", "k", "s")
        assert decision.fee == 1.0
        assert tracker.cash == pytest.approx(1000.0 - 500.0 - 1.0)


class _SimPriced(PaperBroker):
    def __init__(self, price):
        self.price = price

    def get_current_price(self, symbol, key, secret, feed="iex"):
        return self.price
