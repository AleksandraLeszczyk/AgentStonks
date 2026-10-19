from datetime import datetime, timedelta, timezone

import pytest

from agent_stonks import notifications
from agent_stonks.decisions import Decision, DecisionTracker
from agent_stonks.notifications import Settings, Watcher
from agent_stonks.state import AppState

T0 = datetime(2026, 10, 16, 14, 0, tzinfo=timezone.utc)  # 10:00 ET


@pytest.fixture
def outbox(monkeypatch):
    sent = []

    def fake_send(to, subject, body, smtp=None):
        sent.append({"to": to, "subject": subject, "body": body})

    monkeypatch.setattr(notifications, "send_email", fake_send)
    return sent


def _fill(ts, action="buy", qty=10.0, price=230.15, symbol="AAPL", status="filled"):
    return Decision(
        ts=ts.isoformat(), symbol=symbol, action=action, requested_quantity=qty,
        filled_quantity=qty if status == "filled" else 0.0, price=price,
        reasoning="Price touched the buy level.", status=status,
        cash_after=97_698.50, position_after=qty,
    )


def _state(*decisions, running=True):
    state = AppState()
    state.symbols = ["AAPL"]
    state.trading_mode = state.trading_mode_requested = "alpaca_paper"
    state.agent_running = running
    state.starting_budget = 100_000.0
    tracker = DecisionTracker(starting_cash=100_000.0)
    tracker.decisions = list(decisions)
    state.decision_tracker = tracker
    return state


ON_FILL = Settings(email_to="me@example.com", on_transaction=True)


def test_settings_round_trip_and_missing_file_is_off(tmp_path):
    assert notifications.load_settings() == Settings()
    assert not Settings().email_on
    notifications.save_settings(Settings(email_to="a@b.c", every_n_minutes=True, interval_minutes=15))
    loaded = notifications.load_settings()
    assert loaded == Settings(email_to="a@b.c", every_n_minutes=True, interval_minutes=15)
    assert loaded.email_on


def test_history_is_adopted_silently_and_a_new_fill_is_sent_once(outbox):
    state = _state(_fill(T0))
    watcher = Watcher()
    watcher.poll(state, ON_FILL, T0)
    assert outbox == []

    state.decision_tracker.decisions.append(_fill(T0 + timedelta(minutes=5), "sell", 4, 231.0))
    watcher.poll(state, ON_FILL, T0 + timedelta(minutes=5))
    watcher.poll(state, ON_FILL, T0 + timedelta(minutes=6))
    assert len(outbox) == 1
    assert outbox[0]["to"] == "me@example.com"
    assert outbox[0]["subject"] == "Sold 4 AAPL @ $231.00"
    assert "10:05 ET" in outbox[0]["body"]
    assert "Alpaca paper" in outbox[0]["body"]


def test_rejected_orders_and_alerts_are_not_sent(outbox):
    state = _state()
    watcher = Watcher()
    watcher.poll(state, ON_FILL, T0)
    state.decision_tracker.decisions.append(_fill(T0, status="rejected"))
    watcher.poll(state, ON_FILL, T0)
    assert outbox == []


def test_fills_in_one_poll_share_one_email(outbox):
    state = _state()
    watcher = Watcher()
    watcher.poll(state, ON_FILL, T0)
    state.decision_tracker.decisions += [
        _fill(T0), _fill(T0 + timedelta(seconds=2), "sell", 10, 230.4),
    ]
    watcher.poll(state, ON_FILL, T0)
    assert [m["subject"] for m in outbox] == ["2 trades filled"]
    assert "10:00 ET buy 10 AAPL @ $230.15" in outbox[0]["body"]


def test_a_new_tracker_carrying_the_old_ledger_sends_nothing_old(outbox):
    state = _state(_fill(T0))
    watcher = Watcher()
    watcher.poll(state, ON_FILL, T0)
    carried = DecisionTracker()
    carried.carry_over(state.decision_tracker)
    state.decision_tracker = carried
    watcher.poll(state, ON_FILL, T0 + timedelta(minutes=1))
    assert outbox == []


def test_nothing_is_sent_without_an_address_or_with_the_option_off(outbox):
    state = _state()
    watcher = Watcher()
    watcher.poll(state, Settings(on_transaction=True), T0)
    state.decision_tracker.decisions.append(_fill(T0))
    watcher.poll(state, Settings(on_transaction=True), T0)
    watcher.poll(state, Settings(email_to="me@example.com"), T0)
    assert outbox == []


def test_status_email_every_n_minutes_while_running(outbox, monkeypatch):
    settings = Settings(email_to="me@example.com", every_n_minutes=True, interval_minutes=30)
    state = _state()
    monkeypatch.setattr(state, "mark_to_market", lambda: 101_234.56)
    watcher = Watcher()
    watcher.poll(state, settings, T0)
    watcher.poll(state, settings, T0 + timedelta(minutes=29))
    assert outbox == []

    state.decision_tracker.decisions.append(_fill(T0 + timedelta(minutes=10)))
    state.decision_tracker.positions = {"AAPL": 10.0}
    state.agent_activity = ("🟢", "Agent holding 10 AAPL")
    state.status = "✅ Streaming AAPL (SIP)"
    watcher.poll(state, settings, T0 + timedelta(minutes=30))
    assert len(outbox) == 1
    mail = outbox[0]
    assert mail["subject"] == "Agent Stonks 10:30 ET: $101,234.56 (+$1,234.56)"
    assert mail["body"].startswith(
        "10:30 ET. Agent running on Alpaca paper (AAPL).\n"
        "Agent holding 10 AAPL.\n"
        "Stream: ✅ Streaming AAPL (SIP)\n"
    )
    assert "+1.23% since start" in mail["body"]
    assert "Open: 10 AAPL" in mail["body"]
    assert "Trades since last update: 1" in mail["body"]
    assert "10:10 ET buy 10 AAPL @ $230.15" in mail["body"]

    # The next one 30 minutes after that, listing only what came since.
    watcher.poll(state, settings, T0 + timedelta(minutes=45))
    watcher.poll(state, settings, T0 + timedelta(minutes=60))
    assert len(outbox) == 2
    assert "Trades since last update: 0" in outbox[1]["body"]


def test_no_status_email_while_the_agent_is_stopped(outbox):
    settings = Settings(email_to="me@example.com", every_n_minutes=True, interval_minutes=5)
    state = _state(running=False)
    watcher = Watcher()
    for minute in range(0, 60, 5):
        watcher.poll(state, settings, T0 + timedelta(minutes=minute))
    assert outbox == []


def test_test_send_is_the_status_email_now_with_todays_trades(monkeypatch):
    state = _state(
        _fill(T0 - timedelta(days=1)),
        _fill(T0, "buy", 10, 230.15),
        _fill(T0 + timedelta(minutes=5), "sell", 10, 231.0),
        running=False,
    )
    state.agent_activity = ("🟢", "Agent holding 10 AAPL")  # stale: not running
    monkeypatch.setattr(state, "mark_to_market", lambda: 100_008.50)
    subject, body = notifications.current_status_email(state, T0 + timedelta(minutes=30))
    assert subject == "Test: Agent Stonks 10:30 ET: $100,008.50 (+$8.50)"
    assert body.startswith("10:30 ET. Agent stopped on Alpaca paper (AAPL).\nStream: Idle\n")
    assert "Trades today: 2" in body
    assert "10:00 ET buy 10 AAPL @ $230.15" in body
    assert "10:05 ET sell 10 AAPL @ $231.00" in body


def test_test_send_before_the_agent_ever_ran():
    state = AppState()
    state.symbols = ["AAPL", "INTC"]
    subject, body = notifications.current_status_email(state, T0)
    assert subject == "Test: Agent Stonks 10:00 ET"
    assert body == (
        "10:00 ET. Agent stopped (no venue chosen yet) (AAPL, INTC).\n"
        "Stream: Idle\n"
        "No ledger yet: the agent has not been started.\n"
    )


def test_a_failed_send_is_logged_not_raised(monkeypatch):
    def broken(*a, **k):
        raise OSError("connection refused")

    monkeypatch.setattr(notifications, "send_email", broken)
    state = _state()
    watcher = Watcher()
    watcher.poll(state, ON_FILL, T0)
    state.decision_tracker.decisions.append(_fill(T0))
    watcher.poll(state, ON_FILL, T0)
    (entry,) = notifications.sent_log(state)
    assert entry["error"] == "OSError: connection refused"
    # Not retried on the next poll.
    watcher.poll(state, ON_FILL, T0 + timedelta(seconds=5))
    assert len(notifications.sent_log(state)) == 1


class FakeSMTP:
    """Records what send_email does; `reject` answers AUTH like Gmail does
    to a wrong password."""

    calls: list = []
    reject = False

    def __init__(self, host, port, timeout):
        self.calls.append(("connect", host, port))
        self.esmtp_features = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def starttls(self, context):
        self.calls.append(("starttls",))

    def ehlo_or_helo_if_needed(self):
        self.esmtp_features = {"auth": " LOGIN PLAIN XOAUTH2"}

    def auth_plain(self):
        return f"\0{self.user}\0{self.password}"

    def auth(self, mechanism, authobject):
        self.calls.append(("auth", mechanism, self.user, self.password))
        if self.reject:
            raise notifications.smtplib.SMTPAuthenticationError(
                535, b"5.7.8 Username and Password not accepted. For more information, go to\n"
                b"5.7.8  https://support.google.com/mail/?p=BadCredentials - gsmtp",
            )
        return 235, b"2.7.0 Accepted"

    def send_message(self, msg):
        self.calls.append(("send", msg["To"], msg["Subject"], msg.get_content_type(), msg.get_content()))


@pytest.fixture
def fake_smtp(monkeypatch):
    monkeypatch.setattr(FakeSMTP, "calls", [])
    monkeypatch.setattr(notifications.smtplib, "SMTP", FakeSMTP)
    monkeypatch.setenv("SMTP_USER", "bot@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "app-password")
    monkeypatch.delenv("SMTP_HOST", raising=False)
    monkeypatch.delenv("SMTP_PORT", raising=False)
    return FakeSMTP


def test_send_email_is_plain_text_over_starttls(fake_smtp):
    notifications.send_email("me@example.com", "Hi", "Body\n")
    assert fake_smtp.calls == [
        ("connect", "smtp.gmail.com", 587),
        ("starttls",),
        ("auth", "PLAIN", "bot@example.com", "app-password"),
        ("send", "me@example.com", "Hi", "text/plain", "Body\n"),
    ]


def test_a_rejected_login_says_why_and_tries_once(fake_smtp, monkeypatch):
    # smtplib's own login retried with AUTH LOGIN on the connection Gmail had
    # closed, and the 535 surfaced as "Connection unexpectedly closed".
    monkeypatch.setattr(fake_smtp, "reject", True)
    error = notifications.send_and_log(_state(), "me@example.com", "Hi", "Body\n")
    assert [c[0] for c in fake_smtp.calls] == ["connect", "starttls", "auth"]
    assert error.startswith(
        "SMTPAuthenticationError: 535 5.7.8 Username and Password not accepted. "
        "For more information, go to 5.7.8 https://support.google.com/"
    )
    assert "app password" in error


def test_send_email_refuses_without_credentials(monkeypatch):
    monkeypatch.delenv("SMTP_USER", raising=False)
    monkeypatch.delenv("SMTP_PASSWORD", raising=False)
    with pytest.raises(RuntimeError, match="SMTP_USER, SMTP_PASSWORD"):
        notifications.send_email("me@example.com", "Hi", "Body")


def test_a_replay_is_never_watched():
    state = AppState()
    state.replay = object()
    notifications.start_watch(state)
    assert "_notify_watch" not in state.__dict__


def _tab_app():
    from agent_stonks.notifications_ui import notifications_panel
    from agent_stonks.state import AppState

    notifications_panel(AppState())


def test_tab_shows_stored_settings_and_writes_a_change(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.delenv("SMTP_USER", raising=False)
    monkeypatch.delenv("SMTP_PASSWORD", raising=False)
    notifications.save_settings(Settings(email_to="me@example.com", on_transaction=True))
    at = AppTest.from_function(_tab_app).run()
    assert not at.exception
    assert at.text_input(key="notify_email_to").value == "me@example.com"
    assert at.checkbox(key="notify_on_transaction").value is True
    assert at.number_input(key="notify_interval_min").disabled
    assert "SMTP_USER and SMTP_PASSWORD" in at.warning[0].value
    assert at.button(key="notify_test").disabled

    at.checkbox(key="notify_every_n").check().run()
    at.number_input(key="notify_interval_min").set_value(15).run()
    assert notifications.load_settings() == Settings(
        email_to="me@example.com", on_transaction=True, every_n_minutes=True, interval_minutes=15,
    )
    assert not at.number_input(key="notify_interval_min").disabled


def test_tab_picks_up_a_change_made_in_another_tab(monkeypatch):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(_tab_app).run()
    assert at.text_input(key="notify_email_to").value == ""
    notifications.save_settings(Settings(email_to="other@example.com"))
    at.run()
    assert at.text_input(key="notify_email_to").value == "other@example.com"


def test_tab_test_button_sends_the_status_email(monkeypatch, outbox):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("SMTP_USER", "bot@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "app-password")
    notifications.save_settings(Settings(email_to="me@example.com"))
    at = AppTest.from_function(_tab_app).run()
    at.button(key="notify_test").click().run()
    assert not at.exception
    assert len(outbox) == 1
    assert outbox[0]["to"] == "me@example.com"
    assert outbox[0]["subject"].startswith("Test: Agent Stonks ")
    assert "No ledger yet: the agent has not been started." in outbox[0]["body"]
    assert at.success[0].value == "Sent to me@example.com."
