"""Stop by email: what counts as a stop, whose stop is obeyed, once, and what
obeying does -- against a fake mailbox, with the real sell-everything path."""
import threading
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage

import pytest

from agent_stonks import email_stop, notifications
from agent_stonks.broker import Broker
from agent_stonks.decisions import DecisionTracker
from agent_stonks.email_stop import StopWatcher
from agent_stonks.notifications import Settings
from agent_stonks.state import AppState

T0 = datetime(2026, 10, 20, 15, 0, tzinfo=timezone.utc)  # 11:00 ET
ME = "me@gmail.com"
GMAIL_PASS = (
    "mx.google.com; dkim=pass header.i=@gmail.com header.s=20230601 header.b=AbC/d+9=; "
    "spf=pass (google.com: domain of me@gmail.com designates 209.85.220.41 as permitted "
    "sender) smtp.mailfrom=me@gmail.com; dmarc=pass (p=NONE sp=QUARANTINE dis=NONE) "
    "header.from=gmail.com"
)
ON = Settings(email_to=ME, stop_by_email=True, stop_armed_at=(T0 - timedelta(hours=2)).isoformat())


def message(body="stop", sender=ME, auth=(GMAIL_PASS,), msg_id="<a1@mail.gmail.com>", html=None):
    msg = EmailMessage()
    for header in auth:
        msg["Authentication-Results"] = header
    msg["From"] = f"Me <{sender}>"
    msg["To"] = "bot@gmail.com"
    msg["Subject"] = ""
    if msg_id:
        msg["Message-ID"] = msg_id
    if body is not None:
        msg.set_content(body)
    if html is not None:
        if body is None:
            msg.set_content(html, subtype="html")
        else:
            msg.add_alternative(html, subtype="html")
    return msg


class FakeMailbox:
    def __init__(self, *mails):
        self.mails = {}
        self.fetched = []
        for received, msg in mails:
            self.add(received, msg)

    def add(self, received, msg):
        self.mails[f"7:{len(self.mails) + 1}"] = (received, msg.as_bytes())

    def uids_since(self, since):
        return [k for k, (received, _) in self.mails.items() if received.date() >= since.date()]

    def fetch(self, key):
        self.fetched.append(key)
        return self.mails[key]


class ClosedMailbox:
    def uids_since(self, since):
        raise AssertionError("the mailbox was read")

    fetch = uids_since


class FakeBroker(Broker):
    def get_current_price(self, symbol, key, secret, feed="iex") -> float:
        return 230.0

    def submit_order(self, symbol, side, quantity, price) -> dict:
        return {"status": "filled", "filled_qty": quantity, "filled_price": price}


@pytest.fixture
def outbox(monkeypatch):
    sent = []
    monkeypatch.setattr(notifications, "send_email",
                        lambda to, subject, body, smtp=None: sent.append(
                            {"to": to, "subject": subject, "body": body}))
    return sent


@pytest.fixture
def saves(monkeypatch):
    saved = []
    monkeypatch.setattr(email_stop.session_store, "save", lambda state, **kw: saved.append(state))
    return saved


def running(positions=None):
    state = AppState()
    state.set_symbols(["AAPL"])
    state.trading_mode = state.trading_mode_requested = "alpaca_paper"
    tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(), trade_cost=0.0)
    tracker.positions = dict(positions or {"AAPL": 10.0})
    state.decision_tracker = tracker
    state.agent_running = True
    state.agent_stop_event = threading.Event()
    return state


class TestTheCommand:
    @pytest.mark.parametrize("body", ["stop", "STOP", "  Stop \n", "\r\n\tsToP\r\n\r\n", "stop "])
    def test_the_word_alone_any_case_whitespace_round_it(self, body):
        assert email_stop.is_stop(message(body))

    @pytest.mark.parametrize("body", [
        "", "stop please", "don't stop", "stop\n\nSent from my iPhone",
        "stop\n\nOn Tue, Oct 20 the app wrote:\n> Bought 10 AAPL", "sto p", "stopp",
    ])
    def test_anything_else_is_not(self, body):
        assert not email_stop.is_stop(message(body))

    def test_an_html_only_email_is_read_as_its_text(self):
        assert email_stop.is_stop(message(None, html='<div dir="ltr">STOP<br></div>\r\n'))
        assert not email_stop.is_stop(message(None, html="<div>stop</div><div>later</div>"))

    def test_the_plain_part_wins_over_the_html(self):
        assert email_stop.is_stop(message("stop", html="<div>stop</div>"))


class TestTheSender:
    def test_gmail_vouching_for_gmail(self):
        assert email_stop.authenticated(message())

    def test_no_results_or_failing_ones_are_not_enough(self):
        assert not email_stop.authenticated(message(auth=()))
        failed = ("mx.google.com; dkim=fail header.i=@gmail.com; spf=softfail "
                  "smtp.mailfrom=me@gmail.com; dmarc=fail (p=NONE) header.from=gmail.com")
        assert not email_stop.authenticated(message(auth=(failed,)))

    def test_a_pass_for_another_domain_is_not_one_for_this(self):
        other = "mx.google.com; dkim=pass header.i=@evil.example; dmarc=pass header.from=evil.example"
        assert not email_stop.authenticated(message(auth=(other,)))

    def test_only_the_receiving_servers_own_header_counts(self):
        """Headers are added on top: one under the server's came with the
        message, and a forger can write a pass there."""
        failed = "mx.google.com; dkim=none; spf=fail smtp.mailfrom=me@gmail.com; dmarc=fail header.from=gmail.com"
        forged = "mx.google.com; dkim=pass header.i=@gmail.com; dmarc=pass header.from=gmail.com"
        assert not email_stop.authenticated(message(auth=(failed, forged)))

    def test_a_parent_domains_signature_aligns(self):
        signed = "mx.example.net; dkim=pass header.d=example.com header.s=k1"
        assert email_stop.authenticated(message(sender="me@mail.example.com", auth=(signed,)))
        assert not email_stop.authenticated(message(sender="me@notexample.com", auth=(signed,)))


class TestWatcher:
    def test_a_stop_sells_everything_stops_saves_and_answers(self, outbox, saves):
        state = running({"AAPL": 10.0})
        mailbox = FakeMailbox((T0 - timedelta(minutes=1), message("  STOP\n")))
        StopWatcher(handled=set()).poll(state, ON, mailbox, T0)
        assert state.agent_running is False and state.agent_stop_event.is_set()
        assert state.decision_tracker.snapshot()["positions"] == {"AAPL": 0.0}
        assert saves == [state]
        assert "Stop email from me@gmail.com (10:59 ET)" in state.agent_log[0]["text"]
        (reply,) = outbox
        assert reply["to"] == ME
        assert reply["subject"] == "Stopped: sold 10 AAPL @ $230.00"
        assert "obeyed at 11:00 ET: the agent stopped on Alpaca paper" in reply["body"]
        assert "Open: none." in reply["body"]

    def test_each_stop_once_also_across_a_restart(self, outbox, saves):
        state = running()
        mailbox = FakeMailbox((T0, message()))
        watcher = StopWatcher()
        watcher.poll(state, ON, mailbox, T0)
        watcher.poll(state, ON, mailbox, T0 + timedelta(seconds=10))
        assert len(saves) == 1 and mailbox.fetched == ["7:1"]
        # The user started it again; then the app restarted.
        again = running()
        StopWatcher().poll(again, ON, FakeMailbox((T0, message())), T0 + timedelta(minutes=5))
        assert again.agent_running is True and len(outbox) == 1

    def test_a_stop_that_came_while_the_app_was_down_is_obeyed_on_return(self, outbox, saves):
        mailbox = FakeMailbox((T0 - timedelta(minutes=20), message()))
        state = running()
        StopWatcher().poll(state, ON, mailbox, T0)
        assert state.agent_running is False and len(outbox) == 1

    def test_a_second_stop_is_obeyed_too(self, outbox, saves):
        state = running()
        mailbox = FakeMailbox((T0, message()))
        watcher = StopWatcher(handled=set())
        watcher.poll(state, ON, mailbox, T0)
        state.agent_running = True
        mailbox.add(T0 + timedelta(minutes=3), message(msg_id="<a2@mail.gmail.com>"))
        watcher.poll(state, ON, mailbox, T0 + timedelta(minutes=3))
        assert state.agent_running is False and len(outbox) == 2

    def test_a_stranger_is_neither_obeyed_nor_answered(self, outbox, saves):
        state = running()
        stranger = message(sender="someone@gmail.com")
        StopWatcher(handled=set()).poll(state, ON, FakeMailbox((T0, stranger)), T0)
        assert state.agent_running is True and outbox == [] and saves == []

    def test_an_unverified_sender_is_answered_not_obeyed(self, outbox, saves):
        state = running()
        forged = message(auth=("mx.google.com; dkim=none; spf=fail; dmarc=fail header.from=gmail.com",))
        StopWatcher(handled=set()).poll(state, ON, FakeMailbox((T0, forged)), T0)
        assert state.agent_running is True and saves == []
        assert outbox[0]["subject"] == "Stop NOT obeyed: sender not verified"

    def test_a_stop_from_an_earlier_day_is_answered_not_obeyed(self, outbox, saves):
        state = running()
        yesterday = T0 - timedelta(hours=20)  # 15:00 ET the day before
        settings = Settings(email_to=ME, stop_by_email=True,
                            stop_armed_at=(T0 - timedelta(days=3)).isoformat())
        StopWatcher(handled=set()).poll(state, settings, FakeMailbox((yesterday, message())), T0)
        assert state.agent_running is True
        assert outbox[0]["subject"] == "Stop NOT obeyed: sent on an earlier day"

    def test_a_stop_sent_before_it_was_switched_on_is_not_one(self, outbox, saves):
        state = running()
        armed_now = Settings(email_to=ME, stop_by_email=True, stop_armed_at=T0.isoformat())
        old = FakeMailbox((T0 - timedelta(minutes=5), message()))
        StopWatcher(handled=set()).poll(state, armed_now, old, T0)
        assert state.agent_running is True and outbox == []

    def test_without_an_armed_time_this_runs_first_look_arms_it(self, outbox, saves):
        state = running()
        settings = Settings(email_to=ME, stop_by_email=True)
        mailbox = FakeMailbox((T0 - timedelta(minutes=5), message(msg_id="<old@x>")))
        watcher = StopWatcher(handled=set())
        watcher.poll(state, settings, mailbox, T0)
        assert state.agent_running is True
        mailbox.add(T0 + timedelta(seconds=30), message(msg_id="<new@x>"))
        watcher.poll(state, settings, mailbox, T0 + timedelta(minutes=1))
        assert state.agent_running is False

    def test_switched_off_or_with_nobody_to_obey_the_mailbox_is_not_read(self, outbox):
        state = running()
        StopWatcher(handled=set()).poll(state, Settings(email_to=ME), ClosedMailbox(), T0)
        no_address = Settings(stop_by_email=True, stop_armed_at=T0.isoformat())
        StopWatcher(handled=set()).poll(state, no_address, ClosedMailbox(), T0)

    def test_other_mail_is_read_once_and_left_alone(self, outbox, saves):
        state = running()
        mailbox = FakeMailbox((T0, message("Thanks for the update", msg_id="<c@x>")))
        watcher = StopWatcher(handled=set())
        watcher.poll(state, ON, mailbox, T0)
        watcher.poll(state, ON, mailbox, T0 + timedelta(seconds=10))
        assert mailbox.fetched == ["7:1"] and outbox == [] and state.agent_running is True

    def test_a_session_with_no_agent_or_ledger_does_nothing(self, outbox, saves):
        state = AppState()
        StopWatcher(handled=set()).poll(state, ON, FakeMailbox((T0, message())), T0)
        assert outbox == [] and saves == []

    def test_a_stopped_agent_still_sells_what_is_open(self, outbox, saves):
        state = running({"AAPL": 4.0})
        state.agent_running = False
        StopWatcher(handled=set()).poll(state, ON, FakeMailbox((T0, message())), T0)
        assert state.decision_tracker.snapshot()["positions"] == {"AAPL": 0.0}
        assert "the agent was not running" in outbox[0]["body"]

    def test_a_sale_that_fails_says_so_in_the_subject(self, outbox, saves, monkeypatch):
        state = running()

        def no_quote(*a, **k):
            raise RuntimeError("no quote for AAPL")

        monkeypatch.setattr(state.decision_tracker.broker, "get_current_price", no_quote)
        StopWatcher(handled=set()).poll(state, ON, FakeMailbox((T0, message())), T0)
        assert state.agent_running is False
        assert outbox[0]["subject"] == "Stop: agent stopped, NOT everything sold"
        assert "Not sold:" in outbox[0]["body"] and "Still open: 10 AAPL." in outbox[0]["body"]


def test_handled_stops_are_kept_a_week(tmp_path):
    path = tmp_path / "handled.json"
    email_stop.remember_handled("<old@x>", T0 - timedelta(days=8), path)
    email_stop.remember_handled("<new@x>", T0, path)
    assert set(email_stop.load_handled(path)) == {"<new@x>"}


def test_the_mailbox_is_the_sending_account_on_imap(monkeypatch):
    for name in ("IMAP_HOST", "IMAP_PORT", "IMAP_USER", "IMAP_PASSWORD", "SMTP_HOST"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SMTP_USER", "bot@gmail.com")
    monkeypatch.setenv("SMTP_PASSWORD", "app-password")
    config = email_stop.ImapConfig.from_env()
    assert (config.host, config.port, config.user, config.password) == (
        "imap.gmail.com", 993, "bot@gmail.com", "app-password")
    assert config.missing() == []
    monkeypatch.setenv("IMAP_HOST", "mail.example.com")
    assert email_stop.ImapConfig.from_env().host == "mail.example.com"
    monkeypatch.delenv("SMTP_PASSWORD")
    assert email_stop.ImapConfig.from_env().missing() == ["SMTP_PASSWORD"]


def test_a_replay_is_never_watched():
    state = AppState()
    state.replay = object()
    email_stop.start_watch(state)
    assert "_email_stop_watch" not in state.__dict__


def _tab_app():
    from agent_stonks.notifications_ui import notifications_panel
    from agent_stonks.state import AppState

    notifications_panel(AppState())


def test_tab_switches_it_on_from_now(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("SMTP_USER", "bot@gmail.com")
    monkeypatch.setenv("SMTP_PASSWORD", "app-password")
    notifications.save_settings(Settings(email_to=ME))
    at = AppTest.from_function(_tab_app).run()
    assert at.checkbox(key="notify_stop_by_email").value is False
    before = datetime.now(timezone.utc)
    at.checkbox(key="notify_stop_by_email").check().run()
    assert not at.exception
    stored = notifications.load_settings()
    assert stored.stop_by_email is True
    assert datetime.fromisoformat(stored.stop_armed_at) >= before
    assert any("Email **stop** to bot@gmail.com from me@gmail.com" in c.value for c in at.caption)


class FakeImap:
    """imaplib.IMAP4_SSL as Gmail answers it: the shapes `Mailbox` parses."""

    instances: list = []

    def __init__(self, host, port, ssl_context=None, timeout=None):
        self.calls = [("connect", host, port)]
        FakeImap.instances.append(self)

    def login(self, user, password):
        self.calls.append(("login", user))
        return "OK", [b"bot@gmail.com authenticated (Success)"]

    def select(self, mailbox, readonly=False):
        self.calls.append(("select", mailbox, readonly))
        return "OK", [b"1"]

    def response(self, code):
        return code, [b"42"]

    def noop(self):
        self.calls.append(("noop",))
        return "OK", [b"Success"]

    def uid(self, command, *args):
        self.calls.append(("uid", command, *args))
        if command == "search":
            return "OK", [b"7 9"]
        raw = message().as_bytes()
        head = b'2 (UID 9 INTERNALDATE "20-Oct-2026 15:04:05 +0000" BODY[] {%d}' % len(raw)
        return "OK", [(head, raw), b")"]

    def logout(self):
        self.calls.append(("logout",))


def test_the_mailbox_reads_inbox_read_only_over_one_connection(monkeypatch):
    FakeImap.instances.clear()
    monkeypatch.setattr(email_stop.imaplib, "IMAP4_SSL", FakeImap)
    box = email_stop.Mailbox(email_stop.ImapConfig("imap.gmail.com", 993, "bot@gmail.com", "pw"))
    assert box.uids_since(T0) == ["42:7", "42:9"]
    received, raw = box.fetch("42:9")
    assert received == datetime(2026, 10, 20, 15, 4, 5, tzinfo=timezone.utc)
    assert b"stop" in raw
    box.uids_since(T0)
    box.close()
    (conn,) = FakeImap.instances
    assert ("select", "INBOX", True) in conn.calls
    assert ("uid", "search", None, "SINCE", "20-Oct-2026") in conn.calls
    # Never marked read.
    assert ("uid", "fetch", "9", "(INTERNALDATE BODY.PEEK[])") in conn.calls
    assert [c[0] for c in conn.calls].count("noop") == 2 and conn.calls[-1] == ("logout",)
