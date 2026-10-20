"""The live dashboard's Notifications tab: where and when to email.

The settings are written to ``data/notifications.json`` as they change rather
than kept in session state, because the sender (`notifications.start_watch`)
runs on a background thread with no browser session to read them from. Each
field is written on its own, so a second tab that is only looking cannot undo
a change made in the first; a change made in another tab is picked up here on
the next run.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone

import streamlit as st

from . import email_stop, notifications
from .market_hours import MARKET_TZ

# Widget key -> Settings field.
_FIELDS = {
    "notify_email_to": "email_to",
    "notify_on_transaction": "on_transaction",
    "notify_every_n": "every_n_minutes",
    "notify_interval_min": "interval_minutes",
    "notify_stop_by_email": "stop_by_email",
}
# The file's settings as this session last saw (or wrote) them.
_SEEN_KEY = "notify_seen_settings"


def _seed() -> None:
    """Put the file's settings into the widgets, where this session has not
    drawn them yet or another tab has changed the file since."""
    stored = asdict(notifications.load_settings())
    if st.session_state.get(_SEEN_KEY) == stored and all(k in st.session_state for k in _FIELDS):
        return
    for key, field in _FIELDS.items():
        st.session_state[key] = stored[field]
    st.session_state[_SEEN_KEY] = stored


def _write(key: str) -> None:
    """on_change: store one widget's new value."""
    settings = notifications.load_settings()
    value = st.session_state[key]
    if key == "notify_email_to":
        value = value.strip()
    if key == "notify_stop_by_email" and value:
        # A stop sent before it was switched on is not one.
        settings.stop_armed_at = datetime.now(timezone.utc).isoformat()
    setattr(settings, _FIELDS[key], value)
    notifications.save_settings(settings)
    st.session_state[_SEEN_KEY] = asdict(settings)


@st.fragment(run_every=10)
def _sent_log(state) -> None:
    log = notifications.sent_log(state)
    if not log:
        st.caption("Nothing sent yet this session.")
        return
    for entry in log:
        at = datetime.fromtimestamp(entry["at"], MARKET_TZ).strftime("%H:%M:%S ET")
        # Subjects carry prices: a bare $ pair is read as LaTeX.
        subject = entry["subject"].replace("$", "\\$")
        if entry["error"]:
            error = entry["error"].replace("$", "\\$")
            st.caption(f":red[✗] {at} — {subject} — failed: {error}")
        else:
            st.caption(f":green[✓] {at} — {subject} → {entry['to']}")


def notifications_panel(state) -> None:
    """Email settings, a test send, and what has gone out."""
    _seed()
    smtp = notifications.SmtpConfig.from_env()
    missing = smtp.missing()

    with st.container(border=True):
        st.markdown("**Email**")
        st.caption(
            "Short plain-text emails, sent from the app's server: they arrive with "
            "this page closed. Nothing is sent while Dummy data replays a past session."
        )
        st.text_input(
            "Send to",
            key="notify_email_to",
            placeholder="you@example.com",
            on_change=_write,
            args=("notify_email_to",),
            help="Leave empty to send nothing.",
        )
        st.checkbox(
            "On transaction",
            key="notify_on_transaction",
            on_change=_write,
            args=("notify_on_transaction",),
            help="One email per filled buy or sell: what, how many, the price, the "
            "position and cash after it, and the reason. Fills within a few seconds "
            "of each other share one email. Trades already in the ledger when the "
            "app (re)starts are not sent.",
        )
        c1, c2 = st.columns([1, 2], vertical_alignment="center")
        with c1:
            st.checkbox(
                "Every N minutes",
                key="notify_every_n",
                on_change=_write,
                args=("notify_every_n",),
                help="A status email while the agent is running: portfolio value and "
                "its change since the start, cash, open positions and the trades since "
                "the last one. The first comes N minutes after the agent starts.",
            )
        with c2:
            st.number_input(
                "N (minutes)",
                min_value=1,
                max_value=480,
                step=5,
                key="notify_interval_min",
                on_change=_write,
                args=("notify_interval_min",),
                disabled=not st.session_state.get("notify_every_n"),
            )

        if missing:
            st.warning(
                f"Email can't be sent yet: set {' and '.join(missing)} in .env, then "
                "refresh this page. For Gmail, SMTP_PASSWORD is an app password "
                "(Google account → Security → App passwords). SMTP_HOST and SMTP_PORT "
                "default to smtp.gmail.com and 587; SMTP_FROM defaults to SMTP_USER.",
                icon=":material/mail_lock:",
            )
        else:
            st.caption(f"Sending as {smtp.sender} via {smtp.host}:{smtp.port}.")

        to = (st.session_state.get("notify_email_to") or "").strip()
        if st.button(
            "Send test email",
            icon=":material/send:",
            disabled=bool(missing) or not to,
            key="notify_test",
            help="Sends the status email now, as it reads at this moment, with "
            "today's trades — whether or not the agent is running.",
        ):
            with st.spinner(f"Sending to {to}…"):
                subject, body = notifications.current_status_email(state)
                error = notifications.send_and_log(state, to, subject, body)
            if error:
                st.error(f"Not sent: {error}")
            else:
                st.success(f"Sent to {to}.")

    _stop_by_email_box(state, to)

    st.markdown("**Sent**")
    _sent_log(state)


@st.fragment(run_every=10)
def _stop_watch_status(state) -> None:
    if not st.session_state.get("notify_stop_by_email"):
        return
    seen = email_stop.status(state)
    if seen is None:
        st.caption("Not looked at the mailbox yet.")
        return
    at = datetime.fromtimestamp(seen["checked_at"], MARKET_TZ).strftime("%H:%M:%S ET")
    if seen["error"]:
        st.caption(f":red[✗] {at} — the mailbox could not be read: {seen['error']}. "
                   f"Trying again every {email_stop.RETRY_SEC:.0f} s.")
    else:
        st.caption(f":green[✓] Mailbox checked {at}, every {email_stop.POLL_SEC:.0f} s.")


def _stop_by_email_box(state, to: str) -> None:
    imap = email_stop.ImapConfig.from_env()
    with st.container(border=True):
        st.markdown("**Stop by email**")
        st.checkbox(
            "Stop the agent by email",
            key="notify_stop_by_email",
            on_change=_write,
            args=("notify_stop_by_email",),
            help="An email whose body is just \"stop\" (any case, spaces and line "
            "breaks round it) sells every open position at market and stops the "
            "agent — the sidebar's Sell all — and the app answers with what it "
            "sold. Only from the Send to address, and only when its mail server "
            "signed it (DKIM/DMARC), so a forged sender is refused. Each stop "
            "counts once, on the day it is sent, and only after this box was "
            "ticked; one that arrives while the app is down is obeyed when it is "
            "back. A signature or quoted text under the word makes it not a stop.",
        )
        if imap.missing():
            st.warning(
                f"The mailbox can't be read yet: set {' and '.join(imap.missing())} in "
                ".env — the same account the emails are sent from.",
                icon=":material/mail_lock:",
            )
        elif not to:
            st.caption("Set **Send to** above: only emails from that address are obeyed.")
        else:
            st.caption(
                f"Email **stop** to {imap.user} from {to}. Read from "
                f"{imap.host}:{imap.port} with the sending account's password."
            )
        _stop_watch_status(state)
