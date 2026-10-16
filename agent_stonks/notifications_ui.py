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
from datetime import datetime

import streamlit as st

from . import notifications
from .market_hours import MARKET_TZ

# Widget key -> Settings field.
_FIELDS = {
    "notify_email_to": "email_to",
    "notify_on_transaction": "on_transaction",
    "notify_every_n": "every_n_minutes",
    "notify_interval_min": "interval_minutes",
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
        ):
            with st.spinner(f"Sending to {to}…"):
                error = notifications.send_and_log(
                    state, to, "Agent Stonks test email",
                    "Notifications from Agent Stonks reach this address.\n",
                )
            if error:
                st.error(f"Not sent: {error}")
            else:
                st.success(f"Sent to {to}.")

    st.markdown("**Sent**")
    _sent_log(state)
