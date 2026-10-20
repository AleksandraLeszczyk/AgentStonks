"""Stop the agent by email: an email whose body is just "stop" sells every
open position at market and stops the agent -- the sidebar's Sell all,
from wherever the user is -- and the app answers with what it sold.

- **Read from the app's own mailbox**, the account the notifications are sent
  from: IMAP over SSL with the same SMTP_USER and SMTP_PASSWORD (a Gmail app
  password reads IMAP too), at the SMTP host's "imap." twin -- imap.gmail.com
  -- unless IMAP_HOST / IMAP_PORT / IMAP_USER / IMAP_PASSWORD say otherwise.
  Read-only: nothing in the mailbox is marked, moved or deleted.
- **Polled every POLL_SEC** on a background thread per AppState, the way
  `notifications.start_watch` sends: the agent runs with the page closed, so
  the stop has to work with it closed too. One connection, kept open.
- **The command** is the body, case and the whitespace round it aside: "stop"
  and nothing else. The plain-text part, or the HTML part's text without one.
  A signature or a quoted reply under the word makes it not a command, so a
  reply to a notification that happens to mention stopping cannot sell.
- **Only from the notification address** (the Notifications tab's "Send to"),
  and only when the receiving server vouches for it: anyone can type a From
  line, so the topmost Authentication-Results header -- the one the mailbox's
  own server added -- must show DMARC, or a DKIM signature, passing for the
  From address's domain. A stop that fails this is answered, not obeyed; a
  stranger's mail is not answered at all.
- **Once each, received since stop-by-email was switched on, the same ET
  day.** The Message-IDs handled are written to ``data/email_stop.json``, so a
  restart neither misses a stop that arrived while the app was down -- it
  would otherwise resume the run -- nor obeys one twice. A stop from an
  earlier day is answered, not obeyed.
- **Obeying** is `agent.sell_everything_and_stop` and an immediate save, so a
  restart right after does not resume the stopped run. Each AppState with a
  ledger or a running agent obeys on its own, and answers.
- **Never a replay**: Dummy data trades nothing real.
"""
from __future__ import annotations

import email
import html
import imaplib
import json
import logging
import os
import re
import ssl
import threading
import time
import weakref
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email import policy
from email.message import EmailMessage
from email.utils import parseaddr
from pathlib import Path

from . import notifications, session_store
from .agent import sell_everything_and_stop
from .clock import parse_iso
from .market_hours import MARKET_TZ
from .notifications import _VENUES, _et, _money, _qty
from .state import append_agent_log

logger = logging.getLogger(__name__)

HANDLED_PATH = Path(__file__).resolve().parent.parent / "data" / "email_stop.json"

# How often the mailbox is looked at: a stop is obeyed within this.
POLL_SEC = 10.0
# After the mailbox could not be read, wait this long before trying again.
RETRY_SEC = 60.0
IMAP_TIMEOUT_SEC = 20
# Handled Message-IDs are kept this long: past it, the same-day rule refuses
# the message anyway.
HANDLED_KEEP_DAYS = 7

_file_lock = threading.Lock()


# --- the mailbox ---------------------------------------------------------------


def _imap_host_for(smtp_host: str) -> str:
    host = smtp_host.strip().lower()
    return "imap." + host[len("smtp."):] if host.startswith("smtp.") else host


@dataclass
class ImapConfig:
    host: str
    port: int
    user: str
    password: str

    @classmethod
    def from_env(cls) -> "ImapConfig":
        smtp = notifications.SmtpConfig.from_env()
        return cls(
            host=os.getenv("IMAP_HOST", "").strip() or _imap_host_for(smtp.host),
            port=int(os.getenv("IMAP_PORT", "").strip() or 993),
            user=os.getenv("IMAP_USER", "").strip() or smtp.user,
            password=os.getenv("IMAP_PASSWORD", "") or smtp.password,
        )

    def missing(self) -> "list[str]":
        """The environment variables still to set, empty when it can read."""
        return [
            name for name, value in (
                ("SMTP_USER", self.user), ("SMTP_PASSWORD", self.password),
            ) if not value
        ]


class Mailbox:
    """The app account's INBOX over one IMAP connection, opened on first use
    and kept between polls. Raises on any failure; `close` before retrying."""

    def __init__(self, config: ImapConfig) -> None:
        self.config = config
        self.conn: "imaplib.IMAP4 | None" = None
        self.uidvalidity = ""

    def _open(self) -> imaplib.IMAP4:
        if self.conn is None:
            conn = imaplib.IMAP4_SSL(
                self.config.host, self.config.port,
                ssl_context=ssl.create_default_context(), timeout=IMAP_TIMEOUT_SEC,
            )
            conn.login(self.config.user, self.config.password)
            conn.select("INBOX", readonly=True)
            self.uidvalidity = (conn.response("UIDVALIDITY")[1] or [b""])[0].decode()
            self.conn = conn
        else:
            # Lets the server tell a kept connection what has arrived since.
            self.conn.noop()
        return self.conn

    def uids_since(self, since: datetime) -> "list[str]":
        """The messages received on or after `since`'s date, as
        "<uidvalidity>:<uid>" -- the date by the server's clock, so callers
        filter by the time `fetch` gives."""
        conn = self._open()
        typ, data = conn.uid("search", None, "SINCE", since.strftime("%d-%b-%Y"))
        if typ != "OK":
            raise RuntimeError(f"IMAP search refused: {data}")
        return [f"{self.uidvalidity}:{uid.decode()}" for uid in (data[0] or b"").split()]

    def fetch(self, key: str) -> "tuple[datetime, bytes]":
        """(received, the raw message) for one of `uids_since`'s keys, without
        marking it read."""
        conn = self._open()
        uid = key.rpartition(":")[2]
        typ, data = conn.uid("fetch", uid, "(INTERNALDATE BODY.PEEK[])")
        part = next((p for p in data or [] if isinstance(p, tuple)), None)
        if typ != "OK" or part is None:
            raise RuntimeError(f"IMAP fetch of {uid} refused: {data}")
        stamp = imaplib.Internaldate2tuple(part[0])
        received = (
            datetime.fromtimestamp(time.mktime(stamp), timezone.utc)
            if stamp else datetime.now(timezone.utc)
        )
        return received, part[1]

    def close(self) -> None:
        conn, self.conn = self.conn, None
        if conn is not None:
            try:
                conn.logout()
            except Exception:
                pass


# --- what a message says ----------------------------------------------------------


def body_text(msg: EmailMessage) -> str:
    """The message's text: its plain-text part, or its HTML part's text."""
    part = msg.get_body(preferencelist=("plain", "html"))
    if part is None:
        return ""
    try:
        text = part.get_content()
    except (LookupError, ValueError):
        return ""
    if part.get_content_subtype() == "html":
        text = re.sub(r"(?is)<(head|style|script)\b.*?</\1\s*>", " ", text)
        text = html.unescape(re.sub(r"<[^>]*>", " ", text))
    return str(text)


def is_stop(msg: EmailMessage) -> bool:
    """Whether the body is the word "stop", any case, with nothing else but
    whitespace round it."""
    text = body_text(msg).replace("​", "").replace("﻿", "")
    return " ".join(text.split()).casefold() == "stop"


def sender(msg: EmailMessage) -> str:
    return parseaddr(str(msg.get("From") or ""))[1].strip().lower()


def _aligned(signer: str, domain: str) -> bool:
    signer = signer.strip().lower().lstrip("@")
    return bool(signer) and (domain == signer or domain.endswith("." + signer))


def authenticated(msg: EmailMessage) -> bool:
    """Whether the receiving server vouches for the From address: DMARC, or a
    DKIM signature, passing for its domain in the topmost Authentication-
    Results header. That one is the mailbox's own server's; any under it
    came with the message, and could say anything."""
    results = msg.get_all("Authentication-Results") or []
    domain = sender(msg).rpartition("@")[2]
    if not results or not domain:
        return False
    # The first clause names the server that checked; the rest are results.
    for clause in str(results[0]).lower().split(";")[1:]:
        words = re.sub(r"\([^)]*\)", " ", clause).split()
        if not words:
            continue
        method, _, result = words[0].partition("=")
        if result != "pass":
            continue
        props = dict(w.split("=", 1) for w in words[1:] if "=" in w)
        if method == "dmarc" and _aligned(props.get("header.from", ""), domain):
            return True
        if method == "dkim":
            signer = props.get("header.d") or props.get("header.i", "").rpartition("@")[2]
            if _aligned(signer, domain):
                return True
    return False


# --- handled stops ----------------------------------------------------------------


def load_handled(path: "Path | None" = None) -> "dict[str, str]":
    """{Message-ID: when handled, UTC ISO} of the stops already answered."""
    path = path or HANDLED_PATH
    try:
        raw = json.loads(path.read_text())
        return {str(k): str(v) for k, v in (raw.get("handled") or {}).items()}
    except FileNotFoundError:
        return {}
    except (OSError, ValueError, AttributeError):
        logger.exception("Reading %s failed; no stop email counts as handled", path)
        return {}


def remember_handled(key: str, now: datetime, path: "Path | None" = None) -> None:
    path = path or HANDLED_PATH
    with _file_lock:
        handled = load_handled(path)
        handled[key] = now.isoformat()
        keep_from = now - timedelta(days=HANDLED_KEEP_DAYS)
        handled = {k: v for k, v in handled.items() if (parse_iso(v) or now) >= keep_from}
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"handled": handled}, indent=2))
            os.replace(tmp, path)
        except OSError:
            # Still handled in this process; a restart might answer it again.
            logger.exception("Writing %s failed", path)


# --- obeying ----------------------------------------------------------------------


def _venue(state) -> str:
    return _VENUES.get(state.trading_mode, state.trading_mode)


def obey(state, from_addr: str, received: datetime, now: datetime) -> "tuple[str, str]":
    """Sell everything and stop `state`'s agent, save at once, and return the
    answer's (subject, body)."""
    append_agent_log(state, {
        "type": "status",
        "text": f"Stop email from {from_addr} ({_et(received)}): stopping the agent "
                "and selling every open position.",
    })
    was_running = bool(state.agent_running)
    sold, errors = sell_everything_and_stop(state)
    # At once, as the Sell all button does: a crash right after must not
    # resume the run this stopped.
    session_store.save(state)

    filled = [d for d in sold if d.status == "filled"]
    refused = [d for d in sold if d.status != "filled"]
    lines = [
        f"Your stop email ({_et(received)}) was obeyed at {_et(now)}: the agent "
        + ("stopped" if was_running else "was not running")
        + f" on {_venue(state)}."
    ]
    for d in filled:
        lines.append(f"Sold {_qty(d.filled_quantity)} {d.symbol} @ {_money(d.price or 0.0)}.")
    for d in refused:
        lines.append(f"Not sold: {d.symbol} ({' '.join((d.reasoning or d.status).split())}).")
    lines.extend(f"Not sold: {e}" for e in errors)
    if not sold and not errors:
        lines.append("There were no open positions to sell.")
    tracker = state.decision_tracker
    if tracker is not None:
        snap = tracker.snapshot()
        lines.append(f"Cash: {_money(float(snap['cash']))}.")
        held = [f"{_qty(q)} {s}" for s, q in snap["positions"].items() if q]
        lines.append(f"Still open: {', '.join(held)}." if held else "Open: none.")
    lines.append("It stays stopped until ▶ Start Agent in the app.")

    if refused or errors:
        subject = "Stop: agent stopped, NOT everything sold"
    elif filled:
        subject = "Stopped: sold " + ", ".join(
            f"{_qty(d.filled_quantity)} {d.symbol} @ {_money(d.price or 0.0)}" for d in filled
        )
    else:
        subject = "Stopped: nothing to sell"
    return subject, "\n".join(lines) + "\n"


class StopWatcher:
    """Decides, one poll at a time, which stop emails `state` obeys.

    Separate from the thread so a test can drive `poll` with its own mailbox
    and clock."""

    def __init__(self, handled: "set[str] | None" = None) -> None:
        # Stops answered, here or by an earlier run (`HANDLED_PATH`).
        self.handled = set(load_handled()) if handled is None else set(handled)
        # Messages already read this run, stop or not: each is fetched once.
        self.examined: set[str] = set()
        # Since when, when the settings carry no time: this run's first look.
        self.armed_at: "datetime | None" = None

    def poll(self, state, settings: notifications.Settings, mailbox,
             now: "datetime | None" = None) -> None:
        now = now or datetime.now(timezone.utc)
        allowed = {a.strip().lower() for a in settings.email_to.split(",") if a.strip()}
        if not settings.stop_by_email or not allowed:
            self.armed_at = None
            return
        if self.armed_at is None:
            self.armed_at = now
        armed = parse_iso(settings.stop_armed_at) or self.armed_at
        today = now.astimezone(MARKET_TZ).date()
        # Yesterday's too, so a stop sent while the app was down overnight is
        # answered -- and the server's date may not be ET's.
        since = max(armed, now - timedelta(days=1))
        for key in mailbox.uids_since(since):
            if key in self.examined:
                continue
            received, raw = mailbox.fetch(key)
            self.examined.add(key)
            msg = email.message_from_bytes(raw, policy=policy.default)
            if received < armed or not is_stop(msg):
                continue
            from_addr = sender(msg)
            if from_addr not in allowed:
                logger.warning("Stop email from %s ignored: not the notification address", from_addr)
                continue
            msg_id = str(msg.get("Message-ID") or "").strip() or key
            if msg_id in self.handled:
                continue
            self.handled.add(msg_id)
            remember_handled(msg_id, now)
            self._answer(state, msg, from_addr, received, today, now)

    def _answer(self, state, msg, from_addr: str, received: datetime, today, now) -> None:
        if not authenticated(msg):
            logger.warning("Stop email from %s not obeyed: not authenticated", from_addr)
            notifications.send_and_log(
                state, from_addr, "Stop NOT obeyed: sender not verified",
                f"A stop email from {from_addr} ({_et(received)}) was not obeyed: the "
                "mail server did not confirm it came from that address (no DKIM or "
                "DMARC pass). The agent is unchanged.\n",
            )
            return
        if received.astimezone(MARKET_TZ).date() != today:
            notifications.send_and_log(
                state, from_addr, "Stop NOT obeyed: sent on an earlier day",
                f"Your stop email of {received.astimezone(MARKET_TZ):%Y-%m-%d %H:%M ET} "
                "reached the app only today, and a stop counts on the day it is sent. "
                "The agent is unchanged; send stop again to stop it now.\n",
            )
            return
        if state.decision_tracker is None and not state.agent_running:
            logger.info("Stop email from %s: this session has no agent or ledger", from_addr)
            return
        subject, body = obey(state, from_addr, received, now)
        notifications.send_and_log(state, from_addr, subject, body)


def status(state) -> "dict | None":
    """{"checked_at": epoch s, "error": str | None} of `state`'s watcher's
    last look at the mailbox, None before its first."""
    return state.__dict__.get("_email_stop_status")


def start_watch(state, interval: float = POLL_SEC) -> None:
    """Watch the app's mailbox for stop emails for `state` on a background
    thread. Idempotent.

    Holds only a weak reference, so the thread ends with the session."""
    if getattr(state, "replay", None) is not None:
        return
    thread = state.__dict__.get("_email_stop_watch")
    if thread is not None and thread.is_alive():
        return
    ref = weakref.ref(state)
    watcher = StopWatcher()

    def run() -> None:
        mailbox: "Mailbox | None" = None
        last_error = None
        while True:
            current = ref()
            if current is None:
                break
            wait = interval
            try:
                settings = notifications.load_settings()
                config = ImapConfig.from_env()
                if settings.stop_by_email and not config.missing():
                    if mailbox is None or mailbox.config != config:
                        if mailbox is not None:
                            mailbox.close()
                        mailbox = Mailbox(config)
                    watcher.poll(current, settings, mailbox)
                    current.__dict__["_email_stop_status"] = {"checked_at": time.time(), "error": None}
                    last_error = None
                elif mailbox is not None:
                    mailbox.close()
                    mailbox = None
            except Exception as exc:  # one bad look must not end the watching
                error = f"{type(exc).__name__}: {exc}"
                if error != last_error:
                    logger.warning("Stop-email check failed: %s", error)
                last_error = error
                current.__dict__["_email_stop_status"] = {"checked_at": time.time(), "error": error}
                if mailbox is not None:
                    mailbox.close()
                    mailbox = None
                wait = RETRY_SEC
            del current
            time.sleep(wait)
        if mailbox is not None:
            mailbox.close()

    thread = threading.Thread(target=run, daemon=True, name="email-stop")
    state.__dict__["_email_stop_watch"] = thread
    thread.start()
