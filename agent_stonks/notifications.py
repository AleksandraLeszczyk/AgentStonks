"""Email notifications: a short plain-text email when a trade fills, and/or a
status email every N minutes while the agent runs.

The agent trades on background threads whether or not a browser is looking,
so the emails are sent from a background thread too (`start_watch`), one per
AppState, the way `session_store.start_autosave` saves it. The trade chime
(`trade_sound`) is the in-page cue; this is the one that reaches a user who
has closed the page.

- **Settings** live in ``data/notifications.json``, not in a browser session:
  the watcher reads them without one. Written from the Notifications tab.
- **SMTP credentials** come from the environment (.env), never from that file:
  SMTP_USER and SMTP_PASSWORD (for Gmail, an app password), optionally
  SMTP_HOST / SMTP_PORT (smtp.gmail.com / 587) and SMTP_FROM (SMTP_USER).
- **Only fills** are announced, each once. What a ledger already holds when
  the watcher first sees it -- a restored day, a continued run -- is adopted
  silently, so a restart does not send an email per old trade. Fills that land
  inside one poll go out in one email.
- **The status email** goes every N minutes while the agent is running, the
  first one N minutes after it started (or after the option was switched on).
  The tab's "Send test email" sends it once, as it reads at that moment.
- **Never a replay**: Dummy data trades nothing real.
- A failed send is logged and shown in the tab, not retried: retrying a dead
  server every poll would only fill the log.
"""
from __future__ import annotations

import json
import logging
import os
import smtplib
import ssl
import threading
import time
import weakref
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from pathlib import Path

from .clock import parse_iso
from .market_hours import MARKET_TZ

logger = logging.getLogger(__name__)

SETTINGS_PATH = Path(__file__).resolve().parent.parent / "data" / "notifications.json"

# How often the watcher looks at the ledger. A fill is emailed within this.
POLL_SEC = 5.0
SMTP_TIMEOUT_SEC = 20
# The tab's list of recent sends, per AppState.
SENT_LOG_LIMIT = 20
# A trade's reasoning can run to paragraphs; the email keeps the start.
REASON_CHARS = 200
# Trades listed in a status email; the rest are counted.
SUMMARY_TRADE_LINES = 10

_VENUES = {
    "local": "local simulation",
    "alpaca_paper": "Alpaca paper",
    "alpaca_live": "Alpaca LIVE",
}


@dataclass
class Settings:
    email_to: str = ""
    on_transaction: bool = False
    every_n_minutes: bool = False
    interval_minutes: int = 30

    @property
    def email_on(self) -> bool:
        return bool(self.email_to.strip()) and (self.on_transaction or self.every_n_minutes)


def load_settings(path: "Path | None" = None) -> Settings:
    path = path or SETTINGS_PATH
    try:
        raw = json.loads(path.read_text())
    except FileNotFoundError:
        return Settings()
    except (OSError, ValueError):
        logger.exception("Reading %s failed; notifications are off", path)
        return Settings()
    known = {f.name for f in fields(Settings)}
    settings = Settings(**{k: v for k, v in raw.items() if k in known})
    settings.interval_minutes = max(1, int(settings.interval_minutes or 1))
    return settings


def save_settings(settings: Settings, path: "Path | None" = None) -> None:
    path = path or SETTINGS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(asdict(settings), indent=2))
    os.replace(tmp, path)


# --- sending -------------------------------------------------------------


@dataclass
class SmtpConfig:
    host: str
    port: int
    user: str
    password: str
    sender: str

    @classmethod
    def from_env(cls) -> "SmtpConfig":
        user = os.getenv("SMTP_USER", "").strip()
        return cls(
            host=os.getenv("SMTP_HOST", "").strip() or "smtp.gmail.com",
            port=int(os.getenv("SMTP_PORT", "").strip() or 587),
            user=user,
            password=os.getenv("SMTP_PASSWORD", ""),
            sender=os.getenv("SMTP_FROM", "").strip() or user,
        )

    def missing(self) -> "list[str]":
        """The environment variables still to set, empty when it can send."""
        return [
            name for name, value in (
                ("SMTP_USER", self.user), ("SMTP_PASSWORD", self.password),
            ) if not value
        ]


def send_email(
    to: str, subject: str, body: str, smtp: "SmtpConfig | None" = None
) -> None:
    """Send one plain-text email. Raises on any failure."""
    smtp = smtp or SmtpConfig.from_env()
    missing = smtp.missing()
    if missing:
        raise RuntimeError(f"{', '.join(missing)} not set in the environment")
    msg = EmailMessage()
    msg["From"] = smtp.sender
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)
    context = ssl.create_default_context()
    if smtp.port == 465:
        server = smtplib.SMTP_SSL(smtp.host, smtp.port, timeout=SMTP_TIMEOUT_SEC, context=context)
    else:
        server = smtplib.SMTP(smtp.host, smtp.port, timeout=SMTP_TIMEOUT_SEC)
    with server:
        if smtp.port != 465:
            server.starttls(context=context)
        _login(server, smtp.user, smtp.password)
        server.send_message(msg)


def _login(server: smtplib.SMTP, user: str, password: str) -> None:
    """`server.login`, but with one AUTH method. Refused on its first method,
    smtplib's login tries the next one on the same connection; Gmail has hung
    up by then, so its "535 Username and Password not accepted" reached the
    tab as "SMTPServerDisconnected: Connection unexpectedly closed"."""
    server.ehlo_or_helo_if_needed()
    advertised = server.esmtp_features.get("auth", "").upper().split()
    method = next((m for m in ("PLAIN", "LOGIN") if m in advertised), None)
    if method is None:
        # No AUTH, or only methods we don't pick: smtplib's own error.
        server.login(user, password)
        return
    server.user, server.password = user, password
    server.auth(method, getattr(server, f"auth_{method.lower()}"))


def _describe(exc: Exception) -> str:
    """One readable line for the log and the tab. An SMTP reply's text is
    bytes with line breaks; a 535 gets the usual Gmail cause."""
    if not isinstance(exc, smtplib.SMTPResponseException):
        return f"{type(exc).__name__}: {exc}"
    reply = exc.smtp_error
    if isinstance(reply, bytes):
        reply = reply.decode(errors="replace")
    text = f"{type(exc).__name__}: {exc.smtp_code} {' '.join(str(reply).split())}"
    if exc.smtp_code == 535:
        text += (
            " (for Gmail, SMTP_PASSWORD must be an app password from Google "
            "account → Security → App passwords, not the account password)"
        )
    return text


def send_and_log(state, to: str, subject: str, body: str) -> "str | None":
    """Send, and note the outcome in `state`'s sent log. Returns the error,
    or None when it went."""
    error = None
    try:
        send_email(to, subject, body)
        logger.info("Email sent to %s: %s", to, subject)
    except Exception as exc:
        error = _describe(exc)
        logger.warning("Email to %s failed (%s): %s", to, subject, error)
    log = state.__dict__.setdefault("_notify_sent", [])
    log.append({"at": time.time(), "to": to, "subject": subject, "error": error})
    del log[:-SENT_LOG_LIMIT]
    return error


def sent_log(state) -> "list[dict]":
    """Recent sends for `state`, newest first."""
    return list(reversed(state.__dict__.get("_notify_sent", [])))


# --- what the emails say ----------------------------------------------------


def _money(value: float) -> str:
    return f"-${-value:,.2f}" if value < 0 else f"${value:,.2f}"


def _signed_money(value: float) -> str:
    return f"+{_money(value)}" if value >= 0 else _money(value)


def _qty(value: float) -> str:
    return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.4f}".rstrip("0")


def _et(dt: "datetime | None") -> str:
    return dt.astimezone(MARKET_TZ).strftime("%H:%M ET") if dt else "?"


def fill_key(decision) -> tuple:
    return (decision.ts, decision.symbol, decision.action, decision.filled_quantity, decision.price)


def filled_trades(decisions) -> list:
    return [
        d for d in decisions
        if getattr(d, "action", None) in ("buy", "sell")
        and getattr(d, "status", None) == "filled"
    ]


def _trade_line(d) -> str:
    return (
        f"{_et(parse_iso(d.ts))} {d.action} {_qty(d.filled_quantity)} {d.symbol} "
        f"@ {_money(d.price or 0.0)}"
    )


def transaction_email(trades: list, venue: str) -> "tuple[str, str]":
    """(subject, body) for one or more fills."""
    venue_name = _VENUES.get(venue, venue)
    if len(trades) == 1:
        d = trades[0]
        verb = "Bought" if d.action == "buy" else "Sold"
        subject = f"{verb} {_qty(d.filled_quantity)} {d.symbol} @ {_money(d.price or 0.0)}"
        reason = " ".join((d.reasoning or "").split())
        if len(reason) > REASON_CHARS:
            reason = reason[: REASON_CHARS - 1].rstrip() + "…"
        lines = [
            f"{verb} {_qty(d.filled_quantity)} {d.symbol} at {_money(d.price or 0.0)}, "
            f"{_et(parse_iso(d.ts))}.",
            f"Position: {_qty(d.position_after)} {d.symbol}. Cash: {_money(d.cash_after)}.",
            f"Venue: {venue_name}.",
        ]
        if reason:
            lines.append(f"Why: {reason}")
        return subject, "\n".join(lines) + "\n"
    last = trades[-1]
    subject = f"{len(trades)} trades filled"
    lines = [_trade_line(d) for d in trades]
    lines.append(f"Cash: {_money(last.cash_after)}. Venue: {venue_name}.")
    return subject, "\n".join(lines) + "\n"


def summary_email(
    state,
    trades_since: list,
    now: "datetime | None" = None,
    trades_label: str = "Trades since last update",
) -> "tuple[str, str]":
    """(subject, body) for the status email: what the agent and the stream are
    doing, the portfolio, and `trades_since` under `trades_label`."""
    now = now or datetime.now(timezone.utc)
    tracker = state.decision_tracker
    symbols = ", ".join(state.symbols) or "no symbols"
    running = "running" if state.agent_running else "stopped"
    # Before the first ▶ Start Agent no venue has been resolved, and
    # trading_mode's "local" default is not one the user chose.
    if state.trading_mode_requested:
        venue = f"on {_VENUES.get(state.trading_mode, state.trading_mode)}"
    else:
        venue = "(no venue chosen yet)"

    subject = f"Agent Stonks {_et(now)}"
    lines = [f"{_et(now)}. Agent {running} {venue} ({symbols})."]
    activity = state.agent_activity if state.agent_running else None
    if activity:
        lines.append(f"{activity[1]}.")
    lines.append(f"Stream: {state.status}")
    if tracker is None:
        lines.append("No ledger yet: the agent has not been started.")
        return subject, "\n".join(lines) + "\n"

    snap = tracker.snapshot()
    value = state.mark_to_market()
    start = float(getattr(state, "starting_budget", 0.0) or 0.0)
    if value is not None:
        change = value - start if start else None
        if change is not None:
            pct = change / start * 100
            subject += f": {_money(value)} ({_signed_money(change)})"
            lines.append(
                f"Portfolio: {_money(value)} ({_signed_money(change)}, {pct:+.2f}% since start)."
            )
        else:
            subject += f": {_money(value)}"
            lines.append(f"Portfolio: {_money(value)}.")
    lines.append(f"Cash: {_money(float(snap['cash']))}.")
    held = [(s, q) for s, q in snap["positions"].items() if q]
    if held:
        parts = []
        for symbol, qty in held:
            price = state.mark_price(symbol)
            parts.append(
                f"{_qty(qty)} {symbol}" + (f" @ {_money(price)}" if price is not None else "")
            )
        lines.append(f"Open: {', '.join(parts)}.")
    else:
        lines.append("Open: none.")
    lines.append(f"{trades_label}: {len(trades_since)}")
    for d in trades_since[-SUMMARY_TRADE_LINES:]:
        lines.append(f"  {_trade_line(d)}")
    if len(trades_since) > SUMMARY_TRADE_LINES:
        lines.append(f"  (and {len(trades_since) - SUMMARY_TRADE_LINES} earlier)")
    return subject, "\n".join(lines) + "\n"


def current_status_email(state, now: "datetime | None" = None) -> "tuple[str, str]":
    """(subject, body) for the tab's test send: the status email as it reads
    now, whether or not the agent runs, listing today's trades (there is no
    last update to count from)."""
    now = now or datetime.now(timezone.utc)
    tracker = state.decision_tracker
    fills = filled_trades(tracker.snapshot()["decisions"]) if tracker is not None else []
    today = now.astimezone(MARKET_TZ).date()
    todays = [
        d for d in fills
        if (ts := parse_iso(d.ts)) is not None and ts.astimezone(MARKET_TZ).date() == today
    ]
    subject, body = summary_email(state, todays, now, trades_label="Trades today")
    return f"Test: {subject}", body


# --- the watcher -------------------------------------------------------------


class Watcher:
    """Decides, one poll at a time, what `state` should be emailed about.

    Separate from the thread so a test can drive `poll` with its own clock."""

    def __init__(self) -> None:
        self.seen: "set[tuple] | None" = None
        self.summary_from: "datetime | None" = None

    def poll(self, state, settings: Settings, now: "datetime | None" = None) -> None:
        now = now or datetime.now(timezone.utc)
        tracker = getattr(state, "decision_tracker", None)
        fills = filled_trades(tracker.snapshot()["decisions"]) if tracker is not None else []

        keys = [fill_key(d) for d in fills]
        if self.seen is None:
            # First look: the ledger's history is not news.
            self.seen = set(keys)
            new = []
        else:
            new = [d for d, k in zip(fills, keys) if k not in self.seen]
            self.seen.update(keys)

        if not settings.email_to.strip():
            return
        to = settings.email_to.strip()

        if new and settings.on_transaction:
            subject, body = transaction_email(new, state.trading_mode)
            send_and_log(state, to, subject, body)

        if settings.every_n_minutes and state.agent_running and tracker is not None:
            if self.summary_from is None:
                self.summary_from = now
            elif now - self.summary_from >= timedelta(minutes=settings.interval_minutes):
                since = [
                    d for d in fills
                    if (parse_iso(d.ts) or now) > self.summary_from
                ]
                subject, body = summary_email(state, since, now)
                send_and_log(state, to, subject, body)
                self.summary_from = now
        else:
            self.summary_from = None


def start_watch(state, interval: float = POLL_SEC) -> None:
    """Watch `state` for things to email on a background thread. Idempotent.

    Holds only a weak reference, so the thread ends with the session."""
    if getattr(state, "replay", None) is not None:
        return
    thread = state.__dict__.get("_notify_watch")
    if thread is not None and thread.is_alive():
        return
    ref = weakref.ref(state)
    watcher = Watcher()

    def run() -> None:
        while True:
            current = ref()
            if current is None:
                return
            try:
                watcher.poll(current, load_settings())
            except Exception:  # one bad poll must not end the watching
                logger.exception("Notification check failed")
            del current
            time.sleep(interval)

    thread = threading.Thread(target=run, daemon=True, name="notifications")
    state.__dict__["_notify_watch"] = thread
    thread.start()
