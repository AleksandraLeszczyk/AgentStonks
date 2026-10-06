"""Replay a past session in the live app, at the speed of the real one.

The sidebar's **Dummy data** switch. A demo of the agent needs a session to
happen in front of an audience, and the market keeps its own hours; this plays
a session that already happened back through the live app as though it were
happening now -- the same chart, the same tabs, the same agents -- one second
per second.

What is borrowed from SimLab, and why
-------------------------------------
SimLab already knows how to make the agent path believe it is a past day: its
store (`simlab.data`) holds the minute tape, the daily history, the news and
the market indicators per day, `SimMarket` answers every "as of t" question
over them, and `simlab.patches` reroutes each live fetch an agent cycle makes
to those answers. A replay here reuses all three, so an agent that trades a
replay trades exactly what a SimLab run of that day would have shown it. What
SimLab does not have is the live app's side: the page, the bar buffer the chart
draws, the forming candle, the agents' own threads and cadence. That is this
module.

The clock is scoped, not pinned
-------------------------------
SimLab pins the whole process (`clock.set_simulated`); the live app cannot,
because a live session -- possibly trading -- runs in the same process. So the
replay's clock (`ReplayClock`) is bound to the replay's own threads
(`clock.bind`): the page's script runs while Dummy data is on, the feed thread
below, and the agent the Agent tab launches, which inherits it
(`clock.inherit`). The patched fetches dispatch the same way: a call from a
bound thread gets the replay's dataset answer, any other thread the real
fetch (`_install_dispatchers`).

How the tape is played
----------------------
The sidebar's ▶ Start downloads the day (and the week before it, for the
history and the momentum band) into SimLab's store, seeds the bar buffer with
everything completed at the start time (09:29 ET by default), prepares the
pre-market briefing -- from the replay's own cache when it has one -- and
waits there. The Agent tab's ▶ Start Agent sets the clock running. From then
on a feed thread does what the Finnhub socket does live: the minute in progress
is a forming candle that walks from the stored bar's open through its extremes
to its close (the order inside a minute is not stored, so it is made up --
the low first on an up minute, the high first on a down one), and at the end of
the minute the stored bar replaces it, exactly, and is published as closed.
News arrives when it was published and wakes the agent as the news socket does.

What a replay will not do
-------------------------
- Send an order anywhere. The venue is always `ReplayBroker`, which fills at
  the replay's own price; the venue picker is not consulted.
- Write anything of the live day's: no session file (`session_store` refuses a
  replay state), no scoring journal, no briefing verdict for the days-off
  check, no SimLab dataset entry. It writes only SimLab's shared store (the
  downloaded day) and its own briefing cache under ``data/replay/``.
- Show what has no point-in-time history: the options chain (walls, net
  gamma) and bid/ask quotes. The chart draws without them.

One replay per process: a demo is one session, and every tab with Dummy data
on shows that one -- so a page that reconnects finds it still running. A replay
whose pages have all gone for `ABANDON_SEC` stops itself, agent included,
rather than spending LLM calls all afternoon for nobody.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from bisect import bisect_right
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path

from . import clock
from . import event_days
from . import minute_momentum
from . import scoring
from . import stream_common
from .broker import Broker
from .config import MAX_BARS
from .market_hours import MARKET_OPEN, MARKET_TZ
from .news import merge_news
from .state import AppState, SymbolState

logger = logging.getLogger(__name__)

REPLAY_DIR = Path(__file__).resolve().parent.parent / "data" / "replay"
BRIEFING_DIR = REPLAY_DIR / "briefings"

# Where a replay starts: a minute before the bell, so the audience sees the
# agent arm before the open, as it does on a real morning.
DEFAULT_START = dtime(9, 29)
# The pre-market briefing is written as of this moment (SimLab's
# `session_context.BRIEFING_TIME`), or the start time if that is earlier.
BRIEFING_TIME = dtime(9, 25)
# How often the feed moves the tape: the forming candle, the price, the news.
TICK_SEC = 1.0
# Weekdays of news before the replayed one, so the News tab does not open
# empty at 09:29 -- the live app loads the latest articles at start too.
PRIOR_NEWS_DAYS = 2
# A replay no page has looked at for this long is stopped (see the docstring).
ABANDON_SEC = 15 * 60
BAR_SEC = 60.0

# The tape a resolved history feed downloads the replay on. Delayed SIP is the
# whole consolidated tape for any day before today.
_STORE_FEED = {"sip": "sip", "sip_delayed": "sip", "iex": "iex", "yfinance": "yfinance"}


# --- which day -----------------------------------------------------------------


def previous_session(today: "date | None" = None) -> date:
    """The weekday before `today` (ET): the replay's default day."""
    day = (today or datetime.now(timezone.utc).astimezone(MARKET_TZ).date()) - timedelta(days=1)
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return day


def last_replayable_day(now: "datetime | None" = None) -> date:
    """The newest weekday whose whole tape can be downloaded at `now`: a day is
    not over, for the store, until 20:30 ET (`simlab.data.day_final_at`)."""
    from simlab import data as sim_data

    day = sim_data.last_final_day(now or datetime.now(timezone.utc))
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return day


def start_moment(day: date, start: dtime = DEFAULT_START) -> datetime:
    """`start` ET on `day`, in UTC, whole seconds."""
    return datetime.combine(day, start.replace(microsecond=0), tzinfo=MARKET_TZ).astimezone(
        timezone.utc
    )


# --- the clock -----------------------------------------------------------------


class ReplayClock:
    """Replayed time: stands still until `run`, then moves with the wall clock.

    Held as one tuple (anchor, monotonic-at-anchor) so a reader on another
    thread never sees a torn pair.
    """

    def __init__(self, start: datetime) -> None:
        self._state: "tuple[datetime, float | None]" = (start.astimezone(timezone.utc), None)

    def now(self) -> datetime:
        anchor, since = self._state
        if since is None:
            return anchor
        return anchor + timedelta(seconds=time.monotonic() - since)

    @property
    def running(self) -> bool:
        return self._state[1] is not None

    def run(self) -> None:
        anchor, since = self._state
        if since is None:
            self._state = (anchor, time.monotonic())

    def pause(self) -> None:
        if self.running:
            self._state = (self.now(), None)


# --- the patched fetches, per thread --------------------------------------------

# (module, attribute) -> the real function the dispatcher stands in front of.
_originals: "dict[tuple[object, str], object]" = {}
_install_lock = threading.Lock()


def _original(module: object, name: str) -> object:
    return _originals.get((module, name)) or getattr(module, name)


def _dispatcher(module: object, name: str, original):
    def dispatch(*args, **kwargs):
        fakes = getattr(clock.scope(), "fakes", None)
        if fakes:
            fake = fakes.get((module, name))
            if fake is not None:
                return fake(*args, **kwargs)
        return original(*args, **kwargs)

    dispatch.__wrapped__ = original
    dispatch.__name__ = getattr(original, "__name__", name)
    return dispatch


def _install_dispatchers(table: "list[tuple[object, str, object]]") -> None:
    """Stand a dispatcher in front of every attribute in `table`, once.

    A dispatcher is transparent to every thread not bound to a replay, so it
    stays installed after the replay ends; a later replay reuses it."""
    with _install_lock:
        for module, name, _ in table:
            key = (module, name)
            if key in _originals:
                continue
            real = getattr(module, name)  # raises if renamed upstream
            _originals[key] = real
            setattr(module, name, _dispatcher(module, name, real))


def _uninstall_dispatchers() -> None:
    """Put every real function back (tests)."""
    with _install_lock:
        for (module, name), real in _originals.items():
            setattr(module, name, real)
        _originals.clear()


# --- the venue -----------------------------------------------------------------


class ReplayBroker(Broker):
    """Fills at the replay's own price -- the tape's price at the replayed
    moment, as `PaperBroker` fills at the latest real trade. Nothing leaves the
    process, whatever venue the Agent tab shows."""

    def __init__(self, session: "ReplaySession") -> None:
        self.session = session

    def get_current_price(self, symbol: str, key: str, secret: str, feed: str = "iex") -> float:
        price = self.session.price(symbol)
        if price is None:
            raise RuntimeError(f"no replayed price for {symbol} at {self.session.now():%H:%M:%S}")
        return price

    def submit_order(self, symbol: str, side: str, quantity: float, price: float) -> dict:
        return {"status": "filled", "filled_qty": quantity, "filled_price": price}

    @property
    def venue(self) -> str:
        return "replay (dummy data)"


# --- the minute in progress -----------------------------------------------------


def _waypoints(bar: dict) -> "list[tuple[float, float]]":
    """(fraction of the minute, price) the forming candle walks through: open,
    both extremes, close. The low comes first on a minute that closed up."""
    o, h, l, c = (float(bar[k]) for k in ("o", "h", "l", "c"))
    first, second = (l, h) if c >= o else (h, l)
    return [(0.0, o), (1 / 3, first), (2 / 3, second), (1.0, c)]


def _round_price(price: float) -> float:
    return round(price, 2) if price >= 1 else round(price, 4)


def forming(bar: dict, frac: float) -> "tuple[float, float, float]":
    """(price, high so far, low so far) `frac` of the way through `bar`."""
    frac = min(max(frac, 0.0), 1.0)
    points = _waypoints(bar)
    price = points[-1][1]
    for (t0, p0), (t1, p1) in zip(points, points[1:]):
        if frac <= t1:
            price = p0 + (p1 - p0) * ((frac - t0) / (t1 - t0) if t1 > t0 else 1.0)
            break
    price = _round_price(price)
    reached = [p for t, p in points if t <= frac] + [price]
    return price, max(reached), min(reached)


def _ts_key(raw: object) -> str:
    return stream_common.bar_ts_key(raw)


def _trade_ts(moment: datetime) -> str:
    """A trade's time the way the Finnhub stream writes it (milliseconds, 'Z'):
    the chart parses every trade's time as one format, so they must all match."""
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


# --- the session ---------------------------------------------------------------


class ReplaySession:
    """One replayed day: its dataset, its clock, its AppState and its feed.

    Doubles as the clock scope its threads are bound to (`now`, `fakes`)."""

    def __init__(
        self,
        market,
        day: date,
        start: datetime,
        feed: str,
        key: str,
        secret: str,
        history_feed: str,
        history_feed_resolved: str,
    ) -> None:
        from simlab import patches

        self.market = market
        self.day = day
        self.start = start
        self.feed = feed
        self.clock = ReplayClock(start)
        table = patches.patch_table(market, original=_original)
        self.fakes = {(module, name): replacement for module, name, replacement in table}
        _install_dispatchers(table)
        # The days-off check reads the morning briefing's shock verdict. Live
        # that is the briefing the Pre-Market tab shows; here it is the
        # replay's own, so the trader and the tab tell the same story. Without
        # one, SimLab's answer: what the live app recorded that morning.
        recorded = self.fakes.get((event_days, "briefing_verdict"))

        def briefing_verdict(symbol, day):
            briefing = (self.app.premarket_briefings or {}).get(str(symbol).upper())
            if briefing is not None and event_days.as_date(day) == self.day:
                return {
                    "shock": briefing.shock,
                    "reason": briefing.shock_reason,
                    "made_at": self.briefing_as_of().astimezone(MARKET_TZ).isoformat(),
                }
            return recorded(symbol, day) if recorded is not None else None

        self.fakes[(event_days, "briefing_verdict")] = briefing_verdict
        self.stop_event = threading.Event()
        self.finished = False
        self.last_seen = time.monotonic()
        self._cursor: "dict[str, int]" = {}
        self._news_after = start
        self._formed_volume: "dict[str, float]" = {}

        app = AppState()
        app.replay = self
        app.set_symbols(market.symbols)
        app.api_key, app.api_secret = key, secret
        app.history_feed = history_feed
        app.history_feed_resolved = history_feed_resolved
        app.timeframe = "1Min"
        # The tape the volumes are on, for everything that reads it -- and the
        # signal Apple Trader takes as "a replay" (`bar_tape_override`): its
        # forecasts read HighLow's history with the environment's keys and its
        # opening window off the buffer, as in a SimLab run.
        app.bar_tape_override = feed
        app.news_status = "Replay — news arrives as it was published"
        self.app = app
        for ss in app.iter_symbol_states():
            self._seed(ss)
        self._set_status()

    # --- the scope ----------------------------------------------------------

    def now(self) -> datetime:
        return self.clock.now()

    def touch(self) -> None:
        """A page is looking (see ABANDON_SEC)."""
        self.last_seen = time.monotonic()

    @property
    def running(self) -> bool:
        return self.clock.running and not self.stop_event.is_set()

    # --- prices -------------------------------------------------------------

    def price(self, symbol: str) -> "float | None":
        ss = self.app.sym(symbol)
        if ss is not None and ss.last_price is not None:
            return float(ss.last_price)
        return self.market.price_at(str(symbol).upper(), self.now())

    # --- seeding ------------------------------------------------------------

    def _seed(self, ss: SymbolState) -> None:
        """Everything a live start would have loaded at the start moment."""
        sym, t0 = ss.symbol, self.start
        series = self.market.series[sym]
        completed = self.market.completed_bars(sym, t0)
        self._cursor[sym] = len(completed)
        src = self.feed
        bars = [{**bar, "src": src} for bar in completed[-MAX_BARS:]]
        today = [
            bar for bar, ts in zip(series.minute_bars, series.minute_ts)
            if ts.astimezone(MARKET_TZ).date() == self.day and ts + timedelta(seconds=BAR_SEC) <= t0
        ]
        with ss.lock:
            ss.bars.clear()
            ss.bars.extend(bars)
            ss.provisional_bar_keys = set()
            ss.daily_bars = self.market.daily_bars_at(sym, t0)
            ss.prev_close = self.market.prev_close(sym, t0)
            last = bars[-1] if bars else None
            ss.last_price = float(last["c"]) if last else ss.prev_close
            ss.previous_minute_high = last.get("h") if last else None
            ss.previous_minute_low = last.get("l") if last else None
            ss.previous_minute_close = last.get("c") if last else None
            ss.day_volume = self.market.day_volume(sym, t0)
            ss.recent_prices.clear()
            if ss.last_price is not None:
                ss.recent_prices.append((time.monotonic(), float(ss.last_price)))
            # The volume-at-price panel reads trades; the session's minutes so
            # far stand in for them, one print per minute at its close.
            ss.trades = [
                {"p": float(b["c"]), "s": float(b.get("v") or 0.0),
                 "t": _trade_ts(clock.parse_iso_strict(b["t"]))}
                for b in today
            ]
            ss.news = self.market.news_at(sym, t0)
            ss.news_impacts = {}
            ss.news_impact_details = {}
        self._momentum_band(ss)

    def _momentum_band(self, ss: SymbolState) -> None:
        """`abs_mean_minute_momentum` and its bands from the stored week before
        the day -- what the live stream start measures from last week, and
        what a SimLab replay of the day reads (`SimMarket.abs_mean_minute_momentum`)."""
        from simlab import data as sim_data

        bars = [
            bar
            for prior in minute_momentum.prior_week_days(self.day)
            for bar in sim_data.load_day_bars(ss.symbol, prior, self.feed)
        ]
        result = minute_momentum.compute(bars, self.day.isoformat())
        if result is None:
            return
        ss.abs_mean_minute_momentum = result["abs_mean_minute_momentum"]
        ss.minute_momentum_profile = minute_momentum.band(result.get("per_minute_moves") or {}) or None
        ss.minute_momentum_change_profile = (
            minute_momentum.band(result.get("per_minute_deltas") or {}) or None
        )

    # --- running ------------------------------------------------------------

    def launch_feed(self) -> None:
        threading.Thread(
            target=self._feed_loop, name=f"replay-feed-{self.day}", daemon=True
        ).start()

    def run(self) -> None:
        """Set the replayed clock running (▶ Start Agent). Idempotent."""
        if self.stop_event.is_set():
            return
        self.clock.run()
        self.app.bars_connected = True
        self._set_status()

    def stop(self, reason: str = "") -> None:
        """Stop the tape and the agent on it; the clock stands where it was."""
        from .agent import stop_agent

        first = not self.stop_event.is_set()
        self.stop_event.set()
        self.clock.pause()
        self.app.bars_connected = False
        stop_agent(self.app)
        if first:
            self._set_status(reason or "stopped")

    def _set_status(self, ended: str = "") -> None:
        et = self.now().astimezone(MARKET_TZ)
        head = f"🎬 Replay of {self.day:%a %Y-%m-%d} · {et:%H:%M:%S} ET"
        if ended:
            self.app.status = f"{head} · {ended}"
        elif self.finished:
            self.app.status = f"{head} · the stored session is over"
        elif self.clock.running:
            self.app.status = f"{head} · playing (dummy data)"
        else:
            self.app.status = f"{head} · ready — ▶ Start Agent in the Agent tab plays it"

    def _feed_loop(self) -> None:
        clock.bind(self)
        try:
            while not self.stop_event.wait(TICK_SEC):
                if time.monotonic() - self.last_seen > ABANDON_SEC:
                    logger.info("Replay of %s: no page for %ss, stopping", self.day, ABANDON_SEC)
                    self.stop("stopped: no page had looked at it for 15 minutes")
                    return
                if not self.clock.running:
                    continue
                self.tick(self.now())
        except Exception:  # a broken tape must say so, not freeze silently
            logger.exception("Replay feed failed")
            self.app.status = "🎬 Replay feed failed — see the app log"
        finally:
            clock.unbind()

    def tick(self, now: datetime) -> None:
        """Bring every symbol's tape up to `now` (the feed thread, every second)."""
        for ss in self.app.iter_symbol_states():
            self._advance(ss, now)
        self._deliver_news(now)
        if not self.finished and all(
            self._cursor[sym] >= len(self.market.series[sym].minute_ts)
            for sym in self.app.symbols
        ):
            self.finished = True
        self._set_status()

    def _advance(self, ss: SymbolState, now: datetime) -> None:
        series = self.market.series[ss.symbol]
        stamps = series.minute_ts
        i = self._cursor[ss.symbol]
        end = bisect_right(stamps, now - timedelta(seconds=BAR_SEC))
        while i < end:
            self._close_bar(ss, series.minute_bars[i], stamps[i])
            i += 1
        self._cursor[ss.symbol] = i
        if i < len(stamps) and stamps[i] <= now:
            frac = (now - stamps[i]).total_seconds() / BAR_SEC
            self._form_bar(ss, series.minute_bars[i], frac, now)

    def _form_bar(self, ss: SymbolState, bar: dict, frac: float, now: datetime) -> None:
        """The minute in progress, as a Finnhub candle is built live: rewritten
        in place on every tick, never published as closed."""
        price, high, low = forming(bar, frac)
        total = float(bar.get("v") or 0.0)
        volume = round(total * frac)
        key = _ts_key(bar["t"])
        with ss.lock:
            last = ss.bars[-1] if ss.bars else None
            if last is not None and _ts_key(last["t"]) == key:
                before = float(last.get("v") or 0.0)
                last.update({"h": max(float(last["h"]), high), "l": min(float(last["l"]), low),
                             "c": price, "v": max(before, volume)})
            else:
                before = 0.0
                ss.bars.append({"t": bar["t"], "o": float(bar["o"]), "h": high, "l": low,
                                "c": price, "v": volume, "src": self.feed})
            added = max(0.0, volume - before)
            self._formed_volume[ss.symbol] = max(before, volume)
            ss.last_price = price
            ss.recent_prices.append((time.monotonic(), price))
            ss.day_volume = self.market.day_volume(ss.symbol, now) + self._formed_volume[ss.symbol]
            if added:
                ss.trades.append({"p": price, "s": added, "t": _trade_ts(now)})
        self._after_tick(ss, price)

    def _close_bar(self, ss: SymbolState, bar: dict, ts: datetime) -> None:
        """The stored bar, exactly, in place of the forming one -- then published
        as the symbol's last completed bar, as the live flush does."""
        final = {**bar, "src": self.feed}
        key = _ts_key(bar["t"])
        done = ts + timedelta(seconds=BAR_SEC)
        close = float(bar["c"])
        with ss.lock:
            last = ss.bars[-1] if ss.bars else None
            if last is not None and _ts_key(last["t"]) == key:
                ss.bars[-1] = final
            else:
                ss.bars.append(final)
            rest = float(bar.get("v") or 0.0) - self._formed_volume.pop(ss.symbol, 0.0)
            if rest > 0:
                ss.trades.append({"p": close, "s": rest, "t": _trade_ts(done)})
            ss.last_price = close
            ss.recent_prices.append((time.monotonic(), close))
            ss.day_volume = self.market.day_volume(ss.symbol, done)
        ss.daily_bars = self.market.daily_bars_at(ss.symbol, done)
        stream_common.record_bar_close(ss, final)
        self._after_tick(ss, close)

    def _after_tick(self, ss: SymbolState, price: float) -> None:
        scoring.record_price(self.app, ss.symbol, price)
        if self.app.decision_tracker is not None:
            self.app.mark_to_market()
        stream_common.fire_due_alerts(ss)

    def _deliver_news(self, now: datetime) -> None:
        """Articles published since the last tick, as the news socket delivers
        them -- and, like it, each one wakes the agent."""
        after, self._news_after = self._news_after, now
        for ss in self.app.iter_symbol_states():
            fresh = self.market.fresh_news(ss.symbol, after, now)
            if not fresh:
                continue
            with ss.lock:
                ss.news, added = merge_news(ss.news, fresh)
            if added:
                article = added[-1]
                text = f"Fresh news arrived for {ss.symbol}."
                if article.get("headline"):
                    text += f" Latest: {article['headline']}"
                self.app.agent_wake_reason = text
                self.app.agent_wake_event.set()

    # --- the briefing -------------------------------------------------------

    def briefing_as_of(self) -> datetime:
        return min(start_moment(self.day, BRIEFING_TIME), self.start)

    def launch_briefing(self, provider: str, model: "str | None", api_key: str,
                        force: bool = False) -> bool:
        """Load or write the pre-market briefing on a background thread, as
        the live ▶ Start does. False (and why, on the app) without a key and
        nothing cached."""
        from . import premarket

        model = model or premarket.DEFAULT_PREMARKET_MODELS.get(provider)
        app = self.app
        app.premarket_phase = "premarket"
        app.premarket_errors = {}
        if not force:
            cached = {
                sym: b for sym in app.symbols
                if (b := load_briefing(sym, self.day, provider, model)) is not None
            }
            if len(cached) == len(app.symbols):
                app.premarket_briefings = cached
                app.premarket_pending = []
                app.premarket_generated_at = self.briefing_as_of()
                app.premarket_status = "Pre-market briefing ready (from the replay's cache)"
                return True
        if not api_key:
            app.premarket_briefings = {}
            app.premarket_pending = []
            app.premarket_status = (
                f"No API key for {provider} and no cached {provider}/{model} briefing for "
                f"{self.day} — set {provider.upper()}_API_KEY to write one."
            )
            return False
        app.premarket_briefings = {}
        app.premarket_pending = list(app.symbols)
        app.premarket_status = "Generating pre-market briefing…"
        threading.Thread(
            target=clock.inherit(self._brief), args=(provider, model, api_key, force),
            name=f"replay-briefing-{self.day}", daemon=True,
        ).start()
        return True

    def _brief(self, provider: str, model: str, api_key: str, force: bool) -> None:
        app = self.app
        cached_count = 0
        for sym in list(app.symbols):
            try:
                briefing = None if force else load_briefing(sym, self.day, provider, model)
                if briefing is not None:
                    cached_count += 1
                else:
                    briefing = write_briefing(
                        sym, self.day, self.feed, self.briefing_as_of(), provider, model,
                        api_key, app.api_key, app.api_secret,
                    )
                app.premarket_briefings = {**app.premarket_briefings, sym: briefing}
            except Exception as exc:
                app.premarket_errors = {**app.premarket_errors, sym: str(exc) or type(exc).__name__}
            app.premarket_pending = [s for s in app.premarket_pending if s != sym]
        app.premarket_generated_at = self.briefing_as_of()
        done, failed = len(app.premarket_briefings), len(app.premarket_errors)
        app.premarket_status = (
            f"Pre-market briefing ready ({done} of {done + failed} symbols"
            + (f", {cached_count} from the replay's cache" if cached_count else "")
            + ")"
            if done else "Briefing failed"
        )


# --- the briefing cache -----------------------------------------------------------


def briefing_path(symbol: str, day: date, provider: str, model: str) -> Path:
    return BRIEFING_DIR / symbol.upper() / f"{day.isoformat()}.{provider}.{model}.json"


def load_briefing(symbol: str, day: date, provider: str, model: str):
    """The cached briefing for (symbol, day, provider, model), or None."""
    from .premarket import PremarketBriefing

    try:
        record = json.loads(briefing_path(symbol, day, provider, model).read_text())
        return PremarketBriefing.model_validate(record["briefing"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def write_briefing(
    symbol: str, day: date, feed: str, as_of: datetime, provider: str, model: str,
    api_key: str, alpaca_key: str = "", alpaca_secret: str = "",
):
    """The live app's pre-market briefing for `day`, from what was known at
    `as_of` (`premarket.generate_premarket_from_data`, the same point-in-time
    inputs SimLab's session briefings read), kept so the next replay of the day
    tells the same story. Raises when the model gives nothing back; a failure
    is not cached."""
    from simlab import session_context as sc

    from . import premarket

    sym = symbol.upper()
    news = None
    if alpaca_key and alpaca_secret:
        try:
            news = sc._fetch_news_before(sym, as_of, alpaca_key, alpaca_secret)
        except Exception as exc:  # noqa: BLE001 -- the store's copy is thinner, not wrong
            logger.warning("Replay briefing for %s: Alpaca news failed (%s); using the store", sym, exc)
    if news is None:
        news = sc._stored_news_before(sym, as_of)
    briefing = premarket.generate_premarket_from_data(
        sym, provider, api_key, as_of=as_of,
        closes=sc._closes_before(sym, feed, day.isoformat()),
        indicators=sc._indicators_before(day.isoformat()),
        news_items=news,
        earnings_text=premarket._earnings_block(sym, as_of),
        model=model,
    )
    if briefing is None:
        raise RuntimeError("the model returned no briefing")
    path = briefing_path(sym, day, provider, model)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "symbol": sym, "day": day.isoformat(), "as_of": as_of.isoformat(),
        "provider": provider, "model": model, "news": len(news),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "briefing": briefing.model_dump(mode="json"),
    }, indent=1))
    return briefing


# --- the process's replay -----------------------------------------------------------

_current: "ReplaySession | None" = None
_idle: "AppState | None" = None
_lock = threading.Lock()


def current() -> "ReplaySession | None":
    return _current


def session_of(state: object) -> "ReplaySession | None":
    """The replay `state` belongs to, or None for a live state (or the idle
    page before a replay's first ▶ Start)."""
    replay = getattr(state, "replay", None)
    return replay if isinstance(replay, ReplaySession) else None


def is_replay(state: object) -> bool:
    return getattr(state, "replay", None) is not None


def app_state() -> AppState:
    """What a page with Dummy data on shows: the current replay's state, or an
    empty one marked as a replay until the first ▶ Start."""
    global _idle
    if _current is not None:
        return _current.app
    with _lock:
        if _idle is None:
            idle = AppState()
            idle.replay = "idle"
            idle.status = "🎬 Dummy data — ▶ Start in the sidebar loads the replayed day"
            _idle = idle
        return _idle


def scope_for(state: object) -> "ReplaySession | None":
    """The clock scope a page showing `state` runs in (None: the real clock)."""
    return session_of(state)


def prepare(
    symbols: "list[str]",
    day: date,
    start: dtime,
    key: str,
    secret: str,
    history_feed: str,
    progress=lambda _msg: None,
) -> ReplaySession:
    """Download `day`, build its replay paused at `start` ET and make it the
    process's replay, stopping the one before. Raises with a sentence the page
    can show when the day cannot be replayed."""
    from simlab import data as sim_data
    from simlab.market import SimMarket

    from . import bar_history

    global _current
    symbols = [s.strip().upper() for s in symbols if s.strip()]
    if not symbols:
        raise ValueError("enter at least one symbol")
    if day.weekday() >= 5:
        raise ValueError(f"{day:%a %Y-%m-%d} is a weekend; pick a weekday")
    newest = last_replayable_day()
    if day > newest:
        raise ValueError(
            f"{day:%Y-%m-%d} is not over yet for the store (a day's tape is whole from "
            f"20:30 ET); the newest day that can be replayed is {newest:%Y-%m-%d}"
        )
    resolved = bar_history.resolve_history_feed(history_feed, symbols[0], key, secret)
    feed = _STORE_FEED.get(resolved, "yfinance")
    if feed == "yfinance" and day < date.today() - timedelta(days=sim_data.YF_MINUTE_WINDOW_DAYS):
        raise ValueError(
            f"{day:%Y-%m-%d} is older than the {sim_data.YF_MINUTE_WINDOW_DAYS} days of minute "
            "history yfinance keeps, and this key has no SIP history"
        )

    if _current is not None:
        _current.stop("replaced by a new replay")

    sessions = sim_data.download_days(symbols, day, day, key, secret, feed, progress)
    if day.isoformat() not in sessions:
        raise ValueError(f"there was no session on {day:%a %Y-%m-%d} (a market holiday?)")
    missing = [s for s in symbols if not sim_data.load_day_bars(s, day, feed)]
    if missing:
        raise ValueError(f"no {feed} minute bars for {', '.join(missing)} on {day:%Y-%m-%d}")
    prior_news = []
    back = day
    while len(prior_news) < PRIOR_NEWS_DAYS:
        back -= timedelta(days=1)
        if back.weekday() < 5:
            prior_news.append(back)
    for sym in symbols:
        for news_day in prior_news:
            if not sim_data.news_path(sym, news_day).exists():
                sim_data._store_news(sym, news_day, key, secret, progress)

    days = [*minute_momentum.prior_week_days(day), day]
    market = SimMarket(symbols, days, feed)
    session = ReplaySession(
        market, day, start_moment(day, start), feed, key, secret, history_feed, resolved,
    )
    session.launch_feed()
    _current = session
    return session


def score_initial_news(session: ReplaySession, provider: str, llm_key: str) -> None:
    """The news the replay opens with, scored by the LLM on a background
    thread, as the live start scores its initial load (symbols with a
    news-impact model score themselves from the News tab)."""
    from . import newsimpact_model
    from .news import score_news_impacts

    app = session.app
    targets = [
        ss for ss in app.iter_symbol_states()
        if ss.news and not newsimpact_model.uses_model(ss.symbol, app.news_impact_method)
    ]
    if not (targets and llm_key):
        return

    def run() -> None:
        for ss in targets:
            try:
                impacts = score_news_impacts(ss.symbol, list(ss.news), provider, llm_key)
            except Exception as exc:  # noqa: BLE001 -- the badges just stay unknown
                logger.warning("Replay news scoring failed for %s: %s", ss.symbol, exc)
                continue
            with ss.lock:
                ss.news_impacts = impacts

    threading.Thread(target=clock.inherit(run), name="replay-news-impacts", daemon=True).start()


def stop_current(reason: str = "") -> None:
    if _current is not None:
        _current.stop(reason)


def reset() -> None:
    """Forget the process's replay (tests)."""
    global _current, _idle
    if _current is not None:
        _current.stop()
    _current = None
    _idle = None
