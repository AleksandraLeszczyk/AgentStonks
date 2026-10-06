"""The live app's replay of a past session (the sidebar's Dummy data)."""
import json
import threading
from datetime import date, datetime, time, timedelta, timezone

import pytest

from agent_stonks import clock, historical, replay, rule_agent, scoring, session_store, ui
from agent_stonks.apple_trader import AppleTraderConfig
from agent_stonks.decisions import DecisionTracker
from agent_stonks.market_hours import MARKET_TZ
from agent_stonks.premarket import PremarketBriefing
from agent_stonks.state import AppState
from simlab import data as sim_data
from simlab.market import SimMarket

DAY = date(2026, 10, 5)  # a Monday
PRIOR = date(2026, 10, 2)  # the Friday before
FEED = "sip"


def et(day: date, hh: int, mm: int, ss: int = 0) -> datetime:
    return datetime.combine(day, time(hh, mm, ss), tzinfo=MARKET_TZ).astimezone(timezone.utc)


def stamp(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def minute_bars(day: date, start: tuple, count: int, base: float = 100.0) -> list[dict]:
    """`count` one-minute bars from `start` (hh, mm) ET, each closing 0.10 up
    with a dip to the open minus 0.05 and a peak 0.05 over the close."""
    out = []
    for k in range(count):
        o = base + 0.10 * k
        c = o + 0.10
        out.append({
            "t": stamp(et(day, *start) + timedelta(minutes=k)),
            "o": o, "h": c + 0.05, "l": o - 0.05, "c": c, "v": 1000.0 + k,
        })
    return out


@pytest.fixture
def store(monkeypatch, tmp_path):
    """A SimLab store with the replayed day (09:20-09:40 ET), the Friday
    before, a daily history, and news either side of 09:29."""
    monkeypatch.setattr(sim_data, "STORE_DIR", tmp_path / "store")
    sim_data._write_gz(sim_data.bars_path("AAPL", DAY, FEED), minute_bars(DAY, (9, 20), 21))
    sim_data._write_gz(sim_data.bars_path("AAPL", PRIOR, FEED), minute_bars(PRIOR, (9, 30), 30, 90.0))
    daily = [
        {"t": f"{d.isoformat()}T04:00:00Z", "o": 99.0, "h": 101.0, "l": 98.0, "c": c, "v": 5e6}
        for d, c in [(date(2026, 10, 1), 98.5), (PRIOR, 99.5), (DAY, 102.0)]
    ]
    sim_data._write_gz(sim_data.daily_path("AAPL", FEED), {"bars": daily})
    sim_data._write_gz(sim_data.news_path("AAPL", DAY), [
        {"id": 1, "headline": "Early story", "created_at": stamp(et(DAY, 9, 0))},
        {"id": 2, "headline": "Story at the bell", "created_at": stamp(et(DAY, 9, 29, 40))},
    ])
    return tmp_path


@pytest.fixture(autouse=True)
def _clean():
    yield
    replay.reset()
    replay._uninstall_dispatchers()
    clock.unbind()


def make_session(start=(9, 29)) -> replay.ReplaySession:
    market = SimMarket(["AAPL"], [PRIOR, DAY], FEED)
    return replay.ReplaySession(
        market, DAY, et(DAY, *start), FEED, "key", "secret", "auto", "sip_delayed",
    )


# --- the clock ----------------------------------------------------------------


class Fixed:
    def __init__(self, moment):
        self.moment = moment

    def now(self):
        return self.moment


def test_a_bound_thread_reads_its_scope_and_others_the_wall_clock():
    then = et(DAY, 9, 29)
    clock.bind(Fixed(then))
    seen = {}
    other = threading.Thread(target=lambda: seen.setdefault("other", clock.now()))
    other.start()
    other.join()
    assert clock.now() == then
    assert abs((seen["other"] - datetime.now(timezone.utc)).total_seconds()) < 5
    clock.unbind()
    assert clock.now() != then


def test_inherit_carries_the_scope_into_a_new_thread_and_no_further():
    then = et(DAY, 9, 29)
    clock.bind(Fixed(then))
    seen = {}

    def work():
        seen["inside"] = clock.now()

    thread = threading.Thread(target=clock.inherit(work))
    thread.start()
    thread.join()
    assert seen["inside"] == then
    clock.unbind()
    # Captured when wrapped: an unbound caller's thread stays on the real time.
    plain = threading.Thread(target=clock.inherit(lambda: seen.setdefault("plain", clock.now())))
    plain.start()
    plain.join()
    assert seen["plain"] != then


def test_a_scope_wins_over_the_simlab_pin():
    clock.set_simulated(et(DAY, 12, 0))
    try:
        clock.bind(Fixed(et(DAY, 9, 29)))
        assert clock.now() == et(DAY, 9, 29)
        clock.unbind()
        assert clock.now() == et(DAY, 12, 0)
    finally:
        clock.clear()


def test_a_rule_agent_thread_inherits_the_launching_scope():
    then = et(DAY, 9, 31)
    seen = threading.Event()
    got = {}

    def target(state, tracker, stop_event):
        got["now"] = clock.now()
        seen.set()

    state = AppState()
    clock.bind(Fixed(then))
    rule_agent.launch(
        state, DecisionTracker(starting_cash=1000.0), "apple_trader", "AAPL",
        target=target, args=(state, None), stop_agent=lambda s: None,
    )
    assert seen.wait(5)
    assert got["now"] == then


def test_the_replay_clock_stands_still_until_run(monkeypatch):
    ticks = iter([100.0, 100.0, 130.0, 130.0, 500.0])
    monkeypatch.setattr(replay.time, "monotonic", lambda: next(ticks))
    start = et(DAY, 9, 29)
    rc = replay.ReplayClock(start)
    assert rc.now() == start and not rc.running
    rc.run()  # at 100
    assert rc.now() == start  # 100
    assert rc.now() == start + timedelta(seconds=30)  # 130
    rc.pause()  # anchors at 130
    assert rc.now() == start + timedelta(seconds=30)  # monotonic no longer read


# --- the forming candle -----------------------------------------------------------


def test_a_minute_that_closed_up_dips_first_and_ends_on_its_close():
    bar = {"o": 100.0, "h": 100.5, "l": 99.5, "c": 100.2}
    assert replay.forming(bar, 0.0) == (100.0, 100.0, 100.0)
    price, high, low = replay.forming(bar, 1 / 3)
    assert (price, low) == (99.5, 99.5) and high == 100.0
    price, high, low = replay.forming(bar, 2 / 3)
    assert (price, high, low) == (100.5, 100.5, 99.5)
    assert replay.forming(bar, 1.0) == (100.2, 100.5, 99.5)


def test_a_minute_that_closed_down_peaks_first():
    bar = {"o": 100.0, "h": 100.5, "l": 99.5, "c": 99.8}
    price, high, low = replay.forming(bar, 1 / 3)
    assert price == 100.5 and low == 100.0


# --- seeding and playing ------------------------------------------------------------


def test_the_page_opens_as_it_stood_at_the_start_time(store):
    session = make_session()
    ss = session.app.sym("AAPL")
    assert ss.bars[-1]["t"] == stamp(et(DAY, 9, 28))
    assert all(bar["src"] == FEED for bar in ss.bars)
    assert ss.last_price == pytest.approx(ss.bars[-1]["c"])
    assert ss.prev_close == 99.5
    assert [a["headline"] for a in ss.news] == ["Early story"]
    # Today's partial daily bar, from the minutes completed by 09:29 only.
    assert ss.daily_bars[-1]["t"].startswith(DAY.isoformat())
    assert ss.daily_bars[-1]["h"] == pytest.approx(ss.bars[-1]["h"])
    assert session.app.bar_tape_override == FEED
    assert "ready" in session.app.status


def test_the_minute_in_progress_forms_and_then_closes_exactly(store):
    session = make_session()
    ss = session.app.sym("AAPL")
    stored_0929 = sim_data.load_day_bars("AAPL", DAY, FEED)[9]
    session.tick(et(DAY, 9, 29, 30))
    forming_bar = ss.bars[-1]
    assert forming_bar["t"] == stored_0929["t"]
    assert 0 < forming_bar["v"] < stored_0929["v"]
    assert stored_0929["l"] <= ss.last_price <= stored_0929["h"]
    before_close = ss.previous_minute_close

    session.tick(et(DAY, 9, 30, 1))
    closed = [b for b in ss.bars if b["t"] == stored_0929["t"]]
    assert closed == [{**stored_0929, "src": FEED}]
    assert ss.previous_minute_close == stored_0929["c"] != before_close
    # The 09:30 minute is forming now, and the volume adds up to the tape's.
    assert ss.bars[-1]["t"] == stamp(et(DAY, 9, 30))
    tape = sum(b["v"] for b in sim_data.load_day_bars("AAPL", DAY, FEED)[:10])
    assert ss.day_volume == pytest.approx(tape + ss.bars[-1]["v"])


def test_every_trade_time_parses_as_one_format(store):
    """The chart reads all trades' times with one `pd.to_datetime` call, which
    infers its format from the first: a seeded print and a tick must match."""
    import pandas as pd

    session = make_session()
    session.tick(et(DAY, 9, 29, 30))
    session.tick(et(DAY, 9, 30, 1))
    trades = session.app.sym("AAPL").trades
    assert len(trades) > 9
    parsed = pd.to_datetime(pd.Series([t["t"] for t in trades]))
    assert parsed.is_monotonic_increasing


def test_news_arrives_when_it_was_published_and_wakes_the_agent(store):
    session = make_session()
    ss = session.app.sym("AAPL")
    session.tick(et(DAY, 9, 29, 30))
    assert [a["headline"] for a in ss.news] == ["Early story"]
    assert not session.app.agent_wake_event.is_set()
    session.tick(et(DAY, 9, 29, 45))
    assert ss.news[0]["headline"] == "Story at the bell"
    assert session.app.agent_wake_event.is_set()
    assert "Story at the bell" in session.app.agent_wake_reason


def test_the_end_of_the_stored_tape_is_said(store):
    session = make_session()
    session.clock.run()
    session.tick(et(DAY, 9, 45))
    assert session.finished
    assert "over" in session.app.status
    assert session.app.sym("AAPL").bars[-1]["t"] == stamp(et(DAY, 9, 40))


def test_stop_ends_the_tape_and_the_agent(store):
    session = make_session()
    session.run()
    session.app.agent_running = True
    session.stop("stopped")
    assert session.stop_event.is_set() and not session.clock.running
    assert not session.app.agent_running
    assert "stopped" in session.app.status


# --- the patched fetches ------------------------------------------------------------


def test_fetches_read_the_dataset_on_a_bound_thread_only(store, monkeypatch):
    monkeypatch.setattr(historical, "fetch_daily_ohlc_bars", lambda *a, **k: ["live"])
    session = make_session()
    # The dispatcher stands in front of the attribute now.
    assert historical.fetch_daily_ohlc_bars("AAPL") == ["live"]
    clock.bind(session)
    rows = historical.fetch_daily_ohlc_bars("AAPL")
    assert [r["t"] for r in rows] == ["2026-10-01", "2026-10-02"]  # never the day itself
    seen = {}
    thread = threading.Thread(target=lambda: seen.setdefault("x", historical.fetch_daily_ohlc_bars("AAPL")))
    thread.start()
    thread.join()
    assert seen["x"] == ["live"]


def test_the_session_open_is_hidden_until_the_bell(store):
    session = make_session()
    clock.bind(session)
    assert historical.fetch_session_open("AAPL") is None
    session.tick(et(DAY, 9, 30, 30))
    session.clock = replay.ReplayClock(et(DAY, 9, 30, 30))
    assert historical.fetch_session_open("AAPL") == 99.0


# --- the venue and the live day's files -------------------------------------------------


def test_orders_fill_at_the_replayed_price(store):
    session = make_session()
    session.tick(et(DAY, 9, 29, 30))
    tracker = DecisionTracker(starting_cash=10_000.0, broker=replay.ReplayBroker(session))
    decision = tracker.record_trade("AAPL", "buy", 10, "test", "key", "secret")
    assert decision.status == "filled"
    assert decision.price == session.app.sym("AAPL").last_price


def test_a_replay_never_writes_the_live_day(store, tmp_path):
    session = make_session()
    app = session.app
    app.decision_tracker = DecisionTracker(starting_cash=1000.0)
    app.agent_log = [{"type": "status", "text": "x"}]
    session_store.claim(app)
    assert session_store.save(app, force=True) is None
    assert not (tmp_path / "sessions").exists()
    session_store.start_autosave(app)
    assert "_session_autosave" not in app.__dict__
    scoring.begin_session(app, "apple_trader", ["AAPL"])
    assert app.scorecard is None


# --- the briefing ----------------------------------------------------------------------


def briefing() -> PremarketBriefing:
    return PremarketBriefing.model_validate({
        "overall_bias": "bullish", "confidence": "medium", "summary": "Up into the open.",
        "catalysts": [], "technical_levels": [], "risk_factors": [],
        "key_levels_to_watch": [], "macro_context": "calm",
    })


def test_a_cached_briefing_is_shown_without_asking_the_model(store, monkeypatch, tmp_path):
    monkeypatch.setattr(replay, "BRIEFING_DIR", tmp_path / "briefings")
    path = replay.briefing_path("AAPL", DAY, "openai", "m")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"briefing": briefing().model_dump(mode="json")}))
    session = make_session()
    assert session.launch_briefing("openai", "m", api_key="")
    assert session.app.premarket_briefings["AAPL"].summary == "Up into the open."
    assert session.app.premarket_phase == "premarket"
    assert session.app.premarket_generated_at == et(DAY, 9, 25)


def test_the_days_off_check_reads_the_replays_own_briefing(store, monkeypatch):
    from agent_stonks import event_days

    monkeypatch.setattr(event_days, "recorded_verdict", lambda *a, **k: {"shock": "geo"})
    session = make_session()
    clock.bind(session)
    # No briefing yet: what the live app recorded that morning (SimLab's answer).
    assert event_days.briefing_verdict("AAPL", DAY) == {"shock": "geo"}
    shocked = briefing().model_copy(update={"shock": "market", "shock_reason": "a crash"})
    session.app.premarket_briefings = {"AAPL": shocked}
    verdict = event_days.briefing_verdict("AAPL", DAY)
    assert (verdict["shock"], verdict["reason"]) == ("market", "a crash")
    assert verdict["made_at"].startswith("2026-10-05T09:25")


def test_no_key_and_no_cache_says_so(store, monkeypatch, tmp_path):
    monkeypatch.setattr(replay, "BRIEFING_DIR", tmp_path / "briefings")
    session = make_session()
    assert not session.launch_briefing("openai", "m", api_key="")
    assert "No API key" in session.app.premarket_status


def test_a_written_briefing_is_cached_and_records_no_verdict(store, monkeypatch, tmp_path):
    monkeypatch.setattr(replay, "BRIEFING_DIR", tmp_path / "briefings")
    calls = []

    def fake_generate(sym, provider, api_key, as_of, **kw):
        calls.append(as_of)
        return briefing()

    from agent_stonks import event_days, premarket

    monkeypatch.setattr(premarket, "generate_premarket_from_data", fake_generate)
    monkeypatch.setattr(premarket, "_earnings_block", lambda *a, **k: "")
    from simlab import session_context

    monkeypatch.setattr(
        session_context, "_fetch_news_before",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("401")),
    )
    monkeypatch.setattr(event_days, "record_verdict", lambda *a, **k: pytest.fail("verdict written"))
    session = make_session()
    session._brief("openai", "m", "llm-key", force=False)
    assert calls == [et(DAY, 9, 25)]
    assert replay.load_briefing("AAPL", DAY, "openai", "m").summary == "Up into the open."
    assert "ready" in session.app.premarket_status


# --- preparing ------------------------------------------------------------------------


def test_a_day_that_is_not_over_is_refused(monkeypatch):
    monkeypatch.setattr(replay, "last_replayable_day", lambda now=None: DAY)
    with pytest.raises(ValueError, match="not over yet"):
        replay.prepare(["AAPL"], DAY + timedelta(days=1), time(9, 29), "k", "s", "auto")
    with pytest.raises(ValueError, match="weekend"):
        replay.prepare(["AAPL"], date(2026, 10, 4), time(9, 29), "k", "s", "auto")


def test_prepare_builds_the_replay_from_the_store(store, monkeypatch):
    from agent_stonks import bar_history

    monkeypatch.setattr(replay, "last_replayable_day", lambda now=None: DAY)
    monkeypatch.setattr(bar_history, "resolve_history_feed", lambda *a, **k: "sip_delayed")
    monkeypatch.setattr(sim_data, "download_days", lambda *a, **k: [DAY.isoformat()])
    monkeypatch.setattr(sim_data, "_store_news", lambda *a, **k: None)
    monkeypatch.setattr(replay.ReplaySession, "launch_feed", lambda self: None)
    session = replay.prepare(["aapl"], DAY, time(9, 29), "k", "s", "auto")
    assert replay.current() is session and replay.app_state() is session.app
    assert session.feed == "sip" and session.app.history_feed_resolved == "sip_delayed"
    assert session.now() == et(DAY, 9, 29)


def test_a_holiday_is_refused(store, monkeypatch):
    from agent_stonks import bar_history

    monkeypatch.setattr(replay, "last_replayable_day", lambda now=None: DAY)
    monkeypatch.setattr(bar_history, "resolve_history_feed", lambda *a, **k: "sip")
    monkeypatch.setattr(sim_data, "download_days", lambda *a, **k: [])
    with pytest.raises(ValueError, match="no session"):
        replay.prepare(["AAPL"], DAY, time(9, 29), "k", "s", "auto")


def test_the_previous_session_skips_the_weekend():
    assert replay.previous_session(date(2026, 10, 5)) == date(2026, 10, 2)
    assert replay.previous_session(date(2026, 10, 6)) == date(2026, 10, 5)


# --- the Agent tab's ▶ Start Agent ------------------------------------------------------


def test_start_agent_plays_the_replay_on_its_own_broker(store, monkeypatch):
    session = make_session()
    replay._current = session
    monkeypatch.setattr(ui.st, "session_state", {ui.DUMMY_DATA_KEY: True})
    for name in ("info", "success", "warning", "error"):
        monkeypatch.setattr(ui.st, name, lambda *a, **k: None)
    monkeypatch.setattr(session_store, "archive", lambda *a, **k: pytest.fail("archived"))
    launched = {}

    def fake_launch(state, tracker, config, cycle_sec):
        launched.update(state=state, broker=tracker.broker, scope=clock.scope(), now=clock.now())

    monkeypatch.setattr(ui, "launch_apple_trader", fake_launch)
    clock.bind(session)
    ready = ui._start_agent(
        session.app, ["AAPL"], personality="apple_trader", provider="openai", model="",
        apple_config=AppleTraderConfig(), trading_mode_choice="alpaca_live",
        starting_budget=10_000.0, continue_today=True,
    )
    assert ready
    assert launched["state"] is session.app
    assert isinstance(launched["broker"], replay.ReplayBroker)
    assert launched["scope"] is session
    assert session.clock.running
    assert session.app.trading_mode == "local"
    assert session.app.agent_start_time >= et(DAY, 9, 29)
