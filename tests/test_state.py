import threading

from agent_stonks.config import MAX_BARS, PAPER_STARTING_CASH, VOLUME_ALERT_DEFAULT_MULTIPLIER
from agent_stonks.state import (
    AppState,
    average_daily_volume,
    completed_daily_bars,
    current_volume_ratio,
    today_daily_bar,
    today_daily_volume,
)


def _daily(date: str, volume: float) -> dict:
    return {"t": f"{date}T04:00:00Z", "v": volume}


def test_initial_values():
    s = AppState()
    assert s.symbols == []
    assert s.symbol_states == {}
    assert s.symbol == ""
    assert s.status == "Idle"
    assert s.ws is None
    assert s.ws_news is None
    assert s.agent_log == []
    assert s.agent_running is False
    assert s.agent_stop_event is None
    assert s.decision_tracker is None
    assert s.starting_budget == PAPER_STARTING_CASH


def test_set_symbols_creates_and_keeps_symbol_states():
    s = AppState()
    s.set_symbols(["aapl", "TSLA", "AAPL"])
    assert s.symbols == ["AAPL", "TSLA"]
    assert s.symbol == "AAPL"
    aapl = s.sym("aapl")
    assert aapl is not None and aapl.symbol == "AAPL"
    assert aapl.app is s
    # Re-setting keeps the existing state instance for symbols that stay.
    s.set_symbols(["AAPL", "MSFT"])
    assert s.sym("AAPL") is aapl
    assert s.sym("TSLA") is None
    assert [ss.symbol for ss in s.iter_symbol_states()] == ["AAPL", "MSFT"]


def test_symbol_state_initial_values():
    s = AppState()
    s.set_symbols(["AAPL"])
    ss = s.sym("AAPL")
    assert list(ss.bars) == []
    assert ss.trades == []
    assert ss.news == []
    assert ss.status == "Idle"
    assert ss.alerts == []
    assert ss.tactics is None


def test_bars_deque_respects_max_bars():
    s = AppState()
    s.set_symbols(["AAPL"])
    ss = s.sym("AAPL")
    for i in range(MAX_BARS + 50):
        ss.bars.append({"t": i})
    assert len(ss.bars) == MAX_BARS


def test_lock_is_reentrant_from_multiple_threads():
    s = AppState()
    s.set_symbols(["AAPL"])
    ss = s.sym("AAPL")
    errors = []

    def writer(value):
        try:
            with ss.lock:
                ss.bars.append({"v": value})
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert len(ss.bars) == 50


def test_volume_alert_defaults_on():
    s = AppState()
    s.set_symbols(["AAPL"])
    ss = s.sym("AAPL")
    assert s.volume_alert_enabled is True
    assert s.volume_alert_multiplier == VOLUME_ALERT_DEFAULT_MULTIPLIER
    assert ss.volume_alert_triggered is False
    assert ss.day_volume is None


def test_completed_daily_bars_excludes_today():
    bars = [_daily("2026-06-23", 100), _daily("2026-06-24", 200), _daily("2026-06-25", 50)]
    completed = completed_daily_bars(bars, today="2026-06-25")
    assert [b["v"] for b in completed] == [100, 200]


def test_today_daily_volume_uses_today_partial_bar():
    bars = [_daily("2026-06-24", 200), _daily("2026-06-25", 50)]
    assert today_daily_volume(bars, today="2026-06-25") == 50
    # Latest bar isn't today (e.g. weekend / pre-open) -> nothing accumulated yet.
    assert today_daily_volume([_daily("2026-06-24", 200)], today="2026-06-25") == 0.0


def test_today_daily_bar_returns_latest_bar_when_dated_today():
    bars = [
        {"t": "2026-06-24T04:00:00Z", "h": 152.30, "l": 148.10},
        {"t": "2026-06-25T04:00:00Z", "h": 146.00, "l": 144.50},
    ]
    assert today_daily_bar(bars, today="2026-06-25") == bars[-1]


def test_today_daily_bar_none_when_latest_bar_is_stale():
    # Latest daily bar is yesterday's (e.g. pre-open, weekend, or a lagging
    # feed) -- must not be mistaken for today's range.
    bars = [{"t": "2026-06-24T04:00:00Z", "h": 152.30, "l": 148.10}]
    assert today_daily_bar(bars, today="2026-06-25") is None
    assert today_daily_bar([], today="2026-06-25") is None


def test_average_daily_volume_means_completed_days():
    bars = [_daily(f"2026-05-{d:02d}", 100 * d) for d in range(1, 11)]  # 10 completed days
    bars.append(_daily("2026-06-25", 999))  # today's partial bar, excluded
    avg = average_daily_volume(bars, window=20, min_days=5, today="2026-06-25")
    expected = sum(100 * d for d in range(1, 11)) / 10
    assert avg == expected


def test_average_daily_volume_window_caps_history():
    bars = [_daily(f"2026-04-{d:02d}", d) for d in range(1, 26)]  # 25 completed days, vols 1..25
    avg = average_daily_volume(bars, window=20, min_days=5, today="2026-06-25")
    # Only the last 20 days (vols 6..25) count.
    assert avg == sum(range(6, 26)) / 20


def test_average_daily_volume_early_session_falls_back_to_yesterday():
    bars = [_daily("2026-06-23", 100), _daily("2026-06-24", 300)]  # only 2 completed days
    avg = average_daily_volume(bars, window=20, min_days=5, today="2026-06-25")
    assert avg == 300  # yesterday's single-day volume


def test_average_daily_volume_none_without_history():
    assert average_daily_volume([_daily("2026-06-25", 50)], today="2026-06-25") is None


def test_current_volume_ratio():
    bars = [_daily(f"2026-05-{d:02d}", 1000) for d in range(1, 8)]  # ADV = 1000
    ratio, baseline = current_volume_ratio(1700, bars, today="2026-06-25")
    assert baseline == 1000
    assert ratio == 1.7


def test_current_volume_ratio_none_without_volume_or_baseline():
    bars = [_daily(f"2026-05-{d:02d}", 1000) for d in range(1, 8)]
    assert current_volume_ratio(None, bars, today="2026-06-25")[0] is None
    assert current_volume_ratio(500, [], today="2026-06-25")[0] is None


# --- time-of-day-adjusted relative volume (rvol_pace) -----------------------

def _et(iso: str):
    from datetime import datetime

    return datetime.fromisoformat(iso)


def test_intraday_cumulative_volume_fraction_anchors():
    from agent_stonks.state import intraday_cumulative_volume_fraction

    # 2026-07-16 is EDT (UTC-4): 09:30 ET = 13:30Z.
    assert intraday_cumulative_volume_fraction(_et("2026-07-16T13:00:00+00:00")) is None  # pre-open
    assert intraday_cumulative_volume_fraction(_et("2026-07-16T13:30:00+00:00")) == 0.0  # the bell
    assert intraday_cumulative_volume_fraction(_et("2026-07-16T14:30:00+00:00")) == 0.24  # +60 min
    assert intraday_cumulative_volume_fraction(_et("2026-07-16T21:00:00+00:00")) == 1.0  # after close
    # Interpolates between anchors: +45 min sits halfway between 0.15 and 0.24.
    mid = intraday_cumulative_volume_fraction(_et("2026-07-16T14:15:00+00:00"))
    assert abs(mid - 0.195) < 1e-9


def test_rvol_pace_measures_pace_not_full_day():
    from agent_stonks.state import rvol_pace

    daily = [_daily(f"2026-05-{d:02d}", 1_000_000) for d in range(1, 8)]  # ADV = 1M
    # One hour in, an average day has done 24% of its volume (240k). Today has
    # done 480k -- exactly twice the normal pace.
    pace = rvol_pace(480_000, daily, now=_et("2026-07-16T14:30:00+00:00"), today="2026-06-25")
    assert pace == 2.0
    # The naive cumulative ratio would call the same tape "0.48x" -- the trap
    # rvol_pace exists to avoid.
    ratio, _ = current_volume_ratio(480_000, daily, today="2026-06-25")
    assert ratio == 0.48


def test_rvol_pace_none_outside_session_or_without_baseline():
    from agent_stonks.state import rvol_pace

    daily = [_daily(f"2026-05-{d:02d}", 1_000_000) for d in range(1, 8)]
    assert rvol_pace(480_000, daily, now=_et("2026-07-16T12:00:00+00:00"), today="2026-06-25") is None
    assert rvol_pace(480_000, [], now=_et("2026-07-16T14:30:00+00:00"), today="2026-06-25") is None
    assert rvol_pace(None, daily, now=_et("2026-07-16T14:30:00+00:00"), today="2026-06-25") is None


def test_previous_minute_close_is_watchable():
    from agent_stonks.state import ALERTABLE_FIELDS, PRICE_AXIS_ALERT_FIELDS, alert_field_value

    assert "previous_minute_close" in ALERTABLE_FIELDS
    assert "previous_minute_close" in PRICE_AXIS_ALERT_FIELDS
    assert "rvol_pace" in ALERTABLE_FIELDS
    app = AppState()
    app.set_symbols(["AAPL"])
    state = app.sym("AAPL")
    assert alert_field_value(state, "previous_minute_close") is None
    state.previous_minute_close = 101.25
    assert alert_field_value(state, "previous_minute_close") == 101.25
