"""SPY's move from its open against its usual move at that minute (`market_fall`)."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from agent_stonks import fastfurious_model as ff
from agent_stonks import market_fall as mf
from agent_stonks.market_hours import MARKET_TZ

DAY = date(2026, 11, 18)


def bars(day: date, path, base: float = 100.0, minutes=None) -> pd.DataFrame:
    """Minute bars of `day` whose closes are `base` moved by `path` percent."""
    path = list(path)
    minutes = list(range(len(path))) if minutes is None else list(minutes)
    open_at = pd.Timestamp(day.isoformat(), tz=MARKET_TZ) + pd.Timedelta(hours=9, minutes=30)
    index = pd.DatetimeIndex([open_at + pd.Timedelta(minutes=m) for m in minutes], name="ts")
    closes = [base * (1 + p / 100) for p in path]
    return pd.DataFrame(
        {"open": [base] * len(path), "high": closes, "low": closes, "close": closes,
         "volume": [1000.0] * len(path), "vwap": closes},
        index=index,
    )


def sessions_before(day: date, count: int, size: float = 0.5) -> pd.DataFrame:
    """`count` weekdays before `day`, each drifting to +/-`size`% by the close."""
    frames, stamp, sign = [], day, 1
    while len(frames) < count:
        stamp -= timedelta(days=1)
        if stamp.weekday() < 5:
            frames.append(bars(stamp, np.linspace(0, sign * size, 390)))
            sign = -sign
    return pd.concat(frames).sort_index()


@pytest.fixture(autouse=True)
def _no_memos():
    mf.clear_memos()
    yield
    mf.clear_memos()


class TestSessionMoves:
    def test_the_move_is_from_the_session_open_and_gaps_carry_the_last_close(self):
        frame = bars(DAY, [0.1] * 30 + [0.3] * 40, minutes=[*range(30), *range(35, 75)])
        moves = mf.session_moves(frame).loc[DAY.isoformat()]
        assert moves[0] == pytest.approx(0.1)
        assert moves[32] == pytest.approx(0.1)        # no bar: the last close
        assert moves[35] == pytest.approx(0.3)

    def test_nothing_is_invented_after_the_last_bar(self):
        moves = mf.session_moves(bars(DAY, [0.1] * 210)).loc[DAY.isoformat()]
        assert moves[209] == pytest.approx(0.1) and np.isnan(moves[210])

    def test_a_session_without_its_opening_bar_is_left_out(self):
        assert mf.session_moves(bars(DAY, [0.1] * 100, minutes=range(10, 110))).empty

    def test_a_scrap_of_a_session_is_left_out(self):
        assert mf.session_moves(bars(DAY, [0.1] * 20)).empty


class TestUsualMoves:
    def test_only_sessions_before_the_day_count(self):
        history = sessions_before(DAY, 20)
        today = bars(DAY, np.linspace(0, -9.0, 390))
        with_today = mf.usual_moves(pd.concat([history, today]), DAY)
        np.testing.assert_allclose(with_today, mf.usual_moves(history, DAY))

    def test_it_grows_through_the_day(self):
        usual = mf.usual_moves(sessions_before(DAY, 20), DAY)
        assert usual[30] < usual[200] < usual[380]
        assert usual[389] == pytest.approx(0.5, abs=0.02)

    def test_a_crash_does_not_make_the_next_falls_look_ordinary(self):
        calm = sessions_before(DAY, 20)
        crash_day = sorted(set(calm.index.date))[-3]
        crashed = pd.concat([
            calm[calm.index.date != crash_day],
            bars(crash_day, np.linspace(0, -6.0, 390)),
        ]).sort_index()
        before, after = mf.usual_moves(calm, DAY), mf.usual_moves(crashed, DAY)
        assert after[300] == pytest.approx(before[300], rel=0.1)

    def test_too_few_sessions_is_not_a_usual_move(self):
        with pytest.raises(mf.Unavailable, match="only 9 complete SPY sessions"):
            mf.usual_moves(sessions_before(DAY, 9), DAY)


class TestMoveAt:
    USUAL = np.full(390, 0.25)

    def test_sips_open_and_iexs_newest_close(self):
        sip = bars(DAY, [0.0] * 20, base=100.0)
        iex = bars(DAY, [-1.0] * 36, base=100.5)
        move = mf.move_at(sip, iex, 35, self.USUAL)
        assert move.open == 100.0 and move.price == pytest.approx(99.495)
        assert move.move_pct == pytest.approx(-0.505)

    def test_before_sip_has_a_bar_the_open_is_iexs(self):
        move = mf.move_at(bars(DAY, [])[:0], bars(DAY, [-0.5] * 6, base=100.5), 5, self.USUAL)
        assert move.open == 100.5 and move.move_pct == pytest.approx(-0.5)

    def test_a_stale_tape_is_not_now(self):
        with pytest.raises(mf.Unavailable, match="minutes old"):
            mf.move_at(bars(DAY, [])[:0], bars(DAY, [-0.5] * 6), 5 + mf.MAX_BAR_AGE + 1, self.USUAL)

    def test_no_bar_at_all(self):
        with pytest.raises(mf.Unavailable, match="no bar yet"):
            mf.move_at(bars(DAY, [])[:0], bars(DAY, [])[:0], 5, self.USUAL)

    def test_no_usual_move_for_the_minute(self):
        with pytest.raises(mf.Unavailable, match="no usual"):
            mf.move_at(bars(DAY, [])[:0], bars(DAY, [-0.5] * 6), 5, np.full(390, np.nan))

    @pytest.mark.parametrize("move_pct, k, over", [
        (-1.0, 3, True), (-0.75, 3, False), (-0.74, 3, False), (0.9, 3, False), (-0.51, 2, True),
    ])
    def test_a_fall_is_over_the_limit_only_past_it(self, move_pct, k, over):
        assert mf.SpyMove(35, 100.0, 99.0, move_pct, 0.25).is_fall_over(k) is over


class TestRead:
    """A past session is read whole and cut; today is fetched at the bar."""

    PAST = date(2026, 11, 17)

    def _past(self, monkeypatch, path):
        history = sessions_before(self.PAST, 20)
        day = bars(self.PAST, path)
        calls = []

        def raw_history(symbol, first, before, key=None, secret=None):
            calls.append((symbol, first, before))
            whole = pd.concat([history, day]).sort_index()
            stamps = whole.index.tz_convert(MARKET_TZ).date
            inside = whole[(stamps >= first) & (stamps < before)]
            return {"sip": inside, "iex": inside}

        monkeypatch.setattr(ff, "raw_history", raw_history)
        monkeypatch.setattr(ff, "_today_et", lambda: DAY)
        return calls

    def test_a_replay_cannot_see_past_the_bar(self, monkeypatch):
        self._past(monkeypatch, [-0.4] * 60 + [-5.0] * 330)
        assert mf.read(self.PAST, 59).move_pct == pytest.approx(-0.4)
        assert mf.read(self.PAST, 60).move_pct == pytest.approx(-5.0)

    def test_a_replay_reads_each_session_once(self, monkeypatch):
        calls = self._past(monkeypatch, [-0.4] * 390)
        for minute in range(30, 60):
            mf.read(self.PAST, minute)
        assert len(calls) == 2                      # the history, the session

    def _today(self, monkeypatch, iex, settled=True):
        history = sessions_before(DAY, 20)
        asked = []

        def live_tapes(symbol, day, through, key=None, secret=None, retries=None):
            asked.append((symbol, day, through, retries))
            return iex[:0], iex, settled

        monkeypatch.setattr(ff, "raw_history", lambda *a, **k: {"sip": history, "iex": history})
        monkeypatch.setattr(ff, "live_tapes", live_tapes)
        monkeypatch.setattr(ff, "_today_et", lambda: DAY)
        return asked

    def test_today_is_fetched_at_the_bar_and_never_waited_for(self, monkeypatch):
        asked = self._today(monkeypatch, bars(DAY, [-0.5] * 11))
        assert mf.read(DAY, 10).move_pct == pytest.approx(-0.5)
        assert mf.read(DAY, 10).move_pct == pytest.approx(-0.5)
        assert asked == [("SPY", DAY, 10, 0)]        # remembered: one request

    def test_a_bar_not_published_yet_is_read_again(self, monkeypatch):
        asked = self._today(monkeypatch, bars(DAY, [-0.5] * 10), settled=False)
        mf.read(DAY, 10)
        mf.read(DAY, 10)
        assert len(asked) == 2

    def test_a_failure_is_not_known_and_is_not_retried_at_once(self, monkeypatch):
        asked = []

        def refused(*a, **k):
            asked.append(a)
            raise RuntimeError("HTTP 500")

        monkeypatch.setattr(ff, "raw_history", refused)
        monkeypatch.setattr(ff, "_today_et", lambda: DAY)
        for _ in range(2):
            with pytest.raises(mf.Unavailable, match="HTTP 500"):
                mf.read(DAY, 10)
        assert len(asked) == 1


def test_minute_of_counts_from_the_open_in_new_york():
    assert mf.minute_of(pd.Timestamp("2026-11-18 14:30", tz="UTC")) == 0
    assert mf.minute_of(pd.Timestamp("2026-11-18 10:15", tz=MARKET_TZ)) == 45
    assert mf.minute_of(pd.Timestamp("2026-11-18 09:29", tz=MARKET_TZ)) == -1
