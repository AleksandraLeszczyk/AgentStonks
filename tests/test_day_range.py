"""The day-range rules' shared arithmetic (`agent_stonks.day_range`): the IEX
volume caveat, how a 9:35 forecast range moves once the tape trades through
it, the minutes left to the bell and the session's date.

Arithmetic only -- what a trader then does with the moved numbers is pinned in
`tests/test_apple_trader.py`. These came over from TimeToChange3's mirror
tests when that model was removed (2026-10-25); the functions did too.
"""

from datetime import datetime, timezone

import pandas as pd
import pytest

from agent_stonks import clock
from agent_stonks import day_range as D


class TestVolumeScaleWarning:
    def test_iex_is_flagged(self):
        warning = D.volume_scale_warning("iex")
        assert warning is not None and "consolidated" in warning

    @pytest.mark.parametrize("feed", ["sip", "yfinance", None, ""])
    def test_consolidated_tapes_are_not(self, feed):
        assert D.volume_scale_warning(feed) is None


class TestIntradayUpdate:
    """The two policies that move the forecast the session has traded through.

    Arithmetic only -- what a trader then does with the moved numbers is pinned
    in `tests/test_apple_trader.py`.
    """

    # A $10 average daily range around a 100-110 forecast, so a breach and the
    # Brownian reach both land on numbers that can be read by eye.
    FORECAST = {"pred_high": 110.0, "pred_low": 100.0, "adr14_abs": 10.0}
    NOON = 240.0  # minutes left at 12:00

    def _update(self, policy, high, low, minutes_left=NOON, forecast=None):
        return D.updated_range(
            forecast or self.FORECAST,
            session_high=high, session_low=low,
            minutes_left=minutes_left, policy=policy,
        )

    @pytest.mark.parametrize("policy", [D.BREACH_OFF, "nonsense", None])
    def test_off_and_anything_unrecognised_hold_the_forecast(self, policy):
        """An unknown policy reads as off rather than as today's default: it
        arrives from a stored record or a form, and a typo that silently traded
        a different strategy would be worse than one that traded the notebook's."""
        assert self._update(policy, 120.0, 90.0) == (110.0, 100.0)

    def test_an_unbreached_forecast_is_untouched(self):
        """A day trading inside the range is a day the forecast still describes,
        under every policy."""
        for policy in (D.BREACH_SHIFT, D.BREACH_EXTREME, D.BREACH_BROWNIAN):
            assert self._update(policy, 109.99, 100.01) == (110.0, 100.0)

    def test_shift_moves_both_sides_by_the_breach(self):
        """"Move to the extreme so far": the breached side to the extreme, the
        other by the same amount, so the range keeps its $10 width."""
        assert self._update(D.BREACH_SHIFT, 112.5, 103.0) == (112.5, 102.5)
        assert self._update(D.BREACH_SHIFT, 107.0, 97.25) == (107.25, 97.25)

    def test_shift_never_moves_the_other_side_past_what_the_session_printed(self):
        # Up $2.50, but the session already traded $101.00: a predicted low of
        # $102.50 would be a low the day has already disproved.
        assert self._update(D.BREACH_SHIFT, 112.5, 101.0) == (112.5, 101.0)
        # Down $2.75, but the session already traded $109.00.
        assert self._update(D.BREACH_SHIFT, 109.0, 97.25) == (109.0, 97.25)

    def test_shift_through_both_sides_on_one_bar_holds_the_session(self):
        # Up $1, down $2: the range moves down $1 and then holds the session.
        assert self._update(D.BREACH_SHIFT, 111.0, 98.0) == (111.0, 98.0)

    def test_shift_ratchets_from_where_it_last_moved(self):
        first = self._update(D.BREACH_SHIFT, 112.5, 103.0)
        assert first == (112.5, 102.5)
        moved = {"pred_high": first[0], "pred_low": first[1], "adr14_abs": 10.0}
        # Nothing new printed: nothing moves.
        assert self._update(D.BREACH_SHIFT, 112.5, 103.0, forecast=moved) == first
        # A new high moves both again, from the moved range.
        assert self._update(D.BREACH_SHIFT, 113.5, 103.0, forecast=moved) == (113.5, 103.0)

    def test_extreme_moves_the_breached_side_to_the_extreme(self):
        assert self._update(D.BREACH_EXTREME, 112.5, 101.0) == (112.5, 100.0)
        assert self._update(D.BREACH_EXTREME, 109.0, 97.25) == (110.0, 97.25)

    def test_only_the_breached_side_moves(self):
        """A high that has been exceeded says nothing about the low."""
        high, low = self._update(D.BREACH_EXTREME, 112.5, 100.5)
        assert (high, low) == (112.5, 100.0)

    def test_brownian_extends_past_the_extreme_by_the_expected_excursion(self):
        # 240 of 390 minutes left -> ADR x sqrt(240/390) / 2 = $3.92.
        reach = 10.0 * (240 / 390) ** 0.5 / 2
        high, low = self._update(D.BREACH_BROWNIAN, 112.5, 99.0)
        assert high == pytest.approx(112.5 + reach)
        assert low == pytest.approx(99.0 - reach)

    def test_the_brownian_reach_is_half_an_adr_with_a_whole_session_left(self):
        """The one number worth being able to check by hand: E[range] over a
        session is two expected one-sided excursions, so each is half an ADR."""
        assert D.brownian_reach(10.0, D.SESSION_MINUTES) == pytest.approx(5.0)

    def test_the_brownian_reach_shrinks_to_nothing_at_the_bell(self):
        reaches = [D.brownian_reach(10.0, m) for m in (390, 240, 60, 1, 0)]
        assert reaches == sorted(reaches, reverse=True)
        assert reaches[-1] == 0.0

    @pytest.mark.parametrize("adr", [0.0, None, -1.0])
    def test_a_missing_adr_leaves_the_brownian_policy_at_the_extreme(self, adr):
        """No volatility to imply an extension from, so it degrades to the
        weaker claim rather than to a NaN level."""
        forecast = {**self.FORECAST, "adr14_abs": adr}
        assert self._update(D.BREACH_BROWNIAN, 112.5, 105.0, forecast=forecast) == (
            112.5, 100.0
        )

    @pytest.mark.parametrize("policy", [D.BREACH_EXTREME, D.BREACH_BROWNIAN])
    def test_fed_its_own_answer_each_bar_the_range_only_widens(self, policy):
        """The ratchet. A high that could come back down would drag the sell
        level towards the price and turn one move into a stream of exits."""
        forecast = dict(self.FORECAST)
        highs, lows = [], []
        # A session that runs up, falls back inside, then runs again -- with the
        # clock (and so the Brownian reach) shrinking all the way.
        for extreme, minutes in ((111.0, 300), (110.5, 240), (114.0, 120), (113.0, 30)):
            high, low = self._update(policy, extreme, 101.0, minutes_left=minutes,
                                     forecast=forecast)
            forecast = {**forecast, "pred_high": high, "pred_low": low}
            highs.append(high)
            lows.append(low)
        assert highs == sorted(highs)
        assert lows == sorted(lows, reverse=True)

    def test_a_brownian_high_is_not_re_extended_until_it_is_breached_again(self):
        """It already anticipates the excursion, so the tape catching up to the
        extreme it was built on is not new information."""
        high, _ = self._update(D.BREACH_BROWNIAN, 112.0, 101.0)
        forecast = {**self.FORECAST, "pred_high": high}
        assert high > 112.0
        again, _ = self._update(D.BREACH_BROWNIAN, high - 0.01, 101.0, forecast=forecast)
        assert again == high


class TestUpdatedRangeKeepsTheWidth:
    """`width`: a breach moves the day, it does not make it wider.

    The breached side goes where the policy puts it and the other side follows
    at `width` -- the level unit's dollars at 9:35 -- so the only thing that
    can make the range wider than that is a price the session already printed.
    """

    FORECAST = {"pred_high": 110.0, "pred_low": 100.0, "adr14_abs": 10.0}
    NOON = 240.0

    def _update(self, policy, high, low, width, forecast=None, minutes_left=NOON):
        return D.updated_range(
            forecast or self.FORECAST,
            session_high=high, session_low=low,
            minutes_left=minutes_left, policy=policy, width=width,
        )

    def test_shift_puts_the_other_side_the_width_away(self):
        # The forecast's own $10: the same answer as without `width`.
        assert self._update(D.BREACH_SHIFT, 112.5, 103.0, 10.0) == (112.5, 102.5)
        # An $8 ADR under the "adr" unit: the low follows $8 under the high.
        assert self._update(D.BREACH_SHIFT, 112.5, 106.0, 8.0) == (112.5, 104.5)
        assert self._update(D.BREACH_SHIFT, 104.0, 97.0, 8.0) == (105.0, 97.0)

    def test_brownian_moves_both_sides_and_keeps_the_width(self):
        reach = D.brownian_reach(10.0, self.NOON)
        high, low = self._update(D.BREACH_BROWNIAN, 112.5, 107.0, 10.0)
        assert high == pytest.approx(112.5 + reach)
        assert high - low == pytest.approx(10.0)
        low, high = self._update(D.BREACH_BROWNIAN, 103.0, 99.0, 10.0)[::-1]
        assert low == pytest.approx(99.0 - reach)
        assert high - low == pytest.approx(10.0)

    @pytest.mark.parametrize("policy", [D.BREACH_SHIFT, D.BREACH_BROWNIAN])
    def test_it_widens_only_to_hold_what_the_session_printed(self, policy):
        # Up through $110 while $101 has already traded: a low $10 under the
        # new high would exclude it, so the low stops at $101 -- and no lower.
        high, low = self._update(policy, 112.5, 101.0, 10.0)
        assert low == 101.0
        assert high >= 112.5

    def test_a_range_that_was_widened_comes_back_to_the_width(self):
        """The width is the unit's, not whatever the range last had: once the
        session no longer needs the extra, the next breach drops it."""
        widened = {**self.FORECAST, "pred_high": 115.0, "pred_low": 100.0}
        assert self._update(D.BREACH_SHIFT, 116.0, 107.0, 10.0, forecast=widened) == (
            116.0, 106.0
        )
        # Without `width`, "shift" keeps the $15 it had.
        assert D.updated_range(
            widened, session_high=116.0, session_low=107.0, minutes_left=self.NOON,
            policy=D.BREACH_SHIFT,
        ) == (116.0, 101.0)

    def test_an_unbreached_range_is_untouched(self):
        for policy in (D.BREACH_SHIFT, D.BREACH_BROWNIAN):
            assert self._update(policy, 109.0, 101.0, 8.0) == (110.0, 100.0)

    @pytest.mark.parametrize("policy", [D.BREACH_OFF, D.BREACH_EXTREME])
    def test_the_one_sided_policies_ignore_it(self, policy):
        assert self._update(policy, 112.5, 106.0, 8.0) == D.updated_range(
            self.FORECAST, session_high=112.5, session_low=106.0,
            minutes_left=self.NOON, policy=policy,
        )


class TestConditionalExtension:
    """"conditional": "shift", plus the Brownian reach on a side whose breach
    the caller has seen hold (`extend`)."""

    FORECAST = {"pred_high": 110.0, "pred_low": 100.0, "adr14_abs": 10.0}
    NOON = 240.0

    def _update(self, policy, high, low, width=10.0, extend=(False, False), forecast=None):
        return D.updated_range(
            forecast or self.FORECAST,
            session_high=high, session_low=low,
            minutes_left=self.NOON, policy=policy, width=width, extend=extend,
        )

    @pytest.mark.parametrize("width", [10.0, 8.0, None])
    def test_until_a_breach_holds_it_is_shift(self, width):
        for high, low in ((112.5, 103.0), (104.0, 97.0), (109.0, 101.0)):
            assert self._update(D.BREACH_CONDITIONAL, high, low, width) == self._update(
                D.BREACH_SHIFT, high, low, width
            )

    def test_a_held_breach_is_extended_like_brownian(self):
        held = self._update(D.BREACH_CONDITIONAL, 112.5, 107.0, extend=(True, False))
        assert held == pytest.approx(self._update(D.BREACH_BROWNIAN, 112.5, 107.0))
        held = self._update(D.BREACH_CONDITIONAL, 103.0, 99.0, extend=(False, True))
        assert held == pytest.approx(self._update(D.BREACH_BROWNIAN, 103.0, 99.0))

    def test_a_breach_confirmed_on_a_bar_with_no_new_extreme_is_extended_from_the_extreme(self):
        """The range was moved to $112.50 when the breach printed it; the bar
        that confirms the breach traded no higher, and is extended all the same."""
        moved = {**self.FORECAST, "pred_high": 112.5, "pred_low": 102.5}
        reach = D.brownian_reach(10.0, self.NOON)
        high, low = self._update(
            D.BREACH_CONDITIONAL, 112.5, 107.0, extend=(True, False), forecast=moved
        )
        assert high == pytest.approx(112.5 + reach)
        assert high - low == pytest.approx(10.0)
        # Without a width only the held side moves, as under "brownian".
        high, low = self._update(
            D.BREACH_CONDITIONAL, 112.5, 107.0, None, extend=(True, False), forecast=moved
        )
        assert (high, low) == pytest.approx((112.5 + reach, 102.5))

    def test_it_never_pulls_a_side_back_in(self):
        wide = {**self.FORECAST, "pred_high": 130.0, "pred_low": 120.0}
        assert self._update(
            D.BREACH_CONDITIONAL, 112.5, 107.0, extend=(True, False), forecast=wide
        )[0] == 130.0

    @pytest.mark.parametrize("policy", [D.BREACH_SHIFT, D.BREACH_BROWNIAN, D.BREACH_OFF])
    def test_the_other_policies_ignore_extend(self, policy):
        assert self._update(policy, 112.5, 107.0, extend=(True, True)) == self._update(
            policy, 112.5, 107.0
        )


class TestMinutesLeft:
    @staticmethod
    def _et(stamp):
        return pd.Timestamp(stamp, tz=D.market_hours.MARKET_TZ)

    def test_counts_down_to_the_four_oclock_close(self):
        assert D.minutes_left_at(self._et("2026-08-07 12:00")) == 240.0
        assert D.minutes_left_at(self._et("2026-08-07 09:35")) == 385.0

    def test_a_naive_timestamp_is_read_as_exchange_time(self):
        assert D.minutes_left_at(pd.Timestamp("2026-08-07 12:00")) == 240.0

    def test_a_utc_timestamp_is_converted_rather_than_taken_at_face_value(self):
        """The trader's bars are tz-aware; a UTC index read as ET would put the
        whole session after the close and zero every Brownian reach."""
        assert D.minutes_left_at(pd.Timestamp("2026-08-07 16:00", tz="UTC")) == 240.0

    def test_never_goes_negative_after_the_close(self):
        assert D.minutes_left_at(pd.Timestamp("2026-08-07 17:30")) == 0.0


class TestMarketDate:
    def test_the_session_date_follows_the_simulated_clock(self):
        # 00:30 UTC is still the previous day in New York, which is the
        # boundary a UTC date would get wrong.
        clock.set_simulated(datetime(2026, 8, 8, 0, 30, tzinfo=timezone.utc))
        try:
            assert D.market_date() == pd.Timestamp("2026-08-07")
        finally:
            clock.clear()
