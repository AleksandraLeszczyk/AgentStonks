from datetime import datetime, timezone

from agent_stonks.market_hours import MARKET_TZ
from agent_stonks.session_timer import SESSION_SEC, before_label, session_seconds


def _et(year: int, month: int, day: int, hour: int, minute: int, second: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, second, tzinfo=MARKET_TZ)


# 2026-07-06 is a Monday; 2026-07-04 a Saturday; 2026-07-10 a Friday.
class TestSessionSeconds:
    def test_zero_at_the_bell(self):
        assert session_seconds(_et(2026, 7, 6, 9, 30)) == 0

    def test_counts_up_through_the_session(self):
        assert session_seconds(_et(2026, 7, 6, 10, 53, 21)) == 3600 + 23 * 60 + 21

    def test_negative_before_the_bell(self):
        assert session_seconds(_et(2026, 7, 6, 9, 17, 26)) == -(12 * 60 + 34)

    def test_last_second_of_the_session(self):
        assert session_seconds(_et(2026, 7, 6, 15, 59, 59)) == SESSION_SEC - 1

    def test_the_close_counts_down_to_the_next_open(self):
        assert session_seconds(_et(2026, 7, 6, 16, 0)) == -17.5 * 3600

    def test_the_weekend_counts_down_to_monday(self):
        # Friday 17:00 -> Monday 09:30 is 2 days 16.5 hours.
        assert session_seconds(_et(2026, 7, 10, 17, 0)) == -(2 * 86400 + 16.5 * 3600)

    def test_accepts_utc_input(self):
        # 14:00 UTC on a summer Monday = 10:00 ET.
        assert session_seconds(datetime(2026, 7, 6, 14, 0, tzinfo=timezone.utc)) == 1800

    def test_session_is_six_and_a_half_hours(self):
        assert SESSION_SEC == 6.5 * 3600


class TestBeforeLabel:
    def test_same_day_open_has_no_weekday(self):
        assert before_label(_et(2026, 7, 6, 8, 0)) == "Before the open · 09:30 ET"

    def test_after_the_close_names_tomorrow(self):
        assert before_label(_et(2026, 7, 6, 16, 30)) == "Before the open · Tue 09:30 ET"

    def test_weekend_names_monday(self):
        assert before_label(_et(2026, 7, 4, 12, 0)) == "Before the open · Mon 09:30 ET"
