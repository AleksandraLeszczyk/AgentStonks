"""Helpers more than one analysis family needs, and none of them owns."""

from __future__ import annotations

from datetime import datetime, timedelta

from .. import clock


def _news_datetimes(news_times: "list[str] | None") -> "list[datetime]":
    """The parseable timestamps out of a news-time list, as aware UTC."""
    parsed = (clock.parse_iso(raw) for raw in news_times or [])
    return [dt for dt in parsed if dt is not None]

def _news_near(news_dts: "list[datetime]", when: datetime, window_min: int) -> bool:
    """True if any news timestamp lands within `window_min` minutes (before OR
    after) of `when`. A spike that a catalyst straddles is news-driven whichever
    side the print fell on."""
    tol = timedelta(minutes=window_min)
    return any(abs(nd - when) <= tol for nd in news_dts)
