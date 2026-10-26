"""The day-range rules' arithmetic that no one model owns.

Every model that drives the day-range rule set (`apple_models.STRATEGY_DAYRANGE`
-- HighLow, HighLow2) forecasts the same two numbers once, at 9:35, and hands
`apple_trader.DayRangeTrader` the same keys (`pred_high`, `pred_low`,
`adr14_abs`, ...). What happens to that forecast afterwards is the rule set's,
not the model's, and it lives here: how long the opening window is, which
session "today" is, the caveat for an opening read on IEX volume, and how the
range moves once the tape trades through it.

It came out of TimeToChange3's module when that model was removed
(2026-10-25), because the HighLow models had been leaning on it since they
were added. Nothing here imports torch or LightGBM, so the trader imports
it at module scope.
"""

from __future__ import annotations

import math

import pandas as pd

from . import market_hours
# The intraday-update policy names. They live in `config` rather than here
# because everything that has to *name* one -- the config dataclass, both
# forms, a stored record -- reads config anyway. The arithmetic that gives them
# meaning is below (`updated_range`), and that is what belongs here.
from .config import (  # noqa: F401  -- BREACH_POLICIES is re-exported
    BREACH_BROWNIAN,
    BREACH_CONDITIONAL,
    BREACH_EXTREME,
    BREACH_KEEPS_WIDTH,
    BREACH_OFF,
    BREACH_POLICIES,
    BREACH_SHIFT,
)

# How many minutes of the open a forecast looks at, for a bundle that does not
# say. Every bundle shipped today says 5.
OPENING_MINUTES = 5


def opening_minutes(bundle: "dict | None" = None) -> int:
    """The opening window the bundle's forecast was fitted on."""
    try:
        return int((bundle or {})["opening_minutes"])
    except (KeyError, TypeError, ValueError):
        return OPENING_MINUTES


def market_date() -> "pd.Timestamp":
    """Today, in the exchange's timezone -- the session a forecast belongs to."""
    from .clock import now as _now

    return pd.Timestamp(_now().astimezone(market_hours.MARKET_TZ).date())


def volume_scale_warning(feed: "str | None") -> "str | None":
    """Whether this tape's minute volume is on the scale the forecast was fitted on.

    A bundle whose opening was read from SIP was fitted on consolidated minute
    volume (HighLow's `or_volume_rel` and `or_volume_z`). Alpaca's IEX feed is
    one venue at roughly 4% of it, which shifts a volume ratio by about -3.2 in
    log space -- outside anything the model saw, so the feature's contribution
    stops being information and becomes a constant bias. SIP and yfinance are
    consolidated and fine.

    This takes the feed and not the bundle: the caller skips it for a bundle
    fitted on IEX openings (`opening_feed`), which is on IEX's scale.
    """
    if str(feed or "").lower() != "iex":
        return None
    return (
        "the IEX feed carries about 4% of consolidated minute volume, and the "
        "forecast's opening-volume features were fitted on consolidated volume — "
        "their contribution to this forecast is biased. Run on the SIP or yfinance tape."
    )


# --- the intraday update ------------------------------------------------------
#
# A model's `forecast_session` says it once, at 9:35, and it is already not a
# pure model output: the forecast is clipped to what the tape had printed by
# then, on the grounds that the day's high cannot be below the opening window's
# high. Nothing carried that reasoning past 9:35, so a session that traded
# *through* the predicted high at 11:00 spent the rest of the day being measured
# against a number the tape had already falsified -- and every level derived
# from it (see `apple_trader.DayRangeTrader`) was wrong by the same amount.
#
# These policies extend that one constraint through the session. Three rules
# hold for all of them, and they are what make the result usable as a trading
# level rather than as a running statistic:
#
# * **only on a breach.** The model is never re-run -- it cannot be, its
#   opening features are a fixed five-minute window -- so the only new
#   information here is the tape, and the only place the tape contradicts the
#   forecast is where it has traded through it.
# * **one side at a time.** A high that has been exceeded says nothing about the
#   low, so the unbreached side keeps the number the model gave it.
# * **outward only.** Fed its own last answer each bar (which is how
#   `DayRangeTrader` calls it) each side ratchets: a breach can widen the range
#   and nothing narrows it. A level that could walk back towards the price would
#   turn one move into a stream of entries and exits chasing it.

# Minutes in a regular session (09:30-16:00) -- the unit the Brownian extension
# measures time in, so that a whole session left is tau = 1.
SESSION_MINUTES = 390

# Where the Brownian extension's one constant comes from. Take price over the
# rest of the session as a driftless Brownian motion with per-session volatility
# sigma, run for a fraction tau of a session. Then
#
#     E[max]       = sigma * sqrt(2 * tau / pi)
#     E[max - min] = 2 * E[max] = sigma * sqrt(8 * tau / pi)
#
# and the second is exactly what ADR measures -- the average high-to-low range
# of a whole session, tau = 1. So sigma = ADR * sqrt(pi / 8), and the expected
# one-sided excursion over the minutes still to come is
#
#     ADR * sqrt(pi / 8) * sqrt(2 * tau / pi) = ADR * sqrt(tau) / 2
#
# The two constants cancel to one half. They are written out rather than folded
# into a 0.5 because the halving is a consequence of the model -- the expected
# range is two expected excursions -- and a bare 0.5 in the code would read like
# a fudge factor that could be tuned.
_SIGMA_PER_ADR = math.sqrt(math.pi / 8.0)
_EXPECTED_MAX_PER_SIGMA = math.sqrt(2.0 / math.pi)


def brownian_reach(adr: float, minutes_left: float) -> float:
    """How much further a driftless walk is expected to run, in dollars.

    `adr` is the trailing 14-day average daily range (`adr14_abs`), which is
    what implies the volatility; `minutes_left` the minutes to the close. With
    a whole session left this is half an ADR, and it shrinks with the square
    root of the time remaining -- 0 at the bell.
    """
    if not adr or adr <= 0:
        return 0.0
    tau = max(0.0, min(float(minutes_left), SESSION_MINUTES)) / SESSION_MINUTES
    return float(adr) * _SIGMA_PER_ADR * _EXPECTED_MAX_PER_SIGMA * math.sqrt(tau)


def minutes_left_at(ts) -> float:
    """Minutes from a bar's timestamp to the 16:00 close, never negative.

    Read off the bar rather than off the wall clock, so that a SimLab replay of
    stored bars and the live session it came from get the same number. The
    timestamp is the minute the bar opened, so this is one minute long -- the
    conservative side, and 1/390 of a session under a square root is not a
    distinction a trading level can carry.
    """
    et = pd.Timestamp(ts)
    et = (
        et.tz_localize(market_hours.MARKET_TZ)
        if et.tzinfo is None
        else et.tz_convert(market_hours.MARKET_TZ)
    )
    close = et.normalize() + pd.Timedelta(
        hours=market_hours.MARKET_CLOSE.hour, minutes=market_hours.MARKET_CLOSE.minute
    )
    return max(0.0, (close - et).total_seconds() / 60.0)


def updated_range(
    forecast: dict,
    *,
    session_high: "float | None",
    session_low: "float | None",
    minutes_left: float,
    policy: str = BREACH_OFF,
    contain: bool = False,
    width: "float | None" = None,
    extend: "tuple[bool, bool]" = (False, False),
) -> "tuple[float, float]":
    """`(pred_high, pred_low)` after what the session has actually printed.

    `forecast` is the current forecast -- today's original under `off`, and
    under the other two whatever this function last returned, which is what
    makes each side ratchet. `session_high` and `session_low` are the extremes
    of the session so far, `minutes_left` the minutes to the close (see
    `minutes_left_at`).

    Under `shift` both sides move: the breached side to the extreme, the other
    by the same distance, so the range keeps its width and is re-centred on
    where the day has gone. The side that followed never passes the session's
    own extreme on its side -- a predicted low above a low already printed is
    not a forecast -- so once the session's own range is wider than the
    forecast, the forecast is the session's range.

    Under `extreme` a breached side moves to the extreme itself: the flat
    statement that the day's high is at least what has already traded, which is
    the forecast's 9:35 clip applied to the whole session instead of to its
    first five minutes. Under `brownian` it moves past the extreme by
    `brownian_reach` -- the excursion still expected of a driftless walk with
    ADR-implied volatility over the time left -- on the grounds that a day whose
    high has just been printed is very unlikely to have made its high exactly
    now. The first cannot over-reach and will always be a touch late; the second
    leads the tape and pays for that with a level the day may never come back to.

    `width` (the level unit's dollars at 9:35 -- the predicted range, or the
    ADR) changes what the side that was *not* breached does under `shift` and
    `brownian`: it is put `width` away from the breached side, wherever that
    landed, so the range keeps the width the levels were written against rather
    than whatever width it had drifted to. It is then held to what the session
    has printed (`contain_session`), which is the only thing ever allowed to
    widen it. Without `width` the two policies behave as they did before it
    existed -- `shift` keeps the current width, `brownian` moves one side --
    which is what a stored record replays.

    `conditional` is `shift`, plus the Brownian reach on a side `extend` names
    (`(high, low)`): one whose breach the caller has seen hold
    (`apple_trader.DayRangeTrader._held_breaches`). That side goes to the
    session's extreme plus `brownian_reach` whether or not this bar printed a
    new extreme -- the bar that confirms a breach need not be the one that
    made it, and the extreme was already the range's from the breach itself.
    It never pulls a side back in. Without `width` only that side moves, as
    under `brownian`.

    An unknown policy reads as `off`. Callers get their policy from a stored
    record or a form, and a typo that silently traded a different strategy would
    be worse than one that traded the notebook's.

    `contain` then applies `contain_session` on top, whatever the policy: the
    returned range always holds everything the session has printed. It is the
    difference between `off` meaning "one forecast, held through anything" and
    "do not lead the tape, but do not argue with it either". Note that it makes
    `off` and `extreme` agree on a breached side -- `extreme` already satisfies
    it -- so the two policies only still differ when it is switched off.
    """
    high = float(forecast["pred_high"])
    low = float(forecast["pred_low"])
    reach = (
        brownian_reach(float(forecast.get("adr14_abs") or 0.0), minutes_left)
        if policy in (BREACH_BROWNIAN, BREACH_CONDITIONAL)
        else 0.0
    )
    # Under "conditional" the reach is only a held breach's, and a held side
    # is extended from the extreme even on a bar that printed no new one.
    held_up = policy == BREACH_CONDITIONAL and extend[0] and session_high is not None
    held_down = policy == BREACH_CONDITIONAL and extend[1] and session_low is not None
    if width and width > 0 and policy in BREACH_KEEPS_WIDTH:
        up = session_high is not None and float(session_high) > high
        down = session_low is not None and float(session_low) < low
        reach_up = reach if policy == BREACH_BROWNIAN or held_up else 0.0
        reach_down = reach if policy == BREACH_BROWNIAN or held_down else 0.0
        # The breached side goes where the policy puts it and the other follows
        # at `width`. A bar through both sides leaves no side to follow: both go
        # to their extremes, which is already wider than the width.
        if up or held_up:
            high = max(high, float(session_high) + reach_up)
            if not (down or held_down):
                low = high - float(width)
        if down or held_down:
            low = min(low, float(session_low) - reach_down)
            if not (up or held_up):
                high = low + float(width)
        # A followed side that would exclude a price already printed is the
        # one reason the range may come out wider than `width`.
        high, low = contain_session(
            high, low, session_high=session_high, session_low=session_low
        )
    elif policy in (BREACH_SHIFT, BREACH_CONDITIONAL):
        # The breach on each side, as a distance past the forecast. A bar wide
        # enough to breach both sides moves the range by the difference.
        up = max(0.0, float(session_high) - high) if session_high is not None else 0.0
        down = max(0.0, low - float(session_low)) if session_low is not None else 0.0
        high, low = high + up - down, low + up - down
        # The side that followed is held to what the session has printed: a
        # predicted low above today's low (or a high below today's high) is a
        # number the tape has already disproved. That also puts the breached
        # side exactly on its extreme.
        high, low = contain_session(
            high, low, session_high=session_high, session_low=session_low
        )
        # And a held breach goes on past the extreme, that side alone.
        if held_up:
            high = max(high, float(session_high) + reach)
        if held_down:
            low = min(low, float(session_low) - reach)
    elif policy in (BREACH_EXTREME, BREACH_BROWNIAN):
        if session_high is not None and float(session_high) > high:
            high = float(session_high) + reach
        if session_low is not None and float(session_low) < low:
            low = float(session_low) - reach
    if contain:
        high, low = contain_session(
            high, low, session_high=session_high, session_low=session_low
        )
    return high, low


def contain_session(
    high: float,
    low: float,
    *,
    session_high: "float | None",
    session_low: "float | None",
) -> "tuple[float, float]":
    """The forecast widened to hold everything the session has already printed.

    The forecast's 9:35 clip -- "the day's high cannot be below what already
    printed" -- applied to the whole session rather than to its first
    five minutes. A forecast the tape has traded outside of is not a forecast
    any more, it is a number that has been disproved, and every level hanging
    off it is measured from somewhere the day has already been.

    This is arithmetic rather than a policy: it never invents a price, it only
    refuses to keep one the tape has passed. It is therefore applied whatever
    `updated_range`'s policy is, including `off` -- which is what makes `off`
    mean "do not *lead* the tape" rather than "ignore it".

    Only ever widens, so it cannot move a side back towards the price or cross
    the two over each other.
    """
    if session_high is not None:
        high = max(high, float(session_high))
    if session_low is not None:
        low = min(low, float(session_low))
    return high, low

