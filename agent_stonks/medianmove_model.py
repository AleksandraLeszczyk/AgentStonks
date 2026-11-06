"""Where the day's high and low will land, with no model at all (MedianMove).

Every HighLow notebook measures its model against a forecast with no model in
it, and trades that forecast beside its own: HighLow_1h's and HighLow_4h's "no
model" is the price plus the median excursion of the previous 60 sessions at
the same minute (`naive_forecast` in both). MedianMove asks HighLow's question
-- where will the session's high and low land, asked at 9:35 -- the same way,
with the median of HighLow's own targets standing in for the model:

    up   = log(high  / close5) / adr14     per past session, HighLow's targets
    down = log(close5 / low)   / adr14

    pred_high = max(close5 * exp(median(up)   * adr14), high5)
    pred_low  = min(close5 * exp(-median(down) * adr14), low5)

The medians are over the last `LOOKBACK` sessions before today (at least
`MIN_SESSIONS` of them), each in its own 14-day range; `close5`, `high5`, `low5`
and `adr14` are today's, decoded and clipped to the opening range exactly as
HighLow decodes its predictions (`highlow_model.to_prices`). So the two differ
in one thing only -- where `up` and `down` come from -- and a SimLab replay or
tuning job of one beside the other measures what the model adds.

(HighLow_5m's and HighLow2_5m's notebooks anchored their no-model forecast on
the 9:35 price too, with the 14-day *mean* excursion. The median over 60 is the
1h and 4h notebooks' choice: an excursion has a long right tail -- the trend
days -- and a mean of 14 is pulled about by one of them.)

Apple Trader uses it only for the forecast, as it uses HighLow:
`forecast_session` returns the keys `highlow_model.forecast_session` returns,
and `apple_trader.DayRangeTrader` runs on them unchanged -- the two resting
levels, the breach update, the managed exit, the circuit breaker. Its levels
start from the shared defaults: the notebooks tune every reference forecast's
rule on its own, and so should a Tuning job here.

What it reads
-------------
HighLow's SIP history: the per-session rollups `highlow_model.history_frame`
caches under `data/highlow/`, kept under the settings HighLow's sidecar names
for the ticker (`highlow_model.history_settings`), so the two read one file and
never rebuild each other's. A ticker no HighLow bundle exists for (GOOG) is
kept at HighLow's defaults -- SIP openings, 385 bars. For a ticker whose HighLow
bundle reads the open from IEX (MU) the past openings are IEX's, and so is
today's: the window is fetched here, as HighLow fetches it, whatever tape the
caller's bars are on. Everywhere else today's first five bars are the caller's.

Nothing here reads volume, so the IEX volume caveat
(`day_range.volume_scale_warning`) does not describe it (`reads_volume`).

The rule is arithmetic and this module imports nothing heavier than pandas;
`highlow_model`, which brings PyTorch with it, is imported only to read the
history.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pandas as pd

from . import market_hours

KIND = "medianmove"
DEFAULT_TICKER = "AAPL"

# The sessions behind each median, and how few still make one: the 1h and 4h
# notebooks' `naive_forecast(panel, lookback=60)` with `min_periods=20`.
LOOKBACK = 60
MIN_SESSIONS = 20
# The window today's price and opening range are read from: HighLow's.
OPENING_MINUTES = 5
# The sessions behind `adr14`, the unit every excursion is measured in.
ADR_SESSIONS = 14

OPENING_FEED_SIP = "sip"
OHLC = ["open", "high", "low", "close"]

_forecast_cache: "dict[tuple, dict]" = {}
_FORECAST_CACHE_MAX = 512
_forecast_lock = threading.Lock()


def _highlow():
    """`agent_stonks.highlow_model`, imported on first use: its history cache is
    the one this rule reads, and the module imports PyTorch."""
    from . import highlow_model

    return highlow_model


def rule_path() -> Path:
    """This file: there is no saved model, so the rule is what a stored replay
    of it was made from (SimLab's tuning fingerprints it as it fingerprints a
    bundle)."""
    return Path(__file__).resolve()


def load_bundle(ticker: "str | None" = None) -> "dict | None":
    """What the trader is handed in place of a saved model: the rule's settings
    and how the ticker's history is kept -- or None when `highlow_model`, whose
    history this reads, cannot be imported."""
    symbol = (ticker or DEFAULT_TICKER).upper()
    try:
        settings = _highlow().history_settings(symbol)
    except ImportError:
        return None
    return {
        "kind": KIND,
        "ticker": symbol,
        "opening_minutes": OPENING_MINUTES,
        **settings,
        "lookback": LOOKBACK,
        "min_sessions": MIN_SESSIONS,
        "reads_volume": False,
        "path": str(rule_path()),
    }


def opening_minutes(bundle: "dict | None" = None) -> int:
    try:
        return int((bundle or {})["opening_minutes"])
    except (KeyError, TypeError, ValueError):
        return OPENING_MINUTES


def opening_feed(bundle: "dict | None" = None) -> str:
    """The tape this morning is read from: "sip" (the caller's bars), or "iex" (MU)."""
    return str((bundle or {}).get("opening_feed") or OPENING_FEED_SIP)


def _history_kwargs(bundle: "dict | None") -> dict:
    bundle = bundle or {}
    return {
        "opening_feed": opening_feed(bundle),
        "min_bars": int(bundle.get("min_bars") or 385),
        "max_first_minute": int(bundle.get("max_first_minute") or 0),
    }


# --- the rule -----------------------------------------------------------------

def excursions(history: pd.DataFrame) -> pd.DataFrame:
    """Each past session's HighLow targets -- how far above and below its own
    9:35 price (`close5`) its high and low landed, in its own 14-day ranges --
    beside that `adr14`, oldest first.

    `history` is `highlow_model.history_frame`'s: one row per kept session.
    `adr14` rolls over every row, as HighLow's `daily_features` rolls over its
    daily frame; a session with no opening (an IEX opening feed that printed
    nothing) or without 14 sessions before it has no excursion.
    """
    rng = np.log(history["high"] / history["low"])
    adr14 = rng.rolling(ADR_SESSIONS).mean().shift(1)
    out = pd.DataFrame(
        {
            "up": np.log(history["high"] / history["close5"]) / adr14,
            "down": np.log(history["close5"] / history["low"]) / adr14,
            "adr14": adr14,
        },
        index=history.index,
    )
    return out[np.isfinite(out).all(axis=1)]


def _market_indexed(frame: pd.DataFrame) -> pd.DataFrame:
    """`frame` with its index in exchange-local time (naive read as local)."""
    idx = pd.DatetimeIndex(frame.index)
    idx = (
        idx.tz_localize(market_hours.MARKET_TZ) if idx.tz is None
        else idx.tz_convert(market_hours.MARKET_TZ)
    )
    return frame.set_axis(idx)


def _opening(bars: pd.DataFrame, day: pd.Timestamp, want: int) -> pd.DataFrame:
    """`day`'s sane bars inside the first `want` minutes, exchange-local."""
    local = _market_indexed(bars[OHLC].astype(float))
    minute = local.index.hour * 60 + local.index.minute - (9 * 60 + 30)
    sane = (
        (local > 0).all(axis=1)
        & (local["high"] >= local["low"])
    )
    on_day = local.index.normalize().tz_localize(None) == day
    return local[sane & on_day & (minute >= 0) & (minute < want)]


def forecast_from(
    bundle: dict,
    history: pd.DataFrame,
    opening_bars: pd.DataFrame,
    session_date,
    opening_window: "pd.DataFrame | None" = None,
) -> dict:
    """The day's predicted high and low from a history of session rollups
    (`highlow_model.history_frame`) and today's first five minute bars.

    `opening_window` is today's window on the bundle's opening feed when that
    is not SIP (MU's IEX): the price and the opening range are read from it, as
    HighLow reads them. It may be short, but not empty.

    Returns the keys every day-range model's `forecast_session` returns --
    `{"pred_high", "pred_low", "prev_avg", "adr14_abs", "or_high", "or_low"}`
    in dollars -- plus `median_move`, the two medians in 14-day ranges and the
    sessions they were taken over, for the agent's log. Raises ValueError when
    the inputs cannot support a forecast.
    """
    want = opening_minutes(bundle)
    if len(opening_bars) < want:
        raise ValueError(
            f"the forecast is read off the first {want} minutes and only "
            f"{len(opening_bars)} bars have closed."
        )
    day = pd.Timestamp(session_date).normalize()
    if day.tzinfo is not None:
        day = day.tz_localize(None)

    history = history[history.index < day]
    past = excursions(history).tail(int(bundle.get("lookback") or LOOKBACK))
    need = int(bundle.get("min_sessions") or MIN_SESSIONS)
    if len(past) < need:
        raise ValueError(
            f"only {len(past)} SIP sessions before {day.date()} have a 9:35 price and "
            f"{ADR_SESSIONS} sessions before them; MedianMove's medians need {need}."
        )

    # HighLow's checks, in its order: the window starts at the open and is
    # today's; the price and range are then read off its bars inside the
    # first `want` minutes.
    window = _market_indexed(opening_bars.iloc[:want])
    first = window.index[0]
    if first.hour * 60 + first.minute - (9 * 60 + 30) > int(bundle.get("max_first_minute") or 0):
        raise ValueError("the opening window does not start at 09:30.")
    if (window.index.normalize().tz_localize(None) == day).sum() < want:
        raise ValueError(f"the opening window is not {day.date()}'s first {want} minutes.")
    opening = _opening(window, day, want)
    if not len(opening):
        raise ValueError(f"no sane bar in {day.date()}'s first {want} minutes.")
    feed = opening_feed(bundle)
    if feed != OPENING_FEED_SIP:
        thin = (
            _opening(opening_window, day, want)
            if opening_window is not None and len(opening_window) else opening.iloc[:0]
        )
        if not len(thin):
            # The notebook's inner join drops such a session.
            raise ValueError(
                f"{feed.upper()} printed nothing in {day.date()}'s first {want} minutes, and "
                f"this ticker's 9:35 price is read from {feed.upper()} only (HighLow's history "
                "is kept on its openings)."
            )
        opening = thin

    close5 = float(opening["close"].iloc[-1])
    high5, low5 = float(opening["high"].max()), float(opening["low"].min())
    recent = history.tail(ADR_SESSIONS)
    adr14 = float(np.log(recent["high"] / recent["low"]).mean())
    up, down = float(past["up"].median()), float(past["down"].median())
    last = history.iloc[-1]
    return {
        "pred_high": max(close5 * float(np.exp(up * adr14)), high5),
        "pred_low": min(close5 * float(np.exp(-down * adr14)), low5),
        "prev_avg": (float(last["open"]) + float(last["close"])) / 2,
        "adr14_abs": float((recent["high"] - recent["low"]).mean()),
        "or_high": high5,
        "or_low": low5,
        "median_move": {
            "up": up,
            "down": down,
            "sessions": len(past),
            "from": past.index[0].strftime("%Y-%m-%d"),
            "to": past.index[-1].strftime("%Y-%m-%d"),
            "price": close5,
            "adr14": adr14,
        },
    }


# --- one session's forecast ---------------------------------------------------

def history(bundle: dict, ticker: str, before, key: "str | None" = None,
            secret: "str | None" = None) -> pd.DataFrame:
    """The ticker's sessions strictly before `before`, from HighLow's cache
    (fetched from Alpaca SIP where it falls short)."""
    try:
        return _highlow().history_frame(ticker, before, key, secret, **_history_kwargs(bundle))
    except ValueError as exc:
        raise ValueError(f"MedianMove reads the SIP history HighLow keeps, and {exc}") from exc


def warm_history(bundle: dict, ticker: str, before, key: "str | None" = None,
                 secret: "str | None" = None) -> None:
    """Stretch the history cache `forecast_session` reads over the sessions before `before`."""
    history(bundle, ticker, before, key, secret)


def forecast_session(
    bundle: dict,
    ticker: str,
    opening_bars: pd.DataFrame,
    session_date,
    key: "str | None" = None,
    secret: "str | None" = None,
) -> dict:
    """`forecast_from` with the history read (and cached) for `ticker`.

    Memoised on the inputs, as HighLow's is: a SimLab tuning grid replays the
    same session under dozens of configurations, and the forecast depends on
    none of them. An IEX-opening ticker (MU) is handed today's IEX window,
    fetched here.
    """
    want = opening_minutes(bundle)
    settings = _history_kwargs(bundle)
    window = opening_bars.iloc[:want]
    memo = (
        str(ticker).upper(), pd.Timestamp(session_date).date(), tuple(settings.values()),
        int(bundle.get("lookback") or LOOKBACK), int(bundle.get("min_sessions") or MIN_SESSIONS),
        tuple(map(str, window.index)),
        tuple(np.round(window[OHLC].to_numpy(dtype=float), 6).ravel()),
    )
    with _forecast_lock:
        cached = _forecast_cache.get(memo)
    if cached is not None:
        return {**cached, "median_move": dict(cached["median_move"])}
    past = history(bundle, ticker, session_date, key, secret)
    thin = None
    if settings["opening_feed"] != OPENING_FEED_SIP:
        thin = _highlow().fetch_opening_window(
            ticker, session_date, want, settings["opening_feed"], key, secret,
        )
    out = forecast_from(bundle, past, opening_bars, session_date, thin)
    with _forecast_lock:
        if len(_forecast_cache) >= _FORECAST_CACHE_MAX:
            _forecast_cache.clear()
        _forecast_cache[memo] = {**out, "median_move": dict(out["median_move"])}
    return out
