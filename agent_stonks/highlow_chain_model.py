"""HighLow Chain: one day-range forecast, re-made by whichever model's hour it is.

Three FinNotebooks models each answer "where will the high and low land?" for
a different stretch of the session, and each is the best of them there:

    from          model                       forecasts the high and low of
    09:35         HighLow2_5m (`highlow2_model`)  the whole session, once
    10:30 - 13:25 HighLow_1h (`highlow_window_model`)  now to 13:30, every 5 min
    13:30 - 15:55 HighLow_4h (`highlow_window_model`)  now to the close, every 5 min

The chain hands Apple Trader's day-range rules (`apple_trader.DayRangeTrader`)
HighLow2's forecast at 9:35, exactly as the HighLow2 model does, and from 10:30
replaces it with the window model's latest at each of their decision minutes:
the forecast at minute m is made once bar m - 1 has closed and covers the bars
from m on. Between decisions the breach rules move it as they move any
day-range forecast, on the bars after the decision. Both window models
measured against HighLow2 read for the same windows: -74% (1h) and -75% (4h)
error on their 2026 holdouts.

What the chain does *not* claim: a measured trading edge. Its three parts were
fitted and scored apart, nobody has replayed them as one, and the window
models' own trading rules did not survive their notebooks' holdouts. Its day
range is untuned: the levels are the shared fallback until a SimLab Tuning
pick is made for it. And a window forecast is narrower than a day forecast --
at 13:25 it is five minutes wide -- so a sell level measured under it comes
down to the price as its window runs out.

On a 13:00 half day there is no window forecast (the notebooks keep no such
session) and HighLow2's stands all session.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import highlow2_model, highlow_window_model as windows
from .highlow_window_model import HEADS

KIND = "highlow_chain"
DEFAULT_TICKER = highlow2_model.DEFAULT_TICKER


def load_bundle(ticker: "str | None" = None) -> "dict | None":
    """HighLow2's bundle and the two window models' for one ticker, or None
    when any of the three cannot be assembled."""
    day = highlow2_model.load_bundle(ticker)
    parts = {head: windows.load_bundle(ticker, head) for head in HEADS}
    if day is None or any(part is None for part in parts.values()):
        return None
    return {
        "kind": KIND,
        "highlow2": day,
        "windows": parts,
        "opening_minutes": highlow2_model.opening_minutes(day),
        "opening_feed": highlow2_model.opening_feed(day),
        "path": day["path"],
        "ticker": day.get("ticker"),
    }


def paths(ticker: "str | None" = None) -> "list[Path]":
    """The three saved models' files, HighLow2's first."""
    return [highlow2_model.model_path(ticker)] + [windows.model_path(ticker, head) for head in HEADS]


def unavailable_reason(ticker: "str | None" = None) -> str:
    """Which of the three is missing, by name."""
    missing = []
    if highlow2_model.load_bundle(ticker) is None:
        missing.append(f"HighLow2 ({highlow2_model.model_path(ticker)})")
    for head in HEADS.values():
        if windows.load_bundle(ticker, head.key) is None:
            missing.append(f"{head.label} ({windows.model_path(ticker, head.key)})")
    if not missing:
        return "All three models loaded; try again."
    return (
        "HighLow Chain needs all three of its models, and "
        + ", ".join(missing)
        + " could not be loaded (or LightGBM and joblib are not installed)."
    )


def opening_minutes(bundle: "dict | None" = None) -> int:
    return highlow2_model.opening_minutes((bundle or {}).get("highlow2"))


def opening_feed(bundle: "dict | None" = None) -> str:
    return highlow2_model.opening_feed((bundle or {}).get("highlow2"))


def decisions() -> "list[int]":
    """Every decision minute of the chain's window models (minutes after 9:30)."""
    return sorted(m for head in HEADS.values() for m in head.decisions)


def bar_minute(ts) -> int:
    """Minutes after 9:30 of the bar stamped `ts` (0 = the 09:30 bar)."""
    local = pd.Timestamp(ts)
    if local.tzinfo is not None:
        local = local.tz_convert(windows.MARKET_TZ)
    return int(local.hour * 60 + local.minute - (9 * 60 + 30))


def window_forecasts(
    bundle: dict,
    ticker: str,
    session_date,
    through: "int | None" = None,
    key: "str | None" = None,
    secret: "str | None" = None,
    wait: bool = True,
) -> "tuple[dict[int, dict], dict[int, str]]":
    """The window models' forecasts made by the close of bar `through` (all of
    the session's when None): `highlow_window_model.session_forecasts`."""
    return windows.session_forecasts(
        bundle["windows"], ticker, session_date, through, key, secret, wait=wait,
    )


def forecast_session(
    bundle: dict,
    ticker: str,
    opening_bars: pd.DataFrame,
    session_date,
    key: "str | None" = None,
    secret: "str | None" = None,
    through: "int | None" = None,
) -> dict:
    """HighLow2's 9:35 forecast (`highlow2_model.forecast_session`'s keys),
    marked as the chain's, and -- with `through` -- the window forecasts made
    by the close of that bar under `windows` (minute -> forecast), for a caller
    that walks the session at once (the chart, SimLab's tuning chart).

    The trader asks for the windows itself, bar by bar
    (`DayRangeTrader._extend_windows`). A session the window models cannot
    forecast (a half day, a short history) gets HighLow2's alone, with the
    reason under `windows_problem`.
    """
    out = highlow2_model.forecast_session(bundle["highlow2"], ticker, opening_bars, session_date, key, secret)
    out["chain"] = True
    if through is not None:
        try:
            rows, _ = window_forecasts(bundle, ticker, session_date, through, key, secret, wait=False)
            out["windows"] = rows
        except Exception as exc:  # noqa: BLE001 -- HighLow2's forecast still stands
            out["windows"] = {}
            out["windows_problem"] = str(exc)
    return out


def warm_history(bundle: dict, ticker: str, before, key: "str | None" = None,
                 secret: "str | None" = None) -> None:
    """Stretch the caches the session's forecasts read: HighLow2's history, and
    the window models' tape, option table and event history."""
    highlow2_model.warm_history(bundle["highlow2"], ticker, before, key, secret)
    windows.warm(ticker, before, tuple(bundle["windows"]), key, secret)
