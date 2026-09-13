"""
LLM trading agent.

The agent reads data that the app has already fetched (intraday bars, daily
bars, news, quotes — all living on `AppState`) via tool calls, reasons about
the trading regime and a fitting strategy, then finalizes each cycle with
exactly one `submit_decision` tool call (buy / sell / alert). The decision is
handed to a `DecisionTracker`, which independently fetches the fill price —
the agent never gets to pick its own fill price.

Where those decisions go is not this module's business: `DecisionTracker` holds
a `Broker`, which may be the in-process paper ledger, a simulated tape, or a
real Alpaca account (see `agent_stonks.trading_mode`). The one place the venue
does reach the agent is its system prompt — an agent that believes its orders
are inert would reason differently from one spending real money, so
`execution_venue_addendum` tells it which it is.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, time as dt_time, timedelta, timezone
from typing import TYPE_CHECKING, Any, Callable

from . import clock
from . import historical
from . import market_hours
from . import observability as obs
from . import scoring
from . import technical_analysis as ta
from .config import (
    AGENT_MAX_TOOL_ITERS,
    PREMARKET_LEAD_SEC,
    PREMARKET_WAIT_POLL_SEC,
    QUOTE_STALE_SEC,
    QUOTE_WIDE_SPREAD_PCT,
)
from .decisions import whole_shares
from .llm import DEFAULT_AGENT_MODELS, get_agent_client
from .rest import fetch_bars_window, fetch_corporate_actions, fetch_news_window
from .state import (
    ALERTABLE_FIELDS,
    alert_triggered,
    append_agent_log,
    format_alert,
    normalize_alert,
    rvol_pace,
)
from .tactics import (
    TacticsExecutor,
    normalize_tactics,
    tactics_summaries,
)

if TYPE_CHECKING:
    from .decisions import DecisionTracker
    from .state import AppState, SymbolState

# The prompts and the tool schemas moved to their own modules when this one
# outgrew being readable (see `agent_prompts` and `agent_tools`). They are
# re-exported here because `agent_stonks.agent` is the name the app, SimLab and
# the tests import a personality from, and that seam is not worth churning to
# record where the text now lives.
from .agent_prompts import (  # noqa: F401
    AGENT_PERSONALITIES,
    AUTOMATIC_MODE_ADDENDUM,
    BREAKOUT_SYSTEM_PROMPT,
    DEFAULT_PERSONALITY,
    DISABLED_PERSONALITIES,
    MOMENTUM_ADVANCED_LEVELS_ADDENDUM,
    MOMENTUM_SYSTEM_PROMPT,
    MULTI_SYMBOL_ADDENDUM,
    PREMARKET_PERSONALITY,
    PREMARKET_SYSTEM_PROMPT,
    REVERSAL_SYSTEM_PROMPT,
    SESSION_CLOSED_ADDENDUM,
    SMART_MONEY_SYSTEM_PROMPT,
    TACTICS_ADDENDUM,
    VOLUME_DETECTIVE_SYSTEM_PROMPT,
    _session_closed_addendum,
    execution_venue_addendum,
    premarket_briefing_addendum,
    selectable_personalities,
)
from .agent_tools import (  # noqa: F401
    BREAKOUT_TOOLS,
    MOMENTUM_TOOLS,
    PERSONALITY_TOOLS,
    PREMARKET_TOOLS,
    REVERSAL_TOOLS,
    SMART_MONEY_TOOLS,
    VOLUME_DETECTIVE_TOOLS,
    _TOOL_STAND_DOWN,
)



# The agent log belongs to the state, not to this module -- see
# `state.append_agent_log`. Kept as a name here because `apple_trader`,
# `apple_trader2` and `automatic` have imported `_log` from `agent` since
# before that was true.
_log = append_agent_log


def _quote_age_sec(quote_ts: "str | None") -> "float | None":
    """Seconds elapsed since an RFC-3339 quote timestamp, or None if absent/unparseable."""
    ts = clock.parse_iso(quote_ts) if quote_ts else None
    if ts is None:
        return None
    return max(0.0, (clock.now() - ts).total_seconds())


def _tool_get_quote(state: "SymbolState") -> dict:
    with state.lock:
        result = {
            "last_price": state.last_price,
            "prev_close": state.prev_close,
            "bid_price": state.bid_price,
            "bid_size": state.bid_size,
            "ask_price": state.ask_price,
            "ask_size": state.ask_size,
            "quote_time": state.quote_ts,
        }

    # The IEX feed's top-of-book is not the consolidated NBBO: off-hours or
    # with an empty IEX book it degrades to a placeholder-wide (or crossed, or
    # hours-old) quote. Surface spread and age, and attach an explicit warning
    # so the agent leans on last_price instead of unexecutable bid/ask levels.
    warnings = []
    bid, ask = result["bid_price"], result["ask_price"]
    if bid is not None and ask is not None:
        mid = (bid + ask) / 2
        result["spread"] = round(ask - bid, 4)
        if mid > 0:
            spread_pct = (ask - bid) / mid * 100
            result["spread_pct"] = round(spread_pct, 3)
            if ask < bid:
                warnings.append("crossed quote (ask below bid) -- bid/ask unreliable, use last_price")
            elif spread_pct > QUOTE_WIDE_SPREAD_PCT:
                warnings.append(
                    f"spread is {spread_pct:.1f}% of the mid -- placeholder-wide quote from a thin "
                    "IEX book (likely pre/post-market); bid/ask are not executable prices, use last_price"
                )
    age = _quote_age_sec(result["quote_time"])
    if age is not None:
        result["quote_age_sec"] = round(age, 1)
        if age > QUOTE_STALE_SEC:
            warnings.append(
                f"quote is {age / 60:.0f} min old (market closed or stream down) -- bid/ask may be stale"
            )
    if warnings:
        result["warning"] = "; ".join(warnings)
    return result


def _bar_et_date(bar: dict):
    dt = clock.bar_dt(bar)
    return dt.astimezone(market_hours.MARKET_TZ).date() if dt else None


def _fetch_prev_session_bars(symbol: str, today_et) -> "list[dict] | None":
    """Yesterday's intraday session for `symbol`, recovered from yfinance when the
    live buffer doesn't reach back that far. Walks back up to a week so weekends
    and holidays resolve to the last actual trading day."""
    for back in range(1, 6):
        day = today_et - timedelta(days=back)
        try:
            prev = historical.fetch_intraday_bars_for_date(symbol, day.isoformat())
        except Exception:
            prev = []
        if prev:
            return prev
    return None


def _market_window_return(bars: list[dict]) -> "float | None":
    """SPY's % move over the same clock window as `bars`, for beta-adjustment.

    Aligns to the recent window by timestamp so the market move being removed is
    contemporaneous with the ticker's; falls back to the same bar count when
    timestamps are missing (e.g. synthetic bars)."""
    try:
        spy = historical.fetch_intraday_volume_bars(historical.SPY_SYMBOL)
    except Exception:
        spy = []
    if not spy or len(spy) < 2:
        return None
    first_t, last_t = _bar_dt(bars[0]), _bar_dt(bars[-1])
    window = spy
    if first_t and last_t:
        window = [b for b in spy if (bt := _bar_dt(b)) and first_t <= bt <= last_t]
    if len(window) < 2:
        window = spy[-len(bars):]
    if len(window) < 2:
        return None
    start = float(window[0]["c"])
    end = float(window[-1]["c"])
    return (end / start - 1) * 100 if start else None


def _tool_analyze_intraday_momentum(state: "SymbolState", limit: object = None) -> dict:
    n = max(1, min(int(limit or 50), 300))
    with state.lock:
        all_bars = list(state.bars)
    if not all_bars:
        return {"note": "no intraday bars available yet"}
    bars = all_bars[-n:]

    # Split the live buffer into today's and yesterday's ET sessions so the
    # analysis can report today's total move and yesterday's momentum.
    today_et = _bar_et_date(all_bars[-1])
    full_session_bars = None
    prev_session_bars = None
    if today_et is not None:
        full_session_bars = [b for b in all_bars if _bar_et_date(b) == today_et]
        prior_dates = sorted(
            {d for b in all_bars if (d := _bar_et_date(b)) is not None and d < today_et}
        )
        if prior_dates:
            prev_et = prior_dates[-1]
            prev_session_bars = [b for b in all_bars if _bar_et_date(b) == prev_et]
        if not prev_session_bars:
            prev_session_bars = _fetch_prev_session_bars(state.symbol, today_et)

    # Market-neutral inputs: the ticker's long-term beta to SPY and SPY's move
    # over the same recent window. Best-effort -- any fetch failure just omits
    # the market-neutral block rather than failing the whole read.
    market_return_pct = None
    beta_val = None
    beta_window = None
    try:
        beta_info = historical.fetch_market_beta(state.symbol)
        if beta_info:
            market_return_pct = _market_window_return(bars)
            if market_return_pct is not None:
                beta_val = beta_info["beta"]
                beta_window = beta_info["window_days"]
    except Exception:
        market_return_pct = None
        beta_val = None

    return ta.analyze_intraday(
        bars,
        prev_session_bars=prev_session_bars,
        full_session_bars=full_session_bars,
        market_return_pct=market_return_pct,
        beta=beta_val,
        beta_window_days=beta_window,
    )


def _tool_analyze_daily_trend(state: "SymbolState", limit: object = None) -> dict:
    n = max(1, min(int(limit or 60), 365))
    bars = list(state.daily_bars)[-n:]
    if not bars:
        return {"note": "no daily bars available yet"}
    return ta.analyze_trend(bars)


def _opening_range_for(state: "SymbolState", minutes: int, allow_fetch: bool = True) -> "dict | None":
    """Today's opening range for one symbol, from the most reliable source
    available: the per-symbol cache, else measured from the live bar buffer,
    else recovered with a targeted REST fetch of the 09:30 ET window (which
    survives buffer eviction and mid-session starts). Completed ranges are
    cached on the SymbolState. Returns None when the range genuinely cannot
    be established -- never a fabricated range."""
    now = clock.now()
    today_et = now.astimezone(market_hours.MARKET_TZ).date()

    cached = state.opening_range
    if (
        cached
        and cached.get("date") == today_et.isoformat()
        and cached.get("minutes") == minutes
        and cached.get("complete")
    ):
        return cached

    with state.lock:
        bars = list(state.bars)
    rng = ta.compute_opening_range(bars, minutes)
    if "high" not in rng and allow_fetch and state.api_key and state.api_secret:
        open_utc = datetime.combine(
            today_et, market_hours.MARKET_OPEN, tzinfo=market_hours.MARKET_TZ
        ).astimezone(timezone.utc)
        if now >= open_utc:
            try:
                window = fetch_bars_window(
                    state.symbol,
                    "1Min",
                    open_utc,
                    open_utc + timedelta(minutes=minutes),
                    state.api_key,
                    state.api_secret,
                    state.feed,
                )
            except Exception:
                window = []
            if window:
                rng = ta.compute_opening_range(window, minutes, assume_coverage=True)
    if "high" not in rng:
        return None
    if rng.get("complete"):
        state.opening_range = rng
    return rng


def breakout_preconditions(app: "AppState", symbols: list[str], minutes: int = 15) -> "str | None":
    """Deterministic gate for activating the Breakout strategy: returns the
    reason it is NOT currently tradeable, or None when it is.

    The Breakout Trader's whole edge hangs on a real, measurable opening range
    and a session window where breaks tend to follow through -- activating it
    in the midday dead zone, or when no symbol's 09:30 ET window can be
    established, deploys it into a tape it was never designed for.
    """
    clock = ta.session_time_window()
    if not clock.get("favorable_for_breakouts"):
        return f"breakout is not selectable right now: {clock.get('summary')}"
    for symbol in symbols:
        ss = app.sym(symbol)
        if ss is not None and _opening_range_for(ss, minutes) is not None:
            return None
    return (
        "breakout is not selectable: no symbol has a measurable opening range for "
        "today's session (bar history does not reach back to the 09:30 ET open and "
        "the opening window could not be recovered)"
    )


def _tool_analyze_opening_range(state: "SymbolState", minutes: object = None) -> dict:
    n = max(1, min(int(minutes or 15), 120))
    with state.lock:
        bars = list(state.bars)
    if not bars:
        return {"note": "no intraday bars available yet"}
    rng = _opening_range_for(state, n)
    if rng is None:
        # Fall through with the honest measurement note (no cache, no fetch).
        return ta.analyze_opening_range(bars, minutes=n)
    return ta.analyze_opening_range(bars, minutes=n, opening_range=rng)


def _tool_analyze_market(state: "AppState") -> dict:
    data = historical.fetch_market_indicators()
    return ta.analyze_market(
        vix_close=data.get("vix"),
        spy_close=data.get("spy"),
        vix3m_close=data.get("vix3m"),
    )


def _tool_analyze_volume(state: "SymbolState") -> dict:
    # Volume is read from yfinance (consolidated tape across every exchange)
    # rather than Alpaca's single-venue IEX feed, whose few-percent tape share
    # makes absolute volume and volume ratios unreliable. Today's cumulative
    # volume and the ADV baseline both come from yfinance so rvol_pace stays a
    # like-for-like ratio. Fall back to Alpaca's own bars only when yfinance is
    # unavailable.
    #
    # The tactics executor evaluates armed `rvol_pace` conditions from the
    # trading feed's own counters (state.day_volume / state.daily_bars), NOT
    # from the consolidated tape shown here -- the two can diverge by the
    # inverse of the feed's tape share. Report that armable value alongside so
    # the agent chooses thresholds against the number that will actually gate
    # its tactics.
    with state.lock:
        state_day_volume = state.day_volume
    pace_armable = rvol_pace(state_day_volume, state.daily_bars)

    yf_bars = historical.fetch_intraday_volume_bars(state.symbol)
    if yf_bars:
        day_volume = sum(float(b.get("v") or 0.0) for b in yf_bars)
        daily_bars = historical.fetch_daily_volume_bars(state.symbol) or state.daily_bars
        pace = rvol_pace(day_volume, daily_bars)
        result = ta.analyze_volume(yf_bars, rvol_pace=pace, partial_volume_feed=False)
    else:
        with state.lock:
            bars = list(state.bars)
        if not bars:
            return {"note": "no intraday bars available yet"}
        result = ta.analyze_volume(
            bars, rvol_pace=pace_armable, partial_volume_feed=state.feed == "iex"
        )
    if isinstance(result, dict) and "rvol_pace" in result:
        result["rvol_pace_armable"] = (
            round(pace_armable, 2) if pace_armable is not None else None
        )
        if pace_armable is not None and result.get("summary"):
            result["summary"] += (
                f" Armed tactic conditions on rvol_pace evaluate the trading feed's "
                f"value, currently {pace_armable:.2f} -- pick armed thresholds "
                "against that number, not the consolidated pace above."
            )
    return result


def _tool_analyze_consolidation(state: "SymbolState", base_bars: object = None) -> dict:
    n = max(5, min(int(base_bars or 10), 60))
    with state.lock:
        bars = list(state.bars)
    if not bars:
        return {"note": "no intraday bars available yet"}
    return ta.analyze_consolidation(bars, base_bars=n)


def _tool_get_key_levels(state: "SymbolState") -> dict:
    with state.lock:
        bars = list(state.bars)
        spot = state.last_price
    daily = list(state.daily_bars)
    # Same cached/recovered range the ORB tool uses, so the opening-range
    # levels here can never disagree with analyze_opening_range's.
    rng = _opening_range_for(state, 15)
    return ta.key_levels(bars, daily_bars=daily, spot=spot, opening_range=rng)


def _tool_analyze_swing_levels(state: "SymbolState", swing: object = None) -> dict:
    k = max(2, min(int(swing or 3), 10))
    with state.lock:
        bars = list(state.bars)
        spot = state.last_price
    if not bars:
        return {"note": "no intraday bars available yet"}
    return ta.swing_levels(bars, swing=k, spot=spot)


def _tool_analyze_volume_profile(state: "SymbolState", bins: object = None, date: object = None) -> dict:
    n = max(8, min(int(bins or 24), 60))
    day = str(date or "").strip()
    if day:
        # A prior session: pull that day's intraday bars from yfinance rather than
        # today's live buffer. spot stays the live price so the profile still
        # reports where the current price sits relative to the old day's levels.
        try:
            bars = historical.fetch_intraday_bars_for_date(state.symbol, day)
        except ValueError as exc:
            return {"error": str(exc)}
        except Exception:
            return {"note": f"could not fetch intraday bars for {day}"}
        if not bars:
            return {"note": f"no intraday bars for {day} (non-trading day, or outside yfinance's ~60-day window)"}
        with state.lock:
            spot = state.last_price
        result = ta.volume_profile_levels(bars, bins=n, spot=spot)
        result["date"] = day
        return result
    with state.lock:
        bars = list(state.bars)
        spot = state.last_price
    if not bars:
        return {"note": "no intraday bars available yet"}
    return ta.volume_profile_levels(bars, bins=n, spot=spot)


def _news_times_for_date(state: "SymbolState", session_date) -> list[str]:
    """ISO timestamps of `symbol`'s news for one ET session, for the spike
    news-driven classification. Today's session reads the already-loaded live
    news; a prior session is recovered with a dated Alpaca news query. Best
    effort -- any failure just yields no timestamps (spikes fall back to
    supply/demand/unsure)."""
    today_et = clock.now().astimezone(market_hours.MARKET_TZ).date()
    if session_date >= today_et:
        with state.lock:
            return [str(item.get("created_at")) for item in state.news if item.get("created_at")]
    if not (state.api_key and state.api_secret):
        return []
    start = datetime.combine(session_date, dt_time(0, 0), tzinfo=market_hours.MARKET_TZ)
    end = start + timedelta(days=1)
    try:
        articles = fetch_news_window(state.symbol, start, end, state.api_key, state.api_secret)
    except Exception:
        return []
    return [str(item.get("created_at")) for item in articles if item.get("created_at")]


def _tool_analyze_volume_profile_2(state: "SymbolState", bins: object = None, date: object = None) -> dict:
    n = max(8, min(int(bins or 24), 60))
    day = str(date or "").strip()
    with state.lock:
        spot = state.last_price
    if day:
        # A prior session: minute bars for that ET day from yfinance (same
        # consolidated-tape source as today's live volume).
        try:
            bars = historical.fetch_intraday_bars_for_date(state.symbol, day)
        except ValueError as exc:
            return {"error": str(exc)}
        except Exception:
            return {"note": f"could not fetch intraday bars for {day}"}
        if not bars:
            return {"note": f"no intraday bars for {day} (non-trading day, or outside yfinance's ~60-day window)"}
        try:
            session_date = datetime.strptime(day, "%Y-%m-%d").date()
        except ValueError as exc:
            return {"error": str(exc)}
    else:
        # Today's live session: consolidated-tape 1-min bars from yfinance so
        # the volume series is accurate, not Alpaca's single-venue IEX feed.
        bars = historical.fetch_intraday_volume_bars(state.symbol)
        if not bars:
            with state.lock:
                bars = list(state.bars)
        if not bars:
            return {"note": "no intraday bars available yet"}
        session_date = clock.now().astimezone(market_hours.MARKET_TZ).date()

    news_times = _news_times_for_date(state, session_date)
    return ta.analyze_volume_profile_2(
        bars,
        news_times=news_times,
        date=day or None,
        spot=spot,
        price_bins=n,
    )


def _tool_detect_regime_shift(state: "SymbolState") -> dict:
    # Deliberately reads the live streamed buffer rather than yfinance: the
    # whole point is catching the turn as it happens, and the consolidated-tape
    # bars the level tools use are ~15 minutes delayed. Volume is therefore
    # single-venue, but turn_volume_rel is a within-session ratio, so the
    # comparison stays like-for-like.
    with state.lock:
        bars = list(state.bars)
        spot = state.last_price
        news_times = [str(item.get("created_at")) for item in state.news if item.get("created_at")]
    if not bars:
        return {"note": "no intraday bars available yet"}
    return ta.detect_regime_shift(bars, news_times=news_times, spot=spot)


def _tool_get_floor_pivots(state: "SymbolState") -> dict:
    with state.lock:
        spot = state.last_price
    daily = list(state.daily_bars)
    return ta.floor_pivots(daily, spot=spot)


def _tool_get_put_call_walls(state: "SymbolState") -> dict:
    with state.lock:
        data = state.options_chain
        history = list(state.options_wall_history)
    if not data:
        return {"note": "no options chain data available yet"}
    return ta.get_put_call_walls_and_gamma(
        strikes=data["strikes"],
        calls_oi=data["calls_oi"],
        puts_oi=data["puts_oi"],
        calls_gamma_exposure=data["calls_gamma_exposure"],
        puts_gamma_exposure=data["puts_gamma_exposure"],
        spot=data["spot"],
        wall_history=history,
    )


def _tool_get_news(state: "SymbolState", limit: object = None) -> dict:
    n = max(1, min(int(limit or 10), 30))
    with state.lock:
        news = list(state.news)[:n]
        impacts = dict(state.news_impacts)
    return {
        "articles": [
            {
                "headline": item.get("headline"),
                "summary": item.get("summary"),
                "created_at": item.get("created_at"),
                "source": item.get("source"),
                "impact": impacts.get(str(item.get("id", "")), "unknown"),
            }
            for item in news
        ]
    }


def _tool_get_position(app: "AppState", tracker: "DecisionTracker") -> dict:
    snap = tracker.snapshot()
    # Standing conditional orders still armed from a previous cycle, per ticker;
    # a set_tactics call replaces that ticker's plan, actions=[] cancels it.
    armed = {
        ss.symbol: tactics_summaries(ss.tactics)
        for ss in app.iter_symbol_states()
        if ss.tactics is not None
    }
    return {
        "cash": snap["cash"],
        "positions": {sym: qty for sym, qty in snap["positions"].items() if qty},
        # Kept fresh independently by the price stream, not fetched here.
        "portfolio_value": app.portfolio_value,
        "decisions_so_far": len(snap["decisions"]),
        "armed_tactics": armed or None,
    }


def _resolve_symbol_state(app: "AppState", args: dict) -> "tuple[SymbolState | None, str | None]":
    """Resolve a tool call's `symbol` argument to its SymbolState. A missing
    symbol falls back to the sole streamed ticker; otherwise it must name one
    of the streamed tickers. Returns (state, None) or (None, error)."""
    raw = str(args.get("symbol") or "").strip().upper()
    if not raw and len(app.symbols) == 1:
        raw = app.symbols[0]
    state = app.sym(raw) if raw else None
    if state is None:
        return None, (
            f"unknown or missing symbol {raw!r}; pass one of your streamed tickers: "
            f"{', '.join(app.symbols) or '(none)'}"
        )
    return state, None


def _handle_set_tactics(args: dict, app: "AppState", tracker: "DecisionTracker") -> dict:
    """Arm (or cancel) one symbol's tactics as requested by a set_tactics tool call."""
    state, error = _resolve_symbol_state(app, args)
    if error is not None:
        return {"error": error}
    symbol = state.symbol
    raw_actions = args.get("actions")
    reasoning = str(args.get("reasoning") or "")

    if isinstance(raw_actions, list) and not raw_actions:
        had = tactics_summaries(state.tactics)
        state.tactics = None
        _log(app, {"type": "tactics_set", "symbol": symbol, "cancelled": had, "reasoning": reasoning})
        return {"status": "cancelled", "symbol": symbol, "cancelled_tactics": had}

    tactics, error = normalize_tactics(symbol, raw_actions, reasoning)
    if error is not None:
        return {"error": error}

    replaced = tactics_summaries(state.tactics)
    state.tactics = tactics
    summaries = tactics_summaries(tactics)
    with state.lock:
        price = state.last_price
    # Recorded as a no-op "tactics" decision so the arming moment shows up on
    # the portfolio-value chart and in the decision history/report.
    tracker.record_tactics(symbol, summaries, reasoning, price)
    _log(app, {"type": "tactics_set", "symbol": symbol, "tactics": summaries, "replaced": replaced, "reasoning": reasoning})
    return {
        "status": "armed",
        "symbol": symbol,
        "tactics": summaries,
        "replaced_tactics": replaced or None,
        "note": (
            "You are woken the instant any action executes (that ticker's remaining "
            "actions are disarmed). Now finalize the cycle with submit_decision -- "
            "action 'alert' may carry an empty alerts array while tactics are armed."
        ),
    }


def _tool_breakout_trade_geometry(
    entry: object,
    stop: object,
    atr: object = None,
    base_height: object = None,
    overhead_resistance: object = None,
) -> dict:
    return ta.breakout_trade_geometry(
        float(entry),
        float(stop),
        base_height=float(base_height) if base_height is not None else None,
        atr=float(atr) if atr is not None else None,
        overhead_resistance=float(overhead_resistance) if overhead_resistance is not None else None,
    )


def _tool_analyze_vwap_bands(state: "SymbolState", num_std: object = None) -> dict:
    with state.lock:
        bars = list(state.bars)
    if not bars:
        return {"note": "no intraday bars available yet"}
    return ta.analyze_vwap_bands(bars, num_std=float(num_std) if num_std is not None else 2.0)


def _tool_vwap_reversion_geometry(entry: object, vwap: object, std_dev: object, side: object = None
) -> dict:
    return ta.vwap_reversion_geometry(
        float(entry),
        float(vwap),
        float(std_dev),
        side=str(side) if side is not None else "long",
    )


def _tool_analyze_order_blocks(state: "SymbolState") -> dict:
    bars = list(state.daily_bars)
    if not bars:
        return {"note": "no daily bars available yet"}
    with state.lock:
        spot = state.last_price
    return ta.analyze_order_blocks(bars, spot=spot)


def _tool_analyze_fair_value_gaps(state: "SymbolState", limit: object = None) -> dict:
    n = max(1, min(int(limit or 50), 300))
    with state.lock:
        bars = list(state.bars)[-n:]
        spot = state.last_price
    if not bars:
        return {"note": "no intraday bars available yet"}
    return ta.analyze_fair_value_gaps(bars, spot=spot)


def _tool_analyze_smart_money_setup(state: "SymbolState") -> dict:
    daily = list(state.daily_bars)
    if not daily:
        return {"note": "no daily bars available yet"}
    with state.lock:
        intraday = list(state.bars)
        spot = state.last_price
    return ta.analyze_smart_money_setup(daily, intraday_bars=intraday, spot=spot)


def _tool_smart_money_trade_geometry(entry: object, stop: object, target: object) -> dict:
    return ta.smart_money_trade_geometry(float(entry), float(stop), float(target))


def _tool_analyze_liquidity(state: "SymbolState") -> dict:
    with state.lock:
        bars = list(state.bars)
        spot = state.last_price
    if not bars:
        return {"note": "no intraday bars available yet"}
    return ta.analyze_liquidity(bars, spot=spot)


def _tool_analyze_premium_discount(state: "SymbolState") -> dict:
    daily = list(state.daily_bars)
    if not daily:
        return {"note": "no daily bars available yet"}
    with state.lock:
        spot = state.last_price
    return ta.analyze_premium_discount(daily, spot=spot)


def _tool_get_smart_money_flow(state: "SymbolState") -> dict:
    symbol = state.symbol
    if not symbol:
        return {"note": "no symbol set"}
    return historical.fetch_smart_money_flow(symbol)


def _tool_get_analyst_targets(state: "SymbolState") -> dict:
    symbol = state.symbol
    if not symbol:
        return {"note": "no symbol set"}
    # Pass the live streamed price so the upside math is anchored to the current
    # tape rather than Yahoo's slower quote.
    with state.lock:
        spot = state.last_price
    return historical.fetch_analyst_targets(symbol, current_price=spot)


def _tool_get_corporate_actions(state: "SymbolState", days_ahead: object = None) -> dict:
    symbol = state.symbol
    if not symbol:
        return {"note": "no symbol set"}
    key, secret = state.api_key, state.api_secret
    if not key or not secret:
        return {"note": "Alpaca API keys are not configured; corporate actions unavailable"}
    days = max(1, min(int(days_ahead or 14), 90))
    actions = fetch_corporate_actions(symbol, key, secret, days_ahead=days)
    if not actions:
        return {"note": f"no corporate actions scheduled for {symbol} in the next {days} days"}
    return {"window_days": days, "upcoming_corporate_actions": actions}


def _tool_analyze_premarket(state: "SymbolState") -> dict:
    now = clock.now()
    # The session the read is about: the one in progress (edge case: the bell
    # already rang while the analyst was reasoning) or the upcoming one.
    open_dt = market_hours.session_open(now) or market_hours.next_market_open(now)
    with state.lock:
        bars = list(state.bars)
        last_price = state.last_price
        prev_close = state.prev_close

    result: dict = {
        "market_is_open": market_hours.is_market_open(now),
        "market_open_at": open_dt.isoformat(),
        "minutes_until_open": round(max(0.0, (open_dt - now).total_seconds()) / 60.0, 1),
        "prev_close": prev_close,
        "last_price": last_price,
    }
    if last_price is not None and prev_close:
        result["implied_gap_pct"] = round((last_price / prev_close - 1.0) * 100.0, 2)

    # Pre-market bars: same trading day as the open, printed before the bell.
    session_date = open_dt.astimezone(market_hours.MARKET_TZ).date()
    pre_bars = []
    for bar in bars:
        ts = clock.bar_dt(bar)
        if ts is None:
            continue
        if ts < open_dt and ts.astimezone(market_hours.MARKET_TZ).date() == session_date:
            pre_bars.append(bar)
    try:
        if pre_bars:
            result["premarket_session"] = {
                "bars": len(pre_bars),
                "high": max(float(b["h"]) for b in pre_bars),
                "low": min(float(b["l"]) for b in pre_bars),
                "volume": sum(float(b.get("v") or 0.0) for b in pre_bars),
                "last_bar_close": float(pre_bars[-1]["c"]),
                "last_bar_time": pre_bars[-1].get("t"),
            }
        else:
            result["premarket_session"] = {
                "note": "no pre-market bars for the upcoming session yet"
            }
    except (KeyError, TypeError, ValueError):
        result["premarket_session"] = {"note": "pre-market bars are malformed"}
    return result


def _per_symbol(handler: "Callable[..., dict]", *arg_names: str) -> "Callable[[dict, AppState, DecisionTracker], dict]":
    """Wrap a SymbolState-reading tool helper: resolve the call's `symbol` to
    its SymbolState (erroring on unknown tickers), then forward the named args."""

    def run(args: dict, app: "AppState", tracker: "DecisionTracker") -> dict:
        state, error = _resolve_symbol_state(app, args)
        if error is not None:
            return {"error": error}
        return handler(state, *[args.get(name) for name in arg_names])

    return run


_DISPATCH: dict[str, Callable[[dict, "AppState", "DecisionTracker"], dict]] = {
    "get_quote": _per_symbol(_tool_get_quote),
    "analyze_intraday_momentum": _per_symbol(_tool_analyze_intraday_momentum, "limit"),
    "analyze_daily_trend": _per_symbol(_tool_analyze_daily_trend, "limit"),
    "analyze_opening_range": _per_symbol(_tool_analyze_opening_range, "minutes"),
    "analyze_market": lambda args, app, tracker: _tool_analyze_market(app),
    "analyze_volume": _per_symbol(_tool_analyze_volume),
    "analyze_consolidation": _per_symbol(_tool_analyze_consolidation, "base_bars"),
    "get_key_levels": _per_symbol(_tool_get_key_levels),
    "analyze_swing_levels": _per_symbol(_tool_analyze_swing_levels, "swing"),
    "analyze_volume_profile": _per_symbol(_tool_analyze_volume_profile, "bins", "date"),
    "analyze_volume_profile_2": _per_symbol(_tool_analyze_volume_profile_2, "bins", "date"),
    "detect_regime_shift": _per_symbol(_tool_detect_regime_shift),
    "get_floor_pivots": _per_symbol(_tool_get_floor_pivots),
    "breakout_trade_geometry": lambda args, app, tracker: _tool_breakout_trade_geometry(
        args.get("entry"),
        args.get("stop"),
        args.get("atr"),
        args.get("base_height"),
        args.get("overhead_resistance"),
    ),
    "analyze_vwap_bands": _per_symbol(_tool_analyze_vwap_bands, "num_std"),
    "vwap_reversion_geometry": lambda args, app, tracker: _tool_vwap_reversion_geometry(
        args.get("entry"), args.get("vwap"), args.get("std_dev"), args.get("side")
    ),
    "analyze_order_blocks": _per_symbol(_tool_analyze_order_blocks),
    "analyze_fair_value_gaps": _per_symbol(_tool_analyze_fair_value_gaps, "limit"),
    "analyze_smart_money_setup": _per_symbol(_tool_analyze_smart_money_setup),
    "analyze_liquidity": _per_symbol(_tool_analyze_liquidity),
    "analyze_premium_discount": _per_symbol(_tool_analyze_premium_discount),
    "get_smart_money_flow": _per_symbol(_tool_get_smart_money_flow),
    "get_analyst_targets": _per_symbol(_tool_get_analyst_targets),
    "get_corporate_actions": _per_symbol(_tool_get_corporate_actions, "days_ahead"),
    "analyze_premarket": _per_symbol(_tool_analyze_premarket),
    "smart_money_trade_geometry": lambda args, app, tracker: _tool_smart_money_trade_geometry(
        args.get("entry"), args.get("stop"), args.get("target")
    ),
    "get_put_call_walls": _per_symbol(_tool_get_put_call_walls),
    "get_session_clock": lambda args, app, tracker: ta.session_time_window(),
    "get_news": _per_symbol(_tool_get_news, "limit"),
    "get_position": lambda args, app, tracker: _tool_get_position(app, tracker),
}


def _dispatch_tool(name: str, args: dict, app: "AppState", tracker: "DecisionTracker") -> dict:
    handler = _DISPATCH.get(name)
    if handler is None:
        result = {"error": f"unknown tool {name}"}
    else:
        try:
            result = handler(args, app, tracker)
        except Exception as exc:
            result = {"error": str(exc)}
    scoring.record_tool_call(app, name, result)
    return result


def _reject(messages: list[dict], tool_call_id: str, error: str) -> None:
    """Hand a malformed submit_decision back to the model as a tool error so it can
    retry. Used for the cases that have no valid resting state -- an empty alert, a
    zero-quantity trade, or an unrecognized action -- since there is no longer a
    do-nothing decision to silently fall back to."""
    messages.append(
        {"role": "tool", "tool_call_id": tool_call_id, "content": json.dumps({"error": error})}
    )


@obs.observe(name="agent-cycle")
def run_agent_cycle(
    client: Any,
    model: str,
    symbols: list[str],
    state: "AppState",
    tracker: "DecisionTracker",
    max_iters: int = AGENT_MAX_TOOL_ITERS,
    personality: str = DEFAULT_PERSONALITY,
    under_automatic: bool = False,
    system_prompt_override: "str | None" = None,
) -> str:
    """Run one analyze-then-decide cycle over the whole symbol basket. Always
    ends with exactly one recorded decision.

    When Langfuse is configured, the whole cycle is one trace: every LLM turn
    nests under it as a generation, so per-cycle latency, token usage, and cost
    roll up automatically (see `agent_stonks.observability`).

    When `under_automatic` is True the strategy is running under the Automatic
    orchestrator: it also gets a `stand_down` tool to relinquish control when the
    regime no longer fits it. Returns "stand_down" in that case (so the
    orchestrator can re-assess and pick another strategy), or "decided" when the
    cycle finalized with a normal buy/sell/alert (or the forced-sleep fallback).
    """
    symbols_label = ", ".join(symbols)
    obs.update_trace(
        name=f"agent-cycle:{symbols_label}",
        input=symbols_label,
        metadata={"model": model, "symbols": symbols_label, "personality": personality},
    )
    # An override replaces only the personality's base prompt (simlab's editable
    # prompts); the operational addenda below are always appended unchanged.
    system_prompt = system_prompt_override or AGENT_PERSONALITIES.get(
        personality, AGENT_PERSONALITIES[DEFAULT_PERSONALITY]
    )["system_prompt"]
    system_prompt = (
        system_prompt
        + MULTI_SYMBOL_ADDENDUM.format(symbols=symbols_label)
        + TACTICS_ADDENDUM
        # What this run's orders actually do. The venue is a property of the
        # run, not of the personality, and getting it wrong in the prompt is
        # the one error here that could cost real money.
        + execution_venue_addendum(state.trading_mode)
        # The day's research, if a briefing has been generated for these
        # tickers. Empty string when none exists (nothing generated yet, a
        # symbol added mid-session, or a simulation), so the prompt is
        # unchanged from before in that case.
        + premarket_briefing_addendum(
            state.premarket_briefings,
            symbols,
            generated_at=state.premarket_generated_at,
            phase=state.premarket_phase,
        )
    )
    if personality != PREMARKET_PERSONALITY:
        system_prompt = system_prompt + _session_closed_addendum()
    tools = PERSONALITY_TOOLS.get(personality, MOMENTUM_TOOLS)
    if under_automatic:
        system_prompt = system_prompt + AUTOMATIC_MODE_ADDENDUM
        tools = [*tools, _TOOL_STAND_DOWN]
    state.clear_alerts()
    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Tickers: {symbols_label}. Run your analysis process and finish by calling submit_decision.",
        },
    ]
    _log(state, {"type": "cycle_start", "text": f"Starting analysis cycle for {symbols_label}"})

    decision_made = False
    stood_down = False
    for _ in range(max_iters):
        try:
            response = client.chat.completions.create(
                model=model, messages=messages, tools=tools, tool_choice="auto"
            )
        except Exception as exc:
            _log(state, {"type": "error", "text": f"LLM call failed: {exc}"})
            break

        msg = response.choices[0].message
        if msg.content:
            _log(state, {"type": "analysis", "text": msg.content})

        assistant_msg: dict = {"role": "assistant", "content": msg.content}
        tool_calls = getattr(msg, "tool_calls", None) or []
        if tool_calls:
            calls = []
            for tc in tool_calls:
                call = {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                # Gemini 3+ "thinking" models attach a thought_signature here that must be
                # echoed back verbatim on the next turn, or the API rejects the request with
                # "Function call ... is missing a thought_signature".
                extra_content = getattr(tc, "extra_content", None)
                if extra_content:
                    call["extra_content"] = extra_content
                calls.append(call)
            assistant_msg["tool_calls"] = calls
        messages.append(assistant_msg)

        if not tool_calls:
            messages.append(
                {"role": "user", "content": "Please finalize this cycle by calling submit_decision now."}
            )
            continue

        for tc in tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            if name == "stand_down" and under_automatic:
                reasoning = args.get("reasoning", "")
                quiet = args.get("expected_quiet_minutes")
                _log(
                    state,
                    {
                        "type": "stand_down",
                        "personality": personality,
                        "reasoning": reasoning,
                        "expected_quiet_minutes": quiet,
                    },
                )
                obs.update_trace(output={"action": "stand_down", "reasoning": reasoning})
                # The relinquishing strategy's conditional orders must not keep
                # trading under whatever regime/strategy comes next.
                for ss in state.iter_symbol_states():
                    ss.tactics = None
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": json.dumps({"status": "relinquished"})}
                )
                stood_down = True
                decision_made = True
                break

            if name == "set_tactics":
                # _handle_set_tactics writes its own "tactics_set" log entry on
                # success; only a validation failure is logged as a plain tool call.
                result = _handle_set_tactics(args, state, tracker)
                scoring.record_tactics_call(state, ok="error" not in result)
                result_content = json.dumps(result)
                if "error" in result:
                    _log(state, {"type": "tool_call", "name": name, "args": args, "result": result})
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_content})
                continue

            if name == "submit_decision":
                action = args.get("action", "")
                # Whole shares only: a fractional request is rounded down here,
                # so the model is told when that leaves nothing to trade.
                quantity = whole_shares(float(args.get("quantity") or 0))
                reasoning = args.get("reasoning", "")
                regime = args.get("regime", "unknown")
                default_symbol = symbols[0] if len(symbols) == 1 else None
                # Validate each requested condition against the watchable-field
                # and streamed-symbol registries; silently drop malformed specs
                # so one bad entry doesn't sink a valid bracket.
                alerts = [
                    a
                    for a in (
                        normalize_alert(r, symbols=symbols, default_symbol=default_symbol)
                        for r in (args.get("alerts") or [])
                    )
                    if a is not None
                ]

                if action == "alert" and not alerts and not state.any_tactics():
                    # Some models (small/cheap ones especially) pick action="alert" but
                    # forget the conditions, or name a field that isn't watchable. Reject
                    # and let the model retry instead of silently recording a no-op -- it
                    # never sees that happen otherwise, so it can't course-correct.
                    _reject(
                        messages,
                        tc.id,
                        "action 'alert' requires a non-empty 'alerts' array, each "
                        f"entry having symbol (one of: {', '.join(symbols)}), field "
                        f"(one of: {', '.join(ALERTABLE_FIELDS)}), condition ('above' or "
                        "'below'), and a numeric value. Call submit_decision again "
                        "with at least one valid condition (or arm a plan with "
                        "set_tactics first -- an empty alerts array is only allowed "
                        "while tactics are armed). Standing aside always means "
                        "setting an alert -- there is no do-nothing action.",
                    )
                    continue

                if action in ("buy", "sell") and quantity <= 0:
                    # A trade with no size is not a trade. Don't silently record a
                    # no-op -- make the model either commit to a size or stand aside
                    # explicitly with an alert.
                    _reject(
                        messages,
                        tc.id,
                        f"action '{action}' requires a quantity of at least 1 whole "
                        "share (quantities are rounded down to whole shares). Call "
                        "submit_decision again with a positive integer quantity, or use action "
                        "'alert' with one or more conditions to watch if you don't want "
                        "to trade right now.",
                    )
                    continue

                if action in ("buy", "sell"):
                    trade_state, symbol_error = _resolve_symbol_state(app=state, args=args)
                    if symbol_error is not None:
                        _reject(
                            messages,
                            tc.id,
                            f"action '{action}' requires a valid 'symbol': {symbol_error}. "
                            "Call submit_decision again with the ticker to trade.",
                        )
                        continue
                    decision = tracker.record_trade(
                        trade_state.symbol, action, quantity, reasoning,
                        state.api_key, state.api_secret, state.feed,
                    )
                elif action == "alert":
                    involved = sorted({a["symbol"] for a in alerts}) or list(symbols)
                    decision = tracker.record_alert(", ".join(involved), alerts, reasoning)
                    # Distribute each condition to the SymbolState whose stream
                    # watches it (cycle start cleared the previous set).
                    for a in alerts:
                        alert_state = state.sym(a["symbol"])
                        if alert_state is not None:
                            alert_state.alerts = [*alert_state.alerts, a]
                else:
                    # Unknown / removed action (e.g. a model still reaching for the old
                    # "sleep"). There is no do-nothing path: reject and retry.
                    _reject(
                        messages,
                        tc.id,
                        "action must be one of 'buy', 'sell', or 'alert'. To stand aside "
                        "without trading, use action 'alert' with one or more conditions "
                        "to watch -- there is no 'sleep' or do-nothing action. Call "
                        "submit_decision again.",
                    )
                    continue
                _log(
                    state,
                    {
                        "type": "decision",
                        "action": decision.action,
                        "symbol": decision.symbol,
                        "status": decision.status,
                        "price": decision.price,
                        "quantity": decision.filled_quantity,
                        "reasoning": reasoning,
                        "regime": regime,
                        "alerts": decision.alerts,
                    },
                )
                result_content = json.dumps(
                    {
                        "status": decision.status,
                        "filled_quantity": decision.filled_quantity,
                        "price": decision.price,
                        "cash_after": decision.cash_after,
                        "position_after": decision.position_after,
                    }
                )
                decision_made = True
                obs.update_trace(
                    output={"action": decision.action, "regime": regime, "reasoning": reasoning}
                )
            else:
                result = _dispatch_tool(name, args, state, tracker)
                result_content = json.dumps(result)
                _log(state, {"type": "tool_call", "name": name, "args": args, "result": result})

            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_content})

        if decision_made:
            break

    # Deterministic numeric-faithfulness check over the cycle's full transcript
    # (see agent_stonks.scoring); aggregated into the daily scoring session.
    scoring.record_cycle_grounding(state, messages, personality)

    if stood_down:
        return "stand_down"

    if not decision_made:
        forced = tracker.record_sleep(
            symbols_label, "Max reasoning iterations reached without a finalized decision; defaulting to sleep."
        )
        obs.update_trace(output={"action": forced.action, "regime": "unknown", "reasoning": forced.reasoning})
        _log(
            state,
            {
                "type": "decision",
                "action": forced.action,
                "status": forced.status,
                "price": forced.price,
                "quantity": forced.filled_quantity,
                "reasoning": forced.reasoning,
                "regime": "unknown",
            },
        )

    return "decided"


def _wait_for_next_cycle(state: "AppState", stop_event: threading.Event, cycle_sec: int) -> None:
    """Block until the next cycle is actually due.

    With no active alert or armed tactics, this is a plain `cycle_sec` timer
    (woken early only by fresh news). With an active alert or armed tactics,
    the fixed timer is disabled -- the agent committed to "nothing changes
    until a watched condition fires, a tactic executes, or news arrives", so it
    should wait indefinitely for `state.agent_wake_event` rather than also
    waking on the next scheduled tick. The price/news stream threads and the
    TacticsExecutor set that event directly the moment a condition is met, a
    conditional trade fills, or fresh news arrives -- never on a timer just to
    check state.
    """
    state.agent_wake_event.clear()
    state.agent_wake_reason = None
    if stop_event.is_set():
        return

    # The stream only signals on the *next* tick, so an alert condition that's
    # already satisfied the instant it's set would otherwise wait for a tick
    # that may not come. Catch that once, up front.
    alert_pairs = state.iter_alerts()
    if alert_pairs:
        hit = next((a for ss, a in alert_pairs if alert_triggered(ss, a)), None)
        if hit is not None:
            state.clear_alerts()
            _log(
                state,
                {"type": "status", "text": f"Alert already met: {format_alert(hit)}; waking early."},
            )
            return

    # An active alert or armed tactics (on any symbol) mean the agent should
    # sleep until a condition fires, a tactic executes, or news arrives -- not
    # get woken by the regular cycle timer too. (Armed tactics are watched
    # independently by the per-symbol TacticsExecutors, which wake this thread
    # on execution.)
    timeout = None if (alert_pairs or state.any_tactics()) else cycle_sec
    woke_early = state.agent_wake_event.wait(timeout=timeout)
    if stop_event.is_set():
        return
    if woke_early and state.agent_wake_reason:
        _log(state, {"type": "status", "text": f"{state.agent_wake_reason} Waking early."})
    state.agent_wake_event.clear()
    state.agent_wake_reason = None


def _wait_for_premarket_window(state: "AppState", stop_event: threading.Event) -> bool:
    """Block until PREMARKET_LEAD_SEC before the next opening bell -- the
    earliest moment the Premarket Analyst is allowed to start its analysis.
    Returns False when the agent was stopped while holding."""
    logged = False
    while not stop_event.is_set():
        remaining = market_hours.seconds_until_next_open() - PREMARKET_LEAD_SEC
        if remaining <= 0:
            return True
        if not logged:
            open_at = market_hours.next_market_open()
            _log(
                state,
                {
                    "type": "status",
                    "text": (
                        f"Premarket analyst holding until {PREMARKET_LEAD_SEC // 60} min "
                        f"before the bell (opens {open_at.strftime('%Y-%m-%d %H:%M UTC')})."
                    ),
                },
            )
            logged = True
        stop_event.wait(min(remaining, PREMARKET_WAIT_POLL_SEC))
    return False


def run_premarket_session(
    client: Any,
    model: str,
    symbols: list[str],
    state: "AppState",
    tracker: "DecisionTracker",
    stop_event: threading.Event,
) -> str:
    """Run the Premarket Analyst end to end: hold until PREMARKET_LEAD_SEC
    before the opening bell, run one opening-tactics cycle, then sleep until an
    armed tactic executes (the opening trade is simulated by the
    TacticsExecutor). Fresh news before the bell wakes it to revise the plan;
    any wake after the open just keeps it sleeping until a tactic fires.

    Returns "executed" once an opening tactic filled, "done" when the bell rang
    with nothing armed (nothing to perform), or "stopped".
    """
    while not stop_event.is_set():
        if not _wait_for_premarket_window(state, stop_event):
            return "stopped"

        state.agent_wake_event.clear()
        state.agent_wake_reason = None
        try:
            run_agent_cycle(
                client, model, symbols, state, tracker, personality=PREMARKET_PERSONALITY
            )
        except Exception as exc:
            _log(state, {"type": "error", "text": f"Premarket cycle failed: {exc}"})

        if not state.any_tactics():
            # No opening plan -- hold through the bell (so a caller that
            # re-assesses on return doesn't spin pre-open) and retire.
            _log(
                state,
                {
                    "type": "status",
                    "text": "Premarket analyst armed no opening tactics; retiring at the bell.",
                },
            )
            while not stop_event.is_set() and not market_hours.is_market_open():
                stop_event.wait(PREMARKET_WAIT_POLL_SEC)
            return "stopped" if stop_event.is_set() else "done"

        # Opening tactics armed: sleep until the executor performs one.
        while not stop_event.is_set():
            state.agent_wake_event.wait()
            if stop_event.is_set():
                return "stopped"
            reason = state.agent_wake_reason or ""
            state.agent_wake_event.clear()
            state.agent_wake_reason = None
            if reason.startswith("Tactics executed"):
                _log(
                    state,
                    {
                        "type": "status",
                        "text": "Opening tactic executed; premarket analyst retiring.",
                    },
                )
                return "executed"
            if not state.any_tactics():
                # Tactics were cleared without an execution (external cancel).
                return "done"
            if not market_hours.is_market_open():
                # Pre-bell wake (fresh news / alert): revise the opening plan.
                _log(
                    state,
                    {
                        "type": "status",
                        "text": f"{reason} Premarket analyst revising the opening plan.",
                    },
                )
                break
            # Post-open wake that wasn't an execution: the bracket is still
            # armed and watched -- keep sleeping until a tactic fires.
    return "stopped"


def _premarket_loop(
    state: "AppState",
    tracker: "DecisionTracker",
    symbols: list[str],
    provider: str,
    api_key: str,
    model: str,
    stop_event: threading.Event,
) -> None:
    """Standalone Premarket Analyst run: one premarket session, then the agent
    disables itself -- the opening tactics were performed (or there was nothing
    to perform) and this personality never trades the session that follows."""
    client = get_agent_client(provider, api_key)
    outcome = run_premarket_session(client, model, symbols, state, tracker, stop_event)
    if outcome != "stopped" and state.agent_stop_event is stop_event:
        stop_agent(state)
    scoring.end_session(state, tracker)
    state.agent_running = False
    _log(state, {"type": "status", "text": "Premarket analyst disabled."})
    obs.flush()


def _agent_loop(
    state: "AppState",
    tracker: "DecisionTracker",
    symbols: list[str],
    provider: str,
    api_key: str,
    model: str,
    cycle_sec: int,
    stop_event: threading.Event,
    personality: str = DEFAULT_PERSONALITY,
) -> None:
    client = get_agent_client(provider, api_key)
    while not stop_event.is_set():
        try:
            run_agent_cycle(client, model, symbols, state, tracker, personality=personality)
        except Exception as exc:
            _log(state, {"type": "error", "text": f"Agent cycle failed: {exc}"})
        # Daily scoring may come due mid-session on a long-running agent; the
        # check is one stat() call once the day is scored.
        scoring.maybe_score_day(state, tracker)
        _wait_for_next_cycle(state, stop_event, cycle_sec)
    scoring.end_session(state, tracker)
    state.agent_running = False
    _log(state, {"type": "status", "text": "Agent stopped"})


def start_tactics_executor(state: "AppState", tracker: "DecisionTracker") -> None:
    """Start one background matcher per streamed symbol for its armed tactics;
    stopped by `stop_agent`. Shared by `launch_agent` and the Automatic
    orchestrator."""
    for sym_state in state.iter_symbol_states():
        executor = TacticsExecutor(sym_state, tracker)
        sym_state.tactics_executor = executor
        executor.start()


def launch_agent(
    state: "AppState",
    tracker: "DecisionTracker",
    symbols: list[str],
    api_key: str,
    provider: str = "openai",
    model: "str | None" = None,
    cycle_sec: int = 60,
    personality: str = DEFAULT_PERSONALITY,
) -> None:
    """Stop any running agent for this state, then start a new background cycle
    loop trading the whole symbol basket."""
    model = model or DEFAULT_AGENT_MODELS[provider]
    stop_agent(state)
    stop_event = threading.Event()
    state.agent_stop_event = stop_event
    state.agent_running = True
    scoring.begin_session(state, personality, symbols)
    start_tactics_executor(state, tracker)
    if personality == PREMARKET_PERSONALITY:
        # One-shot pre-open specialist: holds for the opening window, arms the
        # opening tactics, and disables itself once they execute.
        threading.Thread(
            target=_premarket_loop,
            args=(state, tracker, symbols, provider, api_key, model, stop_event),
            daemon=True,
        ).start()
        return
    threading.Thread(
        target=_agent_loop,
        args=(state, tracker, symbols, provider, api_key, model, cycle_sec, stop_event, personality),
        daemon=True,
    ).start()


def stop_agent(state: "AppState") -> None:
    if state.agent_stop_event:
        state.agent_stop_event.set()
    state.agent_running = False
    # Disarm every symbol's standing conditional orders and their matchers --
    # with no agent to wake, tactics must not keep trading on their own.
    for sym_state in state.iter_symbol_states():
        if sym_state.tactics_executor is not None:
            sym_state.tactics_executor.stop()
            sym_state.tactics_executor = None
        sym_state.tactics = None
    # Interrupt a blocked _wait_for_next_cycle immediately instead of letting
    # it sit until the timeout expires.
    state.agent_wake_event.set()
    # Push any buffered traces from the cycle(s) that just ran to Langfuse
    # before the background flusher would otherwise get to them.
    obs.flush()
