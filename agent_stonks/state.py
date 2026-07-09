import json
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Iterable, Iterator

if TYPE_CHECKING:
    import websocket

    from .decisions import DecisionTracker

from .config import (
    MAX_BARS,
    PAPER_STARTING_CASH,
    VOLUME_ADV_MIN_DAYS,
    VOLUME_ADV_WINDOW,
    VOLUME_ALERT_DEFAULT_MULTIPLIER,
)

# Continuously-updated state fields the agent can attach a condition alert to.
# Every entry is refreshed on the live price/quote stream (and the REST
# fallback), so an alert on any of them can fire between scheduled cycles. The
# value is a human description used in tool schemas and the UI. `spread` and
# `volume_ratio` are derived (see `alert_field_value`) rather than stored
# directly, but update just as continuously as their inputs. All fields except
# `portfolio_value` are per-symbol; an alert always carries the symbol whose
# stream it watches.
ALERTABLE_FIELDS: dict[str, str] = {
    "last_price": "Latest traded price",
    "bid_price": "Best (highest) bid price",
    "ask_price": "Best (lowest) ask price",
    "bid_size": "Shares offered at the best bid",
    "ask_size": "Shares offered at the best ask",
    "spread": "Ask price minus bid price (absolute, same units as price)",
    "previous_minute_high": "High of the last completed 1-minute bar",
    "previous_minute_low": "Low of the last completed 1-minute bar",
    "day_volume": "Cumulative shares traded so far today",
    "volume_ratio": (
        "Today's cumulative volume divided by the average FULL day's volume -- climbs "
        "from ~0 toward ~1 over a normal session, so it is NOT an intraday-pace measure "
        "and makes a poor breakout-confirmation condition"
    ),
    "portfolio_value": "Paper portfolio value (cash + all positions marked to last price)",
}

# Subset of alertable fields that live on the price axis, so a triggered/pending
# alert on them can be drawn as a horizontal line on the price chart.
PRICE_AXIS_ALERT_FIELDS: frozenset[str] = frozenset(
    {"last_price", "bid_price", "ask_price", "previous_minute_high", "previous_minute_low"}
)


class SymbolState:
    """Per-ticker slice of the app state: everything the stream fills for one
    symbol (bars, trades, news, quotes) plus the per-symbol agent artifacts
    (pending condition alerts, armed tactics and their executor, options data).

    Shared, app-wide values -- API credentials, the paper ledger, the agent's
    wake plumbing -- live on the parent `AppState`; the delegation properties
    below expose them so per-symbol code (stream handlers, the tactics
    executor, agent tools) can keep reading/writing them through this object.
    """

    def __init__(self, symbol: str, app: "AppState") -> None:
        self.symbol = symbol
        self.app = app
        self.lock = threading.Lock()
        self.bars: deque[dict] = deque(maxlen=MAX_BARS)
        self.daily_bars: list[dict] = []
        self.trades: list[dict] = []
        self.news: list[dict] = []
        self.news_impacts: dict[str, str] = {}
        self.status: str = "Idle"
        self.news_status: str = "Idle"
        self.last_price: float | None = None
        self.prev_close: float | None = None
        self.bid_price: float | None = None
        self.bid_size: float | None = None
        self.ask_price: float | None = None
        self.ask_size: float | None = None
        # RFC-3339 timestamp of the last quote applied to bid/ask, so consumers
        # can tell a live quote from an hours-old off-session snapshot.
        self.quote_ts: str | None = None
        self.previous_minute_high: float | None = None
        self.previous_minute_low: float | None = None
        self.day_volume: float | None = None
        self.volume_alert_triggered: bool = False
        self.volume_alert_ratio: float | None = None
        # Ring buffer of (monotonic_timestamp, price) for every trade tick in the
        # last ~minute. Used by last_price alerts to check any price in the window,
        # not only the single most-recent tick.
        self.recent_prices: deque = deque(maxlen=500)
        # Pending condition alerts for THIS symbol: each {symbol, field,
        # condition, value} watching a continuously-updated field (see
        # ALERTABLE_FIELDS). All symbols' alerts clear together once any fires.
        self.alerts: list[dict] = []
        # Armed conditional trade plan for this symbol (see agent_stonks.tactics)
        # and the background executor matching it against live data.
        self.tactics = None  # "Tactics | None"
        self.tactics_executor = None  # "TacticsExecutor | None"
        self.options_chain: "dict | None" = None
        self.options_wall_history: list[dict] = []
        self.options_status: str = ""

    # --- delegation to the shared AppState -------------------------------
    @property
    def api_key(self) -> str:
        return self.app.api_key

    @property
    def api_secret(self) -> str:
        return self.app.api_secret

    @property
    def feed(self) -> str:
        return self.app.feed

    @property
    def timeframe(self) -> str:
        return self.app.timeframe

    @property
    def decision_tracker(self) -> "DecisionTracker | None":
        return self.app.decision_tracker

    @property
    def agent_log(self) -> list[dict]:
        return self.app.agent_log

    @property
    def agent_wake_event(self) -> threading.Event:
        return self.app.agent_wake_event

    @property
    def agent_wake_reason(self) -> "str | None":
        return self.app.agent_wake_reason

    @agent_wake_reason.setter
    def agent_wake_reason(self, value: "str | None") -> None:
        self.app.agent_wake_reason = value

    @property
    def portfolio_value(self) -> "float | None":
        return self.app.portfolio_value

    @property
    def volume_alert_enabled(self) -> bool:
        return self.app.volume_alert_enabled

    @property
    def volume_alert_multiplier(self) -> float:
        return self.app.volume_alert_multiplier


_DEFAULTS: dict[str, object] = {
    "symbols": [],
    "symbol_states": {},
    "feed": "iex",
    "api_key": "",
    "api_secret": "",
    "status": "Idle",
    "news_status": "Idle",
    "bars_connected": False,
    "news_connected": False,
    "ws": None,
    "ws_news": None,
    "bars_fallback_stop_event": None,
    "news_fallback_stop_event": None,
    "timeframe": "1Min",
    "ma_periods": [],
    "show_fib": False,
    "show_7d_avg": False,
    "show_28d_avg": False,
    "show_1y_avg": False,
    "mixture_distribution": "none",
    "mixture_max_components": 0,
    "vwap_style": "hide",
    "show_candle_body": True,
    "show_percentile_body": False,
    "show_whiskers": True,
    "fill_gaps": True,
    "volume_alert_enabled": True,
    "volume_alert_multiplier": VOLUME_ALERT_DEFAULT_MULTIPLIER,
    "agent_log": [],
    "agent_running": False,
    "agent_stop_event": None,
    "decision_tracker": None,
    "starting_budget": PAPER_STARTING_CASH,
    "agent_start_time": None,
    "agent_equity_history": [],
    "portfolio_value": None,
    "agent_wake_event": None,  # handled specially
    "agent_wake_reason": None,
    "llm_provider": "openai",
    "llm_model": "",
    "llm_personality": "automatic",
    "automatic_active_strategy": None,
    "automatic_regime": None,
    "automatic_reason": None,
    "news_llm_provider": "openai",
    "scorecard": None,
}


class AppState:
    """Shared application state: the set of streamed symbols with one
    `SymbolState` each, plus everything that spans symbols -- credentials,
    the single WebSocket streams, chart settings, and the trading agent's
    ledger/wake plumbing (one agent trades the whole basket)."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.symbols: list[str] = []
        self.symbol_states: dict[str, SymbolState] = {}
        self.feed: str = "iex"
        self.api_key: str = ""
        self.api_secret: str = ""
        self.status: str = "Idle"
        self.news_status: str = "Idle"
        self.bars_connected: bool = False
        self.news_connected: bool = False
        self.ws: "websocket.WebSocketApp | None" = None
        self.ws_news: "websocket.WebSocketApp | None" = None
        self.bars_fallback_stop_event: "threading.Event | None" = None
        self.news_fallback_stop_event: "threading.Event | None" = None
        self.timeframe: str = "1Min"
        self.ma_periods: list[int] = []
        self.show_fib: bool = False
        self.show_7d_avg: bool = False
        self.show_28d_avg: bool = False
        self.show_1y_avg: bool = False
        self.mixture_distribution: str = "none"
        self.mixture_max_components: int = 0
        self.vwap_style: str = "hide"
        self.show_candle_body: bool = True
        self.show_percentile_body: bool = False
        self.show_whiskers: bool = True
        # Draw synthetic flat bars at feed minutes without any trade, so the
        # candle/volume series has no visual holes.
        self.fill_gaps: bool = True
        self.volume_alert_enabled: bool = True
        self.volume_alert_multiplier: float = VOLUME_ALERT_DEFAULT_MULTIPLIER
        self.agent_log: list[dict] = []
        self.agent_running: bool = False
        self.agent_stop_event: "threading.Event | None" = None
        self.decision_tracker: "DecisionTracker | None" = None
        self.starting_budget: float = PAPER_STARTING_CASH
        self.agent_start_time: "datetime | None" = None
        self.agent_equity_history: list[dict] = []
        self.portfolio_value: float | None = None
        self.agent_wake_event: threading.Event = threading.Event()
        self.agent_wake_reason: str | None = None
        self.llm_provider: str = "openai"
        self.llm_model: str = ""
        self.llm_personality: str = "automatic"
        # Automatic orchestrator: which strategy it has currently activated (None
        # when idle or assessing the regime), plus the regime read and reasoning
        # behind that choice. Surfaced in the UI/report.
        self.automatic_active_strategy: str | None = None
        self.automatic_regime: str | None = None
        self.automatic_reason: str | None = None
        self.news_llm_provider: str = "openai"
        # Per-session scoring collector (see agent_stonks.scoring); attached by
        # launch_agent/launch_automatic, flushed to the journal at session end.
        self.scorecard = None  # "scoring.Scorecard | None"

    def __getattr__(self, name: str) -> object:
        # Provide defaults for attributes missing on old cached session-state instances.
        if name not in _DEFAULTS:
            raise AttributeError(f"'AppState' object has no attribute '{name}'")
        if name == "lock":
            value: object = threading.Lock()
        elif name == "agent_wake_event":
            value = threading.Event()
        else:
            raw = _DEFAULTS[name]
            # Return a fresh copy for mutables so instances don't share state.
            value = type(raw)() if isinstance(raw, (list, dict, deque)) else raw
        object.__setattr__(self, name, value)
        return value

    # --- symbol management ------------------------------------------------
    @property
    def symbol(self) -> str:
        """Primary (first) symbol, for display fallbacks. '' when none set."""
        return self.symbols[0] if self.symbols else ""

    def set_symbols(self, symbols: Iterable[str]) -> None:
        """Replace the streamed symbol set, keeping existing SymbolStates for
        symbols that stay and creating fresh ones for new symbols."""
        ordered: list[str] = []
        for raw in symbols:
            sym = str(raw).strip().upper()
            if sym and sym not in ordered:
                ordered.append(sym)
        self.symbols = ordered
        self.symbol_states = {
            sym: self.symbol_states.get(sym) or SymbolState(sym, self) for sym in ordered
        }

    def sym(self, symbol: str) -> "SymbolState | None":
        return self.symbol_states.get(str(symbol).strip().upper())

    def iter_symbol_states(self) -> Iterator[SymbolState]:
        return iter(list(self.symbol_states.values()))

    # --- cross-symbol agent helpers ----------------------------------------
    def any_tactics(self) -> bool:
        return any(ss.tactics is not None for ss in self.iter_symbol_states())

    def iter_alerts(self) -> "list[tuple[SymbolState, dict]]":
        return [(ss, a) for ss in self.iter_symbol_states() for a in list(ss.alerts)]

    def clear_alerts(self) -> None:
        for ss in self.iter_symbol_states():
            ss.alerts = []

    def mark_price(self, symbol: str) -> "float | None":
        """Best available marking price for a symbol: live trade price, else the
        latest bar close, else the previous close."""
        ss = self.sym(symbol)
        if ss is None:
            return None
        with ss.lock:
            if ss.last_price is not None:
                return float(ss.last_price)
            if ss.bars:
                close = ss.bars[-1].get("c")
                if close is not None:
                    return float(close)
            return float(ss.prev_close) if ss.prev_close is not None else None

    def mark_to_market(self) -> "float | None":
        """Recompute and store the paper portfolio value: cash + every position
        marked to its symbol's best available price. None while no tracker runs."""
        tracker = self.decision_tracker
        if tracker is None:
            return None
        snap = tracker.snapshot()
        value = float(snap["cash"])
        for symbol, position in snap["positions"].items():
            if not position:
                continue
            price = self.mark_price(symbol)
            if price is None:
                continue
            value += position * price
        self.portfolio_value = value
        return value


def alert_field_value(state: "SymbolState", field: "str | None") -> "float | None":
    """Current numeric value of a continuously-updated alertable field for one
    symbol, or None if it isn't available yet (no data, or an input is unset)."""
    if field == "portfolio_value":
        value = state.app.portfolio_value
        return float(value) if value is not None else None
    if field == "spread":
        if state.ask_price is None or state.bid_price is None:
            return None
        return state.ask_price - state.bid_price
    if field == "volume_ratio":
        ratio, _ = current_volume_ratio(state.day_volume, state.daily_bars)
        return ratio
    if field not in ALERTABLE_FIELDS:
        return None
    value = getattr(state, field, None)
    return float(value) if value is not None else None


def compare(value: "float | None", condition: "str | None", target: "float | None") -> bool:
    """Whether `value` satisfies `condition` ('above' = >=, 'below' = <=) vs `target`."""
    if value is None or target is None:
        return False
    if condition == "above":
        return value >= target
    if condition == "below":
        return value <= target
    return False


_PRICE_WINDOW_SEC = 60


def alert_triggered(state: "SymbolState", alert: dict) -> bool:
    """Whether a generic condition alert's watched field currently meets its threshold.

    `alert` is shaped {"symbol": <ticker>, "field": <ALERTABLE_FIELDS key>,
    "condition": "above"|"below", "value": <threshold>} and is checked against
    the `SymbolState` it was attached to.

    For last_price, any trade price recorded within the last minute counts — not
    only the single most-recent tick. This prevents a fast wick from being missed
    when the alert check runs slightly after the price reverted.
    """
    field = alert.get("field")
    condition = alert.get("condition")
    threshold = alert.get("value")
    if field == "last_price":
        cutoff = time.monotonic() - _PRICE_WINDOW_SEC
        return any(
            compare(price, condition, threshold)
            for ts, price in state.recent_prices
            if ts >= cutoff
        )
    value = alert_field_value(state, field)
    return compare(value, condition, threshold)


def normalize_alert(
    raw: object,
    symbols: "list[str] | None" = None,
    default_symbol: "str | None" = None,
) -> "dict | None":
    """Validate one alert spec (e.g. from the LLM) and return a clean
    {symbol, field, condition, value} dict, or None if it isn't a valid,
    watchable alert. When `symbols` is given the alert's symbol must be one of
    them; a missing symbol falls back to `default_symbol` (the sole streamed
    ticker, typically)."""
    if not isinstance(raw, dict):
        return None
    field = raw.get("field")
    condition = raw.get("condition")
    if field not in ALERTABLE_FIELDS or condition not in ("above", "below"):
        return None
    try:
        value = float(raw.get("value"))
    except (TypeError, ValueError):
        return None
    symbol = str(raw.get("symbol") or default_symbol or "").strip().upper()
    if not symbol:
        return None
    if symbols is not None and symbol not in symbols:
        return None
    return {"symbol": symbol, "field": field, "condition": condition, "value": value}


def format_alert(alert: dict) -> str:
    """One-line human-readable description of a condition alert, e.g.
    'AAPL last_price above 150' or 'TSLA day_volume above 5,000,000'."""
    field = alert.get("field", "?")
    condition = alert.get("condition", "?")
    value = alert.get("value")
    if isinstance(value, (int, float)):
        value_str = f"{value:,.0f}" if abs(value) >= 1000 else f"{value:,.4f}".rstrip("0").rstrip(".")
    else:
        value_str = str(value)
    symbol = alert.get("symbol")
    prefix = f"{symbol} " if symbol else ""
    return f"{prefix}{field} {condition} {value_str}"


def format_tool_kv(data: dict, max_items: int = 8, max_value_len: int = 48) -> list[tuple[str, str]]:
    """Flatten a tool call's args/result dict into (key, value) pairs for log
    display. Truncates each value individually rather than the whole blob, so
    long results never get cut off mid-token."""
    if not isinstance(data, dict):
        return [("", _format_tool_value(data, max_value_len))]
    items = list(data.items())
    rows = [(str(k), _format_tool_value(v, max_value_len)) for k, v in items[:max_items]]
    if len(items) > max_items:
        rows.append(("…", f"+{len(items) - max_items} more"))
    return rows


def _format_tool_value(value: object, max_len: int) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        s = "true" if value else "false"
    elif isinstance(value, float):
        s = f"{value:,.4g}"
    elif isinstance(value, (list, dict)):
        s = json.dumps(value, separators=(",", ":"))
    else:
        s = str(value)
    return s if len(s) <= max_len else s[: max_len - 1] + "…"


def _daily_bar_date(bar: dict) -> str:
    """Trading-day date (YYYY-MM-DD) of an Alpaca daily bar."""
    return str(bar.get("t", ""))[:10]


def _today_iso(today: "str | None" = None) -> str:
    return today or datetime.now(timezone.utc).strftime("%Y-%m-%d")


def completed_daily_bars(daily_bars: list[dict], today: "str | None" = None) -> list[dict]:
    """Daily bars strictly before today -- excludes today's still-forming bar."""
    cutoff = _today_iso(today)
    return [b for b in daily_bars if _daily_bar_date(b) and _daily_bar_date(b) < cutoff]


def today_daily_volume(daily_bars: list[dict], today: "str | None" = None) -> float:
    """Volume already printed on today's (partial) daily bar, or 0 if none yet.

    Used to seed today's running volume when the stream starts mid-session, so
    the alert sees the full day's accumulation rather than only bars that arrive
    after connection.
    """
    cutoff = _today_iso(today)
    for bar in reversed(daily_bars):
        if _daily_bar_date(bar) == cutoff:
            return float(bar.get("v") or 0.0)
    return 0.0


def today_daily_bar(daily_bars: list[dict], today: "str | None" = None) -> "dict | None":
    """Today's still-forming daily bar, or None if the latest daily bar isn't today's
    (e.g. pre-open/weekend, or a lagging feed that hasn't published today's bar yet).
    """
    if not daily_bars:
        return None
    bar = daily_bars[-1]
    return bar if _daily_bar_date(bar) == _today_iso(today) else None


def average_daily_volume(
    daily_bars: list[dict],
    window: int = VOLUME_ADV_WINDOW,
    min_days: int = VOLUME_ADV_MIN_DAYS,
    today: "str | None" = None,
) -> "float | None":
    """Baseline daily volume for the high-volume alert.

    Normally the mean of the last `window` completed daily volumes. When fewer
    than `min_days` completed days are available (thin history / early session),
    fall back to yesterday's single-day volume. Returns None when no completed
    day with volume is available at all.
    """
    vols = [
        float(bar.get("v") or 0.0)
        for bar in completed_daily_bars(daily_bars, today)
    ]
    vols = [v for v in vols if v > 0]
    if not vols:
        return None
    if len(vols) < min_days:
        return vols[-1]  # yesterday's volume
    recent = vols[-window:]
    return sum(recent) / len(recent)


def current_volume_ratio(
    day_volume: "float | None",
    daily_bars: list[dict],
    today: "str | None" = None,
) -> "tuple[float | None, float | None]":
    """(ratio, baseline) of today's cumulative volume vs the ADV baseline.

    `ratio` is None when there isn't enough data (no baseline, or no volume
    accumulated yet) to compute it.
    """
    baseline = average_daily_volume(daily_bars, today=today)
    if not baseline or day_volume is None:
        return None, baseline
    return day_volume / baseline, baseline
