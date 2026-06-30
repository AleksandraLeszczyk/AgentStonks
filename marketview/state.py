import threading
from collections import deque
from datetime import datetime, timezone
from typing import TYPE_CHECKING

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

_DEFAULTS: dict[str, object] = {
    "bars": None,  # handled specially
    "daily_bars": [],
    "trades": [],
    "news": [],
    "news_impacts": {},
    "symbol": "",
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
    "last_price": None,
    "prev_close": None,
    "bid_price": None,
    "bid_size": None,
    "ask_price": None,
    "ask_size": None,
    "previous_minute_high": None,
    "previous_minute_low": None,
    "day_volume": None,
    "volume_alert_enabled": True,
    "volume_alert_multiplier": VOLUME_ALERT_DEFAULT_MULTIPLIER,
    "volume_alert_triggered": False,
    "volume_alert_ratio": None,
    "agent_log": [],
    "agent_running": False,
    "agent_stop_event": None,
    "decision_tracker": None,
    "starting_budget": PAPER_STARTING_CASH,
    "agent_start_time": None,
    "agent_equity_history": [],
    "alerts": [],
    "portfolio_value": None,
    "agent_wake_event": None,  # handled specially
    "agent_wake_reason": None,
    "llm_provider": "gemini",
    "llm_model": "",
    "llm_personality": "swing",
    "automatic_active_strategy": None,
    "automatic_regime": None,
    "automatic_reason": None,
    "news_llm_provider": "gemini",
    "options_chain": None,
    "options_wall_history": [],
    "options_status": "",
}


# Continuously-updated state fields the agent can attach a condition alert to.
# Every entry is refreshed on the live price/quote stream (and the REST
# fallback), so an alert on any of them can fire between scheduled cycles. The
# value is a human description used in tool schemas and the UI. `spread` and
# `volume_ratio` are derived (see `alert_field_value`) rather than stored
# directly, but update just as continuously as their inputs.
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
    "volume_ratio": "Today's cumulative volume divided by average daily volume",
    "portfolio_value": "Paper portfolio value (cash + position marked to last price)",
}

# Subset of alertable fields that live on the price axis, so a triggered/pending
# alert on them can be drawn as a horizontal line on the price chart.
PRICE_AXIS_ALERT_FIELDS: frozenset[str] = frozenset(
    {"last_price", "bid_price", "ask_price", "previous_minute_high", "previous_minute_low"}
)


def alert_field_value(state: "AppState", field: "str | None") -> "float | None":
    """Current numeric value of a continuously-updated alertable field, or None
    if it isn't available yet (no data, or an input field is unset)."""
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


def alert_triggered(state: "AppState", alert: dict) -> bool:
    """Whether a generic condition alert's watched field currently meets its threshold.

    `alert` is shaped {"field": <ALERTABLE_FIELDS key>, "condition": "above"|"below",
    "value": <threshold>}.
    """
    value = alert_field_value(state, alert.get("field"))
    return compare(value, alert.get("condition"), alert.get("value"))


def normalize_alert(raw: object) -> "dict | None":
    """Validate one alert spec (e.g. from the LLM) and return a clean
    {field, condition, value} dict, or None if it isn't a valid, watchable alert."""
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
    return {"field": field, "condition": condition, "value": value}


def format_alert(alert: dict) -> str:
    """One-line human-readable description of a condition alert, e.g.
    'last_price above 150' or 'day_volume above 5,000,000'."""
    field = alert.get("field", "?")
    condition = alert.get("condition", "?")
    value = alert.get("value")
    if isinstance(value, (int, float)):
        value_str = f"{value:,.0f}" if abs(value) >= 1000 else f"{value:,.4f}".rstrip("0").rstrip(".")
    else:
        value_str = str(value)
    return f"{field} {condition} {value_str}"


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


class AppState:
    def __init__(self) -> None:
        self.bars: deque[dict] = deque(maxlen=MAX_BARS)
        self.daily_bars: list[dict] = []
        self.trades: list[dict] = []
        self.news: list[dict] = []
        self.news_impacts: dict[str, str] = {}
        self.symbol: str = ""
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
        self.lock = threading.Lock()
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
        self.last_price: float | None = None
        self.prev_close: float | None = None
        self.bid_price: float | None = None
        self.bid_size: float | None = None
        self.ask_price: float | None = None
        self.ask_size: float | None = None
        self.previous_minute_high: float | None = None
        self.previous_minute_low: float | None = None
        self.day_volume: float | None = None
        self.volume_alert_enabled: bool = True
        self.volume_alert_multiplier: float = VOLUME_ALERT_DEFAULT_MULTIPLIER
        self.volume_alert_triggered: bool = False
        self.volume_alert_ratio: float | None = None
        self.agent_log: list[dict] = []
        self.agent_running: bool = False
        self.agent_stop_event: "threading.Event | None" = None
        self.decision_tracker: "DecisionTracker | None" = None
        self.starting_budget: float = PAPER_STARTING_CASH
        self.agent_start_time: "datetime | None" = None
        self.agent_equity_history: list[dict] = []
        # Pending condition alerts: each {field, condition, value} watching a
        # continuously-updated state field (see ALERTABLE_FIELDS). Cleared once
        # any one fires.
        self.alerts: list[dict] = []
        self.portfolio_value: float | None = None
        self.agent_wake_event: threading.Event = threading.Event()
        self.agent_wake_reason: str | None = None
        self.llm_provider: str = "gemini"
        self.llm_model: str = ""
        self.llm_personality: str = "swing"
        # Automatic orchestrator: which strategy it has currently activated (None
        # when idle or assessing the regime), plus the regime read and reasoning
        # behind that choice. Surfaced in the UI/report.
        self.automatic_active_strategy: str | None = None
        self.automatic_regime: str | None = None
        self.automatic_reason: str | None = None
        self.news_llm_provider: str = "gemini"
        self.options_chain: "dict | None" = None
        self.options_wall_history: list[dict] = []
        self.options_status: str = ""

    def __getattr__(self, name: str) -> object:
        # Provide defaults for attributes missing on old cached session-state instances.
        if name not in _DEFAULTS:
            raise AttributeError(f"'AppState' object has no attribute '{name}'")
        if name == "bars":
            value: object = deque(maxlen=MAX_BARS)
        elif name == "lock":
            value = threading.Lock()
        elif name == "agent_wake_event":
            value = threading.Event()
        else:
            raw = _DEFAULTS[name]
            # Return a fresh copy for mutables so instances don't share state.
            value = type(raw)() if isinstance(raw, (list, dict, deque)) else raw
        object.__setattr__(self, name, value)
        return value
