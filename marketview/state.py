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
    "day_high": None,
    "day_low": None,
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
    "price_alerts": [],
    "portfolio_value": None,
    "agent_wake_event": None,  # handled specially
    "agent_wake_reason": None,
    "llm_provider": "gemini",
    "llm_model": "",
    "llm_personality": "swing",
    "news_llm_provider": "gemini",
    "options_chain": None,
    "options_wall_history": [],
    "options_status": "",
}


def alert_triggered(price: float, alert: dict) -> bool:
    target = alert.get("price")
    condition = alert.get("condition")
    if target is None:
        return False
    if condition == "above":
        return price >= target
    if condition == "below":
        return price <= target
    return False


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
        self.day_high: float | None = None
        self.day_low: float | None = None
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
        self.price_alerts: list[dict] = []
        self.portfolio_value: float | None = None
        self.agent_wake_event: threading.Event = threading.Event()
        self.agent_wake_reason: str | None = None
        self.llm_provider: str = "gemini"
        self.llm_model: str = ""
        self.llm_personality: str = "swing"
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
