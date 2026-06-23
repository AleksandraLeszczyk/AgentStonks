import threading
from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime

    import websocket

    from .decisions import DecisionTracker

from .config import MAX_BARS, PAPER_STARTING_CASH

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
    "ws": None,
    "ws_news": None,
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
        self.ws: "websocket.WebSocketApp | None" = None
        self.ws_news: "websocket.WebSocketApp | None" = None
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
