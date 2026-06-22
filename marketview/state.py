import threading
from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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
    "gaussian_max_components": 0,
    "show_gaussian_centers": False,
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
    "price_alerts": [],
    "llm_provider": "gemini",
    "llm_model": "",
}


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
        self.gaussian_max_components: int = 0
        self.show_gaussian_centers: bool = False
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
        self.price_alerts: list[dict] = []
        self.llm_provider: str = "gemini"
        self.llm_model: str = ""

    def __getattr__(self, name: str) -> object:
        # Provide defaults for attributes missing on old cached session-state instances.
        if name not in _DEFAULTS:
            raise AttributeError(f"'AppState' object has no attribute '{name}'")
        if name == "bars":
            value: object = deque(maxlen=MAX_BARS)
        elif name == "lock":
            value = threading.Lock()
        else:
            raw = _DEFAULTS[name]
            # Return a fresh copy for mutables so instances don't share state.
            value = type(raw)() if isinstance(raw, (list, dict, deque)) else raw
        object.__setattr__(self, name, value)
        return value
