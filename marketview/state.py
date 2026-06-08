import threading
from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import websocket

from .config import MAX_BARS


class AppState:
    def __init__(self) -> None:
        self.bars: deque[dict] = deque(maxlen=MAX_BARS)
        self.daily_bars: list[dict] = []
        self.trades: list[dict] = []
        self.news: list[dict] = []
        self.symbol: str = ""
        self.feed: str = "iex"
        self.status: str = "Idle"
        self.ws: "websocket.WebSocketApp | None" = None
        self.ws_news: "websocket.WebSocketApp | None" = None
        self.lock = threading.Lock()
        self.ma_periods: list[int] = []
        self.show_fib: bool = False
        self.show_7d_avg: bool = False
        self.show_28d_avg: bool = False
        self.show_1y_avg: bool = False
        self.gaussian_max_components: int = 0
        self.show_gaussian_centers: bool = False
