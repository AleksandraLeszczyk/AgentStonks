import threading
from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import websocket

from .config import MAX_BARS


class AppState:
    def __init__(self) -> None:
        self.bars: deque[dict] = deque(maxlen=MAX_BARS)
        self.trades: list[dict] = []
        self.news: list[dict] = []
        self.symbol: str = ""
        self.feed: str = "iex"
        self.status: str = "Idle"
        self.ws: "websocket.WebSocketApp | None" = None
        self.ws_news: "websocket.WebSocketApp | None" = None
        self.lock = threading.Lock()
