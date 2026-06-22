from datetime import datetime, timezone

DATA_REST = "https://data.alpaca.markets"
BARS_STREAM_URL = "wss://stream.data.alpaca.markets/v2/{feed}"
NEWS_STREAM_URL = "wss://stream.data.alpaca.markets/v1beta1/news"
MAX_BARS = 300
POLL_SEC = 3
CHART_POLL_SEC = 30
TIMEFRAMES = ["1Min", "5Min", "15Min", "30Min", "1Hour", "1Day"]
FEEDS = ["iex", "sip"]

# Trading agent
AGENT_CYCLE_SEC = 60
AGENT_LOG_POLL_SEC = 4
AGENT_MAX_TOOL_ITERS = 8
AGENT_ALERT_POLL_SEC = 2
PAPER_STARTING_CASH = 100_000.0
TRADE_FIXED_COST = 1.15

# 13:20 UTC = 09:20 ET, just before market open (09:30 ET)
SESSION_START = datetime.now(tz=timezone.utc).replace(
    hour=13, minute=20, second=0, microsecond=0
)

PALETTE: dict[str, str] = {
    "bg": "#0f1117",
    "panel": "#1a1d27",
    "grid": "#2a2d3a",
    "up": "#26c6a2",
    "down": "#ef5350",
    "text": "#e0e0e0",
    "muted": "#888",
    "accent": "#60a5fa",
    "orange": "#fb923c",
}

MA_COLORS: dict[int, str] = {
    5:  "#60a5fa",  # blue
    15: "#fb923c",  # orange
    60: "#a78bfa",  # violet
}

AVG_LINE_COLORS: dict[str, str] = {
    "7d":  "#34d399",  # green
    "28d": "#fbbf24",  # amber
    "1y":  "#f472b6",  # pink
}

FIB_LEVELS: list[tuple[float, str]] = [
    (0.0,   "0%"),
    (0.236, "23.6%"),
    (0.382, "38.2%"),
    (0.5,   "50%"),
    (0.618, "61.8%"),
    (0.786, "78.6%"),
    (1.0,   "100%"),
]
