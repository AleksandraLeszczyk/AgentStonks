from datetime import datetime, timezone

DATA_REST = "https://data.alpaca.markets"
BARS_STREAM_URL = "wss://stream.data.alpaca.markets/v2/{feed}"
NEWS_STREAM_URL = "wss://stream.data.alpaca.markets/v1beta1/news"
MAX_BARS = 300
POLL_SEC = 3
CHART_POLL_SEC = 30

# REST-polling fallback for bars/trades and news, used only while the
# corresponding WebSocket stream is not connected (e.g. Alpaca's
# "connection limit exceeded" rejecting a second concurrent stream on the
# same API key/feed). REST calls aren't subject to that per-key streaming
# connection cap, so they keep working even while the socket is stuck.
FALLBACK_POLL_SEC = 15
NEWS_FALLBACK_POLL_SEC = 60

# Periodic REST backfill that repairs holes in the live bar series while the
# WebSocket IS connected: the stream never re-delivers bars that closed during
# a reconnect, and thin symbols get no bar at all for minutes without a trade
# on the subscribed feed.
BACKFILL_POLL_SEC = 60
OPTIONS_POLL_SEC = 60
OPTIONS_WALL_HISTORY_MAXLEN = 200
TIMEFRAMES = ["1Min", "5Min", "15Min", "30Min", "1Hour", "1Day"]
FEEDS = ["iex", "sip"]

# High-volume alert: trigger when today's cumulative volume exceeds
# VOLUME_ALERT_DEFAULT_MULTIPLIER x the average daily volume. The baseline is
# the mean of the last VOLUME_ADV_WINDOW completed daily volumes; with fewer
# than VOLUME_ADV_MIN_DAYS completed days (thin history / early session), it
# falls back to yesterday's single-day volume.
VOLUME_ALERT_DEFAULT_MULTIPLIER = 1.5
VOLUME_ADV_WINDOW = 20
VOLUME_ADV_MIN_DAYS = 5

# Quote reliability thresholds for get_quote. The IEX feed reports IEX's own
# top-of-book, not the consolidated NBBO: outside regular hours or when IEX's
# book is empty near the touch, the "latest quote" is a placeholder-wide
# two-sided quote (e.g. ±5% around the mid, 100x100) or an hours-old snapshot.
# Quotes wider than QUOTE_WIDE_SPREAD_PCT percent of the mid, or older than
# QUOTE_STALE_SEC, get a warning attached so the agent doesn't treat them as
# executable prices.
QUOTE_WIDE_SPREAD_PCT = 1.0
QUOTE_STALE_SEC = 120.0

# Trading agent
AGENT_CYCLE_SEC = 60
AGENT_LOG_POLL_SEC = 4
AGENT_PERFORMANCE_POLL_SEC = 60
AGENT_EQUITY_HISTORY_MAXLEN = 5000
AGENT_MAX_TOOL_ITERS = 8
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

# Dot / marker colors per LLM-estimated news impact label, shared by the
# News tab badges and the Live chart news markers.
NEWS_IMPACT_COLORS: dict[str, str] = {
    "positive": "#26c6a2",
    "negative": "#ef5350",
    "neutral":  "#888",
    "small":    "#fb923c",
    "unknown":  "#555",
}

# News dots on the Live chart sit above the high of the minute bar containing
# the article's timestamp, offset by this fraction of the session's price range
# so the spacing looks right at any price scale.
NEWS_MARKER_OFFSET_FRAC = 0.04

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
