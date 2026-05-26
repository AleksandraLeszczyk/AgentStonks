from datetime import datetime, timezone

DATA_REST = "https://data.alpaca.markets"
BARS_STREAM_URL = "wss://stream.data.alpaca.markets/v2/{feed}"
NEWS_STREAM_URL = "wss://stream.data.alpaca.markets/v1beta1/news"
MAX_BARS = 300
POLL_SEC = 3
TIMEFRAMES = ["1Min", "5Min", "15Min", "30Min", "1Hour", "1Day"]
FEEDS = ["iex", "sip"]

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

CUSTOM_CSS = f"""
body, .gradio-container {{ background: {PALETTE['bg']} !important; }}
.gr-button-primary {{ background: {PALETTE['accent']} !important; border:none !important; }}
.gr-button {{ border-radius: 6px !important; }}
footer {{ display: none !important; }}
label {{ color: {PALETTE['muted']} !important; font-size:12px !important; }}
"""
