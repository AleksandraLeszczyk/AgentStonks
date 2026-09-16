from datetime import datetime, timezone

DATA_REST = "https://data.alpaca.markets"
BARS_STREAM_URL = "wss://stream.data.alpaca.markets/v2/{feed}"
NEWS_STREAM_URL = "wss://stream.data.alpaca.markets/v1beta1/news"

# Finnhub's real-time socket. It authenticates in the query string rather than
# with an auth frame, and for US equities it serves the trade tape only -- no
# bars and no book. Candles are aggregated from those trades locally
# (agent_stonks.finnhub_stream), which is why it can be the default source
# despite carrying less than Alpaca's stream does: what it *does* carry is the
# consolidated tape rather than one venue's, so the price and volume it prints
# are the market's rather than IEX's ~2% slice of it, and it keeps working
# outside the hours a free Alpaca key can stream.
FINNHUB_STREAM_URL = "wss://ws.finnhub.io?token={token}"

# Live bar/trade sources, best first. Alpaca stays available as the fallback
# choice because it is the only one of the two that also streams quotes, and
# because its bar volumes match the REST bars the buffer is seeded and
# backfilled with.
DATA_SOURCES = ["finnhub", "alpaca"]
DEFAULT_DATA_SOURCE = "finnhub"

# Where REST bars come from: the initial history load, the timeframe reload, the
# periodic backfill and the stream-down fallback poll all fill the same buffer
# the live socket is filling, so they share one setting (see
# agent_stonks.bar_history for the measurements behind it).
#
# IEX carries under 4% of consolidated volume -- 1.56M vs 41.6M shares over the
# same 390 AAPL minutes -- so pairing IEX history with a consolidated live stream
# puts a ~26x volume step in the middle of the series that every volume-derived
# read then sums across. "auto" therefore means the consolidated tape wherever
# one is reachable: Alpaca SIP if the key is subscribed, else yfinance (within
# 1.5% of SIP, free, but ~15 minutes delayed and only ~7 days of minute
# history), and IEX only when neither can answer.
HISTORY_FEEDS = ["auto", "sip", "yfinance", "iex"]
DEFAULT_HISTORY_FEED = "auto"

# Alpaca's free/basic plans grant SIP only outside a trailing 15-minute window:
# a request whose `end` reaches into it is refused with 403 "subscription does
# not permit querying recent SIP data", while the same request ending 16 minutes
# back returns the full consolidated tape. That is a real entitlement worth
# using -- the backfill repairs *holes*, and a hole 16 minutes old still needs
# filling -- so a key like that is served SIP with the window held back by this
# many minutes rather than being demoted to yfinance. The live socket owns the
# recent window either way.
SIP_DELAY_MIN = 16

# How often the Finnhub aggregator checks whether the bar it is filling has run
# past its bucket. A bar must close on the clock rather than on the next trade,
# or a symbol that stops printing freezes `previous_minute_close` -- the field
# every alert and rule trader reads -- for as long as the quiet lasts. Small
# enough that a closed minute is published within a couple of seconds of ending,
# which is well inside APPLE_TRADER_BAR_LAG_SEC.
FINNHUB_BAR_FLUSH_SEC = 2.0

# Alpaca's Trading API (orders, positions, account) -- a different host from the
# market-data API above, and a different key pair per venue. See
# agent_stonks.trading_rest.
TRADING_REST_PAPER = "https://paper-api.alpaca.markets"
TRADING_REST_LIVE = "https://api.alpaca.markets"

# Where a filled decision actually goes:
#   "local"         the in-memory ledger this app has always kept. Nothing
#                   leaves the process. Still the only mode SimLab can use.
#   "alpaca_paper"  real orders against Alpaca's paper account -- real routing,
#                   real fills, real rejections, fake money.
#   "alpaca_live"   real orders against the live account. Real money.
#
# The default is the paper account rather than the local ledger: the point of
# routing orders at all is to find out what the broker does with them -- partial
# fills, rejections, buying-power limits, queued-until-open -- and none of that
# shows up in a ledger that always says yes.
TRADING_MODES = ["local", "alpaca_paper", "alpaca_live"]
DEFAULT_TRADING_MODE = "alpaca_paper"

# Live trading is off unless this environment variable is truthy, and the UI
# asks for a typed confirmation on top of it. Two independent gates, because
# the failure mode here is not a crash or a bad chart -- it is real money moved
# by an automated agent that the user did not intend to have running.
LIVE_TRADING_ENV_FLAG = "ALPACA_ENABLE_LIVE_TRADING"
LIVE_TRADING_CONFIRM_PHRASE = "TRADE LIVE"

# A market order is accepted immediately and fills asynchronously. The tracker
# waits this long for a terminal state before recording whatever filled so far;
# anything still working is left with the broker rather than cancelled, since
# cancelling a partially-filled order is a trading decision, not a timeout.
ORDER_FILL_TIMEOUT_SEC = 20.0
ORDER_POLL_SEC = 0.5

# Audible cue when an order fills (see agent_stonks.trade_sound). A raw Web
# Audio gain multiplier, not decibels: loud enough to hear from across a room,
# quiet enough not to startle on the twentieth fill of a session.
TRADE_SOUND_VOLUME = 0.22

# Full regular session is 390 one-minute bars; 420 keeps the 09:30 ET open in
# the buffer through the close (plus a little premarket) so session-anchored
# reads (opening range, VWAP) never silently lose their anchor mid-afternoon.
MAX_BARS = 420
POLL_SEC = 3
CHART_POLL_SEC = 30
# How often the Pre-Market tab re-reads the briefing the stream start kicked
# off. Briefing a basket is several seconds of LLM time per symbol and results
# land one symbol at a time, so this only has to be fast enough that a finished
# symbol appears promptly.
PREMARKET_POLL_SEC = 3

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

# Daily agent-accuracy scoring (see agent_stonks.scoring): a scoring session
# runs at most once per UTC day, and only after the day has accumulated at
# least this much total agent runtime -- short experiments alone never score.
SCORING_MIN_TOTAL_RUNTIME_SEC = 3600

# Premarket analyst: it may start its single opening-tactics cycle no earlier
# than PREMARKET_LEAD_SEC before the opening bell; while holding for that
# window it re-checks the clock every PREMARKET_WAIT_POLL_SEC.
PREMARKET_LEAD_SEC = 120
PREMARKET_WAIT_POLL_SEC = 30.0

# Apple Trader: the rule-based (non-LLM) loop that trades its configured symbol
# off a saved model's forecast (see agent_stonks.apple_trader).
#
# MODEL names the model it runs on -- a key of agent_stonks.apple_models.MODELS,
# of which "dayrange" is the only one.
#
# The day-range model forecasts where the whole session's high and low will land, once, at 9:35,
# and the rules built on it are TimeToChange3 notebook 05's: two resting levels
# a fixed number of average daily ranges below the predicted high H, with A the
# 14-day average daily range in dollars.
#
#     buy_level  = H - BUY_K  * A
#     sell_level = H - SELL_K * A
#
# The notebook specifies 0.75 and 0.10 and only ever swept them over five
# sessions. The levels used here are per instrument instead, because the stocks
# do not dip alike: notebook 05's own grid (buy 0.30..1.30, sell 0.05..0.50,
# step 0.05) re-run with its own fill rules over *every* session each ticker has
# minute bars and a forecast for. `TimeToChange3/scripts/sweep_levels.py`
# reproduces the table. A cell is eligible only if it trades on at least half
# the sessions -- otherwise the deepest buy distances "win" on a handful of
# fills -- and the pick is the eligible cell with the best 3x3 neighbourhood
# mean: the middle of a profitable plateau rather than its sharpest cell.
#
#   ticker  days  buy   sell  total  traded  up  1st half  2nd half  at 0.75/0.10
#   AAPL     36   0.40  0.25  +1230    34    22    +1001     +229       +1077
#   GOOGL    31   0.65  0.05   +534    23    14     +244     +291        +146
#   INTC     31   0.50  0.05   +403    28    15    -1444    +1847        -258
#
# Dollars on $10,000 per session, limit fills, no costs, and picked on the same
# sessions it is scored on. How far to trust each differs: AAPL's whole grid is
# profitable and the pick sits on a broad plateau; GOOGL's holds up in both
# halves; INTC's flips sign between halves, so it is the best cell of a surface
# that is mostly noise. GOOGL and INTC both pick the grid's smallest sell
# distance, so their optimum may lie past the edge that was searched. ORCL was
# swept too (it is not wired up) and no cell makes money.
#
# The live ledger fills at market rather than at the level (see
# `DayRangeTrader`), so expect less than these totals. A symbol with no entry
# falls back to the notebook's 0.75 / 0.10, which is also what a SimLab record
# written without the two fields replays at.
#
APPLE_TRADER_MODEL = "dayrange"
APPLE_TRADER_BUY_K = 0.75
APPLE_TRADER_SELL_K = 0.10
# (buy_k, sell_k) per instrument -- see the day-range block above.
APPLE_TRADER_DAYRANGE_LEVELS: "dict[str, tuple[float, float]]" = {
    "AAPL": (0.40, 0.25),
    "GOOGL": (0.65, 0.05),
    "INTC": (0.50, 0.05),
}
# The day-range managed exit, on top of the sell level and the closing flatten
# (`DayRangeTrader._exit`). Unlike the levels these were never swept: they are
# specified starting points, and none of them is in the notebook's numbers.
#   stop       sell everything once a bar's low is STOP_K x ADR under the fill,
#              and take no new entry for the rest of the session
#   take       once the momentum score has fallen MOMENTUM_DROP sigmas from its
#              best since the entry with the position in profit, sell
#              TAKE_FRACTION of it ...
#   runner     ... and keep the rest for the sell level only if that is still
#              HOLD_MIN_GAIN_K x ADR above the fill (otherwise sell it all);
#              a runner is sold if the price comes back to the fill
APPLE_TRADER_STOP_K = 0.20
APPLE_TRADER_MOMENTUM_DROP = 1.0
APPLE_TRADER_TAKE_FRACTION = 0.70
APPLE_TRADER_HOLD_MIN_GAIN_K = 0.30
# What the agent does when the session trades through the forecast it was given
# at 9:35 -- the vocabulary lives here rather than in `dayrange_model`, which
# defines the arithmetic (`updated_range`) but costs 200 MB of torch to import,
# and both the form and the config dataclass need to name a policy without
# paying that. Same reason `TRADING_MODES` is a list here.
#
#   "off"       the 9:35 forecast stands all session, whatever the tape prints
#   "extreme"   a breached side moves to the session's own high (or low)
#   "brownian"  ... and then past it by what a driftless walk with ADR-implied
#               volatility is still expected to add, ADR x sqrt(session left)/2
#
# The forecast already refuses to sit under the opening window's high
# (`dayrange_model.apply_open_constraint`); the latter two carry that same
# correction through the rest of the day, and `DayRangeTrader` rebuilds its two
# levels from the updated high each time it moves.
BREACH_OFF = "off"
BREACH_EXTREME = "extreme"
BREACH_BROWNIAN = "brownian"
BREACH_POLICIES = (BREACH_OFF, BREACH_EXTREME, BREACH_BROWNIAN)
BREACH_LABELS = {
    BREACH_OFF: "Hold the 9:35 forecast",
    BREACH_EXTREME: "Move to the extreme so far",
    BREACH_BROWNIAN: "Brownian extension, volatility implied by ADR",
}
# The default is "extreme" because it is the weaker claim of the two: "the day's
# high is at least what has already traded" is arithmetic, not a forecast.
# Neither policy has been swept, and APPLE_TRADER_DAYRANGE_LEVELS above was --
# under "off", with the forecast held fixed all day -- so the two numbers there
# were picked against a rule this setting changes. A SimLab record written
# before the setting existed replays under "off"
# (`simlab.rule_agents._APPLE_LEGACY`), so no stored result moves.
APPLE_TRADER_BREACH_UPDATE = BREACH_EXTREME

# What the two levels hang off -- the reference `buy_k` and `sell_k` are
# measured below (`apple_trader.DayRangeTrader._set_levels`). Here for the same
# reason the breach policies are: naming one must not cost an import of torch.
#
#   "dayrange"  TimeToChange3's predicted high, one number for the session.
#   "intraday"  the upper curve of that forecast stretched by IntradayVolatility's
#               time-of-day shape -- the "predicted intraday range x day range"
#               overlay, as a level rather than as a decoration. It is the same
#               forecast, re-read minute by minute: at the 09:30 peak it IS the
#               predicted high, by midday it has pulled in to roughly a fifth of
#               the distance from the open, and it opens back up into the close.
#
# The second needs an IntradayVolatility export for the symbol on top of the
# day-range bundle (`intraday_vol_model`), which `apple_trader.config_error`
# checks before a run starts.
LEVELS_DAYRANGE = "dayrange"
LEVELS_INTRADAY = "intraday"
LEVEL_SOURCES = (LEVELS_DAYRANGE, LEVELS_INTRADAY)
LEVEL_SOURCE_LABELS = {
    LEVELS_DAYRANGE: "Predicted high (flat all session)",
    LEVELS_INTRADAY: "Predicted range × intraday volatility",
}
# Unchanged from the notebook, unlike APPLE_TRADER_BREACH_UPDATE above: the
# intraday shape is a second model's claim rather than arithmetic on the tape,
# it is not available for every symbol, and it changes what the strategy *is*
# (a level that moves with the clock, and can therefore walk a target down
# towards an open position). Opt in and measure it in SimLab.
APPLE_TRADER_LEVEL_SOURCE = LEVELS_DAYRANGE
# The session circuit breaker: once a trade has closed for no more than this
# many ADRs of profit per share, the agent buys nothing else that day
# (`apple_trader.DayRangeTrader._close_out`). The reasoning is that a round trip
# which barely paid is evidence the setup was not there today, and re-arming the
# same levels on the same tape is how one weak trade becomes five.
#
# Read it against the levels above before trusting the default. The most a
# target exit can net is (buy_k - sell_k) x ADR -- 0.15 on AAPL's swept pair,
# 0.60 on GOOGL's, 0.45 on INTC's -- so at 0.20 an AAPL run stands down after
# its first completed trade however well it went, while GOOGL and INTC stand
# down only on an exit worse than the target. That is a real strategy choice
# ("one trade a day unless it runs"), not a bug, and the form says so where the
# two settings disagree. 0 switches it off, which is what every record written
# before it existed replays as.
APPLE_TRADER_MIN_WIN_K = 0.20
APPLE_TRADER_CYCLE_SEC = 60
APPLE_TRADER_POSITION_PCT = 95.0
# Flatten this many minutes before the close: the day-range forecast is a
# statement about one session, and the momentum regime does not survive the
# overnight gap either.
APPLE_TRADER_FLATTEN_BEFORE_CLOSE_MIN = 5
# Seconds after a minute boundary to score, giving the stream time to deliver
# the bar that just closed.
APPLE_TRADER_BAR_LAG_SEC = 5.0

# Tactics executor: the stream nudges it on every tick, so this poll is only a
# fallback cadence covering the non-stream condition fields (vix, momentum) and
# REST-fallback sessions. The momentum window is the lookback (in minutes) for
# the `momentum_pct` tactic condition field.
TACTICS_POLL_SEC = 2.0
TACTICS_MOMENTUM_WINDOW_MIN = 10

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

# The day boundary on a chart covering more than one session: a rule at 09:30
# and 16:00. Muted on purpose -- it is the frame the price is read inside, not
# a signal competing with the candles.
SESSION_MARKER_COLOR = "#5b6478"

# Model-prediction overlays on the price chart (see model_overlays.py). One
# color per overlay, so a level, its band and its label are recognisably the
# same prediction.
MODEL_OVERLAY_COLORS: dict[str, str] = {
    "day_range":     "#22d3ee",  # cyan, as the ML predicted profile curve
    "profile_range": "#a78bfa",  # violet
    "profile_poc":   "#f472b6",  # pink -- one number inside the violet band
    # The two time-of-day envelopes: yellow for IntradayVolatility alone, teal
    # for its shape stretched to the day-range forecast (a sibling of that cyan).
    "intraday_range":    "#facc15",
    "intraday_dayrange": "#2dd4bf",
}

# Alpha for the semi-transparent backgrounds overlays paint behind the candles.
# A predicted band covers most of the plot, so it has to read as a tint rather
# than a fill; a predicted window is narrow and can afford a little more.
MODEL_OVERLAY_BAND_ALPHA = 0.08
MODEL_OVERLAY_SPAN_ALPHA = 0.14

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
