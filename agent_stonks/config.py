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

# Apple Trader: the rule-based (non-LLM) loop that watches every closed minute
# bar of its configured symbol for a momentum-regime change into positive and
# asks a saved model about it (see agent_stonks.apple_trader).
#
# ENTRY_MODE decides *when* it asks, and it is the setting that changes what the
# agent does most:
#   "anticipate"  buy while the regime is still negative or balanced, on the
#                 model's forecast that it turns positive on the next bar. Needs
#                 a forecasting model, so only "nbeats" can run it.
#   "confirm"     buy the bar the change has already happened on, if the model
#                 rates it likely to hold. Either model can run this, and it is
#                 what the agent did before anticipation existed -- but by then
#                 momentum has already crossed the enter threshold, so the entry
#                 lands after the move that produced the signal.
#
# MODEL names which of the saved models answers that question -- a key of
# agent_stonks.apple_models.MODELS ("nbeats", the forecast-derived one, or
# "persistence", the incumbent classifier). It defaults to the forecaster
# because the default entry mode is one only the forecaster can answer.
# PROB_THRESHOLD is the probability a candidate has to clear to be bought; None
# uses the cut-off the chosen model picked on its own validation block, which is
# the intended setting because those cut-offs are not on a shared scale -- and
# since TimeToChange2 was re-run per ticker, not on a shared scale across
# symbols either. On AAPL they are 0.05 (the classifier's posterior) and 0.21
# (N-BEATS' gated survival probability); on GOOGL 0.43 and 0.41; on INTC 0.22
# and 0.05. Every one of them moved when the symbol was last retrained, so this
# list is an illustration and the bundle is the authority. Each is picked on
# that symbol's own validation events, so a number
# typed here means something different on each instrument, and None is the only
# setting that means the same thing everywhere.
#
# There are two exits, and either one closes the position. TRAIL_PCT is the
# price rule: sell once price is that far below the highest price seen since
# the entry. REVERSAL_THRESHOLD is the model rule: sell once the forecaster
# puts the positive regime at that probability or better of flipping to
# negative inside its 15-bar horizon -- the trailing stop waits for the
# give-back to happen, this one acts on the same forecast the entry was taken
# on. None switches it off, which is the only setting the incumbent classifier
# can run: like "anticipate", it is a question about bars that are not regime
# changes, so only a forecaster can be asked it.
#
# 0.30 is NOT a tuned number -- nothing in TimeToChange2 ever grid-searched an
# exit -- but it is not a guess either. It was measured on **AAPL only**, and
# nothing re-measured it when the forecaster was fitted for GOOGL and INTC, so
# on those two it is a borrowed default rather than a placed one. It is now a
# borrowed default on AAPL too: the curve below was read off the AAPL
# checkpoint retired on 2026-09-09, and retraining moved the forecast fan the
# reversal probability is drawn from. Re-measure before treating 0.30 as
# placed on any symbol. Over 428
# positive-regime bars on five AAPL sessions (2026-07-27 SIP, 2026-08-03..06 yfinance) the reversal
# probability separates bars within 3 of the end of a positive run from bars
# with 8+ bars still to go at AUC 0.89, and the cut-off picks where on that
# curve to sit:
#
#     >= 0.20   11.2% of held bars (~10/session)   52% land near the run's end
#     >= 0.30    2.6% of held bars (~2/session)    55%
#     >= 0.40    0.9% of held bars (~1/session)    75%
#
# against a 14.7% base rate. 0.30 is the knee: a few signals a session at ~3.7x
# base-rate precision, well past the 0.21 ninetieth percentile of the whole
# distribution, so it stays an outlier rather than becoming a second trailing
# stop. Lower exits earlier and far more often; much above 0.40 the rule stops
# firing at all.
#
# That the rule fires in the right places is measured. That it *helps* is not:
# A/B-ing those same five sessions moved the SIP day +0.140% -> +0.042% and the
# four-day yfinance run -0.577% -> -0.637%, i.e. nothing either way on six and
# twelve round trips. Sweep it in SimLab before trusting it, and treat None as
# a live option rather than the old behaviour.
# The "dayrange" model is the odd one out and ignores everything above. It
# forecasts where the whole session's high and low will land, once, at 9:35,
# and the rules built on it are TimeToChange3 notebook 05's: two resting levels
# a fixed number of average daily ranges below the predicted high H, with A the
# 14-day average daily range in dollars.
#
#     buy_level  = H - BUY_K  * A
#     sell_level = H - SELL_K * A
#
# 0.75 and 0.10 are the notebook's own settings, and they were specified rather
# than fitted -- which is the honest reason to leave them alone here. Notebook
# 05.9 swept both over five sessions: the week total peaks away from them, but
# the *count* of profitable sessions is flat at three in five across the whole
# region where the rule trades at all. Moving the levels changes the price paid
# on the same winning days, not how often the rule is right, and the grid's
# best cell beats the specified one without winning a single extra day. Five
# sessions across a 195-cell grid is selection noise; sweep them in SimLab
# before believing any peak.
#
# What the sweep does establish is the shape: out to a buy distance of about
# 0.85 every session trades and deeper entries simply fill better; past 0.90
# days start dropping out entirely and the totals turn erratic on a handful of
# trades. 0.75 sits inside the first regime, on the rising part of it.
#
# The "momentum_change" model is the third strategy and ignores both blocks above. It
# predicts how far the momentum score will move over the next fifteen bars, in
# bps/min, and TimeToChange notebook 05's rules read that number as a direction
# call on a regime the tape has already printed:
#
#     BUY   the previous minute's regime is negative and pred >=  BUY_THR
#     SELL  the previous minute's regime is positive and pred <= -SELL_THR
#     SELL  momentum falls below M1_MULT x theta  (the momentum floor)
#     SELL  price falls STOP_PCT below the entry
#
# 0.30 / 0.30 / -2.0 / 0.5% are the notebook's, and like the day-range pair they
# were specified rather than fitted. `scripts/simulate_week.py` sweeps all four
# over the reserved holdout week of each ticker, and what it establishes is
# mostly negative: on GOOGL and INTC alike the model's *exits* are the only
# profitable component, the momentum floor churns one-minute round trips
# whenever it sits above -theta (entries only happen while momentum is below
# -theta, so a floor above it is already breached at entry), and 0.5 bp per
# side turns both tickers negative. Sweep them in SimLab before believing any
# cell; five sessions per ticker is a sanity check, not an edge.
APPLE_TRADER_ENTRY_MODE = "anticipate"
APPLE_TRADER_MODEL = "nbeats"
APPLE_TRADER_BUY_K = 0.75
APPLE_TRADER_SELL_K = 0.10
APPLE_TRADER_BUY_THR = 0.30
APPLE_TRADER_SELL_THR = 0.30
APPLE_TRADER_M1_MULT = -2.0
# Percent, like APPLE_TRADER_TRAIL_PCT -- the notebook's 0.005 fraction.
APPLE_TRADER_STOP_PCT = 0.5
APPLE_TRADER_CYCLE_SEC = 60
APPLE_TRADER_PROB_THRESHOLD: "float | None" = None
APPLE_TRADER_TRAIL_PCT = 0.5
APPLE_TRADER_REVERSAL_THRESHOLD: "float | None" = 0.30
APPLE_TRADER_POSITION_PCT = 95.0
# Flatten this many minutes before the close: momentum, regimes and the model's
# whole feature set are intraday, and none of it survives the overnight gap.
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
# same prediction; the momentum marks borrow the up/down palette because they
# name a direction.
MODEL_OVERLAY_COLORS: dict[str, str] = {
    "day_range":     "#22d3ee",  # cyan, as the ML predicted profile curve
    "profile_range": "#a78bfa",  # violet
    "profile_poc":   "#f472b6",  # pink -- one number inside the violet band
    "momentum_up":   "#26c6a2",
    "momentum_down": "#ef5350",
    "momentum_flat": "#888",
    "momentum_hold": "#26c6a2",
    "momentum_turn": "#fbbf24",  # amber: the one mark about the future
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
