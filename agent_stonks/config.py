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
#
# The source and the Alpaca feed are offered as one choice, because they are
# one decision: "Alpaca" says nothing until the feed it streams is named, and
# Finnhub takes no feed at all -- it streams the consolidated trade tape. The
# two still travel separately through the app (the quote poll, the agents'
# fill-price lookups and the REST fallback all read the feed, whichever socket
# is running), so Finnhub is paired with IEX: the one Alpaca feed every plan
# serves.
LIVE_SOURCES: dict[str, tuple[str, str]] = {
    "finnhub": ("finnhub", "iex"),
    "alpaca:iex": ("alpaca", "iex"),
    "alpaca:sip": ("alpaca", "sip"),
}
LIVE_SOURCE_LABELS: dict[str, str] = {
    "finnhub": "Finnhub (trades \u2192 local candles)",
    "alpaca:iex": "Alpaca (iex)",
    "alpaca:sip": "Alpaca (sip)",
}
DEFAULT_LIVE_SOURCE = "finnhub"
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

# Live trading is off unless this environment variable is truthy. It is set
# outside the app, so nothing the agent does in-process -- and no timer that
# starts a run on its own -- can reach the live account on its own initiative.
# Picking "Alpaca LIVE" in the UI is then the deliberate act; the run is still
# announced in red for as long as it lasts.
LIVE_TRADING_ENV_FLAG = "ALPACA_ENABLE_LIVE_TRADING"

# A market order is accepted immediately and fills asynchronously. The tracker
# waits this long for a terminal state before recording whatever filled so far;
# anything still working is left with the broker rather than cancelled, since
# cancelling a partially-filled order is a trading decision, not a timeout.
ORDER_FILL_TIMEOUT_SEC = 20.0
ORDER_POLL_SEC = 0.5

# How long a reading of the venue's own account value (Alpaca's `equity`) is
# reused before it is refreshed. Portfolio value is marked to market on every
# streamed trade -- thousands of times a session -- and the account endpoint is
# neither free nor unmetered, so one read is shared across a short window and
# refreshed on a background thread rather than in the stream's hot path.
VENUE_VALUE_REFRESH_SEC = 10.0

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

# The age at which a backfilled bar counts as settled. Older bars come from a
# consolidated source (yfinance for regular-session minutes, the resolved
# history feed for the rest). Younger bars, other than the minute in progress,
# come from Alpaca IEX and are kept as provisional: IEX carries ~3-4% of the
# consolidated volume, so they are replaced once they are old enough for a
# consolidated source to serve them. The minute in progress is never backfilled;
# the live socket owns it. See `bar_history.fetch_live_bars`.
SETTLED_BAR_AGE_MIN = 15
OPTIONS_POLL_SEC = 60
OPTIONS_WALL_HISTORY_MAXLEN = 200
TIMEFRAMES = ["1Min", "5Min", "15Min", "30Min", "1Hour", "1Day"]

# Average daily volume (`state.current_volume_ratio`): the mean of the last
# VOLUME_ADV_WINDOW completed daily volumes; with fewer than
# VOLUME_ADV_MIN_DAYS completed days (thin history / early session), it falls
# back to yesterday's single-day volume.
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
# a fixed number of level units below the predicted high H, with U the unit
# APPLE_TRADER_LEVEL_UNIT names -- the notebook's 14-day average daily range,
# or the forecast's own predicted range.
#
#     buy_level  = H - BUY_K  * U
#     sell_level = H - SELL_K * U
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
# (buy_k, sell_k) per instrument -- see the day-range block above. Since
# 2026-09-26 only the fallback for a (model, instrument) pair with no SimLab
# tuning pick below: today that is the Day Range × Intraday Volatility model.
APPLE_TRADER_DAYRANGE_LEVELS: "dict[str, tuple[float, float]]" = {
    "AAPL": (0.40, 0.25),
    "GOOGL": (0.65, 0.05),
    "INTC": (0.50, 0.05),
}
# (buy_k, sell_k) per (model, instrument): the pick of SimLab's Tuning tab, one
# job per pair, each a buy × sell grid (buy 0.45..0.95, sell 0.05..0.45, step
# 0.10) replayed through the real engine -- market fills, the managed exit --
# on three weeks of yfinance tape (tech_week_2026-09-07, -09-14, -09-19;
# 14 sessions) at $100,000 per week. The pick is the best total profit of the
# three weeks' heatmaps summed (highest cell, no minimum share of days traded):
#
#   model     ticker  buy   sell  total   return  weeks up  days traded  job
#   dayrange  AAPL    0.55  0.05  +3757   +3.76%    3/3        14/14     20260923-181947-96f7cb
#   dayrange  GOOGL   0.85  0.45  +1317   +1.32%    2/3         9/14     20260923-182822-178bfa
#   dayrange  INTC    0.65  0.05  +7191   +7.19%    2/3         9/14     20260923-182628-bdbe37
#   highlow   AAPL    0.75  0.35  +3775   +3.77%    3/3        12/14     20260923-191233-fd16ae
#   highlow   INTC    0.85  0.35  +5971   +5.97%    3/3         7/14     20260923-191641-b7ded5
#
# How far to trust them. Every week is in sample -- the pick is read off the
# sum it is scored on -- and three weeks is 14 sessions. The grids were swept
# under the managed exit as it stood on 2026-09-23 (momentum confirmation off,
# the legacy fade / negative-momentum take on, no scale-in, no keep_width, and
# the brownian breach rule on dayrange), not under today's defaults, so a
# replay with today's exit will not reproduce these totals. On INTC and GOOGL
# (dayrange) the picked cell is well clear of the runner-up, i.e. a peak rather
# than a plateau. Every pick made most of its total in the first week
# (tech_week_2026-09-07); dayrange GOOGL and INTC lost money on the third. On
# dayrange INTC the untuned 0.50/0.05 made +7262 on the same weeks, more than
# the pick (0.50 sits between the grid's 0.10 steps). Re-read the Tuning tab's
# jobs after adding a week before trusting any of these over the fallback.
APPLE_TRADER_TUNED_LEVELS: "dict[tuple[str, str], tuple[float, float]]" = {
    ("dayrange", "AAPL"): (0.55, 0.05),
    ("dayrange", "GOOGL"): (0.85, 0.45),
    ("dayrange", "INTC"): (0.65, 0.05),
    ("highlow", "AAPL"): (0.75, 0.35),
    ("highlow", "INTC"): (0.85, 0.35),
}
# The day-range managed exit, on top of the sell level and the closing flatten
# (`DayRangeTrader._exit`). Unlike the levels these were never swept: they are
# specified starting points, and none of them is in the notebook's numbers.
#   stop       sell everything once a bar's low is STOP_GAIN_FRACTION of the
#              predicted gain under the fill, and take no new entry for the
#              rest of the session
#   take       once the momentum over the last NEGATIVE_MOMENTUM_BARS bars
#              (close - close N bars earlier) has been negative for
#              NEGATIVE_FOR_BARS bars in a row while holding, with the position
#              in profit, sell TAKE_FRACTION of it ...
#   runner     ... and keep the rest for the sell level only if that is still
#              HOLD_MIN_GAIN_K x ADR above the fill (otherwise sell it all);
#              a runner is sold if the price comes back to the fill
#
# The stop is written against the *predicted gain* rather than against the ADR:
# the trade is playing for the distance between the two levels, which is
# (buy_k - sell_k) level units -- both levels hang off the same reference and
# are counted in the same unit, so the gap between them does not move when the
# reference does -- and the only question a stop answers is how much of that to
# risk to make it. (Under APPLE_TRADER_LEVEL_UNIT = "pred_range" the *unit*
# itself can widen mid-session on a breach, which moves the gap; the stop is
# fixed in dollars at the fill either way.)
# 0.5 is one dollar risked for every two the target is worth. Written that way
# the number means the same thing on every instrument, which STOP_K x ADR did
# not: 0.20 ADR was a third of AAPL's 0.15-ADR target and a third of GOOGL's
# 0.60-ADR one is 0.20 too -- same number, wildly different bets.
#
# Read the two together before changing either. On the shipped pairs 0.5 of the
# predicted gain is 0.075 ADR on AAPL, 0.30 on GOOGL and 0.225 on INTC, so this
# default is a much tighter stop on AAPL than the 0.20 ADR it replaces and a
# wider one on GOOGL. 0 switches the stop off, which is what every record
# written before the managed exit existed replays as.
APPLE_TRADER_STOP_GAIN_FRACTION = 0.50
# The momentum take as it was 2026-09-23 to -24 (`DayRangeTrader._momentum_negative`),
# kept so the records made then replay; a new config uses the momentum
# confirmation below instead, and leaves this look-back at 0. Momentum is the
# N-bar price change, `close - close[N bars ago]` in dollars, and the take
# fires once that has sat under zero for NEGATIVE_FOR_BARS bars in a row since
# the entry.
# The streak is what keeps a single down bar in an up move from counting: 5 of
# 15 is a third of the look-back spent falling. Neither number was swept. 0
# bars switches the take off. Replaced (2026-09-23) the positive-to-balanced
# turn of the sigma score over `momentum_fade_bars`, which stored records still
# carry and replay (`simlab.rule_agents._APPLE_LEGACY`).
APPLE_TRADER_NEGATIVE_MOMENTUM_BARS = 15
APPLE_TRADER_NEGATIVE_FOR_BARS = 5
# The momentum confirmation (2026-09-24, `DayRangeTrader._momentum_read`): one
# look-back, in bars, for both sides. Momentum is the average per-bar move over
# it, `(close - close[N bars ago]) / N`, and its change the average bar-to-bar
# change of the 1-bar momentum, `(m1 - m1[N bars ago]) / N` -- so both are per
# bar, the same scale as `abs_mean_minute_momentum`, which sets the neutral band:
# a value is neutral while its size is under MOMENTUM_NEUTRAL_FRACTION of last
# week's mean absolute one-minute move. The user's behaviour table then decides:
# buy at the buy level only on positive momentum, or neutral with a change
# that is not negative; sell at the sell level unless momentum is positive; below it, take
# gains (in profit) on negative momentum whose change is neutral or negative.
# 5 bars is not swept; 0 switches it off, which is what every SimLab record
# written before it existed replays as (`simlab.rule_agents._APPLE_LEGACY`).
APPLE_TRADER_MOMENTUM_CONFIRMATION_BARS = 5
MOMENTUM_NEUTRAL_FRACTION = 0.10
APPLE_TRADER_TAKE_FRACTION = 0.70
APPLE_TRADER_HOLD_MIN_GAIN_K = 0.30
# What the agent does when the session trades through the forecast it was given
# at 9:35 -- the vocabulary lives here rather than in `dayrange_model`, which
# defines the arithmetic (`updated_range`) but costs 200 MB of torch to import,
# and both the form and the config dataclass need to name a policy without
# paying that. Same reason `TRADING_MODES` is a list here.
#
#   "off"       the 9:35 forecast stands all session, whatever the tape prints
#   "shift"     a breach moves both sides: the breached one to the session's own
#               extreme, the other by the same amount, so the forecast keeps its
#               width and is re-centred on where the day has gone -- never so
#               far that it would exclude a price the session already printed
#   "extreme"   only the breached side moves to the extreme, the other stays.
#               The rule "Move to the extreme so far" meant until 2026-09-23;
#               kept so records made under it replay as run, not offered
#   "brownian"  a breached side moves to the extreme and then past it by what
#               a driftless walk with ADR-implied volatility is still expected
#               to add, ADR x sqrt(session left)/2; with APPLE_TRADER_KEEP_WIDTH
#               the other side follows at the unit's width, without it it stays
#
# The forecast already refuses to sit under the opening window's high
# (`dayrange_model.apply_open_constraint`); the moving policies carry that same
# correction through the rest of the day, and `DayRangeTrader` rebuilds its two
# levels from the updated high each time it moves.
BREACH_OFF = "off"
BREACH_SHIFT = "shift"
BREACH_EXTREME = "extreme"
BREACH_BROWNIAN = "brownian"
# Every policy a config may carry -- including one a stored record carries
# that is no longer offered.
BREACH_POLICIES = (BREACH_OFF, BREACH_SHIFT, BREACH_EXTREME, BREACH_BROWNIAN)
# What a form or a new sweep offers.
BREACH_OFFERED = (BREACH_OFF, BREACH_SHIFT, BREACH_BROWNIAN)
BREACH_LABELS = {
    BREACH_OFF: "Hold the 9:35 forecast",
    BREACH_SHIFT: "Move to the extreme so far",
    BREACH_EXTREME: "Move only the breached side to the extreme (earlier rule)",
    BREACH_BROWNIAN: "Brownian extension, volatility implied by ADR",
}
# The default moves to the extreme because that is the weaker claim: "the
# day's high is at least what has already traded" is arithmetic, not a
# forecast. Since 2026-09-23 it moves the other side with it ("shift"), at the
# user's request: a day that has broken out of its forecast has moved, not
# widened, so the range keeps its predicted width. Under the default
# "pred_range" unit that also keeps the unit fixed, and both levels translate
# by exactly the breach. No policy has been swept, and
# APPLE_TRADER_DAYRANGE_LEVELS above was -- under "off", with the forecast held
# fixed all day -- so the two numbers there were picked against a rule this
# setting changes. A SimLab record written before the setting existed replays
# under "off" and one written under "extreme" replays that
# (`simlab.rule_agents._APPLE_LEGACY`), so no stored result moves.
APPLE_TRADER_BREACH_UPDATE = BREACH_SHIFT

# Whether the forecast is always widened to hold what the session has actually
# printed (`dayrange_model.contain_session`). The 9:35 forecast is already
# clipped to contain the opening five minutes (`apply_open_constraint`); this
# is that same rule applied for the rest of the day, so a predicted high the
# tape has traded through stops being the number the levels are measured from.
#
# It is arithmetic rather than a forecast -- it never leads the tape, it only
# declines to keep a number the tape has passed -- which is why it applies
# under every APPLE_TRADER_BREACH_UPDATE policy including "off".
#
# Consequence worth knowing before switching it on: "extreme" already satisfies
# this, so with it on "off" and "extreme" behave identically and the only
# policy that still differs is "brownian" (which leads the tape by
# `brownian_reach` on a breached side). The two remain separate settings
# because switching this off restores the distinction.
#
# A SimLab record written before this existed replays with it off
# (`simlab.rule_agents._APPLE_LEGACY`), so no stored result moves.
APPLE_TRADER_CONTAIN_RANGE = True

# Whether a breach under "shift" or "brownian" keeps the range at the level
# unit's width -- the 9:35 predicted range under "pred_range", the ADR under
# "adr" -- rather than whatever width it had come to. The breached side goes
# where the policy puts it (the extreme, or past it by the Brownian reach) and
# the other side follows at that width; the only thing allowed to make the
# range wider is a followed side that would exclude a price the session has
# already printed (`dayrange_model.contain_session`). At the user's request,
# 2026-09-25: a moved forecast is the same day moved, not a wider day.
#
# Without it "shift" keeps the width the range had just before the breach
# (which containment may already have widened, and never narrows again) and
# "brownian" moves only the breached side, so every breach widens the range by
# the reach. Under "pred_range" that widening is also a wider unit, so the two
# levels drift apart through a breached day; with it they stay the gap they
# were written as.
#
# A SimLab record written before this existed replays with it off
# (`simlab.rule_agents._APPLE_LEGACY`), so no stored result moves.
APPLE_TRADER_KEEP_WIDTH = True

# Whether a bar that trades through the predicted high closes an open position
# (`apple_trader.DayRangeTrader._exit`).
#
# The levels are a bet that the day tops out near the predicted high, so a bar
# that trades through it has settled that bet -- in the position's favour, at a
# better price than the sell level was ever going to offer. Without this the
# breach instead *moves the forecast*, and under "brownian" the sell level is
# carried past the bar that breached: the position rides on against a target
# that stepped out of its way, with the profit it had already earned given back
# if the day turns. Under "off" and "extreme" the target is reached on the same
# bar anyway, so this only changes what "brownian" does -- but it is checked
# under every policy, because which one is running should not decide whether a
# resolved bet is banked.
#
# Measured against the predicted high *as it stood when the bar opened*, not
# after this bar's own update: a level that moves on the strength of the bar it
# is being tested against is lookahead, however the range policy is set.
#
# The sell level is checked first, so a breach that also reaches the target is
# logged as the target exit it is. 0/False switches this off, which is what
# every record written before it existed replays as.
APPLE_TRADER_BREACH_EXIT = True

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

# What one "k" is worth in dollars -- the yardstick `buy_k` and `sell_k` are
# counted in, and with them everything written against the gap between the two
# levels (`apple_trader.level_unit`).
#
#   "adr"         the trailing 14-day average daily range (`adr14_abs`): what
#                 this symbol's day has been worth lately, a fixed number for
#                 the session, computed off closed bars and no part of the
#                 model's output.
#   "pred_range"  `pred_high - pred_low` from the day-range forecast itself:
#                 what the model says *today* is worth.
#
# The second is the reason this setting exists. Under "adr" the levels hang off
# a predicted high but are spaced by a historical average, so a day the model
# calls unusually wide gets the same distances as a day it calls unusually
# narrow -- the forecast sets where the levels sit and has no say in how far
# apart they are. Under "pred_range" one forecast decides both.
#
# Three consequences worth having in mind before switching:
#
#   * APPLE_TRADER_DAYRANGE_LEVELS below was swept in ADRs. A k means a
#     different number of dollars here, so those pairs are starting points
#     under this unit rather than swept ones -- the same caveat
#     APPLE_TRADER_BREACH_UPDATE carries, for the same reason.
#   * The unit is no longer constant within a session. APPLE_TRADER_BREACH_UPDATE
#     ratchets `pred_high` up and `pred_low` down, so a breached day *widens*
#     the range and the two levels spread apart, where under "adr" a breach
#     shifts both by the same dollar and the gap never moves. A position's
#     target can therefore move away from its fill mid-session, which under
#     "adr" it cannot. The stop is still fixed at the fill (`_stop_price`).
#   * It needs a forecast with both sides. Nothing else here reads `pred_low`,
#     so a bundle predicting only the high would go unnoticed until now;
#     `apple_trader.level_unit` falls back to the ADR rather than to a zero
#     width, which would put both levels on the reference.
#
# A SimLab record written before this setting existed replays under "adr"
# (`simlab.rule_agents._APPLE_LEGACY`), so no stored result moves.
UNIT_ADR = "adr"
UNIT_PRED_RANGE = "pred_range"
LEVEL_UNITS = (UNIT_ADR, UNIT_PRED_RANGE)
LEVEL_UNIT_LABELS = {
    UNIT_ADR: "ADR (14-day average daily range)",
    UNIT_PRED_RANGE: "Predicted Range (predicted high − low)",
}
# Short tokens for the run signature and the log lines, where "0.4 x ADR" and
# "0.4 x the predicted range" are different strategies and must not read alike.
LEVEL_UNIT_TOKENS = {UNIT_ADR: "A", UNIT_PRED_RANGE: "R"}
LEVEL_UNIT_PHRASES = {
    UNIT_ADR: "ADR",
    UNIT_PRED_RANGE: "predicted range",
}
APPLE_TRADER_LEVEL_UNIT = UNIT_PRED_RANGE
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
#
# AAPL was the instrument that disagreed, which is why it got 0.10 of its own
# first: at 0.20 its 0.15-ADR target exit could never clear the bar and every
# completed trade ended the session. 0.10 leaves a target exit (and anything
# better) alive and catches what the rule is for -- a breakeven runner, a
# flatten at the fill, a take that gave out early. Since 2026-09-23 (the user's
# call) 0.10 is every instrument's default, so the per-symbol table below is
# empty; it stays for the same reason the levels are per instrument
# (APPLE_TRADER_DAYRANGE_LEVELS): the number only means something against that
# symbol's own buy/sell distances. Counted in the level unit, like every k.
# With the tuned pairs of 2026-09-26 (APPLE_TRADER_TUNED_LEVELS) the target is
# 0.40 to 0.60 units on every model and instrument, so 0.10 stands a run down
# only on an exit well short of it.
APPLE_TRADER_MIN_WIN_K = 0.10
APPLE_TRADER_MIN_WIN: "dict[str, float]" = {}
APPLE_TRADER_CYCLE_SEC = 60
APPLE_TRADER_POSITION_PCT = 95.0
# Whether a day-range position may be added to on the way down
# (`apple_trader.DayRangeTrader._add`). Only does anything under a position size
# below 100%, since that is what leaves cash for a second buy. After every fill
# the next buy rests half-way between the last one and the bottom of the range
# (reference - 1 unit: the predicted low under "pred_range", H - ADR under
# "adr"), so with AAPL's 0.40 the ladder is 0.40, 0.70, 0.85, 0.925... units
# under the reference -- each add half as far below the last, never past the
# bottom. The stop sits its usual distance under the last actual fill (since
# 2026-09-28; before that it sat under the next rung while an add was
# affordable), so an add is placed only while its rung is above that stop -- a
# bar that reaches a rung under the stop is stopped out first. At the default
# stop (half the predicted gain) and a fill at the buy level the first rung is
# above it on most model/ticker pairs, but not on Day Range x Intraday Volatility
# for AAPL (stop 0.075, rung 0.30 units down) or INTC (0.225 vs 0.25), and a fill
# under the level leaves less room. Not swept.
# A SimLab record written before the setting existed replays with it off
# (`simlab.rule_agents._APPLE_LEGACY`), so no stored result moves.
APPLE_TRADER_SCALE_IN = True
# No buy into a sharp fall (`apple_trader.DayRangeTrader._falling`), as it was
# 2026-09-23 to -24 -- kept so the records made then replay. A new config
# leaves it at 0: the momentum confirmation above decides entries now. An entry,
# first buy or add alike, is refused while the price has dropped more than this
# many level units over the momentum look-back (APPLE_TRADER_NEGATIVE_MOMENTUM_BARS
# unless the run sets its own) -- the same `close - close[N bars ago]` the
# momentum take reads. The bar that reaches the buy level is usually a
# falling one, so this does not refuse dips; it refuses the steep part of one
# and lets the next bar buy once the fall eases, if the price is still there.
#
# 0.30 from the stored yfinance minutes (Aug-Sep 2026, ~30 sessions each): the
# 15-bar change is under -0.30 ADR on about 1% of minutes after 09:35 on AAPL,
# GOOGL and INTC alike (-0.24 at 2%, -0.16 at 5%) -- the same in ADRs on all
# three, which is why one default serves every instrument. About half the
# sessions touch it at least once, mostly in the first hour. Not swept. 0
# switches it off, which is what every SimLab record written before it existed
# replays as (`simlab.rule_agents._APPLE_LEGACY`), so no stored result moves.
#
# The default is 0.10 since 2026-09-23 (the user's call), a much stricter gate
# than the 0.30 measured above: in ADRs a 15-bar fall of 0.16 is already seen on
# 5% of minutes, so 0.10 refuses most buys made while the price is still
# sliding and waits for the fall to flatten out. Not swept either.
APPLE_TRADER_MAX_FALL_K = 0.10
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
    # HighLow's forecast of the same two numbers as "day_range": a sibling of
    # that cyan, bluer, so the two ranges can be drawn together and told apart.
    "highlow_range": "#60a5fa",
    # Apple Trader's two resting levels. Deliberately not a sibling of the cyan
    # above: these are an *agent's* orders, not a model's forecast, and reading
    # the chart means telling the two apart at a glance.
    "trader_levels": "#fb923c",  # orange
    # And the stop under that buy: red, because it is the one line on the chart
    # that marks a loss rather than an order meant to make money.
    "trader_stop": "#f87171",
}

# Alpha for the semi-transparent backgrounds overlays paint behind the candles.
# A predicted band covers most of the plot, so it has to read as a tint rather
# than a fill; a predicted window is narrow and can afford a little more.
MODEL_OVERLAY_BAND_ALPHA = 0.08
MODEL_OVERLAY_SPAN_ALPHA = 0.14

# Candle-pattern overlays (see candle_patterns.py). A fair value gap is tinted
# by its direction in the candles' own up/down colors; a gap price has since
# traded through is drawn fainter, ending at the bar that filled it.
CANDLE_PATTERN_COLORS: dict[str, str] = {
    "bullish": "#26c6a2",
    "bearish": "#ef5350",
}
CANDLE_PATTERN_OPEN_ALPHA = 0.22
CANDLE_PATTERN_FILLED_ALPHA = 0.08

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
