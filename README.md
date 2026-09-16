<img width="1025" height="672" alt="image" src="https://github.com/user-attachments/assets/6f68cbeb-dfa3-40dc-b7dd-222f1583a07b" />


# AgentStonks

Real-time market data dashboard built with Streamlit and the Alpaca streaming API. Tracks a whole basket of US equities at once — live candlestick bars, trade volume profile, and news — plus historical context and an autonomous LLM paper-trading agent that trades the basket from a single shared cash balance.

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)

## Features

### Multi-symbol basket
Enter any number of tickers in the sidebar; every panel and the agent operate across the whole basket, each symbol tracked by its own `SymbolState` (bars, trades, news, position) sharing one `AppState` and one paper cash balance.

### 📡 Live tab
- **Live candlestick chart** — minute bars streamed via WebSocket, seeded with REST history on start. Two live sources, switchable in the sidebar's *Connection* expander: **Finnhub** (the default) streams the consolidated trade tape and the candles are aggregated from it locally, so the newest candle is the minute *in progress* rather than the last one to close; **Alpaca** streams ready-made bars off the chosen feed and is the only one of the two that also streams bid/ask. Whichever is running, Alpaca REST still seeds the history, backfills holes in the bar series, takes over whenever the socket is down, and — under Finnhub — polls the quotes that tape doesn't carry
- **Current price** — live last price with change vs previous close
- **Candle display options** — toggle open-close body, 20%–80% percentile body, and whiskers independently
- **VWAP** — display as dot markers or a continuous line
- **Volume profile** — rainbow-coded price distribution histogram showing trade density over the session
- **VWMA overlays** — volume-weighted moving averages at 5, 15, and 60 periods
- **Average lines** — 7-day, 28-day, and 1-year daily average price overlays
- **Fibonacci levels** — session high/low retracement levels
- **Price profile fit** — fit a Gaussian or Cauchy mixture model to the volume profile (1–5 components), with optional centers shown on the candle chart
- **ML predicted profile** — overlay of where today's volume is predicted to trade, from a LightGBM quantile-function (density/EMD) model trained on the LevelsML workflow at the 9:30 open; the mixture fit can target either the live volume or this predicted curve. Needs the optional `lightgbm` dependency and the trained pack at `../Models/open_profile_lgbm.json.gz` (retrain with `Models/train_open_profile.py`; override the location with `OPEN_PROFILE_MODEL`)
- **Model predictions on the chart** — a *Model Predictions* picker in Chart Settings draws what the trained models say about today, in the idiom each answer calls for. **Predicted day range** (TimeToChange3, made once at 09:35) and **predicted price profile range** (the LevelsML density model's outer quantiles and point of control) are horizontal lines, drawn in the candle chart *and* across the volume profile beside it, with a semi-transparent band over the session they cover. **Predicted intraday range** draws a price envelope that changes with the time of day: IntradayVolatility's volatility curve (power-law decay from the open, flat midday, a short ramp into the close) around the open, widest at 09:30. On its own it is scaled to that model's daily-bar forecast of the day's range; **× day range** stretches the same curve so it tops out exactly at TimeToChange3's predicted high and bottoms out at its predicted low — the better choice, since the daily-bar range forecast is weak out of sample. Export the model with `FinNotebooks/IntradayVolatility/scripts/export_app_model.py`, which writes `../Models/intravol_<TICKER>.json` (override with `INTRAVOL_MODEL_<TICKER>`). Each overlay is offered only for the symbols its model was fitted on, and a missing bundle or too little history is reported as a caption under the chart rather than a silently empty overlay
- **Multi-timeframe** — 1Min, 5Min, 15Min, 30Min, 1Hour, 1Day
- **IEX and SIP feeds** — switch between free (IEX) and paid (SIP) Alpaca data for the live socket and the bid/ask quote poll
- **Consolidated bar history** — historical bars have their own source setting, separate from the stream's feed, because **IEX carries under 4% of consolidated volume** (measured on AAPL: 1.56M vs 41.6M shares over the same 390 one-minute bars). Pairing IEX history with a consolidated live stream would put a ~26x volume step mid-series that relative volume, the volume profile and the models' volume features all sum straight across. `auto` therefore prefers Alpaca SIP — real-time on a paid plan, or held back 16 minutes on a free/basic plan, which refuses only the trailing 15 minutes; delayed SIP is still the right backfill source, since backfill repairs *holes* and the live stream owns the recent window. Failing that it uses yfinance (within 1.5% of SIP, per-minute correlation 0.96), and IEX only when neither can answer

### 📰 News tab
- Latest headlines from Alpaca news (falling back to WorldNews API), with orange vertical markers on the Live chart and impact scoring, filterable per symbol
- **Impact estimate** — by default a symbol with its own **news-impact model** (`Code/Models/newsimpact_trajectory_<TICKER>.joblib` + `.json`, from `FinNotebooks/NewsImpact`; AAPL today) is scored by it, and every other symbol by the LLM; the selector can switch every symbol to the LLM. The model reads the market-adjusted momentum over the 15 minutes *before* an intraday release (plus the news count and hour) and badges the momentum state it expects over the 15 minutes after — it does not read the article text, and releases outside regular hours or in the first 15 minutes stay unknown (the badge tooltip says why). It needs 40 earlier sessions of ticker and SPY minute bars from Alpaca SIP (the volatility profile averages 20 sessions of beta-adjusted returns, and each session's beta averages the 20 before it), fetched once a day in the background, and scores new articles as they arrive. Adding a ticker is dropping its two files in; `NEWS_IMPACT_MODEL_<TICKER>` relocates one

### 🌅 Pre-Market tab
- **Premarket briefing** — an LLM synthesis of recent news, historical price context, macro indicators, and fundamentals into a structured morning briefing (catalysts, technical levels, outlook) per symbol, generated on demand before the session opens

### 🗂️ Historical tab
- **Price history** — daily closes over 7 days to 5 years, plotted against SPY and VIX
- **Expert price targets** — each analyst firm's dated price targets (from Yahoo's analyst actions feed) drawn as piecewise lines over the shown period, toggleable per symbol and per firm (via the chart legend)
- **Dividend and earnings markers** — overlaid on the historical chart
- **Static analysis** — trailing P/E, estimated annual return (growth + dividend), and estimated 10-year cumulative dividend return

### 🔬 Technical Analysis tab
- Daily trend regime, intraday momentum, and broad-market risk environment (VIX level/trend/term structure, S&P 500 trend and drawdown), each summarized in plain language with a gauge chart, per symbol

### 🏦 Smart Money tab
- Higher-timeframe bullish **order blocks** and **fair value gaps** drawn as demand/supply zones over daily candles, with entry/stop/target geometry overlaid when a setup is active

### 🧱 Put/Call Walls tab
- Call Wall / Put Wall (open-interest-based resistance/support) and net dealer gamma regime, computed from a yfinance options chain on its own independent poll loop

### 🤖 Agent tab
- **LLM paper-trading agent** — runs on a fixed cycle, reads already-fetched data for every symbol in the basket (bars, quotes, volume stats, news, options walls, positions) via tool calls, and reasons about a trading regime and strategy, trading from one shared cash balance across the whole basket
- **Agent personalities** — Momentum, Breakout, VWAP Mean-Reversion, and Premarket Analyst, each with its own system prompt, decision playbook, and tool set (default: Momentum). Two more stay wired but switched off, listed below:
  - *Momentum* — screens for gap + relative-volume + news catalyst, trades bull flags and VWAP reclaims
  - *Breakout* — waits for a volume-confirmed opening-range break, sizes via ATR-based `breakout_trade_geometry` targets requiring a minimum 2:1 reward/risk
  - *VWAP Mean-Reversion* — gates on an ADX-confirmed range (below 20), fades 2σ stretches from session VWAP back to the mean via `analyze_vwap_bands` + `vwap_reversion_geometry` (target VWAP, stop one σ beyond entry, minimum 1.5:1 reward/risk), preferring rejection candles at the bands; long-only, so it trims/exits into upside stretches rather than shorting
  - *Smart Money (Highest-Edge)* — **currently disabled** (listed in `agent.DISABLED_PERSONALITIES`, so it is not offered in the app or SimLab and the Automatic orchestrator never activates it; its prompt and tools stay wired for re-enabling). The composite institutional setup: identifies higher-timeframe bullish **order blocks** (`analyze_order_blocks`) and enters only when price *returns* into a demand zone during the session with intraday confirmation — a rejection candle, a filled **fair value gap** (`analyze_fair_value_gaps`), or a breaker/break-of-structure — via the composite `analyze_smart_money_setup` read. Stop sits just beyond the block; target is the next opposing structural level, sized with `smart_money_trade_geometry` to a minimum 3:1 (typically 3:1–5:1) reward/risk; long-only
  - *Volume Signal Detective* — **currently disabled**, on the same terms as Smart Money above (in `agent_prompts.DISABLED_PERSONALITIES`, so no picker offers it and the Automatic orchestrator never activates it; prompt, tools and avatar stay wired for re-enabling). A structure-first session trader: spike-classified demand/supply lines from `analyze_volume_profile_2` are the primary read, cross-examined against session structure, participation, the approach into the level and the catalyst, with a 2:1 room-to-run gate before any size. `detect_regime_shift` is the counterweight — levels describe the past, and a level map alone keeps bidding demand into a tape that has already turned, so the trajectory read gates every plan the levels suggest. It stays wired rather than deleted because a personality key is also the identity of every run that used it: `data/simlab/experiments` holds finished Detective runs, and the label lookup falls back to the *default* for an unknown key, so deleting the entry would silently re-file those results under Momentum Trader
  - *Premarket Analyst* — a one-shot pre-open specialist, gated to a window just before the bell and never selectable by the Automatic orchestrator once the session is live: it reads the premarket briefing and arms opening Tactics instead of trading directly, then retires once those tactics execute or the session starts
- **Automatic (regime-adaptive orchestrator)** — a meta-agent that detects the current market regime and activates the single best-fitting strategy above for you. Before the session opens it deterministically hands off to the Premarket Analyst; during the session it reads the same analysis tools (daily trend, broad-market backdrop, intraday momentum, volume, VWAP/ADX range gate, opening range, order blocks, options walls, news) and calls `select_strategy`. It then goes to sleep and hands control to the chosen strategy, which trades normally. When that strategy judges its edge has faded (e.g. a breakout agent in a dead range, a mean-reversion agent once a trend takes hold, a momentum agent after the move and volume dry up) it calls **`stand_down`** with reasoning instead of idling on an alert — waking the orchestrator to re-assess the regime and switch to a better-suited strategy. The Agent tab status line shows the currently active strategy and detected regime
- **🍎 Apple Trader (rule-based, no LLM)** — the one personality with no model reasoning in the loop: once a minute it reads the minute bar that just closed and applies fixed rules built on a saved model, so the same tape always produces the same trades. The model is the **Day-range forecast (TimeToChange3)**: at **9:35** it forecasts where the *whole session's* high and low will land, from a year of daily bars plus the first five 1-minute bars, and then asks the model nothing else all day. Three predictors blend equally (LightGBM + N-BEATS + N-HiTS over a 32-day lookback of eight per-day channels), a heavily-regularised ridge learns what the opening five minutes add on top of the daily prediction, and the result is clipped to contain the range that has already printed. **Fitted per ticker**: `Models/timetochange3_dayrange_<TICKER>.joblib` (AAPL, GOOGL and INTC) plus the two `.pt` checkpoints and the metadata JSON beside each, every symbol with its own daily models, its own opening ridge and its own held-out error ($2.12, $2.75 and $1.72 respectively). Needs `torch`, `lightgbm`, `scikit-learn` and `joblib`; locations override with `APPLE_DAYRANGE_MODEL_<TICKER>`. Not selectable by the Automatic orchestrator, which drives LLM prompts

  **Which instrument** is picked in the same panel, and the model constrains it: every rule here is a saved model's output, so the picker offers only the symbols the model was fitted on. A run trades that one symbol, which must be streamed. A missing model or dependency is reported instead of traded around.

  The rules are TimeToChange3 notebook 05's, with both levels adjustable. With **H** the predicted high and **A** the trailing 14-day average daily range in dollars, it **buys** when a bar's low reaches `H − buy × A` and **sells** when a bar's high reaches `H − sell × A`, re-arming as often as the day allows, with anything still open flattened before the close. The levels default per instrument — AAPL 0.40/0.25, GOOGL 0.65/0.05, INTC 0.50/0.05 — from re-running the notebook's own grid over every session with a forecast (`FinNotebooks/TimeToChange3/scripts/sweep_levels.py`); the notebook itself specified 0.75/0.10, which any other symbol falls back to. On top of that sits a **managed exit**, each distance measured from the fill and each part switchable off with 0: a **stop** that sells when a bar's low reaches `fill − stop × A` (0.20) and ends new entries for the session; a **momentum take** that, once the tape's momentum score has fallen `drop` σ (1.0) from its best since the entry while the position is in profit, sells 70% and keeps the rest as a **runner** — but only when the sell level is still at least `hold × A` (0.30) above the fill, otherwise it sells everything; and a **breakeven** on that runner, sold if the price comes back to the fill. None of the exit is in the notebook's numbers, and SimLab records from before it replay with it off. It is a mean-reversion bet by construction: what the model forecasts well is the **width** of the day, not its direction, so the rule buys well below where the day is expected to top out and sells just under it, and on a day that never dips that far it does nothing.

  **What H is, and whether it holds still.** Two settings, and they answer different questions. The first is *what the two distances are measured below*. **Predicted high** is the notebook's: one number for the session. **Predicted range × intraday volatility** points them at the upper curve of the *predicted intraday range × day range* band instead — the same TimeToChange3 high and low stretched by IntradayVolatility's fitted time-of-day shape around the session's opening print, read at the minute of the bar that just closed. It *is* the predicted high at 09:30, pulls in to roughly a fifth of the distance to the open by midday, and opens back up into the last half hour, and the whole ladder moves with it. Three consequences worth knowing before using it: an entry needs a deeper dip to fill as the day quiets; the **target descends too**, so a morning position can be closed by a sell level that came down to it rather than by a price that rose to it; and the band is anchored on the **open**, not on the price, so a day that trends away from the opening print leaves the levels behind and leans on the stop. It needs that symbol's own `Models/intravol_<TICKER>.json` on top of the day-range bundle — checked before a run starts, and offered in the form only where it exists — and the shipped buy/sell distances were swept against a reference that does not move, so sweep them again under it rather than reading the first result as the strategy's worth. It is the same curve `model_overlays` draws behind the candles, from the same function, so the level and the picture agree.

  **H is not quite fixed for the day either.** The model runs once — its opening features are a fixed five-minute window, so it cannot be re-run — but a session that trades *through* the predicted high has already falsified it, and both levels hang off that number. What happens then is a setting with three choices. **Hold the 9:35 forecast** is the notebook's rule and what every stored record replays as. **Move to the extreme so far** replaces the breached side with the session's own high (or low) — the same claim `apply_open_constraint` already makes against the opening five minutes, carried through the day. **Brownian extension** goes past the extreme by `A × √(session left) ÷ 2`, the one-sided excursion a driftless walk whose volatility is implied by the ADR would still be expected to make (`E[range] = 2 × E[max]` over a session is what makes the constant a half): half an ADR with the whole day ahead, nothing at the bell. Only the breached side moves, only ever outward, and both levels are rebuilt from the new high the moment it moves — including for a position already open, whose target moves away from it. The choice is part of the Results signature unless it is off, so the three queue as three configurations; neither updating policy has been swept, and the buy/sell distances above were picked with the forecast held fixed all day. Note that the two settings are much less than additive: under the intraday reference a dollar added to the predicted high moves the levels by the shape at that minute, about a fifth of a dollar in the afternoon.

  What the numbers are. On a 129-session test window the blend's mean absolute error on the two targets is **0.0077 in log units ($2.12)**, against 0.0110 for a 14-day rolling baseline and 0.0135 for persistence — 30% of the baseline's error removed, and the ordering holds across all four walk-forward refits. That is a statement about the forecast, not about the trading rule: the default levels were picked on the same 31–36 sessions they are scored on, with limit fills and no costs, and how far each deserves trust differs by ticker (INTC's pick flips sign between the two halves of its sample) — `config.APPLE_TRADER_DAYRANGE_LEVELS` carries the table.

  Two things the live path does differently from the notebook, both worth knowing before comparing results. The **fill**: the notebook rests limit orders and fills a touch *at* the level, while this ledger is market-order only and buys near the close of the bar that touched it, so a bar that dipped and recovered fills worse here — a real cost, and it runs one way. And the **minute tape's volume scale**: the opening ridge's `or_volume_share` feature was fitted on consolidated volume, and Alpaca's IEX feed carries about 4% of it, which puts that one feature outside anything it saw in training. Run the stream and any SimLab dataset on a consolidated tape (`yfinance` or `sip`); the app says so when it does not.

  The TimeToChange2 persistence classifier, its N-BEATS forecaster and the TimeToChange delta-momentum regressor used to be offered here too and have been removed. SimLab runs recorded on them still show in Results, but replaying one is refused rather than run on the day-range model.
- **🍏 Apple Trader 2 (adjustable rules, no LLM)** — the same fixed loop over one symbol's minute bars, the same ledger and the same flatten-before-close, with the strategy taken out of the code. Instead of running one hard-coded strategy, a run is configured with an **instrument** and a **list of action items**, both built in the Agent tab (and in SimLab's Simulate tab, where the list stands in for a prompt). Each item is:
  - **an action** — buy or sell;
  - **a size** — a **percentage** (of cash on a buy, of the position on a sell), a **dollar amount**, or a **share count**. Every mode is clipped to what the ledger can do, so "sell 200 shares" out of 50 sells 50 and a rule set is portable across starting balances;
  - **one or more conditions**, each a named signal against a number (`above` = ≥, `below` = ≤), joined by **AND** (all of them) or **OR** (any of them). One joiner per item on purpose: `A and B or C` has no meaning without precedence rules, and a genuine mix is two items;
  - optionally a **cooldown** in bars, so a rule whose condition stays true ladders in only when that was the intent (390 ≈ one session, i.e. "once a day"), and an on/off switch.

  The **signals** a condition can read are the whole vocabulary, and the day-range model appears here as signals rather than as a strategy: the **bar** (close, high, low, change on the day); the **momentum regime** (score, regime, bars in regime, the dwell of the regime just left, whether this bar is a change into positive); the **day-range forecast** (predicted high/low, the 14-day ADR, and the distance from this bar's low/high/close to the predicted high **in ADRs**, which is the form notebook 05's rule is written in); the **position** (shares, open P&L, give-back from the peak since entry — `below -0.5` is a 0.5% trailing stop — bars held, value, cash, and **the change since the last buy**, measured against `max(that fill price, every high since)`, which unlike the position give-back survives the exit and so can arm a re-entry: "buy again once price is 1% below the best price since I last bought"); and the **clock** (minutes since the open, minutes to the close).

  **Which of those exist depends on the instrument**, because a model is fitted on a symbol and none of the notebooks claims transfer. **AAPL**, **GOOGL** and **INTC** carry the day-range forecast, so all three get the whole catalogue; **any other symbol** gets the model-free half — the bar, the momentum regime, the position and the clock, all computed here from the tape and meaning the same thing everywhere. The picker offers the modelled symbols plus whatever the app is streaming (or, in SimLab, whatever the selected datasets carry) and accepts anything typed into it. Switching the instrument never rewrites the rules: conditions that no longer read are marked in place and the set is refused as unrunnable until they are gone, and the presets narrow the same way (every instrument keeps at least the model-free one). Adding a symbol is one entry in `apple_models.DAYRANGE_TICKERS` plus its bundle in `Code/Models` — ORCL is trained in `FinNotebooks/Models` and not wired up.

  What the engine adds around whatever list it is handed: **at most one action per closed bar** and **the first matching rule wins**, so the order is the priority — but a rule that matches and *cannot transact* (a sell with the book flat, a buy with no cash) is passed over rather than eating the bar, so an exit written above an entry never blocks it. **An absent signal never matches**, in an AND or an OR alike: a model asked about a bar it was not fitted for, a day-range forecast before 9:35, a P&L with no position all read as nothing, and nothing fires a rule. **The closing flatten is not a rule** and cannot be deleted — every signal here is intraday. And **nothing is computed that no rule reads**: the model bundles loaded are the ones the conditions name (a rule set written on price and momentum loads none at all and needs no saved artifacts), the day's high/low forecast is made only if something asks for it, and conditions short-circuit within a rule.

  Three **presets** ship. The first is Apple Trader's own strategy written in this vocabulary — the day-range levels, at the notebook's 0.75/0.10 — so a rule set can be compared against the thing it was meant to improve on. The other two ("scale in, take half off, trail the rest" and a tape-only regime-turn rule) are not recommendations and are not measured; they say things the first agent structurally cannot, and they run on any instrument. Nothing here tunes or validates a rule set — the vocabulary makes it just as easy to write one that fires every bar or holds a position no exit can reach; `ruleset_error` refuses only the ones that cannot work at all (no enabled rules, a rule with no conditions, a set that can only sell, a model that is not installed), and SimLab is where the rest gets answered
- **Tactics (standing conditional trade plans)** — instead of trading at the current price, the agent can arm one or more conditional actions via `set_tactics` (e.g. "buy 10 shares if last_price below X", "sell 20% of the position if last_price above Y and vix below Z"). A background executor evaluates armed tactics against every live tick and fires the first action whose conditions all hold, through the same decision-tracker path as a manual agent decision — then disarms the set and wakes the agent to reevaluate with the fill in hand. While a sell bracket is armed on an open long position (take-profit above, protective stop below the entry price), the executor also trails the stop up mechanically: when the price's high-water mark covers a fraction of the entry→target distance, the stop rises to cover the same fraction of its own distance to the target — the take-profit never moves, and the stop only ever ratchets up
- **Provider-agnostic** — works with Gemini, OpenAI, or Anthropic, via a unified chat-completions client
- **Smart wake-ups** — when it doesn't want to trade, the agent sets condition alerts on any continuously-updated value (price, bid/ask, spread, day high/low, volume, relative volume, short-window momentum, portfolio value) to wake early the moment one is crossed, and always wakes early on fresh news for any symbol in the basket regardless of any alerts set — there is no idle "do nothing" decision
- **Dynamic position management** — while holding a position the agent doesn't just sleep on a static stop/target bracket: it arms checkpoint alerts at intermediate favorable levels (e.g. +1R) and momentum-fade conditions, so it is woken mid-trade to ratchet the stop up (breakeven, then trailing under fresh structure) instead of letting a winner round-trip back to the original stop
- **Independent fill pricing** — decisions (buy/sell/alert) are handed to a separate decision tracker that fetches its own fill price, so the agent never picks the price its own trade is recorded at
- **Paper broker only** — no real orders are ever placed; each filled buy/sell costs a fixed simulated fee
- **Equity curve & performance summary** — replays recorded decisions against streamed bars to reconstruct portfolio value over time
- **Live agent log** — cycle starts, tool calls, analysis, decisions, news alerts, and errors, streamed as they happen
- **HTML report export** — generates a self-contained HTML file with starting conditions, charts, and full decision/activity history
- **Daily accuracy scoring** — a per-session scorecard accumulates a deterministic grounding check (every number the model states in a decision must trace back to a number it was actually shown) plus tool errors and tactics-validation rejections; at most once per UTC day, and only once the day has accumulated an hour of agent runtime, these are aggregated into a scoring report and, when Langfuse is configured, registered there as a `daily-grounding` score; each finished session additionally registers a `session-profit-efficiency` score — the session's portfolio return divided by the maximum profit an oracle's best single round trip could have made on the session's symbols (buy the session minimum and sell the highest later price, or sell the session maximum having bought the lowest earlier price, whichever is larger)
- **Optional LLM observability** — when Langfuse credentials are set, each agent cycle is traced end-to-end (tool calls, token usage, latency) via `observability.py`; a no-op otherwise
- **Data-source logging** — every fetch (WebSocket stream, Alpaca REST, yfinance, WorldNews) logs which source served the data and which fallbacks were tried, de-duplicated so repeated identical outcomes don't flood the console

### 🧪 SimLab — strategy testing suite (`sim_main.py`)
A separate app that replays the trading agents against **stored historical sessions** instead of the live tape — same prompts, same tools, same execution path (`run_agent_cycle`, `DecisionTracker`, `TacticsExecutor`), so a strategy tested here is exactly the strategy that trades live. Hours of "wait for the condition" collapse into minutes: between LLM cycles the engine fast-forwards bar by bar, firing armed tactics, condition alerts, and news wake-ups deterministically from the stored data.

- **Agents tab** — every personality with its avatar, an editable system prompt (saved overrides apply only to simulations; the live app keeps the built-in), and the agent's exact tool set, each tool runnable by hand against any stored moment (pick dataset + symbol + time, see the JSON the agent would see). Apple Trader has no prompt or tools to edit, so its card shows its rules plus the provenance and held-out metrics of the saved model behind them — one block per symbol the model was fitted on, since GOOGL's bundle is a different model from AAPL's rather than the same one on another tape. Apple Trader 2's card has no strategy to describe at all, so it documents the **vocabulary** instead: every signal a condition can read, and the rules the engine applies around whatever list it is handed
- **Datasets tab** — download named datasets (symbols + date range + **feed**): minute bars 04:00–20:00 ET, ~1.5y of daily history, per-day news, and SPY/VIX/VIX3M context, stored as gzip JSON under `data/simlab/store/` deduplicated per (feed, symbol, day) — overlapping datasets never re-download a day. **1-minute bars come from yfinance** (the default feed): consolidated-tape OHLCV across every venue, free, no Alpaca data subscription, and the same source the live volume tools read, so a simulated volume ratio is like-for-like with the live one. Two limits come with it — Yahoo serves 1-minute history for the **last 30 days only** (earlier days download empty), and it publishes no per-bar VWAP, so yfinance bars carry no `vw` and the VWAP note/line is absent on those runs; extended-hours minutes carry real prices but are reported at zero volume. Alpaca's `iex` and `sip` remain selectable — for a window older than 30 days, or to re-run a dataset already downloaded on them. The feed is part of the data, not a download setting: `iex` is one venue (~4% of consolidated volume on a large cap), so its bars carry different closes, far smaller volumes and occasionally an extra or missing minute, and any agent whose rules are thresholds over those bars can trade a different day on each tape. The same day on two feeds is stored twice, deliberately, and every run record states the tape it read
- **Simulate tab** — pick agent + dataset + day(s) + provider/model and run: a pinned simulated clock (`agent_stonks/clock.py`) drives real agent cycles at historical moments, `SimBroker` fills at the stored tape price, and live fetches are rerouted to the dataset (`simlab/patches.py`). Results: equity curve, per-symbol candlesticks with trade markers, the full decision ledger and agent log, an **oracle ceiling** (best single round trip available on the tape) with profit efficiency against it, and an **LLM judge** that grades every entry on the information available at entry time (outcome shown only to calibrate) plus an overall strategy-adherence review. Runs persist under `data/simlab/runs/`; with Langfuse configured, cycles are traced and run scores (`sim-return-pct`, `sim-profit-efficiency`, `sim-judge-overall`) are registered there
- **Drift tab** — how a saved model's accuracy has moved since it was trained. Pick a model (day-range forecast, intraday volatility, open price profile), a tape and an instrument, and every stored session is scored the way a live run would have seen it — daily bars strictly before the day, its opening print and minutes — against what the day then did: the day-range model by the error of its predicted high and low (and their bias), intraday volatility by the error of its log day-range forecast and how closely each session follows the time-of-day curve, the profile model by the earth mover's distance to the realised volume profile and the share of volume inside its predicted outer quantiles. The metric is plotted **by day or by week**, beside the model's own reported number where one exists (the held-out test MAE, the walk-forward EMD, a nominal band coverage), with each **training cutoff read from the model's saved metadata** drawn as a line and the stretch inside the training data shaded — the day-range model has two (daily models through the start of the minute data, the opening ridge through its held-out day), so its drift is read to the right of the second. Cards compare the mean inside the training data with the mean after the last cutoff. Scoring loads the model, so it runs on request and is cached until the store changes
- **Tuning tab** — a parameter grid for Apple Trader, after TimeToChange3 notebook 05's profit heatmap. Pick **one or two parameters** to sweep (the day-range buy/sell distances, the stop, the momentum fade and take, the runner threshold, the size or the flatten window) with a from/to/step each; everything else comes from a base configuration edited with the same form as Simulate. Pick a **tuning dataset** and a **test dataset**: the grid is swept on the first, one cell is chosen there — the highest, or the middle of the best 3×3 plateau as `sweep_levels.py` chose the shipped levels, among cells that traded on enough of the days — and that pick is replayed on the second next to the untuned base configuration, with a verdict on whether its profit held up. Optionally the whole grid is swept on the test dataset too, which shows whether the profitable region itself moved. Every cell is a real replay through the same engine and trader as Simulate — market fills, managed exit, flatten — so a cell's number is what that configuration would show in a Simulate run (lower than the notebook's limit-fill heatmap). Combinations the configuration refuses (a sell distance at or above the buy distance) are left as holes, datasets sharing sessions are flagged as not out of sample, and sessions the model could not forecast are named. A job runs in a detached worker (`python -m simlab.tuning <id>`) that fans cells out over a process pool; the tab shows its progress, can stop it (keeping the cells already swept), and stores results under `data/simlab/tuning/`. The rule replay was made several times faster to afford this — the market's daily-bar and day-volume reads bisect a per-day index instead of rescanning the tape every minute, and the rule agents' minute frame is sliced from one frame per replayed day — each pinned against the code it replaced
- **Apple Trader in SimLab** — the rule-based agent replays on the same engine, the same tape and the same ledger, but on its own day loop: it reads *every* closed bar of the session instead of sleeping on armed conditions, so there is nothing to fast-forward past. It needs no LLM, so it is queued once per dataset rather than once per LLM model, and its **rule set stands in for a model name** — moving either day-range level, what they are measured below, the intraday update, the size or the instrument is a new configuration to test, not a repeat of one already run. **It is never scored by the LLM judge**: the judge grades an agent's stated reasoning against the tape it cited, and this one states none of its own, so profit, profit efficiency and the oracle ceiling are its whole scorecard. It requires the symbol its configuration names in the dataset — checked before anything is queued, and part of the signature, so the same levels over AAPL and over GOOGL are two configurations to compare rather than one averaged row. It reads the dataset's stored **daily** history too (a 252-day window, well inside the 420 days a dataset stores), plus the simulated day's official opening print — the one field of that day's stored daily bar exposed during the run, since the 9:30 auction is fixed before the forecast is made. Stored runs on the removed momentum models still appear in Results, but a replay of one is refused

- **Continuous multi-day price charts** — a run covering several days spends two thirds of its time axis on nights and weekends, which a time axis draws as blank space. Those stretches are removed from the axis instead (computed from the bars themselves, so a weekend, a holiday or a half-day is just one longer gap and needs no special case; an intraday hole where nobody traded is left alone, since collapsing it would make the axis lie about how long a move took), and the day boundary they used to provide is put back as a rule at **09:30** and **16:00** ET, the open labelled with its date. Extended-hours bars then read as what they are: the stretch outside the rules. A session the bars never reach — a run that stopped in the pre-market — is given no opening bell it never saw
- **Model predictions over a replayed day** — the per-symbol chart in Results carries the same *Model predictions* picker as the live chart, and the same overlays, drawn over the tape that actually happened. Offered on **every** run, not only the ones a model drove: what a model would have said about a session is as worth seeing on an LLM agent's day as on Apple Trader's, and nothing here reads the run's own decisions. Each day is scored from the state of the world at its 09:31 — completed daily bars strictly before it, plus that day's stored opening print — so a replay shows the forecast that was makeable that morning rather than one built off the outcome. On a multi-day run each session's levels are drawn over that session only

- **Several setups of one rule agent per batch** — an LLM agent varies over models; a rule agent has no model, so its **setups** take that place. Add Apple Trader or Apple Trader 2 as many times as there are configurations to compare and each is queued against every selected dataset, so a threshold sweep, two instruments, one rule switched on and off, or the classifier against the forecaster is a single batch rather than several trips through the tab. Each setup is its own editor; the newest stays open and the rest collapse to their **signature**, which is the string Results groups runs by, so a glance answers whether they are really different configurations. Two setups that sign the same are queued **once** and the page says so — that signature is the run record's identity everywhere downstream, so running both would produce two runs shown as one row. Note that not every field is in it (neither agent's carries the closing flatten), so setups can differ on screen and still be one configuration

- **Apple Trader 2 in SimLab** — replays on the same rule day loop, queued once per dataset like Apple Trader, and never judged for the same reason. What stands in for a model name here is the **instrument plus the rule list**: the signature Results groups on carries both, so moving one threshold, changing a size, adding a condition, reordering two items or pointing the same rules at another symbol queues a new configuration to compare rather than a repeat of one already run (a rule that is switched *off* does not change it — it did not run). The list is edited in the Simulate tab and stored verbatim on the experiment record, so a run is replayable from what it carries; a record written before the instrument was configurable replays as the AAPL run it was. The symbol it names must be in the dataset, which is checked before anything is queued. Which models a run needs — and therefore which bundles must be installed for it to start at all — falls out of which signals the rules read *and* which symbol they read them on; a set written on price, momentum and the position needs none

```bash
streamlit run sim_main.py
```

## Quickstart

```bash
cp .env.example .env        # fill in your Alpaca credentials (and FINNHUB_API_KEY for the default live source)
pip install -r requirements.txt
streamlit run main.py       # live dashboard
streamlit run sim_main.py   # SimLab strategy testing
```

Open http://localhost:8501.

## Docker

```bash
docker build -t agent_stonks .
docker run -p 8501:8501 --env-file .env agentstonks
```

## Configuration

| Env var | Description |
|---|---|
| `ALPACA_API_KEY` | Alpaca API key ID |
| `ALPACA_SECRET` | Alpaca secret key |
| `ALPACA_PAPER_API_KEY` / `ALPACA_PAPER_SECRET` | (optional) Alpaca **paper** trading keys. Orders default to the paper account; if `ALPACA_API_KEY` is already a paper key these are unnecessary — paper falls back to it |
| `ALPACA_ENABLE_LIVE_TRADING` | Set truthy to make **live** trading selectable at all. Off by default; the app also requires a typed confirmation per run |
| `ALPACA_LIVE_API_KEY` / `ALPACA_LIVE_SECRET` | (optional) Alpaca **live** trading keys — real money. Never inherited from the data or paper variables |
| `FINNHUB_API_KEY` | Finnhub API key — powers the default live data source (trades → locally built candles). Without it the app streams bars from Alpaca instead |
| `GEMINI_API_KEY` | (optional) Gemini key — for LLM news scoring and/or the trading agent |
| `OPENAI_API_KEY` | (optional) OpenAI key — for LLM news scoring and/or the trading agent |
| `ANTHROPIC_API_KEY` | (optional) Anthropic key — for LLM news scoring and/or the trading agent |
| `WORLD_NEWS_API_KEY` | (optional) WorldNews API key, used as a fallback news source |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | (optional) Langfuse keys — enables agent cycle tracing; no-op if unset |
| `LANGFUSE_HOST` | (optional) Langfuse host — Langfuse Cloud is used if unset |
| `OPEN_PROFILE_MODEL` | (optional) path to the LevelsML density-model pack — defaults to `../Models/open_profile_lgbm.json.gz` |
| `APPLE_DAYRANGE_MODEL` | (optional) path to the **AAPL** TimeToChange3 day-range bundle Apple Trader runs on — defaults to `../Models/timetochange3_dayrange_AAPL.joblib`; the two `.pt` checkpoints and the metadata JSON are read from beside it under the same stem. It names one file, so it stands in for AAPL's bundle only |
| `APPLE_DAYRANGE_MODEL_<TICKER>` | (optional) path to another ticker's day-range bundle (e.g. `APPLE_DAYRANGE_MODEL_GOOGL`) — defaults to `../Models/timetochange3_dayrange_<TICKER>.joblib`, one bundle per symbol the model was fitted on |

Credentials can also be entered directly in the sidebar (Alpaca) or the Agent tab (LLM provider); env vars are used as fallback. At least one LLM provider key is required to use the Agent tab or LLM news impact scoring.

> **Note:** Free Alpaca accounts stream the IEX feed only, and only during US market hours (9:30–16:00 ET). They *do* get the consolidated SIP tape over REST outside a trailing 15-minute window, which is what the history/backfill source uses by default — IEX's ~4% share of consolidated volume makes it a poor match for a consolidated live stream.

## Project layout

```
agent_stonks/
  config.py     — constants, color palette, agent timing/cost settings
  state.py      — shared mutable AppState (per-symbol SymbolState: bars, trades, news, position;
                  agent log, WebSocket handles, shared paper cash balance)
  market_hours.py — US regular-session clock (09:30-16:00 ET, Mon-Fri), used to gate the
                  Premarket Analyst and Automatic orchestrator's pre-open handoff
  rest.py       — Alpaca market-data REST helpers (fetch_bars, fetch_trades, fetch_daily_bars, fetch_news)
  trading_rest.py — Alpaca *Trading* API (accounts, positions, orders) — a different host and a
                  different key pair per venue; the only module that can move money
  trading_mode.py — picks the venue for a run (local / Alpaca paper / Alpaca live) and refuses
                  toward simulation on every misconfiguration; live needs an env flag AND a
                  typed confirmation
  broker.py     — the Broker abstraction: PaperBroker (invented fills), SimBroker (stored tape),
                  AlpacaBroker (real orders, real rejections, account is the source of truth)
  stream.py     — Alpaca's WebSocket streaming threads (bars/trades/quotes, news), plus the REST
                  safety nets every source shares: the stream-down fallback poll, the periodic
                  bar backfill, and the quote poll the Finnhub source runs on
  finnhub_stream.py — the default live source: Finnhub's trade tape aggregated into candles locally
                  (CandleBuilder), with a timer that closes each bar on the clock rather than on
                  the next trade
  stream_common.py — what both live sources do once a tick lands: alert/tactics sweep, bar-close
                  publication, volume-alert latch, timestamp bucketing and bar de-duplication
  bar_history.py — the one place that decides where REST bars come from, shared by the initial
                  load, the timeframe reload, the backfill and the stream-down fallback poll so
                  one buffer never mixes feeds whose volumes differ by 26x; probes the key's SIP
                  tier (real-time / 15min-delayed / none) and degrades consolidated-first
  datalog.py    — de-duplicated console logging of which data source served each fetch
                  (WebSocket, Alpaca REST, yfinance, WorldNews) and which fallbacks were tried
  historical.py — yfinance-based historical prices, dividends, earnings dates, static analysis
  technical_analysis.py — trend/intraday/market-regime reads, volume & consolidation analysis,
                  opening-range breakout geometry, VWAP std-dev bands + ADX + reversion geometry,
                  put/call wall + gamma exposure analysis, Smart Money order blocks / fair value
                  gaps / composite setup + geometry
  options.py    — yfinance options chain fetching (open interest, Black-Scholes gamma per strike)
  charts.py     — Plotly chart builders (candlestick + volume profile, gamma, Smart Money zones,
                  performance, historical), the model-overlay renderer, and the multi-day session
                  helpers (rangebreaks collapsing off-session time, 09:30/16:00 boundary rules)
  profile_model.py — ML predicted price profile: loads the LevelsML density-model pack
                  (../Models/open_profile_lgbm.json.gz), rebuilds its at-open features from
                  daily bars + the opening print, and turns predicted volume-quantiles into
                  a smooth density for the Live chart overlay / mixture fit
  news.py       — optional LLM pipeline for news impact scoring (Alpaca + WorldNews sources)
  premarket.py  — LLM synthesis of news/historical/macro/fundamental/alternative data into a
                  structured briefing per symbol, generated automatically when the stream starts
                  and framed by the session phase it runs in (pre-open / intraday / post-close /
                  weekend). Finnhub's alt-data feeds enter as *structural* context with explicit
                  instructions not to treat a quarterly filing as an intraday catalyst
  llm.py        — unified chat-completions client over Gemini, OpenAI, and Anthropic
  observability.py — optional Langfuse tracing for the LLM pipeline (no-op if unconfigured)
  finnhub_rest.py — Finnhub's six alternative-data feeds (insider transactions & sentiment,
                  US federal contract awards, Senate lobbying, USPTO patents, H-1B visas),
                  fetched concurrently and summarised for the pre-market briefing; per-dataset
                  failures degrade rather than losing the rest
  trade_sound.py — optional audible cue on a fill: an inline Custom Component v2 that
                  synthesises a two-note chime with the Web Audio API (rising for a buy,
                  falling for a sell). No audio asset, no visible player. Off by default
  decisions.py  — independent decision ledger; fetches its own fill price per trade. On a
                  simulated broker it owns the cash balance; on a real one the Alpaca account
                  does, and the ledger is written from an account read after each order
  tactics.py    — standing conditional trade plans (`set_tactics`) and the background
                  TacticsExecutor that arms/fires them against live ticks
  automatic.py  — the orchestrator: reads the broad market once, then each ticker separately,
                  and assigns EVERY ticker its own strategy (one cycle per distinct strategy,
                  run sequentially over a shared cash balance). A stand-down re-assesses only
                  that strategy's tickers
  agent_prompts.py — what each personality is told, plus the per-run addenda: multi-symbol,
                  tactics, market-closed, the EXECUTION VENUE (local / Alpaca paper / live —
                  an agent that thinks its orders are inert reasons differently from one
                  spending real money), and the day's research briefing rendered as prose
  agent.py      — LLM trading agent loop (personalities incl. Premarket Analyst, tool calls,
                  reasoning, one decision per cycle across the whole symbol basket);
                  `stand_down` tool when run under Automatic
  automatic.py  — Automatic orchestrator: regime-detection cycle (`select_strategy`) that activates
                  and switches between strategy agents, handing off to the Premarket Analyst pre-open
  momentum_regime.py — the intraday momentum score and Schmitt-trigger regime (mirroring
                  FinNotebooks/TimeToChange2's mshift), plus the live minute grid the rule
                  agents and the chart overlays read. Tape only, no model
  dayrange_model.py — the model behind Apple Trader (FinNotebooks/TimeToChange3):
                  mirrors dayrange's daily/opening features and blends LightGBM + N-BEATS + N-HiTS
                  into one forecast of the session's high and low, made once at 09:35.
                  Also the intraday update of that forecast (`updated_range`): what a session
                  that trades through it replaces the breached side with
  intraday_vol_model.py — IntradayVolatility's export: the time-of-day volatility curve (power
                  decay + close ramp) and a daily-bar HAR forecast of the day's range, plus the
                  envelope between a day's high and low that follows that curve -- drawn whole
                  by the chart overlay, read one minute at a time by Apple Trader. JSON + numpy
  model_overlays.py — what the saved models predict, as drawing instructions: one catalogue of
                  chart overlays (day range, predicted profile range, intraday range alone and
                  × day range), each computed from bars the caller supplies, so the live chart
                  and SimLab's replay chart draw the same items. Four item kinds — a price
                  level, a moment, a stretch of time, a price range that changes with the time
                  of day — and `charts.add_model_overlays` is the only renderer
  apple_models.py — the registry of models Apple Trader can run on, so the loop, SimLab and the UI
                  ask for a model by name; each names the rule set it drives, which is the one
                  thing callers do branch on
  apple_trader.py — Apple Trader: rule-based (no LLM) loop on the day-range forecast. It rests
                  a buy and a sell at fixed distances below a reference (per-instrument
                  defaults) -- the forecast high, or the intraday band's upper curve at this
                  minute -- and rebuilds both whenever either the reference or the clock moves
  apple_rules.py — Apple Trader 2's rule language: the catalogue of signals a condition can
                  name (bar, momentum regime, day-range distances,
                  position, clock), the action item (buy/sell + size + AND/OR conditions),
                  evaluation, validation, the Results signature, and the presets (one of
                  them Apple Trader's own strategy)
  apple_rules_ui.py — the Streamlit rule builder for the above, shared by the live Agent tab
                  and SimLab's Simulate tab (widget keys are namespaced by a prefix)
  apple_trader2.py — Apple Trader 2: the same loop with the strategy taken out of the code.
                  A SignalBus computes each named signal on demand (and only if some rule
                  reads it), the rule list decides, and one order at most is sent per bar
  scoring.py    — per-session grounding/accuracy scorecard and daily (UTC day, 1hr-runtime-gated)
                  aggregate scoring report
  performance.py— replays decisions against price bars to build the equity curve
  report.py     — self-contained HTML report of an agent run
  ui.py         — Streamlit layout (Live / News / Pre-Market / Historical / Technical Analysis /
                  Smart Money / Put-Call Walls / Agent tabs), event callbacks
  clock.py      — swappable time source: wall clock live, pinned to the replayed
                  moment under SimLab (agent path reads time through here)
simlab/
  data.py       — dataset download + smart local store (minute bars from yfinance by
                  default, Alpaca iex/sip on request; gzip JSON, deduplicated per
                  feed-symbol-day so one tape never shadows another; manifest of
                  named datasets, each recording its tape)
  market.py     — time-windowed views over a stored dataset ("as of simulated t")
  engine.py     — simulation engine: real agent cycles at a pinned clock, SimBroker
                  fills from the tape, deterministic bar-by-bar fast-forward between
                  cycles (tactics / alerts / news wake-ups); a second day loop reads
                  every closed bar for the rule-based agents
  patches.py    — simulation context rerouting every live-fetch call site to the dataset
  judge.py      — LLM-as-judge: per-entry reasonableness + overall strategy adherence
                  (LLM agents only; the rule-based agents are scored on profit alone)
  results.py    — run records, oracle/profit scoring, persistence, Langfuse export
  prompts.py    — editable per-personality prompt overrides (simulation-only; a no-op
                  for agents that have no prompt)
  rule_agents.py— the rule-based (non-LLM) agents SimLab can replay, behind one interface:
                  label/ticker, config round trip, run signature, and a uniform run_cycle
  runner.py     — worker process for one queued experiment (replay, judge, persist)
  experiments.py— queued-experiment store and the process scheduler behind the pipeline
  tuning.py     — Apple Trader parameter tuning: a one- or two-parameter grid swept on a tuning
                  dataset, a pick (highest cell or best plateau) replayed on a test dataset
                  beside the base configuration; jobs run detached over a process pool
  drift.py      — model drift: each stored session scored point-in-time by a saved model
                  (day range, intraday volatility, open profile), grouped by day or week,
                  with training cutoffs and reference metrics read from the model's metadata
  app.py        — the SimLab Streamlit UI (agents / ML models / drift / datasets / simulate /
                  tuning / summary / results)
main.py         — entry point (loads .env, launches Streamlit)
sim_main.py     — SimLab entry point (streamlit run sim_main.py)
tests/          — pytest suite, mirrors most modules 1:1
```

## Running tests

```bash
pip install pytest pytest-mock requests-mock
pytest tests/ -v
```
