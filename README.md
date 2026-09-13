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
- **Model predictions on the chart** — a *Model Predictions* picker in Chart Settings draws what the trained models say about today, in the idiom each answer calls for. **Predicted day range** (TimeToChange3, made once at 09:35) and **predicted price profile range** (the LevelsML density model's outer quantiles and point of control) are horizontal lines, drawn in the candle chart *and* across the volume profile beside it, with a semi-transparent band over the session they cover. **Momentum regime changes** (TimeToChange2 — pick which of the two bundles answers) mark every change the session made with an icon, and a change the model expects to *persist* also gets a vertical line and a shaded window over the bars it should hold for; on a forecasting bundle the newest bar carries the turn probability, and its window reaches past the last bar, which is the one thing that widens the time axis. Each overlay is offered only for the symbols its model was fitted on, and a missing bundle or too little history is reported as a caption under the chart rather than a silently empty overlay
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
- **🍎 Apple Trader (rule-based, no LLM)** — the one personality with no model reasoning in the loop: once a minute it reads the minute bar that just closed and applies fixed rules, so the same tape always produces the same trades. It tracks the momentum regime the way FinNotebooks/TimeToChange2 defines it — a volatility-normalised momentum score fed through a Schmitt trigger (enter a directional regime at |mom| > 0.90, leave it only below 0.40), so a score hovering near the line cannot emit a burst of fake changes. When the bar that just closed is a regime change **into positive**, it hands the 20 bars leading into that change to **the saved model chosen for the run** and **buys** if the probability that the change will hold clears the **persistence threshold** (default: the cut-off that model chose on its own validation block). That is the whole entry — one bar, one question — because the thing a multi-bar confirmation window would wait for is exactly what the model is being asked. The **exit** is pure price: **sell** once the price is **0.5%** (configurable) below the highest price seen since the entry. The peak only ratchets up, so the rule starts as a stop under the entry and turns into a profit lock as the move runs; nothing else closes the position except the closing bell, since every feature the model uses is intraday. Two models can answer the entry question, and the rules around them are identical either way:
  - **Persistence classifier (HGB)** — the default. A gradient-boosted classifier trained on the label directly (`Models/timetochange2_persistence_<TICKER>.joblib`, TimeToChange2 notebook 04). **Fitted per ticker**, and its cut-off is picked on that ticker's own validation events, so the thresholds are not close: 0.05 on AAPL, 0.43 on GOOGL, 0.22 on INTC (they move on every retrain — the bundle is the authority, not this list). Needs the optional `scikit-learn`/`joblib` dependencies; override with `APPLE_MOMENTUM_MODEL_<TICKER>`
  - **N-BEATS forecast → persistence** — five seeds of N-BEATS forecast the next 15 bars of the momentum score, 500 sampled futures are run through the same Schmitt trigger, and the fraction that survive 15 bars is the probability (`Models/timetochange2_nbeats_<TICKER>.pt`, notebooks 06–08). **Fitted per ticker**, each with its own thresholds (0.21 on AAPL, 0.41 on GOOGL, 0.05 on INTC) and its own residual sidecar. Needs the optional `torch` dependency plus `Models/timetochange2_nbeats_<TICKER>_residuals.npz` and the metadata JSON beside the checkpoint; override with `APPLE_NBEATS_MODEL_<TICKER>`

  **Which instrument** is picked in the same panel, and the two choices constrain each other: every rule here is a saved model's output, so the picker offers only symbols something was fitted on and then only the models fitted on the symbol chosen. Picking **GOOGL** or **INTC** therefore means the day-range strategy — the momentum models are not on the menu there and pairing them is refused before the run, with the reason ("fitted on AAPL only") rather than a missing-file error. A run trades that one symbol, which must be streamed.

  Both are re-trainable into the app's own store, and have to be: the notebooks' venv pickles with scikit-learn 1.7.2 and this app runs 1.9.0, in which a 1.7.2 `HistGradientBoosting*` will not unpickle at all. `cd FinNotebooks/TimeToChange2 && ../../AgentStonks/.venv/bin/python ../../Models/train_timetochange2.py AAPL GOOGL INTC` trains both models for a symbol under notebook 08's protocol (last trading week held out of every split) in about a minute. All three symbols were last retrained this way on 2026-09-09, AAPL included — it had been the one still carrying notebook 06/07's yfinance-cache checkpoint. It is deliberately the *shipping subset* — no DeepAR, no classification heads, no walk-forward folds, no bootstrap intervals — so run it to get the file and read notebook 08 for whether the model is any good.

  Be clear about what either one is. TimeToChange2 measures 0.82 out-of-fold AUC for the classifier over all regime changes but ~**0.50** over the ones that already pass the observable “the old regime had held 15 bars” pre-condition — it separates the impossible from the possible, not the likely from the unlikely. N-BEATS is the one model in that benchmark above chance on that hard half (**0.67 ± 0.07** across four walk-forward folds, on 35 events, with an interval that only just excludes chance), which is why it is offered as a switch rather than promoted to the default; replayed over one held-out session the two made *identical* trades. Either way an entry is best read as a change the model did not veto. Those numbers are notebook 6's, measured on a 19-session yfinance cache; **every checkpoint was retrained on 2026-09-09** against the 32–36 session Data Collection archive under notebook 08's protocol, and the shape of the finding only partly survives. On a single held-out split each, the hard half now reads N-BEATS 0.67 against the classifier's 0.44 on GOOGL and 0.68 against 0.61 on INTC — but **0.45 against 0.56 on AAPL, where the ordering reverses.** So the forecast route beats the incumbent on two of three symbols, and on AAPL it is now the weaker of the two on the half of the problem that matters. That is not retraining damage: scored on the same 63 unseen events, the old and new AAPL checkpoints are indistinguishable (0.780 full-label both, 0.45 hard half both), which says the 0.713 was one small test block of 35 events and a larger, later one did not reproduce it. Treat the per-symbol numbers as a single held-out split rather than as notebook 6's four-fold walk-forward, which has not been rerun on the larger archive. A missing model or dependency is reported instead of traded around. Not selectable by the Automatic orchestrator, which drives LLM prompts

  A **third** and a **fourth** model are offered, and picking either changes the strategy rather than the answer:
  - **Day-range forecast (TimeToChange3)** — at **9:35** it forecasts where the *whole session's* high and low will land, from a year of daily bars plus the first five 1-minute bars, and then asks the model nothing else all day. Three predictors blend equally (LightGBM + N-BEATS + N-HiTS over a 32-day lookback of eight per-day channels), a heavily-regularised ridge learns what the opening five minutes add on top of the daily prediction, and the result is clipped to contain the range that has already printed. **Fitted per ticker** — as all four now are: `Models/timetochange3_dayrange_<TICKER>.joblib` (AAPL, GOOGL and INTC) plus the two `.pt` checkpoints and the metadata JSON beside each, every symbol with its own daily models, its own opening ridge and its own held-out error ($2.12, $2.75 and $1.72 respectively). Needs `torch`, `lightgbm`, `scikit-learn` and `joblib`; locations override with `APPLE_DAYRANGE_MODEL_<TICKER>`

  The rules that come with it are TimeToChange3 notebook 05's, with both levels adjustable. With **H** the predicted high and **A** the trailing 14-day average daily range in dollars, it **buys** when a bar's low reaches `H − 0.75 × A` and **sells** when a bar's high reaches `H − 0.10 × A`, re-arming as often as the day allows, with anything still open flattened before the close. There is no trailing stop, no probability and no threshold — none of them mean anything to a forecast of the day's range, and `config_signature` leaves them out so a day-range run is never filed in Results beside a momentum one. It is a mean-reversion bet by construction: what the model forecasts well is the **width** of the day, not its direction, so the rule buys well below where the day is expected to top out and sells just under it, and on a day that never dips that far it does nothing.

  What the numbers are. On a 129-session test window the blend's mean absolute error on the two targets is **0.0077 in log units ($2.12)**, against 0.0110 for a 14-day rolling baseline and 0.0135 for persistence — 30% of the baseline's error removed, and the ordering holds across all four walk-forward refits. That is a statement about the forecast, not about the trading rule: notebook 05 ran the rule across all 21 sessions with minute data (15 traded, 10 profitable, $949 total on a fresh $10,000 each) with no commissions and no slippage, which is a sanity check and not an edge. The 0.75 and 0.10 were specified rather than fitted, and the notebook's own five-session sweep is the reason not to retune them off a small sample: the week total peaks elsewhere, but the *count* of profitable sessions is flat at three in five across the whole region where the rule trades at all — the levels change the price paid on the same winning days, not how often the rule is right.

  Two things the live path does differently from the notebook, both worth knowing before comparing results. The **fill**: the notebook rests limit orders and fills a touch *at* the level, while this ledger is market-order only and buys near the close of the bar that touched it, so a bar that dipped and recovered fills worse here — a real cost, and it runs one way. And the **minute tape's volume scale**: the opening ridge's `or_volume_share` feature was fitted on consolidated volume, and Alpaca's IEX feed carries about 4% of it, which puts that one feature outside anything it saw in training. Run the stream and any SimLab dataset on a consolidated tape (`yfinance` or `sip`); the app says so when it does not.

  - **Delta-momentum regressor (TimeToChange)** — the fourth model and the third strategy, wired up for **AAPL and INTC**. It predicts a *quantity* rather than a probability: given the minute that just closed, how many **bps/min** the smoothed 15-minute momentum score will have moved fifteen bars from now (`mom[t+15] − mom[t−1]`). One regressor per ticker, and the estimator is *selected* per ticker on that ticker's own validation days rather than fixed — **Ridge** for AAPL, **HistGradientBoosting** for INTC — over the same 26 point-in-time features TimeToChange's notebooks 02-03 settled on. It is fitted for **GOOGL** too (a RandomForest, R² 0.48 validation / 0.19 holdout — the weakest of the three and the one that fell furthest between the two), and that bundle is installed in `Code/Models` again after the refit on the full weekly archive; the ticker is still out of `apple_models.MOMENTUM_CHANGE_TICKERS`, which is the reversible half of the rule that took it out. A model listed in the registry with no file behind it is not a wider menu, it is an agent that reports itself broken whenever somebody picks it — a file no entry points at costs nothing, so re-adding GOOGL is one entry there plus the tests that use the pairing as their never-fitted fixture. Needs `scikit-learn` and `joblib`; `Models/momentum_change_<TICKER>.joblib` with a readable `.json` sidecar beside it (the name `momlib` itself writes, so a retrain lands where the app looks), overridable with `APPLE_MOMENTUM_CHANGE_MODEL_<TICKER>`

  The rules are TimeToChange notebook 05's, all four thresholds adjustable. The tape picks the situation and the model picks the direction: **buy** when the previous minute's regime is **negative** and the prediction is `≥ 0.30` bps/min, **sell** when it is **positive** and the prediction is `≤ −0.30`, with two risk exits underneath — a **momentum floor** at `−2 × θ` (θ being the day's regime threshold, set from yesterday's minute volatility) and a fixed **0.5% stop** below the entry. No trailing stop, no probability, no entry mode; as with the day-range rules, `config_signature` leaves out everything the strategy does not read.

  What the numbers are, and they are mixed on purpose. All three transfer: on the **holdout week** (2026-08-24…28, five sessions removed before either fitting or selection) AAPL scores R² **0.48**, GOOGL **0.19** and INTC **0.43** against a predict-zero baseline, and the *sign* of the prediction is right on **94%**, **91%** and **86%** of that week's regime changes. The timing is the weak half — over every minute of an unseen day the absolute prediction separates "near a change" from "quiet" at AUC ~0.53, barely better than chance — which is exactly why the entry is gated on a regime the tape has already printed. The **rules** are a different matter: over that week they made −$80.26 on $10k on GOOGL and +$60.35 on INTC (buy-and-hold: +$123.50 and −$118.36), the ablations say the model's *exits* are the only profitable component on either ticker, dropping the model entirely beats the full rules on INTC, and 0.5 bp per side turns both negative. Sweep them in SimLab; five sessions per ticker is a sanity check, not an edge.

  Three things this one does differently from every other model here. It is the only one that reads **minute bars from days before the session being traded** — six of them, because the regime threshold is yesterday's minute volatility and the multi-day features reach back six sessions — fetched once each morning and, in SimLab, read out of the dataset's own stored days; a run without that much history behind it logs a refusal per session instead of scoring on a threshold it invented. Its **event history is `persist` (15) bars stale by construction**, since a change is only persistent once the new regime has held that long, where training used no lag at all. If you compare anything here against the notebooks, pass `momentum_change_model.live_confirm_lag(bundle)` and not `persist`: momlib's `confirm_lag=N` re-stamps an event at the bar N later and the history lookup takes events *strictly* before the current bar, so N models a lag of N+1 and the live equivalent is **`persist − 1` = 14**. The off-by-one hides on the two tree bundles, which quantise the difference away on most bars; AAPL's Ridge is what exposed it. And its bundle is **re-fitted rather than copied** — the notebooks' venv pickles with scikit-learn 1.7.2 and this app runs 1.9.0, in which those files will not load at all — by running TimeToChange's own training script with this project's interpreter: `cd FinNotebooks/TimeToChange && ../../AgentStonks/.venv/bin/python scripts/train_ticker.py AAPL GOOGL INTC --model-dir ../../Models`. Same data, same code, same split, same estimator selected; the metrics move in the third decimal. The AAPL bundle in `FinNotebooks/Models` is left alone deliberately — it is notebook 03's demo (a month of yfinance bars, no reserved holdout week) and notebooks 04-05 were executed against it, so the app reads the `Code/Models` one trained under the same protocol as INTC's
- **🍏 Apple Trader 2 (adjustable rules, no LLM)** — the same fixed loop over one symbol's minute bars, the same ledger and the same flatten-before-close, with the strategy taken out of the code. Instead of picking one of two hard-coded strategies by picking a model, a run is configured with an **instrument** and a **list of action items**, both built in the Agent tab (and in SimLab's Simulate tab, where the list stands in for a prompt). Each item is:
  - **an action** — buy or sell;
  - **a size** — a **percentage** (of cash on a buy, of the position on a sell), a **dollar amount**, or a **share count**. Every mode is clipped to what the ledger can do, so "sell 200 shares" out of 50 sells 50 and a rule set is portable across starting balances;
  - **one or more conditions**, each a named signal against a number (`above` = ≥, `below` = ≤), joined by **AND** (all of them) or **OR** (any of them). One joiner per item on purpose: `A and B or C` has no meaning without precedence rules, and a genuine mix is two items;
  - optionally a **cooldown** in bars, so a rule whose condition stays true ladders in only when that was the intent (390 ≈ one session, i.e. "once a day"), and an on/off switch.

  The **signals** a condition can read are the whole vocabulary, and the models appear here as signals rather than as strategies — a rule set may name two of them at once: the **bar** (close, high, low, change on the day); the **momentum regime** (score, regime, bars in regime, the dwell of the regime just left, whether this bar is a change into positive); the **model forecasts** (`persistence.proba`, `nbeats.proba`, `nbeats.turn_proba`, `nbeats.reversal_proba`); the **day-range forecast** (predicted high/low, the 14-day ADR, and the distance from this bar's low/high/close to the predicted high **in ADRs**, which is the form notebook 05's rule is written in); the **position** (shares, open P&L, give-back from the peak since entry — `below -0.5` is a 0.5% trailing stop — bars held, value, cash, and **the change since the last buy**, measured against `max(that fill price, every high since)`, which unlike the position give-back survives the exit and so can arm a re-entry: "buy again once price is 1% below the best price since I last bought"); and the **clock** (minutes since the open, minutes to the close).

  **Which of those exist depends on the instrument**, because a model is fitted on a symbol and none of the notebooks claims transfer. **AAPL**, **GOOGL** and **INTC** each carry both TimeToChange2 momentum models and the day-range forecast, so all three get the whole catalogue; **any other symbol** gets the model-free half — the bar, the momentum regime, the position and the clock, all computed here from the tape and meaning the same thing everywhere. The picker offers the modelled symbols plus whatever the app is streaming (or, in SimLab, whatever the selected datasets carry) and accepts anything typed into it. Switching the instrument never rewrites the rules: conditions that no longer read are marked in place and the set is refused as unrunnable until they are gone, and the presets narrow the same way (every instrument keeps at least the model-free one). Adding a symbol is one entry in `apple_models.DAYRANGE_TICKERS` plus its bundle in `Code/Models` — ORCL is trained in `FinNotebooks/Models` and not wired up. Note the TimeToChange delta-momentum regressor (AAPL and INTC) has **no signal in this catalogue yet**: only Apple Trader runs it.

  What the engine adds around whatever list it is handed: **at most one action per closed bar** and **the first matching rule wins**, so the order is the priority — but a rule that matches and *cannot transact* (a sell with the book flat, a buy with no cash) is passed over rather than eating the bar, so an exit written above an entry never blocks it. **An absent signal never matches**, in an AND or an OR alike: a model asked about a bar it was not fitted for, a day-range forecast before 9:35, a P&L with no position all read as nothing, and nothing fires a rule. **The closing flatten is not a rule** and cannot be deleted — every signal here is intraday. And **nothing is computed that no rule reads**: the model bundles loaded are the ones the conditions name (a rule set written on price and momentum loads none at all and needs no saved artifacts), the day's high/low forecast is made only if something asks for it, and conditions short-circuit within a rule.

  Four **presets** ship, and the first three are Apple Trader's own strategies written in this vocabulary — anticipate, confirm, and the day-range levels. They are reproductions rather than approximations: replayed against Apple Trader on the 2026-07-27 SIP session through the real saved bundles, each produced the *same trades* — same minutes, same quantities, same fills. That is what makes them worth loading as a starting point, since a rule set can then be compared against the thing it was meant to improve on. The fourth ("scale in, take half off, trail the rest") is not a recommendation and is not measured; it is the shortest list that says something the first agent structurally cannot. Nothing here tunes or validates a rule set — the vocabulary makes it just as easy to write one that fires every bar or holds a position no exit can reach; `ruleset_error` refuses only the ones that cannot work at all (no enabled rules, a rule with no conditions, a set that can only sell, a model that cannot answer a signal it was asked for), and SimLab is where the rest gets answered
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

- **Agents tab** — every personality with its avatar, an editable system prompt (saved overrides apply only to simulations; the live app keeps the built-in), and the agent's exact tool set, each tool runnable by hand against any stored moment (pick dataset + symbol + time, see the JSON the agent would see). Apple Trader has no prompt or tools to edit, so its card shows both its rule sets plus the provenance and held-out metrics of each saved model it can be pointed at — in each model's own units, since a day-range forecast's dollar error and a persistence classifier's AUC are not comparable numbers, and one block per symbol the model was fitted on, since GOOGL's bundle is a different model from AAPL's rather than the same one on another tape. Apple Trader 2's card has no strategy to describe at all, so it documents the **vocabulary** instead: every signal a condition can read, and the rules the engine applies around whatever list it is handed
- **Datasets tab** — download named datasets (symbols + date range + **feed**): minute bars 04:00–20:00 ET, ~1.5y of daily history, per-day news, and SPY/VIX/VIX3M context, stored as gzip JSON under `data/simlab/store/` deduplicated per (feed, symbol, day) — overlapping datasets never re-download a day. **1-minute bars come from yfinance** (the default feed): consolidated-tape OHLCV across every venue, free, no Alpaca data subscription, and the same source the live volume tools read, so a simulated volume ratio is like-for-like with the live one. Two limits come with it — Yahoo serves 1-minute history for the **last 30 days only** (earlier days download empty), and it publishes no per-bar VWAP, so yfinance bars carry no `vw` and the VWAP note/line is absent on those runs; extended-hours minutes carry real prices but are reported at zero volume. Alpaca's `iex` and `sip` remain selectable — for a window older than 30 days, or to re-run a dataset already downloaded on them. The feed is part of the data, not a download setting: `iex` is one venue (~4% of consolidated volume on a large cap), so its bars carry different closes, far smaller volumes and occasionally an extra or missing minute, and any agent whose rules are thresholds over those bars can trade a different day on each tape. The same day on two feeds is stored twice, deliberately, and every run record states the tape it read
- **Simulate tab** — pick agent + dataset + day(s) + provider/model and run: a pinned simulated clock (`agent_stonks/clock.py`) drives real agent cycles at historical moments, `SimBroker` fills at the stored tape price, and live fetches are rerouted to the dataset (`simlab/patches.py`). Results: equity curve, per-symbol candlesticks with trade markers, the full decision ledger and agent log, an **oracle ceiling** (best single round trip available on the tape) with profit efficiency against it, and an **LLM judge** that grades every entry on the information available at entry time (outcome shown only to calibrate) plus an overall strategy-adherence review. Runs persist under `data/simlab/runs/`; with Langfuse configured, cycles are traced and run scores (`sim-return-pct`, `sim-profit-efficiency`, `sim-judge-overall`) are registered there
- **Apple Trader in SimLab** — the rule-based agent replays on the same engine, the same tape and the same ledger, but on its own day loop: it reads *every* closed bar of the session instead of sleeping on armed conditions, so there is nothing to fast-forward past. It needs no LLM, so it is queued once per dataset rather than once per LLM model, and its **rule set stands in for a model name** — swapping the classifier for the N-BEATS forecaster, retuning the persistence threshold or the trailing stop, arming the forecast-reversal exit, or moving either day-range level, is a new configuration to test, not a repeat of one already run. Running one dataset through both momentum models is a straight comparison of them, since everything before the entry question is identical; running it through the day-range model instead compares two strategies rather than two models, and only the fields that strategy actually reads appear in its signature. **It is never scored by the LLM judge**: the judge grades an agent's stated reasoning against the tape it cited, and this one states none of its own, so profit, profit efficiency and the oracle ceiling are its whole scorecard. It requires the symbol its configuration names in the dataset — checked before anything is queued, and part of the signature, so the same day-range levels over AAPL and over GOOGL are two configurations to compare rather than one averaged row. For the momentum rules a single day is enough, since every indicator they consume is session-local and nothing crosses the overnight gap. The other two strategies are the exceptions. The day-range rules read the dataset's stored **daily** history too (a 252-day window, well inside the 420 days a dataset stores), plus the simulated day's official opening print — the one field of that day's stored daily bar exposed during the run, since the 9:30 auction is fixed before the forecast is made. The delta-momentum rules read the dataset's stored **minute** bars for the **six sessions before** each simulated day, because the regime threshold is yesterday's minute volatility and the multi-day features reach back six sessions; a dataset therefore needs six sessions of run-up before the first day it is meant to trade, and a day without them is refused in the log rather than traded on an invented threshold

- **Continuous multi-day price charts** — a run covering several days spends two thirds of its time axis on nights and weekends, which a time axis draws as blank space. Those stretches are removed from the axis instead (computed from the bars themselves, so a weekend, a holiday or a half-day is just one longer gap and needs no special case; an intraday hole where nobody traded is left alone, since collapsing it would make the axis lie about how long a move took), and the day boundary they used to provide is put back as a rule at **09:30** and **16:00** ET, the open labelled with its date. Extended-hours bars then read as what they are: the stretch outside the rules. A session the bars never reach — a run that stopped in the pre-market — is given no opening bell it never saw
- **Model predictions over a replayed day** — the per-symbol chart in Results carries the same *Model predictions* picker as the live chart, and the same three overlays, drawn over the tape that actually happened. Offered on **every** run, not only the ones a model drove: what a model would have said about a session is as worth seeing on an LLM agent's day as on Apple Trader's, and nothing here reads the run's own decisions. Each day is scored from the state of the world at its 09:31 — completed daily bars strictly before it, plus that day's stored opening print — so a replay shows the forecast that was makeable that morning rather than one built off the outcome. On a multi-day run each session's levels are drawn over that session only

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
| `APPLE_MOMENTUM_MODEL` | (optional) path to the **AAPL** momentum-persistence bundle — defaults to `../Models/timetochange2_persistence_AAPL.joblib`. It names one file, so it stands in for AAPL's bundle only |
| `APPLE_MOMENTUM_MODEL_<TICKER>` | (optional) path to another ticker's persistence bundle (e.g. `APPLE_MOMENTUM_MODEL_GOOGL`) — defaults to `../Models/timetochange2_persistence_<TICKER>.joblib` |
| `APPLE_NBEATS_MODEL` | (optional) path to the **AAPL** N-BEATS checkpoint — defaults to `../Models/timetochange2_nbeats_AAPL.pt`; the metadata JSON and the residual sidecar (`<stem>_residuals.npz`) are read from beside it. One file, so it answers for AAPL only |
| `APPLE_NBEATS_MODEL_<TICKER>` | (optional) path to another ticker's N-BEATS checkpoint (e.g. `APPLE_NBEATS_MODEL_INTC`) — defaults to `../Models/timetochange2_nbeats_<TICKER>.pt` |
| `APPLE_DAYRANGE_MODEL` | (optional) path to the **AAPL** TimeToChange3 day-range bundle Apple Trader can use instead — defaults to `../Models/timetochange3_dayrange_AAPL.joblib`; the two `.pt` checkpoints and the metadata JSON are read from beside it under the same stem. It names one file, so it stands in for AAPL's bundle only |
| `APPLE_DAYRANGE_MODEL_<TICKER>` | (optional) path to another ticker's day-range bundle (e.g. `APPLE_DAYRANGE_MODEL_GOOGL`) — defaults to `../Models/timetochange3_dayrange_<TICKER>.joblib`, one bundle per symbol the model was fitted on |
| `APPLE_MOMENTUM_CHANGE_MODEL` | (optional) path to the **AAPL** TimeToChange delta-momentum bundle — defaults to `../Models/momentum_change_AAPL.joblib`, with its `.json` sidecar beside it. It names one file, so like every bare override here it stands in for the default ticker's bundle only |
| `APPLE_MOMENTUM_CHANGE_MODEL_<TICKER>` | (optional) path to another ticker's delta-momentum bundle (e.g. `APPLE_MOMENTUM_CHANGE_MODEL_INTC`) — defaults to `../Models/momentum_change_<TICKER>.joblib` |

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
  persistence_model.py — momentum-persistence classifier (FinNotebooks/TimeToChange2): mirrors
                  mshift's momentum score, hysteresis regimes and 25 causal features, loads
                  ../Models/timetochange2_persistence_<TICKER>.joblib (standing in for the notebooks' package so the
                  pickled pipeline resolves), and scores the 20-bar sequence ending at a regime change
  nbeats_model.py — the alternative brain: mirrors mshift's N-BEATS ensemble, forecasts 15 bars
                  of momentum, bootstraps 500 futures from the exported training residuals and
                  replays the regime trigger over them to get the same persistence probability.
                  One checkpoint per ticker, each with its own residual sidecar and threshold
  dayrange_model.py — the third brain, and a different question (FinNotebooks/TimeToChange3):
                  mirrors dayrange's daily/opening features and blends LightGBM + N-BEATS + N-HiTS
                  into one forecast of the session's high and low, made once at 09:35
  momentum_change_model.py — the fourth brain, and the one whose estimator is chosen per ticker
                  (FinNotebooks/TimeToChange): mirrors momlib's momentum, adaptive per-day regime
                  threshold, persistent-change detection and 26 point-in-time features, and
                  predicts the *size* of the next momentum move in bps/min, per ticker.
                  The only model here that reads days before the session — six sessions of
                  minute bars behind today, through the fetch SimLab patches
  model_overlays.py — what the saved models predict, as drawing instructions: one catalogue of
                  chart overlays (day range, predicted profile range, momentum regime changes),
                  each computed from bars the caller supplies, so the live chart and SimLab's
                  replay chart draw the same items. Three item kinds — a price level, a moment,
                  a stretch of time — and `charts.add_model_overlays` is the only renderer
  apple_models.py — the registry of models Apple Trader can run on, so the loop, SimLab and the UI
                  ask for a model by name; each names the rule set it drives, which is the one
                  thing callers do branch on
  apple_trader.py — Apple Trader: rule-based (no LLM) loop, in three strategies the chosen model
                  selects between. The momentum rules buy a change into positive momentum the
                  model expects to persist and exit on a trailing stop (or a forecast reversal);
                  the day-range rules rest a buy and a sell at fixed distances below the
                  forecast high and hold them all session; the delta-momentum rules buy a
                  negative regime the model expects to turn up and sell a positive one it
                  expects to turn down, under a momentum floor and a fixed stop
  apple_rules.py — Apple Trader 2's rule language: the catalogue of signals a condition can
                  name (bar, momentum regime, each model's forecasts, day-range distances,
                  position, clock), the action item (buy/sell + size + AND/OR conditions),
                  evaluation, validation, the Results signature, and the presets that
                  reproduce Apple Trader's own strategies
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
  app.py        — the four-tab SimLab Streamlit UI (agents / datasets / simulate / results)
main.py         — entry point (loads .env, launches Streamlit)
sim_main.py     — SimLab entry point (streamlit run sim_main.py)
tests/          — pytest suite, mirrors most modules 1:1
```

## Running tests

```bash
pip install pytest pytest-mock requests-mock
pytest tests/ -v
```
