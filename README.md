<img width="1025" height="672" alt="image" src="https://github.com/user-attachments/assets/6f68cbeb-dfa3-40dc-b7dd-222f1583a07b" />


# AgentStonks

Real-time market data dashboard built with Streamlit and the Alpaca streaming API. Tracks a whole basket of US equities at once — live candlestick bars, trade volume profile, and news — plus historical context and an autonomous LLM paper-trading agent that trades the basket from a single shared cash balance.

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)

## Features

### Multi-symbol basket
Enter any number of tickers in the sidebar; every panel and the agent operate across the whole basket, each symbol tracked by its own `SymbolState` (bars, trades, news, position) sharing one `AppState` and one paper cash balance.

### 📡 Live tab
- **Live candlestick chart** — minute bars streamed via WebSocket, seeded with REST history on start
- **Current price** — live last price with change vs previous close
- **Candle display options** — toggle open-close body, 20%–80% percentile body, and whiskers independently
- **VWAP** — display as dot markers or a continuous line
- **Volume profile** — rainbow-coded price distribution histogram showing trade density over the session
- **VWMA overlays** — volume-weighted moving averages at 5, 15, and 60 periods
- **Average lines** — 7-day, 28-day, and 1-year daily average price overlays
- **Fibonacci levels** — session high/low retracement levels
- **Price profile fit** — fit a Gaussian or Cauchy mixture model to the volume profile (1–5 components), with optional centers shown on the candle chart
- **Multi-timeframe** — 1Min, 5Min, 15Min, 30Min, 1Hour, 1Day
- **IEX and SIP feeds** — switch between free (IEX) and paid (SIP) data

### 📰 News tab
- Latest headlines from Alpaca news (falling back to WorldNews API), with orange vertical markers on the Live chart and optional LLM impact scoring, filterable per symbol

### 🌅 Pre-Market tab
- **Premarket briefing** — an LLM synthesis of recent news, historical price context, macro indicators, and fundamentals into a structured morning briefing (catalysts, technical levels, outlook) per symbol, generated on demand before the session opens

### 🗂️ Historical tab
- **Price history** — daily closes over 7 days to 5 years, plotted against SPY and VIX
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
- **Agent personalities** — Momentum, Breakout, VWAP Mean-Reversion, Smart Money, and Premarket Analyst, each with its own system prompt, decision playbook, and tool set (default: Momentum):
  - *Momentum* — screens for gap + relative-volume + news catalyst, trades bull flags and VWAP reclaims
  - *Breakout* — waits for a volume-confirmed opening-range break, sizes via ATR-based `breakout_trade_geometry` targets requiring a minimum 2:1 reward/risk
  - *VWAP Mean-Reversion* — gates on an ADX-confirmed range (below 20), fades 2σ stretches from session VWAP back to the mean via `analyze_vwap_bands` + `vwap_reversion_geometry` (target VWAP, stop one σ beyond entry, minimum 1.5:1 reward/risk), preferring rejection candles at the bands; long-only, so it trims/exits into upside stretches rather than shorting
  - *Smart Money (Highest-Edge)* — the composite institutional setup: identifies higher-timeframe bullish **order blocks** (`analyze_order_blocks`) and enters only when price *returns* into a demand zone during the session with intraday confirmation — a rejection candle, a filled **fair value gap** (`analyze_fair_value_gaps`), or a breaker/break-of-structure — via the composite `analyze_smart_money_setup` read. Stop sits just beyond the block; target is the next opposing structural level, sized with `smart_money_trade_geometry` to a minimum 3:1 (typically 3:1–5:1) reward/risk; long-only
  - *Premarket Analyst* — a one-shot pre-open specialist, gated to a window just before the bell and never selectable by the Automatic orchestrator once the session is live: it reads the premarket briefing and arms opening Tactics instead of trading directly, then retires once those tactics execute or the session starts
- **Automatic (regime-adaptive orchestrator)** — a meta-agent that detects the current market regime and activates the single best-fitting strategy above for you. Before the session opens it deterministically hands off to the Premarket Analyst; during the session it reads the same analysis tools (daily trend, broad-market backdrop, intraday momentum, volume, VWAP/ADX range gate, opening range, order blocks, options walls, news) and calls `select_strategy`. It then goes to sleep and hands control to the chosen strategy, which trades normally. When that strategy judges its edge has faded (e.g. a breakout agent in a dead range, a mean-reversion agent once a trend takes hold, a momentum agent after the move and volume dry up) it calls **`stand_down`** with reasoning instead of idling on an alert — waking the orchestrator to re-assess the regime and switch to a better-suited strategy. The Agent tab status line shows the currently active strategy and detected regime
- **Tactics (standing conditional trade plans)** — instead of trading at the current price, the agent can arm one or more conditional actions via `set_tactics` (e.g. "buy 10 shares if last_price below X", "sell 20% of the position if last_price above Y and vix below Z"). A background executor evaluates armed tactics against every live tick and fires the first action whose conditions all hold, through the same decision-tracker path as a manual agent decision — then disarms the set and wakes the agent to reevaluate with the fill in hand
- **Provider-agnostic** — works with Gemini, OpenAI, or Anthropic, via a unified chat-completions client
- **Smart wake-ups** — when it doesn't want to trade, the agent sets condition alerts on any continuously-updated value (price, bid/ask, spread, day high/low, volume, relative volume, portfolio value) to wake early the moment one is crossed, and always wakes early on fresh news for any symbol in the basket regardless of any alerts set — there is no idle "do nothing" decision
- **Independent fill pricing** — decisions (buy/sell/alert) are handed to a separate decision tracker that fetches its own fill price, so the agent never picks the price its own trade is recorded at
- **Paper broker only** — no real orders are ever placed; each filled buy/sell costs a fixed simulated fee
- **Equity curve & performance summary** — replays recorded decisions against streamed bars to reconstruct portfolio value over time
- **Live agent log** — cycle starts, tool calls, analysis, decisions, news alerts, and errors, streamed as they happen
- **HTML report export** — generates a self-contained HTML file with starting conditions, charts, and full decision/activity history
- **Daily accuracy scoring** — a per-session scorecard accumulates a deterministic grounding check (every number the model states in a decision must trace back to a number it was actually shown) plus tool errors and tactics-validation rejections; at most once per UTC day, and only once the day has accumulated an hour of agent runtime, these are aggregated into a scoring report and, when Langfuse is configured, registered there as a `daily-grounding` score
- **Optional LLM observability** — when Langfuse credentials are set, each agent cycle is traced end-to-end (tool calls, token usage, latency) via `observability.py`; a no-op otherwise
- **Data-source logging** — every fetch (WebSocket stream, Alpaca REST, yfinance, WorldNews) logs which source served the data and which fallbacks were tried, de-duplicated so repeated identical outcomes don't flood the console

## Quickstart

```bash
cp .env.example .env        # fill in your Alpaca credentials
pip install -r requirements.txt
streamlit run main.py
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
| `GEMINI_API_KEY` | (optional) Gemini key — for LLM news scoring and/or the trading agent |
| `OPENAI_API_KEY` | (optional) OpenAI key — for LLM news scoring and/or the trading agent |
| `ANTHROPIC_API_KEY` | (optional) Anthropic key — for LLM news scoring and/or the trading agent |
| `WORLD_NEWS_API_KEY` | (optional) WorldNews API key, used as a fallback news source |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | (optional) Langfuse keys — enables agent cycle tracing; no-op if unset |
| `LANGFUSE_HOST` | (optional) Langfuse host — Langfuse Cloud is used if unset |

Credentials can also be entered directly in the sidebar (Alpaca) or the Agent tab (LLM provider); env vars are used as fallback. At least one LLM provider key is required to use the Agent tab or LLM news impact scoring.

> **Note:** Free Alpaca accounts have access to the IEX feed only during US market hours (9:30–16:00 ET).

## Project layout

```
agent_stonks/
  config.py     — constants, color palette, agent timing/cost settings
  state.py      — shared mutable AppState (per-symbol SymbolState: bars, trades, news, position;
                  agent log, WebSocket handles, shared paper cash balance)
  market_hours.py — US regular-session clock (09:30-16:00 ET, Mon-Fri), used to gate the
                  Premarket Analyst and Automatic orchestrator's pre-open handoff
  rest.py       — Alpaca REST helpers (fetch_bars, fetch_trades, fetch_daily_bars, fetch_news)
  stream.py     — WebSocket streaming threads for bars/trades and news
  datalog.py    — de-duplicated console logging of which data source served each fetch
                  (WebSocket, Alpaca REST, yfinance, WorldNews) and which fallbacks were tried
  historical.py — yfinance-based historical prices, dividends, earnings dates, static analysis
  technical_analysis.py — trend/intraday/market-regime reads, volume & consolidation analysis,
                  opening-range breakout geometry, VWAP std-dev bands + ADX + reversion geometry,
                  put/call wall + gamma exposure analysis, Smart Money order blocks / fair value
                  gaps / composite setup + geometry
  options.py    — yfinance options chain fetching (open interest, Black-Scholes gamma per strike)
  charts.py     — Plotly chart builders (candlestick + volume profile, gamma, Smart Money zones, performance, historical)
  news.py       — optional LLM pipeline for news impact scoring (Alpaca + WorldNews sources)
  premarket.py  — LLM synthesis of news/historical/macro/fundamental data into a structured
                  pre-open briefing (catalysts, technical levels, outlook) per symbol
  llm.py        — unified chat-completions client over Gemini, OpenAI, and Anthropic
  observability.py — optional Langfuse tracing for the LLM pipeline (no-op if unconfigured)
  broker.py     — order execution abstraction (Broker / PaperBroker)
  decisions.py  — independent decision ledger; fetches its own fill price per trade
  tactics.py    — standing conditional trade plans (`set_tactics`) and the background
                  TacticsExecutor that arms/fires them against live ticks
  agent.py      — LLM trading agent loop (personalities incl. Premarket Analyst, tool calls,
                  reasoning, one decision per cycle across the whole symbol basket);
                  `stand_down` tool when run under Automatic
  automatic.py  — Automatic orchestrator: regime-detection cycle (`select_strategy`) that activates
                  and switches between strategy agents, handing off to the Premarket Analyst pre-open
  scoring.py    — per-session grounding/accuracy scorecard and daily (UTC day, 1hr-runtime-gated)
                  aggregate scoring report
  performance.py— replays decisions against price bars to build the equity curve
  report.py     — self-contained HTML report of an agent run
  ui.py         — Streamlit layout (Live / News / Pre-Market / Historical / Technical Analysis /
                  Smart Money / Put-Call Walls / Agent tabs), event callbacks
main.py         — entry point (loads .env, launches Streamlit)
tests/          — pytest suite, mirrors most modules 1:1
```

## Running tests

```bash
pip install pytest pytest-mock requests-mock
pytest tests/ -v
```
