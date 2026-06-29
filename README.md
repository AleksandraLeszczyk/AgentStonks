<img width="1025" height="672" alt="image" src="https://github.com/user-attachments/assets/6f68cbeb-dfa3-40dc-b7dd-222f1583a07b" />


# MarketView

Real-time market data dashboard built with Streamlit and the Alpaca streaming API. Shows live candlestick bars, trade volume profile, and news for any US equity — plus historical context and an autonomous LLM paper-trading agent.

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)

## Features

### Live tab
- **Live candlestick chart** — minute bars streamed via WebSocket, seeded with REST history on start
- **Current price** — live last price with change vs previous close
- **Candle display options** — toggle open-close body, 20%–80% percentile body, and whiskers independently
- **VWAP** — display as dot markers or a continuous line
- **Volume profile** — rainbow-coded price distribution histogram showing trade density over the session
- **VWMA overlays** — volume-weighted moving averages at 5, 15, and 60 periods
- **Average lines** — 7-day, 28-day, and 1-year daily average price overlays
- **Fibonacci levels** — session high/low retracement levels
- **Price profile fit** — fit a Gaussian or Cauchy mixture model to the volume profile (1–5 components), with optional centers shown on the candle chart
- **News feed** — latest headlines from Alpaca news (falling back to WorldNews API), with orange vertical markers on the chart and optional LLM impact scoring
- **Multi-timeframe** — 1Min, 5Min, 15Min, 30Min, 1Hour, 1Day
- **IEX and SIP feeds** — switch between free (IEX) and paid (SIP) data
- **Technical analysis panel** — daily trend regime, intraday momentum, and broad-market risk environment (VIX level/trend/term structure, S&P 500 trend and drawdown), each summarized in plain language with a gauge chart
- **Options walls panel** — Call Wall / Put Wall (open-interest-based resistance/support) and net dealer gamma regime, computed from a yfinance options chain on its own independent poll loop

### Historical tab
- **Price history** — daily closes over 7 days to 5 years, plotted against SPY and VIX
- **Dividend and earnings markers** — overlaid on the historical chart
- **Static analysis** — trailing P/E, estimated annual return (growth + dividend), and estimated 10-year cumulative dividend return

### Agent tab
- **LLM paper-trading agent** — runs on a fixed cycle, reads already-fetched ticker data (bars, quotes, volume stats, news, options walls, position) via tool calls, and reasons about a trading regime and strategy
- **Agent personalities** — Swing/Position, Momentum, Breakout, VWAP Mean-Reversion, and Smart Money traders, each with its own system prompt, decision playbook, and tool set:
  - *Swing* — daily trend + market regime first, then intraday/volume confirmation and put/call walls as support/resistance
  - *Momentum* — screens for gap + relative-volume + news catalyst, trades bull flags and VWAP reclaims
  - *Breakout* — waits for a volume-confirmed opening-range break, sizes via ATR-based `breakout_trade_geometry` targets requiring a minimum 2:1 reward/risk
  - *VWAP Mean-Reversion* — gates on an ADX-confirmed range (below 20), fades 2σ stretches from session VWAP back to the mean via `analyze_vwap_bands` + `vwap_reversion_geometry` (target VWAP, stop one σ beyond entry, minimum 1.5:1 reward/risk), preferring rejection candles at the bands; long-only, so it trims/exits into upside stretches rather than shorting
  - *Smart Money (Highest-Edge)* — the composite institutional setup: identifies higher-timeframe bullish **order blocks** (`analyze_order_blocks`) and enters only when price *returns* into a demand zone during the session with intraday confirmation — a rejection candle, a filled **fair value gap** (`analyze_fair_value_gaps`), or a breaker/break-of-structure — via the composite `analyze_smart_money_setup` read. Stop sits just beyond the block; target is the next opposing structural level, sized with `smart_money_trade_geometry` to a minimum 3:1 (typically 3:1–5:1) reward/risk; long-only. The **🏦 Smart Money** tab draws the order blocks as demand/supply zones over daily candles with the entry/stop/target geometry overlaid
- **Automatic (regime-adaptive orchestrator)** — a meta-agent that detects the current market regime and activates the single best-fitting strategy above for you. Each round it reads the same analysis tools (daily trend, broad-market backdrop, intraday momentum, volume, VWAP/ADX range gate, opening range, order blocks, options walls, news) and calls `select_strategy`. It then goes to sleep and hands control to the chosen strategy, which trades normally. When that strategy judges its edge has faded (e.g. a breakout agent in a dead range, a mean-reversion agent once a trend takes hold) it calls **`stand_down`** with reasoning instead of idling on an alert — waking the orchestrator to re-assess the regime and switch to a better-suited strategy. The Agent tab status line shows the currently active strategy and detected regime
- **Provider-agnostic** — works with Gemini, OpenAI, or Anthropic, via a unified chat-completions client
- **Smart wake-ups** — when it doesn't want to trade, the agent sets condition alerts on any continuously-updated value (price, bid/ask, spread, day high/low, volume, relative volume, portfolio value) to wake early the moment one is crossed, and always wakes early on fresh news for the ticker regardless of any alerts set — there is no idle "do nothing" decision
- **Independent fill pricing** — decisions (buy/sell/alert) are handed to a separate decision tracker that fetches its own fill price, so the agent never picks the price its own trade is recorded at
- **Paper broker only** — no real orders are ever placed; each filled buy/sell costs a fixed simulated fee
- **Equity curve & performance summary** — replays recorded decisions against streamed bars to reconstruct portfolio value over time
- **Live agent log** — cycle starts, tool calls, analysis, decisions, news alerts, and errors, streamed as they happen
- **HTML report export** — generates a self-contained HTML file with starting conditions, charts, and full decision/activity history
- **Optional LLM observability** — when Langfuse credentials are set, each agent cycle is traced end-to-end (tool calls, token usage, latency) via `observability.py`; a no-op otherwise

## Quickstart

```bash
cp .env.example .env        # fill in your Alpaca credentials
pip install -r requirements.txt
streamlit run main.py
```

Open http://localhost:8501.

## Docker

```bash
docker build -t marketview .
docker run -p 8501:8501 --env-file .env marketview
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
marketview/
  config.py     — constants, color palette, agent timing/cost settings
  state.py      — shared mutable AppState (bars, trades, news, agent log, WebSocket handles)
  rest.py       — Alpaca REST helpers (fetch_bars, fetch_trades, fetch_daily_bars, fetch_news)
  stream.py     — WebSocket streaming threads for bars/trades and news
  historical.py — yfinance-based historical prices, dividends, earnings dates, static analysis
  technical_analysis.py — trend/intraday/market-regime reads, volume & consolidation analysis,
                  opening-range breakout geometry, VWAP std-dev bands + ADX + reversion geometry,
                  put/call wall + gamma exposure analysis, Smart Money order blocks / fair value
                  gaps / composite setup + geometry
  options.py    — yfinance options chain fetching (open interest, Black-Scholes gamma per strike)
  charts.py     — Plotly chart builders (candlestick + volume profile, gamma, Smart Money zones, performance, historical)
  news.py       — optional LLM pipeline for news impact scoring (Alpaca + WorldNews sources)
  llm.py        — unified chat-completions client over Gemini, OpenAI, and Anthropic
  observability.py — optional Langfuse tracing for the LLM pipeline (no-op if unconfigured)
  broker.py     — order execution abstraction (Broker / PaperBroker)
  decisions.py  — independent decision ledger; fetches its own fill price per trade
  agent.py      — LLM trading agent loop (personalities, tool calls, reasoning, one decision per cycle); `stand_down` tool when run under Automatic
  automatic.py  — Automatic orchestrator: regime-detection cycle (`select_strategy`) that activates and switches between strategy agents
  performance.py— replays decisions against price bars to build the equity curve
  report.py     — self-contained HTML report of an agent run
  ui.py         — Streamlit layout (Live / Historical / Agent tabs), event callbacks
main.py         — entry point (loads .env, launches Streamlit)
tests/          — pytest suite, mirrors most modules 1:1
```

## Running tests

```bash
pip install pytest pytest-mock requests-mock
pytest tests/ -v
```
