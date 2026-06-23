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
- **Gaussian price profile fit** — fit a mixture model to the volume profile (1–5 components), with optional centers shown on the candle chart
- **News feed** — latest headlines from Alpaca news (falling back to WorldNews API), with orange vertical markers on the chart and optional LLM impact scoring
- **Multi-timeframe** — 1Min, 5Min, 15Min, 30Min, 1Hour, 1Day
- **IEX and SIP feeds** — switch between free (IEX) and paid (SIP) data

### Historical tab
- **Price history** — daily closes over 7 days to 5 years, plotted against SPY and VIX
- **Dividend and earnings markers** — overlaid on the historical chart
- **Static analysis** — trailing P/E, estimated annual return (growth + dividend), and estimated 10-year cumulative dividend return

### Agent tab
- **LLM paper-trading agent** — runs on a fixed cycle, reads already-fetched ticker data (bars, quotes, volume stats, news, position) via tool calls, and reasons about a trading regime and strategy
- **Provider-agnostic** — works with Gemini, OpenAI, or Anthropic, via a unified chat-completions client
- **Smart wake-ups** — instead of sleeping blind between cycles, the agent can set a price alert (low, high, or both) to wake early if price crosses a watched level, and always wakes early on fresh news for the ticker
- **Independent fill pricing** — decisions (buy/sell/sleep/alert) are handed to a separate decision tracker that fetches its own fill price, so the agent never picks the price its own trade is recorded at
- **Paper broker only** — no real orders are ever placed; each filled buy/sell costs a fixed simulated fee
- **Equity curve & performance summary** — replays recorded decisions against streamed bars to reconstruct portfolio value over time
- **Live agent log** — cycle starts, tool calls, analysis, decisions, news alerts, and errors, streamed as they happen
- **HTML report export** — generates a self-contained HTML file with starting conditions, charts, and full decision/activity history

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
  charts.py     — Plotly chart builders (candlestick + volume profile, performance, historical)
  news.py       — optional LLM pipeline for news impact scoring (Alpaca + WorldNews sources)
  llm.py        — unified chat-completions client over Gemini, OpenAI, and Anthropic
  broker.py     — order execution abstraction (Broker / PaperBroker)
  decisions.py  — independent decision ledger; fetches its own fill price per trade
  agent.py      — LLM trading agent loop (tool calls, reasoning, one decision per cycle)
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
