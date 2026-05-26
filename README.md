# MarketView

Real-time market data dashboard built with Gradio and the Alpaca streaming API. Shows live candlestick bars, trade volume profile, and news for any US equity.

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)

## Features

- **Live candlestick chart** — minute bars streamed via WebSocket, seeded with REST history on start
- **Volume profile** — rainbow-coded price distribution histogram showing trade density over the session
- **News feed** — latest headlines from Alpaca news, with orange vertical markers on the chart
- **Multi-timeframe** — 1Min, 5Min, 15Min, 30Min, 1Hour, 1Day
- **IEX and SIP feeds** — switch between free (IEX) and paid (SIP) data

## Quickstart

```bash
cp .env.example .env        # fill in your Alpaca credentials
pip install -r requirements.txt
python main.py
```

Open http://localhost:7860 — default credentials are `user` / `pass` (change in `main.py`).

## Docker

```bash
docker build -t marketview .
docker run -p 7860:7860 --env-file .env marketview
```

## Configuration

| Env var | Description |
|---|---|
| `ALPACA_API_KEY` | Alpaca API key ID |
| `ALPACA_SECRET` | Alpaca secret key |
| `GEMINI_API_KEY` | (optional) Gemini key for LLM news scoring |
| `WORLD_NEWS_API_KEY` | (optional) WorldNews API key |

Credentials can also be entered directly in the UI; env vars are used as fallback.

> **Note:** Free Alpaca accounts have access to the IEX feed only during US market hours (9:30–16:00 ET).

## Project layout

```
marketview/
  config.py    — constants, color palette, session start time
  state.py     — shared mutable AppState (bars, trades, news, WebSocket handles)
  rest.py      — Alpaca REST helpers (fetch_bars, fetch_trades, fetch_news)
  stream.py    — WebSocket streaming threads for bars/trades and news
  charts.py    — Plotly chart builders (candlestick + volume profile)
  news.py      — optional LLM pipeline for news impact scoring
  ui.py        — Gradio layout, event callbacks, news HTML renderer
main.py        — entry point (loads .env, launches Gradio)
tests/         — pytest suite
```

## Running tests

```bash
pip install pytest pytest-mock requests-mock
pytest tests/ -v
```
