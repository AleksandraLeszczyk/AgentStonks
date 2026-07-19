"""The simulation context: reroute every live-fetch call site to the dataset.

`simulation_context(market)` is the single place that knows which parts of
``agent_stonks`` would otherwise touch the network (or the wall clock) during
an agent cycle, and swaps each one for a dataset-backed equivalent for the
duration of a simulation:

- ``clock``                       -> pinned by the engine per step (cleared here on exit)
- ``agent.fetch_bars_window``     -> stored minute bars (opening-range recovery)
- ``agent.fetch_corporate_actions`` -> none scheduled (not part of the dataset)
- ``historical.fetch_market_indicators`` -> stored SPY/VIX/VIX3M closes,
  clipped to the simulated date (also covers ``tactics.fetch_vix_level`` and
  the ``vix`` tactic condition, which read through the same function)
- ``historical.fetch_analyst_targets`` / ``fetch_smart_money_flow`` ->
  honest "not available in simulation" notes (point-in-time histories of
  these aren't stored; a note keeps the agent reasoning on real data instead
  of on today's targets leaking into the past)

Keeping every patch point in this one module means a new live fetch added to
the app fails loudly here (the setattr asserts the attribute exists) instead
of silently leaking real-time data into simulated sessions.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import pandas as pd

from agent_stonks import agent as agent_mod
from agent_stonks import clock, historical

from .market import SimMarket


def _series(pairs: list[tuple[str, float]]) -> pd.Series:
    if not pairs:
        return pd.Series(dtype=float)
    dates, closes = zip(*pairs)
    return pd.Series(closes, index=pd.to_datetime(dates))


@contextmanager
def simulation_context(market: SimMarket) -> Iterator[None]:
    def fake_bars_window(symbol, timeframe, start, end, key, secret, feed="iex", limit=200):
        return market.bars_window(str(symbol).upper(), start, end)

    def fake_corporate_actions(symbol, key, secret, days_ahead=14):
        return []

    def fake_market_indicators(days: int = 365, ttl_sec: int = 300) -> dict:
        now = clock.now()
        return {
            name: _series(market.indicator_closes(name, now))
            for name in ("spy", "vix", "vix3m")
        }

    def fake_analyst_targets(symbol, current_price=None):
        return {
            "note": (
                "analyst price targets are not available in simulation "
                "(no point-in-time history stored) -- weigh the technical read instead"
            )
        }

    def fake_smart_money_flow(symbol):
        return {
            "note": (
                "insider/institutional ownership data is not available in simulation "
                "(no point-in-time history stored) -- weigh the technical read instead"
            )
        }

    patches = [
        (agent_mod, "fetch_bars_window", fake_bars_window),
        (agent_mod, "fetch_corporate_actions", fake_corporate_actions),
        (historical, "fetch_market_indicators", fake_market_indicators),
        (historical, "fetch_analyst_targets", fake_analyst_targets),
        (historical, "fetch_smart_money_flow", fake_smart_money_flow),
    ]
    saved = []
    for module, name, replacement in patches:
        saved.append((module, name, getattr(module, name)))  # raises if renamed upstream
        setattr(module, name, replacement)
    try:
        yield
    finally:
        for module, name, original in saved:
            setattr(module, name, original)
        clock.clear()
