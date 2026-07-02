import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

HISTORICAL_PERIODS: dict[str, int] = {
    "7 Days": 7,
    "28 Days": 28,
    "Quarter": 91,
    "1 Year": 365,
    "5 Years": 1825,
}

VIX_SYMBOL = "^VIX"
VIX3M_SYMBOL = "^VIX3M"
SPY_SYMBOL = "SPY"

# Broad-market indicator series change slowly relative to the agent's cycle and
# are identical for every ticker, so cache them briefly to avoid re-hitting
# yfinance on each agent call.
_market_cache: dict = {"ts": None, "data": None}


def fetch_close_series(symbol: str, days: int) -> pd.Series:
    """Fetch daily close prices for `symbol` over the trailing `days`. Raises on failure."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    try:
        df = yf.download(symbol, start=start, end=end, interval="1d", auto_adjust=True, progress=False)
    except Exception:
        logger.exception("yfinance download failed for %s (start=%s, end=%s)", symbol, start, end)
        raise
    if df.empty:
        logger.warning("yfinance returned no data for %s (start=%s, end=%s)", symbol, start, end)
        return pd.Series(dtype=float)
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close.index = pd.to_datetime(close.index)
    return close.dropna()


def fetch_intraday_bars(symbol: str, interval: str = "1m") -> list[dict]:
    """Today's intraday bars from yfinance -- last-resort price fallback when both
    Alpaca's stream and REST API are unavailable. No API key required, but quotes
    are delayed (typically ~15 minutes) rather than real-time.

    Returns bars in the same {"t","o","h","l","c","v"} shape as Alpaca's REST/stream
    bars (UTC ISO timestamps) so callers don't need to branch on the source.
    """
    try:
        df = yf.download(symbol, period="1d", interval=interval, auto_adjust=False, progress=False)
    except Exception:
        logger.exception("yfinance intraday download failed for %s", symbol)
        raise
    if df.empty:
        return []
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize("America/New_York")
    idx = idx.tz_convert("UTC")
    return [
        {
            "t": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "o": float(row.Open),
            "h": float(row.High),
            "l": float(row.Low),
            "c": float(row.Close),
            "v": float(row.Volume),
        }
        for ts, row in zip(idx, df.itertuples(index=False))
    ]


def fetch_market_indicators(days: int = 365, ttl_sec: int = 300) -> dict:
    """Fetch the broad-market condition series (SPY, VIX, VIX3M) for `analyze_market`.

    Returns a dict of {"spy", "vix", "vix3m"} -> daily close Series. A failed or
    unavailable symbol yields an empty Series rather than raising, so one bad
    feed never sinks the whole read. Results are cached for `ttl_sec` seconds.
    """
    now = datetime.now(timezone.utc)
    cached = _market_cache["data"]
    cached_ts = _market_cache["ts"]
    if cached is not None and cached_ts is not None and (now - cached_ts).total_seconds() < ttl_sec:
        return cached

    data: dict[str, pd.Series] = {}
    for key, symbol in (("spy", SPY_SYMBOL), ("vix", VIX_SYMBOL), ("vix3m", VIX3M_SYMBOL)):
        try:
            data[key] = fetch_close_series(symbol, days)
        except Exception:
            logger.error("Market indicator fetch failed for %s (%s); using empty series", symbol, key)
            data[key] = pd.Series(dtype=float)

    _market_cache["ts"] = now
    _market_cache["data"] = data
    return data


def fetch_dividends(symbol: str, days: int) -> pd.Series:
    """Fetch dividend payouts for `symbol` over the trailing `days`. Raises on failure."""
    try:
        div = yf.Ticker(symbol).dividends
    except Exception:
        logger.exception("yfinance dividends lookup failed for %s", symbol)
        raise
    if div.empty:
        return div
    cutoff = pd.Timestamp.now(tz=div.index.tz) - pd.Timedelta(days=days)
    return div[div.index >= cutoff]


def fetch_earnings_dates(symbol: str, days: int) -> pd.DataFrame:
    """Fetch past and upcoming earnings dates for `symbol` over the trailing `days`."""
    try:
        earnings = yf.Ticker(symbol).get_earnings_dates(limit=20)
    except Exception:
        logger.error("yfinance earnings dates lookup failed for %s", symbol, exc_info=True)
        return pd.DataFrame()
    if earnings is None or earnings.empty:
        return pd.DataFrame()
    cutoff = pd.Timestamp.now(tz=earnings.index.tz) - pd.Timedelta(days=days)
    return earnings[earnings.index >= cutoff]


def fetch_static_analysis(symbol: str) -> dict:
    """Fetch the raw inputs (P/E, growth, dividend yield) for a simple static valuation estimate."""
    ticker = yf.Ticker(symbol)
    try:
        info = ticker.info
        growth_rate = info.get("earningsGrowth") or info.get("revenueGrowth")
        return {
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "dividend_yield": info.get("trailingAnnualDividendYield"),
            "growth_rate": growth_rate,
        }
    except Exception:
        # Yahoo Finance periodically restricts the quoteSummary endpoint; fall back
        # to computing dividend yield from the chart endpoint (less restricted).
        logger.warning("yfinance .info unavailable for %s; falling back to fast_info", symbol, exc_info=True)

    dividend_yield = None
    try:
        last_price = ticker.fast_info.last_price
        annual_div = float(ticker.dividends.last("365D").sum())
        if last_price and annual_div:
            dividend_yield = annual_div / last_price
    except Exception:
        pass

    return {
        "pe_ratio": None,
        "forward_pe": None,
        "dividend_yield": dividend_yield,
        "growth_rate": None,
    }


SMART_MONEY_TTL_SEC = 6 * 3600
_smart_money_cache: dict = {}


def _net_insider_shares(purchases) -> "dict | None":
    """Parse yfinance `insider_purchases` (a 6-month buy/sell summary) into a net
    direction. The frame indexes rows like 'Total Shares Purchased' / 'Sold' /
    'Net Shares Purchased (Sold)' against a single value column."""
    try:
        frame = purchases
        if frame is None or getattr(frame, "empty", True):
            return None
        rows = {str(k).strip().lower(): v for k, v in frame.iloc[:, 0].items()}
        bought = rows.get("total shares purchased")
        sold = rows.get("total shares sold")
        net = rows.get("net shares purchased (sold)")
        if net is None and bought is not None and sold is not None:
            net = float(bought) - float(sold)
        if net is None:
            return None
        net = float(net)
        return {
            "net_shares_6mo": int(net),
            "bought_6mo": int(bought) if bought is not None else None,
            "sold_6mo": int(sold) if sold is not None else None,
            "direction": "buying" if net > 0 else ("selling" if net < 0 else "flat"),
        }
    except Exception:
        logger.warning("Could not parse insider purchases", exc_info=True)
        return None


def fetch_smart_money_flow(symbol: str) -> dict:
    """The institutional 'smart money' footprint for `symbol` from free yfinance data.

    Pulls three slow-moving but high-signal disclosures Yahoo aggregates from SEC
    filings: aggregate ownership breakdown (% held by insiders vs institutions),
    net insider buying/selling over the trailing 6 months (Form 4), and the
    largest institutional holders with their quarter-over-quarter share changes
    (13F). These are quarterly/Form-4 cadence -- not intraday signals -- so the
    result is cached for several hours. A net insider/institutional accumulation
    behind a bullish demand block corroborates the technical Smart Money read;
    distribution is a caution flag. Never raises -- missing fields come back None.
    """
    now = datetime.now(timezone.utc)
    cached = _smart_money_cache.get(symbol)
    if cached and (now - cached["ts"]).total_seconds() < SMART_MONEY_TTL_SEC:
        return cached["data"]

    ticker = yf.Ticker(symbol)
    result: dict = {
        "symbol": symbol,
        "insiders_pct_held": None,
        "institutions_pct_held": None,
        "institutions_count": None,
        "insider_flow": None,
        "top_institutional_holders": [],
        "institutional_net_pct_change": None,
    }

    try:
        mh = ticker.major_holders
        if mh is not None and not mh.empty:
            col = mh.iloc[:, 0]
            for label, value in col.items():
                key = str(label).strip().lower()
                try:
                    val = float(value)
                except (TypeError, ValueError):
                    continue
                if key == "insiderspercentheld":
                    result["insiders_pct_held"] = round(val, 4)
                elif key == "institutionspercentheld":
                    result["institutions_pct_held"] = round(val, 4)
                elif key == "institutionscount":
                    result["institutions_count"] = int(val)
    except Exception:
        logger.warning("yfinance major_holders unavailable for %s", symbol, exc_info=True)

    try:
        result["insider_flow"] = _net_insider_shares(ticker.insider_purchases)
    except Exception:
        logger.warning("yfinance insider_purchases unavailable for %s", symbol, exc_info=True)

    try:
        inst = ticker.institutional_holders
        if inst is not None and not inst.empty:
            net_change = 0.0
            for _, row in inst.head(10).iterrows():
                pct_change = row.get("pctChange")
                if pct_change is not None and pd.notna(pct_change):
                    net_change += float(pct_change)
                result["top_institutional_holders"].append({
                    "holder": str(row.get("Holder", "")),
                    "shares": int(row["Shares"]) if pd.notna(row.get("Shares")) else None,
                    "pct_held": round(float(row["pctHeld"]), 4) if pd.notna(row.get("pctHeld")) else None,
                    "pct_change": round(float(pct_change), 4) if pct_change is not None and pd.notna(pct_change) else None,
                })
            result["top_institutional_holders"] = result["top_institutional_holders"][:5]
            result["institutional_net_pct_change"] = round(net_change, 4)
    except Exception:
        logger.warning("yfinance institutional_holders unavailable for %s", symbol, exc_info=True)

    result["summary"] = _summarize_smart_money_flow(result)
    _smart_money_cache[symbol] = {"ts": now, "data": result}
    return result


def _summarize_smart_money_flow(flow: dict) -> str:
    parts: list[str] = []
    inst_pct = flow.get("institutions_pct_held")
    ins_pct = flow.get("insiders_pct_held")
    if inst_pct is not None:
        parts.append(f"Institutions hold {inst_pct * 100:.1f}%" + (f" across {flow['institutions_count']} holders" if flow.get("institutions_count") else "") + ".")
    if ins_pct is not None:
        parts.append(f"Insiders hold {ins_pct * 100:.1f}%.")
    insider = flow.get("insider_flow")
    if insider:
        parts.append(f"Insiders net {insider['direction']} {abs(insider['net_shares_6mo']):,} shares over 6mo (Form 4).")
    net = flow.get("institutional_net_pct_change")
    if net is not None:
        lean = "accumulating" if net > 0 else ("distributing" if net < 0 else "flat")
        parts.append(f"Top institutions {lean} (net {net * 100:+.1f}% q/q, 13F).")
    return " ".join(parts) if parts else "No institutional ownership data available."


def estimate_total_return(dividend_yield: Optional[float], growth_rate: Optional[float]) -> Optional[float]:
    """Rough estimate of annual total return on the asset: dividend yield + earnings/revenue growth."""
    if dividend_yield is None or growth_rate is None:
        return None
    return dividend_yield + growth_rate


def estimate_dividend_return_10y(
    dividend_yield: Optional[float], growth_rate: Optional[float], years: int = 10
) -> Optional[float]:
    """Cumulative dividends collected over `years`, as a fraction of the current price.

    Assumes the dividend grows annually at `growth_rate` from today's yield.
    """
    if dividend_yield is None or growth_rate is None:
        return None
    return sum(dividend_yield * (1 + growth_rate) ** t for t in range(years))
