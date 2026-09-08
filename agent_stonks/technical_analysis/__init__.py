"""Human-readable technical analysis over OHLCV bars.

Raw bars (Alpaca format: keys o/h/l/c/v/t/vw) are fine for charting but a
flat array of numbers isn't a signal -- an LLM agent has to redo the same
arithmetic every cycle to notice anything. This module does that arithmetic
once and returns the kind of read a human technical analyst would give:
trend regime, momentum, volatility, support/resistance, and volume
confirmation, each as a labeled value plus a one-line summary the agent can
reason over directly.

Split into one module per family of read, because a single 3,100-line file had
stopped being navigable: the classic indicators, the ICT concepts, the options
book and the market backdrop share a vocabulary of bars and nothing else.

    indicators    trend, momentum, volatility, VWAP bands, volume, consolidation
    regime        has the trajectory changed?
    levels        opening range, structure, swings, volume profile, pivots
    market        the broad-market backdrop
    options_flow  put/call walls and dealer gamma
    geometry      risk/reward arithmetic and the session clock
    smart_money   order blocks, fair-value gaps, liquidity, premium/discount

`indicators` is the base layer; every other module imports from it and none of
them import from each other, apart from the two news-timestamp helpers in
`_shared`. Everything public is re-exported here, so `technical_analysis.X`
still resolves for the agent tools, the UI and SimLab, which is what let the
split happen without touching any caller.
"""

from .indicators import (  # noqa: F401
    PARTICIPATION_MIN_LOCAL,
    PARTICIPATION_MIN_PACE,
    VOLUME_BURST_BARS,
    adx,
    analyze_consolidation,
    analyze_intraday,
    analyze_trend,
    analyze_volume,
    analyze_vwap_bands,
    atr,
    obv_trend,
    piecewise_regimes,
    rsi,
    sma,
    support_resistance,
)
from .regime import (  # noqa: F401
    detect_regime_shift,
)
from .levels import (  # noqa: F401
    analyze_opening_range,
    analyze_volume_profile_2,
    compute_opening_range,
    floor_pivots,
    key_levels,
    swing_levels,
    volume_profile_levels,
)
from .market import (  # noqa: F401
    analyze_market,
)
from .options_flow import (  # noqa: F401
    get_put_call_walls_and_gamma,
)
from .geometry import (  # noqa: F401
    breakout_trade_geometry,
    session_time_window,
    vwap_reversion_geometry,
)
from .smart_money import (  # noqa: F401
    analyze_fair_value_gaps,
    analyze_liquidity,
    analyze_order_blocks,
    analyze_premium_discount,
    analyze_smart_money_setup,
    find_fair_value_gaps,
    find_order_blocks,
    smart_money_trade_geometry,
)

# Private helpers that callers outside the package name directly.
from ._shared import _news_datetimes, _news_near  # noqa: F401
from .indicators import _closes, _rejection_candle, _session_bars  # noqa: F401
from .levels import _OPENING_RANGE_COVERAGE_GRACE_MIN  # noqa: F401
