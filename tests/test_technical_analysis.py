import pandas as pd

from marketview.technical_analysis import (
    analyze_intraday,
    analyze_trend,
    analyze_volume,
    atr,
    obv_trend,
    rsi,
    sma,
    support_resistance,
)


def _make_bars(closes, highs=None, lows=None, volumes=None, vwaps=None):
    highs = highs or [c + 0.5 for c in closes]
    lows = lows or [c - 0.5 for c in closes]
    volumes = volumes or [1000] * len(closes)
    bars = []
    for i, c in enumerate(closes):
        bar = {"o": c, "h": highs[i], "l": lows[i], "c": c, "v": volumes[i]}
        if vwaps is not None:
            bar["vw"] = vwaps[i]
        bars.append(bar)
    return bars


class TestIndicatorHelpers:
    def test_sma_basic(self):
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        assert sma(series, 3) == 4.0

    def test_sma_returns_none_when_not_enough_data(self):
        series = pd.Series([1.0, 2.0])
        assert sma(series, 3) is None

    def test_rsi_is_100_for_strictly_increasing_series(self):
        series = pd.Series([float(i) for i in range(10)])
        assert rsi(series, period=5) == 100.0

    def test_rsi_returns_none_when_not_enough_data(self):
        series = pd.Series([1.0, 2.0])
        assert rsi(series, period=5) is None

    def test_atr_equals_constant_high_low_range_when_close_unchanged(self):
        closes = [100.0] * 16
        bars = _make_bars(closes, highs=[101.0] * 16, lows=[99.0] * 16)
        assert atr(bars, period=14) == 2.0

    def test_atr_returns_none_when_not_enough_data(self):
        bars = _make_bars([100.0, 101.0])
        assert atr(bars, period=14) is None

    def test_obv_trend_rising_when_price_climbs_on_volume(self):
        closes = [100.0 + i for i in range(12)]
        bars = _make_bars(closes)
        assert obv_trend(bars, window=5) == "rising"

    def test_obv_trend_falling_when_price_drops_on_volume(self):
        closes = [100.0 - i for i in range(12)]
        bars = _make_bars(closes)
        assert obv_trend(bars, window=5) == "falling"

    def test_support_resistance_uses_lookback_window(self):
        closes = list(range(1, 31))
        bars = _make_bars([float(c) for c in closes])
        levels = support_resistance(bars, lookback=5)
        assert levels["lookback_bars"] == 5
        assert levels["support"] == bars[-5]["l"]
        assert levels["resistance"] == bars[-1]["h"]


class TestAnalyzeTrend:
    def test_not_enough_bars_returns_note(self):
        bars = _make_bars([100.0, 101.0])
        assert "note" in analyze_trend(bars)

    def test_steady_uptrend_is_classified_bullish(self):
        closes = [100.0 + i * 0.5 for i in range(70)]
        bars = _make_bars(closes)
        result = analyze_trend(bars)
        assert result["regime"] == "bullish"
        assert result["pct_change_over_period"] > 0
        assert "summary" in result

    def test_steady_downtrend_is_classified_bearish(self):
        closes = [150.0 - i * 0.5 for i in range(70)]
        bars = _make_bars(closes)
        result = analyze_trend(bars)
        assert result["regime"] == "bearish"
        assert result["pct_change_over_period"] < 0

    def test_flat_choppy_series_is_classified_neutral(self):
        closes = [100.0, 100.5, 99.8, 100.2, 99.9, 100.1, 100.0, 99.7, 100.3, 100.0] * 6
        bars = _make_bars(closes)
        result = analyze_trend(bars)
        assert result["regime"] == "neutral"


class TestAnalyzeIntraday:
    def test_not_enough_bars_returns_note(self):
        bars = _make_bars([100.0, 101.0])
        assert "note" in analyze_intraday(bars)

    def test_detects_higher_highs_and_higher_lows(self):
        closes = [100.0 + i for i in range(20)]
        bars = _make_bars(closes)
        result = analyze_intraday(bars)
        assert "uptrend" in result["momentum_pattern"]

    def test_detects_lower_highs_and_lower_lows(self):
        closes = [120.0 - i for i in range(20)]
        bars = _make_bars(closes)
        result = analyze_intraday(bars)
        assert "downtrend" in result["momentum_pattern"]

    def test_vwap_position_reported_when_present(self):
        closes = [100.0] * 10
        bars = _make_bars(closes, vwaps=[99.0] * 10)
        result = analyze_intraday(bars)
        assert "above" in result["vwap_position"]


class TestAnalyzeVolume:
    def test_no_bars_returns_note(self):
        assert "note" in analyze_volume([])

    def test_rising_price_and_volume_is_confirming(self):
        closes = [100.0 + i for i in range(20)]
        volumes = [1000 + i * 100 for i in range(20)]
        bars = _make_bars(closes, volumes=volumes)
        result = analyze_volume(bars)
        assert result["volume_trend"] == "increasing"
        assert result["confirmation"] == "confirming"

    def test_rising_price_with_falling_volume_is_diverging(self):
        closes = [100.0 + i for i in range(20)]
        volumes = [5000 - i * 200 for i in range(20)]
        bars = _make_bars(closes, volumes=volumes)
        result = analyze_volume(bars)
        assert result["volume_trend"] == "decreasing"
        assert "diverging" in result["confirmation"]
