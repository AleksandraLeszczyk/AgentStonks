import pandas as pd

from marketview.technical_analysis import (
    analyze_intraday,
    analyze_market,
    analyze_trend,
    analyze_volume,
    atr,
    get_put_call_walls_and_gamma,
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


class TestAnalyzeMarket:
    def test_no_data_returns_note(self):
        assert "note" in analyze_market(vix_close=None, spy_close=None)

    def test_calm_vix_and_uptrending_spy_is_risk_on(self):
        vix = pd.Series([13.0] * 10)
        spy = pd.Series([300.0 + i for i in range(260)])  # steady uptrend, above 50/200d
        result = analyze_market(vix_close=vix, spy_close=spy)
        assert result["risk_environment"] == "risk-on"
        assert result["risk_score"] > 0
        assert result["vix_label"] == "low (calm)"
        assert result["insights"]

    def test_high_vix_and_falling_spy_is_risk_off(self):
        vix = pd.Series([38.0] * 10)
        spy = pd.Series([500.0 - i for i in range(260)])  # steady downtrend, below 50/200d
        result = analyze_market(vix_close=vix, spy_close=spy)
        assert result["risk_environment"] == "risk-off"
        assert result["risk_score"] < 0
        assert "extreme (panic)" in result["vix_label"]

    def test_inverted_term_structure_flagged_as_backwardation(self):
        vix = pd.Series([30.0] * 10)
        vix3m = pd.Series([26.0] * 10)
        result = analyze_market(vix_close=vix, vix3m_close=vix3m, spy_close=None)
        assert result["vix_term_structure"] == "backwardation"
        assert any("inverted" in i for i in result["insights"])

    def test_contango_term_structure_when_vix3m_above_vix(self):
        vix = pd.Series([18.0] * 10)
        vix3m = pd.Series([20.0] * 10)
        result = analyze_market(vix_close=vix, vix3m_close=vix3m, spy_close=None)
        assert result["vix_term_structure"] == "contango"

    def test_spy_drawdown_reported(self):
        # Peak at 100, ends 15% lower -> correction territory.
        spy = pd.Series([100.0] * 5 + [85.0])
        result = analyze_market(vix_close=None, spy_close=spy)
        assert result["spy_drawdown_from_high_pct"] == -15.0
        assert any("correction" in i for i in result["insights"])

    def test_rising_vix_adds_caution_insight(self):
        vix = pd.Series([20.0, 21.0, 22.0, 24.0, 26.0, 28.0])  # +40% over 5 sessions
        result = analyze_market(vix_close=vix, spy_close=None)
        assert result["vix_5d_change_pct"] > 10
        assert any("rising fast" in i for i in result["insights"])


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


class TestGetPutCallWallsAndGamma:
    def test_no_strikes_returns_note(self):
        assert "note" in get_put_call_walls_and_gamma([], [], [], [], [], spot=100.0)

    def test_identifies_call_wall_and_put_wall_from_peak_open_interest(self):
        strikes = [90.0, 95.0, 100.0, 105.0, 110.0]
        calls_oi = [100, 200, 300, 400, 1000]
        puts_oi = [900, 300, 200, 100, 50]
        gamma = [0.0] * 5
        result = get_put_call_walls_and_gamma(strikes, calls_oi, puts_oi, gamma, gamma, spot=100.0)
        assert result["call_wall"] == 110.0
        assert result["put_wall"] == 90.0
        assert result["in_range"] is True

    def test_positive_net_gamma_is_dampening_regime(self):
        strikes = [95.0, 100.0, 105.0]
        calls_oi = [100, 100, 100]
        puts_oi = [100, 100, 100]
        calls_gamma = [10.0, 10.0, 10.0]
        puts_gamma = [-1.0, -1.0, -1.0]
        result = get_put_call_walls_and_gamma(strikes, calls_oi, puts_oi, calls_gamma, puts_gamma, spot=100.0)
        assert result["net_gamma"] > 0
        assert "positive" in result["gamma_regime"]

    def test_negative_net_gamma_is_amplifying_regime(self):
        strikes = [95.0, 100.0, 105.0]
        calls_oi = [100, 100, 100]
        puts_oi = [100, 100, 100]
        calls_gamma = [1.0, 1.0, 1.0]
        puts_gamma = [-10.0, -10.0, -10.0]
        result = get_put_call_walls_and_gamma(strikes, calls_oi, puts_oi, calls_gamma, puts_gamma, spot=100.0)
        assert result["net_gamma"] < 0
        assert "negative" in result["gamma_regime"]

    def test_gamma_flip_found_between_negative_and_positive_strikes(self):
        strikes = [95.0, 100.0, 105.0]
        calls_oi = puts_oi = [100, 100, 100]
        calls_gamma = [0.0, 5.0, 20.0]
        puts_gamma = [-10.0, 0.0, 0.0]
        result = get_put_call_walls_and_gamma(strikes, calls_oi, puts_oi, calls_gamma, puts_gamma, spot=100.0)
        assert result["gamma_flip"] is not None
        assert 100.0 < result["gamma_flip"] < 105.0

    def test_spot_above_call_wall_flags_breach(self):
        strikes = [90.0, 95.0, 100.0]
        calls_oi = [500, 100, 50]
        puts_oi = [50, 100, 500]
        gamma = [0.0] * 3
        result = get_put_call_walls_and_gamma(strikes, calls_oi, puts_oi, gamma, gamma, spot=110.0)
        assert result["in_range"] is False
        assert any("above the call wall" in i for i in result["insights"])

    def test_rising_call_wall_trend_detected_from_history(self):
        strikes = [95.0, 100.0, 105.0]
        calls_oi = [100, 100, 500]
        puts_oi = [500, 100, 100]
        gamma = [0.0] * 3
        history = [{"call_wall": 95.0, "put_wall": 90.0}, {"call_wall": 100.0, "put_wall": 90.0}]
        result = get_put_call_walls_and_gamma(
            strikes, calls_oi, puts_oi, gamma, gamma, spot=100.0, wall_history=history
        )
        assert result["call_wall_trend"] == "rising"
        assert any("rising" in i for i in result["insights"])
