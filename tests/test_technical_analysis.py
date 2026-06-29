import pandas as pd

from marketview.technical_analysis import (
    adx,
    analyze_consolidation,
    analyze_fair_value_gaps,
    analyze_intraday,
    analyze_market,
    analyze_order_blocks,
    analyze_smart_money_setup,
    analyze_trend,
    analyze_volume,
    analyze_vwap_bands,
    atr,
    breakout_trade_geometry,
    find_fair_value_gaps,
    find_order_blocks,
    get_put_call_walls_and_gamma,
    obv_trend,
    rsi,
    session_time_window,
    smart_money_trade_geometry,
    sma,
    support_resistance,
    vwap_reversion_geometry,
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


class TestAnalyzeConsolidation:
    def test_not_enough_bars_returns_note(self):
        bars = _make_bars([100.0, 101.0])
        assert "note" in analyze_consolidation(bars)

    def test_contracting_range_and_declining_volume_reads_as_coiling(self):
        prior_closes = [90.0, 110.0] * 10  # wide swings -> wide prior range
        prior_bars = _make_bars(prior_closes, highs=[c + 10 for c in prior_closes], lows=[c - 10 for c in prior_closes], volumes=[2000] * 20)
        base_closes = [100.0] * 10  # tight range
        base_bars = _make_bars(base_closes, highs=[100.5] * 10, lows=[99.5] * 10, volumes=[500] * 10)
        result = analyze_consolidation(prior_bars + base_bars, base_bars=10, prior_bars=20)

        assert result["base_high"] == 100.5
        assert result["base_low"] == 99.5
        assert result["base_height"] == 1.0
        assert result["range_contraction_pct"] > 10
        assert result["volume_trend_in_base"] == "declining"
        assert result["is_coiling"] is True

    def test_expanding_range_is_not_coiling(self):
        prior_closes = [100.0] * 20
        prior_bars = _make_bars(prior_closes, highs=[100.5] * 20, lows=[99.5] * 20, volumes=[500] * 20)
        base_closes = [90.0, 110.0] * 5
        base_bars = _make_bars(base_closes, highs=[c + 10 for c in base_closes], lows=[c - 10 for c in base_closes], volumes=[2000] * 10)
        result = analyze_consolidation(prior_bars + base_bars, base_bars=10, prior_bars=20)

        assert result["is_coiling"] is False

    def test_counts_touches_at_resistance_and_support(self):
        prior_bars = _make_bars([100.0] * 20, volumes=[1000] * 20)
        highs = [100.5, 99.0, 100.5, 99.0, 100.5, 99.0, 99.0, 99.0, 99.0, 99.0]
        lows = [99.5, 98.5, 99.5, 98.5, 99.5, 98.5, 98.5, 98.5, 98.5, 98.5]
        base_bars = _make_bars([99.0] * 10, highs=highs, lows=lows, volumes=[1000] * 10)
        result = analyze_consolidation(prior_bars + base_bars, base_bars=10, prior_bars=20)

        assert result["touches_at_resistance"] == 3
        assert result["well_tested"] is True


class TestSessionTimeWindow:
    def test_opening_window_is_favorable(self):
        result = session_time_window("2024-07-15T13:35:00Z")  # 09:35 ET (EDT, UTC-4)
        assert result["window"] == "opening_window"
        assert result["favorable_for_breakouts"] is True

    def test_midday_dead_zone_is_unfavorable(self):
        result = session_time_window("2024-07-15T17:30:00Z")  # 13:30 ET
        assert result["window"] == "midday_dead_zone"
        assert result["favorable_for_breakouts"] is False

    def test_power_hour_is_favorable(self):
        result = session_time_window("2024-07-15T19:45:00Z")  # 15:45 ET
        assert result["window"] == "power_hour"
        assert result["favorable_for_breakouts"] is True

    def test_outside_regular_hours_is_unfavorable(self):
        result = session_time_window("2024-07-15T03:00:00Z")  # 23:00 ET prior day
        assert result["window"] == "outside_regular_hours"
        assert result["favorable_for_breakouts"] is False

    def test_falls_back_to_now_when_no_timestamp_given(self):
        result = session_time_window(None)
        assert "et_time" in result
        assert "window" in result


def _osc_bars(n=40, center=100.0, amp=1.0, vol=1000):
    """Choppy, alternating bars -- a ranging tape (balanced +DI/-DI, low ADX)."""
    bars = []
    for i in range(n):
        c = center + (amp if i % 2 == 0 else -amp)
        bars.append({"o": c, "h": c + 0.3, "l": c - 0.3, "c": c, "v": vol})
    return bars


class TestADX:
    def test_returns_none_when_not_enough_bars(self):
        assert adx(_make_bars([100.0] * 10), period=14) is None

    def test_strong_uptrend_reads_as_trending(self):
        closes = [100.0 + i for i in range(40)]
        assert adx(_make_bars(closes), period=14) >= 25

    def test_choppy_range_reads_as_non_trending(self):
        # Balanced up/down movement -> +DI and -DI cancel -> low ADX.
        assert adx(_osc_bars(40), period=14) < 25


class TestAnalyzeVwapBands:
    def test_not_enough_bars_returns_note(self):
        assert "note" in analyze_vwap_bands(_make_bars([100.0, 101.0]))

    def test_bands_are_ordered_around_vwap(self):
        result = analyze_vwap_bands(_osc_bars(40))
        assert result["lower_band_3sd"] < result["lower_band_2sd"] < result["lower_band_1sd"]
        assert result["lower_band_1sd"] < result["vwap"] < result["upper_band_1sd"]
        assert result["upper_band_1sd"] < result["upper_band_2sd"] < result["upper_band_3sd"]

    def test_oversold_stretch_in_range_is_long_setup(self):
        bars = _osc_bars(40)
        # A deep dip on the final bar with a long lower wick (bullish rejection).
        bars.append({"o": 97.0, "h": 97.2, "l": 95.5, "c": 97.0, "v": 1000})
        result = analyze_vwap_bands(bars, num_std=2.0)
        assert result["z_score"] <= -2.0
        assert result["is_ranging"] is True
        assert result["signal"] == "long_setup"
        assert result["rejection_candle"] == "bullish_rejection"

    def test_strong_trend_is_not_a_fade_setup(self):
        closes = [100.0 + i for i in range(40)]
        result = analyze_vwap_bands(_make_bars(closes), num_std=2.0)
        assert result["is_ranging"] is False
        assert result["signal"] in ("no_setup", "no_setup_trending")


class TestVwapReversionGeometry:
    def test_non_positive_inputs_return_note(self):
        assert "note" in vwap_reversion_geometry(entry=0, vwap=100.0, std_dev=1.0)

    def test_long_entry_must_be_below_vwap(self):
        assert "note" in vwap_reversion_geometry(entry=101.0, vwap=100.0, std_dev=1.0, side="long")

    def test_long_reversion_geometry_and_reward_risk(self):
        # Entry 2sd below VWAP, stop 1sd further down, target VWAP -> 2:1.
        result = vwap_reversion_geometry(entry=98.0, vwap=100.0, std_dev=1.0, side="long")
        assert result["stop"] == 97.0
        assert result["target"] == 100.0
        assert result["risk_per_share"] == 1.0
        assert result["reward_per_share"] == 2.0
        assert result["reward_risk_ratio"] == 2.0
        assert result["meets_min_reward_risk"] is True

    def test_shallow_stretch_fails_min_reward_risk(self):
        # Entry only 1sd below VWAP -> reward == risk -> 1:1, below the 1.5 minimum.
        result = vwap_reversion_geometry(entry=99.0, vwap=100.0, std_dev=1.0, side="long")
        assert result["reward_risk_ratio"] == 1.0
        assert result["meets_min_reward_risk"] is False

    def test_short_reversion_geometry(self):
        result = vwap_reversion_geometry(entry=102.0, vwap=100.0, std_dev=1.0, side="short")
        assert result["stop"] == 103.0
        assert result["risk_per_share"] == 1.0
        assert result["reward_per_share"] == 2.0
        assert result["meets_min_reward_risk"] is True


class TestBreakoutTradeGeometry:
    def test_non_positive_entry_or_stop_returns_note(self):
        assert "note" in breakout_trade_geometry(entry=0, stop=10)

    def test_stop_above_entry_returns_note(self):
        assert "note" in breakout_trade_geometry(entry=100.0, stop=105.0)

    def test_base_height_target_meets_min_reward_risk(self):
        result = breakout_trade_geometry(entry=100.0, stop=98.0, base_height=4.0)
        assert result["risk_per_share"] == 2.0
        assert result["target1_base_height"] == 104.0
        assert result["rr1_base_height"] == 2.0
        assert result["target2_base_height"] == 108.0
        assert result["rr2_base_height"] == 4.0
        assert result["meets_min_reward_risk"] is True

    def test_wide_stop_fails_min_reward_risk(self):
        result = breakout_trade_geometry(entry=100.0, stop=90.0, base_height=5.0)
        assert result["best_reward_risk_ratio"] == 1.0
        assert result["meets_min_reward_risk"] is False

    def test_atr_target_computed_alongside_base_height(self):
        result = breakout_trade_geometry(entry=100.0, stop=99.0, base_height=2.0, atr=1.0)
        assert result["target1_atr"] == 101.0
        assert result["rr1_atr"] == 1.0
        assert result["target2_atr"] == 102.0

    def test_no_target_inputs_returns_no_targets(self):
        result = breakout_trade_geometry(entry=100.0, stop=99.0)
        assert result["best_reward_risk_ratio"] is None
        assert result["meets_min_reward_risk"] is False
        assert "cannot project a target" in result["summary"]


def _ohlc(o, h, l, c, v=1000):
    return {"o": o, "h": h, "l": l, "c": c, "v": v}


# A daily series with one clean bullish order block at index 8: a bearish candle
# (the zone 98.5-101.5) followed by a three-bar up impulse that takes out the
# prior swing high (a bullish break of structure), then a return back into the zone.
def _smart_money_daily_bars():
    bars = [_ohlc(100, 100.5, 99.5, 100) for _ in range(8)]
    bars.append(_ohlc(101, 101.5, 98.5, 99))      # 8: bearish order block
    bars.append(_ohlc(99.2, 104.5, 99.0, 104))    # 9: impulse up...
    bars.append(_ohlc(104, 107.5, 103.5, 107))    # 10
    bars.append(_ohlc(107, 110.5, 106.5, 110))    # 11 (breaks structure)
    bars.append(_ohlc(109, 109.5, 105.5, 106))    # 12: return down...
    bars.append(_ohlc(106, 106.5, 102.5, 103))    # 13
    bars.append(_ohlc(103, 103.5, 100.5, 101))    # 14
    bars.append(_ohlc(101, 101.5, 99.5, 100))     # 15: back into the zone
    bars.append(_ohlc(100, 100.8, 99.2, 100))     # 16
    bars.append(_ohlc(100, 100.6, 99.4, 100))     # 17
    return bars


class TestOrderBlocks:
    def test_detects_bullish_order_block_at_displacement(self):
        result = find_order_blocks(_smart_money_daily_bars())
        bull = [b for b in result["order_blocks"] if b["type"] == "bullish"]
        assert len(bull) == 1
        block = bull[0]
        assert block["index"] == 8
        assert block["bottom"] == 98.5
        assert block["top"] == 101.5

    def test_block_marked_mitigated_after_price_returns(self):
        block = [b for b in find_order_blocks(_smart_money_daily_bars())["order_blocks"] if b["type"] == "bullish"][0]
        assert block["mitigated"] is True  # price returned into the zone after the impulse

    def test_not_enough_bars_returns_note(self):
        assert "note" in find_order_blocks(_make_bars([100.0, 101.0, 102.0]))

    def test_analyze_order_blocks_finds_nearest_demand(self):
        result = analyze_order_blocks(_smart_money_daily_bars(), spot=100.0)
        demand = result["nearest_bullish_ob"]
        assert demand is not None
        assert demand["bottom"] == 98.5 and demand["top"] == 101.5


class TestFairValueGaps:
    def test_detects_bullish_gap(self):
        # Middle bar gaps up: bar3 low (102) is above bar1 high (100).
        bars = [_ohlc(99, 100, 98, 99), _ohlc(100, 103, 100, 102.5), _ohlc(103, 104, 102, 103.5)]
        gaps = find_fair_value_gaps(bars)["fair_value_gaps"]
        assert len(gaps) == 1
        assert gaps[0]["type"] == "bullish"
        assert gaps[0]["bottom"] == 100.0 and gaps[0]["top"] == 102.0
        assert gaps[0]["filled"] is False

    def test_gap_marked_filled_when_price_returns(self):
        bars = [
            _ohlc(99, 100, 98, 99),
            _ohlc(100, 103, 100, 102.5),
            _ohlc(103, 104, 102, 103.5),
            _ohlc(103, 103.5, 101, 101.5),  # trades back down into the 100-102 gap
        ]
        gaps = find_fair_value_gaps(bars)["fair_value_gaps"]
        assert gaps[0]["filled"] is True

    def test_analyze_fair_value_gaps_not_enough_bars(self):
        assert "note" in analyze_fair_value_gaps([_ohlc(100, 101, 99, 100)])


class TestSmartMoneyTradeGeometry:
    def test_meets_three_to_one(self):
        result = smart_money_trade_geometry(entry=100.0, stop=98.0, target=107.0)
        assert result["risk_per_share"] == 2.0
        assert result["reward_per_share"] == 7.0
        assert result["reward_risk_ratio"] == 3.5
        assert result["meets_min_reward_risk"] is True

    def test_below_minimum_reward_risk(self):
        result = smart_money_trade_geometry(entry=100.0, stop=98.0, target=104.0)
        assert result["reward_risk_ratio"] == 2.0
        assert result["meets_min_reward_risk"] is False

    def test_invalid_ordering_returns_note(self):
        assert "note" in smart_money_trade_geometry(entry=100.0, stop=101.0, target=105.0)


class TestSmartMoneySetup:
    def test_long_setup_on_confirmed_return_into_demand(self):
        daily = _smart_money_daily_bars()
        # Last intraday bar is a bullish rejection candle (long lower wick, small body).
        intraday = [_ohlc(100, 100.2, 99.8, 100) for _ in range(5)]
        intraday.append(_ohlc(100.0, 100.3, 99.0, 100.2))
        result = analyze_smart_money_setup(daily, intraday_bars=intraday, spot=100.0)
        assert result["signal"] == "long_setup"
        assert result["price_in_order_block"] is True
        assert "rejection_candle" in result["intraday_confirmation"]
        assert result["order_block"]["bottom"] == 98.5
        assert result["meets_min_reward_risk"] is True

    def test_watching_when_price_not_in_block(self):
        daily = _smart_money_daily_bars()
        result = analyze_smart_money_setup(daily, intraday_bars=None, spot=108.0)
        assert result["signal"] == "watching"
        assert result["price_in_order_block"] is False
        assert result["order_block"] is not None

    def test_no_setup_without_enough_daily_bars(self):
        result = analyze_smart_money_setup(_make_bars([100.0, 101.0, 102.0]))
        assert result["signal"] == "no_setup"
        assert "note" in result
