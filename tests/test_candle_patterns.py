import numpy as np
import pandas as pd
import pytest
import plotly.graph_objects as go

from agent_stonks import candle_patterns
from agent_stonks.charts import add_candle_patterns, build_chart
from agent_stonks.technical_analysis import fair_value_gaps

# 09:30 ET on a summer day.
OPEN = pd.Timestamp("2026-07-28T13:30:00Z")


def _bars(rows, start=OPEN):
    """(o, h, l, c) rows as consecutive 1-minute bars from `start`."""
    return [
        {"t": (start + pd.Timedelta(minutes=i)).isoformat(), "o": o, "h": h, "l": l, "c": c, "v": 100}
        for i, (o, h, l, c) in enumerate(rows)
    ]


# Candle 1 tops out at 100, candle 3 bottoms at 102: a bullish gap 100-102.
BULL = [(99, 100, 98, 99.5), (100, 103, 100, 102.5), (103, 104, 102, 103.5)]


class TestFairValueGaps:
    def test_touch_and_fill_are_separate_moments(self):
        rows = BULL + [
            (103, 103.5, 101, 101.5),  # into the gap: touched, not filled
            (101.5, 102, 99.5, 99.8),  # through its bottom: filled
        ]
        (gap,) = fair_value_gaps(_bars(rows))
        assert gap["type"] == "bullish" and gap["index"] == 1
        assert (gap["bottom"], gap["top"]) == (100.0, 102.0)
        assert gap["touched_at"] == 3 and gap["filled_at"] == 4

    def test_bearish_mirror(self):
        rows = [(101, 102, 100, 100.5), (100, 100, 97, 97.5), (97, 98, 96, 96.5)]
        (gap,) = fair_value_gaps(_bars(rows))
        assert gap["type"] == "bearish"
        assert (gap["bottom"], gap["top"]) == (98.0, 100.0)
        assert gap["touched_at"] is None and gap["filled_at"] is None

    def test_overlapping_wicks_are_not_a_gap(self):
        rows = [(99, 100, 98, 99.5), (100, 103, 100, 102.5), (103, 104, 99.9, 103.5)]
        assert fair_value_gaps(_bars(rows)) == []


class TestCompute:
    def test_open_gap_item(self):
        (item,) = candle_patterns.compute([candle_patterns.FVG_KEY], _bars(BULL), min_size=0)
        assert item["kind"] == "fvg" and item["direction"] == "bullish"
        assert (item["y0"], item["y1"]) == (100.0, 102.0)
        assert pd.Timestamp(item["x0"]) == OPEN  # candle 1
        assert pd.Timestamp(item["formed"]) == OPEN + pd.Timedelta(minutes=2)  # candle 3
        assert pd.Timestamp(item["x1"]) == OPEN + pd.Timedelta(minutes=2)  # last bar
        assert item["filled"] is False

    def test_filled_gap_ends_at_the_filling_bar(self):
        rows = BULL + [(103, 103.5, 99, 99.5)]
        (item,) = candle_patterns.compute(["fvg"], _bars(rows), min_size=0)
        assert item["filled"] is True
        assert pd.Timestamp(item["x1"]) == OPEN + pd.Timedelta(minutes=3)
        assert candle_patterns.compute(["fvg"], _bars(rows), min_size=0, hide_filled=True) == []

    def test_min_size_is_relative_to_recent_bar_ranges(self):
        # The 2.0 gap against candle 1's 2.0 range is 1.0x.
        assert candle_patterns.compute(["fvg"], _bars(BULL), min_size=1.0)
        assert not candle_patterns.compute(["fvg"], _bars(BULL), min_size=1.25)

    def test_unknown_or_no_keys_draw_nothing(self):
        assert candle_patterns.compute([], _bars(BULL)) == []
        assert candle_patterns.compute(["nope"], _bars(BULL)) == []

    def test_open_gap_stops_at_its_own_session(self):
        day2 = _bars([(103, 103.5, 102.5, 103)] * 3, start=OPEN + pd.Timedelta(days=1))
        (item,) = candle_patterns.compute(["fvg"], _bars(BULL) + day2, min_size=0)
        assert pd.Timestamp(item["x1"]) == OPEN + pd.Timedelta(minutes=2)

    def test_pre_market_is_ignored(self):
        early = OPEN - pd.Timedelta(minutes=10)
        assert candle_patterns.compute(["fvg"], _bars(BULL, start=early), min_size=0) == []

    def test_gap_across_the_overnight_close_is_not_an_fvg(self):
        # Candle 1 is the day's last regular bar, candles 2-3 the next open.
        day1 = _bars(BULL[:1], start=pd.Timestamp("2026-07-27T19:59:00Z"))
        day2 = _bars(BULL[1:], start=OPEN)
        assert candle_patterns.compute(["fvg"], day1 + day2, min_size=0) == []


class TestRenderer:
    def test_boxes_go_behind_the_candles_with_one_legend_entry_per_direction(self):
        # A bullish gap filled at bar 3, then a second one (100-101) left open.
        bars = _bars(BULL + [
            (103, 103.5, 99, 99.5), (99.5, 100, 99, 99.8),
            (99.8, 104, 99.8, 103.9), (104, 105, 101, 104.5),
        ])
        items = candle_patterns.compute(["fvg"], bars, min_size=0)
        assert any(i["filled"] for i in items) and any(not i["filled"] for i in items)

        fig = go.Figure(go.Candlestick(
            x=[b["t"] for b in bars], open=[b["o"] for b in bars], high=[b["h"] for b in bars],
            low=[b["l"] for b in bars], close=[b["c"] for b in bars],
        ))
        add_candle_patterns(
            items, fig, pd.Timestamp(bars[0]["t"]), pd.Timestamp(bars[-1]["t"]), row=None, col=None,
        )
        polygons = [t for t in fig.data if getattr(t, "fill", None) == "toself"]
        assert polygons and fig.data[: len(polygons)] == tuple(polygons)
        assert isinstance(fig.data[len(polygons)], go.Candlestick)
        legend = [t.name for t in fig.data if t.showlegend]
        assert legend.count("Bullish FVG") == 1

    def test_build_chart_draws_them(self):
        bars = _bars(BULL)
        items = candle_patterns.compute(["fvg"], bars, min_size=0)
        fig = build_chart(bars, [], [], "AAPL", OPEN - pd.Timedelta(minutes=10), candle_patterns=items)
        assert any(t.name == "Bullish FVG" for t in fig.data)


# The stall scenario, u = $0.10 a minute, N = 3: twenty heavy bars build a
# volume node at 100, price slides to 99 and sits there, then rallies $0.95 in
# three bars (bar 42) and goes flat under the node -- stalled at bar 45.
U = 0.10
NODE_CLOSES = [100.0] * 20 + [round(100.0 - 0.1 * k, 2) for k in range(1, 11)] + [99.0] * 10
RALLY = [99.35, 99.7, 99.95, 99.96, 99.95, 99.96]


def _session(closes, volumes=None, start=OPEN):
    """1-minute bars through `closes`, each opening at the previous close."""
    volumes = volumes or [10_000] * 20 + [1_000] * (len(closes) - 20)
    bars, prev = [], closes[0]
    for i, (c, v) in enumerate(zip(closes, volumes)):
        bars.append({
            "t": (start + pd.Timedelta(minutes=i)).isoformat(),
            "o": prev, "h": max(prev, c) + 0.02, "l": min(prev, c) - 0.02, "c": c, "v": v,
        })
        prev = c
    return bars


def _stalls(bars, **kwargs):
    kwargs.setdefault("stall_bars", 3)
    kwargs.setdefault("minute_move", U)
    return candle_patterns.compute([candle_patterns.STALL_KEY], bars, **kwargs)


def _bar(i):
    return OPEN + pd.Timedelta(minutes=i)


class TestStall:
    def test_rally_that_stalls_under_a_node_is_resistance(self):
        (item,) = _stalls(_session(NODE_CLOSES + RALLY))
        assert item["kind"] == "stall" and item["side"] == "resistance"
        assert pd.Timestamp(item["x"]) == _bar(45)
        assert pd.Timestamp(item["x0"]) == _bar(42)
        assert abs(item["level"] - 100.0) < 0.05 and item["poc"] is True
        assert item["approach"] == pytest.approx(0.95)
        assert item["stall_move"] == pytest.approx(0.01)
        assert item["y"] > 99.96 + 0.02  # above the bar
        assert item["broken"] is False

    def test_the_mirror_is_support(self):
        mirrored = [round(200.0 - c, 2) for c in NODE_CLOSES + RALLY]
        (item,) = _stalls(_session(mirrored))
        assert item["side"] == "support"
        assert pd.Timestamp(item["x"]) == _bar(45)
        assert abs(item["level"] - 100.0) < 0.05
        assert item["y"] < 100.04 - 0.02  # below the bar

    def test_no_volume_node_there_no_mark(self):
        closes = [101.0] * 20 + NODE_CLOSES[20:] + RALLY
        assert _stalls(_session(closes)) == []

    def test_a_slow_creep_to_the_node_is_no_approach(self):
        creep = [round(99.0 + 0.95 * k / 40, 3) for k in range(1, 41)]
        assert _stalls(_session(NODE_CLOSES + creep + [creep[-1]] * 5)) == []

    def test_one_mark_per_visit_until_price_leaves_the_node(self):
        # A dip and a second, smaller rally while still near the node, then a
        # trip back down to 99 and a third rally.
        closes = NODE_CLOSES + RALLY + [99.85, 99.85, 99.85, 99.9, 99.93, 99.96, 99.96, 99.96, 99.96]
        closes += [99.0] * 6 + RALLY
        items = _stalls(_session(closes), stall_approach=0.25)
        resistance = [pd.Timestamp(i["x"]) for i in items if i["side"] == "resistance"]
        assert resistance == [_bar(45), _bar(len(closes) - 1)]

    def test_volume_check_compares_with_the_bars_before_the_move(self):
        bars = _session(NODE_CLOSES + RALLY)
        (item,) = _stalls(bars)
        # 1,000 a bar in the stall against bars 9-38 (11 heavy, 19 light).
        assert item["volume_ratio"] == pytest.approx(1_000 / ((11 * 10_000 + 19 * 1_000) / 30), abs=0.01)
        assert _stalls(bars, stall_min_volume=1.0) == []
        for bar in bars[43:46]:
            bar["v"] = 10_000
        assert _stalls(bars, stall_min_volume=1.0)

    def test_a_close_through_the_node_ends_its_line(self):
        closes = NODE_CLOSES + RALLY + [100.05, 100.2, 100.3]
        (item,) = _stalls(_session(closes))
        assert item["broken"] is True
        assert pd.Timestamp(item["x1"]) == _bar(47)  # first close > node + u

    def test_a_mark_never_depends_on_later_bars(self):
        rng = np.random.default_rng(7)
        closes = list(np.round(100 + np.cumsum(rng.normal(0, U, 390)), 2))
        volumes = list(rng.lognormal(8, 0.6, 390).round())
        bars = _session(closes, volumes)
        full = _stalls(bars, stall_approach=0.25)
        assert len(full) >= 3
        for k in (120, 200, 300):
            cutoff = pd.Timestamp(bars[k - 1]["t"])
            early = [(i["x"], i["side"], i["level"]) for i in full if pd.Timestamp(i["x"]) <= cutoff]
            prefix = [(i["x"], i["side"], i["level"]) for i in _stalls(bars[:k], stall_approach=0.25)]
            assert prefix == early

    def test_minute_move_per_day_and_the_sessions_own_fallback(self):
        bars = _session(NODE_CLOSES + RALLY)
        day = OPEN.tz_convert("America/New_York").date()
        assert _stalls(bars, minute_move={day: U}) == _stalls(bars)
        (own,) = _stalls(bars, minute_move=None, stall_approach=0.5)
        closes = np.array([b["c"] for b in bars[:46]])
        assert own["unit"] == pytest.approx(np.abs(np.diff(closes)).mean(), abs=1e-4)

    def test_build_chart_draws_marks_with_a_hover_and_one_legend_entry(self):
        bars = _session(NODE_CLOSES + RALLY)
        items = _stalls(bars)
        fig = build_chart(bars, [], [], "AAPL", OPEN - pd.Timedelta(minutes=10), candle_patterns=items)
        stall = [t for t in fig.data if t.name == "Resistance stall"]
        marks = [t for t in stall if t.mode == "markers"]
        assert len(marks) == 1 and marks[0].marker.symbol == "triangle-down"
        assert "Resistance stall</b> at 100.0" in marks[0].customdata[0]
        assert [t.showlegend for t in stall].count(False) == 1  # the line rides on the mark's entry


# The zone scenario, u = $0.10, so the default 3u band is $0.30: a slide in
# $0.20 steps, 25 bars with every close at 100.0-100.2 and the volume on the
# flat 100.2 bars, then a slide out below and a rally back through.
SLIDE_IN = [round(102.0 - 0.2 * k, 2) for k in range(8)]  # 102.0 .. 100.6
ZONE = ([100.2, 100.2, 100.2, 100.0, 100.2] * 5)
SLIDE_OUT = [99.8, 99.6, 99.4, 99.2]
RALLY_BACK = [99.6, 100.0, 100.4, 100.6]


def _zone_session(closes, start=OPEN):
    """Bars through `closes`, heavy where a close repeats the one before:
    the zone's flat 100.2 bars carry its volume."""
    volumes = [5_000 if i and c == closes[i - 1] else 500 for i, c in enumerate(closes)]
    return _session(closes, volumes, start=start)


def _zones(bars, **kwargs):
    kwargs.setdefault("minute_move", U)
    return candle_patterns.compute([candle_patterns.ZONE_KEY], bars, **kwargs)


class TestStallZone:
    def test_a_band_left_below_is_resistance_at_its_volume_peak(self):
        bars = _zone_session(SLIDE_IN + ZONE + SLIDE_OUT)
        (zone,) = _zones(bars)
        assert zone["kind"] == "zone" and zone["role"] == "resistance"
        assert pd.Timestamp(zone["x0"]) == _bar(8)
        assert pd.Timestamp(zone["x_end"]) == _bar(32)
        assert pd.Timestamp(zone["left_at"]) == _bar(33)
        assert (zone["y0"], zone["y1"]) == (100.0, 100.2)
        # The peak of the zone's own profile, not the middle of its band.
        assert abs(zone["level"] - 100.2) < 0.03
        assert zone["bars"] == 25 and zone["broken"] is False
        assert pd.Timestamp(zone["x1"]) == _bar(len(bars) - 1)

    def test_a_close_back_through_the_level_ends_its_line(self):
        bars = _zone_session(SLIDE_IN + ZONE + SLIDE_OUT + RALLY_BACK)
        (zone,) = _zones(bars)
        assert zone["broken"] is True
        # The first close above the level + u: 100.4, the third rally bar.
        assert pd.Timestamp(zone["x1"]) == _bar(len(SLIDE_IN + ZONE + SLIDE_OUT) + 2)

    def test_the_mirror_is_support(self):
        closes = [round(200.0 - c, 2) for c in SLIDE_IN + ZONE + SLIDE_OUT]
        (zone,) = _zones(_zone_session(closes))
        assert zone["role"] == "support"
        assert abs(zone["level"] - 99.8) < 0.03

    def test_a_band_price_is_still_in_is_forming(self):
        bars = _zone_session(SLIDE_IN + ZONE)
        (zone,) = _zones(bars)
        assert zone["role"] == "forming" and zone["left_at"] is None
        assert pd.Timestamp(zone["x1"]) == pd.Timestamp(zone["x_end"]) == _bar(len(bars) - 1)

    def test_too_short_or_too_wide_is_no_zone(self):
        bars = _zone_session(SLIDE_IN + ZONE + SLIDE_OUT)
        assert _zones(bars, zone_min_bars=30) == []
        # A band narrower than the zone's $0.20 of closes.
        assert _zones(bars, zone_width=1.5) == []

    def test_zones_do_not_overlap(self):
        second = [round(c - 1.2, 2) for c in ZONE]  # 98.8-99.0 after the slide out
        closes = SLIDE_IN + ZONE + SLIDE_OUT + second + [99.4, 99.8, 100.2]
        first, later = _zones(_zone_session(closes))
        assert pd.Timestamp(later["x0"]) > pd.Timestamp(first["x_end"])
        assert first["role"] == "resistance" and later["role"] == "support"

    def test_a_finished_zone_never_depends_on_later_bars(self):
        rng = np.random.default_rng(11)
        closes = list(np.round(100 + np.cumsum(rng.normal(0, U, 390)), 2))
        volumes = list(rng.lognormal(8, 0.6, 390).round())
        bars = _session(closes, volumes)
        full = _zones(bars, zone_min_bars=10)
        assert len(full) >= 3

        def key(zone):
            return zone["x0"], zone["x_end"], zone["role"], zone["level"]

        for k in (150, 250, 350):
            cutoff = pd.Timestamp(bars[k - 1]["t"])
            done = [key(z) for z in full if z["left_at"] and pd.Timestamp(z["left_at"]) <= cutoff]
            prefix = [key(z) for z in _zones(bars[:k], zone_min_bars=10) if z["role"] != "forming"]
            assert prefix == done

    def test_build_chart_draws_box_behind_the_candles_line_and_hover(self):
        bars = _zone_session(SLIDE_IN + ZONE + SLIDE_OUT)
        items = _zones(bars)
        fig = build_chart(bars, [], [], "AAPL", OPEN - pd.Timedelta(minutes=10), candle_patterns=items)
        zone = [t for t in fig.data if t.name == "Resistance zone"]
        box = [t for t in zone if t.fill == "toself"]
        assert len(box) == 1
        assert fig.data[0] is box[0]  # behind the candles and everything else
        assert [t.showlegend for t in zone].count(True) == 1
        hover = [t for t in zone if t.mode == "markers"]
        assert "Resistance zone</b> level 100.2" in hover[0].customdata[0]


class TestFormingZone:
    """`forming_zone`: the chart's grey box as one call on a session's closes,
    which Apple Trader's zone take re-runs on every bar."""

    def test_the_band_price_is_still_in(self):
        closes = SLIDE_IN + ZONE
        zone = candle_patterns.forming_zone(closes, U)
        assert zone["start"] == 8 and zone["bars"] == 25
        # The 20th bar of the zone is the first it is one at all.
        assert zone["confirmed"] == 8 + 19
        assert (zone["low"], zone["high"]) == (100.0, 100.2)
        assert zone["width"] == pytest.approx(3 * U)

    def test_none_once_price_has_left_or_before_the_length(self):
        assert candle_patterns.forming_zone(SLIDE_IN + ZONE + SLIDE_OUT, U) is None
        assert candle_patterns.forming_zone(SLIDE_IN + ZONE[:19], U) is None
        assert candle_patterns.forming_zone(SLIDE_IN + ZONE[:20], U)["bars"] == 20

    def test_the_settings_are_the_charts(self):
        closes = SLIDE_IN + ZONE
        assert candle_patterns.forming_zone(closes, U, min_bars=30) is None
        assert candle_patterns.forming_zone(closes, U, width_k=1.5) is None

    def test_it_agrees_with_the_charts_box_on_every_bar(self):
        rng = np.random.default_rng(7)
        closes = list(np.round(100 + np.cumsum(rng.normal(0, U, 390)), 2))
        bars = _session(closes, list(rng.lognormal(8, 0.6, 390).round()))
        seen = 0
        for k in range(20, 391, 7):
            drawn = [z for z in _zones(bars[:k], zone_min_bars=10) if z["role"] == "forming"]
            zone = candle_patterns.forming_zone(closes[:k], U, min_bars=10)
            if not drawn:
                assert zone is None
                continue
            seen += 1
            (box,) = drawn
            assert pd.Timestamp(box["x0"]) == _bar(zone["start"])
            assert (box["y0"], box["y1"]) == (zone["low"], zone["high"])
        assert seen >= 3
