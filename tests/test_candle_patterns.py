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
