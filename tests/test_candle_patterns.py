import pandas as pd
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
