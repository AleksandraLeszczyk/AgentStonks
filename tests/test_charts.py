from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from agent_stonks import charts
from agent_stonks.charts import build_chart, build_performance_chart, empty_chart


SESSION_START = datetime(2024, 1, 15, 13, 20, tzinfo=timezone.utc)

BARS = [
    {"t": "2024-01-15T14:00:00Z", "o": 100.0, "h": 102.0, "l": 99.0, "c": 101.0, "v": 5000},
    {"t": "2024-01-15T14:01:00Z", "o": 101.0, "h": 103.0, "l": 100.5, "c": 102.0, "v": 3000},
    {"t": "2024-01-15T14:02:00Z", "o": 102.0, "h": 102.5, "l": 101.0, "c": 101.5, "v": 2000},
]

TRADES = [
    {"p": 100.5, "s": 100, "t": "2024-01-15T14:00:10Z"},
    {"p": 101.0, "s": 200, "t": "2024-01-15T14:00:30Z"},
    {"p": 101.5, "s": 150, "t": "2024-01-15T14:01:05Z"},
]

NEWS = [
    {
        "headline": "Apple beats earnings",
        "created_at": "2024-01-15T14:00:00Z",
        "url": "http://example.com",
    }
]


class TestEmptyChart:
    def test_returns_figure(self):
        assert isinstance(empty_chart(), go.Figure)

    def test_contains_custom_message(self):
        fig = empty_chart("Test message")
        texts = [a["text"] for a in fig.layout.annotations]
        assert "Test message" in texts

    def test_default_message(self):
        fig = empty_chart()
        texts = [a["text"] for a in fig.layout.annotations]
        assert any("symbol" in t.lower() or "start" in t.lower() for t in texts)


class TestBuildChart:
    def test_returns_figure_with_bars(self):
        fig = build_chart(BARS, [], [], "AAPL", SESSION_START)
        assert isinstance(fig, go.Figure)

    @pytest.mark.parametrize("show_momentum", [False, True])
    def test_the_volume_axis_is_logarithmic(self, show_momentum):
        fig = build_chart(BARS, [], [], "AAPL", SESSION_START, show_momentum=show_momentum)
        volume = next(t for t in fig.data if t.name == "Volume")
        axis = fig.layout["yaxis" + volume.yaxis[1:]]
        assert axis.type == "log"
        # The price panel stays linear.
        assert fig.layout.yaxis.type != "log"

    def test_empty_bars_returns_waiting_chart(self):
        fig = build_chart([], [], [], "AAPL", SESSION_START)
        texts = [a["text"] for a in fig.layout.annotations]
        assert any("Waiting" in t for t in texts)

    def test_title_contains_symbol_and_price(self):
        fig = build_chart(BARS, [], [], "TSLA", SESSION_START)
        assert "TSLA" in fig.layout.title.text
        assert "101.50" in fig.layout.title.text

    def test_works_with_empty_trades(self):
        fig = build_chart(BARS, [], [], "AAPL", SESSION_START)
        assert isinstance(fig, go.Figure)

    def test_works_with_empty_news(self):
        fig = build_chart(BARS, [], TRADES, "AAPL", SESSION_START)
        assert isinstance(fig, go.Figure)

    def test_works_with_trades_and_news(self):
        fig = build_chart(BARS, NEWS, TRADES, "AAPL", SESSION_START)
        assert isinstance(fig, go.Figure)

    def test_bars_before_session_start_are_filtered(self):
        old_bar = {"t": "2024-01-14T10:00:00Z", "o": 50.0, "h": 51.0, "l": 49.0, "c": 50.5, "v": 1000}
        fig = build_chart([old_bar] + BARS, [], [], "AAPL", SESSION_START)
        # Should still render (BARS are after session_start)
        assert "AAPL" in fig.layout.title.text

    def test_only_old_bars_returns_waiting_chart(self):
        old_bar = {"t": "2024-01-14T10:00:00Z", "o": 50.0, "h": 51.0, "l": 49.0, "c": 50.5, "v": 1000}
        fig = build_chart([old_bar], [], [], "AAPL", SESSION_START)
        texts = [a["text"] for a in fig.layout.annotations]
        assert any("Waiting" in t for t in texts)

    def test_decision_markers_plot_buy_and_sell(self):
        decisions = [
            {"ts": "2024-01-15T14:00:30Z", "action": "buy", "price": 100.5, "filled_quantity": 2, "status": "filled"},
            {"ts": "2024-01-15T14:01:30Z", "action": "sell", "price": 102.0, "filled_quantity": 2, "status": "filled"},
        ]
        fig = build_chart(BARS, [], [], "AAPL", SESSION_START, decisions=decisions)
        names = [t.name for t in fig.data]
        assert "Agent buy" in names
        assert "Agent sell" in names

    def test_decision_markers_ignored_when_no_price(self):
        decisions = [{"ts": "2024-01-15T14:00:30Z", "action": "sleep", "price": None, "filled_quantity": 0}]
        fig = build_chart(BARS, [], [], "AAPL", SESSION_START, decisions=decisions)
        names = [t.name for t in fig.data]
        assert "Agent sleep" not in names

    def test_decision_marker_hover_carries_reasoning(self):
        decisions = [
            {
                "ts": "2024-01-15T14:00:30Z", "action": "buy", "price": 100.5,
                "filled_quantity": 2, "status": "filled",
                "reasoning": "Opened below the predicted day range low.",
            },
        ]
        fig = build_chart(BARS, [], [], "AAPL", SESSION_START, decisions=decisions)
        trace = next(t for t in fig.data if t.name == "Agent buy")
        assert "%{customdata[1]}" in trace.hovertemplate
        assert trace.customdata[0][1] == "Opened below the predicted day range low."

    def test_decision_marker_hover_drops_why_without_reasoning(self):
        decisions = [
            {"ts": "2024-01-15T14:00:30Z", "action": "buy", "price": 100.5,
             "filled_quantity": 2, "status": "filled"},
        ]
        fig = build_chart(BARS, [], [], "AAPL", SESSION_START, decisions=decisions)
        trace = next(t for t in fig.data if t.name == "Agent buy")
        assert "Why" not in trace.hovertemplate
        assert "%{customdata[0]}" in trace.hovertemplate

    def test_decision_marker_hover_still_shows_quantity(self):
        decisions = [
            {"ts": "2024-01-15T14:00:30Z", "action": "sell", "price": 102.0,
             "filled_quantity": 3, "status": "filled", "reasoning": "Flattening into the close."},
        ]
        fig = build_chart(BARS, [], [], "AAPL", SESSION_START, decisions=decisions)
        trace = next(t for t in fig.data if t.name == "Agent sell")
        assert trace.customdata[0][0] == "3.00"


class TestHoverParagraph:
    def test_wraps_long_prose_into_lines(self):
        text = "word " * 60
        out = charts._hover_paragraph(text)
        assert "<br>" in out
        assert all(len(line) <= charts._HOVER_WRAP_COLS for line in out.split("<br>"))

    def test_escapes_angle_brackets(self):
        out = charts._hover_paragraph("bought because <model> said so")
        assert "<model>" not in out
        assert "&lt;model&gt;" in out

    def test_truncates_beyond_the_line_cap(self):
        out = charts._hover_paragraph("reason " * 400)
        assert len(out.split("<br>")) == charts._HOVER_WRAP_LINES
        assert out.endswith("\u2026")

    def test_empty_text_is_empty(self):
        assert charts._hover_paragraph("") == ""
        assert charts._hover_paragraph(None) == ""

    def test_no_decisions_does_not_error(self):
        fig = build_chart(BARS, [], [], "AAPL", SESSION_START, decisions=None)
        assert isinstance(fig, go.Figure)

    def test_price_alerts_plot_as_shapes(self):
        alerts = [
            {"field": "last_price", "condition": "above", "value": 150.0},
            {"field": "day_low", "condition": "below", "value": 95.0},
        ]
        fig = build_chart(BARS, [], [], "AAPL", SESSION_START, price_alerts=alerts)
        shape_levels = [s.y0 for s in fig.layout.shapes]
        assert 150.0 in shape_levels
        assert 95.0 in shape_levels
        texts = [a["text"] for a in fig.layout.annotations]
        assert any("above" in t and "150.00" in t for t in texts)
        assert any("below" in t and "95.00" in t for t in texts)

    def test_no_price_alerts_does_not_error(self):
        fig = build_chart(BARS, [], [], "AAPL", SESSION_START, price_alerts=None)
        assert isinstance(fig, go.Figure)


class TestBuildPerformanceChart:
    def test_no_points_returns_placeholder(self):
        fig = build_performance_chart([], [], "AAPL")
        texts = [a["text"] for a in fig.layout.annotations]
        assert any("No agent performance" in t for t in texts)

    def test_returns_figure_with_value_line(self):
        points = [
            {"ts": "2024-01-15T14:00:00Z", "price": 100.0, "cash": 1000.0, "position": 0.0, "value": 1000.0},
            {"ts": "2024-01-15T14:01:00Z", "price": 102.0, "cash": 1000.0, "position": 0.0, "value": 1000.0},
        ]
        fig = build_performance_chart(points, [], "AAPL")
        names = [t.name for t in fig.data]
        assert "Portfolio value" in names

    def test_markers_for_buy_and_sell_decisions(self):
        points = [{"ts": "2024-01-15T14:00:00Z", "price": 100.0, "cash": 900.0, "position": 1.0, "value": 1000.0}]
        markers = [
            {"ts": "2024-01-15T14:00:00Z", "action": "buy", "value": 1000.0},
            {"ts": "2024-01-15T14:00:30Z", "action": "sell", "value": 1005.0},
        ]
        fig = build_performance_chart(points, markers, "AAPL")
        names = [t.name for t in fig.data]
        assert "Agent buy" in names
        assert "Agent sell" in names


class TestFillIntradayGaps:
    GAPPY_BARS = [
        {"t": "2024-01-15T14:00:00Z", "o": 100.0, "h": 102.0, "l": 99.0, "c": 101.0, "v": 5000},
        {"t": "2024-01-15T14:01:00Z", "o": 101.0, "h": 103.0, "l": 100.5, "c": 102.0, "v": 3000},
        # 14:02 and 14:03 missing (no trades on the feed)
        {"t": "2024-01-15T14:04:00Z", "o": 102.0, "h": 102.5, "l": 101.0, "c": 101.5, "v": 2000},
    ]

    def _df(self, bars):
        df = pd.DataFrame(bars)
        df["t"] = pd.to_datetime(df["t"], utc=True)
        return df.sort_values("t").reset_index(drop=True)

    def test_fills_missing_buckets_with_flat_zero_volume_bars(self):
        from agent_stonks.charts import _fill_intraday_gaps

        filled = _fill_intraday_gaps(self._df(self.GAPPY_BARS))
        assert len(filled) == 5
        synth = filled[filled["synthetic"]]
        assert list(synth["t"].dt.strftime("%H:%M")) == ["14:02", "14:03"]
        # Flat at the previous close, zero volume
        assert (synth["o"] == 102.0).all()
        assert (synth["c"] == 102.0).all()
        assert (synth["v"] == 0).all()
        # Real bars untouched
        assert filled[~filled["synthetic"]]["v"].tolist() == [5000, 3000, 2000]

    def test_does_not_fill_across_days(self):
        from agent_stonks.charts import _fill_intraday_gaps

        bars = self.GAPPY_BARS + [
            {"t": "2024-01-16T14:00:00Z", "o": 103.0, "h": 104.0, "l": 102.0, "c": 103.5, "v": 1000},
        ]
        filled = _fill_intraday_gaps(self._df(bars))
        # Only the two intraday holes are filled -- not the overnight gap.
        assert int(filled["synthetic"].sum()) == 2

    def test_build_chart_with_fill_gaps_adds_no_trades_markers(self):
        fig = build_chart(self.GAPPY_BARS, [], [], "AAPL", SESSION_START, fill_gaps=True)
        names = [tr.name for tr in fig.data]
        assert "No trades" in names

    def test_build_chart_without_fill_gaps_has_no_markers(self):
        fig = build_chart(self.GAPPY_BARS, [], [], "AAPL", SESSION_START, fill_gaps=False)
        names = [tr.name for tr in fig.data]
        assert "No trades" not in names

    def test_build_chart_dedupes_mixed_timestamp_formats(self):
        # Same bucket delivered twice: REST 'Z' format and stream '+00:00' format.
        dup = dict(self.GAPPY_BARS[1], t="2024-01-15T14:01:00+00:00", c=999.0)
        fig = build_chart(self.GAPPY_BARS + [dup], [], [], "AAPL", SESSION_START)
        assert isinstance(fig, go.Figure)


class TestSessionRangebreaks:
    """Collapsing the hours the exchange is shut, on a multi-day chart.

    A run covering four days spends two thirds of its time axis on nights and
    weekends, which a time axis draws as blank space. These pin that the right
    stretches are removed -- and, just as importantly, that no real bar is.
    """

    def bars(self, days=("2024-01-15", "2024-01-16"), start="04:00", end="19:59"):
        """Extended-hours minute bars (04:00-19:59 ET), the shape SimLab stores."""
        out = []
        for day in days:
            idx = pd.date_range(
                f"{day} {start}", f"{day} {end}", freq="1min", tz="America/New_York"
            )
            for i, ts in enumerate(idx):
                out.append({
                    "t": ts.tz_convert("UTC").isoformat(),
                    "o": 100.0 + i * 0.001, "h": 100.5, "l": 99.5,
                    "c": 100.0 + i * 0.001, "v": 1000.0,
                })
        return out

    def test_one_break_per_night(self):
        breaks = charts.session_rangebreaks(
            self.bars(("2024-01-15", "2024-01-16", "2024-01-17"))
        )
        assert len(breaks) == 2

    def test_the_break_is_in_utc_wall_clock_like_the_axis(self):
        """Bar timestamps reach plotly as UTC, so the bounds have to be too.

        Emitting exchange-local bounds hides the wrong eight hours -- in
        January that would cut off the whole post-market instead of the night.
        """
        breaks = charts.session_rangebreaks(self.bars())
        lo, hi = breaks[0]["bounds"]
        # 19:59 ET + 1min = 20:00 EST = 01:00 UTC the next day; 04:00 EST = 09:00 UTC.
        assert lo == "2024-01-16T01:00:00"
        assert hi == "2024-01-16T09:00:00"

    def test_no_real_bar_falls_inside_a_break(self):
        bars = self.bars(("2024-01-15", "2024-01-16", "2024-01-17"))
        stamps = pd.to_datetime([b["t"] for b in bars], utc=True).tz_localize(None)
        for brk in charts.session_rangebreaks(bars):
            lo, hi = (pd.Timestamp(b) for b in brk["bounds"])
            assert not ((stamps >= lo) & (stamps < hi)).any()

    def test_a_weekend_is_one_break_not_three(self):
        # Friday to Monday: one gap between two consecutive bars.
        breaks = charts.session_rangebreaks(self.bars(("2024-01-19", "2024-01-22")))
        assert len(breaks) == 1
        lo, hi = breaks[0]["bounds"]
        assert pd.Timestamp(hi) - pd.Timestamp(lo) > pd.Timedelta(days=2)

    def test_a_single_day_needs_no_breaks(self):
        assert charts.session_rangebreaks(self.bars(("2024-01-15",))) == []

    def test_an_intraday_hole_is_not_collapsed(self):
        """A few minutes nobody traded is real elapsed time, not a closed
        exchange; hiding it would make the axis lie about how long a move took."""
        bars = self.bars(("2024-01-15",))
        thinned = bars[:100] + bars[130:]
        assert charts.session_rangebreaks(thinned) == []

    def test_no_bars_is_no_breaks(self):
        assert charts.session_rangebreaks([]) == []
        assert charts.session_rangebreaks(BARS[:1]) == []


class TestSessionMarkers:
    def figure(self, bars):
        fig = go.Figure(
            go.Candlestick(
                x=[b["t"] for b in bars], open=[b["o"] for b in bars],
                high=[b["h"] for b in bars], low=[b["l"] for b in bars],
                close=[b["c"] for b in bars],
            )
        )
        charts.add_session_markers(fig, bars)
        return fig

    def marks(self, fig):
        return sorted(
            s.x0 for s in fig.layout.shapes
            if s.line.color == charts.SESSION_MARKER_COLOR
        )

    def test_marks_the_regular_open_and_close_of_each_day(self):
        bars = TestSessionRangebreaks().bars(("2024-01-15", "2024-01-16"))
        marks = self.marks(self.figure(bars))
        # 09:30 and 16:00 EST are 14:30 and 21:00 UTC.
        assert marks == [
            "2024-01-15T14:30:00", "2024-01-15T21:00:00",
            "2024-01-16T14:30:00", "2024-01-16T21:00:00",
        ]

    def test_each_day_is_labelled_once_at_its_open(self):
        bars = TestSessionRangebreaks().bars(("2024-01-15", "2024-01-16"))
        fig = self.figure(bars)
        labels = [a["text"].strip() for a in fig.layout.annotations]
        assert labels == ["Mon 15 Jan", "Tue 16 Jan"]

    def test_a_session_the_bars_never_reach_gets_no_marker(self):
        """A run that stopped in the pre-market never saw an opening bell."""
        bars = TestSessionRangebreaks().bars(("2024-01-15",), end="08:30")
        assert self.marks(self.figure(bars)) == []

    def test_a_session_that_ended_early_keeps_its_open_and_loses_its_close(self):
        bars = TestSessionRangebreaks().bars(("2024-01-15",), end="11:00")
        assert self.marks(self.figure(bars)) == ["2024-01-15T14:30:00"]

    def test_no_bars_draws_nothing(self):
        fig = go.Figure()
        charts.add_session_markers(fig, [])
        assert not fig.layout.shapes


class TestMomentumPanel:
    """Bar-by-bar momentum and its change, in two rows under the volume.

    Momentum is `close - previous close`; its change is `momentum - previous
    momentum`; both one value per chart bar. These pin those definitions, that
    the panels are off unless asked for, that switching them on leaves the
    price and price-profile subplots where `add_model_overlays` addresses them,
    and that the warm-up is drawn as a gap with a note rather than as zeros.
    """

    @staticmethod
    def rth_bars(n: int, start: str = "2024-01-15T14:30:00Z", step_min: int = 1) -> list[dict]:
        """`n` bars of a drifting tape from the open (14:30Z = 09:30 ET)."""
        t0 = pd.Timestamp(start)
        steps = np.random.default_rng(7).normal(0.02, 0.05, n)
        out = []
        close = 100.0
        for i, step in enumerate(steps):
            open_, close = close, close + float(step)
            out.append({
                "t": (t0 + pd.Timedelta(minutes=i * step_min)).isoformat(),
                "o": open_, "h": max(open_, close) + 0.02,
                "l": min(open_, close) - 0.02, "c": close, "v": 1000 + i,
            })
        return out

    @staticmethod
    def trace(fig, name):
        (tr,) = [tr for tr in fig.data if tr.name == name]
        return tr

    def test_off_by_default(self):
        fig = build_chart(BARS, [], [], "AAPL", SESSION_START)
        assert "yaxis5" not in fig.layout
        names = [tr.name for tr in fig.data]
        assert "Momentum" not in names and "Momentum change" not in names

    def test_momentum_in_row_3_and_its_change_in_row_4(self):
        fig = build_chart(self.rth_bars(60), [], [], "AAPL", SESSION_START, show_momentum=True)
        # Two columns per row: row 3 col 1 is y5, row 4 col 1 is y7.
        assert self.trace(fig, "Momentum").yaxis == "y5"
        assert self.trace(fig, "Momentum change").yaxis == "y7"

    def gamma(self, bars, sign=1.0):
        return {"t": [b["t"] for b in bars],
                "value": [sign * 2.5e8 * (i + 1) for i in range(len(bars))], "note": ""}

    def test_net_gamma_sits_under_momentum_change(self):
        bars = self.rth_bars(60)
        fig = build_chart(bars, [], [], "AAPL", SESSION_START, show_momentum=True,
                          net_gamma=self.gamma(bars))
        # Row 5 col 1 is y9, under momentum Δ's y7.
        tr = self.trace(fig, "Net gamma")
        assert tr.yaxis == "y9"
        assert list(tr.y)[:2] == pytest.approx([250.0, 500.0])   # $M
        # The candles keep their height: the chart grows instead.
        assert fig.layout.height > 760

    def test_net_gamma_is_under_the_volume_without_momentum(self):
        bars = self.rth_bars(60)
        fig = build_chart(bars, [], [], "AAPL", SESSION_START,
                          net_gamma=self.gamma(bars, sign=-1.0))
        tr = self.trace(fig, "Net gamma")
        assert tr.yaxis == "y5"
        assert set(tr.marker.color) == {charts.PALETTE["down"]}

    def test_net_gamma_only_over_the_candles_drawn(self):
        """The series can reach back into pre-market bars the chart's start
        has cut; none of those are drawn."""
        bars = self.rth_bars(60)
        early = pd.Timestamp(bars[0]["t"]) - pd.Timedelta(hours=5)
        series = self.gamma(bars)
        series["t"] = [early.isoformat()] + series["t"]
        series["value"] = [9.9e9] + series["value"]
        fig = build_chart(bars, [], [], "AAPL", SESSION_START, show_momentum=True,
                          net_gamma=series)
        tr = self.trace(fig, "Net gamma")
        assert pd.to_datetime(list(tr.x), utc=True).min() >= pd.Timestamp(bars[0]["t"])
        assert len(tr.y) == len(bars)

    def test_no_chain_yet_says_so_in_the_panel(self):
        fig = build_chart(self.rth_bars(60), [], [], "AAPL", SESSION_START, show_momentum=True,
                          net_gamma={"t": [], "value": [], "note": "Waiting for the chain"})
        assert any(a.text == "Waiting for the chain" for a in fig.layout.annotations)

    def test_no_net_gamma_panel_by_default(self):
        fig = build_chart(self.rth_bars(60), [], [], "AAPL", SESSION_START, show_momentum=True)
        assert "Net gamma" not in [tr.name for tr in fig.data]
        assert fig.layout.height == 760

    def test_price_profile_keeps_its_axis_ids(self):
        # add_model_overlays addresses the profile column as x2/y2, so the
        # extra rows have to be appended rather than inserted.
        fig = build_chart(self.rth_bars(60), [], TRADES, "AAPL", SESSION_START, show_momentum=True)
        assert any(tr.yaxis == "y2" for tr in fig.data)

    @pytest.mark.parametrize("step_min", [1, 5])
    def test_momentum_is_the_change_from_the_previous_bar(self, step_min):
        bars = self.rth_bars(40, step_min=step_min)
        fig = build_chart(bars, [], [], "AAPL", SESSION_START, show_momentum=True)
        mom = self.trace(fig, "Momentum")
        closes = [b["c"] for b in bars]
        values = list(mom.y)
        assert len(values) == len(bars)
        assert pd.isna(values[0])
        assert values[1:] == pytest.approx([closes[i] - closes[i - 1] for i in range(1, len(closes))])
        # One bar per chart bar, at the chart's own resolution.
        xs = pd.to_datetime(list(mom.x), utc=True)
        assert (xs[1] - xs[0]) == pd.Timedelta(minutes=step_min)

    def test_change_is_momentum_minus_the_previous_momentum(self):
        bars = self.rth_bars(40)
        fig = build_chart(bars, [], [], "AAPL", SESSION_START, show_momentum=True)
        c = [b["c"] for b in bars]
        values = list(self.trace(fig, "Momentum change").y)
        assert pd.isna(values[0]) and pd.isna(values[1])
        expected = [(c[i] - c[i - 1]) - (c[i - 1] - c[i - 2]) for i in range(2, len(c))]
        assert values[2:] == pytest.approx(expected)

    def test_bars_are_colored_by_sign(self):
        fig = build_chart(self.rth_bars(40), [], [], "AAPL", SESSION_START, show_momentum=True)
        for name in ("Momentum", "Momentum change"):
            tr = self.trace(fig, name)
            for v, color in zip(tr.y, tr.marker.color):
                if pd.notna(v) and v != 0:
                    assert color == (charts.PALETTE["up"] if v > 0 else charts.PALETTE["down"])

    def test_each_session_starts_fresh(self):
        # Yesterday's close to today's open is an overnight gap, not a bar's move.
        day1 = self.rth_bars(10)
        day2 = [dict(b, c=b["c"] + 5.0, o=b["o"] + 5.0, h=b["h"] + 5.0, l=b["l"] + 5.0)
                for b in self.rth_bars(10, start="2024-01-16T14:30:00Z")]
        fig = build_chart(day1 + day2, [], [], "AAPL", "2024-01-15T00:00:00Z", show_momentum=True)
        values = list(self.trace(fig, "Momentum").y)
        assert pd.isna(values[0]) and pd.isna(values[10])
        assert all(abs(v) < 1.0 for v in values if pd.notna(v))

    def test_one_bar_draws_notes_instead(self):
        fig = build_chart(self.rth_bars(1), [], [], "AAPL", SESSION_START, show_momentum=True)
        names = [tr.name for tr in fig.data]
        assert "Momentum" not in names and "Momentum change" not in names
        texts = [a["text"] for a in fig.layout.annotations]
        assert any("second regular-session bar" in t for t in texts)
        assert any("third regular-session bar" in t for t in texts)

    def test_two_bars_draw_momentum_but_not_yet_its_change(self):
        fig = build_chart(self.rth_bars(2), [], [], "AAPL", SESSION_START, show_momentum=True)
        names = [tr.name for tr in fig.data]
        assert "Momentum" in names and "Momentum change" not in names
        texts = [a["text"] for a in fig.layout.annotations]
        assert any("third regular-session bar" in t for t in texts)

    def test_warming_up_panels_still_anchor_the_shared_time_axis(self):
        # An empty momentum panel left its x axis (matched to the price and
        # volume axes) without data, and plotly then autoranged the whole
        # group over 1912-2034: right after the open the candles vanished.
        bars = self.rth_bars(1)
        fig = build_chart(bars, [], [], "AAPL", SESSION_START, show_momentum=True)
        for axis in ("y5", "y7"):
            anchors = [tr for tr in fig.data if tr.yaxis == axis]
            assert len(anchors) == 1
            assert all(y is None for y in anchors[0].y)

    def test_bars_outside_the_regular_session_draw_a_note(self):
        # 13:30Z is 08:30 ET: after SESSION_START, so the bars are drawn, but
        # pre-market, so the momentum frame drops every one of them.
        pre = self.rth_bars(40, start="2024-01-15T13:30:00Z")[:50]
        pre = [b for b in pre if pd.Timestamp(b["t"]) < pd.Timestamp("2024-01-15T14:30:00Z")]
        fig = build_chart(pre, [], [], "AAPL", SESSION_START, show_momentum=True)
        texts = [a["text"] for a in fig.layout.annotations]
        assert any("second regular-session bar" in t for t in texts)

    def test_labels_the_latest_values_in_dollars(self):
        bars = self.rth_bars(60)
        fig = build_chart(bars, [], [], "AAPL", SESSION_START, show_momentum=True)
        c = [b["c"] for b in bars]
        mom = c[-1] - c[-2]
        change = mom - (c[-2] - c[-3])
        texts = [a["text"].strip() for a in fig.layout.annotations]
        for value in (mom, change):
            assert f"{'+' if value >= 0 else '-'}${abs(value):.2f}" in texts

    def test_the_axes_are_in_dollars(self):
        fig = build_chart(self.rth_bars(60), [], [], "AAPL", SESSION_START, show_momentum=True)
        assert fig.layout.yaxis5.tickprefix == "$"
        assert fig.layout.yaxis7.tickprefix == "$"
        assert fig.layout.yaxis5.title.text == "Momentum"
        assert fig.layout.yaxis7.title.text == "Momentum Δ"

    def test_only_the_zero_lines_are_drawn(self):
        fig = build_chart(self.rth_bars(60), [], [], "AAPL", SESSION_START, show_momentum=True)
        assert {s.y0 for s in fig.layout.shapes if s.yref == "y5"} == {0}
        assert {s.y0 for s in fig.layout.shapes if s.yref == "y7"} == {0}


    def ref_traces(self, fig):
        return [tr for tr in fig.data if tr.name == "Mean minute move"]

    # 09:30 ET = 570; a profile that grows by a cent per minute of the session.
    PROFILE = {570 + m: 0.10 + 0.01 * m for m in range(390)}

    def test_the_minute_profile_is_drawn_at_plus_and_minus(self):
        fig = build_chart(
            self.rth_bars(40), [], [], "AAPL", SESSION_START,
            show_momentum=True, minute_momentum_profile=self.PROFILE,
        )
        refs = self.ref_traces(fig)
        assert len(refs) == 2
        upper, lower = sorted(refs, key=lambda tr: np.nanmax(tr.y), reverse=True)
        # Each bar gets its own clock minute's value, not one flat level.
        assert list(upper.y) == pytest.approx([self.PROFILE[570 + i] for i in range(40)])
        assert list(lower.y) == pytest.approx([-self.PROFILE[570 + i] for i in range(40)])
        # On the momentum panel, not the change panel.
        assert {tr.yaxis for tr in refs} == {"y5"}
        texts = [a["text"] for a in fig.layout.annotations]
        assert any("by minute" in t for t in texts)

    def test_the_profile_is_a_dark_band_behind_the_bars(self):
        fig = build_chart(
            self.rth_bars(40), [], [], "AAPL", SESSION_START,
            show_momentum=True, minute_momentum_profile=self.PROFILE,
        )
        refs = self.ref_traces(fig)
        assert {tr.fill for tr in refs} == {"tozeroy"}
        bars = [tr for tr in fig.data if tr.name == "Momentum"]
        # Plotly puts bar traces over scatter traces unless zorder says otherwise.
        assert all(tr.zorder < (bars[0].zorder or 0) for tr in refs)

    def test_a_minute_the_profile_lacks_is_a_gap(self):
        profile = {m: v for m, v in self.PROFILE.items() if m != 575}
        fig = build_chart(
            self.rth_bars(10), [], [], "AAPL", SESSION_START,
            show_momentum=True, minute_momentum_profile=profile,
        )
        upper = max(self.ref_traces(fig), key=lambda tr: np.nanmax(tr.y))
        assert np.isnan(upper.y[5])
        assert upper.y[4] == pytest.approx(self.PROFILE[574])

    def test_no_minute_reference_on_coarser_bars(self):
        # A per-minute value says nothing about a 5-minute bar's move.
        fig = build_chart(
            self.rth_bars(40, step_min=5), [], [], "AAPL", SESSION_START,
            show_momentum=True, minute_momentum_profile=self.PROFILE,
        )
        assert not self.ref_traces(fig)

    @pytest.mark.parametrize("profile", [None, {}])
    def test_no_minute_reference_until_it_is_known(self, profile):
        fig = build_chart(
            self.rth_bars(40), [], [], "AAPL", SESSION_START,
            show_momentum=True, minute_momentum_profile=profile,
        )
        assert not self.ref_traces(fig)

    def test_no_minute_reference_while_momentum_warms_up(self):
        fig = build_chart(
            self.rth_bars(1), [], [], "AAPL", SESSION_START,
            show_momentum=True, minute_momentum_profile=self.PROFILE,
        )
        assert not self.ref_traces(fig)


    def change_ref_traces(self, fig):
        return [tr for tr in fig.data if tr.name == "Mean minute move change"]

    def test_the_change_profile_is_a_band_on_the_change_panel(self):
        change_profile = {m: v / 2 for m, v in self.PROFILE.items()}
        fig = build_chart(
            self.rth_bars(40), [], [], "AAPL", SESSION_START,
            show_momentum=True, minute_momentum_profile=self.PROFILE,
            minute_momentum_change_profile=change_profile,
        )
        refs = self.change_ref_traces(fig)
        assert len(refs) == 2
        assert {tr.yaxis for tr in refs} == {"y7"}
        assert {tr.fill for tr in refs} == {"tozeroy"}
        assert all(tr.zorder < 0 for tr in refs)
        upper = max(refs, key=lambda tr: np.nanmax(tr.y))
        assert list(upper.y) == pytest.approx([change_profile[570 + i] for i in range(40)])
        # The momentum panel keeps its own band.
        assert {tr.yaxis for tr in self.ref_traces(fig)} == {"y5"}
        texts = [a["text"] for a in fig.layout.annotations]
        assert any("\u0394| mean + 1\u03c3" in t for t in texts)

    def test_no_change_band_while_the_change_warms_up(self):
        # Two bars: one momentum bar, no change yet.
        fig = build_chart(
            self.rth_bars(2), [], [], "AAPL", SESSION_START,
            show_momentum=True, minute_momentum_profile=self.PROFILE,
            minute_momentum_change_profile=self.PROFILE,
        )
        assert self.ref_traces(fig)
        assert not self.change_ref_traces(fig)

    def agent_trace(self, fig, what):
        return [tr for tr in fig.data if tr.name == f"Agent {what}"]

    @pytest.mark.parametrize("n", [5, 15])
    def test_agent_lines_are_the_n_bar_momentum_and_its_change(self, n):
        bars = self.rth_bars(60)
        fig = build_chart(
            bars, [], [], "AAPL", SESSION_START, show_momentum=True,
            agent_momentum_bars=n, agent_momentum_label="Apple Trader",
        )
        c = [b["c"] for b in bars]
        (mom,) = self.agent_trace(fig, "momentum")
        (chg,) = self.agent_trace(fig, "momentum change")
        assert (mom.yaxis, chg.yaxis) == ("y5", "y7")
        assert mom.mode == "lines" and 0 < mom.opacity < 1
        # Averages per bar: (c - c[n]) / n, and (m1 - m1[n]) / n.
        m = [(c[i] - c[i - n]) / n for i in range(n, len(c))]
        assert all(pd.isna(v) for v in list(mom.y)[:n])
        assert list(mom.y)[n:] == pytest.approx(m)
        m1 = [None] + [c[i] - c[i - 1] for i in range(1, len(c))]
        d = [(m1[i] - m1[i - n]) / n for i in range(n + 1, len(c))]
        assert all(pd.isna(v) for v in list(chg.y)[:n + 1])
        assert list(chg.y)[n + 1:] == pytest.approx(d)
        texts = [a["text"] for a in fig.layout.annotations]
        assert sum(f"{n}-bar avg \u00b7 Apple Trader" in t for t in texts) == 2

    def test_a_one_bar_look_back_draws_no_extra_line(self):
        # It would be the bars themselves.
        fig = build_chart(
            self.rth_bars(40, step_min=5), [], [], "AAPL", SESSION_START,
            show_momentum=True, agent_momentum_bars=1,
        )
        assert not self.agent_trace(fig, "momentum")

    def test_no_agent_line_until_the_look_back_has_filled(self):
        fig = build_chart(
            self.rth_bars(10), [], [], "AAPL", SESSION_START,
            show_momentum=True, agent_momentum_bars=15,
        )
        assert "Momentum" in [tr.name for tr in fig.data]
        assert not self.agent_trace(fig, "momentum")

    def test_no_agent_line_over_a_warming_up_panel(self):
        fig = build_chart(
            self.rth_bars(1), [], [], "AAPL", SESSION_START,
            show_momentum=True, agent_momentum_bars=5,
        )
        assert not self.agent_trace(fig, "momentum")
        assert not self.agent_trace(fig, "momentum change")


class TestVolumeBand:
    """The per-source mean + 1 sigma band behind the volume bars."""

    MINUTE = 9 * 60  # BARS sit at 09:00-09:02 ET

    def band(self, level: float, history: str = "sip") -> dict:
        return {"per_minute": {self.MINUTE + i: level for i in range(3)}, "sessions": 5,
                "dates": [], "history": history}

    def baseline(self, span: int = 1, **by_source) -> dict:
        return {"key": "week_band", "label": "x", "span": span, "band_by_source": by_source}

    def tagged(self, *sources):
        return [{**b, "src": s} for b, s in zip(BARS, sources)]

    def trace(self, fig):
        return [t for t in fig.data if t.name == "Volume band"]

    def test_each_bar_reads_its_own_sources_band(self):
        fig = build_chart(
            self.tagged("sip", "iex", "sip"), [], [], "AAPL", SESSION_START,
            volume_baseline=self.baseline(sip=self.band(9000.0), iex=self.band(300.0, "iex")),
        )
        (band,) = self.trace(fig)
        assert list(band.y) == [9000.0, 300.0, 9000.0]
        assert list(band.customdata) == ["SIP", "IEX", "SIP"]
        assert band.fill == "tozeroy"
        assert band.zorder < 0  # behind the volume bars
        # Instead of, not as well as, the mean line and shape.
        assert not [t for t in fig.data if (t.name or "").startswith(("Mean volume", "Avg volume"))]

    def test_a_finnhub_bar_names_the_history_it_is_read_against(self):
        fig = build_chart(
            self.tagged("finnhub", "finnhub", "finnhub"), [], [], "AAPL", SESSION_START,
            volume_baseline=self.baseline(finnhub=self.band(9000.0)),
        )
        (band,) = self.trace(fig)
        assert band.customdata[0] == "SIP (Finnhub bar)"

    def test_a_source_without_a_band_is_a_gap(self):
        fig = build_chart(
            self.tagged("sip", "iex", "sip"), [], [], "AAPL", SESSION_START,
            volume_baseline=self.baseline(sip=self.band(9000.0)),
        )
        (band,) = self.trace(fig)
        assert np.isnan(band.y[1])

    def test_untagged_bars_get_no_band(self):
        fig = build_chart(
            BARS, [], [], "AAPL", SESSION_START,
            volume_baseline=self.baseline(sip=self.band(9000.0)),
        )
        assert not self.trace(fig)

    def test_a_band_for_another_bar_width_is_not_drawn(self):
        fig = build_chart(
            self.tagged("sip", "sip", "sip"), [], [], "AAPL", SESSION_START,
            volume_baseline=self.baseline(span=5, sip=self.band(9000.0)),
        )
        assert not self.trace(fig)

    def test_a_gap_filled_minute_takes_the_source_before_it(self):
        bars = [
            {**BARS[0], "src": "sip"},
            {**BARS[1], "src": "iex"},
            {**BARS[2], "t": "2024-01-15T14:03:00Z", "src": "sip"},  # 09:02 missing
        ]
        levels = {"per_minute": {self.MINUTE + i: 300.0 + i for i in range(4)}, "sessions": 5,
                  "dates": [], "history": "iex"}
        fig = build_chart(
            bars, [], [], "AAPL", SESSION_START, fill_gaps=True,
            volume_baseline=self.baseline(sip=self.band(9000.0), iex=levels),
        )
        (band,) = self.trace(fig)
        # 09:03 has no SIP level in `band` (it covers 09:00-09:02): a gap.
        assert list(band.y[:3]) == [9000.0, 301.0, 302.0]


class TestVolumeBaseline:
    """The "usual volume" references drawn under the live volume bars."""

    # 14:00 UTC is 09:00 ET on 2024-01-15, so the bars sit at minutes 540-542.
    MINUTE = 9 * 60

    def baseline(self, **over) -> dict:
        base = {
            "key": "week",
            "label": "Last trading week",
            "mean_per_minute": 1000.0,
            "per_minute": {self.MINUTE + i: 4000.0 for i in range(3)},
            "sessions": 5,
            "dates": ["2024-01-08"],
        }
        return {**base, **over}

    def traces(self, fig, prefix: str) -> list:
        return [t for t in fig.data if (t.name or "").startswith(prefix)]

    def test_nothing_is_drawn_without_a_baseline(self):
        fig = build_chart(BARS, [], [], "AAPL", SESSION_START, volume_baseline=None)
        assert not self.traces(fig, "Mean volume")
        assert not self.traces(fig, "Avg volume")

    def test_the_mean_is_a_flat_line(self):
        fig = build_chart(BARS, [], [], "AAPL", SESSION_START, volume_baseline=self.baseline())
        (line,) = self.traces(fig, "Mean volume")
        assert list(line.y) == [1000.0, 1000.0]

    def test_the_mean_spans_the_drawn_bars(self):
        fig = build_chart(BARS, [], [], "AAPL", SESSION_START, volume_baseline=self.baseline())
        (line,) = self.traces(fig, "Mean volume")
        assert [pd.Timestamp(x) for x in line.x] == [
            pd.Timestamp("2024-01-15T14:00:00Z"),
            pd.Timestamp("2024-01-15T14:02:00Z"),
        ]

    def test_the_shape_is_aligned_to_each_bars_clock_minute(self):
        shape = {self.MINUTE: 9000.0, self.MINUTE + 1: 8000.0, self.MINUTE + 2: 7000.0}
        fig = build_chart(
            BARS, [], [], "AAPL", SESSION_START,
            volume_baseline=self.baseline(per_minute=shape),
        )
        (ghost,) = self.traces(fig, "Avg volume")
        assert list(ghost.y) == [9000.0, 8000.0, 7000.0]

    def test_both_references_land_in_the_volume_panel(self):
        fig = build_chart(BARS, [], [], "AAPL", SESSION_START, volume_baseline=self.baseline())
        volume = next(t for t in fig.data if t.name == "Volume")
        for prefix in ("Mean volume", "Avg volume"):
            (trace,) = self.traces(fig, prefix)
            assert trace.yaxis == volume.yaxis

    def test_the_shape_is_semi_transparent(self):
        fig = build_chart(BARS, [], [], "AAPL", SESSION_START, volume_baseline=self.baseline())
        (ghost,) = self.traces(fig, "Avg volume")
        assert ghost.opacity < 1
        assert "rgba" in ghost.fillcolor

    def test_a_window_with_no_shape_still_draws_its_mean(self):
        # "This session": the mean line alone.
        fig = build_chart(
            BARS, [], [], "AAPL", SESSION_START,
            volume_baseline=self.baseline(key="session", label="This session", per_minute={}),
        )
        assert self.traces(fig, "Mean volume")
        assert not self.traces(fig, "Avg volume")

    def test_minutes_outside_the_window_are_gaps_not_zeroes(self):
        fig = build_chart(
            BARS, [], [], "AAPL", SESSION_START,
            volume_baseline=self.baseline(per_minute={self.MINUTE: 9000.0}),
        )
        (ghost,) = self.traces(fig, "Avg volume")
        assert ghost.y[0] == 9000.0
        assert all(v != v for v in ghost.y[1:])

    def test_the_session_count_is_named_on_both(self):
        fig = build_chart(BARS, [], [], "AAPL", SESSION_START, volume_baseline=self.baseline())
        for prefix in ("Mean volume", "Avg volume"):
            (trace,) = self.traces(fig, prefix)
            assert "5 sessions" in trace.name
            assert "Last trading week" in trace.name

    def test_one_session_is_not_called_sessions(self):
        fig = build_chart(
            BARS, [], [], "AAPL", SESSION_START,
            volume_baseline=self.baseline(label="Yesterday", sessions=1),
        )
        (line,) = self.traces(fig, "Mean volume")
        assert "1 session)" in line.name

    def test_the_mean_is_annotated_on_the_panel(self):
        fig = build_chart(BARS, [], [], "AAPL", SESSION_START, volume_baseline=self.baseline())
        assert any("mean 1,000" in a.text for a in fig.layout.annotations)

    def test_the_mean_label_sits_at_the_mean_on_the_log_axis(self):
        # Annotations on a log axis are placed in log10 units.
        fig = build_chart(BARS, [], [], "AAPL", SESSION_START, volume_baseline=self.baseline())
        (note,) = [a for a in fig.layout.annotations if "mean 1,000" in a.text]
        assert note.y == pytest.approx(3.0)

    def test_a_zero_mean_draws_nothing_on_the_log_axis(self):
        fig = build_chart(
            BARS, [], [], "AAPL", SESSION_START,
            volume_baseline=self.baseline(mean_per_minute=0.0),
        )
        assert not self.traces(fig, "Mean volume")

    def test_a_wider_timeframe_scales_the_reference_to_its_bars(self):
        # A 5-minute bar holds five minutes of volume; comparing it against a
        # one-minute average would make every bar look like an outlier.
        five_min = [
            {"t": f"2024-01-15T14:{m:02d}:00Z", "o": 100.0, "h": 101.0,
             "l": 99.0, "c": 100.5, "v": 5000}
            for m in (0, 5, 10)
        ]
        shape = {self.MINUTE + i: 100.0 for i in range(15)}
        fig = build_chart(
            five_min, [], [], "AAPL", SESSION_START,
            volume_baseline=self.baseline(per_minute=shape, mean_per_minute=100.0),
        )
        (line,) = self.traces(fig, "Mean volume")
        (ghost,) = self.traces(fig, "Avg volume")
        assert list(line.y) == [500.0, 500.0]
        assert list(ghost.y) == [500.0, 500.0, 500.0]


class TestBarMinutes:
    def make(self, stamps) -> pd.Series:
        return pd.Series(pd.to_datetime(stamps, utc=True))

    def test_one_minute_bars(self):
        assert charts._bar_minutes(
            self.make(["2024-01-15T14:00Z", "2024-01-15T14:01Z", "2024-01-15T14:02Z"])
        ) == 1

    def test_fifteen_minute_bars(self):
        assert charts._bar_minutes(
            self.make(["2024-01-15T14:00Z", "2024-01-15T14:15Z", "2024-01-15T14:30Z"])
        ) == 15

    def test_a_hole_does_not_widen_the_bar(self):
        # The median gap, not the mean: sessions have holes in them.
        assert charts._bar_minutes(
            self.make([
                "2024-01-15T14:00Z", "2024-01-15T14:01Z",
                "2024-01-15T14:40Z", "2024-01-15T14:41Z",
            ])
        ) == 1

    def test_a_single_bar_falls_back_to_one_minute(self):
        assert charts._bar_minutes(self.make(["2024-01-15T14:00Z"])) == 1


class TestDayRangeLines:
    def _annotations(self, fig):
        return [a.text.strip() for a in fig.layout.annotations if a.text]

    def test_both_lines_drawn(self):
        fig = build_chart(BARS, [], TRADES, "AAPL", SESSION_START,
                          day_range_lines={"day_low": 99.0, "day_high": 103.0})
        texts = self._annotations(fig)
        assert "Day low 99.00" in texts
        assert "Day high 103.00" in texts
        levels = {s.y0 for s in fig.layout.shapes if s.y0 == s.y1}
        assert {99.0, 103.0} <= levels

    def test_only_selected_line_drawn(self):
        fig = build_chart(BARS, [], TRADES, "AAPL", SESSION_START,
                          day_range_lines={"day_high": 103.0})
        texts = self._annotations(fig)
        assert "Day high 103.00" in texts
        assert not any(t.startswith("Day low") for t in texts)

    def test_none_by_default(self):
        fig = build_chart(BARS, [], TRADES, "AAPL", SESSION_START)
        assert not any(t.startswith("Day ") for t in self._annotations(fig))


class TestOptionWalls:
    def _annotations(self, fig):
        return [a.text.strip() for a in fig.layout.annotations if a.text]

    def test_both_walls_drawn(self):
        fig = build_chart(BARS, [], TRADES, "AAPL", SESSION_START,
                          option_walls={"call_wall": 105.0, "put_wall": 95.0})
        texts = self._annotations(fig)
        assert "Call wall 105.00" in texts
        assert "Put wall 95.00" in texts
        levels = {s.y0 for s in fig.layout.shapes if s.y0 == s.y1}
        assert {105.0, 95.0} <= levels

    def test_only_selected_wall_drawn(self):
        fig = build_chart(BARS, [], TRADES, "AAPL", SESSION_START,
                          option_walls={"put_wall": 95.0})
        texts = self._annotations(fig)
        assert "Put wall 95.00" in texts
        assert not any(t.startswith("Call wall") for t in texts)

    def test_no_walls_by_default(self):
        fig = build_chart(BARS, [], TRADES, "AAPL", SESSION_START)
        assert not any("wall" in t for t in self._annotations(fig))

    def test_far_wall_is_an_edge_label_not_a_line(self):
        # Session trades 99-103; a put wall 20% below must not stretch the axis.
        fig = build_chart(BARS, [], TRADES, "AAPL", SESSION_START,
                          option_walls={"call_wall": 103.5, "put_wall": 80.0})
        levels = {s.y0 for s in fig.layout.shapes if s.y0 == s.y1}
        assert 103.5 in levels
        assert 80.0 not in levels
        edge = [a for a in fig.layout.annotations if a.text and "Put wall" in a.text]
        assert edge and edge[0].yref == "y domain" and edge[0].y == 0
        assert "▼" in edge[0].text
