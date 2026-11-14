"""The Tuning tab's per-session context: the VIX open and the pre-market bias.

The briefing is an LLM call on a past morning, so what matters here is what it
is *shown*: nothing dated at or after 09:25 ET that day may reach the prompt.
"""
from datetime import datetime

import pandas as pd
import pytest

from agent_stonks import premarket
from agent_stonks.market_hours import MARKET_TZ
from agent_stonks.premarket import PremarketBriefing
from simlab import session_context as sc


@pytest.fixture(autouse=True)
def _tmp_context(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "CONTEXT_DIR", tmp_path)
    monkeypatch.setattr(sc, "VIX_OPEN_PATH", tmp_path / "vix_open.json")
    monkeypatch.setattr(sc, "BRIEFING_DIR", tmp_path / "briefings")


def _briefing(bias="bullish") -> PremarketBriefing:
    return PremarketBriefing(
        overall_bias=bias, confidence="medium", summary="s", catalysts=[],
        technical_levels=[], risk_factors=[], macro_context="m", key_levels_to_watch=[],
    )


class TestVixOpens:
    def test_downloads_only_the_days_not_kept(self):
        calls = []

        def fetch(start, end):
            calls.append((start.isoformat(), end.isoformat()))
            return {"2026-09-14": 17.5, "2026-09-15": 17.57}

        assert sc.vix_opens(["2026-09-14", "2026-09-15"], fetch) == {
            "2026-09-14": 17.5, "2026-09-15": 17.57,
        }
        assert calls == [("2026-09-14", "2026-09-15")]
        # Kept on disk: asking again downloads nothing.
        assert sc.vix_opens(["2026-09-15"], fetch) == {"2026-09-15": 17.57}
        assert len(calls) == 1

    def test_a_failed_download_leaves_the_day_out(self):
        def fetch(start, end):
            raise RuntimeError("yahoo down")

        assert sc.vix_opens(["2026-09-14"], fetch) == {}
        assert not sc.VIX_OPEN_PATH.exists()


class TestPointInTime:
    def test_closes_and_indicators_stop_the_day_before(self, monkeypatch):
        # Daily bars stamped at midnight ET, in UTC.
        monkeypatch.setattr(sc.sim_data, "load_daily_bars", lambda sym, feed: [
            {"t": "2026-09-11T04:00:00Z", "c": 230.0},
            {"t": "2026-09-14T04:00:00Z", "c": 231.0},
            {"t": "2026-09-15T04:00:00Z", "c": 999.0},
        ])
        monkeypatch.setattr(sc.sim_data, "load_market_indicators", lambda: {
            "vix": [{"date": "2026-09-11", "close": 15.8}, {"date": "2026-09-14", "close": 99.0}],
        })
        closes = sc._closes_before("AAPL", "sip", "2026-09-14")
        assert list(closes) == [230.0]
        assert list(sc._indicators_before("2026-09-14")["vix"]) == [15.8]

    def test_stored_news_stops_at_the_briefing_time(self, monkeypatch):
        stored = {
            "2026-09-13": [{"headline": "weekend", "created_at": "2026-09-13T15:00:00Z"}],
            # 09:20 ET is 13:20Z; 09:40 ET is 13:40Z.
            "2026-09-14": [
                {"headline": "pre-open", "created_at": "2026-09-14T13:20:00Z"},
                {"headline": "after the open", "created_at": "2026-09-14T13:40:00Z"},
            ],
        }
        monkeypatch.setattr(
            sc.sim_data, "load_news", lambda sym, day: stored.get(day.isoformat(), [])
        )
        items = sc._stored_news_before("AAPL", sc.briefing_as_of("2026-09-14"))
        assert [i["headline"] for i in items] == ["pre-open", "weekend"]

    def test_the_prompt_never_sees_the_session_or_after(self, monkeypatch):
        seen = {}

        def fake_parse(provider, api_key, model, system, user, response_model):
            seen["system"], seen["user"] = system, user
            return _briefing()

        monkeypatch.setattr(premarket, "parse_structured", fake_parse)
        closes = pd.Series(
            [100.0, 110.0, 5000.0],
            index=pd.to_datetime(["2026-09-10", "2026-09-11", "2026-09-14"]),
        )
        as_of = datetime(2026, 9, 14, 9, 25, tzinfo=MARKET_TZ)
        premarket.generate_premarket_from_data(
            "aapl", "anthropic", "k", as_of=as_of, closes=closes,
            indicators={"vix": pd.Series([15.8], index=pd.to_datetime(["2026-09-11"]))},
            news_items=[{"headline": "H", "summary": "", "created_at": "2026-09-13T15:00:00Z"}],
        )
        assert "Analysis time: 2026-09-14 09:25 ET (before today's open)" in seen["user"]
        assert "7d: +10.0%" in seen["user"] and "last close 110" in seen["user"]
        assert "5000" not in seen["user"]
        assert "VIX 15.8" in seen["user"]
        assert "before today's opening bell" in seen["system"]


class TestSessionBiases:
    def test_cached_days_are_read_and_the_rest_generated(self, monkeypatch):
        generated = []

        def generate(symbol, day, feed, provider, model, api_key, key, secret):
            generated.append((day, feed))
            if day == "2026-09-15":
                raise RuntimeError("rate limited")
            return {"day": day, "bias": "bearish", "confidence": "low", "summary": ""}

        path = sc.briefing_path("AAPL", "2026-09-11", "anthropic", "m")
        path.parent.mkdir(parents=True)
        path.write_text('{"day": "2026-09-11", "bias": "bullish"}')

        records, errors = sc.session_biases(
            "AAPL", {"sip": ["2026-09-11", "2026-09-14"], "yfinance": ["2026-09-15"]},
            "anthropic", "m", "k", generate=generate,
        )
        assert records["2026-09-11"]["bias"] == "bullish"
        assert records["2026-09-14"]["bias"] == "bearish"
        assert errors == {"2026-09-15": "rate limited"}
        assert sorted(generated) == [("2026-09-14", "sip"), ("2026-09-15", "yfinance")]

    def test_generate_briefing_caches_what_it_wrote(self, monkeypatch):
        monkeypatch.setattr(sc, "_closes_before", lambda *a: pd.Series(dtype=float))
        monkeypatch.setattr(sc, "_indicators_before", lambda day: {})
        monkeypatch.setattr(sc, "_stored_news_before", lambda sym, as_of: [])
        monkeypatch.setattr(premarket, "_earnings_block", lambda sym, as_of=None: "")
        monkeypatch.setattr(
            premarket, "generate_premarket_from_data", lambda *a, **k: _briefing("neutral")
        )
        record = sc.generate_briefing("aapl", "2026-09-14", "sip", "anthropic", "m", "k")
        assert record["bias"] == "neutral" and record["news_source"] == "store"
        assert sc.cached_briefing("AAPL", "2026-09-14", "anthropic", "m") == record


class TestCachedBiases:
    @staticmethod
    def _keep(symbol, day, provider, model, bias):
        path = sc.briefing_path(symbol, day, provider, model)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f'{{"day": "{day}", "bias": "{bias}"}}')

    def test_reads_the_model_covering_most_days_and_writes_nothing(self, monkeypatch):
        monkeypatch.setattr(
            sc, "generate_briefing", lambda *a, **k: pytest.fail("a briefing was written")
        )
        self._keep("AAPL", "2026-09-14", "gemini", "g", "bearish")
        self._keep("AAPL", "2026-09-14", "openai", "o", "bullish")
        self._keep("AAPL", "2026-09-15", "openai", "o", "neutral")
        before = sorted(sc.BRIEFING_DIR.rglob("*.json"))
        source, records = sc.cached_biases(
            "AAPL", ["2026-09-14", "2026-09-15", "2026-09-16"], prefer=("gemini", "g")
        )
        assert source == ("openai", "o")
        assert {d: r["bias"] for d, r in records.items()} == {
            "2026-09-14": "bullish", "2026-09-15": "neutral",
        }
        assert sorted(sc.BRIEFING_DIR.rglob("*.json")) == before

    def test_a_tie_goes_to_the_preferred_model(self):
        self._keep("AAPL", "2026-09-14", "gemini", "g", "bearish")
        self._keep("AAPL", "2026-09-14", "openai", "o", "bullish")
        assert sc.cached_biases("AAPL", ["2026-09-14"], prefer=("gemini", "g"))[0] == (
            "gemini", "g"
        )

    def test_nothing_cached_is_no_source(self):
        assert sc.cached_biases("AAPL", ["2026-09-14"]) == (None, {})


class TestTuningContextWritesNoBriefing:
    def test_the_tuning_tab_reads_cached_briefings_only(self, monkeypatch):
        from simlab import app as sim_app

        monkeypatch.setattr(
            sim_app.sim_session, "session_biases",
            lambda *a, **k: pytest.fail("the Tuning tab wrote a briefing"),
        )
        monkeypatch.setattr(
            sim_app.sim_session, "generate_briefing",
            lambda *a, **k: pytest.fail("the Tuning tab wrote a briefing"),
        )
        monkeypatch.setattr(sim_app.sim_session, "vix_opens", lambda days: {})
        monkeypatch.setattr(
            sim_app.sim_session, "briefing_provider", lambda: ("openai", "o", "key")
        )
        TestCachedBiases._keep("AAPL", "2026-09-14", "openai", "o", "bearish")
        cell = {"overrides": {}, "profit": 1.0,
                "daily": {"2026-09-14": 1.0, "2026-09-15": 0.0}}
        job = {
            "spec": {"base": {"ticker": "AAPL"}, "datasets": [{"name": "wk", "feed": "sip"}]},
            "cells": {"wk": [cell]},
            "best": cell,
        }
        monkeypatch.setattr(sim_app.sim_tuning, "is_scored", lambda c: bool(c))
        monkeypatch.setattr(sim_app, "_tuning_shown_cell", lambda job, name, chosen: cell)
        context = sim_app._tuning_session_context(job, chosen={})
        assert context["provider"] == "openai · o"
        assert context["biases"]["2026-09-14"]["bias"] == "bearish"
        assert context["missing"] == ["2026-09-15"]


class TestSessionChart:
    def test_each_session_label_carries_its_bias_and_vix(self, monkeypatch):
        from simlab import app as sim_app

        monkeypatch.setattr(sim_app, "_tuning_session_bars", lambda sym, feed, days: {})
        cell = {"overrides": {}, "profit": 10.0,
                "daily": {"2026-09-14": 12.0, "2026-09-15": -2.0}}
        job = {
            "spec": {
                "base": {"ticker": "AAPL"},
                "datasets": [{"name": "wk", "feed": "sip"}],
            },
            "cells": {"wk": [cell]},
            "best": cell,
        }
        monkeypatch.setattr(sim_app.sim_tuning, "is_scored", lambda c: bool(c))
        context = {
            "vix": {"2026-09-14": 17.5},
            "biases": {"2026-09-14": {"bias": "bearish", "confidence": "high"}},
        }
        fig = sim_app._tuning_daily_chart(job, context, chosen={})
        ticks = dict(zip(fig.layout.xaxis4.tickvals, fig.layout.xaxis4.ticktext))
        assert "▼ bearish" in ticks["2026-09-14"] and "VIX 17.5" in ticks["2026-09-14"]
        assert "no bias" in ticks["2026-09-15"] and "VIX –" in ticks["2026-09-15"]
        assert "bearish (high confidence)" in fig.data[0].hovertext[0]

    def test_under_the_candles_each_sessions_move_and_range(self, monkeypatch):
        from simlab import app as sim_app

        monkeypatch.setattr(
            sim_app, "_tuning_session_bars",
            lambda sym, feed, days: {
                "2026-09-14": {"o": 230.0, "h": 234.0, "l": 229.0, "c": 233.0},
                "2026-09-15": {"o": 233.0, "h": 233.5, "l": 228.0, "c": 229.0},
            },
        )
        cell = {"overrides": {}, "profit": 10.0,
                "daily": {"2026-09-14": 12.0, "2026-09-15": -2.0}}
        job = {
            "spec": {"base": {"ticker": "AAPL"}, "datasets": [{"name": "wk", "feed": "sip"}]},
            "cells": {"wk": [cell]},
            "best": cell,
        }
        monkeypatch.setattr(sim_app.sim_tuning, "is_scored", lambda c: bool(c))
        fig = sim_app._tuning_daily_chart(job, chosen={})
        traces = {t.name: t for t in fig.data}
        move, span = traces["Open → close"], traces["High − low"]
        assert (move.xaxis, move.yaxis, span.yaxis) == ("x3", "y3", "y4")
        assert list(move.y) == [3.0, -4.0] and list(span.y) == [5.0, 5.5]
        assert list(move.marker.color) == [sim_app.PALETTE["up"], sim_app.PALETTE["down"]]
        assert "open 233.00 → close 229.00: -4.00 (-1.72%)" in move.customdata[1]
        # A profit's hover carries its session's move and range too.
        profit = traces["Selected on wk"]
        assert "high 234.00 − low 229.00: 5.00" in profit.hovertext[0]

    def test_the_price_panel_is_one_daily_candle_per_session(self, monkeypatch):
        from simlab import app as sim_app

        stored = [
            {"t": "2026-09-14T04:00:00Z", "o": 230.0, "h": 234.0, "l": 229.0, "c": 233.0},
            {"t": "2026-09-15T04:00:00Z", "o": 233.0, "h": 233.5, "l": 228.0, "c": 229.0},
            {"t": "2026-09-16T04:00:00Z", "o": 1.0, "h": 1.0, "l": 1.0, "c": 1.0},
        ]
        monkeypatch.setattr(sim_app.sim_data, "load_daily_bars", lambda sym, feed: stored)
        cell = {"overrides": {}, "profit": 10.0,
                "daily": {"2026-09-14": 12.0, "2026-09-15": -2.0}}
        job = {
            "spec": {"base": {"ticker": "AAPL"}, "datasets": [{"name": "wk", "feed": "sip"}]},
            "cells": {"wk": [cell]},
            "best": cell,
        }
        monkeypatch.setattr(sim_app.sim_tuning, "is_scored", lambda c: bool(c))
        fig = sim_app._tuning_daily_chart(job, chosen={})
        candles = [t for t in fig.data if t.type == "candlestick"]
        assert len(candles) == 1 and candles[0].xaxis == "x2"
        assert list(candles[0].x) == ["2026-09-14", "2026-09-15"]
        assert list(candles[0].open) == [230.0, 233.0]
        assert list(candles[0].high) == [234.0, 233.5]
        assert list(candles[0].low) == [229.0, 228.0]
        assert list(candles[0].close) == [233.0, 229.0]
        assert not [t for t in fig.data if t.type == "bar" and t.xaxis == "x2"]
        assert fig.layout.xaxis2.rangeslider.visible is False
