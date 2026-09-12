from datetime import datetime, timezone

import pytest

from agent_stonks import clock, market_hours, premarket
from agent_stonks.premarket import PremarketBriefing
from agent_stonks.state import AppState


def _at(iso: str):
    """Pin the shared clock to an ET wall time expressed in UTC."""
    clock.set_simulated(datetime.fromisoformat(iso).replace(tzinfo=timezone.utc))


@pytest.fixture(autouse=True)
def _restore_clock():
    yield
    clock.clear()


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Every outside fetch the briefing makes is stubbed; tests override what they
    are about. Without this the generator quietly reaches yfinance and Alpaca."""
    monkeypatch.setattr(premarket, "fetch_news_with_fallback", lambda *a, **k: [])
    monkeypatch.setattr(premarket, "get_last_week_news", lambda *a, **k: [])
    monkeypatch.setattr(premarket, "_price_context", lambda sym: ("Price history:", {"7d": 100.0}))
    monkeypatch.setattr(premarket, "_macro_context", lambda days=30: "Macro:")
    monkeypatch.setattr(premarket, "_fundamentals_block", lambda sym: "")
    monkeypatch.setattr(premarket, "_earnings_block", lambda sym: "")
    monkeypatch.setattr(premarket, "_corporate_actions_block", lambda *a, **k: "")
    monkeypatch.setattr(premarket, "_targets_block", lambda sym, current_price=None: "")


def _briefing() -> PremarketBriefing:
    return PremarketBriefing(
        overall_bias="neutral",
        confidence="low",
        summary="s",
        catalysts=[],
        technical_levels=[],
        risk_factors=[],
        macro_context="m",
        key_levels_to_watch=[],
    )


def _bars(*specs) -> list[dict]:
    """(iso_ts, o, h, l, c, v) tuples -> bar dicts."""
    return [
        {"t": t, "o": o, "h": h, "l": lo, "c": c, "v": v} for t, o, h, lo, c, v in specs
    ]


class TestSessionPhase:
    @pytest.mark.parametrize(
        "utc_iso, expected",
        [
            ("2026-09-11T12:00:00", "premarket"),    # 08:00 ET Friday
            ("2026-09-11T13:29:00", "premarket"),    # 09:29 ET, one minute early
            ("2026-09-11T13:30:00", "open"),         # the bell
            ("2026-09-11T17:00:00", "open"),         # 13:00 ET
            ("2026-09-11T19:59:00", "open"),         # 15:59 ET
            ("2026-09-11T20:00:00", "after_hours"),  # 16:00 ET, the close
            ("2026-09-11T23:00:00", "after_hours"),
            ("2026-09-12T17:00:00", "weekend"),      # Saturday
            ("2026-09-13T17:00:00", "weekend"),      # Sunday
        ],
    )
    def test_phase_boundaries(self, utc_iso, expected):
        _at(utc_iso)
        assert market_hours.session_phase() == expected


class TestIntradayBlock:
    def test_reports_the_sessions_own_structure(self):
        _at("2026-09-11T17:00:00")  # 13:00 ET, session in progress
        bars = _bars(
            ("2026-09-11T13:30:00Z", 100.0, 102.0, 99.0, 101.0, 1000),
            ("2026-09-11T13:31:00Z", 101.0, 105.0, 100.0, 104.0, 2000),
            ("2026-09-11T13:32:00Z", 104.0, 104.5, 103.0, 103.5, 500),
        )
        text = premarket._intraday_block(bars, prev_close=98.0)

        assert "Opening print: 100.00" in text
        assert "Range so far: 99.00 - 105.00" in text
        assert "Last price: 103.50" in text
        # (103.5 - 99) / (105 - 99) = 75%
        assert "Position in today's range: 75%" in text
        assert "Change from the open: +3.50%" in text
        assert "previous close (98.00): +5.61%" in text
        assert "3,500 shares over 3 minute bars" in text

    def test_projects_a_relative_volume_pace_against_a_normal_day(self):
        # Raw share count says nothing on its own: heavy by 10:00 and heavy by
        # 15:30 are different statements, so the block projects today to a full
        # session before comparing. Two of 390 minutes at 1,000 shares each
        # projects to 390,000 against a 195,000-share average day = 2.00x.
        _at("2026-09-11T17:00:00")
        bars = _bars(
            ("2026-09-11T13:30:00Z", 100.0, 102.0, 99.0, 101.0, 1000),
            ("2026-09-11T13:31:00Z", 101.0, 102.0, 100.0, 101.5, 1000),
        )
        daily = [{"t": f"2026-09-0{d}T05:00:00Z", "v": 195_000} for d in (1, 2, 3, 4, 5)]

        text = premarket._intraday_block(bars, prev_close=None, daily_bars=daily)
        assert "Relative volume pace: 2.00x" in text

    def test_omits_the_pace_when_there_is_no_daily_baseline(self):
        _at("2026-09-11T17:00:00")
        bars = _bars(("2026-09-11T13:30:00Z", 100.0, 102.0, 99.0, 101.0, 1000))
        text = premarket._intraday_block(bars, prev_close=None, daily_bars=[])
        assert "Volume so far" in text
        assert "Relative volume pace" not in text

    def test_excludes_bars_from_before_the_bell(self):
        _at("2026-09-11T17:00:00")
        bars = _bars(
            ("2026-09-11T11:00:00Z", 90.0, 90.0, 90.0, 90.0, 10),   # premarket print
            ("2026-09-11T13:30:00Z", 100.0, 102.0, 99.0, 101.0, 1000),
        )
        text = premarket._intraday_block(bars, prev_close=None)

        # The 90.00 premarket bar must not become the session's open or its low.
        assert "Opening print: 100.00" in text
        assert "Range so far: 99.00 - 102.00" in text
        assert "1,000 shares" in text

    def test_empty_when_the_session_has_no_bars_yet(self):
        _at("2026-09-11T13:31:00Z".replace("Z", ""))
        assert premarket._intraday_block([], prev_close=100.0) == ""

    def test_empty_when_every_bar_predates_the_session(self):
        _at("2026-09-11T17:00:00")
        bars = _bars(("2026-09-11T11:00:00Z", 90.0, 90.0, 90.0, 90.0, 10))
        assert premarket._intraday_block(bars, prev_close=None) == ""

    @pytest.mark.parametrize("utc_iso", ["2026-09-11T21:00:00", "2026-09-12T17:00:00"])
    def test_silent_outside_the_session(self, utc_iso):
        # Outside trading hours the live buffer holds whatever the last REST
        # lookback caught -- on a Saturday, a sliver of Friday's extended-hours
        # tape. Summarising sixteen thin after-hours minutes as "the session"
        # would have the model quote an 8-cent range as the day's structure.
        # The completed day is already described by the daily close series.
        _at(utc_iso)
        bars = _bars(
            ("2026-09-11T19:44:00Z", 332.52, 332.60, 332.52, 332.58, 900),
            ("2026-09-11T19:59:00Z", 332.58, 332.60, 332.55, 332.58, 800),
        )
        assert premarket._intraday_block(bars, prev_close=None) == ""

    def test_the_block_never_reaches_the_prompt_outside_the_session(self, monkeypatch):
        _at("2026-09-12T17:00:00")  # Saturday
        seen = {}
        monkeypatch.setattr(
            premarket, "parse_structured",
            lambda p, k, m, system, user, rm: seen.update(user=user) or _briefing(),
        )
        bars = _bars(("2026-09-11T19:59:00Z", 332.58, 332.60, 332.55, 332.58, 800))

        premarket.generate_premarket_analysis("AAPL", "openai", "k", intraday_bars=bars)

        assert "TODAY'S SESSION SO FAR" not in seen["user"]


class TestGenerateIsPhaseAware:
    def _capture(self, monkeypatch) -> dict:
        seen: dict = {}

        def _parse(provider, api_key, model, system, user, response_model):
            seen["system"] = system
            seen["user"] = user
            return _briefing()

        monkeypatch.setattr(premarket, "parse_structured", _parse)
        return seen

    def test_mid_session_asks_for_an_intraday_briefing(self, monkeypatch):
        _at("2026-09-11T17:00:00")
        seen = self._capture(monkeypatch)
        bars = _bars(("2026-09-11T13:30:00Z", 100.0, 102.0, 99.0, 101.0, 1000))

        premarket.generate_premarket_analysis(
            "AAPL", "openai", "k", intraday_bars=bars, prev_close=98.0
        )

        assert "ALREADY UNDERWAY" in seen["system"]
        assert "intraday situation briefing for AAPL" in seen["user"]
        assert "TODAY'S SESSION SO FAR" in seen["user"]
        assert "regular session in progress" in seen["user"]

    def test_before_the_bell_asks_for_a_pre_market_briefing(self, monkeypatch):
        _at("2026-09-11T12:00:00")
        seen = self._capture(monkeypatch)

        premarket.generate_premarket_analysis("AAPL", "openai", "k")

        assert "before today's opening bell" in seen["system"]
        assert "pre-market briefing for AAPL" in seen["user"]
        assert "TODAY'S SESSION SO FAR" not in seen["user"]

    def test_after_the_close_frames_the_next_session(self, monkeypatch):
        _at("2026-09-11T21:00:00")
        seen = self._capture(monkeypatch)

        premarket.generate_premarket_analysis("AAPL", "openai", "k")

        assert "has CLOSED" in seen["system"]

    def test_an_explicit_phase_overrides_the_clock(self, monkeypatch):
        _at("2026-09-11T12:00:00")  # clock says premarket
        seen = self._capture(monkeypatch)

        premarket.generate_premarket_analysis("AAPL", "openai", "k", phase="open")

        assert "ALREADY UNDERWAY" in seen["system"]

    def test_target_upside_is_anchored_on_the_live_price_mid_session(self, monkeypatch):
        _at("2026-09-11T17:00:00")
        self._capture(monkeypatch)
        anchors = []
        monkeypatch.setattr(
            premarket, "_targets_block",
            lambda sym, current_price=None: anchors.append(current_price) or "",
        )
        bars = _bars(("2026-09-11T13:30:00Z", 100.0, 108.0, 99.0, 107.0, 1000))

        premarket.generate_premarket_analysis(
            "AAPL", "openai", "k", intraday_bars=bars
        )

        # Not the 100.0 daily close from _price_context -- quoting analyst
        # upside against a stale close while the stock is 7% up misstates every
        # distance-to-target in the briefing.
        assert anchors == [107.0]

    def test_target_upside_uses_the_daily_close_before_the_bell(self, monkeypatch):
        _at("2026-09-11T12:00:00")
        self._capture(monkeypatch)
        anchors = []
        monkeypatch.setattr(
            premarket, "_targets_block",
            lambda sym, current_price=None: anchors.append(current_price) or "",
        )

        premarket.generate_premarket_analysis("AAPL", "openai", "k")

        assert anchors == [100.0]


class TestGenerateForSymbols:
    def _app(self, *symbols):
        app = AppState()
        app.set_symbols(list(symbols) or ["AAPL"])
        return app

    def test_publishes_each_symbol_and_records_the_phase(self, monkeypatch):
        _at("2026-09-11T17:00:00")
        app = self._app("AAPL", "TSLA")
        monkeypatch.setattr(
            premarket, "generate_premarket_analysis", lambda **kw: _briefing()
        )

        premarket.generate_for_symbols(app, ["AAPL", "TSLA"], "openai", "k")

        assert sorted(app.premarket_briefings) == ["AAPL", "TSLA"]
        assert app.premarket_phase == "open"
        assert app.premarket_pending == []
        assert app.premarket_generated_at is not None
        assert "Intraday Situation Briefing" in app.premarket_status

    def test_one_symbol_failing_does_not_stop_the_rest(self, monkeypatch):
        _at("2026-09-11T12:00:00")
        app = self._app("AAPL", "TSLA")

        def _gen(**kw):
            if kw["symbol"] == "AAPL":
                raise RuntimeError("provider exploded")
            return _briefing()

        monkeypatch.setattr(premarket, "generate_premarket_analysis", _gen)

        premarket.generate_for_symbols(app, ["AAPL", "TSLA"], "openai", "k")

        assert list(app.premarket_briefings) == ["TSLA"]
        assert "provider exploded" in app.premarket_errors["AAPL"]
        assert app.premarket_pending == []

    def test_a_none_briefing_is_reported_rather_than_dropped(self, monkeypatch):
        _at("2026-09-11T12:00:00")
        app = self._app()
        monkeypatch.setattr(premarket, "generate_premarket_analysis", lambda **kw: None)

        premarket.generate_for_symbols(app, ["AAPL"], "openai", "k")

        assert app.premarket_briefings == {}
        assert "no briefing" in app.premarket_errors["AAPL"]

    def test_passes_the_symbols_live_bars_through(self, monkeypatch):
        _at("2026-09-11T17:00:00")
        app = self._app()
        bars = _bars(("2026-09-11T13:30:00Z", 100.0, 102.0, 99.0, 101.0, 1000))
        state = app.sym("AAPL")
        state.bars.extend(bars)
        state.prev_close = 98.0
        seen = {}

        def _gen(**kw):
            seen.update(kw)
            return _briefing()

        monkeypatch.setattr(premarket, "generate_premarket_analysis", _gen)
        premarket.generate_for_symbols(app, ["AAPL"], "openai", "k")

        assert seen["intraday_bars"] == bars
        assert seen["prev_close"] == 98.0
        assert seen["phase"] == "open"
        assert seen["daily_bars"] == state.daily_bars

    def test_a_rerun_clears_the_previous_results(self, monkeypatch):
        _at("2026-09-11T12:00:00")
        app = self._app()
        app.premarket_briefings = {"STALE": _briefing()}
        app.premarket_errors = {"STALE": "old"}
        monkeypatch.setattr(
            premarket, "generate_premarket_analysis", lambda **kw: _briefing()
        )

        premarket.generate_for_symbols(app, ["AAPL"], "openai", "k")

        assert list(app.premarket_briefings) == ["AAPL"]
        assert app.premarket_errors == {}


class TestLaunch:
    def test_reports_a_missing_key_instead_of_starting_a_thread(self):
        app = AppState()
        app.set_symbols(["AAPL"])

        assert premarket.launch_premarket_analysis(app, ["AAPL"], "openai", "") is False
        assert "No API key" in app.premarket_status

    def test_does_nothing_without_symbols(self):
        app = AppState()
        assert premarket.launch_premarket_analysis(app, [], "openai", "k") is False

    def test_starts_a_background_thread_when_it_can_run(self, monkeypatch):
        _at("2026-09-11T12:00:00")
        app = AppState()
        app.set_symbols(["AAPL"])
        monkeypatch.setattr(
            premarket, "generate_premarket_analysis", lambda **kw: _briefing()
        )

        assert premarket.launch_premarket_analysis(app, ["AAPL"], "openai", "k") is True
        for _ in range(200):
            if app.premarket_generated_at is not None:
                break
            import time as _t
            _t.sleep(0.01)
        assert list(app.premarket_briefings) == ["AAPL"]
