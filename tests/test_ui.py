from agent_stonks import config
from agent_stonks.state import AppState
from agent_stonks.ui import (
    _agent_momentum,
    _chart_start,
    _cash_label,
    _venue_badge,
    _portfolio_value_label,
    _quote_html,
    _starting_value_label,
    build_news_html,
    wrap_text,
)


class TestWrapText:
    def test_wraps_long_text(self):
        result = wrap_text("word " * 20, width=10)
        assert "<br>" in result

    def test_empty_string_returns_empty(self):
        assert wrap_text("") == ""

    def test_none_returns_empty(self):
        assert wrap_text(None) == ""  # type: ignore[arg-type]

    def test_short_text_no_wrap(self):
        assert wrap_text("short", width=30) == "short"


class TestBuildNewsHtml:
    def test_returns_no_news_message_when_empty(self):
        html = build_news_html([], "AAPL")
        assert "No recent news" in html
        assert "AAPL" in html

    def test_renders_headline(self):
        news = [
            {
                "headline": "Apple hits record high",
                "summary": "Apple stock rallied on strong earnings.",
                "created_at": "2024-01-15T14:30:00Z",
                "url": "http://example.com/apple",
                "source": "Reuters",
            }
        ]
        html = build_news_html(news, "AAPL")
        assert "Apple hits record high" in html
        assert "Reuters" in html
        assert "http://example.com/apple" in html

    def test_caps_at_12_items(self):
        news = [
            {
                "headline": f"Headline {i}",
                "summary": "text",
                "created_at": "2024-01-15T14:30:00Z",
                "url": f"http://example.com/{i}",
                "source": "src",
            }
            for i in range(20)
        ]
        html = build_news_html(news, "AAPL")
        # Only the first 12 headlines should appear
        assert "Headline 11" in html
        assert "Headline 12" not in html

    def test_handles_missing_summary(self):
        news = [
            {
                "headline": "No summary here",
                "summary": None,
                "created_at": "2024-01-15T14:30:00Z",
                "url": "http://example.com",
                "source": "AP",
            }
        ]
        html = build_news_html(news, "AAPL")
        assert "No summary here" in html

    def test_marks_yahoo_articles(self):
        base = {"summary": "", "created_at": "2024-01-15T14:30:00Z", "url": "http://example.com"}
        news = [
            {**base, "id": "yf-1", "headline": "From Yahoo", "source": "Reuters", "feed": "yfinance"},
            {**base, "id": 2, "headline": "From Alpaca", "source": "benzinga"},
        ]
        html = build_news_html(news, "AAPL")
        assert html.count("via Yahoo") == 1
        assert "Reuters" in html


class TestMoneyLabelsNameTheAccount:
    """Paper and live are two separate Alpaca accounts with separate balances.
    A figure that does not say which one it came from is one the user has to
    guess about -- and guessing wrong about live money is the expensive way."""

    def _state(self, mode):
        state = AppState()
        state.trading_mode = mode
        return state

    def test_local_simulation_keeps_its_old_wording(self):
        state = self._state("local")
        assert _portfolio_value_label(state) == "Portfolio value"
        assert _starting_value_label(state) == "Starting budget"
        assert _cash_label(state) == "Paper cash"

    def test_the_paper_account_is_named(self):
        state = self._state("alpaca_paper")
        assert "paper" in _portfolio_value_label(state)
        assert "paper" in _cash_label(state)

    def test_the_live_account_is_named_unmistakably(self):
        state = self._state("alpaca_live")
        assert "LIVE" in _portfolio_value_label(state)
        assert "LIVE" in _cash_label(state)
        assert "paper" not in _portfolio_value_label(state).lower()

    def test_a_real_account_has_a_starting_value_not_a_budget(self):
        # Nothing was budgeted: the run opened on whatever the account held.
        for mode in ("alpaca_paper", "alpaca_live"):
            assert _starting_value_label(self._state(mode)) == "Value at start"


class TestVenueBadge:
    """The status line has to say where orders are going. The bug it replaces:
    a session that asked for live, was refused for want of live keys and
    degraded to simulation, said "Local simulation" under a dropdown still
    reading "Alpaca LIVE" -- and the one message explaining why had already
    scrolled away with the rerun that produced it."""

    def test_every_mode_has_a_distinguishable_badge(self):
        badges = {_venue_badge(m) for m in ("local", "alpaca_paper", "alpaca_live")}
        assert len(badges) == 3

    def test_live_is_not_mistakable_for_paper(self):
        assert "LIVE" in _venue_badge("alpaca_live")
        assert "paper" not in _venue_badge("alpaca_live").lower()
        assert "paper" in _venue_badge("alpaca_paper").lower()
        assert "LIVE" not in _venue_badge("alpaca_paper")

    def test_simulation_says_so(self):
        assert "simulation" in _venue_badge("local").lower()

    def test_an_unknown_mode_falls_back_to_its_own_name(self):
        assert _venue_badge("something_else") == "something_else"


class TestTheRequestedModeIsRemembered:
    """`trading_mode` is what the run got; `trading_mode_requested` is what was
    asked for. Keeping both is what lets the UI say they differ."""

    def test_nothing_is_requested_before_the_first_start(self):
        assert AppState().trading_mode_requested == ""

    def test_a_downgrade_leaves_the_two_disagreeing(self):
        # What resolve_broker does when live keys are missing.
        state = AppState()
        state.trading_mode_requested = "alpaca_live"
        state.trading_mode = "local"
        assert state.trading_mode != state.trading_mode_requested


class TestTheLiveSourceChoice:
    """The sidebar offers the source and the Alpaca feed as one choice."""

    def test_offers_finnhub_and_both_alpaca_feeds(self):
        assert list(config.LIVE_SOURCES) == ["finnhub", "alpaca:iex", "alpaca:sip"]

    def test_each_choice_names_a_source_and_a_feed(self):
        assert config.LIVE_SOURCES["alpaca:iex"] == ("alpaca", "iex")
        assert config.LIVE_SOURCES["alpaca:sip"] == ("alpaca", "sip")

    def test_finnhub_rides_on_iex_quotes(self):
        """Finnhub streams no Alpaca feed, but the quote poll and every agent's
        fill-price lookup still read one, so it is paired with the feed served
        on every Alpaca plan."""
        source, feed = config.LIVE_SOURCES["finnhub"]
        assert source == "finnhub"
        assert feed == "iex"

    def test_default_is_finnhub(self):
        assert config.DEFAULT_LIVE_SOURCE in config.LIVE_SOURCES
        assert config.LIVE_SOURCES[config.DEFAULT_LIVE_SOURCE][0] == config.DEFAULT_DATA_SOURCE

    def test_every_choice_is_labelled(self):
        assert set(config.LIVE_SOURCE_LABELS) == set(config.LIVE_SOURCES)
        assert config.LIVE_SOURCE_LABELS["alpaca:iex"] == "Alpaca (iex)"
        assert config.LIVE_SOURCE_LABELS["alpaca:sip"] == "Alpaca (sip)"


class TestLiveOptionWalls:
    CHAIN = {
        "strikes": [95.0, 100.0, 105.0],
        "calls_oi": [10, 50, 400],
        "puts_oi": [300, 40, 5],
        "calls_gamma_exposure": [1.0, 2.0, 1.0],
        "puts_gamma_exposure": [-1.0, -2.0, -1.0],
        "spot": 100.0,
    }

    def _sym_state(self, chain):
        from agent_stonks.state import SymbolState
        sym_state = SymbolState("AAPL", AppState())
        sym_state.options_chain = chain
        return sym_state

    def _stub_fetch(self, monkeypatch, calls, release=None):
        from agent_stonks import ui
        monkeypatch.setattr(ui, "_option_fetch_tried", {})
        monkeypatch.setattr(ui, "_option_fetch_running", set())

        def fetch(sym, spot=None):
            calls.append(sym)
            if release is not None:
                release.wait(2)
            return self.CHAIN

        monkeypatch.setattr(ui, "fetch_options_walls_data", fetch)

    def test_nothing_selected_fetches_nothing(self, monkeypatch):
        from agent_stonks.ui import _live_option_walls
        calls = []
        self._stub_fetch(monkeypatch, calls)
        assert _live_option_walls(self._sym_state(self.CHAIN), []) is None
        assert calls == []

    def test_selected_walls_from_the_latest_chain(self, monkeypatch):
        from agent_stonks.ui import _live_option_walls
        self._stub_fetch(monkeypatch, [])
        walls = _live_option_walls(self._sym_state(self.CHAIN), ["call_wall", "put_wall"])
        assert walls == {"call_wall": 105.0, "put_wall": 95.0}
        walls = _live_option_walls(self._sym_state(self.CHAIN), ["put_wall"])
        assert walls == {"put_wall": 95.0}

    def test_no_chain_yet_draws_nothing_and_fetches_once(self, monkeypatch):
        import threading
        import time as _time
        from agent_stonks.ui import _live_option_walls
        calls, release = [], threading.Event()
        self._stub_fetch(monkeypatch, calls, release)
        sym_state = self._sym_state(None)
        assert _live_option_walls(sym_state, ["call_wall"]) is None
        release.set()
        deadline = _time.monotonic() + 2
        while sym_state.options_chain is None and _time.monotonic() < deadline:
            _time.sleep(0.01)
        assert sym_state.options_chain == self.CHAIN
        # A second render inside the poll interval must not fetch again.
        assert _live_option_walls(sym_state, ["call_wall"]) == {"call_wall": 105.0}
        assert calls == ["AAPL"]


class TestReportSections:
    CHAIN = TestLiveOptionWalls.CHAIN

    def _app(self, chain=None):
        from agent_stonks.state import SymbolState
        app = AppState()
        sym_state = SymbolState("AAPL", app)
        sym_state.options_chain = chain
        sym_state.news = [
            {"id": "1", "created_at": "2024-01-15T13:00:00Z", "headline": "Apple headline",
             "summary": "", "source": "benzinga", "url": "https://example.com"}
        ]
        app.symbol_states["AAPL"] = sym_state
        return app

    def test_briefing_cards_carry_the_phase_and_failures(self):
        from datetime import datetime, timezone
        from agent_stonks.premarket import PremarketBriefing
        from agent_stonks.ui import _report_briefing
        app = self._app()
        app.premarket_phase = "open"
        app.premarket_generated_at = datetime(2024, 1, 15, 15, 5, tzinfo=timezone.utc)
        app.premarket_briefings = {
            "AAPL": PremarketBriefing(
                overall_bias="bullish", confidence="high", summary="Gap up on earnings.",
                catalysts=[], technical_levels=[], risk_factors=[],
                macro_context="Calm tape.", key_levels_to_watch=[],
            )
        }
        app.premarket_errors = {"TSLA": "timeout"}
        out = _report_briefing(app, ["AAPL", "TSLA"])
        assert out["briefing_title"] == "Intraday Situation Briefing"
        assert out["briefing_note"] == "Generated 2024-01-15 10:05 ET · failed for TSLA: timeout"
        assert len(out["briefing_cards"]) == 1
        assert "Gap up on earnings." in out["briefing_cards"][0]

    def test_news_cards_per_streamed_symbol(self):
        from agent_stonks.ui import _report_news_cards
        cards = _report_news_cards(self._app(), ["AAPL", "MSFT"])
        assert len(cards) == 1
        assert "Apple headline" in cards[0]

    def test_option_walls_from_the_stored_chain(self, monkeypatch):
        from agent_stonks import ui
        monkeypatch.setattr(
            ui, "fetch_options_walls_data",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("fetched")),
        )
        walls = ui._report_option_walls(self._app(self.CHAIN), ["AAPL"])
        assert [w["symbol"] for w in walls] == ["AAPL"]
        assert walls[0]["analysis"]["call_wall"] == 105.0
        assert walls[0]["fig"] is not None

    def test_option_walls_fetch_when_none_stored_and_skip_failures(self, monkeypatch):
        from agent_stonks import ui
        calls = []

        def fetch(sym, spot=None):
            calls.append(sym)
            if sym == "MSFT":
                raise RuntimeError("no chain")
            return self.CHAIN

        monkeypatch.setattr(ui, "fetch_options_walls_data", fetch)
        walls = ui._report_option_walls(self._app(None), ["AAPL", "MSFT"])
        assert calls == ["AAPL", "MSFT"]
        assert [w["symbol"] for w in walls] == ["AAPL"]


class TestAgentMomentum:
    """The look-back the momentum panels' agent lines are drawn over."""

    @staticmethod
    def state(personality="momentum", timeframe="1Min", running=False, form=None,
              levels=None):
        from types import SimpleNamespace

        return SimpleNamespace(
            llm_personality=personality, timeframe=timeframe, agent_running=running,
            apple_trader_config=form, apple_trader_levels=levels,
        )

    @staticmethod
    def sym(tactic_fields=(), alert_fields=(), status="armed"):
        from types import SimpleNamespace

        tactics = None
        if tactic_fields:
            tactics = SimpleNamespace(
                status=status,
                actions=[SimpleNamespace(conditions=[SimpleNamespace(field=f) for f in tactic_fields])],
            )
        return SimpleNamespace(tactics=tactics, alerts=[{"field": f} for f in alert_fields])

    @staticmethod
    def trader(take=15, fade=0, drop=0.0, fall=0.1, confirm=0):
        """A legacy config by default: the take / fall look-back."""
        from agent_stonks.apple_trader import AppleTraderConfig

        return AppleTraderConfig(
            model_key="dayrange", negative_momentum_bars=take, momentum_fade_bars=fade,
            momentum_drop=drop, max_fall_k=fall, momentum_confirmation_bars=confirm,
        )

    def test_apple_traders_confirmation_period(self):
        from agent_stonks.apple_trader import APPLE_TRADER_KEY

        form = self.trader(take=0, fall=0.0, confirm=7)
        assert _agent_momentum(self.state(APPLE_TRADER_KEY, form=form), self.sym()) == (
            7, "Apple Trader",
        )

    def test_no_momentum_falls_back_to_five_minutes(self):
        assert _agent_momentum(self.state(), self.sym()) == (5, "5 min")

    def test_five_minutes_is_one_bar_on_a_5min_chart(self):
        assert _agent_momentum(self.state(timeframe="5Min"), self.sym()) == (1, "5 min")

    def test_apple_traders_look_back(self):
        from agent_stonks.apple_trader import APPLE_TRADER_KEY

        bars, label = _agent_momentum(
            self.state(APPLE_TRADER_KEY, form=self.trader(take=22)), self.sym()
        )
        assert bars == 22 and label

    def test_apple_trader_with_only_the_fall_rule_reads_the_default_look_back(self):
        from agent_stonks.apple_trader import APPLE_TRADER_KEY

        bars, _ = _agent_momentum(
            self.state(APPLE_TRADER_KEY, form=self.trader(take=0, fall=0.1)), self.sym()
        )
        assert bars == config.APPLE_TRADER_NEGATIVE_MOMENTUM_BARS

    def test_apple_trader_reading_no_momentum_falls_back(self):
        from agent_stonks.apple_trader import APPLE_TRADER_KEY

        state = self.state(APPLE_TRADER_KEY, form=self.trader(take=0, fall=0.0))
        assert _agent_momentum(state, self.sym()) == (5, "5 min")

    def test_the_running_apple_traders_look_back_wins_over_the_form(self):
        from agent_stonks.apple_trader import APPLE_TRADER_KEY

        state = self.state(
            APPLE_TRADER_KEY, running=True, form=self.trader(take=22),
            levels={"config": self.trader(take=9)},
        )
        assert _agent_momentum(state, self.sym())[0] == 9

    def test_an_armed_momentum_tactic_uses_its_window(self):
        bars, label = _agent_momentum(self.state(), self.sym(tactic_fields=["momentum_pct"]))
        assert bars == config.TACTICS_MOMENTUM_WINDOW_MIN and label == "armed tactic"

    def test_the_tactic_window_is_converted_to_chart_bars(self):
        bars, _ = _agent_momentum(
            self.state(timeframe="5Min"), self.sym(tactic_fields=["momentum_pct"])
        )
        assert bars == 2

    def test_a_momentum_alert_counts_too(self):
        assert _agent_momentum(self.state(), self.sym(alert_fields=["momentum_pct"]))[0] == 10

    def test_a_spent_or_non_momentum_tactic_does_not(self):
        assert _agent_momentum(self.state(), self.sym(tactic_fields=["last_price"]))[0] == 5
        executed = self.sym(tactic_fields=["momentum_pct"], status="executed")
        assert _agent_momentum(self.state(), executed)[0] == 5


class TestVolumeBandSources:
    """`_volume_band_baseline`: one band per source on the chart, each from that
    source's own history."""

    TODAY = "2026-09-21"
    WEEK = ["2026-09-14", "2026-09-15", "2026-09-16", "2026-09-17", "2026-09-18"]

    def history(self, volume: float) -> list[dict]:
        return [
            {"t": f"{day}T16:00:00Z", "o": 1, "h": 1, "l": 1, "c": 1, "v": volume}
            for day in self.WEEK
        ]

    def setup(self, monkeypatch, sip=True):
        from agent_stonks import ui

        fetched = []
        monkeypatch.setattr(ui, "_VOLUME_BAND_CACHE", {})
        monkeypatch.setattr(
            ui, "fetch_intraday_history_bars",
            lambda symbol, days: fetched.append("yfinance") or self.history(900.0),
        )

        def week(symbol, feed, key, secret, days):
            fetched.append(feed)
            if feed == "sip" and not sip:
                return []
            return self.history({"sip": 1000.0, "iex": 40.0}[feed])

        monkeypatch.setattr(ui.bar_history, "fetch_week_minute_bars", week)
        state = AppState()
        state.timeframe = "1Min"
        return ui, state, fetched

    def bars(self, *sources):
        return [
            {"t": f"{self.TODAY}T16:0{i}:00Z", "o": 1, "h": 1, "l": 1, "c": 1, "v": 1, "src": s}
            for i, s in enumerate(sources)
        ]

    def test_each_source_is_read_against_its_own_history(self, monkeypatch):
        ui, state, fetched = self.setup(monkeypatch)
        base = ui._volume_band_baseline("AAPL", self.bars("yfinance", "iex", "sip"), state, "week_band")
        levels = {src: band["per_minute"][12 * 60] for src, band in base["band_by_source"].items()}
        assert levels == {"yfinance": 900.0, "iex": 40.0, "sip": 1000.0}
        assert sorted(fetched) == ["iex", "sip", "yfinance"]

    def test_finnhub_bars_read_sip_or_else_yfinance(self, monkeypatch):
        ui, state, _ = self.setup(monkeypatch)
        base = ui._volume_band_baseline("AAPL", self.bars("finnhub"), state, "week_band")
        assert base["band_by_source"]["finnhub"]["history"] == "sip"

        ui, state, _ = self.setup(monkeypatch, sip=False)
        base = ui._volume_band_baseline("AAPL", self.bars("finnhub"), state, "week_band")
        assert base["band_by_source"]["finnhub"]["history"] == "yfinance"

    def test_untagged_bars_fetch_nothing(self, monkeypatch):
        ui, state, fetched = self.setup(monkeypatch)
        bars = [{k: v for k, v in b.items() if k != "src"} for b in self.bars("sip")]
        assert ui._volume_band_baseline("AAPL", bars, state, "week_band") is None
        assert not fetched

    def test_the_daily_bars_timeframe_has_no_band(self, monkeypatch):
        ui, state, _ = self.setup(monkeypatch)
        state.timeframe = "1Day"
        assert ui._volume_band_baseline("AAPL", self.bars("sip"), state, "week_band") is None


class TestQuoteHtml:
    def _html(self, **kw):
        return _quote_html(
            100.0, 99.0, 99.9, 100, 100.1, 200, "AAPL",
            today_low=98.0, today_high=101.0, **kw,
        )

    def test_day_cards_are_clickable_toggles(self):
        html = self._html()
        assert 'data-toggle="day_low"' in html
        assert 'data-toggle="day_high"' in html
        assert "Prev Min" not in html

    def test_shown_line_card_is_highlighted(self):
        html = self._html(day_lines={"day_high"})
        high = html.split('data-toggle="day_high"')[1].split(">")[0]
        low = html.split('data-toggle="day_low"')[1].split(">")[0]
        assert config.PALETTE["accent"] in high and "Hide" in high
        assert config.PALETTE["accent"] not in low and "Show" in low


class TestChartStart:
    """The live chart starts five minutes before the open of its bars' own
    ET day, or at that day's first bar with pre-market on."""

    @staticmethod
    def _et(stamp):
        import pandas as pd

        return pd.Timestamp(stamp).tz_convert("America/New_York")

    def test_five_minutes_before_the_open_in_summer_and_winter(self):
        for last_bar in ("2026-09-29T15:00:00+00:00", "2026-12-15T16:00:00+00:00"):
            start = self._et(_chart_start([{"t": last_bar}], pre_market=False))
            assert (start.hour, start.minute) == (9, 25)
            assert start.date() == self._et(last_bar).date()

    def test_pre_market_starts_at_the_days_first_bar(self):
        start = self._et(_chart_start([{"t": "2026-09-29T15:00:00+00:00"}], pre_market=True))
        assert (start.hour, start.minute) == (0, 0)
        assert str(start.date()) == "2026-09-29"

    def test_an_evening_bar_is_still_its_own_day(self):
        # 23:30 ET on the 29th is 03:30 UTC on the 30th.
        start = self._et(_chart_start([{"t": "2026-09-30T03:30:00+00:00"}], pre_market=False))
        assert str(start.date()) == "2026-09-29"


class TestOrchestraIsItsOwnAgent:
    """Orchestra is a personality of its own, not a mode of Apple Trader."""

    def test_listed_as_a_rule_agent_with_its_own_label_and_face(self):
        from agent_stonks import ui
        from agent_stonks.apple_trader import APPLE_TRADER_KEY
        from agent_stonks.orchestra import ORCHESTRA_KEY, ORCHESTRA_LABEL

        assert ORCHESTRA_KEY in ui.RULE_AGENT_KEYS
        assert ui._personality_label(ORCHESTRA_KEY) == ORCHESTRA_LABEL == "Orchestra (rule-based, no LLM)"
        assert ui._avatar_data_uri(ORCHESTRA_KEY)
        assert ui._avatar_data_uri(ORCHESTRA_KEY) != ui._avatar_data_uri(APPLE_TRADER_KEY)

    def test_its_settings_are_its_own_and_remembered(self):
        from agent_stonks import last_setup, ui

        assert ui._ORCHESTRA_COPY.prefix == "orchestra"
        assert ui._APPLE_TRADER_COPY.prefix == "apple_trader"
        assert last_setup.is_kept("orchestra_buy_k_highlow_AAPL")
        assert last_setup.is_kept("orchestra_pairs")
        # A button's value cannot be restored into a session, so it is not kept.
        assert not last_setup.is_kept("add_symbols_for_orchestra")

    def test_the_momentum_window_is_the_pairs_on_this_symbol(self):
        from types import SimpleNamespace

        from agent_stonks.apple_trader import AppleTraderConfig
        from agent_stonks.orchestra import ORCHESTRA_KEY

        state = SimpleNamespace(
            llm_personality=ORCHESTRA_KEY, timeframe="1Min", agent_running=False,
            apple_trader_config=None, apple_trader_levels=None, orchestra=None,
            orchestra_levels={},
            orchestra_configs={
                "INTC:dayrange": AppleTraderConfig(ticker="INTC", momentum_confirmation_bars=7),
            },
        )
        sym = SimpleNamespace(symbol="INTC", tactics=None, alerts=[])
        assert _agent_momentum(state, sym) == (7, "Orchestra")
