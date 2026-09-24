from agent_stonks import config
from agent_stonks.state import AppState
from agent_stonks.ui import (
    _agent_momentum,
    _cash_label,
    _venue_badge,
    _portfolio_value_label,
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


class TestAgentMomentum:
    """The look-back the momentum panels' agent lines are drawn over."""

    @staticmethod
    def state(personality="momentum", timeframe="1Min", running=False, form=None,
              levels=None, rules2=None):
        from types import SimpleNamespace

        return SimpleNamespace(
            llm_personality=personality, timeframe=timeframe, agent_running=running,
            apple_trader_config=form, apple_trader_levels=levels,
            apple_trader2_config=(
                SimpleNamespace(rules=SimpleNamespace(reads_momentum=lambda: rules2))
                if rules2 is not None else None
            ),
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
    def trader(take=15, fade=0, drop=0.0, fall=0.1):
        from agent_stonks.apple_trader import AppleTraderConfig

        return AppleTraderConfig(
            model_key="dayrange", negative_momentum_bars=take, momentum_fade_bars=fade,
            momentum_drop=drop, max_fall_k=fall,
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

    def test_apple_trader_2_reading_momentum_uses_the_regime_horizon(self):
        from agent_stonks.apple_trader2 import APPLE_TRADER2_KEY
        from agent_stonks.momentum_regime import MOMENTUM_DEFAULTS

        bars, _ = _agent_momentum(self.state(APPLE_TRADER2_KEY, rules2=True), self.sym())
        assert bars == MOMENTUM_DEFAULTS["horizon"]

    def test_apple_trader_2_without_momentum_rules_falls_back(self):
        from agent_stonks.apple_trader2 import APPLE_TRADER2_KEY

        state = self.state(APPLE_TRADER2_KEY, rules2=False)
        assert _agent_momentum(state, self.sym()) == (5, "5 min")

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
