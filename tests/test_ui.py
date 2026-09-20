from agent_stonks import config
from agent_stonks.state import AppState
from agent_stonks.ui import (
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
