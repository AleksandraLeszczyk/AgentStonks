from marketview.ui import build_news_html, wrap_text


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
