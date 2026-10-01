"""The live session's two news feeds -- Alpaca and Yahoo Finance -- and how
they are combined into one list (`agent_stonks.news`)."""
import yfinance

from agent_stonks import news


def _yahoo_item(uid, title, pub, provider="Reuters", **content):
    return {
        "id": uid,
        "content": {
            "id": uid,
            "contentType": "STORY",
            "title": title,
            "summary": f"{title} summary",
            "pubDate": pub,
            "provider": {"displayName": provider},
            "canonicalUrl": {"url": f"https://example.com/{uid}"},
            **content,
        },
    }


class _FakeTicker:
    items: list = []
    calls: list = []

    def __init__(self, symbol):
        self.symbol = symbol

    def get_news(self, count=10, tab="news"):
        _FakeTicker.calls.append((self.symbol, count, tab))
        return _FakeTicker.items


class TestFetchYfinanceNews:
    def test_maps_yahoo_items_to_the_alpaca_article_shape(self, monkeypatch):
        _FakeTicker.items = [_yahoo_item("u1", "Apple rallies", "2026-10-01T14:23:58Z")]
        _FakeTicker.calls = []
        monkeypatch.setattr(yfinance, "Ticker", _FakeTicker)

        articles = news.fetch_yfinance_news("AAPL", count=5)

        assert articles == [{
            "id": "yf-u1",
            "headline": "Apple rallies",
            "summary": "Apple rallies summary",
            "created_at": "2026-10-01T14:23:58Z",
            "url": "https://example.com/u1",
            "source": "Reuters",
            "feed": "yfinance",
        }]
        assert _FakeTicker.calls == [("AAPL", 5, "news")]

    def test_skips_items_without_a_title_or_time_and_falls_back_on_url(self, monkeypatch):
        no_title = _yahoo_item("u1", "", "2026-10-01T14:00:00Z")
        no_time = _yahoo_item("u2", "Headline", None)
        click_through = _yahoo_item(
            "u3", "Kept", "2026-10-01T14:00:00Z",
            canonicalUrl=None, clickThroughUrl={"url": "https://click/u3"},
        )
        _FakeTicker.items = [no_title, no_time, click_through]
        monkeypatch.setattr(yfinance, "Ticker", _FakeTicker)

        articles = news.fetch_yfinance_news("AAPL")

        assert [a["id"] for a in articles] == ["yf-u3"]
        assert articles[0]["url"] == "https://click/u3"


class TestMergeNews:
    def test_adds_unseen_articles_newest_first(self):
        existing = [{"id": 1, "headline": "Older", "created_at": "2026-10-01T10:00:00Z"}]
        fresh = [
            {"id": "yf-a", "headline": "Newest", "created_at": "2026-10-01T12:00:00Z"},
            {"id": "yf-b", "headline": "Oldest", "created_at": "2026-10-01T09:00:00Z"},
        ]

        merged, added = news.merge_news(existing, fresh)

        assert [a["headline"] for a in merged] == ["Newest", "Older", "Oldest"]
        assert [a["id"] for a in added] == ["yf-a", "yf-b"]
        assert existing == [{"id": 1, "headline": "Older", "created_at": "2026-10-01T10:00:00Z"}]

    def test_the_same_story_from_the_other_feed_is_a_repeat(self):
        """Benzinga's pieces reach Yahoo under Yahoo's own id; Alpaca's
        headlines carry HTML entities Yahoo's do not."""
        alpaca = [{
            "id": 62099317,
            "headline": "Inquiry Into Apple's Competitor Dynamics In Hardware, Storage &amp; Peripherals",
            "created_at": "2026-10-01T09:59:13Z",
        }]
        yahoo = [{
            "id": "yf-x",
            "headline": "Inquiry into Apple’s competitor dynamics in hardware, storage & peripherals",
            "created_at": "2026-10-01T09:59:13Z",
        }]
        # The typographic apostrophe is a different character: still a repeat,
        # because only letters and digits are compared.
        merged, added = news.merge_news(alpaca, yahoo)

        assert added == []
        assert [a["id"] for a in merged] == [62099317]

    def test_a_repeated_id_is_dropped_even_with_a_new_headline(self):
        existing = [{"id": "7", "headline": "Old wording", "created_at": "2026-10-01T10:00:00Z"}]
        merged, added = news.merge_news(existing, [{"id": 7, "headline": "Updated wording"}])

        assert added == []
        assert merged == existing

    def test_repeats_within_one_batch_are_dropped(self):
        fresh = [
            {"id": "a", "headline": "Same story", "created_at": "2026-10-01T10:00:00Z"},
            {"id": "b", "headline": "Same Story!", "created_at": "2026-10-01T10:01:00Z"},
        ]
        _, added = news.merge_news([], fresh)

        assert [a["id"] for a in added] == ["a"]

    def test_sorts_even_when_nothing_is_added(self):
        """Alpaca's REST answer is not in publication order either."""
        existing = [
            {"id": 1, "headline": "a", "created_at": "2026-09-30T09:43:20Z"},
            {"id": 2, "headline": "b", "created_at": "2026-09-30T11:31:56Z"},
        ]
        merged, _ = news.merge_news(existing, [])

        assert [a["id"] for a in merged] == [2, 1]

    def test_articles_without_a_time_sort_last_in_their_order(self):
        existing = [{"id": "x", "headline": "no time"}, {"id": "y", "headline": "also none"}]
        merged, _ = news.merge_news(
            existing, [{"id": "z", "headline": "timed", "created_at": "2026-10-01T10:00:00Z"}]
        )

        assert [a["id"] for a in merged] == ["z", "x", "y"]

    def test_mixed_time_formats_compare_as_times(self):
        merged, _ = news.merge_news(
            [{"id": 1, "headline": "a", "created_at": "2026-10-01T10:00:00.500Z"}],
            [{"id": 2, "headline": "b", "created_at": "2026-10-01 10:00:01"}],  # WorldNews, naive UTC
        )

        assert [a["id"] for a in merged] == [2, 1]


class TestFetchLiveNews:
    def test_merges_alpaca_and_yahoo(self, monkeypatch):
        monkeypatch.setattr(news, "fetch_news_with_fallback", lambda *a, **k: [
            {"id": 1, "headline": "Shared story", "created_at": "2026-10-01T10:00:00Z"},
        ])
        monkeypatch.setattr(news, "fetch_yfinance_news", lambda symbol: [
            {"id": "yf-1", "headline": "Shared story", "created_at": "2026-10-01T10:00:00Z"},
            {"id": "yf-2", "headline": "Yahoo only", "created_at": "2026-10-01T11:00:00Z"},
        ])

        articles = news.fetch_live_news("AAPL", "k", "s", "")

        assert [a["id"] for a in articles] == ["yf-2", 1]

    def test_a_yahoo_failure_keeps_alpacas_articles(self, monkeypatch):
        alpaca = [{"id": 1, "headline": "Alpaca story", "created_at": "2026-10-01T10:00:00Z"}]
        monkeypatch.setattr(news, "fetch_news_with_fallback", lambda *a, **k: alpaca)

        def _down(symbol):
            raise RuntimeError("Too Many Requests")

        monkeypatch.setattr(news, "fetch_yfinance_news", _down)

        assert news.fetch_live_news("AAPL", "k", "s", "") == alpaca
