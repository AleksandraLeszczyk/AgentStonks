"""Tests for the SimLab simulation suite (clock, store, market, engine, scores)."""
import json
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from agent_stonks import apple_models, clock, persistence_model
from agent_stonks.agent import MOMENTUM_SYSTEM_PROMPT
from agent_stonks.apple_trader import (
    APPLE_TRADER_KEY,
    RULE_PROVIDER,
    AppleTraderConfig,
    config_signature,
)
from agent_stonks.apple_trader2 import APPLE_TRADER2_KEY, AppleTrader2Config
from simlab import app as sim_app
from simlab import data as sim_data
from simlab import prompts as sim_prompts
from simlab import results as sim_results
from simlab.engine import SimulationConfig, SimulationEngine
from simlab.judge import _entry_context, _first_exit_after
from simlab.market import SimMarket, parse_ts
from simlab.patches import simulation_context
from simlab.rule_agents import RULE_AGENTS, rule_agent

DAY = date(2026, 6, 15)  # a Monday
OPEN_UTC = datetime(2026, 6, 15, 13, 30, tzinfo=timezone.utc)  # 09:30 EDT


def _bar(ts: datetime, close: float, volume: float = 1000.0) -> dict:
    return {
        "t": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "o": close - 0.05,
        "h": close + 0.1,
        "l": close - 0.1,
        "c": close,
        "v": volume,
    }


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """Point the SimLab store at a temp dir and populate one synthetic day:
    60 minute bars ramping 100.0 -> 105.9, plus 30 prior daily bars."""
    monkeypatch.setattr(sim_data, "STORE_DIR", tmp_path / "store")
    monkeypatch.setattr(sim_data, "MANIFEST_PATH", tmp_path / "datasets.json")
    minute_bars = [
        _bar(OPEN_UTC + timedelta(minutes=i), 100.0 + 0.1 * i) for i in range(60)
    ]
    sim_data._write_gz(sim_data.bars_path("TEST", DAY), minute_bars)
    daily = [
        _bar(datetime(2026, 6, 15, tzinfo=timezone.utc) - timedelta(days=i), 99.0)
        for i in range(30, 0, -1)
    ]
    sim_data._write_gz(sim_data.daily_path("TEST"), {
        "symbol": "TEST", "start": "2026-05-16", "end": "2026-06-15", "bars": daily,
    })
    return tmp_path


class TestClock:
    def test_pin_and_clear(self):
        pinned = datetime(2026, 6, 15, 14, 0, tzinfo=timezone.utc)
        try:
            clock.set_simulated(pinned)
            assert clock.now() == pinned
            assert clock.monotonic() == pinned.timestamp()
            assert clock.is_simulated()
        finally:
            clock.clear()
        assert not clock.is_simulated()
        assert abs((clock.now() - datetime.now(timezone.utc)).total_seconds()) < 5

    def test_rejects_naive_datetime(self):
        with pytest.raises(ValueError):
            clock.set_simulated(datetime(2026, 6, 15, 14, 0))


class TestMarket:
    def test_a_symbol_the_dataset_does_not_carry_reads_as_empty(self, store):
        """Not `KeyError` -- the contract `simlab.patches` claims to honour.

        Every live fetch these stand in for returns `[]`/`None` when a symbol
        cannot be had, so a model reading a *second* symbol for context (a
        market index for a relative-strength or beta feature, say) must see the
        same thing in a replay as it would live, and refuse in its own words.

        Before this, such a read raised a bare ``'SPY'`` out of the middle of
        the day loop and ended the whole simulation with "partial results
        below" instead of standing the session down. A symbol the run actually
        trades is caught much earlier, when the experiment is queued
        (`app.missing_rule_symbols`), so nothing real is swallowed here.
        """
        market = SimMarket(["TEST"], [DAY])
        t = OPEN_UTC + timedelta(minutes=10)
        assert market.completed_bars("SPY", t) == []
        assert market.daily_bars_at("SPY", t) == []
        assert market.completed_daily_bars("SPY", t) == []
        assert market.bars_window("SPY", t, t + timedelta(minutes=5)) == []
        assert market.news_at("SPY", t) == []
        assert market.fresh_news("SPY", t - timedelta(minutes=5), t) == []
        assert market.price_at("SPY", t) is None
        assert market.session_open_price("SPY", t) is None
        assert market.prev_close("SPY", t) is None
        # ...and the symbol it does carry is unaffected.
        assert market.completed_bars("TEST", t)

    def test_completed_bars_respect_bar_completion(self, store):
        market = SimMarket(["TEST"], [DAY])
        # At 13:31:00 exactly bar 0 (13:30) has completed; bar 1 has not.
        t = OPEN_UTC + timedelta(minutes=1)
        bars = market.completed_bars("TEST", t)
        assert len(bars) == 1
        assert market.price_at("TEST", t) == 100.0

    def test_daily_bars_include_partial_today(self, store):
        market = SimMarket(["TEST"], [DAY])
        t = OPEN_UTC + timedelta(minutes=10)
        daily = market.daily_bars_at("TEST", t)
        assert daily[-1]["t"].startswith("2026-06-15")
        assert daily[-1]["c"] == pytest.approx(100.9)  # close of bar 9
        assert market.prev_close("TEST", t) == 99.0

    def test_step_times_cover_every_bar(self, store):
        market = SimMarket(["TEST"], [DAY])
        steps = market.step_times(DAY)
        assert len(steps) == 60
        assert steps[0] == OPEN_UTC + timedelta(minutes=1)

    def test_completed_daily_bars_stop_before_today(self, store):
        """The stricter daily view, for callers that must never be able to
        reach the simulated day's outcome. `daily_bars_at` appends a partial
        row for today; this one appends nothing."""
        market = SimMarket(["TEST"], [DAY])
        t = OPEN_UTC + timedelta(minutes=10)
        bars = market.completed_daily_bars("TEST", t)
        assert all(str(b["t"])[:10] < "2026-06-15" for b in bars)
        assert len(bars) == len(market.daily_bars_at("TEST", t)) - 1

    def test_the_session_open_is_available_and_the_rest_of_that_row_is_not(self, store):
        """The 9:30 auction print is fixed at the open, so reading it during
        the simulated day is point-in-time honest -- unlike the high, low and
        close of the same stored row, which are the day's outcome."""
        stored = sim_data._read_gz(sim_data.daily_path("TEST"))
        stored["bars"].append(_bar(datetime(2026, 6, 15, tzinfo=timezone.utc), 123.0))
        sim_data._write_gz(sim_data.daily_path("TEST"), stored)
        market = SimMarket(["TEST"], [DAY])

        opening = market.session_open_price("TEST", OPEN_UTC + timedelta(minutes=10))
        assert opening == pytest.approx(122.95)  # `_bar` sets o = close - 0.05
        assert all(
            str(b["t"])[:10] != "2026-06-15"
            for b in market.completed_daily_bars("TEST", OPEN_UTC + timedelta(minutes=10))
        )

    def test_no_session_open_before_the_bell(self, store):
        """Otherwise a pre-market cycle could read the auction that has not
        happened yet."""
        stored = sim_data._read_gz(sim_data.daily_path("TEST"))
        stored["bars"].append(_bar(datetime(2026, 6, 15, tzinfo=timezone.utc), 123.0))
        sim_data._write_gz(sim_data.daily_path("TEST"), stored)
        market = SimMarket(["TEST"], [DAY])
        assert market.session_open_price("TEST", OPEN_UTC - timedelta(minutes=5)) is None


class TestDatasetStore:
    def test_create_dataset_downloads_only_missing_days(self, store, monkeypatch):
        calls = []

        def fake_minute(symbol, day, key, secret, feed="iex"):
            calls.append(day)
            return [_bar(OPEN_UTC, 100.0)] if day.weekday() < 5 else []

        monkeypatch.setattr(sim_data, "fetch_minute_bars_day", fake_minute)
        monkeypatch.setattr(sim_data, "fetch_news_day", lambda *a, **k: [])
        monkeypatch.setattr(sim_data, "fetch_daily_bars_range", lambda *a, **k: [_bar(OPEN_UTC, 99.0)])
        monkeypatch.setattr(sim_data, "fetch_market_indicator_closes", lambda *a, **k: {"spy": [], "vix": [], "vix3m": []})

        ds = sim_data.create_dataset("wk", ["TEST"], date(2026, 6, 15), date(2026, 6, 17), "k", "s")
        # 2026-06-15 already in the store (fixture) -- only 16th and 17th fetched.
        assert calls == [date(2026, 6, 16), date(2026, 6, 17)]
        assert ds.days == ["2026-06-15", "2026-06-16", "2026-06-17"]
        assert sim_data.get_dataset("wk").symbols == ["TEST"]
        sim_data.delete_dataset("wk")
        assert sim_data.get_dataset("wk") is None


class TestFeedIsPartOfTheData:
    """`iex` and `sip` are different bars for the same day, not two routes to
    one answer -- one venue against the consolidated tape. Keying the store on
    (symbol, day) alone made a `sip` request silently reuse `iex` bars, which
    is how a run reproduces a study's dataset but not its trades."""

    def _fetchers(self, monkeypatch, calls):
        def fake_minute(symbol, day, key, secret, feed="iex"):
            calls.append((day, feed))
            # A thin feed is a different tape, not a scaled one: different
            # closes, and here one bar fewer.
            n = 3 if feed == "iex" else 5
            open_utc = OPEN_UTC + timedelta(days=(day - DAY).days)
            return [_bar(open_utc + timedelta(minutes=i),
                         100.0 + i + (0.5 if feed == "iex" else 0.0)) for i in range(n)]

        monkeypatch.setattr(sim_data, "fetch_minute_bars_day", fake_minute)
        monkeypatch.setattr(sim_data, "fetch_news_day", lambda *a, **k: [])
        monkeypatch.setattr(
            sim_data, "fetch_daily_bars_range", lambda *a, **k: [_bar(OPEN_UTC, 99.0)]
        )
        monkeypatch.setattr(
            sim_data, "fetch_market_indicator_closes",
            lambda *a, **k: {"spy": [], "vix": [], "vix3m": []},
        )

    def test_the_two_feeds_are_stored_side_by_side(self, store, monkeypatch):
        calls = []
        self._fetchers(monkeypatch, calls)
        day = date(2026, 6, 16)

        sim_data.create_dataset("d-iex", ["TEST"], day, day, "k", "s", feed="iex")
        sim_data.create_dataset("d-sip", ["TEST"], day, day, "k", "s", feed="sip")

        # The crux: the second request downloads rather than reusing the first.
        assert calls == [(day, "iex"), (day, "sip")]
        assert sim_data.bars_path("TEST", day, "iex") != sim_data.bars_path("TEST", day, "sip")
        assert len(sim_data.load_day_bars("TEST", day, "iex")) == 3
        assert len(sim_data.load_day_bars("TEST", day, "sip")) == 5

    def test_coverage_answers_per_feed(self, store, monkeypatch):
        calls = []
        self._fetchers(monkeypatch, calls)
        day = date(2026, 6, 16)
        sim_data.create_dataset("d-iex", ["TEST"], day, day, "k", "s", feed="iex")

        assert sim_data.coverage(["TEST"], day, day, "iex")[day.isoformat()]["TEST"]
        assert not sim_data.coverage(["TEST"], day, day, "sip")[day.isoformat()]["TEST"]

    def test_the_dataset_records_the_tape_it_holds(self, store, monkeypatch):
        self._fetchers(monkeypatch, [])
        day = date(2026, 6, 16)
        ds = sim_data.create_dataset("d-sip", ["TEST"], day, day, "k", "s", feed="sip")
        assert ds.feed == "sip"
        assert sim_data.get_dataset("d-sip").feed == "sip"

    def test_an_unknown_feed_is_refused_before_anything_downloads(self, store, monkeypatch):
        calls = []
        self._fetchers(monkeypatch, calls)
        with pytest.raises(ValueError, match="unknown feed"):
            sim_data.create_dataset(
                "d", ["TEST"], DAY, DAY, "k", "s", feed="consolidated"
            )
        assert calls == []

    def test_days_stored_before_feeds_were_tracked_still_load_as_iex(self, store):
        """The fixture writes through the feed-scoped path, so this writes the
        pre-feed layout by hand -- that is what every day downloaded before
        this change looks like, and it must not force a re-download."""
        day = date(2026, 6, 18)
        sim_data._write_gz(
            sim_data._legacy_bars_path("TEST", day), [_bar(OPEN_UTC, 101.0)]
        )
        assert len(sim_data.load_day_bars("TEST", day, "iex")) == 1
        assert sim_data.coverage(["TEST"], day, day, "iex")[day.isoformat()]["TEST"]
        # ...and is never mistaken for the consolidated tape it is not.
        assert sim_data.load_day_bars("TEST", day, "sip") == []
        assert not sim_data.coverage(["TEST"], day, day, "sip")[day.isoformat()]["TEST"]

    def test_a_manifest_entry_without_a_feed_reads_as_iex(self, store):
        sim_data.MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        sim_data.MANIFEST_PATH.write_text(json.dumps([{
            "name": "old", "symbols": ["TEST"], "start": "2026-06-15",
            "end": "2026-06-15", "created_at": "", "days": ["2026-06-15"],
        }]))
        assert sim_data.get_dataset("old").feed == "iex"

    def test_the_simulation_reads_the_tape_its_dataset_names(self, store, monkeypatch):
        """The whole point of threading the feed through: a run on `sip` must
        see `sip` bars, not whichever tape happened to be downloaded first."""
        self._fetchers(monkeypatch, [])
        day = date(2026, 6, 16)
        for feed in ("iex", "sip"):
            sim_data.create_dataset(f"d-{feed}", ["TEST"], day, day, "k", "s", feed=feed)

        t = OPEN_UTC + timedelta(days=1, minutes=1)
        assert SimMarket(["TEST"], [day], "iex").price_at("TEST", t) == 100.5
        assert SimMarket(["TEST"], [day], "sip").price_at("TEST", t) == 100.0
        assert SimMarket(["TEST"], [day], "sip").feed == "sip"

    def test_the_run_record_states_which_tape_produced_it(self, store, monkeypatch):
        self._fetchers(monkeypatch, [])
        day = date(2026, 6, 16)
        sim_data.create_dataset("d-sip", ["TEST"], day, day, "k", "s", feed="sip")
        market = SimMarket(["TEST"], [day], "sip")
        config = SimulationConfig(
            personality=APPLE_TRADER_KEY, provider=RULE_PROVIDER, model="rules",
            api_key="", symbols=["TEST"], days=[day], feed="sip",
        )
        result = SimulationEngine(market, config).run()
        assert result.config_summary["feed"] == "sip"


class TestMinuteBarsComeFromYfinance:
    """New datasets are consolidated-tape yfinance bars, not Alpaca IEX.

    IEX is one venue (~4% of the tape); yfinance is the consolidated tape and
    the same source the live volume tools read, so a simulated volume ratio is
    like-for-like with the live one. The Alpaca feeds stay reachable for the
    datasets already downloaded on them.
    """

    def _fake_yfinance(self, monkeypatch, frame, captured=None):
        """Install a stand-in yfinance module for `_yf_frame`'s local import."""
        import sys

        def fake_download(symbol, **kwargs):
            if captured is not None:
                captured.append({"symbol": symbol, **kwargs})
            return frame

        monkeypatch.setitem(
            sys.modules, "yfinance", SimpleNamespace(download=fake_download)
        )

    def _minute_frame(self, rows):
        """rows: [(ET "HH:MM", close, volume)] -> a yfinance-shaped frame."""
        import pandas as pd

        index = pd.to_datetime(
            [f"2026-06-15 {hhmm}" for hhmm, _, _ in rows]
        ).tz_localize("America/New_York").tz_convert("UTC")
        return pd.DataFrame(
            {
                "Open": [c for _, c, _ in rows],
                "High": [c for _, c, _ in rows],
                "Low": [c for _, c, _ in rows],
                "Close": [c for _, c, _ in rows],
                "Volume": [v for _, _, v in rows],
            },
            index=index,
        )

    def test_the_default_feed_is_yfinance(self):
        assert sim_data.DEFAULT_FEED == "yfinance"
        assert "yfinance" in sim_data.FEEDS
        # ...and the pre-feed store layout is still read as the IEX it is.
        assert sim_data.LEGACY_FEED == "iex"

    def test_a_minute_day_is_fetched_from_yahoo_with_extended_hours(self, monkeypatch):
        captured = []
        self._fake_yfinance(
            monkeypatch, self._minute_frame([("09:30", 100.0, 5000.0)]), captured
        )

        bars = sim_data.fetch_minute_bars_day("TEST", DAY)

        assert len(captured) == 1
        call = captured[0]
        assert call["symbol"] == "TEST"
        assert call["interval"] == "1m"
        # prepost is the difference between the stored 04:00-20:00 window and
        # the regular session alone.
        assert call["prepost"] is True
        assert bars == [{
            "t": "2026-06-15T13:30:00Z", "o": 100.0, "h": 100.0,
            "l": 100.0, "c": 100.0, "v": 5000.0,
        }]

    def test_no_alpaca_credentials_are_needed(self, monkeypatch):
        """The signature keeps key/secret for the Alpaca feeds, but the
        default path must not require them -- the datasets tab lets them be
        blank now."""
        self._fake_yfinance(monkeypatch, self._minute_frame([("09:30", 100.0, 1.0)]))

        def no_alpaca(*a, **k):
            raise AssertionError("the yfinance feed must not call Alpaca")

        monkeypatch.setattr(sim_data, "_paged_get", no_alpaca)
        assert len(sim_data.fetch_minute_bars_day("TEST", DAY)) == 1

    def test_extended_hours_bars_are_kept_despite_zero_volume(self, monkeypatch):
        """Yahoo serves real, moving prices before 09:30 but reports every one
        of those minutes at volume 0. Dropping them on that basis would cut the
        stored day down to the regular session and blind every pre-market read."""
        self._fake_yfinance(monkeypatch, self._minute_frame([
            ("04:00", 99.0, 0.0),
            ("04:01", 99.4, 0.0),
            ("09:30", 100.0, 5000.0),
            ("18:00", 101.0, 0.0),
        ]))

        bars = sim_data.fetch_minute_bars_day("TEST", DAY)

        assert [b["c"] for b in bars] == [99.0, 99.4, 100.0, 101.0]
        assert [b["v"] for b in bars] == [0.0, 0.0, 5000.0, 0.0]

    def test_a_minute_with_no_price_at_all_is_dropped(self, monkeypatch):
        frame = self._minute_frame([("09:30", 100.0, 10.0), ("09:31", 101.0, 10.0)])
        frame.iloc[1, :] = float("nan")
        self._fake_yfinance(monkeypatch, frame)

        assert [b["t"] for b in sim_data.fetch_minute_bars_day("TEST", DAY)] == [
            "2026-06-15T13:30:00Z"
        ]

    def test_bars_carry_no_vwap_field(self, monkeypatch):
        """Yahoo publishes no per-bar VWAP. Downstream (`analyze_intraday`,
        the chart's VWAP line) branches on the key's presence, so omitting it
        loses the VWAP note; inventing one would make it wrong instead."""
        self._fake_yfinance(monkeypatch, self._minute_frame([("09:30", 100.0, 10.0)]))
        assert "vw" not in sim_data.fetch_minute_bars_day("TEST", DAY)[0]

    def test_a_day_outside_yahoos_window_is_empty_not_an_error(self, monkeypatch):
        """Yahoo refuses 1-minute data older than 30 days; the fetch returns
        the same "nothing for this day" a holiday gives, which create_dataset
        already handles."""
        self._fake_yfinance(monkeypatch, None)
        assert sim_data.fetch_minute_bars_day("TEST", date(2020, 1, 6)) == []

    def test_a_new_dataset_is_stored_under_the_yfinance_feed(self, store, monkeypatch):
        calls = []

        def fake_minute(symbol, day, key="", secret="", feed=sim_data.DEFAULT_FEED):
            calls.append((day, feed))
            return [_bar(OPEN_UTC + timedelta(days=(day - DAY).days), 100.0)]

        monkeypatch.setattr(sim_data, "fetch_minute_bars_day", fake_minute)
        monkeypatch.setattr(sim_data, "fetch_daily_bars_range", lambda *a, **k: [])
        monkeypatch.setattr(
            sim_data, "fetch_market_indicator_closes",
            lambda *a, **k: {"spy": [], "vix": [], "vix3m": []},
        )
        day = date(2026, 6, 16)

        ds = sim_data.create_dataset("yf", ["TEST"], day, day)

        assert ds.feed == "yfinance"
        assert calls == [(day, "yfinance")]
        assert sim_data.bars_path("TEST", day, "yfinance").exists()
        # The same day on the Alpaca tape is a different file and still absent.
        assert not sim_data.bars_path("TEST", day, "iex").exists()

    def test_the_thirty_day_horizon_is_announced_before_downloading(self, store, monkeypatch):
        """A day Yahoo won't serve stores empty, which on disk is
        indistinguishable from a holiday -- so say it up front."""
        monkeypatch.setattr(sim_data, "fetch_minute_bars_day", lambda *a, **k: [])
        monkeypatch.setattr(sim_data, "fetch_daily_bars_range", lambda *a, **k: [])
        monkeypatch.setattr(
            sim_data, "fetch_market_indicator_closes",
            lambda *a, **k: {"spy": [], "vix": [], "vix3m": []},
        )
        old = date.today() - timedelta(days=sim_data.YF_MINUTE_WINDOW_DAYS + 5)
        messages = []

        sim_data.create_dataset("old", ["TEST"], old, old, progress=messages.append)

        assert any("30 days" in m and m.startswith("warning") for m in messages)

    def test_the_alpaca_feeds_still_require_credentials(self, store, monkeypatch):
        monkeypatch.setattr(
            sim_data, "fetch_minute_bars_day",
            lambda *a, **k: pytest.fail("must not download without credentials"),
        )
        with pytest.raises(ValueError, match="Alpaca credentials"):
            sim_data.create_dataset("d", ["TEST"], DAY, DAY, feed="iex")

    def test_news_is_skipped_rather_than_failing_without_credentials(self, store, monkeypatch):
        """News is Alpaca-only; a credential-free yfinance dataset has none."""
        monkeypatch.setattr(
            sim_data, "fetch_news_day",
            lambda *a, **k: pytest.fail("must not reach Alpaca news without credentials"),
        )
        monkeypatch.setattr(sim_data, "fetch_minute_bars_day", lambda *a, **k: [])
        monkeypatch.setattr(sim_data, "fetch_daily_bars_range", lambda *a, **k: [])
        monkeypatch.setattr(
            sim_data, "fetch_market_indicator_closes",
            lambda *a, **k: {"spy": [], "vix": [], "vix3m": []},
        )
        day = date(2026, 6, 16)

        sim_data.create_dataset("yf", ["TEST"], day, day)

        assert sim_data.load_news("TEST", day) == []


def _tool_call(call_id, name, arguments):
    return SimpleNamespace(
        id=call_id, function=SimpleNamespace(name=name, arguments=json.dumps(arguments))
    )


def _response(tool_calls, content=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        outer = self

        class _Completions:
            def create(self, model, messages, tools, tool_choice):
                outer.calls.append(messages)
                return outer._responses.pop(0)

        self.chat = SimpleNamespace(completions=_Completions())


def _scripted_client():
    """Cycle 1: arm a buy-the-breakout tactic at 103 and finalize with alert.
    Cycle 2 (woken by the fill): sell everything. Cycle 3 (cycle timer): stand
    aside on a far-away alert for the rest of the day."""
    cycle1 = _response([
        _tool_call("c1a", "set_tactics", {
            "symbol": "TEST",
            "actions": [{
                "action": "buy", "quantity": 10,
                "conditions": [{"field": "last_price", "condition": "above", "value": 103.0}],
                "note": "breakout entry",
            }],
            "reasoning": "buy the break of 103",
        }),
        _tool_call("c1b", "submit_decision", {
            "action": "alert", "regime": "bullish", "reasoning": "waiting for the break", "alerts": [],
        }),
    ])
    cycle2 = _response([
        _tool_call("c2a", "submit_decision", {
            "action": "sell", "symbol": "TEST", "quantity": 10,
            "regime": "bullish", "reasoning": "taking the breakout profit",
        }),
    ])
    cycle3 = _response([
        _tool_call("c3a", "submit_decision", {
            "action": "alert", "regime": "neutral", "reasoning": "nothing to do",
            "alerts": [{"symbol": "TEST", "field": "last_price", "condition": "above", "value": 99999}],
        }),
    ])
    return FakeClient([cycle1, cycle2, cycle3])


def _run_sim(store, cycle_minutes=5):
    market = SimMarket(["TEST"], [DAY])
    config = SimulationConfig(
        personality="momentum", provider="openai", model="fake", api_key="",
        symbols=["TEST"], days=[DAY], starting_cash=10_000.0, cycle_minutes=cycle_minutes,
    )
    engine = SimulationEngine(market, config)
    result = engine.run(client=_scripted_client())
    return market, result


class TestEngine:
    def test_full_session_replay(self, store):
        market, result = _run_sim(store)
        assert result.error is None
        assert result.cycles_run == 3

        actions = [(d["action"], d["status"]) for d in result.decisions]
        assert actions == [
            ("tactics", "armed"),  # cycle 1: arm the breakout entry
            ("alert", "noop"),  # cycle 1: finalize (empty alerts, tactics armed)
            ("buy", "filled"),  # tactic fires mid-fast-forward
            ("sell", "filled"),  # cycle 2, woken by the fill
            ("alert", "noop"),  # cycle 3, cycle timer
        ]
        buy, sell = result.decisions[2], result.decisions[3]
        # The tactic fired on the first bar closing >= 103 (bar 30, completes
        # 14:01Z) and filled at that bar's close -- simulated time throughout.
        assert buy["price"] == pytest.approx(103.0)
        assert buy["ts"].startswith("2026-06-15T14:01")
        assert "Tactics triggered" in buy["reasoning"]
        assert sell["price"] >= buy["price"]  # ramping tape: sold at/above the entry

        # Equity: one point per completed bar, valued in simulated time.
        assert len(result.equity) == 60
        assert result.equity[0]["ts"].startswith("2026-06-15")
        fees = 2 * 1.15
        expected_final = 10_000.0 + 10 * (sell["price"] - buy["price"]) - fees
        assert result.final_value == pytest.approx(expected_final)

    def test_clock_restored_after_run(self, store):
        _run_sim(store)
        assert not clock.is_simulated()

    def test_result_records_the_prompt_and_tools_that_produced_it(self, store):
        # `prompt_overridden` alone can't distinguish two revisions of the
        # built-in prompt, so the run has to carry the resolved text itself.
        _, result = _run_sim(store)
        assert result.prompt_used == MOMENTUM_SYSTEM_PROMPT
        assert "get_quote" in result.tool_names
        assert "analyze_swing_levels" in result.tool_names

    def test_result_records_an_overriding_prompt(self, store):
        market = SimMarket(["TEST"], [DAY])
        config = SimulationConfig(
            personality="momentum", provider="openai", model="fake", api_key="",
            symbols=["TEST"], days=[DAY], starting_cash=10_000.0, cycle_minutes=5,
            system_prompt_override="custom plan",
        )
        result = SimulationEngine(market, config).run(client=_scripted_client())
        assert result.prompt_used == "custom plan"

    def test_summary_and_oracle(self, store):
        market, result = _run_sim(store)
        summary = sim_results.summarize_run(result, market)
        assert summary["trades_filled"] == 2
        # Oracle: buy 100.0, sell 105.9 -> 5.9%.
        assert summary["oracle_ceiling_pct"] == pytest.approx(5.9, abs=0.01)
        assert summary["profit_efficiency"] is not None
        assert summary["return_pct"] == pytest.approx(
            (result.final_value / 10_000.0 - 1.0) * 100.0
        )


class TestRuleAgentEngine:
    """The rule-based Apple Trader replays on the engine's other day loop: no
    LLM client, no prompt, no tools, one decision per closed bar."""

    @pytest.fixture()
    def apple_store(self, tmp_path, monkeypatch):
        """A stored AAPL session shaped like the setup the agent trades.

        60 minutes alternating 100.00 / 100.01 -- enough volatility for the
        momentum score to be defined, no net drift, so the regime stays
        *balanced*; then a steady climb that pushes momentum through the +0.90
        entry threshold (the one bar the model is ever asked about, and it lands
        well clear of the 30-bar opening warm-up); then a give-back steep enough
        to take out a trailing stop. Volume varies bar to bar because several
        features z-score it, and a constant would leave them undefined.
        """
        monkeypatch.setattr(sim_data, "STORE_DIR", tmp_path / "store")
        monkeypatch.setattr(sim_data, "MANIFEST_PATH", tmp_path / "datasets.json")
        prices = (
            [100.0 + 0.01 * (i % 2) for i in range(60)]
            + [100.0 + 0.03 * (i + 1) for i in range(40)]
            + [101.2 - 0.06 * (i + 1) for i in range(20)]
        )
        bars = [
            _bar(OPEN_UTC + timedelta(minutes=i), price, volume=1000.0 + 37 * (i % 13))
            for i, price in enumerate(prices)
        ]
        sim_data._write_gz(sim_data.bars_path("AAPL", DAY), bars)
        sim_data._write_gz(sim_data.daily_path("AAPL"), {
            "symbol": "AAPL", "start": "2026-05-16", "end": "2026-06-15", "bars": [],
        })
        return tmp_path

    @staticmethod
    def _bundle(proba: float = 0.9) -> dict:
        class Pipeline:
            def predict_proba(self, X):
                import numpy as np

                return np.column_stack(
                    [np.full(len(X), 1 - proba), np.full(len(X), proba)]
                )

        return {
            "pipeline": Pipeline(),
            "feature_columns": list(persistence_model.FEATURE_COLUMNS),
            "seq_len": 20,
            "threshold": 0.07,
            "settings": {"momentum": dict(persistence_model.MOMENTUM_DEFAULTS)},
            "metrics": {"roc_auc": 0.84},
        }

    # The stub bundle above is a classifier, so these runs pin the `confirm`
    # entry -- the one it can answer. `anticipate`, the default, needs a
    # forecasting bundle and is covered by its own case below.
    CONFIRM = {"model_key": "persistence", "entry_mode": "confirm"}

    def _run(self, monkeypatch, rule_config: dict, proba: float = 0.9):
        monkeypatch.setattr(
            persistence_model, "load_bundle", lambda *a, **k: self._bundle(proba)
        )
        rules = {**self.CONFIRM, **rule_config}
        market = SimMarket(["AAPL"], [DAY])
        config = SimulationConfig(
            personality=APPLE_TRADER_KEY, provider=RULE_PROVIDER,
            model=config_signature(), api_key="", symbols=["AAPL"], days=[DAY],
            starting_cash=10_000.0, rule_config=rules,
        )
        engine = SimulationEngine(market, config)
        return market, engine, engine.run()

    def test_reads_every_session_bar_and_needs_no_llm(self, apple_store, monkeypatch):
        # No client is passed and none is built: a rule run that quietly tried
        # to reach an LLM would raise here instead.
        _, engine, result = self._run(monkeypatch, {"prob_threshold": 0.5})
        assert result.error is None
        assert engine.rule_based
        # 120 stored bars complete at 09:31..11:30 ET -- every one inside the
        # session, so every one is read.
        assert result.cycles_run == 120
        assert not clock.is_simulated()

    def test_buys_the_backed_change_then_trails_out_of_it(self, apple_store, monkeypatch):
        _, _, result = self._run(monkeypatch, {"prob_threshold": 0.5, "trail_pct": 0.5})
        actions = [(d["action"], d["status"]) for d in result.decisions]
        assert actions[:2] == [("buy", "filled"), ("sell", "filled")]
        buy, sell = result.decisions[0], result.decisions[1]
        assert "-> positive" in buy["reasoning"] and "90%" in buy["reasoning"]
        assert "Trailing stop" in sell["reasoning"]
        # Bought early in the climb, held it all the way up -- the stop only
        # trails -- and sold into the give-back, below the $101.20 peak but
        # well above the entry: the profit lock, not the initial stop.
        assert parse_ts(sell["ts"]) > parse_ts(buy["ts"])
        assert 100.0 < float(buy["price"]) < 100.5
        assert float(buy["price"]) < float(sell["price"]) < 101.2

    def test_a_change_the_model_vetoes_is_never_bought(self, apple_store, monkeypatch):
        _, _, result = self._run(
            monkeypatch, {"prob_threshold": 0.5}, proba=0.01  # below the cut-off
        )
        assert result.error is None
        assert [d for d in result.decisions if d["action"] in ("buy", "sell")] == []

    def test_records_its_rules_instead_of_a_prompt(self, apple_store, monkeypatch):
        rules = {"prob_threshold": 0.5, "trail_pct": 0.5}
        _, _, result = self._run(monkeypatch, rules)
        assert result.prompt_used is None
        assert result.tool_names == []
        assert result.config_summary["rule_based"] is True
        assert result.config_summary["rule_config"] == {**self.CONFIRM, **rules}

    def test_a_dataset_without_the_ticker_fails_loudly(self, store, monkeypatch):
        monkeypatch.setattr(persistence_model, "load_bundle", lambda *a, **k: self._bundle())
        market = SimMarket(["TEST"], [DAY])
        config = SimulationConfig(
            personality=APPLE_TRADER_KEY, provider=RULE_PROVIDER, model="rules",
            api_key="", symbols=["TEST"], days=[DAY], rule_config=dict(self.CONFIRM),
        )
        result = SimulationEngine(market, config).run()
        assert "only trades AAPL" in (result.error or "")

    def test_a_missing_model_fails_loudly(self, apple_store, monkeypatch):
        monkeypatch.setattr(persistence_model, "load_bundle", lambda *a, **k: None)
        market = SimMarket(["AAPL"], [DAY])
        config = SimulationConfig(
            personality=APPLE_TRADER_KEY, provider=RULE_PROVIDER, model="rules",
            api_key="", symbols=["AAPL"], days=[DAY], rule_config=dict(self.CONFIRM),
        )
        result = SimulationEngine(market, config).run()
        # Names the file and the dependency: a run that simply never traded
        # would read like a strategy result rather than a setup problem.
        error = result.error or ""
        assert "timetochange2_persistence_AAPL.joblib" in error
        assert "scikit-learn" in error

    def test_anticipating_on_a_model_that_cannot_forecast_fails_loudly(
        self, apple_store, monkeypatch
    ):
        """The default entry needs a forecaster. Paired with the classifier the
        run would otherwise finish clean and empty -- `read_latest` leaves
        `turn_proba` None on every bar -- which reads like a strategy that
        found nothing rather than a rule set that could never fire.
        """
        monkeypatch.setattr(persistence_model, "load_bundle", lambda *a, **k: self._bundle())
        market = SimMarket(["AAPL"], [DAY])
        config = SimulationConfig(
            personality=APPLE_TRADER_KEY, provider=RULE_PROVIDER, model="rules",
            api_key="", symbols=["AAPL"], days=[DAY],
            rule_config={"model_key": "persistence", "entry_mode": "anticipate"},
        )
        result = SimulationEngine(market, config).run()
        error = result.error or ""
        assert "cannot forecast" in error
        assert "'confirm'" in error

    def test_the_named_model_is_the_one_loaded(self, apple_store, monkeypatch):
        """`model_key` selects which saved model answers the entry question.

        A record naming a model that cannot be assembled must fail on *that*
        model rather than quietly falling back to the one that can.
        """
        monkeypatch.setattr(
            apple_models, "load", lambda key, ticker=None: None if key == "nbeats" else self._bundle()
        )
        market = SimMarket(["AAPL"], [DAY])
        config = SimulationConfig(
            personality=APPLE_TRADER_KEY, provider=RULE_PROVIDER, model="rules",
            api_key="", symbols=["AAPL"], days=[DAY],
            rule_config={"model_key": "nbeats", "prob_threshold": 0.5},
        )
        result = SimulationEngine(market, config).run()
        assert "timetochange2_nbeats_AAPL.pt" in (result.error or "")

    def test_the_model_leads_the_configuration_signature(self):
        """Results groups runs on this string, so two models are two
        configurations even when every other knob matches."""
        rules = dict(prob_threshold=0.2, trail_pct=0.5)
        classifier = config_signature(AppleTraderConfig(model_key="persistence", **rules))
        forecaster = config_signature(AppleTraderConfig(model_key="nbeats", **rules))
        assert classifier != forecaster
        assert classifier.startswith("persistence_AAPL(")
        assert forecaster.startswith("nbeats_AAPL(")

    def test_rules_recorded_before_a_field_existed_decode_to_what_they_ran(self):
        """Records written when there was one model and one entry carry neither
        `model_key` nor `entry_mode`.

        They decode to the behaviour of the day they were written, NOT to
        today's defaults: a stored record describes a run that already
        happened, and replaying it as anticipate-on-N-BEATS would file a
        different strategy's numbers in Results beside the original as though
        they matched. (It would also refuse to build at all, since the model
        those records name cannot anticipate.)
        """
        agent = rule_agent(APPLE_TRADER_KEY)
        config = agent.from_record({"prob_threshold": 0.5})
        assert (config.model_key, config.entry_mode) == ("persistence", "confirm")
        assert config.prob_threshold == pytest.approx(0.5)
        # The trailing stop was the only exit then, so replaying such a record
        # must not sell on a forecast the run it describes never consulted.
        assert config.reversal_threshold is None
        assert not config.sells_on_reversal

        # A record that does name them is taken at its word.
        newer = agent.from_record(
            {"prob_threshold": 0.5, "entry_mode": "anticipate", "model_key": "nbeats"}
        )
        assert (newer.model_key, newer.entry_mode) == ("nbeats", "anticipate")

    def test_the_reversal_exit_survives_a_round_trip_through_the_record(self):
        agent = rule_agent(APPLE_TRADER_KEY)
        armed = AppleTraderConfig(model_key="nbeats", reversal_threshold=0.3)
        assert agent.from_record(agent.to_record(armed)).reversal_threshold == pytest.approx(0.3)
        off = AppleTraderConfig(model_key="nbeats", reversal_threshold=None)
        assert agent.from_record(agent.to_record(off)).reversal_threshold is None


class TestDayRangeEngine:
    """The day-range rules replayed end to end on the engine.

    The forecast itself is stubbed -- `tests/test_dayrange_model.py` pins that
    against the notebook -- so what this covers is the wiring: the strategy the
    model selects, the levels working over a real stored tape, and the daily
    history reaching the trader through the patched fetch rather than through
    yfinance.
    """

    # 105.50 predicted high on a $4 average daily range puts the shipped
    # 0.75 / 0.10 levels at 102.50 and 105.10, which the tape below crosses in
    # that order.
    FORECAST = {
        "pred_high": 105.5, "pred_low": 99.0, "prev_avg": 102.0,
        "adr14_abs": 4.0, "or_high": 104.1, "or_low": 103.9,
    }

    @pytest.fixture()
    def dayrange_store(self, tmp_path, monkeypatch, request):
        """A stored session that dips under the buy level and recovers through
        the sell level: five flat opening minutes the forecast is built on, a
        slide to 102.4, then a climb to 105.5.

        Written for AAPL unless the test is parameterised with another symbol
        (`indirect=True`), which is how the same session is replayed under a
        ticker the day-range model was retrained for.
        """
        symbol = getattr(request, "param", "AAPL")
        monkeypatch.setattr(sim_data, "STORE_DIR", tmp_path / "store")
        monkeypatch.setattr(sim_data, "MANIFEST_PATH", tmp_path / "datasets.json")
        prices = (
            [104.0] * 5
            + [104.0 - 0.16 * (i + 1) for i in range(10)]   # down to ~102.4
            + [102.4 + 0.20 * (i + 1) for i in range(16)]   # back up to ~105.6
            + [105.0] * 5
        )
        bars = [
            _bar(OPEN_UTC + timedelta(minutes=i), price, volume=1000.0 + 37 * (i % 13))
            for i, price in enumerate(prices)
        ]
        sim_data._write_gz(sim_data.bars_path(symbol, DAY), bars)
        sim_data._write_gz(sim_data.daily_path(symbol), {
            "symbol": symbol, "start": "2026-05-16", "end": "2026-06-15",
            "bars": [
                _bar(datetime(2026, 6, 15, tzinfo=timezone.utc) - timedelta(days=i), 99.0)
                for i in range(30, 0, -1)
            ],
        })
        return symbol

    def _stub_model(self, monkeypatch) -> dict:
        """The bundle and the forecast, replaced at the model module so the
        registry, the config check and the trader all see the stub."""
        dayrange = pytest.importorskip("agent_stonks.dayrange_model")
        bundle = {
            "model": None, "metadata": {}, "daily_models": ["lgbm", "nbeats", "nhits"],
            "opening_minutes": 5, "lookback": 32, "trained_at": "2026-08-10",
        }
        seen: dict = {}

        def forecast(bundle_, history, opening, session_date, open_price=None):
            seen["history"] = history
            seen["opening"] = opening
            seen["open_price"] = open_price
            return dict(self.FORECAST)

        monkeypatch.setattr(dayrange, "load_bundle", lambda ticker=None: bundle)
        monkeypatch.setattr(dayrange, "forecast_session", forecast)
        return seen

    def _run(self, monkeypatch, rule_config: "dict | None" = None, symbol: str = "AAPL"):
        seen = self._stub_model(monkeypatch)
        rules = {"model_key": "dayrange", "ticker": symbol, **(rule_config or {})}
        market = SimMarket([symbol], [DAY])
        config = SimulationConfig(
            personality=APPLE_TRADER_KEY, provider=RULE_PROVIDER,
            model=config_signature(AppleTraderConfig(**rules)), api_key="",
            symbols=[symbol], days=[DAY], starting_cash=10_000.0, rule_config=rules,
        )
        return seen, SimulationEngine(market, config).run()

    def test_buys_the_dip_under_the_buy_level_and_sells_at_the_target(
        self, dayrange_store, monkeypatch
    ):
        _, result = self._run(monkeypatch)
        assert result.error is None
        actions = [(d["action"], d["status"]) for d in result.decisions]
        assert actions[:2] == [("buy", "filled"), ("sell", "filled")]
        buy, sell = result.decisions[0], result.decisions[1]
        assert "102.50" in buy["reasoning"]
        assert "Target" in sell["reasoning"]
        assert float(buy["price"]) <= 102.6
        assert float(sell["price"]) >= 105.0
        assert parse_ts(sell["ts"]) > parse_ts(buy["ts"])

    def test_moving_the_levels_moves_the_fills(self, dayrange_store, monkeypatch):
        """The two knobs are the strategy: a shallower buy distance enters
        higher up the slide, and an earlier sell exits sooner."""
        _, deep = self._run(monkeypatch)
        _, shallow = self._run(monkeypatch, {"buy_k": 0.2, "sell_k": 0.05})
        assert float(shallow.decisions[0]["price"]) > float(deep.decisions[0]["price"])
        assert parse_ts(shallow.decisions[0]["ts"]) < parse_ts(deep.decisions[0]["ts"])

    def test_the_daily_history_comes_from_the_dataset_not_the_network(
        self, dayrange_store, monkeypatch
    ):
        """The model needs a year of daily bars, which live is a yfinance
        download of the last 420 days -- wall-clock days, not simulated ones.
        Inside a run it has to be the stored history, clipped."""
        seen, result = self._run(monkeypatch)
        assert result.error is None
        history = seen["history"]
        assert len(history) == 30
        assert history.index.max() < pd.Timestamp("2026-06-15")

    def test_the_forecast_is_built_on_the_first_five_stored_minutes(
        self, dayrange_store, monkeypatch
    ):
        seen, _ = self._run(monkeypatch)
        opening = seen["opening"]
        assert len(opening) == 5
        assert opening.index[0].strftime("%H:%M") == "09:30"
        assert opening.index[-1].strftime("%H:%M") == "09:34"

    def test_the_entry_mode_and_reversal_defaults_do_not_block_the_run(
        self, dayrange_store, monkeypatch
    ):
        """A day-range record carries `anticipate` and a reversal threshold
        because one dataclass serves both strategies. Neither means anything
        here, and neither may stop the run the way they would on a classifier.
        """
        _, result = self._run(
            monkeypatch, {"entry_mode": "anticipate", "reversal_threshold": 0.3}
        )
        assert result.error is None
        assert [d for d in result.decisions if d["action"] == "buy"]

    def test_a_missing_bundle_fails_loudly(self, dayrange_store, monkeypatch):
        dayrange = pytest.importorskip("agent_stonks.dayrange_model")
        monkeypatch.setattr(dayrange, "load_bundle", lambda ticker=None: None)
        market = SimMarket(["AAPL"], [DAY])
        config = SimulationConfig(
            personality=APPLE_TRADER_KEY, provider=RULE_PROVIDER, model="rules",
            api_key="", symbols=["AAPL"], days=[DAY],
            rule_config={"model_key": "dayrange"},
        )
        result = SimulationEngine(market, config).run()
        error = result.error or ""
        assert "timetochange3_dayrange_AAPL.joblib" in error
        assert "PyTorch" in error

    @pytest.mark.parametrize("dayrange_store", ["GOOGL", "INTC"], indirect=True)
    def test_the_retrained_symbols_replay_on_the_same_day_loop(
        self, dayrange_store, monkeypatch
    ):
        """TimeToChange3 was fitted per ticker, so GOOGL and INTC are the same
        strategy on another tape -- nothing about the loop, the levels or the
        ledger changes, and the run is filed under its own symbol."""
        symbol = dayrange_store
        seen, result = self._run(monkeypatch, symbol=symbol)
        assert result.error is None
        assert [d["symbol"] for d in result.decisions][:2] == [symbol, symbol]
        assert result.config_summary["rule_config"]["ticker"] == symbol
        # The daily history and the opening window came from that symbol's
        # stored bars -- there are no AAPL bars in this dataset at all, so a
        # path still hard-coded to AAPL could not have produced a forecast.
        assert len(seen["history"]) == 30
        assert len(seen["opening"]) == 5

    def test_a_momentum_model_cannot_be_pointed_at_another_symbol(
        self, dayrange_store, monkeypatch
    ):
        """The pairing check reaches SimLab through the same `config_error`
        the live loop uses, so a record naming one cannot be replayed.

        MSFT rather than GOOGL: since TimeToChange2 was re-run per ticker every
        shipped model covers every shipped symbol, so an unmodelled symbol is
        now the only pairing the registry refuses."""
        market = SimMarket(["MSFT"], [DAY])
        config = SimulationConfig(
            personality=APPLE_TRADER_KEY, provider=RULE_PROVIDER, model="rules",
            api_key="", symbols=["MSFT"], days=[DAY],
            rule_config={"model_key": "nbeats", "ticker": "MSFT"},
        )
        error = SimulationEngine(market, config).run().error or ""
        assert "cannot trade MSFT" in error and "AAPL, GOOGL, INTC only" in error

    def test_the_dataset_is_checked_against_the_configured_symbol(
        self, dayrange_store, monkeypatch
    ):
        """An AAPL dataset and a GOOGL config is a run that cannot trade."""
        self._stub_model(monkeypatch)
        market = SimMarket(["AAPL"], [DAY])
        config = SimulationConfig(
            personality=APPLE_TRADER_KEY, provider=RULE_PROVIDER, model="rules",
            api_key="", symbols=["AAPL"], days=[DAY],
            rule_config={"model_key": "dayrange", "ticker": "GOOGL"},
        )
        assert "only trades GOOGL" in (SimulationEngine(market, config).run().error or "")

    def test_the_strategy_leads_the_configuration_signature(self, dayrange_store, monkeypatch):
        """Results groups on this string. A day-range run and a momentum run
        are different strategies, not two settings of one."""
        _, result = self._run(monkeypatch)
        assert result.config_summary["rule_based"] is True
        assert config_signature(
            AppleTraderConfig(model_key="dayrange")
        ).startswith("dayrange_AAPL(buy=")


class TestMomentumChangeEngine:
    """The delta-momentum rules replayed end to end on the engine.

    The estimator is stubbed -- `tests/test_momentum_change_model.py` pins the model
    against momlib itself -- so what this covers is the wiring, and one part of
    it exists nowhere else in the app: this is the only strategy that reads
    **minute bars from days before the one being simulated**. Live that is six
    yfinance downloads; inside a run it has to be the dataset's stored bars,
    reached through the patched `fetch_intraday_bars_for_date`. A run that
    quietly fetched wall-clock history instead would still trade, and every
    regime it traded would be wrong.
    """

    # Any symbol the delta-momentum model still covers -- the estimator is
    # stubbed, so what the symbol has to satisfy is `model_ticker_error`, which
    # refuses a pairing the registry does not carry before the run starts. It
    # was GOOGL until that bundle was withdrawn from `Code/Models`.
    SYMBOL = "INTC"
    # 09:30 ET on six quiet sessions, then the one that trades. Short days: the
    # rules need twenty bars of warm-up and nothing here needs a real session,
    # so `MIN_BARS_PER_DAY` is lowered to match rather than writing 2,700 bars.
    HISTORY_DAYS = [date(2026, 6, d) for d in (5, 8, 9, 10, 11, 12)]
    BARS_PER_HISTORY_DAY = 40

    def _prices(self) -> list[float]:
        """Flat, then a slide into a negative regime, then a recovery through
        balanced into positive -- the two bars the rules are written for."""
        return (
            [100.0] * 22
            + [100.0 - 0.02 * (i + 1) for i in range(18)]
            + [99.64 + 0.04 * (i + 1) for i in range(25)]
        )

    @pytest.fixture()
    def momentum_change_store(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sim_data, "STORE_DIR", tmp_path / "store")
        monkeypatch.setattr(sim_data, "MANIFEST_PATH", tmp_path / "datasets.json")
        momentum_change = pytest.importorskip("agent_stonks.momentum_change_model")
        monkeypatch.setattr(momentum_change, "MIN_BARS_PER_DAY", 20)
        momentum_change.reset_history_cache()

        for day in self.HISTORY_DAYS:
            open_utc = datetime.combine(
                day, datetime.min.time(), tzinfo=timezone.utc
            ).replace(hour=13, minute=30)
            # A gentle wiggle rather than a flat line: `theta` is 0.4 times the
            # previous session's minute volatility, and a session with none
            # would make every regime threshold zero.
            bars = [
                _bar(open_utc + timedelta(minutes=i), 100.0 * (1 + 0.0002 * ((-1) ** i)))
                for i in range(self.BARS_PER_HISTORY_DAY)
            ]
            sim_data._write_gz(sim_data.bars_path(self.SYMBOL, day), bars)

        sim_data._write_gz(
            sim_data.bars_path(self.SYMBOL, DAY),
            [
                _bar(OPEN_UTC + timedelta(minutes=i), price)
                for i, price in enumerate(self._prices())
            ],
        )
        sim_data._write_gz(sim_data.daily_path(self.SYMBOL), {
            "symbol": self.SYMBOL, "start": "2026-05-16", "end": "2026-06-15", "bars": [],
        })
        return self.SYMBOL

    def _stub_bundle(self, monkeypatch) -> dict:
        """A bundle whose estimator reads the regime out of the feature row.

        Deterministic and direction-aware, which is what makes both rules
        reachable on one tape: a turn called up out of the negative regime, and
        the same regime called over once it is positive.
        """
        momentum_change = pytest.importorskip("agent_stonks.momentum_change_model")

        class Scripted:
            def predict(self, X):
                regime = X["regime_before"].to_numpy(float)
                return np.where(regime < 0, 0.9, np.where(regime > 0, -0.9, 0.0))

        bundle = {
            "estimator": Scripted(),
            "feature_cols": list(momentum_change.FEATURE_COLS),
            "pipeline_params": dict(momentum_change.PIPELINE_DEFAULTS),
            "model_name": "Scripted",
            "metrics": {},
            "saved_at": "2026-09-07",
        }
        monkeypatch.setattr(momentum_change, "load_bundle", lambda ticker=None: bundle)
        return bundle

    def _run(self, monkeypatch, days: "list[date] | None" = None, **overrides):
        self._stub_bundle(monkeypatch)
        rules = {
            "model_key": "momentum_change", "ticker": self.SYMBOL,
            # The two risk exits are pinned by the unit tests; widened here so
            # this one is about the wiring rather than about which rule fires
            # first on a synthetic slide.
            "m1_mult": -6.0, "stop_pct": 5.0,
            **overrides,
        }
        run_days = self.HISTORY_DAYS + [DAY] if days is None else days
        market = SimMarket([self.SYMBOL], run_days)
        config = SimulationConfig(
            personality=APPLE_TRADER_KEY, provider=RULE_PROVIDER,
            model=config_signature(AppleTraderConfig(**rules)), api_key="",
            symbols=[self.SYMBOL], days=run_days, starting_cash=10_000.0,
            rule_config=rules,
        )
        return SimulationEngine(market, config).run()

    def test_it_buys_the_called_turn_and_sells_the_called_top(
        self, momentum_change_store, monkeypatch
    ):
        result = self._run(monkeypatch)
        assert result.error is None
        actions = [(d["action"], d["status"]) for d in result.decisions]
        assert actions[:2] == [("buy", "filled"), ("sell", "filled")]
        buy, sell = result.decisions[0], result.decisions[1]
        assert "negative momentum regime" in buy["reasoning"]
        assert "Model exit" in sell["reasoning"]
        assert parse_ts(sell["ts"]) > parse_ts(buy["ts"])
        # Bought into the slide and sold into the recovery, which is the shape
        # of the rule rather than an accident of the tape.
        assert float(sell["price"]) > float(buy["price"])

    def test_the_previous_sessions_come_from_the_dataset_not_the_network(
        self, momentum_change_store, monkeypatch
    ):
        """The one leak this strategy could have. `fetch_intraday_bars_for_date`
        live is a yfinance download for a wall-clock date; unpatched inside a
        run it would hand the model six real sessions from 2026-09 while the
        simulated day is 2026-06-15."""
        momentum_change = pytest.importorskip("agent_stonks.momentum_change_model")
        calls: list = []
        real = momentum_change.history_bars

        def spy(ticker, session_date, **kwargs):
            bars = real(ticker, session_date, **kwargs)
            calls.append((str(session_date)[:10], len(bars)))
            return bars

        monkeypatch.setattr(momentum_change, "history_bars", spy)
        result = self._run(monkeypatch)
        assert result.error is None

        fetched = dict(calls)
        assert fetched[str(DAY)] == len(self.HISTORY_DAYS) * self.BARS_PER_HISTORY_DAY
        # Every session the model saw is one the dataset holds, and all of them
        # finished before the simulated day.
        assert max(d for d, _ in calls) == str(DAY)

    def test_a_dataset_with_nothing_behind_it_refuses_rather_than_trades(
        self, momentum_change_store, monkeypatch
    ):
        """The regime threshold is yesterday's volatility, so the first day of
        a dataset has nothing to build one from. Scoring anyway would produce
        confident numbers off a threshold the model never saw -- so the day is
        skipped, once, in the log."""
        result = self._run(monkeypatch, days=[DAY])
        assert result.error is None
        assert not [d for d in result.decisions if d["action"] == "buy"]

    def test_the_run_is_filed_under_its_own_signature(
        self, momentum_change_store, monkeypatch
    ):
        result = self._run(monkeypatch)
        assert result.error is None
        assert result.config_summary["rule_config"]["ticker"] == self.SYMBOL
        assert result.config_summary["model"].startswith(
            f"momentum_change_{self.SYMBOL}("
        )

    def test_a_missing_bundle_fails_loudly(self, momentum_change_store, monkeypatch):
        momentum_change = pytest.importorskip("agent_stonks.momentum_change_model")
        monkeypatch.setattr(momentum_change, "load_bundle", lambda ticker=None: None)
        market = SimMarket([self.SYMBOL], [DAY])
        config = SimulationConfig(
            personality=APPLE_TRADER_KEY, provider=RULE_PROVIDER, model="rules",
            api_key="", symbols=[self.SYMBOL], days=[DAY],
            rule_config={"model_key": "momentum_change", "ticker": self.SYMBOL},
        )
        result = SimulationEngine(market, config).run()
        error = result.error or ""
        assert f"momentum_change_{self.SYMBOL}.joblib" in error



class TestAppleTrader2Engine:
    """Apple Trader 2 on the same rule day loop, driven by a rule list.

    The rule sets here are written on price and the position alone, which is
    the point: a strategy that names no model needs no bundle on disk, so these
    exercise the engine wiring rather than a stubbed forecast.
    """

    @pytest.fixture()
    def apple_store(self, tmp_path, monkeypatch):
        """A stored AAPL session that climbs, then gives back."""
        monkeypatch.setattr(sim_data, "STORE_DIR", tmp_path / "store")
        monkeypatch.setattr(sim_data, "MANIFEST_PATH", tmp_path / "datasets.json")
        prices = (
            [100.0 + 0.03 * (i + 1) for i in range(40)]
            + [101.2 - 0.06 * (i + 1) for i in range(20)]
        )
        bars = [
            _bar(OPEN_UTC + timedelta(minutes=i), price, volume=1000.0 + 37 * (i % 13))
            for i, price in enumerate(prices)
        ]
        sim_data._write_gz(sim_data.bars_path("AAPL", DAY), bars)
        sim_data._write_gz(sim_data.daily_path("AAPL"), {
            "symbol": "AAPL", "start": "2026-05-16", "end": "2026-06-15", "bars": [],
        })
        return tmp_path

    RULES = {
        "rules": {
            "items": [
                {
                    "action": "sell", "size_mode": "pct", "size": 100.0, "join": "all",
                    "label": "Trailing stop", "enabled": True, "cooldown_bars": 0,
                    "conditions": [
                        {"field": "pos.drawdown_pct", "op": "below", "value": -0.5}
                    ],
                },
                {
                    "action": "buy", "size_mode": "pct", "size": 95.0, "join": "all",
                    "label": "Buy the dip", "enabled": True, "cooldown_bars": 0,
                    # "while flat" is a condition here rather than a property of
                    # the agent -- without it the same rule would add on every
                    # later bar its price condition still holds.
                    "conditions": [
                        {"field": "pos.shares", "op": "below", "value": 0.0},
                        {"field": "bar.price", "op": "above", "value": 100.5},
                    ],
                },
            ]
        }
    }

    def _run(self, rule_config: "dict | None" = None, symbol: str = "AAPL"):
        market = SimMarket([symbol], [DAY])
        config = SimulationConfig(
            personality=APPLE_TRADER2_KEY, provider=RULE_PROVIDER, model="rules",
            api_key="", symbols=[symbol], days=[DAY], starting_cash=10_000.0,
            rule_config=rule_config or self.RULES,
        )
        return SimulationEngine(market, config).run()

    def test_a_rule_list_replays_with_no_llm_and_no_model(self, apple_store):
        result = self._run()
        assert result.error is None
        assert result.prompt_used is None and result.tool_names == []
        assert result.cycles_run == 60

    def test_it_buys_on_its_entry_rule_and_exits_on_its_stop(self, apple_store):
        result = self._run()
        actions = [(d["action"], d["status"]) for d in result.decisions]
        assert actions[:2] == [("buy", "filled"), ("sell", "filled")]
        buy, sell = result.decisions[0], result.decisions[1]
        assert "Buy the dip" in buy["reasoning"] and "Rule 2" in buy["reasoning"]
        assert "Trailing stop" in sell["reasoning"] and "Rule 1" in sell["reasoning"]
        assert float(buy["price"]) < float(sell["price"])

    def test_the_rule_list_is_what_the_record_carries(self, apple_store):
        result = self._run()
        assert result.config_summary["rule_based"] is True
        stored = result.config_summary["rule_config"]
        assert stored["rules"]["items"][1]["conditions"][1]["field"] == "bar.price"

    def test_a_rule_set_that_cannot_work_fails_loudly(self, apple_store):
        """Before the run, not after: an unrunnable set would otherwise finish
        clean with an empty ledger that reads like a strategy result."""
        result = self._run({"rules": {"items": []}})
        assert "no enabled rules" in (result.error or "")

    def test_a_dataset_without_the_ticker_fails_loudly(self, store):
        market = SimMarket(["TEST"], [DAY])
        config = SimulationConfig(
            personality=APPLE_TRADER2_KEY, provider=RULE_PROVIDER, model="rules",
            api_key="", symbols=["TEST"], days=[DAY], rule_config=self.RULES,
        )
        assert "only trades AAPL" in (SimulationEngine(market, config).run().error or "")

    def test_the_symbol_it_trades_comes_from_the_record(self, tmp_path, monkeypatch):
        """The same rules over another instrument. Nothing about the day loop
        changes -- which is the claim worth pinning, since the ticker moved from
        a constant on the agent to a field on the config."""
        monkeypatch.setattr(sim_data, "STORE_DIR", tmp_path / "store")
        monkeypatch.setattr(sim_data, "MANIFEST_PATH", tmp_path / "datasets.json")
        prices = (
            [100.0 + 0.03 * (i + 1) for i in range(40)]
            + [101.2 - 0.06 * (i + 1) for i in range(20)]
        )
        sim_data._write_gz(
            sim_data.bars_path("GOOGL", DAY),
            [_bar(OPEN_UTC + timedelta(minutes=i), price) for i, price in enumerate(prices)],
        )

        result = self._run({**self.RULES, "ticker": "GOOGL"}, symbol="GOOGL")
        assert result.error is None
        assert [d["symbol"] for d in result.decisions][:2] == ["GOOGL", "GOOGL"]
        assert result.config_summary["rule_config"]["ticker"] == "GOOGL"

    def test_the_dataset_is_checked_against_the_configured_symbol(self, apple_store):
        """An AAPL dataset and a GOOGL rule set is a run that cannot trade, and
        it is caught before it produces an empty ledger."""
        result = self._run({**self.RULES, "ticker": "GOOGL"})
        assert "only trades GOOGL" in (result.error or "")


class TestRuleAgentRegistry:
    """What the engine, the runner and the UI rely on being true of *every*
    rule agent, so adding one cannot half-wire it."""

    def test_every_rule_agent_is_replayable(self):
        assert set(RULE_AGENTS) == {APPLE_TRADER_KEY, APPLE_TRADER2_KEY}
        for key, agent in RULE_AGENTS.items():
            assert agent.key == key
            assert agent.default_ticker and agent.label
            # The symbol is asked of a config, not of the agent: one of the two
            # has it fixed and the other configurable, and the engine's dataset
            # check has to work the same way for both.
            assert agent.ticker(agent.from_record(None)) == agent.default_ticker
            # Prompt-driven and rule-driven are the two kinds of agent, and no
            # agent is both: the UI splits the picker on exactly this.
            assert not sim_prompts.has_prompt(key)

    def test_a_config_survives_the_experiment_record(self):
        for agent in RULE_AGENTS.values():
            defaults = agent.from_record(None)
            record = json.loads(json.dumps(agent.to_record(defaults)))
            assert agent.from_record(record) == defaults

    def test_signatures_separate_configurations(self):
        # Results groups runs on this string, so two rule sets that trade
        # differently must not collapse into one "already tested" row. Signed
        # on the day-range model, whose signature reads no threshold and so
        # needs no bundle on disk to answer.
        agent = rule_agent(APPLE_TRADER_KEY)
        base = agent.from_record({"model_key": "dayrange"})
        assert agent.signature(base) == agent.signature(
            agent.from_record({"model_key": "dayrange"})
        )
        for field, value in (
            ("buy_k", 0.2), ("sell_k", 0.3), ("position_pct", 50.0),
        ):
            other = replace(base, **{field: value})
            assert agent.signature(other) != agent.signature(base), field


class TestOracle:
    def test_best_round_trip_orders_matter(self):
        assert sim_results.oracle_best_round_trip([105, 100, 104]) == pytest.approx(4.0)
        assert sim_results.oracle_best_round_trip([105, 104, 103]) == 0.0
        assert sim_results.oracle_best_round_trip([]) == 0.0


class TestJudgeContext:
    def test_entry_context_includes_tape_and_outcome(self, store):
        market, result = _run_sim(store)
        buy = result.decisions[2]
        exit_decision = _first_exit_after(buy, result.decisions)
        assert exit_decision is not None and exit_decision["action"] == "sell"
        context = _entry_context(buy, exit_decision, market)
        assert "ENTRY: buy" in context
        assert "TAPE BEFORE ENTRY" in context
        assert "max favorable excursion" in context
        assert "EXIT: sold" in context


class TestPatches:
    def test_market_indicators_clip_to_sim_time(self, store):
        sim_data._write_gz(sim_data.market_path(), {
            "spy": [
                {"date": "2026-06-12", "close": 500.0},
                {"date": "2026-06-15", "close": 501.0},
                {"date": "2026-06-16", "close": 502.0},
            ],
            "vix": [{"date": "2026-06-15", "close": 15.0}],
            "vix3m": [],
        })
        market = SimMarket(["TEST"], [DAY])
        from agent_stonks import historical

        with simulation_context(market):
            clock.set_simulated(OPEN_UTC)
            series = historical.fetch_market_indicators()
            # 2026-06-16 is the future from the pinned clock -- must be clipped.
            assert list(series["spy"].values) == [500.0, 501.0]
            assert float(series["vix"].iloc[-1]) == 15.0
        assert not clock.is_simulated()

    def test_the_daily_history_a_session_model_reads_is_dataset_backed(self, store):
        """Live this is a yfinance download of the last 420 days. Inside a
        simulation it has to be the stored bars, clipped to the simulated day,
        or a per-session model forecasts today off a history that runs to
        wall-clock today."""
        from agent_stonks import historical

        stored = sim_data._read_gz(sim_data.daily_path("TEST"))
        stored["bars"].append(_bar(datetime(2026, 6, 15, tzinfo=timezone.utc), 123.0))
        sim_data._write_gz(sim_data.daily_path("TEST"), stored)

        with simulation_context(SimMarket(["TEST"], [DAY])):
            clock.set_simulated(OPEN_UTC + timedelta(minutes=10))
            bars = historical.fetch_daily_ohlc_bars("TEST")
            assert bars and all(str(b["t"])[:10] < "2026-06-15" for b in bars)
            assert set(bars[0]) == {"t", "o", "h", "l", "c", "v"}
            assert historical.fetch_session_open("TEST") == pytest.approx(122.95)

    def test_patches_are_restored(self, store):
        from agent_stonks import historical

        original = historical.fetch_market_indicators
        with simulation_context(SimMarket(["TEST"], [DAY])):
            assert historical.fetch_market_indicators is not original
        assert historical.fetch_market_indicators is original


class TestPrompts:
    def test_override_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sim_prompts, "PROMPTS_DIR", tmp_path / "prompts")
        assert sim_prompts.get_prompt("momentum") == sim_prompts.default_prompt("momentum")
        assert not sim_prompts.has_override("momentum")
        sim_prompts.save_override("momentum", "You are a test agent.")
        assert sim_prompts.get_prompt("momentum") == "You are a test agent."
        sim_prompts.reset_override("momentum")
        assert sim_prompts.get_prompt("momentum") == sim_prompts.default_prompt("momentum")

    def test_unknown_personality_rejected(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sim_prompts, "PROMPTS_DIR", tmp_path / "prompts")
        with pytest.raises(KeyError):
            sim_prompts.save_override("nope", "x")

    def test_the_rule_agent_has_no_prompt(self, tmp_path, monkeypatch):
        """It is configured by numbers, not text, so every prompt helper is a
        no-op for it rather than a KeyError."""
        monkeypatch.setattr(sim_prompts, "PROMPTS_DIR", tmp_path / "prompts")
        assert not sim_prompts.has_prompt(APPLE_TRADER_KEY)
        assert sim_prompts.get_prompt(APPLE_TRADER_KEY) == ""
        assert not sim_prompts.has_override(APPLE_TRADER_KEY)
        assert sim_prompts.get_override(APPLE_TRADER_KEY) is None


class TestExperiments:
    @pytest.fixture(autouse=True)
    def exp_store(self, tmp_path, monkeypatch):
        from simlab import experiments as sim_experiments

        monkeypatch.setattr(sim_experiments, "EXPERIMENTS_DIR", tmp_path / "experiments")
        self.experiments = sim_experiments

    def _submit(self, **overrides):
        config = {
            "personality": "momentum", "provider": "openai", "model": "gpt-test",
            "api_key": "sk-secret", "symbols": ["TEST"], "days": [DAY.isoformat()],
            "starting_cash": 100_000.0, "cycle_minutes": 5,
            "max_cycles_per_day": 40, "system_prompt_override": None,
            "run_judge": False,
        }
        config.update(overrides)
        return self.experiments.submit("ds1", config)

    def test_submit_and_finalize_scrubs_api_key(self):
        exp = self._submit(judge_api_key="sk-judge")
        assert exp["status"] == self.experiments.WAITING
        stored = self.experiments.get_experiment(exp["experiment_id"])
        assert stored["config"]["api_key"] == "sk-secret"

        done = self.experiments.finalize(
            exp["experiment_id"], self.experiments.FINISHED, run_id="run-1",
            result_summary={"return_pct": 1.5},
        )
        assert done["status"] == self.experiments.FINISHED
        assert done["run_id"] == "run-1"
        assert done["config"]["api_key"] == ""
        assert done["config"]["judge_api_key"] == ""
        assert done["finished_at"] is not None

    def test_tick_spawns_up_to_limit_oldest_first(self, monkeypatch):
        first = self._submit()
        second = self._submit()
        third = self._submit()
        spawned = []
        monkeypatch.setattr(self.experiments, "spawn", spawned.append)
        self.experiments.tick(max_parallel=2)
        assert spawned == [first["experiment_id"], second["experiment_id"]]
        assert third["experiment_id"] not in spawned

    def test_tick_counts_running_against_limit(self, monkeypatch):
        running = self._submit()
        self.experiments.update(
            running["experiment_id"], status=self.experiments.RUNNING, pid=99999999
        )
        waiting = self._submit()
        spawned = []
        monkeypatch.setattr(self.experiments, "spawn", spawned.append)
        monkeypatch.setattr(self.experiments, "_pid_alive", lambda pid: True)
        self.experiments.tick(max_parallel=1)
        assert spawned == []
        self.experiments.tick(max_parallel=2)
        assert spawned == [waiting["experiment_id"]]

    def test_stop_kills_the_worker_and_finalizes(self, monkeypatch):
        exp = self._submit()
        self.experiments.update(
            exp["experiment_id"], status=self.experiments.RUNNING, pid=4242
        )
        killed = []
        monkeypatch.setattr(self.experiments, "_pid_alive", lambda pid: True)
        monkeypatch.setattr(self.experiments.os, "getpgid", lambda pid: pid)
        monkeypatch.setattr(
            self.experiments.os, "killpg", lambda pgid, sig: killed.append((pgid, sig))
        )
        stopped = self.experiments.stop(exp["experiment_id"])
        assert killed == [(4242, self.experiments.signal.SIGTERM)]
        assert stopped["status"] == self.experiments.FAILED
        assert stopped["error"] == self.experiments.STOPPED_ERROR
        assert stopped["config"]["api_key"] == ""

    def test_stop_leaves_a_finished_experiment_alone(self, monkeypatch):
        exp = self._submit()
        self.experiments.finalize(
            exp["experiment_id"], self.experiments.FINISHED, run_id="run-1"
        )
        monkeypatch.setattr(self.experiments, "_pid_alive", lambda pid: True)
        untouched = self.experiments.stop(exp["experiment_id"])
        assert untouched["status"] == self.experiments.FINISHED
        assert untouched["run_id"] == "run-1"
        assert untouched["error"] is None

    def test_stop_finalizes_a_waiting_experiment_without_a_worker(self):
        exp = self._submit()
        stopped = self.experiments.stop(exp["experiment_id"])
        assert stopped["status"] == self.experiments.FAILED
        assert stopped["error"] == self.experiments.STOPPED_ERROR

    def test_stopped_experiment_is_not_reaped_again(self, monkeypatch):
        exp = self._submit()
        self.experiments.update(
            exp["experiment_id"], status=self.experiments.RUNNING, pid=4242
        )
        monkeypatch.setattr(self.experiments, "_pid_alive", lambda pid: False)
        monkeypatch.setattr(self.experiments, "spawn", lambda _id: None)
        self.experiments.stop(exp["experiment_id"])
        self.experiments.tick(max_parallel=1)
        after = self.experiments.get_experiment(exp["experiment_id"])
        assert after["error"] == self.experiments.STOPPED_ERROR

    def test_tick_reaps_dead_workers(self, monkeypatch):
        exp = self._submit()
        self.experiments.update(
            exp["experiment_id"], status=self.experiments.RUNNING, pid=99999999
        )
        monkeypatch.setattr(self.experiments, "_pid_alive", lambda pid: False)
        monkeypatch.setattr(self.experiments, "spawn", lambda _id: None)
        self.experiments.tick(max_parallel=1)
        reaped = self.experiments.get_experiment(exp["experiment_id"])
        assert reaped["status"] == self.experiments.FAILED
        assert "died" in reaped["error"]

    def test_clear_finished_keeps_active(self):
        active = self._submit()
        done = self._submit()
        self.experiments.finalize(done["experiment_id"], self.experiments.FINISHED)
        assert self.experiments.clear_finished() == 1
        ids = [e["experiment_id"] for e in self.experiments.list_experiments()]
        assert ids == [active["experiment_id"]]


class TestBreakdown:
    def _run(self, model="gpt-a", dataset="ds1", personality="momentum",
             return_pct=1.0, efficiency=0.5, judge_score=7.0):
        record = {
            "dataset": dataset,
            "config_summary": {"provider": "openai", "model": model, "personality": personality},
            "summary": {"return_pct": return_pct, "profit_efficiency": efficiency},
        }
        if judge_score is not None:
            record["judge"] = {"overall_score": judge_score}
        return record

    def test_groups_by_model(self):
        rows = sim_results.breakdown(
            [
                self._run(model="gpt-a", return_pct=2.0, efficiency=0.8, judge_score=8.0),
                self._run(model="gpt-a", return_pct=0.0, efficiency=0.2, judge_score=None),
                self._run(model="gpt-b", return_pct=-1.0, efficiency=0.1),
            ],
            by="model",
        )
        assert [r["group"] for r in rows] == ["openai/gpt-a", "openai/gpt-b"]
        top = rows[0]
        assert top["runs"] == 2
        assert top["avg_return_pct"] == 1.0
        assert top["best_return_pct"] == 2.0
        assert top["avg_profit_efficiency"] == 0.5
        assert top["avg_judge_score"] == 8.0  # unjudged run skipped

    def test_groups_by_dataset_and_agent(self):
        runs = [
            self._run(dataset="ds1", personality="momentum"),
            self._run(dataset="ds2", personality="contrarian", efficiency=None),
        ]
        by_dataset = sim_results.breakdown(runs, by="dataset")
        assert {r["group"] for r in by_dataset} == {"ds1", "ds2"}
        by_agent = sim_results.breakdown(runs, by="agent")
        assert {r["group"] for r in by_agent} == {"momentum", "contrarian"}
        no_eff = next(r for r in by_agent if r["group"] == "contrarian")
        assert no_eff["avg_profit_efficiency"] is None
        # None-efficiency groups sort after scored ones.
        assert by_agent[-1]["group"] == "contrarian"

    def test_best_run_names_the_winning_run(self):
        rows = sim_results.breakdown(
            [
                self._run(model="gpt-a", dataset="ds1", personality="momentum", return_pct=2.0),
                self._run(model="gpt-a", dataset="ds2", personality="breakout", return_pct=5.0),
            ],
            by="model",
        )
        best = rows[0]["best_run"]
        assert rows[0]["best_return_pct"] == 5.0
        assert best["return_pct"] == 5.0
        assert (best["provider"], best["model"]) == ("openai", "gpt-a")
        assert best["personality"] == "breakout"
        assert best["dataset"] == "ds2"

    def test_best_run_absent_without_returns(self):
        rows = sim_results.breakdown([self._run(return_pct=None)], by="model")
        assert rows[0]["best_return_pct"] is None
        assert rows[0]["best_run"] is None

    def test_unknown_dimension_rejected(self):
        with pytest.raises(ValueError):
            sim_results.breakdown([], by="provider-only")


class TestInstrumentBreakdown:
    """Which symbol a stored run is filed under, and why it is one row.

    A rule run's setup names its instrument; an LLM run trades whatever basket
    it was handed. Both have to land somewhere honest without a whole-run
    return ever being counted under more than one symbol.
    """

    def _run(self, symbols=("AAPL",), ticker=None, return_pct=1.0, **extra):
        config = {
            "provider": "openai", "model": "gpt-a", "personality": "momentum",
            "symbols": list(symbols),
        }
        if ticker is not None:
            config["rule_based"] = True
            config["rule_config"] = {"ticker": ticker}
        config.update(extra)
        return {
            "run_id": "r1",
            "dataset": "ds1",
            "config_summary": config,
            "summary": {"return_pct": return_pct, "profit_efficiency": 0.5},
        }

    def test_a_single_symbol_run_is_that_instrument(self):
        assert sim_results.instrument_key(self._run(symbols=["GOOGL"])) == "GOOGL"

    def test_a_rule_setup_names_the_instrument(self):
        assert sim_results.instrument_key(
            self._run(symbols=["AAPL"], ticker="INTC")
        ) == "INTC"

    def test_a_legacy_rule_run_is_filed_under_what_it_traded(self):
        """Runs stored before the Simulate tab narrowed a rule experiment's
        symbols carry the whole basket they were handed. The setup's ticker is
        what that run actually traded, and reading the basket instead would
        file an ordinary AAPL run under "several"."""
        record = self._run(symbols=["NVDA", "AAPL", "GOOGL", "INTC"], ticker="AAPL")
        assert sim_results.instrument_key(record) == "AAPL"

    def test_a_basket_run_gets_its_own_row(self):
        """Only an LLM agent can produce one. Its return is a fact about the
        basket, so attributing it to each symbol would count it four times."""
        record = self._run(symbols=["NVDA", "AAPL", "GOOGL", "INTC"])
        assert sim_results.instrument_key(record) == sim_results.MULTI_INSTRUMENT

    def test_a_record_without_symbols_is_not_lost(self):
        assert sim_results.instrument_key({}) == sim_results.UNKNOWN_INSTRUMENT

    def test_symbols_are_normalised(self):
        assert sim_results.instrument_key(self._run(symbols=[" spy "])) == "SPY"
        assert sim_results.instrument_key(
            self._run(symbols=["AAPL"], ticker=" intc ")
        ) == "INTC"

    def test_runs_group_by_instrument(self):
        rows = sim_results.breakdown(
            [
                self._run(symbols=["AAPL"], return_pct=2.0),
                self._run(symbols=["AAPL"], return_pct=0.0),
                self._run(symbols=["INTC"], return_pct=5.0),
            ],
            by="instrument",
        )
        by_group = {r["group"]: r for r in rows}
        assert set(by_group) == {"AAPL", "INTC"}
        assert by_group["AAPL"]["runs"] == 2
        assert by_group["AAPL"]["avg_return_pct"] == 1.0
        assert by_group["INTC"]["runs"] == 1
        assert by_group["INTC"]["best_return_pct"] == 5.0

    def test_every_run_lands_in_exactly_one_group(self):
        """What keeps the instrument rows readable beside the other three:
        the columns mean the same thing in every view because a run is counted
        once, whatever it was handed."""
        runs = [
            self._run(symbols=["AAPL"]),
            self._run(symbols=["AAPL"], ticker="GOOGL"),
            self._run(symbols=["NVDA", "META"]),
            self._run(symbols=[]),
        ]
        rows = sim_results.breakdown(runs, by="instrument")
        assert sum(r["runs"] for r in rows) == len(runs)

    def test_the_winning_run_names_its_instrument(self):
        """In a breakdown along any other dimension the instrument is invisible,
        which is what the best-run tooltip is for."""
        rows = sim_results.breakdown(
            [
                self._run(symbols=["AAPL"], return_pct=2.0),
                self._run(symbols=["INTC"], return_pct=9.0),
            ],
            by="model",
        )
        assert rows[0]["best_run"]["instrument"] == "INTC"

    def test_instrument_is_an_offered_dimension(self):
        assert "instrument" in sim_results.BREAKDOWN_DIMENSIONS
        assert set(sim_app._BREAKDOWN_DIMENSIONS.values()) <= set(
            sim_results.BREAKDOWN_DIMENSIONS
        )
        assert "instrument" in sim_app._BREAKDOWN_DIMENSIONS.values()


class TestDecisionTrigger:
    """What fired a fill, named in two or three words.

    The strings below are copied verbatim from `apple_trader`,
    `apple_trader2` and the tactic executor -- the whole classifier is a bet on
    those openings, so the test has to be a sample of the real thing rather
    than a paraphrase. If a trader is reworded, this fails and the tag is
    updated; the failure mode in the app meanwhile is a missing headline, never
    a wrong one.
    """

    def _reason(self, text):
        return {"action": "sell", "reasoning": text}

    def test_a_trailing_stop_is_named(self):
        assert sim_results.decision_trigger(self._reason(
            "Trailing stop: $89.46 is -0.52% off the $89.93 high since the $89.77 "
            "entry, past the 0.50% give-back. Selling at market (-0.34%)."
        )) == "Trailing stop"

    def test_a_hard_stop_is_named(self):
        assert sim_results.decision_trigger(self._reason(
            "Stop: $88.10 is below the $88.50 floor 0.50% under the $88.94 entry. "
            "Selling at market (-0.94%)."
        )) == "Stop loss"

    def test_the_two_stops_are_not_confused(self):
        """They are different exits -- one gives back from a peak, the other
        breaks a floor under the entry -- and 'Stop:' is a prefix of neither."""
        trailing = sim_results.decision_trigger(self._reason("Trailing stop: ..."))
        hard = sim_results.decision_trigger(self._reason("Stop: ..."))
        assert trailing != hard

    def test_a_model_entry_is_named_as_a_forecast(self):
        assert sim_results.decision_trigger(self._reason(
            "Momentum regime is still balanced on this bar (momentum +0.73, and it "
            "has held 20 bars), but the forecast puts it at 9% (>= 5%) to turn "
            "positive on the next bar and stay there."
        )) == "ML forecast — anticipated turn"

    def test_a_persistence_entry_is_named_as_a_probability(self):
        assert sim_results.decision_trigger(self._reason(
            "Momentum regime turned balanced -> positive on this bar (momentum +1.11, "
            "the old regime had held 6 bars), and the persistence model puts it at "
            "24% (>= 22%) to hold."
        )) == "ML probability — change holds"

    def test_a_forecast_reversal_exit_is_named(self):
        assert sim_results.decision_trigger(self._reason(
            "Forecast reversal: the momentum regime is still positive (momentum "
            "+0.42, 6 bars in), but the model puts it at 41% (>= 30%) to flip "
            "negative inside the forecast horizon."
        )) == "ML forecast — reversal"

    def test_the_day_range_levels_are_named_apart(self):
        buy = sim_results.decision_trigger(self._reason(
            "The bar traded down to $341.16, at or through the $341.42 buy level — "
            "0.75 average daily ranges ($6.34 each) below the $346.17 high."
        ))
        sell = sim_results.decision_trigger(self._reason(
            "Target: the bar traded up to $88.83, at or through the $88.63 sell "
            "level (H − 0.15 × ADR). Selling at market (+2.90%)."
        ))
        assert buy == "Predicted-range buy level"
        assert sell == "Predicted-range sell level"

    def test_the_delta_momentum_exits_are_named(self):
        assert sim_results.decision_trigger(self._reason(
            "Momentum floor: the score is -1.20 bps/min, below -2 × θ (-0.90)."
        )) == "Momentum floor"
        assert sim_results.decision_trigger(self._reason(
            "Model exit: the regime is positive and the model puts the next 15 bars "
            "at -0.44 bps/min, at or past the -0.3 sell threshold."
        )) == "ML forecast — move over"

    def test_the_closing_bell_is_not_a_signal(self):
        """Worth its own tag: a flatten at the close says nothing about the
        strategy, and reading it as an exit signal would misjudge the run."""
        assert sim_results.decision_trigger(self._reason(
            "Session ends in 5 min. Momentum, regimes and every model feature are "
            "intraday, so the position is flattened rather than carried overnight."
        )) == "Flattened at the close"

    def test_a_rule_agent_fill_is_named(self):
        assert sim_results.decision_trigger(self._reason(
            "Rule 2 — Buy the dip below the predicted high, while flat: buy 50% of "
            "cash when pos.shares <= 0 (0 now) and dayrange.pred_high_dip_adr >= "
            "-0.75 (0.528 now)."
        )) == "Rule fired"

    def test_an_llm_tactic_fill_is_named(self):
        assert sim_results.decision_trigger(self._reason(
            "Tactics triggered (last_price below 304.4569 held 60s): Reassess "
            "sustained lower 2σ stretch."
        )) == "Armed tactic"

    def test_an_unrecognised_reason_gets_no_tag(self):
        """The safe failure: a reworded trader loses its headline and keeps its
        prose. A wrong tag would be worse than none."""
        assert sim_results.decision_trigger(self._reason(
            "Exit the full position now. Trajectory overrides the volume levels."
        )) == ""

    def test_a_decision_without_a_reason_gets_no_tag(self):
        assert sim_results.decision_trigger({"action": "buy"}) == ""
        assert sim_results.decision_trigger({}) == ""


class TestDecisionHover:
    """The marker tooltip. What was traded, and the condition that fired."""

    def _fill(self, **over):
        return {
            "ts": "2026-08-31T18:14:00+00:00", "symbol": "INTC", "action": "buy",
            "filled_quantity": 556.9882, "price": 89.76850128173828,
            "status": "filled",
            "reasoning": "Trailing stop: $89.46 is -0.52% off the $89.93 high.",
            **over,
        }

    def test_it_leads_with_the_trade(self):
        card = sim_app._decision_hover(self._fill())
        assert "<b>BUY</b>" in card
        assert "$89.77" in card
        assert "2026-08-31 18:14" in card

    def test_fractional_shares_keep_their_precision(self):
        """The fraction is what the position-sizing rule chose, not noise."""
        assert "556.9882 sh" in sim_app._decision_hover(self._fill())
        assert "3.2849 sh" in sim_app._decision_hover(
            self._fill(filled_quantity=3.2848810798042805)
        )

    def test_a_whole_share_count_is_not_padded(self):
        assert "10 sh" in sim_app._decision_hover(self._fill(filled_quantity=10.0))

    def test_it_carries_the_trigger_and_the_reason(self):
        card = sim_app._decision_hover(self._fill())
        assert "Trailing stop" in card
        assert "-0.52% off the $89.93 high" in card.replace("<br>", " ")

    def test_the_reason_is_wrapped(self):
        """Plotly sizes a hover label to its longest line, so an unwrapped
        400-character reason is a tooltip wider than the chart."""
        card = sim_app._decision_hover(self._fill(reasoning="word " * 200))
        assert all(
            len(line) <= sim_app._HOVER_WIDTH + 10 for line in card.split("<br>")
        )

    def test_a_very_long_reason_is_capped(self):
        """An LLM can justify a fill at length, and a tooltip taller than the
        plot covers the tape it is explaining."""
        card = sim_app._decision_hover(self._fill(reasoning="word " * 500))
        assert card.count("<br>") <= sim_app._HOVER_MAX_LINES + 4
        assert "Decisions below" in card

    def test_markup_in_a_reason_cannot_break_the_card(self):
        card = sim_app._decision_hover(
            self._fill(reasoning="proba >= 0.05 <b>not bold</b>")
        )
        assert "<b>not bold</b>" not in card
        assert "&gt;=" in card

    def test_a_fill_with_no_reason_says_so(self):
        card = sim_app._decision_hover(self._fill(reasoning=""))
        assert "No reason recorded" in card

    def test_an_unfilled_quantity_falls_back_to_what_was_asked(self):
        card = sim_app._decision_hover(
            self._fill(filled_quantity=None, requested_quantity=12.5)
        )
        assert "12.5 sh" in card


class TestMLModelBreakdown:
    """Which saved model produced a run's decisions.

    A different axis from `model`, which is the LLM (or the rule set's
    signature). Three kinds of run answer it three ways, and the point of the
    tests is that each lands somewhere honest: one model, a decoded set of
    them, or -- for an LLM, which loads none -- its provider.
    """

    def _llm(self, provider="openai", **extra):
        return {
            "run_id": "r1", "dataset": "ds1",
            "config_summary": {
                "provider": provider, "model": "gpt-a", "personality": "momentum",
                "symbols": ["AAPL"], **extra,
            },
            "summary": {"return_pct": 1.0, "profit_efficiency": 0.5},
        }

    def _apple(self, model_key="nbeats", return_pct=1.0):
        return {
            "run_id": "r1", "dataset": "ds1",
            "config_summary": {
                "provider": "rules", "model": "sig", "personality": APPLE_TRADER_KEY,
                "symbols": ["AAPL"], "rule_based": True,
                "rule_config": {"model_key": model_key, "ticker": "AAPL"},
            },
            "summary": {"return_pct": return_pct, "profit_efficiency": 0.5},
        }

    def _apple2(self, fields=(), return_pct=1.0):
        """An Apple Trader 2 run whose one rule reads the given signal fields."""
        rules = {"items": [{
            "action": "buy", "size_mode": "pct", "size": 50.0,
            "conditions": [{"field": f, "op": "above", "value": 0.0} for f in fields],
            "join": "all", "enabled": True,
        }]}
        return {
            "run_id": "r1", "dataset": "ds1",
            "config_summary": {
                "provider": "rules", "model": "sig", "personality": APPLE_TRADER2_KEY,
                "symbols": ["AAPL"], "rule_based": True,
                "rule_config": {"rules": rules, "ticker": "AAPL"},
            },
            "summary": {"return_pct": return_pct, "profit_efficiency": 0.5},
        }

    def test_apple_trader_is_the_model_it_names(self):
        assert sim_results.ml_model_key(self._apple("dayrange")) == "dayrange"

    def test_an_llm_run_is_grouped_by_provider(self):
        assert sim_results.ml_model_key(self._llm("anthropic")) == (
            sim_results.LLM_MODEL_PREFIX + "anthropic"
        )

    def test_two_llm_models_from_one_provider_share_a_row(self):
        """The whole point of grouping LLMs by provider on this axis: the rows
        compare approaches, and one row per LLM model would swamp the four ML
        ones."""
        runs = [self._llm("openai"), self._llm("openai")]
        runs[1]["config_summary"]["model"] = "gpt-b"
        rows = sim_results.breakdown(runs, by="ml_model")
        assert len(rows) == 1 and rows[0]["runs"] == 2

    def test_apple_trader_2_reads_its_conditions(self):
        assert sim_results.ml_model_key(
            self._apple2(["dayrange.pred_high_dip_adr"])
        ) == "dayrange"

    def test_a_rule_set_naming_several_models_is_one_row(self):
        """A whole-run return counted under two models would double it, so the
        set is the group -- joined in registry order, so the same combination
        is always the same key."""
        key = sim_results.ml_model_key(
            self._apple2(["dayrange.pred_high_dip_adr", "persistence.proba"])
        )
        assert key == "persistence" + sim_results.ML_MODEL_JOIN + "dayrange"
        assert sim_results.ml_model_key(
            self._apple2(["persistence.proba", "dayrange.pred_high_dip_adr"])
        ) == key

    def test_a_tape_only_rule_set_is_its_own_answer(self):
        """Not a missing value: a set on price and the position is a complete
        strategy, and it is the row that says what the models are worth."""
        assert sim_results.ml_model_key(
            self._apple2(["bar.price", "pos.shares"])
        ) == sim_results.NO_ML_MODEL

    def test_an_undecodable_rule_set_is_not_lost(self):
        record = self._apple2([])
        record["config_summary"]["rule_config"]["rules"] = {"items": "not a list"}
        assert sim_results.ml_model_key(record) == sim_results.UNKNOWN_INSTRUMENT

    def test_every_run_lands_in_exactly_one_group(self):
        runs = [
            self._llm("openai"), self._llm("gemini"),
            self._apple("nbeats"), self._apple("persistence"),
            self._apple2(["persistence.proba"]), self._apple2(["bar.price"]),
        ]
        rows = sim_results.breakdown(runs, by="ml_model")
        assert sum(r["runs"] for r in rows) == len(runs)

    def test_the_winning_run_names_its_ml_model(self):
        rows = sim_results.breakdown(
            [self._apple("nbeats", return_pct=1.0),
             self._apple("dayrange", return_pct=9.0)],
            by="agent",
        )
        assert rows[0]["best_run"]["ml_model"] == "dayrange"


class TestMLModelLabels:
    """The row labels. Keys are stored; labels are rendered, so a model renamed
    in `apple_models` moves its row without touching stored runs."""

    def test_a_model_key_becomes_its_registry_name(self):
        assert sim_app._ml_model_label("nbeats") == "N-BEATS forecast"
        assert sim_app._ml_model_label("persistence") == "Persistence classifier"

    def test_a_provider_is_marked_as_an_llm(self):
        assert sim_app._ml_model_label("llm:openai") == "openai (LLM)"

    def test_a_combination_joins_both_names(self):
        assert sim_app._ml_model_label("persistence+dayrange") == (
            "Persistence classifier + Day-range forecast"
        )

    def test_sentinels_are_shown_as_they_are(self):
        for key in (sim_results.NO_ML_MODEL, sim_results.UNKNOWN_INSTRUMENT):
            assert sim_app._ml_model_label(key) == key

    def test_an_unknown_key_is_never_shown_as_a_real_model(self):
        """`apple_models.get` falls back to the default model for an unknown
        key so a stored run still replays. Printing that model's name over a
        key that is not one would be a quiet lie about which model ran."""
        assert sim_app._ml_model_label("") == sim_results.UNKNOWN_INSTRUMENT
        assert sim_app._ml_model_label("retired_model") == "retired_model"

    def test_labels_stay_short_enough_for_the_table(self):
        """The breakdown table does not wrap, so every label a real run can
        produce has to fit a first column."""
        for key in list(apple_models.MODELS) + ["llm:anthropic", "persistence+dayrange"]:
            assert len(sim_app._ml_model_label(key)) <= 48


class TestTopRuns:
    def _run(self, run_id="r1", return_pct=1.0, efficiency=0.5, **config):
        return {
            "run_id": run_id,
            "dataset": config.pop("dataset", "ds1"),
            "config_summary": {
                "provider": "openai", "model": "gpt-a", "personality": "momentum",
                **config,
            },
            "summary": {"return_pct": return_pct, "profit_efficiency": efficiency},
        }

    def test_ranks_by_the_chosen_metric(self):
        runs = [
            self._run(run_id="r1", return_pct=1.0, efficiency=0.9),
            self._run(run_id="r2", return_pct=5.0, efficiency=0.1),
            self._run(run_id="r3", return_pct=3.0, efficiency=0.5),
        ]
        by_return = sim_results.top_runs(runs, by="return_pct")
        assert [r["run_id"] for r in by_return] == ["r2", "r3", "r1"]
        by_efficiency = sim_results.top_runs(runs, by="profit_efficiency")
        assert [r["run_id"] for r in by_efficiency] == ["r1", "r3", "r2"]

    def test_limit_and_identity(self):
        runs = [self._run(run_id=f"r{i}", return_pct=float(i)) for i in range(5)]
        top = sim_results.top_runs(runs, by="return_pct", limit=3)
        assert [r["run_id"] for r in top] == ["r4", "r3", "r2"]
        assert top[0]["model"] == "openai/gpt-a"
        assert top[0]["personality"] == "momentum"
        assert top[0]["dataset"] == "ds1"
        # Both metrics travel with the row, whichever one it was ranked on.
        assert top[0]["profit_efficiency"] == 0.5

    def test_runs_without_the_metric_drop_out(self):
        runs = [self._run(run_id="r1", efficiency=None), self._run(run_id="r2")]
        assert [r["run_id"] for r in sim_results.top_runs(runs, by="profit_efficiency")] == ["r2"]
        # ...but still rank on a metric they do have.
        assert len(sim_results.top_runs(runs, by="return_pct")) == 2

    def test_unknown_metric_rejected(self):
        with pytest.raises(ValueError):
            sim_results.top_runs([], by="judge_score")


class TestRunFilters:
    def _run(self, model="gpt-a", dataset="ds1", provider="openai",
             personality="momentum", symbols=("AAPL",)):
        return {
            "dataset": dataset,
            "config_summary": {
                "provider": provider, "model": model,
                "personality": personality, "symbols": list(symbols),
            },
        }

    def _rule_run(self, ticker="AAPL", model_key="persistence", dataset="ds1"):
        """A rule run, whose instrument and ML model come out of its setup
        rather than out of the LLM fields."""
        return {
            "dataset": dataset,
            "config_summary": {
                "provider": "rules", "model": f"rules/{model_key}",
                "personality": "apple_trader", "rule_based": True,
                "symbols": [ticker],
                "rule_config": {"ticker": ticker, "model_key": model_key},
            },
        }

    def _runs(self):
        return [
            self._run(model="gpt-a", dataset="ds1"),
            self._run(model="gpt-b", dataset="ds1"),
            self._run(model="gpt-a", dataset="ds2"),
        ]

    def test_options_are_the_distinct_keys(self):
        options = sim_results.filter_options(self._runs() + [self._run(dataset="")])
        assert options["datasets"] == ["ds1", "ds2", sim_results.NO_DATASET]
        assert options["models"] == ["openai/gpt-a", "openai/gpt-b"]

    def test_options_cover_the_new_dimensions(self):
        runs = [
            self._run(personality="momentum", symbols=["AAPL"]),
            self._run(personality="contrarian", symbols=["INTC"]),
            self._rule_run(ticker="GOOGL", model_key="dayrange"),
        ]
        options = sim_results.filter_options(runs)
        assert options["agents"] == ["apple_trader", "contrarian", "momentum"]
        assert options["instruments"] == ["AAPL", "GOOGL", "INTC"]
        # LLM runs group by provider on this axis; the rule run names its model.
        assert options["ml_models"] == [
            "dayrange", f"{sim_results.LLM_MODEL_PREFIX}openai",
        ]

    def test_sentinel_options_sort_last(self):
        runs = [
            self._run(dataset="", symbols=["AAPL", "INTC"]),
            self._run(dataset="zz", symbols=["AAPL"]),
        ]
        options = sim_results.filter_options(runs)
        assert options["datasets"] == ["zz", sim_results.NO_DATASET]
        assert options["instruments"] == ["AAPL", sim_results.MULTI_INSTRUMENT]

    def test_empty_filters_keep_everything(self):
        runs = self._runs()
        assert sim_results.filter_runs(runs) == runs
        assert sim_results.filter_runs(runs, datasets=[], models=[]) == runs
        assert sim_results.filter_runs(
            runs, agents=[], instruments=[], ml_models=[]
        ) == runs

    def test_filters_combine(self):
        runs = self._runs()
        assert len(sim_results.filter_runs(runs, datasets=["ds1"])) == 2
        assert len(sim_results.filter_runs(runs, models=["openai/gpt-a"])) == 2
        both = sim_results.filter_runs(runs, datasets=["ds1"], models=["openai/gpt-a"])
        assert both == [runs[0]]

    def test_new_filters_select_and_intersect(self):
        aapl = self._rule_run(ticker="AAPL", model_key="persistence")
        intc = self._rule_run(ticker="INTC", model_key="persistence")
        googl = self._rule_run(ticker="GOOGL", model_key="dayrange")
        llm = self._run(personality="momentum", symbols=["AAPL"])
        runs = [aapl, intc, googl, llm]

        assert sim_results.filter_runs(runs, instruments=["AAPL"]) == [aapl, llm]
        assert sim_results.filter_runs(runs, ml_models=["persistence"]) == [aapl, intc]
        assert sim_results.filter_runs(runs, agents=["momentum"]) == [llm]
        # Dimensions are AND-ed: narrowing one never widens another.
        assert sim_results.filter_runs(
            runs, instruments=["AAPL"], ml_models=["persistence"]
        ) == [aapl]
        assert sim_results.filter_runs(
            runs, instruments=["AAPL"], agents=["momentum"]
        ) == [llm]
        assert sim_results.filter_runs(
            runs, instruments=["INTC"], ml_models=["dayrange"]
        ) == []

    def test_filter_keys_match_breakdown_groups(self):
        runs = self._runs()
        by_model = {r["group"] for r in sim_results.breakdown(runs, by="model")}
        assert by_model == set(sim_results.filter_options(runs)["models"])
        by_dataset = {r["group"] for r in sim_results.breakdown(runs, by="dataset")}
        assert by_dataset == set(sim_results.filter_options(runs)["datasets"])

    def test_every_filter_agrees_with_its_breakdown(self):
        """A selection and the row it narrows to have to be the same key, or
        picking a row's value in the filter would empty the page."""
        runs = [
            self._run(personality="momentum", symbols=["AAPL"]),
            self._run(personality="contrarian", symbols=["INTC"], dataset="ds2"),
            self._rule_run(ticker="GOOGL", model_key="dayrange"),
        ]
        options = sim_results.filter_options(runs)
        pairs = {
            "agents": "agent", "instruments": "instrument",
            "ml_models": "ml_model", "models": "model", "datasets": "dataset",
        }
        for option_key, dimension in pairs.items():
            groups = {r["group"] for r in sim_results.breakdown(runs, by=dimension)}
            assert groups == set(options[option_key]), dimension
            for value in options[option_key]:
                assert sim_results.filter_runs(runs, **{option_key: [value]}), (
                    dimension, value,
                )


class TestPriorRuns:
    def _run(self, run_id="r1", personality="momentum", provider="openai",
             model="gpt-a", dataset="ds1", agent_log=None, **fields):
        record = {
            "run_id": run_id,
            "dataset": dataset,
            "config_summary": {
                "personality": personality, "provider": provider, "model": model,
                "days": [DAY.isoformat()],
            },
            "summary": {"return_pct": 1.0, "profit_efficiency": 0.5},
            "cycles_run": 10,
            "agent_log": agent_log if agent_log is not None else [{"type": "cycle"}],
        }
        record.update(fields)
        return record

    def test_counts_only_error_log_entries(self):
        record = self._run(agent_log=[
            {"type": "cycle"},
            {"type": "error", "text": "LLM call failed: Error code: 429"},
            {"type": "error", "text": "Agent cycle failed: boom"},
        ])
        assert sim_results.cycle_error_count(record) == 2
        assert sim_results.cycle_error_count(self._run()) == 0
        assert sim_results.cycle_error_count({}) == 0

    def test_matches_only_the_full_combination(self):
        runs = [
            self._run(run_id="match"),
            self._run(run_id="other-agent", personality="breakout"),
            self._run(run_id="other-model", model="gpt-b"),
            self._run(run_id="other-provider", provider="anthropic"),
            self._run(run_id="other-dataset", dataset="ds2"),
        ]
        prior = sim_results.find_prior_runs(runs, "momentum", "openai", "gpt-a", "ds1")
        assert [r["run_id"] for r in prior["clean"]] == ["match"]
        assert prior["degraded"] == []

    def test_llm_errors_make_a_run_degraded_not_clean(self):
        runs = [
            self._run(run_id="clean"),
            self._run(run_id="llm-error", agent_log=[
                {"type": "error", "text": "LLM call failed: Error code: 429"}]),
            self._run(run_id="engine-error", error="no simulated tape price"),
            self._run(run_id="interrupted", interrupted=True),
            self._run(run_id="no-cycles", cycles_run=0),
        ]
        prior = sim_results.find_prior_runs(runs, "momentum", "openai", "gpt-a", "ds1")
        assert [r["run_id"] for r in prior["clean"]] == ["clean"]
        assert [r["run_id"] for r in prior["degraded"]] == [
            "llm-error", "engine-error", "interrupted", "no-cycles"
        ]

    def test_no_prior_runs(self):
        prior = sim_results.find_prior_runs([], "momentum", "openai", "gpt-a", "ds1")
        assert prior == {"clean": [], "degraded": []}

    def test_store_signature_tracks_changes(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sim_results, "RUNS_DIR", tmp_path / "runs")
        assert sim_results.store_signature() == ()
        (tmp_path / "runs").mkdir()
        (tmp_path / "runs" / "a.json").write_text("{}")
        first = sim_results.store_signature()
        assert len(first) == 1
        (tmp_path / "runs" / "b.json").write_text("{}")
        assert sim_results.store_signature() != first

    def test_delete_all_runs_empties_the_store(self, tmp_path, monkeypatch):
        runs_dir = tmp_path / "runs"
        monkeypatch.setattr(sim_results, "RUNS_DIR", runs_dir)
        # No store yet -- nothing to delete, and no directory is created.
        assert sim_results.delete_all_runs() == 0
        runs_dir.mkdir()
        for name in ("a.json", "b.json"):
            (runs_dir / name).write_text("{}")
        (runs_dir / "notes.txt").write_text("keep me")

        assert sim_results.delete_all_runs() == 2
        assert sim_results.list_runs() == []
        assert sim_results.store_signature() == ()
        assert (runs_dir / "notes.txt").exists()


class TestRunnerJudgeSelection:
    """The judge LLM is picked independently of the agent's, falling back to it."""

    def _run_with(self, monkeypatch, config: dict) -> dict:
        from simlab import runner

        captured: dict = {}

        def fake_judge_run(decisions, summary, prompt, market, provider, api_key,
                           model, progress=lambda msg: None):
            captured.update(provider=provider, api_key=api_key, model=model)
            return {}

        class FakeEngine:
            def __init__(self, *a, **kw):
                pass

            def run(self):
                return SimpleNamespace(cycles_run=1, decisions=[])

        monkeypatch.setattr(
            runner.experiments, "get_experiment",
            lambda _id: {"config": config, "dataset": "ds1"},
        )
        monkeypatch.setattr(runner, "SimMarket", lambda *a, **kw: object())
        monkeypatch.setattr(runner, "SimulationEngine", FakeEngine)
        monkeypatch.setattr(runner.sim_results, "summarize_run", lambda *a, **kw: {})
        monkeypatch.setattr(runner.sim_results, "save_run", lambda *a, **kw: {"run_id": "r1"})
        monkeypatch.setattr(runner.sim_prompts, "get_prompt", lambda _p: "brief")
        monkeypatch.setattr(runner.sim_judge, "judge_run", fake_judge_run)
        runner.run_experiment("exp-1")
        return captured

    @staticmethod
    def _config(**overrides) -> dict:
        config = {
            "personality": "momentum", "provider": "openai", "model": "gpt-test",
            "api_key": "sk-agent", "symbols": ["TEST"], "days": [DAY.isoformat()],
            "starting_cash": 100_000.0, "cycle_minutes": 5,
            "max_cycles_per_day": 40, "system_prompt_override": None,
            "run_judge": True,
        }
        config.update(overrides)
        return config

    def test_separate_judge_llm_is_used(self, monkeypatch):
        captured = self._run_with(monkeypatch, self._config(
            judge_provider="anthropic", judge_model="claude-sonnet-5",
            judge_api_key="sk-judge",
        ))
        assert captured == {
            "provider": "anthropic", "model": "claude-sonnet-5", "api_key": "sk-judge",
        }

    def test_falls_back_to_the_agent_llm(self, monkeypatch):
        """Experiments queued before the judge picker existed still judge."""
        captured = self._run_with(monkeypatch, self._config())
        assert captured == {
            "provider": "openai", "model": "gpt-test", "api_key": "sk-agent",
        }

    def test_the_rule_agent_is_never_judged(self, monkeypatch):
        """Profit metrics only. The judge grades stated reasoning against the
        tape; the rule agent states none of its own, so it is skipped even when
        the experiment record explicitly asks for a judge."""
        captured = self._run_with(monkeypatch, self._config(
            personality=APPLE_TRADER_KEY, provider=RULE_PROVIDER,
            model=config_signature(), api_key="", run_judge=True,
        ))
        assert captured == {}


class TestVolumeFetchPatches:
    """The volume tools' yfinance fetches must serve stored bars clipped to the
    simulated clock -- live they would return wall-clock-today data (a
    different day entirely inside a simulation) or a full day including the
    simulated future."""

    def test_intraday_volume_bars_clip_to_sim_now(self, store):
        from agent_stonks import historical

        with simulation_context(SimMarket(["TEST"], [DAY])):
            clock.set_simulated(OPEN_UTC + timedelta(minutes=30))
            bars = historical.fetch_intraday_volume_bars("TEST")
            # Bar stamped T is visible once T+60s <= now: bars 0..29.
            assert len(bars) == 30
            assert bars[-1]["t"] == (OPEN_UTC + timedelta(minutes=29)).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )

    def test_dated_fetch_of_sim_day_clips_to_sim_now(self, store):
        from agent_stonks import historical

        with simulation_context(SimMarket(["TEST"], [DAY])):
            clock.set_simulated(OPEN_UTC + timedelta(minutes=10))
            bars = historical.fetch_intraday_bars_for_date("TEST", DAY.isoformat())
            assert len(bars) == 10  # not the stored day's full 60

    def test_dated_fetch_of_unstored_day_is_empty(self, store):
        from agent_stonks import historical

        with simulation_context(SimMarket(["TEST"], [DAY])):
            clock.set_simulated(OPEN_UTC)
            assert historical.fetch_intraday_bars_for_date("TEST", "2026-06-12") == []

    def test_dated_fetch_rejects_bad_format(self, store):
        from agent_stonks import historical

        with simulation_context(SimMarket(["TEST"], [DAY])):
            clock.set_simulated(OPEN_UTC)
            with pytest.raises(ValueError):
                historical.fetch_intraday_bars_for_date("TEST", "yesterday")

    def test_daily_volume_bars_shape_and_clip(self, store):
        from agent_stonks import historical

        with simulation_context(SimMarket(["TEST"], [DAY])):
            clock.set_simulated(OPEN_UTC + timedelta(minutes=30))
            daily = historical.fetch_daily_volume_bars("TEST")
            assert daily, "expected prior daily bars plus today's partial"
            assert set(daily[0]) == {"t", "v"}
            # Last row is the sim day's partial; nothing beyond the sim day.
            assert daily[-1]["t"] == DAY.isoformat()
            assert all(row["t"] <= DAY.isoformat() for row in daily)


class TestPriceChartSessions:
    """The Results price chart across more than one day.

    `_price_chart` is where the two session helpers are actually applied, so
    these pin the wiring rather than the arithmetic (`tests/test_charts.py`
    covers that).
    """

    def bars(self, days=("2024-01-15", "2024-01-16")):
        out = []
        for day in days:
            idx = pd.date_range(
                f"{day} 04:00", f"{day} 19:59", freq="1min", tz="America/New_York"
            )
            out += [
                {"t": ts.tz_convert("UTC").isoformat(), "o": 100.0, "h": 100.5,
                 "l": 99.5, "c": 100.0, "v": 1000.0}
                for ts in idx
            ]
        return out

    def test_the_night_between_two_days_is_removed_from_the_axis(self):
        from simlab.app import _price_chart

        fig = _price_chart("AAPL", self.bars(), [])
        assert len(fig.layout.xaxis.rangebreaks) == 1

    def test_session_boundaries_are_drawn(self):
        from agent_stonks.charts import SESSION_MARKER_COLOR
        from simlab.app import _price_chart

        fig = _price_chart("AAPL", self.bars(), [])
        marks = [
            s for s in fig.layout.shapes if s.line.color == SESSION_MARKER_COLOR
        ]
        assert len(marks) == 4  # open + close, two days

    def test_the_breaks_survive_the_shared_chart_layout(self):
        """`_chart_layout` sets `xaxis=dict(gridcolor=...)` afterwards, which
        merges rather than replaces -- pinned because a replace would silently
        put the empty nights back."""
        from simlab.app import _price_chart

        fig = _price_chart("AAPL", self.bars(), [])
        assert fig.layout.xaxis.gridcolor
        assert fig.layout.xaxis.rangebreaks

    def test_no_bars_is_still_a_figure(self):
        from simlab.app import _price_chart

        assert _price_chart("AAPL", [], []) is not None


class TestRuleSetupSlots:
    """The Simulate tab's per-agent setup slots.

    A slot is an opaque id owning one set of widget keys, which is the whole
    point: removing the middle of three setups must leave the other two's
    inputs where they were, and indexing by position would shuffle them.
    """

    @pytest.fixture(autouse=True)
    def session(self, monkeypatch):
        state: dict = {}
        monkeypatch.setattr(sim_app.st, "session_state", state)
        return state

    def test_an_agent_starts_with_one_setup(self):
        assert len(sim_app._rule_slots(APPLE_TRADER_KEY)) == 1

    def test_adding_appends_a_new_distinct_slot(self):
        first = list(sim_app._rule_slots(APPLE_TRADER_KEY))
        sim_app._add_rule_slot(APPLE_TRADER_KEY)
        slots = sim_app._rule_slots(APPLE_TRADER_KEY)
        assert len(slots) == 2
        assert slots[0] == first[0]
        assert slots[0] != slots[1]

    def test_removing_the_middle_leaves_the_others_untouched(self):
        for _ in range(2):
            sim_app._add_rule_slot(APPLE_TRADER_KEY)
        first, middle, last = sim_app._rule_slots(APPLE_TRADER_KEY)
        sim_app._drop_rule_slot(APPLE_TRADER_KEY, middle)
        assert sim_app._rule_slots(APPLE_TRADER_KEY) == [first, last]

    def test_the_last_setup_cannot_be_removed(self):
        only = sim_app._rule_slots(APPLE_TRADER_KEY)[0]
        sim_app._drop_rule_slot(APPLE_TRADER_KEY, only)
        assert sim_app._rule_slots(APPLE_TRADER_KEY) == [only]

    def test_removing_an_unknown_slot_is_a_no_op(self):
        slots = list(sim_app._rule_slots(APPLE_TRADER_KEY))
        sim_app._add_rule_slot(APPLE_TRADER_KEY)
        sim_app._drop_rule_slot(APPLE_TRADER_KEY, "not-a-slot")
        assert sim_app._rule_slots(APPLE_TRADER_KEY)[0] == slots[0]

    def test_the_two_agents_keep_separate_slots(self):
        sim_app._add_rule_slot(APPLE_TRADER_KEY)
        assert len(sim_app._rule_slots(APPLE_TRADER_KEY)) == 2
        assert len(sim_app._rule_slots(APPLE_TRADER2_KEY)) == 1


class TestRuleCombinations:
    """Several setups of one agent are several experiments."""

    def setups(self, *configs):
        return {APPLE_TRADER_KEY: list(configs)}

    def test_one_combination_per_setup_per_dataset(self):
        setups = self.setups(
            AppleTraderConfig(trail_pct=0.35),
            AppleTraderConfig(trail_pct=0.50),
        )
        combos, _ = sim_app._rule_combinations(setups, ["ds1", "ds2"])
        assert len(combos) == 4
        assert {c[3] for c in combos} == {"ds1", "ds2"}

    def test_each_setup_gets_its_own_signature(self):
        setups = self.setups(
            AppleTraderConfig(trail_pct=0.35),
            AppleTraderConfig(trail_pct=0.50),
        )
        combos, _ = sim_app._rule_combinations(setups, ["ds1"])
        assert len({c[2] for c in combos}) == 2

    def test_a_combination_maps_back_to_the_settings_that_made_it(self):
        slow, fast = AppleTraderConfig(trail_pct=0.35), AppleTraderConfig(trail_pct=0.50)
        combos, by_combo = sim_app._rule_combinations(self.setups(slow, fast), ["ds1"])
        assert [by_combo[c].trail_pct for c in combos] == [0.35, 0.50]

    def test_two_identical_setups_are_one_experiment(self):
        """Identical settings are the same configuration; Results would file
        the two runs as one row whatever this did."""
        same = AppleTraderConfig(trail_pct=0.35)
        combos, _ = sim_app._rule_combinations(self.setups(same, same), ["ds1"])
        assert len(combos) == 1

    def test_setups_the_signature_cannot_tell_apart_are_one_experiment(self):
        """The closing flatten is in neither agent's signature.

        So two setups differing only in it are one configuration everywhere
        downstream -- the run record's `model` field, the already-tested check
        and the Results grouping are all that string. Queueing both would
        produce two runs shown as a single row.
        """
        combos, _ = sim_app._rule_combinations(
            self.setups(
                AppleTraderConfig(flatten_before_close_min=5),
                AppleTraderConfig(flatten_before_close_min=9),
            ),
            ["ds1"],
        )
        assert len(combos) == 1

    def test_setups_on_different_instruments_are_different_combinations(self):
        combos, _ = sim_app._rule_combinations(
            {APPLE_TRADER_KEY: [
                AppleTraderConfig(model_key="dayrange", ticker="AAPL"),
                AppleTraderConfig(model_key="dayrange", ticker="GOOGL"),
            ]},
            ["ds1"],
        )
        assert len(combos) == 2
        assert len({c[2] for c in combos}) == 2

    def test_both_agents_contribute_their_own_setups(self):
        combos, _ = sim_app._rule_combinations(
            {
                APPLE_TRADER_KEY: [
                    AppleTraderConfig(trail_pct=0.35),
                    AppleTraderConfig(trail_pct=0.50),
                ],
                APPLE_TRADER2_KEY: [AppleTrader2Config()],
            },
            ["ds1"],
        )
        assert len(combos) == 3
        assert {c[0] for c in combos} == {APPLE_TRADER_KEY, APPLE_TRADER2_KEY}

    def test_no_datasets_is_no_combinations(self):
        assert sim_app._rule_combinations(self.setups(AppleTraderConfig()), []) == ([], {})


class TestExperimentSymbols:
    """What each queued experiment is handed, which is now per combination.

    The whole point of the change: a rule run loads its own symbol's bars and
    no others, so the symbol boxes stop being a step that can only be got
    wrong, while the LLM basket -- which really is a configuration -- stays a
    choice.
    """

    def test_a_rule_agent_gets_only_its_own_instrument(self):
        assert sim_app._experiment_symbols(
            APPLE_TRADER_KEY,
            AppleTraderConfig(model_key="dayrange", ticker="GOOGL"),
            ["AAPL", "GOOGL", "SPY"],
        ) == ["GOOGL"]

    def test_the_basket_does_not_have_to_contain_it(self):
        """A rule run is no longer gated by the selection, so an instrument
        absent from the basket is still queued -- whether the *dataset* carries
        it is what `_rule_agents_missing_ticker` answers."""
        assert sim_app._experiment_symbols(
            APPLE_TRADER_KEY,
            AppleTraderConfig(model_key="dayrange", ticker="INTC"),
            ["AAPL"],
        ) == ["INTC"]

    def test_apple_trader_2_gets_its_own_instrument_too(self):
        assert sim_app._experiment_symbols(
            APPLE_TRADER2_KEY, AppleTrader2Config(ticker="SPY"), ["AAPL", "SPY"]
        ) == ["SPY"]

    def test_an_llm_agent_gets_the_whole_basket(self):
        assert sim_app._experiment_symbols(
            "momentum", None, ["AAPL", "GOOGL"]
        ) == ["AAPL", "GOOGL"]

    def test_the_basket_is_copied_not_aliased(self):
        """Every combination is submitted from one scope dict; a shared list
        would let one experiment record mutate another's."""
        basket = ["AAPL"]
        out = sim_app._experiment_symbols("momentum", None, basket)
        assert out == basket and out is not basket


class TestConfiguredRuleTickers:
    """Which instruments the Simulate tab will actually trade.

    The dataset scope is drawn above the agent setups but derives its symbols
    from them, so the answer is read out of the widgets' stored state rather
    than from the rendered setups. These pin that reading -- including the case
    that makes it safe, which is a slot whose instrument widget has not been
    drawn yet reporting exactly the value that widget will come up with.
    """

    @pytest.fixture(autouse=True)
    def session(self, monkeypatch):
        state: dict = {}
        monkeypatch.setattr(sim_app.st, "session_state", state)
        return state

    def test_an_unrendered_setup_reports_the_agent_default(self, session):
        tickers = sim_app._configured_rule_tickers([APPLE_TRADER_KEY])
        assert tickers == {
            APPLE_TRADER_KEY: [RULE_AGENTS[APPLE_TRADER_KEY].default_ticker]
        }

    def test_a_stored_instrument_wins_over_the_default(self, session):
        slot = sim_app._rule_slots(APPLE_TRADER_KEY)[0]
        session[f"sim_rule_{APPLE_TRADER_KEY}_{slot}_ticker"] = "GOOGL"
        assert sim_app._configured_rule_tickers([APPLE_TRADER_KEY]) == {
            APPLE_TRADER_KEY: ["GOOGL"]
        }

    def test_every_slot_contributes_its_own_instrument(self, session):
        sim_app._add_rule_slot(APPLE_TRADER_KEY)
        first, second = sim_app._rule_slots(APPLE_TRADER_KEY)
        session[f"sim_rule_{APPLE_TRADER_KEY}_{first}_ticker"] = "AAPL"
        session[f"sim_rule_{APPLE_TRADER_KEY}_{second}_ticker"] = "INTC"
        assert sim_app._configured_rule_tickers([APPLE_TRADER_KEY]) == {
            APPLE_TRADER_KEY: ["AAPL", "INTC"]
        }

    def test_two_slots_on_one_instrument_report_it_once(self, session):
        sim_app._add_rule_slot(APPLE_TRADER_KEY)
        for slot in sim_app._rule_slots(APPLE_TRADER_KEY):
            session[f"sim_rule_{APPLE_TRADER_KEY}_{slot}_ticker"] = "AAPL"
        assert sim_app._configured_rule_tickers([APPLE_TRADER_KEY]) == {
            APPLE_TRADER_KEY: ["AAPL"]
        }

    def test_a_free_text_instrument_is_normalised(self, session):
        """Apple Trader 2's box accepts anything typed into it, so what comes
        back has to be squared with the dataset's own spelling."""
        slot = sim_app._rule_slots(APPLE_TRADER2_KEY)[0]
        session[f"sim_rule_{APPLE_TRADER2_KEY}_{slot}_ticker"] = "  spy "
        assert sim_app._configured_rule_tickers([APPLE_TRADER2_KEY]) == {
            APPLE_TRADER2_KEY: ["SPY"]
        }

    def test_an_llm_agent_contributes_nothing(self, session):
        """It has no instrument of its own -- it trades the basket it is
        handed, which is the one thing on that row still worth choosing."""
        assert sim_app._configured_rule_tickers(["momentum"]) == {}

    def test_both_rule_agents_are_reported_separately(self, session):
        apple = sim_app._rule_slots(APPLE_TRADER_KEY)[0]
        apple2 = sim_app._rule_slots(APPLE_TRADER2_KEY)[0]
        session[f"sim_rule_{APPLE_TRADER_KEY}_{apple}_ticker"] = "AAPL"
        session[f"sim_rule_{APPLE_TRADER2_KEY}_{apple2}_ticker"] = "GOOGL"
        assert sim_app._configured_rule_tickers(
            [APPLE_TRADER_KEY, APPLE_TRADER2_KEY]
        ) == {APPLE_TRADER_KEY: ["AAPL"], APPLE_TRADER2_KEY: ["GOOGL"]}


class TestRuleSetupTickerCheck:
    """A setup whose symbol a dataset does not carry cannot trade at all.

    Checked against the *dataset*, not against what is ticked on the page:
    since a rule run is queued with its own symbol whichever boxes are
    selected, the only thing that can defeat it is the bars not being there.
    """

    def datasets(self, **symbols_by_name):
        return {
            name: SimpleNamespace(name=name, symbols=syms)
            for name, syms in symbols_by_name.items()
        }

    def test_a_setup_the_dataset_covers_is_not_reported(self):
        missing = sim_app._rule_agents_missing_ticker(
            {APPLE_TRADER_KEY: [AppleTraderConfig(ticker="AAPL")]},
            self.datasets(ds1=["AAPL", "SPY"]), ["ds1"],
        )
        assert missing == []

    def test_only_the_setup_that_is_missing_is_reported(self):
        missing = sim_app._rule_agents_missing_ticker(
            {APPLE_TRADER_KEY: [
                AppleTraderConfig(model_key="dayrange", ticker="AAPL"),
                AppleTraderConfig(model_key="dayrange", ticker="GOOGL"),
            ]},
            self.datasets(ds1=["AAPL"]), ["ds1"],
        )
        assert missing == [(APPLE_TRADER_KEY, "GOOGL", ["ds1"])]

    def test_one_symbol_is_reported_once_however_many_setups_use_it(self):
        missing = sim_app._rule_agents_missing_ticker(
            {APPLE_TRADER_KEY: [
                AppleTraderConfig(ticker="AAPL", trail_pct=0.35),
                AppleTraderConfig(ticker="AAPL", trail_pct=0.50),
            ]},
            self.datasets(ds1=["SPY"]), ["ds1"],
        )
        assert missing == [(APPLE_TRADER_KEY, "AAPL", ["ds1"])]

    def test_every_dataset_missing_the_symbol_is_named(self):
        missing = sim_app._rule_agents_missing_ticker(
            {APPLE_TRADER_KEY: [AppleTraderConfig(ticker="AAPL")]},
            self.datasets(ds1=["SPY"], ds2=["AAPL"], ds3=[]), ["ds1", "ds2", "ds3"],
        )
        assert missing == [(APPLE_TRADER_KEY, "AAPL", ["ds1", "ds3"])]
