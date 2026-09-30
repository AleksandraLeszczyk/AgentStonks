"""The HighLow mirror: the SIP history cache, the forecast's seams, and the bundle.

The tests that need neither the saved model nor the notebook pin the seams --
what the app has to get right to hand the model correct inputs: which sessions
the history cache fetches and when, a split being noticed, a short history
being refused.

The one that needs both pins the mirror itself: the notebook's own minute tape,
rolled up the way the live path rolls up SIP bars, must reproduce what the
notebook's `highlow` package predicts for the held-out week.
"""

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agent_stonks import apple_models

H = pytest.importorskip("agent_stonks.highlow_model")

NOTEBOOK = Path("/Users/aleksandra/Documents/playground/Code/FinNotebooks/HighLow_5m")

# What the notebook's own `highlow` package predicts from its own panel for the
# held-out week of 31 Aug 2026 plus one later day and one test-window day
# (`highlow.models.load_bundle(...).predict_prices`), per ticker's bundle:
# AAPL saved 2026-09-15, INTC 2026-09-21. (pred_high, pred_low, adr14_usd)
AAPL_FORECASTS = {
    "2026-03-02": (265.63175058123767, 259.76960582407247, 6.819571428571438),
    "2026-08-31": (321.235, 315.78224457142767, 5.9200214285714265),
    "2026-09-01": (317.8488540510188, 313.09447158886843, 6.009664285714284),
    "2026-09-02": (328.7640088736563, 323.0533824041481, 6.543949999999995),
    "2026-09-03": (328.4214399925502, 323.1039373069979, 6.609664285714282),
    "2026-09-04": (329.76991231125703, 324.7885355428754, 6.860378571428567),
    "2026-09-11": (336.3560774590717, 326.3, 7.329599999999999),
}
INTC_FORECASTS = {
    "2026-03-02": (45.88399158706168, 43.88328505541922, 2.331371428571429),
    "2026-08-31": (92.52259005688143, 88.91520912956068, 4.119050000000001),
    "2026-09-01": (88.20672619422156, 85.05890732577451, 4.1104785714285725),
    "2026-09-02": (90.92706444298031, 87.5004722200887, 4.157135714285715),
    "2026-09-03": (90.08890242862724, 87.19492111916887, 3.8492785714285733),
    "2026-09-04": (95.81487950584402, 91.81971111056774, 3.8367785714285736),
    "2026-09-11": (104.88463667029384, 101.07, 3.8160571428571433),
}
# MU, saved 2026-09-30, reads its opening five minutes from IEX. Beside the
# held-out week: three sessions whose IEX window is short (2024-05-03: 3 bars;
# 2024-05-09: 4, no 09:30 bar; 2025-08-26: 4) and two after 2025-03-10, the one
# session IEX printed nothing in the window of -- its opening row is missing from
# the 28-session `or_volume_z` windows these read.
MU_FORECASTS = {
    "2024-05-03": (115.30969869214596, 112.23142878513893, 4.4696),
    "2024-05-09": (119.7100917670105, 117.35568271451045, 3.551385714285715),
    "2025-03-11": (89.83187273312511, 86.43684487323075, 4.9813857142857145),
    "2025-04-01": (88.8252353457782, 86.27698365175549, 3.6731214285714304),
    "2025-08-26": (118.87739195199754, 115.53119054623781, 4.19375),
    "2026-03-02": (425.0276091905608, 397.27, 21.79093571428572),
    "2026-08-31": (958.5099101163877, 920.1623735778495, 41.976057142857144),
    "2026-09-01": (965.8477957297089, 930.9890304491993, 41.60462142857143),
    "2026-09-02": (958.998621751499, 924.1214764145235, 42.375335714285725),
    "2026-09-03": (959.63, 902.2897049413292, 39.47127142857145),
    "2026-09-04": (1020.1816140024187, 969.415, 40.40555000000001),
    "2026-09-11": (1004.0314071814445, 973.5282062200738, 39.09912142857146),
}
# BE, saved 2026-09-30: IEX opening, sessions kept from 370 bars, and both nets
# read `lead_or_ret_adr` -- VST's opening move, or the three peers' mean when VST
# has no session (2024-01-25, 2024-01-30 and 2025-12-31 are such days).
# 2025-03-20 follows a 380-bar session the 385 threshold would have dropped.
BE_FORECASTS = {
    "2024-01-25": (12.385294410373472, 11.885134703937394, 0.6581000000000004),
    "2024-01-30": (11.881398568113182, 11.366851973681667, 0.6709571428571429),
    "2025-03-03": (24.983474312555384, 23.119309288859345, 1.6507214285714285),
    "2025-03-20": (26.055005754281275, 24.464228977126428, 1.8423714285714288),
    "2025-12-31": (89.345984174502, 84.049977889037, 7.901228571428571),
    "2026-03-02": (159.82499929282542, 148.27041193059029, 15.58590714285714),
    "2026-08-31": (210.73049738457146, 198.0394663021734, 15.116421428571426),
    "2026-09-01": (210.54852122632903, 198.7941043872093, 14.943564285714283),
    "2026-09-02": (212.38925753695253, 201.09077201113618, 14.754278571428566),
    "2026-09-03": (219.88020886985672, 206.8663942954915, 14.263428571428566),
    "2026-09-04": (246.9925999891292, 230.829461243144, 14.648428571428566),
    "2026-09-11": (277.22314330940884, 261.28178609658266, 15.301199999999994),
}
NOTEBOOK_FORECASTS = {
    "AAPL": AAPL_FORECASTS, "INTC": INTC_FORECASTS, "MU": MU_FORECASTS, "BE": BE_FORECASTS,
}
# The candidates each bundle ships with non-zero weight.
SHIPPED = {
    "AAPL": ["lgbm", "nbeats"], "INTC": ["nhits"], "MU": ["lgbm", "nbeats", "nhits"],
    "BE": ["nbeats", "nhits"],
}
# The tape each bundle reads the opening from.
OPENING_FEEDS = {"AAPL": "sip", "INTC": "sip", "MU": "iex", "BE": "iex"}


# --- synthetic sessions -------------------------------------------------------

def _session_bars(day: date, base: float, rng) -> "list[dict]":
    """A full 390-bar regular session in Alpaca's bar-dict shape."""
    start = pd.Timestamp(day).tz_localize("America/New_York") + pd.Timedelta(hours=9, minutes=30)
    closes = base * np.exp(np.cumsum(rng.normal(0, 0.0008, 390)))
    bars = []
    prev = base
    for i, close in enumerate(closes):
        hi, lo = max(prev, close) * 1.0004, min(prev, close) * 0.9996
        bars.append({
            "t": (start + pd.Timedelta(minutes=i)).tz_convert("UTC").isoformat(),
            "o": prev, "h": hi, "l": lo, "c": close, "v": 1000 + int(rng.integers(0, 500)),
        })
        prev = close
    return bars


def _tape(first: date, last: date, scale: float = 1.0) -> "dict[date, list[dict]]":
    """Deterministic sessions for every weekday in [first, last]."""
    rng = np.random.default_rng(3)
    out, day, base = {}, first, 100.0
    while day <= last:
        if day.weekday() < 5:
            bars = _session_bars(day, base, rng)
            base = bars[-1]["c"]
            out[day] = [
                {**b, "o": b["o"] * scale, "h": b["h"] * scale, "l": b["l"] * scale, "c": b["c"] * scale}
                for b in bars
            ]
        day += timedelta(days=1)
    return out


class FakeSip:
    """Stands in for `highlow_model._fetch_rollups`, counting what is asked."""

    def __init__(self, tape):
        self.tape = tape
        self.calls: "list[tuple[date, date]]" = []

    def __call__(self, symbol, first, before, key, secret, opening_feed="sip",
                 min_bars=H.MIN_BARS_PER_SESSION):
        self.calls.append((first, before))
        bars = [b for d, day_bars in self.tape.items() if first <= d < before for b in day_bars]
        return H.session_rollups(H.minute_frame_from_bars(bars), min_bars=min_bars)


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(H, "HISTORY_DIR", tmp_path / "highlow")
    return tmp_path / "highlow"


class TestRegistry:
    def test_highlow_is_offered_where_a_bundle_was_saved(self):
        for symbol in ("AAPL", "INTC", "MU", "BE"):
            assert apple_models.HIGHLOW_KEY in apple_models.keys_for(symbol)
        assert apple_models.HIGHLOW_KEY not in apple_models.keys_for("GOOGL")

    def test_it_drives_the_day_range_rules(self):
        assert apple_models.strategy(apple_models.HIGHLOW_KEY) == apple_models.STRATEGY_DAYRANGE


class TestRollups:
    def test_a_half_day_is_dropped_like_the_notebook_drops_it(self):
        tape = _tape(date(2026, 3, 2), date(2026, 3, 3))
        half = [b for b in tape[date(2026, 3, 3)]][:210]  # 13:00 close
        rollups, dropped = H.session_rollups(
            H.minute_frame_from_bars(tape[date(2026, 3, 2)] + half)
        )
        assert list(rollups.index) == [pd.Timestamp("2026-03-02")]
        assert [pd.Timestamp(d) for d in dropped] == [pd.Timestamp("2026-03-03")]

    def test_rolling_up_in_chunks_matches_rolling_up_at_once(self):
        tape = _tape(date(2026, 3, 2), date(2026, 3, 13))
        bars = [b for day in tape.values() for b in day]
        whole, _ = H.session_rollups(H.minute_frame_from_bars(bars))
        days = sorted(tape)
        halves = [
            H.session_rollups(H.minute_frame_from_bars([b for d in part for b in tape[d]]))[0]
            for part in (days[:4], days[4:])
        ]
        pd.testing.assert_frame_equal(pd.concat(halves), whole)


class TestThinOpeningFeed:
    """An IEX-opening bundle's rollups: SIP decides which sessions exist, IEX
    supplies their opening summaries, and a session IEX missed keeps its row."""

    def test_the_opening_summary_is_the_thin_feeds(self):
        tape = _tape(date(2026, 3, 2), date(2026, 3, 3))
        sip = H.minute_frame_from_bars([b for d in sorted(tape) for b in tape[d]])
        iex_bars = [
            {**b, "h": b["h"] * 0.9999, "v": 10} for d in sorted(tape) for b in tape[d][:5]
        ]
        rollups, _ = H.session_rollups(sip, H.minute_frame_from_bars(iex_bars))
        sip_only, _ = H.session_rollups(sip)
        pd.testing.assert_frame_equal(rollups[H._ROLLUP_COLS], sip_only[H._ROLLUP_COLS])
        assert (rollups["volume5"] == 50).all()
        assert (rollups["high5"] < sip_only["high5"]).all()

    def test_a_session_the_thin_feed_missed_keeps_its_daily_row(self):
        tape = _tape(date(2026, 3, 2), date(2026, 3, 4))
        sip = H.minute_frame_from_bars([b for d in sorted(tape) for b in tape[d]])
        # three bars on the 2nd, none on the 3rd, a late start on the 4th
        iex = H.minute_frame_from_bars(
            tape[date(2026, 3, 2)][:3] + tape[date(2026, 3, 4)][1:5]
        )
        rollups, _ = H.session_rollups(sip, iex)
        assert len(rollups) == 3
        assert np.isnan(rollups.loc["2026-03-03", "open5"])
        assert rollups.loc["2026-03-04", "open5"] == pytest.approx(tape[date(2026, 3, 4)][1]["o"])

    def test_a_cache_of_the_other_feeds_openings_is_rebuilt(self, cache_dir, monkeypatch):
        sip = FakeSip(_tape(date(2025, 12, 1), date(2026, 9, 30)))
        monkeypatch.setattr(H, "_fetch_rollups", sip)
        H.history_frame("MU", date(2026, 9, 14), "k", "s")
        assert len(sip.calls) == 1

        H.history_frame("MU", date(2026, 9, 14), "k", "s", opening_feed="iex")
        assert len(sip.calls) == 2  # SIP openings are not IEX ones: refetched whole
        assert H._read_cache("MU", "iex")["opening_feed"] == "iex"
        assert H._read_cache("MU")["sessions"] == {}  # and the SIP reader sees none

    def test_forecast_session_reads_the_window_from_the_bundles_feed(self, monkeypatch):
        H._forecast_cache.clear()
        asked = {}
        monkeypatch.setattr(
            H, "history_frame",
            lambda symbol, before, key=None, secret=None, opening_feed="sip", min_bars=385:
                asked.setdefault("history", opening_feed) and pd.DataFrame(),
        )
        monkeypatch.setattr(
            H, "fetch_opening_window",
            lambda symbol, day, want, feed, key=None, secret=None:
                asked.setdefault("window", feed) and pd.DataFrame(),
        )
        monkeypatch.setattr(
            H, "forecast_from",
            lambda bundle, history, opening, day, window=None, peer_data=None: {"window": window},
        )
        opening = H.minute_frame_from_bars(_tape(date(2026, 7, 1), date(2026, 7, 1))[date(2026, 7, 1)][:5])
        H.forecast_session({"opening_feed": "iex", "path": "mu"}, "MU", opening, date(2026, 7, 1))
        assert asked == {"history": "iex", "window": "iex"}

        asked.clear()
        out = H.forecast_session({"path": "aapl"}, "AAPL", opening, date(2026, 7, 1))
        assert asked == {"history": "sip"} and out["window"] is None
        H._forecast_cache.clear()


class TestThemePeers:
    """BE's bundle reads its theme peers: each one's history and IEX window go
    to the forecast in the sidecar's order, lead first."""

    def test_forecast_session_hands_every_peer_its_history_and_window(self, monkeypatch):
        H._forecast_cache.clear()
        histories, windows = [], []
        monkeypatch.setattr(
            H, "history_frame",
            lambda symbol, before, key=None, secret=None, opening_feed="sip", min_bars=385:
                histories.append((symbol, opening_feed, min_bars)) or pd.DataFrame({"s": [symbol]}),
        )
        monkeypatch.setattr(
            H, "fetch_opening_window",
            lambda symbol, day, want, feed, key=None, secret=None:
                windows.append((symbol, feed)) or pd.DataFrame({"w": [symbol]}),
        )
        monkeypatch.setattr(
            H, "forecast_from",
            lambda bundle, history, opening, day, window=None, peer_data=None: {"peers": peer_data},
        )
        opening = H.minute_frame_from_bars(_tape(date(2026, 7, 1), date(2026, 7, 1))[date(2026, 7, 1)][:5])
        bundle = {"opening_feed": "iex", "min_bars": 370, "peers": ("VST", "PLUG", "XLU"), "path": "be"}
        out = H.forecast_session(bundle, "BE", opening, date(2026, 7, 1))
        assert histories == [("BE", "iex", 370), ("VST", "iex", 385), ("PLUG", "iex", 385),
                             ("XLU", "iex", 385)]
        assert windows == [("BE", "iex"), ("VST", "iex"), ("PLUG", "iex"), ("XLU", "iex")]
        assert list(out["peers"]) == ["VST", "PLUG", "XLU"]
        assert out["peers"]["VST"][0]["s"][0] == "VST" and out["peers"]["VST"][1]["w"][0] == "VST"
        H._forecast_cache.clear()

    def test_a_bundle_reading_peers_refuses_without_them(self):
        bundle, raw, thin, history, _ = notebook_inputs("BE")
        stamp = pd.Timestamp("2026-09-04")
        with pytest.raises(ValueError, match="theme peers"):
            H.forecast_from(bundle, history, _one_day(raw, stamp).iloc[:5], stamp,
                            _one_day(thin, stamp, 5))

    def test_a_lead_peer_without_a_window_falls_back_to_the_group(self):
        """2025-12-31: VST's SIP session was short (380 bars), so the notebook has
        no VST row and reads the mean of PLUG and XLU. Handing the mirror VST's
        IEX window instead -- which is all 9:35 knows live -- moves the forecast."""
        bundle, raw, thin, history, peer_tapes = notebook_inputs("BE")
        stamp = pd.Timestamp("2025-12-31")
        peer_data = _peer_data(peer_tapes, stamp)
        assert peer_data["VST"][1].empty
        opening, window = _one_day(raw, stamp).iloc[:5], _one_day(thin, stamp, 5)
        kept = H.forecast_from(bundle, history, opening, stamp, window, peer_data)
        vst_raw = peer_tapes["VST"][2]
        live = H.forecast_from(bundle, history, opening, stamp, window,
                               {**peer_data, "VST": (peer_data["VST"][0], _one_day(vst_raw, stamp, 5))})
        assert kept["pred_high"] == pytest.approx(BE_FORECASTS["2025-12-31"][0], abs=1e-5)
        assert 1e-5 < abs(live["pred_high"] - kept["pred_high"]) < 0.01

    def test_a_cache_cleaned_at_another_bar_count_is_rebuilt(self, cache_dir, monkeypatch):
        sip = FakeSip(_tape(date(2025, 12, 1), date(2026, 9, 30)))
        monkeypatch.setattr(H, "_fetch_rollups", sip)
        H.history_frame("BE", date(2026, 9, 14), "k", "s", "iex")
        H.history_frame("BE", date(2026, 9, 14), "k", "s", "iex", min_bars=370)
        assert len(sip.calls) == 2
        assert H._read_cache("BE", "iex", 370)["min_bars"] == 370
        assert H._read_cache("BE", "iex")["sessions"] == {}

    def test_a_short_quiet_session_is_kept_at_the_bundles_threshold(self):
        tape = _tape(date(2026, 3, 2), date(2026, 3, 2))
        # 380 bars: the last ten minutes of a quiet day dropped, still printing near the close
        bars = tape[date(2026, 3, 2)][:375] + tape[date(2026, 3, 2)][385:]
        frame = H.minute_frame_from_bars(bars)
        assert H.session_rollups(frame)[0].empty
        assert len(H.session_rollups(frame, min_bars=370)[0]) == 1

    def test_theme_columns_need_the_sidecar_to_name_the_peers(self, tmp_path):
        src = H.model_path("BE")
        if not src.exists():
            pytest.skip("the HighLow BE bundle is not installed")
        import json
        for extra in (src, src.with_name(f"{src.stem}_nbeats.pt"), src.with_name(f"{src.stem}_nhits.pt")):
            (tmp_path / extra.name).write_bytes(extra.read_bytes())
        meta = json.loads(src.with_suffix(".json").read_text())
        (tmp_path / src.with_suffix(".json").name).write_text(json.dumps({**meta, "peers": []}))
        assert H._build_bundle(tmp_path / src.name) is None


class TestHalfDays:
    def test_the_rule_reproduces_the_notebooks_list(self):
        """`config.EARLY_CLOSE_DATES` in HighLow_5m, which ends in 2025."""
        listed = {
            "2023-07-03", "2023-11-24", "2024-07-03", "2024-11-29", "2024-12-24",
            "2025-07-03", "2025-11-28", "2025-12-24",
        }
        days = pd.bdate_range("2023-01-01", "2025-12-31")
        assert set(days[H.early_close(days)].strftime("%Y-%m-%d")) == listed

    def test_a_half_day_with_a_full_bar_count_is_still_dropped(self):
        """INTC's 28 Nov 2025: SIP printed past the 13:00 close, so the session
        passed the bar count. The date is what drops it."""
        tape = _tape(date(2025, 11, 27), date(2025, 11, 28))
        rollups, dropped = H.session_rollups(H.minute_frame_from_bars(tape[date(2025, 11, 28)]))
        assert rollups.empty
        assert [pd.Timestamp(d) for d in dropped] == [pd.Timestamp("2025-11-28")]


class TestHistoryCache:
    def test_a_cold_cache_fetches_the_whole_window_once(self, cache_dir, monkeypatch):
        sip = FakeSip(_tape(date(2025, 12, 1), date(2026, 9, 30)))
        monkeypatch.setattr(H, "_fetch_rollups", sip)
        day = date(2026, 9, 14)

        frame = H.history_frame("AAPL", day, "k", "s")
        assert sip.calls == [(day - timedelta(days=H.HISTORY_CALENDAR_DAYS), day)]
        assert frame.index.max() < pd.Timestamp(day)
        assert len(frame) >= H.MIN_PRIOR_SESSIONS

        H.history_frame("AAPL", day, "k", "s")
        assert len(sip.calls) == 1  # served from disk

    def test_the_next_morning_fetches_only_the_new_session(self, cache_dir, monkeypatch):
        sip = FakeSip(_tape(date(2025, 12, 1), date(2026, 9, 30)))
        monkeypatch.setattr(H, "_fetch_rollups", sip)
        H.history_frame("AAPL", date(2026, 9, 14), "k", "s")

        frame = H.history_frame("AAPL", date(2026, 9, 15), "k", "s")
        # From the newest cached session (Friday the 11th, the seam) up to the
        # session being forecast.
        assert sip.calls[-1] == (date(2026, 9, 11), date(2026, 9, 15))
        assert frame.index[-1] == pd.Timestamp("2026-09-14")

    def test_a_replay_of_an_earlier_day_fetches_only_the_older_edge(
        self, cache_dir, monkeypatch
    ):
        sip = FakeSip(_tape(date(2025, 12, 1), date(2026, 9, 30)))
        monkeypatch.setattr(H, "_fetch_rollups", sip)
        H.history_frame("AAPL", date(2026, 9, 14), "k", "s")
        oldest = min(H._read_cache("AAPL")["sessions"])

        frame = H.history_frame("AAPL", date(2026, 9, 1), "k", "s")
        # Only the thirteen days its window reaches further back were missing,
        # read up to and including the oldest cached session (the seam).
        assert sip.calls[1:] == [(
            date(2026, 9, 1) - timedelta(days=H.HISTORY_CALENDAR_DAYS),
            date.fromisoformat(oldest) + timedelta(days=1),
        )]
        assert frame.index.max() < pd.Timestamp("2026-09-01")

        H.history_frame("AAPL", date(2026, 8, 20), "k", "s")
        assert len(sip.calls) == 3  # a little further back again, nothing forward

    def test_a_split_rebuilds_the_cache_rather_than_mixing_scales(self, cache_dir, monkeypatch):
        before_split = FakeSip(_tape(date(2025, 12, 1), date(2026, 9, 30)))
        monkeypatch.setattr(H, "_fetch_rollups", before_split)
        H.history_frame("AAPL", date(2026, 9, 14), "k", "s")

        after_split = FakeSip(_tape(date(2025, 12, 1), date(2026, 9, 30), scale=0.25))
        monkeypatch.setattr(H, "_fetch_rollups", after_split)
        frame = H.history_frame("AAPL", date(2026, 9, 15), "k", "s")

        # The seam session came back at a quarter of its cached close: refetched whole.
        assert after_split.calls[-1] == (date(2026, 9, 15) - timedelta(days=H.HISTORY_CALENDAR_DAYS),
                                         date(2026, 9, 15))
        assert frame["close"].max() < 50  # every row on the new scale

    def test_no_credentials_is_a_refusal_not_a_guess(self, cache_dir, monkeypatch):
        monkeypatch.delenv("ALPACA_API_KEY", raising=False)
        monkeypatch.delenv("ALPACA_SECRET", raising=False)
        with pytest.raises(ValueError, match="credentials"):
            H.history_frame("AAPL", date(2026, 9, 14))


class TestForecastSeams:
    def test_a_short_history_is_refused(self):
        tape = _tape(date(2026, 6, 1), date(2026, 7, 31))
        history, _ = H.session_rollups(
            H.minute_frame_from_bars([b for d in sorted(tape)[:-1] for b in tape[d]])
        )
        last = sorted(tape)[-1]
        opening = H.minute_frame_from_bars(tape[last][:5])
        with pytest.raises(ValueError, match="sessions of history"):
            H.forecast_from({"model": None, "opening_minutes": 5}, history, opening, last)

    def test_too_few_opening_bars_is_refused(self):
        with pytest.raises(ValueError, match="first 5 minutes"):
            H.forecast_from(
                {"model": None, "opening_minutes": 5}, pd.DataFrame(),
                H.minute_frame_from_bars(_tape(date(2026, 7, 1), date(2026, 7, 1))[date(2026, 7, 1)][:3]),
                date(2026, 7, 1),
            )


_NOTEBOOK_INPUTS: dict = {}


def notebook_inputs(ticker: str):
    """The bundle, and the notebook's *raw* SIP files rolled up by the live
    path's code -- raw rather than the cleaned `minute.parquet`, so the session
    cleaning (half days included) is part of what is checked."""
    if ticker in _NOTEBOOK_INPUTS:
        return _NOTEBOOK_INPUTS[ticker]
    folder = NOTEBOOK / "data" / ticker / "raw"
    files = sorted(folder.glob(f"{ticker}_20??.parquet"))
    if not files:
        pytest.skip(f"the HighLow_5m {ticker} notebook data is not on this machine")
    bundle = H.load_bundle(ticker)
    if bundle is None:
        pytest.skip(f"the HighLow {ticker} bundle is not installed")

    def tape(paths):
        raw = pd.concat(pd.read_parquet(f) for f in paths).sort_index()
        raw = raw[~raw.index.duplicated()].loc["2023-01-01":"2026-09-11 23:59"]
        return raw.tz_convert("America/New_York").between_time("09:30", "15:59")[H.OHLCV]

    raw = tape(files)
    thin = None
    feed = H.opening_feed(bundle)
    if feed != "sip":
        thin = tape(sorted(folder.glob(f"{ticker}_20??_{feed}.parquet")))
    history, _ = H.session_rollups(raw, thin, H.min_bars(bundle))
    # Each theme peer as the notebook's `load_peers` reads it: SIP rollups at 385
    # bars, IEX openings trimmed to the sessions SIP kept -- and the raw IEX tape.
    peer_tapes = {}
    for peer in H.peers(bundle):
        peer_raw = NOTEBOOK / "data" / peer / "raw"
        p_iex = tape(sorted(peer_raw.glob(f"{peer}_20??_iex.parquet")))
        p_history, _ = H.session_rollups(tape(sorted(peer_raw.glob(f"{peer}_20??.parquet"))), p_iex)
        peer_tapes[peer] = (p_history, H.opening_minute_frame(p_iex, p_history.index)[H.OHLCV], p_iex)
    _NOTEBOOK_INPUTS[ticker] = (bundle, raw, thin, history, peer_tapes)
    return _NOTEBOOK_INPUTS[ticker]


def _peer_data(peer_tapes, stamp):
    """`forecast_from`'s `peer_data` for one day, windows as the notebook kept them."""
    return {p: (h, _one_day(kept, stamp, 5)) for p, (h, kept, _) in peer_tapes.items()} or None


def _one_day(frame, stamp, minutes=None):
    day = frame[frame.index.normalize().tz_localize(None) == stamp]
    if minutes is None or day.empty:
        return day
    return day[day.index < day.index[0].normalize() + pd.Timedelta(hours=9, minutes=30 + minutes)]


class TestAgainstTheNotebook:
    """The mirror contract: the notebook's minute tape, rolled up by the live
    path's code, reproduces the notebook package's forecasts. Not bit-exact --
    the N-BEATS half runs in float32 and the two venvs carry different pandas
    -- but within a millionth of a dollar, far inside anything a level reads."""

    @pytest.mark.parametrize(
        "ticker,day",
        [(t, d) for t, days in NOTEBOOK_FORECASTS.items() for d in sorted(days)],
    )
    def test_reproduces_the_notebook_forecast(self, ticker, day):
        bundle, raw, thin, history, peer_tapes = notebook_inputs(ticker)
        stamp = pd.Timestamp(day)
        opening = _one_day(raw, stamp).iloc[:5]
        window = None if thin is None else _one_day(thin, stamp, 5)
        out = H.forecast_from(bundle, history, opening, stamp, window, _peer_data(peer_tapes, stamp))
        high, low, adr = NOTEBOOK_FORECASTS[ticker][day]
        # MU trades near $1,000: the same float32 noise is a larger dollar figure.
        assert out["pred_high"] == pytest.approx(high, abs=1e-5, rel=1e-8)
        assert out["pred_low"] == pytest.approx(low, abs=1e-5, rel=1e-8)
        assert out["adr14_abs"] == pytest.approx(adr, abs=1e-9)

    def test_a_session_iex_printed_nothing_in_is_refused(self):
        """MU 2025-03-10: the notebook's panel has no row for it."""
        bundle, raw, thin, history, _ = notebook_inputs("MU")
        stamp = pd.Timestamp("2025-03-10")
        assert _one_day(thin, stamp, 5).empty
        with pytest.raises(ValueError, match="IEX printed nothing"):
            H.forecast_from(bundle, history, _one_day(raw, stamp).iloc[:5], stamp,
                            _one_day(thin, stamp, 5))

    @pytest.mark.parametrize("ticker", sorted(OPENING_FEEDS))
    def test_the_opening_feed_is_read_off_the_sidecar(self, ticker):
        bundle, *_ = notebook_inputs(ticker)
        assert H.opening_feed(bundle) == OPENING_FEEDS[ticker]

    @pytest.mark.parametrize("ticker", sorted(SHIPPED))
    def test_only_the_weighted_candidates_are_loaded(self, ticker):
        bundle, *_ = notebook_inputs(ticker)
        assert bundle["kind"] == "highlow"
        assert bundle["ticker"] == ticker  # the file that loaded is the right one
        assert sorted(bundle["daily_models"]) == SHIPPED[ticker]

    def test_a_bundle_whose_shipped_candidate_reads_a_custom_group_is_refused(
        self, tmp_path, monkeypatch
    ):
        """INTC's linear median reads microstructure columns this module never
        builds; at weight 0 that is harmless, weighted it must not load."""
        import joblib

        src = H.model_path("INTC")
        if not src.exists():
            pytest.skip("the HighLow INTC bundle is not installed")
        H._register_unpickle_alias()
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            blob = joblib.load(src)
        blob["weights"] = {**blob["weights"], "linear": 1.0}
        target = tmp_path / src.name
        joblib.dump(blob, target)
        for extra in (src.with_name(f"{src.stem}_nhits.pt"), src.with_suffix(".json")):
            (tmp_path / extra.name).write_bytes(extra.read_bytes())
        assert H._build_bundle(target) is None
