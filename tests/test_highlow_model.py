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
# AVGO, saved 2026-10-04: base features, sessions kept from 320 bars. The two
# 2024 days sit among the pre-split sessions of 329-384 bars the 385 rule would
# drop -- at 385, 2024-05-03's high would be $0.48 lower.
AVGO_FORECASTS = {
    "2024-05-03": (127.3698548374877, 123.9940384456728, 4.501428571428578),
    "2024-06-13": (173.82198972533345, 167.12403198972592, 3.7571428571428607),
    "2026-03-02": (314.7496307012425, 305.4854750385193, 12.973792857142863),
    "2026-08-31": (376.37418046961415, 367.44915793162437, 11.280750000000003),
    "2026-09-01": (367.28976454490817, 359.2512686175464, 10.77503571428572),
    "2026-09-02": (369.84648315202537, 360.83229156351285, 10.59503571428572),
    "2026-09-03": (353.6499, 341.5438502477626, 10.09003571428572),
    "2026-09-04": (364.39602275294214, 353.0329087468992, 9.595000000000008),
    "2026-09-11": (366.32834780285083, 358.7532893099266, 9.227),
}
# NVDA, saved 2026-10-04, fitted through 2026-09-11 and held out on the three
# weeks after: reads SPY's opening, the pre-market, the previous evening, the
# opening's shape and the regime. 2024-07-05 and 2025-12-26 follow a half day
# (its evening is trimmed away, so they read the one before: 3 days old), as
# does 2025-12-01 (the 26 Nov evening, 5 days old, the oldest still read);
# 2024-11-21 and 2026-08-27 are the mornings after results evenings; 09-14 a Monday.
NVDA_FORECASTS = {
    "2024-07-05": (129.59301174198674, 125.84376282333966, 5.342142857142856),
    "2024-11-21": (152.89, 144.4880111216915, 4.02143571428572),
    "2025-12-01": (179.48268470156032, 173.4542650381043, 7.5282071428571395),
    "2025-12-26": (193.1154652922541, 189.0722972144385, 4.5086928571428535),
    "2026-03-02": (182.49686417910112, 174.62, 5.793721428571429),
    "2026-08-27": (226.71037878112529, 219.28034039636856, 4.769121428571426),
    "2026-09-14": (212.4665030058211, 207.3424562526983, 6.372857142857143),
    "2026-09-21": (225.2065022842249, 221.0988011605764, 4.872142857142857),
    "2026-10-02": (238.31206881509854, 233.84640363077662, 4.331221428571425),
}
NOTEBOOK_FORECASTS = {
    "AAPL": AAPL_FORECASTS, "INTC": INTC_FORECASTS, "MU": MU_FORECASTS, "BE": BE_FORECASTS,
    "AVGO": AVGO_FORECASTS, "NVDA": NVDA_FORECASTS,
}
# The candidates each bundle ships with non-zero weight.
SHIPPED = {
    "AAPL": ["lgbm", "nbeats"], "INTC": ["nhits"], "MU": ["lgbm", "nbeats", "nhits"],
    "BE": ["nbeats", "nhits"], "AVGO": ["lgbm", "nbeats"], "NVDA": ["lgbm", "nhits"],
}
# The tape each bundle reads the opening from.
OPENING_FEEDS = {"AAPL": "sip", "INTC": "sip", "MU": "iex", "BE": "iex", "AVGO": "sip", "NVDA": "sip"}
# The last session of each notebook's tape the mirror test reads.
NOTEBOOK_TAPE_END = {"NVDA": "2026-10-02"}


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
            lambda symbol, before, key=None, secret=None, opening_feed="sip", min_bars=385, extended=False:
                asked.setdefault("history", opening_feed) and pd.DataFrame(),
        )
        monkeypatch.setattr(
            H, "fetch_opening_window",
            lambda symbol, day, want, feed, key=None, secret=None:
                asked.setdefault("window", feed) and pd.DataFrame(),
        )
        monkeypatch.setattr(
            H, "forecast_from",
            lambda bundle, history, opening, day, window=None, peer_data=None, **_: {"window": window},
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
            lambda symbol, before, key=None, secret=None, opening_feed="sip", min_bars=385, extended=False:
                histories.append((symbol, opening_feed, min_bars)) or pd.DataFrame({"s": [symbol]}),
        )
        monkeypatch.setattr(
            H, "fetch_opening_window",
            lambda symbol, day, want, feed, key=None, secret=None:
                windows.append((symbol, feed)) or pd.DataFrame({"w": [symbol]}),
        )
        monkeypatch.setattr(
            H, "forecast_from",
            lambda bundle, history, opening, day, window=None, peer_data=None, **_: {"peers": peer_data},
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
        bundle, raw, thin, history, *_ = notebook_inputs("BE")
        stamp = pd.Timestamp("2026-09-04")
        with pytest.raises(ValueError, match="theme peers"):
            H.forecast_from(bundle, history, _one_day(raw, stamp).iloc[:5], stamp,
                            _one_day(thin, stamp, 5))

    def test_a_lead_peer_without_a_window_falls_back_to_the_group(self):
        """2025-12-31: VST's SIP session was short (380 bars), so the notebook has
        no VST row and reads the mean of PLUG and XLU. Handing the mirror VST's
        IEX window instead -- which is all 9:35 knows live -- moves the forecast."""
        bundle, raw, thin, history, peer_tapes, _ = notebook_inputs("BE")
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


def _bar(day: date, hh: int, mm: int, price: float, volume: int = 100) -> dict:
    """One extended-hours minute bar in Alpaca's shape."""
    ts = pd.Timestamp(day).tz_localize("America/New_York") + pd.Timedelta(hours=hh, minutes=mm)
    return {"t": ts.tz_convert("UTC").isoformat(), "o": price, "h": price + 0.1, "l": price - 0.1,
            "c": price, "v": volume}


class TestExtendedHoursAndMarket:
    """NVDA's groups: the pre-market and the previous evening come off the
    cached SIP tape, SPY off its own cache and this morning's window."""

    def test_the_evening_leaves_out_the_closing_auction_and_is_read_the_next_session(self):
        days = [date(2026, 9, 9), date(2026, 9, 10), date(2026, 9, 11)]
        post = H.all_hours_frame_from_bars(
            [_bar(days[0], 16, 0, 999.0, 10**7), _bar(days[0], 16, 30, 101.0, 500), _bar(days[0], 19, 59, 102.0, 500)]
            + [_bar(days[1], 17, 0, 103.0, 300)]
        )
        ev = H.evening_summary(H.add_session_columns(post))
        assert ev.loc["2026-09-09", "ah_high"] == pytest.approx(102.1)  # the 16:00 print is not the evening
        assert ev.loc["2026-09-09", "ah_volume"] == 1000
        daily = pd.DataFrame({"close": [100.0, 101.0, 102.0, 103.0, 104.0]},
                             index=pd.DatetimeIndex(["2026-09-09", "2026-09-10", "2026-09-11",
                                                     "2026-09-14", "2026-09-21"]))
        f = H.afterhours_from(ev, daily)
        assert np.isnan(f.loc["2026-09-09", "ah_last"])         # no evening before it
        assert f.loc["2026-09-10", "ah_last"] == 102.0          # the evening before
        assert f.loc["2026-09-11", "ah_last"] == 103.0
        assert f.loc["2026-09-14", "ah_last"] == 103.0          # Monday reads Thursday's: Friday's printed nothing
        assert np.isnan(f.loc["2026-09-21", "ah_last"])         # 11 days old: a hole, not last evening
        assert f.loc["2026-09-10", "ah_prev_close"] == 100.0

    def test_extended_summaries_keep_the_notebooks_windows_and_sessions(self):
        tape = _tape(date(2025, 11, 26), date(2025, 12, 1))
        day, half_day = date(2025, 11, 26), date(2025, 11, 28)
        extended = [_bar(day, 4, 0, 50.0), _bar(day, 9, 19, 51.0), _bar(day, 9, 25, 80.0),  # 09:25: past 09:19
                    _bar(day, 16, 0, 90.0), _bar(day, 16, 1, 52.0), _bar(day, 20, 0, 70.0),  # 20:00: past 19:59
                    _bar(half_day, 8, 0, 55.0), _bar(half_day, 16, 30, 56.0)]
        bars = [b for d in sorted(tape) for b in tape[d]] + extended
        all_hours = H.all_hours_frame_from_bars(bars)
        rollups, dropped = H.session_rollups(H.minute_frame_from_bars(bars), all_hours=all_hours)
        row = rollups.loc["2025-11-26"]
        assert (row["pm_last"], row["pm_high"], row["pm_bars"]) == (51.0, pytest.approx(51.1), 2)
        assert (row["ah_last"], row["ah_high"]) == (52.0, pytest.approx(52.1))
        assert pd.Timestamp(half_day) in dropped and pd.Timestamp(half_day) not in rollups.index
        assert np.isnan(rollups.loc["2025-12-01", "pm_last"])  # printed nothing before the open

    def test_a_cache_without_extended_summaries_is_rebuilt_only_when_asked(self, cache_dir, monkeypatch):
        old = FakeSip(_tape(date(2025, 12, 1), date(2026, 9, 30)))  # rollups without them
        monkeypatch.setattr(H, "_fetch_rollups", old)
        H.history_frame("NVDA", date(2026, 9, 14), "k", "s")
        assert H._read_cache("NVDA")["extended"] is False
        H.history_frame("NVDA", date(2026, 9, 14), "k", "s")
        assert len(old.calls) == 1

        def with_extended(symbol, first, before, key, secret, opening_feed="sip", min_bars=385):
            bars = [b for d, day_bars in old.tape.items() if first <= d < before for b in day_bars]
            return H.session_rollups(H.minute_frame_from_bars(bars), min_bars=min_bars,
                                     all_hours=H.all_hours_frame_from_bars(bars))

        monkeypatch.setattr(H, "_fetch_rollups", with_extended)
        frame = H.history_frame("NVDA", date(2026, 9, 14), "k", "s", extended=True)
        assert set(H._EXTENDED_COLS) <= set(frame.columns)
        assert H._read_cache("NVDA", extended=True)["extended"] is True

    def test_the_market_window_is_sip_once_released_and_iex_before(self):
        tz = "America/New_York"
        at = lambda hh, mm, ss=0: pd.Timestamp(2026, 10, 5, hh, mm, ss, tz=tz).to_pydatetime()
        assert H.market_window_feed(date(2026, 10, 5), 5, at(9, 35, 5)) == "iex"   # live at 9:35
        assert H.market_window_feed(date(2026, 10, 5), 5, at(9, 50, 4)) == "sip"   # released
        assert H.market_window_feed(date(2026, 9, 14), 5, at(9, 35, 5)) == "sip"   # a replay

    def test_forecast_session_hands_an_nvda_bundle_spy_and_the_premarket(self, monkeypatch):
        H._forecast_cache.clear()
        asked = []
        monkeypatch.setattr(
            H, "history_frame",
            lambda symbol, before, key=None, secret=None, opening_feed="sip", min_bars=385, extended=False:
                asked.append(("history", symbol, opening_feed, extended)) or pd.DataFrame({"s": [symbol]}),
        )
        monkeypatch.setattr(
            H, "fetch_opening_window",
            lambda symbol, day, want, feed, key=None, secret=None:
                asked.append(("window", symbol, feed)) or pd.DataFrame({"w": [symbol]}),
        )
        monkeypatch.setattr(H, "market_window_feed", lambda day, want, now=None: "sip")
        monkeypatch.setattr(H, "fetch_premarket", lambda symbol, day, key=None, secret=None:
                            asked.append(("premarket", symbol)) or pd.DataFrame({"p": [symbol]}))
        monkeypatch.setattr(H, "forecast_from", lambda *a, **k: k)
        opening = H.minute_frame_from_bars(_tape(date(2026, 7, 1), date(2026, 7, 1))[date(2026, 7, 1)][:5])
        bundle = {"path": "nvda", "groups": ("market", "premarket", "afterhours", "opening_shape", "regime"),
                  "market_proxy": "SPY"}
        out = H.forecast_session(bundle, "NVDA", opening, date(2026, 7, 1))
        assert asked == [("history", "NVDA", "sip", True), ("history", "SPY", "sip", False),
                         ("window", "SPY", "sip"), ("premarket", "NVDA")]
        assert out["market_data"][0]["s"][0] == "SPY" and out["market_data"][1]["w"][0] == "SPY"
        assert out["premarket_bars"]["p"][0] == "NVDA"
        H._forecast_cache.clear()

    def test_a_bundle_reading_the_evening_refuses_a_history_without_it(self):
        tape = _tape(date(2025, 12, 1), date(2026, 7, 31))
        last = sorted(tape)[-1]
        history, _ = H.session_rollups(
            H.minute_frame_from_bars([b for d in sorted(tape)[:-1] for b in tape[d]])
        )
        opening = H.minute_frame_from_bars(tape[last][:5])
        bundle = {"model": None, "opening_minutes": 5, "groups": ("afterhours",)}
        with pytest.raises(ValueError, match="extended-hours"):
            H.forecast_from(bundle, history, opening, last)

    def _copy(self, ticker, tmp_path, meta_edit=None):
        import json
        src = H.model_path(ticker)
        if not src.exists():
            pytest.skip(f"the HighLow {ticker} bundle is not installed")
        for extra in (src, src.with_name(f"{src.stem}_nbeats.pt"), src.with_name(f"{src.stem}_nhits.pt")):
            (tmp_path / extra.name).write_bytes(extra.read_bytes())
        meta = json.loads(src.with_suffix(".json").read_text())
        (tmp_path / src.with_suffix(".json").name).write_text(json.dumps((meta_edit or (lambda m: m))(meta)))
        return tmp_path / src.name

    def test_nvda_reads_five_groups_and_spy_but_not_its_peers(self, tmp_path):
        bundle = H._build_bundle(self._copy("NVDA", tmp_path))
        assert H.groups(bundle) == ("market", "premarket", "afterhours", "opening_shape", "regime")
        assert H.market_proxy(bundle) == "SPY" and H.reads_extended_hours(bundle)
        assert H.peers(bundle) == ()  # theme built in the notebook, read by nothing shipped

    def test_the_market_group_needs_the_sidecar_to_name_the_proxy(self, tmp_path):
        path = self._copy("NVDA", tmp_path, lambda m: {**m, "market_proxy": None})
        assert H._build_bundle(path) is None

    def test_theme_peers_on_a_sip_opening_are_refused(self, tmp_path):
        """NVDA's notebook reads its peers on SIP; only BE's IEX peers are mirrored."""
        path = self._copy("BE", tmp_path, lambda m: {**m, "opening_feed": "sip"})
        assert H._build_bundle(path) is None

    def test_avgo_keeps_sessions_from_its_notebooks_threshold(self, tmp_path):
        bundle = H._build_bundle(self._copy("AVGO", tmp_path))
        assert "min_bars_per_session" not in bundle["metadata"]
        assert H.min_bars(bundle) == 320 and H.groups(bundle) == ()


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
    end = NOTEBOOK_TAPE_END.get(ticker, "2026-09-11")

    def tape(paths, hours=("09:30", "15:59")):
        raw = pd.concat(pd.read_parquet(f) for f in paths).sort_index()
        raw = raw[~raw.index.duplicated()].loc["2023-01-01":f"{end} 23:59"]
        return raw.tz_convert("America/New_York").between_time(*hours)[H.OHLCV]

    raw = tape(files)
    thin = None
    feed = H.opening_feed(bundle)
    if feed != "sip":
        thin = tape(sorted(folder.glob(f"{ticker}_20??_{feed}.parquet")))
    # The pre-market and after-hours files the notebook downloaded beside the
    # session (04:00-09:19, 16:00-19:59), as one all-hours tape.
    extras = {}
    all_hours = None
    if H.reads_extended_hours(bundle):
        extras["pre"] = tape(sorted(folder.glob(f"{ticker}_20??_pre.parquet")), ("00:00", "23:59"))
        post = tape(sorted(folder.glob(f"{ticker}_20??_post.parquet")), ("00:00", "23:59"))
        all_hours = pd.concat([raw, extras["pre"], post]).sort_index()
    history, _ = H.session_rollups(raw, thin, H.min_bars(bundle), all_hours)
    proxy = H.market_proxy(bundle)
    if proxy:
        m_raw = tape(sorted((NOTEBOOK / "data" / proxy / "raw").glob(f"{proxy}_20??.parquet")))
        extras["market"] = (H.session_rollups(m_raw)[0], m_raw)
    # Each theme peer as the notebook's `load_peers` reads it: SIP rollups at 385
    # bars, IEX openings trimmed to the sessions SIP kept -- and the raw IEX tape.
    peer_tapes = {}
    for peer in H.peers(bundle):
        peer_raw = NOTEBOOK / "data" / peer / "raw"
        p_iex = tape(sorted(peer_raw.glob(f"{peer}_20??_iex.parquet")))
        p_history, _ = H.session_rollups(tape(sorted(peer_raw.glob(f"{peer}_20??.parquet"))), p_iex)
        peer_tapes[peer] = (p_history, H.opening_minute_frame(p_iex, p_history.index)[H.OHLCV], p_iex)
    _NOTEBOOK_INPUTS[ticker] = (bundle, raw, thin, history, peer_tapes, extras)
    return _NOTEBOOK_INPUTS[ticker]


def _extra_inputs(extras, stamp):
    """`forecast_from`'s market and pre-market inputs for one day, the market's
    window as the notebook read it (SIP's first five minutes)."""
    out = {}
    if "market" in extras:
        m_history, m_raw = extras["market"]
        out["market_data"] = (m_history, _one_day(m_raw, stamp, 5))
    if "pre" in extras:
        out["premarket_bars"] = _one_day(extras["pre"], stamp)
    return out


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
        bundle, raw, thin, history, peer_tapes, extras = notebook_inputs(ticker)
        stamp = pd.Timestamp(day)
        opening = _one_day(raw, stamp).iloc[:5]
        window = None if thin is None else _one_day(thin, stamp, 5)
        out = H.forecast_from(bundle, history, opening, stamp, window, _peer_data(peer_tapes, stamp),
                              **_extra_inputs(extras, stamp))
        high, low, adr = NOTEBOOK_FORECASTS[ticker][day]
        # MU trades near $1,000: the same float32 noise is a larger dollar figure.
        assert out["pred_high"] == pytest.approx(high, abs=1e-5, rel=1e-8)
        assert out["pred_low"] == pytest.approx(low, abs=1e-5, rel=1e-8)
        assert out["adr14_abs"] == pytest.approx(adr, abs=1e-9)

    def test_a_session_iex_printed_nothing_in_is_refused(self):
        """MU 2025-03-10: the notebook's panel has no row for it."""
        bundle, raw, thin, history, *_ = notebook_inputs("MU")
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
