"""The HighLow_3m mirror: the history cache, the 9:33 opening, the option tables, the bundle.

The tests that need neither the saved model nor the notebook pin the seams --
what the app has to get right to hand the model correct inputs: which days the
caches fetch and when, the 9:33 line the opening is cut at, an option row being
built only from what was known at that close (and kept only once it is final),
and the refusals.

The ones that need both pin the mirror itself: the notebook's own raw minute
files and option bars, run through the live path's code, must reproduce what
the notebook's `highlow3m` package built and predicted.
"""

import json
import warnings
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agent_stonks import apple_models

H = pytest.importorskip("agent_stonks.highlow3m_model")

NOTEBOOK = Path("/Users/aleksandra/Documents/playground/Code/FinNotebooks/HighLow_3m")

# What the notebook's own `highlow3m` package predicts from its own panel
# (`pipeline.panel`, then `models.load_bundle(...).predict_prices`) with the
# AAPL bundle saved 2026-10-05: (pred_high, pred_low, adr14_usd). The test
# window (8-18 Sep 2026) and the traded week (21-25 Sep), plus the sessions
# after a half day (2025-07-07, 2025-12-01, 2025-12-26) -- whose option row is
# the half day's close, 2025-07-03's without a 7-day wall on either side --
# and two first-of-month Mondays.
AAPL_FORECASTS = {
    "2025-03-03": (244.72835589342122, 240.57052671713765, 5.208178571428576),
    "2025-07-07": (216.54963124885492, 213.17729397643777, 3.788707142857141),
    "2025-12-01": (279.96157611790915, 276.3195632181682, 5.790285714285716),
    "2025-12-26": (276.12614769436686, 273.24800566762843, 4.327450000000007),
    "2026-03-02": (265.87888190775055, 260.3295638169582, 6.819571428571438),
    "2026-09-08": (321.6640526843638, 315.12525450922436, 7.313878571428567),
    "2026-09-09": (318.4222443985974, 312.7850853800904, 7.317449999999996),
    "2026-09-10": (321.27649842254704, 314.6495484821407, 7.286742857142855),
    "2026-09-11": (331.93668979171804, 325.8212607054519, 7.329599999999999),
    "2026-09-14": (335.0782428050734, 329.2309056848788, 7.6546),
    "2026-09-15": (333.53224042120536, 327.99283875208505, 7.709600000000003),
    "2026-09-16": (336.02030486195724, 330.56667433714244, 7.570314285714285),
    "2026-09-17": (336.14477556676604, 330.81073206097363, 7.438178571428572),
    "2026-09-18": (340.1237715083028, 334.31191373803256, 7.592235714285716),
    "2026-09-21": (337.0922322515308, 331.95180960987676, 7.523692857142862),
    "2026-09-22": (346.98847748463953, 342.0316496183337, 7.3919071428571455),
    "2026-09-23": (343.52561851727364, 338.4317713324233, 6.9647642857142875),
    "2026-09-24": (338.30175301592425, 333.3287252554556, 7.066907142857145),
    "2026-09-25": (337.2669785369002, 332.6607104342691, 6.917621428571432),
}


# --- synthetic sessions -------------------------------------------------------

TZ = "America/New_York"
_RTH = [9 * 60 + 30 + i for i in range(390)]


def _bars(times: pd.DatetimeIndex, base: float, rng, scale: float = 1.0) -> "tuple[list[dict], float]":
    closes = base * np.exp(np.cumsum(rng.normal(0, 0.0008, len(times))))
    opens = np.r_[base, closes[:-1]]
    highs = np.maximum(opens, closes) * 1.0004
    lows = np.minimum(opens, closes) * 0.9996
    volumes = 1000 + rng.integers(0, 500, len(times))
    stamps = times.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    out = [
        {"t": t, "o": o * scale, "h": h * scale, "l": lo * scale, "c": c * scale, "v": int(v)}
        for t, o, h, lo, c, v in zip(stamps, opens, highs, lows, closes, volumes)
    ]
    return out, float(closes[-1])


@lru_cache(maxsize=None)
def _tape(first: date, last: date, scale: float = 1.0) -> "dict[date, tuple[list, list]]":
    """Deterministic (SIP, IEX) regular-session bars for every NYSE session in
    [first, last]; IEX is SIP at a twentieth of the volume. Read-only."""
    rng = np.random.default_rng(3)
    out, day, base = {}, first, 100.0
    holidays = set(H.NYSE_HOLIDAYS)
    while day <= last:
        if day.weekday() < 5 and day.isoformat() not in holidays:
            midnight = pd.Timestamp(day).tz_localize(TZ)
            sip, base = _bars(midnight + pd.to_timedelta(_RTH, unit="min"), base, rng, scale)
            out[day] = (sip, [{**b, "v": b["v"] // 20} for b in sip])
        day += timedelta(days=1)
    return out


def _frames(tape, days) -> "tuple[pd.DataFrame, pd.DataFrame]":
    return (H.bars_frame([b for d in days for b in tape[d][0]]),
            H.bars_frame([b for d in days for b in tape[d][1]]))


class FakeAlpaca:
    """Stands in for `_fetch_stretch` and `fetch_opening`, counting what is asked."""

    def __init__(self, tape):
        self.tape = tape
        self.stretches: "list[tuple[date, date]]" = []
        self.openings: "list[date]" = []

    def stretch(self, symbol, first, before, key, secret, min_bars=H.MIN_BARS_PER_SESSION):
        self.stretches.append((first, before))
        sip, iex = _frames(self.tape, [d for d in self.tape if first <= d < before])
        return H.stretch_from(sip, iex, min_bars)

    def opening(self, symbol, session_date, n_minutes=H.OPENING_MINUTES, key=None, secret=None):
        day = pd.Timestamp(session_date).date()
        self.openings.append(day)
        _, iex = _frames(self.tape, [day])
        return H.opening_from_bars(iex, day, n_minutes), True


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(H, "HISTORY_DIR", tmp_path / "highlow3m")
    return tmp_path / "highlow3m"


@pytest.fixture
def alpaca(cache_dir, monkeypatch):
    fake = FakeAlpaca(_tape(date(2025, 12, 1), date(2026, 9, 30)))
    monkeypatch.setattr(H, "_fetch_stretch", fake.stretch)
    monkeypatch.setattr(H, "fetch_opening", fake.opening)
    monkeypatch.setattr(H, "_forecast_cache", {})
    return fake


# --- a synthetic option chain ---------------------------------------------------

class FakeChain:
    """Weekly-ish expirations, strikes 75-125, every contract trading every
    session of its last 120 days at a Black-Scholes price off the synthetic
    tape's close. Stands in for the three downloaders and counts the asks."""

    STRIKES = range(75, 130, 5)

    def __init__(self, tape, irx_through: "date | None" = None):
        closes = pd.Series(
            {pd.Timestamp(d): sip[-1]["c"] for d, (sip, _) in tape.items()}, dtype=float,
        )
        self.closes = closes.sort_index()
        days = self.closes.index
        expirations = pd.date_range("2026-05-01", "2027-02-26", freq="14D")
        rows, bars = [], []
        for e in expirations:
            for kind in ("call", "put"):
                for k in self.STRIKES:
                    symbol = f"SYN{e:%y%m%d}{kind[0].upper()}{k:05d}"
                    rows.append({"symbol": symbol, "type": kind, "strike": float(k), "expiration": e})
                    life = days[(days >= e - pd.Timedelta(120, "D")) & (days <= e)]
                    if not len(life):
                        continue
                    spot = self.closes.loc[life].to_numpy()
                    t = np.maximum((e - life).days.to_numpy() / 365, 1e-6)
                    price = H.bsm_price(spot, float(k), t, 0.04, 0.0, 0.3, kind == "call")
                    for d, px, i in zip(life, price, range(len(life))):
                        # Puts trade more than calls, so dealer gamma does not
                        # cancel to float noise.
                        volume = 20 + (i * 7 + k) % 50 + (15 if kind == "put" else 0)
                        bars.append({"date": d, "symbol": symbol, "open": px, "high": px,
                                     "low": px, "close": max(px, 0.01) + 0.05,
                                     "volume": volume, "trades": 1 + i % 4, "vwap": px})
        self.contracts = pd.DataFrame(rows)
        self.bars = pd.DataFrame(bars)
        self.irx_through = irx_through
        self.contract_asks: "list[tuple[date, date]]" = []
        self.bar_asks: "list[tuple[pd.Timestamp, pd.Timestamp, int]]" = []
        self.irx_asks = 0

    def download_contracts(self, underlying, lo, hi, key, secret):
        self.contract_asks.append((lo, hi))
        c = self.contracts
        return c[(c["expiration"] >= pd.Timestamp(lo)) & (c["expiration"] <= pd.Timestamp(hi))].reset_index(drop=True)

    def download_bars(self, symbols, start, end, key, secret):
        self.bar_asks.append((pd.Timestamp(start), pd.Timestamp(end), len(symbols)))
        b = self.bars
        out = b[b["symbol"].isin(symbols) & (b["date"] >= start) & (b["date"] <= end)]
        return out[H._BAR_COLUMNS].reset_index(drop=True)

    def download_irx(self, start, end):
        self.irx_asks += 1
        last = min(end - timedelta(days=1), self.irx_through or end)
        # A function of the date alone, as a real close is.
        return {
            d.strftime("%Y-%m-%d"): 4.0 + 0.001 * (d.toordinal() % 50)
            for d in pd.bdate_range(start, last)
        }


@pytest.fixture
def chain(alpaca, monkeypatch):
    fake = FakeChain(alpaca.tape)
    monkeypatch.setattr(H, "_download_contracts", fake.download_contracts)
    monkeypatch.setattr(H, "_download_option_bars", fake.download_bars)
    monkeypatch.setattr(H, "_download_irx", fake.download_irx)
    monkeypatch.setitem(H.NOTEBOOK_OPTIONS, "SYN", dict(H.NOTEBOOK_OPTIONS["AAPL"]))
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET", "s")
    return fake


def _assert_rows_equal(got: dict, want: dict) -> None:
    """Row dicts equal field by field, NaN equal to NaN (a missing gamma flip)."""
    assert got.keys() == want.keys()
    for day in want:
        assert got[day].keys() == want[day].keys()
        for key, value in want[day].items():
            np.testing.assert_array_equal(np.asarray(got[day][key], float),
                                          np.asarray(value, float), err_msg=f"{day} {key}")


# --- a stand-in model -------------------------------------------------------------

class Constant:
    """A shipped component that answers one excursion pair whatever it is
    shown, and remembers the rows it was shown."""

    kind = "lgbm"
    cols = list(H.FEATURE_COLS)

    def __init__(self):
        self.rows = []

    def predict(self, X):
        self.rows.append(X)
        return pd.DataFrame({"up": 1.0, "down": 0.5}, index=X.index)


def _model(cols, component):
    model = H.HighLowModel()
    model.components, model.weights, model.feature_cols, model.metadata = (
        {"c": component}, {"c": 1.0}, list(cols), {},
    )
    return model


def _bundle(cols=("or_down", "prev_rv", "down_hist14", "mom126", "iv1d_move", "gex_open_adv")):
    component = Constant()
    bundle = {"kind": "highlow3m", "model": _model(cols, component), "opening_minutes": 3,
              "path": "synthetic", "reads_options": True}
    return bundle, component


def _option_row(spot: float) -> dict:
    """One previous close's positioning row, every field a plausible number."""
    row = {
        "spot": spot, "call_wall": spot * 1.05, "put_wall": spot * 0.95,
        "call_wall_7d": spot * 1.02, "put_wall_7d": spot * 0.98,
        "call_gamma_wall": spot * 1.01, "put_gamma_wall": spot * 0.99,
        "net_gex": 1.0e9, "gross_gex": 2.0e9, "gex_ratio": 0.5, "gamma_flip": np.nan,
        "iv_1d": 0.25, "iv_30d": 0.24, "call_volume": 5.0e5, "put_volume": 3.0e5,
        "call_oi": 1.0e6, "put_oi": 5.0e5, "next_is_opex": 0.0,
    }
    row["profile_net"] = [1.0e9 * (1 - 4 * m) for m in H.GEX_GRID]
    row["profile_gross"] = [2.0e9] * len(H.GEX_GRID)
    return row


def _stub_options(monkeypatch, alpaca):
    """`positioning_rows` answering a fixed row for whatever close it is asked."""
    asked = []

    def rows(symbol, days, key=None, secret=None, min_bars=H.MIN_BARS_PER_SESSION):
        asked.extend(pd.Timestamp(d).date() for d in days)
        return {f"{pd.Timestamp(d):%Y-%m-%d}": _option_row(100.0) for d in days}

    monkeypatch.setattr(H, "positioning_rows", rows)
    return asked


# --- the registry ---------------------------------------------------------------

class TestRegistry:
    def test_highlow3m_is_offered_where_a_bundle_was_saved(self):
        assert apple_models.HIGHLOW3M_KEY in apple_models.keys_for("AAPL")
        assert apple_models.HIGHLOW3M_KEY not in apple_models.keys_for("INTC")

    def test_it_drives_the_day_range_rules(self):
        assert apple_models.strategy(apple_models.HIGHLOW3M_KEY) == apple_models.STRATEGY_DAYRANGE

    def test_every_ticker_it_covers_has_the_notebooks_option_settings(self):
        for ticker in apple_models.HIGHLOW3M_TICKERS:
            assert ticker in H.NOTEBOOK_OPTIONS


# --- the history and this morning ------------------------------------------------------

class TestHistoryCache:
    def test_a_cold_cache_fetches_the_whole_window_once(self, alpaca):
        day = date(2026, 9, 14)
        history = H.history_inputs("AAPL", day, "k", "s")
        assert alpaca.stretches == [(day - timedelta(days=H.HISTORY_CALENDAR_DAYS), day)]
        assert history.sessions.index.max() < pd.Timestamp(day)
        assert len(history.sessions) >= H.MIN_PRIOR_SESSIONS
        # The targets the 14-session averages read, and every close.
        assert history.sessions[["rest_high", "rest_low", "close3"]].notna().all().all()
        assert history.closes.index.equals(history.sessions.index)

        H.history_inputs("AAPL", day, "k", "s")
        assert len(alpaca.stretches) == 1  # served from disk

    def test_the_next_morning_fetches_only_the_new_session(self, alpaca):
        H.history_inputs("AAPL", date(2026, 9, 14), "k", "s")
        history = H.history_inputs("AAPL", date(2026, 9, 15), "k", "s")
        assert alpaca.stretches[-1] == (date(2026, 9, 11), date(2026, 9, 15))
        assert history.sessions.index[-1] == pd.Timestamp("2026-09-14")

    def test_a_split_rebuilds_the_cache_rather_than_mixing_scales(self, alpaca, monkeypatch):
        H.history_inputs("AAPL", date(2026, 9, 14), "k", "s")
        after_split = FakeAlpaca(_tape(date(2025, 12, 1), date(2026, 9, 30), scale=0.25))
        monkeypatch.setattr(H, "_fetch_stretch", after_split.stretch)
        H.history_inputs("AAPL", date(2026, 9, 15), "k", "s")
        assert after_split.stretches[-1] == (
            date(2026, 9, 15) - timedelta(days=H.HISTORY_CALENDAR_DAYS), date(2026, 9, 15)
        )

    def test_highlow2s_cache_is_not_this_ones(self):
        """Its rows hold five-minute openings and no rest-of-session extremes."""
        hl2 = pytest.importorskip("agent_stonks.highlow2_model")
        assert H.HISTORY_DIR != hl2.HISTORY_DIR

    def test_no_credentials_is_a_refusal_not_a_guess(self, cache_dir, monkeypatch):
        monkeypatch.delenv("ALPACA_API_KEY", raising=False)
        monkeypatch.delenv("ALPACA_SECRET", raising=False)
        with pytest.raises(ValueError, match="credentials"):
            H.history_inputs("AAPL", date(2026, 9, 14))


class TestSessionCloses:
    def test_a_half_days_close_is_its_1259_bar_and_it_is_kept(self):
        """The option tables price off every session's close; the models drop
        half days, the closes do not."""
        day = date(2026, 11, 27)  # the day after Thanksgiving
        tape = _tape(date(2026, 11, 23), date(2026, 11, 27))
        sip, iex = _frames(tape, sorted(tape))
        history = H.stretch_from(sip, iex)
        stamp = pd.Timestamp(day)
        assert stamp in history.closes.index and stamp not in history.sessions.index
        assert history.closes[stamp] == pytest.approx(tape[day][0][12 * 60 + 59 - 9 * 60 - 30]["c"])


class TestOpening:
    def test_bars_past_the_933_line_cannot_move_the_opening(self):
        day = date(2026, 9, 14)
        tape = _tape(date(2026, 9, 14), date(2026, 9, 14))
        _, iex = _frames(tape, [day])
        whole = H.opening_from_bars(iex, day)
        window = H.opening_from_bars(iex.iloc[:3], day)
        pd.testing.assert_frame_equal(whole, window)
        assert whole["bars3"].iloc[0] == 3
        assert whole["close3"].iloc[0] == pytest.approx(tape[day][1][2]["c"])

    def test_an_empty_iex_window_is_refused(self):
        day = date(2026, 9, 14)
        _, iex = _frames(_tape(date(2026, 9, 14), date(2026, 9, 14)), [day])
        with pytest.raises(ValueError, match="IEX printed nothing"):
            H.opening_from_bars(iex.iloc[5:], day)


# --- the option tables --------------------------------------------------------------

class TestOptionTables:
    def test_a_close_still_trading_is_refused(self, chain, monkeypatch):
        monkeypatch.setattr(H, "_today_et", lambda: date(2026, 9, 18))
        with pytest.raises(ValueError, match="not final"):
            H.positioning_rows("SYN", [date(2026, 9, 18)])

    def test_a_cold_build_fetches_each_window_once_and_keeps_its_rows(self, chain):
        day = pd.Timestamp("2026-09-17")
        rows = H.positioning_rows("SYN", [day])
        assert set(rows) == {"2026-09-17"}
        row = rows["2026-09-17"]
        assert row["spot"] == pytest.approx(chain.closes[day])
        assert np.isfinite([row["iv_1d"], row["iv_30d"], row["net_gex"], row["call_volume"]]).all()
        # Close to the 0.3 the chain was priced at.
        assert row["iv_30d"] == pytest.approx(0.3, abs=0.02)
        assert len(row["profile_net"]) == len(H.GEX_GRID)
        # Out to everything listed, so the next close is covered by the same list.
        [(lo, hi)] = chain.contract_asks
        assert lo <= day.date() and hi >= (day + pd.Timedelta(120, "D")).date()
        # Every expiration's window starts 120 days before it; none reaches past the close.
        assert all(end <= day for _, end, _ in chain.bar_asks)
        asks = (len(chain.contract_asks), len(chain.bar_asks), chain.irx_asks)

        _assert_rows_equal(H.positioning_rows("SYN", [day]), rows)  # from disk, no fetch
        assert (len(chain.contract_asks), len(chain.bar_asks), chain.irx_asks) == asks
        kept = json.loads((H.HISTORY_DIR / "SYN_positioning.json").read_text())["rows"]
        assert set(kept) == {"2026-09-17"}

    def test_the_next_close_fetches_only_the_new_day(self, chain):
        H.positioning_rows("SYN", ["2026-09-17"])
        before = len(chain.bar_asks)
        H.positioning_rows("SYN", ["2026-09-18"])
        new = chain.bar_asks[before:]
        # Old contracts over the one new day; a strike the move brought into the
        # band, or an expiration entering the 120 days, over its whole window.
        increments = [a for a in new if a[0] == pd.Timestamp("2026-09-18")]
        assert increments and all(end == pd.Timestamp("2026-09-18") for _, end, _ in new)

    def test_a_row_is_the_same_whatever_the_cache_has_seen_since(self, chain, cache_dir, monkeypatch):
        """Point in time: the row of a close built from a cache that already
        reaches a week past it equals the one built that evening."""
        late = H.positioning_rows("SYN", ["2026-09-17", "2026-09-24"])["2026-09-17"]
        monkeypatch.setattr(H, "HISTORY_DIR", cache_dir.parent / "fresh")
        fresh = H.positioning_rows("SYN", ["2026-09-17"])["2026-09-17"]
        _assert_rows_equal({"d": late}, {"d": fresh})

    def test_a_row_waits_for_its_own_rate_before_it_is_kept(self, chain, monkeypatch):
        """Without a T-bill close dated on or after the close, its rate is
        yesterday's carried forward: the row is used, not kept."""
        chain.irx_through = date(2026, 9, 16)
        assert "2026-09-17" in H.positioning_rows("SYN", ["2026-09-17"])
        kept = json.loads((H.HISTORY_DIR / "SYN_positioning.json").read_text())["rows"]
        assert "2026-09-17" not in kept

    def test_closes_months_apart_are_built_apart(self, chain, monkeypatch):
        spans = []
        build = H._build_rows

        def spy(symbol, lo, hi, *a, **k):
            spans.append((lo.date(), hi.date()))
            return build(symbol, lo, hi, *a, **k)

        monkeypatch.setattr(H, "_build_rows", spy)
        H.positioning_rows("SYN", ["2026-06-15", "2026-09-16", "2026-09-17"])
        assert spans == [(date(2026, 6, 15), date(2026, 6, 15)),
                         (date(2026, 9, 16), date(2026, 9, 17))]

    def test_a_ticker_without_the_notebooks_settings_is_refused(self, chain):
        with pytest.raises(ValueError, match="option settings"):
            H.positioning_rows("ZZZ", ["2026-09-17"])

    def test_the_contract_list_is_refetched_once_a_close_is_newer_than_it(self, chain, monkeypatch):
        """A contract listed on a session trades on it, so each new close wants
        a list from after it -- and a replay of older closes needs none."""
        monkeypatch.setattr(H, "_today_et", lambda: date(2026, 9, 18))
        H.positioning_rows("SYN", ["2026-09-17"])
        monkeypatch.setattr(H, "_today_et", lambda: date(2026, 9, 21))
        H.positioning_rows("SYN", ["2026-09-18"])
        assert len(chain.contract_asks) == 2
        H.positioning_rows("SYN", ["2026-09-16"])
        assert len(chain.contract_asks) == 2

    def test_a_dropped_connection_is_retried(self, monkeypatch):
        import requests

        calls = []

        class Ok:
            status_code = 200

            def json(self):
                return {"ok": True}

        def get(*a, **k):
            calls.append(1)
            if len(calls) == 1:
                raise requests.ConnectionError("reset")
            return Ok()

        monkeypatch.setattr(requests, "get", get)
        monkeypatch.setattr(H.time, "sleep", lambda s: None)
        assert H._get_json("https://example.invalid", {}, "k", "s") == {"ok": True}
        assert len(calls) == 2


# --- one session's forecast ------------------------------------------------------------

class TestForecast:
    DAY = date(2026, 9, 14)

    def test_live_the_opening_is_fetched_and_the_answer_memoised(self, alpaca, monkeypatch):
        asked = _stub_options(monkeypatch, alpaca)
        bundle, component = _bundle()
        opening = pd.DataFrame(index=range(3))
        out = H.forecast_session(bundle, "AAPL", opening, self.DAY, "k", "s")
        assert alpaca.openings == [self.DAY]
        # Last night's tables: Friday's close.
        assert asked == [date(2026, 9, 11)]
        row = component.rows[-1]
        close3 = row["close3"].iloc[0]
        adr = row["adr14"].iloc[0]
        assert out["pred_high"] == pytest.approx(close3 * np.exp(1.0 * adr))
        assert out["pred_low"] == pytest.approx(close3 * np.exp(-0.5 * adr))
        assert out["range_after_opening"] is True
        # Never clipped to the window: the range is the one after it.
        assert out["or_high"] == pytest.approx(row["high3"].iloc[0])

        H.forecast_session(bundle, "AAPL", opening, self.DAY, "k", "s")
        assert len(alpaca.openings) == 1

    def test_a_replay_of_a_day_the_cache_covers_reads_its_opening_from_there(self, alpaca, monkeypatch):
        _stub_options(monkeypatch, alpaca)
        H.history_inputs("AAPL", date(2026, 9, 18), "k", "s")  # cache through the 17th
        bundle, _ = _bundle()
        H.forecast_session(bundle, "AAPL", pd.DataFrame(index=range(3)), self.DAY, "k", "s")
        assert alpaca.openings == []

    def test_an_unsettled_live_window_is_not_memoised(self, alpaca, monkeypatch):
        _stub_options(monkeypatch, alpaca)
        monkeypatch.setattr(
            H, "fetch_opening",
            lambda *a, **k: (alpaca.opening(*a, **k)[0], False),
        )
        bundle, _ = _bundle()
        H.forecast_session(bundle, "AAPL", pd.DataFrame(index=range(3)), self.DAY, "k", "s")
        H.forecast_session(bundle, "AAPL", pd.DataFrame(index=range(3)), self.DAY, "k", "s")
        assert len(alpaca.openings) == 2

    def test_nothing_after_the_window_reaches_the_row(self, alpaca, monkeypatch):
        """Today's row reads no target and no SIP bar of today."""
        _stub_options(monkeypatch, alpaca)
        bundle, component = _bundle()
        H.forecast_session(bundle, "AAPL", pd.DataFrame(index=range(3)), self.DAY, "k", "s")
        row = component.rows[-1]
        assert row[["rest_high", "rest_low", "up", "down"]].isna().all().all()

    def test_a_short_history_is_refused(self, alpaca, monkeypatch):
        _stub_options(monkeypatch, alpaca)
        bundle, _ = _bundle()
        with pytest.raises(ValueError, match="sessions of history"):
            H.forecast_session(bundle, "AAPL", pd.DataFrame(index=range(3)), date(2026, 1, 5), "k", "s")

    def test_too_few_opening_bars_is_refused(self, alpaca):
        bundle, _ = _bundle()
        with pytest.raises(ValueError, match="first 3 minutes"):
            H.forecast_session(bundle, "AAPL", pd.DataFrame(index=range(2)), self.DAY, "k", "s")

    def test_a_missing_option_input_is_refused_not_imputed(self, alpaca, monkeypatch):
        def rows(symbol, days, *a, **k):
            return {f"{pd.Timestamp(d):%Y-%m-%d}": {**_option_row(100.0), "iv_1d": float("nan")}
                    for d in days}

        monkeypatch.setattr(H, "positioning_rows", rows)
        bundle, _ = _bundle()
        with pytest.raises(ValueError, match="iv1d_move"):
            H.forecast_session(bundle, "AAPL", pd.DataFrame(index=range(3)), self.DAY, "k", "s")

    def test_a_missing_wall_is_forecast_through(self, alpaca, monkeypatch):
        """The notebook's panel keeps a day without a 7-day wall; so does this."""
        def rows(symbol, days, *a, **k):
            return {f"{pd.Timestamp(d):%Y-%m-%d}": {**_option_row(100.0), "call_wall_7d": float("nan")}
                    for d in days}

        monkeypatch.setattr(H, "positioning_rows", rows)
        bundle, _ = _bundle(cols=("or_down", "call_wall7_dist", "iv1d_move"))
        out = H.forecast_session(bundle, "AAPL", pd.DataFrame(index=range(3)), self.DAY, "k", "s")
        assert np.isfinite(out["pred_high"])


# --- the saved bundle ---------------------------------------------------------------

def _installed_bundle():
    bundle = H.load_bundle("AAPL")
    if bundle is None:
        pytest.skip("the HighLow_3m AAPL bundle is not installed")
    return bundle


class TestBundle:
    def test_the_shipped_blend_is_lightgbm_and_the_linear_median(self):
        bundle = _installed_bundle()
        model = bundle["model"]
        assert bundle["kind"] == "highlow3m" and bundle["reads_options"]
        assert model.weights == {"lgbm": 0.8, "linear": 0.2}
        assert set(model.feature_cols) <= set(H.FEATURE_COLS)
        assert len(model.feature_cols) == 21
        lgbm = model.components["lgbm"]
        assert lgbm.seeds == (7, 11, 23) and len(lgbm.models) == 6
        fill, mean, scale, coef, _ = model.components["linear"]._arrays("up")
        assert fill.shape == mean.shape == scale.shape == coef.shape == (21,)

    def _resaved(self, tmp_path, meta=None, mutate=None):
        import joblib

        src = H.model_path("AAPL")
        if not src.exists():
            pytest.skip("the HighLow_3m AAPL bundle is not installed")
        H._register_unpickle_alias()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # the 1.7.2 pipeline, read from its arrays
            model = joblib.load(src)
        if mutate:
            mutate(model)
        target = tmp_path / "highlow3m_AAPL.joblib"
        joblib.dump(model, target)
        sidecar = json.loads(H.metadata_path(src).read_text())
        sidecar.update(meta or {})
        H.metadata_path(target).write_text(json.dumps(sidecar))
        return target

    def test_a_faithful_copy_loads(self, tmp_path):
        assert H._build_bundle(self._resaved(tmp_path)) is not None

    def test_a_five_minute_opening_is_refused(self, tmp_path):
        assert H._build_bundle(self._resaved(tmp_path, {"opening_minutes": 5})) is None

    def test_a_sip_opening_is_refused(self, tmp_path):
        assert H._build_bundle(self._resaved(tmp_path, {"opening_feed": "sip"})) is None

    def test_a_component_reading_an_unmirrored_column_is_refused(self, tmp_path):
        def mutate(model):
            model.components["lgbm"].cols = [*model.components["lgbm"].cols, "an_overnight"]

        assert H._build_bundle(self._resaved(tmp_path, mutate=mutate)) is None

    def test_an_option_reading_ticker_without_its_settings_is_refused(self, tmp_path):
        assert H._build_bundle(self._resaved(tmp_path, {"ticker": "MSFT"})) is None


# --- the mirror, against the notebook ------------------------------------------------

_NOTEBOOK: dict = {}


def notebook_inputs():
    """The bundle, and the notebook's *raw* minute files run through the live
    path's code, and its option bars, contract list and ^IRX through the
    positioning build: the session cleaning, the half days, the 9:33 cut and
    every option step are all part of what is checked."""
    if _NOTEBOOK:
        return _NOTEBOOK["inputs"]
    data = NOTEBOOK / "data" / "AAPL"
    if not sorted((data / "raw").glob("AAPL_20??_iex.parquet")) or not (data / "options" / "bars").exists():
        pytest.skip("the HighLow_3m AAPL notebook data is not on this machine")
    bundle = _installed_bundle()

    def tape(pattern):
        raw = pd.concat(pd.read_parquet(f) for f in sorted((data / "raw").glob(pattern)))
        return raw.sort_index()[H.OHLCV]

    sip, iex = tape("AAPL_20??.parquet"), tape("AAPL_20??_iex.parquet")
    history = H.stretch_from(sip, iex)
    contracts = pd.read_parquet(data / "options" / "contracts.parquet")[H._CONTRACT_COLUMNS]
    bars = pd.concat(
        [pd.read_parquet(p) for p in sorted((data / "options" / "bars").glob("*.parquet"))],
        ignore_index=True,
    )[H._BAR_COLUMNS]
    irx = pd.read_parquet(NOTEBOOK / "data" / "yahoo" / "_IRX.parquet")["close"]
    table, profile = H.positioning_from(
        contracts, bars, history.closes.loc[:"2026-09-25"], irx, H.NOTEBOOK_OPTIONS["AAPL"],
    )
    _NOTEBOOK["inputs"] = (bundle, sip, iex, history, table, profile)
    return _NOTEBOOK["inputs"]


def _one_day(frame, stamp):
    return frame[frame.index.normalize().tz_localize(None) == stamp]


class TestAgainstTheNotebook:
    """The mirror contract: the notebook's tapes and option bars, through the
    live path's code, reproduce the notebook package's tables and forecasts.
    Same LightGBM, the linear part from its fitted arrays: to the last digit."""

    def test_the_session_closes_and_daily_bars_are_the_notebooks(self):
        _, _, _, history, _, _ = notebook_inputs()
        data = NOTEBOOK / "data" / "AAPL"
        closes = pd.read_parquet(data / "session_closes.parquet")["close"]
        pd.testing.assert_series_equal(history.closes, closes, check_names=False,
                                       check_index_type=False, check_freq=False)
        daily = pd.read_parquet(data / "daily.parquet")
        cols = ["open", "high", "low", "close", "volume", "rv"]
        np.testing.assert_array_equal(history.sessions[cols].to_numpy(float),
                                      daily[cols].to_numpy(float))

    def test_the_option_tables_are_the_notebooks(self):
        _, _, _, _, table, profile = notebook_inputs()
        data = NOTEBOOK / "data" / "AAPL" / "options"
        want = pd.read_parquet(data / "positioning.parquet")
        # Past the notebook's 40-session warm-up, which it blanks and this
        # module does not need (its cache always holds a contract's whole life).
        days = want.index[want["history_sessions"] >= 40]
        cols = [c for c in H._ROW_FIELDS if c != "next_is_opex"]
        np.testing.assert_array_equal(table.loc[days, cols].to_numpy(float),
                                      want.loc[days, cols].to_numpy(float))
        assert (table.loc[days, "next_is_opex"].astype(bool) == want.loc[days, "next_is_opex"]).all()
        got = profile.set_index(["date", "m"]).sort_index()
        ref = pd.read_parquet(data / "gex_profile.parquet").set_index(["date", "m"]).sort_index()
        np.testing.assert_array_equal(got.loc[ref.index].to_numpy(float), ref.to_numpy(float))

    @pytest.mark.parametrize("day", sorted(AAPL_FORECASTS))
    def test_reproduces_the_notebook_forecast(self, day):
        bundle, _, iex, history, table, profile = notebook_inputs()
        stamp = pd.Timestamp(day)
        opening = H.opening_from_bars(_one_day(iex, stamp), stamp)
        last = H.previous_close(history.before(stamp), stamp)
        rows = {f"{last:%Y-%m-%d}": H._row_payload(table, profile, last)}
        positioning, prof = H.positioning_frames(rows)
        out = H.forecast_from(bundle, history, opening, stamp, positioning, prof)
        high, low, adr = AAPL_FORECASTS[day]
        assert out["pred_high"] == pytest.approx(high, abs=1e-9)
        assert out["pred_low"] == pytest.approx(low, abs=1e-9)
        assert out["adr14_abs"] == pytest.approx(adr, abs=1e-9)

    def test_the_cached_opening_is_the_fetched_one(self, cache_dir):
        _, _, iex, history, _, _ = notebook_inputs()
        cache = H._empty_cache()
        H._merge(cache, history, None)
        cache["from"], cache["through"] = "2024-01-02", "2026-09-25"
        H._write_cache("AAPL", cache)
        for day in ("2026-09-10", "2026-09-21"):
            stamp = pd.Timestamp(day)
            cached = H._cached_opening("AAPL", stamp.date())
            fetched = H.opening_from_bars(_one_day(iex, stamp), stamp)
            pd.testing.assert_frame_equal(cached, fetched, check_dtype=False, check_index_type=False)
