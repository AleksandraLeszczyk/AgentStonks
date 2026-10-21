"""The option tables: the session closes' cache, the option chain's cache, the positioning rows.

The tests that need no notebook pin the seams -- which days the caches fetch
and when, an option row being built only from what was known at that close
(and kept only once it is final), and the refusals.

The one that needs the notebook pins the mirror itself: HighLow_3m's own raw
minute files and option bars, run through the live path's code, must
reproduce the tables its `highlow3m` package built.
"""

import json
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

P = pytest.importorskip("agent_stonks.option_positioning")
hl2 = pytest.importorskip("agent_stonks.highlow2_model")

NOTEBOOK = Path("/Users/aleksandra/Documents/playground/Code/FinNotebooks/HighLow_3m")


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
def _tape(first: date, last: date, scale: float = 1.0) -> "dict[date, list]":
    """Deterministic SIP regular-session bars for every NYSE session in
    [first, last]. Read-only."""
    rng = np.random.default_rng(3)
    out, day, base = {}, first, 100.0
    holidays = set(P.NYSE_HOLIDAYS)
    while day <= last:
        if day.weekday() < 5 and day.isoformat() not in holidays:
            midnight = pd.Timestamp(day).tz_localize(TZ)
            out[day], base = _bars(midnight + pd.to_timedelta(_RTH, unit="min"), base, rng, scale)
        day += timedelta(days=1)
    return out


def _frame(tape, days) -> pd.DataFrame:
    return hl2.bars_frame([b for d in days for b in tape[d]])


class FakeAlpaca:
    """Stands in for Alpaca's SIP tape (`_fetch_tape`), counting what is asked."""

    def __init__(self, tape):
        self.tape = tape
        self.stretches: "list[tuple[date, date]]" = []

    def fetch(self, symbol, start, end, key, secret, feed):
        assert feed == "sip"
        first, before = start.date(), end.date()
        self.stretches.append((first, before))
        return _frame(self.tape, [d for d in self.tape if first <= d < before])


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "HISTORY_DIR", tmp_path / "option_positioning")
    return tmp_path / "option_positioning"


@pytest.fixture
def alpaca(cache_dir, monkeypatch):
    fake = FakeAlpaca(_tape(date(2025, 12, 1), date(2026, 9, 30)))
    monkeypatch.setattr(P, "_fetch_tape", fake.fetch)
    return fake


# --- a synthetic option chain ---------------------------------------------------

class FakeChain:
    """Weekly-ish expirations, strikes 75-125, every contract trading every
    session of its last 120 days at a Black-Scholes price off the synthetic
    tape's close. Stands in for the three downloaders and counts the asks."""

    STRIKES = range(75, 130, 5)

    def __init__(self, tape, irx_through: "date | None" = None):
        closes = pd.Series({pd.Timestamp(d): sip[-1]["c"] for d, sip in tape.items()}, dtype=float)
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
                    price = P.bsm_price(spot, float(k), t, 0.04, 0.0, 0.3, kind == "call")
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
        return out[P._BAR_COLUMNS].reset_index(drop=True)

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
    monkeypatch.setattr(P, "_download_contracts", fake.download_contracts)
    monkeypatch.setattr(P, "_download_option_bars", fake.download_bars)
    monkeypatch.setattr(P, "_download_irx", fake.download_irx)
    monkeypatch.setitem(P.NOTEBOOK_OPTIONS, "SYN", dict(P.NOTEBOOK_OPTIONS["AAPL"]))
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


# --- the session closes ---------------------------------------------------------------

class TestClosesCache:
    def test_a_cold_cache_fetches_the_whole_window_once(self, alpaca):
        day = date(2026, 9, 14)
        closes = P.sip_closes("SYN", day, "k", "s")
        assert alpaca.stretches == [(day - timedelta(days=P.HISTORY_CALENDAR_DAYS), day)]
        assert closes.index.max() < pd.Timestamp(day)
        # Every session of the window, reaching past a contract's 120 days.
        assert closes.index.min() <= pd.Timestamp(day) - pd.Timedelta(P.OPTION_MAX_DTE, "D")
        assert closes.notna().all()

        P.sip_closes("SYN", day, "k", "s")
        assert len(alpaca.stretches) == 1  # served from disk

    def test_the_next_morning_fetches_only_the_new_session(self, alpaca):
        P.sip_closes("SYN", date(2026, 9, 14), "k", "s")
        closes = P.sip_closes("SYN", date(2026, 9, 15), "k", "s")
        assert alpaca.stretches[-1] == (date(2026, 9, 11), date(2026, 9, 15))
        assert closes.index[-1] == pd.Timestamp("2026-09-14")

    def test_a_split_rebuilds_the_cache_rather_than_mixing_scales(self, alpaca, monkeypatch):
        P.sip_closes("SYN", date(2026, 9, 14), "k", "s")
        after_split = FakeAlpaca(_tape(date(2025, 12, 1), date(2026, 9, 30), scale=0.25))
        monkeypatch.setattr(P, "_fetch_tape", after_split.fetch)
        closes = P.sip_closes("SYN", date(2026, 9, 15), "k", "s")
        assert after_split.stretches[-1] == (
            date(2026, 9, 15) - timedelta(days=P.HISTORY_CALENDAR_DAYS), date(2026, 9, 15)
        )
        assert closes.max() < 50  # all on the new scale (the old one is ~100)

    def test_no_credentials_is_a_refusal_not_a_guess(self, cache_dir, monkeypatch):
        monkeypatch.delenv("ALPACA_API_KEY", raising=False)
        monkeypatch.delenv("ALPACA_SECRET", raising=False)
        with pytest.raises(ValueError, match="credentials"):
            P.sip_closes("SYN", date(2026, 9, 14))


class TestSessionCloses:
    def test_a_half_days_close_is_its_1259_bar_and_it_is_kept(self):
        """The option tables price off every session's close, half days too."""
        day = date(2026, 11, 27)  # the day after Thanksgiving
        tape = _tape(date(2026, 11, 23), date(2026, 11, 27))
        closes = P.session_closes(_frame(tape, sorted(tape)))
        stamp = pd.Timestamp(day)
        assert stamp in closes.index
        assert closes[stamp] == pytest.approx(tape[day][12 * 60 + 59 - 9 * 60 - 30]["c"])


# --- the option tables --------------------------------------------------------------

class TestOptionTables:
    def test_a_window_with_no_bars_leaves_the_rest_typed(self, chain, monkeypatch):
        """A job Alpaca answers with no bars is an empty frame typed object; in
        pandas 3 it would make the whole fetch object, and the tables built off
        it fail (GoodMoments' AMD row for 2026-10-20)."""
        asks = []

        def download(symbols, start, end, key, secret):
            asks.append(start)
            if len(asks) == 1:
                return pd.DataFrame(columns=P._BAR_COLUMNS)
            return chain.download_bars(symbols, start, end, key, secret)

        monkeypatch.setattr(P, "_download_option_bars", download)
        through = pd.Timestamp("2026-09-17")
        _, bars = P.option_history("SYN", through, through + pd.Timedelta(120, "D"),
                                   chain.closes[chain.closes.index <= through], through)
        assert len(asks) > 1 and len(bars)
        assert bars["volume"].dtype.kind in "iu" and bars["close"].dtype.kind == "f"

    def test_a_close_still_trading_is_refused(self, chain, monkeypatch):
        monkeypatch.setattr(P, "_today_et", lambda: date(2026, 9, 18))
        with pytest.raises(ValueError, match="not final"):
            P.positioning_rows("SYN", [date(2026, 9, 18)])

    def test_a_cold_build_fetches_each_window_once_and_keeps_its_rows(self, chain):
        day = pd.Timestamp("2026-09-17")
        rows = P.positioning_rows("SYN", [day])
        assert set(rows) == {"2026-09-17"}
        row = rows["2026-09-17"]
        assert row["spot"] == pytest.approx(chain.closes[day])
        assert np.isfinite([row["iv_1d"], row["iv_30d"], row["net_gex"], row["call_volume"]]).all()
        # Close to the 0.3 the chain was priced at.
        assert row["iv_30d"] == pytest.approx(0.3, abs=0.02)
        assert len(row["profile_net"]) == len(P.GEX_GRID)
        # Out to everything listed, so the next close is covered by the same list.
        [(lo, hi)] = chain.contract_asks
        assert lo <= day.date() and hi >= (day + pd.Timedelta(120, "D")).date()
        # Every expiration's window starts 120 days before it; none reaches past the close.
        assert all(end <= day for _, end, _ in chain.bar_asks)
        asks = (len(chain.contract_asks), len(chain.bar_asks), chain.irx_asks)

        _assert_rows_equal(P.positioning_rows("SYN", [day]), rows)  # from disk, no fetch
        assert (len(chain.contract_asks), len(chain.bar_asks), chain.irx_asks) == asks
        kept = json.loads((P.HISTORY_DIR / "SYN_positioning.json").read_text())["rows"]
        assert set(kept) == {"2026-09-17"}

    def test_the_next_close_fetches_only_the_new_day(self, chain):
        P.positioning_rows("SYN", ["2026-09-17"])
        before = len(chain.bar_asks)
        P.positioning_rows("SYN", ["2026-09-18"])
        new = chain.bar_asks[before:]
        # Old contracts over the one new day; a strike the move brought into the
        # band, or an expiration entering the 120 days, over its whole window.
        increments = [a for a in new if a[0] == pd.Timestamp("2026-09-18")]
        assert increments and all(end == pd.Timestamp("2026-09-18") for _, end, _ in new)

    def test_a_row_is_the_same_whatever_the_cache_has_seen_since(self, chain, cache_dir, monkeypatch):
        """Point in time: the row of a close built from a cache that already
        reaches a week past it equals the one built that evening."""
        late = P.positioning_rows("SYN", ["2026-09-17", "2026-09-24"])["2026-09-17"]
        monkeypatch.setattr(P, "HISTORY_DIR", cache_dir.parent / "fresh")
        fresh = P.positioning_rows("SYN", ["2026-09-17"])["2026-09-17"]
        _assert_rows_equal({"d": late}, {"d": fresh})

    def test_a_row_waits_for_its_own_rate_before_it_is_kept(self, chain, monkeypatch):
        """Without a T-bill close dated on or after the close, its rate is
        yesterday's carried forward: the row is used, not kept."""
        chain.irx_through = date(2026, 9, 16)
        assert "2026-09-17" in P.positioning_rows("SYN", ["2026-09-17"])
        kept = json.loads((P.HISTORY_DIR / "SYN_positioning.json").read_text())["rows"]
        assert "2026-09-17" not in kept

    def test_closes_months_apart_are_built_apart(self, chain, monkeypatch):
        spans = []
        build = P._build_rows

        def spy(symbol, lo, hi, *a, **k):
            spans.append((lo.date(), hi.date()))
            return build(symbol, lo, hi, *a, **k)

        monkeypatch.setattr(P, "_build_rows", spy)
        P.positioning_rows("SYN", ["2026-06-15", "2026-09-16", "2026-09-17"])
        assert spans == [(date(2026, 6, 15), date(2026, 6, 15)),
                         (date(2026, 9, 16), date(2026, 9, 17))]

    def test_a_ticker_without_the_notebooks_settings_is_refused(self, chain):
        with pytest.raises(ValueError, match="option settings"):
            P.positioning_rows("ZZZ", ["2026-09-17"])

    def test_the_contract_list_is_refetched_once_a_close_is_newer_than_it(self, chain, monkeypatch):
        """A contract listed on a session trades on it, so each new close wants
        a list from after it -- and a replay of older closes needs none."""
        monkeypatch.setattr(P, "_today_et", lambda: date(2026, 9, 18))
        P.positioning_rows("SYN", ["2026-09-17"])
        monkeypatch.setattr(P, "_today_et", lambda: date(2026, 9, 21))
        P.positioning_rows("SYN", ["2026-09-18"])
        assert len(chain.contract_asks) == 2
        P.positioning_rows("SYN", ["2026-09-16"])
        assert len(chain.contract_asks) == 2

    def test_every_option_reading_model_has_the_notebooks_settings(self):
        """Fast&Furious's stop reads these tables; a ticker it covers without
        an entry here would lose its stop to a refusal."""
        from agent_stonks import apple_models

        for ticker in apple_models.FASTFURIOUS_TICKERS:
            assert ticker in P.NOTEBOOK_OPTIONS

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
        monkeypatch.setattr(P.time, "sleep", lambda s: None)
        assert P._get_json("https://example.invalid", {}, "k", "s") == {"ok": True}
        assert len(calls) == 2


# --- the mirror, against the notebook ------------------------------------------------

_NOTEBOOK: dict = {}


def notebook_inputs():
    """The notebook's *raw* SIP minute files run through the live path's
    session closes, and its option bars, contract list and ^IRX through the
    positioning build: the session cleaning, the half days and every option
    step are all part of what is checked."""
    if _NOTEBOOK:
        return _NOTEBOOK["inputs"]
    data = NOTEBOOK / "data" / "AAPL"
    if not sorted((data / "raw").glob("AAPL_20??.parquet")) or not (data / "options" / "bars").exists():
        pytest.skip("the HighLow_3m AAPL notebook data is not on this machine")
    raw = pd.concat(pd.read_parquet(f) for f in sorted((data / "raw").glob("AAPL_20??.parquet")))
    sip = raw.sort_index()[hl2.OHLCV]
    closes = P.session_closes(sip.between_time(P.RTH_START, P.RTH_END))
    contracts = pd.read_parquet(data / "options" / "contracts.parquet")[P._CONTRACT_COLUMNS]
    bars = pd.concat(
        [pd.read_parquet(p) for p in sorted((data / "options" / "bars").glob("*.parquet"))],
        ignore_index=True,
    )[P._BAR_COLUMNS]
    irx = pd.read_parquet(NOTEBOOK / "data" / "yahoo" / "_IRX.parquet")["close"]
    table, profile = P.positioning_from(
        contracts, bars, closes.loc[:"2026-09-25"], irx, P.NOTEBOOK_OPTIONS["AAPL"],
    )
    _NOTEBOOK["inputs"] = (closes, table, profile)
    return _NOTEBOOK["inputs"]


class TestAgainstTheNotebook:
    """The mirror contract: the notebook's tapes and option bars, through the
    live path's code, reproduce the notebook package's tables to the last digit."""

    def test_the_session_closes_are_the_notebooks(self):
        closes, _, _ = notebook_inputs()
        want = pd.read_parquet(NOTEBOOK / "data" / "AAPL" / "session_closes.parquet")["close"]
        pd.testing.assert_series_equal(closes, want, check_names=False,
                                       check_index_type=False, check_freq=False)

    def test_the_option_tables_are_the_notebooks(self):
        _, table, profile = notebook_inputs()
        data = NOTEBOOK / "data" / "AAPL" / "options"
        want = pd.read_parquet(data / "positioning.parquet")
        # Past the notebook's 40-session warm-up, which it blanks and this
        # module does not need (its cache always holds a contract's whole life).
        days = want.index[want["history_sessions"] >= 40]
        cols = [c for c in P._ROW_FIELDS if c != "next_is_opex"]
        np.testing.assert_array_equal(table.loc[days, cols].to_numpy(float),
                                      want.loc[days, cols].to_numpy(float))
        assert (table.loc[days, "next_is_opex"].astype(bool) == want.loc[days, "next_is_opex"]).all()
        got = profile.set_index(["date", "m"]).sort_index()
        ref = pd.read_parquet(data / "gex_profile.parquet").set_index(["date", "m"]).sort_index()
        np.testing.assert_array_equal(got.loc[ref.index].to_numpy(float), ref.to_numpy(float))
