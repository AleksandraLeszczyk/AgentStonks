"""HighLow Chain: HighLow2 at 9:35, then HighLow_1h and HighLow_4h every five minutes.

The tests that need no notebook pin the chain's seams: which model's forecast
a decision minute belongs to, the trader swapping each window forecast in at
its decision bar (live, in a restart's catch-up and in the chart's walk alike),
the breach rules measuring against the window forecast after it, a restart
taking the window back, and the refusals said once.

The ones that need the notebooks pin the mirrors: HighLow2_5m's processed
minute files, run through this app's code, must reproduce HighLow_1h's and
HighLow_4h's own panels feature for feature and their shipped models'
forecasts, and HighLow2_5m's cached option bars its own option table -- also
through the live path's assembly (the 70 sessions before the day, the FOMC
history stacked in front).
"""

import glob
import types
from dataclasses import replace
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

W = pytest.importorskip("agent_stonks.highlow_window_model")
O = pytest.importorskip("agent_stonks.highlow2_options")
chain = pytest.importorskip("agent_stonks.highlow_chain_model")

from agent_stonks import apple_models, clock  # noqa: E402
from agent_stonks import apple_trader as at  # noqa: E402
from agent_stonks.decisions import DecisionTracker  # noqa: E402
from agent_stonks.state import AppState  # noqa: E402

from tests.test_apple_trader import FORECAST, FakeBroker, Tape, dayrange_config  # noqa: E402

FINNOTEBOOKS = Path("/Users/aleksandra/Documents/playground/Code/FinNotebooks")
HL2_DATA = FINNOTEBOOKS / "HighLow2_5m" / "data" / "AAPL"
CHAIN_BUNDLE = {"kind": "highlow_chain", "opening_minutes": 5}


def chain_config(**kwargs):
    return replace(dayrange_config(**kwargs), model_key=apple_models.HIGHLOW_CHAIN_KEY)


def window(minute: int, high: float, low: float, price: float) -> dict:
    head = W.head_at(minute)
    return {"head": head.key, "minute": minute, "pred_high": high, "pred_low": low,
            "price": price, "scale": 0.004, "window_end": head.end, "adr14_abs": 10.0}


# --- which model, when ------------------------------------------------------------


class TestDecisionGrid:
    def test_each_minute_belongs_to_the_model_whose_hour_it_is(self):
        assert W.head_at(59) is None                       # 10:29: HighLow2's still
        assert W.head_at(60).key == "1h"                   # 10:30
        assert W.head_at(235).key == "1h"                  # 13:25, the 1h model's last
        assert W.head_at(240).key == "4h"                  # 13:30
        assert W.head_at(385).key == "4h"                  # 15:55, the 4h model's last
        assert W.head_at(62) is None                       # only every fifth minute
        assert W.head_at(390) is None
        assert len(W.HEADS["1h"].decisions) == 36 and len(W.HEADS["4h"].decisions) == 30
        assert chain.decisions() == list(range(60, 390, 5))

    def test_a_bar_is_read_as_minutes_after_the_open(self):
        assert chain.bar_minute(pd.Timestamp("2026-07-21 10:29", tz="America/New_York")) == 59
        assert chain.bar_minute(pd.Timestamp("2026-07-21 14:29", tz="UTC")) == 59

    def test_a_half_day_has_no_window_forecasts(self):
        with pytest.raises(ValueError, match="13:00"):
            W.session_forecasts({"1h": {"path": "x"}}, "AAPL", "2026-11-27")

    def test_the_fomc_days_run_past_the_calendar(self):
        days = W.fomc_dates()
        assert pd.Timestamp("2026-09-16") in days          # the calendar's last
        assert pd.Timestamp("2026-10-28") in days and pd.Timestamp("2026-12-09") in days
        assert pd.Timestamp("2024-01-31") in days

    def test_the_option_table_read_is_the_previous_option_session(self):
        # The Friday after Thanksgiving closes at 13:00 but is an option session.
        assert O.previous_session("2026-11-30") == pd.Timestamp("2026-11-27")
        assert O.previous_session("2026-11-27") == pd.Timestamp("2026-11-25")


class TestRegistry:
    def test_the_chain_exists_where_all_three_models_do(self):
        assert apple_models.covers(apple_models.HIGHLOW_CHAIN_KEY, "AAPL")
        assert not apple_models.covers(apple_models.HIGHLOW_CHAIN_KEY, "GOOG")
        assert apple_models.strategy(apple_models.HIGHLOW_CHAIN_KEY) == apple_models.STRATEGY_DAYRANGE

    def test_tuning_fingerprints_all_three_models(self):
        names = {p.stem for p in apple_models.bundle_files(apple_models.HIGHLOW_CHAIN_KEY, "AAPL")}
        assert {"highlow2_5m_AAPL", "highlow_1h_AAPL", "highlow_4h_AAPL"} <= names

    def test_a_missing_window_model_is_named(self, monkeypatch):
        monkeypatch.setattr(W, "load_bundle", lambda ticker=None, head="1h": None if head == "4h" else {})
        monkeypatch.setattr(chain.highlow2_model, "load_bundle", lambda ticker=None: {"path": "x"})
        assert chain.load_bundle("AAPL") is None
        why = chain.unavailable_reason("AAPL")
        assert "HighLow_4h" in why and "HighLow_1h" not in why

    def test_a_bundle_reading_a_feature_not_mirrored_is_refused(self, monkeypatch, tmp_path):
        model = types.SimpleNamespace(feature_cols=["minutes_left", "spy_gap"], models={"up": [], "down": []})
        monkeypatch.setattr("joblib.load", lambda path: types.SimpleNamespace(model=model, metadata={}))
        assert W._build_bundle(tmp_path / "x.joblib", W.HEADS["4h"]) is None
        model.feature_cols = ["minutes_left", "hist_up"]
        assert W._build_bundle(tmp_path / "x.joblib", W.HEADS["4h"]) is not None


# --- the trader: a window forecast swapped in at its decision bar -------------------


OPEN = pd.Timestamp("2026-07-21 09:30", tz="America/New_York")


def session(minutes: int, close: float = 100.0, spikes: "dict[int, float] | None" = None) -> pd.DataFrame:
    """`minutes` bars from 09:30, flat at `close`, a bar's high raised where
    `spikes` says."""
    rows = []
    for i in range(minutes):
        high = (spikes or {}).get(i, close + 0.05)
        rows.append({"open": close, "high": high, "low": close - 0.05, "close": close, "volume": 1e5})
    index = pd.DatetimeIndex([OPEN + pd.Timedelta(minutes=i) for i in range(minutes)])
    return at.momentum_regime.add_session_columns(pd.DataFrame(rows, index=index))


DAY = {**FORECAST, "pred_high": 104.0, "pred_low": 96.0, "or_high": 100.05, "or_low": 99.95, "chain": True}


class TestChartWalk:
    """`session_levels`: the one walk the chart, a restart's catch-up and the
    live trader share."""

    def test_each_window_forecast_takes_over_at_its_decision_bar(self):
        windows = {60: window(60, 101.5, 99.0, 100.0), 65: window(65, 101.2, 99.2, 100.0),
                   240: window(240, 100.8, 99.5, 100.0)}
        frame = session(245)
        rows = at.session_levels(chain_config(), {**DAY, "windows": windows}, frame, frame.index[4])
        by_minute = {chain.bar_minute(r["t"]): r for r in rows}
        assert (by_minute[58]["pred_high"], by_minute[58]["pred_low"]) == (104.0, 96.0)
        # Made at the close of the 10:29 bar, for the bars from 10:30.
        assert (by_minute[59]["pred_high"], by_minute[59]["pred_low"]) == (101.5, 99.0)
        assert by_minute[59]["window"]["head"] == "1h" and by_minute[59]["window"]["minute"] == 60
        assert by_minute[63]["pred_high"] == 101.5
        assert by_minute[64]["pred_high"] == 101.2
        # The 1h model's last forecast stands until the 4h model's first.
        assert by_minute[238]["pred_high"] == 101.2
        assert by_minute[239]["pred_high"] == 100.8 and by_minute[239]["window"]["head"] == "4h"
        # The levels hang off each: buy = H - 0.75 ADR.
        assert by_minute[59]["buy"] == pytest.approx(101.5 - 7.5)

    def test_a_breach_is_measured_against_the_window_forecast(self):
        # The 10:31 bar trades through the 10:30 forecast's high: under "shift"
        # the range follows it, from the window forecast, not from 9:35's.
        frame = session(66, spikes={61: 102.0})
        rows = at.session_levels(
            chain_config(breach_update="shift"),
            {**DAY, "windows": {60: window(60, 101.5, 99.0, 100.0)}}, frame, frame.index[4],
        )
        by_minute = {chain.bar_minute(r["t"]): r for r in rows}
        assert by_minute[60]["pred_high"] == 101.5
        assert by_minute[61]["pred_high"] == pytest.approx(102.0)
        assert by_minute[61]["pred_low"] == pytest.approx(99.5)     # shifted by the same 0.50

    def test_the_opening_extremes_are_no_breach_of_a_window_forecast(self):
        # The morning traded up to 103 before 10:30; the window forecast from
        # 100 is not "breached" by it.
        frame = session(64, spikes={20: 103.0})
        rows = at.session_levels(
            chain_config(breach_update="shift", contain_range=True),
            {**DAY, "windows": {60: window(60, 101.5, 99.0, 100.0)}}, frame, frame.index[4],
        )
        assert rows[-1]["pred_high"] == 101.5 and rows[-1]["pred_low"] == 99.0

    def test_a_restart_takes_the_window_forecast_back(self):
        frame = session(70, spikes={63: 101.3})
        trader = at.DayRangeTrader(chain_config(breach_update="shift"))
        trader.plan = {"date": OPEN.normalize(), "opening_end": frame.index[4], "open_price": 100.0, **DAY}
        made = window(60, 101.5, 99.0, 100.0)
        trader.plan["windows"] = {60: made}
        trader._set_levels(frame.index[4])
        for ts in frame.index[5:62]:
            trader._move_range(frame[frame.index <= ts], ts)
        row = at._levels_row(trader.config, trader.plan, frame.index[61])

        again = at.DayRangeTrader(chain_config(breach_update="shift"))
        again.plan = {"date": OPEN.normalize(), "opening_end": frame.index[4], "open_price": 100.0, **DAY}
        assert again._resume_range(row, frame)
        assert again.plan["window"]["minute"] == 60
        assert again.plan["forecast_width"] == pytest.approx(2.5)
        # The morning before 10:30 is not read again; the bars after it are.
        assert again.plan["seen_to"] == frame.index[61]
        assert (again.plan["pred_high"], again.plan["pred_low"]) == (101.5, 99.0)


class TestTrader:
    @pytest.fixture
    def run(self, monkeypatch):
        clock.set_simulated(datetime(2026, 7, 21, 14, 31, tzinfo=timezone.utc))
        state = AppState()
        state.set_symbols(["AAPL"])
        state.api_key, state.api_secret, state.feed = "k", "s", "iex"
        tape = Tape(monkeypatch)
        monkeypatch.setattr(at, "session_forecast", lambda *a, **k: ({**DAY}, None))
        calls = []

        def windows(bundle, ticker, day, through, key=None, secret=None, wait=True):
            calls.append(through)
            return {m: window(m, 101.5, 99.0, 100.0) for m in chain.decisions() if m - 1 <= through}, {}

        monkeypatch.setattr(chain, "window_forecasts", windows)
        tracker = DecisionTracker(starting_cash=10_000.0, broker=FakeBroker(100.0))
        yield state, tape, tracker, calls, monkeypatch
        clock.clear()

    def test_the_decision_bar_swaps_the_window_forecast_in_and_says_so(self, run):
        state, tape, tracker, calls, _ = run
        trader = at.DayRangeTrader(chain_config())
        tape.append(100.0, offset=58)
        trader.run_cycle(CHAIN_BUNDLE, state, tracker)
        assert trader.plan["pred_high"] == 104.0
        assert "made again every five minutes" in " ".join(e.get("text", "") for e in state.agent_log)
        tape.append(100.0, offset=59)
        trader.run_cycle(CHAIN_BUNDLE, state, tracker)
        assert (trader.plan["pred_high"], trader.plan["pred_low"]) == (101.5, 99.0)
        assert trader.plan["buy_level"] == pytest.approx(101.5 - 7.5)
        said = [e["text"] for e in state.agent_log if "made again at 10:30" in e.get("text", "")]
        assert len(said) == 1 and "HighLow_1h" in said[0] and "to 13:30" in said[0]
        assert state.apple_trader_levels["rows"][-1]["window"]["minute"] == 60
        # Asked once the 10:29 bar closed, and not again for the same bar.
        trader.run_cycle(CHAIN_BUNDLE, state, tracker)
        assert calls.count(59) == 1

    def test_a_run_started_late_catches_up_on_every_window_forecast(self, run):
        state, tape, tracker, calls, _ = run
        for offset in range(5, 72):
            tape.append(100.0, offset=offset)
        trader = at.DayRangeTrader(chain_config())
        trader.run_cycle(CHAIN_BUNDLE, state, tracker)
        rows = state.apple_trader_levels["rows"]
        made = sorted({r["window"]["minute"] for r in rows if r.get("window")})
        assert made == [60, 65, 70]
        assert any("made again by HighLow_1h at 10:30, 10:35, 10:40" in e.get("text", "")
                   for e in state.agent_log)

    def test_a_session_without_window_forecasts_keeps_highlow2s_and_says_so_once(self, run):
        state, tape, tracker, calls, monkeypatch = run

        def refused(*a, **k):
            calls.append("refused")
            raise ValueError("only 40 complete SIP sessions of history")

        monkeypatch.setattr(chain, "window_forecasts", refused)
        trader = at.DayRangeTrader(chain_config())
        for offset in (59, 64, 69):
            tape.append(100.0, offset=offset)
            trader.run_cycle(CHAIN_BUNDLE, state, tracker)
        assert trader.plan["pred_high"] == 104.0
        assert calls.count("refused") == 1
        said = [e for e in state.agent_log if "makes no HighLow_1h / HighLow_4h" in e.get("text", "")]
        assert len(said) == 1 and "only 40 complete" in said[0]["text"]

    def test_a_decision_without_a_forecast_is_said_once_per_reason(self, run):
        state, tape, tracker, calls, monkeypatch = run
        why = "last night's option table could not be read: timeout."

        def no_options(bundle, ticker, day, through, key=None, secret=None, wait=True):
            got = {m: window(m, 101.5, 99.0, 100.0) for m in chain.decisions() if m - 1 <= min(through, 239)
                   and m < 240}
            return got, {m: why for m in chain.decisions() if 240 <= m <= through + 1}

        monkeypatch.setattr(chain, "window_forecasts", no_options)
        trader = at.DayRangeTrader(chain_config())
        for offset in (234, 239, 244):
            tape.append(100.0, offset=offset)
            trader.run_cycle(CHAIN_BUNDLE, state, tracker)
        # The 1h model's last, made at 13:25, stands past 13:30.
        assert trader.plan["window"]["head"] == "1h" and trader.plan["window"]["minute"] == 235
        said = [e for e in state.agent_log if "could not forecast" in e.get("text", "")]
        assert len(said) == 1 and "HighLow_4h" in said[0]["text"] and "13:30" in said[0]["text"]


# --- the mirrors, against the notebooks' own files ------------------------------------


needs_notebooks = pytest.mark.skipif(
    not (HL2_DATA / "minute.parquet").exists()
    or not (FINNOTEBOOKS / "HighLow_1h" / "data" / "panel_AAPL.parquet").exists()
    or not (FINNOTEBOOKS / "HighLow_4h" / "data" / "panel_AAPL.parquet").exists()
    or W.load_bundle("AAPL", "1h") is None or W.load_bundle("AAPL", "4h") is None,
    reason="the HighLow2_5m / HighLow_1h / HighLow_4h notebooks' data or the bundles are not here",
)

HISTORY_END = "2026-09-25"


@lru_cache(maxsize=None)
def _notebook_inputs():
    sip = pd.read_parquet(HL2_DATA / "minute.parquet")
    iex = pd.read_parquet(HL2_DATA / "minute_iex.parquet")
    daily = pd.read_parquet(HL2_DATA / "daily.parquet").loc[:HISTORY_END]
    pre = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(HL2_DATA / "raw" / "AAPL_*_pre_sip.parquet")))])
    pre = W.premarket_frame(pre.sort_index())
    pos = pd.read_parquet(HL2_DATA / "options" / "positioning.parquet")
    return (sip[sip["date"] <= HISTORY_END], iex[iex["date"] <= HISTORY_END], daily,
            pre[pre["date"] <= HISTORY_END], pos)


def _notebook_context():
    """The notebooks' own `_context`: every session from 2024, the FOMC curve
    on every FOMC day."""
    sip, iex, daily, pre, pos = _notebook_inputs()
    dates = pd.DatetimeIndex(daily.index, name="date")
    ctx = W._context(daily, sip, iex, pre, dates, None, _previous_rows(pos, dates))
    events = W.fomc_dates()
    ctx.hist_var_event = W.event_history(ctx.tape.c_var, ctx.hist_var, ctx.day["adr14"].to_numpy(),
                                         dates, events[events <= HISTORY_END])
    return ctx


@lru_cache(maxsize=None)
def _notebook_panel(head: str) -> pd.DataFrame:
    folder = {"1h": "HighLow_1h", "4h": "HighLow_4h"}[head]
    return pd.read_parquet(FINNOTEBOOKS / folder / "data" / "panel_AAPL.parquet")


def _previous_rows(table: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """The notebook's `previous_row`."""
    left = pd.DataFrame({"session": dates})
    right = table.sort_index().rename_axis("asof").reset_index()
    out = pd.merge_asof(left, right, left_on="session", right_on="asof", allow_exact_matches=False)
    return out.set_index("session").rename_axis("date")


@needs_notebooks
class TestMirror:
    def test_the_panels_are_the_notebooks_feature_for_feature(self):
        ctx = _notebook_context()
        for head in ("1h", "4h"):
            want = _notebook_panel(head)
            got = W.build_panel(ctx, W.HEADS[head], W.HEADS[head].decisions).loc[want.index]
            for col in W.load_bundle("AAPL", head)["feature_cols"] + ["price", "scale"]:
                assert np.array_equal(got[col].to_numpy(float), want[col].to_numpy(float), equal_nan=True), (head, col)

    def test_the_shipped_models_forecast_the_notebooks_test_window_exactly(self):
        # The test window has an FOMC afternoon, 2026-09-16.
        ctx = _notebook_context()
        for head, folder in (("1h", "HighLow_1h"), ("4h", "HighLow_4h")):
            fc = pd.read_parquet(FINNOTEBOOKS / folder / "data" / "forecasts_AAPL.parquet")
            shipped = fc[fc["source"] == "shipped model"]
            assert len(shipped) > 400
            panel = W.build_panel(ctx, W.HEADS[head], W.HEADS[head].decisions)
            got = W.load_bundle("AAPL", head)["model"].predict_prices(panel.loc[shipped.index])
            col = f"hl{head}"
            assert np.array_equal(got["pred_high"].to_numpy(), shipped[f"{col}_pred_high"].to_numpy())
            assert np.array_equal(got["pred_low"].to_numpy(), shipped[f"{col}_pred_low"].to_numpy())

    @pytest.mark.parametrize("day", ["2026-09-08", "2026-09-16", "2026-09-22", "2025-03-19", "2025-12-10"])
    def test_the_live_assembly_reads_the_same_rows(self, day):
        """The live path's context: the 70 sessions before the day with their
        tape, the day's own, last night's option row, and on an FOMC day (09-16,
        2025-03-19, 2025-12-10) every FOMC session since 2024 stacked in front."""
        sip, iex, daily, pre, pos = _notebook_inputs()
        day = pd.Timestamp(day)
        before = daily[daily.index < day]
        d = pd.concat([before, pd.DataFrame(index=pd.DatetimeIndex([day], name="date"))])
        dates = before.index[-W.HIST_SESSIONS:].append(pd.DatetimeIndex([day])).rename("date")
        opt = pd.DataFrame([pos.loc[pos.index[pos.index < day].max()].to_dict()],
                           index=pd.DatetimeIndex([day], name="date"))
        events = None
        if day in W.fomc_dates():
            whole = W.daily_features(daily)
            past = [e for e in W.fomc_dates() if pd.Timestamp(W.EVENT_HISTORY_START) <= e < day]

            def cvar(e):
                one = pd.DatetimeIndex([e])
                tape = W.LiveTape(W.to_tape(sip[sip["date"] == e], one), W.to_tape(iex.iloc[:0], one),
                                  np.full((1, W.PRE_MINUTES), np.nan), np.full(1, np.nan))
                return tape.c_var[0]

            events = (pd.DatetimeIndex(past, name="date"), np.array([cvar(e) for e in past]),
                      np.array([whole.at[e, "adr14"] for e in past]))
        ctx = W._context(d, sip[sip["date"].isin(dates)], iex[iex["date"].isin(dates)],
                         pre[pre["date"] == day], dates, events, opt)
        for head in ("1h", "4h"):
            want = _notebook_panel(head).xs(day, level="date")
            got = W.build_panel(ctx, W.HEADS[head], W.HEADS[head].decisions).xs(day, level="date")
            for col in W.load_bundle("AAPL", head)["feature_cols"]:
                assert np.array_equal(got.loc[want.index, col].to_numpy(float), want[col].to_numpy(float),
                                      equal_nan=True), (head, col)


@pytest.mark.skipif(not (HL2_DATA / "options" / "positioning.parquet").exists(),
                    reason="HighLow2_5m's option data is not here")
def test_the_option_table_is_the_notebooks():
    """HighLow2_5m's cached option bars and contract list, priced with its
    calibration, reproduce its positioning table on every close."""
    from agent_stonks import option_positioning as op

    contracts = pd.read_parquet(HL2_DATA / "options" / "contracts.parquet")[["symbol", "type", "strike", "expiration"]]
    bars = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(HL2_DATA / "options" / "bars" / "*.parquet")))],
                     ignore_index=True)
    raw = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(HL2_DATA / "raw" / "AAPL_20??.parquet")))])
    closes = op.session_closes(raw.sort_index().between_time("09:30", "15:59"))
    want = pd.read_parquet(HL2_DATA / "options" / "positioning.parquet")
    closes = closes.loc[want.index.min():want.index.max()]
    assert np.array_equal(closes.to_numpy(), want["spot"].to_numpy())
    irx_path = FINNOTEBOOKS / "HighLow2_5m" / "data" / "yahoo" / "_IRX.parquet"
    irx = pd.read_parquet(irx_path)["close"]
    table = O.positioning_from(contracts, bars, closes, irx, O.NOTEBOOK_OPTIONS["AAPL"])
    for col in O.ROW_FIELDS:
        assert np.array_equal(table[col].to_numpy(float), want[col].reindex(table.index).to_numpy(float),
                              equal_nan=True), col
