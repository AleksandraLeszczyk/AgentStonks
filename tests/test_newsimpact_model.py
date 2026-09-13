"""The NewsImpact trajectory mirror: pre-release momentum, the seams, the badge.

Two kinds of test here, as for the other model mirrors.

The synthetic ones need neither the saved model nor the notebooks. They pin
what the app has to get right to hand the model correct inputs: bars becoming
NewsImpact's frame, a release outside the session (or in its first minutes, or
before its bars exist) never reaching the model, the news count and the hour,
and the momentum reading only bars from before the release.

The notebook one pins the mirror itself: the notebook's own labelled events,
re-scored from its cached SIP bars through this module, must reproduce the
momentum before every release and the model's probabilities on the
notebook's own feature recipe. It is the test that fails if
`newsimpact/trajectory.py` changes and this copy does not.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from agent_stonks import newsimpact_model as M
from agent_stonks.state import AppState, SymbolState

NOTEBOOK_DATA = Path("/Users/aleksandra/Documents/playground/Code/FinNotebooks/NewsImpact/data/AAPL")
MARKET_TZ = "America/New_York"
PRESTATE_INPUTS = ["state_pre", "z_pre_c", "t_pre_c", "n_prior_news_30m", "hour"]


# --------------------------------------------------------------- synthetic tape

class FakePipeline:
    """Fixed class probabilities; records every frame it is asked about."""

    classes_ = np.array(M.CLASSES, dtype=object)

    def __init__(self, probs=None):
        # "positive" whatever the state before: P(after = pos) >= 0.8.
        self.probs = probs or {"neu->pos": 0.8, "no_change": 0.1, "neu->neg": 0.1}
        self.seen = []

    def predict_proba(self, X):
        self.seen.append(X.copy())
        row = np.array([self.probs.get(c, 0.0) for c in M.CLASSES])
        return np.tile(row, (len(X), 1))


def _bundle(pipeline=None) -> dict:
    return {
        "pipeline": pipeline or FakePipeline(),
        "family": "prestate",
        "inputs": list(PRESTATE_INPUTS),
        "pre": 15,
        "post": 15,
        "threshold": 1.0,
        "basis": "excess",
        "trajectory": dict(M.TRAJECTORY_DEFAULTS),
        "market_symbol": "SPY",
        "min_bars_per_session": 180,
        "trained_on": {},
    }


def _tape(days, seed=0, minutes=390):
    """App-shape minute bars for a stock with beta 1.2 to a synthetic SPY."""
    rng = np.random.default_rng(seed)
    stock, market = [], []
    ps, pm = 200.0, 500.0
    for day in days:
        open_ts = pd.Timestamp(day).tz_localize(MARKET_TZ) + pd.Timedelta(hours=9, minutes=30)
        rm = rng.normal(0, 4e-4, minutes)
        rs = 1.2 * rm + rng.normal(0, 6e-4, minutes)
        for m in range(minutes):
            pm *= np.exp(rm[m])
            ps *= np.exp(rs[m])
            t = (open_ts + pd.Timedelta(minutes=m)).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
            stock.append({"t": t, "o": ps, "h": ps, "l": ps, "c": ps, "v": 1000.0})
            market.append({"t": t, "o": pm, "h": pm, "l": pm, "c": pm, "v": 1000.0})
    return stock, market


def _article(news_id: str, when: str) -> dict:
    ts = pd.Timestamp(when, tz=MARKET_TZ).tz_convert("UTC")
    return {"id": news_id, "created_at": ts.strftime("%Y-%m-%dT%H:%M:%SZ")}


# 45 weekdays ending 2026-09-04 (a synthetic tape, so holidays do not matter).
# A release needs 40 sessions behind it (20 of profile, each netted by a beta
# over 20 more), so 2026-09-04 is scorable and 2026-08-10 (index 25) is not.
DAYS = list(pd.bdate_range(end="2026-09-04", periods=45))
AFTER_TAPE = pd.Timestamp("2026-09-05 12:00", tz=MARKET_TZ)  # a Saturday


@pytest.fixture(scope="module")
def tape():
    stock, market = _tape(DAYS)
    return M.bars_frame(stock), M.bars_frame(market)


class TestBarsAndCalendar:
    def test_bars_frame_keeps_the_regular_session_in_market_time(self):
        bars = [
            {"t": "2026-09-04T13:29:00Z", "o": 1, "h": 1, "l": 1, "c": 1},   # 09:29 ET
            {"t": "2026-09-04T13:30:00Z", "o": 2, "h": 2, "l": 2, "c": 2},   # 09:30
            {"t": "2026-09-04T14:00:00Z", "o": 0, "h": 0, "l": 0, "c": 0},   # no price
            {"t": "2026-09-04T19:59:00Z", "o": 3, "h": 3, "l": 3, "c": 3},   # 15:59
            {"t": "2026-09-04T20:00:00Z", "o": 4, "h": 4, "l": 4, "c": 4},   # 16:00 auction bar
        ]
        frame = M.bars_frame(bars)
        assert [ts.strftime("%H:%M") for ts in frame.index] == ["09:30", "15:59"]
        assert list(frame["close"]) == [2.0, 3.0]
        assert (frame["date"] == pd.Timestamp("2026-09-04")).all()

    def test_half_days_follow_the_exchange_rule(self):
        dates = pd.DatetimeIndex(["2025-11-28", "2025-12-24", "2025-07-03", "2026-07-03", "2026-09-04"])
        # 2026-07-03 is a Friday: the exchange is closed for the observed holiday.
        assert list(M.early_close(dates)) == [True, True, True, False, False]

    def test_today_is_kept_short_and_closes_at_the_bell(self, tape):
        bars, _ = tape
        today = pd.Timestamp("2026-09-04")
        partial = bars[(bars["date"] < today) | (bars.index.strftime("%H:%M") < "10:30")]
        cal = M.session_calendar(partial, 180, today)
        assert today in cal.index                        # 60 bars, under the 180 floor
        assert cal.loc[today, "close_ts"].strftime("%H:%M") == "16:00"
        assert cal.loc[today, "span_bars"] == 390
        # The same short day in the past is dropped, as the notebook drops it.
        assert today not in M.session_calendar(partial, 180, None).index


class TestScoring:
    def test_scores_an_intraday_release(self, tape):
        bars, market = tape
        out = M.score_articles(_bundle(), [_article("a", "2026-09-04 11:00")], bars, market, AFTER_TAPE)
        result = out["a"]
        assert result["status"] == M.STATUS_SCORED
        assert result["label"] == "positive"
        assert result["p_up"] + result["p_flat"] + result["p_down"] == pytest.approx(1.0)
        assert result["state_pre"] in M.STATES
        assert "15-min momentum after the release" in result["reason"]

    def test_what_the_model_cannot_read_never_reaches_it(self, tape):
        bars, market = tape
        pipe = FakePipeline()
        articles = [
            _article("pre", "2026-09-04 08:00"),
            _article("after", "2026-09-04 16:30"),
            _article("weekend", "2026-08-29 11:00"),
            _article("open", "2026-09-04 09:45:30"),
            _article("old", "2026-08-10 11:00"),
        ]
        out = M.score_articles(_bundle(pipe), articles, bars, market, AFTER_TAPE)
        assert {k: v["short"] for k, v in out.items()} == {
            "pre": "premarket",
            "after": "after hours",
            "weekend": "outside session",
            "open": "first 15 min",
            "old": "no history",
        }
        assert all(v["status"] == M.STATUS_NOT_SCORABLE and v["label"] == "unknown" for v in out.values())
        assert pipe.seen == []

    def test_the_first_scorable_minute_is_the_sixteenth(self, tape):
        bars, market = tape
        out = M.score_articles(_bundle(), [_article("a", "2026-09-04 09:46")], bars, market, AFTER_TAPE)
        assert out["a"]["status"] == M.STATUS_SCORED

    def test_a_release_today_waits_for_its_bars(self, tape):
        bars, market = tape
        today = pd.Timestamp("2026-09-04")
        keep = lambda f: f[(f["date"] < today) | (f.index.strftime("%H:%M") < "10:30")]  # noqa: E731
        now = pd.Timestamp("2026-09-04 10:35", tz=MARKET_TZ)
        articles = [
            _article("early", "2026-09-04 10:20"),   # needs the 10:19 bar: in
            _article("late", "2026-09-04 10:34"),    # needs the 10:33 bar: not yet
            _article("future", "2026-09-04 11:00"),
        ]
        out = M.score_articles(_bundle(), articles, keep(bars), keep(market), now)
        assert out["early"]["status"] == M.STATUS_SCORED
        # Not "after hours": today's close is the bell, not its latest bar.
        assert out["late"]["status"] == M.STATUS_PENDING
        assert out["future"]["status"] == M.STATUS_PENDING

    def test_news_count_and_hour_are_the_notebooks(self, tape):
        bars, market = tape
        pipe = FakePipeline()
        articles = [
            _article("premarket", "2026-09-04 08:00"),  # counted, but not within 30 min
            _article("a", "2026-09-04 10:29:59"),
            _article("b", "2026-09-04 11:00"),          # 10:29:59 is just outside [10:30, 11:00)
            _article("c", "2026-09-04 11:10"),
            _article("d", "2026-09-04 11:29:59"),
        ]
        M.score_articles(_bundle(pipe), articles, bars, market, AFTER_TAPE)
        (X,) = pipe.seen
        assert list(X.columns) == PRESTATE_INPUTS
        assert list(X["n_prior_news_30m"]) == [0.0, 0.0, 1.0, 2.0]
        assert list(X["hour"]) == ["10", "11", "11", "11"]

    def test_momentum_reads_only_bars_before_the_release(self, tape):
        bars, market = tape
        release = pd.Timestamp("2026-09-04 11:00", tz=MARKET_TZ)
        article = [_article("a", "2026-09-04 11:00")]
        base = M.score_articles(_bundle(), article, bars, market, AFTER_TAPE)["a"]["z_pre"]

        def bump(frame, from_ts):
            out = frame.copy()
            out.loc[out.index >= from_ts, "close"] *= 1.03
            return out

        after = M.score_articles(
            _bundle(), article, bump(bars, release), bump(market, release), AFTER_TAPE
        )["a"]["z_pre"]
        assert after == base
        # P0 is the close of the bar before the release, so that bar does count.
        p0 = release - pd.Timedelta(minutes=1)
        moved = M.score_articles(_bundle(), article, bump(bars, p0), market, AFTER_TAPE)["a"]["z_pre"]
        assert moved != base

    def test_badge_is_the_most_likely_state_after(self):
        assert M.impact_label(0.1, 0.8, 0.1) == "neutral"
        assert M.impact_label(0.5, 0.3, 0.2) == "positive"
        assert M.impact_label(0.2, 0.3, 0.5) == "negative"
        assert M.impact_label(0.4, 0.4, 0.2) == "neutral"


class TestBundle:
    def _write(self, tmp_path, inputs=PRESTATE_INPUTS, label=True):
        path = tmp_path / "newsimpact_trajectory_TEST.joblib"
        joblib.dump(
            M.Predictor(pipeline=FakePipeline(), family="prestate", classes=M.CLASSES, inputs=list(inputs)),
            path,
        )
        meta = {"settings": {"trajectory": {"vol_lookback_sessions": 10}, "data": {"market_symbol": "QQQ"}}}
        if label:
            meta["label"] = {"pre": 15, "post": 15, "threshold": 1.0, "basis": "excess"}
        path.with_suffix(".json").write_text(json.dumps(meta))
        return path

    def test_reads_the_label_and_settings_from_the_sidecar(self, tmp_path):
        bundle = M._build_bundle(self._write(tmp_path))
        assert (bundle["pre"], bundle["post"], bundle["threshold"], bundle["basis"]) == (15, 15, 1.0, "excess")
        assert bundle["trajectory"]["vol_lookback_sessions"] == 10
        assert bundle["trajectory"]["beta_lookback_sessions"] == 20
        assert bundle["market_symbol"] == "QQQ"
        # excess basis: the profile's 10 sessions, each netted by a 20-session beta
        assert M.history_sessions(bundle) == 30
        assert M.history_sessions({**bundle, "basis": "raw"}) == 10

    def test_refuses_what_it_cannot_use(self, tmp_path):
        assert M._build_bundle(tmp_path / "missing.joblib") is None
        assert M._build_bundle(self._write(tmp_path, label=False)) is None
        # The news-reading family: its text inputs cannot be built here.
        assert M._build_bundle(self._write(tmp_path, inputs=PRESTATE_INPUTS + ["text"])) is None

    def test_auto_uses_the_model_only_where_one_is_fitted(self, monkeypatch):
        monkeypatch.setattr(M, "has_model", lambda ticker: ticker == "AAPL")
        assert M.uses_model("AAPL", M.IMPACT_METHOD_AUTO)
        assert not M.uses_model("MSFT", M.IMPACT_METHOD_AUTO)
        assert not M.uses_model("AAPL", M.IMPACT_METHOD_LLM)


class TestPublishing:
    def test_refresh_merges_and_clear_removes_only_the_models_labels(self, monkeypatch):
        sym_state = SymbolState("AAPL", AppState())
        sym_state.news = [{"id": "1"}, {"id": "2"}]
        sym_state.news_impacts = {"2": "negative"}  # an LLM label
        verdict = {"label": "positive", "source": M.SOURCE_MODEL, "status": M.STATUS_SCORED}
        monkeypatch.setattr(M, "score_symbol_news", lambda *a, **k: {"1": verdict})

        M.refresh_impacts(sym_state, "", "", "sip")
        assert sym_state.news_impacts == {"1": "positive", "2": "negative"}
        assert sym_state.news_impact_details == {"1": verdict}
        assert M.needs_refresh(sym_state)  # "2" has no model verdict yet

        M.clear_model_impacts(sym_state)
        assert sym_state.news_impacts == {"2": "negative"}
        assert sym_state.news_impact_details == {}

    def test_pending_articles_keep_asking_for_a_refresh(self):
        sym_state = SymbolState("AAPL", AppState())
        sym_state.news = [{"id": "1"}]
        sym_state.news_impact_details = {"1": {"source": M.SOURCE_MODEL, "status": M.STATUS_PENDING}}
        assert M.needs_refresh(sym_state)
        sym_state.news_impact_details = {"1": {"source": M.SOURCE_MODEL, "status": M.STATUS_NOT_SCORABLE}}
        assert not M.needs_refresh(sym_state)

    def test_badge_names_the_model_and_carries_its_reason(self):
        from agent_stonks.ui import build_news_html

        news = [
            {"id": "1", "headline": "h1", "summary": "s", "created_at": "2026-09-04T15:00:00Z",
             "url": "u", "source": "benzinga"},
            {"id": "2", "headline": "h2", "summary": "s", "created_at": "2026-09-04T11:00:00Z",
             "url": "u", "source": "benzinga"},
        ]
        details = {
            "1": {"source": "model", "status": "scored", "short": "model", "reason": "up 60% <b>"},
            "2": {"source": "model", "status": "not_scorable", "short": "premarket", "reason": "before"},
        }
        html = build_news_html(news, "AAPL", {"1": "positive", "2": "unknown"}, details)
        assert "positive impact · model" in html
        assert 'title="up 60% &lt;b&gt;"' in html
        assert "premarket · model" in html


# ------------------------------------------------------------- against the notebook

def _notebook_ready() -> bool:
    return (NOTEBOOK_DATA / "events.parquet").exists() and M.load_bundle("AAPL") is not None


@pytest.mark.skipif(not _notebook_ready(), reason="NewsImpact notebook data or the AAPL model is not on this machine")
def test_reproduces_the_notebooks_momentum_and_probabilities():
    bundle = M.load_bundle("AAPL")
    choice = json.loads((NOTEBOOK_DATA / "label_choice.json").read_text())
    assert (bundle["pre"], bundle["threshold"], bundle["basis"]) == (
        choice["pre"], choice["threshold"], choice["basis"]
    )

    events = pd.read_parquet(NOTEBOOK_DATA / "events.parquet")
    news = pd.read_parquet(NOTEBOOK_DATA / "news.parquet")
    cal = pd.read_parquet(NOTEBOOK_DATA / "calendar.parquet")
    bars = {sym: pd.read_parquet(NOTEBOOK_DATA / "bars" / f"{sym}_2026.parquet") for sym in ("AAPL", "SPY")}
    all_ts = np.sort(news["ts"].dt.tz_convert("UTC").astype("int64").to_numpy())

    # From April, so the 45-session window below fits inside the 2026 bar file.
    labelled = events[events["label_ok"] & (events["session_date"] >= "2026-04-01")]
    sessions = sorted(labelled["session_date"].unique())
    picked = sorted(np.random.default_rng(0).choice(len(sessions), size=min(12, len(sessions)), replace=False))
    lookback = M.history_sessions(bundle) + 5

    checked = 0
    for i in picked:
        day = pd.Timestamp(sessions[i])
        pos = cal.index.get_loc(day)
        window = cal.index[pos - lookback: pos + 1]
        frames = {}
        for sym, b in bars.items():
            w = b[b.index.tz_localize(None).normalize().isin(window)]
            frames[sym] = M.bars_frame([
                {"t": ts.tz_convert("UTC").isoformat(), "o": r.open, "h": r.high, "l": r.low, "c": r.close}
                for ts, r in zip(w.index, w.itertuples())
            ])
        start = day.tz_localize(MARKET_TZ)
        day_news = news[(news["ts"] >= start - pd.Timedelta(days=1)) & (news["ts"] < start + pd.Timedelta(days=1))]
        articles = [{"id": str(r.id), "created_at": r.created_at} for r in day_news.itertuples()]

        out = M.score_articles(bundle, articles, frames["AAPL"], frames["SPY"], start + pd.Timedelta(days=1))

        for r in labelled[labelled["session_date"] == day].itertuples():
            got = out[str(r.id)]
            assert got["status"] == M.STATUS_SCORED, got
            assert got["state_pre"] == r.state_pre
            np.testing.assert_allclose(got["z_pre"], r.z_pre, rtol=1e-9)

            # modeling.prepare_frame's recipe, from the notebook's own columns
            t = r.ts.tz_convert("UTC").value
            X = pd.DataFrame({
                "state_pre": [r.state_pre],
                "z_pre_c": [float(np.clip(r.z_pre, -5, 5))],
                "t_pre_c": [0.0 if np.isnan(r.t_pre) else float(np.clip(r.t_pre, -10, 10))],
                "n_prior_news_30m": [float(np.searchsorted(all_ts, t, "left")
                                           - np.searchsorted(all_ts, t - 30 * 60 * 10**9, "left"))],
                "hour": [str(r.ts.hour)],
            })
            P = pd.DataFrame(M.proba_full(bundle["pipeline"], X), columns=list(M.CLASSES))
            post = M.post_state_probs(P, X["state_pre"])
            np.testing.assert_allclose(
                [got["p_up"], got["p_down"]],
                [post["p_post_pos"].iloc[0], post["p_post_neg"].iloc[0]],
                rtol=1e-7,
            )
            checked += 1
    assert checked >= 20
