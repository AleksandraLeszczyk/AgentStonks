import colorsys
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import (
    AVG_LINE_COLORS,
    FIB_LEVELS,
    MA_COLORS,
    NEWS_IMPACT_COLORS,
    NEWS_MARKER_OFFSET_FRAC,
    PALETTE,
)


def empty_chart(msg: str = "Enter a symbol and click Start") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=PALETTE["bg"],
        plot_bgcolor=PALETTE["panel"],
        annotations=[
            dict(
                text=msg,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=16, color=PALETTE["muted"]),
            )
        ],
        margin=dict(l=10, r=10, t=40, b=10),
        height=520,
    )
    return fig


_MIXTURE_COLORS = ["#e0e0e0", "#60a5fa", "#fb923c", "#a78bfa", "#34d399"]


def _gmm_em(
    centers: np.ndarray,
    bin_weights: np.ndarray,
    n: int,
    seed: int,
    max_iter: int = 300,
    tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Weighted EM for a 1-D n-component GMM fitted directly to histogram bins.
    bin_weights are the volume counts per bin.
    Returns (mixing_weights, means, stds, mixture_density_per_bin).
    """
    rng = np.random.default_rng(seed)
    total_w = bin_weights.sum()
    cdf = np.cumsum(bin_weights) / total_w
    q = rng.uniform(0, 1, n)
    means = centers[np.searchsorted(cdf, q)].astype(float)
    w_norm = bin_weights / total_w
    global_mean = float(np.dot(w_norm, centers))
    global_std = float(np.sqrt(np.dot(w_norm, (centers - global_mean) ** 2)))
    stds = np.full(n, max(global_std / n, 1e-6))
    mix = np.ones(n) / n
    log_lik = -np.inf
    density = np.ones(len(centers))

    for _ in range(max_iter):
        # E-step
        resp = np.column_stack([
            mix[k] * np.exp(-0.5 * ((centers - means[k]) / stds[k]) ** 2)
            / (stds[k] * np.sqrt(2 * np.pi))
            for k in range(n)
        ])
        density = resp.sum(axis=1)
        density_safe = np.where(density == 0, 1e-300, density)
        resp /= density_safe[:, None]

        # Weighted M-step
        eff = resp * bin_weights[:, None]
        Nk = eff.sum(axis=0)
        mix = Nk / total_w
        means = (eff * centers[:, None]).sum(axis=0) / np.maximum(Nk, 1e-10)
        stds = np.sqrt((eff * (centers[:, None] - means) ** 2).sum(axis=0) / np.maximum(Nk, 1e-10))
        stds = np.maximum(stds, 1e-6)

        new_log_lik = float((bin_weights * np.log(density_safe)).sum())
        if abs(new_log_lik - log_lik) < tol:
            break
        log_lik = new_log_lik

    return mix, means, stds, density


def _cmm_em(
    centers: np.ndarray,
    bin_weights: np.ndarray,
    n: int,
    seed: int,
    max_iter: int = 300,
    tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Weighted ECM for a 1-D n-component Cauchy mixture fitted directly to histogram
    bins, using the t-distribution EM scheme (Peel & McLachlan) fixed at nu=1.
    Returns (mixing_weights, locations, scales, mixture_density_per_bin).
    """
    rng = np.random.default_rng(seed)
    total_w = bin_weights.sum()
    cdf = np.cumsum(bin_weights) / total_w
    q = rng.uniform(0, 1, n)
    locs = centers[np.searchsorted(cdf, q)].astype(float)
    w_norm = bin_weights / total_w
    global_mean = float(np.dot(w_norm, centers))
    global_std = float(np.sqrt(np.dot(w_norm, (centers - global_mean) ** 2)))
    scales = np.full(n, max(global_std / n, 1e-6))
    mix = np.ones(n) / n
    log_lik = -np.inf
    density = np.ones(len(centers))

    for _ in range(max_iter):
        # E-step
        delta = np.column_stack([((centers - locs[k]) / scales[k]) ** 2 for k in range(n)])
        comp_density = np.column_stack([
            mix[k] / (np.pi * scales[k] * (1.0 + delta[:, k])) for k in range(n)
        ])
        density = comp_density.sum(axis=1)
        density_safe = np.where(density == 0, 1e-300, density)
        resp = comp_density / density_safe[:, None]
        z = resp * bin_weights[:, None]
        Nk = z.sum(axis=0)

        # CM-step 1: update locations using nu=1 t-weights at the current scale
        u = 2.0 / (1.0 + delta)
        eff = z * u
        locs = (eff * centers[:, None]).sum(axis=0) / np.maximum(eff.sum(axis=0), 1e-10)

        # CM-step 2: recompute t-weights at the new locations, then update scales
        delta = np.column_stack([((centers - locs[k]) / scales[k]) ** 2 for k in range(n)])
        u = 2.0 / (1.0 + delta)
        scales_sq = (z * u * (centers[:, None] - locs) ** 2).sum(axis=0) / np.maximum(Nk, 1e-10)
        scales = np.maximum(np.sqrt(scales_sq), 1e-6)
        mix = Nk / total_w

        new_log_lik = float((bin_weights * np.log(density_safe)).sum())
        if abs(new_log_lik - log_lik) < tol:
            break
        log_lik = new_log_lik

    return mix, locs, scales, density


_MIXTURE_EM = {"gaussian": _gmm_em, "cauchy": _cmm_em}


def _mixture_pdf(x: np.ndarray, loc: float, scale: float, dist: str) -> np.ndarray:
    if dist == "cauchy":
        return 1.0 / (np.pi * scale * (1.0 + ((x - loc) / scale) ** 2))
    return np.exp(-0.5 * ((x - loc) / scale) ** 2) / (scale * np.sqrt(2 * np.pi))


def _fit_mixture(
    bin_centers: np.ndarray,
    weights: np.ndarray,
    n_components: int,
    dist: str,
    n_init: int = 5,
) -> list[tuple[float, float, float]]:
    """
    Fit exactly n_components Gaussian or Cauchy distributions to a weighted
    histogram via EM (pure numpy). Runs n_init random restarts and returns
    the best by weighted log-likelihood.
    Returns a list of (mixing_weight, location, scale) tuples.
    """
    if weights.sum() == 0 or bin_centers.std() == 0:
        return []

    em_fn = _MIXTURE_EM[dist]
    best_wll = -np.inf
    best_params: tuple = ()
    for init in range(n_init):
        mix, loc, scale, density = em_fn(bin_centers, weights, n_components, seed=init)
        wll = float((weights * np.log(np.maximum(density, 1e-300))).sum())
        if wll > best_wll:
            best_wll, best_params = wll, (mix, loc, scale)

    mix, loc, scale = best_params
    return [(float(mix[i]), float(loc[i]), float(scale[i])) for i in range(len(mix))]


def _plot_price_distribution(
    df_trades: pd.DataFrame,
    fig: go.Figure,
    n_price_bins: int = 50,
    n_time_buckets: int = 20,
    mixture_distribution: str = "none",
    mixture_max_components: int = 0,
) -> tuple[go.Figure, list[tuple[float, float, float]]]:
    """
    Add a horizontal volume-weighted price distribution histogram to col 2.
    Colors run violet→red from oldest to newest trades.
    """
    df = df_trades.sort_values("t").reset_index(drop=True)

    price_min, price_max = df["p"].min(), df["p"].max()
    bin_edges = np.linspace(price_min, price_max, n_price_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]
    df["bin_idx"] = np.searchsorted(bin_edges[1:-1], df["p"].values)

    n = len(df)
    df["bucket"] = np.clip(
        df.index.to_numpy() * n_time_buckets // max(n, 1), 0, n_time_buckets - 1
    )

    def _hsv_to_rgb(hue_deg: float, s: float = 0.85, v: float = 0.95) -> str:
        r, g, b = colorsys.hsv_to_rgb(hue_deg / 360.0, s, v)
        return f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})"

    bucket_colors = [
        _hsv_to_rgb(270.0 * (1.0 - b / max(n_time_buckets - 1, 1)))
        for b in range(n_time_buckets)
    ]

    agg = (
        df.groupby(["bin_idx", "bucket"])["s"]
        .sum()
        .reindex(
            pd.MultiIndex.from_product(
                [range(n_price_bins), range(n_time_buckets)],
                names=["bin_idx", "bucket"],
            ),
            fill_value=0,
        )
        .reset_index()
    )

    for bucket in range(n_time_buckets):
        bdata = agg[agg["bucket"] == bucket]
        t_lo = bucket / n_time_buckets
        t_hi = (bucket + 1) / n_time_buckets
        fig.add_trace(
            go.Bar(
                orientation="h",
                y=bin_centers[bdata["bin_idx"].values],
                x=bdata["s"].values,
                width=bin_width * 0.9,
                marker_color=bucket_colors[bucket],
                marker_line_width=0,
                showlegend=False,
                name=f"Time {t_lo:.0%}–{t_hi:.0%}",
                hovertemplate=(
                    "<b>Price bin:</b> %{y:.4f}<br>"
                    "<b>Volume:</b> %{x:,.0f}<br>"
                    f"<b>Period:</b> {t_lo:.0%}–{t_hi:.0%} of session"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=2,
        )

    fit_enabled = mixture_distribution in _MIXTURE_EM and mixture_max_components > 0
    components: list[tuple[float, float, float]] = []
    if fit_enabled:
        total_per_bin = (
            agg.groupby("bin_idx")["s"]
            .sum()
            .reindex(range(n_price_bins), fill_value=0)
            .values
        )
        total_vol = total_per_bin.sum()
        if total_vol > 0:
            components = _fit_mixture(bin_centers, total_per_bin, mixture_max_components, mixture_distribution)
            if components:
                price_smooth = np.linspace(price_min, price_max, 400)
                scale = total_vol * bin_width
                prefix = "C" if mixture_distribution == "cauchy" else "G"
                loc_sym = "x₀" if mixture_distribution == "cauchy" else "μ"
                scale_sym = "γ" if mixture_distribution == "cauchy" else "σ"

                if len(components) > 1:
                    mixture_pdf = sum(
                        w * _mixture_pdf(price_smooth, mu, sigma, mixture_distribution)
                        for w, mu, sigma in components
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=mixture_pdf * scale,
                            y=price_smooth,
                            mode="lines",
                            name=f"{prefix}MM envelope",
                            line=dict(color="#ffffff", width=2),
                            opacity=0.9,
                            hovertemplate=(
                                "<b>Price:</b> %{y:.4f}<br>"
                                "<b>Fitted vol:</b> %{x:,.0f}"
                                "<extra></extra>"
                            ),
                        ),
                        row=1,
                        col=2,
                    )

                for i, (w, mu, sigma) in enumerate(components):
                    pdf = w * _mixture_pdf(price_smooth, mu, sigma, mixture_distribution)
                    color = _MIXTURE_COLORS[i % len(_MIXTURE_COLORS)]
                    fig.add_trace(
                        go.Scatter(
                            x=pdf * scale,
                            y=price_smooth,
                            mode="lines",
                            name=f"{prefix}{i + 1}  {loc_sym}={mu:.2f}  {scale_sym}={sigma:.2f}",
                            line=dict(color=color, width=1.5, dash="dot"),
                            opacity=0.85,
                            hovertemplate=(
                                f"<b>{prefix}{i + 1}</b>  {loc_sym}={mu:.2f}  {scale_sym}={sigma:.2f}<br>"
                                "<b>Price:</b> %{y:.4f}<br>"
                                "<b>Fitted vol:</b> %{x:,.0f}"
                                "<extra></extra>"
                            ),
                        ),
                        row=1,
                        col=2,
                    )

    # Invisible scatter to attach a colorbar legend
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(
                colorscale=[
                    [i / (n_time_buckets - 1), bucket_colors[i]]
                    for i in range(n_time_buckets)
                ],
                cmin=0,
                cmax=1,
                color=[0],
                colorbar=dict(
                    tickvals=[0, 0.5, 1],
                    ticktext=["Oldest", "Mid", "Newest"],
                    lenmode="fraction",
                    len=0.5,
                    thickness=12,
                    x=1.01,
                    y=0.5,
                    outlinewidth=0,
                ),
                showscale=True,
            ),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        barmode="stack",
        xaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.15)", zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.15)", tickformat=".4f"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        bargap=0,
        bargroupgap=0,
        height=700,
    )
    return fig, components


def _add_moving_averages(df: pd.DataFrame, fig: go.Figure, periods: list[int]) -> None:
    for period in periods:
        if len(df) < period:
            continue
        vwap = (df["c"] * df["v"]).rolling(window=period).sum() / df["v"].rolling(window=period).sum()
        fig.add_trace(
            go.Scatter(
                x=df["t"],
                y=vwap,
                mode="lines",
                name=f"VWMA({period})",
                line=dict(color=MA_COLORS.get(period, "#ffffff"), width=1.5, dash="solid"),
                opacity=0.85,
            ),
            row=1,
            col=1,
        )


def _add_avg_lines(
    df_all: pd.DataFrame,
    fig: go.Figure,
    show_7d: bool = True,
    show_28d: bool = True,
    show_1y: bool = False,
    x0_dt: Optional[datetime] = None,
    df_daily: Optional[pd.DataFrame] = None,
) -> None:
    """Horizontal VWAP-style average price lines for 7-day, 28-day, and 1-year lookbacks."""
    if df_all.empty:
        return
    latest = df_all["t"].max()
    # Use daily bars for multi-day lookbacks when available so the windows
    # contain the right data regardless of how much intraday history is loaded.
    df_hist = df_daily if (df_daily is not None and not df_daily.empty) else df_all
    lookbacks = [
        ("7d",  pd.Timedelta(days=7),   "7d avg",  show_7d),
        ("28d", pd.Timedelta(days=28),  "28d avg", show_28d),
        ("1y",  pd.Timedelta(days=365), "1y avg",  show_1y),
    ]
    x0 = pd.Timestamp(x0_dt) if x0_dt is not None else df_all["t"].iloc[0]
    x1 = latest
    for key, delta, label, visible in lookbacks:
        if not visible:
            continue
        hist_latest = df_hist["t"].max()
        window = df_hist[df_hist["t"] >= hist_latest - delta]
        if window.empty:
            continue
        total_vol = window["v"].sum()
        if total_vol == 0:
            continue
        avg = (window["c"] * window["v"]).sum() / total_vol
        color = AVG_LINE_COLORS[key]
        fig.add_shape(
            type="line",
            x0=x0, x1=x1,
            y0=avg, y1=avg,
            line=dict(color=color, width=1, dash="dash"),
            row=1, col=1,
        )
        fig.add_annotation(
            xref="x", yref="y",
            x=x1, y=avg,
            text=f" {label} {avg:.2f}",
            font=dict(color=color, size=10, family="monospace"),
            showarrow=False,
            xanchor="left",
        )


def _add_vwap(df: pd.DataFrame, fig: go.Figure, vwap_style: str = "dot", show_candle_body: bool = True) -> None:
    """Per-bar VWAP from Alpaca's vw field. vwap_style: 'line', 'dot', or 'hide'."""
    if "vw" not in df.columns or vwap_style == "hide":
        return

    if vwap_style == "line":
        fig.add_trace(
            go.Scatter(
                x=df["t"],
                y=df["vw"],
                mode="lines",
                name="VWAP",
                line=dict(color="#000000", width=1.5),
                opacity=0.9,
                hovertemplate="<b>VWAP:</b> %{y:.4f}<extra></extra>",
            ),
            row=1, col=1,
        )
    else:
        marker_colors = (
            "#000000"
            if show_candle_body
            else [PALETTE["up"] if c >= o else PALETTE["down"] for c, o in zip(df["c"], df["o"])]
        )
        fig.add_trace(
            go.Scatter(
                x=df["t"],
                y=df["vw"],
                mode="markers",
                name="VWAP",
                marker=dict(symbol="circle", color=marker_colors, size=6),
                opacity=0.9,
                hovertemplate="<b>VWAP:</b> %{y:.4f}<extra></extra>",
            ),
            row=1, col=1,
        )


def _add_fibonacci_levels(df: pd.DataFrame, fig: go.Figure) -> None:
    price_high = df["h"].max()
    price_low = df["l"].min()
    price_range = price_high - price_low
    if price_range == 0:
        return

    x_start = df["t"].iloc[0]
    x_end = df["t"].iloc[-1]

    for ratio, label in FIB_LEVELS:
        level = price_high - ratio * price_range
        fig.add_shape(
            type="line",
            x0=x_start,
            x1=x_end,
            y0=level,
            y1=level,
            line=dict(color="#d4af37", width=1, dash="dot"),
            row=1,
            col=1,
        )
        fig.add_annotation(
            xref="x",
            yref="y",
            x=x_end,
            y=level,
            text=f" {label} {level:.2f}",
            font=dict(color="#d4af37", size=10, family="monospace"),
            showarrow=False,
            xanchor="left",
        )


def _add_price_alerts(price_alerts: list[dict], fig: go.Figure, x0: datetime, x1: datetime) -> None:
    """Horizontal lines marking price-axis levels the agent is waiting to wake up on.

    Expects condition alerts shaped {"field", "condition", "value"} whose field
    lives on the price axis (price/bid/ask/day high/low); other alert fields
    (volume, spread, portfolio value) have no price level and are filtered out
    by the caller.
    """
    for alert in price_alerts:
        level = alert.get("value")
        condition = alert.get("condition")
        if level is None:
            continue
        field = alert.get("field", "price")
        color = PALETTE["up"] if condition == "above" else PALETTE["down"]
        fig.add_shape(
            type="line",
            x0=x0, x1=x1,
            y0=level, y1=level,
            line=dict(color=color, width=1.5, dash="dashdot"),
            row=1, col=1,
        )
        fig.add_annotation(
            xref="x", yref="y",
            x=x1, y=level,
            text=f" ⏰ {field} {condition} {level:.2f}",
            font=dict(color=color, size=10, family="monospace"),
            showarrow=False,
            xanchor="left",
        )


def _add_tactic_levels(tactic_levels: list[dict], fig: go.Figure, x0: datetime, x1: datetime) -> None:
    """Horizontal lines marking price-axis conditions of the agent's armed tactics --
    the levels at which a standing conditional buy/sell will execute.

    Expects entries shaped {"action", "label", "field", "condition", "value"}
    (see tactics.tactic_price_levels); conditions on non-price fields are
    filtered out by that helper.
    """
    for level_spec in tactic_levels:
        level = level_spec.get("value")
        if level is None:
            continue
        color = PALETTE["up"] if level_spec.get("action") == "buy" else PALETTE["down"]
        fig.add_shape(
            type="line",
            x0=x0, x1=x1,
            y0=level, y1=level,
            line=dict(color=color, width=1.5, dash="dot"),
            row=1, col=1,
        )
        fig.add_annotation(
            xref="x", yref="y",
            x=x1, y=level,
            text=(
                f" 🎯 {level_spec.get('label', '')} @ {level_spec.get('field', 'price')} "
                f"{level_spec.get('condition', '')} {level:.2f}"
            ),
            font=dict(color=color, size=10, family="monospace"),
            showarrow=False,
            xanchor="left",
        )


def _add_decision_markers(decisions: list[dict], fig: go.Figure, session_start: datetime) -> None:
    """Plot filled agent buy/sell decisions as markers on the price chart."""
    if not decisions:
        return
    df = pd.DataFrame(decisions)
    if df.empty or "ts" not in df.columns or "price" not in df.columns:
        return
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df[(df["ts"] > session_start) & df["price"].notna()]
    if df.empty:
        return

    markers = (
        ("buy", "triangle-up", PALETTE["up"]),
        ("sell", "triangle-down", PALETTE["down"]),
    )
    for action, marker_symbol, color in markers:
        sub = df[df["action"] == action]
        if sub.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=sub["ts"],
                y=sub["price"],
                mode="markers",
                marker=dict(symbol=marker_symbol, size=14, color=color, line=dict(width=1.5, color="#ffffff")),
                name=f"Agent {action}",
                customdata=sub.get("filled_quantity", sub.get("quantity")),
                hovertemplate=(
                    f"<b>Agent {action}</b><br>Price: %{{y:.4f}}<br>Qty: %{{customdata:.2f}}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )


def _fill_intraday_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Insert synthetic flat bars (o=h=l=c=prev close, v=0) at missing buckets.

    A feed only prints a bar for periods in which it saw at least one trade
    (on IEX that's a small slice of consolidated volume), so thin symbols
    leave holes in the candle/volume series. The grid step is the smallest
    observed bar spacing, applied per calendar day so overnight/weekend gaps
    are never filled. Synthetic rows are flagged in a 'synthetic' column.
    """
    if len(df) < 3:
        return df
    step = df["t"].diff().dropna().min()
    if pd.isna(step) or step <= pd.Timedelta(0):
        return df

    days = []
    for _, day_df in df.groupby(df["t"].dt.date, sort=True):
        idx = pd.DatetimeIndex(day_df["t"])
        grid = pd.date_range(idx[0], idx[-1], freq=step)
        gridded = day_df.set_index("t").reindex(idx.union(grid))
        synth = gridded["c"].isna()
        gridded["c"] = gridded["c"].ffill()
        for col in ("o", "h", "l", "vw"):
            if col in gridded.columns:
                gridded.loc[synth, col] = gridded.loc[synth, "c"]
        gridded["v"] = gridded["v"].fillna(0.0)
        gridded["synthetic"] = synth
        days.append(gridded)
    return pd.concat(days).rename_axis("t").reset_index()


def _bar_width_ms(df: pd.DataFrame) -> float:
    if len(df) < 2:
        return 60_000
    delta_ms = float(df["t"].diff().dropna().dt.total_seconds().median() * 1000)
    return delta_ms * 0.8


def build_chart(
    bars: list[dict],
    news: list[dict],
    trades: list[dict],
    symbol: str,
    session_start: datetime,
    ma_periods: Optional[list] = None,
    show_fib: bool = False,
    show_7d_avg: bool = True,
    show_28d_avg: bool = True,
    show_1y_avg: bool = False,
    mixture_distribution: str = "none",
    mixture_max_components: int = 0,
    daily_bars: Optional[list[dict]] = None,
    vwap_style: str = "hide",
    show_candle_body: bool = True,
    show_percentile_body: bool = False,
    show_whiskers: bool = True,
    decisions: Optional[list[dict]] = None,
    price_alerts: Optional[list[dict]] = None,
    tactic_levels: Optional[list[dict]] = None,
    news_impacts: Optional[dict] = None,
    fill_gaps: bool = False,
) -> go.Figure:
    if not bars:
        return empty_chart("Waiting for data…")

    df_all = pd.DataFrame(bars)
    df_all["t"] = pd.to_datetime(df_all["t"], utc=True)
    df_all = (
        df_all.sort_values("t")
        .drop_duplicates(subset="t", keep="last")
        .reset_index(drop=True)
    )
    df = df_all[df_all["t"] > session_start].reset_index(drop=True)
    if df.empty:
        return empty_chart("Waiting for data…")
    if fill_gaps:
        df = _fill_intraday_gaps(df)

    df_trades = pd.DataFrame.from_records(trades) if trades else pd.DataFrame(columns=["p", "s", "t"])
    if not df_trades.empty:
        df_trades["t"] = pd.to_datetime(df_trades["t"])
        df_trades = df_trades[df_trades["t"] > session_start]

    fig = make_subplots(
        rows=2,
        cols=2,
        shared_xaxes=True,
        shared_yaxes=True,
        vertical_spacing=0.02,
        horizontal_spacing=0.02,
        row_heights=[0.75, 0.25],
        column_widths=[0.8, 0.2],
    )

    if not df_trades.empty:
        fig, mixture_components = _plot_price_distribution(
            df_trades, fig,
            mixture_distribution=mixture_distribution,
            mixture_max_components=mixture_max_components,
        )
    else:
        mixture_components = []

    body_colors = [PALETTE["up"] if c >= o else PALETTE["down"] for c, o in zip(df["c"], df["o"])]
    body_base = [min(o, c) for o, c in zip(df["o"], df["c"])]
    body_height = [abs(c - o) for o, c in zip(df["o"], df["c"])]

    if show_candle_body:
        fig.add_trace(
            go.Bar(
                x=df["t"],
                y=body_height,
                base=body_base,
                marker_color=body_colors,
                marker_line_width=0,
                name=symbol,
                width=_bar_width_ms(df),
            ),
            row=1,
            col=1,
        )

    if show_percentile_body:
        # Fallback: OHLCV-based approximation (20%/80% of the bar's wick range)
        p20 = (df["l"] + 0.2 * (df["h"] - df["l"])).values.copy()
        p80 = (df["l"] + 0.8 * (df["h"] - df["l"])).values.copy()

        if not df_trades.empty:
            # Override fallback with actual trade percentiles where data exists
            bar_ts = df["t"].sort_values()
            last_gap = (bar_ts.iloc[-1] - bar_ts.iloc[-2]) if len(bar_ts) >= 2 else pd.Timedelta(minutes=1)
            bins = pd.DatetimeIndex(bar_ts.tolist() + [bar_ts.iloc[-1] + last_gap])
            trade_t = df_trades["t"]
            if trade_t.dt.tz is None:
                trade_t = trade_t.dt.tz_localize("UTC")
            else:
                trade_t = trade_t.dt.tz_convert("UTC")
            trade_t = trade_t.dt.as_unit("us")
            bin_idx = pd.cut(trade_t, bins=bins, right=False, labels=False)
            df_binned = pd.DataFrame({"bin_idx": bin_idx, "p": df_trades["p"].values}).dropna(subset=["bin_idx"])
            df_binned["bin_idx"] = df_binned["bin_idx"].astype(int)
            pct = df_binned.groupby("bin_idx")["p"].quantile([0.2, 0.8]).unstack()
            pct.columns = ["p20", "p80"]
            for i, row in pct.iterrows():
                p20[i] = row["p20"]
                p80[i] = row["p80"]

        pct_colors = [PALETTE["up"] if c >= o else PALETTE["down"] for c, o in zip(df["c"], df["o"])]
        fig.add_trace(
            go.Bar(
                x=df["t"],
                y=p80 - p20,
                base=p20,
                marker_color=pct_colors,
                marker_line_width=0,
                opacity=0.45,
                name="20%-80%",
                width=_bar_width_ms(df),
            ),
            row=1,
            col=1,
        )

    if show_whiskers:
        for direction, color in (("up", PALETTE["up"]), ("down", PALETTE["down"])):
            wx, wy = [], []
            for t, lo, hi, o, c in zip(df["t"], df["l"], df["h"], df["o"], df["c"]):
                if (c >= o) != (direction == "up"):
                    continue
                wx += [t, t, None]
                wy += [lo, hi, None]
            if wx:
                fig.add_trace(
                    go.Scatter(
                        x=wx,
                        y=wy,
                        mode="lines",
                        line=dict(color=color, width=1),
                        name="Wicks",
                        showlegend=False,
                    ),
                    row=1,
                    col=1,
                )

    if "synthetic" in df.columns and df["synthetic"].any():
        # Gap-filled buckets: a zero-range candle body renders as nothing, so
        # mark them as muted flat ticks at the carried-forward close instead.
        sdf = df[df["synthetic"]]
        fig.add_trace(
            go.Scatter(
                x=sdf["t"],
                y=sdf["c"],
                mode="markers",
                marker=dict(
                    symbol="line-ew",
                    size=7,
                    line=dict(color=PALETTE["muted"], width=1),
                ),
                name="No trades",
                showlegend=False,
                hovertemplate="<b>No trades</b> — last %{y:.4f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    vol_colors = [
        PALETTE["up"] if c >= o else PALETTE["down"]
        for c, o in zip(df["c"], df["o"])
    ]
    fig.add_trace(
        go.Bar(
            x=df["t"],
            y=df["v"],
            marker_color=vol_colors,
            opacity=0.75,
            name="Volume",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    df_news = pd.DataFrame(news) if news else pd.DataFrame(columns=["created_at", "headline"])
    if not df_news.empty:
        df_news["created_at"] = pd.to_datetime(df_news["created_at"])
        df_news = df_news[df_news["created_at"] > session_start]

    if not df_news.empty:
        impacts = news_impacts or {}
        if "id" in df_news.columns:
            impact_labels = [
                impacts.get(str(news_id), "unknown") for news_id in df_news["id"]
            ]
        else:
            impact_labels = ["unknown"] * len(df_news)
        impact_colors = [
            NEWS_IMPACT_COLORS.get(label, NEWS_IMPACT_COLORS["unknown"])
            for label in impact_labels
        ]

        # Anchor each dot to the high of the minute bar containing the article,
        # so markers sit just above the price action at publication time.
        news_t = df_news["created_at"]
        news_t = (
            news_t.dt.tz_localize("UTC") if news_t.dt.tz is None else news_t.dt.tz_convert("UTC")
        )
        bar_idx = np.searchsorted(df["t"].values, news_t.values, side="right") - 1
        bar_idx = np.clip(bar_idx, 0, len(df) - 1)
        price_range = float(df["h"].max() - df["l"].min())
        # Fall back to a fraction of the price itself when the session range is
        # flat (single bar / pre-open), so the dot still clears the candle.
        offset = price_range * NEWS_MARKER_OFFSET_FRAC or float(df["h"].max()) * 0.001
        news_y = df["h"].to_numpy()[bar_idx] + offset

        fig.add_trace(
            go.Scatter(
                x=df_news["created_at"],
                y=news_y,
                mode="markers",
                marker=dict(
                    color=impact_colors,
                    size=9,
                    line=dict(width=1, color="#ffffff"),
                ),
                text=df_news["headline"],
                customdata=impact_labels,
                hovertemplate=(
                    "<b>%{text}</b><br>Impact: %{customdata}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        for created_at, color in zip(df_news["created_at"], impact_colors):
            fig.add_vline(
                x=created_at,
                line=dict(color=color, width=1, dash="dot"),
                row=1,
                col=1,
            )

    if vwap_style != "hide":
        _add_vwap(df, fig, vwap_style=vwap_style, show_candle_body=show_candle_body or show_percentile_body)

    if ma_periods:
        _add_moving_averages(df, fig, ma_periods)

    df_daily: Optional[pd.DataFrame] = None
    if daily_bars:
        df_daily = pd.DataFrame(daily_bars)
        df_daily["t"] = pd.to_datetime(df_daily["t"], utc=True)

    _add_avg_lines(
        df_all, fig,
        show_7d=show_7d_avg, show_28d=show_28d_avg, show_1y=show_1y_avg,
        x0_dt=session_start,
        df_daily=df_daily,
    )

    if mixture_components:
        x0 = df["t"].iloc[0]
        x1 = df["t"].iloc[-1]
        prefix = "C" if mixture_distribution == "cauchy" else "G"
        for i, (_, mu, _) in enumerate(mixture_components):
            color = _MIXTURE_COLORS[i % len(_MIXTURE_COLORS)]
            fig.add_shape(
                type="line",
                x0=x0, x1=x1,
                y0=mu, y1=mu,
                line=dict(color=color, width=1, dash="dash"),
                row=1, col=1,
            )
            fig.add_annotation(
                xref="x", yref="y",
                x=x1, y=mu,
                text=f" {prefix}{i + 1} {mu:.2f}",
                font=dict(color=color, size=10, family="monospace"),
                showarrow=False,
                xanchor="left",
            )

    if show_fib:
        _add_fibonacci_levels(df, fig)

    _add_decision_markers(decisions or [], fig, session_start)

    if price_alerts:
        _add_price_alerts(price_alerts, fig, df["t"].iloc[0], df["t"].iloc[-1])
    if tactic_levels:
        _add_tactic_levels(tactic_levels, fig, df["t"].iloc[0], df["t"].iloc[-1])

    last = df.iloc[-1]
    color = PALETTE["up"] if last["c"] >= last["o"] else PALETTE["down"]
    fig.add_annotation(
        xref="paper",
        yref="y",
        x=1.01,
        y=last["c"],
        text=f" {last['c']:.2f}",
        font=dict(color=color, size=13, family="monospace"),
        showarrow=False,
        bgcolor=PALETTE["panel"],
        bordercolor=color,
        borderwidth=1,
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=PALETTE["bg"],
        plot_bgcolor=PALETTE["panel"],
        font=dict(color=PALETTE["text"], family="Inter, sans-serif"),
        title=dict(
            text=f"<b>{symbol}</b>  <span style='color:{color}'>{last['c']:.2f}</span>",
            font=dict(size=18),
            x=0.02,
        ),
        xaxis=dict(
            range=[
                pd.Timestamp(session_start).isoformat(),
                df["t"].max().isoformat(),
            ],
            rangeslider=dict(visible=False),
        ),
        xaxis2=dict(showgrid=True, gridcolor=PALETTE["grid"]),
        yaxis=dict(showgrid=True, gridcolor=PALETTE["grid"], tickfont=dict(size=11)),
        yaxis2=dict(showgrid=True, gridcolor=PALETTE["grid"], tickfont=dict(size=10)),
        legend=dict(orientation="h", y=1.04, bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=10, t=50, b=10),
        height=520,
    )
    return fig


def build_performance_chart(
    points: list[dict],
    markers: list[dict],
    symbol: str,
) -> go.Figure:
    """Agent equity curve: portfolio value per bar, with markers at filled decisions."""
    if not points:
        return empty_chart("No agent performance data yet")

    df = pd.DataFrame(points)
    df["ts"] = pd.to_datetime(df["ts"], format="ISO8601")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["ts"],
            y=df["value"],
            mode="lines+markers",
            name="Portfolio value",
            line=dict(color=PALETTE["accent"], width=2),
            marker=dict(size=4),
            hovertemplate="<b>%{x|%H:%M}</b><br>Value: $%{y:,.2f}<extra></extra>",
        )
    )

    if markers:
        df_m = pd.DataFrame(markers)
        df_m["ts"] = pd.to_datetime(df_m["ts"], format="ISO8601")
        for action, marker_symbol, color in (
            ("buy", "triangle-up", PALETTE["up"]),
            ("sell", "triangle-down", PALETTE["down"]),
        ):
            sub = df_m[df_m["action"] == action]
            if sub.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=sub["ts"],
                    y=sub["value"],
                    mode="markers",
                    marker=dict(symbol=marker_symbol, size=12, color=color, line=dict(width=1.5, color="#ffffff")),
                    name=f"Agent {action}",
                    hovertemplate=f"<b>Agent {action}</b><br>Value: $%{{y:,.2f}}<extra></extra>",
                )
            )
        # Moments the agent armed a conditional trade plan (tactics), so the
        # curve shows when a standing order was set, not only when it filled.
        sub = df_m[df_m["action"] == "tactics"]
        if not sub.empty:
            fig.add_trace(
                go.Scatter(
                    x=sub["ts"],
                    y=sub["value"],
                    mode="markers",
                    marker=dict(symbol="diamond", size=11, color=PALETTE["orange"], line=dict(width=1.5, color="#ffffff")),
                    name="Tactics armed",
                    customdata=sub.get("label"),
                    hovertemplate="<b>🎯 Tactics armed</b><br>%{customdata}<br>Value: $%{y:,.2f}<extra></extra>",
                )
            )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=PALETTE["bg"],
        plot_bgcolor=PALETTE["panel"],
        font=dict(color=PALETTE["text"], family="Inter, sans-serif"),
        title=dict(text=f"<b>{symbol}</b> agent portfolio value", font=dict(size=16), x=0.02),
        xaxis=dict(showgrid=True, gridcolor=PALETTE["grid"]),
        yaxis=dict(showgrid=True, gridcolor=PALETTE["grid"], tickprefix="$"),
        legend=dict(orientation="h", y=1.08, bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=10, t=50, b=10),
        height=420,
    )
    return fig


def build_gamma_chart(data: dict, analysis: dict, symbol: str) -> go.Figure:
    """Per-strike call/put open interest (mirrored bars) with net dealer-gamma exposure
    overlaid, plus the spot price, call wall, put wall, and gamma flip level marked."""
    strikes = data.get("strikes") or []
    if not strikes:
        return empty_chart(f"No options chain data for {symbol}")

    calls_oi = data["calls_oi"]
    puts_oi = data["puts_oi"]
    net_gamma = [c + p for c, p in zip(data["calls_gamma_exposure"], data["puts_gamma_exposure"])]
    spot = data["spot"]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=strikes,
            y=calls_oi,
            name="Call OI",
            marker_color=PALETTE["up"],
            opacity=0.75,
            hovertemplate="<b>Strike %{x}</b><br>Call OI: %{y:,.0f}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(
            x=strikes,
            y=[-v for v in puts_oi],
            name="Put OI",
            marker_color=PALETTE["down"],
            opacity=0.75,
            customdata=puts_oi,
            hovertemplate="<b>Strike %{x}</b><br>Put OI: %{customdata:,.0f}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=strikes,
            y=net_gamma,
            name="Net gamma exposure",
            mode="lines",
            line=dict(color=PALETTE["accent"], width=2),
            hovertemplate="<b>Strike %{x}</b><br>Net gamma: %{y:,.0f}<extra></extra>",
        ),
        secondary_y=True,
    )

    levels = (
        ("call_wall", "Call Wall", PALETTE["up"]),
        ("put_wall", "Put Wall", PALETTE["down"]),
        ("gamma_flip", "Gamma Flip", PALETTE["orange"]),
    )
    for key, label, color in levels:
        level = analysis.get(key)
        if level is None:
            continue
        fig.add_vline(
            x=level,
            line=dict(color=color, width=1.5, dash="dash"),
            annotation_text=f"{label} {level:.2f}",
            annotation_font=dict(color=color, size=10),
            annotation_position="top",
        )

    fig.add_vline(
        x=spot,
        line=dict(color=PALETTE["text"], width=1.5, dash="dot"),
        annotation_text=f"Spot {spot:.2f}",
        annotation_font=dict(color=PALETTE["text"], size=10),
        annotation_position="bottom",
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=PALETTE["bg"],
        plot_bgcolor=PALETTE["panel"],
        font=dict(color=PALETTE["text"], family="Inter, sans-serif"),
        title=dict(
            text=f"<b>{symbol}</b> put/call walls &amp; gamma — expiry {data.get('expiry', '')}",
            font=dict(size=18),
            x=0.02,
        ),
        barmode="relative",
        xaxis=dict(title="Strike", showgrid=True, gridcolor=PALETTE["grid"]),
        yaxis=dict(title="Open interest (puts mirrored)", showgrid=True, gridcolor=PALETTE["grid"]),
        yaxis2=dict(title="Net gamma exposure ($/1%)", showgrid=False),
        legend=dict(orientation="h", y=1.08, bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=10, t=60, b=10),
        height=520,
    )
    return fig


def build_smart_money_chart(bars: list[dict], analysis: dict, symbol: str) -> go.Figure:
    """Smart Money Concepts overlay: candlesticks with the detected order blocks and
    fair value gaps drawn as shaded zones, plus the suggested entry/stop/target lines.

    Bullish (demand) zones are shaded green, bearish (supply) zones red. The order
    block the agent is actually watching (`analysis['order_block']`) is highlighted,
    its target supply zone marked, and the entry/stop/target geometry drawn as
    horizontal lines so a setup reads at a glance.
    """
    if not bars:
        return empty_chart(f"No data for {symbol}")

    df = pd.DataFrame(bars)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    df = df.sort_values("t").reset_index(drop=True)
    x0, x1 = df["t"].iloc[0], df["t"].iloc[-1]

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df["t"],
            open=df["o"],
            high=df["h"],
            low=df["l"],
            close=df["c"],
            name=symbol,
            increasing_line_color=PALETTE["up"],
            decreasing_line_color=PALETTE["down"],
            showlegend=False,
        )
    )

    def _index_to_x(idx: "int | None"):
        if idx is None or idx < 0 or idx >= len(df):
            return x0
        return df["t"].iloc[idx]

    watched = analysis.get("order_block")
    watched_idx = watched.get("index") if watched else None

    # All detected order blocks as shaded zones; the watched one drawn brighter.
    for blk in (analysis.get("order_blocks") or []):
        is_bull = blk.get("type") == "bullish"
        base_color = PALETTE["up"] if is_bull else PALETTE["down"]
        watched_block = watched is not None and blk.get("index") == watched_idx
        fig.add_shape(
            type="rect",
            xref="x", yref="y",
            x0=_index_to_x(blk.get("index")), x1=x1,
            y0=blk["bottom"], y1=blk["top"],
            fillcolor=base_color,
            opacity=0.30 if watched_block else 0.12,
            line=dict(color=base_color, width=1.5 if watched_block else 0),
            layer="below",
        )
        fig.add_annotation(
            xref="x", yref="y", x=x1, y=blk["top"],
            text=f" {'Demand' if is_bull else 'Supply'} OB {blk['bottom']:.2f}-{blk['top']:.2f}",
            font=dict(color=base_color, size=9, family="monospace"),
            showarrow=False, xanchor="left",
        )

    # Fair value gaps (intraday-derived) as thin amber zones.
    for gap in (analysis.get("fair_value_gaps") or []):
        if gap.get("filled"):
            continue
        fig.add_shape(
            type="rect",
            xref="x", yref="y",
            x0=_index_to_x(gap.get("index")), x1=x1,
            y0=gap["bottom"], y1=gap["top"],
            fillcolor=PALETTE["orange"], opacity=0.15,
            line=dict(width=0), layer="below",
        )

    # Premium / discount: equilibrium line with the discount half shaded faintly
    # green (where Smart Money buys). Drawn below the candles.
    pd_read = analysis.get("premium_discount") or {}
    eq = pd_read.get("equilibrium")
    r_lo, r_hi = pd_read.get("range_low"), pd_read.get("range_high")
    if eq is not None and r_lo is not None:
        fig.add_shape(
            type="rect", xref="x", yref="y",
            x0=x0, x1=x1, y0=r_lo, y1=eq,
            fillcolor=PALETTE["up"], opacity=0.05, line=dict(width=0), layer="below",
        )
        fig.add_shape(
            type="line", xref="x", yref="y", x0=x0, x1=x1, y0=eq, y1=eq,
            line=dict(color=PALETTE["muted"], width=1, dash="dashdot"),
        )
        fig.add_annotation(
            xref="x", yref="y", x=x1, y=eq,
            text=f" Equilibrium {eq:.2f} ({pd_read.get('zone', '')})",
            font=dict(color=PALETTE["muted"], size=9, family="monospace"),
            showarrow=False, xanchor="left",
        )

    # Liquidity pools: nearest buy-side (overhead, a target) and sell-side (below,
    # resting stops) as thin violet dotted lines.
    liq = analysis.get("liquidity") or {}
    for pool_key, tag in (("nearest_bsl_above", "BSL"), ("nearest_ssl_below", "SSL")):
        pool = liq.get(pool_key)
        if not pool:
            continue
        label = f"{tag}{' (equal)' if pool.get('equal') else ''} {pool['price']:.2f}"
        fig.add_shape(
            type="line", xref="x", yref="y", x0=x0, x1=x1, y0=pool["price"], y1=pool["price"],
            line=dict(color="#a78bfa", width=1, dash="dot"),
        )
        fig.add_annotation(
            xref="x", yref="y", x=x0, y=pool["price"], text=f"{label} ",
            font=dict(color="#a78bfa", size=9, family="monospace"),
            showarrow=False, xanchor="right",
        )

    # Entry / stop / target geometry.
    levels = (
        ("suggested_entry", "Entry", PALETTE["accent"], "dash"),
        ("suggested_stop", "Stop", PALETTE["down"], "dot"),
        ("structural_target", "Target", PALETTE["up"], "dot"),
    )
    for key, label, color, dash in levels:
        level = analysis.get(key)
        if level is None:
            continue
        fig.add_shape(
            type="line", xref="x", yref="y",
            x0=x0, x1=x1, y0=level, y1=level,
            line=dict(color=color, width=1.5, dash=dash),
        )
        fig.add_annotation(
            xref="x", yref="y", x=x0, y=level,
            text=f"{label} {level:.2f} ",
            font=dict(color=color, size=10, family="monospace"),
            showarrow=False, xanchor="right",
        )

    signal = analysis.get("signal", "")
    quality = analysis.get("quality", "")
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=PALETTE["bg"],
        plot_bgcolor=PALETTE["panel"],
        font=dict(color=PALETTE["text"], family="Inter, sans-serif"),
        title=dict(
            text=f"<b>{symbol}</b> Smart Money setup — <span style='color:{PALETTE['accent']}'>"
            f"{signal.replace('_', ' ')} ({quality})</span>",
            font=dict(size=18), x=0.02,
        ),
        xaxis=dict(showgrid=True, gridcolor=PALETTE["grid"], rangeslider=dict(visible=False)),
        yaxis=dict(showgrid=True, gridcolor=PALETTE["grid"], tickfont=dict(size=11)),
        margin=dict(l=10, r=10, t=50, b=10),
        height=520,
    )
    return fig


def build_analysis_gauges(trend: dict, intraday: dict, market: dict) -> go.Figure:
    """Three-gauge snapshot: daily-trend RSI, intraday momentum, and the VIX fear
    gauge -- the headline number from each `technical_analysis` read, at a glance."""
    fig = make_subplots(rows=1, cols=3, specs=[[{"type": "indicator"}] * 3])

    rsi14 = trend.get("rsi_14")
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=rsi14 if rsi14 is not None else 50,
            number=dict(suffix="" if rsi14 is not None else " (n/a)", font=dict(color=PALETTE["text"])),
            title=dict(
                text=f"Trend RSI(14)<br><span style='font-size:11px;color:{PALETTE['muted']}'>"
                f"{trend.get('regime', 'n/a')}</span>"
            ),
            gauge=dict(
                axis=dict(range=[0, 100], tickcolor=PALETTE["muted"]),
                bar=dict(color=PALETTE["accent"]),
                bgcolor=PALETTE["panel"],
                steps=[
                    dict(range=[0, 30], color="rgba(239,83,80,0.35)"),
                    dict(range=[30, 70], color="rgba(136,136,136,0.2)"),
                    dict(range=[70, 100], color="rgba(239,83,80,0.35)"),
                ],
            ),
        ),
        row=1,
        col=1,
    )

    pct_chg = intraday.get("pct_change_in_window") or 0.0
    bound = max(2.0, abs(pct_chg) * 1.5)
    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=pct_chg,
            number=dict(suffix="%", font=dict(color=PALETTE["text"])),
            delta=dict(reference=0, suffix="%"),
            title=dict(
                text="Intraday Momentum<br><span style='font-size:11px;color:"
                f"{PALETTE['muted']}'>{(intraday.get('momentum_pattern') or 'n/a')[:40]}</span>"
            ),
            gauge=dict(
                axis=dict(range=[-bound, bound], tickcolor=PALETTE["muted"]),
                bar=dict(color=PALETTE["up"] if pct_chg >= 0 else PALETTE["down"]),
                bgcolor=PALETTE["panel"],
                steps=[
                    dict(range=[-bound, 0], color="rgba(239,83,80,0.2)"),
                    dict(range=[0, bound], color="rgba(38,198,162,0.2)"),
                ],
            ),
        ),
        row=1,
        col=2,
    )

    vix = market.get("vix")
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=vix if vix is not None else 20,
            number=dict(suffix="" if vix is not None else " (n/a)", font=dict(color=PALETTE["text"])),
            title=dict(
                text=f"VIX<br><span style='font-size:11px;color:{PALETTE['muted']}'>"
                f"{market.get('risk_environment', 'n/a')}</span>"
            ),
            gauge=dict(
                axis=dict(range=[0, 40], tickcolor=PALETTE["muted"]),
                bar=dict(color=PALETTE["orange"]),
                bgcolor=PALETTE["panel"],
                steps=[
                    dict(range=[0, 17], color="rgba(38,198,162,0.25)"),
                    dict(range=[17, 26], color="rgba(251,146,60,0.25)"),
                    dict(range=[26, 40], color="rgba(239,83,80,0.3)"),
                ],
            ),
        ),
        row=1,
        col=3,
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=PALETTE["bg"],
        font=dict(color=PALETTE["text"], family="Inter, sans-serif"),
        margin=dict(l=20, r=20, t=70, b=10),
        height=260,
    )
    return fig


def build_historical_chart(
    ticker_close: pd.Series,
    spy_close: pd.Series,
    vix_close: pd.Series,
    symbol: str,
    period_label: str,
    dividends: Optional[pd.Series] = None,
    earnings: Optional[pd.DataFrame] = None,
) -> go.Figure:
    """Plot `symbol` and SPY as % change over the period, with VIX on a secondary axis.

    Earnings and dividend dates for `symbol` are marked as vertical lines.
    """
    if ticker_close is None or ticker_close.empty:
        return empty_chart(f"No historical data for {symbol}")

    ticker_pct = (ticker_close / ticker_close.iloc[0] - 1.0) * 100
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=ticker_pct.index,
            y=ticker_pct.values,
            mode="lines",
            name=symbol,
            line=dict(color=PALETTE["accent"], width=2),
        )
    )

    if spy_close is not None and not spy_close.empty:
        spy_pct = (spy_close / spy_close.iloc[0] - 1.0) * 100
        fig.add_trace(
            go.Scatter(
                x=spy_pct.index,
                y=spy_pct.values,
                mode="lines",
                name="SPY",
                line=dict(color=PALETTE["muted"], width=2, dash="dot"),
            )
        )

    if vix_close is not None and not vix_close.empty:
        fig.add_trace(
            go.Scatter(
                x=vix_close.index,
                y=vix_close.values,
                mode="lines",
                name="VIX",
                line=dict(color=PALETTE["orange"], width=1.5),
                yaxis="y2",
            )
        )

    if earnings is not None and not earnings.empty:
        for dt in earnings.index:
            naive = (dt.tz_localize(None) if dt.tzinfo else dt).to_pydatetime()
            fig.add_vline(x=naive, line=dict(color=PALETTE["accent"], width=1, dash="dash"))
            fig.add_annotation(
                xref="x", yref="paper", x=naive, y=1.0,
                text="Earnings", font=dict(size=9, color=PALETTE["accent"]),
                showarrow=False, yanchor="bottom",
            )

    if dividends is not None and not dividends.empty:
        for dt in dividends.index:
            naive = (dt.tz_localize(None) if dt.tzinfo else dt).to_pydatetime()
            fig.add_vline(x=naive, line=dict(color=PALETTE["up"], width=1, dash="dash"))
            fig.add_annotation(
                xref="x", yref="paper", x=naive, y=0.0,
                text="Dividend", font=dict(size=9, color=PALETTE["up"]),
                showarrow=False, yanchor="top",
            )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=PALETTE["bg"],
        plot_bgcolor=PALETTE["panel"],
        font=dict(color=PALETTE["text"], family="Inter, sans-serif"),
        title=dict(
            text=f"<b>{symbol}</b> vs <b>SPY</b> vs <b>VIX</b> — {period_label}",
            font=dict(size=18),
            x=0.02,
        ),
        xaxis=dict(showgrid=True, gridcolor=PALETTE["grid"]),
        yaxis=dict(
            title="% change",
            showgrid=True,
            gridcolor=PALETTE["grid"],
            ticksuffix="%",
        ),
        yaxis2=dict(
            title="VIX",
            overlaying="y",
            side="right",
            showgrid=False,
        ),
        legend=dict(orientation="h", y=1.08, bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=10, t=60, b=10),
        height=520,
    )
    return fig
