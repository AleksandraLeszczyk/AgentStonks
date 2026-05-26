import colorsys
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import PALETTE


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


def _plot_price_distribution(
    df_trades: pd.DataFrame,
    fig: go.Figure,
    n_price_bins: int = 50,
    n_time_buckets: int = 20,
) -> go.Figure:
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
    return fig


def build_chart(
    bars: list[dict],
    news: list[dict],
    trades: list[dict],
    symbol: str,
    session_start: datetime,
) -> go.Figure:
    if not bars:
        return empty_chart("Waiting for data…")

    df = pd.DataFrame(bars)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    df = df[df["t"] > session_start].sort_values("t").reset_index(drop=True)
    if df.empty:
        return empty_chart("Waiting for data…")

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
        fig = _plot_price_distribution(df_trades, fig)

    fig.add_trace(
        go.Candlestick(
            x=df["t"],
            open=df["o"],
            high=df["h"],
            low=df["l"],
            close=df["c"],
            name=symbol,
            increasing=dict(line=dict(color=PALETTE["up"], width=1), fillcolor=PALETTE["up"]),
            decreasing=dict(line=dict(color=PALETTE["down"], width=1), fillcolor=PALETTE["down"]),
            whiskerwidth=0.2,
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
        price_ref = df_trades["p"].max() if not df_trades.empty else df["h"].max()
        fig.add_trace(
            go.Scatter(
                x=df_news["created_at"],
                y=[price_ref + 0.1] * len(df_news),
                mode="markers",
                text=df_news["headline"],
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        for item in df_news.itertuples():
            fig.add_vline(
                x=item.created_at,
                line=dict(color=PALETTE["orange"], width=1, dash="dot"),
                row=1,
                col=1,
            )

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
        xaxis_rangeslider_visible=False,
        xaxis2=dict(showgrid=True, gridcolor=PALETTE["grid"]),
        yaxis=dict(showgrid=True, gridcolor=PALETTE["grid"], tickfont=dict(size=11)),
        yaxis2=dict(showgrid=True, gridcolor=PALETTE["grid"], tickfont=dict(size=10)),
        legend=dict(orientation="h", y=1.04, bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=10, t=50, b=10),
        height=520,
    )
    return fig
