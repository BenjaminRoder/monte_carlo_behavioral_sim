"""
Calibration Plots
==================
Validates that the simulation's return distribution matches real historical data.
These charts are the "proof" that the model is data-driven, not made up.

Four calibration charts:
  1. Simulated vs historical return distribution (per asset, overlaid)
  2. Q-Q plot — simulated quantiles vs historical quantiles
  3. Rolling correlation — empirical vs simulated stability
  4. Tail comparison — how often do large losses occur in sim vs history
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from src.data import MarketData

# Reuse theme from visualizations
THEME = dict(
    bg       = "#0D1117",
    panel    = "#161B22",
    border   = "#30363D",
    text_pri = "#E6EDF3",
    text_sec = "#8B949E",
    gridline = "#21262D",
    font_family = "IBM Plex Mono, monospace",
)

ASSET_COLORS = ["#4A9EFF", "#AB47BC", "#3FB950"]


def _base_layout(**kwargs) -> dict:
    base = dict(
        paper_bgcolor=THEME["bg"],
        plot_bgcolor=THEME["panel"],
        font=dict(family=THEME["font_family"], color=THEME["text_pri"], size=11),
        margin=dict(l=50, r=20, t=50, b=50),
    )
    base.update(kwargs)
    return base


def plot_return_distributions(
    market_data: MarketData,
    sim_daily_returns: np.ndarray,  # (n_sim, n_days, n_assets)
    n_sample: int = 500,
) -> go.Figure:
    """
    Overlay simulated vs historical daily return distributions for each asset.
    Uses a random sample of simulation paths to keep it fast.
    """
    assets      = market_data.fitted_assets
    log_returns = market_data.log_returns
    n_assets    = len(assets)

    fig = make_subplots(
        rows=1, cols=n_assets,
        subplot_titles=[a.name.split("(")[0].strip() for a in assets],
    )

    rng     = np.random.default_rng(0)
    sim_idx = rng.choice(sim_daily_returns.shape[0], size=min(n_sample, sim_daily_returns.shape[0]), replace=False)

    for i, (asset, col) in enumerate(zip(assets, log_returns.columns)):
        hist_r = log_returns[col].values
        sim_r  = sim_daily_returns[sim_idx, :, i].ravel()

        # Clip to 1-99th percentile for clean display
        lo = np.percentile(hist_r, 0.5)
        hi = np.percentile(hist_r, 99.5)
        bins = np.linspace(lo, hi, 80)

        # Historical
        h_counts, h_edges = np.histogram(hist_r, bins=bins, density=True)
        h_centres = (h_edges[:-1] + h_edges[1:]) / 2
        fig.add_trace(go.Scatter(
            x=h_centres, y=h_counts,
            mode="lines", name="Historical" if i == 0 else None,
            showlegend=(i == 0),
            line=dict(color=ASSET_COLORS[i], width=2),
            legendgroup="hist",
        ), row=1, col=i + 1)

        # Simulated
        s_counts, s_edges = np.histogram(sim_r, bins=bins, density=True)
        s_centres = (s_edges[:-1] + s_edges[1:]) / 2
        fig.add_trace(go.Scatter(
            x=s_centres, y=s_counts,
            mode="lines", name="Simulated" if i == 0 else None,
            showlegend=(i == 0),
            line=dict(color="white", width=1.5, dash="dash"),
            opacity=0.7,
            legendgroup="sim",
        ), row=1, col=i + 1)

        # Fitted t-distribution curve
        x_range = np.linspace(lo, hi, 200)
        fitted_pdf = stats.t.pdf(x_range, df=asset.nu, loc=asset.mu, scale=asset.sigma)
        fig.add_trace(go.Scatter(
            x=x_range, y=fitted_pdf,
            mode="lines", name="Fitted t-dist" if i == 0 else None,
            showlegend=(i == 0),
            line=dict(color="#FFD700", width=1.5, dash="dot"),
            legendgroup="fitted",
        ), row=1, col=i + 1)

    fig.update_layout(
        title="Calibration: simulated vs historical return distributions",
        **_base_layout(),
    )
    for i in range(1, n_assets + 1):
        fig.update_xaxes(gridcolor=THEME["gridline"], row=1, col=i)
        fig.update_yaxes(gridcolor=THEME["gridline"], row=1, col=i)

    return fig


def plot_qq(
    market_data: MarketData,
    sim_daily_returns: np.ndarray,
    n_sample: int = 300,
) -> go.Figure:
    """
    Q-Q plot: simulated quantiles vs historical quantiles.
    A straight diagonal line = perfect calibration.
    """
    assets      = market_data.fitted_assets
    log_returns = market_data.log_returns
    n_assets    = len(assets)

    fig = make_subplots(
        rows=1, cols=n_assets,
        subplot_titles=[a.name.split("(")[0].strip() for a in assets],
    )

    rng     = np.random.default_rng(1)
    sim_idx = rng.choice(sim_daily_returns.shape[0], size=min(n_sample, sim_daily_returns.shape[0]), replace=False)
    probs   = np.linspace(0.01, 0.99, 100)

    for i, (asset, col) in enumerate(zip(assets, log_returns.columns)):
        hist_r = log_returns[col].values
        sim_r  = sim_daily_returns[sim_idx, :, i].ravel()

        hist_q = np.quantile(hist_r, probs)
        sim_q  = np.quantile(sim_r,  probs)

        # Scatter of quantile pairs
        fig.add_trace(go.Scatter(
            x=hist_q, y=sim_q,
            mode="markers",
            marker=dict(color=ASSET_COLORS[i], size=5, opacity=0.8),
            name=asset.name.split("(")[0].strip() if i == 0 else None,
            showlegend=False,
        ), row=1, col=i + 1)

        # 45-degree reference line
        mn = min(hist_q.min(), sim_q.min())
        mx = max(hist_q.max(), sim_q.max())
        fig.add_trace(go.Scatter(
            x=[mn, mx], y=[mn, mx],
            mode="lines",
            line=dict(color="#FFD700", width=1, dash="dash"),
            showlegend=(i == 0),
            name="Perfect fit" if i == 0 else None,
        ), row=1, col=i + 1)

    fig.update_layout(
        title="Q-Q plot: simulated quantiles vs historical quantiles (diagonal = perfect fit)",
        **_base_layout(),
    )
    for i in range(1, n_assets + 1):
        fig.update_xaxes(title_text="Historical", gridcolor=THEME["gridline"], row=1, col=i)
        fig.update_yaxes(title_text="Simulated", gridcolor=THEME["gridline"], row=1, col=i)

    return fig


def plot_rolling_correlation(market_data: MarketData) -> go.Figure:
    """
    Empirical rolling 1-year correlations — shows how correlations shift
    over time and highlights the 2008/2020 crisis spikes.
    """
    roll_corr = market_data.rolling_corr
    colors    = ["#4A9EFF", "#AB47BC", "#3FB950"]

    fig = go.Figure()
    for i, col in enumerate(roll_corr.columns):
        fig.add_trace(go.Scatter(
            x=roll_corr.index,
            y=roll_corr[col],
            mode="lines",
            name=col,
            line=dict(color=colors[i % len(colors)], width=1.5),
        ))

    # Mark crisis periods
    crises = [
        ("2008-09-15", "2009-03-09", "GFC"),
        ("2020-02-19", "2020-03-23", "COVID"),
        ("2022-01-03", "2022-10-13", "Rate hikes"),
    ]
    for start, end, label in crises:
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor="#FF7043", opacity=0.08,
            line_width=0,
            annotation_text=label,
            annotation_position="top left",
            annotation_font=dict(color="#FF7043", size=9),
        )

    fig.add_hline(y=0, line=dict(color=THEME["text_sec"], width=0.8, dash="dash"))

    fig.update_layout(
        title="Rolling 1-year correlations — empirical (shaded = crisis periods)",
        xaxis_title="Date",
        yaxis_title="Pearson correlation",
        **_base_layout(),
        xaxis=dict(gridcolor=THEME["gridline"]),
        yaxis=dict(gridcolor=THEME["gridline"], range=[-0.6, 1.0]),
    )
    return fig


def plot_tail_comparison(
    market_data: MarketData,
    sim_daily_returns: np.ndarray,
    n_sample: int = 500,
) -> go.Figure:
    """
    Tail loss comparison: frequency of large daily losses in simulation vs history.
    Key test — does the fat-tailed model reproduce crash frequency correctly?
    """
    assets      = market_data.fitted_assets
    log_returns = market_data.log_returns
    n_assets    = len(assets)

    thresholds  = np.array([-0.005, -0.010, -0.015, -0.020, -0.030, -0.040, -0.050])
    threshold_labels = [f"{t:.1%}" for t in thresholds]

    rng     = np.random.default_rng(2)
    sim_idx = rng.choice(sim_daily_returns.shape[0], size=min(n_sample, sim_daily_returns.shape[0]), replace=False)

    fig = go.Figure()

    for i, (asset, col) in enumerate(zip(assets, log_returns.columns)):
        hist_r = log_returns[col].values
        sim_r  = sim_daily_returns[sim_idx, :, i].ravel()

        hist_freq = np.array([(hist_r < t).mean() for t in thresholds])
        sim_freq  = np.array([(sim_r  < t).mean() for t in thresholds])

        fig.add_trace(go.Scatter(
            x=threshold_labels, y=hist_freq,
            mode="lines+markers",
            name=f"{asset.name.split('(')[0].strip()} — Historical",
            line=dict(color=ASSET_COLORS[i], width=2),
            marker=dict(size=7),
        ))
        fig.add_trace(go.Scatter(
            x=threshold_labels, y=sim_freq,
            mode="lines+markers",
            name=f"{asset.name.split('(')[0].strip()} — Simulated",
            line=dict(color=ASSET_COLORS[i], width=1.5, dash="dash"),
            marker=dict(size=5, symbol="diamond"),
        ))

    fig.update_layout(
        title="Tail calibration: frequency of large daily losses — simulated vs historical",
        xaxis_title="Daily loss threshold",
        yaxis_title="Frequency (fraction of days)",
        yaxis_tickformat=".2%",
        **_base_layout(),
        xaxis=dict(gridcolor=THEME["gridline"]),
        yaxis=dict(gridcolor=THEME["gridline"]),
    )
    return fig


def compute_calibration_stats(
    market_data: MarketData,
    sim_daily_returns: np.ndarray,
    n_sample: int = 500,
) -> list[dict]:
    """
    Summary table of calibration statistics.
    Returns list of dicts suitable for a dataframe display.
    """
    log_returns = market_data.log_returns
    rng         = np.random.default_rng(3)
    sim_idx     = rng.choice(sim_daily_returns.shape[0], size=min(n_sample, sim_daily_returns.shape[0]), replace=False)

    rows = []
    for i, (asset, col) in enumerate(zip(market_data.fitted_assets, log_returns.columns)):
        hist_r = log_returns[col].values
        sim_r  = sim_daily_returns[sim_idx, :, i].ravel()
        a      = asset

        rows.append({
            "Asset":            a.name.split("(")[0].strip(),
            "Hist. ann. return":f"{hist_r.mean() * 252:.2%}",
            "Sim. ann. return": f"{sim_r.mean()  * 252:.2%}",
            "Hist. ann. vol":   f"{hist_r.std() * np.sqrt(252):.2%}",
            "Sim. ann. vol":    f"{sim_r.std()  * np.sqrt(252):.2%}",
            "Hist. skew":       f"{pd.Series(hist_r).skew():.3f}",
            "Sim. skew":        f"{pd.Series(sim_r).skew():.3f}",
            "Hist. kurtosis":   f"{pd.Series(hist_r).kurtosis():.2f}",
            "Sim. kurtosis":    f"{pd.Series(sim_r).kurtosis():.2f}",
            "Fitted nu":        f"{a.nu:.2f}",
            "KS p-value":       f"{a.ks_pval:.4f}",
        })
    return rows


def plot_regime_analysis(
    market_data: MarketData,
    sim_regimes: np.ndarray,   # (n_sim, n_days)
    n_years: int = 10,
) -> go.Figure:
    """
    Two-panel regime chart:
      Top: Historical realised vol with regime bands coloured
      Bottom: Simulated regime distribution over time (stacked area)
    """
    from plotly.subplots import make_subplots

    lr      = market_data.log_returns
    spy_vol = lr.iloc[:, 0].rolling(21).std() * np.sqrt(252)
    spy_vol = spy_vol.dropna()

    # Thresholds from calibration
    p40 = spy_vol.quantile(0.40)
    p75 = spy_vol.quantile(0.75)

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=[
            "Historical SPY realised volatility — regime classification",
            "Simulated regime occupancy over horizon (% of paths per regime)",
        ],
        vertical_spacing=0.14,
        row_heights=[0.55, 0.45],
    )

    # --- Top: historical vol with regime colouring ---
    fig.add_trace(go.Scatter(
        x=spy_vol.index, y=spy_vol.values,
        mode="lines", name="Realised vol (21d)",
        line=dict(color="#E6EDF3", width=1.2),
        showlegend=True,
    ), row=1, col=1)

    # Regime threshold lines
    fig.add_hline(y=p40, line=dict(color="#4A9EFF", width=1, dash="dot"),
                  annotation_text=f"Calm/Normal ({p40:.1%})",
                  annotation_font=dict(color="#4A9EFF", size=9),
                  annotation_position="bottom right", row=1, col=1)
    fig.add_hline(y=p75, line=dict(color="#FF7043", width=1, dash="dot"),
                  annotation_text=f"Normal/Crisis ({p75:.1%})",
                  annotation_font=dict(color="#FF7043", size=9),
                  annotation_position="top right", row=1, col=1)

    # Shade calm and crisis bands
    calm_mask   = spy_vol < p40
    crisis_mask = spy_vol > p75
    for mask, color, label in [
        (calm_mask,   "rgba(74,158,255,0.08)",  "Calm"),
        (crisis_mask, "rgba(255,112,67,0.12)",  "Crisis"),
    ]:
        # Find contiguous runs
        diff = mask.astype(int).diff().fillna(0)
        starts = spy_vol.index[diff == 1].tolist()
        ends   = spy_vol.index[diff == -1].tolist()
        if mask.iloc[0]:
            starts = [spy_vol.index[0]] + starts
        if mask.iloc[-1]:
            ends   = ends + [spy_vol.index[-1]]
        for s, e in zip(starts, ends):
            fig.add_vrect(x0=s, x1=e, fillcolor=color, line_width=0,
                          row=1, col=1)

    # Crisis annotations
    for start, end, lbl in [
        ("2008-09-01", "2009-06-01", "GFC"),
        ("2020-02-01", "2020-06-01", "COVID"),
        ("2022-01-01", "2022-12-01", "Hikes"),
    ]:
        fig.add_annotation(
            x=start, y=spy_vol.max() * 0.92,
            text=lbl, font=dict(color="#FF7043", size=9),
            showarrow=False, row=1, col=1,
        )

    # --- Bottom: simulated regime occupancy ---
    n_sim, n_days = sim_regimes.shape
    days_x = np.linspace(0, n_years, n_days)

    calm_pct   = (sim_regimes == 0).mean(axis=0) * 100
    normal_pct = (sim_regimes == 1).mean(axis=0) * 100
    crisis_pct = (sim_regimes == 2).mean(axis=0) * 100

    for pct, name, color in [
        (crisis_pct, "Crisis",  "#FF7043"),
        (normal_pct, "Normal",  "#8B949E"),
        (calm_pct,   "Calm",    "#4A9EFF"),
    ]:
        fig.add_trace(go.Scatter(
            x=days_x, y=pct,
            mode="lines", name=name,
            line=dict(color=color, width=0),
            fill="tonexty" if name != "Crisis" else "tozeroy",
            fillcolor=color.replace(")", ", 0.5)").replace("rgb(", "rgba(").replace("#FF7043", "rgba(255,112,67,0.45)").replace("#8B949E", "rgba(139,148,158,0.45)").replace("#4A9EFF", "rgba(74,158,255,0.45)"),
            stackgroup="regimes",
        ), row=2, col=1)

    fig.update_layout(
        **_base_layout(
            title="Regime analysis — historical classification & simulated occupancy",
            margin=dict(l=50, r=20, t=60, b=50),
        ),
        xaxis=dict(gridcolor=THEME["gridline"]),
        yaxis=dict(gridcolor=THEME["gridline"], tickformat=".0%"),
        xaxis2=dict(gridcolor=THEME["gridline"], title="Years"),
        yaxis2=dict(gridcolor=THEME["gridline"], title="% of paths", ticksuffix="%"),
        legend=dict(bgcolor=THEME["panel"], bordercolor=THEME["border"], borderwidth=1),
    )
    return fig
