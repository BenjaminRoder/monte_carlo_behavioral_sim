"""
Visualization Module
Plotly-based charts for the behavioral finance dashboard.
All charts use a consistent dark financial aesthetic.
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from src.investors import InvestorResult
from src.metrics import PortfolioMetrics


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------
THEME = dict(
    bg          = "#07070A",
    panel       = "#0D0D12",
    border      = "rgba(255,255,255,0.055)",
    text_pri    = "#DDD7CB",
    text_sec    = "#6E665E",
    gridline    = "rgba(255,255,255,0.042)",
    gold        = "#C8960C",
    font_family = "Fira Code, monospace",
)

BASE_LAYOUT = dict(
    paper_bgcolor = THEME["bg"],
    plot_bgcolor  = THEME["panel"],
    font          = dict(family=THEME["font_family"], color=THEME["text_pri"], size=11),
    margin        = dict(l=50, r=20, t=50, b=50),
    xaxis         = dict(gridcolor=THEME["gridline"], linecolor=THEME["border"], zerolinecolor=THEME["gridline"]),
    yaxis         = dict(gridcolor=THEME["gridline"], linecolor=THEME["border"], zerolinecolor=THEME["gridline"]),
    legend        = dict(bgcolor=THEME["panel"], bordercolor=THEME["border"], borderwidth=1),
)


def _apply_theme(fig: go.Figure) -> go.Figure:
    fig.update_layout(**BASE_LAYOUT)
    return fig


# ---------------------------------------------------------------------------
# 1. Terminal Wealth Distribution (overlapping KDE / histogram)
# ---------------------------------------------------------------------------

def plot_wealth_distributions(results: list[InvestorResult], initial_wealth: float) -> go.Figure:
    fig = go.Figure()

    for r in results:
        W = r.terminal_wealth
        # Bin edges
        lo, hi = np.percentile(W, 1), np.percentile(W, 99)
        bins = np.linspace(lo, hi, 120)
        counts, edges = np.histogram(W, bins=bins, density=True)
        centres = (edges[:-1] + edges[1:]) / 2

        fig.add_trace(go.Scatter(
            x=centres, y=counts,
            mode="lines",
            name=r.name,
            line=dict(color=r.color, width=2.5),
            fill="tozeroy",
            fillcolor=r.color.replace(")", ", 0.10)").replace("rgb(", "rgba("),
        ))

    # Vertical line for initial wealth
    fig.add_vline(
        x=initial_wealth,
        line=dict(color=THEME["gold"], width=1.2, dash="dash"),
        annotation_text="Initial",
        annotation_font_color=THEME["gold"],
    )

    fig.update_layout(
        title="Terminal wealth distribution after 10 years",
        xaxis_title="Terminal wealth ($)",
        yaxis_title="Probability density",
        **BASE_LAYOUT,
    )
    return fig


# ---------------------------------------------------------------------------
# 2. Median wealth paths (fan chart)
# ---------------------------------------------------------------------------

def plot_wealth_paths(results: list[InvestorResult], n_years: int) -> go.Figure:
    fig = go.Figure()
    days = results[0].wealth_paths.shape[1]
    x    = np.linspace(0, n_years, days)

    for r in results:
        W = r.wealth_paths
        med   = np.median(W, axis=0)
        p25   = np.percentile(W, 25, axis=0)
        p75   = np.percentile(W, 75, axis=0)
        p10   = np.percentile(W, 10, axis=0)
        p90   = np.percentile(W, 90, axis=0)

        # 10-90 band
        fig.add_trace(go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([p90, p10[::-1]]),
            fill="toself",
            fillcolor=r.color.replace(")", ", 0.06)").replace("rgb(", "rgba("),
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False, hoverinfo="skip",
        ))
        # 25-75 band
        fig.add_trace(go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([p75, p25[::-1]]),
            fill="toself",
            fillcolor=r.color.replace(")", ", 0.14)").replace("rgb(", "rgba("),
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False, hoverinfo="skip",
        ))
        # Median line
        fig.add_trace(go.Scatter(
            x=x, y=med,
            mode="lines",
            name=r.name,
            line=dict(color=r.color, width=2.5),
        ))

    fig.update_layout(
        title="Median wealth path (with 25–75 and 10–90 percentile bands)",
        xaxis_title="Years",
        yaxis_title="Portfolio value ($)",
        **BASE_LAYOUT,
    )
    return fig


# ---------------------------------------------------------------------------
# 3. Drawdown comparison
# ---------------------------------------------------------------------------

def plot_drawdowns(results: list[InvestorResult], n_years: int) -> go.Figure:
    fig = go.Figure()
    days = results[0].drawdown_paths.shape[1]
    x    = np.linspace(0, n_years, days)

    for r in results:
        D = r.drawdown_paths
        med = np.median(D, axis=0)
        p10 = np.percentile(D, 10, axis=0)   # worst 10%

        fig.add_trace(go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([med, p10[::-1]]),
            fill="toself",
            fillcolor=r.color.replace(")", ", 0.10)").replace("rgb(", "rgba("),
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=x, y=med,
            mode="lines",
            name=r.name,
            line=dict(color=r.color, width=2),
        ))

    fig.update_layout(
        title="Median drawdown path (shaded: 10th percentile — worst scenarios)",
        xaxis_title="Years",
        yaxis_title="Drawdown from peak",
        yaxis_tickformat=".0%",
        **BASE_LAYOUT,
    )
    return fig


# ---------------------------------------------------------------------------
# 4. Bias cost bar chart
# ---------------------------------------------------------------------------

def plot_bias_cost(metrics_list: list[PortfolioMetrics], initial_wealth: float) -> go.Figure:
    non_rational = [m for m in metrics_list if m.name != "Rational"]

    fig = go.Figure(go.Bar(
        x=[m.name for m in non_rational],
        y=[m.bias_cost_vs_rational for m in non_rational],
        marker_color=[m.color for m in non_rational],
        text=[f"${m.bias_cost_vs_rational:,.0f}<br>({m.bias_cost_pct:.1%})" for m in non_rational],
        textposition="outside",
        textfont=dict(color=THEME["text_pri"]),
        width=0.4,
    ))

    fig.update_layout(
        title="Behavioral bias cost vs rational investor (median terminal wealth)",
        xaxis_title="Investor type",
        yaxis_title="Wealth gap ($)",
        **BASE_LAYOUT,
    )
    return fig


# ---------------------------------------------------------------------------
# 5. Risk-Return scatter
# ---------------------------------------------------------------------------

def plot_risk_return(metrics_list: list[PortfolioMetrics]) -> go.Figure:
    fig = go.Figure()

    for m in metrics_list:
        fig.add_trace(go.Scatter(
            x=[m.annualised_volatility],
            y=[m.annualised_return],
            mode="markers+text",
            name=m.name,
            marker=dict(color=m.color, size=18, line=dict(color="white", width=1.5)),
            text=[m.name],
            textposition="top center",
            textfont=dict(color=m.color, size=11),
        ))

    fig.update_layout(
        title="Risk-return profile",
        xaxis_title="Annualised volatility",
        yaxis_title="Annualised return (geometric)",
        xaxis_tickformat=".1%",
        yaxis_tickformat=".1%",
        showlegend=False,
        **BASE_LAYOUT,
    )
    return fig


# ---------------------------------------------------------------------------
# 6. Metric heatmap
# ---------------------------------------------------------------------------

def plot_metric_heatmap(metrics_list: list[PortfolioMetrics]) -> go.Figure:
    names = [m.name for m in metrics_list]
    metrics_raw = {
        "Ann. Return":     [m.annualised_return for m in metrics_list],
        "Ann. Volatility": [m.annualised_volatility for m in metrics_list],
        "Sharpe":          [m.sharpe_ratio for m in metrics_list],
        "P(Beat Infl.)":   [m.probability_beat_inflation for m in metrics_list],
        "P(Loss)":         [m.probability_of_loss for m in metrics_list],
        "Max DD (med)":    [abs(m.max_drawdown_median) for m in metrics_list],
    }

    z   = np.array(list(metrics_raw.values()))
    z_n = (z - z.min(axis=1, keepdims=True)) / (np.ptp(z, axis=1, keepdims=True) + 1e-9)

    # Invert "bad" metrics so green = good
    bad_rows = [1, 4, 5]  # volatility, P(loss), max DD
    z_n[bad_rows] = 1 - z_n[bad_rows]

    text_vals = []
    fmts = [".2%", ".2%", ".2f", ".2%", ".2%", ".2%"]
    for i, (k, vals) in enumerate(metrics_raw.items()):
        row = []
        for v in vals:
            row.append(f"{v:{fmts[i]}}")
        text_vals.append(row)

    fig = go.Figure(go.Heatmap(
        z=z_n,
        x=names,
        y=list(metrics_raw.keys()),
        text=text_vals,
        texttemplate="%{text}",
        colorscale=[[0, "#3A0E08"], [0.5, "#0D0D12"], [1, "#0D3D26"]],
        showscale=False,
        hoverinfo="skip",
    ))

    fig.update_layout(
        title="Metric comparison (green = better)",
        paper_bgcolor=THEME["bg"],
        plot_bgcolor=THEME["panel"],
        font=dict(family=THEME["font_family"], color=THEME["text_pri"], size=11),
        margin=dict(l=120, r=20, t=50, b=50),
        yaxis=dict(autorange="reversed", gridcolor=THEME["gridline"]),
    )
    return fig
