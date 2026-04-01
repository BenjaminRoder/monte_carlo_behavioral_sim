"""
Historical Backtest Validation
================================
Runs the Rational investor strategy on the ACTUAL historical SPY/EFA/AGG
return sequence (not simulated) to verify the model is correctly calibrated.

A well-calibrated simulation should produce a distribution of outcomes whose
median is close to what actually happened over the same period.

Three checks:
  1. Realised historical return for the 60/20/20 portfolio
  2. Simulated median vs actual — should be in the same ballpark
  3. Where the actual outcome falls in the simulated distribution

This is the "sanity check" that proves the simulation isn't generating
fantasy numbers disconnected from reality.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data import MarketData
from src.investors import RationalInvestor, InvestorResult

THEME = dict(
    bg="       #0D1117", panel="#161B22", border="#30363D",
    text_pri="#E6EDF3",  text_sec="#8B949E", gridline="#21262D",
    font_family="IBM Plex Mono, monospace",
)
THEME["bg"] = "#0D1117"   # strip whitespace artefact


def run_historical_backtest(
    market_data: MarketData,
    weights: np.ndarray = np.array([0.60, 0.20, 0.20]),
    initial_wealth: float = 100_000.0,
) -> dict:
    """
    Run the buy-and-hold strategy on the actual historical return sequence.

    Returns dict with:
      wealth_path   : pd.Series — daily portfolio value indexed by date
      total_return  : float — total return over full period
      ann_return    : float — geometric annualised return
      ann_vol       : float — annualised daily vol
      sharpe        : float — Sharpe ratio (rf=0 for simplicity)
      max_drawdown  : float — maximum peak-to-trough drawdown
      cagr_spy      : float — SPY-only CAGR over same period (benchmark)
    """
    lr     = market_data.log_returns
    prices = market_data.prices
    n_days = len(lr)

    # Rebalance quarterly on actual data
    rebal_freq = 63
    asset_vals = np.array(weights) * initial_wealth
    path = [initial_wealth]

    for d in range(n_days):
        r          = lr.iloc[d].values
        asset_vals = asset_vals * (1 + r)
        total      = asset_vals.sum()
        if (d + 1) % rebal_freq == 0:
            asset_vals = total * weights
        path.append(total)

    wealth_series = pd.Series(path, index=[prices.index[0]] + list(lr.index))
    total_return  = wealth_series.iloc[-1] / initial_wealth - 1
    n_years       = n_days / 252
    ann_return    = (1 + total_return) ** (1 / n_years) - 1

    port_daily    = lr @ weights
    ann_vol       = port_daily.std() * np.sqrt(252)
    sharpe        = ann_return / ann_vol if ann_vol > 0 else 0.0

    peak       = wealth_series.cummax()
    drawdown   = (wealth_series - peak) / peak
    max_dd     = drawdown.min()

    spy_total  = prices.iloc[-1, 0] / prices.iloc[0, 0] - 1
    cagr_spy   = (1 + spy_total) ** (1 / n_years) - 1

    return {
        "wealth_path":   wealth_series,
        "total_return":  total_return,
        "ann_return":    ann_return,
        "ann_vol":       ann_vol,
        "sharpe":        sharpe,
        "max_drawdown":  max_dd,
        "cagr_spy":      cagr_spy,
        "n_years":       n_years,
        "start_date":    market_data.start_date,
        "end_date":      market_data.end_date,
    }


def plot_backtest_vs_simulation(
    backtest: dict,
    sim_result: InvestorResult,
    config,
    market_data: MarketData,
) -> go.Figure:
    """
    Two-panel chart:
      Top: Actual historical wealth path vs simulated median + percentile bands
      Bottom: Where the actual terminal wealth falls in the simulated distribution
    """
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=[
            "Historical portfolio path vs simulated distribution",
            "Actual terminal wealth in simulated distribution",
        ],
        vertical_spacing=0.14,
        row_heights=[0.60, 0.40],
    )

    # ── Top panel: wealth paths ──
    W      = sim_result.wealth_paths
    n_days_sim = W.shape[1] - 1
    x_sim  = np.linspace(0, config.n_years, W.shape[1])

    for plo, phi, alpha in [(5, 95, 0.06), (25, 75, 0.12)]:
        lo = np.percentile(W, plo, axis=0)
        hi = np.percentile(W, phi, axis=0)
        fig.add_trace(go.Scatter(
            x=np.concatenate([x_sim, x_sim[::-1]]),
            y=np.concatenate([hi, lo[::-1]]),
            fill="toself",
            fillcolor=f"rgba(74,158,255,{alpha})",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=True if plo == 5 else False,
            name=f"Sim p{plo}–p{phi}",
            hoverinfo="skip",
        ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=x_sim, y=np.median(W, axis=0),
        mode="lines", name="Sim median",
        line=dict(color="#4A9EFF", width=2),
    ), row=1, col=1)

    # Scale historical path to match simulation start year spacing
    hist_path  = backtest["wealth_path"]
    x_hist     = np.linspace(0, backtest["n_years"], len(hist_path))
    fig.add_trace(go.Scatter(
        x=x_hist, y=hist_path.values,
        mode="lines", name="Actual history",
        line=dict(color="#FFD700", width=2.5),
    ), row=1, col=1)

    # Initial wealth reference
    fig.add_hline(
        y=config.initial_wealth,
        line=dict(color="#8B949E", width=0.8, dash="dash"),
        row=1, col=1,
    )

    # ── Bottom panel: terminal wealth distribution ──
    terminal = sim_result.terminal_wealth
    lo_b, hi_b = np.percentile(terminal, 0.5), np.percentile(terminal, 99.5)
    bins       = np.linspace(lo_b, hi_b, 100)
    counts, edges = np.histogram(terminal, bins=bins, density=True)
    centres       = (edges[:-1] + edges[1:]) / 2

    fig.add_trace(go.Bar(
        x=centres, y=counts,
        marker_color="rgba(74,158,255,0.35)",
        marker_line_color="#4A9EFF",
        marker_line_width=0.5,
        name="Simulated distribution",
        showlegend=True,
    ), row=2, col=1)

    # Mark actual terminal wealth
    actual_terminal = backtest["wealth_path"].iloc[-1]
    pct_rank = (terminal < actual_terminal).mean() * 100
    fig.add_vline(
        x=actual_terminal,
        line=dict(color="#FFD700", width=2),
        annotation_text=f"Actual: ${actual_terminal:,.0f} (p{pct_rank:.0f})",
        annotation_font=dict(color="#FFD700", size=10),
        annotation_position="top right",
        row=2, col=1,
    )

    base = dict(
        paper_bgcolor=THEME["bg"], plot_bgcolor=THEME["panel"],
        font=dict(family=THEME["font_family"], color=THEME["text_pri"], size=11),
        margin=dict(l=60, r=20, t=60, b=50),
        legend=dict(bgcolor=THEME["panel"], bordercolor=THEME["border"], borderwidth=1),
    )
    fig.update_layout(
        title=(f"Backtest validation: {backtest['start_date']} → {backtest['end_date']} "
               f"| Actual ann. return: {backtest['ann_return']:.2%}"),
        **base,
    )
    for row in [1, 2]:
        fig.update_xaxes(gridcolor=THEME["gridline"], row=row, col=1)
        fig.update_yaxes(gridcolor=THEME["gridline"], row=row, col=1)
    fig.update_xaxes(title_text="Years", row=1, col=1)
    fig.update_xaxes(title_text="Terminal wealth ($)", row=2, col=1)
    fig.update_yaxes(title_text="Portfolio value ($)", row=1, col=1)

    return fig


def backtest_summary_table(backtest: dict, sim_result: InvestorResult, config) -> list[dict]:
    """Comparison table: actual history vs simulation statistics."""
    terminal     = sim_result.terminal_wealth
    actual_term  = backtest["wealth_path"].iloc[-1]
    pct_rank     = (terminal < actual_term).mean() * 100

    return [
        {"Metric":           "Period",
         "Actual history":   f"{backtest['start_date']} → {backtest['end_date']}",
         "Simulation":       f"{config.n_years}-year forward"},
        {"Metric":           "Ann. return (geometric)",
         "Actual history":   f"{backtest['ann_return']:.2%}",
         "Simulation":       f"{(np.median(terminal)/config.initial_wealth)**(1/config.n_years)-1:.2%} (median path)"},
        {"Metric":           "Ann. volatility",
         "Actual history":   f"{backtest['ann_vol']:.2%}",
         "Simulation":       f"{np.diff(sim_result.wealth_paths, axis=1).std(axis=1).mean() / np.mean(sim_result.wealth_paths[:,:-1]) * np.sqrt(252):.2%}"},
        {"Metric":           "Max drawdown",
         "Actual history":   f"{backtest['max_drawdown']:.2%}",
         "Simulation":       f"{sim_result.drawdown_paths.min(axis=1).mean():.2%} (mean across sims)"},
        {"Metric":           "Terminal wealth",
         "Actual history":   f"${actual_term:,.0f}",
         "Simulation":       f"${np.median(terminal):,.0f} (median)"},
        {"Metric":           "Actual outcome percentile",
         "Actual history":   f"p{pct_rank:.0f} of simulated distribution",
         "Simulation":       "—"},
        {"Metric":           "SPY buy-and-hold CAGR",
         "Actual history":   f"{backtest['cagr_spy']:.2%}",
         "Simulation":       "Portfolio CAGR shown above"},
    ]
