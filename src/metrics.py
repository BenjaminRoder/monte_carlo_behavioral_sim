"""
Portfolio Risk & Performance Metrics
Computes quant-standard metrics across all investor types.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from src.investors import InvestorResult
from src.simulation import SimulationConfig  # noqa: F401 — re-exported for convenience


@dataclass
class PortfolioMetrics:
    """Full metric suite for a single investor type."""
    name: str
    color: str

    # Return metrics
    median_terminal_wealth: float
    mean_terminal_wealth: float
    annualised_return: float          # geometric mean
    annualised_volatility: float

    # Risk metrics
    sharpe_ratio: float
    var_95: float                     # 5th percentile of terminal wealth
    cvar_95: float                    # mean of bottom 5%
    max_drawdown_median: float        # median of per-sim max drawdown
    max_drawdown_p95: float           # 95th pctile of per-sim max drawdown

    # Behavioural
    avg_trades: float
    probability_of_loss: float        # P(terminal < initial)
    probability_beat_inflation: float # P(terminal > inflation-adjusted initial)

    # Convenience
    bias_cost_vs_rational: Optional[float] = None   # filled in after comparison
    bias_cost_pct: Optional[float] = None


def compute_metrics(
    result: InvestorResult,
    config: SimulationConfig,
    risk_free_rate: float = 0.04,
    inflation_rate: float = 0.025,
) -> PortfolioMetrics:
    """Compute full metric suite from an InvestorResult."""
    W0   = config.initial_wealth
    W_T  = result.terminal_wealth
    T    = config.n_years

    median_W   = np.median(W_T)
    mean_W     = np.mean(W_T)

    # Geometric annualised return from median terminal wealth
    ann_return = (median_W / W0) ** (1 / T) - 1

    # Annualised volatility from daily portfolio returns
    # Re-derive daily returns from wealth path
    port_daily = np.diff(result.wealth_paths, axis=1) / result.wealth_paths[:, :-1]
    ann_vol    = port_daily.std(axis=1).mean() * np.sqrt(252)

    sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol > 0 else 0.0

    # Risk
    var_95  = np.percentile(W_T, 5)
    cvar_95 = W_T[W_T <= var_95].mean() if (W_T <= var_95).any() else var_95

    per_sim_max_dd = result.drawdown_paths.min(axis=1)
    max_dd_med     = np.median(per_sim_max_dd)
    max_dd_p95     = np.percentile(per_sim_max_dd, 95)

    prob_loss  = (W_T < W0).mean()
    inflation_W = W0 * (1 + inflation_rate) ** T
    prob_beat_inf = (W_T > inflation_W).mean()

    return PortfolioMetrics(
        name=result.name,
        color=result.color,
        median_terminal_wealth=median_W,
        mean_terminal_wealth=mean_W,
        annualised_return=ann_return,
        annualised_volatility=ann_vol,
        sharpe_ratio=sharpe,
        var_95=var_95,
        cvar_95=cvar_95,
        max_drawdown_median=max_dd_med,
        max_drawdown_p95=max_dd_p95,
        avg_trades=result.trade_counts.mean(),
        probability_of_loss=prob_loss,
        probability_beat_inflation=prob_beat_inf,
    )


def compute_bias_costs(
    metrics_list: list[PortfolioMetrics],
    rational_name: str = "Rational",
) -> list[PortfolioMetrics]:
    """
    Add bias_cost_vs_rational to each non-rational investor.
    Bias cost = rational median wealth − investor median wealth.
    """
    rational = next((m for m in metrics_list if m.name == rational_name), None)
    if rational is None:
        return metrics_list

    for m in metrics_list:
        if m.name != rational_name:
            m.bias_cost_vs_rational = rational.median_terminal_wealth - m.median_terminal_wealth
            m.bias_cost_pct = m.bias_cost_vs_rational / rational.median_terminal_wealth
        else:
            m.bias_cost_vs_rational = 0.0
            m.bias_cost_pct = 0.0

    return metrics_list


def summary_table(metrics_list: list[PortfolioMetrics]) -> list[dict]:
    """Return a list of dicts suitable for display in a dataframe or table."""
    rows = []
    for m in metrics_list:
        rows.append({
            "Investor Type":        m.name,
            "Median Wealth":        f"${m.median_terminal_wealth:,.0f}",
            "Ann. Return":          f"{m.annualised_return:.2%}",
            "Ann. Volatility":      f"{m.annualised_volatility:.2%}",
            "Sharpe Ratio":         f"{m.sharpe_ratio:.2f}",
            "VaR (5%)":             f"${m.var_95:,.0f}",
            "CVaR (5%)":            f"${m.cvar_95:,.0f}",
            "Max Drawdown (med)":   f"{m.max_drawdown_median:.2%}",
            "P(Loss)":              f"{m.probability_of_loss:.2%}",
            "P(Beat Inflation)":    f"{m.probability_beat_inflation:.2%}",
            "Avg Trades":           f"{m.avg_trades:.0f}",
            "Bias Cost ($)":        f"${m.bias_cost_vs_rational:,.0f}" if m.bias_cost_vs_rational is not None else "—",
        })
    return rows
