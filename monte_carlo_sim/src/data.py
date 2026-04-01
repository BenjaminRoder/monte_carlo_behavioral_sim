"""
Data Layer
==========
Loads real market data (SPY, EFA, AGG) from local CSVs downloaded from Stooq,
computes daily log-returns, fits a Student's t-distribution to each asset,
and estimates the empirical correlation matrix.

All simulation parameters are derived from this module — nothing is hardcoded.

Live data upgrade:
    Replace load_csv_data() with load_yfinance_data() when running locally
    with internet access. The rest of the pipeline is identical.
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent.parent / "data"

ASSET_FILES = {
    "US Equities (SPY)":    DATA_DIR / "spy.csv",
    "Intl Equities (EFA)":  DATA_DIR / "efa.csv",
    "US Bonds (AGG)":       DATA_DIR / "agg.csv",
}

TARGET_WEIGHTS = np.array([0.60, 0.20, 0.20])


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_csv_data(start: str = "2005-01-01", end: str = "2024-12-31") -> pd.DataFrame:
    """
    Load price data from local Stooq CSVs.
    Returns DataFrame of adjusted close prices, aligned on common trading days.
    """
    frames = {}
    for name, path in ASSET_FILES.items():
        df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
        df = df.sort_index()
        df = df.loc[start:end, ["Close"]]
        df.columns = [name]
        frames[name] = df

    prices = pd.concat(frames.values(), axis=1, join="inner")
    prices.columns = list(ASSET_FILES.keys())
    prices = prices.dropna()
    return prices


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log-returns from price series."""
    return np.log(prices / prices.shift(1)).dropna()


# ---------------------------------------------------------------------------
# Distribution fitting
# ---------------------------------------------------------------------------

@dataclass
class FittedAsset:
    """Fitted parameters for a single asset."""
    name: str
    weight: float

    # Fitted t-distribution parameters (daily scale)
    nu: float               # degrees of freedom — captures fat tails
    mu: float               # location (daily mean log-return)
    sigma: float            # scale parameter from scipy t.fit
    empirical_daily_std: float  # actual std of return series (for vol scaling)
    skew: float             # empirical skewness of the return series

    # Annualised equivalents (for display)
    ann_return: float
    ann_vol: float

    # Goodness of fit
    ks_stat: float   # Kolmogorov-Smirnov statistic vs fitted t
    ks_pval: float


def fit_t_distribution(returns: pd.Series) -> tuple[float, float, float]:
    """
    Fit a Student's t-distribution to a return series.
    Returns (nu, mu, sigma) — degrees of freedom, location, scale.
    scipy uses parameterisation: X = mu + sigma * T(nu)
    """
    nu, mu, sigma = stats.t.fit(returns, method="MLE")
    # Clip nu: must be > 2 for finite variance, cap at 30 (near-normal above that)
    # Floor at 2.1 to honour genuine fat tails in real data (SPY nu ≈ 2.4 historically)
    nu = float(np.clip(nu, 2.1, 30.0))
    return nu, float(mu), float(sigma)


def fit_assets(
    log_returns: pd.DataFrame,
    weights: np.ndarray = TARGET_WEIGHTS,
) -> list[FittedAsset]:
    """
    Fit t-distribution to each asset's return series.
    Returns list of FittedAsset objects with all parameters data-driven.
    """
    fitted = []
    for i, col in enumerate(log_returns.columns):
        r = log_returns[col].dropna()
        nu, mu, sigma = fit_t_distribution(r)

        # KS test against the fitted distribution
        ks_stat, ks_pval = stats.kstest(r, "t", args=(nu, mu, sigma))

        # Empirical daily std (used for vol normalisation in simulation)
        empirical_daily_std = float(r.std())

        # Annualise using empirical std (more stable than theoretical t-std at low nu)
        ann_return = mu * 252
        ann_vol    = empirical_daily_std * np.sqrt(252)

        fitted.append(FittedAsset(
            name=col,
            weight=weights[i],
            nu=nu,
            mu=mu,
            sigma=sigma,
            empirical_daily_std=empirical_daily_std,
            skew=float(r.skew()),
            ann_return=ann_return,
            ann_vol=ann_vol,
            ks_stat=float(ks_stat),
            ks_pval=float(ks_pval),
        ))

    return fitted


# ---------------------------------------------------------------------------
# Correlation matrix
# ---------------------------------------------------------------------------

def estimate_correlation(log_returns: pd.DataFrame) -> np.ndarray:
    """
    Estimate empirical Pearson correlation matrix from daily log-returns.
    Uses the full overlapping sample period.
    """
    corr = log_returns.corr().values
    # Ensure positive semi-definite (numerical safety)
    eigvals = np.linalg.eigvalsh(corr)
    if eigvals.min() < 0:
        # Nearest PSD via eigenvalue clipping
        eigvals, eigvecs = np.linalg.eigh(corr)
        eigvals = np.maximum(eigvals, 1e-8)
        corr = eigvecs @ np.diag(eigvals) @ eigvecs.T
        # Re-normalise to correlation matrix
        d = np.sqrt(np.diag(corr))
        corr = corr / np.outer(d, d)
    return corr


# ---------------------------------------------------------------------------
# Regime-aware correlation (bonus: rolling 1-year correlation)
# ---------------------------------------------------------------------------

def rolling_correlation(
    log_returns: pd.DataFrame,
    window: int = 252,
) -> pd.DataFrame:
    """
    Rolling pairwise correlations over a 1-year window.
    Returns DataFrame with columns = asset pairs, index = date.
    Useful for showing correlation breakdown in crises.
    """
    pairs = []
    cols  = log_returns.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pair_name = f"{cols[i]} / {cols[j]}"
            roll = log_returns[cols[i]].rolling(window).corr(log_returns[cols[j]])
            pairs.append(roll.rename(pair_name))

    return pd.concat(pairs, axis=1).dropna()


# ---------------------------------------------------------------------------
# Master loader — single call returns everything the simulation needs
# ---------------------------------------------------------------------------

@dataclass
class MarketData:
    prices:       pd.DataFrame
    log_returns:  pd.DataFrame
    fitted_assets: list[FittedAsset]
    correlation:  np.ndarray
    rolling_corr: pd.DataFrame
    start_date:   str
    end_date:     str
    n_obs:        int


def load_market_data(
    start: str = "2005-01-01",
    end:   str = "2024-12-31",
) -> MarketData:
    """
    Single entry point — load, clean, fit, correlate.
    Returns MarketData with everything the simulation and dashboard need.
    """
    prices      = load_csv_data(start=start, end=end)
    log_returns = compute_log_returns(prices)
    fitted      = fit_assets(log_returns)
    corr        = estimate_correlation(log_returns)
    roll_corr   = rolling_correlation(log_returns)

    return MarketData(
        prices=prices,
        log_returns=log_returns,
        fitted_assets=fitted,
        correlation=corr,
        rolling_corr=roll_corr,
        start_date=str(log_returns.index[0].date()),
        end_date=str(log_returns.index[-1].date()),
        n_obs=len(log_returns),
    )
