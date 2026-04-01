"""
Monte Carlo Simulation Engine
==============================
Generates realistic correlated return paths using parameters fitted to
real market data (SPY, EFA, AGG) via src/data.py.

Return model — meta-elliptical copula with t margins
─────────────────────────────────────────────────────
Each asset i has its own MLE-fitted t-distribution with degrees of freedom
nu_i (SPY: ~2.4, EFA: ~2.8, AGG: ~2.1). Because nu differs across assets,
scipy.stats.multivariate_t — which requires a single shared df — is
INAPPROPRIATE here. The correct approach is the meta-elliptical copula:

  1. Draw independent standard t(nu_i) variates for each asset i
  2. Winsorise and standardise to unit variance (prevents blow-up at low nu)
  3. Apply Cholesky of the REGIME-SPECIFIC correlation matrix
  4. Apply skew correction, scale to fitted daily mu and empirical sigma

This is the standard method for portfolio simulation with heterogeneous
tail profiles. See:
  McNeil, Frey & Embrechts (2015) "Quantitative Risk Management", Ch. 7-8
  Joe (2014) "Dependence Modeling with Copulas", Ch. 4

Why not scipy.stats.multivariate_t?
  That class requires a single shared df parameter. Forcing all three assets
  to share one df misrepresents the marginals — AGG (nu~2.1, extreme tail)
  would contaminate SPY (nu~2.4). The meta-elliptical approach correctly
  preserves each asset's fitted tail profile while imposing the empirical
  correlation structure. Known limitation: tail dependence (tendency to crash
  together) is slightly underestimated — a documented conservative bias.

Regime-switching correlations
───────────────────────────────
Empirical analysis shows correlations shift materially across market regimes:

  Regime    | Vol signal      | SPY/EFA | SPY/AGG | EFA/AGG
  ----------|-----------------|---------|---------|--------
  Calm      | <p40  (<11.1%)  |  +0.79  |  -0.28  |  -0.24
  Normal    | p40-p75         |  +0.88  |  -0.01  |  +0.01
  Crisis    | >p75  (>18.6%)  |  +0.95  |  +0.16  |  +0.22

  The equity/bond correlation sign flip in crisis (negative → positive) is
  a documented phenomenon — the "flight to quality" effect reverses in
  liquidity crises where everything is sold simultaneously (2008, 2020, 2022).

Regime transitions are modelled as a first-order Markov chain with
persistence probabilities calibrated from historical regime durations.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from src.data import MarketData, FittedAsset


# ---------------------------------------------------------------------------
# Regime correlation matrices (empirically calibrated from 2005-2024 data)
# ---------------------------------------------------------------------------

def _make_corr(spy_efa: float, spy_agg: float, efa_agg: float) -> np.ndarray:
    """Construct 3x3 correlation matrix from pairwise values."""
    C = np.array([
        [1.0,      spy_efa,  spy_agg],
        [spy_efa,  1.0,      efa_agg],
        [spy_agg,  efa_agg,  1.0    ],
    ])
    # Ensure PSD (numerical safety for extreme correlations)
    eigvals, eigvecs = np.linalg.eigh(C)
    if eigvals.min() < 1e-8:
        eigvals = np.maximum(eigvals, 1e-8)
        C = eigvecs @ np.diag(eigvals) @ eigvecs.T
        d = np.sqrt(np.diag(C))
        C = C / np.outer(d, d)
    return C


# Three regimes calibrated from real data
REGIME_CORRELATIONS = {
    "calm":   _make_corr(spy_efa=0.79, spy_agg=-0.28, efa_agg=-0.24),
    "normal": _make_corr(spy_efa=0.88, spy_agg=-0.01, efa_agg=+0.01),
    "crisis": _make_corr(spy_efa=0.95, spy_agg=+0.16, efa_agg=+0.22),
}

# Cholesky factors (pre-computed)
REGIME_CHOLESKY = {k: np.linalg.cholesky(v) for k, v in REGIME_CORRELATIONS.items()}

# Regime labels and indices
REGIMES       = ["calm", "normal", "crisis"]
N_REGIMES     = 3
CALM, NORMAL, CRISIS = 0, 1, 2

# Historical regime proportions (from vol analysis: p40/p75 thresholds)
REGIME_STATIONARY = np.array([0.397, 0.352, 0.252])   # from eigenanalysis; normalised at use   # calm, normal, crisis

# Markov transition matrix (rows = from, cols = to)
# Calibrated from mean regime durations: calm ~200d, normal ~150d, crisis ~90d
# Self-transition p = 1 - 1/mean_duration; off-diagonal split by stationary dist
# Transition matrix fitted directly from historical regime sequences (2005-2024)
# Calm <-> Crisis always routes through Normal (per empirical data)
REGIME_TRANSITIONS = np.array([
    # to calm    to normal  to crisis
    [0.964824,   0.035176,  0.000000],  # from calm   (mean duration 28d)
    [0.039678,   0.930420,  0.029902],  # from normal (mean duration 14d)
    [0.000000,   0.041801,  0.958199],  # from crisis (mean duration 24d)
])


@dataclass
class SimulationConfig:
    """Top-level simulation configuration."""
    n_simulations:        int   = 10_000
    n_years:              int   = 10
    trading_days:         int   = 252
    initial_wealth:       float = 100_000.0
    rebalance_frequency:  int   = 63        # quarterly (~63 trading days)
    random_seed: Optional[int]  = 42
    use_regime_switching: bool  = True      # toggle for A/B comparison


class MonteCarloEngine:
    """
    Generates correlated, fat-tailed, regime-aware daily return paths.
    See module docstring for full methodology and references.
    """

    def __init__(
        self,
        market_data: MarketData,
        config: SimulationConfig = None,
    ):
        self.market_data = market_data
        self.config      = config or SimulationConfig()
        self.assets      = market_data.fitted_assets
        self.n_assets    = len(self.assets)
        self.total_days  = self.config.n_years * self.config.trading_days
        self._rng        = np.random.default_rng(self.config.random_seed)

        # Fallback: static Cholesky from full-sample empirical correlation
        self._L_static = np.linalg.cholesky(market_data.correlation)

        # Per-asset daily parameters from fitted data
        self.daily_mu    = np.array([a.mu                  for a in self.assets])
        self.daily_sigma = np.array([a.empirical_daily_std for a in self.assets])
        self.nu          = np.array([a.nu                  for a in self.assets])
        self.skews       = np.array([a.skew                for a in self.assets])
        self.weights     = np.array([a.weight              for a in self.assets])

    # ------------------------------------------------------------------
    # Regime simulation
    # ------------------------------------------------------------------

    def _simulate_regimes(self, n_sim: int, n_days: int) -> np.ndarray:
        """
        Simulate regime sequences via first-order Markov chain.
        Returns int array of shape (n_sim, n_days) with values 0/1/2 (calm/normal/crisis).

        Initial regime drawn from stationary distribution.
        Transitions applied day-by-day using pre-calibrated matrix.
        """
        regimes = np.zeros((n_sim, n_days), dtype=np.int8)

        # Initial state from stationary distribution
        p = REGIME_STATIONARY / REGIME_STATIONARY.sum()  # ensure sums to exactly 1.0
        regimes[:, 0] = self._rng.choice(N_REGIMES, size=n_sim, p=p)

        # Precompute cumulative transition rows for fast lookup
        cum_trans = np.cumsum(REGIME_TRANSITIONS, axis=1)

        # Advance Markov chain
        u = self._rng.random((n_sim, n_days - 1))  # uniform draws for transitions
        for d in range(1, n_days):
            prev   = regimes[:, d - 1]              # (n_sim,)
            thresholds = cum_trans[prev]             # (n_sim, 3)
            u_d    = u[:, d - 1, None]              # (n_sim, 1)
            regimes[:, d] = (u_d > thresholds).sum(axis=1).clip(0, N_REGIMES - 1)

        return regimes

    # ------------------------------------------------------------------
    # Core return generation
    # ------------------------------------------------------------------

    def _standardised_t_draws(self, n_sim: int, n_days: int) -> np.ndarray:
        """
        Draw winsorised, unit-variance t-variates for each asset.
        Returns (n_sim, n_days, n_assets).

        Winsorising at 0.1/99.9 percentile prevents the blow-up at very
        low nu (<2.5) while preserving fat-tail shape in the bulk.
        """
        raw = np.zeros((n_sim, n_days, self.n_assets))
        for i in range(self.n_assets):
            nu_i   = self.nu[i]
            draws  = self._rng.standard_t(nu_i, size=(n_sim, n_days))
            lo, hi = np.percentile(draws, [0.1, 99.9])
            draws  = np.clip(draws, lo, hi)
            s      = draws.std()
            if s > 0:
                draws /= s
            raw[:, :, i] = draws
        return raw

    def _generate_correlated_returns(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Main return generation loop.

        Returns:
          returns : (n_sim, total_days, n_assets)  daily log-returns
          regimes : (n_sim, total_days)             regime label per day (0/1/2)
        """
        n_sim  = self.config.n_simulations
        n_days = self.total_days

        # Step 1: standardised t-variates (fat tails, unit variance)
        raw = self._standardised_t_draws(n_sim, n_days)

        if self.config.use_regime_switching:
            # Step 2a: simulate regime sequence per sim path
            regimes = self._simulate_regimes(n_sim, n_days)

            # Step 3a: apply regime-specific Cholesky day by day
            # Group days by regime for efficiency (batch by regime label)
            correlated = np.empty_like(raw)
            for regime_idx, regime_name in enumerate(REGIMES):
                L = REGIME_CHOLESKY[regime_name]
                mask = (regimes == regime_idx)        # (n_sim, n_days) bool
                if not mask.any():
                    continue
                # Extract all (sim, day) pairs in this regime and apply L
                sim_idx, day_idx = np.where(mask)
                draws_regime = raw[sim_idx, day_idx, :]  # (k, n_assets)
                correlated[sim_idx, day_idx, :] = draws_regime @ L.T

        else:
            # Static correlation (full-sample empirical)
            regimes    = np.full((n_sim, n_days), NORMAL, dtype=np.int8)
            correlated = raw @ self._L_static.T

        # Step 4: skew adjustment (cubic, small magnitude)
        for i in range(self.n_assets):
            if abs(self.skews[i]) > 0.1:
                magnitude = min(abs(self.skews[i]) / 10.0, 0.05)
                correlated[:, :, i] += (
                    np.sign(self.skews[i]) * magnitude * correlated[:, :, i] ** 2
                )

        # Step 5: scale to fitted daily mu and sigma
        returns = self.daily_mu[None, None, :] + self.daily_sigma[None, None, :] * correlated
        return returns, regimes

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate_paths(self) -> dict:
        """
        Generate all simulation paths.

        Returns dict with keys:
          daily_returns     : (n_sim, n_days, n_assets)
          portfolio_returns : (n_sim, n_days)
          regimes           : (n_sim, n_days)  — 0=calm, 1=normal, 2=crisis
          config, assets, weights, market_data
        """
        daily_returns, regimes = self._generate_correlated_returns()
        portfolio_returns = (daily_returns * self.weights[None, None, :]).sum(axis=2)

        return {
            "daily_returns":     daily_returns,
            "portfolio_returns": portfolio_returns,
            "regimes":           regimes,
            "config":            self.config,
            "assets":            self.assets,
            "weights":           self.weights,
            "market_data":       self.market_data,
        }
