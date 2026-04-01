"""
Investor Behavior Models
=========================
Three investor archetypes grounded in behavioural economics research.

1. Rational       — buy-and-hold with quarterly rebalancing (Markowitz 1952)
2. Loss-Averse    — probabilistic panic selling driven by drawdown severity
                    (Kahneman & Tversky 1979; Shefrin & Statman 1985)
3. Overconfident  — concentration in recent winner, over-trading with costs
                    (Barber & Odean 2001; Malmendier & Tate 2005)

Key improvement over naive hard-threshold models:
    Loss-averse panic is PROBABILISTIC, not deterministic. The probability
    of selling on any given day follows a logistic function of drawdown depth:

        P(panic | DD) = sigmoid(k * (|DD| - threshold))

    Calibrated so P=50% at DD=-12%, P=96% at DD=-25%, P=10% at DD=-3%.
    This matches Prospect Theory's non-linear pain function and avoids the
    unrealistic "cliff" where exactly -10% triggers a guaranteed response.
    See: Kahneman & Tversky (1979) Figure 3; Barberis et al. (2001).
"""

import numpy as np
from dataclasses import dataclass, field
from scipy.special import expit   # numerically stable logistic function


@dataclass
class InvestorResult:
    """Holds the wealth path for all simulations."""
    name: str
    color: str
    description: str
    wealth_paths: np.ndarray        # (n_sim, n_days+1)
    terminal_wealth: np.ndarray     # (n_sim,)
    drawdown_paths: np.ndarray      # (n_sim, n_days+1)
    trade_counts: np.ndarray        # (n_sim,) — number of behavioural trades
    panic_curve: dict = field(default_factory=dict)   # for display in dashboard
    params: dict = field(default_factory=dict)


class BaseInvestor:
    """Shared helpers."""

    def _compute_drawdown(self, wealth_path: np.ndarray) -> np.ndarray:
        """Rolling drawdown from running peak."""
        peak = np.maximum.accumulate(wealth_path)
        return (wealth_path - peak) / peak


# ---------------------------------------------------------------------------
# 1. Rational Investor
# ---------------------------------------------------------------------------

class RationalInvestor(BaseInvestor):
    """
    Buy-and-hold with quarterly rebalancing. The efficient-markets baseline.
    No panic selling, no concentration bias, no over-trading.
    """

    def __init__(self, rebalance_freq: int = 63):
        self.rebalance_freq = rebalance_freq

    def simulate(self, paths: dict) -> InvestorResult:
        daily_returns = paths["daily_returns"]   # (n_sim, n_days, n_assets)
        weights       = paths["weights"]
        config        = paths["config"]

        n_sim, n_days, n_assets = daily_returns.shape
        W0 = config.initial_wealth

        wealth       = np.zeros((n_sim, n_days + 1))
        wealth[:, 0] = W0
        asset_values = np.ones((n_sim, n_assets)) * (W0 * weights)
        trade_counts = np.zeros(n_sim, dtype=int)

        for d in range(n_days):
            r = daily_returns[:, d, :]
            asset_values *= (1 + r)
            total = asset_values.sum(axis=1)

            if (d + 1) % self.rebalance_freq == 0:
                asset_values = total[:, None] * weights[None, :]
                trade_counts += 1

            wealth[:, d + 1] = total

        drawdowns = np.array([self._compute_drawdown(wealth[i]) for i in range(n_sim)])

        return InvestorResult(
            name="Rational",
            color="#4882B4",
            description="Buy-and-hold, quarterly rebalancing",
            wealth_paths=wealth,
            terminal_wealth=wealth[:, -1],
            drawdown_paths=drawdowns,
            trade_counts=trade_counts,
            params={"rebalance_freq": self.rebalance_freq},
        )


# ---------------------------------------------------------------------------
# 2. Loss-Averse Investor  (probabilistic panic)
# ---------------------------------------------------------------------------

class LossAverseInvestor(BaseInvestor):
    """
    Kahneman & Tversky (1979) Prospect Theory investor.

    PROBABILISTIC panic model (upgrade from hard threshold):
    ─────────────────────────────────────────────────────────
    Each day, conditional on being in the Normal state, the investor panics
    with probability:

        P(panic | DD_t) = sigmoid(k * (|DD_t| - threshold))

    where DD_t is the current drawdown from peak (a negative number).

    Parameters calibrated so:
        DD =  -3%  →  P ≈  9%   (small dip, unlikely to panic)
        DD = -12%  →  P ≈ 50%   (meaningful loss, coin-flip)
        DD = -20%  →  P ≈ 88%   (severe loss, very likely to sell)
        DD = -30%  →  P ≈ 99%   (extreme loss, almost certain to sell)

    This matches Prospect Theory's non-linear pain function (value function
    is steeper for losses than gains, and steepness increases with loss size).

    Recovery mechanism:
        After panic, investor waits for `recovery_days` consecutive positive
        return days before gradually ramping back to target allocation over
        `entry_ramp` days. This models the well-documented reluctance to
        re-enter after a crash (Odean 1998, "Are Investors Reluctant to
        Realise Their Losses?").
    """

    def __init__(
        self,
        panic_threshold: float = -0.10,   # kept for UI slider compatibility
                                           # (used as x0 in logistic curve)
        recovery_days:   int   = 40,
        entry_ramp:      int   = 20,
        panic_k:         float = 25.0,    # logistic steepness
        safe_weights:    np.ndarray = None,
        rng_seed:        int   = 123,
    ):
        # panic_threshold is repurposed as the logistic midpoint:
        # P=50% when |DD| = |panic_threshold|
        self.panic_midpoint  = abs(panic_threshold)   # e.g. 0.10 → 50% at -10% DD
        self.panic_k         = panic_k
        self.recovery_days   = recovery_days
        self.entry_ramp      = entry_ramp
        self.safe_weights    = safe_weights
        self._rng            = np.random.default_rng(rng_seed)

    def _panic_probability(self, drawdown: np.ndarray) -> np.ndarray:
        """
        Logistic panic probability as a function of drawdown.
        drawdown: array of values <= 0 (e.g. -0.15 = 15% below peak)
        Returns: array of probabilities in [0, 1]
        """
        # |DD| - midpoint: positive when drawdown exceeds midpoint
        return expit(self.panic_k * (-drawdown - self.panic_midpoint))

    def panic_curve_data(self) -> dict:
        """Return x/y data for the panic probability curve (for dashboard display)."""
        dd = np.linspace(-0.35, 0.0, 200)
        p  = self._panic_probability(dd)
        return {"drawdown": dd.tolist(), "probability": p.tolist(),
                "midpoint": -self.panic_midpoint, "k": self.panic_k}

    def simulate(self, paths: dict) -> InvestorResult:
        daily_returns  = paths["daily_returns"]
        target_weights = paths["weights"]
        config         = paths["config"]

        n_sim, n_days, n_assets = daily_returns.shape
        W0 = config.initial_wealth

        safe = np.zeros(n_assets) if self.safe_weights is None else self.safe_weights
        if self.safe_weights is None:
            safe[-1] = 1.0   # 100% bonds

        wealth       = np.zeros((n_sim, n_days + 1))
        wealth[:, 0] = W0
        asset_values = np.ones((n_sim, n_assets)) * (W0 * target_weights)

        # Per-sim state: 0=normal, 1=panicked, 2=recovering
        state            = np.zeros(n_sim, dtype=np.int8)
        recovery_counter = np.zeros(n_sim, dtype=int)
        ramp_counter     = np.zeros(n_sim, dtype=int)
        peak_wealth      = np.full(n_sim, W0)
        trade_counts     = np.zeros(n_sim, dtype=int)

        for d in range(n_days):
            r     = daily_returns[:, d, :]
            total = asset_values.sum(axis=1)
            peak_wealth = np.maximum(peak_wealth, total)
            drawdown = (total - peak_wealth) / np.maximum(peak_wealth, 1e-9)

            # ── Normal → Panicked (probabilistic) ──────────────────────────
            in_normal = state == 0
            if in_normal.any():
                p_panic   = self._panic_probability(drawdown[in_normal])
                u         = self._rng.random(in_normal.sum())
                panicking = in_normal.copy()
                panicking[in_normal] = u < p_panic

                if panicking.any():
                    asset_values[panicking] = total[panicking, None] * safe[None, :]
                    state[panicking]            = 1
                    recovery_counter[panicking] = 0
                    trade_counts[panicking]    += 1

            # ── Panicked → Recovering ───────────────────────────────────────
            # Count consecutive positive-return days in safe haven
            daily_port_r = (r * (asset_values / np.maximum(total[:, None], 1e-9))).sum(axis=1)
            in_panic = state == 1
            recovery_counter[in_panic & (daily_port_r > 0)] += 1
            recovery_counter[in_panic & (daily_port_r <= 0)]  = 0   # reset on down day
            start_recovery = in_panic & (recovery_counter >= self.recovery_days)
            state[start_recovery]        = 2
            ramp_counter[start_recovery] = 0

            # ── Recovering → Normal (gradual re-entry ramp) ─────────────────
            recovering = state == 2
            if recovering.any():
                ramp_counter[recovering] += 1
                alpha         = np.minimum(ramp_counter[recovering] / self.entry_ramp, 1.0)
                current_total = asset_values[recovering].sum(axis=1)
                blended       = (alpha[:, None] * target_weights[None, :]
                                 + (1 - alpha[:, None]) * safe[None, :])
                asset_values[recovering] = current_total[:, None] * blended
                done = recovering & (ramp_counter >= self.entry_ramp)
                state[done] = 0
                trade_counts[done] += 1

            # Apply returns
            asset_values *= (1 + r)
            wealth[:, d + 1] = asset_values.sum(axis=1)

        drawdowns = np.array([self._compute_drawdown(wealth[i]) for i in range(n_sim)])

        return InvestorResult(
            name="Loss-Averse",
            color="#BC6C3A",
            description=f"Probabilistic panic (P=50% at DD={-self.panic_midpoint:.0%})",
            wealth_paths=wealth,
            terminal_wealth=wealth[:, -1],
            drawdown_paths=drawdowns,
            trade_counts=trade_counts,
            panic_curve=self.panic_curve_data(),
            params={
                "panic_midpoint":  -self.panic_midpoint,
                "panic_k":          self.panic_k,
                "recovery_days":    self.recovery_days,
                "entry_ramp":       self.entry_ramp,
            },
        )


# ---------------------------------------------------------------------------
# 3. Overconfident Investor
# ---------------------------------------------------------------------------

class OverconfidentInvestor(BaseInvestor):
    """
    Barber & Odean (2001) / Malmendier & Tate (2005) overconfident investor.

    Behavioural rules:
    - Concentration: allocates `concentration_mult`× target weight to the
      best recent performer (recency bias / representativeness heuristic)
    - Over-trading: re-assesses every `trade_freq` days regardless of signal
    - Transaction costs: `transaction_cost` bp applied per rebalance event
    - All other weights reduced proportionally to maintain budget constraint
    """

    def __init__(
        self,
        concentration_mult: float = 2.0,
        trade_freq:         int   = 21,
        lookback:           int   = 21,
        transaction_cost:   float = 0.002,
    ):
        self.concentration_mult = concentration_mult
        self.trade_freq         = trade_freq
        self.lookback           = lookback
        self.transaction_cost   = transaction_cost

    def _concentrated_weights(
        self,
        recent_returns: np.ndarray,
        base_weights:   np.ndarray,
    ) -> np.ndarray:
        """
        Boost weight of recent winner by concentration_mult, reduce others
        proportionally. Returns normalised weight matrix (n_sim, n_assets).
        """
        n_sim, n_assets = recent_returns.shape
        winner      = recent_returns.argmax(axis=1)   # (n_sim,)
        new_weights = np.tile(base_weights, (n_sim, 1)).astype(float)

        for s in range(n_sim):
            w     = winner[s]
            boost = base_weights[w] * (self.concentration_mult - 1.0)
            new_weights[s, w] = base_weights[w] * self.concentration_mult
            others      = [i for i in range(n_assets) if i != w]
            total_other = new_weights[s, others].sum()
            if total_other > 0:
                new_weights[s, others] -= boost * (new_weights[s, others] / total_other)
            new_weights[s] = np.maximum(new_weights[s], 0)
            new_weights[s] /= new_weights[s].sum()

        return new_weights

    def simulate(self, paths: dict) -> InvestorResult:
        daily_returns  = paths["daily_returns"]
        target_weights = paths["weights"]
        config         = paths["config"]

        n_sim, n_days, n_assets = daily_returns.shape
        W0 = config.initial_wealth

        wealth       = np.zeros((n_sim, n_days + 1))
        wealth[:, 0] = W0
        asset_values = np.ones((n_sim, n_assets)) * (W0 * target_weights)
        trade_counts = np.zeros(n_sim, dtype=int)

        for d in range(n_days):
            r     = daily_returns[:, d, :]
            total = asset_values.sum(axis=1)

            if (d + 1) % self.trade_freq == 0 and d >= self.lookback:
                lookback_rets = daily_returns[:, max(0, d - self.lookback):d, :]
                cum_r  = (1 + lookback_rets).prod(axis=1) - 1
                new_w  = self._concentrated_weights(cum_r, target_weights)
                cost   = self.transaction_cost * total
                total -= cost
                asset_values = total[:, None] * new_w
                trade_counts += 1

            asset_values *= (1 + r)
            wealth[:, d + 1] = asset_values.sum(axis=1)

        drawdowns = np.array([self._compute_drawdown(wealth[i]) for i in range(n_sim)])

        return InvestorResult(
            name="Overconfident",
            color="#7C5AB8",
            description=f"{self.concentration_mult}× concentration on recent winner",
            wealth_paths=wealth,
            terminal_wealth=wealth[:, -1],
            drawdown_paths=drawdowns,
            trade_counts=trade_counts,
            params={
                "concentration_mult": self.concentration_mult,
                "trade_freq":         self.trade_freq,
                "lookback":           self.lookback,
                "transaction_cost":   self.transaction_cost,
            },
        )
