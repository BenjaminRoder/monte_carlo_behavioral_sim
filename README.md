# Behavioral Finance Monte Carlo Simulator

A quantitative portfolio simulation that models how investor psychology
affects long-run wealth outcomes, using fat-tailed return distributions
and behavioral economics research.

## Live Demo

[Streamlit Community Cloud](YOUR_DEPLOYED_URL_HERE)

## Overview

This project simulates 10,000+ portfolio paths for three investor archetypes:

| Archetype | Behavior | Based on |
|---|---|---|
| **Rational** | Buy-and-hold, quarterly rebalancing | Markowitz (1952) |
| **Loss-Averse** | Panic-sells after drawdown threshold, slow re-entry | Kahneman & Tversky (1979) |
| **Overconfident** | Concentrates in recent winners, over-trades | Barber & Odean (2001) |

The "bias cost" — the dollar gap between rational and behavioral outcomes —
is the central finding.

## Features

- **Fat-tailed returns**: Meta-elliptical copula with per-asset MLE-fitted t-distributions (NOT multivariate normal or a shared-df multivariate-t)
- **Regime-switching correlations**: Three-regime Markov chain (Calm / Normal / Crisis) calibrated from historical volatility percentiles
- **Behavioral state machines**: Each investor has realistic, research-grounded decision rules
- **Probabilistic panic model**: Loss-Averse investor uses a logistic probability curve, not a hard threshold
- **Full quant metrics**: Sharpe, VaR, CVaR, max drawdown, P(loss), bias cost
- **Interactive Streamlit dashboard**: Sliders for every parameter, tooltips on all controls
- **Eight analysis tabs**: Wealth Paths, Distributions, Drawdowns, Bias Cost, Risk/Return, Heatmap, Calibration, Backtest

## Installation

```bash
pip install -r requirements.txt
```

## Running

```bash
streamlit run app.py
```

## Project Structure

```
monte_carlo_behavioral_sim/
├── app.py                        ← Streamlit dashboard
├── src/
│   ├── data.py                   ← Market data loading & t-distribution fitting
│   ├── simulation.py             ← Monte Carlo engine (regime-switching copula)
│   ├── investors.py              ← Rational, Loss-Averse, Overconfident classes
│   ├── metrics.py                ← VaR, CVaR, Sharpe, bias cost
│   ├── visualizations.py         ← Plotly charts
│   ├── backtest.py               ← Historical backtest validation
│   └── calibration.py            ← Distribution fit diagnostics
├── data/
│   ├── spy.csv                   ← SPY daily prices (Stooq, 2005–2024)
│   ├── efa.csv                   ← EFA daily prices (Stooq, 2005–2024)
│   └── agg.csv                   ← AGG daily prices (Stooq, 2005–2024)
├── notebooks/
│   └── behavioral_monte_carlo.ipynb
└── requirements.txt
```

## Key Academic References

- **Kahneman, D. & Tversky, A. (1979)** — Prospect Theory: An Analysis of Decision Under Risk. *Econometrica*.
- **Barber, B. & Odean, T. (2001)** — Boys Will Be Boys: Gender, Overconfidence, and Common Stock Investment. *Quarterly Journal of Economics*.
- **Malmendier, U. & Tate, G. (2005)** — CEO Overconfidence and Corporate Investment. *Journal of Finance*.
- **Shefrin, H. & Statman, M. (1985)** — The Disposition to Sell Winners Too Early and Ride Losers Too Long. *Journal of Finance*.

## Methodological Notes

**Data**
Real daily prices for SPY, EFA, and AGG sourced from Stooq, covering 2005–2024
(~4,800 trading days). All simulation parameters are derived via MLE from this
data — nothing is hardcoded.

**Why t-distribution instead of normal?**
Equity returns exhibit excess kurtosis and negative skew. MLE-fitted Student's
t-distributions per asset capture crash frequency far better than a Gaussian
model. Fitted degrees of freedom land in the ν ≈ 2–3 range for all three assets,
indicating genuinely fat tails.

**Why meta-elliptical copula instead of multivariate-t?**
Each asset has a different fitted ν (SPY ~2.4, EFA ~2.8, AGG ~2.1). scipy's
`multivariate_t` requires a single shared df — forcing all three assets to share
one df misrepresents the marginals. The meta-elliptical copula correctly preserves
each asset's fitted tail profile while imposing the empirical correlation structure
via Cholesky decomposition.

**Regime-switching correlations**
Empirical analysis shows correlations shift materially across market regimes.
A three-state Markov chain (Calm / Normal / Crisis) is calibrated from historical
volatility percentiles, with separate correlation matrices per regime. Notably,
the equity/bond correlation flips sign in crises — the "flight to quality" effect
reverses in liquidity crises where everything is sold simultaneously (2008, 2020, 2022).

**Logistic panic probability**
The Loss-Averse investor does not panic at a fixed threshold. Panic probability
follows a logistic curve as a function of drawdown depth:

    P(panic | DD) = sigmoid(k × (|DD| − threshold))

Calibrated so P = 50% at the user-set trigger level. This matches Prospect Theory's
non-linear pain function and avoids the unrealistic cliff of a hard threshold.

**Bias cost interpretation**
"Loss aversion cost you $X over 10 years on a $100k portfolio" — this
translates Prospect Theory into a concrete dollar figure suitable for
client communication or academic illustration.

## Portfolio Extensions

- Add momentum factor (returns persistence)
- Model transaction costs for loss-averse investor
- Add a "disposition effect" investor (sells winners too early)
- Bootstrap historical returns instead of parametric distribution
- Live data pull via `yfinance`
