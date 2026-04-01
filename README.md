# Behavioral Finance Monte Carlo Simulator

A quantitative portfolio simulation that models how investor psychology
affects long-run wealth outcomes, using fat-tailed return distributions
and behavioral economics research.

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

- **Fat-tailed returns**: Correlated multivariate t-distribution (NOT normal)
- **Behavioral state machines**: Each investor has realistic decision rules
- **Full quant metrics**: Sharpe, VaR, CVaR, max drawdown, P(loss)
- **Interactive Streamlit dashboard**: Sliders for every parameter
- **Six chart types**: Wealth paths, distributions, drawdowns, bias cost, risk-return, heatmap

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
monte_carlo_behavioral/
├── app.py                    ← Streamlit dashboard
├── src/
│   ├── simulation.py         ← Monte Carlo engine (fat-tailed returns)
│   ├── investors.py          ← Rational, Loss-Averse, Overconfident classes
│   ├── metrics.py            ← VaR, CVaR, Sharpe, bias cost
│   └── visualizations.py    ← Plotly charts
├── requirements.txt
└── README.md
```

## Key Academic References

- **Kahneman, D. & Tversky, A. (1979)** — Prospect Theory: An Analysis of Decision Under Risk. *Econometrica*.
- **Barber, B. & Odean, T. (2001)** — Boys Will Be Boys: Gender, Overconfidence, and Common Stock Investment. *Quarterly Journal of Economics*.
- **Malmendier, U. & Tate, G. (2005)** — CEO Overconfidence and Corporate Investment. *Journal of Finance*.
- **Shefrin, H. & Statman, M. (1985)** — The Disposition to Sell Winners Too Early and Ride Losers Too Long. *Journal of Finance*.

## Methodological Notes

**Why t-distribution instead of normal?**
Equity returns exhibit excess kurtosis (~6) and negative skew. The Student's
t-distribution with low degrees of freedom (ν ≈ 5) captures crash frequency
far better than a Gaussian model.

**Correlation structure**
Cholesky decomposition of a historical-calibrated correlation matrix:
- Equity-equity: ρ = 0.75 (positive, diversification limited)
- Equity-bond: ρ = -0.15 (slight flight-to-safety effect)

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
