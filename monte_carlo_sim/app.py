"""
Behavioral Finance Monte Carlo Dashboard
=========================================
Portfolio simulation using real SPY/EFA/AGG market data (Stooq, 2005-2024).
Parameters are fitted from actual returns — nothing is hardcoded.

Run:
    streamlit run app.py
"""

import sys
sys.path.insert(0, ".")

import streamlit as st
import numpy as np
import pandas as pd

from src.data import load_market_data
from src.simulation import MonteCarloEngine, SimulationConfig
from src.investors import RationalInvestor, LossAverseInvestor, OverconfidentInvestor
from src.metrics import compute_metrics, compute_bias_costs, summary_table
from src.visualizations import (
    plot_wealth_distributions,
    plot_wealth_paths,
    plot_drawdowns,
    plot_bias_cost,
    plot_risk_return,
    plot_metric_heatmap,
)
from src.backtest import run_historical_backtest, plot_backtest_vs_simulation, backtest_summary_table
from src.calibration import (
    plot_return_distributions,
    plot_qq,
    plot_rolling_correlation,
    plot_tail_comparison,
    compute_calibration_stats,
    plot_regime_analysis,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Behavioral Monte Carlo",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+SC:wght@300;500;600&family=Fira+Code:wght@400;500&family=Instrument+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

/* ── Design tokens ──────────────────────────────────────────────────────── */
:root {
    --bg-base:    #07070A;
    --bg-surface: #0D0D12;
    --bg-raised:  #131319;
    --bd-dim:     rgba(255,255,255,0.055);
    --bd-mid:     rgba(255,255,255,0.10);
    --bd-gold:    rgba(200,150,12,0.38);
    --gold:       #C8960C;
    --gold-hi:    #D9A820;
    --gold-glow:  rgba(200,150,12,0.10);
    --text-hi:    #DDD7CB;
    --text-mid:   #6E665E;
    --text-lo:    #343028;
    --gain:       #4C8A60;
    --loss:       #AA4030;
    --r-color:    #4882B4;
    --l-color:    #BC6C3A;
    --o-color:    #7C5AB8;
}

/* ── Base reset ─────────────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Instrument Sans', sans-serif !important;
    background-color: var(--bg-base) !important;
    color: var(--text-hi) !important;
}
.stApp { background-color: var(--bg-base) !important; }
.main .block-container { padding-top: 2.2rem !important; max-width: 1380px !important; }

/* Subtle grid texture on main content */
.main {
    background-image:
        linear-gradient(var(--bd-dim) 1px, transparent 1px),
        linear-gradient(90deg, var(--bd-dim) 1px, transparent 1px);
    background-size: 28px 28px;
}

/* ── Sidebar ────────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background-color: var(--bg-raised) !important;
    border-right: 1px solid var(--bd-dim) !important;
    background-image: none !important;
}
section[data-testid="stSidebar"] .stMarkdown p {
    font-family: 'Fira Code', monospace !important;
    font-size: 0.60rem !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    color: var(--gold) !important;
    font-weight: 500 !important;
    margin: 0.2rem 0 !important;
}
section[data-testid="stSidebar"] hr { border-color: var(--bd-dim) !important; }
section[data-testid="stSidebar"] label {
    font-family: 'Fira Code', monospace !important;
    font-size: 0.66rem !important;
    color: var(--text-mid) !important;
    letter-spacing: 0.04em !important;
}

/* ── Buttons ────────────────────────────────────────────────────────────── */
.stButton > button[kind="primary"] {
    background: var(--gold) !important;
    color: #07070A !important;
    border: none !important;
    border-radius: 0 !important;
    font-family: 'Fira Code', monospace !important;
    font-size: 0.68rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.16em !important;
    text-transform: uppercase !important;
    padding: 0.65rem 1rem !important;
    transition: background 0.12s !important;
}
.stButton > button[kind="primary"]:hover { background: var(--gold-hi) !important; }

/* ── Section labels ─────────────────────────────────────────────────────── */
.sect {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 2.2rem 0 1rem;
}
.sect-txt {
    font-family: 'Fira Code', monospace;
    font-size: 0.56rem;
    font-weight: 500;
    letter-spacing: 0.24em;
    text-transform: uppercase;
    color: var(--gold);
    white-space: nowrap;
}
.sect::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, var(--bd-gold) 0%, transparent 70%);
}

/* ── Display header ─────────────────────────────────────────────────────── */
.disp-title {
    font-family: 'Cormorant SC', serif;
    font-size: 3.4rem;
    font-weight: 600;
    line-height: 0.92;
    letter-spacing: -0.01em;
    color: var(--text-hi);
    margin-bottom: 0.35rem;
}
.disp-sub {
    font-family: 'Instrument Sans', sans-serif;
    font-size: 0.85rem;
    font-weight: 300;
    font-style: italic;
    color: var(--text-mid);
    letter-spacing: 0.015em;
    margin-bottom: 0.85rem;
}
.data-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: var(--gold-glow);
    color: var(--gold);
    border: 1px solid var(--bd-gold);
    padding: 3px 11px;
    font-family: 'Fira Code', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.07em;
    margin-bottom: 1.8rem;
}
.data-pill::before { content: '◆'; font-size: 0.42rem; opacity: 0.75; }

/* ── Investor tags ───────────────────────────────────────────────────────── */
.tag {
    display: inline-block;
    font-family: 'Fira Code', monospace;
    font-size: 0.63rem;
    letter-spacing: 0.05em;
    padding: 2px 9px;
    border: 1px solid;
    margin-right: 4px;
    vertical-align: middle;
}
.tag-r { color: var(--r-color); border-color: rgba(72,130,180,0.38); background: rgba(72,130,180,0.07); }
.tag-l { color: var(--l-color); border-color: rgba(188,108,58,0.38);  background: rgba(188,108,58,0.07); }
.tag-o { color: var(--o-color); border-color: rgba(124,90,184,0.38);  background: rgba(124,90,184,0.07); }

/* ── Fitted parameter cards ─────────────────────────────────────────────── */
.fit-card {
    background: var(--bg-raised);
    border: 1px solid var(--bd-dim);
    border-top: 2px solid var(--gold);
    padding: 0.85rem 0.95rem 0.95rem;
    font-family: 'Fira Code', monospace;
    height: 100%;
}
.fit-name {
    font-size: 0.70rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-hi);
    margin-bottom: 0.55rem;
    padding-bottom: 0.45rem;
    border-bottom: 1px solid var(--bd-dim);
}
.fit-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.66rem;
    color: var(--text-mid);
    padding: 2.5px 0;
}
.fv { color: var(--text-hi); }

/* ── KPI cards ───────────────────────────────────────────────────────────── */
.kpi {
    background: var(--bg-raised);
    border: 1px solid var(--bd-dim);
    border-left: 3px solid;
    padding: 0.9rem 1rem 0.95rem;
}
.kpi-r { border-left-color: var(--r-color); }
.kpi-l { border-left-color: var(--l-color); }
.kpi-o { border-left-color: var(--o-color); }
.kpi-label {
    font-family: 'Fira Code', monospace;
    font-size: 0.56rem;
    font-weight: 500;
    letter-spacing: 0.20em;
    text-transform: uppercase;
    color: var(--text-mid);
    margin-bottom: 0.22rem;
}
.kpi-value {
    font-family: 'Cormorant SC', serif;
    font-size: 2.15rem;
    font-weight: 600;
    line-height: 1.0;
    letter-spacing: -0.01em;
    color: var(--text-hi);
}
.kpi-sub {
    font-family: 'Fira Code', monospace;
    font-size: 0.61rem;
    color: var(--text-mid);
    margin-top: 0.3rem;
    letter-spacing: 0.02em;
}
.kpi-pos { font-family: 'Fira Code', monospace; font-size: 0.60rem; color: var(--gain);  margin-top: 0.18rem; }
.kpi-neg { font-family: 'Fira Code', monospace; font-size: 0.60rem; color: var(--loss);  margin-top: 0.18rem; }

/* ── Tabs ────────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background-color: transparent !important;
    border-bottom: 1px solid var(--bd-dim) !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Fira Code', monospace !important;
    font-size: 0.61rem !important;
    letter-spacing: 0.11em !important;
    text-transform: uppercase !important;
    color: var(--text-mid) !important;
    padding: 9px 16px !important;
    background: transparent !important;
    border-bottom: 2px solid transparent !important;
    margin-bottom: -1px !important;
}
.stTabs [aria-selected="true"] {
    color: var(--gold) !important;
    border-bottom: 2px solid var(--gold) !important;
    background: transparent !important;
}

/* ── Expander ────────────────────────────────────────────────────────────── */
.streamlit-expanderHeader {
    font-family: 'Fira Code', monospace !important;
    font-size: 0.66rem !important;
    color: var(--text-mid) !important;
    background: var(--bg-raised) !important;
    border: 1px solid var(--bd-dim) !important;
    border-radius: 0 !important;
}
.streamlit-expanderContent {
    background: var(--bg-raised) !important;
    border: 1px solid var(--bd-dim) !important;
    border-top: none !important;
}

/* ── Captions ────────────────────────────────────────────────────────────── */
.stCaptionContainer p, .stCaption {
    font-family: 'Fira Code', monospace !important;
    font-size: 0.62rem !important;
    color: var(--text-mid) !important;
    letter-spacing: 0.02em !important;
}

/* ── Dataframe ───────────────────────────────────────────────────────────── */
.stDataFrame { border: 1px solid var(--bd-dim) !important; }

/* ── Scrollbar ───────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.08); }

/* ── Footer ──────────────────────────────────────────────────────────────── */
.footer {
    margin-top: 3rem;
    padding-top: 1.2rem;
    border-top: 1px solid var(--bd-dim);
    font-family: 'Fira Code', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.14em;
    color: var(--text-lo);
    text-align: right;
    text-transform: uppercase;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# UI component helpers  (composition pattern — single-responsibility templates)
# ---------------------------------------------------------------------------

def _section(text: str) -> str:
    """Renders a gold section heading with extending rule."""
    return f'<div class="sect"><span class="sect-txt">{text}</span></div>'


def _investor_tag(name: str) -> str:
    """Explicit per-variant tag — no boolean prop branching."""
    variant_cls = {
        "Rational":      "tag-r",
        "Loss-Averse":   "tag-l",
        "Overconfident": "tag-o",
    }.get(name, "tag-r")
    return f'<span class="tag {variant_cls}">{name}</span>'


def _kpi_color(name: str) -> str:
    return {"Rational": "r", "Loss-Averse": "l", "Overconfident": "o"}.get(name, "r")


def _kpi_card(label: str, value: str, sub: str, delta_html: str, investor_name: str) -> str:
    """KPI card — each concern isolated, no conditional prop soup."""
    c = _kpi_color(investor_name)
    return (
        f'<div class="kpi kpi-{c}">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-sub">{sub}</div>'
        f'{delta_html}'
        f'</div>'
    )


def _fitted_card(asset, top_color: str) -> str:
    """Fitted parameter card — minimal, data-forward."""
    return (
        f'<div class="fit-card" style="border-top-color:{top_color}">'
        f'<div class="fit-name">{asset.name}</div>'
        f'<div class="fit-row"><span>Ann. return</span><span class="fv">{asset.ann_return:.2%}</span></div>'
        f'<div class="fit-row"><span>Ann. vol</span><span class="fv">{asset.ann_vol:.2%}</span></div>'
        f'<div class="fit-row"><span>t-dist ν</span><span class="fv">{asset.nu:.2f}</span></div>'
        f'<div class="fit-row"><span>Skew</span><span class="fv">{asset.skew:.3f}</span></div>'
        f'<div class="fit-row"><span>Weight</span><span class="fv">{asset.weight:.0%}</span></div>'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Load real market data (cached — only runs once per session)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading market data…")
def get_market_data():
    return load_market_data()

market_data = get_market_data()

@st.cache_data(show_spinner=False)
def get_backtest():
    return run_historical_backtest(market_data)

historical_backtest = get_backtest()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("**Portfolio**")
    n_simulations  = st.slider("Simulations",         1_000, 15_000, 5_000, step=1_000)
    n_years        = st.slider("Horizon (years)",     5,     30,     10,    step=1)
    initial_wealth = st.number_input("Initial wealth ($)", 10_000, 5_000_000, 100_000, step=10_000)

    st.markdown("---")
    st.markdown("**Loss-Averse investor**")
    panic_threshold = st.slider("Panic trigger (drawdown)", -0.25, -0.05, -0.10, step=0.01, format="%.0f%%")
    recovery_days   = st.slider("Recovery days before re-entry", 10, 120, 40, step=5)

    st.markdown("---")
    st.markdown("**Overconfident investor**")
    conc_mult  = st.slider("Winner concentration multiplier", 1.5, 4.0, 2.0, step=0.25)
    trade_freq = st.slider("Re-concentration frequency (days)", 10, 63, 21, step=5)

    st.markdown("---")
    risk_free = st.slider("Risk-free rate", 0.00, 0.07, 0.04, step=0.005, format="%.1f%%")
    seed      = st.number_input("Random seed", 0, 9999, 42)

    st.markdown("---")
    st.markdown("**Model settings**")
    use_regime = st.toggle(
        "Regime-switching correlations",
        value=True,
        help="When ON: correlations shift between Calm / Normal / Crisis regimes "
             "(calibrated from historical vol). When OFF: static full-sample correlation.",
    )
    st.markdown("---")
    st.markdown(
        f"<span style='font-family:Fira Code,monospace;font-size:0.60rem;"
        f"color:#6E665E;letter-spacing:0.05em'>"
        f"{market_data.start_date} → {market_data.end_date} &nbsp;·&nbsp; "
        f"{market_data.n_obs:,} days</span>",
        unsafe_allow_html=True,
    )
    run_btn = st.button("Run Simulation", use_container_width=True, type="primary")


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="disp-title">Behavioral<br>Monte Carlo</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="disp-sub">The financial cost of acting on instinct — '
    'quantified across 10,000 simulated portfolios.</div>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<div class="data-pill">Real data &nbsp;·&nbsp; SPY / EFA / AGG &nbsp;·&nbsp; '
    f'{market_data.start_date} – {market_data.end_date} &nbsp;·&nbsp; '
    f'{market_data.n_obs:,} trading days</div>',
    unsafe_allow_html=True,
)

# Investor archetype row
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(
        _investor_tag("Rational") +
        '&nbsp;<span style="font-size:0.78rem;color:#6E665E">Buy-and-hold, quarterly rebalance</span>',
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        _investor_tag("Loss-Averse") +
        '&nbsp;<span style="font-size:0.78rem;color:#6E665E">Panic-sells on drawdowns</span>',
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        _investor_tag("Overconfident") +
        '&nbsp;<span style="font-size:0.78rem;color:#6E665E">Over-concentrates on winners</span>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Fitted parameters panel
# ---------------------------------------------------------------------------
st.markdown(_section("Fitted market parameters — from real data"), unsafe_allow_html=True)

asset_colors = ["#4882B4", "#7C5AB8", "#4C8A60"]
param_cols   = st.columns(len(market_data.fitted_assets))
for col, asset, color in zip(param_cols, market_data.fitted_assets, asset_colors):
    with col:
        st.markdown(_fitted_card(asset, color), unsafe_allow_html=True)

with st.expander("Empirical correlation matrix"):
    corr_df = pd.DataFrame(
        np.round(market_data.correlation, 3),
        index=[a.name.split("(")[0].strip() for a in market_data.fitted_assets],
        columns=[a.name.split("(")[0].strip() for a in market_data.fitted_assets],
    )
    st.dataframe(corr_df, use_container_width=True)
    st.caption("Pearson correlations estimated from overlapping daily return sample 2005–2024.")


# ---------------------------------------------------------------------------
# Run simulation
# ---------------------------------------------------------------------------
def plot_panic_curve_inline(panic_midpoint: float) -> None:
    """Render a small panic probability curve using Plotly."""
    import plotly.graph_objects as go
    from scipy.special import expit as _expit
    k   = 25.0
    dd  = [x / 1000 for x in range(-350, 5, 5)]
    prb = [float(_expit(k * (-d - abs(panic_midpoint)))) for d in dd]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[d * 100 for d in dd], y=[p * 100 for p in prb],
        mode="lines",
        line=dict(color="#BC6C3A", width=2),
        fill="tozeroy",
        fillcolor="rgba(188,108,58,0.08)",
    ))
    fig.add_vline(
        x=panic_midpoint * 100,
        line=dict(color="#BC6C3A", width=1.2, dash="dash"),
        annotation_text=f"P=50% at {panic_midpoint:.0%}",
        annotation_font=dict(color="#BC6C3A", size=9),
    )
    fig.add_hline(y=50, line=dict(color="rgba(255,255,255,0.12)", width=0.8, dash="dot"))
    fig.update_layout(
        paper_bgcolor="#07070A",
        plot_bgcolor="#0D0D12",
        font=dict(family="Fira Code, monospace", color="#DDD7CB", size=10),
        margin=dict(l=40, r=10, t=30, b=40),
        xaxis=dict(title="Drawdown (%)", gridcolor="rgba(255,255,255,0.04)"),
        yaxis=dict(title="P(panic) %",   gridcolor="rgba(255,255,255,0.04)", range=[-2, 102]),
        height=220,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


@st.cache_data(show_spinner="Running Monte Carlo simulation…")
def run_simulation(n_sim, n_yrs, W0, panic_thr, rec_days, conc, tfreq, rf, seed_val,
                   data_start, data_end, regime_switching=True):
    config = SimulationConfig(
        n_simulations=n_sim,
        n_years=n_yrs,
        initial_wealth=W0,
        use_regime_switching=regime_switching,
        random_seed=seed_val,
    )
    engine = MonteCarloEngine(market_data=market_data, config=config)
    paths  = engine.generate_paths()

    rational    = RationalInvestor().simulate(paths)
    loss_averse = LossAverseInvestor(panic_threshold=panic_thr, recovery_days=rec_days).simulate(paths)
    overconf    = OverconfidentInvestor(concentration_mult=conc, trade_freq=tfreq).simulate(paths)

    results = [rational, loss_averse, overconf]
    m_list  = compute_bias_costs([compute_metrics(r, config, risk_free_rate=rf) for r in results])

    return results, m_list, config, paths["daily_returns"], paths["regimes"]


if "results" not in st.session_state or run_btn:
    with st.spinner("Running simulation…"):
        results, m_list, config, daily_returns, sim_regimes = run_simulation(
            n_simulations, n_years, initial_wealth,
            panic_threshold, recovery_days, conc_mult, trade_freq,
            risk_free, seed,
            market_data.start_date, market_data.end_date,
            regime_switching=use_regime,
        )
    st.session_state.update({
        "results": results, "m_list": m_list,
        "config": config, "daily_returns": daily_returns,
        "sim_regimes": sim_regimes,
    })
else:
    results       = st.session_state["results"]
    m_list        = st.session_state["m_list"]
    config        = st.session_state["config"]
    daily_returns = st.session_state["daily_returns"]
    sim_regimes   = st.session_state["sim_regimes"]


# ---------------------------------------------------------------------------
# KPI cards
# ---------------------------------------------------------------------------
st.markdown(_section("Key metrics"), unsafe_allow_html=True)

kpi_cols = st.columns(len(m_list) * 2)
for i, m in enumerate(m_list):
    with kpi_cols[i * 2]:
        st.markdown(
            _kpi_card(
                label=f"{m.name} — Median Wealth",
                value=f"${m.median_terminal_wealth:,.0f}",
                sub=f"Ann. return {m.annualised_return:.2%}  ·  Sharpe {m.sharpe_ratio:.2f}",
                delta_html=(
                    f'<div class="kpi-pos">Rational baseline</div>'
                    if m.name == "Rational"
                    else f'<div class="kpi-neg">▼ ${m.bias_cost_vs_rational:,.0f} bias cost ({m.bias_cost_pct:.1%})</div>'
                ),
                investor_name=m.name,
            ),
            unsafe_allow_html=True,
        )
    with kpi_cols[i * 2 + 1]:
        st.markdown(
            _kpi_card(
                label=f"{m.name} — Max Drawdown",
                value=f"{m.max_drawdown_median:.1%}",
                sub=f"P(loss) {m.probability_of_loss:.1%}  ·  P(beat inflation) {m.probability_beat_inflation:.1%}",
                delta_html="",
                investor_name=m.name,
            ),
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Analysis tabs
# ---------------------------------------------------------------------------
st.markdown(_section("Analysis"), unsafe_allow_html=True)

chart_cfg = {"displayModeBar": False}

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "Wealth Paths",
    "Distributions",
    "Drawdowns",
    "Bias Cost",
    "Risk / Return",
    "Heatmap",
    "Calibration",
    "Backtest",
])

with tab1:
    st.plotly_chart(plot_wealth_paths(results, config.n_years), use_container_width=True, config=chart_cfg)
    st.caption("Median path with 25–75th and 10–90th percentile bands.")

with tab2:
    st.plotly_chart(plot_wealth_distributions(results, config.initial_wealth), use_container_width=True, config=chart_cfg)
    st.caption("Kernel-smoothed density of terminal wealth across all simulations.")

with tab3:
    st.plotly_chart(plot_drawdowns(results, config.n_years), use_container_width=True, config=chart_cfg)
    st.caption("Median drawdown from peak. Shaded = 10th percentile (worst-case scenarios).")

with tab4:
    st.plotly_chart(plot_bias_cost(m_list, config.initial_wealth), use_container_width=True, config=chart_cfg)
    non_rat = [m for m in m_list if m.name != "Rational"]
    rat     = next(m for m in m_list if m.name == "Rational")
    bias_rows = [{
        "Investor":        m.name,
        "Rational median": f"${rat.median_terminal_wealth:,.0f}",
        "Their median":    f"${m.median_terminal_wealth:,.0f}",
        "$ gap":           f"${m.bias_cost_vs_rational:,.0f}",
        "% gap":           f"{m.bias_cost_pct:.2%}",
        "Avg trades":      f"{m.avg_trades:.0f}",
    } for m in non_rat]
    st.dataframe(pd.DataFrame(bias_rows), hide_index=True, use_container_width=True)

with tab5:
    st.plotly_chart(plot_risk_return(m_list), use_container_width=True, config=chart_cfg)
    st.caption("Annualised return vs volatility. Upper-left = better.")

with tab6:
    st.plotly_chart(plot_metric_heatmap(m_list), use_container_width=True, config=chart_cfg)
    st.caption("Normalised metric heatmap. Green = better. Risk metrics are inverted.")

with tab7:
    st.markdown(
        '<div style="font-family:\'Cormorant SC\',serif;font-size:1.4rem;font-weight:500;'
        'color:#DDD7CB;margin-bottom:0.3rem">Model Calibration</div>'
        '<div style="font-family:\'Instrument Sans\',sans-serif;font-size:0.80rem;font-style:italic;'
        'color:#6E665E;margin-bottom:1rem">Simulated vs real historical data — validates the return model.</div>',
        unsafe_allow_html=True,
    )

    cal_tab1, cal_tab2, cal_tab3, cal_tab4, cal_tab5 = st.tabs([
        "Return Distributions", "Q-Q Plot", "Rolling Correlations", "Tail Frequencies", "Regimes"
    ])

    with cal_tab1:
        st.plotly_chart(
            plot_return_distributions(market_data, daily_returns),
            use_container_width=True, config=chart_cfg,
        )
        st.caption("Solid = historical, dashed = simulated, dotted = fitted t-distribution.")

    with cal_tab2:
        st.plotly_chart(
            plot_qq(market_data, daily_returns),
            use_container_width=True, config=chart_cfg,
        )
        st.caption("Points on the diagonal = perfect calibration. Deviations in tails show where the model diverges.")

    with cal_tab3:
        st.plotly_chart(
            plot_rolling_correlation(market_data),
            use_container_width=True, config=chart_cfg,
        )
        st.caption("Empirical 1-year rolling correlations. Crisis periods shaded. Note equity-bond breakdown in 2022.")

    with cal_tab4:
        st.plotly_chart(
            plot_tail_comparison(market_data, daily_returns),
            use_container_width=True, config=chart_cfg,
        )
        st.caption("Solid = historical, dashed = simulated. Close match validates fat-tail model.")

    with cal_tab5:
        st.plotly_chart(
            plot_regime_analysis(market_data, sim_regimes, config.n_years),
            use_container_width=True, config=chart_cfg,
        )
        st.caption(
            "Top: historical SPY realised vol with calm/crisis regime bands. "
            "Bottom: fraction of simulated paths in each regime over the investment horizon."
        )

    st.markdown(_section("Calibration statistics"), unsafe_allow_html=True)
    cal_rows = compute_calibration_stats(market_data, daily_returns, n_sample=500)
    st.dataframe(pd.DataFrame(cal_rows), hide_index=True, use_container_width=True)
    st.caption("KS p-value > 0.05 indicates acceptable fit.")

with tab8:
    st.markdown(
        '<div style="font-family:\'Cormorant SC\',serif;font-size:1.4rem;font-weight:500;'
        'color:#DDD7CB;margin-bottom:0.3rem">Historical Backtest</div>'
        f'<div style="font-family:\'Instrument Sans\',sans-serif;font-size:0.80rem;font-style:italic;'
        f'color:#6E665E;margin-bottom:1rem">Rational strategy on actual '
        f'{historical_backtest["start_date"]} → {historical_backtest["end_date"]} returns.</div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        plot_backtest_vs_simulation(historical_backtest, results[0], config, market_data),
        use_container_width=True, config=chart_cfg,
    )
    bt_rows = backtest_summary_table(historical_backtest, results[0], config)
    st.dataframe(pd.DataFrame(bt_rows), hide_index=True, use_container_width=True)
    st.caption(
        "If the simulation is well-calibrated, the actual historical outcome should "
        "fall between the 10th and 90th percentile of the simulated distribution."
    )

    st.markdown(_section("Loss-averse panic curve"), unsafe_allow_html=True)
    st.caption(
        "P(panic | drawdown) follows a logistic function. "
        "Adjust the panic trigger slider in the sidebar to shift the curve."
    )
    plot_panic_curve_inline(panic_threshold)


# ---------------------------------------------------------------------------
# Full stats table
# ---------------------------------------------------------------------------
st.markdown(_section("Full statistics"), unsafe_allow_html=True)
st.dataframe(pd.DataFrame(summary_table(m_list)), hide_index=True, use_container_width=True)


# ---------------------------------------------------------------------------
# Methodology
# ---------------------------------------------------------------------------
with st.expander("Methodology & data sources"):
    st.markdown(f"""
**Data**
- SPY (US equities), EFA (international equities), AGG (US bonds)
- Source: Stooq.com historical daily prices
- Period: {market_data.start_date} → {market_data.end_date} ({market_data.n_obs:,} trading days)

**Return model: meta-elliptical copula**
1. Fit Student's t-distribution to each asset's daily log-returns via MLE
2. Estimate empirical Pearson correlation matrix from overlapping return sample
3. In simulation: draw independent t(ν) variates per asset, apply Cholesky decomposition,
   scale to fitted daily μ and σ

**Regime-switching correlations**

| Regime | Vol signal | SPY/EFA | SPY/AGG |
|--------|-----------|---------|---------|
| Calm | < 11.1% ann. vol | +0.79 | −0.28 |
| Normal | 11.1% – 18.6% | +0.88 | −0.01 |
| Crisis | > 18.6% | +0.95 | +0.16 |

**Investor archetypes**

| Archetype | Mechanism | Reference |
|---|---|---|
| **Rational** | Buy-and-hold, quarterly rebalancing | Markowitz (1952) |
| **Loss-Averse** | De-risks to bonds after threshold drawdown; gradual re-entry | Kahneman & Tversky (1979) |
| **Overconfident** | Concentrates in recent winner; over-trades with transaction costs | Barber & Odean (2001) |

**Key references**
- Kahneman, D. & Tversky, A. (1979). Prospect Theory. *Econometrica*.
- Barber, B. & Odean, T. (2001). Boys Will Be Boys. *Quarterly Journal of Economics*.
- McNeil, A., Frey, R. & Embrechts, P. (2015). *Quantitative Risk Management*. Princeton UP.
    """)

st.markdown(
    '<div class="footer">Real data &nbsp;·&nbsp; Monte Carlo &nbsp;·&nbsp; Behavioral Finance</div>',
    unsafe_allow_html=True,
)
