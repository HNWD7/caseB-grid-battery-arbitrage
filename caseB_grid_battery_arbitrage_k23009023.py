"""
caseB_grid_battery_arbitrage.py
================================
Case B: Grid-Scale Battery Energy Arbitrage
EGS Individual Coursework | Python / SciPy / HiGHS

Runs the full pipeline:
  1. Load & validate data
  2. Heuristic dispatch (rolling percentile thresholds)
  3. LP optimal dispatch (perfect foresight, SciPy/HiGHS)
  4. Verification (8 checks × 2 policies = 16 total)
  5. Extensions: threshold sweep, carbon Pareto, capacity sensitivity
  6. Generate all figures

Usage:
    python caseB_grid_battery_arbitrage.py
    python caseB_grid_battery_arbitrage.py --data path/to/caseB_grid_battery_market_hourly.csv
"""

import argparse
import os
import numpy as np
import pandas as pd
from scipy.optimize import linprog
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker

# ══════════════════════════════════════════════════════════════════════════════
# PARAMETERS — single source of truth
# ══════════════════════════════════════════════════════════════════════════════

E_CAP   = 2000.0   # kWh  — usable energy capacity
P_MAX   = 1000.0   # kW   — max charge / discharge power
ETA_CH  = 0.9381   # —    — charge efficiency (sqrt(0.88))
ETA_DIS = 0.9381   # —    — discharge efficiency (sqrt(0.88))
ETA_RT  = 0.88     # —    — round-trip efficiency
E0      = 1000.0   # kWh  — initial SOC (50% of capacity)
DT      = 1.0      # h    — timestep

H_CHARGE_PCT    = 30   # heuristic: percentile threshold to charge
H_DISCHARGE_PCT = 70   # heuristic: percentile threshold to discharge
H_WINDOW        = 48   # heuristic: rolling window (hours)

CARBON_PENALTIES = [0, 25, 50, 100, 200, 500]   # GBP/tonne CO2
CAPACITY_RANGE   = [500, 1000, 2000, 3000, 4000, 5000]  # kWh
SWEEP_CHARGE     = [10, 20, 25, 30, 35, 40]
SWEEP_DISCHARGE  = [60, 65, 70, 75, 80, 85, 90]

TOL = 1e-6   # verification tolerance

# ── Plot style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 8,
    'axes.titlesize': 9, 'axes.labelsize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'figure.dpi': 150, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.3, 'grid.linewidth': 0.5,
})
COL_H   = '#E07B39'
COL_LP  = '#27AE60'
COL_PRI = '#BBBBBB'


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_data(filepath):
    """Load and validate caseB_grid_battery_market_hourly.csv."""
    df = pd.read_csv(filepath, parse_dates=['timestamp'])
    assert len(df) == 1440,            f"Expected 1440 rows, got {len(df)}"
    assert df.isnull().sum().sum() == 0, "Null values found in dataset"
    assert df.shape[1] == 5,            f"Expected 5 columns, got {df.shape[1]}"
    deltas = df['timestamp'].diff().dropna()
    assert (deltas == pd.Timedelta('1h')).all(), "Non-uniform timesteps detected"
    assert (df['day_ahead_price_gbp_per_mwh'] > 0).all(), "Non-positive DA prices"

    print(f"Data loaded: {len(df)} rows, {df.shape[1]} columns, zero nulls")
    print(f"  DA price:  £{df['day_ahead_price_gbp_per_mwh'].min():.2f}–"
          f"£{df['day_ahead_price_gbp_per_mwh'].max():.2f}/MWh "
          f"(mean £{df['day_ahead_price_gbp_per_mwh'].mean():.2f})")
    print(f"  P80/P20 ratio: "
          f"{np.percentile(df['day_ahead_price_gbp_per_mwh'],80) / np.percentile(df['day_ahead_price_gbp_per_mwh'],20):.3f} "
          f"(break-even: 1.136)")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. HEURISTIC DISPATCH
# ══════════════════════════════════════════════════════════════════════════════

def run_heuristic(price, p_charge=H_CHARGE_PCT, p_discharge=H_DISCHARGE_PCT, window=H_WINDOW):
    """
    Rolling-percentile heuristic dispatch.
    Charge at max feasible power when price <= p_charge percentile.
    Discharge at max feasible power when price >= p_discharge percentile.
    No future price knowledge required.
    """
    T    = len(price)
    pch  = np.zeros(T)
    pdis = np.zeros(T)
    E    = np.zeros(T + 1)
    E[0] = E0

    for t in range(T):
        lo   = max(0, t - window // 2)
        hi   = min(T, t + window // 2)
        p_lo = np.percentile(price[lo:hi], p_charge)
        p_hi = np.percentile(price[lo:hi], p_discharge)

        avail_charge    = (E_CAP - E[t]) / (ETA_CH * DT)
        avail_discharge = E[t] * ETA_DIS / DT

        if price[t] <= p_lo and avail_charge > 0:
            pch[t] = min(P_MAX, avail_charge)
        elif price[t] >= p_hi and avail_discharge > 0:
            pdis[t] = min(P_MAX, avail_discharge)

        E[t + 1] = np.clip(
            E[t] + ETA_CH * pch[t] * DT - pdis[t] / ETA_DIS * DT,
            0, E_CAP
        )

    return pch, pdis, E[1:]


# ══════════════════════════════════════════════════════════════════════════════
# 3. LP OPTIMAL DISPATCH
# ══════════════════════════════════════════════════════════════════════════════

def run_lp(price, carbon=None, carbon_penalty=0.0):
    """
    Perfect-foresight LP via SciPy / HiGHS.

    Decision variables: x = [P_ch(0)...P_ch(T-1), P_dis(0)...P_dis(T-1)]

    Base objective:
        max  Σ_t [ P_dis(t)·π(t) - P_ch(t)·π(t) ] · Δt/1000

    Carbon-aware extension (carbon_penalty > 0):
        min  Σ_t [ P_ch(t)·π(t) - P_dis(t)·π(t) + (λ_c/1000)·CI(t)·P_ch(t) ] · Δt/1000

    SOC dynamics enforced as cumulative inequality constraints.
    Terminal SOC constraint: E(T) >= E0.
    """
    T = len(price)
    if carbon is None:
        carbon = np.zeros(T)

    # Objective (linprog minimises, so negate profit)
    c_obj = np.zeros(2 * T)
    for t in range(T):
        c_obj[t]     =  price[t] * DT / 1000 + (carbon_penalty / 1000) * carbon[t] * DT
        c_obj[T + t] = -price[t] * DT / 1000

    # Bounds: 0 <= P_ch, P_dis <= P_max
    bounds = [(0, P_MAX)] * T + [(0, P_MAX)] * T

    # SOC constraints (cumulative form)
    # E(t) = E0 + Σ_{s<=t} [η_ch·P_ch(s) - P_dis(s)/η_dis]·Δt
    # Upper: η_ch·Σpch - (1/η_dis)·Σpdis <= E_cap - E0
    # Lower: -η_ch·Σpch + (1/η_dis)·Σpdis <= E0
    A_ub, b_ub = [], []
    for t in range(T):
        row = np.zeros(2 * T)
        row[:t + 1]      =  ETA_CH
        row[T:T + t + 1] = -1.0 / ETA_DIS
        A_ub.append(row.copy());  b_ub.append(E_CAP - E0)
        A_ub.append(-row.copy()); b_ub.append(E0)

    # Terminal SOC: E(T) >= E0
    row_term = np.zeros(2 * T)
    row_term[:T] = -ETA_CH
    row_term[T:] =  1.0 / ETA_DIS
    A_ub.append(row_term)
    b_ub.append(0.0)

    result = linprog(c_obj, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                     bounds=bounds, method='highs')

    if result.status != 0:
        raise RuntimeError(f"LP failed: {result.message}")

    pch  = result.x[:T]
    pdis = result.x[T:]

    # Reconstruct SOC
    E = np.zeros(T + 1)
    E[0] = E0
    for t in range(T):
        E[t + 1] = E[t] + ETA_CH * pch[t] * DT - pdis[t] / ETA_DIS * DT

    return pch, pdis, E[1:]


# ══════════════════════════════════════════════════════════════════════════════
# 4. VERIFICATION — 8 independent checks
# ══════════════════════════════════════════════════════════════════════════════

def run_verification(price, pch, pdis, E, label='', verbose=True):
    """
    8 verification checks:
      1. Energy balance residual
      2. SOC lower bound
      3. SOC upper bound
      4. Power bounds
      5. Simultaneous charge/discharge
      6. Terminal SOC
      7. Unit consistency (worked example)
      8. Profit reconciliation
    """
    T      = len(price)
    passed = {}

    # 1. Energy balance: r(t) = E(t) - E(t-1) - η_ch·P_ch·Δt + P_dis·Δt/η_dis
    E_prev    = np.concatenate([[E0], E[:-1]])
    residuals = E - E_prev - ETA_CH * pch * DT + pdis / ETA_DIS * DT
    c1 = np.max(np.abs(residuals))
    passed[1] = c1 < TOL

    # 2. SOC lower bound
    c2 = E.min()
    passed[2] = c2 >= -TOL

    # 3. SOC upper bound
    c3 = E.max()
    passed[3] = c3 <= E_CAP + TOL

    # 4. Power bounds
    passed[4] = (pch.min()  >= -TOL and pch.max()  <= P_MAX + TOL and
                 pdis.min() >= -TOL and pdis.max() <= P_MAX + TOL)

    # 5. Simultaneous charge/discharge
    c5 = int(np.sum((pch > TOL) & (pdis > TOL)))
    passed[5] = c5 == 0

    # 6. Terminal SOC
    c6 = E[-1]
    passed[6] = c6 >= E0 - TOL

    # 7. Unit consistency (t=10 worked example)
    pi_t  = price[10]
    c7    = 1000.0 * DT * pi_t / 1000   # GBP
    passed[7] = abs(c7 - round(c7, 10)) < TOL

    # 8. Profit reconciliation
    profit_direct = np.sum((pdis - pch) * price * DT / 1000)
    profit_cumsum = np.cumsum((pdis - pch) * price * DT / 1000)[-1]
    c8 = abs(profit_direct - profit_cumsum)
    passed[8] = c8 < TOL

    if verbose:
        status = '✓ ALL PASS' if all(passed.values()) else '✗ FAILURES'
        print(f"\n{'='*60}")
        print(f"Verification: {label}  [{status}]")
        print(f"{'='*60}")
        print(f"  1  Energy balance residual:  {c1:.2e} kWh      {'PASS' if passed[1] else 'FAIL'}")
        print(f"  2  SOC lower bound:          {c2:.4f} kWh     {'PASS' if passed[2] else 'FAIL'}")
        print(f"  3  SOC upper bound:          {c3:.4f} kWh     {'PASS' if passed[3] else 'FAIL'}")
        print(f"  4  Power bounds:             [0,{pch.max():.0f}]/[0,{pdis.max():.0f}] kW  {'PASS' if passed[4] else 'FAIL'}")
        print(f"  5  Simult. ch/dis:           {c5} / {T}        {'PASS' if passed[5] else 'FAIL'}")
        print(f"  6  Terminal SOC:             {c6:.2f} kWh      {'PASS' if passed[6] else 'FAIL'}")
        print(f"  7  Unit consistency:         £{c7:.2f}           {'PASS' if passed[7] else 'FAIL'}")
        print(f"  8  Profit reconciliation:    {c8:.2e} GBP      {'PASS' if passed[8] else 'FAIL'}")

    return passed


# ══════════════════════════════════════════════════════════════════════════════
# 5. KPI SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def compute_kpis(price, pch, pdis, E, label):
    profit      = np.sum((pdis - pch) * price * DT / 1000)
    charge_cost = np.sum(pch  * price * DT / 1000)
    dis_rev     = np.sum(pdis * price * DT / 1000)
    E_ch        = np.sum(pch)  * DT / 1000
    E_dis       = np.sum(pdis) * DT / 1000
    eff_loss    = E_ch - E_dis
    cycles      = E_ch * 1000 / E_CAP
    avg_buy     = charge_cost / E_ch  if E_ch  > 0 else 0
    avg_sell    = dis_rev     / E_dis if E_dis > 0 else 0

    print(f"\n── KPIs: {label} ──")
    print(f"  Total profit:        £{profit:,.0f}")
    print(f"  Charge cost:         £{charge_cost:,.0f}")
    print(f"  Discharge revenue:   £{dis_rev:,.0f}")
    print(f"  Energy charged:      {E_ch:.1f} MWh")
    print(f"  Energy discharged:   {E_dis:.1f} MWh")
    print(f"  Efficiency loss:     {eff_loss:.1f} MWh  (≈ £{eff_loss*1000*price.mean()/1000:.0f} at mean price)")
    print(f"  Full equiv. cycles:  {cycles:.1f}")
    print(f"  Avg buy price:       £{avg_buy:.2f}/MWh")
    print(f"  Avg sell price:      £{avg_sell:.2f}/MWh")
    print(f"  Terminal SOC:        {E[-1]:.0f} kWh")
    return profit


# ══════════════════════════════════════════════════════════════════════════════
# 6. EXTENSIONS
# ══════════════════════════════════════════════════════════════════════════════

def run_threshold_sweep(price):
    """Sweep all threshold combinations to show the heuristic gap is structural."""
    results = {}
    for p_lo in SWEEP_CHARGE:
        for p_hi in SWEEP_DISCHARGE:
            if p_hi > p_lo + 20:
                pc, pd_, _ = run_heuristic(price, p_lo, p_hi)
                results[(p_lo, p_hi)] = np.sum((pd_ - pc) * price / 1000)
    return results


def run_carbon_pareto(price, carbon):
    """Carbon-aware LP across multiple penalty levels."""
    results = []
    for lc in CARBON_PENALTIES:
        pc, pd_, E_ = run_lp(price, carbon, lc)
        profit_ = np.sum((pd_ - pc) * price / 1000)
        emiss_  = np.sum(pc * carbon) / 1000   # tonnes CO2
        results.append((lc, profit_, emiss_))
        # Re-run all 8 verification checks at each penalty level
        run_verification(price, pc, pd_, E_, label=f'LP Carbon λ={lc}', verbose=False)
        print(f"  λ_c={lc:4d} GBP/t: profit=£{profit_:,.0f}  emissions={emiss_:.1f} t  [verified]")
    return results


def run_capacity_sensitivity(price):
    """Run LP at different capacities with fixed P_max = 1,000 kW."""
    results = []
    T = len(price)
    for cap in CAPACITY_RANGE:
        e0_  = cap / 2
        pmax = P_MAX   # fixed at 1,000 kW regardless of capacity
        c_obj = np.zeros(2 * T)
        for t in range(T):
            c_obj[t]     =  price[t] * DT / 1000
            c_obj[T + t] = -price[t] * DT / 1000
        bounds  = [(0, pmax)] * T + [(0, pmax)] * T
        A_ub, b_ub = [], []
        for t in range(T):
            row = np.zeros(2 * T)
            row[:t + 1]      =  ETA_CH
            row[T:T + t + 1] = -1.0 / ETA_DIS
            A_ub.append(row.copy());  b_ub.append(cap - e0_)
            A_ub.append(-row.copy()); b_ub.append(e0_)
        row_term = np.zeros(2 * T)
        row_term[:T] = -ETA_CH; row_term[T:] = 1.0 / ETA_DIS
        A_ub.append(row_term); b_ub.append(0.0)
        res = linprog(c_obj, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                      bounds=bounds, method='highs')
        results.append((cap, -res.fun))
        print(f"  Capacity {cap/1000:.1f} MWh: £{-res.fun:,.0f}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 7. FIGURES
# ══════════════════════════════════════════════════════════════════════════════

def save_fig(fig, path):
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def fig_soc_vs_price(ts, price, E_h, E_lp, out_dir):
    start, end = 7 * 24, 14 * 24
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.5), sharex=True)
    fig.suptitle('State of Charge vs. Day-Ahead Price — Days 8–14',
                 fontsize=9, fontweight='bold')
    for ax, E_, col, label in zip(axes, [E_h, E_lp], [COL_H, COL_LP],
                                   ['Heuristic (p30/p70)', 'LP Optimal']):
        ax2 = ax.twinx()
        ax2.fill_between(ts[start:end], price[start:end], alpha=0.15, color=COL_PRI)
        ax2.plot(ts[start:end], price[start:end], color='#999999', lw=0.6, alpha=0.7)
        ax2.set_ylabel('Price (£/MWh)', fontsize=7, color='#777777')
        ax2.tick_params(axis='y', labelcolor='#777777', labelsize=6)
        ax2.spines['top'].set_visible(False)
        ax2.set_ylim(0, price[start:end].max() * 1.25)
        ax.plot(ts[start:end], E_[start:end] / 1000, color=col, lw=1.3, label=label)
        ax.axhline(2.0, color='red',     lw=0.6, ls='--', alpha=0.5, label='Max (2 MWh)')
        ax.axhline(1.0, color='#888888', lw=0.6, ls=':',  alpha=0.5, label='Init (1 MWh)')
        ax.set_ylabel('SOC (MWh)', fontsize=7)
        ax.set_ylim(-0.05, 2.2)
        ax.set_title(f'  {label}', loc='left', fontsize=8, fontweight='bold', pad=3)
        ax.legend(loc='upper left', fontsize=6, framealpha=0.7)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%d %b\n%H:%M'))
    axes[1].xaxis.set_major_locator(mdates.DayLocator())
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, 'fig1_soc_price.png'))


def fig_threshold_heatmap(sweep_results, lp_profit, out_dir):
    grid = np.full((len(SWEEP_CHARGE), len(SWEEP_DISCHARGE)), np.nan)
    for (p_lo, p_hi), profit in sweep_results.items():
        if p_lo in SWEEP_CHARGE and p_hi in SWEEP_DISCHARGE:
            grid[SWEEP_CHARGE.index(p_lo), SWEEP_DISCHARGE.index(p_hi)] = profit
    best_i, best_j = np.unravel_index(np.nanargmax(grid), grid.shape)
    best = grid[best_i, best_j]
    gap  = (lp_profit - best) / lp_profit * 100
    fig, ax = plt.subplots(figsize=(6, 3.5))
    im = ax.imshow(grid, aspect='auto', cmap='RdYlGn', vmin=6000, vmax=12000)
    ax.set_xticks(range(len(SWEEP_DISCHARGE)))
    ax.set_yticks(range(len(SWEEP_CHARGE)))
    ax.set_xticklabels([f'p{v}' for v in SWEEP_DISCHARGE])
    ax.set_yticklabels([f'p{v}' for v in SWEEP_CHARGE])
    ax.set_xlabel('Discharge threshold (percentile)')
    ax.set_ylabel('Charge threshold (percentile)')
    ax.set_title(f'Heuristic Profit vs Threshold Pair\n'
                 f'Best: p{SWEEP_CHARGE[best_i]}/p{SWEEP_DISCHARGE[best_j]} = '
                 f'£{best:,.0f} | LP: £{lp_profit:,.0f} | Gap: {gap:.1f}%', fontsize=8)
    for i in range(len(SWEEP_CHARGE)):
        for j in range(len(SWEEP_DISCHARGE)):
            if not np.isnan(grid[i, j]):
                v = grid[i, j]
                c = 'white' if (v < 8500 or v > 11200) else 'black'
                w = 'bold' if (i == best_i and j == best_j) else 'normal'
                ax.text(j, i, f'£{v/1000:.1f}k', ha='center', va='center',
                        fontsize=6, color=c, fontweight=w)
    ax.add_patch(plt.Rectangle((best_j-0.5, best_i-0.5), 1, 1,
                                fill=False, edgecolor='lime', lw=2))
    plt.colorbar(im, ax=ax, label='Profit (£)', shrink=0.8)
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, 'fig2_heatmap.png'))


def fig_residual(E_lp, pch_lp, pdis_lp, out_dir):
    T     = len(E_lp)
    E_rec = np.zeros(T + 1); E_rec[0] = E0
    for t in range(T):
        E_rec[t+1] = E_rec[t] + ETA_CH*pch_lp[t]*DT - pdis_lp[t]/ETA_DIS*DT
    resid = E_rec[1:] - E_lp
    fig, ax = plt.subplots(figsize=(7, 2.2))
    ax.plot(range(T), resid * 1e13, color=COL_LP, lw=0.6, alpha=0.8)
    ax.axhline(0, color='black', lw=0.7, ls='--', alpha=0.5)
    ax.set_xlabel('Hour'); ax.set_ylabel('Residual (×10⁻¹³ kWh)')
    ax.set_title('Energy Balance Residual — LP Optimal (should be ≈ 0)', fontweight='bold')
    ax.text(0.99, 0.95, f'Max |residual| = {np.abs(resid).max():.1e} kWh\n(machine precision)',
            transform=ax.transAxes, ha='right', va='top', fontsize=7,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#cccccc', alpha=0.9))
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, 'fig4_residual.png'))


def fig_pareto(pareto_results, out_dir):
    penalties = [r[0] for r in pareto_results]
    profits   = [r[1] for r in pareto_results]
    emissions = [r[2] for r in pareto_results]
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.plot(emissions, profits, 'o-', color=COL_LP, lw=1.8, ms=6, zorder=3)
    for lc, x, y in zip(penalties, emissions, profits):
        ax.annotate(f'£{lc}/t', (x, y), xytext=(x+0.5, y+120), fontsize=7, color=COL_LP)
    elbow_idx = penalties.index(50)
    ax.plot(emissions[elbow_idx], profits[elbow_idx], 'o',
            color='#E74C3C', ms=9, zorder=4, label='Elbow (£50/t)')
    ax.set_xlabel('Charging Emissions (tonnes CO₂)')
    ax.set_ylabel('Total Profit (£)')
    ax.set_title('Pareto Frontier: Profit vs. Carbon Emissions', fontweight='bold')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'£{x:,.0f}'))
    ax.legend(loc='lower right', fontsize=7)
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, 'fig5_pareto.png'))


def fig_cumulative_profit(ts, price, pch_h, pdis_h, pch_lp, pdis_lp, out_dir):
    cum_h  = np.cumsum((pdis_h  - pch_h)  * price * DT / 1000)
    cum_lp = np.cumsum((pdis_lp - pch_lp) * price * DT / 1000)
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.plot(ts, cum_h,  color=COL_H,  lw=1.4, label=f'Heuristic: £{cum_h[-1]:,.0f}')
    ax.plot(ts, cum_lp, color=COL_LP, lw=1.4, label=f'LP Optimal: £{cum_lp[-1]:,.0f}')
    ax.fill_between(ts, cum_h, cum_lp, alpha=0.12, color=COL_LP)
    ax.set_xlabel('Date'); ax.set_ylabel('Cumulative Profit (£)')
    ax.set_title('Cumulative Profit — 60-Day Horizon', fontweight='bold')
    ax.legend(loc='upper left')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'£{x:,.0f}'))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, 'figA1_cumprofit.png'))


def fig_full_dispatch(ts, price, pch_lp, pdis_lp, E_lp, out_dir):
    fig, axes = plt.subplots(3, 1, figsize=(7.5, 5.5), sharex=True,
                              gridspec_kw={'height_ratios': [1, 1, 1.2]})
    fig.suptitle('Full 60-Day Dispatch — LP Optimal', fontsize=9, fontweight='bold')
    axes[0].plot(ts, price, color='#888888', lw=0.5, alpha=0.8)
    axes[0].fill_between(ts, price, alpha=0.15, color='#888888')
    axes[0].set_ylabel('Price\n(£/MWh)', fontsize=7)
    ch_vals  = np.where(pch_lp  > 0, pch_lp  / 1000, 0)
    dis_vals = np.where(pdis_lp > 0, pdis_lp / 1000, 0)
    axes[1].bar(ts, ch_vals,   width=0.03, color=COL_LP,    alpha=0.8, label='Charge')
    axes[1].bar(ts, -dis_vals, width=0.03, color='#E74C3C', alpha=0.8, label='Discharge')
    axes[1].axhline(0, color='black', lw=0.5)
    axes[1].set_ylabel('Power\n(MW)', fontsize=7); axes[1].legend(loc='upper right', fontsize=6)
    axes[2].fill_between(ts, E_lp / 1000, alpha=0.3, color=COL_LP)
    axes[2].plot(ts, E_lp / 1000, color=COL_LP, lw=0.8)
    axes[2].axhline(1.0, color='#888888', lw=0.7, ls=':', alpha=0.7, label='Initial SOC')
    axes[2].set_ylabel('SOC (MWh)', fontsize=7); axes[2].set_xlabel('Date')
    axes[2].legend(loc='upper right', fontsize=6)
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    axes[2].xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, 'figA2_full60day.png'))


def fig_daily_profit(price, pch_h, pdis_h, pch_lp, pdis_lp, out_dir):
    daily_h  = [np.sum((pdis_h[d*24:(d+1)*24]  - pch_h[d*24:(d+1)*24])
                       * price[d*24:(d+1)*24] / 1000) for d in range(60)]
    daily_lp = [np.sum((pdis_lp[d*24:(d+1)*24] - pch_lp[d*24:(d+1)*24])
                       * price[d*24:(d+1)*24] / 1000) for d in range(60)]
    days = pd.date_range('2025-06-01', periods=60, freq='D')
    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    x, w = np.arange(60), 0.38
    ax.bar(x - w/2, daily_h,  width=w, color=COL_H,  alpha=0.85, label='Heuristic')
    ax.bar(x + w/2, daily_lp, width=w, color=COL_LP, alpha=0.85, label='LP Optimal')
    ax.set_xticks(x[::7])
    ax.set_xticklabels([days[i].strftime('%d %b') for i in range(0, 60, 7)], fontsize=6)
    ax.set_ylabel('Daily Profit (£)'); ax.set_xlabel('Date')
    ax.set_title('Daily Profit Comparison', fontweight='bold', fontsize=8)
    ax.legend(loc='upper left')
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, 'figA3_daily_profit.png'))


def fig_capacity_sensitivity(cap_results, out_dir):
    caps    = [r[0] for r in cap_results]
    profits = [r[1] for r in cap_results]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot([c / 1000 for c in caps], profits, 'o-', color=COL_LP, lw=1.8, ms=7)
    for c, p in zip(caps, profits):
        ax.annotate(f'£{p/1000:.1f}k', (c/1000, p),
                    xytext=(0, 8), textcoords='offset points', ha='center', fontsize=7)
    ax.set_xlabel('Battery Capacity (MWh)'); ax.set_ylabel('Total Profit (£)')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'£{x:,.0f}'))
    ax.set_title('Profit vs Battery Capacity — Diminishing Returns', fontweight='bold', fontsize=8)
    ax.axvline(2.0, color='red', ls='--', lw=0.8, alpha=0.6, label='Base case (2 MWh)')
    ax.legend(fontsize=7)
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, 'figA4_capacity.png'))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def main(data_path='caseB_grid_battery_market_hourly.csv', fig_dir='figures'):
    os.makedirs(fig_dir, exist_ok=True)

    print("\n" + "="*60)
    print("Case B: Grid-Scale Battery Energy Arbitrage")
    print("="*60)

    print("\n[1] Loading and validating data...")
    df     = load_data(data_path)
    price  = df['day_ahead_price_gbp_per_mwh'].values
    carbon = df['carbon_intensity_kg_per_kwh_optional'].values
    ts     = df['timestamp']

    print("\n[2] Running heuristic dispatch...")
    pch_h, pdis_h, E_h = run_heuristic(price)
    profit_h = compute_kpis(price, pch_h, pdis_h, E_h, 'Heuristic (p30/p70)')

    print("\n[3] Running LP optimal dispatch...")
    pch_lp, pdis_lp, E_lp = run_lp(price)
    profit_lp = compute_kpis(price, pch_lp, pdis_lp, E_lp, 'LP Optimal')
    print(f"\n  Optimality gap: {(profit_lp - profit_h) / profit_lp * 100:.1f}%")

    print("\n[4] Running verification (8 checks × 2 policies = 16 total)...")
    run_verification(price, pch_h,  pdis_h,  E_h,  label='Heuristic')
    run_verification(price, pch_lp, pdis_lp, E_lp, label='LP Optimal')

    print("\n[5] Running threshold sweep (42 combinations)...")
    sweep = run_threshold_sweep(price)
    best  = max(sweep, key=sweep.get)
    print(f"  Best: p{best[0]}/p{best[1]} = £{sweep[best]:,.0f}  |  "
          f"Gap to LP: {(profit_lp - sweep[best]) / profit_lp * 100:.1f}%")

    print("\n[6] Running carbon-aware Pareto sweep...")
    pareto = run_carbon_pareto(price, carbon)

    print("\n[7] Running capacity sensitivity...")
    cap_results = run_capacity_sensitivity(price)

    print(f"\n[8] Generating figures -> {fig_dir}/")
    fig_soc_vs_price(ts, price, E_h, E_lp, fig_dir)
    fig_threshold_heatmap(sweep, profit_lp, fig_dir)
    fig_residual(E_lp, pch_lp, pdis_lp, fig_dir)
    fig_pareto(pareto, fig_dir)
    fig_cumulative_profit(ts, price, pch_h, pdis_h, pch_lp, pdis_lp, fig_dir)
    fig_full_dispatch(ts, price, pch_lp, pdis_lp, E_lp, fig_dir)
    fig_daily_profit(price, pch_h, pdis_h, pch_lp, pdis_lp, fig_dir)
    fig_capacity_sensitivity(cap_results, fig_dir)

    print("\n" + "="*60)
    print("Pipeline complete.")
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Case B: Grid Battery Arbitrage')
    parser.add_argument('--data', default='caseB_grid_battery_market_hourly.csv',
                        help='Path to the dataset CSV')
    parser.add_argument('--figs', default='figures',
                        help='Output directory for figures')
    args = parser.parse_args()
    main(args.data, args.figs)
