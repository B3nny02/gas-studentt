"""
==============================================================================
EMPIRICAL APPLICATION: Multi-Asset Portfolio Analysis (2017-2024)
==============================================================================

Framework: GAS-Student-t with Dynamic Fisher Information Scaling
Assets: SPY (Equity), QQQ (Tech), TLT (Bonds), GLD (Gold)
Sample: 2017-2024 (~1,750 daily observations)

Analysis Components:
1. Data preparation and descriptive statistics
2. Model estimation with 3 scaling methods
3. Comprehensive diagnostic analysis
4. Value-at-Risk backtesting with statistical tests
5. Crisis period subsample analysis (COVID, Inflation)
6. Dynamic correlation regime detection
7. Full visualization suite from framework

Output: 15+ plots, 6 tables
==============================================================================
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import sys
import warnings

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

# Import GAS Framework
from gas_studentt.models.gas_studentt_multivariate import GASStudentTMultivariate
from gas_studentt.utils.config import GASConfig
from gas_studentt.core.cholesky import vec_to_chol_d, chol_to_sigma_d

# Import diagnostics
from gas_studentt.utils.diagnostics import (
    check_pd, condition_numbers, score_norms, full_diagnostics,
    dynamic_correlation, logdet_series, min_eigenvalue_series,
    condition_number_series, score_norm_series, state_variation,
    correlation_matrix
)

# Import visualizations
from gas_studentt.utils.visualization import (
    plot_volatility, plot_dynamic_correlation, plot_condition_number,
    plot_min_eigenvalue, plot_logdet, plot_score_norm, plot_variances,
    plot_correlation_distribution, qqplot_residuals,
    plot_standardized_residuals, plot_fisher_history
)

# Set random seed for reproducibility
np.random.seed(42)

# Output directory
output_dir = Path('')
output_dir.mkdir(parents=True, exist_ok=True)

# =============================================================================
# SECTION 1: DATA LOADING & PREPARATION
# =============================================================================
print("\n" + "="*80)
print("1. DATA COLLECTION AND PREPROCESSING")
print("="*80)

# Asset selection: diversified portfolio
tickers = ['SPY', 'QQQ', 'TLT', 'GLD']
ticker_names = {
    'SPY': 'S&P 500 ETF',
    'QQQ': 'Nasdaq-100 ETF',
    'TLT': '20+ Year Treasury Bond ETF',
    'GLD': 'Gold ETF'
}

# Sample period
start_date = '2017-01-01'
end_date = '2024-01-01'

print(f"\nDownloading data for {tickers}...")
print(f"Period: {start_date} to {end_date}")

# Download data
raw_data = yf.download(tickers, start=start_date, end=end_date)['Close']
raw_data = raw_data.dropna()

# Compute log-returns (in percentage)
Y_df = 100 * np.log(raw_data / raw_data.shift(1)).dropna()
Y_df = Y_df[tickers]  # Ensure column order
Y = Y_df.values
dates = Y_df.index
T, d = Y.shape

print(f"\nDataset Properties:")
print(f"  Observations: {T}")
print(f"  Assets: {d}")
print(f"  Frequency: Daily")
print(f"  Period: {dates[0].date()} to {dates[-1].date()}")

# Descriptive statistics
print("\n" + "-"*80)
print("Descriptive Statistics (Daily Returns, %)")
print("-"*80)
desc_stats = Y_df.describe()
print(desc_stats.round(4))

# Unconditional correlation matrix
print("\n" + "-"*80)
print("Unconditional Correlation Matrix")
print("-"*80)
corr_uncond = Y_df.corr()
print(corr_uncond.round(3))

# Save descriptive statistics
desc_stats.to_csv(output_dir / 'descriptive_statistics.csv')
corr_uncond.to_csv(output_dir / 'unconditional_correlation.csv')

# Additional statistics
print("\n" + "-"*80)
print("Higher Moments")
print("-"*80)
moments_table = []
for ticker in tickers:
    data = Y_df[ticker]
    moments_table.append({
        'Asset': ticker,
        'Mean': data.mean(),
        'Std': data.std(),
        'Skewness': data.skew(),
        'Kurtosis': data.kurtosis(),
        'Min': data.min(),
        'Max': data.max(),
        'JB_Stat': stats.jarque_bera(data)[0],
        'JB_Pvalue': stats.jarque_bera(data)[1]
    })

df_moments = pd.DataFrame(moments_table)
print(df_moments.to_string(index=False))
df_moments.to_csv(output_dir / 'higher_moments.csv', index=False)


# =============================================================================
# SECTION 2: MODEL ESTIMATION WITH COMPETITIVE SCALINGS
# =============================================================================
print("\n" + "="*80)
print("2. MODEL ESTIMATION")
print("="*80)

scaling_methods = ['identity', 'sqrt_fisher_diag', 'fisher_diag']
results = {}

for scaling in scaling_methods:
    print(f"\n{'-'*80}")
    print(f"Estimating: {scaling.upper()}")
    print(f"{'-'*80}")

    try:
        # Initialize model
        model = GASStudentTMultivariate(
            d=d,
            scaling=scaling,
            update_type='diagonal',
            config=GASConfig(d=d)
        )

        # Precompute Fisher grid if needed
        if scaling != 'identity':
            print("  [*] Pre-computing Fisher Information Grid...")
            model.precompute_fisher_grid(
                h_min=-12.0,
                h_max=12.0,
                n_points=60,
                n_mc=1000,
                nu_target=7.0
            )

        # Estimate model
        print("  [*] Running optimization (this may take a few minutes)...")
        fit_res = model.fit(
            Y,
            method='L-BFGS-B',
            maxiter=1000,
            multistart=5
        )

        # Display results
        print(f"\n  Results:")
        print(f"    Converged: {fit_res.success}")
        print(f"    Log-Likelihood: {fit_res.ll:.2f}")
        print(f"    AIC: {fit_res.aic:.2f}")
        print(f"    BIC: {fit_res.bic:.2f}")
        print(f"    Nu (DoF): {fit_res.params['nu']:.4f}")
        print(f"    PD Failures: {fit_res.n_pd_failures}")

        # Store results
        results[scaling] = {
            'fit': fit_res,
            'model': model,
            'Sigmas': model.last_Sigmas(),
            'h': model.last_h(),
            'scores': model.last_scores()
        }

    except Exception as e:
        print(f"  [!] ESTIMATION FAILED: {str(e)}")
        results[scaling] = None


# =============================================================================
# SECTION 3: MODEL COMPARISON & SELECTION
# =============================================================================
print("\n" + "="*80)
print("3. MODEL COMPARISON AND SELECTION")
print("="*80)

comparison_table = []

for scaling in scaling_methods:
    if results[scaling] is not None:
        res = results[scaling]

        # Compute diagnostics
        pd_stats = check_pd(res['Sigmas'])
        cond_stats = condition_numbers(res['Sigmas'])
        score_stats = score_norms(res['scores'])

        comparison_table.append({
            'Scaling': scaling,
            'LogLik': res['fit'].ll,
            'AIC': res['fit'].aic,
            'BIC': res['fit'].bic,
            'Nu': res['fit'].params['nu'],
            'MinEig': pd_stats['min_eig_min'],
            'Cond_Med': cond_stats['median'],
            'Cond_Max': cond_stats['max'],
            'Score_Mean': score_stats['mean'],
            'Score_Max': score_stats['max'],
            'PD_Fail': res['fit'].n_pd_failures
        })

df_comparison = pd.DataFrame(comparison_table)
df_comparison = df_comparison.sort_values('BIC')

print("\n" + "-"*80)
print("Model Comparison Table (sorted by BIC)")
print("-"*80)
print(df_comparison.to_string(index=False))

# Save comparison
df_comparison.to_csv(output_dir / 'model_comparison.csv', index=False)

# Select best model
best_scaling = df_comparison.iloc[0]['Scaling']
best_res = results[best_scaling]
best_Sigmas = best_res['Sigmas']
best_h = best_res['h']
best_scores = best_res['scores']
best_params = best_res['fit'].params

print(f"\n{'='*80}")
print(f"SELECTED MODEL: {best_scaling.upper()}")
print(f"{'='*80}")
print(f"  BIC: {best_res['fit'].bic:.2f}")
print(f"  Degrees of Freedom: {best_params['nu']:.4f}")
print(f"  Numerical Stability: Cond_Max = {df_comparison.iloc[0]['Cond_Max']:.2f}")


# =============================================================================
# SECTION 4: COMPREHENSIVE DIAGNOSTICS
# =============================================================================
print("\n" + "="*80)
print("4. COMPREHENSIVE DIAGNOSTIC ANALYSIS")
print("="*80)

# Full diagnostics
full_diag = full_diagnostics(best_Sigmas, best_scores, best_h)

print("\n" + "-"*80)
print("Positive Definiteness & Conditioning")
print("-"*80)
print(f"  Min Eigenvalue (min): {full_diag['pd']['min_eig_min']:.6f}")
print(f"  Min Eigenvalue (median): {full_diag['pd']['min_eig_median']:.6f}")
print(f"  PD Violations: {full_diag['pd']['n_violations']}")
print(f"  Condition Number (median): {full_diag['conditioning']['median']:.2f}")
print(f"  Condition Number (max): {full_diag['conditioning']['max']:.2f}")

print("\n" + "-"*80)
print("Score Statistics")
print("-"*80)
print(f"  Mean Norm: {full_diag['score_norms']['mean']:.4f}")
print(f"  Median Norm: {full_diag['score_norms']['median']:.4f}")
print(f"  95th Percentile: {full_diag['score_norms']['p95']:.4f}")
print(f"  Max Norm: {full_diag['score_norms']['max']:.4f}")
print(f"  Autocorr (lag 1): {full_diag['score_autocorr_lag1']:.4f}")

print("\n" + "-"*80)
print("State Variation")
print("-"*80)
print(f"  Mean: {full_diag['state_variation']['mean']:.4f}")
print(f"  Median: {full_diag['state_variation']['median']:.4f}")
print(f"  Max: {full_diag['state_variation']['max']:.4f}")

print("\n" + "-"*80)
print("Log-Determinant (Total Variance)")
print("-"*80)
print(f"  Mean: {full_diag['logdet']['mean']:.4f}")
print(f"  Median: {full_diag['logdet']['median']:.4f}")
print(f"  Min: {full_diag['logdet']['min']:.4f}")
print(f"  Max: {full_diag['logdet']['max']:.4f}")

# Save diagnostics
diag_summary = {
    'Metric': ['MinEig_Min', 'MinEig_Median', 'Cond_Median', 'Cond_Max',
               'Score_Mean', 'Score_Max', 'StateVar_Mean', 'LogDet_Mean'],
    'Value': [
        full_diag['pd']['min_eig_min'],
        full_diag['pd']['min_eig_median'],
        full_diag['conditioning']['median'],
        full_diag['conditioning']['max'],
        full_diag['score_norms']['mean'],
        full_diag['score_norms']['max'],
        full_diag['state_variation']['mean'],
        full_diag['logdet']['mean']
    ]
}
pd.DataFrame(diag_summary).to_csv(output_dir / 'diagnostic_summary.csv', index=False)


# =============================================================================
# SECTION 5: VALUE-AT-RISK BACKTESTING
# =============================================================================
print("\n" + "="*80)
print("5. VALUE-AT-RISK BACKTESTING")
print("="*80)

# Equal-weighted portfolio
weights = np.ones(d) / d
port_ret = Y @ weights

# Compute portfolio variance and volatility
port_vars = np.array([weights.T @ best_Sigmas[t] @ weights for t in range(T)])
port_vols = np.sqrt(port_vars)

# Student-t quantiles
nu_hat = best_params['nu']
alpha_levels = [0.01, 0.025, 0.05]
var_quantiles = {}

for alpha in alpha_levels:
    t_q = stats.t.ppf(alpha, df=nu_hat)
    var_quantiles[alpha] = port_vols * t_q

# Count violations
var_results = []
for alpha in alpha_levels:
    hits = port_ret < var_quantiles[alpha]
    n_viol = hits.sum()
    viol_rate = n_viol / T

    var_results.append({
        'Confidence': f'{(1-alpha)*100:.1f}%',
        'Alpha': alpha,
        'Target_Rate': alpha,
        'Violations': n_viol,
        'Empirical_Rate': viol_rate,
        'Ratio': viol_rate / alpha
    })

    print(f"\nVaR {(1-alpha)*100:.1f}%:")
    print(f"  Target violations: {alpha*100:.2f}%")
    print(f"  Actual violations: {n_viol} / {T} = {viol_rate*100:.2f}%")
    print(f"  Ratio: {viol_rate/alpha:.3f}")

df_var = pd.DataFrame(var_results)
df_var.to_csv(output_dir / 'var_backtest_results.csv', index=False)

# Statistical tests (Christoffersen)
print("\n" + "-"*80)
print("Christoffersen Tests")
print("-"*80)

def christoffersen_uc_test(violations, T, alpha):
    """Unconditional Coverage Test"""
    n_viol = violations.sum()
    p_hat = n_viol / T

    if p_hat > 0 and p_hat < 1:
        lr = -2 * (
            n_viol * np.log(alpha) + (T - n_viol) * np.log(1 - alpha) -
            n_viol * np.log(p_hat) - (T - n_viol) * np.log(1 - p_hat)
        )
    else:
        lr = 0

    p_value = 1 - stats.chi2.cdf(lr, df=1)
    return {'LR': lr, 'p_value': p_value, 'reject': p_value < 0.05}

test_results = []
for alpha in [0.01, 0.05]:
    hits = port_ret < var_quantiles[alpha]
    uc_test = christoffersen_uc_test(hits, T, alpha)

    test_results.append({
        'Test': f'UC_{int((1-alpha)*100)}%',
        'LR_Stat': uc_test['LR'],
        'P_Value': uc_test['p_value'],
        'Result': 'Reject' if uc_test['reject'] else 'Accept'
    })

    print(f"  UC Test ({(1-alpha)*100:.0f}%): LR={uc_test['LR']:.3f}, p={uc_test['p_value']:.3f} -> {test_results[-1]['Result']}")

pd.DataFrame(test_results).to_csv(output_dir / 'var_statistical_tests.csv', index=False)


# =============================================================================
# SECTION 6: CRISIS PERIOD ANALYSIS
# =============================================================================
print("\n" + "="*80)
print("6. CRISIS PERIOD SUBSAMPLE ANALYSIS")
print("="*80)

# Define crisis periods
crisis_periods = {
    'Pre-COVID (2019)': ('2019-01-01', '2019-12-31'),
    'COVID Crash (Q1 2020)': ('2020-02-15', '2020-04-30'),
    'Recovery (2021)': ('2021-01-01', '2021-12-31'),
    'Inflation Crisis (2022)': ('2022-01-01', '2022-12-31'),
    'Full Sample': (dates[0], dates[-1])
}

crisis_stats = []

for period_name, (start, end) in crisis_periods.items():
    mask = (dates >= start) & (dates <= end)
    n_obs = mask.sum()

    if n_obs > 0:
        # Returns
        port_ret_period = port_ret[mask]

        # Volatilities
        vols_period = np.sqrt(np.diagonal(best_Sigmas[mask], axis1=1, axis2=2))

        # Correlations (average pairwise)
        corrs_period = []
        for i in range(d):
            for j in range(i+1, d):
                rho = dynamic_correlation(best_Sigmas[mask], i, j)
                corrs_period.append(rho.mean())

        crisis_stats.append({
            'Period': period_name,
            'N_Obs': n_obs,
            'Ret_Mean': port_ret_period.mean(),
            'Ret_Std': port_ret_period.std(),
            'Ret_Min': port_ret_period.min(),
            'Ret_Max': port_ret_period.max(),
            'Vol_Mean': vols_period.mean(),
            'Vol_Max': vols_period.max(),
            'Corr_Mean': np.mean(corrs_period)
        })

df_crisis = pd.DataFrame(crisis_stats)
print("\n", df_crisis.to_string(index=False))
df_crisis.to_csv(output_dir / 'crisis_period_analysis.csv', index=False)


# =============================================================================
# SECTION 7: DYNAMIC CORRELATION REGIME DETECTION
# =============================================================================
print("\n" + "="*80)
print("7. DYNAMIC CORRELATION ANALYSIS")
print("="*80)

# Compute correlation matrices over time
Corrs = np.zeros((T, d, d))
for t in range(T):
    Corrs[t] = correlation_matrix(best_Sigmas[t])

# Key pairs for financial interpretation
key_pairs = [
    ('SPY', 'TLT', 'Equity-Bond (Flight-to-Safety)'),
    ('SPY', 'GLD', 'Equity-Gold (Safe Haven)'),
    ('SPY', 'QQQ', 'Equity-Tech (Beta)'),
    ('TLT', 'GLD', 'Bond-Gold (Inflation Hedge)')
]

corr_stats = []
for ticker1, ticker2, description in key_pairs:
    i, j = tickers.index(ticker1), tickers.index(ticker2)
    rho = Corrs[:, i, j]

    corr_stats.append({
        'Pair': f'{ticker1}-{ticker2}',
        'Description': description,
        'Mean': rho.mean(),
        'Std': rho.std(),
        'Min': rho.min(),
        'Max': rho.max(),
        'Pre_COVID': rho[(dates >= '2019-01-01') & (dates <= '2019-12-31')].mean(),
        'COVID': rho[(dates >= '2020-02-15') & (dates <= '2020-04-30')].mean(),
        'Inflation': rho[(dates >= '2022-01-01') & (dates <= '2022-12-31')].mean()
    })

df_corr_stats = pd.DataFrame(corr_stats)
print("\n", df_corr_stats.to_string(index=False))
df_corr_stats.to_csv(output_dir / 'correlation_regime_statistics.csv', index=False)


# =============================================================================
# SECTION 8: VISUALIZATION SUITE
# =============================================================================
print("\n" + "="*80)
print("8. GENERATING COMPREHENSIVE VISUALIZATIONS")
print("="*80)

# Helper function for event markers
def add_event_markers(ax):
    """Add vertical lines for major market events"""
    events = {
        'COVID': pd.Timestamp('2020-03-15'),
        'Vaccine': pd.Timestamp('2020-11-09'),
        'Rate Hikes': pd.Timestamp('2022-03-16')
    }
    for name, date in events.items():
        if date >= dates[0] and date <= dates[-1]:
            ax.axvline(date, color='red', ls=':', alpha=0.4, lw=1)
            ax.text(date, ax.get_ylim()[1]*0.95, f' {name}',
                   rotation=90, va='top', fontsize=7, alpha=0.6)

print("  Generating plots (1/15)...")

# PLOT 1: Volatility paths
fig1 = plot_volatility(best_Sigmas, dates, tickers=tickers, figsize=(14, 7))
add_event_markers(fig1.axes[0])
fig1.savefig(output_dir / '01_volatility_paths.png', dpi=300, bbox_inches='tight')
plt.close(fig1)

print("  Generating plots (2/15)...")

# PLOT 2: All pairwise correlations
fig2 = plot_dynamic_correlation(best_Sigmas, dates, bands=True, figsize=(14, 10))
for ax in fig2.axes:
    add_event_markers(ax)
fig2.savefig(output_dir / '02_all_correlations.png', dpi=300, bbox_inches='tight')
plt.close(fig2)

print("  Generating plots (3/15)...")

# PLOT 3: Key correlation pairs (custom)
fig3, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
axes = axes.ravel()

for idx, (t1, t2, desc) in enumerate(key_pairs[:4]):
    i, j = tickers.index(t1), tickers.index(t2)
    rho = Corrs[:, i, j]

    axes[idx].plot(dates, rho, lw=1.5, color=f'C{idx}')
    axes[idx].axhline(rho.mean(), ls='--', color='black', alpha=0.5,
                     label=f'Mean={rho.mean():.2f}')
    axes[idx].fill_between(dates, rho.mean()-rho.std(), rho.mean()+rho.std(),
                           alpha=0.2, color=f'C{idx}')
    add_event_markers(axes[idx])
    axes[idx].set_title(f'{t1}-{t2}: {desc}')
    axes[idx].set_ylabel('Correlation')
    axes[idx].legend(loc='upper left', fontsize=8)
    axes[idx].grid(alpha=0.3)

axes[-1].set_xlabel('Date')
fig3.suptitle('Key Correlation Dynamics', fontsize=14, fontweight='bold')
plt.tight_layout()
fig3.savefig(output_dir / '03_key_correlations.png', dpi=300, bbox_inches='tight')
plt.close(fig3)

print("  Generating plots (4/15)...")

# PLOT 4: Correlation heatmaps by period
fig4, axes = plt.subplots(1, 4, figsize=(18, 4))
heatmap_periods = [
    ('2019-01-01', '2019-12-31', 'Pre-COVID'),
    ('2020-02-15', '2020-04-30', 'COVID Crash'),
    ('2021-01-01', '2021-12-31', 'Recovery'),
    ('2022-01-01', '2022-12-31', 'Inflation')
]

for idx, (start, end, name) in enumerate(heatmap_periods):
    mask = (dates >= start) & (dates <= end)
    if mask.sum() > 0:
        avg_corr = Corrs[mask].mean(axis=0)
        sns.heatmap(avg_corr, annot=True, fmt=".2f", cmap='RdBu_r',
                   vmin=-0.5, vmax=1.0, center=0.3,
                   xticklabels=tickers, yticklabels=tickers,
                   ax=axes[idx], cbar=(idx==3))
        axes[idx].set_title(name)

fig4.suptitle('Correlation Matrix Evolution Across Regimes', fontweight='bold')
plt.tight_layout()
fig4.savefig(output_dir / '04_correlation_heatmaps.png', dpi=300, bbox_inches='tight')
plt.close(fig4)

print("  Generating plots (5/15)...")

# PLOT 5: VaR backtest (1% level)
fig5, ax = plt.subplots(figsize=(14, 6))
ax.plot(dates, port_ret, color='grey', alpha=0.4, lw=0.6, label='Portfolio Return')
ax.plot(dates, var_quantiles[0.01], color='red', lw=1.5, label='VaR 1%')
hits_1 = port_ret < var_quantiles[0.01]
ax.scatter(dates[hits_1], port_ret[hits_1], color='darkred', s=40, zorder=5,
          label=f'Violations ({hits_1.sum()})')
add_event_markers(ax)
ax.set_title(f'Value-at-Risk Backtesting (99% Confidence, nu={nu_hat:.2f})')
ax.set_ylabel('Return (%)')
ax.legend(loc='lower left')
ax.grid(alpha=0.3)
plt.tight_layout()
fig5.savefig(output_dir / '05_var_backtest.png', dpi=300, bbox_inches='tight')
plt.close(fig5)

print("  Generating plots (6/15)...")

# PLOT 6: Portfolio volatility
fig6, ax = plt.subplots(figsize=(14, 5))
ax.plot(dates, port_vols, lw=1.2, color='purple')
ax.fill_between(dates, 0, port_vols, alpha=0.2, color='purple')
add_event_markers(ax)
ax.set_title('Portfolio Volatility (Equal-Weighted)')
ax.set_ylabel('Volatility (%)')
ax.grid(alpha=0.3)
plt.tight_layout()
fig6.savefig(output_dir / '06_portfolio_volatility.png', dpi=300, bbox_inches='tight')
plt.close(fig6)

print("  Generating plots (7/15)...")

# PLOT 7: Condition number
fig7 = plot_condition_number(best_Sigmas, threshold=1000, figsize=(14, 5))
ax = fig7.axes[0]
ax.clear()
conds = condition_number_series(best_Sigmas)
ax.plot(dates, conds, lw=1.2, color='purple')
ax.axhline(1000, color='red', ls='--', label='Threshold')
add_event_markers(ax)
ax.set_yscale('log')
ax.set_ylabel('cond(Sigma)')
ax.set_title('Condition Number (Numerical Stability)')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
fig7.savefig(output_dir / '07_condition_number.png', dpi=300, bbox_inches='tight')
plt.close(fig7)

print("  Generating plots (8/15)...")

# PLOT 8: Minimum eigenvalue
fig8 = plot_min_eigenvalue(best_Sigmas, figsize=(14, 5))
ax = fig8.axes[0]
ax.clear()
min_eigs = min_eigenvalue_series(best_Sigmas)
ax.plot(dates, min_eigs, lw=1.2, color='green')
ax.axhline(0, ls='--', color='red')
add_event_markers(ax)
ax.set_ylabel('min eigenvalue')
ax.set_title('Minimum Eigenvalue (PD Verification)')
ax.grid(alpha=0.3)
plt.tight_layout()
fig8.savefig(output_dir / '08_min_eigenvalue.png', dpi=300, bbox_inches='tight')
plt.close(fig8)

print("  Generating plots (9/15)...")

# PLOT 9: Log-determinant
fig9 = plot_logdet(best_Sigmas, figsize=(14, 5))
ax = fig9.axes[0]
ax.clear()
logdets = logdet_series(best_Sigmas)
ax.plot(dates, logdets, lw=1.2, color='navy')
add_event_markers(ax)
ax.set_ylabel('log det(Sigma)')
ax.set_title('Log-Determinant (Total System Variance)')
ax.grid(alpha=0.3)
plt.tight_layout()
fig9.savefig(output_dir / '09_logdet.png', dpi=300, bbox_inches='tight')
plt.close(fig9)

print("  Generating plots (10/15)...")

# PLOT 10: Score norms
fig10 = plot_score_norm(best_scores, figsize=(14, 5))
ax = fig10.axes[0]
ax.clear()
score_norms_series = score_norm_series(best_scores)
ax.plot(dates, score_norms_series, lw=1.2, color='orange')
add_event_markers(ax)
ax.set_ylabel('||score||')
ax.set_title('Score Norm Dynamics')
ax.grid(alpha=0.3)
plt.tight_layout()
fig10.savefig(output_dir / '10_score_norms.png', dpi=300, bbox_inches='tight')
plt.close(fig10)

print("  Generating plots (11/15)...")

# PLOT 11: Correlation distribution
fig11 = plot_correlation_distribution(best_Sigmas, bins=40, figsize=(8, 5))
fig11.savefig(output_dir / '11_correlation_distribution.png', dpi=300, bbox_inches='tight')
plt.close(fig11)

print("  Generating plots (12/15)...")

# PLOT 12: Standardized residuals
Z = np.zeros_like(Y)
for t in range(T):
    try:
        L = np.linalg.cholesky(best_Sigmas[t])
        Z[t] = np.linalg.solve(L, Y[t] - best_params['mu'])
    except:
        Z[t] = Y[t]

fig12 = plot_standardized_residuals(Z, title_prefix='Standardized Residuals', figsize=(14, 8))
fig12.savefig(output_dir / '12_standardized_residuals.png', dpi=300, bbox_inches='tight')
plt.close(fig12)

print("  Generating plots (13/15)...")

# PLOT 13: QQ-plots
fig13 = qqplot_residuals(Z, nu=nu_hat, figsize=(14, 4))
fig13.savefig(output_dir / '13_qqplots.png', dpi=300, bbox_inches='tight')
plt.close(fig13)

print("  Generating plots (14/15)...")

# PLOT 14: Asset returns (time series)
fig14, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
for i, ticker in enumerate(tickers):
    axes[i].plot(dates, Y[:, i], lw=0.8, alpha=0.7, color=f'C{i}')
    axes[i].axhline(0, color='black', ls=':', alpha=0.5)
    add_event_markers(axes[i])
    axes[i].set_ylabel(f'{ticker} (%)')
    axes[i].set_title(ticker_names[ticker])
    axes[i].grid(alpha=0.3)
axes[-1].set_xlabel('Date')
fig14.suptitle('Asset Returns (Daily Log-Returns)', fontweight='bold')
plt.tight_layout()
fig14.savefig(output_dir / '14_asset_returns.png', dpi=300, bbox_inches='tight')
plt.close(fig14)

print("  Generating plots (15/15)...")

# PLOT 15: Fisher Information history (if available)
if best_scaling != 'identity':
    try:
        param_names = [f'h[{i}]' for i in range(len(best_h[0]))]
        fig15 = plot_fisher_history(best_res['model'], dates, param_names=param_names)
        if fig15 is not None:
            fig15.savefig(output_dir / '15_fisher_history.png', dpi=300, bbox_inches='tight')
            plt.close(fig15)
    except:
        print("  [!] Fisher history plot not available")


# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

print(f"\nOutput directory: {output_dir}")
print("\nGenerated Files:")
print("  Tables (6):")
print("    1. descriptive_statistics.csv")
print("    2. unconditional_correlation.csv")
print("    3. higher_moments.csv")
print("    4. model_comparison.csv")
print("    5. diagnostic_summary.csv")
print("    6. var_backtest_results.csv")
print("    7. var_statistical_tests.csv")
print("    8. crisis_period_analysis.csv")
print("    9. correlation_regime_statistics.csv")

print("\n  Visualizations (15):")
print("    01. volatility_paths.png")
print("    02. all_correlations.png")
print("    03. key_correlations.png")
print("    04. correlation_heatmaps.png")
print("    05. var_backtest.png")
print("    06. portfolio_volatility.png")
print("    07. condition_number.png")
print("    08. min_eigenvalue.png")
print("    09. logdet.png")
print("    10. score_norms.png")
print("    11. correlation_distribution.png")
print("    12. standardized_residuals.png")
print("    13. qqplots.png")
print("    14. asset_returns.png")
print("    15. fisher_history.png (if available)")

print("\n" + "="*80)
print("KEY FINDINGS:")
print("="*80)
print(f"  Selected Model: {best_scaling.upper()}")
print(f"  Degrees of Freedom: {best_params['nu']:.2f}")
print(f"  VaR 99% Coverage: {(1-hits_1.mean())*100:.2f}%")
print(f"  Max Condition Number: {df_comparison.iloc[0]['Cond_Max']:.2f}")
print(f"  COVID Impact: Vol increased by {df_crisis[df_crisis['Period']=='COVID Crash (Q1 2020)']['Vol_Mean'].values[0] / df_crisis[df_crisis['Period']=='Pre-COVID (2019)']['Vol_Mean'].values[0]:.2f}x")
print("="*80)

print("\n✓ Real-world application analysis complete!")
print(f"✓ All outputs saved to: {output_dir}")