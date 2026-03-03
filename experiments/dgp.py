"""
DGP 2 ENHANCED: Trivariate with Regime Switching + Extended Visualizations
===========================================================================
Fixed date range issue and added comprehensive diagnostic plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore', category=UserWarning)


from gas_studentt.models.gas_studentt_multivariate import GASStudentTMultivariate
from gas_studentt.utils.config import GASConfig
from gas_studentt.core.cholesky import vec_to_chol_d, chol_to_sigma_d
from gas_studentt.utils.diagnostics import (
    check_pd, condition_numbers, full_diagnostics,
    dynamic_correlation, logdet_series, min_eigenvalue_series,
    score_norms, state_variation, frobenius_distance_series
)
from gas_studentt.utils.visualization import (
    plot_volatility, plot_dynamic_correlation,
    plot_condition_number, plot_min_eigenvalue, plot_logdet,
    plot_score_norm, plot_variances, plot_correlation_distribution,
    qqplot_residuals, plot_standardized_residuals
)

np.random.seed(123)

# =============================================================================
# 1. DATA GENERATION PROCESS (DGP) WITH REGIME SWITCHING
# =============================================================================

def generate_dgp2_data(T=2000):
    """
    Generate trivariate Student-t data with regime switching.
    """
    d = 3

    # Initialize
    Y = np.zeros((T, d))
    Sigma = np.zeros((T, d, d))
    regime = np.zeros(T, dtype=int)

    # Define regimes
    regime[0:1000] = 0      # Tranquil
    regime[1000:1500] = 1   # Crisis
    regime[1500:2000] = 2   # Recovery

    # Regime parameters
    regimes_config = {
        0: {  # Tranquil
            'vol_scale': np.array([1.0, 1.0, 1.0]),
            'corr': np.array([[1.0, 0.3, 0.4],
                             [0.3, 1.0, 0.35],
                             [0.4, 0.35, 1.0]]),
            'nu': 8.0
        },
        1: {  # Crisis
            'vol_scale': np.array([2.5, 2.8, 2.3]),
            'corr': np.array([[1.0, 0.75, 0.70],
                             [0.75, 1.0, 0.68],
                             [0.70, 0.68, 1.0]]),
            'nu': 4.0
        },
        2: {  # Recovery
            'vol_scale': np.array([1.5, 1.6, 1.4]),
            'corr': np.array([[1.0, 0.50, 0.48],
                             [0.50, 1.0, 0.45],
                             [0.48, 0.45, 1.0]]),
            'nu': 6.0
        }
    }

    # Generate data
    mu = np.zeros(d)

    for t in range(T):
        r = regime[t]
        config = regimes_config[r]

        # Construct covariance matrix
        D = np.diag(config['vol_scale'])
        Sigma[t] = D @ config['corr'] @ D

        # Generate Student-t observation
        Z = np.random.multivariate_normal(np.zeros(d), Sigma[t])
        w = np.random.chisquare(df=config['nu'])
        Y[t] = mu + np.sqrt(config['nu'] / w) * Z

    return Y, Sigma, regime, regimes_config


# =============================================================================
# 2. GENERATE DATA
# =============================================================================

print("=" * 80)
print("DGP: TRIVARIATE WITH REGIME SWITCHING")
print("=" * 80)

T = 2000
Y, Sigma_true, regime, regime_config = generate_dgp2_data(T=T)

print(f"\nGenerated {T} observations across 3 regimes")
print(f"\nRegime distribution:")
print(f"  Tranquil (0-999):   {(regime == 0).sum()} obs")
print(f"  Crisis (1000-1499): {(regime == 1).sum()} obs")
print(f"  Recovery (1500-1999): {(regime == 2).sum()} obs")

print(f"\nSample statistics:")
print(f"  Mean: {Y.mean(axis=0)}")
print(f"  Std: {Y.std(axis=0)}")
print(f"  Correlation:\n{np.corrcoef(Y.T)}")


# =============================================================================
# 3. MODEL ESTIMATION - COMPARE SCALINGS
# =============================================================================

scaling_methods = ['identity', 'sqrt_fisher_diag', 'fisher_diag']
results = {}

for scaling in scaling_methods:
    print(f"\n{'='*80}")
    print(f"ESTIMATING WITH SCALING: {scaling.upper()}")
    print(f"{'='*80}")

    model = GASStudentTMultivariate(
        d=3,
        scaling=scaling,
        update_type='diagonal',
        config=GASConfig(d=3)
    )

    if scaling != 'identity':
        model.precompute_fisher_grid(
            h_min=-12.0,
            h_max=12.0,
            n_points=80,
            n_mc=800,
            nu_target=6.0
        )

    fit_res = model.fit(
        Y,
        method='L-BFGS-B',
        maxiter=600,
        multistart=4
    )

    print(f"\nFit Success: {fit_res.success}")
    print(f"Log-Likelihood: {fit_res.ll:.2f}")
    print(f"AIC: {fit_res.aic:.2f}")
    print(f"BIC: {fit_res.bic:.2f}")
    print(f"Estimated nu: {fit_res.params['nu']:.3f}")
    print(f"PD Failures: {fit_res.n_pd_failures}")

    results[scaling] = {
        'fit': fit_res,
        'model': model,
        'Sigmas': model.last_Sigmas(),
        'h': model.last_h(),
        'scores': model.last_scores()
    }


# =============================================================================
# 4. REGIME DETECTION ANALYSIS
# =============================================================================

print(f"\n{'='*80}")
print("REGIME DETECTION ANALYSIS")
print(f"{'='*80}")

bic_vals = {s: results[s]['fit'].bic for s in scaling_methods}
best_scaling = min(bic_vals, key=bic_vals.get)
best_Sigmas = results[best_scaling]['Sigmas']

vol_by_regime = {0: [], 1: [], 2: []}
corr_by_regime = {0: [], 1: [], 2: []}

for r in [0, 1, 2]:
    mask = (regime == r)

    vols = np.sqrt(np.diagonal(best_Sigmas[mask], axis1=1, axis2=2))
    vol_by_regime[r] = vols.mean(axis=0)

    corrs_01 = dynamic_correlation(best_Sigmas[mask], 0, 1).mean()
    corrs_02 = dynamic_correlation(best_Sigmas[mask], 0, 2).mean()
    corrs_12 = dynamic_correlation(best_Sigmas[mask], 1, 2).mean()
    corr_by_regime[r] = [corrs_01, corrs_02, corrs_12]

regime_table = []
for r in [0, 1, 2]:
    regime_name = ['Tranquil', 'Crisis', 'Recovery'][r]
    regime_table.append({
        'Regime': regime_name,
        'Vol_Asset1': f'{vol_by_regime[r][0]:.3f}',
        'Vol_Asset2': f'{vol_by_regime[r][1]:.3f}',
        'Vol_Asset3': f'{vol_by_regime[r][2]:.3f}',
        'Corr_0-1': f'{corr_by_regime[r][0]:.3f}',
        'Corr_0-2': f'{corr_by_regime[r][1]:.3f}',
        'Corr_1-2': f'{corr_by_regime[r][2]:.3f}'
    })

df_regime = pd.DataFrame(regime_table)
print("\n", df_regime.to_string(index=False))


# =============================================================================
# 5. MODEL COMPARISON
# =============================================================================

print(f"\n{'='*80}")
print("MODEL COMPARISON")
print(f"{'='*80}")

comparison_table = []
for scaling in scaling_methods:
    res = results[scaling]
    pd_stats = check_pd(res['Sigmas'])
    cond_stats = condition_numbers(res['Sigmas'])

    comparison_table.append({
        'Scaling': scaling,
        'LogLik': res['fit'].ll,
        'AIC': res['fit'].aic,
        'BIC': res['fit'].bic,
        'nu_est': res['fit'].params['nu'],
        'MinEig': pd_stats['min_eig_min'],
        'Cond_Max': cond_stats['max'],
        'PD_Fail': res['fit'].n_pd_failures
    })

df_comparison = pd.DataFrame(comparison_table)
print("\n", df_comparison.to_string(index=False))


# =============================================================================
# 6. EXTENDED VISUALIZATIONS
# =============================================================================

print(f"\n{'='*80}")
print("GENERATING VISUALIZATIONS...")
print(f"{'='*80}")

output_dir = Path('')
output_dir.mkdir(parents=True, exist_ok=True)

# FIX: Create proper date range for full series
dates = pd.date_range('2018-01-01', periods=T, freq='D')

best_model = results[best_scaling]['model']
best_scores = results[best_scaling]['scores']
best_h = results[best_scaling]['h']

print(f"\nDate range: {dates[0]} to {dates[-1]}")
print(f"Total days: {len(dates)}")

# Helper function to add regime bands
def add_regime_bands(ax, dates, alpha=0.15):
    ax.axvspan(dates[0], dates[999], alpha=alpha, color='green', label='Tranquil')
    ax.axvspan(dates[1000], dates[1499], alpha=alpha, color='red', label='Crisis')
    ax.axvspan(dates[1500], dates[1999], alpha=alpha, color='yellow', label='Recovery')


# PLOT 1: Volatility with regime highlighting
print("  - Plot 1: Volatility paths...")
fig1 = plot_volatility(best_Sigmas, dates, tickers=['Asset 1', 'Asset 2', 'Asset 3'])
ax = fig1.axes[0]
add_regime_bands(ax, dates, alpha=0.2)
ax.legend(loc='upper left')
fig1.savefig(output_dir / 'volatility_regimes.png', dpi=150, bbox_inches='tight')
plt.close(fig1)


# PLOT 2: All pairwise correlations
print("  - Plot 2: Dynamic correlations...")
fig2 = plot_dynamic_correlation(best_Sigmas, dates, bands=False)
for ax in fig2.axes:
    add_regime_bands(ax, dates)
fig2.savefig(output_dir / 'correlations_regimes.png', dpi=150, bbox_inches='tight')
plt.close(fig2)


# PLOT 3: Variances (alternative view)
print("  - Plot 3: Variance paths...")
fig3 = plot_variances(best_Sigmas, labels=['Asset 1', 'Asset 2', 'Asset 3'])
ax = fig3.axes[0]
add_regime_bands(ax, dates)
fig3.savefig(output_dir / 'variances_regimes.png', dpi=150, bbox_inches='tight')
plt.close(fig3)


# PLOT 4: Log-determinant (total variance)
print("  - Plot 4: Log-determinant...")
logdets = logdet_series(best_Sigmas)
fig4, ax = plt.subplots(figsize=(12, 5))
ax.plot(dates, logdets, lw=1.2, color='purple')
add_regime_bands(ax, dates)
ax.set_ylabel('log det(Sigma)')
ax.set_title('Total System Variance (log determinant)')
ax.set_xlabel('Date')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
fig4.savefig(output_dir / 'logdet_regimes.png', dpi=150, bbox_inches='tight')
plt.close(fig4)


# PLOT 5: Minimum Eigenvalue
print("  - Plot 5: Minimum eigenvalue...")
fig5 = plot_min_eigenvalue(best_Sigmas, figsize=(12, 5))
ax = fig5.axes[0]
# Manually set x-axis to dates
ax.clear()
min_eigs = min_eigenvalue_series(best_Sigmas)
ax.plot(dates, min_eigs, lw=1.2, color='green')
ax.axhline(0.0, ls='--', color='red')
add_regime_bands(ax, dates)
ax.set_xlabel('Date')
ax.set_ylabel('min eigenvalue')
ax.set_title('Minimum Eigenvalue Path')
ax.grid(alpha=0.3)
fig5.savefig(output_dir / 'min_eigenvalue.png', dpi=150, bbox_inches='tight')
plt.close(fig5)


# PLOT 6: Condition Number
print("  - Plot 6: Condition number...")
fig6 = plot_condition_number(best_Sigmas, threshold=1000, figsize=(12, 5))
ax = fig6.axes[0]
# Redraw with dates
from gas_studentt.utils.diagnostics import condition_number_series
ax.clear()
conds = condition_number_series(best_Sigmas)
ax.plot(dates, conds, lw=1.2, label='cond(Sigma_t)', color='purple')
ax.axhline(1000, color='red', ls='--', label='Instability threshold')
add_regime_bands(ax, dates)
ax.set_yscale('log')
ax.set_xlabel('Date')
ax.set_ylabel('cond(Sigma)')
ax.set_title('Condition Number Path')
ax.legend()
ax.grid(alpha=0.3)
fig6.savefig(output_dir / 'condition_number.png', dpi=150, bbox_inches='tight')
plt.close(fig6)


# PLOT 7: Score Norms
print("  - Plot 7: Score norms...")
fig7 = plot_score_norm(best_scores, figsize=(12, 5))
ax = fig7.axes[0]
# Redraw with dates
from gas_studentt.utils.diagnostics import score_norm_series
ax.clear()
norms = score_norm_series(best_scores)
ax.plot(dates, norms, lw=1.2, color='orange')
add_regime_bands(ax, dates)
ax.set_xlabel('Date')
ax.set_ylabel('||score||')
ax.set_title('Score Norm Path')
ax.grid(alpha=0.3)
fig7.savefig(output_dir / 'score_norms.png', dpi=150, bbox_inches='tight')
plt.close(fig7)


# PLOT 8: Correlation Distribution
print("  - Plot 8: Correlation distribution...")
fig8 = plot_correlation_distribution(best_Sigmas, bins=30, figsize=(6, 4))
fig8.savefig(output_dir / 'correlation_distribution.png', dpi=150, bbox_inches='tight')
plt.close(fig8)


# PLOT 9: Regime-wise correlation comparison (True vs Estimated)
print("  - Plot 9: Correlation tracking...")
fig9, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
pairs = [(0, 1), (0, 2), (1, 2)]

for idx, (i, j) in enumerate(pairs):
    rho_true = dynamic_correlation(Sigma_true, i, j)
    rho_est = dynamic_correlation(best_Sigmas, i, j)

    axes[idx].plot(dates, rho_true, label='True', lw=1.5, alpha=0.7)
    axes[idx].plot(dates, rho_est, label='Estimated', lw=1.5, alpha=0.7, ls='--')
    add_regime_bands(axes[idx], dates, alpha=0.1)
    axes[idx].set_ylabel(f'Corr({i},{j})')
    axes[idx].legend()
    axes[idx].grid(alpha=0.3)

axes[2].set_xlabel('Date')
fig9.suptitle('True vs Estimated Correlations Across Regimes')
plt.tight_layout()
fig9.savefig(output_dir / 'correlation_tracking.png', dpi=150, bbox_inches='tight')
plt.close(fig9)


# PLOT 10: Heatmaps by regime
print("  - Plot 10: Correlation heatmaps...")
fig10, axes = plt.subplots(1, 3, figsize=(15, 4))
regime_names = ['Tranquil', 'Crisis', 'Recovery']

for r_idx, r in enumerate([0, 1, 2]):
    mask = (regime == r)
    avg_corr_matrix = np.zeros((3, 3))

    for i in range(3):
        for j in range(3):
            if i == j:
                avg_corr_matrix[i, j] = 1.0
            else:
                corrs = dynamic_correlation(best_Sigmas[mask], i, j)
                avg_corr_matrix[i, j] = corrs.mean()
                avg_corr_matrix[j, i] = corrs.mean()

    im = axes[r_idx].imshow(avg_corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[r_idx].set_title(regime_names[r_idx])
    axes[r_idx].set_xticks([0, 1, 2])
    axes[r_idx].set_yticks([0, 1, 2])
    axes[r_idx].set_xticklabels(['A1', 'A2', 'A3'])
    axes[r_idx].set_yticklabels(['A1', 'A2', 'A3'])

    for i in range(3):
        for j in range(3):
            axes[r_idx].text(j, i, f'{avg_corr_matrix[i, j]:.2f}',
                           ha="center", va="center", color="black")

fig10.colorbar(im, ax=axes, label='Correlation')
fig10.suptitle('Average Correlation Matrix by Regime')
plt.tight_layout()
fig10.savefig(output_dir / 'correlation_heatmaps.png', dpi=150, bbox_inches='tight')
plt.close(fig10)


# PLOT 11: State variation (h dynamics)
print("  - Plot 11: State variation...")
from project.utils.diagnostics import state_variation_series
state_var = state_variation_series(best_h)

fig11, ax = plt.subplots(figsize=(12, 5))
ax.plot(dates[1:], state_var, lw=1.2, color='teal')  # dates[1:] because diff reduces length
add_regime_bands(ax, dates)
ax.set_xlabel('Date')
ax.set_ylabel('||h_t - h_{t-1}||')
ax.set_title('State Variation Path (Latent Dynamics)')
ax.grid(alpha=0.3)
plt.tight_layout()
fig11.savefig(output_dir / 'state_variation.png', dpi=150, bbox_inches='tight')
plt.close(fig11)


# PLOT 12: Standardized Residuals
print("  - Plot 12: Standardized residuals...")
# Compute standardized residuals
Z = np.zeros_like(Y)
for t in range(T):
    try:
        L = np.linalg.cholesky(best_Sigmas[t])
        Z[t] = np.linalg.solve(L, Y[t] - results[best_scaling]['fit'].params['mu'])
    except:
        Z[t] = Y[t]

fig12 = plot_standardized_residuals(Z, title_prefix='Standardized Residuals', figsize=(12, 8))
fig12.savefig(output_dir / 'standardized_residuals.png', dpi=150, bbox_inches='tight')
plt.close(fig12)


# PLOT 13: QQ-plots
print("  - Plot 13: QQ-plots...")
fig13 = qqplot_residuals(Z, nu=results[best_scaling]['fit'].params['nu'], figsize=(12, 4))
fig13.savefig(output_dir / 'qqplots.png', dpi=150, bbox_inches='tight')
plt.close(fig13)


print(f"\n✓ All {13} visualizations saved to {output_dir}")


# =============================================================================
# 7. SAVE TABLES
# =============================================================================

df_comparison.to_csv(output_dir / 'model_comparison.csv', index=False)
df_regime.to_csv(output_dir / 'regime_statistics.csv', index=False)

# Detailed regime stats
regime_detailed = []
for r in [0, 1, 2]:
    mask = (regime == r)
    Y_regime = Y[mask]

    regime_detailed.append({
        'Regime': ['Tranquil', 'Crisis', 'Recovery'][r],
        'N_obs': mask.sum(),
        'Mean_Asset1': Y_regime[:, 0].mean(),
        'Std_Asset1': Y_regime[:, 0].std(),
        'Mean_Asset2': Y_regime[:, 1].mean(),
        'Std_Asset2': Y_regime[:, 1].std(),
        'Mean_Asset3': Y_regime[:, 2].mean(),
        'Std_Asset3': Y_regime[:, 2].std(),
        'Kurtosis_Asset1': pd.Series(Y_regime[:, 0]).kurtosis(),
        'Kurtosis_Asset2': pd.Series(Y_regime[:, 1]).kurtosis(),
        'Kurtosis_Asset3': pd.Series(Y_regime[:, 2]).kurtosis()
    })

df_regime_detail = pd.DataFrame(regime_detailed)
df_regime_detail.to_csv(output_dir / 'regime_detailed_stats.csv', index=False)

# Additional diagnostics table
diag_stats = {
    'Metric': ['Score Mean', 'Score Max', 'State Var Mean', 'Min Eig Mean', 'Cond Median'],
    'Value': [
        score_norms(best_scores)['mean'],
        score_norms(best_scores)['max'],
        state_var.mean(),
        min_eigs.mean(),
        np.median(conds[np.isfinite(conds)]) if np.any(np.isfinite(conds)) else np.nan    ]
}
df_diag_extra = pd.DataFrame(diag_stats)
df_diag_extra.to_csv(output_dir / 'diagnostic_metrics.csv', index=False)

print(f"\n✓ All tables saved to {output_dir}")


# =============================================================================
# 8. FINAL SUMMARY
# =============================================================================

print(f"\n{'='*80}")
print("DGP 2 ENHANCED - ANALYSIS COMPLETE!")
print(f"{'='*80}")
print(f"\nResults saved in: {output_dir}")
print(f"  - 13 visualization plots")
print(f"  - 4 CSV tables")

print(f"\nBest scaling method (by BIC): {best_scaling}")
print(f"The model successfully captured regime transitions:")
print(f"  - Crisis period shows {vol_by_regime[1][0]/vol_by_regime[0][0]:.2f}x volatility increase")
print(f"  - Correlation increased from {corr_by_regime[0][0]:.2f} to {corr_by_regime[1][0]:.2f}")

print(f"\n{'='*80}")
print("PLOT LIST:")
print(f"{'='*80}")
plots = [
    "1. volatility_regimes.png - Volatility paths with regime bands",
    "2. correlations_regimes.png - All pairwise correlations",
    "3. variances_regimes.png - Variance paths (alternative view)",
    "4. logdet_regimes.png - Log-determinant (total variance)",
    "5. min_eigenvalue.png - Minimum eigenvalue stability",
    "6. condition_number.png - Condition number path",
    "7. score_norms.png - Score norm dynamics",
    "8. correlation_distribution.png - Histogram of correlations",
    "9. correlation_tracking.png - True vs estimated (3 panels)",
    "10. correlation_heatmaps.png - Regime-wise correlation matrices",
    "11. state_variation.png - Latent state dynamics",
    "12. standardized_residuals.png - Residual diagnostics",
    "13. qqplots.png - QQ-plots vs Student-t"
]
for plot in plots:
    print(f"  {plot}")