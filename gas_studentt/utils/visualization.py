import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Sequence, Dict, Any
from scipy import stats
import itertools

from .diagnostics import (
    dynamic_correlation,
    condition_number_series,
    min_eigenvalue_series,
    score_norm_series,
    logdet_series,
)


# 1. STANDARDIZED RESIDUALS AND Q-Q PLOTS

def plot_standardized_residuals(
        Z: np.ndarray,
        title_prefix: str = "Standardized Residuals",
        figsize=(12, 6)
):
    """Plots the time series of standardized residuals for each dimension."""
    Z = np.asarray(Z)
    T, d = Z.shape

    fig, axes = plt.subplots(d, 1, figsize=figsize, sharex=True)
    if d == 1:
        axes = [axes]

    for i in range(d):
        axes[i].plot(Z[:, i], lw=0.8)
        axes[i].axhline(0.0, ls="--", alpha=0.6)
        axes[i].set_title(f"{title_prefix} - Dimension {i + 1}")
        axes[i].grid(alpha=0.3)

    plt.tight_layout()
    return fig


def qqplot_residuals(
        Z: np.ndarray,
        nu: float,
        figsize=(12, 5)
):
    """
    Generates Q-Q plots of the standardized residuals against the theoretical
    Student-t distribution. Dynamically adapts the layout based on dimensions.
    """
    Z = np.asarray(Z)
    T, d = Z.shape

    fig, axes = plt.subplots(1, d, figsize=figsize)
    if d == 1:
        axes = [axes]

    for i in range(d):
        stats.probplot(
            Z[:, i],
            dist=stats.t,
            sparams=(nu,),
            plot=axes[i]
        )
        axes[i].set_title(f"Q-Q Plot Dim {i + 1} vs t({nu:.2f})")

    plt.tight_layout()
    return fig


# 2. PROBABILITY INTEGRAL TRANSFORM (PIT)

def plot_pit_histogram(
        pit: np.ndarray,
        bins: int = 20,
        ks_test: bool = True,
        figsize=(6, 4)
):
    """
    Plots the Probability Integral Transform (PIT) histogram and compares it
    against a theoretical Uniform(0,1) distribution to assess model fit.
    """
    pit = np.asarray(pit)

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(pit, bins=bins, density=True, alpha=0.7, edgecolor="k")

    ax.axhline(1.0, color="red", ls="--", label="Uniform(0,1)")

    title = "PIT Histogram"
    if ks_test:
        ks_p = stats.kstest(pit, "uniform").pvalue
        title += f" (KS p={ks_p:.3f})"

    ax.set_title(title)
    ax.set_xlabel("PIT Values")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


# 3. NUMERICAL STABILITY (CONDITION NUMBER & EIGENVALUES)

def plot_condition_number(
        Sigmas: np.ndarray,
        threshold: Optional[float] = None,
        figsize=(10, 4)
):
    """Plots the time series of the covariance matrix condition number."""
    conds = condition_number_series(Sigmas)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(conds, lw=1.2, label="cond($\Sigma_t$)", color="purple")

    if threshold is not None:
        ax.axhline(threshold, color="red", ls="--", label="Instability threshold")

    ax.set_yscale("log")
    ax.set_title("Condition Number Path")
    ax.set_xlabel("Time")
    ax.set_ylabel("cond($\Sigma$)")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def plot_min_eigenvalue(
        Sigmas: np.ndarray,
        figsize=(10, 4)
):
    """Plots the time series of the minimum eigenvalue of the covariance matrices."""
    mins = min_eigenvalue_series(Sigmas)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(mins, lw=1.2, color="green")
    ax.axhline(0.0, ls="--", color="red")

    ax.set_title("Minimum Eigenvalue Path")
    ax.set_xlabel("Time")
    ax.set_ylabel("Min Eigenvalue")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


# 4. DYNAMIC CORRELATIONS

def plot_dynamic_correlation(
        Sigmas: np.ndarray,
        dates,
        bands: bool = True,
        figsize=(10, 4)
):
    """
    Plots the dynamic conditional correlations.
    If d=2, plots the single correlation pair.
    If d>2, creates subplots for all unique asset combinations.
    """
    Sigmas = np.asarray(Sigmas, float)
    d = Sigmas.shape[1]

    if d == 2:
        pairs = [(0, 1)]
        rows, cols = 1, 1
    else:
        pairs = list(itertools.combinations(range(d), 2))
        rows = len(pairs)
        cols = 1
        figsize = (figsize[0], 3 * rows)

    fig, axes = plt.subplots(rows, cols, figsize=figsize, sharex=True)
    if rows == 1:
        axes = [axes]

    for idx, (i, j) in enumerate(pairs):
        ax = axes[idx]

        rho = dynamic_correlation(Sigmas, i, j)
        mu = rho.mean()
        sd = rho.std()

        ax.plot(dates, rho, lw=1.2, color="crimson", label=f"Corr({i},{j})")
        ax.axhline(mu, ls="--", color="black", alpha=0.7, label=f"Mean={mu:.2f}")

        if bands:
            ax.fill_between(
                dates,
                mu - sd,
                mu + sd,
                color="crimson",
                alpha=0.1,
                label="±1 SD"
            )

        ax.set_ylim(-1.05, 1.05)
        ax.set_title(f"Dynamic Correlation: Asset {i} vs Asset {j}")
        ax.legend(loc="upper right", fontsize="small")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def plot_correlation_distribution(
        Sigmas: np.ndarray,
        bins: int = 30,
        figsize=(6, 4)
):
    """
    Plots the empirical distribution of the dynamic correlation.
    Defaults to showing the distribution for the (0, 1) asset pair.
    """
    rho = dynamic_correlation(Sigmas, 0, 1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(rho, bins=bins, alpha=0.7, edgecolor="k", density=True)

    ax.axvline(rho.mean(), color="red", ls="--", label=f"Mean = {rho.mean():.3f}")
    ax.axvline(np.median(rho), color="green", ls="--", label=f"Median = {np.median(rho):.3f}")

    ax.set_title("Correlation Distribution (Asset 0 vs 1)")
    ax.set_xlabel("Correlation")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


# 5. DYNAMIC VARIANCES / VOLATILITY

def plot_variances(
        Sigmas: np.ndarray,
        labels: Optional[Sequence[str]] = None,
        figsize=(12, 4)
):
    """Plots the dynamic conditional variances $\Sigma_{ii}(t)$."""
    Sigmas = np.asarray(Sigmas)
    d = Sigmas.shape[1]

    fig, ax = plt.subplots(figsize=figsize)

    for i in range(d):
        lbl = labels[i] if labels else f"Var {i}"
        ax.plot(Sigmas[:, i, i], lw=1.2, label=lbl)

    ax.set_title("Dynamic Variances")
    ax.set_xlabel("Time")
    ax.set_ylabel("Variance")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def plot_volatility(
        Sigmas: np.ndarray,
        dates,
        tickers: Sequence[str],
        figsize=(12, 6)
):
    """
    Plots the conditional standard deviations (volatility).
    Often more interpretable than variance for financial asset returns.
    """
    Sigmas = np.asarray(Sigmas)
    T, d, _ = Sigmas.shape
    vols = np.sqrt(np.diagonal(Sigmas, axis1=1, axis2=2))

    fig, ax = plt.subplots(figsize=figsize)
    for i in range(d):
        label = tickers[i] if i < len(tickers) else f"Asset {i}"
        ax.plot(dates, vols[:, i], label=label, alpha=0.8, lw=1.2)

    ax.set_title("Conditional Volatility (Std Dev)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylabel("$\sigma_t$ (Std Dev)")

    plt.tight_layout()
    return fig


# 6. SCORE AND SYSTEM ENTROPY

def plot_score_norm(
        scores: np.ndarray,
        figsize=(10, 4)
):
    """Plots the time series of the score vector norm $||s_t||$."""
    norms = score_norm_series(scores)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(norms, lw=1.2, color="orange")
    ax.set_title("Score Norm Path")
    ax.set_xlabel("Time")
    ax.set_ylabel("$||score||$")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def plot_logdet(
        Sigmas: np.ndarray,
        figsize=(10, 4)
):
    """Plots the time series of the covariance matrix log-determinant (generalized variance)."""
    vals = logdet_series(Sigmas)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(vals, lw=1.2, color="navy")
    ax.set_title("log det($\Sigma_t$)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Log-Determinant")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


# 7. RISK MANAGEMENT (VaR)

def plot_var_backtest(
        y: np.ndarray,
        var: np.ndarray,
        alpha: float = 0.01,
        figsize=(10, 4)
):
    """
    Univariate Value-at-Risk (VaR) backtest plot.
    Highlights periods where actual returns breach the conditional VaR threshold.
    """
    y = np.asarray(y)
    var = np.asarray(var)

    violations = y < var
    n_viol = violations.sum()

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(y, lw=0.8, alpha=0.6, label="Returns")
    ax.plot(var, lw=1.5, ls="--", color="red", label=f"VaR {int(alpha * 100)}%")

    ax.scatter(
        np.where(violations)[0],
        y[violations],
        color="black",
        zorder=5,
        s=15,
        label=f"Violations ({n_viol})"
    )

    ax.set_title(f"VaR Backtest (Violations: {n_viol})")
    ax.set_xlabel("Time")
    ax.set_ylabel("Returns")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


# 8. FISHER INFORMATION DYNAMICS

def plot_fisher_history(
        model: Any,
        dates,
        param_names: Optional[Sequence[str]] = None
):
    """
    Reconstructs and plots the history of the Fisher Information $\mathcal{I}(h_t)$
    by interpolating the model's pre-computed grid. Essential for validating
    dynamic scaling mechanisms.
    """
    if model.scaling not in ["fisher_diag", "sqrt_fisher_diag", "block_diag", "dynamic_fisher"]:
        print(f"Skipping Fisher plot: scaling '{model.scaling}' is static or identity.")
        return None

    try:
        h_t = model.last_h()
        grid_x = model.fisher_grid_x
        grid_val = model.fisher_grid_val

        if grid_x is None or grid_val is None:
            raise ValueError("Grid not found in model.")

    except Exception as e:
        print(f"Cannot plot Fisher history: {e}")
        return None

    T, m = h_t.shape

    fisher_hist = np.zeros_like(h_t)
    for t in range(T):
        for k in range(m):
            fisher_hist[t, k] = np.interp(h_t[t, k], grid_x, grid_val[k])

    cols = 2
    rows = int(np.ceil(m / cols))

    fig, axs = plt.subplots(rows, cols, figsize=(15, 3 * rows), sharex=True)
    if m > 1:
        axs = axs.ravel()
    else:
        axs = [axs]

    for i in range(m):
        title = param_names[i] if param_names and i < len(param_names) else f"Param {i}"

        ln1 = axs[i].plot(dates, fisher_hist[:, i], color='darkred', lw=1.5, label='Fisher Info $\mathcal{I}_t$')
        axs[i].set_ylabel("Information", color='darkred')
        axs[i].tick_params(axis='y', labelcolor='darkred')
        axs[i].grid(True, alpha=0.3)

        ax2 = axs[i].twinx()
        ln2 = ax2.plot(dates, h_t[:, i], color='navy', lw=1.0, alpha=0.5, label='Param $h_t$')
        ax2.set_ylabel("Param Value", color='navy')
        ax2.tick_params(axis='y', labelcolor='navy')

        axs[i].set_title(f"Fisher Dynamics: {title}")

        lns = ln1 + ln2
        labs = [l.get_label() for l in lns]
        axs[i].legend(lns, labs, loc='upper left', fontsize='small')

    plt.suptitle(f"Time-Varying Fisher Information ({model.scaling})", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig