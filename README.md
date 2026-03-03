# GAS-StudentT: Multivariate Time-Varying Covariance with Score-Driven Models

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Numba](https://img.shields.io/badge/powered%20by-Numba-orange.svg)](http://numba.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A robust, high-performance Python implementation of **Generalized Autoregressive Score (GAS)** models for time-varying covariance matrices with multivariate Student-t innovations. This library provides a complete framework for estimating, forecasting, and analyzing dynamic covariance structures in financial time series and other heavy-tailed data.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Theoretical Background](#theoretical-background)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
- [Software Architecture](#software-architecture)
- [Examples](#examples)
- [API Reference](#api-reference)
- [Performance Considerations](#performance-considerations)
- [Citation](#citation)
- [License](#license)

---

## Overview

The estimation of time-varying covariance matrices is a central challenge in financial econometrics, complicated by:

- **Heavy tails** — financial returns exhibit leptokurtosis (fat tails)
- **Positive definiteness** — covariance matrices must remain symmetric positive definite (SPD)
- **Numerical stability** — high-dimensional estimation requires robust algorithms

This library implements a **Cholesky-GAS-Student-t** framework that addresses these challenges through:

1. **Score-driven dynamics** — parameters update based on the score (gradient) of the log-likelihood, providing information-theoretic optimality
2. **Student-t innovations** — automatic down-weighting of outliers for robust estimation
3. **Cholesky parameterization** — guarantees SPD matrices without constrained optimization
4. **Dynamic Fisher scaling** — grid-based interpolation for efficient curvature adjustment
5. **JIT compilation** — Numba-accelerated core for C-like performance

---

## Key Features

| Feature | Description |
|---------|-------------|
| Flexible model specs | Scalar, diagonal, or full dynamics for A and B matrices |
| Multiple scaling strategies | Identity, Fisher Information, Square-Root Fisher |
| Numerical safeguards | PD repair, gradient clipping, jittering |
| Grid-based Fisher estimation | Pre-computed interpolation for dynamic scaling |
| Comprehensive diagnostics | PD checks, condition numbers, residual analysis |
| Multi-start optimization | Global optimization to avoid local minima |
| Forecasting | Multi-step ahead covariance and VaR predictions |
| Modular architecture | Clean separation of statistical specification and numerical core |

---

## Theoretical Background

### The GAS Framework

The Generalized Autoregressive Score (GAS) model links parameter updates directly to the shape of the predictive density. For a time-varying parameter vector $f_t$, the update equation is:

$$f_{t+1} = \omega + A s_t + B(f_t - \omega)$$

where $\omega$ is the long-run equilibrium, $s_t = S_t \nabla_t$ is the **scaled score**, $A$ controls sensitivity to new information, and $B$ controls persistence. This approach locally minimizes the Kullback-Leibler divergence between the true and model densities.

### Multivariate Student-t Distribution

For a $d$-dimensional observation $y_t \mid \mathcal{F}_{t-1} \sim t_\nu(\mu, \Sigma_t)$, the log-likelihood is:

$$\ell_t = \log\Gamma\!\left(\frac{\nu+d}{2}\right) - \log\Gamma\!\left(\frac{\nu}{2}\right) - \frac{d}{2}\log(\nu\pi) - \frac{1}{2}\log|\Sigma_t| - \frac{\nu+d}{2}\log\!\left(1 + \frac{q_t}{\nu}\right)$$

where $q_t = (y_t - \mu)^\top \Sigma_t^{-1}(y_t - \mu)$ is the squared Mahalanobis distance.

The adaptive weight $w_t = \frac{\nu+d}{\nu+q_t}$ automatically down-weights outliers — large $q_t$ produces small $w_t$.

### Cholesky Parameterization

We model the Cholesky factor $L_t$ where $\Sigma_t = L_t L_t^\top$. The unconstrained vector $h_t \in \mathbb{R}^m$ (with $m = d(d+1)/2$) maps to $L_t$ as:

- **Diagonal** entries: $L_{ii,t} = \exp(h_{k,t})$
- **Off-diagonal** entries: $L_{ij,t} = h_{k,t}$ for $i > j$
- **Upper triangle**: $L_{ij,t} = 0$ for $i < j$

This guarantees $\Sigma_t$ is SPD by construction — no constrained optimization needed.

### Score-Driven Dynamics

The raw score $\nabla_t = \partial \ell_t / \partial h_t$ is computed via chain rule:

**Step 1** — gradient w.r.t. $\Sigma_t$:

$$\nabla_{\Sigma_t}\ell_t = \frac{1}{2}\left(w_t\Sigma_t^{-1}x_t x_t^\top \Sigma_t^{-1} - \Sigma_t^{-1}\right)$$

**Step 2** — propagate through Cholesky:

$$\nabla_{L_t}\ell_t = 2(\nabla_{\Sigma_t}\ell_t)L_t$$

**Step 3** - Mapping to the Unconstrained Parameter Vector ($h_t$) The final score vector components are mapped based on their position in the lower-triangular matrix:

$$[\nabla_t]_k = \begin{cases} 
2 [(\nabla_{\Sigma_t} \ell_t) L_t]_{ii} L_{ii} & \text{if } i = j \text{ (Diagonal)} \\\\ 
2 [(\nabla_{\Sigma_t} \ell_t) L_t]_{ij} & \text{if } i > j \text{ (Off-diagonal)} 
\end{cases}$$

### Fisher Information and Dynamic Scaling

The Fisher Information Matrix 

$$\mathcal{I}_t = \mathbb{E}_{t-1}[\nabla_t \nabla_t^\top]$$ 

measures local curvature. Since analytical derivation is intractable, we use **grid-based interpolation**:

1. **Pre-computation** — estimate $\mathcal{I}_{kk}(h)$ on a grid $h \in [-12, 12]$ via Monte Carlo
2. **Runtime interpolation** — retrieve

$$\mathcal{I}_{kk}(h_{k,t})$$

via linear interpolation during filtering

3. **Scaling** — apply
$$s_t = \nabla_t / \sqrt{\mathcal{I}_t}$$
(Square-Root Fisher) for variance stabilization

---

## Installation

```bash
git clone https://github.com/yourusername/gas-studentt.git
cd gas-studentt
pip install -r requirements.txt
pip install -e .
```

**Requirements:** Python 3.8+, NumPy >= 1.20, SciPy >= 1.7, Numba >= 0.55, Matplotlib >= 3.4, Pandas >= 1.3 (optional)

---

## Quick Start

```python
import numpy as np
from gas_studentt.models import GASStudentTMultivariate

T, d = 1000, 3
np.random.seed(42)
Y = np.random.randn(T, d) * 0.01  # replace with actual returns

# Initialize and configure model
model = GASStudentTMultivariate(d=d, scaling="sqrt_fisher_diag", update_type="diagonal")
model.precompute_fisher_grid(h_min=-12.0, h_max=12.0, n_points=50, nu_target=10.0)

# Fit
result = model.fit(Y, multistart=3, method="L-BFGS-B")
print(f"Log-Likelihood: {result.ll:.4f}")
print(f"AIC: {result.aic:.2f} | BIC: {result.bic:.2f}")
print(f"Estimated degrees of freedom: {result.params['nu']:.2f}")

# Filtered states and forecasts
Sigma_t    = model.last_Sigmas()
forecast   = model.forecast(h_last=model.last_h()[-1], H=10)
Sigma_pred = forecast["Sigma_forecast"]
```

---

## Detailed Usage

### Model Configuration

```python
model = GASStudentTMultivariate(
    d=4,
    scaling="sqrt_fisher_diag",    # 'identity', 'fisher_diag', 'sqrt_fisher_diag'
    update_type="diagonal",        # 'scalar', 'diagonal', 'full'
    constraint_bounds=(0.0, 0.999)
)
```

**Scaling options:**

| Option | Formula | Notes |
|--------|---------|-------|
| `"identity"` | $s_t = \nabla_t$ | Baseline; may be unstable |
| `"fisher_diag"` | $s_t = \nabla_t / \mathcal{I}_t$ | Inverse Fisher |
| `"sqrt_fisher_diag"` | $s_t = \nabla_t / \sqrt{\mathcal{I}_t}$ | **Recommended** |

**Update types:**

| Option | Structure | Notes |
|--------|-----------|-------|
| `"scalar"` | $A = aI,\ B = bI$ | Single parameters |
| `"diagonal"` | $A = \text{diag}(a_1,\dots,a_m)$ | Per-parameter dynamics |
| `"full"` | Full matrices | Most flexible, most parameters |

### Pre-computing Fisher Grid

```python
model.precompute_fisher_grid(
    h_min=-12.0,
    h_max=12.0,
    n_points=50,
    nu_target=10.0,
    n_mc=1000,
    random_state=42
)
```

Generates a lookup table of shape `(m, n_points)` with the score variance for each parameter across the state space.

### Estimation (Fitting)

```python
result = model.fit(
    Y,
    multistart=5,
    method="L-BFGS-B",
    max_iter=800,
    tol=1e-6,
    verbose=True
)
```

The optimizer works in **unconstrained space** with automatic transformations: $A, B$ via logistic transform to $(0, 0.999)$; $\nu$ via softplus to $(2, \infty)$; $h_0$ clipped to $[-20, 20]$.

### Forecasting

```python
forecast       = model.forecast(h_last=model.last_h()[-1], H=10)
Sigma_forecast = forecast["Sigma_forecast"]  # shape (H, d, d)

weights       = np.ones(d) / d
portfolio_vol = np.sqrt([weights @ S @ weights for S in Sigma_forecast])
VaR           = stats.t.ppf(0.01, df=result.params['nu']) * portfolio_vol
```

### Diagnostics

```python
from gas_studentt.utils.diagnostics import full_diagnostics

diag = full_diagnostics(
    Sigma=model.last_Sigmas(),
    scores=model.last_scores(),
    h=model.last_h(),
    Y=Y,
    mu=result.params['mu']
)

print(f"PD failures:            {diag['pd']['n_violations']}")
print(f"Median condition number: {diag['conditioning']['median']:.2f}")
```

---

## Software Architecture

```
gas-studentt/
├── gas_studentt/
│   ├── models/
│   │   └── gas_studentt_multivariate.py  # Main model class
│   ├── core/
│   │   ├── gas_filter_numba.py           # JIT-compiled filter
│   │   ├── score.py                      # Analytical score computation
│   │   ├── student_t.py                  # Log-density and utilities
│   │   ├── cholesky.py                   # Cholesky transformations
│   │   ├── fisher.py                     # Fisher grid computation
│   │   └── scaling.py                    # Dynamic scaling
│   └── utils/
│       ├── constraints.py                # Parameter transformations
│       ├── diagnostics.py                # Validation tools
│       ├── config.py                     # Numerical hyperparameters
│       └── visualization.py              # Plotting utilities
├── experiments/
        ├──dgp.py
        ├──dgp_MC.py
        └── real_data.py
├── requirements.txt
└── setup.py
```

### Core Components

| Module | Description |
|--------|-------------|
| `models/gas_studentt_multivariate.py` | Orchestrates initialization, fitting, forecasting |
| `core/gas_filter_numba.py` | JIT-compiled GAS recursive filter |
| `core/score.py` | Analytical gradient via chain rule |
| `core/fisher.py` | Monte Carlo Fisher grid computation |
| `core/scaling.py` | Dynamic score scaling with interpolation |

### Numerical Safeguards

1. **PD repair** — jitter $\Sigma_t \leftarrow \Sigma_t + \delta I$ when ill-conditioned
2. **Double clipping** — raw and scaled scores clipped to $[-C, C]$
3. **Mahalanobis clipping** — $q_t \leftarrow \min(q_t,\, q_{\max})$ to prevent overflow
4. **Fisher floor** — avoid division by zero

$$\mathcal{I}_{ii} \leftarrow \max(\mathcal{I}_{ii}, \epsilon_{\min})$$ 

All thresholds are configurable via `utils/config.py`.

---

## Examples

### Example 1: Basic Bivariate Model

```python
import numpy as np
import matplotlib.pyplot as plt
from gas_studentt.models import GASStudentTMultivariate

T, d = 1500, 2
Y = np.zeros((T, d))
Y[:750] = np.random.multivariate_normal([0, 0], [[0.5, 0.2], [0.2, 0.5]], 750)
Y[750:] = np.random.multivariate_normal([0, 0], [[2.0, 1.5], [1.5, 2.0]], 750)
Y[500]  *= 5   # outlier
Y[1000] *= 5   # outlier

model = GASStudentTMultivariate(d=2, scaling="sqrt_fisher_diag")
model.precompute_fisher_grid()
result = model.fit(Y, multistart=3)

print(f"Estimated nu: {result.params['nu']:.2f}")

Sigma_t = model.last_Sigmas()
plt.plot(np.sqrt(Sigma_t[:, 0, 0]), label='Asset 1 Vol')
plt.plot(np.sqrt(Sigma_t[:, 1, 1]), label='Asset 2 Vol')
plt.axvline(x=750, color='r', linestyle='--', label='Regime Switch')
plt.legend()
plt.title('Estimated Conditional Volatilities')
plt.show()
```

### Example 2: Multi-Asset Portfolio with Diagnostics

```python
import numpy as np
from gas_studentt.models import GASStudentTMultivariate
from gas_studentt.utils.diagnostics import full_diagnostics

np.random.seed(42)
T, d = 1759, 4
Y = np.random.randn(T, d) * 0.01

results, models = {}, {}
for scaling in ["identity", "fisher_diag", "sqrt_fisher_diag"]:
    model  = GASStudentTMultivariate(d=d, scaling=scaling)
    model.precompute_fisher_grid()
    result = model.fit(Y, multistart=2)
    models[scaling]  = model
    results[scaling] = result
    print(f"{scaling}: LL={result.ll:.2f}, BIC={result.bic:.2f}")

best = min(results, key=lambda x: results[x].bic)
diag = full_diagnostics(
    Sigma=models[best].last_Sigmas(),
    scores=models[best].last_scores(),
    h=models[best].last_h(),
    Y=Y,
    mu=results[best].params['mu']
)
print(f"Best model: {best} | PD failures: {diag['pd']['n_violations']}")
```

### Example 3: Forecasting and VaR

```python
import numpy as np
from scipy import stats
from gas_studentt.models import GASStudentTMultivariate

T, d = 2000, 3
np.random.seed(42)
Y = np.random.randn(T, d) * 0.01
Y_train, Y_test = Y[:1500], Y[1500:]

model = GASStudentTMultivariate(d=d, scaling="sqrt_fisher_diag")
model.precompute_fisher_grid()
result = model.fit(Y_train, multistart=3)

forecast       = model.forecast(h_last=model.last_h()[-1], H=20)
Sigma_forecast = forecast["Sigma_forecast"]

weights       = np.ones(d) / d
nu            = result.params['nu']
portfolio_vol = np.sqrt([weights @ S @ weights for S in Sigma_forecast])
VaR_1pct      = stats.t.ppf(0.01, df=nu) * portfolio_vol

violations = (Y_test[:20] @ weights) < -VaR_1pct
print(f"1% VaR violation rate: {violations.mean():.2%}")
```

---

## API Reference

### `GASStudentTMultivariate`

```python
GASStudentTMultivariate(
    d: int,
    scaling: str = "sqrt_fisher_diag",
    update_type: str = "diagonal",
    constraint_bounds: tuple = (0.0, 0.999)
)
```

| Method | Description |
|--------|-------------|
| `precompute_fisher_grid(h_min, h_max, n_points, nu_target, n_mc, random_state)` | Pre-compute Fisher grid |
| `fit(Y, multistart, method, max_iter, tol, verbose)` | Estimate parameters via MLE |
| `forecast(h_last, H)` | Generate multi-step forecasts |
| `last_h()` | Filtered states $h_t$ |
| `last_Sigmas()` | Filtered covariance matrices $\Sigma_t$ |
| `last_scores()` | Raw scores $\nabla_t$ |
| `last_scaled_scores()` | Scaled scores $s_t$ |

### `FitResult`

| Attribute | Description |
|-----------|-------------|
| `params` | Dictionary of estimated parameters |
| `ll` | Log-likelihood value |
| `aic` | Akaike Information Criterion |
| `bic` | Bayesian Information Criterion |
| `converged` | Boolean convergence flag |
| `message` | Optimizer message |
| `nfev` | Number of function evaluations |

---

## Performance Considerations

- **Filtering complexity**: $O(T \cdot m \cdot d^3)$ where $m = d(d+1)/2$
- **Fisher grid** (one-time cost): $O(m \cdot K \cdot N_{MC} \cdot d^3)$ with $K \approx 50$, $N_{MC} \approx 1000$
- **Memory**: $O(T \cdot d^2)$ for stored covariance matrices

**Optimization tips:**
- Start with $d \leq 5$ for initial testing
- Use `"sqrt_fisher_diag"` for the best stability/performance trade-off
- For volatile data, increase `h_max` to 15–20; for better precision increase `n_points` to 80–100
- 3–5 multi-starts are usually sufficient; increase for difficult problems
- Adjust clipping thresholds in `utils/config.py` if encountering instability

---

## Citation

If you use this code in your research, please cite:

```bibtex
@mastersthesis{benincasafrancesco2026gas,
  title  = {Scaling matrices and score dynamics in multivariate GAS Student-t models},
  author = {Francesco Benincasa},
  year   = {2026},
  school = {University of Bologna}
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

This implementation builds on the theoretical framework from:

- Creal, D., Koopman, S.J., & Lucas, A. (2011). *A dynamic multivariate heavy-tailed model for time-varying volatilities and correlations.*
- Harvey, A.C. (2013). *Dynamic Models for Volatility and Heavy Tails.*
- Zheng, T., & Ye, S. (2024). *Cholesky GAS models for large time-varying covariance matrices.*

---

**Maintainer**: Francesco Benincasa
**Contributions welcome!** Please open an issue or pull request on GitHub.
