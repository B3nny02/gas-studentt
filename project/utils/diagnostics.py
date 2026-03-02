import numpy as np
from typing import Dict, Optional


# 1. POSITIVE DEFINITENESS (PD) AND CONDITION NUMBERS

def condition_numbers(Sigmas: np.ndarray) -> Dict[str, float]:
    """Summary statistics of the condition numbers for a sequence of covariance matrices."""
    Sigmas = np.asarray(Sigmas, float)
    if len(Sigmas) == 0:
        return {"min": 0.0, "median": 0.0, "mean": 0.0, "max": 0.0}

    conds = []
    for S in Sigmas:
        w = np.linalg.eigvalsh(S)
        wmin = max(w.min(), 1e-18)
        conds.append(w.max() / wmin)

    conds = np.asarray(conds)
    return {
        "min": float(conds.min()),
        "median": float(np.median(conds)),
        "mean": float(conds.mean()),
        "max": float(conds.max())
    }


def condition_number_series(Sigmas: np.ndarray) -> np.ndarray:
    """Time series of the condition number."""
    Sigmas = np.asarray(Sigmas, float)
    conds = np.empty(len(Sigmas))

    for t, S in enumerate(Sigmas):
        w = np.linalg.eigvalsh(S)
        conds[t] = w.max() / max(w.min(), 1e-18)

    return conds


def check_pd(Sigmas: np.ndarray) -> Dict[str, float]:
    """
    Checks the positive definiteness of a sequence of matrices.
    Note: If PD-repair (jittering) is active during filtering, negative
    eigenvalues may have been artificially corrected.
    """
    Sigmas = np.asarray(Sigmas, float)
    if len(Sigmas) == 0:
        return {"min_eig_min": 0.0, "n_violations": 0}

    min_eigs = np.array([np.linalg.eigvalsh(S).min() for S in Sigmas])
    n_viol = int(np.sum(min_eigs < 0))

    return {
        "min_eig_min": float(min_eigs.min()),
        "min_eig_median": float(np.median(min_eigs)),
        "min_eig_mean": float(np.mean(min_eigs)),
        "n_violations": n_viol,
        "pct_violations": float(100.0 * n_viol / len(min_eigs))
    }


def min_eigenvalue_series(Sigmas: np.ndarray) -> np.ndarray:
    """Time series of the minimum eigenvalue."""
    Sigmas = np.asarray(Sigmas, float)
    mins = np.empty(len(Sigmas))

    for t, S in enumerate(Sigmas):
        mins[t] = np.linalg.eigvalsh(S).min()

    return mins


# 2. CORRELATIONS AND ECONOMETRIC STRUCTURE

def dynamic_correlation(Sigmas: np.ndarray, i: int = 0, j: int = 1) -> np.ndarray:
    """Dynamic conditional correlation between asset i and asset j."""
    Sigmas = np.asarray(Sigmas, float)
    d = Sigmas.shape[1]

    if i >= d or j >= d:
        raise ValueError(f"Invalid indices ({i},{j}) for dimension d={d}")
    if i == j:
        return np.ones(Sigmas.shape[0])

    s_ii = Sigmas[:, i, i]
    s_jj = Sigmas[:, j, j]
    s_ij = Sigmas[:, i, j]

    denom = np.sqrt(np.maximum(s_ii * s_jj, 1e-18))
    return s_ij / denom


def correlation_matrix(Sigma: np.ndarray) -> np.ndarray:
    """Extracts the correlation matrix from a given covariance matrix."""
    Sigma = np.asarray(Sigma, float)
    D_inv = np.diag(1.0 / np.sqrt(np.maximum(np.diag(Sigma), 1e-18)))
    return D_inv @ Sigma @ D_inv


# 3. SCORE DYNAMICS AND GAS MECHANISM STABILITY

def score_norms(scores: np.ndarray) -> Dict[str, float]:
    """Summary statistics for the norms of the score vector $s_t = \nabla_{h_t} \ell_t$."""
    scores = np.asarray(scores, float)
    norms = np.linalg.norm(scores, axis=1)

    return {
        "min": float(norms.min()),
        "median": float(np.median(norms)),
        "mean": float(np.mean(norms)),
        "p95": float(np.percentile(norms, 95)),
        "max": float(norms.max())
    }


def score_norm_series(scores: np.ndarray) -> np.ndarray:
    """Time series of the score norm $||s_t||$."""
    scores = np.asarray(scores, float)
    return np.linalg.norm(scores, axis=1)


def score_autocorrelation(scores: np.ndarray, lag: int = 1) -> float:
    """Average autocorrelation of the score components at a specified lag."""
    scores = np.asarray(scores, float)
    ac_vals = []

    for i in range(scores.shape[1]):
        x = scores[:-lag, i]
        y = scores[lag:, i]

        if np.std(x) > 0 and np.std(y) > 0:
            ac_vals.append(np.corrcoef(x, y)[0, 1])

    return float(np.mean(ac_vals)) if ac_vals else 0.0


# 4. LATENT STATE DYNAMICS (h_t)

def state_variation(h: np.ndarray) -> Dict[str, float]:
    """Summary statistics of the latent state step variations."""
    h = np.asarray(h, float)
    dh = np.diff(h, axis=0)
    norms = np.linalg.norm(dh, axis=1)

    return {
        "min": float(norms.min()),
        "median": float(np.median(norms)),
        "mean": float(np.mean(norms)),
        "max": float(norms.max())
    }


def state_variation_series(h: np.ndarray) -> np.ndarray:
    """Time series of the step variations $||h_t - h_{t-1}||$."""
    h = np.asarray(h, float)
    return np.linalg.norm(np.diff(h, axis=0), axis=1)


# 5. LOG-DETERMINANT AND SYSTEM ENTROPY

def log_det_safe(Sigma: np.ndarray, eps: float = 1e-12) -> float:
    """Numerically safe computation of the log-determinant."""
    w = np.linalg.eigvalsh(Sigma)
    w = np.maximum(w, eps)
    return float(np.sum(np.log(w)))


def logdet_series(Sigmas: np.ndarray) -> np.ndarray:
    """Time series of the log-determinant of $\Sigma_t$."""
    Sigmas = np.asarray(Sigmas, float)
    return np.array([log_det_safe(S) for S in Sigmas])


def logdet_stats(Sigmas: np.ndarray) -> Dict[str, float]:
    """Summary statistics of the log-determinant."""
    vals = logdet_series(Sigmas)

    return {
        "min": float(vals.min()),
        "median": float(np.median(vals)),
        "mean": float(np.mean(vals)),
        "max": float(vals.max())
    }


# 6. MATRIX DISTANCES

def frobenius_distance(S1: np.ndarray, S2: np.ndarray) -> float:
    """Frobenius distance between two matrices."""
    return float(np.linalg.norm(S1 - S2, ord="fro"))


def frobenius_distance_series(Sigmas: np.ndarray) -> np.ndarray:
    """Time series of the Frobenius distance $||\Sigma_t - \Sigma_{t-1}||_F$."""
    Sigmas = np.asarray(Sigmas, float)
    dists = np.empty(len(Sigmas) - 1)

    for t in range(1, len(Sigmas)):
        dists[t - 1] = frobenius_distance(Sigmas[t], Sigmas[t - 1])

    return dists


# 7. COMPREHENSIVE DIAGNOSTIC REPORT

def full_diagnostics(
        Sigmas: np.ndarray,
        scores: Optional[np.ndarray] = None,
        h: Optional[np.ndarray] = None
) -> Dict[str, Dict]:
    """Generates a comprehensive diagnostic report for the fitted GAS model."""
    out = {
        "pd": check_pd(Sigmas),
        "conditioning": condition_numbers(Sigmas),
        "logdet": logdet_stats(Sigmas)
    }

    if scores is not None:
        out["score_norms"] = score_norms(scores)
        out["score_autocorr_lag1"] = score_autocorrelation(scores, lag=1)

    if h is not None:
        out["state_variation"] = state_variation(h)

    return out