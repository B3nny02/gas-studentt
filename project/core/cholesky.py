import numpy as np
from numba import njit
from math import exp


@njit(cache=False)
def vec_to_chol_d(h_vec: np.ndarray, d: int) -> np.ndarray:
    """
    Maps the unconstrained vector h_vec to a lower-triangular Cholesky factor L.
    Diagonal elements are exponentiated to ensure strict positivity.
    """
    L = np.zeros((d, d))
    idx = 0
    for i in range(d):
        for j in range(i + 1):
            if i == j:
                L[i, j] = exp(h_vec[idx])
            else:
                L[i, j] = h_vec[idx]
            idx += 1
    return L


@njit(cache=False)
def chol_to_sigma_d(L: np.ndarray) -> np.ndarray:
    """Reconstructs the covariance matrix: Sigma = L @ L.T"""
    return L @ L.T


@njit(cache=False)
def L_to_Sigma_2d(l11: float, l21: float, l22: float):
    """Analytical covariance reconstruction for the 2D case."""
    s11 = l11 * l11
    s12 = l11 * l21
    s22 = l21 * l21 + l22 * l22
    return s11, s12, s22


@njit(cache=False)
def sigma_inv_and_logdet_safe(
        Sigma: np.ndarray,
        eps_pd: float,
        jitter_start: float,
        jitter_max: float,
        jitter_factor: float,
        max_iter: int,
):
    """
    Computes the inverse and log-determinant of Sigma with numerical safeguards.

    Numba-compatible strategy:
    1. Computes the minimum eigenvalue.
    2. If near-singular, applies diagonal loading (jitter).
    3. Performs Cholesky decomposition and inversion.

    Returns: tuple (Sigma_inv, logdet, is_valid)
    """
    d = Sigma.shape[0]
    S = Sigma.copy()

    jitter = jitter_start
    ok = False

    for _ in range(max_iter):
        evals = np.linalg.eigvalsh(S)
        min_eig = evals[0]

        if not np.isfinite(min_eig):
            ok = False
            break

        if min_eig <= eps_pd:
            # Apply diagonal loading (jitter) to restore positive definiteness
            add = (eps_pd - min_eig) + jitter
            if add > jitter_max:
                add = jitter_max
            for i in range(d):
                S[i, i] = S[i, i] + add

            jitter = jitter * jitter_factor
            if jitter > jitter_max:
                jitter = jitter_max
            continue

        L = np.linalg.cholesky(S)

        logdet = 0.0
        for i in range(d):
            lii = L[i, i]
            if lii <= 0.0 or (not np.isfinite(lii)):
                ok = False
                break
            logdet += 2.0 * np.log(lii)

        if not ok and logdet == 0.0:
            break

        # Compute inverse using L
        I = np.eye(d)
        Linv = np.linalg.solve(L, I)
        Sigma_inv = Linv.T @ Linv

        # Sanity check for numerical stability
        if (not np.isfinite(logdet)) or (not np.all(np.isfinite(Sigma_inv))):
            ok = False
            break

        ok = True
        return Sigma_inv, logdet, True

    return np.zeros((d, d)), 0.0, False