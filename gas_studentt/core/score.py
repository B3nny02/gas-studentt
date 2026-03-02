import numpy as np
from numba import njit

from .cholesky import vec_to_chol_d, chol_to_sigma_d, sigma_inv_and_logdet_safe,L_to_Sigma_2d
from .student_t import student_t_logpdf_d,student_t_logpdf_2d
from math import exp
@njit(cache=False)
def score_h_d(
    y: np.ndarray,
    mu: np.ndarray,
    h_vec: np.ndarray,
    nu: float,
    d: int,
    cfg_eps_pd: float,
    cfg_jitter_start: float,
    cfg_jitter_max: float,
    cfg_jitter_factor: float,
    cfg_max_iter: int,
    cfg_clip_q: float,
):
    """
    Log-likelihood score with respect to h (Cholesky parameters).

    Parameterization:
        L is lower-triangular with:
            L[i,i] = exp(h_k)
            L[i,j] = h_k   (for i > j)
        Sigma = L @ L^T

    Efficient computation:
        G = ∂ℓ/∂Sigma  (symmetric)
        ∂ℓ/∂L = 2 * G @ L

    For each parameter h_k:
      - off-diag (i > j):  ∂ℓ/∂h_k = (∂ℓ/∂L[i,j]) * 1 = 2 * (G @ L)[i,j]
      - diag (i == j):     ∂ℓ/∂h_k = (∂ℓ/∂L[i,i]) * ∂L[i,i]/∂h_k
                                   = 2 * (G @ L)[i,i] * L[i,i]

    Returns:
        tuple: (ll, score_vec, Sigma, is_valid)
    """
    m = (d * (d + 1)) // 2

    # h → L → Sigma
    L = vec_to_chol_d(h_vec, d)
    Sigma = chol_to_sigma_d(L)

    # Inverse e log-det
    Sigma_inv, logdet, is_valid = sigma_inv_and_logdet_safe(
        Sigma,
        cfg_eps_pd,
        cfg_jitter_start,
        cfg_jitter_max,
        cfg_jitter_factor,
        cfg_max_iter,
    )

    if not is_valid:
        return -1e10, np.zeros(m), Sigma, False

    # Log-pdf e gradient wrt Sigma
    ll, q_eff, grad_Sigma = student_t_logpdf_d(
        y, mu, Sigma_inv, logdet, nu, d, cfg_clip_q
    )

    # GL = grad_Sigma @ L
    GL = grad_Sigma @ L

    # Score in h
    score_vec = np.empty(m)
    idx = 0
    for i in range(d):
        for j in range(i + 1):
            if i == j:
                # diag param: L[i,i] = exp(h)  => multiply by L[i,i]
                score_vec[idx] = 2.0 * GL[i, i] * L[i, i]
            else:
                # off-diag param: L[i,j] = h
                score_vec[idx] = 2.0 * GL[i, j]
            idx += 1

    return ll, score_vec, Sigma, True



# Specialized 2D implementation
@njit(cache=False)
def score_h_2d(
    y1: float,
    y2: float,
    mu1: float,
    mu2: float,
    h1: float,
    h2: float,
    h3: float,
    nu: float,
    eps_pd: float,
    clip_q: float,
):
    """
    Specialized 2D score computation.

    h1 -> l11 = exp(h1)
    h2 -> l21 = h2
    h3 -> l22 = exp(h3)

    Returns:
        tuple: (ll, s1, s2, s3, s11, s12, s22)
    """
    

    l11 = exp(h1)
    l21 = h2
    l22 = exp(h3)

    s11, s12, s22 = L_to_Sigma_2d(l11, l21, l22)

    ll, q_eff, inv11, inv12, inv22 = student_t_logpdf_2d(
        y1, y2, mu1, mu2, s11, s12, s22, nu, eps_pd, clip_q
    )

    # grad_Sigma (2x2)
    x1 = y1 - mu1
    x2 = y2 - mu2
    v1 = inv11 * x1 + inv12 * x2
    v2 = inv12 * x1 + inv22 * x2

    w = (nu + 2.0) / (nu + q_eff)

    g11 = 0.5 * (w * v1 * v1 - inv11)
    g12 = 0.5 * (w * v1 * v2 - inv12)
    g22 = 0.5 * (w * v2 * v2 - inv22)

    # L = [[l11, 0],
    #      [l21, l22]]
    # GL = G @ L
    # G = [[g11, g12],
    #      [g12, g22]]
    GL00 = g11 * l11 + g12 * l21
    GL10 = g12 * l11 + g22 * l21
    GL11 = g12 * 0.0 + g22 * l22  # = g22*l22

    # score in h:
    # s1 = 2*GL00 * l11
    # s2 = 2*GL10
    # s3 = 2*GL11 * l22
    s1 = 2.0 * GL00 * l11
    s2 = 2.0 * GL10
    s3 = 2.0 * GL11 * l22

    return ll, s1, s2, s3, s11, s12, s22
