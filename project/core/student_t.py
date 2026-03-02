import numpy as np
from numba import njit
from math import lgamma, log, pi


@njit(cache=False)
def student_t_logpdf_d(
    y: np.ndarray,
    mu: np.ndarray,
    Sigma_inv: np.ndarray,
    logdet: float,
    nu: float,
    d: int,
    clip_q: float
):
    """
    Computes the multivariate Student-t log-pdf and its gradient
    with respect to the covariance matrix Sigma.

    Returns:
        tuple: (ll, q_eff, grad_Sigma) where `ll` is the log-likelihood,
               `q_eff` is the clipped Mahalanobis distance, and `grad_Sigma`
               is the gradient matrix.
    """
    # Safety guard: degrees of freedom must be strictly positive
    if not (nu > 0.0):
        return -1e20, 0.0, -0.5 * Sigma_inv

    x = y - mu
    v = Sigma_inv @ x

    q = 0.0
    for i in range(d):
        q += x[i] * v[i]

    # Clip Mahalanobis distance to prevent numerical overflow
    q_eff = q
    if q_eff > clip_q:
        q_eff = clip_q

    # Normalization constant and log-likelihood computation
    c = lgamma((nu + d) / 2.0) - lgamma(nu / 2.0) - (d / 2.0) * log(nu * pi)
    ll = c - 0.5 * logdet - ((nu + d) / 2.0) * log(1.0 + q_eff / nu)

    # Gradient with respect to Sigma
    w = (nu + d) / (nu + q_eff)
    grad_Sigma = 0.5 * (w * np.outer(v, v) - Sigma_inv)

    return ll, q_eff, grad_Sigma


@njit(cache=False)
def student_t_logpdf_2d(
    y1: float,
    y2: float,
    mu1: float,
    mu2: float,
    s11: float,
    s12: float,
    s22: float,
    nu: float,
    eps_pd: float,
    clip_q: float
):
    """
    Specialized 2D implementation of the multivariate Student-t log-pdf
    for optimized execution speed.

    Returns:
        tuple: (ll, q_eff, inv11, inv12, inv22)
    """
    # Safety guard
    if not (nu > 0.0):
        inv11 = 1.0 / eps_pd
        inv12 = 0.0
        inv22 = 1.0 / eps_pd
        return -1e20, 0.0, inv11, inv12, inv22

    # Analytical 2x2 inversion
    det = s11 * s22 - s12 * s12
    if det < eps_pd:
        det = eps_pd

    inv11 = s22 / det
    inv12 = -s12 / det
    inv22 = s11 / det

    logdet = log(det)

    x1 = y1 - mu1
    x2 = y2 - mu2

    v1 = inv11 * x1 + inv12 * x2
    v2 = inv12 * x1 + inv22 * x2

    q = x1 * v1 + x2 * v2

    # Clip Mahalanobis distance to prevent numerical overflow
    q_eff = q
    if q_eff > clip_q:
        q_eff = clip_q

    d2 = 2.0
    c = lgamma((nu + d2) / 2.0) - lgamma(nu / 2.0) - (d2 / 2.0) * log(nu * pi)
    ll = c - 0.5 * logdet - ((nu + d2) / 2.0) * log(1.0 + q_eff / nu)

    return ll, q_eff, inv11, inv12, inv22