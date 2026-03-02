"""
Numba-optimized GAS filter for multivariate Student-t models.
This module implements the core filtering recursion.
"""

import numpy as np
from numba import njit

from .score import score_h_d
from .scaling import (
    scale_identity,
    scale_fisher_diag,
    scale_sqrt_fisher_diag,
    scale_block_diag,
    clip_vector,
)

# Scaling type constants (must match SCALING_MAP in the model class)
SCALE_IDENTITY = 0
SCALE_FISHER_DIAG = 1  # Dynamic Inverse Fisher
SCALE_SQRT_FISHER_DIAG = 2  # Dynamic Sqrt Fisher
SCALE_BLOCK_DIAG = 3  # Dynamic Block Fisher


@njit(cache=False)
def gas_filter_d_studentt(
        Y: np.ndarray,  # (T, d)
        mu: np.ndarray,  # (d,)
        omega: np.ndarray,  # (m,)
        A: np.ndarray,  # (m,) or (m, m)
        B: np.ndarray,  # (m,) or (m, m)
        h0: np.ndarray,  # (m,)
        nu: float,
        d: int,
        scaling_id: int,
        grid_x: np.ndarray,  # (n_points,) Base grid points for h
        grid_fisher: np.ndarray,  # (m, n_points) Pre-computed Fisher values

        # Positive definiteness / numerical safety parameters
        eps_pd: float,
        jitter_start: float,
        jitter_max: float,
        jitter_factor: float,
        max_jitter_iter: int,
        clip_q: float,
        clip_score: float,
        clip_scale: float,
        min_cond: float,
        penalty_non_pd: float,
):
    """
    Multivariate Student-t GAS filter with dynamic Fisher scaling.

    The scaling step dynamically interpolates the Fisher Information matrix
    based on the current state vector h_t.
    """
    T = Y.shape[0]
    m = h0.shape[0]

    # Output arrays
    h_arr = np.zeros((T, m))
    Sigmas = np.zeros((T, d, d))
    scores = np.zeros((T, m))
    scaled_scores = np.zeros((T, m))

    # Initialization
    h_arr[0, :] = h0
    ll_total = 0.0
    n_pd_failures = 0
    last_valid_Sigma = np.eye(d)

    # Matrix format detection for A/B (Scalar, Diagonal, or Full)
    A_is_diag = (A.ndim == 1)
    B_is_diag = (B.ndim == 1)

    # Buffer for scaled score
    s_scaled = np.zeros(m)

    for t in range(T):
        y_t = Y[t, :]
        h_t = h_arr[t, :]

        # 1. Conditional Density & Raw Score
        ll_t, score_t, Sigma_t, is_valid = score_h_d(
            y_t, mu, h_t, nu, d,
            eps_pd, jitter_start, jitter_max, jitter_factor,
            max_jitter_iter, clip_q
        )

        if is_valid:
            ll_total += ll_t
            last_valid_Sigma = Sigma_t
        else:
            # Robust failure handling: penalize likelihood and reuse last valid state
            n_pd_failures += 1
            ll_total -= penalty_non_pd
            Sigma_t = last_valid_Sigma
            score_t = np.zeros(m)

        # Universal score clipping to prevent exploding gradients
        score_t = clip_vector(score_t, clip_score)

        # Store outputs
        scores[t, :] = score_t
        Sigmas[t, :, :] = Sigma_t

        # 2. Dynamic Scaling Dispatch
        if scaling_id == SCALE_IDENTITY:
            s_scaled = scale_identity(score_t)

        elif scaling_id == SCALE_FISHER_DIAG:
            s_scaled = scale_fisher_diag(
                score_t, h_t, grid_x, grid_fisher, min_cond, clip_scale
            )

        elif scaling_id == SCALE_SQRT_FISHER_DIAG:
            s_scaled = scale_sqrt_fisher_diag(
                score_t, h_t, grid_x, grid_fisher, min_cond, clip_scale
            )

        elif scaling_id == SCALE_BLOCK_DIAG:
            s_scaled = scale_block_diag(
                score_t, h_t, grid_x, grid_fisher, d, min_cond, clip_scale
            )

        else:
            s_scaled = scale_identity(score_t)

        scaled_scores[t, :] = s_scaled

        # 3. Update State h_{t+1}
        if t < T - 1:
            diff = h_t - omega

            if A_is_diag:
                term_A = A * s_scaled
            else:
                term_A = A @ s_scaled

            if B_is_diag:
                term_B = B * diff
            else:
                term_B = B @ diff

            h_arr[t + 1, :] = omega + term_A + term_B

    return ll_total, h_arr, Sigmas, scores, scaled_scores, n_pd_failures