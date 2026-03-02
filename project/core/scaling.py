"""
Scaling matrices for multivariate GAS models.

This module implements dynamic Fisher information-based scaling strategies.
All methods rely on a pre-computed grid of Fisher information values and
interpolate the local curvature based on the current state parameter h_t.
"""

import numpy as np
from numba import njit


@njit(cache=False)
def clip_vector(x: np.ndarray, clip_value: float) -> np.ndarray:
    """Clips vector elements to [-clip_value, clip_value] for numerical stability."""
    if clip_value <= 0.0:
        return x
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        v = x[i]
        if v > clip_value:
            v = clip_value
        elif v < -clip_value:
            v = -clip_value
        else:
            v = v
        out[i] = v
    return out


@njit(cache=False)
def _floor_pos(x: float, min_cond: float) -> float:
    """Ensures positivity and applies a numerical floor to prevent division by zero."""
    if not np.isfinite(x):
        return min_cond
    if x < min_cond:
        return min_cond
    return x


@njit(cache=False)
def scale_identity(score: np.ndarray) -> np.ndarray:
    """
    Identity scaling: returns the raw score without modification.
    This method is static and ignores curvature entirely.
    """
    return score


@njit(cache=False)
def scale_fisher_diag(
        score: np.ndarray,
        h_t: np.ndarray,
        grid_x: np.ndarray,
        grid_fisher: np.ndarray,
        min_cond: float,
        clip_scale: float
) -> np.ndarray:
    """
    Dynamic Diagonal Inverse Fisher Information scaling.
    Interpolates the Fisher Information from the grid and applies inverse scaling.
    """
    m = score.shape[0]
    out = np.empty_like(score)

    for i in range(m):
        fisher_val = np.interp(h_t[i], grid_x, grid_fisher[i])
        denom = _floor_pos(fisher_val, min_cond)
        out[i] = score[i] / denom

    return clip_vector(out, clip_scale)


@njit(cache=False)
def scale_sqrt_fisher_diag(
        score: np.ndarray,
        h_t: np.ndarray,
        grid_x: np.ndarray,
        grid_fisher: np.ndarray,
        min_cond: float,
        clip_scale: float
) -> np.ndarray:
    r"""
    Dynamic Diagonal Square-Root Fisher Information scaling.

    Interpolates the local curvature and scales the score by its inverse square root.
    This ensures the scaled score has approximately unit variance conditionally.
    """
    m = score.shape[0]
    out = np.empty_like(score)

    for i in range(m):
        fisher_val = np.interp(h_t[i], grid_x, grid_fisher[i])
        denom = _floor_pos(fisher_val, min_cond)
        out[i] = score[i] / np.sqrt(denom)

    return clip_vector(out, clip_scale)


@njit(cache=False)
def scale_block_diag(
        score: np.ndarray,
        h_t: np.ndarray,
        grid_x: np.ndarray,
        grid_fisher: np.ndarray,
        d: int,
        min_cond: float,
        clip_scale: float
) -> np.ndarray:
    r"""
    Dynamic Block-Diagonal Fisher scaling.

    Applies square-root scaling dynamically, explicitly separating logic
    for Volatility and Correlation blocks for potential future customization.
    """
    m = score.shape[0]
    out = np.empty_like(score)

    for i in range(d):
        fisher_val = np.interp(h_t[i], grid_x, grid_fisher[i])
        denom = _floor_pos(fisher_val, min_cond)
        out[i] = score[i] / np.sqrt(denom)

    for i in range(d, m):
        fisher_val = np.interp(h_t[i], grid_x, grid_fisher[i])
        denom = _floor_pos(fisher_val, min_cond)
        out[i] = score[i] / np.sqrt(denom)

    return clip_vector(out, clip_scale)