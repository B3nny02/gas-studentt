import numpy as np


def _clip_array(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(x, lo, hi)


def logistic(x: np.ndarray, lo: float = 0.0, hi: float = 0.999) -> np.ndarray:
    """
    Stable logistic transformation mapping an unconstrained input to the interval (lo, hi).
    """
    x = np.asarray(x, float)
    z = _clip_array(x, -50.0, 50.0)
    sig = 1.0 / (1.0 + np.exp(-z))
    return lo + (hi - lo) * sig


def inv_logistic(y: np.ndarray, lo: float = 0.0, hi: float = 0.999) -> np.ndarray:
    """
    Inverse of the logistic transformation.
    """
    y = np.asarray(y, float)
    eps = 1e-12
    y = _clip_array(y, lo + eps, hi - eps)
    ratio = (y - lo) / (hi - lo)
    ratio = _clip_array(ratio, eps, 1.0 - eps)
    return -np.log((1.0 / ratio) - 1.0)


def softplus(x: float) -> float:
    """Numerically stable Softplus function."""
    if x > 30.0:
        return float(x)
    if x < -30.0:
        return float(np.exp(x))
    return float(np.log1p(np.exp(x)))


def inv_softplus(y: float) -> float:
    """Inverse of the Softplus function."""
    y = max(float(y), 1e-12)
    if y > 30.0:
        return float(y)
    return float(np.log(np.expm1(y)))


def nu_from_raw(nu_raw: float) -> float:
    """
    Transformation ensuring the Student-t degrees of freedom strictly exceed 2.
    """
    return 2.0 + softplus(float(nu_raw))


def raw_from_nu(nu: float) -> float:
    """
    Inverse transformation for the degrees of freedom.
    """
    nu = max(float(nu), 2.000001)
    return inv_softplus(nu - 2.0)


def matrix_log_transform(Sigma: np.ndarray) -> np.ndarray:
    """
    Matrix logarithm transformation to ensure symmetric positive definiteness.
    Included as a parametric alternative, but not utilized in the core GAS model.
    """
    Sigma = np.asarray(Sigma, float)
    w, V = np.linalg.eigh(Sigma)
    w = np.maximum(w, 1e-12)
    return V @ np.diag(np.log(w)) @ V.T


def matrix_exp_transform(M: np.ndarray) -> np.ndarray:
    """
    Matrix exponential transformation (inverse of the matrix logarithm).
    """
    M = np.asarray(M, float)
    w, V = np.linalg.eigh(M)
    return V @ np.diag(np.exp(w)) @ V.T