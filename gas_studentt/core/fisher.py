import numpy as np
from .score import score_h_d
from .cholesky import vec_to_chol_d, chol_to_sigma_d


def compute_fisher_grid(
        d: int,
        nu: float,
        h_min: float = -5.0,
        h_max: float = 5.0,
        n_points: int = 50,
        n_mc_samples: int = 1000,
        seed: int = 42,
        # Safety config defaults
        eps_pd: float = 1e-8,
        jitter_start: float = 1e-8,
        jitter_max: float = 1e-3,
        jitter_factor: float = 10.0,
        max_jitter_iter: int = 5,
        clip_q: float = 1e6
):
    r"""
    Compute a grid of Fisher Information values for dynamic scaling.

    For each parameter k and each grid value x, this function:
    1. Sets h_k = x (keeping other parameters at a neutral baseline, e.g., 0).
    2. Simulates $Y \sim t_\nu(0, \Sigma(h))$.
    3. Computes the expected squared score $E[\nabla_k \ell \cdot \nabla_k \ell]$.

    Returns
    -------
    fisher_table : np.ndarray (m, n_points)
        The computed information values.
    grid_x : np.ndarray (n_points,)
        The grid points used for h.
    """
    m = (d * (d + 1)) // 2
    grid_x = np.linspace(h_min, h_max, n_points)
    fisher_table = np.zeros((m, n_points))

    rng = np.random.default_rng(seed)
    mu_dummy = np.zeros(d)

    # Pre-allocate reuse buffers
    h_temp = np.zeros(m)

    for j, val_h in enumerate(grid_x):
        for k in range(m):
            # Construct the test vector h.
            # Set the k-th element to the grid value, keep others at 0.0.
            # (0.0 implies vol=1.0 for diagonal elements, and correlation=0.0 for off-diagonal)
            h_temp[:] = 0.0
            h_temp[k] = val_h

            # 1. Reconstruct Sigma
            L = vec_to_chol_d(h_temp, d)
            Sigma = chol_to_sigma_d(L)

            # 2. Monte Carlo Generation: Y ~ Student-t(0, Sigma, nu)
            # Y = mu + sqrt(nu / w) * Z, where Z ~ N(0, Sigma) and w ~ Chi2(nu)
            Z = rng.multivariate_normal(np.zeros(d), Sigma, size=n_mc_samples)
            w = rng.chisquare(df=nu, size=n_mc_samples)

            # Prevent division by zero
            w = np.maximum(w, 1e-8)
            scale_factor = np.sqrt(nu / w)

            # Broadcasting: (N, 1) * (N, d)
            Y_sim = mu_dummy + Z * scale_factor[:, None]

            # 3. Compute Expected Squared Score
            scores_sq_sum = 0.0
            valid_count = 0

            for i_mc in range(n_mc_samples):
                y_curr = Y_sim[i_mc]

                _, s_vec, _, valid = score_h_d(
                    y_curr, mu_dummy, h_temp, nu, d,
                    eps_pd, jitter_start, jitter_max, jitter_factor, max_jitter_iter, clip_q
                )

                if valid:
                    # Extract the k-th component of the score
                    # I_kk(h) = E[ (d_ell / d_hk)^2 ]
                    s_val = s_vec[k]

                    # Robustness: ignore extreme outliers resulting from rare numerical instabilities
                    if np.abs(s_val) < 1e10:
                        scores_sq_sum += (s_val * s_val)
                        valid_count += 1

            if valid_count > 10:  # Minimum statistical threshold
                fisher_val = scores_sq_sum / valid_count
            else:
                fisher_val = 1.0  # Conservative fallback

            # Ensure a positive and finite value
            if not np.isfinite(fisher_val) or fisher_val < 1e-9:
                fisher_val = 1.0

            fisher_table[k, j] = fisher_val

    return fisher_table, grid_x