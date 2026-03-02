from dataclasses import dataclass

@dataclass(frozen=True)
class GASConfig:
    """
    Configuration for multivariate GAS models with Fisher-based scaling.
    
    All parameters are tuned for numerical stability in Cholesky-parameterized
    Student-t volatility models.
    """

    # DIMENSION
    d: int = 2

    # POSITIVITY DEFINITENESS / NUMERICAL SAFETY
    eps_pd: float = 1e-10
    jitter_start: float = 1e-8
    jitter_max: float = 1e-4
    jitter_factor: float = 10.0
    max_jitter_iter: int = 10

    # CLIPPING
    clip_q: float = 1e6
    clip_score: float = 1e4
    clip_scale: float = 1e3
    clip_h_min: float = -20.0
    clip_h_max: float = 20.0

    # CONDITIONING
    min_cond: float = 1e-12

    # OPTIMIZATON
    opt_maxiter: int = 800
    opt_multistart: int = 5
    opt_seed: int = 123
    opt_tol: float = 1e-6

    # PENALTY FOR NON-PD
    penalty_non_pd: float = 1e10

    # NUMERICAL FISHER ESTIMATION
    fisher_eps: float = 1e-5
    fisher_T_sub: int = 200

    # STUDENT t INIZIALIZATION
    nu_init: float = 10.0
