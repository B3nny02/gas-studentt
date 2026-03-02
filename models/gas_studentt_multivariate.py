import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, List, Any, Tuple
from scipy.optimize import minimize

from gas_studentt.utils.config import GASConfig
from gas_studentt.utils.constraints import logistic, inv_logistic, nu_from_raw, raw_from_nu

from gas_studentt.core.gas_filter_numba import gas_filter_d_studentt
from gas_studentt.core.fisher import compute_fisher_grid
from gas_studentt.core.cholesky import vec_to_chol_d, chol_to_sigma_d

# SCALING MAP
SCALING_MAP = {
    "identity": 0,
    "fisher_diag": 1,  # Dynamic Inverse Fisher
    "sqrt_fisher_diag": 2,  # Dynamic Sqrt Fisher (Recommended)
    "block_diag": 3,  # Dynamic Block Fisher
}


@dataclass
class FitResult:
    """Encapsulates the results of the Maximum Likelihood Estimation."""
    success: bool
    message: str
    nfev: int
    params: Dict[str, np.ndarray]
    ll: float
    aic: float
    bic: float
    x_opt: np.ndarray
    n_pd_failures: int


class GASStudentTMultivariate:
    """
    Multivariate Generalized Autoregressive Score (GAS) model with Student-t innovations.
    Utilizes Cholesky parameterization to ensure Positive Definiteness (PD) of the
    conditional covariance matrix by algebraic construction.
    """

    def __init__(
            self,
            d: int = 2,
            scaling: str = "sqrt_fisher_diag",  # Robust default
            update_type: str = "diagonal",
            config: Optional[GASConfig] = None,
    ):
        if scaling not in SCALING_MAP:
            raise ValueError(
                f"Scaling '{scaling}' is not supported. Valid options: {list(SCALING_MAP.keys())}"
            )
        if update_type not in ("scalar", "diagonal", "full"):
            raise ValueError("update_type must be one of: 'scalar', 'diagonal', 'full'.")

        self.d = int(d)
        self.m = (self.d * (self.d + 1)) // 2

        self.scaling = scaling
        self.scaling_id = SCALING_MAP[scaling]
        self.update_type = update_type
        self.cfg = GASConfig(d=self.d) if config is None else config

        # Grid for Dynamic Scaling
        self.fisher_grid_val: Optional[np.ndarray] = None  # (m, n_points)
        self.fisher_grid_x: Optional[np.ndarray] = None  # (n_points,)
        self._fisher_grid_ready: bool = False

        # Fit state
        self.params_: Optional[Dict[str, np.ndarray]] = None
        self.last_filter_: Optional[Dict[str, Any]] = None

        # Debug tracking
        self._n_obj_failures: int = 0
        self._last_obj_error: str = ""

    @staticmethod
    def _pack_params(mu, omega, A, B, nu_raw, h0, update_type):
        if update_type == "scalar":
            A_vec = np.array([A[0]])
            B_vec = np.array([B[0]])
        elif update_type == "diagonal":
            A_vec = A
            B_vec = B
        else:  # full
            A_vec = A.ravel()
            B_vec = B.ravel()

        return np.concatenate([mu, omega, A_vec, B_vec, np.array([nu_raw]), h0])

    @staticmethod
    def _unpack_params(x, d, m, update_type):
        idx = 0
        mu = x[idx:idx + d];
        idx += d
        omega = x[idx:idx + m];
        idx += m

        if update_type == "scalar":
            A = np.array([x[idx]]);
            idx += 1
            B = np.array([x[idx]]);
            idx += 1
        elif update_type == "diagonal":
            A = x[idx:idx + m];
            idx += m
            B = x[idx:idx + m];
            idx += m
        else:  # full
            A = x[idx:idx + m * m].reshape(m, m);
            idx += m * m
            B = x[idx:idx + m * m].reshape(m, m);
            idx += m * m

        nu_raw = x[idx];
        idx += 1
        h0 = x[idx:idx + m]

        return mu, omega, A, B, nu_raw, h0

    def _transform_params(self, mu, omega, A_raw, B_raw, nu_raw, h0):
        if self.update_type in ("scalar", "diagonal"):
            A = logistic(A_raw, lo=0.0, hi=0.999)
            B = logistic(B_raw, lo=0.0, hi=0.999)
        else:
            A = np.tanh(A_raw) * 0.5
            B = logistic(B_raw, lo=0.0, hi=0.999)

        nu = nu_from_raw(float(nu_raw))
        h0 = np.clip(h0, self.cfg.clip_h_min, self.cfg.clip_h_max)
        return mu, omega, A, B, nu, h0

    def precompute_fisher_grid(
            self,
            h_min: float = -12.0,
            h_max: float = 12.0,
            n_points: int = 50,
            n_mc: int = 1000,
            nu_target: Optional[float] = None
    ):
        """
        Computes the grid for Fisher Information interpolation.
        Must be called before fitting if a dynamic Fisher scaling method is selected.
        """
        if nu_target is None:
            nu_target = self.cfg.nu_init

        print(f"[GAS] Pre-computing Dynamic Fisher Grid (nu={nu_target:.1f}, points={n_points})...")

        table, grid_x = compute_fisher_grid(
            d=self.d,
            nu=float(nu_target),
            h_min=h_min,
            h_max=h_max,
            n_points=n_points,
            n_mc_samples=n_mc,
            eps_pd=self.cfg.eps_pd,
            clip_q=self.cfg.clip_q
        )

        self.fisher_grid_val = table
        self.fisher_grid_x = grid_x
        self._fisher_grid_ready = True
        print("[GAS] Grid computation complete.")

    def _ensure_grid_ready(self):
        """Ensures the Fisher grid is initialized before running the filter."""
        if self.scaling == "identity":
            return

        if not self._fisher_grid_ready:
            print("Warning: Fisher Grid required but not found. Computing default grid (-12, 12)...")
            self.precompute_fisher_grid()

    def filter(
            self,
            Y: np.ndarray,
            params: Dict[str, np.ndarray],
            *,
            override_cfg: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Executes the GAS filter using the Numba-optimized backend.
        Passes the pre-computed grids (or valid dummies) for dynamic scaling.
        """
        Y = np.asarray(Y, float)
        T, d_obs = Y.shape

        if d_obs != self.d:
            raise ValueError(f"Observation dimension {d_obs} does not match model dimension {self.d}")

        # Unpack parameters
        mu = np.asarray(params["mu"], float)
        omega = np.asarray(params["omega"], float)
        A = params["A"]
        B = params["B"]
        nu = float(params["nu"])
        h0 = np.asarray(params["h0"], float)

        # Expand A/B based on update type
        if self.update_type == "scalar":
            A_f = np.full(self.m, float(A[0]))
            B_f = np.full(self.m, float(B[0]))
        elif self.update_type == "diagonal":
            A_f = np.asarray(A, float)
            B_f = np.asarray(B, float)
        else:
            A_f = np.asarray(A, float)
            B_f = np.asarray(B, float)

        # Prepare grids for Numba execution
        if self._fisher_grid_ready:
            gx = self.fisher_grid_x
            gf = self.fisher_grid_val
        else:
            gx = np.array([0.0, 1.0], dtype=float)
            gf = np.ones((self.m, 2), dtype=float)

        # Configuration override function
        def _cfg(name: str):
            if override_cfg is not None and name in override_cfg:
                return override_cfg[name]
            return getattr(self.cfg, name)

        ll, h, Sigmas, sc, ssc, n_fail = gas_filter_d_studentt(
            Y=Y,
            mu=mu,
            omega=omega,
            A=A_f,
            B=B_f,
            h0=h0,
            nu=nu,
            d=self.d,
            scaling_id=self.scaling_id,
            grid_x=gx,
            grid_fisher=gf,
            eps_pd=_cfg("eps_pd"),
            jitter_start=_cfg("jitter_start"),
            jitter_max=_cfg("jitter_max"),
            jitter_factor=_cfg("jitter_factor"),
            max_jitter_iter=_cfg("max_jitter_iter"),
            clip_q=_cfg("clip_q"),
            clip_score=_cfg("clip_score"),
            clip_scale=_cfg("clip_scale"),
            min_cond=_cfg("min_cond"),
            penalty_non_pd=_cfg("penalty_non_pd"),
        )

        out = {
            "ll": float(ll),
            "h": h,
            "Sigmas": Sigmas,
            "scores": sc,
            "scaled_scores": ssc,
            "n_pd_failures": int(n_fail),
        }
        self.last_filter_ = out
        return out

    def _neg_ll(self, x: np.ndarray, Y: np.ndarray) -> float:
        try:
            mu, omega, A_raw, B_raw, nu_raw, h0 = self._unpack_params(
                x, self.d, self.m, self.update_type
            )
            mu, omega, A, B, nu, h0 = self._transform_params(
                mu, omega, A_raw, B_raw, nu_raw, h0
            )
            params = {"mu": mu, "omega": omega, "A": A, "B": B, "nu": nu, "h0": h0}

            out = self.filter(Y, params)

            val = -out["ll"]
            if not np.isfinite(val):
                return 1e15
            return float(val)

        except Exception as e:
            self._n_obj_failures += 1
            self._last_obj_error = str(e)
            return 1e15

    def _init_x0(self, Y: np.ndarray) -> np.ndarray:
        mu0 = Y.mean(axis=0)
        omega0 = np.full(self.m, -2.0)  # Start from a low covariance state

        if self.update_type == "scalar":
            A0, B0 = np.array([0.05]), np.array([0.90])
        elif self.update_type == "diagonal":
            A0, B0 = np.full(self.m, 0.05), np.full(self.m, 0.90)
        else:
            A0, B0 = np.eye(self.m) * 0.05, np.eye(self.m) * 0.90

        nu0 = float(self.cfg.nu_init)
        h00 = omega0.copy()

        if self.update_type in ("scalar", "diagonal"):
            A_raw0 = inv_logistic(A0)
            B_raw0 = inv_logistic(B0)
        else:
            A_raw0 = np.arctanh(np.clip(A0 / 0.5, -0.999, 0.999))
            B_raw0 = inv_logistic(B0)

        nu_raw0 = raw_from_nu(nu0)

        return self._pack_params(mu0, omega0, A_raw0, B_raw0, nu_raw0, h00, self.update_type)

    def _jitter_x0(self, x0, rng, scale=0.2):
        noise = rng.normal(0.0, scale, size=x0.shape)
        noise[:self.d] *= 0.1
        return x0 + noise

    def fit(
            self,
            Y: np.ndarray,
            x0: Optional[np.ndarray] = None,
            method: str = "L-BFGS-B",
            maxiter: Optional[int] = None,
            multistart: Optional[int] = None,
            seed: Optional[int] = None,
            fallback_methods: Optional[List[str]] = None,
            use_pure_ll_for_ic: bool = False,
    ) -> FitResult:
        """
        Estimates the static parameters using Maximum Likelihood via a multistart
        optimization strategy to avoid local optima.
        """
        Y = np.asarray(Y, float)
        T = Y.shape[0]

        # Setup Configuration
        if maxiter is None: maxiter = self.cfg.opt_maxiter
        if multistart is None: multistart = self.cfg.opt_multistart
        if seed is None: seed = self.cfg.opt_seed
        if fallback_methods is None: fallback_methods = ["Powell", "Nelder-Mead"]

        # Ensure scaling grid is ready
        self._ensure_grid_ready()

        if x0 is None:
            base = self._init_x0(Y)
        else:
            base = np.asarray(x0, float)

        rng = np.random.default_rng(int(seed))
        starts = [base] + [
            self._jitter_x0(base, rng)
            for _ in range(max(1, int(multistart)) - 1)
        ]

        best_res = None
        best_fun = np.inf

        for i_s, x_init in enumerate(starts):
            current_methods = [method] + [m for m in fallback_methods if m != method]

            for meth in current_methods:
                try:
                    res = minimize(
                        fun=lambda z: self._neg_ll(z, Y),
                        x0=x_init,
                        method=meth,
                        options={"maxiter": int(maxiter), "disp": False},
                    )
                    if res.fun < best_fun:
                        best_fun = float(res.fun)
                        best_res = res

                    if res.success:
                        break
                except Exception:
                    continue

        if best_res is None:
            raise RuntimeError("Optimization failed completely.")

        res = best_res

        mu, omega, A_raw, B_raw, nu_raw, h0 = self._unpack_params(
            res.x, self.d, self.m, self.update_type
        )
        mu, omega, A, B, nu, h0 = self._transform_params(
            mu, omega, A_raw, B_raw, nu_raw, h0
        )

        self.params_ = {"mu": mu, "omega": omega, "A": A, "B": B, "nu": nu, "h0": h0}

        out = self.filter(Y, self.params_)
        ll_final = out["ll"]

        if use_pure_ll_for_ic:
            override = {
                "penalty_non_pd": 0.0,
                "clip_q": 1e18,
                "clip_score": 1e18,
                "clip_scale": 1e18
            }
            out_pure = self.filter(Y, self.params_, override_cfg=override)
            ll_final = out_pure["ll"]

        k_params = len(res.x)
        aic = 2.0 * k_params - 2.0 * ll_final
        bic = float(k_params * np.log(T) - 2.0 * ll_final)

        return FitResult(
            success=bool(res.success),
            message=str(res.message),
            nfev=int(res.nfev),
            params=self.params_,
            ll=float(ll_final),
            aic=float(aic),
            bic=float(bic),
            x_opt=np.asarray(res.x, float),
            n_pd_failures=int(out["n_pd_failures"]),
        )

    def forecast(self, h_last: np.ndarray, H: int = 1) -> Dict[str, np.ndarray]:
        """
        Generates multi-step ahead forecasts for the latent states and conditional
        covariance matrices. Assumes the expected future score is zero.
        """
        if self.params_ is None:
            raise RuntimeError("Model is not fitted. Run .fit() first.")

        A, B = self.params_["A"], self.params_["B"]
        omega = self.params_["omega"]

        h_fc = np.zeros((H, self.m))
        Sigma_fc = np.zeros((H, self.d, self.d))

        h_t = np.asarray(h_last, float).copy()

        if self.update_type == "scalar":
            B_op = lambda h: np.full(self.m, float(B[0])) * h
        elif self.update_type == "diagonal":
            B_op = lambda h: B * h
        else:  # full
            B_op = lambda h: B @ h

        for t in range(H):
            h_next = omega + B_op(h_t - omega)
            h_fc[t] = h_next
            L = vec_to_chol_d(h_next, self.d)
            Sigma_fc[t] = chol_to_sigma_d(L)
            h_t = h_next

        return {"h_forecast": h_fc, "Sigma_forecast": Sigma_fc}

    def last_Sigmas(self):
        if self.last_filter_ is None:
            raise RuntimeError("Run filter or fit first.")
        return self.last_filter_["Sigmas"]

    def last_h(self):
        if self.last_filter_ is None:
            raise RuntimeError("Run filter or fit first.")
        return self.last_filter_["h"]

    def last_scores(self):
        if self.last_filter_ is None:
            raise RuntimeError("Run filter or fit first.")
        return self.last_filter_["scores"]