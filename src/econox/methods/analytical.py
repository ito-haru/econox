# src/econox/methods/analytical.py
"""
Analytical estimation methods (formula-based).
Linear models like OLS and 2SLS that can be solved directly using matrix algebra.
"""

from typing import Any, Dict
import jax.numpy as jnp
import equinox as eqx
import lineax as lx
from jaxtyping import Array, PyTree

from econox.protocols import StructuralModel
from econox.utils import get_from_pytree
from econox.structures import EstimationResult
from econox.methods.base import EstimationMethod
from econox.methods.variance import Hessian, Variance

class LinearMethod(EstimationMethod):
    """
    Base class for linear estimators. 
    Provides data preparation and a fallback `compute_loss`.
    """
    solver: lx.AbstractLinearSolver = lx.AutoLinearSolver(well_posed=None)
    add_intercept: bool = True
    target_key: str = "y"
    
    variance: Variance | None = eqx.field(default_factory=Hessian, kw_only=True)

    def _prepare_data(self, data: Array, n_obs: int) -> Array:
        """Helper to reshape and add intercept."""
        if data.ndim == 1:
            data = data[:, None]
        if self.add_intercept:
            ones = jnp.ones((n_obs, 1))
            data = jnp.hstack([ones, data])
        return data

    def _format_params(self, beta: Array) -> Dict[str, Array]:
        """Vector -> Dict conversion."""
        start = 0
        names = []
        if self.add_intercept:
            names.append("intercept")
            start = 1
        names.extend([f"beta_{i}" for i in range(len(beta) - start)])
        return {k: v for k, v in zip(names, beta)}

    @staticmethod
    def _to_1d(name: str, val) -> jnp.ndarray:
        arr = jnp.asarray(val)
        if arr.ndim == 0:
            return arr[None]
        elif arr.ndim == 1:
            return arr
        else:
            raise ValueError(
                f"LinearMethod expects scalar or 1D vector parameters, "
                f"but got shape {arr.shape} for parameter '{name}'."
            )

    def _reconstruct_beta(self, params: PyTree) -> Array:
        """Dict -> Vector conversion (Inverse of _format_params)."""
        beta_list = []
        if self.add_intercept:
            val = get_from_pytree(params, "intercept")
            beta_list.append(self._to_1d("intercept", val))
        
        i = 0
        while True:
            try:
                val = get_from_pytree(params, f"beta_{i}")
                beta_list.append(self._to_1d(f"beta_{i}", val))
                i += 1
            except (KeyError, AttributeError):
                break
        
        if not beta_list:
            raise ValueError("No parameters found to reconstruct beta vector.")

        return jnp.concatenate(beta_list)

    def _get_regressors(self, model: StructuralModel, n_obs: int) -> Array:
        """
        Returns the independent variables matrix (X).
        Overridden by subclasses (e.g., 2SLS).
        """
        key = getattr(self, "feature_key", "X")
        X_raw = get_from_pytree(model.data, key)
        return self._prepare_data(X_raw, n_obs)

    def compute_loss(
        self,
        result: Any | None, 
        observations: Any,
        params: PyTree,
        model: StructuralModel
    ) -> Any:
        y = get_from_pytree(observations, self.target_key, default=observations)
        y = jnp.asarray(y).ravel()
        n_obs = y.shape[0]
        
        beta = self._reconstruct_beta(params)
        X = self._get_regressors(model, n_obs)
        
        residuals = y - X @ beta
        return jnp.sum(residuals**2)


class LeastSquares(LinearMethod):
    """Ordinary Least Squares (OLS) Estimator."""
    feature_key: str = "X"

    def solve(self, model: StructuralModel, observations: Any, param_space: Any) -> EstimationResult | None:
        y = get_from_pytree(observations, self.target_key, default=observations)
        y = jnp.asarray(y).ravel()
        X_raw = get_from_pytree(model.data, self.feature_key)
        
        n_obs = y.shape[0]
        X = self._prepare_data(X_raw, n_obs)
        n_params = X.shape[1]

        # 1. Solve Normal Equations
        Xt = X.T
        XtX = Xt @ X
        op = lx.MatrixLinearOperator(XtX)
        rhs = Xt @ y
        beta = lx.linear_solve(op, rhs, solver=self.solver).value

        # 2. Compute Statistics
        residuals = y - X @ beta
        ssr = jnp.sum(residuals**2)
        sigma2 = ssr / (n_obs - n_params)
        
        XtX_inv = jnp.linalg.solve(XtX, jnp.eye(n_params))
        vcov = sigma2 * XtX_inv
        std_errors = jnp.sqrt(jnp.maximum(jnp.diag(vcov), 0.0))

        y_mean = jnp.mean(y)
        sst = jnp.sum((y - y_mean)**2)
        r_squared = 1.0 - (ssr / sst)

        # 3. Pack Results
        params_dict = self._format_params(beta)
        std_errors_dict = self._format_params(std_errors)

        return EstimationResult(
            params=params_dict,
            loss=ssr,
            success=jnp.array(True),
            std_errors=std_errors_dict,
            vcov=vcov,
            r_squared=r_squared,
            meta={
                "estimator": "OLS", 
                "method": "analytical", 
                "n_obs": n_obs
            }
        )


class TwoStageLeastSquares(LinearMethod):
    """Two-Stage Least Squares (2SLS) Estimator."""
    endog_key: str = "X"
    instrument_key: str = "Z"

    def _get_regressors(self, model: StructuralModel, n_obs: int) -> Array:
        X_raw = get_from_pytree(model.data, self.endog_key)
        Z_raw = get_from_pytree(model.data, self.instrument_key)
        
        X = self._prepare_data(X_raw, n_obs)
        Z = self._prepare_data(Z_raw, n_obs)

        ZtZ = Z.T @ Z
        ZtX = Z.T @ X
    
        gamma = jnp.linalg.solve(ZtZ, ZtX)
        X_hat = Z @ gamma
        
        return X_hat

    def solve(self, model: StructuralModel, observations: Any, param_space: Any) -> EstimationResult | None:
        y = get_from_pytree(observations, self.target_key, default=observations)
        y = jnp.asarray(y).ravel()
        X_raw = get_from_pytree(model.data, self.endog_key)
        Z_raw = get_from_pytree(model.data, self.instrument_key)

        n_obs = y.shape[0]
        X = self._prepare_data(X_raw, n_obs)
        Z = self._prepare_data(Z_raw, n_obs)
        n_params = X.shape[1]

        # Stage 1: Projection
        ZtZ = Z.T @ Z
        ZtX = Z.T @ X
        gamma = jnp.linalg.solve(ZtZ, ZtX)
        X_hat = Z @ gamma

        # Stage 2: Regression 
        XtX_hat = X_hat.T @ X_hat
        op_xhat = lx.MatrixLinearOperator(XtX_hat)
        beta = lx.linear_solve(op_xhat, X_hat.T @ y, solver=self.solver).value

        # Statistics (using original X)
        residuals = y - X @ beta
        ssr = jnp.sum(residuals**2)
        sigma2 = ssr / (n_obs - n_params)
        
        XtX_hat_inv = jnp.linalg.solve(XtX_hat, jnp.eye(n_params))
        vcov = sigma2 * XtX_hat_inv
        std_errors = jnp.sqrt(jnp.maximum(jnp.diag(vcov), 0.0))
        
        y_mean = jnp.mean(y)
        sst = jnp.sum((y - y_mean)**2)
        r_squared = 1.0 - (ssr / sst)

        return EstimationResult(
            params=self._format_params(beta),
            loss=ssr,
            success=jnp.array(True),
            std_errors=self._format_params(std_errors),
            vcov=vcov,
            r_squared=r_squared,
            meta={
                "estimator": "2SLS", 
                "method": "analytical",
                "n_obs": n_obs
            }
        )