# src/econox/methods/analytical.py
"""
Analytical linear estimation methods (OLS, 2SLS) with fixed parameter support.

This module provides:
- OLS (Ordinary Least Squares)
- 2SLS (Two-Stage Least Squares)

Both support:
- Fixed parameter constraints
- Numerical stability via QR decomposition
- Automatic fallback to numerical optimization for complex constraints

Example:
    >>> ols = LeastSquares(feature_key="X", target_key="y")
    >>> # Use Estimator (Recommended)
    >>> est = Estimator(model=model, param_space=param_space, method=ols)
    >>> result = est.fit(observations=observations)
    >>> # Or call solve() directly
    >>> result = ols.solve(model=model, observations=observations, param_space=param_space)
"""

from __future__ import annotations
from typing import Any, Dict, Tuple, List
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, PyTree

from econox.protocols import StructuralModel
from econox.utils import get_from_pytree
from econox.structures import EstimationResult, ParameterSpace
from econox.methods.base import EstimationMethod
from econox.methods.variance import Hessian, Variance


class AnalyticalParameterHandler(eqx.Module):
    """
    Handles fixed parameter constraints for analytical linear methods.
    
    This handler:
    - Separates fixed and free parameters
    - Transforms design matrices to account for fixed values
    - Reconstructs full parameter vectors and covariance matrices
    
    Attributes:
        is_fixed_mask: Boolean mask (True = fixed, False = free)
        fixed_values_vec: Values for fixed parameters
        param_names: Names of all parameters
        n_total: Total number of parameters
        n_fixed: Number of fixed parameters
        n_free: Number of free parameters (n_total - n_fixed)
    """
    is_fixed_mask: Array
    fixed_values_vec: Array
    
    param_names: List[str] = eqx.field(static=True)
    n_total: int = eqx.field(static=True)
    n_fixed: int = eqx.field(static=True)
    n_free: int = eqx.field(static=True)

    @classmethod
    def from_params(cls, param_space: ParameterSpace, param_names: List[str]) -> AnalyticalParameterHandler:
        n_total = len(param_names)
        constraints = param_space.constraints
        initials = param_space.initial_params

        is_fixed = []
        fixed_vals = []
        
        for name in param_names:
            kind = constraints.get(name, "free")
            if kind == "fixed":
                is_fixed.append(True)
                val = float(jnp.asarray(initials.get(name, 0.0)))
                fixed_vals.append(val)
            elif kind == "free":
                is_fixed.append(False)
                fixed_vals.append(0.0)
            else:
                raise ValueError(f"Constraint '{kind}' not supported.")

        is_fixed_mask = jnp.array(is_fixed, dtype=bool)
        fixed_values_vec = jnp.array(fixed_vals, dtype=float)
        n_fixed = int(jnp.sum(is_fixed_mask))
        n_free = n_total - n_fixed

        return cls(is_fixed_mask, fixed_values_vec, param_names, n_total, n_fixed, n_free)

    def transform(self, X: Array, y: Array) -> Tuple[Array, Array]:
        if self.n_fixed > 0:
            X_fixed = X[:, self.is_fixed_mask]
            y_adj = y - X_fixed @ self.fixed_values_vec[self.is_fixed_mask]
            X_free = X[:, ~self.is_fixed_mask]
            return X_free, y_adj
        return X, y

    def reconstruct(self, beta_free: Array, vcov_free: Array) -> Tuple[Array, Array]:
        beta_full = jnp.zeros(self.n_total)
        if self.n_free > 0:
            beta_full = beta_full.at[~self.is_fixed_mask].set(beta_free)
        if self.n_fixed > 0:
            beta_full = beta_full.at[self.is_fixed_mask].set(
                self.fixed_values_vec[self.is_fixed_mask]
            )

        vcov_full = jnp.zeros((self.n_total, self.n_total))
        if self.n_free > 0:
            free_indices = jnp.where(~self.is_fixed_mask)[0]
            ix_grid, iy_grid = jnp.meshgrid(free_indices, free_indices, indexing='ij')
            vcov_full = vcov_full.at[ix_grid, iy_grid].set(vcov_free)
            
        return beta_full, vcov_full


class LinearMethod(EstimationMethod):
    """
    Base class providing the template method for analytical solving.
    """
    add_intercept: bool = True
    target_key: str = "y"
    # Optional parameter names for fixed parameters
    # If None, defaults to "intercept", "beta_0", "beta_1", ...
    param_names: List[str] | None = eqx.field(default=None)
    
    variance: Variance | None = eqx.field(default_factory=Hessian, kw_only=True)

    # --- Template Method ---
    def solve(self, model: StructuralModel, observations: Any, param_space: ParameterSpace) -> EstimationResult | None:
        """
        Analytical estimation workflow with fixed parameter support.

        Workflow:
            1. Data Preparation: Extract y, construct X (and Z for 2SLS)
            2. Constraint Handling: Separate fixed/free parameters
            3. Core Estimation: Subclass-specific OLS/2SLS logic
            4. Reconstruction: Merge fixed and estimated parameters
            5. Statistics: Compute residuals, R², standard errors
    
        Args:
            model: Structural model containing data
            observations: Observation data (used to extract target y)
            param_space: Parameter space with constraints and initial values
        
        Returns:
            EstimationResult if successful, None if fallback to numerical needed
                - loss: The Sum of Squared Residuals (SSR). 
                        Note that this is the total sum, not the mean (MSE).
        
        Note:
            Returns None when constraints other than 'fixed'/'free' are present,
            signaling Estimator to use numerical optimization instead.
        """
        # A. Data Preparation
        y_raw = get_from_pytree(observations, self.target_key, default=observations)
        y = jnp.asarray(y_raw).ravel()
        n_obs = y.shape[0]
        
        # Call subclass implementation to get X (and Z)
        X, Z = self._get_design_matrices(model, n_obs)
        n_params_total = X.shape[1]

        # B. Constraint Handling
        # Determine parameter names
        if self.param_names is not None:
            if len(self.param_names) != n_params_total:
                raise ValueError(f"param_names length {len(self.param_names)} != X columns {n_params_total}")
            p_names = self.param_names
        else:
            p_names = self._get_default_param_names(n_params_total)

        try:
            handler = AnalyticalParameterHandler.from_params(param_space, p_names)
        except ValueError:
            return None # Fallback to numerical

        # Transform data (Offsetting)
        X_free, y_adj = handler.transform(X, y)
        n_free = X_free.shape[1]

        # C. Core Estimation (Subclass logic)
        if n_free > 0:
            # Call subclass implementation
            beta_free, vcov_free = self._estimate_core(X_free, y_adj, Z)
        else:
            # All fixed case
            beta_free = jnp.array([])
            vcov_free = jnp.zeros((0, 0))

        # D. Reconstruction
        beta_final, vcov_final = handler.reconstruct(beta_free, vcov_free)

        # E. Statistics
        residuals = y - X @ beta_final
        ssr = jnp.sum(residuals**2)
        y_mean = jnp.mean(y)
        sst = jnp.sum((y - y_mean)**2)
        # Robust R² calculation
        r_squared = jnp.where(
            sst > 1e-10,
            1.0 - (ssr / sst),
            jnp.nan
        )
        
        # Robust sqrt for standard errors
        std_errors = jnp.sqrt(jnp.maximum(jnp.diag(vcov_final), 0.0))

        return EstimationResult(
            params=self._format_params(beta_final, p_names),
            loss=ssr,
            success=jnp.array(True),
            std_errors=self._format_params(std_errors, p_names),
            vcov=vcov_final,
            diagnostics={"r_squared": r_squared},
            meta={
                "computation": "Analytical", 
                "estimation_method": self.__class__.__name__,
                "inference_method": "Analytical", 
                "n_obs": n_obs,
                "n_params": n_params_total,
                "n_free_params": handler.n_free,
                "n_fixed": handler.n_fixed
            },
            fixed_mask=param_space.fixed_mask
        )

    # --- Fallback for Numerical Optimization ---
    def compute_loss(self, result: Any | None, observations: Any, params: PyTree, model: StructuralModel) -> Any:
        y_raw = get_from_pytree(observations, self.target_key, default=observations)
        y = jnp.asarray(y_raw).ravel()
        n_obs = y.shape[0]
        
        # Use the same data logic as solve()
        X, Z = self._get_design_matrices(model, n_obs)

        if Z is not None:
            Q_z, _ = jnp.linalg.qr(Z, mode='reduced')
            regressor = Q_z @ (Q_z.T @ X)
        else:
            regressor = X

        beta = self._reconstruct_beta_from_dict(params)

        if regressor.shape[1] != beta.shape[0]:
            raise ValueError(
                f"Shape mismatch in {self.__class__.__name__}.compute_loss(): "
                f"regressor has {regressor.shape[1]} columns, "
                f"but beta has {beta.shape[0]} elements. "
                f"Check param_names consistency."
            )
        
        residuals = y - regressor @ beta
        return jnp.sum(residuals**2)

    # --- Abstract Methods (Subclasses MUST implement) ---
    def _get_design_matrices(self, model: StructuralModel, n_obs: int) -> Tuple[Array, Array | None]:
        raise NotImplementedError
        
    def _estimate_core(self, X: Array, y: Array, Z: Array | None = None) -> Tuple[Array, Array]:
        raise NotImplementedError

    # --- Helpers ---
    def _prepare_data(self, data: Array, n_obs: int, add_intercept: bool = False) -> Array:
        """
        Prepare data array for regression.

        Args:
            data: Input data (can be 1D or 2D)
            n_obs: Expected number of observations
            add_intercept: If True, prepend column of ones

        Returns:
            2D array with optional intercept column
        """
        if data.ndim == 1:
            data = data[:, None]

        if data.shape[0] != n_obs:
            raise ValueError(f"Data has {data.shape[0]} rows, expected {n_obs}")

        if add_intercept:
            ones = jnp.ones((n_obs, 1))
            data = jnp.hstack([ones, data])
        
        return data

    def _format_params(self, beta: Array, names: List[str]) -> Dict[str, Array]:
        return {k: v for k, v in zip(names, beta)}

    def _get_default_param_names(self, n_params: int) -> List[str]:
        start = 0
        names = []
        if self.add_intercept:
            names.append("intercept")
            start = 1
        names.extend([f"beta_{i}" for i in range(n_params - start)])
        return names

    def _reconstruct_beta_from_dict(self, params: PyTree) -> Array:
        """Returns parameter vector from PyTree based on param_names or default naming."""
        # If param_names is given, use it to extract values in order
        if self.param_names is not None:
            beta_list = [jnp.asarray(get_from_pytree(params, name)) 
                     for name in self.param_names]
        
        # Use default naming convention
        else:
            beta_list = []
            if self.add_intercept:
                val = get_from_pytree(params, "intercept")
                beta_list.append(jnp.asarray(val))
            i = 0
            while True:
                try:
                    val = get_from_pytree(params, f"beta_{i}")
                    beta_list.append(jnp.asarray(val))
                    i += 1
                except (KeyError, AttributeError):
                    break
        return jnp.stack(beta_list)


# =============================================================================
# Concrete Classes (Only logic, no flow control)
# =============================================================================

class LeastSquares(LinearMethod):
    feature_key: str = "X"

    def _get_design_matrices(self, model: StructuralModel, n_obs: int) -> Tuple[Array, Array | None]:
        X_raw = get_from_pytree(model.data, self.feature_key)
        # OLS always adds intercept if configured
        X = self._prepare_data(X_raw, n_obs, add_intercept=self.add_intercept)
        return X, None

    def _estimate_core(self, X: Array, y: Array, Z: Array | None = None) -> Tuple[Array, Array]:
        """
        Ordinary Least Squares estimation using QR decomposition for numerical stability.
        Args:
            X: Design matrix (N x K)
            y: Target vector (N,)
        Returns:
            beta: Estimated coefficients (K,)
            vcov: Variance-covariance matrix of estimates (K x K)
        """
        n_obs, n_params = X.shape
        
        # 1. QR decomposition (reduced mode)
        # Q: (N, K), R: (K, K)
        Q, R = jnp.linalg.qr(X, mode='reduced')
        
        # 2. Calculate coefficients beta
        # Solve R @ beta = Q.T @ y
        # Since R is an upper triangular matrix, solve_triangular is faster and more accurate than a regular solve
        qty = Q.T @ y
        beta = jax.scipy.linalg.solve_triangular(R, qty, lower=False)
        
        # 3. Estimate variance
        residuals = y - X @ beta
        ssr = jnp.sum(residuals**2)
        sigma2 = ssr / (n_obs - n_params)
        
        # 4. Variance-covariance matrix (R^{-1} @ R^{-T})
        # Calculate R_inv (also using the property of triangular matrix)
        R_inv = jax.scipy.linalg.solve_triangular(R, jnp.eye(n_params), lower=False)
        vcov = sigma2 * (R_inv @ R_inv.T)
        
        return beta, vcov


class TwoStageLeastSquares(LinearMethod):
    endog_key: str = "X"
    instrument_key: str = "Z"
    controls_key: str | None = None

    def _get_design_matrices(self, model: StructuralModel, n_obs: int) -> Tuple[Array, Array | None]:
        # 1. Get Components
        endog = get_from_pytree(model.data, self.endog_key)
        instrument = get_from_pytree(model.data, self.instrument_key)
        
        # Ensure 2D (No intercept yet)
        endog = self._prepare_data(endog, n_obs, add_intercept=False)
        instrument = self._prepare_data(instrument, n_obs, add_intercept=False)

        # 2. Add Controls
        if self.controls_key is not None:
            controls = get_from_pytree(model.data, self.controls_key)
            controls = self._prepare_data(controls, n_obs, add_intercept=False)
            
            # Combine: X = [endog, controls], Z = [instrument, controls]
            X = jnp.hstack([endog, controls])
            Z = jnp.hstack([instrument, controls])
        else:
            X = endog
            Z = instrument

        # 3. Add Intercept (Finally)
        # Calling prepare_data with True adds intercept to the combined matrix
        if self.add_intercept:
            ones = jnp.ones((n_obs, 1))
            X = jnp.hstack([ones, X])
            Z = jnp.hstack([ones, Z])
            
        return X, Z

    def _estimate_core(self, X: Array, y: Array, Z: Array | None = None) -> Tuple[Array, Array]:
        """
        Two-Stage Least Squares estimation using QR decomposition for numerical stability.

        Stage 1: Project X onto the column space of Z
        Stage 2: Regress y on the projected X_hat

        Note: Control variables (if present) are both in X and Z, so projecting 
        them onto Z is an identity operation. This allows efficient "whole matrix" 
        projection without separating endogenous and control variables.
        """
        if Z is None:
            raise ValueError("Instruments Z required for 2SLS.")

        n_obs, n_params = X.shape

        # === Stage 1: Compute Fitted Values ===
        # Project X onto instruments: X_hat = P_Z @ X, where P_Z = Q_z @ Q_z.T
        Q_z, _ = jnp.linalg.qr(Z, mode='reduced')
        X_hat = Q_z @ (Q_z.T @ X)

        # === Stage 2: Estimate Coefficients ===
        # Solve: X_hat @ beta = y using QR decomposition
        Q_x, R_x = jnp.linalg.qr(X_hat, mode='reduced')
        beta = jax.scipy.linalg.solve_triangular(R_x, Q_x.T @ y, lower=False)

        # === Variance Estimation ===
        # Structural errors: u = y - X @ beta (using ORIGINAL X, not X_hat)
        residuals = y - X @ beta
        sigma2 = jnp.sum(residuals**2) / (n_obs - n_params)

        # Covariance: sigma^2 * (X_hat.T @ X_hat)^-1 = sigma^2 * R_inv @ R_inv.T
        R_inv = jax.scipy.linalg.solve_triangular(R_x, jnp.eye(n_params), lower=False)
        vcov = sigma2 * (R_inv @ R_inv.T)
    
        return beta, vcov