# src/econox/methods/variance.py
"""
Variance calculation strategies for statistical inference.
Handles the computation of standard errors and covariance matrices.
"""

from typing import Callable, Any
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import PyTree, Scalar, Array, Float

class Variance(eqx.Module):
    """
    Base class for variance computation strategies.
    """
    def compute(
        self,
        loss_fn: Callable[[PyTree], Scalar],
        params: PyTree,
        observations: Any,
        num_observations: int
    ) -> tuple[PyTree | None, Float[Array, "n_params n_params"] | None]:
        """
        Calculates the standard errors and variance-covariance matrix.

        Args:
            loss_fn: A differentiable function `f(params) -> loss`.
                     The objective function with `result` and `model` applied via closure.
            params: The estimated optimal parameters.
            observations: The observed data.
            num_observations: Number of data points (N).

        Returns:
            A tuple containing:
            - std_errors: PyTree of standard errors (same structure as params).
            - vcov: Variance-covariance matrix (n_params x n_params).
        """
        return None, None


class Hessian(Variance):
    """
    Calculates variance using the inverse Hessian of the loss function.
    
    Standard approach for Maximum Likelihood Estimation (MLE).
    Assumes the loss function is the negative log-likelihood.
    V = H^{-1} / N
    """
    
    def compute(
        self,
        loss_fn: Callable[[PyTree], Scalar],
        params: PyTree,
        observations: Any,
        num_observations: int
    ) -> tuple[PyTree | None, Float[Array, "n_params n_params"] | None]:
        
        try:
            # 1. Compute Hessian of the loss function at the optimum
            H = jax.hessian(loss_fn)(params)
            
            # 2. Invert the Hessian
            # Using pinv for numerical stability against singular matrices
            vcov = jnp.linalg.pinv(H) / num_observations
            
            # 3. Extract Standard Errors (sqrt of diagonal elements)
            # Use maximum(0) to avoid NaNs from negative diagonal elements (numerical noise)
            std_errors_flat = jnp.sqrt(jnp.maximum(jnp.diag(vcov), 0.0))
            
            # Note: We return the flat std_errors here. 
            # The Estimator will verify reshaping, but for now we return the flat array 
            # or we can leave reconstruction to the Estimator logic which holds the structure.
            # In this design, Estimator handles the PyTree reconstruction using unravel_fn.
            
            return std_errors_flat, vcov
            
        except Exception as e:
            # Return None if Hessian computation fails (e.g. out of memory, singular)
            return None, None