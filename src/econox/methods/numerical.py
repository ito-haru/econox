# src/econox/methods/numerical.py
"""
Numerical estimation methods (loss-based).
Standard methods like Maximum Likelihood (MLE) and GMM.
"""

from __future__ import annotations
from typing import Any, Callable
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import PyTree, Scalar
import logging

from econox.protocols import StructuralModel
from econox.utils import get_from_pytree
from econox.config import LOSS_PENALTY
from econox.methods.base import EstimationMethod
from econox.methods.variance import Variance, Hessian

logger = logging.getLogger(__name__)


class NumericalMethod(EstimationMethod):
    """
    Base class for numerical estimation methods based on minimizing a scalar loss function.
    
    Users should implement the `compute_loss` method to define the loss function.

    Attributes:
        variance: Variance | None
            Optional variance calculation strategy for inference.
    """

    @classmethod
    def from_function(cls, func: Callable, variance: Variance | None = Hessian()) -> NumericalMethod:
        """
        Creates an `NumericalMethod` instance from a simple loss function.
        
        This factory method allows users to define objectives using a simple function 
        instead of defining a full class. The created objective will rely on numerical 
        optimization (solve returns None) and will compute standard errors using the Hessian method by default.

        Args:
            func: A function with the signature:
                  `(result, observations, params, model) -> Scalar`

        Returns:
            An instance of a dynamically created `NumericalMethod` subclass.

        Example:
            >>> def mse_loss(result, observations, params, model):
            ...     return jnp.mean((result.solution - observations) ** 2)
            >>> method = NumericalMethod.from_function(mse_loss, variance=Hessian())
        """
        # Dynamically create a subclass to wrap the function
        class WrapperMethod(NumericalMethod):
            def compute_loss(self, result, observations, params, model):
                return func(result, observations, params, model)

            def __repr__(self):
                return f"WrapperMethod({func.__name__})"

        return WrapperMethod(variance=variance)

# Decorator
def method_from_loss(
    func: Callable | None = None,
    *,
    variance: Variance | None = Hessian()
) -> NumericalMethod | Callable[[Callable], NumericalMethod]:
    """
    Decorator version of `NumericalMethod.from_function` for convenience.

    Example:
        >>> # No variance specified (default: Hessian)
        >>> @method_from_loss
        ... def mse_loss(result, observations, params, model):
        ...     return jnp.mean((result.solution - observations) ** 2)
        
        >>> # With variance specified
        >>> @method_from_loss(variance=Hessian()) 
        ... def mse_loss(result, observations, params, model):
        ...     return jnp.mean((result.solution - observations) ** 2)
    """
    def decorator(f: Callable) -> NumericalMethod:
        return NumericalMethod.from_function(f, variance=variance)

    # Case 1: @method_from_loss(func) (no arguments decorator)
    if func is not None:
        return decorator(func)
    
    # Case 2: @method_from_loss(variance=...) (with arguments decorator)
    return decorator


class MaximumLikelihood(NumericalMethod):
    """
    Standard MLE for Discrete Choice (Migration/Occupation).
    Computes Negative Log-Likelihood (NLL) based on choice probabilities.
    """
    choice_probs_key: str = "profile"  # Field name in SolverResult containing P(a|s)
    
    variance: Variance | None = eqx.field(default_factory=Hessian, kw_only=True)
    """
    Variance calculation strategy for standard errors (default: Hessian).
    """

    def compute_loss(
        self,
        result: Any | None,
        observations: Any,
        params: PyTree,
        model: StructuralModel
    ) -> Scalar:
        if result is None:
            raise ValueError("MaximumLikelihood requires a SolverResult (numerical solution), but got None.")

        choice_probs = getattr(result, self.choice_probs_key, None)

        if choice_probs is None:
            raise ValueError(
                f"SolverResult does not contain '{self.choice_probs_key}'. "
                "MaximumLikelihood requires choice probabilities (e.g. 'profile')."
            )

        # Retrieve Observed Data
        obs_states = get_from_pytree(observations, "state_indices")
        obs_choices = get_from_pytree(observations, "choice_indices")
        obs_weights = get_from_pytree(observations, "weights", default=1.0)

        sum_weights = jnp.sum(obs_weights) if jnp.ndim(obs_weights) > 0 else obs_states.shape[0]

        if choice_probs.ndim == 3:
            # Mixed model: choice_probs is (R, S, A) → SMLE
            R = choice_probs.shape[0]
            # p_selected[r, n] = P(a_n | s_n, β^(r))  shape: (R, N)
            p_selected = jnp.clip(choice_probs[:, obs_states, obs_choices], 1e-10, 1.0)
            log_p = jnp.log(p_selected)  # (R, N)

            obs_individuals = get_from_pytree(observations, "individual_indices", default=None)
            if obs_individuals is not None:
                # Panel: Σ_t log L_{n,t}(β^(r)) within individual, then average over draws.
                # num_individuals must be a concrete int for JAX tracing compatibility.
                n_ind = int(get_from_pytree(
                    observations, "num_individuals",
                    default=int(obs_individuals.max()) + 1,
                ))
                weighted_log_p = log_p * obs_weights  # (R, N)
                per_draw_per_ind = jax.vmap(
                    lambda lp: jax.ops.segment_sum(lp, obs_individuals, n_ind)
                )(weighted_log_p)  # (R, n_ind)
                log_pn = jax.scipy.special.logsumexp(per_draw_per_ind, axis=0) - jnp.log(R)
                ll = jnp.sum(log_pn)
                sum_weights = jnp.array(n_ind, dtype=log_pn.dtype)
            else:
                # Cross-sectional (T=1 per individual):
                # log P_n = logsumexp_r(log L_n(β^(r))) - log(R)  →  (N,)
                log_pn = jax.scipy.special.logsumexp(log_p, axis=0) - jnp.log(R)
                ll = jnp.sum(log_pn * obs_weights)
        else:
            # Standard: choice_probs is (S, A)
            p_selected = jnp.clip(choice_probs[obs_states, obs_choices], 1e-10, 1.0)
            ll = jnp.sum(jnp.log(p_selected) * obs_weights)

        nll = -(ll / sum_weights)
        return jnp.where(jnp.isfinite(nll), nll, jnp.array(LOSS_PENALTY))


class GaussianMomentMatch(NumericalMethod):
    """
    Fits a continuous model variable (e.g. Rent, Wage) to observed data
    assuming a Gaussian (or Log-Normal) error structure.
    """
    obs_key: str 
    model_key: str 
    scale_param_key: str 
    
    log_transform: bool = False
    variance: Variance | None = eqx.field(default=None, kw_only=True)

    def compute_loss(
        self,
        result: Any | None,
        observations: Any,
        params: PyTree,
        model: StructuralModel
    ) -> Scalar:

        if result is None:
            raise ValueError("GaussianMomentMatch requires a SolverResult (numerical solution), but got None.")

        # Try to find equilibrium data in auxiliary info, otherwise fallback to model data
        if hasattr(result, "aux") and isinstance(result.aux, dict) and "equilibrium_data" in result.aux:
            source = result.aux["equilibrium_data"]
        else:
            source = model.data 
            
        pred_val = get_from_pytree(source, self.model_key)
        obs_val = get_from_pytree(observations, self.obs_key)
        sigma = get_from_pytree(params, self.scale_param_key)
        
        if self.log_transform:
            epsilon = 1e-10
            pred_val = jnp.log(jnp.maximum(pred_val, epsilon))
            obs_val = jnp.log(jnp.maximum(obs_val, epsilon))
            
        # Compute Gaussian NLL: log(sigma) + 0.5 * ((y - mu) / sigma)^2
        sigma_safe = jnp.maximum(sigma, 1e-10)
        residuals = obs_val - pred_val
        
        nll = jnp.log(sigma_safe) + 0.5 * jnp.mean((residuals / sigma_safe) ** 2)
        robust_nll = jnp.where(jnp.isfinite(nll), nll, jnp.array(LOSS_PENALTY))

        return robust_nll

