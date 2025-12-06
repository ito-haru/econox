# src/econox/strategies/objective.py
"""
Objective function definitions for the Econox framework.
Defines criteria for comparing model outputs with observed data.
"""

from __future__ import annotations
from abc import abstractmethod
import logging
import jax
import jax.numpy as jnp
import equinox as eqx
import warnings
from typing import Any, Sequence, Callable, Optional
from jaxtyping import PyTree, Scalar, Float, Array

from econox.protocols import StructuralModel
from econox.structures import SolverResult
from econox.utils import get_from_pytree
from econox.config import LOSS_PENALTY

logger = logging.getLogger(__name__)

# =============================================================================
#  Base Objective
# =============================================================================

class Objective(eqx.Module):
    """
    Base class for all objective functions in Econox.
    
    This class serves three main purposes:
    1. **Strategy Definition**: Defines the loss function to be minimized during numerical estimation.
    2. **Analytical Solution**: Optionally provides a direct solution method (e.g., for OLS/2SLS).
    3. **Inference**: Optionally defines how to calculate standard errors (e.g., Hessian, Sandwich).

    Users can create custom objectives by subclassing this class or by using the 
    `@custom_objective` decorator.
    """

    @abstractmethod
    def compute_loss(
        self,
        result: Any | None, 
        observations: Any,
        params: PyTree, 
        model: StructuralModel
    ) -> Scalar:
        """
        Calculates the scalar loss metric to be minimized.
        
        This method is the core of the numerical estimation loop. It compares the 
        model's prediction (`result`) with the real-world data (`observations`).

        Args:
            result: The output from the Solver (e.g., `SolverResult`). 
                    If an analytical solution is being evaluated, this may be None.
            observations: Observed data to fit the model against.
            params: Current model parameters (useful for regularization terms).
            model: The structural model environment.

        Returns:
            A scalar JAX array representing the loss (e.g., Negative Log-Likelihood).
        """
        ...

    def solve(
        self,
        model: StructuralModel,
        observations: Any,
        param_space: Any
    ) -> Any | None:
        """
        Computes the analytical solution for the parameters, if available.

        This method allows the `Estimator` to bypass the numerical optimization loop 
        for models that have a closed-form solution (e.g., OLS, 2SLS).

        Args:
            model: The structural model environment.
            observations: Observed data.
            param_space: The parameter space definition.

        Returns:
            - `EstimationResult`: If an analytical solution is found.
            - `None`: If no analytical solution exists (default). 
                      The Estimator will fall back to numerical optimization using `compute_loss`.
        """
        return None

    def calculate_variance(
        self,
        loss_fn: Callable[[PyTree], Scalar],
        params: PyTree,
        observations: Any,
        num_observations: int
    ) -> Float[Array, "n_params n_params"] | None:
        """
        Calculates the variance-covariance matrix of the estimated parameters.

        Args:
            loss_fn: A differentiable function `f(params) -> loss`. 
                     This is the objective function with `result` and `model` already applied via closure.
            params: The estimated optimal parameters (in raw/unconstrained space).
            observations: The observed data (used for computing scores/gradients).
            num_observations: Number of data points (N), used for scaling the covariance.

        Returns:
            The variance-covariance matrix (n_params x n_params), or `None` if 
            variance calculation is not supported or failed.
        """
        return None

    @classmethod
    def from_function(cls, func: Callable) -> "Objective":
        """
        Creates an `Objective` instance from a simple loss function.
        
        This factory method allows users to define objectives using a simple function 
        instead of defining a full class. The created objective will rely on numerical 
        optimization (solve returns None) and will not compute standard errors by default.

        Args:
            func: A function with the signature:
                  `(result, observations, params, model) -> Scalar`

        Returns:
            An instance of a dynamically created `Objective` subclass.

        Example:
            >>> @custom_objective
            ... def mse_loss(result, observations, params, model):
            ...     return jnp.mean((result.solution - observations) ** 2)
        """
        # Dynamically create a subclass to wrap the function
        class WrapperObjective(Objective):
            def compute_loss(self, result, observations, params, model):
                return func(result, observations, params, model)

            def __repr__(self):
                return f"WrapperObjective({func.__name__})"

        return WrapperObjective()

# Alias for decorator usage
custom_objective = Objective.from_function


# =============================================================================
#  Aggregator (The Composite)
# =============================================================================

class CompositeObjective(Objective):
    """
    Combines multiple objective functions into a single scalar loss.
    Loss = sum( weight_i * loss_i )
    """
    objectives: Sequence[Objective]
    weights: Sequence[float] | None = None
        
    def compute_loss(
        self,
        result: Any | None,
        observations: Any,
        params: PyTree,
        model: StructuralModel
    ) -> Scalar:

        if result is None:
             raise ValueError("MaximumLikelihood requires a SolverResult (numerical solution), but got None.")

        current_weights = self.weights
        
        if current_weights is None:
            current_weights = [1.0] * len(self.objectives)
        elif len(current_weights) != len(self.objectives):
            raise ValueError("Weights and objectives must have the same length.")
        
        total_loss = jnp.array(0.0)
        
        for obj, w in zip(self.objectives, current_weights):
            loss = obj.compute_loss(result, observations, params, model)
            total_loss += w * loss
            
        return total_loss

    def calculate_variance(
        self,
        loss_fn: Callable[[PyTree], Scalar],
        params: PyTree,
        observations: Any,
        num_observations: int
    ) -> Optional[Float[Array, "n_params n_params"]]:
        """
        Calculates the variance-covariance matrix of the composite objective.
        
        Currently returns None because standard errors for composite objectives 
        (e.g. Micro + Macro) cannot be reliably estimated using the simple inverse Hessian 
        method without correct relative weighting or sandwich estimators.
        """
        warnings.warn(
            "Standard error calculation for CompositeObjective is currently disabled "
            "in this version to prevent misleading results. "
            "Please rely on parameter point estimates or use bootstrap methods manually.",
            UserWarning
        )

        return None


# =============================================================================
#  Discrete Choice Objective (Maximum Likelihood)
# =============================================================================

class MaximumLikelihood(Objective):
    """
    Standard MLE for Discrete Choice (Migration/Occupation).
    Computes Negative Log-Likelihood (NLL) based on choice probabilities.
    """
    choice_probs_key: str = "profile"  # Field name in SolverResult containing P(a|s)

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

        p_selected = choice_probs[obs_states, obs_choices]
        
        # Clip for numerical stability to avoid log(0)
        p_selected = jnp.clip(p_selected, 1e-10, 1.0)
        
        # Calculate weighted log-likelihood
        sum_weights = jnp.sum(obs_weights) if jnp.ndim(obs_weights) > 0 else obs_states.shape[0]
        ll_choice = jnp.sum(jnp.log(p_selected) * obs_weights)
        
        nll = - (ll_choice / sum_weights)
        robust_nll = jnp.where(jnp.isfinite(nll), nll, jnp.array(LOSS_PENALTY))

        return robust_nll

    def calculate_variance(
        self,
        loss_fn: Callable[[PyTree], Scalar],
        params: PyTree,
        observations: Any,
        num_observations: int
    ) -> Optional[Float[Array, "n_params n_params"]]:
        """
        Calculates the variance-covariance matrix using the inverse Hessian of the NLL.
        V = H^{-1} / N (Assuming loss_fn returns mean NLL)
        """
        try:
            # 1. Compute Hessian of the loss function at the optimum
            H = jax.hessian(loss_fn)(params)
            
            # 2. Invert the Hessian
            # Using pinv for numerical stability
            vcov = jnp.linalg.pinv(H) / num_observations
            return vcov
            
        except Exception as e:
            logger.warning(f"Failed to compute Hessian for standard errors in MaximumLikelihood: {e}")
            return None


# =============================================================================
#  Continuous Variable Objective (Gaussian Moment Matching)
# =============================================================================

class GaussianMomentMatch(Objective):
    """
    Fits a continuous model variable (e.g. Rent, Wage) to observed data
    assuming a Gaussian (or Log-Normal) error structure.
    """
    obs_key: str
    model_key: str
    scale_param_key: str # Key in 'params' for standard deviation (sigma)
    log_transform: bool = False
    
    def compute_loss(
        self,
        result: Any | None,
        observations: Any,
        params: PyTree,
        model: StructuralModel
    ) -> Scalar:

        if result is None:
             raise ValueError("GaussianMomentMatch requires a SolverResult (numerical solution), but got None.")

        if "equilibrium_data" in result.aux:
            source = result.aux["equilibrium_data"]
        else:
            source = model.data # Fallback to static model data
            
        pred_val = get_from_pytree(source, self.model_key)
        obs_val = get_from_pytree(observations, self.obs_key)
        sigma = get_from_pytree(params, self.scale_param_key)
        
        if self.log_transform:
            epsilon = 1e-10
            pred_val = jnp.log(jnp.maximum(pred_val, epsilon))
            obs_val = jnp.log(jnp.maximum(obs_val, epsilon))
            
        # Compute Gaussian NLL
        sigma_safe = jnp.maximum(sigma, 1e-10)
        residuals = obs_val - pred_val
        
        nll = jnp.log(sigma_safe) + 0.5 * jnp.mean((residuals / sigma_safe) ** 2)
        robust_nll = jnp.where(jnp.isfinite(nll), nll, jnp.array(LOSS_PENALTY))

        return robust_nll

    def calculate_variance(
        self,
        loss_fn: Callable[[PyTree], Scalar],
        params: PyTree,
        observations: Any,
        num_observations: int
    ) -> Optional[Float[Array, "n_params n_params"]]:
        """
        Standard variance calculation for GMM/OLS logic not yet implemented.
        Returns None to indicate lack of standard errors.
        """
        warnings.warn(
            "Standard errors are not yet implemented for GaussianMomentMatch. "
            "Returning None.", 
            UserWarning
        )
        return None

# =============================================================================
#  Generalized Method of Moments (GMM) Objective
# =============================================================================
class GeneralizedMethodOfMoments(Objective):
    """
    GMM Objective for matching model-implied moments to observed moments.
    """
    model_moments_key: str
    observed_moments_key: str
    weights_matrix: Optional[Float[Array, "n_moments n_moments"]] = None

    def compute_loss(
        self,
        result: Any | None,
        observations: Any,
        params: PyTree,
        model: StructuralModel
    ) -> Scalar:
        raise NotImplementedError(
            "GMM Objective is not yet implemented."
        )

    def calculate_variance(
        self,
        loss_fn: Callable[[PyTree], Scalar],
        params: PyTree,
        observations: Any,
        num_observations: int
    ) -> Optional[Float[Array, "n_params n_params"]]:

        raise NotImplementedError(
            "Variance calculation for GMM Objective is not yet implemented."
        )
        