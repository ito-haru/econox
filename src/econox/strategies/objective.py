# src/econox/strategies/objective.py
"""
Objective function definitions for the Econox framework.
Defines criteria for comparing model outputs with observed data.
"""

import logging
import jax
import jax.numpy as jnp
import equinox as eqx
import warnings
from typing import Any, Sequence, Callable, Optional
from jaxtyping import PyTree, Scalar, Float, Array

from econox.protocols import StructuralModel, Objective
from econox.structures import SolverResult
from econox.utils import get_from_pytree
from econox.config import LOSS_PENALTY

logger = logging.getLogger(__name__)

# =============================================================================
# 1. Aggregator (The Composite)
# =============================================================================

class CompositeObjective(eqx.Module):
    """
    Combines multiple objective functions into a single scalar loss.
    Loss = sum( weight_i * loss_i )
    """
    objectives: Sequence[Objective]
    weights: Sequence[float] | None = None

    def __init__(
        self, 
        objectives: Sequence[Objective], 
        weights: Sequence[float] | None = None
    ):
        self.objectives = objectives
        
    def compute_loss(
        self,
        result: SolverResult,
        observations: Any,
        params: PyTree,
        model: StructuralModel
    ) -> Scalar:
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
# 2. Discrete Choice Objective (Maximum Likelihood)
# =============================================================================

class MaximumLikelihood(eqx.Module):
    """
    Standard MLE for Discrete Choice (Migration/Occupation).
    Computes Negative Log-Likelihood (NLL) based on choice probabilities.
    """
    choice_probs_key: str = "profile"  # Field name in SolverResult containing P(a|s)

    def compute_loss(
        self,
        result: SolverResult,
        observations: Any,
        params: PyTree,
        model: StructuralModel
    ) -> Scalar:

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
# 3. Continuous Variable Objective (Gaussian Moment Matching)
# =============================================================================

class GaussianMomentMatch(eqx.Module):
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
        result: SolverResult,
        observations: Any,
        params: PyTree,
        model: StructuralModel
    ) -> Scalar:

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
# 4. Generalized Method of Moments (GMM) Objective
# =============================================================================
class GeneralizedMethodOfMoments(eqx.Module):
    """
    GMM Objective for matching model-implied moments to observed moments.
    """
    model_moments_key: str
    observed_moments_key: str
    weights_matrix: Optional[Float[Array, "n_moments n_moments"]] = None

    def compute_loss(
        self,
        result: SolverResult,
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
        