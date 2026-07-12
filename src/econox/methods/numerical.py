# src/econox/methods/numerical.py
"""
Numerical estimation methods (loss-based).
Standard methods like Maximum Likelihood (MLE) and GMM.
"""

from __future__ import annotations
from typing import Any, Callable
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import PyTree, Scalar, Array
import logging

from econox.protocols import StructuralModel
from econox.utils import get_from_pytree
from econox.config import LOSS_PENALTY
from econox.methods.base import EstimationMethod
from econox.methods.variance import Variance, Hessian, Sandwich

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
    def from_function(
        cls,
        func: Callable,
        variance: Variance | None = None,
        loss_scale: Callable[[Any], Scalar] | float | None = None,
    ) -> NumericalMethod:
        """
        Creates an `NumericalMethod` instance from a simple loss function.

        This factory method allows users to define objectives using a simple function
        instead of defining a full class. The created objective relies on numerical
        optimization (solve returns None).

        Inference is **off by default** (``variance=None``): an arbitrary loss function
        has no guaranteed statistical interpretation. Pass ``variance=Hessian()`` to opt
        in — it also needs ``loss_scale`` (see below), so the two go together.

        Args:
            func: A function with the signature:
                  `(result, observations, params, model) -> Scalar`
            variance: Variance strategy for inference. ``None`` (default) means no
                  standard errors are computed.
            loss_scale: The constant `func` normalizes by (its divisor), needed by a
                  Hessian ``variance`` to recover the un-normalized objective. A constant,
                  or a callable ``(observations) -> scalar`` for a data-dependent divisor
                  (e.g. the observation count of a mean loss). Only consumed by Hessian
                  variance, which raises rather than guess if it is undeclared.

        Returns:
            An instance of a dynamically created `NumericalMethod` subclass.

        Example:
            >>> def mse_loss(result, observations, params, model):
            ...     return jnp.mean((result.solution - observations["y"]) ** 2)
            >>> # No inference (default)
            >>> method = NumericalMethod.from_function(mse_loss)
            >>> # Opt into Hessian standard errors, declaring the divisor
            >>> method = NumericalMethod.from_function(
            ...     mse_loss, variance=Hessian(), loss_scale=lambda obs: obs["y"].shape[0]
            ... )
        """
        # Dynamically create a subclass to wrap the function
        class WrapperMethod(NumericalMethod):
            def compute_loss(self, result, observations, params, model):
                return func(result, observations, params, model)

            def _loss_scale(self, observations):
                # Report the user-declared divisor; ``None`` (default) leaves it
                # undeclared and disables the Hessian path (see base `_loss_scale`).
                if loss_scale is None:
                    return None
                if callable(loss_scale):
                    return loss_scale(observations)
                return loss_scale

            def __repr__(self):
                return f"WrapperMethod({func.__name__})"

        return WrapperMethod(variance=variance)

# Decorator
def method_from_loss(
    func: Callable | None = None,
    *,
    variance: Variance | None = None,
    loss_scale: Callable[[Any], Scalar] | float | None = None,
) -> NumericalMethod | Callable[[Callable], NumericalMethod]:
    """
    Decorator version of `NumericalMethod.from_function` for convenience.

    Inference is off by default (``variance=None``). See
    :meth:`NumericalMethod.from_function` for the ``loss_scale`` contract: when a
    Hessian variance is requested it needs the loss divisor declared, else it raises
    rather than guessing.

    Example:
        >>> # No inference (default)
        >>> @method_from_loss
        ... def mse_loss(result, observations, params, model):
        ...     return jnp.mean((result.solution - observations["y"]) ** 2)

        >>> # Opt into Hessian standard errors, declaring the divisor
        >>> @method_from_loss(variance=Hessian(), loss_scale=lambda obs: obs["y"].shape[0])
        ... def mse_loss(result, observations, params, model):
        ...     return jnp.mean((result.solution - observations["y"]) ** 2)
    """
    def decorator(f: Callable) -> NumericalMethod:
        return NumericalMethod.from_function(f, variance=variance, loss_scale=loss_scale)

    # Case 1: @method_from_loss(func) (no arguments decorator)
    if func is not None:
        return decorator(func)
    
    # Case 2: @method_from_loss(variance=...) (with arguments decorator)
    return decorator


class MaximumLikelihood(NumericalMethod):
    """
    Standard MLE for Discrete Choice (Migration/Occupation).
    Computes Negative Log-Likelihood (NLL) based on choice probabilities.

    Observations:
        The ``observations`` container (a dict or any PyTree readable by
        :func:`~econox.utils.get_from_pytree`) must provide:

        * ``state_indices`` (**required**): 1-D integer array indexing the observed
          state of each observation.
        * ``choice_indices`` (**required**): 1-D integer array (same length as
          ``state_indices``) indexing the chosen action of each observation.
        * ``weights`` (*optional*): 1-D float array of per-observation **frequency
          weights** (replication counts), same length as ``state_indices``; omitted means
          equal weighting. Estimates and standard errors (including
          :class:`~econox.methods.variance.Sandwich`) behave as if each observation
          appeared ``weight`` times, and ``sum(weights)`` is the effective sample size.
          Sampling / probability weights are not supported. Must be 1-D or omitted (see
          :meth:`~econox.methods.base.EstimationMethod._get_validated_weights`).

        The choice probabilities themselves come from the solver output (``result``),
        under the field named by :attr:`choice_probs_key`; they are indexed as
        ``choice_probs[state_indices, choice_indices]``.
    """
    choice_probs_key: str = "profile"  # Field name in SolverResult containing P(a|s)

    variance: Variance | None = eqx.field(default_factory=Sandwich, kw_only=True)
    """
    Variance calculation strategy for standard errors (default: Sandwich / robust Hessian).

    Pass ``variance=Hessian()`` explicitly to use the non-robust inverse-Hessian estimator.
    """

    def compute_loss(
        self,
        result: Any | None,
        observations: Any,
        params: PyTree,
        model: StructuralModel
    ) -> Scalar:
        # Aggregate NLL is just the weighted, normalized sum of the per-obs scores.
        # Reusing them as the single source of truth keeps this loss consistent with the
        # sandwich bread/meat, which are built from the same scores.
        per_obs = self.compute_loss_per_obs(result, observations, params, model)

        # Weight here, not in compute_loss_per_obs: the scores must stay un-weighted so
        # the sandwich meat scales linearly in the weights.
        obs_weights = self._get_validated_weights(observations)
        numerator = jnp.sum(per_obs if obs_weights is None else obs_weights * per_obs)

        # Mean NLL = sum_i (w_i * -log P_i) / (sum of weights).
        nll = numerator / self._loss_scale(observations)

        # Return huge penalty if NLL is NaN/Inf.
        return jnp.where(jnp.isfinite(nll), nll, jnp.array(LOSS_PENALTY))

    def _loss_scale(self, observations: Any) -> Scalar:
        """
        Divisor used by :meth:`compute_loss`: the sum of (frequency) weights, or the
        observation count when weights are absent. Shared by the loss and the variance
        estimators, so dividing the summed per-obs loss by exactly this recovers the
        un-normalized objective. Under frequency weights this equals the effective
        sample size. See :meth:`EstimationMethod._loss_scale`.
        """
        obs_weights = self._get_validated_weights(observations)
        if obs_weights is not None:
            return jnp.sum(obs_weights)
        obs_states = get_from_pytree(observations, "state_indices")
        return obs_states.shape[0]

    def compute_loss_per_obs(
        self,
        result: Any | None,
        observations: Any,
        params: PyTree,
        model: StructuralModel
    ) -> Array:
        """
        Un-weighted per-observation NLL contributions, the source of the scores used by
        robust / sandwich variance.

        Returns the vector :math:`\\ell_i = -\\log P_i`, **un-weighted**: weighting is
        left to the callers so the sandwich meat scales *linearly* (not quadratically) in
        the weights. Kept raw (no ``LOSS_PENALTY`` clamp, no division by N) so the scores
        are exact.
        """
        if result is None:
            raise ValueError("MaximumLikelihood requires a SolverResult (numerical solution), but got None.")

        choice_probs = getattr(result, self.choice_probs_key, None)

        if choice_probs is None:
            raise ValueError(
                f"SolverResult does not contain '{self.choice_probs_key}'. "
                "MaximumLikelihood requires choice probabilities (e.g. 'profile')."
            )

        obs_states = get_from_pytree(observations, "state_indices")
        obs_choices = get_from_pytree(observations, "choice_indices")

        p_selected = choice_probs[obs_states, obs_choices]
        p_selected = jnp.clip(p_selected, 1e-10, 1.0)

        # Un-weighted per-observation NLL. Weighting is intentionally deferred to the
        # callers (see docstring) so the raw scores g_i stay available for the meat.
        return -jnp.log(p_selected)


class GaussianMomentMatch(NumericalMethod):
    """
    Fits a continuous model variable (e.g. Rent, Wage) to observed data
    assuming a Gaussian (or Log-Normal) error structure.

    Unlike :class:`MaximumLikelihood`, the data keys are not fixed names but are
    configured per instance via the fields below, so the same method can target any
    observed/predicted variable pair.

    Observations:
        ``observations`` must provide the observed values under the key named by
        :attr:`obs_key`. The matching model prediction is read under :attr:`model_key`
        from ``result.aux["equilibrium_data"]`` when present, otherwise from
        ``model.data``. The noise scale is read under :attr:`scale_param_key` from the
        estimated ``params``.

    Attributes:
        obs_key: Key of the observed values in ``observations``.
        model_key: Key of the predicted values in the equilibrium data / ``model.data``.
        scale_param_key: Key of the Gaussian scale (sigma) in ``params``.
        log_transform: If True, compare values in log space (Log-Normal error).
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

