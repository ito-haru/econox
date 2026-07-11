# src/econox/methods/variance.py
"""
Variance calculation strategies for statistical inference.
Handles the computation of standard errors and covariance matrices.
"""

from dataclasses import dataclass
from typing import Callable, Any
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import PyTree, Scalar, Array, Float

import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InferenceInputs:
    """
    Everything a :class:`Variance` strategy may need, bundled into one object.

    The estimation loop builds a single ``InferenceInputs`` and hands it to
    ``Variance.compute``; each strategy reads only the fields it needs. This keeps
    the strategy interface stable: adding a new estimator that requires more inputs
    (e.g. clustered/HAC covariance) means adding a field here, not changing every
    ``compute`` signature.

    Attributes:
        params: The estimated optimal parameters (flat free-parameter vector).
        observations: The observed data.
        total_loss_fn: The **summed** (un-normalized) loss of ``params``, i.e.
            ``mean_loss * loss_scale``. In sum form the ``1/N`` scaling is already baked
            in, so Hessian-based estimators need no separate normalization constant.
            ``None`` when the method declares no loss scale
            (:meth:`~econox.methods.base.EstimationMethod._loss_scale`); a strategy that
            needs it (e.g. :class:`Hessian`) rejects that in :meth:`Variance.validate`.
    """
    params: Array
    observations: Any
    total_loss_fn: Callable[[Array], Scalar] | None = None


class Variance(eqx.Module):
    """
    Base class for variance computation strategies.

    A strategy implements :meth:`compute` (produce the covariance) and, optionally,
    :meth:`validate` (fail fast before the expensive optimization if the chosen
    method cannot supply what the strategy needs).
    """

    def validate(self, method: Any, observations: Any) -> None:
        """
        Check that ``method`` can supply this strategy's prerequisites, and raise a
        clear error if not. Called once, *before* optimization, so a misconfiguration
        is caught up-front rather than after a costly fit.

        The default is a no-op: a strategy that degrades gracefully at compute time need
        not override it. Keeping the check inside the strategy, rather than having the
        estimation loop special-case concrete types, lets new strategies declare their own
        needs without touching the caller.

        Args:
            method: The :class:`~econox.methods.base.EstimationMethod` being fit.
            observations: The observed data.
        """
        return None

    def compute(
        self, inputs: InferenceInputs
    ) -> tuple[PyTree | None, Float[Array, "n_params n_params"] | None]:
        """
        Calculate the standard errors and variance-covariance matrix.

        Args:
            inputs: The :class:`InferenceInputs` bundle (see its docstring for fields).

        Returns:
            A tuple containing:
            - std_errors: PyTree of standard errors (same structure as ``inputs.params``).
            - vcov: Variance-covariance matrix (n_params x n_params).
        """
        return None, None


class Hessian(Variance):
    """
    Calculates variance using the inverse Hessian of the loss function.

    Standard approach for Maximum Likelihood Estimation (MLE).
    Assumes the loss function is the negative log-likelihood.
    :math:`V = H^{-1}`, where :math:`H` is the Hessian of the *summed* NLL.
    """

    def validate(self, method: Any, observations: Any) -> None:
        """
        Require the loss normalization constant. We work in sum form
        (:attr:`InferenceInputs.total_loss_fn`), which needs the method to declare
        :meth:`~econox.methods.base.EstimationMethod._loss_scale`; without it the inverse
        Hessian of a *mean* loss is mis-scaled. Refuse rather than guess a divisor and
        silently return wrong standard errors.
        """
        if method._loss_scale(observations) is None:
            raise ValueError(
                f"{type(method).__name__} does not declare a loss normalization "
                "constant (`_loss_scale` returned None), which Hessian variance "
                "requires. Declare it (e.g. `loss_scale=...` on method_from_loss, or "
                "override `_loss_scale`), or use `variance=None`."
            )

    def compute(
        self, inputs: InferenceInputs
    ) -> tuple[PyTree | None, Float[Array, "n_params n_params"] | None]:
        """
        Calculates standard errors and covariance from the inverse Hessian.

        Uses :attr:`InferenceInputs.total_loss_fn` (the summed loss), so the inverse
        Hessian is already the observed-information covariance — no extra ``1/N``
        scaling. ``validate`` guarantees ``total_loss_fn`` is present.

        Returns:
            tuple: ``(std_errors, vcov)``; ``(None, None)`` if the Hessian is singular
            or its inversion fails.
        """
        try:
            # Hessian of the *summed* loss: the 1/N scaling is already baked in, so the
            # inverse is directly the covariance (no division by a sample-size constant).
            H = jax.hessian(inputs.total_loss_fn)(inputs.params)

            # pinv for numerical stability against singular matrices.
            vcov = jnp.linalg.pinv(H)

            # Standard Errors: sqrt of the diagonal; maximum(0) guards against negative
            # diagonal elements from numerical noise.
            std_errors_flat = jnp.sqrt(jnp.maximum(jnp.diag(vcov), 0.0))

            return std_errors_flat, vcov

        except (np.linalg.LinAlgError, ValueError) as e:
            logger.warning(f"Hessian inversion failed due to numerical instability: {e}")
            return None, None
