# src/econox/methods/variance.py
"""
Variance calculation strategies for statistical inference.
Handles the computation of standard errors and covariance matrices.
"""

from dataclasses import dataclass
from typing import Callable, Any
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
        per_obs_loss_fn: The **un-weighted** per-observation loss contributions as a
            function of ``params`` (shape ``(n_obs,)``), or ``None`` when the method
            cannot decompose its objective. Required by robust estimators
            (:class:`Sandwich`) for the "meat" term.
        weights: Optional 1-D **frequency weights** (replication counts) aligned with
            ``per_obs_loss_fn``. ``None`` means unweighted.
    """
    params: Array
    observations: Any
    total_loss_fn: Callable[[Array], Scalar] | None = None
    per_obs_loss_fn: Callable[[Array], Array] | None = None
    weights: Array | None = None


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

        The default is a no-op: a strategy with no prerequisites need not override it.
        Keeping the check inside the strategy, rather than having the estimation loop
        special-case concrete types, lets new strategies declare their own needs without
        touching the caller.

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
                "override `_loss_scale`), or use `variance=None` / a sum-form strategy "
                "like Sandwich."
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
        # Hessian of the *summed* loss: the 1/N scaling is already baked in, so the
        # inverse is directly the covariance (no division by a sample-size constant).
        H = jax.hessian(inputs.total_loss_fn)(inputs.params)

        # pinv for numerical stability against singular matrices.
        vcov = jnp.linalg.pinv(H)

        # pinv never raises on a singular/ill-conditioned Hessian; a degenerate or
        # non-finite H instead propagates as NaN/inf through the inverse. Detect that
        # explicitly and signal failure with (None, None) rather than returning a
        # silently-invalid covariance.
        if not bool(jnp.all(jnp.isfinite(vcov))):
            logger.warning("Hessian inversion produced non-finite values; returning None.")
            return None, None

        # Standard Errors: sqrt of the diagonal; maximum(0) guards against negative
        # diagonal elements from numerical noise.
        std_errors_flat = jnp.sqrt(jnp.maximum(jnp.diag(vcov), 0.0))

        return std_errors_flat, vcov


class Sandwich(Variance):
    r"""
    Robust (Huber-White / QMLE) "sandwich" variance estimator.

    Combines the inverse Hessian ("bread") with the outer product of per-observation
    scores ("meat") to obtain standard errors that are consistent even under
    distributional misspecification or heteroskedasticity. Also known as the robust
    Hessian estimator.

    :math:`V = H^{-1} B H^{-1}`

    where :math:`H = \sum_i \partial^2 \ell_i` is the Hessian of the *summed* loss
    (bread) and :math:`B = \sum_i g_i g_i^\top` is the outer product of the
    per-observation scores :math:`g_i = \partial \ell_i` (meat).

    Working in this un-normalized (sum) form means no explicit sample-size or
    normalization constant is needed: the :math:`1/N` scaling emerges naturally from
    the magnitudes of the bread and meat. Both are derived from the *same* un-weighted
    per-observation scores, so they are mutually consistent by construction.

    Under correct specification the information matrix equality gives :math:`H \approx B`,
    so :math:`V \approx H^{-1}`, matching the observed-information (:class:`Hessian`)
    estimator.

    **Weights are treated as frequency weights** (replication counts). Both the bread
    and the meat scale *linearly* in the weights,

    .. math:: H = \sum_i w_i \partial^2 \ell_i, \qquad B = \sum_i w_i\, g_i g_i^\top,

    so a weight of :math:`w_i` behaves exactly like :math:`w_i` identical copies of
    observation :math:`i`, consistent with the frequency-weight semantics used
    throughout the library.

    .. note::
       Sampling / probability weights (Horvitz--Thompson, :math:`w_i = 1/\pi_i`) are
       **not** supported: they would need a *quadratic* meat
       :math:`B = \sum_i w_i^2 g_i g_i^\top`, deliberately left disabled so the
       frequency-weight contract stays unambiguous.

    Requires the method to supply per-observation scores
    (:meth:`~econox.methods.base.EstimationMethod.compute_loss_per_obs`). This is checked
    up-front in :meth:`validate`, so a method that cannot provide them is rejected before
    the costly fit rather than silently yielding ``None`` standard errors afterwards.
    """

    def validate(self, method: Any, observations: Any) -> None:
        """
        Require per-observation scores. The sandwich "meat"
        :math:`B = \\sum_i g_i g_i^\\top` is the outer product of per-observation score
        contributions, so the method must decompose its objective additively over
        observations via
        :meth:`~econox.methods.base.EstimationMethod.compute_loss_per_obs`. Reject up-front
        rather than let the user pay for a fit and only then discover the robust standard
        errors could not be computed.
        """
        if not method._supports_per_obs_loss():
            raise ValueError(
                f"{type(method).__name__} does not provide a per-observation loss "
                "(`compute_loss_per_obs` is not overridden), which Sandwich (robust) "
                "variance requires for the score / meat term. Use `variance=Hessian()` "
                "or `variance=None`, or implement `compute_loss_per_obs`."
            )

    def compute(
        self, inputs: InferenceInputs
    ) -> tuple[PyTree | None, Float[Array, "n_params n_params"] | None]:
        """
        Calculates robust standard errors and covariance via the sandwich formula.

        Reads only ``per_obs_loss_fn``, ``params`` and ``weights`` from ``inputs``:
        both bread and meat come from the same un-weighted per-observation scores, so
        the estimate is independent of how the aggregate objective was scaled.

        Returns:
            tuple: ``(std_errors, vcov)``; ``(None, None)`` if no per-observation loss
            is available or the computation fails numerically.
        """
        # :meth:`validate` already rejects methods without per-observation scores before
        # the fit, so a missing `per_obs_loss_fn` here means `compute` was called directly
        # (bypassing the estimation loop). Keep a defensive guard rather than crash.
        if inputs.per_obs_loss_fn is None:
            logger.warning(
                "Sandwich (robust) variance requires a per-observation loss, but the "
                "estimation method does not provide one. Returning None."
            )
            return None, None

        # Both bread and meat are built from `per_obs_loss_fn`, NOT from
        # `total_loss_fn`. At a finite point `jax.hessian(total_loss_fn)` equals the
        # weighted bread below, but Sandwich must also work when the method declares no
        # loss scale (`_loss_scale` -> None, so `total_loss_fn` is None): deriving
        # everything from the raw per-obs scores keeps it independent of that
        # normalization, and avoids the LOSS_PENALTY clamp `total_loss_fn` carries.
        # Do not "simplify" by reusing `total_loss_fn` for the bread.
        #
        # Scores: un-weighted per-observation gradients. jacfwd is cheaper when
        # n_params << n_obs. G has shape (n_obs, n_params); row i is g_i = d l_i/d p.
        G = jax.jacfwd(inputs.per_obs_loss_fn)(inputs.params)

        if inputs.weights is None:
            # Bread: H = sum_i d^2 l_i / d params^2. Meat: B = sum_i g_i g_i^T.
            H = jax.hessian(lambda p: jnp.sum(inputs.per_obs_loss_fn(p)))(inputs.params)
            B = G.T @ G
        else:
            w = jnp.asarray(inputs.weights)
            # Bread: H = sum_i w_i d^2 l_i / d params^2 (Hessian of the weighted sum).
            H = jax.hessian(lambda p: jnp.sum(w * inputs.per_obs_loss_fn(p)))(inputs.params)
            # Meat: B = sum_i w_i g_i g_i^T  (linear in the frequency weights).
            B = (G * w[:, None]).T @ G

        # Sandwich: V = H^{-1} B H^{-1}. The 1/N scaling emerges from the magnitudes
        # of H (~N) and B (~N); pinv guards against singular matrices.
        H_inv = jnp.linalg.pinv(H)
        vcov = H_inv @ B @ H_inv

        # pinv never raises on a singular/ill-conditioned bread; a degenerate or
        # non-finite H (or scores) instead propagates as NaN/inf through the sandwich.
        # Detect that explicitly and signal failure with (None, None) rather than
        # returning a silently-invalid covariance.
        if not bool(jnp.all(jnp.isfinite(vcov))):
            logger.warning("Sandwich variance produced non-finite values; returning None.")
            return None, None

        # Standard Errors: sqrt of diagonal, clamped for numerical noise.
        std_errors_flat = jnp.sqrt(jnp.maximum(jnp.diag(vcov), 0.0))

        return std_errors_flat, vcov
