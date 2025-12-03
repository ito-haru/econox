# src/econox/workflow/estimator.py
"""
Estimator module for the Econox framework.
Orchestrates the estimation process by connecting Data, Model, Solver, and Objective.
"""

import logging
import jax
import jax.numpy as jnp
import equinox as eqx
from jax.flatten_util import ravel_pytree
from typing import Any
from jaxtyping import PyTree, Scalar, Array

from econox.protocols import FeedbackMechanism, StructuralModel, Solver, Objective, Utility, Distribution
from econox.structures import ParameterSpace, EstimationResult
from econox.strategies.numerical import Minimizer, MinimizerResult
from econox.config import LOSS_PENALTY
from econox.utils import get_from_pytree

logger = logging.getLogger(__name__)


class Estimator(eqx.Module):
    """
    Orchestrates the structural estimation process.
    
    Handles:
    1. Parameter transformation (Raw <-> Constrained) via ParameterSpace.
    2. Solving the model (Single run or Batched Simulation/SMM).
    3. Evaluating the objective function (Loss calculation).
    4. Minimizing the loss using an Optimizer.

    Attributes:
        model: StructuralModel - The structural model to estimate.
        param_space: ParameterSpace - Parameter transformation and constraints.
        solver: Solver - Solver to compute model solutions.
        utility: Utility - Utility specification for the model.
        dist: Distribution - Distribution specification for the model.
        objective: Objective - Objective function to evaluate fit.
        optimizer: Minimizer - Optimization strategy for minimizing the loss.
        feedback: FeedbackMechanism | None - Optional feedback mechanism during solving.
        num_simulations: int | None - Number of simulations for SMM (if applicable).
        key: jax.Array | None - Random key for stochastic simulations. (Currently unused)
        verbose: bool - If True, enables detailed logging for debugging.
    """
    model: StructuralModel
    param_space: ParameterSpace
    solver: Solver
    utility: Utility
    dist: Distribution
    objective: Objective
    optimizer: Minimizer = eqx.field(default_factory=Minimizer)
    feedback: FeedbackMechanism | None = None
    # SMM Configuration
    num_simulations: int | None = eqx.field(default=None, static=True)
    key: jax.Array | None = None
    
    # Debugging
    verbose: bool = eqx.field(default=False, static=True)

    def fit(
        self,
        observations: Any, 
        initial_params: dict | None = None,
        sample_size: int | None = None
        ) -> EstimationResult:
        """
        Estimates the model parameters to minimize the objective function.

        Args:
            observations: Observed data to match (passed to Objective).
            initial_params: Dictionary of initial parameter values (Constrained space).
                            If None, uses initial_params from ParameterSpace.
            sample_size: Effective sample size for variance calculations (if needed).

        Returns:
            EstimationResult containing estimated parameters, final loss, and details.
        """
        # Prepare Initial Parameters
        # Convert constrained initial params to raw (unconstrained) space for the optimizer
        if initial_params is None:
            constrained_init = self.param_space.initial_params
        else:
            constrained_init = initial_params
            
        raw_init = self.param_space.inverse_transform(constrained_init)


        # Sample Size Handling
        sum_weights = self._get_sum_weights(observations)
        final_N = None
    
        if sample_size is not None:
            # Use provided sample size
            final_N = sample_size
            # Warn if provided sample size differs from sum of weights
            if sum_weights is not None:
                if abs(final_N - sum_weights) > 1.0: 
                    logger.warning(
                        f"Provided sample_size ({final_N}) differs from sum of weights ({sum_weights}). "
                        "Using provided sample_size."
                    )
        else:
            # Try to infer sample size from weights in observations
            if sum_weights is not None:
                final_N = int(sum_weights)
                logger.info(f"Using sum of weights (N={final_N}) as sample size.")
            else:
                # Unable to determine sample size
                raise ValueError(
                    "Sample size could not be determined.\n"
                    "Please provide `sample_size` argument explicitly, or ensure `observations` contains 'weights'."
                )

        # ----------------------------------------

        # Define Loss Function (The core pipeline)
        @eqx.filter_jit
        def loss_fn(raw_params: PyTree, args: Any) -> Scalar:
            # A. Transform Parameters: Raw (Optimizer) -> Constrained (Model)
            params = self.param_space.transform(raw_params)

            # Debug output if verbose
            if self.verbose:
                jax.debug.print("Estimator: Checking Params: {}", params)

            # B. Solve the Model
            # Case 1: SMM (Simulated Method of Moments) - Batched Execution
            if self.num_simulations is not None:
                raise NotImplementedError(
                    "SMM (Simulated Method of Moments) is not yet implemented. "
                    "Please set num_simulations=None to use standard estimation."
                )

            # Case 2: Standard Estimation (MLE / NFXP) - Single Execution
            else:
                result = self.solver.solve(
                    params, 
                    self.model, 
                    self.utility, 
                    self.dist,
                    feedback=self.feedback
                )

            # C. Evaluate Objective
            loss = self.objective.compute_loss(result, observations, params, self.model)
            
            # Debug output if verbose
            if self.verbose:
                jax.debug.print("Estimator: Loss: {}", loss)
                
            return loss

        # Run Optimization
        logger.info(f"Starting estimation with {self.optimizer.__class__.__name__}...")
        opt_result: MinimizerResult = self.optimizer.minimize(
            loss_fn=loss_fn,
            init_params=raw_init,
            args=observations # Passed as args to loss_fn
        )

        # 4. Process Results
        final_raw_params = opt_result.params
        final_constrained_params = self.param_space.transform(final_raw_params)
        final_loss = opt_result.loss
        
        final_solver_result = self.solver.solve(
            final_constrained_params, self.model, self.utility, self.dist, self.feedback
        )

        std_errors = None
        vcov = None

        if opt_result.success:
            try:
                # Handle parameter space transformation with fixed parameter masking  
                # A. Flatten Raw Params
                flat_raw_params, unravel_fn = ravel_pytree(final_raw_params)
                
                # B. Handle Fixed Parameters Masking
                fixed_mask_pytree = self.param_space.fixed_mask
                flat_fixed_mask, _ = ravel_pytree(fixed_mask_pytree)
                is_free = jnp.logical_not(flat_fixed_mask)
                flat_free_params = flat_raw_params[is_free]
                
                # C. Define wrapper loss for free params only
                def loss_fn_for_inference(free_params_vec: Array) -> Scalar:
                    full_params_vec = flat_raw_params.at[is_free].set(free_params_vec)
                    raw_pytree = unravel_fn(full_params_vec)
                    return loss_fn(raw_pytree, observations)

                # D. Compute Variance in FREE Raw Space
                vcov_free = self.objective.calculate_variance(
                    loss_fn=loss_fn_for_inference,
                    params=flat_free_params,
                    observations=observations,
                    num_observations=final_N
                )

                if vcov_free is not None:
                    # E. Expand to Full Raw Space (N x N)
                    n_total = flat_raw_params.shape[0]
                    vcov_raw = jnp.zeros((n_total, n_total))
                    
                    free_indices = jnp.where(is_free)[0]
                    ix_grid, iy_grid = jnp.meshgrid(free_indices, free_indices, indexing='ij')
                    vcov_raw = vcov_raw.at[ix_grid, iy_grid].set(vcov_free)

                    # F. Delta Method: Raw -> Model Space
                    def transform_flat(flat_raw_vec):
                        p_raw = unravel_fn(flat_raw_vec)
                        p_model = self.param_space.transform(p_raw)
                        p_model_flat, _ = ravel_pytree(p_model)
                        return p_model_flat
                    
                    # Calculate Jacobian
                    J = jax.jacfwd(transform_flat)(flat_raw_params)
                    
                    # Validate dimensions
                    if self.verbose:
                        jax.debug.print("Delta Method - J shape: {}, vcov_raw shape: {}", 
                                       J.shape, vcov_raw.shape)
                    
                    if J.shape[1] != vcov_raw.shape[0]:
                        raise ValueError(
                            f"Jacobian columns ({J.shape[1]}) don't match "
                            f"vcov_raw rows ({vcov_raw.shape[0]})"
                        )
                    
                    # Apply transformation
                    vcov_model_flat = J @ vcov_raw @ J.T
                    vcov = vcov_model_flat
                    
                    # Extract Standard Errors
                    std_errors_flat = jnp.sqrt(jnp.maximum(jnp.diag(vcov_model_flat), 0.0))
                    
                    # Reshape back to PyTree
                    _, unravel_model_fn = ravel_pytree(final_constrained_params)
                    std_errors = unravel_model_fn(std_errors_flat)
                
                else:
                    if self.verbose:
                        jax.debug.print("Warning: vcov_free is None")

            except Exception as e:
                logger.warning(f"Failed to compute standard errors: {e}")
                std_errors = None
                vcov = None

        return EstimationResult(
            params=final_constrained_params,
            loss=final_loss,
            success=opt_result.success,
            std_errors=std_errors,
            vcov=vcov,
            solver_result=final_solver_result,
            meta={
                "steps": int(opt_result.steps), 
                "optimizer": self.optimizer.__class__.__name__
            }
        )

    def _get_sum_weights(self, observations: Any) -> int | None:
        """
        Extract sum of weights from observations if available.
        Returns None if 'weights' key is not found.
        """
        weights = get_from_pytree(observations, "weights", default=None)
        if weights is not None:
            return int(jnp.sum(weights))
        return None