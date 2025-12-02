# src/econox/workflow/estimator.py
"""
Estimator module for the Econox framework.
Orchestrates the estimation process by connecting Data, Model, Solver, and Objective.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from jax.flatten_util import ravel_pytree
from typing import Any
from jaxtyping import PyTree, Scalar, Array

from econox.logic import feedback
from econox.protocols import FeedbackMechanism, StructuralModel, Solver, Objective, Utility, Distribution
from econox.structures import ParameterSpace, EstimationResult, SolverResult
from econox.strategies.numerical import Minimizer, MinimizerResult
from econox.config import LOSS_PENALTY
from econox.utils import get_from_pytree


class Estimator(eqx.Module):
    """
    Orchestrates the structural estimation process.
    
    Handles:
    1. Parameter transformation (Raw <-> Constrained) via ParameterSpace.
    2. Solving the model (Single run or Batched Simulation/SMM).
    3. Evaluating the objective function (Loss calculation).
    4. Minimizing the loss using an Optimizer.
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

    def fit(self, observations: Any, initial_params: dict | None = None) -> EstimationResult:
        """
        Estimates the model parameters to minimize the objective function.

        Args:
            observations: Observed data to match (passed to Objective).
            initial_params: Dictionary of initial parameter values (Constrained space).
                            If None, uses initial_params from ParameterSpace.

        Returns:
            EstimationResult containing estimated parameters, final loss, and details.
        """
        # 1. Prepare Initial Parameters
        # Convert constrained initial params to raw (unconstrained) space for the optimizer
        if initial_params is None:
            constrained_init = self.param_space.initial_params
        else:
            constrained_init = initial_params
            
        raw_init = self.param_space.inverse_transform(constrained_init)

        # 2. Define Loss Function (The core pipeline)
        @eqx.filter_jit
        def loss_fn(raw_params: PyTree, args: Any) -> Scalar:
            # A. Transform Parameters: Raw (Optimizer) -> Constrained (Model)
            try:
                params = self.param_space.transform(raw_params)
            except ValueError:
                return jnp.array(LOSS_PENALTY)

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

        # 3. Run Optimization
        print(f"Starting estimation with {self.optimizer.__class__.__name__}...")
        opt_result: MinimizerResult = self.optimizer.minimize(
            loss_fn=loss_fn,
            init_params=raw_init,
            args=observations # Passed as args to loss_fn
        )

        # 4. Process Results
        final_raw_params = opt_result.params
        final_constrained_params = self.param_space.transform(final_raw_params)
        final_loss = opt_result.loss
        
        # --- [Correction 4] Run Solver one last time to get the final state ---
        final_solver_result = self.solver.solve(
            final_constrained_params, self.model, self.utility, self.dist, self.feedback
        )

        # 5. Post-Estimation Inference (Standard Errors with Delta Method)
        vcov_model = None
        std_errors = None

        if opt_result.success:
            try:
                # --- Get number of observations ---
                num_obs = self._get_num_observations(observations)
                
                # --- [Correction 1 & 2] Jacobian Correction & Fixed Masking ---
                
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
                    num_observations=num_obs
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
                    
                    # Extract Standard Errors
                    std_errors_flat = jnp.sqrt(jnp.maximum(jnp.diag(vcov_model_flat), 0.0))
                    
                    # Reshape back to PyTree
                    _, unravel_model_fn = ravel_pytree(final_constrained_params)
                    std_errors = unravel_model_fn(std_errors_flat)
                    
                    # Optional: Store covariance matrix
                    # vcov_model = vcov_model_flat
                
                else:
                    if self.verbose:
                        jax.debug.print("Warning: vcov_free is None")

            except Exception as e:
                print(f"Warning: Failed to compute standard errors: {e}")
                std_errors = None

        return EstimationResult(
            params=final_constrained_params,
            loss=final_loss,
            success=opt_result.success,
            std_errors=std_errors,
            vcov=None,
            solver_result=final_solver_result,
            meta={
                "steps": opt_result.steps, 
                "optimizer": self.optimizer.__class__.__name__
            }
        )

    def _get_num_observations(self, observations: Any) -> int:
        """Extract effective sample size from observations."""
        weights = get_from_pytree(observations, "weights", default=None)
        if weights is not None:
            return int(jnp.sum(weights))
        
        leaves = jax.tree_util.tree_leaves(observations)
        for leaf in leaves:
            if hasattr(leaf, 'shape') and leaf.ndim > 0:
                return int(leaf.shape[0])
        
        return 1