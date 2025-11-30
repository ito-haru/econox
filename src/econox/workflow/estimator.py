# src/econox/workflow/estimator.py
"""
Estimator module for the Econox framework.
Orchestrates the estimation process by connecting Data, Model, Solver, and Objective.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Any
from jaxtyping import PyTree, Scalar

from econox.protocols import StructuralModel, Solver, Objective, Utility, Distribution
from econox.structures import ParameterSpace, EstimationResult, SolverResult
from econox.strategies.numerical import Minimizer, MinimizerResult
from econox.config import LOSS_PENALTY


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
                # Currently disabled to reduce debugging load as per requirements.
                # Future implementation will involve jax.vmap over the solver.
                raise NotImplementedError(
                    "SMM (Simulated Method of Moments) is not yet implemented. "
                    "Please set num_simulations=None to use standard estimation."
                )

            # Case 2: Standard Estimation (MLE / NFXP) - Single Execution
            else:
                result = self.solver.solve(
                    params, self.model, self.utility, self.dist
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
        
        # 5. Post-Estimation Inference (Standard Errors)
        # Determine N (sample size) from observations for scaling
        leaves = jax.tree_util.tree_leaves(observations)
        num_obs = leaves[0].shape[0] if leaves else 1
        
        vcov = None
        std_errors = None
        
        if opt_result.success:
            loss_fn_for_hessian = lambda p: loss_fn(p, observations)
            # Delegate variance calculation to the Objective (Method Injection)
            vcov = self.objective.calculate_variance(
                loss_fn=loss_fn_for_hessian,
                params=opt_result.params, # Using RAW params for differentiation
                observations=observations,
                num_observations=num_obs
            )
            
            if vcov is not None:
                 # Diagonal elements are variances
                 std_errors = jnp.sqrt(jnp.diag(vcov))

        return EstimationResult(
            params=final_constrained_params,
            loss=final_loss,
            success=opt_result.success,
            std_errors=std_errors,
            vcov=vcov,
            meta={
                "steps": opt_result.steps, 
                "optimizer": self.optimizer.__class__.__name__
            }
        )