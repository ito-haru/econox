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

from econox.protocols import StructuralModel, Solver
from econox.methods.base import EstimationMethod
from econox.structures import ParameterSpace, EstimationResult
from econox.optim import Minimizer, MinimizerResult
from econox.utils import get_from_pytree

logger = logging.getLogger(__name__)


class Estimator(eqx.Module):
    """
    Orchestrates the structural estimation process.
    
    Handles:
    1. Parameter transformation (Raw <-> Constrained) via ParameterSpace.
    2. Solving the model (Single run or Batched Simulation/SMM).
    3. Evaluating the loss function via EstimationMethod.
    4. Minimizing the loss using an Optimizer.

    Attributes:
        model: StructuralModel - The structural model to estimate.
        param_space: ParameterSpace - Parameter transformation and constraints.
        method: EstimationMethod - Strategy for estimation (Loss definition & Inference).
        solver: Solver | None - Solver to compute model solutions. Not required for reduced-form estimation.
        optimizer: Minimizer - Optimization strategy for minimizing the loss.
        verbose: bool - If True, enables detailed logging for debugging.
    
    Examples:
        >>> # 1. Setup components
        >>> model = Model.from_data(...)
        >>> param_space = ParameterSpace.create(...)
        >>> solver = ValueIterationSolver(utility=..., dist=..., discount_factor=0.95)
        >>> method = MaximumLikelihood(model_key="choice_probs", obs_key="actions")
        
        >>> # 2. Initialize Estimator
        >>> estimator = Estimator(
        ...     model=model,
        ...     param_space=param_space,
        ...     solver=solver,
        ...     method=method
        ... )
        
        >>> # 3. Run estimation
        >>> result = estimator.fit(observations=data)
        >>> print(result.params)
    """
    model: StructuralModel
    param_space: ParameterSpace
    method: EstimationMethod

    solver: Solver | None = None
    optimizer: Minimizer = eqx.field(default_factory=Minimizer)
    
    # Debugging
    verbose: bool = eqx.field(default=False, static=True)

    def fit(
        self,
        observations: Any, 
        initial_params: dict | None = None,
        sample_size: int | None = None,
        force_numerical: bool = False
        ) -> EstimationResult:
        """
        Estimates the model parameters to minimize the objective function.

        Args:
            observations: Observed data to match (passed to Objective).
            initial_params: Dictionary of initial parameter values (Constrained space).
                            If None, uses initial_params from ParameterSpace.
            sample_size: Effective sample size for variance calculations.
                         Note: This argument is primarily for numerical estimation.
                         If an analytical solution is found, this argument 
                         is ignored and the actual data size (n_obs) is used instead.
            force_numerical: If True, forces numerical optimization even if an analytical solution is available.

        Returns:
            EstimationResult containing:
            * **params**: Estimated parameters (Constrained space).
            * **loss**: Final loss value.
            * **success**: Whether optimization was successful.
            * **std_errors**: Standard errors of estimates (if computed).
            * **vcov**: Variance-covariance matrix (if computed).
            * **t_values**: t-statistics of estimates (if computed).
            * **solver_result**: Final solver result (if applicable).
        """
        # =========================================================
        # 1. Try Analytical Solution (Priority)
        # =========================================================
        if not force_numerical:
            analytical_result: EstimationResult | None = self.method.solve(
                self.model, observations, self.param_space
            )
            
            if analytical_result is not None:
                return analytical_result

        # =========================================================
        # 2. Numerical Optimization (Structural Route)
        # =========================================================
        solver = self.solver

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
            # Case1: Structural (With Solver)
            if solver is not None:
                result = solver.solve(
                        params, 
                        self.model
                    )
            # Case2: Reduced Form (No Solver)
            else:
                result = None

            # C. Evaluate Objective
            loss = self.method.compute_loss(result, observations, params, self.model)
            
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
        final_raw_params_free = opt_result.params
        final_constrained_params = self.param_space.transform(final_raw_params_free)
        final_loss = opt_result.loss
        
        if solver is not None:
            final_solver_result = solver.solve(
                final_constrained_params, self.model
            )
        else:
            final_solver_result = None

        # =========================================================
        # 3. Statistical Inference (Variance Calculation)
        # =========================================================
        std_errors = None
        vcov = None

        if opt_result.success and final_N is not None and self.method.variance is not None:
            try:
                # A. Create separate unravel functions for raw and constrained spaces
                # Use the actual optimization results as templates to ensure structure matching
                _, unravel_raw_fn = ravel_pytree(final_raw_params_free)
                _, unravel_constrained_fn = ravel_pytree(final_constrained_params)

                # B. Get flat vector of free parameters (optimizer output)
                flat_raw_params_free, _ = ravel_pytree(final_raw_params_free)

                # C. Define wrapper loss for free params only
                # Must match the structure used during JIT-compilation of loss_fn
                def loss_fn_for_inference(free_params_vec: Array) -> Scalar:
                    raw_pytree = unravel_raw_fn(free_params_vec)
                    return loss_fn(raw_pytree, observations)

                # D. Compute variance in the free raw space
                _, vcov_free = self.method.variance.compute(
                    loss_fn=loss_fn_for_inference,
                    params=flat_raw_params_free,
                    observations=observations,
                    num_observations=final_N
                    )

                if vcov_free is not None:
                    # E. Delta method: Map free raw vector -> full constrained vector
                    # This handles the transformation and internal fixed-parameter filling
                    def transform_flat(free_vec):
                        p_raw = unravel_raw_fn(free_vec)
                        p_model = self.param_space.transform(p_raw) # Fills fixed params internally
                        p_model_flat, _ = ravel_pytree(p_model)
                        return p_model_flat

                    # Jacobian of the transformation: (n_total, n_free)
                    J = jax.jacfwd(transform_flat)(flat_raw_params_free)
            
                    # Project variance to constrained space
                    vcov_model_flat = J @ vcov_free @ J.T
                    vcov = vcov_model_flat
            
                    # Extract standard errors and unravel to constrained PyTree structure
                    std_errors_flat = jnp.sqrt(jnp.maximum(jnp.diag(vcov_model_flat), 0.0))
                    std_errors = unravel_constrained_fn(std_errors_flat)
                
                else:
                    if self.verbose:
                        logger.warning("Variance calculation returned None (e.g. Hessian failed).")

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
                "optimizer": self.optimizer.method_name,
                "optimizer_steps": int(opt_result.steps),
                "computation": "Numerical",
                "estimation_method": self.method.__class__.__name__,
                "inference_method": 
                    self.method.variance.__class__.__name__ if self.method.variance is not None else None,
                "n_obs": final_N,
                "n_params": self.param_space.num_total_params,
                "n_free_params": self.param_space.num_free_params,
                "n_fixed": self.param_space.num_total_params - self.param_space.num_free_params
            },
            initial_params=constrained_init,
            fixed_mask=self.param_space.fixed_mask
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