# src/econox/methods/base.py
"""
Base module for method functions in the Econox framework.
"""

from __future__ import annotations
from typing import Sequence, Any
import equinox as eqx
from jaxtyping import PyTree, Scalar, Array
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import logging

from econox.optim import Minimizer, MinimizerResult
from econox.protocols import StructuralModel, Solver
from econox.structures import EstimationResult
from econox.methods.variance import Variance
from econox.structures import ParameterSpace
from econox.utils import get_from_pytree

logger = logging.getLogger(__name__)


class EstimationMethod(eqx.Module):
    """
    Base class for all estimation method functions in Econox.
    
    This class serves three main purposes:
    1. **Strategy Definition**: Defines the loss function to be minimized during numerical estimation.
    2. **Analytical Solution**: Optionally provides a direct solution method (e.g., for OLS/2SLS).
    3. **Inference**: Optionally defines how to calculate standard errors (e.g., Hessian, Sandwich).

    Users can create custom objectives by subclassing this class or by using the 
    `@method_from_loss` decorator.

    Attributes:
        variance: Variance | None
    """
    variance: Variance | None = eqx.field(default=None, kw_only=True)
    """
    Optional variance calculation strategy for inference.
    """

    def fit(
        self,
        observations: Any, 
        model: StructuralModel,
        param_space: ParameterSpace,
        solver: Solver | None = None,
        optimizer: Minimizer = Minimizer(),
        verbose: bool = False,
        initial_params: dict | None = None,
        sample_size: int | None = None
        ) -> EstimationResult:
        """
        Estimates the model parameters to minimize the objective function.

        Args:
            model: The structural model being estimated.
            param_space: The parameter space definition.
            solver: Optional solver to compute model solutions.
            observations: Observed data to match (passed to Objective).
            initial_params: Dictionary of initial parameter values (Constrained space).
                            If None, uses initial_params from ParameterSpace.
            sample_size: Effective sample size for variance calculations.
                         Note: This argument is primarily for numerical estimation.
                         If an analytical solution is found, this argument 
                         is ignored and the actual data size (n_obs) is used instead.

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
        result = self._minimize_loss(
            observations=observations,
            model=model,
            param_space=param_space,
            solver=solver,
            optimizer=optimizer,
            verbose=verbose,
            initial_params=initial_params,
            sample_size=sample_size
        )
        return result
    
    def _minimize_loss(
        self,
        observations: Any, 
        model: StructuralModel,
        param_space: ParameterSpace,
        solver: Solver | None = None,
        optimizer: Minimizer = Minimizer(),
        verbose: bool = False,
        initial_params: dict | None = None,
        sample_size: int | None = None,
        ) -> EstimationResult:
        # Prepare Initial Parameters
        # Convert constrained initial params to raw (unconstrained) space for the optimizer
        if initial_params is None:
            constrained_init = param_space.initial_params
        else:
            constrained_init = initial_params
            
        raw_init = param_space.inverse_transform(constrained_init)

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
        
        # Define Loss Function (The core pipeline)
        @eqx.filter_jit
        def loss_fn(raw_params: PyTree, args: Any) -> Scalar:
            # A. Transform Parameters: Raw (Optimizer) -> Constrained (Model)
            params = param_space.transform(raw_params)

            # Debug output if verbose
            if verbose:
                jax.debug.print("Estimator: Checking Params: {}", params)

            # B. Solve the Model
            # Case1: Structural (With Solver)
            if solver is not None:
                result = solver.solve(
                        params, 
                        model
                    )
            # Case2: Reduced Form (No Solver)
            else:
                result = None

            # C. Evaluate Objective
            loss = self.compute_loss(result, observations, params, model)
            
            # Debug output if verbose
            if verbose:
                jax.debug.print("Estimator: Loss: {}", loss)
                
            return loss

        # Run Optimization
        logger.info(f"Starting estimation with {optimizer.__class__.__name__}...")
        opt_result: MinimizerResult = optimizer.minimize(
            loss_fn=loss_fn,
            init_params=raw_init,
            args=observations # Passed as args to loss_fn
        )

        # 4. Process Results
        final_raw_params_free = opt_result.params
        final_constrained_params = param_space.transform(final_raw_params_free)
        final_loss = opt_result.loss
        
        if solver is not None:
            final_solver_result = solver.solve(
                final_constrained_params, model
            )
        else:
            final_solver_result = None

        # =========================================================
        # 2. Statistical Inference (Variance Calculation)
        # =========================================================
        
        std_errors = None
        vcov = None

        if opt_result.success and final_N is not None and self.variance is not None:
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
                _, vcov_free = self.variance.compute(
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
                        p_model = param_space.transform(p_raw) # Fills fixed params internally
                        p_model_flat, _ = ravel_pytree(p_model)
                        return p_model_flat

                    # Jacobian of the transformation: (n_total, n_free)
                    J = jax.jacfwd(transform_flat)(flat_raw_params_free)
            
                    # Project variance to constrained space
                    vcov_model_flat = J @ vcov_free @ J.T
                    vcov = vcov_model_flat

                    # Create a dummy dict mapping keys to their insertion index
                    user_keys = list(param_space.initial_params.keys())
                    indices_struct = {k: i for i, k in enumerate(user_keys)}
                    
                    # Flatten this structure using JAX to see where each index ended up
                    flat_indices, _ = ravel_pytree(indices_struct)
                    flat_indices = jnp.array(flat_indices, dtype=int)
                    
                    # Compute permutation to sort JAX order back to User order
                    perm_order = jnp.argsort(flat_indices)
                    
                    # Apply permutation to rows and columns of vcov
                    vcov = vcov[perm_order][:, perm_order]
            
                    # Extract standard errors and unravel to constrained PyTree structure
                    variances = jnp.diag(vcov_model_flat)
                    std_errors_flat_jax = jnp.sqrt(jnp.where(variances < -1e-10, jnp.nan, jnp.maximum(variances, 0.0)))
                    std_errors_jax = unravel_constrained_fn(std_errors_flat_jax)
                    
                    if isinstance(std_errors_jax, dict):
                         std_errors = {k: std_errors_jax[k] for k in user_keys if k in std_errors_jax}
                    else:
                         std_errors = std_errors_jax
                
                else:
                    if verbose:
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
                "optimizer": optimizer.method_name,
                "optimizer_steps": int(opt_result.steps),
                "computation": "Numerical",
                "estimation_method": self.__class__.__name__,
                "inference_method": 
                    self.variance.__class__.__name__ if self.variance is not None else None,
                "n_obs": final_N,
                "n_params": param_space.num_total_params,
                "n_free_params": param_space.num_free_params,
                "n_fixed": param_space.num_total_params - param_space.num_free_params
            },
            initial_params=constrained_init,
            fixed_mask=param_space.fixed_mask
        )

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
        raise NotImplementedError(
            "compute_loss is not implemented for this EstimationMethod."
        )

    def solve(
        self,
        model: StructuralModel,
        observations: Any,
        param_space: Any
    ) -> EstimationResult | None:
        """
        Computes the analytical solution for the parameters, if available.

        This method allows the `Estimator` to bypass the numerical optimization loop 
        for models that have a closed-form solution (e.g., OLS, 2SLS).

        Args:
            model: The structural model environment.
            observations: Observed data.
            param_space: The parameter space definition.

        Returns:
            EstimationResult | None:
            Returns an ``EstimationResult`` if an analytical solution is found.
            Returns ``None`` otherwise (default), and the Estimator will fall back 
            to numerical optimization using `compute_loss`.
        """
        return None
    
    def _get_sum_weights(self, observations: Any) -> int | None:
        """
        Extract sum of weights from observations if available.
        Returns None if 'weights' key is not found.
        """
        weights = get_from_pytree(observations, "weights", default=None)
        if weights is not None:
            return int(jnp.sum(weights))
        return None


class CompositeMethod(EstimationMethod):
    """
    Combines multiple estimation methods into a single scalar loss.
    Assumes methods are independent (Block-Diagonal Weighting).
    
    Loss = sum( weight_i * loss_i )

    Attributes:
        methods: Sequence[EstimationMethod]
        weights: Sequence[float] | None
            Optional weights for each method. If None, equal weights are used.
        variance: Variance | None
            Optional variance calculation strategy for inference.
            Note: By default, variance is not computed for composite methods because
            the combined loss may not correspond to a valid statistical model.
    """
    methods: Sequence[EstimationMethod]
    weights: Sequence[float] | None = eqx.field(default=None)
    
    variance: Variance | None = eqx.field(default=None, kw_only=True)

    def compute_loss(
        self,
        result: Any | None,
        observations: Any,
        params: PyTree,
        model: StructuralModel
    ) -> Scalar:
    
        current_weights = self.weights
        if current_weights is None:
            current_weights = [1.0] * len(self.methods)
        elif len(current_weights) != len(self.methods):
            raise ValueError("Weights length must match methods length.")
        
        total_loss = jnp.array(0.0)
        
        for method, w in zip(self.methods, current_weights):
            loss = method.compute_loss(result, observations, params, model)
            total_loss = total_loss + (w * loss)
            
        return total_loss
