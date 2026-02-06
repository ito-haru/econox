# src/econox/workflow/estimator.py
"""
Estimator module for the Econox framework.
Orchestrates the estimation process by connecting Data, Model, Solver, and Objective.
"""

import logging
import equinox as eqx
from typing import Any

from econox.protocols import StructuralModel, Solver
from econox.methods.base import EstimationMethod
from econox.structures import ParameterSpace, EstimationResult
from econox.optim import Minimizer

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
        sample_size: int | None = None
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
        result = self.method.fit(
            observations=observations,
            model=self.model,
            param_space=self.param_space,
            solver=self.solver,
            optimizer=self.optimizer,
            verbose=self.verbose,
            initial_params=initial_params,
            sample_size=sample_size
        )
        return result
        