# src/econox/solvers/fixed_point.py
"""
Fixed-point solver for Structural Models in the Econox framework.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Any
from jaxtyping import PyTree, Array, Float

from econox.protocols import StructuralModel, Utility, Distribution, FeedbackMechanism
from econox.strategies import FixedPoint, FixedPointResult


class ValueIterationSolver(eqx.Module):
    """
    Fixed-point solver using value function iteration.
    
    Attributes:
        max_iterations: Maximum number of iterations to perform.
        tolerance: Convergence tolerance.
    """
    numerical_solver: FixedPoint = eqx.field(default_factory=FixedPoint)
    discount_factor: float = 0.90

    def solve(
        self,
        params: PyTree,
        model: StructuralModel, 
        utility: Utility, 
        dist: Distribution, 
        feedback: None = None
    ) -> Any:
        """
        Solves for the fixed point of the structural model using value iteration.

        Parameters
        ----------
        params : PyTree
            Model parameters.
        model : StructuralModel
            The structural model instance.
        utility : Utility
            Utility function instance.
        dist : Distribution
            Distribution of agents in the model.
        feedback : None
            Feedback mechanism (not used in this solver).

        Returns
        -------
        FixedPointResult
            The result of the fixed-point computation.
        """
        
        transitions: PyTree | None = model.transitions
        if transitions is None:
            raise ValueError("Model transitions must be defined for ValueIterationSolver.")

        flow_utility: Array = utility.compute_flow_utility(params, model)

        def bellman_operator(current_value: Array, _) -> Array:
            expected_future_value = transitions @ current_value
            choice_values = flow_utility + self.discount_factor * expected_future_value[:, None]
            next_value: Array = dist.expected_max(choice_values)
            return next_value
        
