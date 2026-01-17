# src/econox/solvers/dynamic_programming.py
"""
Dynamic programming solver module for economic models.
Can be used for static models as well by setting discount_factor=0.
"""

import jax.numpy as jnp
import equinox as eqx
from typing import Any
from jaxtyping import PyTree, Array

from econox.protocols import StructuralModel, Utility, Distribution, TerminalApproximator
from econox.optim import FixedPoint, FixedPointResult
from econox.structures import SolverResult
from econox.logic import IdentityTerminal 

class ValueIterationSolver(eqx.Module):
    """
    Fixed-point solver using value function iteration.
    
    Attributes:
        utility (Utility): Utility function to compute flow utilities.
        dist (Distribution): Probability distribution for choice modeling.
        discount_factor (float): Discount factor for future utilities.
        terminal_approximator (TerminalApproximator): Approximator for terminal value function.
        numerical_solver (FixedPoint): Numerical solver for finding fixed points.

    
    Examples:
        >>> # Define structural components
        >>> utility = MyUtilityFunction()
        >>> dist = Type1ExtremeValue()
        
        >>> # Initialize solver
        >>> solver = ValueIterationSolver(
        ...     utility=utility,
        ...     dist=dist,
        ...     discount_factor=0.99,
        ...     terminal_approximator=IdentityTerminal(),
        ... )
        
        >>> # Solve the model
        >>> result = solver.solve(params, model)
        
        >>> # Access results
        >>> EV = result.solution  # Expected Value Function EV(s)
        >>> P = result.profile    # Choice Probabilities P(a|s)
    
    """
    utility: Utility
    dist: Distribution
    discount_factor: float
    terminal_approximator: TerminalApproximator = eqx.field(default_factory=IdentityTerminal)
    numerical_solver: FixedPoint = eqx.field(default_factory=FixedPoint)

    def solve(
        self,
        params: PyTree,
        model: StructuralModel
    ) -> Any:
        """
        Solves for the fixed point of the structural model using value iteration.

        Args:
            params (PyTree): Model parameters.
            model (StructuralModel): The structural model instance.

        Returns:
            SolverResult: The result of the solver containing the solution and additional information containing:

            * **solution** (Array): The computed Expected Value Function :math:`EV(s)` (Integrated Value Function / Emax).
            * **profile** (Array): The Conditional Choice Probabilities (CCP) :math:`P(a|s)` derived from the value function.
            * **success** (Bool): Whether the solver converged successfully.
            * **aux** (Dict): Auxiliary information, including number of steps taken.
        """
        utility = self.utility
        dist = self.dist

        data: PyTree = model.data
        transitions: Any = model.transitions

        if transitions is None:
            raise ValueError("Model transitions must be defined for ValueIterationSolver.")
        
        if hasattr(transitions, "ndim") and transitions.ndim != 2:
            raise ValueError(f"MVP Version only supports (S*A, S) shape. Got {transitions.shape}")
        
        expected_rows: int = model.num_states * model.num_actions
        expected_cols: int = model.num_states

        if hasattr(transitions, "shape"):
            if transitions.shape != (expected_rows, expected_cols):
                raise ValueError(
                    f"Transitions shape mismatch.\n"
                    f"Expected: ({expected_rows}, {expected_cols}) for (S*A, S)\n"
                    f"Got:      {transitions.shape}"
                )
        
        num_states: int = model.num_states
        num_actions: int = model.num_actions

        flow_utility: Array = utility.compute_flow_utility(params, model)

        # ---------------------------------------------------------
        # Bellman Operator
        # ---------------------------------------------------------
        def bellman_operator(current_ev: Array, args=None) -> Array:
            """
            Bellman operator for value iteration.
            
            Args:
                current_ev: Current expected value vector (S,)
                args: Unused. Required by FixedPoint.find_fixed_point signature.
            
            Returns:
                Updated expected value vector (S,)
            """
            # current_ev: (S,)
            expected_flat = transitions @ current_ev
            expected = expected_flat.reshape(num_states, num_actions)
            
            # Apply terminal value approximation
            expected = self.terminal_approximator.approximate(expected, params, model)
            
            choice_values = flow_utility + self.discount_factor * expected
            next_ev = dist.expected_max(choice_values)
            return next_ev
        
        initial_ev = jnp.zeros((num_states,))

        result: FixedPointResult = self.numerical_solver.find_fixed_point(
            step_fn=bellman_operator,
            init_val=initial_ev
        )

        final_ev: Array = result.value

        # ---------------------------------------------------------
        # Post-Processing
        # ---------------------------------------------------------
        expected_final_flat = transitions @ final_ev
        expected_final = expected_final_flat.reshape(num_states, num_actions)
        
        # Apply terminal value approximation
        expected_final = self.terminal_approximator.approximate(expected_final, params, model)
            
        value_choices = flow_utility + self.discount_factor * expected_final
        choice_probs = dist.choice_probabilities(value_choices)
        
        return SolverResult(
            solution=final_ev,
            profile=choice_probs,
            success=result.success,
            aux={"num_steps": result.steps}
        )
        