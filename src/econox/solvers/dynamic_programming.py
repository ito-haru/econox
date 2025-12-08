# src/econox/solvers/dynamic_programming.py
"""
Dynamic programming solver module for economic models.
Can be used for static models as well by setting discount_factor=0.
"""

import jax.numpy as jnp
import equinox as eqx
from typing import Any
from jaxtyping import PyTree, Array, Int

from econox.protocols import StructuralModel, Utility, Distribution, FeedbackMechanism
from econox.optim import FixedPoint, FixedPointResult
from econox.structures import SolverResult
from econox.utils import get_from_pytree


class ValueIterationSolver(eqx.Module):
    """
    Fixed-point solver using value function iteration.
    
    Attributes:
        numerical_solver: FixedPoint
        discount_factor: float

    Optional Data Keys:
        The solver looks for the following keys in `model.data` to enable 
        terminal state approximation (EV(T-1) = EV(T)):
        
        * "terminal_state_indices" (Int[Array, "n"]): Indices of states at T.
        * "previous_state_indices" (Int[Array, "n"]): Indices of states at T-1 to copy from.
        
        If provided, both must be present and have the same shape.
    """
    numerical_solver: FixedPoint = eqx.field(default_factory=FixedPoint)
    discount_factor: float = 0.90

    def solve(
        self,
        params: PyTree,
        model: StructuralModel, 
        utility: Utility, 
        dist: Distribution, 
        feedback: FeedbackMechanism | None = None
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
        feedback : FeedbackMechanism | None
            Feedback mechanism (not used in this solver).

        Returns
        -------
        SolverResult
            The result of the solver containing the solution and additional info.
        """
        
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
        
        # If finite, approximate terminal value EV[T-1] = EV[T]
        term_idx: Int[Array, "n_terminal"] | None = get_from_pytree(data, "terminal_state_indices", default=None)
        prev_idx: Int[Array, "n_terminal"] | None = get_from_pytree(data, "previous_state_indices", default=None)

        # Validate terminal approximation indices
        if (term_idx is None) != (prev_idx is None):
            raise ValueError(
                "ValueIteration: 'terminal_state_indices' and 'previous_state_indices' "
                "must be provided together in model.data to enable terminal approximation."
            )
        if term_idx is not None and prev_idx is not None:
            if term_idx.shape != prev_idx.shape:
                raise ValueError(
                    f"ValueIteration: Shape mismatch. terminal_state_indices {term_idx.shape} "
                    f"!= previous_state_indices {prev_idx.shape}"
                )
        num_states: int = model.num_states
        num_actions: int = model.num_actions

        flow_utility: Array = utility.compute_flow_utility(params, model)

        # ---------------------------------------------------------
        # Helper Function (Closure)
        # ---------------------------------------------------------
        def apply_terminal_approximation(expected: Array) -> Array:
            """Apply terminal state approximation EV(T-1) = EV(T) if configured."""
            if term_idx is not None and prev_idx is not None:
                expected_at_prev = expected[prev_idx, :]
                return expected.at[term_idx, :].set(expected_at_prev)
            return expected

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
            
            expected = apply_terminal_approximation(expected)
            
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
        
        expected_final = apply_terminal_approximation(expected_final)
            
        value_choices = flow_utility + self.discount_factor * expected_final
        choice_probs = dist.choice_probabilities(value_choices)
        
        return SolverResult(
            solution=final_ev,
            profile=choice_probs,
            success=result.success,
            aux={"num_steps": result.steps}
        )
        