# src/econox/solvers/equilibrium.py
"""
Equilibrium solver module for dynamic economic models.
Can be used for static models as well by setting discount_factor=0.
"""

import jax.numpy as jnp
import equinox as eqx
from jaxtyping import PyTree, Float, Array

from econox.protocols import Distribution, FeedbackMechanism, StructuralModel, Solver, Utility, Dynamics
from econox.optim import FixedPoint, FixedPointResult
from econox.logic import SimpleDynamics
from econox.structures import SolverResult
from econox.solvers.dynamic_programming import ValueIterationSolver


class EquilibriumSolver(eqx.Module):
    """
    Fixed-point solver using equilibrium conditions.
    
    Attributes:
        numerical_solver: FixedPoint
        inner_solver: Solver (typically ValueIterationSolver)
    """
    numerical_solver: FixedPoint = eqx.field(default_factory=FixedPoint)
    inner_solver: Solver = eqx.field(default_factory=ValueIterationSolver)
    default_initial_distribution: Float[Array, "num_states"] | None = None
    default_dynamics: Dynamics | None = None

    def solve(
        self,
        params: PyTree,
        model: StructuralModel,
        utility: Utility,
        dist: Distribution,
        feedback: FeedbackMechanism | None = None,
        dynamics: Dynamics | None = None,
        initial_distribution: Float[Array, "num_states"] | None = None,
        damping: float = 1.0
    ) -> SolverResult:
        """
        Solves for the fixed point of the structural model using equilibrium conditions.

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
        feedback : FeedbackMechanism | None = None
            Feedback mechanism (updates model based on distribution).
            If None, raises ValueError.
        dynamics : Dynamics | None
            Dynamics logic for distribution evolution. If None, SimpleDynamics is used.
        initial_distribution : Float[Array, "num_states"] | None
            Initial guess for the distribution. If None, uses uniform distribution.
        damping : float
            Damping factor for fixed-point updates (0 < damping <= 1).

        Returns
        -------
        SolverResult
            solution: Equilibrium Distribution D*
            profile: Equilibrium Policy P*
            inner_result: Full result from the inner solver (Value Function etc.)
        """

        # Validate damping
        if not (0 < damping <= 1.0):
             raise ValueError(f"Damping must be in range (0, 1], got {damping}")
        
        if feedback is None:
            raise ValueError("Feedback mechanism must be provided for EquilibriumSolver.")
        
        num_states = model.num_states

        if dynamics is None:
            dynamics = self.default_dynamics
        apply_dynamics = dynamics if dynamics is not None else SimpleDynamics()

        if initial_distribution is None:
            initial_distribution = self.default_initial_distribution
        if initial_distribution is None:
            initial_distribution = jnp.ones(num_states) / num_states
        
        def equilibrium_step(current_dist: Array, args: None) -> Array:
            current_result = {"solution": current_dist}
            model_updated: StructuralModel = feedback.update(params, current_result, model)

            inner_result: SolverResult = self.inner_solver.solve(
                params=params, 
                model=model_updated, 
                utility=utility, 
                dist=dist)

            policy: Array | None = inner_result.profile
            if policy is None:
                raise ValueError("Inner solver must return a policy (profile) for equilibrium computation.")
            
            new_dist = apply_dynamics(
                distribution=current_dist, 
                policy=policy, 
                model=model_updated)
            
            updated_dist = damping * new_dist + (1 - damping) * current_dist

            return updated_dist
        
        result: FixedPointResult = self.numerical_solver.find_fixed_point(
            step_fn=equilibrium_step,
            init_val=initial_distribution
        )

        final_dist = result.value
        final_result = {"solution": final_dist}
        final_model: StructuralModel = feedback.update(params, final_result, model)
        final_inner_result = self.inner_solver.solve(
            params=params, 
            model=final_model, 
            utility=utility, 
            dist=dist
        )

        return SolverResult(
            solution=final_dist,           # Equilibrium Distribution D*
            profile=final_inner_result.profile,  # Equilibrium Policy P*
            inner_result=final_inner_result,     # Full inner details (V*)
            success=result.success,
            aux={"steps": result.steps,
            "diff": jnp.max(jnp.abs(result.value - initial_distribution)),
            "equilibrium_data": final_model.data}
        )