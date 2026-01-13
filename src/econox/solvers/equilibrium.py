# src/econox/solvers/equilibrium.py
"""
Equilibrium solver module for dynamic economic models.
Can be used for static models as well by setting discount_factor=0.
"""

import jax.numpy as jnp
import equinox as eqx
from jaxtyping import PyTree, Float, Array

from econox.protocols import FeedbackMechanism, StructuralModel, Solver, Dynamics
from econox.optim import FixedPoint, FixedPointResult
from econox.structures import SolverResult


class EquilibriumSolver(eqx.Module):
    """
    Fixed-point solver for General Equilibrium (GE) or Stationary Equilibrium.

    This solver searches for a distribution of agents (or prices) :math:`D^*` such that:
    
    .. math:: D^* = \\Phi(D^*, \\theta)

    where :math:`\\Phi` represents the compound operator of:
    1. Updating the environment (Feedback): :math:`\\Omega' = \\Gamma(D, \\Omega)`
    2. Solving the agent's problem (Inner Solver): :math:`\\sigma^* = \\text{argmax} \\, V(s; \\Omega')`
    3. Applying the law of motion (Dynamics): :math:`D' = f(D, \\sigma^*)`

    Attributes:
        inner_solver (Solver): The solver used to compute the optimal policy given a fixed environment.
        feedback (FeedbackMechanism): Logic to update model data based on aggregate results.
        dynamics (Dynamics): Law of motion describing how the distribution evolves.
        numerical_solver (FixedPoint): The numerical algorithm for the outer loop (e.g., Anderson Acceleration).
        damping (float): Damping factor for the update step :math:`D_{k+1} = (1-\\lambda)D_k + \\lambda D_{new}`.
        initial_distribution (Array | None): Initial guess for the distribution.
    
    Examples:
        >>> # 1. Inner agent problem (e.g., Household optimization)
        >>> inner_solver = ValueIterationSolver(...)
        
        >>> # 2. Market clearing logic (e.g., Supply = Demand)
        >>> feedback = FunctionFeedback(func=WageFeedback, target_key="wage")
        
        >>> # 3. Dynamics (Law of Motion)
        >>> dynamics = SimpleDynamics()
        
        >>> # 4. Equilibrium Solver
        >>> eq_solver = EquilibriumSolver(
        ...     inner_solver=inner_solver,
        ...     feedback=feedback,
        ...     dynamics=dynamics,
        ...     damping=0.5
        ... )
        
        >>> # Solve for stationary equilibrium
        >>> result = eq_solver.solve(params, model)
    """
    # ---------------------------------------------------------------
    # 1. Structural Components (The "What")
    # ---------------------------------------------------------------
    inner_solver: Solver          # The Agent (holds Utility/Dist)
    feedback: FeedbackMechanism   # The Market Clearing logic
    dynamics: Dynamics            # The Law of Motion

    # ---------------------------------------------------------------
    # 2. Solver Configuration (The "How")
    # ---------------------------------------------------------------
    numerical_solver: FixedPoint = eqx.field(default_factory=FixedPoint)
    damping: float = 1.0
    initial_distribution: Float[Array, "num_states"] | None = None

    def solve(
        self,
        params: PyTree,
        model: StructuralModel,
    ) -> SolverResult:
        """
        Solves for the fixed point of the structural model using equilibrium conditions.

        Args:
            params (PyTree): Model parameters.
            model (StructuralModel): The structural model instance.

        Returns:
            SolverResult: The result object containing:

            * **solution**: Equilibrium Distribution :math:`D^*`
            * **profile**: Equilibrium Policy :math:`P^*`
            * **inner_result**: Full result from the inner solver (Value Function etc.)
        """
        feedback = self.feedback
        dynamics = self.dynamics
        damping = self.damping
        initial_distribution = self.initial_distribution

        # Validate damping
        if not (0 < damping <= 1.0):
             raise ValueError(f"Damping must be in range (0, 1], got {damping}")
        
        if feedback is None:
            raise ValueError("Feedback mechanism must be provided for EquilibriumSolver.")
        
        num_states = model.num_states

        if initial_distribution is None:
            initial_distribution = jnp.ones(num_states) / num_states
        
        def equilibrium_step(current_dist: Array, args: None) -> Array:
            current_result = {"solution": current_dist}
            model_updated: StructuralModel = feedback.update(params, current_result, model)

            inner_result: SolverResult = self.inner_solver.solve(
                params=params, 
                model=model_updated
                )

            policy: Array | None = inner_result.profile
            if policy is None:
                raise ValueError("Inner solver must return a policy (profile) for equilibrium computation.")
            
            new_dist = dynamics(
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
            model=final_model
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