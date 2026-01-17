# src/econox/protocols.py
"""
Protocol definitions for the Econox framework.

This module defines the core interfaces (contracts) that enable modularity.
By adhering to these protocols, users can swap out components (e.g., changing 
utility functions or solver algorithms) without modifying the rest of the workflow.
"""

from __future__ import annotations
from typing import Protocol, Any, TypeAlias, runtime_checkable
from jaxtyping import Array, Float, PyTree

Scalar: TypeAlias = Float[Array, ""]

# =============================================================================
# 1. Data Containers (Model)
# =============================================================================

@runtime_checkable
class StructuralModel(Protocol):
    """
    Represents the economic environment (State Space and Constraints).
    
    In structural estimation, a model :math:`M` is defined by the tuple 
    :math:`(S, A, T, \\Omega)`, where:
    
    * :math:`S`: State space (num_states)
    * :math:`A`: Action space (num_actions)
    * :math:`T`: Time horizon (num_periods)
    * :math:`\\Omega`: Information set / Data constants (data)

    This protocol abstracts away the storage details of these elements.
    """
    @property
    def num_states(self) -> int: ...
    
    @property
    def num_actions(self) -> int: ...

    @property
    def num_periods(self) -> int | float:
        """
        Number of periods `T` in the model.
        Should be a positive integer for finite horizon or `np.inf` for infinite horizon.
        """
        ...

    @property
    def data(self) -> PyTree:
        """Immutable constants :math:`\\Omega` (features, matrices, etc.)."""
        ...

    @property
    def transitions(self) -> PyTree | None:
        """
        Transition structure (e.g., transition matrix or adjacency).
        Corresponds to :math:`P(s' | s, a)`.
        """
        ...
        
    @property
    def availability(self) -> PyTree | None:
        """
        Feasible action set :math:`A(s)`.
        Boolean mask of shape (num_states, num_actions).
        """
        ...
    # ----------------------------------------------------------------

    def replace_data(self, key: str, value: Any) -> StructuralModel:
        """
        Returns a new instance of the model with the specified data key updated.
        Required for Feedback mechanisms to update the environment (e.g., prices).
        
        Args:
            key: The name of the data field to update.
            value: The new value for that field.
            
        Returns:
            A new StructuralModel instance (immutable update).
        """
        ...

# =============================================================================
# 2. Logic Components (The Physics)
# =============================================================================

@runtime_checkable
class Utility(Protocol):
    """
    Structural Utility Function :math:`u(s, a; \\theta)`.
    
    Defines the instantaneous payoff an agent receives from taking action :math:`a`
    in state :math:`s`.
    """
    def compute_flow_utility(self, params: PyTree, model: StructuralModel) -> Float[Array, "n_states n_actions"]:
        """Calculates the utility matrix given parameters and model state."""
        ...

@runtime_checkable
class Distribution(Protocol):
    """
    Stochastic Shock Distribution :math:`F(\\epsilon)`.
    
    Defines the properties of the unobserved state variables (error terms).
    Handles the smoothing of the max operator (Emax) and choice probabilities (CCP).
    
    Common examples: Type-I Extreme Value (Logit), Normal (Probit).
    """
    def expected_max(self, values: Float[Array, "n_states n_actions"]) -> Float[Array, "n_states"]:
        """
        Computes the expected maximum value: 
        :math:`E[\\max_a (v(s, a) + \\epsilon(a))]`
        """
        ...

    def choice_probabilities(self, values: Float[Array, "n_states n_actions"]) -> Float[Array, "n_states n_actions"]:
        """
        Computes conditional choice probabilities (CCP):
        :math:`P(a | s) = P(v(s, a) + \\epsilon(a) \\ge v(s, a') + \\epsilon(a'), \\forall a')`
        """
        ...

@runtime_checkable
class FeedbackMechanism(Protocol):
    """
    Equilibrium/Market Clearing Condition.
    
    Defines how aggregate agent behaviors affect the environment (e.g., prices, congestion).
     Mathematically: :math:`\\Omega' = \\Gamma(\\sigma, \\Omega)`, where :math:`\\sigma` is the policy.
    """
    def update(self, params: PyTree, current_result: Any, model: StructuralModel) -> StructuralModel:
        """Updates the model environment based on the current solution."""
        ...

@runtime_checkable
class Dynamics(Protocol):
    """
    State Transition Law of Motion :math:`s' = f(s, a, \\xi)`.
    
    Defines how the distribution of agents over states evolves over time.
    Used for simulation and calculating steady-state distributions.
    """
    def __call__(
        self, 
        distribution: Float[Array, "num_states"], 
        policy: Float[Array, "num_states num_actions"], 
        model: StructuralModel
    ) -> Float[Array, "num_states"]:
        """Computes :math:`D_{t+1}` given :math:`D_t` and Policy."""
        ...

@runtime_checkable
class TerminalApproximator(Protocol):
    """
    Terminal Value Function Approximator for Finite Horizon Models.
    """
    def approximate(
        self, 
        expected: Float[Array, "num_states num_actions"], 
        params: PyTree, 
        model: StructuralModel
    ) -> Float[Array, "num_states num_actions"]:
        """
        Computes the terminal value function approximation.
        
        Args:
            expected: The expected future value matrix before terminal adjustment.
            params: Model parameters (may include growth rates, etc.).
            model: The structural model instance providing data and metadata.
            
        Returns:
            The adjusted expected future value matrix.
        """
        ...

# =============================================================================
# 3. Core Engine (Solver)
# =============================================================================

@runtime_checkable
class Solver(Protocol):
    """
    Computational Engine for the Model.
    
    A Solver finds the solution (Policy function / Value function) that satisfies
    the optimality conditions defined by the Model Primitives.
    
    Implementation Note:
        Concrete solvers should store their specific logic (Utility, Distribution)
        internally. The `solve` method strictly takes the parameters and the environment.
    """
    def solve(
        self,
        params: PyTree,
        model: StructuralModel, 
    ) -> Any:
        """
        Executes the solution algorithm.

        Args:
            params: Structural parameters :math:`\\theta`.
            model: The economic environment :math:`(S, A, ...)`.

        Returns:
            Any: A result object (e.g., SolverResult) containing the solution.
        """
        ...
