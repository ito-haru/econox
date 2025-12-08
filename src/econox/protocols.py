# src/econox/protocols.py
"""
Protocol definitions for the Econox framework.
"""

from __future__ import annotations
from typing import Protocol, Any, TypeAlias, Callable, runtime_checkable
from jaxtyping import Array, Float, PyTree

Scalar: TypeAlias = Float[Array, ""]

# =============================================================================
# 1. Data Containers (Model)
# =============================================================================

@runtime_checkable
class StructuralModel(Protocol):
    """
    Container that holds the "environment" of an economic model.
    Contains state space, transition probabilities, constant data, etc.
    """
    @property
    def num_states(self) -> int: ...
    
    @property
    def num_actions(self) -> int: ...

    @property
    def num_periods(self) -> int | float:
        """
        Number of periods in the model.
        Should be a positive integer for finite horizon or `np.inf` for infinite horizon.
        """
        ...

    @property
    def data(self) -> PyTree:
        """
        Holds environment constants (features, matrices, etc.).
        Ideally a PyTree (dict, NamedTuple, etc.).
        """
        ...

    @property
    def transitions(self) -> PyTree | None:
        """Transition logic or matrix (e.g., P(s'|s,a) or Adjacency)."""
        ...
        
    @property
    def availability(self) -> PyTree | None:
        """Availability mask for actions."""
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
    Utility function (Instantaneous Utility / Reward).
    Takes parameters and data, returns a utility matrix of shape (n_states, n_actions).
    """
    def compute_flow_utility(self, params: PyTree, model: StructuralModel) -> Float[Array, "n_states n_actions"]:
        ...

@runtime_checkable
class Distribution(Protocol):
    """
    Distribution of error terms (Stochasticity).
    Provides computation logic for expected maximum value (Emax) and choice probabilities (P).
    """
    def expected_max(self, values: Float[Array, "n_states n_actions"]) -> Float[Array, "n_states"]:
        """E[max(v + epsilon)]"""
        ...

    def choice_probabilities(self, values: Float[Array, "n_states n_actions"]) -> Float[Array, "n_states n_actions"]:
        """P(a|s)"""
        ...

@runtime_checkable
class FeedbackMechanism(Protocol):
    """
    Equilibrium feedback (General Equilibrium / Game Interaction).
    Receives aggregated results and updates model state (such as prices).
    """
    def update(self, params: PyTree, current_result: Any, model: StructuralModel) -> StructuralModel:
        ...

@runtime_checkable
class Dynamics(Protocol):
    """
    Protocol for the Law of Motion logic.
    Physics: D_{t+1} = f(D_t, Policy_t, Model)
    """
    def __call__(
        self, 
        distribution: Float[Array, "num_states"], 
        policy: Float[Array, "num_states num_actions"], 
        model: StructuralModel
    ) -> Float[Array, "num_states"]:
        ...

# =============================================================================
# 3. Core Engine (Solver)
# =============================================================================

@runtime_checkable
class Solver(Protocol):
    """
    Computational engine.
    Uses Utility and Distribution to solve for fixed points or optimal policies.
    """
    def solve(
        self,
        params: PyTree,
        model: StructuralModel, 
        utility: Utility, 
        dist: Distribution, 
        feedback: FeedbackMechanism | None = None
    ) -> Any:
        ...
