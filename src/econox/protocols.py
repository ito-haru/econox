# src/econox/protocols.py
"""
Protocol definitions for the Econox framework.
"""

from __future__ import annotations
from typing import Callable, Protocol, Any, TypeAlias, runtime_checkable
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

@runtime_checkable
class ParameterSpace(Protocol):
    """
    Interface for parameter management and transformation.
    Responsible for mutual transformation between real space (for optimization) and model space (with constraints).
    """
    def transform(self, raw_params: PyTree) -> PyTree:
        """Real parameters (Unconstrained) -> Model parameters (Constrained)"""
        ...

    def inverse_transform(self, model_params: PyTree) -> PyTree:
        """Model parameters (Constrained) -> Real parameters (Unconstrained)"""
        ...
    
    def get_bounds(self) -> tuple[PyTree, PyTree] | None:
        """Returns parameter bounds (if necessary)"""
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

# =============================================================================
# 4. Strategy Layer (Estimator Logic)
# =============================================================================

@runtime_checkable
class Objective(Protocol):
    """
    Loss function definition (Strategy).
    Compares solver results with observed data and returns the loss to minimize.
    """
    def compute_loss(
        self,
        params: PyTree, 
        model: StructuralModel, 
        solver: Solver, 
        observations: Any,
        utility: Utility, 
        dist: Distribution | None = None,
        feedback: FeedbackMechanism | None = None
    ) -> Scalar:
        ...

@runtime_checkable
class Optimizer(Protocol):
    """
    Optimization algorithm (Wrapper).
    Takes a loss function and initial values, returns optimal parameters.
    """
    def minimize(self, loss_fn: Callable[[PyTree], Scalar], init_params: PyTree) -> PyTree:
        ...