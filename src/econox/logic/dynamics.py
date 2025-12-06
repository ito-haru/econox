# src/econox/logic/dynamics.py
"""
Dynamics logic components for the Econox framework.
Defines how the population/state distribution evolves over time (Law of Motion).
"""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree

from econox.protocols import StructuralModel
from econox.utils import get_from_pytree


class SimpleDynamics(eqx.Module):
    """
    Standard Law of Motion for dynamic models.
    
    Supports two modes:

    1. Explicit Transition: :math:`D_{t+1} = (D_t * P(a|s)) @ T(s'|s,a)`
       Used when a transition matrix is provided in the model.
       
    2. Direct Mapping: :math:`D_{t+1} = D_t @ P(a|s)`
       Used when no transition matrix is provided.
       Assumes Action Space maps 1-to-1 to State Space (A=S).
    """
    # If True, enforces usage of transition matrix. 
    # If False (default), it auto-detects based on model.transitions.
    use_transitions: bool = eqx.field(default=False, static=True)
    
    def __call__(
        self, 
        distribution: Float[Array, "num_states"], 
        policy: Float[Array, "num_states num_actions"], 
        model: StructuralModel
    ) -> Float[Array, "num_states"]:
        
        # 1. Try to use Explicit Transition Matrix
        # Auto-detect: if use_transitions is True OR model has transitions data
        trans_mat = model.transitions
        
        if self.use_transitions or trans_mat is not None:
            if trans_mat is None:
                raise ValueError("SimpleDynamics: use_transitions=True but model.transitions is None.")
            
            # Calculate Mass Movement (Flow)
            # (S,) * (S, A) -> (S, A)
            mass_sa = distribution[:, None] * policy
            
            # Flatten to (S*A,) and apply transition
            mass_flat = mass_sa.ravel()
            
            # (S*A,) @ (S*A, S) -> (S,)
            next_dist = mass_flat @ trans_mat
            return next_dist
        
        # 2. Fallback to Direct Mapping (A=S Assumption)
        else:
            if model.num_states != model.num_actions:
                raise ValueError(
                    f"SimpleDynamics (Direct Mode) requires num_states ({model.num_states}) == "
                    f"num_actions ({model.num_actions}).\n"
                    "If your model has complex transitions (S != A), please provide "
                    "'transitions' in the model or use a custom Dynamics class."
                )

            # (S,) @ (S, A) -> (A,) interpreted as (S,)
            return distribution @ policy


class TrajectoryDynamics(eqx.Module):
    """
    Dynamics for solving path-dependent problems (e.g., Rational Expectations).
    
    Handles the evolution of the entire state trajectory over a finite horizon,
    often involving:
    1. Explicit Transition Matrices (S*A -> S) to handle complex flows.
    2. Boundary Conditions (e.g., Fixing t=0 population).
    
    This class expects the following in the model:
    - The transition matrix must be accessible via the `transitions` property of the model (i.e., `model.transitions` should return a (S*A, S) matrix defining physical movement).
    - "initial_year_indices": Indices to enforce boundary conditions (in `model.data`).
    - "initial_year_values": Values to enforce at the boundary (in `model.data`).
    """
    enforce_boundary: bool = eqx.field(default=True, static=True)

    def __call__(
        self, 
        distribution: Float[Array, "num_states"], 
        policy: Float[Array, "num_states num_actions"], 
        model: StructuralModel
    ) -> Float[Array, "num_states"]:
        
        # Retrieve Data (Safe access via helper is recommended in production)
        trans_mat = model.transitions
        
        if trans_mat is None:
            raise ValueError("TrajectoryDynamics requires a 'transitions' matrix in the model.")
            
        # Calculate Flow (Mass Movement)
        # (S,) * (S, A) -> (S, A)
        mass_sa = distribution[:, None] * policy
        
        # Flatten to (S*A,) to apply transition matrix
        mass_flat = mass_sa.ravel()
        
        # Apply Time Evolution (Trajectory Update)
        # (S*A,) @ (S*A, S) -> (S,)
        next_path = mass_flat @ trans_mat

        # Return if no boundary enforcement is needed
        if not self.enforce_boundary:
            return next_path
        
        # Apply Boundary Conditions (Fix t=0)
        data: PyTree = model.data
        init_idx = get_from_pytree(data, "initial_year_indices", None)
        init_val = get_from_pytree(data, "initial_year_values", None)
        
        if init_idx is None or init_val is None:
            raise ValueError(
                "TrajectoryDynamics requires 'initial_year_indices' and "
                "'initial_year_values' in model.data for boundary conditions."
            )

        init_idx = jnp.asarray(init_idx)
        init_val = jnp.asarray(init_val)

        if init_idx.shape[0] != init_val.shape[0]:
            raise ValueError(
                "'initial_year_indices' and 'initial_year_values' must have the same length."
            )
        next_path = next_path.at[init_idx].set(init_val)
        
        return next_path
