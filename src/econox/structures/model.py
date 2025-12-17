# src/econox/structures/model.py

from __future__ import annotations
import dataclasses
from typing import Dict, Any
import numpy as np
import jax.numpy as jnp
from jax.experimental import sparse
import equinox as eqx
from jaxtyping import Array, Float, Int

class Model(eqx.Module):
    """
    Immutable container representing the structural environment.
    
    This class holds all exogenous data (states, transitions, covariates) required 
    to define the economic model. It is designed to be purely data-centric and 
    logic-free, ensuring separation between the environment and the behavioral 
    logic (Utility/Solver).

    As an Equinox Module, this class is a valid JAX PyTree, meaning it can be 
    passed into JIT-compiled functions or differentiated with respect to.
    """
    
    # ---Required Fields (Protocol: StructuralModel)---
    
    num_states: int = eqx.field(static=True)
    """Total cardinality of the state space (:math:`S`). Used to determine array shapes."""
    
    num_actions: int = eqx.field(static=True)
    """Total cardinality of the action space (:math:`A`)."""
    
    data: Dict[str, Float[Array, "..."]]
    """
    Dictionary of environment constants and exogenous variables.
    
    Keys are identifiers (e.g., 'wage', 'distance', 'rent') that must match 
    the keys expected by the `Utility` component. Values are typically JAX arrays 
    of shape (:math:`S`, :math:`A`), (:math:`S`,), or scalars.
    """

    # ---Optional Fields & Protocol Extensions---
    
    num_periods: int | float = eqx.field(default=np.inf, static=True)
    """
    Time horizon (:math:`T`) of the model.
    
    * `np.inf`: Infinite horizon (default).
    * `int`: Finite horizon.
    """
    
    availability: Int[Array, "num_states num_actions"] | None = None
    """
    Binary mask indicating feasible actions. 
    Shape (:math:`S`, :math:`A`). `1` (or `True`) indicates action :math:`a` is available in state :math:`s`, 
    while `0` (or `False`) indicates it is physically impossible.
    """
    
    transitions: Float[Array, "..."] | sparse.BCOO | None = None
    """
    Exogenous transition structure.
    
    Depending on the model type, this could be:
    * Transition Probability Matrix: :math:`P(s' | s, a)`
    * Adjacency Matrix (for spatial models)
    * Deterministic mapping logic
    """

    def __check_init__(self):
        if self.num_periods != np.inf and self.num_periods % 1 != 0:
            raise ValueError("num_periods must be an integer if finite.")
    
    # ---Factory Method---
    @classmethod
    def from_data(
        cls,
        num_states: int,
        num_actions: int,
        data: Dict[str, Any],
        availability: Any | None = None,
        transitions: Any | None = None,
        num_periods: int | float = np.inf,
    ) -> Model:
        """
        Factory method to initialize a Model from raw Python/NumPy data.
        
        This is the recommended entry point. It automatically handles the conversion 
        of Python lists and NumPy arrays into JAX DeviceArrays, ensuring compatibility 
        with JIT compilation.

        Args:
            num_states: Total number of states (:math:`S`). Must be positive.
            num_actions: Total number of actions (:math:`A`). Must be positive.
            data: Dictionary of feature matrices (e.g., `{'wage': [...]}`).
                  Keys should be strings, values can be lists, NumPy arrays, or JAX arrays.
            availability: Optional mask for feasible actions. Shape must be (num_states, num_actions).
            transitions: Optional transition matrices.
            num_periods: Time horizon. Defaults to `np.inf` (Infinite). Must be positive.

        Returns:
            A frozen, JAX-ready `Model` instance.
            
        Raises:
            ValueError: If dimensions are invalid.
            TypeError: If data types are incompatible.
            
        Examples:
            >>> # Infinite horizon model
            >>> model = Model.from_data(
            ...     num_states=10,
            ...     num_actions=3,
            ...     data={'wage': np.random.randn(10, 3)}
            ... )
            
            >>> # Finite horizon with availability constraints
            >>> model = Model.from_data(
            ...     num_states=5,
            ...     num_actions=2,
            ...     data={'utility': [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]},
            ...     availability=np.ones((5, 2)),
            ...     num_periods=10
            ... )
        """
        # Validate dimensions
        if num_states <= 0:
            raise ValueError(f"num_states must be positive, got {num_states}")
        if num_actions <= 0:
            raise ValueError(f"num_actions must be positive, got {num_actions}")
        if num_periods is not None and num_periods <= 0:
            raise ValueError(f"num_periods must be positive or None, got {num_periods}")
        
        # Validate and convert data dictionary
        if not isinstance(data, dict):
            raise TypeError(f"data must be a dictionary, got {type(data).__name__}")
        if not data:
            raise ValueError("data dictionary cannot be empty")
        
        try:
            jax_data = {k: jnp.asarray(v) for k, v in data.items()}
        except (ValueError, TypeError) as e:
            raise TypeError(f"Failed to convert data to JAX arrays: {e}") from e
        
        # Validate and convert availability
        jax_avail = None
        if availability is not None:
            try:
                jax_avail = jnp.asarray(availability)
            except (ValueError, TypeError) as e:  
                raise TypeError(f"Failed to convert availability to JAX array: {e}") from e  
            
            if jax_avail.shape != (num_states, num_actions):
                raise ValueError(
                    f"availability shape {jax_avail.shape} doesn't match "
                    f"expected ({num_states}, {num_actions})"
                )
        
        # Validate and convert transitions
        jax_trans = None
        if transitions is not None:
            if isinstance(transitions, (jnp.ndarray, sparse.BCOO)):
                jax_trans = transitions
            else:
                try:
                    jax_trans = jnp.asarray(transitions)
                except (ValueError, TypeError) as e:
                    raise TypeError(f"Failed to convert transitions to JAX array: {e}") from e

        return cls(
            num_states=num_states,
            num_actions=num_actions,
            data=jax_data,
            availability=jax_avail,
            transitions=jax_trans,
            num_periods=num_periods,
        )

    # ---Protocol Implementation (Immutable Setter)---
    def replace_data(self, key: str, value: Any) -> Model:
        """
        Creates a NEW Model instance with specific data updated.
        
        This method is essential for counterfactual simulations. It allows you 
        to create a "What-if" world (e.g., "What if subsidies increase by 10%?") 
        without mutating the original baseline model.

        Args:
            key: The name of the data field to update (must exist in `data`).
            value: The new value for that field.

        Returns:
            A new `Model` instance with the updated data.
        """
        # Validate key existence
        if key not in self.data:
            raise KeyError(
                f"Key '{key}' not found in model data. "
                f"Available keys: {list(self.data.keys())}"
            )
    
        new_data = self.data.copy()
        new_data[key] = jnp.asarray(value)

        return dataclasses.replace(self, data=new_data)

    # ---Helpers (Backward Compatibility / Convenience)---
    @property
    def features(self) -> Dict[str, Array]:
        """Alias for `data`, provided for semantic convenience."""
        return self.data

    def get(self, name: str) -> Array:
        """
        Retrieves a data array by name with error handling.
        
        Raises:
            KeyError: If the name is not found in `data`.
        """
        if name not in self.data:
            raise KeyError(f"Data '{name}' not found in model.")
        return self.data[name]
    
    @property
    def shapes(self) -> Dict[str, tuple[int, ...]]:
        """Returns the shape of each array in `data` for debugging."""
        return {k: v.shape for k, v in self.data.items()}