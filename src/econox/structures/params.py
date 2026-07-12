# src/econox/structures/params.py

from __future__ import annotations
from enum import StrEnum
from typing import Dict, Any
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import PyTree

from econox.config import LOG_CLIP_MIN, LOG_CLIP_MAX, NUMERICAL_EPSILON


class ConstraintKind(StrEnum):
    """
    Specifies the type of constraint applied to a parameter.

    Being a :class:`~enum.StrEnum`, each member compares equal to its string
    value (``ConstraintKind.FIXED == "fixed"``), so the plain-string spelling
    keeps working at runtime while ``ecx.ConstraintKind.FIXED`` is the
    type-checked form.

    Options:
        - **FREE**: No constraints (-inf, +inf).
        - **POSITIVE**: Must be positive (0, +inf). Used for variances, etc.
        - **NEGATIVE**: Must be negative (-inf, 0).
        - **PROBABILITY**: Constrained to (0, 1).
        - **UNIT_INTERVAL**: Alias for PROBABILITY.
        - **FIXED**: Parameter is fixed to its initial value and not optimized.
        - **BOUNDED**: Constrained to a specific range [lower, upper].
    """

    FREE = "free"                   # (-inf, +inf)
    POSITIVE = "positive"           # (0, +inf)
    NEGATIVE = "negative"           # (-inf, 0)
    PROBABILITY = "probability"     # (0, 1)
    UNIT_INTERVAL = "unit_interval"  # (0, 1) Alias for PROBABILITY
    FIXED = "fixed"                 # Fixed value
    BOUNDED = "bounded"             # (lower, upper)

class ParameterSpace(eqx.Module):
    """
    Manages parameter constraints and transformations with numerical stability.
    Compliant with the ParameterSpace protocol.
    
    Handles the mapping between:
    1. Raw Parameters (Real space, R^n): For the optimizer.
    2. Model Parameters (Constrained space): For the economic model.

    Examples:
        >>> # Define initial values
        >>> init_params = {
        ...     "beta": 0.95,
        ...     "sigma": 1.0,
        ...     "alpha": 0.5,
        ...     "gamma": 2.0
        ... }
        
        >>> # Define constraints (ConstraintKind members or their string values)
        >>> constraints = {
        ...     "beta": ConstraintKind.FIXED,        # Not optimized
        ...     "sigma": ConstraintKind.POSITIVE,    # Domain: (0, inf)
        ...     "alpha": ConstraintKind.PROBABILITY, # Domain: (0, 1)
        ...     "gamma": ConstraintKind.FREE         # Domain: (-inf, inf) (Default)
        ... }
        
        >>> # Create the parameter space
        >>> pspace = ParameterSpace.create(init_params, constraints)
    """
    
    # ---Fields (Immutable)---
    initial_params: Dict[str, Any]
    """Initial values of the parameters (Constrained space)."""
    
    constraints: Dict[str, ConstraintKind]
    """Dictionary mapping parameter names to their constraint types."""
    
    bounds: Dict[str, tuple[float, float]]
    """Dictionary mapping parameter names to (lower, upper) bounds."""

    # ---Numerical Stability Constants---
    eps: float = eqx.field(default=NUMERICAL_EPSILON, static=True)
    """Small constant for numerical stability."""
    
    log_clip_min: float = eqx.field(default=LOG_CLIP_MIN, static=True)
    """Minimum value for log transformations."""
    
    log_clip_max: float = eqx.field(default=LOG_CLIP_MAX, static=True)
    """Maximum value for log transformations."""

    # ---Factory Method (Replace __init__)---
    @classmethod
    def create(
        cls,
        initial_params: Dict[str, Any],
        constraints: Dict[str, ConstraintKind] | None = None,
        bounds: Dict[str, tuple[float, float]] | None = None,
    ) -> ParameterSpace:
        """
        Factory method to initialize ParameterSpace.
        Validates keys, bounds, and fills default constraints (FREE).

        Args:
            initial_params (Dict[str, Any]): Dictionary of initial parameter values.
            constraints (Dict[str, ConstraintKind] | None): Optional dictionary specifying constraints for each parameter. Values may be ConstraintKind members or their string equivalents; they are normalized to ConstraintKind. Defaults to FREE for unspecified parameters.
            bounds (Dict[str, tuple[float, float]] | None): Optional dictionary specifying (lower, upper) bounds for BOUNDED parameters.
        """
        # Validate inputs
        if not initial_params:
            raise ValueError("initial_params cannot be empty")

        # Validate unknown keys in constraints (before normalization for a clean message)
        if constraints:
            unknown_keys = set(constraints.keys()) - set(initial_params.keys())
            if unknown_keys:
                raise ValueError(f"Constraints defined for unknown parameters: {unknown_keys}")

        # Fill defaults and normalize to ConstraintKind (validates each value early)
        filled_constraints: Dict[str, ConstraintKind] = {}
        for k in initial_params.keys():
            if constraints and k in constraints:
                try:
                    filled_constraints[k] = ConstraintKind(constraints[k])
                except ValueError as e:
                    raise ValueError(
                        f"Invalid constraint for '{k}': {constraints[k]!r}. "
                        f"Must be one of {[c.value for c in ConstraintKind]}."
                    ) from e
            else:
                filled_constraints[k] = ConstraintKind.FREE

        # Validate bounds
        filled_bounds = bounds or {}
        for k, kind in filled_constraints.items():
            if kind == ConstraintKind.BOUNDED:
                if k not in filled_bounds:
                    raise ValueError(f"Parameter '{k}' has 'bounded' constraint but no bounds specified.")
                
                # Validate bounds correctness
                lower, upper = filled_bounds[k]
                if lower > upper:
                    raise ValueError(f"Bounds for '{k}' must satisfy lower <= upper, got ({lower}, {upper}).")

                # Treat as fixed if bounds are equal
                elif lower == upper:
                    filled_constraints[k] = ConstraintKind.FIXED

                # Validate initial value within bounds
                init_val = initial_params[k]
                if not (lower <= init_val <= upper):
                     raise ValueError(f"Initial value for '{k}' ({init_val}) is out of bounds ({lower}, {upper}).")

        return cls(
            initial_params=initial_params,
            constraints=filled_constraints,
            bounds=filled_bounds
        )

    # ---Protocol Implementation---
    
    def transform(self, raw_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform raw (unconstrained) parameters to model (constrained) parameters.

        Args:
            raw_params: Dictionary of unconstrained parameters. Fixed parameters 
                       should NOT be included in this dictionary.

        Returns:
            Dictionary of constrained parameters including fixed parameters.

        Raises:
            ValueError: If required (non-fixed) parameters are missing or 
                       unexpected parameters are present.
        """
        if not isinstance(raw_params, dict):
             raise TypeError("ParameterSpace currently expects a dictionary of parameters.")

        input_keys = set(raw_params.keys())
        all_keys = set(self.initial_params.keys())

        # Check for missing required parameters
        required_keys = {
            k for k in all_keys
            if self.constraints.get(k, ConstraintKind.FREE) != ConstraintKind.FIXED
        }
        if not input_keys.issuperset(required_keys):
            missing = required_keys - input_keys
            raise ValueError(f"Missing required parameters in raw_params: {missing}")

        # Check for unexpected extra parameters
        extra = input_keys - all_keys
        if extra:
            raise ValueError(f"Unexpected parameters in raw_params: {extra}")
        

        def _transform_leaf(value, name):
            kind = self.constraints.get(name, ConstraintKind.FREE)

            if kind == ConstraintKind.FIXED:
                # FIXED parameters should return their initial value, not the raw value
                return self.initial_params[name]

            if kind == ConstraintKind.FREE:
                return value

            # Common clip for numerical stability (after free/fixed check)
            clipped = jnp.clip(value, self.log_clip_min, self.log_clip_max)

            if kind == ConstraintKind.POSITIVE:
                return jnp.exp(clipped)
            elif kind == ConstraintKind.NEGATIVE:
                return -jnp.exp(clipped)
            elif kind in (ConstraintKind.PROBABILITY, ConstraintKind.UNIT_INTERVAL):
                return jax.nn.sigmoid(clipped)
            elif kind == ConstraintKind.BOUNDED:
                lower, upper = self.bounds[name]
                normalized = jax.nn.sigmoid(clipped)
                return lower + (upper - lower) * normalized
            else:
                raise ValueError(f"Unhandled constraint type: {kind}")

        return {
            name: _transform_leaf(raw_params.get(name), name)
            for name in self.initial_params.keys()
        }

    def inverse_transform(self, model_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Model parameters (Constrained) -> Raw parameters (Unconstrained).
        """
        if not isinstance(model_params, dict):
            raise TypeError("ParameterSpace currently expects a dictionary of parameters.")
        
        input_keys = set(model_params.keys())
        expected_keys = set(self.initial_params.keys())

        if input_keys != expected_keys:
            missing = expected_keys - input_keys
            extra = input_keys - expected_keys
            error_msg = "Parameter keys mismatch."
            if missing:
                error_msg += f" Missing: {missing}."
            if extra:
                error_msg += f" Extra (unexpected): {extra}."
            raise ValueError(error_msg)
        # -----------------------------------

        def _inv_transform_leaf(value, name):
            kind = self.constraints.get(name, ConstraintKind.FREE)

            if kind == ConstraintKind.FIXED:
                # FIXED parameters: return 0 in raw space (will be ignored by optimizer)
                return 0.0

            if kind == ConstraintKind.FREE:
                return value

            if kind == ConstraintKind.POSITIVE:
                safe_value = jnp.maximum(value, self.eps)
                return jnp.log(safe_value)
            elif kind == ConstraintKind.NEGATIVE:
                safe_value = jnp.maximum(-value, self.eps)
                return jnp.log(safe_value)
            elif kind in (ConstraintKind.PROBABILITY, ConstraintKind.UNIT_INTERVAL):
                safe_value = jnp.clip(value, self.eps, 1.0 - self.eps)
                return jax.scipy.special.logit(safe_value)
            elif kind == ConstraintKind.BOUNDED:
                lower, upper = self.bounds[name]
                denom = upper - lower

                # Handle potential degeneracy in bounds
                is_degenerate = jnp.abs(denom) < self.eps
                safe_denom = jnp.where(is_degenerate, 1.0, denom)

                # Normalize and clip
                normalized = (value - lower) / safe_denom
                normalized_safe = jnp.clip(normalized, 1e-6, 1.0 - 1e-6)

                # Apply logit transformation to the clipped normalized value
                unconstrained = jax.scipy.special.logit(normalized_safe)

                # If degenerate, return 0.0
                return jnp.where(is_degenerate, 0.0, unconstrained)
            else:
                raise ValueError(f"Unhandled constraint type: {kind}")
                
        return {
            name: _inv_transform_leaf(value, name)
            for name, value in model_params.items()
            if self.constraints.get(name, ConstraintKind.FREE) != ConstraintKind.FIXED # Exclude FIXED params
        }
    
    def get_bounds(self) -> tuple[PyTree, PyTree] | None:
        """
        Protocol: Returns parameter bounds for the optimizer.
        
        Returns None because this class uses the 'Transformation Method' (Unconstrained Optimization).
        The optimizer operates on 'raw_params' which are unbounded (-inf, +inf).
        Constraints are enforced via the 'transform' method, not by the optimizer's bound constraints.
        """
        return None

    # ---Helper Properties---
    
    @property
    def fixed_mask(self) -> Dict[str, bool]:
        """
        Returns a boolean mask where True indicates a parameter is FIXED.
        Useful for masking gradients in the Estimator.
        """
        return {
            k: (v == ConstraintKind.FIXED)
            for k, v in self.constraints.items()
        }
    
    @property
    def num_total_params(self) -> int:
        """
        Returns the number of all parameters.
        """
        return len(self.constraints)
    
    @property
    def num_free_params(self) -> int:
        """
        Returns the number of free (non-fixed) parameters.
        """
        return sum(
            1 for kind in self.constraints.values()
            if kind != ConstraintKind.FIXED
        )
