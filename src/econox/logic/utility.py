# src/econox/logic/utility.py
"""
Utility components for the Econox framework.
"""

import jax.numpy as jnp
import equinox as eqx
from typing import Sequence
from jaxtyping import Float, Array, PyTree

from econox.protocols import StructuralModel
from econox.utils import get_from_pytree

class LinearUtility(eqx.Module):
    """
    Computes flow utility as a linear combination of features and parameters.
    Formula: U(s, a) = sum_k ( param_k * feature_k(s, a) )
    """
    param_keys: tuple[str, ...]
    """Keys in params for the coefficients corresponding to each feature."""
    feature_key: str
    """Key in model.data for the feature tensor of shape (num_states, num_actions, num_features)."""

    def compute_flow_utility(
        self, 
        params: PyTree, 
        model: StructuralModel
    ) -> Float[Array, "num_states num_actions"]:
        """
        Calculates flow utility using matrix multiplication (einsum).
        
        Args:
            params: Parameter PyTree containing the coefficients.
            model: StructuralModel containing the feature tensor.
                   Expected shape: (num_states, num_actions, num_features)

        Returns:
            Flow utility matrix of shape (num_states, num_actions).
        """
        # 1. Retrieve Feature Tensor
        X = get_from_pytree(model.data, self.feature_key)
        if X.ndim != 3:
            raise ValueError(f"Feature '{self.feature_key}' must be 3D (states, actions, features), got {X.shape}")

        # 2. Retrieve & Process Parameters (Concise & Robust)
        # Extract all params, ensure they are arrays, stack them, and flatten to 1D.
        # This handles both scalars (0.5) and 1-element arrays (jnp.array([0.5])) gracefully.
        try:
            coeffs_list = [jnp.asarray(get_from_pytree(params, k)) for k in self.param_keys]
            coeffs = jnp.stack(coeffs_list).flatten()
        except Exception as e:
            raise ValueError(f"Failed to stack parameters {self.param_keys}: {e}")

        # 3. Validation: Ensure parameter count matches feature dimension
        if coeffs.shape[0] != X.shape[-1]:
            raise ValueError(
                f"Dimension mismatch: Feature '{self.feature_key}' has {X.shape[-1]} dims, "
                f"but {coeffs.shape[0]} parameters were provided ({self.param_keys})."
            )

        # 4. Compute Utility (Dot Product)
        return jnp.einsum("saf, f -> sa", X, coeffs)