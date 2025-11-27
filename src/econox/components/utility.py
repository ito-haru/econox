# src/econox/components/utility.py
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
    feature_key: str

    def __init__(self, param_keys: Sequence[str], feature_key: str = "features") -> None:
        """
        Args:
            param_keys: List of keys to extract from the params PyTree.
                        Order must match the last dimension of the feature tensor.
            feature_key: The key in model.data where the feature tensor is stored.
                         Default is "features".
        """
        self.param_keys = tuple(param_keys)
        self.feature_key = feature_key

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
        # 1. Stack parameters into a vector (beta)
        # Assuming params is a dictionary-like PyTree.
        # Shape: (num_features,)
        coeffs: Float[Array, "num_features"] = jnp.stack(
            [get_from_pytree(params, k) for k in self.param_keys]
        )

        # 2. Retrieve the feature tensor
        # Shape: (num_states, num_actions, num_features)
        X: Float[Array, "num_states num_actions num_features"] = model.data[self.feature_key]

        # 3. Compute dot product
        # Contract over the feature dimension (last dimension)
        flow_utility: Float[Array, "num_states num_actions"] = jnp.einsum(
            "saf, f -> sa", X, coeffs
        )
        
        return flow_utility