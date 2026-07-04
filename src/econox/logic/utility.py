# src/econox/logic/utility.py
"""
Utility components for the Econox framework.
"""

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
from typing import Callable
from jaxtyping import Float, Array, PRNGKeyArray, PyTree

from econox.protocols import StructuralModel, Utility, Distribution
from econox.utils import get_from_pytree, set_in_pytree

class FunctionUtility(eqx.Module):
    """
    Wraps a user-defined function to satisfy the Utility protocol.
    Allows defining utility logic as a simple function.

    Attributes:
        func (Callable):
            A function with signature `(params: PyTree, model: StructuralModel) -> Float[Array, "num_states num_actions"]`.
            This function computes the flow utility matrix for the given parameters and model.
    
    Example:
        >>> import econox as ecx
        >>> # Define a custom utility function
        >>> def my_utility(params, model):
        ...     # params["beta"] is a scalar, model.data["x"] is (num_states, num_actions)
        ...     return params["beta"] * model.data["x"]
        >>> # Wrap it as a FunctionUtility
        >>> utility: econox.protocols.Utility = ecx.utility(my_utility)
        >>> u = utility.compute_flow_utility(params, model)
        >>> u.shape
        (num_states, num_actions)
    """
    func: Callable

    def compute_flow_utility(
        self, 
        params: PyTree, 
        model: StructuralModel
    ) -> Float[Array, "num_states num_actions"]:
        """
        Calls the user-defined function to compute flow utility.
        Args:
            params: Parameter PyTree.
            model: StructuralModel instance.

        Returns:
            Float[Array, "num_states num_actions"]: 
                The computed flow utility matrix.
        """
        result = self.func(params, model)
        if result.shape != (model.num_states, model.num_actions):
            raise ValueError(f"Utility function returned shape {result.shape}, expected {(model.num_states, model.num_actions)}")
        return result


def utility(func: Callable) -> FunctionUtility:
    """
    Decorator to convert a function into a Utility module.

    This allows you to define a utility function with the standard signature and
    automatically wrap it as a module compatible with the Econox framework.

    Args:
        func (Callable): 
            A function with signature `(params: PyTree, model: StructuralModel) -> Float[Array, "num_states num_actions"]`.
            The function should compute and return the flow utility matrix for the given parameters and model.
    
    Returns:
        FunctionUtility: 
            An object with a `compute_flow_utility(params, model)` method that calls the provided function.
    
    Example:
        >>> import econox as ecx
        >>> # Define a custom utility function
        >>> @ecx.utility
        ... def my_utility(params, model):
        ...     # params["beta"] is a scalar, model.data["x"] is (num_states, num_actions)
        ...     return params["beta"] * model.data["x"]
        >>> # my_utility is now a FunctionUtility instance
        >>> u = my_utility.compute_flow_utility(params, model)
        >>> u.shape
        (num_states, num_actions)
    """
    return FunctionUtility(func=func)


class LinearUtility(eqx.Module):
    """
    Computes flow utility as a linear combination of features and parameters.
    
    This module implements the standard linear utility specification:
    .. math:: U(s, a) = \\sum_{k} \\beta_k \\cdot X_k(s, a)

    It expects parameters to be provided as individual scalars (or consistent arrays) 
    corresponding to the last dimension of the feature tensor. This design enforces 
    explicit naming of parameters, facilitating integration with `ParameterSpace` 
    for constraints and interpretation.

    Attributes:
        param_keys (tuple[str, ...]): 
            A sequence of keys to retrieve coefficients from the parameter PyTree.
            Order must match the last dimension of the feature tensor.
            Example: `("beta_income", "beta_distance")`
        feature_key (str): 
            Key to retrieve the feature tensor from `model.data`.
            The tensor must have shape `(num_states, num_actions, num_features)`.

    Example:
        >>> # Model data has features shape (100, 5, 2) -> 2 features
        >>> utility = LinearUtility(param_keys=("beta_0", "beta_1"), feature_key="X")
        >>> # Params must contain "beta_0" and "beta_1"
        >>> u = utility.compute_flow_utility(params, model)
        >>> u.shape
        (100, 5)
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
        Calculates flow utility using matrix multiplication.

        The method retrieves parameters specified by `param_keys`, stacks them into 
        a single vector, and computes the dot product with the feature tensor.

        Args:
            params: Parameter PyTree containing the coefficients. 
                    Values for `param_keys` should typically be scalars.
            model: StructuralModel containing the feature tensor at `model.data[self.feature_key]`.
                   Expected shape: `(num_states, num_actions, num_features)`.

        Returns:
            Float[Array, "num_states num_actions"]: 
                The calculated flow utility matrix.

        Raises:
            ValueError: If the feature tensor is not 3D.
            ValueError: If the number of `param_keys` does not match the 
                        last dimension (num_features) of the feature tensor.
            ValueError: If parameters cannot be stacked (e.g., shape mismatch).
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

class MixedUtility(eqx.Module):
    """
    Wrapper for random coefficients utility (mixed models).
    """
    base_utility: Utility
    """Base utility function (e.g., LinearUtility)."""
    distribution: Distribution
    """Distribution of random coefficients (e.g., Normal, Gumbel)."""
    draws: np.ndarray
    """Pre-generated random draws. Stored as np.ndarray so eqx.filter_grad excludes it from differentiation."""
    coeff_keys: tuple[str, ...]
    """Keys in params for random coefficients."""
    scale_keys: tuple[str, ...]
    """Keys in params for scales of random coefficients."""

    def __init__(
        self,
        base_utility: Utility,
        distribution: Distribution,
        coeff_keys: tuple[str, ...],
        scale_keys: tuple[str, ...],
        num_draws: int,
        num_params: int,
        key: PRNGKeyArray
    ):
        """
        Initializes MixedUtility by generating standard draws.

        Args:
            base_utility: Base utility function (e.g., LinearUtility).
            distribution: Distribution of random coefficients.
            coeff_keys: Keys in params for random coefficients.
            scale_keys: Keys in params for scales of random coefficients.
            num_draws: Number of random draws to generate.
            num_params: Number of random coefficients.
            key: Random key for reproducibility.
        """
        self.base_utility = base_utility
        self.distribution = distribution
        self.coeff_keys = coeff_keys
        self.scale_keys = scale_keys
        self.draws = np.array(distribution.generate_standard_draws(
            key, shape=(num_draws, num_params)
        ))

    def compute_flow_utility(
        self,
        params: PyTree,
        model: StructuralModel
    ) -> Float[Array, "num_draws n_states n_actions"]:
        """
        Computes flow utility for mixed models with random coefficients.

        Args:
            params: Parameter PyTree containing means and scales.
            model: StructuralModel instance.

        Returns:
            Float[Array, "num_draws n_states n_actions"]:
                The computed flow utility matrix for each draw.
        """
        loc = jnp.array([
            get_from_pytree(params, k) for k in self.coeff_keys
        ])
        scale = jnp.array([
            get_from_pytree(params, k) for k in self.scale_keys
        ])

        # Transform all draws at once outside vmap: (num_draws, num_params)
        all_rand_coeffs = self.distribution.transform(
            draws=jnp.asarray(self.draws),
            loc=loc,
            scale=scale,
        )

        def single_draw_utility(rand_coeffs):
            current_params = params
            for i, k in enumerate(self.coeff_keys):
                current_params = set_in_pytree(current_params, k, rand_coeffs[i])
            return self.base_utility.compute_flow_utility(current_params, model)

        return jax.vmap(single_draw_utility)(all_rand_coeffs)
