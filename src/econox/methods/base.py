# src/econox/methods/base.py
"""
Base module for method functions in the Econox framework.
"""

from __future__ import annotations
from typing import Callable, Any
from abc import abstractmethod
import equinox as eqx
from jaxtyping import PyTree, Scalar

from econox.protocols import StructuralModel
from econox.structures import EstimationResult
from econox.methods.variance import Variance


class EstimationMethod(eqx.Module):
    """
    Base class for all estimation method functions in Econox.
    
    This class serves three main purposes:
    1. **Strategy Definition**: Defines the loss function to be minimized during numerical estimation.
    2. **Analytical Solution**: Optionally provides a direct solution method (e.g., for OLS/2SLS).
    3. **Inference**: Optionally defines how to calculate standard errors (e.g., Hessian, Sandwich).

    Users can create custom objectives by subclassing this class or by using the 
    `@method_from_loss` decorator.

    Attributes:
        variance: Variance | None
    """
    variance: Variance | None = eqx.field(default=None, kw_only=True)
    """
    Optional variance calculation strategy for inference.
    """

    @abstractmethod
    def compute_loss(
        self,
        result: Any | None, 
        observations: Any,
        params: PyTree, 
        model: StructuralModel
    ) -> Scalar:
        """
        Calculates the scalar loss metric to be minimized.
        
        This method is the core of the numerical estimation loop. It compares the 
        model's prediction (`result`) with the real-world data (`observations`).

        Args:
            result: The output from the Solver (e.g., `SolverResult`). 
                    If an analytical solution is being evaluated, this may be None.
            observations: Observed data to fit the model against.
            params: Current model parameters (useful for regularization terms).
            model: The structural model environment.

        Returns:
            A scalar JAX array representing the loss (e.g., Negative Log-Likelihood).
        """
        ...

    def solve(
        self,
        model: StructuralModel,
        observations: Any,
        param_space: Any
    ) -> EstimationResult | None:
        """
        Computes the analytical solution for the parameters, if available.

        This method allows the `Estimator` to bypass the numerical optimization loop 
        for models that have a closed-form solution (e.g., OLS, 2SLS).

        Args:
            model: The structural model environment.
            observations: Observed data.
            param_space: The parameter space definition.

        Returns:
            EstimationResult | None:
            Returns an ``EstimationResult`` if an analytical solution is found.
            Returns ``None`` otherwise (default), and the Estimator will fall back 
            to numerical optimization using `compute_loss`.
        """
        return None

    @classmethod
    def from_function(cls, func: Callable) -> EstimationMethod:
        """
        Creates an `EstimationMethod` instance from a simple loss function.
        
        This factory method allows users to define objectives using a simple function 
        instead of defining a full class. The created objective will rely on numerical 
        optimization (solve returns None) and will not compute standard errors by default.

        Args:
            func: A function with the signature:
                  `(result, observations, params, model) -> Scalar`

        Returns:
            An instance of a dynamically created `EstimationMethod` subclass.

        Example:
            >>> @method_from_loss
            ... def mse_loss(result, observations, params, model):
            ...     return jnp.mean((result.solution - observations) ** 2)
        """
        # Dynamically create a subclass to wrap the function
        class WrapperMethod(EstimationMethod):
            def compute_loss(self, result, observations, params, model):
                return func(result, observations, params, model)

            def __repr__(self):
                return f"WrapperMethod({func.__name__})"

        return WrapperMethod()

# Alias for decorator usage
method_from_loss = EstimationMethod.from_function
