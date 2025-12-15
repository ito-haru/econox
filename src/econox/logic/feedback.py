"""
Feedback mechanisms for General Equilibrium (GE) interactions.

This module provides the infrastructure for defining and executing feedback loops
in structural models. It supports both component-wise updates (FunctionFeedback)
and full-model updates (CustomUpdateFeedback).
"""

import equinox as eqx
import jax.numpy as jnp
from typing import Sequence, Callable, Any
from jaxtyping import PyTree
from econox.protocols import StructuralModel, FeedbackMechanism


class CompositeFeedback(eqx.Module):
    """
    A container that executes multiple feedback mechanisms sequentially.
    Useful when you want to chain multiple update steps.
    """
    feedbacks: Sequence[FeedbackMechanism]

    def update(
        self, 
        params: PyTree, 
        current_result: Any, 
        model: StructuralModel
    ) -> StructuralModel:
        """Sequential application of feedback mechanisms."""
        for fb in self.feedbacks:
            model = fb.update(params, current_result, model)
        return model


class CustomUpdateFeedback(eqx.Module):
    """
    A feedback mechanism that allows the user to define a custom function
    to update the ENTIRE model structure.
    
    This is the most flexible approach, allowing for complex dependencies
    (e.g., wage and rent depending on the same density calculation) without
    redundant computations or shape mismatches.

    Attributes:
        func: A callable with signature (params, result, model) -> StructuralModel.
    """
    func: Callable

    def update(
        self, 
        params: PyTree, 
        current_result: Any, 
        model: StructuralModel
    ) -> StructuralModel:
        """Delegates the update logic entirely to the user-defined function."""
        return self.func(params, current_result, model)


def model_feedback(func: Callable) -> CustomUpdateFeedback:
    """
    Decorator to register a function as a CustomUpdateFeedback.

    Usage:
        @ecx.model_feedback
        def my_ge_loop(params, result, model):
           # ... calculation ...
           return new_model
    """
    return CustomUpdateFeedback(func=func)


class FunctionFeedback(eqx.Module):
    """
    A simpler wrapper for updating a specific key in model.data.
    Best for independent, single-variable updates.
    """
    func: Callable
    target_key: str

    def update(
        self, 
        params: PyTree, 
        current_result: Any, 
        model: StructuralModel
    ) -> StructuralModel:
        # Execute user logic
        new_values = self.func(model.data, params, current_result)
        
        # Update model data safely
        new_data = model.data.copy()
        new_data[self.target_key] = new_values
        
        return eqx.tree_at(lambda m: m.data, model, new_data)


def function_feedback(target_key: str) -> Callable[[Callable[..., Any]], FunctionFeedback]:
    """Decorator for simple single-variable updates.
    Usage:
        @ecx.function_feedback(target_key="wage")
        > def wage_update(data, params, result):
        >   # ... calculation ...
        >   return new_wage_values
    """
    def wrapper(func: Callable) -> FunctionFeedback:
        return FunctionFeedback(func=func, target_key=target_key)
    return wrapper