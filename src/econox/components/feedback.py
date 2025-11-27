# src/econox/components/feedback.py
"""
Feedback mechanisms for General Equilibrium (GE) interactions.
Updates environmental variables (e.g., Prices) based on agent distributions.
"""

import jax.numpy as jnp
import equinox as eqx
from typing import Any
from jaxtyping import PyTree, Array, Float

from econox.protocols import StructuralModel

class LogLinearFeedback(eqx.Module):
    """
    Updates a target variable based on population density using a log-linear rule.
    
    Physics:
        ln(Target) = Intercept + Elasticity * ln(Density)
        Density = (Population_Share * Total_Population) / Area_Size
    """
    target_data_key: str
    result_metric_key: str
    elasticity_param_key: str
    intercept_param_key: str
    area_data_key: str
    total_pop_data_key: str

    def __init__(
        self,
        target_data_key: str,          # e.g., "rent" (The variable to update)
        result_metric_key: str,        # e.g., "population_distribution" (From solver result)
        elasticity_param_key: str,     # e.g., "rent_elasticity"
        intercept_param_key: str,      # e.g., "rent_intercepts"
        area_data_key: str = "area_size",
        total_pop_data_key: str = "total_pop"
    ) -> None:
        self.target_data_key = target_data_key
        self.result_metric_key = result_metric_key
        self.elasticity_param_key = elasticity_param_key
        self.intercept_param_key = intercept_param_key
        self.area_data_key = area_data_key
        self.total_pop_data_key = total_pop_data_key

    def update(
        self, 
        params: PyTree, 
        current_result: Any, 
        model: StructuralModel
    ) -> StructuralModel:
        """
        Calculates the new equilibrium values and returns a generic updated model.
        
        Args:
            params: Parameter PyTree containing elasticity and intercepts.
            current_result: Solver result containing the population distribution.
            model: Current StructuralModel containing constants (area, total_pop).

        Returns:
            A new StructuralModel instance with updated data.
        """
        # 1. Retrieve Parameters
        # Elasticity: scalar or (num_states,)
        elasticity: PyTree = params[self.elasticity_param_key]
        # Intercept: scalar or (num_states,)
        # Note: If intercepts are defined per "Area" but the model has "States",
        # params[key] should already be mapped to (num_states,) by ParameterSpace
        # or be broadcastable.
        intercept: PyTree = params[self.intercept_param_key]

        # 2. Retrieve Data from Model and Result
        model_data: PyTree = model.data
        
        # Result Metric (Share): (num_states,)
        pop_share: Float[Array, "num_states"] = getattr(current_result, self.result_metric_key)
        
        # Constants
        area_size = model_data[self.area_data_key]          # (num_states,)
        total_pop = model_data.get(self.total_pop_data_key, 1.0) # scalar or (num_states,)

        # 3. Calculate Density
        # Density = (Share * Total) / Area
        abs_population = pop_share * total_pop
        density = abs_population / area_size
        
        # 4. Compute Log-Linear Update
        # Formula: ln(Y) = alpha + beta * ln(Density)
        # Using maximum(..., 1e-8) for numerical stability
        ln_density: Array = jnp.log(jnp.maximum(density, 1e-8))
        pred_ln_val: Array = intercept + elasticity * ln_density

        # 5. Return New Model with Updated Data
        # Since StructuralModel is immutable (via Protocol), we rely on a method
        # `replace_data` if it exists, or assume a specific implementation.
        # For the generic protocol, we assume the user's Model class has a way to update.
        if hasattr(model, "replace_data"):
             return model.replace_data(self.target_data_key, pred_ln_val) # type: ignore
        
        # Fallback: Raise error if the model doesn't support updates
        raise NotImplementedError(
            f"The model class {type(model).__name__} does not implement 'replace_data' method required for feedback."
        )