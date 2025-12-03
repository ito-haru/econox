# src/econox/logic/feedback.py
"""
Feedback mechanisms for General Equilibrium (GE) interactions.
Updates environmental variables (e.g., prices) based on agent distributions.
"""

import jax.numpy as jnp
import equinox as eqx
from typing import Any
from jaxtyping import PyTree, Array, Float

from econox.config import LOG_CLIP_MAX, LOG_CLIP_MIN, NUMERICAL_EPSILON
from econox.protocols import StructuralModel
from econox.utils import get_from_pytree

class LogLinearFeedback(eqx.Module):
    """
    Updates a target variable based on population density using a log-linear rule.
    
    Physics:
        ln(Target) = Intercept + Elasticity * ln(Density)
        Density = (Population_Share * Total_Population) / Area_Size
    
    Attributes:
        target_data_key (str): Key in model.data to update (e.g., "rent").
        result_metric_key (str): Key in solver result for population share metric (e.g., "pop_share").
        elasticity_param_key (str): Key in params for elasticity coefficient (e.g., "rent_elasticity").
        intercept_param_key (str): Key in params for intercept term (e.g., "rent_intercepts").
        area_data_key (str): Key in model.data for area sizes.
        total_pop_data_key (str): Key in model.data for total population.
    """
    target_data_key: str
    result_metric_key: str
    elasticity_param_key: str
    intercept_param_key: str
    area_data_key: str
    total_pop_data_key: str

    # Configuration
    clip_min: float = LOG_CLIP_MIN
    clip_max: float = LOG_CLIP_MAX
    epsilon: float = NUMERICAL_EPSILON

    def __init__(
        self,
        target_data_key: str, 
        result_metric_key: str,      
        elasticity_param_key: str,    
        intercept_param_key: str,     
        area_data_key: str = "area_size",
        total_pop_data_key: str = "total_pop",
        clip_min: float = LOG_CLIP_MIN,
        clip_max: float = LOG_CLIP_MAX,
        epsilon: float = NUMERICAL_EPSILON
    ) -> None:
        """
        Args:
            target_data_key (str): Key in model.data to update with new values (e.g., "rent").
            result_metric_key (str): Key in solver result for population share metric (e.g., "pop_share").
            elasticity_param_key (str): Key in params for elasticity coefficient (e.g., "rent_elasticity").
            intercept_param_key (str): Key in params for intercept term (e.g., "rent_intercepts").
            area_data_key (str): Key in model.data for area sizes. Default "area_size".
            total_pop_data_key (str): Key in model.data for total population. Default "total_pop".
            clip_min (float): Minimum clipping value for log predictions. Default LOG_CLIP_MIN.
            clip_max (float): Maximum clipping value for log predictions. Default LOG_CLIP_MAX.
            epsilon (float): Small constant for numerical stability. Default NUMERICAL_EPSILON.
        """
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
        elasticity: PyTree = get_from_pytree(params, self.elasticity_param_key)
        intercept: PyTree = get_from_pytree(params, self.intercept_param_key)

        # 2. Retrieve Data from Model and Result
        model_data: PyTree = model.data
        
        # Result Metric (Share): (num_states,)
        pop_share: Float[Array, "num_states"] = get_from_pytree(
            current_result, self.result_metric_key
        )
        
        # Constants
        area_size = get_from_pytree(model_data, self.area_data_key)
        total_pop = get_from_pytree(model_data, self.total_pop_data_key, default=1.0)

        # 3. Calculate Density
        # Density = (Share * Total) / Area
        abs_population = pop_share * total_pop
        density = abs_population / jnp.maximum(area_size, self.epsilon)
        
        # 4. Compute Log-Linear Update
        # Formula: ln(Y) = alpha + beta * ln(Density)
        # Using maximum(..., epsilon) for numerical stability
        ln_density: Array = jnp.log(jnp.maximum(density, self.epsilon))
        pred_ln_val: Array = intercept + elasticity * ln_density
        pred_ln_val_safe: Array = jnp.clip(pred_ln_val, self.clip_min, self.clip_max)
        new_val: Array = jnp.exp(pred_ln_val_safe)

        # 5. Return New Model with Updated Data
        # StructuralModel protocol guarantees `replace_data` method.
        return model.replace_data(self.target_data_key, new_val)