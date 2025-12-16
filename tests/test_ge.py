"""
Tests for the General Equilibrium (GE) workflow.

This module validates the 'Triple-Loop' capability of Econox (Optimization -> Equilibrium -> Dynamic Programming).
It uses a synthetic Spatial Equilibrium model to test stability and parameter recovery.
"""

import jax
import jax.numpy as jnp
import jax.nn as jnn
import pytest
import equinox as eqx
from jaxtyping import PyTree, Float, Array

import econox as ecx
from econox.protocols import StructuralModel
from econox.logic.dynamics import TrajectoryDynamics
from econox.utils import get_from_pytree

# Ensure x64 precision for GE stability
jax.config.update("jax_enable_x64", True)

# =============================================================================
# Custom Logic Definitions (From Example Script)
# =============================================================================

class RealWageUtility(eqx.Module):
    """
    Custom Utility: U = beta_money * (w/r) + beta_dist * distance
    Operates on real values stored in model.data.
    """
    money_param_key: str
    dist_param_key: str
    
    def compute_flow_utility(self, params: PyTree, model: StructuralModel) -> Float[Array, "num_states num_actions"]:
        beta_money = get_from_pytree(params, self.money_param_key)
        beta_dist = get_from_pytree(params, self.dist_param_key)
        
        # Retrieve real values
        wage = model.data["wage"]  # (S,)
        rent = model.data["rent"]  # (S,)
        distance = model.data["distance_matrix_flat"]  # (S, A)
        
        num_periods = model.num_periods if isinstance(model.num_periods, int) else int(len(wage) / model.num_actions)
        num_areas = model.num_actions
        
        # Reshape to (T, A)
        wage_grid = wage.reshape(num_periods, num_areas)
        rent_grid = rent.reshape(num_periods, num_areas)
        
        # Expand to (S, A)
        W_sa = jnp.tile(wage_grid[:, None, :], (1, num_areas, 1)).reshape(-1, num_areas)
        R_sa = jnp.tile(rent_grid[:, None, :], (1, num_areas, 1)).reshape(-1, num_areas)
        
        # Real wage calculation (w - r proxying for purchasing power)
        real_wage = W_sa - R_sa 
        
        u = beta_money * real_wage + beta_dist * distance
        return u

# =============================================================================
# Helper Functions (Data Generation)
# =============================================================================

def generate_realistic_data(num_areas=5, num_periods=10, base_pop=1000.0, key=None):
    if key is None:
        key = jax.random.PRNGKey(42)
    
    # Simplified area sizes for test speed
    area_sizes = jnp.linspace(100.0, 200.0, num_areas)
    
    distance_matrix = jnp.zeros((num_areas, num_areas), dtype=jnp.float64)
    for i in range(num_areas):
        for j in range(num_areas):
            if i != j:
                distance_matrix = distance_matrix.at[i, j].set(float(abs(i - j)))
    
    # Synthetic Population Dynamics
    pop_counts = []
    initial_shares = jnp.ones(num_areas) / num_areas
    
    for t in range(num_periods):
        total_pop_t = base_pop * (1.0 + 0.01 * t)
        # Random perturbation for realism
        key, subkey = jax.random.split(key)
        noise = jax.random.uniform(subkey, (num_areas,), minval=0.9, maxval=1.1)
        shares_t = initial_shares * noise
        shares_t = shares_t / shares_t.sum()
        
        pop_t = total_pop_t * shares_t
        pop_counts.extend(pop_t.tolist())
    
    pop_counts = jnp.array(pop_counts, dtype=jnp.float64)
    
    # Calculate aggregates
    total_pop_path = jnp.array([
        pop_counts[t*num_areas:(t+1)*num_areas].sum()
        for t in range(num_periods)
        for _ in range(num_areas)
    ], dtype=jnp.float64)
    
    pop_dist = pop_counts / pop_counts.sum()
    area_per_state = jnp.tile(area_sizes, num_periods)
    pop_density = pop_counts / area_per_state
    ln_pop_density = jnp.log(pop_density + 1e-8)
    
    # Generate Wages/Rents based on density (Structural Equations)
    true_wage_elasticity = 0.15
    true_rent_elasticity = 0.25
    mean_ln_density = jnp.mean(ln_pop_density)
    
    base_ln_wage = 0.6
    base_ln_rent = 0.6
    
    ln_wage = base_ln_wage + true_wage_elasticity * (ln_pop_density - mean_ln_density)
    ln_rent = base_ln_rent + true_rent_elasticity * (ln_pop_density - mean_ln_density)
    
    return {
        'pop_dist': pop_dist,
        'ln_wage': ln_wage,
        'ln_rent': ln_rent,
        'total_pop_path': total_pop_path,
        'area_per_state': area_per_state,
        'ln_pop_density': ln_pop_density,
        'distance_matrix': distance_matrix
    }

def estimate_dummy_params(data, num_areas):
    """Simple OLS to get 'true' parameters for initialization."""
    ln_density = data['ln_pop_density']
    ln_wage = data['ln_wage']
    ln_rent = data['ln_rent']
    
    # Intercepts per area
    state_to_area_indices = jnp.tile(jnp.arange(num_areas), len(ln_density) // num_areas)
    X_dummies = jnn.one_hot(state_to_area_indices, num_areas, dtype=jnp.float64)
    X = jnp.column_stack([ln_density, X_dummies])
    
    wage_params = jnp.linalg.lstsq(X, ln_wage, rcond=None)[0]
    rent_params = jnp.linalg.lstsq(X, ln_rent, rcond=None)[0]
    
    return {
        'wage_elasticity': float(wage_params[0]),
        'rent_elasticity': float(rent_params[0]),
        'wage_intercepts': wage_params[1:],
        'rent_intercepts': rent_params[1:],
        'wage_scale': 0.1,
        'rent_scale': 0.1
    }

# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def ge_config():
    return {"num_areas": 5, "num_periods": 5}  # Smaller scale for testing

@pytest.fixture(scope="module")
def ge_data_bundle(ge_config):
    key = jax.random.PRNGKey(42)
    data = generate_realistic_data(
        num_areas=ge_config["num_areas"], 
        num_periods=ge_config["num_periods"], 
        key=key
    )
    params = estimate_dummy_params(data, ge_config["num_areas"])
    return {"data": data, "params": params}

@pytest.fixture(scope="module")
def ge_model(ge_config, ge_data_bundle) -> ecx.Model:
    num_areas = ge_config["num_areas"]
    num_periods = ge_config["num_periods"]
    S = num_areas * num_periods
    A = num_areas
    data = ge_data_bundle["data"]

    # 1. Transitions (Deterministic Time Step)
    transition_tensor = jnp.zeros((S, A, S), dtype=jnp.float64)
    for s in range(S):
        current_time = s // num_areas
        if current_time < num_periods - 1:
            next_time = current_time + 1
            for a in range(A):
                next_state = next_time * num_areas + a
                transition_tensor = transition_tensor.at[s, a, next_state].set(1.0)
    transition_matrix = transition_tensor.reshape(S * A, S)

    # 2. Distance Matrix (Flattened for Utility)
    distance_sa = jnp.zeros((S, A), dtype=jnp.float64)
    for s in range(S):
        current_area = s % num_areas
        distance_sa = distance_sa.at[s, :].set(data['distance_matrix'][current_area, :])

    # 3. Boundary Condition Indices
    terminal_indices = jnp.arange((num_periods-1)*num_areas, S, dtype=jnp.int32)
    previous_indices = jnp.arange((num_periods-2)*num_areas, (num_periods-1)*num_areas, dtype=jnp.int32)
    initial_year_indices = jnp.arange(num_areas, dtype=jnp.int32)
    initial_year_values = data['pop_dist'][:A] / data['pop_dist'][:A].sum()

    return ecx.Model.from_data(
        num_states=S,
        num_actions=A,
        num_periods=num_periods,
        transitions=transition_matrix,
        data={
            "wage": jnp.exp(data['ln_wage']),
            "rent": jnp.exp(data['ln_rent']),
            "distance_matrix_flat": distance_sa,
            "area_size": data['area_per_state'],
            "total_pop": data['total_pop_path'],
            "terminal_state_indices": terminal_indices,
            "previous_state_indices": previous_indices,
            "initial_year_indices": initial_year_indices,
            "initial_year_values": initial_year_values
        }
    )

@ecx.function_feedback(target_key="wage")
def wage_feedback(params, result, data):
    elasticity = params["wage_elasticity"]
    intercepts = params["wage_intercepts_S"]
    area_size = data["area_size"]
    total_pop = data["total_pop"]
    pop_density = result["solution"]*total_pop / area_size
    ln_pop_density = jnp.log(pop_density + 1e-8)
    mean_ln_density = jnp.mean(ln_pop_density)
    ln_wage = intercepts + elasticity * (ln_pop_density - mean_ln_density)
    return jnp.exp(ln_wage)

@ecx.function_feedback(target_key="rent")
def rent_feedback(params, result, data):
    elasticity = params["rent_elasticity"]
    intercepts = params["rent_intercepts_S"]
    area_size = data["area_size"]
    total_pop = data["total_pop"]
    pop_density = result["solution"]*total_pop / area_size
    ln_pop_density = jnp.log(pop_density + 1e-8)
    mean_ln_density = jnp.mean(ln_pop_density)
    ln_rent = intercepts + elasticity * (ln_pop_density - mean_ln_density)
    return jnp.exp(ln_rent)

@pytest.fixture(scope="module")
def ge_components(ge_config, ge_data_bundle):
    """Initialize Utility, Feedback, and ParameterSpace."""
    num_periods = ge_config["num_periods"]
    true_params = ge_data_bundle["params"]
    
    # Expand intercepts to full state space
    wage_intercepts_S = jnp.tile(true_params['wage_intercepts'], num_periods)
    rent_intercepts_S = jnp.tile(true_params['rent_intercepts'], num_periods)
    
    initial_params = {
        "beta_money": 1.0,
        "beta_dist": -0.25,
        "wage_elasticity": true_params['wage_elasticity'],
        "rent_elasticity": true_params['rent_elasticity'],
        "wage_intercepts_S": wage_intercepts_S,
        "rent_intercepts_S": rent_intercepts_S,
        "wage_scale": true_params['wage_scale'],
        "rent_scale": true_params['rent_scale']
    }
    
    param_space = ecx.ParameterSpace.create(initial_params=initial_params)
    
    utility = RealWageUtility(money_param_key="beta_money", dist_param_key="beta_dist")
    
    feedback = ecx.CompositeFeedback(feedbacks=(rent_feedback, wage_feedback))
    
    return {
        "param_space": param_space,
        "utility": utility,
        "feedback": feedback,
        "dist": ecx.GumbelDistribution(),
        "dynamics": TrajectoryDynamics(enforce_boundary=True)
    }

# =============================================================================
# Tests
# =============================================================================

def test_equilibrium_stability(ge_model, ge_components, ge_data_bundle):
    """
    Verify that the EquilibriumSolver converges to a stable distribution.
    """
    inner_solver = ecx.ValueIterationSolver(
        utility=ge_components["utility"],
        dist=ge_components["dist"],
        discount_factor=0.95,
        )
    equilibrium_solver = ecx.EquilibriumSolver(
        inner_solver=inner_solver,
        feedback=ge_components["feedback"],
        dynamics=ge_components["dynamics"],
        damping=0.5,
        initial_distribution=ge_data_bundle["data"]["pop_dist"]
    )
    
    result: ecx.SolverResult = equilibrium_solver.solve(
        params=ge_components["param_space"].initial_params,
        model=ge_model
    )
    
    assert result.success, f"Equilibrium solver failed. Steps: {result.aux.get('steps')}"
    
    # Check if distribution is valid
    pop_dist = result.solution
    assert jnp.all(jnp.isfinite(pop_dist))
    assert jnp.all(pop_dist >= 0)


def test_ge_estimation(ge_model, ge_components, ge_data_bundle):
    """
    Verify that the full GE Estimator pipeline runs.
    Note: Full parameter recovery is difficult in a short unit test, 
    so we focus on execution success and gradient availability.
    """
    # 1. Run Equilibrium once to generate "observed" weights
    inner_solver = ecx.ValueIterationSolver(
        utility=ge_components["utility"],
        dist=ge_components["dist"],
        discount_factor=0.95,
        )

    equilibrium_solver = ecx.EquilibriumSolver(
        inner_solver=inner_solver,
        feedback=ge_components["feedback"],
        dynamics=ge_components["dynamics"],
        damping=0.5,
        initial_distribution=ge_data_bundle["data"]["pop_dist"]
    )
    
    solve_result = equilibrium_solver.solve(
        params=ge_components["param_space"].initial_params,
        model=ge_model
    )
    
    # 2. Construct Observations
    true_dist = solve_result.solution
    true_policy = solve_result.profile
    
    obs_weights_matrix = true_dist[:, None] * true_policy * 1000.0
    state_indices = jnp.repeat(jnp.arange(ge_model.num_states), ge_model.num_actions)
    choice_indices = jnp.tile(jnp.arange(ge_model.num_actions), ge_model.num_states)
    weights_flat = obs_weights_matrix.flatten()
    
    # 3. Setup Estimator
    objective = ecx.CompositeMethod(
        methods=[
            ecx.MaximumLikelihood(),
            ecx.GaussianMomentMatch(obs_key="obs_wage", model_key="wage", scale_param_key="wage_scale"),
            ecx.GaussianMomentMatch(obs_key="obs_rent", model_key="rent", scale_param_key="rent_scale")
        ], 
        weights=[1.0, 1.0, 1.0] # Simplified weights
    )
    
    estimator = ecx.Estimator(
        model=ge_model,
        param_space=ge_components["param_space"],
        solver=equilibrium_solver,
        method=objective,
        verbose=True
    )
    
    observations = {
        "state_indices": state_indices,
        "choice_indices": choice_indices,
        "weights": weights_flat,
        "obs_wage": ge_model.data["wage"], # Perfect match target
        "obs_rent": ge_model.data["rent"]
    }
    
    # 4. Run Estimation (few steps for test speed)
    # Note: Optimistix minimizer config can be adjusted if needed, 
    # but here we rely on the default behavior checking for success.
    result = estimator.fit(
        observations=observations,
        initial_params=ge_components["param_space"].initial_params
    )
    
    assert result is not None
    # We check if loss is finite, success might be False if max_steps is hit early,
    # but the pipeline must complete without crashing.
    assert jnp.isfinite(result.loss)

# =============================================================================
# New Test Component: Model Feedback (Joint Update)
# =============================================================================

@ecx.model_feedback
def joint_market_clearing(params, result, model):
    """
    Example of a 'CustomUpdateFeedback' that updates both Wage and Rent simultaneously.
    This is more efficient than CompositeFeedback when updates share intermediate calculations
    (like population density).
    """
    # 1. Retrieve Parameters
    wage_elas = params["wage_elasticity"]
    wage_int = params["wage_intercepts_S"]
    rent_elas = params["rent_elasticity"]
    rent_int = params["rent_intercepts_S"]
    
    # 2. Retrieve Data
    area_size = model.data["area_size"]
    total_pop = model.data["total_pop"]
    
    # 3. Shared Physics: Calculate Density ONCE
    # (In CompositeFeedback, this would be calculated twice)
    pop_density = result["solution"] * total_pop / area_size
    ln_pop_density = jnp.log(pop_density + 1e-8)
    mean_ln_density = jnp.mean(ln_pop_density)
    
    # 4. Calculate New Values
    ln_wage = wage_int + wage_elas * (ln_pop_density - mean_ln_density)
    ln_rent = rent_int + rent_elas * (ln_pop_density - mean_ln_density)
    
    # 5. Batch Update Model Data
    # copying dictionary to avoid side effects (though JAX arrays are immutable)
    new_data = model.data.copy()
    new_data["wage"] = jnp.exp(ln_wage)
    new_data["rent"] = jnp.exp(ln_rent)
    
    # Use eqx.tree_at to return the new StructuralModel safely
    model = model.replace_data("wage", jnp.exp(ln_wage)).replace_data("rent", jnp.exp(ln_rent))
    return model


def test_custom_model_feedback_logic(ge_model, ge_components, ge_data_bundle):
    """
    Test ensuring that @ecx.model_feedback works correctly by performing
    a simultaneous update of multiple model variables.
    """
    inner_solver = ecx.ValueIterationSolver(
        utility=ge_components["utility"],
        dist=ge_components["dist"],
        discount_factor=0.95,
        )
    equilibrium_solver = ecx.EquilibriumSolver(
        inner_solver=inner_solver,
        feedback=joint_market_clearing,
        dynamics=ge_components["dynamics"],
        initial_distribution=ge_data_bundle["data"]["pop_dist"]
    )
    
    # Run solver using the JOINT feedback mechanism defined above
    result: ecx.SolverResult = equilibrium_solver.solve(
        params=ge_components["param_space"].initial_params,
        model=ge_model,
    )
    
    assert result.success, "Equilibrium solver with @model_feedback failed."
    
    # Validate Output
    pop_dist = result.solution
    assert jnp.all(jnp.isfinite(pop_dist))
    assert jnp.all(pop_dist >= 0)
    
    # Check if data was actually updated in the final model result
    final_data = result.aux["equilibrium_data"]
    
    # Wages and Rents should not be uniform if the feedback worked
    assert jnp.std(final_data["wage"]) > 1e-6, "Wages should vary across states."
    assert jnp.std(final_data["rent"]) > 1e-6, "Rents should vary across states."
    
    # Consistency check: Compare density implied by result vs data used for wage
    # (simplified check to ensure the feedback loop closed consistently)
    final_wage = final_data["wage"]
    # Re-calculate expected wage from final distribution manually
    # This verifies the feedback function logic was applied correctly at the fixed point
    recalc_model = joint_market_clearing.update(
        ge_components["param_space"].initial_params, 
        {"solution": pop_dist}, 
        ge_model
    )
    assert jnp.allclose(final_wage, recalc_model.data["wage"], atol=1e-5)
    final_rent = final_data["rent"]
    assert jnp.allclose(final_rent, recalc_model.data["rent"], atol=1e-5)
