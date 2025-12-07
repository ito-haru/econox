"""
Tests for Estimation Methods and Estimator mechanics.
Focuses on:
1. Analytical vs Numerical equivalence (LinearMethod).
2. CompositeMethod weighting logic.
3. Parameter constraints (Fixed parameters).
"""

import jax
import jax.numpy as jnp
import pytest

from econox import (
    Model,
    ParameterSpace,
    Estimator,
    LeastSquares,
    GaussianMomentMatch,
    CompositeMethod,
)

# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def linear_data():
    """Generates simple linear data y = 2x + 1 + noise."""
    key = jax.random.PRNGKey(42)
    N = 100
    x = jax.random.normal(key, (N, 1))
    true_beta = jnp.array([2.0])
    true_intercept = jnp.array(1.0)
    
    # y = X*beta + intercept + noise
    noise = 0.1 * jax.random.normal(key, (N,))
    y = (x @ true_beta).ravel() + true_intercept + noise
    
    return {
        "x": x,
        "y": y,
        "N": N
    }

@pytest.fixture
def simple_model(linear_data):
    """Creates a dummy model container."""
    return Model.from_data(
        num_states=linear_data["N"],
        num_actions=1,
        data={"x": linear_data["x"], "y": linear_data["y"]}
    )

# =============================================================================
# Tests
# =============================================================================

def test_ols_numerical_equivalence(simple_model, linear_data):
    """
    Test that Analytical OLS and Numerical OLS (via optimizer) yield the same results.
    This verifies that `LinearMethod.compute_loss` is consistent with `LinearMethod.solve`.
    """
    # Setup
    initial_params = {"beta_0": jnp.array(0.0), "intercept": jnp.array(0.0)}
    param_space = ParameterSpace.create(initial_params)
    
    # Method: OLS
    # Note: LeastSquares defaults to feature_key="X", we override to "x"
    ols_method = LeastSquares(feature_key="x", target_key="y")
    
    # 1. Analytical Solution (solve)
    estimator_analytical = Estimator(
        model=simple_model,
        param_space=param_space,
        method=ols_method
    )
    res_analytical = estimator_analytical.fit(linear_data, sample_size=linear_data["N"])
    
    # 2. Numerical Solution (compute_loss + optimizer)
    # force_numerical=True prevents calling solve()
    estimator_numerical = Estimator(
        model=simple_model,
        param_space=param_space,
        method=ols_method
    )
    res_numerical = estimator_numerical.fit(
        linear_data, 
        sample_size=linear_data["N"], 
        force_numerical=True
    )
    
    # Verification
    print("Analytical:", res_analytical.params)
    print("Numerical: ", res_numerical.params)
    
    # Check parameters match (allow small tolerance for float errors)
    for k in res_analytical.params:
        assert jnp.allclose(
            res_analytical.params[k], 
            res_numerical.params[k], 
            atol=1e-3
        ), f"Parameter {k} mismatch between Analytical and Numerical OLS."


def test_composite_weights(simple_model, linear_data):
    """
    Test that CompositeMethod weights correctly influence the estimation.
    We set up two conflicting objectives and see if weights shift the result.
    """
    # Objective A: Match y (True target)
    method_a = GaussianMomentMatch(
        model_key="x", # Dummy mapping: predict 'x' from model data
        obs_key="y",   # Match against 'y'
        scale_param_key="sigma"
    )
    # Ideally we use the OLS logic, but let's use GMM for simplicity.
    # Actually, let's use two Gaussian targets.
    
    # Setup: We want to estimate a parameter 'mu'.
    # Data 1 suggests mu = 0
    # Data 2 suggests mu = 10
    
    dummy_model = Model.from_data(
        num_states=10, num_actions=1, 
        data={"zeros": jnp.zeros(10), "tens": jnp.ones(10) * 10}
    )
    
    # Params: mu is the value we estimate. 
    # We use a dummy model_key that we replace via 'replace_data' in a real setting,
    # but here let's just optimize a parameter directly against data.
    # To do this with current Estimator, we need the model to generate the prediction.
    # For this test, let's trust the logic:
    # Loss = w1 * L1 + w2 * L2.
    
    # Let's stick to the Linear Regression case.
    # Case 1: Standard OLS on y
    method_1 = LeastSquares(feature_key="x", target_key="y") # True relation
    
    # Case 2: OLS on a garbage target (zeros)
    # This will pull parameters towards 0
    method_2 = LeastSquares(feature_key="x", target_key="zeros_target")
    
    data_with_noise = {
        "x": linear_data["x"], 
        "y": linear_data["y"],
        "zeros_target": jnp.zeros_like(linear_data["y"])
    }
    model = Model.from_data(100, 1, data={"x": linear_data["x"]})
    
    param_space = ParameterSpace.create({"beta_0": 0.5, "intercept": 0.5})

    # Run 1: Weight [1.0, 0.0] -> Should act like pure Method 1 (Valid OLS)
    comp_method_1 = CompositeMethod(methods=[method_1, method_2], weights=[1.0, 0.0])
    res_1 = Estimator(model, param_space, comp_method_1).fit(
        data_with_noise, sample_size=100, force_numerical=True
    )
    
    # Run 2: Weight [0.5, 0.5] -> Should be a mix (parameters shrink towards 0)
    comp_method_mix = CompositeMethod(methods=[method_1, method_2], weights=[0.5, 0.5])
    res_mix = Estimator(model, param_space, comp_method_mix).fit(
        data_with_noise, sample_size=100, force_numerical=True
    )
    
    # Verification
    # Result 1 should be close to true params (beta ~ 2)
    # Result Mix should be smaller (pulled by method 2 which wants y=0 -> beta=0)
    
    beta_1 = res_1.params["beta_0"]
    beta_mix = res_mix.params["beta_0"]
    
    print(f"Beta (Pure): {beta_1}, Beta (Mix): {beta_mix}")
    
    assert beta_1 > 1.8  # Close to 2.0
    assert beta_mix < beta_1  # Should be pulled down
    assert beta_mix > 0.0     # But not zero


def test_fixed_parameter(simple_model, linear_data):
    """
    Verify that parameters marked as 'fixed' do not change during estimation.
    """
    # Fix 'intercept' to 10.0 (True is 1.0, so this is wrong but should stay fixed)
    initial_params = {"beta_0": 0.0, "intercept": 10.0}
    param_space = ParameterSpace.create(
        initial_params=initial_params,
        constraints={"intercept": "fixed"}
    )
    
    method = LeastSquares(feature_key="x", target_key="y")
    
    estimator = Estimator(
        model=simple_model,
        param_space=param_space,
        method=method
    )
    
    # Force numerical to ensure the optimizer respects the mask 
    # (Analytical solve might handle it differently depending on implementation, 
    # but Estimator.fit logic handles fixing for numerical optimization mainly).
    # *Note*: Current analytical implementation in `LeastSquares.solve` solves for ALL params.
    # So 'fixed' constraints are primarily for Numerical Optimization in this library version.
    
    result = estimator.fit(
        linear_data, 
        sample_size=linear_data["N"],
        force_numerical=True
    )
    
    # Check intercept
    assert result.params["intercept"] == 10.0, "Fixed parameter 'intercept' changed!"
    
    # Check beta (should optimize to compensate, though model will be bad)
    assert result.params["beta_0"] != 0.0, "Free parameter 'beta_0' did not update."