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
    TwoStageLeastSquares
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

# =============================================================================
# 2SLS Tests
# =============================================================================

@pytest.fixture
def iv_data():
    """
    Generates data with endogeneity to test 2SLS.
    Structure:
        Z (Instrument) -> X (Endogenous) -> Y (Outcome)
        U (Unobserved) affects both X and Y (Confounder)
    """
    key = jax.random.PRNGKey(999)
    N = 1000  # Need larger sample for IV convergence
    
    # 1. Instrument Z (Exogenous)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    z = jax.random.normal(k1, (N, 1))
    
    # 2. Unobserved Confounder U
    u = jax.random.normal(k2, (N, 1))
    
    # 3. Endogenous Variable X
    # X depends on Z (relevance) and U (endogeneity)
    # True First Stage: X = 0.8*Z + 0.5*U + noise
    x = 0.8 * z + 0.5 * u + 0.1 * jax.random.normal(k3, (N, 1))
    
    # 4. Outcome Y
    # Y depends on X and U (bias source)
    # True Structural Equation: Y = 2.0*X + 1.0 + (U + noise)
    true_beta = 2.0
    true_intercept = 1.0
    
    # Note: Because X contains 0.5*U, and Y contains 1.0*U, 
    # OLS will overestimate beta (Positive Bias).
    y = (true_beta * x).ravel() + true_intercept + (u + 0.1 * jax.random.normal(k4, (N, 1))).ravel()
    
    return {
        "x": x,
        "y": y,
        "z": z,
        "N": N,
        "true_beta": true_beta,
        "true_intercept": true_intercept
    }

def test_2sls_recovery(iv_data):
    """
    Verifies that 2SLS correctly recovers parameters in the presence of endogeneity,
    whereas OLS fails (is biased).
    """
    # Setup Data Container
    data = {
        "y": iv_data["y"],
        "X": iv_data["x"], # Endogenous
        "Z": iv_data["z"]  # Instrument
    }
    
    model = Model.from_data(
        num_states=iv_data["N"],
        num_actions=1,
        data=data
    )
    
    # Initial Params (same for both)
    # Note: 2SLS class in analytical.py likely parses "beta_0", "intercept" etc.
    initial_params = {"intercept": 0.0, "beta_0": 0.0}
    param_space = ParameterSpace.create(initial_params)
    
    # ---------------------------------------------------------
    # 1. Run OLS (Expected to be Biased)
    # ---------------------------------------------------------
    ols_method = LeastSquares(feature_key="X", target_key="y")
    estimator_ols = Estimator(model, param_space, ols_method)
    
    # Use analytical solve
    res_ols = estimator_ols.fit(data, sample_size=iv_data["N"])
    beta_ols = res_ols.params["beta_0"]
    
    # ---------------------------------------------------------
    # 2. Run 2SLS (Expected to be Consistent)
    # ---------------------------------------------------------
    tsls_method = TwoStageLeastSquares(
        target_key="y",
        endog_key="X",
        instrument_key="Z"
    )
    estimator_tsls = Estimator(model, param_space, tsls_method)
    
    # Use analytical solve
    res_tsls = estimator_tsls.fit(data, sample_size=iv_data["N"])
    beta_tsls = res_tsls.params["beta_0"]
    intercept_tsls = res_tsls.params["intercept"]
    
    # ---------------------------------------------------------
    # Verification
    # ---------------------------------------------------------
    print(f"\nTrue Beta: {iv_data['true_beta']}")
    print(f"OLS Beta : {beta_ols:.4f} (Should be biased upward)")
    print(f"2SLS Beta: {beta_tsls:.4f} (Should be close to True)")
    
    # 1. 2SLS should be accurate (allow some sampling noise)
    assert jnp.abs(beta_tsls - iv_data["true_beta"]) < 0.15, \
        f"2SLS failed to recover beta. Got {beta_tsls}, expected {iv_data['true_beta']}"
        
    assert jnp.abs(intercept_tsls - iv_data["true_intercept"]) < 0.15, \
        f"2SLS failed to recover intercept. Got {intercept_tsls}"

    # 2. OLS should be significantly biased (Bias check)
    # In this simulation, bias should be roughly cov(X,U)/var(X) > 0
    assert beta_ols > iv_data["true_beta"] + 0.1, \
        "OLS should be biased in this endogeneity setup, but it wasn't."

def test_2sls_numerical_equivalence(iv_data):
    """
    Verify that 2SLS analytical solution matches numerical optimization
    (Assuming the objective function is defined consistently).
    
    Note: Standard 2SLS is usually closed-form. If EstimationMethod.compute_loss 
    is implemented for 2SLS (e.g. via IV-GMM objective), this test verifies consistency.
    """
    data = {
        "y": iv_data["y"],
        "X": iv_data["x"],
        "Z": iv_data["z"]
    }
    model = Model.from_data(iv_data["N"], 1, data)
    
    initial_params = {"intercept": 0.0, "beta_0": 0.0}
    param_space = ParameterSpace.create(initial_params)
    
    tsls_method = TwoStageLeastSquares(
        target_key="y",
        endog_key="X",
        instrument_key="Z"
    )
    
    # 1. Analytical
    res_analytical = Estimator(model, param_space, tsls_method).fit(
        data, sample_size=iv_data["N"]
    )
    
    # 2. Numerical
    # This requires TwoStageLeastSquares.compute_loss to be implemented correctly
    res_numerical = Estimator(model, param_space, tsls_method).fit(
        data, sample_size=iv_data["N"], force_numerical=True
    )
    
    print("\n2SLS Analytical:", res_analytical.params)
    print("2SLS Numerical: ", res_numerical.params)
    
    for k in res_analytical.params:
        assert jnp.allclose(
            res_analytical.params[k], 
            res_numerical.params[k], 
            atol=1e-2
        ), f"Parameter {k} mismatch between Analytical and Numerical 2SLS."