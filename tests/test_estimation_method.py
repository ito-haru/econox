"""
Tests for Estimation Methods and Estimator mechanics.
Focuses on:
1. Analytical vs Numerical equivalence (LinearMethod).
2. CompositeMethod weighting logic.
3. Parameter constraints (Fixed parameters).
4. Two-Stage Least Squares (2SLS) for endogeneity.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from econox import (
    Model,
    ParameterSpace,
    Estimator,
    LeastSquares,
    CompositeMethod,
    TwoStageLeastSquares,
    MaximumLikelihood,
    ValueIterationSolver,
    LinearUtility,
    GumbelDistribution,
)
from econox.methods.variance import Hessian

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
    
    # y = x*beta + intercept + noise
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
    # Setup: Parameters are beta_0 (slope) and intercept
    initial_params = {"beta_0": jnp.array(0.0), "intercept": jnp.array(0.0)}
    param_space = ParameterSpace.create(initial_params)
    
    # Method: OLS with feature_key="x", target_key="y"
    ols_method = LeastSquares(feature_key="x", target_key="y")
    
    # 1. Analytical Solution (solve)
    estimator_analytical = Estimator(
        model=simple_model,
        param_space=param_space,
        method=ols_method
    )
    res_analytical = estimator_analytical.fit(linear_data, sample_size=linear_data["N"])
    
    # 2. Numerical Solution (compute_loss + optimizer)
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
    
    # Verification: Check parameters match
    print("Analytical:", res_analytical.params)
    print("Numerical: ", res_numerical.params)
    
    for k in res_analytical.params:
        assert jnp.allclose(
            res_analytical.params[k], 
            res_numerical.params[k], 
            atol=1e-3
        ), f"Parameter {k} mismatch between Analytical and Numerical OLS."


def test_composite_weights(simple_model, linear_data):
    """
    Test that CompositeMethod weights correctly influence the estimation.
    Method 1: OLS on true target (y)
    Method 2: OLS on zeros (pulls parameters toward zero)
    """
    # Case 1: Standard OLS on y (true relation: y = 2x + 1)
    method_1 = LeastSquares(feature_key="x", target_key="y")
    
    # Case 2: OLS on zeros target (pulls beta toward 0)
    method_2 = LeastSquares(feature_key="x", target_key="zeros_target")
    
    data_with_noise = {
        "x": linear_data["x"], 
        "y": linear_data["y"],
        "zeros_target": jnp.zeros_like(linear_data["y"])
    }
    model = Model.from_data(100, 1, data={"x": linear_data["x"]})
    
    param_space = ParameterSpace.create({"beta_0": 0.5, "intercept": 0.5})

    # Run 1: Weight [1.0, 0.0] -> Pure Method 1 (valid OLS, beta should be ~2.0)
    comp_method_1 = CompositeMethod(methods=[method_1, method_2], weights=[1.0, 0.0])
    res_1 = Estimator(model, param_space, comp_method_1).fit(
        data_with_noise, sample_size=100, force_numerical=True
    )
    
    # Run 2: Weight [0.5, 0.5] -> Mixed (beta should be pulled toward 0)
    comp_method_mix = CompositeMethod(methods=[method_1, method_2], weights=[0.5, 0.5])
    res_mix = Estimator(model, param_space, comp_method_mix).fit(
        data_with_noise, sample_size=100, force_numerical=True
    )
    
    # Verification
    beta_1 = res_1.params["beta_0"]
    beta_mix = res_mix.params["beta_0"]
    
    print(f"Beta (Pure): {beta_1}, Beta (Mix): {beta_mix}")
    
    assert beta_1 > 1.8  # Close to true value 2.0
    assert beta_mix < beta_1  # Mixed should be pulled down
    assert beta_mix > 0.0     # But not completely zero


def test_fixed_parameter(simple_model, linear_data):
    """
    Verify that parameters marked as 'fixed' do not change during estimation.
    We fix intercept=10.0 (incorrect value) and check it stays fixed.
    """
    # Fix 'intercept' to 10.0 (true value is 1.0, but should stay at 10.0)
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
    
    result = estimator.fit(
        linear_data, 
        sample_size=linear_data["N"],
        force_numerical=True
    )
    
    # Verification
    assert result.params["intercept"] == 10.0, "Fixed parameter 'intercept' changed!"
    assert result.params["beta_0"] != 0.0, "Free parameter 'beta_0' did not update."


def test_fixed_parameter_with_variance_dynamic_model():
    """
    Test variance calculation (Hessian) with fixed parameters in dynamic discrete choice model.
    Uses MaximumLikelihood to ensure Hessian-based variance is computed.
    Verifies that:
    1. Fixed parameters have zero standard errors
    2. Free parameters have non-zero standard errors
    3. No shape mismatch errors occur during variance computation
    """
    # Simple 2-state, 2-action model
    num_states = 2
    num_actions = 2
    
    features = jnp.array([
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],  # State 0
        [[1.0, 1.0, 0.0], [0.0, 0.5, 0.0]],  # State 1
    ])
    
    transitions = jnp.array([
        [0.7, 0.3],  # State 0, Action 0
        [0.4, 0.6],  # State 0, Action 1
        [0.6, 0.4],  # State 1, Action 0
        [0.3, 0.7],  # State 1, Action 1
    ])
    
    model = Model.from_data(
        num_states=num_states,
        num_actions=num_actions,
        data={"features": features},
        transitions=transitions,
        num_periods=np.inf
    )
    
    # True parameters
    true_params = {"beta_0": 1.0, "beta_1": 0.5, "beta_2": 0.1}
    
    # Generate synthetic data
    utility = LinearUtility(param_keys=("beta_0", "beta_1", "beta_2"), feature_key="features")
    dist = GumbelDistribution()
    solver = ValueIterationSolver(
        utility=utility,
        dist=dist,
        discount_factor=0.9
    )
    
    result_true = solver.solve(true_params, model)
    choice_probs = result_true.profile
    
    # Generate observations
    np.random.seed(42)
    n_obs = 5000
    states = np.random.randint(0, num_states, size=n_obs)
    choices = np.array([
        np.random.choice(num_actions, p=np.array(choice_probs[s]))
        for s in states
    ])
    
    observations = {
        "state_indices": jnp.array(states),
        "choice_indices": jnp.array(choices)
    }
    
    # Estimation with fixed beta_2
    initial_params = {"beta_0": 0.5, "beta_1": 0.3, "beta_2": 0.1}
    param_space = ParameterSpace.create(
        initial_params=initial_params,
        constraints={"beta_2": "fixed"}
    )
    
    method = MaximumLikelihood(variance=Hessian())
    
    estimator = Estimator(
        model=model,
        param_space=param_space,
        solver=ValueIterationSolver(
            utility=LinearUtility(param_keys=("beta_0", "beta_1", "beta_2"), feature_key="features"),
            dist=GumbelDistribution(),
            discount_factor=0.9
        ),
        method=method
    )
    
    result = estimator.fit(observations, sample_size=n_obs)
    
    # Verify estimation success
    assert result.success, "Estimation failed"
    assert result.std_errors is not None, "Standard errors not computed"
    assert result.vcov is not None, "Variance-covariance matrix not computed"
    
    # Verify fixed parameter constraint
    assert result.params["beta_2"] == 0.1, "Fixed parameter changed"
    
    # Verify standard errors structure
    assert "beta_0" in result.std_errors, "Missing std error for beta_0"
    assert "beta_1" in result.std_errors, "Missing std error for beta_1"
    assert "beta_2" in result.std_errors, "Missing std error for beta_2"
    
    # Verify free parameters have positive std errors
    assert result.std_errors["beta_0"] > 0.0, "Free parameter should have positive std error"
    assert result.std_errors["beta_1"] > 0.0, "Free parameter should have positive std error"
    
    # Verify fixed parameter has zero or near-zero std error
    assert result.std_errors["beta_2"] == 0, "Fixed parameter should have zero std error"

# =============================================================================
# 2SLS Tests
# =============================================================================

@pytest.fixture
def iv_data():
    """
    Generates data with endogeneity to test 2SLS.
    Structure:
        Z (Instrument) -> X (Endogenous) -> Y (Outcome)
        U (Unobserved) affects both X and Y (creates endogeneity)
    
    True model: Y = 2.0*X + 1.0 + error
    OLS will be biased due to correlation between X and U.
    """
    key = jax.random.PRNGKey(999)
    N = 1000
    
    # 1. Instrument Z (exogenous)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    z = jax.random.normal(k1, (N, 1))
    
    # 2. Unobserved Confounder U (creates endogeneity)
    u = jax.random.normal(k2, (N, 1))
    
    # 3. Endogenous Variable X
    # First Stage: X = 0.8*Z + 0.5*U + noise
    x = 0.8 * z + 0.5 * u + 0.1 * jax.random.normal(k3, (N, 1))
    
    # 4. Outcome Y
    # Structural Equation: Y = 2.0*X + 1.0 + U + noise
    true_beta = 2.0
    true_intercept = 1.0
    
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
    whereas OLS is biased upward.
    """
    # Setup Data
    data = {
        "y": iv_data["y"],
        "X": iv_data["x"], # Endogenous regressor
        "Z": iv_data["z"]  # Exogenous instrument
    }
    
    model = Model.from_data(
        num_states=iv_data["N"],
        num_actions=1,
        data=data
    )
    
    # Initial Parameters
    initial_params = {"intercept": 0.0, "beta_0": 0.0}
    param_space = ParameterSpace.create(initial_params)
    
    # 1. Run OLS (expected to be biased upward)
    ols_method = LeastSquares(feature_key="X", target_key="y")
    estimator_ols = Estimator(model, param_space, ols_method)
    
    res_ols = estimator_ols.fit(data, sample_size=iv_data["N"])
    beta_ols = res_ols.params["beta_0"]
    
    # 2. Run 2SLS (expected to be consistent)
    tsls_method = TwoStageLeastSquares(
        target_key="y",
        endog_key="X",
        instrument_key="Z"
    )
    estimator_tsls = Estimator(model, param_space, tsls_method)
    
    res_tsls = estimator_tsls.fit(data, sample_size=iv_data["N"])
    beta_tsls = res_tsls.params["beta_0"]
    intercept_tsls = res_tsls.params["intercept"]
    
    # Verification
    print(f"\nTrue Beta: {iv_data['true_beta']}")
    print(f"OLS Beta : {beta_ols:.4f} (Should be biased upward)")
    print(f"2SLS Beta: {beta_tsls:.4f} (Should be close to {iv_data['true_beta']})")
    
    # 2SLS should be accurate (allowing for sampling variation)
    assert jnp.abs(beta_tsls - iv_data["true_beta"]) < 0.15, \
        f"2SLS failed to recover beta. Got {beta_tsls}, expected {iv_data['true_beta']}"
        
    assert jnp.abs(intercept_tsls - iv_data["true_intercept"]) < 0.15, \
        f"2SLS failed to recover intercept. Got {intercept_tsls}, expected {iv_data['true_intercept']}"

    # OLS should be significantly biased upward
    assert beta_ols > iv_data["true_beta"] + 0.1, \
        "OLS should be biased upward due to endogeneity, but it wasn't."

def test_2sls_numerical_equivalence(iv_data):
    """
    Verify that 2SLS analytical solution matches numerical optimization.
    This checks consistency between TwoStageLeastSquares.solve() and compute_loss().
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
    
    # 1. Analytical solution
    res_analytical = Estimator(model, param_space, tsls_method).fit(
        data, sample_size=iv_data["N"]
    )
    
    # 2. Numerical solution (via optimizer)
    res_numerical = Estimator(model, param_space, tsls_method).fit(
        data, sample_size=iv_data["N"], force_numerical=True
    )
    
    print("\n2SLS Analytical:", res_analytical.params)
    print("2SLS Numerical: ", res_numerical.params)
    
    # Verification: Parameters should match closely
    for k in res_analytical.params:
        assert jnp.allclose(
            res_analytical.params[k], 
            res_numerical.params[k], 
            atol=1e-2
        ), f"Parameter {k} mismatch between Analytical and Numerical 2SLS."