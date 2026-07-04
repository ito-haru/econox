"""
Tests for Utility components.
Validates LinearUtility and the functional utility decorator API.
"""

import jax
import jax.numpy as jnp
import pytest
import equinox as eqx
from jaxtyping import PyTree

import econox as ecx

# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_model_data():
    """Creates a minimal dummy model and parameters for testing."""
    S, A, F = 4, 2, 3  # States, Actions, Features
    
    # Random feature matrix (S, A, F)
    key = jax.random.PRNGKey(0)
    features = jax.random.normal(key, (S, A, F))
    
    # Dummy Model Wrapper (Minimal implementation of StructuralModel protocol)
    class DummyModel(eqx.Module):
        data: PyTree
        num_states: int = S
        num_actions: int = A
        num_periods: int = 1
        transitions: None = None
        availability: None = None

        def replace_data(self, key, value):
            new_data = self.data.copy()
            new_data[key] = value
            return eqx.tree_at(lambda m: m.data, self, new_data)

    model = DummyModel(data={"features": features, "other_val": 10.0})
    
    # Parameters
    params = {
        "beta_0": 1.0,
        "beta_1": 2.0,
        "beta_2": -1.0,
        "gamma": 0.5
    }
    
    return model, params, features

# =============================================================================
# Tests
# =============================================================================

def test_linear_utility(mock_model_data):
    """Test standard LinearUtility matrix multiplication."""
    model, params, features = mock_model_data
    
    # 1. Define Utility
    utility = ecx.LinearUtility(
        param_keys=("beta_0", "beta_1", "beta_2"),
        feature_key="features"
    )
    
    # 2. Compute
    u_matrix = utility.compute_flow_utility(params, model)
    
    # 3. Validation
    # Manual calculation: dot product of features and weights [1, 2, -1]
    weights = jnp.array([1.0, 2.0, -1.0])
    expected_u = jnp.einsum("saf, f -> sa", features, weights)
    
    assert u_matrix.shape == (model.num_states, model.num_actions)
    assert jnp.allclose(u_matrix, expected_u)


def test_linear_utility_mismatch(mock_model_data):
    """Test error handling for dimension mismatch in LinearUtility."""
    model, params, _ = mock_model_data
    
    # Only 2 params provided for 3 features -> Should raise ValueError
    utility = ecx.LinearUtility(
        param_keys=("beta_0", "beta_1"), 
        feature_key="features"
    )
    
    with pytest.raises(ValueError, match="Dimension mismatch"):
        utility.compute_flow_utility(params, model)


def test_function_utility_class(mock_model_data):
    """Test manual usage of FunctionUtility class."""
    model, params, _ = mock_model_data
    
    # Define a simple function logic
    def simple_logic(p, m):
        # U = gamma * other_val (constant across s, a)
        val = p["gamma"] * m.data["other_val"]
        return jnp.full((m.num_states, m.num_actions), val)
    
    # Wrap in class
    utility = ecx.FunctionUtility(func=simple_logic)
    
    u_matrix = utility.compute_flow_utility(params, model)
    
    expected_val = 0.5 * 10.0 # 5.0
    assert jnp.allclose(u_matrix, expected_val)


def test_utility_decorator(mock_model_data):
    """Test the @ecx.utility decorator syntax."""
    model, params, features = mock_model_data
    
    # Define using decorator
    @ecx.utility
    def my_custom_utility(params, model):
        # Logic: beta_0 * feature[0] + 100
        b0 = params["beta_0"]
        f0 = model.data["features"][..., 0] # (S, A)
        return b0 * f0 + 100.0
        
    # Check if it satisfies the interface
    assert isinstance(my_custom_utility, ecx.FunctionUtility)
    
    u_matrix = my_custom_utility.compute_flow_utility(params, model)
    
    # Validation
    expected = 1.0 * features[..., 0] + 100.0
    assert jnp.allclose(u_matrix, expected)

def test_utility_error_handling(mock_model_data):
    model, params, _ = mock_model_data
    
    # 1. Shape mismatch
    @ecx.utility
    def bad_shape_utility(p, m):
        return jnp.zeros((1,)) # Wrong shape
    
    with pytest.raises((ValueError, TypeError)): 
        bad_shape_utility.compute_flow_utility(params, model)

    # 2. User exception
    @ecx.utility
    def error_utility(p, m):
        raise RuntimeError("User logic failed")

    with pytest.raises(RuntimeError, match="User logic failed"):
        error_utility.compute_flow_utility(params, model)


# =============================================================================
# MixedUtility
# =============================================================================

@pytest.fixture
def mixed_setup(mock_model_data):
    """Builds a MixedUtility with NormalDistribution over LinearUtility."""
    model, params, features = mock_model_data
    S, A = model.num_states, model.num_actions  # 4, 2
    F = features.shape[-1]  # 3

    # Add scale parameters to params
    mixed_params = dict(params)
    mixed_params["sigma_beta_0"] = 0.5
    mixed_params["sigma_beta_1"] = 0.3

    base_utility = ecx.LinearUtility(
        param_keys=("beta_0", "beta_1", "beta_2"),
        feature_key="features"
    )
    dist = ecx.NormalDistribution()
    key = jax.random.PRNGKey(7)
    num_draws = 32

    mixed_utility = ecx.MixedUtility(
        base_utility=base_utility,
        distribution=dist,
        coeff_keys=("beta_0", "beta_1"),
        scale_keys=("sigma_beta_0", "sigma_beta_1"),
        num_draws=num_draws,
        num_params=2,
        key=key
    )
    return mixed_utility, mixed_params, model, num_draws


def test_mixed_utility_output_shape(mixed_setup):
    """compute_flow_utility returns (num_draws, S, A)."""
    mixed_utility, mixed_params, model, num_draws = mixed_setup
    result = mixed_utility.compute_flow_utility(mixed_params, model)
    assert result.shape == (num_draws, model.num_states, model.num_actions)


def test_mixed_utility_output_finite(mixed_setup):
    """All output values are finite."""
    import jax.numpy as jnp
    mixed_utility, mixed_params, model, num_draws = mixed_setup
    result = mixed_utility.compute_flow_utility(mixed_params, model)
    assert jnp.all(jnp.isfinite(result))


def test_mixed_utility_draws_vary_across_draws(mixed_setup):
    """Different draws should produce different utility values (not all identical)."""
    import jax.numpy as jnp
    mixed_utility, mixed_params, model, _ = mixed_setup
    result = mixed_utility.compute_flow_utility(mixed_params, model)
    # At least two draw-slices should differ
    assert not jnp.allclose(result[0], result[1])


def test_mixed_utility_zero_scale_collapses_to_mean(mock_model_data):
    """With scale=0, all draws collapse to the mean → all draw slices equal."""
    import jax.numpy as jnp
    model, params, _ = mock_model_data

    params_zero_scale = dict(params)
    params_zero_scale["sigma_beta_0"] = 0.0
    params_zero_scale["sigma_beta_1"] = 0.0

    base_utility = ecx.LinearUtility(
        param_keys=("beta_0", "beta_1", "beta_2"),
        feature_key="features"
    )
    dist = ecx.NormalDistribution()
    mixed_utility = ecx.MixedUtility(
        base_utility=base_utility,
        distribution=dist,
        coeff_keys=("beta_0", "beta_1"),
        scale_keys=("sigma_beta_0", "sigma_beta_1"),
        num_draws=16,
        num_params=2,
        key=jax.random.PRNGKey(0)
    )
    result = mixed_utility.compute_flow_utility(params_zero_scale, model)
    # All draws should give the same utility matrix
    assert jnp.allclose(result[0], result[1])


def test_mixed_utility_with_gumbel_distribution(mock_model_data):
    """MixedUtility also works with GumbelDistribution."""
    import jax.numpy as jnp
    model, params, _ = mock_model_data

    mixed_params = dict(params)
    mixed_params["sigma_beta_0"] = 0.5
    mixed_params["sigma_beta_1"] = 0.3

    base_utility = ecx.LinearUtility(
        param_keys=("beta_0", "beta_1", "beta_2"),
        feature_key="features"
    )
    dist = ecx.GumbelDistribution(scale=1.0)
    num_draws = 20
    mixed_utility = ecx.MixedUtility(
        base_utility=base_utility,
        distribution=dist,
        coeff_keys=("beta_0", "beta_1"),
        scale_keys=("sigma_beta_0", "sigma_beta_1"),
        num_draws=num_draws,
        num_params=2,
        key=jax.random.PRNGKey(1)
    )
    result = mixed_utility.compute_flow_utility(mixed_params, model)
    assert result.shape == (num_draws, model.num_states, model.num_actions)
    assert jnp.all(jnp.isfinite(result))