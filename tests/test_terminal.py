"""
Tests for Terminal Value Function Approximators.
Validates IdentityTerminal, StationaryTerminal, ExponentialTrendTerminal, and LinearTrendTerminal.
"""

import jax
import jax.numpy as jnp
import pytest
import equinox as eqx
from jaxtyping import PyTree

from econox.logic.terminal import (
    IdentityTerminal,
    StationaryTerminal,
    ExponentialTrendTerminal,
    LinearTrendTerminal,
    _retrieve_and_validate_param
)
from econox.config import NUMERICAL_EPSILON


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_model():
    """Creates a minimal dummy model for testing."""
    class DummyModel(eqx.Module):
        num_states: int = 10
        num_actions: int = 3
        num_periods: int = 20

    return DummyModel()


@pytest.fixture
def sample_expected_values():
    """Creates a sample expected value matrix (S x A)."""
    key = jax.random.PRNGKey(42)
    # 10 states, 3 actions
    return jax.random.uniform(key, (10, 3), minval=0.0, maxval=100.0)


@pytest.fixture
def sample_params():
    """Creates sample parameters for testing."""
    return {
        "g": 0.02,
        "g1": 0.01,
        "g2": 0.03,
        "g3": 0.02,
        "g_vector": jnp.array([0.01, 0.02, 0.03]),
        "drift": 5.0,
        "drift1": 3.0,
        "drift2": 7.0,
        "drift3": 5.0,
        "drift_vector": jnp.array([3.0, 5.0, 7.0])
    }


# =============================================================================
# Helper Function Tests
# =============================================================================

def test_retrieve_and_validate_param_scalar_key(sample_params):
    """Test parameter retrieval with a single scalar key."""
    prev_idx = (0, 1, 2)
    result = _retrieve_and_validate_param("g", sample_params, prev_idx, "test_param")
    
    # Should return scalar value
    assert result == 0.02
    assert jnp.isscalar(result)


def test_retrieve_and_validate_param_vector_key(sample_params):
    """Test parameter retrieval with a vector key."""
    prev_idx = (0, 1, 2)
    result = _retrieve_and_validate_param("g_vector", sample_params, prev_idx, "test_param")
    
    # Should return (3, 1) array for broadcasting
    assert isinstance(result, jnp.ndarray)
    assert result.shape == (3, 1)
    assert jnp.allclose(result.flatten(), jnp.array([0.01, 0.02, 0.03]))


def test_retrieve_and_validate_param_list_keys(sample_params):
    """Test parameter retrieval with a list of keys."""
    prev_idx = (0, 1, 2)
    result = _retrieve_and_validate_param(["g1", "g2", "g3"], sample_params, prev_idx, "test_param")
    
    # Should aggregate into (3, 1) array
    assert isinstance(result, jnp.ndarray)
    assert result.shape == (3, 1)
    assert jnp.allclose(result.flatten(), jnp.array([0.01, 0.03, 0.02]))


def test_retrieve_and_validate_param_shape_mismatch(sample_params):
    """Test that shape validation raises ValueError on mismatch."""
    prev_idx = (0, 1, 2)  # Length 3
    
    with pytest.raises(ValueError, match="incompatible shape"):
        # g_vector has length 3, but we'll create a prev_idx of different length
        wrong_idx = (0, 1)  # Length 2
        _retrieve_and_validate_param("g_vector", sample_params, wrong_idx, "test_param")


def test_retrieve_and_validate_param_missing_key():
    """Test behavior when key is missing from params."""
    prev_idx = (0, 1, 2)
    result = _retrieve_and_validate_param("nonexistent_key", {}, prev_idx, "test_param")
    
    # Should return default 0.0
    assert result == 0.0


# =============================================================================
# IdentityTerminal Tests
# =============================================================================

def test_identity_terminal_no_modification(sample_expected_values, sample_params, mock_model):
    """Test that IdentityTerminal returns values unchanged."""
    approximator = IdentityTerminal()
    result = approximator.approximate(sample_expected_values, sample_params, mock_model)
    
    # Should be identical
    assert jnp.allclose(result, sample_expected_values)
    assert result.shape == sample_expected_values.shape


def test_identity_terminal_preserves_dtype(mock_model):
    """Test that IdentityTerminal preserves data type."""
    expected = jnp.ones((5, 2), dtype=jnp.float32)
    approximator = IdentityTerminal()
    result = approximator.approximate(expected, {}, mock_model)
    
    assert result.dtype == jnp.float32


# =============================================================================
# StationaryTerminal Tests
# =============================================================================

def test_stationary_terminal_basic(sample_expected_values, sample_params, mock_model):
    """Test basic stationary approximation."""
    # States 8, 9 are terminal, copy from states 6, 7
    term_idx = (8, 9)
    prev_idx = (6, 7)
    
    approximator = StationaryTerminal(term_idx=term_idx, prev_idx=prev_idx)
    result = approximator.approximate(sample_expected_values, sample_params, mock_model)
    
    # Terminal states should match previous states
    assert jnp.allclose(result[8, :], sample_expected_values[6, :])
    assert jnp.allclose(result[9, :], sample_expected_values[7, :])
    
    # Other states should be unchanged
    assert jnp.allclose(result[0, :], sample_expected_values[0, :])
    assert jnp.allclose(result[5, :], sample_expected_values[5, :])


def test_stationary_terminal_shape_validation(sample_expected_values, sample_params, mock_model):
    """Test that StationaryTerminal validates index shape consistency."""
    term_idx = (8, 9)
    prev_idx = (6,)  # Mismatched length
    
    approximator = StationaryTerminal(term_idx=term_idx, prev_idx=prev_idx)
    
    with pytest.raises(ValueError, match="must have the same shape"):
        approximator.approximate(sample_expected_values, sample_params, mock_model)


def test_stationary_terminal_single_state(sample_expected_values, sample_params, mock_model):
    """Test stationary approximation for a single terminal state."""
    term_idx = (9,)
    prev_idx = (7,)
    
    approximator = StationaryTerminal(term_idx=term_idx, prev_idx=prev_idx)
    result = approximator.approximate(sample_expected_values, sample_params, mock_model)
    
    assert jnp.allclose(result[9, :], sample_expected_values[7, :])


# =============================================================================
# ExponentialTrendTerminal Tests
# =============================================================================

def test_exponential_trend_exogenous_scalar(sample_expected_values, sample_params, mock_model):
    """Test exponential trend with scalar growth rate."""
    term_idx = (8, 9)
    prev_idx = (6, 7)
    
    approximator = ExponentialTrendTerminal(
        term_idx=term_idx,
        prev_idx=prev_idx,
        growth_rate_keys="g"
    )
    result = approximator.approximate(sample_expected_values, sample_params, mock_model)
    
    # Expected: val_t_minus_1 * (1 + 0.02)
    expected_8 = sample_expected_values[6, :] * 1.02
    expected_9 = sample_expected_values[7, :] * 1.02
    
    assert jnp.allclose(result[8, :], expected_8)
    assert jnp.allclose(result[9, :], expected_9)


def test_exponential_trend_exogenous_vector(sample_expected_values, sample_params, mock_model):
    """Test exponential trend with vector growth rates."""
    term_idx = (7, 8, 9)
    prev_idx = (4, 5, 6)
    
    approximator = ExponentialTrendTerminal(
        term_idx=term_idx,
        prev_idx=prev_idx,
        growth_rate_keys="g_vector"
    )
    result = approximator.approximate(sample_expected_values, sample_params, mock_model)
    
    # Expected: val_t_minus_1 * (1 + gamma_i)
    expected_7 = sample_expected_values[4, :] * 1.01
    expected_8 = sample_expected_values[5, :] * 1.02
    expected_9 = sample_expected_values[6, :] * 1.03
    
    assert jnp.allclose(result[7, :], expected_7)
    assert jnp.allclose(result[8, :], expected_8)
    assert jnp.allclose(result[9, :], expected_9)


def test_exponential_trend_exogenous_list_keys(sample_expected_values, sample_params, mock_model):
    """Test exponential trend with aggregated regional scalars."""
    term_idx = (7, 8, 9)
    prev_idx = (4, 5, 6)
    
    approximator = ExponentialTrendTerminal(
        term_idx=term_idx,
        prev_idx=prev_idx,
        growth_rate_keys=["g1", "g2", "g3"]
    )
    result = approximator.approximate(sample_expected_values, sample_params, mock_model)
    
    # Expected: val_t_minus_1 * (1 + gamma_i)
    expected_7 = sample_expected_values[4, :] * 1.01
    expected_8 = sample_expected_values[5, :] * 1.03
    expected_9 = sample_expected_values[6, :] * 1.02
    
    assert jnp.allclose(result[7, :], expected_7)
    assert jnp.allclose(result[8, :], expected_8)
    assert jnp.allclose(result[9, :], expected_9)


def test_exponential_trend_endogenous(sample_expected_values, sample_params, mock_model):
    """Test exponential trend with endogenous extrapolation."""
    term_idx = (8, 9)
    prev_idx = (6, 7)
    pre_prev_idx = (4, 5)
    
    approximator = ExponentialTrendTerminal(
        term_idx=term_idx,
        prev_idx=prev_idx,
        pre_prev_idx=pre_prev_idx
    )
    result = approximator.approximate(sample_expected_values, sample_params, mock_model)
    
    # Expected: val_t_minus_1 * (val_t_minus_1 / val_t_minus_2)
    val_t_minus_1 = sample_expected_values[prev_idx, :]
    val_t_minus_2 = sample_expected_values[pre_prev_idx, :]
    
    # Check numerical stability: ratio should be 1.0 where denominator is near zero
    denominator = jnp.abs(val_t_minus_2)
    ratio = jnp.where(
        denominator > NUMERICAL_EPSILON,
        val_t_minus_1 / val_t_minus_2,
        1.0
    )
    expected_vals = val_t_minus_1 * ratio
    
    assert jnp.allclose(result[8, :], expected_vals[0, :])
    assert jnp.allclose(result[9, :], expected_vals[1, :])


def test_exponential_trend_endogenous_numerical_stability(sample_params, mock_model):
    """Test numerical stability when val_t_minus_2 is near zero."""
    # Create expected values where some T-2 values are very small
    expected = jnp.array([
        [100.0, 50.0, 30.0],
        [1e-10, 1e-9, 20.0],  # Near-zero values
        [120.0, 60.0, 40.0],
        [0.0, 1e-12, 25.0]    # Zero and near-zero
    ])
    
    term_idx = (2, 3)
    prev_idx = (2, 3)
    pre_prev_idx = (1, 1)
    
    approximator = ExponentialTrendTerminal(
        term_idx=term_idx,
        prev_idx=prev_idx,
        pre_prev_idx=pre_prev_idx
    )
    result = approximator.approximate(expected, sample_params, mock_model)
    
    # Should not produce NaN or Inf
    assert jnp.all(jnp.isfinite(result))
    
    # When denominator is near zero, ratio should be 1.0 (stationary)
    # So result should be close to val_t_minus_1
    assert jnp.allclose(result[2, 0], expected[2, 0], rtol=1e-5)  # Stationary for near-zero


def test_exponential_trend_negative_values(sample_params, mock_model):
    """Test exponential trend with negative value functions (cost minimization)."""
    # Create expected values with negative costs
    expected = jnp.array([
        [-100.0, -80.0, -60.0],
        [-50.0, -40.0, -30.0],
        [-120.0, -100.0, -70.0],
        [-60.0, -50.0, -35.0]
    ])
    
    term_idx = (2, 3)
    prev_idx = (2, 3)
    pre_prev_idx = (0, 1)
    
    approximator = ExponentialTrendTerminal(
        term_idx=term_idx,
        prev_idx=prev_idx,
        pre_prev_idx=pre_prev_idx
    )
    result = approximator.approximate(expected, sample_params, mock_model)
    
    # Should handle negative values correctly
    val_t_minus_1 = expected[prev_idx, :]
    val_t_minus_2 = expected[pre_prev_idx, :]
    denominator = jnp.abs(val_t_minus_2)
    ratio = jnp.where(
        denominator > NUMERICAL_EPSILON,
        val_t_minus_1 / val_t_minus_2,
        1.0
    )
    expected_vals = val_t_minus_1 * ratio
    
    assert jnp.allclose(result[2, :], expected_vals[0, :])
    assert jnp.allclose(result[3, :], expected_vals[1, :])


def test_exponential_trend_requires_keys_or_indices(sample_expected_values, mock_model):
    """Test that ExponentialTrendTerminal requires either keys or pre_prev_idx."""
    term_idx = (8, 9)
    prev_idx = (6, 7)
    
    approximator = ExponentialTrendTerminal(
        term_idx=term_idx,
        prev_idx=prev_idx,
        growth_rate_keys=None,
        pre_prev_idx=None
    )
    
    with pytest.raises(ValueError, match="requires either"):
        approximator.approximate(sample_expected_values, {}, mock_model)


def test_exponential_trend_index_shape_validation(sample_expected_values, sample_params, mock_model):
    """Test index shape validation for pre_prev_idx."""
    term_idx = (8, 9)
    prev_idx = (6, 7)
    pre_prev_idx = (4,)  # Mismatched length
    
    approximator = ExponentialTrendTerminal(
        term_idx=term_idx,
        prev_idx=prev_idx,
        pre_prev_idx=pre_prev_idx
    )
    
    with pytest.raises(ValueError, match="must have the same shape"):
        approximator.approximate(sample_expected_values, sample_params, mock_model)


# =============================================================================
# LinearTrendTerminal Tests
# =============================================================================

def test_linear_trend_exogenous_scalar(sample_expected_values, sample_params, mock_model):
    """Test linear trend with scalar drift."""
    term_idx = (8, 9)
    prev_idx = (6, 7)
    
    approximator = LinearTrendTerminal(
        term_idx=term_idx,
        prev_idx=prev_idx,
        drift_keys="drift"
    )
    result = approximator.approximate(sample_expected_values, sample_params, mock_model)
    
    # Expected: val_t_minus_1 + 5.0
    expected_8 = sample_expected_values[6, :] + 5.0
    expected_9 = sample_expected_values[7, :] + 5.0
    
    assert jnp.allclose(result[8, :], expected_8)
    assert jnp.allclose(result[9, :], expected_9)


def test_linear_trend_exogenous_vector(sample_expected_values, sample_params, mock_model):
    """Test linear trend with vector drifts."""
    term_idx = (7, 8, 9)
    prev_idx = (4, 5, 6)
    
    approximator = LinearTrendTerminal(
        term_idx=term_idx,
        prev_idx=prev_idx,
        drift_keys="drift_vector"
    )
    result = approximator.approximate(sample_expected_values, sample_params, mock_model)
    
    # Expected: val_t_minus_1 + delta_i
    expected_7 = sample_expected_values[4, :] + 3.0
    expected_8 = sample_expected_values[5, :] + 5.0
    expected_9 = sample_expected_values[6, :] + 7.0
    
    assert jnp.allclose(result[7, :], expected_7)
    assert jnp.allclose(result[8, :], expected_8)
    assert jnp.allclose(result[9, :], expected_9)


def test_linear_trend_exogenous_list_keys(sample_expected_values, sample_params, mock_model):
    """Test linear trend with aggregated regional drifts."""
    term_idx = (7, 8, 9)
    prev_idx = (4, 5, 6)
    
    approximator = LinearTrendTerminal(
        term_idx=term_idx,
        prev_idx=prev_idx,
        drift_keys=["drift1", "drift2", "drift3"]
    )
    result = approximator.approximate(sample_expected_values, sample_params, mock_model)
    
    # Expected: val_t_minus_1 + delta_i
    expected_7 = sample_expected_values[4, :] + 3.0
    expected_8 = sample_expected_values[5, :] + 7.0
    expected_9 = sample_expected_values[6, :] + 5.0
    
    assert jnp.allclose(result[7, :], expected_7)
    assert jnp.allclose(result[8, :], expected_8)
    assert jnp.allclose(result[9, :], expected_9)


def test_linear_trend_endogenous(sample_expected_values, sample_params, mock_model):
    """Test linear trend with endogenous extrapolation."""
    term_idx = (8, 9)
    prev_idx = (6, 7)
    pre_prev_idx = (4, 5)
    
    approximator = LinearTrendTerminal(
        term_idx=term_idx,
        prev_idx=prev_idx,
        pre_prev_idx=pre_prev_idx
    )
    result = approximator.approximate(sample_expected_values, sample_params, mock_model)
    
    # Expected: val_t_minus_1 + (val_t_minus_1 - val_t_minus_2)
    val_t_minus_1 = sample_expected_values[prev_idx, :]
    val_t_minus_2 = sample_expected_values[pre_prev_idx, :]
    diff = val_t_minus_1 - val_t_minus_2
    expected_vals = val_t_minus_1 + diff
    
    assert jnp.allclose(result[8, :], expected_vals[0, :])
    assert jnp.allclose(result[9, :], expected_vals[1, :])


def test_linear_trend_negative_values(sample_params, mock_model):
    """Test linear trend with negative value functions."""
    expected = jnp.array([
        [-100.0, -80.0, -60.0],
        [-50.0, -40.0, -30.0],
        [-120.0, -100.0, -70.0],
        [-60.0, -50.0, -35.0]
    ])
    
    term_idx = (2, 3)
    prev_idx = (2, 3)
    pre_prev_idx = (0, 1)
    
    approximator = LinearTrendTerminal(
        term_idx=term_idx,
        prev_idx=prev_idx,
        pre_prev_idx=pre_prev_idx
    )
    result = approximator.approximate(expected, sample_params, mock_model)
    
    # Should handle negative values correctly
    val_t_minus_1 = expected[prev_idx, :]
    val_t_minus_2 = expected[pre_prev_idx, :]
    diff = val_t_minus_1 - val_t_minus_2
    expected_vals = val_t_minus_1 + diff
    
    assert jnp.allclose(result[2, :], expected_vals[0, :])
    assert jnp.allclose(result[3, :], expected_vals[1, :])


def test_linear_trend_requires_keys_or_indices(sample_expected_values, mock_model):
    """Test that LinearTrendTerminal requires either keys or pre_prev_idx."""
    term_idx = (8, 9)
    prev_idx = (6, 7)
    
    approximator = LinearTrendTerminal(
        term_idx=term_idx,
        prev_idx=prev_idx,
        drift_keys=None,
        pre_prev_idx=None
    )
    
    with pytest.raises(ValueError, match="requires either"):
        approximator.approximate(sample_expected_values, {}, mock_model)


def test_linear_trend_structure_consistency(sample_expected_values, sample_params, mock_model):
    """Test that LinearTrendTerminal returns consistently at the end (not in branches)."""
    term_idx = (8, 9)
    prev_idx = (6, 7)
    
    # Test with exogenous keys
    approx1 = LinearTrendTerminal(term_idx=term_idx, prev_idx=prev_idx, drift_keys="drift")
    result1 = approx1.approximate(sample_expected_values, sample_params, mock_model)
    assert result1.shape == sample_expected_values.shape
    
    # Test with endogenous
    pre_prev_idx = (4, 5)
    approx2 = LinearTrendTerminal(term_idx=term_idx, prev_idx=prev_idx, pre_prev_idx=pre_prev_idx)
    result2 = approx2.approximate(sample_expected_values, sample_params, mock_model)
    assert result2.shape == sample_expected_values.shape


# =============================================================================
# Integration Tests
# =============================================================================

def test_all_approximators_preserve_shape(sample_expected_values, sample_params, mock_model):
    """Test that all approximators preserve the input shape."""
    term_idx = (8, 9)
    prev_idx = (6, 7)
    
    approximators = [
        IdentityTerminal(),
        StationaryTerminal(term_idx=term_idx, prev_idx=prev_idx),
        ExponentialTrendTerminal(term_idx=term_idx, prev_idx=prev_idx, growth_rate_keys="g"),
        LinearTrendTerminal(term_idx=term_idx, prev_idx=prev_idx, drift_keys="drift")
    ]
    
    for approx in approximators:
        result = approx.approximate(sample_expected_values, sample_params, mock_model)
        assert result.shape == sample_expected_values.shape


def test_approximators_are_jax_compatible(sample_expected_values, sample_params, mock_model):
    """Test that approximators work with JAX transformations (jit, grad)."""
    term_idx = (8, 9)
    prev_idx = (6, 7)
    
    approximator = ExponentialTrendTerminal(
        term_idx=term_idx,
        prev_idx=prev_idx,
        growth_rate_keys="g"
    )
    
    # Test JIT compilation
    @jax.jit
    def approximate_jit(expected, params):
        return approximator.approximate(expected, params, mock_model)
    
    result = approximate_jit(sample_expected_values, sample_params)
    assert jnp.all(jnp.isfinite(result))
    
    # Test gradient computation (through expected values)
    def loss_fn(expected):
        result = approximator.approximate(expected, sample_params, mock_model)
        return jnp.sum(result)
    
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(sample_expected_values)
    assert grads.shape == sample_expected_values.shape
    assert jnp.all(jnp.isfinite(grads))
