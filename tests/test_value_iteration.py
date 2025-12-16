"""
Tests for the ValueIterationSolver component.

This module validates the correctness of the fixed-point iteration solver by 
reproducing the logic of the canonical NFXP (Nested Fixed Point) example. 
It verifies convergence, output shapes, and adherence to the Bellman equation 
under various discount factors.
"""

import logging
from typing import Dict, Any, Generator, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from econox import (
    Model,
    SolverResult, 
    LinearUtility, 
    GumbelDistribution,
    ValueIterationSolver
)

# =============================================================================
# Configuration & Constants
# =============================================================================

TOLERANCE: float = 5e-6

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module", autouse=True)
def configure_jax_precision() -> Generator[None, None, None]:
    """Enable x64 precision for the duration of this test module."""
    orig_val = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", orig_val)


@pytest.fixture(scope="module")
def nfxp_example_data() -> Dict[str, Any]:
    """Provides synthetic data for a simple 4-state, 2-action DDCM."""
    # Feature tensor: (State=4, Action=2, Feature=3)
    state_variables = jnp.array([
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [[0.0, 0.5, 0.0], [1.0, 0.0, 0.5]],
        [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]],
        [[0.0, 1.5, 0.0], [1.0, 0.0, 1.5]],
    ], dtype=jnp.float64)

    # Transition probabilities: (State=4, Action=2, NextState=4)
    transition_tensor = jnp.array([
        [[0.4, 0.3, 0.2, 0.1], [0.1, 0.2, 0.3, 0.4]],
        [[0.2, 0.3, 0.3, 0.2], [0.3, 0.2, 0.2, 0.3]],
        [[0.1, 0.2, 0.4, 0.3], [0.25, 0.25, 0.25, 0.25]],
        [[0.3, 0.2, 0.1, 0.4], [0.2, 0.4, 0.2, 0.2]],
    ], dtype=jnp.float64)
    
    assert jnp.allclose(jnp.sum(transition_tensor, axis=2), 1.0)
    transition_matrix = transition_tensor.reshape(8, 4)
    true_coeffs = jnp.array([0.5, 1.0, -1.0], dtype=jnp.float64)

    return {
        "features": state_variables,
        "transitions": transition_matrix,
        "coeffs": true_coeffs,
        "num_states": 4,
        "num_actions": 2
    }


@pytest.fixture(params=[0.0, 0.5, 0.9, 0.99])
def solver_run(request, nfxp_example_data) -> Tuple[SolverResult, float, Dict[str, Any]]:
    """
    Executes the solver for a specific discount factor.
    Returns: (result, discount_factor, data)
    """
    discount_factor = request.param
    data = nfxp_example_data
    
    model = Model.from_data(
        num_states=data["num_states"],
        num_actions=data["num_actions"],
        data={"features": data["features"]},
        transitions=data["transitions"],
        num_periods=np.inf
    )

    utility_logic = LinearUtility(param_keys=("coeffs",), feature_key="features")
    distribution_logic = GumbelDistribution(scale=1.0)
    solver = ValueIterationSolver(
        utility=utility_logic,
        dist=distribution_logic,
        discount_factor=discount_factor,
    )
    params = {"coeffs": data["coeffs"]}

    logger.debug(f"Running solver with beta={discount_factor}...")
    result = solver.solve(params, model)
    
    return result, discount_factor, data


# =============================================================================
# Test Cases (Separated)
# =============================================================================

def test_convergence(solver_run: Tuple[SolverResult, float, Dict[str, Any]]) -> None:
    """Check if the solver reports success."""
    result, beta, _ = solver_run
    steps = result.aux.get("num_steps", -1)
    
    assert result.success, \
        f"Solver failed to converge with beta={beta}. Steps: {steps}"


def test_output_shapes(solver_run: Tuple[SolverResult, float, Dict[str, Any]]) -> None:
    """Check if the output arrays have correct dimensions."""
    result, _, data = solver_run
    
    expected_v_shape = (data["num_states"],)
    expected_p_shape = (data["num_states"], data["num_actions"])
    
    assert result.solution.shape == expected_v_shape
    assert result.profile is not None
    assert result.profile.shape == expected_p_shape


def test_fixed_point_property(solver_run: Tuple[SolverResult, float, Dict[str, Any]]) -> None:
    """Verify that the solution satisfies V = T(V)."""
    result, beta, data = solver_run
    
    # Manually compute one Bellman step
    expected_utility = jnp.einsum("saf,f->sa", data["features"], data["coeffs"])
    ev_flat = data["transitions"] @ result.solution
    ev_reshaped = ev_flat.reshape(data["num_states"], data["num_actions"])
    
    q_values = expected_utility + beta * ev_reshaped
    v_check = jax.scipy.special.logsumexp(q_values, axis=1)

    max_diff = jnp.max(jnp.abs(result.solution - v_check))
    
    assert max_diff < TOLERANCE, \
        f"Fixed point property violated (beta={beta}). Diff: {max_diff:.2e}"


def test_probability_constraints(solver_run: Tuple[SolverResult, float, Dict[str, Any]]) -> None:
    """Verify that choice probabilities sum to 1.0."""
    result, beta, _ = solver_run
    
    assert result.profile is not None
    probs_sum = jnp.sum(result.profile, axis=1)
    
    assert jnp.allclose(probs_sum, 1.0, atol=TOLERANCE), \
        f"Probabilities do not sum to 1 (beta={beta}). Got: {probs_sum}"


# =============================================================================
# Invalid Input Tests (Unchanged)
# =============================================================================

def test_value_iteration_invalid_inputs() -> None:
    """Test error handling for invalid inputs."""
    dummy_data = {"dummy": jnp.array([1])}

    # Case 1: Missing transitions
    model_no_trans = Model.from_data(
        num_states=4, num_actions=2, data=dummy_data, transitions=None
    )
    solver = ValueIterationSolver(
        utility=LinearUtility((), feature_key="dummy"),
        dist=GumbelDistribution(),
        discount_factor=0.9
    )
    
    with pytest.raises(ValueError, match="Model transitions must be defined"):
        solver.solve({}, model_no_trans)

    # Case 2: Invalid Shape
    invalid_trans_3d = jnp.zeros((2, 4, 4)) 
    model_3d = Model.from_data(
        num_states=4, num_actions=2, data=dummy_data, transitions=invalid_trans_3d
    )
    
    with pytest.raises(ValueError, match="MVP Version only supports"):
        solver.solve({}, model_3d)