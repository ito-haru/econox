# tests/test_simulator.py

import pytest
import jax.numpy as jnp
import equinox as eqx
from typing import Any
from jaxtyping import PyTree

from econox import Model, SolverResult, Scenario
from econox.workflow.simulator import Simulator, simulator_objective_from_func


class SimpleMockSolver(eqx.Module):
    def solve(self, params: PyTree, model: Any) -> SolverResult:
        success = model.data.get("force_fail", jnp.array(False)) == False
        return SolverResult(
            solution=jnp.array([1.0]),
            profile=jnp.array([0.5, 0.5]),
            success=jnp.array(success) if isinstance(success, bool) else success,
            aux={"val": model.data.get("test_val", jnp.array(0.0))}
        )


@simulator_objective_from_func
def mock_objective(cf: Scenario, base: Scenario, params: PyTree) -> Any:
    return cf.result.aux["val"] - base.result.aux["val"]


@pytest.fixture
def base_model():
    return Model.from_data(
        num_states=1, 
        num_actions=2, 
        data={"test_val": 1.0, "force_fail": False}
    )


@pytest.fixture
def solver():
    return SimpleMockSolver()


@pytest.fixture
def objective():
    return mock_objective


def test_scenario_data_access():
    """Scenario correctly bundles model and result."""
    model = Model.from_data(num_states=1, num_actions=2, data={"key": 10})
    result = SolverResult(
        solution=jnp.array([1.0]), 
        profile=jnp.array([0.5, 0.5]), 
        success=True
    )
    
    scenario = Scenario(model=model, result=result)
    
    assert scenario.model.data["key"] == 10
    assert jnp.array_equal(scenario.result.solution, jnp.array([1.0]))
    assert scenario.result.success is True


def test_simulator_successful_simulation(base_model, solver, objective):
    """Normal simulation flow produces correct outcome."""
    sim = Simulator(solver=solver, base_model=base_model, objective_function=objective)
    
    result = sim(params={}, updates={"test_val": 5.0})
    
    assert jnp.allclose(result, 4.0)  # 5.0 - 1.0


def test_simulator_with_precomputed_base(base_model, solver, objective):
    """Using precomputed base_result skips baseline resolution."""
    sim = Simulator(solver=solver, base_model=base_model, objective_function=objective)
    
    precomputed = solver.solve(params={}, model=base_model)
    
    result = sim(params={}, updates={"test_val": 8.0}, base_result=precomputed)
    
    assert jnp.allclose(result, 7.0)  # 8.0 - 1.0


def test_simulator_no_updates(base_model, solver, objective):
    """Simulation with no updates returns zero difference."""
    sim = Simulator(solver=solver, base_model=base_model, objective_function=objective)
    
    result = sim(params={}, updates={})
    
    assert jnp.allclose(result, 0.0)


def test_simulator_multiple_updates(solver, objective):
    """Multiple data fields can be updated simultaneously."""
    model = Model.from_data(
        num_states=1, 
        num_actions=2, 
        data={"test_val": 2.0, "force_fail": False, "other_key": 10.0}
    )
    sim = Simulator(solver=solver, base_model=model, objective_function=objective)
    
    result = sim(params={}, updates={"test_val": 7.0, "other_key": 20.0})
    
    assert jnp.allclose(result, 5.0)  # 7.0 - 2.0


def test_simulator_invalid_key(base_model, solver, objective):
    """Invalid update key raises KeyError."""
    sim = Simulator(solver=solver, base_model=base_model, objective_function=objective)
    
    with pytest.raises(KeyError, match="Key 'wrong_key' not found in model data"):
        sim(params={}, updates={"wrong_key": 99.0})