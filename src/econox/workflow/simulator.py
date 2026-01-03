# src/econox/workflow/simulator.py
"""
Counterfactual Simulation and Policy Evaluation Workflow.

This module provides the infrastructure for structural simulation in Econox. 
It enables researchers to conduct "What-if" analyses by modifying model 
environments (Scenario), solving for new agent behaviors, and evaluating 
outcomes through differentiable objective functions.

Key Components:
    * Scenario: A container pairing a model environment with its solution.
    * SimulatorObjective: A flexible wrapper for outcome evaluation.
    * Simulator: The orchestrator for the simulation workflow.
"""

import equinox as eqx
from typing import Callable, Any, Dict, TypeVar, Generic
from jaxtyping import PyTree

from econox.structures import SolverResult
from econox.protocols import Solver, StructuralModel

T = TypeVar("T")

class Scenario(eqx.Module):
    """
    A context container that pairs a structural model with its computed solution.

    In economic simulation, a result (e.g., choice probabilities) must always be 
    interpreted relative to its environment (e.g., prices, taxes). This class 
    ensures that the model's data and the solver's output are bundled together.

    Attributes:
        model: The StructuralModel instance representing the environment.
        result: The SolverResult containing the policy or equilibrium solution.

    Note:
        Scenario is immutable (frozen) and should be treated as a read-only 
        container. This prevents accidental modifications during simulation 
        workflows and supports JAX/Equinox functional-style usage.

    Examples:
        >>> # Accessing data within a scenario
        >>> tax_rate = scenario.model.data['tax']
        >>> choices = scenario.result.profile
    """
    model: StructuralModel
    result: SolverResult

class SimulatorObjective(eqx.Module, Generic[T]):
    """
    Interface for evaluating simulation outcomes.

    A SimulatorObjective calculates metrics such as welfare changes or tax revenue 
    by comparing scenarios. It remains differentiable and can carry its own 
    internal state (e.g., social welfare weights).

    Examples:
        >>> # Usage via the decorator (recommended)
        >>> @simulator_objective_from_func
        >>> def welfare_change(cf, base, params):
        ...     # Difference in average outcomes between scenarios
        ...     return cf.result.solution.mean() - base.result.solution.mean()
    """
    func: Callable[[Scenario, Scenario, PyTree], T]

    def __call__(
        self, 
        cf: Scenario,
        base: Scenario,
        params: PyTree
    ) -> T:
        """
        Executes the objective evaluation logic.

        Args:
            cf: The counterfactual scenario (Post-intervention).
            base: The baseline scenario (Status quo).
            params: The structural parameters used.

        Returns:
            T: The computed metric (Scalar, Array, or PyTree).
        """
        return self.func(cf, base, params)

# Decorator to create a SimulatorObjective from a function
def simulator_objective_from_func(func: Callable[[Scenario, Scenario, PyTree], T]) -> SimulatorObjective[T]:
    """
    A decorator that converts a Python function into a SimulatorObjective.

    Args:
        func: A function with signature (cf, base, params) -> T.

    Returns:
        SimulatorObjective[T]: A JAX-compatible objective module.

    Examples:
        >>> @simulator_objective_from_func
        ... def tax_revenue_gain(cf, base, params):
        ...     # Revenue = tax_rate * income
        ...     rev_cf = cf.model.data['tax'] * cf.result.aux['income']
        ...     rev_base = base.model.data['tax'] * base.result.aux['income']
        ...     return rev_cf - rev_base
    """
    return SimulatorObjective(func=func)


class Simulator(eqx.Module):
    """
    Counterfactual Simulation Engine.

    The Simulator automates the process of updating model data, solving for 
    new agent behaviors, and evaluating outcomes. It is fully compatible with 
    JAX transformations (JIT, Grad), making it ideal for policy optimization.

    Attributes:
        solver: The solver used to find the model's solution.
        base_model: The reference structural model.
        objective_function: The logic used to evaluate results.

    Examples:
        >>> # 1. Define objective
        >>> @simulator_objective_from_func
        ... def diff_obj(cf, base, params):
        ...     return cf.result.solution.mean() - base.result.solution.mean()
        >>>
        >>> # 2. Initialize Simulator
        >>> sim = Simulator(solver=my_solver, base_model=model, objective_function=diff_obj)
        >>>
        >>> # 3. Run simulation
        >>> benefits = sim(params, updates={'tax': 0.25})
    """
    solver: Solver
    base_model: StructuralModel
    objective_function: SimulatorObjective[Any]

    def __call__(
        self, 
        params: PyTree,
        updates: Dict[str, Any],
        base_result: SolverResult | None = None
    ) -> PyTree:
        """
        Performs a counterfactual simulation and evaluates the objective.

        Args:
            params: Structural parameters :math:`\\theta`.
            updates: Dictionary of fields in the `base_model` data to be modified.
                Example: `{'tax_rate': 0.25}`.
            base_result: Optional precomputed result for the `base_model`.
                Providing this bypasses the baseline solver step, significantly 
                accelerating policy optimization or "what-if" loops.

                Warning: The caller is responsible for ensuring that `base_result`
                corresponds to solving `base_model` with the provided `params`; 
                passing an inconsistent result may lead to invalid outcomes.

        Returns:
            The output of the `objective_function`.
        """
        cf_model = self.base_model
        for key, val in updates.items():
            try:
                cf_model = cf_model.replace_data(key, val)
            except KeyError as e:
                raise 
        
        if base_result is None:
            base_result = self.solver.solve(params=params, model=self.base_model)
            if base_result is None:
                raise RuntimeError("Solver failed to produce a base_result.")

        cf_result = self.solver.solve(params=params, model=cf_model)
        if cf_result is None:
            raise RuntimeError("Solver failed to produce a cf_result.")

        base = Scenario(model=self.base_model, result=base_result)
        cf = Scenario(model=cf_model, result=cf_result)

        return self.objective_function(cf, base, params)
