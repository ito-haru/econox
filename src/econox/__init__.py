# src/econox/__init__.py
"""
Econox: Structural modeling and estimation in JAX.

Econox provides an Equinox-based framework for defining economic models,
simulating data, and performing structural estimation.

Main components:
    - models: Base classes for structural models (e.g., Logit, DDCM).
    - estimation: Tools for MLE, GMM, and other estimation methods.
    - simulation: Utilities for forward simulation and data generation.
"""

# =============================================================================
# User-Facing API (Shortcuts)
# =============================================================================

# 1. Structures (Data Containers & Results)
from econox.structures import (
    Model,
    ParameterSpace,
    SolverResult,
    EstimationResult
)

# 2. Logic Components (Building Blocks)
from econox.logic import (
    LinearUtility,
    utility,
    FunctionUtility,
    GumbelDistribution,
    CompositeFeedback,
    function_feedback,
    model_feedback,
    FunctionFeedback,
    CustomUpdateFeedback,
    SimpleDynamics,
    TrajectoryDynamics,
    IdentityTerminal,
    StationaryTerminal,
    ExponentialTrendTerminal,
    LinearTrendTerminal,
)

# 3. Solvers (Computational Engines)
from econox.solvers import ValueIterationSolver, EquilibriumSolver

# 4. Workflow (High-level APIs)
from econox.workflow import Estimator, Simulator, simulator_objective_from_func, SimulatorObjective, Scenario

# 5. Methods (Estimation Techniques)
from econox.methods import MaximumLikelihood, GaussianMomentMatch, CompositeMethod, LeastSquares, TwoStageLeastSquares

__all__ = [
    # Structures
    "Model",
    "ParameterSpace",
    "SolverResult",
    "EstimationResult",
    # Logic
    "LinearUtility",
    "utility",
    "FunctionUtility",
    "GumbelDistribution",
    "CompositeFeedback",
    "function_feedback",
    "model_feedback",
    "FunctionFeedback",
    "CustomUpdateFeedback",
    "SimpleDynamics",
    "TrajectoryDynamics",
    "IdentityTerminal",
    "StationaryTerminal",
    "ExponentialTrendTerminal",
    "LinearTrendTerminal",
    # Solvers
    "ValueIterationSolver",
    "EquilibriumSolver",
    # Workflow
    "Estimator",
    "Scenario",
    "SimulatorObjective",
    "Simulator",
    "simulator_objective_from_func",
    # Methods
    "MaximumLikelihood",
    "GaussianMomentMatch",
    "CompositeMethod",
    "LeastSquares",
    "TwoStageLeastSquares"
]