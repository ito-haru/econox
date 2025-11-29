# src/econox/__init__.py
"""
Econox: A JAX-based structural estimation library for dynamic economic models.
"""

__version__ = "0.1.0"

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
    GumbelDistribution,
    LogLinearFeedback
)

# 3. Solvers (Computational Engines)
from econox.solvers.fixed_point import ValueIterationSolver

# 4. Workflow (High-level APIs) -> 将来 Estimator などができたらここに追加
# from econox.workflow import Estimator, Simulator

__all__ = [
    # Structures
    "Model",
    "ParameterSpace",
    "SolverResult",
    "EstimationResult",
    # Logic
    "LinearUtility",
    "GumbelDistribution",
    "LogLinearFeedback",
    # Solvers
    "ValueIterationSolver",
]