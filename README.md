# Econox
[![PyPI version](https://img.shields.io/pypi/v/econox.svg)](https://pypi.org/project/econox/)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![JAX](https://img.shields.io/badge/backend-JAX-informational)](https://github.com/jax-ml/jax)
[![Status](https://img.shields.io/badge/status-alpha-orange)](https://github.com/ito-haru/econox)
[![License](https://img.shields.io/badge/license-Apache_2.0-green)](./LICENSE)

> [!WARNING]
> **Early Development Status**
> 
> Econox is currently under active development. The API is unstable and subject to **breaking changes** at any time.  
> Use with caution in production environments.
> (本ライブラリは現在開発中です。APIは頻繁に変更される可能性があります。)

**A JAX-based toolkit for structural modeling and estimation.**

Econox is a Python library built on [JAX](https://github.com/jax-ml/jax) and [Equinox](https://github.com/patrick-kidger/equinox) designed to streamline the research workflow for structural economists.

While originally motivated by dynamic discrete choice models, Econox provides a modular, object-oriented framework suitable for a wide range of economic models.

### Key Features
* **Modeling:** Define models as clean, composable Python classes using Equinox.
* **Estimation:** Perform gradient-based estimation (Maximum Likelihood Estimation etc.) efficiently with JAX's automatic differentiation and JIT compilation.
* **Simulation (Coming Soon):** Forward-simulation capabilities are currently under active development.

### Requirements

Econox is built upon the modern JAX ecosystem. The core dependencies include:

* **[JAX](https://github.com/jax-ml/jax):** For high-performance array computing and automatic differentiation.
* **[Equinox](https://github.com/patrick-kidger/equinox):** For defining parameterized models and neural networks.
* **[Optimistix](https://github.com/patrick-kidger/optimistix):** For nonlinear optimization and root-finding.
* **[Lineax](https://github.com/patrick-kidger/lineax):** For linear solvers.
* **[Jaxtyping](https://github.com/patrick-kidger/jaxtyping):** For type annotations and shape checking.

### Installation

Requires Python 3.11+ and JAX.

**Using pip:**
```bash
pip install econox
#Using uv (Recommended):
uv add econox
```

### Quick Start

Here is a simple example of solving a dynamic programming problem (Value Iteration).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ito-haru/econox/blob/main/examples/quickstart.ipynb)

```python
import jax.numpy as jnp
from econox import Model, LinearUtility, GumbelDistribution, ValueIterationSolver

# 1. Define the Environment (Data)
# Create a dummy model with 10 states, 3 actions, and random feature 'x'
num_states, num_actions = 10, 3
model = Model.from_data(
    num_states=num_states,
    num_actions=num_actions,
    data={"x": jnp.zeros((num_states, num_actions, 1))},  # Feature tensor
    transitions=jnp.ones((num_states * num_actions, num_states)) / num_states # Uniform transitions
)

# 2. Define the Physics (Logic)
# Utility is a linear function of feature 'x'
utility = LinearUtility(param_keys=("beta",), feature_key="x")
solver = ValueIterationSolver(discount_factor=0.95)

# 3. Solve
params = {"beta": jnp.array([1.0])}
result = solver.solve(
    params=params,
    model=model,
    utility=utility,
    dist=GumbelDistribution() # Type I Extreme Value errors
)

print(f"Converged: {result.success}")
print(f"Value Function: {result.solution}") # shape: (num_states,)
print(f"Choice Probs: {result.profile}")    # shape: (num_states, num_actions)
```