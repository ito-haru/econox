
<div align="center">

# Econox

### Build structural models like Lego blocks.

[![PyPI version](https://img.shields.io/pypi/v/econox.svg)](https://pypi.org/project/econox/)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![Documentation](https://img.shields.io/badge/docs-online-blue)](https://ito-haru.github.io/econox/)
[![Status](https://img.shields.io/badge/status-alpha-orange)](https://github.com/ito-haru/econox)
[![License](https://img.shields.io/badge/license-Apache_2.0-green)](./LICENSE)
</div>

> [!WARNING]
> **Status: Alpha Release**
> 
> Econox is currently in the **alpha stage**. We are actively expanding the library of solvers and structural components.
>

Econox is a **JAX-based toolkit for structural modeling and estimation**. 
It leverages [Equinox](https://github.com/patrick-kidger/equinox) to provide a highly modular, object-oriented framework where researchers can assemble economic models from reusable components.

## 🧩 Core Philosophy: Composable Design

In Econox, a structural model is not a black box. It is a composition of independent, interchangeable blocks.

$$\text{Model} + \text{Utility} + \text{Solver} \xrightarrow{\text{Estimator}} \text{Results}$$

* **Define Physics:** Swap `LinearUtility` with **your own custom utility classes** instantly.
* **Define Solver:** Switch between `ValueIteration` and `Equilibrium` without rewriting the loop.
* **Estimate:** Feed the assembled parts to `Estimator` to find parameters that match the data.

---

### Key Features

* **🐍 Pure Python & Easy Setup:**
    * **No Fortran/C++ required:** Unlike legacy libraries that rely on complex compilation chains, Econox is **Pure Python**.
    * **Hackable:** Since the logic is written in standard Python (via Equinox), it is readable, debuggable, and easy to extend with custom logic.
    * **JIT Compilation:** While you write Python, JAX compiles it to XLA (machine code) at runtime, offering C++ level performance.

* **🎯 Precision via Auto-Diff:**
    * **Exact Derivatives:** Econox uses **Automatic Differentiation** to compute gradients and Hessians.
    * **Better Inference:** This eliminates approximation errors associated with numerical differentiation, leading to **more accurate standard errors and t-statistics** for structural parameters.

* **🧩 Modular Modeling:**
    * Define models as clean, composable classes. Swap utility functions or solvers instantly without rewriting the estimation loop.

* **🚀 Hardware Accelerated:**
    * Seamlessly scale to **GPUs or TPUs** to accelerate large-scale state spaces ($S>10,000$) simply by changing the JAX backend.

## ⚡ Validation & Performance

We demonstrate both the **accuracy** and **scalability** of Econox in a single unified benchmark.

### Benchmark & Replication Report
The following notebook contains two key validations:
1.  **Accuracy (Rust 1987 Replication):** Successfully replicates the structural parameters $(\theta, RC)$ from John Rust's classic paper using the original dataset.
2.  **Scalability (Large-Scale Benchmark):** Measures estimation speed on massive synthetic problems ($S=3000, P=50$) to demonstrate hardware acceleration.

[![Open Benchmark Report In Colab](https://img.shields.io/badge/Colab-Run_Benchmark-F9AB00?logo=googlecolab)](https://colab.research.google.com/github/ito-haru/econox/blob/main/examples/0_benchmark_rust.ipynb)

### Scalability Results (Large-Scale DDCM)
*Synthetic data estimation based on the structure of Rust (1987) fixed-point model.*
*(State Space $S=3000$, Parameters $P=50$, 100 Estimation Steps)*

| Implementation | Hardware | Est. Total Time | Speedup |
| :--- | :--- | :--- | :--- |
| NumPy (Numerical Diff) | CPU | ~2.0 days | 1x (Baseline) |
| **Econox (Auto-Diff)** | **CPU** | **~1.1 hours** | **~44x** |
| **Econox (Auto-Diff)** | **GPU (T4)** | **~19 min** | **~157x** |

> *Note: Times for Econox include JIT compilation overhead (Warmup). NumPy estimates are extrapolated from single-step performance using an honest solver (Utility $\to$ Bellman $\to$ Likelihood).* Benchmark conducted on Google Colab (vCPU: Intel Xeon, GPU: NVIDIA T4).

## Quick Start: Structural Estimation

Here is how you can build and estimate a Dynamic Discrete Choice Model (Rust 1987 style) by assembling reusable components.

[![Open Tutorial In Colab](https://img.shields.io/badge/Colab-Run_Quick_Start-F9AB00?logo=googlecolab)](https://colab.research.google.com/github/ito-haru/econox/blob/main/examples/1_quickstart.ipynb)

```python
import jax.numpy as jnp
import econox as ecx

# 1. [Environment] Define Data & Transitions
num_states, num_actions = 10, 3
model = ecx.Model.from_data(
    num_states=num_states,
    num_actions=num_actions,
    data={"x": jnp.zeros((10, 3, 1))}, # Dummy feature
    transitions=jnp.ones((30, 10)) / 10
)

# 2. [Physics] Define Utility Function
utility = ecx.LinearUtility(param_keys=("beta",), feature_key="x")

# 3. [Algorithm] Define Solver
solver = ecx.ValueIterationSolver(
    utility=utility,
    dist=ecx.GumbelDistribution(), # Logit
    discount_factor=0.95
)

# 4. [Interface] Assemble the Estimator
param_space = ecx.ParameterSpace.create(initial_params={"beta": jnp.array([0.0])})

estimator = ecx.Estimator(
    model=model,
    param_space=param_space,
    method=ecx.MaximumLikelihood(),  # Use MLE
    solver=solver
)

# 5. Estimate!
observations = {
    "state_indices": jnp.array([0, 1, 2]), 
    "choice_indices": jnp.array([0, 2, 1])
}
result = estimator.fit(observations)

print(f"Success: {result.success}")
print(f"Estimated Params: {result.params}")
```

### Installation

Requires Python 3.11+ and JAX.

**Using pip:**
```bash
pip install econox
```
**Using uv (Recommended):**
```bash
uv add econox
```

### Requirements

Econox is built upon the modern JAX ecosystem. The core dependencies include:

* **[JAX](https://github.com/jax-ml/jax):** For high-performance array computing and automatic differentiation.
* **[Equinox](https://github.com/patrick-kidger/equinox):** For defining parameterized models.
* **[Optimistix](https://github.com/patrick-kidger/optimistix):** For nonlinear optimization and root-finding.
* **[Jaxtyping](https://github.com/patrick-kidger/jaxtyping):** For type annotations and shape checking.

## 📚 Documentation

For detailed API references and tutorials, please visit the **[Official Documentation](https://ito-haru.github.io/econox/)**.

## ℹ️ Project Information

### 📄 Citation
If you use **Econox** in your research, please cite the software artifact as follows until the official paper is published:

```bibtex
@software{econox,
  author = {Ito, Haruto},
  title = {Econox: A JAX-based toolkit for structural modeling and estimation},
  url = {[https://github.com/ito-haru/econox](https://github.com/ito-haru/econox)},
  version = {0.1.3},
  year = {2026},
  note = {Alpha Release}
}
```

### 👥 Credits & Acknowledgements
#### Author

* **[Haruto Ito (伊藤晴人)](https://github.com/ito-haru)** - *Lead Developer, Department of Civil Engineering, The University of Tokyo*

#### Acknowledgements

This library originates from an undergraduate thesis conducted at the **[Lab for Innovative Infrastructure Systems (LIIS)](https://liis.t.u-tokyo.ac.jp/)**, The University of Tokyo.

We are deeply grateful to **[Prof. Daisuke Fukuda (福田 大輔)](https://researchmap.jp/fukudai)** and **[Asst. Prof. Keishi Fujiwara (藤原啓示)](https://researchmap.jp/fjwkc25)** for their valuable academic guidance and theoretical advice regarding the structural estimation frameworks used in this library.

> **Note on Maintenance:** This is an independent open-source project maintained by Haruto Ito. Please adhere to the license terms regarding liability.