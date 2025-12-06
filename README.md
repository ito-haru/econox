# Econox

> [!WARNING]
> **Early Development Status**
> 
> Econox is currently under active development. The API is unstable and subject to **breaking changes** at any time.  
> Use with caution in production environments.
> (本ライブラリは現在開発中です。APIは頻繁に変更される可能性があります。)

**A JAX-based toolkit for structural modeling and estimation.**

Econox is a Python library built on [JAX](https://github.com/google/jax) and [Equinox](https://github.com/patrick-kidger/equinox) designed to streamline the research workflow for structural economists.

While originally motivated by dynamic discrete choice models, Econox provides a modular, object-oriented framework suitable for a wide range of economic models.

### Key Features
* **Modeling:** Define models as clean, composable Python classes using Equinox.
* **Estimation:** Perform gradient-based estimation (MLE, GMM, etc.) efficiently with JAX's automatic differentiation and JIT compilation.
* **Simulation (Coming Soon):** Forward-simulation capabilities are currently under active development.

### Installation

```bash
pip install econox