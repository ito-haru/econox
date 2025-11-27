# src/econox/config.py
"""
Global configuration defaults for Econox.
"""
import jax

# Small constant for numerical stability
NUMERICAL_EPSILON: float = 1e-8

# Clipping bounds for log-linear feedback updates
LOG_CLIP_MIN: float = -20.0
LOG_CLIP_MAX: float = 20.0

# JAX configuration
jax.config.update("jax_enable_x64", True)