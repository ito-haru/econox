# src/econox/config.py
"""
Global configuration defaults for Econox.
"""
import jax

# Small constant for numerical stability
NUMERICAL_EPSILON: float = 1e-8

# Clipping bounds for logarithmic transformations
LOG_CLIP_MIN: float = -20.0
LOG_CLIP_MAX: float = 20.0

# JAX configuration
jax.config.update("jax_enable_x64", True)

# =============================================================================
# Result Saving Configuration
# =============================================================================

# Threshold for saving arrays inline vs. as CSV files
INLINE_ARRAY_SIZE_THRESHOLD: int = 10
"""Arrays with size <= this value will be saved inline in summary.txt and metadata.json."""

# Maximum string length in summary.txt before truncation
SUMMARY_STRING_MAX_LENGTH: int = 50
"""Maximum length for string representations in summary.txt before adding '...'."""

# CSV saving behavior
FLATTEN_MULTIDIM_ARRAYS: bool = True
"""If True, arrays with >2 dimensions will be flattened when saving to CSV."""

# Summary text formatting
SUMMARY_FIELD_WIDTH: int = 25
"""Width for field name padding in summary.txt."""

SUMMARY_SEPARATOR_LENGTH: int = 60
"""Length of separator lines (=== bars) in summary.txt."""

LOSS_PENALTY: float = float("inf")
"""
Penalty value returned for invalid model solutions during optimization.
Used to steer the optimizer away from unstable parameter regions.
Default is positive infinity.
"""