# src/econox/structures/results.py
"""
Data structures for holding computation results.
Uses Equinox modules to allow mixins and PyTree registration.
"""

from __future__ import annotations
import logging
import json
import dataclasses
import shutil
import numpy as np
import jax
import jax.numpy as jnp
import pandas as pd
import equinox as eqx
from pathlib import Path
from typing import Any, Dict, Union
from jaxtyping import Array, Float, Bool, PyTree, Scalar

from econox.config import (
    INLINE_ARRAY_SIZE_THRESHOLD,
    SUMMARY_STRING_MAX_LENGTH,
    FLATTEN_MULTIDIM_ARRAYS,
    SUMMARY_FIELD_WIDTH,
    SUMMARY_DICT_KEY_WIDTH,
    SUMMARY_SEPARATOR_LENGTH,
)

logger = logging.getLogger(__name__)

# =============================================================================
# 1. Save Logic (Mixin)
# =============================================================================

class ResultMixin:
    """
    Provides a generic `save()` method for Result objects.
    Implements the 'Directory Bundle' strategy.
    """
    def save(self, path: Union[str, Path], overwrite: bool = False) -> None:
        """
        Save the result object to a directory using the 'Directory Bundle' strategy.  

        Args: 
            path (Union[str, Path]): The target directory path where the result will be saved.  
            overwrite (bool, optional): If True, overwrite the directory if it already exists. Default is False.   

        Raises: 
            FileExistsError: If the target directory already exists and `overwrite` is False.  
        """
        base_path = Path(path)
        if base_path.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Directory '{base_path}' already exists. "
                    f"Use overwrite=True to replace."
                )
            else:
                shutil.rmtree(base_path)
        
        base_path.mkdir(parents=True, exist_ok=True)
        data_dir = base_path / "data"
        data_dir.mkdir(exist_ok=True)

        summary_lines = []
        metadata = {}
        
        # Header
        class_name = self.__class__.__name__
        summary_lines.append("=" * SUMMARY_SEPARATOR_LENGTH)
        summary_lines.append(f"Result Report: {class_name}")
        summary_lines.append("=" * SUMMARY_SEPARATOR_LENGTH + "\n")

        # Get all field names from eqx.Module (which is a dataclass)
        field_names = []
        if dataclasses.is_dataclass(self):
            field_names = [f.name for f in dataclasses.fields(self)]
        else:
            # Fallback for non-dataclass objects
            field_names = list(vars(self).keys())

        for field_name in field_names:
            value = getattr(self, field_name)
            
            # 1. Nested Result (Recursive Save)
            if hasattr(value, 'save') and isinstance(value, ResultMixin):
                sub_path = base_path / field_name
                value.save(sub_path, overwrite=overwrite)
                summary_lines.append(f"{field_name:<{SUMMARY_FIELD_WIDTH}}: [Nested result saved in ./{field_name}/]")
                metadata[field_name] = f"./{field_name}/"
                continue
            
            # 2. Dictionary Fields (aux, meta)
            if isinstance(value, dict):
                if not value:
                    summary_lines.append(f"{field_name:<{SUMMARY_FIELD_WIDTH}}: {{}}")
                    metadata[field_name] = {}
                else:
                    dict_dir = base_path / field_name
                    dict_dir.mkdir(exist_ok=True)

                    summary_lines.append(f"{field_name:<{SUMMARY_FIELD_WIDTH}}: [Saved as dir ./{field_name}/]")
                    dict_metadata = {}

                    for k, v in value.items():
                        # Handle nested arrays in dictionaries
                        if isinstance(v, jax.Array):
                            arr = jax.device_get(v)
                            if arr.size < INLINE_ARRAY_SIZE_THRESHOLD and arr.ndim <= 1:
                                arr_list = arr.tolist()
                                summary_lines.append(f"  - {k:<{SUMMARY_DICT_KEY_WIDTH}}: {str(arr_list)[:SUMMARY_STRING_MAX_LENGTH]}")
                                dict_metadata[k] = arr_list
                            else:
                                csv_name = f"{k}.csv"
                                csv_path = dict_dir / csv_name
                                self._save_array_to_csv(arr, csv_path)
                                shape_str = str(arr.shape)
                                summary_lines.append(f"  - {k:<{SUMMARY_DICT_KEY_WIDTH}}: [./{field_name}/{csv_name}] Shape={shape_str}")
                                dict_metadata[k] = f"./{field_name}/{csv_name}"
                        else:
                            # Primitive values
                            val_str = str(v)
                            if len(val_str) > SUMMARY_STRING_MAX_LENGTH:
                                val_str = val_str[:SUMMARY_STRING_MAX_LENGTH - 3] + "..."
                            summary_lines.append(f"  - {k:<{SUMMARY_DICT_KEY_WIDTH}}: {val_str}")
                            try:
                                json.dumps(v)
                                dict_metadata[k] = v
                            except (TypeError, OverflowError):
                                dict_metadata[k] = str(v)

                    metadata[field_name] = dict_metadata
                continue
            
            # 3. JAX Arrays
            if isinstance(value, jax.Array):
                arr = jax.device_get(value)
                
                # Scalar or Small Array -> Save in Text/JSON
                if arr.size < INLINE_ARRAY_SIZE_THRESHOLD and arr.ndim <= 1:
                    arr_list = arr.tolist()
                    val_str = str(arr_list)
                    summary_lines.append(f"{field_name:<{SUMMARY_FIELD_WIDTH}}: {val_str}")
                    metadata[field_name] = arr_list
                
                # Large Array -> Save as CSV
                else:
                    csv_name = f"{field_name}.csv"
                    csv_path = data_dir / csv_name
                    self._save_array_to_csv(arr, csv_path)
                    
                    shape_str = str(arr.shape)
                    summary_lines.append(f"{field_name:<{SUMMARY_FIELD_WIDTH}}: [Saved as data/{csv_name}] Shape={shape_str}")
                    metadata[field_name] = f"data/{csv_name}"

            # 4. Boolean values
            elif isinstance(value, (bool, jnp.bool_)):
                bool_val = bool(value)
                summary_lines.append(f"{field_name:<{SUMMARY_FIELD_WIDTH}}: {bool_val}")
                metadata[field_name] = bool_val

            # 5. None or Primitives
            elif value is None:
                summary_lines.append(f"{field_name:<{SUMMARY_FIELD_WIDTH}}: None")
                metadata[field_name] = None
            
            else:
                # Python primitives (int, float, str)
                val_str = str(value)
                if len(val_str) > SUMMARY_STRING_MAX_LENGTH:
                    val_str = val_str[:SUMMARY_STRING_MAX_LENGTH - 3] + "..."
                summary_lines.append(f"{field_name:<{SUMMARY_FIELD_WIDTH}}: {val_str}")
                
                # Try to add to metadata if JSON serializable
                try:
                    json.dumps(value)
                    metadata[field_name] = value
                except (TypeError, OverflowError):
                    metadata[field_name] = str(value)

        # Write Summary Text
        with open(base_path / "summary.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(summary_lines))
        
        # Write Metadata JSON
        with open(base_path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

        logger.info(f"Results saved to: {base_path}")

    def _save_array_to_csv(self, arr, path: Path) -> None:
        """
        Helper to save arrays to CSV using Pandas.
        
        Args:
            arr (jax.Array or numpy.ndarray): The array to save. Must be a JAX array or NumPy array.
            path (Path): The file path where the CSV will be saved.
            
        Raises:
            TypeError: If arr is not a JAX array or NumPy array.
            ValueError: If arr is empty or has invalid dimensions.
        """
        # Validate input type
        if not isinstance(arr, (jax.Array, jnp.ndarray, np.ndarray)):
            raise TypeError(
                f"Expected JAX array or NumPy array, got {type(arr).__name__}"
            )
        
        # Validate array is not empty
        if arr.size == 0:
            raise ValueError("Cannot save empty array to CSV")
        
        # Handle multi-dimensional arrays
        if FLATTEN_MULTIDIM_ARRAYS and hasattr(arr, "ndim") and arr.ndim > 2:
            # Flatten >2D arrays for CSV (e.g. T x S x A -> T*S rows)  
            flattened = arr.reshape(arr.shape[0], -1)
            pd.DataFrame(flattened).to_csv(path, index=False)
        else:
            pd.DataFrame(arr).to_csv(path, index=False)


# =============================================================================
# 2. Concrete Result Classes (Using eqx.Module)
# =============================================================================

class SolverResult(ResultMixin, eqx.Module):
    """
    Container for the output of a Solver (Inner/Outer Loop).
    """

    solution: PyTree
    r"""
    Main solution returned by the solver (the fixed point).

    - **DP**: Expected Value Function :math:`EV(s)` (Integrated Value Function / Emax).
      Represents the expected value *before* the realization of the shock :math:`\epsilon`.
    - **GE**: Equilibrium allocations (e.g., Population Distribution :math:`D`) or Prices :math:`P`.
    """

    profile: PyTree | None = None
    """
    Associated profile information derived from the solution.

    - **DP**: Conditional Choice Probabilities (CCP) :math:`P(a|s)`.
      The probability of choosing action :math:`a` given state :math:`s`.
    - **GE**: Market prices (Wage, Rent) or aggregate states corresponding to the solution.
    """

    inner_result: SolverResult | None = None
    """
    Associated inner solver result used during nested solving.
    """

    success: Bool[Array, ""] | bool = False
    """Whether the solver converged successfully."""
    aux: Dict[str, Any] = eqx.field(default_factory=dict)
    """Additional auxiliary information (e.g., diagnostics)."""

class EstimationResult(ResultMixin, eqx.Module):
    """
    Container for the output of an Estimator.
    """
    
    params: PyTree
    """Estimated parameters."""
    loss: Scalar | float
    """Final value of the loss function (e.g., negative log-likelihood)."""
    success: Bool[Array, ""] | bool = False
    """Whether the estimation converged successfully."""
    solver_result: SolverResult | None = None
    """Associated (outermost) solver result used during estimation."""
    
    std_errors: PyTree | None = None
    """Standard errors of the estimated parameters, if available."""
    vcov: Float[Array, "n_params n_params"] | None = None
    """Variance-covariance matrix of the estimated parameters, if available."""
    r_squared: Scalar | None = None
    """R-squared statistic for goodness-of-fit, if available."""
    meta: Dict[str, Any] = eqx.field(default_factory=dict, static=True)
    """Additional metadata about the estimation process (e.g., convergence criteria, iteration counts, duration)."""

    @property
    def t_values(self) -> PyTree | None:
        """Compute t-values if standard errors are available."""
        if self.std_errors is None:
            return None
        
        return jax.tree_util.tree_map(
            lambda p, se: jnp.where(se != 0, p / se, jnp.nan),
            self.params,
            self.std_errors
        )