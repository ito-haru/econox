# src/econox/structures/results.py
"""
Data structures for holding computation results.
Uses Equinox modules to allow mixins and PyTree registration.
"""

import json
import dataclasses
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import equinox as eqx
from pathlib import Path
from typing import Any, Dict, Union
from jaxtyping import Array, Float, Bool, PyTree

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
        Save the result object to a directory.
        """
        base_path = Path(path)
        if base_path.exists() and not overwrite:
            raise FileExistsError(f"Directory '{base_path}' already exists. Use overwrite=True to replace.")
        
        base_path.mkdir(parents=True, exist_ok=True)
        data_dir = base_path / "data"
        data_dir.mkdir(exist_ok=True)

        summary_lines = []
        metadata = {}
        
        # Header
        class_name = self.__class__.__name__
        summary_lines.append("=" * 60)
        summary_lines.append(f"Result Report: {class_name}")
        summary_lines.append("=" * 60 + "\n")

        # Iterate over fields
        # eqx.Module uses dataclass fields, accessible via __dataclass_fields__ or just iterate vars if dynamic
        # Ideally, we iterate over the fields defined in the class.
        # For eqx.Module, we can use vars(self) or serialization approaches, 
        # but explicit field access is safer.
        
        # Get all field names
        if dataclasses.is_dataclass(self):
            field_names = [f.name for f in dataclasses.fields(self)]
        else:
            # Fallback for non-dataclass objects (vars() or __dict__)
            field_names = list(getattr(self, "__dict__", {}).keys())

        for field_name in field_names:
            value = getattr(self, field_name)
            
            # 1. Nested Result (Recursive Save)
            if hasattr(value, 'save') and isinstance(value, ResultMixin):
                sub_path = base_path / field_name
                value.save(sub_path, overwrite=overwrite)
                summary_lines.append(f"{field_name:<25}: [Saved in ./{field_name}/]")
                continue
            
            # 2. JAX/Numpy Arrays
            if isinstance(value, (jnp.ndarray, np.ndarray)):
                # Convert to numpy for saving
                arr = np.array(value)
                
                # Scalar or Small Array -> Save in Text/JSON
                if arr.size < 10 and arr.ndim <= 1:
                    arr_list = arr.tolist()
                    val_str = str(arr_list)
                    summary_lines.append(f"{field_name:<25}: {val_str}")
                    metadata[field_name] = arr_list
                
                # Large Array -> Save as CSV
                else:
                    csv_name = f"{field_name}.csv"
                    csv_path = data_dir / csv_name
                    self._save_array_to_csv(arr, csv_path)
                    
                    shape_str = str(arr.shape)
                    summary_lines.append(f"{field_name:<25}: [Saved as data/{csv_name}] Shape={shape_str}")
                    metadata[field_name] = f"data/{csv_name}"

            # 3. None or Primitives
            elif value is None:
                summary_lines.append(f"{field_name:<25}: None")
                metadata[field_name] = None
            
            else:
                # Python primitives (int, float, str, dict)
                val_str = str(value)
                if len(val_str) > 50:
                    val_str = val_str[:47] + "..."
                summary_lines.append(f"{field_name:<25}: {val_str}")
                
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

        print(f"✅ Results saved to: {base_path}")

    def _save_array_to_csv(self, arr: np.ndarray, path: Path) -> None:
        """Helper to save arrays to CSV using Pandas."""
        if arr.ndim <= 2:
            pd.DataFrame(arr).to_csv(path, index=False)
        else:
            # Flatten >2D arrays for CSV
            flattened = arr.reshape(arr.shape[0], -1)
            pd.DataFrame(flattened).to_csv(path, index=False)


# =============================================================================
# 2. Concrete Result Classes (Using eqx.Module)
# =============================================================================

class SolverResult(ResultMixin, eqx.Module):
    """
    Container for the output of a Solver (Inner/Outer Loop).
    """
    values: PyTree
    policy: PyTree
    success: Bool[Array, ""] | bool
    aux: Dict[str, Any] = eqx.field(default_factory=dict)
    
    # Market clearing / GE results
    equilibrium: PyTree | None = None


class EstimationResult(ResultMixin, eqx.Module):
    """
    Container for the output of an Estimator.
    """
    params: PyTree
    loss: float
    success: Bool[Array, ""] | bool
    
    std_errors: PyTree | None = None
    vcov: Float[Array, "n_params n_params"] | None = None
    meta: Dict[str, Any] = eqx.field(default_factory=dict)

    # 推定されたパラメータでの均衡状態
    equilibrium: SolverResult | None = None

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