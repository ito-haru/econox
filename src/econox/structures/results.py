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
from scipy.stats import norm
import jax
import jax.numpy as jnp
import pandas as pd
import equinox as eqx
from pathlib import Path
from typing import Any, Dict, Union, Callable
from jaxtyping import Array, Float, Bool, PyTree, Scalar

from econox.config import (
    INLINE_ARRAY_SIZE_THRESHOLD,
    SUMMARY_STRING_MAX_LENGTH,
    FLATTEN_MULTIDIM_ARRAYS,
    SUMMARY_FIELD_WIDTH,
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
    def __repr__(self) -> str:
        return self.summary(short=True, print_summary=False)

    def summary(self, short: bool = False, print_summary: bool = True) -> str:
        """
        Generate a summary string of the result object.
        Args:
            short (bool, optional): If True, generate a shorter summary. Default is False.
            print_summary (bool, optional): If True, print the summary to console. Default is True.
        Returns:
            str: The summary string.
        """
        lines = []
        lines.append("=" * SUMMARY_SEPARATOR_LENGTH)
        lines.append(f"{self.__class__.__name__} Summary".center(SUMMARY_SEPARATOR_LENGTH))
        lines.append("=" * SUMMARY_SEPARATOR_LENGTH)

        # Get all field names from eqx.Module (which is a dataclass)
        field_names = []
        if dataclasses.is_dataclass(self):
            field_names = [f.name for f in dataclasses.fields(self)]
        else:
            # Fallback for non-dataclass objects
            field_names = list(vars(self).keys())
        
        if short:
            parts = [f"{self.__class__.__name__}(success={getattr(self, 'success', 'N/A')})"]
            summary_text = " ".join(parts)
        
        else:
            for field_name in field_names:
                value = getattr(self, field_name)
                display_str, _ = self._format_field_value(field_name, value)
                lines.append(f"{field_name:<{SUMMARY_FIELD_WIDTH}} : {display_str}")

            summary_text = "\n".join(lines)

        if print_summary:
            print(summary_text)

        return summary_text
        
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
        metadata = {}

        # Get all field names from eqx.Module (which is a dataclass)
        field_names = []
        if dataclasses.is_dataclass(self):
            field_names = [f.name for f in dataclasses.fields(self)]
        else:
            # Fallback for non-dataclass objects
            field_names = list(vars(self).keys())

        for field_name in field_names:
            value = getattr(self, field_name)
            _, meta_value = self._format_field_value(field_name, value)

            # Nested Result
            if isinstance(value, ResultMixin):
                value.save(base_path / field_name, overwrite=overwrite)
                metadata[field_name] = meta_value
            
            # Dictionary
            elif isinstance(value, dict) and value:
                dict_dir = base_path / field_name
                metadata[field_name] = self._save_dict_contents(value, dict_dir)
            
            # Long Arrays saved to CSV
            elif isinstance(value, jax.Array) and "data/" in str(meta_value):
                data_dir.mkdir(exist_ok=True)
                array_path = data_dir / f"{field_name}.csv"
                self._save_array_to_csv(value, array_path)
                metadata[field_name] = meta_value
            
            else:
                metadata[field_name] = meta_value
           

        # Write Summary Text
        with open(base_path / "summary.txt", "w", encoding="utf-8") as f:
            f.write(self.summary(short=False, print_summary=False))
        
        # Write Metadata JSON
        with open(base_path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
        
        logger.info(f"Results saved to: {base_path}")
        
        to_latex_method = getattr(self, "to_latex", None)
        
        if callable(to_latex_method):
            contents = to_latex_method()
            if isinstance(contents, str):
                latex_path = base_path / "result_table.tex"
                with open(latex_path, "w", encoding="utf-8") as f:
                    f.write(contents)
                logger.info(f"LaTeX table saved to: {latex_path}")

    def _format_field_value(self, name: str, value: Any) -> tuple[str, Any]:
        """
        Helper to format a field value for summary display and metadata.
        
        Args:
            name (str): The field name.
            value (Any): The field value.
        Returns:
            tuple[str, Any]: Formatted string for summary and value for metadata.
        """
        # 1. Nested Result
        if isinstance(value, ResultMixin):
            return f"[Nested result saved in ./{name}/]", f"./{name}/"
        
        # 2. Dictionary
        if isinstance(value, dict):
            if not value:
                return "{}", {}
            else:
                return f"[Saved as dir ./{name}/]", "DIR_PLACEHOLDER"
        
        # 3. JAX Array
        if isinstance(value, jax.Array):
            arr = jax.device_get(value)
            if arr.size < INLINE_ARRAY_SIZE_THRESHOLD and arr.ndim <= 1:
                arr_list = arr.tolist()
                return str(arr_list)[:SUMMARY_STRING_MAX_LENGTH], arr_list
            else:
                shape_str = str(arr.shape)
                path = f"data/{name}.csv"
                return f"[Saved as {path}] Shape={shape_str}", path
        
        # 4. Boolean
        if isinstance(value, (bool, jnp.bool_)):
            return str(bool(value)), bool(value)
        
        # 5. None
        if value is None:
            return "None", None
        
        # 6. Primitive
        val_str = str(value)
        if len(val_str) > SUMMARY_STRING_MAX_LENGTH:
            val_str = val_str[:SUMMARY_STRING_MAX_LENGTH - 3] + "..."
        
        try:
            json.dumps(value)
            meta = value
        except (TypeError, OverflowError):
            meta = str(value)
        
        return val_str, meta

    def _save_dict_contents(self, data_dict: Dict[str, Any], dict_dir: Path) -> Dict[str, Any]:
        dict_metadata = {}
        for k, v in data_dict.items():
            if isinstance(v, jax.Array):
                arr = jax.device_get(v)
                if arr.size < INLINE_ARRAY_SIZE_THRESHOLD and arr.ndim <= 1:
                    dict_metadata[k] = arr.tolist()
                else:
                    csv_name = f"{k}.csv"
                    dict_dir.mkdir(exist_ok=True)
                    self._save_array_to_csv(v, dict_dir / csv_name)
                    dict_metadata[k] = f"./{dict_dir.name}/{csv_name}"
            else:
                try:
                    json.dumps(v)
                    dict_metadata[k] = v
                except (TypeError, OverflowError):
                    dict_metadata[k] = str(v)
        return dict_metadata

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
    diagnostics: Dict[str, Any] = eqx.field(default_factory=dict)
    """Additional diagnostics about the estimation process."""
    meta: Dict[str, Any] = eqx.field(default_factory=dict, static=True)
    """Additional metadata about the estimation process (e.g., convergence criteria, iteration counts, duration)."""
    initial_params: PyTree | None = None
    """Initial parameters used for estimation, if available."""
    fixed_mask: PyTree | None = None
    """Boolean mask indicating which parameters were fixed during estimation."""

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

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the estimation results to a Pandas DataFrame for easy analysis.
        
        Returns:
            pd.DataFrame: DataFrame containing parameters, standard errors, and t-values.
        """
        def flatten_ordered(tree, parent_key=""):
            items = []
            if isinstance(tree, dict):
                for k, v in tree.items():
                    new_key = f"{parent_key}.{k}" if parent_key else k
                    items.extend(flatten_ordered(v, new_key))
            elif isinstance(tree, (list, tuple)):
                for i, v in enumerate(tree):
                    new_key = f"{parent_key}[{i}]" if parent_key else str(i)
                    items.extend(flatten_ordered(v, new_key))
            else:
                items.append((parent_key, tree))
            return items
        
        def path_to_str(path):
            return ".".join(str(p.key) if hasattr(p, 'key') else str(p) for p in path)

        ordered_items = flatten_ordered(self.params)
        names = [item[0] for item in ordered_items]
        vals = np.array([float(item[1]) for item in ordered_items], dtype=float)
        init_val = []
        if self.initial_params is not None:
            init_items = flatten_ordered(self.initial_params)
            init_dict = {item[0]: float(item[1]) for item in init_items}
            init_val = np.array([init_dict.get(name, np.nan) for name in names], dtype=float)
        else:
            init_val = np.array([np.nan] * len(names))

        if self.fixed_mask is not None:
            flat_mask_with_path, _ = jax.tree_util.tree_flatten_with_path(self.fixed_mask)
            mask_dict = {path_to_str(p): v for p, v in flat_mask_with_path}
        else:
            mask_dict = {}

        se_dict = {}
        if self.std_errors is not None:
            flat_ses_with_path, _ = jax.tree_util.tree_flatten_with_path(self.std_errors)
            se_dict = {path_to_str(p): v for p, v in flat_ses_with_path}

        ses = []
        final_names = []
        for name in names:
            is_fixed = mask_dict.get(name, False)
            if is_fixed:
                ses.append(np.nan)
                final_names.append(f"{name} (fixed)")
            else:
                ses.append(se_dict.get(name, np.nan))
                final_names.append(name)

        df = pd.DataFrame({
            "Initial": init_val,
            "Estimate": vals,
            "Std. Error": np.array(ses),
        }, index=pd.Index(final_names, name="Parameter"))

        df["t-stat"] = df["Estimate"] / df["Std. Error"]
        p_values = 2 * (1 - norm.cdf(np.abs(df["t-stat"].to_numpy())))
        df["p-value"] = p_values

        def get_stars(p, name) -> str:
            if "(fixed)" in name: return ""
            if p < 0.01: return "***"
            if p < 0.05: return "**"
            if p < 0.1:  return "*"
            return ""
        df["Sig"] = [get_stars(p, name) for p, name in zip(p_values, final_names)]

        return df
    
    def to_latex(self, split_cols: bool = True, threshold: int = 25) -> str:
        """
        Convert the estimation results to a LaTeX table.
        Returns:
            str: LaTeX table as a string.
        """
        df = self.to_dataframe()
        
        rows = []
        for name, row in df.iterrows():
            name_clean = str(name).replace("_", " ").replace(".", " ")
            coef_str = f"{row['Estimate']:.4f}{row['Sig']}"
            rows.append({"Parameter": name_clean, "Value": coef_str})
            if not np.isnan(row['Std. Error']):
                se_str = f"({row['Std. Error']:.4f})"
                rows.append({"Parameter": "", "Value": se_str})
            else:
                pass
        
        if split_cols and (len(df) > threshold):
            n_params = len(df)
            n_left = (n_params + 1) // 2
            mid = n_left * 2

            left_rows = rows[:mid]
            right_rows = rows[mid:]
            
            while len(right_rows) < len(left_rows):
                right_rows.append({"Parameter": "", "Value": ""})

            latex_df_l = pd.DataFrame(left_rows)
            latex_df_r = pd.DataFrame(right_rows)
            
            table_l = latex_df_l.to_latex(index=False, header=True, column_format="lr", escape=False)
            table_r = latex_df_r.to_latex(index=False, header=True, column_format="lr", escape=False)
            
            begin_tab = 'begin{tabular}{lr}'
            end_tab = '\\end{tabular}'
            begin_tab_escaped = '\\begin{tabular}[t]{lr}'
            quad = '\\quad'
            content = f"{begin_tab_escaped} {table_l.split(begin_tab)[1].split(end_tab)[0]} {end_tab}\n"
            content += f"{quad}\n"
            content += f"{begin_tab_escaped} {table_r.split(begin_tab)[1].split(end_tab)[0]} {end_tab}"
        else:
            latex_df = pd.DataFrame(rows)
            content = latex_df.to_latex(index=False, header=True, column_format="lr", escape=False)

        # Stats
        stats_rows = []
        stats_rows.append("\\midrule")

        if "n_obs" in self.meta:
            stats_rows.append(f"Observations & {self.meta['n_obs']} \\\\")
        
        if self.loss is not None:
            stats_rows.append(f"Loss ({self.meta.get('estimation_method', 'Value')}) & {float(self.loss):.4e} \\\\")
        
        if "r_squared" in self.diagnostics:
            r2 = self.diagnostics["r_squared"]
            stats_rows.append(f"$R^2$ & {r2:.4f} \\\\")
        
        latex_lines = content.splitlines()
        final_lines = []
        for line in latex_lines:
            if "\\bottomrule" in line:
                final_lines.extend(stats_rows)
            final_lines.append(line)

        latex_str = "\\begin{table}\n"
        latex_str += "\\centering\n" 
        latex_str += f"\\caption{{Estimation Results}}\n"
        latex_str += f"\\label{{tab:results}}\n"
        latex_str += "\n".join(final_lines)
        latex_str += "\\end{table}"
        
        return latex_str
    
    def summary(self, short: bool = False, print_summary: bool = True) -> str:
        """
        Generate a summary string of the estimation result.
        Args:
            short (bool, optional): If True, generate a shorter summary. Default is False.
            print_summary (bool, optional): If True, print the summary to console. Default is True.
        Returns:
            str: The summary string.
        """
        if short:
            return super().summary(short=True, print_summary=print_summary)
        
        df = self.to_dataframe()
        display_df = df.copy()
        if "Initial" in display_df.columns and bool(display_df["Initial"].isna().all()):
            display_df = display_df.drop(columns=["Initial"])
        loss_str = f"{self.loss:.4e}" if self.loss is not None else "N/A"
        header = f"Estimation Result Summary (Loss: {loss_str}, Success: {self.success})"
        formatters: dict[Any, Callable[[Any], str]] = {
            'Initial': lambda x: f"{x: .4f}",
            'Estimate': lambda x: f"{x: .4f}",
            'Std. Error': lambda x: f"{x: .4f}",
            't-stat': lambda x: f"{x: .2f}",
            'p-value': lambda x: f"{x: .3f}"
        }
        active_formats = {col: fmt for col, fmt in formatters.items() if col in display_df.columns}
        table_str = display_df.to_string(
            index=True, 
            justify='right', 
            formatters=active_formats
        )

        lines = [header, "-" * len(header), table_str]

        # Diagnostics
        if self.diagnostics:
            lines.append("\nDiagnostics:")
            for k, v in self.diagnostics.items():
                if isinstance(v, (float, np.floating)) or (isinstance(v, jax.Array) and v.ndim == 0):
                    val_str = f"{float(v):.4f}"
                else:
                    val_str = str(v)
                lines.append(f"  {k:<{SUMMARY_FIELD_WIDTH}}: {val_str}")
        
        if self.meta:
            lines.append("\nMetadata:")
            for k, v in self.meta.items():
                val_str = str(v)
                lines.append(f"  {k:<{SUMMARY_FIELD_WIDTH}}: {val_str}")

        summary_text = "\n".join(lines)
        if print_summary:
            print(summary_text)
        return summary_text