# src/econox/logic/terminal.py
"""
Terminal value function approximators for dynamic programming solvers.

These classes define strategies to close finite-horizon dynamic models by 
approximating the expected value function :math:`EV(T)` at the simulation 
horizon.
"""
import jax.numpy as jnp
import equinox as eqx
from typing import Union, List, Tuple
from jaxtyping import Array, Float, PyTree

from econox.protocols import StructuralModel
from econox.utils import get_from_pytree
from econox.config import NUMERICAL_EPSILON


def _retrieve_and_validate_param(
    param_keys: Union[str, List[str], Tuple[str, ...]],
    params: PyTree,
    prev_idx: tuple[int, ...],
    param_name: str
) -> Union[Float[Array, "n 1"], float]:
    """
    Retrieve and validate trend parameters from PyTree.
    
    Args:
        param_keys: Single key or list of keys to retrieve from params.
        params: Parameter PyTree.
        prev_idx: Index tuple to validate shape against.
        param_name: Name for error messages.
    
    Returns:
        Parameter value, reshaped for broadcasting if multidimensional.
    
    Raises:
        ValueError: If parameter shape doesn't match prev_idx.
    """
    if isinstance(param_keys, (list, tuple)):
        param = jnp.array([get_from_pytree(params, k, 0.0) for k in param_keys])
    else:
        param = jnp.asarray(get_from_pytree(params, param_keys, 0.0))
    
    if param.ndim > 0:
        if param.shape[0] != len(prev_idx):
            raise ValueError(
                f"{param_name} has incompatible shape {param.shape}; "
                f"expected leading dimension {len(prev_idx)} to match prev_idx."
            )
        return param[:, jnp.newaxis]
    else:
        return param


class IdentityTerminal(eqx.Module):
    r"""
    Identity terminal approximator (Zero modification).
    
    This strategy assumes the terminal value is already correctly initialized 
    and performs no modification to the input matrix.

    .. math::
        \mathbb{E}V_T(s, a) = \mathbb{E}V_{T}^{input}(s, a)

    Examples:
        >>> approximator = IdentityTerminal()
        >>> # Returns expected value matrix as-is
        >>> adjusted_ev = approximator.approximate(expected, params, model)
    """
    def approximate(
        self, 
        expected: Float[Array, "S A"], 
        params: PyTree, 
        model: StructuralModel
    ) -> Float[Array, "S A"]:
        """
        Returns the expected value matrix without any modifications.
        
        This method acts as a pass-through, preserving the original Emax values 
        computed by the Bellman operator.
        """
        return expected


class StationaryTerminal(eqx.Module):
    r"""
    Stationary terminal approximator (Steady-state Boundary).
    
    Approximates the terminal period by assuming the system has reached a 
    time-invariant steady state, where the value function at :math:`T` 
    replicates the value at :math:`T-1`.

    .. math::
        \mathbb{E}V_T(s, a) = \mathbb{E}V_{T-1}(s', a) \quad \forall s \in \mathcal{S}_{term}

    Args:
        term_idx (tuple[int, ...]): Indices of the terminal states :math:`\mathcal{S}_{term}`.
        prev_idx (tuple[int, ...]): Indices of the predecessor states :math:`\mathcal{S}_{prev}`.
    
    Examples:
        >>> # Assuming states (4, 5) are terminal and (2, 3) are T-1
        >>> approximator = StationaryTerminal(
        ...     term_idx=(4, 5),
        ...     prev_idx=(2, 3)
        ... )
        >>> adjusted_ev = approximator.approximate(expected, params, model)
    """
    term_idx: tuple[int, ...] = eqx.field(static=True)
    prev_idx: tuple[int, ...] = eqx.field(static=True)

    def approximate(
        self, 
        expected: Float[Array, "S A"], 
        params: PyTree, 
        model: StructuralModel
    ) -> Float[Array, "S A"]:
        r"""
        Overwrites terminal state values with values from predecessor states.

        This implementation performs a scatter operation where:
        
        .. math::
            EV_{adj}[term\_idx, :] = EV_{raw}[prev\_idx, :]
        """
        if len(self.term_idx) != len(self.prev_idx):
            raise ValueError(
                f"StationaryTerminal: term_idx and prev_idx must have the same shape. "
                f"Got {len(self.term_idx)} and {len(self.prev_idx)}."
            )

        # Extract values from predecessor states
        expected_at_prev = expected[self.prev_idx, :]
        # Map them to terminal states
        return expected.at[self.term_idx, :].set(expected_at_prev)


class ExponentialTrendTerminal(eqx.Module):
    r"""
    Exponential trend terminal approximator with Adaptive Branching.
    
    This approximator handles non-stationary growth at the horizon by applying 
    a growth rate :math:`\gamma_s` to the value function. 

    **Branching Priority**:
    
    1. **Exogenous (Parameter-driven)**: If `growth_rate_keys` is not None, 
       the solver uses growth rates :math:`\gamma` from `params`.
    2. **Endogenous (Data-driven)**: If `growth_rate_keys` is None and 
       `pre_prev_idx` is provided, the solver extrapolates the trend from 
       the model's internal dynamics (:math:`T-1` and :math:`T-2`).
    
    It supports growth rates through three parameter specification patterns:
    
    1. **Global Scalar**: A single key mapping to a scalar value (e.g., ``"g"``). 
       The same growth rate is applied to all terminal states.
    2. **Aggregated Scalars**: A list or tuple of keys (e.g., ``["g1", "g2", "g3"]``). 
       The number of keys must match the length of ``term_idx`` (and ``prev_idx``). 
       Each scalar parameter corresponds to a terminal state in order.
    3. **State-Indexed Vector**: A single key mapping to an array of length :math:`n` 
       (e.g., ``"g_vector"``), where :math:`n` is the length of ``term_idx``. 
       Each element corresponds to a terminal state in order.

    .. math::
        \mathbb{E}V_T(s, a) = \begin{cases} 
        (1 + \gamma_s) \mathbb{E}V_{T-1}(s', a) & \text{if keys provided (Exogenous)} \\
        \frac{\mathbb{E}V_{T-1}(s', a)}{\mathbb{E}V_{T-2}(s'', a)} \mathbb{E}V_{T-1}(s', a) & \text{if pre_prev_idx provided (Endogenous)}
        \end{cases}

    Args:
        term_idx: Indices of the terminal states :math:`T`.
        prev_idx: Indices of the predecessor states :math:`T-1`.
        pre_prev_idx: Indices of the states :math:`T-2`. Required for endogenous mode.
        growth_rate_keys: Identifier(s) for growth rate :math:`\gamma`. 
            Accepts a single ``str`` for global/vector parameters, or a ``list[str]`` 
            to aggregate multiple regional scalars.
    
    Raises:
        ValueError: If both `growth_rate_keys` and `pre_prev_idx` are None.

    Examples:
        >>> # Pattern 1: Global scalar growth
        >>> approx = ExponentialTrendTerminal(term_idx, prev_idx, growth_rate_keys="g")
        >>> params = {"g": 0.02}
        
        >>> # Pattern 2: Aggregated scalars (3 terminal states)
        >>> term_idx = (13, 14, 15)
        >>> prev_idx = (10, 11, 12)
        >>> approx = ExponentialTrendTerminal(
        ...     term_idx, prev_idx, growth_rate_keys=["g1", "g2", "g3"]
        ... )
        >>> params = {"g1": 0.02, "g2": 0.03, "g3": 0.01}
        
        >>> # Pattern 3: Endogenous dynamic extrapolation (No params needed)
        >>> approx = ExponentialTrendTerminal(term_idx, prev_idx, pre_prev_idx=pre_prev)
        >>> adjusted_ev = approx.approximate(expected, {}, model)
    """
    term_idx: tuple[int, ...] = eqx.field(static=True)
    prev_idx: tuple[int, ...] = eqx.field(static=True)
    pre_prev_idx: tuple[int, ...] | None = eqx.field(static=True, default=None)
    growth_rate_keys: Union[str, List[str], Tuple[str, ...]] | None = None

    def approximate(
        self, 
        expected: Float[Array, "S A"], 
        params: PyTree, 
        model: StructuralModel
    ) -> Float[Array, "S A"]:
        r"""
        Applies exponential growth to the terminal horizon.

        The method adaptively switches between:
        
        * **Exogenous Growth**: Multiplying :math:`T-1` values by :math:`(1 + \gamma)` from `params`.
        * **Endogenous Extrapolation**: Multiplying :math:`T-1` values by the ratio :math:`EV_{T-1} / EV_{T-2}`.
        
        It automatically handles spatial heterogeneity by mapping parameter keys 
        or vector elements to the corresponding state indices.
        """
        if len(self.term_idx) != len(self.prev_idx):
            raise ValueError(
                f"ExponentialTrendTerminal: term_idx and prev_idx must have the same shape. "
                f"Got {len(self.term_idx)} and {len(self.prev_idx)}."
            )
        val_t_minus_1 = expected[self.prev_idx, :]

        # Case 1: Growth rates provided
        if self.growth_rate_keys is not None:
            gamma_eff = _retrieve_and_validate_param(
                self.growth_rate_keys, params, self.prev_idx, "ExponentialTrendTerminal: gamma"
            )
            updated_val = val_t_minus_1 * (1.0 + gamma_eff)

        # Case 2: Pre-previous indices provided    
        elif self.pre_prev_idx is not None:
            if len(self.pre_prev_idx) != len(self.prev_idx):
                raise ValueError(
                    f"ExponentialTrendTerminal: pre_prev_idx and prev_idx must have the same shape. "
                    f"Got {len(self.pre_prev_idx)} and {len(self.prev_idx)}."
                )
            val_t_minus_2 = expected[self.pre_prev_idx, :]
            # Use jnp.where for numerical stability: fallback to stationary (ratio=1.0) when denominator is near zero
            # Note: jnp.where evaluates both branches, so we must avoid division by zero in the computation itself
            denominator = jnp.abs(val_t_minus_2)
            safe_denom = jnp.where(denominator > NUMERICAL_EPSILON, val_t_minus_2, 1.0)
            ratio = jnp.where(
                denominator > NUMERICAL_EPSILON,
                val_t_minus_1 / safe_denom,
                1.0
            )
            updated_val = val_t_minus_1 * ratio
            
        else:
            raise ValueError(
                "ExponentialTrendTerminal requires either 'growth_rate_keys' or 'pre_prev_idx' to be set."
            )

        return expected.at[self.term_idx, :].set(updated_val)


class LinearTrendTerminal(eqx.Module):
    r"""
    Linear trend terminal approximator with Adaptive Branching.
    
    Approximates the terminal value by adding a drift component :math:`\delta_s`. 

    **Branching Priority**:
    
    1. **Exogenous (Parameter-driven)**: If `drift_keys` is not None, 
       uses drift terms :math:`\delta` from `params`.
    2. **Endogenous (Data-driven)**: If `drift_keys` is None and 
       `pre_prev_idx` is provided, extrapolates the linear difference 
       between :math:`T-1` and :math:`T-2`.

    Similar to the exponential variant, it supports three patterns for :math:`\delta`:
    
    1. **Global Drift**: A single key mapping to a scalar drift value applied to all terminal states.
    2. **Aggregated Drifts**: A list or tuple of keys. The number of keys must match 
       the length of ``term_idx`` (and ``prev_idx``). Each scalar corresponds to a terminal state in order.
    3. **Drift Vector**: A single key mapping to an array of length :math:`n`, 
       where :math:`n` is the length of ``term_idx``. Each element corresponds to a terminal state in order.

    .. math::
        \mathbb{E}V_T(s, a) = \begin{cases} 
        \mathbb{E}V_{T-1}(s', a) + \delta_s & \text{if keys provided (Exogenous)} \\
        \mathbb{E}V_{T-1}(s', a) + (\mathbb{E}V_{T-1}(s', a) - \mathbb{E}V_{T-2}(s'', a)) & \text{if pre_prev_idx provided (Endogenous)}
        \end{cases}

    Args:
        term_idx: Indices of the terminal states :math:`T`.
        prev_idx: Indices of the predecessor states :math:`T-1`.
        pre_prev_idx: Indices of the states :math:`T-2`. Required for endogenous mode.
        drift_keys: Identifier(s) for drift :math:`\delta`. Accepts a single ``str`` 
            for global/vector parameters, or a ``list[str]`` to aggregate 
            multiple regional scalars.

    Raises:
        ValueError: If both `drift_keys` and `pre_prev_idx` are None.

    Examples:
        >>> # Linear drift via parameter keys
        >>> approx = LinearTrendTerminal(term_idx, prev_idx, drift_keys="drift")
        >>> params = {"drift": 500.0}
        >>> adjusted_ev = approx.approximate(expected, params, model)
    """
    term_idx: tuple[int, ...] = eqx.field(static=True)
    prev_idx: tuple[int, ...] = eqx.field(static=True)
    pre_prev_idx: tuple[int, ...] | None = eqx.field(static=True, default=None)
    drift_keys: Union[str, List[str], Tuple[str, ...]] | None = None

    def approximate(
        self, 
        expected: Float[Array, "S A"], 
        params: PyTree, 
        model: StructuralModel
    ) -> Float[Array, "S A"]:
        r"""
        Applies linear drift to the terminal horizon.

        The method adaptively switches between:
        
        * **Exogenous Drift**: Adding :math:`\delta` from `params` to :math:`T-1` values.
        * **Endogenous Extrapolation**: Adding the difference :math:`(EV_{T-1} - EV_{T-2})` to :math:`EV_{T-1}`.
        """
        val_t_minus_1 = expected[self.prev_idx, :]

        # Case 1: Drift keys provided
        if self.drift_keys is not None:
            delta_effective = _retrieve_and_validate_param(
                self.drift_keys, params, self.prev_idx, "LinearTrendTerminal: delta"
            )
            updated_val = val_t_minus_1 + delta_effective
        
        # Case 2: Pre-previous indices provided
        elif self.pre_prev_idx is not None:
            if len(self.pre_prev_idx) != len(self.prev_idx):
                raise ValueError(
                    f"LinearTrendTerminal: pre_prev_idx and prev_idx must have the same shape. "
                    f"Got {len(self.pre_prev_idx)} and {len(self.prev_idx)}."
                )
            val_t_minus_2 = expected[self.pre_prev_idx, :]
            diff = val_t_minus_1 - val_t_minus_2
            updated_val = val_t_minus_1 + diff
        
        else:
            raise ValueError(
                "LinearTrendTerminal requires either 'drift_keys' or 'pre_prev_idx' to be set."
            )
        
        return expected.at[self.term_idx, :].set(updated_val)
