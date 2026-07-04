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
from econox.config import STABILITY_MARGIN


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
        >>> adjusted_ev = approximator.approximate(expected, params, model, discount_factor)
    """
    def approximate(
        self, 
        expected: Float[Array, "S A"], 
        params: PyTree, 
        model: StructuralModel,
        discount_factor: float
    ) -> tuple[Float[Array, "S A"], Array | None]:
        """
        Returns the expected value matrix without any modifications.
        
        This method acts as a pass-through, preserving the original Emax values 
        computed by the Bellman operator.
        """
        return expected, None


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
        >>> adjusted_ev = approximator.approximate(expected, params, model, discount_factor)
    """
    term_idx: tuple[int, ...] = eqx.field(static=True)
    prev_idx: tuple[int, ...] = eqx.field(static=True)

    def approximate(
        self, 
        expected: Float[Array, "S A"], 
        params: PyTree, 
        model: StructuralModel,
        discount_factor: float
    ) -> tuple[Float[Array, "S A"], Array | None]:
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
        return expected.at[self.term_idx, :].set(expected_at_prev), None


class ExponentialTrendTerminal(eqx.Module):
    r"""
    Exponential trend terminal approximator.

    This approximator handles non-stationary growth at the horizon by applying
    an exogenous growth rate :math:`\gamma_s` (from `params`) to the value function.

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
        \mathbb{E}V_T(s, a) = (1 + \gamma_s) \mathbb{E}V_{T-1}(s', a)

    **Stability Mechanism (Soft-Clipping)**:

    To prevent the value function from diverging (which occurs if growth rate :math:`\ge 1/\beta`),
    this class applies a differentiable **Tanh Soft-Clipping** mechanism.
    The effective growth rate :math:`\gamma_{safe}` is constrained as:

    .. math::
        \gamma_{limit} &= 1/\beta - 1 - \epsilon \\
        \gamma_{raw} &= \text{input\_ratio} - 1 \\
        \gamma_{safe} &= \gamma_{limit} \cdot \tanh\left(\frac{\gamma_{raw}}{\gamma_{limit}}\right)

    This ensures that the growth rate smoothly approaches the theoretical limit
    without hitting a hard boundary, preserving gradients for the optimizer.

    Args:
        term_idx: Indices of the terminal states :math:`T`.
        prev_idx: Indices of the predecessor states :math:`T-1`.
        growth_rate_keys: Identifier(s) for growth rate :math:`\gamma`.
            Accepts a single ``str`` for global/vector parameters, or a ``list[str]``
            to aggregate multiple regional scalars.
        scale: Scaling factor for numerical stability. Input parameters are divided
            by this value (e.g., set to 100.0 if parameters are estimated in percentage).
            This helps the optimizer by keeping gradients in a manageable range.

    Raises:
        ValueError: If `growth_rate_keys` is None.

    Examples:
        >>> # Pattern 1: Global scalar growth
        >>> approx = ExponentialTrendTerminal(term_idx, prev_idx, growth_rate_keys="g", scale=100.0)
        >>> params = {"g": 2.0}  # 2% growth

        >>> # Pattern 2: Aggregated scalars (3 terminal states)
        >>> term_idx = (13, 14, 15)
        >>> prev_idx = (10, 11, 12)
        >>> approx = ExponentialTrendTerminal(
        ...     term_idx, prev_idx, growth_rate_keys=["g1", "g2", "g3"], scale=100.0
        ... )
        >>> params = {"g1": 2.0, "g2": 3.0, "g3": 1.0}

    Note:
        The ``is_clipped`` flag in the results returns ``True`` if the raw growth rate
        exceeded the theoretical stability limit :math:`1/\beta`. In this case,
        the actual used growth rate was compressed by the Tanh function.
    """
    term_idx: tuple[int, ...] = eqx.field(static=True)
    prev_idx: tuple[int, ...] = eqx.field(static=True)
    growth_rate_keys: Union[str, List[str], Tuple[str, ...]] | None = eqx.field(static=True, default=None)
    scale: float = eqx.field(static=True, default=1.0)

    def __check_init__(self) -> None:
        if len(self.term_idx) != len(self.prev_idx):
            raise ValueError(
                f"ExponentialTrendTerminal: term_idx and prev_idx must have the same shape. "
                f"Got {len(self.term_idx)} and {len(self.prev_idx)}."
            )
        if self.growth_rate_keys is None:
            raise ValueError(
                "ExponentialTrendTerminal requires 'growth_rate_keys' to be set."
            )

    def approximate(
        self,
        expected: Float[Array, "S A"],
        params: PyTree,
        model: StructuralModel,
        discount_factor: float
    ) -> tuple[Float[Array, "S A"], Array | None]:
        r"""
        Applies exponential growth to the terminal horizon.

        Multiplies :math:`T-1` values by :math:`(1 + \gamma)`, with :math:`\gamma`
        retrieved from `params`. Automatically handles spatial heterogeneity by
        mapping parameter keys or vector elements to the corresponding state indices.
        """
        val_t_minus_1 = expected[self.prev_idx, :]

        gamma_eff = _retrieve_and_validate_param(
            self.growth_rate_keys, params, self.prev_idx, "ExponentialTrendTerminal: gamma"
        )
        ratio = (1.0 + gamma_eff / self.scale)

        limit_growth = 1.0 / discount_factor - STABILITY_MARGIN - 1
        raw_growth = ratio - 1.0
        safe_growth = limit_growth * jnp.tanh(raw_growth / limit_growth)
        safe_ratio = safe_growth + 1.0
        actual_ratio = jnp.clip(safe_ratio, min=0.0, max=limit_growth + 1.0)
        is_clipped = jnp.any(raw_growth > limit_growth)
        updated_val = val_t_minus_1 * actual_ratio

        return expected.at[self.term_idx, :].set(updated_val), is_clipped

class LinearTrendTerminal(eqx.Module):
    r"""
    Linear trend terminal approximator.

    Approximates the terminal value by adding an exogenous drift component
    :math:`\delta_s` (from `params`) to the value function.

    It supports three patterns for :math:`\delta`:

    1. **Global Drift**: A single key mapping to a scalar drift value applied to all terminal states.
    2. **Aggregated Drifts**: A list or tuple of keys. The number of keys must match
       the length of ``term_idx`` (and ``prev_idx``). Each scalar corresponds to a terminal state in order.
    3. **Drift Vector**: A single key mapping to an array of length :math:`n`,
       where :math:`n` is the length of ``term_idx``. Each element corresponds to a terminal state in order.

    .. math::
        \mathbb{E}V_T(s, a) = \mathbb{E}V_{T-1}(s', a) + \delta_s

    Args:
        term_idx: Indices of the terminal states :math:`T`.
        prev_idx: Indices of the predecessor states :math:`T-1`.
        drift_keys: Identifier(s) for drift :math:`\delta`. Accepts a single ``str``
            for global/vector parameters, or a ``list[str]`` to aggregate
            multiple regional scalars.
        scale: Scaling factor for numerical stability. Input parameters are divided
            by this value. Useful when drift values have a large absolute magnitude.

    Raises:
        ValueError: If `drift_keys` is None.

    Examples:
        >>> # Linear drift via parameter keys
        >>> approx = LinearTrendTerminal(term_idx, prev_idx, drift_keys="drift")
        >>> params = {"drift": 500.0}
        >>> adjusted_ev = approx.approximate(expected, params, model, discount_factor=0.99)
    """
    term_idx: tuple[int, ...] = eqx.field(static=True)
    prev_idx: tuple[int, ...] = eqx.field(static=True)
    drift_keys: Union[str, List[str], Tuple[str, ...]] | None = None
    scale: float = eqx.field(static=True, default=1.0)

    def __check_init__(self) -> None:
        if len(self.term_idx) != len(self.prev_idx):
            raise ValueError(
                f"LinearTrendTerminal: term_idx and prev_idx must have the same shape. "
                f"Got {len(self.term_idx)} and {len(self.prev_idx)}."
            )
        if self.drift_keys is None:
            raise ValueError(
                "LinearTrendTerminal requires 'drift_keys' to be set."
            )

    def approximate(
        self,
        expected: Float[Array, "S A"],
        params: PyTree,
        model: StructuralModel,
        discount_factor: float
    ) -> tuple[Float[Array, "S A"], Array | None]:
        r"""
        Applies linear drift to the terminal horizon.

        Adds :math:`\delta` from `params` to :math:`T-1` values.
        """
        val_t_minus_1 = expected[self.prev_idx, :]

        delta_effective = _retrieve_and_validate_param(
            self.drift_keys, params, self.prev_idx, "LinearTrendTerminal: delta"
        )
        updated_val = val_t_minus_1 + delta_effective / self.scale

        return expected.at[self.term_idx, :].set(updated_val), None
