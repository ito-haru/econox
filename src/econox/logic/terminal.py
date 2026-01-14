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
from jaxtyping import Array, Float, Int, PyTree

from econox.protocols import StructuralModel
from econox.utils import get_from_pytree


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
        term_idx (Int[Array, "n"]): Indices of the terminal states :math:`\mathcal{S}_{term}`.
        prev_idx (Int[Array, "n"]): Indices of the predecessor states :math:`\mathcal{S}_{prev}`.
    
    Examples:
        >>> # Assuming states [4, 5] are terminal and [2, 3] are T-1
        >>> approximator = StationaryTerminal(
        ...     term_idx=jnp.array([4, 5]),
        ...     prev_idx=jnp.array([2, 3])
        ... )
        >>> adjusted_ev = approximator.approximate(expected, params, model)
    """
    term_idx: Int[Array, "n"]
    prev_idx: Int[Array, "n"]

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
    
    It supports spatial heterogeneity in growth rates through three parameter 
    specification patterns:
    
    1. **Global Scalar**: A single key mapping to a scalar value (e.g., ``"g"``). 
       The same growth rate is applied to all states.
    2. **Aggregated Scalars**: A list or tuple of keys (e.g., ``["g_tokyo", "g_osaka"]``). 
       Individual scalar parameters are collected into a spatial vector.
    3. **Spatial Vector**: A single key mapping to an array of length :math:`S` 
       (e.g., ``"g_vector"``). Each element corresponds to a specific state.

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
        
        >>> # Pattern 2: Aggregated regional scalars (Spatial Heterogeneity)
        >>> approx = ExponentialTrendTerminal(
        ...     term_idx, prev_idx, growth_rate_keys=["g_tokyo", "g_osaka"]
        ... )
        >>> params = {"g_tokyo": 0.03, "g_osaka": 0.01}
        
        >>> # Pattern 3: Endogenous dynamic extrapolation (No params needed)
        >>> approx = ExponentialTrendTerminal(term_idx, prev_idx, pre_prev_idx=pre_prev)
        >>> adjusted_ev = approx.approximate(expected, {}, model)
    """
    term_idx: Int[Array, "n"]
    prev_idx: Int[Array, "n"]
    pre_prev_idx: Int[Array, "n"] | None = None
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
        val_t_minus_1 = expected[self.prev_idx, :]

        # Case 1: Growth rates provided
        if self.growth_rate_keys is not None:
            if isinstance(self.growth_rate_keys, (list, tuple)):
                gamma = jnp.array([get_from_pytree(params, k, 0.0) for k in self.growth_rate_keys])
            else:
                gamma = jnp.asarray(get_from_pytree(params, self.growth_rate_keys, 0.0))
            
            gamma_eff = gamma[self.prev_idx, jnp.newaxis] if gamma.ndim > 0 else gamma
            updated_val = val_t_minus_1 * (1.0 + gamma_eff)

        # Case 2: Pre-previous indices provided    
        elif self.pre_prev_idx is not None:
            val_t_minus_2 = expected[self.pre_prev_idx, :]
            ratio = val_t_minus_1 / (jnp.abs(val_t_minus_2) + 1e-8)
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

    Similar to the exponential variant, it supports three patterns for :math:`\delta_s`:
    
    1. **Global Drift**: A single key mapping to a scalar drift value.
    2. **Aggregated Drifts**: A collection of keys mapping to region-specific scalars.
    3. **Spatial Drift Vector**: A single key mapping to a pre-formed drift array 
       of length :math:`S`.

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
    term_idx: Int[Array, "n"]
    prev_idx: Int[Array, "n"]
    pre_prev_idx: Int[Array, "n"] | None = None
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
            
            # Retrieve and aggregate drift parameters
            if isinstance(self.drift_keys, (list, tuple)):
                delta = jnp.array([
                    get_from_pytree(params, k, default=0.0) 
                    for k in self.drift_keys
                ])
            else:
                delta = jnp.asarray(get_from_pytree(params, self.drift_keys, default=0.0))

            val_prev = expected[self.prev_idx, :]

            if delta.ndim > 0:
                delta_effective = delta[self.prev_idx, jnp.newaxis]
            else:
                delta_effective = delta

            updated_val = val_prev + delta_effective
            return expected.at[self.term_idx, :].set(updated_val)
        
        # Case 2: Pre-previous indices provided
        elif self.pre_prev_idx is not None:
            val_t_minus_2 = expected[self.pre_prev_idx, :]
            diff = val_t_minus_1 - val_t_minus_2
            updated_val = val_t_minus_1 + diff
            return expected.at[self.term_idx, :].set(updated_val)
        
        else:
            raise ValueError(
                "LinearTrendTerminal requires either 'drift_keys' or 'pre_prev_idx' to be set."
            )
