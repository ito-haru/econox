# src/econox/components/distribution.py
"""
Distribution components for Econox.
Handles stochastic parts of the model (error terms).
"""

import jax
import equinox as eqx
from jaxtyping import Array, Float

class GumbelDistribution(eqx.Module):
    """
    Type I Extreme Value (Gumbel) distribution logic for Logit models.
    Provides Emax (LogSumExp) and choice probabilities (Softmax).
    """
    scale: float = 1.0

    def expected_max(
        self, 
        values: Float[Array, "num_states num_actions"]
    ) -> Float[Array, "num_states"]:
        """
        Computes the expected maximum value E[max(v + epsilon)].
        
        Formula: scale * log( sum( exp(v / scale) ) )
        (Note: Standard implementation commonly refers to this as the 'Inclusive Value'.)
        """
        # axis=-1 assumes the last dimension is actions (num_states, num_actions)
        return self.scale * jax.scipy.special.logsumexp(values / self.scale, axis=-1)

    def choice_probabilities(
        self, 
        values: Float[Array, "num_states num_actions"]
    ) -> Float[Array, "num_states num_actions"]:
        """
        Computes the choice probabilities P(choice | state).
        
        Formula: exp(v / scale) / sum( exp(v / scale) )
        """
        return jax.nn.softmax(values / self.scale, axis=-1)