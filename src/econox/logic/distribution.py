# src/econox/logic/distribution.py
"""
Distribution components for Econox.
Handles stochastic parts of the model (error terms).
"""

import jax
import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray

class GumbelDistribution(eqx.Module):
    """
    Type I Extreme Value (Gumbel) distribution logic for Logit models.
    Provides Emax (LogSumExp), choice probabilities (Softmax) and 
    transformation of random draws for Mixed Models.

    Attributes:
        scale (float): Scale parameter of the Gumbel distribution.
    """
    scale: float = 1.0

    def __check_init__(self) -> None:
        if self.scale <= 0:
            raise ValueError(f"Gumbel scale must be positive, got {self.scale}")

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
    
    def transform(
        self,
        draws: Float[Array, "..."],
        loc: Float[Array, "..."],
        scale: Float[Array, "..."]
    )-> Float[Array, "..."]:
        """
        Transforms standard uniform draws into Gumbel-distributed variables.
        
        Uses the inverse CDF method:
        Gumbel = loc - scale * log(-log(U)), where U ~ Uniform(0, 1)
        """
        return loc - scale * jax.numpy.log(-jax.numpy.log(draws))
    
    def generate_standard_draws(
        self, 
        key: PRNGKeyArray, 
        shape: tuple[int, ...]
        ) -> Float[Array, "..."]:
        """
        Generates standard uniform random draws for Gumbel transformation.
        
        Args:
            key: Random key/seed for reproducibility.
            shape: Desired shape of the output array.
        Returns:
            Array of random variables uniformly distributed in (0, 1).
        """
        # minval=1e-6 to avoid log(-log(0)) = -inf in the Gumbel transform
        return jax.random.uniform(key, shape=shape, minval=1e-6, maxval=1.0)
    
class NormalDistribution(eqx.Module):
    """
    Normal distribution for Mixed Models (random coefficients).

    NOTE: `expected_max` and `choice_probabilities` are not yet implemented.
    Only `transform` and `generate_standard_draws` are available for use with MixedUtility.
    """

    def expected_max(
        self,
        values: Float[Array, "num_states num_actions"]
    ) -> Float[Array, "num_states"]:
        """
        Expected max under normal distribution.
        # TODO: Implement via numerical integration.
        """
        raise NotImplementedError("Emax for Normal distribution is not implemented.")

    def choice_probabilities(
        self,
        values: Float[Array, "num_states num_actions"]
    ) -> Float[Array, "num_states num_actions"]:
        """
        Choice probabilities under normal distribution (Probit).
        # TODO: Implement via GHK simulator or accept Monte Carlo draws from MixedUtility.
        """
        raise NotImplementedError("Choice probabilities for Normal distribution are not implemented.")
    
    def transform(
        self,
        draws: Float[Array, "..."],
        loc: Float[Array, "..."],
        scale: Float[Array, "..."]
    ) -> Float[Array, "..."]:
        """
        Transforms standard normal draws into specified normal distribution.
        
        Formula: X = loc + scale * Z, where Z ~ N(0, 1)
        """
        return loc + scale * draws
    
    def generate_standard_draws(
        self, 
        key: PRNGKeyArray, 
        shape: tuple[int, ...]
        ) -> Float[Array, "..."]:
        """
        Generates standard normal random draws.
        
        Args:
            key: Random key/seed for reproducibility.
            shape: Desired shape of the output array.
        Returns:
            Array of random variables following N(0, 1).
        """
        return jax.random.normal(key, shape=shape)
    