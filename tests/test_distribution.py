"""
Tests for Distribution components.
Validates GumbelDistribution (transform/generate_standard_draws)
and NormalDistribution.
"""

import jax
import jax.numpy as jnp
import pytest

import econox as ecx

KEY = jax.random.PRNGKey(42)


# =============================================================================
# GumbelDistribution
# =============================================================================

class TestGumbelDistribution:
    def setup_method(self):
        self.dist = ecx.GumbelDistribution(scale=1.0)

    def test_generate_standard_draws_shape(self):
        draws = self.dist.generate_standard_draws(KEY, shape=(100, 3))
        assert draws.shape == (100, 3)

    def test_generate_standard_draws_range(self):
        draws = self.dist.generate_standard_draws(KEY, shape=(10000,))
        assert jnp.all(draws > 0.0)
        assert jnp.all(draws < 1.0)

    def test_transform_shape(self):
        draws = self.dist.generate_standard_draws(KEY, shape=(50, 2))
        loc = jnp.zeros(2)
        scale = jnp.ones(2)
        result = self.dist.transform(draws, loc, scale)
        assert result.shape == (50, 2)

    def test_transform_loc_shift(self):
        """Shifting loc should shift the transformed output by the same amount."""
        draws = self.dist.generate_standard_draws(KEY, shape=(100,))
        scale = jnp.ones(())
        result_0 = self.dist.transform(draws, jnp.zeros(()), scale)
        result_5 = self.dist.transform(draws, jnp.full((), 5.0), scale)
        assert jnp.allclose(result_5, result_0 + 5.0)

    def test_transform_scale_effect(self):
        """Larger scale should increase the spread of the output."""
        draws = self.dist.generate_standard_draws(KEY, shape=(1000,))
        loc = jnp.zeros(())
        out_small = self.dist.transform(draws, loc, jnp.ones(()) * 0.5)
        out_large = self.dist.transform(draws, loc, jnp.ones(()) * 2.0)
        assert jnp.std(out_large) > jnp.std(out_small)

    def test_transform_finite(self):
        """All transformed values must be finite (no -inf from log(0))."""
        draws = self.dist.generate_standard_draws(KEY, shape=(1000,))
        result = self.dist.transform(draws, jnp.zeros(()), jnp.ones(()))
        assert jnp.all(jnp.isfinite(result))

    def test_transform_inverse_cdf_formula(self):
        """Verify the inverse CDF formula: loc - scale * log(-log(U))."""
        draws = jnp.array([0.1, 0.5, 0.9])
        loc = jnp.array([1.0, 1.0, 1.0])
        scale = jnp.array([2.0, 2.0, 2.0])
        expected = loc - scale * jnp.log(-jnp.log(draws))
        result = self.dist.transform(draws, loc, scale)
        assert jnp.allclose(result, expected)

    def test_reproducibility(self):
        d1 = self.dist.generate_standard_draws(KEY, shape=(10,))
        d2 = self.dist.generate_standard_draws(KEY, shape=(10,))
        assert jnp.allclose(d1, d2)

    def test_different_keys_differ(self):
        key2 = jax.random.PRNGKey(99)
        d1 = self.dist.generate_standard_draws(KEY, shape=(100,))
        d2 = self.dist.generate_standard_draws(key2, shape=(100,))
        assert not jnp.allclose(d1, d2)


# =============================================================================
# NormalDistribution
# =============================================================================

class TestNormalDistribution:
    def setup_method(self):
        self.dist = ecx.NormalDistribution()

    def test_generate_standard_draws_shape(self):
        draws = self.dist.generate_standard_draws(KEY, shape=(200, 4))
        assert draws.shape == (200, 4)

    def test_generate_standard_draws_approx_standard_normal(self):
        draws = self.dist.generate_standard_draws(KEY, shape=(10000,))
        assert jnp.abs(jnp.mean(draws)) < 0.05
        assert jnp.abs(jnp.std(draws) - 1.0) < 0.05

    def test_transform_formula(self):
        """X = loc + scale * Z."""
        draws = jnp.array([0.0, 1.0, -1.0, 2.0])
        loc = jnp.array([3.0, 3.0, 3.0, 3.0])
        scale = jnp.array([2.0, 2.0, 2.0, 2.0])
        expected = loc + scale * draws
        result = self.dist.transform(draws, loc, scale)
        assert jnp.allclose(result, expected)

    def test_transform_shape(self):
        draws = self.dist.generate_standard_draws(KEY, shape=(50, 3))
        loc = jnp.zeros(3)
        scale = jnp.ones(3)
        result = self.dist.transform(draws, loc, scale)
        assert result.shape == (50, 3)

    def test_transform_loc_shift(self):
        draws = self.dist.generate_standard_draws(KEY, shape=(100,))
        scale = jnp.ones(())
        result_0 = self.dist.transform(draws, jnp.zeros(()), scale)
        result_5 = self.dist.transform(draws, jnp.full((), 5.0), scale)
        assert jnp.allclose(result_5, result_0 + 5.0)

    def test_transform_scale_effect(self):
        draws = self.dist.generate_standard_draws(KEY, shape=(1000,))
        loc = jnp.zeros(())
        out_small = self.dist.transform(draws, loc, jnp.ones(()) * 0.5)
        out_large = self.dist.transform(draws, loc, jnp.ones(()) * 2.0)
        assert jnp.std(out_large) > jnp.std(out_small)

    def test_transform_produces_correct_mean_std(self):
        """After transform, empirical mean ≈ loc and std ≈ scale."""
        draws = self.dist.generate_standard_draws(KEY, shape=(50000,))
        loc = jnp.array(3.0)
        scale = jnp.array(2.0)
        result = self.dist.transform(draws, loc, scale)
        assert jnp.abs(jnp.mean(result) - loc) < 0.05
        assert jnp.abs(jnp.std(result) - scale) < 0.05

    def test_reproducibility(self):
        d1 = self.dist.generate_standard_draws(KEY, shape=(10,))
        d2 = self.dist.generate_standard_draws(KEY, shape=(10,))
        assert jnp.allclose(d1, d2)

    def test_expected_max_not_implemented(self):
        values = jnp.zeros((4, 2))
        with pytest.raises(NotImplementedError):
            self.dist.expected_max(values)

    def test_choice_probabilities_not_implemented(self):
        values = jnp.zeros((4, 2))
        with pytest.raises(NotImplementedError):
            self.dist.choice_probabilities(values)
