"""Observation and action spaces."""

import jax
import jax.numpy as jnp
import numpy as np

from particlegym.envs import Continuous, Discrete


def test_discrete_space():
    space = Discrete(4)
    sample = jax.jit(lambda key: space.sample(key, (1000,)))(jax.random.key(0))
    assert space.shape == () and sample.shape == (1000,)
    np.testing.assert_array_equal(np.unique(sample), [0, 1, 2, 3])
    assert bool(space.contains(sample)) and bool(space.contains(3))
    assert not bool(space.contains(4)) and not bool(space.contains(1.0))


def test_continuous_space():
    space = Continuous(-1.0, 2.0, (3,))
    sample = jax.jit(lambda key: space.sample(key, (1000, 2)))(jax.random.key(0))
    assert sample.shape == (1000, 2, 3)
    assert float(sample.min()) >= -1.0 and float(sample.max()) <= 2.0
    assert float(sample.min()) < -0.95 and float(sample.max()) > 1.95
    assert bool(space.contains(sample)) and not bool(space.contains(jnp.array([0.0, 0.0, 2.5])))
    assert not bool(space.contains(jnp.zeros(2)))

    unbounded = Continuous(-jnp.inf, jnp.inf, (2,))
    sample = jax.jit(lambda key: unbounded.sample(key, (4000,)))(jax.random.key(0))
    assert sample.shape == (4000, 2) and bool(unbounded.contains(sample))
    np.testing.assert_allclose([sample.mean(), sample.std()], [0.0, 1.0], atol=0.05)
