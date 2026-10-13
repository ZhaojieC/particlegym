"""Descriptions of observation and action spaces.

Every environment describes the observation and the action of one agent with
one of these classes. They are small stand-ins for the spaces of Gymnasium,
so that the core needs no extra dependency.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp


class Discrete(NamedTuple):
    """Integers ``0, ..., n - 1``."""

    n: int

    @property
    def shape(self) -> tuple[int, ...]:
        return ()

    def sample(self, key: jax.Array, batch: tuple[int, ...] = ()) -> jax.Array:
        """Uniform random elements with the leading shape ``batch``."""
        return jax.random.randint(key, batch, 0, self.n)

    def contains(self, x) -> jax.Array:
        x = jnp.asarray(x)
        return jnp.all((x >= 0) & (x < self.n)) & jnp.issubdtype(x.dtype, jnp.integer)


class Continuous(NamedTuple):
    """Real arrays of a given shape with entries in ``[low, high]``."""

    low: float
    high: float
    shape: tuple[int, ...]

    def sample(self, key: jax.Array, batch: tuple[int, ...] = ()) -> jax.Array:
        """Uniform random elements with the leading shape ``batch``.

        Unbounded entries are drawn from a standard normal distribution.
        """
        shape = (*batch, *self.shape)
        if math.isfinite(self.low) and math.isfinite(self.high):
            return jax.random.uniform(key, shape, minval=self.low, maxval=self.high)
        return jax.random.normal(key, shape)

    def contains(self, x) -> jax.Array:
        x = jnp.asarray(x)
        return jnp.all((x >= self.low) & (x <= self.high)) & (
            x.shape[-len(self.shape) :] == self.shape
        )
