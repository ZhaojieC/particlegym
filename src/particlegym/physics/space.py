"""Simulation domain: an axis-aligned box with optional periodicity per axis."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np


@dataclasses.dataclass(frozen=True)
class Box:
    """Axis-aligned simulation domain in two or three dimensions.

    Periodic axes use the minimum-image convention. Non-periodic axes are open:
    particles are confined by walls (see :mod:`particlegym.physics.geometry`),
    not by the box itself.

    Args:
        lo: Lower corner, one value per axis.
        hi: Upper corner, one value per axis.
        periodic: Periodicity of each axis, or a single value for all axes.
    """

    lo: tuple[float, ...]
    hi: tuple[float, ...]
    periodic: tuple[bool, ...] | bool = False

    def __post_init__(self) -> None:
        lo = tuple(float(x) for x in self.lo)
        hi = tuple(float(x) for x in self.hi)
        if len(lo) not in (2, 3) or len(hi) != len(lo):
            raise ValueError("lo and hi must both have 2 or 3 entries")
        if any(high <= low for low, high in zip(lo, hi, strict=True)):
            raise ValueError("hi must be larger than lo along every axis")
        periodic = self.periodic
        if isinstance(periodic, bool | np.bool_):
            periodic = (bool(periodic),) * len(lo)
        periodic = tuple(bool(p) for p in periodic)
        if len(periodic) != len(lo):
            raise ValueError("periodic must have one entry per axis")
        object.__setattr__(self, "lo", lo)
        object.__setattr__(self, "hi", hi)
        object.__setattr__(self, "periodic", periodic)

    @property
    def dim(self) -> int:
        """Number of spatial dimensions."""
        return len(self.lo)

    @property
    def length(self) -> np.ndarray:
        """Edge lengths of the box."""
        return np.subtract(self.hi, self.lo)

    def displacement(self, a: jax.Array, b: jax.Array, axis: int = -1) -> jax.Array:
        """Return ``a - b``, using the minimum image along periodic axes.

        Args:
            a: Positions.
            b: Positions, broadcastable against ``a``.
            axis: Array axis that holds the spatial coordinates.
        """
        dr = a - b
        if not any(self.periodic):
            return dr
        shape = [1] * dr.ndim
        shape[axis] = self.dim
        length = jnp.asarray(self.length, dtype=dr.dtype).reshape(shape)
        wrapped = dr - length * jnp.round(dr / length)
        return jnp.where(jnp.asarray(self.periodic).reshape(shape), wrapped, dr)

    def wrap(self, pos: jax.Array) -> jax.Array:
        """Map positions into the box along periodic axes; other axes are unchanged."""
        if not any(self.periodic):
            return pos
        lo = jnp.asarray(self.lo, dtype=pos.dtype)
        length = jnp.asarray(self.length, dtype=pos.dtype)
        wrapped = lo + jnp.mod(pos - lo, length)
        return jnp.where(jnp.asarray(self.periodic), wrapped, pos)
