"""Scalar fields on a regular grid, coupled to the particles.

A field, for example the concentration of a chemical, is an array with one
value per grid cell. Particles add to it (:meth:`Grid.deposit`) and read it or
its gradient at their positions (:meth:`Grid.sample`,
:meth:`Grid.sample_gradient`); :meth:`Grid.diffuse` advances the diffusion
equation by one explicit step.

Deposit and sampling use the same linear weights. A particle therefore feels
no gradient from the field it has deposited itself, as long as it is away
from the boundaries.
"""

from __future__ import annotations

import dataclasses
import itertools

import jax
import jax.numpy as jnp
import numpy as np


@dataclasses.dataclass(frozen=True)
class Grid:
    """A regular grid of cells on the region ``[lo, hi]``.

    Cell ``(i, j, ...)`` has its centre at ``lo + (index + 1/2) * spacing``.
    Along a non-periodic axis the boundary is closed: nothing diffuses
    through it.

    Args:
        lo: Lower corner of the region.
        hi: Upper corner of the region.
        shape: Number of cells along each axis.
        periodic: Periodicity of each axis, or a single value for all axes.
    """

    lo: tuple[float, ...]
    hi: tuple[float, ...]
    shape: tuple[int, ...]
    periodic: tuple[bool, ...] | bool = False

    def __post_init__(self) -> None:
        lo = tuple(float(x) for x in self.lo)
        hi = tuple(float(x) for x in self.hi)
        shape = tuple(int(n) for n in self.shape)
        if len(lo) not in (2, 3) or len(hi) != len(lo) or len(shape) != len(lo):
            raise ValueError("lo, hi and shape must all have 2 or 3 entries")
        if any(b <= a for a, b in zip(lo, hi, strict=True)) or any(n < 2 for n in shape):
            raise ValueError("hi must be larger than lo, with at least 2 cells per axis")
        periodic = self.periodic
        if isinstance(periodic, bool | np.bool_):
            periodic = (bool(periodic),) * len(lo)
        periodic = tuple(bool(p) for p in periodic)
        if len(periodic) != len(lo):
            raise ValueError("periodic must have one entry per axis")
        object.__setattr__(self, "lo", lo)
        object.__setattr__(self, "hi", hi)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "periodic", periodic)

    @property
    def dim(self) -> int:
        """Number of spatial dimensions."""
        return len(self.lo)

    @property
    def spacing(self) -> np.ndarray:
        """Edge lengths of a cell."""
        return np.subtract(self.hi, self.lo) / np.asarray(self.shape)

    @property
    def cell_volume(self) -> float:
        """Volume (area in 2D) of a cell."""
        return float(np.prod(self.spacing))

    def centres(self, dtype=None) -> jax.Array:
        """Coordinates of the cell centres, shape ``(*shape, d)``."""
        axes = [
            a + (jnp.arange(n, dtype=dtype) + 0.5) * h
            for a, n, h in zip(self.lo, self.shape, self.spacing, strict=True)
        ]
        return jnp.stack(jnp.meshgrid(*axes, indexing="ij"), axis=-1)

    def max_time_step(self, diffusivity: float) -> float:
        """Largest stable time step of :meth:`diffuse` for the given diffusivity."""
        return 0.5 / (diffusivity * float(np.sum(1.0 / self.spacing**2)))

    def _stencil(self, pos: jax.Array):
        """Indices and linear weights of the 2^d cells around every point.

        Returns a list with one ``(index, weight)`` pair per corner; ``index``
        is a tuple of d integer arrays of shape (N,), ``weight`` has shape (N,).
        """
        lo = jnp.asarray(self.lo, dtype=pos.dtype)
        spacing = jnp.asarray(self.spacing, dtype=pos.dtype)
        x = (pos - lo) / spacing - 0.5
        base = jnp.floor(x)
        fraction = x - base
        base = base.astype(jnp.int32)
        stencil = []
        for corner in itertools.product((0, 1), repeat=self.dim):
            weight = jnp.ones(pos.shape[0], dtype=pos.dtype)
            index = []
            for axis, upper in enumerate(corner):
                f = fraction[:, axis]
                weight = weight * (f if upper else 1.0 - f)
                i = base[:, axis] + upper
                n = self.shape[axis]
                # Outside a closed boundary the value of the last cell is used.
                index.append(jnp.mod(i, n) if self.periodic[axis] else jnp.clip(i, 0, n - 1))
            stencil.append((tuple(index), weight))
        return stencil

    def sample(self, values: jax.Array, pos: jax.Array) -> jax.Array:
        """Values of a field at the points ``pos`` (N, d), by linear interpolation.

        ``values`` may have leading axes, for example the d components of a
        gradient; the result then has shape (..., N).
        """
        return sum(weight * values[(..., *index)] for index, weight in self._stencil(pos))

    def deposit(self, pos: jax.Array, amount=1.0, alive: jax.Array | None = None) -> jax.Array:
        """Density field of point sources at ``pos`` (N, d) with the given amounts.

        Every amount is shared between the cells around the point with the
        weights of :meth:`sample` and divided by the cell volume, so that the
        integral of the result is the sum of the amounts.
        """
        amount = jnp.broadcast_to(jnp.asarray(amount, dtype=pos.dtype), pos.shape[:1])
        if alive is not None:
            amount = jnp.where(alive, amount, 0.0)
        field = jnp.zeros(self.shape, dtype=pos.dtype)
        for index, weight in self._stencil(pos):
            field = field.at[index].add(weight * amount)
        return field / self.cell_volume

    def _padded(self, values: jax.Array, axis: int) -> jax.Array:
        """``values`` with one more cell at both ends of ``axis``, set by the boundary type."""
        pad = [(0, 0)] * values.ndim
        pad[axis] = (1, 1)
        return jnp.pad(values, pad, mode="wrap" if self.periodic[axis] else "edge")

    def gradient(self, values: jax.Array) -> jax.Array:
        """Gradient of a field by central differences, shape ``(d, *shape)``."""
        parts = []
        for axis in range(self.dim):
            padded = self._padded(values, axis)
            upper = jax.lax.slice_in_dim(padded, 2, None, axis=axis)
            lower = jax.lax.slice_in_dim(padded, 0, -2, axis=axis)
            parts.append((upper - lower) / (2.0 * self.spacing[axis]))
        return jnp.stack(parts)

    def sample_gradient(self, values: jax.Array, pos: jax.Array) -> jax.Array:
        """Gradient of a field at the points ``pos``, shape (N, d)."""
        return self.sample(self.gradient(values), pos).T

    def laplacian(self, values: jax.Array) -> jax.Array:
        """Laplacian of a field with the standard second-order stencil."""
        total = jnp.zeros_like(values)
        for axis in range(self.dim):
            padded = self._padded(values, axis)
            upper = jax.lax.slice_in_dim(padded, 2, None, axis=axis)
            lower = jax.lax.slice_in_dim(padded, 0, -2, axis=axis)
            total = total + (upper - 2.0 * values + lower) / self.spacing[axis] ** 2
        return total

    def diffuse(self, values: jax.Array, diffusivity, dt, *, decay=0.0, source=None) -> jax.Array:
        """One explicit step of ``dC/dt = D laplacian(C) - decay * C + source``.

        The scheme is forward in time and central in space. Without decay and
        source the integral of the field is conserved exactly.

        Raises:
            ValueError: If ``dt`` exceeds :meth:`max_time_step` (checked when
                ``diffusivity`` and ``dt`` are Python numbers).
        """
        numbers = isinstance(diffusivity, int | float) and isinstance(dt, int | float)
        if numbers and diffusivity > 0 and dt > self.max_time_step(diffusivity) * (1 + 1e-12):
            raise ValueError(
                f"time step {dt} exceeds the stability limit "
                f"{self.max_time_step(diffusivity):.4g} of the explicit diffusion scheme"
            )
        rate = diffusivity * self.laplacian(values) - decay * values
        if source is not None:
            rate = rate + source
        return values + dt * rate
