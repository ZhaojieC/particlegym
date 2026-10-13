"""Interactions between listed particles: bonds, angles and enclosed areas.

Every term is defined by its energy; the force field obtains the forces by
automatic differentiation. A term is switched off while one of its particles
is dead.
"""

from __future__ import annotations

import abc
import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np

from particlegym.physics.space import Box


def _length(x: jax.Array) -> jax.Array:
    """Euclidean norm over the last axis, with a zero gradient at the origin."""
    s = jnp.sum(x * x, axis=-1)
    positive = s > 0
    return jnp.where(positive, jnp.sqrt(jnp.where(positive, s, 1.0)), 0.0)


def _indices(values, width: int, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.int32)
    if values.ndim != 2 or values.shape[1] != width:
        raise ValueError(f"{name} must have shape (M, {width})")
    return values


class BondedTerm(abc.ABC):
    """An energy of the positions of listed particles."""

    @abc.abstractmethod
    def energy(self, pos: jax.Array, box: Box, alive: jax.Array) -> jax.Array:
        """Total energy of the term."""


@dataclasses.dataclass(frozen=True, eq=False)
class Bonds(BondedTerm):
    """Harmonic bonds: ``E = (k / 2) (r - r0)^2`` for every listed pair.

    With ``r0 = 0`` the bonds are the entropic springs of an ideal (Gaussian)
    chain.

    Args:
        pairs: (M, 2) particle indices.
        k: Stiffness, scalar or (M,).
        r0: Rest length, scalar or (M,).
    """

    pairs: np.ndarray
    k: float | np.ndarray = 1.0
    r0: float | np.ndarray = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "pairs", _indices(self.pairs, 2, "pairs"))

    def energy(self, pos, box, alive):
        i, j = self.pairs[:, 0], self.pairs[:, 1]
        r = _length(box.displacement(pos[i], pos[j]))
        energy = 0.5 * jnp.asarray(self.k, pos.dtype) * (r - jnp.asarray(self.r0, pos.dtype)) ** 2
        return jnp.sum(jnp.where(alive[i] & alive[j], energy, 0.0))


@dataclasses.dataclass(frozen=True, eq=False)
class Angles(BondedTerm):
    """Bending energy ``E = k (1 - cos(theta - theta0))`` for every listed triplet.

    ``theta`` is the angle at the middle particle j between the directions to
    i and to k. Near the rest angle the energy is ``(k / 2) (theta - theta0)^2``.
    In 2D the angle is signed (counter-clockwise from i to k), so ``theta0``
    and ``-theta0`` are different shapes. In 3D it is between 0 and pi. The
    default ``theta0 = pi`` describes a chain that resists bending.

    Args:
        triplets: (M, 3) particle indices (i, j, k).
        k: Bending stiffness (an energy).
        theta0: Rest angle in radians.
    """

    triplets: np.ndarray
    k: float = 1.0
    theta0: float = math.pi

    def __post_init__(self) -> None:
        object.__setattr__(self, "triplets", _indices(self.triplets, 3, "triplets"))

    def energy(self, pos, box, alive):
        i, j, k = self.triplets[:, 0], self.triplets[:, 1], self.triplets[:, 2]
        a = box.displacement(pos[i], pos[j])
        b = box.displacement(pos[k], pos[j])
        norm = _length(a) * _length(b)
        on = alive[i] & alive[j] & alive[k] & (norm > 0)
        norm = jnp.where(on, norm, 1.0)
        cos = jnp.sum(a * b, axis=-1) / norm
        aligned = math.cos(self.theta0) * cos
        sin0 = math.sin(self.theta0)
        if abs(sin0) > 1e-12:  # the sine is not needed for a straight rest shape
            if pos.shape[1] == 2:
                sin = (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) / norm
            else:
                sin = _length(jnp.cross(a, b)) / norm
            aligned = aligned + sin0 * sin
        return jnp.sum(jnp.where(on, self.k * (1.0 - aligned), 0.0))


@dataclasses.dataclass(frozen=True, eq=False)
class AreaConstraint(BondedTerm):
    """Area elasticity of closed rings of particles in 2D.

    ``E = (k / 2) (A - A0)^2`` for every ring, with A the area of the polygon
    through the particles of the ring. Together with :class:`Bonds` along the
    ring this is a simple model of a vesicle or a cell with an incompressible
    interior. The area is positive for particles listed counter-clockwise.

    Args:
        rings: (M, n) particle indices, in order around each ring.
        k: Area stiffness (energy per area squared).
        area0: Rest area, scalar or (M,).
    """

    rings: np.ndarray
    k: float = 1.0
    area0: float | np.ndarray = 1.0

    def __post_init__(self) -> None:
        rings = np.asarray(self.rings, dtype=np.int32)
        if rings.ndim != 2 or rings.shape[1] < 3:
            raise ValueError("rings must have shape (M, n) with n of at least 3")
        object.__setattr__(self, "rings", rings)

    def area(self, pos: jax.Array, box: Box) -> jax.Array:
        """(M,) signed areas of the rings."""
        if pos.shape[1] != 2:
            raise ValueError("AreaConstraint is defined in 2D only")
        # Positions relative to the first particle of each ring, so that a
        # ring may cross a periodic boundary.
        q = box.displacement(pos[self.rings], pos[self.rings[:, :1]])
        nxt = jnp.roll(q, -1, axis=1)
        return 0.5 * jnp.sum(q[..., 0] * nxt[..., 1] - nxt[..., 0] * q[..., 1], axis=1)

    def energy(self, pos, box, alive):
        on = jnp.all(alive[self.rings], axis=1)
        deviation = self.area(pos, box) - jnp.asarray(self.area0, pos.dtype)
        return jnp.sum(jnp.where(on, 0.5 * self.k * deviation**2, 0.0))
