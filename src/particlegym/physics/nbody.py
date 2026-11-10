# ParticleGym, https://github.com/ZhaojieC/particlegym (MIT license).
# Please cite it in published work: particlegym.cite() gives the references.
"""Newtonian gravity between particles.

:class:`NewtonGravity` is the gravitational attraction of every particle by
every other one (the N-body problem), or by a few heavy particles only (the
restricted N-body problem). :func:`orbital_elements` gives the semi-major
axis and the eccentricity vector of a Kepler orbit, and
:func:`hohmann_delta_v` the velocity change of the Hohmann transfer between
two circular orbits.
"""

from __future__ import annotations

import dataclasses
import math

import jax
import jax.numpy as jnp

from particlegym.physics.external import ExternalForce


@dataclasses.dataclass(frozen=True)
class NewtonGravity(ExternalForce):
    """Newtonian gravity with Plummer softening.

    The force on particle i is

        F_i = -G m_i sum_j m_j (x_i - x_j) / (|x_i - x_j|^2 + eps^2)^(3/2),

    the gradient of the potential energy

        U = -G sum_{i<j} m_i m_j / sqrt(|x_i - x_j|^2 + eps^2).

    The softening length eps keeps close encounters finite; eps = 0 gives
    Newton's law. With ``sources = k`` only the first k particles attract:
    the others are test particles, which move in the field of the sources
    without acting on anything (the restricted N-body problem, as in the
    galaxy encounters of Toomre & Toomre, 1972). Every particle is paired
    with every source, so the cost grows as N k, or N^2 without ``sources``:
    this is meant for up to a few thousand particles. Positions are not
    wrapped into a periodic box.

    It is added to a :class:`~particlegym.physics.ForceField` as an external
    term, although it depends on the positions of all particles.

    Args:
        G: Gravitational constant.
        softening: Plummer softening length eps.
        sources: Number of leading particles that attract; all if None.
    """

    G: float = 1.0
    softening: float = 0.0
    sources: int | None = None

    def __post_init__(self) -> None:
        if self.G < 0 or self.softening < 0:
            raise ValueError("G and the softening length must not be negative")
        if self.sources is not None and self.sources < 1:
            raise ValueError("sources must be at least 1, or None for all particles")

    def _pairs(self, pos: jax.Array):
        """Separations (N, k, d) from the sources, softened squared distances
        (N, k) and the mask of self-pairs."""
        n = pos.shape[0]
        k = n if self.sources is None else min(self.sources, n)
        separation = pos[:, None, :] - pos[None, :k, :]
        same = jnp.arange(n)[:, None] == jnp.arange(k)[None, :]
        r2 = jnp.sum(separation * separation, axis=-1) + self.softening**2
        return separation, jnp.where(same, 1.0, r2), same, k

    def force(self, pos, vel, mass):
        separation, r2, same, k = self._pairs(pos)
        weight = jnp.where(same, 0.0, mass[None, :k] * r2**-1.5)
        return -self.G * mass[:, None] * jnp.einsum("ik,ikd->id", weight, separation)

    def energy(self, pos, mass):
        """(N,) potential energies.

        A pair of sources gives half of its energy to each of its two
        particles; a test particle carries all of its energy in the field of
        the sources. Without ``sources`` the energies add up to U. In the
        restricted problem, the energy of a test particle is its potential
        energy in the field of the sources, and the force on it is minus the
        gradient of that energy.
        """
        _, r2, same, k = self._pairs(pos)
        pair = jnp.where(same, 0.0, mass[None, :k] / jnp.sqrt(r2))
        share = jnp.where(jnp.arange(pos.shape[0]) < k, 0.5, 1.0)
        return -self.G * mass * share * jnp.sum(pair, axis=1)


def orbital_elements(position: jax.Array, velocity: jax.Array, gm) -> tuple[jax.Array, jax.Array]:
    """Semi-major axis and eccentricity vector of a Kepler orbit.

    For the position r and velocity v of one body relative to the other,
    and ``gm = G (M + m)``:

        a = -gm / (2 E),  E = |v|^2 / 2 - gm / |r|,
        e = ((|v|^2 - gm / |r|) r - (r . v) v) / gm.

    ``a`` is negative for hyperbolic orbits. The eccentricity vector points
    to the periapsis; its length is the eccentricity.

    Args:
        position: (..., d) relative positions.
        velocity: (..., d) relative velocities.
        gm: The product G (M + m).

    Returns:
        ``(a, e)`` with shapes (...) and (..., d).
    """
    r = jnp.linalg.norm(position, axis=-1)
    v2 = jnp.sum(velocity * velocity, axis=-1)
    energy = 0.5 * v2 - gm / r
    a = -gm / (2.0 * energy)
    radial = jnp.sum(position * velocity, axis=-1)
    e = ((v2 - gm / r)[..., None] * position - radial[..., None] * velocity) / gm
    return a, e


def hohmann_delta_v(r0: float, r1: float, gm: float = 1.0) -> float:
    """Total velocity change of the Hohmann transfer between circular orbits.

    Two impulsive, tangential burns: at the start onto the ellipse that
    touches both orbits, and at the other end of it onto the second orbit.
    Among transfers with two impulses it is the one with the smallest total
    velocity change (for r1 / r0 below 11.94 it is the optimum among all
    impulsive transfers).
    """
    a = 0.5 * (r0 + r1)
    first = abs(math.sqrt(gm / r0) * (math.sqrt(r1 / a) - 1.0))
    second = abs(math.sqrt(gm / r1) * (1.0 - math.sqrt(r0 / a)))
    return first + second
