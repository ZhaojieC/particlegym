"""Pair interactions.

A pair potential is an energy ``phi(r, sigma)`` of the centre-to-centre
distance ``r`` and the contact distance ``sigma`` (the sum of the two radii).
Forces are obtained from it by automatic differentiation, so forces and
energies are consistent by construction and Newton's third law holds. The same
potentials describe particle-wall contacts, with ``r`` the distance from the
particle centre to the wall and ``sigma`` the particle radius.

Dissipative pair forces, which depend on velocities and have no potential,
are written explicitly as subclasses of :class:`PairDissipation`.
"""

from __future__ import annotations

import abc
import dataclasses

import jax
import jax.numpy as jnp

_WCA_CUTOFF = 2.0 ** (1.0 / 6.0)


class PairPotential(abc.ABC):
    """Conservative pair interaction defined by its energy."""

    @abc.abstractmethod
    def energy(self, r: jax.Array, sigma: jax.Array) -> jax.Array:
        """Elementwise energy; must be zero for ``r >= cutoff(sigma)``."""

    @abc.abstractmethod
    def cutoff(self, sigma):
        """Distance beyond which the interaction vanishes."""


class PairDissipation(abc.ABC):
    """Velocity-dependent pair force without a potential."""

    @abc.abstractmethod
    def force(self, unit: jax.Array, r: jax.Array, dv: jax.Array, sigma: jax.Array) -> jax.Array:
        """Force on particle i from particle j, shape (d, ...).

        Vectors have the coordinate axis first, which lets the compiler
        vectorise over pairs.

        Args:
            unit: Unit vectors pointing from j to i, shape (d, ...).
            r: Distances, shape (...).
            dv: Relative velocities ``v_j - v_i``, shape (d, ...).
            sigma: Contact distances, shape (...).
        """

    @abc.abstractmethod
    def cutoff(self, sigma):
        """Distance beyond which the force vanishes."""


@dataclasses.dataclass(frozen=True)
class Harmonic(PairPotential):
    """Soft repulsion between overlapping particles.

    ``phi(r) = (k / 2) (sigma - r)^2`` for ``r < sigma``, zero otherwise.
    """

    k: float = 1.0

    def energy(self, r, sigma):
        overlap = jnp.maximum(sigma - r, 0.0)
        return 0.5 * self.k * overlap**2

    def cutoff(self, sigma):
        return sigma


@dataclasses.dataclass(frozen=True)
class WCA(PairPotential):
    """Weeks-Chandler-Andersen repulsion (Weeks, Chandler & Andersen, 1971).

    The Lennard-Jones potential with length scale ``sigma``, truncated at its
    minimum ``2^(1/6) sigma`` and shifted so that energy and force vanish
    there.
    """

    epsilon: float = 1.0

    def energy(self, r, sigma):
        x = (sigma / r) ** 6
        e = 4.0 * self.epsilon * (x * x - x) + self.epsilon
        return jnp.where(r < _WCA_CUTOFF * sigma, e, 0.0)

    def cutoff(self, sigma):
        return _WCA_CUTOFF * sigma


@dataclasses.dataclass(frozen=True)
class SocialForce(PairPotential):
    """Conservative part of the social force model of Helbing, Farkas & Vicsek (2000).

    The repulsive force between pedestrians at distance ``r`` is
    ``A exp((sigma - r) / B) + k g(sigma - r)`` with ``g(x) = max(x, 0)``: an
    exponential social repulsion plus a body force during contact. It derives
    from ``phi(r) = A B exp((sigma - r) / B) + (k / 2) g(sigma - r)^2``, which
    is truncated at ``sigma + interaction_range`` and shifted to zero there.
    The tangential sliding friction of the same model is dissipative; use
    :class:`TangentialFriction` for it.

    Defaults are the published values in SI units (N, m, kg).

    Args:
        A: Strength of the social repulsion (N).
        B: Range of the social repulsion (m).
        k: Body-force stiffness during contact (kg s^-2).
        interaction_range: Truncation distance beyond contact (m). With the
            defaults the neglected force is below 0.01 N.
    """

    A: float = 2.0e3
    B: float = 0.08
    k: float = 1.2e5
    interaction_range: float = 1.0

    def energy(self, r, sigma):
        gap = r - sigma
        e = self.A * self.B * (jnp.exp(-gap / self.B) - jnp.exp(-self.interaction_range / self.B))
        e = e + 0.5 * self.k * jnp.maximum(-gap, 0.0) ** 2
        return jnp.where(gap < self.interaction_range, e, 0.0)

    def cutoff(self, sigma):
        return sigma + self.interaction_range


@dataclasses.dataclass(frozen=True)
class TangentialFriction(PairDissipation):
    """Sliding friction between touching particles (Helbing, Farkas & Vicsek, 2000).

    ``f_i = kappa g(sigma - r) [dv - (dv . n) n]`` with ``dv = v_j - v_i`` and
    ``n`` the unit vector between the centres. The force is antisymmetric, so
    momentum is conserved, and it can only remove kinetic energy.

    Args:
        kappa: Friction coefficient (kg m^-1 s^-1 in SI units).
    """

    kappa: float = 2.4e5

    def force(self, unit, r, dv, sigma):
        overlap = jnp.maximum(sigma - r, 0.0)
        dv_tangential = dv - jnp.sum(dv * unit, axis=0) * unit
        return self.kappa * overlap * dv_tangential

    def cutoff(self, sigma):
        return sigma
