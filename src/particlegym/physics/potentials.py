"""Pair interactions.

A pair potential is an energy ``phi(r, sigma)`` of the centre-to-centre
distance ``r`` and the contact distance ``sigma`` (the sum of the two radii).
Forces are obtained from it by automatic differentiation, so forces and
energies are consistent by construction and Newton's third law holds. The same
potentials describe particle-wall contacts, with ``r`` the distance from the
particle centre to the wall and ``sigma`` the particle radius.

Dissipative pair forces, which depend on velocities and have no potential,
are written explicitly as subclasses of :class:`PairDissipation`.
:class:`PairThermostat` adds the matching random pair force, and
:class:`Mixture` makes the strength of a potential depend on the species of
the two particles.
"""

from __future__ import annotations

import abc
import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np

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


@dataclasses.dataclass(frozen=True)
class ContactExponential(PairPotential):
    """Exponential repulsion that acts only during contact.

    The force is ``B exp((sigma - r) / E)`` for ``r < sigma`` and zero
    otherwise, so it jumps to ``B`` at contact. It derives from
    ``phi(r) = B E (exp((sigma - r) / E) - 1)``. Zhang, Chai, Sun &
    Lykotrafitis (arXiv:2007.03190) use this force, with ``B = 100`` and
    ``E = 0.08``, for the contact inhibition of locomotion between cells.

    Args:
        B: Force at contact.
        E: Decay length of the force.
    """

    B: float = 100.0
    E: float = 0.08

    def energy(self, r, sigma):
        overlap = jnp.maximum(sigma - r, 0.0)
        return self.B * self.E * jnp.expm1(overlap / self.E)

    def cutoff(self, sigma):
        return sigma


@dataclasses.dataclass(frozen=True, eq=False)
class Mixture(PairPotential):
    """A pair potential whose strength depends on the species of the two particles.

    The energy of a pair of species a and b is
    ``strength[a, b] * potential.energy(r, sigma)``. Species are the ``kind``
    labels of the particle state.

    Args:
        potential: The potential that is scaled.
        strength: (S, S) symmetric matrix of factors for S species.
    """

    potential: PairPotential
    strength: np.ndarray

    def __post_init__(self) -> None:
        strength = np.asarray(self.strength, dtype=float)
        if strength.ndim != 2 or strength.shape[0] != strength.shape[1]:
            raise ValueError("strength must be a square matrix")
        if not np.allclose(strength, strength.T):
            raise ValueError("strength must be symmetric")
        object.__setattr__(self, "strength", strength)

    def energy(self, r, sigma, kind_i=None, kind_j=None):
        if kind_i is None or kind_j is None:
            raise ValueError("a Mixture needs the species (kind) of the particles")
        factor = jnp.asarray(self.strength, dtype=r.dtype)[kind_i, kind_j]
        return factor * self.potential.energy(r, sigma)

    def cutoff(self, sigma):
        return self.potential.cutoff(sigma)


@dataclasses.dataclass(frozen=True)
class NormalDamping(PairDissipation):
    """Damping of the approach velocity of touching particles.

    ``f_i = gamma (dv . n) n`` for ``r < sigma``, with ``dv = v_j - v_i`` and
    ``n`` the unit vector from j to i. Together with :class:`Harmonic` this
    is the linear spring-dashpot contact model, which has a constant
    coefficient of restitution; see :func:`restitution_coefficient`.

    Args:
        gamma: Damping coefficient (mass per time).
    """

    gamma: float = 1.0

    def force(self, unit, r, dv, sigma):
        approach = jnp.sum(dv * unit, axis=0)
        return jnp.where(r < sigma, self.gamma * approach, 0.0) * unit

    def cutoff(self, sigma):
        return sigma


def restitution_coefficient(gamma: float, k: float, reduced_mass: float) -> float:
    """Coefficient of restitution of a linear spring-dashpot contact.

    Two particles with reduced mass ``m* = m_i m_j / (m_i + m_j)`` that touch
    with stiffness ``k`` (:class:`Harmonic`) and damping ``gamma``
    (:class:`NormalDamping`) separate with ``e`` times their approach speed,
    ``e = exp(-pi zeta / sqrt(1 - zeta^2))`` with the damping ratio
    ``zeta = gamma / (2 sqrt(k m*))``. For a particle and a fixed wall the
    reduced mass is the particle mass. The result is 0 for ``zeta >= 1``.
    """
    zeta = gamma / (2.0 * math.sqrt(k * reduced_mass))
    if zeta >= 1.0:
        return 0.0
    return math.exp(-math.pi * zeta / math.sqrt(1.0 - zeta * zeta))


def damping_for_restitution(restitution: float, k: float, reduced_mass: float) -> float:
    """Damping coefficient that gives the coefficient of restitution ``restitution``.

    The inverse of :func:`restitution_coefficient`.
    """
    if not 0.0 < restitution <= 1.0:
        raise ValueError("the coefficient of restitution must be in (0, 1]")
    log_e = math.log(restitution)
    zeta = -log_e / math.sqrt(math.pi**2 + log_e**2)
    return 2.0 * zeta * math.sqrt(k * reduced_mass)


@dataclasses.dataclass(frozen=True)
class PairThermostat(PairDissipation):
    """Pairwise thermostat of dissipative particle dynamics (DPD).

    Every pair closer than the contact distance ``sigma`` feels a dissipative
    force ``gamma w^2 (dv . n) n`` and a random force
    ``sqrt(2 gamma kT / dt) w theta n`` on particle i, with the weight
    ``w = 1 - r / sigma``, ``dv = v_j - v_i`` and ``n`` the unit vector from
    j to i. The random number ``theta`` has zero mean and unit variance and
    is the same for both particles of a pair, so both forces conserve
    momentum. The relation between the two amplitudes (the
    fluctuation-dissipation relation) keeps the temperature at ``kT``.
    Together with :class:`Harmonic` as
    conservative force, ``F = a (1 - r / sigma)`` with ``a = k sigma``, this
    is the DPD fluid of Groot & Warren (1997); particles of radius 0.5 have
    the usual cut-off of 1.

    The force field needs a random key for this term; see
    :func:`particlegym.physics.velocity_verlet`.

    Args:
        gamma: Friction coefficient.
        kT: Thermal energy.
        dt: Time step of the integrator.
    """

    gamma: float = 4.5
    kT: float = 1.0
    dt: float = 0.01

    def force(self, unit, r, dv, sigma):
        w = jnp.maximum(1.0 - r / sigma, 0.0)
        return self.gamma * w * w * jnp.sum(dv * unit, axis=0) * unit

    def random_force(self, unit, r, sigma, noise):
        """Random force on particle i for the symmetric pair noise ``noise``."""
        w = jnp.maximum(1.0 - r / sigma, 0.0)
        return math.sqrt(2.0 * self.gamma * self.kT / self.dt) * w * noise * unit

    def cutoff(self, sigma):
        return sigma
