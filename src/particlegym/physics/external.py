"""Forces that act on single particles, independent of the other particles."""

from __future__ import annotations

import abc
import dataclasses

import jax
import jax.numpy as jnp


class ExternalForce(abc.ABC):
    """A force on every particle that depends on its own position and velocity."""

    @abc.abstractmethod
    def force(self, pos: jax.Array, vel: jax.Array, mass: jax.Array) -> jax.Array:
        """(N, d) forces on particles at ``pos`` with velocities ``vel`` and masses ``mass``."""

    def energy(self, pos: jax.Array, mass: jax.Array) -> jax.Array:
        """(N,) potential energies; zero for forces without a potential."""
        return jnp.zeros(pos.shape[0], dtype=pos.dtype)


@dataclasses.dataclass(frozen=True)
class Gravity(ExternalForce):
    """Uniform gravity: the force on a particle of mass m is ``m g``.

    Args:
        g: Acceleration vector, one value per axis.
    """

    g: tuple[float, ...] = (0.0, -9.81)

    def force(self, pos, vel, mass):
        return mass[:, None] * jnp.asarray(self.g, dtype=pos.dtype)

    def energy(self, pos, mass):
        return -mass * (pos @ jnp.asarray(self.g, dtype=pos.dtype))


@dataclasses.dataclass(frozen=True)
class KineticFriction(ExternalForce):
    """Friction of constant strength against the motion, as for a body on a table.

    ``f = -m a v / sqrt(|v|^2 + v0^2)``: a particle that moves much faster
    than ``v0`` slows down with the constant deceleration ``a`` (the friction
    coefficient times the gravitational acceleration), and comes to rest
    smoothly below ``v0``. The smoothing replaces the jump of Coulomb
    friction at zero velocity, which an explicit integrator cannot follow.
    The time step must be smaller than ``2 v0 / a``.

    Args:
        deceleration: The deceleration ``a`` of a fast particle.
        speed_scale: The speed ``v0`` below which the friction weakens.
    """

    deceleration: float = 1.0
    speed_scale: float = 1e-2

    def force(self, pos, vel, mass):
        speed = jnp.sqrt(jnp.sum(vel * vel, axis=-1, keepdims=True) + self.speed_scale**2)
        return -mass[:, None] * self.deceleration * vel / speed
