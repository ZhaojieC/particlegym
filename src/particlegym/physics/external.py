# ParticleGym, https://github.com/ZhaojieC/particlegym (MIT license).
# Please cite it in published work: particlegym.cite() gives the references.
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


@dataclasses.dataclass(frozen=True)
class Ground(ExternalForce):
    """A flat floor with an elastic contact and Coulomb friction.

    The floor is where the last coordinate (y in 2D, z in 3D) equals ``level``.
    A particle closer to it than ``radius`` overlaps it by delta and is pushed
    out with the normal force ``F_n = max(k delta - c v_n, 0)``: an elastic
    contact with damping that never pulls. Friction opposes the sliding
    velocity ``v_t`` (the other components),

        F_t = -mu F_n v_t / sqrt(|v_t|^2 + v_s^2),

    Coulomb's law with a smooth transition below the slip speed ``v_s``, which
    an explicit integrator can follow (as for :class:`KineticFriction`). Well
    below ``v_s`` the friction acts like a stiff viscous friction and holds a
    foot nearly in place. The time step must resolve the contact, dt well
    below ``sqrt(m / k)``, and the friction, dt below ``m v_s / (mu F_n)``.

    Args:
        level: Height of the floor.
        stiffness: Contact stiffness k.
        damping: Damping c of the normal velocity during contact.
        friction: Friction coefficient mu.
        slip_speed: Slip speed v_s of the smooth transition.
        radius: Contact radius of the particles.
    """

    level: float = 0.0
    stiffness: float = 1000.0
    damping: float = 10.0
    friction: float = 1.0
    slip_speed: float = 0.1
    radius: float = 0.0

    def overlap(self, pos: jax.Array) -> jax.Array:
        """(N,) depths by which the particles overlap the floor."""
        return jnp.maximum(self.level + self.radius - pos[:, -1], 0.0)

    def force(self, pos, vel, mass):
        overlap = self.overlap(pos)
        push = jnp.maximum(self.stiffness * overlap - self.damping * vel[:, -1], 0.0)
        normal = jnp.where(overlap > 0.0, push, 0.0)
        sliding = vel[:, :-1]
        speed = jnp.sqrt(jnp.sum(sliding * sliding, axis=1, keepdims=True) + self.slip_speed**2)
        friction = -self.friction * normal[:, None] * sliding / speed
        return jnp.concatenate([friction, normal[:, None]], axis=1)

    def energy(self, pos, mass):
        return 0.5 * self.stiffness * self.overlap(pos) ** 2
