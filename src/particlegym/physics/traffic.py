# ParticleGym, https://github.com/ZhaojieC/particlegym (MIT license).
# Please cite it in published work: particlegym.cite() gives the references.
"""Car-following models of traffic in one lane of a ring road.

A car-following model gives the acceleration of a vehicle from its gap to
the vehicle ahead (bumper to bumper), its speed, and the speed of the
vehicle ahead. The vehicles drive in one lane and do not overtake, so on a
ring road the vehicle ahead of vehicle i is vehicle i + 1, and the last one
follows the first. Positions are arc lengths in [0, L).

:class:`IntelligentDriver` is the model of Treiber, Hennecke & Helbing
(2000) and :class:`OptimalVelocity` the one of Bando et al. (1995).
:func:`ballistic_step` advances the vehicles with the ballistic update, which
Treiber & Kanagaraj (2015) found more accurate than Euler's method of the
same order and well suited to vehicles that stop. :func:`ring_modes` gives
the linear stability of uniform flow: whether small disturbances grow into
stop-and-go waves.
"""

from __future__ import annotations

import abc
import dataclasses
import math

import jax
import jax.numpy as jnp


class CarFollowing(abc.ABC):
    """A car-following model: the acceleration of a driver."""

    @abc.abstractmethod
    def acceleration(self, gap, speed, leader_speed):
        """Accelerations at the given gaps, speeds and speeds of the vehicles ahead."""

    @abc.abstractmethod
    def equilibrium_speed(self, gap):
        """Speed of uniform flow at the given gap, at which the acceleration vanishes."""


@dataclasses.dataclass(frozen=True)
class IntelligentDriver(CarFollowing):
    """The intelligent driver model (IDM) of Treiber, Hennecke & Helbing (2000).

        dv/dt = a [1 - (v / v0)^delta - (s*(v, w) / s)^2],
        s*(v, w) = s0 + max(0, v T + v w / (2 sqrt(a b))),

    with the gap s, the speed v and the approach rate w = v - v_ahead. A
    driver accelerates with at most a towards the desired speed v0, keeps
    the time headway T and the minimum gap s0, and brakes with about b in
    normal situations, harder when a collision threatens: the braking term
    grows without bound as the gap closes, so that without noise a vehicle
    does not run into the one ahead. The defaults are those of the Flow
    framework (Wu et al. 2022).

    Args:
        v0: Desired speed, m/s.
        T: Time headway, s.
        a: Maximum acceleration, m/s^2.
        b: Comfortable deceleration, m/s^2.
        delta: Exponent of the free-road term.
        s0: Minimum gap, m.
    """

    v0: float = 30.0
    T: float = 1.0
    a: float = 1.0
    b: float = 1.5
    delta: float = 4.0
    s0: float = 2.0

    def __post_init__(self) -> None:
        if min(self.v0, self.T, self.a, self.b, self.delta) <= 0 or self.s0 < 0:
            raise ValueError("v0, T, a, b and delta must be positive and s0 not negative")

    def desired_gap(self, speed, leader_speed):
        """The dynamic desired gap s*(v, w)."""
        approach = speed - leader_speed
        dynamic = speed * self.T + speed * approach / (2.0 * math.sqrt(self.a * self.b))
        return self.s0 + jnp.maximum(dynamic, 0.0)

    def acceleration(self, gap, speed, leader_speed):
        free = 1.0 - (speed / self.v0) ** self.delta
        return self.a * (free - (self.desired_gap(speed, leader_speed) / gap) ** 2)

    def equilibrium_speed(self, gap):
        """Solves 1 - (v / v0)^delta = ((s0 + v T) / s)^2 by bisection; 0 for s <= s0."""
        gap = jnp.asarray(gap)

        def excess(speed):  # decreases from 1 - (s0 / s)^2 at v = 0 to below 0 at v = v0
            return 1.0 - (speed / self.v0) ** self.delta - ((self.s0 + speed * self.T) / gap) ** 2

        def bisect(_, bounds):
            low, high = bounds
            middle = 0.5 * (low + high)
            above = excess(middle) > 0.0
            return jnp.where(above, middle, low), jnp.where(above, high, middle)

        low = jnp.zeros_like(gap)
        low, high = jax.lax.fori_loop(0, 64, bisect, (low, jnp.full_like(gap, self.v0)))
        return jnp.where(gap > self.s0, 0.5 * (low + high), 0.0)


@dataclasses.dataclass(frozen=True)
class OptimalVelocity(CarFollowing):
    """The optimal velocity model of Bando et al. (1995).

        dv/dt = k [V(s) - v],   V(s) = v_max [tanh(s / w - 2) + tanh 2] / (1 + tanh 2).

    A driver relaxes at the rate k to the optimal velocity V of its gap,
    which rises from 0 at contact to v_max for large gaps, with the steepest
    rise at s = 2 w; the shape of V is that of Bando et al. Uniform flow is
    unstable where V'(s) > k / 2. The model can lead to collisions.

    In the form of :func:`~particlegym.physics.langevin_baoab` the driver
    relaxes to the drift velocity V(s) with the friction rate k, as a
    pedestrian of the social force model relaxes to its desired velocity.

    Args:
        k: Sensitivity, 1/s.
        v_max: Optimal velocity at large gaps, m/s.
        w: Width of the rise of V, m.
    """

    k: float = 1.0
    v_max: float = 30.0
    w: float = 5.0

    def __post_init__(self) -> None:
        if min(self.k, self.v_max, self.w) <= 0:
            raise ValueError("k, v_max and w must be positive")

    def optimal_velocity(self, gap):
        """V(s)."""
        rise = jnp.tanh(gap / self.w - 2.0) + math.tanh(2.0)
        return self.v_max * rise / (1.0 + math.tanh(2.0))

    def acceleration(self, gap, speed, leader_speed):
        return self.k * (self.optimal_velocity(gap) - speed)

    def equilibrium_speed(self, gap):
        return self.optimal_velocity(jnp.asarray(gap))


def ring_gaps(position: jax.Array, ring_length, vehicle_length) -> jax.Array:
    """(N,) gaps from every vehicle to the one ahead on a ring of the given length.

    The vehicles must be ordered along the road. A single vehicle follows
    itself, one ring length ahead.
    """
    if position.shape[0] == 1:
        return jnp.asarray(ring_length, position.dtype)[None] - vehicle_length
    distance = jnp.mod(jnp.roll(position, -1) - position, ring_length)
    return distance - vehicle_length


def ring_accelerations(
    model: CarFollowing, position: jax.Array, speed: jax.Array, ring_length, vehicle_length
) -> jax.Array:
    """(N,) accelerations of the drivers of all vehicles on a ring."""
    gap = ring_gaps(position, ring_length, vehicle_length)
    return model.acceleration(gap, speed, jnp.roll(speed, -1))


def ballistic_step(
    position: jax.Array, speed: jax.Array, acceleration: jax.Array, dt, ring_length
) -> tuple[jax.Array, jax.Array]:
    """Advance vehicles with a constant acceleration over the step.

        v' = v + a dt,   x' = x + v dt + a dt^2 / 2,

    except for a vehicle that would come to a stop within the step: it stops
    after the braking distance v^2 / (2 |a|) and stays at rest. Speeds never
    become negative. Positions are wrapped into [0, ring_length).

    Returns:
        ``(position, speed)`` after the step.
    """
    new_speed = speed + acceleration * dt
    stops = new_speed < 0.0
    braking = jnp.where(stops, acceleration, -1.0)
    advance = jnp.where(
        stops, -(speed**2) / (2.0 * braking), speed * dt + 0.5 * acceleration * dt**2
    )
    return jnp.mod(position + advance, ring_length), jnp.maximum(new_speed, 0.0)


def ring_modes(model: CarFollowing, n_vehicles: int, ring_length, vehicle_length) -> jax.Array:
    """Linear stability of uniform flow on a ring road.

    In uniform flow all N vehicles keep the gap s = L / N - l at the
    equilibrium speed. A small displacement y_n = exp(i theta n + lambda t)
    of vehicle n, with theta = 2 pi k / N, grows or decays at the rate Re
    lambda, where lambda solves

        lambda^2 - lambda (f_v + f_u z) - f_s (z - 1) = 0,   z = exp(i theta),

    and f_s, f_v, f_u are the derivatives of the acceleration with respect to
    the gap, the own speed and the speed of the vehicle ahead, evaluated by
    automatic differentiation. Uniform flow is linearly unstable if a mode
    has Re lambda > 0; disturbances then grow into stop-and-go waves.

    Args:
        model: The car-following model of all drivers.
        n_vehicles: N.
        ring_length: L.
        vehicle_length: l.

    Returns:
        (N - 1, 2) complex rates lambda of the modes k = 1, ..., N - 1; the
        mode k = 0 (moving all vehicles together) is neutral.
    """
    gap = jnp.asarray(ring_length / n_vehicles - vehicle_length, dtype=float)
    speed = model.equilibrium_speed(gap)
    f_s, f_v, f_u = jax.grad(model.acceleration, argnums=(0, 1, 2))(gap, speed, speed)
    k = jnp.arange(1, n_vehicles)
    z = jnp.exp(2j * jnp.pi * k / n_vehicles)
    b = f_v + f_u * z
    root = jnp.sqrt(b * b + 4.0 * f_s * (z - 1.0))
    return jnp.stack([0.5 * (b + root), 0.5 * (b - root)], axis=-1)
