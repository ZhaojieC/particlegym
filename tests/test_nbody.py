"""Newtonian gravity: forces, the restricted problem, Kepler orbits and Hohmann transfers."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from particlegym import physics as pp

BOX3 = pp.Box((-100.0,) * 3, (100.0,) * 3)
BOX2 = pp.Box((-100.0,) * 2, (100.0,) * 2)


def _run(state, field, dt, steps):
    """Velocity Verlet for ``steps`` steps; returns the final state and the trajectory."""
    state, _ = pp.init_forces(state, field)

    def body(s, _):
        s, _ = pp.velocity_verlet(s, field, dt)
        return s, (s.pos, s.vel)

    return jax.jit(lambda s: jax.lax.scan(body, s, None, length=steps))(state)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("softening", [0.0, 0.3])
def test_forces_are_minus_the_gradient_of_the_energy(dim, softening):
    gravity = pp.NewtonGravity(G=1.7, softening=softening)
    pos = jax.random.normal(jax.random.key(dim), (7, dim))
    mass = jnp.linspace(0.5, 2.0, 7)
    force = gravity.force(pos, None, mass)
    grad = jax.grad(lambda p: jnp.sum(gravity.energy(p, mass)))(pos)
    np.testing.assert_allclose(force, -grad, atol=1e-12)
    np.testing.assert_allclose(force.sum(axis=0), 0.0, atol=1e-12)  # Newton's third law
    # The pair formula directly, for one pair.
    d = pos[0] - pos[1]
    pair = -1.7 * mass[0] * mass[1] / jnp.sqrt(d @ d + softening**2)
    two = pp.NewtonGravity(G=1.7, softening=softening)
    np.testing.assert_allclose(jnp.sum(two.energy(pos[:2], mass[:2])), pair, rtol=1e-12)


def test_test_particles_feel_the_sources_and_do_not_act_back():
    pos = jax.random.normal(jax.random.key(1), (6, 3))
    mass = jnp.array([3.0, 2.0, 1.0, 1.0, 0.5, 0.5])
    restricted = pp.NewtonGravity(softening=0.1, sources=2)
    full = pp.NewtonGravity(softening=0.1)
    force = restricted.force(pos, None, mass)
    # The two sources feel only each other.
    np.testing.assert_allclose(force[:2], full.force(pos[:2], None, mass[:2]), atol=1e-12)
    # A test particle feels the sources only: minus the gradient of its own energy.
    for i in range(2, 6):
        grad = jax.grad(lambda x, i=i: restricted.energy(pos.at[i].set(x), mass)[i])(pos[i])
        np.testing.assert_allclose(force[i], -grad, atol=1e-12)
    # The acceleration of a test particle, summed directly over the two sources.
    p, m = np.asarray(pos), np.asarray(mass)
    for i in range(2, 6):
        d = p[i] - p[:2]
        r2 = np.sum(d * d, axis=1) + 0.01
        expected = -np.sum(m[:2, None] * d / r2[:, None] ** 1.5, axis=0)
        np.testing.assert_allclose(force[i] / mass[i], expected, rtol=1e-12)


def test_two_bodies_on_a_circular_orbit_keep_keplers_period_and_conserve_energy():
    m1, m2, separation = 1.0, 0.5, 1.0
    gm = m1 + m2  # G = 1
    period = 2 * math.pi * math.sqrt(separation**3 / gm)
    speed = math.sqrt(gm / separation)  # relative speed
    # Positions and velocities about the centre of mass.
    pos = jnp.array([[-m2 / gm * separation, 0.0, 0.0], [m1 / gm * separation, 0.0, 0.0]])
    vel = jnp.array([[0.0, -m2 / gm * speed, 0.0], [0.0, m1 / gm * speed, 0.0]])
    field = pp.ForceField(BOX3, external=[pp.NewtonGravity()])
    state = pp.create_state(pos, vel, mass=jnp.array([m1, m2]), radius=0.0)
    steps = 2000
    final, (positions, velocities) = _run(state, field, period / steps, steps)

    relative = positions[:, 1] - positions[:, 0]
    distance = jnp.linalg.norm(relative, axis=1)
    np.testing.assert_allclose(distance, separation, rtol=1e-5)  # the orbit stays circular
    np.testing.assert_allclose(final.pos, pos, atol=1e-4)  # back after one period
    mass = jnp.array([m1, m2])
    kinetic = 0.5 * jnp.sum(mass[None, :, None] * velocities**2, axis=(1, 2))
    potential = -m1 * m2 / distance
    energy = kinetic + potential
    np.testing.assert_allclose(energy, energy[0], rtol=1e-8)
    momentum = jnp.sum(mass[None, :, None] * velocities, axis=1)
    np.testing.assert_allclose(momentum, 0.0, atol=1e-12)
    angular = jnp.sum(mass[None, :, None] * jnp.cross(positions, velocities), axis=1)
    np.testing.assert_allclose(angular - angular[0], 0.0, atol=1e-12)


def test_an_elliptic_orbit_keeps_its_elements_and_period():
    a, e, gm = 1.5, 0.6, 1.0
    periapsis = a * (1 - e)
    speed = math.sqrt(gm * (1 + e) / periapsis)
    period = 2 * math.pi * math.sqrt(a**3 / gm)
    # A massive body at rest and a test particle (restricted problem).
    pos = jnp.array([[0.0, 0.0], [periapsis, 0.0]])
    vel = jnp.array([[0.0, 0.0], [0.0, speed]])
    field = pp.ForceField(BOX2, external=[pp.NewtonGravity(sources=1)])
    state = pp.create_state(pos, vel, mass=jnp.array([1.0, 1.0]), radius=0.0)
    steps = 20000
    final, (positions, velocities) = _run(state, field, period / steps, steps)

    semi_major, ecc = pp.orbital_elements(positions[:, 1], velocities[:, 1], gm)
    np.testing.assert_allclose(semi_major, a, rtol=1e-4)
    np.testing.assert_allclose(jnp.linalg.norm(ecc, axis=1), e, atol=1e-4)
    np.testing.assert_allclose(ecc[:, 1], 0.0, atol=1e-4)  # the periapsis stays on the x axis
    np.testing.assert_allclose(final.pos[1], pos[1], atol=1e-3)  # back at the periapsis
    np.testing.assert_allclose(final.pos[0], 0.0, atol=0.0)  # the source does not move


def test_a_ring_of_test_particles_circles_a_source():
    radius, n, eps = 2.0, 36, 0.05
    angle = 2 * math.pi * jnp.arange(n) / n
    ring = radius * jnp.stack([jnp.cos(angle), jnp.sin(angle)], axis=1)
    # Circular speed in the softened field of a unit mass: v^2 = R^2 / (R^2 + eps^2)^(3/2).
    speed = math.sqrt(radius**2 / (radius**2 + eps**2) ** 1.5)
    tangent = jnp.stack([-jnp.sin(angle), jnp.cos(angle)], axis=1)
    pos = jnp.concatenate([jnp.zeros((1, 2)), ring])
    vel = jnp.concatenate([jnp.zeros((1, 2)), speed * tangent])
    # Massive test particles: they must not pull on the source or on each other.
    field = pp.ForceField(BOX2, external=[pp.NewtonGravity(softening=eps, sources=1)])
    state = pp.create_state(pos, vel, mass=jnp.full(n + 1, 1.0), radius=0.0)
    period = 2 * math.pi * radius / speed
    steps = 4000
    final, (positions, _) = _run(state, field, period / steps, steps)
    distance = jnp.linalg.norm(positions[:, 1:], axis=2)
    np.testing.assert_allclose(distance, radius, rtol=1e-5)
    np.testing.assert_allclose(final.pos, pos, atol=2e-4)


def test_hohmann_delta_v():
    # Raising a circular orbit from r = 1 to r = 2 (G M = 1), from the vis-viva equation.
    first = math.sqrt(4 / 3) - 1
    second = math.sqrt(1 / 2) * (1 - math.sqrt(2 / 3))
    assert pp.hohmann_delta_v(1.0, 2.0) == pytest.approx(first + second, rel=1e-12)
    assert pp.hohmann_delta_v(2.0, 1.0) == pytest.approx(first + second, rel=1e-12)
    assert pp.hohmann_delta_v(1.0, 1.0) == 0.0
