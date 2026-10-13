"""Bonded terms: consistency of forces, ideal chain, bending, area of a ring."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from particlegym import physics as pp


def _ring(n, radius, centre=(5.0, 5.0)):
    angle = 2 * np.pi * np.arange(n) / n
    return np.asarray(centre) + radius * np.stack([np.cos(angle), np.sin(angle)], axis=1)


def _ring_terms(n, side, area, k_bond=200.0, k_area=50.0, k_bend=2.0):
    index = np.arange(n)
    pairs = np.stack([index, np.roll(index, -1)], axis=1)
    triplets = np.stack([np.roll(index, 1), index, np.roll(index, -1)], axis=1)
    return [
        pp.Bonds(pairs, k=k_bond, r0=side),
        pp.Angles(triplets, k=k_bend, theta0=np.pi),
        pp.AreaConstraint(index[None, :], k=k_area, area0=area),
    ]


def test_forces_are_the_gradient_of_the_energy_and_sum_to_zero():
    box = pp.Box((0.0, 0.0), (6.0, 6.0), periodic=True)
    n = 12
    pos = jnp.asarray(_ring(n, 1.5, centre=(5.5, 0.4)))  # the ring crosses two boundaries
    pos = box.wrap(pos + 0.1 * jax.random.normal(jax.random.key(0), (n, 2)))
    state = pp.create_state(pos, radius=0.1)
    field = pp.ForceField(box, bonded=_ring_terms(n, side=0.7, area=5.0))
    force, info = field.forces(state.pos, state.vel, state.radius, state.alive)
    grad = jax.grad(field.potential_energy)(state.pos, state.radius, state.alive)
    np.testing.assert_allclose(force, -grad, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(
        info.potential_energy, field.potential_energy(state.pos, state.radius, state.alive)
    )
    np.testing.assert_allclose(jnp.sum(force, axis=0), 0.0, atol=1e-9)
    assert float(jnp.abs(force).max()) > 1.0
    # A term is off while one of its particles is dead.
    dead = state.alive.at[0].set(False)
    force_dead, _ = field.forces(state.pos, state.vel, state.radius, dead)
    np.testing.assert_array_equal(force_dead[0], 0.0)
    two_bonds = pp.ForceField(box, bonded=[pp.Bonds([[0, 1], [2, 3]], k=3.0, r0=0.0)])
    force_two, _ = two_bonds.forces(state.pos, state.vel, state.radius, dead)
    np.testing.assert_array_equal(force_two[:2], 0.0)
    assert float(jnp.abs(force_two[2:4]).max()) > 0.1


def test_area_of_a_ring():
    box = pp.Box((0.0, 0.0), (4.0, 4.0), periodic=True)
    square = jnp.array([[3.5, 3.5], [0.5, 3.5], [0.5, 0.5], [3.5, 0.5]])  # across the corner
    constraint = pp.AreaConstraint([[0, 1, 2, 3], [3, 2, 1, 0]], area0=1.0)
    np.testing.assert_allclose(constraint.area(square, box), [1.0, -1.0])
    with pytest.raises(ValueError, match="2D only"):
        constraint.area(jnp.zeros((4, 3)), pp.Box((0.0,) * 3, (4.0,) * 3))
    with pytest.raises(ValueError):
        pp.Bonds([[0, 1, 2]])


def test_ring_conserves_energy_and_relaxes_to_its_rest_shape():
    n, radius = 16, 1.0
    side = 2 * radius * np.sin(np.pi / n)
    area = 0.5 * n * radius**2 * np.sin(2 * np.pi / n)
    box = pp.Box((0.0, 0.0), (10.0, 10.0))
    terms = _ring_terms(n, side, area)
    field = pp.ForceField(box, bonded=terms)
    # Start from an ellipse with the same number of particles.
    pos = (_ring(n, radius) - 5.0) * np.array([1.3, 0.75]) + 5.0
    state, info = pp.init_forces(pp.create_state(jnp.asarray(pos), radius=0.05), field)
    first = float(info.potential_energy)
    assert first > 1.0

    @jax.jit
    def run(s, gamma):
        def body(s, _):
            s, info = pp.langevin_baoab(s, field, 2e-3, gamma=gamma)
            return s, pp.kinetic_energy(s) + info.potential_energy

        return jax.lax.scan(body, s, None, length=20_000)

    _, energy = run(state, 0.0)
    np.testing.assert_allclose(energy, first, rtol=1e-3)

    relaxed, energy = run(state, 2.0)
    # The rest shape is the regular polygon. Bonds and area are relaxed there;
    # the bending energy of its n equal corners remains.
    np.testing.assert_allclose(energy[-1], n * 2.0 * (1 - np.cos(2 * np.pi / n)), rtol=1e-6)
    np.testing.assert_allclose(terms[2].area(relaxed.pos, box), [area], rtol=1e-5)
    bond = jnp.linalg.norm(relaxed.pos - jnp.roll(relaxed.pos, -1, axis=0), axis=1)
    np.testing.assert_allclose(bond, side, rtol=1e-4)


def test_signed_rest_angle_in_2d():
    box = pp.Box((0.0, 0.0), (10.0, 10.0))
    field = pp.ForceField(
        box,
        bonded=[
            pp.Bonds([[0, 1], [1, 2]], k=100.0, r0=1.0),
            pp.Angles([[0, 1, 2]], k=5.0, theta0=np.pi / 2),
        ],
    )
    pos = jnp.array([[6.0, 5.0], [5.0, 5.0], [4.2, 4.6]])
    state, _ = pp.init_forces(pp.create_state(pos, radius=0.1), field)

    @jax.jit
    def relax(s):
        def body(s, _):
            return pp.langevin_baoab(s, field, 2e-3, gamma=3.0)[0], None

        return jax.lax.scan(body, s, None, length=20_000)[0]

    final = relax(state)
    a, b = final.pos[0] - final.pos[1], final.pos[2] - final.pos[1]
    angle = jnp.arctan2(a[0] * b[1] - a[1] * b[0], a @ b)
    # Counter-clockwise from the first to the second bond, not clockwise.
    np.testing.assert_allclose(angle, np.pi / 2, atol=1e-5)
    # Near the rest angle the energy is (k / 2) (theta - theta0)^2.
    bent = pp.Angles([[0, 1, 2]], k=5.0, theta0=np.pi)
    straight = jnp.array([[0.0, 0.0], [1.0, 0.0], [1.0 + np.cos(0.02), np.sin(0.02)]])
    energy = bent.energy(straight, box, jnp.ones(3, bool))
    np.testing.assert_allclose(energy, 0.5 * 5.0 * 0.02**2, rtol=1e-3)
    straight_3d = jnp.concatenate([straight, jnp.zeros((3, 1))], axis=1)
    box_3d = pp.Box((0.0,) * 3, (10.0,) * 3)
    tilted = pp.Angles([[0, 1, 2]], k=5.0, theta0=np.pi - 0.02)
    np.testing.assert_allclose(
        tilted.energy(straight_3d, box_3d, jnp.ones(3, bool)), 0.0, atol=1e-9
    )


def test_ideal_chain_has_the_gaussian_end_to_end_distance():
    beads, chains, k, kT, dt = 20, 400, 3.0, 1.0, 0.02
    box = pp.Box((0.0, 0.0), (1e3, 1e3))
    index = np.arange(beads - 1)
    field = pp.ForceField(box, bonded=[pp.Bonds(np.stack([index, index + 1], axis=1), k=k)])
    state, _ = pp.init_forces(pp.create_state(jnp.full((beads, 2), 500.0), radius=0.1), field)

    @jax.jit
    @jax.vmap
    def run(key):
        def body(s, key):
            s, _ = pp.langevin_baoab(s, field, dt, gamma=1.0, kT=kT, key=key)
            end_to_end = jnp.sum((s.pos[-1] - s.pos[0]) ** 2)
            return s, (end_to_end, pp.temperature(s))

        _, (r2, temperature) = jax.lax.scan(body, state, jax.random.split(key, 8000))
        return jnp.mean(r2[3000:]), jnp.mean(temperature[3000:])

    r2, temperature = run(jax.random.split(jax.random.key(0), chains))
    # <R^2> = (n - 1) d kT / k for n beads joined by springs of zero rest length.
    np.testing.assert_allclose(float(jnp.mean(r2)), (beads - 1) * 2 * kT / k, rtol=0.05)
    np.testing.assert_allclose(float(jnp.mean(temperature)), kT, rtol=0.01)
