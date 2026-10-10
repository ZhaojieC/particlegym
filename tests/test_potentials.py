"""Forces must be consistent with energies and reproduce the published force laws."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from particlegym import physics as pp


def _system_with_everything():
    """3D particles with several pair potentials, walls and obstacles."""
    rng = np.random.default_rng(3)
    n = 150
    radius = rng.uniform(0.3, 0.5, n)
    # Insert with smaller radii so that many particles overlap slightly.
    pos = pp.random_positions(rng, (0.2, 0.2, 0.2), (7.8, 7.8, 7.8), 0.8 * radius)
    box = pp.Box((0.0, 0.0, 0.0), (8.0, 8.0, 8.0))
    spheres = pp.Spheres([[4.0, 4.0, 4.0]], [1.0])
    boxes = pp.Boxes([[1.0, 1.0, 1.0]], [[2.0, 3.0, 2.0]])
    # Particles whose centres lie inside an obstacle are switched off; they
    # also exercise the alive mask.
    inside = (np.asarray(spheres.distance(jnp.asarray(pos))).min(axis=1) < 0.1) | (
        np.asarray(boxes.distance(jnp.asarray(pos))).min(axis=1) < 0.1
    )
    state = pp.create_state(pos, vel=rng.normal(size=(n, 3)), radius=radius, alive=~inside)
    social = pp.SocialForce(A=5.0, B=0.2, k=50.0, interaction_range=0.6)
    field = pp.ForceField(
        box,
        pair=[pp.WCA(epsilon=0.5), social, pp.Harmonic(k=20.0)],
        walls=[
            pp.Walls(pp.box_walls(box.lo, box.hi), pp.Harmonic(k=100.0)),
            pp.Walls(spheres, pp.WCA()),
            pp.Walls(boxes, social),
        ],
        neighbors=pp.CellList(cutoff=1.6, capacity=32),
    )
    field.check(state)
    return state, field


def test_forces_equal_minus_energy_gradient():
    state, field = _system_with_everything()
    force, info = jax.jit(field.forces)(state.pos, state.vel, state.radius, state.alive)
    energy, grad = jax.jit(jax.value_and_grad(field.potential_energy))(
        state.pos, state.radius, state.alive
    )

    assert not bool(info.overflow)
    scale = float(jnp.abs(force).max())
    assert scale > 1.0
    np.testing.assert_allclose(force, -grad, rtol=1e-10, atol=1e-10 * scale)
    np.testing.assert_allclose(info.potential_energy, energy, rtol=1e-12)
    assert np.all(np.asarray(force)[~np.asarray(state.alive)] == 0.0)


def test_energy_gradient_matches_finite_differences():
    state, field = _system_with_everything()
    force, _ = jax.jit(field.forces)(state.pos, state.vel, state.radius, state.alive)
    energy = jax.jit(field.potential_energy)
    h = 1e-6
    rng = np.random.default_rng(0)
    for i in rng.choice(np.flatnonzero(np.asarray(state.alive)), size=8, replace=False):
        for axis in range(3):
            shift = jnp.zeros_like(state.pos).at[i, axis].set(h)
            numeric = (
                energy(state.pos + shift, state.radius, state.alive)
                - energy(state.pos - shift, state.radius, state.alive)
            ) / (2 * h)
            np.testing.assert_allclose(-numeric, force[i, axis], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("distance", [0.45, 0.6, 0.75, 1.2])
def test_social_force_matches_published_law(distance):
    a, b, k = 2.0e3, 0.08, 1.2e5
    box = pp.Box((-5.0, -5.0), (5.0, 5.0))
    state = pp.create_state([[0.0, 0.0], [distance, 0.0]], radius=0.3)
    field = pp.ForceField(box, pair=[pp.SocialForce(A=a, B=b, k=k)])
    force, _ = field.forces(state.pos, state.vel, state.radius, state.alive)

    sigma = 0.6
    expected = a * np.exp((sigma - distance) / b) + k * max(sigma - distance, 0.0)
    np.testing.assert_allclose(force[1], [expected, 0.0], rtol=1e-12)
    np.testing.assert_allclose(force[0], [-expected, 0.0], rtol=1e-12)


def test_social_force_vanishes_beyond_interaction_range():
    box = pp.Box((-5.0, -5.0), (5.0, 5.0))
    state = pp.create_state([[0.0, 0.0], [1.61, 0.0]], radius=0.3)
    field = pp.ForceField(box, pair=[pp.SocialForce(interaction_range=1.0)])
    force, info = field.forces(state.pos, state.vel, state.radius, state.alive)
    assert float(jnp.abs(force).max()) == 0.0
    assert float(info.potential_energy) == 0.0


def test_wca_energy_and_force_vanish_at_cutoff():
    wca = pp.WCA(epsilon=2.0)
    sigma = 1.0
    cutoff = 2.0 ** (1.0 / 6.0)
    np.testing.assert_allclose(wca.energy(jnp.array(1.0), sigma), 2.0)
    below = jnp.array(cutoff * (1.0 - 1e-9))
    np.testing.assert_allclose(wca.energy(below, sigma), 0.0, atol=1e-12)
    np.testing.assert_allclose(jax.grad(wca.energy)(below, sigma), 0.0, atol=1e-6)
    assert float(wca.energy(jnp.array(cutoff * 1.01), sigma)) == 0.0


def test_pair_forces_obey_newtons_third_law():
    rng = np.random.default_rng(5)
    n = 200
    box = pp.Box((0.0, 0.0, 0.0), (6.0, 6.0, 6.0), periodic=True)
    state = pp.create_state(
        rng.uniform(0.0, 6.0, (n, 3)),
        vel=rng.normal(size=(n, 3)),
        radius=rng.uniform(0.3, 0.5, n),
    )
    field = pp.ForceField(
        box,
        pair=[
            pp.Harmonic(k=10.0),
            pp.SocialForce(A=5.0, B=0.2, k=50.0, interaction_range=0.6),
            pp.TangentialFriction(kappa=5.0),
        ],
        neighbors=pp.CellList(cutoff=1.6, capacity=32),
    )
    force, _ = jax.jit(field.forces)(state.pos, state.vel, state.radius, state.alive)
    scale = float(jnp.abs(force).max())
    np.testing.assert_allclose(jnp.sum(force, axis=0), 0.0, atol=1e-12 * n * scale)


def test_tangential_friction_conserves_momentum_and_only_dissipates():
    kappa = 7.0
    box = pp.Box((-5.0, -5.0, -5.0), (5.0, 5.0, 5.0))
    vel = np.array([[0.3, -1.0, 0.5], [-0.2, 0.4, 2.0]])
    state = pp.create_state([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], vel=vel, radius=0.3)
    field = pp.ForceField(box, pair=[pp.TangentialFriction(kappa=kappa)])
    force, _ = field.forces(state.pos, state.vel, state.radius, state.alive)

    np.testing.assert_allclose(force[0], -force[1], atol=1e-14)
    np.testing.assert_allclose(force[0, 0], 0.0, atol=1e-14)  # no normal component
    dv = vel[1] - vel[0]
    dv_tangential = dv - dv[0] * np.array([1.0, 0.0, 0.0])
    np.testing.assert_allclose(force[0], kappa * 0.1 * dv_tangential, rtol=1e-12)
    power = float(jnp.sum(force * state.vel))
    np.testing.assert_allclose(power, -kappa * 0.1 * np.sum(dv_tangential**2), rtol=1e-12)
