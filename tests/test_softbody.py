"""Soft bodies of voxels: the lattice, dashpots, the ground with Coulomb friction, the dynamics."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from particlegym import physics as pp

BOX = pp.Box((-1e3, -1e3), (1e3, 1e3))


def _spring(body, a, b):
    """Index of the spring between the corners at lattice points a and b."""
    nodes = {tuple(p): i for i, p in enumerate(np.rint(body.positions).astype(int).tolist())}
    pair = {nodes[a], nodes[b]}
    return next(s for s, (i, j) in enumerate(body.springs.tolist()) if {i, j} == pair)


def _field(body, command, *, gravity=0.0, ground=None, damping=0.0, k=100.0):
    external = []
    if gravity:
        external.append(pp.Gravity((0.0, -gravity)))
    if ground is not None:
        external.append(ground)
    if damping:
        external.append(pp.Dashpots(body.springs, damping))
    bonds = pp.Bonds(body.springs, k=body.stiffness(k), r0=body.rest_lengths(command, 0.25))
    return pp.ForceField(BOX, bonded=[bonds], external=external)


def _simulate(body, field_of, steps, dt=0.005, pos=None, vel=None):
    """Velocity Verlet; ``field_of(i)`` gives the force field of step i."""
    pos = jnp.asarray(body.positions if pos is None else pos)
    state = pp.create_state(pos, vel, mass=jnp.asarray(body.masses), radius=0.0)
    state, _ = pp.init_forces(state, field_of(0))

    def step(s, i):
        s, info = pp.velocity_verlet(s, field_of(i), dt)
        return s, (s.pos, s.vel, info.potential_energy)

    return jax.lax.scan(step, state, jnp.arange(1, steps + 1))


def test_voxels_share_corners_and_edges():
    body = pp.VoxelBody(("HS", "SV"))
    assert (body.n_nodes, body.n_voxels, len(body.springs)) == (9, 4, 12 + 8)
    assert body.actuators.tolist() == [0, 3]  # in reading order
    np.testing.assert_allclose(body.masses.sum(), 4.0)  # one per voxel
    centre = _spring(body, (1, 1), (1, 2))  # an inner edge
    assert body.edge_voxels[centre].tolist() != [body.edge_voxels[centre][0]] * 2
    np.testing.assert_allclose(sorted(body.masses), [0.25] * 4 + [0.5] * 4 + [1.0])
    # Without commands the rest lengths are the lattice distances: the body is at rest.
    rest = body.rest_lengths(jnp.zeros(2), 0.25)
    ends = body.positions[body.springs]
    np.testing.assert_allclose(rest, np.linalg.norm(ends[:, 0] - ends[:, 1], axis=1))


def test_actuators_lengthen_and_shorten_their_springs():
    body = pp.VoxelBody(("HS", "SV"))  # H at the top left, V at the bottom right
    rest = np.asarray(body.rest_lengths(jnp.array([1.0, -1.0]), 0.25))
    assert rest[_spring(body, (0, 2), (1, 2))] == pytest.approx(1.25)  # outer edge of H
    assert rest[_spring(body, (0, 1), (1, 1))] == pytest.approx(1.125)  # shared with S below
    assert rest[_spring(body, (0, 1), (0, 2))] == pytest.approx(1.0)  # vertical edge of H
    assert rest[_spring(body, (2, 0), (2, 1))] == pytest.approx(0.75)  # outer edge of V
    assert rest[_spring(body, (1, 0), (1, 1))] == pytest.approx(0.875)  # shared with S
    assert rest[_spring(body, (0, 1), (1, 2))] == pytest.approx(math.hypot(1.25, 1.0))
    assert rest[_spring(body, (1, 0), (2, 1))] == pytest.approx(math.hypot(1.0, 0.75))
    rigid = pp.VoxelBody(("RS",))
    k = rigid.stiffness(100.0, rigidity=3.0)
    assert k[_spring(rigid, (1, 0), (1, 1))] == pytest.approx(200.0)  # shared: the mean
    assert k[_spring(rigid, (0, 0), (1, 1))] == pytest.approx(300.0)  # diagonal of R
    assert k[_spring(rigid, (1, 0), (2, 1))] == pytest.approx(100.0)  # diagonal of S


@pytest.mark.parametrize("layout", [(), ("SS", "S"), ("SX",), ("S.S",), ("...",)])
def test_invalid_layouts_are_rejected(layout):
    with pytest.raises(ValueError):
        pp.VoxelBody(layout)


def test_dashpots_resist_only_changes_of_distance():
    key1, key2 = jax.random.split(jax.random.key(0))
    pos = jax.random.normal(key1, (6, 2))
    vel = jax.random.normal(key2, (6, 2))
    pairs = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0], [0, 3]])
    damper = pp.Dashpots(pairs, c=2.0)
    force = damper.force(pos, vel, None)
    np.testing.assert_allclose(force.sum(axis=0), 0.0, atol=1e-12)  # momentum
    torque = pos[:, 0] * force[:, 1] - pos[:, 1] * force[:, 0]
    np.testing.assert_allclose(torque.sum(), 0.0, atol=1e-12)  # angular momentum
    d = pos[pairs[:, 0]] - pos[pairs[:, 1]]
    rate = jnp.sum((vel[pairs[:, 0]] - vel[pairs[:, 1]]) * d, axis=1) / jnp.linalg.norm(d, axis=1)
    np.testing.assert_allclose(jnp.sum(force * vel), -2.0 * jnp.sum(rate**2), rtol=1e-12)
    # A rigid motion, translation plus rotation, is not damped.
    rigid = jnp.array([0.3, -0.7]) + 1.3 * jnp.stack([-pos[:, 1], pos[:, 0]], axis=1)
    np.testing.assert_allclose(damper.force(pos, rigid, None), 0.0, atol=1e-12)


def test_ground_pushes_out_with_the_gradient_of_its_energy():
    ground = pp.Ground(stiffness=1000.0, damping=0.0, friction=0.0)
    pos = jnp.array([[0.3, -0.01], [1.0, 0.2], [2.0, -0.003], [3.0, 0.0]])
    force = ground.force(pos, jnp.zeros_like(pos), None)
    grad = jax.grad(lambda p: jnp.sum(ground.energy(p, None)))(pos)
    np.testing.assert_allclose(force, -grad, atol=1e-12)
    np.testing.assert_allclose(force, [[0.0, 10.0], [0.0, 0.0], [0.0, 3.0], [0.0, 0.0]])
    # The damped contact never pulls a particle that leaves the floor quickly.
    leaving = pp.Ground(stiffness=1000.0, damping=50.0)
    assert float(leaving.force(pos[:1], jnp.array([[0.0, 5.0]]), None)[0, 1]) == 0.0


def test_ground_friction_follows_coulombs_law():
    ground = pp.Ground(stiffness=1000.0, damping=0.0, friction=0.5, slip_speed=0.01)
    pos = jnp.array([[0.0, -0.02], [1.0, -0.02], [2.0, -0.02]])
    vel = jnp.array([[2.0, 0.0], [-0.5, 0.0], [0.0, 0.0]])
    force = ground.force(pos, vel, None)
    np.testing.assert_allclose(force[:, 1], 20.0)  # normal force k delta
    np.testing.assert_allclose(force[:2, 0], [-10.0, 10.0], rtol=1e-3)  # mu F_n, opposing
    assert float(force[2, 0]) == 0.0  # no sliding, no friction
    # In 3D the friction opposes the horizontal velocity; z is up.
    ground3 = pp.Ground(stiffness=1000.0, friction=0.5, slip_speed=0.01)
    f3 = ground3.force(jnp.array([[0.0, 0.0, -0.02]]), jnp.array([[3.0, 4.0, 0.0]]), None)
    np.testing.assert_allclose(f3, [[-6.0, -8.0, 20.0]], rtol=1e-3)


def test_a_block_slides_to_rest_after_the_coulomb_distance():
    # A particle of unit mass slides on the ground under gravity g = 1 with
    # mu = 0.5: it decelerates at mu g and stops after v^2 / (2 mu g) = 4.
    ground = pp.Ground(stiffness=1e4, damping=100.0, friction=0.5, slip_speed=0.01)
    field = pp.ForceField(BOX, external=[pp.Gravity((0.0, -1.0)), ground])
    state = pp.create_state(jnp.array([[0.0, -1e-4]]), jnp.array([[2.0, 0.0]]), radius=0.0)
    state, _ = pp.init_forces(state, field)

    def step(s, _):
        return pp.velocity_verlet(s, field, 0.001)[0], None

    state, _ = jax.lax.scan(step, state, None, length=8000)  # 8 time units
    assert float(state.pos[0, 0]) == pytest.approx(4.0, rel=0.01)
    assert abs(float(state.vel[0, 0])) < 0.02


def test_a_floating_body_cannot_move_its_centre_of_mass():
    # Without gravity and ground all forces are internal: the commands change
    # the shape, but momentum and angular momentum stay zero.
    body = pp.VoxelBody(("HHHHH", "SSSSS", "V.V.V"))
    commands = jax.random.uniform(jax.random.key(1), (100, body.n_actuators), minval=-1, maxval=1)

    def field_of(i):
        return _field(body, commands[(i - 1) // 20], damping=5.0)

    _, (pos, vel, _) = _simulate(body, field_of, 2000)
    mass = jnp.asarray(body.masses)[None, :, None]
    momentum = jnp.sum(mass * vel, axis=1)
    np.testing.assert_allclose(momentum, 0.0, atol=1e-11)
    centre = jnp.sum(mass * pos, axis=1) / mass.sum()
    np.testing.assert_allclose(centre - centre[0], 0.0, atol=1e-10)
    cross = pos[..., 0] * vel[..., 1] - pos[..., 1] * vel[..., 0]
    angular = jnp.sum(mass[..., 0] * cross, axis=1)
    np.testing.assert_allclose(angular, 0.0, atol=1e-10)
    assert float(jnp.max(jnp.abs(vel))) > 0.1  # the body does move its parts


def test_without_damping_a_falling_body_conserves_energy():
    # A deformed body with a rigid voxel in free fall: velocity Verlet keeps
    # the energy within fluctuations of order dt^2.
    body = pp.VoxelBody(("SSS", "SRS"))
    start = body.positions + 0.05 * jax.random.normal(jax.random.key(2), body.positions.shape)
    field = _field(body, jnp.zeros(0), gravity=1.0)

    def fluctuation(dt):
        steps = round(5.0 / dt)
        _, (_, vel, potential) = _simulate(body, lambda i: field, steps, dt=dt, pos=start)
        kinetic = 0.5 * jnp.sum(jnp.asarray(body.masses)[None, :, None] * vel**2, axis=(1, 2))
        energy = kinetic + potential
        return float(jnp.max(jnp.abs(energy - energy[0]))), float(jnp.max(kinetic))

    coarse, _ = fluctuation(0.004)
    fine, kinetic = fluctuation(0.002)
    assert fine < 2e-4 * kinetic
    assert 3.6 < coarse / fine < 4.4


def test_a_dropped_body_comes_to_rest_and_the_ground_carries_its_weight():
    body = pp.VoxelBody(("SSS", "SSS"))
    ground = pp.Ground(stiffness=3000.0, damping=40.0)
    field = _field(body, jnp.zeros(0), gravity=1.0, ground=ground, damping=5.0)
    start = body.positions + jnp.array([0.0, 0.5])
    final, _ = _simulate(body, lambda i: field, 6000, pos=start)  # 30 time units
    np.testing.assert_allclose(final.vel, 0.0, atol=1e-6)
    carried = jnp.sum(ground.force(final.pos, final.vel, None)[:, 1])
    assert float(carried) == pytest.approx(body.masses.sum(), rel=1e-6)  # weight m g
