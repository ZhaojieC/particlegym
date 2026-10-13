"""Contacts: restitution, moving bodies and shapes that are switched off."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from particlegym import physics as pp

BOX = pp.Box((0.0, 0.0), (10.0, 10.0))
K = 1.0e5


@pytest.mark.parametrize("restitution", [0.9, 0.6, 0.3])
def test_two_body_collision_has_the_set_restitution(restitution):
    masses = jnp.array([1.0, 3.0])
    reduced = 1.0 * 3.0 / 4.0
    gamma = pp.damping_for_restitution(restitution, K, reduced)
    np.testing.assert_allclose(pp.restitution_coefficient(gamma, K, reduced), restitution)
    field = pp.ForceField(BOX, pair=[pp.Harmonic(K), pp.NormalDamping(gamma)])
    state = pp.create_state(jnp.array([[4.0, 5.0], [6.0, 5.0]]), mass=masses, radius=0.5)
    state, _ = pp.init_forces(state._replace(vel=jnp.array([[2.0, 0.0], [-1.0, 0.0]])), field)

    @jax.jit
    def run(s):
        def body(s, _):
            return pp.velocity_verlet(s, field, 1e-5)[0], None

        return jax.lax.scan(body, s, None, length=60_000)[0]

    final = run(state)
    np.testing.assert_allclose(pp.total_momentum(final), pp.total_momentum(state), atol=1e-10)
    approach = state.vel[0, 0] - state.vel[1, 0]
    separation = final.vel[1, 0] - final.vel[0, 0]
    # The damping force starts abruptly at contact, so the error of the
    # integration is of first order in the time step.
    np.testing.assert_allclose(separation / approach, restitution, rtol=3e-3)
    # The kinetic energy of the relative motion drops by the factor e^2.
    loss = pp.kinetic_energy(state) - pp.kinetic_energy(final)
    np.testing.assert_allclose(loss, 0.5 * reduced * approach**2 * (1 - restitution**2), rtol=3e-3)


def test_restitution_helpers_reject_bad_values():
    assert pp.restitution_coefficient(1e9, K, 1.0) == 0.0  # overdamped contact
    np.testing.assert_allclose(pp.damping_for_restitution(1.0, K, 1.0), 0.0, atol=1e-12)
    with pytest.raises(ValueError):
        pp.damping_for_restitution(0.0, K, 1.0)


def test_bounce_off_a_damped_wall():
    mass, restitution = 0.5, 0.8
    gamma = pp.damping_for_restitution(restitution, K, mass)
    walls = pp.Walls(pp.box_walls(BOX.lo, BOX.hi), pp.Harmonic(K), damping=gamma)
    field = pp.ForceField(BOX, walls=[walls])
    state = pp.create_state(jnp.array([[5.0, 1.0]]), mass=mass, radius=0.3)
    state, _ = pp.init_forces(state._replace(vel=jnp.array([[0.7, -3.0]])), field)

    @jax.jit
    def run(s):
        def body(s, _):
            return pp.velocity_verlet(s, field, 1e-5)[0], None

        return jax.lax.scan(body, s, None, length=40_000)[0]

    final = run(state)
    # The normal velocity is reversed and reduced; the tangential one is unchanged.
    np.testing.assert_allclose(final.vel[0], [0.7, 3.0 * restitution], rtol=1e-3)


def _bounce(shapes_at, velocity, steps=30_000, dt=1e-5):
    """Final velocity of a particle with the given velocity that is hit by a moving shape."""
    state = pp.create_state(jnp.array([[5.0, 5.0]]), radius=0.25)
    state = state._replace(vel=jnp.array([velocity]))

    def field_at(time):
        return pp.ForceField(BOX, walls=[pp.Walls(shapes_at(time), pp.Harmonic(K))])

    state, _ = pp.init_forces(state, field_at(0.0))

    @jax.jit
    def run(s):
        def body(s, step):
            return pp.velocity_verlet(s, field_at((step + 1) * dt), dt)[0], None

        return jax.lax.scan(body, s, jnp.arange(steps))[0]

    return run(state).vel[0]


def test_bounce_off_a_moving_wall():
    speed = 1.5

    def plate(time):  # a vertical wall that starts at x = 4.5 and moves to the right
        x = 4.5 + speed * time
        return pp.Plates(jnp.array([[x, 0.0]]), jnp.array([[[0.0, 10.0]]]))

    # In the frame of the wall the normal velocity is reversed: v' = 2V - v.
    np.testing.assert_allclose(_bounce(plate, [-0.5, 0.2]), [2 * speed + 0.5, 0.2], rtol=1e-3)


def test_bounce_off_a_moving_disc():
    speed = 2.0

    def disc(time):  # a paddle of radius 1 that hits the particle head-on
        return pp.Spheres(jnp.array([[3.5 + speed * time, 5.0]]), 1.0)

    velocity = _bounce(disc, [-0.5, 0.0], steps=40_000)
    np.testing.assert_allclose(velocity, [2 * speed + 0.5, 0.0], rtol=1e-3, atol=1e-9)


def test_shapes_built_from_traced_arrays_match_fixed_shapes():
    pos = jax.random.uniform(jax.random.key(1), (40, 2), minval=1.0, maxval=9.0)
    state = pp.create_state(pos, radius=0.4)
    lo, hi = jnp.array([[2.0, 2.0], [6.0, 5.0]]), jnp.array([[3.5, 4.0], [8.0, 6.0]])
    centres, radii = jnp.array([[5.0, 5.0]]), jnp.array([1.2])
    origins, edges = jnp.array([[1.0, 8.0]]), jnp.array([[[6.0, 1.0]]])
    potential = pp.Harmonic(50.0)

    def forces(lo, hi, centres, radii, origins, edges):
        walls = [
            pp.Walls(pp.Boxes(lo, hi), potential),
            pp.Walls(pp.Spheres(centres, radii), potential),
            pp.Walls(pp.Plates(origins, edges), potential),
        ]
        field = pp.ForceField(BOX, walls=walls)
        return field.forces(state.pos, state.vel, state.radius, state.alive)[0]

    fixed = forces(lo, hi, centres, radii, origins, edges)
    traced = jax.jit(forces)(lo, hi, centres, radii, origins, edges)
    assert float(jnp.abs(fixed).max()) > 1.0
    np.testing.assert_allclose(traced, fixed, rtol=1e-12, atol=1e-12)


def test_inactive_shapes_do_not_act():
    bricks = pp.Boxes([[2.0, 2.0], [6.0, 2.0]], [[4.0, 3.0], [8.0, 3.0]])
    state = pp.create_state(jnp.array([[3.0, 3.1], [7.0, 3.1]]), radius=0.2)
    state = state._replace(vel=jnp.array([[1.0, 0.0], [1.0, 0.0]]))

    def forces(active):
        walls = pp.Walls(bricks, pp.Harmonic(100.0), friction=5.0, damping=2.0, active=active)
        field = pp.ForceField(BOX, walls=[walls])
        return field.forces(state.pos, state.vel, state.radius, state.alive)

    force, info = jax.jit(forces)(jnp.array([True, False]))
    assert float(force[0, 1]) > 0 and float(force[0, 0]) < 0  # pushed up, slowed by friction
    np.testing.assert_array_equal(force[1], 0.0)
    both, info_both = forces(None)
    np.testing.assert_allclose(both[1], force[0])
    np.testing.assert_allclose(info_both.potential_energy, 2 * info.potential_energy)


def test_wall_friction_uses_the_velocity_relative_to_the_shape():
    plate = pp.Plates.from_segments([[(0.0, 1.0), (10.0, 1.0)]])
    state = pp.create_state(jnp.array([[5.0, 1.2]]), radius=0.3)
    state = state._replace(vel=jnp.array([[2.0, 0.5]]))

    def dissipative(velocity):
        walls = pp.Walls(plate, pp.Harmonic(0.0), friction=10.0, damping=3.0, velocity=velocity)
        return walls.dissipative_force(state.pos, state.vel, state.radius, state.alive)[0]

    # Overlap 0.1: friction 10 * 0.1 * v_t against the sliding, damping 3 * v_n.
    np.testing.assert_allclose(dissipative(None), [-2.0, -1.5], rtol=1e-12)
    # A wall that moves with the particle exerts no dissipative force.
    np.testing.assert_allclose(dissipative(jnp.array([[2.0, 0.5]])), [0.0, 0.0], atol=1e-12)
    # A conveyor belt drags the particle along.
    np.testing.assert_allclose(dissipative(jnp.array([[5.0, 0.0]])), [3.0, -1.5], rtol=1e-12)
