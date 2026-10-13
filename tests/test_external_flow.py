"""External forces and flow fields: free fall, terminal velocity, friction, streamlines."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from particlegym import physics as pp

BOX = pp.Box((0.0, 0.0), (10.0, 10.0))


def test_free_fall_follows_the_parabola():
    g, dt, steps = 9.81, 1e-3, 1000
    field = pp.ForceField(BOX, external=[pp.Gravity((0.0, -g))])
    state = pp.create_state(jnp.array([[2.0, 9.0], [5.0, 8.0]]), mass=jnp.array([1.0, 3.0]))
    state = state._replace(vel=jnp.array([[1.0, 0.0], [0.0, 2.0]]))
    start, _ = pp.init_forces(state, field)

    @jax.jit
    def run(s):
        def body(s, _):
            s, info = pp.velocity_verlet(s, field, dt)
            return s, pp.kinetic_energy(s) + info.potential_energy

        return jax.lax.scan(body, s, None, length=steps)

    final, energy = run(start)
    t = dt * steps
    expected = start.pos + start.vel * t + 0.5 * jnp.array([0.0, -g]) * t**2
    # Velocity Verlet is exact for a constant force; the mass does not matter.
    np.testing.assert_allclose(final.pos, expected, rtol=1e-12)
    np.testing.assert_allclose(energy, energy[0], rtol=1e-12)


def test_bouncing_ball_conserves_energy():
    field = pp.ForceField(
        BOX,
        walls=[pp.Walls(pp.box_walls(BOX.lo, BOX.hi), pp.Harmonic(k=1e4))],
        external=[pp.Gravity((0.0, -9.81))],
    )
    state = pp.create_state(jnp.array([[5.0, 3.0]]), radius=0.3)
    state, info = pp.init_forces(state._replace(vel=jnp.array([[1.5, 0.0]])), field)
    first = pp.kinetic_energy(state) + info.potential_energy

    @jax.jit
    def run(s):
        def body(s, _):
            s, info = pp.velocity_verlet(s, field, 2e-4)
            return s, (pp.kinetic_energy(s) + info.potential_energy, s.pos[0, 1])

        return jax.lax.scan(body, s, None, length=40_000)

    _, (energy, height) = run(state)
    np.testing.assert_allclose(energy, first, rtol=1e-4)
    # The ball bounces several times and returns to the height where it started.
    assert int(np.sum(np.diff(np.sign(np.diff(height))) > 0)) >= 4
    np.testing.assert_allclose(height.max(), 3.0, atol=1e-3)


def test_terminal_velocity_in_a_flow():
    g, gamma, dt = 9.81, 2.0, 1e-3
    flow = pp.UniformFlow((0.5, 0.2))
    field = pp.ForceField(pp.Box((0.0, 0.0), (1e3, 1e3)), external=[pp.Gravity((0.0, -g))])
    state, _ = pp.init_forces(pp.create_state(jnp.array([[500.0, 900.0]]), mass=2.5), field)

    @jax.jit
    def run(s):
        def body(s, _):
            s, _ = pp.langevin_baoab(s, field, dt, gamma=gamma, drift=flow.velocity(s.pos))
            return s, None

        return jax.lax.scan(body, s, None, length=20_000)[0]

    final = run(state)
    # Stokes drag m gamma (v - u) balances the weight: v = u + g / gamma.
    np.testing.assert_allclose(final.vel[0], [0.5, 0.2 - g / gamma], rtol=1e-5)


def test_kinetic_friction_stops_a_sliding_body():
    a, v0, dt = 0.5, 1e-3, 1e-3
    field = pp.ForceField(pp.Box((0.0, 0.0), (100.0, 100.0)), external=[pp.KineticFriction(a, v0)])
    state = pp.create_state(jnp.array([[10.0, 50.0]]), mass=0.17)
    state, _ = pp.init_forces(state._replace(vel=jnp.array([[2.0, 0.0]])), field)

    @jax.jit
    def run(s):
        def body(s, _):
            s, _ = pp.velocity_verlet(s, field, dt)
            return s, s.vel[0, 0]

        return jax.lax.scan(body, s, None, length=6000)

    final, speed = run(state)
    # Constant deceleration a: the body stops after v / a = 4 s and v^2 / (2a) = 4 m.
    np.testing.assert_allclose(speed[1999], 2.0 - a * 2.0, rtol=1e-3)
    np.testing.assert_allclose(final.pos[0, 0] - 10.0, 2.0**2 / (2 * a), rtol=5e-3)
    assert abs(float(final.vel[0, 0])) < 1e-6


def test_flow_fields_are_consistent():
    pos = jax.random.uniform(jax.random.key(0), (50, 2), minval=-3.0, maxval=3.0)
    flow = pp.TaylorGreen(u0=1.3, k=0.7)

    def divergence(point):
        return jnp.trace(jax.jacfwd(lambda p: flow.velocity(p[None])[0])(point))

    np.testing.assert_allclose(jax.vmap(divergence)(pos), 0.0, atol=1e-12)
    # The closed-form vorticity equals the curl obtained by differentiation.
    np.testing.assert_allclose(flow.vorticity(pos), pp.FlowField.vorticity(flow, pos), rtol=1e-12)
    # The velocity is the curl of the stream function: u = (d psi/dy, -d psi/dx).
    grad = jax.vmap(jax.grad(lambda p: flow.stream_function(p[None])[0]))(pos)
    expected = jnp.stack([grad[:, 1], -grad[:, 0]], axis=1)
    np.testing.assert_allclose(flow.velocity(pos), expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(jnp.max(jnp.linalg.norm(flow.velocity(pos), axis=1)) <= 0.65, True)

    shear = pp.ShearFlow(rate=2.0, origin=1.0)
    np.testing.assert_allclose(shear.velocity(jnp.array([[0.3, 2.5]])), [[3.0, 0.0]])
    np.testing.assert_allclose(shear.vorticity(pos), -2.0, rtol=1e-12)
    with pytest.raises(ValueError, match="2D flows"):
        shear.vorticity(jnp.zeros((1, 3)))


def test_tracer_follows_the_streamlines_of_the_vortex_flow():
    flow = pp.TaylorGreen(u0=1.0, k=1.0)
    side = 2 * np.pi
    field = pp.ForceField(pp.Box((0.0, 0.0), (side, side), periodic=True))
    pos = jnp.array([[0.3, 0.2], [1.0, 0.4], [2.5, 2.9]])
    state, _ = pp.init_forces(pp.create_state(pos), field)
    dt = 2e-3

    @jax.jit
    def run(s):
        def body(s, _):
            # A strong drag keeps the particle velocity at the fluid velocity.
            s, _ = pp.langevin_baoab(s, field, dt, gamma=1e4, drift=flow.velocity(s.pos))
            return s, flow.stream_function(s.pos)

        return jax.lax.scan(body, s, None, length=20_000)

    final, psi = run(state)
    np.testing.assert_allclose(
        psi, np.broadcast_to(flow.stream_function(pos), psi.shape), atol=5e-3
    )
    assert float(jnp.linalg.norm(final.pos - pos, axis=1).max()) > 0.1  # the tracers have moved
