"""Active particles: persistent random walk, steering, alignment."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from particlegym import physics as pp
from particlegym.physics import active


def test_active_brownian_particle_has_the_analytic_mean_square_displacement():
    n, speed, d_rot, dt, steps = 4000, 1.5, 1.0, 0.01, 1000
    field = pp.ForceField(pp.Box((0.0, 0.0), (1e4, 1e4)))
    theta = jax.random.uniform(jax.random.key(0), (n,), minval=-np.pi, maxval=np.pi)
    state = pp.create_state(jnp.full((n, 2), 5e3), speed * active.heading_vector(theta))
    state, _ = pp.init_forces(state, field)

    @jax.jit
    def run(s, theta, key):
        def body(carry, k):
            s, theta = carry
            # A strong drag makes the velocity equal to the self-propulsion velocity.
            s, _ = pp.langevin_baoab(
                s, field, dt, gamma=1e5, drift=speed * active.heading_vector(theta)
            )
            theta = active.rotate(theta, dt, diffusion=d_rot, key=k)
            return (s, theta), (s.pos, theta)

        return jax.lax.scan(body, (s, theta), jax.random.split(key, steps))[1]

    pos, angle = run(state, theta, jax.random.key(1))
    for step in (9, 99, 999):
        t = dt * (step + 1)
        measured = float(jnp.mean(jnp.sum((pos[step] - state.pos) ** 2, axis=1)))
        expected = 2 * speed**2 / d_rot**2 * (d_rot * t - 1 + np.exp(-d_rot * t))
        np.testing.assert_allclose(measured, expected, rtol=0.06)
        # The heading forgets its direction with the rate D_r.
        memory = float(jnp.mean(jnp.cos(angle[step] - theta)))
        np.testing.assert_allclose(memory, np.exp(-d_rot * t), atol=0.04)


def test_rotation_with_constant_angular_velocity_and_missing_key():
    theta = active.rotate(jnp.array([0.0, 3.0]), 0.5, omega=jnp.array([1.0, 1.0]))
    np.testing.assert_allclose(theta, [0.5, 3.5 - 2 * np.pi])
    with pytest.raises(ValueError, match="random key"):
        active.rotate(jnp.zeros(3), 0.1, diffusion=1.0)


def test_steering_relaxes_the_heading_to_the_target():
    rate, dt, steps, target = 2.0, 1e-3, 1500, 0.5
    start = jnp.array([2.5, -1.0, 0.4])

    def body(theta, _):
        return active.rotate(theta, dt, omega=active.steering_rate(theta, target, rate)), None

    final = jax.lax.scan(body, start, None, length=steps)[0]
    # Solution of d(theta)/dt = rate sin(target - theta).
    expected = target - 2 * jnp.arctan(jnp.tan((target - start) / 2) * np.exp(-rate * dt * steps))
    np.testing.assert_allclose(final, expected, rtol=2e-3)


def test_mean_heading_averages_over_neighbours_across_the_boundary():
    box = pp.Box((0.0, 0.0), (10.0, 10.0), periodic=True)
    pos = jnp.array([[0.2, 5.0], [9.7, 5.0], [5.0, 5.0], [5.0, 5.5]])
    theta = jnp.array([0.0, np.pi / 2, 1.0, 2.0])
    mean = active.mean_heading(pos, theta, box, radius=1.0)
    np.testing.assert_allclose(mean, [np.pi / 4, np.pi / 4, 1.5, 1.5], rtol=1e-12)
    alive = jnp.array([True, False, True, True])
    np.testing.assert_allclose(
        active.mean_heading(pos, theta, box, 1.0, alive)[0], 0.0, atol=1e-12
    )
    np.testing.assert_allclose(active.polar_order(jnp.array([0.3, 0.3, 0.3])), 1.0)
    np.testing.assert_allclose(active.polar_order(jnp.array([0.0, np.pi])), 0.0, atol=1e-12)
    np.testing.assert_allclose(
        active.polar_order(theta, alive), active.polar_order(theta[jnp.array([0, 2, 3])])
    )


def _vicsek_order(noise, n=400, side=10.0, speed=0.03, steps=600, seed=0):
    """Mean polar order of the Vicsek model over the second half of a run."""
    box = pp.Box((0.0, 0.0), (side, side), periodic=True)
    k_pos, k_theta, k_run = jax.random.split(jax.random.key(seed), 3)
    pos = jax.random.uniform(k_pos, (n, 2), maxval=side)
    theta = jax.random.uniform(k_theta, (n,), minval=-np.pi, maxval=np.pi)

    @jax.jit
    def run(pos, theta):
        def body(carry, key):
            pos, theta = carry
            turn = noise * (jax.random.uniform(key, (n,)) - 0.5)
            theta = active.wrap_angle(active.mean_heading(pos, theta, box, 1.0) + turn)
            pos = box.wrap(pos + speed * active.heading_vector(theta))
            return (pos, theta), active.polar_order(theta)

        return jax.lax.scan(body, (pos, theta), jax.random.split(k_run, steps))[1]

    return float(jnp.mean(run(pos, theta)[steps // 2 :]))


def test_alignment_orders_the_headings_at_low_noise_only():
    # Density 4, as in the model of Vicsek et al.: order at weak noise,
    # disorder when the headings are fully randomised at every step.
    assert _vicsek_order(noise=0.5) > 0.9
    assert _vicsek_order(noise=2 * np.pi) < 0.15
