"""The orbit-transfer environment: orbits, thrust, fuel, episode ends and the Hohmann baseline."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from particlegym import envs
from particlegym import physics as pp


def _rollout(env, params, policy, steps, key=0):
    """Run ``steps`` compiled control steps from a reset; ``policy(state)`` gives the actions."""
    _, state = env.reset(jax.random.key(key), params)

    def body(s, k):
        obs, s, reward, terminated, truncated, info = env.step(k, s, policy(s), params)
        return s, (obs, reward, terminated, truncated, info)

    keys = jax.random.split(jax.random.key(key + 1), steps)
    return jax.jit(lambda s: jax.lax.scan(body, s, keys))(state)


def test_reset_puts_the_spacecraft_on_the_circular_orbit_and_coasting_keeps_it():
    env = envs.OrbitTransfer(n_agents=4)
    params = env.default_params
    obs, state = env.reset(jax.random.key(0), params)
    np.testing.assert_allclose(obs, [[1.0, 0.0, 1.0, 2.0, 1.0]] * 4, atol=1e-12)
    np.testing.assert_allclose(state.particles.pos[0], 0.0)  # the planet

    coast = lambda s: jnp.zeros((4, 2))  # noqa: E731
    state, (obs, reward, terminated, _, _) = _rollout(env, params, coast, 100)
    # Velocity Verlet with omega dt = 0.01 keeps the circle to (omega dt)^2.
    np.testing.assert_allclose(obs[:, :, 0], 1.0, atol=1e-4)  # radius, over 1.6 periods
    np.testing.assert_allclose(obs[:, :, 1], 0.0, atol=1e-4)  # radial velocity
    np.testing.assert_allclose(state.particles.pos[0], 0.0)
    assert np.all(np.asarray(reward) == 0.0) and not np.asarray(terminated).any()
    np.testing.assert_allclose(state.fuel_used, 0.0)


def test_thrust_acts_in_the_orbital_frame_and_spends_fuel():
    env = envs.OrbitTransfer(n_agents=2, action="discrete")
    params = env.default_params
    _, state = env.reset(jax.random.key(1), params)
    _, coasted, *_ = env.step(None, state, jnp.array([0, 0]), params)
    _, pushed, reward, *_ = env.step(None, state, jnp.array([1, 3]), params)
    radial, transverse = env._frame(state.particles)
    kick = pushed.particles.vel[1:] - coasted.particles.vel[1:]
    dv = params.max_thrust * env.control_interval  # 0.05
    # Prograde for the first spacecraft, radially out for the second, to first order in the step.
    np.testing.assert_allclose(kick[0], dv * transverse[0], atol=2e-3)
    np.testing.assert_allclose(kick[1], dv * radial[1], atol=2e-3)
    np.testing.assert_allclose(pushed.fuel_used, dv, rtol=1e-12)
    np.testing.assert_allclose(reward, -dv, rtol=1e-12)
    # A tenth of the thrust for actions 5 to 8.
    _, fine, reward, *_ = env.step(None, state, jnp.array([5, 6]), params)
    np.testing.assert_allclose(fine.fuel_used, 0.1 * dv, rtol=1e-12)


def test_the_thrust_stops_when_the_fuel_is_used_up():
    env = envs.OrbitTransfer(n_agents=1)
    params = env.default_params._replace(fuel=0.07)
    full = lambda s: jnp.array([[0.0, 1.0]])  # noqa: E731
    state, (_, reward, *_rest) = _rollout(env, params, full, 3)
    np.testing.assert_allclose(reward[:, 0], [-0.05, -0.02, 0.0], atol=1e-12)
    np.testing.assert_allclose(state.fuel_used, 0.07, rtol=1e-12)


def test_crashing_ends_the_episode_and_later_steps_give_nothing():
    env = envs.OrbitTransfer(n_agents=1, action="discrete")
    params = env.default_params._replace(fuel=10.0)
    retrograde = lambda s: jnp.array([2])  # noqa: E731
    _, (_, reward, terminated, truncated, info) = _rollout(env, params, retrograde, 60)
    crash = int(np.argmax(np.asarray(info["crashed"][:, 0])))
    assert bool(info["crashed"][crash, 0]) and 0 < crash < 60
    assert bool(terminated[crash, 0]) and not np.asarray(truncated).any()
    np.testing.assert_allclose(reward[crash + 1 :, 0], 0.0)  # no reward after the end


@pytest.mark.parametrize("target", [1.5, 2.0, 3.0, 4.0])
def test_hohmann_controller_reaches_the_orbit_with_little_more_than_the_impulsive_optimum(target):
    env = envs.OrbitTransfer(n_agents=1)
    params = env.default_params._replace(target_radius=target)
    state, (_, reward, _, _, info) = _rollout(
        env, params, lambda s: env.hohmann_policy(s, params), 200
    )
    arrival = int(np.argmax(np.asarray(info["arrived"][:, 0])))
    assert bool(info["arrived"][arrival, 0])
    used = float(state.fuel_used[0])
    optimum = pp.hohmann_delta_v(1.0, target)
    # The last step of the second burn makes the orbit circular at r1, and
    # burns of finite thrust centred on the apsides need a little more than
    # the impulsive burns of the optimum.
    assert optimum <= used <= 1.01 * optimum
    np.testing.assert_allclose(np.sum(reward[:, 0]), params.arrival_reward - used, rtol=1e-9)
    assert abs(float(info["semi_major_axis"][arrival, 0]) - target) < 0.02 * target
    assert float(info["eccentricity"][arrival, 0]) < 0.02


@pytest.mark.parametrize("target", [1.5, 2.0, 4.0])
def test_discrete_hohmann_controller_reaches_the_orbit_on_one_transfer(target):
    env = envs.OrbitTransfer(n_agents=1, action="discrete")
    params = env.default_params._replace(target_radius=target)
    state, (_, _, _, truncated, info) = _rollout(
        env, params, lambda s: env.hohmann_policy(s, params), params.max_steps
    )
    arrived = np.asarray(info["arrived"][:, 0])
    assert arrived.any() and not np.asarray(truncated).any()
    # Half a period of the transfer ellipse, plus a few steps for the burns:
    # the spacecraft arrives on the first pass through the apoapsis.
    half_period = np.pi * (0.5 * (1.0 + target)) ** 1.5 / env.control_interval
    assert arrived.argmax() + 1 < half_period + 40  # another revolution: over 100 steps
    used = float(state.fuel_used[0])
    optimum = pp.hohmann_delta_v(1.0, target)
    # Steps of a tenth of the thrust can stop as soon as the orbit is within
    # the tolerances, a little short of the circular orbit at r1, so a little
    # less than the optimum is possible.
    assert 0.97 * optimum <= used <= 1.01 * optimum


def test_reset_and_step_compile_and_batch_over_targets():
    env = envs.OrbitTransfer(n_agents=2)
    params = jax.tree.map(lambda x: jnp.full(3, x), env.default_params)
    params = params._replace(target_radius=jnp.array([1.5, 2.0, 3.0]))
    keys = jax.random.split(jax.random.key(3), 3)
    obs, states = jax.jit(jax.vmap(env.reset))(keys, params)
    assert obs.shape == (3, 2, 5)
    np.testing.assert_allclose(obs[:, 0, 3], [1.5, 2.0, 3.0])
    actions = jax.vmap(env.hohmann_policy)(states, params)
    out = jax.jit(jax.vmap(env.step))(keys, states, actions, params)
    assert out[2].shape == (3, 2) and out[3].shape == (3, 2)
    assert env.render(jax.tree.map(lambda x: x[0], states)).shape[2] == 3


def test_invalid_settings_are_rejected():
    with pytest.raises(ValueError, match="n_agents"):
        envs.OrbitTransfer(n_agents=0)
    with pytest.raises(ValueError, match="action"):
        envs.OrbitTransfer(action="impulse")
    with pytest.raises(ValueError, match="planet_radius"):
        envs.OrbitTransfer(planet_radius=1.5)
