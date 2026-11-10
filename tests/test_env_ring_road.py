"""The ring road: the initial wave, rewards, the safety rule, collisions and the baselines."""

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

    keys = jax.random.split(jax.random.key(key + 1000), steps)
    return jax.jit(lambda s: jax.lax.scan(body, s, keys))(state)


def test_reset_starts_in_a_wave_with_every_vehicle_at_its_equilibrium_speed():
    env = envs.RingRoad()
    params = env.default_params
    obs, state = env.reset(jax.random.key(0), params)
    gaps = np.asarray(pp.ring_gaps(state.position, 230.0, 5.0))
    mean_gap = 230.0 / 22 - 5.0
    # The gaps are a cosine of relative amplitude 0.5 around the ring.
    spectrum = np.fft.fft(gaps - mean_gap)
    np.testing.assert_allclose(np.abs(spectrum[1]) * 2 / 22, 0.5 * mean_gap, rtol=1e-10)
    np.testing.assert_allclose(np.abs(spectrum[2:-1]), 0.0, atol=1e-10)
    np.testing.assert_allclose(gaps.mean(), mean_gap, rtol=1e-12)
    np.testing.assert_allclose(state.speed, env.driver.equilibrium_speed(gaps), rtol=1e-12)
    # The agent sees its speed, the speed of the vehicle ahead and the gap to it.
    np.testing.assert_allclose(obs, [[state.speed[0], state.speed[1], gaps[0]]], rtol=1e-12)


def test_uniform_flow_without_fluctuations_pays_the_distance_driven():
    env = envs.RingRoad(n_agents=2)
    params = env.default_params._replace(noise=0.0, wave=0.0)
    speed = float(env.uniform_speed(params))  # 3.45 m/s
    _, (obs, reward, terminated, truncated, _) = _rollout(
        env, params, lambda s: env.idm_policy(s, params), 100
    )
    np.testing.assert_allclose(reward, speed * env.control_interval, rtol=1e-9)
    np.testing.assert_allclose(obs[..., :2], speed, rtol=1e-9)
    np.testing.assert_allclose(obs[..., 2], 230.0 / 22 - 5.0, rtol=1e-9)
    assert not np.asarray(terminated).any() and not np.asarray(truncated).any()


def test_reward_is_the_mean_distance_driven_in_the_step():
    env = envs.RingRoad()
    params = env.default_params
    _, state = env.reset(jax.random.key(1), params)
    _, new, reward, *_ = env.step(jax.random.key(2), state, jnp.array([[0.3]]), params)
    moved = jnp.mod(new.position - state.position, params.ring_length)
    np.testing.assert_allclose(reward, [jnp.mean(moved)], rtol=1e-12)


def test_the_safety_rule_caps_the_acceleration_at_the_one_of_a_human_driver():
    env = envs.RingRoad(substeps=1)
    params = env.default_params
    _, state = env.reset(jax.random.key(3), params)
    gaps = pp.ring_gaps(state.position, params.ring_length, env.vehicle_length)
    human = float(env.driver.acceleration(gaps[0], state.speed[0], state.speed[1]))
    for command in (1.0, -0.5):
        _, new, *_ = env.step(jax.random.key(4), state, jnp.array([[command]]), params)
        accel = (float(new.speed[0]) - float(state.speed[0])) / env.dt
        assert accel == pytest.approx(min(command, human), abs=1e-9)
    # Full throttle for a whole episode: no collision.
    full = lambda s: jnp.ones((1, 1))  # noqa: E731
    _, (_, _, terminated, _, info) = _rollout(env, params, full, 600)
    assert not np.asarray(terminated).any() and float(jnp.min(info["min_gap"])) > 0.0


def test_without_the_safety_rule_full_throttle_crashes_and_ends_the_episode():
    env = envs.RingRoad(safe=False)
    params = env.default_params
    full = lambda s: jnp.ones((1, 1))  # noqa: E731
    state, (_, reward, terminated, truncated, info) = _rollout(env, params, full, 200)
    crash = int(np.argmax(np.asarray(terminated[:, 0])))
    assert bool(terminated[crash, 0]) and 0 < crash < 199
    assert float(info["min_gap"][crash]) < 0.0
    np.testing.assert_allclose(reward[crash + 1 :], 0.0)  # the vehicles stand still
    np.testing.assert_allclose(state.speed, 0.0)
    assert not np.asarray(truncated).any()


def test_discrete_actions_are_accelerations():
    env = envs.RingRoad(action="discrete", safe=False, substeps=1)
    params = env.default_params
    _, state = env.reset(jax.random.key(5), params)
    for index, level in enumerate(env.levels):
        _, new, *_ = env.step(jax.random.key(6), state, jnp.array([index]), params)
        accel = (float(new.speed[0]) - float(state.speed[0])) / env.dt
        assert accel == pytest.approx(level * env.max_accel, abs=1e-9)
    assert np.asarray(env.smoothing_policy(state, params)).shape == (1,)


@pytest.mark.parametrize("action", ["continuous", "discrete"])
def test_a_gently_cruising_vehicle_dissolves_the_stop_and_go_wave(action):
    env = envs.RingRoad(action=action)
    params = env.default_params
    results = {}
    for name in ("idm_policy", "smoothing_policy"):
        policy = getattr(env, name)
        returns, spread = [], []
        for seed in range(3):
            _, (_, reward, terminated, _, info) = _rollout(
                env, params, lambda s, policy=policy: policy(s, params), 600, key=seed
            )
            assert not np.asarray(terminated).any()
            returns.append(float(jnp.sum(reward[:, 0])))
            spread.append(float(jnp.mean(info["speed_std"][300:])))  # the last 150 s
        results[name] = np.mean(returns), np.mean(spread)
    human, smooth = results["idm_policy"], results["smoothing_policy"]
    assert human[1] > 2.0  # the jam persists when the vehicle drives like a human
    assert smooth[1] < 0.5  # and is gone with the gently cruising vehicle
    assert smooth[0] > 1.15 * human[0]  # every vehicle drives more than 15 % farther


def test_reset_and_step_compile_and_batch_over_ring_lengths():
    env = envs.RingRoad(n_agents=2)
    params = jax.tree.map(lambda x: jnp.full(3, x), env.default_params)
    params = params._replace(ring_length=jnp.array([220.0, 230.0, 260.0]))
    keys = jax.random.split(jax.random.key(7), 3)
    obs, states = jax.jit(jax.vmap(env.reset))(keys, params)
    assert obs.shape == (3, 2, 3)
    actions = jax.vmap(env.smoothing_policy)(states, params)
    out = jax.jit(jax.vmap(env.step))(keys, states, actions, params)
    assert out[2].shape == (3, 2) and out[3].shape == (3, 2)
    frame = env.render(jax.tree.map(lambda x: x[0], states), params=env.default_params)
    assert frame.ndim == 3 and frame.shape[2] == 3


def test_invalid_settings_are_rejected():
    with pytest.raises(ValueError, match="n_agents"):
        envs.RingRoad(n_vehicles=3, n_agents=4)
    with pytest.raises(ValueError, match="action"):
        envs.RingRoad(action="lane change")
    env = envs.RingRoad()
    _, state = env.reset(jax.random.key(0))
    with pytest.raises(ValueError, match="random key"):
        env.step(None, state, jnp.zeros((1, 1)))
