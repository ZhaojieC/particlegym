"""The soft walker: the start, the reward, the actuators, friction and the baselines."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from particlegym import envs


def _rollout(env, params, policy, steps, key=0):
    """Run ``steps`` compiled control steps from a reset; ``policy(key, state)`` gives actions."""
    _, state = env.reset(jax.random.key(key), params)

    def body(s, k):
        obs, s, reward, terminated, truncated, info = env.step(k, s, policy(k, s), params)
        return s, (obs, reward, terminated, truncated, info)

    keys = jax.random.split(jax.random.key(key + 1), steps)
    return jax.jit(lambda s: jax.lax.scan(body, s, keys))(state)


def test_the_robot_starts_at_rest_and_stands_still_without_commands():
    env = envs.SoftWalker()
    params = env.default_params
    obs, state = env.reset(jax.random.key(0), params)
    assert env.body.n_nodes == 24 and env.n_actuators == 8
    assert obs.shape == (1, 3 + 4 * 24 + 8) == (1, env.observation_space.shape[0])
    centre, velocity = env.centre_of_mass(state.particles)
    np.testing.assert_allclose(centre[0], 0.0, atol=1e-12)
    np.testing.assert_allclose(velocity, 0.0)
    idle = lambda k, s: jnp.zeros((1, env.n_actuators))  # noqa: E731
    _, (_, reward, terminated, truncated, info) = _rollout(env, params, idle, 200)
    assert abs(float(jnp.sum(reward))) < 1e-3  # no motion along the ground
    assert abs(float(info["speed"][-1])) < 1e-6
    assert 1.6 < float(info["height"][-1]) < 1.7  # it settles on its legs
    assert not np.asarray(terminated).any() and not np.asarray(truncated).any()


def test_reward_is_the_forward_motion_of_the_centre_of_mass():
    env = envs.SoftWalker()
    params = env.default_params
    _, state = env.reset(jax.random.key(0), params)
    for _ in range(5):
        _, new, reward, *_ = env.step(None, state, env.gait_policy(state, params), params)
        before = env.centre_of_mass(state.particles)[0][0]
        after = env.centre_of_mass(new.particles)[0][0]
        np.testing.assert_allclose(reward, [after - before], rtol=1e-12)
        state = new


def test_activations_follow_the_commands_with_a_delay():
    env = envs.SoftWalker()
    params = env.default_params
    _, state = env.reset(jax.random.key(0), params)
    command = jnp.linspace(-2.0, 2.0, env.n_actuators)[None]
    _, state, *_ = env.step(None, state, command, params)
    # Commands are clipped to [-1, 1]; after one control step of 0.1 the
    # activation has covered 1 - exp(-0.1 / 0.2) of the way.
    expected = np.clip(np.asarray(command[0]), -1, 1) * (1.0 - math.exp(-0.5))
    np.testing.assert_allclose(state.activation, expected, rtol=1e-9)


def test_walking_needs_friction():
    env = envs.SoftWalker()
    for friction in (1.0, 0.0):
        params = env.default_params._replace(friction=friction)
        gait = lambda k, s, params=params: env.gait_policy(s, params)  # noqa: E731
        _, (_, reward, _, _, info) = _rollout(env, params, gait, 300)
        if friction:
            assert float(jnp.sum(reward)) > 10.0  # about 24 voxels in 30 time units
        else:
            # Without friction no horizontal force acts on the robot as a
            # whole: its centre of mass cannot leave its place.
            np.testing.assert_allclose(info["x"], 0.0, atol=1e-9)


def test_the_gait_walks_forward_and_random_commands_do_not():
    env = envs.SoftWalker()
    params = env.default_params
    gait = lambda k, s: env.gait_policy(s, params)  # noqa: E731
    _, (_, reward, _, truncated, _) = _rollout(env, params, gait, params.max_steps)
    # 30 to 45 voxels in an episode of 50 time units, depending on the start:
    # walking is chaotic.
    assert float(jnp.sum(reward)) > 20.0
    assert bool(truncated[-1, 0]) and not np.asarray(truncated[:-1]).any()
    for seed in range(3):
        random = lambda k, s: env.action_space.sample(k, (1,))  # noqa: E731
        _, (_, reward, *_) = _rollout(env, params, random, params.max_steps, key=seed)
        assert abs(float(jnp.sum(reward))) < 5.0


def test_reset_and_step_compile_and_batch_over_friction():
    env = envs.SoftWalker()
    params = jax.tree.map(lambda x: jnp.full(3, x), env.default_params)
    params = params._replace(friction=jnp.array([0.5, 1.0, 1.5]))
    keys = jax.random.split(jax.random.key(3), 3)
    obs, states = jax.jit(jax.vmap(env.reset))(keys, params)
    assert obs.shape == (3, 1, env.observation_space.shape[0])
    actions = jax.vmap(env.gait_policy)(states, params)
    out = jax.jit(jax.vmap(env.step))(keys, states, actions, params)
    assert out[2].shape == (3, 1) and out[3].shape == (3, 1)
    frame = env.render(jax.tree.map(lambda x: x[0], states))
    assert frame.ndim == 3 and frame.shape[2] == 3


def test_invalid_settings_are_rejected():
    with pytest.raises(ValueError, match="actuator"):
        envs.SoftWalker(layout=("SS",))
    with pytest.raises(ValueError, match="strength"):
        envs.SoftWalker(strength=1.5)
