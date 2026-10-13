"""Microswimmer: interface, equations of motion, coarse states, naive gyrotaxis."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from particlegym.envs import Microswimmer

QUIET = {"rotational_diffusion": 0.0, "translational_diffusion": 0.0}


def _place(state, pos, theta):
    pos = jnp.asarray(pos, state.particles.pos.dtype).reshape(-1, 2)
    theta = jnp.asarray(theta, pos.dtype).reshape(-1)
    return state._replace(particles=state.particles._replace(pos=pos), theta=theta)


def _run(env, state, action, params, steps, seed=0):
    def body(s, k):
        _, s, reward, _, _, _ = env.step(k, s, action, params)
        return s, (reward, s.theta)

    return jax.jit(lambda s: jax.lax.scan(body, s, jax.random.split(jax.random.key(seed), steps)))(
        state
    )


def test_interface_shapes_and_batching():
    env = Microswimmer(n_agents=3)
    params = env.default_params
    assert env.observation_space.shape == (3,) and env.action_space.n == 4
    obs, state = jax.jit(env.reset)(jax.random.key(0), params)
    assert obs.shape == (3, 3) and state.theta.shape == (3,)
    assert bool(env.observation_space.contains(obs[0]))
    action = env.action_space.sample(jax.random.key(1), (3,))
    obs, state, reward, terminated, truncated, info = jax.jit(env.step)(
        jax.random.key(2), state, action, params
    )
    assert obs.shape == (3, 3) and reward.shape == terminated.shape == truncated.shape == (3,)
    np.testing.assert_allclose(reward, info["altitude"])
    assert not bool(terminated.any()) and not bool(truncated.any())
    _, _, _, _, truncated, _ = env.step(
        jax.random.key(2), state, action, params._replace(max_steps=2)
    )
    assert bool(truncated.all())

    keys = jax.random.split(jax.random.key(3), 5)
    _, batch = jax.vmap(env.reset, in_axes=(0, None))(keys, params)
    actions = env.action_space.sample(jax.random.key(4), (5, 3))
    out = jax.vmap(env.step, in_axes=(0, 0, 0, None))(keys, batch, actions, params)
    assert out[0].shape == (5, 3, 3) and out[2].shape == (5, 3)

    coarse = Microswimmer(n_agents=2, observation="discrete", action="continuous")
    assert coarse.observation_space.n == 12 and coarse.action_space.shape == (2,)
    obs, state = coarse.reset(jax.random.key(0))
    assert obs.shape == (2,) and obs.dtype == jnp.int32
    np.testing.assert_allclose(
        coarse.preferred_angle(jnp.array([[0.0, 1.0], [-1.0, 0.0]])), [np.pi / 2, np.pi]
    )
    np.testing.assert_allclose(env.preferred_angle(jnp.arange(3)), np.pi * np.array([0.5, 1, 1.5]))
    with pytest.raises(ValueError, match="observation"):
        Microswimmer(observation="pixels")


def test_swimmer_in_still_fluid_turns_and_swims_as_the_equations_say():
    env = Microswimmer()
    params = env.default_params._replace(flow_speed=0.0, alignment_time=0.7, **QUIET)
    _, state = env.reset(jax.random.key(0), params)
    state = _place(state, [1.0, 2.0], -2.0)
    # The preferred direction is up (action 0): theta_k = pi / 2.
    final, (reward, theta) = _run(env, state, jnp.array([0]), params, 60)
    time = 0.1 * np.arange(1, 61)
    # d(theta)/dt = sin(theta_k - theta) / (2 B) has the solution
    # tan((theta_k - theta) / 2) = tan((theta_k - theta_0) / 2) exp(-t / (2 B)).
    exact = np.pi / 2 - 2 * np.arctan(np.tan((np.pi / 2 + 2.0) / 2) * np.exp(-time / 1.4))
    error = np.angle(np.exp(1j * (np.asarray(theta[:, 0]) - exact)))  # angles are wrapped
    # The heading is advanced with Euler steps of 0.01: the error stays below 3.2e-3 rad.
    np.testing.assert_allclose(error, 0.0, atol=5e-3)
    # The swimmer moves at the swimming speed along its heading.
    np.testing.assert_allclose(reward[-1, 0], 0.3 * 0.1 * np.sin(exact[-1]), rtol=2e-3)
    np.testing.assert_allclose(final.displacement[0, 1], reward.sum(), rtol=1e-6)
    path = 0.3 * 0.01 * 600
    assert float(np.linalg.norm(final.displacement[0])) < path


def test_flow_carries_and_turns_the_swimmer():
    env = Microswimmer(n_agents=2)
    # No swimming and no steering: a tracer of the flow.
    params = env.default_params._replace(swim_speed=0.0, alignment_time=1e9, **QUIET)
    _, state = env.reset(jax.random.key(0), params)
    # The first swimmer is at the centre of a vortex, where the fluid is at
    # rest and the vorticity is u0 k = 1; the second is on a closed streamline.
    state = _place(state, [[0.0, 0.0], [0.6, 0.3]], [0.2, 0.0])
    flow = env.flow(params)
    stream = float(flow.stream_function(state.particles.pos)[1])
    final, (_, theta) = _run(env, state, jnp.array([0, 0]), params, 100)
    np.testing.assert_allclose(final.displacement[0], 0.0, atol=1e-9)
    np.testing.assert_allclose(theta[-1, 0], 0.2 + 0.5 * 10.0 - 2 * np.pi, rtol=1e-6)
    np.testing.assert_allclose(flow.stream_function(final.particles.pos)[1], stream, rtol=1e-2)
    assert float(np.linalg.norm(final.displacement[1])) > 0.2


def test_coarse_states():
    env = Microswimmer(n_agents=4, observation="discrete")
    params = env.default_params
    _, state = env.reset(jax.random.key(0), params)
    # Vorticity cos(x) cos(y): 1 at (0, 0), -1 at (pi, 0), 0 at (pi / 2, 0), 0.4 at (acos(0.4), 0).
    pos = [[0.0, 0.0], [np.pi, 0.0], [np.pi / 2, 0.0], [np.arccos(0.4), 0.0]]
    state = _place(state, pos, [np.pi / 2, 0.1, -np.pi / 2 + 0.2, np.pi - 0.3])
    # 4 * (negative, near zero, positive) + (up, left, down, right)
    np.testing.assert_array_equal(env.discrete_state(state, params), [8, 3, 6, 9])
    np.testing.assert_array_equal(env._observe(state, params), [8, 3, 6, 9])
    wide = Microswimmer(n_agents=4, vorticity_threshold=0.5)
    np.testing.assert_array_equal(wide.discrete_state(state, params), [8, 3, 6, 5])
    smooth = Microswimmer(n_agents=4)
    obs = smooth._observe(state, params)
    np.testing.assert_allclose(obs[:, 0], [1.0, -1.0, 0.0, 0.4], atol=1e-12)
    np.testing.assert_allclose(obs[0, 1:], [0.0, 1.0], atol=1e-12)


def test_naive_swimmers_are_trapped_or_rise_depending_on_the_parameters():
    env = Microswimmer(n_agents=128)

    def climb_speed(swim_speed, alignment_time):
        params = env.default_params._replace(swim_speed=swim_speed, alignment_time=alignment_time)
        _, state = env.reset(jax.random.key(0), params)
        _, (reward, _) = _run(env, state, env.naive_policy(), params, 1000, seed=1)
        return float(reward[500:].sum(axis=0).mean()) / 50.0

    # Slow swimmers with a slow alignment stay in the vortices (reference: 0.000).
    assert abs(climb_speed(0.3, 1.0)) < 0.02
    # Fast swimmers with a fast alignment rise at about their swimming speed
    # (reference: 0.989 of it).
    np.testing.assert_allclose(climb_speed(3.0, 0.1), 3.0, rtol=0.03)
