"""Counter-flow corridor: interface, own-frame observations, lane formation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from particlegym.envs import CounterFlow


def test_interface_shapes_and_batching():
    env = CounterFlow(length=12.0, width=4.0, n_agents=3, n_crowd=9, n_neighbors=4)
    params = env.default_params
    assert env.observation_space.shape == (23,)
    assert env.action_space.shape == (2,)
    obs, state = jax.jit(env.reset)(jax.random.key(0), params)
    assert obs.shape == (3, 23)
    action = env.action_space.sample(jax.random.key(1), (3,))
    obs, state, reward, terminated, truncated, info = jax.jit(env.step)(
        jax.random.key(2), state, action, params
    )
    assert obs.shape == (3, 23) and reward.shape == (3,)
    assert not terminated.any() and not truncated.any()
    assert np.isfinite(obs).all() and not bool(info["overflow"])

    keys = jax.random.split(jax.random.key(3), 5)
    _, batch_state = jax.vmap(env.reset, in_axes=(0, None))(keys, params)
    out = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
        keys, batch_state, jnp.zeros((5, 3, 2)), params
    )
    assert out[0].shape == (5, 3, 23)

    discrete = CounterFlow(length=12.0, width=4.0, n_agents=2, n_crowd=4, action="discrete")
    assert discrete.action_space.n == 8
    np.testing.assert_allclose(discrete.directions[0], [1.0, 0.0], atol=1e-12)
    _, state = discrete.reset(jax.random.key(0))
    discrete.step(jax.random.key(1), state, jnp.array([0, 3]))
    with pytest.raises(ValueError, match="space for"):
        CounterFlow(length=5.0, width=2.0, n_agents=1, n_crowd=60)


def test_reward_is_the_progress_along_the_walking_direction():
    env = CounterFlow(length=40.0, width=6.0, n_agents=2, n_crowd=0, n_neighbors=1)
    params = env.default_params._replace(max_steps=3)
    _, state = env.reset(jax.random.key(0), params)
    step = jax.jit(env.step)
    total = jnp.zeros(2)
    start = state.particles.pos
    for i in range(3):
        # Straight ahead in the own frame: +x for agent 0, -x for agent 1.
        _, state, reward, _, truncated, _ = step(
            jax.random.key(i), state, jnp.array([[1.0, 0.0], [1.0, 0.0]]), params
        )
        total = total + reward
    assert truncated.all()
    moved = env.box.displacement(state.particles.pos, start)[:, 0]
    np.testing.assert_allclose(total, [moved[0], -moved[1]], rtol=1e-6)
    assert float(moved[0]) > 0.05 and float(moved[1]) < -0.05


def test_observations_are_given_in_the_own_frame():
    env = CounterFlow(length=12.0, width=4.0, n_agents=4, n_crowd=4, n_neighbors=3, radius=0.25)
    _, state = env.reset(jax.random.key(5))
    p = state.particles
    vel = jax.random.normal(jax.random.key(6), p.vel.shape)
    state = state._replace(particles=p._replace(vel=vel))
    obs = env._observe(state)
    # Rotate the corridor by 180 degrees and exchange the two groups: every
    # pedestrian then sees what its partner saw before.
    partner = jnp.arange(8) ^ 1
    turned = p._replace(
        pos=(jnp.array([12.0, 4.0]) - p.pos)[partner], vel=-vel[partner], radius=p.radius[partner]
    )
    turned_obs = env._observe(state._replace(particles=turned))
    np.testing.assert_allclose(turned_obs, obs[partner[:4]], atol=1e-9)


def test_lane_order_of_lanes_and_of_a_mixture():
    env = CounterFlow(length=20.0, width=4.0, n_agents=0, n_crowd=40, radius=0.25)
    _, state = env.reset(jax.random.key(0))
    index = np.arange(40)
    lanes = np.stack([(index // 2 + 0.5) * 1.0, np.where(index % 2 == 0, 1.0, 3.0)], axis=1)
    order = env.lane_order(state.particles._replace(pos=jnp.asarray(lanes)))
    np.testing.assert_allclose(order, 1.0)
    assert float(env.lane_order(state.particles)) < 0.3


@pytest.mark.slow
def test_lanes_form_and_the_flow_speeds_up():
    env = CounterFlow(length=30.0, width=6.0, n_agents=0, n_crowd=80)
    params = env.default_params

    @jax.jit
    @jax.vmap
    def rollout(key):
        reset_key, step_key = jax.random.split(key)
        _, state = env.reset(reset_key, params)

        def body(s, k):
            _, s, _, _, _, info = env.step(k, s, jnp.zeros((0, 2)), params)
            return s, (info["lane_order"], info["speed"])

        return jax.lax.scan(body, state, jax.random.split(step_key, 900))[1]

    order, speed = rollout(jax.random.split(jax.random.key(0), 4))
    # Reference run: the order parameter rises from 0.08 to 0.98 and the mean
    # speed from 0.97 m/s (after 10 s) to 1.30 m/s, the desired speed.
    assert float(order[:, 0].mean()) < 0.2
    assert float(order[:, -1].mean()) > 0.8
    assert float(speed[:, -1].mean()) > float(speed[:, 99].mean()) + 0.15
