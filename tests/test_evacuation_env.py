"""The evacuation environment: API contract, compilation, batching and physics."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from particlegym import envs
from particlegym import physics as pp

TOP_DOOR = pp.Opening("y+", (4.5, 5.5))


def _env(**kwargs):
    kwargs.setdefault("exits", [TOP_DOOR])
    return envs.Evacuation(room=(10.0, 10.0), **kwargs)


def _rollout(env, state, actions_fn, params, steps, seed=0):
    """Run `steps` compiled control steps; `actions_fn(state)` gives the actions."""

    def body(s, key):
        _, s, reward, terminated, truncated, info = env.step(key, s, actions_fn(s), params)
        return s, (reward, terminated, truncated, info["inside"])

    keys = jax.random.split(jax.random.key(seed), steps)
    return jax.jit(lambda s: jax.lax.scan(body, s, keys))(state)


def test_reset_places_pedestrians_inside_without_overlap():
    pillar = pp.Spheres([[3.0, 3.0]], [1.0])
    env = _env(n_agents=3, n_crowd=40, obstacles=[pillar], radius=(0.25, 0.3), min_gap=0.2)
    obs, state = jax.jit(env.reset)(jax.random.key(0))
    p = state.particles
    pos, radius = np.asarray(p.pos), np.asarray(p.radius)

    assert obs.shape == (3, 4)
    assert p.pos.shape == (43, 2)
    assert np.all((radius >= 0.25) & (radius <= 0.3))
    assert np.all((pos > radius[:, None]) & (pos < 10.0 - radius[:, None]))
    gaps = np.linalg.norm(pos[:, None] - pos[None], axis=-1) - (radius[:, None] + radius[None])
    assert gaps[~np.eye(43, dtype=bool)].min() >= 0.2 - 1e-9
    assert np.all(np.linalg.norm(pos - [3.0, 3.0], axis=1) - 1.0 > radius)
    np.testing.assert_array_equal(p.kind, [0] * 3 + [1] * 40)
    assert int(state.step) == 0
    assert np.all(np.isinf(np.asarray(state.exit_time)))
    np.testing.assert_allclose(obs[:, :2], pos[:3] / 10.0 - 0.5)
    np.testing.assert_allclose(obs[:, 2:], 0.0)


def test_reset_depends_only_on_the_key():
    env = _env(n_crowd=10)
    reset = jax.jit(env.reset)
    a = reset(jax.random.key(1))[1].particles.pos
    b = reset(jax.random.key(1))[1].particles.pos
    c = reset(jax.random.key(2))[1].particles.pos
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)


def test_room_too_small_for_the_crowd_is_rejected():
    with pytest.raises(ValueError, match="space for"):
        envs.Evacuation(room=(3.0, 3.0), n_agents=1, n_crowd=50)


def test_discrete_directions_follow_the_physica_a_order():
    s = np.sqrt(0.5)
    expected = [(0, 1), (-s, s), (-1, 0), (-s, -s), (0, -1), (s, -s), (1, 0), (s, s)]
    np.testing.assert_allclose(_env().directions, expected, atol=1e-12)


def test_actions_set_the_desired_velocity():
    params = envs.EvacuationParams(desired_speed=1.5)
    discrete = _env(n_agents=2)
    np.testing.assert_allclose(
        discrete.agent_velocity(jnp.array([6, 0]), params, jnp.float64),
        [[1.5, 0.0], [0.0, 1.5]],
        atol=1e-12,
    )
    continuous = _env(n_agents=2, action="continuous")
    velocity = continuous.agent_velocity(jnp.array([[0.5, 0.0], [3.0, 4.0]]), params, jnp.float64)
    # Lengths below 1 scale the desired speed; longer vectors are capped at it.
    np.testing.assert_allclose(velocity, [[0.75, 0.0], [0.9, 1.2]], atol=1e-12)


def test_crowd_policies():
    params = envs.EvacuationParams(desired_speed=1.2)
    key = jax.random.key(0)
    exits = [TOP_DOOR, pp.Opening("x-", (2.0, 3.0))]
    state = jax.jit(_env(n_crowd=20, exits=exits).reset)(key)[1]

    stand = _env(n_crowd=20, exits=exits, crowd_policy="stand")
    np.testing.assert_allclose(stand.crowd_velocity(key, state.particles, params), 0.0)

    random = _env(n_crowd=20, exits=exits, crowd_policy="random")
    velocity = np.asarray(random.crowd_velocity(key, state.particles, params))
    np.testing.assert_allclose(np.linalg.norm(velocity, axis=1), 1.2)
    other = np.asarray(random.crowd_velocity(jax.random.key(1), state.particles, params))
    assert not np.allclose(velocity, other)

    nearest = _env(n_crowd=20, exits=exits, crowd_policy="nearest_exit")
    velocity = np.asarray(nearest.crowd_velocity(key, state.particles, params))
    pos = np.asarray(state.particles.pos[1:])
    centres = np.array([[5.0, 10.0], [0.0, 2.5]])
    targets = np.array([[5.0, 11.0], [-1.0, 2.5]])  # 1 m beyond each door centre
    choice = np.argmin(np.linalg.norm(pos[:, None] - centres, axis=-1), axis=1)
    expected = targets[choice] - pos
    expected *= 1.2 / np.linalg.norm(expected, axis=1, keepdims=True)
    np.testing.assert_allclose(velocity, expected, rtol=1e-10)


def test_agent_walking_out_of_the_door_terminates_with_the_exit_reward():
    env = _env(action="continuous")
    params = env.default_params._replace(exit_reward=5.0)
    _, state = jax.jit(env.reset)(jax.random.key(3), params)
    target = jnp.array([5.0, 11.0])  # beyond the door centre

    def towards_door(s):
        offset = target - s.particles.pos[:1]
        return offset / jnp.linalg.norm(offset)

    state, (reward, terminated, truncated, _) = _rollout(env, state, towards_door, params, 150)
    reward, terminated = np.asarray(reward[:, 0]), np.asarray(terminated[:, 0])
    exit_step = int(np.argmax(terminated))

    assert terminated[-1] and not np.asarray(truncated).any()
    assert np.all(terminated[exit_step:]) and not terminated[:exit_step].any()
    np.testing.assert_allclose(reward[:exit_step], -0.1)
    assert reward[exit_step] == 5.0
    np.testing.assert_allclose(reward[exit_step + 1 :], 0.0)
    np.testing.assert_allclose(state.exit_time[0], (exit_step + 1) * env.dt * env.substeps)
    x, y = np.asarray(state.particles.pos[0])
    assert 4.5 < x < 5.5 and y > 10.0  # it left through the door


def test_episode_is_truncated_at_max_steps():
    env = _env(action="continuous")
    params = env.default_params._replace(max_steps=5)
    _, state = jax.jit(env.reset)(jax.random.key(4), params)
    _, (_, terminated, truncated, _) = _rollout(env, state, lambda s: jnp.zeros((1, 2)), params, 5)
    np.testing.assert_array_equal(np.asarray(truncated[:, 0]), [False] * 4 + [True])
    assert not np.asarray(terminated).any()


def test_crowd_leaves_only_through_the_doors():
    exits = [pp.Opening("y+", (4.4, 5.6)), pp.Opening("x-", (4.4, 5.6))]
    env = _env(exits=exits, n_agents=0, n_crowd=40, action="continuous", radius=(0.25, 0.3))
    obs, state = jax.jit(env.reset)(jax.random.key(5))
    assert obs.shape == (0, 4)
    state, (_, _, _, inside) = _rollout(
        env, state, lambda s: jnp.zeros((0, 2)), env.default_params, 600
    )
    pos = np.asarray(state.particles.pos)
    radius = np.asarray(state.particles.radius)

    assert int(inside[-1]) == 0
    assert np.all(np.isfinite(np.asarray(state.exit_time)))
    # Pedestrians are frozen where they were removed: just outside a doorway.
    through_top = (pos[:, 1] > 10.0) & (np.abs(pos[:, 0] - 5.0) < 0.6 + radius)
    through_left = (pos[:, 0] < 0.0) & (np.abs(pos[:, 1] - 5.0) < 0.6 + radius)
    assert np.all(through_top | through_left)


def test_reset_and_step_compile_and_batch():
    env = _env(n_agents=2, n_crowd=5)
    keys = jax.random.split(jax.random.key(6), 4)
    obs, states = jax.jit(jax.vmap(env.reset))(keys)
    actions = jnp.array([[0, 6], [2, 4], [1, 7], [3, 5]])

    out = jax.jit(jax.vmap(env.step))(keys, states, actions)
    obs, next_states, reward, terminated, truncated, info = out

    assert obs.shape == (4, 2, 4)
    assert reward.shape == terminated.shape == truncated.shape == (4, 2)
    assert info["inside"].shape == (4,)
    np.testing.assert_array_equal(next_states.step, 1)
    single = jax.jit(env.step)(keys[1], jax.tree.map(lambda x: x[1], states), actions[1])
    np.testing.assert_allclose(single[0], obs[1], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(single[1].particles.pos, next_states.particles.pos[1], rtol=1e-9)
    np.testing.assert_allclose(single[2], reward[1])


def test_params_can_be_batched_for_parameter_sweeps():
    env = _env(action="continuous")
    params = jax.tree.map(lambda x: jnp.full(2, x), env.default_params)
    params = params._replace(desired_speed=jnp.array([1.0, 2.0]))
    # Start both runs from the same place, far from the walls, walking up.
    _, state = jax.jit(env.reset)(jax.random.key(7))
    particles, _ = jax.jit(pp.init_forces, static_argnums=1)(
        state.particles._replace(pos=jnp.array([[5.0, 3.0]])), env.force_field
    )
    states = jax.tree.map(lambda x: jnp.stack([x, x]), state._replace(particles=particles))
    keys = jax.random.split(jax.random.key(8), 2)

    def run(key, state, p):
        def body(s, k):
            return env.step(k, s, jnp.array([[0.0, 1.0]]), p)[1], None

        return jax.lax.scan(body, state, jax.random.split(key, 10))[0]

    final = jax.jit(jax.vmap(run))(keys, states, params)
    distance = np.asarray(final.particles.pos[:, 0, 1]) - 3.0
    # The walking distance is proportional to the desired speed.
    np.testing.assert_allclose(distance[1] / distance[0], 2.0, rtol=1e-6)
    # After 1 s from rest: v0 * (t - tau * (1 - exp(-t / tau))).
    expected = np.array([1.0, 2.0]) * (1.0 - 0.5 * (1.0 - np.exp(-2.0)))
    np.testing.assert_allclose(distance, expected, rtol=1e-3)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_agents": 0, "n_crowd": 0},
        {"crowd_policy": "panic"},
        {"action": "joystick"},
        {"radius": (0.3, 0.2)},
        {"exits": [pp.Opening("z+", ((0.0, 1.0), (0.0, 1.0)))]},
    ],
)
def test_invalid_configurations_are_rejected(kwargs):
    with pytest.raises(ValueError):
        _env(**kwargs)
