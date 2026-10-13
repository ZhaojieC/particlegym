"""Herding: interface, flock rules, reward, scripted shepherd."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from particlegym.envs import Herding


def _place(state, shepherds, sheep):
    """The state with shepherds and sheep at the given positions, at rest."""
    pos = jnp.asarray(np.concatenate([shepherds, sheep]), state.particles.pos.dtype)
    particles = state.particles._replace(pos=pos, vel=jnp.zeros_like(pos))
    return state._replace(particles=particles)


def test_interface_shapes_and_batching():
    env = Herding(n_sheep=12, n_agents=2, n_neighbors=8)
    params = env.default_params
    assert env.observation_space.shape == (13,) and env.action_space.shape == (2,)
    obs, state = jax.jit(env.reset)(jax.random.key(0), params)
    assert obs.shape == (2, 13) and state.particles.pos.shape == (14, 2)
    np.testing.assert_array_equal(state.particles.kind, [0, 0] + [1] * 12)
    sheep = np.asarray(state.particles.pos[2:])
    assert sheep.min() > 0.5 * env.field and sheep.max() < env.field
    np.testing.assert_allclose(np.linalg.norm(state.heading, axis=1), 1.0, rtol=1e-6)

    action = env.action_space.sample(jax.random.key(1), (2,))
    obs, state, reward, terminated, truncated, info = jax.jit(env.step)(
        jax.random.key(2), state, action, params
    )
    assert obs.shape == (2, 13)
    assert reward.shape == terminated.shape == truncated.shape == (2,)
    assert float(info["distance"]) > 100.0 and float(info["spread"]) > 0.0
    assert bool(jnp.isfinite(obs).all())

    keys = jax.random.split(jax.random.key(3), 5)
    obs, batch = jax.vmap(env.reset, in_axes=(0, None))(keys, params)
    actions = env.action_space.sample(jax.random.key(4), (5, 2))
    out = jax.vmap(env.step, in_axes=(0, 0, 0, None))(keys, batch, actions, params)
    assert out[0].shape == (5, 2, 13) and out[2].shape == (5, 2)

    single = Herding(n_sheep=12, n_neighbors=8, action="discrete")
    assert single.observation_space.shape == (11,) and single.action_space.n == 8
    _, state = single.reset(jax.random.key(0))
    start = state.particles.pos[0]
    for _ in range(5):
        _, state, *_ = single.step(jax.random.key(5), state, jnp.array([2]))
    # Direction 2 of 8 is -x. The shepherd walks at 1.5 m/s for 5 s, less the
    # distance lost while its velocity relaxes (speed times relaxation time).
    moved = np.asarray(state.particles.pos[0] - start)
    np.testing.assert_allclose(moved, [-(7.5 - 1.5 * 0.25), 0.0], atol=0.05)
    with pytest.raises(ValueError, match="n_neighbors"):
        Herding(n_sheep=10, n_neighbors=10)


def test_sheep_flee_graze_gather_and_keep_apart():
    env = Herding(n_sheep=4, n_agents=1, n_neighbors=3)
    off = {"inertia": 0.0, "noise": 0.0, "graze_probability": 0.0}
    key = jax.random.key(0)
    _, state = env.reset(key)
    # The last sheep is farther from the shepherd than the detection distance of 65 m.
    sheep = np.array([[10.0, 0.0], [0.0, 20.0], [-30.0, 0.0], [100.0, 0.0]])
    state = _place(state, [[0.0, 0.0]], sheep)

    flee = env.default_params._replace(attraction=0.0, repulsion=0.0, **off)
    heading, velocity = env.sheep_velocity(key, state, flee)
    np.testing.assert_allclose(velocity, [[1, 0], [0, 1], [-1, 0], [0, 0]], atol=1e-12)
    np.testing.assert_allclose(heading[3], state.heading[3])  # a grazing sheep keeps its heading

    # A grazing sheep moves with the set probability.
    moving = flee._replace(graze_probability=1.0)
    _, velocity = env.sheep_velocity(key, state, moving)
    np.testing.assert_allclose(np.linalg.norm(velocity[3]), 1.0, rtol=1e-6)

    # Attraction alone: towards the centre of the three other sheep.
    gather = env.default_params._replace(shepherd_repulsion=0.0, repulsion=0.0, **off)
    _, velocity = env.sheep_velocity(key, state, gather)
    for i in range(3):
        to_centre = np.delete(sheep, i, axis=0).mean(axis=0) - sheep[i]
        np.testing.assert_allclose(velocity[i], to_centre / np.linalg.norm(to_centre), rtol=1e-6)

    # Repulsion alone: sheep closer than the interaction distance of 2 m move apart.
    close = np.array([[10.0, 0.0], [11.2, 0.9], [10.0, 5.0], [30.0, 0.0]])
    apart = env.default_params._replace(shepherd_repulsion=0.0, attraction=0.0, **off)
    _, velocity = env.sheep_velocity(key, _place(state, [[0.0, 0.0]], close), apart)
    np.testing.assert_allclose(velocity[0], [-0.8, -0.6], rtol=1e-6)
    np.testing.assert_allclose(velocity[1], [0.8, 0.6], rtol=1e-6)
    np.testing.assert_allclose(velocity[2:], 0.0, atol=1e-12)

    # The inertia keeps a share of the previous direction.
    inertia = flee._replace(inertia=1.0)
    state = state._replace(heading=jnp.array([[0.0, 1.0]] * 4))
    heading, _ = env.sheep_velocity(key, state, inertia)
    np.testing.assert_allclose(heading[0], np.array([1.0, 1.0]) / np.sqrt(2.0), rtol=1e-6)


def test_reward_is_the_progress_of_the_flock_and_success_ends_the_episode():
    env = Herding(n_sheep=6, n_agents=2, n_neighbors=5)
    params = env.default_params
    key = jax.random.key(0)
    _, state = env.reset(key, params)
    sheep = np.array([[40.0, 40.0], [42, 40], [40, 42], [42, 42], [44, 41], [41, 44]])
    # A shepherd behind the flock pushes it towards the target at the origin.
    state = _place(state, [[48.0, 48.0], [80.0, 0.0]], sheep)
    before = np.linalg.norm(sheep.mean(axis=0))
    total = 0.0
    for i in range(5):
        _, state, reward, terminated, truncated, info = env.step(
            jax.random.fold_in(key, i), state, jnp.zeros((2, 2)), params
        )
        assert float(reward[0]) == float(reward[1])
        total += float(reward[0])
    np.testing.assert_allclose(total, before - float(info["distance"]), rtol=1e-5)
    assert total > 1.0 and not bool(terminated.any()) and not bool(truncated.any())

    _, _, _, terminated, truncated, _ = env.step(
        key, state, jnp.zeros((2, 2)), params._replace(max_steps=6)
    )
    assert bool(truncated.all()) and not bool(terminated.any())

    # The centre of the flock comes within the target radius of 10 m.
    near = _place(state, [[20.0, 20.0], [80.0, 0.0]], sheep - 40.0 + 7.5)
    start = np.linalg.norm((sheep - 40.0 + 7.5).mean(axis=0))
    assert start > params.target_radius
    done, steps, total = False, 0, 0.0
    while not done and steps < 20:
        _, near, reward, terminated, truncated, info = env.step(
            jax.random.fold_in(key, steps), near, jnp.zeros((2, 2)), params
        )
        done, steps, total = bool(terminated[0]), steps + 1, total + float(reward[0])
    assert done and not bool(truncated.any())
    assert float(info["distance"]) < params.target_radius
    np.testing.assert_allclose(
        total, start - float(info["distance"]) + params.success_reward, rtol=1e-5
    )


def _games(env, params, policy, n_games, steps):
    @jax.jit
    @jax.vmap
    def game(key):
        reset_key, run_key = jax.random.split(key)
        _, state = env.reset(reset_key, params)

        def body(carry, k):
            s, done, used = carry
            action_key, step_key = jax.random.split(k)
            _, new, _, terminated, _, info = env.step(step_key, s, policy(action_key, s), params)
            s = jax.tree.map(lambda old, new: jnp.where(done, old, new), s, new)
            sheep = s.particles.pos[env.n_agents :]
            gap = jnp.linalg.norm(sheep[:, None] - sheep[None], axis=-1)
            gap = jnp.min(gap + 1e3 * jnp.eye(env.n_sheep))
            return (s, done | terminated[0], used + ~done), (info["distance"], gap)

        (_, done, used), (distance, gap) = jax.lax.scan(
            body, (state, False, 0), jax.random.split(run_key, steps)
        )
        return done, used, distance[-1], gap.min()

    return game(jax.random.split(jax.random.key(0), n_games))


def test_scripted_shepherd_brings_the_flock_to_the_target():
    env = Herding()
    params = env.default_params
    done, used, _, gap = _games(
        env, params, lambda k, s: env.shepherd_policy(k, s, params), n_games=8, steps=1000
    )
    # Reference run: solved in 430 to 610 control steps.
    assert bool(done.all()) and int(used.max()) < 800
    # The bodies (diameter 1 m) overlap by less than a fifth of their diameter.
    assert float(gap.min()) > 0.8

    # A shepherd that stands still is too far away to move the flock.
    done, _, distance, _ = _games(
        env, params, lambda k, s: jnp.zeros((1, 2)), n_games=4, steps=300
    )
    assert not bool(done.any()) and float(distance.min()) > 120.0


def test_two_shepherds_also_solve_the_task():
    env = Herding(n_agents=2)
    params = env.default_params
    done, used, _, _ = _games(
        env, params, lambda k, s: env.shepherd_policy(k, s, params), n_games=4, steps=1000
    )
    assert bool(done.all()) and int(used.max()) < 900
