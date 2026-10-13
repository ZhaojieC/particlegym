"""Billiards: interface, collisions, friction, cushions, pockets, simple player."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from particlegym import physics as pp
from particlegym.envs import Billiards

NO_FRICTION = {"friction": 0.0, "max_shot_time": 0.4}


def _place(env, state, positions):
    """The state with the cue ball and the object balls at the given positions."""
    pos = jnp.asarray(positions, state.balls.pos.dtype)
    return state._replace(balls=state.balls._replace(pos=pos))


def _shot(angle, speed, params):
    strength = 2.0 * (speed - params.min_speed) / (params.max_speed - params.min_speed) - 1.0
    return jnp.array([[angle / np.pi, strength]])


def test_interface_shapes_and_batching():
    env = Billiards()
    params = env.default_params
    assert env.n_agents == 1 and env.observation_space.shape == (48,)
    assert env.action_space.shape == (2,)
    obs, state = jax.jit(env.reset)(jax.random.key(0), params)
    assert obs.shape == (1, 48)
    pos = np.asarray(state.balls.pos)
    distance = np.linalg.norm(pos[:, None] - pos[None], axis=-1) + np.eye(16)
    assert distance.min() > 2 * env.ball_radius  # no overlaps in the rack
    assert np.all(pos > env.ball_radius) and np.all(pos < np.array(env.table) - env.ball_radius)
    np.testing.assert_allclose(pos[1], env.foot_spot)

    obs, state, reward, terminated, truncated, info = jax.jit(env.step)(
        jax.random.key(1), state, jnp.array([[0.0, 1.0]]), params
    )
    assert obs.shape == (1, 48) and reward.shape == terminated.shape == truncated.shape == (1,)
    assert bool(info["rest"]) and float(info["time"]) < params.max_shot_time
    np.testing.assert_allclose(state.balls.vel, 0.0)
    assert np.abs(np.asarray(obs[0]).reshape(16, 3)[:, :2]).max() <= 0.5

    keys = jax.random.split(jax.random.key(2), 4)
    _, batch = jax.vmap(env.reset, in_axes=(0, None))(keys, params)
    actions = env.action_space.sample(jax.random.key(3), (4, 1))
    out = jax.vmap(env.step, in_axes=(0, 0, 0, None))(keys, batch, actions, params)
    assert out[0].shape == (4, 1, 48) and bool(out[5]["rest"].all())
    with pytest.raises(ValueError, match="between 1 and 15"):
        Billiards(n_balls=16)


@pytest.mark.parametrize("restitution", [0.95, 0.7])
def test_head_on_collision_has_the_set_restitution(restitution):
    env = Billiards(n_balls=1, ball_restitution=restitution)
    params = env.default_params._replace(**NO_FRICTION)
    _, state = env.reset(jax.random.key(0), params)
    state = _place(env, state, [[0.6, 0.56], [0.9, 0.56]])
    balls = env.strike(state.balls, _shot(0.0, 2.0, params), params)
    field = env.force_field(params)
    balls, _ = pp.init_forces(balls, field)
    balls = jax.jit(lambda b: env._advance(b, field, 3000))(balls)  # 0.3 s
    # Equal masses: v1' = (1 - e) v / 2 and v2' = (1 + e) v / 2. The error of
    # the restitution is of first order in the time step and grows with the
    # damping: 0.25 % for e = 0.95 and 1.3 % for e = 0.7.
    measured = float(balls.vel[1, 0] - balls.vel[0, 0]) / 2.0
    np.testing.assert_allclose(measured, restitution, rtol=4e-3 if restitution > 0.9 else 2e-2)
    np.testing.assert_allclose(balls.vel[:, 1], 0.0, atol=1e-12)
    np.testing.assert_allclose(pp.total_momentum(balls), [0.17 * 2.0, 0.0], rtol=1e-10, atol=1e-12)


def test_elastic_balls_separate_at_a_right_angle():
    env = Billiards(n_balls=1, ball_restitution=1.0)
    params = env.default_params._replace(**NO_FRICTION)
    _, state = env.reset(jax.random.key(0), params)
    # The cue ball moves along x and hits the other ball off centre.
    state = _place(env, state, [[0.6, 0.56], [0.9, 0.56 + 0.03]])
    balls = env.strike(state.balls, _shot(0.0, 2.0, params), params)
    field = env.force_field(params)
    balls, _ = pp.init_forces(balls, field)
    balls = jax.jit(lambda b: env._advance(b, field, 3000))(balls)
    v = np.asarray(balls.vel)
    cosine = v[0] @ v[1] / (np.linalg.norm(v[0]) * np.linalg.norm(v[1]))
    assert np.linalg.norm(v[0]) > 0.5 and np.linalg.norm(v[1]) > 0.5
    np.testing.assert_allclose(cosine, 0.0, atol=2e-3)
    np.testing.assert_allclose(pp.kinetic_energy(balls), 0.5 * 0.17 * 2.0**2, rtol=1e-3)


def test_friction_and_cushion():
    env = Billiards(n_balls=1, cushion_restitution=0.8)
    params = env.default_params
    _, state = env.reset(jax.random.key(0), params)
    # A ball that slides freely stops after v^2 / (2 a).
    state = _place(env, state, [[0.3, 0.3], [2.0, 1.0]])
    _, after, reward, _, _, info = env.step(None, state, _shot(0.0, 1.2, params), params)
    travelled = float(after.balls.pos[0, 0]) - 0.3
    np.testing.assert_allclose(travelled, 1.2**2 / (2 * params.friction), rtol=1e-2)
    np.testing.assert_allclose(info["time"], 1.2 / params.friction, rtol=0.1)
    assert float(reward[0]) == 0.0 and bool(info["rest"])

    # A cushion returns the ball with the speed multiplied by the restitution.
    free = params._replace(**NO_FRICTION)
    balls = env.strike(state.balls, _shot(-np.pi / 2, 1.0, free), free)
    field = env.force_field(free)
    balls, _ = pp.init_forces(balls, field)
    balls = jax.jit(lambda b: env._advance(b, field, 4000))(balls)
    # The error depends on when the contact starts within a time step:
    # between -0.2 % and +0.5 % for time steps from 2e-5 to 2e-4 s.
    np.testing.assert_allclose(balls.vel[0], [0.0, 0.8], rtol=1e-2, atol=1e-9)


def test_pocketing_a_ball_and_the_cue_ball():
    env = Billiards(n_balls=2)
    params = env.default_params
    _, state = env.reset(jax.random.key(0), params)
    # Ball 1 lies on the line from the cue ball to the corner pocket at (0, 0).
    state = _place(env, state, [[0.6, 0.6], [0.3, 0.3], [1.8, 0.9]])
    step = jax.jit(env.step)
    obs, after, reward, terminated, _, info = step(
        None, state, _shot(-0.75 * np.pi, 1.5, params), params
    )
    np.testing.assert_array_equal(after.balls.alive, [True, False, True])
    assert float(reward[0]) == 1.0 and int(info["pocketed"]) == 1 and not bool(info["scratch"])
    assert not bool(terminated[0])
    np.testing.assert_allclose(obs[0].reshape(3, 3)[1], 0.0)
    # The aiming player finds the same shot.
    aimed = env.aim_policy(state)
    np.testing.assert_allclose(aimed[0, 0], -0.75, atol=1e-6)

    # The cue ball alone into the pocket: a penalty, and it returns to the head line.
    alone = _place(env, state, [[0.4, 0.4], [1.5, 0.9], [1.8, 0.9]])
    _, after, reward, _, _, info = step(None, alone, _shot(-0.75 * np.pi, 1.5, params), params)
    assert bool(info["scratch"]) and float(reward[0]) == params.scratch_penalty
    np.testing.assert_array_equal(after.balls.alive, [True, True, True])
    np.testing.assert_allclose(after.balls.pos[0], env.head_spot)
    # If a ball lies on the head spot, the next free place is used.
    blocked = _place(env, state, [[0.4, 0.4], [*env.head_spot], [1.8, 0.9]])
    _, after, *_ = step(None, blocked, _shot(-0.75 * np.pi, 1.5, params), params)
    np.testing.assert_allclose(
        after.balls.pos[0], env.head_spot + np.array([0.0, 3 * env.ball_radius])
    )

    # Pocketing the last ball ends the game.
    single = Billiards(n_balls=1)
    _, state = single.reset(jax.random.key(0))
    state = _place(single, state, [[0.6, 0.6], [0.3, 0.3]])
    _, after, reward, terminated, truncated, _ = single.step(
        None, state, _shot(-0.75 * np.pi, 1.5, params), params
    )
    assert bool(terminated[0]) and not bool(truncated[0]) and float(reward[0]) == 1.0


def test_trajectory_matches_the_step():
    env = Billiards(n_balls=3)
    params = env.default_params
    _, state = env.reset(jax.random.key(1), params)
    action = jnp.array([[0.02, 0.6]])
    path = jax.jit(lambda s, a: env.trajectory(s, a, params, 200, 0.05))(state, action)
    assert path.pos.shape == (200, 4, 2)
    _, after, _, _, _, info = env.step(None, state, action, params)
    assert bool(info["rest"]) and float(info["time"]) < 10.0
    np.testing.assert_array_equal(path.alive[-1], after.balls.alive | ~path.alive[-1])
    still = np.asarray(path.alive[-1])
    np.testing.assert_allclose(path.pos[-1][still], after.balls.pos[still], atol=2e-3)


@pytest.mark.slow
def test_aiming_player_clears_most_of_the_table():
    env = Billiards()
    params = env.default_params

    @jax.jit
    @jax.vmap
    def game(key):
        reset_key, step_key = jax.random.split(key)
        _, state = env.reset(reset_key, params)

        def body(carry, k):
            s, done = carry
            _, new, _, terminated, _, info = env.step(k, s, env.aim_policy(s), params)
            s = jax.tree.map(lambda old, new: jnp.where(done, old, new), s, new)
            return (s, done | terminated[0]), info["rest"]

        (final, _), rest = jax.lax.scan(body, (state, False), jax.random.split(step_key, 30))
        return final, rest

    final, rest = game(jax.random.split(jax.random.key(0), 16))
    pocketed = 15 - np.asarray(final.balls.alive[:, 1:]).sum(axis=1)
    # Reference run: 14.25 of 15 balls on average in 30 shots.
    assert bool(rest.all())
    assert pocketed.mean() > 12.0
