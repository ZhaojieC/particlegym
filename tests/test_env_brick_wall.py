"""Brick wall game: interface, bounces, bricks, lost balls, images, scripted player."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from particlegym import physics as pp
from particlegym import viz
from particlegym.envs import BrickWall


def _with_ball(state, pos, vel):
    ball = pp.create_state(jnp.array([pos]), jnp.array([vel]), radius=state.ball.radius)
    return state._replace(ball=ball)


def _run(env, state, params, steps, action=0):
    step = jax.jit(env.step)
    out = []
    for i in range(steps):
        result = step(jax.random.key(i), state, jnp.array([action]), params)
        state = result[1]
        out.append(result)
    return state, out


def test_interface_shapes_and_batching():
    env = BrickWall()
    params = env.default_params
    assert env.n_agents == 1 and env.n_bricks == 40
    assert env.observation_space.shape == (45,) and env.action_space.n == 3
    obs, state = jax.jit(env.reset)(jax.random.key(0), params)
    assert obs.shape == (1, 45)
    obs, state, reward, terminated, truncated, info = jax.jit(env.step)(
        jax.random.key(1), state, jnp.array([2]), params
    )
    assert obs.shape == (1, 45) and reward.shape == terminated.shape == truncated.shape == (1,)
    assert float(state.paddle) > 4.0 and int(info["lives"]) == 3
    np.testing.assert_allclose(obs[0, 5:], 1.0)

    keys = jax.random.split(jax.random.key(2), 6)
    _, batch = jax.vmap(env.reset, in_axes=(0, None))(keys, params)
    out = jax.vmap(env.step, in_axes=(0, 0, 0, None))(keys, batch, jnp.ones((6, 1), int), params)
    assert out[0].shape == (6, 1, 45)

    continuous = BrickWall(action="continuous")
    assert continuous.action_space.shape == (1,)
    _, state = continuous.reset(jax.random.key(0))
    _, moved, *_ = continuous.step(jax.random.key(1), state, jnp.array([[-0.5]]))
    np.testing.assert_allclose(moved.paddle, 4.0 - 0.5 * 8.0 * 0.02, rtol=1e-6)
    with pytest.raises(ValueError, match="square pixels"):
        BrickWall(size=(8.0, 10.0), observation="pixels")


def test_ball_bounces_elastically_between_the_side_walls():
    env = BrickWall()
    params = env.default_params._replace(speed_relaxation=0.0)
    _, state = env.reset(jax.random.key(0), params)
    state = _with_ball(state, [4.0, 3.0], [6.0, 0.0])
    state, out = _run(env, state, params, 150)
    velocity = np.array([o[1].ball.vel[0] for o in out])
    # 18 length units in 3 s: the ball has hit both walls.
    assert (np.diff(np.sign(velocity[:, 0])) != 0).sum() >= 2
    free = np.abs(np.abs(velocity[:, 0]) - 6.0) < 0.5  # steps that do not end in a contact
    assert free.sum() > 130
    # A bounce changes the speed by about 0.1 %: (k / m)^(1/2) dt is 0.14.
    np.testing.assert_allclose(np.abs(velocity[free, 0]), 6.0, rtol=3e-3)
    np.testing.assert_allclose(velocity[:, 1], 0.0, atol=1e-9)
    np.testing.assert_allclose(state.ball.pos[0, 1], 3.0, atol=1e-9)
    assert int(state.score) == 0 and int(state.lives) == 3


def test_brick_disappears_after_the_contact_and_gives_a_reward():
    env = BrickWall()
    params = env.default_params._replace(speed_relaxation=0.0)
    _, state = env.reset(jax.random.key(0), params)
    brick = 4 * 8  # lowest row, first column: spans x in [0, 1], y in [4.8, 5.2]
    np.testing.assert_allclose(env.brick_lo[brick], [0.0, 4.8])
    only = jnp.zeros(40, bool).at[jnp.array([brick, 0])].set(True)
    state = _with_ball(state, [0.5, 4.0], [0.0, 6.0])._replace(bricks=only)
    state, out = _run(env, state, params, 12)
    touched = np.array([bool(o[1].touched[brick]) for o in out])
    present = np.array([bool(o[1].bricks[brick]) for o in out])
    reward = np.array([float(o[2][0]) for o in out])
    # The brick stays in place as long as the ball touches it.
    assert touched.any() and present[touched].all() and not present[-1]
    assert reward.sum() == 1.0 and reward[np.argmin(present)] == 1.0
    assert int(state.score) == 1 and bool(state.bricks[0])
    np.testing.assert_allclose(state.ball.vel[0], [0.0, -6.0], atol=2e-3)
    assert not bool(out[-1][3][0])

    # Removing the last brick ends the game.
    last = jnp.zeros(40, bool).at[brick].set(True)
    state = _with_ball(state, [0.5, 4.0], [0.0, 6.0])._replace(bricks=last)
    state, out = _run(env, state, params, 12)
    assert bool(out[-1][3][0]) and not bool(out[-1][4][0]) and not bool(state.bricks.any())


def test_lost_ball_costs_a_life_and_the_last_one_ends_the_game():
    env = BrickWall()
    params = env.default_params._replace(lives=2)
    _, state = env.reset(jax.random.key(0), params)
    state = _with_ball(state, [7.5, 0.6], [0.0, -6.0])
    state, out = _run(env, state, params, 8)
    lost = np.array([bool(o[5]["lost"]) for o in out])
    assert lost.sum() == 1 and int(state.lives) == 1 and not bool(out[-1][3][0])
    first = int(np.argmax(lost))
    # A new ball is launched below the bricks.
    np.testing.assert_allclose(out[first][1].ball.pos[0], [4.0, 3.0])
    np.testing.assert_allclose(np.linalg.norm(out[first][1].ball.vel[0]), 6.0, rtol=1e-6)
    state = _with_ball(state, [7.5, 0.6], [0.0, -6.0])
    state, out = _run(env, state, params, 8)
    assert int(state.lives) == 0 and bool(out[-1][3][0])


def test_pixel_observation_shows_the_ball_the_bricks_and_the_last_frame():
    env = BrickWall(observation="pixels")
    params = env.default_params
    assert env.observation_space.shape == (84, 84, 2)
    obs, state = jax.jit(env.reset)(jax.random.key(0), params)
    assert obs.shape == (1, 84, 84, 2)
    np.testing.assert_array_equal(obs[0, :, :, 0], obs[0, :, :, 1])
    state = _with_ball(state, [2.31, 2.74], [3.0, 3.0])
    new_obs, new_state, *_ = jax.jit(env.step)(jax.random.key(1), state, jnp.array([1]), params)
    np.testing.assert_array_equal(new_obs[0, :, :, 1], obs[0, :, :, 0])
    image = np.asarray(new_obs[0, :, :, 0])
    assert image.min() >= 0.0 and image.max() <= 1.0
    # The ball is the only object brighter than the bricks.
    rows, columns = np.nonzero(image > 0.75)
    weight = image[rows, columns]
    x = (columns + 0.5) * env.pixel
    y = 8.0 - (rows + 0.5) * env.pixel
    centre = np.array([np.sum(weight * x), np.sum(weight * y)]) / weight.sum()
    np.testing.assert_allclose(centre, new_state.ball.pos[0], atol=0.03)
    # The band of the bricks, y from 4.8 to 6.8, has the grey value 0.7. The
    # joints between the bricks are darker, which lowers the sum by 6 %.
    height = 8.0 - (np.arange(84) + 0.5) * env.pixel
    band = image[(height > 4.4) & (height < 7.2)]
    area = band.sum() / 0.7 * env.pixel**2
    assert 0.92 * 16.0 < area < 16.0
    assert np.all(image[(height > 3.3) & (height < 4.4)] == 0.0)


def test_pixel_image_of_the_bricks_equals_the_test_of_every_pixel_against_every_brick():
    env = BrickWall(observation="pixels")
    _, state = env.reset(jax.random.key(0))
    present = jax.random.bernoulli(jax.random.key(1), 0.5, (env.n_bricks,))
    image = jax.jit(env._image)(state._replace(bricks=present))
    points = viz.pixel_centres((0.0, 0.0), env.size, env.pixels)
    lo, hi = jnp.asarray(env.brick_lo), jnp.asarray(env.brick_hi)
    bricks = viz.box_coverage(points, lo, hi, env.pixel, mask=present)
    above = np.asarray(points[..., 1] > 4.0)  # the ball and the paddle are below
    assert float(bricks.max()) == 1.0
    np.testing.assert_allclose(image[above], 0.7 * bricks[above], atol=2e-5)


def test_scripted_player_clears_the_wall():
    env = BrickWall()
    params = env.default_params

    @jax.jit
    @jax.vmap
    def play(key):
        reset_key, step_key = jax.random.split(key)
        _, state = env.reset(reset_key, params)

        def body(carry, k):
            s, done = carry
            _, new, _, terminated, _, _ = env.step(k, s, env.scripted_policy(s, params), params)
            s = jax.tree.map(lambda old, new: jnp.where(done, old, new), s, new)
            return (s, done | terminated[0]), s.score

        (final, _), score = jax.lax.scan(body, (state, False), jax.random.split(step_key, 5000))
        return final, score

    final, score = play(jax.random.split(jax.random.key(0), 8))
    # Reference run: all eight games reach 39 or 40 bricks without losing a
    # ball; the mean score is 24 after 2000 steps.
    assert int(final.lives.min()) == 3
    assert int(final.score.min()) >= 35
    assert float(score[:, 1999].mean()) > 15
