"""Cell migration: interface, cell mechanics, concentration field, rewards, baseline."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from particlegym import physics as pp
from particlegym.envs import CellMigration


def _place(env, state, pos, params=None, settle=0.0):
    """The state with the cells at ``pos``; the field is rebuilt for ``settle`` time units."""
    params = env.default_params if params is None else params
    pos = jnp.asarray(pos, state.particles.pos.dtype).reshape(-1, 2)
    particles = state.particles._replace(pos=pos, vel=jnp.zeros_like(pos))
    particles, _ = pp.init_forces(particles, env.force_field)
    c3a = jnp.zeros_like(state.c3a)
    source = env._secretion(pos, params)
    step = jax.jit(lambda c: env._diffuse(c, source, params))
    for _ in range(round(settle / env.dt)):
        c3a = step(c3a)
    return state._replace(particles=particles, c3a=c3a)


def _angle(psi):
    """Action for the direction angle psi."""
    return jnp.asarray(psi) / jnp.pi - 1.0


def test_interface_shapes_and_batching():
    env = CellMigration(n_leaders=3, n_followers=4)
    params = env.default_params
    assert env.n_agents == 7 and env.observation_space.shape == (7,)
    assert env.action_space.shape == (1,)
    obs, state = jax.jit(env.reset)(jax.random.key(0), params)
    assert obs.shape == (7, 7) and state.c3a.shape == (2, 100, 100)
    np.testing.assert_array_equal(state.particles.kind, [0, 0, 0, 1, 1, 1, 1])
    np.testing.assert_array_equal(obs[:, 6], [1, 1, 1, 0, 0, 0, 0])
    pos = np.asarray(state.particles.pos)
    gap = np.linalg.norm(pos[:, None] - pos[None], axis=-1) + 10 * np.eye(7)
    assert gap.min() > 2 * env.cell_radius
    # The leaders start nearer to the source than the followers.
    assert pos[:3, 1].min() > pos[3:, 1].max()
    # After one decay time, the field holds (1 - 1/e) of its final amount per cell.
    amount = np.asarray(state.c3a.sum(axis=(1, 2))) * env.grid.cell_volume
    expected = np.array([3, 4]) * params.secretion / params.decay * (1 - np.exp(-1.0))
    np.testing.assert_allclose(amount, expected, rtol=1e-3)

    action = env.action_space.sample(jax.random.key(1), (7,))
    obs, state, reward, terminated, truncated, info = jax.jit(env.step)(
        None, state, action, params
    )
    assert obs.shape == (7, 7) and reward.shape == terminated.shape == truncated.shape == (7,)
    assert bool(env.observation_space.contains(obs[0])) and bool(jnp.isfinite(obs).all())
    np.testing.assert_allclose(np.linalg.norm(obs[:, 2:4], axis=1), 1.0, rtol=1e-6)
    assert float(info["distance"]) > 3.0 and info["arrived"].shape == (7,)

    keys = jax.random.split(jax.random.key(2), 3)
    _, batch = jax.vmap(env.reset, in_axes=(0, None))(keys, params)
    actions = env.action_space.sample(jax.random.key(3), (3, 7))
    out = jax.vmap(env.step, in_axes=(0, 0, 0, None))(keys, batch, actions, params)
    assert out[0].shape == (3, 7, 7) and out[2].shape == (3, 7)

    wall = CellMigration(n_leaders=2, n_followers=2, source="top", action="vector")
    obs, state = wall.reset(jax.random.key(0))
    assert wall.action_space.shape == (2,)
    np.testing.assert_allclose(obs[:, 2:4], np.broadcast_to([0.0, 1.0], (4, 2)))
    np.testing.assert_allclose(
        wall.source_distance(state.particles.pos), 10 - state.particles.pos[:, 1]
    )
    spread = CellMigration(n_leaders=5, n_followers=5, start="uniform")
    _, state = spread.reset(jax.random.key(0))
    assert float(spread.source_distance(state.particles.pos).min()) > 1.0
    with pytest.raises(ValueError, match="source"):
        CellMigration(source="left")
    with pytest.raises(ValueError, match="stability"):
        CellMigration(dt=5e-3)


def test_single_leader_moves_as_the_equation_of_motion_says_and_is_paid_per_step():
    env = CellMigration(n_leaders=1)
    params = env.default_params._replace(leader_local_weight=0.0)
    _, state = env.reset(jax.random.key(0), params)
    state = _place(env, state, [[5.0, 7.5]])
    step = jax.jit(env.step)
    # Towards the source at (7.5, 7.5): the direction angle is 0.
    np.testing.assert_allclose(env.gradient_policy(state), [[-1.0]], atol=1e-6)
    np.testing.assert_allclose(env.direction(_angle([[np.pi / 2]])), [[0.0, 1.0]], atol=1e-6)
    _, after, reward, terminated, _, _ = step(None, state, env.gradient_policy(state), params)
    # m dv/dt = (m / tau) (u - v): v = u (1 - exp(-t / tau)).
    np.testing.assert_allclose(after.particles.vel[0, 0], 0.3 * (1 - np.exp(-0.2)), rtol=1e-5)
    np.testing.assert_allclose(
        after.particles.pos[0, 0] - 5.0, 0.3 * (0.1 - 0.5 * (1 - np.exp(-0.2))), rtol=1e-3
    )
    np.testing.assert_allclose(reward, [-0.1])
    assert not bool(terminated[0])

    total, steps, done = float(reward[0]), 1, False
    while not done and steps < 200:
        _, after, reward, terminated, truncated, info = step(
            None, after, env.gradient_policy(after), params
        )
        total, steps, done = total + float(reward[0]), steps + 1, bool(terminated[0])
    # 2 length units at the speed 0.3, plus the relaxation time 0.5, in steps of 0.1.
    assert done and not bool(truncated[0]) and bool(info["arrived"][0])
    np.testing.assert_allclose(steps, (2.0 / 0.3 + 0.5) / 0.1, atol=2)
    # -0.1 per step, and 0 for the step in which the source is reached.
    np.testing.assert_allclose(total, -0.1 * (steps - 1), rtol=1e-6)
    _, _, reward, _, _, _ = step(None, after, env.gradient_policy(after), params)
    np.testing.assert_allclose(reward, [0.0])


def test_cells_attract_through_the_field_and_the_local_reward_follows_the_angle():
    env = CellMigration(n_leaders=1, n_followers=2, source="top")
    params = env.default_params
    _, state = env.reset(jax.random.key(0), params)
    # Two followers side by side; the leader is in the centre of the region.
    state = _place(env, state, [[5.0, 5.0], [1.5, 5.0], [2.5, 5.0]], settle=2.0)
    gradient = np.asarray(env.c3a_gradient(state))
    # Each follower senses the other one more than the leader, which is farther away.
    assert gradient[1, 0] > 0.01 and gradient[2, 0] < -0.01
    np.testing.assert_allclose(gradient[1:, 1], 0.0, atol=1e-6)
    # The leader senses the leaders only, that is itself: no gradient in the
    # centre of the region, where the reflections from the walls cancel.
    np.testing.assert_allclose(gradient[0], 0.0, atol=1e-9)
    obs = env._observe(state)
    np.testing.assert_allclose(obs[:, 4:6], [[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0]], atol=1e-4)
    # Nearer to a wall than to the centre, the reflection of its own secretion draws a cell.
    alone = CellMigration(n_leaders=1)
    _, lone = alone.reset(jax.random.key(0), params)
    lone = _place(alone, lone, [[5.0, 8.5]], settle=2.0)
    pull = np.asarray(alone.c3a_gradient(lone))[0]
    assert pull[1] > 1e-3 and abs(pull[0]) < 1e-9

    # Angle between action and gradient: 0, pi / 3 (and the leader moves up).
    psi = np.arctan2(gradient[1, 1], gradient[1, 0])
    psi2 = np.arctan2(gradient[2, 1], gradient[2, 0])
    action = _angle(jnp.mod(jnp.array([[np.pi / 2], [psi], [psi2 + np.pi / 3]]), 2 * np.pi))
    _, _, reward, _, _, _ = env.step(None, state, action, params)
    np.testing.assert_allclose(reward[1], 0.0, atol=1e-6)
    np.testing.assert_allclose(reward[2], np.exp(-((np.pi / 3) ** 2)) - 1.0, rtol=1e-4)
    # The leader senses no gradient: only the step reward, with the weight 1 - omega_l.
    np.testing.assert_allclose(reward[0], 0.7 * -0.1, rtol=1e-6)
    expected = np.asarray(action[1:, 0]) - np.array([0.0, 1.0 / 3.0])
    np.testing.assert_allclose(env.gradient_policy(state)[1:, 0], expected, atol=1e-5)


def test_contact_inhibition_and_walls_keep_cells_apart():
    env = CellMigration(n_leaders=2, source="top", action="vector")
    params = env.default_params
    _, state = env.reset(jax.random.key(0), params)
    state = _place(env, state, [[4.0, 5.0], [6.0, 5.0]])
    step = jax.jit(env.step)
    # The cells are driven against each other for 10 time units.
    gaps = []
    for _ in range(100):
        _, state, *_ = step(None, state, jnp.array([[1.0, 0.0], [-1.0, 0.0]]), params)
        gaps.append(float(state.particles.pos[1, 0] - state.particles.pos[0, 0]))
    # The contact force is 100 at contact; the driving force is 0.6.
    assert min(gaps) > 0.99 * 2 * env.cell_radius and gaps[-1] < 1.2 * 2 * env.cell_radius
    # Driven into a corner, the cells stay inside the region.
    for _ in range(300):
        _, state, *_ = step(None, state, jnp.array([[-1.0, -1.0], [1.0, -1.0]]), params)
    pos = np.asarray(state.particles.pos)
    assert pos.min() > 0.99 * env.cell_radius and pos.max() < 10 - 0.99 * env.cell_radius
    np.testing.assert_allclose(pos[:, 1], env.cell_radius, atol=0.05)


def test_arrival_spreads_through_touching_leaders():
    env = CellMigration(n_leaders=4, action="vector")
    params = env.default_params._replace(leader_local_weight=0.0)
    _, state = env.reset(jax.random.key(0), params)
    # A chain of three leaders from the source at (7.5, 7.5), and one leader apart.
    chain = [[7.1, 7.5], [6.59, 7.5], [6.08, 7.5], [3.0, 3.0]]
    state = _place(env, state, chain)
    still = jnp.zeros((4, 2))
    arrived = []
    for _ in range(3):
        _, state, reward, terminated, _, info = env.step(None, state, still, params)
        arrived.append(np.asarray(info["arrived"]).tolist())
    assert arrived == [
        [True, False, False, False],
        [True, True, False, False],
        [True, True, True, False],
    ]
    np.testing.assert_allclose(reward, [0.0, 0.0, 0.0, -0.1], atol=1e-6)
    assert not bool(terminated.any())


def _games(env, weight, steps, n_games=6):
    params = env.default_params

    @jax.jit
    @jax.vmap
    def game(key):
        _, state = env.reset(key, params)

        def body(s, _):
            _, s, _, terminated, _, info = env.step(
                None, s, env.gradient_policy(s, weight), params
            )
            pos = s.particles.pos
            centre = jnp.mean(pos, axis=0)
            spread = jnp.sqrt(jnp.mean(jnp.sum((pos - centre) ** 2, axis=1)))
            return s, (terminated[0], info["distance"], spread)

        return jax.lax.scan(body, state, None, length=steps)[1]

    return [np.asarray(x) for x in game(jax.random.split(jax.random.key(0), n_games))]


def test_gradient_following_leaders_reach_the_source_and_co_attraction_keeps_them_together():
    env = CellMigration()
    done, distance, spread = _games(env, 0.0, 300)
    # Reference run: all leaders at the source after 180 to 240 steps.
    assert done[:, -1].all() and done.argmax(axis=1).max() < 280
    together, distance_together, spread_together = _games(env, 0.5, 400)
    assert together[:, -1].all()
    # With co-attraction the group is more compact on the way and slower
    # (reference at step 100: spread 0.69 against 1.01, distance 4.6 against 3.4).
    assert spread_together[:, 99].max() < 0.85 < spread[:, 99].min()
    assert (distance_together[:, 99] - distance[:, 99]).min() > 0.5  # same starts
