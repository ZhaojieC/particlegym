"""The evacuation environment in 3D: API contract, directions and consistency with 2D."""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from particlegym import envs
from particlegym import physics as pp

ROOM = (10.0, 10.0, 4.0)
# A door 1 m wide and 2 m high in the wall at y = 10, standing on the floor.
DOOR = pp.Opening("y+", ((4.5, 5.5), (0.0, 2.0)))
# A 2 m x 2 m hatch in the middle of the ceiling.
HATCH = pp.Opening("z+", ((4.0, 6.0), (4.0, 6.0)))


def _env(**kwargs):
    kwargs.setdefault("exits", [DOOR])
    return envs.Evacuation(room=ROOM, **kwargs)


def _place(env, state, pos):
    """Move the pedestrians of `state` to `pos` and update the forces."""
    particles = state.particles._replace(pos=jnp.asarray(pos))
    particles, _ = jax.jit(pp.init_forces, static_argnums=1)(particles, env.force_field)
    return state._replace(particles=particles)


def _rollout(env, state, actions_fn, params, steps, seed=0):
    """Run `steps` compiled control steps; `actions_fn(state)` gives the actions."""

    def body(s, key):
        _, s, reward, terminated, truncated, info = env.step(key, s, actions_fn(s), params)
        return s, (reward, terminated, truncated, info["inside"], s.particles.pos)

    keys = jax.random.split(jax.random.key(seed), steps)
    return jax.jit(lambda s: jax.lax.scan(body, s, keys))(state)


def test_reset_places_pedestrians_inside_without_overlap():
    pillar = pp.Spheres([[3.0, 3.0, 2.0]], [1.0])
    env = _env(n_agents=3, n_crowd=60, obstacles=[pillar], radius=(0.25, 0.3), min_gap=0.2)
    obs, state = jax.jit(env.reset)(jax.random.key(0))
    p = state.particles
    pos, radius = np.asarray(p.pos), np.asarray(p.radius)

    assert env.dim == 3 and env.size == ROOM
    assert obs.shape == (3, 6)
    assert p.pos.shape == (63, 3)
    assert np.all((pos > radius[:, None]) & (pos < np.array(ROOM) - radius[:, None]))
    assert np.ptp(pos[:, 2]) > 2.0  # the pedestrians fill the height of the room
    gaps = np.linalg.norm(pos[:, None] - pos[None], axis=-1) - (radius[:, None] + radius[None])
    assert gaps[~np.eye(63, dtype=bool)].min() >= 0.2 - 1e-9
    assert np.all(np.linalg.norm(pos - [3.0, 3.0, 2.0], axis=1) - 1.0 > radius)
    np.testing.assert_allclose(obs[:, :3], pos[:3] / np.array(ROOM) - 0.5)
    np.testing.assert_allclose(obs[:, 3:], 0.0)


@pytest.mark.parametrize(("n", "in_plane", "max_nonzero"), [(6, 4, 1), (18, 8, 2), (26, 8, 3)])
def test_discrete_directions_point_to_the_faces_edges_and_corners_of_a_cube(
    n, in_plane, max_nonzero
):
    directions = _env(n_directions=n).directions
    assert directions.shape == (n, 3)
    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0)
    cube = [
        np.array(v) / np.linalg.norm(v)
        for v in itertools.product((-1, 0, 1), repeat=3)
        if 0 < np.count_nonzero(v) <= max_nonzero
    ]
    assert len(cube) == n
    # Every direction of the cube appears exactly once.
    match = np.isclose(directions[:, None, :], np.array(cube)[None]).all(axis=-1)
    np.testing.assert_array_equal(match.sum(axis=0), 1)

    # The directions in the xy plane come first, in the order of the 2D room.
    planar = envs.Evacuation(room=(10.0, 10.0), n_directions=in_plane).directions
    np.testing.assert_allclose(directions[:in_plane, :2], planar, atol=1e-12)
    np.testing.assert_allclose(directions[:in_plane, 2], 0.0)
    # Then +z and the directions tilted towards it, then the same for -z.
    half = (n - in_plane) // 2
    np.testing.assert_allclose(directions[in_plane], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(directions[in_plane + half], [0.0, 0.0, -1.0])
    assert np.all(directions[in_plane : in_plane + half, 2] > 0)
    assert np.all(directions[in_plane + half :, 2] < 0)
    np.testing.assert_allclose(
        directions[in_plane + half :], directions[in_plane : in_plane + half] * [1, 1, -1]
    )


def test_default_number_of_directions_depends_on_the_dimension():
    assert envs.Evacuation(room=(10.0, 10.0)).directions.shape == (8, 2)
    assert _env().directions.shape == (26, 3)


def test_actions_set_the_desired_velocity():
    params = envs.EvacuationParams(desired_speed=1.5)
    discrete = _env(n_agents=3)
    velocity = discrete.agent_velocity(jnp.array([6, 8, 10]), params, jnp.float64)
    corner = 1.5 / np.sqrt(3.0)  # direction 10 points to the corner (-1, 1, 1) of the cube
    np.testing.assert_allclose(
        velocity, [[1.5, 0.0, 0.0], [0.0, 0.0, 1.5], [-corner, corner, corner]], atol=1e-12
    )
    continuous = _env(n_agents=2, action="continuous")
    velocity = continuous.agent_velocity(
        jnp.array([[0.0, 0.0, -0.5], [3.0, 4.0, 12.0]]), params, jnp.float64
    )
    # Lengths below 1 scale the desired speed; longer vectors are capped at it.
    np.testing.assert_allclose(
        velocity, [[0.0, 0.0, -0.75], np.array([3.0, 4.0, 12.0]) * 1.5 / 13.0], atol=1e-12
    )


def test_door_centres_and_targets():
    side = pp.Opening("x-", ((2.0, 3.0), (1.0, 3.0)))
    env = _env(exits=[DOOR, HATCH, side])
    np.testing.assert_allclose(
        env.door_centres, [[5.0, 10.0, 1.0], [5.0, 5.0, 4.0], [0.0, 2.5, 2.0]]
    )
    np.testing.assert_allclose(
        env.door_targets, [[5.0, 11.0, 1.0], [5.0, 5.0, 5.0], [-1.0, 2.5, 2.0]]
    )


def test_agents_leave_through_the_door_but_not_through_the_wall_above_it():
    env = _env(n_agents=2, action="continuous")
    params = env.default_params._replace(exit_reward=5.0)
    _, state = jax.jit(env.reset)(jax.random.key(3), params)
    # Both agents fly in the +y direction: the first at the height of the door,
    # the second 1 m above the top of the door.
    state = _place(env, state, [[5.0, 5.0, 1.0], [5.0, 5.0, 3.0]])
    forward = jnp.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])

    state, (reward, terminated, truncated, _, _) = _rollout(
        env, state, lambda s: forward, params, 100
    )
    reward, terminated = np.asarray(reward), np.asarray(terminated)
    exit_step = int(np.argmax(terminated[:, 0]))

    assert terminated[-1, 0] and not terminated[:, 1].any()
    assert not np.asarray(truncated).any()
    np.testing.assert_allclose(reward[:exit_step, 0], -0.1)
    assert reward[exit_step, 0] == 5.0
    np.testing.assert_allclose(reward[exit_step + 1 :, 0], 0.0)
    np.testing.assert_allclose(reward[:, 1], -0.1)
    np.testing.assert_allclose(state.exit_time[0], (exit_step + 1) * env.dt * env.substeps)
    assert np.isinf(state.exit_time[1])
    x, y, z = np.asarray(state.particles.pos).T
    assert 4.5 < x[0] < 5.5 and y[0] > 10.0 and 0.0 < z[0] < 2.0  # left through the door
    assert 9.0 < y[1] < 10.0 and 2.5 < z[1] < 3.5  # stopped by the wall above the door


def test_crowd_leaves_only_through_the_exits():
    door = pp.Opening("y+", ((4.4, 5.6), (0.0, 2.0)))
    env = _env(exits=[door, HATCH], n_agents=0, n_crowd=40, radius=(0.25, 0.3))
    obs, state = jax.jit(env.reset)(jax.random.key(5))
    assert obs.shape == (0, 6)
    state, (_, _, _, inside, _) = _rollout(
        env, state, lambda s: jnp.zeros((0,), int), env.default_params, 600
    )
    pos = np.asarray(state.particles.pos)
    radius = np.asarray(state.particles.radius)

    assert int(inside[-1]) == 0
    assert np.all(np.isfinite(np.asarray(state.exit_time)))
    # Pedestrians are frozen where they were removed: just outside an exit.
    through_door = (
        (pos[:, 1] > 10.0)
        & (np.abs(pos[:, 0] - 5.0) < 0.6 + radius)
        & (np.abs(pos[:, 2] - 1.0) < 1.0 + radius)
    )
    through_hatch = (
        (pos[:, 2] > 4.0)
        & (np.abs(pos[:, 0] - 5.0) < 1.0 + radius)
        & (np.abs(pos[:, 1] - 5.0) < 1.0 + radius)
    )
    assert np.all(through_door | through_hatch)
    assert through_door.any() and through_hatch.any()


def test_random_crowd_directions_are_uniform_on_the_sphere():
    env = envs.Evacuation(room=(20.0, 20.0, 20.0), n_agents=0, n_crowd=4000, crowd_policy="random")
    params = envs.EvacuationParams(desired_speed=1.2)
    particles = pp.create_state(jnp.zeros((4000, 3)))
    velocity = np.asarray(env.crowd_velocity(jax.random.key(1), particles, params))
    direction = velocity / 1.2

    np.testing.assert_allclose(np.linalg.norm(direction, axis=1), 1.0)
    # For a uniform distribution, each component has mean 0 and variance 1/3.
    np.testing.assert_allclose(direction.mean(axis=0), 0.0, atol=4 / np.sqrt(3 * 4000))
    np.testing.assert_allclose((direction**2).mean(axis=0), 1 / 3, atol=0.02)


def test_reset_and_step_compile_and_batch():
    env = _env(n_agents=2, n_crowd=5)
    keys = jax.random.split(jax.random.key(6), 4)
    obs, states = jax.jit(jax.vmap(env.reset))(keys)
    actions = jnp.array([[0, 8], [17, 4], [9, 25], [3, 12]])

    out = jax.jit(jax.vmap(env.step))(keys, states, actions)
    obs, next_states, reward, terminated, truncated, info = out

    assert obs.shape == (4, 2, 6)
    assert reward.shape == terminated.shape == truncated.shape == (4, 2)
    assert info["inside"].shape == (4,)
    np.testing.assert_array_equal(next_states.step, 1)
    single = jax.jit(env.step)(keys[1], jax.tree.map(lambda x: x[1], states), actions[1])
    np.testing.assert_allclose(single[0], obs[1], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(single[1].particles.pos, next_states.particles.pos[1], rtol=1e-9)
    np.testing.assert_allclose(single[2], reward[1])


def test_large_crowds_use_a_cell_list_that_gives_the_brute_force_forces():
    small = envs.Evacuation(room=(30.0, 30.0, 10.0), n_crowd=600)
    assert isinstance(small.force_field.neighbors, pp.BruteForce)
    env = envs.Evacuation(room=(30.0, 30.0, 10.0), n_agents=0, n_crowd=2100)
    assert isinstance(env.force_field.neighbors, pp.CellList)

    _, state = jax.jit(env.reset)(jax.random.key(0))
    p = state.particles
    force_field = env.force_field
    brute = pp.ForceField(env.box, pair=force_field.pair, walls=force_field.walls)
    reference, _ = jax.jit(brute.forces)(p.pos, p.vel, p.radius, p.alive)
    _, info = jax.jit(force_field.forces)(p.pos, p.vel, p.radius, p.alive)

    assert not bool(info.overflow)
    assert float(jnp.abs(reference).max()) > 10.0  # neighbours do interact
    np.testing.assert_allclose(p.force, reference, rtol=1e-9, atol=1e-9)


def test_mid_plane_of_a_3d_room_reproduces_the_2d_room():
    """Pedestrians in the mid-plane of a 3D room with full-height doors move as in 2D.

    The floor and the ceiling are 1.5 m away, beyond the range of the wall
    force, and all other forces lie in the plane.
    """
    kwargs = {"n_agents": 2, "n_crowd": 12, "action": "continuous", "radius": (0.25, 0.3)}
    flat = envs.Evacuation(
        room=(10.0, 10.0),
        exits=[pp.Opening("y+", (4.5, 5.5)), pp.Opening("x-", (2.0, 3.0))],
        **kwargs,
    )
    room = envs.Evacuation(
        room=(10.0, 10.0, 3.0),
        exits=[
            pp.Opening("y+", ((4.5, 5.5), (0.0, 3.0))),
            pp.Opening("x-", ((2.0, 3.0), (0.0, 3.0))),
        ],
        **kwargs,
    )
    _, flat_state = jax.jit(flat.reset)(jax.random.key(9))
    p = flat_state.particles
    lifted = pp.create_state(
        jnp.column_stack([p.pos, jnp.full(p.n, 1.5)]), mass=p.mass, radius=p.radius, kind=p.kind
    )
    lifted, _ = jax.jit(pp.init_forces, static_argnums=1)(lifted, room.force_field)
    room_state = envs.EvacuationState(lifted, flat_state.step, flat_state.exit_time)

    steps = 80
    action = jnp.array([[0.6, 0.8], [-1.0, 0.0]])
    flat_final, flat_out = _rollout(flat, flat_state, lambda s: action, flat.default_params, steps)
    action_3d = jnp.column_stack([action, jnp.zeros(2)])
    room_final, room_out = _rollout(
        room, room_state, lambda s: action_3d, room.default_params, steps
    )

    assert int(flat_out[3][-1]) < 14  # some pedestrians have left, so doors are compared too
    np.testing.assert_allclose(room_out[4][..., :2], flat_out[4], rtol=0, atol=1e-8)
    np.testing.assert_allclose(room_out[4][..., 2], 1.5, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(room_final.particles.alive, flat_final.particles.alive)
    np.testing.assert_allclose(room_final.exit_time, flat_final.exit_time)
    np.testing.assert_allclose(room_out[0], flat_out[0])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"room": (10.0, 10.0, 4.0, 2.0)},
        {"room": (10.0, 10.0, 0.0)},
        {"n_directions": 8},
        {"exits": [pp.Opening("y+", (4.5, 5.5))]},
        {"exits": [pp.Opening("w+", ((4.5, 5.5), (0.0, 2.0)))]},
        {"obstacles": [pp.Spheres([[3.0, 3.0]], [1.0])]},
    ],
)
def test_invalid_configurations_are_rejected(kwargs):
    kwargs.setdefault("room", ROOM)
    with pytest.raises(ValueError):
        envs.Evacuation(**kwargs)
