"""Rooms of published evacuation studies: layouts and evacuation by the crowd."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from particlegym import envs

DIAG = math.sqrt(2) / 2
# The eight actions of the code of Zhang, Chai & Lykotrafitis (2021), in its order.
LEGACY_ACTIONS = [
    [0, 1], [-DIAG, DIAG], [-1, 0], [-DIAG, -DIAG], [0, -1], [DIAG, -DIAG], [1, 0], [DIAG, DIAG]
]  # fmt: skip
# Exits and obstacles of that code, in units of the room size of 10 m. The
# obstacle sizes are diameters in metres.
LEGACY_LAYOUTS = {
    "four_exits": {"exits": [(0.5, 1.0), (0.5, 0.0), (0.0, 0.5), (1.0, 0.5)], "obstacles": []},
    "three_exits_two_obstacles": {
        "exits": [(0.7, 1.0), (0.5, 0.0), (0.0, 0.7)],
        "obstacles": [((0.8, 0.8), 2.0), ((0.3, 0.5), 3.0)],
    },
}


def _obstacles(env):
    """Centres (M, 2) and radii (M,) of all round obstacles of a room."""
    centres = np.concatenate([s.centers for s in env.obstacles] or [np.zeros((0, 2))])
    radii = np.concatenate([s.radii for s in env.obstacles] or [np.zeros(0)])
    return centres, radii


@pytest.mark.parametrize("layout", list(LEGACY_LAYOUTS))
def test_layouts_and_parameters_are_those_of_the_paper(layout):
    env = envs.zhang2021(layout)
    legacy = LEGACY_LAYOUTS[layout]

    assert env.size == (10.0, 10.0)
    np.testing.assert_allclose(env.door_centres, 10.0 * np.array(legacy["exits"]))
    np.testing.assert_allclose([np.ptp(door.span) for door in env.exits], 1.0)
    centres, radii = _obstacles(env)
    np.testing.assert_allclose(
        centres, 10.0 * np.array([c for c, _ in legacy["obstacles"]]).reshape(-1, 2)
    )
    np.testing.assert_allclose(radii, [size / 2 for _, size in legacy["obstacles"]])

    # One agent alone in the room, with the pedestrians, actions, rewards and
    # decision interval of the paper.
    assert (env.n_agents, env.n_crowd) == (1, 0)
    assert env.radius == (0.25, 0.25) and env.mass == 80.0
    np.testing.assert_allclose(env.directions, LEGACY_ACTIONS, atol=1e-15)
    params = env.default_params
    assert (params.desired_speed, params.relaxation_time) == (2.0, 0.5)
    assert (params.step_penalty, params.exit_reward) == (-0.1, 0.0)
    assert env.dt * env.substeps == pytest.approx(0.1)


def test_other_arguments_are_passed_on_and_the_layout_is_fixed():
    env = envs.zhang2021(
        "three_exits_two_obstacles",
        n_agents=2,
        n_crowd=78,
        crowd_policy="stand",
        action="continuous",
        radius=(0.2, 0.3),
    )
    assert (env.n_agents, env.n_crowd, env.crowd_policy) == (2, 78, "stand")
    assert env.action == "continuous" and env.radius == (0.2, 0.3)
    with pytest.raises(TypeError, match="exits"):
        envs.zhang2021("four_exits", exits=[])
    with pytest.raises(ValueError, match="layout must be one of"):
        envs.zhang2021("five_exits")


@pytest.mark.parametrize(
    ("layout", "time_limit"), [("four_exits", 25.0), ("three_exits_two_obstacles", 35.0)]
)
def test_80_pedestrians_walking_to_the_nearest_door_leave_through_the_doors(layout, time_limit):
    env = envs.zhang2021(layout, n_agents=0, n_crowd=80, action="continuous")
    _, state = jax.jit(env.reset)(jax.random.key(0))
    pos = np.asarray(state.particles.pos)
    radius = np.asarray(state.particles.radius)
    centres, radii = _obstacles(env)
    gap = np.linalg.norm(pos[:, None] - centres[None], axis=-1) - radii - radius[:, None]
    assert np.all(gap > 0.0)  # nobody starts inside an obstacle

    def body(s, key):
        _, s, _, _, _, info = env.step(key, s, jnp.zeros((0, 2)))
        return s, info["overflow"]

    keys = jax.random.split(jax.random.key(1), round(time_limit / 0.1))
    state, overflow = jax.jit(lambda s: jax.lax.scan(body, s, keys))(state)

    exit_time = np.asarray(state.exit_time)
    assert np.all(np.isfinite(exit_time)) and not np.asarray(overflow).any()
    # Pedestrians are frozen where they were removed: just outside a doorway.
    pos = np.asarray(state.particles.pos)
    through_a_door = np.zeros(len(pos), bool)
    for door, centre in zip(env.exits, env.door_centres, strict=True):
        axis = "xy".index(door.face[0])
        outside = pos[:, axis] < 0.0 if door.face[1] == "-" else pos[:, axis] > 10.0
        along = np.abs(pos[:, 1 - axis] - centre[1 - axis]) < 0.5 + radius
        through_a_door |= outside & along
    assert through_a_door.all()
