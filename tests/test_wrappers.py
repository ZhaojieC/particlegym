"""Gymnasium and PettingZoo wrappers."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from particlegym import envs
from particlegym import physics as pp

gymnasium = pytest.importorskip("gymnasium")
pettingzoo = pytest.importorskip("pettingzoo")

from gymnasium.utils.env_checker import check_env  # noqa: E402
from pettingzoo.test import parallel_api_test  # noqa: E402

from particlegym import wrappers  # noqa: E402

DOOR = pp.Opening("x+", (4.4, 5.6))


@pytest.mark.parametrize(
    "name",
    ["Evacuation2D-v0", "BrickWall-v0", "Billiards-v0", "Herding-v0", "Microswimmer-v0"],
)
def test_registered_environments_pass_the_gymnasium_checker(name):
    env = gymnasium.make(f"particlegym/{name}", render_mode="rgb_array")
    check_env(env.unwrapped, skip_render_check=name == "Billiards-v0")
    obs, _ = env.reset(seed=3)
    again, _ = env.reset(seed=3)
    np.testing.assert_array_equal(obs, again)
    image = env.render()
    assert image.ndim == 3 and image.shape[2] == 3 and image.dtype == np.uint8


def test_gymnasium_env_gives_the_results_of_the_functional_environment():
    env = envs.Billiards(n_balls=3)  # its step is deterministic
    wrapped = wrappers.GymnasiumEnv(env)
    obs, _ = wrapped.reset(seed=0)
    state = wrapped.env_state
    for action in ([0.0, 1.0], [0.3, 0.2], [-0.8, 0.5]):
        action = np.array(action, np.float32)
        expected = env.step(None, state, jnp.asarray(action)[None], env.default_params)
        obs, reward, terminated, truncated, info = wrapped.step(action)
        state = expected[1]
        # The wrapper returns observations in single precision.
        np.testing.assert_allclose(obs, expected[0][0], rtol=1e-5, atol=1e-6)
        assert reward == float(expected[2][0])
        assert terminated == bool(expected[3][0]) and truncated == bool(expected[4][0])
        assert bool(info["rest"])
    with pytest.raises(ValueError, match="one agent"):
        wrappers.GymnasiumEnv(envs.Herding(n_agents=2))
    with pytest.raises(ValueError, match="render_mode"):
        wrappers.GymnasiumEnv(env, render_mode="human")


def test_vector_environment_steps_copies_and_resets_finished_episodes():
    env = envs.Microswimmer()
    params = env.default_params._replace(max_steps=5)
    vector = wrappers.GymnasiumVectorEnv(env, 7, params)
    assert vector.observation_space.shape == (7, 3) and vector.action_space.shape == (7,)
    assert vector.metadata["autoreset_mode"] == gymnasium.vector.AutoresetMode.SAME_STEP
    obs, _ = vector.reset(seed=1)
    again, _ = vector.reset(seed=1)
    np.testing.assert_array_equal(obs, again)
    assert obs.shape == (7, 3) and vector.observation_space.contains(obs)
    for step in range(1, 8):
        actions = vector.action_space.sample()
        obs, reward, terminated, truncated, info = vector.step(actions)
        assert obs.shape == (7, 3) and reward.shape == terminated.shape == truncated.shape == (7,)
        assert not terminated.any() and truncated.all() == (step == 5)
        if step == 5:
            # The copies start again; the last observations are kept.
            assert int(vector.env_state.step.max()) == 0
            assert info["final_obs"].shape == (7, 3) and not np.allclose(info["final_obs"], obs)
        else:
            np.testing.assert_array_equal(info["final_obs"], obs)
    assert int(vector.env_state.step.max()) == 2
    # The copies differ from each other.
    assert np.unique(np.round(obs[:, 0], 6)).size == 7


@pytest.mark.parametrize(
    "env",
    [
        envs.Evacuation(room=(10.0, 10.0), exits=[DOOR], n_agents=5, n_crowd=3),
        envs.CounterFlow(n_agents=4, n_crowd=20),
        envs.Herding(n_sheep=10, n_agents=2, n_neighbors=5),
        envs.CellMigration(n_leaders=2, n_followers=1),
    ],
    ids=["evacuation", "counterflow", "herding", "cell_migration"],
)
def test_environments_pass_the_pettingzoo_parallel_api_test(env):
    wrapped = wrappers.PettingZooEnv(env, render_mode="rgb_array")
    parallel_api_test(wrapped, num_cycles=30)
    assert wrapped.possible_agents == [f"agent_{i}" for i in range(env.n_agents)]
    assert wrapped.render().dtype == np.uint8


def test_agents_that_have_left_are_removed_until_the_episode_ends():
    env = envs.Evacuation(room=(10.0, 10.0), exits=[DOOR], n_agents=6)
    wrapped = wrappers.PettingZooEnv(env)
    observations, _ = wrapped.reset(seed=0)
    assert sorted(observations) == sorted(wrapped.possible_agents)
    # Direction 6 of 8 is +x, towards the door in the right wall.
    assert np.allclose(env.directions[6], [1.0, 0.0])
    door = jnp.array([10.5, 5.0])
    seen_partial = False
    for _ in range(200):
        if not wrapped.agents:
            break
        pos = np.asarray(wrapped.env_state.particles.pos)
        actions = {}
        for name in wrapped.agents:
            to_door = np.asarray(door) - pos[int(name.split("_")[1])]
            actions[name] = int(np.argmax(env.directions @ to_door))
        live = list(wrapped.agents)
        observations, rewards, terminated, _, _ = wrapped.step(actions)
        assert sorted(observations) == sorted(live) == sorted(rewards)
        assert all(name not in wrapped.agents for name in live if terminated[name])
        seen_partial |= 0 < len(wrapped.agents) < 6
    assert seen_partial and not wrapped.agents
    # A new episode has all agents again.
    wrapped.reset(seed=1)
    assert len(wrapped.agents) == 6
    assert jax.tree.all(
        jax.tree.map(lambda x: bool(jnp.all(jnp.isfinite(x))), wrapped.env_state.particles.pos)
    )
