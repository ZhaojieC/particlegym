"""Learning baselines: PPO and tabular Q-learning."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from cartpole import CartPole
from particlegym import envs
from particlegym import physics as pp
from particlegym.baselines import ppo, q_learning
from particlegym.envs.spaces import Discrete

TINY = ppo.PPOConfig(n_envs=4, n_steps=8, n_iterations=4, n_minibatches=2, n_epochs=2)


def test_gae_matches_a_direct_sum():
    rng = np.random.default_rng(0)
    n = 12
    value, reward, next_value = rng.normal(size=(3, n, 2))
    terminated = np.zeros((n, 2), bool)
    truncated = np.zeros((n, 2), bool)
    terminated[4, 0] = truncated[7, 0] = terminated[9, 1] = True
    ended = terminated | truncated
    gamma, lam = 0.9, 0.8

    expected = np.zeros((n, 2))
    for j in range(2):
        for t in range(n):
            total, weight = 0.0, 1.0
            for k in range(t, n):
                follows = 0.0 if terminated[k, j] else next_value[k, j]
                total += weight * (reward[k, j] + gamma * follows - value[k, j])
                weight *= gamma * lam
                if ended[k, j]:
                    break
            expected[t, j] = total

    advantage, target = ppo.gae(
        jnp.asarray(value),
        jnp.asarray(reward),
        jnp.asarray(next_value),
        jnp.asarray(terminated),
        jnp.asarray(ended),
        gamma,
        lam,
    )
    np.testing.assert_allclose(advantage, expected, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(target, expected + value, rtol=1e-6, atol=1e-9)


def test_ppo_learns_to_balance_the_pole():
    """Cart-pole: the largest return is 500, and 475 counts as solved."""
    env = CartPole()
    config = ppo.PPOConfig(
        n_envs=16,
        n_steps=128,
        n_iterations=100,
        n_minibatches=32,
        n_epochs=10,
        entropy=0.0,
        normalize=False,
    )
    train = jax.jit(jax.vmap(lambda key: ppo.train(key, env, None, config)))
    policies, curves = train(jax.random.split(jax.random.key(0), 3))

    def play(policy, key):
        def policy_fn(key, obs, state):
            return ppo.act(policy, env, obs, normalize=False)

        return ppo.evaluate(key, env, None, policy_fn, 32, 500)

    stats = jax.jit(jax.vmap(play))(policies, jax.random.split(jax.random.key(1), 3))
    returns = np.asarray(stats["episode_return"]).mean(axis=1)
    first = np.asarray(curves["episode_return"])[:, 0]
    assert np.all(first < 40.0), first  # a random policy drops the pole after about 22 steps
    assert np.all(returns > 475.0), returns


def test_training_is_a_function_of_the_key_and_can_be_cut_into_pieces():
    env = CartPole()
    key = jax.random.key(3)
    policy, curves = ppo.train(key, env, None, TINY)
    again, _ = ppo.train(key, env, None, TINY)
    other, _ = ppo.train(jax.random.key(4), env, None, TINY)
    for a, b in zip(jax.tree.leaves(policy), jax.tree.leaves(again), strict=True):
        np.testing.assert_array_equal(a, b)
    assert not np.allclose(policy.actor[0][0], other.actor[0][0])

    state = ppo.init(key, env, None, TINY)
    state, first = ppo.run(state, env, None, TINY, 1)
    state, rest = ppo.run(state, env, None, TINY, 3)
    assert int(state.iteration) == 4
    for a, b in zip(jax.tree.leaves(policy), jax.tree.leaves(state.policy), strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-7)
    joined = np.concatenate([first["reward"], rest["reward"]])
    np.testing.assert_allclose(joined, curves["reward"], rtol=1e-6)


def test_ppo_runs_with_several_agents_that_finish_at_different_times():
    env = envs.Evacuation(room=(4.0, 4.0), exits=(pp.Opening("x+", (1.4, 2.6)),), n_agents=3)
    params = env.default_params._replace(max_steps=12)
    policy, curves = jax.jit(lambda key: ppo.train(key, env, params, TINY))(jax.random.key(0))
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree.leaves(policy))
    assert np.all(np.isfinite(curves["reward"]))
    # Episodes end after at most 12 steps, so every update of 8 steps sees some end.
    assert np.asarray(curves["episodes"]).sum() >= 4
    assert np.nanmax(curves["episode_length"]) <= 12
    # The return per agent is the penalty of 0.1 per step in the room.
    finite = np.isfinite(curves["episode_return"])
    assert np.all(np.asarray(curves["episode_return"])[finite] >= -1.2 - 1e-6)

    action = ppo.act(policy, env, env.reset(jax.random.key(1), params)[0])
    assert action.shape == (3,) and env.action_space.contains(action[0])


def test_ppo_runs_with_continuous_actions_and_stores_its_policy(tmp_path):
    env = envs.Microswimmer(n_agents=2, action="continuous")
    params = env.default_params._replace(max_steps=10)
    policy, curves = jax.jit(lambda key: ppo.train(key, env, params, TINY))(jax.random.key(0))
    assert np.all(np.isfinite(curves["reward"]))
    obs, _ = env.reset(jax.random.key(1), params)
    best = ppo.act(policy, env, obs)
    noisy = ppo.act(policy, env, obs, jax.random.key(2))
    assert best.shape == noisy.shape == (2, 2)
    assert bool(jnp.all(jnp.abs(noisy) <= 1.0))
    assert not np.allclose(best, noisy)

    ppo.save_policy(tmp_path / "policy.npz", policy)
    loaded = ppo.load_policy(tmp_path / "policy.npz", env, TINY)
    np.testing.assert_array_equal(ppo.act(loaded, env, obs), best)


def test_evaluate_counts_the_rewards_until_the_episode_ends():
    env = CartPole()

    def push_right(key, obs, state):
        return jnp.ones(1, jnp.int32)

    stats = ppo.evaluate(jax.random.key(0), env, None, push_right, 8, 100)
    # With a constant push the pole falls within a few steps; one reward per step.
    np.testing.assert_allclose(stats["episode_return"], stats["episode_length"])
    assert bool(jnp.all(stats["finished"])) and int(jnp.max(stats["episode_length"])) < 20


class _RingState(NamedTuple):
    site: jax.Array


class _Ring:
    """Four sites on a ring: action 0 steps forwards for a reward of 1, action 1 backwards."""

    n_agents = 2
    observation_space = Discrete(4)
    action_space = Discrete(2)
    default_params = None

    def reset(self, key, params=None):
        site = jax.random.randint(key, (2,), 0, 4)
        return site, _RingState(site)

    def step(self, key, state, action, params=None):
        site = (state.site + jnp.where(action == 0, 1, -1)) % 4
        reward = jnp.where(action == 0, 1.0, 0.0)
        return site, _RingState(site), reward, jnp.zeros(2, bool), jnp.zeros(2, bool), {}


def test_q_learning_finds_the_values_of_a_small_problem():
    config = q_learning.QConfig(
        n_episodes=10, n_steps=300, learning_rate=0.5, gamma=0.9, initial_value=100.0
    )
    tables, returns = q_learning.train(jax.random.key(0), _Ring(), None, config)
    assert tables.shape == (2, 4, 2) and returns.shape == (10, 2)
    np.testing.assert_array_equal(jnp.argmax(tables, axis=-1), 0)
    # Always forwards: 1 + 0.9 + 0.81 + ... = 10.
    np.testing.assert_allclose(tables[:, :, 0], 10.0, rtol=1e-3)
    np.testing.assert_allclose(returns[-1], 300.0)
    obs = jnp.array([2, 3])
    np.testing.assert_array_equal(q_learning.greedy(tables, obs), 0)


def test_q_learning_swimmers_climb_faster_than_naive_swimmers():
    """The direction of the result of Colabrese et al. (2017) at Phi = 0.3, Psi = 1.

    A short run: three seeds gave 0.37 to 0.43 for the learners and 0.18 to
    0.26 for the naive swimmers. The run of ``examples/learning`` is longer.
    """
    env = envs.Microswimmer(n_agents=16, observation="discrete")
    params = env.default_params
    config = q_learning.QConfig(n_episodes=200, n_steps=500, learning_rate=0.1, gamma=0.99)
    train = jax.jit(lambda key: q_learning.train(key, env, params, config))
    _, returns = train(jax.random.key(0))

    def naive_policy(key, obs, state):
        return env.naive_policy()

    stats = ppo.evaluate(jax.random.key(10), env, params, naive_policy, 8, 500)
    # Altitude gained in 50 time units, as a speed in units of the swim speed.
    to_speed = 1.0 / (50.0 * params.swim_speed)
    learners = to_speed * float(jnp.mean(returns[-50:]))
    naive = to_speed * float(jnp.mean(stats["episode_return"]))
    assert learners > 1.3 * naive, (learners, naive)
