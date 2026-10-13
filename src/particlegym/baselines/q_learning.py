"""Tabular Q-learning for environments with discrete observations and actions.

The table is updated when the observed state of an agent changes, with the
reward collected since the previous change, as in the microswimmer study of
Colabrese et al. (2017):

    Q(s, a) <- Q(s, a) + alpha [R + gamma max_a' Q(s', a') - Q(s, a)],

where the action a was chosen in the state s and kept until the state
changed to s'. The table starts with large equal entries (optimistic
initial values), so that every action is tried, and is carried over from
episode to episode. Every agent of the environment learns its own table.

S. Colabrese, K. Gustavsson, A. Celani and L. Biferale, Flow navigation by
smart microswimmers via reinforcement learning, Phys. Rev. Lett. 118, 158004
(2017).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class QConfig(NamedTuple):
    """Settings of a training run.

    Attributes:
        n_episodes: Number of episodes; each starts from a new reset.
        n_steps: Control steps per episode.
        learning_rate: Step size alpha.
        gamma: Discount factor per change of state.
        initial_value: Initial value of all entries of the table.
        epsilon: Probability of a random action instead of the best one.
    """

    n_episodes: int = 1000
    n_steps: int = 2000
    learning_rate: float = 0.1
    gamma: float = 0.999
    initial_value: float = 100.0
    epsilon: float = 0.0


def greedy(table: jax.Array, obs: jax.Array) -> jax.Array:
    """Best actions for the observations of the agents; ``table`` is (n_agents, S, A)."""
    return jnp.argmax(jnp.take_along_axis(table, obs[:, None, None], axis=1)[:, 0], axis=-1)


def train(key: jax.Array, env, params=None, config: QConfig | None = None):
    """Learn a table of action values for every agent of the environment.

    Args:
        key: JAX random key; the run is a pure function of it.
        env: An environment with discrete observations and actions.
        params: Episode parameters; the defaults of the environment if None.
        config: Settings; the defaults if None.

    Returns:
        ``(table, returns)``: the tables, shape (n_agents, S, A), and the
        sum of the rewards of every episode, shape (n_episodes, n_agents).
    """
    config = QConfig() if config is None else config
    params = env.default_params if params is None else params
    n_states, n_actions, n = env.observation_space.n, env.action_space.n, env.n_agents
    rows = jnp.arange(n)

    def choose(table, obs, key):
        explore_key, action_key = jax.random.split(key)
        explore = jax.random.uniform(explore_key, (n,)) < config.epsilon
        return jnp.where(
            explore, jax.random.randint(action_key, (n,), 0, n_actions), greedy(table, obs)
        )

    def episode(table, key):
        reset_key, first_key, run_key = jax.random.split(key, 3)
        obs, state = env.reset(reset_key, params)
        action = choose(table, obs, first_key)

        def body(carry, key):
            table, state, obs, action, collected, total = carry
            step_key, action_key = jax.random.split(key)
            new_obs, state, reward, _, _, _ = env.step(step_key, state, action, params)
            reward = reward.astype(table.dtype)
            collected, total = collected + reward, total + reward
            changed = new_obs != obs
            best = jnp.max(table[rows, new_obs], axis=-1)
            old = table[rows, obs, action]
            target = collected + config.gamma * best
            update = jnp.where(changed, config.learning_rate * (target - old), 0.0)
            table = table.at[rows, obs, action].add(update)
            action = jnp.where(changed, choose(table, new_obs, action_key), action)
            collected = jnp.where(changed, 0.0, collected)
            return (table, state, new_obs, action, collected, total), None

        zeros = jnp.zeros(n, table.dtype)
        carry = (table, state, obs, action, zeros, zeros)
        carry, _ = jax.lax.scan(body, carry, jax.random.split(run_key, config.n_steps))
        return carry[0], carry[5]

    table = jnp.full((n, n_states, n_actions), config.initial_value, jnp.float32)
    return jax.lax.scan(episode, table, jax.random.split(key, config.n_episodes))
