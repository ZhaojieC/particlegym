"""Proximal policy optimisation (PPO) in JAX, in one file.

A compact implementation for the functional environments of ParticleGym. The
environments, the roll-outs and the updates compile into one program, and
``jax.vmap`` over the random key trains several seeds at once:

    train = jax.jit(jax.vmap(lambda key: ppo.train(key, env, params, config)))
    policies, curves = train(jax.random.split(key, 5))

A long run can be cut into pieces with :func:`init` and :func:`run`, to save
the policy in between.

All agents of an environment share one policy and learn from their own
observations and rewards (independent learners with shared parameters). The
networks are small multilayer perceptrons, so pixel observations are not
supported.

The method: the clipped objective of Schulman et al. (2017) with generalized
advantage estimation (Schulman et al. 2016) and the Adam optimiser (Kingma
and Ba 2015). An episode that is cut off (truncated) is not treated as a
failure: the value of the last observation is used in its place.

J. Schulman, F. Wolski, P. Dhariwal, A. Radford and O. Klimov, Proximal
policy optimization algorithms, arXiv:1707.06347 (2017).
J. Schulman, P. Moritz, S. Levine, M. Jordan and P. Abbeel, High-dimensional
continuous control using generalized advantage estimation, arXiv:1506.02438.
D. P. Kingma and J. Ba, Adam: a method for stochastic optimization,
arXiv:1412.6980.
"""

from __future__ import annotations

import itertools
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from particlegym.envs.spaces import Discrete


class PPOConfig(NamedTuple):
    """Settings of a training run.

    Attributes:
        n_envs: Number of environments stepped in parallel.
        n_steps: Control steps per environment between two updates.
        n_iterations: Number of updates.
        n_epochs: Passes over the collected data per update.
        n_minibatches: Minibatches per pass.
        gamma: Discount factor.
        gae_lambda: Decay of the advantage estimator.
        clip: Clipping range of the probability ratio.
        entropy: Weight of the entropy bonus.
        value: Weight of the value loss.
        learning_rate: Step size of Adam; decreases linearly to zero if
            ``anneal`` is set.
        max_grad_norm: Largest norm of the gradient.
        hidden: Widths of the hidden layers of policy and value network.
        anneal: Decrease the learning rate linearly to zero.
        log_std: Initial logarithm of the standard deviation of continuous
            actions.
        normalize: Divide every entry of the observations by its running
            root mean square (not used for discrete observations). The mean
            is not subtracted, so that the origin of positions stays fixed.
        reward_scale: Factor applied to the rewards for learning. The curves
            report the rewards of the environment, without this factor.
    """

    n_envs: int = 256
    n_steps: int = 64
    n_iterations: int = 200
    n_epochs: int = 4
    n_minibatches: int = 8
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    entropy: float = 0.01
    value: float = 0.5
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5
    hidden: tuple[int, ...] = (64, 64)
    anneal: bool = True
    log_std: float = -0.5
    normalize: bool = True
    reward_scale: float = 1.0


class Policy(NamedTuple):
    """Parameters of a trained policy. A pytree.

    Attributes:
        actor: Layers ``(weights, bias)`` of the policy network.
        critic: Layers of the value network.
        log_std: Logarithm of the standard deviation of continuous actions.
        square: Running mean of the squared observations.
        count: Number of observations in the running mean.
    """

    actor: list
    critic: list
    log_std: jax.Array
    square: jax.Array
    count: jax.Array


def features(env, obs: jax.Array) -> jax.Array:
    """Network input for observations of shape ``(..., n_agents, *obs_shape)``."""
    space = env.observation_space
    if isinstance(space, Discrete):
        return jax.nn.one_hot(obs, space.n)
    batch = obs.shape[: obs.ndim - len(space.shape)]
    return jnp.reshape(obs, (*batch, -1)).astype(jnp.float32)


def _n_features(env) -> int:
    space = env.observation_space
    return space.n if isinstance(space, Discrete) else int(np.prod(space.shape))


def _n_outputs(env) -> int:
    space = env.action_space
    return space.n if isinstance(space, Discrete) else int(np.prod(space.shape))


def _layers(key, sizes, last_scale):
    layers = []
    for i, (n_in, n_out) in enumerate(itertools.pairwise(sizes)):
        key, sub = jax.random.split(key)
        scale = last_scale if i == len(sizes) - 2 else np.sqrt(2.0)
        weights = jax.nn.initializers.orthogonal(scale)(sub, (n_in, n_out), jnp.float32)
        layers.append((weights, jnp.zeros(n_out, jnp.float32)))
    return layers


def _mlp(layers, x):
    for weights, bias in layers[:-1]:
        x = jnp.tanh(x @ weights + bias)
    weights, bias = layers[-1]
    return x @ weights + bias


def init_policy(key: jax.Array, env, config: PPOConfig) -> Policy:
    """A policy with random weights for the given environment."""
    n_in, n_out = _n_features(env), _n_outputs(env)
    actor_key, critic_key = jax.random.split(key)
    return Policy(
        actor=_layers(actor_key, (n_in, *config.hidden, n_out), 0.01),
        critic=_layers(critic_key, (n_in, *config.hidden, 1), 1.0),
        log_std=jnp.full(n_out, config.log_std, jnp.float32),
        square=jnp.ones(n_in, jnp.float32),
        count=jnp.asarray(1e-4, jnp.float32),
    )


def _scaled(policy: Policy, x: jax.Array, normalize: bool) -> jax.Array:
    if not normalize:
        return x
    return jnp.clip(x / jnp.sqrt(policy.square + 1e-8), -10.0, 10.0)


def _outputs(policy: Policy, env, obs, normalize: bool):
    """Policy output (logits or means) and values for observations."""
    discrete = isinstance(env.observation_space, Discrete)
    x = _scaled(policy, features(env, obs), normalize and not discrete)
    return _mlp(policy.actor, x), _mlp(policy.critic, x)[..., 0]


def _log_prob_entropy(policy: Policy, env, out, action):
    """Log-probability of the actions and entropy of the distributions."""
    if isinstance(env.action_space, Discrete):
        log_p = jax.nn.log_softmax(out)
        chosen = jnp.take_along_axis(log_p, action[..., None], axis=-1)[..., 0]
        return chosen, -jnp.sum(jnp.exp(log_p) * log_p, axis=-1)
    z = (action.reshape(out.shape) - out) / jnp.exp(policy.log_std)
    log_p = -0.5 * z**2 - policy.log_std - 0.5 * jnp.log(2.0 * jnp.pi)
    entropy = jnp.sum(policy.log_std + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e))
    return jnp.sum(log_p, axis=-1), jnp.broadcast_to(entropy, out.shape[:-1])


def _sample(policy: Policy, env, out, key):
    space = env.action_space
    if isinstance(space, Discrete):
        return jax.random.categorical(key, out)
    noise = jax.random.normal(key, out.shape, out.dtype)
    return (out + jnp.exp(policy.log_std) * noise).reshape(*out.shape[:-1], *space.shape)


def _to_env(env, action):
    """The action that is passed to the environment: within its bounds."""
    space = env.action_space
    if isinstance(space, Discrete):
        return action
    return jnp.clip(action, space.low, space.high)


def act(
    policy: Policy, env, obs: jax.Array, key: jax.Array | None = None, normalize: bool = True
) -> jax.Array:
    """Actions of the policy for observations of shape ``(..., n_agents, *obs_shape)``.

    Without a key the most probable action is returned (the mean for
    continuous actions); with a key the action is sampled.
    """
    out, _ = _outputs(policy, env, obs, normalize)
    space = env.action_space
    if key is not None:
        return _to_env(env, _sample(policy, env, out, key))
    if isinstance(space, Discrete):
        return jnp.argmax(out, axis=-1)
    return _to_env(env, out.reshape(*out.shape[:-1], *space.shape))


def _where(mask, new, old):
    """Select between two pytrees with a mask over the leading axis."""

    def select(a, b):
        return jnp.where(mask.reshape(mask.shape + (1,) * (a.ndim - mask.ndim)), a, b)

    return jax.tree.map(select, new, old)


class TrainState(NamedTuple):
    """What a training run carries from one update to the next. A pytree.

    Attributes:
        policy: The current policy.
        moments: First and second moments of the gradient (Adam).
        gradient_steps: Number of gradient steps taken.
        runner: The environments and the statistics of the running episodes.
        key: JAX random key for the next update.
        iteration: Number of updates done.
    """

    policy: Policy
    moments: tuple
    gradient_steps: jax.Array
    runner: tuple
    key: jax.Array
    iteration: jax.Array


def gae(value, reward, next_value, terminated, ended, gamma: float, gae_lambda: float):
    """Generalized advantage estimates along the leading (time) axis.

    Args:
        value: Values of the observations, shape (n_steps, ...).
        reward: Rewards of the steps.
        next_value: Values of the observations after the steps, before any reset.
        terminated: True where the step ended the episode for good; the value
            of what follows is then zero.
        ended: True where the step ended the episode (terminated or
            truncated); the estimate does not reach across such a step.
        gamma: Discount factor.
        gae_lambda: Decay of the estimator.

    Returns:
        ``(advantage, target)``; the target of the value is their sum
        ``advantage + value``.
    """

    def body(carry, x):
        value, reward, next_value, terminated, ended = x
        delta = reward + gamma * next_value * ~terminated - value
        carry = delta + gamma * gae_lambda * ~ended * carry
        return carry, carry

    xs = (value, reward, next_value, terminated, ended)
    _, advantage = jax.lax.scan(body, jnp.zeros_like(value[0]), xs, reverse=True)
    return advantage, advantage + value


def init(key: jax.Array, env, params=None, config: PPOConfig | None = None) -> TrainState:
    """The start of a training run: a random policy and freshly reset environments."""
    config = PPOConfig() if config is None else config
    params = env.default_params if params is None else params
    n_envs, n_agents = config.n_envs, env.n_agents
    key, policy_key, reset_key = jax.random.split(key, 3)
    policy = init_policy(policy_key, env, config)
    reset = jax.vmap(env.reset, in_axes=(0, None))
    obs, state = reset(jax.random.split(reset_key, n_envs), params)
    runner = (
        state,
        obs,
        jnp.zeros((n_envs, n_agents), bool),  # agents whose episode has ended
        jnp.zeros((n_envs, n_agents), jnp.float32),  # return so far
        jnp.zeros(n_envs, jnp.int32),  # length so far
    )
    zeros = jax.tree.map(jnp.zeros_like, (policy.actor, policy.critic, policy.log_std))
    count = jnp.zeros((), jnp.int32)
    return TrainState(policy, (zeros, zeros), count, runner, key, count)


def run(
    state: TrainState,
    env,
    params=None,
    config: PPOConfig | None = None,
    n_iterations: int | None = None,
):
    """Continue a training run by ``n_iterations`` updates (default: all of them).

    Returns:
        ``(state, curves)``; see :func:`train` for the curves.
    """
    config = PPOConfig() if config is None else config
    params = env.default_params if params is None else params
    n_iterations = config.n_iterations if n_iterations is None else n_iterations
    n_envs, n_steps, n_agents = config.n_envs, config.n_steps, env.n_agents
    reset = jax.vmap(env.reset, in_axes=(0, None))
    step = jax.vmap(env.step, in_axes=(0, 0, 0, None))

    def collect(policy, runner, key):
        def body(runner, key):
            state, obs, finished, returns, length = runner
            action_key, step_key, reset_key = jax.random.split(key, 3)
            out, value = _outputs(policy, env, obs, config.normalize)
            action = _sample(policy, env, out, action_key)
            log_p, _ = _log_prob_entropy(policy, env, out, action)
            next_obs, next_state, reward, terminated, truncated, _ = step(
                jax.random.split(step_key, n_envs), state, _to_env(env, action), params
            )
            reward = reward.astype(jnp.float32)
            _, next_value = _outputs(policy, env, next_obs, config.normalize)
            active = ~finished
            ended = terminated | truncated
            returns = returns + jnp.where(active, reward, 0.0)
            length = length + 1
            finished = finished | ended
            over = jnp.all(finished, axis=1)
            sample = (obs, action, log_p, value, reward, next_value, terminated, ended, active)
            episode = (over, jnp.mean(returns, axis=1), length)

            fresh_obs, fresh_state = reset(jax.random.split(reset_key, n_envs), params)
            state = _where(over, fresh_state, next_state)
            obs = _where(over, fresh_obs, next_obs)
            finished = jnp.where(over[:, None], False, finished)
            returns = jnp.where(over[:, None], 0.0, returns)
            length = jnp.where(over, 0, length)
            return (state, obs, finished, returns, length), (sample, episode)

        return jax.lax.scan(body, runner, jax.random.split(key, n_steps))

    def loss(trained, stats, batch):
        policy = Policy(*trained, *stats)
        obs, action, old_log_p, advantage, target, weight = batch
        out, value = _outputs(policy, env, obs, config.normalize)
        log_p, entropy = _log_prob_entropy(policy, env, out, action)
        total = jnp.maximum(jnp.sum(weight), 1.0)
        mean = jnp.sum(weight * advantage) / total
        std = jnp.sqrt(jnp.sum(weight * (advantage - mean) ** 2) / total)
        advantage = (advantage - mean) / (std + 1e-8)
        ratio = jnp.exp(log_p - old_log_p)
        clipped = jnp.clip(ratio, 1.0 - config.clip, 1.0 + config.clip)
        surrogate = jnp.minimum(ratio * advantage, clipped * advantage)
        terms = -surrogate + config.value * 0.5 * (value - target) ** 2 - config.entropy * entropy
        return jnp.sum(weight * terms) / total

    def adam(trained, moments, grads, count, rate):
        norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree.leaves(grads)))
        grads = jax.tree.map(lambda g: g * jnp.minimum(1.0, config.max_grad_norm / norm), grads)
        first, second = moments
        first = jax.tree.map(lambda m, g: 0.9 * m + 0.1 * g, first, grads)
        second = jax.tree.map(lambda v, g: 0.999 * v + 0.001 * g**2, second, grads)
        t = count.astype(jnp.float32)
        scale = rate * jnp.sqrt(1.0 - 0.999**t) / (1.0 - 0.9**t)
        trained = jax.tree.map(
            lambda p, m, v: p - scale * m / (jnp.sqrt(v) + 1e-5), trained, first, second
        )
        return trained, (first, second)

    def iteration(train_state, _):
        policy, moments, count, runner, key, index = train_state
        key, collect_key, update_key = jax.random.split(key, 3)
        runner, (sample, episode) = collect(policy, runner, collect_key)
        obs, action, log_p, value, reward, next_value, terminated, ended, active = sample
        advantage, target = gae(
            value,
            config.reward_scale * reward,
            next_value,
            terminated,
            ended,
            config.gamma,
            config.gae_lambda,
        )

        n = n_steps * n_envs * n_agents
        flat = jax.tree.map(
            lambda a: a.reshape(n, *a.shape[3:]),
            (obs, action, log_p, advantage, target, active.astype(jnp.float32)),
        )
        rate = config.learning_rate
        if config.anneal:
            done = index.astype(jnp.float32) / config.n_iterations
            rate = rate * jnp.maximum(1.0 - done, 0.0)
        stats = (policy.square, policy.count)

        def epoch(carry, key):
            def minibatch(carry, rows):
                trained, moments, count = carry
                batch = jax.tree.map(lambda a: a[rows], flat)
                grads = jax.grad(loss)(trained, stats, batch)
                count = count + 1
                trained, moments = adam(trained, moments, grads, count, rate)
                return (trained, moments, count), None

            order = jax.random.permutation(key, n)
            size = n // config.n_minibatches
            rows = order[: size * config.n_minibatches].reshape(config.n_minibatches, size)
            return jax.lax.scan(minibatch, carry, rows)

        trained = (policy.actor, policy.critic, policy.log_std)
        (trained, moments, count), _ = jax.lax.scan(
            epoch, (trained, moments, count), jax.random.split(update_key, config.n_epochs)
        )

        # Running mean of the squared observations, for the next roll-out.
        x = features(env, obs).reshape(n, -1)
        weight = active.reshape(n, 1).astype(jnp.float32)
        total = policy.count + jnp.sum(weight)
        square = (policy.square * policy.count + jnp.sum(weight * x**2, axis=0)) / total
        policy = Policy(*trained, square, total)

        over, returns, length = episode
        episodes = jnp.sum(over)
        curve = {
            "episode_return": jnp.sum(jnp.where(over, returns, 0.0)) / episodes,
            "episode_length": jnp.sum(jnp.where(over, length, 0)) / episodes,
            "episodes": episodes,
            "reward": jnp.sum(jnp.where(active, reward, 0.0)) / jnp.maximum(jnp.sum(active), 1),
        }
        return TrainState(policy, moments, count, runner, key, index + 1), curve

    return jax.lax.scan(iteration, state, None, length=n_iterations)


def train(key: jax.Array, env, params=None, config: PPOConfig | None = None):
    """Train a policy.

    Args:
        key: JAX random key; the run is a pure function of it.
        env: An environment of ParticleGym.
        params: Episode parameters; the defaults of the environment if None.
        config: Settings; the defaults if None.

    Returns:
        ``(policy, curves)``. ``curves`` holds one entry per update:
        ``episode_return`` (return per agent, averaged over the agents and
        over the episodes that ended since the previous update; NaN if none
        ended), ``episode_length``, ``episodes`` (their number) and
        ``reward`` (mean reward per agent and step).
    """
    state, curves = run(init(key, env, params, config), env, params, config)
    return state.policy, curves


def save_policy(path, policy: Policy) -> None:
    """Write a policy, or a batch of policies, to a ``.npz`` file."""
    leaves = jax.tree.leaves(policy)
    np.savez(path, **{f"leaf_{i:02d}": np.asarray(leaf) for i, leaf in enumerate(leaves)})


def load_policy(path, env, config: PPOConfig | None = None) -> Policy:
    """Read a policy written by :func:`save_policy` for the same environment and settings."""
    config = PPOConfig() if config is None else config
    template = init_policy(jax.random.key(0), env, config)
    with np.load(path) as data:
        leaves = [jnp.asarray(data[name]) for name in sorted(data.files)]
    return jax.tree.unflatten(jax.tree.structure(template), leaves)


def evaluate(key: jax.Array, env, params, policy_fn, n_episodes: int, max_steps: int):
    """Play ``n_episodes`` episodes with a policy and return their statistics.

    Args:
        key: JAX random key.
        env: An environment of ParticleGym.
        params: Episode parameters.
        policy_fn: ``policy_fn(key, obs, state) -> actions`` for one environment.
        n_episodes: Number of episodes, played in parallel.
        max_steps: Largest number of control steps; an episode that is still
            running then is counted with the return so far.

    Returns:
        A dictionary with one entry per episode: ``episode_return`` (return
        per agent, averaged over the agents), ``episode_length`` and
        ``finished``.
    """
    params = env.default_params if params is None else params

    def episode(key):
        reset_key, run_key = jax.random.split(key)
        obs, state = env.reset(reset_key, params)

        def body(carry, key):
            state, obs, finished, returns, length = carry
            action_key, step_key = jax.random.split(key)
            action = policy_fn(action_key, obs, state)
            new_obs, new_state, reward, terminated, truncated, _ = env.step(
                step_key, state, action, params
            )
            over = jnp.all(finished)
            returns = returns + jnp.where(finished, 0.0, reward.astype(jnp.float32))
            length = length + ~over
            finished = finished | terminated | truncated
            state = jax.tree.map(lambda new, old: jnp.where(over, old, new), new_state, state)
            obs = jnp.where(over, obs, new_obs)
            return (state, obs, finished, returns, length), None

        carry = (
            state,
            obs,
            jnp.zeros(env.n_agents, bool),
            jnp.zeros(env.n_agents, jnp.float32),
            jnp.zeros((), jnp.int32),
        )
        (_, _, finished, returns, length), _ = jax.lax.scan(
            body, carry, jax.random.split(run_key, max_steps)
        )
        return {
            "episode_return": jnp.mean(returns),
            "episode_length": length,
            "finished": jnp.all(finished),
        }

    return jax.vmap(episode)(jax.random.split(key, n_episodes))
