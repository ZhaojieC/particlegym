"""The tasks of the learning examples: environment, settings and baselines."""

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp

from particlegym import envs
from particlegym import physics as pp
from particlegym.baselines import ppo


class Task(NamedTuple):
    """A task for :mod:`train`.

    Attributes:
        env: The environment.
        params: Its episode parameters.
        config: Settings of PPO.
        baselines: Policies to compare with, ``name -> policy(key, obs, state)``.
        label: Axis label of the return in the figures.
        display: Environment for the animations, if it differs from ``env``
            (the policy is shared, so it also steers a larger group).
        view: Further arguments of ``render`` for the animations.
    """

    env: object
    params: object
    config: ppo.PPOConfig
    baselines: dict[str, Callable]
    label: str
    display: object = None
    view: dict | None = None


def random_policy(env):
    """Actions drawn uniformly from the action space."""
    return lambda key, obs, state: env.action_space.sample(key, (env.n_agents,))


def microswimmer() -> Task:
    """Swim upwards through the vortices; the naive swimmer always steers up."""
    env = envs.Microswimmer(n_agents=16)
    config = ppo.PPOConfig(
        n_envs=256, n_steps=64, n_iterations=300, n_minibatches=16, reward_scale=10.0
    )
    baselines = {
        "naive (always up)": lambda key, obs, state: env.naive_policy(),
        "random": random_policy(env),
    }
    label = "altitude gained in 100 time units"
    # The animation shows 4 x 4 periods of the flow and the altitude of the swimmers.
    view = {"periods": (4, 4), "start_row": 1}
    return Task(env, env.default_params, config, baselines, label, envs.Microswimmer(400), view)


def evacuation() -> Task:
    """Leave the four-door room of Zhang, Chai and Lykotrafitis (2021)."""
    doors = [pp.Opening(face, (4.5, 5.5)) for face in ("y+", "y-", "x-", "x+")]
    env = envs.Evacuation(room=(10.0, 10.0), exits=doors, n_agents=40)
    params = env.default_params._replace(max_steps=300)
    config = ppo.PPOConfig(n_envs=128, n_steps=64, n_iterations=300, n_minibatches=16)
    directions = jnp.asarray(env.directions)
    centres = jnp.asarray(env.door_centres)
    targets = jnp.asarray(env.door_targets)

    def nearest_door(key, obs, state):
        pos = state.particles.pos[: env.n_agents]
        nearest = jnp.argmin(jnp.sum((pos[:, None, :] - centres) ** 2, axis=-1), axis=1)
        return jnp.argmax((targets[nearest] - pos) @ directions.T, axis=1)

    baselines = {"walk to the nearest door": nearest_door, "random": random_policy(env)}
    return Task(env, params, config, baselines, "return per pedestrian")


def herding() -> Task:
    """Drive a flock of 40 sheep to the target with one shepherd."""
    env = envs.Herding()
    params = env.default_params._replace(max_steps=1000)
    config = ppo.PPOConfig(
        n_envs=512,
        n_steps=128,
        n_iterations=600,
        n_minibatches=16,
        gamma=0.995,
        reward_scale=0.1,
    )
    baselines = {
        "scripted shepherd": lambda key, obs, state: env.shepherd_policy(key, state, params),
        "random": random_policy(env),
    }
    return Task(env, params, config, baselines, "return")


def brick_wall() -> Task:
    """Knock the 40 bricks out of the wall with three balls."""
    env = envs.BrickWall()
    params = env.default_params._replace(max_steps=3000)
    config = ppo.PPOConfig(
        n_envs=1024, n_steps=128, n_iterations=400, n_minibatches=16, gamma=0.995
    )
    baselines = {
        "scripted player": lambda key, obs, state: env.scripted_policy(state, params),
        "random": random_policy(env),
    }
    return Task(env, params, config, baselines, "bricks removed")


TASKS = {
    "microswimmer": microswimmer,
    "evacuation": evacuation,
    "herding": herding,
    "brick_wall": brick_wall,
}


def greedy(task: Task, policy: ppo.Policy):
    """The trained policy without exploration noise, as a baseline-style function."""
    return lambda key, obs, state: ppo.act(policy, task.env, obs, normalize=task.config.normalize)


def sampled(task: Task, policy: ppo.Policy):
    """The trained policy with its exploration noise."""
    return lambda key, obs, state: ppo.act(
        policy, task.env, obs, key, normalize=task.config.normalize
    )


def evaluate(task: Task, policy_fn, key: jax.Array, n_episodes: int):
    """Statistics of ``n_episodes`` episodes; see :func:`particlegym.baselines.ppo.evaluate`."""
    return ppo.evaluate(key, task.env, task.params, policy_fn, n_episodes, task.params.max_steps)
