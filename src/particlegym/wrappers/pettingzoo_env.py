"""PettingZoo wrapper: a multi-agent environment as a ``ParallelEnv``."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

from particlegym.envs.spaces import Discrete

try:
    import pettingzoo

    from particlegym.wrappers.gymnasium_env import _info, _to_numpy, to_gymnasium_space
except ImportError as error:  # pragma: no cover
    raise ImportError(
        'the PettingZoo wrapper needs PettingZoo: pip install "particlegym[pettingzoo]"'
    ) from error


class PettingZooEnv(pettingzoo.ParallelEnv):
    """An environment of ParticleGym as a PettingZoo ``ParallelEnv``.

    The agents are named ``agent_0``, ``agent_1``, ... in the order of the
    agent axis of the environment. An agent whose episode has ended (for
    example a pedestrian that has left the room) is removed from ``agents``;
    the environment itself keeps its fixed size, and the missing action is
    replaced by zeros.

    Args:
        env: An environment of ParticleGym.
        params: Episode parameters; the defaults of the environment if None.
        render_mode: None or ``"rgb_array"``.
    """

    metadata = {"render_modes": ["rgb_array"], "name": "particlegym_v0"}  # noqa: RUF012

    def __init__(self, env, params=None, render_mode: str | None = None) -> None:
        if render_mode not in (None, "rgb_array"):
            raise ValueError("render_mode must be None or 'rgb_array'")
        self.env = env
        self.params = env.default_params if params is None else params
        self.render_mode = render_mode
        self.possible_agents = [f"agent_{i}" for i in range(env.n_agents)]
        self.agents = []
        self._index = {name: i for i, name in enumerate(self.possible_agents)}
        self._reset = jax.jit(env.reset)
        self._step = jax.jit(env.step)
        self._key = None
        self.env_state = None

    @functools.cache  # noqa: B019
    def observation_space(self, agent):
        return to_gymnasium_space(self.env.observation_space)

    @functools.cache  # noqa: B019
    def action_space(self, agent):
        return to_gymnasium_space(self.env.action_space)

    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None or self._key is None:
            self._key = jax.random.key(0 if seed is None else seed)
        self._key, key = jax.random.split(self._key)
        obs, self.env_state = self._reset(key, self.params)
        self.agents = list(self.possible_agents)
        obs = _to_numpy(self.env.observation_space, obs)
        return {name: obs[i] for name, i in self._index.items()}, {
            name: {} for name in self.agents
        }

    def step(self, actions: dict):
        if self.env_state is None:
            raise RuntimeError("call reset before step")
        space = self.env.action_space
        dtype = np.int32 if isinstance(space, Discrete) else np.float32
        joint = np.zeros((self.env.n_agents, *space.shape), dtype)
        for name, action in actions.items():
            joint[self._index[name]] = action
        self._key, key = jax.random.split(self._key)
        obs, self.env_state, reward, terminated, truncated, info = self._step(
            key, self.env_state, jnp.asarray(joint), self.params
        )
        obs = _to_numpy(self.env.observation_space, obs)
        reward, terminated, truncated = (np.asarray(x) for x in (reward, terminated, truncated))
        info = _info(info)
        live = list(self.agents)
        self.agents = [
            name
            for name in live
            if not (terminated[self._index[name]] or truncated[self._index[name]])
        ]
        return (
            {name: obs[self._index[name]] for name in live},
            {name: float(reward[self._index[name]]) for name in live},
            {name: bool(terminated[self._index[name]]) for name in live},
            {name: bool(truncated[self._index[name]]) for name in live},
            {name: dict(info) for name in live},
        )

    def render(self):
        if self.render_mode is None or self.env_state is None:
            return None
        return np.asarray(self.env.render(self.env_state))
