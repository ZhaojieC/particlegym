"""Gymnasium wrappers: a single environment and a vector environment."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from particlegym.envs.spaces import Continuous, Discrete

try:
    import gymnasium
    from gymnasium.vector.utils import batch_space
except ImportError as error:  # pragma: no cover
    raise ImportError(
        'the Gymnasium wrappers need Gymnasium: pip install "particlegym[gymnasium]"'
    ) from error


def to_gymnasium_space(space: Discrete | Continuous, dtype=np.float32) -> gymnasium.Space:
    """The Gymnasium space for a space of ParticleGym."""
    if isinstance(space, Discrete):
        return gymnasium.spaces.Discrete(space.n)
    low = space.low if math.isfinite(space.low) else -np.inf
    high = space.high if math.isfinite(space.high) else np.inf
    return gymnasium.spaces.Box(low, high, space.shape, dtype)


def _to_numpy(space: Discrete | Continuous, x) -> np.ndarray:
    return np.asarray(x, np.int64 if isinstance(space, Discrete) else np.float32)


def _info(info: dict) -> dict:
    return {name: np.asarray(value) for name, value in info.items()}


class GymnasiumEnv(gymnasium.Env):
    """A single-agent environment of ParticleGym as a ``gymnasium.Env``.

    Args:
        env: An environment with ``n_agents == 1``.
        params: Episode parameters; the defaults of the environment if None.
        render_mode: None or ``"rgb_array"``.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}  # noqa: RUF012

    def __init__(self, env, params=None, render_mode: str | None = None) -> None:
        if env.n_agents != 1:
            raise ValueError("GymnasiumEnv needs one agent; use PettingZooEnv for several")
        if render_mode not in (None, "rgb_array"):
            raise ValueError("render_mode must be None or 'rgb_array'")
        self.env = env
        self.params = env.default_params if params is None else params
        self.render_mode = render_mode
        self.observation_space = to_gymnasium_space(env.observation_space)
        self.action_space = to_gymnasium_space(env.action_space)
        self._reset = jax.jit(env.reset)
        self._step = jax.jit(env.step)
        self._key = None
        self.env_state = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None or self._key is None:
            self._key = jax.random.key(int(self.np_random.integers(2**31 - 1)))
        self._key, key = jax.random.split(self._key)
        obs, self.env_state = self._reset(key, self.params)
        return _to_numpy(self.env.observation_space, obs[0]), {}

    def step(self, action):
        if self.env_state is None:
            raise RuntimeError("call reset before step")
        self._key, key = jax.random.split(self._key)
        action = jnp.asarray(action)[None]
        obs, self.env_state, reward, terminated, truncated, info = self._step(
            key, self.env_state, action, self.params
        )
        return (
            _to_numpy(self.env.observation_space, obs[0]),
            float(reward[0]),
            bool(terminated[0]),
            bool(truncated[0]),
            _info(info),
        )

    def render(self):
        if self.render_mode is None or self.env_state is None:
            return None
        return np.asarray(self.env.render(self.env_state))


class GymnasiumVectorEnv(gymnasium.vector.VectorEnv):
    """``num_envs`` copies of a single-agent environment, batched with ``jax.vmap``.

    A finished episode is reset in the same step, and the observation
    returned is the first one of the new episode
    (``AutoresetMode.SAME_STEP``); the last observation of the finished
    episode is in ``info["final_obs"]``.

    Args:
        env: An environment with ``n_agents == 1``.
        num_envs: Number of copies.
        params: Episode parameters, shared by all copies.
    """

    def __init__(self, env, num_envs: int, params=None) -> None:
        if env.n_agents != 1:
            raise ValueError("GymnasiumVectorEnv needs one agent per environment")
        self.env = env
        self.num_envs = int(num_envs)
        self.params = env.default_params if params is None else params
        self.single_observation_space = to_gymnasium_space(env.observation_space)
        self.single_action_space = to_gymnasium_space(env.action_space)
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)
        self.action_space = batch_space(self.single_action_space, self.num_envs)
        self.metadata = {"autoreset_mode": gymnasium.vector.AutoresetMode.SAME_STEP}
        self._reset = jax.jit(jax.vmap(env.reset, in_axes=(0, None)))

        def step(key, state, action, params):
            step_key, reset_key = jax.random.split(key)
            obs, state, reward, terminated, truncated, info = env.step(
                step_key, state, action[None], params
            )
            done = terminated[0] | truncated[0]
            new_obs, new_state = env.reset(reset_key, params)
            state = jax.tree.map(lambda new, old: jnp.where(done, new, old), new_state, state)
            out = jnp.where(done, new_obs[0], obs[0])
            return out, state, reward[0], terminated[0], truncated[0], info, obs[0]

        self._step = jax.jit(jax.vmap(step, in_axes=(0, 0, 0, None)))
        self._key = None
        self.env_state = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None or self._key is None:
            self._key = jax.random.key(int(self.np_random.integers(2**31 - 1)))
        self._key, key = jax.random.split(self._key)
        obs, self.env_state = self._reset(jax.random.split(key, self.num_envs), self.params)
        return _to_numpy(self.env.observation_space, obs[:, 0]), {}

    def step(self, actions):
        if self.env_state is None:
            raise RuntimeError("call reset before step")
        self._key, key = jax.random.split(self._key)
        obs, self.env_state, reward, terminated, truncated, info, final = self._step(
            jax.random.split(key, self.num_envs), self.env_state, jnp.asarray(actions), self.params
        )
        info = _info(info)
        info["final_obs"] = _to_numpy(self.env.observation_space, final)
        return (
            _to_numpy(self.env.observation_space, obs),
            np.asarray(reward, np.float64),
            np.asarray(terminated, bool),
            np.asarray(truncated, bool),
            info,
        )


_REGISTERED = {
    "particlegym/Evacuation2D-v0": "_evacuation_2d",
    "particlegym/BrickWall-v0": "_brick_wall",
    "particlegym/Billiards-v0": "_billiards",
    "particlegym/Herding-v0": "_herding",
    "particlegym/Microswimmer-v0": "_microswimmer",
}


def _evacuation_2d(render_mode=None, **kwargs):
    from particlegym import envs
    from particlegym import physics as pp

    kwargs.setdefault("room", (10.0, 10.0))
    kwargs.setdefault("exits", [pp.Opening("x+", (4.4, 5.6))])
    return GymnasiumEnv(envs.Evacuation(n_agents=1, **kwargs), render_mode=render_mode)


def _brick_wall(render_mode=None, **kwargs):
    from particlegym import envs

    return GymnasiumEnv(envs.BrickWall(**kwargs), render_mode=render_mode)


def _billiards(render_mode=None, **kwargs):
    from particlegym import envs

    return GymnasiumEnv(envs.Billiards(**kwargs), render_mode=render_mode)


def _herding(render_mode=None, **kwargs):
    from particlegym import envs

    return GymnasiumEnv(envs.Herding(n_agents=1, **kwargs), render_mode=render_mode)


def _microswimmer(render_mode=None, **kwargs):
    from particlegym import envs

    return GymnasiumEnv(envs.Microswimmer(n_agents=1, **kwargs), render_mode=render_mode)


def register_envs() -> None:
    """Register the single-agent environments with Gymnasium.

    Called when :mod:`particlegym.wrappers` is imported. Afterwards, for
    example, ``gymnasium.make("particlegym/BrickWall-v0")`` works.
    """
    for name, entry in _REGISTERED.items():
        if name not in gymnasium.registry:
            gymnasium.register(
                id=name,
                entry_point=f"particlegym.wrappers.gymnasium_env:{entry}",
                disable_env_checker=True,
            )


register_envs()
