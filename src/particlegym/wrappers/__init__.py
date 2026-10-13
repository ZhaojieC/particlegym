"""Wrappers that expose the environments through Gymnasium and PettingZoo.

The libraries are optional: install them with ``pip install
"particlegym[gymnasium]"`` or ``"particlegym[pettingzoo]"``. The wrappers
convert between JAX and NumPy arrays and keep the state of the episode, so
they are meant for interoperability; training code in JAX is faster with the
functional interface of the environments.
"""

from particlegym.wrappers.gymnasium_env import (
    GymnasiumEnv,
    GymnasiumVectorEnv,
    register_envs,
    to_gymnasium_space,
)
from particlegym.wrappers.pettingzoo_env import PettingZooEnv

__all__ = [
    "GymnasiumEnv",
    "GymnasiumVectorEnv",
    "PettingZooEnv",
    "register_envs",
    "to_gymnasium_space",
]
