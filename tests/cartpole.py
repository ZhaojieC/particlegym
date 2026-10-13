"""The cart-pole balancing task, as a check for learning algorithms.

The equations of motion, the parameters and the limits are those of
``CartPole-v1`` of Gymnasium (after Barto, Sutton and Anderson, 1983): a
reward of 1 per step until the pole falls or the cart leaves the track,
and at most 500 steps. It is not a particle environment and therefore not
part of the package.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from particlegym.envs.spaces import Continuous, Discrete


class CartPoleParams(NamedTuple):
    max_steps: int = 500


class CartPoleState(NamedTuple):
    x: jax.Array  # position and velocity of the cart, angle and angular velocity of the pole
    step: jax.Array


class CartPole:
    n_agents = 1
    observation_space = Continuous(-jnp.inf, jnp.inf, (4,))
    action_space = Discrete(2)

    @property
    def default_params(self):
        return CartPoleParams()

    def reset(self, key, params=None):
        x = jax.random.uniform(key, (4,), minval=-0.05, maxval=0.05)
        return x[None], CartPoleState(x, jnp.zeros((), jnp.int32))

    def step(self, key, state, action, params=None):
        params = self.default_params if params is None else params
        gravity, cart, pole, half_length, push, dt = 9.8, 1.0, 0.1, 0.5, 10.0, 0.02
        pos, vel, angle, spin = state.x
        force = jnp.where(jnp.reshape(action, ()) == 1, push, -push)
        cos, sin = jnp.cos(angle), jnp.sin(angle)
        temp = (force + pole * half_length * spin**2 * sin) / (cart + pole)
        spin_rate = (gravity * sin - cos * temp) / (
            half_length * (4.0 / 3.0 - pole * cos**2 / (cart + pole))
        )
        acceleration = temp - pole * half_length * spin_rate * cos / (cart + pole)
        x = jnp.stack(
            [pos + dt * vel, vel + dt * acceleration, angle + dt * spin, spin + dt * spin_rate]
        )
        step = state.step + 1
        terminated = (jnp.abs(x[0]) > 2.4) | (jnp.abs(x[2]) > 12.0 * jnp.pi / 180.0)
        truncated = ~terminated & (step >= params.max_steps)
        return (
            x[None],
            CartPoleState(x, step),
            jnp.ones(1, x.dtype),
            terminated[None],
            truncated[None],
            {},
        )
