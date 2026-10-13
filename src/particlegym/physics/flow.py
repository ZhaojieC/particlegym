"""Prescribed flow fields of a surrounding fluid.

A flow field gives the fluid velocity at any point. Particles are coupled to
it through the ``drift`` argument of
:func:`~particlegym.physics.langevin_baoab`: with ``drift = flow.velocity(pos)``
the friction term of the integrator is the Stokes drag towards the local
fluid velocity. The particles do not act back on the flow.
"""

from __future__ import annotations

import abc
import dataclasses

import jax
import jax.numpy as jnp


class FlowField(abc.ABC):
    """A steady velocity field."""

    @abc.abstractmethod
    def velocity(self, pos: jax.Array) -> jax.Array:
        """(N, d) fluid velocities at the points ``pos`` (N, d)."""

    def vorticity(self, pos: jax.Array) -> jax.Array:
        """(N,) vorticity ``du_y/dx - du_x/dy`` of a 2D flow at the points ``pos``.

        A small body carried by the flow rotates with half this rate.
        """
        if pos.shape[-1] != 2:
            raise ValueError("the scalar vorticity is defined for 2D flows")

        def one(point):
            jacobian = jax.jacfwd(lambda p: self.velocity(p[None])[0])(point)
            return jacobian[1, 0] - jacobian[0, 1]

        return jax.vmap(one)(pos)


@dataclasses.dataclass(frozen=True)
class UniformFlow(FlowField):
    """The same velocity everywhere.

    Args:
        u: Velocity vector, one value per axis.
    """

    u: tuple[float, ...] = (1.0, 0.0)

    def velocity(self, pos):
        return jnp.broadcast_to(jnp.asarray(self.u, dtype=pos.dtype), pos.shape)


@dataclasses.dataclass(frozen=True)
class ShearFlow(FlowField):
    """Simple shear: the velocity along one axis grows linearly along another.

    ``u[flow_axis] = rate * (x[gradient_axis] - origin)``; the other
    components are zero.

    Args:
        rate: Shear rate.
        flow_axis: Axis of the velocity.
        gradient_axis: Axis along which the velocity changes.
        origin: Coordinate along the gradient axis where the velocity is zero.
    """

    rate: float = 1.0
    flow_axis: int = 0
    gradient_axis: int = 1
    origin: float = 0.0

    def velocity(self, pos):
        speed = self.rate * (pos[..., self.gradient_axis] - self.origin)
        return jnp.zeros_like(pos).at[..., self.flow_axis].set(speed)


@dataclasses.dataclass(frozen=True)
class TaylorGreen(FlowField):
    """Steady 2D array of counter-rotating vortices (Taylor-Green flow).

    ``u = (u0 / 2) (-cos(kx) sin(ky), sin(kx) cos(ky))``, the form used by
    Colabrese, Gustavsson, Celani & Biferale (2017) for microswimmers. The
    flow is periodic with period ``2 pi / k`` along both axes, has the stream
    function ``psi = u0 / (2k) cos(kx) cos(ky)`` and the vorticity
    ``u0 k cos(kx) cos(ky)``. The largest speed is ``u0 / 2``.

    Args:
        u0: Velocity scale.
        k: Wave number.
    """

    u0: float = 1.0
    k: float = 1.0

    def velocity(self, pos):
        x, y = self.k * pos[..., 0], self.k * pos[..., 1]
        u = jnp.stack([-jnp.cos(x) * jnp.sin(y), jnp.sin(x) * jnp.cos(y)], axis=-1)
        return 0.5 * self.u0 * u

    def vorticity(self, pos):
        return self.u0 * self.k * jnp.cos(self.k * pos[..., 0]) * jnp.cos(self.k * pos[..., 1])

    def stream_function(self, pos: jax.Array) -> jax.Array:
        """(N,) stream function; it is constant along the path of a fluid element."""
        x, y = self.k * pos[..., 0], self.k * pos[..., 1]
        return 0.5 * self.u0 / self.k * jnp.cos(x) * jnp.cos(y)
