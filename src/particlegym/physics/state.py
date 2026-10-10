"""Particle state."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class ParticleState(NamedTuple):
    """State of N particles in d dimensions. It is a JAX pytree.

    Attributes:
        pos: (N, d) positions.
        vel: (N, d) velocities.
        force: (N, d) total force at ``pos``. Integrators keep it up to date.
        mass: (N,) masses.
        radius: (N,) radii. Two particles touch when their distance equals the
            sum of their radii.
        alive: (N,) booleans. Dead particles (for example pedestrians that have
            left through an exit) are frozen and do not interact. Using a mask
            instead of removing particles keeps array shapes static.
        kind: (N,) integer species labels, used by environments and for output.
    """

    pos: jax.Array
    vel: jax.Array
    force: jax.Array
    mass: jax.Array
    radius: jax.Array
    alive: jax.Array
    kind: jax.Array

    @property
    def n(self) -> int:
        """Number of particles, including dead ones."""
        return self.pos.shape[0]

    @property
    def dim(self) -> int:
        """Number of spatial dimensions."""
        return self.pos.shape[1]


def create_state(pos, vel=None, *, mass=1.0, radius=0.5, alive=True, kind=0) -> ParticleState:
    """Create a :class:`ParticleState`, broadcasting scalar properties.

    The floating-point type of ``pos`` sets the type of the whole state. Forces
    start at zero; call :func:`particlegym.physics.init_forces` before the
    first integration step.

    Args:
        pos: (N, d) initial positions with d = 2 or 3.
        vel: (N, d) initial velocities; zero if omitted.
        mass: Particle masses, scalar or (N,).
        radius: Particle radii, scalar or (N,).
        alive: Alive flags, scalar or (N,).
        kind: Species labels, scalar or (N,).
    """
    pos = jnp.asarray(pos)
    if not jnp.issubdtype(pos.dtype, jnp.floating):
        pos = pos.astype(jax.dtypes.canonicalize_dtype(np.float64))
    if pos.ndim != 2 or pos.shape[1] not in (2, 3):
        raise ValueError("pos must have shape (N, 2) or (N, 3)")
    n, d = pos.shape
    dtype = pos.dtype
    vel = jnp.zeros_like(pos) if vel is None else jnp.broadcast_to(jnp.asarray(vel, dtype), (n, d))
    return ParticleState(
        pos=pos,
        vel=vel,
        force=jnp.zeros_like(pos),
        mass=jnp.broadcast_to(jnp.asarray(mass, dtype), (n,)),
        radius=jnp.broadcast_to(jnp.asarray(radius, dtype), (n,)),
        alive=jnp.broadcast_to(jnp.asarray(alive, bool), (n,)),
        kind=jnp.broadcast_to(jnp.asarray(kind, jnp.int32), (n,)),
    )
