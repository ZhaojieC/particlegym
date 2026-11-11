# ParticleGym, https://github.com/ZhaojieC/particlegym (MIT license).
# Please cite it in published work: particlegym.cite() gives the references.
"""Cells that grow, divide and die: helpers for agent-based models of tissues.

In an agent-based (centre-based) model a cell is a soft particle; it grows,
divides into two daughters and dies (Drasdo & Hoehme 2005). The population
lives in the slots of a :class:`~particlegym.physics.ParticleState` of fixed
size, so that arrays keep their shapes under :func:`jax.jit`: a slot whose
``alive`` flag is false holds no cell, and a daughter is born into such a
slot.

- :func:`birth_slots` pairs the cells that are ready to divide with free
  slots.
- :func:`divide` places the two daughters of every dividing cell.
- :func:`overlap_sum` measures how strongly a cell is compressed by its
  neighbours, for contact inhibition of growth.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from particlegym.physics.neighbors import BruteForce, CellList
from particlegym.physics.space import Box


def birth_slots(alive: jax.Array, ready: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Free slots for the daughters of the cells that are ready to divide.

    The ready cells divide in the order of their indices while free slots
    last; the others wait.

    Args:
        alive: (N,) slots that hold a cell.
        ready: (N,) cells that are ready to divide.

    Returns:
        ``(divides, mother)``: (N,) the cells that divide now, and (N,) for
        every slot the index of the cell whose daughter it receives, or -1.
    """
    n = alive.shape[0]
    ready = ready & alive
    rank = jnp.cumsum(ready) - 1
    free = jnp.nonzero(~alive, size=n, fill_value=n)[0]
    divides = ready & (rank < jnp.sum(~alive))
    target = jnp.where(divides, free[jnp.clip(rank, 0, n - 1)], n)
    mother = jnp.full(n + 1, -1, dtype=jnp.int32).at[target].set(jnp.arange(n, dtype=jnp.int32))
    return divides, mother[:n]


def divide(
    pos: jax.Array,
    radius: jax.Array,
    alive: jax.Array,
    divides: jax.Array,
    mother: jax.Array,
    direction: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Positions, radii and alive flags after the divisions.

    A dividing cell of radius r becomes two daughters with half of its volume
    each, of radius ``r / 2^(1/d)`` in d dimensions: one stays in its slot,
    the other is born in the free slot that :func:`birth_slots` gave it. The
    two sit at a quarter of the mother's diameter on either side of the
    mother's centre along ``direction``, so that the centre of the pair is
    where the mother was; contact forces then push them apart.

    Args:
        pos: (N, d) positions.
        radius: (N,) radii.
        alive: (N,) slots that hold a cell.
        divides: (N,) the dividing cells, from :func:`birth_slots`.
        mother: (N,) the mother of every slot, from :func:`birth_slots`.
        direction: (N, d) unit vectors, the direction of each division.

    Returns:
        ``(pos, radius, alive)`` with the daughters.
    """
    d = pos.shape[1]
    born = mother >= 0
    parent = jnp.where(born, mother, 0)
    shift = (0.5 * radius)[:, None] * direction  # a quarter of the diameter
    pos = jnp.where(divides[:, None], pos - shift, pos)
    pos = jnp.where(born[:, None], pos[parent] + 2.0 * shift[parent], pos)
    small = radius * 2.0 ** (-1.0 / d)
    radius = jnp.where(divides, small, radius)
    radius = jnp.where(born, small[parent], radius)
    return pos, radius, alive | born


def overlap_sum(
    pos: jax.Array,
    radius: jax.Array,
    alive: jax.Array,
    box: Box,
    neighbors: BruteForce | CellList | None = None,
) -> jax.Array:
    """(N,) sum of the overlaps ``r_i + r_j - |x_i - x_j|`` of every cell with its neighbours.

    With a harmonic contact of stiffness k it is the sum of the magnitudes of
    the contact forces divided by k: a measure of how strongly a cell is
    compressed, which stays positive when the forces from all sides cancel.
    Dead slots give zero.
    """
    neighbors = BruteForce() if neighbors is None else neighbors
    idx, _ = neighbors.candidates(pos, alive, box)
    n = pos.shape[0]
    other = jnp.broadcast_to(jnp.arange(n), (n, n)) if idx is None else idx
    valid = other < n
    j = jnp.where(valid, other, 0)
    distance = jnp.linalg.norm(box.displacement(pos[:, None, :], pos[j]), axis=-1)
    overlap = radius[:, None] + radius[j] - distance
    touching = valid & alive[:, None] & alive[j] & (j != jnp.arange(n)[:, None])
    return jnp.sum(jnp.where(touching, jnp.maximum(overlap, 0.0), 0.0), axis=1)
