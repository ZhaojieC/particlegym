"""Neighbour search with static array shapes.

A strategy returns, for every particle, a fixed-size row of candidate
neighbour indices padded with the sentinel value ``N`` (the number of
particles), or ``None`` to request a dense evaluation over all pairs. Fixed
shapes let a whole simulation step be compiled with ``jax.jit`` and batched
with ``jax.vmap``. Candidates are a superset of the interacting pairs; the
exact cut-off is applied when forces are computed.
"""

from __future__ import annotations

import dataclasses
import itertools

import jax
import jax.numpy as jnp
import numpy as np

from particlegym.physics.space import Box


@dataclasses.dataclass(frozen=True)
class BruteForce:
    """Every particle is a candidate neighbour of every other particle.

    The work grows as N^2, but there is nothing to tune. This is usually the
    fastest choice for systems of up to a few hundred particles.
    """

    def validate(self, box: Box) -> None:
        """Brute force works for any box."""

    def candidates(self, pos: jax.Array, alive: jax.Array, box: Box) -> tuple[None, jax.Array]:
        """Return ``None`` (all pairs, evaluated densely) and an overflow flag (always False)."""
        return None, jnp.zeros((), dtype=bool)


@dataclasses.dataclass(frozen=True)
class CellList:
    """Cell-list neighbour search: the work grows linearly with N.

    The box is divided into cells no smaller than ``cutoff``. The candidates of
    a particle are the members of its own cell and of the adjacent cells. Each
    cell stores at most ``capacity`` particles; if any cell holds more, the
    overflow flag is raised and some interactions are missing. Increase
    ``capacity`` and run again in that case.

    Along non-periodic axes, particles outside the box are assigned to the
    nearest boundary cell. This never separates two particles that are closer
    than ``cutoff``, so no interaction is lost.

    Args:
        cutoff: Largest interaction distance between any two particles.
        capacity: Maximum number of particles per cell.
    """

    cutoff: float
    capacity: int = 16

    def __post_init__(self) -> None:
        if self.cutoff <= 0:
            raise ValueError("cutoff must be positive")
        if self.capacity < 1:
            raise ValueError("capacity must be at least 1")

    def grid(self, box: Box) -> tuple[int, ...]:
        """Number of cells along each axis of ``box``."""
        return tuple(max(1, int(length // self.cutoff)) for length in box.length)

    def validate(self, box: Box) -> None:
        """Check that the cell grid is compatible with the periodicity of ``box``."""
        for axis, (n_cells, periodic) in enumerate(zip(self.grid(box), box.periodic, strict=True)):
            if periodic and n_cells < 3:
                raise ValueError(
                    f"the box is too small along periodic axis {axis} for a cell list "
                    f"with cutoff {self.cutoff}; use BruteForce instead"
                )

    def candidates(
        self, pos: jax.Array, alive: jax.Array, box: Box
    ) -> tuple[jax.Array, jax.Array]:
        """Return (N, 3^d * capacity) candidate indices and the overflow flag."""
        n, d = pos.shape
        grid_np = np.asarray(self.grid(box))
        n_cells = int(np.prod(grid_np))
        capacity = self.capacity
        # Per-axis constants as (d, 1) columns: arrays below are coordinate-first.
        grid = jnp.asarray(grid_np[:, None], dtype=jnp.int32)
        strides = jnp.asarray(np.cumprod(np.r_[1, grid_np[:-1]])[:, None], dtype=jnp.int32)
        periodic = jnp.asarray(np.asarray(box.periodic)[:, None])

        # Cell coordinates (d, N) of every particle; dead particles go to an
        # extra cell with index n_cells that is never visited.
        lo = jnp.asarray(np.asarray(box.lo)[:, None], dtype=pos.dtype)
        cell_size = jnp.asarray((box.length / grid_np)[:, None], dtype=pos.dtype)
        coords = jnp.floor((pos.T - lo) / cell_size).astype(jnp.int32)
        coords = jnp.where(periodic, jnp.mod(coords, grid), jnp.clip(coords, 0, grid - 1))
        cell = jnp.where(alive, jnp.sum(coords * strides, axis=0), n_cells)

        # Sort particles by cell and collect at most `capacity` members per cell.
        # After sorting, the members of cell c start at the number of particles
        # in cells with a smaller index (an exclusive prefix sum of the counts).
        order = jnp.argsort(cell).astype(jnp.int32)
        counts = jnp.zeros(n_cells + 1, dtype=jnp.int32).at[cell].add(1)
        count = counts[:n_cells]
        start = (jnp.cumsum(counts) - counts)[:n_cells]
        overflow = jnp.any(count > capacity)
        slot = jnp.arange(capacity)
        members = order[jnp.minimum(start[:, None] + slot, n - 1)]
        members = jnp.where(slot < count[:, None], members, n)
        # Extra empty row used for neighbour cells outside a non-periodic box.
        members = jnp.concatenate([members, jnp.full((1, capacity), n, dtype=jnp.int32)])

        # Visit the 3^d cells around each particle's cell: arrays (d, 3^d, N).
        offsets = np.asarray(list(itertools.product((-1, 0, 1), repeat=d))).T
        neighbor = coords[:, None, :] + jnp.asarray(offsets[:, :, None], dtype=jnp.int32)
        inside = (neighbor >= 0) & (neighbor < grid[:, None])
        valid = jnp.all(inside | periodic[:, None], axis=0)
        # Offsets are -1, 0 or 1, so periodic wrapping needs at most one addition
        # or subtraction (cheaper than an integer modulo).
        neighbor = jnp.where(neighbor < 0, neighbor + grid[:, None], neighbor)
        neighbor = jnp.where(neighbor >= grid[:, None], neighbor - grid[:, None], neighbor)
        neighbor_cell = jnp.sum(neighbor * strides[:, None], axis=0)
        neighbor_cell = jnp.where(valid, neighbor_cell, n_cells)  # (3^d, N)

        idx = members[neighbor_cell.T].reshape(n, -1)
        idx = jnp.where(idx == jnp.arange(n, dtype=jnp.int32)[:, None], n, idx)
        return idx, overflow
