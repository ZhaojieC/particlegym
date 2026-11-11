# ParticleGym, https://github.com/ZhaojieC/particlegym (MIT license).
# Please cite it in published work: particlegym.cite() gives the references.
"""Soft bodies made of square voxels: point masses at the corners, springs between them.

A :class:`VoxelBody` is drawn as text, one string per row from top to bottom:

    "HHH"
    "S.S"

Every filled voxel puts point masses at its four corners, shared with its
neighbours, and springs along its four edges and its two diagonals. Edges
shared by two voxels carry one spring. This mass-spring voxel is the model
of voxel-based soft robots in two dimensions, as in Evolution Gym (Bhatia et
al. 2021). The kinds of voxel are

- ``S`` soft: passive springs;
- ``R`` rigid: passive springs, stiffer by ``rigidity``;
- ``H`` horizontal actuator: the rest length of its horizontal edges, and
  with it of its diagonals, follows a command;
- ``V`` vertical actuator: the same for its vertical edges;
- ``.`` empty.

An actuator with the command a in [-1, 1] takes the rest length ``(1 + s a)``
times the voxel size along its direction, with the strength s. An edge shared
by two voxels takes the mean of their rest lengths.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

KINDS = "SRHV"


@dataclasses.dataclass(frozen=True, eq=False)
class VoxelBody:
    """The lattice of a body of square voxels.

    Args:
        layout: Rows of the body from top to bottom, one character per
            voxel: ``S``, ``R``, ``H``, ``V`` or ``.`` (see the module).
        size: Edge length of a voxel.

    Attributes:
        positions: (n_nodes, 2) corners at rest, the lowest row at y = 0 and
            the leftmost column at x = 0.
        masses: (n_nodes,) a quarter of the mass of a voxel from each voxel
            that a node belongs to (voxels have unit mass).
        voxels: (n_voxels, 4) corner nodes, counter-clockwise from the lower left.
        kinds: (n_voxels,) index of the kind in ``KINDS``.
        cells: (n_voxels, 2) column and row of every voxel, the lowest row 0.
        springs: (n_springs, 2) node pairs: first the edges, then two
            diagonals per voxel.
        actuators: (n_actuators,) the voxels of kind ``H`` or ``V``, in
            reading order of the layout.
    """

    layout: tuple[str, ...]
    size: float = 1.0

    def __post_init__(self) -> None:
        rows = tuple(self.layout)
        if not rows or len({len(row) for row in rows}) != 1:
            raise ValueError("layout must be rows of equal length")
        if any(c not in KINDS + "." for row in rows for c in row):
            raise ValueError(f"layout characters must be among {KINDS!r} and '.'")
        cells = [
            (x, len(rows) - 1 - r, KINDS.index(c))
            for r, row in enumerate(rows)
            for x, c in enumerate(row)
            if c != "."
        ]
        if not cells:
            raise ValueError("layout has no voxels")
        nodes: dict[tuple[int, int], int] = {}
        voxels = []
        for x, y, _ in cells:
            corners = ((x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1))
            voxels.append([nodes.setdefault(corner, len(nodes)) for corner in corners])
        edges: dict[tuple[int, int], list[int]] = {}
        horizontal: dict[tuple[int, int], bool] = {}
        for v, (a, b, c, d) in enumerate(voxels):
            for p, q, along_x in ((a, b, True), (b, c, False), (d, c, True), (a, d, False)):
                key = (min(p, q), max(p, q))
                edges.setdefault(key, []).append(v)
                horizontal[key] = along_x
        if not _connected(len(nodes), list(edges)):
            raise ValueError("the voxels of the layout must form one connected body")
        edge_pairs = np.array(list(edges), dtype=np.int32)
        # The voxels of each edge; an edge of one voxel lists it twice.
        edge_voxels = np.array([(vs[0], vs[-1]) for vs in edges.values()], dtype=np.int32)
        voxels = np.array(voxels, dtype=np.int32)
        diagonals = np.concatenate([voxels[:, [0, 2]], voxels[:, [1, 3]]])
        counts = np.bincount(voxels.ravel(), minlength=len(nodes))
        object.__setattr__(self, "layout", rows)
        object.__setattr__(self, "positions", self.size * np.array(list(nodes), dtype=float))
        object.__setattr__(self, "masses", counts / 4.0)
        object.__setattr__(self, "voxels", voxels)
        object.__setattr__(self, "kinds", np.array([k for _, _, k in cells], dtype=np.int32))
        object.__setattr__(self, "cells", np.array([(x, y) for x, y, _ in cells], dtype=np.int32))
        object.__setattr__(self, "springs", np.concatenate([edge_pairs, diagonals]))
        object.__setattr__(self, "edge_voxels", edge_voxels)
        object.__setattr__(
            self, "edge_horizontal", np.array([horizontal[key] for key in edges], dtype=bool)
        )
        object.__setattr__(
            self, "actuators", np.flatnonzero(np.isin(self.kinds, (2, 3))).astype(np.int32)
        )

    @property
    def n_nodes(self) -> int:
        """Number of point masses."""
        return len(self.positions)

    @property
    def n_voxels(self) -> int:
        """Number of voxels."""
        return len(self.voxels)

    @property
    def n_actuators(self) -> int:
        """Number of voxels of kind ``H`` or ``V``."""
        return len(self.actuators)

    def stiffness(self, k: float, rigidity: float = 5.0) -> np.ndarray:
        """(n_springs,) spring constants: k, times ``rigidity`` in rigid voxels.

        An edge between a rigid and another voxel takes the mean.
        """
        factor = np.where(self.kinds == 1, rigidity, 1.0)
        edges = factor[self.edge_voxels].mean(axis=1)
        return k * np.concatenate([edges, factor, factor])

    def rest_lengths(self, command: jax.Array, strength: float) -> jax.Array:
        """(n_springs,) rest lengths for the actuator commands (n_actuators,) in [-1, 1]."""
        stretch = 1.0 + strength * jnp.asarray(command)
        ones = jnp.ones(self.n_voxels, stretch.dtype)
        kinds = self.kinds[self.actuators]
        along_x = ones.at[self.actuators].set(jnp.where(kinds == 2, stretch, 1.0))
        along_y = ones.at[self.actuators].set(jnp.where(kinds == 3, stretch, 1.0))
        edges = jnp.where(
            self.edge_horizontal,
            along_x[self.edge_voxels].mean(axis=1),
            along_y[self.edge_voxels].mean(axis=1),
        )
        diagonal = jnp.sqrt(along_x**2 + along_y**2)
        return self.size * jnp.concatenate([edges, diagonal, diagonal])


def _connected(n_nodes: int, pairs: list[tuple[int, int]]) -> bool:
    """Whether the edges connect all nodes."""
    parent = list(range(n_nodes))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in pairs:
        parent[root(a)] = root(b)
    return len({root(i) for i in range(n_nodes)}) == 1
