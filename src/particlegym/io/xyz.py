"""Extended XYZ output, readable by OVITO and ASE."""

from __future__ import annotations

import os
import string
from typing import TextIO

import numpy as np

from particlegym.physics.space import Box
from particlegym.physics.state import ParticleState

_PROPERTIES = "species:S:1:pos:R:3:velo:R:3:radius:R:1:id:I:1"


def write_extxyz(
    file: str | os.PathLike | TextIO,
    state: ParticleState,
    box: Box,
    *,
    time: float | None = None,
    type_names=None,
) -> None:
    """Append one frame of ``state`` to an extended XYZ file.

    Dead particles are omitted; the ``id`` column keeps the original particle
    indices so that trajectories can be followed. 2D systems are written with
    z = 0 in a cell of unit thickness.

    Args:
        file: Path (opened in append mode) or text file object.
        state: Particle state to write.
        box: Simulation box, written as the cell.
        time: Simulation time stored in the frame header.
        type_names: Species name for each ``kind`` value; defaults to A, B, C, ...
    """
    alive = np.asarray(state.alive, dtype=bool)
    ids = np.flatnonzero(alive)
    pos = np.asarray(state.pos, dtype=float)[ids]
    vel = np.asarray(state.vel, dtype=float)[ids]
    radius = np.asarray(state.radius, dtype=float)[ids]
    kind = np.asarray(state.kind)[ids]
    lo, length = np.asarray(box.lo, dtype=float), box.length.astype(float)
    pbc = ["T" if p else "F" for p in box.periodic]
    if box.dim == 2:
        pos = np.column_stack([pos, np.zeros(len(ids))])
        vel = np.column_stack([vel, np.zeros(len(ids))])
        lo, length, pbc = np.r_[lo, -0.5], np.r_[length, 1.0], [*pbc, "F"]
    names = list(type_names) if type_names is not None else list(string.ascii_uppercase)

    lattice = f"{length[0]:.8g} 0 0 0 {length[1]:.8g} 0 0 0 {length[2]:.8g}"
    header = (
        f'Lattice="{lattice}" Origin="{lo[0]:.8g} {lo[1]:.8g} {lo[2]:.8g}" '
        f'Properties={_PROPERTIES} pbc="{" ".join(pbc)}"'
    )
    if time is not None:
        header += f" Time={time:.8g}"
    lines = [str(len(ids)), header]
    for i, p, v, r, k in zip(ids, pos, vel, radius, kind, strict=True):
        lines.append(
            f"{names[k]} {p[0]:.8g} {p[1]:.8g} {p[2]:.8g} "
            f"{v[0]:.8g} {v[1]:.8g} {v[2]:.8g} {r:.8g} {i}"
        )
    text = "\n".join(lines) + "\n"
    if hasattr(file, "write"):
        file.write(text)
    else:
        with open(file, "a") as f:
            f.write(text)
