"""Assembly of all forces acting on the particles."""

from __future__ import annotations

import dataclasses
from typing import NamedTuple

import jax
import jax.numpy as jnp

from particlegym.physics.geometry import Walls
from particlegym.physics.neighbors import BruteForce, CellList
from particlegym.physics.potentials import PairDissipation, PairPotential
from particlegym.physics.space import Box
from particlegym.physics.state import ParticleState


class ForceInfo(NamedTuple):
    """Diagnostics returned with every force evaluation.

    Attributes:
        potential_energy: Total conservative (pair plus wall) energy.
        overflow: True if the neighbour search dropped candidates; the forces
            are then incomplete and the cell-list capacity must be increased.
    """

    potential_energy: jax.Array
    overflow: jax.Array


def _pad(x: jax.Array, value) -> jax.Array:
    """Append one row filled with ``value``; it is gathered by the sentinel index N."""
    return jnp.concatenate([x, jnp.full((1, *x.shape[1:]), value, dtype=x.dtype)])


@dataclasses.dataclass(frozen=True, eq=False)
class ForceField:
    """Sum of all interactions acting on the particles.

    A force field is a static description of the model. Capture it in a
    closure (for example with :func:`functools.partial`) when compiling a step
    function with :func:`jax.jit`.

    Args:
        box: Simulation domain, used for minimum-image distances and wrapping.
        pair: Pair potentials and dissipative pair forces.
        walls: Particle-geometry interactions.
        neighbors: Neighbour search for the pair terms.
    """

    box: Box
    pair: tuple = ()
    walls: tuple = ()
    neighbors: BruteForce | CellList = dataclasses.field(default_factory=BruteForce)

    def __post_init__(self) -> None:
        object.__setattr__(self, "pair", tuple(self.pair))
        object.__setattr__(self, "walls", tuple(self.walls))
        for term in self.pair:
            if not isinstance(term, PairPotential | PairDissipation):
                raise TypeError(f"{term!r} is not a PairPotential or PairDissipation")
        for wall in self.walls:
            if not isinstance(wall, Walls):
                raise TypeError(f"{wall!r} is not a Walls instance")
            if wall.shapes.dim != self.box.dim:
                raise ValueError("wall geometry and box have different dimensions")
        self.neighbors.validate(self.box)

    def interaction_range(self, max_radius: float) -> float:
        """Largest pair interaction distance for particles of radius at most ``max_radius``."""
        return max((float(term.cutoff(2.0 * max_radius)) for term in self.pair), default=0.0)

    def check(self, state: ParticleState) -> None:
        """Check, outside of compiled code, that the setup can see all interactions.

        Raises:
            ValueError: If the state and box dimensions differ, if the cell list
                cut-off is shorter than the interaction range, or if the range
                exceeds half the box length along a periodic axis.
        """
        if state.dim != self.box.dim:
            raise ValueError("state and box have different dimensions")
        if not self.pair:
            return
        reach = self.interaction_range(float(jnp.max(state.radius)))
        if isinstance(self.neighbors, CellList) and self.neighbors.cutoff < reach:
            raise ValueError(
                f"cell list cutoff {self.neighbors.cutoff} is shorter than the "
                f"interaction range {reach}"
            )
        for length, periodic in zip(self.box.length, self.box.periodic, strict=True):
            if periodic and reach > length / 2:
                raise ValueError("interaction range exceeds half the periodic box length")

    def forces(
        self, pos: jax.Array, vel: jax.Array, radius: jax.Array, alive: jax.Array
    ) -> tuple[jax.Array, ForceInfo]:
        """Total force on every particle (zero for dead particles) and diagnostics."""
        force = jnp.zeros_like(pos)
        energy = jnp.zeros((), dtype=pos.dtype)
        overflow = jnp.zeros((), dtype=bool)
        if self.pair:
            idx, overflow = self.neighbors.candidates(pos, alive, self.box)
            pair_force, pair_energy = self._pair_forces(pos, vel, radius, alive, idx)
            force = force + pair_force
            energy = energy + pair_energy
        for wall in self.walls:
            wall_energy, grad = jax.value_and_grad(wall.energy)(pos, radius, alive)
            force = force - grad
            energy = energy + wall_energy
            if wall.friction:
                force = force + wall.dissipative_force(pos, vel, radius, alive)
        return jnp.where(alive[:, None], force, 0.0), ForceInfo(energy, overflow)

    def potential_energy(self, pos: jax.Array, radius: jax.Array, alive: jax.Array) -> jax.Array:
        """Total conservative energy, written directly as a function of positions.

        :meth:`forces` assembles pair forces from per-pair derivatives instead;
        comparing it with the gradient of this function validates the assembly.
        """
        energy = jnp.zeros((), dtype=pos.dtype)
        potentials = [term for term in self.pair if isinstance(term, PairPotential)]
        if potentials:
            idx, _ = self.neighbors.candidates(pos, alive, self.box)
            pairs = self._pair_geometry(pos, None, radius, alive, idx)
            r = jnp.sqrt(jnp.where(pairs.near, pairs.r2, 1.0))
            for term in potentials:
                e = term.energy(r, pairs.sigma)
                energy = energy + 0.5 * jnp.sum(jnp.where(pairs.near, e, 0.0))
        for wall in self.walls:
            energy = energy + wall.energy(pos, radius, alive)
        return energy

    def _pair_geometry(self, pos, vel, radius, alive, idx) -> _Pairs:
        """Geometry of every particle's candidate pairs.

        Vectors are stored coordinate-first, with shape (d, N, K), so that the
        compiler vectorises over pairs rather than over the two or three
        coordinates. For dense evaluation (``idx is None``) the neighbour data
        are broadcast instead of gathered. Relative velocities are computed
        only if ``vel`` is given.
        """
        q = pos.T
        if idx is None:
            q_j = q[:, None, :]
            radius_j = radius[None, :]
            alive_j = alive[None, :]
        else:
            q_j = _pad(pos, 0.0).T[:, idx]
            radius_j = _pad(radius, 0.0)[idx]
            alive_j = _pad(alive, False)[idx]  # the sentinel row is dead

        dr = self.box.displacement(q[:, :, None], q_j, axis=0)
        r2 = jnp.sum(dr * dr, axis=0)
        sigma = radius[:, None] + radius_j
        reach = self.pair[0].cutoff(sigma)
        for term in self.pair[1:]:
            reach = jnp.maximum(reach, term.cutoff(sigma))
        # Self pairs and coincident particles (r = 0, no defined direction) are skipped.
        near = alive[:, None] & alive_j & (r2 < reach * reach) & (r2 > 0)
        dv = None
        if vel is not None:
            vel_j = vel.T[:, None, :] if idx is None else _pad(vel, 0.0).T[:, idx]
            dv = vel_j - vel.T[:, :, None]
        return _Pairs(dr, r2, sigma, near, dv)

    def _pair_forces(self, pos, vel, radius, alive, idx):
        """Pair forces summed per particle over its candidate pairs."""
        dissipative = any(isinstance(term, PairDissipation) for term in self.pair)
        pairs = self._pair_geometry(pos, vel if dissipative else None, radius, alive, idx)
        near, sigma = pairs.near, pairs.sigma
        r = jnp.sqrt(jnp.where(near, pairs.r2, 1.0))
        unit = pairs.dr / r
        force = jnp.zeros_like(pos.T)
        energy = jnp.zeros((), dtype=pos.dtype)
        for term in self.pair:
            if isinstance(term, PairPotential):

                def total(r_, term=term):
                    return jnp.sum(jnp.where(near, term.energy(r_, sigma), 0.0))

                pair_energy, de_dr = jax.value_and_grad(total)(r)
                force = force - jnp.sum(de_dr * unit, axis=2)
                # Every pair appears twice, once from each side.
                energy = energy + 0.5 * pair_energy
            else:
                f = term.force(unit, r, pairs.dv, sigma)
                force = force + jnp.sum(jnp.where(near, f, 0.0), axis=2)
        return force.T, energy


class _Pairs(NamedTuple):
    """Candidate pairs; vectors have shape (d, N, K), scalars (N, K)."""

    dr: jax.Array  # displacement from neighbour j to particle i
    r2: jax.Array  # squared distance
    sigma: jax.Array  # contact distance r_i + r_j
    near: jax.Array  # pair interacts
    dv: jax.Array | None  # relative velocity v_j - v_i
