"""Assembly of all forces acting on the particles."""

from __future__ import annotations

import dataclasses
from typing import NamedTuple

import jax
import jax.numpy as jnp

from particlegym.physics.bonded import BondedTerm
from particlegym.physics.external import ExternalForce
from particlegym.physics.geometry import Walls
from particlegym.physics.neighbors import BruteForce, CellList
from particlegym.physics.potentials import Mixture, PairDissipation, PairPotential, PairThermostat
from particlegym.physics.space import Box
from particlegym.physics.state import ParticleState


class ForceInfo(NamedTuple):
    """Diagnostics returned with every force evaluation.

    Attributes:
        potential_energy: Total conservative energy (pair, wall, bonded and
            external terms).
        overflow: True if the neighbour search dropped candidates; the forces
            are then incomplete and the cell-list capacity must be increased.
    """

    potential_energy: jax.Array
    overflow: jax.Array


def _pad(x: jax.Array, value) -> jax.Array:
    """Append one row filled with ``value``; it is gathered by the sentinel index N."""
    return jnp.concatenate([x, jnp.full((1, *x.shape[1:]), value, dtype=x.dtype)])


def _hash(x: jax.Array) -> jax.Array:
    """Mix the bits of 32-bit unsigned integers (the "lowbias32" integer hash)."""
    x = x ^ (x >> 16)
    x = x * jnp.uint32(0x7FEB352D)
    x = x ^ (x >> 15)
    x = x * jnp.uint32(0x846CA68B)
    return x ^ (x >> 16)


def _pair_noise(key: jax.Array, n: int, idx, dtype) -> jax.Array:
    """Random numbers with zero mean and unit variance, one for every candidate pair.

    The number of the pair (i, j) is a hash of the two particle indices and
    of the key. It is therefore the same when seen from i and from j, and
    the same for a dense and a cell-list evaluation. The numbers are
    uniformly distributed.
    """
    seed = jax.random.bits(key, (), jnp.uint32)
    i = jnp.arange(n, dtype=jnp.uint32)[:, None]
    j = i.T if idx is None else idx.astype(jnp.uint32)
    bits = _hash(_hash(jnp.minimum(i, j) ^ seed) + jnp.maximum(i, j) * jnp.uint32(0x9E3779B9))
    uniform = (bits >> 8).astype(dtype) * (1.0 / (1 << 24))
    return (uniform - 0.5) * (12.0**0.5)


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
        external: Forces on single particles, such as gravity.
        bonded: Bonds, angles and other terms between listed particles.
    """

    box: Box
    pair: tuple = ()
    walls: tuple = ()
    neighbors: BruteForce | CellList = dataclasses.field(default_factory=BruteForce)
    external: tuple = ()
    bonded: tuple = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "pair", tuple(self.pair))
        object.__setattr__(self, "walls", tuple(self.walls))
        object.__setattr__(self, "external", tuple(self.external))
        object.__setattr__(self, "bonded", tuple(self.bonded))
        for term in self.external:
            if not isinstance(term, ExternalForce):
                raise TypeError(f"{term!r} is not an ExternalForce")
        for term in self.bonded:
            if not isinstance(term, BondedTerm):
                raise TypeError(f"{term!r} is not a bonded term")
        for term in self.pair:
            if not isinstance(term, PairPotential | PairDissipation):
                raise TypeError(f"{term!r} is not a PairPotential or PairDissipation")
        for wall in self.walls:
            if not isinstance(wall, Walls):
                raise TypeError(f"{wall!r} is not a Walls instance")
            if wall.shapes.dim != self.box.dim:
                raise ValueError("wall geometry and box have different dimensions")
        self.neighbors.validate(self.box)

    @property
    def stochastic(self) -> bool:
        """True if a term draws random numbers, so that :meth:`forces` needs a key."""
        return any(isinstance(term, PairThermostat) for term in self.pair)

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
        self,
        pos: jax.Array,
        vel: jax.Array,
        radius: jax.Array,
        alive: jax.Array,
        *,
        mass: jax.Array | None = None,
        kind: jax.Array | None = None,
        key: jax.Array | None = None,
    ) -> tuple[jax.Array, ForceInfo]:
        """Total force on every particle (zero for dead particles) and diagnostics.

        Args:
            pos: (N, d) positions.
            vel: (N, d) velocities, used by the dissipative terms.
            radius: (N,) radii.
            alive: (N,) alive flags.
            mass: (N,) masses; required by external forces.
            kind: (N,) species; required by :class:`Mixture` potentials.
            key: Random key; required by :class:`PairThermostat`.
        """
        force = jnp.zeros_like(pos)
        energy = jnp.zeros((), dtype=pos.dtype)
        overflow = jnp.zeros((), dtype=bool)
        if self.pair:
            if self.stochastic and key is None:
                raise ValueError("a random key is required by the pair thermostat")
            idx, overflow = self.neighbors.candidates(pos, alive, self.box)
            pair_force, pair_energy = self._pair_forces(pos, vel, radius, alive, idx, kind, key)
            force = force + pair_force
            energy = energy + pair_energy
        for wall in self.walls:
            wall_energy, grad = jax.value_and_grad(wall.energy)(pos, radius, alive)
            force = force - grad
            energy = energy + wall_energy
            if wall.friction or wall.damping:
                force = force + wall.dissipative_force(pos, vel, radius, alive)
        for term in self.bonded:
            bonded_energy, grad = jax.value_and_grad(term.energy)(pos, self.box, alive)
            force = force - grad
            energy = energy + bonded_energy
        if self.external:
            if mass is None:
                raise ValueError("the particle masses are required by external forces")
            for term in self.external:
                force = force + term.force(pos, vel, mass)
                energy = energy + jnp.sum(jnp.where(alive, term.energy(pos, mass), 0.0))
        return jnp.where(alive[:, None], force, 0.0), ForceInfo(energy, overflow)

    def potential_energy(
        self,
        pos: jax.Array,
        radius: jax.Array,
        alive: jax.Array,
        *,
        mass: jax.Array | None = None,
        kind: jax.Array | None = None,
    ) -> jax.Array:
        """Total conservative energy, written directly as a function of positions.

        :meth:`forces` assembles pair forces from per-pair derivatives instead;
        comparing it with the gradient of this function validates the assembly.
        """
        energy = jnp.zeros((), dtype=pos.dtype)
        potentials = [term for term in self.pair if isinstance(term, PairPotential)]
        if potentials:
            idx, _ = self.neighbors.candidates(pos, alive, self.box)
            pairs = self._pair_geometry(pos, None, radius, alive, idx, kind)
            for term in potentials:
                e = _pair_energy(term, pairs.r, pairs)
                energy = energy + 0.5 * jnp.sum(jnp.where(pairs.near, e, 0.0))
        for wall in self.walls:
            energy = energy + wall.energy(pos, radius, alive)
        for term in self.bonded:
            energy = energy + term.energy(pos, self.box, alive)
        if self.external:
            if mass is None:
                raise ValueError("the particle masses are required by external forces")
            for term in self.external:
                energy = energy + jnp.sum(jnp.where(alive, term.energy(pos, mass), 0.0))
        return energy

    def pair_virial(
        self, pos: jax.Array, radius: jax.Array, alive: jax.Array, kind: jax.Array | None = None
    ) -> jax.Array:
        """Virial ``sum_{i<j} r_ij . F_ij`` of the conservative pair forces.

        The pressure of a periodic system of volume V in d dimensions is
        ``(2 K + W) / (d V)`` with the kinetic energy K and this virial W; see
        :func:`particlegym.physics.pressure`.
        """
        virial = jnp.zeros((), dtype=pos.dtype)
        potentials = [term for term in self.pair if isinstance(term, PairPotential)]
        if not potentials:
            return virial
        idx, _ = self.neighbors.candidates(pos, alive, self.box)
        pairs = self._pair_geometry(pos, None, radius, alive, idx, kind)
        for term in potentials:

            def total(r_, term=term):
                return jnp.sum(jnp.where(pairs.near, _pair_energy(term, r_, pairs), 0.0))

            de_dr = jax.grad(total)(pairs.r)
            # Every pair appears twice, once from each side.
            virial = virial - 0.5 * jnp.sum(jnp.where(pairs.near, pairs.r * de_dr, 0.0))
        return virial

    def _pair_geometry(self, pos, vel, radius, alive, idx, kind=None) -> _Pairs:
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
        # Pairs that do not interact are evaluated at the cut-off, where every
        # potential is zero and smooth, so their discarded gradients stay finite.
        far = jnp.where(reach > 0, reach, 1.0)
        r = jnp.where(near, jnp.sqrt(jnp.where(near, r2, 1.0)), far)
        dv = None
        if vel is not None:
            vel_j = vel.T[:, None, :] if idx is None else _pad(vel, 0.0).T[:, idx]
            dv = vel_j - vel.T[:, :, None]
        kind_i = kind_j = None
        if kind is not None:
            kind_i = kind[:, None]
            kind_j = kind[None, :] if idx is None else _pad(kind, 0)[idx]
        return _Pairs(dr, r, sigma, near, dv, kind_i, kind_j)

    def _pair_forces(self, pos, vel, radius, alive, idx, kind=None, key=None):
        """Pair forces summed per particle over its candidate pairs."""
        dissipative = any(isinstance(term, PairDissipation) for term in self.pair)
        pairs = self._pair_geometry(pos, vel if dissipative else None, radius, alive, idx, kind)
        near, sigma, r = pairs.near, pairs.sigma, pairs.r
        unit = pairs.dr / r
        force = jnp.zeros_like(pos.T)
        energy = jnp.zeros((), dtype=pos.dtype)
        for number, term in enumerate(self.pair):
            if isinstance(term, PairPotential):

                def total(r_, term=term):
                    return jnp.sum(jnp.where(near, _pair_energy(term, r_, pairs), 0.0))

                pair_energy, de_dr = jax.value_and_grad(total)(r)
                force = force - jnp.sum(de_dr * unit, axis=2)
                # Every pair appears twice, once from each side.
                energy = energy + 0.5 * pair_energy
            else:
                f = term.force(unit, r, pairs.dv, sigma)
                if isinstance(term, PairThermostat):
                    noise = _pair_noise(
                        jax.random.fold_in(key, number), pos.shape[0], idx, r.dtype
                    )
                    f = f + term.random_force(unit, r, sigma, noise)
                force = force + jnp.sum(jnp.where(near, f, 0.0), axis=2)
        return force.T, energy


def _pair_energy(term: PairPotential, r: jax.Array, pairs: _Pairs) -> jax.Array:
    """Energy of every candidate pair at the distances ``r``."""
    if isinstance(term, Mixture):
        return term.energy(r, pairs.sigma, pairs.kind_i, pairs.kind_j)
    return term.energy(r, pairs.sigma)


class _Pairs(NamedTuple):
    """Candidate pairs; vectors have shape (d, N, K), scalars (N, K)."""

    dr: jax.Array  # displacement from neighbour j to particle i
    r: jax.Array  # distance for interacting pairs, the cut-off otherwise
    sigma: jax.Array  # contact distance r_i + r_j
    near: jax.Array  # pair interacts
    dv: jax.Array | None  # relative velocity v_j - v_i
    kind_i: jax.Array | None = None  # species of particle i, shape (N, 1)
    kind_j: jax.Array | None = None  # species of the neighbour
