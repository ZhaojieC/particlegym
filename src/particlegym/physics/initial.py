"""Initial conditions."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def lattice_positions(lo, hi, n: int) -> np.ndarray:
    """Place ``n`` particles on a square (2D) or simple cubic (3D) lattice.

    The lattice fills the region ``[lo, hi]`` with the same number of sites
    along every axis; sites are centred in their lattice cells.

    Returns:
        (n, d) array of positions.
    """
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    d = lo.shape[0]
    per_axis = int(np.ceil(n ** (1.0 / d) - 1e-9))
    spacing = (hi - lo) / per_axis
    sites = np.stack(np.meshgrid(*[np.arange(per_axis)] * d, indexing="ij"), axis=-1)
    sites = sites.reshape(-1, d)[:n]
    return lo + (sites + 0.5) * spacing


def random_positions(
    rng: np.random.Generator,
    lo,
    hi,
    radius,
    *,
    gap: float = 0.0,
    max_attempts: int = 100_000,
) -> np.ndarray:
    """Random sequential insertion of non-overlapping discs (2D) or spheres (3D).

    Particle centres are kept at least one radius away from the region
    boundary, so the particles fit inside ``[lo, hi]``.

    Args:
        rng: NumPy random generator.
        lo: Lower corner of the region.
        hi: Upper corner of the region.
        radius: (n,) particle radii; the number of particles is ``len(radius)``.
        gap: Extra clearance required between particle surfaces.
        max_attempts: Total number of trial insertions before giving up.

    Returns:
        (n, d) array of positions.
    """
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    radius = np.asarray(radius, dtype=float)
    pos = np.empty((radius.shape[0], lo.shape[0]))
    placed = 0
    for _ in range(max_attempts):
        if placed == radius.shape[0]:
            break
        r = radius[placed]
        trial = rng.uniform(lo + r, hi - r)
        dist = np.linalg.norm(pos[:placed] - trial, axis=1)
        if np.all(dist >= radius[:placed] + r + gap):
            pos[placed] = trial
            placed += 1
    if placed < radius.shape[0]:
        raise RuntimeError(
            f"placed only {placed} of {radius.shape[0]} particles; "
            "lower the packing fraction or increase max_attempts"
        )
    return pos


def maxwell_boltzmann(key: jax.Array, mass, kT, dim: int, *, remove_drift: bool = True):
    """Sample velocities from the Maxwell-Boltzmann distribution.

    Args:
        key: JAX random key.
        mass: (N,) particle masses.
        kT: Thermal energy (Boltzmann constant times temperature).
        dim: Number of spatial dimensions.
        remove_drift: Subtract the centre-of-mass velocity.

    Returns:
        (N, dim) velocities with the floating-point type of ``mass``.
    """
    mass = jnp.asarray(mass)
    vel = jax.random.normal(key, (mass.shape[0], dim), dtype=mass.dtype)
    vel = vel * jnp.sqrt(kT / mass)[:, None]
    if remove_drift:
        vel = vel - jnp.sum(mass[:, None] * vel, axis=0) / jnp.sum(mass)
    return vel
