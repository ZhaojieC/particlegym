"""Observables of a particle state. Dead particles are excluded."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from particlegym.physics.state import ParticleState


def kinetic_energy(state: ParticleState) -> jax.Array:
    """Total kinetic energy of the alive particles."""
    per_particle = 0.5 * state.mass * jnp.sum(state.vel * state.vel, axis=-1)
    return jnp.sum(jnp.where(state.alive, per_particle, 0.0))


def temperature(state: ParticleState) -> jax.Array:
    """Kinetic temperature ``2 K / (d N)`` in energy units (k_B T).

    No correction is made for conserved quantities such as total momentum.
    """
    n_alive = jnp.sum(state.alive)
    return 2.0 * kinetic_energy(state) / (state.dim * n_alive)


def total_momentum(state: ParticleState) -> jax.Array:
    """Total momentum (d,) of the alive particles."""
    momentum = state.mass[:, None] * state.vel
    return jnp.sum(jnp.where(state.alive[:, None], momentum, 0.0), axis=0)
