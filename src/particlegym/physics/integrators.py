"""Time integrators.

Every step function is pure and returns ``(new_state, ForceInfo)``, so it can
be compiled with :func:`jax.jit`, iterated with :func:`jax.lax.scan` and
batched with :func:`jax.vmap`. Dead particles are left unchanged with zero
velocity.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from particlegym.physics.forces import ForceField, ForceInfo
from particlegym.physics.state import ParticleState


def init_forces(state: ParticleState, force_field: ForceField) -> tuple[ParticleState, ForceInfo]:
    """Evaluate and store the forces at the current positions.

    Call this once before the first step, and again whenever positions are
    changed outside an integrator.
    """
    force, info = force_field.forces(state.pos, state.vel, state.radius, state.alive)
    return state._replace(force=force), info


def velocity_verlet(
    state: ParticleState, force_field: ForceField, dt: float
) -> tuple[ParticleState, ForceInfo]:
    """One velocity Verlet step.

    With conservative forces only, this is a symplectic, time-reversible
    integrator of Newton's equations: the total energy is conserved up to
    bounded fluctuations of order ``dt^2``.
    """
    alive = state.alive[:, None]
    inv_mass = 1.0 / state.mass[:, None]
    vel = state.vel + 0.5 * dt * state.force * inv_mass
    pos = jnp.where(alive, force_field.box.wrap(state.pos + dt * vel), state.pos)
    force, info = force_field.forces(pos, vel, state.radius, state.alive)
    vel = jnp.where(alive, vel + 0.5 * dt * force * inv_mass, 0.0)
    return state._replace(pos=pos, vel=vel, force=force), info


def langevin_baoab(
    state: ParticleState,
    force_field: ForceField,
    dt: float,
    *,
    gamma,
    kT=0.0,
    drift: jax.Array | None = None,
    key: jax.Array | None = None,
) -> tuple[ParticleState, ForceInfo]:
    """One BAOAB step of Langevin dynamics with a drift velocity.

    Integrates

        m dv/dt = F(x, v) - m gamma (v - u) + sqrt(2 m gamma kT) xi(t),

    with Gaussian white noise ``xi``, using the BAOAB splitting of Leimkuhler &
    Matthews (2013). The O part (friction, drift and noise) is solved exactly,
    so the step is stable for any ``gamma * dt``.

    The drift velocity ``u`` is what an agent controls. For pedestrians, the
    driving term of the social force model, ``m (v0 e - v) / tau``, is exactly
    the friction term with ``gamma = 1 / tau`` and ``u = v0 e``. For passive
    Brownian particles ``u = 0``. With ``gamma = 0`` the step reduces to
    velocity Verlet.

    Args:
        state: Current state with up-to-date forces.
        force_field: The interactions.
        dt: Time step.
        gamma: Friction rate, scalar or (N,).
        kT: Thermal energy of the noise.
        drift: (N, d) drift velocities; zero if omitted.
        key: JAX random key. Required unless ``kT`` is exactly 0.
    """
    if key is None and not (isinstance(kT, int | float) and kT == 0):
        raise ValueError("a random key is required when kT is not 0")
    alive = state.alive[:, None]
    mass = state.mass[:, None]
    dtype = state.pos.dtype
    gamma = jnp.asarray(gamma, dtype=dtype)
    if gamma.ndim == 1:
        gamma = gamma[:, None]
    u = jnp.zeros((), dtype=dtype) if drift is None else drift
    c1 = jnp.exp(-gamma * dt)

    vel = state.vel + 0.5 * dt * state.force / mass  # B
    pos = state.pos + 0.5 * dt * vel  # A
    vel = u + c1 * (vel - u)  # O: exact relaxation towards the drift velocity
    if key is not None:
        c2 = jnp.sqrt(-jnp.expm1(-2.0 * gamma * dt))
        noise = jax.random.normal(key, vel.shape, dtype=dtype)
        vel = vel + c2 * jnp.sqrt(kT / mass) * noise
    pos = pos + 0.5 * dt * vel  # A
    pos = jnp.where(alive, force_field.box.wrap(pos), state.pos)
    force, info = force_field.forces(pos, vel, state.radius, state.alive)
    vel = jnp.where(alive, vel + 0.5 * dt * force / mass, 0.0)  # B
    return state._replace(pos=pos, vel=vel, force=force), info
