"""Self-propelled particles with an orientation, in 2D.

An active particle has a heading angle ``theta`` and propels itself along
``(cos(theta), sin(theta))``. The heading is kept by the environment next to
the particle state; the functions here advance and analyse it. To move the
particles, pass ``speed * heading_vector(theta)`` (plus the fluid velocity,
if any) as the ``drift`` of :func:`~particlegym.physics.langevin_baoab`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from particlegym.physics.space import Box


def heading_vector(theta: jax.Array) -> jax.Array:
    """Unit vectors ``(cos(theta), sin(theta))`` with shape (..., 2)."""
    return jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=-1)


def wrap_angle(theta: jax.Array) -> jax.Array:
    """Map angles to the interval (-pi, pi]."""
    return jnp.arctan2(jnp.sin(theta), jnp.cos(theta))


def rotate(
    theta: jax.Array, dt: float, *, omega=0.0, diffusion=0.0, key: jax.Array | None = None
) -> jax.Array:
    """Advance the headings by one step of ``d(theta) = omega dt + sqrt(2 D_r) dW``.

    Args:
        theta: Heading angles.
        dt: Time step.
        omega: Angular velocity, scalar or one value per particle, for example
            half the vorticity of the surrounding flow plus a steering rate.
        diffusion: Rotational diffusion coefficient ``D_r``.
        key: Random key; required if ``diffusion`` is not zero.

    Returns:
        The new angles in (-pi, pi].
    """
    theta = theta + omega * dt
    if key is not None:
        noise = jax.random.normal(key, theta.shape, dtype=theta.dtype)
        theta = theta + jnp.sqrt(2.0 * diffusion * dt) * noise
    elif not (isinstance(diffusion, int | float) and diffusion == 0):
        raise ValueError("a random key is required when the rotational diffusion is not 0")
    return wrap_angle(theta)


def steering_rate(theta: jax.Array, target: jax.Array, rate) -> jax.Array:
    """Angular velocity ``rate * sin(target - theta)`` that turns a heading towards ``target``.

    This is the restoring torque of a bottom-heavy (gyrotactic) swimmer; the
    heading relaxes to ``target`` in a time of about ``1 / rate``.
    """
    return rate * jnp.sin(target - theta)


def mean_heading(
    pos: jax.Array, theta: jax.Array, box: Box, radius: float, alive: jax.Array | None = None
) -> jax.Array:
    """Direction of the mean heading of the particles within ``radius``, itself included.

    This is the alignment rule of the model of Vicsek et al. (1995). All
    pairs are compared, so the work grows as N^2.

    Returns:
        (N,) angles.
    """
    dr = box.displacement(pos[:, None, :], pos[None, :, :])
    near = jnp.sum(dr * dr, axis=-1) <= radius * radius
    if alive is not None:
        near = near & alive[None, :]
    weight = near.astype(theta.dtype)
    return jnp.arctan2(weight @ jnp.sin(theta), weight @ jnp.cos(theta))


def polar_order(theta: jax.Array, alive: jax.Array | None = None) -> jax.Array:
    """Length of the mean heading vector.

    It is 1 for parallel headings and about ``N^(-1/2)`` for random ones.
    """
    weight = jnp.ones_like(theta) if alive is None else alive.astype(theta.dtype)
    mean = jnp.sum(weight[..., None] * heading_vector(theta), axis=-2) / jnp.sum(weight)
    return jnp.sqrt(jnp.sum(mean * mean, axis=-1))
