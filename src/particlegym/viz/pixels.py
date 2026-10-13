"""Pixel images computed in JAX, for image observations.

The functions return, for every pixel, the fraction of it that is covered by
a set of discs or boxes, a value from 0 to 1. They are pure and have static
shapes, so they can be used inside a compiled environment step. Every pixel is
tested against every shape, which is efficient for the small images (for
example 84 x 84 pixels) and few objects of game-like environments. For large
pictures use :class:`particlegym.viz.Canvas`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def pixel_centres(lo, hi, shape: tuple[int, int], dtype=None) -> jax.Array:
    """Coordinates of the pixel centres of an image of the region ``[lo, hi]``.

    Args:
        lo: Lower-left corner of the region.
        hi: Upper-right corner of the region.
        shape: Image size as ``(height, width)``.

    Returns:
        (height, width, 2) points. Row 0 is the top of the region, so that
        the image has the usual orientation when displayed.
    """
    height, width = shape
    x = lo[0] + (jnp.arange(width, dtype=dtype) + 0.5) * ((hi[0] - lo[0]) / width)
    y = hi[1] - (jnp.arange(height, dtype=dtype) + 0.5) * ((hi[1] - lo[1]) / height)
    return jnp.stack(jnp.meshgrid(x, y), axis=-1)


def _coverage(distance: jax.Array, mask, pixel) -> jax.Array:
    """Largest pixel coverage over the shapes (last axis), from signed distances."""
    cover = jnp.clip(0.5 - distance / pixel, 0.0, 1.0)
    if mask is not None:
        cover = jnp.where(mask, cover, 0.0)
    return jnp.max(cover, axis=-1, initial=0.0)


def disc_coverage(points: jax.Array, pos, radius, pixel, mask=None) -> jax.Array:
    """Coverage of the pixels at ``points`` by discs.

    The coverage of a pixel falls linearly from 1 to 0 over one pixel size
    across the edge of a disc, which approximates the covered area fraction.

    Args:
        points: (..., 2) pixel centres, from :func:`pixel_centres`.
        pos: (N, 2) disc centres.
        radius: Disc radii, scalar or (N,).
        pixel: Edge length of a pixel.
        mask: (N,) booleans; discs with a false entry are left out.

    Returns:
        Array with the leading shape of ``points``, values in [0, 1].
    """
    offset = points[..., None, :] - pos
    distance = jnp.sqrt(jnp.sum(offset * offset, axis=-1)) - radius
    return _coverage(distance, mask, pixel)


def box_coverage(points: jax.Array, lo, hi, pixel, mask=None) -> jax.Array:
    """Coverage of the pixels at ``points`` by boxes with corners ``lo``, ``hi`` (both (M, 2))."""
    centre, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    q = jnp.abs(points[..., None, :] - centre) - half
    outside = jnp.sqrt(jnp.sum(jnp.maximum(q, 0.0) ** 2, axis=-1))
    distance = outside + jnp.minimum(jnp.max(q, axis=-1), 0.0)
    return _coverage(distance, mask, pixel)
