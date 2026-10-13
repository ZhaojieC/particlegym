"""Pixel images computed in JAX, for image observations.

The functions return, for every pixel, the fraction of it that is covered by
a set of discs or boxes, a value from 0 to 1. They are pure and have static
shapes, so they can be used inside a compiled environment step. Every pixel is
tested against every shape, which is efficient for the small images (for
example 84 x 84 pixels) and few objects of game-like environments. Boxes
that do not move, like the bricks of a wall, are drawn faster from a table
(:func:`box_coverage_table`). For large pictures use
:class:`particlegym.viz.Canvas`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


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


def _largest(values: jax.Array) -> jax.Array:
    """Maximum over the last axis, and at least 0.

    A few entries are compared one by one: a reduction over a short axis
    is much slower in compiled code (for 84 x 84 pixels and one disc, 0.4 ms
    instead of 0.03 ms on a CPU).
    """
    if values.shape[-1] > 8:
        return jnp.max(values, axis=-1, initial=0.0)
    largest = jnp.zeros(values.shape[:-1], values.dtype)
    for k in range(values.shape[-1]):
        largest = jnp.maximum(largest, values[..., k])
    return largest


def _coverage(distance: jax.Array, mask, pixel) -> jax.Array:
    """Largest pixel coverage over the shapes (last axis), from signed distances."""
    cover = jnp.clip(0.5 - distance / pixel, 0.0, 1.0)
    if mask is not None:
        cover = jnp.where(mask, cover, 0.0)
    return _largest(cover)


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
    pos = jnp.asarray(pos)
    dx = points[..., None, 0] - pos[..., 0]
    dy = points[..., None, 1] - pos[..., 1]
    distance = jnp.sqrt(dx * dx + dy * dy) - radius
    return _coverage(distance, mask, pixel)


def box_coverage(points: jax.Array, lo, hi, pixel, mask=None) -> jax.Array:
    """Coverage of the pixels at ``points`` by boxes with corners ``lo``, ``hi`` (both (M, 2)).

    Every pixel is tested against every box. With many boxes this compiled
    very slowly on a GPU (84 x 84 pixels and 40 boxes: the first call did
    not finish within 300 s with JAX 0.11.2). For boxes that do not move,
    use :func:`box_coverage_table`.
    """
    lo, hi = jnp.asarray(lo), jnp.asarray(hi)
    centre, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    qx = jnp.abs(points[..., None, 0] - centre[..., 0]) - half[..., 0]
    qy = jnp.abs(points[..., None, 1] - centre[..., 1]) - half[..., 1]
    outside = jnp.sqrt(jnp.maximum(qx, 0.0) ** 2 + jnp.maximum(qy, 0.0) ** 2)
    distance = outside + jnp.minimum(jnp.maximum(qx, qy), 0.0)
    return _coverage(distance, mask, pixel)


def box_coverage_table(points, lo, hi, pixel) -> tuple[np.ndarray, np.ndarray]:
    """Table of the coverage of the pixels by boxes that do not move.

    A pixel is touched by few of the boxes. The table keeps, for every pixel,
    those boxes and their coverage, so that an image of the boxes that are
    present costs a few look-ups per pixel instead of a test against every
    box (:func:`table_coverage`). The table is computed once with NumPy,
    outside compiled code.

    Args:
        points: (..., 2) pixel centres, from :func:`pixel_centres`.
        lo: (M, 2) lower-left corners of the boxes.
        hi: (M, 2) upper-right corners of the boxes.
        pixel: Edge length of a pixel.

    Returns:
        ``(index, cover)``, both of shape (..., K): the K boxes that cover
        each pixel most and their coverage, as in :func:`box_coverage`. K is
        the largest number of boxes that touch one pixel; unused entries have
        the coverage 0.
    """
    points = np.asarray(points, dtype=float)
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    centre, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    q = np.abs(points[..., None, :] - centre) - half
    outside = np.sqrt(np.sum(np.maximum(q, 0.0) ** 2, axis=-1))
    distance = outside + np.minimum(np.max(q, axis=-1), 0.0)
    cover = np.clip(0.5 - distance / pixel, 0.0, 1.0)
    k = max(1, int(np.count_nonzero(cover, axis=-1).max()))
    index = np.argsort(-cover, axis=-1, kind="stable")[..., :k]
    return index.astype(np.int32), np.take_along_axis(cover, index, axis=-1)


def table_coverage(index, cover, mask) -> jax.Array:
    """Coverage of the pixels by the boxes of a table that are present.

    Args:
        index: (..., K) box indices, from :func:`box_coverage_table`.
        cover: (..., K) coverage of the pixel by these boxes.
        mask: (M,) booleans; boxes with a false entry are left out.

    Returns:
        Array with the leading shape of ``index``, values in [0, 1]; equal to
        :func:`box_coverage` with the same mask.
    """
    return _largest(jnp.where(mask[index], cover, 0.0))
