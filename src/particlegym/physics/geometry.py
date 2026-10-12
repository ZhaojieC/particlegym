"""Static geometry described by distance functions.

A shape set holds M shapes of one kind and returns the distances from
particle centres to the shape surfaces. :class:`Walls` turns these distances
into forces with any :class:`~particlegym.physics.potentials.PairPotential`,
evaluated with ``sigma`` equal to the particle radius, so wall forces are again
obtained by automatic differentiation.

Internally, points are stored coordinate-first, shape (d, N), and distances as
(M, N), so that the compiler vectorises over particles.

Geometry is not periodic: distances are computed from the (wrapped) particle
positions without minimum images.
"""

from __future__ import annotations

import abc
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from particlegym.physics.potentials import PairPotential

_AXES = "xyz"


def _norm(x: jax.Array) -> jax.Array:
    """Euclidean norm over the first (coordinate) axis, with a zero gradient at the origin."""
    s = jnp.sum(x * x, axis=0)
    positive = s > 0
    return jnp.where(positive, jnp.sqrt(jnp.where(positive, s, 1.0)), 0.0)


class ShapeSet(abc.ABC):
    """A set of shapes of one kind."""

    @property
    @abc.abstractmethod
    def dim(self) -> int:
        """Number of spatial dimensions."""

    @abc.abstractmethod
    def distance_t(self, points: jax.Array) -> jax.Array:
        """(M, N) distances from ``points`` (d, N), given coordinate-first, to the shapes."""

    def distance(self, pos: jax.Array) -> jax.Array:
        """(N, M) distances from the points ``pos`` (N, d) to the shape surfaces."""
        return self.distance_t(pos.T).T


@dataclasses.dataclass(frozen=True, eq=False)
class Plates(ShapeSet):
    """Thin flat walls: line segments in 2D, rectangles in 3D.

    Plate m is the set ``origins[m] + sum_k s_k edges[m, k]`` with every
    ``s_k`` in [0, 1]. Each plate has d - 1 mutually orthogonal edge vectors.
    Distances are unsigned, so plates repel particles from both sides, and the
    plate edges act like corners (door jambs, for example).

    Args:
        origins: (M, d) corner of each plate.
        edges: (M, d - 1, d) edge vectors of each plate.
    """

    origins: np.ndarray
    edges: np.ndarray

    def __post_init__(self) -> None:
        origins = np.asarray(self.origins, dtype=float)
        edges = np.asarray(self.edges, dtype=float)
        if origins.ndim != 2 or origins.shape[1] not in (2, 3):
            raise ValueError("origins must have shape (M, 2) or (M, 3)")
        m, d = origins.shape
        if edges.shape != (m, d - 1, d):
            raise ValueError(f"edges must have shape ({m}, {d - 1}, {d})")
        lengths = np.linalg.norm(edges, axis=-1)
        if np.any(lengths == 0):
            raise ValueError("plate edges must have non-zero length")
        if d == 3:
            cosine = np.abs(np.sum(edges[:, 0] * edges[:, 1], axis=-1)) / np.prod(lengths, axis=1)
            if np.any(cosine > 1e-9):
                raise ValueError("the two edges of a rectangular plate must be orthogonal")
        object.__setattr__(self, "origins", origins)
        object.__setattr__(self, "edges", edges)

    @classmethod
    def from_segments(cls, segments) -> Plates:
        """Build 2D plates from ``[(start, end), ...]`` point pairs."""
        segments = np.asarray(segments, dtype=float)
        if segments.ndim != 3 or segments.shape[1:] != (2, 2):
            raise ValueError("segments must have shape (M, 2, 2)")
        return cls(segments[:, 0], (segments[:, 1] - segments[:, 0])[:, None, :])

    @classmethod
    def from_rectangles(cls, rectangles) -> Plates:
        """Build 3D plates from ``[(corner, edge_1, edge_2), ...]`` triples."""
        rectangles = np.asarray(rectangles, dtype=float)
        if rectangles.ndim != 3 or rectangles.shape[1:] != (3, 3):
            raise ValueError("rectangles must have shape (M, 3, 3)")
        return cls(rectangles[:, 0], rectangles[:, 1:])

    @property
    def dim(self) -> int:
        return self.origins.shape[1]

    def distance_t(self, points):
        origins = jnp.asarray(self.origins.T[:, :, None], dtype=points.dtype)  # (d, M, 1)
        rel = points[:, None, :] - origins  # (d, M, N)
        # The edges are orthogonal, so the closest point of the plate has the
        # coordinate s along each edge, clipped to the plate. The edges are
        # handled one by one: with a reduction over a stacked edge axis, the
        # 3D kernel compiled to code that was 30 times slower in single
        # precision on CPU (JAX 0.11).
        offset = jnp.zeros_like(rel)
        for k in range(self.dim - 1):
            edge = jnp.asarray(self.edges[:, k].T[:, :, None], dtype=points.dtype)  # (d, M, 1)
            s = jnp.sum(rel * edge, axis=0) / jnp.sum(edge * edge, axis=0)  # (M, N)
            offset = offset + jnp.clip(s, 0.0, 1.0) * edge
        return _norm(rel - offset)


@dataclasses.dataclass(frozen=True, eq=False)
class Spheres(ShapeSet):
    """Spherical (3D) or circular (2D) obstacles; the distance is negative inside.

    Args:
        centers: (M, d) centres.
        radii: (M,) radii.
    """

    centers: np.ndarray
    radii: np.ndarray

    def __post_init__(self) -> None:
        centers = np.asarray(self.centers, dtype=float)
        radii = np.broadcast_to(np.asarray(self.radii, dtype=float), centers.shape[:1])
        if centers.ndim != 2 or centers.shape[1] not in (2, 3):
            raise ValueError("centers must have shape (M, 2) or (M, 3)")
        if np.any(radii <= 0):
            raise ValueError("radii must be positive")
        object.__setattr__(self, "centers", centers)
        object.__setattr__(self, "radii", radii)

    @property
    def dim(self) -> int:
        return self.centers.shape[1]

    def distance_t(self, points):
        centers = jnp.asarray(self.centers.T[:, :, None], dtype=points.dtype)  # (d, M, 1)
        radii = jnp.asarray(self.radii[:, None], dtype=points.dtype)
        return _norm(points[:, None, :] - centers) - radii


@dataclasses.dataclass(frozen=True, eq=False)
class Boxes(ShapeSet):
    """Solid axis-aligned box obstacles; the distance is negative inside.

    Args:
        lo: (M, d) lower corners.
        hi: (M, d) upper corners.
    """

    lo: np.ndarray
    hi: np.ndarray

    def __post_init__(self) -> None:
        lo = np.asarray(self.lo, dtype=float)
        hi = np.asarray(self.hi, dtype=float)
        if lo.ndim != 2 or lo.shape[1] not in (2, 3) or hi.shape != lo.shape:
            raise ValueError("lo and hi must have the same shape (M, 2) or (M, 3)")
        if np.any(hi <= lo):
            raise ValueError("hi must be larger than lo along every axis")
        object.__setattr__(self, "lo", lo)
        object.__setattr__(self, "hi", hi)

    @property
    def dim(self) -> int:
        return self.lo.shape[1]

    def distance_t(self, points):
        center = jnp.asarray(((self.lo + self.hi) / 2).T[:, :, None], dtype=points.dtype)
        half = jnp.asarray(((self.hi - self.lo) / 2).T[:, :, None], dtype=points.dtype)
        q = jnp.abs(points[:, None, :] - center) - half  # (d, M, N)
        outside = _norm(jnp.maximum(q, 0.0))
        inside = jnp.minimum(jnp.max(q, axis=0), 0.0)
        return outside + inside


@dataclasses.dataclass(frozen=True)
class Opening:
    """A rectangular hole, such as a door, in one face of :func:`box_walls`.

    Args:
        face: ``"x-"``, ``"x+"``, ``"y-"``, ``"y+"``, ``"z-"`` or ``"z+"``; for
            example ``"x+"`` is the face at ``hi[0]``.
        span: Extent of the hole within the face. In 2D a ``(start, end)`` pair
            along the other axis; in 3D one ``(start, end)`` pair for each of the
            two other axes, in increasing axis order.
    """

    face: str
    span: tuple


def box_walls(lo, hi, openings=()) -> Plates:
    """Walls on the faces of an axis-aligned room, with optional openings.

    In 2D any number of openings per face is supported; in 3D at most one per
    face.

    Args:
        lo: Lower corner of the room.
        hi: Upper corner of the room.
        openings: :class:`Opening` instances.

    Returns:
        The wall plates.
    """
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    d = lo.shape[0]
    if d not in (2, 3) or hi.shape != lo.shape or np.any(hi <= lo):
        raise ValueError("lo and hi must be 2D or 3D corners with hi > lo")
    holes_by_face: dict[str, list] = {}
    for opening in openings:
        holes_by_face.setdefault(opening.face, []).append(opening.span)

    origins, edges = [], []
    for axis in range(d):
        tangential = [a for a in range(d) if a != axis]
        face_extent = [(lo[a], hi[a]) for a in tangential]
        for sign, value in (("-", lo[axis]), ("+", hi[axis])):
            spans = holes_by_face.pop(f"{_AXES[axis]}{sign}", [])
            if d == 2:
                pieces = [[piece] for piece in _subtract_intervals(face_extent[0], spans)]
            else:
                pieces = _subtract_rectangle(face_extent, spans)
            for piece in pieces:
                origin = np.empty(d)
                origin[axis] = value
                edge = np.zeros((d - 1, d))
                for k, (a, (start, end)) in enumerate(zip(tangential, piece, strict=True)):
                    origin[a] = start
                    edge[k, a] = end - start
                origins.append(origin)
                edges.append(edge)
    if holes_by_face:
        raise ValueError(f"unknown faces {sorted(holes_by_face)}; use e.g. 'x-' or 'y+'")
    return Plates(np.asarray(origins), np.asarray(edges))


def _subtract_intervals(extent, spans) -> list[tuple[float, float]]:
    """Pieces of the interval ``extent`` left after removing the ``spans``."""
    low, high = extent
    spans = sorted((float(a), float(b)) for a, b in spans)
    pieces, cursor = [], low
    for a, b in spans:
        if not low <= a < b <= high or a < cursor:
            raise ValueError(f"opening ({a}, {b}) must lie inside ({low}, {high}) and not overlap")
        if a > cursor:
            pieces.append((cursor, a))
        cursor = b
    if cursor < high:
        pieces.append((cursor, high))
    return pieces


def _subtract_rectangle(extent, spans) -> list[list[tuple[float, float]]]:
    """Rectangles covering the face ``extent`` minus at most one rectangular hole."""
    if not spans:
        return [list(extent)]
    if len(spans) > 1:
        raise NotImplementedError("at most one opening per face is supported in 3D")
    (u0, u1), (v0, v1) = extent
    (a0, a1), (b0, b1) = ((float(s), float(e)) for s, e in spans[0])
    if not (u0 <= a0 < a1 <= u1 and v0 <= b0 < b1 <= v1):
        raise ValueError("the opening must lie inside the face")
    candidates = [
        [(u0, a0), (v0, v1)],
        [(a1, u1), (v0, v1)],
        [(a0, a1), (v0, b0)],
        [(a0, a1), (b1, v1)],
    ]
    return [rect for rect in candidates if all(end > start for start, end in rect)]


@dataclasses.dataclass(frozen=True, eq=False)
class Walls:
    """Interaction between particles and a set of shapes.

    The wall energy of particle i is ``sum_m potential.energy(d_im, r_i)``,
    where ``d_im`` is the distance from the particle centre to shape m and
    ``r_i`` the particle radius.

    Args:
        shapes: The geometry.
        potential: Contact potential between particles and the shapes.
        friction: Tangential wall friction coefficient kappa of Helbing, Farkas
            & Vicsek (2000), ``f = -kappa g(r_i - d) v_t``; 0 disables it.
    """

    shapes: ShapeSet
    potential: PairPotential
    friction: float = 0.0

    def energy(self, pos: jax.Array, radius: jax.Array, alive: jax.Array) -> jax.Array:
        """Total wall energy of the alive particles."""
        dist = self.shapes.distance_t(pos.T)  # (M, N)
        sigma = radius[None, :]
        reach = self.potential.cutoff(sigma)
        near = alive[None, :] & (dist < reach)
        # Evaluate far pairs at the cut-off, where the potential is zero and
        # smooth, so that their discarded gradients stay finite.
        far = jnp.where(reach > 0, reach, 1.0)
        energy = self.potential.energy(jnp.where(near, dist, far), sigma)
        return jnp.sum(jnp.where(near, energy, 0.0))

    def dissipative_force(
        self, pos: jax.Array, vel: jax.Array, radius: jax.Array, alive: jax.Array
    ) -> jax.Array:
        """Tangential friction force on particles touching the shapes."""
        points = pos.T
        d = points.shape[0]
        # The surface normal is the gradient of the distance. Each distance
        # depends on one particle only, so moving every particle along axis c
        # gives the c-th normal component for all particles at once.
        dist, distance_jvp = jax.linearize(self.shapes.distance_t, points)
        directions = jnp.eye(d, dtype=points.dtype)[:, :, None] * jnp.ones_like(points)
        normal = jax.vmap(distance_jvp)(directions)  # (d, M, N)
        overlap = jnp.where(alive[None, :], jnp.maximum(radius[None, :] - dist, 0.0), 0.0)
        v = vel.T[:, None, :]
        v_tangential = v - jnp.sum(v * normal, axis=0) * normal
        return -self.friction * jnp.sum(overlap * v_tangential, axis=1).T
