"""Images of two-dimensional particle systems.

A :class:`Canvas` maps a rectangular region of the simulation domain to an
image and draws particles and geometry on it. A frame is a NumPy array of
shape (height, width, 3) and type ``uint8``; :func:`save_gif` and
:func:`save_mp4` write a list of frames as an animation.

Drawing runs on the host, outside compiled code, and uses Pillow, an optional
dependency: ``pip install "particlegym[viz]"``.
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess

import numpy as np

from particlegym.physics.geometry import Boxes, Plates, ShapeSet, Spheres

WHITE = (255, 255, 255)
BLACK = (25, 25, 25)
GREY = (150, 150, 150)
LIGHT_GREY = (225, 225, 225)
RED = (230, 30, 30)
BLUE = (20, 90, 235)
GREEN = (0, 160, 70)
ORANGE = (255, 130, 0)


def _pillow():
    try:
        from PIL import Image, ImageDraw
    except ImportError as error:
        raise ImportError('drawing needs Pillow: pip install "particlegym[viz]"') from error
    return Image, ImageDraw


class Canvas:
    """An image of the region ``[lo, hi]`` of a 2D simulation domain.

    The y axis points up, as in the simulation. Lengths are given in
    simulation units. Shapes are drawn at ``supersample`` times the output
    resolution and averaged down, which smooths their edges.

    Args:
        lo: Lower-left corner of the region.
        hi: Upper-right corner of the region.
        width: Image width in pixels. The height follows from the aspect
            ratio of the region.
        background: Background colour, an (r, g, b) triple of values 0 to 255.
        supersample: Linear oversampling factor; 1 gives hard pixel edges.
    """

    def __init__(self, lo, hi, width: int = 480, background=WHITE, supersample: int = 3) -> None:
        image, image_draw = _pillow()
        self.lo = np.asarray(lo, dtype=float)
        self.hi = np.asarray(hi, dtype=float)
        if self.lo.shape != (2,) or self.hi.shape != (2,) or np.any(self.hi <= self.lo):
            raise ValueError("lo and hi must be 2D corners with hi > lo")
        if width < 1 or supersample < 1:
            raise ValueError("width and supersample must be at least 1")
        extent = self.hi - self.lo
        self.width = int(width)
        self.height = max(1, round(width * extent[1] / extent[0]))
        self.supersample = int(supersample)
        # Pixels of the oversampled image per unit length.
        self.scale = self.width * self.supersample / extent[0]
        size = (self.width * self.supersample, self.height * self.supersample)
        self._image = image.new("RGB", size, tuple(background))
        self._draw = image_draw.Draw(self._image)

    def _to_pixels(self, points) -> np.ndarray:
        """Coordinates in the oversampled image; pixel k covers the interval [k, k + 1]."""
        points = np.asarray(points, dtype=float)
        x = (points[..., 0] - self.lo[0]) * self.scale
        y = (self.hi[1] - points[..., 1]) * self.scale
        return np.stack([x, y], axis=-1)

    def discs(self, pos, radius, color, mask=None) -> None:
        """Draw filled discs.

        Args:
            pos: (N, 2) centres.
            radius: Radii, scalar or (N,).
            color: One (r, g, b) colour, or (N, 3) colours.
            mask: (N,) booleans; discs with a false entry are not drawn (pass
                the ``alive`` flags of a particle state).
        """
        pos = np.asarray(pos, dtype=float).reshape(-1, 2)
        n = len(pos)
        centres = self._to_pixels(pos)
        radii = np.broadcast_to(np.asarray(radius, dtype=float), (n,)) * self.scale
        colors = np.broadcast_to(np.asarray(color, dtype=np.uint8), (n, 3))
        mask = np.ones(n, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        # Pillow places ellipses on whole pixels: the first and last pixel that
        # the disc covers along each axis.
        first = np.rint(centres - radii[:, None]).astype(int)
        last = np.maximum(np.rint(centres + radii[:, None]).astype(int) - 1, first)
        for i in np.flatnonzero(mask):
            self._draw.ellipse((*first[i], *last[i]), fill=tuple(colors[i].tolist()))

    def rectangles(self, lo, hi, color, mask=None) -> None:
        """Draw filled axis-aligned rectangles with corners ``lo`` and ``hi``, both (M, 2)."""
        lo = np.asarray(lo, dtype=float).reshape(-1, 2)
        hi = np.asarray(hi, dtype=float).reshape(-1, 2)
        a, b = self._to_pixels(lo), self._to_pixels(hi)
        colors = np.broadcast_to(np.asarray(color, dtype=np.uint8), (len(lo), 3))
        mask = np.ones(len(lo), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        # A pixel is filled if its centre is inside the rectangle. The image y
        # axis points down, so the upper corner comes first.
        first = np.ceil(np.stack([a[:, 0], b[:, 1]], axis=1) - 0.5).astype(int)
        last = np.ceil(np.stack([b[:, 0], a[:, 1]], axis=1) - 0.5).astype(int) - 1
        for i in np.flatnonzero(mask & np.all(last >= first, axis=1)):
            self._draw.rectangle((*first[i], *last[i]), fill=tuple(colors[i].tolist()))

    def segments(self, start, end, color, width: float) -> None:
        """Draw line segments with round ends from ``start`` to ``end``, both (M, 2)."""
        start = np.asarray(start, dtype=float).reshape(-1, 2)
        end = np.asarray(end, dtype=float).reshape(-1, 2)
        # Lines are drawn through pixel centres, which are at k + 0.5.
        a, b = self._to_pixels(start) - 0.5, self._to_pixels(end) - 0.5
        pixels = max(1, round(width * self.scale))
        color = tuple(int(c) for c in color)
        for p, q in zip(a, b, strict=True):
            self._draw.line([tuple(p), tuple(q)], fill=color, width=pixels)
        if pixels > 2:
            self.discs(np.concatenate([start, end]), 0.5 * pixels / self.scale, color)

    def polyline(self, points, color, width: float) -> None:
        """Draw a connected line through ``points`` (K, 2), for example a particle path."""
        points = self._to_pixels(np.asarray(points, dtype=float).reshape(-1, 2)) - 0.5
        if len(points) < 2:
            return
        pixels = max(1, round(width * self.scale))
        self._draw.line(
            [tuple(p) for p in points], fill=tuple(int(c) for c in color), width=pixels,
            joint="curve",
        )  # fmt: skip

    def shapes(self, shapes: ShapeSet, color=BLACK, thickness: float = 0.1) -> None:
        """Draw a 2D shape set.

        Plates are thin in the simulation; they are drawn as lines of the
        given ``thickness``. Spheres (discs in 2D) and boxes are filled.
        """
        if shapes.dim != 2:
            raise ValueError("only 2D shapes can be drawn; write 3D systems with write_extxyz")
        if isinstance(shapes, Plates):
            origins = np.asarray(shapes.origins)
            self.segments(origins, origins + np.asarray(shapes.edges)[:, 0], color, thickness)
        elif isinstance(shapes, Spheres):
            self.discs(np.asarray(shapes.centers), np.asarray(shapes.radii), color)
        elif isinstance(shapes, Boxes):
            self.rectangles(np.asarray(shapes.lo), np.asarray(shapes.hi), color)
        else:
            raise TypeError(f"cannot draw {type(shapes).__name__}")

    def frame(self) -> np.ndarray:
        """The image as an array of shape (height, width, 3) and type ``uint8``."""
        image, _ = _pillow()
        picture = self._image
        if self.supersample > 1:
            picture = picture.resize((self.width, self.height), image.Resampling.BOX)
        return np.asarray(picture)


def upscale(frame: np.ndarray, factor: int) -> np.ndarray:
    """Repeat every pixel ``factor`` times along both axes, which keeps pixels sharp."""
    return np.repeat(np.repeat(np.asarray(frame), factor, axis=0), factor, axis=1)


def save_gif(path, frames, fps: float = 20.0, colors: int = 64) -> None:
    """Write frames as an animated GIF that loops forever.

    All frames share one colour table, built from a sample of the frames.
    GIF stores frame durations in units of 10 ms, so the frame rate is
    rounded: 20 and 25 frames per second are exact, 30 becomes 33.

    Args:
        path: Output file.
        frames: Sequence of (height, width, 3) ``uint8`` arrays of equal shape.
        fps: Frames per second.
        colors: Size of the colour table, at most 256.
    """
    image, _ = _pillow()
    frames = [np.asarray(frame, dtype=np.uint8) for frame in frames]
    if not frames:
        raise ValueError("no frames to write")
    sample = frames[:: max(1, len(frames) // 8)]
    palette = image.fromarray(np.concatenate(sample, axis=1)).quantize(
        colors, method=image.Quantize.MEDIANCUT, dither=image.Dither.NONE
    )
    pictures = [
        image.fromarray(frame).quantize(palette=palette, dither=image.Dither.NONE)
        for frame in frames
    ]
    pictures[0].save(
        pathlib.Path(path),
        save_all=True,
        append_images=pictures[1:],
        duration=10 * max(1, round(100.0 / fps)),
        loop=0,
        optimize=False,
    )


def save_mp4(path, frames, fps: float = 30.0, quality: int = 20) -> None:
    """Write frames as an H.264 video with the ``ffmpeg`` program.

    Args:
        path: Output file.
        frames: Sequence of (height, width, 3) ``uint8`` arrays of equal shape.
        fps: Frames per second.
        quality: Constant rate factor of the encoder; lower is better and larger.

    Raises:
        RuntimeError: If ``ffmpeg`` is not installed or fails.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("save_mp4 needs the ffmpeg program on the search path")
    frames = [np.asarray(frame, dtype=np.uint8) for frame in frames]
    if not frames:
        raise ValueError("no frames to write")
    height, width = frames[0].shape[:2]
    command = [
        ffmpeg, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", f"{fps}",
        "-i", "-",
        # H.264 with this pixel format needs even image sizes.
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2:color=white",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(quality),
        str(path),
    ]  # fmt: skip
    data = b"".join(frame.tobytes() for frame in frames)
    result = subprocess.run(command, input=data, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.decode(errors='replace').strip()}")
