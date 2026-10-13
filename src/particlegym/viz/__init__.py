"""Pictures and animations of 2D systems, and pixel images for observations."""

from particlegym.viz.canvas import (
    BLACK,
    BLUE,
    GREEN,
    GREY,
    LIGHT_GREY,
    ORANGE,
    RED,
    WHITE,
    Canvas,
    save_gif,
    save_mp4,
    upscale,
)
from particlegym.viz.pixels import box_coverage, disc_coverage, pixel_centres

__all__ = [
    "BLACK",
    "BLUE",
    "GREEN",
    "GREY",
    "LIGHT_GREY",
    "ORANGE",
    "RED",
    "WHITE",
    "Canvas",
    "box_coverage",
    "disc_coverage",
    "pixel_centres",
    "save_gif",
    "save_mp4",
    "upscale",
]
