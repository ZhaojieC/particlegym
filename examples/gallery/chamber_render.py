"""Render the 3D evacuation example with OVITO (Tachyon renderer).

Reads the trajectory `evacuation_env_3d.xyz` and the wall meshes written by
`chamber_walls.py` from the scene directory. Agents are coloured by the exit
they use: orange for the ceiling hatch, blue for the door. The two walls
nearest to the camera are left out and the ceiling is drawn translucent, so
that the inside of the chamber is visible. Dark frames mark the two exits.

Run with a Python that has the `ovito` package; without a display:

    QT_QPA_PLATFORM=offscreen python examples/gallery/chamber_render.py scene [--still FRAME]

The frames are written to scene/frames/.
"""

import argparse
import pathlib

import numpy as np
from ovito.data import TriangleMesh
from ovito.io import import_file
from ovito.vis import TachyonRenderer, Viewport

DOOR_CENTRE = np.array([5.0, 10.0, 1.0])
HATCH_CENTRE = np.array([5.0, 5.0, 4.0])
ORANGE, BLUE = (1.0, 0.45, 0.0), (0.0, 0.3, 1.0)
WALL_GREY, FLOOR_GREY, FRAME_GREY = (0.93, 0.93, 0.93), (0.78, 0.78, 0.78), (0.2, 0.2, 0.2)
# Colour and transparency of each wall; the walls x- and y- face the camera and are not drawn.
WALLS = {
    "x+": (WALL_GREY, 0.0),
    "y+": (WALL_GREY, 0.0),
    "z-": (FLOOR_GREY, 0.0),
    "z+": (WALL_GREY, 0.8),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("scene", type=pathlib.Path, help="directory with the scene files")
    parser.add_argument("--still", type=int, default=None, help="render one frame to a PNG")
    parser.add_argument("--size", type=int, nargs=2, default=(800, 600))
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    scene = args.scene
    pipeline = import_file(str(scene / "evacuation_env_3d.xyz"), multiple_frames=True)
    n_frames = pipeline.source.num_frames

    # The exit of each agent is the one nearest to the last position at which it was inside.
    last = {}
    for frame in range(n_frames):
        data = pipeline.compute(frame)
        ids = data.particles["Particle Identifier"]
        for i, pos in zip(ids, data.particles.positions, strict=True):
            last[int(i)] = np.array(pos)
    colours = np.zeros((max(last) + 1, 3))
    for i, pos in last.items():
        by_hatch = np.linalg.norm(pos - HATCH_CENTRE) < np.linalg.norm(pos - DOOR_CENTRE)
        colours[i] = ORANGE if by_hatch else BLUE
    print(f"{n_frames} frames, {len(last)} agents")

    def colour_by_exit(frame, data):
        data.cell_.vis.enabled = False
        ids = np.asarray(data.particles["Particle Identifier"])
        data.particles_.create_property("Color", data=colours[ids])

    pipeline.modifiers.append(colour_by_exit)
    pipeline.add_to_scene()

    meshes = {f"walls_{face}.obj": style for face, style in WALLS.items()}
    meshes["frames.obj"] = (FRAME_GREY, 0.0)
    for filename, (colour, transparency) in meshes.items():
        walls = import_file(str(scene / filename))
        mesh = next(o for o in walls.compute().objects if isinstance(o, TriangleMesh))
        mesh.vis.color = colour
        mesh.vis.transparency = transparency
        mesh.vis.highlight_edges = False
        walls.add_to_scene()

    viewport = Viewport(type=Viewport.Type.Perspective)
    camera, target = np.array([-9.0, -15.0, 12.5]), np.array([5.0, 5.0, 1.6])
    viewport.camera_pos = tuple(camera)
    viewport.camera_dir = tuple(target - camera)
    viewport.fov = np.radians(27.0)
    renderer = TachyonRenderer(
        ambient_occlusion=True,
        ambient_occlusion_brightness=0.95,
        shadows=False,
        antialiasing_samples=12,
    )
    if args.still is not None:
        viewport.render_image(
            filename=str(scene / f"still{args.still:04d}.png"),
            size=tuple(args.size),
            frame=args.still,
            renderer=renderer,
        )
    else:
        (scene / "frames").mkdir(exist_ok=True)
        viewport.render_anim(
            filename=str(scene / "frames" / "frame.png"),
            size=tuple(args.size),
            fps=args.fps,
            renderer=renderer,
        )


if __name__ == "__main__":
    main()
