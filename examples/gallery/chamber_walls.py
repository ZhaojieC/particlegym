"""Write the walls of the 3D chamber of `examples/evacuation_env_3d.py` as OBJ meshes.

One file per face of the room (walls_x-.obj, ..., walls_z+.obj), so that the
renderer can hide or fade individual walls, and one file with a frame around
every exit (frames.obj), which makes the exits easy to see. The meshes are
read by `chamber_render.py`.

    python examples/gallery/chamber_walls.py scene
"""

import argparse
import pathlib

import numpy as np

from particlegym import physics as pp

ROOM = (10.0, 10.0, 4.0)
EXITS = [
    pp.Opening("y+", ((4.5, 5.5), (0.0, 2.0))),
    pp.Opening("z+", ((4.0, 6.0), (4.0, 6.0))),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("scene", type=pathlib.Path, help="directory for the mesh files")
    args = parser.parse_args()
    args.scene.mkdir(parents=True, exist_ok=True)

    plates = pp.box_walls((0.0, 0.0, 0.0), ROOM, EXITS)
    origins, edges = np.asarray(plates.origins), np.asarray(plates.edges)
    faces: dict[str, list[np.ndarray]] = {}
    for origin, (e1, e2) in zip(origins, edges, strict=True):
        axis = int(np.argmax(np.abs(np.cross(e1, e2))))
        name = "xyz"[axis] + ("+" if origin[axis] > 0.5 * ROOM[axis] else "-")
        corners = np.array([origin, origin + e1, origin + e1 + e2, origin + e2])
        faces.setdefault(name, []).append(corners)
    for name, rectangles in sorted(faces.items()):
        with open(args.scene / f"walls_{name}.obj", "w") as f:
            for corners in rectangles:
                for x, y, z in corners:
                    f.write(f"v {x:.6g} {y:.6g} {z:.6g}\n")
            for i in range(len(rectangles)):
                a = 4 * i + 1
                f.write(f"f {a} {a + 1} {a + 2}\nf {a} {a + 2} {a + 3}\n")
        print(f"walls_{name}.obj: {len(rectangles)} rectangles")
    write_frames(args.scene / "frames.obj")


def write_frames(filename: pathlib.Path, width: float = 0.12, lift: float = 0.02) -> None:
    """Write a flat frame of the given width around every exit, just inside the wall."""
    quads = []
    for opening in EXITS:
        axis = "xyz".index(opening.face[0])
        outward = 1.0 if opening.face[1] == "+" else -1.0
        level = (ROOM[axis] if outward > 0 else 0.0) - outward * lift
        u, v = (a for a in range(3) if a != axis)
        (u0, u1), (v0, v1) = opening.span
        # Four bars: two along u (including the corners) and two along v.
        bars = [
            (u0 - width, u1 + width, v0 - width, v0),
            (u0 - width, u1 + width, v1, v1 + width),
            (u0 - width, u0, v0, v1),
            (u1, u1 + width, v0, v1),
        ]
        for a0, a1, b0, b1 in bars:
            # Keep the frame inside the wall, for example at the foot of a door.
            a0, a1 = max(a0, 0.0), min(a1, ROOM[u])
            b0, b1 = max(b0, 0.0), min(b1, ROOM[v])
            if a1 <= a0 or b1 <= b0:
                continue
            quad = np.full((4, 3), level)
            quad[:, u] = [a0, a1, a1, a0]
            quad[:, v] = [b0, b0, b1, b1]
            quads.append(quad)
    with open(filename, "w") as f:
        for quad in quads:
            for x, y, z in quad:
                f.write(f"v {x:.6g} {y:.6g} {z:.6g}\n")
        for i in range(len(quads)):
            a = 4 * i + 1
            f.write(f"f {a} {a + 1} {a + 2}\nf {a} {a + 2} {a + 3}\n")
    print(f"{filename.name}: {len(quads)} bars")


if __name__ == "__main__":
    main()
