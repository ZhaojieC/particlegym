"""Join numbered PNG frames, for example from OVITO, into an animated GIF.

python examples/gallery/frames_to_gif.py scene/frames evacuation_chamber_3d.gif --width 480
"""

import argparse
import pathlib

import numpy as np
from PIL import Image

from particlegym import viz


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("frames", type=pathlib.Path, help="directory with the PNG frames")
    parser.add_argument("output")
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--colors", type=int, default=64)
    args = parser.parse_args()

    frames = []
    for path in sorted(args.frames.glob("*.png")):
        with Image.open(path) as picture:
            height = round(args.width * picture.height / picture.width)
            small = picture.convert("RGB").resize((args.width, height), Image.Resampling.LANCZOS)
            frames.append(np.asarray(small))
    viz.save_gif(args.output, frames, fps=args.fps, colors=args.colors)
    print(f"{len(frames)} frames written to {args.output}")


if __name__ == "__main__":
    main()
