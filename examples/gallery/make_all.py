"""Regenerate the animations of the gallery.

Runs every script of the gallery that needs only ParticleGym and Pillow and
writes the GIF files to the output directory. The 3D animation needs OVITO and
is made by hand; see the README in this directory.

    python examples/gallery/make_all.py [--output docs/assets/gallery] [--skip-large]
"""

import argparse
import pathlib
import subprocess
import sys

SCRIPTS = [
    "evacuation_room",
    "crowd_10000",
    "counterflow",
    "brick_wall",
    "billiards",
    "herding",
    "microswimmer",
    "cell_migration",
    "dpd_demixing",
]
LARGE = {"crowd_10000"}  # several minutes on a CPU


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("docs/assets/gallery"))
    parser.add_argument("--skip-large", action="store_true", help="skip the long simulations")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    here = pathlib.Path(__file__).parent
    for name in SCRIPTS:
        if args.skip_large and name in LARGE:
            print(f"skipping {name}")
            continue
        print(f"--- {name}", flush=True)
        command = [sys.executable, here / f"{name}.py", "--output", args.output / f"{name}.gif"]
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
