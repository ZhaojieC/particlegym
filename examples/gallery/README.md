# Gallery scripts

Every animation in the project README and on the gallery page of the
documentation is made by a script in this directory, from a simulation with a
fixed random seed.

| Animation | Made by | Needs |
|---|---|---|
| `evacuation_room.gif` | `evacuation_room.py` | Pillow |
| `crowd_10000.gif` | `crowd_10000.py` (about six minutes on 12 CPU cores) | Pillow |
| `counterflow.gif` | `counterflow.py` | Pillow |
| `brick_wall.gif` | `brick_wall.py` | Pillow |
| `billiards.gif` | `billiards.py` | Pillow |
| `herding.gif` | `herding.py` | Pillow |
| `microswimmer.gif` | `microswimmer.py` | Pillow |
| `cell_migration.gif` | `cell_migration.py` | Pillow |
| `dpd_demixing.gif` | `dpd_demixing.py` | Pillow |
| `evacuation_chamber_3d.gif` | the four steps below | Pillow, OVITO |
| `encircle_3d.gif` | the [drone-encircle](https://github.com/ZhaojieC/drone-encircle) repository | |

```bash
pip install -e ".[viz]"
python examples/gallery/make_all.py            # all animations that need only Pillow
```

The 3D chamber is rendered with the Python package of
[OVITO](https://www.ovito.org/) (`pip install ovito`):

```bash
mkdir scene
python examples/evacuation_env_3d.py --output scene/evacuation_env_3d.xyz
python examples/gallery/chamber_walls.py scene
QT_QPA_PLATFORM=offscreen python examples/gallery/chamber_render.py scene
python examples/gallery/frames_to_gif.py scene/frames docs/assets/gallery/evacuation_chamber_3d.gif
```
