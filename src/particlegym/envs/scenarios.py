"""Rooms of published evacuation studies, as ready-made environments.

:func:`zhang2021` builds the two rooms of Zhang, Chai & Lykotrafitis (2021),
the study that ParticleGym grows out of. The layouts are taken from the code
that accompanies the paper: a 10 m x 10 m room with doors of 1 m, either
empty with a door in every wall or with three doors and two round obstacles.

The physics is ParticleGym's: real door openings, the social force model with
body force and sliding friction, and several physics steps per decision. The
differences from the code of the paper are listed in the documentation
(``docs/environments.md``, "Rooms of Zhang, Chai & Lykotrafitis (2021)").
"""

from __future__ import annotations

from typing import NamedTuple

from particlegym import physics as pp
from particlegym.envs.evacuation import Evacuation


class RoomLayout(NamedTuple):
    """The geometry of a room: the ``room``, ``exits`` and ``obstacles`` of an
    :class:`~particlegym.envs.Evacuation`."""

    room: tuple[float, ...]
    exits: tuple[pp.Opening, ...]
    obstacles: tuple[pp.ShapeSet, ...] = ()


# Positions in metres, with the room spanning [0, 10] m along x and y. In the
# code of the paper, exits and obstacles are given in units of the room size:
# exits at (0.5, 1), (0.5, 0), (0, 0.5) and (1, 0.5) for the empty room; exits
# at (0.7, 1), (0.5, 0) and (0, 0.7), and obstacles of diameter 2 m at
# (0.8, 0.8) and 3 m at (0.3, 0.5) for the room with obstacles.
ZHANG2021_LAYOUTS = {
    "four_exits": RoomLayout(
        room=(10.0, 10.0),
        exits=tuple(pp.Opening(face, (4.5, 5.5)) for face in ("y+", "y-", "x-", "x+")),
    ),
    "three_exits_two_obstacles": RoomLayout(
        room=(10.0, 10.0),
        exits=(
            pp.Opening("y+", (6.5, 7.5)),
            pp.Opening("y-", (4.5, 5.5)),
            pp.Opening("x-", (6.5, 7.5)),
        ),
        obstacles=(pp.Spheres([[8.0, 8.0], [3.0, 5.0]], [1.0, 1.5]),),
    ),
}


def zhang2021(layout: str = "four_exits", **kwargs) -> Evacuation:
    """A room of Zhang, Chai & Lykotrafitis (2021) as an evacuation environment.

    Both rooms are 10 m x 10 m with doors of 1 m:

    - ``"four_exits"``: an empty room with a door at the centre of every wall;
    - ``"three_exits_two_obstacles"``: doors in the top wall at x = 7 m, in the
      bottom wall at x = 5 m and in the left wall at y = 7 m, and round
      obstacles of diameter 2 m at (8, 8) m and 3 m at (3, 5) m.

    Pedestrians have the size and mass of the paper (diameter 0.5 m, 80 kg)
    unless ``radius`` or ``mass`` is given. The default episode parameters and
    decisions of :class:`Evacuation` are those of the paper: desired speed
    2 m/s, relaxation time 0.5 s, reward -0.1 per step and 0 at the exit, one
    of eight directions every 0.1 s. The paper trained one agent alone in the
    room, which is the default here, and tested the learned policy on 80
    pedestrians.

    Args:
        layout: ``"four_exits"`` or ``"three_exits_two_obstacles"``.
        **kwargs: Other arguments of :class:`Evacuation`, for example
            ``n_agents``, ``n_crowd`` or ``crowd_policy``. The layout fixes the
            room, the doors and the obstacles.

    Returns:
        The environment.
    """
    if layout not in ZHANG2021_LAYOUTS:
        raise ValueError(f"layout must be one of {list(ZHANG2021_LAYOUTS)}, not {layout!r}")
    fixed = sorted({"room", "exits", "obstacles"} & kwargs.keys())
    if fixed:
        raise TypeError(
            f"the layout sets {', '.join(fixed)}; build an Evacuation directly for other rooms"
        )
    kwargs.setdefault("radius", 0.25)
    kwargs.setdefault("mass", 80.0)
    return Evacuation(**ZHANG2021_LAYOUTS[layout]._asdict(), **kwargs)
