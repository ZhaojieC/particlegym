"""Pedestrians leave a room through a door (social force model).

Every pedestrian walks towards the door centre with the desired speed; once
past the door, straight on. Pedestrians 1 m beyond the door have left and are
switched off with the alive mask. Prints the evacuation time and the flow, and
writes a trajectory that OVITO can open.

Compare desired speeds: Helbing et al. (2000) found that above some speed,
clogging at the door makes the evacuation slower ("faster is slower").
Whether and where this happens depends on the crowd size, door width and
parameters.

    python examples/room_evacuation_2d.py --pedestrians 100 --speed 1.5
    python examples/room_evacuation_2d.py --pedestrians 100 --speed 4.0
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from particlegym import physics as pp
from particlegym.io import write_extxyz


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pedestrians", type=int, default=100)
    parser.add_argument("--speed", type=float, default=1.5, help="desired speed v0 (m/s)")
    parser.add_argument("--room", type=float, default=10.0, help="side of the square room (m)")
    parser.add_argument("--door", type=float, default=1.2, help="door width (m)")
    parser.add_argument("--max-time", type=float, default=300.0, help="seconds")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="evacuation.xyz")
    args = parser.parse_args()

    room, half_door = args.room, args.door / 2
    door = pp.Opening("x+", (room / 2 - half_door, room / 2 + half_door))
    walls = pp.box_walls((0.0, 0.0), (room, room), openings=[door])
    box = pp.Box((-1.0, -1.0), (room + 3.0, room + 1.0))
    field = pp.ForceField(
        box,
        pair=[pp.SocialForce(), pp.TangentialFriction()],
        walls=[pp.Walls(walls, pp.SocialForce(), friction=2.4e5)],
        neighbors=pp.BruteForce()
        if args.pedestrians <= 500
        else pp.CellList(cutoff=1.7, capacity=32),
    )

    rng = np.random.default_rng(args.seed)
    radius = rng.uniform(0.25, 0.30, args.pedestrians)
    pos = pp.random_positions(rng, (0.0, 0.0), (room, room), radius, gap=0.05)
    state = pp.create_state(pos, mass=80.0, radius=radius)
    field.check(state)
    state, _ = pp.init_forces(state, field)

    dt, tau, steps_per_frame = 0.005, 0.5, 100
    target = jnp.array([room, room / 2])

    @jax.jit
    def advance(state):
        def body(s, _):
            inside = s.pos[:, 0] < room
            direction = jnp.where(inside[:, None], target - s.pos, jnp.array([1.0, 0.0]))
            direction = direction / jnp.linalg.norm(direction, axis=1, keepdims=True)
            s, info = pp.langevin_baoab(
                s, field, dt, gamma=1.0 / tau, drift=args.speed * direction
            )
            s = s._replace(alive=s.alive & (s.pos[:, 0] < room + 1.0))
            return s, info.overflow

        state, overflow = jax.lax.scan(body, state, None, length=steps_per_frame)
        return state, overflow.any()

    open(args.output, "w").close()
    write_extxyz(args.output, state, box, time=0.0)
    time, remaining = 0.0, args.pedestrians
    while time < args.max_time:
        state, overflow = advance(state)
        if overflow:
            raise RuntimeError("a cell list cell overflowed; increase its capacity")
        time += steps_per_frame * dt
        write_extxyz(args.output, state, box, time=time)
        remaining = int(state.alive.sum())
        if remaining == 0:
            break
    left = args.pedestrians - remaining
    print(f"{left} of {args.pedestrians} pedestrians left in {time:.1f} s")
    if remaining == 0:
        flow = args.pedestrians / time
        print(f"mean flow {flow:.2f} persons/s, {flow / args.door:.2f} persons/(m s)")
    print(f"trajectory written to {args.output}")


if __name__ == "__main__":
    main()
