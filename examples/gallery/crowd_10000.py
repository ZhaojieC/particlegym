"""Animation: 10,000 pedestrians leave a hall of 120 m x 90 m through ten doors.

Every pedestrian walks to the nearest door (the crowd policy
``"nearest_exit"`` of the evacuation environment) at a desired speed of
1.3 m/s, and is coloured by the door that is nearest at the start.

The run takes about six minutes on 12 CPU cores. With the default desired
speed of 2 m/s, pedestrians are pressed so hard against the door posts that
the explicit sliding friction becomes unstable at the default time step of
5 ms; the script stops if that happens.

    python examples/gallery/crowd_10000.py --output crowd_10000.gif
"""

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

from particlegym import envs, viz
from particlegym import physics as pp

HALL = (120.0, 90.0)
DOOR = 6.0  # door width (m)
COLORS = np.array([viz.BLUE, viz.ORANGE, viz.GREEN, viz.RED])


def doors() -> list[pp.Opening]:
    """Three doors in each long wall and two in each short wall."""
    along_x = [(x - DOOR / 2, x + DOOR / 2) for x in (20.0, 60.0, 100.0)]
    along_y = [(y - DOOR / 2, y + DOOR / 2) for y in (22.5, 67.5)]
    return [pp.Opening(face, span) for face in ("y-", "y+") for span in along_x] + [
        pp.Opening(face, span) for face in ("x-", "x+") for span in along_y
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pedestrians", type=int, default=10_000)
    parser.add_argument("--frames", type=int, default=150)
    parser.add_argument("--stride", type=int, default=3, help="control steps of 0.1 s per frame")
    parser.add_argument("--speed", type=float, default=1.3, help="desired speed (m/s)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--output", default="crowd_10000.gif")
    args = parser.parse_args()

    env = envs.Evacuation(room=HALL, exits=doors(), n_agents=0, n_crowd=args.pedestrians)
    params = env.default_params._replace(desired_speed=args.speed)
    no_action = jnp.zeros((0,), jnp.int32)

    @jax.jit
    def rollout(key):
        reset_key, step_key = jax.random.split(key)
        _, state = env.reset(reset_key, params)

        def frame(s, key):
            def step(carry, k):
                s, overflow = carry
                s, info = env.step(k, s, no_action, params)[1::4]
                return (s, overflow | info["overflow"]), None

            (s, overflow), _ = jax.lax.scan(
                step, (s, jnp.zeros((), bool)), jax.random.split(key, args.stride)
            )
            speed = jnp.max(jnp.linalg.norm(s.particles.vel, axis=1))
            return s, (s.particles.pos, s.particles.alive, overflow, speed)

        _, frames = jax.lax.scan(frame, state, jax.random.split(step_key, args.frames))
        return state, frames

    start_time = time.perf_counter()
    start, (pos, alive, overflow, speed) = rollout(jax.random.key(args.seed))
    pos, alive = np.asarray(pos), np.asarray(alive)
    elapsed = time.perf_counter() - start_time
    print(f"largest speed of a pedestrian: {float(np.max(speed)):.1f} m/s")
    if bool(np.any(overflow)) or float(np.max(speed)) > 10 * args.speed:
        raise RuntimeError("the simulation became unstable; lower --speed")
    seconds = args.frames * args.stride * env.dt * env.substeps
    print(f"simulated {seconds:.0f} s in {elapsed:.0f} s, including compilation")
    print(f"{args.pedestrians - alive[-1].sum()} of {args.pedestrians} pedestrians left the hall")

    first = np.asarray(start.particles.pos)
    nearest = np.argmin(np.linalg.norm(first[:, None] - env.door_centres[None], axis=-1), axis=1)
    colors = COLORS[nearest % len(COLORS)]
    radius = np.asarray(start.particles.radius)
    walls = env.force_field.walls[0].shapes

    def draw(pos, alive):
        canvas = viz.Canvas((-1.0, -1.0), (HALL[0] + 1.0, HALL[1] + 1.0), args.width)
        canvas.shapes(walls, viz.BLACK, thickness=0.6)
        canvas.discs(pos, radius, colors, alive)
        return canvas.frame()

    frames = [draw(first, np.ones(len(first), bool))]
    frames += [draw(p, a) for p, a in zip(pos, alive, strict=True)]
    viz.save_gif(args.output, frames, fps=20, colors=32)
    print(f"{len(frames)} frames written to {args.output}")


if __name__ == "__main__":
    main()
