"""Animation: 100 pedestrians leave a room with two doors and a pillar.

Every pedestrian is an agent that walks to the nearest door. The pillar stands
in front of the right door. Pedestrians are coloured by the door they use:
blue for the left door, orange for the right door.

    python examples/gallery/evacuation_room.py --output evacuation_room.gif
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from particlegym import envs, viz
from particlegym import physics as pp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--agents", type=int, default=100)
    parser.add_argument("--steps", type=int, default=320, help="control steps of 0.1 s")
    parser.add_argument("--stride", type=int, default=2, help="control steps per frame")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--output", default="evacuation_room.gif")
    args = parser.parse_args()

    env = envs.Evacuation(
        room=(14.0, 10.0),
        exits=[pp.Opening("x-", (4.4, 5.6)), pp.Opening("x+", (4.4, 5.6))],
        obstacles=[pp.Spheres([[12.3, 5.0]], [0.6])],
        n_agents=args.agents,
        action="continuous",
        radius=(0.22, 0.28),
    )
    params = env.default_params
    centres = jnp.asarray(env.door_centres)
    targets = jnp.asarray(env.door_targets)

    def nearest_door(state):
        pos = state.particles.pos
        return jnp.argmin(jnp.sum((pos[:, None, :] - centres) ** 2, axis=-1), axis=1)

    @jax.jit
    def rollout(key):
        reset_key, step_key = jax.random.split(key)
        _, state = env.reset(reset_key, params)

        def body(s, k):
            # A continuous action is the desired direction; lengths above 1 are capped.
            s = env.step(k, s, targets[nearest_door(s)] - s.particles.pos, params)[1]
            return s, s

        final, states = jax.lax.scan(body, state, jax.random.split(step_key, args.steps))
        return state, final, states

    start, final, states = rollout(jax.random.key(args.seed))

    exit_time = np.asarray(final.exit_time)
    left = np.isfinite(exit_time)
    print(f"{left.sum()} of {args.agents} pedestrians left the room")
    if left.all():
        print(f"evacuation time {exit_time.max():.1f} s")

    # The door of a pedestrian is the one nearest to its last position.
    by_right_door = np.asarray(final.particles.pos)[:, 0] > 0.5 * env.size[0]
    colors = np.where(by_right_door[:, None], viz.ORANGE, viz.BLUE)
    last = int(np.ceil(exit_time[left].max() / (env.dt * env.substeps))) if left.any() else 0
    frames = [env.render(start, args.width, colors)]
    for i in range(args.stride - 1, min(args.steps, last + 5 * args.stride), args.stride):
        state = jax.tree.map(lambda x, i=i: x[i], states)
        frames.append(env.render(state, args.width, colors))
    viz.save_gif(args.output, frames, fps=20)
    print(f"{len(frames)} frames written to {args.output}")


if __name__ == "__main__":
    main()
