"""Agents leave a 3D chamber through the nearest exit, using the evacuation environment.

The chamber measures 10 m x 10 m x 4 m and has a door in a side wall and a
hatch in the ceiling. The agents move freely in all three directions (there
is no gravity). Every agent picks, among the 26 action directions, the one
closest to the direction of the nearest exit. Prints the evacuation time and
writes a trajectory that OVITO can open.

    python examples/evacuation_env_3d.py --agents 60
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from particlegym import envs
from particlegym import physics as pp
from particlegym.io import write_extxyz


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--agents", type=int, default=60)
    parser.add_argument("--steps", type=int, default=300, help="control steps of 0.1 s")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="evacuation_env_3d.xyz")
    args = parser.parse_args()

    exits = [
        pp.Opening("y+", ((4.5, 5.5), (0.0, 2.0))),  # a door 1 m wide and 2 m high
        pp.Opening("z+", ((4.0, 6.0), (4.0, 6.0))),  # a 2 m x 2 m hatch in the ceiling
    ]
    env = envs.Evacuation(room=(10.0, 10.0, 4.0), exits=exits, n_agents=args.agents)
    params = env.default_params
    directions = jnp.asarray(env.directions)
    centres = jnp.asarray(env.door_centres)
    targets = jnp.asarray(env.door_targets)

    def nearest_exit_policy(state):
        pos = state.particles.pos[: env.n_agents]
        nearest = jnp.argmin(jnp.sum((pos[:, None, :] - centres) ** 2, axis=-1), axis=1)
        heading = targets[nearest] - pos
        return jnp.argmax(heading @ directions.T, axis=1)

    @jax.jit
    def rollout(key):
        reset_key, step_key = jax.random.split(key)
        _, state = env.reset(reset_key, params)

        def body(s, k):
            s = env.step(k, s, nearest_exit_policy(s), params)[1]
            return s, s.particles

        return jax.lax.scan(body, state, jax.random.split(step_key, args.steps))

    final, frames = rollout(jax.random.key(args.seed))

    exit_time = np.asarray(final.exit_time)
    left = np.isfinite(exit_time)
    through_hatch = left & (np.asarray(final.particles.pos)[:, 2] > env.size[2])
    print(f"{left.sum()} of {args.agents} agents left the chamber")
    print(f"{through_hatch.sum()} through the hatch, {(left & ~through_hatch).sum()} by the door")
    if left.all():
        print(f"evacuation time {exit_time.max():.1f} s")
    open(args.output, "w").close()
    interval = env.dt * env.substeps
    for i in range(0, args.steps, 2):
        frame = jax.tree.map(lambda x, i=i: x[i], frames)
        write_extxyz(args.output, frame, env.box, time=(i + 1) * interval)
        if not bool(frame.alive.any()):
            break
    print(f"trajectory written to {args.output}")


if __name__ == "__main__":
    main()
