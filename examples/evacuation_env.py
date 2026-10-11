"""Agents leave a room through the nearest door, using the evacuation environment.

The room is the four-exit room of Zhang, Chai & Lykotrafitis (2021). Every
agent picks, among the eight action directions, the one closest to the
direction of the nearest door. Prints the evacuation time and writes a
trajectory that OVITO can open.

    python examples/evacuation_env.py --agents 80
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
    parser.add_argument("--agents", type=int, default=80)
    parser.add_argument("--steps", type=int, default=600, help="control steps of 0.1 s")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="evacuation_env.xyz")
    args = parser.parse_args()

    doors = [pp.Opening(face, (4.5, 5.5)) for face in ("y+", "y-", "x-", "x+")]
    env = envs.Evacuation(room=(10.0, 10.0), exits=doors, n_agents=args.agents)
    params = env.default_params
    directions = jnp.asarray(env.directions)
    centres = jnp.asarray(env.door_centres)
    targets = jnp.asarray(env.door_targets)

    def nearest_door_policy(state):
        pos = state.particles.pos[: env.n_agents]
        nearest = jnp.argmin(jnp.sum((pos[:, None, :] - centres) ** 2, axis=-1), axis=1)
        heading = targets[nearest] - pos
        return jnp.argmax(heading @ directions.T, axis=1)

    @jax.jit
    def rollout(key):
        reset_key, step_key = jax.random.split(key)
        _, state = env.reset(reset_key, params)

        def body(s, k):
            s = env.step(k, s, nearest_door_policy(s), params)[1]
            return s, s.particles

        return jax.lax.scan(body, state, jax.random.split(step_key, args.steps))

    final, frames = rollout(jax.random.key(args.seed))

    exit_time = np.asarray(final.exit_time)
    left = np.isfinite(exit_time)
    print(f"{left.sum()} of {args.agents} agents left the room")
    if left.all():
        print(f"evacuation time {exit_time.max():.1f} s")
    open(args.output, "w").close()
    interval = env.dt * env.substeps
    for i in range(0, args.steps, 5):
        frame = jax.tree.map(lambda x, i=i: x[i], frames)
        write_extxyz(args.output, frame, env.box, time=(i + 1) * interval)
        if not bool(frame.alive.any()):
            break
    print(f"trajectory written to {args.output}")


if __name__ == "__main__":
    main()
