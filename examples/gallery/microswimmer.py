"""Animation: gyrotactic swimmers in a field of vortices.

Swimmers that always steer upwards (naive gyrotaxis) in a Taylor-Green flow,
for the parameters Phi = 0.3 and Psi = 1 of Colabrese et al. (2017): after a
start phase the vortices trap them, and they gain no more altitude. The
picture shows one period of the flow; the grey levels show the vorticity.

    python examples/gallery/microswimmer.py --output microswimmer.gif
"""

import argparse

import jax
import numpy as np

from particlegym import envs, viz


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--swimmers", type=int, default=400)
    parser.add_argument("--steps", type=int, default=450, help="control steps of 0.1 time units")
    parser.add_argument("--stride", type=int, default=4, help="control steps per frame")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=280)
    parser.add_argument("--output", default="microswimmer.gif")
    args = parser.parse_args()

    env = envs.Microswimmer(n_agents=args.swimmers)
    params = env.default_params
    step = jax.jit(env.step)
    key = jax.random.key(args.seed)
    _, state = env.reset(key, params)
    action = env.naive_policy()
    frames = [env.render(state, args.width, params, viz.RED)]
    before = np.zeros(args.swimmers)
    for i in range(args.steps):
        _, state, _, _, _, info = step(jax.random.fold_in(key, i), state, action, params)
        if (i + 1) % args.stride == 0:
            frames.append(env.render(state, args.width, params, viz.RED))
        if (i + 1) % 150 == 0:
            altitude = np.asarray(info["altitude"])
            rising = np.mean(altitude - before > 0.5 * env.period)
            print(
                f"time {0.1 * (i + 1):.0f}: mean altitude gain {altitude.mean():+.2f}; "
                f"{100 * rising:.0f} % of the swimmers rose by more than half a flow period "
                "in the last 15 time units"
            )
            before = altitude
    viz.save_gif(args.output, frames, fps=20, colors=32)
    print(f"{len(frames)} frames written to {args.output}")


if __name__ == "__main__":
    main()
