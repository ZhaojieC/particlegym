"""Animation: two groups of pedestrians form lanes in a corridor.

80 pedestrians walk in a corridor of 30 m x 6 m that is periodic along its
length: the red ones to the right, the blue ones to the left. Every
pedestrian wants to walk straight on. Lanes form through the interactions.

    python examples/gallery/counterflow.py --output counterflow.gif
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from particlegym import envs, viz


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pedestrians", type=int, default=80)
    parser.add_argument("--steps", type=int, default=900, help="control steps of 0.1 s")
    parser.add_argument("--stride", type=int, default=6, help="control steps per frame")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--output", default="counterflow.gif")
    args = parser.parse_args()

    env = envs.CounterFlow(length=30.0, width=6.0, n_agents=0, n_crowd=args.pedestrians)
    params = env.default_params

    @jax.jit
    def rollout(key):
        reset_key, step_key = jax.random.split(key)
        _, state = env.reset(reset_key, params)

        def body(s, k):
            _, s, _, _, _, info = env.step(k, s, jnp.zeros((0, 2)), params)
            return s, (s, info["lane_order"], info["speed"])

        _, out = jax.lax.scan(body, state, jax.random.split(step_key, args.steps))
        return state, out

    start, (states, order, speed) = rollout(jax.random.key(args.seed))
    order, speed = np.asarray(order), np.asarray(speed)
    for t in (0, 99, 299, 599, args.steps - 1):
        print(
            f"t = {0.1 * (t + 1):5.1f} s: lane order {order[t]:.2f}, mean speed {speed[t]:.2f} m/s"
        )

    frames = [env.render(start, args.width)]
    for i in range(args.stride - 1, args.steps, args.stride):
        frames.append(env.render(jax.tree.map(lambda x, i=i: x[i], states), args.width))
    viz.save_gif(args.output, frames, fps=20, colors=32)
    print(f"{len(frames)} frames written to {args.output}")


if __name__ == "__main__":
    main()
