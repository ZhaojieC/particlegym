"""Animation: a scripted player knocks bricks out of a wall.

The ball moves by Newton's equations and bounces off the walls, the bricks
and the round paddle. The paddle is moved by a simple rule: go to where the
ball will come down.

    python examples/gallery/brick_wall.py --output brick_wall.gif
"""

import argparse

import jax
import numpy as np

from particlegym import envs, viz


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=1000, help="control steps of 0.02 s")
    parser.add_argument("--stride", type=int, default=5, help="control steps per frame")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--output", default="brick_wall.gif")
    args = parser.parse_args()

    env = envs.BrickWall()
    params = env.default_params

    @jax.jit
    def rollout(key):
        reset_key, step_key = jax.random.split(key)
        _, state = env.reset(reset_key, params)

        def body(s, k):
            s = env.step(k, s, env.scripted_policy(s, params), params)[1]
            return s, s

        _, states = jax.lax.scan(body, state, jax.random.split(step_key, args.steps))
        return state, states

    start, states = rollout(jax.random.key(args.seed))
    score, lives = np.asarray(states.score), np.asarray(states.lives)
    print(f"{score[-1]} of {env.n_bricks} bricks removed in {args.steps} steps")
    print(f"{lives[-1]} of {params.lives} balls left")

    frames = [env.render(start, args.width)]
    for i in range(args.stride - 1, args.steps, args.stride):
        frames.append(env.render(jax.tree.map(lambda x, i=i: x[i], states), args.width))
    viz.save_gif(args.output, frames, fps=25, colors=32)
    print(f"{len(frames)} frames written to {args.output}")


if __name__ == "__main__":
    main()
