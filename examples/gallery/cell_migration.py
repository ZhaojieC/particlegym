"""Animation: cells migrate together to a source of chemoattractant.

Leader cells (purple) move towards the source (red) and up the gradient of a
second chemoattractant that they secrete themselves (orange), which keeps the
group together. Cells repel each other on contact. Model after Zhang, Chai,
Sun and Lykotrafitis (arXiv:2007.03190); the cells follow the gradients, no
learning is involved.

    python examples/gallery/cell_migration.py --output cell_migration.gif
"""

import argparse

import jax

from particlegym import envs, viz


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--leaders", type=int, default=9)
    parser.add_argument("--followers", type=int, default=0)
    parser.add_argument("--weight", type=float, default=0.5, help="weight of the co-attraction")
    parser.add_argument("--steps", type=int, default=400, help="control steps of 0.1 time units")
    parser.add_argument("--stride", type=int, default=3, help="control steps per frame")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--output", default="cell_migration.gif")
    args = parser.parse_args()

    env = envs.CellMigration(n_leaders=args.leaders, n_followers=args.followers)
    params = env.default_params
    step = jax.jit(env.step)
    policy = jax.jit(lambda s: env.gradient_policy(s, args.weight))
    _, state = env.reset(jax.random.key(args.seed), params)
    frames = [env.render(state, args.width)]
    finished = None
    for i in range(args.steps):
        _, state, _, terminated, _, info = step(None, state, policy(state), params)
        if (i + 1) % args.stride == 0:
            frames.append(env.render(state, args.width))
        if (i + 1) % 50 == 0:
            print(
                f"step {i + 1}: mean distance of the leaders to the source "
                f"{float(info['distance']):.2f}, spread of the cells {float(info['spread']):.2f}"
            )
        if finished is None and bool(terminated[0]):
            finished = i + 1
            print(f"all leaders at the source after {finished} steps")
        if finished is not None and i + 1 >= finished + 30:
            break
    viz.save_gif(args.output, frames, fps=20, colors=128)
    print(f"{len(frames)} frames written to {args.output}")


if __name__ == "__main__":
    main()
