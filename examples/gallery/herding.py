"""Animation: a scripted shepherd drives a flock to the target.

The sheep (blue) follow the flock rules adapted from Strömbom et al. (2014).
The shepherd (red) collects stray sheep and then drives the flock to the
green target area.

    python examples/gallery/herding.py --output herding.gif
"""

import argparse

import jax

from particlegym import envs, viz


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sheep", type=int, default=40)
    parser.add_argument("--shepherds", type=int, default=1)
    parser.add_argument("--stride", type=int, default=4, help="control steps per frame")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=360)
    parser.add_argument("--output", default="herding.gif")
    args = parser.parse_args()

    env = envs.Herding(n_sheep=args.sheep, n_agents=args.shepherds)
    params = env.default_params
    step = jax.jit(env.step)
    policy = jax.jit(lambda k, s: env.shepherd_policy(k, s, params))

    key = jax.random.key(args.seed)
    _, state = env.reset(key, params)
    frames = [env.render(state, args.width)]
    for i in range(params.max_steps):
        action_key, step_key = jax.random.split(jax.random.fold_in(key, i))
        _, state, _, terminated, _, info = step(step_key, state, policy(action_key, state), params)
        if (i + 1) % args.stride == 0 or bool(terminated[0]):
            frames.append(env.render(state, args.width))
        if (i + 1) % 100 == 0:
            print(
                f"step {i + 1}: flock centre {float(info['distance']):.1f} m from the target, "
                f"spread {float(info['spread']):.1f} m"
            )
        if bool(terminated[0]):
            print(f"flock at the target after {i + 1} steps")
            break
    frames += [frames[-1]] * 10
    viz.save_gif(args.output, frames, fps=20, colors=32)
    print(f"{len(frames)} frames written to {args.output}")


if __name__ == "__main__":
    main()
