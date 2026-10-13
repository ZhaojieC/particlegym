"""Animation: a simple player clears a billiards table.

The first shot breaks the rack. After that the cue ball is aimed at the
easiest straight shot. The balls slide with friction and collide with a
coefficient of restitution of 0.95 (cushions: 0.8).

    python examples/gallery/billiards.py --output billiards.gif
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from particlegym import envs, viz


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shots", type=int, default=7)
    parser.add_argument("--interval", type=float, default=0.1, help="seconds per frame")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--output", default="billiards.gif")
    args = parser.parse_args()

    env = envs.Billiards()
    params = env.default_params
    step = jax.jit(env.step)
    path = jax.jit(lambda s, a: env.trajectory(s, a, params, 120, args.interval))

    _, state = env.reset(jax.random.key(args.seed), params)
    frames = [env.render(state, args.width)] * 5
    total = 0
    for shot in range(args.shots):
        if shot == 0:
            aim = jnp.asarray(env.foot_spot) - state.balls.pos[0]
            action = jnp.stack([jnp.arctan2(aim[1], aim[0]) / jnp.pi, 1.0])[None, :]
        else:
            action = env.aim_policy(state, strength=0.2)
        balls = path(state, action)
        _, state, _, terminated, _, info = step(None, state, action, params)
        total += int(info["pocketed"])
        print(
            f"shot {shot + 1}: {int(info['pocketed'])} pocketed, "
            f"{'scratch, ' if bool(info['scratch']) else ''}{float(info['time']):.1f} s"
        )
        shown = min(120, int(np.ceil(float(info["time"]) / args.interval)) + 1)
        for i in range(shown):
            frames.append(env.render(jax.tree.map(lambda x, i=i: x[i], balls), args.width))
        frames += [env.render(state, args.width)] * 4
        if bool(terminated[0]):
            break
    print(f"{total} of {env.n_balls} balls pocketed in {shot + 1} shots")
    viz.save_gif(args.output, frames, fps=20, colors=48)
    print(f"{len(frames)} frames written to {args.output}")


if __name__ == "__main__":
    main()
