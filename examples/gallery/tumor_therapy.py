# ParticleGym, https://github.com/ZhaojieC/particlegym (MIT license).
# Please cite it in published work: particlegym.cite() gives the references.
"""Animation: a tumour under the maximum tolerated dose and under two adaptive therapies.

The agent-based tumour of the tumour-therapy environment: 200 cells, most of
them sensitive to the drug (green), a few resistant (red). All three tumours
start from the same cells. Left: treatment every day at the maximum
tolerated dose. Middle: the rule of the adaptive abiraterone trial, which
pauses the treatment when the tumour has halved and resumes it when the
tumour is back at its initial size. Right: containment, which treats
whenever the tumour has grown back to its initial size. The bar on top
shows the size of the tumour, with marks at the initial size and at
progression (1.2 times it); it is purple on days of treatment. Each panel
stops at progression.

    python examples/gallery/tumor_therapy.py --output tumor_therapy.gif
"""

import argparse

import jax
import numpy as np

from particlegym import envs, viz


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--days", type=int, default=150, help="longest simulated time")
    parser.add_argument("--width", type=int, default=280, help="width of one panel")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--output", default="tumor_therapy.gif")
    args = parser.parse_args()

    env = envs.TumorTherapy()
    params = env.default_params
    _, start = env.reset(jax.random.key(args.seed), params)
    step = jax.jit(env.step)
    policies = (
        jax.jit(env.mtd_policy),
        jax.jit(env.adaptive_policy),
        jax.jit(lambda s, p: env.adaptive_policy(s, p, stop=1.0, restart=1.0)),
    )
    states, ends = [start] * 3, [None] * 3
    keys = jax.random.split(jax.random.key(args.seed + 1), args.days)
    frames = []
    for day in range(args.days + 1):
        panels = [env.render(s, args.width, params) for s in states]
        gap = np.full((panels[0].shape[0], args.width // 20, 3), 255, np.uint8)
        frames.append(np.concatenate([panels[0], gap, panels[1], gap, panels[2]], axis=1))
        if all(end is not None for end in ends) or day == args.days:
            break
        for side in range(3):
            if ends[side] is None:
                action = policies[side](states[side], params)
                states[side] = step(keys[day], states[side], action, params)[1]
                if bool(states[side].progressed):
                    ends[side] = day + 1
    viz.save_gif(args.output, frames + frames[-1:] * 10, fps=8)
    names = ("maximum tolerated dose", "adaptive therapy", "containment")
    for name, end in zip(names, ends, strict=True):
        print(f"{name}: progression after {end} days")
    print(f"{len(frames)} frames written to {args.output}")


if __name__ == "__main__":
    main()
