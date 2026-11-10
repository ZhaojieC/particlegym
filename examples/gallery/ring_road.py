# ParticleGym, https://github.com/ZhaojieC/particlegym (MIT license).
# Please cite it in published work: particlegym.cite() gives the references.
"""Animation: a phantom traffic jam, and one automated vehicle that dissolves it.

22 vehicles on a ring road of 230 m, as in the experiment of Sugiyama et al.
(2008). Both rings start from the same stop-and-go wave, and their human
drivers have the same random fluctuations. Left: the blue vehicle drives like
the others (the intelligent driver model), and the jam keeps travelling
backward around the ring. Right: the blue vehicle cruises gently at 90 % of
the speed of uniform flow, and the jam dissolves. The vehicles drive
counter-clockwise; human drivers turn red as they slow down below half the
speed of uniform flow. The bar under each ring shows the mean speed of all
vehicles; the black mark is the speed of uniform flow. The default seed is
the one with the median gain of the seeds 0 to 20.

    python examples/gallery/ring_road.py --output ring_road.gif
"""

import argparse

import jax
import numpy as np

from particlegym import envs, viz


def speed_bar(mean_speed: float, uniform: float, width: int) -> np.ndarray:
    """A strip with a bar for the mean speed and a mark at the speed of uniform flow."""
    canvas = viz.Canvas((0.0, 0.0), (1.0, 0.1), width)
    full = 1.25 * uniform
    canvas.rectangles([0.1, 0.035], [0.9, 0.065], viz.LIGHT_GREY)
    canvas.rectangles([0.1, 0.035], [0.1 + 0.8 * min(mean_speed / full, 1.0), 0.065], (90, 90, 90))
    mark = 0.1 + 0.8 * uniform / full
    canvas.rectangles([mark - 0.004, 0.02], [mark + 0.004, 0.08], viz.BLACK)
    return canvas.frame()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--time", type=float, default=300.0, help="simulated seconds")
    parser.add_argument("--frame-every", type=int, default=3, help="control steps per frame")
    parser.add_argument("--width", type=int, default=300, help="width of one ring in pixels")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output", default="ring_road.gif")
    args = parser.parse_args()

    env = envs.RingRoad()
    params = env.default_params
    uniform = float(env.uniform_speed(params))
    _, start = env.reset(jax.random.key(args.seed), params)
    step = jax.jit(env.step)
    policies = (jax.jit(env.idm_policy), jax.jit(env.smoothing_policy))
    states = [start, start]
    distance = [0.0, 0.0]
    steps = round(args.time / env.control_interval)
    keys = jax.random.split(jax.random.key(args.seed + 1), steps)
    gap = np.full((args.width + 2 * (args.width // 10), args.width // 30, 3), 255, np.uint8)
    frames = []
    for i in range(steps + 1):
        if i % args.frame_every == 0:
            panels = [
                np.concatenate(
                    [
                        env.render(s, args.width, params),
                        speed_bar(float(np.mean(s.speed)), uniform, args.width),
                    ]
                )
                for s in states
            ]
            frames.append(np.concatenate([panels[0], gap[: panels[0].shape[0]], panels[1]], 1))
        if i < steps:
            for side in (0, 1):
                action = policies[side](states[side], params)
                out = step(keys[i], states[side], action, params)
                states[side] = out[1]
                distance[side] += float(out[2][0])
    viz.save_gif(args.output, frames, fps=15)
    for name, d, s in zip(("human-like", "gently cruising"), distance, states, strict=True):
        print(
            f"{name} automated vehicle: {d:.0f} m per vehicle in {args.time:g} s, "
            f"final speed spread {float(np.std(s.speed)):.2f} m/s"
        )
    print(f"{len(frames)} frames written to {args.output}")


if __name__ == "__main__":
    main()
