# ParticleGym, https://github.com/ZhaojieC/particlegym (MIT license).
# Please cite it in published work: particlegym.cite() gives the references.
"""Figure: trajectories of the vehicles on the ring road, coloured by speed.

The same two runs as ``ring_road.py``: 22 vehicles on a ring of 230 m, from
the same stop-and-go wave with the same fluctuations of the human drivers.
Left, the automated vehicle drives like a human; the jam (dark band) travels
backward around the ring. Right, it cruises gently below the speed of
uniform flow (blue trajectory); the jam dissolves. Needs matplotlib.

    python examples/gallery/ring_road_spacetime.py --output ring_road_spacetime.png
"""

import argparse

import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from particlegym import envs


def run(env, params, policy, seed: int, steps: int):
    """Positions and speeds of all vehicles after every control step."""
    _, state = env.reset(jax.random.key(seed), params)
    keys = jax.random.split(jax.random.key(seed + 1), steps)

    def body(s, k):
        s = env.step(k, s, policy(s, params), params)[1]
        return s, (s.position, s.speed)

    _, (position, speed) = jax.jit(lambda s: jax.lax.scan(body, s, keys))(state)
    return np.asarray(position), np.asarray(speed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--time", type=float, default=300.0, help="simulated seconds")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output", default="ring_road_spacetime.png")
    args = parser.parse_args()

    env = envs.RingRoad()
    params = env.default_params
    steps = round(args.time / env.control_interval)
    t = env.control_interval * np.arange(1, steps + 1)
    runs = [
        run(env, params, env.idm_policy, args.seed, steps),
        run(env, params, env.smoothing_policy, args.seed, steps),
    ]
    top = 1.5 * float(env.uniform_speed(params))
    # Sequential: stopped vehicles dark red, fast ones light grey.
    colors = LinearSegmentedColormap.from_list("speed", ["#7a0c0c", "#e2584b", "#d9d9d9"])
    titles = ("Automated vehicle drives like a human", "Automated vehicle cruises gently (blue)")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True, constrained_layout=True)
    for ax, (position, speed), title in zip(axes, runs, titles, strict=True):
        times = np.broadcast_to(t[:, None], position.shape)
        points = ax.scatter(
            times.ravel(), position.ravel(), c=speed.ravel(), s=0.6, cmap=colors, vmin=0,
            vmax=top, linewidths=0, rasterized=True,
        )  # fmt: skip
        agent = env.agents[0]
        ax.scatter(t, position[:, agent], s=0.8, color="#1f5fd6", linewidths=0)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("time (s)")
        ax.set_xlim(0, args.time)
        ax.set_ylim(0, float(params.ring_length))
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("position on the ring (m)")
    bar = fig.colorbar(points, ax=axes, shrink=0.85, pad=0.01)
    bar.set_label("speed (m/s)")
    fig.savefig(args.output, dpi=160)
    print(f"written to {args.output}")


if __name__ == "__main__":
    main()
