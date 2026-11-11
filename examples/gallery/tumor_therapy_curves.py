# ParticleGym, https://github.com/ZhaojieC/particlegym (MIT license).
# Please cite it in published work: particlegym.cite() gives the references.
"""Figure: sensitive and resistant cells under the maximum tolerated dose and adaptive therapies.

The same three runs as ``tumor_therapy.py``. The numbers of sensitive (green)
and resistant (red) cells are stacked; shaded days are days of treatment;
the dashed lines mark the initial size, half of it, and progression at 1.2
times it. Needs matplotlib.

    python examples/gallery/tumor_therapy_curves.py --output tumor_therapy_curves.png
"""

import argparse

import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from particlegym import envs


def run(env, params, policy, seed: int, days: int):
    """Daily numbers of sensitive and resistant cells, doses, and whether the tumour progressed."""
    _, state = env.reset(jax.random.key(seed), params)
    step = jax.jit(env.step)
    policy = jax.jit(policy)
    keys = jax.random.split(jax.random.key(seed + 1), days)
    sensitive, resistant, doses = [], [], []
    alive, kind = np.asarray(state.cells.alive), np.asarray(state.cells.kind)
    sensitive.append(int(np.sum(alive & (kind == 0))))
    resistant.append(int(np.sum(alive & (kind == 1))))
    for day in range(days):
        _, state, _, terminated, _, info = step(keys[day], state, policy(state, params), params)
        sensitive.append(int(info["sensitive"]))
        resistant.append(int(info["resistant"]))
        doses.append(float(state.dose))
        if bool(terminated[0]):
            break
    return np.array(sensitive), np.array(resistant), np.array(doses), bool(state.progressed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--days", type=int, default=150)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--output", default="tumor_therapy_curves.png")
    args = parser.parse_args()

    env = envs.TumorTherapy()
    params = env.default_params

    def containment(state, params):
        return env.adaptive_policy(state, params, stop=1.0, restart=1.0)

    runs = [
        ("Maximum tolerated dose", run(env, params, env.mtd_policy, args.seed, args.days)),
        ("Adaptive therapy", run(env, params, env.adaptive_policy, args.seed, args.days)),
        ("Containment", run(env, params, containment, args.seed, args.days)),
    ]
    initial = env.initial_cells
    longest = max(len(r[1][0]) for r in runs)
    colors = {
        "sensitive cells": "#3caa6e",
        "resistant cells": "#dc3c28",
        "day of treatment": "#e9e0f5",
    }
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True, layout="constrained")
    for ax, (title, (sensitive, resistant, doses, progressed)) in zip(axes, runs, strict=True):
        days = np.arange(len(sensitive))
        for day in np.flatnonzero(doses > 0):
            ax.axvspan(day, day + 1, color=colors["day of treatment"], lw=0)
        ax.stackplot(
            days, sensitive, resistant, step="post",
            colors=[colors["sensitive cells"], colors["resistant cells"]],
        )  # fmt: skip
        for level, label in ((1.0, "start"), (0.5, "half"), (params.progression, "progression")):
            ax.axhline(level * initial, color="#555555", lw=0.8, ls="--")
            y = level * initial
            ax.text(longest, y, f" {label}", va="center", fontsize=8, color="#555555")
        outcome = "progression after" if progressed else "no progression in"
        ax.set_title(f"{title}: {outcome} {len(doses)} days", fontsize=9)
        ax.set_xlabel("day")
        ax.set_xlim(0, longest)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("cells")
    handles = [Patch(color=color, label=label) for label, color in colors.items()]
    fig.legend(handles=handles, loc="outside lower center", ncols=3, frameon=False, fontsize=8)
    fig.savefig(args.output, dpi=160)
    print(f"written to {args.output}")


if __name__ == "__main__":
    main()
