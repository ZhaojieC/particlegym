# ParticleGym, https://github.com/ZhaojieC/particlegym (MIT license).
# Please cite it in published work: particlegym.cite() gives the references.
"""Figure: sensitive and resistant cells under the maximum tolerated dose and adaptive therapy.

The same two runs as ``tumor_therapy.py``. The numbers of sensitive (green)
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

from particlegym import envs


def run(env, params, policy, seed: int, days: int):
    """Daily numbers of sensitive and resistant cells and doses, until progression."""
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
    return np.array(sensitive), np.array(resistant), np.array(doses)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--days", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="tumor_therapy_curves.png")
    args = parser.parse_args()

    env = envs.TumorTherapy()
    params = env.default_params
    runs = [
        ("Maximum tolerated dose", run(env, params, env.mtd_policy, args.seed, args.days)),
        ("Adaptive therapy", run(env, params, env.adaptive_policy, args.seed, args.days)),
    ]
    initial = env.initial_cells
    longest = max(len(r[1][0]) for r in runs)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True, constrained_layout=True)
    for ax, (title, (sensitive, resistant, doses)) in zip(axes, runs, strict=True):
        days = np.arange(len(sensitive))
        for day in np.flatnonzero(doses > 0):
            ax.axvspan(day, day + 1, color="#e9e0f5", lw=0)
        ax.stackplot(
            days, sensitive, resistant, colors=["#3caa6e", "#dc3c28"],
            labels=["sensitive", "resistant"], step="post",
        )  # fmt: skip
        for level, label in ((1.0, "start"), (0.5, "half"), (params.progression, "progression")):
            ax.axhline(level * initial, color="#555555", lw=0.8, ls="--")
            y = level * initial
            ax.text(longest, y, f" {label}", va="center", fontsize=8, color="#555555")
        ax.set_title(f"{title}: progression after {len(doses)} days", fontsize=10)
        ax.set_xlabel("day")
        ax.set_xlim(0, longest)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("cells")
    axes[1].legend(loc="upper left", frameon=False, fontsize=8)
    fig.savefig(args.output, dpi=160)
    print(f"written to {args.output}")


if __name__ == "__main__":
    main()
