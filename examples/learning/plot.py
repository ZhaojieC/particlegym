"""Draw the learning curves written by ``train.py`` and ``microswimmer_q.py``.

One figure per task: the return of the episodes that ended during training
(mean over the seeds, and the range from the lowest to the highest seed)
and, as horizontal lines, the mean return of the baselines in the
evaluation episodes. For the Q-learning runs the band reaches from the 10th
to the 90th percentile of the learners. Needs Matplotlib.

    python examples/learning/plot.py --input results --output results
"""

import argparse
import itertools
import pathlib
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tasks import TASKS

COLORS = ("#d62728", "#1f77b4", "#2ca02c", "#ff7f0e")


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "font.size": 11,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "legend.frameon": False,
            "legend.fontsize": 9,
        }
    )


def binned(values: np.ndarray, n_bins: int):
    """Mean of the finite values in ``n_bins`` bins along the last axis."""
    edges = np.linspace(0, values.shape[-1], n_bins + 1).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # bins in which no episode ended
        parts = [np.nanmean(values[..., a:b], axis=-1) for a, b in itertools.pairwise(edges)]
    return 0.5 * (edges[:-1] + edges[1:]), np.stack(parts, axis=-1)


def save(fig, path: pathlib.Path) -> None:
    fig.tight_layout(pad=0.4)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"wrote {path}")


def plot_task(name: str, source: pathlib.Path, target: pathlib.Path) -> None:
    curves = np.load(source / f"{name}_curves.npz")
    evaluation = np.load(source / f"{name}_evaluation.npz")
    task = TASKS[name]()
    returns = curves["episode_return"]  # (seeds, updates)
    updates, values = binned(returns, min(40, returns.shape[1]))
    steps = updates * float(curves["agent_steps_per_update"]) / 1e6
    keep = np.isfinite(values).all(axis=0)

    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    ax.fill_between(
        steps[keep], values.min(axis=0)[keep], values.max(axis=0)[keep], color=COLORS[0], alpha=0.2
    )
    ax.plot(steps[keep], values.mean(axis=0)[keep], color=COLORS[0], lw=1.6, label="PPO")
    for color, label in zip(COLORS[1:], task.baselines, strict=False):
        level = evaluation[f"{label}/episode_return"].mean()
        ax.axhline(level, color=color, lw=1.2, ls="--", label=label)
    ax.set_xlabel("agent steps (millions)")
    ax.set_ylabel(task.label)
    ax.set_xlim(0.0, steps[-1])
    ax.legend()
    save(fig, target / f"learning_{name}.png")


def plot_q_learning(source: pathlib.Path, target: pathlib.Path) -> None:
    data = np.load(source / "microswimmer_q_curves.npz")
    scale = 1.0 / (float(data["duration"]) * float(data["swim_speed"]))
    returns = scale * data["returns"].T  # (learners, episodes)
    episodes, values = binned(returns, min(50, returns.shape[1]))

    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    low, high = np.percentile(values, [10, 90], axis=0)
    ax.fill_between(episodes, low, high, color=COLORS[0], alpha=0.2)
    ax.plot(episodes, values.mean(axis=0), color=COLORS[0], lw=1.6, label="Q-learning")
    ax.axhline(scale * data["naive"].mean(), color=COLORS[1], lw=1.2, ls="--", label="naive")
    ax.set_xlabel("episode")
    ax.set_ylabel("climbing speed / swim speed")
    ax.set_xlim(0.0, returns.shape[1])
    ax.legend(loc="lower right")
    save(fig, target / "learning_microswimmer_q.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", default="results")
    parser.add_argument("--output", default="results")
    args = parser.parse_args()
    source, target = pathlib.Path(args.input), pathlib.Path(args.output)
    target.mkdir(parents=True, exist_ok=True)
    style()
    for name in TASKS:
        if (source / f"{name}_evaluation.npz").exists():
            plot_task(name, source, target)
    if (source / "microswimmer_q_curves.npz").exists():
        plot_q_learning(source, target)


if __name__ == "__main__":
    main()
