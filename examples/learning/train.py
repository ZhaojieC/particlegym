"""Train policies with PPO on one of the tasks, several seeds at once.

All seeds train in one compiled program (``jax.vmap`` over the random key).
The run is cut into chunks of updates; after every chunk the curves and the
policies are written, so that a run that is stopped early leaves its results.
At the end the policies and the baselines play the same evaluation episodes.

    python examples/learning/train.py --task microswimmer --output results

Files written to the output directory, with the name of the task as prefix:
``_curves.npz`` (learning curves, one row per seed), ``_policy.npz`` (the
final policies), ``_untrained.npz`` (the policies before training),
``_evaluation.npz`` and ``_summary.txt`` (returns of the evaluation episodes).
"""

import argparse
import pathlib
import time
import warnings

import jax
import numpy as np
from tasks import TASKS, evaluate, greedy, sampled

from particlegym.baselines import ppo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, help="number of updates (default: of the task)")
    parser.add_argument("--envs", type=int, help="parallel environments (default: of the task)")
    parser.add_argument("--chunk", type=int, default=20, help="updates between two saves")
    parser.add_argument("--minutes", type=float, help="stop training after this time")
    parser.add_argument("--episodes", type=int, default=64, help="evaluation episodes")
    parser.add_argument("--output", default="results")
    args = parser.parse_args()

    task = TASKS[args.task]()
    config = task.config
    if args.iterations:
        config = config._replace(n_iterations=args.iterations)
    if args.envs:
        config = config._replace(n_envs=args.envs)
    task = task._replace(config=config)
    env, params = task.env, task.params
    out = pathlib.Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    prefix = str(out / args.task)
    per_update = config.n_envs * config.n_steps * env.n_agents
    lines = [
        f"task {args.task}, {args.seeds} seeds, device {jax.devices()[0]}",
        f"settings {config}",
        f"{per_update} agent steps per update",
    ]
    print("\n".join(lines), flush=True)

    key = jax.random.key(args.seed)
    train_key, eval_key = jax.random.split(key)
    state = jax.jit(jax.vmap(lambda k: ppo.init(k, env, params, config)))(
        jax.random.split(train_key, args.seeds)
    )
    ppo.save_policy(prefix + "_untrained.npz", state.policy)
    chunk = min(args.chunk, config.n_iterations)
    run = jax.jit(jax.vmap(lambda s: ppo.run(s, env, params, config, chunk)))

    curves, done, start = [], 0, time.time()
    while done + chunk <= config.n_iterations:
        state, part = run(state)
        curves.append({name: np.asarray(value) for name, value in part.items()})
        done += chunk
        merged = {name: np.concatenate([c[name] for c in curves], axis=1) for name in part}
        np.savez(prefix + "_curves.npz", agent_steps_per_update=per_update, **merged)
        ppo.save_policy(prefix + "_policy.npz", state.policy)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # no episode ended in the chunk
            recent = np.nanmean(merged["episode_return"][:, -chunk:], axis=1)
        elapsed = time.time() - start
        print(
            f"update {done:5d}  {elapsed:7.0f} s  return per seed {np.round(recent, 2)}",
            flush=True,
        )
        if args.minutes and elapsed > 60.0 * args.minutes:
            break
    elapsed = time.time() - start
    lines.append(
        f"{done} of {config.n_iterations} updates in {elapsed:.0f} s "
        f"({done * per_update * args.seeds / elapsed:.0f} agent steps per second, "
        "compilation included)"
    )

    def play(policy_fn):
        return evaluate(task, policy_fn, eval_key, args.episodes)

    results = {}
    for name, make in (("PPO", greedy), ("PPO with exploration noise", sampled)):
        stats = jax.jit(jax.vmap(lambda p, make=make: play(make(task, p))))(state.policy)
        results[name] = {k: np.asarray(v) for k, v in stats.items()}  # (seeds, episodes)
    for name, policy_fn in task.baselines.items():
        stats = jax.jit(lambda policy_fn=policy_fn: play(policy_fn))()
        results[name] = {k: np.asarray(v) for k, v in stats.items()}  # (episodes,)

    lines.append(
        f"evaluation: {args.episodes} episodes of at most {params.max_steps} steps, the same "
        "starts for every policy; return per agent, mean over the episodes"
    )
    for name, stats in results.items():
        returns = stats["episode_return"]
        if returns.ndim == 2:
            per_seed = [round(float(x), 3) for x in returns.mean(axis=1)]
            spread = f"seeds: {per_seed}"
        else:
            spread = f"std over episodes {returns.std():.3f}"
        lines.append(
            f"{name:28s} {returns.mean():9.3f}  ({spread}; "
            f"episodes finished {stats['finished'].mean():.2f}; "
            f"length {stats['episode_length'].mean():.0f})"
        )
    flat = {f"{name}/{k}": v for name, stats in results.items() for k, v in stats.items()}
    np.savez(prefix + "_evaluation.npz", **flat)
    pathlib.Path(prefix + "_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[3:]), flush=True)


if __name__ == "__main__":
    main()
