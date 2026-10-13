"""Tabular Q-learning of the microswimmer task, after Colabrese et al. (2017).

Every swimmer senses one of 12 states (three levels of the vorticity times
four sectors of its heading), chooses one of four preferred directions and
learns its own table of action values from the altitude it gains. The table
starts with large equal entries and is carried from episode to episode. The
swimmers do not interact, so they are independent learning runs.

    python examples/learning/microswimmer_q.py --output results

Writes ``microswimmer_q_curves.npz`` (the altitude gained in every episode,
one column per swimmer), ``microswimmer_q_tables.npz`` and
``microswimmer_q_summary.txt`` to the output directory.
"""

import argparse
import pathlib
import time

import jax
import numpy as np

from particlegym import envs
from particlegym.baselines import ppo, q_learning


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--swimmers", type=int, default=32, help="independent learners")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=1000, help="control steps per episode")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="results")
    args = parser.parse_args()

    env = envs.Microswimmer(n_agents=args.swimmers, observation="discrete")
    params = env.default_params._replace(max_steps=args.steps)
    config = q_learning.QConfig(
        n_episodes=args.episodes, n_steps=args.steps, learning_rate=0.1, gamma=0.99
    )
    train_key, eval_key = jax.random.split(jax.random.key(args.seed))
    start = time.time()
    tables, returns = jax.jit(lambda k: q_learning.train(k, env, params, config))(train_key)
    returns = np.asarray(returns)
    elapsed = time.time() - start

    def play(policy_fn):
        stats = ppo.evaluate(eval_key, env, params, policy_fn, 16, args.steps)
        return np.asarray(stats["episode_return"])

    learned = play(jax.jit(lambda key, obs, state: q_learning.greedy(tables, obs)))
    naive = play(lambda key, obs, state: env.naive_policy())

    duration = args.steps * env.dt * env.substeps
    to_speed = 1.0 / (duration * params.swim_speed)  # altitude gain to units of the swim speed
    out = pathlib.Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(
        out / "microswimmer_q_curves.npz",
        returns=returns,
        duration=duration,
        swim_speed=params.swim_speed,
        learned=learned,
        naive=naive,
    )
    np.savez(out / "microswimmer_q_tables.npz", tables=np.asarray(tables))
    first, last = returns[:100].mean(axis=0), returns[-100:].mean(axis=0)
    lines = [
        f"{args.swimmers} learners, {args.episodes} episodes of {duration:.0f} time units, "
        f"swim speed {params.swim_speed}, alignment time {params.alignment_time}, "
        f"flow speed {params.flow_speed}; settings {config}; {elapsed:.0f} s",
        "climbing speed in units of the swim speed, mean over the learners (smallest, largest):",
        f"  first 100 episodes of learning  {to_speed * first.mean():.3f} "
        f"({to_speed * first.min():.3f}, {to_speed * first.max():.3f})",
        f"  last 100 episodes of learning   {to_speed * last.mean():.3f} "
        f"({to_speed * last.min():.3f}, {to_speed * last.max():.3f})",
        "16 evaluation episodes with new starts, mean over episodes and swimmers:",
        f"  learned tables, best action     {to_speed * learned.mean():.3f}",
        f"  naive (always up)               {to_speed * naive.mean():.3f}",
    ]
    (out / "microswimmer_q_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
