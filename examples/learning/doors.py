"""Evacuation: the doors through which the pedestrians of each policy leave.

Reads the policies written by ``train.py`` for the task ``evacuation`` and
counts, over a number of episodes, how many pedestrians leave through each
door: for every seed of PPO (best action) and for the rule "walk to the
nearest door". A pedestrian is assigned to the door closest to its last
position in the room.

    python examples/learning/doors.py --input results --output results
"""

import argparse
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
from tasks import TASKS

from particlegym.baselines import ppo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", default="results")
    parser.add_argument("--output", default="results")
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    task = TASKS["evacuation"]()
    env, params, config = task.env, task.params, task.config
    centres = jnp.asarray(env.door_centres)
    n = env.n_agents

    def count(policy_fn, key):
        reset_key, run_key = jax.random.split(key)
        obs, state = env.reset(reset_key, params)

        def body(carry, key):
            state, obs, last = carry
            action_key, step_key = jax.random.split(key)
            action = policy_fn(action_key, obs, state)
            obs, state, *_ = env.step(step_key, state, action, params)
            inside = state.particles.alive[:n]
            last = jnp.where(inside[:, None], state.particles.pos[:n], last)
            return (state, obs, last), None

        carry = (state, obs, state.particles.pos[:n])
        keys = jax.random.split(run_key, params.max_steps)
        (state, _, last), _ = jax.lax.scan(body, carry, keys)
        left = ~state.particles.alive[:n]
        door = jnp.argmin(jnp.sum((last[:, None, :] - centres) ** 2, axis=-1), axis=1)
        return jnp.stack([jnp.sum(left & (door == d)) for d in range(centres.shape[0])])

    def mean_count(policy_fn):
        keys = jax.random.split(jax.random.key(args.seed), args.episodes)
        counts = jax.jit(jax.vmap(lambda key: count(policy_fn, key)))(keys)
        return np.asarray(counts).mean(axis=0)

    source = pathlib.Path(args.input)
    policies = ppo.load_policy(source / "evacuation_policy.npz", env, config)
    returns = np.load(source / "evacuation_evaluation.npz")["PPO/episode_return"].mean(axis=1)
    doors = "  ".join(f"({x:g}, {y:g})" for x, y in np.asarray(centres))
    lines = [
        f"task evacuation: pedestrians of {n} that leave through each door, "
        f"mean of {args.episodes} episodes of at most {params.max_steps} steps",
        f"door centres: {doors}",
    ]
    for seed in range(returns.shape[0]):
        policy = jax.tree.map(lambda leaf, seed=seed: leaf[seed], policies)
        counts = mean_count(
            lambda key, obs, state, policy=policy: ppo.act(
                policy, env, obs, normalize=config.normalize
            )
        )
        row = "  ".join(f"{c:5.1f}" for c in counts)
        lines.append(f"PPO, seed {seed} (return {returns[seed]:7.3f})   {row}")
    counts = mean_count(task.baselines["walk to the nearest door"])
    lines.append("walk to the nearest door          " + "  ".join(f"{c:5.1f}" for c in counts))

    text = "\n".join(lines) + "\n"
    print(text, end="")
    output = pathlib.Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "evacuation_doors.txt").write_text(text)


if __name__ == "__main__":
    main()
