"""Animation: a policy before and after training, side by side.

Reads the policies written by ``train.py``. Of the seeds, the one with the
median return in the evaluation is shown, not the best one. Both halves of
the picture start from the same state: on the left the policy before
training (with its exploration noise), on the right the trained policy
(without noise).

    python examples/learning/animate.py --task brick_wall --input results \
        --output learned_brick_wall.gif
"""

import argparse
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
from tasks import TASKS

from particlegym import viz
from particlegym.baselines import ppo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--input", default="results")
    parser.add_argument("--steps", type=int, default=400, help="control steps")
    parser.add_argument("--stride", type=int, default=4, help="control steps per frame")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=220, help="width of one half in pixels")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--colors", type=int, default=48)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    task = TASKS[args.task]()
    env = task.env if task.display is None else task.display
    params, config = task.params, task.config
    source = pathlib.Path(args.input)
    evaluation = np.load(source / f"{args.task}_evaluation.npz")
    per_seed = evaluation["PPO/episode_return"].mean(axis=1)
    seed = int(np.argsort(per_seed)[len(per_seed) // 2])
    print(f"returns of the seeds {np.round(per_seed, 3)}; showing seed {seed}")

    def one(policies):
        return jax.tree.map(lambda leaf: leaf[seed], policies)

    untrained = one(ppo.load_policy(source / f"{args.task}_untrained.npz", task.env, config))
    trained = one(ppo.load_policy(source / f"{args.task}_policy.npz", task.env, config))

    @jax.jit
    def rollout(policy, noisy, key):
        reset_key, run_key = jax.random.split(key)
        obs, state = env.reset(reset_key, params)

        def body(carry, key):
            state, obs, over = carry
            action_key, step_key = jax.random.split(key)
            sampled = ppo.act(policy, env, obs, action_key, normalize=config.normalize)
            best = ppo.act(policy, env, obs, normalize=config.normalize)
            action = jnp.where(noisy, sampled, best)
            new_obs, new_state, _, terminated, truncated, _ = env.step(
                step_key, state, action, params
            )
            # The picture stops when the episode is over.
            state = jax.tree.map(lambda new, old: jnp.where(over, old, new), new_state, state)
            obs = jnp.where(over, obs, new_obs)
            over = over | jnp.all(terminated | truncated)
            return (state, obs, over), state

        _, states = jax.lax.scan(body, (state, obs, False), jax.random.split(run_key, args.steps))
        return state, states

    key = jax.random.key(args.seed)
    halves = []
    for policy, noisy in ((untrained, True), (trained, False)):
        first, states = rollout(policy, noisy, key)
        frames = [env.render(first, args.width)]
        for i in range(args.stride - 1, args.steps, args.stride):
            frames.append(env.render(jax.tree.map(lambda leaf, i=i: leaf[i], states), args.width))
        halves.append(frames)

    gap = np.full((halves[0][0].shape[0], 8, 3), 255, np.uint8)
    frames = [np.concatenate([a, gap, b], axis=1) for a, b in zip(*halves, strict=True)]
    output = args.output or f"learned_{args.task}.gif"
    viz.save_gif(output, frames, fps=args.fps, colors=args.colors)
    print(f"{len(frames)} frames written to {output}")


if __name__ == "__main__":
    main()
