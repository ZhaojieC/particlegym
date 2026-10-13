"""Throughput of the environments: control steps per second with random actions.

Each case steps a batch of environments (``jax.vmap``) for a number of control
steps inside one compiled ``jax.lax.scan``, after a warm-up run that compiles
it, and reports the best of three timings. Episodes that end are not reset.
A control step consists of several physics steps (the column "Substeps"); a
step of the billiards game is a whole shot, and in a batch it lasts until the
balls of the slowest table have stopped. A case that fails, for example for
lack of memory, is reported as "failed".

    python benchmarks/bench_envs.py --batches 1 64
    python benchmarks/bench_envs.py --batches 1024 16384   # on a GPU
"""

import argparse
import time

import jax

from particlegym import envs
from particlegym import physics as pp

DOOR = (pp.Opening("x+", (4.4, 5.6)),)
FOUR_DOORS = tuple(pp.Opening(face, (4.5, 5.5)) for face in ("y+", "y-", "x-", "x+"))
# Name, constructor, control steps per timing.
CASES = (
    ("Evacuation, 1 pedestrian", lambda: envs.Evacuation((10.0, 10.0), DOOR), 200),
    (
        "Evacuation, 40 pedestrians, 4 doors",
        lambda: envs.Evacuation((10.0, 10.0), FOUR_DOORS, n_agents=40),
        50,
    ),
    ("Counter-flow, 80 pedestrians", lambda: envs.CounterFlow(), 50),
    ("Brick wall, state observations", lambda: envs.BrickWall(), 200),
    ("Brick wall, 84 x 84 pixels", lambda: envs.BrickWall(observation="pixels"), 100),
    ("Billiards, 15 balls (one shot per step)", lambda: envs.Billiards(), 2),
    ("Herding, 40 sheep", lambda: envs.Herding(), 50),
    ("Microswimmer", lambda: envs.Microswimmer(), 200),
    ("Cell migration, 9 cells", lambda: envs.CellMigration(), 20),
)


def measure(env, batch: int, steps: int) -> float:
    """Control steps per second of a batch of environments."""
    params = env.default_params
    reset = jax.vmap(env.reset, in_axes=(0, None))
    step = jax.vmap(env.step, in_axes=(0, 0, 0, None))

    @jax.jit
    def run(key):
        reset_key, run_key = jax.random.split(key)
        _, state = reset(jax.random.split(reset_key, batch), params)

        def body(state, key):
            action_key, step_key = jax.random.split(key)
            action = env.action_space.sample(action_key, (batch, env.n_agents))
            out = step(jax.random.split(step_key, batch), state, action, params)
            return out[1], out[2].sum()

        _, rewards = jax.lax.scan(body, state, jax.random.split(run_key, steps))
        return rewards.sum()

    run(jax.random.key(0)).block_until_ready()
    best = float("inf")
    for i in range(3):
        start = time.perf_counter()
        run(jax.random.key(i + 1)).block_until_ready()
        best = min(best, time.perf_counter() - start)
    return batch * steps / best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 64])
    parser.add_argument("--only", help="run the cases whose name contains this text")
    args = parser.parse_args()

    print(f"JAX {jax.__version__} on {jax.devices()[0]}, float32")
    header = " | ".join(f"Batch {b}" for b in args.batches)
    print(f"| Environment | Substeps | {header} |")
    print("|---|---|" + "---|" * len(args.batches))
    for name, make, steps in CASES:
        if args.only and args.only.lower() not in name.lower():
            continue
        env = make()
        cells = []
        for batch in args.batches:
            try:
                cells.append(f"{measure(env, batch, steps):,.0f}")
            except Exception as error:  # report it and go on with the other cases
                cells.append("failed")
                print(f"{name}, batch {batch}: {type(error).__name__}", flush=True)
        cells = " | ".join(cells)
        print(f"| {name} | {getattr(env, 'substeps', '')} | {cells} |", flush=True)


if __name__ == "__main__":
    main()
