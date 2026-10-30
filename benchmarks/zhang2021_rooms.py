"""Evacuation of the rooms of Zhang, Chai & Lykotrafitis (2021) by 80 pedestrians.

Every pedestrian walks to the nearest door, with one of two rules:

- ``nearest_exit``: the built-in crowd policy, straight towards a point 1 m
  beyond the centre of the nearest door;
- ``eight directions``: the one of the eight action directions that is
  closest to that heading. The code of the paper compares its learned
  policies with this rule (aimed at the door centre there).

Each room and rule is run from 20 random starts for 60 s. The script prints
the number of runs in which everybody left, their evacuation time (when the
last pedestrian left: mean ± standard deviation, and range), the time at
which 95 % of the pedestrians had left (over all runs), the number of
pedestrians still inside after 60 s (over all runs), and the mean number of
pedestrians per door, in the order of the doors of the layout. With
``--picture``, it also draws both rooms 3 s after the first start with the
built-in rule, the pedestrians coloured by the door they leave through.

    python benchmarks/zhang2021_rooms.py > benchmarks/zhang2021_rooms.txt
    python benchmarks/zhang2021_rooms.py --picture docs/assets/environments/zhang2021_rooms.png
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from particlegym import envs, viz

PEDESTRIANS = 80
STARTS = 20
STEPS = 600  # control steps of 0.1 s
SNAPSHOT = 30  # control steps before the picture
RULES = ("nearest_exit", "eight directions")
DOOR_COLORS = np.array([viz.BLUE, viz.ORANGE, viz.GREEN, viz.RED])


def make_env(layout: str, rule: str):
    """The room with 80 pedestrians, and their actions as a function of the state."""
    if rule == "nearest_exit":
        env = envs.zhang2021(layout, n_agents=0, n_crowd=PEDESTRIANS, action="continuous")
        return env, lambda state: jnp.zeros((0, 2))

    env = envs.zhang2021(layout, n_agents=PEDESTRIANS)
    directions = jnp.asarray(env.directions)
    centres = jnp.asarray(env.door_centres)
    targets = jnp.asarray(env.door_targets)

    def eight_directions(state):
        pos = state.particles.pos
        nearest = jnp.argmin(jnp.sum((pos[:, None, :] - centres) ** 2, axis=-1), axis=1)
        return jnp.argmax((targets[nearest] - pos) @ directions.T, axis=1)

    return env, eight_directions


def evacuate(env, policy, keys):
    """Final states and, for every run, the pedestrians at the time of the picture."""
    params = env.default_params

    def episode(key):
        reset_key, step_key = jax.random.split(key)
        _, state = env.reset(reset_key, params)

        def body(s, k):
            s = env.step(k, s, policy(s), params)[1]
            return s, s.particles

        state, history = jax.lax.scan(body, state, jax.random.split(step_key, STEPS))
        return state, jax.tree.map(lambda x: x[SNAPSHOT - 1], history)

    return jax.jit(jax.vmap(episode))(keys)


def doors_used(env, pos: np.ndarray) -> np.ndarray:
    """Door of every pedestrian: the one nearest to where it was removed."""
    return np.argmin(((pos[..., None, :] - env.door_centres) ** 2).sum(-1), axis=-1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--picture", help="also draw both rooms into this PNG file")
    parser.add_argument("--width", type=int, default=400, help="width of one room in pixels")
    args = parser.parse_args()

    keys = jax.random.split(jax.random.key(0), STARTS)
    print(
        f"JAX {jax.__version__} on {jax.devices()[0].platform.upper()}, "
        f"{PEDESTRIANS} pedestrians, {STARTS} random starts per row"
    )
    print(
        "| Room | Rule | Runs in which all left | Evacuation time (s) | 95 % out (s) "
        "| Inside after 60 s | Pedestrians per door |"
    )
    print("|---|---|---|---|---|---|---|")
    pictures = []
    for layout in envs.ZHANG2021_LAYOUTS:
        for rule in RULES:
            env, policy = make_env(layout, rule)
            final, snapshot = evacuate(env, policy, keys)
            exit_time = np.asarray(final.exit_time)
            out = np.isfinite(exit_time)
            complete = out.all(axis=1)
            last = exit_time[complete].max(axis=1)
            time = "n/a"
            if complete.sum() > 1:
                time = f"{last.mean():.1f} ± {last.std(ddof=1):.1f}"
                time += f" ({last.min():.1f} to {last.max():.1f})"
            most = np.sort(exit_time, axis=1)[:, round(0.95 * PEDESTRIANS) - 1]
            doors = doors_used(env, np.asarray(final.particles.pos))
            counts = np.mean([np.bincount(d, minlength=len(env.exits)) for d in doors], axis=0)
            per_door = " / ".join(f"{c:.1f}" for c in counts)
            print(
                f"| `{layout}` | {rule} | {complete.sum()} of {STARTS} | {time} "
                f"| {most.mean():.1f} ± {most.std(ddof=1):.1f} | {(~out).sum()} | {per_door} |",
                flush=True,
            )

            if args.picture and rule == "nearest_exit":
                first = jax.tree.map(lambda x: x[0], final)
                state = first._replace(particles=jax.tree.map(lambda x: x[0], snapshot))
                pictures.append(env.render(state, args.width, DOOR_COLORS[doors[0]]))

    if args.picture:
        from PIL import Image

        gap = np.full((pictures[0].shape[0], args.width // 20, 3), 255, np.uint8)
        Image.fromarray(np.concatenate([pictures[0], gap, pictures[1]], axis=1)).save(args.picture)


if __name__ == "__main__":
    main()
