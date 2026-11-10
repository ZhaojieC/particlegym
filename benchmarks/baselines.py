"""Scripted and random policies in the experimental environments.

The numbers quoted in the documentation of the environments come from this
script. Every part uses fixed random seeds and double precision. The
reference output is in ``benchmarks/baselines.txt``. Run with

    python benchmarks/baselines.py > benchmarks/baselines.txt

or with the names of the parts to run (herding, microswimmer, cell_migration,
billiards, orbit_transfer).
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np

from particlegym import envs
from particlegym import physics as pp

jax.config.update("jax_enable_x64", True)


def herding(n_games=16, steps=1000):
    print(f"# Herding: 40 sheep, {n_games} games of at most {steps} steps")
    print("# shepherds  policy  solved  steps_min  steps_mean  steps_max  final_distance_mean")
    for n_agents in (1, 2):
        env = envs.Herding(n_agents=n_agents)
        params = env.default_params
        policies = {
            "scripted": lambda k, s, env=env, params=params: env.shepherd_policy(k, s, params),
            "random": lambda k, s, env=env: env.action_space.sample(k, (env.n_agents,)),
        }
        for name, policy in policies.items():

            @jax.jit
            @jax.vmap
            def game(key, env=env, params=params, policy=policy):
                reset_key, run_key = jax.random.split(key)
                _, state = env.reset(reset_key, params)

                def body(carry, k):
                    s, done, used = carry
                    action_key, step_key = jax.random.split(k)
                    out = env.step(step_key, s, policy(action_key, s), params)
                    s = jax.tree.map(lambda old, new: jnp.where(done, old, new), s, out[1])
                    return (s, done | out[3][0], used + ~done), out[5]["distance"]

                carry, distance = jax.lax.scan(
                    body, (state, False, 0), jax.random.split(run_key, steps)
                )
                return carry[1], carry[2], distance[-1]

            done, used, distance = (
                np.asarray(x) for x in game(jax.random.split(jax.random.key(0), n_games))
            )
            used = used[done] if done.any() else np.array([np.nan])
            print(
                f"{n_agents}  {name}  {done.sum()}/{n_games}  {used.min():.0f}  "
                f"{used.mean():.1f}  {used.max():.0f}  {distance.mean():.1f}"
            )


def microswimmer(n=256, steps=2000):
    print(f"# Microswimmer: {n} swimmers, {steps} steps of 0.1 time units; climbing speed")
    print("# from the altitude gained in the second half, in units of the swimming speed")
    print("# Phi  Psi  policy  mean  standard_deviation_over_swimmers")
    env = envs.Microswimmer(n_agents=n)
    points = [(0.3, 1.0), (0.1, 0.3), (0.3, 0.1), (0.3, 0.3), (0.3, 10.0), (1.0, 1.0), (3.0, 1.0)]
    for phi, psi in points:
        params = env.default_params._replace(swim_speed=phi, alignment_time=psi)
        policies = {
            "up": lambda k, s: env.naive_policy(s),
            "random": lambda k, s: env.action_space.sample(k, (n,)),
        }
        for name, policy in policies.items():

            @jax.jit
            def run(key, params=params, policy=policy):
                reset_key, run_key = jax.random.split(key)
                _, state = env.reset(reset_key, params)

                def body(s, k):
                    action_key, step_key = jax.random.split(k)
                    s = env.step(step_key, s, policy(action_key, s), params)[1]
                    return s, s.displacement[:, 1]

                return jax.lax.scan(body, state, jax.random.split(run_key, steps))[1]

            altitude = np.asarray(run(jax.random.key(0)))
            time = 0.5 * steps * env.dt * env.substeps
            speed = (altitude[-1] - altitude[steps // 2 - 1]) / time / phi
            print(f"{phi:g}  {psi:g}  {name}  {speed.mean():.4f}  {speed.std():.4f}")


def cell_migration(n_games=8, steps=400):
    print(f"# CellMigration: 9 leaders, source at (7.5, 7.5), {n_games} games of {steps} steps,")
    print("# gradient-following cells with a weight on the direction to the other cells")
    print("# weight  arrived  steps_min  steps_mean  steps_max  distance_at_100  spread_at_100")
    env = envs.CellMigration()
    params = env.default_params
    for weight in (0.0, 0.3, 0.5):

        @jax.jit
        @jax.vmap
        def game(key, weight=weight):
            _, state = env.reset(key, params)

            def body(s, _):
                out = env.step(None, s, env.gradient_policy(s, weight), params)
                return out[1], (out[3][0], out[5]["distance"], out[5]["spread"])

            return jax.lax.scan(body, state, None, length=steps)[1]

        done, distance, spread = (
            np.asarray(x) for x in game(jax.random.split(jax.random.key(0), n_games))
        )
        first = done.argmax(axis=1)[done.any(axis=1)] + 1
        print(
            f"{weight:g}  {done.any(axis=1).sum()}/{n_games}  {first.min()}  {first.mean():.1f}  "
            f"{first.max()}  {distance[:, 99].mean():.2f}  {spread[:, 99].mean():.3f}"
        )


def billiards(n_games=16, shots=30):
    print(f"# Billiards: 15 object balls, {n_games} games of {shots} shots")
    print("# policy  pocketed_mean  pocketed_min  pocketed_max  scratches_mean")
    env = envs.Billiards()
    params = env.default_params
    policies = {
        "aim": lambda k, s: env.aim_policy(s),
        "random": lambda k, s: env.action_space.sample(k, (1,)),
    }
    for name, policy in policies.items():

        @jax.jit
        @jax.vmap
        def game(key, policy=policy):
            reset_key, run_key = jax.random.split(key)
            _, state = env.reset(reset_key, params)

            def body(carry, k):
                s, done = carry
                action_key, step_key = jax.random.split(k)
                out = env.step(step_key, s, policy(action_key, s), params)
                s = jax.tree.map(lambda old, new: jnp.where(done, old, new), s, out[1])
                return (s, done | out[3][0]), jnp.where(done, False, out[5]["scratch"])

            (final, _), scratch = jax.lax.scan(
                body, (state, False), jax.random.split(run_key, shots)
            )
            return 15 - jnp.sum(final.balls.alive[1:]), jnp.sum(scratch)

        pocketed, scratch = (
            np.asarray(x) for x in game(jax.random.split(jax.random.key(0), n_games))
        )
        print(
            f"{name}  {pocketed.mean():.2f}  {pocketed.min()}  {pocketed.max()}  "
            f"{scratch.mean():.2f}"
        )


def orbit_transfer(n_games=16, steps=400):
    print(f"# OrbitTransfer: from the circular orbit r = 1 to r = r1 (G M = 1), at most {steps}")
    print("# control steps of 0.1 time units. Hohmann controller: the step of arrival and the")
    print("# delta-v spent, against the impulsive optimum")
    print("# action  r1  arrived_at_step  delta_v  optimum  ratio")
    for action in ("continuous", "discrete"):
        env = envs.OrbitTransfer(action=action)
        for target in (1.5, 2.0, 3.0, 4.0):
            params = env.default_params._replace(target_radius=target)

            @jax.jit
            def run(key, env=env, params=params):
                _, state = env.reset(key, params)

                def body(s, _):
                    out = env.step(None, s, env.hohmann_policy(s, params), params)
                    return out[1], out[5]["arrived"][0]

                return jax.lax.scan(body, state, None, length=steps)

            final, arrived = run(jax.random.key(0))
            arrived = np.asarray(arrived)
            used = float(final.fuel_used[0])
            optimum = pp.hohmann_delta_v(1.0, target)
            at = f"{arrived.argmax() + 1}" if arrived.any() else "no"
            print(f"{action}  {target:g}  {at}  {used:.4f}  {optimum:.4f}  {used / optimum:.3f}")

    print(f"# Random actions, r1 = 2, {n_games} games")
    print("# action  arrived  crashed  escaped  delta_v_mean")
    for action in ("continuous", "discrete"):
        env = envs.OrbitTransfer(action=action)
        params = env.default_params

        @jax.jit
        @jax.vmap
        def game(key, env=env, params=params):
            reset_key, run_key = jax.random.split(key)
            _, state = env.reset(reset_key, params)

            def body(s, k):
                out = env.step(None, s, env.action_space.sample(k, (1,)), params)
                info = out[5]
                return out[1], (info["arrived"][0], info["crashed"][0], info["escaped"][0])

            final, ends = jax.lax.scan(body, state, jax.random.split(run_key, steps))
            return final.fuel_used[0], *(jnp.any(end) for end in ends)

        used, arrived, crashed, escaped = (
            np.asarray(x) for x in game(jax.random.split(jax.random.key(0), n_games))
        )
        print(
            f"{action}  {arrived.sum()}/{n_games}  {crashed.sum()}/{n_games}  "
            f"{escaped.sum()}/{n_games}  {used.mean():.3f}"
        )


PARTS = {
    "herding": herding,
    "microswimmer": microswimmer,
    "cell_migration": cell_migration,
    "billiards": billiards,
    "orbit_transfer": orbit_transfer,
}

if __name__ == "__main__":
    for part in sys.argv[1:] or PARTS:
        PARTS[part]()
        print(flush=True)
