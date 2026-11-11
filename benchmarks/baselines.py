"""Scripted and random policies in the experimental environments.

The numbers quoted in the documentation of the environments come from this
script. Every part uses fixed random seeds and double precision. The
reference output is in ``benchmarks/baselines.txt``. Run with

    python benchmarks/baselines.py > benchmarks/baselines.txt

or with the names of the parts to run (herding, microswimmer, cell_migration,
billiards, orbit_transfer, ring_road, soft_walker, tumor_therapy).
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


def ring_road(n_games=32):
    print(f"# RingRoad: 22 vehicles, one automated, {n_games} games of 600 control steps (300 s)")
    print("# from a stop-and-go wave. Return: mean distance driven per vehicle (m); speed")
    print("# spread: standard deviation of the speeds over the vehicles, mean over the last 150 s")
    print("# ring_length  action  policy  return_mean  return_min  return_max  speed_spread")
    for ring_length in (220.0, 230.0, 250.0, 270.0):
        for action in ("continuous", "discrete"):
            env = envs.RingRoad(action=action)
            params = env.default_params._replace(ring_length=ring_length)
            results = {}
            policies = {
                "idm": lambda k, s, env=env, params=params: env.idm_policy(s, params),
                "smoothing": lambda k, s, env=env, params=params: env.smoothing_policy(s, params),
                "random": lambda k, s, env=env: env.action_space.sample(k, (env.n_agents,)),
            }
            for name, policy in policies.items():

                @jax.jit
                @jax.vmap
                def game(key, env=env, params=params, policy=policy):
                    reset_key, run_key = jax.random.split(key)
                    _, state = env.reset(reset_key, params)

                    def body(s, k):
                        action_key, step_key = jax.random.split(k)
                        out = env.step(step_key, s, policy(action_key, s), params)
                        return out[1], (out[2][0], out[5]["speed_std"])

                    keys = jax.random.split(run_key, params.max_steps)
                    _, (reward, spread) = jax.lax.scan(body, state, keys)
                    return jnp.sum(reward), spread

                returns, spread = (
                    np.asarray(x) for x in game(jax.random.split(jax.random.key(0), n_games))
                )
                results[name] = returns, spread
                low, high = returns.min(), returns.max()
                late = spread[:, params.max_steps // 2 :].mean()
                print(
                    f"{ring_length:g}  {action}  {name}  {returns.mean():.1f}  {low:.1f}  "
                    f"{high:.1f}  {late:.3f}"
                )
            # The same games with and without smoothing: the gain of every game, and
            # the time after which the spread first falls below 0.6 m/s.
            gain = results["smoothing"][0] / results["idm"][0] - 1.0
            below = results["smoothing"][1] < 0.6
            calm = np.where(below.any(axis=1), below.argmax(axis=1) + 1, np.inf)
            calm = calm * env.control_interval
            print(
                f"# smoothing, gain per game: min {gain.min():.3f}, max {gain.max():.3f}; "
                f"spread below 0.6 m/s after {np.median(calm):.1f} s (median), "
                f"{calm.max():.1f} s (latest)"
            )


def soft_walker(n_games=8):
    print("# SoftWalker: the default robot, episodes of 500 control steps (50 time units);")
    print(f"# distance of the centre of mass in voxels, {n_games} games from random starts")
    print("# friction  policy  distance_mean  distance_min  distance_max")
    env = envs.SoftWalker()
    policies = {
        "gait": lambda k, s, p: env.gait_policy(s, p),
        "idle": lambda k, s, p: jnp.zeros((1, env.n_actuators)),
        "random": lambda k, s, p: env.action_space.sample(k, (1,)),
    }
    for friction in (0.5, 1.0, 1.5):
        params = env.default_params._replace(friction=friction)
        for name, policy in policies.items():

            @jax.jit
            @jax.vmap
            def game(key, params=params, policy=policy):
                reset_key, run_key = jax.random.split(key)
                _, state = env.reset(reset_key, params)

                def body(s, k):
                    action_key, step_key = jax.random.split(k)
                    out = env.step(step_key, s, policy(action_key, s, params), params)
                    return out[1], out[2][0]

                keys = jax.random.split(run_key, params.max_steps)
                return jnp.sum(jax.lax.scan(body, state, keys)[1])

            distance = np.asarray(game(jax.random.split(jax.random.key(0), n_games)))
            print(
                f"{friction:g}  {name}  {distance.mean():.2f}  {distance.min():.2f}  "
                f"{distance.max():.2f}"
            )


def tumor_therapy(n_games=32):
    print(f"# TumorTherapy: 200 cells, {n_games} games from random starts for each share of")
    print("# resistant cells. Days to progression (1.2 times the initial size), or 365 if the")
    print("# tumour did not progress within the episode; games that progressed, games in which")
    print("# all cells died, share of days treated, and the share of resistant cells at")
    print("# progression. Adaptive: the rule of the abiraterone trial (pause at half the")
    print("# initial size, resume at the initial size); containment: treat whenever the")
    print("# tumour has grown back to its initial size.")
    print(
        "# resistant  policy  days_mean  days_median  days_min  days_max  progressed  extinct  "
        "treated  resistant_share"
    )
    env = envs.TumorTherapy()
    for fraction in (0.02, 0.05):
        params = env.default_params._replace(resistant_fraction=fraction)
        policies = {
            "none": lambda s, p: env.no_treatment_policy(s, p),
            "mtd": lambda s, p: env.mtd_policy(s, p),
            "adaptive": lambda s, p: env.adaptive_policy(s, p),
            "containment": lambda s, p: env.adaptive_policy(s, p, stop=1.0, restart=1.0),
        }
        for name, policy in policies.items():

            @jax.jit
            @jax.vmap
            def game(key, policy=policy, params=params):
                reset_key, run_key = jax.random.split(key)
                _, state = env.reset(reset_key, params)

                def running(carry):
                    state, *_ = carry
                    cells = jnp.sum(state.cells.alive)
                    return ~state.progressed & (cells > 0) & (state.day < params.max_days)

                def day(carry):
                    state, treated, _ = carry
                    key = jax.random.fold_in(run_key, state.day)
                    out = env.step(key, state, policy(state, params), params)
                    info = out[5]
                    cells = jnp.maximum(info["sensitive"] + info["resistant"], 1)
                    return out[1], treated + out[1].dose, info["resistant"] / cells

                zero = jnp.zeros((), state.dose.dtype)
                state, treated, share = jax.lax.while_loop(running, day, (state, zero, zero))
                extinct = jnp.sum(state.cells.alive) == 0
                days = jnp.where(state.progressed, state.day, params.max_days)
                return days, state.progressed, extinct, treated / state.day, share

            days, progressed, extinct, treated, share = (
                np.asarray(x) for x in game(jax.random.split(jax.random.key(0), n_games))
            )
            resistant = share[progressed].mean() if progressed.any() else float("nan")
            print(
                f"{fraction:g}  {name}  {days.mean():.1f}  {np.median(days):.1f}  {days.min()}  "
                f"{days.max()}  "
                f"{progressed.sum()}/{n_games}  {extinct.sum()}/{n_games}  "
                f"{treated.mean():.2f}  {resistant:.2f}"
            )


PARTS = {
    "herding": herding,
    "microswimmer": microswimmer,
    "cell_migration": cell_migration,
    "billiards": billiards,
    "orbit_transfer": orbit_transfer,
    "ring_road": ring_road,
    "soft_walker": soft_walker,
    "tumor_therapy": tumor_therapy,
}

if __name__ == "__main__":
    for part in sys.argv[1:] or PARTS:
        PARTS[part]()
        print(flush=True)
