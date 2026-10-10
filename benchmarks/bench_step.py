"""Throughput of the physics core: particle-steps per second.

Each case runs a compiled `jax.lax.scan` over many BAOAB steps, after a warm-up
run that compiles it, and reports the best of three timings.

    python benchmarks/bench_step.py           # all cases
    python benchmarks/bench_step.py --quick   # small cases only
"""

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

from particlegym import physics as pp


def crowd(n, neighbors, seed=0):
    """2D social-force crowd at about 1 person per square metre in a room with a door."""
    side = float(np.sqrt(n))
    rng = np.random.default_rng(seed)
    radius = rng.uniform(0.25, 0.30, n)
    pos = pp.random_positions(rng, (0.0, 0.0), (side, side), radius, gap=0.05)
    door = pp.Opening("x+", (side / 2 - 0.6, side / 2 + 0.6))
    field = pp.ForceField(
        pp.Box((-1.0, -1.0), (side + 3.0, side + 1.0)),
        pair=[pp.SocialForce(), pp.TangentialFriction()],
        walls=[pp.Walls(pp.box_walls((0.0, 0.0), (side, side), [door]), pp.SocialForce(), 2.4e5)],
        neighbors=neighbors,
    )
    state = pp.create_state(pos, mass=80.0, radius=radius)
    target = jnp.array([side, side / 2])

    def drift(s):
        direction = target - s.pos
        return 1.3 * direction / jnp.linalg.norm(direction, axis=1, keepdims=True)

    return field, state, dict(dt=0.005, gamma=2.0, drift=drift)


def wca(n, neighbors, density=0.5):
    """3D WCA fluid in a periodic cube, thermostatted at kT = 1."""
    length = (n / density) ** (1.0 / 3.0)
    box = pp.Box((0.0, 0.0, 0.0), (length,) * 3, periodic=True)
    field = pp.ForceField(box, pair=[pp.WCA()], neighbors=neighbors)
    state = pp.create_state(pp.lattice_positions(box.lo, box.hi, n), radius=0.5)
    return field, state, dict(dt=0.005, gamma=1.0, kT=1.0)


def measure(field, state, options, steps, batch=1):
    """Microseconds per step and particle-steps per second."""
    field.check(state)
    state, _ = jax.jit(pp.init_forces, static_argnums=1)(state, field)
    options = dict(options)
    drift = options.pop("drift", None)
    noisy = options.get("kT", 0.0) != 0.0

    def run(s, key):
        def body(s, k):
            s, info = pp.langevin_baoab(
                s,
                field,
                drift=None if drift is None else drift(s),
                key=k if noisy else None,
                **options,
            )
            return s, info.overflow

        s, overflow = jax.lax.scan(body, s, jax.random.split(key, steps))
        return s.pos, overflow.any()

    keys = jax.random.key(0)
    if batch > 1:
        state = jax.tree.map(lambda x: jnp.broadcast_to(x, (batch, *x.shape)), state)
        run = jax.vmap(run)
        keys = jax.random.split(keys, batch)
    run = jax.jit(run)
    pos, overflow = run(state, keys)
    pos.block_until_ready()
    if bool(jnp.any(overflow)):
        raise RuntimeError("cell list overflow; increase the capacity")
    best = np.inf
    for _ in range(3):
        start = time.perf_counter()
        run(state, keys)[0].block_until_ready()
        best = min(best, time.perf_counter() - start)
    per_step = best / steps
    return per_step * 1e6, batch * state.pos.shape[-2] / per_step


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="small cases only")
    args = parser.parse_args()

    cases = [
        ("2D crowd", crowd, 100, pp.BruteForce(), 1, 500),
        ("2D crowd", crowd, 100, pp.BruteForce(), 64, 100),
        ("3D WCA fluid", wca, 1000, pp.CellList(cutoff=1.2, capacity=8), 1, 200),
    ]
    if not args.quick:
        cases += [
            ("2D crowd", crowd, 1000, pp.CellList(cutoff=1.7, capacity=16), 1, 200),
            ("2D crowd", crowd, 10_000, pp.CellList(cutoff=1.7, capacity=16), 1, 20),
            ("3D WCA fluid", wca, 10_000, pp.CellList(cutoff=1.2, capacity=8), 1, 20),
        ]

    print(f"JAX {jax.__version__} on {jax.devices()[0].platform.upper()}")
    print("| system | particles | batch | neighbours | us/step | particle-steps/s |")
    print("|---|---|---|---|---|---|")
    for name, build, n, neighbors, batch, steps in cases:
        field, state, options = build(n, neighbors)
        us, rate = measure(field, state, options, steps, batch)
        kind = type(neighbors).__name__
        print(f"| {name} | {n} | {batch} | {kind} | {us:,.0f} | {rate:,.0f} |", flush=True)


if __name__ == "__main__":
    main()
