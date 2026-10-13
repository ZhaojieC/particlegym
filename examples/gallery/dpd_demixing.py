"""Animation: two fluids that do not mix separate (dissipative particle dynamics).

A two-dimensional DPD fluid of two species in a periodic box. Particles of
the same species repel with a = 25, particles of different species with a
larger value, so the mixture separates into domains that grow with time. The
pair thermostat (gamma = 4.5, kT = 1) conserves momentum, so the growing
domains drive a flow. The script prints the kinetic temperature with and
without this flow (the mean velocity in boxes of side 2).

    python examples/gallery/dpd_demixing.py --output dpd_demixing.gif
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from particlegym import physics as pp
from particlegym import viz


def like_fraction(pos, kind, box):
    """Fraction of the pairs closer than the cut-off 1 that are of the same species."""
    offset = box.displacement(pos[:, None, :], pos[None, :, :])
    near = (jnp.sum(offset**2, axis=-1) < 1.0) & ~jnp.eye(pos.shape[0], dtype=bool)
    same = kind[:, None] == kind[None, :]
    return jnp.sum(near & same) / jnp.sum(near)


def temperatures(state, box, bins):
    """Kinetic temperature, and the same without the mean velocity in ``bins`` x ``bins`` boxes."""
    n, dim = state.pos.shape
    kinetic = 2.0 * pp.kinetic_energy(state) / (dim * (n - 1))
    cell = jnp.clip((state.pos / box.length * bins).astype(jnp.int32), 0, bins - 1)
    index = cell[:, 0] * bins + cell[:, 1]
    count = jnp.zeros(bins * bins).at[index].add(1.0)
    flow = (
        jnp.zeros((bins * bins, dim)).at[index].add(state.vel) / jnp.maximum(count, 1.0)[:, None]
    )
    thermal = jnp.sum((state.vel - flow[index]) ** 2) / (dim * (n - bins * bins))
    return kinetic, thermal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--side", type=float, default=20.0)
    parser.add_argument("--density", type=float, default=3.0)
    parser.add_argument("--unlike", type=float, default=60.0, help="a between the species")
    parser.add_argument("--time", type=float, default=60.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=280)
    parser.add_argument("--output", default="dpd_demixing.gif")
    args = parser.parse_args()

    n = round(args.density * args.side**2)
    box = pp.Box((0.0, 0.0), (args.side, args.side), periodic=True)
    key_pos, key_vel, key_run = jax.random.split(jax.random.key(args.seed), 3)
    kind = jnp.arange(n) % 2
    state = pp.create_state(args.side * jax.random.uniform(key_pos, (n, 2)), radius=0.5, kind=kind)
    vel = pp.maxwell_boltzmann(key_vel, state.mass, 1.0, 2)
    state = state._replace(vel=vel - jnp.mean(vel, axis=0))
    repulsion = pp.Mixture(pp.Harmonic(k=1.0), [[25.0, args.unlike], [args.unlike, 25.0]])
    thermostat = pp.PairThermostat(gamma=4.5, kT=1.0, dt=args.dt)
    field = pp.ForceField(
        box, pair=[repulsion, thermostat], neighbors=pp.CellList(cutoff=1.0, capacity=24)
    )
    field.check(state)
    state, _ = pp.init_forces(state, field, key_run)
    stride = max(1, round(args.time / args.dt / args.frames))
    bins = max(1, round(args.side / 2.0))

    @jax.jit
    def advance(state, key):
        def body(s, k):
            s, info = pp.velocity_verlet(s, field, args.dt, key=k)
            return s, (info.overflow, *temperatures(s, box, bins))

        state, (overflow, kinetic, thermal) = jax.lax.scan(
            body, state, jax.random.split(key, stride)
        )
        fraction = like_fraction(state.pos, state.kind, box)
        return state, jnp.any(overflow), jnp.mean(kinetic), jnp.mean(thermal), fraction

    def picture(state):
        canvas = viz.Canvas(box.lo, box.hi, args.width)
        colors = np.where(np.asarray(state.kind)[:, None] == 0, viz.RED, viz.BLUE)
        canvas.discs(state.pos, 0.33, colors)
        return canvas.frame()

    frames = [picture(state)]
    start = float(like_fraction(state.pos, state.kind, box))
    print(f"{n} particles; like-neighbour fraction at the start {start:.3f}")
    block = args.frames // 5
    kinetic, thermal = [], []
    for i in range(args.frames):
        state, overflow, t_kinetic, t_thermal, fraction = advance(
            state, jax.random.fold_in(key_run, i)
        )
        if bool(overflow):
            raise RuntimeError("the cell list is too small; raise its capacity")
        frames.append(picture(state))
        kinetic.append(float(t_kinetic))
        thermal.append(float(t_thermal))
        if (i + 1) % block == 0:
            print(
                f"time {(i + 1) * stride * args.dt:.0f}: like-neighbour fraction "
                f"{float(fraction):.3f}; mean temperature since the last line "
                f"{np.mean(kinetic[-block:]):.3f}, without flow {np.mean(thermal[-block:]):.3f}"
            )
    viz.save_gif(args.output, frames, fps=20, colors=16)
    print(f"{len(frames)} frames written to {args.output}")


if __name__ == "__main__":
    main()
