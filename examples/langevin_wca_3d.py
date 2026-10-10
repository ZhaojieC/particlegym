"""A WCA fluid in a periodic 3D box, thermostatted with Langevin dynamics.

Prints the kinetic temperature while the fluid equilibrates and writes a
trajectory that OVITO can open.

    python examples/langevin_wca_3d.py --particles 1000 --kT 1.0
"""

import argparse

import jax

from particlegym import physics as pp
from particlegym.io import write_extxyz


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--particles", type=int, default=1000)
    parser.add_argument("--density", type=float, default=0.5)
    parser.add_argument("--kT", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--output", default="wca_3d.xyz")
    args = parser.parse_args()

    length = (args.particles / args.density) ** (1.0 / 3.0)
    box = pp.Box((0.0, 0.0, 0.0), (length,) * 3, periodic=True)
    state = pp.create_state(pp.lattice_positions(box.lo, box.hi, args.particles), radius=0.5)
    field = pp.ForceField(box, pair=[pp.WCA()], neighbors=pp.CellList(cutoff=1.2, capacity=16))
    field.check(state)
    state, _ = pp.init_forces(state, field)

    dt, steps_per_frame = 0.005, 200

    @jax.jit
    def advance(state, key):
        def body(s, k):
            s, info = pp.langevin_baoab(s, field, dt, gamma=1.0, kT=args.kT, key=k)
            return s, info.overflow

        state, overflow = jax.lax.scan(body, state, jax.random.split(key, steps_per_frame))
        return state, overflow.any()

    open(args.output, "w").close()
    key = jax.random.key(0)
    for frame in range(1, args.steps // steps_per_frame + 1):
        key, subkey = jax.random.split(key)
        state, overflow = advance(state, subkey)
        if overflow:
            raise RuntimeError("a cell list cell overflowed; increase its capacity")
        time = frame * steps_per_frame * dt
        write_extxyz(args.output, state, box, time=time)
        print(f"t = {time:5.1f}   kT = {float(pp.temperature(state)):.3f}")
    print(f"trajectory written to {args.output}")


if __name__ == "__main__":
    main()
