"""Temperature and pressure of the DPD fluid.

The fluid has the conservative force a (1 - r) with a = 25, the pair
thermostat with gamma = 4.5 and kT = 1, and the cut-off 1, in a periodic cube.
The script prints

1. the kinetic temperature for several time steps (the thermostat is
   integrated explicitly, so the temperature rises with the time step), at
   density 3 and at density 4, the density of Fig. 1 of Groot and Warren,
   J. Chem. Phys. 107, 4423 (1997), and
2. the pressure for several densities, with the coefficient
   alpha = (p - rho kT) / (a rho^2) of the excess pressure.

The reference output is in ``benchmarks/dpd_fluid.txt``. Run with

    python benchmarks/dpd_fluid.py > benchmarks/dpd_fluid.txt
"""

from functools import partial

import jax
import jax.numpy as jnp

from particlegym import physics as pp

jax.config.update("jax_enable_x64", True)

A = 25.0


def fluid(density, side, dt, seed=0):
    n = round(density * side**3)
    box = pp.Box((0.0,) * 3, (side,) * 3, periodic=True)
    key_pos, key_vel = jax.random.split(jax.random.key(seed))
    state = pp.create_state(side * jax.random.uniform(key_pos, (n, 3)), radius=0.5)
    vel = pp.maxwell_boltzmann(key_vel, state.mass, 1.0, 3)
    state = state._replace(vel=vel - jnp.mean(vel, axis=0))
    pair = [pp.Harmonic(k=A), pp.PairThermostat(gamma=4.5, kT=1.0, dt=dt)]
    return state, pp.ForceField(box, pair=pair)


@partial(jax.jit, static_argnames=("field", "steps", "dt"))
def run(state, field, key, steps, dt):
    """Returns the final state, the kinetic temperatures and the excess pressures."""
    state, _ = pp.init_forces(state, field, key)
    volume = jnp.prod(field.box.length)

    def body(s, k):
        s, _ = pp.velocity_verlet(s, field, dt, key=k)
        # The total momentum is conserved, which removes d degrees of freedom.
        temperature = 2.0 * pp.kinetic_energy(s) / (s.dim * (s.n - 1))
        virial = field.pair_virial(s.pos, s.radius, s.alive)
        return s, (temperature, virial / (s.dim * volume))

    return jax.lax.scan(body, state, jax.random.split(key, steps))


def temperature_against_time_step(
    density=3.0, time_steps=(0.01, 0.02, 0.04), runs=16, discard=500, time=40.0
):
    particles = fluid(density, 4.0, time_steps[0])[0].n
    print(
        f"# Kinetic temperature, density {density:g}, side 4 ({particles} particles), {runs} runs,"
    )
    print(f"# {discard} steps discarded, then {time:g} time units")
    print("# dt  mean  standard_error  lowest  highest")
    for dt in time_steps:
        state, field = fluid(density, 4.0, dt)
        steps = round(time / dt)

        def one(key, state=state, field=field, steps=steps, dt=dt):
            s, _ = run(state, field, key, discard, dt)
            _, (temperature, _) = run(s, field, jax.random.fold_in(key, 1), steps, dt)
            return jnp.mean(temperature)

        out = jax.vmap(one)(jax.random.split(jax.random.key(7), runs))
        error = float(jnp.std(out, ddof=1)) / runs**0.5
        print(
            f"{dt}  {float(out.mean()):.4f}  {error:.4f}  {float(out.min()):.4f}  "
            f"{float(out.max()):.4f}",
            flush=True,
        )


def pressure_against_density(dt=0.01, discard=1000):
    print("\n# Pressure, dt = 0.01, 1000 steps discarded; the error of alpha is the")
    print("# standard error of 20 block means")
    print("# density  side  particles  steps  temperature  pressure  alpha  error_of_alpha")
    cases = [(3.0, 4.0, 8000), (3.0, 6.0, 4000), (4.0, 4.0, 8000), (5.0, 4.0, 8000)]
    cases += [(6.0, 4.0, 8000), (8.0, 4.0, 6000)]
    for density, side, steps in cases:
        state, field = fluid(density, side, dt)
        state, _ = run(state, field, jax.random.key(1), discard, dt)
        _, (temperature, excess) = run(state, field, jax.random.key(2), steps, dt)
        mean_temperature = float(jnp.mean(temperature))
        pressure = density * mean_temperature + float(jnp.mean(excess))
        alpha = excess / (A * density**2)
        blocks = jnp.mean(alpha.reshape(20, -1), axis=1)
        error = float(jnp.std(blocks, ddof=1)) / 20**0.5
        print(
            f"{density:g}  {side:g}  {state.n}  {steps}  {mean_temperature:.4f}  "
            f"{pressure:.3f}  {float(jnp.mean(alpha)):.4f}  {error:.1e}",
            flush=True,
        )


if __name__ == "__main__":
    temperature_against_time_step()
    print()
    temperature_against_time_step(density=4.0, time_steps=(0.04, 0.05))
    pressure_against_density()
