"""Dissipative particle dynamics: pair noise, momentum, temperature, pressure, mixtures."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from particlegym import physics as pp
from particlegym.physics.forces import _pair_noise

DT = 0.01


def _fluid(density, a=25.0, side=4.0, kT=1.0, neighbors=None, seed=0, extra=()):
    """A DPD fluid in a periodic cube: particles of radius 0.5, so the cut-off is 1."""
    n = round(density * side**3)
    box = pp.Box((0.0,) * 3, (side,) * 3, periodic=True)
    key_pos, key_vel = jax.random.split(jax.random.key(seed))
    state = pp.create_state(side * jax.random.uniform(key_pos, (n, 3)), radius=0.5)
    vel = pp.maxwell_boltzmann(key_vel, state.mass, kT, 3)
    state = state._replace(vel=vel - jnp.mean(vel, axis=0))
    pair = [pp.Harmonic(k=a), pp.PairThermostat(gamma=4.5, kT=kT, dt=DT), *extra]
    field = pp.ForceField(box, pair=pair, neighbors=neighbors or pp.BruteForce())
    field.check(state)
    return state, field


@partial(jax.jit, static_argnames=("field", "steps"))
def _run(state, field, key, steps):
    state, _ = pp.init_forces(state, field, key)

    def body(s, k):
        s, _ = pp.velocity_verlet(s, field, DT, key=k)
        # The total momentum is conserved, which removes d degrees of freedom.
        kinetic_temperature = 2.0 * pp.kinetic_energy(s) / (s.dim * (s.n - 1))
        return s, (kinetic_temperature, pp.pressure(s, field), pp.total_momentum(s))

    return jax.lax.scan(body, state, jax.random.split(key, steps))


def test_pair_noise_is_symmetric_with_unit_variance():
    n = 600
    noise = _pair_noise(jax.random.key(3), n, None, jnp.float64)
    assert noise.shape == (n, n)
    np.testing.assert_array_equal(noise, noise.T)
    upper = np.asarray(noise)[np.triu_indices(n, 1)]
    np.testing.assert_allclose(upper.mean(), 0.0, atol=0.01)
    np.testing.assert_allclose(upper.var(), 1.0, rtol=0.01)
    assert np.abs(upper).max() <= 3**0.5
    # Neighbouring pairs and successive keys are not correlated.
    other = np.asarray(_pair_noise(jax.random.key(4), n, None, jnp.float64))[np.triu_indices(n, 1)]
    assert abs(np.corrcoef(upper, other)[0, 1]) < 0.01
    assert abs(np.corrcoef(upper[:-1], upper[1:])[0, 1]) < 0.01
    # A list of candidates gives the same numbers as the dense evaluation.
    idx = jax.random.randint(jax.random.key(5), (n, 7), 0, n)
    listed = _pair_noise(jax.random.key(3), n, idx, jnp.float64)
    np.testing.assert_array_equal(listed, jnp.take_along_axis(noise, idx, axis=1))


def test_thermostat_needs_a_key_and_gives_the_same_forces_with_a_cell_list():
    state, dense = _fluid(3.0, side=5.0)
    _, listed = _fluid(3.0, side=5.0, neighbors=pp.CellList(cutoff=1.0, capacity=24))
    assert dense.stochastic and not pp.ForceField(dense.box, pair=[pp.Harmonic()]).stochastic
    with pytest.raises(ValueError, match="random key"):
        pp.init_forces(state, dense)
    key = jax.random.key(11)
    a, _ = pp.init_forces(state, dense, key)
    b, info = pp.init_forces(state, listed, key)
    assert not bool(info.overflow)
    np.testing.assert_allclose(a.force, b.force, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(jnp.sum(a.force, axis=0), 0.0, atol=1e-9)
    # Random forces dominate: another key gives other forces.
    c, _ = pp.init_forces(state, dense, jax.random.key(12))
    assert float(jnp.abs(a.force - c.force).mean()) > 1.0


def test_dpd_fluid_keeps_momentum_and_temperature():
    state, field = _fluid(3.0)

    @jax.vmap
    def run(key):
        s, _ = _run(state, field, key, 500)  # equilibrate
        _, (temperature, _, momentum) = _run(s, field, jax.random.fold_in(key, 1), 2000)
        return jnp.mean(temperature), jnp.max(jnp.abs(momentum))

    temperature, momentum = run(jax.random.split(jax.random.key(1), 8))
    np.testing.assert_allclose(momentum, 0.0, atol=1e-9)
    # Single runs scatter by about 1 %. In benchmarks/dpd_fluid.txt the mean
    # of 16 longer runs is 1.001 +- 0.003 at this time step and rises to
    # 1.026 at a time step of 0.04.
    np.testing.assert_allclose(float(jnp.mean(temperature)), 1.0, rtol=0.02)


def test_pressure_of_an_ideal_gas():
    box = pp.Box((0.0,) * 3, (5.0,) * 3, periodic=True)
    state = pp.create_state(5.0 * jax.random.uniform(jax.random.key(0), (500, 3)))
    state = state._replace(vel=pp.maxwell_boltzmann(jax.random.key(1), state.mass, 2.0, 3))
    density = 500 / 125.0
    np.testing.assert_allclose(
        pp.pressure(state, pp.ForceField(box)), density * pp.temperature(state), rtol=1e-12
    )


@pytest.mark.slow
@pytest.mark.parametrize(("density", "low", "high"), [(3.0, 0.090, 0.094), (5.0, 0.096, 0.100)])
def test_excess_pressure_is_nearly_quadratic_in_the_density(density, low, high):
    a = 25.0
    state, field = _fluid(density, a=a)
    state, _ = _run(state, field, jax.random.key(1), 1000)
    _, (temperature, pressure, _) = _run(state, field, jax.random.key(2), 4000)
    excess = float(jnp.mean(pressure)) - density * float(jnp.mean(temperature))
    # Without correlations between the particles the excess pressure is
    # (pi / 30) a rho^2 = 0.1047 a rho^2. The measured coefficient is lower
    # and rises slowly with the density: 0.092 at rho = 3 (a pressure of
    # 23.65) and 0.098 at rho = 5 (benchmarks/dpd_fluid.txt).
    assert low < excess / (a * density**2) < high < np.pi / 30


def test_mixture_scales_the_interaction_of_unlike_particles():
    box = pp.Box((0.0, 0.0), (6.0, 6.0), periodic=True)
    mixture = pp.Mixture(pp.Harmonic(k=25.0), [[1.0, 1.6], [1.6, 1.0]])
    pos = jnp.array([[1.0, 1.0], [1.6, 1.0], [4.0, 4.0], [4.6, 4.0]])
    kind = jnp.array([0, 0, 0, 1])
    state = pp.create_state(pos, radius=0.5, kind=kind)
    field = pp.ForceField(box, pair=[mixture])
    state, info = pp.init_forces(state, field)
    np.testing.assert_allclose(state.force[0], [-25.0 * 0.4, 0.0], rtol=1e-12)
    np.testing.assert_allclose(state.force[2], 1.6 * state.force[0], rtol=1e-12)
    np.testing.assert_allclose(info.potential_energy, 2.6 * 0.5 * 25.0 * 0.4**2, rtol=1e-12)
    with pytest.raises(ValueError, match="species"):
        field.forces(state.pos, state.vel, state.radius, state.alive)
    with pytest.raises(ValueError, match="symmetric"):
        pp.Mixture(pp.Harmonic(), [[1.0, 2.0], [3.0, 1.0]])

    # Random mixture: forces are the gradient of the energy, for both searches.
    n = 200
    key_pos, key_kind = jax.random.split(jax.random.key(0))
    dense = pp.create_state(
        6.0 * jax.random.uniform(key_pos, (n, 2)),
        radius=0.5,
        kind=jax.random.randint(key_kind, (n,), 0, 2),
    )
    listed = pp.ForceField(box, pair=[mixture], neighbors=pp.CellList(cutoff=1.0, capacity=24))
    force, _ = pp.init_forces(dense, field)
    energy = partial(field.potential_energy, kind=dense.kind)
    grad = jax.grad(energy)(dense.pos, dense.radius, dense.alive)
    np.testing.assert_allclose(force.force, -grad, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(pp.init_forces(dense, listed)[0].force, force.force, atol=1e-9)
    np.testing.assert_allclose(
        field.pair_virial(dense.pos, dense.radius, dense.alive, dense.kind),
        listed.pair_virial(dense.pos, dense.radius, dense.alive, dense.kind),
    )


def test_two_species_with_a_stronger_unlike_repulsion_separate():
    side, dt, steps = 8.0, 0.02, 750
    n = round(3.0 * side**2)
    box = pp.Box((0.0, 0.0), (side, side), periodic=True)
    kind = jnp.arange(n) % 2

    def like_fraction(pos):
        offset = box.displacement(pos[:, None, :], pos[None, :, :])
        near = (jnp.sum(offset**2, axis=-1) < 1.0) & ~jnp.eye(n, dtype=bool)
        return jnp.sum(near & (kind[:, None] == kind[None, :])) / jnp.sum(near)

    def final_fraction(unlike):
        key_pos, key_vel, key_run = jax.random.split(jax.random.key(0), 3)
        state = pp.create_state(side * jax.random.uniform(key_pos, (n, 2)), radius=0.5, kind=kind)
        state = state._replace(vel=pp.maxwell_boltzmann(key_vel, state.mass, 1.0, 2))
        repulsion = pp.Mixture(pp.Harmonic(k=1.0), [[25.0, unlike], [unlike, 25.0]])
        field = pp.ForceField(box, pair=[repulsion, pp.PairThermostat(4.5, 1.0, dt)])
        state, _ = pp.init_forces(state, field, key_run)
        start = like_fraction(state.pos)

        @jax.jit
        def run(state):
            def body(s, k):
                return pp.velocity_verlet(s, field, dt, key=k)[0], None

            return jax.lax.scan(body, state, jax.random.split(key_run, steps))[0]

        return float(start), float(like_fraction(run(state).pos))

    start, same = final_fraction(25.0)
    _, separated = final_fraction(60.0)
    # A random mixture has half of its neighbour pairs of the same species.
    # Reference run of the gallery script (1200 particles): 0.50 at the start, 0.91 at time 25.
    assert abs(start - 0.5) < 0.05 and abs(same - 0.5) < 0.05
    assert separated > 0.8
