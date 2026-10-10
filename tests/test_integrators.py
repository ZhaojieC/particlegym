"""Integrators: energy and momentum conservation, thermostat, diffusion, drift."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from particlegym import physics as pp


def _wca_fluid(kT=1.0, density=0.6, n_side=5, seed=0):
    """Lattice start of a WCA fluid in a periodic cube."""
    n = n_side**3
    length = (n / density) ** (1.0 / 3.0)
    box = pp.Box((0.0, 0.0, 0.0), (length,) * 3, periodic=True)
    state = pp.create_state(pp.lattice_positions(box.lo, box.hi, n), radius=0.5)
    state = state._replace(vel=pp.maxwell_boltzmann(jax.random.key(seed), state.mass, kT, 3))
    field = pp.ForceField(box, pair=[pp.WCA()], neighbors=pp.CellList(cutoff=1.2, capacity=16))
    field.check(state)
    state, _ = pp.init_forces(state, field)
    return state, field


@partial(jax.jit, static_argnames=("field", "dt", "steps"))
def _nve(state, field, dt, steps):
    def body(s, _):
        s, info = pp.velocity_verlet(s, field, dt)
        energy = pp.kinetic_energy(s) + info.potential_energy
        return s, (energy, pp.total_momentum(s), info.overflow)

    return jax.lax.scan(body, state, None, length=steps)


@pytest.fixture(scope="module")
def melted_fluid():
    state, field = _wca_fluid()
    state, _ = _nve(state, field, 0.004, 1000)  # melt the lattice
    return state, field


def test_velocity_verlet_conserves_energy_and_momentum(melted_fluid):
    state, field = melted_fluid
    _, (energy, momentum, overflow) = _nve(state, field, 0.004, 2500)
    assert not bool(overflow.any())
    kinetic = 1.5 * state.n  # order of magnitude of the kinetic energy (kT = 1)
    assert float(energy.max() - energy.min()) / kinetic < 2e-3
    # No drift: the first and last fifths of the run have the same mean energy.
    np.testing.assert_allclose(energy[:500].mean(), energy[-500:].mean(), atol=2e-4 * kinetic)
    np.testing.assert_allclose(momentum, 0.0, atol=1e-10)


def test_velocity_verlet_energy_error_scales_with_dt_squared(melted_fluid):
    state, field = melted_fluid
    _, (coarse, _, _) = _nve(state, field, 0.004, 1000)
    _, (fine, _, _) = _nve(state, field, 0.002, 2000)
    ratio = float(jnp.std(coarse) / jnp.std(fine))
    assert 3.0 < ratio < 5.0, ratio


def test_baoab_without_friction_is_velocity_verlet(melted_fluid):
    state, field = melted_fluid
    vv, _ = pp.velocity_verlet(state, field, 0.004)
    baoab, _ = pp.langevin_baoab(state, field, 0.004, gamma=0.0)
    np.testing.assert_allclose(baoab.pos, vv.pos, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(baoab.vel, vv.vel, rtol=1e-12, atol=1e-12)


def _free_particles(n, dim, mass, pos=None, vel=None):
    box = pp.Box((-1.0,) * dim, (1.0,) * dim)
    pos = jnp.zeros((n, dim)) if pos is None else pos
    return pp.create_state(pos, vel=vel, mass=mass), pp.ForceField(box)


@pytest.mark.slow
def test_langevin_thermostat_samples_maxwell_boltzmann():
    kT, mass, gamma, dt = 1.5, 2.0, 1.0, 0.05
    state, field = _free_particles(4000, 3, mass)

    def body(s, key):
        s, _ = pp.langevin_baoab(s, field, dt, gamma=gamma, kT=kT, key=key)
        return s, (jnp.mean(s.mass[:, None] * s.vel**2), jnp.mean(s.vel**4))

    keys = jax.random.split(jax.random.key(1), 800)
    _, (second, fourth) = jax.jit(lambda s: jax.lax.scan(body, s, keys))(state)
    # Discard the first 400 steps (20 relaxation times) as equilibration.
    np.testing.assert_allclose(second[400:].mean(), kT, rtol=0.02)
    np.testing.assert_allclose(fourth[400:].mean(), 3.0 * (kT / mass) ** 2, rtol=0.05)


@pytest.mark.slow
def test_langevin_thermostat_with_interactions(melted_fluid):
    state, field = melted_fluid
    kT = 1.3

    def body(s, key):
        s, info = pp.langevin_baoab(s, field, 0.004, gamma=1.0, kT=kT, key=key)
        return s, (pp.temperature(s), info.overflow)

    keys = jax.random.split(jax.random.key(2), 12_000)
    _, (temperature, overflow) = jax.jit(lambda s: jax.lax.scan(body, s, keys))(state)
    assert not bool(overflow.any())
    # 125 particles fluctuate by several percent, so compare with the statistical
    # error of the mean, estimated from 20 blocks after 2000 equilibration steps.
    blocks = np.asarray(temperature[2000:]).reshape(20, -1).mean(axis=1)
    error = blocks.std(ddof=1) / np.sqrt(len(blocks))
    assert error < 0.02 * kT  # the run is long enough to be a meaningful test
    assert abs(blocks.mean() - kT) < 4.0 * error, (blocks.mean(), error)


@pytest.mark.slow
def test_free_diffusion_matches_exact_ornstein_uhlenbeck_msd():
    n, dim, kT, mass, gamma, dt, steps = 4000, 3, 1.0, 1.0, 2.0, 0.01, 1000
    vel = pp.maxwell_boltzmann(jax.random.key(3), jnp.full(n, mass), kT, dim, remove_drift=False)
    state, field = _free_particles(n, dim, mass, vel=vel)

    def body(s, key):
        s, _ = pp.langevin_baoab(s, field, dt, gamma=gamma, kT=kT, key=key)
        return s, jnp.mean(jnp.sum(s.pos**2, axis=1))

    keys = jax.random.split(jax.random.key(4), steps)
    _, msd = jax.jit(lambda s: jax.lax.scan(body, s, keys))(state)

    t = dt * np.arange(1, steps + 1)
    diffusion = kT / (mass * gamma)  # Einstein relation
    exact = 2 * dim * diffusion * (t - (1.0 - np.exp(-gamma * t)) / gamma)
    for time in (0.1, 0.5, 2.0, 10.0):
        i = round(time / dt) - 1
        np.testing.assert_allclose(msd[i], exact[i], rtol=0.05, err_msg=f"t = {time}")


def test_drift_relaxation_is_exact_without_noise():
    u = jnp.array([[1.5, -0.5]])
    v_start = jnp.array([[0.0, 2.0]])
    gamma, dt, steps = 3.0, 0.01, 200
    state, field = _free_particles(1, 2, 80.0, vel=v_start)

    def body(s, _):
        s, _ = pp.langevin_baoab(s, field, dt, gamma=gamma, drift=u)
        return s, None

    state, _ = jax.lax.scan(body, state, None, length=steps)
    t = dt * steps
    np.testing.assert_allclose(state.vel, u + (v_start - u) * np.exp(-gamma * t), rtol=1e-12)
    exact_pos = u * t + (v_start - u) * (1.0 - np.exp(-gamma * t)) / gamma
    np.testing.assert_allclose(state.pos, exact_pos, rtol=1e-3)


def test_noise_requires_a_key():
    state, field = _free_particles(2, 2, 1.0)
    with pytest.raises(ValueError, match="random key"):
        pp.langevin_baoab(state, field, 0.01, gamma=1.0, kT=1.0)


def test_dead_particles_are_frozen_and_do_not_interact():
    box = pp.Box((-5.0, -5.0), (5.0, 5.0))
    field = pp.ForceField(box, pair=[pp.Harmonic(k=100.0)])
    state = pp.create_state(
        [[0.0, 0.0], [0.3, 0.0]], vel=[[0.1, 0.0], [0.0, 1.0]], alive=[True, False]
    )
    state, _ = pp.init_forces(state, field)
    np.testing.assert_allclose(state.force, 0.0)
    for _ in range(10):
        state, _ = pp.velocity_verlet(state, field, 0.01)
    np.testing.assert_allclose(state.pos[1], [0.3, 0.0])
    np.testing.assert_allclose(state.vel[1], [0.0, 0.0])
    np.testing.assert_allclose(state.pos[0], [0.01, 0.0], rtol=1e-12)


def test_float32_states_stay_float32():
    rng = np.random.default_rng(0)
    pos = rng.uniform(1.0, 9.0, size=(64, 3)).astype(np.float32)
    box = pp.Box((0.0, 0.0, 0.0), (10.0, 10.0, 10.0))
    field = pp.ForceField(
        box,
        pair=[pp.WCA(), pp.TangentialFriction(kappa=1.0)],
        walls=[pp.Walls(pp.box_walls(box.lo, box.hi), pp.Harmonic(k=10.0), friction=1.0)],
        neighbors=pp.CellList(cutoff=1.2),
    )
    state, _ = pp.init_forces(pp.create_state(pos, radius=0.3), field)
    step = jax.jit(partial(pp.langevin_baoab, force_field=field, dt=0.002, gamma=1.0, kT=1.0))
    key = jax.random.key(0)
    for _ in range(5):
        key, subkey = jax.random.split(key)
        state, info = step(state, key=subkey)
    for leaf in (state.pos, state.vel, state.force, info.potential_energy):
        assert leaf.dtype == jnp.float32
        assert bool(jnp.all(jnp.isfinite(leaf)))
