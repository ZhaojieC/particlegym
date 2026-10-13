"""Scalar fields on a grid: interpolation, deposit, diffusion."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from particlegym import physics as pp


def test_linear_field_is_sampled_exactly():
    grid = pp.Grid((0.0, -1.0), (4.0, 2.0), (40, 60))
    centres = grid.centres()
    assert centres.shape == (40, 60, 2)
    np.testing.assert_allclose(centres[0, 0], [0.05, -0.975])
    values = 2.0 * centres[..., 0] - 3.0 * centres[..., 1] + 1.0
    pos = jax.random.uniform(jax.random.key(0), (100, 2), minval=0.2, maxval=1.8)
    expected = 2.0 * pos[:, 0] - 3.0 * pos[:, 1] + 1.0
    np.testing.assert_allclose(grid.sample(values, pos), expected, rtol=1e-12)
    gradient = jax.jit(grid.sample_gradient)(values, pos)
    np.testing.assert_allclose(gradient, np.broadcast_to([2.0, -3.0], (100, 2)), rtol=1e-10)
    # Outside the region the value of the nearest cell is returned.
    outside = grid.sample(values, jnp.array([[-1.0, -5.0]]))
    np.testing.assert_allclose(outside, values[0, 0])


def test_deposit_conserves_the_amount_and_is_the_adjoint_of_sampling():
    grid = pp.Grid((0.0, 0.0, 0.0), (2.0, 3.0, 1.0), (8, 12, 5), periodic=(True, False, True))
    key_pos, key_amount, key_field = jax.random.split(jax.random.key(1), 3)
    pos = jax.random.uniform(key_pos, (50, 3)) * jnp.array([2.0, 3.0, 1.0])
    amount = jax.random.uniform(key_amount, (50,))
    alive = jnp.arange(50) % 5 != 0
    density = grid.deposit(pos, amount, alive)
    assert density.shape == (8, 12, 5)
    np.testing.assert_allclose(
        density.sum() * grid.cell_volume, jnp.sum(jnp.where(alive, amount, 0.0)), rtol=1e-12
    )
    field = jax.random.normal(key_field, grid.shape)
    np.testing.assert_allclose(
        jnp.sum(density * field) * grid.cell_volume,
        jnp.sum(jnp.where(alive, amount * grid.sample(field, pos), 0.0)),
        rtol=1e-12,
    )


def test_point_source_spreads_with_the_diffusion_law():
    grid = pp.Grid((0.0, 0.0), (10.0, 10.0), (100, 100))
    diffusivity, dt, steps = 1.0, 0.001, 400
    np.testing.assert_allclose(grid.max_time_step(diffusivity), 0.0025)
    start = grid.deposit(jnp.array([[5.03, 4.98]]), 2.0)

    @jax.jit
    def run(values):
        return jax.lax.scan(
            lambda v, _: (grid.diffuse(v, diffusivity, dt), None), values, None, length=steps
        )[0]

    final = run(start)
    centres = grid.centres()

    def moments(values):
        total = values.sum() * grid.cell_volume
        mean = jnp.einsum("ij,ijk->k", values, centres) * grid.cell_volume / total
        spread = jnp.einsum("ij,ijk->k", values, (centres - mean) ** 2) * grid.cell_volume / total
        return total, mean, spread

    total, mean, spread = moments(final)
    first_spread = moments(start)[2]
    np.testing.assert_allclose(total, 2.0, rtol=1e-12)
    np.testing.assert_allclose(mean, [5.03, 4.98], rtol=1e-8)
    # The variance along every axis grows by 2 D t.
    np.testing.assert_allclose(spread - first_spread, 2 * diffusivity * dt * steps, rtol=1e-6)
    # The profile is the Gaussian solution of the diffusion equation.
    variance = 2 * diffusivity * dt * steps
    exact = (
        2.0
        / (2 * np.pi * variance)
        * jnp.exp(-jnp.sum((centres - mean) ** 2, axis=-1) / (2 * variance))
    )
    np.testing.assert_allclose(final, exact, atol=0.02 * float(exact.max()))
    with pytest.raises(ValueError, match="stability limit"):
        grid.diffuse(start, diffusivity, 0.003)


def test_closed_and_periodic_boundaries_and_decay():
    closed = pp.Grid((0.0, 0.0), (1.0, 1.0), (10, 10))
    periodic = pp.Grid((0.0, 0.0), (1.0, 1.0), (10, 10), periodic=True)
    start = closed.deposit(jnp.array([[0.04, 0.5]]), 1.0)

    def run(grid, decay=0.0, source=None, steps=2000):
        step = jax.jit(lambda v: grid.diffuse(v, 0.5, 0.002, decay=decay, source=source))
        values = start
        for _ in range(steps):
            values = step(values)
        return values

    # Nothing leaves a closed region; the field becomes uniform.
    np.testing.assert_allclose(run(closed), 1.0, rtol=1e-6)
    np.testing.assert_allclose(run(periodic), 1.0, rtol=1e-6)
    # After a few steps, a periodic grid has passed material through the left edge.
    few = run(periodic, steps=5)
    assert float(few[-1, 5]) > 1e-3 and float(run(closed, steps=5)[-1, 5]) < 1e-12
    # A uniform source balances the decay at source / decay.
    np.testing.assert_allclose(run(closed, decay=2.0, source=3.0, steps=5000), 1.5, rtol=1e-6)


def test_particle_feels_no_gradient_from_its_own_deposit():
    grid = pp.Grid((0.0, 0.0), (6.0, 6.0), (60, 60))
    pos = jnp.array([[2.537, 3.118], [4.0, 3.0]])

    def gradient_at(first_only):
        amount = jnp.array([1.0, 0.0]) if first_only else jnp.array([1.0, 1.0])
        values = jnp.zeros(grid.shape)
        for _ in range(100):
            values = grid.diffuse(values, 1.0, 0.002, decay=1.0, source=grid.deposit(pos, amount))
        return grid.sample_gradient(values, pos)

    alone = gradient_at(True)
    both = gradient_at(False)
    np.testing.assert_allclose(alone[0], 0.0, atol=1e-12)
    # A second source attracts: the gradient points to it.
    assert float(both[0, 0]) > 1e-3 and abs(float(both[0, 1])) < 0.2 * float(both[0, 0])
