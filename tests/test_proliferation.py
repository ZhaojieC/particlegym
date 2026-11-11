"""Cells that divide and die: free slots, the placement of daughters, compression."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from particlegym import physics as pp

BOX = pp.Box((-20.0, -20.0), (20.0, 20.0))


def test_daughters_are_born_into_free_slots_while_they_last():
    alive = jnp.array([True, True, False, True, False, False])
    ready = jnp.array([True, False, True, True, False, False])  # slot 2 is empty
    divides, mother = pp.birth_slots(alive, ready)
    np.testing.assert_array_equal(divides, [True, False, False, True, False, False])
    np.testing.assert_array_equal(mother, [-1, -1, 0, -1, 3, -1])
    # One free slot for three ready cells: the first divides, the others wait.
    alive = jnp.array([True, True, True, False])
    divides, mother = pp.birth_slots(alive, jnp.array([False, True, True, False]))
    np.testing.assert_array_equal(divides, [False, True, False, False])
    np.testing.assert_array_equal(mother, [-1, -1, -1, 1])


@pytest.mark.parametrize("d", [2, 3])
def test_division_keeps_the_volume_and_the_centre(d):
    pos = jnp.zeros((3, d)).at[0].set(jnp.arange(1.0, d + 1.0))
    radius = jnp.array([0.5, 0.4, 0.3])
    alive = jnp.array([True, True, False])
    divides, mother = pp.birth_slots(alive, jnp.array([True, False, False]))
    direction = jnp.zeros((3, d)).at[:, 0].set(1.0)
    new_pos, new_radius, new_alive = pp.divide(pos, radius, alive, divides, mother, direction)
    np.testing.assert_array_equal(new_alive, [True, True, True])
    np.testing.assert_allclose(new_pos[0], pos[0] - jnp.eye(d)[0] * 0.25)
    np.testing.assert_allclose(new_pos[2], pos[0] + jnp.eye(d)[0] * 0.25)
    np.testing.assert_allclose(0.5 * (new_pos[0] + new_pos[2]), pos[0])  # the centre stays
    np.testing.assert_allclose(2 * new_radius[0] ** d, radius[0] ** d)  # the volume is shared
    np.testing.assert_allclose(np.asarray(new_radius)[[0, 2]], radius[0] * 2 ** (-1 / d))
    np.testing.assert_allclose(new_pos[1], pos[1])  # others are not touched
    assert float(new_radius[1]) == 0.4


def test_overlap_sum_measures_compression():
    pos = jnp.array([[0.0, 0.0], [0.8, 0.0], [-0.9, 0.0], [5.0, 5.0], [0.0, 0.7]])
    radius = jnp.full(5, 0.5)
    alive = jnp.array([True, True, True, True, False])  # the last cell is dead
    total = pp.overlap_sum(pos, radius, alive, BOX)
    np.testing.assert_allclose(total, [0.2 + 0.1, 0.2, 0.1, 0.0, 0.0], atol=1e-12)
    # The forces on the middle cell nearly cancel, its compression does not.
    many = jax.random.uniform(jax.random.key(0), (300, 2), minval=-8, maxval=8)
    alive = jnp.ones(300, bool).at[::7].set(False)
    radius = jnp.full(300, 0.5)
    cells = pp.CellList(cutoff=1.0, capacity=12)
    np.testing.assert_allclose(
        pp.overlap_sum(many, radius, alive, BOX, cells),
        pp.overlap_sum(many, radius, alive, BOX),
        atol=1e-12,
    )
