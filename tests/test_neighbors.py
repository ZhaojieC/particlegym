"""The cell list must find exactly the interactions that brute force finds."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from particlegym import physics as pp


def _random_system(dim, periodic, seed, n=300):
    rng = np.random.default_rng(seed)
    hi = np.full(dim, 6.0 if dim == 3 else 12.0)
    pos = rng.uniform(0.0, hi, size=(n, dim))
    if not periodic:
        # Some particles outside the box, e.g. pedestrians that walked out of a door.
        pos[:20] = rng.uniform(-3.0, hi + 3.0, size=(20, dim))
    box = pp.Box(np.zeros(dim), hi, periodic=periodic)
    state = pp.create_state(
        pos,
        vel=rng.normal(size=(n, dim)),
        radius=rng.uniform(0.3, 0.5, size=n),
        alive=rng.random(n) > 0.1,
    )
    return box, state


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("periodic", [True, False])
def test_cell_list_forces_match_brute_force(dim, periodic):
    box, state = _random_system(dim, periodic, seed=0)
    pair = [pp.Harmonic(k=10.0), pp.TangentialFriction(kappa=3.0)]
    brute = pp.ForceField(box, pair=pair)
    cells = pp.ForceField(box, pair=pair, neighbors=pp.CellList(cutoff=1.0, capacity=32))
    cells.check(state)
    args = (state.pos, state.vel, state.radius, state.alive)

    force_brute, info_brute = jax.jit(brute.forces)(*args)
    force_cells, info_cells = jax.jit(cells.forces)(*args)

    assert not bool(info_cells.overflow)
    assert float(jnp.abs(force_brute).max()) > 1.0  # the configuration does interact
    np.testing.assert_allclose(force_cells, force_brute, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        info_cells.potential_energy, info_brute.potential_energy, rtol=1e-12
    )


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("periodic", [True, False])
def test_cell_list_candidates_cover_all_close_pairs_once(dim, periodic):
    box, state = _random_system(dim, periodic, seed=1)
    cutoff = 1.0
    cell_list = pp.CellList(cutoff=cutoff, capacity=32)
    idx, overflow = jax.jit(lambda p, a: cell_list.candidates(p, a, box))(state.pos, state.alive)
    assert not bool(overflow)
    idx = np.asarray(idx)
    alive = np.asarray(state.alive)
    n = state.n
    dr = np.asarray(box.displacement(state.pos[:, None, :], state.pos[None, :, :]))
    dist = np.linalg.norm(dr, axis=-1)
    for i in np.flatnonzero(alive):
        row = idx[i][idx[i] < n]
        assert len(row) == len(set(row.tolist())), "duplicate candidates"
        assert i not in row
        assert alive[row].all(), "dead particles must not be candidates"
        close = np.flatnonzero((dist[i] < cutoff) & alive & (np.arange(n) != i))
        assert set(close.tolist()) <= set(row.tolist())


def test_cell_list_reports_overflow():
    box = pp.Box((0.0, 0.0), (10.0, 10.0))
    pos = 5.0 + 0.01 * np.arange(5)[:, None] * np.ones((5, 2))
    state = pp.create_state(pos, radius=0.1)
    _, overflow = pp.CellList(cutoff=1.0, capacity=4).candidates(state.pos, state.alive, box)
    assert bool(overflow)
    _, overflow = pp.CellList(cutoff=1.0, capacity=5).candidates(state.pos, state.alive, box)
    assert not bool(overflow)


def test_cell_list_rejects_small_periodic_box():
    box = pp.Box((0.0, 0.0), (2.5, 10.0), periodic=True)
    with pytest.raises(ValueError, match="too small"):
        pp.ForceField(box, pair=[pp.Harmonic()], neighbors=pp.CellList(cutoff=1.0))


def test_check_detects_cutoff_shorter_than_interaction_range():
    box = pp.Box((0.0, 0.0), (10.0, 10.0))
    field = pp.ForceField(box, pair=[pp.SocialForce()], neighbors=pp.CellList(cutoff=1.0))
    state = pp.create_state(np.full((1, 2), 5.0), radius=0.3)
    with pytest.raises(ValueError, match="shorter than the interaction range"):
        field.check(state)


def test_check_detects_range_beyond_half_periodic_box():
    box = pp.Box((0.0, 0.0), (3.0, 3.0), periodic=True)
    field = pp.ForceField(box, pair=[pp.SocialForce()])
    state = pp.create_state(np.full((1, 2), 1.0), radius=0.3)
    with pytest.raises(ValueError, match="half the periodic box"):
        field.check(state)
