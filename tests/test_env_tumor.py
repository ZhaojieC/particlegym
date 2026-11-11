"""Tumour therapy: the start, drug and turnover rates, contact inhibition, the therapies."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from particlegym import envs
from particlegym import physics as pp


def _rollout(env, params, policy, days, key=0):
    """Run ``days`` compiled steps from a reset; ``policy(state)`` gives the actions."""
    _, state = env.reset(jax.random.key(key), params)

    def body(s, k):
        obs, s, reward, terminated, truncated, info = env.step(k, s, policy(s), params)
        return s, (obs, reward, terminated, truncated, info)

    keys = jax.random.split(jax.random.key(key + 1000), days)
    return jax.jit(lambda s: jax.lax.scan(body, s, keys))(state)


def test_reset_puts_a_round_tumour_of_mostly_sensitive_cells():
    env = envs.TumorTherapy()
    obs, state = env.reset(jax.random.key(0))
    cells = state.cells
    alive = np.asarray(cells.alive)
    assert alive.sum() == 200 and not alive[200:].any()
    np.testing.assert_allclose(obs, [[1.0, 0.0, 0.0]])
    resistant = int(np.sum(alive & (np.asarray(cells.kind) == 1)))
    assert 0 < resistant < 25  # 5 % of 200 cells
    distance = np.linalg.norm(np.asarray(cells.pos)[alive], axis=1)
    assert distance.max() < 0.85 * math.sqrt(200 / math.pi * 2 / math.sqrt(3)) + 0.5
    age, cycle = np.asarray(state.age)[alive], np.asarray(state.cycle)[alive]
    assert np.all((age >= 0) & (age < cycle))
    radius = np.asarray(cells.radius)[alive]
    assert np.all((radius > 0.5 / math.sqrt(2) - 1e-12) & (radius < 0.5 + 1e-12))


@pytest.mark.parametrize("dose", [1.0, 0.5])
def test_the_drug_kills_cycling_sensitive_cells_at_its_rate(dose):
    # No divisions, no contact inhibition, no other deaths: within one day a
    # sensitive cell survives with probability exp(-kill_rate * dose * 24).
    env = envs.TumorTherapy(initial_cells=400, capacity=450, cycle_time=1e6, inhibition=1e9)
    params = env.default_params._replace(death_rate=0.0, resistant_fraction=0.5)
    _, state = env.reset(jax.random.key(1), params)
    alive, kind = np.asarray(state.cells.alive), np.asarray(state.cells.kind)
    sensitive, resistant = np.sum(alive & (kind == 0)), np.sum(alive & (kind == 1))
    *_, info = env.step(jax.random.key(2), state, jnp.array([[dose]]), params)
    survival = math.exp(-params.kill_rate * dose * 24)
    expected = sensitive * survival
    sigma = math.sqrt(sensitive * survival * (1 - survival))
    assert abs(int(info["sensitive"]) - expected) < 4 * sigma
    assert int(info["resistant"]) == resistant  # the drug spares resistant cells


def test_every_cell_dies_at_the_turnover_rate():
    env = envs.TumorTherapy(initial_cells=400, capacity=450, cycle_time=1e6, inhibition=1e9)
    params = env.default_params._replace(resistant_fraction=0.5)
    _, state = env.reset(jax.random.key(3), params)
    _, _, *_, info = env.step(jax.random.key(4), state, jnp.zeros((1, 1)), params)
    survival = math.exp(-params.death_rate * 24)
    sigma = math.sqrt(400 * survival * (1 - survival))
    assert abs(int(info["sensitive"]) + int(info["resistant"]) - 400 * survival) < 4 * sigma


def test_contact_inhibition_confines_cycling_to_the_rim():
    env = envs.TumorTherapy()
    params = env.default_params._replace(progression=10.0)
    state, (_, _, _, _, info) = _rollout(env, params, env.no_treatment_policy, 8)
    cells = state.cells
    pressure = pp.overlap_sum(cells.pos, cells.radius, cells.alive, env.box, env.neighbors)
    cycling = np.asarray(cells.alive & (pressure < env.inhibition))
    resting = np.asarray(cells.alive) & ~cycling
    alive = np.asarray(cells.alive)
    centre = np.asarray(cells.pos)[alive].mean(axis=0)
    distance = np.linalg.norm(np.asarray(cells.pos) - centre, axis=1)
    assert 0.1 < cycling.sum() / alive.sum() < 0.7
    assert distance[cycling].mean() > distance[resting].mean() + 1.0  # cycling cells sit outside
    assert int(info["cycling"][-1]) > 0


def test_untreated_tumours_progress_within_days_and_end_the_episode():
    env = envs.TumorTherapy()
    params = env.default_params
    _, (obs, reward, terminated, truncated, _) = _rollout(env, params, env.no_treatment_policy, 15)
    day = int(np.argmax(np.asarray(terminated[:, 0])))
    assert bool(terminated[day, 0]) and day < 10
    assert float(obs[day, 0, 0]) > 1.2  # the tumour has grown past 1.2 times its start
    np.testing.assert_allclose(reward[: day + 1, 0], 1.0)  # a reward for every day...
    np.testing.assert_allclose(reward[day + 1 :, 0], 0.0)  # ...before progression
    assert not np.asarray(truncated).any()


def test_maximum_dose_selects_resistant_cells_and_adaptive_therapy_treats_less():
    env = envs.TumorTherapy()
    params = env.default_params
    for policy in (env.mtd_policy, env.adaptive_policy):
        _, (obs, _, terminated, _, info) = _rollout(env, params, policy, 120)
        day = int(np.argmax(np.asarray(terminated[:, 0])))
        assert bool(terminated[day, 0]) and day > 10  # treatment delays progression
        treated = float(jnp.mean(obs[: day + 1, 0, 2]))
        if policy == env.mtd_policy:
            assert treated == 1.0
            # At progression the tumour consists mostly of resistant cells.
            assert int(info["resistant"][day]) > int(info["sensitive"][day])
        else:
            assert treated < 0.8


def test_the_adaptive_rule_pauses_at_half_and_resumes_at_the_start():
    env = envs.TumorTherapy()
    _, state = env.reset(jax.random.key(0))
    cases = [(1.0, 120, 1.0), (1.0, 95, 0.0), (0.0, 190, 0.0), (0.0, 200, 1.0)]
    for dose, cells, expected in cases:
        alive = jnp.arange(env.capacity) < cells
        s = state._replace(cells=state.cells._replace(alive=alive), dose=jnp.asarray(dose))
        np.testing.assert_allclose(env.adaptive_policy(s), [[expected]])
    # Containment: treat whenever the tumour has grown back to its initial size.
    cases = [(1.0, 202, 1.0), (1.0, 198, 0.0), (0.0, 198, 0.0), (0.0, 200, 1.0)]
    for dose, cells, expected in cases:
        alive = jnp.arange(env.capacity) < cells
        s = state._replace(cells=state.cells._replace(alive=alive), dose=jnp.asarray(dose))
        np.testing.assert_allclose(env.adaptive_policy(s, stop=1.0, restart=1.0), [[expected]])


def test_discrete_actions_are_doses():
    env = envs.TumorTherapy(initial_cells=50, capacity=100, action="discrete")
    params = env.default_params
    _, state = env.reset(jax.random.key(0), params)
    for index, dose in enumerate(env.levels):
        _, new, *_ = env.step(jax.random.key(1), state, jnp.array([index]), params)
        assert float(new.dose) == dose
    assert int(env.mtd_policy(state)[0]) == 2 and int(env.no_treatment_policy(state)[0]) == 0


def test_reset_and_step_compile_and_batch_over_parameters():
    env = envs.TumorTherapy(initial_cells=50, capacity=100)
    params = jax.tree.map(lambda x: jnp.full(3, x), env.default_params)
    params = params._replace(resistant_fraction=jnp.array([0.0, 0.1, 0.5]))
    keys = jax.random.split(jax.random.key(5), 3)
    obs, states = jax.jit(jax.vmap(env.reset))(keys, params)
    assert obs.shape == (3, 1, 3)
    resistant = jnp.sum(states.cells.alive & (states.cells.kind == 1), axis=1)
    assert int(resistant[0]) == 0 and int(resistant[2]) > int(resistant[1])
    actions = jax.vmap(env.adaptive_policy)(states, params)
    out = jax.jit(jax.vmap(env.step))(keys, states, actions, params)
    assert out[2].shape == (3, 1) and not bool(jnp.any(out[5]["overflow"]))
    frame = env.render(jax.tree.map(lambda x: x[0], states))
    assert frame.ndim == 3 and frame.shape[2] == 3


def test_invalid_settings_are_rejected():
    with pytest.raises(ValueError, match="initial_cells"):
        envs.TumorTherapy(initial_cells=600, capacity=512)
    with pytest.raises(ValueError, match="action"):
        envs.TumorTherapy(action="surgery")
    env = envs.TumorTherapy(initial_cells=20, capacity=40)
    _, state = env.reset(jax.random.key(0))
    with pytest.raises(ValueError, match="random key"):
        env.step(None, state, jnp.zeros((1, 1)))
