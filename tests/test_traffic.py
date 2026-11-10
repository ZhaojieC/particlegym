"""Car-following models: driver laws, the ballistic update, uniform flow and its stability."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from particlegym import physics as pp

RING, CARS, LENGTH = 230.0, 22, 5.0  # the circuit of Sugiyama et al. (2008)


def _ring(model, pos, speed, dt, steps, ring=RING, noise=0.0, key=None):
    """Simulate all drivers of a ring; returns positions, speeds and gaps of every step."""
    keys = jax.random.split(jax.random.key(0) if key is None else key, steps)

    def body(carry, k):
        p, s = carry
        a = pp.ring_accelerations(model, p, s, ring, LENGTH)
        a = a + noise * jax.random.normal(k, s.shape, s.dtype) / math.sqrt(dt)
        p, s = pp.ballistic_step(p, s, a, dt, ring)
        return (p, s), (p, s, pp.ring_gaps(p, ring, LENGTH))

    return jax.jit(lambda c: jax.lax.scan(body, c, keys)[1])((pos, speed))


def test_intelligent_driver_model_follows_its_formula():
    idm = pp.IntelligentDriver()  # v0 = 30, T = 1, a = 1, b = 1.5, delta = 4, s0 = 2
    assert float(idm.acceleration(jnp.inf, 0.0, 0.0)) == 1.0  # free road, at rest: a
    assert float(idm.acceleration(jnp.inf, 30.0, 30.0)) == 0.0  # free road at v0
    assert float(idm.acceleration(2.0, 0.0, 0.0)) == pytest.approx(0.0, abs=1e-15)  # at s0
    s, v, u = 7.0, 4.0, 3.0
    desired = 2.0 + v * 1.0 + v * (v - u) / (2.0 * math.sqrt(1.5))
    expected = 1.0 - (v / 30.0) ** 4 - (desired / s) ** 2
    assert float(idm.acceleration(s, v, u)) == pytest.approx(expected, rel=1e-12)
    # A leader that drives away much faster does not shrink the desired gap below s0.
    assert float(idm.desired_gap(4.0, 40.0)) == 2.0
    with pytest.raises(ValueError, match="positive"):
        pp.IntelligentDriver(T=0.0)


@pytest.mark.parametrize(
    "model",
    [pp.IntelligentDriver(), pp.IntelligentDriver(T=1.5, a=2.0, delta=1.0), pp.OptimalVelocity()],
)
def test_equilibrium_speeds_make_the_acceleration_vanish(model):
    gaps = jnp.linspace(2.5, 60.0, 40)
    speed = model.equilibrium_speed(gaps)
    np.testing.assert_allclose(model.acceleration(gaps, speed, speed), 0.0, atol=1e-12)
    assert np.all(np.diff(np.asarray(speed)) > 0)  # faster at larger gaps
    if isinstance(model, pp.IntelligentDriver):
        assert float(model.equilibrium_speed(1.5)) == 0.0  # standing below the minimum gap


def test_ballistic_step_is_exact_for_constant_accelerations_and_stops_vehicles():
    pos, speed = jnp.array([0.0, 50.0]), jnp.array([2.0, 10.0])
    acc = jnp.array([0.5, -2.0])
    for _ in range(30):  # 3 s
        pos, speed = pp.ballistic_step(pos, speed, acc, 0.1, 1000.0)
    np.testing.assert_allclose(pos, [2.0 * 3 + 0.25 * 9, 50.0 + 30.0 - 9.0], rtol=1e-12)
    np.testing.assert_allclose(speed, [3.5, 4.0], rtol=1e-12)
    for _ in range(30):  # the braking vehicle stops at t = 5 s, 25 m after its start
        pos, speed = pp.ballistic_step(pos, speed, acc, 0.1, 1000.0)
    np.testing.assert_allclose([pos[1], speed[1]], [75.0, 0.0], atol=1e-12)
    # Positions are wrapped onto the ring.
    pos, _ = pp.ballistic_step(jnp.array([995.0]), jnp.array([10.0]), jnp.zeros(1), 1.0, 1000.0)
    np.testing.assert_allclose(pos, [5.0], atol=1e-12)


def test_free_road_relaxation_converges_at_first_order():
    # With delta = 1 the free-road equation dv/dt = a (1 - v / v0) is linear:
    # v = v0 (1 - exp(-a t / v0)) from rest.
    model = pp.IntelligentDriver(delta=1.0)

    def errors(dt):
        def body(carry, _):
            p, s = carry
            return pp.ballistic_step(p, s, model.acceleration(jnp.inf, s, s), dt, 1e9), None

        steps = round(10.0 / dt)
        (p, s), _ = jax.lax.scan(body, (jnp.zeros(1), jnp.zeros(1)), None, length=steps)
        decay = 1.0 - math.exp(-10.0 / 30.0)
        return abs(float(s[0]) - 30.0 * decay), abs(float(p[0]) - (300.0 - 900.0 * decay))

    coarse, fine = errors(0.1), errors(0.05)
    assert coarse[0] < 0.015 and coarse[1] < 0.1
    np.testing.assert_allclose(np.divide(coarse, fine), 2.0, rtol=0.01)


@pytest.mark.parametrize(("speed", "gap"), [(10.0, 30.0), (15.0, 100.0), (30.0, 150.0)])
def test_a_driver_stops_behind_a_standing_vehicle_without_collision(speed, gap):
    idm = pp.IntelligentDriver()

    def body(carry, _):
        x, v = carry
        a = idm.acceleration(gap - x, v, jnp.zeros(1))
        x, v = pp.ballistic_step(x, v, a, 0.1, 1e6)
        return (x, v), (gap - x[0], v[0])

    _, (gaps, speeds) = jax.lax.scan(body, (jnp.zeros(1), jnp.full(1, speed)), None, length=3000)
    assert float(jnp.min(gaps)) > 1.5  # close to s0 = 2 m, never touching
    assert float(speeds[-1]) == 0.0 and float(jnp.min(speeds)) >= 0.0


def test_ring_gaps_wrap_around_the_ring():
    pos = jnp.array([225.0, 5.0, 100.0])  # the first vehicle is just before the end
    np.testing.assert_allclose(pp.ring_gaps(pos, RING, LENGTH), [5.0, 90.0, 120.0])
    np.testing.assert_allclose(pp.ring_gaps(jnp.array([3.0]), RING, LENGTH), [225.0])


def test_uniform_flow_is_a_steady_state():
    idm = pp.IntelligentDriver()
    gap = RING / CARS - LENGTH
    v = float(idm.equilibrium_speed(gap))
    pos = jnp.arange(CARS) * RING / CARS
    _, speeds, gaps = _ring(idm, pos, jnp.full(CARS, v), 0.1, 1000)  # 100 s
    np.testing.assert_allclose(speeds, v, atol=1e-9)
    np.testing.assert_allclose(gaps, gap, atol=1e-9)


@pytest.mark.parametrize("k", [1.0, 2.5, 2.9, 4.0])
def test_optimal_velocity_modes_follow_the_exact_stability_condition(k):
    model = pp.OptimalVelocity(k=k)
    gap = RING / CARS - LENGTH
    slope = float(jax.grad(model.optimal_velocity)(gap))  # V'(s) = 1.47 /s
    rates = np.asarray(pp.ring_modes(model, CARS, RING, LENGTH))
    theta = 2 * np.pi * np.arange(1, CARS) / CARS
    z = np.exp(1j * theta)[:, None]
    # For dv/dt = k (V(s) - v) the rates solve lambda^2 + k lambda - k V' (z - 1) = 0 ...
    np.testing.assert_allclose(rates**2 + k * rates - k * slope * (z - 1), 0.0, atol=1e-12)
    # ... and a mode is unstable exactly where V' > k / (1 + cos theta); for long
    # waves this is the condition V' > k / 2 of Bando et al. (1995).
    unstable = rates.real.max(axis=1) > 0
    np.testing.assert_array_equal(unstable, slope * (1 + np.cos(theta)) > k)


@pytest.mark.parametrize(
    "model", [pp.IntelligentDriver(), pp.IntelligentDriver(a=2.0), pp.OptimalVelocity(k=4.0)]
)
def test_disturbances_grow_or_decay_at_the_rate_of_linear_theory(model):
    expected = float(np.asarray(pp.ring_modes(model, CARS, RING, LENGTH))[0].real.max())
    gap = RING / CARS - LENGTH
    theta = 2 * np.pi * np.arange(CARS) / CARS
    pos = jnp.asarray(np.arange(CARS) * RING / CARS + 1e-3 * np.cos(theta))
    speed = jnp.full(CARS, float(model.equilibrium_speed(gap)))
    dt = 0.01
    _, _, gaps = _ring(model, pos, speed, dt, 30000)  # 300 s
    amplitude = np.abs(np.asarray(gaps - gap) @ np.exp(-1j * theta))
    t = dt * np.arange(1, 30001)
    late = t > 50.0  # after the decay of the second, fast root
    measured = np.polyfit(t[late], np.log(amplitude[late]), 1)[0]
    # IDM with a = 1: unstable (growth 0.023 /s); a = 2 and the OVM with k = 4: stable.
    assert measured == pytest.approx(expected, rel=0.015)


def test_a_phantom_jam_forms_and_moves_backward():
    idm = pp.IntelligentDriver()
    v = float(idm.equilibrium_speed(RING / CARS - LENGTH))  # 3.45 m/s
    key = jax.random.key(1)
    jitter = jax.random.uniform(key, (CARS,), minval=-0.5, maxval=0.5)
    pos = jnp.arange(CARS) * RING / CARS + jitter
    positions, speeds, gaps = _ring(
        idm, pos, jnp.full(CARS, v), 0.1, 6000, noise=0.2, key=jax.random.fold_in(key, 1)
    )
    late = slice(3000, None)  # the last 300 of 600 s
    assert float(jnp.min(gaps)) > 1.0  # no collision
    assert float(jnp.mean(jnp.std(speeds[late], axis=1))) > 2.0  # stop and go
    assert float(jnp.min(speeds[late])) == 0.0  # vehicles stand in the jam
    assert float(jnp.mean(speeds[late])) < 0.8 * v
    # The slowest vehicle marks the jam; it moves against the traffic.
    slowest = np.asarray(positions)[np.arange(6000), np.asarray(jnp.argmin(speeds, axis=1))]
    track = np.unwrap(slowest[late] * 2 * np.pi / RING) * RING / (2 * np.pi)
    velocity = np.polyfit(0.1 * np.arange(3000), track, 1)[0]
    assert -6.0 < velocity < -3.0  # m/s
