"""Distance functions, rooms with openings, and wall interactions."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from particlegym import physics as pp


def test_segment_distance():
    plates = pp.Plates.from_segments([[(0.0, 0.0), (2.0, 0.0)]])
    points = jnp.array([[1.0, 3.0], [3.0, 0.0], [-1.0, -1.0], [1.0, 0.0]])
    np.testing.assert_allclose(plates.distance(points)[:, 0], [3.0, 1.0, np.sqrt(2.0), 0.0])


def test_rectangle_distance():
    plates = pp.Plates.from_rectangles([[(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 1.0, 0.0)]])
    points = jnp.array([[1.0, 0.5, 2.0], [3.0, 0.5, 0.0], [3.0, 2.0, 0.0], [1.0, 0.5, -0.5]])
    np.testing.assert_allclose(plates.distance(points)[:, 0], [2.0, 1.0, np.sqrt(2.0), 0.5])


def test_sphere_and_box_distances_are_signed():
    spheres = pp.Spheres([[0.0, 0.0, 0.0]], [1.0])
    np.testing.assert_allclose(
        spheres.distance(jnp.array([[2.0, 0.0, 0.0], [0.5, 0.0, 0.0]]))[:, 0], [1.0, -0.5]
    )
    boxes = pp.Boxes([[0.0, 0.0]], [[2.0, 1.0]])
    np.testing.assert_allclose(
        boxes.distance(jnp.array([[3.0, 0.5], [3.0, 2.0], [1.0, 0.5]]))[:, 0],
        [1.0, np.sqrt(2.0), -0.5],
    )


def test_non_orthogonal_rectangle_is_rejected():
    with pytest.raises(ValueError, match="orthogonal"):
        pp.Plates.from_rectangles([[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]])


def test_box_walls_leave_a_door_open_in_2d():
    plates = pp.box_walls((0.0, 0.0), (10.0, 8.0), openings=[pp.Opening("x+", (3.0, 4.0))])
    assert plates.origins.shape == (5, 2)  # four walls, the right one split in two
    d = plates.distance(jnp.array([[10.0, 3.5], [10.0, 6.0]])).min(axis=1)
    np.testing.assert_allclose(d, [0.5, 0.0])  # door centre is 0.5 m from both jambs


def test_box_walls_leave_a_door_open_in_3d():
    door = pp.Opening("y-", ((1.0, 2.0), (0.0, 2.0)))  # x in [1, 2], z in [0, 2]
    plates = pp.box_walls((0.0, 0.0, 0.0), (4.0, 4.0, 3.0), openings=[door])
    assert plates.origins.shape == (8, 3)  # five full faces + three pieces around the door
    d = plates.distance(jnp.array([[1.5, 0.0, 1.0], [3.0, 0.0, 1.0]])).min(axis=1)
    np.testing.assert_allclose(d, [0.5, 0.0])


@pytest.mark.parametrize(
    "opening",
    [pp.Opening("w+", (0.2, 0.4)), pp.Opening("x+", (0.5, 1.5)), pp.Opening("x+", (0.6, 0.4))],
)
def test_box_walls_reject_invalid_openings(opening):
    with pytest.raises(ValueError):
        pp.box_walls((0.0, 0.0), (1.0, 1.0), openings=[opening])


def _shoot(start, velocity, walls, steps, dt=1e-3):
    """Fire one particle (radius 0.3) at the walls and return its trajectory."""
    box = pp.Box((-20.0, -20.0), (20.0, 20.0))
    field = pp.ForceField(box, walls=[pp.Walls(walls, pp.Harmonic(k=1.0e4))])
    state = pp.create_state([start], vel=[velocity], radius=0.3)
    state, _ = pp.init_forces(state, field)
    step = partial(pp.velocity_verlet, force_field=field, dt=dt)

    def body(s, _):
        s, info = step(s)
        return s, (s.pos[0], info.potential_energy + pp.kinetic_energy(s))

    final, (path, energy) = jax.jit(lambda s: jax.lax.scan(body, s, None, length=steps))(state)
    return final, np.asarray(path), np.asarray(energy)


def test_particle_bounces_off_a_wall_elastically():
    wall = pp.Plates.from_segments([[(5.0, -5.0), (5.0, 5.0)]])
    # Contact frequency omega = sqrt(k / m) = 100; omega * dt = 0.025 resolves the bounce.
    final, path, energy = _shoot((3.0, 0.0), (2.0, 0.0), wall, steps=10_000, dt=2.5e-4)
    assert path[:, 0].max() < 5.0 - 0.25  # never gets close to crossing
    np.testing.assert_allclose(final.vel[0], [-2.0, 0.0], atol=5e-4)
    np.testing.assert_allclose(energy, energy[0], rtol=1e-3)


def test_particles_pass_through_the_door_but_not_the_wall():
    room = pp.box_walls((0.0, 0.0), (10.0, 10.0), openings=[pp.Opening("x+", (4.0, 6.0))])
    _, path, _ = _shoot((8.0, 5.0), (1.0, 0.0), room, steps=4000)
    assert path[-1, 0] > 11.0
    _, path, _ = _shoot((8.0, 2.0), (1.0, 0.0), room, steps=4000)
    assert path[:, 0].max() < 10.0
    assert path[-1, 0] < 8.0  # it came back
