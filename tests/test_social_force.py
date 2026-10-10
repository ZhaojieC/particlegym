"""Crowd dynamics: pedestrians in a room leave through the door and only through it."""

import jax
import jax.numpy as jnp
import numpy as np

from particlegym import physics as pp

ROOM = 8.0
DOOR = (3.4, 4.6)  # a 1.2 m wide door in the right wall
EXIT_X = ROOM + 1.0  # pedestrians 1 m past the door have left


def _evacuate(n=30, v0=1.2, tau=0.5, dt=0.005, seconds=60.0, seed=7):
    rng = np.random.default_rng(seed)
    radius = rng.uniform(0.25, 0.30, n)
    pos = pp.random_positions(rng, (0.0, 0.0), (ROOM, ROOM), radius, gap=0.05)
    walls = pp.box_walls((0.0, 0.0), (ROOM, ROOM), openings=[pp.Opening("x+", DOOR)])
    field = pp.ForceField(
        pp.Box((-1.0, -1.0), (ROOM + 3.0, ROOM + 1.0)),
        pair=[pp.SocialForce(), pp.TangentialFriction()],
        walls=[pp.Walls(walls, pp.SocialForce(), friction=2.4e5)],
    )
    state = pp.create_state(pos, mass=80.0, radius=radius)
    field.check(state)
    state, _ = pp.init_forces(state, field)
    door_centre = jnp.array([ROOM, sum(DOOR) / 2])

    def body(carry, _):
        s, crossing_y, escaped = carry
        inside = s.pos[:, 0] < ROOM
        direction = jnp.where(inside[:, None], door_centre - s.pos, jnp.array([1.0, 0.0]))
        direction = direction / jnp.linalg.norm(direction, axis=1, keepdims=True)
        new, _ = pp.langevin_baoab(s, field, dt, gamma=1.0 / tau, drift=v0 * direction)
        # Height at which each pedestrian crosses the plane of the right wall.
        crossed = s.alive & (s.pos[:, 0] < ROOM) & (new.pos[:, 0] >= ROOM)
        frac = (ROOM - s.pos[:, 0]) / (new.pos[:, 0] - s.pos[:, 0])
        y = s.pos[:, 1] + frac * (new.pos[:, 1] - s.pos[:, 1])
        crossing_y = jnp.where(crossed, y, crossing_y)
        # Anyone inside the x-range of the room but beyond its other walls escaped.
        outside = (new.pos[:, 0] < 0.0) | (new.pos[:, 1] < 0.0) | (new.pos[:, 1] > ROOM)
        escaped = escaped | (new.alive & (new.pos[:, 0] < ROOM) & outside)
        new = new._replace(alive=new.alive & (new.pos[:, 0] < EXIT_X))
        return (new, crossing_y, escaped), jnp.sum(new.alive)

    carry = (state, jnp.full(n, jnp.nan), jnp.zeros(n, dtype=bool))
    steps = round(seconds / dt)
    (_, crossing_y, escaped), remaining = jax.jit(
        lambda c: jax.lax.scan(body, c, None, length=steps)
    )(carry)
    return np.asarray(crossing_y), np.asarray(escaped), np.asarray(remaining), dt


def test_pedestrians_leave_only_through_the_door():
    crossing_y, escaped, remaining, dt = _evacuate()

    assert remaining[-1] == 0, "everybody should have left within 60 s"
    assert not escaped.any(), "nobody may pass through a wall"
    assert np.all(np.isfinite(crossing_y))
    assert np.all((crossing_y > DOOR[0]) & (crossing_y < DOOR[1]))

    evacuation_time = (np.argmax(remaining == 0) + 1) * dt
    # 30 people through a 1.2 m door: tens of seconds, not seconds and not minutes.
    assert 10.0 < evacuation_time < 60.0, evacuation_time
