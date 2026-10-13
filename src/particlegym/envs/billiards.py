"""Billiards: pocket the balls on a table with a cue ball.

Balls are discs that slide on a table with a constant friction, collide with
each other and with the cushions through damped elastic contacts (a
coefficient of restitution below 1), and drop into six pockets. An action is
a shot: the direction and the speed given to the cue ball. One environment
step simulates the shot until all balls are at rest.

The model has no spin: the balls slide without rolling, and contacts act
along the line of centres only. This environment is experimental; its
parameters are plausible values, not measurements.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from particlegym import physics as pp
from particlegym import viz
from particlegym.envs.spaces import Continuous

_BALL_COLORS = np.array(
    [
        (245, 245, 245), (240, 200, 0), (20, 90, 235), (230, 30, 30), (120, 40, 160),
        (255, 130, 0), (0, 130, 60), (130, 30, 30), (25, 25, 25), (250, 225, 110),
        (110, 160, 250), (245, 130, 130), (190, 140, 220), (255, 190, 120), (120, 200, 150),
        (200, 130, 130),
    ],
    dtype=np.uint8,
)  # fmt: skip
_REST_SPEED = 1e-3  # m/s; below this speed a ball counts as at rest


class BilliardsParams(NamedTuple):
    """Episode parameters. A pytree, so values may be traced or batched.

    Attributes:
        min_speed: Speed of the cue ball (m/s) for the weakest shot.
        max_speed: Speed of the cue ball (m/s) for the strongest shot.
        friction: Deceleration of a moving ball (m/s^2).
        scratch_penalty: Reward for a shot that pockets the cue ball.
        max_shot_time: Longest simulated time of a shot (s).
        max_shots: Episode length in shots.
    """

    min_speed: float = 0.3
    max_speed: float = 3.0
    friction: float = 0.6
    scratch_penalty: float = -1.0
    max_shot_time: float = 12.0
    max_shots: int = 30


class BilliardsState(NamedTuple):
    """Environment state. A pytree.

    Attributes:
        balls: The balls; ball 0 is the cue ball. Pocketed balls are not
            alive.
        shot: Number of shots taken.
    """

    balls: pp.ParticleState
    shot: jax.Array


class Billiards:
    """A table with six pockets, a cue ball and up to 15 object balls.

    The playing surface spans ``[0, table[0]] x [0, table[1]]``. The pockets
    are at the four corners and at the middle of the two long sides; a ball
    whose centre comes within ``pocket_radius`` of a pocket centre is
    pocketed. The object balls start in a triangle with its tip at three
    quarters of the table length, the cue ball at one quarter.

    Args:
        n_balls: Number of object balls, 1 to 15.
        table: Length and width of the playing surface (m).
        ball_radius: Radius of the balls (m).
        ball_mass: Mass of the balls (kg).
        pocket_radius: Capture radius of the pockets (m).
        ball_restitution: Coefficient of restitution between two balls.
        cushion_restitution: Coefficient of restitution of the cushions.
        stiffness: Contact stiffness (N/m).
        dt: Physics time step (s). With the default values the measured
            coefficients of restitution are within about 0.5 % of the set
            ones; the error grows with the time step and with the damping
            (1.3 % for a restitution of 0.7 between balls).
    """

    n_agents = 1

    def __init__(
        self,
        n_balls: int = 15,
        table=(2.24, 1.12),
        ball_radius: float = 0.0286,
        ball_mass: float = 0.17,
        pocket_radius: float = 0.07,
        ball_restitution: float = 0.95,
        cushion_restitution: float = 0.8,
        stiffness: float = 4e4,
        dt: float = 1e-4,
    ) -> None:
        if not 1 <= n_balls <= 15:
            raise ValueError("n_balls must be between 1 and 15")
        self.n_balls = int(n_balls)
        self.n = self.n_balls + 1
        self.table = (float(table[0]), float(table[1]))
        length, width = self.table
        self.ball_radius, self.ball_mass = float(ball_radius), float(ball_mass)
        self.pocket_radius = float(pocket_radius)
        self.dt = float(dt)
        self._chunk = 50  # physics steps between two checks for rest

        contact = pp.Harmonic(k=float(stiffness))
        ball_damping = pp.damping_for_restitution(ball_restitution, stiffness, ball_mass / 2)
        cushion_damping = pp.damping_for_restitution(cushion_restitution, stiffness, ball_mass)
        self._pair = [contact, pp.NormalDamping(ball_damping)]
        self._walls = [
            pp.Walls(pp.box_walls((0.0, 0.0), self.table), contact, damping=cushion_damping)
        ]
        self.box = pp.Box((-0.5, -0.5), (length + 0.5, width + 0.5))
        self.pockets = np.array(
            [[x, y] for y in (0.0, width) for x in (0.0, length / 2, length)], dtype=float
        )
        self.head_spot = np.array([length / 4, width / 2])
        self.foot_spot = np.array([3 * length / 4, width / 2])

        spacing = 2.0 * self.ball_radius * 1.001
        rack = [
            self.foot_spot + spacing * np.array([row * np.sqrt(3) / 2, i - row / 2])
            for row in range(5)
            for i in range(row + 1)
        ]
        self._rack = np.array(rack[: self.n_balls])
        # Places for the cue ball after a scratch, tried in this order.
        offsets = np.array([0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5]) * 3.0 * self.ball_radius
        self._spots = self.head_spot + np.stack([np.zeros_like(offsets), offsets], axis=1)

    @property
    def default_params(self) -> BilliardsParams:
        """Default episode parameters."""
        return BilliardsParams()

    @property
    def observation_space(self) -> Continuous:
        """Observation of the player: see :meth:`step`."""
        return Continuous(-0.5, 1.0, (3 * self.n,))

    @property
    def action_space(self) -> Continuous:
        """A shot: the direction (angle / pi) and the strength, both from -1 to 1."""
        return Continuous(-1.0, 1.0, (2,))

    def reset(
        self, key: jax.Array, params: BilliardsParams | None = None
    ) -> tuple[jax.Array, BilliardsState]:
        """Rack the object balls; the cue ball starts at a random place on the head line."""
        width = self.table[1]
        offset = jax.random.uniform(key, (), minval=-0.3 * width, maxval=0.3 * width)
        cue = jnp.asarray(self.head_spot) + jnp.stack([jnp.zeros_like(offset), offset])
        pos = jnp.concatenate([cue[None, :], jnp.asarray(self._rack, cue.dtype)])
        balls = pp.create_state(
            pos, mass=self.ball_mass, radius=self.ball_radius, kind=jnp.arange(self.n)
        )
        state = BilliardsState(balls, jnp.zeros((), jnp.int32))
        return self._observe(state), state

    def step(
        self,
        key: jax.Array,
        state: BilliardsState,
        action: jax.Array,
        params: BilliardsParams | None = None,
    ):
        """Play one shot and simulate it until all balls are at rest.

        Args:
            key: JAX random key (not used; the motion is deterministic).
            state: Current state, with all balls at rest.
            action: Shape (1, 2): the direction of the shot as an angle
                divided by pi, and its strength; -1 gives ``min_speed`` and 1
                gives ``max_speed``.
            params: Episode parameters.

        Returns:
            ``(obs, state, reward, terminated, truncated, info)`` with a
            leading axis of length 1 on the per-player arrays. The
            observation holds, for every ball, the position scaled to
            [-0.5, 0.5] and 1 if it is on the table (0 and a position of 0 if
            pocketed). The reward is the number of object balls pocketed by
            the shot, plus ``scratch_penalty`` if the cue ball was pocketed;
            the cue ball then returns to the head line. The game is
            terminated when no object ball is left. ``info`` holds the number
            of balls ``pocketed`` by the shot, the ``scratch`` flag, the
            simulated ``time`` and whether all balls came to ``rest``.
        """
        del key
        params = self.default_params if params is None else params
        field = self.force_field(params)
        before = state.balls
        balls, _ = pp.init_forces(self.strike(before, action, params), field)
        max_chunks = jnp.ceil(params.max_shot_time / (self._chunk * self.dt)).astype(jnp.int32)

        def moving(carry):
            b, chunks = carry
            speed = jnp.max(jnp.where(b.alive, jnp.sum(b.vel**2, axis=1), 0.0))
            return (speed > _REST_SPEED**2) & (chunks < max_chunks)

        def advance(carry):
            b, chunks = carry
            return self._advance(b, field, self._chunk), chunks + 1

        balls, chunks = jax.lax.while_loop(moving, advance, (balls, jnp.zeros((), jnp.int32)))
        rest = ~moving((balls, jnp.zeros((), jnp.int32)))
        balls = balls._replace(vel=jnp.zeros_like(balls.vel), force=jnp.zeros_like(balls.force))

        pocketed = jnp.sum(before.alive[1:] & ~balls.alive[1:])
        scratch = ~balls.alive[0]
        balls = self._replace_cue_ball(balls, scratch)
        shot = state.shot + 1
        state = BilliardsState(balls, shot)
        reward = pocketed + jnp.where(scratch, params.scratch_penalty, 0.0)
        done = ~jnp.any(balls.alive[1:])
        info = {
            "pocketed": pocketed,
            "scratch": scratch,
            "time": chunks * (self._chunk * self.dt),
            "rest": rest,
        }
        return (
            self._observe(state),
            state,
            jnp.reshape(reward.astype(balls.pos.dtype), (1,)),
            jnp.reshape(done, (1,)),
            jnp.reshape(~done & (shot >= params.max_shots), (1,)),
            info,
        )

    def force_field(self, params: BilliardsParams | None = None) -> pp.ForceField:
        """Contacts between balls and with the cushions, and the table friction."""
        params = self.default_params if params is None else params
        # Below 1 cm/s the friction weakens, so that a ball comes to rest smoothly.
        friction = pp.KineticFriction(params.friction, speed_scale=0.01)
        return pp.ForceField(self.box, pair=self._pair, walls=self._walls, external=[friction])

    def strike(self, balls: pp.ParticleState, action, params: BilliardsParams) -> pp.ParticleState:
        """The balls just after the cue has hit the cue ball."""
        action = jnp.reshape(jnp.asarray(action, balls.pos.dtype), (2,))
        angle = jnp.pi * action[0]
        strength = 0.5 * (jnp.clip(action[1], -1.0, 1.0) + 1.0)
        speed = params.min_speed + strength * (params.max_speed - params.min_speed)
        velocity = speed * jnp.stack([jnp.cos(angle), jnp.sin(angle)])
        return balls._replace(vel=jnp.zeros_like(balls.vel).at[0].set(velocity))

    def trajectory(
        self,
        state: BilliardsState,
        action: jax.Array,
        params: BilliardsParams | None = None,
        frames: int = 100,
        interval: float = 0.05,
    ) -> pp.ParticleState:
        """The balls at ``frames`` times, ``interval`` seconds apart, during a shot.

        For animations: the motion is the same as in :meth:`step`, but the
        number of steps is fixed and the cue ball is not put back.
        """
        params = self.default_params if params is None else params
        field = self.force_field(params)
        balls, _ = pp.init_forces(self.strike(state.balls, action, params), field)
        steps = max(1, round(interval / self.dt))

        def body(b, _):
            b = self._advance(b, field, steps)
            return b, b

        return jax.lax.scan(body, balls, None, length=frames)[1]

    def aim_policy(self, state: BilliardsState, strength: float = 0.0) -> jax.Array:
        """A simple player: shoot the cue ball at the easiest straight shot.

        For every object ball and pocket, the cue ball is aimed at the point
        from which it would push the ball towards the pocket. The pair with
        the smallest cut angle is chosen; balls in the way are ignored.

        Returns:
            An action, shape (1, 2).
        """
        b = state.balls
        cue, balls = b.pos[0], b.pos[1:]
        pockets = jnp.asarray(self.pockets, cue.dtype)
        to_pocket = pockets[None, :, :] - balls[:, None, :]
        to_pocket = to_pocket / jnp.linalg.norm(to_pocket, axis=-1, keepdims=True)
        ghost = balls[:, None, :] - 2.0 * self.ball_radius * to_pocket
        to_ghost = ghost - cue
        distance = jnp.linalg.norm(to_ghost, axis=-1)
        cut = jnp.sum(to_ghost * to_pocket, axis=-1) / jnp.maximum(distance, 1e-9)
        cut = jnp.where(b.alive[1:, None], cut, -jnp.inf)
        best = jnp.argmax(cut)
        aim = to_ghost.reshape(-1, 2)[best]
        angle = jnp.arctan2(aim[1], aim[0]) / jnp.pi
        return jnp.stack([angle, jnp.asarray(strength, angle.dtype)])[None, :]

    def render(self, balls: pp.ParticleState | BilliardsState, width: int = 480) -> np.ndarray:
        """Draw the table. Runs on the host and needs Pillow."""
        balls = balls.balls if isinstance(balls, BilliardsState) else balls
        length, height = self.table
        border = 0.08
        canvas = viz.Canvas(
            (-border, -border), (length + border, height + border), width, background=(95, 60, 35)
        )
        canvas.rectangles([[0.0, 0.0]], [[length, height]], (20, 125, 80))
        canvas.discs(self.pockets, self.pocket_radius, viz.BLACK)
        canvas.discs(balls.pos, balls.radius, _BALL_COLORS[np.asarray(balls.kind)], balls.alive)
        return canvas.frame()

    def _advance(self, balls: pp.ParticleState, field: pp.ForceField, steps: int):
        """Run physics steps; balls that reach a pocket are removed."""
        pockets = jnp.asarray(self.pockets, balls.pos.dtype)

        def body(b, _):
            b, _ = pp.velocity_verlet(b, field, self.dt)
            offset = b.pos[:, None, :] - pockets[None, :, :]
            inside = jnp.any(jnp.sum(offset * offset, axis=-1) < self.pocket_radius**2, axis=1)
            return b._replace(alive=b.alive & ~inside), None

        return jax.lax.scan(body, balls, None, length=steps)[0]

    def _replace_cue_ball(self, balls: pp.ParticleState, scratch: jax.Array) -> pp.ParticleState:
        """Put a pocketed cue ball on the first free place of the head line."""
        spots = jnp.asarray(self._spots, balls.pos.dtype)
        offset = spots[:, None, :] - balls.pos[None, 1:, :]
        blocked = (
            jnp.sum(offset * offset, axis=-1) < (2.2 * self.ball_radius) ** 2
        ) & balls.alive[1:]
        spot = spots[jnp.argmin(jnp.any(blocked, axis=1))]
        pos = balls.pos.at[0].set(jnp.where(scratch, spot, balls.pos[0]))
        return balls._replace(pos=pos, alive=balls.alive.at[0].set(True))

    def _observe(self, state: BilliardsState) -> jax.Array:
        b = state.balls
        size = jnp.asarray(self.table, b.pos.dtype)
        position = jnp.where(b.alive[:, None], b.pos / size - 0.5, 0.0)
        return jnp.concatenate(
            [position, b.alive[:, None].astype(position.dtype)], axis=1
        ).reshape(1, -1)
