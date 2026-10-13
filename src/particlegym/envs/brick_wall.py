"""Brick wall: a paddle keeps a ball in play that knocks bricks out of a wall.

An arcade game built from the physics core. The ball is a particle that moves
by Newton's equations and bounces off the walls, the bricks and the paddle
through a stiff elastic contact potential. The paddle is the cap of a large
disc that slides along the bottom, so the point where the ball hits it sets
the direction of the rebound. A brick disappears when the ball leaves it
after a contact.

The observation is either a short state vector or an image computed in JAX,
so the whole game compiles with :func:`jax.jit` and batches with
:func:`jax.vmap`. This environment is experimental.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from particlegym import physics as pp
from particlegym import viz
from particlegym.envs.spaces import Continuous, Discrete

_ROW_COLORS = (viz.RED, viz.ORANGE, viz.GREEN, viz.BLUE, (150, 60, 200))
# Radius of the paddle disc and the height of its cap above the bottom line.
_PADDLE_RADIUS = 3.0
_PADDLE_HEIGHT = 0.5


class BrickWallParams(NamedTuple):
    """Episode parameters. A pytree, so values may be traced or batched.

    Attributes:
        ball_speed: Speed of the ball at launch, and the speed to which it
            returns after a hit by the moving paddle.
        speed_relaxation: Rate at which the speed returns to ``ball_speed``,
            without a change of direction; 0 gives purely Newtonian motion.
        paddle_speed: Largest speed of the paddle.
        lives: Number of balls.
        max_steps: Episode length in control steps.
    """

    ball_speed: float = 6.0
    speed_relaxation: float = 2.0
    paddle_speed: float = 8.0
    lives: int = 3
    max_steps: int = 5000


class BrickWallState(NamedTuple):
    """Environment state. A pytree.

    Attributes:
        ball: The ball, a single particle.
        paddle: x coordinate of the centre of the paddle.
        bricks: (n_bricks,) booleans, true for bricks that are in the wall.
        touched: (n_bricks,) booleans, true for bricks that the ball is
            touching; they disappear when the contact ends.
        lives: Balls left, the one in play included.
        score: Number of bricks removed.
        step: Number of control steps taken.
        frame: The image of the previous control step (pixel observations).
    """

    ball: pp.ParticleState
    paddle: jax.Array
    bricks: jax.Array
    touched: jax.Array
    lives: jax.Array
    score: jax.Array
    step: jax.Array
    frame: jax.Array


class BrickWall:
    """A ball, a paddle and a wall of bricks.

    The arena spans ``[0, size[0]] x [0, size[1]]`` with walls on the left,
    on the right and at the top. The bottom is open: a ball that passes the
    paddle is lost.

    Args:
        size: Width and height of the arena.
        rows: Number of rows of bricks.
        columns: Number of bricks per row; the bricks fill the width.
        brick_height: Height of a brick.
        top_gap: Free space between the top wall and the bricks.
        ball_radius: Radius of the ball.
        observation: ``"state"`` (ball position and velocity, paddle
            position, and one entry per brick) or ``"pixels"`` (two images:
            the current one and the one of the previous control step).
        action: ``"discrete"`` (0: stay, 1: left, 2: right) or
            ``"continuous"`` (the paddle velocity from -1 to 1, in units of
            ``paddle_speed``).
        pixels: Image size ``(height, width)`` of pixel observations; the
            pixels must be square.
        stiffness: Stiffness of the elastic contacts.
        dt: Physics time step.
        substeps: Physics steps per control step.
    """

    n_agents = 1

    def __init__(
        self,
        size=(8.0, 8.0),
        rows: int = 5,
        columns: int = 8,
        brick_height: float = 0.4,
        top_gap: float = 1.2,
        ball_radius: float = 0.15,
        observation: str = "state",
        action: str = "discrete",
        pixels: tuple[int, int] = (84, 84),
        stiffness: float = 2e4,
        dt: float = 1e-3,
        substeps: int = 20,
    ) -> None:
        if observation not in ("state", "pixels"):
            raise ValueError("observation must be 'state' or 'pixels'")
        if action not in ("discrete", "continuous"):
            raise ValueError("action must be 'discrete' or 'continuous'")
        self.size = (float(size[0]), float(size[1]))
        width, height = self.size
        if rows < 1 or columns < 1 or top_gap + rows * brick_height > height - 3.0:
            raise ValueError("the bricks must leave at least 3 length units above the bottom")
        self.rows, self.columns = int(rows), int(columns)
        self.n_bricks = self.rows * self.columns
        self.ball_radius = float(ball_radius)
        self.observation, self.action = observation, action
        self.pixels = (int(pixels[0]), int(pixels[1]))
        if (
            observation == "pixels"
            and abs(width / self.pixels[1] - height / self.pixels[0]) > 1e-9
        ):
            raise ValueError("pixel observations need square pixels: size and pixels must agree")
        self.pixel = width / self.pixels[1]
        self.dt, self.substeps = float(dt), int(substeps)

        brick_width = width / self.columns
        top = height - float(top_gap)
        column, row = np.meshgrid(np.arange(self.columns), np.arange(self.rows))
        lo = np.stack([column * brick_width, top - (row + 1) * brick_height], axis=-1)
        self.brick_lo = lo.reshape(-1, 2)
        self.brick_hi = self.brick_lo + np.array([brick_width, brick_height])
        self.bricks = pp.Boxes(self.brick_lo, self.brick_hi)
        self.brick_row = row.reshape(-1)

        self.contact = pp.Harmonic(k=float(stiffness))
        # The side walls continue below the bottom line, so a ball that is
        # about to be lost still bounces off them.
        self.walls = pp.Walls(
            pp.Plates.from_segments(
                [
                    [(0.0, -2.0), (0.0, height)],
                    [(width, -2.0), (width, height)],
                    [(0.0, height), (width, height)],
                ]
            ),
            self.contact,
        )
        self.box = pp.Box((-1.0, -4.0), (width + 1.0, height + 1.0))
        self.paddle_y = _PADDLE_HEIGHT - _PADDLE_RADIUS
        half_width = float(np.sqrt(_PADDLE_RADIUS**2 - self.paddle_y**2))
        self.paddle_half_width = half_width
        self.paddle_range = (0.5 * half_width, width - 0.5 * half_width)

    @property
    def default_params(self) -> BrickWallParams:
        """Default episode parameters."""
        return BrickWallParams()

    @property
    def observation_space(self) -> Continuous:
        """Observation of the player."""
        if self.observation == "pixels":
            return Continuous(0.0, 1.0, (*self.pixels, 2))
        return Continuous(-jnp.inf, jnp.inf, (5 + self.n_bricks,))

    @property
    def action_space(self) -> Discrete | Continuous:
        """Action of the player."""
        return Discrete(3) if self.action == "discrete" else Continuous(-1.0, 1.0, (1,))

    def reset(
        self, key: jax.Array, params: BrickWallParams | None = None
    ) -> tuple[jax.Array, BrickWallState]:
        """Start a game: all bricks in place, the paddle in the middle, a ball launched."""
        params = self.default_params if params is None else params
        ball = self._launch(key, params)
        frame = jnp.zeros(self.pixels if self.observation == "pixels" else (1, 1), ball.pos.dtype)
        state = BrickWallState(
            ball=ball,
            paddle=jnp.asarray(0.5 * self.size[0], ball.pos.dtype),
            bricks=jnp.ones(self.n_bricks, bool),
            touched=jnp.zeros(self.n_bricks, bool),
            lives=jnp.asarray(params.lives, jnp.int32),
            score=jnp.zeros((), jnp.int32),
            step=jnp.zeros((), jnp.int32),
            frame=frame,
        )
        state = state._replace(frame=self._image(state))
        return self._observe(state, state.frame, state.frame, params), state

    def step(
        self,
        key: jax.Array,
        state: BrickWallState,
        action: jax.Array,
        params: BrickWallParams | None = None,
    ):
        """Advance the game by one control step.

        Returns:
            ``(obs, state, reward, terminated, truncated, info)`` with a
            leading axis of length 1 (one player) on ``obs``, ``reward``,
            ``terminated`` and ``truncated``. The reward is the number of
            bricks removed in this step. The game is terminated when the wall
            is cleared or the last ball is lost. ``info`` holds the
            ``score``, the ``lives`` and whether a ball was ``lost`` in this
            step.
        """
        params = self.default_params if params is None else params
        dtype = state.ball.pos.dtype
        if self.action == "discrete":
            command = jnp.asarray([0.0, -1.0, 1.0], dtype)[jnp.reshape(action, ())]
        else:
            command = jnp.clip(jnp.reshape(jnp.asarray(action, dtype), ()), -1.0, 1.0)
        paddle_velocity = params.paddle_speed * command
        lo, hi = self.paddle_range

        def substep(carry, _):
            ball, paddle, bricks, touched, removed = carry
            paddle = jnp.clip(paddle + paddle_velocity * self.dt, lo, hi)
            speed = jnp.sqrt(jnp.sum(ball.vel**2, axis=-1, keepdims=True))
            ball, _ = pp.langevin_baoab(
                ball,
                self.force_field(paddle, bricks),
                self.dt,
                gamma=params.speed_relaxation,
                drift=params.ball_speed * ball.vel / jnp.maximum(speed, 1e-9),
            )
            touching = bricks & (self.bricks.distance(ball.pos)[0] < self.ball_radius)
            left = touched & ~touching
            carry = (
                ball,
                paddle,
                bricks & ~left,
                touching,
                removed + jnp.sum(left, dtype=jnp.int32),
            )
            return carry, None

        start = (state.ball, state.paddle, state.bricks, state.touched, jnp.zeros((), jnp.int32))
        (ball, paddle, bricks, touched, removed), _ = jax.lax.scan(
            substep, start, None, length=self.substeps
        )

        lost = ball.pos[0, 1] < -2.0 * self.ball_radius
        lives = state.lives - lost.astype(jnp.int32)
        ball = jax.tree.map(
            lambda new, old: jnp.where(lost, new, old), self._launch(key, params), ball
        )
        previous = state.frame
        state = BrickWallState(
            ball=ball,
            paddle=paddle,
            bricks=bricks,
            touched=touched,
            lives=lives,
            score=state.score + removed,
            step=state.step + 1,
            frame=previous,
        )
        image = self._image(state)
        obs = self._observe(state, image, previous, params)
        state = state._replace(frame=image)
        done = (lives <= 0) | ~jnp.any(bricks)
        info = {"score": state.score, "lives": lives, "lost": lost}
        return (
            obs,
            state,
            jnp.reshape(removed.astype(dtype), (1,)),
            jnp.reshape(done, (1,)),
            jnp.reshape(~done & (state.step >= params.max_steps), (1,)),
            info,
        )

    def force_field(self, paddle: jax.Array, bricks: jax.Array) -> pp.ForceField:
        """The contact forces on the ball for a paddle position and a set of bricks."""
        centre = jnp.stack([paddle, jnp.full_like(paddle, self.paddle_y)])[None, :]
        radius = jnp.full((1,), _PADDLE_RADIUS, centre.dtype)
        walls = [
            self.walls,
            pp.Walls(pp.Spheres(centre, radius), self.contact),
            pp.Walls(self.bricks, self.contact, active=bricks),
        ]
        return pp.ForceField(self.box, walls=walls)

    def scripted_policy(self, state: BrickWallState, params: BrickWallParams | None = None):
        """A simple player: move the paddle to where the ball will come down.

        The path of the ball is continued in a straight line and folded at
        the side walls. The paddle aims a little off the centre, by an amount
        that changes slowly, so that the rebounds vary.
        """
        width = self.size[0]
        pos, vel = state.ball.pos[0], state.ball.vel[0]
        time = (pos[1] - _PADDLE_HEIGHT) / jnp.maximum(-vel[1], 1e-6)
        r = self.ball_radius
        span = width - 2.0 * r
        landing = jnp.mod(pos[0] + vel[0] * time - r, 2.0 * span)
        landing = r + jnp.where(landing > span, 2.0 * span - landing, landing)
        target = jnp.where(vel[1] < 0, landing, pos[0])
        target = target + 0.8 * jnp.sin(0.05 * state.step)
        offset = target - state.paddle
        if self.action == "continuous":
            scale = (self.default_params if params is None else params).paddle_speed
            return jnp.clip(offset / (scale * self.dt * self.substeps), -1.0, 1.0)[None]
        return jnp.where(jnp.abs(offset) < 0.1, 0, jnp.where(offset < 0, 1, 2))[None]

    def render(self, state: BrickWallState, width: int = 320) -> np.ndarray:
        """Draw the game with coloured bricks. Runs on the host and needs Pillow."""
        w, h = self.size
        canvas = viz.Canvas((-0.15, 0.0), (w + 0.15, h + 0.15), width, background=viz.BLACK)
        canvas.rectangles([[0.0, 0.0]], [[w, h]], (245, 245, 245))
        colors = np.array(_ROW_COLORS)[self.brick_row % len(_ROW_COLORS)]
        # A thin gap between the bricks makes them visible as single bricks.
        gap = 0.03
        canvas.rectangles(
            self.brick_lo + gap, self.brick_hi - gap, colors, np.asarray(state.bricks)
        )
        canvas.discs([[float(state.paddle), self.paddle_y]], _PADDLE_RADIUS, (60, 60, 60))
        # The side walls hide the ends of the paddle that reach beyond them.
        canvas.rectangles([[-0.15, 0.0], [w, 0.0]], [[0.0, h], [w + 0.15, h]], viz.BLACK)
        canvas.discs(np.asarray(state.ball.pos), self.ball_radius, viz.BLACK)
        return canvas.frame()

    def _launch(self, key: jax.Array, params: BrickWallParams) -> pp.ParticleState:
        """A ball below the bricks that flies down within 30 degrees of the vertical."""
        angle = jax.random.uniform(key, (), minval=-jnp.pi / 6, maxval=jnp.pi / 6)
        velocity = params.ball_speed * jnp.stack([jnp.sin(angle), -jnp.cos(angle)])
        pos = jnp.asarray([[0.5 * self.size[0], 3.0]], velocity.dtype)
        return pp.create_state(pos, velocity[None, :], radius=self.ball_radius)

    def _image(self, state: BrickWallState) -> jax.Array:
        """Grey-scale image: ball 1, bricks 0.7, paddle 0.4."""
        if self.observation != "pixels":
            return state.frame
        dtype = state.ball.pos.dtype
        points = viz.pixel_centres((0.0, 0.0), self.size, self.pixels, dtype)
        ball = viz.disc_coverage(points, state.ball.pos, self.ball_radius, self.pixel)
        centre = jnp.stack([state.paddle, jnp.full_like(state.paddle, self.paddle_y)])[None, :]
        paddle = viz.disc_coverage(points, centre, _PADDLE_RADIUS, self.pixel)
        bricks = viz.box_coverage(
            points,
            jnp.asarray(self.brick_lo, dtype),
            jnp.asarray(self.brick_hi, dtype),
            self.pixel,
            state.bricks,
        )
        return jnp.maximum(ball, jnp.maximum(0.7 * bricks, 0.4 * paddle))

    def _observe(self, state: BrickWallState, image, previous, params) -> jax.Array:
        if self.observation == "pixels":
            return jnp.stack([image, previous], axis=-1)[None]
        size = jnp.asarray(self.size, state.ball.pos.dtype)
        ball = state.ball.pos[0] / size - 0.5
        velocity = state.ball.vel[0] / params.ball_speed
        paddle = state.paddle / size[0] - 0.5
        bricks = state.bricks.astype(ball.dtype)
        return jnp.concatenate([ball, velocity, paddle[None], bricks])[None]
