"""Microswimmer: swim upwards through a field of vortices.

The task of Colabrese et al. (2017). A swimmer moves at a constant speed
along its heading while the flow carries it along. The heading turns towards
a preferred direction, which the agent sets, and with the fluid, at half the
vorticity. The reward is the gain in altitude. A swimmer that always prefers
"up" is trapped by the vortices for some parameters; a swimmer that senses
the vorticity can learn to escape.

Equations of motion, in the plane (x, y) with y as the altitude:

    dx/dt = u(x) + v_s p + sqrt(2 D_0) eta(t),
    d(theta)/dt = sin(theta_k - theta) / (2 B) + omega(x) / 2 + sqrt(2 D_R) xi(t),

with the heading ``p = (cos(theta), sin(theta))``, the preferred direction at
the angle ``theta_k``, and the Taylor-Green flow ``u`` with the vorticity
``omega``. The two parameters of the paper are the swimming number
``Phi = v_s / u0`` and the stability number ``Psi = B omega_0``, where
``omega_0 = u0 k``.

This environment is experimental. The sizes of the noise terms and the
thresholds of the discrete observation are in the supplementary material of
the paper, which was not available; the values here were chosen freely.

S. Colabrese, K. Gustavsson, A. Celani and L. Biferale, Flow navigation by
smart microswimmers via reinforcement learning, Phys. Rev. Lett. 118, 158004
(2017).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from particlegym import physics as pp
from particlegym import viz
from particlegym.envs.evacuation import _ring
from particlegym.envs.spaces import Continuous, Discrete


class MicroswimmerParams(NamedTuple):
    """Episode parameters. A pytree, so values may be traced or batched.

    The defaults are the point ``Phi = 0.3``, ``Psi = 1`` for which
    Colabrese et al. (2017) show learning curves.

    Attributes:
        swim_speed: Swimming speed v_s.
        alignment_time: Time scale B on which the heading turns to the
            preferred direction.
        flow_speed: Velocity scale u0 of the flow; its largest speed is u0 / 2.
        rotational_diffusion: Diffusion coefficient D_R of the heading.
        translational_diffusion: Diffusion coefficient D_0 of the position.
        max_steps: Episode length in control steps.
    """

    swim_speed: float = 0.3
    alignment_time: float = 1.0
    flow_speed: float = 1.0
    rotational_diffusion: float = 0.01
    translational_diffusion: float = 0.001
    max_steps: int = 1000


class MicroswimmerState(NamedTuple):
    """Environment state. A pytree.

    Attributes:
        particles: The swimmers; positions are kept inside one period of the flow.
        theta: (n_agents,) heading angles.
        displacement: (n_agents, 2) total displacement since the reset, not
            reduced to the period; its second component is the altitude gained.
        step: Number of control steps taken.
    """

    particles: pp.ParticleState
    theta: jax.Array
    displacement: jax.Array
    step: jax.Array


class Microswimmer:
    """Gyrotactic swimmers in a Taylor-Green flow.

    The swimmers are points without inertia and do not interact, so
    ``n_agents`` swimmers are ``n_agents`` independent copies of the task in
    the same flow.

    Args:
        n_agents: Number of swimmers.
        wave_number: Wave number k of the flow; its period is ``2 pi / k``.
        observation: ``"continuous"`` (the vorticity in units of ``u0 k`` and
            the cosine and sine of the heading angle) or ``"discrete"`` (one
            of 12 states, see :meth:`discrete_state`).
        action: ``"discrete"`` (the preferred direction is one of up, left,
            down and right, in this order) or ``"continuous"`` (a 2D vector
            that points in the preferred direction).
        vorticity_threshold: The vorticity counts as "close to zero" if its
            magnitude is below this fraction of ``u0 k``.
        dt: Physics time step.
        substeps: Physics steps per control step.
    """

    def __init__(
        self,
        n_agents: int = 1,
        wave_number: float = 1.0,
        observation: str = "continuous",
        action: str = "discrete",
        vorticity_threshold: float = 1.0 / 3.0,
        dt: float = 0.01,
        substeps: int = 10,
    ) -> None:
        if n_agents < 1:
            raise ValueError("n_agents must be at least 1")
        if observation not in ("continuous", "discrete"):
            raise ValueError("observation must be 'continuous' or 'discrete'")
        if action not in ("discrete", "continuous"):
            raise ValueError("action must be 'discrete' or 'continuous'")
        self.n_agents = int(n_agents)
        self.wave_number = float(wave_number)
        self.period = 2.0 * np.pi / self.wave_number
        self.observation, self.action = observation, action
        self.vorticity_threshold = float(vorticity_threshold)
        self.dt, self.substeps = float(dt), int(substeps)
        self.directions = _ring(4)
        self.box = pp.Box((0.0, 0.0), (self.period, self.period), periodic=True)
        self.force_field = pp.ForceField(self.box)

    @property
    def default_params(self) -> MicroswimmerParams:
        """Default episode parameters."""
        return MicroswimmerParams()

    @property
    def observation_space(self) -> Discrete | Continuous:
        """Observation of one swimmer."""
        if self.observation == "discrete":
            return Discrete(12)
        return Continuous(-1.0, 1.0, (3,))

    @property
    def action_space(self) -> Discrete | Continuous:
        """Action of one swimmer."""
        if self.action == "discrete":
            return Discrete(4)
        return Continuous(-1.0, 1.0, (2,))

    def flow(self, params: MicroswimmerParams) -> pp.TaylorGreen:
        """The flow field for the given parameters."""
        return pp.TaylorGreen(params.flow_speed, self.wave_number)

    def reset(
        self, key: jax.Array, params: MicroswimmerParams | None = None
    ) -> tuple[jax.Array, MicroswimmerState]:
        """Start an episode at random positions with random headings."""
        params = self.default_params if params is None else params
        pos_key, angle_key = jax.random.split(key)
        pos = self.period * jax.random.uniform(pos_key, (self.n_agents, 2))
        theta = jax.random.uniform(angle_key, (self.n_agents,), pos.dtype, -jnp.pi, jnp.pi)
        particles = pp.create_state(pos, radius=0.0)
        particles, _ = pp.init_forces(particles, self.force_field)
        state = MicroswimmerState(particles, theta, jnp.zeros_like(pos), jnp.zeros((), jnp.int32))
        return self._observe(state, params), state

    def step(
        self,
        key: jax.Array,
        state: MicroswimmerState,
        action: jax.Array,
        params: MicroswimmerParams | None = None,
    ):
        """Advance the environment by one control step.

        Returns:
            ``(obs, state, reward, terminated, truncated, info)``. The reward
            of a swimmer is the altitude it gained in this step. Episodes are
            never terminated; they are truncated after ``max_steps``.
            ``info`` holds the ``altitude`` gained since the reset.
        """
        params = self.default_params if params is None else params
        dtype = state.particles.pos.dtype
        flow = self.flow(params)
        target = self.preferred_angle(action, dtype)
        rate = 0.5 / params.alignment_time
        period = self.period

        def substep(carry, k):
            particles, theta, displacement = carry
            pos_key, angle_key = jax.random.split(k)
            pos = particles.pos
            turn = pp.active.steering_rate(theta, target, rate) + 0.5 * flow.vorticity(pos)
            drift = flow.velocity(pos) + params.swim_speed * pp.active.heading_vector(theta)
            particles, _ = pp.brownian(
                particles,
                self.force_field,
                self.dt,
                gamma=1.0,
                kT=params.translational_diffusion,
                drift=drift,
                key=pos_key,
            )
            moved = particles.pos - pos
            displacement = displacement + moved - period * jnp.round(moved / period)
            theta = pp.active.rotate(
                theta, self.dt, omega=turn, diffusion=params.rotational_diffusion, key=angle_key
            )
            return (particles, theta, displacement), None

        carry = (state.particles, state.theta, state.displacement)
        (particles, theta, displacement), _ = jax.lax.scan(
            substep, carry, jax.random.split(key, self.substeps)
        )
        step = state.step + 1
        reward = displacement[:, 1] - state.displacement[:, 1]
        state = MicroswimmerState(particles, theta, displacement, step)
        return (
            self._observe(state, params),
            state,
            reward,
            jnp.zeros(self.n_agents, bool),
            jnp.broadcast_to(step >= params.max_steps, (self.n_agents,)),
            {"altitude": displacement[:, 1]},
        )

    def preferred_angle(self, action: jax.Array, dtype=None) -> jax.Array:
        """(n_agents,) angles of the preferred directions for the given actions."""
        if self.action == "discrete":
            index = jnp.reshape(action, (self.n_agents,))
            return jnp.asarray(0.5 * jnp.pi * (1 + index), dtype)
        vector = jnp.reshape(jnp.asarray(action, dtype), (self.n_agents, 2))
        return jnp.arctan2(vector[:, 1], vector[:, 0])

    def discrete_state(self, state: MicroswimmerState, params: MicroswimmerParams) -> jax.Array:
        """(n_agents,) indices of the 12 coarse states of Colabrese et al. (2017).

        The index is ``4 * level + sector``. The level is 0, 1 or 2 for a
        vorticity that is negative, close to zero or positive; a positive
        vorticity turns the swimmer counter-clockwise. The sector is the
        direction closest to the heading: 0 up, 1 left, 2 down, 3 right.
        """
        scale = params.flow_speed * self.wave_number
        vorticity = self.flow(params).vorticity(state.particles.pos)
        threshold = self.vorticity_threshold * scale
        level = 1 + jnp.sign(vorticity).astype(jnp.int32) * (jnp.abs(vorticity) > threshold)
        sector = jnp.round((state.theta - 0.5 * jnp.pi) / (0.5 * jnp.pi)).astype(jnp.int32) % 4
        return 4 * level + sector

    def naive_policy(self, state: MicroswimmerState | None = None) -> jax.Array:
        """The preferred direction is always up (naive gyrotaxis)."""
        if self.action == "discrete":
            return jnp.zeros(self.n_agents, jnp.int32)
        return jnp.broadcast_to(jnp.array([0.0, 1.0]), (self.n_agents, 2))

    def render(
        self,
        state: MicroswimmerState,
        width: int = 360,
        params: MicroswimmerParams | None = None,
        color=viz.BLUE,
        periods: tuple[int, int] = (1, 1),
        start_row: int = 0,
    ) -> np.ndarray:
        """Draw the swimmers over the vorticity (grey scale, dark where it is negative).

        By default the picture shows one period of the flow; swimmers that
        leave on one side come back on the other. With several rows the
        picture shows the altitude instead: a swimmer is drawn at the height
        it has reached since the reset, and it is not drawn while it is
        above or below the picture.

        Args:
            state: Environment state.
            width: Width of the picture in pixels.
            params: Episode parameters.
            color: Colour of the swimmers.
            periods: Numbers of periods of the flow shown along x and along
                the altitude, ``(columns, rows)``.
            start_row: Row, counted from the bottom, in which the swimmers
                are at the reset. Used with more than one row.
        """
        params = self.default_params if params is None else params
        columns, rows = periods
        size = self.period
        hi = (columns * size, rows * size)
        canvas = viz.Canvas((0.0, 0.0), hi, width)
        grid = pp.Grid((0.0, 0.0), hi, (64 * columns, 64 * rows))
        vorticity = self.flow(params).vorticity(grid.centres().reshape(-1, 2)).reshape(grid.shape)
        canvas.field(vorticity, colors=((95, 95, 95), viz.WHITE))
        pos = np.asarray(state.particles.pos)
        # The swimmers are spread over the columns; the flow is the same in each.
        x = np.mod(pos[:, 0] + size * (np.arange(self.n_agents) % columns), hi[0])
        y = np.mod(pos[:, 1], size)
        if rows > 1:
            gained = np.asarray(state.displacement)[:, 1]
            y = np.mod(pos[:, 1] - gained, size) + gained + start_row * size
        inside = (y >= 0.0) & (y <= hi[1])
        canvas.discs(np.stack([x, y], axis=1)[inside], 0.012 * hi[0], color)
        return canvas.frame()

    def _observe(self, state: MicroswimmerState, params: MicroswimmerParams) -> jax.Array:
        if self.observation == "discrete":
            return self.discrete_state(state, params)
        scale = params.flow_speed * self.wave_number
        vorticity = self.flow(params).vorticity(state.particles.pos) / scale
        return jnp.stack([vorticity, jnp.cos(state.theta), jnp.sin(state.theta)], axis=1)
