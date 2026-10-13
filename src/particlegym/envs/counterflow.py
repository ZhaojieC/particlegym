"""Counter-flow in a corridor.

Two groups of pedestrians walk in opposite directions along a corridor that
is periodic along its length. The physics is the social force model of the
evacuation environment. Agents choose their desired direction; the other
pedestrians want to walk straight ahead.

Observations and actions of an agent are given in its own frame, in which it
walks along +x: for the group that walks along -x, all vectors are rotated by
180 degrees. One policy can therefore be shared by both groups, and a rule
such as "keep to the right" means the same for everybody.

This environment is experimental: the parameters of the social force model
are the published ones and are not calibrated against measured flows.
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


class CounterFlowParams(NamedTuple):
    """Episode parameters. A pytree, so values may be traced or batched.

    Attributes:
        desired_speed: Desired walking speed v0 (m/s).
        relaxation_time: Relaxation time tau of the social force model (s).
        noise_kT: Thermal energy of random velocity fluctuations (J).
        max_steps: Episode length in control steps.
    """

    desired_speed: float = 1.3
    relaxation_time: float = 0.5
    noise_kT: float = 0.0
    max_steps: int = 600


class CounterFlowState(NamedTuple):
    """Environment state. A pytree.

    Attributes:
        particles: The pedestrians; the first ``n_agents`` are the agents.
            ``kind`` is the group: 0 walks along +x, 1 along -x.
        step: Number of control steps taken.
    """

    particles: pp.ParticleState
    step: jax.Array


class CounterFlow:
    """Two groups of pedestrians that pass each other in a corridor.

    The corridor spans ``[0, length]`` along x, where it is periodic, and
    ``[0, width]`` along y, between two walls. Pedestrians with an even index
    walk along +x, those with an odd index along -x.

    Args:
        length: Length of the periodic corridor (m).
        width: Distance between the walls (m).
        n_agents: Number of pedestrians controlled by actions.
        n_crowd: Number of other pedestrians, which want to walk straight on.
        action: ``"discrete"`` (one of ``n_directions`` directions) or
            ``"continuous"`` (a 2D vector; its length, capped at 1, scales
            the desired speed). Direction 0 of the discrete set is straight
            ahead, the following ones turn counter-clockwise.
        n_directions: Number of discrete directions.
        n_neighbors: Number of nearest pedestrians in an observation.
        radius: Pedestrian radius (m), or a ``(min, max)`` range.
        mass: Pedestrian mass (kg).
        social_force: Contact model; defaults to
            :class:`~particlegym.physics.SocialForce`.
        friction: Sliding friction coefficient kappa.
        dt: Physics time step (s).
        substeps: Physics steps per control step.
        min_gap: Minimum clearance between pedestrians at reset (m).
    """

    def __init__(
        self,
        length: float = 30.0,
        width: float = 6.0,
        n_agents: int = 1,
        n_crowd: int = 79,
        action: str = "continuous",
        n_directions: int = 8,
        n_neighbors: int = 6,
        radius=(0.22, 0.28),
        mass: float = 80.0,
        social_force: pp.SocialForce | None = None,
        friction: float = 2.4e5,
        dt: float = 0.005,
        substeps: int = 20,
        min_gap: float = 0.2,
    ) -> None:
        if length <= 0 or width <= 0:
            raise ValueError("length and width must be positive")
        if n_agents < 0 or n_crowd < 0 or n_agents + n_crowd < 2:
            raise ValueError("need at least two pedestrians, and no negative counts")
        if action not in ("discrete", "continuous"):
            raise ValueError("action must be 'discrete' or 'continuous'")
        r_min, r_max = (float(radius), float(radius)) if np.isscalar(radius) else radius
        if not 0 < r_min <= r_max:
            raise ValueError("radius must be positive, and min <= max for a range")
        self.size = (float(length), float(width))
        self.n_agents, self.n_crowd = int(n_agents), int(n_crowd)
        self.n = self.n_agents + self.n_crowd
        self.n_neighbors = min(int(n_neighbors), self.n - 1)
        self.action = action
        self.radius = (r_min, r_max)
        self.mass = float(mass)
        self.dt, self.substeps = float(dt), int(substeps)
        # Rotated so that direction 0 is straight ahead (+x in the own frame).
        self.directions = _ring(n_directions) @ np.array([[0.0, -1.0], [1.0, 0.0]])
        self.sign = np.where(np.arange(self.n) % 2 == 0, 1.0, -1.0)

        social_force = pp.SocialForce() if social_force is None else social_force
        # Geometry is not periodic, so the two walls extend beyond the box.
        walls = pp.Plates.from_segments(
            [[(-length, y), (2 * length, y)] for y in (0.0, float(width))]
        )
        pair = [social_force] + ([pp.TangentialFriction(friction)] if friction else [])
        margin = 1.0
        self.box = pp.Box((0.0, -margin), (length, width + margin), periodic=(True, False))
        neighbors = pp.BruteForce()
        if self.n > 512:
            neighbors = pp.CellList(cutoff=social_force.cutoff(2.0 * r_max), capacity=32)
        self.force_field = pp.ForceField(
            self.box,
            pair=pair,
            walls=[pp.Walls(walls, social_force, friction)],
            neighbors=neighbors,
        )

        self._jitter = 0.1
        spacing = 2 * r_max + float(min_gap) + 2 * self._jitter
        nx, ny = int(length // spacing), int((width - spacing) // spacing) + 1
        if nx * ny < self.n:
            raise ValueError(
                f"the corridor has space for {nx * ny} pedestrians at reset, "
                f"but {self.n} were requested"
            )
        x = (np.arange(nx) + 0.5) * (length / nx)
        y = width / 2 + (np.arange(ny) - (ny - 1) / 2) * spacing
        self._sites = np.stack(np.meshgrid(x, y), axis=-1).reshape(-1, 2)

    @property
    def default_params(self) -> CounterFlowParams:
        """Default episode parameters."""
        return CounterFlowParams()

    @property
    def observation_space(self) -> Continuous:
        """Observation of one agent; see :meth:`step`."""
        return Continuous(-jnp.inf, jnp.inf, (3 + 5 * self.n_neighbors,))

    @property
    def action_space(self) -> Discrete | Continuous:
        """Action of one agent."""
        if self.action == "discrete":
            return Discrete(len(self.directions))
        return Continuous(-1.0, 1.0, (2,))

    def reset(
        self, key: jax.Array, params: CounterFlowParams | None = None
    ) -> tuple[jax.Array, CounterFlowState]:
        """Start an episode with the two groups mixed at random positions."""
        site_key, jitter_key, radius_key = jax.random.split(key, 3)
        sites = jnp.asarray(self._sites)
        chosen = jax.random.choice(site_key, sites.shape[0], (self.n,), replace=False)
        jitter = jax.random.uniform(
            jitter_key, (self.n, 2), sites.dtype, -self._jitter, self._jitter
        )
        r_min, r_max = self.radius
        radius = r_min + (r_max - r_min) * jax.random.uniform(radius_key, (self.n,), sites.dtype)
        particles = pp.create_state(
            sites[chosen] + jitter, mass=self.mass, radius=radius, kind=jnp.arange(self.n) % 2
        )
        particles, _ = pp.init_forces(particles, self.force_field)
        state = CounterFlowState(particles, jnp.zeros((), jnp.int32))
        return self._observe(state), state

    def step(
        self,
        key: jax.Array,
        state: CounterFlowState,
        action: jax.Array,
        params: CounterFlowParams | None = None,
    ):
        """Advance the environment by one control step.

        Returns:
            ``(obs, state, reward, terminated, truncated, info)``. An
            observation holds, in the agent's own frame, its lateral position
            (y scaled to [-0.5, 0.5]), its velocity, and for each of the
            ``n_neighbors`` nearest pedestrians the relative position, the
            relative velocity and +1 or -1 for the same or the other group.
            The reward of an agent is the distance (m) it has moved along its
            walking direction during the step. Episodes are never terminated;
            they are truncated after ``max_steps``. ``info`` holds the
            ``lane_order`` parameter, the mean ``speed`` along the walking
            directions and a neighbour-list ``overflow`` flag.
        """
        params = self.default_params if params is None else params
        particles = state.particles
        dtype = particles.pos.dtype
        sign = jnp.asarray(self.sign, dtype)[:, None]
        straight = jnp.asarray([1.0, 0.0], dtype) * jnp.ones((self.n_crowd, 1), dtype)
        direction = jnp.concatenate([self._agent_direction(action, dtype), straight])
        drift = params.desired_speed * sign * direction
        gamma = 1.0 / params.relaxation_time

        def substep(carry, substep_key):
            p, moved, overflow = carry
            new, info = pp.langevin_baoab(
                p, self.force_field, self.dt, gamma=gamma, kT=params.noise_kT, drift=drift,
                key=substep_key,
            )  # fmt: skip
            moved = moved + self.box.displacement(new.pos, p.pos)[:, 0]
            return (new, moved, overflow | info.overflow), None

        start = (particles, jnp.zeros(self.n, dtype), jnp.zeros((), bool))
        (particles, moved, overflow), _ = jax.lax.scan(
            substep, start, jax.random.split(key, self.substeps)
        )
        step = state.step + 1
        state = CounterFlowState(particles, step)
        reward = (sign[:, 0] * moved)[: self.n_agents]
        info = {
            "lane_order": self.lane_order(particles),
            "speed": jnp.mean(sign[:, 0] * particles.vel[:, 0]),
            "overflow": overflow,
        }
        terminated = jnp.zeros(self.n_agents, bool)
        truncated = jnp.broadcast_to(step >= params.max_steps, (self.n_agents,))
        return self._observe(state), state, reward, terminated, truncated, info

    def lane_order(self, particles: pp.ParticleState, band: float | None = None) -> jax.Array:
        """How well the two groups are separated into lanes, from 0 to 1.

        For every pedestrian, the pedestrians whose lateral position y
        differs by less than ``band`` (default: the largest diameter) are
        counted, itself included. With ``s`` of them in the same group and
        ``o`` in the other group, the pedestrian contributes
        ``((s - o) / (s + o))^2``. The result is the mean over the
        pedestrians: 1 for perfect lanes, small for a random mixture.
        """
        band = 2.0 * self.radius[1] if band is None else band
        y = particles.pos[:, 1]
        close = jnp.abs(y[:, None] - y[None, :]) < band
        same = particles.kind[:, None] == particles.kind[None, :]
        s = jnp.sum(close & same, axis=1)
        o = jnp.sum(close & ~same, axis=1)
        return jnp.mean(((s - o) / (s + o)) ** 2)

    def render(self, state: CounterFlowState, width: int = 640) -> np.ndarray:
        """Draw the corridor; the group walking to the right is red, the other blue."""
        length, height = self.size
        canvas = viz.Canvas((0.0, -0.5), (length, height + 0.5), width)
        canvas.segments(
            [[0.0, 0.0], [0.0, height]], [[length, 0.0], [length, height]], viz.BLACK, 0.12
        )
        p = state.particles
        colors = np.where(np.asarray(p.kind)[:, None] == 0, viz.RED, viz.BLUE)
        pos, radius = np.asarray(p.pos), np.asarray(p.radius)
        # Pedestrians that overlap an end of the corridor are drawn at both ends.
        for shift in (-length, 0.0, length):
            canvas.discs(pos + np.array([shift, 0.0]), radius, colors)
        return canvas.frame()

    def _agent_direction(self, action: jax.Array, dtype) -> jax.Array:
        """Desired directions (n_agents, 2) in the agents' own frames."""
        if self.action == "discrete":
            return jnp.asarray(self.directions, dtype)[jnp.reshape(action, (self.n_agents,))]
        direction = jnp.reshape(jnp.asarray(action, dtype), (self.n_agents, 2))
        length = jnp.linalg.norm(direction, axis=1, keepdims=True)
        return direction / jnp.maximum(length, 1.0)

    def _observe(self, state: CounterFlowState) -> jax.Array:
        p = state.particles
        k, m = self.n_neighbors, self.n_agents
        sign = jnp.asarray(self.sign, p.pos.dtype)
        dr = self.box.displacement(p.pos[None, :, :], p.pos[:m, None, :])  # from agent to other
        distance = jnp.sum(dr * dr, axis=-1)
        distance = jnp.where(
            jnp.arange(self.n)[None, :] == jnp.arange(m)[:, None], jnp.inf, distance
        )
        _, nearest = jax.lax.top_k(-distance, k)
        own = sign[:m, None]
        relative = jnp.take_along_axis(dr, nearest[:, :, None], axis=1) * own[:, :, None]
        velocity = (p.vel[nearest] - p.vel[:m, None, :]) * own[:, :, None]
        same = sign[nearest] * own
        lateral = (p.pos[:m, 1] / self.size[1] - 0.5) * sign[:m]
        return jnp.concatenate(
            [
                lateral[:, None],
                p.vel[:m] * own,
                relative.reshape(m, 2 * k),
                velocity.reshape(m, 2 * k),
                same.reshape(m, k),
            ],
            axis=1,
        )
