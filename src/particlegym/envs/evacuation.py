"""Room evacuation environment.

Pedestrians move in a rectangular room with doors in its walls, following the
social force model. Some of them are reinforcement-learning agents: an action
sets an agent's desired direction of motion, and the motion follows from the
physics (design principle P1 in ``docs/design.md``). Pedestrians who have left
the room are switched off with the alive mask.

The environment is functional and JAX-native::

    env = Evacuation(exits=[Opening("y+", (4.5, 5.5))])
    params = env.default_params
    obs, state = env.reset(key, params)
    obs, state, reward, terminated, truncated, info = env.step(key, state, action, params)

``env`` holds the static structure (room, doors, obstacles, numbers of
pedestrians, physics time step). ``params`` holds values that may change
between episodes and can be batched with :func:`jax.vmap` (desired speed,
rewards, episode length). :meth:`Evacuation.reset` and :meth:`Evacuation.step`
are pure functions that can be compiled with :func:`jax.jit`.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from particlegym import physics as pp

CROWD_POLICIES = ("nearest_exit", "random", "stand")

# Pedestrians walking to a door aim at a point this far (m) beyond its centre,
# so that they keep walking once they are in the doorway.
_TARGET_DISTANCE = 1.0
# Outward unit normal of each face of the room.
_FACE_NORMALS = {"x-": (-1.0, 0.0), "x+": (1.0, 0.0), "y-": (0.0, -1.0), "y+": (0.0, 1.0)}


class EvacuationParams(NamedTuple):
    """Episode parameters. A pytree, so values may be traced or batched.

    Attributes:
        desired_speed: Desired walking speed v0 (m/s).
        relaxation_time: Relaxation time tau of the social force model (s).
        noise_kT: Thermal energy of random velocity fluctuations (J); 0 for
            deterministic motion.
        step_penalty: Reward for every control step an agent is in the room.
        exit_reward: Reward for the control step in which an agent leaves.
        max_steps: Episode length in control steps.
    """

    desired_speed: float = 2.0
    relaxation_time: float = 0.5
    noise_kT: float = 0.0
    step_penalty: float = -0.1
    exit_reward: float = 0.0
    max_steps: int = 1000


class EvacuationState(NamedTuple):
    """Environment state. A pytree.

    Attributes:
        particles: The pedestrians; the first ``n_agents`` are the agents.
        step: Number of control steps taken.
        exit_time: (N,) time (s) at which each pedestrian left the room;
            infinite while inside.
    """

    particles: pp.ParticleState
    step: jax.Array
    exit_time: jax.Array


class Evacuation:
    """Pedestrians evacuating a rectangular room through doors.

    The room spans ``[0, width] x [0, height]``. Its walls are closed except at
    the doors, so a pedestrian whose centre is outside the room by more than
    its radius has passed through a door and is removed.

    Args:
        room: Width and height of the room (m).
        exits: The doors, as :class:`~particlegym.physics.Opening` instances on
            the faces ``"x-"``, ``"x+"``, ``"y-"`` and ``"y+"``.
        obstacles: Shape sets that pedestrians cannot enter, for example
            :class:`~particlegym.physics.Spheres` for pillars.
        n_agents: Number of pedestrians controlled by actions.
        n_crowd: Number of other pedestrians, which follow ``crowd_policy``.
        crowd_policy: ``"nearest_exit"`` (walk to the nearest door),
            ``"random"`` (a new random direction every control step) or
            ``"stand"`` (no desired motion).
        action: ``"discrete"`` (one of ``n_directions`` evenly spaced
            directions) or ``"continuous"`` (a 2D vector; its length, capped
            at 1, scales the desired speed).
        n_directions: Number of discrete directions. Direction k points at
            ``90 + 360 k / n_directions`` degrees, so the eight default
            directions are up, up-left, left, down-left, down, down-right,
            right and up-right, as in Zhang, Chai & Lykotrafitis (2021).
        radius: Pedestrian radius (m), or a ``(min, max)`` range sampled
            uniformly at reset.
        mass: Pedestrian mass (kg).
        social_force: Contact model between pedestrians and with walls and
            obstacles. Defaults to :class:`~particlegym.physics.SocialForce`.
        friction: Sliding friction coefficient kappa (0 disables friction).
        dt: Physics time step (s).
        substeps: Physics steps per control step; the control interval is
            ``dt * substeps``.
        min_gap: Minimum clearance between pedestrians at reset (m).

    Attributes:
        directions: (n_directions, 2) unit vectors of the discrete actions.
        door_centres: (n_exits, 2) centres of the doors.
        door_targets: (n_exits, 2) points 1 m beyond the door centres, at which
            pedestrians walking to a door aim.
        box: Simulation domain (the room plus a 2 m margin).
        force_field: The interactions between pedestrians, walls and obstacles.
    """

    def __init__(
        self,
        room=(10.0, 10.0),
        exits=(),
        obstacles=(),
        n_agents: int = 1,
        n_crowd: int = 0,
        crowd_policy: str = "nearest_exit",
        action: str = "discrete",
        n_directions: int = 8,
        radius=0.25,
        mass: float = 80.0,
        social_force: pp.SocialForce | None = None,
        friction: float = 2.4e5,
        dt: float = 0.005,
        substeps: int = 20,
        min_gap: float = 0.2,
    ) -> None:
        self.width, self.height = (float(x) for x in room)
        if self.width <= 0 or self.height <= 0:
            raise ValueError("room dimensions must be positive")
        if n_agents < 0 or n_crowd < 0 or n_agents + n_crowd == 0:
            raise ValueError("need at least one pedestrian, and no negative counts")
        if crowd_policy not in CROWD_POLICIES:
            raise ValueError(f"crowd_policy must be one of {CROWD_POLICIES}")
        if action not in ("discrete", "continuous"):
            raise ValueError("action must be 'discrete' or 'continuous'")
        if n_directions < 2:
            raise ValueError("n_directions must be at least 2")
        if dt <= 0 or substeps < 1:
            raise ValueError("dt must be positive and substeps at least 1")
        r_min, r_max = (float(radius), float(radius)) if np.isscalar(radius) else radius
        if not 0 < r_min <= r_max:
            raise ValueError("radius must be positive, and min <= max for a range")

        self.exits = tuple(exits)
        for opening in self.exits:
            if opening.face not in _FACE_NORMALS:
                raise ValueError(f"exit face {opening.face!r} is not one of {list(_FACE_NORMALS)}")
        self.obstacles = tuple(obstacles)
        for shapes in self.obstacles:
            if shapes.dim != 2:
                raise ValueError("obstacles must be two-dimensional")
        self.n_agents, self.n_crowd = int(n_agents), int(n_crowd)
        self.n = self.n_agents + self.n_crowd
        self.crowd_policy = crowd_policy
        self.action = action
        self.radius = (r_min, r_max)
        self.mass = float(mass)
        self.dt, self.substeps = float(dt), int(substeps)

        social_force = pp.SocialForce() if social_force is None else social_force
        walls = [pp.Walls(pp.box_walls((0.0, 0.0), room, self.exits), social_force, friction)]
        walls += [pp.Walls(shapes, social_force, friction) for shapes in self.obstacles]
        pair = [social_force] + ([pp.TangentialFriction(friction)] if friction else [])
        if self.n <= 512:
            neighbors = pp.BruteForce()
        else:
            neighbors = pp.CellList(cutoff=social_force.cutoff(2.0 * r_max), capacity=32)
        margin = 2.0  # room around the walls for pedestrians walking out of a door
        self.box = pp.Box((-margin, -margin), (self.width + margin, self.height + margin))
        self.force_field = pp.ForceField(self.box, pair=pair, walls=walls, neighbors=neighbors)

        centres = np.array([_door_centre(o, self.width, self.height) for o in self.exits])
        normals = np.array([_FACE_NORMALS[o.face] for o in self.exits])
        self.door_centres = centres.reshape(-1, 2)
        self.door_targets = self.door_centres + _TARGET_DISTANCE * normals.reshape(-1, 2)

        angles = math.pi / 2 + 2 * math.pi * np.arange(n_directions) / n_directions
        self.directions = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        self.directions[np.abs(self.directions) < 1e-12] = 0.0

        self._jitter = 0.1
        self._sites = self._initial_sites(r_max, float(min_gap))
        if len(self._sites) < self.n:
            raise ValueError(
                f"the room has space for {len(self._sites)} pedestrians at reset, "
                f"but {self.n} were requested; lower min_gap or the number of pedestrians"
            )

    @property
    def default_params(self) -> EvacuationParams:
        """Default episode parameters."""
        return EvacuationParams()

    def reset(
        self, key: jax.Array, params: EvacuationParams | None = None
    ) -> tuple[jax.Array, EvacuationState]:
        """Start an episode with pedestrians at random, non-overlapping positions.

        Returns:
            The agents' observations, shape (n_agents, 4), and the state.
        """
        site_key, jitter_key, radius_key = jax.random.split(key, 3)
        sites = jnp.asarray(self._sites)
        chosen = jax.random.choice(site_key, sites.shape[0], (self.n,), replace=False)
        jitter = jax.random.uniform(
            jitter_key, (self.n, 2), sites.dtype, -self._jitter, self._jitter
        )
        r_min, r_max = self.radius
        radius = r_min + (r_max - r_min) * jax.random.uniform(radius_key, (self.n,), sites.dtype)
        kind = jnp.concatenate(
            [jnp.zeros(self.n_agents, jnp.int32), jnp.ones(self.n_crowd, jnp.int32)]
        )
        particles = pp.create_state(
            sites[chosen] + jitter, mass=self.mass, radius=radius, kind=kind
        )
        particles, _ = pp.init_forces(particles, self.force_field)
        state = EvacuationState(
            particles=particles,
            step=jnp.zeros((), jnp.int32),
            exit_time=jnp.full((self.n,), jnp.inf, sites.dtype),
        )
        return self._observe(state), state

    def step(
        self,
        key: jax.Array,
        state: EvacuationState,
        action: jax.Array,
        params: EvacuationParams | None = None,
    ):
        """Advance the environment by one control step.

        Args:
            key: JAX random key.
            state: Current state.
            action: Agent actions: integer directions, shape (n_agents,), or
                vectors, shape (n_agents, 2), depending on ``action``.
            params: Episode parameters; defaults to :attr:`default_params`.

        Returns:
            ``(obs, state, reward, terminated, truncated, info)``. ``obs`` has
            shape (n_agents, 4); ``reward``, ``terminated`` and ``truncated``
            have shape (n_agents,). ``terminated`` is true once an agent has
            left the room, ``truncated`` when the episode reaches
            ``max_steps`` with the agent still inside. ``info`` holds the
            elapsed ``time``, the number of pedestrians ``inside``, the mask
            of pedestrians that ``left`` in this step and a neighbour-list
            ``overflow`` flag.
        """
        params = self.default_params if params is None else params
        particles = state.particles
        crowd_key, noise_key = jax.random.split(key)
        drift = jnp.concatenate(
            [
                self.agent_velocity(action, params, particles.pos.dtype),
                self.crowd_velocity(crowd_key, particles, params),
            ]
        )
        gamma = 1.0 / params.relaxation_time

        def substep(carry, substep_key):
            p, overflow = carry
            p, info = pp.langevin_baoab(
                p,
                self.force_field,
                self.dt,
                gamma=gamma,
                kT=params.noise_kT,
                drift=drift,
                key=substep_key,
            )
            return (p, overflow | info.overflow), None

        (particles, overflow), _ = jax.lax.scan(
            substep,
            (particles, jnp.zeros((), bool)),
            jax.random.split(noise_key, self.substeps),
        )
        left = particles.alive & self._outside(particles)
        particles = particles._replace(alive=particles.alive & ~left)
        step = state.step + 1
        time = step * (self.substeps * self.dt)
        state = EvacuationState(particles, step, jnp.where(left, time, state.exit_time))

        # Agents still inside pay the step penalty, agents that left in this
        # step get the exit reward, and agents that left earlier get nothing.
        inside = particles.alive[: self.n_agents]
        reward = jnp.where(
            inside,
            params.step_penalty,
            jnp.where(left[: self.n_agents], params.exit_reward, 0.0),
        )
        terminated = ~inside
        truncated = inside & (step >= params.max_steps)
        info = {
            "time": time,
            "inside": jnp.sum(particles.alive),
            "left": left,
            "overflow": overflow,
        }
        return self._observe(state), state, reward, terminated, truncated, info

    def agent_velocity(self, action: jax.Array, params: EvacuationParams, dtype) -> jax.Array:
        """Desired velocities (n_agents, 2) of the agents for the given actions."""
        if self.action == "discrete":
            index = jnp.reshape(action, (self.n_agents,))
            direction = jnp.asarray(self.directions, dtype)[index]
        else:
            direction = jnp.reshape(jnp.asarray(action, dtype), (self.n_agents, 2))
            length = jnp.linalg.norm(direction, axis=1, keepdims=True)
            direction = direction / jnp.maximum(length, 1.0)
        return params.desired_speed * direction

    def crowd_velocity(
        self, key: jax.Array, particles: pp.ParticleState, params: EvacuationParams
    ) -> jax.Array:
        """Desired velocities (n_crowd, 2) of the crowd under ``crowd_policy``."""
        pos = particles.pos[self.n_agents :]
        if self.crowd_policy == "stand" or (
            self.crowd_policy == "nearest_exit" and not self.exits
        ):
            return jnp.zeros_like(pos)
        if self.crowd_policy == "random":
            angle = jax.random.uniform(key, (self.n_crowd,), pos.dtype, 0.0, 2 * math.pi)
            direction = jnp.stack([jnp.cos(angle), jnp.sin(angle)], axis=1)
        else:
            centres = jnp.asarray(self.door_centres, pos.dtype)
            nearest = jnp.argmin(jnp.sum((pos[:, None, :] - centres) ** 2, axis=-1), axis=1)
            offset = jnp.asarray(self.door_targets, pos.dtype)[nearest] - pos
            length = jnp.linalg.norm(offset, axis=1, keepdims=True)
            direction = offset / jnp.maximum(length, 1e-9)
        return params.desired_speed * direction

    def _observe(self, state: EvacuationState) -> jax.Array:
        """Agent observations: position scaled to [-0.5, 0.5] in the room, and velocity."""
        p = state.particles
        size = jnp.asarray([self.width, self.height], p.pos.dtype)
        position = p.pos[: self.n_agents] / size - 0.5
        return jnp.concatenate([position, p.vel[: self.n_agents]], axis=1)

    def _outside(self, particles: pp.ParticleState) -> jax.Array:
        """Pedestrians whose centre is outside the room by more than their radius."""
        x, y, r = particles.pos[:, 0], particles.pos[:, 1], particles.radius
        return (x < -r) | (x > self.width + r) | (y < -r) | (y > self.height + r)

    def _initial_sites(self, r_max: float, min_gap: float) -> np.ndarray:
        """Lattice of candidate start positions, clear of walls and obstacles.

        Neighbouring sites are far enough apart that pedestrians keep a
        clearance of ``min_gap`` after a random shift of up to ``_jitter``.
        """
        spacing = 2 * r_max + min_gap + 2 * self._jitter
        margin = r_max + min_gap / 2 + self._jitter

        def axis(length):
            count = int(np.floor((length - 2 * margin) / spacing)) + 1
            if count < 1:
                return np.zeros(0)
            start = (length - (count - 1) * spacing) / 2
            return start + spacing * np.arange(count)

        sites = np.stack(np.meshgrid(axis(self.width), axis(self.height)), axis=-1)
        sites = sites.reshape(-1, 2)
        for shapes in self.obstacles:
            if len(sites):
                clearance = np.asarray(shapes.distance(jnp.asarray(sites))).min(axis=1)
                sites = sites[clearance > margin]
        return sites


def _door_centre(opening: pp.Opening, width: float, height: float) -> tuple[float, float]:
    start, end = (float(s) for s in opening.span)
    middle = (start + end) / 2
    return {
        "x-": (0.0, middle),
        "x+": (width, middle),
        "y-": (middle, 0.0),
        "y+": (middle, height),
    }[opening.face]
