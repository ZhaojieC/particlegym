"""Room evacuation environment.

Pedestrians move in a rectangular room with doors in its walls, following the
social force model. Some of them are reinforcement-learning agents: an action
sets an agent's desired direction of motion, and the motion follows from the
physics (design principle P1 in ``docs/design.md``). Pedestrians who have left
the room are switched off with the alive mask.

The room is two- or three-dimensional, depending on the number of entries of
``room``. In 2D, pedestrians are discs on a floor. In 3D, they are spheres
that move freely in all three directions, without gravity: a model for
swimmers and other self-propelled particles that leave a chamber, not for
people walking on a floor.

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
_AXES = "xyz"
# Sizes of the discrete direction sets in 3D, with the number of directions
# that are tilted out of the xy plane on each side of it.
_TILTED_DIRECTIONS = {6: 0, 18: 4, 26: 8}


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
    """Pedestrians evacuating a rectangular room through doors, in 2D or 3D.

    The room spans ``[0, room[a]]`` along every axis ``a``; the number of
    entries of ``room`` sets the number of dimensions d. Its walls are closed
    except at the doors, so a pedestrian whose centre is outside the room by
    more than its radius has passed through a door and is removed.

    In 3D the pedestrians are spheres that move freely in all three
    directions; there is no gravity.

    Args:
        room: Edge lengths of the room (m): two values for a 2D room, three
            for a 3D room.
        exits: The doors, as :class:`~particlegym.physics.Opening` instances on
            the faces ``"x-"``, ``"x+"``, ``"y-"`` and ``"y+"``, and in 3D also
            ``"z-"`` and ``"z+"``. In 2D the span of a door is a
            ``(start, end)`` pair; in 3D it is one ``(start, end)`` pair for
            each of the two other axes, and a face can have one door.
        obstacles: Shape sets that pedestrians cannot enter, for example
            :class:`~particlegym.physics.Spheres` for pillars.
        n_agents: Number of pedestrians controlled by actions.
        n_crowd: Number of other pedestrians, which follow ``crowd_policy``.
        crowd_policy: ``"nearest_exit"`` (walk to the nearest door),
            ``"random"`` (a new random direction every control step) or
            ``"stand"`` (no desired motion).
        action: ``"discrete"`` (one of ``n_directions`` directions) or
            ``"continuous"`` (a d-dimensional vector; its length, capped at 1,
            scales the desired speed).
        n_directions: Number of discrete directions; 8 in 2D and 26 in 3D if
            omitted. In 2D, direction k points at ``90 + 360 k / n_directions``
            degrees, so the eight default directions are up, up-left, left,
            down-left, down, down-right, right and up-right, as in Zhang, Chai
            & Lykotrafitis (2021). In 3D, the directions point from the centre
            of a cube to its 6 faces, to its faces and 12 edges (18), or to
            its faces, edges and 8 corners (26). They are ordered as the
            directions in the xy plane (in the 2D order), then +z followed by
            the directions tilted towards +z, then -z followed by the
            directions tilted towards -z.
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
        dim: Number of dimensions d of the room, 2 or 3.
        size: Edge lengths of the room.
        directions: (n_directions, d) unit vectors of the discrete actions.
        door_centres: (n_exits, d) centres of the doors.
        door_targets: (n_exits, d) points 1 m beyond the door centres, at which
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
        n_directions: int | None = None,
        radius=0.25,
        mass: float = 80.0,
        social_force: pp.SocialForce | None = None,
        friction: float = 2.4e5,
        dt: float = 0.005,
        substeps: int = 20,
        min_gap: float = 0.2,
    ) -> None:
        self.size = tuple(float(x) for x in room)
        self.dim = len(self.size)
        if self.dim not in (2, 3) or any(x <= 0 for x in self.size):
            raise ValueError("room must have two or three positive edge lengths")
        if n_agents < 0 or n_crowd < 0 or n_agents + n_crowd == 0:
            raise ValueError("need at least one pedestrian, and no negative counts")
        if crowd_policy not in CROWD_POLICIES:
            raise ValueError(f"crowd_policy must be one of {CROWD_POLICIES}")
        if action not in ("discrete", "continuous"):
            raise ValueError("action must be 'discrete' or 'continuous'")
        if n_directions is None:
            n_directions = 8 if self.dim == 2 else 26
        if self.dim == 2 and n_directions < 2:
            raise ValueError("n_directions must be at least 2")
        if self.dim == 3 and n_directions not in _TILTED_DIRECTIONS:
            raise ValueError(f"n_directions must be one of {list(_TILTED_DIRECTIONS)} in 3D")
        if dt <= 0 or substeps < 1:
            raise ValueError("dt must be positive and substeps at least 1")
        r_min, r_max = (float(radius), float(radius)) if np.isscalar(radius) else radius
        if not 0 < r_min <= r_max:
            raise ValueError("radius must be positive, and min <= max for a range")

        faces = [axis + sign for axis in _AXES[: self.dim] for sign in "-+"]
        span_shape = (2,) if self.dim == 2 else (2, 2)
        self.exits = tuple(exits)
        for opening in self.exits:
            if opening.face not in faces:
                raise ValueError(f"exit face {opening.face!r} is not one of {faces}")
            if np.shape(opening.span) != span_shape:
                raise ValueError(
                    f"the span of an exit of a {self.dim}D room must have shape {span_shape}"
                )
        self.obstacles = tuple(obstacles)
        for shapes in self.obstacles:
            if shapes.dim != self.dim:
                raise ValueError(f"obstacles of a {self.dim}D room must be {self.dim}D shapes")
        self.n_agents, self.n_crowd = int(n_agents), int(n_crowd)
        self.n = self.n_agents + self.n_crowd
        self.crowd_policy = crowd_policy
        self.action = action
        self.radius = (r_min, r_max)
        self.mass = float(mass)
        self.dt, self.substeps = float(dt), int(substeps)

        social_force = pp.SocialForce() if social_force is None else social_force
        origin = (0.0,) * self.dim
        walls = [pp.Walls(pp.box_walls(origin, self.size, self.exits), social_force, friction)]
        walls += [pp.Walls(shapes, social_force, friction) for shapes in self.obstacles]
        pair = [social_force] + ([pp.TangentialFriction(friction)] if friction else [])
        # With the default sizes, a cell holds about 10 close-packed discs but
        # about 40 close-packed spheres. A 3D cell list then checks 27 cells of
        # up to 48 pedestrians, and was faster than brute force only above
        # about 2000 pedestrians (measured on CPU).
        capacity, brute_force_limit = (32, 512) if self.dim == 2 else (48, 2000)
        if self.n <= brute_force_limit:
            neighbors = pp.BruteForce()
        else:
            neighbors = pp.CellList(cutoff=social_force.cutoff(2.0 * r_max), capacity=capacity)
        margin = 2.0  # room around the walls for pedestrians walking out of a door
        self.box = pp.Box((-margin,) * self.dim, tuple(x + margin for x in self.size))
        self.force_field = pp.ForceField(self.box, pair=pair, walls=walls, neighbors=neighbors)

        centres = np.array([_door_centre(o, self.size) for o in self.exits])
        normals = np.array([_face_normal(o.face, self.dim) for o in self.exits])
        self.door_centres = centres.reshape(-1, self.dim)
        self.door_targets = self.door_centres + _TARGET_DISTANCE * normals.reshape(-1, self.dim)

        self.directions = _directions(self.dim, n_directions)

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
            The agents' observations, shape (n_agents, 2 d), and the state.
        """
        site_key, jitter_key, radius_key = jax.random.split(key, 3)
        sites = jnp.asarray(self._sites)
        chosen = jax.random.choice(site_key, sites.shape[0], (self.n,), replace=False)
        jitter = jax.random.uniform(
            jitter_key, (self.n, self.dim), sites.dtype, -self._jitter, self._jitter
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
                vectors, shape (n_agents, d), depending on ``action``.
            params: Episode parameters; defaults to :attr:`default_params`.

        Returns:
            ``(obs, state, reward, terminated, truncated, info)``. ``obs`` has
            shape (n_agents, 2 d); ``reward``, ``terminated`` and ``truncated``
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
        """Desired velocities (n_agents, d) of the agents for the given actions."""
        if self.action == "discrete":
            index = jnp.reshape(action, (self.n_agents,))
            direction = jnp.asarray(self.directions, dtype)[index]
        else:
            direction = jnp.reshape(jnp.asarray(action, dtype), (self.n_agents, self.dim))
            length = jnp.linalg.norm(direction, axis=1, keepdims=True)
            direction = direction / jnp.maximum(length, 1.0)
        return params.desired_speed * direction

    def crowd_velocity(
        self, key: jax.Array, particles: pp.ParticleState, params: EvacuationParams
    ) -> jax.Array:
        """Desired velocities (n_crowd, d) of the crowd under ``crowd_policy``."""
        pos = particles.pos[self.n_agents :]
        if self.crowd_policy == "stand" or (
            self.crowd_policy == "nearest_exit" and not self.exits
        ):
            return jnp.zeros_like(pos)
        if self.crowd_policy == "random" and self.dim == 2:
            angle = jax.random.uniform(key, (self.n_crowd,), pos.dtype, 0.0, 2 * math.pi)
            direction = jnp.stack([jnp.cos(angle), jnp.sin(angle)], axis=1)
        elif self.crowd_policy == "random":
            # A normalised Gaussian vector is uniformly distributed on the sphere.
            direction = jax.random.normal(key, pos.shape, pos.dtype)
            direction = direction / jnp.linalg.norm(direction, axis=1, keepdims=True)
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
        size = jnp.asarray(self.size, p.pos.dtype)
        position = p.pos[: self.n_agents] / size - 0.5
        return jnp.concatenate([position, p.vel[: self.n_agents]], axis=1)

    def _outside(self, particles: pp.ParticleState) -> jax.Array:
        """Pedestrians whose centre is outside the room by more than their radius."""
        pos, r = particles.pos, particles.radius[:, None]
        size = jnp.asarray(self.size, pos.dtype)
        return jnp.any((pos < -r) | (pos > size + r), axis=1)

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

        sites = np.stack(np.meshgrid(*[axis(length) for length in self.size]), axis=-1)
        sites = sites.reshape(-1, self.dim)
        for shapes in self.obstacles:
            if len(sites):
                clearance = np.asarray(shapes.distance(jnp.asarray(sites))).min(axis=1)
                sites = sites[clearance > margin]
        return sites


def _face_normal(face: str, dim: int) -> np.ndarray:
    """Outward unit normal of a face of the room, such as ``"x-"``."""
    normal = np.zeros(dim)
    normal[_AXES.index(face[0])] = 1.0 if face[1] == "+" else -1.0
    return normal


def _door_centre(opening: pp.Opening, size: tuple[float, ...]) -> np.ndarray:
    """Centre of a door in a wall of the room ``[0, size]``."""
    dim = len(size)
    axis = _AXES.index(opening.face[0])
    spans = np.reshape(np.asarray(opening.span, dtype=float), (dim - 1, 2))
    centre = np.empty(dim)
    centre[axis] = size[axis] if opening.face[1] == "+" else 0.0
    centre[[a for a in range(dim) if a != axis]] = spans.mean(axis=1)
    return centre


def _ring(n: int) -> np.ndarray:
    """(n, 2) unit vectors at ``90 + 360 k / n`` degrees."""
    angles = math.pi / 2 + 2 * math.pi * np.arange(n) / n
    ring = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    ring[np.abs(ring) < 1e-12] = 0.0
    return ring


def _directions(dim: int, n: int) -> np.ndarray:
    """(n, dim) unit vectors of the discrete actions; see :class:`Evacuation`."""
    if dim == 2:
        return _ring(n)
    # Directions from the centre of a cube to its faces, edges and corners,
    # which have the components -1, 0 or 1 before normalisation.
    in_plane = np.sign(_ring(4 if n == 6 else 8))
    layers = [np.column_stack([in_plane, np.zeros(len(in_plane))])]
    tilted = np.sign(_ring(_TILTED_DIRECTIONS[n])) if _TILTED_DIRECTIONS[n] else np.zeros((0, 2))
    for z in (1.0, -1.0):
        layers.append(np.array([[0.0, 0.0, z]]))
        layers.append(np.column_stack([tilted, np.full(len(tilted), z)]))
    directions = np.concatenate(layers)
    return directions / np.linalg.norm(directions, axis=1, keepdims=True)
