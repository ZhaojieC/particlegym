"""Herding: shepherds drive a flock to a target.

The flock follows the rules of the model of Strömbom et al. (2014): a sheep
that sees a shepherd is repelled by it, repelled by sheep that are very close
and attracted to the centre of its nearest neighbours; a sheep that sees no
shepherd grazes. The agents are the shepherds. An action sets a shepherd's
desired direction of motion.

The rules give every sheep a desired velocity; the motion itself follows from
the physics core, in which sheep and shepherds are soft discs whose velocity
relaxes to the desired one. This differs from the paper, where positions are
updated in discrete steps without inertia or contacts, so results of the
paper are not expected to be reproduced in detail. The formulas of the paper
were not available in text form when this was written; the rules implemented
here are stated in the documentation. This environment is experimental.

D. Strömbom, R. P. Mann, A. M. Wilson, S. Hailes, A. J. Morton, D. J. T.
Sumpter and A. J. King, Solving the shepherding problem: heuristics for
herding autonomous, interacting agents, J. R. Soc. Interface 11, 20140719
(2014).
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


class HerdingParams(NamedTuple):
    """Episode parameters. A pytree, so values may be traced or batched.

    The defaults of the flock are the typical values in Table 1 of Strömbom
    et al. (2014), with one time step of the paper taken as one second.

    Attributes:
        sheep_speed: Speed of a sheep that flees from a shepherd (m/s).
        shepherd_speed: Largest speed of a shepherd (m/s).
        detection_distance: Distance r_s within which a sheep sees a shepherd (m).
        interaction_distance: Distance r_a within which sheep repel each other (m).
        repulsion: Weight rho_a of the repulsion from close sheep.
        attraction: Weight c of the attraction to the nearest neighbours.
        shepherd_repulsion: Weight rho_s of the repulsion from the shepherds.
        inertia: Weight h of the previous direction.
        noise: Weight e of a random direction.
        graze_probability: Probability that a grazing sheep moves in a control step.
        relaxation_time: Time in which velocities relax to the desired ones (s).
        target_radius: The task is solved when the centre of the flock is
            within this distance of the target (m).
        success_reward: Reward for solving the task.
        max_steps: Episode length in control steps.
    """

    sheep_speed: float = 1.0
    shepherd_speed: float = 1.5
    detection_distance: float = 65.0
    interaction_distance: float = 2.0
    repulsion: float = 2.0
    attraction: float = 1.05
    shepherd_repulsion: float = 1.0
    inertia: float = 0.5
    noise: float = 0.3
    graze_probability: float = 0.05
    relaxation_time: float = 0.25
    target_radius: float = 10.0
    success_reward: float = 10.0
    max_steps: int = 2000


class HerdingState(NamedTuple):
    """Environment state. A pytree.

    Attributes:
        particles: Shepherds (the first ``n_agents`` particles, kind 0) and
            sheep (kind 1).
        heading: (n_sheep, 2) unit vectors, the directions in which the sheep
            wanted to move in the previous control step.
        step: Number of control steps taken.
    """

    particles: pp.ParticleState
    heading: jax.Array
    step: jax.Array


class Herding:
    """Shepherds that drive a flock of sheep to a target.

    The sheep start at random places in the upper right quarter of a square
    field of side ``field``, the shepherds near the target. There are no
    fences.

    Args:
        n_sheep: Number of sheep.
        n_agents: Number of shepherds.
        n_neighbors: Number n of nearest neighbours to whose centre a sheep
            is attracted.
        field: Side length L of the field (m).
        target: The point to which the flock is to be driven.
        action: ``"continuous"`` (a 2D vector; its length, capped at 1, scales
            the shepherd speed) or ``"discrete"`` (one of ``n_directions``
            directions at full speed; direction k is at ``90 + 360 k / n``
            degrees, so 0 is +y).
        n_directions: Number of discrete directions.
        radius: Body radius of sheep and shepherds (m).
        stiffness: Stiffness of the contact between bodies.
        dt: Physics time step (s).
        substeps: Physics steps per control step; the control interval
            ``dt * substeps`` should be 1 s for the default parameters.
    """

    def __init__(
        self,
        n_sheep: int = 40,
        n_agents: int = 1,
        n_neighbors: int = 30,
        field: float = 150.0,
        target=(0.0, 0.0),
        action: str = "continuous",
        n_directions: int = 8,
        radius: float = 0.5,
        stiffness: float = 100.0,
        dt: float = 0.05,
        substeps: int = 20,
    ) -> None:
        if n_sheep < 2 or n_agents < 1:
            raise ValueError("need at least two sheep and one shepherd")
        if not 1 <= n_neighbors < n_sheep:
            raise ValueError("n_neighbors must be between 1 and n_sheep - 1")
        if action not in ("discrete", "continuous"):
            raise ValueError("action must be 'discrete' or 'continuous'")
        self.n_sheep, self.n_agents = int(n_sheep), int(n_agents)
        self.n = self.n_sheep + self.n_agents
        self.n_neighbors = int(n_neighbors)
        self.field = float(field)
        self.target = np.asarray(target, dtype=float)
        self.action = action
        self.radius = float(radius)
        self.dt, self.substeps = float(dt), int(substeps)
        self.directions = _ring(n_directions)
        span = 20.0 * self.field
        self.box = pp.Box((-span, -span), (span, span))
        self.force_field = pp.ForceField(self.box, pair=[pp.Harmonic(k=float(stiffness))])
        # Start places of the sheep: a lattice in the upper right quarter.
        spacing = 4.0 * self.radius
        axis = np.arange(0.5 * self.field + spacing, self.field, spacing)
        self._sites = np.stack(np.meshgrid(axis, axis), axis=-1).reshape(-1, 2)
        if len(self._sites) < self.n_sheep:
            raise ValueError("the field is too small for the flock")

    @property
    def default_params(self) -> HerdingParams:
        """Default episode parameters."""
        return HerdingParams()

    @property
    def observation_space(self) -> Continuous:
        """Observation of one shepherd; see :meth:`step`."""
        return Continuous(-jnp.inf, jnp.inf, (11 + (2 if self.n_agents > 1 else 0),))

    @property
    def action_space(self) -> Discrete | Continuous:
        """Action of one shepherd."""
        if self.action == "discrete":
            return Discrete(len(self.directions))
        return Continuous(-1.0, 1.0, (2,))

    def reset(
        self, key: jax.Array, params: HerdingParams | None = None
    ) -> tuple[jax.Array, HerdingState]:
        """Start an episode with a scattered flock."""
        site_key, shepherd_key, heading_key = jax.random.split(key, 3)
        sites = jnp.asarray(self._sites)
        chosen = jax.random.choice(site_key, sites.shape[0], (self.n_sheep,), replace=False)
        offset = jax.random.uniform(
            shepherd_key, (self.n_agents, 2), sites.dtype, 0.05 * self.field, 0.15 * self.field
        )
        shepherds = jnp.asarray(self.target, sites.dtype) + offset
        kind = jnp.concatenate(
            [jnp.zeros(self.n_agents, jnp.int32), jnp.ones(self.n_sheep, jnp.int32)]
        )
        particles = pp.create_state(
            jnp.concatenate([shepherds, sites[chosen]]), radius=self.radius, kind=kind
        )
        particles, _ = pp.init_forces(particles, self.force_field)
        angle = jax.random.uniform(heading_key, (self.n_sheep,), sites.dtype, 0.0, 2 * jnp.pi)
        state = HerdingState(particles, pp.active.heading_vector(angle), jnp.zeros((), jnp.int32))
        return self._observe(state), state

    def step(
        self,
        key: jax.Array,
        state: HerdingState,
        action: jax.Array,
        params: HerdingParams | None = None,
    ):
        """Advance the environment by one control step.

        Returns:
            ``(obs, state, reward, terminated, truncated, info)``. An
            observation holds, with lengths in units of the field size: the
            shepherd's position relative to the target, its velocity (in
            units of the shepherd speed), the centre of the flock relative to
            the shepherd, the target relative to the centre of the flock, the
            sheep farthest from the centre relative to the shepherd, the
            largest distance of a sheep from the centre and, with several
            shepherds, the nearest other shepherd. All shepherds get the same
            reward: the distance (m) by which the centre of the flock has
            approached the target, plus ``success_reward`` when the task is
            solved, which terminates the episode. ``info`` holds the
            ``distance`` of the centre of the flock to the target and the
            ``spread`` of the flock (largest distance from its centre).
        """
        params = self.default_params if params is None else params
        particles = state.particles
        dtype = particles.pos.dtype
        heading, sheep_velocity = self.sheep_velocity(key, state, params)
        drift = jnp.concatenate([self.shepherd_velocity(action, params, dtype), sheep_velocity])
        gamma = 1.0 / params.relaxation_time

        def substep(p, _):
            p, _ = pp.langevin_baoab(p, self.force_field, self.dt, gamma=gamma, drift=drift)
            return p, None

        before = self._flock(particles)[1]
        particles, _ = jax.lax.scan(substep, particles, None, length=self.substeps)
        step = state.step + 1
        state = HerdingState(particles, heading, step)
        _, distance, spread, _ = self._flock(particles)
        solved = distance < params.target_radius
        reward = before - distance + jnp.where(solved, params.success_reward, 0.0)
        info = {"distance": distance, "spread": spread}
        return (
            self._observe(state),
            state,
            jnp.broadcast_to(reward, (self.n_agents,)),
            jnp.broadcast_to(solved, (self.n_agents,)),
            jnp.broadcast_to(~solved & (step >= params.max_steps), (self.n_agents,)),
            info,
        )

    def shepherd_velocity(self, action: jax.Array, params: HerdingParams, dtype) -> jax.Array:
        """Desired velocities (n_agents, 2) of the shepherds for the given actions."""
        if self.action == "discrete":
            index = jnp.reshape(action, (self.n_agents,))
            direction = jnp.asarray(self.directions, dtype)[index]
        else:
            direction = jnp.reshape(jnp.asarray(action, dtype), (self.n_agents, 2))
            length = jnp.linalg.norm(direction, axis=1, keepdims=True)
            direction = direction / jnp.maximum(length, 1.0)
        return params.shepherd_speed * direction

    def sheep_velocity(self, key: jax.Array, state: HerdingState, params: HerdingParams):
        """The flock rules: new headings and desired velocities of the sheep.

        A sheep within ``detection_distance`` of a shepherd wants to move at
        ``sheep_speed`` along the normalised sum of

        - ``inertia`` times its previous direction,
        - ``attraction`` times the direction to the centre of its
          ``n_neighbors`` nearest neighbours,
        - ``repulsion`` times the direction of the summed unit vectors that
          point away from the sheep within ``interaction_distance``,
        - ``shepherd_repulsion`` times the direction of the summed unit
          vectors that point away from the shepherds it sees, and
        - ``noise`` times a random direction.

        Other sheep graze: they stand still, and with ``graze_probability``
        they walk in a random direction for one control step.

        Returns:
            The new headings and the desired velocities, both (n_sheep, 2).
        """
        m = self.n_agents
        pos = state.particles.pos
        sheep, shepherds = pos[m:], pos[:m]
        noise_key, graze_key, direction_key = jax.random.split(key, 3)

        offset = sheep[:, None, :] - sheep[None, :, :]  # from sheep j to sheep i
        squared = jnp.sum(offset * offset, axis=-1)
        other = ~jnp.eye(self.n_sheep, dtype=bool)
        close = other & (squared < params.interaction_distance**2)
        away = jnp.sum(jnp.where(close[:, :, None], _unit(offset), 0.0), axis=1)

        _, nearest = jax.lax.top_k(-jnp.where(other, squared, jnp.inf), self.n_neighbors)
        centre = jnp.mean(sheep[nearest], axis=1)

        from_shepherd = sheep[:, None, :] - shepherds[None, :, :]
        seen = jnp.sum(from_shepherd**2, axis=-1) < params.detection_distance**2
        flee = jnp.sum(jnp.where(seen[:, :, None], _unit(from_shepherd), 0.0), axis=1)

        angle = jax.random.uniform(noise_key, (self.n_sheep,), pos.dtype, 0.0, 2 * jnp.pi)
        heading = _unit(
            params.inertia * state.heading
            + params.attraction * _unit(centre - sheep)
            + params.repulsion * _unit(away)
            + params.shepherd_repulsion * _unit(flee)
            + params.noise * pp.active.heading_vector(angle)
        )
        alarmed = jnp.any(seen, axis=1)[:, None]
        heading = jnp.where(alarmed, heading, state.heading)

        moves = jax.random.uniform(graze_key, (self.n_sheep, 1)) < params.graze_probability
        angle = jax.random.uniform(direction_key, (self.n_sheep,), pos.dtype, 0.0, 2 * jnp.pi)
        grazing = jnp.where(moves, pp.active.heading_vector(angle), 0.0)
        return heading, params.sheep_speed * jnp.where(alarmed, heading, grazing)

    def shepherd_policy(
        self, key: jax.Array, state: HerdingState, params: HerdingParams | None = None
    ) -> jax.Array:
        """A scripted shepherd that collects stray sheep and drives the flock.

        Adapted from the heuristic of Strömbom et al. (2014). With N sheep
        and the interaction distance r_a: if all sheep are within
        ``r_a N^(2/3)`` of the centre of the flock, the shepherd walks to the
        driving point, ``r_a N^(1/2)`` behind the centre as seen from the
        target; otherwise it walks to the collecting point, ``r_a`` behind
        the sheep farthest from the centre. Within ``3 r_a`` of a sheep it
        walks at a fifth of its speed. Its direction has the same random
        part as that of the sheep. With several shepherds, shepherd k
        collects the k-th farthest sheep.

        Returns:
            Continuous actions, shape (n_agents, 2).
        """
        params = self.default_params if params is None else params
        m = self.n_agents
        pos = state.particles.pos
        sheep, shepherds = pos[m:], pos[:m]
        r_a = params.interaction_distance
        centre, _, spread, order = self._flock(state.particles)
        target = jnp.asarray(self.target, pos.dtype)
        driving = centre + r_a * self.n_sheep**0.5 * _unit(centre - target)
        stray = sheep[order[:m]]
        collecting = stray + r_a * _unit(stray - centre)
        goal = jnp.where(spread < r_a * self.n_sheep ** (2 / 3), driving[None, :], collecting)

        angle = jax.random.uniform(key, (m,), pos.dtype, 0.0, 2 * jnp.pi)
        direction = _unit(_unit(goal - shepherds) + params.noise * pp.active.heading_vector(angle))
        gap = jnp.min(jnp.linalg.norm(shepherds[:, None, :] - sheep[None, :, :], axis=-1), axis=1)
        return jnp.where((gap < 3.0 * r_a)[:, None], 0.2, 1.0) * direction

    def render(self, state: HerdingState, width: int = 400) -> np.ndarray:
        """Draw the field: sheep blue, shepherds red, the target area green.

        The view is larger than the field, because a flock that flees from a
        shepherd coming from the target leaves the field at first.
        """
        view = 1.4 * self.field
        lo = np.minimum(self.target, 0.0) - 0.1 * self.field
        canvas = viz.Canvas(lo, lo + view, width, background=(235, 242, 225))
        canvas.discs([self.target], self.default_params.target_radius, (150, 205, 140))
        p = state.particles
        # Bodies of 0.5 m are too small to see on a field of 150 m.
        shown = max(self.radius, 0.007 * view)
        is_shepherd = np.asarray(p.kind) == 0
        colors = np.where(is_shepherd[:, None], viz.RED, viz.BLUE)
        canvas.discs(p.pos, np.where(is_shepherd, 1.5 * shown, shown), colors)
        return canvas.frame()

    def _flock(self, particles: pp.ParticleState):
        """Centre of the flock, its distance to the target, its spread, sheep by distance."""
        sheep = particles.pos[self.n_agents :]
        centre = jnp.mean(sheep, axis=0)
        distance = jnp.linalg.norm(centre - jnp.asarray(self.target, sheep.dtype))
        from_centre = jnp.linalg.norm(sheep - centre, axis=1)
        order = jnp.argsort(-from_centre)
        return centre, distance, from_centre[order[0]], order

    def _observe(self, state: HerdingState) -> jax.Array:
        p = state.particles
        m, size = self.n_agents, self.field
        shepherds, sheep = p.pos[:m], p.pos[m:]
        centre, _, spread, order = self._flock(p)
        target = jnp.asarray(self.target, p.pos.dtype)
        parts = [
            (shepherds - target) / size,
            p.vel[:m] / self.default_params.shepherd_speed,
            (centre - shepherds) / size,
            jnp.broadcast_to((target - centre) / size, (m, 2)),
            (sheep[order[0]] - shepherds) / size,
            jnp.broadcast_to(spread / size, (m, 1)),
        ]
        if m > 1:
            offset = shepherds[None, :, :] - shepherds[:, None, :]
            squared = jnp.sum(offset * offset, axis=-1) + jnp.where(
                jnp.eye(m, dtype=bool), jnp.inf, 0
            )
            nearest = jnp.argmin(squared, axis=1)
            parts.append(offset[jnp.arange(m), nearest] / size)
        return jnp.concatenate(parts, axis=1)


def _unit(x: jax.Array) -> jax.Array:
    """Vectors scaled to unit length; zero vectors stay zero."""
    squared = jnp.sum(x * x, axis=-1, keepdims=True)
    safe = jnp.where(squared > 0, squared, 1.0)
    return jnp.where(squared > 0, x / jnp.sqrt(safe), 0.0)
