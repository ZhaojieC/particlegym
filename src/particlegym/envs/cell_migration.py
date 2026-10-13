"""Cell migration: leader and follower cells move to a source of chemoattractant.

After the model of neural crest cell migration of Zhang, Chai, Sun and
Lykotrafitis (2020). Cells are soft discs in a closed square region. An
action is the direction of a cell's self-driven force. Leader cells sense the
gradient of a chemoattractant (Sdf1) that comes from a fixed source, and are
rewarded for reaching the source quickly. All cells secrete a second
chemoattractant (C3a), which diffuses; cells are rewarded for moving up its
gradient, which keeps the group together (co-attraction). Cells repel each
other on contact (contact inhibition of locomotion).

Taken from the paper: the region of 10 x 10, the cell diameter 0.5, the
contact force ``100 exp(overlap / 0.08)``, the viscous force ``-(m / tau) v``
with ``m = 1`` and ``tau = 0.5``, the control interval 0.1, the diffusion
coefficient 1 on a grid of spacing 0.1, the observation (position and
direction of the concentration gradient), the action (an angle) and the
rewards.

Different from the paper, and chosen here:

- The self-driven force has a constant magnitude (the paper: 0.4 times the
  magnitude of the concentration gradient), set by the speed of a free cell.
- Only the direction of the Sdf1 gradient is used: towards the source.
- C3a is secreted at a constant rate and decays (the paper holds the
  concentration at 1 where the cells are, without decay), so that the field
  becomes steady. The field that a cell senses includes what it secreted
  itself: in open space this gives no gradient at a cell at rest, but a
  moving cell senses its own trail, and a cell near a wall senses what the
  wall reflects, which draws it to the wall.
- The equations of motion are integrated with a time step of 0.002 (the
  paper: 0.1), which the stiff contact force needs.

This environment is experimental.

Y. Zhang, Z. Chai, Y. Sun and G. Lykotrafitis, A deep reinforcement learning
model based on deterministic policy gradient for collective neural crest cell
migration, arXiv:2007.03190 (2020).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from particlegym import physics as pp
from particlegym import viz
from particlegym.envs.herding import _unit
from particlegym.envs.spaces import Continuous

PURPLE = (150, 40, 160)


class CellMigrationParams(NamedTuple):
    """Episode parameters. A pytree, so values may be traced or batched.

    Attributes:
        speed: Speed of a free cell: the self-driven force times
            ``relaxation_time / mass``.
        relaxation_time: Relaxation time tau of the viscous force.
        secretion: Amount of C3a that a cell secretes per unit time.
        decay: Decay rate of C3a.
        leader_local_weight: Weight omega_l of the local reward in the reward
            of a leader; the step reward has the weight ``1 - omega_l``.
        step_reward: Reward of a leader per step before it reaches the source.
        target_radius: A leader has reached the source when it is closer to
            it than this distance, or touches a leader that has reached it.
        max_steps: Episode length in control steps.
    """

    speed: float = 0.3
    relaxation_time: float = 0.5
    secretion: float = 1.0
    decay: float = 0.25
    leader_local_weight: float = 0.3
    step_reward: float = -0.1
    target_radius: float = 0.5
    max_steps: int = 600


class CellMigrationState(NamedTuple):
    """Environment state. A pytree.

    Attributes:
        particles: The cells: leaders (kind 0) first, then followers (kind 1).
        c3a: (2, nx, ny) concentrations of the C3a secreted by the leaders
            and by the followers.
        arrived: (n_agents,) flags of the leaders that have reached the source.
        step: Number of control steps taken.
    """

    particles: pp.ParticleState
    c3a: jax.Array
    arrived: jax.Array
    step: jax.Array


class CellMigration:
    """Leader and follower cells that migrate to a source of chemoattractant.

    Every cell is an agent. Leaders sense the C3a of the other leaders,
    followers the C3a of all cells.

    Args:
        n_leaders: Number of leader cells.
        n_followers: Number of follower cells.
        source: The position (x, y) of a point source, or ``"top"`` (the
            source is the upper wall).
        start: ``"groups"`` (the leaders start in one part of the region and
            the followers in another, farther from the source) or
            ``"uniform"`` (all cells anywhere, at least 1 away from the
            source).
        action: ``"angle"`` (one number a in [-1, 1]; the direction angle is
            ``pi (a + 1)``) or ``"vector"`` (a 2D vector along the direction).
        size: Side length of the region.
        cell_radius: Radius of a cell.
        contact: Contact force between cells and with the walls.
        diffusivity: Diffusion coefficient of C3a.
        grid: Number of grid cells per side for the C3a field.
        dt: Physics time step, also used for the diffusion.
        substeps: Physics steps per control step.
    """

    def __init__(
        self,
        n_leaders: int = 9,
        n_followers: int = 0,
        source=(7.5, 7.5),
        start: str = "groups",
        action: str = "angle",
        size: float = 10.0,
        cell_radius: float = 0.25,
        contact: pp.PairPotential | None = None,
        diffusivity: float = 1.0,
        grid: int = 100,
        dt: float = 2e-3,
        substeps: int = 50,
    ) -> None:
        if n_leaders < 1 or n_followers < 0:
            raise ValueError("need at least one leader")
        if start not in ("groups", "uniform"):
            raise ValueError("start must be 'groups' or 'uniform'")
        if action not in ("angle", "vector"):
            raise ValueError("action must be 'angle' or 'vector'")
        self.n_leaders, self.n_followers = int(n_leaders), int(n_followers)
        self.n_agents = self.n_leaders + self.n_followers
        self.size, self.cell_radius = float(size), float(cell_radius)
        self.source = None if isinstance(source, str) else np.asarray(source, dtype=float)
        if isinstance(source, str) and source != "top":
            raise ValueError("source must be 'top' or a position (x, y)")
        self.start, self.action = start, action
        self.diffusivity = float(diffusivity)
        self.dt, self.substeps = float(dt), int(substeps)
        self.is_leader = np.arange(self.n_agents) < self.n_leaders
        contact = pp.ContactExponential(B=100.0, E=0.08) if contact is None else contact
        self.box = pp.Box((0.0, 0.0), (self.size, self.size))
        self.walls = pp.box_walls((0.0, 0.0), (self.size, self.size))
        self.force_field = pp.ForceField(
            self.box, pair=[contact], walls=[pp.Walls(self.walls, contact)]
        )
        self.grid = pp.Grid((0.0, 0.0), (self.size, self.size), (int(grid), int(grid)))
        if self.dt > self.grid.max_time_step(self.diffusivity):
            raise ValueError("dt is above the stability limit of the diffusion step")
        self._sites = self._start_sites()

    def _start_sites(self) -> tuple[np.ndarray, np.ndarray]:
        """Lattice sites on which the leaders and the followers may start."""
        spacing = 2.4 * self.cell_radius

        def lattice(lo, hi):
            x = np.arange(lo[0] * self.size, hi[0] * self.size + 1e-9, spacing)
            y = np.arange(lo[1] * self.size, hi[1] * self.size + 1e-9, spacing)
            return np.stack(np.meshgrid(x, y), axis=-1).reshape(-1, 2)

        if self.start == "groups" and self.source is None:
            sites = lattice((0.25, 0.46), (0.75, 0.6)), lattice((0.15, 0.08), (0.85, 0.38))
        elif self.start == "groups":
            sites = lattice((0.1, 0.25), (0.4, 0.5)), lattice((0.05, 0.04), (0.5, 0.18))
        else:
            margin = 2.0 * self.cell_radius / self.size
            everywhere = lattice((margin, margin), (1.0 - margin, 1.0 - margin))
            if self.source is None:
                far = everywhere[:, 1] < self.size - 1.0
            else:
                far = np.linalg.norm(everywhere - self.source, axis=1) > 1.0
            sites = everywhere[far], everywhere[far]
        if len(sites[0]) < self.n_leaders or len(sites[1]) < self.n_followers:
            raise ValueError("the region is too small for this number of cells")
        if self.start == "uniform" and len(sites[0]) < self.n_agents:
            raise ValueError("the region is too small for this number of cells")
        return sites

    @property
    def default_params(self) -> CellMigrationParams:
        """Default episode parameters."""
        return CellMigrationParams()

    @property
    def observation_space(self) -> Continuous:
        """Observation of one cell; see :meth:`step`."""
        return Continuous(-1.0, 1.0, (7,))

    @property
    def action_space(self) -> Continuous:
        """Action of one cell."""
        return Continuous(-1.0, 1.0, (1,) if self.action == "angle" else (2,))

    def reset(
        self, key: jax.Array, params: CellMigrationParams | None = None
    ) -> tuple[jax.Array, CellMigrationState]:
        """Start an episode; the C3a field of the resting cells is built up first."""
        params = self.default_params if params is None else params
        leader_sites, follower_sites = (jnp.asarray(s) for s in self._sites)

        def pick(k, sites, count):
            return sites[jax.random.choice(k, sites.shape[0], (count,), replace=False)]

        if self.start == "groups":
            leader_key, follower_key = jax.random.split(key)
            pos = jnp.concatenate(
                [
                    pick(leader_key, leader_sites, self.n_leaders),
                    pick(follower_key, follower_sites, self.n_followers),
                ]
            )
        else:
            pos = pick(key, leader_sites, self.n_agents)
        kind = jnp.asarray(~self.is_leader, jnp.int32)
        particles = pp.create_state(pos, radius=self.cell_radius, kind=kind)
        particles, _ = pp.init_forces(particles, self.force_field)

        # Let the field of the resting cells form for one decay time.
        source = self._secretion(particles.pos, params)
        steps = round(1.0 / (self.default_params.decay * self.dt))
        c3a, _ = jax.lax.scan(
            lambda c, _: (self._diffuse(c, source, params), None),
            jnp.zeros((2, *self.grid.shape), pos.dtype),
            None,
            length=steps,
        )
        state = CellMigrationState(
            particles, c3a, jnp.zeros(self.n_agents, bool), jnp.zeros((), jnp.int32)
        )
        return self._observe(state), state

    def step(
        self,
        key: jax.Array | None,
        state: CellMigrationState,
        action: jax.Array,
        params: CellMigrationParams | None = None,
    ):
        """Advance the environment by one control step.

        Returns:
            ``(obs, state, reward, terminated, truncated, info)``. An
            observation holds the position of the cell (scaled to -0.5 to
            0.5), the direction of the Sdf1 gradient, the direction of the
            gradient of the C3a that the cell senses, and 1 for a leader or 0
            for a follower. With the angle beta between the direction of the
            action and the C3a gradient, the local reward is
            ``exp(-beta^2) - 1``. A follower gets the local reward. A leader
            gets ``omega_l`` times the local reward plus ``1 - omega_l``
            times ``step_reward``, and 0 once it has reached the source: it
            is closer to it than ``target_radius``, or it touches a leader
            that has reached it. The episode terminates when all leaders
            have reached the source.
            ``info`` holds ``arrived``, the mean ``distance`` of the leaders
            to the source and the ``spread`` of all cells (the root mean
            square distance from their centre). The ``key`` is not used.
        """
        params = self.default_params if params is None else params
        dtype = state.particles.pos.dtype
        direction = self.direction(action, dtype)
        drift = params.speed * direction
        gamma = 1.0 / params.relaxation_time
        local = self._local_reward(direction, self.c3a_direction(state))

        def substep(carry, _):
            particles, c3a = carry
            particles, _ = pp.langevin_baoab(
                particles, self.force_field, self.dt, gamma=gamma, drift=drift
            )
            c3a = self._diffuse(c3a, self._secretion(particles.pos, params), params)
            return (particles, c3a), None

        (particles, c3a), _ = jax.lax.scan(
            substep, (state.particles, state.c3a), None, length=self.substeps
        )
        is_leader = jnp.asarray(self.is_leader)
        distance = self.source_distance(particles.pos)
        gap = jnp.linalg.norm(particles.pos[:, None, :] - particles.pos[None, :, :], axis=-1)
        touches_arrived = jnp.any(state.arrived[None, :] & (gap < 2.2 * self.cell_radius), axis=1)
        arrived = state.arrived | (
            is_leader & ((distance < params.target_radius) | touches_arrived)
        )
        step = state.step + 1
        state = CellMigrationState(particles, c3a, arrived, step)

        weight = params.leader_local_weight
        leader_reward = weight * local + (1.0 - weight) * params.step_reward
        reward = jnp.where(is_leader, jnp.where(arrived, 0.0, leader_reward), local)
        done = jnp.all(arrived | ~is_leader)
        centre = jnp.mean(particles.pos, axis=0)
        info = {
            "arrived": arrived,
            "distance": jnp.sum(jnp.where(is_leader, distance, 0.0)) / self.n_leaders,
            "spread": jnp.sqrt(jnp.mean(jnp.sum((particles.pos - centre) ** 2, axis=1))),
        }
        return (
            self._observe(state),
            state,
            reward,
            jnp.broadcast_to(done, (self.n_agents,)),
            jnp.broadcast_to(~done & (step >= params.max_steps), (self.n_agents,)),
            info,
        )

    def direction(self, action: jax.Array, dtype=None) -> jax.Array:
        """(n_agents, 2) unit vectors of the self-driven forces for the given actions."""
        if self.action == "angle":
            angle = jnp.pi * (jnp.reshape(jnp.asarray(action, dtype), (self.n_agents,)) + 1.0)
            return pp.active.heading_vector(angle)
        return _unit(jnp.reshape(jnp.asarray(action, dtype), (self.n_agents, 2)))

    def source_distance(self, pos: jax.Array) -> jax.Array:
        """(N,) distances of the points ``pos`` to the Sdf1 source."""
        if self.source is None:
            return self.size - pos[:, 1]
        return jnp.linalg.norm(pos - jnp.asarray(self.source, pos.dtype), axis=1)

    def sdf1_direction(self, pos: jax.Array) -> jax.Array:
        """(N, 2) directions of the Sdf1 gradient: towards the source."""
        if self.source is None:
            return jnp.broadcast_to(jnp.array([0.0, 1.0], pos.dtype), pos.shape)
        return _unit(jnp.asarray(self.source, pos.dtype) - pos)

    def c3a_gradient(self, state: CellMigrationState) -> jax.Array:
        """(n_agents, 2) gradients of the C3a that each cell senses, at its position.

        Leaders sense the C3a of the leaders, followers that of all cells.
        """
        pos = state.particles.pos
        of_leaders = self.grid.sample_gradient(state.c3a[0], pos)
        of_all = self.grid.sample_gradient(state.c3a[0] + state.c3a[1], pos)
        return jnp.where(jnp.asarray(self.is_leader)[:, None], of_leaders, of_all)

    def c3a_direction(self, state: CellMigrationState) -> jax.Array:
        """(n_agents, 2) unit vectors along :meth:`c3a_gradient`; zero where it vanishes.

        A gradient below ``1e-6`` counts as zero.
        """
        gradient = self.c3a_gradient(state)
        norm = jnp.linalg.norm(gradient, axis=1, keepdims=True)
        return jnp.where(norm > 1e-6, gradient / jnp.maximum(norm, 1e-6), 0.0)

    def gradient_policy(self, state: CellMigrationState, weight: float = 0.0) -> jax.Array:
        """Cells that move along the concentration gradients.

        A follower moves up the gradient of C3a. A leader moves along
        ``(1 - weight)`` times the direction to the source plus ``weight``
        times the direction of the C3a gradient; with ``weight = 0`` it takes
        the shortest path to the source.

        Returns:
            Actions in the form that the environment expects.
        """
        pos = state.particles.pos
        to_others = self.c3a_direction(state)
        leader = _unit((1.0 - weight) * self.sdf1_direction(pos) + weight * to_others)
        direction = jnp.where(jnp.asarray(self.is_leader)[:, None], leader, to_others)
        if self.action == "vector":
            return direction
        angle = jnp.mod(jnp.arctan2(direction[:, 1], direction[:, 0]), 2.0 * jnp.pi)
        return (angle / jnp.pi - 1.0)[:, None]

    def render(self, state: CellMigrationState, width: int = 360) -> np.ndarray:
        """Draw the cells over the C3a concentration.

        Leaders are purple, followers blue, the Sdf1 source red; the
        concentration of C3a is shown in orange.
        """
        size, radius = self.size, self.cell_radius
        margin = 0.03 * size
        canvas = viz.Canvas((-margin, -margin), (size + margin, size + 3 * radius), width)
        total = np.asarray(state.c3a[0] + state.c3a[1])
        canvas.field(total, (0.0, 0.0), (size, size), low=0.0, colors=(viz.WHITE, (255, 190, 110)))
        canvas.shapes(self.walls, viz.BLACK, 0.012 * size)
        if self.source is None:
            x = np.arange(radius, size, 2 * radius)
            canvas.discs(np.stack([x, np.full_like(x, size + radius)], axis=1), radius, viz.RED)
        else:
            canvas.discs([self.source], radius, viz.RED)
        colors = np.where(self.is_leader[:, None], PURPLE, viz.BLUE)
        canvas.discs(state.particles.pos, radius, colors)
        return canvas.frame()

    def _secretion(self, pos: jax.Array, params: CellMigrationParams) -> jax.Array:
        """(2, nx, ny) source terms of the C3a of the leaders and of the followers."""
        is_leader = jnp.asarray(self.is_leader)
        return params.secretion * jnp.stack(
            [self.grid.deposit(pos, 1.0, is_leader), self.grid.deposit(pos, 1.0, ~is_leader)]
        )

    def _diffuse(self, c3a: jax.Array, source: jax.Array, params: CellMigrationParams):
        return jax.vmap(
            lambda c, s: self.grid.diffuse(
                c, self.diffusivity, self.dt, decay=params.decay, source=s
            )
        )(c3a, source)

    def _local_reward(self, direction: jax.Array, to_others: jax.Array) -> jax.Array:
        """``exp(-beta^2) - 1`` for the angle beta between the two directions.

        Zero where the cell senses no gradient.
        """
        cosine = jnp.sum(direction * to_others, axis=1)
        beta = jnp.arccos(jnp.clip(cosine, -1.0, 1.0))
        senses = jnp.any(to_others != 0, axis=1)
        return jnp.where(senses, jnp.exp(-(beta**2)) - 1.0, 0.0)

    def _observe(self, state: CellMigrationState) -> jax.Array:
        pos = state.particles.pos
        return jnp.concatenate(
            [
                pos / self.size - 0.5,
                self.sdf1_direction(pos),
                self.c3a_direction(state),
                jnp.asarray(self.is_leader, pos.dtype)[:, None],
            ],
            axis=1,
        )
