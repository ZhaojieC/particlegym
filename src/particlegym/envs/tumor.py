# ParticleGym, https://github.com/ZhaojieC/particlegym (MIT license).
# Please cite it in published work: particlegym.cite() gives the references.
"""Tumour therapy: dose a drug so that a tumour stays under control for as long as possible.

An agent-based model of a tumour in two dimensions: every cell is a soft
disc that grows and divides into two daughters (Drasdo & Hoehme 2005). A cell
pressed by its neighbours stops cycling, so a dense tumour grows only at its
rim. Most cells are sensitive to the drug, a few are resistant; resistance
has a cost, a longer cell cycle. The drug kills sensitive cells while they
cycle.

The agent, the oncologist, chooses the dose every day and is rewarded for
every day before the tumour progresses, that is, grows to 1.2 times its
size at the start. Treating at the maximum tolerated dose kills the
sensitive cells, but frees the resistant ones from competition; adaptive
therapy (Gatenby et al. 2009) pauses treatment so that sensitive cells keep
the resistant ones in check, as in the adaptive abiraterone trial of Zhang
et al. (2017) and the agent-based model of Gallaher et al. (2018). This
environment is experimental.

Units: the diameter of a cell at division, and hours; a control step is one
day.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from particlegym import physics as pp
from particlegym import viz
from particlegym.citation import remind
from particlegym.envs.spaces import Continuous, Discrete

SENSITIVE, RESISTANT = 0, 1
_RADIUS = 0.5  # radius at division: the unit of length is the diameter
_NEWBORN = _RADIUS / math.sqrt(2.0)  # half the area
_LEVELS = np.array([0.0, 0.5, 1.0])  # doses of the discrete actions


class TumorTherapyParams(NamedTuple):
    """Episode parameters. A pytree, so values may be traced or batched.

    Attributes:
        resistant_fraction: Probability that a cell of the initial tumour is resistant.
        resistance_cost: Relative lengthening of the cell cycle of resistant cells.
        kill_rate: Rate at which a cycling sensitive cell dies at full dose (per hour).
        death_rate: Rate at which any cell dies of other causes (per hour).
        progression: The tumour progresses when it grows to this multiple
            of its number of cells at the start; the episode ends.
        toxicity: Reward deducted per day for a full dose.
        max_days: Episode length in days.
    """

    resistant_fraction: float = 0.02
    resistance_cost: float = 0.3
    kill_rate: float = 0.05
    death_rate: float = 0.0
    progression: float = 1.2
    toxicity: float = 0.0
    max_days: int = 1000


class TumorTherapyState(NamedTuple):
    """Environment state. A pytree.

    Attributes:
        cells: The cells; ``kind`` is 0 for sensitive and 1 for resistant
            cells, and slots without a cell are not alive.
        age: (capacity,) hours that each cell has spent cycling since its birth.
        cycle: (capacity,) length of the cell cycle of each cell (hours).
        initial: Number of cells at the start.
        dose: Dose of the last day.
        previous: Number of cells a day before.
        day: Number of days taken.
        progressed: Whether the tumour has progressed.
    """

    cells: pp.ParticleState
    age: jax.Array
    cycle: jax.Array
    initial: jax.Array
    dose: jax.Array
    previous: jax.Array
    day: jax.Array
    progressed: jax.Array


class TumorTherapy:
    """Daily dosing of a drug against a growing tumour of sensitive and resistant cells.

    The agent observes three numbers, as an oncologist observes a tumour
    marker: the number of cells relative to the start, its change over the
    last day, and the dose of the last day. It chooses the dose of the next
    day, between 0 and 1 (the maximum tolerated dose). The reward is 1 for
    every day before progression, minus ``toxicity`` times the dose.

    Every hour, a cell whose neighbours press it less than ``inhibition``
    (the sum of its overlaps, in cell diameters) advances in its cycle and
    grows; when its cycle is complete it divides. A cycling sensitive cell
    dies at the rate ``kill_rate`` times the dose. In between, the cells move
    by overdamped dynamics under a harmonic contact repulsion.

    Args:
        initial_cells: Number of cells at the start.
        capacity: Largest number of cells.
        action: ``"continuous"`` (a dose in [0, 1]) or ``"discrete"`` (0, 0.5 or 1).
        cycle_time: Mean cell cycle of sensitive cells (hours).
        cycle_spread: Relative standard deviation of the cell cycle.
        inhibition: Sum of overlaps above which a cell stops cycling.
        stiffness: Stiffness of the contact repulsion.
        friction: Friction coefficient of the overdamped motion.
        substeps: Steps of the mechanics per hour.
        size: Half the width of the square domain (cell diameters).
    """

    # Papers behind this environment, listed by particlegym.cite.
    references = ("drasdo2005", "gatenby2009", "zhang2017", "gallaher2018")

    def __init__(
        self,
        initial_cells: int = 400,
        capacity: int = 1024,
        action: str = "continuous",
        cycle_time: float = 24.0,
        cycle_spread: float = 0.15,
        inhibition: float = 0.3,
        stiffness: float = 10.0,
        friction: float = 2.0,
        substeps: int = 5,
        size: float = 40.0,
    ) -> None:
        if not 1 <= initial_cells < capacity:
            raise ValueError("need 1 <= initial_cells < capacity")
        if action not in ("continuous", "discrete"):
            raise ValueError("action must be 'continuous' or 'discrete'")
        if min(cycle_time, inhibition, stiffness, friction, size) <= 0 or substeps < 1:
            raise ValueError("cycle times, forces and the domain size must be positive")
        self.n_agents = 1
        self.initial_cells, self.capacity = int(initial_cells), int(capacity)
        self.action = action
        self.cycle_time, self.cycle_spread = float(cycle_time), float(cycle_spread)
        self.inhibition = float(inhibition)
        self.friction, self.substeps = float(friction), int(substeps)
        self.dt = 1.0 / self.substeps
        self.box = pp.Box((-size, -size), (size, size))
        self.neighbors = pp.CellList(cutoff=2 * _RADIUS, capacity=8)
        self.force_field = pp.ForceField(
            self.box, pair=[pp.Harmonic(k=float(stiffness))], neighbors=self.neighbors
        )
        self.levels = _LEVELS
        # A hexagonal patch: the initial tumour takes its innermost points.
        spacing = 0.85
        side = math.ceil(math.sqrt(initial_cells)) + 4
        i, j = np.meshgrid(np.arange(-side, side + 1), np.arange(-side, side + 1))
        points = np.stack([i + 0.5 * (j % 2), j * math.sqrt(3) / 2], -1).reshape(-1, 2)
        points = spacing * points[np.argsort(np.hypot(*points.T), kind="stable")]
        self._patch = points[:initial_cells]
        remind(self)

    @property
    def default_params(self) -> TumorTherapyParams:
        """Default episode parameters."""
        return TumorTherapyParams()

    @property
    def observation_space(self) -> Continuous:
        """Size of the tumour relative to the start, its change over a day, the last dose."""
        return Continuous(-jnp.inf, jnp.inf, (3,))

    @property
    def action_space(self) -> Discrete | Continuous:
        """The dose of the next day."""
        if self.action == "discrete":
            return Discrete(len(self.levels))
        return Continuous(0.0, 1.0, (1,))

    def reset(
        self, key: jax.Array, params: TumorTherapyParams | None = None
    ) -> tuple[jax.Array, TumorTherapyState]:
        """A round tumour of cells at random stages of their cycle, a few of them resistant."""
        params = self.default_params if params is None else params
        n, n0 = self.capacity, self.initial_cells
        jitter_key, kind_key, cycle_key, age_key = jax.random.split(key, 4)
        pos = jnp.zeros((n, 2)).at[:n0].set(jnp.asarray(self._patch))
        pos = pos + 0.05 * jax.random.normal(jitter_key, (n, 2))
        alive = jnp.arange(n) < n0
        kind = jnp.where(
            jax.random.uniform(kind_key, (n,)) < params.resistant_fraction, RESISTANT, SENSITIVE
        )
        cycle = self._cycles(cycle_key, kind, params)
        age = jax.random.uniform(age_key, (n,)) * cycle
        radius = self._radius(age, cycle)
        cells = pp.create_state(pos, mass=1.0, radius=radius, alive=alive, kind=kind)
        cells, _ = pp.init_forces(cells, self.force_field)
        count = jnp.asarray(n0, pos.dtype)
        zero = jnp.zeros((), pos.dtype)
        state = TumorTherapyState(
            cells, age, cycle, count, zero, count, jnp.zeros((), jnp.int32), jnp.zeros((), bool)
        )
        return self._observe(state), state

    def step(
        self,
        key: jax.Array,
        state: TumorTherapyState,
        action: jax.Array,
        params: TumorTherapyParams | None = None,
    ):
        """Advance the tumour by one day at the chosen dose.

        Returns:
            ``(obs, state, reward, terminated, truncated, info)``. ``info``
            holds the numbers of ``sensitive`` and ``resistant`` cells and
            ``overflow``, true if the neighbour search dropped candidates.
        """
        if key is None:
            raise ValueError("a random key is required for the cell cycles and deaths")
        params = self.default_params if params is None else params
        dose = self._dose(action, state.cells.pos.dtype)
        death = 1.0 - jnp.exp(-params.kill_rate * dose)  # probability within an hour
        turnover = 1.0 - jnp.exp(-params.death_rate)

        def hour(carry, k):
            cells, age, cycle, overflow, _ = carry
            death_key, angle_key, cycle_key, turnover_key = jax.random.split(k, 4)
            pressure = pp.overlap_sum(
                cells.pos, cells.radius, cells.alive, self.box, self.neighbors
            )
            cycling = cells.alive & (pressure < self.inhibition)
            age = age + cycling.astype(age.dtype)
            killed = cycling & (cells.kind == SENSITIVE)
            killed = killed & (jax.random.uniform(death_key, age.shape) < death)
            killed = killed | (jax.random.uniform(turnover_key, age.shape) < turnover)
            alive = cells.alive & ~killed
            divides, mother = pp.birth_slots(alive, cycling & (age >= cycle))
            angle = jax.random.uniform(angle_key, age.shape, maxval=2 * jnp.pi)
            direction = jnp.stack([jnp.cos(angle), jnp.sin(angle)], axis=1)
            pos, _, alive = pp.divide(cells.pos, cells.radius, alive, divides, mother, direction)
            born = mother >= 0
            kind = jnp.where(born, cells.kind[jnp.where(born, mother, 0)], cells.kind)
            renewed = divides | born
            cycle = jnp.where(renewed, self._cycles(cycle_key, kind, params), cycle)
            age = jnp.where(renewed, 0.0, age)
            radius = self._radius(age, cycle)
            cells = cells._replace(pos=pos, radius=radius, alive=alive, kind=kind)
            cells, info = pp.init_forces(cells, self.force_field)
            overflow = overflow | info.overflow

            def move(c, _):
                c, info = pp.brownian(c, self.force_field, self.dt, gamma=self.friction)
                return c, info.overflow

            cells, flags = jax.lax.scan(move, cells, None, length=self.substeps)
            return (cells, age, cycle, overflow | jnp.any(flags), jnp.sum(cycling)), None

        none = jnp.zeros((), jnp.int32)
        carry = (state.cells, state.age, state.cycle, jnp.zeros((), bool), none)
        (cells, age, cycle, overflow, cycling), _ = jax.lax.scan(
            hour, carry, jax.random.split(key, 24)
        )
        count = jnp.sum(cells.alive).astype(state.initial.dtype)
        day = state.day + 1
        progressed = state.progressed | (count > params.progression * state.initial)
        reward = jnp.where(state.progressed, 0.0, 1.0 - params.toxicity * dose)
        state = TumorTherapyState(
            cells, age, cycle, state.initial, dose, self._count(state), day, progressed
        )
        resistant = jnp.sum(cells.alive & (cells.kind == RESISTANT))
        info = {
            "sensitive": jnp.sum(cells.alive) - resistant,
            "resistant": resistant,
            "cycling": cycling,
            "overflow": overflow,
        }
        terminated = jnp.reshape(progressed, (1,))
        truncated = jnp.reshape(~progressed & (day >= params.max_days), (1,))
        return self._observe(state), state, jnp.reshape(reward, (1,)), terminated, truncated, info

    def mtd_policy(self, state: TumorTherapyState, params: TumorTherapyParams | None = None):
        """Treat every day at the maximum tolerated dose."""
        return self._to_action(jnp.ones(()))

    def adaptive_policy(
        self,
        state: TumorTherapyState,
        params: TumorTherapyParams | None = None,
        stop: float = 0.5,
        restart: float = 1.0,
    ):
        """Adaptive therapy: pause when the tumour has halved, resume at its initial size.

        This is the rule of the adaptive abiraterone trial (NCT02415621; Zhang
        et al. 2017) applied to the number of cells: treat at the maximum
        tolerated dose until the tumour has fallen to ``stop`` times its
        initial size, then pause until it has grown back to ``restart`` times
        it.
        """
        size = self._count(state) / state.initial
        treating = state.dose > 0
        dose = jnp.where(treating, size > stop, size >= restart)
        return self._to_action(dose.astype(state.cells.pos.dtype))

    def no_treatment_policy(
        self, state: TumorTherapyState, params: TumorTherapyParams | None = None
    ):
        """Never treat."""
        return self._to_action(jnp.zeros(()))

    def render(
        self, state: TumorTherapyState, width: int = 360, params: TumorTherapyParams | None = None
    ) -> np.ndarray:
        """Draw the cells: sensitive ones green, resistant ones red.

        A bar at the top shows the size of the tumour relative to the start,
        with marks at the start and at progression; it is purple on days of
        treatment.
        """
        params = self.default_params if params is None else params
        extent = 0.75 * math.sqrt(self.initial_cells * params.progression) + 2.0
        canvas = viz.Canvas((-extent, -extent), (extent, extent + 0.12 * extent), width)
        cells = state.cells
        alive = np.asarray(cells.alive)
        kind = np.asarray(cells.kind)
        colors = np.where((kind == RESISTANT)[:, None], [[220, 60, 40]], [[60, 170, 110]])
        canvas.discs(np.asarray(cells.pos), np.asarray(cells.radius), colors, mask=alive)
        top, scale = extent + 0.03 * extent, 2 * extent / (1.5 * params.progression)
        size = alive.sum() / float(state.initial)
        canvas.rectangles([-extent, top], [extent, top + 0.06 * extent], viz.LIGHT_GREY)
        bar = (150, 90, 200) if float(state.dose) > 0 else viz.GREY
        canvas.rectangles([-extent, top], [-extent + scale * size, top + 0.06 * extent], bar)
        for mark in (1.0, float(params.progression)):
            x = -extent + scale * mark
            lo = [x - 0.01 * extent, top - 0.02 * extent]
            canvas.rectangles(lo, [x + 0.01 * extent, top + 0.08 * extent], viz.BLACK)
        return canvas.frame()

    def _cycles(self, key, kind, params) -> jax.Array:
        """Cell cycles drawn for every slot; resistant cells take longer."""
        spread = jnp.clip(1.0 + self.cycle_spread * jax.random.normal(key, kind.shape), 0.5, 1.5)
        return self.cycle_time * (1.0 + params.resistance_cost * kind) * spread

    @staticmethod
    def _radius(age, cycle) -> jax.Array:
        """The area grows linearly over the cycle from half to the full size."""
        return _NEWBORN * jnp.sqrt(1.0 + jnp.clip(age / cycle, 0.0, 1.0))

    def _count(self, state: TumorTherapyState) -> jax.Array:
        return jnp.sum(state.cells.alive).astype(state.initial.dtype)

    def _dose(self, action, dtype) -> jax.Array:
        if self.action == "discrete":
            return jnp.asarray(self.levels, dtype)[jnp.reshape(action, ())]
        return jnp.clip(jnp.reshape(jnp.asarray(action, dtype), ()), 0.0, 1.0)

    def _to_action(self, dose) -> jax.Array:
        if self.action == "discrete":
            return jnp.reshape(jnp.argmin(jnp.abs(jnp.asarray(self.levels) - dose)), (1,))
        return jnp.reshape(dose, (1, 1))

    def _observe(self, state: TumorTherapyState) -> jax.Array:
        size = self._count(state) / state.initial
        change = (self._count(state) - state.previous) / state.initial
        return jnp.stack([size, change, state.dose])[None, :]
