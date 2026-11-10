# ParticleGym, https://github.com/ZhaojieC/particlegym (MIT license).
# Please cite it in published work: particlegym.cite() gives the references.
"""Orbit transfer: raise a spacecraft to a higher circular orbit on little fuel.

A spacecraft circles a planet on a circular orbit of radius r0. With a
thruster of limited acceleration it must reach the circular orbit of radius
r1 and spend as little velocity change (delta-v, a measure of the fuel) as
possible. For impulsive burns between circular orbits with r1 / r0 < 11.94
the optimum is the Hohmann transfer: a prograde burn onto the ellipse that
touches both orbits, half a revolution, and a second prograde burn at the
far end of the ellipse. Its velocity change is
:func:`~particlegym.physics.hohmann_delta_v` (0.2845 for r1 = 2 r0); a
finite thrust needs slightly more. :meth:`OrbitTransfer.hohmann_policy`
flies this transfer and is the baseline.

Units: G M = 1 and r0 = 1, so the speed on the first orbit is 1 and its
period is 2 pi. The planet and the spacecraft are particles under
:class:`~particlegym.physics.NewtonGravity`; the spacecraft are test
particles, so the planet stays at rest and several spacecraft are independent
copies of the task.

This environment is experimental: the dynamics are tested against Kepler
orbits and the Hohmann transfer (see the validation page), but no
reinforcement-learning result on it has been published.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from particlegym import physics as pp
from particlegym import viz
from particlegym.citation import remind
from particlegym.envs.spaces import Continuous, Discrete

_GM = 1.0
_R0 = 1.0
# Thrusts of the discrete actions in the (radial, transverse) frame, as
# fractions of the largest thrust: coast; prograde, retrograde, radially out
# and radially in at full thrust; the same four at a tenth of it.
_FULL = np.array([[0.0, 1.0], [0.0, -1.0], [1.0, 0.0], [-1.0, 0.0]])
_DIRECTIONS = np.concatenate([np.zeros((1, 2)), _FULL, 0.1 * _FULL])


class OrbitTransferParams(NamedTuple):
    """Episode parameters. A pytree, so values may be traced or batched.

    Attributes:
        target_radius: Radius r1 of the target orbit (the first orbit has r0 = 1).
        max_thrust: Largest acceleration of the thruster.
        fuel: Velocity change available for the whole episode.
        radius_tolerance: The orbit counts as reached when its semi-major
            axis is within this fraction of r1 ...
        eccentricity_tolerance: ... and its eccentricity below this value.
        arrival_reward: Reward for reaching the target orbit.
        max_steps: Episode length in control steps.
    """

    target_radius: float = 2.0
    max_thrust: float = 0.5
    fuel: float = 1.0
    radius_tolerance: float = 0.02
    eccentricity_tolerance: float = 0.02
    arrival_reward: float = 1.0
    max_steps: int = 400


class OrbitTransferState(NamedTuple):
    """Environment state. A pytree.

    Attributes:
        particles: The planet (first) and the spacecraft.
        fuel_used: (n_agents,) velocity change spent so far.
        done: (n_agents,) whether a spacecraft has reached the target orbit,
            crashed or escaped.
        step: Number of control steps taken.
    """

    particles: pp.ParticleState
    fuel_used: jax.Array
    done: jax.Array
    step: jax.Array


class OrbitTransfer:
    """Spacecraft that transfer from a circular orbit of radius 1 to radius r1.

    Each spacecraft observes, in its own orbital frame, its distance from the
    planet, its radial and transverse velocity, the target radius and the
    fuel it has left: five numbers that do not depend on where on the orbit
    it is. An action is a thrust in the same frame. The reward is minus the
    velocity change spent in the step, plus ``arrival_reward`` when the
    spacecraft reaches the target orbit; the episode of a spacecraft
    terminates then, and when it hits the planet or escapes.

    Args:
        n_agents: Number of spacecraft; they do not interact.
        action: ``"continuous"`` (a vector (radial, transverse) whose length,
            capped at 1, scales the largest thrust) or ``"discrete"`` (one of
            nine: 0 coast; 1 prograde, 2 retrograde, 3 radially out and
            4 radially in at full thrust; 5 to 8 the same at a tenth of it).
        planet_radius: A spacecraft closer to the centre than this crashes.
        escape_radius: A spacecraft farther than this has escaped.
        dt: Physics time step.
        substeps: Physics steps per control step.
    """

    # Papers behind this environment, listed by particlegym.cite.
    references = ()

    def __init__(
        self,
        n_agents: int = 1,
        action: str = "continuous",
        planet_radius: float = 0.3,
        escape_radius: float = 10.0,
        dt: float = 0.01,
        substeps: int = 10,
    ) -> None:
        if n_agents < 1:
            raise ValueError("n_agents must be at least 1")
        if action not in ("continuous", "discrete"):
            raise ValueError("action must be 'continuous' or 'discrete'")
        if not 0 < planet_radius < _R0 < escape_radius:
            raise ValueError("need 0 < planet_radius < 1 < escape_radius")
        if dt <= 0 or substeps < 1:
            raise ValueError("dt must be positive and substeps at least 1")
        self.n_agents = int(n_agents)
        self.action = action
        self.planet_radius, self.escape_radius = float(planet_radius), float(escape_radius)
        self.dt, self.substeps = float(dt), int(substeps)
        self.control_interval = self.dt * self.substeps
        self.directions = _DIRECTIONS
        box = pp.Box((-2 * self.escape_radius,) * 2, (2 * self.escape_radius,) * 2)
        self.force_field = pp.ForceField(box, external=[pp.NewtonGravity(G=_GM, sources=1)])
        remind(self)

    @property
    def default_params(self) -> OrbitTransferParams:
        """Default episode parameters."""
        return OrbitTransferParams()

    @property
    def observation_space(self) -> Continuous:
        """Observation of one spacecraft: r, radial and transverse velocity, r1, fuel left."""
        return Continuous(-jnp.inf, jnp.inf, (5,))

    @property
    def action_space(self) -> Discrete | Continuous:
        """Action of one spacecraft."""
        if self.action == "discrete":
            return Discrete(len(self.directions))
        return Continuous(-1.0, 1.0, (2,))

    def reset(
        self, key: jax.Array, params: OrbitTransferParams | None = None
    ) -> tuple[jax.Array, OrbitTransferState]:
        """Start every spacecraft on the circular orbit of radius 1, at a random angle."""
        params = self.default_params if params is None else params
        angle = jax.random.uniform(key, (self.n_agents,), minval=0.0, maxval=2 * jnp.pi)
        radial = jnp.stack([jnp.cos(angle), jnp.sin(angle)], axis=1)
        transverse = jnp.stack([-radial[:, 1], radial[:, 0]], axis=1)
        speed = jnp.sqrt(_GM / _R0)
        pos = jnp.concatenate([jnp.zeros((1, 2), radial.dtype), _R0 * radial])
        vel = jnp.concatenate([jnp.zeros((1, 2), radial.dtype), speed * transverse])
        radius = jnp.concatenate([jnp.full(1, self.planet_radius), jnp.zeros(self.n_agents)])
        particles = pp.create_state(pos, vel, mass=1.0, radius=radius)
        particles, _ = pp.init_forces(particles, self.force_field)
        state = OrbitTransferState(
            particles,
            jnp.zeros(self.n_agents, pos.dtype),
            jnp.zeros(self.n_agents, bool),
            jnp.zeros((), jnp.int32),
        )
        return self._observe(state, params), state

    def step(
        self,
        key: jax.Array,
        state: OrbitTransferState,
        action: jax.Array,
        params: OrbitTransferParams | None = None,
    ):
        """Advance the environment by one control step.

        The thrust is held fixed in the inertial frame during the step and is
        applied as half kicks around each velocity Verlet step. A spacecraft
        without fuel left, or whose episode has ended, does not thrust.

        Returns:
            ``(obs, state, reward, terminated, truncated, info)``. ``info``
            holds the ``delta_v`` spent in the step, whether each spacecraft
            ``arrived``, ``crashed`` or ``escaped`` in it, and the
            ``semi_major_axis`` and ``eccentricity`` of the orbits.
        """
        params = self.default_params if params is None else params
        particles = state.particles
        dtype = particles.pos.dtype
        radial, transverse = self._frame(particles)
        local = self._thrust_command(action, dtype)  # (n, 2), length <= 1
        fuel_left = jnp.maximum(params.fuel - state.fuel_used, 0.0)
        wanted = params.max_thrust * jnp.linalg.norm(local, axis=1) * self.control_interval
        delta_v = jnp.where(state.done, 0.0, jnp.minimum(wanted, fuel_left))
        scale = jnp.where(wanted > 0.0, delta_v / jnp.where(wanted > 0.0, wanted, 1.0), 0.0)
        accel = (
            params.max_thrust
            * scale[:, None]
            * (local[:, :1] * radial + local[:, 1:] * transverse)
        )
        accel = jnp.concatenate([jnp.zeros((1, 2), dtype), accel])
        half = 0.5 * self.dt * accel

        def substep(p, _):
            p = p._replace(vel=p.vel + half)
            p, _ = pp.velocity_verlet(p, self.force_field, self.dt)
            return p._replace(vel=p.vel + half), None

        particles, _ = jax.lax.scan(substep, particles, None, length=self.substeps)
        fuel_used = state.fuel_used + delta_v

        position, velocity = self._relative(particles)
        r = jnp.linalg.norm(position, axis=1)
        a, e = pp.orbital_elements(position, velocity, _GM)
        ecc = jnp.linalg.norm(e, axis=1)
        target = params.target_radius
        arrived = (
            (jnp.abs(a - target) < params.radius_tolerance * target)
            & (ecc < params.eccentricity_tolerance)
            & ~state.done
        )
        crashed = (r < self.planet_radius) & ~state.done
        escaped = (r > self.escape_radius) & ~state.done
        done = state.done | arrived | crashed | escaped
        reward = jnp.where(state.done, 0.0, -delta_v) + jnp.where(
            arrived, params.arrival_reward, 0.0
        )
        step = state.step + 1
        state = OrbitTransferState(particles, fuel_used, done, step)
        info = {
            "delta_v": delta_v,
            "arrived": arrived,
            "crashed": crashed,
            "escaped": escaped,
            "semi_major_axis": a,
            "eccentricity": ecc,
        }
        truncated = ~done & (step >= params.max_steps)
        return self._observe(state, params), state, reward, done, truncated, info

    def hohmann_policy(self, state: OrbitTransferState, params: OrbitTransferParams | None = None):
        """The Hohmann transfer as a closed-loop controller.

        First it thrusts prograde (transverse) until the far end of the orbit,
        the apoapsis, is at r1: the transverse speed it aims for follows from
        the energy and the angular momentum, which a spacecraft has at the
        apoapsis of the new orbit as well. Then it coasts, and starts the
        second burn half the duration of that burn before the apoapsis, so
        that the burn is centred on it; the time to the apoapsis follows from
        Kepler's equation. The second burn makes the orbit circular: it
        cancels the radial velocity and reaches the circular speed. In each
        step the controller asks for exactly the velocity change it needs, at
        most the largest thrust. With discrete actions it thrusts prograde
        only: at full thrust while it needs at least that much, then at a
        tenth of it.
        """
        params = self.default_params if params is None else params
        position, velocity = self._relative(state.particles)
        radial, transverse = self._frame(state.particles)
        r = jnp.linalg.norm(position, axis=1)
        v_r = jnp.sum(velocity * radial, axis=1)
        v_t = jnp.sum(velocity * transverse, axis=1)
        a, e = pp.orbital_elements(position, velocity, _GM)
        ecc = jnp.linalg.norm(e, axis=1)
        apoapsis, periapsis = a * (1.0 + ecc), a * (1.0 - ecc)
        target = params.target_radius
        full = params.max_thrust * self.control_interval  # velocity change of a full step

        # Transverse speed that puts the apoapsis at r1: at the apoapsis the
        # speed is h / r1 with the angular momentum h = r v_t, and the energy
        # (v_r^2 + v_t^2) / 2 - GM / r is the same as there.
        ratio = jnp.minimum(r / target, 0.999)
        apoapsis_speed2 = (2.0 * _GM * (1.0 / r - 1.0 / target) - v_r**2) / (1.0 - ratio**2)
        middle = 0.5 * (_R0 + target)
        raise_apoapsis = (periapsis < middle) & (
            apoapsis < target * (1.0 - 0.1 * params.radius_tolerance)
        )

        # Time to the apoapsis: the eccentric anomaly E has cos E = (1 - r / a) / e
        # and E < pi on the way out; Kepler's equation gives the mean anomaly
        # E - e sin E, which grows at the rate sqrt(GM / a^3) and is pi there.
        bound = a > 0.0
        a_bound = jnp.where(bound, a, 1.0)
        cos_eccentric = jnp.clip((1.0 - r / a_bound) / jnp.maximum(ecc, 1e-12), -1.0, 1.0)
        eccentric = jnp.arccos(cos_eccentric)
        eccentric = jnp.where(v_r >= 0.0, eccentric, 2 * jnp.pi - eccentric)
        mean_anomaly = eccentric - ecc * jnp.sin(eccentric)
        to_apoapsis = jnp.mod(jnp.pi - mean_anomaly, 2 * jnp.pi) * jnp.sqrt(a_bound**3 / _GM)
        # Duration of the second burn: the velocity change it needs at the
        # apoapsis over the thrust, in whole steps for the discrete actions.
        second = jnp.sqrt(_GM / apoapsis) - r * v_t / apoapsis
        if self.action == "continuous":
            duration = second / params.max_thrust
        else:
            whole = jnp.floor(second / full)
            tenths = jnp.ceil((second - whole * full) / (0.1 * full))
            duration = (whole + tenths) * self.control_interval
        # Burn from half the duration before the apoapsis until the orbit is
        # circular, or until the spacecraft has fallen 5 % below the apoapsis.
        window = (bound & (to_apoapsis <= 0.5 * duration)) | ((v_r <= 0.0) & (r > 0.95 * apoapsis))
        circularize = ~raise_apoapsis & (ecc > 0.25 * params.eccentricity_tolerance) & window
        # The discrete controller thrusts prograde only: a radial correction
        # would cost a full or a tenth of a thrust where it needs less.
        cancel_radial = circularize if self.action == "continuous" else False
        needed_r = jnp.where(cancel_radial, -v_r, 0.0)
        needed_t = jnp.where(
            raise_apoapsis,
            jnp.sqrt(jnp.maximum(apoapsis_speed2, 0.0)) - v_t,
            jnp.where(circularize, jnp.sqrt(_GM / r) - v_t, 0.0),
        )
        local = jnp.stack([needed_r, jnp.maximum(needed_t, 0.0)], axis=1)
        local = local / full
        local = local / jnp.maximum(jnp.linalg.norm(local, axis=1, keepdims=True), 1.0)
        if self.action == "continuous":
            return local
        prograde = local[:, 1]
        return jnp.where(prograde >= 0.999, 1, jnp.where(prograde >= 0.05, 5, 0))

    def render(
        self,
        state: OrbitTransferState,
        width: int = 360,
        params: OrbitTransferParams | None = None,
    ) -> np.ndarray:
        """Draw the planet, the first and the target orbit, and the spacecraft."""
        params = self.default_params if params is None else params
        extent = 1.15 * max(float(params.target_radius), _R0)
        canvas = viz.Canvas((-extent, -extent), (extent, extent), width)
        angle = np.linspace(0.0, 2 * np.pi, 181)
        circle = np.stack([np.cos(angle), np.sin(angle)], axis=1)
        canvas.polyline(_R0 * circle, viz.LIGHT_GREY, 0.012 * extent)
        canvas.polyline(float(params.target_radius) * circle, viz.GREEN, 0.012 * extent)
        canvas.discs(np.zeros((1, 2)), self.planet_radius, viz.BLUE)
        pos = np.asarray(state.particles.pos)[1:]
        canvas.discs(pos, 0.035 * extent, viz.RED)
        return canvas.frame()

    def _relative(self, particles: pp.ParticleState):
        """Positions and velocities of the spacecraft relative to the planet."""
        return (
            particles.pos[1:] - particles.pos[:1],
            particles.vel[1:] - particles.vel[:1],
        )

    def _frame(self, particles: pp.ParticleState):
        """Unit vectors (n, 2): radially out, and transverse in the direction of the motion."""
        position, velocity = self._relative(particles)
        radial = position / jnp.linalg.norm(position, axis=1, keepdims=True)
        normal = jnp.stack([-radial[:, 1], radial[:, 0]], axis=1)
        sense = jnp.where(jnp.sum(velocity * normal, axis=1) < 0.0, -1.0, 1.0)
        return radial, sense[:, None] * normal

    def _thrust_command(self, action: jax.Array, dtype) -> jax.Array:
        """(n, 2) thrust in the (radial, transverse) frame, of length at most 1."""
        if self.action == "discrete":
            index = jnp.reshape(action, (self.n_agents,))
            return jnp.asarray(self.directions, dtype)[index]
        local = jnp.reshape(jnp.asarray(action, dtype), (self.n_agents, 2))
        length = jnp.linalg.norm(local, axis=1, keepdims=True)
        return local / jnp.maximum(length, 1.0)

    def _observe(self, state: OrbitTransferState, params: OrbitTransferParams) -> jax.Array:
        position, velocity = self._relative(state.particles)
        radial, transverse = self._frame(state.particles)
        r = jnp.linalg.norm(position, axis=1)
        v_r = jnp.sum(velocity * radial, axis=1)
        v_t = jnp.sum(velocity * transverse, axis=1)
        target = jnp.broadcast_to(jnp.asarray(params.target_radius, r.dtype), r.shape)
        fuel_left = jnp.maximum(params.fuel - state.fuel_used, 0.0)
        return jnp.stack([r, v_r, v_t, target, fuel_left], axis=1)
