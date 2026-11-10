# ParticleGym, https://github.com/ZhaojieC/particlegym (MIT license).
# Please cite it in published work: particlegym.cite() gives the references.
"""Ring road: automated vehicles smooth out the stop-and-go waves of human drivers.

N vehicles drive in one lane of a ring road, as in the experiment of
Sugiyama et al. (2008), where 22 cars on a circuit of 230 m formed a traffic
jam without any bottleneck. The human drivers follow the intelligent driver
model (Treiber, Hennecke & Helbing 2000) with random fluctuations of their
acceleration. At this density uniform flow is unstable: small fluctuations
grow into a stop-and-go wave, a "phantom jam" that travels backward around
the ring. Some vehicles are automated, and the agents choose their
acceleration. This is the mixed-autonomy ring of the Flow framework (Wu et
al. 2022) and of the field experiment of Stern et al. (2018), in which one
automated vehicle dissipated the waves of more than 20 human drivers.

The vehicles move by the car-following physics of
:mod:`particlegym.physics.traffic`. An episode starts in a stop-and-go
wave: the gaps vary around the ring like a cosine, and every vehicle drives
at the speed of uniform flow at its gap. This environment is experimental.
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

# Accelerations of the discrete actions, as fractions of the largest one.
_LEVELS = np.array([-1.0, -0.3, -0.1, -0.03, 0.0, 0.03, 0.1, 0.3, 1.0])


class RingRoadParams(NamedTuple):
    """Episode parameters. A pytree, so values may be traced or batched.

    Attributes:
        ring_length: Length L of the ring road (m); 230 m in the experiment
            of Sugiyama et al. (2008).
        noise: Strength sigma of the random acceleration of human drivers
            (m s^-3/2): in a time dt their speed changes by an extra
            sigma sqrt(dt) times a standard normal number.
        wave: Relative amplitude of the gaps of the initial wave; 0 starts
            in uniform flow.
        max_steps: Episode length in control steps.
    """

    ring_length: float = 230.0
    noise: float = 0.2
    wave: float = 0.5
    max_steps: int = 600


class RingRoadState(NamedTuple):
    """Environment state. A pytree.

    Attributes:
        position: (N,) positions of the vehicle fronts along the ring (m);
            vehicle i + 1 drives ahead of vehicle i.
        speed: (N,) speeds (m/s).
        crashed: Whether two vehicles have collided; the vehicles then stand still.
        step: Number of control steps taken.
    """

    position: jax.Array
    speed: jax.Array
    crashed: jax.Array
    step: jax.Array


class RingRoad:
    """Mixed traffic on a ring road: human drivers and automated vehicles.

    Each agent drives one automated vehicle. It observes its speed, the
    speed of the vehicle ahead and the gap to it, and chooses an
    acceleration. With ``safe=True`` an automated vehicle never accelerates
    more than a human driver would in its place, which prevents collisions.
    All agents receive the same reward: the distance that the vehicles have
    driven in the step, averaged over all vehicles. Its sum over an episode
    is the mean distance driven, which stop-and-go waves reduce.

    Args:
        n_vehicles: Number of vehicles N.
        n_agents: Number of automated vehicles, spread evenly over the ring.
        action: ``"continuous"`` (an acceleration in [-1, 1] times
            ``max_accel``) or ``"discrete"`` (one of nine: -1, -0.3, -0.1,
            -0.03, 0, 0.03, 0.1, 0.3 or 1 times ``max_accel``).
        driver: Car-following model of the human drivers; the intelligent
            driver model with the parameters of Flow if None.
        safe: Cap the acceleration of the automated vehicles at the one of a
            human driver.
        vehicle_length: Length of every vehicle (m).
        max_accel: Largest acceleration and deceleration that an agent can
            ask for (m/s^2).
        dt: Time step of the physics (s).
        substeps: Physics steps per control step.
    """

    # Papers behind this environment, listed by particlegym.cite.
    references = ("sugiyama2008", "treiber2000", "wu2022")

    def __init__(
        self,
        n_vehicles: int = 22,
        n_agents: int = 1,
        action: str = "continuous",
        driver: pp.CarFollowing | None = None,
        safe: bool = True,
        vehicle_length: float = 5.0,
        max_accel: float = 1.0,
        dt: float = 0.1,
        substeps: int = 5,
    ) -> None:
        if not 1 <= n_agents <= n_vehicles:
            raise ValueError("need 1 <= n_agents <= n_vehicles")
        if action not in ("continuous", "discrete"):
            raise ValueError("action must be 'continuous' or 'discrete'")
        if vehicle_length <= 0 or max_accel <= 0 or dt <= 0 or substeps < 1:
            raise ValueError("vehicle_length, max_accel and dt must be positive, substeps >= 1")
        self.n_vehicles, self.n_agents = int(n_vehicles), int(n_agents)
        self.action = action
        self.driver = pp.IntelligentDriver() if driver is None else driver
        self.safe = bool(safe)
        self.vehicle_length, self.max_accel = float(vehicle_length), float(max_accel)
        self.dt, self.substeps = float(dt), int(substeps)
        self.control_interval = self.dt * self.substeps
        # The automated vehicles, evenly spaced along the ring.
        self.agents = np.round(np.arange(n_agents) * n_vehicles / n_agents).astype(np.int32)
        self.levels = _LEVELS
        remind(self)

    @property
    def default_params(self) -> RingRoadParams:
        """Default episode parameters."""
        return RingRoadParams()

    @property
    def observation_space(self) -> Continuous:
        """Observation of one agent: its speed, the speed of the vehicle ahead, the gap to it."""
        return Continuous(-jnp.inf, jnp.inf, (3,))

    @property
    def action_space(self) -> Discrete | Continuous:
        """Action of one agent."""
        if self.action == "discrete":
            return Discrete(len(self.levels))
        return Continuous(-1.0, 1.0, (1,))

    def uniform_speed(self, params: RingRoadParams | None = None) -> jax.Array:
        """Speed of uniform flow: all vehicles at the same gap L / N - l."""
        params = self.default_params if params is None else params
        gap = params.ring_length / self.n_vehicles - self.vehicle_length
        return self.driver.equilibrium_speed(jnp.asarray(gap, dtype=float))

    def reset(
        self, key: jax.Array, params: RingRoadParams | None = None
    ) -> tuple[jax.Array, RingRoadState]:
        """Start in a stop-and-go wave with a random phase and a random place on the ring."""
        params = self.default_params if params is None else params
        n = self.n_vehicles
        phase_key, shift_key = jax.random.split(key)
        mean_gap = params.ring_length / n - self.vehicle_length
        theta = 2 * jnp.pi * jnp.arange(n) / n + jax.random.uniform(phase_key, maxval=2 * jnp.pi)
        gaps = mean_gap * (1.0 + params.wave * jnp.cos(theta))
        fronts = jnp.concatenate([jnp.zeros(1), jnp.cumsum(gaps + self.vehicle_length)[:-1]])
        shift = jax.random.uniform(shift_key, dtype=fronts.dtype) * params.ring_length
        state = RingRoadState(
            position=jnp.mod(fronts + shift, params.ring_length),
            speed=self.driver.equilibrium_speed(gaps),
            crashed=jnp.zeros((), bool),
            step=jnp.zeros((), jnp.int32),
        )
        return self._observe(state, params), state

    def step(
        self,
        key: jax.Array,
        state: RingRoadState,
        action: jax.Array,
        params: RingRoadParams | None = None,
    ):
        """Advance the environment by one control step.

        The human drivers follow the car-following model plus their random
        accelerations; the automated vehicles keep the acceleration that the
        agents ask for, capped by the safety rule. After a collision every
        vehicle stands still and the reward is zero.

        Returns:
            ``(obs, state, reward, terminated, truncated, info)``. ``info`` holds
            the ``mean_speed`` and the ``speed_std`` of all vehicles, the
            smallest gap ``min_gap`` and whether the vehicles have ``crashed``.
        """
        if key is None:
            raise ValueError("a random key is required for the fluctuations of the drivers")
        params = self.default_params if params is None else params
        command = self._command(action)
        length, ring, agents = self.vehicle_length, params.ring_length, self.agents
        dtype = state.speed.dtype

        def substep(carry, k):
            position, speed, crashed, distance = carry
            gaps = pp.ring_gaps(position, ring, length)
            driver = self.driver.acceleration(gaps, speed, jnp.roll(speed, -1))
            noise = jax.random.normal(k, speed.shape, dtype) / math.sqrt(self.dt)
            agent_accel = jnp.minimum(command, driver[agents]) if self.safe else command
            accel = (driver + params.noise * noise).at[agents].set(agent_accel)
            new_position, new_speed = pp.ballistic_step(position, speed, accel, self.dt, ring)
            moved = jnp.mod(new_position - position, ring)
            new_position = jnp.where(crashed, position, new_position)
            new_speed = jnp.where(crashed, speed, new_speed)
            distance = distance + jnp.where(crashed, 0.0, moved)
            crashed = crashed | jnp.any(pp.ring_gaps(new_position, ring, length) < 0.0)
            return (new_position, new_speed, crashed, distance), None

        carry = (state.position, state.speed, state.crashed, jnp.zeros_like(state.speed))
        keys = jax.random.split(key, self.substeps)
        (position, speed, crashed, distance), _ = jax.lax.scan(substep, carry, keys)
        speed = jnp.where(crashed, 0.0, speed)
        step = state.step + 1
        state = RingRoadState(position, speed, crashed, step)
        gaps = pp.ring_gaps(position, ring, length)
        shape = (self.n_agents,)
        reward = jnp.broadcast_to(jnp.mean(distance), shape)
        terminated = jnp.broadcast_to(crashed, shape)
        truncated = jnp.broadcast_to(~crashed & (step >= params.max_steps), shape)
        info = {
            "mean_speed": jnp.mean(speed),
            "speed_std": jnp.std(speed),
            "min_gap": jnp.min(gaps),
            "crashed": crashed,
        }
        return self._observe(state, params), state, reward, terminated, truncated, info

    def idm_policy(self, state: RingRoadState, params: RingRoadParams | None = None):
        """Ask for the acceleration of a human driver, without the fluctuations.

        With this policy the ring behaves almost as if all vehicles were
        driven by humans: the reference without control. The agent decides
        once per control step, while the human drivers adapt after every
        physics step.
        """
        params = self.default_params if params is None else params
        gaps = pp.ring_gaps(state.position, params.ring_length, self.vehicle_length)
        leader = jnp.roll(state.speed, -1)
        accel = self.driver.acceleration(gaps, state.speed, leader)
        return self._to_action(accel[self.agents])

    def smoothing_policy(
        self,
        state: RingRoadState,
        params: RingRoadParams | None = None,
        fraction: float = 0.9,
        gain: float = 0.1,
    ):
        """Cruise gently at a fixed speed below the speed of uniform flow.

        The automated vehicle accelerates with ``gain * (U - v)`` towards the
        speed U = ``fraction`` times the speed of uniform flow, which it
        computes from the ring length and the number of vehicles. A small
        ``gain`` (a time constant of 10 s) lets it leave the jam slowly and
        open a gap to the vehicle ahead, which absorbs the next wave; the
        safety rule brakes when needed. Without that gap, a vehicle that
        cruises at the speed of uniform flow runs into the back of the jam.
        """
        params = self.default_params if params is None else params
        target = fraction * self.uniform_speed(params)
        return self._to_action(gain * (target - state.speed[self.agents]))

    def render(
        self, state: RingRoadState, width: int = 360, params: RingRoadParams | None = None
    ) -> np.ndarray:
        """Draw the ring road and the vehicles; they drive counter-clockwise.

        Human drivers are grey, and turn red as they slow down below half the
        speed of uniform flow; automated vehicles are blue.
        """
        params = self.default_params if params is None else params
        ring = float(params.ring_length)
        radius = ring / (2 * np.pi)
        extent = radius + 4 * self.vehicle_length
        canvas = viz.Canvas((-extent, -extent), (extent, extent), width)
        lane = 0.8 * self.vehicle_length  # half the width of the road
        canvas.discs(np.zeros((1, 2)), radius + lane, viz.LIGHT_GREY)
        canvas.discs(np.zeros((1, 2)), radius - lane, viz.WHITE)
        position, speed = np.asarray(state.position), np.asarray(state.speed)
        phi = 2 * np.pi * position / ring
        xy = radius * np.stack([np.cos(phi), np.sin(phi)], axis=1)
        slow = 0.5 * max(float(self.uniform_speed(params)), 1e-6)
        fast = np.clip(speed / slow, 0.0, 1.0)
        colors = np.asarray(viz.RED) + fast[:, None] * (np.subtract(viz.GREY, viz.RED))
        colors[self.agents] = viz.BLUE
        canvas.discs(xy, 0.5 * self.vehicle_length, colors.astype(np.uint8))
        return canvas.frame()

    def _command(self, action: jax.Array) -> jax.Array:
        """(n_agents,) accelerations asked for by the agents."""
        if self.action == "discrete":
            index = jnp.reshape(action, (self.n_agents,))
            return self.max_accel * jnp.asarray(self.levels)[index]
        value = jnp.reshape(jnp.asarray(action, dtype=float), (self.n_agents,))
        return self.max_accel * jnp.clip(value, -1.0, 1.0)

    def _to_action(self, accel: jax.Array) -> jax.Array:
        """The action that asks for an acceleration, or the nearest discrete one."""
        if self.action == "discrete":
            levels = self.max_accel * jnp.asarray(self.levels)
            return jnp.argmin(jnp.abs(accel[:, None] - levels[None, :]), axis=1)
        return jnp.clip(accel / self.max_accel, -1.0, 1.0)[:, None]

    def _observe(self, state: RingRoadState, params: RingRoadParams) -> jax.Array:
        gaps = pp.ring_gaps(state.position, params.ring_length, self.vehicle_length)
        leader = jnp.roll(state.speed, -1)
        agents = self.agents
        return jnp.stack([state.speed[agents], leader[agents], gaps[agents]], axis=1)
