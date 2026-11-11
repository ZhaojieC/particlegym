# ParticleGym, https://github.com/ZhaojieC/particlegym (MIT license).
# Please cite it in published work: particlegym.cite() gives the references.
"""Soft walker: a robot of soft voxels learns to walk.

The robot is a :class:`~particlegym.physics.VoxelBody`: point masses at the
corners of square voxels, joined by damped springs along the edges and the
diagonals. Actuator voxels lengthen or shorten their springs on command, as
muscles do. Under gravity the robot stands on the ground, which pushes back
and holds its feet by Coulomb friction; only these ground forces can move the
robot as a whole, so it has to find a gait. The agent sets the commands of
all actuators and is rewarded for the distance its centre of mass moves
forward (to the right). The voxel model follows the voxel-based soft robots
of Evolution Gym (Bhatia et al. 2021); the robots, the units and the contact
model are those of ParticleGym. This environment is experimental.

Units: the voxel size, the mass of a voxel and the gravitational
acceleration are 1, so the time unit is sqrt(voxel size / g); for voxels of
10 cm it is 0.1 s.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from particlegym import physics as pp
from particlegym import viz
from particlegym.citation import remind
from particlegym.envs.spaces import Continuous

# A robot with a trunk of horizontal actuators, a soft layer and three legs
# of vertical actuators.
WALKER = ("HHHHH", "SSSSS", "V.V.V")
_COLORS = {0: (170, 180, 195), 1: (90, 95, 105), 2: (240, 150, 60), 3: (70, 130, 230)}


class SoftWalkerParams(NamedTuple):
    """Episode parameters. A pytree, so values may be traced or batched.

    Attributes:
        friction: Coulomb friction coefficient between the robot and the ground.
        max_steps: Episode length in control steps.
    """

    friction: float = 1.0
    max_steps: int = 500


class SoftWalkerState(NamedTuple):
    """Environment state. A pytree.

    Attributes:
        particles: The corners of the voxels.
        activation: (n_actuators,) the activations of the actuators, which
            follow the commands with a delay.
        step: Number of control steps taken.
    """

    particles: pp.ParticleState
    activation: jax.Array
    step: jax.Array


class SoftWalker:
    """A soft robot of voxels that walks on flat ground.

    The single agent observes the velocity and the height of the centre of
    mass, the positions and velocities of all corners relative to it, and the
    activations of the actuators; it sets a command in [-1, 1] for every
    actuator voxel, which lengthens or shortens the voxel by ``strength``
    along its direction. The reward is the distance that the centre of mass
    moves to the right in the step.

    Args:
        layout: The robot, as rows of :class:`~particlegym.physics.VoxelBody`
            characters from top to bottom.
        strength: Largest relative change of the rest length of an actuator.
        stiffness: Spring constant of soft voxels.
        rigidity: Stiffness of rigid voxels relative to soft ones.
        damping: Damping coefficient of the dashpots along every spring.
        ground_stiffness: Stiffness of the contact with the ground.
        ground_damping: Damping of the contact with the ground.
        slip_speed: Slip speed of the smooth Coulomb friction.
        activation_time: Time in which the activation of an actuator relaxes
            to its command, as for a muscle.
        jitter: Largest random displacement of a corner at the start.
        dt: Time step of the physics.
        substeps: Physics steps per control step.
    """

    # Papers behind this environment, listed by particlegym.cite.
    references = ("bhatia2021",)

    def __init__(
        self,
        layout=WALKER,
        strength: float = 0.25,
        stiffness: float = 200.0,
        rigidity: float = 3.0,
        damping: float = 7.0,
        ground_stiffness: float = 3000.0,
        ground_damping: float = 40.0,
        slip_speed: float = 0.1,
        activation_time: float = 0.2,
        jitter: float = 1e-3,
        dt: float = 0.005,
        substeps: int = 20,
    ) -> None:
        self.body = pp.VoxelBody(tuple(layout))
        if self.body.n_actuators == 0:
            raise ValueError("the robot needs at least one actuator voxel (H or V)")
        if not 0 < strength < 1:
            raise ValueError("strength must be between 0 and 1")
        if dt <= 0 or substeps < 1 or activation_time <= 0:
            raise ValueError("dt and activation_time must be positive, substeps at least 1")
        self.n_agents = 1
        self.strength = float(strength)
        self.spring_constants = self.body.stiffness(stiffness, rigidity)
        self.damping = float(damping)
        self.ground_stiffness, self.ground_damping = float(ground_stiffness), float(ground_damping)
        self.slip_speed = float(slip_speed)
        self.dt, self.substeps = float(dt), int(substeps)
        self.control_interval = self.dt * self.substeps
        self.activation_time = float(activation_time)
        self.jitter = float(jitter)
        self.box = pp.Box((-1e5, -1e3), (1e5, 1e3))
        remind(self)

    @property
    def n_actuators(self) -> int:
        """Number of actuator voxels, the length of an action."""
        return self.body.n_actuators

    @property
    def default_params(self) -> SoftWalkerParams:
        """Default episode parameters."""
        return SoftWalkerParams()

    @property
    def observation_space(self) -> Continuous:
        """Velocity and height of the centre of mass, corners relative to it, activations."""
        size = 3 + 4 * self.body.n_nodes + self.n_actuators
        return Continuous(-jnp.inf, jnp.inf, (size,))

    @property
    def action_space(self) -> Continuous:
        """One command in [-1, 1] per actuator voxel."""
        return Continuous(-1.0, 1.0, (self.n_actuators,))

    def force_field(self, activation: jax.Array, params: SoftWalkerParams) -> pp.ForceField:
        """The forces on the corners for the activations of the actuators."""
        springs = self.body.springs
        rest = self.body.rest_lengths(activation, self.strength)
        ground = pp.Ground(
            stiffness=self.ground_stiffness,
            damping=self.ground_damping,
            friction=params.friction,
            slip_speed=self.slip_speed,
        )
        return pp.ForceField(
            self.box,
            bonded=[pp.Bonds(springs, k=self.spring_constants, r0=rest)],
            external=[pp.Gravity((0.0, -1.0)), ground, pp.Dashpots(springs, self.damping)],
        )

    def reset(
        self, key: jax.Array, params: SoftWalkerParams | None = None
    ) -> tuple[jax.Array, SoftWalkerState]:
        """Put the robot at rest on the ground, its centre of mass above x = 0.

        The corners are displaced at random by up to ``jitter`` voxels. Walking
        is chaotic, so these displacements give every episode its own
        course.
        """
        params = self.default_params if params is None else params
        masses = jnp.asarray(self.body.masses)
        positions = jnp.asarray(self.body.positions)
        shape = positions.shape
        positions = positions + self.jitter * jax.random.uniform(key, shape, minval=-1, maxval=1)
        positions = positions.at[:, 1].add(-jnp.min(positions[:, 1]))  # standing on the ground
        centre = jnp.sum(masses * positions[:, 0]) / jnp.sum(masses)
        positions = positions.at[:, 0].add(-centre)
        particles = pp.create_state(positions, mass=masses, radius=0.0)
        activation = jnp.zeros(self.n_actuators, positions.dtype)
        particles, _ = pp.init_forces(particles, self.force_field(activation, params))
        state = SoftWalkerState(particles, activation, jnp.zeros((), jnp.int32))
        return self._observe(state), state

    def step(
        self,
        key: jax.Array,
        state: SoftWalkerState,
        action: jax.Array,
        params: SoftWalkerParams | None = None,
    ):
        """Advance the environment by one control step.

        The activation of every actuator relaxes exponentially to its command
        with the time constant ``activation_time``, and sets the rest lengths
        of the actuator's springs at every physics step.

        Returns:
            ``(obs, state, reward, terminated, truncated, info)``. ``info`` holds
            the position ``x``, the velocity ``speed`` and the height of the
            centre of mass.
        """
        params = self.default_params if params is None else params
        dtype = state.particles.pos.dtype
        command = jnp.clip(jnp.reshape(jnp.asarray(action, dtype), (self.n_actuators,)), -1, 1)
        relax = float(np.exp(-self.dt / self.activation_time))

        def substep(carry, _):
            p, activation = carry
            activation = command + (activation - command) * relax
            p, _ = pp.velocity_verlet(p, self.force_field(activation, params), self.dt)
            return (p, activation), None

        carry = (state.particles, state.activation)
        (particles, activation), _ = jax.lax.scan(substep, carry, None, length=self.substeps)
        before, _ = self.centre_of_mass(state.particles)
        after, velocity = self.centre_of_mass(particles)
        step = state.step + 1
        state = SoftWalkerState(particles, activation, step)
        reward = jnp.reshape(after[0] - before[0], (1,))
        terminated = jnp.zeros(1, bool)
        truncated = jnp.reshape(step >= params.max_steps, (1,))
        info = {"x": after[0], "speed": velocity[0], "height": after[1]}
        return self._observe(state), state, reward, terminated, truncated, info

    def centre_of_mass(self, particles: pp.ParticleState) -> tuple[jax.Array, jax.Array]:
        """Position and velocity (2,) of the centre of mass."""
        weights = particles.mass / jnp.sum(particles.mass)
        return weights @ particles.pos, weights @ particles.vel

    def gait_policy(
        self,
        state: SoftWalkerState,
        params: SoftWalkerParams | None = None,
        period: float = 1.0,
        wavelength: float = 8.0,
    ) -> jax.Array:
        """An open-loop gait: a wave of contraction that runs along the body.

        Actuator i, in column x_i, gets the command
        ``sin(2 pi (t / period - x_i / wavelength))``: a wave that travels
        forward at ``wavelength / period``. The policy ignores the state
        except for the time.
        """
        t = state.step * self.control_interval
        column = jnp.asarray(self.body.cells[self.body.actuators, 0], dtype=float)
        phase = 2 * jnp.pi * (t / period - column / wavelength)
        return jnp.sin(phase)[None, :]

    def render(
        self,
        state: SoftWalkerState,
        width: int = 480,
        params: SoftWalkerParams | None = None,
        view: float = 12.0,
    ) -> np.ndarray:
        """Draw the robot and the ground around it.

        The view follows the robot: ``view`` voxels wide, with a mark on the
        ground at every whole voxel. Soft voxels are light grey, rigid ones
        dark grey, horizontal actuators orange and vertical ones blue; an
        actuator is lighter when it lengthens and darker when it shortens.
        """
        pos = np.asarray(state.particles.pos)
        centre = float(np.asarray(self.centre_of_mass(state.particles)[0])[0])
        height = max(self.body.positions[:, 1].max() + 2.0, 0.5 * view)
        lo, hi = (centre - 0.5 * view, -1.0), (centre + 0.5 * view, height - 1.0)
        canvas = viz.Canvas(lo, hi, width)
        canvas.rectangles([lo[0], -1.0], [hi[0], 0.0], viz.LIGHT_GREY)
        marks = np.arange(np.floor(lo[0]), np.ceil(hi[0]) + 1)
        canvas.rectangles(
            np.stack([marks - 0.03, np.full_like(marks, -0.35)], 1),
            np.stack([marks + 0.03, np.zeros_like(marks)], 1),
            viz.GREY,
        )
        colors = np.array([_COLORS[k] for k in self.body.kinds], dtype=float)
        shade = np.zeros(self.body.n_voxels)
        shade[self.body.actuators] = np.asarray(state.activation)
        white, black = np.array(viz.WHITE, float), np.array(viz.BLACK, float)
        lighter = colors + 0.45 * np.clip(shade, 0, 1)[:, None] * (white - colors)
        colors = lighter + 0.45 * np.clip(-shade, 0, 1)[:, None] * (black - lighter)
        canvas.polygons(pos[self.body.voxels], colors.astype(np.uint8))
        return canvas.frame()

    def _observe(self, state: SoftWalkerState) -> jax.Array:
        position, velocity = self.centre_of_mass(state.particles)
        relative = state.particles.pos - position
        moving = state.particles.vel - velocity
        parts = [velocity, position[1:], relative.ravel(), moving.ravel(), state.activation]
        return jnp.concatenate(parts)[None, :]
