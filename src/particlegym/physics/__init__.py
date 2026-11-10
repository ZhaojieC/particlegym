# ParticleGym, https://github.com/ZhaojieC/particlegym (MIT license).
# Please cite it in published work: particlegym.cite() gives the references.
"""Particle-dynamics core: 2D/3D, written in JAX, independent of any RL interface.

The module :mod:`particlegym.physics.traffic` adds vehicles in one lane of a ring road.
"""

from particlegym.physics import active
from particlegym.physics.bonded import Angles, AreaConstraint, BondedTerm, Bonds
from particlegym.physics.external import ExternalForce, Gravity, KineticFriction
from particlegym.physics.fields import Grid
from particlegym.physics.flow import FlowField, ShearFlow, TaylorGreen, UniformFlow
from particlegym.physics.forces import ForceField, ForceInfo
from particlegym.physics.geometry import (
    Boxes,
    Opening,
    Plates,
    ShapeSet,
    Spheres,
    Walls,
    box_walls,
)
from particlegym.physics.initial import lattice_positions, maxwell_boltzmann, random_positions
from particlegym.physics.integrators import (
    brownian,
    init_forces,
    langevin_baoab,
    velocity_verlet,
)
from particlegym.physics.nbody import NewtonGravity, hohmann_delta_v, orbital_elements
from particlegym.physics.neighbors import BruteForce, CellList
from particlegym.physics.observables import kinetic_energy, pressure, temperature, total_momentum
from particlegym.physics.potentials import (
    WCA,
    ContactExponential,
    Harmonic,
    Mixture,
    NormalDamping,
    PairDissipation,
    PairPotential,
    PairThermostat,
    SocialForce,
    TangentialFriction,
    damping_for_restitution,
    restitution_coefficient,
)
from particlegym.physics.space import Box
from particlegym.physics.state import ParticleState, create_state
from particlegym.physics.traffic import (
    CarFollowing,
    IntelligentDriver,
    OptimalVelocity,
    ballistic_step,
    ring_accelerations,
    ring_gaps,
    ring_modes,
)

__all__ = [
    "WCA",
    "Angles",
    "AreaConstraint",
    "BondedTerm",
    "Bonds",
    "Box",
    "Boxes",
    "BruteForce",
    "CarFollowing",
    "CellList",
    "ContactExponential",
    "ExternalForce",
    "FlowField",
    "ForceField",
    "ForceInfo",
    "Gravity",
    "Grid",
    "Harmonic",
    "IntelligentDriver",
    "KineticFriction",
    "Mixture",
    "NewtonGravity",
    "NormalDamping",
    "Opening",
    "OptimalVelocity",
    "PairDissipation",
    "PairPotential",
    "PairThermostat",
    "ParticleState",
    "Plates",
    "ShapeSet",
    "ShearFlow",
    "SocialForce",
    "Spheres",
    "TangentialFriction",
    "TaylorGreen",
    "UniformFlow",
    "Walls",
    "active",
    "ballistic_step",
    "box_walls",
    "brownian",
    "create_state",
    "damping_for_restitution",
    "hohmann_delta_v",
    "init_forces",
    "kinetic_energy",
    "langevin_baoab",
    "lattice_positions",
    "maxwell_boltzmann",
    "orbital_elements",
    "pressure",
    "random_positions",
    "restitution_coefficient",
    "ring_accelerations",
    "ring_gaps",
    "ring_modes",
    "temperature",
    "total_momentum",
    "velocity_verlet",
]
