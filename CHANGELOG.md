# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [Unreleased]

### Added

- Functional room-evacuation environment `particlegym.envs.Evacuation`:
  `reset`/`step` as pure functions that compile with `jax.jit` and batch with
  `jax.vmap`, including over episode parameters; discrete or continuous
  desired directions; several physics substeps per control step; exits
  through real doors with the alive mask and recorded exit times; crowd
  policies for pedestrians that are not agents
  ([#3](https://github.com/ZhaojieC/particlegym/issues/3)).
- Example: room evacuation with agents that walk to the nearest door.
- Three-dimensional rooms in `Evacuation`: a `room` with three edge lengths
  gives a 3D chamber in which pedestrians move freely in all directions
  (no gravity), with rectangular doors in the walls, the floor and the
  ceiling, and 6, 18 or 26 discrete directions. The 2D room is unchanged.
- Example: evacuation of a 3D chamber through a door and a ceiling hatch.
- Module `particlegym.viz`: pictures of 2D systems (`Canvas`), GIF and MP4
  writers, `Evacuation.render`, and pixel images computed in JAX for image
  observations. Pillow is an optional dependency (`pip install
  "particlegym[viz]"`).
- Gallery: scripts in `examples/gallery` that make the animations shown in
  the README and on the new gallery page of the documentation.
- Workflow that publishes the documentation with GitHub Pages.
- Physics capabilities for more kinds of environments, each with validation
  tests (see `docs/validation.md`):
  - external forces (`Gravity`, `KineticFriction`) and prescribed flow fields
    (`UniformFlow`, `ShearFlow`, `TaylorGreen`);
  - shapes that move or are switched off during a simulation
    (`Walls(active=..., velocity=...)`, shape sets built inside compiled code);
  - contacts with a coefficient of restitution (`NormalDamping`,
    `Walls(damping=...)`, `damping_for_restitution`);
  - active particles in 2D (`particlegym.physics.active`): rotational
    diffusion, steering, alignment, polar order;
  - bonded terms (`Bonds`, `Angles`, `AreaConstraint`);
  - dissipative particle dynamics (`PairThermostat`) with momentum-conserving
    pair noise, and the virial `pressure`;
  - scalar fields on a grid (`Grid`): deposit, sampling, gradient, diffusion;
  - species-dependent interactions (`Mixture`) and the contact force of the
    cell-migration model (`ContactExponential`).
- Benchmark `benchmarks/dpd_fluid.py`: temperature and pressure of the DPD
  fluid, with its reference output.

- Project scaffold: packaging, CI, documentation site, contribution guidelines.
- Physics core (2D and 3D, JAX):
  - simulation box with per-axis periodicity;
  - particle state with an `alive` mask for absorbing exits;
  - brute-force and cell-list neighbour search with overflow detection;
  - pair potentials (harmonic, WCA, Helbing social force) with forces from
    automatic differentiation, plus tangential sliding friction;
  - geometry from distance functions (plates, spheres, boxes, rooms with real
    door openings), with wall potentials and wall friction;
  - velocity Verlet and BAOAB Langevin integrators, where the Langevin drift
    velocity carries the self-propulsion of the social force model;
  - observables and an extended-XYZ writer for OVITO.
- Physics validation test suite (see `docs/validation.md`).
- Design document, physics reference, roadmap and AI usage log.
- Examples: 3D Langevin WCA fluid, 2D room evacuation through a real door.
- Benchmark script.

### Changed

- `Evacuation` has the attributes `dim` and `size` instead of `width` and
  `height`, and `n_directions` defaults to 8 in 2D and 26 in 3D.
- The distance to 3D plates is computed edge by edge, which makes forces from
  3D walls about 10 times faster in single precision on CPU (JAX 0.11);
  distances are unchanged and forces agree to round-off.
- Version 1.0 focuses on pedestrian evacuation (design decision D9); the
  roadmap is reordered accordingly.
