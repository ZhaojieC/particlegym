# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [Unreleased]

### Added

- `particlegym.cite`: the references for ParticleGym and for the published
  models behind each environment (`env.references`), as text or BibTeX. The
  first environment of each kind that a program creates prints them once;
  `PARTICLEGYM_CITE=0` turns this off.
- Every source file of the package and of the examples starts with a line
  that asks for a citation.
- `physics.NewtonGravity`: Newtonian gravity with Plummer softening, between
  all particles or from the first few particles on test particles (the
  restricted N-body problem); `physics.orbital_elements` and
  `physics.hohmann_delta_v` for Kepler orbits.
- Experimental environment `OrbitTransfer` (Gymnasium id
  `particlegym/OrbitTransfer-v0`): spacecraft raise their orbit on as little
  fuel as possible, with continuous or discrete thrust; the Hohmann transfer
  flown in closed loop (`hohmann_policy`) is the baseline.
- Gallery animations of a galaxy encounter after Toomre & Toomre (1972) and of
  an orbit transfer.
- `physics.traffic`: the car-following models `IntelligentDriver` (Treiber,
  Hennecke & Helbing 2000) and `OptimalVelocity` (Bando et al. 1995) in one
  lane of a ring road, the ballistic update with stops, and `ring_modes`, the
  linear stability of uniform flow for any car-following model by automatic
  differentiation.
- Experimental environment `RingRoad` (Gymnasium id
  `particlegym/RingRoad-v0`): automated vehicles among human drivers on the
  ring of Sugiyama et al. (2008). A gently cruising automated vehicle
  (`smoothing_policy`) dissolves the stop-and-go wave; every vehicle then
  drives 21 to 25 % farther.
- Gallery: a traffic jam without a bottleneck and its dissolution, and a
  space-time diagram of the vehicles.
- `physics.VoxelBody` (soft bodies of square voxels with point masses at the
  corners, springs along edges and diagonals, and horizontal and vertical
  actuators, as in Evolution Gym), `physics.Dashpots` (linear dampers between
  listed particles) and `physics.Ground` (a flat floor with an elastic
  contact and Coulomb friction); `viz.Canvas.polygons`.
- Experimental environment `SoftWalker` (Gymnasium id
  `particlegym/SoftWalker-v0`): a robot of 13 soft voxels learns to walk;
  an open-loop gait (`gait_policy`) covers about 40 voxels in an episode.
- Gallery: the soft walker.

## [0.2.0] - 2026-10-13

The first release. It contains what the roadmap planned for the versions
0.1 and 0.2; a version 0.1 was not released.

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
- Six experimental environments with the interface of `Evacuation`, each
  with tests, a scripted policy and an animation (see
  `docs/experimental.md`): `CounterFlow` (two groups of pedestrians in a
  corridor), `BrickWall` (an arcade game with state or pixel observations),
  `Billiards`, `Herding` (flock rules adapted from Strömbom et al. 2014),
  `Microswimmer` (the task of Colabrese et al. 2017) and `CellMigration`
  (after Zhang et al. 2020).
- `observation_space` and `action_space` on every environment
  (`envs.Discrete`, `envs.Continuous`).
- Brownian dynamics integrator `brownian` for particles without inertia.
- `Canvas.field` draws a scalar field behind the particles.
- Gallery animation of two DPD fluids that separate.
- Benchmark `benchmarks/baselines.py`: scripted and random policies in the
  experimental environments, with its reference output.

- Wrappers `particlegym.wrappers`: `GymnasiumEnv` and `GymnasiumVectorEnv`
  (batched with `jax.vmap`) for single-agent environments, with the registered
  ids `particlegym/Evacuation2D-v0`, `BrickWall-v0`, `Billiards-v0`,
  `Herding-v0` and `Microswimmer-v0`, and `PettingZooEnv` for the parallel
  multi-agent interface. Optional dependencies: `pip install
  "particlegym[gymnasium,pettingzoo]"`
  ([#5](https://github.com/ZhaojieC/particlegym/issues/5)).
- Baselines `particlegym.baselines`: PPO with a policy shared by all agents
  (`ppo`) and tabular Q-learning (`q_learning`), in pure JAX; several seeds
  train in one compiled program.
- Examples `examples/learning`: Q-learning of the microswimmer task after
  Colabrese et al. (2017), PPO on four tasks with scripted and random
  baselines, learning curves, and animations of a policy before and after
  training.
- Documentation page "Learning", with the results of PPO on the four tasks
  (five seeds each): it learns the microswimmer task and the brick wall, is
  slower than a simple rule in the evacuation task and fails at herding.
- `Microswimmer.render` can show several periods of the flow and the
  altitude that the swimmers have reached (`periods`, `start_row`).
- Notebook `examples/quickstart.ipynb`: a room evacuation, batched rooms, a
  game through the Gymnasium interface and a short PPO run.
- Benchmark `benchmarks/bench_envs.py`: control steps per second of the
  environments for batches of several sizes, with a table for one GPU and
  four CPU cores in the roadmap. Pixel observations of the brick wall do
  not compile on a GPU within 600 s yet.
- Validation: the equation of state and the temperature of the DPD fluid
  are compared with Groot & Warren (1997).
- Packaging: the README shown by the package index uses absolute addresses
  for pictures and links, the source distribution leaves out the animations,
  and CI builds and checks the package.
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
- Design decision D9: pedestrian evacuation is the validated flagship of
  version 1.0, and other families of environments on the same physics core
  are included as experimental environments. Before, they were to follow
  after 1.0. The roadmap is updated: the reproduction of the published
  evacuation results and the Verlet neighbour lists moved from 0.2 to 0.3.
