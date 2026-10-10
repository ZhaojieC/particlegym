# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [Unreleased]

### Added

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
