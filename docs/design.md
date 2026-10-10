# Design

This document records the design of ParticleGym: what it is for, the
principles it follows, and the decisions taken so far, with the reasons and
the alternatives that were considered. It is a living document. Decisions that
are still open are listed at the end and marked **open**.

## Purpose

Reinforcement learning (RL) is increasingly used to study how agents should
move through crowded, physical environments: pedestrians evacuating a
building, active colloids navigating a channel, cells migrating collectively,
robots in a swarm. Researchers in these fields need environments that are

1. **physically consistent**, so that the learned behaviour is meaningful for
   the physical system and not an artefact of the simulator;
2. **fast**, because RL needs millions of environment steps; and
3. **simple** to install, read, modify and extend.

General RL physics engines target rigid-body robotics, and particle-based RL
environments are usually two-dimensional with simplified contact physics.
Molecular-dynamics and pedestrian-dynamics simulators have the right physics
but no RL interface. ParticleGym aims to fill this gap: many-body particle
dynamics with soft-matter and pedestrian force models, in 2D and 3D, with a
standard RL interface and automated physical validation.

ParticleGym grows out of the environment of Zhang, Chai & Lykotrafitis,
*Physica A* 571, 125845 (2021), which coupled a social force model to deep
Q-learning for room evacuation. Pedestrian evacuation is the first
application domain and the focus of version 1.0 (decision D9); the physics
core is general and will serve other particle systems later.

## Goals and non-goals

Goals:

- Interacting particles in 2D and 3D, with pair forces, walls and obstacles,
  and thermal noise.
- Agents control the *intent* of selected particles (a desired velocity).
- Standard RL interfaces (Gymnasium and PettingZoo, planned for v0.2) and a
  functional JAX interface for end-to-end compiled training loops.
- Every physical claim backed by an automated test.
- Throughput high enough for RL: thousands of batched environments on a GPU.

Non-goals:

- Rigid bodies with rotational dynamics, joints or articulated robots (use a
  robotics engine).
- Production molecular dynamics (use LAMMPS, HOOMD-blue, GROMACS).
- Calibrated, regulation-grade evacuation analysis.

## Design principles

**P1 — Agents choose intent, physics produces motion.** An action never sets
positions or velocities directly. It sets a desired velocity (the drift
velocity of the Langevin equation, below), and the motion follows from forces
and the integrator. This keeps learned policies physically meaningful.

**P2 — Conservative forces come from energies.** Pair and wall interactions are
written as energy functions; forces are obtained by automatic differentiation.
Forces and energies are therefore consistent by construction, Newton's third
law holds, and energy conservation can be tested. Velocity-dependent
(dissipative) forces are written explicitly and kept separate.

**P3 — One well-founded integrator for driven, damped, noisy dynamics.** The
driving term of the social force model, \(m(v_0 \mathbf e - \mathbf v)/\tau\),
is the friction term of Langevin dynamics with friction rate
\(\gamma = 1/\tau\) and drift velocity \(\mathbf u = v_0 \mathbf e\). Passive
Brownian particles have \(\mathbf u = 0\); active particles steer
\(\mathbf u\). A single BAOAB Langevin integrator therefore covers pedestrians,
colloids and active particles, and integrates the relaxation term exactly.

**P4 — Pure functions and static shapes.** All physics functions are pure and
all arrays have shapes fixed at compile time, so a step compiles with
`jax.jit`, loops compile with `jax.lax.scan`, environments batch with
`jax.vmap`, and gradients are available. Particles that leave through an
exit are masked out (`alive = False`) instead of being removed.

**P5 — Dimension-generic code.** The same code handles 2D and 3D. 2D scenarios
are genuinely two-dimensional rather than 3D scenarios with a frozen axis.

**P6 — Validation is part of the code.** Every model comes with tests against
analytical or published results (see [Validation](validation.md)), run in CI.

**P7 — Few dependencies.** The core depends only on JAX and NumPy. RL
libraries are optional extras.

## Architecture

ParticleGym is organised in layers; each layer depends only on the layers
below it.

| Layer | Contents | Status |
|---|---|---|
| Physics core (`particlegym.physics`) | state, box, neighbour search, potentials, geometry, integrators, observables | v0.1 |
| Environment core | functional `reset`/`step`, actions, observations, rewards, exits | planned v0.2 |
| Wrappers | Gymnasium, Gymnasium vector, PettingZoo parallel | planned v0.2 |
| Scenarios | room evacuation (incl. Physica A 2021), bottleneck, multi-agent evacuation, rooms with obstacles, multi-storey buildings | planned v0.2–0.4 |

Data flow in one physics step: the integrator updates velocities and
positions, asks the `ForceField` for forces, and the force field combines

- pair terms, evaluated on candidate pairs from the neighbour search, and
- wall terms, evaluated from distance functions of the geometry.

## Decisions

Each decision lists the choice, the reasons, and the alternatives considered.

### D1 — JAX as the only backend

*Choice.* Write the core in JAX.

*Reasons.* `jit` compiles a whole step into one XLA program for CPU or GPU;
`vmap` batches environments without code changes; `grad` gives forces from
energies (P2) and enables differentiable-physics research; explicit random
keys make runs reproducible.

*Alternatives.* PyTorch (used by VMAS and TorchRL) is more familiar to many RL
users but has no equally simple function-level compilation and batching.
NumPy with Numba is fast on CPU but has no GPU batching or autodiff. Supporting
two backends would double the maintenance work.

### D2 — Static shapes with an alive mask

*Choice.* The number of particles is fixed per environment; leaving particles
are frozen and excluded from all interactions.

*Reasons.* JAX compiles for fixed shapes. A mask keeps `jit`, `scan` and `vmap`
working while particles exit.

*Alternative.* Dynamic removal would force recompilation or host-side
bookkeeping on every exit.

### D3 — Forces from energies by automatic differentiation

*Choice.* A pair potential is a scalar function \(\phi(r, \sigma)\) of the
distance \(r\) and contact distance \(\sigma = r_i + r_j\). The force
magnitude is \(-\partial\phi/\partial r\), obtained with `jax.grad`; forces
are gathered per particle over its candidate pairs.

*Reasons.* New potentials need one line of physics and no hand-written
derivative; consistency between force and energy is guaranteed (and tested).
Gathering per particle avoids scatter operations, which are slow on
accelerators.

*Alternative.* Hand-written force laws are error-prone (the predecessor code
had a discontinuous and non-monotonic wall force).

### D4 — Geometry from distance functions; walls reuse pair potentials

*Choice.* Shapes provide the distance from a point to their surface (plates,
spheres, boxes). A wall interaction evaluates a pair potential at
\((d, r_i)\). Rooms are built from plates with real openings (doors).

*Reasons.* One mechanism covers all shapes; any pair potential can serve as a
wall potential; door jambs behave like corners automatically. Doors are true
openings, so clogging at bottlenecks can emerge. (In the predecessor code an
exit was an absorbing region on a closed wall.)

*Alternative.* Special-cased wall code per shape and per potential.

### D5 — BAOAB Langevin with drift as the main integrator

*Choice.* Use the BAOAB splitting (Leimkuhler & Matthews, 2013) with the O
step solved exactly, extended by a drift velocity. Velocity Verlet is kept for
energy-conserving runs (it is the \(\gamma = 0\) limit).

*Reasons.* See P3. The exact O step is stable for any \(\gamma\,\Delta t\), so
the overdamped limit (Brownian dynamics) is reached without changing
integrators. BAOAB has excellent configurational sampling accuracy.

*Alternative.* Explicit Euler for the self-driving force (as in many crowd
codes) is only first-order accurate and unstable for large \(\Delta t/\tau\).

### D6 — Neighbour search with fixed-capacity cell lists

*Choice.* Two strategies with the same interface: brute force (dense, no
gathers) and a cell list with a fixed number of slots per cell and an overflow
flag.

*Reasons.* Fixed capacity gives static shapes (D2). Overflow cannot raise an
error inside compiled code, so it is reported and checked by the caller.

*Planned improvement.* A compact Verlet neighbour list with a skin, rebuilt
only when particles have moved far enough, would remove most empty slots and
amortise the cell-list construction (see [Roadmap](roadmap.md)).

### D7 — Coordinate-first array layout inside kernels

*Choice.* The public API uses (N, d) arrays, but pair and wall kernels work on
coordinate-first arrays, (d, N, K) and (d, M, N).

*Reasons.* XLA vectorises over the innermost array axis. With (N, K, d) that
axis has length 2 or 3; with the coordinate axis first, it is the long pair
axis. On CPU this made the dense pair and wall kernels about 4 to 5 times
faster and the cell-list kernels 1.3 to 1.8 times faster.

*Rejected.* Packing all per-particle data into one array to gather it at once
was 10 times slower on CPU: the compiler repeated the gather for every use.

### D8 — Separate control and physics time steps (planned for the environment layer)

*Choice.* One RL step will advance the physics by several integrator steps.

*Reasons.* The physics time step is set by numerical stability (stiff
contacts need milliseconds), while decisions of pedestrians or swimmers happen
on a much slower time scale. Coupling them would either make physics unstable
or RL needlessly expensive.

### D9 — Pedestrian evacuation first

*Choice.* Version 1.0 concentrates on pedestrian evacuation: rooms with real
doors and obstacles, bottlenecks, single- and multi-agent evacuation, and
multi-storey buildings. Active colloids and cell migration follow after 1.0,
on the same physics core.

*Reasons.* It builds directly on the authors' published work (Physica A 2021),
which gives a first reproduction target. Pedestrian dynamics has established
experimental benchmarks (speed-density relations, bottleneck flows) for
validation. And a focused scope gets a complete, well-documented release to
users sooner.

*Alternative.* Covering crowds, active matter and cells at once would spread
the validation and documentation work thinly before any domain is complete.

## Planned environment interface (v0.2 sketch)

```python
env = particlegym.make("Evacuation2D-v0")  # Gymnasium-compatible
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(action)

# Functional interface for compiled training loops
params = particlegym.envs.Evacuation2D.default_params()
obs, state = reset(key, params)
obs, state, reward, done, info = step(key, state, action, params)
```

Actions map to desired directions (discrete direction sets or continuous
vectors). Observations are composed from modules (own state, target
direction, local density on a grid, nearest neighbours, ray casts). Rewards are
composed from terms (time penalty, exit reward, collision penalty).

## Open questions

- **open** — How to represent multi-storey buildings: floors as separate 2D
  domains linked by stairs, or one 3D domain with gravity and stair geometry.
- **open** — Observation defaults for evacuation: own state only (as in the
  Physica A 2021 setup), or also local density and nearest neighbours.
- **open** — Default floating-point precision for environments (float32 for
  speed; float64 is used in the validation tests).
- **open** — Whether to add dissipative particle dynamics (DPD) with
  hydrodynamic interactions after 1.0.
