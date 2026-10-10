# ParticleGym

**Fast, physically validated 2D/3D particle-dynamics environments for reinforcement learning.**

> **Status: pre-alpha (v0.1.0.dev0).** The physics core and its validation
> tests are in place; the reinforcement-learning interface (Gymnasium,
> PettingZoo) is the next milestone. APIs may change. See the
> [roadmap](docs/roadmap.md).

ParticleGym simulates many interacting particles — pedestrians, active
colloids, cells, swarm robots — and lets reinforcement-learning agents control
some of them. It is built around three goals:

- **Physically consistent.** Agents choose what they *want* (a desired
  direction of motion); the motion itself always follows from forces and a
  proper integrator. Conservative forces are derived from energies by automatic
  differentiation, and every physical claim is checked by an automated test.
- **Fast.** Everything is written in [JAX](https://github.com/jax-ml/jax): one
  simulation step compiles to a single XLA program that runs on CPU or GPU, and
  environments can be batched with `jax.vmap`. On a 4-core CPU a step of a
  100-pedestrian crowd takes about 0.15 ms (see the
  [benchmarks](docs/roadmap.md#performance-plan)).
- **Simple.** Pure Python, two dependencies (`jax`, `numpy`), the same code for
  2D and 3D, and readable models with references.

## Features (physics core)

| Component | Available |
|---|---|
| Domain | 2D or 3D box, periodic or open along each axis |
| Neighbour search | brute force; cell list with overflow detection |
| Pair interactions | harmonic, WCA, social force (Helbing et al. 2000), tangential friction |
| Geometry | plates (segments/rectangles), spheres, boxes, rooms with real door openings |
| Integrators | velocity Verlet; BAOAB Langevin with a drift velocity (self-propulsion) |
| Exits | `alive` mask: particles that leave are frozen, shapes stay static |
| Output | extended XYZ for [OVITO](https://www.ovito.org/) and ASE |

## Installation

ParticleGym requires Python 3.12 or later.

```bash
git clone <repository-url>
cd particlegym
pip install -e .
```

For GPU support, install the matching JAX build first, following the
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html).

## Quick start: pedestrians leaving a room through a door

```python
import jax
import jax.numpy as jnp
import numpy as np

from particlegym import physics as pp

# A 10 m x 10 m room with a 1.2 m wide door in the right wall.
box = pp.Box(lo=(-1.0, -1.0), hi=(13.0, 11.0))
walls = pp.box_walls((0, 0), (10, 10), openings=[pp.Opening("x+", (4.4, 5.6))])

force_field = pp.ForceField(
    box,
    pair=[pp.SocialForce(), pp.TangentialFriction()],
    walls=[pp.Walls(walls, pp.SocialForce(), friction=2.4e5)],
)

rng = np.random.default_rng(0)
radius = rng.uniform(0.25, 0.30, size=50)
state = pp.create_state(
    pp.random_positions(rng, (0, 0), (10, 10), radius), mass=80.0, radius=radius
)
state, _ = pp.init_forces(state, force_field)

door = jnp.array([10.0, 5.0])


@jax.jit
def step(state):
    # Desired direction: towards the door, then straight out.
    to_door = door - state.pos
    direction = jnp.where(state.pos[:, :1] < 10.0, to_door, jnp.array([1.0, 0.0]))
    direction = direction / jnp.linalg.norm(direction, axis=1, keepdims=True)
    state, info = pp.langevin_baoab(
        state, force_field, dt=0.005, gamma=1 / 0.5, drift=1.5 * direction
    )
    # Pedestrians that have walked 1 m past the door have left the room.
    return state._replace(alive=state.alive & (state.pos[:, 0] < 11.0)), info


for _ in range(20_000):  # 100 s
    state, info = step(state)
print("still inside:", int(state.alive.sum()))
```

The driving term of the social force model, `m (v0 e - v) / tau`, is exactly
the friction term of Langevin dynamics with `gamma = 1 / tau` and drift
velocity `u = v0 e`; see [Physics models](docs/physics.md). In the upcoming
environment layer, the RL action sets the desired direction `e`.

More examples are in [`examples/`](examples).

## Validation

The test suite checks, among others: energy conservation with the expected
`dt^2` scaling, momentum conservation, forces equal to minus the energy
gradient, cell list against brute force, the Langevin temperature, the
Einstein diffusion law against the exact Ornstein-Uhlenbeck result, the
published social-force law, and that pedestrians leave a room only through
the door. See [Validation](docs/validation.md).

```bash
pip install -e ".[test]"
pytest
```

## Background

ParticleGym grows out of the particle-dynamics environment used in

> Y. Zhang, Z. Chai, G. Lykotrafitis. *Deep reinforcement learning with a
> particle dynamics environment applied to emergency evacuation of a room with
> obstacles.* Physica A 571, 125845 (2021).
> [doi:10.1016/j.physa.2021.125845](https://doi.org/10.1016/j.physa.2021.125845)

It generalises that environment to 3D, to other particle systems, and to
modern, fast, tested software.

## Contributing and citing

Contributions are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md). Citation
metadata is in [CITATION.cff](CITATION.cff). ParticleGym is released under
the [MIT license](LICENSE).
