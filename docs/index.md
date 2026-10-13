# ParticleGym

**Fast, physically validated 2D/3D particle-dynamics environments for
reinforcement learning.**

!!! warning "Pre-alpha"
    Version 0.2.0 provides the physics core with its validation tests,
    a room-evacuation environment, six experimental environments, Gymnasium
    and PettingZoo wrappers and two learning baselines. Interfaces may
    change; see the [roadmap](roadmap.md).

ParticleGym simulates many interacting particles — pedestrians, active
colloids, cells, swarm robots — and lets reinforcement-learning agents control
some of them. Pedestrian evacuation is the first application and the one
that is validated against published work; games, herding, microswimmers and
cell migration are included as experimental environments.

- **Physically consistent.** Agents choose a desired direction; the motion
  follows from forces and a proper integrator. Conservative forces come from
  energies by automatic differentiation, and each model is checked by tests.
- **Fast.** Written in JAX: steps compile to XLA for CPU and GPU, and
  environments batch with `jax.vmap`.
- **Simple.** Two dependencies, the same code for 2D and 3D, readable models.

![10,000 pedestrians leave a hall](assets/gallery/crowd_10000.gif)

More animations are in the [gallery](gallery.md).

## Installation

```bash
git clone https://github.com/ZhaojieC/particlegym
cd particlegym
pip install -e ".[viz]"
```

## A first simulation

A Lennard-Jones (WCA) fluid in a periodic 3D box, thermostatted with Langevin
dynamics:

```python
import jax
from particlegym import physics as pp

box = pp.Box((0, 0, 0), (10, 10, 10), periodic=True)
state = pp.create_state(pp.lattice_positions(box.lo, box.hi, 500), radius=0.5)
field = pp.ForceField(box, pair=[pp.WCA()], neighbors=pp.CellList(cutoff=1.2))
field.check(state)
state, _ = pp.init_forces(state, field)


def body(state, key):
    state, info = pp.langevin_baoab(state, field, 0.005, gamma=1.0, kT=1.0, key=key)
    return state, pp.temperature(state)


keys = jax.random.split(jax.random.key(0), 2000)
state, temperature = jax.jit(lambda s: jax.lax.scan(body, s, keys))(state)
print(temperature[-500:].mean())  # close to 1.0
```

The quick start in the project README shows pedestrians leaving a room through
a door. The notebook
[`examples/quickstart.ipynb`](https://github.com/ZhaojieC/particlegym/blob/main/examples/quickstart.ipynb)
goes from a room evacuation to a game and a trained policy in four steps, and
the `examples/` directory contains complete scripts.

## Documentation

- [Environments](environments.md): the interface and the evacuation
  environment; [experimental environments](experimental.md): games, herding,
  microswimmers and cell migration.
- [Learning](learning.md): Gymnasium and PettingZoo wrappers, the PPO and
  Q-learning baselines, and first results.
- [Design](design.md): principles, architecture and design decisions.
- [Physics models](physics.md): equations, parameters, units and references.
- [Validation](validation.md): what is tested and the results.
- [Roadmap](roadmap.md): milestones and the path to a JOSS paper.
- [AI usage log](ai-usage.md): record of AI assistance.
