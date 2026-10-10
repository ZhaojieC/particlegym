# Roadmap

ParticleGym is developed in the open towards a 1.0 release and a paper in the
[Journal of Open Source Software](https://joss.theoj.org/) (JOSS). Every
milestone is a tagged release with a Zenodo DOI.

## Milestones

| Version | Content | Target |
|---|---|---|
| **0.1** | Physics core, validation tests, packaging, CI, documentation site | month 1–2 |
| **0.2** | Environment layer: functional `reset`/`step`, actions as desired directions, observation and reward modules, exits; Gymnasium, Gymnasium vector and PettingZoo wrappers; Verlet neighbour lists with skin | month 2–3 |
| **0.3** | Scenarios: `Evacuation2D` (reproducing Zhang, Chai & Lykotrafitis 2021), `Bottleneck` with real doors; RL baselines (Stable-Baselines3 PPO/DQN, pure-JAX PPO); benchmarks | month 3–4 |
| **0.4** | 3D scenarios: `ActiveNavigation3D` (self-propelled particles with orientation and noise), `Swarm3D` (multi-agent), `CellMigration3D` (chemotaxis); multi-agent baselines | month 4–5 |
| **0.5** | Tutorials (notebooks), API reference, performance guide, external users | month 5–6 |
| **1.0** | Stable API, JOSS submission | month 6–7 |

Later (1.x): dissipative particle dynamics with hydrodynamic interactions,
multi-floor buildings, calibrated pedestrian parameters.

## Performance plan

Measured with `benchmarks/bench_step.py` on a 4-core CPU (JAX 0.11.2,
float32), v0.1.0.dev0:

| System | Particles | Neighbour search | µs per step | Particle-steps per second |
|---|---|---|---|---|
| 2D crowd (social force + friction, walls, door) | 100 | brute force | 147 | 680,000 |
| 2D crowd, 64 batched environments | 64 × 100 | brute force | 14,959 | 430,000 |
| 2D crowd | 1,000 | cell list | 4,408 | 230,000 |
| 2D crowd | 10,000 | cell list | 31,389 | 320,000 |
| 3D WCA fluid | 1,000 | cell list | 2,963 | 340,000 |
| 3D WCA fluid | 10,000 | cell list | 20,296 | 490,000 |

The predecessor code runs at about 7,500 particle-steps per second on the same
machine, so a step is 30 to 90 times faster. Two caveats:

- The default contact model of Helbing et al. (2000) is stiff and needs time
  steps of about 5 ms. The predecessor used 0.1 s steps with much softer,
  truncated forces. Per simulated second, the gain is therefore about 4 to 5
  times, in exchange for more realistic contacts; softer parameters allow
  larger steps.
- Batching with `jax.vmap` does not help on a CPU; it is meant for GPUs.

Planned improvements:

- Verlet neighbour lists with a skin, rebuilt only when needed, to remove most
  empty candidate slots and amortise the cell-list construction.
- GPU benchmarks of batched environments, compared with VMAS and PettingZoo
  MPE on comparable 2D tasks.

## JOSS readiness checklist

JOSS (2026 criteria) requires at least six months of public, active
development with evidence of research use, and a paper of 750–1750 words with
the sections Summary, Statement of need, State of the field, Software design,
Research impact statement and AI usage disclosure.

- [x] OSI-approved license (MIT)
- [x] Automated tests and continuous integration
- [x] Contribution guidelines and code of conduct
- [x] Design document ([Design](design.md)) as the basis of the Software design section
- [x] AI usage log ([AI usage](ai-usage.md)) as the basis of the AI usage disclosure
- [ ] Public repository with open development (issues, pull requests, releases) for at least six months
- [ ] Installable from PyPI
- [ ] Complete API documentation and tutorials
- [ ] Tagged releases archived on Zenodo
- [ ] Research use: at least one study, preprint or thesis chapter using ParticleGym
- [ ] External users or integrations (for example the Gymnasium third-party environment list)
- [ ] Paper (`paper/paper.md`, `paper/paper.bib`)

## State of the field (for the paper)

| Software | Scope | Difference |
|---|---|---|
| VMAS | vectorised 2D physics for multi-agent RL (PyTorch) | 2D only; rigid-body contacts; no thermostat or soft-matter and pedestrian force models |
| PettingZoo MPE | classic 2D particle environments | 2D, CPU, simplified physics |
| SwarmRL | RL for active colloids | focused on colloids and microswimmers |
| JuPedSim | pedestrian dynamics simulator | not an RL environment |
| Brax, MuJoCo MJX, Genesis, Isaac Lab | 3D rigid-body robotics | articulated robots, not many-body particle systems |
| JAX-MD | differentiable molecular dynamics | no agents, observations, rewards or RL interface |
