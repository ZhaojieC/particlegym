# Roadmap

ParticleGym is developed in the open towards a 1.0 release and a paper in the
[Journal of Open Source Software](https://joss.theoj.org/) (JOSS). Every
milestone is a tagged release with a Zenodo DOI.

## Milestones

Pedestrian evacuation is the validated flagship of version 1.0; other
families of environments are included as experimental
([design decision D9](design.md#d9-evacuation-is-the-validated-flagship-other-families-are-experimental)).

| Version | Content | Target |
|---|---|---|
| **0.1** | Physics core, validation tests, packaging, CI, documentation site | month 1–2 |
| **0.2** | Environment layer: functional `reset`/`step`, actions as desired directions, exits, 2D and 3D rooms; Gymnasium and PettingZoo wrappers; pictures and animations; more physics (external forces, flows, moving shapes, contact damping, active particles, bonds, DPD, scalar fields); six experimental environments; PPO and Q-learning baselines | month 2–3 |
| **0.3** | `Evacuation2D` reproducing Zhang, Chai & Lykotrafitis (2021) (four exits; three exits and two obstacles); Verlet neighbour lists with skin; crowd validation against experiments (speed-density relation, bottleneck flow) and the faster-is-slower effect; calibrated parameter sets; `Bottleneck` scenario; a DQN baseline and a Stable-Baselines3 example | month 3–4 |
| **0.4** | Multi-agent evacuation (shared and independent policies), rooms with complex obstacles, multi-storey buildings in 3D; GPU benchmarks | month 4–5 |
| **0.5** | Tutorials (notebooks), API reference, performance guide, external users | month 5–6 |
| **1.0** | Stable API, JOSS submission | month 6–7 |

Two items planned for 0.2 moved to 0.3: the reproduction of the evacuation
results of Zhang, Chai & Lykotrafitis (2021) and the Verlet neighbour lists.
The experimental environments and the learning baselines came earlier than
planned.

Experimental environments leave that status one by one, when their results
have been compared with published ones (see
[Planned validation](validation.md#planned-validation)). After 1.0:
microswimmers and active colloids in 3D (orientations, rotational diffusion)
and DPD with walls and flows.

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

On one NVIDIA L40S GPU, the PPO baseline trained five seeds at once at 14.7
(microswimmer), 11.0 (evacuation with 40 pedestrians), 6.1 (brick wall) and
1.4 (herding) million agent steps per second, with the physics substeps, the
updates of the networks and the compilation included (see
[Learning](learning.md#results-of-ppo)). `benchmarks/bench_envs.py` measures
the environments alone, with random actions: control steps per second of all
environments of a batch together, on one NVIDIA L40S GPU and on four cores of
an AMD EPYC 9554 CPU (JAX 0.11.2, float32; the output is in
`benchmarks/bench_envs.txt`, rounded here to two digits). A control step
contains the number of physics steps of the column "Substeps" and the
computation of the observations.

| Environment | Substeps | GPU, batch 1024 | GPU, batch 16384 | CPU, batch 1 | CPU, batch 64 |
|---|---|---|---|---|---|
| Evacuation, 1 pedestrian | 20 | 7,200,000 | 68,000,000 | 39,000 | 310,000 |
| Evacuation, 40 pedestrians, 4 doors | 20 | 1,000,000 | 550,000 | 3,600 | 2,300 |
| Counter-flow, 80 pedestrians | 20 | 360,000 | 100,000 | 1,100 | 2,000 |
| Brick wall, state observations | 20 | 3,500,000 | 28,000,000 | 47,000 | 37,000 |
| Brick wall, 84 × 84 pixels | 20 | 3,100,000 | 8,300,000 | 15,000 | 26,000 |
| Herding, 40 sheep | 20 | 1,800,000 | 3,000,000 | 7,300 | 5,400 |
| Microswimmer | 10 | 11,000,000 | 120,000,000 | 63,000 | 550,000 |
| Cell migration, 9 cells | 50 | 10,000 | 9,900 | 330 | 1,300 |
| Billiards, 15 balls | one shot | 2,100 | 1,200 | 22 | 5 |

- With several agents in an environment, the number of agent steps is larger
  by the number of agents: 40 times for the room with four doors.
- A larger batch does not always help: for the crowds and the cell
  migration, the throughput does not rise from 1024 to 16384 environments.
- A step of the billiards game is a whole shot, and in a batch it lasts until
  the balls of the slowest table have stopped.
- Pixel observations cost about three times as much as state observations
  for one environment on the CPU and for 16384 environments on the GPU (1.4
  times for 64 on the CPU, 1.1 times for 1024 on the GPU). The row of the
  pixels comes from a later run than the other rows, after the drawing code
  was changed; the row of the state observations agreed within 4 % between
  the two runs. In version 0.2.0 every pixel was tested against every
  brick: 1,900 control steps per second for one environment on the CPU, and
  on the GPU the code was still being compiled after 600 s.
- The numbers hold within about a factor of two. A run of an earlier version
  of the script, which did less work in a step, gave values up to two times
  lower on another node with the same GPU and CPU; the cause was not
  investigated.

Planned improvements:

- Verlet neighbour lists with a skin, rebuilt only when needed, to remove most
  empty candidate slots and amortise the cell-list construction.
- Images of many boxes that move: `viz.box_coverage` tests every pixel
  against every box, and with 84 × 84 pixels and 40 boxes its first call on
  a GPU did not finish within 300 s. The brick wall avoids it with a table,
  which needs boxes that stay in place.
- A comparison of batched environments on a GPU with VMAS and the JAX
  version of MPE on a comparable 2D task and the same hardware. This has not
  been done yet.

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
| JuPedSim | pedestrian dynamics simulator | not an RL environment |
| Vadere | crowd simulation framework (Java) | not an RL environment |
| CrowdNav | RL for a robot navigating among simulated pedestrians | the pedestrians are not the learning agents; 2D |
| VMAS | vectorised 2D physics for multi-agent RL (PyTorch) | 2D only; rigid-body contacts; no pedestrian or soft-matter force models, no thermostat |
| PettingZoo MPE | classic 2D particle environments | 2D, CPU, simplified physics |
| SwarmRL | RL for active colloids | focused on colloids and microswimmers |
| Brax, MuJoCo MJX, Genesis, Isaac Lab | 3D rigid-body robotics | articulated robots, not many-body particle systems |
| JAX-MD | differentiable molecular dynamics | no agents, observations, rewards or RL interface |
