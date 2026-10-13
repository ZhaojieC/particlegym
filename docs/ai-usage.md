# AI usage log

JOSS asks every paper to disclose whether and how generative AI was used,
and how its output was checked. This log records AI assistance as it happens,
so that the disclosure can be written from facts. Add an entry for every pull
request that used AI tools.

| Date | Tool | Scope | Human review and validation |
|---|---|---|---|
| 2026-10-10 | Claude Code (Anthropic) | Initial project scaffold, physics core, test suite, documentation drafts, examples and benchmark script | Pending: review of every module by the maintainers. The physics is checked by the automated validation tests in [Validation](validation.md). |
| 2026-10-11 | Claude Code (Anthropic) | Evacuation environment (`particlegym.envs`), its tests, the Environments page and an example ([#3](https://github.com/ZhaojieC/particlegym/issues/3)) | Pending review of the pull request by the maintainer. |
| 2026-10-12 | Claude Code (Anthropic) | 3D rooms in the evacuation environment, their tests, documentation and an example; faster distance to 3D plates | Pending review by the maintainer. Checked by tests that compare the 3D room with the 2D room and the new plate distance with the previous one. |
| 2026-10-12 | Claude Code (Anthropic) | Drawing module `particlegym.viz`, gallery scripts and animations, README front page, gallery page, documentation workflow | Pending review by the maintainer. Tests check that discs, shapes and pedestrians are drawn at their positions and with their areas; every animation is made by a script from a simulation with a fixed seed. |
| 2026-10-12 | Claude Code (Anthropic) | Physics capabilities: external forces, flow fields, moving and removable shapes, contact damping, active particles, bonded terms, the DPD thermostat, scalar fields and species; their tests, documentation and a DPD benchmark | Pending review by the maintainer. Every capability is tested against an analytical result (see [Validation](validation.md)). The DPD equation of state has not yet been compared with the published coefficient. |
| 2026-10-12 | Claude Code (Anthropic) | Experimental environments (counter-flow, brick wall, billiards, herding, microswimmer, cell migration), spaces, the Brownian integrator, their tests, gallery scripts, animations, documentation and a baseline benchmark | Pending review by the maintainer. The mechanics are tested against the equations of each model; the papers behind three of the models were read in the original, and the documentation states what was taken from them and what was chosen freely. No published result is reproduced in numbers yet. |
| 2026-10-13 | Claude Code (Anthropic) | Gymnasium and PettingZoo wrappers, the PPO and Q-learning baselines, the learning examples, their tests and documentation | Pending review by the maintainer. The wrappers pass the checkers of the two libraries; PPO is tested on the cart-pole task (largest return reached by three seeds) and its advantage estimates against a direct sum; Q-learning against the known values of a small problem. The microswimmer result agrees with Colabrese et al. (2017) in direction only. |
| 2026-10-13 | Claude Code (Anthropic) | Training runs of PPO on four tasks (five seeds each, one GPU), their figures, animations and the results section of the Learning page | Pending review by the maintainer. Every number is taken from the outputs in `benchmarks/learning`; the results include the task that PPO does not learn (herding) and the one in which it is slower than a simple rule (evacuation). One run per task; the settings were not tuned. |
| 2026-10-13 | Claude Code (Anthropic) | Preparation of version 0.2.0: design decision D9 and the roadmap, the quick-start notebook, the benchmark of the environments, packaging, the comparison of the DPD fluid with Groot & Warren (1997) | Pending review by the maintainer. The benchmark numbers are the recorded output of the scripts in `benchmarks/`; the comparison with the paper uses its Eq. 16 and the points of its Fig. 4, read from the scanned page. |

## How to write an entry

- **Tool**: name and version of the tool or model.
- **Scope**: which files or features, and the kind of help (generated code,
  refactoring, documentation, review).
- **Review**: who reviewed it and how correctness was verified (tests added,
  results compared with references, manual derivation).
