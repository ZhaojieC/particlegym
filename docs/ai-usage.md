# AI usage log

JOSS asks every paper to disclose whether and how generative AI was used,
and how its output was checked. This log records AI assistance as it happens,
so that the disclosure can be written from facts. Add an entry for every pull
request that used AI tools.

| Date | Tool | Scope | Human review and validation |
|---|---|---|---|
| 2026-10-10 | Claude Code (Anthropic) | Initial project scaffold, physics core, test suite, documentation drafts, examples and benchmark script | Pending: the maintainers review every module before v0.1.0. The physics is checked by the automated validation tests in [Validation](validation.md). |
| 2026-10-11 | Claude Code (Anthropic) | Evacuation environment (`particlegym.envs`), its tests, the Environments page and an example ([#3](https://github.com/ZhaojieC/particlegym/issues/3)) | Pending review of the pull request by the maintainer. |

## How to write an entry

- **Tool**: name and version of the tool or model.
- **Scope**: which files or features, and the kind of help (generated code,
  refactoring, documentation, review).
- **Review**: who reviewed it and how correctness was verified (tests added,
  results compared with references, manual derivation).
