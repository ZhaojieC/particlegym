# Contributing to ParticleGym

Thank you for your interest in ParticleGym! Bug reports, new scenarios, new
physics models, documentation and benchmark results are all welcome.

## Development setup

```bash
git clone <repository-url>
cd particlegym
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

Run the checks before opening a pull request:

```bash
ruff check . && ruff format --check .
pytest
mkdocs build --strict
```

## How we work

All changes go through GitHub issues and pull requests, including changes by
the maintainers. This keeps design discussions in the open.

1. Open an issue describing the problem or proposal (skip this for typos).
2. Keep pull requests small and focused on one change.
3. Add tests. Every physics claim must have an automated validation test (see
   [`docs/validation.md`](docs/validation.md)). New force models need at least
   a check that forces equal the negative energy gradient, plus a test against a
   known analytical or published result.
4. Update the documentation and `CHANGELOG.md`.

## Code style

- Physics code is written in JAX with pure functions and static array shapes,
  so it can be compiled with `jax.jit` and batched with `jax.vmap`.
- Conservative interactions are written as energies; forces come from automatic
  differentiation. Dissipative forces are written explicitly.
- The same code must work in 2D and 3D.
- Formatting and linting use `ruff` (configured in `pyproject.toml`).

## AI-assisted contributions

You may use AI tools, but you are responsible for every line you submit.
Please disclose in the pull request which tools you used, for which parts, and
how you checked the result. Maintainers record this in
[`docs/ai-usage.md`](docs/ai-usage.md) so that it can be reported in
publications about the software.

## Code of conduct

This project follows the [Code of Conduct](CODE_OF_CONDUCT.md).
