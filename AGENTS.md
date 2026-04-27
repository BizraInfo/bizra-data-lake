# Repository Guidelines

## Project Structure & Module Organization
`core/` contains the primary Python runtime and domain packages. `tests/` mirrors that layout with focused suites under `tests/core/`, plus `tests/integration/`, `tests/e2e/`, and `tests/property_based/`. `bizra-omega/` is the Rust workspace. UI work is split between `frontend/` (Vite + React) and `dema-console/` (Next.js). Deployment and operator assets live in `deploy/`, `runtime/`, `config/`, `scripts/`, and `docs/`. Treat `evidence/`, `artifacts/`, and archived bundles as generated outputs unless the task explicitly targets them.

## Build, Test, and Development Commands
Set up Python with `pip install -e ".[dev]"` and activate `.venv/` before running repo-level commands.

- `make test` runs the fast proof-engine gate.
- `make test-all` runs the filtered Python suite and writes a log under `/data/bizra/logs`.
- `make lint` runs the pre-commit checks; `make check` and `make build` validate the Rust workspace in `bizra-omega/`.
- `cd bizra-omega && cargo test --workspace` runs Rust tests directly.
- `cd frontend && npm run ci` runs typecheck, lint, test, and build for the Vite app.
- `cd dema-console && npm run lint && npm run build` validates the Next.js console.

## Coding Style & Naming Conventions
Python uses 4-space indentation, type hints on public interfaces, and Black-compatible formatting with a line length of 88. Ruff enforces `E/F/W` rules, and imports are sorted with `isort`. Use `snake_case` for Python modules/functions/tests, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants. For Rust, keep `cargo fmt` and `cargo clippy -- -D warnings` clean. In React/TypeScript packages, follow existing component naming (`PascalCase`) and hook naming (`useSomething`).

## Testing Guidelines
Pytest is authoritative at the repo root. Name files `test_*.py`, classes `Test*`, and functions `test_*`. The default config skips `slow` tests; use markers such as `requires_ollama`, `requires_gpu`, and `requires_network` explicitly when needed. Python coverage is measured against `core/` with `fail_under = 65`. Add regression tests for behavior, API, contract, or governance changes. For UI changes, add or update Vitest coverage in `frontend/tests/`.

## Commit & Pull Request Guidelines
Recent history follows scoped Conventional Commits such as `feat(face): ...`, `fix(contracts): ...`, and `chore(smoke): ...`; keep that format. PRs should summarize the affected area, list verification commands run, and call out config, schema, or contract changes. Include screenshots for UI work and link the relevant issue or PR context when available.

## Agent Notes
More specific `AGENTS.md` files in subdirectories override this root guide. Check `bizra-omega/AGENTS.md` before editing that workspace.
