# Repository Guidelines

## Project Structure & Module Organization
- Data lake tiers live at the repo root: `00_INTAKE/` (drop zone), `01_RAW/` (immutable originals), `02_PROCESSED/` (typed outputs), `03_INDEXED/` (embeddings/graph), `04_GOLD/` (curated assets), `99_QUARANTINE/` (duplicates/corruption). Use these paths rather than creating new top-level tiers.
- Python source is a mix of root-level modules (e.g., `bizra_*.py`), `core/`, and `src/`. Keep new modules close to related systems and avoid duplicating similar utilities.
- Supporting assets and docs: `config/`, `docs/`, `static/`, `models/`, `scripts/`, plus many architecture and roadmap markdown files at the root.

## Build, Test, and Development Commands
- Install dependencies (editable, dev): `pip install -e ".[dev]"`
- Install pinned environment: `pip install -r requirements.lock`
- Run intake processing once: `.\DataLakeProcessor.ps1 -ProcessOnce`
- Start continuous monitoring: `.\DataLakeProcessor.ps1 -Watch`
- Cloud ingestion dry run: `.\CloudIngestion.ps1 -DryRun`

## Coding Style & Naming Conventions
- Python 3.9+; 4-space indentation; prefer type hints on public functions.
- Formatting: `black` with line length 88; keep imports organized with `isort`.
- Naming: `snake_case` for files/functions, `PascalCase` for classes, `UPPER_SNAKE` for constants.

## Testing Guidelines
- Framework: `pytest` (see `pytest.ini`).
- Tests live in `tests/` and root-level `test_*.py` files.
- Run all tests: `pytest`
- Use markers when needed: `pytest -m "integration"`, `pytest -m "not slow"`; special markers include `requires_gpu` and `requires_ollama`.

## Commit & Pull Request Guidelines
- Current history uses Conventional Commit style (e.g., `feat: Initial commit of BIZRA Sovereign codebase`). Follow `<type>: <short summary>` where possible.
- PRs should include: a concise summary, affected areas (e.g., `02_PROCESSED/`, `core/`), tests run, and any data/migration notes. Add screenshots or metrics for dashboards/visualizations.

## Security & Data Handling
- Do not modify files in `01_RAW/`; treat it as immutable provenance.
- Use `00_INTAKE/` for new data so the pipeline can track and deduplicate.
- Treat `99_QUARANTINE/` as read-only unless you are resolving a known duplicate issue.

## References
- Architecture: `ARCHITECTURE.md`, `ARCHITECTURE_LOCK.md`
- Pipeline ops: `QUICK-START.md`, `DataLakeProcessor.ps1`, `CloudIngestion.ps1`
- Agent roles: `AGENT_ROLES.md`, `CLAUDE.md`
