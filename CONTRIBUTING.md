# Contributing to BIZRA

Thank you for your interest in contributing to BIZRA. Every contribution matters.

## Getting Started

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes
4. Run tests and linters (see below)
5. Submit a pull request

## Development Setup

### Python

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
pip install -e ".[dev]"
```

### Rust

```bash
cd bizra-omega
cargo build --workspace
cargo test --workspace
```

## Code Standards

### Quality Thresholds

All code must satisfy constitutional constraints defined in `core/integration/constants.py`:

| Threshold | Value | Purpose |
|-----------|-------|---------|
| Ihsan (Excellence) | >= 0.95 | Minimum quality for any output |
| SNR (Signal Quality) | >= 0.85 | Information quality filter |
| ADL (Justice) Gini | <= 0.40 | Resource distribution fairness |

### Linting (Python)

CI enforces all of these on `core/`:

```bash
ruff check core/                           # Fast linter (primary)
black --check core/                        # Formatting
isort --check-only core/                   # Import order
mypy core/ --ignore-missing-imports        # Type checking
```

### Linting (Rust)

```bash
cd bizra-omega
cargo fmt --all -- --check                 # Formatting
cargo clippy --workspace --all-targets -- -D warnings  # Zero warnings
```

### Style

- **Python**: Black formatter, isort for imports, ruff for linting
- **Rust**: `cargo fmt` and `cargo clippy` (zero warnings enforced)
- **Commits**: Conventional commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `ci:`)

### Security

- Never hardcode secrets or API keys
- Use environment variables for all credentials
- Validate all external input
- No `eval()` or equivalent

## Testing

### Run Tests Before Submitting

```bash
# Python — fast local gate (recommended before every commit)
pytest -q tests/core/sovereign/test_runtime_types.py --capture=no
pytest -q tests/core/proof_engine/test_receipt.py --capture=no
pytest -q tests/core/sovereign/test_api_metrics.py --capture=no

# Python — full suite (exclude GPU/network tests)
pytest tests/ -m "not requires_ollama and not requires_gpu and not slow"

# Rust
cd bizra-omega && cargo test --workspace
```

### Test Markers

| Marker | Description |
|--------|-------------|
| `@pytest.mark.slow` | Long-running tests (>30s) |
| `@pytest.mark.integration` | Requires external services |
| `@pytest.mark.requires_ollama` | Requires Ollama running |
| `@pytest.mark.requires_gpu` | Requires CUDA GPU |
| `@pytest.mark.requires_network` | Requires internet access |

See [docs/TESTING.md](docs/TESTING.md) for full testing guide.

## Pull Request Process

1. Ensure all tests pass locally
2. Run `ruff check core/` and `black --check core/` with no errors
3. Update documentation if you changed behavior, contracts, or APIs
4. Add test coverage for new features
5. One approval required for merge

### PR Checklist

- [ ] Tests pass: `pytest tests/ -m "not slow"`
- [ ] Linters pass: `ruff check core/ && black --check core/`
- [ ] Docs updated (if contract-sensitive change)
- [ ] No secrets or credentials in diff
- [ ] Commit messages follow conventional format

## Documentation Quality Gate

Documentation quality is enforced in CI by `.github/workflows/docs-quality.yml`.

Before opening a PR, run:

```bash
python scripts/ci_docs_quality.py
```

Expectations:

- Contract-sensitive changes (`core/sovereign/`, `core/proof_engine/`, `core/pci/`, `deploy/`, etc.) must include documentation updates in the same PR.
- Canonical documentation entrypoint is [docs/README.md](docs/README.md).
- Operational and validation guidance must stay current in:
  - [docs/OPERATIONS_RUNBOOK.md](docs/OPERATIONS_RUNBOOK.md)
  - [docs/TESTING.md](docs/TESTING.md)

## Branch Naming

- `feature/` — New features
- `fix/` — Bug fixes
- `docs/` — Documentation only
- `refactor/` — Code restructuring
- `ci/` — CI/CD changes

## Reporting Issues

Use GitHub Issues with clear reproduction steps and expected vs actual behavior.

## Code of Conduct

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
