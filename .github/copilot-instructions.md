# Project Guidelines

## Overview

BIZRA-DATA-LAKE is a proof-native constitutional intelligence system. Monorepo with Python sovereignty infrastructure (`core/`, ~75 subpackages) and a high-performance Rust workspace (`bizra-omega/`, 26 crates). For full architecture, see [CLAUDE.md](../CLAUDE.md) and [ARCHITECTURE.md](../ARCHITECTURE.md).

## Build and Test

### Python

```bash
source .venv/bin/activate            # Linux/WSL venv (NOT .venv on Windows)
pip install -e ".[dev]"

pytest tests/                        # All tests (excludes @slow by default)
pytest tests/core/pci/               # Single module
pytest tests/ -m "not requires_ollama and not requires_gpu and not slow"  # CI-safe

ruff check core/                     # Linter (primary)
black --check core/                  # Formatting
isort --check-only core/             # Import order
mypy core/ --ignore-missing-imports  # Type checking
```

### Rust (`bizra-omega/`)

```bash
cd bizra-omega
sudo apt install libz3-dev           # Required: Z3 solver
cargo build --workspace --release
cargo test --workspace --release
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings  # Zero warnings
```

### Frontend (`frontend/`)

```bash
cd frontend && npm install && npm run ci  # Full gate: typecheck + lint + test + build
```

## Architecture

### Constitutional Thresholds (non-negotiable)

All thresholds live in `core/integration/constants.py` — single source of truth. Never define local copies.

- **Ihsan >= 0.95** — ethical floor
- **SNR >= 0.85** — signal-to-noise minimum
- **ADL Gini <= 0.35** — inequality hard gate

### Constitutional Spine (frozen 2026-03-30)

Five root objects in `bizra-omega/bizra-core/src/`: `CanonicalReceipt`, `MissionState`, `TopologyCanon`, `GenesisSeal`, `ReceiptStateMachine`. Python mirror: `core/proof_engine/canonical_receipt_adapter.py`. See [CLAUDE.md § Constitutional Spine](../CLAUDE.md) for details.

### Python–Rust Mirror

Python lives in `core/`, Rust in `bizra-omega/`. Key pairs: `core/pci/` ↔ `bizra-core/`, `core/proof_engine/` ↔ `bizra-proofspace/`, `core/sovereign/` ↔ `bizra-cli/`. When decomposed modules exist (`governance/`, `reasoning/`, `orchestration/`, `treasury/`), prefer those over the monolithic `sovereign/`.

### Data Pipeline

`00_INTAKE/` → `01_RAW/` (immutable) → `02_PROCESSED/` → `03_INDEXED/` → `04_GOLD/`. Duplicates → `99_QUARANTINE/`. Never modify `01_RAW/`.

## Conventions

### Code Style

- **Python:** PEP 8, type hints on public functions, Google-style docstrings, specific exceptions only (bare `except:` forbidden)
- **Rust:** 2021 edition, `snake_case` functions, `PascalCase` types, zero clippy warnings
- **Commits:** Conventional Commits — `feat:`, `fix:`, `test:`, `docs:`

### Truth Labels (required on all claims)

Every function, module, and architectural boundary must declare truth status:

- `[ENFORCEMENT: PROVEN]` — verified by test/proof
- `[ENFORCEMENT: WIRED]` — enforced by type system/guard
- `[OPTIMIZATION: PARTIAL]` — heuristic, not guaranteed
- `[OPTIMIZATION: PLANNED]` — aspirational, not implemented

### Import Gotchas

- TierPolicy: `core.spearpoint.config` (NOT `core.sovereign.tier_policy`)
- EvidenceLedger: `core.proof_engine.evidence_ledger` (NOT `core.spearpoint.evidence_chain`)
- SNR: `core.iaas.snr_v2_adapter` (NOT `core.iaas.snr_maximizer`)
- PCI Gates: `core.pci.gates.PCIGateKeeper` (NOT `run_gate_chain`)
- Constants: `ADL_GINI_THRESHOLD` (NOT `ADL_GINI_HARD_GATE`)

### Testing

- **asyncio:** Use `asyncio.run()`, never `get_event_loop().run_until_complete()` (crashes under xdist)
- **Heavy tests:** Mark with `@pytest.mark.xdist_group("runtime_heavy")`
- **Optional deps:** Use `pytest.importorskip()` from day one
- **New modules:** Module + all files importing it must be in the SAME commit
- **Timeout:** 60s per test. asyncio_mode is `auto` (no `@pytest.mark.asyncio` needed)

## Key Files

| Purpose | File |
|---------|------|
| Constitutional thresholds | `core/integration/constants.py` |
| Runtime config | `bizra_config.py` |
| API server entry | `core/sovereign/__main__.py` |
| Node gateway | `services/node_gateway/app/routers.py` |
| CI pipeline | `.github/workflows/ci.yml` |
| Rust workspace | `bizra-omega/Cargo.toml` |
| PAT/SAT agents | `core/pat/agent.py`, `core/sat/` |
| URP service | `core/urp/service.py` |
| Proof engine | `core/proof_engine/` |
| PCI crypto | `core/pci/crypto.py` |

## References

- Full agent knowledge base: [CLAUDE.md](../CLAUDE.md)
- Architecture: [ARCHITECTURE.md](../ARCHITECTURE.md), [docs/ARCHITECTURE_BLUEPRINT_v2.3.0.md](../docs/ARCHITECTURE_BLUEPRINT_v2.3.0.md)
- Contributing: [CONTRIBUTING.md](../CONTRIBUTING.md)
- Rust hunter agent: [bizra-omega/AGENTS.md](../bizra-omega/AGENTS.md)
- Data pipeline conventions: [docs/internal/AGENTS.md](../docs/internal/AGENTS.md)
- CI details: [CLAUDE.md § CI Pipeline](../CLAUDE.md)
