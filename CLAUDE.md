# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**BIZRA-DATA-LAKE** is the persistent memory and knowledge layer of the BIZRA ecosystem — a decentralized agentic system built on Proof-Carrying Inference, FATE gates, and constitutional AI governance. This repo contains both a Python sovereignty infrastructure (`core/`) and a high-performance Rust workspace (`bizra-omega/`, 13 crates).

**Environment:** WSL Ubuntu on Windows | RTX 4090 (16GB VRAM), 128GB RAM | Python 3.11+ | Rust stable (1.88+)

## Common Commands

### Python

```bash
# Setup
source .venv/bin/activate          # WSL/Linux
pip install -e ".[dev]"            # Install with dev dependencies
pip install -e ".[full]"           # Includes torch, transformers, sentence-transformers

# Testing
pytest tests/                                              # All tests
pytest tests/core/pci/                                     # Single module
pytest tests/test_snr_engine.py::test_function_name        # Single test
pytest tests/ -m "not requires_ollama and not requires_gpu and not slow"  # CI-safe subset
pytest tests/ --cov=core --cov-report=term-missing         # With coverage

# Linting (CI enforces all of these on core/)
ruff check core/                   # Fast linter (primary)
black --check core/                # Formatting check
isort --check-only core/           # Import order
mypy core/ --ignore-missing-imports  # Type checking (incremental — many pre-existing errors)

# Data pipeline (run in order)
python corpus_manager.py           # Layer 1: Build 04_GOLD/documents.parquet
python vector_engine.py            # Layer 2: Generate embeddings → 04_GOLD/chunks.parquet
python langextract_engine.py       # Layer 4: LLM extraction → assertions.jsonl
python arte_engine.py              # ARTE: SNR validation
```

### Rust (bizra-omega/)

```bash
cd bizra-omega
cargo build --workspace --release
cargo test --workspace --release
cargo test --doc --workspace
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings   # Zero warnings enforced in CI

# PyO3 bindings (requires maturin)
cd bizra-python && maturin develop --release

# Maximum optimization (AVX-512)
RUSTFLAGS="-C target-cpu=native" cargo build --profile omega
```

## Architecture

The codebase has two major layers that mirror each other:

```
Python (core/)                          Rust (bizra-omega/)
├── pci/        Proof-Carrying Inference    bizra-core/        Constitution + FATE + Identity
├── federation/ P2P gossip + BFT consensus  bizra-federation/  Gossip + signed messages
├── inference/  Tiered LLM gateway          bizra-inference/   Inference backends
├── sovereign/  Runtime engine (largest)    bizra-cli/         Terminal UI dashboard
├── integration/ Cross-module bridge        bizra-api/         REST API server
├── iaas/       SNR calculation engine      bizra-proofspace/  Proof verification
├── governance/ Constitutional gates        bizra-telescript/  Mobile agent scripts
├── reasoning/  Graph-of-Thoughts           bizra-autopoiesis/ Self-healing
├── orchestration/ Event bus + agents       bizra-resourcepool/ Compute allocation
├── treasury/   Resource management         bizra-python/      PyO3 bindings
└── a2a/        Agent-to-Agent protocol     bizra-hunter/      Bounty system
```

### Key Architectural Concepts

**Constitutional Thresholds** — All defined in `core/integration/constants.py` (single source of truth). Every module must import from there, not define its own:
- Ihsan (excellence): 0.95 production, 0.90 CI, 0.99 strict/consensus
- SNR (signal quality): 0.85 minimum, 0.95 T1, 0.98 T0/elite
- ADL Gini (justice): <= 0.40 hard gate

**Inference Tiers** — Local-first with tiered fallback, configured in `bizra_config.py`:
1. LM Studio at `192.168.56.1:1234` (primary)
2. Ollama at `localhost:11434` (fallback)
3. Cloud API (emergency fallback)

**Data Pipeline** — Files flow through numbered directories: `00_INTAKE/` → `01_RAW/` → `02_PROCESSED/` → `03_INDEXED/` → `04_GOLD/`. Duplicates go to `99_QUARANTINE/` via SHA-256 detection. Downloads are always COPIED, never moved.

**`core/sovereign/`** is the largest module (~60 files). It contains the runtime engine, Graph-of-Thoughts reasoning, guardian council, treasury, and most integration points. When decomposed modules exist (governance/, reasoning/, orchestration/, treasury/), prefer using those over the monolithic sovereign/ equivalents.

## CI Pipeline

Defined in `.github/workflows/ci.yml`. Stages run in order:
1. **Lint** — ruff, black, isort, mypy (Python) + cargo fmt, clippy (Rust)
2. **Test** — pytest matrix (3.11, 3.12 with coverage) + cargo test
3. **PyO3 Bindings** — maturin build + smoke test
4. **Quality Gates** — SNR/Ihsan score validation (can be skipped via workflow_dispatch)
5. **Security** — bandit, pip-audit, cargo-audit, Trivy
6. **Docker Build** — `deploy/Dockerfile.elite` (Python), `bizra-omega/Dockerfile` (Rust)

Coverage floor: 60% enforced (ratcheting toward 95%).

## Test Organization

```
tests/
├── core/           # Unit tests mirroring core/ structure (one subdir per module)
├── integration/    # Cross-module and external service tests
├── property_based/ # Hypothesis-based property tests
└── root_legacy/    # Legacy tests (excluded from pytest via addopts)
```

**Markers:** `@pytest.mark.slow`, `@pytest.mark.integration`, `@pytest.mark.requires_ollama`, `@pytest.mark.requires_gpu`, `@pytest.mark.requires_network`

**Timeout:** 60 seconds per test (configured in pyproject.toml).

**Fixtures:** `bizra_root`, `sample_documents`, `sample_chunks` available session-wide from `tests/conftest.py`.

## Configuration

- **`bizra_config.py`** — All paths and hyperparameters. Paths auto-resolve across Windows/WSL/Linux via `BIZRA_DATA_LAKE_ROOT` env var.
- **`core/integration/constants.py`** — Constitutional thresholds (authoritative). Cross-repo sync with Dual-Agentic-System and bizra-omega Rust crate.
- **`.env`** / **`.env.example`** — LLM backend URLs, API keys. Copy `.env.example` to `.env` for local setup.
- **`pyproject.toml`** — All tool configs (pytest, coverage, ruff, black, isort, mypy) are centralized here.

## Rust Workspace (bizra-omega/)

13 crates with workspace-level dependency management. Key dependencies: `ed25519-dalek` (crypto), `tokio` (async), `serde` (serialization), `blake3` (hashing with rayon parallelism).

Release profile uses fat LTO + single codegen unit + `panic = "abort"`. The `omega` profile adds AVX-512 native CPU targeting.

## Important Patterns

- All Python paths use forward slashes for cross-platform compatibility
- Metadata stored as `.meta.json` alongside processed files
- `core/__init__.py` re-exports all subpackages — imports like `from core import pci` work
- The `core/protocols/` package defines interface contracts via structural typing (Protocol classes)
- Ruff ignores `E402` (deferred imports for performance) and `E501` (Black handles line length)
- MyPy runs in strict mode globally but relaxes `core.*` and `tests.*` modules — strict enforcement is being adopted incrementally
