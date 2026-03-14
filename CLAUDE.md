# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**BIZRA-DATA-LAKE** is the persistent memory and knowledge layer of the BIZRA ecosystem — a decentralized agentic system built on Proof-Carrying Inference, FATE gates, and constitutional AI governance. The repo contains a Python sovereignty infrastructure (`core/`) and a high-performance Rust workspace (`bizra-omega/`, 23 crates).

**Environment:** WSL Ubuntu on Windows | Python 3.11+ | Rust stable (1.91+)

## Common Commands

### Python

```bash
# Setup (WSL — use .venv-linux, NOT .venv which is the Windows venv)
source .venv-linux/bin/activate
pip install -e ".[dev]"            # Dev dependencies (pytest, linters)
pip install -e ".[full]"           # Adds torch, transformers, sentence-transformers

# Testing
pytest tests/                                              # All tests (excludes @slow by default)
pytest tests/core/pci/                                     # Single module
pytest tests/test_snr_engine.py::test_function_name        # Single test
pytest tests/ -m "not requires_ollama and not requires_gpu and not slow"  # CI-safe subset
pytest tests/ --cov=core --cov-report=term-missing         # With coverage

# Linting (CI enforces all on core/)
ruff check core/                   # Fast linter (primary)
black --check core/                # Formatting
isort --check-only core/           # Import order
mypy core/ --ignore-missing-imports  # Type checking (incremental)

# Data pipeline (run in order)
python corpus_manager.py           # Layer 1: Build 04_GOLD/documents.parquet
python vector_engine.py            # Layer 2: Generate embeddings → 04_GOLD/chunks.parquet
python langextract_engine.py       # Layer 4: LLM extraction → assertions.jsonl
python arte_engine.py              # ARTE: SNR validation
```

### Rust (bizra-omega/)

```bash
cd bizra-omega

# Prerequisite: Z3 solver
sudo apt install libz3-dev

cargo build --workspace --release
cargo test --workspace --release
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings   # Zero warnings enforced in CI

# PyO3 bindings (requires maturin + correct venv)
cd bizra-python
VIRTUAL_ENV=/mnt/c/BIZRA-DATA-LAKE/.venv-linux maturin develop --release

# Maximum optimization (AVX-512)
RUSTFLAGS="-C target-cpu=native" cargo build --profile omega
```

### Frontend (frontend/)

```bash
cd frontend
npm install                        # Install deps (Node >= 20)
npm run dev                        # Vite dev server
npm run build                      # tsc -b + vite build
npm run typecheck                  # tsc --noEmit
npm run lint                       # ESLint (zero warnings enforced)
npm run test                       # Vitest (single run)
npm run test:watch                 # Vitest (watch mode)
npm run ci                         # Full gate: typecheck + lint + test + build
```

React 18 + TypeScript + Vite. Phase state machine (trust→splash→genesis→teach→assembly→dashboard). Dashboard has 6 tabs: cmd, char, skill, quest, comm, prog. Design tokens in `frontend/src/tokens.ts` (synced with `constants.py`).

## Architecture

### Python (`core/`) and Rust (`bizra-omega/`) Mirror

```
Python (core/)                          Rust (bizra-omega/ — 23 crates)
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
├── a2a/        Agent-to-Agent protocol     bizra-hunter/      Bounty system
├── living_memory/ Proactive retrieval      bizra-memory/      Memory synthesis pipeline
├── proof_engine/ Evidence + gates          bizra-hooks/       Nervous system + Ihsan gate
├── spearpoint/ Benchmark campaigns         bizra-action/      Event→Action→Receipt bus
├── auth/       Middleware + auth            bizra-ttrl/        On-device RL (SSO spectral norm)
├── benchmark/  Guardrails + scoring        bizra-agent/       OmniKernel cognitive cycle
├── zpk/        Zero-Point Kernel           bizra-node/        Desktop sovereign binary
├── (FATE gates)                            fate-binding/      Z3 + Dilithium post-quantum
└── (IPC)                                   iceoryx-bridge/    Zero-copy shared memory
```

`core/` has ~58 subpackages total. The table above shows the most-used modules. Additional modules include: `apex/`, `autopoiesis/`, `bridges/`, `elite/`, `graph/`, `hypergraph/`, `nexus/`, `pat/`, `personaplex/`, `resonance/`, `swarm/`, `vault/`, `zpk/`.

`core/sovereign/` is the largest module (~60 files). When decomposed modules exist (governance/, reasoning/, orchestration/, treasury/), prefer using those over the monolithic sovereign/ equivalents.

### Key Architectural Concepts

**Constitutional Thresholds** — All defined in `core/integration/constants.py` (single source of truth). Every module must import from there, not define its own:
- Ihsan (excellence): 0.95 production, 0.90 CI, 0.99 strict/consensus, 1.0 runtime
- SNR (signal quality): 0.85 minimum, 0.95 T1, 0.98 T0/elite
- ADL Gini (justice): <= 0.35 hard gate

**Inference Tiers** — Local-first with tiered fallback, configured in `bizra_config.py`:
1. LM Studio at WSL gateway:1234 (auto-detected, env: LMSTUDIO_HOST)
2. Ollama at `localhost:11434` (fallback)
3. Cloud API (emergency fallback)

**Data Pipeline** — Files flow through numbered directories: `00_INTAKE/` → `01_RAW/` → `02_PROCESSED/` → `03_INDEXED/` → `04_GOLD/`. Duplicates go to `99_QUARANTINE/` via SHA-256 detection. Downloads are always COPIED, never moved.

**Unified Concurrency Fabric (UCF)** — Sharded EventBus (8 namespace shards via FNV-1a) in `bizra-hooks`, two-phase OmniKernel (try_cache_hit + complete_cache_hit) in `bizra-agent`, and PyO3 event bridge (`PyEventBridge` Rust → `RustEventBridge` Python wrapper). Bridge gracefully returns None when PyO3 isn't built.

## CI Pipeline

Defined in `.github/workflows/ci.yml`. Stages:
1. **Lint** — ruff, black, isort, mypy (Python) + cargo fmt, clippy (Rust)
2. **Cross-Language Sync** — validates constants.py ↔ Rust thresholds match
3. **Test** — pytest matrix (3.11, 3.12) + cargo test
4. **PyO3 Bindings** — maturin build + smoke test
5. **Quality Gates** — SNR/Ihsan score validation (skippable via workflow_dispatch)
6. **Security** — bandit, pip-audit, cargo-audit, Trivy
7. **Docker Build** — `deploy/Dockerfile.elite` (Python), `bizra-omega/Dockerfile` (Rust)

Coverage floor: **75%** enforced in pyproject.toml (ratcheted from 38%).

## Test Organization

```
tests/
├── core/           # Unit tests mirroring core/ structure (one subdir per module)
├── integration/    # Cross-module and external service tests
├── property_based/ # Hypothesis-based property tests
└── root_legacy/    # Legacy tests (excluded from pytest via addopts)
```

**Markers:** `@pytest.mark.slow`, `@pytest.mark.integration`, `@pytest.mark.requires_ollama`, `@pytest.mark.requires_gpu`, `@pytest.mark.requires_network`, `@pytest.mark.xdist_group`

**Timeout:** 60 seconds per test. **asyncio_mode:** `auto` (no need for `@pytest.mark.asyncio`).

**Fixtures:** `bizra_root`, `sample_documents`, `sample_chunks` from `tests/conftest.py`.

## Critical Patterns and Gotchas

### Import Paths (Verified — Not Obvious)
- TierPolicy: `core.spearpoint.config` (NOT core.sovereign.tier_policy)
- EvidenceLedger: `core.proof_engine.evidence_ledger` (NOT core.spearpoint.evidence_chain)
- SNR: `core.iaas.snr_v2_adapter` or `core.apex.snr_apex_engine` (NOT core.iaas.snr_maximizer)
- PCI Gates: `core.pci.gates.PCIGateKeeper` (NOT run_gate_chain)
- Constants: `ADL_GINI_THRESHOLD` (NOT ADL_GINI_HARD_GATE)

### Testing Gotchas
- **asyncio**: Never use `asyncio.get_event_loop().run_until_complete()` — always use `asyncio.run()` for sync→async bridges. The old pattern crashes under pytest-xdist.
- **Heavy tests**: Mark with `@pytest.mark.xdist_group("runtime_heavy")` to prevent OOM under parallel execution.
- **Optional deps**: Use `pytest.importorskip("prometheus_client")` etc. from day one — numpy, httpx, prometheus_client are not in base env.
- **New module rule**: New module + ALL files that import it must be in the SAME commit. Never commit code that imports an untracked file.

### PyO3 Bindings
- `bizra-omega/bizra-python/python/bizra/` is gitignored — use `git add -f` for `__init__.py`
- `VIRTUAL_ENV` must point to `.venv-linux` (not `.venv` which is the Windows venv)
- maturin build needs `/root/.cargo/bin` + `/usr/bin` (for `cc` linker) in PATH
- `PyEventBridge` registered in `bizra-python/src/lib.rs`, re-exported via `__init__.py`

### Code Patterns
- All Python paths use forward slashes for cross-platform compatibility
- `core/__init__.py` re-exports all subpackages — `from core import pci` works
- `core/protocols/` defines interface contracts via structural typing (Protocol classes)
- FATE fallback: `_conservative_fallback_check()` (NOT `_manual_constraint_check`) — default-deny, stricter than Z3
- Bare `except:` is forbidden — always use specific exception types

## Configuration

- **`bizra_config.py`** — All paths and hyperparameters. Paths auto-resolve across Windows/WSL/Linux via `BIZRA_DATA_LAKE_ROOT` env var.
- **`core/integration/constants.py`** — Constitutional thresholds (authoritative). Cross-repo sync with Dual-Agentic-System and bizra-omega Rust crates.
- **`.env`** / **`.env.example`** — LLM backend URLs, API keys. Copy `.env.example` to `.env` for local setup.
- **`pyproject.toml`** — All tool configs (pytest, coverage, ruff, black, isort, mypy) centralized here.

## Rust Workspace (bizra-omega/)

23 crates in a unified workspace (v2.0.0). Six layers:

- **Platform** (14): bizra-core, hypergraph, inference, autopoiesis, federation, installer, python, api, tests, hunter, telescript, proofspace, resourcepool, cli
- **Cognitive** (2): bizra-hooks (nervous system), bizra-memory (synthesis pipeline)
- **Action** (1): bizra-action (Event→Action→Receipt bus)
- **TTRL** (1): bizra-ttrl (on-device RL with SSO spectral norm)
- **Desktop** (2): bizra-agent (OmniKernel), bizra-node (sovereign binary)
- **Numeric** (1): bizra-sippar (exact regular number arithmetic for token splits)
- **Bindings** (2): fate-binding (Z3 + Dilithium post-quantum), iceoryx-bridge (zero-copy IPC)

Key deps: `ed25519-dalek` (crypto), `tokio` (async), `blake3` (hashing+rayon), `z3` (formal verification), `pyo3` (Python bindings), `iceoryx2` (IPC), `pqcrypto-mldsa` (post-quantum).

Release profile: fat LTO, single codegen unit, `panic = "abort"`, `strip = true`. Z3 is required: `sudo apt install libz3-dev`.

**Note:** `native/` is deprecated. All Rust development happens in `bizra-omega/`.

## Claude Rules (`.claude/rules/`)

Path-scoped rules are auto-loaded by Claude Code:
- `security.md` — all files: no hardcoded secrets, validate external input, use env vars
- `python-code.md` — `*.py`, `core/**`: PEP 8/484, Google-style docstrings, specific exceptions only
- `typescript-code.md` — `*.ts`/`*.tsx`: strict mode, no `any`, functional components, Vitest
- `sovereign-engine.md` — `core/sovereign/**`, `core/iaas/**`, `core/personaplex/**`: Ihsan SNR >= 0.95, provenance on every inference, Graph-of-Thoughts branching

## Lint Quirks

- Ruff ignores `E402` (deferred imports for performance) and `E501` (Black handles line length)
- MyPy: strict mode globally, relaxed for `core.*` and `tests.*` — strict enforcement adopted incrementally
- Clippy `uninlined_format_args` is a Rust 1.88 lint — pre-existing across workspace
- `# noqa: SEC-001` marks intentional legacy SHA-256 usage (BLAKE3 gate)
