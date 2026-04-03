# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

BIZRA (بذرة - "seed") is a dual-agentic AI orchestration system with two parallel agent teams:

- **PAT (Personal Agentic Team)**: 7 specialized agents for task execution (Strategic, Creative, Analytical, Implementation, Quality, User Advocate, Coordination)
- **SAT (System Agentic Team)**: 5 guardian agents for validation (Security, Ethics, Performance, Consistency, Resources)

**Request Flow**: `User → SAT Validation (3/5 consensus) → PAT Execution → SAT Evaluation → Receipt Emission → Response`

**The Law**: لا نفترض — "We do not assume." All outputs gated by Ihsan threshold (≥0.95). Fail-closed defaults. SAT can veto any PAT action.

## System Layers

The codebase has five major layers. Understanding their relationships is critical:

```
┌──────────────────────────────────────────────────────────────┐
│  constellation/     29-agent roster with SNR-tier routing    │
│                     (cross-pollination teams, GoT/ToT/CoT)  │
├──────────────────────────────────────────────────────────────┤
│  bizra_kernel/      Intelligence layer: federation, PAT      │
│                     enforcement, knowledge graph, chaos      │
├──────────────────────────────────────────────────────────────┤
│  core/              Python FastAPI kernel (port 8010)        │
│                     SAPE, FATE, LLM routing, memory, apex   │
├──────────────────────────────────────────────────────────────┤
│  src/               Rust production engine (port 8080)       │
│                     PAT/SAT, SAPE, FATE, receipts, HTTP API │
├──────────────────────────────────────────────────────────────┤
│  crates/            Workspace: finance-v1, gateway, bridge   │
└──────────────────────────────────────────────────────────────┘
```

- **`src/` (Rust)** and **`core/` (Python)** are parallel implementations (see Rust-Python Parity section).
- **`bizra_kernel/`** sits above `core/` — handles federation consensus, PAT enforcement pipelines, knowledge graph sharding, chaos testing, and sovereign identity.
- **`constellation/`** is the agent roster layer — 29 specialized agents (historical Islamic scholars as personas) organized into 8 cross-pollination teams with SNR-tier routing (T0-T6).
- **`crates/`** contains: `finance-v1` (MaaSP pricing engine, port 3002), `bizra-gateway` (WebSocket/REST bridge for React UI), `bizra_bridge` (Python FFI via PyO3).

Entry points: `src/main.rs` → `src/cli/mod.rs` (Clap CLI), `src/lib.rs` (library root), `core/main.py` (FastAPI).

## Build & Run Commands

```bash
# Rust — build and test
cargo build --release
cargo test
cargo clippy --all-targets
cargo fmt --check

# Run single Rust test
cargo test sape::tests                    # By module
cargo test --test pat_sat_runtime_tests   # By integration test file
BIZRA_ADAPTER_MODE=simulated cargo test   # Without real LLMs

# Python kernel
pip install -r requirements-kernel.txt
python -m core.main                       # Starts FastAPI on :8010
pytest tests/                             # All Python tests
pytest tests/test_kg_receipts.py          # Single file
pytest -m "not slow"                      # Skip slow tests
pytest --cov=core --cov-report=html       # With coverage

# Docker (full stack — 11 services)
docker compose up -d
docker compose logs -f elite              # Rust service logs
docker compose logs -f kernel             # Python service logs
```

### Rust CLI Subcommands

```bash
cargo run -- serve --port 8080   # Start HTTP API server (production)
cargo run -- task "prompt"       # Execute PAT/SAT task interactively
cargo run -- status              # Check system health
cargo run -- models              # List available models
cargo run -- demo                # Run verification demo
```

### Rust Feature Flags

Default features: `http`, `observability`, `crypto` (Ed25519 signing). Optional: `z3-solver` (formal verification, requires system Z3 library).

### MSSC CLI (Minimal Self-Sovereign Computation)

```bash
python mssc/mssc.py genesis build    # Generate Block 0
python mssc/mssc.py api up           # Start Node-0 Validation API
python mssc/mssc.py contribute run   # Privacy-preserving contribution
python mssc/mssc.py poi attest       # Generate Proof-of-Impact
python mssc/mssc.py poi verify       # Deterministic verification
```

## Docker Service Ports

Docker maps to non-standard host ports to avoid conflicts. All bind to `127.0.0.1` only.

| Service | Host Port | Purpose |
|---------|-----------|---------|
| **elite** | 8080 | Rust PAT+SAT+SAPE engine |
| **kernel** | 8010 | Python FastAPI |
| **postgres** | **5433** | PostgreSQL + pgvector |
| **synapse** (Redis) | **6380** | State, receipts, FATE |
| **wisdom** (Neo4j) | 7474/7687 | Graph evidence |
| **vectors** (ChromaDB) | 8001 | Embeddings |
| **refinery** | 8081 | Ingestion daemon |
| **finance** | 3002 | MaaSP pricing engine |
| **agentic-flow** | 3100 | Swarm intelligence |
| **prometheus** | 9090 | Metrics collection |
| **grafana** | 3000 | Dashboards (admin/bizra_glass) |

Optional: `docker compose --profile optional up -d` also starts `fate_auditor`.

## Core Concepts

### Ihsan (إحسان) — Excellence Score

Weighted ethical score across 8 dimensions. **Single source of truth**: `constitution/ihsan_v1.yaml`

Dimensions (by weight): `correctness` (0.22), `safety` (0.22), `user_benefit` (0.14), `efficiency` (0.12), `auditability` (0.12), `anti_centralization` (0.08), `robustness` (0.06), `adl_fairness` (0.04).

**Thresholds**: production=0.95, staging=0.95, ci=0.90, dev=0.80. Configurable via `BIZRA_IHSAN_ENV`.

### SAPE (Symbolic-Abstraction Probe Elevation)

9-probe verification: `threat_scan`, `compliance`, `bias`, `user_benefit`, `correctness`, `safety`, `groundedness`, `relevance`, `fluency`. Auto-elevates patterns with >3 repetitions into kernel shortcuts.

- **Sequential** (`src/sape.rs`): Full-featured ~900ms
- **Parallel** (`src/sape_parallel.rs`): 3-batch ~300ms

### FATE (Fail-Safe Agentic Trust Escalation)

Escalation levels: Low → Medium → High → Critical. Handles quarantine, human review routing, rejection receipts via Redis (synapse).

### Receipt-Native Architecture

All decisions emit structured receipts: `receipt_id`, `timestamp`, `task_summary`, `rejection_codes`, `escalation_level`, `integrity_hash` (SHA-256). Receipts are **append-only** JSONL in `docs/evidence/receipts/`.

**Receipt Schema Guard**: Changes to receipt fields require updating:
1. `src/receipts.rs` (Rust struct)
2. `core/fate.py` (Python equivalent)
3. Parsers in `tests/` and `scripts/`
4. Evidence docs in `docs/execution/`

### Constellation (29-Agent Multi-Expert System)

Defined in `constellation/agents/roster.yaml`. Agents are organized by domain with SNR-tier routing:

| SNR Tier | Precision | Domains |
|----------|-----------|---------|
| T1 (96-98%) | Maximal | Authentication, security |
| T2 (93-97%) | Scientific | Natural sciences, optics |
| T3 (92-95%) | Medical | Medicine, diagnostics |
| T4 (93-96%) | Mathematical | Algorithms, computation |
| T5 (88-92%) | Philosophical | Ethics, theology |
| T6 (85-90%) | Creative | Poetry, exploration |

Reasoning modes: CoT (default/fastest), ToT (planning/branching), GoT (interdisciplinary synthesis). Configured in `constellation/router/policy.yaml`.

8 cross-pollination teams defined in `constellation/teams/configurations.yaml` (e.g., Scientific Method Elite, Systems Architecture Dream, Legal Reasoning Panel).

## Rust-Python Parity

When modifying SAPE, FATE, or receipt logic, check the counterpart: `src/sape.rs` ↔ `core/sape.py`, `src/fate.rs` ↔ `core/fate.py`, `src/receipts.rs` ↔ `core/fate.py`, `src/ihsan.rs` ↔ constitution YAML. Run `task das:validate:parity` to verify.

## Critical Development Rules

### Fail-Closed Error Handling

Errors must fail visibly. Never proceed without SAT approval:
```rust
if !validation.consensus_reached {
    fate.escalate_rejection(...);
    receipts.emit_rejection(...);
    return Err(...);  // NEVER silently continue
}
```

### Receipt-First Development

New behavior must either emit/extend a receipt in `src/receipts.rs` (and Python equivalent), or explicitly document why it's non-receipted.

### Constitution as Code

`constitution/ihsan_v1.yaml` contains executable constraints — don't change semantics casually. The `constitution/` directory holds 11 YAML files that serve as governance rules (PAT enforcement, token economics, scaling policies, ontology, lexicon).

## Key Files

| File | Purpose |
|------|---------|
| `src/bridge.rs` | PAT-SAT coordination entry point (BridgeCoordinator) |
| `src/pat.rs` / `src/sat.rs` | PAT (7 agents) and SAT (5 guardians) |
| `src/ihsan.rs` | Constitution loading, threshold enforcement |
| `src/sape.rs` / `src/sape_parallel.rs` | SAPE probe engine (sequential / 3-batch parallel) |
| `src/fate.rs` | FATE escalation handling |
| `src/receipts.rs` | Receipt schemas and emission |
| `src/http.rs` | Axum HTTP API server |
| `src/mcp.rs` / `src/a2a.rs` | MCP tools / Agent-to-Agent protocol |
| `src/synapse.rs` | Redis state persistence |
| `core/main.py` | Python kernel FastAPI entry point |
| `core/sape.py` / `core/fate.py` | Python SAPE/FATE implementations |
| `core/llm.py` | LLM routing (Ollama/LM Studio) |
| `core/unified_memory.py` | Multi-tier memory (M1-M6) |
| `bizra_kernel/kernel.py` | Main kernel executive |
| `bizra_kernel/ihsan_gate.py` | Ihsan scoring |
| `bizra_kernel/pat_enforcement_pipeline.py` | PAT validation pipeline |
| `constellation/orchestrator.py` | Multi-agent orchestrator |
| `constellation/agents/roster.yaml` | Agent definitions and domains |
| `constitution/ihsan_v1.yaml` | Ihsan constitution (single source of truth) |
| `config/cognition_contract.json` | Unified Rust/Python API schema |
| `model-family-genesis-v1-SEALED.yaml` | Model routing config |

### Rust `src/` Subsystem Directories

`apex/` (orchestration, circuit breaker), `federation/` (gossip, consensus), `sovereign/` (thermal router, nodes), `sovereignty/` (policy, key mgmt, isolation), `autopoietic/` (self-generating loops, proof chains), `blockchain/` (chain, tokens), `pci/` (gates, envelope), `unified/` (cognitive bridge), `kernel/` (contract engine), `cli/` (Clap subcommands).

## HTTP API

All endpoints require bearer token auth (`BIZRA_API_TOKEN`) and are rate-limited (100 req/min/IP).

```
GET  /health     — Ihsan status check
POST /execute    — Standard dual-agentic execution
POST /enhanced   — Enhanced with reasoning/tools
POST /mcp/call   — MCP tool invocation
```

## Model Routing

Defined in `model-family-genesis-v1-SEALED.yaml`, accessed via Ollama (`http://localhost:11434`):

| Role | Model | Purpose |
|------|-------|---------|
| cold_core | deepseek-r1:8b | Deterministic reasoning |
| warm_surface | mistral:latest | User-facing |
| embeddings | nomic-embed-text:latest | Vector embeddings |
| primary_reasoning | bizra-planner:latest | Planning |

## Testing

### Python Test Markers

Defined in `pytest.ini` with `asyncio_mode = auto`:
- `@pytest.mark.slow` — Skip with `-m "not slow"`
- `@pytest.mark.integration` — Requires running services
- `@pytest.mark.tls` — Requires `BIZRA_TLS_TESTS=1`

### Key Test Files

**Rust**: `tests/pat_sat_runtime_tests.rs`, `tests/sape_integration_tests.rs`, `tests/sat_rejection_tests.rs`, `tests/integration_harness.rs`

**Python**: `tests/test_kg_receipts.py`, `tests/test_sat_consensus.py`, `tests/test_synapse_security.py`, `tests/test_seed_lifecycle.py`, `tests/test_apex_integration.py`, `tests/test_snr_enforcer.py`, `tests/test_agentic_flow_bridge.py`

## Environment Variables

See `.env.example` for full list. Critical ones:

```bash
BIZRA_ADAPTER_MODE=real|simulated   # simulated = no real LLMs (for testing)
BIZRA_API_TOKEN=<token>             # API auth
BIZRA_IHSAN_ENV=development         # Controls Ihsan threshold tier
OLLAMA_BASE_URL=http://localhost:11434
LMSTUDIO_BASE_URL=http://localhost:1234
IHSAN_THRESHOLD=0.95
RUST_LOG=info,bizra=debug           # Rust log levels
AGENTIC_FLOW_URL=http://localhost:3100
AGENTIC_FLOW_ENABLED=true
```

## CI/CD Gates

GitHub Actions: `.github/workflows/elite-ci-cd.yml`

1. **Preflight**: Repository hygiene
2. **Security Gate**: cargo-audit, cargo-deny, gitleaks
3. **Quality Gate**: fmt, clippy, tests
4. **Ihsan Gate**: Ethics threshold (0.95)
5. **Performance Gate**: Build time, binary size
6. **Container Gate**: Build, scan, push to ghcr.io

## State Storage

| Data | Storage | Key Pattern / Location |
|------|---------|------------------------|
| Session Memory | Redis (synapse) | `bizra:session:{id}` |
| Agent Presence | Redis (synapse) | `bizra:presence:{id}` (TTL: 30s) |
| SAPE Elevations | Redis (synapse) | `bizra:sape:elevation:{hash}` |
| FATE Escalations | Redis (synapse) | `bizra:fate:escalation:{id}` |
| Knowledge Graph | PostgreSQL + pgvector | Relational + vector similarity |
| Graph Evidence | Neo4j (wisdom) | Cypher queries |
| Embeddings | ChromaDB (vectors) | Collection-based |
| Receipts | Filesystem | `docs/evidence/receipts/` (append-only JSONL) |
| Multi-tier Memory | `core/unified_memory.py` | M1 (working) through M6 (archival) |

## Task Runner

`Taskfile.yml` orchestrates all components. Run `task --list` for available commands. Key validation tasks: `task das:validate:parity`, `task das:validate:secrets`, `task das:validate:ihsan`, `task das:sape:audit`, `task das:quality`.

## Debug Logging

```bash
RUST_LOG=trace cargo run                                    # Full debug
RUST_LOG=bizra::sape=trace,bizra::mcp=debug cargo run       # Component-specific
```
