# BIZRA Dual-Agentic System - AI Agent Instructions

<!-- CONTRACT HEADER — DO NOT REMOVE -->
<!--
  PURPOSE: Maximize agent productivity + enforce repo invariants
  NON-GOALS: No architecture rewrites, no invented practices, no guessing paths
  EVIDENCE RULE: Claims must link to repo paths or logs (never hallucinate file locations)
  FAIL-CLOSED RULE: If Ihsān gate missing/invalid → structured rejection (never best-effort)
  AUTHORITATIVE PATH: .github/copilot-instructions.md (this file; no other copies allowed)
  LAST VALIDATED: 2025-12-21T20:00+04:00 | Rust 45/45 tests | Docker 5/5 healthy
-->

---

## Architecture Overview

This is a **dual-agentic orchestration system** with two parallel agent teams:

- **PAT (Personal Agentic Team)**: 7 specialized agents for task execution (Strategic, Creative, Analytical, Implementation, Quality, User Advocate, Coordination)
- **SAT (System Agentic Team)**: 5 guardian agents for validation (Security, Ethics, Performance, Consistency, Resources)

All requests flow: `User → SAT Validation (3/5 consensus) → PAT Execution → SAT Evaluation → Response`

**Dual Implementation**:
- `src/` - Rust core (production, port 8080) - handles PAT/SAT, MCP, A2A, SAPE, FATE
- `core/` - Python kernel (port 8000) - FastAPI, SAPE planning, FATE engine, LLM routing

## Critical Concepts

### Ihsān (إحسان) - Excellence Score
- Weighted ethical score across 8 dimensions defined in [constitution/ihsan_v1.yaml](../constitution/ihsan_v1.yaml)
- **Threshold**: 0.95 production, 0.90 CI, 0.80 dev (configurable via `BIZRA_IHSAN_ENV`)
- Key dimensions: `correctness` (0.22), `safety` (0.22), `user_benefit` (0.14)
- All outputs gated by Ihsān threshold - if score < threshold, execution fails

### SAPE (Symbolic-Abstraction Probe Elevation)
- 9-probe verification system: threat_scan, compliance, bias, user_benefit, correctness, safety, groundedness, relevance, fluency
- Auto-elevates patterns with >3 repetitions into optimized kernel shortcuts
- See [src/sape.rs](../src/sape.rs) and [core/sape.py](../core/sape.py)

### FATE (Fail-Safe Agentic Trust Escalation)
- Escalation levels: Low → Medium → High → Critical
- Handles quarantine, human review routing, rejection receipts
- Uses Redis (Synapse) for persistent escalation storage

### Receipt-Native Architecture
All decisions emit structured receipts with:
```rust
// From src/receipts.rs
pub struct RejectionReceipt {
    receipt_id, timestamp, task_summary,
    rejection_codes, escalation_level,
    integrity_hash  // SHA-256
}
```
Receipts are append-only, stored in `docs/evidence/receipts/`

> **⚠️ Receipt Schema Guard**: Changing receipt fields requires updating:
> 1. `src/receipts.rs` (Rust struct)
> 2. `core/fate.py` (Python equivalent if applicable)
> 3. Any parsers in `tests/` or `scripts/`
> 4. Evidence docs in `docs/execution/`

## Build & Run Commands

```powershell
# Rust (Elite engine)
cargo build --release
cargo run                    # Starts on :8080
cargo test                   # Run tests
cargo clippy                 # Linting

# Python kernel
pip install -r requirements-kernel.txt
python -m core.main          # Starts on :8000

# Docker (full stack)
docker compose up -d         # Starts: postgres, redis (synapse), neo4j (wisdom), chromadb (vectors), kernel, elite
docker compose logs -f elite # Watch Rust service
```

## Service Dependencies (docker-compose.yml)

| Service  | Image | Port | Purpose |
|----------|-------|------|---------|
| postgres | pgvector/pgvector:pg16 | 5432 | Knowledge graph + vector store |
| synapse  | redis:7-alpine | 6379 | State persistence, receipts, FATE |
| wisdom   | neo4j:5.15-community | 7474/7687 | Graph evidence for SAPE |
| vectors  | chromadb/chroma | 8001 | Embeddings |
| kernel   | Dockerfile | 8010 | Python FastAPI |
| refinery | Dockerfile.refinery | 8081 | Python refinery daemon |
| elite    | Dockerfile.rust | 8080 | Rust PAT+SAT+SAPE |

> **Port reference**: See `docker-compose.yml` service definitions for `refinery` and `kernel`.

## Patterns & Conventions

### Fail-Closed Error Handling
```rust
// REQUIRED: Errors fail visibly, never silently
if !validation.consensus_reached {
    let escalation = fate.escalate_rejection(...);
    receipts.emit_rejection(...);
    return Err(...);  // Never proceed without SAT approval
}
```

### Model Routing (Ollama)
Models defined in [model-family-genesis-v1-SEALED.yaml](../model-family-genesis-v1-SEALED.yaml):
- `cold_core`: deepseek-r1:8b (deterministic reasoning)
- `warm_surface`: mistral:latest (user-facing)
- `embeddings`: nomic-embed-text:latest
- `primary_reasoning`: bizra-planner:latest

### API Patterns (src/http.rs)
```rust
// All endpoints require bearer token auth
// Rate limited: 100 req/min per IP
// Request ID header: X-Request-ID
POST /execute   -> DualAgenticResponse
POST /enhanced  -> EnhancedDualAgenticResponse
POST /mcp/call  -> MCP tool invocation
GET  /health    -> Ihsān status check
```

## Key Files

- [src/bridge.rs](../src/bridge.rs) - PAT-SAT coordination entry point
- [src/ihsan.rs](../src/ihsan.rs) - Constitution loading and threshold enforcement
- [src/sape.rs](../src/sape.rs) / [core/sape.py](../core/sape.py) - SAPE probe engine
- [src/fate.rs](../src/fate.rs) / [core/fate.py](../core/fate.py) - Escalation handling
- [constitution/ihsan_v1.yaml](../constitution/ihsan_v1.yaml) - Single source of truth for ethical weights

## Testing

```powershell
# Rust tests
cargo test                              # All tests
cargo test --test pat_sat_runtime_tests # PAT/SAT integration

# Python tests  
pytest tests/test_kg_receipts.py

# Full integration
docker compose up -d
curl http://localhost:8080/health
```

## CI/CD Gates (.github/workflows/elite-ci-cd.yml)

1. **Security Gate**: cargo-audit, cargo-deny, gitleaks
2. **Quality Gate**: fmt, clippy, tests
3. **Ihsān Gate**: Ethics threshold enforcement
4. **Performance Gate**: Build time, binary size
5. **Container Gate**: Build, scan, push

## External Integrations

- **HyperGraphRAG** ([HyperGraphRAG/](../HyperGraphRAG/)) - Hypergraph-structured knowledge retrieval
- **ACE Framework** ([ace-framework/](../ace-framework/)) - Multi-agent team orchestration with Ollama
- **Neo4j** (Wisdom) - Graph evidence for high-stakes SAPE probes
- **MCP Protocol** - Tool access via Model Context Protocol

## Quick Start

```powershell
# Rust validation
cargo build && cargo test --lib && cargo clippy --all-targets

# Python validation  
python -m compileall core
python -c "from core import main, sape, fate; print('core import OK')"

# Integration truth (Docker)
docker compose up -d
docker compose ps --all --no-trunc
docker compose logs refinery --tail=200
```

## Conventions

1. **Fail-closed**: If a gate/score is missing or invalid, return a structured rejection — never "best-effort" past ethics/gates
2. **Receipts first**: New behavior must either emit/extend a receipt in `src/receipts.rs` (and Python equivalent), or explicitly document why it's non-receipted
3. **Keep schemas stable**: Receipts/gates are evidence artifacts; changes require updating docs + any parsers
4. **Evidence-driven docs**: Link to concrete logs/artifacts under `docs/execution/evidence/` or `docs/evidence/`
5. **Constitution as code**: `constitution/ihsan_v1.yaml` is executable constraints — don't change semantics casually

## Where to Start (implementation checklist)

1. **Find the enforcement point**: Rust `src/*` vs Python `core/*` vs compose wiring
2. **Identify receipt/gate impact**: Update `src/receipts.rs` / constitution refs if needed
3. **Add/adjust tests**: Unit → integration → compose (closest to boundary you changed)
4. **Emit evidence**: If behavior is gated, ensure it produces auditable output
