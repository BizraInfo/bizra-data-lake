# Dual-Bus Kernel Blueprint — Implementation Status

## Mapping: Blueprint vs Codebase Reality

| Blueprint Section | Existing Implementation | Status | Gap |
|---|---|---|---|
| **EventBus (Immutable Truth)** | `core/bus/subscribers.py` (CQRS) + `core/sovereign/event_bus.py` (async) + `core/bus/event_publisher.py` (fanout) | IMPLEMENTED | Needs Rust `bizra-eventbus` crate for deterministic hashing |
| **ActionBus (Gated Executor)** | `core/bus/action_bus.py` + `bizra-omega/bizra-action/` | IMPLEMENTED | TeleScript + FATE gates exist; Skill Tier gate partial |
| **BLAKE3 Chain** | `bizra-core/src/canonical.rs` (219 LOC) + `bizra-mission/src/receipt.rs` | IMPLEMENTED | Domain separation, 7 prefixes, 5 invariants |
| **Reducers** | `core/sovereign/helix3.py` (aggregator) + `core/node0/heartbeat.py` (breath) | PARTIAL | Wallet/Memory/Reflex reducers inline, not standalone |
| **Event Enum (Rust)** | `bizra-hooks/src/types.rs` (Event struct) + `bizra-action/src/types.rs` | PARTIAL | Not unified enum — spread across crates |
| **Redis Store** | `docker-compose.yml` (Redis 6379+6380) | RUNNING | Python store module not yet Redis-backed (uses JSONL) |
| **K8s Persistence** | `deploy/k8s/` manifests exist | STRUCTURAL | PVC for Redis not deployed |
| **Prometheus Metrics** | Grafana + Prometheus running (ports 3000/9090) | RUNNING | ServiceMonitor not yet applied |
| **Ihsan Thresholds** | `core/integration/constants.py` (single source of truth) | IMPLEMENTED | Blueprint values align with existing constants |
| **CI/CD Security** | `.github/workflows/ci.yml` (24 gates) + `phase56-security-gate.yml` | IMPLEMENTED | Missing: dedicated security.yml with SBOM |
| **HDA Executor** | `bizra-omega/bizra-action/src/dispatcher.rs` | IMPLEMENTED | Sandboxed execution via channel handlers |
| **Dead-Letter Evidence** | `core/node0/heartbeat.py` (event_dead_letters.jsonl) | IMPLEMENTED | Added this session |

## What Already Exists (No Build Needed)

1. BLAKE3 receipt chains with domain separation — `canonical.rs`
2. Ed25519 signing — `receipt.rs` sign()/verify()
3. Constitutional gates — Ihsan floor, Gini halt, FATE filtering
4. Event fanout — `event_publisher.py` (CQRS + sovereign unified)
5. ActionBus with TeleScript policy — `core/bus/action_bus.py`
6. 9-stage mission pipeline — `core/sovereign/mission_executor.py`
7. Living Memory — `core/living_memory/brain.py`
8. Reflex compilation — `bizra-agent/src/reflex_compiler.rs`
9. Token ledger — `core/token/ledger.py` (hash-chained, Ed25519 signed)
10. Prometheus + Grafana — running on ports 9090/3000

## What Needs Building (Priority Order)

### P0: Unified Rust EventBus crate
- Blueprint Section II — deterministic Event enum + BLAKE3 chain
- Existing: event types spread across `bizra-hooks`, `bizra-action`, `bizra-mission`
- Action: Create `bizra-eventbus` crate that unifies the enum
- Estimate: 200 LOC, 1 session

### P1: Standalone Reducers
- Blueprint Section II.2 — Wallet, Memory, Reflex, Network reducers
- Existing: reduction logic inline in `helix3.py` and `heartbeat.py`
- Action: Extract into `core/bus/reducers/` module
- Estimate: 150 LOC, 1 session

### P2: Redis-backed Python store
- Blueprint Section IV — Replace JSONL with Redis for hot state
- Existing: Redis running, JSONL used for cold persistence
- Action: Create `core/store.py` with Redis + JSONL dual-write
- Estimate: 100 LOC, 1 session

### P3: Security CI workflow
- Blueprint Section V — Dedicated security scanning pipeline
- Existing: bandit + cargo-audit in main CI, no SBOM
- Action: Create `.github/workflows/security.yml`
- Estimate: 50 LOC YAML, 1 session

### P4: K8s PVC for Redis
- Blueprint Section VII — Persistent Redis in K8s
- Existing: Redis deployment without PVC
- Action: Add PVC + resource limits
- Estimate: Deploy config only

## Constants Alignment Check

| Blueprint | Codebase (`constants.py`) | Match? |
|-----------|--------------------------|--------|
| Mission Floor: 0.85 | `SNR_THRESHOLD = 0.85` | YES |
| Minting: 0.95 | `UNIFIED_IHSAN_THRESHOLD = 0.95` | YES |
| Gini: 0.35 | `ADL_GINI_THRESHOLD = 0.35` | YES |
| Zakat: 2.5% | `ZAKAT_RATE = 0.025` | YES |
| Harberger: 5% | `ADL_HARBERGER_TAX_RATE = 0.05` | YES |

All thresholds align. No drift between blueprint and implementation.
