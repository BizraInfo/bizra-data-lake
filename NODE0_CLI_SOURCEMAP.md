# NODE0 CLI Source Map — Canonical Four-Plane Architecture

> **Frozen:** 2026-04-03 | **Authority:** Layer 1 (Constitutional Core)
> **Doctrine:** The kernel is physics; the higher constitution is jurisprudence.

---

## Four Planes

### 1. Kernel Plane (Law)

The kernel defines hard invariants. All gate verdicts originate here.

| Component | Path | Language | Purpose |
|-----------|------|----------|---------|
| PAT/SAT Bridge | src/bridge.rs | Rust | PAT-SAT coordination entry point |
| PAT Agents (7) | src/pat.rs | Rust | Strategic, Creative, Analytical, Implementation, Quality, User Advocate, Coordination |
| SAT Guardians (5) | src/sat.rs | Rust | Security, Ethics, Performance, Consistency, Resources |
| Mission Contracts | src/mission.rs | Rust | MissionEnvelope, GateVerdict, ReceiptArtifact, ManifestArtifact |
| Ihsan Gate | src/ihsan.rs | Rust | Constitution loading, threshold enforcement |
| SAPE Engine | src/sape.rs, src/sape_parallel.rs | Rust | 9-probe verification (sequential ~900ms, parallel ~300ms) |
| FATE Escalation | src/fate.rs | Rust | Fail-Safe Agentic Trust Escalation |
| Receipt Schema | src/receipts.rs | Rust | Rejection/Execution/KEP/Autopoietic receipts |
| PCI Gate Chain | src/pci/ | Rust | 10-gate CHEAP→EXPENSIVE rejection chain |
| IFC Taint | src/ifc.rs | Rust | Information flow control |
| Signing | src/signing.rs | Rust | Ed25519 domain-separated receipt signatures |
| Constitutional YAMLs | constitution/*.yaml | YAML | 11 governance files (single source of truth) |
| Crates | crates/ | Rust | finance-v1, bizra-gateway, bizra_bridge |

### 2. Graph Plane (Knowledge)

The graph stores evidence and retrieves context. It supports but never overrides the kernel.

| Component | Path | Language | Purpose |
|-----------|------|----------|---------|
| PostgreSQL + pgvector | docker: postgres:5433 | SQL | Knowledge graph + vector store |
| Neo4j (Wisdom) | src/wisdom.rs, docker: wisdom:7474 | Rust/Cypher | Graph evidence for SAPE probes |
| ChromaDB (Vectors) | src/vectors.rs, docker: ectors:8001 | Rust/HTTP | Embeddings storage |
| Unified Memory | core/unified_memory.py | Python | Multi-tier memory M1-M6 |
| HyperGraphRAG | HyperGraphRAG/ | Python | Hypergraph-structured knowledge retrieval |
| Embeddings | src/embeddings.rs | Rust | fastembed integration |

### 3. Proof Plane (Evidence)

All decisions produce receipts. Receipts are append-only and cryptographically chained.

| Component | Path | Language | Purpose |
|-----------|------|----------|---------|
| Receipt Storage | docs/evidence/receipts/ | JSONL | Append-only receipt lineage |
| Receipt Batching | core/receipt_batching.py | Python | Batch receipt emission |
| PAT Receipt Pipeline | izra_kernel/pat_receipt_pipeline.py | Python | PAT-specific receipt generation |
| Autopoietic Proofs | src/autopoietic/loop_engine.rs | Rust | 11-step self-evolution proof chain |
| Merkle Trees | src/merkle.rs | Rust | Merkle root for proof anchoring |
| Blockchain Anchoring | src/blockchain/ | Rust | Proof chain blockchain anchors |
| Ed25519 Signatures | src/signing.rs | Rust | Receipt signing (domain: izra-receipt-v1:) |
| BLAKE3 Hashing | Cargo dependency | Rust | Fast receipt payload hashing |
| Evidence Sync | core/evidence_sync.py | Python | Cross-service evidence synchronization |
| Gate Evidence | docs/evidence/gates/ | JSON | Gate evaluation snapshots |

### 4. Face Plane (Operator Surface)

The operator surface reveals proof. It never defines or interprets law.

| Component | Path | Language | Purpose |
|-----------|------|----------|---------|
| Rust HTTP API | src/http.rs | Rust | Axum server on port 8080 |
| Python Kernel API | core/main.py | Python | FastAPI server on port 8010 |
| CLI | src/cli/mod.rs | Rust | Clap subcommands (serve, task, status, models, demo) |
| UX Proposal | workspace-ux-proposal/ | TypeScript | Trust Panel, canonical type mirrors |
| Finance Engine | crates/finance-v1/ | Rust | MaaSP pricing (port 3002) |
| Gateway | crates/bizra-gateway/ | Rust | WebSocket/REST bridge for React UI |
| Agentic Flow Bridge | core/agentic_flow_bridge.py | Python | 66-agent swarm integration |
| Constellation | constellation/ | Python | 29-agent multi-expert orchestrator |
| Node0 Identity | NODE0_IDENTITY.yaml | YAML | Sovereign identity |

---

## Canonical Contracts (Frozen)

| Contract | Rust (Law) | Python (Enforcement) | TypeScript (Surface) |
|----------|-----------|---------------------|---------------------|
| MissionEnvelope | src/mission.rs | core/mission.py | workspace-ux-proposal/src/types/bizra.ts |
| GateVerdict | src/mission.rs | core/mission.py | workspace-ux-proposal/src/types/bizra.ts |
| ReceiptArtifact | src/mission.rs | core/mission.py | workspace-ux-proposal/src/types/bizra.ts |
| ManifestArtifact | src/mission.rs | core/mission.py | workspace-ux-proposal/src/types/bizra.ts |

## Gate Order (Frozen)

`
Ingress → State → Proposal → Constitution → Proof → Receipt → Refinement → Reflex
`

## Authority Model (5 Layers — Frozen)

| Layer | Role | Enforcement | Verdicts |
|-------|------|-------------|----------|
| 1 | Defines law | Hard invariants, blocking | PERMIT / REJECT |
| 2 | Interprets | Bounded review, timeout-aware | REVIEW / SCORE_ONLY |
| 3 | Enforces | Kernel admission, fail-closed | PERMIT / REJECT |
| 4 | Experiments | Non-blocking, observability | SCORE_ONLY |
| 5 | Reveals | UI, public proof, trust panel | (read-only) |

**Rule:** Graph is support. Kernel is law. No UI surface may become its own authority centre.

## Killer Product Loop

`
work → proof → memory → reflex → reward → marketplace → sovereignty
`

## Doctrine

> The kernel is physics; the higher constitution is jurisprudence.
> → Layer 1 hard law (decidable, blocking)
> → Layer 2 bounded review (budgeted, timeout-aware)
> → Layer 3 judiciary/advisory (expressive, score-producing)
> → Verdicts: PERMIT | REJECT | REVIEW | SCORE_ONLY
