# Phase 37 — DDAGI OS v4.0-GENESIS: Implementation Component Mapping

> Maps the v4.0 architecture to concrete technology choices and existing codebase artifacts.

Standing on Giants: Brooks (No Silver Bullet, 1986) + Conway (Org/Arch isomorphism, 1968) + Lamport (Distributed Systems, 1978)

---

## 1. Implementation Components

| Module | Technology | Layer | Responsibility |
|--------|-----------|-------|----------------|
| **Embodiment** | AutoHotkey v2 + Win32 API | L0 | Global hotkeys, UI scraping, file-system control |
| **Bridge** | TCP/JSON-RPC (loopback) | L1 | Secure WSL-Windows conduit |
| **Inference** | BIZRA-7B / LM Studio / Ollama | L2 | Local LLM execution (zero-API dependency) |
| **Reasoning** | Python (Graph-of-Thoughts) | L3 | Non-linear hypothesis exploration |
| **Consensus** | PBFT over Python (libp2p future) | L4 | Distributed agreement on proposals |
| **Governance** | Python (Constitutional Gate) | L5 | Ethical hard constraints |
| **Ledger** | Blake3 + Ed25519 (Python + Rust) | L6 | Cryptographic receipt signing + chain |
| **Runtime** | Rust (Tokio/Axum) + Python | All | Request routing + state machine |
| **Storage** | SQLite (WAL) + HNSW (hnswlib) | L3,L6 | Persistent memory + vector search |

---

## 2. Existing Codebase Artifact Map

### Layer 0: Neural Nervous System

```
ARTIFACT MAP (Layer 0):
  core/bridges/desktop_bridge.py     # TCP bridge endpoint (Python side)
  core/bridges/bridge.py             # Skill routing from bridge commands
  core/bridges/bridge_receipt.py     # HMAC auth + receipt generation

  # AHK-v2 side (Windows-native, not in this repo):
  # C:\BIZRA-OS\ahk\bizra_ahk.ahk   # Global hotkeys + perception
  # C:\BIZRA-OS\ahk\screen_reader.ahk # Pixel-level screen capture

STATUS: Bridge server active, auth hardened (SAPE session).
        AHK scripts exist in BIZRA-OS repo.
GAP:    No unified AHK<->Python type contract. Needs IDL or schema.
```

### Layer 1: Sovereign Bridge

```
ARTIFACT MAP (Layer 1):
  core/bridges/desktop_bridge.py:953  # Server startup (TCP)
  core/bridges/desktop_bridge.py:992  # Health check endpoint
  docker-compose.yml:100-120          # Compose service config
  core/bridges/bridge_receipt.py:44   # Receipt signing

STATUS: Hardened in this session (SAPE-003).
        Required env vars: BIZRA_BRIDGE_TOKEN, BIZRA_RECEIPT_PRIVATE_KEY_HEX
        Healthcheck includes auth headers.
GAP:    Protocol version negotiation not implemented.
        Heartbeat loop not implemented (spec calls for 15s interval).
```

### Layer 2: Intelligence Core (RDVE)

```
ARTIFACT MAP (Layer 2):
  core/spearpoint/auto_researcher.py       # Hypothesis generation
  core/spearpoint/recursive_loop.py        # RDVE iteration controller
  core/spearpoint/auto_evaluator.py        # Verdict gate
  core/spearpoint/benchmark_dominance.py   # SOTA tracking
  core/spearpoint/ablation_engine.py       # Component ablation
  core/spearpoint/pattern_selector.py      # System 1 cache

  # Inference backends:
  core/inference/backends/lmstudio_backend.py   # LM Studio (primary)
  core/inference/backends/ollama_backend.py      # Ollama (fallback)
  core/inference/tiered_gateway.py               # Tier routing

STATUS: Full RDVE loop operational. 3-tier inference (LM Studio > Ollama > Cloud).
GAP:    Entropy Router exists conceptually in pattern_selector.py but
        not formalized as System 1/2 decision boundary.
        No explicit complexity/reversibility/stakes scoring.
```

### Layer 3: Cognitive Backbone (GoT)

```
ARTIFACT MAP (Layer 3):
  core/reasoning/graph_core.py               # Node/Edge primitives
  core/reasoning/graph_operations.py         # Expand/Aggregate/Refine
  core/reasoning/graph_reasoner.py           # GoT orchestrator
  core/reasoning/graph_types.py              # Type definitions
  core/reasoning/bicameral_engine.py         # Dual-hemisphere reasoning
  core/reasoning/collective_intelligence.py  # Multi-agent synthesis
  core/reasoning/snr_maximizer.py            # Signal quality optimizer

  # HyperGraph RAG (Phase 33):
  core/hypergraph_rag/engine.py              # Multi-hop retrieval
  core/hypergraph_rag/hypergraph.py          # Hyperedge storage

  # V3 AgentDB Memory:
  core/memory/agent_db.py                    # Unified facade
  core/memory/hnsw_index.py                  # HNSW vector search
  core/memory/hybrid_query.py                # Score fusion engine

STATUS: GoT fully operational (Phase 33). HyperGraph RAG complete.
        AgentDB with HNSW indexing live (V3 Memory Unification).
GAP:    GoT pruning strategy uses fixed 60% cutoff — should be adaptive
        based on graph density and convergence velocity.
```

### Layer 4: SAT-49 Verification

```
ARTIFACT MAP (Layer 4):
  core/federation/consensus.py               # PBFT state machine
  core/federation/gossip.py                  # Peer discovery
  core/federation/protocol.py                # Message format
  core/federation/secure_transport.py        # DTLS transport
  core/reasoning/guardian_council.py         # Multi-agent review
  core/sovereign/constitutional_gate.py      # Admission control
  core/pci/crypto.py                         # Ed25519 + Blake3

STATUS: PBFT consensus engine operational (Phase 35).
        Guardian Council active with 5 validators (SAT-5).
GAP:    SAT-49 spec calls for 49 departments, current impl has 5.
        Need to scale from SAT-5 to SAT-49 with department taxonomy.
        libp2p transport not implemented (using TCP).
        View-change timeout not tuned for production.
```

### Layer 5: FATE Gate

```
ARTIFACT MAP (Layer 5):
  core/integration/constants.py:86           # UNIFIED_IHSAN_THRESHOLD = 0.95
  core/integration/constants.py:122          # UNIFIED_SNR_THRESHOLD = 0.85
  core/integration/constants.py:93           # STRICT_IHSAN_THRESHOLD = 0.99
  core/sovereign/constitutional_gate.py      # Z3-proven admission
  core/governance/adaptive_ihsan.py          # Dynamic threshold
  core/governance/constitutional_gate.py     # Re-export wrapper
  core/pci/gates.py                          # PCI gate chain
  core/governance/ihsan_vector.py            # Multi-dimensional Ihsan

STATUS: Ihsan constraint enforced across all modules.
        Constitutional gate uses Z3 theorem prover for formal verification.
        Constants are SSOT (v2.2.2).
GAP:    Ihsan weight vector {C:0.30, S:0.30, E:0.15, B:0.25} not
        configurable — hardcoded in spec. Should be in constants.py.
        Daughter Test not implemented as code (currently documentation only).
```

### Layer 6: Evidence Ledger

```
ARTIFACT MAP (Layer 6):
  core/sovereign/experience_ledger.py        # SEL (hash-chained)
  core/proof_engine/evidence_ledger.py       # Evidence receipts
  core/proof_engine/canonical.py             # Canonical JSON + hex_digest
  core/pci/crypto.py                         # Ed25519 sign/verify
  core/pci/envelope.py                       # PCI envelope format
  core/bridges/bridge_receipt.py             # Bridge-specific receipts

  # Rust implementation:
  bizra-omega/bizra-proofspace/              # Proof verification crate

STATUS: Hash-chained evidence ledger operational.
        Ed25519 signing active via PCI crypto.
        SEL is read-only in AgentDB adapter (V3 memory).
GAP:    Merkle tree (BlockGraph) not implemented — current impl is
        linear hash chain. Need DAG structure for concurrent branches.
        No chain compaction/archival strategy for growth beyond 10M entries.
```

---

## 3. Gap Analysis Summary

| Gap ID | Layer | Severity | Description | Effort |
|--------|-------|----------|-------------|--------|
| G-01 | L0 | MEDIUM | No AHK<->Python IDL/schema contract | 2 days |
| G-02 | L1 | LOW | Protocol version negotiation missing | 1 day |
| G-03 | L1 | MEDIUM | Heartbeat loop not implemented | 1 day |
| G-04 | L2 | HIGH | Entropy Router not formalized | 3 days |
| G-05 | L3 | LOW | GoT pruning cutoff is static (60%) | 1 day |
| G-06 | L4 | HIGH | SAT-5 -> SAT-49 scaling | 5 days |
| G-07 | L4 | MEDIUM | libp2p transport not implemented | 5 days |
| G-08 | L5 | LOW | Ihsan weights not in constants.py | 0.5 day |
| G-09 | L5 | MEDIUM | Daughter Test not coded | 2 days |
| G-10 | L6 | HIGH | Linear chain -> Merkle DAG | 5 days |
| G-11 | L6 | LOW | No chain compaction strategy | 2 days |

### Priority Order (by impact)

1. **G-04** Entropy Router — Prevents deadlock-of-caution
2. **G-06** SAT-49 scaling — Core consensus architecture
3. **G-10** Merkle DAG — Enables concurrent evidence branches
4. **G-09** Daughter Test — Ethical governance completeness
5. **G-07** libp2p transport — Federation readiness

---

## 4. Dependency Graph

```
                    ┌─────────────┐
                    │  G-04       │ Entropy Router
                    │  (Layer 2)  │
                    └──────┬──────┘
                           │ requires
                    ┌──────▼──────┐
                    │  G-06       │ SAT-49 Scaling
                    │  (Layer 4)  │ (uses entropy for spot-check)
                    └──────┬──────┘
                           │ requires
              ┌────────────┼────────────┐
              │                         │
     ┌────────▼──────┐        ┌────────▼──────┐
     │  G-07         │        │  G-10         │
     │  libp2p       │        │  Merkle DAG   │
     │  (Layer 4)    │        │  (Layer 6)    │
     └───────────────┘        └───────────────┘
              │                         │
              └─────────┬───────────────┘
                        │
               ┌────────▼────────┐
               │  G-09           │ Daughter Test
               │  (Layer 5)      │ (can be parallel)
               └─────────────────┘
```

---

## 5. Technology Decision Record

### TDR-001: Why AHK-v2 over Playwright/Puppeteer?

| Criterion | AHK-v2 | Playwright |
|-----------|--------|------------|
| Win32 native access | Full (SendInput, BitBlt) | Browser only |
| Latency | <10ms | 50-200ms |
| Non-browser apps | Yes (any Win32 window) | No |
| GPU screen capture | BitBlt + DXGIOutputDuplication | Not applicable |
| Footprint | ~2MB binary | ~200MB + browser |

**Decision**: AHK-v2 for Layer 0. Playwright reserved for web-specific testing.

### TDR-002: Why TCP/JSON-RPC over WebSocket?

| Criterion | TCP/JSON-RPC | WebSocket |
|-----------|-------------|-----------|
| Protocol overhead | Minimal (newline-delimited) | HTTP upgrade + framing |
| Reliability | Deterministic (no browser runtime) | Browser-dependent |
| Loopback binding | Native | Requires HTTP server |
| Debugging | `nc 127.0.0.1 9742` | Needs WS client |
| Latency | <1ms loopback | 2-5ms (HTTP overhead) |

**Decision**: TCP/JSON-RPC for Layer 1. WebSocket reserved for browser UI.

### TDR-003: Why PBFT over Raft?

| Criterion | PBFT | Raft |
|-----------|------|------|
| Byzantine tolerance | Yes (f of 3f+1) | No (crash-fault only) |
| Malicious actors | Handles | Cannot handle |
| Message complexity | O(n^2) | O(n) |
| Scaling ceiling | ~100 nodes | ~1000 nodes |
| Use case fit | Adversarial P2P | Trusted cluster |

**Decision**: PBFT for Layer 4. Raft is insufficient for untrusted DDAGI network.
For SAT-49 (49 nodes), O(n^2) = 2,401 messages per round — acceptable.

### TDR-004: Why Blake3 over SHA-256?

| Criterion | Blake3 | SHA-256 |
|-----------|--------|---------|
| Speed (single-thread) | 1.8 GB/s | 0.3 GB/s |
| Parallelism | SIMD + tree hashing | Sequential |
| Security margin | 256-bit, BLAKE family | 256-bit, SHA-2 |
| Rust ecosystem | `blake3` crate (rayon) | `sha2` crate |
| Determinism | Cross-platform | Cross-platform |

**Decision**: Blake3 for all hashing in Layer 6. Already used in PCI crypto.

---

## 6. Runtime Architecture

```
MODULE RuntimeOrchestrator:

  STRUCT SystemState:
    layers: Dict[int, LayerStatus]
    uptime_s: int
    node_id: str
    mode: ThinkingMode              # Current System 1/2 mode
    active_proposals: int           # In-flight SAT-49 votes
    chain_depth: int                # Evidence ledger depth

  FUNCTION boot_sequence():
    """Deterministic startup order. Each layer depends on the one below."""
    # Phase 1: Foundation
    layer6 = init_evidence_ledger()       # Ledger first (records everything)
    layer5 = init_fate_gate(layer6)       # Governance needs ledger
    layer4 = init_sat49(layer5, layer6)   # Consensus needs governance + ledger

    # Phase 2: Intelligence
    layer3 = init_got_engine()            # GoT is self-contained
    layer2 = init_rdve(layer3, layer4)    # RDVE needs GoT + consensus

    # Phase 3: Interface
    layer1 = init_bridge()                # Bridge is transport only
    layer0 = init_ahk_perception()        # AHK connects through bridge

    # Record boot receipt
    layer6.append_receipt(
      action="SYSTEM_BOOT",
      ihsan=1.0, snr=1.0,
      consensus="GENESIS"
    )

    RETURN SystemState(
      layers={0: layer0, 1: layer1, 2: layer2,
              3: layer3, 4: layer4, 5: layer5, 6: layer6},
      mode=SYSTEM_1,
      chain_depth=1
    )

  FUNCTION shutdown_sequence():
    """Reverse order. Drain in-flight proposals before stopping."""
    # Drain Layer 4 proposals (30s timeout)
    drain_proposals(timeout_s=30)

    # Stop accepting new inputs
    layer0.stop()
    layer1.stop()

    # Flush and persist
    layer2.flush_cache()
    layer3.persist_graph()
    layer6.flush_wal()

    # Record shutdown receipt (last entry)
    layer6.append_receipt(
      action="SYSTEM_SHUTDOWN",
      ihsan=1.0, snr=1.0,
      consensus="ORDERLY"
    )

  # TDD ANCHOR: test_boot_sequence_layers_in_order
  # TDD ANCHOR: test_boot_records_genesis_receipt
  # TDD ANCHOR: test_shutdown_drains_proposals
  # TDD ANCHOR: test_shutdown_records_final_receipt
  # TDD ANCHOR: test_layer_dependency_enforced
```

---

## 7. TDD Anchor Summary

| Module | Test Count | Key Assertion |
|--------|-----------|---------------|
| RuntimeOrchestrator | 5 | Boot order, genesis receipt, shutdown |
| Gap Analysis | 0 | Documentation only |
| TDR decisions | 0 | Documentation only |
| **Total** | **5** | |
