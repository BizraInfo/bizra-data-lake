# Node0 Kernel Specification — Overview

Last updated: 2026-02-14

## Purpose

This specification formalizes the **Node0 Kernel** — the sovereign execution substrate that powers every BIZRA node. Node0 is not a folder or a service; it is the complete machine identity, from hardware attestation to cognitive reconciliation. The kernel implements a 7-layer architecture where each layer is fail-closed, hash-chained, and formally verifiable.

This document is the entry point. Each layer has its own phase file with pseudocode, TDD anchors, and source-of-truth mappings.

---

## Design Principles

| Principle | Enforcement | Source |
|-----------|-------------|--------|
| **Fail-Closed** | Every gate rejects by default; explicit pass required | Lampson (1971) |
| **Hash-Chain Everything** | Every mutation produces a BLAKE3-chained receipt | Merkle (1979) |
| **Formal Before Execution** | Z3 SMT solver verifies constraints before any commit | Floyd (1967) |
| **Dual Implementation** | Python (dev/integration) + Rust (performance/production) | Defense-in-depth |
| **Constitutional Invariants** | Ihsan >= 0.95, SNR >= 0.85, Gini <= 0.40 — not configurable | `core/integration/constants.py` |
| **Observable** | Every operation emits metrics, receipts, or ledger entries | Shannon (1948) |
| **Sovereign** | Data never leaves the node unless the human explicitly federates | Constitution Article I |

---

## 7-Layer Architecture

```
┌───────────────────────────────────────────────────────────────┐
│ Layer 7: AUDIT SURFACE                                        │
│   REST API · WebSocket · CLI Dashboard · Prometheus Metrics   │
├───────────────────────────────────────────────────────────────┤
│ Layer 6: RECONCILIATION LOOP                                  │
│   AutoResearcher · AutoEvaluator · RecursiveLoop (OODA)      │
│   AutopoieticEngine · SpearPoint Pipeline                    │
├───────────────────────────────────────────────────────────────┤
│ Layer 5: SAT CONSTRAINT ENGINE                                │
│   Z3 SMT Solver · FATE Gate Verification · SAT Controller    │
├───────────────────────────────────────────────────────────────┤
│ Layer 4: MEMORY LEDGER                                        │
│   Tamper-Evident Log · Experience Ledger · Token Ledger      │
│   SQLite (materialized) + JSONL (immutable chain)            │
├───────────────────────────────────────────────────────────────┤
│ Layer 3: POI / FATE ENGINE                                    │
│   4-Stage PoI Scoring · 6-Gate Chain · Ihsan/SNR Thresholds  │
├───────────────────────────────────────────────────────────────┤
│ Layer 2: DETERMINISTIC EXECUTION                              │
│   RuntimeCore · ContractBoundary · SovereignEngine Modes     │
├───────────────────────────────────────────────────────────────┤
│ Layer 1: IDENTITY                                             │
│   Ed25519 Keypair · Genesis Ceremony · Hardware Attestation  │
│   Capability Cards · Post-Quantum Readiness (Dilithium-5)    │
└───────────────────────────────────────────────────────────────┘
```

---

## Layer Summary

| Layer | Phase File | Key Modules (Python) | Key Modules (Rust) | Lines |
|-------|-----------|---------------------|--------------------|----|
| 1. Identity | [phase_01](phase_01_identity.md) | `core/pci/crypto.py`, `core/sovereign/genesis_identity.py` | `bizra-core/src/identity.rs`, `bizra-core/src/genesis.rs` | ~2,200 |
| 2. Execution | [phase_02](phase_02_execution.md) | `core/sovereign/runtime_core.py`, `core/sovereign/engine.py` | `bizra-core/src/sovereign/omega.rs` | ~3,900 |
| 3. PoI/FATE | [phase_03](phase_03_poi_fate.md) | `core/proof_engine/poi_engine.py`, `core/pci/gates.py` | `bizra-core/src/pci/gates.rs`, `fate-binding/` | ~3,100 |
| 4. Ledger | [phase_04](phase_04_ledger.md) | `core/sovereign/tamper_evident_log.py`, `core/token/ledger.py` | `bizra-core/src/sovereign/experience_ledger.rs` | ~3,800 |
| 5-6. Reconciliation + Audit | [phase_05](phase_05_reconciliation.md) | `core/spearpoint/`, `core/sovereign/api.py` | `bizra-cli/` | ~9,600 |
| Boot Sequence | [phase_06](phase_06_boot_sequence.md) | `core/sovereign/launch.py`, `core/sovereign/__main__.py` | `bizra-cli/src/main.rs` | ~1,800 |

**Total kernel surface:** ~24,400 lines (Python + Rust)

---

## Cross-Cutting Concerns

### Constitutional Thresholds (Single Source of Truth)

All thresholds are defined in `core/integration/constants.py` and mirrored in `bizra-omega/bizra-core/src/lib.rs`. No module may define its own thresholds.

```
UNIFIED_IHSAN_THRESHOLD       = 0.95   # Production gate
UNIFIED_SNR_THRESHOLD         = 0.85   # Signal quality floor
UNIFIED_ADL_GINI_THRESHOLD    = 0.40   # Justice ceiling
UNIFIED_IHSAN_STRICT          = 0.99   # Consensus/critical
UNIFIED_IHSAN_CI              = 0.90   # CI tolerance
UNIFIED_CLOCK_SKEW_TOLERANCE  = 120    # Seconds
```

### Cryptographic Stack

| Operation | Algorithm | Domain Prefix | Source |
|-----------|-----------|---------------|--------|
| Content hashing | BLAKE3 | `bizra-pci-v1:` | `core/proof_engine/canonical.hex_digest()` |
| Signatures | Ed25519 | — | `core/pci/crypto.sign_message()` |
| PQ signatures | Dilithium-5 | — | `native/fate-binding/src/dilithium.rs` |
| JSON canonicalization | RFC 8785 | — | `core/pci/crypto.canonicalize_json()` |
| Token hashing | BLAKE3 | `bizra-token-v1:` | `core/token/types.py` |
| Tamper-evident log | HMAC-SHA256 | — | `core/sovereign/tamper_evident_log.py` |

### Dual Implementation Strategy

Critical paths have both Python and Rust implementations:

| Component | Python | Rust | Binding |
|-----------|--------|------|---------|
| Identity | `core/pci/crypto.py` | `bizra-core/src/identity.rs` | PyO3 |
| Gate chain | `core/pci/gates.py` | `bizra-core/src/pci/gates.rs` | PyO3 |
| Experience ledger | `core/sovereign/experience_ledger.py` | `bizra-core/src/sovereign/experience_ledger.rs` | PyO3 |
| SNR engine | `core/iaas/sanitization.py` | `bizra-core/src/sovereign/snr_engine.rs` | PyO3 |
| GoT reasoning | `core/autopoiesis/got_integration.py` | `bizra-core/src/sovereign/graph_of_thoughts.rs` | PyO3 |

Python is the integration layer; Rust is the performance layer. When both exist, Rust is preferred at runtime with Python as fallback.

---

## Standing on Giants

| Giant | Contribution | Kernel Layer |
|-------|-------------|--------------|
| Lampson (1971) | Protection domains, fail-closed access | Layer 2: Contract boundaries |
| Merkle (1979) | Hash chains for tamper detection | Layer 4: All ledgers |
| Lamport (1978) | Logical clocks, Byzantine fault tolerance | Layer 4: Sequence numbers |
| Shannon (1948) | Information theory, SNR measurement | Layer 3: PoI scoring |
| Bernstein (2012) | Ed25519 high-speed signatures | Layer 1: Identity |
| Nakamoto (2008) | Hash-chained ledger, genesis block | Layer 4: Token ledger |
| Al-Ghazali (1058-1111) | Ihsan (excellence), zakat (distributive justice) | Layer 3: Quality gates |
| Floyd (1967) | Assertion-based verification | Layer 5: Z3 FATE gate |
| Boyd (1995) | OODA loop (Observe-Orient-Decide-Act) | Layer 6: Reconciliation |
| Maturana & Varela (1972) | Autopoiesis (self-creating systems) | Layer 6: Autopoietic engine |
| Besta (2024) | Graph-of-Thoughts reasoning | Layer 2: GoT integration |
| Ostrom (1990) | Common-pool resource governance | Layer 3: ADL Gini gate |
| Tulving (1972) | Episodic memory | Layer 4: Experience ledger |
| Kocher (1996) | Timing-safe cryptographic operations | Layer 1: Constant-time compare |

---

## Reading Path

1. **Start here** — This overview
2. **[Phase 01: Identity](phase_01_identity.md)** — Ed25519, genesis ceremony, capability cards
3. **[Phase 02: Execution](phase_02_execution.md)** — RuntimeCore, contract boundaries, engine modes
4. **[Phase 03: PoI/FATE](phase_03_poi_fate.md)** — 4-stage PoI, 6-gate chain, Z3 verification
5. **[Phase 04: Ledger](phase_04_ledger.md)** — Hash-chained ledgers, token system, dual storage
6. **[Phase 05: Reconciliation + Audit](phase_05_reconciliation.md)** — OODA loop, autopoiesis, API surface
7. **[Phase 06: Boot Sequence](phase_06_boot_sequence.md)** — Deterministic startup, service lifecycle

---

*Source of truth: This specification. Implementation: `core/`, `bizra-omega/`, `native/`.*
