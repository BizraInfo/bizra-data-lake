# Module 01 — Constitutional Layer

> **Domain:** FATE gates, thresholds, kernel invariants, proof-carrying inference
> **Source Specs:** Phase 0-3 (foundations), Phase 56 (SAPE engine), Phase 60 (boundary), Phase 72 (kernel)
> **Rust Mirror:** `bizra-omega/bizra-core/`, `bizra-omega/fate-binding/`

## 1.1 Constitutional Thresholds (Single Source of Truth)

**Status:** [x] BUILT
**Path:** `core/integration/constants.py`

All thresholds defined once, imported everywhere. Cross-language sync validated in CI
(`Cross-Language Sync` job checks Python vs Rust parity).

| Constant | Value | Purpose |
|----------|-------|---------|
| `UNIFIED_IHSAN_THRESHOLD` | 0.95 | Production gate |
| `STRICT_IHSAN_THRESHOLD` | 0.99 | Consensus gate |
| `RUNTIME_IHSAN_THRESHOLD` | 1.0 | Apoptosis safeguard |
| `UNIFIED_SNR_THRESHOLD` | 0.85 | Minimum quality floor |
| `SNR_THRESHOLD_T1` | 0.95 | T1 expert tier |
| `SNR_THRESHOLD_T0_ELITE` | 0.98 | T0 elite tier |
| `ADL_GINI_THRESHOLD` | 0.35 | Justice invariant |
| `ADL_HARBERGER_TAX_RATE` | 0.05 | Annual wealth tax |
| `TOKEN_ZAKAT_RATE` | 0.025 | Computational zakat |
| `CONFIDENCE_HIGH` | 0.95 | Evaluator gate |
| `FAISS_SIMILARITY_FLOOR` | 0.35 | Retrieval cutoff |
| `GOT_MAX_HYPOTHESES` | 5 | Planner branching |

**Tests:** `tests/core/integration/` — threshold import, cross-repo sync

---

## 1.2 FATE Gates (Fairness, Accountability, Transparency, Ethics)

**Status:** [x] BUILT
**Path:** `core/pci/gates.py` — `PCIGateKeeper` class

FATE gates are the constitutional firewall. Every inference, agent action, and economic
transaction passes through FATE before execution. Default-deny semantics.

**Key classes:**
- `PCIGateKeeper` — orchestrates gate chain (core/pci/gates.py)
- `FATEEvaluator` — individual FATE dimension scoring (core/pci/fate.py)
- `_conservative_fallback_check()` — default-deny when Z3 unavailable

**Rust mirror:** `bizra-omega/fate-binding/` — Z3 formal verification + Dilithium post-quantum signatures

**Tests:** `tests/core/pci/` — gate pass/fail, fallback behavior, Z3 integration

---

## 1.3 Proof-Carrying Inference (PCI)

**Status:** [x] BUILT
**Path:** `core/pci/`

Every inference carries a cryptographic proof of its derivation chain. Proofs are
append-only, hash-chained, and auditable.

**Components:**
- `core/pci/gates.py` — PCIGateKeeper orchestration
- `core/pci/proof.py` — proof construction and verification
- `core/pci/verifier.py` — proof chain validation
- `core/pci/types.py` — proof type definitions

**Tests:** `tests/core/pci/` — 17+ test files mirroring module structure

---

## 1.4 Evidence Ledger (Hash-Chained Receipts)

**Status:** [x] BUILT
**Path:** `core/proof_engine/evidence_ledger.py`

Append-only ledger with hash-chained entries. Each entry contains action, result,
timestamp, and cryptographic link to previous entry.

**API:**
```
EvidenceLedger(path=, validate_on_append=True)
.append(receipt={...})        # receipt dict with reason_codes
.verify_chain() -> (bool, List[str])  # sequence + hash verification
```

**Tests:** `tests/core/proof_engine/` — chain integrity, append validation, chaos tests

---

## 1.5 BLAKE3 Canonical Hashing

**Status:** [x] BUILT
**Path:** `core/proof_engine/canonical.py`

BLAKE3 is the canonical hash function. Legacy SHA-256 usage marked with `# noqa: SEC-001`.
Rust side uses `blake3` crate with rayon parallelism.

**Tests:** Hash consistency, cross-language parity

---

## 1.6 Constitutional Gate (Governance)

**Status:** [x] BUILT
**Path:** `core/governance/constitutional_gate.py`

Higher-level governance gate that wraps FATE + Ihsan + SNR checks into a single
constitutional pass/fail decision. Used by SovereignRuntime before any state mutation.

**Tests:** `tests/core/governance/`

---

## 1.7 Apoptosis Safeguard

**Status:** [x] BUILT
**Path:** `core/integration/constants.py` (`RUNTIME_IHSAN_THRESHOLD = 1.0`)
**Implementation:** Checked in SovereignRuntime — if Ihsan drops below 1.0 at runtime,
the node self-terminates rather than operate in degraded ethical state.

---

## 1.8 Constitutional Simulation

**Status:** [x] BUILT
**Path:** `core/constitutional/simulation.py`

Simulation harness for testing constitutional invariants under adversarial conditions.
BLOOM linear accrual regression test included.

**Tests:** `tests/constitutional/test_simulation.py`

---

## 1.9 SAPE Framework (SNR-Anchored Performance Evaluation)

**Status:** [x] BUILT
**Path:** `core/apex/snr_apex_engine.py`, `core/iaas/snr_v2_adapter.py`

Composite scoring: SNR + Ihsan + coverage + security. Quality gate in CI
(SAPE-003 composite = 0.704, soft-gated due to signing keys).

---

## 1.10 Z3 Formal Verification Binding

**Status:** [x] BUILT
**Path:** `bizra-omega/fate-binding/`

Rust crate binding Z3 solver for formal FATE verification. Post-quantum signatures
via Dilithium (pqcrypto-mldsa). Falls back to conservative check when Z3 unavailable.

**Deps:** `z3` (system lib), `pqcrypto-mldsa`
**Tests:** `bizra-omega/fate-binding/tests/`

---

## 1.11 Cross-Language Constant Sync

**Status:** [x] BUILT
**Path:** `.github/workflows/ci.yml` (Cross-Language Sync job)

CI gate validates that Python `constants.py` thresholds match Rust `bizra-core` constants.
Prevents constitutional drift between language runtimes.

---

## 1.12 Constitutional Kernel CLI

**Status:** [x] BUILT
**Path:** `core/constitutional/__main__.py`

CLI entry point for running constitutional checks standalone.

---

## 1.13 Ihsan Scoring Engine

**Status:** [x] BUILT
**Path:** Distributed across `core/apex/`, `core/iaas/`, `core/governance/`

Ihsan (excellence) scoring computed from multiple dimensions: correctness, completeness,
ethical compliance, SNR quality. Used as universal gate.

---

## 1.14 ADL Justice Invariant (Gini Gate)

**Status:** [x] BUILT
**Path:** `core/integration/constants.py` (ADL_GINI_THRESHOLD = 0.35)
**Enforcement:** Token system simulates post-transaction Gini coefficient, rejects if
> 0.35 AND transaction increases concentration. Genesis mint exempt.

**Tests:** Token minting tests verify Gini gate enforcement

---

## Completion

| Feature | Status | Coverage |
|---------|--------|----------|
| 1.1 Thresholds | BUILT | CI-enforced |
| 1.2 FATE Gates | BUILT | 17+ tests |
| 1.3 PCI | BUILT | Full |
| 1.4 Evidence Ledger | BUILT | Chain + chaos |
| 1.5 BLAKE3 | BUILT | Cross-lang |
| 1.6 Constitutional Gate | BUILT | Full |
| 1.7 Apoptosis | BUILT | Runtime check |
| 1.8 Simulation | BUILT | Regression |
| 1.9 SAPE | BUILT | CI gate |
| 1.10 Z3 Binding | BUILT | Rust tests |
| 1.11 Cross-Lang Sync | BUILT | CI gate |
| 1.12 Kernel CLI | BUILT | Smoke |
| 1.13 Ihsan Scoring | BUILT | Full |
| 1.14 ADL Gini Gate | BUILT | Token tests |
| **TOTAL** | **14/14** | **100%** |
