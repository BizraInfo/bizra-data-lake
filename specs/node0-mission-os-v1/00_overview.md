# Node0 Mission OS v1 — Specification Overview

**Status:** [ENFORCEMENT: WIRED]
**SNR Score:** 8.8/10 → Target: 9.5+
**Ihsan Alignment:** 91.5% → Target: 95%+
**Sprint Duration:** 14 days (4 phases)
**Date:** 2026-03-28

## Core Runtime Law

```
Mission → Proof → Receipt → Refinement → Reflex → Trust
```

Every action must be preceded by admissibility verification.
Every decision must produce immutable evidence.
Every learning must be bounded by constitutional constraints.

## Phase Map

| Phase | Days | Objective | Ihsan Gate |
|-------|------|-----------|------------|
| 0: Symbolic Freeze | 1-2 | Lock invariants, remove fallbacks | Adl (consistency) |
| 1: Layer Unification | 3-7 | Cross-layer contracts, reflex default-live | Excellence |
| 2: Probe Rarely Fired | 8-10 | Fail-closed under all adverse conditions | Amanah (completeness) |
| 3: Elevation | 11-14 | Public proof artifact, v1.0.0 release | Benevolence (transparency) |

## Five Critical Gaps (from Blueprint)

| # | Gap | Severity | Phase |
|---|-----|----------|-------|
| G1 | Cross-language sealing (Rust/Python digest mismatch) | HIGH | Phase 0 |
| G2 | Dilithium fallback returns `true` for any signature | CRITICAL | Phase 0 |
| G3 | Reflex default-live feature-flagged | MEDIUM | Phase 1 |
| G4 | Redis secret mismatch | **FIXED** (2026-03-28) | Done |
| G5 | Documentation overclaims | MEDIUM | Phase 3 |

## Four Cross-Layer Contracts

| Contract | Purpose | Fields |
|----------|---------|--------|
| MissionEnvelope | Canonical cross-layer mission object | mission_id, initiator_id, payload, constitutional_context, timestamps, canonical_hash |
| GateVerdict | Authoritative gate result | gate_status, proof_status, ihsan_status, snr_status, reason_codes, policy_bundle_version |
| ReceiptArtifact | Evidence for every evaluated transition | state_hashes, signatures, lineage |
| ManifestArtifact | Bundle summary for review | receipt_references, integrity_hashes |

## Authority Hierarchy

```
Layer 1 defines law    → Rust omega (constitutional core)
Layer 2 interprets     → Python sovereign cognition
Layer 3 enforces       → .bizra-kernel (runtime bridge)
Layer 4 experiments    → Runtime prototypes
Layer 5 reveals        → Operator surface
```

## Success Criteria (All Must Be True)

1. Golden-vector CI passes with identical Rust/Python digests
2. Dilithium fallback removed with error receipt on native failure
3. Redis bound to localhost with aligned secret name
4. IHSAN_THRESHOLD and ADL_GINI enforced across layers
5. Four cross-layer contracts implemented in mission flow
6. Reflex default-live with correct status semantics
7. 24-hour heartbeat runs without failure
8. Evidence bundle exports with all required artifacts
9. All SAPE probes pass in CI
10. Public repo updated with truth-label matrix
11. v1.0.0 tagged with evidence bundle
