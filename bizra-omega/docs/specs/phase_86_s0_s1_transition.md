# Phase 86: S0→S1 Transition Specification
## From Proven Architecture to Self-Sustaining Loop

**Document**: BIZRA-PHASE86-SPEC-v1.0
**Date**: 2026-03-17
**Evidence Base**: Phase 84-85 audit (4,197 tests GREEN), 6 strategic documents, 4-agent convergent analysis
**Authority**: Quran → Hadith → Enforceable Spine v1.1 → This Spec → Code
**Constitutional**: IHSAN >= 0.95 · GINI <= 0.35 · SNR >= 0.85

---

## 1. HHMM State Context

```
S0 (Proven Architecture)  ──gate──>  S1 (Self-Sustaining Loop)
                                          │
Gate requires exactly 3:                  │
  [x] B1: Persistence (Phase 85)         │
  [ ] B2: Ed25519 receipt signing         │ ← THIS SPEC
  [ ] D:  4-loop HHMM wiring             │ ← THIS SPEC
                                          │
S1 gate proof:                            │
  NODE0 runs 24h continuous               │
  Each task → receipt → bus → memory      │
  Self-sustaining loop PROVEN             │
```

## 2. Scope (3 Workstreams)

| ID | Workstream | Sprint Ref | Acceptance Gate |
|----|-----------|-----------|-----------------|
| 86-A | Ed25519 receipt signing | B2 | `verify()` returns True for valid sigs, False for tampered |
| 86-B | 4-loop HHMM EventBus wiring | D1-D4 | Heartbeat events reach all 13 subscribers end-to-end |
| 86-C | Deployment verification | Horizon 1 | Release binary processes 100 governed missions, 0 crashes |

## 3. Dependencies (Cascading Order)

```
86-A (signing) must complete before 86-C (deployment)
  └─ Receipts must be signed before 24h continuous run
86-B (wiring) must complete before 86-C (deployment)
  └─ Bus must propagate before self-sustaining loop
86-A and 86-B are independent — can parallelize
```

## 4. Constitutional Constraints

- No receipt without signature (ZANN_ZERO / CLAIM_MUST_BIND)
- No silent failures in bus wiring (advance! pattern from Phase 85)
- All new code must pass clippy with zero warnings
- Every new module + dependents in SAME commit
- Ed25519 keys from `ed25519-dalek` (already in workspace deps)
- Hash algorithm: BLAKE3 only (Spine v1.1 Amendment A6)

## 5. Files Affected (Estimated)

```
86-A: Ed25519 Signing
  bizra-mission/src/receipt.rs       — add sign() + verify_signature()
  bizra-mission/Cargo.toml           — add ed25519-dalek dep
  bizra-node/src/node.rs             — hold signing key in Node
  bizra-node/src/mission_bridge.rs   — sign receipt after emission
  bizra-node/src/lib.rs              — integration tests

86-B: 4-Loop HHMM Wiring
  bizra-hooks/src/subscribers.rs     — verify all 13 subscribers
  bizra-hooks/src/event_bus.rs       — async heartbeat bridge
  bizra-node/src/node.rs             — bus.start() in heartbeat cycle
  bizra-node/src/lib.rs              — integration tests

86-C: Deployment Verification
  bizra-omega/scripts/ops/smoke_test.sh  — 100-mission binary test
```

## 6. Risk Analysis

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Ed25519 key management complexity | Medium | High | Use ephemeral test keys; production keys from genesis ceremony |
| Bus wiring breaks existing tests | Low | High | Run full 1,381 test suite after each loop |
| 24h run reveals memory leak | Medium | Medium | Monitor RSS via HEALTH command at intervals |
| Signing overhead degrades p95 latency | Low | Low | Ed25519 sign is ~50μs — negligible vs 10ms mission |
