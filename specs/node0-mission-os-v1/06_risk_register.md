# Risk Register — Node0 Mission OS v1

**Status:** [ENFORCEMENT: WIRED]

## Active Risks

| ID | Risk | Probability | Impact | Phase | Mitigation | Owner |
|----|------|-------------|--------|-------|------------|-------|
| R1 | Cross-language sealing fails | Medium | High | 0 | Golden-vector CI catches early; dedicated engineering | Eng |
| R2 | Reflex promotion over-promises | Medium | High | 1 | Strict 3-execution provenance; promote only after probes | Eng |
| R3 | Performance regression | Medium | Medium | 2 | Benchmark suite in CI; 10% regression gate | CI |
| R4 | Documentation-runtime drift | High | Medium | 3 | Truth labels mandatory; evidence bundle forces sync | Docs |
| R5 | Fallback vulnerability persists | Low | Critical | 0 | Code review + security scan; Phase 0 removal | Sec |
| R6 | Reviewer confusion | Medium | Medium | 3 | Evidence bundle includes replay instructions + demo | PM |
| R7 | 24h heartbeat failure | Low | High | 2 | Auto-restart with evidence preservation; root cause analysis | Ops |

## Mitigated Risks (Resolved This Session)

| ID | Risk | Resolution | Date |
|----|------|------------|------|
| R8 | Redis auth mismatch | Added `BIZRA_REDIS_PASS` to secrets file | 2026-03-28 |
| R9 | Ed25519 key on NTFS | Key already on ext4 at `/root/.bizra-keys/` | 2026-03-28 |
| R10 | Wildcard permissions | Confirmed false alarm — 75 explicit patterns | 2026-03-28 |
| R11 | Clippy CI failure | Fixed `too_many_arguments` + `missing_safety_doc` | 2026-03-28 |

## Noise Items (Overclaims to Correct)

| Claim | Current Label | Required Action |
|-------|--------------|-----------------|
| "Kernel-level halt" | OVERCLAIM | Correct to "Python API boundary enforcement" |
| "800M TPS planetary scale" | OVERCLAIM | Correct to "validated in EV&V environment" |
| "HHMM/diffusion reasoning amplifier" | OVERCLAIM | Correct to PLANNED or remove |
| "Physically impossible security" | OVERCLAIM | Correct to "fail-closed membrane" |

These overclaims MUST be corrected in Phase 3 truth-label audit.
