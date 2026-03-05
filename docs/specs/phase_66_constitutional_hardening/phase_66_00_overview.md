# Phase 66: Constitutional Hardening Sprint

## Context

SPARC integration review (2026-03-05, commit `ba3b6b3`) scored the system at
composite SNR 0.83 — above the 0.85 diagnostic floor but below the 0.95
operational-grade threshold. The gap is **not in capabilities but in wiring
discipline**: thresholds duplicated across 11 files, audit trails broken by
silent `except: pass`, and hot-path functions missing trivial caching.

The HHMM state model predicts the system is in ORGANIZING → ANALYZING
transition. This sprint completes the transition to CREATING by sealing
the 5 highest-ROI defects identified across 4 analysis agents.

## Axiom Grounding (from Peak Hidden Thoughts Analysis)

> "If a constraint can be violated at runtime, it is not a constraint —
>  it is a suggestion."

Every fix in this sprint converts a **suggestion** into a **constraint**:
- Duplicated thresholds → single import (compile-time truth)
- Silent `except: pass` → logged failure (audit-time truth)
- Uncached pure functions → memoized (performance truth)
- Blocking sync in async → executor-wrapped (correctness truth)
- N+1 commits → batch transaction (efficiency truth)

## Spec Modules

| # | Spec File | Focus | Files Changed | Lines |
|---|-----------|-------|---------------|-------|
| 01 | `phase_66_01_threshold_canonicalization.md` | Single source of truth for all thresholds | 11 | ~30 |
| 02 | `phase_66_02_audit_trail_integrity.md` | Eliminate silent swallows in safety paths | 6 | ~20 |
| 03 | `phase_66_03_performance_hardening.md` | SNR cache, async urllib, N+1 batch | 4 | ~25 |
| 04 | `phase_66_04_tdd_anchors.md` | Test specifications for all changes | 3 new | ~120 |

## Verification Criteria

1. `pytest tests/` — zero regressions (8,500+ tests pass)
2. `ruff check core/` — no new lint violations
3. `grep -rn "= 0.35" core/ | grep -v constants.py | grep -v test` → 0 results
4. `grep -rn "except.*:$" core/sovereign/mission.py` → 0 bare except blocks
5. Benchmark: SNR scoring latency reduced by >50% on repeated inputs

## Standing on Giants

- Shannon (1948): SNR as measurable, cacheable signal
- Lamport (1978): Single source of truth for distributed constants
- Deming (1950): Fix the highest-variance process first (PDCA)
- Boyd (1976): Smallest action, fastest OODA loop
- Al-Ghazali (1095): Ihsan — excellence is the floor, not the ceiling
