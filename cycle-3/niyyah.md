# Cycle 3 — Phase 1: NIYYAH (نية) — Intent Declaration

**Cycle:** 3
**Date:** 2026-04-15
**Predecessor:** Cycle 2 (chain hash `4312035fb50254c860c5f6b55b4c3456802e0c7617f32c0a59295e266e4ab9ee`)

---

## WHAT

Canonicalize the **Constitutional Hygiene** subsystem — the drift-detection-and-closure methodology
that keeps implementation and test expectations synchronized across Python, Rust, and TypeScript.

Three commits represent this work:
1. `34eb09a0` — close cross-language constant drift surface (IHSAN_THRESHOLD, ADL_GINI, ADL_HARBERGER, MIN_CONFIDENCE, MAX_HARM_SCORE)
2. `f558e228` — close SNR_FLOOR drift + codify workspace sweep
3. `19260543` — sync heartbeat test truth_label assertions with DEFAULT-LIVE reflex path

## WHY

Drift between implementation and tests has occurred three separate times (IHSAN_THRESHOLD, SNR_FLOOR,
heartbeat labels). Each time, the implementation evolved correctly but test expectations lagged —
creating a false-failure that masks real regressions. This is the highest-frequency class of
constitutional violation: not malice, but neglect. The methodology that detects and closes this
drift is itself a subsystem worth canonicalizing.

**Constitutional trace:** Root Invariant 4 (Ihsan ≥ 0.95) → The system must be honest about its own
state. Stale test expectations violate Ihsan because they create a false picture of system health.
Al-Ghazali (honest labeling, 1096) — label what is proven vs. what is partial.

## SUCCESS_CONDITION

1. All 6 Tier-1 constitutional constants verified identical across Python + Rust (re-verification)
2. All test suites GREEN: constitutional (296+), core (3142+), Rust (79+)
3. Zero frozen-anchor violations
4. The drift-detection methodology documented as a repeatable procedure
5. BLAKE3 receipt chained from Cycle 2 hash
6. Subsystem status elevated from TESTED → PROVEN in TOPOLOGY_CANON.md
