# Phase 60: Constitutional Boundary Sprint — Overview

## Standing on Giants: Shannon (SNR) | Lamport (Boundary Contracts) | Al-Ghazali (Ihsan) | Dijkstra (Fail-Closed) | Besta (Graph-of-Thoughts)

**Date:** 2026-03-03 | **SAPE Composite:** 0.93 → Target 0.96
**Scope:** 7 steps closing boundary gaps identified by SAPE multi-lens audit
**Risk:** Medium (steps 1-4, 6 are additive; step 5 is refactor; step 7 is new)
**Prerequisite:** Phase 59 Consolidation Sprint (SNR normalization, constants artifact, Ihsan 6-dim)

## Graph of Thoughts — Dependency Map

```
Node A: Ontology (Souls)          ← Ihsan, Adl, Amanah axioms
    │ HONESTY
    ↓
Node B: Architecture (Skeleton)   ← 7-layer stack, constitution.toml, evidence spine
    │ DETERMINISM
    ↓
Node C: Operations (Muscle)       ← CI gates, evidence accumulation, health tiering
    │ REPRODUCIBILITY
    ↓
Node D: Verification (Immune)     ← Z3 SMT2, conformance topology, SNR contract
```

**Edge Properties:**
- A→D (Honesty): Every theorem cites evidence — Z3 proofs bind axioms to code
- B→C (Determinism): Build output identical across environments — evidence.json is deterministic
- C→D (Reproducibility): Evidence footprints immutable — hash-chained, timestamped

## SNR Framework Application

| Category | Signal (Keep) | Noise (Filter) |
|----------|---------------|-----------------|
| Architecture | constitution.toml externalizes hardcoded gates | Nix flake (deferred — infrastructure for Alpha-100) |
| Security | URP auth normalization is already done | HSM procurement (blocked by hardware) |
| Performance | Health tiering gives O(1) liveness | Full pipeline parallelization (already implemented) |
| Contracts | SNR canonical function exists; enforce it | Regex-based cross-repo validation (deprecated) |
| Persistence | StorageBackend abstraction for URP | Full distributed consensus (deferred) |
| Verification | Z3 starter axioms for kernel invariants | Complete formal proof (requires stable axioms) |

## SAPE Audit Findings → Phase 60 Step Mapping

| Finding | Status | Phase 60 Step | Rationale |
|---------|--------|---------------|-----------|
| F3: Gateway no auth | Fixed in artifact | — | `require_admin()` exists on all POST endpoints |
| F4: URP auth mismatch | Fixed in artifact | — | `URP_ADMIN_TOKEN or URP_ADMIN_KEY` fallback exists |
| F5: In-memory _DB | TRUE | Step 5 | Split-brain under multi-worker — needs persistence |
| F6: Conformance ports | TRUE | Step 6 | CI tests target wrong ports |
| F7: Sequential pipeline | FALSE | — | apex_engine.py uses asyncio.gather in 2 batches |
| F8: Health O(n) | TRUE | Step 3 | 11-subsystem check on every liveness probe |
| F9: SNR divergence | PARTIAL | Step 4 | Canonical function exists but not enforced |
| F10: Dependency lock | TRUE | — | Phase 59 addresses this |
| F12: OpenAPI mismatch | TRUE | Step 6 | 0/8 endpoints implemented vs spec |

## Step Index

| Step | File | Target | Risk | Effort |
|------|------|--------|------|--------|
| 1 | `01_constitution_toml.md` | Externalize α4-α10 gates into TOML codex | Low | 8-12h |
| 2 | `02_evidence_accumulation.md` | CI produces evidence.json per commit | Low | 6-8h |
| 3 | `03_health_tiering.md` | Split health into live/ready/deep tiers | Zero | 4-6h |
| 4 | `04_snr_contract.md` | Enforce canonical SNR normalization | Zero | 3-4h |
| 5 | `05_urp_persistence.md` | Replace in-memory _DB with StorageBackend | Medium | 12-16h |
| 6 | `06_conformance_topology.md` | Align test ports with service topology | Low | 4-6h |
| 7 | `07_z3_axiom_starter.md` | Z3 SMT2 axioms for kernel invariants | Medium | 10-14h |

## SAPE Gap Closure Matrix

| Dimension | After Phase 59 | After Phase 60 | Delta |
|-----------|----------------|----------------|-------|
| Architecture | 0.95 | 0.97 | +0.02 (constitution.toml) |
| Security | 0.88 | 0.90 | +0.02 (conformance + SNR contract) |
| Performance | 0.88 | 0.92 | +0.04 (health tiering) |
| Scalability | 0.90 | 0.93 | +0.03 (URP persistence) |
| Verification | 0.90 | 0.94 | +0.04 (Z3 axioms + evidence accumulation) |
| **Composite** | **0.93** | **0.96** | **+0.03** |

## Dependency Order

Steps 3, 4, 6 are independent — can run in parallel.
Step 1 depends on Phase 59 Step 4 (constants artifact must exist).
Step 2 depends on Step 1 (evidence.json references constitution gate results).
Step 5 is independent but largest — start early.
Step 7 depends on Step 1 (Z3 axioms codify what constitution.toml declares).

```
Phase 59 Step 4 (Constants) ──▶ Step 1 (Constitution.toml) ──┬──▶ Step 2 (Evidence)
                                                              └──▶ Step 7 (Z3 Axioms)
Step 3 (Health Tiering) ────────────────────────────────────────▶ (independent)
Step 4 (SNR Contract) ─────────────────────────────────────────▶ (independent)
Step 5 (URP Persistence) ──────────────────────────────────────▶ (independent)
Step 6 (Conformance) ──────────────────────────────────────────▶ (independent)
```

## Hidden Flow Pattern (HHMM + Diffusion Reasoning)

**Observed HHMM State Transition:**
```
IDLE → EXPLORING (codebase scan) → ANALYZING (SAPE audit) → CREATING (Phase 57-59)
     → ORGANIZING (boundary hardening) → CREATING (Phase 60 implementation)
```

**Diffusion Reasoning Amplifier Output:**
The system's hidden thought pattern reveals a **convergence cycle**:
1. Intelligence concentrates in core runtime (Phase 57 MissionOrchestrator)
2. Service shell receives decomposed distribution (URP artifact pack)
3. Boundary mismatches create drift (auth/config/ports/scoring)
4. Permissive fallbacks keep flow alive but reduce trust
5. Re-concentration back into core runtime

**Phase 60 breaks this cycle** by:
- Making boundaries declarative, not hardcoded (constitution.toml)
- Making evidence first-class in CI (evidence.json)
- Making health probes cheap (tiered endpoints)
- Making state durable (StorageBackend)
- Making invariants machine-verifiable (Z3)

The cycle converts from "drift loop" to "diffusion loop" — intelligence
radiates outward with verified contracts, not hopeful assumptions.

## Golden Gems Extracted (SNR ≥ 0.98)

1. **constitution.toml is the Invariant Verification Layer** — binds philosophy to compiler
2. **Evidence accumulation is the missing audit trail** — zero evidence.json in any CI workflow today
3. **Health tiering is the cheapest trust gain** — O(1) liveness vs O(11) subsystem scan
4. **Z3 axioms formalize what code already enforces** — prove the existing gates correct
5. **StorageBackend abstraction unlocks horizontal scaling** — prerequisite for multi-node

## Success Criteria

- All 7 steps pass their TDD anchors
- Full test suite remains GREEN (7,911+ tests)
- No new required dependencies (Z3 is already in dev env)
- Evidence ledger receipt emitted for each step
- SAPE composite re-scored at ≥ 0.96
- constitution.toml parses without errors and generates valid test assertions
- `evidence.json` produced by CI and uploaded as artifact
- Z3 returns SAT for all kernel invariant axioms

## Total Spec Size

| File | Lines |
|------|-------|
| `00_overview.md` | 152 |
| `01_constitution_toml.md` | 414 |
| `02_evidence_accumulation.md` | 306 |
| `03_health_tiering.md` | 222 |
| `04_snr_contract.md` | 188 |
| `05_urp_persistence.md` | 375 |
| `06_conformance_topology.md` | 266 |
| `07_z3_axiom_starter.md` | 457 |
| **Total** | **2,380** |
