# BIZRA Sovereign Engine — Unified Specification Index
# Single Source of Truth — All Phases (0-68 + Experimental)

**Last Updated:** 2026-03-06
**Status:** ACTIVE
**Spec Locations:** 3 directories, 170+ files, ~50K LOC
**Total TDD Anchors:** 162 (Phase 67-68) + anchors in earlier phases
**Implementation Gradient:** Phases 0-43 ~100%, 44-58 ~80%, 59-66 ~60%, 67 ~92%, 68 ~0%

---

## 1. Architecture — Three Spec Directories, One Engine

### 1.1 Directory Map

```
/mnt/c/BIZRA-DATA-LAKE/
  docs/specs/          <-- Primary (70+ files, phases 0-68 + frameworks)
  specs/               <-- Experimental (86 files, phases 42-50 + programs)
  docs/specs/UNIFIED_SPEC_INDEX.md  <-- THIS FILE (SSoT)
```

| Location | Files | LOC | Phase Range | Role |
|----------|-------|-----|-------------|------|
| `docs/specs/` | 70+ | ~28K | 0-68 | Canonical specs — all implemented phases |
| `specs/` | 86 | ~22K | 42-50 + experimental | Research, programs, experimental |
| **Total** | **156+** | **~50K** | **0-68** | |

**Overlap:** Phases 42-50 appear in BOTH directories. `docs/specs/` is authoritative
for implementation. `specs/` contains deeper research, alternative approaches, and
program-level specs (Alpha-100, SAP, User Zero) not in `docs/specs/`.

### 1.2 Layer Stack

```
Phase 68: Nervous System (buses, loops, config)          [docs/specs/]
  |
  | ActionBus.propose() -> TeleScript -> FATE -> Channel -> Receipt
  | OmegaLoop.run() -> proof-based iteration
  | ConfigLoader -> 3-scope YAML
  | CapsuleRuntime -> workflow execution
  | TopicRegistry -> 38 canonical events
  |
  v
Phase 67: Constitutional Kernel (algorithms, types, fixed-point)  [docs/specs/]
  |
  | 15 native algorithms (A1-A15)
  | Fixed-point arithmetic (FP_PRECISION = 1,000,000)
  | 12-step ticker heartbeat
  | Declaration genesis + covenant chain
  | Sovereignty CLI (init/work/attest/status)
  |
  v
Phases 42-66: Foundation Stack (SNR, identity, cognition, federation)  [both]
  |
  v
Phases 0-41: Genesis (NTU, bridges, RAG, hypergraph, production)  [docs/specs/]
  |
  v
core/integration/constants.py — Single Source of Truth for ALL thresholds
```

**Rule:** Higher phases USE lower phases. Never the reverse. Dependencies flow
downward only.

---

## 2. Complete Spec Inventory

### Phase 67 — Sovereign Instantiation (Constitutional Kernel)

| # | File | Lines | Status | Code Location |
|---|------|-------|--------|---------------|
| 67.00 | overview.md | 123 | REFERENCE | — |
| 67.01 | fixed_point_arithmetic.md | 213 | IMPLEMENTED | core/constitutional/fixed_point.py (194 LOC) |
| 67.02 | native_algorithms.md | 639 | IMPLEMENTED | core/constitutional/algorithms.py (600 LOC) |
| 67.03a | asabiyyah_gini_coupling.md | 311 | SPEC-ONLY | NOT WIRED — highest-priority gap |
| 67.03b | declaration_genesis.md | 267 | IMPLEMENTED | core/constitutional/declaration.py (150 LOC) |
| 67.04 | sovereignty_cli.md | 353 | IMPLEMENTED | core/constitutional/cli.py (473 LOC) |
| 67.05 | akis_pipeline.md | 342 | SPEC-ONLY | core/akis/ does not exist |
| 67.06 | chaos_validators.md | 382 | IMPLEMENTED | tests/constitutional/test_chaos.py (610 LOC) |
| 67.07 | tdd_anchors.md | 359 | IMPLEMENTED | 3,563 LOC across 8 test files |

### Phase 68 — Bus Architecture (MMORPG Nervous System)

| # | File | Lines | Status | Code Location |
|---|------|-------|--------|---------------|
| 68.00 | overview.md | 201 | REFERENCE | — |
| 68.01 | action_bus.md | 296 | SPEC-ONLY | core/bus/action_bus.py (planned) |
| 68.02 | omega_loop.md | 327 | SPEC-ONLY | core/bus/omega_loop.py (planned) |
| 68.03 | config_system.md | 284 | SPEC-ONLY | core/config/ (planned) |
| 68.04 | capsule_runtime.md | 287 | SPEC-ONLY | core/bus/capsule_runtime.py (planned) |
| 68.05 | telescript_python.md | 323 | SPEC-ONLY | core/bus/telescript.py (planned) |
| 68.06 | topic_registry.md | 258 | SPEC-ONLY | core/bus/topics.py (planned) |

### `specs/` — Experimental & Program Specs (Phases 42-50+)

| # | Directory | Files | LOC | Status | Unique Content |
|---|-----------|-------|-----|--------|----------------|
| 42 | phase-42-snr-unification | 6 | 1,012 | COMPLETED | SNR v2 adapter design |
| 43 | phase-43-node0-identity-awakening | 6 | 1,018 | COMPLETED | Ed25519 identity bootstrap |
| 45 | phase-45-distributed-cognitive-scaling | 6 | 2,456 | ACTIVE | **Reverse Scale Hypothesis** — N nodes > sum(N isolated) |
| 47 | phase-47-cognitive-resonance-activation | 7 | 1,785 | COMPLETED | Resonance + reflex myelination |
| 48 | phase-48-rust-workspace-unification | 5 | 916 | COMPLETED | 22-crate workspace audit |
| 49 | phase-49-refinement-consolidation | 5 | 594 | ACTIVE | Sprint consolidation patterns |
| 50 | phase-50-rlm-sovereign-cognition | 6 | 1,959 | ACTIVE | **RLM integration** (MIT CSAIL 2026) |
| — | alpha100-sprint3 | 5 | 1,842 | ACTIVE | Alpha-100 release program |
| — | bizra-harness | 8 | 2,323 | ACTIVE | Regression test harness framework |
| — | node0-empower-one-human | 6 | 1,818 | ACTIVE | Mission: empower one human |
| — | sap-v0 | 6 | 313 | ACTIVE | **Sovereign Agent Protocol** — consent-first negotiation |
| — | user-zero-bootstrap | 6 | 2,263 | ACTIVE | User Zero onboarding + agent-as-marketing |
| — | user-zero-shadow-marketing-v0 | 2 | 42 | MINIMAL | Shadow marketing stub |
| — | v3-memory-unification | 7 | 2,098 | ACTIVE | HNSW+SQLite hybrid query engine |
| — | v3-swarm-coordination | 5 | 1,311 | ACTIVE | 15-agent hierarchical mesh |

**Key specs only in `specs/` (not in `docs/specs/`):**

1. **Reverse Scale Hypothesis** (phase-45) — The thesis that distributed
   cognition exhibits reverse scaling: collective intelligence exceeds the
   sum of isolated nodes. Unprecedented in the literature.

2. **RLM Sovereign Cognition** (phase-50) — Integration plan for MIT CSAIL's
   Recursive Language Model (2026). Enables on-device recursive reasoning
   without cloud dependency.

3. **SAP v0** — Sovereign Agent Protocol: consent-first agent-to-agent
   negotiation overlay. Pre-requisite for federation beyond gossip.

4. **bizra-harness** — Comprehensive regression testing framework with
   campaign management, evidence chains, and reproducible benchmarks.

5. **v3-memory-unification** — HNSW+SQLite hybrid backend for unified memory.
   150x-12,500x search improvement over linear scan. ADR-006/ADR-009.

---

## 3. Implementation Status Matrix

```
                      docs/specs/ (Phases 0-68)
Phases 0-41:  [######################################] 100%  (foundation, deployed)
Phases 42-50: [################################------]  80%  (4/15 specs/ items completed)
Phases 51-58: [##############################--------]  80%  (core + mission wired)
Phases 59-66: [########################--------------]  60%  (constitutional hardening)
Phase 67:     [####################################--]  92%  (7/9 implemented)
Phase 68:     [                                      ]   0%  (0/7 implemented)

                      specs/ (Experimental)
Completed:    [####                                  ]  27%  (4/15: ph42,43,47,48)
Active:       [##############################        ]  60%  (9/15: research + programs)
Minimal:      [#                                     ]   7%  (1/15: shadow-marketing stub)
```

---

## 4. Critical Gaps (Priority Order)

### GAP-1: Asabiyyah-Gini Coupling [CRITICAL]

**Spec:** phase_67_03_asabiyyah_gini_coupling.md
**Status:** Specified, not implemented
**Impact:** The single highest-SNR unwired connection in the codebase

**What's missing:**
- Function `asabiyyah_adjustment()` in algorithms.py (~20 LOC)
- Modified `khaldunian_throttle(gini, asabiyyah=FP_ZERO)` signature
- Modified `progressive_mint()` to pass asabiyyah through
- Reordered `process_tick()` — asabiyyah computed at Step 3.5, not Step 12
- 3 new constants in constants.py:
  - ASABIYYAH_COUPLING_FLOOR = 0.80
  - ASABIYYAH_COUPLING_CEIL = 1.20
  - ASABIYYAH_NEUTRAL = 0.50

**Fix effort:** ~100 LOC + 12 tests
**Blocks:** Phase 68 economy.asabiyyah events (TopicRegistry tier 3)

### GAP-2: Python ActionBus [HIGH]

**Spec:** phase_68_01_action_bus.md
**Status:** Specified, not implemented
**Impact:** MissionOrchestrator calls channels directly (no CQRS, no gates)

**Fix effort:** ~300 LOC + 16 tests
**Blocks:** OmegaLoop, CapsuleRuntime

### GAP-3: TopicRegistry [HIGH]

**Spec:** phase_68_06_topic_registry.md
**Status:** Specified, not implemented
**Impact:** No canonical topic validation; Python and Rust topics can drift

**Fix effort:** ~200 LOC + 10 tests
**Blocks:** Cross-runtime sync CI gate

### GAP-4: TeleScript Python [HIGH]

**Spec:** phase_68_05_telescript_python.md
**Status:** Specified, not implemented
**Impact:** No Python-side capability enforcement

**Fix effort:** ~250 LOC + 14 tests
**Blocks:** ActionBus (capability check step)

### GAP-5: AKIS Pipeline [MEDIUM]

**Spec:** phase_67_05_akis_pipeline.md
**Status:** Specified, not implemented
**Impact:** Sensory layer deferred; no auto-extraction from external sources

**Fix effort:** ~1,000 LOC + 8 tests
**Blocks:** Nothing (standalone)

### GAP-6: OmegaLoop, ConfigSystem, CapsuleRuntime [MEDIUM]

**Specs:** phase_68_02, 68_03, 68_04
**Status:** Specified, not implemented
**Impact:** No proof-based iteration, no unified config, no skill runtime

**Fix effort:** ~800 LOC + 36 tests
**Blocks:** Each other (dependency chain)

---

## 5. Dependency Graph

```
constants.py (SSoT) ─────────────────────────────────────┐
                                                          |
Phase 67 (Constitutional Kernel)                          |
  fixed_point.py ──────────────────────────────┐          |
  types.py ────────────────────────────────────┤          |
  algorithms.py ───────────────────────────────┤          |
    + asabiyyah_adjustment() [GAP-1] ──────────┤          |
  ticker.py ───────────────────────────────────┤          |
  declaration.py ──────────────────────────────┤          |
  cli.py ──────────────────────────────────────┘          |
                                                          |
Phase 68 (Nervous System)                                 |
  topics.py ─────── [no deps] ─────────────────┐          |
  telescript.py ─── [depends on config] ───────┤          |
  types.py (bus) ── [depends on fp types] ─────┤          |
  action_bus.py ─── [depends on above 3] ──────┤          |
  config/loader.py ─ [depends on constants] ───┤          |
  omega_loop.py ─── [depends on action_bus] ───┤          |
  capsule_runtime.py [depends on action_bus] ──┘          |
                                                          |
Existing Infrastructure (already complete)                 |
  core/sovereign/event_bus.py ─────────────────── EventBus|
  core/sovereign/mission.py ──────── MissionOrchestrator  |
  core/bridges/channel_dispatcher.py  ChannelDispatcher   |
  core/proof_engine/evidence_ledger.py  EvidenceLedger    |
  bizra-hooks/src/event_bus.rs ──── Rust 8-shard EventBus |
  bizra-agent/src/action_bus.rs ─── Rust ActionBus        |
  bizra-action/src/dispatcher.rs ── Rust OODA Dispatcher  |
```

---

## 6. Unified Type System

All types across both phases, in dependency order:

### Layer 0: Fixed-Point (Phase 67.01)
```
FP_PRECISION = 1,000,000
fp(float) -> int
fp_float(int) -> float
fp_add, fp_sub, fp_mul, fp_div, fp_clamp
```

### Layer 1: Constitutional Types (Phase 67.02)
```
ActionReceipt [frozen] — 11 fields (receipt_id through co_actors)
WalletState — 11 fields (node_id through cooperative_actions)
Proposal — 7 fields
Reflex [frozen] — 5 fields
Attestation [frozen] — 5 fields
Event — 7 fields (event_id through hash)
ConstitutionalInvariant [frozen] — 4 fields
```

### Layer 2: Bus Types (Phase 68.01)
```
ActionEnvelope [frozen] — 10 fields (action_id through timestamp)
ActionBudget [frozen] — 3 fields (time_ms, s2_tokens_max, retry_max)
ActionStatus [enum] — 8 states
BusActionReceipt [frozen] — 8 fields (merkle-chained)
ChannelResult — 5 fields
```

### Layer 3: Control Types (Phase 68.02-06)
```
OmegaLoopState — 10 fields
OmegaStatus [enum] — 7 states
LoopBudget — 3 fields
ProofCondition [frozen] — 4 fields
TeleScriptPolicy [frozen] — 5 fields
TeleScriptVerdict [frozen] — 4 fields
Capability [enum] — 17 values
BusEvent [frozen] — 8 fields (canonical envelope)
TopicTier [enum] — 8 tiers
TopicDef — 3 fields
CapsuleManifest — 8 fields
CapsuleResult — 4 fields
BizraConfig [pydantic] — 8 sections
```

### Naming Convention
- Phase 67 `ActionReceipt` = constitutional receipt (ihsan scores)
- Phase 68 `BusActionReceipt` = bus receipt (merkle chain, guardian verdict)
- Both are frozen dataclasses. Bus receipt WRAPS constitutional receipt.

---

## 7. Canonical Flow (Integrated Phase 67 + 68)

```
USER INTENT
  |
  v
OmegaLoop.run(mission, proof_conditions, budget)     [68.02]
  |
  | ITERATION START
  |   |
  |   v
  |  ActionBus.propose(action)                         [68.01]
  |   |
  |   v
  |  TeleScript.check(capabilities, paths)             [68.05]
  |   |-- deny --> EventBus.emit("policy.telescript.denied")
  |   +-- allow
  |         |
  |         v
  |  FATE Gate.evaluate(action)                        [existing]
  |   |-- deny --> EventBus.emit("policy.fate.vetoed")
  |   +-- allow
  |         |
  |         v
  |  Channel.execute(action)                           [existing]
  |   |
  |   v
  |  Verifier.check(pre, post)
  |   |
  |   v
  |  BusActionReceipt = sign(outcome_hash, ihsan)      [68.01]
  |   |
  |   v
  |  EventBus.emit("action.receipt", receipt)           [68.06]
  |   |
  |   v
  |  Post-Receipt Hooks:
  |   |-- ConstitutionalTicker.process_tick()           [67.02]
  |   |     |-- intent_gate()                           [A1]
  |   |     |-- ihsan_score()                           [A1]
  |   |     |-- compute_gini()                          [A4]
  |   |     |-- network_asabiyyah()                     [A15]
  |   |     |-- asabiyyah_adjustment()                  [67.03 GAP-1]
  |   |     |-- progressive_mint(gini, asabiyyah)       [A4]
  |   |     |-- accrue_bloom()                          [A3]
  |   |     |-- apply_demurrage()                       [A7]
  |   |     |-- compute_zakat()                         [A5]
  |   |     |-- compile_reflex()                        [A10]
  |   |     +-- append_event()                          [A14]
  |   |
  |   +-- CapsuleRuntime.match_trigger()                [68.04]
  |         |-- reflex_compile_if_eligible
  |         +-- format/lint/test hooks
  |
  | CHECK PROOFS
  |   |-- ihsan >= 0.95?
  |   |-- tests pass?
  |   |-- ledger committed?
  |   |-- all conditions met? --> PROVED --> EXIT
  |   +-- not met? --> NEXT ITERATION
  |
  | ITERATION END
  v
OmegaLoop.state stored in EventLog (resumable)
```

---

## 8. Constants Alignment (All Sources)

All values sourced from `core/integration/constants.py`:

| Constant | Value | Used By | Status |
|----------|-------|---------|--------|
| UNIFIED_IHSAN_THRESHOLD | 0.95 | A1, ActionBus, OmegaLoop | OK |
| INTENT_FLOOR | 0.90 | A1 intent_gate | OK |
| ADL_GINI_THRESHOLD | 0.35 | A4, ConfigLoader | OK |
| GINI_HEALTHY | 0.30 | A4 khaldunian_throttle | OK |
| GINI_WARNING | 0.50 | A4 | OK |
| GINI_CRISIS | 0.70 | A4 | OK |
| ZAKAT_RATE | 0.025 | A5 | OK |
| NISAB_THRESHOLD | 85.0 | A5 | OK |
| EQUITY_FACTOR_MIN | 1.0 | A4 ghazali_equity | OK |
| EQUITY_FACTOR_MAX | 5.0 | A4 | OK |
| SNR_MINIMUM_THRESHOLD | 0.85 | ConfigLoader | OK |
| ASABIYYAH_WEIGHTS | (0.4, 0.3, 0.3) | A15 | OK |
| **ASABIYYAH_COUPLING_FLOOR** | **0.80** | **asabiyyah_adjustment** | **MISSING** |
| **ASABIYYAH_COUPLING_CEIL** | **1.20** | **asabiyyah_adjustment** | **MISSING** |
| **ASABIYYAH_NEUTRAL** | **0.50** | **asabiyyah_adjustment** | **MISSING** |

---

## 9. Test Contract Summary

| Phase | Component | Tests | Status |
|-------|-----------|-------|--------|
| 67.01 | Fixed-Point | 13 | ALL PASS |
| 67.02 | 15 Algorithms | 35 | ALL PASS |
| 67.03a | Asabiyyah Coupling | 12 | NOT WRITTEN |
| 67.03b | Declaration | 7 | ALL PASS |
| 67.04 | CLI | 8 | ALL PASS |
| 67.05 | AKIS | 8 | NOT WRITTEN |
| 67.06 | Chaos | 10 | ALL PASS |
| 67.07 | Red Team + Integration | ~15 | ALL PASS |
| 68.01 | ActionBus | 16 | NOT WRITTEN |
| 68.02 | OmegaLoop | 14 | NOT WRITTEN |
| 68.03 | ConfigSystem | 12 | NOT WRITTEN |
| 68.04 | CapsuleRuntime | 10 | NOT WRITTEN |
| 68.05 | TeleScript | 14 | NOT WRITTEN |
| 68.06 | TopicRegistry | 10 | NOT WRITTEN |
| **Total** | | **162+** | **88 pass / 96 not written** |

---

## 10. Implementation Roadmap (Critical Path)

### Sprint 1: Wire the Gap (1 day)
**Goal:** Close GAP-1 (Asabiyyah-Gini coupling)

1. Add 3 constants to `constants.py`
2. Add `asabiyyah_adjustment()` to `algorithms.py`
3. Modify `khaldunian_throttle()` signature
4. Modify `progressive_mint()` signature
5. Reorder `process_tick()` steps
6. Write 12 tests
7. Run full test suite (regression)

**Why first:** Highest SNR. Closes the constitutional kernel. Unblocks
Phase 68 economy.asabiyyah events.

### Sprint 2: Bus Foundation (2-3 days)
**Goal:** Implement Phase 68 Layer 0 (types + topics + telescript)

1. Create `core/bus/__init__.py`
2. Implement `core/bus/types.py` (ActionEnvelope, ActionBudget, etc.)
3. Implement `core/bus/topics.py` (TopicRegistry + 38 topics)
4. Implement `core/bus/telescript.py` (TeleScriptEngine)
5. Write 34 tests (10 + 14 + 10 type tests)

**Why second:** Zero inter-dependencies. Three files parallelizable.

### Sprint 3: ActionBus (2 days)
**Goal:** Implement Phase 68 command pipeline

1. Implement `core/bus/action_bus.py`
2. Wire to existing EventBus
3. Write 16 tests
4. Integration test with MissionOrchestrator

### Sprint 4: OmegaLoop + Config (3 days)
**Goal:** Proof-based iteration + unified config

1. Implement `core/bus/omega_loop.py`
2. Implement `core/config/loader.py`
3. Write 26 tests (14 + 12)

### Sprint 5: Capsule Runtime + Integration (2-3 days)
**Goal:** Skill execution engine + full wiring

1. Implement `core/bus/capsule_runtime.py`
2. Wire MissionOrchestrator through ActionBus + OmegaLoop
3. Wire Ticker to emit economy.* events
4. Write 10 + integration tests

### Sprint 6: AKIS Pipeline (3-5 days)
**Goal:** External knowledge extraction

1. Implement `core/akis/` (spec 67.05)
2. Write 8 tests

---

## 11. Spec File Locations (All Three Directories)

```
docs/specs/                                    <-- CANONICAL (implementation authority)
  phase0_foundation_integrity.md
  phase_01_ntu_rust_*.md                       (3 files)
  phase2_bridge_skill_routing.md
  phase19_*.md - phase_50_*.md                 (50+ individual specs)
  phase_51_integration_index.md
  phase_52_lifecycle_emulation/
  phase_53_sovereign_migration/
  phase_54_pat_sat_architecture/
  phase_55_unified_message_backbone/
  phase_56_security_hardening/
  phase_57_first_heartbeat/                    (7 files)
  phase_58_optimization_sprint.md
  phase_59_consolidation_sprint/
  phase_60_constitutional_boundary/
  phase_61_proof_chain_v2/
  phase_62_node0_v6_deployment/
  phase_64_self_harness_sovereign/
  phase_65_lifecycle_protocol/
  phase_66_constitutional_hardening/
  phase_67_sovereign_instantiation/            (9 files)
    phase_67_03_asabiyyah_gini_coupling.md     <-- GAP-1
    phase_67_05_akis_pipeline.md               <-- GAP-5
  phase_68_bus_architecture/                   (7 files)
    phase_68_01_action_bus.md                  <-- GAP-2
    phase_68_02_omega_loop.md                  <-- GAP-6
    phase_68_03_config_system.md               <-- GAP-6
    phase_68_04_capsule_runtime.md             <-- GAP-6
    phase_68_05_telescript_python.md           <-- GAP-4
    phase_68_06_topic_registry.md              <-- GAP-3
  UNIFIED_SPEC_INDEX.md                        <-- THIS FILE (SSoT)

specs/                                         <-- EXPERIMENTAL (research authority)
  phase-42-snr-unification/                    COMPLETED
  phase-43-node0-identity-awakening/           COMPLETED
  phase-45-distributed-cognitive-scaling/      ACTIVE — Reverse Scale Hypothesis
  phase-47-cognitive-resonance-activation/     COMPLETED
  phase-48-rust-workspace-unification/         COMPLETED
  phase-49-refinement-consolidation/           ACTIVE
  phase-50-rlm-sovereign-cognition/            ACTIVE — RLM integration
  alpha100-sprint3/                            ACTIVE — release program
  bizra-harness/                               ACTIVE — regression framework
  node0-empower-one-human/                     ACTIVE — mission spec
  sap-v0/                                      ACTIVE — Sovereign Agent Protocol
  user-zero-bootstrap/                         ACTIVE — onboarding
  user-zero-shadow-marketing-v0/               MINIMAL — stub
  v3-memory-unification/                       ACTIVE — HNSW+SQLite hybrid
  v3-swarm-coordination/                       ACTIVE — 15-agent mesh
```

### Phase 69 — Sovereign Synthesis (Multi-Lens Analysis + Wiring Sprint)

| # | File | Lines | Status | Purpose |
|---|------|-------|--------|---------|
| 69.00 | multi_lens_analysis.md | 340 | REFERENCE | 8-lens codebase audit + GoT dependency graph |
| 69.01 | wiring_sprint.md | 304 | SPEC-ONLY | Sprints 1-2: Asabiyyah + Bus Foundation |
| 69.02 | wiring_sprint_continued.md | 287 | SPEC-ONLY | Sprints 3-6: ActionBus + Security + Omega + Capsules |

### Conflict Resolution Rule

When the same phase appears in both `docs/specs/` and `specs/`:
- **`docs/specs/`** is authoritative for implementation decisions
- **`specs/`** may contain deeper research, alternative approaches, or
  program-level context not in `docs/specs/`
- If they contradict, `docs/specs/` wins
- Unique content in `specs/` (SAP, User Zero, harness, v3-*) has no
  counterpart in `docs/specs/` and stands independently

---

## 12. Standing on Giants

| Scholar | Contribution | Where Used |
|---------|-------------|------------|
| Al-Ghazali (1058-1111) | Intent as ethical pre-gate | A1 intent_gate |
| Ibn Khaldun (1332-1406) | Asabiyyah + progressive throttle | A4, A15, GAP-1 |
| Al-Khwarizmi (780-850) | Algorithm as deterministic procedure | Fixed-point kernel |
| Kahneman (2002) | System-1/System-2 cognitive split | A10 reflex, OmegaLoop |
| Shannon (1948) | Information entropy, SNR | SNR gates, phase-42 |
| Nakamoto (2008) | Event log as consensus | A14, EventLog |
| Lamport (1978) | Logical clocks | BusEvent ordering |
| Fowler (2005) | CQRS + Event Sourcing | ActionBus + EventBus split |
| Gray & Reuter (1993) | Two-phase commit | Action lifecycle |
| Rawls (1971) | Veil of ignorance | Ghazali equity factor |
| MIT CSAIL (2026) | Recursive Language Models | phase-50 RLM integration |
| Malkov & Yashunin (2018) | HNSW graph indexing | v3-memory-unification |
| Dunbar (1992) | Social group size limits | phase-45 distributed cognition |
