# BIZRA Sovereign Engine — Unified Specification Index

**Last Updated:** 2026-03-09
**Status:** ACTIVE — Single canonical location
**Location:** `docs/specs/` (all specs consolidated here)
**Files:** 348 | **LOC:** ~111K | **TDD Anchors:** 162+
**Implementation Gradient:** Phases 0-43 ~100%, 44-58 ~80%, 59-66 ~60%, 67 ~92%, 68 ~0%

---

## Directory Structure

```
docs/specs/
├── UNIFIED_SPEC_INDEX.md        ← THIS FILE (SSoT)
├── phase_*.md                   ← Canonical phase specs (0-79)
├── phase_*/                     ← Multi-file phase specs (37, 52-79)
├── _experimental/               ← Research & program specs (ex-specs/)
│   ├── alpha100-sprint3/
│   ├── node0-empower-one-human/        [BACKLOG]
│   ├── phase-42-snr-unification/       [BUILT]
│   ├── phase-43-node0-identity-awakening/  [BACKLOG]
│   ├── phase-45-distributed-cognitive-scaling/  [BUILT]
│   ├── phase-47-cognitive-resonance-activation/  [BACKLOG]
│   ├── phase-48-rust-workspace-unification/  [BUILT]
│   ├── phase-49-refinement-consolidation/  [BUILT]
│   ├── phase-50-rlm-sovereign-cognition/  [BUILT]
│   ├── sap-v0/                         [BUILT]
│   ├── user-zero-bootstrap/            [BUILT]
│   ├── user-zero-shadow-marketing-v0/  [BUILT]
│   ├── v3-memory-unification/          [BUILT]
│   ├── v3-swarm-coordination/          [BUILT]
│   └── bizra-harness/                  [BUILT]
├── _terminal/                   ← Terminal UI specs (ex-terminal/specs/)
│   ├── phase_01_terminal_architecture.md   [BACKLOG]
│   ├── phase_02_view_briefing.md           [BACKLOG]
│   ├── phase_03_view_mission.md            [BACKLOG]
│   ├── phase_04_view_receipts.md           [BACKLOG]
│   ├── phase_05_view_wallet.md             [BACKLOG]
│   ├── phase_06_view_memory_skills.md      [BACKLOG]
│   ├── phase_07_view_system.md             [BACKLOG]
│   ├── phase_08_event_bus_wiring.md        [BACKLOG]
│   └── phase_09_build_plan.md              [BACKLOG]
├── _normalizers/                ← Normalizer specs (ex-bizra-normalizers/specs/)
│   └── pat_collection_agent.md             [BACKLOG]
├── apex_integration/            ← SAPE/Apex framework
├── certification_framework/     ← Certification gates
├── ddagi_os_atlas_v5/           ← DDAGI OS atlas
├── installer_v2/                ← Installer specs
├── node0_kernel/                ← Node0 kernel design
├── true_spearpoint_v9/          ← Spearpoint benchmark
└── ui_ux_apex/                  ← UI/UX design
```

**Consolidation note (2026-03-09):** Four satellite directories were merged:
- `specs/` → `docs/specs/_experimental/`
- `terminal/specs/` → `docs/specs/_terminal/`
- `bizra-normalizers/specs/` → `docs/specs/_normalizers/`
- `scripts/spec/` → validators moved to `scripts/` (not specs)

---

## Backlog: 20 Unbuilt Specs

### P0 — Constitutional Gaps (Phase 67)

| # | Spec | Lines | Code Target | Dependency |
|---|------|-------|-------------|------------|
| 67.03a | `phase_67_sovereign_instantiation/67_03a_asabiyyah_gini_coupling.md` | 311 | `core/constitutional/asabiyyah.py` | ADL Gini from constants.py |
| 67.05 | `phase_67_sovereign_instantiation/67_05_akis_pipeline.md` | 342 | `core/akis/` | Living memory, inference |

### P1 — Nervous System (Phase 68, 0% built)

| # | Spec | Lines | Code Target |
|---|------|-------|-------------|
| 68.00 | `phase_68_bus_architecture/phase_68_00_overview.md` | 201 | Reference |
| 68.01 | `phase_68_bus_architecture/phase_68_01_action_bus.md` | 296 | `core/bus/action_bus.py` |
| 68.02 | `phase_68_bus_architecture/phase_68_02_omega_loop.md` | 327 | `core/bus/omega_loop.py` |
| 68.03 | `phase_68_bus_architecture/phase_68_03_config_system.md` | 284 | `core/config/` |
| 68.04 | `phase_68_bus_architecture/phase_68_04_capsule_runtime.md` | 287 | `core/bus/capsule_runtime.py` |
| 68.05 | `phase_68_bus_architecture/phase_68_05_telescript_python.md` | 323 | `core/bus/telescript.py` |
| 68.06 | `phase_68_bus_architecture/phase_68_06_topic_registry.md` | 258 | `core/bus/topics.py` |

### P2 — Experimental (3 unbuilt)

| Spec | Lines | Code Target |
|------|-------|-------------|
| `_experimental/node0-empower-one-human/` | 1,820 | Node0 empowerment loop |
| `_experimental/phase-43-node0-identity-awakening/` | 639 | `core/sovereign/identity.py` |
| `_experimental/phase-47-cognitive-resonance-activation/` | 1,770 | `core/resonance/` |

### P3 — Terminal UI (9 unbuilt)

| Spec | Lines | Code Target |
|------|-------|-------------|
| `_terminal/phase_01_terminal_architecture.md` | 125 | Terminal architecture |
| `_terminal/phase_02_view_briefing.md` | 140 | Briefing view |
| `_terminal/phase_03_view_mission.md` | 156 | Mission view |
| `_terminal/phase_04_view_receipts.md` | 134 | Receipts view |
| `_terminal/phase_05_view_wallet.md` | 164 | Wallet view |
| `_terminal/phase_06_view_memory_skills.md` | 180 | Memory/skills view |
| `_terminal/phase_07_view_system.md` | 150 | System view |
| `_terminal/phase_08_event_bus_wiring.md` | 126 | Event bus integration |
| `_terminal/phase_09_build_plan.md` | 170 | Build plan |

### P4 — Normalizers (1 unbuilt)

| Spec | Lines | Code Target |
|------|-------|-------------|
| `_normalizers/pat_collection_agent.md` | 105 | PAT collection agent |

---

## Built Specs (Reference)

Phases 0-66 and most of 67 are implemented. These specs remain as historical
design records and audit trail. The code is the authoritative source for current
behavior; specs document the original design intent.

### Layer Stack (Dependencies Flow Down)

```
Phase 79: Genesis Gate (constitutional go-live)        [BUILT]
Phase 78: Terminal v1 (frontend shell)                 [BUILT]
Phase 77: API contract + deploy smoke                  [BUILT]
Phase 72: Constitutional kernel (algorithms, types)    [BUILT]
Phase 69: Synthesis + wiring sprints                   [SPEC-ONLY]
Phase 68: Nervous system (buses, loops, config)        [SPEC-ONLY]
Phase 67: Sovereign instantiation (92% built)          [2 GAPS]
Phase 60-66: Constitutional boundary + hardening       [BUILT]
Phase 52-59: Lifecycle, migration, PAT/SAT, security   [BUILT]
Phase 42-51: SNR, identity, chat, dashboard, AaaS      [~80%]
Phase 30-41: DDAGI OS, hypergraph, federation, Rust     [BUILT]
Phase 20-29: RSL, genesis, guild, HRM, NorthStar       [BUILT]
Phase 0-19: Foundation, NTU, bridges, consolidation     [BUILT]
```

### Implementation Status by Phase

| Range | Phases | Status | Built % |
|-------|--------|--------|---------|
| 0-19 | Foundation | All implemented | ~100% |
| 20-29 | RSL + Genesis | All implemented | ~100% |
| 30-41 | DDAGI + Production | All implemented | ~100% |
| 42-51 | SNR to Integration | Mostly implemented | ~80% |
| 52-59 | Lifecycle to Sprint | Mostly implemented | ~80% |
| 60-66 | Constitutional | Implemented | ~60% |
| 67 | Sovereign Instantiation | 92% (2 gaps) | 92% |
| 68 | Bus Architecture | **0% — all SPEC-ONLY** | 0% |
| 69 | Synthesis | **SPEC-ONLY** | 0% |
| 72-79 | Kernel to Genesis Gate | Implemented | ~85% |

---

## Standing on Giants

- **PMBOK**: Organizational process assets — specs are retained, not deleted
- **IEEE 830**: Spec ↔ code traceability via this index
- **Lamport**: Hash-chained evidence links specs to proof receipts
- **Brooks**: "Plan to throw one away" — early specs are learning artifacts
- **Deming**: Continuous improvement — backlog specs feed the Genesis Sprint
