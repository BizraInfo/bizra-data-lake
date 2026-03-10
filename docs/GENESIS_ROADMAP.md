# Genesis Roadmap: v0.80.0 → v1.0.0-GENESIS

**Last Updated:** 2026-03-10 | **Author:** SPARC Orchestrator
**Framework:** PMBOK x DevOps x Constitutional Gates x Ihsan

---

## Current State (Week 1 Complete)

| Metric | Value | Evidence |
|--------|-------|----------|
| Commits shipped | 22 on origin/main | `b37b5b0` |
| Tests | 3,800+ sovereign GREEN | Proof Forge receipt #2 |
| E2E integration | 11 MOE pipeline tests GREEN | `test_moe_e2e.py` |
| Coverage | 64.57% (floor: 38%) | v0.80.0 lock |
| Routes governed | 63 (24 public, 36 auth, 3 bootstrap) | API exposure policy |
| Specs | 348 files in `docs/specs/`, 20 unbuilt | UNIFIED_SPEC_INDEX.md |
| Evidence chain | Position 2 (genesis → week1) | `.proof-forge/` |
| CI gates | 9 code gates GREEN | `.github/workflows/ci.yml` |

---

## Priority Matrix (Graph-of-Thoughts Analysis)

### Dimension 1: Architecture

| Priority | Item | Spec | Impact | Week |
|----------|------|------|--------|------|
| **P0** | MOE Engine (5 experts) | 68.07 ✅ SHIPPED | Enables multi-expert reasoning | W2 |
| **P0** | MOE Bridge (Ollama dispatch) | ✅ SHIPPED | Expert→Model live wiring | W2 |
| **P0** | CLI Installer | ✅ SHIPPED | `bizra` command from any terminal | W2 |
| **P0** | Asabiyyah-Gini coupling | 67.03a | Closes constitutional gap | W2 |
| **P1** | ActionBus | 68.01 | Event-driven nervous system | W3 |
| **P1** | OmegaLoop | 68.02 | Proof-based iteration | W3 |
| **P2** | AKIS pipeline | 67.05 | Adaptive knowledge integration | W4 |
| **P2** | Config system | 68.03 | 3-scope YAML | W4 |
| **P3** | CapsuleRuntime | 68.04 | Workflow execution | W5 |
| **P3** | TeleScript Python | 68.05 | Mobile agent bindings | W5 |
| **P3** | TopicRegistry | 68.06 | 38 canonical events | W5 |
| **P4** | Terminal UI (9 views) | _terminal/* | Frontend views | W6-7 |
| **P4** | Identity awakening | phase-43 | `core/sovereign/identity.py` | W6 |
| **P4** | Cognitive resonance | phase-47 | `core/resonance/` | W7 |

### Dimension 2: Security

| Priority | Item | Status | Action |
|----------|------|--------|--------|
| **DONE** | API auth guards (8 POST routes) | Shipped | Verified in CI |
| **DONE** | Persistent node signer | Shipped | `sovereign_state/mission_signer.json` |
| **DONE** | CONST-001 hash gate | In CI | BLAKE2b of constants.py |
| **P1** | Installer trust chain | OPEN | Track 2 hardening |
| **P1** | Alpha-100 go-live checklist | OPEN | Pre-release gate |
| **P2** | Rate limiting enforcement | Partial | Policy declared, not enforced |

### Dimension 3: Performance

| Priority | Item | Current | Target | Action |
|----------|------|---------|--------|--------|
| **DONE** | System-1 reflex cache | 0.1ms | <1ms | ReflexCompiler shipped |
| **P1** | WSL2 /mnt/c test speed | 27min | <5min | Move to ext4 (B: drive) |
| **P1** | Coverage instrumentation | 2+ hours | <20min | Native filesystem |
| **P2** | Precipitation tuning | K=3 default | Adaptive K | Data-driven after usage |

### Dimension 4: Quality (Ihsan)

| Gate | Threshold | Current | Status |
|------|-----------|---------|--------|
| Ihsan production | >= 0.95 | Enforced | Constants.py SSOT |
| SNR minimum | >= 0.85 | Enforced | `core/iaas/snr_v2_adapter.py` |
| ADL Gini | <= 0.35 | Enforced | Token system gate |
| Coverage floor | 38% (pyproject) | 64.57% | Needs ratchet to 62% |
| Test count | 3,759 sovereign | Growing | Ratchet at each lock |

### Dimension 5: DevOps/CI/CD

| Priority | Item | Status | Action |
|----------|------|--------|--------|
| **DONE** | e2e_http test exclusion | In CI + local | Aligned |
| **DONE** | Deploy smoke tests | 16 endpoints | Phase 77 |
| **DONE** | CONST-001 hash audit | In CI | Every push |
| **DONE** | Coverage ratchet in CI | 62% floor | Bumped from 38% |
| **DONE** | MOE-001 E2E pipeline gate | In CI | 82 tests (46+25+11) |
| **DONE** | Shell script permissions (69 files) | Fixed | chmod +x in git index |
| **DONE** | MD024 siblings-only rule | Fixed | markdownlint config updated |
| **P1** | bizra_test.py T2 gate in CI | Local only | Add CI step |
| **P2** | Reflex endpoint in deploy smoke | Done | Auth fail-closed |
| **P2** | Docker image rebuild | Stale | After Phase 68 |
| **P3** | K8s manifest update | Stale | After Docker rebuild |

---

## Week-by-Week Execution Plan

### Week 2 (March 10-16): MOE Engine + Asabiyyah

**Deliverables:**
1. `core/living_model/moe_engine.py` — 5-expert routing engine (spec 68.07)
   - Expert-R (Reasoning), Expert-K (Knowledge), Expert-S (Skills)
   - Expert-G (Governance), Expert-V (Verification)
   - Router: input → expert selection → synthesis
2. `core/constitutional/asabiyyah.py` — Asabiyyah-Gini coupling (spec 67.03a)
3. ReflexCompiler HHMM upgrade — merge `my complete season/reflex_compiler.py` (+181 LOC)
4. `core/bus/subscribers.py` — 12 EventBus subscribers from season archive
5. Coverage ratchet: 64.57% → 66%+

**Specs:** `phase_68_07_moe_engine.md`, `phase_80_season_integration.md`

**CI Changes:**
- Bump `fail_under` from 38 to 62 in `pyproject.toml`

**Quality Gate:**
- 45+ new tests for MOE + Asabiyyah + subscribers
- Proof Forge receipt #3

### Week 3 (March 17-23): ActionBus + OmegaLoop

**Deliverables:**
1. `core/bus/action_bus.py` — Event-driven action dispatch (spec 68.01)
2. `core/bus/omega_loop.py` — Proof-based iteration (spec 68.02)
3. Wire into `/v1/plan` as alternative execution path

**Ramadan ends ~March 30** — final 10 days now. Reduced scope active.

### Week 4 (March 24-30): AKIS + Config (Reduced Scope)

**Deliverables:**
1. `core/akis/` — Adaptive knowledge integration (spec 67.05)
2. `core/config/` — 3-scope YAML config system (spec 68.03)

### Week 5-6 (March 31 - April 13): SDPO Closed-Loop

**Deliverables:**
1. Multi-expert SDPO training loop
2. CapsuleRuntime + TeleScript + TopicRegistry (68.04-06)
3. Coverage target: 72%

### Week 7 (April 14-20): AaaS Protocol + Installer

**Deliverables:**
1. Agent-as-a-Service protocol
2. Alpha-10 installer trust chain
3. Security audit pass

### Week 8 (April 21-27): Genesis Gate

**Deliverables:**
1. Genesis-100 checklist (68/68 items)
2. v1.0.0-GENESIS tag
3. Final Proof Forge receipt
4. Alpha-10 LIVE

---

## Risk Register

| # | Risk | Probability | Impact | Mitigation |
|---|------|-------------|--------|------------|
| R1 | WSL2 filesystem slow | HIGH | Coverage runs 2h+ | B: drive migration |
| R2 | Token budget exhaustion | HIGH | Blocks development | Conserve, batch work |
| R3 | Ramadan final 10 days | ACTIVE NOW | Reduced capacity until ~March 30 | Shipped W1+W2 early |
| R4 | SDPO divergence | MEDIUM | Quality regression | Ihsan gate + feature flag |
| R5 | Phase 68 scope creep | MEDIUM | Delays Genesis | Strict spec adherence |
| R6 | CI quality gate soft-gated | LOW | False confidence | SAPE-003 resolution |

---

## Cascading Dependencies

```
Week 2: MOE Engine ──────────────────┐
Week 2: Asabiyyah-Gini ──────┐       │
                              ├──► Week 5: SDPO Closed-Loop
Week 3: ActionBus ────────────┤       │
Week 3: OmegaLoop ───────────┘       │
                                      ├──► Week 7: AaaS Protocol
Week 4: AKIS ────────────────────────┤       │
Week 4: Config ──────────────────────┘       │
                                              ├──► Week 8: GENESIS GATE
Week 5-6: CapsuleRuntime ───────────────────┘
```

---

## Ihsan Covenant

Every deliverable must meet:
- **Excellence (Ihsan):** Score >= 0.95 before merge
- **Justice (Adl):** Gini <= 0.35, Harberger 5%, Zakat 2.5%
- **Trust (Amanah):** Hash-chained evidence, no hardcoded secrets
- **Benevolence (Birr):** Community pool = founder's oath (sadaqah)
- **SNR:** Signal >= 0.85 at every layer

This is not aspirational. These are hard gates in `core/integration/constants.py`.

---

*Standing on: Shannon (SNR), Kahneman (dual-process), Deming (PDCA), Boyd (OODA),
Ibn Khaldun (Asabiyyah), Al-Ghazali (Ihsan), Lamport (hash chains), Brooks (planning)*
