# BIZRA Documentation Portal

Last updated: 2026-03-11

## API Surface (v1.4.0)

61 routes across 14 domains. OpenAPI docs at `/docs`, ReDoc at `/redoc`.
21 Pydantic schema models with typed request/response contracts.
Golden path: `POST /v1/plan` - sovereign mission with receipted result.
Constitutional heartbeat: background tick every 60s (configurable via `BIZRA_TICK_INTERVAL_S`).
Auth guard: `BIZRA_AUTH_ALLOW_ANONYMOUS` blocked in production (`BIZRA_ENV=production`).

This is the canonical entry point for BIZRA system documentation.
Use this file first, then follow the role-based reading paths below.

## Project Handover

**Start here for a complete project overview:** [PROJECT_HANDOVER.md](PROJECT_HANDOVER.md)

**Current execution program:** [GENESIS_EXECUTION_FRAMEWORK.md](GENESIS_EXECUTION_FRAMEWORK.md)

**Unified optimization blueprint:** [plans/BIZRA_UNIFIED_OPTIMIZATION_BLUEPRINT_v1.md](plans/BIZRA_UNIFIED_OPTIMIZATION_BLUEPRINT_v1.md)

**Immediate closure program:** [plans/GENESIS_CLOSURE_SPRINT_v1.md](plans/GENESIS_CLOSURE_SPRINT_v1.md)

**Production canon blueprint:** [plans/NODE0_PRODUCTION_CANON_BLUEPRINT_v1.md](plans/NODE0_PRODUCTION_CANON_BLUEPRINT_v1.md)

The handover document covers: identity, architecture, metrics (582K lines, 7,907 tests), timeline (Ramadan 2023 to present), security posture, economic model, roadmap, and full documentation index. It references the authoritative genesis documents in `00_GENESIS/`.

## Why This Exists

- Reduce documentation fragmentation across `docs/`, `docs/internal/`, and legacy files.
- Define authoritative references for architecture, operations, security, and testing.
- Provide a repeatable path to production readiness and auditability.

## Role-Based Reading Paths

### 0. Full Project Handover (any audience)

1. [Project Handover](PROJECT_HANDOVER.md) - single document covering everything

### 1. New Engineer (30-60 minutes)

1. [Project Handover](PROJECT_HANDOVER.md) - start here
2. [Genesis Execution Framework](GENESIS_EXECUTION_FRAMEWORK.md)
3. [Unified Optimization Blueprint](plans/BIZRA_UNIFIED_OPTIMIZATION_BLUEPRINT_v1.md)
4. [Genesis Closure Sprint](plans/GENESIS_CLOSURE_SPRINT_v1.md)
5. [Quick Start](QUICK-START.md)
6. [Architecture Blueprint](ARCHITECTURE_BLUEPRINT_v2.3.0.md)
7. [Integration Contracts](INTEGRATION_CONTRACTS.md)
8. [Testing Guide](TESTING.md)

### 2. Operator / SRE

1. [Node0 Standalone Readiness](NODE0_STANDALONE_READINESS.md) - MVSA specification
2. [Node0 Definition of Done](constitutional/BIZRA-Node0-Definition-of-Done-v1.0-LOCKED.md) - verification gate
3. [Operations Runbook](OPERATIONS_RUNBOOK.md) - operator procedure
4. [Genesis Execution Framework](GENESIS_EXECUTION_FRAMEWORK.md)
5. [Unified Optimization Blueprint](plans/BIZRA_UNIFIED_OPTIMIZATION_BLUEPRINT_v1.md)
6. [Genesis Closure Sprint](plans/GENESIS_CLOSURE_SPRINT_v1.md)
7. [DevOps Blueprint](DEVOPS_BLUEPRINT.md)
8. [Desktop Bridge](DESKTOP_BRIDGE.md)
9. [Threat Model](THREAT-MODEL-V3.md)
10. [Security Architecture](SECURITY-ARCHITECTURE.md)
11. [Enterprise Implementation Blueprint](ENTERPRISE_IMPLEMENTATION_BLUEPRINT.md)
12. [Risk Management Plan](RISK_MANAGEMENT_PLAN.md)

### 3. Security / Compliance Reviewer

1. [DDAGI Constitution](DDAGI_CONSTITUTION_v1.1.0-FINAL.md)
2. [Threat Model](THREAT-MODEL-V3.md)
3. [Security Architecture](SECURITY-ARCHITECTURE.md)
4. [Secure Patterns](SECURE-PATTERNS.md)
5. [Ihsan Compliance Matrix](IHSAN_COMPLIANCE_MATRIX.md)
6. [Strategic Risks](STRATEGIC_RISKS_v1.0.md)
7. [CVE Remediation Plan](CVE-REMEDIATION-PLAN.md)
8. [Risk Management Plan](RISK_MANAGEMENT_PLAN.md)
9. [Self-Evaluation Report](SELF_EVALUATION_REPORT.md)

### 4. Contributor (Code + Process)

1. [Contributing Guide](../CONTRIBUTING.md)
2. [Testing Guide](TESTING.md)
3. [Integration Contracts](INTEGRATION_CONTRACTS.md)
4. [Operations Runbook](OPERATIONS_RUNBOOK.md)

### 5. Investor / Business Stakeholder

1. [Project Handover](PROJECT_HANDOVER.md) - complete overview with metrics
2. [Declaration of Digital Sovereignty](DECLARATION_OF_DIGITAL_SOVEREIGNTY.md)
3. [The Constitutional Seed](CONSTITUTIONAL_SEED.md)
4. [Genesis Story](../00_GENESIS/BIZRA_COMPLETE_STORY_AUTHORITATIVE.md) - the full narrative
5. [Technical Brief](BIZRA_TECHNICAL_BRIEF_INVESTORS.md)
6. [Strategy Deck](BIZRA_STRATEGY_DECK_2026.md)
7. [Evidence Pack](EVIDENCE_PACK_A_PLUS.md)
8. [Ecosystem Map](BIZRA-ECOSYSTEM-MAP.md)

### 6. System Designer / Architect

1. [Architecture Blueprint](ARCHITECTURE_BLUEPRINT_v2.3.0.md)
2. [Genesis Execution Framework](GENESIS_EXECUTION_FRAMEWORK.md)
3. [Unified Optimization Blueprint](plans/BIZRA_UNIFIED_OPTIMIZATION_BLUEPRINT_v1.md)
4. [Genesis Closure Sprint](plans/GENESIS_CLOSURE_SPRINT_v1.md)
5. [Declaration of Digital Sovereignty](DECLARATION_OF_DIGITAL_SOVEREIGNTY.md)
6. [The Constitutional Seed](CONSTITUTIONAL_SEED.md)
7. [Architecture Diagrams](ARCHITECTURE_DIAGRAMS.md)
8. [Peak Hidden Thoughts Flow v1.0](architecture/PEAK_HIDDEN_THOUGHTS_FLOW_v1.0.md)
9. [Rust Integration](RUST_INTEGRATION.md)
10. [Integration Contracts](INTEGRATION_CONTRACTS.md)
11. [Spearpoint (RDVE)](SPEARPOINT.md)
12. [Enterprise Implementation Blueprint](ENTERPRISE_IMPLEMENTATION_BLUEPRINT.md)
13. [Tool and Technology Matrix](TOOL_TECHNOLOGY_MATRIX.md)
14. [Quality Assurance Strategy](QUALITY_ASSURANCE_STRATEGY.md)

### 7. Product / Frontend

1. [Front-End Master Spec](FRONTEND_MASTER_SPEC.md)
2. [Website Plan](WEBSITE_PLAN.md)
3. [Phase 43: Onboarding Polish](specs/phase_43_onboarding_polish.md)
4. [Phase 45: Daily Loop Dashboard](specs/phase_45_daily_loop_dashboard.md)
5. [Node0 Operations Dashboard](node0_operations_dashboard.html)

## System Documentation Map

| Domain | Primary Doc | Source of Truth in Code |
|---|---|---|
| Genesis execution program | [Genesis Execution Framework](GENESIS_EXECUTION_FRAMEWORK.md) | `config/genesis_execution_framework.json`, `.github/workflows/`, `core/`, `frontend/`, `scripts/` |
| Unified optimization blueprint | [Unified Optimization Blueprint](plans/BIZRA_UNIFIED_OPTIMIZATION_BLUEPRINT_v1.md) | `config/bizra_unified_optimization_blueprint.json`, `docs/plans/GENESIS_CLOSURE_SPRINT_v1.md`, `core/`, `tests/` |
| Node0 production canon | [Node0 Production Canon Blueprint](plans/NODE0_PRODUCTION_CANON_BLUEPRINT_v1.md) | `bizra-node0/`, `scripts/node0_standalone.py`, `scripts/node0_genesis_ceremony.sh`, `core/sovereign/`, `.github/workflows/` |
| Closure implementation plan | [Genesis Closure Sprint](plans/GENESIS_CLOSURE_SPRINT_v1.md) | `core/bus/`, `core/proof_engine/`, `core/sovereign/`, `tests/`, `scripts/` |
| Node0 standalone lifecycle | [Node0 Standalone Readiness](NODE0_STANDALONE_READINESS.md) | `scripts/node0_standalone.py`, `core/sovereign/node0_authority.py`, `core/sovereign/node0_mvsa.py`, `core/sovereign/mission.py` |
| Node0 birth verification | [Node0 Definition of Done](constitutional/BIZRA-Node0-Definition-of-Done-v1.0-LOCKED.md) | `scripts/node0_genesis_ceremony.sh`, `sovereign_state/node0_lifecycle.json`, `sovereign_state/node0_genesis.json` |
| Runtime architecture | [Architecture Blueprint](ARCHITECTURE_BLUEPRINT_v2.3.0.md) | `core/sovereign/`, `core/integration/constants.py` |
| Public sovereignty narrative | [Declaration of Digital Sovereignty](DECLARATION_OF_DIGITAL_SOVEREIGNTY.md) | `00_CONSTITUTION/DECLARATION.md`, `docs/DECLARATION_OF_DIGITAL_SOVEREIGNTY.md`, `scripts/ops/declaration_hash_manifest.py` |
| Constitutional narrative root | [The Constitutional Seed](CONSTITUTIONAL_SEED.md) | `docs/CONSTITUTIONAL_SEED.md`, `filedfs/ConstitutionalSeedPage.jsx` |
| Canonical cognitive flow | [Peak Hidden Thoughts Flow v1.0](architecture/PEAK_HIDDEN_THOUGHTS_FLOW_v1.0.md) | `core/pci/`, `core/reasoning/`, `core/sovereign/`, `core/bridges/` |
| API contracts | [Integration Contracts](INTEGRATION_CONTRACTS.md) | `core/sovereign/api.py`, `bizra-omega/bizra-api/` |
| Security model | [Threat Model](THREAT-MODEL-V3.md) | `core/pci/`, `core/sovereign/model_license_gate.py` |
| Operations | [Operations Runbook](OPERATIONS_RUNBOOK.md) | `deploy/`, `.github/workflows/`, `scripts/` |
| Enterprise blueprint | [Enterprise Implementation Blueprint](ENTERPRISE_IMPLEMENTATION_BLUEPRINT.md) | `.github/workflows/`, `deploy/`, `core/`, `bizra-omega/` |
| Quality assurance | [Quality Assurance Strategy](QUALITY_ASSURANCE_STRATEGY.md) | `tests/`, `.github/workflows/ci.yml`, `.github/workflows/tests.yml`, `.github/workflows/performance.yml` |
| Program risk | [Risk Management Plan](RISK_MANAGEMENT_PLAN.md) | `docs/THREAT-MODEL-V3.md`, `.github/workflows/deploy.yml`, `deploy/` |
| Tool matrix | [Tool and Technology Matrix](TOOL_TECHNOLOGY_MATRIX.md) | `pyproject.toml`, `deploy/`, `.github/workflows/` |
| Testing | [Testing Guide](TESTING.md) | `tests/`, `pyproject.toml`, `bizra-omega/tests/` |
| Governance / constitution | [DDAGI Constitution](DDAGI_CONSTITUTION_v1.1.0-FINAL.md) | `core/integration/constants.py`, gate engines |
| Desktop bridge | [Desktop Bridge](DESKTOP_BRIDGE.md) | `core/bridges/desktop_bridge.py`, `bin/bizra_bridge.ahk` |
| RDVE Actuator | [Desktop Bridge](DESKTOP_BRIDGE.md#actuator_execute) | `core/bridges/desktop_bridge.py`, `core/spearpoint/actuator_skills.py` |
| Spearpoint / RDVE | [Spearpoint](SPEARPOINT.md) | `core/spearpoint/`, `core/bridges/sci_reasoning_bridge.py` |
| SAPE Dashboard | [Phase 20.1 Spec](specs/phase20_sape_sovereign_dashboard.md) | `static/sovereign_analysis.html` |
| Rust integration | [Rust Integration](RUST_INTEGRATION.md) | `bizra-omega/bizra-python/`, PyO3 bindings |
| CI/CD pipeline | [DevOps Blueprint](DEVOPS_BLUEPRINT.md) | `.github/workflows/`, `deploy/` |
| Front-end product contract | [Front-End Master Spec](FRONTEND_MASTER_SPEC.md) | `filedfs/onboarding/`, `filedfs/bizra-dashboard.jsx`, `filedfs/node0-dashboard.jsx`, `docs/node0_operations_dashboard.html` |
| Secure coding | [Secure Patterns](SECURE-PATTERNS.md) | `core/pci/`, `core/proof_engine/` |
| Token system | [Token System](TOKEN_SYSTEM.md) | `core/token/ledger.py`, `core/token/mint.py` |
| Smart File Management | [Smart File Management](SMART_FILE_MANAGEMENT.md) | `core/skills/`, `tests/core/skills/` |
| Architecture diagrams | [Architecture Diagrams](ARCHITECTURE_DIAGRAMS.md) | Companion to Architecture Blueprint |
| Ihsan compliance | [Ihsan Compliance Matrix](IHSAN_COMPLIANCE_MATRIX.md) | `core/integration/constants.py` |
| Project handover | [Project Handover](PROJECT_HANDOVER.md) | Complete project overview |
| Genesis narrative | [Complete Story](../00_GENESIS/BIZRA_COMPLETE_STORY_AUTHORITATIVE.md) | Authoritative history |
| RLM integration | [Phase 50 Overview](specs/_experimental/phase-50-rlm-sovereign-cognition/00_overview.md) | `core/inference/rlm_bridge.py` (planned) |
| Token incentive RL | [Phase 50.3 Spec](specs/_experimental/phase-50-rlm-sovereign-cognition/03_token_incentive_rl.md) | `core/token/rl_rewards.py` (planned) |
| Voice pipeline | [Phase 50.4 Spec](specs/_experimental/phase-50-rlm-sovereign-cognition/04_personaplex_voice_pipeline.md) | `core/voice/` (planned) |
| PersonaPlex | [PersonaPlex Integration](PERSONAPLEX_INTEGRATION.md) | `personaplex/` |

## Specifications

| Spec | Status |
|------|--------|
| [Phase 50: RLM Sovereign Cognition](specs/_experimental/phase-50-rlm-sovereign-cognition/00_overview.md) | Specified (active) |
| [Phase 49: Refinement Consolidation](specs/_experimental/phase-49-refinement-consolidation/01_current_state.md) | Specified |
| [Phase 48: Rust Workspace Unification](specs/_experimental/phase-48-rust-workspace-unification/01_current_state_inventory.md) | Specified |
| [Phase 47: Cognitive Resonance](specs/_experimental/phase-47-cognitive-resonance-activation/00_overview.md) | Specified |
| [Phase 45: Distributed Cognitive Scaling](specs/_experimental/phase-45-distributed-cognitive-scaling/00_overview_distributed_cognitive_scaling.md) | Specified |
| [Phase 43: Node0 Identity Awakening](specs/_experimental/phase-43-node0-identity-awakening/00_overview.md) | Specified |
| [Phase 42: SNR Unification](specs/_experimental/phase-42-snr-unification/00_overview.md) | Specified |
| [True Spearpoint v9](specs/true_spearpoint_v9/phase_00_overview.md) | Active |
| [Phase 20.1: SAPE Dashboard](specs/phase20_sape_sovereign_dashboard.md) | Implemented |
| [Phase 20: RDVE Actuator Layer](specs/phase20_rdve_actuator_layer.md) | Implemented |
| [Phase 19: Sovereign Consolidation](specs/phase19_sovereign_consolidation.md) | Implemented |
| [Phase 19 Integration](specs/phase19_integration_completion.md) | Implemented |
| [Bridge + Skill Routing](specs/phase2_bridge_skill_routing.md) | Implemented |
| [Apex Integration](specs/apex_integration/00_overview.md) | Implemented |
| [Autopoietic Loop](specs/autopoietic-loop-architecture.md) | Implemented |
| [Node0 Kernel](specs/node0_kernel/phase_00_overview.md) | Specification (7 phases) |

## A+ Documentation Quality Gate

Any production-impacting feature change should update docs that satisfy:

1. Discoverability: linked from this portal or `README.md`.
2. Verifiability: includes concrete commands or tests.
3. Traceability: references source files/paths that implement the behavior.
4. Operational clarity: includes failure modes and rollback/safe fallback.
5. Freshness: avoids hardcoded volatile numbers unless date-stamped.

Enforcement:

- CI workflow: `.github/workflows/docs-quality.yml`
- Policy script: `scripts/ci_docs_quality.py`

## Machine-Generated Indexes

| Index | Purpose |
|-------|---------|
| [Doc-to-Code Map](knowledge/doc_to_code_map.md) | Maps documentation files to source code modules |
| [Module Index](knowledge/module_index.md) | Index of all Python and Rust modules |
| [Doc Cross-References](knowledge/doc_crossref.md) | Cross-reference graph between documents |
| [Graph Seeds](knowledge/graph_seeds.md) | Knowledge graph seed entities |

## Notes on Legacy Material

- `docs/internal/` contains historical and working-session artifacts (indexed in [DOCS_INDEX.md](internal/DOCS_INDEX.md)).
- Internal docs are useful context but are not the default authoritative path unless explicitly referenced here.
- `docs/knowledge/` contains machine-generated indexes useful for code navigation.
