# BIZRA Documentation Portal

Last updated: 2026-02-14

This is the canonical entry point for BIZRA system documentation.
Use this file first, then follow the role-based reading paths below.

## Why This Exists

- Reduce documentation fragmentation across `docs/`, `docs/internal/`, and legacy files.
- Define authoritative references for architecture, operations, security, and testing.
- Provide a repeatable path to production readiness and auditability.

## Role-Based Reading Paths

### 1. New Engineer (30-60 minutes)

1. [README.md](../README.md)
2. [Quick Start](QUICK-START.md)
3. [Architecture Blueprint](ARCHITECTURE_BLUEPRINT_v2.3.0.md)
4. [Integration Contracts](INTEGRATION_CONTRACTS.md)
5. [Testing Guide](TESTING.md)

### 2. Operator / SRE

1. [Operations Runbook](OPERATIONS_RUNBOOK.md)
2. [DevOps Blueprint](DEVOPS_BLUEPRINT.md)
3. [Desktop Bridge](DESKTOP_BRIDGE.md)
4. [Threat Model](THREAT-MODEL-V3.md)
5. [Security Architecture](SECURITY-ARCHITECTURE.md)

### 3. Security / Compliance Reviewer

1. [DDAGI Constitution](DDAGI_CONSTITUTION_v1.1.0-FINAL.md)
2. [Threat Model](THREAT-MODEL-V3.md)
3. [Security Architecture](SECURITY-ARCHITECTURE.md)
4. [Secure Patterns](SECURE-PATTERNS.md)
5. [Ihsan Compliance Matrix](IHSAN_COMPLIANCE_MATRIX.md)
6. [Strategic Risks](STRATEGIC_RISKS_v1.0.md)
7. [CVE Remediation Plan](CVE-REMEDIATION-PLAN.md)

### 4. Contributor (Code + Process)

1. [Contributing Guide](../CONTRIBUTING.md)
2. [Testing Guide](TESTING.md)
3. [Integration Contracts](INTEGRATION_CONTRACTS.md)
4. [Operations Runbook](OPERATIONS_RUNBOOK.md)

### 5. Investor / Business Stakeholder

1. [Technical Brief](BIZRA_TECHNICAL_BRIEF_INVESTORS.md)
2. [Strategy Deck](BIZRA_STRATEGY_DECK_2026.md)
3. [Evidence Pack](EVIDENCE_PACK_A_PLUS.md)
4. [Ecosystem Map](BIZRA-ECOSYSTEM-MAP.md)

### 6. System Designer / Architect

1. [Architecture Blueprint](ARCHITECTURE_BLUEPRINT_v2.3.0.md)
2. [Architecture Diagrams](ARCHITECTURE_DIAGRAMS.md)
3. [Rust Integration](RUST_INTEGRATION.md)
4. [Integration Contracts](INTEGRATION_CONTRACTS.md)
5. [Spearpoint (RDVE)](SPEARPOINT.md)

## System Documentation Map

| Domain | Primary Doc | Source of Truth in Code |
|---|---|---|
| Runtime architecture | [Architecture Blueprint](ARCHITECTURE_BLUEPRINT_v2.3.0.md) | `core/sovereign/`, `core/integration/constants.py` |
| API contracts | [Integration Contracts](INTEGRATION_CONTRACTS.md) | `core/sovereign/api.py`, `bizra-omega/bizra-api/` |
| Security model | [Threat Model](THREAT-MODEL-V3.md) | `core/pci/`, `core/sovereign/model_license_gate.py` |
| Operations | [Operations Runbook](OPERATIONS_RUNBOOK.md) | `deploy/`, `.github/workflows/`, `scripts/` |
| Testing | [Testing Guide](TESTING.md) | `tests/`, `pyproject.toml`, `bizra-omega/tests/` |
| Governance / constitution | [DDAGI Constitution](DDAGI_CONSTITUTION_v1.1.0-FINAL.md) | `core/integration/constants.py`, gate engines |
| Desktop bridge | [Desktop Bridge](DESKTOP_BRIDGE.md) | `core/bridges/desktop_bridge.py`, `bin/bizra_bridge.ahk` |
| RDVE Actuator | [Desktop Bridge](DESKTOP_BRIDGE.md#actuator_execute) | `core/bridges/desktop_bridge.py`, `core/spearpoint/actuator_skills.py` |
| Spearpoint / RDVE | [Spearpoint](SPEARPOINT.md) | `core/spearpoint/`, `core/bridges/sci_reasoning_bridge.py` |
| SAPE Dashboard | [Phase 20.1 Spec](specs/phase20_sape_sovereign_dashboard.md) | `static/sovereign_analysis.html` |
| Rust integration | [Rust Integration](RUST_INTEGRATION.md) | `bizra-omega/bizra-python/`, PyO3 bindings |
| CI/CD pipeline | [DevOps Blueprint](DEVOPS_BLUEPRINT.md) | `.github/workflows/`, `deploy/` |
| Secure coding | [Secure Patterns](SECURE-PATTERNS.md) | `core/pci/`, `core/proof_engine/` |
| Token system | [Token System](TOKEN_SYSTEM.md) | `core/token/ledger.py`, `core/token/mint.py` |
| Smart File Management | [Smart File Management](SMART_FILE_MANAGEMENT.md) | `core/skills/`, `tests/core/skills/` |
| Architecture diagrams | [Architecture Diagrams](ARCHITECTURE_DIAGRAMS.md) | Companion to Architecture Blueprint |
| Ihsan compliance | [Ihsan Compliance Matrix](IHSAN_COMPLIANCE_MATRIX.md) | `core/integration/constants.py` |

## Specifications

| Spec | Status |
|------|--------|
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
